package de.kherud.llama.gguf;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * Library-friendly GGUF file inspection utility.
 *
 * This refactored version provides a fluent API for inspecting GGUF files,
 * with builder pattern configuration, streaming capabilities, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic inspection
 * try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(path)) {
 *     InspectionResult result = inspector.inspect();
 *     System.out.println("File version: " + result.getFileInfo().getVersion());
 * }
 *
 * // Configured inspection
 * try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(path)
 *         .includeMetadata(true)
 *         .includeTensors(false)
 *         .filterByKey("model")) {
 *     InspectionResult result = inspector.inspect();
 * }
 *
 * // Streaming metadata
 * try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(path)) {
 *     inspector.streamMetadata()
 *             .filter(entry -> entry.getKey().contains("attention"))
 *             .forEach(entry -> System.out.println(entry.getKey() + ": " + entry.getValue()));
 * }
 *
 * // Async inspection
 * GGUFInspectorLibrary.open(path)
 *     .inspectAsync()
 *     .thenAccept(result -> System.out.println("Inspection complete"))
 *     .exceptionally(throwable -> {
 *         System.err.println("Inspection failed: " + throwable.getMessage());
 *         return null;
 *     });
 * }</pre>
 */
public class GGUFInspectorLibrary implements AutoCloseable {
	private static final System.Logger logger = System.getLogger(GGUFInspectorLibrary.class.getName());
	private static final ExecutorService defaultExecutor = Executors.newCachedThreadPool(r -> {
		Thread t = new Thread(r, "GGUFInspector-" + System.nanoTime());
		t.setDaemon(true);
		return t;
	});

	// Configuration
	private final Path filePath;
	private final GGUFReader reader;
	private boolean includeMetadata = true;
	private boolean includeTensors = true;
	private boolean includeTensorData = false;
	private boolean includeFileInfo = true;
	private boolean verbose = false;
	private String keyFilter = null;
	private Predicate<String> keyPredicate = null;
	private int maxStringLength = 60;
	private ExecutorService executor = defaultExecutor;

	// Factory method
	public static GGUFInspectorLibrary open(Path filePath) throws IOException {
		return new GGUFInspectorLibrary(filePath);
	}

	public static GGUFInspectorLibrary open(String filePath) throws IOException {
		return new GGUFInspectorLibrary(Path.of(filePath));
	}

	private GGUFInspectorLibrary(Path filePath) throws IOException {
		this.filePath = filePath;
		if (!Files.exists(filePath)) {
			throw new IOException("File not found: " + filePath);
		}
		if (!Files.isRegularFile(filePath)) {
			throw new IOException("Not a regular file: " + filePath);
		}
		this.reader = new GGUFReader(filePath);
	}

	// Fluent configuration methods
	public GGUFInspectorLibrary includeMetadata(boolean include) {
		this.includeMetadata = include;
		return this;
	}

	public GGUFInspectorLibrary includeTensors(boolean include) {
		this.includeTensors = include;
		return this;
	}

	public GGUFInspectorLibrary includeTensorData(boolean include) {
		this.includeTensorData = include;
		return this;
	}

	public GGUFInspectorLibrary includeFileInfo(boolean include) {
		this.includeFileInfo = include;
		return this;
	}

	public GGUFInspectorLibrary verbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}

	public GGUFInspectorLibrary filterByKey(String keyPattern) {
		this.keyFilter = keyPattern;
		this.keyPredicate = keyPattern != null ? key -> key.contains(keyPattern) : null;
		return this;
	}

	public GGUFInspectorLibrary filterByKeyPredicate(Predicate<String> predicate) {
		this.keyPredicate = predicate;
		this.keyFilter = null; // Clear simple filter
		return this;
	}

	public GGUFInspectorLibrary maxStringLength(int length) {
		this.maxStringLength = Math.max(1, length);
		return this;
	}

	public GGUFInspectorLibrary executor(ExecutorService executor) {
		this.executor = Objects.requireNonNull(executor, "Executor cannot be null");
		return this;
	}

	// Primary inspection methods
	public InspectionResult inspect() throws IOException {
		InspectionResult.Builder resultBuilder = new InspectionResult.Builder(filePath);

		if (includeFileInfo) {
			resultBuilder.fileInfo(createFileInfo());
		}

		if (includeMetadata) {
			resultBuilder.metadata(collectMetadata());
		}

		if (includeTensors) {
			resultBuilder.tensors(collectTensors());
		}

		return resultBuilder.build();
	}

	public CompletableFuture<InspectionResult> inspectAsync() {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return inspect();
			} catch (IOException e) {
				throw new RuntimeException("Inspection failed", e);
			}
		}, executor);
	}

	public void inspectWithProgress(Consumer<InspectionProgress> progressCallback) throws IOException {
		Objects.requireNonNull(progressCallback, "Progress callback cannot be null");

		progressCallback.accept(new InspectionProgress("Starting inspection", 0.0));

		FileInfo fileInfo = null;
		if (includeFileInfo) {
			progressCallback.accept(new InspectionProgress("Reading file info", 0.1));
			fileInfo = createFileInfo();
		}

		Map<String, Object> metadata = null;
		if (includeMetadata) {
			progressCallback.accept(new InspectionProgress("Reading metadata", 0.3));
			metadata = collectMetadata();
		}

		Map<String, TensorInfo> tensors = null;
		if (includeTensors) {
			progressCallback.accept(new InspectionProgress("Reading tensors", 0.7));
			tensors = collectTensors();
		}

		progressCallback.accept(new InspectionProgress("Building result", 0.9));
		InspectionResult result = new InspectionResult.Builder(filePath)
			.fileInfo(fileInfo)
			.metadata(metadata)
			.tensors(tensors)
			.build();

		progressCallback.accept(new InspectionProgress("Inspection complete", 1.0, result));
	}

	// Streaming APIs
	public Stream<MetadataEntry> streamMetadata() {
		Map<String, GGUFReader.GGUFField> rawMetadata = reader.getFields();
		return rawMetadata.entrySet().stream()
			.filter(entry -> keyPredicate == null || keyPredicate.test(entry.getKey()))
			.map(entry -> new MetadataEntry(entry.getKey(), entry.getValue().value, entry.getValue().type.ordinal()));
	}

	public Stream<TensorInfo> streamTensors() {
		List<GGUFReader.GGUFTensor> rawTensors = reader.getTensors();
		return rawTensors.stream()
			.filter(tensor -> keyPredicate == null || keyPredicate.test(tensor.name))
			.map(this::createTensorInfo);
	}

	// Individual query methods
	public Optional<Object> getMetadata(String key) {
		Map<String, GGUFReader.GGUFField> metadata = reader.getFields();
		GGUFReader.GGUFField field = metadata.get(key);
		return field != null ? Optional.of(field.value) : Optional.empty();
	}

	public List<String> getMetadataKeys() {
		return new ArrayList<>(reader.getFields().keySet());
	}

	public List<String> getTensorNames() {
		return reader.getTensors().stream()
			.map(tensor -> tensor.name)
			.collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
	}

	public Optional<TensorInfo> getTensorInfo(String name) {
		return reader.getTensors().stream()
			.filter(tensor -> tensor.name.equals(name))
			.map(this::createTensorInfo)
			.findFirst();
	}

	public FileInfo getFileInfo() throws IOException {
		return createFileInfo();
	}

	public boolean hasMetadata(String key) {
		return reader.getFields().containsKey(key);
	}

	public boolean hasTensor(String name) {
		return reader.getTensors().stream()
			.anyMatch(tensor -> tensor.name.equals(name));
	}

	// Validation methods
	public ValidationResult validate() throws IOException {
		ValidationResult.Builder builder = new ValidationResult.Builder();

		// File existence and readability
		builder.fileExists(Files.exists(filePath));
		builder.fileReadable(Files.isReadable(filePath));

		try {
			// Header validation
			reader.getFields(); // This will throw if header is invalid
			builder.headerValid(true);

			// Metadata validation
			int metadataCount = reader.getFields().size();
			builder.metadataCount(metadataCount);

			// Tensor validation
			List<GGUFReader.GGUFTensor> tensors = reader.getTensors();
			builder.tensorCount(tensors.size());

			// Check for common required metadata
			Map<String, GGUFReader.GGUFField> metadata = reader.getFields();
			builder.hasRequiredMetadata("general.name", metadata.containsKey("general.name"));
			builder.hasRequiredMetadata("general.architecture", metadata.containsKey("general.architecture"));

		} catch (Exception e) {
			builder.headerValid(false);
			builder.validationError(e.getMessage());
		}

		return builder.build();
	}

	public CompletableFuture<ValidationResult> validateAsync() {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return validate();
			} catch (IOException e) {
				throw new RuntimeException("Validation failed", e);
			}
		}, executor);
	}

	// Helper methods
	private FileInfo createFileInfo() throws IOException {
		FileInfo.Builder builder = new FileInfo.Builder();

		// File size
		builder.fileSize(Files.size(filePath));

		// GGUF specific info
		Map<String, GGUFReader.GGUFField> metadata = reader.getFields();
		List<GGUFReader.GGUFTensor> tensors = reader.getTensors();

		builder.metadataCount(metadata.size());
		builder.tensorCount(tensors.size());

		// Calculate checksum if verbose mode
		if (verbose) {
			builder.checksum(calculateChecksum());
		}

		return builder.build();
	}

	private Map<String, Object> collectMetadata() {
		Map<String, Object> result = new LinkedHashMap<>();
		Map<String, GGUFReader.GGUFField> rawMetadata = reader.getFields();

		for (Map.Entry<String, GGUFReader.GGUFField> entry : rawMetadata.entrySet()) {
			String key = entry.getKey();
			if (keyPredicate == null || keyPredicate.test(key)) {
				Object value = entry.getValue().value;
				if (value instanceof String && ((String) value).length() > maxStringLength) {
					value = ((String) value).substring(0, maxStringLength) + "...";
				}
				result.put(key, value);
			}
		}

		return result;
	}

	private Map<String, TensorInfo> collectTensors() {
		Map<String, TensorInfo> result = new LinkedHashMap<>();
		List<GGUFReader.GGUFTensor> rawTensors = reader.getTensors();

		for (GGUFReader.GGUFTensor tensor : rawTensors) {
			if (keyPredicate == null || keyPredicate.test(tensor.name)) {
				result.put(tensor.name, createTensorInfo(tensor));
			}
		}

		return result;
	}

	private TensorInfo createTensorInfo(GGUFReader.GGUFTensor tensor) {
		return new TensorInfo.Builder()
			.name(tensor.name)
			.type(tensor.type.ordinal())
			.shape(Arrays.copyOf(tensor.shape, tensor.shape.length))
			.offset(tensor.dataOffset)
			.build();
	}

	private String calculateChecksum() throws IOException {
		try {
			MessageDigest md = MessageDigest.getInstance("SHA-256");
			try (InputStream is = new FileInputStream(filePath.toFile())) {
				byte[] buffer = new byte[8192];
				int bytesRead;
				while ((bytesRead = is.read(buffer)) != -1) {
					md.update(buffer, 0, bytesRead);
				}
			}

			byte[] hash = md.digest();
			StringBuilder hexString = new StringBuilder();
			for (byte b : hash) {
				String hex = Integer.toHexString(0xff & b);
				if (hex.length() == 1) {
					hexString.append('0');
				}
				hexString.append(hex);
			}
			return hexString.toString();
		} catch (Exception e) {
			throw new IOException("Failed to calculate checksum", e);
		}
	}

	@Override
	public void close() throws IOException {
		if (reader != null) {
			reader.close();
		}
	}

	// Result classes with builders
	public static class InspectionResult {
		private final Path filePath;
		private final FileInfo fileInfo;
		private final Map<String, Object> metadata;
		private final Map<String, TensorInfo> tensors;

		private InspectionResult(Builder builder) {
			this.filePath = builder.filePath;
			this.fileInfo = builder.fileInfo;
			this.metadata = builder.metadata != null ? Collections.unmodifiableMap(builder.metadata) : Collections.emptyMap();
			this.tensors = builder.tensors != null ? Collections.unmodifiableMap(builder.tensors) : Collections.emptyMap();
		}

		public Path getFilePath() { return filePath; }
		public Optional<FileInfo> getFileInfo() { return Optional.ofNullable(fileInfo); }
		public Map<String, Object> getMetadata() { return metadata; }
		public Map<String, TensorInfo> getTensors() { return tensors; }

		public static class Builder {
			private final Path filePath;
			private FileInfo fileInfo;
			private Map<String, Object> metadata;
			private Map<String, TensorInfo> tensors;

			public Builder(Path filePath) {
				this.filePath = filePath;
			}

			public Builder fileInfo(FileInfo fileInfo) {
				this.fileInfo = fileInfo;
				return this;
			}

			public Builder metadata(Map<String, Object> metadata) {
				this.metadata = metadata;
				return this;
			}

			public Builder tensors(Map<String, TensorInfo> tensors) {
				this.tensors = tensors;
				return this;
			}

			public InspectionResult build() {
				return new InspectionResult(this);
			}
		}
	}

	public static class FileInfo {
		private final long fileSize;
		private final int metadataCount;
		private final int tensorCount;
		private final String checksum;

		private FileInfo(Builder builder) {
			this.fileSize = builder.fileSize;
			this.metadataCount = builder.metadataCount;
			this.tensorCount = builder.tensorCount;
			this.checksum = builder.checksum;
		}

		public long getFileSize() { return fileSize; }
		public int getMetadataCount() { return metadataCount; }
		public int getTensorCount() { return tensorCount; }
		public Optional<String> getChecksum() { return Optional.ofNullable(checksum); }

		public static class Builder {
			private long fileSize;
			private int metadataCount;
			private int tensorCount;
			private String checksum;

			public Builder fileSize(long size) {
				this.fileSize = size;
				return this;
			}

			public Builder metadataCount(int count) {
				this.metadataCount = count;
				return this;
			}

			public Builder tensorCount(int count) {
				this.tensorCount = count;
				return this;
			}

			public Builder checksum(String checksum) {
				this.checksum = checksum;
				return this;
			}

			public FileInfo build() {
				return new FileInfo(this);
			}
		}
	}

	public static class TensorInfo {
		private final String name;
		private final int type;
		private final long[] shape;
		private final long offset;

		private TensorInfo(Builder builder) {
			this.name = builder.name;
			this.type = builder.type;
			this.shape = builder.shape;
			this.offset = builder.offset;
		}

		public String getName() { return name; }
		public int getType() { return type; }
		public long[] getShape() { return Arrays.copyOf(shape, shape.length); }
		public long getOffset() { return offset; }

		public static class Builder {
			private String name;
			private int type;
			private long[] shape;
			private long offset;

			public Builder name(String name) {
				this.name = name;
				return this;
			}

			public Builder type(int type) {
				this.type = type;
				return this;
			}

			public Builder shape(long[] shape) {
				this.shape = shape;
				return this;
			}

			public Builder offset(long offset) {
				this.offset = offset;
				return this;
			}

			public TensorInfo build() {
				return new TensorInfo(this);
			}
		}
	}

	public static class MetadataEntry {
		private final String key;
		private final Object value;
		private final int type;

		public MetadataEntry(String key, Object value, int type) {
			this.key = key;
			this.value = value;
			this.type = type;
		}

		public String getKey() { return key; }
		public Object getValue() { return value; }
		public int getType() { return type; }
	}

	public static class InspectionProgress {
		private final String message;
		private final double progress;
		private final InspectionResult result;

		public InspectionProgress(String message, double progress) {
			this(message, progress, null);
		}

		public InspectionProgress(String message, double progress, InspectionResult result) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.result = result;
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Optional<InspectionResult> getResult() { return Optional.ofNullable(result); }
	}

	public static class ValidationResult {
		private final boolean fileExists;
		private final boolean fileReadable;
		private final boolean headerValid;
		private final int metadataCount;
		private final int tensorCount;
		private final Map<String, Boolean> requiredMetadata;
		private final String validationError;

		private ValidationResult(Builder builder) {
			this.fileExists = builder.fileExists;
			this.fileReadable = builder.fileReadable;
			this.headerValid = builder.headerValid;
			this.metadataCount = builder.metadataCount;
			this.tensorCount = builder.tensorCount;
			this.requiredMetadata = Collections.unmodifiableMap(builder.requiredMetadata);
			this.validationError = builder.validationError;
		}

		public boolean isFileExists() { return fileExists; }
		public boolean isFileReadable() { return fileReadable; }
		public boolean isHeaderValid() { return headerValid; }
		public int getMetadataCount() { return metadataCount; }
		public int getTensorCount() { return tensorCount; }
		public Map<String, Boolean> getRequiredMetadata() { return requiredMetadata; }
		public Optional<String> getValidationError() { return Optional.ofNullable(validationError); }

		public boolean isValid() {
			return fileExists && fileReadable && headerValid && validationError == null;
		}

		public static class Builder {
			private boolean fileExists;
			private boolean fileReadable;
			private boolean headerValid;
			private int metadataCount;
			private int tensorCount;
			private final Map<String, Boolean> requiredMetadata = new LinkedHashMap<>();
			private String validationError;

			public Builder fileExists(boolean exists) {
				this.fileExists = exists;
				return this;
			}

			public Builder fileReadable(boolean readable) {
				this.fileReadable = readable;
				return this;
			}

			public Builder headerValid(boolean valid) {
				this.headerValid = valid;
				return this;
			}

			public Builder metadataCount(int count) {
				this.metadataCount = count;
				return this;
			}

			public Builder tensorCount(int count) {
				this.tensorCount = count;
				return this;
			}

			public Builder hasRequiredMetadata(String key, boolean present) {
				this.requiredMetadata.put(key, present);
				return this;
			}

			public Builder validationError(String error) {
				this.validationError = error;
				return this;
			}

			public ValidationResult build() {
				return new ValidationResult(this);
			}
		}
	}
}
