package de.kherud.llama.converters;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.gguf.GGUFConstants;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Library-friendly HuggingFace to GGUF converter.
 *
 * This refactored version provides a fluent API for converting HuggingFace models to GGUF format,
 * with builder pattern configuration, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic conversion
 * ConversionResult result = HuggingFaceToGGUFConverterLibrary.builder()
 *     .source(Paths.get("model/"))
 *     .destination(Paths.get("model.gguf"))
 *     .build()
 *     .convert();
 *
 * // Configured conversion
 * ConversionResult result = HuggingFaceToGGUFConverterLibrary.builder()
 *     .source(modelPath)
 *     .destination(outputPath)
 *     .quantization(GGUFConstants.GGMLQuantizationType.Q4_0)
 *     .threads(8)
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .build()
 *     .convert();
 *
 * // Async conversion
 * HuggingFaceToGGUFConverterLibrary.builder()
 *     .source(modelPath)
 *     .destination(outputPath)
 *     .build()
 *     .convertAsync()
 *     .thenAccept(result -> System.out.println("Conversion complete"))
 *     .exceptionally(throwable -> {
 *         System.err.println("Conversion failed: " + throwable.getMessage());
 *         return null;
 *     });
 * }</pre>
 */
public class HuggingFaceToGGUFConverterLibrary {
	private static final System.Logger logger = System.getLogger(HuggingFaceToGGUFConverterLibrary.class.getName());
	private static final ExecutorService defaultExecutor = Executors.newCachedThreadPool(r -> {
		Thread t = new Thread(r, "HFConverter-" + System.nanoTime());
		t.setDaemon(true);
		return t;
	});

	// Configuration
	private final Path sourcePath;
	private final Path destinationPath;
	private final GGUFConstants.GGMLQuantizationType quantizationType;
	private final int threadCount;
	private final boolean verbose;
	private final Map<String, String> overrideMetadata;
	private final Consumer<ConversionProgress> progressCallback;
	private final ExecutorService executor;
	private final Duration timeout;

	private HuggingFaceToGGUFConverterLibrary(Builder builder) {
		this.sourcePath = builder.sourcePath;
		this.destinationPath = builder.destinationPath;
		this.quantizationType = builder.quantizationType;
		this.threadCount = builder.threadCount;
		this.verbose = builder.verbose;
		this.overrideMetadata = new HashMap<>(builder.overrideMetadata);
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor != null ? builder.executor : defaultExecutor;
		this.timeout = builder.timeout;
	}

	public static Builder builder() {
		return new Builder();
	}

	// Primary conversion methods
	public ConversionResult convert() throws ConversionException {
		try {
			validateInputs();

			Instant startTime = Instant.now();
			progress("Starting conversion", 0.0);

			// Load model configuration
			progress("Loading model configuration", 0.1);
			ModelInfo modelInfo = loadModelInfo();

			// Load tensors
			progress("Loading tensors", 0.2);
			Map<String, TensorData> tensors = loadTensors();

			// Convert and quantize if needed
			progress("Converting tensors", 0.5);
			Map<String, TensorData> convertedTensors = convertTensors(tensors);

			// Write GGUF file
			progress("Writing GGUF file", 0.8);
			writeGGUFFile(modelInfo, convertedTensors);

			Duration conversionTime = Duration.between(startTime, Instant.now());
			progress("Conversion complete", 1.0);

			return new ConversionResult.Builder()
				.success(true)
				.sourcePath(sourcePath)
				.destinationPath(destinationPath)
				.modelInfo(modelInfo)
				.tensorCount(convertedTensors.size())
				.conversionTime(conversionTime)
				.quantizationType(quantizationType)
				.build();

		} catch (Exception e) {
			throw new ConversionException("Conversion failed", e);
		}
	}

	public CompletableFuture<ConversionResult> convertAsync() {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return convert();
			} catch (ConversionException e) {
				throw new RuntimeException(e);
			}
		}, executor).orTimeout(timeout.toMillis(), TimeUnit.MILLISECONDS);
	}

	// Validation methods
	public ValidationResult validateSource() {
		ValidationResult.Builder builder = new ValidationResult.Builder();

		try {
			// Check source path exists
			builder.sourceExists(Files.exists(sourcePath));
			if (!Files.exists(sourcePath)) {
				builder.addError("Source path does not exist: " + sourcePath);
				return builder.build();
			}

			// Check if it's a directory
			builder.sourceIsDirectory(Files.isDirectory(sourcePath));

			// Look for config.json
			Path configPath = sourcePath.resolve("config.json");
			builder.hasConfigFile(Files.exists(configPath));

			if (Files.exists(configPath)) {
				try {
					ObjectMapper mapper = new ObjectMapper();
					JsonNode config = mapper.readTree(configPath.toFile());

					// Check for architecture
					if (config.has("architectures")) {
						builder.hasArchitecture(true);
						builder.architecture(config.get("architectures").get(0).asText());
					}

					// Check for model type
					if (config.has("model_type")) {
						builder.modelType(config.get("model_type").asText());
					}

				} catch (Exception e) {
					builder.addError("Failed to parse config.json: " + e.getMessage());
				}
			}

			// Look for model files
			List<Path> modelFiles = findModelFiles();
			builder.modelFileCount(modelFiles.size());
			builder.hasModelFiles(!modelFiles.isEmpty());

			if (modelFiles.isEmpty()) {
				builder.addError("No model files found (*.safetensors, *.bin, *.pth)");
			}

			// Check destination path
			if (destinationPath != null) {
				Path parentDir = destinationPath.getParent();
				if (parentDir != null && !Files.exists(parentDir)) {
					builder.addError("Destination directory does not exist: " + parentDir);
				}
			}

		} catch (Exception e) {
			builder.addError("Validation failed: " + e.getMessage());
		}

		return builder.build();
	}

	public ModelInfo analyzeSource() throws IOException {
		if (!Files.exists(sourcePath)) {
			throw new IOException("Source path does not exist: " + sourcePath);
		}

		return loadModelInfo();
	}

	public boolean isValidSource() {
		return validateSource().isValid();
	}

	// Helper methods
	private void validateInputs() throws ConversionException {
		if (sourcePath == null) {
			throw new ConversionException("Source path is required");
		}
		if (destinationPath == null) {
			throw new ConversionException("Destination path is required");
		}

		ValidationResult validation = validateSource();
		if (!validation.isValid()) {
			throw new ConversionException("Source validation failed: " +
				String.join(", ", validation.getErrors()));
		}
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new ConversionProgress(message, progress));
		}
		if (verbose) {
			logger.log(System.Logger.Level.INFO, String.format("[%.0f%%] %s", progress * 100, message));
		}
	}

	private ModelInfo loadModelInfo() throws IOException {
		Path configPath = sourcePath.resolve("config.json");
		if (!Files.exists(configPath)) {
			throw new IOException("config.json not found in: " + sourcePath);
		}

		ObjectMapper mapper = new ObjectMapper();
		JsonNode config = mapper.readTree(configPath.toFile());

		ModelInfo.Builder builder = new ModelInfo.Builder();

		if (config.has("architectures")) {
			builder.architecture(config.get("architectures").get(0).asText());
		}
		if (config.has("model_type")) {
			builder.modelType(config.get("model_type").asText());
		}
		if (config.has("vocab_size")) {
			builder.vocabSize(config.get("vocab_size").asInt());
		}
		if (config.has("hidden_size")) {
			builder.hiddenSize(config.get("hidden_size").asInt());
		}
		if (config.has("num_hidden_layers")) {
			builder.numLayers(config.get("num_hidden_layers").asInt());
		}
		if (config.has("num_attention_heads")) {
			builder.numHeads(config.get("num_attention_heads").asInt());
		}

		// Add custom metadata
		for (Map.Entry<String, String> entry : overrideMetadata.entrySet()) {
			builder.addMetadata(entry.getKey(), entry.getValue());
		}

		return builder.build();
	}

	private Map<String, TensorData> loadTensors() throws IOException {
		Map<String, TensorData> tensors = new HashMap<>();
		List<Path> modelFiles = findModelFiles();

		for (Path file : modelFiles) {
			if (file.toString().endsWith(".safetensors")) {
				tensors.putAll(loadSafeTensors(file));
			} else if (file.toString().endsWith(".bin") || file.toString().endsWith(".pth")) {
				tensors.putAll(loadPyTorchTensors(file));
			}
		}

		return tensors;
	}

	private List<Path> findModelFiles() throws IOException {
		if (!Files.isDirectory(sourcePath)) {
			return Collections.emptyList();
		}

		return Files.list(sourcePath)
			.filter(path -> {
				String name = path.getFileName().toString().toLowerCase();
				return name.endsWith(".safetensors") || name.endsWith(".bin") || name.endsWith(".pth");
			})
			.collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
	}

	private Map<String, TensorData> loadSafeTensors(Path file) throws IOException {
		// Placeholder implementation - would need actual SafeTensors parsing
		return new HashMap<>();
	}

	private Map<String, TensorData> loadPyTorchTensors(Path file) throws IOException {
		// Placeholder implementation - would need actual PyTorch parsing
		return new HashMap<>();
	}

	private Map<String, TensorData> convertTensors(Map<String, TensorData> tensors) {
		// Apply quantization if needed
		if (quantizationType != GGUFConstants.GGMLQuantizationType.F16) {
			return quantizeTensors(tensors);
		}
		return tensors;
	}

	private Map<String, TensorData> quantizeTensors(Map<String, TensorData> tensors) {
		// Placeholder for quantization logic
		return tensors;
	}

	private void writeGGUFFile(ModelInfo modelInfo, Map<String, TensorData> tensors) throws IOException {
		// Placeholder for GGUF writing logic
		// Would use GGUFWriter to write the final file
	}

	// Builder class
	public static class Builder {
		private Path sourcePath;
		private Path destinationPath;
		private GGUFConstants.GGMLQuantizationType quantizationType = GGUFConstants.GGMLQuantizationType.F16;
		private int threadCount = Runtime.getRuntime().availableProcessors();
		private boolean verbose = false;
		private final Map<String, String> overrideMetadata = new HashMap<>();
		private Consumer<ConversionProgress> progressCallback;
		private ExecutorService executor;
		private Duration timeout = Duration.ofMinutes(30);

		public Builder source(Path sourcePath) {
			this.sourcePath = sourcePath;
			return this;
		}

		public Builder source(String sourcePath) {
			return source(Path.of(sourcePath));
		}

		public Builder destination(Path destinationPath) {
			this.destinationPath = destinationPath;
			return this;
		}

		public Builder destination(String destinationPath) {
			return destination(Path.of(destinationPath));
		}

		public Builder quantization(GGUFConstants.GGMLQuantizationType quantizationType) {
			this.quantizationType = Objects.requireNonNull(quantizationType);
			return this;
		}

		public Builder threads(int threadCount) {
			this.threadCount = Math.max(1, threadCount);
			return this;
		}

		public Builder verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public Builder addMetadata(String key, String value) {
			this.overrideMetadata.put(key, value);
			return this;
		}

		public Builder progressCallback(Consumer<ConversionProgress> callback) {
			this.progressCallback = callback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public Builder timeout(Duration timeout) {
			this.timeout = Objects.requireNonNull(timeout);
			return this;
		}

		public HuggingFaceToGGUFConverterLibrary build() {
			Objects.requireNonNull(sourcePath, "Source path is required");
			Objects.requireNonNull(destinationPath, "Destination path is required");
			return new HuggingFaceToGGUFConverterLibrary(this);
		}
	}

	// Result classes
	public static class ConversionResult {
		private final boolean success;
		private final Path sourcePath;
		private final Path destinationPath;
		private final ModelInfo modelInfo;
		private final int tensorCount;
		private final Duration conversionTime;
		private final GGUFConstants.GGMLQuantizationType quantizationType;
		private final String error;

		private ConversionResult(Builder builder) {
			this.success = builder.success;
			this.sourcePath = builder.sourcePath;
			this.destinationPath = builder.destinationPath;
			this.modelInfo = builder.modelInfo;
			this.tensorCount = builder.tensorCount;
			this.conversionTime = builder.conversionTime;
			this.quantizationType = builder.quantizationType;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public Path getSourcePath() { return sourcePath; }
		public Path getDestinationPath() { return destinationPath; }
		public Optional<ModelInfo> getModelInfo() { return Optional.ofNullable(modelInfo); }
		public int getTensorCount() { return tensorCount; }
		public Optional<Duration> getConversionTime() { return Optional.ofNullable(conversionTime); }
		public Optional<GGUFConstants.GGMLQuantizationType> getQuantizationType() { return Optional.ofNullable(quantizationType); }
		public Optional<String> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private Path sourcePath;
			private Path destinationPath;
			private ModelInfo modelInfo;
			private int tensorCount;
			private Duration conversionTime;
			private GGUFConstants.GGMLQuantizationType quantizationType;
			private String error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder sourcePath(Path path) { this.sourcePath = path; return this; }
			public Builder destinationPath(Path path) { this.destinationPath = path; return this; }
			public Builder modelInfo(ModelInfo info) { this.modelInfo = info; return this; }
			public Builder tensorCount(int count) { this.tensorCount = count; return this; }
			public Builder conversionTime(Duration time) { this.conversionTime = time; return this; }
			public Builder quantizationType(GGUFConstants.GGMLQuantizationType type) { this.quantizationType = type; return this; }
			public Builder error(String error) { this.error = error; return this; }

			public ConversionResult build() {
				return new ConversionResult(this);
			}
		}
	}

	public static class ValidationResult {
		private final boolean sourceExists;
		private final boolean sourceIsDirectory;
		private final boolean hasConfigFile;
		private final boolean hasArchitecture;
		private final boolean hasModelFiles;
		private final String architecture;
		private final String modelType;
		private final int modelFileCount;
		private final List<String> errors;

		private ValidationResult(Builder builder) {
			this.sourceExists = builder.sourceExists;
			this.sourceIsDirectory = builder.sourceIsDirectory;
			this.hasConfigFile = builder.hasConfigFile;
			this.hasArchitecture = builder.hasArchitecture;
			this.hasModelFiles = builder.hasModelFiles;
			this.architecture = builder.architecture;
			this.modelType = builder.modelType;
			this.modelFileCount = builder.modelFileCount;
			this.errors = Collections.unmodifiableList(builder.errors);
		}

		public boolean isSourceExists() { return sourceExists; }
		public boolean isSourceIsDirectory() { return sourceIsDirectory; }
		public boolean isHasConfigFile() { return hasConfigFile; }
		public boolean isHasArchitecture() { return hasArchitecture; }
		public boolean isHasModelFiles() { return hasModelFiles; }
		public Optional<String> getArchitecture() { return Optional.ofNullable(architecture); }
		public Optional<String> getModelType() { return Optional.ofNullable(modelType); }
		public int getModelFileCount() { return modelFileCount; }
		public List<String> getErrors() { return errors; }

		public boolean isValid() {
			return sourceExists && hasConfigFile && hasModelFiles && errors.isEmpty();
		}

		public static class Builder {
			private boolean sourceExists;
			private boolean sourceIsDirectory;
			private boolean hasConfigFile;
			private boolean hasArchitecture;
			private boolean hasModelFiles;
			private String architecture;
			private String modelType;
			private int modelFileCount;
			private final List<String> errors = new ArrayList<>();

			public Builder sourceExists(boolean exists) { this.sourceExists = exists; return this; }
			public Builder sourceIsDirectory(boolean isDir) { this.sourceIsDirectory = isDir; return this; }
			public Builder hasConfigFile(boolean has) { this.hasConfigFile = has; return this; }
			public Builder hasArchitecture(boolean has) { this.hasArchitecture = has; return this; }
			public Builder hasModelFiles(boolean has) { this.hasModelFiles = has; return this; }
			public Builder architecture(String arch) { this.architecture = arch; return this; }
			public Builder modelType(String type) { this.modelType = type; return this; }
			public Builder modelFileCount(int count) { this.modelFileCount = count; return this; }
			public Builder addError(String error) { this.errors.add(error); return this; }

			public ValidationResult build() {
				return new ValidationResult(this);
			}
		}
	}

	public static class ModelInfo {
		private final String architecture;
		private final String modelType;
		private final int vocabSize;
		private final int hiddenSize;
		private final int numLayers;
		private final int numHeads;
		private final Map<String, String> metadata;

		private ModelInfo(Builder builder) {
			this.architecture = builder.architecture;
			this.modelType = builder.modelType;
			this.vocabSize = builder.vocabSize;
			this.hiddenSize = builder.hiddenSize;
			this.numLayers = builder.numLayers;
			this.numHeads = builder.numHeads;
			this.metadata = Collections.unmodifiableMap(builder.metadata);
		}

		public Optional<String> getArchitecture() { return Optional.ofNullable(architecture); }
		public Optional<String> getModelType() { return Optional.ofNullable(modelType); }
		public int getVocabSize() { return vocabSize; }
		public int getHiddenSize() { return hiddenSize; }
		public int getNumLayers() { return numLayers; }
		public int getNumHeads() { return numHeads; }
		public Map<String, String> getMetadata() { return metadata; }

		public static class Builder {
			private String architecture;
			private String modelType;
			private int vocabSize;
			private int hiddenSize;
			private int numLayers;
			private int numHeads;
			private final Map<String, String> metadata = new HashMap<>();

			public Builder architecture(String arch) { this.architecture = arch; return this; }
			public Builder modelType(String type) { this.modelType = type; return this; }
			public Builder vocabSize(int size) { this.vocabSize = size; return this; }
			public Builder hiddenSize(int size) { this.hiddenSize = size; return this; }
			public Builder numLayers(int layers) { this.numLayers = layers; return this; }
			public Builder numHeads(int heads) { this.numHeads = heads; return this; }
			public Builder addMetadata(String key, String value) { this.metadata.put(key, value); return this; }

			public ModelInfo build() {
				return new ModelInfo(this);
			}
		}
	}

	public static class TensorData {
		private final String name;
		private final long[] shape;
		private final String dtype;
		private final long size;

		public TensorData(String name, long[] shape, String dtype, long size) {
			this.name = name;
			this.shape = Arrays.copyOf(shape, shape.length);
			this.dtype = dtype;
			this.size = size;
		}

		public String getName() { return name; }
		public long[] getShape() { return Arrays.copyOf(shape, shape.length); }
		public String getDtype() { return dtype; }
		public long getSize() { return size; }
	}

	public static class ConversionProgress {
		private final String message;
		private final double progress;

		public ConversionProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
	}

	public static class ConversionException extends Exception {
		public ConversionException(String message) {
			super(message);
		}

		public ConversionException(String message, Throwable cause) {
			super(message, cause);
		}
	}
}
