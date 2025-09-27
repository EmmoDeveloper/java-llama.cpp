package de.kherud.llama.gguf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Library-friendly GGUF metadata editor.
 *
 * This refactored version provides a fluent API for editing GGUF metadata,
 * with builder pattern configuration, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic metadata editing
 * EditResult result = GGUFMetadataEditorLibrary.open(ggufPath)
 *     .set("general.name", "My Model")
 *     .set("general.version", "1.0")
 *     .delete("unnecessary.key")
 *     .apply();
 *
 * // Configured editing
 * EditResult result = GGUFMetadataEditorLibrary.open(ggufPath)
 *     .createBackup(true)
 *     .validateChanges(true)
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .set("tokenizer.ggml.tokens", tokenList)
 *     .apply();
 *
 * // Batch operations
 * List<GGUFMetadataEditor.MetadataOperation> ops = Arrays.asList(
 *     GGUFMetadataEditor.MetadataOperation.set("general.name", "Updated Model"),
 *     GGUFMetadataEditor.MetadataOperation.delete("old.key"),
 *     GGUFMetadataEditor.MetadataOperation.rename("old.name", "new.name")
 * );
 * EditResult result = GGUFMetadataEditorLibrary.open(ggufPath)
 *     .applyOperations(ops);
 *
 * // Async editing
 * GGUFMetadataEditorLibrary.open(ggufPath)
 *     .set("general.name", "Async Model")
 *     .applyAsync()
 *     .thenAccept(result -> System.out.println("Edit complete: " + result.getMessage()));
 * }</pre>
 */
public class GGUFMetadataEditorLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(GGUFMetadataEditorLibrary.class.getName());

	private final Path filePath;
	private final List<GGUFMetadataEditor.MetadataOperation> pendingOperations;
	private boolean createBackup;
	private boolean validateChanges;
	private boolean dryRun;
	private Consumer<EditProgress> progressCallback;
	private ExecutorService executor;

	private GGUFMetadataEditorLibrary(Builder builder) {
		this.filePath = Objects.requireNonNull(builder.filePath, "File path cannot be null");
		this.pendingOperations = new ArrayList<>();
		this.createBackup = builder.createBackup;
		this.validateChanges = builder.validateChanges;
		this.dryRun = builder.dryRun;
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static GGUFMetadataEditorLibrary open(Path filePath) throws IOException {
		return builder().filePath(filePath).build();
	}

	// Fluent API for building operations
	public GGUFMetadataEditorLibrary set(String key, Object value) {
		pendingOperations.add(GGUFMetadataEditor.MetadataOperation.set(key, value));
		return this;
	}

	public GGUFMetadataEditorLibrary delete(String key) {
		pendingOperations.add(GGUFMetadataEditor.MetadataOperation.delete(key));
		return this;
	}

	public GGUFMetadataEditorLibrary rename(String oldKey, String newKey) {
		pendingOperations.add(GGUFMetadataEditor.MetadataOperation.rename(oldKey, newKey));
		return this;
	}

	public GGUFMetadataEditorLibrary createBackup(boolean backup) {
		this.createBackup = backup;
		return this;
	}

	public GGUFMetadataEditorLibrary validateChanges(boolean validate) {
		this.validateChanges = validate;
		return this;
	}

	public GGUFMetadataEditorLibrary dryRun(boolean dryRun) {
		this.dryRun = dryRun;
		return this;
	}

	public GGUFMetadataEditorLibrary progressCallback(Consumer<EditProgress> callback) {
		this.progressCallback = callback;
		return this;
	}

	/**
	 * Apply all pending operations
	 */
	public EditResult apply() throws IOException {
		return applyOperations(new ArrayList<>(pendingOperations));
	}

	/**
	 * Apply specific operations (bypassing pending operations)
	 */
	public EditResult applyOperations(List<GGUFMetadataEditor.MetadataOperation> operations) throws IOException {
		validateFilePath();

		progress("Starting metadata edit operation", 0.0);
		Instant startTime = Instant.now();

		try {
			// Create backup if requested
			Path backupPath = null;
			if (createBackup) {
				progress("Creating backup", 0.1);
				backupPath = createBackup(filePath);
			}

			// Build edit options
			GGUFMetadataEditor.EditOptions options = new GGUFMetadataEditor.EditOptions()
				.backup(createBackup)
				.dryRun(dryRun);

			progress("Processing operations", 0.3);

			// Use the original editor for the actual work
			GGUFMetadataEditor editor = new GGUFMetadataEditor(filePath, options);
			GGUFMetadataEditor.EditResult originalResult = editor.applyOperations(operations);

			progress("Operations complete", 0.9);

			// Convert to our result format
			EditResult result = new EditResult.Builder()
				.success(originalResult.isSuccess())
				.message(originalResult.getMessage())
				.changedMetadata(originalResult.getChangedMetadata())
				.deletedKeys(originalResult.getDeletedKeys())
				.renamedKeys(originalResult.getRenamedKeys())
				.filePath(filePath)
				.backupPath(backupPath)
				.operationCount(operations.size())
				.duration(Duration.between(startTime, Instant.now()))
				.build();

			progress("Edit result ready", 1.0);
			return result;

		} catch (Exception e) {
			String errorMsg = "Metadata edit failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new EditResult.Builder()
				.success(false)
				.message(errorMsg)
				.filePath(filePath)
				.operationCount(operations.size())
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Apply pending operations asynchronously
	 */
	public CompletableFuture<EditResult> applyAsync() {
		List<GGUFMetadataEditor.MetadataOperation> operations = new ArrayList<>(pendingOperations);
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();

		return CompletableFuture.supplyAsync(() -> {
			try {
				return applyOperations(operations);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}, exec);
	}

	/**
	 * Get current metadata without applying changes
	 */
	public Map<String, Object> getCurrentMetadata() throws IOException {
		validateFilePath();

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(filePath)) {
			return inspector.inspect().getMetadata();
		}
	}

	/**
	 * Preview what changes would be made without applying them
	 */
	public EditPreview previewChanges() throws IOException {
		return previewOperations(new ArrayList<>(pendingOperations));
	}

	/**
	 * Preview what specific operations would do
	 */
	public EditPreview previewOperations(List<GGUFMetadataEditor.MetadataOperation> operations) throws IOException {
		Map<String, Object> currentMetadata = getCurrentMetadata();
		Map<String, Object> resultingMetadata = new LinkedHashMap<>(currentMetadata);

		Set<String> addedKeys = new HashSet<>();
		Set<String> modifiedKeys = new HashSet<>();
		Set<String> deletedKeys = new HashSet<>();
		Map<String, String> renamedKeys = new LinkedHashMap<>();

		for (GGUFMetadataEditor.MetadataOperation op : operations) {
			switch (op.getType()) {
				case SET:
					if (resultingMetadata.containsKey(op.getKey())) {
						modifiedKeys.add(op.getKey());
					} else {
						addedKeys.add(op.getKey());
					}
					resultingMetadata.put(op.getKey(), op.getValue());
					break;
				case DELETE:
					if (resultingMetadata.containsKey(op.getKey())) {
						deletedKeys.add(op.getKey());
						resultingMetadata.remove(op.getKey());
					}
					break;
				case RENAME:
					if (resultingMetadata.containsKey(op.getKey())) {
						Object value = resultingMetadata.remove(op.getKey());
						resultingMetadata.put(op.getNewKey(), value);
						renamedKeys.put(op.getKey(), op.getNewKey());
					}
					break;
			}
		}

		return new EditPreview.Builder()
			.currentMetadata(currentMetadata)
			.resultingMetadata(resultingMetadata)
			.addedKeys(addedKeys)
			.modifiedKeys(modifiedKeys)
			.deletedKeys(deletedKeys)
			.renamedKeys(renamedKeys)
			.operationCount(operations.size())
			.build();
	}

	/**
	 * Clear all pending operations
	 */
	public GGUFMetadataEditorLibrary clearOperations() {
		pendingOperations.clear();
		return this;
	}

	/**
	 * Get count of pending operations
	 */
	public int getPendingOperationCount() {
		return pendingOperations.size();
	}

	// Helper methods
	private void validateFilePath() throws IOException {
		if (!Files.exists(filePath)) {
			throw new IOException("File does not exist: " + filePath);
		}
		if (!Files.isReadable(filePath)) {
			throw new IOException("File is not readable: " + filePath);
		}
		if (!Files.isWritable(filePath)) {
			throw new IOException("File is not writable: " + filePath);
		}
	}

	private Path createBackup(Path originalPath) throws IOException {
		String timestamp = Instant.now().toString().replaceAll("[:.]", "-");
		Path backupPath = originalPath.resolveSibling(
			originalPath.getFileName() + ".backup-" + timestamp
		);
		Files.copy(originalPath, backupPath);
		return backupPath;
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new EditProgress(message, progress));
		}
	}

	@Override
	public void close() throws IOException {
		if (executor != null) {
			executor.shutdown();
		}
	}

	// Builder class
	public static class Builder {
		private Path filePath;
		private boolean createBackup = false;
		private boolean validateChanges = true;
		private boolean dryRun = false;
		private Consumer<EditProgress> progressCallback;
		private ExecutorService executor;

		public Builder filePath(Path filePath) {
			this.filePath = filePath;
			return this;
		}

		public Builder createBackup(boolean createBackup) {
			this.createBackup = createBackup;
			return this;
		}

		public Builder validateChanges(boolean validateChanges) {
			this.validateChanges = validateChanges;
			return this;
		}

		public Builder dryRun(boolean dryRun) {
			this.dryRun = dryRun;
			return this;
		}

		public Builder progressCallback(Consumer<EditProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public GGUFMetadataEditorLibrary build() throws IOException {
			return new GGUFMetadataEditorLibrary(this);
		}
	}

	// Progress tracking class
	public static class EditProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public EditProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	// Enhanced result class
	public static class EditResult {
		private final boolean success;
		private final String message;
		private final Map<String, Object> changedMetadata;
		private final List<String> deletedKeys;
		private final Map<String, String> renamedKeys;
		private final Path filePath;
		private final Path backupPath;
		private final int operationCount;
		private final Duration duration;
		private final Exception error;

		private EditResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.changedMetadata = Collections.unmodifiableMap(builder.changedMetadata);
			this.deletedKeys = Collections.unmodifiableList(builder.deletedKeys);
			this.renamedKeys = Collections.unmodifiableMap(builder.renamedKeys);
			this.filePath = builder.filePath;
			this.backupPath = builder.backupPath;
			this.operationCount = builder.operationCount;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Map<String, Object> getChangedMetadata() { return changedMetadata; }
		public List<String> getDeletedKeys() { return deletedKeys; }
		public Map<String, String> getRenamedKeys() { return renamedKeys; }
		public Path getFilePath() { return filePath; }
		public Optional<Path> getBackupPath() { return Optional.ofNullable(backupPath); }
		public int getOperationCount() { return operationCount; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public boolean hasChanges() {
			return !changedMetadata.isEmpty() || !deletedKeys.isEmpty() || !renamedKeys.isEmpty();
		}

		public static class Builder {
			private boolean success;
			private String message;
			private Map<String, Object> changedMetadata = new LinkedHashMap<>();
			private List<String> deletedKeys = new ArrayList<>();
			private Map<String, String> renamedKeys = new LinkedHashMap<>();
			private Path filePath;
			private Path backupPath;
			private int operationCount;
			private Duration duration;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder changedMetadata(Map<String, Object> changedMetadata) { this.changedMetadata = changedMetadata; return this; }
			public Builder deletedKeys(List<String> deletedKeys) { this.deletedKeys = deletedKeys; return this; }
			public Builder renamedKeys(Map<String, String> renamedKeys) { this.renamedKeys = renamedKeys; return this; }
			public Builder filePath(Path filePath) { this.filePath = filePath; return this; }
			public Builder backupPath(Path backupPath) { this.backupPath = backupPath; return this; }
			public Builder operationCount(int operationCount) { this.operationCount = operationCount; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public EditResult build() { return new EditResult(this); }
		}
	}

	// Preview result class
	public static class EditPreview {
		private final Map<String, Object> currentMetadata;
		private final Map<String, Object> resultingMetadata;
		private final Set<String> addedKeys;
		private final Set<String> modifiedKeys;
		private final Set<String> deletedKeys;
		private final Map<String, String> renamedKeys;
		private final int operationCount;

		private EditPreview(Builder builder) {
			this.currentMetadata = Collections.unmodifiableMap(builder.currentMetadata);
			this.resultingMetadata = Collections.unmodifiableMap(builder.resultingMetadata);
			this.addedKeys = Collections.unmodifiableSet(builder.addedKeys);
			this.modifiedKeys = Collections.unmodifiableSet(builder.modifiedKeys);
			this.deletedKeys = Collections.unmodifiableSet(builder.deletedKeys);
			this.renamedKeys = Collections.unmodifiableMap(builder.renamedKeys);
			this.operationCount = builder.operationCount;
		}

		public Map<String, Object> getCurrentMetadata() { return currentMetadata; }
		public Map<String, Object> getResultingMetadata() { return resultingMetadata; }
		public Set<String> getAddedKeys() { return addedKeys; }
		public Set<String> getModifiedKeys() { return modifiedKeys; }
		public Set<String> getDeletedKeys() { return deletedKeys; }
		public Map<String, String> getRenamedKeys() { return renamedKeys; }
		public int getOperationCount() { return operationCount; }

		public boolean hasChanges() {
			return !addedKeys.isEmpty() || !modifiedKeys.isEmpty() ||
				   !deletedKeys.isEmpty() || !renamedKeys.isEmpty();
		}

		public static class Builder {
			private Map<String, Object> currentMetadata = new LinkedHashMap<>();
			private Map<String, Object> resultingMetadata = new LinkedHashMap<>();
			private Set<String> addedKeys = new HashSet<>();
			private Set<String> modifiedKeys = new HashSet<>();
			private Set<String> deletedKeys = new HashSet<>();
			private Map<String, String> renamedKeys = new LinkedHashMap<>();
			private int operationCount;

			public Builder currentMetadata(Map<String, Object> currentMetadata) { this.currentMetadata = currentMetadata; return this; }
			public Builder resultingMetadata(Map<String, Object> resultingMetadata) { this.resultingMetadata = resultingMetadata; return this; }
			public Builder addedKeys(Set<String> addedKeys) { this.addedKeys = addedKeys; return this; }
			public Builder modifiedKeys(Set<String> modifiedKeys) { this.modifiedKeys = modifiedKeys; return this; }
			public Builder deletedKeys(Set<String> deletedKeys) { this.deletedKeys = deletedKeys; return this; }
			public Builder renamedKeys(Map<String, String> renamedKeys) { this.renamedKeys = renamedKeys; return this; }
			public Builder operationCount(int operationCount) { this.operationCount = operationCount; return this; }

			public EditPreview build() { return new EditPreview(this); }
		}
	}
}