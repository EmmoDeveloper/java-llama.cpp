package de.kherud.llama.converters;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

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
 * Library-friendly LoRA to GGUF converter.
 *
 * This refactored version provides a fluent API for converting LoRA adapters to GGUF format,
 * with builder pattern configuration, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic conversion
 * ConversionResult result = LoRAToGGUFConverterLibrary.builder()
 *     .adapterPath(Paths.get("lora_adapter/"))
 *     .outputPath(Paths.get("output.gguf"))
 *     .build()
 *     .convert();
 *
 * // Configured conversion
 * ConversionResult result = LoRAToGGUFConverterLibrary.builder()
 *     .adapterPath(adapterPath)
 *     .outputPath(outputPath)
 *     .baseModelArchitecture("llama")
 *     .mergeLayerNorms(true)
 *     .verbose(true)
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .build()
 *     .convert();
 *
 * // Async conversion
 * LoRAToGGUFConverterLibrary.builder()
 *     .adapterPath(adapterPath)
 *     .outputPath(outputPath)
 *     .build()
 *     .convertAsync()
 *     .thenAccept(result -> System.out.println("Conversion complete: " + result.isSuccess()));
 *
 * // Batch conversion
 * List<ConversionTask> tasks = Arrays.asList(
 *     new ConversionTask(adapter1, output1),
 *     new ConversionTask(adapter2, output2)
 * );
 * BatchConversionResult batchResult = converter.convertMultiple(tasks);
 * }</pre>
 */
public class LoRAToGGUFConverterLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(LoRAToGGUFConverterLibrary.class.getName());

	private final Path adapterPath;
	private final Path outputPath;
	private final boolean verbose;
	private final boolean mergeLayerNorms;
	private final String baseModelArchitecture;
	private final Map<String, String> targetModules;
	private final boolean validateTensors;
	private final Consumer<ConversionProgress> progressCallback;
	private final ExecutorService executor;

	private LoRAToGGUFConverterLibrary(Builder builder) {
		this.adapterPath = Objects.requireNonNull(builder.adapterPath, "Adapter path cannot be null");
		this.outputPath = Objects.requireNonNull(builder.outputPath, "Output path cannot be null");
		this.verbose = builder.verbose;
		this.mergeLayerNorms = builder.mergeLayerNorms;
		this.baseModelArchitecture = builder.baseModelArchitecture;
		this.targetModules = Collections.unmodifiableMap(builder.targetModules);
		this.validateTensors = builder.validateTensors;
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Convert LoRA adapter to GGUF format
	 */
	public ConversionResult convert() throws IOException {
		validatePaths();

		progress("Starting LoRA to GGUF conversion", 0.0);
		Instant startTime = Instant.now();

		try {
			// Build conversion config for original converter
			LoRAToGGUFConverter.ConversionConfig config = new LoRAToGGUFConverter.ConversionConfig()
				.verbose(verbose)
				.mergeLayerNorms(mergeLayerNorms)
				.baseModelArchitecture(baseModelArchitecture);

			// Add target modules
			for (Map.Entry<String, String> entry : targetModules.entrySet()) {
				config.addTargetModule(entry.getKey(), entry.getValue());
			}

			progress("Loading adapter configuration", 0.2);

			// Use the original converter for the actual work
			LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterPath, outputPath, config);
			converter.convert();

			progress("Conversion complete", 1.0);

			// Get file info for result
			long outputSize = Files.exists(outputPath) ? Files.size(outputPath) : 0;

			// Extract adapter info
			AdapterInfo adapterInfo = extractAdapterInfo();

			return new ConversionResult.Builder()
				.success(true)
				.message("Conversion successful")
				.adapterPath(adapterPath)
				.outputPath(outputPath)
				.outputSize(outputSize)
				.duration(Duration.between(startTime, Instant.now()))
				.adapterInfo(adapterInfo)
				.build();

		} catch (Exception e) {
			String errorMsg = "Conversion failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new ConversionResult.Builder()
				.success(false)
				.message(errorMsg)
				.adapterPath(adapterPath)
				.outputPath(outputPath)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Convert LoRA adapter asynchronously
	 */
	public CompletableFuture<ConversionResult> convertAsync() {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();

		return CompletableFuture.supplyAsync(() -> {
			try {
				return convert();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}, exec);
	}

	/**
	 * Convert multiple LoRA adapters
	 */
	public BatchConversionResult convertMultiple(List<ConversionTask> tasks) throws IOException {
		List<ConversionResult> results = new ArrayList<>();
		int totalTasks = tasks.size();
		int successCount = 0;
		int errorCount = 0;
		Duration totalDuration = Duration.ZERO;

		progress("Starting batch conversion of " + totalTasks + " adapters", 0.0);

		for (int i = 0; i < totalTasks; i++) {
			ConversionTask task = tasks.get(i);
			progress("Converting adapter " + (i + 1) + "/" + totalTasks, (double) i / totalTasks);

			try {
				// Create a new converter for each task
				LoRAToGGUFConverterLibrary taskConverter = new Builder()
					.adapterPath(task.getAdapterPath())
					.outputPath(task.getOutputPath())
					.verbose(verbose)
					.mergeLayerNorms(mergeLayerNorms)
					.baseModelArchitecture(baseModelArchitecture)
					.targetModules(targetModules)
					.validateTensors(validateTensors)
					.build();

				ConversionResult result = taskConverter.convert();
				results.add(result);

				if (result.isSuccess()) {
					successCount++;
				} else {
					errorCount++;
				}
				totalDuration = totalDuration.plus(result.getDuration());

			} catch (IOException e) {
				errorCount++;
				results.add(new ConversionResult.Builder()
					.success(false)
					.message("Failed to convert: " + e.getMessage())
					.adapterPath(task.getAdapterPath())
					.outputPath(task.getOutputPath())
					.error(e)
					.build());
			}
		}

		progress("Batch conversion complete", 1.0);

		return new BatchConversionResult.Builder()
			.tasks(tasks)
			.results(results)
			.totalTasks(totalTasks)
			.successCount(successCount)
			.errorCount(errorCount)
			.totalDuration(totalDuration)
			.build();
	}

	/**
	 * Validate adapter without converting
	 */
	public ValidationResult validateAdapter() throws IOException {
		validatePaths();

		progress("Validating LoRA adapter", 0.5);

		try {
			// Check if adapter files exist
			boolean hasConfig = Files.exists(adapterPath.resolve("adapter_config.json"));
			boolean hasBinModel = Files.exists(adapterPath.resolve("adapter_model.bin"));
			boolean hasSafetensors = Files.exists(adapterPath.resolve("adapter_model.safetensors"));

			if (!hasConfig && !hasBinModel && !hasSafetensors) {
				return new ValidationResult(false, "No valid adapter files found", adapterPath);
			}

			// Try to extract adapter info
			AdapterInfo adapterInfo = extractAdapterInfo();

			progress("Validation complete", 1.0);

			return new ValidationResult(true, "Adapter is valid", adapterPath, adapterInfo);

		} catch (Exception e) {
			return new ValidationResult(false, "Validation failed: " + e.getMessage(), adapterPath);
		}
	}

	// Helper methods
	private void validatePaths() throws IOException {
		if (!Files.exists(adapterPath)) {
			throw new IOException("Adapter path does not exist: " + adapterPath);
		}
		if (!Files.isDirectory(adapterPath) && !adapterPath.toString().endsWith(".safetensors")) {
			throw new IOException("Adapter path must be a directory or safetensors file: " + adapterPath);
		}

		// Create output directory if needed
		Path outputDir = outputPath.getParent();
		if (outputDir != null && !Files.exists(outputDir)) {
			Files.createDirectories(outputDir);
		}
	}

	private AdapterInfo extractAdapterInfo() {
		try {
			Path configPath = adapterPath.resolve("adapter_config.json");
			if (Files.exists(configPath)) {
				ObjectMapper mapper = new ObjectMapper();
				JsonNode config = mapper.readTree(Files.newBufferedReader(configPath));

				return new AdapterInfo.Builder()
					.loraAlpha(config.path("lora_alpha").floatValue())
					.loraRank(config.path("r").asInt())
					.taskType(config.path("task_type").asText(""))
					.baseModelName(config.path("base_model_name_or_path").asText(""))
					.build();
			}
		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.WARNING, "Failed to extract adapter info", e);
		}

		return new AdapterInfo.Builder().build();
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new ConversionProgress(message, progress));
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
		private Path adapterPath;
		private Path outputPath;
		private boolean verbose = false;
		private boolean mergeLayerNorms = false;
		private String baseModelArchitecture = "llama";
		private Map<String, String> targetModules = new HashMap<>();
		private boolean validateTensors = true;
		private Consumer<ConversionProgress> progressCallback;
		private ExecutorService executor;

		public Builder adapterPath(Path adapterPath) {
			this.adapterPath = adapterPath;
			return this;
		}

		public Builder outputPath(Path outputPath) {
			this.outputPath = outputPath;
			return this;
		}

		public Builder verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public Builder mergeLayerNorms(boolean mergeLayerNorms) {
			this.mergeLayerNorms = mergeLayerNorms;
			return this;
		}

		public Builder baseModelArchitecture(String baseModelArchitecture) {
			this.baseModelArchitecture = baseModelArchitecture;
			return this;
		}

		public Builder targetModule(String key, String value) {
			this.targetModules.put(key, value);
			return this;
		}

		public Builder targetModules(Map<String, String> targetModules) {
			this.targetModules.putAll(targetModules);
			return this;
		}

		public Builder validateTensors(boolean validateTensors) {
			this.validateTensors = validateTensors;
			return this;
		}

		public Builder progressCallback(Consumer<ConversionProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public LoRAToGGUFConverterLibrary build() {
			return new LoRAToGGUFConverterLibrary(this);
		}
	}

	// Progress tracking class
	public static class ConversionProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public ConversionProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	// Result classes
	public static class ConversionResult {
		private final boolean success;
		private final String message;
		private final Path adapterPath;
		private final Path outputPath;
		private final long outputSize;
		private final Duration duration;
		private final AdapterInfo adapterInfo;
		private final Exception error;

		private ConversionResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.adapterPath = builder.adapterPath;
			this.outputPath = builder.outputPath;
			this.outputSize = builder.outputSize;
			this.duration = builder.duration;
			this.adapterInfo = builder.adapterInfo;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Path getAdapterPath() { return adapterPath; }
		public Path getOutputPath() { return outputPath; }
		public long getOutputSize() { return outputSize; }
		public Duration getDuration() { return duration; }
		public Optional<AdapterInfo> getAdapterInfo() { return Optional.ofNullable(adapterInfo); }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private Path adapterPath;
			private Path outputPath;
			private long outputSize;
			private Duration duration = Duration.ZERO;
			private AdapterInfo adapterInfo;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder adapterPath(Path adapterPath) { this.adapterPath = adapterPath; return this; }
			public Builder outputPath(Path outputPath) { this.outputPath = outputPath; return this; }
			public Builder outputSize(long outputSize) { this.outputSize = outputSize; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder adapterInfo(AdapterInfo adapterInfo) { this.adapterInfo = adapterInfo; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public ConversionResult build() { return new ConversionResult(this); }
		}
	}

	public static class BatchConversionResult {
		private final List<ConversionTask> tasks;
		private final List<ConversionResult> results;
		private final int totalTasks;
		private final int successCount;
		private final int errorCount;
		private final Duration totalDuration;

		private BatchConversionResult(Builder builder) {
			this.tasks = Collections.unmodifiableList(builder.tasks);
			this.results = Collections.unmodifiableList(builder.results);
			this.totalTasks = builder.totalTasks;
			this.successCount = builder.successCount;
			this.errorCount = builder.errorCount;
			this.totalDuration = builder.totalDuration;
		}

		public List<ConversionTask> getTasks() { return tasks; }
		public List<ConversionResult> getResults() { return results; }
		public int getTotalTasks() { return totalTasks; }
		public int getSuccessCount() { return successCount; }
		public int getErrorCount() { return errorCount; }
		public Duration getTotalDuration() { return totalDuration; }

		public List<ConversionResult> getSuccessfulConversions() {
			return results.stream().filter(ConversionResult::isSuccess).toList();
		}

		public List<ConversionResult> getFailedConversions() {
			return results.stream().filter(result -> !result.isSuccess()).toList();
		}

		public static class Builder {
			private List<ConversionTask> tasks = new ArrayList<>();
			private List<ConversionResult> results = new ArrayList<>();
			private int totalTasks;
			private int successCount;
			private int errorCount;
			private Duration totalDuration = Duration.ZERO;

			public Builder tasks(List<ConversionTask> tasks) { this.tasks = tasks; return this; }
			public Builder results(List<ConversionResult> results) { this.results = results; return this; }
			public Builder totalTasks(int totalTasks) { this.totalTasks = totalTasks; return this; }
			public Builder successCount(int successCount) { this.successCount = successCount; return this; }
			public Builder errorCount(int errorCount) { this.errorCount = errorCount; return this; }
			public Builder totalDuration(Duration totalDuration) { this.totalDuration = totalDuration; return this; }

			public BatchConversionResult build() { return new BatchConversionResult(this); }
		}
	}

	public static class ConversionTask {
		private final Path adapterPath;
		private final Path outputPath;

		public ConversionTask(Path adapterPath, Path outputPath) {
			this.adapterPath = Objects.requireNonNull(adapterPath);
			this.outputPath = Objects.requireNonNull(outputPath);
		}

		public Path getAdapterPath() { return adapterPath; }
		public Path getOutputPath() { return outputPath; }
	}

	public static class AdapterInfo {
		private final float loraAlpha;
		private final int loraRank;
		private final String taskType;
		private final String baseModelName;

		private AdapterInfo(Builder builder) {
			this.loraAlpha = builder.loraAlpha;
			this.loraRank = builder.loraRank;
			this.taskType = builder.taskType;
			this.baseModelName = builder.baseModelName;
		}

		public float getLoraAlpha() { return loraAlpha; }
		public int getLoraRank() { return loraRank; }
		public String getTaskType() { return taskType; }
		public String getBaseModelName() { return baseModelName; }

		public static class Builder {
			private float loraAlpha = 16.0f;
			private int loraRank = 0;
			private String taskType = "";
			private String baseModelName = "";

			public Builder loraAlpha(float loraAlpha) { this.loraAlpha = loraAlpha; return this; }
			public Builder loraRank(int loraRank) { this.loraRank = loraRank; return this; }
			public Builder taskType(String taskType) { this.taskType = taskType; return this; }
			public Builder baseModelName(String baseModelName) { this.baseModelName = baseModelName; return this; }

			public AdapterInfo build() { return new AdapterInfo(this); }
		}
	}

	public static class ValidationResult {
		private final boolean valid;
		private final String message;
		private final Path adapterPath;
		private final AdapterInfo adapterInfo;

		public ValidationResult(boolean valid, String message, Path adapterPath) {
			this(valid, message, adapterPath, null);
		}

		public ValidationResult(boolean valid, String message, Path adapterPath, AdapterInfo adapterInfo) {
			this.valid = valid;
			this.message = message;
			this.adapterPath = adapterPath;
			this.adapterInfo = adapterInfo;
		}

		public boolean isValid() { return valid; }
		public String getMessage() { return message; }
		public Path getAdapterPath() { return adapterPath; }
		public Optional<AdapterInfo> getAdapterInfo() { return Optional.ofNullable(adapterInfo); }
	}
}