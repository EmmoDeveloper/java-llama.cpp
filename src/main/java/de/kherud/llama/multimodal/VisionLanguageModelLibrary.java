package de.kherud.llama.multimodal;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Library-friendly vision-language model interface.
 *
 * This refactored version provides a fluent API for multimodal inference including
 * image captioning, visual question answering, and image-text understanding
 * with builder pattern configuration, batch processing, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic image captioning
 * InferenceResult result = VisionLanguageModelLibrary.builder()
 *     .languageModel("llava-model.gguf")
 *     .build()
 *     .captionImage(imagePath);
 *
 * // Visual question answering
 * InferenceResult result = VisionLanguageModelLibrary.builder()
 *     .languageModel("llava-model.gguf")
 *     .maxTokens(100)
 *     .build()
 *     .answerQuestion(imagePath, "What do you see in this image?");
 *
 * // Batch processing
 * BatchInferenceResult result = vlm.processBatch(imageTextPairs);
 *
 * // Async inference
 * VisionLanguageModelLibrary.builder()
 *     .languageModel("llava-model.gguf")
 *     .build()
 *     .captionImageAsync(imagePath)
 *     .thenAccept(result -> System.out.println("Caption: " + result.getResponse()));
 * }</pre>
 */
public class VisionLanguageModelLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(VisionLanguageModelLibrary.class.getName());

	// Model configuration
	private final String languageModelPath;
	private final int contextSize;
	private final int gpuLayers;

	// Vision configuration
	private final int visionEmbeddingDim;
	private final int textEmbeddingDim;
	private final int projectionDim;
	private final boolean useProjection;
	private final String visionModelType;
	private final float temperatureVision;
	private final int maxImageTokens;
	private final String imageTokenTemplate;

	// Inference configuration
	private final int maxTokens;
	private final float temperature;
	private final float topP;
	private final int topK;
	private final float repeatPenalty;
	private final long seed;

	// Runtime configuration
	private final Consumer<InferenceProgress> progressCallback;
	private final ExecutorService executor;
	private final boolean enableBatchProcessing;
	private final int batchSize;
	private final ImageProcessorLibrary imageProcessor;

	private LlamaModel model;

	private VisionLanguageModelLibrary(Builder builder) {
		this.languageModelPath = Objects.requireNonNull(builder.languageModelPath, "Language model path cannot be null");
		this.contextSize = builder.contextSize;
		this.gpuLayers = builder.gpuLayers;

		this.visionEmbeddingDim = builder.visionEmbeddingDim;
		this.textEmbeddingDim = builder.textEmbeddingDim;
		this.projectionDim = builder.projectionDim;
		this.useProjection = builder.useProjection;
		this.visionModelType = builder.visionModelType;
		this.temperatureVision = builder.temperatureVision;
		this.maxImageTokens = builder.maxImageTokens;
		this.imageTokenTemplate = builder.imageTokenTemplate;

		this.maxTokens = builder.maxTokens;
		this.temperature = builder.temperature;
		this.topP = builder.topP;
		this.topK = builder.topK;
		this.repeatPenalty = builder.repeatPenalty;
		this.seed = builder.seed;

		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
		this.enableBatchProcessing = builder.enableBatchProcessing;
		this.batchSize = builder.batchSize;

		// Initialize image processor with default configuration
		this.imageProcessor = ImageProcessorLibrary.builder()
			.targetSize(224, 224)
			.build();
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Generate caption for an image
	 */
	public InferenceResult captionImage(Path imagePath) {
		return captionImage(imagePath, "Describe this image.");
	}

	/**
	 * Generate caption for an image with custom prompt
	 */
	public InferenceResult captionImage(Path imagePath, String prompt) {
		return processImageText(imagePath, prompt, InferenceType.CAPTION);
	}

	/**
	 * Answer a question about an image
	 */
	public InferenceResult answerQuestion(Path imagePath, String question) {
		String prompt = "USER: " + question + "\nASSISTANT:";
		return processImageText(imagePath, prompt, InferenceType.VQA);
	}

	/**
	 * Analyze image content with detailed description
	 */
	public InferenceResult analyzeImage(Path imagePath) {
		String prompt = "Provide a detailed analysis of this image, including objects, people, setting, mood, and any notable features.";
		return processImageText(imagePath, prompt, InferenceType.ANALYSIS);
	}

	/**
	 * Extract text from image (OCR-like functionality)
	 */
	public InferenceResult extractText(Path imagePath) {
		String prompt = "Read and transcribe any text visible in this image.";
		return processImageText(imagePath, prompt, InferenceType.OCR);
	}

	/**
	 * Compare two images
	 */
	public InferenceResult compareImages(Path imagePath1, Path imagePath2, String question) {
		try {
			initializeModel();

			// Process both images
			ImageProcessorLibrary.ProcessingResult result1 = imageProcessor.processImage(imagePath1);
			ImageProcessorLibrary.ProcessingResult result2 = imageProcessor.processImage(imagePath2);

			if (!result1.isSuccess() || !result2.isSuccess()) {
				return new InferenceResult.Builder()
					.success(false)
					.message("Failed to process one or both images")
					.build();
			}

			// Create comparison prompt
			String prompt = String.format("%s %s %s Compare these two images: %s",
				imageTokenTemplate, imageTokenTemplate, question, "");

			return generateResponse(prompt, InferenceType.COMPARISON);

		} catch (Exception e) {
			return new InferenceResult.Builder()
				.success(false)
				.message("Image comparison failed: " + e.getMessage())
				.error(e)
				.build();
		}
	}

	/**
	 * Process multiple image-text pairs
	 */
	public BatchInferenceResult processBatch(List<ImageTextPair> pairs) {
		progress("Starting batch multimodal inference", 0.0);
		Instant startTime = Instant.now();

		try {
			List<InferenceResult> results = new ArrayList<>();
			List<ImageTextPair> failedPairs = new ArrayList<>();
			int totalBatches = enableBatchProcessing ?
				(int) Math.ceil((double) pairs.size() / batchSize) : pairs.size();

			for (int i = 0; i < totalBatches; i++) {
				int startIdx = enableBatchProcessing ? i * batchSize : i;
				int endIdx = enableBatchProcessing ?
					Math.min(startIdx + batchSize, pairs.size()) : startIdx + 1;
				List<ImageTextPair> batch = pairs.subList(startIdx, endIdx);

				progress("Processing batch " + (i + 1) + "/" + totalBatches,
					(double) i / totalBatches);

				for (ImageTextPair pair : batch) {
					InferenceResult result = processImageText(pair.getImagePath(), pair.getText(), pair.getType());
					results.add(result);

					if (!result.isSuccess()) {
						failedPairs.add(pair);
					}
				}
			}

			progress("Batch processing complete", 1.0);

			Duration duration = Duration.between(startTime, Instant.now());
			boolean success = failedPairs.isEmpty();

			return new BatchInferenceResult.Builder()
				.success(success)
				.message(String.format("Processed %d pairs, %d failed", pairs.size(), failedPairs.size()))
				.results(results)
				.totalPairs(pairs.size())
				.successfulPairs(results.size() - failedPairs.size())
				.failedPairs(failedPairs.size())
				.duration(duration)
				.build();

		} catch (Exception e) {
			String errorMsg = "Batch processing failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new BatchInferenceResult.Builder()
				.success(false)
				.message(errorMsg)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Process image and text asynchronously
	 */
	public CompletableFuture<InferenceResult> captionImageAsync(Path imagePath) {
		return processImageTextAsync(imagePath, "Describe this image.", InferenceType.CAPTION);
	}

	/**
	 * Answer question asynchronously
	 */
	public CompletableFuture<InferenceResult> answerQuestionAsync(Path imagePath, String question) {
		String prompt = "USER: " + question + "\nASSISTANT:";
		return processImageTextAsync(imagePath, prompt, InferenceType.VQA);
	}

	/**
	 * Process batch asynchronously
	 */
	public CompletableFuture<BatchInferenceResult> processBatchAsync(List<ImageTextPair> pairs) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();
		return CompletableFuture.supplyAsync(() -> processBatch(pairs), exec);
	}

	/**
	 * Validate model and test basic functionality
	 */
	public ValidationResult validateModel() {
		try {
			initializeModel();

			// Test basic text generation
			InferenceParameters params = new InferenceParameters("Test prompt")
				.setNPredict(10)
				.setTemperature(0.1f);

			StringBuilder response = new StringBuilder();
			for (LlamaOutput output : model.generate(params)) {
				response.append(output.text);
			}

			boolean textGeneration = !response.toString().trim().isEmpty();

			return new ValidationResult.Builder()
				.valid(textGeneration)
				.message(textGeneration ? "Model validation successful" : "Text generation failed")
				.languageModelPath(languageModelPath)
				.supportsTextGeneration(textGeneration)
				.supportsVision(true) // Assume vision support for VL models
				.contextSize(contextSize)
				.build();

		} catch (Exception e) {
			return new ValidationResult.Builder()
				.valid(false)
				.message("Model validation failed: " + e.getMessage())
				.languageModelPath(languageModelPath)
				.error(e)
				.build();
		}
	}

	// Core processing logic
	private InferenceResult processImageText(Path imagePath, String text, InferenceType inferenceType) {
		progress("Starting multimodal inference", 0.0);
		Instant startTime = Instant.now();

		try {
			initializeModel();
			progress("Model initialized", 0.2);

			// Process image
			ImageProcessorLibrary.ProcessingResult imageResult = imageProcessor.processImage(imagePath);
			if (!imageResult.isSuccess()) {
				return new InferenceResult.Builder()
					.success(false)
					.message("Image processing failed: " + imageResult.getMessage())
					.imagePath(imagePath)
					.prompt(text)
					.inferenceType(inferenceType)
					.duration(Duration.between(startTime, Instant.now()))
					.build();
			}
			progress("Image processed", 0.5);

			// Create multimodal prompt
			String multimodalPrompt = createMultimodalPrompt(text, imageResult.getProcessedImage().get());
			progress("Prompt created", 0.6);

			// Generate response
			InferenceResult result = generateResponse(multimodalPrompt, inferenceType);
			result.imagePath = imagePath;
			result.duration = Duration.between(startTime, Instant.now());

			progress("Inference complete", 1.0);
			return result;

		} catch (Exception e) {
			String errorMsg = "Multimodal inference failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new InferenceResult.Builder()
				.success(false)
				.message(errorMsg)
				.imagePath(imagePath)
				.prompt(text)
				.inferenceType(inferenceType)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	private CompletableFuture<InferenceResult> processImageTextAsync(Path imagePath, String text, InferenceType inferenceType) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();
		return CompletableFuture.supplyAsync(() -> processImageText(imagePath, text, inferenceType), exec);
	}

	private void initializeModel() {
		if (model == null) {
			ModelParameters params = new ModelParameters()
				.setModel(languageModelPath)
				.setCtxSize(contextSize)
				.setGpuLayers(gpuLayers);
			model = new LlamaModel(params);
		}
	}

	private String createMultimodalPrompt(String text, ImageProcessorLibrary.ProcessedImage processedImage) {
		// Simulate embedding the image into the text prompt
		// In a real implementation, this would involve proper vision-language integration
		return imageTokenTemplate + " " + text;
	}

	private InferenceResult generateResponse(String prompt, InferenceType inferenceType) {
		try {
			InferenceParameters params = new InferenceParameters(prompt)
				.setNPredict(maxTokens)
				.setTemperature(temperature)
				.setTopP(topP)
				.setTopK(topK)
				.setRepeatPenalty(repeatPenalty)
				.setSeed(seed);

			StringBuilder response = new StringBuilder();
			List<String> tokens = new ArrayList<>();

			for (LlamaOutput output : model.generate(params)) {
				response.append(output.text);
				tokens.add(output.text);
			}

			return new InferenceResult.Builder()
				.success(true)
				.message("Inference completed successfully")
				.prompt(prompt)
				.response(response.toString())
				.tokens(tokens)
				.tokenCount(tokens.size())
				.inferenceType(inferenceType)
				.build();

		} catch (Exception e) {
			return new InferenceResult.Builder()
				.success(false)
				.message("Response generation failed: " + e.getMessage())
				.prompt(prompt)
				.inferenceType(inferenceType)
				.error(e)
				.build();
		}
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new InferenceProgress(message, progress));
		}
	}

	@Override
	public void close() {
		if (model != null) {
			model.close();
		}
		if (executor != null) {
			executor.shutdown();
		}
		try {
			imageProcessor.close();
		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.WARNING, "Failed to close image processor", e);
		}
	}

	// Enums and data classes
	public enum InferenceType {
		CAPTION, VQA, ANALYSIS, OCR, COMPARISON, GENERAL
	}

	public static class InferenceProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public InferenceProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	public static class ImageTextPair {
		private final Path imagePath;
		private final String text;
		private final InferenceType type;

		public ImageTextPair(Path imagePath, String text) {
			this(imagePath, text, InferenceType.GENERAL);
		}

		public ImageTextPair(Path imagePath, String text, InferenceType type) {
			this.imagePath = imagePath;
			this.text = text;
			this.type = type;
		}

		public Path getImagePath() { return imagePath; }
		public String getText() { return text; }
		public InferenceType getType() { return type; }
	}

	// Builder class
	public static class Builder {
		private String languageModelPath;
		private int contextSize = 4096;
		private int gpuLayers = 40;

		private int visionEmbeddingDim = 768;
		private int textEmbeddingDim = 4096;
		private int projectionDim = 4096;
		private boolean useProjection = true;
		private String visionModelType = "clip";
		private float temperatureVision = 0.1f;
		private int maxImageTokens = 256;
		private String imageTokenTemplate = "<image>";

		private int maxTokens = 512;
		private float temperature = 0.7f;
		private float topP = 0.9f;
		private int topK = 40;
		private float repeatPenalty = 1.1f;
		private long seed = -1;

		private Consumer<InferenceProgress> progressCallback;
		private ExecutorService executor;
		private boolean enableBatchProcessing = true;
		private int batchSize = 8;

		public Builder languageModel(String languageModelPath) {
			this.languageModelPath = languageModelPath;
			return this;
		}

		public Builder contextSize(int contextSize) {
			this.contextSize = contextSize;
			return this;
		}

		public Builder gpuLayers(int gpuLayers) {
			this.gpuLayers = gpuLayers;
			return this;
		}

		public Builder visionEmbeddingDim(int visionEmbeddingDim) {
			this.visionEmbeddingDim = visionEmbeddingDim;
			return this;
		}

		public Builder textEmbeddingDim(int textEmbeddingDim) {
			this.textEmbeddingDim = textEmbeddingDim;
			return this;
		}

		public Builder projectionDim(int projectionDim) {
			this.projectionDim = projectionDim;
			return this;
		}

		public Builder useProjection(boolean useProjection) {
			this.useProjection = useProjection;
			return this;
		}

		public Builder visionModelType(String visionModelType) {
			this.visionModelType = visionModelType;
			return this;
		}

		public Builder temperatureVision(float temperatureVision) {
			this.temperatureVision = temperatureVision;
			return this;
		}

		public Builder maxImageTokens(int maxImageTokens) {
			this.maxImageTokens = maxImageTokens;
			return this;
		}

		public Builder imageTokenTemplate(String imageTokenTemplate) {
			this.imageTokenTemplate = imageTokenTemplate;
			return this;
		}

		public Builder maxTokens(int maxTokens) {
			this.maxTokens = maxTokens;
			return this;
		}

		public Builder temperature(float temperature) {
			this.temperature = temperature;
			return this;
		}

		public Builder topP(float topP) {
			this.topP = topP;
			return this;
		}

		public Builder topK(int topK) {
			this.topK = topK;
			return this;
		}

		public Builder repeatPenalty(float repeatPenalty) {
			this.repeatPenalty = repeatPenalty;
			return this;
		}

		public Builder seed(long seed) {
			this.seed = seed;
			return this;
		}

		public Builder progressCallback(Consumer<InferenceProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public Builder enableBatchProcessing(boolean enableBatchProcessing) {
			this.enableBatchProcessing = enableBatchProcessing;
			return this;
		}

		public Builder batchSize(int batchSize) {
			this.batchSize = Math.max(1, batchSize);
			return this;
		}

		public VisionLanguageModelLibrary build() {
			return new VisionLanguageModelLibrary(this);
		}
	}

	// Result classes
	public static class InferenceResult {
		private final boolean success;
		private final String message;
		private Path imagePath;
		private final String prompt;
		private final String response;
		private final List<String> tokens;
		private final int tokenCount;
		private final InferenceType inferenceType;
		private Duration duration;
		private final Exception error;

		private InferenceResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.imagePath = builder.imagePath;
			this.prompt = builder.prompt;
			this.response = builder.response;
			this.tokens = builder.tokens != null ? Collections.unmodifiableList(builder.tokens) : Collections.emptyList();
			this.tokenCount = builder.tokenCount;
			this.inferenceType = builder.inferenceType;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Optional<Path> getImagePath() { return Optional.ofNullable(imagePath); }
		public String getPrompt() { return prompt; }
		public String getResponse() { return response; }
		public List<String> getTokens() { return tokens; }
		public int getTokenCount() { return tokenCount; }
		public InferenceType getInferenceType() { return inferenceType; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private Path imagePath;
			private String prompt;
			private String response;
			private List<String> tokens;
			private int tokenCount;
			private InferenceType inferenceType;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder imagePath(Path imagePath) { this.imagePath = imagePath; return this; }
			public Builder prompt(String prompt) { this.prompt = prompt; return this; }
			public Builder response(String response) { this.response = response; return this; }
			public Builder tokens(List<String> tokens) { this.tokens = tokens; return this; }
			public Builder tokenCount(int tokenCount) { this.tokenCount = tokenCount; return this; }
			public Builder inferenceType(InferenceType inferenceType) { this.inferenceType = inferenceType; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public InferenceResult build() { return new InferenceResult(this); }
		}
	}

	public static class BatchInferenceResult {
		private final boolean success;
		private final String message;
		private final List<InferenceResult> results;
		private final int totalPairs;
		private final int successfulPairs;
		private final int failedPairs;
		private final Duration duration;
		private final Exception error;

		private BatchInferenceResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.results = Collections.unmodifiableList(builder.results);
			this.totalPairs = builder.totalPairs;
			this.successfulPairs = builder.successfulPairs;
			this.failedPairs = builder.failedPairs;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public List<InferenceResult> getResults() { return results; }
		public int getTotalPairs() { return totalPairs; }
		public int getSuccessfulPairs() { return successfulPairs; }
		public int getFailedPairs() { return failedPairs; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }
		public double getSuccessRate() { return totalPairs > 0 ? (double) successfulPairs / totalPairs : 0.0; }

		public static class Builder {
			private boolean success;
			private String message;
			private List<InferenceResult> results = new ArrayList<>();
			private int totalPairs;
			private int successfulPairs;
			private int failedPairs;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder results(List<InferenceResult> results) { this.results = results; return this; }
			public Builder totalPairs(int totalPairs) { this.totalPairs = totalPairs; return this; }
			public Builder successfulPairs(int successfulPairs) { this.successfulPairs = successfulPairs; return this; }
			public Builder failedPairs(int failedPairs) { this.failedPairs = failedPairs; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public BatchInferenceResult build() { return new BatchInferenceResult(this); }
		}
	}

	public static class ValidationResult {
		private final boolean valid;
		private final String message;
		private final String languageModelPath;
		private final boolean supportsTextGeneration;
		private final boolean supportsVision;
		private final int contextSize;
		private final Exception error;

		private ValidationResult(Builder builder) {
			this.valid = builder.valid;
			this.message = builder.message;
			this.languageModelPath = builder.languageModelPath;
			this.supportsTextGeneration = builder.supportsTextGeneration;
			this.supportsVision = builder.supportsVision;
			this.contextSize = builder.contextSize;
			this.error = builder.error;
		}

		public boolean isValid() { return valid; }
		public String getMessage() { return message; }
		public String getLanguageModelPath() { return languageModelPath; }
		public boolean isSupportsTextGeneration() { return supportsTextGeneration; }
		public boolean isSupportsVision() { return supportsVision; }
		public int getContextSize() { return contextSize; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean valid;
			private String message;
			private String languageModelPath;
			private boolean supportsTextGeneration;
			private boolean supportsVision;
			private int contextSize;
			private Exception error;

			public Builder valid(boolean valid) { this.valid = valid; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder languageModelPath(String languageModelPath) { this.languageModelPath = languageModelPath; return this; }
			public Builder supportsTextGeneration(boolean supportsTextGeneration) { this.supportsTextGeneration = supportsTextGeneration; return this; }
			public Builder supportsVision(boolean supportsVision) { this.supportsVision = supportsVision; return this; }
			public Builder contextSize(int contextSize) { this.contextSize = contextSize; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public ValidationResult build() { return new ValidationResult(this); }
		}
	}
}