package de.kherud.llama.examples;

import de.kherud.llama.multimodal.ImageProcessorLibrary;
import de.kherud.llama.multimodal.VisionLanguageModelLibrary;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Complete multimodal AI pipeline example showing integration between
 * ImageProcessorLibrary and VisionLanguageModelLibrary.
 * Demonstrates real-world usage patterns for AI IDE integration.
 */
public class MultimodalPipelineExample {
	private static final System.Logger LOGGER = System.getLogger(MultimodalPipelineExample.class.getName());

	public static void main(String[] args) {
		String modelPath = "models/llava-v1.5-7b-q4_k_m.gguf";

		MultimodalPipelineExample pipeline = new MultimodalPipelineExample();

		try {
			// Run pipeline examples
			pipeline.completeImageAnalysisPipeline(modelPath);
			pipeline.batchImageProcessingPipeline(modelPath);
			pipeline.aiIdeIntegrationExample(modelPath);
			pipeline.performanceOptimizedPipeline(modelPath);
			pipeline.errorHandlingExample(modelPath);

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Pipeline execution failed", e);
		}
	}

	/**
	 * Complete image analysis pipeline with preprocessing and inference
	 */
	public void completeImageAnalysisPipeline(String modelPath) {
		System.out.println("\n=== Complete Image Analysis Pipeline ===");

		// Create image processor with AI-optimized settings
		try (ImageProcessorLibrary imageProcessor = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.maintainAspectRatio(true)
				.centerCrop(true)
				.meanNormalization(new float[]{0.485f, 0.456f, 0.406f})
				.stdNormalization(new float[]{0.229f, 0.224f, 0.225f})
				.interpolation(ImageProcessorLibrary.InterpolationMethod.BICUBIC)
				.progressCallback(progress ->
					System.out.printf("Image Processing: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build();

			VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(200)
				.temperature(0.7f)
				.progressCallback(progress ->
					System.out.printf("VLM Inference: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build()) {

			Path imagePath = Paths.get("examples/images/complex_scene.jpg");
			Instant startTime = Instant.now();

			// Step 1: Validate image
			System.out.println("Step 1: Validating image...");
			ImageProcessorLibrary.ValidationResult validation = imageProcessor.validateImage(imagePath);
			if (!validation.isValid()) {
				System.err.println("Image validation failed: " + validation.getMessage());
				return;
			}
			System.out.printf("✓ Image valid: %dx%d %s%n",
				validation.getWidth(), validation.getHeight(), validation.getFormat());

			// Step 2: Process image
			System.out.println("Step 2: Processing image...");
			ImageProcessorLibrary.ProcessingResult processingResult = imageProcessor.processImage(imagePath);
			if (!processingResult.isSuccess()) {
				System.err.println("Image processing failed: " + processingResult.getMessage());
				return;
			}

			ImageProcessorLibrary.ProcessedImage processed = processingResult.getProcessedImage().get();
			ImageProcessorLibrary.ImageMetadata metadata = processingResult.getMetadata().get();
			System.out.printf("✓ Image processed: %dx%d → %dx%d (%.1f KB)%n",
				metadata.getOriginalWidth(), metadata.getOriginalHeight(),
				metadata.getProcessedWidth(), metadata.getProcessedHeight(),
				processed.flatten().length * 4.0 / 1024);

			// Step 3: Extract features
			System.out.println("Step 3: Extracting features...");
			ImageProcessorLibrary.FeatureExtractionResult featureResult = imageProcessor.extractFeatures(imagePath);
			if (!featureResult.isSuccess()) {
				System.err.println("Feature extraction failed: " + featureResult.getMessage());
				return;
			}
			System.out.printf("✓ Features extracted: %d dimensions%n", featureResult.getFeatureCount());

			// Step 4: Generate description
			System.out.println("Step 4: Generating AI description...");
			VisionLanguageModelLibrary.InferenceResult captionResult = vlm.captionImage(imagePath);
			if (!captionResult.isSuccess()) {
				System.err.println("Caption generation failed: " + captionResult.getMessage());
				return;
			}
			System.out.println("✓ AI Caption: " + captionResult.getResponse());

			// Step 5: Answer specific questions
			System.out.println("Step 5: Answering specific questions...");
			String[] questions = {
				"How many objects are visible in this image?",
				"What is the dominant color scheme?",
				"Describe the lighting conditions."
			};

			for (String question : questions) {
				VisionLanguageModelLibrary.InferenceResult qaResult = vlm.answerQuestion(imagePath, question);
				if (qaResult.isSuccess()) {
					System.out.printf("✓ Q: %s%n", question);
					System.out.printf("  A: %s%n", qaResult.getResponse());
				}
			}

			Duration totalTime = Duration.between(startTime, Instant.now());
			System.out.printf("Pipeline completed in %.2f seconds%n", totalTime.toMillis() / 1000.0);

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Complete pipeline failed", e);
		}
	}

	/**
	 * Batch processing pipeline for multiple images
	 */
	public void batchImageProcessingPipeline(String modelPath) {
		System.out.println("\n=== Batch Image Processing Pipeline ===");

		try (ImageProcessorLibrary imageProcessor = ImageProcessorLibrary.builder()
				.targetSize(256, 256)
				.batchSize(16)
				.enableMetrics(true)
				.progressCallback(progress ->
					System.out.printf("Batch Processing: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build();

			VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(100)
				.enableBatchProcessing(true)
				.batchSize(8)
				.progressCallback(progress ->
					System.out.printf("Batch Inference: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build()) {

			// Create dataset of images
			List<Path> imagePaths = new ArrayList<>();
			for (int i = 1; i <= 20; i++) {
				imagePaths.add(Paths.get(String.format("examples/images/dataset/img_%03d.jpg", i)));
			}

			Instant startTime = Instant.now();

			// Step 1: Batch image processing
			System.out.println("Step 1: Batch processing images...");
			ImageProcessorLibrary.BatchProcessingResult batchProcessing = imageProcessor.processBatch(imagePaths);

			if (!batchProcessing.isSuccess()) {
				System.err.println("Batch processing failed: " + batchProcessing.getMessage());
				return;
			}

			System.out.printf("✓ Processed %d/%d images (%.1f%% success rate)%n",
				batchProcessing.getSuccessfulImages(),
				batchProcessing.getTotalImages(),
				batchProcessing.getSuccessRate() * 100);

			// Step 2: Batch inference
			System.out.println("Step 2: Batch AI inference...");
			List<VisionLanguageModelLibrary.ImageTextPair> pairs = new ArrayList<>();

			for (Path imagePath : imagePaths) {
				pairs.add(new VisionLanguageModelLibrary.ImageTextPair(
					imagePath,
					"Provide a brief, one-sentence description of this image.",
					VisionLanguageModelLibrary.InferenceType.CAPTION
				));
			}

			VisionLanguageModelLibrary.BatchInferenceResult batchInference = vlm.processBatch(pairs);

			if (!batchInference.isSuccess()) {
				System.err.println("Batch inference failed: " + batchInference.getMessage());
				return;
			}

			System.out.printf("✓ Generated %d/%d captions (%.1f%% success rate)%n",
				batchInference.getSuccessfulPairs(),
				batchInference.getTotalPairs(),
				batchInference.getSuccessRate() * 100);

			// Step 3: Aggregate results
			System.out.println("Step 3: Aggregating results...");
			List<ImageAnalysisResult> results = new ArrayList<>();

			for (int i = 0; i < imagePaths.size(); i++) {
				Path imagePath = imagePaths.get(i);
				ImageProcessorLibrary.ProcessingResult procResult = batchProcessing.getResults().get(i);
				VisionLanguageModelLibrary.InferenceResult infResult = batchInference.getResults().get(i);

				results.add(new ImageAnalysisResult(
					imagePath,
					procResult.isSuccess(),
					infResult.isSuccess(),
					procResult.isSuccess() ? procResult.getMetadata().orElse(null) : null,
					infResult.isSuccess() ? infResult.getResponse() : null
				));
			}

			// Display summary
			Duration totalTime = Duration.between(startTime, Instant.now());
			System.out.printf("Batch pipeline completed in %.2f seconds%n", totalTime.toMillis() / 1000.0);
			System.out.printf("Average time per image: %.0f ms%n",
				totalTime.toMillis() / (double) imagePaths.size());

			// Show sample results
			System.out.println("Sample results:");
			for (int i = 0; i < Math.min(3, results.size()); i++) {
				ImageAnalysisResult result = results.get(i);
				System.out.printf("  %s: %s%n",
					result.imagePath.getFileName(),
					result.caption != null ? result.caption : "Failed");
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Batch pipeline failed", e);
		}
	}

	/**
	 * AI IDE integration example with code analysis
	 */
	public void aiIdeIntegrationExample(String modelPath) {
		System.out.println("\n=== AI IDE Integration Example ===");

		try (ImageProcessorLibrary imageProcessor = ImageProcessorLibrary.builder()
				.targetSize(512, 512)
				.maintainAspectRatio(true)
				.build();

			VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(300)
				.temperature(0.3f) // Lower temperature for more accurate analysis
				.build()) {

			// Simulate IDE scenarios
			System.out.println("Scenario 1: Code screenshot analysis");
			analyzeCodeScreenshot(imageProcessor, vlm, "examples/images/code_screenshot.png");

			System.out.println("\nScenario 2: UI mockup analysis");
			analyzeUIMockup(imageProcessor, vlm, "examples/images/ui_mockup.png");

			System.out.println("\nScenario 3: Diagram analysis");
			analyzeDiagram(imageProcessor, vlm, "examples/images/architecture_diagram.png");

			System.out.println("\nScenario 4: Documentation image analysis");
			analyzeDocumentation(imageProcessor, vlm, "examples/images/documentation.png");

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "AI IDE integration failed", e);
		}
	}

	/**
	 * Performance-optimized pipeline for high-throughput scenarios
	 */
	public void performanceOptimizedPipeline(String modelPath) {
		System.out.println("\n=== Performance Optimized Pipeline ===");

		ExecutorService executor = Executors.newFixedThreadPool(4);

		try (ImageProcessorLibrary imageProcessor = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.batchSize(32)
				.executor(executor)
				.build();

			VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(50) // Shorter responses for speed
				.temperature(0.5f)
				.enableBatchProcessing(true)
				.batchSize(16)
				.executor(executor)
				.build()) {

			List<Path> imagePaths = new ArrayList<>();
			for (int i = 1; i <= 100; i++) {
				imagePaths.add(Paths.get(String.format("examples/images/large_dataset/img_%04d.jpg", i)));
			}

			Instant startTime = Instant.now();

			// Parallel processing with async operations
			System.out.println("Starting parallel processing of 100 images...");

			CompletableFuture<ImageProcessorLibrary.BatchProcessingResult> processingFuture =
				imageProcessor.processBatchAsync(imagePaths);

			List<VisionLanguageModelLibrary.ImageTextPair> pairs = new ArrayList<>();
			for (Path imagePath : imagePaths) {
				pairs.add(new VisionLanguageModelLibrary.ImageTextPair(
					imagePath, "Brief description:", VisionLanguageModelLibrary.InferenceType.CAPTION));
			}

			CompletableFuture<VisionLanguageModelLibrary.BatchInferenceResult> inferenceFuture =
				vlm.processBatchAsync(pairs);

			// Wait for both to complete
			CompletableFuture<Void> combinedFuture = CompletableFuture.allOf(processingFuture, inferenceFuture)
				.thenRun(() -> {
					try {
						ImageProcessorLibrary.BatchProcessingResult procResult = processingFuture.get();
						VisionLanguageModelLibrary.BatchInferenceResult infResult = inferenceFuture.get();

						Duration totalTime = Duration.between(startTime, Instant.now());

						System.out.printf("Performance Results:%n");
						System.out.printf("  Total time: %.2f seconds%n", totalTime.toMillis() / 1000.0);
						System.out.printf("  Images processed: %d/%d%n",
							procResult.getSuccessfulImages(), procResult.getTotalImages());
						System.out.printf("  Captions generated: %d/%d%n",
							infResult.getSuccessfulPairs(), infResult.getTotalPairs());
						System.out.printf("  Throughput: %.1f images/second%n",
							imagePaths.size() / (totalTime.toMillis() / 1000.0));

					} catch (Exception e) {
						LOGGER.log(System.Logger.Level.ERROR, "Performance pipeline error", e);
					}
				});

			combinedFuture.join();

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Performance pipeline failed", e);
		} finally {
			executor.shutdown();
		}
	}

	/**
	 * Error handling and recovery example
	 */
	public void errorHandlingExample(String modelPath) {
		System.out.println("\n=== Error Handling Example ===");

		try (ImageProcessorLibrary imageProcessor = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.build();

			VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(100)
				.build()) {

			// Test various error scenarios
			String[] testCases = {
				"examples/images/valid.jpg",        // Should succeed
				"examples/images/nonexistent.jpg",  // File not found
				"examples/images/corrupted.jpg",    // Corrupted file
				"examples/images/unsupported.bmp",  // Unsupported format
				"examples/images/empty.txt"         // Not an image
			};

			for (String testCase : testCases) {
				Path imagePath = Paths.get(testCase);
				System.out.printf("Testing: %s%n", imagePath.getFileName());

				try {
					// Attempt validation first
					ImageProcessorLibrary.ValidationResult validation = imageProcessor.validateImage(imagePath);
					if (!validation.isValid()) {
						System.out.printf("  ❌ Validation failed: %s%n", validation.getMessage());
						continue;
					}

					// Attempt processing
					ImageProcessorLibrary.ProcessingResult processingResult = imageProcessor.processImage(imagePath);
					if (!processingResult.isSuccess()) {
						System.out.printf("  ❌ Processing failed: %s%n", processingResult.getMessage());
						continue;
					}

					// Attempt inference
					VisionLanguageModelLibrary.InferenceResult inferenceResult = vlm.captionImage(imagePath);
					if (!inferenceResult.isSuccess()) {
						System.out.printf("  ❌ Inference failed: %s%n", inferenceResult.getMessage());
						continue;
					}

					System.out.printf("  ✅ Success: %s%n", inferenceResult.getResponse());

				} catch (Exception e) {
					System.out.printf("  ❌ Exception: %s%n", e.getMessage());
				}
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Error handling example failed", e);
		}
	}

	// Helper methods for IDE scenarios
	private void analyzeCodeScreenshot(ImageProcessorLibrary imageProcessor, VisionLanguageModelLibrary vlm, String imagePath) {
		Path path = Paths.get(imagePath);
		VisionLanguageModelLibrary.InferenceResult result = vlm.answerQuestion(path,
			"Analyze this code screenshot. What programming language is used and what does the code do?");

		if (result.isSuccess()) {
			System.out.println("Code Analysis: " + result.getResponse());
		} else {
			System.err.println("Code analysis failed: " + result.getMessage());
		}
	}

	private void analyzeUIMockup(ImageProcessorLibrary imageProcessor, VisionLanguageModelLibrary vlm, String imagePath) {
		Path path = Paths.get(imagePath);
		VisionLanguageModelLibrary.InferenceResult result = vlm.answerQuestion(path,
			"Describe the UI elements and layout in this mockup. What type of application is this?");

		if (result.isSuccess()) {
			System.out.println("UI Analysis: " + result.getResponse());
		} else {
			System.err.println("UI analysis failed: " + result.getMessage());
		}
	}

	private void analyzeDiagram(ImageProcessorLibrary imageProcessor, VisionLanguageModelLibrary vlm, String imagePath) {
		Path path = Paths.get(imagePath);
		VisionLanguageModelLibrary.InferenceResult result = vlm.answerQuestion(path,
			"Explain the architecture or flow shown in this diagram. What are the main components?");

		if (result.isSuccess()) {
			System.out.println("Diagram Analysis: " + result.getResponse());
		} else {
			System.err.println("Diagram analysis failed: " + result.getMessage());
		}
	}

	private void analyzeDocumentation(ImageProcessorLibrary imageProcessor, VisionLanguageModelLibrary vlm, String imagePath) {
		Path path = Paths.get(imagePath);
		VisionLanguageModelLibrary.InferenceResult result = vlm.extractText(path);

		if (result.isSuccess()) {
			System.out.println("Documentation Text: " + result.getResponse());
		} else {
			System.err.println("Documentation analysis failed: " + result.getMessage());
		}
	}

	// Helper data class
	private static class ImageAnalysisResult {
		public final Path imagePath;
		public final boolean processingSuccess;
		public final boolean inferenceSuccess;
		public final ImageProcessorLibrary.ImageMetadata metadata;
		public final String caption;

		public ImageAnalysisResult(Path imagePath, boolean processingSuccess, boolean inferenceSuccess,
								 ImageProcessorLibrary.ImageMetadata metadata, String caption) {
			this.imagePath = imagePath;
			this.processingSuccess = processingSuccess;
			this.inferenceSuccess = inferenceSuccess;
			this.metadata = metadata;
			this.caption = caption;
		}
	}
}