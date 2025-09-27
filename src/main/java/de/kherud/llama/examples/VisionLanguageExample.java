package de.kherud.llama.examples;

import de.kherud.llama.multimodal.ImageProcessorLibrary;
import de.kherud.llama.multimodal.VisionLanguageModelLibrary;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Comprehensive examples demonstrating VisionLanguageModelLibrary usage.
 * Shows various multimodal AI capabilities including image captioning,
 * visual question answering, and batch processing.
 */
public class VisionLanguageExample {
	private static final System.Logger LOGGER = System.getLogger(VisionLanguageExample.class.getName());

	public static void main(String[] args) {
		// Example model path - replace with actual model
		String modelPath = "models/llava-v1.5-7b-q4_k_m.gguf";

		VisionLanguageExample example = new VisionLanguageExample();

		try {
			// Run all examples
			example.basicImageCaptioning(modelPath);
			example.visualQuestionAnswering(modelPath);
			example.detailedImageAnalysis(modelPath);
			example.ocrExample(modelPath);
			example.imageComparison(modelPath);
			example.batchProcessingExample(modelPath);
			example.asyncProcessingExample(modelPath);
			example.modelValidationExample(modelPath);

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Example execution failed", e);
		}
	}

	/**
	 * Basic image captioning example
	 */
	public void basicImageCaptioning(String modelPath) {
		System.out.println("\n=== Basic Image Captioning ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(100)
				.temperature(0.7f)
				.progressCallback(progress ->
					System.out.printf("Progress: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build()) {

			Path imagePath = Paths.get("examples/images/sample.jpg");

			// Generate caption
			VisionLanguageModelLibrary.InferenceResult result = vlm.captionImage(imagePath);

			if (result.isSuccess()) {
				System.out.println("Caption: " + result.getResponse());
				System.out.println("Generated " + result.getTokenCount() + " tokens");
				System.out.println("Duration: " + result.getDuration().toMillis() + "ms");
			} else {
				System.err.println("Captioning failed: " + result.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Basic captioning failed", e);
		}
	}

	/**
	 * Visual Question Answering (VQA) example
	 */
	public void visualQuestionAnswering(String modelPath) {
		System.out.println("\n=== Visual Question Answering ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(150)
				.temperature(0.3f) // Lower temperature for more factual answers
				.build()) {

			Path imagePath = Paths.get("examples/images/office.jpg");

			// Ask various questions about the image
			String[] questions = {
				"How many people are in this image?",
				"What objects can you see on the desk?",
				"What is the dominant color in this image?",
				"Is this an indoor or outdoor scene?"
			};

			for (String question : questions) {
				VisionLanguageModelLibrary.InferenceResult result = vlm.answerQuestion(imagePath, question);

				if (result.isSuccess()) {
					System.out.println("Q: " + question);
					System.out.println("A: " + result.getResponse());
					System.out.println();
				} else {
					System.err.println("Failed to answer: " + question + " - " + result.getMessage());
				}
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "VQA example failed", e);
		}
	}

	/**
	 * Detailed image analysis example
	 */
	public void detailedImageAnalysis(String modelPath) {
		System.out.println("\n=== Detailed Image Analysis ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(300)
				.temperature(0.5f)
				.build()) {

			Path imagePath = Paths.get("examples/images/landscape.jpg");

			VisionLanguageModelLibrary.InferenceResult result = vlm.analyzeImage(imagePath);

			if (result.isSuccess()) {
				System.out.println("Detailed Analysis:");
				System.out.println(result.getResponse());
				System.out.println("\nAnalysis completed in: " + result.getDuration().toMillis() + "ms");
			} else {
				System.err.println("Analysis failed: " + result.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Image analysis failed", e);
		}
	}

	/**
	 * OCR (Optical Character Recognition) example
	 */
	public void ocrExample(String modelPath) {
		System.out.println("\n=== OCR Example ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(200)
				.temperature(0.1f) // Very low temperature for accuracy
				.build()) {

			Path imagePath = Paths.get("examples/images/document.jpg");

			VisionLanguageModelLibrary.InferenceResult result = vlm.extractText(imagePath);

			if (result.isSuccess()) {
				System.out.println("Extracted Text:");
				System.out.println("\"" + result.getResponse() + "\"");
			} else {
				System.err.println("OCR failed: " + result.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "OCR example failed", e);
		}
	}

	/**
	 * Image comparison example
	 */
	public void imageComparison(String modelPath) {
		System.out.println("\n=== Image Comparison ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(200)
				.temperature(0.4f)
				.build()) {

			Path image1 = Paths.get("examples/images/cat1.jpg");
			Path image2 = Paths.get("examples/images/cat2.jpg");

			String question = "What are the main differences between these two images?";

			VisionLanguageModelLibrary.InferenceResult result = vlm.compareImages(image1, image2, question);

			if (result.isSuccess()) {
				System.out.println("Comparison Result:");
				System.out.println(result.getResponse());
			} else {
				System.err.println("Comparison failed: " + result.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Image comparison failed", e);
		}
	}

	/**
	 * Batch processing example
	 */
	public void batchProcessingExample(String modelPath) {
		System.out.println("\n=== Batch Processing ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(100)
				.temperature(0.6f)
				.enableBatchProcessing(true)
				.batchSize(4)
				.progressCallback(progress ->
					System.out.printf("Batch Progress: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build()) {

			// Create batch of image-text pairs
			List<VisionLanguageModelLibrary.ImageTextPair> pairs = Arrays.asList(
				new VisionLanguageModelLibrary.ImageTextPair(
					Paths.get("examples/images/nature1.jpg"),
					"Describe the natural scenery in this image.",
					VisionLanguageModelLibrary.InferenceType.CAPTION),
				new VisionLanguageModelLibrary.ImageTextPair(
					Paths.get("examples/images/city1.jpg"),
					"What urban elements can you identify?",
					VisionLanguageModelLibrary.InferenceType.VQA),
				new VisionLanguageModelLibrary.ImageTextPair(
					Paths.get("examples/images/food1.jpg"),
					"What type of cuisine is this?",
					VisionLanguageModelLibrary.InferenceType.ANALYSIS),
				new VisionLanguageModelLibrary.ImageTextPair(
					Paths.get("examples/images/art1.jpg"),
					"Analyze the artistic style and composition.",
					VisionLanguageModelLibrary.InferenceType.ANALYSIS)
			);

			VisionLanguageModelLibrary.BatchInferenceResult batchResult = vlm.processBatch(pairs);

			if (batchResult.isSuccess()) {
				System.out.printf("Batch completed successfully: %d/%d images processed%n",
					batchResult.getSuccessfulPairs(), batchResult.getTotalPairs());
				System.out.printf("Success rate: %.1f%%%n", batchResult.getSuccessRate() * 100);
				System.out.println("Total duration: " + batchResult.getDuration().toSeconds() + "s");

				// Display individual results
				for (int i = 0; i < batchResult.getResults().size(); i++) {
					VisionLanguageModelLibrary.InferenceResult result = batchResult.getResults().get(i);
					System.out.printf("\nResult %d: %s%n", i + 1,
						result.isSuccess() ? result.getResponse() : "Failed - " + result.getMessage());
				}
			} else {
				System.err.println("Batch processing failed: " + batchResult.getMessage());
				System.err.println("Failed pairs: " + batchResult.getFailedPairs());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Batch processing failed", e);
		}
	}

	/**
	 * Asynchronous processing example
	 */
	public void asyncProcessingExample(String modelPath) {
		System.out.println("\n=== Async Processing ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.maxTokens(150)
				.build()) {

			Path imagePath = Paths.get("examples/images/async_test.jpg");

			// Start async captioning
			CompletableFuture<VisionLanguageModelLibrary.InferenceResult> captionFuture =
				vlm.captionImageAsync(imagePath);

			// Start async question answering
			CompletableFuture<VisionLanguageModelLibrary.InferenceResult> qaFuture =
				vlm.answerQuestionAsync(imagePath, "What is the main subject of this image?");

			// Wait for both to complete and combine results
			CompletableFuture<Void> combinedFuture = CompletableFuture.allOf(captionFuture, qaFuture)
				.thenRun(() -> {
					try {
						VisionLanguageModelLibrary.InferenceResult captionResult = captionFuture.get();
						VisionLanguageModelLibrary.InferenceResult qaResult = qaFuture.get();

						System.out.println("Async Caption: " +
							(captionResult.isSuccess() ? captionResult.getResponse() : "Failed"));
						System.out.println("Async Q&A: " +
							(qaResult.isSuccess() ? qaResult.getResponse() : "Failed"));

					} catch (Exception e) {
						LOGGER.log(System.Logger.Level.ERROR, "Async processing error", e);
					}
				});

			// Wait for completion
			combinedFuture.join();
			System.out.println("Async processing completed!");

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Async example failed", e);
		}
	}

	/**
	 * Model validation example
	 */
	public void modelValidationExample(String modelPath) {
		System.out.println("\n=== Model Validation ===");

		try (VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
				.languageModel(modelPath)
				.contextSize(2048)
				.gpuLayers(35)
				.build()) {

			VisionLanguageModelLibrary.ValidationResult validation = vlm.validateModel();

			System.out.println("Model Validation Results:");
			System.out.println("Valid: " + validation.isValid());
			System.out.println("Message: " + validation.getMessage());
			System.out.println("Supports text generation: " + validation.isSupportsTextGeneration());
			System.out.println("Supports vision: " + validation.isSupportsVision());
			System.out.println("Context size: " + validation.getContextSize());

			if (!validation.isValid()) {
				System.err.println("Model validation failed - check model path and format");
				if (validation.getError().isPresent()) {
					validation.getError().get().printStackTrace();
				}
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Model validation failed", e);
		}
	}
}