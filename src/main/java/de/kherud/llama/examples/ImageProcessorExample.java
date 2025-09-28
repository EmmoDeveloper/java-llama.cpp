package de.kherud.llama.examples;

import de.kherud.llama.multimodal.ImageProcessorLibrary;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Comprehensive examples demonstrating ImageProcessorLibrary usage.
 * Shows various image preprocessing capabilities for multimodal AI models.
 */
public class ImageProcessorExample {
	private static final System.Logger LOGGER = System.getLogger(ImageProcessorExample.class.getName());

	public static void main(String[] args) {
		ImageProcessorExample example = new ImageProcessorExample();

		try {
			// Run all examples
			example.basicImageProcessing();
			example.configuredProcessing();
			example.batchProcessing();
			example.asyncProcessing();
			example.featureExtraction();
			example.imageValidation();
			example.resizeExample();
			example.normalizeExample();
			example.performanceTest();

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Example execution failed", e);
		}
	}

	/**
	 * Basic image processing with default settings
	 */
	public void basicImageProcessing() {
		System.out.println("\n=== Basic Image Processing ===");

		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.build()) {

			Path imagePath = Paths.get("examples/images/sample.jpg");

			ImageProcessorLibrary.ProcessingResult result = processor.processImage(imagePath);

			if (result.isSuccess()) {
				ImageProcessorLibrary.ProcessedImage processed = result.getProcessedImage().get();
				ImageProcessorLibrary.ImageMetadata metadata = result.getMetadata().get();

				System.out.printf("Successfully processed image: %s%n", imagePath.getFileName());
				System.out.printf("Original dimensions: %dx%d%n",
					metadata.getOriginalWidth(), metadata.getOriginalHeight());
				System.out.printf("Processed dimensions: %dx%d%n",
					metadata.getProcessedWidth(), metadata.getProcessedHeight());
				System.out.printf("Channels: %d%n", metadata.getChannels());
				System.out.printf("Format: %s%n", metadata.getFormat());
				System.out.printf("Aspect ratio: %.2f%n", metadata.getAspectRatio());
				System.out.printf("Processing time: %d ms%n", result.getDuration().toMillis());

				// Show pixel data info
				float[] flattened = processed.flatten();
				System.out.printf("Total pixels: %d (%.1f KB)%n",
					flattened.length, flattened.length * 4.0 / 1024);
			} else {
				System.err.println("Processing failed: " + result.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Basic processing failed", e);
		}
	}

	/**
	 * Configured processing with custom settings
	 */
	public void configuredProcessing() {
		System.out.println("\n=== Configured Processing ===");

		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(512, 512)
				.maintainAspectRatio(true)
				.centerCrop(true)
				.meanNormalization(new float[]{0.485f, 0.456f, 0.406f}) // ImageNet means
				.stdNormalization(new float[]{0.229f, 0.224f, 0.225f})  // ImageNet stds
				.interpolation(ImageProcessorLibrary.InterpolationMethod.BICUBIC)
				.progressCallback(progress ->
					System.out.printf("Progress: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build()) {

			Path imagePath = Paths.get("examples/images/high_res.jpg");

			ImageProcessorLibrary.ProcessingResult result = processor.processImage(imagePath);

			if (result.isSuccess()) {
				System.out.println("Configured processing completed successfully");

				// Demonstrate different output formats
				ImageProcessorLibrary.ProcessedImage processed = result.getProcessedImage().get();

				float[] chw = processed.flattenCHW();
				float[] hwc = processed.flattenHWC();

				System.out.printf("CHW format: %d elements%n", chw.length);
				System.out.printf("HWC format: %d elements%n", hwc.length);

				// Show pixel value ranges after normalization
				float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
				for (float value : chw) {
					min = Math.min(min, value);
					max = Math.max(max, value);
				}
				System.out.printf("Normalized pixel range: %.3f to %.3f%n", min, max);

			} else {
				System.err.println("Configured processing failed: " + result.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Configured processing failed", e);
		}
	}

	/**
	 * Batch processing multiple images
	 */
	public void batchProcessing() {
		System.out.println("\n=== Batch Processing ===");

		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(256, 256)
				.maintainAspectRatio(true)
				.batchSize(8)
				.progressCallback(progress ->
					System.out.printf("Batch Progress: %.1f%% - %s%n",
						progress.getProgress() * 100, progress.getMessage()))
				.build()) {

			// Create list of image paths
			List<Path> imagePaths = Arrays.asList(
				Paths.get("examples/images/img1.jpg"),
				Paths.get("examples/images/img2.jpg"),
				Paths.get("examples/images/img3.jpg"),
				Paths.get("examples/images/img4.jpg"),
				Paths.get("examples/images/img5.jpg"),
				Paths.get("examples/images/img6.jpg"),
				Paths.get("examples/images/img7.jpg"),
				Paths.get("examples/images/img8.jpg"),
				Paths.get("examples/images/img9.jpg"),
				Paths.get("examples/images/img10.jpg")
			);

			ImageProcessorLibrary.BatchProcessingResult batchResult = processor.processBatch(imagePaths);

			if (batchResult.isSuccess()) {
				System.out.printf("Batch processing completed successfully!%n");
				System.out.printf("Total images: %d%n", batchResult.getTotalImages());
				System.out.printf("Successful: %d%n", batchResult.getSuccessfulImages());
				System.out.printf("Failed: %d%n", batchResult.getFailedImages());
				System.out.printf("Success rate: %.1f%%%n", batchResult.getSuccessRate() * 100);
				System.out.printf("Total duration: %.2f seconds%n", batchResult.getDuration().toMillis() / 1000.0);

				if (batchResult.getFailedImages() > 0) {
					System.out.println("Failed images:");
					batchResult.getFailedPaths().forEach(path ->
						System.out.println("  - " + path.getFileName()));
				}

				// Show processing stats for successful images
				long totalPixels = 0;
				for (ImageProcessorLibrary.ProcessingResult result : batchResult.getResults()) {
					if (result.isSuccess()) {
						ImageProcessorLibrary.ProcessedImage processed = result.getProcessedImage().get();
						totalPixels += processed.flatten().length;
					}
				}
				System.out.printf("Total pixels processed: %,d (%.1f MB)%n",
					totalPixels, totalPixels * 4.0 / (1024 * 1024));

			} else {
				System.err.println("Batch processing failed: " + batchResult.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Batch processing failed", e);
		}
	}

	/**
	 * Asynchronous processing example
	 */
	public void asyncProcessing() {
		System.out.println("\n=== Async Processing ===");

		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.build()) {

			Path imagePath1 = Paths.get("examples/images/async1.jpg");
			Path imagePath2 = Paths.get("examples/images/async2.jpg");
			Path imagePath3 = Paths.get("examples/images/async3.jpg");

			// Start multiple async operations
			CompletableFuture<ImageProcessorLibrary.ProcessingResult> future1 =
				processor.processImageAsync(imagePath1);
			CompletableFuture<ImageProcessorLibrary.ProcessingResult> future2 =
				processor.processImageAsync(imagePath2);
			CompletableFuture<ImageProcessorLibrary.ProcessingResult> future3 =
				processor.processImageAsync(imagePath3);

			// Combine results
			CompletableFuture<Void> allFutures = CompletableFuture.allOf(future1, future2, future3)
				.thenRun(() -> {
					try {
						ImageProcessorLibrary.ProcessingResult result1 = future1.get();
						ImageProcessorLibrary.ProcessingResult result2 = future2.get();
						ImageProcessorLibrary.ProcessingResult result3 = future3.get();

						System.out.println("All async processing completed!");
						System.out.printf("Image 1: %s%n", result1.isSuccess() ? "Success" : "Failed");
						System.out.printf("Image 2: %s%n", result2.isSuccess() ? "Success" : "Failed");
						System.out.printf("Image 3: %s%n", result3.isSuccess() ? "Success" : "Failed");

					} catch (Exception e) {
						LOGGER.log(System.Logger.Level.ERROR, "Async processing error", e);
					}
				});

			// Wait for completion
			allFutures.join();

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Async processing failed", e);
		}
	}

	/**
	 * Feature extraction example
	 */
	public void featureExtraction() {
		System.out.println("\n=== Feature Extraction ===");

		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.meanNormalization(new float[]{0.485f, 0.456f, 0.406f})
				.stdNormalization(new float[]{0.229f, 0.224f, 0.225f})
				.build()) {

			Path imagePath = Paths.get("examples/images/features.jpg");

			ImageProcessorLibrary.FeatureExtractionResult result = processor.extractFeatures(imagePath);

			if (result.isSuccess()) {
				float[] features = result.getFeatures();

				System.out.printf("Features extracted successfully%n");
				System.out.printf("Feature count: %d%n", result.getFeatureCount());
				System.out.printf("Feature vector size: %.1f KB%n", features.length * 4.0 / 1024);

				// Calculate feature statistics
				double sum = 0, sumSquares = 0;
				float min = Float.MAX_VALUE, max = Float.MIN_VALUE;

				for (float feature : features) {
					sum += feature;
					sumSquares += feature * feature;
					min = Math.min(min, feature);
					max = Math.max(max, feature);
				}

				double mean = sum / features.length;
				double variance = (sumSquares / features.length) - (mean * mean);
				double stdDev = Math.sqrt(variance);

				System.out.printf("Feature statistics:%n");
				System.out.printf("  Mean: %.6f%n", mean);
				System.out.printf("  Std Dev: %.6f%n", stdDev);
				System.out.printf("  Min: %.6f%n", min);
				System.out.printf("  Max: %.6f%n", max);

				// Show sample features
				System.out.print("First 10 features: ");
				for (int i = 0; i < Math.min(10, features.length); i++) {
					System.out.printf("%.3f ", features[i]);
				}
				System.out.println();

			} else {
				System.err.println("Feature extraction failed: " + result.getMessage());
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Feature extraction failed", e);
		}
	}

	/**
	 * Image validation example
	 */
	public void imageValidation() {
		System.out.println("\n=== Image Validation ===");

		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder().build()) {

			// Test various image files
			String[] testFiles = {
				"examples/images/valid.jpg",
				"examples/images/valid.png",
				"examples/images/invalid.txt",
				"examples/images/corrupted.jpg",
				"examples/images/nonexistent.jpg"
			};

			for (String filePath : testFiles) {
				Path path = Paths.get(filePath);
				ImageProcessorLibrary.ValidationResult validation = processor.validateImage(path);

				System.out.printf("File: %-30s Status: %s%n",
					path.getFileName(),
					validation.isValid() ? "VALID" : "INVALID");

				if (validation.isValid()) {
					System.out.printf("  Dimensions: %dx%d, Format: %s%n",
						validation.getWidth(), validation.getHeight(), validation.getFormat());
				} else {
					System.out.printf("  Error: %s%n", validation.getMessage());
				}
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Image validation failed", e);
		}
	}

	/**
	 * Image resizing example
	 */
	public void resizeExample() {
		System.out.println("\n=== Image Resizing ===");

		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder().build()) {

			Path imagePath = Paths.get("examples/images/resize_test.jpg");

			// Test different resize operations
			int[][] sizes = {{128, 128}, {256, 256}, {512, 512}, {300, 200}, {800, 600}};

			for (int[] size : sizes) {
				ImageProcessorLibrary.ProcessingResult result =
					processor.resizeImage(imagePath, size[0], size[1]);

				if (result.isSuccess()) {
					ImageProcessorLibrary.ImageMetadata metadata = result.getMetadata().get();
					System.out.printf("Resized to %dx%d: Success (aspect ratio: %.2f)%n",
						size[0], size[1], metadata.getAspectRatio());
				} else {
					System.out.printf("Resized to %dx%d: Failed - %s%n",
						size[0], size[1], result.getMessage());
				}
			}

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Resize example failed", e);
		}
	}

	/**
	 * Different normalization methods example
	 */
	public void normalizeExample() {
		System.out.println("\n=== Normalization Methods ===");

		Path imagePath = Paths.get("examples/images/normalize_test.jpg");

		// Test ImageNet normalization
		try (ImageProcessorLibrary processor1 = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.meanNormalization(new float[]{0.485f, 0.456f, 0.406f})
				.stdNormalization(new float[]{0.229f, 0.224f, 0.225f})
				.normalizeToRange(false)
				.build()) {

			ImageProcessorLibrary.ProcessingResult result1 = processor1.processImage(imagePath);
			if (result1.isSuccess()) {
				float[] pixels1 = result1.getProcessedImage().get().flatten();
				System.out.printf("ImageNet normalization - Range: [%.3f, %.3f]%n",
					getMin(pixels1), getMax(pixels1));
			}
		}

		// Test range normalization [-1, 1]
		try (ImageProcessorLibrary processor2 = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.normalizeToRange(true)
				.build()) {

			ImageProcessorLibrary.ProcessingResult result2 = processor2.processImage(imagePath);
			if (result2.isSuccess()) {
				float[] pixels2 = result2.getProcessedImage().get().flatten();
				System.out.printf("Range normalization - Range: [%.3f, %.3f]%n",
					getMin(pixels2), getMax(pixels2));
			}
		}

		// Test no normalization (raw [0, 1])
		try (ImageProcessorLibrary processor3 = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.meanNormalization(new float[]{0.0f, 0.0f, 0.0f})
				.stdNormalization(new float[]{1.0f, 1.0f, 1.0f})
				.normalizeToRange(false)
				.build()) {

			ImageProcessorLibrary.ProcessingResult result3 = processor3.processImage(imagePath);
			if (result3.isSuccess()) {
				float[] pixels3 = result3.getProcessedImage().get().flatten();
				System.out.printf("No normalization - Range: [%.3f, %.3f]%n",
					getMin(pixels3), getMax(pixels3));
			}
		}
	}

	/**
	 * Performance testing example
	 */
	public void performanceTest() {
		System.out.println("\n=== Performance Test ===");

		// Test different interpolation methods
		ImageProcessorLibrary.InterpolationMethod[] methods = {
			ImageProcessorLibrary.InterpolationMethod.NEAREST,
			ImageProcessorLibrary.InterpolationMethod.BILINEAR,
			ImageProcessorLibrary.InterpolationMethod.BICUBIC
		};

		Path imagePath = Paths.get("examples/images/performance_test.jpg");

		for (ImageProcessorLibrary.InterpolationMethod method : methods) {
			try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
					.targetSize(512, 512)
					.interpolation(method)
					.build()) {

				long startTime = System.currentTimeMillis();

				// Process multiple times to get average
				int iterations = 10;
				int successes = 0;

				for (int i = 0; i < iterations; i++) {
					ImageProcessorLibrary.ProcessingResult result = processor.processImage(imagePath);
					if (result.isSuccess()) {
						successes++;
					}
				}

				long endTime = System.currentTimeMillis();
				double avgTime = (endTime - startTime) / (double) iterations;

				System.out.printf("%-10s: %.1f ms/image (%d/%d successful)%n",
					method, avgTime, successes, iterations);

			} catch (Exception e) {
				System.out.printf("%-10s: Failed - %s%n", method, e.getMessage());
			}
		}
	}

	// Helper methods
	private float getMin(float[] array) {
		float min = Float.MAX_VALUE;
		for (float value : array) {
			min = Math.min(min, value);
		}
		return min;
	}

	private float getMax(float[] array) {
		float max = Float.MIN_VALUE;
		for (float value : array) {
			max = Math.max(max, value);
		}
		return max;
	}
}
