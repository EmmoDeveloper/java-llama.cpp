package de.kherud.llama.examples;

import de.kherud.llama.multimodal.ImageProcessorLibrary;
import de.kherud.llama.multimodal.VisionLanguageModelLibrary;
import org.junit.Test;
import org.junit.Before;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Simple unit tests for VisionLanguageModelLibrary and ImageProcessorLibrary using JUnit 4.
 * Tests basic functionality and builder patterns.
 */
public class VisionLanguageLibrarySimpleTest {

	private Path testImagePath;
	private Path testOutputDir;

	@Before
	public void setUp() throws IOException {
		// Create test directories
		testOutputDir = Files.createTempDirectory("vlm_test");
		Path testImagesDir = testOutputDir.resolve("images");
		Files.createDirectories(testImagesDir);

		// Create a test image
		testImagePath = createTestImage(testImagesDir.resolve("test.jpg"));
	}

	@Test
	public void testImageProcessorLibraryBuilder() {
		ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
			.targetSize(224, 224)
			.maintainAspectRatio(true)
			.centerCrop(true)
			.interpolation(ImageProcessorLibrary.InterpolationMethod.BICUBIC)
			.batchSize(16)
			.build();

		assertNotNull("Processor should be created", processor);
		processor.close();
	}

	@Test
	public void testVisionLanguageModelLibraryBuilder() {
		VisionLanguageModelLibrary vlm = VisionLanguageModelLibrary.builder()
			.languageModel("test-model.gguf")
			.contextSize(2048)
			.maxTokens(100)
			.temperature(0.7f)
			.topP(0.9f)
			.topK(40)
			.build();

		assertNotNull("VLM should be created", vlm);
		vlm.close();
	}

	@Test
	public void testImageProcessing() throws IOException {
		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(256, 256)
				.build()) {

			ImageProcessorLibrary.ProcessingResult result = processor.processImage(testImagePath);

			assertTrue("Image processing should succeed", result.isSuccess());
			assertTrue("Processed image should be present", result.getProcessedImage().isPresent());
			assertTrue("Metadata should be present", result.getMetadata().isPresent());

			ImageProcessorLibrary.ProcessedImage processed = result.getProcessedImage().get();
			assertEquals("Should have 3 channels (RGB)", 3, processed.channels);
			assertEquals("Width should be 256", 256, processed.processedWidth);
			assertEquals("Height should be 256", 256, processed.processedHeight);

			float[] pixels = processed.flatten();
			assertEquals("Pixel array should have correct size", 256 * 256 * 3, pixels.length);
		}
	}

	@Test
	public void testImageValidation() throws IOException {
		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder().build()) {

			// Test valid image
			ImageProcessorLibrary.ValidationResult result = processor.validateImage(testImagePath);
			assertTrue("Test image should be valid", result.isValid());
			assertEquals("Format should be JPG", "jpg", result.getFormat().toLowerCase());

			// Test non-existent image
			Path nonExistent = testOutputDir.resolve("nonexistent.jpg");
			ImageProcessorLibrary.ValidationResult invalidResult = processor.validateImage(nonExistent);
			assertFalse("Non-existent image should be invalid", invalidResult.isValid());
		}
	}

	@Test
	public void testBatchImageProcessing() throws IOException {
		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(128, 128)
				.batchSize(4)
				.build()) {

			// Create multiple test images
			List<Path> imagePaths = Arrays.asList(
				createTestImage(testOutputDir.resolve("batch1.jpg")),
				createTestImage(testOutputDir.resolve("batch2.jpg")),
				createTestImage(testOutputDir.resolve("batch3.jpg"))
			);

			ImageProcessorLibrary.BatchProcessingResult result = processor.processBatch(imagePaths);

			assertTrue("Batch processing should succeed", result.isSuccess());
			assertEquals("Should process 3 images", 3, result.getTotalImages());
			assertEquals("All images should succeed", 3, result.getSuccessfulImages());
			assertEquals("No images should fail", 0, result.getFailedImages());
			assertEquals("Success rate should be 100%", 1.0, result.getSuccessRate(), 0.001);
		}
	}

	@Test
	public void testFeatureExtraction() throws IOException {
		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(64, 64)
				.build()) {

			ImageProcessorLibrary.FeatureExtractionResult result = processor.extractFeatures(testImagePath);

			assertTrue("Feature extraction should succeed", result.isSuccess());
			assertEquals("Should have correct feature count", 64 * 64 * 3, result.getFeatureCount());

			float[] features = result.getFeatures();
			assertNotNull("Features should not be null", features);
			assertEquals("Feature array size should match count", result.getFeatureCount(), features.length);
		}
	}

	@Test
	public void testNormalizationMethods() throws IOException {
		// Test ImageNet normalization
		try (ImageProcessorLibrary processor1 = ImageProcessorLibrary.builder()
				.targetSize(32, 32)
				.meanNormalization(new float[]{0.485f, 0.456f, 0.406f})
				.stdNormalization(new float[]{0.229f, 0.224f, 0.225f})
				.normalizeToRange(false)
				.build()) {

			ImageProcessorLibrary.ProcessingResult result1 = processor1.processImage(testImagePath);
			assertTrue("ImageNet processing should succeed", result1.isSuccess());

			float[] pixels1 = result1.getProcessedImage().get().flatten();
			// ImageNet normalization typically results in values around [-2, 2]
			assertTrue("ImageNet normalized pixels should have negative values", getMin(pixels1) < 0);
			assertTrue("ImageNet normalized pixels should have positive values", getMax(pixels1) > 0);
		}

		// Test range normalization [-1, 1]
		try (ImageProcessorLibrary processor2 = ImageProcessorLibrary.builder()
				.targetSize(32, 32)
				.normalizeToRange(true)
				.build()) {

			ImageProcessorLibrary.ProcessingResult result2 = processor2.processImage(testImagePath);
			assertTrue("Range processing should succeed", result2.isSuccess());

			float[] pixels2 = result2.getProcessedImage().get().flatten();
			assertTrue("Range normalized pixels should be >= -1", getMin(pixels2) >= -1.0f);
			assertTrue("Range normalized pixels should be <= 1", getMax(pixels2) <= 1.0f);
		}
	}

	@Test
	public void testInterpolationMethods() throws IOException {
		ImageProcessorLibrary.InterpolationMethod[] methods = {
			ImageProcessorLibrary.InterpolationMethod.NEAREST,
			ImageProcessorLibrary.InterpolationMethod.BILINEAR,
			ImageProcessorLibrary.InterpolationMethod.BICUBIC
		};

		for (ImageProcessorLibrary.InterpolationMethod method : methods) {
			try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
					.targetSize(128, 128)
					.interpolation(method)
					.build()) {

				ImageProcessorLibrary.ProcessingResult result = processor.processImage(testImagePath);
				assertTrue("Processing with " + method + " should succeed", result.isSuccess());

				ImageProcessorLibrary.ProcessedImage processed = result.getProcessedImage().get();
				assertEquals("Width should be 128 for " + method, 128, processed.processedWidth);
				assertEquals("Height should be 128 for " + method, 128, processed.processedHeight);
			}
		}
	}

	@Test
	public void testPixelFormatConversion() throws IOException {
		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
				.targetSize(4, 4) // Small size for easy testing
				.build()) {

			ImageProcessorLibrary.ProcessingResult result = processor.processImage(testImagePath);
			assertTrue("Processing should succeed", result.isSuccess());

			ImageProcessorLibrary.ProcessedImage processed = result.getProcessedImage().get();

			float[] chw = processed.flattenCHW();
			float[] hwc = processed.flattenHWC();

			assertEquals("CHW and HWC should have same length", chw.length, hwc.length);
			assertEquals("Should have correct total size", 4 * 4 * 3, chw.length);

			// The arrays should contain the same values but in different order
			assertNotEquals("CHW and HWC should have different ordering",
				Arrays.toString(chw), Arrays.toString(hwc));
		}
	}

	@Test
	public void testIntegratedPipeline() throws IOException {
		// Test the integration between ImageProcessorLibrary components
		try (ImageProcessorLibrary imageProcessor = ImageProcessorLibrary.builder()
				.targetSize(224, 224)
				.build()) {

			// First validate and process the image
			ImageProcessorLibrary.ValidationResult validation = imageProcessor.validateImage(testImagePath);
			assertTrue("Image should be valid for pipeline", validation.isValid());

			ImageProcessorLibrary.ProcessingResult processing = imageProcessor.processImage(testImagePath);
			assertTrue("Image processing should succeed in pipeline", processing.isSuccess());

			ImageProcessorLibrary.FeatureExtractionResult features = imageProcessor.extractFeatures(testImagePath);
			assertTrue("Feature extraction should succeed in pipeline", features.isSuccess());

			// Verify the pipeline produces consistent results
			assertNotNull("Processed image should exist", processing.getProcessedImage().get());
			assertNotNull("Features should exist", features.getFeatures());
			assertEquals("Feature count should match processed size", 224 * 224 * 3, features.getFeatureCount());
		}
	}

	@Test
	public void testErrorHandling() throws IOException {
		try (ImageProcessorLibrary processor = ImageProcessorLibrary.builder().build()) {

			// Test with non-existent file
			Path nonExistent = testOutputDir.resolve("does_not_exist.jpg");
			ImageProcessorLibrary.ProcessingResult result = processor.processImage(nonExistent);
			assertFalse("Processing non-existent file should fail", result.isSuccess());
			assertTrue("Error should be present", result.getError().isPresent());

			// Test validation with non-existent file
			ImageProcessorLibrary.ValidationResult validation = processor.validateImage(nonExistent);
			assertFalse("Validation of non-existent file should fail", validation.isValid());
		}
	}

	@Test
	public void testResourceManagement() {
		// Test that resources are properly cleaned up
		ImageProcessorLibrary processor = ImageProcessorLibrary.builder()
			.targetSize(32, 32)
			.build();

		// Use the processor
		try {
			processor.processImage(testImagePath);
		} catch (Exception e) {
			// Ignore for this test
		}

		// Close should not throw
		try {
			processor.close();
		} catch (Exception e) {
			fail("Close should not throw exception: " + e.getMessage());
		}

		// Closing again should not throw
		try {
			processor.close();
		} catch (Exception e) {
			fail("Second close should not throw exception: " + e.getMessage());
		}
	}

	// Helper methods
	private Path createTestImage(Path outputPath) throws IOException {
		BufferedImage image = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = image.createGraphics();

		// Create a simple test pattern
		g2d.setColor(Color.RED);
		g2d.fillRect(0, 0, 50, 50);
		g2d.setColor(Color.GREEN);
		g2d.fillRect(50, 0, 50, 50);
		g2d.setColor(Color.BLUE);
		g2d.fillRect(0, 50, 50, 50);
		g2d.setColor(Color.YELLOW);
		g2d.fillRect(50, 50, 50, 50);

		g2d.dispose();

		// Ensure parent directory exists
		Files.createDirectories(outputPath.getParent());

		// Save image
		ImageIO.write(image, "jpg", outputPath.toFile());
		return outputPath;
	}

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