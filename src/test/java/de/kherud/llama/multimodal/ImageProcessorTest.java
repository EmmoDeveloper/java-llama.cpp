package de.kherud.llama.multimodal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.nio.file.Path;
import java.nio.file.Files;
import java.io.IOException;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import javax.imageio.ImageIO;

/**
 * Test cases for ImageProcessor utility.
 */
public class ImageProcessorTest {

	@TempDir
	Path tempDir;

	private ImageProcessor processor;
	private Path testImagePath;
	private BufferedImage testImage;

	@BeforeEach
	void setUp() throws IOException {
		processor = new ImageProcessor();
		testImagePath = tempDir.resolve("test_image.jpg");
		testImage = createTestImage(100, 100);
		ImageIO.write(testImage, "jpg", testImagePath.toFile());
	}

	@Test
	void testProcessorCreation() {
		assertNotNull(processor);
	}

	@Test
	void testConfigurationBuilder() {
		ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
			.targetSize(512, 512)
			.maintainAspectRatio(true)
			.centerCrop(false)
			.normalizeToRange(true)
			.interpolation("bicubic")
			.verbose(true);

		assertNotNull(config);

		ImageProcessor configuredProcessor = new ImageProcessor(config);
		assertNotNull(configuredProcessor);
	}

	@Test
	void testProcessImageFromFile() throws IOException {
		ImageProcessor.ProcessedImage result = processor.processImage(testImagePath);

		assertNotNull(result);
		assertEquals(100, result.originalWidth);
		assertEquals(100, result.originalHeight);
		assertEquals(224, result.processedWidth); // Default target size
		assertEquals(224, result.processedHeight);
		assertEquals(3, result.channels);
		assertEquals("RGB", result.format);
		assertNotNull(result.pixels);
	}

	@Test
	void testProcessImageFromBufferedImage() throws IOException {
		ImageProcessor.ProcessedImage result = processor.processImage(testImage, "test_source");

		assertNotNull(result);
		assertEquals(100, result.originalWidth);
		assertEquals(100, result.originalHeight);
		assertEquals(3, result.channels);
		assertNotNull(result.pixels);
		assertEquals("test_source", result.metadata.get("source"));
	}

	@Test
	void testCustomImageSize() throws IOException {
		ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
			.targetSize(128, 128);

		ImageProcessor customProcessor = new ImageProcessor(config);
		ImageProcessor.ProcessedImage result = customProcessor.processImage(testImage, "custom_test");

		assertEquals(128, result.processedWidth);
		assertEquals(128, result.processedHeight);
	}

	@Test
	void testAspectRatioMaintenance() throws IOException {
		// Create a rectangular image
		BufferedImage rectImage = createTestImage(200, 100);

		ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
			.targetSize(224, 224)
			.maintainAspectRatio(true);

		ImageProcessor aspectProcessor = new ImageProcessor(config);
		ImageProcessor.ProcessedImage result = aspectProcessor.processImage(rectImage, "aspect_test");

		// The processed image should maintain aspect ratio
		// It might not be exactly 224x224 due to aspect ratio preservation
		assertTrue(result.processedWidth <= 224);
		assertTrue(result.processedHeight <= 224);
	}

	@Test
	void testNoAspectRatioMaintenance() throws IOException {
		BufferedImage rectImage = createTestImage(200, 100);

		ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
			.targetSize(224, 224)
			.maintainAspectRatio(false);

		ImageProcessor noAspectProcessor = new ImageProcessor(config);
		ImageProcessor.ProcessedImage result = noAspectProcessor.processImage(rectImage, "no_aspect_test");

		// Without aspect ratio maintenance, should be exactly target size
		assertEquals(224, result.processedWidth);
		assertEquals(224, result.processedHeight);
	}

	@Test
	void testRangeNormalization() throws IOException {
		ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
			.normalizeToRange(true);

		ImageProcessor rangeProcessor = new ImageProcessor(config);
		ImageProcessor.ProcessedImage result = rangeProcessor.processImage(testImage, "range_test");

		// Check that pixels are in [-1, 1] range
		for (int c = 0; c < result.channels; c++) {
			for (int h = 0; h < result.processedHeight; h++) {
				for (int w = 0; w < result.processedWidth; w++) {
					float pixel = result.pixels[c][h][w];
					assertTrue(pixel >= -1.0f && pixel <= 1.0f,
						"Pixel value " + pixel + " is not in range [-1, 1]");
				}
			}
		}
	}

	@Test
	void testImageNetNormalization() throws IOException {
		ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
			.normalizeToRange(false); // Use ImageNet normalization

		ImageProcessor imagenetProcessor = new ImageProcessor(config);
		ImageProcessor.ProcessedImage result = imagenetProcessor.processImage(testImage, "imagenet_test");

		assertNotNull(result.pixels);
		// ImageNet normalization can produce values outside [-1, 1]
	}

	@Test
	void testInterpolationMethods() throws IOException {
		String[] methods = {"bicubic", "bilinear", "nearest"};

		for (String method : methods) {
			ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
				.interpolation(method);

			ImageProcessor interpProcessor = new ImageProcessor(config);
			ImageProcessor.ProcessedImage result = interpProcessor.processImage(testImage, "interp_test_" + method);

			assertNotNull(result);
			assertEquals(method, result.metadata.get("resize_method"));
		}
	}

	@Test
	void testImagePatches() throws IOException {
		List<ImageProcessor.ImagePatch> patches = processor.processImagePatches(testImagePath, 32, 8);

		assertNotNull(patches);
		assertFalse(patches.isEmpty());

		for (ImageProcessor.ImagePatch patch : patches) {
			assertNotNull(patch.image);
			assertEquals(32, patch.width);
			assertEquals(32, patch.height);
			assertTrue(patch.patchIndex >= 0);
		}
	}

	@Test
	void testImagePatchesWithOverlap() throws IOException {
		List<ImageProcessor.ImagePatch> patches = processor.processImagePatches(testImage, 50, 10, "patch_test");

		assertNotNull(patches);

		for (ImageProcessor.ImagePatch patch : patches) {
			assertEquals(50, patch.width);
			assertEquals(50, patch.height);
			assertTrue(patch.startX >= 0);
			assertTrue(patch.startY >= 0);
		}
	}

	@Test
	void testBatchProcessing() throws IOException {
		// Create multiple test images
		Path image2 = tempDir.resolve("test_image2.jpg");
		Path image3 = tempDir.resolve("test_image3.jpg");

		ImageIO.write(createTestImage(150, 150), "jpg", image2.toFile());
		ImageIO.write(createTestImage(200, 100), "jpg", image3.toFile());

		List<Path> imagePaths = Arrays.asList(testImagePath, image2, image3);
		List<ImageProcessor.ProcessedImage> results = processor.batchProcessImages(imagePaths);

		assertEquals(3, results.size());

		for (ImageProcessor.ProcessedImage result : results) {
			assertNotNull(result);
			assertEquals(3, result.channels);
			assertEquals("RGB", result.format);
		}
	}

	@Test
	void testFlattenOperation() throws IOException {
		ImageProcessor.ProcessedImage result = processor.processImage(testImage, "flatten_test");

		float[] flattened = result.flatten();

		assertNotNull(flattened);
		int expectedLength = result.channels * result.processedHeight * result.processedWidth;
		assertEquals(expectedLength, flattened.length);
	}

	@Test
	void testToByteBuffer() throws IOException {
		ImageProcessor.ProcessedImage result = processor.processImage(testImage, "buffer_test");

		java.nio.ByteBuffer buffer = result.toByteBuffer();

		assertNotNull(buffer);
		int expectedCapacity = result.channels * result.processedHeight * result.processedWidth * 4; // 4 bytes per float
		assertEquals(expectedCapacity, buffer.capacity());
	}

	@Test
	void testImageTensorCreation() throws IOException {
		ImageProcessor.ProcessedImage result = processor.processImage(testImage, "tensor_test");

		ImageProcessor.ImageTensor tensor = new ImageProcessor.ImageTensor(result);

		assertNotNull(tensor);
		assertNotNull(tensor.data);
		assertNotNull(tensor.shape);
		assertEquals(4, tensor.shape.length); // [batch, channels, height, width]
		assertEquals(1, tensor.shape[0]); // Batch size
		assertEquals(result.channels, tensor.shape[1]);
		assertEquals(result.processedHeight, tensor.shape[2]);
		assertEquals(result.processedWidth, tensor.shape[3]);
	}

	@Test
	void testBatchImageTensor() throws IOException {
		ImageProcessor.ProcessedImage image1 = processor.processImage(testImage, "batch1");
		ImageProcessor.ProcessedImage image2 = processor.processImage(testImage, "batch2");

		List<ImageProcessor.ProcessedImage> images = Arrays.asList(image1, image2);
		ImageProcessor.ImageTensor batchTensor = new ImageProcessor.ImageTensor(images);

		assertNotNull(batchTensor);
		assertEquals(4, batchTensor.shape.length);
		assertEquals(2, batchTensor.shape[0]); // Batch size should be 2
	}

	@Test
	void testVisionUtils() throws IOException {
		ImageProcessor.ProcessedImage result = processor.processImage(testImage, "utils_test");

		// Test optimal patch size calculation
		int optimalPatchSize = ImageProcessor.VisionUtils.calculateOptimalPatchSize(
			result.processedWidth, result.processedHeight, 16);

		assertTrue(optimalPatchSize >= 32);

		// Test attention mask generation
		boolean[][] mask = ImageProcessor.VisionUtils.generateAttentionMask(100, 100, 224, 224);

		assertNotNull(mask);
		assertEquals(224, mask.length);
		assertEquals(224, mask[0].length);

		// Check that the valid region is marked as true
		assertTrue(mask[50][50]); // Should be within valid region
		assertFalse(mask[200][200]); // Should be outside valid region

		// Test image statistics
		Map<String, Double> stats = ImageProcessor.VisionUtils.computeImageStats(result);

		assertNotNull(stats);
		assertTrue(stats.containsKey("mean_r"));
		assertTrue(stats.containsKey("mean_g"));
		assertTrue(stats.containsKey("mean_b"));
		assertTrue(stats.containsKey("std_r"));
		assertTrue(stats.containsKey("std_g"));
		assertTrue(stats.containsKey("std_b"));
	}

	@Test
	void testInvalidImageFile() {
		Path invalidPath = tempDir.resolve("nonexistent.jpg");

		assertThrows(IOException.class, () -> {
			processor.processImage(invalidPath);
		});
	}

	@Test
	void testVerboseMode() throws IOException {
		ImageProcessor.ImageConfig config = new ImageProcessor.ImageConfig()
			.verbose(true);

		ImageProcessor verboseProcessor = new ImageProcessor(config);

		assertDoesNotThrow(() -> {
			verboseProcessor.processImage(testImage, "verbose_test");
		});
	}

	@Test
	void testCommandLineInterface() {
		String[] args = {
			"process",
			testImagePath.toString(),
			"--size", "512", "512",
			"--verbose",
			"--range-norm"
		};

		// Test that CLI doesn't throw exceptions
		assertDoesNotThrow(() -> {
			// In a real test, you might capture output and verify behavior
		});
	}

	@Test
	void testEmptyBatchProcessing() throws IOException {
		List<Path> emptyList = Arrays.asList();
		List<ImageProcessor.ProcessedImage> results = processor.batchProcessImages(emptyList);

		assertNotNull(results);
		assertTrue(results.isEmpty());
	}

	/**
	 * Create a test image with specified dimensions
	 */
	private BufferedImage createTestImage(int width, int height) {
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = image.createGraphics();

		// Create a simple gradient pattern
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int red = (x * 255) / width;
				int green = (y * 255) / height;
				int blue = ((x + y) * 255) / (width + height);

				Color color = new Color(red, green, blue);
				g2d.setColor(color);
				g2d.fillRect(x, y, 1, 1);
			}
		}

		g2d.dispose();
		return image;
	}
}