package de.kherud.llama.multimodal;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Image processing utilities for multimodal models.
 *
 * Equivalent to clip.cpp and vision processing utilities - handles image preprocessing,
 * encoding, and feature extraction for vision-language models like LLaVA.
 */
public class ImageProcessor {
	private static final System.Logger logger = System.getLogger(ImageProcessor.class.getName());

	public static class ImageConfig {
		private int targetWidth = 224;
		private int targetHeight = 224;
		private boolean maintainAspectRatio = true;
		private boolean centerCrop = true;
		private float[] meanNormalization = {0.485f, 0.456f, 0.406f}; // ImageNet means
		private float[] stdNormalization = {0.229f, 0.224f, 0.225f}; // ImageNet stds
		private boolean normalizeToRange = false; // If true, normalize to [-1, 1] instead
		private String interpolation = "bicubic"; // bicubic, bilinear, nearest
		private boolean verbose = false;

		public ImageConfig targetSize(int width, int height) {
			this.targetWidth = width;
			this.targetHeight = height;
			return this;
		}

		public ImageConfig maintainAspectRatio(boolean maintain) {
			this.maintainAspectRatio = maintain;
			return this;
		}

		public ImageConfig centerCrop(boolean crop) {
			this.centerCrop = crop;
			return this;
		}

		public ImageConfig meanNormalization(float[] means) {
			this.meanNormalization = means.clone();
			return this;
		}

		public ImageConfig stdNormalization(float[] stds) {
			this.stdNormalization = stds.clone();
			return this;
		}

		public ImageConfig normalizeToRange(boolean normalize) {
			this.normalizeToRange = normalize;
			return this;
		}

		public ImageConfig interpolation(String method) {
			this.interpolation = method;
			return this;
		}

		public ImageConfig verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}
	}

	public static class ProcessedImage {
		public float[][][] pixels; // [channels][height][width]
		public int originalWidth;
		public int originalHeight;
		public int processedWidth;
		public int processedHeight;
		public int channels;
		public String format;
		public Map<String, Object> metadata = new HashMap<>();

		public float[] flatten() {
			int totalElements = channels * processedHeight * processedWidth;
			float[] flattened = new float[totalElements];
			int index = 0;

			for (int c = 0; c < channels; c++) {
				for (int h = 0; h < processedHeight; h++) {
					for (int w = 0; w < processedWidth; w++) {
						flattened[index++] = pixels[c][h][w];
					}
				}
			}

			return flattened;
		}

		public ByteBuffer toByteBuffer() {
			float[] flattened = flatten();
			ByteBuffer buffer = ByteBuffer.allocate(flattened.length * 4);
			for (float value : flattened) {
				buffer.putFloat(value);
			}
			buffer.flip();
			return buffer;
		}
	}

	public static class ImagePatch {
		public ProcessedImage image;
		public int startX;
		public int startY;
		public int width;
		public int height;
		public int patchIndex;
	}

	private final ImageConfig config;

	public ImageProcessor() {
		this(new ImageConfig());
	}

	public ImageProcessor(ImageConfig config) {
		this.config = config;
	}

	/**
	 * Process an image from file path
	 */
	public ProcessedImage processImage(Path imagePath) throws IOException {
		if (config.verbose) {
			logger.log(System.Logger.Level.INFO, "Processing image: " + imagePath);
		}

		BufferedImage image = ImageIO.read(imagePath.toFile());
		if (image == null) {
			throw new IOException("Failed to load image: " + imagePath);
		}

		return processImage(image, imagePath.toString());
	}

	/**
	 * Process an image from BufferedImage
	 */
	public ProcessedImage processImage(BufferedImage image, String sourceName) throws IOException {
		ProcessedImage processed = new ProcessedImage();
		processed.originalWidth = image.getWidth();
		processed.originalHeight = image.getHeight();
		processed.format = "RGB";
		processed.channels = 3;

		if (config.verbose) {
			logger.log(System.Logger.Level.INFO, String.format("Original image size: %dx%d",
				processed.originalWidth, processed.originalHeight));
		}

		// Convert to RGB if necessary
		BufferedImage rgbImage = convertToRGB(image);

		// Resize image
		BufferedImage resizedImage = resizeImage(rgbImage);
		processed.processedWidth = resizedImage.getWidth();
		processed.processedHeight = resizedImage.getHeight();

		if (config.verbose) {
			logger.log(System.Logger.Level.INFO, String.format("Processed image size: %dx%d",
				processed.processedWidth, processed.processedHeight));
		}

		// Extract and normalize pixels
		processed.pixels = extractAndNormalizePixels(resizedImage);

		// Add metadata
		processed.metadata.put("source", sourceName);
		processed.metadata.put("resize_method", config.interpolation);
		processed.metadata.put("normalization_type", config.normalizeToRange ? "range" : "imagenet");

		return processed;
	}

	/**
	 * Process image into patches for patch-based models
	 */
	public List<ImagePatch> processImagePatches(Path imagePath, int patchSize, int overlap) throws IOException {
		BufferedImage image = ImageIO.read(imagePath.toFile());
		if (image == null) {
			throw new IOException("Failed to load image: " + imagePath);
		}

		return processImagePatches(image, patchSize, overlap, imagePath.toString());
	}

	public List<ImagePatch> processImagePatches(BufferedImage image, int patchSize, int overlap, String sourceName) throws IOException {
		List<ImagePatch> patches = new ArrayList<>();

		// Convert to RGB
		BufferedImage rgbImage = convertToRGB(image);

		int step = patchSize - overlap;
		int patchIndex = 0;

		for (int y = 0; y <= rgbImage.getHeight() - patchSize; y += step) {
			for (int x = 0; x <= rgbImage.getWidth() - patchSize; x += step) {
				// Extract patch
				BufferedImage patchImage = rgbImage.getSubimage(x, y, patchSize, patchSize);

				// Process patch
				ProcessedImage processedPatch = processImage(patchImage, sourceName + "_patch_" + patchIndex);

				// Create patch info
				ImagePatch patch = new ImagePatch();
				patch.image = processedPatch;
				patch.startX = x;
				patch.startY = y;
				patch.width = patchSize;
				patch.height = patchSize;
				patch.patchIndex = patchIndex++;

				patches.add(patch);
			}
		}

		if (config.verbose) {
			logger.log(System.Logger.Level.INFO, "Generated " + patches.size() + " patches of size " + patchSize + "x" + patchSize);
		}

		return patches;
	}

	/**
	 * Batch process multiple images
	 */
	public List<ProcessedImage> batchProcessImages(List<Path> imagePaths) throws IOException {
		List<ProcessedImage> processed = new ArrayList<>();

		for (Path imagePath : imagePaths) {
			try {
				ProcessedImage result = processImage(imagePath);
				processed.add(result);
			} catch (IOException e) {
				logger.log(System.Logger.Level.WARNING, "Failed to process image: " + imagePath + " - " + e.getMessage());
			}
		}

		return processed;
	}

	private BufferedImage convertToRGB(BufferedImage image) {
		if (image.getType() == BufferedImage.TYPE_INT_RGB) {
			return image;
		}

		BufferedImage rgbImage = new BufferedImage(
			image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = rgbImage.createGraphics();
		g2d.drawImage(image, 0, 0, null);
		g2d.dispose();

		return rgbImage;
	}

	private BufferedImage resizeImage(BufferedImage image) {
		int targetWidth = config.targetWidth;
		int targetHeight = config.targetHeight;

		if (config.maintainAspectRatio) {
			double aspectRatio = (double) image.getWidth() / image.getHeight();
			double targetAspectRatio = (double) targetWidth / targetHeight;

			if (aspectRatio > targetAspectRatio) {
				// Image is wider than target
				targetHeight = (int) (targetWidth / aspectRatio);
			} else {
				// Image is taller than target
				targetWidth = (int) (targetHeight * aspectRatio);
			}
		}

		// Choose scaling algorithm
		Object interpolationHint = getInterpolationHint();

		BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = resized.createGraphics();
		g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolationHint);
		g2d.drawImage(image, 0, 0, targetWidth, targetHeight, null);
		g2d.dispose();

		// Center crop if needed
		if (config.centerCrop && (targetWidth != config.targetWidth || targetHeight != config.targetHeight)) {
			resized = centerCropImage(resized, config.targetWidth, config.targetHeight);
		}

		return resized;
	}

	private Object getInterpolationHint() {
		switch (config.interpolation.toLowerCase()) {
			case "bicubic":
				return RenderingHints.VALUE_INTERPOLATION_BICUBIC;
			case "bilinear":
				return RenderingHints.VALUE_INTERPOLATION_BILINEAR;
			case "nearest":
				return RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR;
			default:
				return RenderingHints.VALUE_INTERPOLATION_BICUBIC;
		}
	}

	private BufferedImage centerCropImage(BufferedImage image, int targetWidth, int targetHeight) {
		int x = (image.getWidth() - targetWidth) / 2;
		int y = (image.getHeight() - targetHeight) / 2;

		if (x < 0) x = 0;
		if (y < 0) y = 0;

		int cropWidth = Math.min(targetWidth, image.getWidth());
		int cropHeight = Math.min(targetHeight, image.getHeight());

		return image.getSubimage(x, y, cropWidth, cropHeight);
	}

	private float[][][] extractAndNormalizePixels(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		float[][][] pixels = new float[3][height][width]; // RGB channels

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rgb = image.getRGB(x, y);

				// Extract RGB components
				int red = (rgb >> 16) & 0xFF;
				int green = (rgb >> 8) & 0xFF;
				int blue = rgb & 0xFF;

				// Convert to [0, 1] range
				float r = red / 255.0f;
				float g = green / 255.0f;
				float b = blue / 255.0f;

				// Apply normalization
				if (config.normalizeToRange) {
					// Normalize to [-1, 1]
					pixels[0][y][x] = 2.0f * r - 1.0f; // Red
					pixels[1][y][x] = 2.0f * g - 1.0f; // Green
					pixels[2][y][x] = 2.0f * b - 1.0f; // Blue
				} else {
					// Apply ImageNet normalization
					pixels[0][y][x] = (r - config.meanNormalization[0]) / config.stdNormalization[0]; // Red
					pixels[1][y][x] = (g - config.meanNormalization[1]) / config.stdNormalization[1]; // Green
					pixels[2][y][x] = (b - config.meanNormalization[2]) / config.stdNormalization[2]; // Blue
				}
			}
		}

		return pixels;
	}

	/**
	 * Create image tensors for model input
	 */
	public static class ImageTensor {
		public float[] data;
		public long[] shape; // [batch, channels, height, width]
		public String dtype = "float32";

		public ImageTensor(ProcessedImage image) {
			this.data = image.flatten();
			this.shape = new long[]{1, image.channels, image.processedHeight, image.processedWidth};
		}

		public ImageTensor(List<ProcessedImage> images) {
			if (images.isEmpty()) {
				throw new IllegalArgumentException("Image list cannot be empty");
			}

			ProcessedImage first = images.get(0);
			int batchSize = images.size();
			int channels = first.channels;
			int height = first.processedHeight;
			int width = first.processedWidth;

			this.shape = new long[]{batchSize, channels, height, width};
			this.data = new float[batchSize * channels * height * width];

			int batchOffset = 0;
			for (ProcessedImage image : images) {
				float[] imageData = image.flatten();
				System.arraycopy(imageData, 0, this.data, batchOffset, imageData.length);
				batchOffset += imageData.length;
			}
		}
	}

	/**
	 * Utility methods for common vision tasks
	 */
	public static class VisionUtils {

		/**
		 * Calculate optimal patch size for an image
		 */
		public static int calculateOptimalPatchSize(int imageWidth, int imageHeight, int maxPatches) {
			int minPatchSize = 32;
			int maxPatchSize = Math.min(imageWidth, imageHeight);

			for (int patchSize = minPatchSize; patchSize <= maxPatchSize; patchSize += 16) {
				int patchesX = imageWidth / patchSize;
				int patchesY = imageHeight / patchSize;
				int totalPatches = patchesX * patchesY;

				if (totalPatches <= maxPatches) {
					return patchSize;
				}
			}

			return maxPatchSize;
		}

		/**
		 * Generate attention masks for variable-sized images
		 */
		public static boolean[][] generateAttentionMask(int width, int height, int maxWidth, int maxHeight) {
			boolean[][] mask = new boolean[maxHeight][maxWidth];

			for (int y = 0; y < maxHeight; y++) {
				for (int x = 0; x < maxWidth; x++) {
					mask[y][x] = (x < width && y < height);
				}
			}

			return mask;
		}

		/**
		 * Compute image statistics
		 */
		public static Map<String, Double> computeImageStats(ProcessedImage image) {
			Map<String, Double> stats = new HashMap<>();

			double[] channelMeans = new double[image.channels];
			double[] channelStds = new double[image.channels];

			// Calculate means
			for (int c = 0; c < image.channels; c++) {
				double sum = 0;
				int count = 0;

				for (int h = 0; h < image.processedHeight; h++) {
					for (int w = 0; w < image.processedWidth; w++) {
						sum += image.pixels[c][h][w];
						count++;
					}
				}

				channelMeans[c] = sum / count;
			}

			// Calculate standard deviations
			for (int c = 0; c < image.channels; c++) {
				double sumSquares = 0;
				int count = 0;

				for (int h = 0; h < image.processedHeight; h++) {
					for (int w = 0; w < image.processedWidth; w++) {
						double diff = image.pixels[c][h][w] - channelMeans[c];
						sumSquares += diff * diff;
						count++;
					}
				}

				channelStds[c] = Math.sqrt(sumSquares / count);
			}

			stats.put("mean_r", channelMeans[0]);
			stats.put("mean_g", channelMeans[1]);
			stats.put("mean_b", channelMeans[2]);
			stats.put("std_r", channelStds[0]);
			stats.put("std_g", channelStds[1]);
			stats.put("std_b", channelStds[2]);

			return stats;
		}
	}

	/**
	 * Command-line interface for testing
	 */
	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(ImageProcessor::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length < 1) {
			printUsage();
			throw new IllegalArgumentException("No command specified");
		}

		try {
			String command = args[0];
			ImageConfig config = new ImageConfig();

			// Parse options
			int argIndex = 1;
			while (argIndex < args.length && args[argIndex].startsWith("--")) {
				switch (args[argIndex]) {
					case "--size":
						if (argIndex + 2 < args.length) {
							int width = Integer.parseInt(args[++argIndex]);
							int height = Integer.parseInt(args[++argIndex]);
							config.targetSize(width, height);
						}
						break;
					case "--no-aspect":
						config.maintainAspectRatio(false);
						break;
					case "--no-crop":
						config.centerCrop(false);
						break;
					case "--range-norm":
						config.normalizeToRange(true);
						break;
					case "--interpolation":
						if (argIndex + 1 < args.length) {
							config.interpolation(args[++argIndex]);
						}
						break;
					case "--verbose":
					case "-v":
						config.verbose(true);
						break;
					case "--help":
					case "-h":
						printUsage();
						return; // Exit normally after showing help
				}
				argIndex++;
			}

			ImageProcessor processor = new ImageProcessor(config);

			switch (command) {
				case "process":
					handleProcessCommand(args, argIndex, processor);
					break;
				case "patches":
					handlePatchesCommand(args, argIndex, processor);
					break;
				case "stats":
					handleStatsCommand(args, argIndex, processor);
					break;
				default:
					printUsage();
					throw new IllegalArgumentException("Unknown command: " + command);
			}

		} catch (IOException e) {
			throw e; // Re-throw IO exceptions
		} catch (Exception e) {
			throw new RuntimeException("Command failed", e);
		}
	}

	private static void handleProcessCommand(String[] args, int startIndex, ImageProcessor processor) throws IOException {
		if (startIndex >= args.length) {
			throw new IllegalArgumentException("Image path required for process command");
		}

		Path imagePath = Path.of(args[startIndex]);
		ProcessedImage result = processor.processImage(imagePath);

		System.out.println("=== IMAGE PROCESSING RESULT ===");
		System.out.println("Original size: " + result.originalWidth + "x" + result.originalHeight);
		System.out.println("Processed size: " + result.processedWidth + "x" + result.processedHeight);
		System.out.println("Channels: " + result.channels);
		System.out.println("Format: " + result.format);
		System.out.println("Total elements: " + (result.processedWidth * result.processedHeight * result.channels));

		Map<String, Double> stats = VisionUtils.computeImageStats(result);
		System.out.println("Channel means: R=" + String.format("%.3f", stats.get("mean_r")) +
						  " G=" + String.format("%.3f", stats.get("mean_g")) +
						  " B=" + String.format("%.3f", stats.get("mean_b")));
	}

	private static void handlePatchesCommand(String[] args, int startIndex, ImageProcessor processor) throws IOException {
		if (startIndex + 1 >= args.length) {
			throw new IllegalArgumentException("Image path and patch size required for patches command");
		}

		Path imagePath = Path.of(args[startIndex]);
		int patchSize = Integer.parseInt(args[startIndex + 1]);
		int overlap = 0;

		if (startIndex + 2 < args.length) {
			overlap = Integer.parseInt(args[startIndex + 2]);
		}

		List<ImagePatch> patches = processor.processImagePatches(imagePath, patchSize, overlap);

		System.out.println("=== IMAGE PATCHES RESULT ===");
		System.out.println("Total patches: " + patches.size());
		System.out.println("Patch size: " + patchSize + "x" + patchSize);
		System.out.println("Overlap: " + overlap);

		for (int i = 0; i < Math.min(5, patches.size()); i++) {
			ImagePatch patch = patches.get(i);
			System.out.println("Patch " + i + ": position (" + patch.startX + "," + patch.startY + ")");
		}
	}

	private static void handleStatsCommand(String[] args, int startIndex, ImageProcessor processor) throws IOException {
		if (startIndex >= args.length) {
			throw new IllegalArgumentException("Image path required for stats command");
		}

		Path imagePath = Path.of(args[startIndex]);
		ProcessedImage result = processor.processImage(imagePath);
		Map<String, Double> stats = VisionUtils.computeImageStats(result);

		System.out.println("=== IMAGE STATISTICS ===");
		System.out.println("Image: " + imagePath.getFileName());
		System.out.println("Size: " + result.processedWidth + "x" + result.processedHeight);
		System.out.println();
		System.out.println("Channel Statistics:");
		System.out.printf("Red   - Mean: %8.4f, Std: %8.4f%n", stats.get("mean_r"), stats.get("std_r"));
		System.out.printf("Green - Mean: %8.4f, Std: %8.4f%n", stats.get("mean_g"), stats.get("std_g"));
		System.out.printf("Blue  - Mean: %8.4f, Std: %8.4f%n", stats.get("mean_b"), stats.get("std_b"));
	}

	private static void printUsage() {
		System.out.println("Usage: ImageProcessor <command> [options] [args]");
		System.out.println();
		System.out.println("Process images for multimodal vision-language models.");
		System.out.println();
		System.out.println("Commands:");
		System.out.println("  process <image>           Process a single image");
		System.out.println("  patches <image> <size>    Extract patches from image");
		System.out.println("  stats <image>             Show image statistics");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --size <w> <h>            Target image size (default: 224x224)");
		System.out.println("  --no-aspect               Don't maintain aspect ratio");
		System.out.println("  --no-crop                 Don't center crop");
		System.out.println("  --range-norm              Normalize to [-1,1] instead of ImageNet");
		System.out.println("  --interpolation <method>   Interpolation method (bicubic, bilinear, nearest)");
		System.out.println("  --verbose, -v             Verbose output");
		System.out.println("  --help, -h                Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  ImageProcessor process image.jpg");
		System.out.println("  ImageProcessor --size 512 512 process image.jpg");
		System.out.println("  ImageProcessor patches image.jpg 64 8");
		System.out.println("  ImageProcessor stats image.jpg");
	}
}
