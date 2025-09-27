package de.kherud.llama.multimodal;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Library-friendly image processing for multimodal models.
 *
 * This refactored version provides a fluent API for image preprocessing, encoding,
 * and feature extraction for vision-language models with builder pattern configuration,
 * batch processing, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic image processing
 * ProcessingResult result = ImageProcessorLibrary.builder()
 *     .targetSize(224, 224)
 *     .build()
 *     .processImage(imagePath);
 *
 * // Configured processing
 * ProcessingResult result = ImageProcessorLibrary.builder()
 *     .targetSize(512, 512)
 *     .maintainAspectRatio(true)
 *     .centerCrop(true)
 *     .meanNormalization(new float[]{0.485f, 0.456f, 0.406f})
 *     .interpolation(InterpolationMethod.BICUBIC)
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .build()
 *     .processImage(imagePath);
 *
 * // Batch processing
 * BatchProcessingResult result = processor.processBatch(imagePaths);
 *
 * // Async processing
 * ImageProcessorLibrary.builder()
 *     .targetSize(224, 224)
 *     .build()
 *     .processImageAsync(imagePath)
 *     .thenAccept(result -> System.out.println("Processed: " + result.isSuccess()));
 * }</pre>
 */
public class ImageProcessorLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(ImageProcessorLibrary.class.getName());

	// Configuration
	private final int targetWidth;
	private final int targetHeight;
	private final boolean maintainAspectRatio;
	private final boolean centerCrop;
	private final float[] meanNormalization;
	private final float[] stdNormalization;
	private final boolean normalizeToRange;
	private final InterpolationMethod interpolation;
	private final Consumer<ProcessingProgress> progressCallback;
	private final ExecutorService executor;
	private final boolean enableMetrics;
	private final int batchSize;

	private ImageProcessorLibrary(Builder builder) {
		this.targetWidth = builder.targetWidth;
		this.targetHeight = builder.targetHeight;
		this.maintainAspectRatio = builder.maintainAspectRatio;
		this.centerCrop = builder.centerCrop;
		this.meanNormalization = builder.meanNormalization.clone();
		this.stdNormalization = builder.stdNormalization.clone();
		this.normalizeToRange = builder.normalizeToRange;
		this.interpolation = builder.interpolation;
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
		this.enableMetrics = builder.enableMetrics;
		this.batchSize = builder.batchSize;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Process a single image from file path
	 */
	public ProcessingResult processImage(Path imagePath) {
		progress("Starting image processing", 0.0);
		Instant startTime = Instant.now();

		try {
			// Load image
			BufferedImage image = ImageIO.read(imagePath.toFile());
			if (image == null) {
				return new ProcessingResult.Builder()
					.success(false)
					.message("Failed to load image: " + imagePath)
					.imagePath(imagePath)
					.duration(Duration.between(startTime, Instant.now()))
					.build();
			}

			progress("Image loaded", 0.2);

			// Process image
			ProcessedImage processed = processBufferedImage(image);
			progress("Image processed", 0.8);

			// Create metadata
			ImageMetadata metadata = new ImageMetadata.Builder()
				.originalWidth(image.getWidth())
				.originalHeight(image.getHeight())
				.processedWidth(processed.processedWidth)
				.processedHeight(processed.processedHeight)
				.channels(processed.channels)
				.format(getImageFormat(imagePath))
				.aspectRatio((double) image.getWidth() / image.getHeight())
				.build();

			progress("Processing complete", 1.0);

			Duration duration = Duration.between(startTime, Instant.now());

			return new ProcessingResult.Builder()
				.success(true)
				.message("Image processed successfully")
				.imagePath(imagePath)
				.processedImage(processed)
				.metadata(metadata)
				.duration(duration)
				.build();

		} catch (Exception e) {
			String errorMsg = "Image processing failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new ProcessingResult.Builder()
				.success(false)
				.message(errorMsg)
				.imagePath(imagePath)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Process a BufferedImage directly
	 */
	public ProcessingResult processImage(BufferedImage image) {
		Instant startTime = Instant.now();

		try {
			ProcessedImage processed = processBufferedImage(image);

			ImageMetadata metadata = new ImageMetadata.Builder()
				.originalWidth(image.getWidth())
				.originalHeight(image.getHeight())
				.processedWidth(processed.processedWidth)
				.processedHeight(processed.processedHeight)
				.channels(processed.channels)
				.format("BufferedImage")
				.aspectRatio((double) image.getWidth() / image.getHeight())
				.build();

			Duration duration = Duration.between(startTime, Instant.now());

			return new ProcessingResult.Builder()
				.success(true)
				.message("BufferedImage processed successfully")
				.processedImage(processed)
				.metadata(metadata)
				.duration(duration)
				.build();

		} catch (Exception e) {
			return new ProcessingResult.Builder()
				.success(false)
				.message("BufferedImage processing failed: " + e.getMessage())
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Process multiple images in batches
	 */
	public BatchProcessingResult processBatch(List<Path> imagePaths) {
		progress("Starting batch image processing", 0.0);
		Instant startTime = Instant.now();

		try {
			List<ProcessingResult> results = new ArrayList<>();
			List<Path> failedPaths = new ArrayList<>();
			int totalBatches = (int) Math.ceil((double) imagePaths.size() / batchSize);

			for (int i = 0; i < totalBatches; i++) {
				int startIdx = i * batchSize;
				int endIdx = Math.min(startIdx + batchSize, imagePaths.size());
				List<Path> batch = imagePaths.subList(startIdx, endIdx);

				progress("Processing batch " + (i + 1) + "/" + totalBatches,
					(double) i / totalBatches);

				for (Path imagePath : batch) {
					ProcessingResult result = processImage(imagePath);
					results.add(result);

					if (!result.isSuccess()) {
						failedPaths.add(imagePath);
					}
				}
			}

			progress("Batch processing complete", 1.0);

			Duration duration = Duration.between(startTime, Instant.now());
			boolean success = failedPaths.isEmpty();

			return new BatchProcessingResult.Builder()
				.success(success)
				.message(String.format("Processed %d images, %d failed", imagePaths.size(), failedPaths.size()))
				.results(results)
				.totalImages(imagePaths.size())
				.successfulImages(results.size() - failedPaths.size())
				.failedImages(failedPaths.size())
				.failedPaths(failedPaths)
				.duration(duration)
				.build();

		} catch (Exception e) {
			String errorMsg = "Batch processing failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new BatchProcessingResult.Builder()
				.success(false)
				.message(errorMsg)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Process image asynchronously
	 */
	public CompletableFuture<ProcessingResult> processImageAsync(Path imagePath) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();
		return CompletableFuture.supplyAsync(() -> processImage(imagePath), exec);
	}

	/**
	 * Process batch asynchronously
	 */
	public CompletableFuture<BatchProcessingResult> processBatchAsync(List<Path> imagePaths) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();
		return CompletableFuture.supplyAsync(() -> processBatch(imagePaths), exec);
	}

	/**
	 * Resize image to target dimensions
	 */
	public ProcessingResult resizeImage(Path imagePath, int newWidth, int newHeight) {
		return ImageProcessorLibrary.builder()
			.targetSize(newWidth, newHeight)
			.maintainAspectRatio(false)
			.centerCrop(false)
			.build()
			.processImage(imagePath);
	}

	/**
	 * Extract image features as flattened array
	 */
	public FeatureExtractionResult extractFeatures(Path imagePath) {
		ProcessingResult result = processImage(imagePath);

		if (!result.isSuccess()) {
			return new FeatureExtractionResult.Builder()
				.success(false)
				.message("Feature extraction failed: " + result.getMessage())
				.imagePath(imagePath)
				.error(result.getError().orElse(null))
				.build();
		}

		float[] features = result.getProcessedImage().get().flatten();

		return new FeatureExtractionResult.Builder()
			.success(true)
			.message("Features extracted successfully")
			.imagePath(imagePath)
			.features(features)
			.featureCount(features.length)
			.metadata(result.getMetadata().orElse(null))
			.build();
	}

	/**
	 * Validate image file
	 */
	public ValidationResult validateImage(Path imagePath) {
		try {
			if (!imagePath.toFile().exists()) {
				return new ValidationResult(false, "Image file does not exist", imagePath);
			}

			BufferedImage image = ImageIO.read(imagePath.toFile());
			if (image == null) {
				return new ValidationResult(false, "Invalid or unsupported image format", imagePath);
			}

			String format = getImageFormat(imagePath);
			boolean supportedFormat = Arrays.asList("jpg", "jpeg", "png", "bmp", "gif").contains(format.toLowerCase());

			if (!supportedFormat) {
				return new ValidationResult(false, "Unsupported image format: " + format, imagePath);
			}

			return new ValidationResult(true, "Image is valid", imagePath, image.getWidth(), image.getHeight(), format);

		} catch (Exception e) {
			return new ValidationResult(false, "Image validation failed: " + e.getMessage(), imagePath);
		}
	}

	// Core processing logic
	private ProcessedImage processBufferedImage(BufferedImage image) {
		// Calculate target dimensions
		Dimension targetDim = calculateTargetDimensions(image.getWidth(), image.getHeight());

		// Resize image
		BufferedImage resized = resizeImage(image, targetDim.width, targetDim.height);

		// Center crop if needed
		if (centerCrop && maintainAspectRatio) {
			resized = centerCropImage(resized, targetWidth, targetHeight);
		}

		// Convert to RGB if necessary
		BufferedImage rgb = convertToRGB(resized);

		// Extract and normalize pixels
		float[][][] pixels = extractPixels(rgb);
		normalizePixels(pixels);

		ProcessedImage processed = new ProcessedImage();
		processed.pixels = pixels;
		processed.originalWidth = image.getWidth();
		processed.originalHeight = image.getHeight();
		processed.processedWidth = rgb.getWidth();
		processed.processedHeight = rgb.getHeight();
		processed.channels = 3; // RGB
		processed.format = "RGB";

		return processed;
	}

	private Dimension calculateTargetDimensions(int originalWidth, int originalHeight) {
		if (!maintainAspectRatio) {
			return new Dimension(targetWidth, targetHeight);
		}

		double aspectRatio = (double) originalWidth / originalHeight;
		int newWidth, newHeight;

		if (aspectRatio > 1.0) { // Landscape
			newWidth = targetWidth;
			newHeight = (int) (targetWidth / aspectRatio);
		} else { // Portrait or square
			newHeight = targetHeight;
			newWidth = (int) (targetHeight * aspectRatio);
		}

		return new Dimension(newWidth, newHeight);
	}

	private BufferedImage resizeImage(BufferedImage original, int newWidth, int newHeight) {
		BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = resized.createGraphics();

		// Set rendering hints based on interpolation method
		setRenderingHints(g2d);

		g2d.drawImage(original, 0, 0, newWidth, newHeight, null);
		g2d.dispose();

		return resized;
	}

	private void setRenderingHints(Graphics2D g2d) {
		switch (interpolation) {
			case BICUBIC:
				g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
				break;
			case BILINEAR:
				g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
				break;
			case NEAREST:
				g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
				break;
		}
		g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
		g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
	}

	private BufferedImage centerCropImage(BufferedImage image, int targetWidth, int targetHeight) {
		int x = Math.max(0, (image.getWidth() - targetWidth) / 2);
		int y = Math.max(0, (image.getHeight() - targetHeight) / 2);
		int cropWidth = Math.min(targetWidth, image.getWidth());
		int cropHeight = Math.min(targetHeight, image.getHeight());

		return image.getSubimage(x, y, cropWidth, cropHeight);
	}

	private BufferedImage convertToRGB(BufferedImage image) {
		if (image.getType() == BufferedImage.TYPE_INT_RGB) {
			return image;
		}

		BufferedImage rgb = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = rgb.createGraphics();
		g2d.drawImage(image, 0, 0, null);
		g2d.dispose();

		return rgb;
	}

	private float[][][] extractPixels(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		float[][][] pixels = new float[3][height][width]; // [channels][height][width]

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rgb = image.getRGB(x, y);
				pixels[0][y][x] = ((rgb >> 16) & 0xFF) / 255.0f; // Red
				pixels[1][y][x] = ((rgb >> 8) & 0xFF) / 255.0f;  // Green
				pixels[2][y][x] = (rgb & 0xFF) / 255.0f;         // Blue
			}
		}

		return pixels;
	}

	private void normalizePixels(float[][][] pixels) {
		int channels = pixels.length;
		int height = pixels[0].length;
		int width = pixels[0][0].length;

		for (int c = 0; c < channels; c++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					if (normalizeToRange) {
						// Normalize to [-1, 1]
						pixels[c][h][w] = (pixels[c][h][w] - 0.5f) * 2.0f;
					} else {
						// Standard ImageNet normalization
						pixels[c][h][w] = (pixels[c][h][w] - meanNormalization[c]) / stdNormalization[c];
					}
				}
			}
		}
	}

	private String getImageFormat(Path imagePath) {
		String fileName = imagePath.getFileName().toString();
		int lastDot = fileName.lastIndexOf('.');
		return lastDot > 0 ? fileName.substring(lastDot + 1) : "unknown";
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new ProcessingProgress(message, progress));
		}
	}

	@Override
	public void close() {
		if (executor != null) {
			executor.shutdown();
		}
	}

	// Enums and data classes
	public enum InterpolationMethod {
		BICUBIC, BILINEAR, NEAREST
	}

	public static class ProcessingProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public ProcessingProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	public static class ProcessedImage {
		public float[][][] pixels; // [channels][height][width]
		public int originalWidth;
		public int originalHeight;
		public int processedWidth;
		public int processedHeight;
		public int channels;
		public String format;

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

		public float[] flattenCHW() {
			return flatten(); // Already in CHW format
		}

		public float[] flattenHWC() {
			int totalElements = channels * processedHeight * processedWidth;
			float[] flattened = new float[totalElements];
			int index = 0;

			for (int h = 0; h < processedHeight; h++) {
				for (int w = 0; w < processedWidth; w++) {
					for (int c = 0; c < channels; c++) {
						flattened[index++] = pixels[c][h][w];
					}
				}
			}

			return flattened;
		}
	}

	// Builder class
	public static class Builder {
		private int targetWidth = 224;
		private int targetHeight = 224;
		private boolean maintainAspectRatio = true;
		private boolean centerCrop = true;
		private float[] meanNormalization = {0.485f, 0.456f, 0.406f}; // ImageNet means
		private float[] stdNormalization = {0.229f, 0.224f, 0.225f}; // ImageNet stds
		private boolean normalizeToRange = false;
		private InterpolationMethod interpolation = InterpolationMethod.BICUBIC;
		private Consumer<ProcessingProgress> progressCallback;
		private ExecutorService executor;
		private boolean enableMetrics = false;
		private int batchSize = 32;

		public Builder targetSize(int width, int height) {
			this.targetWidth = width;
			this.targetHeight = height;
			return this;
		}

		public Builder maintainAspectRatio(boolean maintainAspectRatio) {
			this.maintainAspectRatio = maintainAspectRatio;
			return this;
		}

		public Builder centerCrop(boolean centerCrop) {
			this.centerCrop = centerCrop;
			return this;
		}

		public Builder meanNormalization(float[] meanNormalization) {
			this.meanNormalization = meanNormalization.clone();
			return this;
		}

		public Builder stdNormalization(float[] stdNormalization) {
			this.stdNormalization = stdNormalization.clone();
			return this;
		}

		public Builder normalizeToRange(boolean normalizeToRange) {
			this.normalizeToRange = normalizeToRange;
			return this;
		}

		public Builder interpolation(InterpolationMethod interpolation) {
			this.interpolation = interpolation;
			return this;
		}

		public Builder progressCallback(Consumer<ProcessingProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public Builder enableMetrics(boolean enableMetrics) {
			this.enableMetrics = enableMetrics;
			return this;
		}

		public Builder batchSize(int batchSize) {
			this.batchSize = Math.max(1, batchSize);
			return this;
		}

		public ImageProcessorLibrary build() {
			return new ImageProcessorLibrary(this);
		}
	}

	// Result classes
	public static class ProcessingResult {
		private final boolean success;
		private final String message;
		private final Path imagePath;
		private final ProcessedImage processedImage;
		private final ImageMetadata metadata;
		private final Duration duration;
		private final Exception error;

		private ProcessingResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.imagePath = builder.imagePath;
			this.processedImage = builder.processedImage;
			this.metadata = builder.metadata;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Optional<Path> getImagePath() { return Optional.ofNullable(imagePath); }
		public Optional<ProcessedImage> getProcessedImage() { return Optional.ofNullable(processedImage); }
		public Optional<ImageMetadata> getMetadata() { return Optional.ofNullable(metadata); }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private Path imagePath;
			private ProcessedImage processedImage;
			private ImageMetadata metadata;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder imagePath(Path imagePath) { this.imagePath = imagePath; return this; }
			public Builder processedImage(ProcessedImage processedImage) { this.processedImage = processedImage; return this; }
			public Builder metadata(ImageMetadata metadata) { this.metadata = metadata; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public ProcessingResult build() { return new ProcessingResult(this); }
		}
	}

	public static class BatchProcessingResult {
		private final boolean success;
		private final String message;
		private final List<ProcessingResult> results;
		private final int totalImages;
		private final int successfulImages;
		private final int failedImages;
		private final List<Path> failedPaths;
		private final Duration duration;
		private final Exception error;

		private BatchProcessingResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.results = Collections.unmodifiableList(builder.results);
			this.totalImages = builder.totalImages;
			this.successfulImages = builder.successfulImages;
			this.failedImages = builder.failedImages;
			this.failedPaths = Collections.unmodifiableList(builder.failedPaths);
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public List<ProcessingResult> getResults() { return results; }
		public int getTotalImages() { return totalImages; }
		public int getSuccessfulImages() { return successfulImages; }
		public int getFailedImages() { return failedImages; }
		public List<Path> getFailedPaths() { return failedPaths; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }
		public double getSuccessRate() { return totalImages > 0 ? (double) successfulImages / totalImages : 0.0; }

		public static class Builder {
			private boolean success;
			private String message;
			private List<ProcessingResult> results = new ArrayList<>();
			private int totalImages;
			private int successfulImages;
			private int failedImages;
			private List<Path> failedPaths = new ArrayList<>();
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder results(List<ProcessingResult> results) { this.results = results; return this; }
			public Builder totalImages(int totalImages) { this.totalImages = totalImages; return this; }
			public Builder successfulImages(int successfulImages) { this.successfulImages = successfulImages; return this; }
			public Builder failedImages(int failedImages) { this.failedImages = failedImages; return this; }
			public Builder failedPaths(List<Path> failedPaths) { this.failedPaths = failedPaths; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public BatchProcessingResult build() { return new BatchProcessingResult(this); }
		}
	}

	public static class FeatureExtractionResult {
		private final boolean success;
		private final String message;
		private final Path imagePath;
		private final float[] features;
		private final int featureCount;
		private final ImageMetadata metadata;
		private final Exception error;

		private FeatureExtractionResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.imagePath = builder.imagePath;
			this.features = builder.features != null ? builder.features.clone() : new float[0];
			this.featureCount = builder.featureCount;
			this.metadata = builder.metadata;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Optional<Path> getImagePath() { return Optional.ofNullable(imagePath); }
		public float[] getFeatures() { return features.clone(); }
		public int getFeatureCount() { return featureCount; }
		public Optional<ImageMetadata> getMetadata() { return Optional.ofNullable(metadata); }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private Path imagePath;
			private float[] features;
			private int featureCount;
			private ImageMetadata metadata;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder imagePath(Path imagePath) { this.imagePath = imagePath; return this; }
			public Builder features(float[] features) { this.features = features; return this; }
			public Builder featureCount(int featureCount) { this.featureCount = featureCount; return this; }
			public Builder metadata(ImageMetadata metadata) { this.metadata = metadata; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public FeatureExtractionResult build() { return new FeatureExtractionResult(this); }
		}
	}

	public static class ImageMetadata {
		private final int originalWidth;
		private final int originalHeight;
		private final int processedWidth;
		private final int processedHeight;
		private final int channels;
		private final String format;
		private final double aspectRatio;

		private ImageMetadata(Builder builder) {
			this.originalWidth = builder.originalWidth;
			this.originalHeight = builder.originalHeight;
			this.processedWidth = builder.processedWidth;
			this.processedHeight = builder.processedHeight;
			this.channels = builder.channels;
			this.format = builder.format;
			this.aspectRatio = builder.aspectRatio;
		}

		public int getOriginalWidth() { return originalWidth; }
		public int getOriginalHeight() { return originalHeight; }
		public int getProcessedWidth() { return processedWidth; }
		public int getProcessedHeight() { return processedHeight; }
		public int getChannels() { return channels; }
		public String getFormat() { return format; }
		public double getAspectRatio() { return aspectRatio; }

		public static class Builder {
			private int originalWidth;
			private int originalHeight;
			private int processedWidth;
			private int processedHeight;
			private int channels;
			private String format;
			private double aspectRatio;

			public Builder originalWidth(int originalWidth) { this.originalWidth = originalWidth; return this; }
			public Builder originalHeight(int originalHeight) { this.originalHeight = originalHeight; return this; }
			public Builder processedWidth(int processedWidth) { this.processedWidth = processedWidth; return this; }
			public Builder processedHeight(int processedHeight) { this.processedHeight = processedHeight; return this; }
			public Builder channels(int channels) { this.channels = channels; return this; }
			public Builder format(String format) { this.format = format; return this; }
			public Builder aspectRatio(double aspectRatio) { this.aspectRatio = aspectRatio; return this; }

			public ImageMetadata build() { return new ImageMetadata(this); }
		}
	}

	public static class ValidationResult {
		private final boolean valid;
		private final String message;
		private final Path imagePath;
		private final int width;
		private final int height;
		private final String format;

		public ValidationResult(boolean valid, String message, Path imagePath) {
			this(valid, message, imagePath, 0, 0, "");
		}

		public ValidationResult(boolean valid, String message, Path imagePath, int width, int height, String format) {
			this.valid = valid;
			this.message = message;
			this.imagePath = imagePath;
			this.width = width;
			this.height = height;
			this.format = format;
		}

		public boolean isValid() { return valid; }
		public String getMessage() { return message; }
		public Path getImagePath() { return imagePath; }
		public int getWidth() { return width; }
		public int getHeight() { return height; }
		public String getFormat() { return format; }
	}
}