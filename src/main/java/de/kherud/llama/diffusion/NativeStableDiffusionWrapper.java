package de.kherud.llama.diffusion;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

/**
 * High-level wrapper for Native Stable Diffusion integration.
 *
 * This class provides an easier-to-use interface for Stable Diffusion image generation,
 * with automatic resource management and sensible defaults.
 */
public class NativeStableDiffusionWrapper implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(NativeStableDiffusionWrapper.class.getName());

	private final long contextHandle;
	private final String modelPath;
	private boolean closed = false;

	/**
	 * Parameters for image generation.
	 */
	public static class GenerationParameters {
		public String prompt = "";
		public String negativePrompt = "";
		public int width = 768;
		public int height = 768;
		public int steps = 30;
		public float cfgScale = 7.0f;
		public float slgScale = 2.5f;  // Good for SD3.5 Medium
		public int seed = -1;
		public int sampleMethod = NativeStableDiffusion.SAMPLE_METHOD_EULER;
		public boolean clipOnCpu = true;

		// ControlNet parameters
		public byte[] controlImage = null;
		public int controlImageWidth = 0;
		public int controlImageHeight = 0;
		public int controlImageChannels = 3;
		public float controlStrength = 0.9f;

		// Image-to-image parameters
		public byte[] initImage = null;
		public int initImageWidth = 0;
		public int initImageHeight = 0;
		public int initImageChannels = 3;
		public float strength = 0.8f;

		// Inpainting parameters
		public byte[] maskImage = null;
		public int maskImageWidth = 0;
		public int maskImageHeight = 0;
		public int maskImageChannels = 1;

		public static GenerationParameters defaults() {
			return new GenerationParameters();
		}

		public static GenerationParameters forSD35Medium() {
			GenerationParameters params = new GenerationParameters();
			params.slgScale = 2.5f;  // Optimal for SD3.5 Medium
			params.cfgScale = 7.0f;
			params.steps = 30;
			return params;
		}

		public GenerationParameters withPrompt(String prompt) {
			this.prompt = prompt;
			return this;
		}

		public GenerationParameters withNegativePrompt(String negativePrompt) {
			this.negativePrompt = negativePrompt;
			return this;
		}

		public GenerationParameters withSize(int width, int height) {
			this.width = width;
			this.height = height;
			return this;
		}

		public GenerationParameters withWidth(int width) {
			this.width = width;
			return this;
		}

		public GenerationParameters withHeight(int height) {
			this.height = height;
			return this;
		}

		public GenerationParameters withSteps(int steps) {
			this.steps = steps;
			return this;
		}

		public GenerationParameters withCfgScale(float cfgScale) {
			this.cfgScale = cfgScale;
			return this;
		}

		public GenerationParameters withSlgScale(float slgScale) {
			this.slgScale = slgScale;
			return this;
		}

		public GenerationParameters withSeed(int seed) {
			this.seed = seed;
			return this;
		}

		public GenerationParameters withSampleMethod(int sampleMethod) {
			this.sampleMethod = sampleMethod;
			return this;
		}

		public GenerationParameters withControlImage(byte[] controlImage, int width, int height, int channels) {
			this.controlImage = controlImage;
			this.controlImageWidth = width;
			this.controlImageHeight = height;
			this.controlImageChannels = channels;
			return this;
		}

		public GenerationParameters withControlImage(byte[] controlImage, int width, int height) {
			return withControlImage(controlImage, width, height, 3);
		}

		public GenerationParameters withControlStrength(float controlStrength) {
			this.controlStrength = controlStrength;
			return this;
		}

		public GenerationParameters withInitImage(byte[] initImage, int width, int height, int channels) {
			this.initImage = initImage;
			this.initImageWidth = width;
			this.initImageHeight = height;
			this.initImageChannels = channels;
			return this;
		}

		public GenerationParameters withInitImage(byte[] initImage, int width, int height) {
			return withInitImage(initImage, width, height, 3);
		}

		public GenerationParameters withStrength(float strength) {
			this.strength = strength;
			return this;
		}

		public GenerationParameters withMaskImage(byte[] maskImage, int width, int height, int channels) {
			this.maskImage = maskImage;
			this.maskImageWidth = width;
			this.maskImageHeight = height;
			this.maskImageChannels = channels;
			return this;
		}

		public GenerationParameters withMaskImage(byte[] maskImage, int width, int height) {
			return withMaskImage(maskImage, width, height, 1);
		}
	}

	/**
	 * Builder for creating NativeStableDiffusionWrapper instances.
	 */
	public static class Builder {
		private String modelPath;
		private String clipLPath;
		private String clipGPath;
		private String t5xxlPath;
		private String controlNetPath;
		private boolean keepClipOnCpu = true;
		private boolean keepControlNetOnCpu = false;

		public Builder modelPath(String modelPath) {
			this.modelPath = modelPath;
			return this;
		}

		public Builder clipL(String clipLPath) {
			this.clipLPath = clipLPath;
			return this;
		}

		public Builder clipG(String clipGPath) {
			this.clipGPath = clipGPath;
			return this;
		}

		public Builder t5xxl(String t5xxlPath) {
			this.t5xxlPath = t5xxlPath;
			return this;
		}

		public Builder keepClipOnCpu(boolean keepClipOnCpu) {
			this.keepClipOnCpu = keepClipOnCpu;
			return this;
		}

		public Builder controlNet(String controlNetPath) {
			this.controlNetPath = controlNetPath;
			return this;
		}

		public Builder keepControlNetOnCpu(boolean keepControlNetOnCpu) {
			this.keepControlNetOnCpu = keepControlNetOnCpu;
			return this;
		}

		public NativeStableDiffusionWrapper build() throws IllegalStateException {
			if (modelPath == null) {
				throw new IllegalArgumentException("Model path is required");
			}
			return new NativeStableDiffusionWrapper(this);
		}
	}

	private NativeStableDiffusionWrapper(Builder builder) throws IllegalStateException {
		this.modelPath = builder.modelPath;

		// Validate model file exists
		if (!Files.exists(Paths.get(modelPath))) {
			throw new IllegalArgumentException("Model file not found: " + modelPath);
		}

		// Resolve symbolic links for better native library compatibility
		String resolvedModelPath = resolveSymbolicLinks(builder.modelPath);
		String resolvedClipLPath = resolveSymbolicLinks(builder.clipLPath);
		String resolvedClipGPath = resolveSymbolicLinks(builder.clipGPath);
		String resolvedT5xxlPath = resolveSymbolicLinks(builder.t5xxlPath);
		String resolvedControlNetPath = resolveSymbolicLinks(builder.controlNetPath);

		// Create the native context
		if (resolvedControlNetPath != null && !resolvedControlNetPath.isEmpty()) {
			this.contextHandle = NativeStableDiffusion.createContextWithControlNet(
				resolvedModelPath,
				resolvedClipLPath,
				resolvedClipGPath,
				resolvedT5xxlPath,
				resolvedControlNetPath,
				builder.keepClipOnCpu,
				builder.keepControlNetOnCpu
			);
		} else {
			this.contextHandle = NativeStableDiffusion.createContext(
				resolvedModelPath,
				resolvedClipLPath,
				resolvedClipGPath,
				resolvedT5xxlPath,
				builder.keepClipOnCpu
			);
		}

		if (contextHandle == 0) {
			String error = NativeStableDiffusion.getLastError();
			throw new IllegalStateException("Failed to create Stable Diffusion context: " + error);
		}

		LOGGER.log(System.Logger.Level.INFO, "Created Stable Diffusion context for model: " + modelPath);
	}

	/**
	 * Private constructor for creating a wrapper with an existing context handle.
	 * Used by createWithXL() for Diffusers models.
	 */
	private NativeStableDiffusionWrapper(long contextHandle, String modelPath) {
		this.contextHandle = contextHandle;
		this.modelPath = modelPath;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Resolves symbolic links to actual file paths for better native library compatibility.
	 */
	private static String resolveSymbolicLinks(String path) {
		if (path == null || path.isEmpty()) {
			return path;
		}
		try {
			Path resolved = Paths.get(path).toRealPath();
			return resolved.toString();
		} catch (Exception e) {
			// If resolution fails, return original path
			LOGGER.log(System.Logger.Level.WARNING, "Failed to resolve symbolic link for: " + path + " - " + e.getMessage());
			return path;
		}
	}

	/**
	 * Generate an image using the specified parameters.
	 *
	 * @param params Generation parameters
	 * @return StableDiffusionResult containing the generated image or error information
	 * @throws IllegalStateException if the wrapper has been closed
	 */
	public StableDiffusionResult generateImage(GenerationParameters params) throws IllegalStateException {
		checkNotClosed();

		if (params.prompt == null || params.prompt.trim().isEmpty()) {
			throw new IllegalArgumentException("Prompt cannot be null or empty");
		}

		// Validate inpainting requirements
		if (params.maskImage != null) {
			if (params.initImage == null) {
				throw new IllegalArgumentException("Inpainting requires an init image along with the mask. Use withInitImage() before withMaskImage()");
			}

			// Check if this is an SD3 model (SD3 doesn't support inpainting)
			String modelPath = this.modelPath.toLowerCase();
			if (modelPath.contains("sd3") || modelPath.contains("sd_3") ||
				modelPath.contains("stable-diffusion-3") || modelPath.contains("sd3.5")) {
				throw new IllegalArgumentException(
					"SD3/SD3.5 models do not support inpainting. " +
					"Use SD1.5-inpaint, SD2-inpaint, or SDXL-inpaint models instead. " +
					"Current model: " + this.modelPath
				);
			}
		}

		// Check if we need to use advanced generation (ControlNet, img2img, or inpainting)
		boolean useAdvancedGeneration = (params.controlImage != null) || (params.initImage != null) || (params.maskImage != null);

		if (useAdvancedGeneration) {
			LOGGER.log(System.Logger.Level.INFO,
				"Generating image (advanced): {}x{}, steps={}, cfg={}, slg={}, controlNet={}, img2img={}, inpainting={}, prompt='{}'",
				params.width, params.height, params.steps, params.cfgScale, params.slgScale,
				params.controlImage != null, params.initImage != null, params.maskImage != null, params.prompt);

			return NativeStableDiffusion.generateImageAdvanced(
				contextHandle,
				params.prompt,
				params.negativePrompt,
				params.width,
				params.height,
				params.steps,
				params.cfgScale,
				params.slgScale,
				params.seed,
				params.sampleMethod,
				params.clipOnCpu,
				params.controlImage,
				params.controlImageWidth,
				params.controlImageHeight,
				params.controlImageChannels,
				params.controlStrength,
				params.initImage,
				params.initImageWidth,
				params.initImageHeight,
				params.initImageChannels,
				params.strength,
				params.maskImage,
				params.maskImageWidth,
				params.maskImageHeight,
				params.maskImageChannels
			);
		} else {
			LOGGER.log(System.Logger.Level.INFO,
				"Generating image: {}x{}, steps={}, cfg={}, slg={}, prompt='{}'",
				params.width, params.height, params.steps, params.cfgScale, params.slgScale,
				params.prompt);

			return NativeStableDiffusion.generateImage(
				contextHandle,
				params.prompt,
				params.negativePrompt,
				params.width,
				params.height,
				params.steps,
				params.cfgScale,
				params.slgScale,
				params.seed,
				params.sampleMethod,
				params.clipOnCpu
			);
		}
	}

	/**
	 * Generate an image with a simple prompt and default parameters.
	 *
	 * @param prompt Text prompt describing the desired image
	 * @return StableDiffusionResult containing the generated image or error information
	 */
	public StableDiffusionResult generateImage(String prompt) {
		return generateImage(GenerationParameters.defaults().withPrompt(prompt));
	}

	/**
	 * Generate an image with a prompt and custom size.
	 *
	 * @param prompt Text prompt describing the desired image
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @return StableDiffusionResult containing the generated image or error information
	 */
	public StableDiffusionResult generateImage(String prompt, int width, int height) {
		return generateImage(GenerationParameters.defaults()
			.withPrompt(prompt)
			.withSize(width, height));
	}

	/**
	 * Save generated image data to a PNG file.
	 *
	 * @param result The generation result containing image data
	 * @param outputPath Path where to save the PNG file
	 * @throws IOException if saving fails
	 * @throws IllegalArgumentException if result doesn't contain valid image data
	 */
	public static void saveImageAsPng(StableDiffusionResult result, Path outputPath) throws IOException {
		if (!result.isSuccess()) {
			throw new IllegalArgumentException("Cannot save failed generation result");
		}

		Optional<byte[]> imageData = result.getImageData();
		if (!imageData.isPresent()) {
			throw new IllegalArgumentException("No image data in result");
		}

		// Convert RGB data to PNG format
		byte[] pngData = convertRgbToPng(imageData.get(), result.getWidth(), result.getHeight(), result.getChannels());
		Files.write(outputPath, pngData);

		LOGGER.log(System.Logger.Level.INFO, "Saved image to: " + outputPath);
	}

	/**
	 * Check if the wrapper is available and ready for use.
	 *
	 * @return true if the wrapper can be used for generation
	 */
	public boolean isAvailable() {
		return !closed && NativeStableDiffusion.isValidHandle(contextHandle);
	}

	/**
	 * Get the model path this wrapper was created with.
	 *
	 * @return Model file path
	 */
	public String getModelPath() {
		return modelPath;
	}

	/**
	 * Get system information about the stable-diffusion.cpp backend.
	 *
	 * @return System information string
	 */
	public static String getSystemInfo() {
		return NativeStableDiffusion.getSystemInfo();
	}

	@Override
	public void close() {
		if (!closed && contextHandle != 0) {
			boolean success = NativeStableDiffusion.destroyContext(contextHandle);
			if (success) {
				LOGGER.log(System.Logger.Level.INFO, "Closed Stable Diffusion context");
			} else {
				LOGGER.log(System.Logger.Level.WARNING, "Failed to properly close Stable Diffusion context");
			}
			closed = true;
		}
	}

	private void checkNotClosed() throws IllegalStateException {
		if (closed) {
			throw new IllegalStateException("NativeStableDiffusionWrapper has been closed");
		}
	}

	/**
	 * Convert RGB image data to PNG format using Java's built-in ImageIO.
	 */
	private static byte[] convertRgbToPng(byte[] rgbData, int width, int height, int channels) {
		try {
			// Create BufferedImage from RGB data
			java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(width, height, java.awt.image.BufferedImage.TYPE_INT_RGB);

			// Convert byte array to int array for BufferedImage
			int[] pixels = new int[width * height];
			for (int i = 0; i < pixels.length; i++) {
				int baseIndex = i * channels;
				int r = rgbData[baseIndex] & 0xFF;
				int g = rgbData[baseIndex + 1] & 0xFF;
				int b = rgbData[baseIndex + 2] & 0xFF;
				pixels[i] = (r << 16) | (g << 8) | b;
			}
			image.setRGB(0, 0, width, height, pixels, 0, width);

			// Convert to PNG bytes
			java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
			javax.imageio.ImageIO.write(image, "png", baos);
			return baos.toByteArray();

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Failed to convert RGB to PNG", e);
			// Fallback: return raw data
			return rgbData;
		}
	}


	public static Optional<NativeStableDiffusionWrapper> createWithXL() {
		Path modelsDir = Paths.get(System.getProperty("user.home"), "ai-models", "noob");

		// Check if this is a valid Diffusers directory structure
		Path unetDir = modelsDir.resolve("unet");
		Path textEncoderDir = modelsDir.resolve("text_encoder");
		Path textEncoder2Dir = modelsDir.resolve("text_encoder_2");

		if (Files.exists(unetDir) && Files.isDirectory(unetDir) &&
			Files.exists(textEncoderDir) && Files.isDirectory(textEncoderDir)) {

			try {
				// Use the new Diffusers-compatible native method
				long contextHandle = NativeStableDiffusion.createContextFromDiffusers(
					modelsDir.toString(),
					Files.exists(textEncoderDir.resolve("model.safetensors")) ?
						textEncoderDir.resolve("model.safetensors").toString() : null,
					Files.exists(textEncoder2Dir.resolve("model.safetensors")) ?
						textEncoder2Dir.resolve("model.safetensors").toString() : null,
					true
				);

				if (contextHandle == 0) {
					String error = NativeStableDiffusion.getLastError();
					throw new IllegalStateException("Failed to create Stable Diffusion context from Diffusers: " + error);
				}

				LOGGER.log(System.Logger.Level.INFO, "Created SDXL Diffusers context for model: " + modelsDir);
				return Optional.of(new NativeStableDiffusionWrapper(contextHandle, modelsDir.toString()));

			} catch (Exception e) {
				LOGGER.log(System.Logger.Level.WARNING, "Failed to create wrapper with Diffusers model", e);
			}
		}
		return Optional.empty();
	}

	/**
	 * Create a wrapper instance with automatic model detection.
	 * Looks for the best available SD3.5 Medium model in the standard location.
	 *
	 * @return NativeStableDiffusionWrapper instance, or empty if no models found
	 */
	public static Optional<NativeStableDiffusionWrapper> createWithAutoDetection() {
		Path modelsDir = Paths.get(System.getProperty("user.home"), "ai-models", "stable-diffusion-v3-5-medium");

		// Check for safetensors files
		Path mainModel = modelsDir.resolve("sd3.5_medium.safetensors");
		Path clipL = modelsDir.resolve("text_encoders/clip_l.safetensors");
		Path clipG = modelsDir.resolve("text_encoders/clip_g.safetensors");
		Path t5xxl = modelsDir.resolve("text_encoders/t5xxl_fp16.safetensors");

		if (Files.exists(mainModel)) {
			try {
				Builder builder = builder().modelPath(mainModel.toString());

				if (Files.exists(clipL)) {
					builder.clipL(clipL.toString());
				}
				if (Files.exists(clipG)) {
					builder.clipG(clipG.toString());
				}
				if (Files.exists(t5xxl)) {
					builder.t5xxl(t5xxl.toString());
				}

				return Optional.of(builder.build());
			} catch (Exception e) {
				LOGGER.log(System.Logger.Level.WARNING, "Failed to create wrapper with safetensors model", e);
			}
		}

		// Fallback to GGUF files
		String[] preferredModels = {
			"stable-diffusion-v3-5-medium-FP16.gguf",
			"stable-diffusion-v3-5-medium-Q8_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_0.gguf",
			"stable-diffusion-v3-5-medium-Q4_1.gguf"
		};

		for (String model : preferredModels) {
			Path modelPath = modelsDir.resolve(model);
			if (Files.exists(modelPath)) {
				try {
					return Optional.of(builder()
						.modelPath(modelPath.toString())
						.build());
				} catch (Exception e) {
					LOGGER.log(System.Logger.Level.WARNING, "Failed to create wrapper with model: " + model, e);
				}
			}
		}

		return Optional.empty();
	}

	/**
	 * Create a grayscale mask from RGB image data.
	 * Converts color image to single-channel mask suitable for inpainting.
	 *
	 * @param rgbData RGB image data (3 or 4 channels)
	 * @param width Image width
	 * @param height Image height
	 * @param channels Number of channels in input (3 for RGB, 4 for RGBA)
	 * @return Grayscale mask data (single channel)
	 */
	public static byte[] createMaskFromRgb(byte[] rgbData, int width, int height, int channels) {
		if (rgbData == null) {
			throw new IllegalArgumentException("RGB data cannot be null");
		}
		if (channels < 3 || channels > 4) {
			throw new IllegalArgumentException("Channels must be 3 (RGB) or 4 (RGBA)");
		}

		int expectedSize = width * height * channels;
		if (rgbData.length != expectedSize) {
			throw new IllegalArgumentException("RGB data size mismatch: expected " + expectedSize + ", got " + rgbData.length);
		}

		byte[] maskData = new byte[width * height];

		for (int i = 0; i < width * height; i++) {
			int baseIndex = i * channels;
			int r = rgbData[baseIndex] & 0xFF;
			int g = rgbData[baseIndex + 1] & 0xFF;
			int b = rgbData[baseIndex + 2] & 0xFF;

			// Convert to grayscale using luminance formula
			int gray = (int)(0.299 * r + 0.587 * g + 0.114 * b);
			maskData[i] = (byte)gray;
		}

		return maskData;
	}

	/**
	 * Create a binary mask where white pixels (255) indicate areas to inpaint
	 * and black pixels (0) indicate areas to preserve.
	 *
	 * @param width Mask width
	 * @param height Mask height
	 * @param fillValue Fill value for the mask (0-255, typically 0 or 255)
	 * @return Binary mask data
	 */
	public static byte[] createUniformMask(int width, int height, int fillValue) {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("Width and height must be positive");
		}
		if (fillValue < 0 || fillValue > 255) {
			throw new IllegalArgumentException("Fill value must be between 0 and 255");
		}

		byte[] maskData = new byte[width * height];
		byte fill = (byte)fillValue;

		for (int i = 0; i < maskData.length; i++) {
			maskData[i] = fill;
		}

		return maskData;
	}

	/**
	 * Create a rectangular mask with specified area to inpaint.
	 *
	 * @param width Mask width
	 * @param height Mask height
	 * @param rectX Rectangle top-left X coordinate
	 * @param rectY Rectangle top-left Y coordinate
	 * @param rectWidth Rectangle width
	 * @param rectHeight Rectangle height
	 * @return Mask data with white rectangle on black background
	 */
	public static byte[] createRectangularMask(int width, int height, int rectX, int rectY, int rectWidth, int rectHeight) {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("Width and height must be positive");
		}
		if (rectX < 0 || rectY < 0 || rectX + rectWidth > width || rectY + rectHeight > height) {
			throw new IllegalArgumentException("Rectangle must be within mask bounds");
		}

		byte[] maskData = new byte[width * height];

		// Fill background with black (preserve)
		java.util.Arrays.fill(maskData, (byte)0);

		// Fill rectangle with white (inpaint)
		for (int y = rectY; y < rectY + rectHeight; y++) {
			for (int x = rectX; x < rectX + rectWidth; x++) {
				maskData[y * width + x] = (byte)255;
			}
		}

		return maskData;
	}

	/**
	 * Validate mask image parameters.
	 *
	 * @param maskData Mask image data
	 * @param width Mask width
	 * @param height Mask height
	 * @param channels Number of channels (should be 1 for grayscale masks)
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public static void validateMask(byte[] maskData, int width, int height, int channels) {
		if (maskData == null) {
			throw new IllegalArgumentException("Mask data cannot be null");
		}
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("Width and height must be positive");
		}
		if (channels != 1) {
			throw new IllegalArgumentException("Mask should have 1 channel (grayscale), got " + channels);
		}

		int expectedSize = width * height * channels;
		if (maskData.length != expectedSize) {
			throw new IllegalArgumentException("Mask data size mismatch: expected " + expectedSize + ", got " + maskData.length);
		}
	}
}
