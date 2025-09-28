package de.kherud.llama.diffusion;

import de.kherud.llama.LlamaLoader;

/**
 * Native interface for Stable Diffusion image generation using stable-diffusion.cpp.
 *
 * This class provides JNI bindings to the stable-diffusion.cpp library for
 * high-performance image generation from text prompts.
 */
public final class NativeStableDiffusion {

	static {
		// Load the native libraries - stable-diffusion first, then jllama
		try {
			// First initialize the loader and load stable-diffusion library
			LlamaLoader.initialize();
			// The jllama library will be loaded automatically when needed
		} catch (Exception e) {
			// Library loading handled by LlamaLoader
		}
	}

	// Private constructor to prevent instantiation
	private NativeStableDiffusion() {
		throw new AssertionError("Utility class should not be instantiated");
	}

	// Sample method constants
	public static final int SAMPLE_METHOD_EULER = 1;
	public static final int SAMPLE_METHOD_HEUN = 2;
	public static final int SAMPLE_METHOD_DPM2 = 3;
	public static final int SAMPLE_METHOD_DPMPP2S_A = 4;
	public static final int SAMPLE_METHOD_DPMPP2M = 5;
	public static final int SAMPLE_METHOD_DPMPP2Mv2 = 6;
	public static final int SAMPLE_METHOD_EULER_A = 50;

	/**
	 * Create a new Stable Diffusion context.
	 *
	 * @param modelPath Path to the main Stable Diffusion model (.gguf file)
	 * @param clipLPath Path to CLIP-L text encoder (optional, can be null)
	 * @param clipGPath Path to CLIP-G text encoder (optional, can be null)
	 * @param t5xxlPath Path to T5XXL text encoder (optional, can be null)
	 * @param keepClipOnCpu Whether to keep CLIP models on CPU to save GPU memory
	 * @return Handle to the created context, or 0 if creation failed
	 */
	public static native long createContext(String modelPath, String clipLPath,
											String clipGPath, String t5xxlPath,
											boolean keepClipOnCpu);

	/**
	 * Create a new Stable Diffusion context with ControlNet support.
	 *
	 * @param modelPath Path to the main Stable Diffusion model (.gguf file)
	 * @param clipLPath Path to CLIP-L text encoder (optional, can be null)
	 * @param clipGPath Path to CLIP-G text encoder (optional, can be null)
	 * @param t5xxlPath Path to T5XXL text encoder (optional, can be null)
	 * @param controlNetPath Path to ControlNet model (optional, can be null)
	 * @param keepClipOnCpu Whether to keep CLIP models on CPU to save GPU memory
	 * @param keepControlNetOnCpu Whether to keep ControlNet on CPU to save GPU memory
	 * @return Handle to the created context, or 0 if creation failed
	 */
	public static native long createContextWithControlNet(String modelPath, String clipLPath,
														  String clipGPath, String t5xxlPath,
														  String controlNetPath, boolean keepClipOnCpu,
														  boolean keepControlNetOnCpu);

	/**
	 * Create a new Stable Diffusion context using Diffusers directory format.
	 *
	 * @param diffusersPath Path to directory containing Diffusers model (unet/, text_encoder/, etc.)
	 * @param clipLPath Path to CLIP-L text encoder (optional, can be null)
	 * @param clipGPath Path to CLIP-G text encoder (optional, can be null)
	 * @param keepClipOnCpu Whether to keep CLIP models on CPU to save GPU memory
	 * @return Handle to the created context, or 0 if creation failed
	 */
	public static native long createContextFromDiffusers(String diffusersPath, String clipLPath,
														 String clipGPath, boolean keepClipOnCpu);

	/**
	 * Destroy a Stable Diffusion context and free its resources.
	 *
	 * @param handle Handle to the context to destroy
	 * @return true if destruction was successful
	 */
	public static native boolean destroyContext(long handle);

	/**
	 * Generate an image using Stable Diffusion.
	 *
	 * @param handle Handle to the Stable Diffusion context
	 * @param prompt Text prompt describing the desired image
	 * @param negativePrompt Text describing what to avoid in the image (optional)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param steps Number of denoising steps (higher = better quality, slower)
	 * @param cfgScale Classifier-free guidance scale (how closely to follow prompt)
	 * @param slgScale Skip Layer Guidance scale (for SD3.5 Medium)
	 * @param seed Random seed for reproducible generation (-1 for random)
	 * @param sampleMethod Sampling method (use SAMPLE_METHOD_* constants)
	 * @param clipOnCpu Whether to run CLIP on CPU
	 * @return StableDiffusionResult containing the generated image or error information
	 */
	public static native StableDiffusionResult generateImage(long handle, String prompt,
															 String negativePrompt, int width, int height,
															 int steps, float cfgScale, float slgScale,
															 int seed, int sampleMethod, boolean clipOnCpu);

	/**
	 * Generate an image with ControlNet, img2img, and/or inpainting support.
	 *
	 * @param handle Handle to the Stable Diffusion context
	 * @param prompt Text prompt describing the desired image
	 * @param negativePrompt Text describing what to avoid in the image (optional)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param steps Number of denoising steps (higher = better quality, slower)
	 * @param cfgScale Classifier-free guidance scale (how closely to follow prompt)
	 * @param slgScale Skip Layer Guidance scale (for SD3.5 Medium)
	 * @param seed Random seed for reproducible generation (-1 for random)
	 * @param sampleMethod Sampling method (use SAMPLE_METHOD_* constants)
	 * @param clipOnCpu Whether to run CLIP on CPU
	 * @param controlImage Control image data (RGB bytes, can be null)
	 * @param controlImageWidth Control image width
	 * @param controlImageHeight Control image height
	 * @param controlImageChannels Control image channels
	 * @param controlStrength ControlNet influence strength (0.0-1.0)
	 * @param initImage Initial image for img2img (RGB bytes, can be null)
	 * @param initImageWidth Initial image width
	 * @param initImageHeight Initial image height
	 * @param initImageChannels Initial image channels
	 * @param strength Img2img denoising strength (0.0-1.0)
	 * @param maskImage Mask image for inpainting (grayscale bytes, can be null)
	 * @param maskImageWidth Mask image width
	 * @param maskImageHeight Mask image height
	 * @param maskImageChannels Mask image channels (typically 1 for grayscale)
	 * @return StableDiffusionResult containing the generated image or error information
	 */
	public static native StableDiffusionResult generateImageAdvanced(long handle, String prompt,
																	 String negativePrompt, int width, int height,
																	 int steps, float cfgScale, float slgScale,
																	 int seed, int sampleMethod, boolean clipOnCpu,
																	 byte[] controlImage, int controlImageWidth,
																	 int controlImageHeight, int controlImageChannels,
																	 float controlStrength, byte[] initImage,
																	 int initImageWidth, int initImageHeight,
																	 int initImageChannels, float strength,
																	 byte[] maskImage, int maskImageWidth,
																	 int maskImageHeight, int maskImageChannels);

	/**
	 * Get system information about the stable-diffusion.cpp backend.
	 *
	 * @return System information string
	 */
	public static native String getSystemInfo();

	/**
	 * Get the last error message from the native library.
	 *
	 * @return Last error message, or empty string if no error
	 */
	public static native String getLastError();

	/**
	 * Apply Canny edge detection preprocessing to an image.
	 *
	 * DISABLED: This method is currently disabled due to a memory management bug
	 * in stable-diffusion.cpp that causes double-free crashes.
	 *
	 * @param imageData Image data (RGB bytes)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @param highThreshold High threshold for edge detection
	 * @param lowThreshold Low threshold for edge detection
	 * @param weak Weak edge value (0.0-1.0)
	 * @param strong Strong edge value (0.0-1.0)
	 * @param inverse Whether to invert the edge detection result
	 * @return true if preprocessing succeeded, false otherwise
	 * @deprecated This method is disabled due to library bugs
	 */
	@Deprecated
	public static native boolean preprocessCanny(byte[] imageData, int width, int height, int channels,
												 float highThreshold, float lowThreshold,
												 float weak, float strong, boolean inverse);

	/**
	 * Convenience method to create a context with default settings.
	 *
	 * @param modelPath Path to the main Stable Diffusion model (.gguf file)
	 * @return Handle to the created context, or 0 if creation failed
	 */
	public static long createContext(String modelPath) {
		return createContext(modelPath, null, null, null, true);
	}

	/**
	 * Convenience method to create a context with text encoders.
	 *
	 * @param modelPath Path to the main Stable Diffusion model (.gguf file)
	 * @param clipLPath Path to CLIP-L text encoder
	 * @param clipGPath Path to CLIP-G text encoder
	 * @param t5xxlPath Path to T5XXL text encoder
	 * @return Handle to the created context, or 0 if creation failed
	 */
	public static long createContextWithEncoders(String modelPath, String clipLPath,
												  String clipGPath, String t5xxlPath) {
		return createContext(modelPath, clipLPath, clipGPath, t5xxlPath, true);
	}

	/**
	 * Generate an image with default parameters.
	 *
	 * @param handle Handle to the Stable Diffusion context
	 * @param prompt Text prompt describing the desired image
	 * @return StableDiffusionResult containing the generated image or error information
	 */
	public static StableDiffusionResult generateImage(long handle, String prompt) {
		return generateImage(handle, prompt, null, 768, 768, 30, 7.0f, 2.5f,
							 -1, SAMPLE_METHOD_EULER, true);
	}

	/**
	 * Generate an image with custom size.
	 *
	 * @param handle Handle to the Stable Diffusion context
	 * @param prompt Text prompt describing the desired image
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @return StableDiffusionResult containing the generated image or error information
	 */
	public static StableDiffusionResult generateImage(long handle, String prompt, int width, int height) {
		return generateImage(handle, prompt, null, width, height, 30, 7.0f, 2.5f,
							 -1, SAMPLE_METHOD_EULER, true);
	}

	/**
	 * Check if a context handle is valid (non-zero).
	 *
	 * @param handle Context handle to check
	 * @return true if handle is valid
	 */
	public static boolean isValidHandle(long handle) {
		return handle != 0;
	}

	/**
	 * Apply Canny edge detection with default parameters.
	 *
	 * @param imageData Image data (RGB bytes)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @return true if preprocessing succeeded, false otherwise
	 * @deprecated This method is disabled due to library bugs
	 */
	@Deprecated
	public static boolean preprocessCanny(byte[] imageData, int width, int height, int channels) {
		throw new UnsupportedOperationException("Canny preprocessing is disabled due to stable-diffusion.cpp library bugs");
	}

	/**
	 * Apply Canny edge detection with custom thresholds.
	 *
	 * @param imageData Image data (RGB bytes)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @param highThreshold High threshold for edge detection
	 * @param lowThreshold Low threshold for edge detection
	 * @return true if preprocessing succeeded, false otherwise
	 * @deprecated This method is disabled due to library bugs
	 */
	@Deprecated
	public static boolean preprocessCanny(byte[] imageData, int width, int height, int channels,
										  float highThreshold, float lowThreshold) {
		throw new UnsupportedOperationException("Canny preprocessing is disabled due to stable-diffusion.cpp library bugs");
	}

	/**
	 * Create a new upscaler context for image upscaling.
	 *
	 * @param esrganPath Path to the ESRGAN model file
	 * @param offloadToCpu Whether to offload parameters to CPU to save GPU memory
	 * @param direct Whether to use direct memory access
	 * @param threads Number of threads for processing
	 * @return Handle to the created upscaler context, or 0 if creation failed
	 */
	public static native long createUpscalerContext(String esrganPath, boolean offloadToCpu,
													boolean direct, int threads);

	/**
	 * Destroy an upscaler context and free its resources.
	 *
	 * @param handle Handle to the upscaler context to destroy
	 * @return true if destruction was successful
	 */
	public static native boolean destroyUpscalerContext(long handle);

	/**
	 * Upscale an image using the specified upscaler context.
	 *
	 * @param handle Handle to the upscaler context
	 * @param imageData Input image data (RGB bytes)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @param upscaleFactor Upscaling factor (2, 4, etc.)
	 * @return UpscaleResult containing the upscaled image or error information
	 */
	public static native UpscaleResult upscaleImage(long handle, byte[] imageData,
													int width, int height, int channels,
													int upscaleFactor);

	/**
	 * Get the default sampling method name.
	 *
	 * @param sampleMethod Sample method constant
	 * @return Human-readable name of the sampling method
	 */
	public static String getSampleMethodName(int sampleMethod) {
		switch (sampleMethod) {
			case SAMPLE_METHOD_EULER: return "Euler";
			case SAMPLE_METHOD_HEUN: return "Heun";
			case SAMPLE_METHOD_DPM2: return "DPM2";
			case SAMPLE_METHOD_DPMPP2S_A: return "DPM++2S a";
			case SAMPLE_METHOD_DPMPP2M: return "DPM++2M";
			case SAMPLE_METHOD_DPMPP2Mv2: return "DPM++2M v2";
			case SAMPLE_METHOD_EULER_A: return "Euler Ancestral";
			default: return "Unknown";
		}
	}
}
