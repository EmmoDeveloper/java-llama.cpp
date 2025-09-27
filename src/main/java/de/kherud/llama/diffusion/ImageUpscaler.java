package de.kherud.llama.diffusion;

/**
 * Image upscaling functionality using ESRGAN models.
 *
 * This class provides high-level access to image upscaling capabilities
 * through the stable-diffusion.cpp library's ESRGAN implementation.
 */
public final class ImageUpscaler implements AutoCloseable {

	private final long handle;
	private volatile boolean closed = false;

	private ImageUpscaler(long handle) {
		if (handle == 0) {
			throw new IllegalStateException("Failed to create upscaler context");
		}
		this.handle = handle;
	}

	/**
	 * Create a new image upscaler with an ESRGAN model.
	 *
	 * @param esrganPath Path to the ESRGAN model file
	 * @param offloadToCpu Whether to offload parameters to CPU to save GPU memory
	 * @param direct Whether to use direct memory access
	 * @param threads Number of threads for processing
	 * @return new ImageUpscaler instance
	 * @throws IllegalArgumentException if esrganPath is null or empty
	 * @throws IllegalStateException if upscaler creation fails
	 */
	public static ImageUpscaler create(String esrganPath, boolean offloadToCpu, boolean direct, int threads) {
		if (esrganPath == null || esrganPath.trim().isEmpty()) {
			throw new IllegalArgumentException("ESRGAN model path cannot be null or empty");
		}
		if (threads <= 0) {
			throw new IllegalArgumentException("Thread count must be positive");
		}

		long handle = NativeStableDiffusion.createUpscalerContext(esrganPath, offloadToCpu, direct, threads);
		return new ImageUpscaler(handle);
	}

	/**
	 * Create a new image upscaler with default settings.
	 *
	 * @param esrganPath Path to the ESRGAN model file
	 * @return new ImageUpscaler instance
	 */
	public static ImageUpscaler create(String esrganPath) {
		return create(esrganPath, true, false, 4);
	}

	/**
	 * Upscale an image by the specified factor.
	 *
	 * @param imageData Input image data (RGB bytes)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @param upscaleFactor Upscaling factor (2, 4, etc.)
	 * @return UpscaleResult containing the upscaled image or error information
	 * @throws IllegalStateException if upscaler is closed
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public UpscaleResult upscale(byte[] imageData, int width, int height, int channels, int upscaleFactor) {
		ensureNotClosed();
		validateUpscaleParameters(imageData, width, height, channels, upscaleFactor);

		return NativeStableDiffusion.upscaleImage(handle, imageData, width, height, channels, upscaleFactor);
	}

	/**
	 * Upscale an image with default 2x factor.
	 *
	 * @param imageData Input image data (RGB bytes)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @return UpscaleResult containing the upscaled image or error information
	 */
	public UpscaleResult upscale(byte[] imageData, int width, int height, int channels) {
		return upscale(imageData, width, height, channels, 2);
	}

	private void validateUpscaleParameters(byte[] imageData, int width, int height, int channels, int upscaleFactor) {
		if (imageData == null) {
			throw new IllegalArgumentException("Image data cannot be null");
		}
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("Image dimensions must be positive");
		}
		if (channels <= 0 || channels > 4) {
			throw new IllegalArgumentException("Channels must be between 1 and 4");
		}
		if (upscaleFactor <= 1 || upscaleFactor > 8) {
			throw new IllegalArgumentException("Upscale factor must be between 2 and 8");
		}

		int expectedSize = width * height * channels;
		if (imageData.length != expectedSize) {
			throw new IllegalArgumentException(
				String.format("Image data size mismatch: expected %d bytes, got %d", expectedSize, imageData.length)
			);
		}
	}

	private void ensureNotClosed() {
		if (closed) {
			throw new IllegalStateException("ImageUpscaler has been closed");
		}
	}

	/**
	 * Check if this upscaler is valid and not closed.
	 *
	 * @return true if upscaler is valid
	 */
	public boolean isValid() {
		return !closed && handle != 0;
	}

	@Override
	public void close() {
		if (!closed && handle != 0) {
			NativeStableDiffusion.destroyUpscalerContext(handle);
			closed = true;
		}
	}
}