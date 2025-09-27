package de.kherud.llama.diffusion;

/**
 * Result of an image upscaling operation.
 *
 * Contains either the upscaled image data or error information if the operation failed.
 */
public final class UpscaleResult {

	private final boolean success;
	private final byte[] imageData;
	private final int width;
	private final int height;
	private final int channels;
	private final String errorMessage;

	private UpscaleResult(boolean success, byte[] imageData, int width, int height, int channels, String errorMessage) {
		this.success = success;
		this.imageData = imageData;
		this.width = width;
		this.height = height;
		this.channels = channels;
		this.errorMessage = errorMessage;
	}

	/**
	 * Create a successful upscale result.
	 *
	 * @param imageData Upscaled image data
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels
	 * @return successful UpscaleResult
	 */
	public static UpscaleResult success(byte[] imageData, int width, int height, int channels) {
		return new UpscaleResult(true, imageData, width, height, channels, null);
	}

	/**
	 * Create a failed upscale result.
	 *
	 * @param errorMessage Error description
	 * @return failed UpscaleResult
	 */
	public static UpscaleResult failure(String errorMessage) {
		return new UpscaleResult(false, null, 0, 0, 0, errorMessage);
	}

	/**
	 * Check if the upscaling operation was successful.
	 *
	 * @return true if successful
	 */
	public boolean isSuccess() {
		return success;
	}

	/**
	 * Get the upscaled image data.
	 *
	 * @return image data bytes, or null if operation failed
	 */
	public byte[] getImageData() {
		return imageData;
	}

	/**
	 * Get the upscaled image width.
	 *
	 * @return image width in pixels, or 0 if operation failed
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * Get the upscaled image height.
	 *
	 * @return image height in pixels, or 0 if operation failed
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * Get the number of image channels.
	 *
	 * @return number of channels, or 0 if operation failed
	 */
	public int getChannels() {
		return channels;
	}

	/**
	 * Get the error message if operation failed.
	 *
	 * @return error message, or null if operation succeeded
	 */
	public String getErrorMessage() {
		return errorMessage;
	}

	/**
	 * Get the total size of image data in bytes.
	 *
	 * @return total bytes, or 0 if operation failed
	 */
	public int getDataSize() {
		return success ? width * height * channels : 0;
	}

	@Override
	public String toString() {
		if (success) {
			return String.format("UpscaleResult{success=%s, size=%dx%d, channels=%d, bytes=%d}",
				success, width, height, channels, getDataSize());
		} else {
			return String.format("UpscaleResult{success=%s, error='%s'}", success, errorMessage);
		}
	}
}