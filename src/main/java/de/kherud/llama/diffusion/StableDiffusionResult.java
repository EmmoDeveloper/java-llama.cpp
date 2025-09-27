package de.kherud.llama.diffusion;

import java.util.Optional;

/**
 * Result of a Stable Diffusion image generation operation.
 */
public class StableDiffusionResult {
	private final boolean success;
	private final String errorMessage;
	private final byte[] imageData;
	private final int width;
	private final int height;
	private final float generationTime;

	public StableDiffusionResult(boolean success, String errorMessage, byte[] imageData,
								 int width, int height, float generationTime) {
		this.success = success;
		this.errorMessage = errorMessage;
		this.imageData = imageData;
		this.width = width;
		this.height = height;
		this.generationTime = generationTime;
	}

	/**
	 * @return true if image generation was successful
	 */
	public boolean isSuccess() {
		return success;
	}

	/**
	 * @return error message if generation failed, empty if successful
	 */
	public Optional<String> getErrorMessage() {
		return Optional.ofNullable(errorMessage);
	}

	/**
	 * @return raw image data (RGB format) if generation was successful
	 */
	public Optional<byte[]> getImageData() {
		return Optional.ofNullable(imageData);
	}

	/**
	 * @return image width in pixels
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * @return image height in pixels
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * @return generation time in seconds
	 */
	public float getGenerationTime() {
		return generationTime;
	}

	/**
	 * @return size of image data in bytes (width * height * channels)
	 */
	public int getImageDataSize() {
		return imageData != null ? imageData.length : 0;
	}

	/**
	 * @return number of color channels (usually 3 for RGB)
	 */
	public int getChannels() {
		if (imageData == null || width == 0 || height == 0) {
			return 0;
		}
		return imageData.length / (width * height);
	}

	@Override
	public String toString() {
		if (success) {
			return String.format("StableDiffusionResult{success=true, size=%dx%d, time=%.2fs, channels=%d}",
				width, height, generationTime, getChannels());
		} else {
			return String.format("StableDiffusionResult{success=false, error='%s'}",
				errorMessage != null ? errorMessage : "Unknown error");
		}
	}
}