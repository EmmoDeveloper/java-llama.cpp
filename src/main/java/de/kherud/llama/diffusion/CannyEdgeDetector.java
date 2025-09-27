package de.kherud.llama.diffusion;

/**
 * Utility class for Canny edge detection preprocessing.
 *
 * IMPORTANT: Canny edge detection is currently DISABLED due to a memory management
 * bug in the stable-diffusion.cpp library. The preprocess_canny() function takes
 * sd_image_t by value instead of by reference, causing double-free crashes.
 *
 * All methods will throw UnsupportedOperationException until the underlying
 * library is fixed. The implementation is complete and ready to be re-enabled
 * when the library issue is resolved.
 *
 * @see <a href="https://github.com/leejet/stable-diffusion.cpp/issues">Report issue</a>
 */
public final class CannyEdgeDetector {

	private CannyEdgeDetector() {
		throw new AssertionError("Utility class should not be instantiated");
	}

	private static void throwUnsupported() {
		throw new UnsupportedOperationException(
			"Canny edge detection is disabled due to a memory management bug in stable-diffusion.cpp. " +
			"The preprocess_canny() function causes double-free crashes. " +
			"This feature will be re-enabled when the underlying library is fixed."
		);
	}

	/**
	 * Default Canny parameters based on stable-diffusion.cpp defaults.
	 */
	public static final float DEFAULT_HIGH_THRESHOLD = 0.08f;
	public static final float DEFAULT_LOW_THRESHOLD = 0.08f;
	public static final float DEFAULT_WEAK = 0.8f;
	public static final float DEFAULT_STRONG = 1.0f;
	public static final boolean DEFAULT_INVERSE = false;

	/**
	 * Apply Canny edge detection with default parameters.
	 *
	 * @param imageData Image data (RGB bytes) - modified in place
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @return true if edge detection succeeded
	 * @throws UnsupportedOperationException always (feature disabled due to library bug)
	 */
	public static boolean detectEdges(byte[] imageData, int width, int height, int channels) {
		throwUnsupported();
		return false; // unreachable
	}

	/**
	 * Apply Canny edge detection with custom thresholds.
	 *
	 * @param imageData Image data (RGB bytes) - modified in place
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @param highThreshold High threshold for edge detection (0.0-1.0)
	 * @param lowThreshold Low threshold for edge detection (0.0-1.0)
	 * @return true if edge detection succeeded
	 * @throws UnsupportedOperationException always (feature disabled due to library bug)
	 */
	public static boolean detectEdges(byte[] imageData, int width, int height, int channels,
									  float highThreshold, float lowThreshold) {
		throwUnsupported();
		return false; // unreachable
	}

	/**
	 * Apply Canny edge detection with full parameter control.
	 *
	 * @param imageData Image data (RGB bytes) - modified in place
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @param highThreshold High threshold for edge detection (0.0-1.0)
	 * @param lowThreshold Low threshold for edge detection (0.0-1.0)
	 * @param weak Weak edge value (0.0-1.0)
	 * @param strong Strong edge value (0.0-1.0)
	 * @param inverse Whether to invert the edge detection result
	 * @return true if edge detection succeeded
	 * @throws UnsupportedOperationException always (feature disabled due to library bug)
	 */
	public static boolean detectEdges(byte[] imageData, int width, int height, int channels,
									  float highThreshold, float lowThreshold,
									  float weak, float strong, boolean inverse) {
		throwUnsupported();
		return false; // unreachable
	}

	/**
	 * Create a copy of image data and apply edge detection to the copy.
	 *
	 * @param originalData Original image data (not modified)
	 * @param width Image width in pixels
	 * @param height Image height in pixels
	 * @param channels Image channels (typically 3 for RGB)
	 * @return new byte array with edge detection applied, or null if failed
	 * @throws UnsupportedOperationException always (feature disabled due to library bug)
	 */
	public static byte[] detectEdgesCopy(byte[] originalData, int width, int height, int channels) {
		throwUnsupported();
		return null; // unreachable
	}

	/**
	 * Validate edge detection parameters.
	 *
	 * @param highThreshold High threshold for edge detection
	 * @param lowThreshold Low threshold for edge detection
	 * @param weak Weak edge value
	 * @param strong Strong edge value
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public static void validateParameters(float highThreshold, float lowThreshold,
										  float weak, float strong) {
		if (highThreshold < 0.0f || highThreshold > 1.0f) {
			throw new IllegalArgumentException("High threshold must be between 0.0 and 1.0");
		}
		if (lowThreshold < 0.0f || lowThreshold > 1.0f) {
			throw new IllegalArgumentException("Low threshold must be between 0.0 and 1.0");
		}
		if (weak < 0.0f || weak > 1.0f) {
			throw new IllegalArgumentException("Weak value must be between 0.0 and 1.0");
		}
		if (strong < 0.0f || strong > 1.0f) {
			throw new IllegalArgumentException("Strong value must be between 0.0 and 1.0");
		}
		if (lowThreshold > highThreshold) {
			throw new IllegalArgumentException("Low threshold cannot be higher than high threshold");
		}
	}
}