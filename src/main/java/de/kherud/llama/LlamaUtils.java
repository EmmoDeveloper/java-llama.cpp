package de.kherud.llama;

/**
 * Utility class providing system information, capabilities, and performance monitoring for llama.cpp.
 * These functions help with deployment optimization, debugging, and performance monitoring.
 */
public class LlamaUtils {
	static {
		// Load the native library
		LlamaLoader.initialize();
	}

	private LlamaUtils() {
	}

	/**
	 * Check if GPU offload is supported on this system.
	 *
	 * @return true if GPU offload is available, false otherwise
	 */
	public static boolean supportsGpuOffload() {
		return supportsGpuOffloadNative();
	}

	/**
	 * Check if memory mapping (mmap) is supported on this system.
	 * Memory mapping can significantly improve model loading performance.
	 *
	 * @return true if mmap is supported, false otherwise
	 */
	public static boolean supportsMmap() {
		return supportsMmapNative();
	}

	/**
	 * Check if memory locking (mlock) is supported on this system.
	 * Memory locking prevents model data from being swapped to disk.
	 *
	 * @return true if mlock is supported, false otherwise
	 */
	public static boolean supportsMlock() {
		return supportsMlockNative();
	}

	/**
	 * Check if RPC (Remote Procedure Call) support is available.
	 *
	 * @return true if RPC is supported, false otherwise
	 */
	public static boolean supportsRpc() {
		return supportsRpcNative();
	}

	/**
	 * Get the maximum number of devices (GPUs) available for computation.
	 *
	 * @return maximum number of devices
	 */
	public static long maxDevices() {
		return maxDevicesNative();
	}

	/**
	 * Get the maximum number of parallel sequences supported.
	 * This helps determine optimal batch sizes for concurrent processing.
	 *
	 * @return maximum number of parallel sequences
	 */
	public static long maxParallelSequences() {
		return maxParallelSequencesNative();
	}

	/**
	 * Get detailed system information for debugging and optimization.
	 * This includes CPU, memory, and GPU information.
	 *
	 * @return system information string
	 */
	public static String printSystemInfo() {
		return printSystemInfoNative();
	}

	/**
	 * Get current time in microseconds for high-precision timing.
	 * Useful for performance benchmarking and monitoring.
	 *
	 * @return current time in microseconds
	 */
	public static long timeUs() {
		return timeUsNative();
	}

	/**
	 * Set a custom logging callback to receive llama.cpp log messages.
	 * Pass null to clear the callback and revert to default stderr logging.
	 *
	 * @param callback custom log callback, or null to clear
	 */
	public static void setLogCallback(LogCallback callback) {
		setLogCallbackNative(callback);
	}

	// Native method declarations
	private static native boolean supportsGpuOffloadNative();
	private static native boolean supportsMmapNative();
	private static native boolean supportsMlockNative();
	private static native boolean supportsRpcNative();
	private static native long maxDevicesNative();
	private static native long maxParallelSequencesNative();
	private static native String printSystemInfoNative();
	private static native long timeUsNative();
	private static native void setLogCallbackNative(LogCallback callback);

	/**
	 * Interface for custom log callbacks.
	 */
	public interface LogCallback {
		/**
		 * Called when llama.cpp emits a log message.
		 *
		 * @param level log level (GGML_LOG_LEVEL_*)
		 * @param message log message text
		 */
		void onLog(int level, String message);
	}
}
