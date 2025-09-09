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

	/**
	 * Build a split file path for multipart model files.
	 * Useful for handling models split across multiple files.
	 *
	 * @param basePath the base path for the model
	 * @param splitIndex the split file index (0, 1, 2, etc.)
	 * @return the complete path for the specified split file
	 */
	public static String buildSplitPath(String basePath, int splitIndex) {
		return splitPathNative(basePath, splitIndex);
	}

	/**
	 * Initialize the llama.cpp backend globally.
	 * This should be called once before creating any models.
	 * Thread-safe and can be called multiple times.
	 */
	public static void initializeBackend() {
		initializeBackendNative();
	}

	/**
	 * Free the llama.cpp backend resources globally.
	 * This should be called when shutting down the application.
	 * After this call, no llama.cpp functions should be used.
	 */
	public static void freeBackend() {
		freeBackendNative();
	}

	/**
	 * Initialize NUMA optimizations for multi-socket systems.
	 * This improves performance on systems with multiple CPU sockets.
	 *
	 * @param strategy the NUMA strategy (0=disabled, 1=distribute, 2=isolate, 3=numactl)
	 */
	public static void initializeNuma(int strategy) {
		initializeNumaNative(strategy);
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
	private static native String splitPathNative(String path, int split);
	
	// Tier 5: Backend management native methods
	private static native void initializeBackendNative();
	private static native void freeBackendNative();
	private static native void initializeNumaNative(int strategy);

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
