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

	/**
	 * Extract the path prefix from a split model file path.
	 * Important for model file management and validation.
	 *
	 * @param path the split model file path
	 * @return the extracted prefix, or empty string if extraction fails
	 */
	public static String extractSplitPrefix(String path) {
		return extractSplitPrefixNative(path);
	}

	/**
	 * Get the default model loading parameters as JSON.
	 * Provides baseline configuration for consistent model initialization.
	 *
	 * @return JSON string containing default model parameters
	 */
	public static String getModelDefaultParams() {
		return getModelDefaultParamsNative();
	}

	/**
	 * Get the default context parameters as JSON.
	 * Provides baseline configuration for context initialization.
	 *
	 * @return JSON string containing default context parameters
	 */
	public static String getContextDefaultParams() {
		return getContextDefaultParamsNative();
	}

	/**
	 * Get the default sampler chain parameters as JSON.
	 * Provides baseline configuration for sampler initialization.
	 *
	 * @return JSON string containing default sampler chain parameters
	 */
	public static String getSamplerChainDefaultParams() {
		return getSamplerChainDefaultParamsNative();
	}

	/**
	 * Get the default quantization parameters as JSON.
	 * Provides baseline configuration for model quantization.
	 *
	 * @return JSON string containing default quantization parameters
	 */
	public static String getQuantizationDefaultParams() {
		return getQuantizationDefaultParamsNative();
	}

	/**
	 * Get the name of a flash attention type.
	 * Provides human-readable description of attention optimization.
	 *
	 * @param flashAttnType the flash attention type ID
	 * @return the name of the flash attention type
	 */
	public static String getFlashAttentionTypeName(int flashAttnType) {
		return getFlashAttentionTypeNameNative(flashAttnType);
	}

	/**
	 * Get the list of built-in chat templates.
	 * Provides available templates for conversation formatting.
	 *
	 * @return array of built-in chat template names
	 */
	public static String[] getChatBuiltinTemplates() {
		return getChatBuiltinTemplatesNative();
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
	
	// Tier 6: Advanced debugging & production management native methods
	private static native String extractSplitPrefixNative(String path);
	
	// Tier 7: Complete utility mastery native methods
	private static native String getModelDefaultParamsNative();
	private static native String getContextDefaultParamsNative();
	private static native String getSamplerChainDefaultParamsNative();
	private static native String getQuantizationDefaultParamsNative();
	private static native String getFlashAttentionTypeNameNative(int flashAttnType);
	private static native String[] getChatBuiltinTemplatesNative();

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
