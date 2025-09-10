package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Simple smart defaults system that applies GPU acceleration by default.
 * No complex detection - just intelligent defaults that work well for most users.
 */
public class SmartDefaults {
	private static final System.Logger logger = System.getLogger(SmartDefaults.class.getName());
	/**
	 * Apply smart defaults to ModelParameters for better out-of-the-box experience.
	 * This is called automatically when creating a LlamaModel.
	 */
	public static ModelParameters apply(ModelParameters params) {

		// GPU Layer defaults - try to use GPU by default
		if (!params.parameters.containsKey("--gpu-layers")) {
			// Default to a high number of GPU layers - llama.cpp will use what's available
			int defaultGpuLayers = 999; // Let llama.cpp decide based on available VRAM
			params.setGpuLayers(defaultGpuLayers);
			logger.log(DEBUG, "🚀 Auto-enabled GPU acceleration (up to " + defaultGpuLayers + " layers)");
		}

		// Context size defaults
		if (!params.parameters.containsKey("--ctx-size")) {
			params.setCtxSize(2048); // Reasonable default for most use cases
			logger.log(DEBUG, "📝 Auto-configured context size: 2048");
		}

		// Apply intelligent batch size optimization
		params = BatchOptimizer.applyBatchOptimization(params);

		// Validate any user-provided batch configurations
		BatchOptimizer.validateBatchConfiguration(params);

		// Apply memory optimizations
		params = MemoryOptimizer.optimizeMemoryUsage(params);

		// Check memory health
		MemoryOptimizer.checkMemoryHealth();

		// Apply intelligent threading optimization
		params = ThreadingOptimizer.optimize(params, ThreadingOptimizer.WorkloadType.GENERAL);

		// Enable Flash Attention by default if not specified
		if (!params.parameters.containsKey("--flash-attn") && !params.parameters.containsKey("--no-flash-attn")) {
			params.enableFlashAttn();
			logger.log(DEBUG, "⚡ Auto-enabled Flash Attention");
		}

		return params;
	}
}
