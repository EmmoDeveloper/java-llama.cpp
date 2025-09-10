package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Intelligent batch size optimization based on GPU memory and model characteristics.
 * Analyzes system capabilities to determine optimal batch and ubatch sizes for maximum throughput.
 */
public class BatchOptimizer {
	private static final System.Logger logger = System.getLogger(BatchOptimizer.class.getName());
	// Based on empirical testing and llama.cpp recommendations
	private static final int MIN_BATCH_SIZE = 128;
	private static final int MAX_BATCH_SIZE = 2048;
	private static final int DEFAULT_BATCH_SIZE = 512;

	/**
	 * Calculate optimal batch size based on context size and GPU memory.
	 */
	public static BatchConfiguration optimizeForGpu(int contextSize, boolean hasGpu) {
		if (!hasGpu) {
			// CPU-only optimization: smaller batches to avoid memory pressure
			return new BatchConfiguration(
				Math.min(256, contextSize / 4),  // Conservative for CPU
				128,  // Small ubatch for CPU
				"CPU-optimized configuration"
			);
		}

		// GPU optimization based on context size and memory efficiency
		int optimalBatch = calculateOptimalBatchSize(contextSize);
		int optimalUbatch = calculateOptimalUbatchSize(optimalBatch);

		return new BatchConfiguration(
			optimalBatch,
			optimalUbatch,
			"GPU-optimized for context size " + contextSize
		);
	}

	/**
	 * Calculate optimal batch size based on context size.
	 * Larger contexts can benefit from larger batches, but with diminishing returns.
	 */
	private static int calculateOptimalBatchSize(int contextSize) {
		if (contextSize <= 512) {
			return 256; // Small contexts work well with medium batches
		} else if (contextSize <= 1024) {
			return 512; // Standard contexts benefit from standard batches
		} else if (contextSize <= 2048) {
			return 1024; // Large contexts can handle larger batches
		} else {
			return 1536; // Very large contexts need even larger batches for efficiency
		}
	}

	/**
	 * Calculate optimal ubatch size based on batch size.
	 * ubatch should be <= batch size and optimized for GPU memory transfer.
	 */
	private static int calculateOptimalUbatchSize(int batchSize) {
		// ubatch should be a reasonable fraction of batch size
		// but not so small that it causes inefficient GPU utilization
		if (batchSize <= 256) {
			return batchSize; // For small batches, ubatch = batch
		} else if (batchSize <= 512) {
			return 256; // Sweet spot for most GPUs
		} else if (batchSize <= 1024) {
			return 512; // Larger ubatch for larger batches
		} else {
			return 512; // Cap ubatch at 512 for memory efficiency
		}
	}

	/**
	 * Apply batch optimizations to ModelParameters.
	 * Only optimizes if user hasn't explicitly set batch parameters.
	 */
	public static ModelParameters applyBatchOptimization(ModelParameters params) {
		boolean hasBatchConfig = params.parameters.containsKey("--batch-size");
		boolean hasUbatchConfig = params.parameters.containsKey("--ubatch-size");

		if (hasBatchConfig && hasUbatchConfig) {
			// User has configured both - don't override
			return params;
		}

		// Determine context size (default to 2048 if not set)
		int contextSize = 2048;
		if (params.parameters.containsKey("--ctx-size")) {
			contextSize = Integer.parseInt(params.parameters.get("--ctx-size"));
		}

		// Check if GPU is likely being used
		boolean hasGpu = params.parameters.containsKey("--gpu-layers") &&
		                 !params.parameters.get("--gpu-layers").equals("0");

		// Get optimal configuration
		BatchConfiguration config = optimizeForGpu(contextSize, hasGpu);

		// Apply optimizations only where user hasn't specified
		if (!hasBatchConfig) {
			params.setBatchSize(config.batchSize);
			logger.log(DEBUG, "🚀 Auto-optimized batch size: " + config.batchSize +
				" (" + config.description + ")");
		}

		if (!hasUbatchConfig) {
			params.setUbatchSize(config.ubatchSize);
			logger.log(DEBUG, "⚡ Auto-optimized ubatch size: " + config.ubatchSize);
		}

		return params;
	}

	/**
	 * Validate and suggest corrections for user-provided batch configurations.
	 */
	public static void validateBatchConfiguration(ModelParameters params) {
		String batchStr = params.parameters.get("--batch-size");
		String ubatchStr = params.parameters.get("--ubatch-size");

		if (batchStr != null && ubatchStr != null) {
			int batchSize = Integer.parseInt(batchStr);
			int ubatchSize = Integer.parseInt(ubatchStr);

			// Validate ubatch <= batch
			if (ubatchSize > batchSize) {
				logger.log(DEBUG, "⚠️  Warning: ubatch size (" + ubatchSize +
					") is larger than batch size (" + batchSize + ")");
				logger.log(DEBUG, "   Consider reducing ubatch size to " + batchSize + " or less");
			}

			// Suggest optimizations for very small or very large batch sizes
			if (batchSize < MIN_BATCH_SIZE) {
				logger.log(DEBUG, "💡 Tip: Batch size " + batchSize + " may be too small for optimal throughput");
				logger.log(DEBUG, "   Consider increasing to " + MIN_BATCH_SIZE + " or higher");
			} else if (batchSize > MAX_BATCH_SIZE) {
				logger.log(DEBUG, "💡 Tip: Batch size " + batchSize + " may cause memory issues");
				logger.log(DEBUG, "   Consider reducing to " + MAX_BATCH_SIZE + " or lower");
			}
		}
	}

	public static class BatchConfiguration {
		public final int batchSize;
		public final int ubatchSize;
		public final String description;

		public BatchConfiguration(int batchSize, int ubatchSize, String description) {
			this.batchSize = batchSize;
			this.ubatchSize = ubatchSize;
			this.description = description;
		}

		@Override
		public String toString() {
			return String.format("Batch: %d, UBatch: %d (%s)",
				batchSize, ubatchSize, description);
		}
	}
}
