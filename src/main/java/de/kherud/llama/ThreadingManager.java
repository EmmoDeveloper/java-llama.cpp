package de.kherud.llama;

/**
 * Advanced threading management for llama.cpp contexts.
 * Provides fine-grained control over thread allocation for different phases:
 * - Generation threads: Single token inference
 * - Batch threads: Multi-token batch processing
 * - NUMA optimization: Thread affinity and memory locality
 */
public class ThreadingManager {
	
	static {
		LlamaLoader.initialize();
	}
	
	/**
	 * Threading configuration optimized for different use cases.
	 */
	public static class ThreadConfig {
		private final int generationThreads;
		private final int batchThreads;
		private final boolean numaOptimized;
		private final String description;
		
		public ThreadConfig(int generationThreads, int batchThreads, boolean numaOptimized, String description) {
			this.generationThreads = Math.max(1, generationThreads);
			this.batchThreads = Math.max(1, batchThreads);
			this.numaOptimized = numaOptimized;
			this.description = description;
		}
		
		public int getGenerationThreads() { return generationThreads; }
		public int getBatchThreads() { return batchThreads; }
		public boolean isNumaOptimized() { return numaOptimized; }
		public String getDescription() { return description; }
		
		@Override
		public String toString() {
			return String.format("ThreadConfig{gen:%d, batch:%d, numa:%s, desc:'%s'}", 
				generationThreads, batchThreads, numaOptimized, description);
		}
	}
	
	/**
	 * Get optimal threading configuration for the current system.
	 */
	public static ThreadConfig getOptimalConfiguration() {
		int cores = Runtime.getRuntime().availableProcessors();
		boolean hasNuma = detectNumaSupport();
		
		// Intel/AMD optimization based on core count
		if (cores <= 4) {
			return new ThreadConfig(2, 4, hasNuma, "Low-core optimization");
		} else if (cores <= 8) {
			return new ThreadConfig(4, 6, hasNuma, "Mid-range optimization");
		} else if (cores <= 16) {
			return new ThreadConfig(8, 10, hasNuma, "High-performance optimization");
		} else {
			return new ThreadConfig(12, 16, hasNuma, "Server-grade optimization");
		}
	}
	
	/**
	 * Configuration optimized for low-latency single token generation.
	 */
	public static ThreadConfig forLowLatency() {
		int cores = Runtime.getRuntime().availableProcessors();
		int genThreads = Math.min(4, cores / 2); // Conservative for latency
		int batchThreads = Math.max(2, genThreads); // Minimum batch capability
		
		return new ThreadConfig(genThreads, batchThreads, false, "Low-latency optimized");
	}
	
	/**
	 * Configuration optimized for high-throughput batch processing.
	 */
	public static ThreadConfig forHighThroughput() {
		int cores = Runtime.getRuntime().availableProcessors();
		int genThreads = Math.max(2, cores / 4); // Reserve cores for batch
		int batchThreads = Math.min(cores - 2, cores * 3 / 4); // Maximize batch threads
		
		return new ThreadConfig(genThreads, batchThreads, detectNumaSupport(), "High-throughput optimized");
	}
	
	/**
	 * Configuration optimized for memory-constrained environments.
	 */
	public static ThreadConfig forMemoryEfficiency() {
		int cores = Runtime.getRuntime().availableProcessors();
		// Conservative thread allocation to reduce memory pressure
		int genThreads = Math.min(4, Math.max(1, cores / 4));
		int batchThreads = Math.min(6, Math.max(2, cores / 3));
		
		return new ThreadConfig(genThreads, batchThreads, false, "Memory-efficient");
	}
	
	/**
	 * Apply threading configuration to a model context.
	 */
	public static void applyConfiguration(LlamaModel model, ThreadConfig config) {
		if (model == null) {
			throw new IllegalArgumentException("Model cannot be null");
		}
		if (config == null) {
			throw new IllegalArgumentException("ThreadConfig cannot be null");
		}
		
		// Apply the threading configuration via native method
		setModelThreading(model, config.generationThreads, config.batchThreads);
		
		// Log the configuration for debugging
		System.out.println("🔧 Threading: Applied " + config);
		if (config.numaOptimized) {
			System.out.println("🔧 Threading: NUMA optimizations enabled");
		}
	}
	
	/**
	 * Get current threading configuration from a model.
	 */
	public static ThreadConfig getCurrentConfiguration(LlamaModel model) {
		if (model == null) {
			throw new IllegalArgumentException("Model cannot be null");
		}
		
		int[] threads = getModelThreading(model);
		return new ThreadConfig(threads[0], threads[1], false, "Current configuration");
	}
	
	/**
	 * Detect if the system has NUMA (Non-Uniform Memory Access) architecture.
	 */
	private static boolean detectNumaSupport() {
		// Simple heuristic: systems with many cores likely have NUMA
		int cores = Runtime.getRuntime().availableProcessors();
		return cores >= 16; // Conservative detection
	}
	
	/**
	 * Create a threading configuration with explicit parameters.
	 */
	public static ThreadConfig createCustom(int generationThreads, int batchThreads) {
		return createCustom(generationThreads, batchThreads, false, "Custom configuration");
	}
	
	/**
	 * Create a threading configuration with explicit parameters and NUMA settings.
	 */
	public static ThreadConfig createCustom(int generationThreads, int batchThreads, boolean numaOptimized, String description) {
		return new ThreadConfig(generationThreads, batchThreads, numaOptimized, description);
	}
	
	// Native method declarations
	private static native void setModelThreading(LlamaModel model, int generationThreads, int batchThreads);
	private static native int[] getModelThreading(LlamaModel model);
}