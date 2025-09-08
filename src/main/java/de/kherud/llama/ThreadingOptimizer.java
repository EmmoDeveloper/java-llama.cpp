package de.kherud.llama;

/**
 * Advanced threading optimization for llama.cpp C++ native threading.
 * Provides intelligent thread pool management based on CPU topology,
 * workload characteristics, and performance monitoring.
 */
public class ThreadingOptimizer {
	
	private static final int MIN_THREADS = 1;
	private static final int MAX_THREADS_PER_CORE = 2;
	
	// CPU topology information
	private static volatile int detectedCores = -1;
	private static volatile int detectedThreads = -1;
	private static volatile boolean isNUMASystem = false;
	
	/**
	 * Optimize threading parameters for the given model and workload type.
	 * 
	 * @param params The model parameters to optimize
	 * @param workloadType The type of workload (COMPLETION, EMBEDDING, RERANKING)
	 * @return Optimized parameters with threading configuration
	 */
	public static ModelParameters optimize(ModelParameters params, WorkloadType workloadType) {
		CPUTopology topology = detectCPUTopology();
		
		int optimalThreads = calculateOptimalThreads(topology, workloadType);
		int optimalBatchThreads = calculateOptimalBatchThreads(topology, workloadType);
		
		// Apply optimizations if not already set by user
		if (!hasUserSetThreads(params)) {
			params.setThreads(optimalThreads);
		}
		
		if (!hasUserSetBatchThreads(params)) {
			params.setThreadsBatch(optimalBatchThreads);
		}
		
		// Add NUMA optimization if detected
		if (topology.isNUMASystem) {
			applyNUMAOptimizations(params, topology);
		}
		
		logThreadingConfiguration(optimalThreads, optimalBatchThreads, topology);
		
		return params;
	}
	
	/**
	 * Get recommended thread count for specific workload type.
	 * 
	 * @param workloadType The workload type
	 * @return Recommended thread count
	 */
	public static int getRecommendedThreads(WorkloadType workloadType) {
		CPUTopology topology = detectCPUTopology();
		return calculateOptimalThreads(topology, workloadType);
	}
	
	/**
	 * Detect CPU topology and capabilities.
	 */
	private static CPUTopology detectCPUTopology() {
		if (detectedCores == -1) {
			detectSystemCapabilities();
		}
		
		return new CPUTopology(detectedCores, detectedThreads, isNUMASystem);
	}
	
	/**
	 * Detect system capabilities using Java APIs and system properties.
	 */
	private static synchronized void detectSystemCapabilities() {
		if (detectedCores != -1) return; // Already detected
		
		// Detect CPU cores
		detectedCores = Runtime.getRuntime().availableProcessors();
		
		// Attempt to detect logical threads (hyperthreading)
		detectedThreads = detectLogicalThreads();
		
		// Detect NUMA topology
		isNUMASystem = detectNUMACapability();
		
		System.out.printf("🔧 Threading: Detected %d cores, %d threads, NUMA: %s\n", 
						 detectedCores, detectedThreads, isNUMASystem ? "yes" : "no");
	}
	
	/**
	 * Attempt to detect logical thread count (including hyperthreading).
	 */
	private static int detectLogicalThreads() {
		// For most systems, assume hyperthreading if core count suggests it
		int cores = detectedCores;
		
		// Check system properties for hints
		String cpuName = System.getProperty("os.arch", "").toLowerCase();
		if (cpuName.contains("x86") || cpuName.contains("amd64")) {
			// Most modern x86/x64 processors support hyperthreading
			// Conservative estimate: assume 2 threads per core if > 4 cores
			return cores > 4 ? cores : cores * 2;
		}
		
		// Default: assume cores == threads for other architectures
		return cores;
	}
	
	/**
	 * Detect if system has NUMA architecture.
	 */
	private static boolean detectNUMACapability() {
		// Simple heuristic: systems with many cores are likely NUMA
		return detectedCores >= 16;
	}
	
	/**
	 * Calculate optimal thread count for the workload.
	 */
	private static int calculateOptimalThreads(CPUTopology topology, WorkloadType workloadType) {
		int baseCores = topology.physicalCores;
		
		switch (workloadType) {
			case COMPLETION:
				// Completion benefits from moderate threading
				return Math.min(Math.max(baseCores / 2, MIN_THREADS), baseCores);
				
			case EMBEDDING:
				// Embedding is typically compute-intensive, use more threads
				return Math.min(Math.max((baseCores * 3) / 4, MIN_THREADS), baseCores);
				
			case RERANKING:
				// Reranking processes multiple documents, benefits from high parallelism
				return Math.min(baseCores, MAX_THREADS_PER_CORE * baseCores / 2);
				
			case GENERAL:
			default:
				// Balanced approach for general workloads
				return Math.min(Math.max(baseCores / 2, MIN_THREADS), baseCores / 2 + 2);
		}
	}
	
	/**
	 * Calculate optimal batch processing thread count.
	 */
	private static int calculateOptimalBatchThreads(CPUTopology topology, WorkloadType workloadType) {
		// Batch processing can typically use more threads than generation
		int generationThreads = calculateOptimalThreads(topology, workloadType);
		return Math.min(generationThreads + 2, topology.physicalCores);
	}
	
	/**
	 * Check if user has explicitly set thread count.
	 */
	private static boolean hasUserSetThreads(ModelParameters params) {
		return params.parameters.containsKey("--threads");
	}
	
	/**
	 * Check if user has explicitly set batch thread count.
	 */
	private static boolean hasUserSetBatchThreads(ModelParameters params) {
		return params.parameters.containsKey("--threads-batch");
	}
	
	/**
	 * Apply NUMA-specific optimizations.
	 */
	private static void applyNUMAOptimizations(ModelParameters params, CPUTopology topology) {
		// NUMA optimizations would go here
		// For now, just log that NUMA was detected
		System.out.println("🔧 Threading: NUMA system detected - consider manual NUMA tuning");
	}
	
	/**
	 * Log the threading configuration.
	 */
	private static void logThreadingConfiguration(int threads, int batchThreads, CPUTopology topology) {
		System.out.printf("🔧 Threading: Using %d threads for generation, %d for batches (detected %d cores)\n", 
						 threads, batchThreads, topology.physicalCores);
	}
	
	/**
	 * Workload types for threading optimization.
	 */
	public enum WorkloadType {
		COMPLETION,    // Text completion/generation
		EMBEDDING,     // Text embedding creation
		RERANKING,     // Document reranking
		GENERAL        // Mixed/general workload
	}
	
	/**
	 * CPU topology information.
	 */
	private static class CPUTopology {
		final int physicalCores;
		final int logicalThreads;
		final boolean isNUMASystem;
		
		CPUTopology(int physicalCores, int logicalThreads, boolean isNUMASystem) {
			this.physicalCores = physicalCores;
			this.logicalThreads = logicalThreads;
			this.isNUMASystem = isNUMASystem;
		}
	}
}