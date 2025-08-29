package de.kherud.llama;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;

/**
 * Memory optimization utilities for better memory allocation patterns and resource management.
 * Provides intelligent memory management strategies based on system capabilities.
 */
public class MemoryOptimizer {
	
	private static final int MB = 1024 * 1024;
	private static final int MINIMUM_HEAP_MB = 512;
	private static final int RECOMMENDED_HEAP_MB = 2048;
	
	/**
	 * Optimize ModelParameters for memory efficiency based on available system memory.
	 */
	public static ModelParameters optimizeMemoryUsage(ModelParameters params) {
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		
		long availableMemoryMB = heapUsage.getMax() / MB;
		long usedMemoryMB = heapUsage.getUsed() / MB;
		long freeMemoryMB = availableMemoryMB - usedMemoryMB;
		
		System.out.printf("💾 Memory Status: %dMB free / %dMB total heap\n", 
			freeMemoryMB, availableMemoryMB);
		
		// Apply memory-conscious optimizations
		params = optimizeContextSize(params, freeMemoryMB);
		params = optimizeBatchSizes(params, freeMemoryMB);
		params = optimizeThreading(params, freeMemoryMB);
		
		// Add memory monitoring warnings
		if (freeMemoryMB < MINIMUM_HEAP_MB) {
			System.out.println("⚠️  Low memory warning: Consider increasing JVM heap size (-Xmx)");
		}
		
		return params;
	}
	
	/**
	 * Optimize context size based on available memory to prevent OOM errors.
	 */
	private static ModelParameters optimizeContextSize(ModelParameters params, long freeMemoryMB) {
		if (params.parameters.containsKey("--ctx-size")) {
			int currentCtxSize = Integer.parseInt(params.parameters.get("--ctx-size"));
			int recommendedCtxSize = calculateOptimalContextSize(freeMemoryMB);
			
			if (currentCtxSize > recommendedCtxSize) {
				System.out.printf("🔧 Memory optimization: Reducing context size from %d to %d for available memory (%dMB)\n", 
					currentCtxSize, recommendedCtxSize, freeMemoryMB);
				params.setCtxSize(recommendedCtxSize);
			}
		}
		return params;
	}
	
	/**
	 * Calculate optimal context size based on available memory.
	 * Context memory usage scales roughly linearly with context size.
	 */
	private static int calculateOptimalContextSize(long freeMemoryMB) {
		if (freeMemoryMB >= 4096) return 4096;  // 4GB+ -> large contexts
		if (freeMemoryMB >= 2048) return 2048;  // 2GB+ -> medium contexts
		if (freeMemoryMB >= 1024) return 1024;  // 1GB+ -> small contexts
		if (freeMemoryMB >= 512) return 512;    // 512MB+ -> minimal contexts
		return 256; // Very low memory -> tiny contexts
	}
	
	/**
	 * Optimize batch sizes for memory efficiency without sacrificing too much performance.
	 */
	private static ModelParameters optimizeBatchSizes(ModelParameters params, long freeMemoryMB) {
		// Conservative batch sizes for low memory situations
		if (freeMemoryMB < MINIMUM_HEAP_MB) {
			if (!params.parameters.containsKey("--batch-size")) {
				params.setBatchSize(128); // Small batch for low memory
				System.out.println("🔧 Memory optimization: Using small batch size (128) for low memory");
			}
			if (!params.parameters.containsKey("--ubatch-size")) {
				params.setUbatchSize(64); // Very small ubatch
				System.out.println("🔧 Memory optimization: Using small ubatch size (64) for low memory");
			}
		}
		return params;
	}
	
	/**
	 * Optimize threading based on memory constraints.
	 */
	private static ModelParameters optimizeThreading(ModelParameters params, long freeMemoryMB) {
		if (!params.parameters.containsKey("--threads")) {
			// Use fewer threads in low memory situations to reduce memory pressure
			int threads = freeMemoryMB < MINIMUM_HEAP_MB ? 2 : 4;
			params.setThreads(threads);
			System.out.printf("🔧 Memory optimization: Using %d threads for memory efficiency\n", threads);
		}
		return params;
	}
	
	/**
	 * Provide memory usage recommendations for model configuration.
	 */
	public static void printMemoryRecommendations() {
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		
		long maxHeapMB = heapUsage.getMax() / MB;
		long usedHeapMB = heapUsage.getUsed() / MB;
		
		System.out.println("\n💡 Memory Usage Recommendations:");
		System.out.printf("   Current heap: %dMB used / %dMB max\n", usedHeapMB, maxHeapMB);
		
		if (maxHeapMB < MINIMUM_HEAP_MB) {
			System.out.println("   🔴 Critical: Increase JVM heap size (-Xmx1g or higher)");
		} else if (maxHeapMB < RECOMMENDED_HEAP_MB) {
			System.out.println("   🟡 Recommended: Consider increasing heap size for better performance (-Xmx2g)");
		} else {
			System.out.println("   🟢 Good: Heap size is adequate for most operations");
		}
		
		System.out.println("   💾 For large models, consider:");
		System.out.println("      - Using GPU offloading to reduce CPU memory usage");
		System.out.println("      - Smaller context sizes (1024-2048) for memory efficiency");
		System.out.println("      - Closing unused model instances promptly");
	}
	
	/**
	 * Monitor memory usage and provide warnings for potential issues.
	 */
	public static void checkMemoryHealth() {
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		
		double memoryUsageRatio = (double) heapUsage.getUsed() / heapUsage.getMax();
		
		if (memoryUsageRatio > 0.9) {
			System.out.println("🚨 Memory Alert: Heap usage > 90% - consider reducing model parameters or increasing heap size");
		} else if (memoryUsageRatio > 0.8) {
			System.out.println("⚠️  Memory Warning: Heap usage > 80% - monitor for potential memory issues");
		}
		
		// Force GC if memory usage is high
		if (memoryUsageRatio > 0.85) {
			System.out.println("🧹 Running garbage collection to free memory...");
			System.gc();
		}
	}
	
	/**
	 * Create a memory-efficient ModelParameters configuration for resource-constrained environments.
	 */
	public static ModelParameters createMemoryEfficientConfig(String modelPath) {
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		long availableMemoryMB = memoryBean.getHeapMemoryUsage().getMax() / MB;
		
		ModelParameters params = new ModelParameters().setModel(modelPath);
		
		// Conservative settings for memory efficiency
		if (availableMemoryMB < MINIMUM_HEAP_MB) {
			// Ultra-low memory configuration
			params.setCtxSize(256)
				  .setBatchSize(64)
				  .setUbatchSize(32)
				  .setThreads(2)
				  .setGpuLayers(999); // Offload to GPU to save CPU memory
			System.out.println("🔧 Created ultra-low memory configuration");
		} else if (availableMemoryMB < RECOMMENDED_HEAP_MB) {
			// Low memory configuration
			params.setCtxSize(512)
				  .setBatchSize(128)
				  .setUbatchSize(64)
				  .setThreads(3)
				  .setGpuLayers(999);
			System.out.println("🔧 Created low memory configuration");
		} else {
			// Standard efficient configuration
			params.setCtxSize(1024)
				  .setBatchSize(256)
				  .setUbatchSize(128)
				  .setGpuLayers(999);
			System.out.println("🔧 Created standard efficient configuration");
		}
		
		return params;
	}
}