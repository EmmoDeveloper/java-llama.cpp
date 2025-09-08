package de.kherud.llama;

import org.junit.Test;
import org.junit.Ignore;

/**
 * Integration test demonstrating the usage of WorkloadOptimizer factory methods.
 * This test shows how to use the workload-specific factory methods for different use cases.
 */
public class WorkloadOptimizerIntegrationTest {

	@Test
	@Ignore // Requires model file to be present
	public void testWorkloadOptimizerFactoryMethods() {
		System.out.println("\n=== WorkloadOptimizer Factory Methods Demo ===");
		
		// Base model parameters
		ModelParameters baseParams = new ModelParameters()
			.setCtxSize(512)
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(43);
		
		// Test completion-optimized model
		System.out.println("\n1. Creating completion-optimized model...");
		try (LlamaModel completionModel = LlamaModel.forCompletion(baseParams)) {
			System.out.println("✅ Completion model created successfully");
			// The model is automatically optimized with:
			// - Balanced threading for low-latency generation
			// - Moderate batch size (256) for latency/throughput balance
			// - Continuous batching enabled for better throughput
		}
		
		// Test embedding-optimized model
		System.out.println("\n2. Creating embedding-optimized model...");
		try (LlamaModel embeddingModel = LlamaModel.forEmbedding(baseParams)) {
			System.out.println("✅ Embedding model created successfully");
			// The model is automatically optimized with:
			// - High threading for compute-intensive operations
			// - Large batch size (1024) for high-throughput processing
			// - Embedding mode enabled
			// - Mean pooling for better quality
		}
		
		// Test reranking-optimized model
		System.out.println("\n3. Creating reranking-optimized model...");
		try (LlamaModel rerankingModel = LlamaModel.forReranking(baseParams)) {
			System.out.println("✅ Reranking model created successfully");
			// The model is automatically optimized with:
			// - Optimized threading for high-parallelism document processing
			// - Medium batch size (512) for document processing balance
			// - Reranking mode enabled
			// - Rank pooling for score computation
		}
		
		System.out.println("\n✅ All WorkloadOptimizer factory methods tested successfully!");
	}
	
	@Test
	public void testWorkloadOptimizerUtilityMethods() {
		System.out.println("\n=== WorkloadOptimizer Utility Methods Demo ===");
		
		// Print threading recommendations
		WorkloadOptimizer.printThreadingRecommendations();
		
		// Test manual optimization
		ModelParameters baseParams = new ModelParameters();
		
		System.out.println("Manual optimization examples:");
		
		ModelParameters completionOptimized = WorkloadOptimizer.optimizeForCompletion(baseParams);
		System.out.println("✅ Completion optimization applied");
		
		ModelParameters embeddingOptimized = WorkloadOptimizer.optimizeForEmbedding(baseParams);
		System.out.println("✅ Embedding optimization applied");
		
		ModelParameters rerankingOptimized = WorkloadOptimizer.optimizeForReranking(baseParams);
		System.out.println("✅ Reranking optimization applied");
		
		// Test generic workload optimization
		ModelParameters genericOptimized = WorkloadOptimizer.optimizeForWorkload(
			baseParams, WorkloadOptimizer.WorkloadType.GENERAL);
		System.out.println("✅ Generic workload optimization applied");
		
		System.out.println("\n✅ All WorkloadOptimizer utility methods tested successfully!");
	}
	
	@Test
	public void testThreadingConfigurationUtils() {
		System.out.println("\n=== Threading Configuration Demo ===");
		
		// Create default threading profiles
		ThreadingConfigUtils.createDefaultProfiles();
		
		// List available profiles
		ThreadingConfigUtils.printAllProfiles();
		
		// Test profile application
		ModelParameters params = new ModelParameters();
		params = ThreadingConfigUtils.applyThreadingProfile(params, "balanced");
		System.out.println("✅ Threading profile applied");
		
		// Print specific profile details
		ThreadingConfigUtils.printProfile("high-performance");
		
		System.out.println("✅ Threading configuration utilities tested successfully!");
	}
}