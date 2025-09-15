package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Ignore;

/**
 * Integration test demonstrating the usage of WorkloadOptimizer factory methods.
 * This test shows how to use the workload-specific factory methods for different use cases.
 */
public class WorkloadOptimizerIntegrationTest {
	private static final System.Logger logger = System.getLogger(WorkloadOptimizerIntegrationTest.class.getName());

	@Test
	@Ignore // Requires model file to be present
	public void testWorkloadOptimizerFactoryMethods() {
		logger.log(DEBUG, "\n=== WorkloadOptimizer Factory Methods Demo ===");

		// Base model parameters
		ModelParameters baseParams = new ModelParameters()
			.setCtxSize(512)
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(43);

		// Test completion-optimized model
		logger.log(DEBUG, "\n1. Creating completion-optimized model...");
		try (LlamaModel completionModel = LlamaModel.forCompletion(baseParams)) {
			logger.log(DEBUG, "✅ Completion model created successfully");
			// The model is automatically optimized with:
			// - Balanced threading for low-latency generation
			// - Moderate batch size (256) for latency/throughput balance
			// - Continuous batching enabled for better throughput
		}

		// Test embedding-optimized model
		logger.log(DEBUG, "\n2. Creating embedding-optimized model...");
		try (LlamaModel embeddingModel = LlamaModel.forEmbedding(baseParams)) {
			logger.log(DEBUG, "✅ Embedding model created successfully");
			// The model is automatically optimized with:
			// - High threading for compute-intensive operations
			// - Large batch size (1024) for high-throughput processing
			// - Embedding mode enabled
			// - Mean pooling for better quality
		}

		// Test reranking-optimized model
		logger.log(DEBUG, "\n3. Creating reranking-optimized model...");
		try (LlamaModel rerankingModel = LlamaModel.forReranking(baseParams)) {
			logger.log(DEBUG, "✅ Reranking model created successfully");
			// The model is automatically optimized with:
			// - Optimized threading for high-parallelism document processing
			// - Medium batch size (512) for document processing balance
			// - Reranking mode enabled
			// - Rank pooling for score computation
		}

		logger.log(DEBUG, "\n✅ All WorkloadOptimizer factory methods tested successfully!");
	}

	@Test
	public void testWorkloadOptimizerUtilityMethods() {
		logger.log(DEBUG, "\n=== WorkloadOptimizer Utility Methods Demo ===");

		// Print threading recommendations
		WorkloadOptimizer.printThreadingRecommendations();

		// Test manual optimization
		ModelParameters baseParams = new ModelParameters();

		logger.log(DEBUG, "Manual optimization examples:");

		ModelParameters completionOptimized = WorkloadOptimizer.optimizeForCompletion(baseParams);
		logger.log(DEBUG, "✅ Completion optimization applied");

		ModelParameters embeddingOptimized = WorkloadOptimizer.optimizeForEmbedding(baseParams);
		logger.log(DEBUG, "✅ Embedding optimization applied");

		ModelParameters rerankingOptimized = WorkloadOptimizer.optimizeForReranking(baseParams);
		logger.log(DEBUG, "✅ Reranking optimization applied");

		// Test generic workload optimization
		ModelParameters genericOptimized = WorkloadOptimizer.optimizeForWorkload(
			baseParams, WorkloadOptimizer.WorkloadType.GENERAL);
		logger.log(DEBUG, "✅ Generic workload optimization applied");

		logger.log(DEBUG, "\n✅ All WorkloadOptimizer utility methods tested successfully!");
	}

	@Test
	public void testThreadingConfigurationUtils() {
		logger.log(DEBUG, "\n=== Threading Configuration Demo ===");

		// Create default threading profiles
		ThreadingConfigUtils.createDefaultProfiles();

		// List available profiles
		ThreadingConfigUtils.printAllProfiles();

		// Test profile application
		ModelParameters params = new ModelParameters();
		params = ThreadingConfigUtils.applyThreadingProfile(params, "balanced");
		logger.log(DEBUG, "✅ Threading profile applied");

		// Print specific profile details
		ThreadingConfigUtils.printProfile("high-performance");

		logger.log(DEBUG, "✅ Threading configuration utilities tested successfully!");
	}
}
