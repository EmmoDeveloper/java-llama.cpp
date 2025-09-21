package de.kherud.llama;

import de.kherud.llama.args.PoolingType;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Workload-specific optimization utilities for different LLM operations.
 * Provides optimized configurations for completion, embedding, and reranking workloads.
 *
 * <p><b>Usage Examples:</b></p>
 * <pre>
 * // Direct usage with factory methods (recommended)
 * ModelParameters params = new ModelParameters().setModel("path/to/model.gguf");
 * LlamaModel completionModel = LlamaModel.forCompletion(params);
 * LlamaModel embeddingModel = LlamaModel.forEmbedding(params);
 * LlamaModel rerankingModel = LlamaModel.forReranking(params);
 *
 * // Manual optimization
 * ModelParameters optimized = WorkloadOptimizer.optimizeForCompletion(params);
 * LlamaModel model = new LlamaModel(optimized);
 *
 * // Print threading recommendations
 * WorkloadOptimizer.printThreadingRecommendations();
 * </pre>
 */
public class WorkloadOptimizer {
	private static final System.Logger logger = System.getLogger(WorkloadOptimizer.class.getName());
	private WorkloadOptimizer() {
	}

	/**
	 * Optimize model parameters for text completion/generation workloads.
	 *
	 * @param baseParams Base model parameters
	 * @return Parameters optimized for completion workloads
	 */
	public static ModelParameters optimizeForCompletion(ModelParameters baseParams) {
		ModelParameters optimized = new ModelParameters();

		// Copy base parameters
		copyParameters(baseParams, optimized);

		// Apply completion-specific threading
		optimized = ThreadingOptimizer.optimize(optimized, ThreadingOptimizer.WorkloadType.COMPLETION);

		// Completion-specific optimizations
		if (!optimized.parameters.containsKey("--batch-size")) {
			// Moderate batch size for completion - balance latency vs throughput
			optimized.setBatchSize(256);
			logger.log(DEBUG, "🔧 Completion: Optimized batch size for low-latency generation");
		}

		// Enable continuous batching for better throughput if not set
		if (!optimized.parameters.containsKey("--cont-batching")) {
			optimized.parameters.put("--cont-batching", null);
			logger.log(DEBUG, "🔧 Completion: Enabled continuous batching");
		}

		return optimized;
	}

	/**
	 * Optimize model parameters for embedding generation workloads.
	 *
	 * @param baseParams Base model parameters
	 * @return Parameters optimized for embedding workloads
	 */
	public static ModelParameters optimizeForEmbedding(ModelParameters baseParams) {
		ModelParameters optimized = new ModelParameters();

		// Copy base parameters
		copyParameters(baseParams, optimized);

		// Apply embedding-specific threading
		optimized = ThreadingOptimizer.optimize(optimized, ThreadingOptimizer.WorkloadType.EMBEDDING);

		// Embedding-specific optimizations
		if (!optimized.parameters.containsKey("--batch-size")) {
			// Large batch size for embeddings - optimize for throughput
			optimized.setBatchSize(1024);
			logger.log(DEBUG, "🔧 Embedding: Optimized batch size for high-throughput processing");
		}

		// Ensure embedding mode is enabled
		if (!optimized.parameters.containsKey("--embedding")) {
			optimized.enableEmbedding();
			logger.log(DEBUG, "🔧 Embedding: Enabled embedding mode");
		}

		// Use pooling mean for better embedding quality
		if (!optimized.parameters.containsKey("--pooling")) {
			optimized.setPoolingType(PoolingType.MEAN);
			logger.log(DEBUG, "🔧 Embedding: Using mean pooling for better quality");
		}

		return optimized;
	}

	/**
	 * Optimize model parameters for document reranking workloads.
	 *
	 * @param baseParams Base model parameters
	 * @return Parameters optimized for reranking workloads
	 */
	public static ModelParameters optimizeForReranking(ModelParameters baseParams) {
		ModelParameters optimized = new ModelParameters();

		// Copy base parameters
		copyParameters(baseParams, optimized);

		// Apply reranking-specific threading
		optimized = ThreadingOptimizer.optimize(optimized, ThreadingOptimizer.WorkloadType.RERANKING);

		// Reranking-specific optimizations
		if (!optimized.parameters.containsKey("--batch-size")) {
			// Medium batch size for reranking - balance between individual doc processing and throughput
			optimized.setBatchSize(512);
			logger.log(DEBUG, "🔧 Reranking: Optimized batch size for document processing");
		}

		// Enable reranking mode
		if (!optimized.parameters.containsKey("--reranking")) {
			optimized.enableReranking();
			logger.log(DEBUG, "🔧 Reranking: Enabled reranking mode");
		}

		// Use appropriate pooling for reranking scores
		if (!optimized.parameters.containsKey("--pooling")) {
			optimized.setPoolingType(PoolingType.RANK);
			logger.log(DEBUG, "🔧 Reranking: Using rank pooling for score computation");
		}

		return optimized;
	}

	/**
	 * Copy parameters from source to destination.
	 */
	private static void copyParameters(ModelParameters source, ModelParameters destination) {
		if (source != null && source.parameters != null) {
			destination.parameters.putAll(source.parameters);
		}
	}

	/**
	 * Get optimized parameters for a specific workload type.
	 *
	 * @param baseParams Base model parameters
	 * @param workloadType The workload type to optimize for
	 * @return Optimized parameters
	 */
	public static ModelParameters optimizeForWorkload(ModelParameters baseParams, WorkloadType workloadType) {
		switch (workloadType) {
			case COMPLETION:
				return optimizeForCompletion(baseParams);
			case EMBEDDING:
				return optimizeForEmbedding(baseParams);
			case RERANKING:
				return optimizeForReranking(baseParams);
			default:
				// Use general threading optimization
				return ThreadingOptimizer.optimize(baseParams, ThreadingOptimizer.WorkloadType.GENERAL);
		}
	}

	/**
	 * Print threading recommendations for manual tuning.
	 */
	public static void printThreadingRecommendations() {
		logger.log(DEBUG, "\n📊 Threading Recommendations:");
		logger.log(DEBUG, "   • Completion:  %d threads (balanced latency/throughput)",
			ThreadingOptimizer.getRecommendedThreads(ThreadingOptimizer.WorkloadType.COMPLETION));
		logger.log(DEBUG, "   • Embedding:   %d threads (compute-intensive)",
			ThreadingOptimizer.getRecommendedThreads(ThreadingOptimizer.WorkloadType.EMBEDDING));
		logger.log(DEBUG, "   • Reranking:   %d threads (high parallelism)",
			ThreadingOptimizer.getRecommendedThreads(ThreadingOptimizer.WorkloadType.RERANKING));
		logger.log(DEBUG, "   • General:     %d threads (balanced workload)",
			ThreadingOptimizer.getRecommendedThreads(ThreadingOptimizer.WorkloadType.GENERAL));
	}

	/**
	 * Workload types for optimization.
	 */
	public enum WorkloadType {
		COMPLETION,
		EMBEDDING,
		RERANKING,
		GENERAL
	}
}
