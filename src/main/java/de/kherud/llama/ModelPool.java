package de.kherud.llama;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.*;

/**
 * Advanced model pool with virtual threads, structured concurrency, and workload-aware routing.
 * Leverages Java 24 features for high-performance concurrent model access.
 */
public class ModelPool implements AutoCloseable {

	public enum ModelType {
		COMPLETION,
		EMBEDDING,
		RERANKING
	}

	public static class PoolConfig {
		private int completionPoolSize = 2;
		private int embeddingPoolSize = 1;
		private int rerankingPoolSize = 1;
		private long healthCheckIntervalMs = 30000;
		private boolean enableVirtualThreads = true;
		private boolean enableStructuredConcurrency = true;

		public PoolConfig setCompletionPoolSize(int size) { this.completionPoolSize = size; return this; }
		public PoolConfig setEmbeddingPoolSize(int size) { this.embeddingPoolSize = size; return this; }
		public PoolConfig setRerankingPoolSize(int size) { this.rerankingPoolSize = size; return this; }
		public PoolConfig setHealthCheckInterval(long ms) { this.healthCheckIntervalMs = ms; return this; }
		public PoolConfig setVirtualThreads(boolean enabled) { this.enableVirtualThreads = enabled; return this; }
		public PoolConfig setStructuredConcurrency(boolean enabled) { this.enableStructuredConcurrency = enabled; return this; }
	}

	private static class ModelInstance {
		final LlamaModel model;
		final ModelType type;
		final AtomicInteger activeRequests;
		volatile boolean healthy;
		volatile long lastUsed;

		ModelInstance(LlamaModel model, ModelType type) {
			this.model = model;
			this.type = type;
			this.activeRequests = new AtomicInteger(0);
			this.healthy = true;
			this.lastUsed = System.currentTimeMillis();
		}

		boolean tryAcquire() {
			if (!healthy) return false;
			activeRequests.incrementAndGet();
			lastUsed = System.currentTimeMillis();
			return true;
		}

		void release() {
			activeRequests.decrementAndGet();
		}

		int getLoad() {
			return activeRequests.get();
		}
	}

	private final PoolConfig config;
	private final Map<ModelType, List<ModelInstance>> modelPools;
	private final Map<ModelType, ModelParameters> modelConfigs;
	private final ReadWriteLock poolLock;
	private final ExecutorService executor;
	private final ScheduledExecutorService healthChecker;
	private volatile boolean closed = false;

	public ModelPool() {
		this(new PoolConfig());
	}

	public ModelPool(PoolConfig config) {
		this.config = config;
		this.modelPools = new EnumMap<>(ModelType.class);
		this.modelConfigs = new EnumMap<>(ModelType.class);
		this.poolLock = new ReentrantReadWriteLock();

		// Use virtual threads if available and enabled
		if (config.enableVirtualThreads) {
			this.executor = Executors.newVirtualThreadPerTaskExecutor();
		} else {
			this.executor = Executors.newCachedThreadPool(r -> {
				Thread t = new Thread(r);
				t.setName("model-pool-" + System.currentTimeMillis());
				t.setDaemon(true);
				return t;
			});
		}

		this.healthChecker = Executors.newScheduledThreadPool(1, r -> {
			Thread t = new Thread(r);
			t.setName("model-pool-health-checker");
			t.setDaemon(true);
			return t;
		});

		// Initialize pools
		for (ModelType type : ModelType.values()) {
			modelPools.put(type, new ArrayList<>());
		}

		// Start health checking
		healthChecker.scheduleAtFixedRate(this::performHealthCheck,
			config.healthCheckIntervalMs, config.healthCheckIntervalMs, TimeUnit.MILLISECONDS);
	}

	/**
	 * Add completion models to the pool
	 */
	public ModelPool withCompletion(ModelParameters params) {
		modelConfigs.put(ModelType.COMPLETION, params);
		initializePool(ModelType.COMPLETION, params, config.completionPoolSize);
		return this;
	}

	/**
	 * Add embedding models to the pool
	 */
	public ModelPool withEmbedding(ModelParameters params) {
		modelConfigs.put(ModelType.EMBEDDING, params);
		initializePool(ModelType.EMBEDDING, params, config.embeddingPoolSize);
		return this;
	}

	/**
	 * Add reranking models to the pool
	 */
	public ModelPool withReranking(ModelParameters params) {
		modelConfigs.put(ModelType.RERANKING, params);
		initializePool(ModelType.RERANKING, params, config.rerankingPoolSize);
		return this;
	}

	/**
	 * Generate text completion using the pool
	 */
	public CompletableFuture<String> complete(String prompt) {
		return complete(new InferenceParameters(prompt));
	}

	/**
	 * Generate text completion with custom parameters
	 */
	public CompletableFuture<String> complete(InferenceParameters params) {
		if (closed) return CompletableFuture.failedFuture(new IllegalStateException("Pool is closed"));

		if (config.enableStructuredConcurrency) {
			return executeWithStructuredConcurrency(() -> {
				ModelInstance instance = acquireModel(ModelType.COMPLETION);
				try {
					return instance.model.complete(params);
				} finally {
					instance.release();
				}
			});
		} else {
			return CompletableFuture.supplyAsync(() -> {
				ModelInstance instance = acquireModel(ModelType.COMPLETION);
				try {
					return instance.model.complete(params);
				} finally {
					instance.release();
				}
			}, executor);
		}
	}

	/**
	 * Generate embeddings using the pool
	 */
	public CompletableFuture<float[]> embed(String text) {
		if (closed) return CompletableFuture.failedFuture(new IllegalStateException("Pool is closed"));

		if (config.enableStructuredConcurrency) {
			return executeWithStructuredConcurrency(() -> {
				ModelInstance instance = acquireModel(ModelType.EMBEDDING);
				try {
					return instance.model.embed(text);
				} finally {
					instance.release();
				}
			});
		} else {
			return CompletableFuture.supplyAsync(() -> {
				ModelInstance instance = acquireModel(ModelType.EMBEDDING);
				try {
					return instance.model.embed(text);
				} finally {
					instance.release();
				}
			}, executor);
		}
	}

	/**
	 * Rerank documents using the pool
	 */
	public CompletableFuture<LlamaOutput> rerank(String query, String... documents) {
		if (closed) return CompletableFuture.failedFuture(new IllegalStateException("Pool is closed"));

		if (config.enableStructuredConcurrency) {
			return executeWithStructuredConcurrency(() -> {
				ModelInstance instance = acquireModel(ModelType.RERANKING);
				try {
					return instance.model.rerank(query, documents);
				} finally {
					instance.release();
				}
			});
		} else {
			return CompletableFuture.supplyAsync(() -> {
				ModelInstance instance = acquireModel(ModelType.RERANKING);
				try {
					return instance.model.rerank(query, documents);
				} finally {
					instance.release();
				}
			}, executor);
		}
	}

	/**
	 * Get pool statistics
	 */
	public PoolStats getStats() {
		poolLock.readLock().lock();
		try {
			Map<ModelType, Integer> totalInstances = new EnumMap<>(ModelType.class);
			Map<ModelType, Integer> activeRequests = new EnumMap<>(ModelType.class);
			Map<ModelType, Integer> healthyInstances = new EnumMap<>(ModelType.class);

			for (ModelType type : ModelType.values()) {
				List<ModelInstance> pool = modelPools.get(type);
				totalInstances.put(type, pool.size());

				int active = pool.stream().mapToInt(ModelInstance::getLoad).sum();
				activeRequests.put(type, active);

				int healthy = (int) pool.stream().filter(i -> i.healthy).count();
				healthyInstances.put(type, healthy);
			}

			return new PoolStats(totalInstances, activeRequests, healthyInstances, closed);
		} finally {
			poolLock.readLock().unlock();
		}
	}

	private void initializePool(ModelType type, ModelParameters params, int poolSize) {
		poolLock.writeLock().lock();
		try {
			List<ModelInstance> pool = modelPools.get(type);
			for (int i = 0; i < poolSize; i++) {
				LlamaModel model = createOptimizedModel(type, params);
				pool.add(new ModelInstance(model, type));
			}
		} finally {
			poolLock.writeLock().unlock();
		}
	}

	private LlamaModel createOptimizedModel(ModelType type, ModelParameters params) {
		return switch (type) {
			case COMPLETION -> LlamaModel.forCompletion(params);
			case EMBEDDING -> LlamaModel.forEmbedding(params);
			case RERANKING -> LlamaModel.forReranking(params);
		};
	}

	private ModelInstance acquireModel(ModelType type) {
		poolLock.readLock().lock();
		try {
			List<ModelInstance> pool = modelPools.get(type);
			if (pool.isEmpty()) {
				throw new IllegalStateException("No models available for type: " + type);
			}

			// Find least loaded healthy instance
			return pool.stream()
				.filter(i -> i.healthy)
				.min(Comparator.comparingInt(ModelInstance::getLoad))
				.filter(ModelInstance::tryAcquire)
				.orElseThrow(() -> new RuntimeException("No healthy models available for type: " + type));
		} finally {
			poolLock.readLock().unlock();
		}
	}

	private <T> CompletableFuture<T> executeWithStructuredConcurrency(Callable<T> task) {
		// For now, fall back to regular async execution since StructuredTaskScope requires preview features
		return CompletableFuture.supplyAsync(() -> {
			try {
				return task.call();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}, executor);
	}

	private void performHealthCheck() {
		if (closed) return;

		poolLock.readLock().lock();
		try {
			for (List<ModelInstance> pool : modelPools.values()) {
				for (ModelInstance instance : pool) {
					// Simple health check - could be enhanced
					try {
						// Test with a simple encode operation
						instance.model.encode("test");
						instance.healthy = true;
					} catch (Exception e) {
						instance.healthy = false;
					}
				}
			}
		} finally {
			poolLock.readLock().unlock();
		}
	}

	@Override
	public void close() {
		closed = true;

		healthChecker.shutdown();
		executor.shutdown();

		poolLock.writeLock().lock();
		try {
			for (List<ModelInstance> pool : modelPools.values()) {
				for (ModelInstance instance : pool) {
					try {
						instance.model.close();
					} catch (Exception e) {
						// Log but continue cleanup
					}
				}
				pool.clear();
			}
		} finally {
			poolLock.writeLock().unlock();
		}

		try {
			if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
				executor.shutdownNow();
			}
			if (!healthChecker.awaitTermination(5, TimeUnit.SECONDS)) {
				healthChecker.shutdownNow();
			}
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	/**
	 * Pool statistics
	 */
	public record PoolStats(
		Map<ModelType, Integer> totalInstances,
		Map<ModelType, Integer> activeRequests,
		Map<ModelType, Integer> healthyInstances,
		boolean poolClosed
	) {
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder("PoolStats{\n");
			for (ModelType type : ModelType.values()) {
				sb.append(String.format("  %s: %d total, %d active, %d healthy\n",
					type,
					totalInstances.getOrDefault(type, 0),
					activeRequests.getOrDefault(type, 0),
					healthyInstances.getOrDefault(type, 0)));
			}
			sb.append("  closed: ").append(poolClosed).append("\n}");
			return sb.toString();
		}
	}
}