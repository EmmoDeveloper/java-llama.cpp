package de.kherud.llama;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;

/**
 * Advanced multi-model management system for production AI IDE deployment.
 * Supports model ensembles, dynamic routing, load balancing, and intelligent fallback.
 */
public class MultiModelManager implements AutoCloseable {

	/**
	 * Model specializations for different AI IDE tasks
	 */
	public enum ModelSpecialization {
		CODE_COMPLETION,        // Fast, syntax-aware models
		CODE_GENERATION,        // Larger models for complex code generation
		DOCUMENTATION,          // Models optimized for natural language
		DEBUGGING,              // Models trained on error analysis
		REFACTORING,           // Models specialized in code transformation
		EXPLANATION,           // Models for code explanation and teaching
		TESTING,               // Models for test generation
		EMBEDDING,             // Embedding models for semantic search
		GENERAL_PURPOSE        // Fallback general models
	}

	/**
	 * Routing strategies for model selection
	 */
	public enum RoutingStrategy {
		ROUND_ROBIN,           // Simple round-robin distribution
		LOAD_BALANCED,         // Route to least loaded model
		PERFORMANCE_BASED,     // Route to fastest responding model
		CONTEXT_AWARE,         // Route based on request context
		ENSEMBLE_VOTING,       // Use multiple models and vote
		CAPABILITY_MATCHED     // Match request to model capabilities
	}

	/**
	 * Model configuration with specialization and routing
	 */
	public static class ModelConfig {
		public final String modelId;
		public final String modelPath;
		public final ModelSpecialization specialization;
		public final ModelParameters parameters;
		public final Map<String, Object> metadata;
		public final Set<String> capabilities;
		public final int priority;
		public final int maxConcurrent;

		private ModelConfig(Builder builder) {
			this.modelId = builder.modelId;
			this.modelPath = builder.modelPath;
			this.specialization = builder.specialization;
			this.parameters = builder.parameters;
			this.metadata = new HashMap<>(builder.metadata);
			this.capabilities = new HashSet<>(builder.capabilities);
			this.priority = builder.priority;
			this.maxConcurrent = builder.maxConcurrent;
		}

		public static class Builder {
			private String modelId;
			private String modelPath;
			private ModelSpecialization specialization = ModelSpecialization.GENERAL_PURPOSE;
			private ModelParameters parameters = new ModelParameters();
			private Map<String, Object> metadata = new HashMap<>();
			private Set<String> capabilities = new HashSet<>();
			private int priority = 5;
			private int maxConcurrent = 1;

			public Builder(String modelId, String modelPath) {
				this.modelId = modelId;
				this.modelPath = modelPath;
			}

			public Builder specialization(ModelSpecialization spec) { this.specialization = spec; return this; }
			public Builder parameters(ModelParameters params) { this.parameters = params; return this; }
			public Builder metadata(String key, Object value) { this.metadata.put(key, value); return this; }
			public Builder capability(String capability) { this.capabilities.add(capability); return this; }
			public Builder priority(int priority) { this.priority = priority; return this; }
			public Builder maxConcurrent(int max) { this.maxConcurrent = max; return this; }

			public ModelConfig build() {
				return new ModelConfig(this);
			}
		}
	}

	/**
	 * Request context for intelligent routing
	 */
	public static class RequestContext {
		public final String language;
		public final String fileType;
		public final String taskType;
		public final Map<String, String> metadata;
		public final Set<String> requiredCapabilities;
		public final int priority;
		public final long timeoutMs;

		private RequestContext(Builder builder) {
			this.language = builder.language;
			this.fileType = builder.fileType;
			this.taskType = builder.taskType;
			this.metadata = new HashMap<>(builder.metadata);
			this.requiredCapabilities = new HashSet<>(builder.requiredCapabilities);
			this.priority = builder.priority;
			this.timeoutMs = builder.timeoutMs;
		}

		public static class Builder {
			private String language = "unknown";
			private String fileType = "unknown";
			private String taskType = "general";
			private Map<String, String> metadata = new HashMap<>();
			private Set<String> requiredCapabilities = new HashSet<>();
			private int priority = 5;
			private long timeoutMs = 30000;

			public Builder language(String language) { this.language = language; return this; }
			public Builder fileType(String fileType) { this.fileType = fileType; return this; }
			public Builder taskType(String taskType) { this.taskType = taskType; return this; }
			public Builder metadata(String key, String value) { this.metadata.put(key, value); return this; }
			public Builder requireCapability(String capability) { this.requiredCapabilities.add(capability); return this; }
			public Builder priority(int priority) { this.priority = priority; return this; }
			public Builder timeout(long timeoutMs) { this.timeoutMs = timeoutMs; return this; }

			public RequestContext build() {
				return new RequestContext(this);
			}
		}
	}

	/**
	 * Enhanced model instance with performance metrics
	 */
	private static class EnhancedModelInstance {
		public final ModelConfig config;
		public final LlamaModel model;
		public final AtomicLong requestCount;
		public final AtomicLong totalResponseTime;
		public final AtomicLong activeRequests;
		public volatile boolean healthy;
		public volatile long lastUsed;
		public volatile double avgResponseTime;
		public volatile int currentLoad;

		public EnhancedModelInstance(ModelConfig config, LlamaModel model) {
			this.config = config;
			this.model = model;
			this.requestCount = new AtomicLong(0);
			this.totalResponseTime = new AtomicLong(0);
			this.activeRequests = new AtomicLong(0);
			this.healthy = true;
			this.lastUsed = System.currentTimeMillis();
			this.avgResponseTime = 0.0;
			this.currentLoad = 0;
		}

		public boolean tryAcquire() {
			if (!healthy || activeRequests.get() >= config.maxConcurrent) {
				return false;
			}
			activeRequests.incrementAndGet();
			lastUsed = System.currentTimeMillis();
			return true;
		}

		public void release(long responseTimeMs) {
			activeRequests.decrementAndGet();
			requestCount.incrementAndGet();
			totalResponseTime.addAndGet(responseTimeMs);

			// Update average response time
			long requests = requestCount.get();
			if (requests > 0) {
				avgResponseTime = (double) totalResponseTime.get() / requests;
			}

			currentLoad = (int) activeRequests.get();
		}

		public double getScore(RequestContext context) {
			double score = 0.0;

			// Priority score (higher priority = higher score)
			score += config.priority * 20;

			// Performance score (lower response time = higher score)
			if (avgResponseTime > 0) {
				score += Math.max(0, 1000 - avgResponseTime) / 10;
			}

			// Load score (lower load = higher score)
			score += Math.max(0, 100 - (currentLoad * 25));

			// Capability match score
			long matchedCapabilities = context.requiredCapabilities.stream()
				.mapToLong(req -> config.capabilities.contains(req) ? 1 : 0)
				.sum();
			score += matchedCapabilities * 50;

			// Specialization match score
			if (isSpecializationMatch(config.specialization, context)) {
				score += 100;
			}

			// Health penalty
			if (!healthy) {
				score *= 0.1;
			}

			return score;
		}

		private boolean isSpecializationMatch(ModelSpecialization spec, RequestContext context) {
			switch (spec) {
				case CODE_COMPLETION:
					return "completion".equals(context.taskType);
				case CODE_GENERATION:
					return "generation".equals(context.taskType);
				case DOCUMENTATION:
					return "documentation".equals(context.taskType);
				case DEBUGGING:
					return "debug".equals(context.taskType);
				case REFACTORING:
					return "refactor".equals(context.taskType);
				case EXPLANATION:
					return "explain".equals(context.taskType);
				case TESTING:
					return "testing".equals(context.taskType);
				case EMBEDDING:
					return "embedding".equals(context.taskType);
				default:
					return true; // General purpose matches anything
			}
		}
	}

	// Core components
	private final Map<String, EnhancedModelInstance> models;
	private final Map<ModelSpecialization, List<String>> specializationIndex;
	private final ReadWriteLock modelsLock;
	private final ScheduledExecutorService healthChecker;
	private final ScheduledExecutorService metricsCollector;
	private volatile boolean closed = false;

	// Routing configuration
	private RoutingStrategy defaultRoutingStrategy = RoutingStrategy.PERFORMANCE_BASED;
	private final Map<String, RoutingStrategy> taskSpecificRouting;

	// Performance metrics
	private final AtomicLong totalRequests;
	private final AtomicLong totalResponseTime;
	private final Map<String, AtomicLong> modelMetrics;

	public MultiModelManager() {
		this.models = new ConcurrentHashMap<>();
		this.specializationIndex = new ConcurrentHashMap<>();
		this.modelsLock = new ReentrantReadWriteLock();
		this.taskSpecificRouting = new ConcurrentHashMap<>();
		this.totalRequests = new AtomicLong(0);
		this.totalResponseTime = new AtomicLong(0);
		this.modelMetrics = new ConcurrentHashMap<>();

		// Initialize specialization index
		for (ModelSpecialization spec : ModelSpecialization.values()) {
			specializationIndex.put(spec, new ArrayList<>());
		}

		// Start health checker
		this.healthChecker = Executors.newScheduledThreadPool(1, r -> {
			Thread t = new Thread(r, "multi-model-health-checker");
			t.setDaemon(true);
			return t;
		});
		healthChecker.scheduleWithFixedDelay(this::performHealthCheck, 30, 30, TimeUnit.SECONDS);

		// Start metrics collector
		this.metricsCollector = Executors.newScheduledThreadPool(1, r -> {
			Thread t = new Thread(r, "multi-model-metrics");
			t.setDaemon(true);
			return t;
		});
		metricsCollector.scheduleWithFixedDelay(this::collectMetrics, 60, 60, TimeUnit.SECONDS);
	}

	/**
	 * Register a new model with the manager
	 */
	public void registerModel(ModelConfig config) throws LlamaException {
		modelsLock.writeLock().lock();
		try {
			if (models.containsKey(config.modelId)) {
				throw new IllegalArgumentException("Model already registered: " + config.modelId);
			}

			// Load the model
			LlamaModel model = new LlamaModel(config.parameters.setModel(config.modelPath));
			EnhancedModelInstance instance = new EnhancedModelInstance(config, model);

			models.put(config.modelId, instance);
			specializationIndex.get(config.specialization).add(config.modelId);
			modelMetrics.put(config.modelId, new AtomicLong(0));

			System.out.println("Registered model: " + config.modelId +
							   " (specialization: " + config.specialization + ")");

		} finally {
			modelsLock.writeLock().unlock();
		}
	}

	/**
	 * Execute a request with intelligent model routing
	 */
	public <T> CompletableFuture<T> execute(RequestContext context,
											Function<LlamaModel, T> task) {
		return CompletableFuture.supplyAsync(() -> {
			long startTime = System.currentTimeMillis();
			totalRequests.incrementAndGet();

			try {
				EnhancedModelInstance selectedModel = selectModel(context);
				if (selectedModel == null) {
					throw new IllegalStateException("No suitable model available for request");
				}

				if (!selectedModel.tryAcquire()) {
					throw new IllegalStateException("Selected model is not available: " +
													selectedModel.config.modelId);
				}

				try {
					T result = task.apply(selectedModel.model);
					long responseTime = System.currentTimeMillis() - startTime;
					selectedModel.release(responseTime);
					totalResponseTime.addAndGet(responseTime);
					modelMetrics.get(selectedModel.config.modelId).incrementAndGet();

					return result;
				} finally {
					// Ensure release is called even if task throws exception
					if (selectedModel.activeRequests.get() > 0) {
						selectedModel.release(System.currentTimeMillis() - startTime);
					}
				}

			} catch (Exception e) {
				throw new RuntimeException("Request execution failed", e);
			}
		});
	}

	/**
	 * Execute request with ensemble voting (multiple models)
	 */
	public <T> CompletableFuture<List<T>> executeEnsemble(RequestContext context,
														  Function<LlamaModel, T> task,
														  int maxModels) {
		List<EnhancedModelInstance> selectedModels = selectEnsembleModels(context, maxModels);

		List<CompletableFuture<T>> futures = selectedModels.stream()
			.map(instance -> CompletableFuture.supplyAsync(() -> {
				long startTime = System.currentTimeMillis();

				if (!instance.tryAcquire()) {
					throw new RuntimeException("Model not available: " + instance.config.modelId);
				}

				try {
					T result = task.apply(instance.model);
					instance.release(System.currentTimeMillis() - startTime);
					return result;
				} finally {
					if (instance.activeRequests.get() > 0) {
						instance.release(System.currentTimeMillis() - startTime);
					}
				}
			}))
			.toList();

		return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
			.thenApply(v -> futures.stream()
				.map(CompletableFuture::join)
				.toList());
	}

	/**
	 * Get models by specialization
	 */
	public List<String> getModelsBySpecialization(ModelSpecialization specialization) {
		return new ArrayList<>(specializationIndex.get(specialization));
	}

	/**
	 * Get system metrics
	 */
	public Map<String, Object> getMetrics() {
		Map<String, Object> metrics = new HashMap<>();

		modelsLock.readLock().lock();
		try {
			metrics.put("totalRequests", totalRequests.get());
			metrics.put("averageResponseTime",
				totalRequests.get() > 0 ? (double) totalResponseTime.get() / totalRequests.get() : 0.0);

			Map<String, Object> modelStats = new HashMap<>();
			for (Map.Entry<String, EnhancedModelInstance> entry : models.entrySet()) {
				EnhancedModelInstance instance = entry.getValue();
				Map<String, Object> stats = new HashMap<>();
				stats.put("healthy", instance.healthy);
				stats.put("activeRequests", instance.activeRequests.get());
				stats.put("totalRequests", instance.requestCount.get());
				stats.put("averageResponseTime", instance.avgResponseTime);
				stats.put("specialization", instance.config.specialization);
				stats.put("capabilities", instance.config.capabilities);

				modelStats.put(entry.getKey(), stats);
			}
			metrics.put("models", modelStats);

			Map<String, Integer> specializationCounts = new HashMap<>();
			for (Map.Entry<ModelSpecialization, List<String>> entry : specializationIndex.entrySet()) {
				specializationCounts.put(entry.getKey().name(), entry.getValue().size());
			}
			metrics.put("specializationCounts", specializationCounts);

		} finally {
			modelsLock.readLock().unlock();
		}

		return metrics;
	}

	/**
	 * Set routing strategy for specific task types
	 */
	public void setTaskRoutingStrategy(String taskType, RoutingStrategy strategy) {
		taskSpecificRouting.put(taskType, strategy);
	}

	@Override
	public void close() {
		if (closed) return;
		closed = true;

		healthChecker.shutdown();
		metricsCollector.shutdown();

		modelsLock.writeLock().lock();
		try {
			for (EnhancedModelInstance instance : models.values()) {
				try {
					instance.model.close();
				} catch (Exception e) {
					System.err.println("Error closing model " + instance.config.modelId + ": " + e.getMessage());
				}
			}
			models.clear();
			specializationIndex.clear();
		} finally {
			modelsLock.writeLock().unlock();
		}
	}

	// Private helper methods

	private EnhancedModelInstance selectModel(RequestContext context) {
		RoutingStrategy strategy = taskSpecificRouting.getOrDefault(context.taskType, defaultRoutingStrategy);

		modelsLock.readLock().lock();
		try {
			List<EnhancedModelInstance> candidates = getCandidateModels(context);
			if (candidates.isEmpty()) {
				return null;
			}

			switch (strategy) {
				case LOAD_BALANCED:
					return candidates.stream()
						.min(Comparator.comparing(m -> m.currentLoad))
						.orElse(null);

				case PERFORMANCE_BASED:
					return candidates.stream()
						.max(Comparator.comparing(m -> m.getScore(context)))
						.orElse(null);

				case CONTEXT_AWARE:
					return candidates.stream()
						.filter(m -> isContextMatch(m, context))
						.max(Comparator.comparing(m -> m.getScore(context)))
						.orElse(candidates.get(0));

				case ROUND_ROBIN:
				default:
					return candidates.get((int) (totalRequests.get() % candidates.size()));
			}
		} finally {
			modelsLock.readLock().unlock();
		}
	}

	private List<EnhancedModelInstance> selectEnsembleModels(RequestContext context, int maxModels) {
		modelsLock.readLock().lock();
		try {
			return getCandidateModels(context).stream()
				.sorted(Comparator.comparing((EnhancedModelInstance m) -> m.getScore(context)).reversed())
				.limit(maxModels)
				.toList();
		} finally {
			modelsLock.readLock().unlock();
		}
	}

	private List<EnhancedModelInstance> getCandidateModels(RequestContext context) {
		return models.values().stream()
			.filter(m -> m.healthy)
			.filter(m -> meetsRequirements(m, context))
			.toList();
	}

	private boolean meetsRequirements(EnhancedModelInstance model, RequestContext context) {
		// Check required capabilities
		if (!model.config.capabilities.containsAll(context.requiredCapabilities)) {
			return false;
		}

		// Check availability
		return model.activeRequests.get() < model.config.maxConcurrent;
	}

	private boolean isContextMatch(EnhancedModelInstance model, RequestContext context) {
		// Simple context matching based on file type and language
		return context.language.equalsIgnoreCase((String) model.config.metadata.get("language")) ||
			   context.fileType.equalsIgnoreCase((String) model.config.metadata.get("fileType"));
	}

	private void performHealthCheck() {
		modelsLock.readLock().lock();
		try {
			for (EnhancedModelInstance instance : models.values()) {
				// Simple health check - could be enhanced with actual model ping
				boolean wasHealthy = instance.healthy;
				instance.healthy = (System.currentTimeMillis() - instance.lastUsed) < 300000; // 5 minutes

				if (wasHealthy != instance.healthy) {
					System.out.println("Model " + instance.config.modelId +
									   " health changed to: " + instance.healthy);
				}
			}
		} finally {
			modelsLock.readLock().unlock();
		}
	}

	private void collectMetrics() {
		// Collect and optionally export metrics
		Map<String, Object> currentMetrics = getMetrics();
		System.out.println("Multi-model metrics: " + currentMetrics.get("totalRequests") +
						   " requests, avg response: " + currentMetrics.get("averageResponseTime") + "ms");
	}
}
