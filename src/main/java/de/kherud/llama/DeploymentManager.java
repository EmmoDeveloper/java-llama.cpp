package de.kherud.llama;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Deployment manager for AI IDE applications.
 * Handles load balancing, auto-scaling, health monitoring, and graceful degradation.
 */
public class DeploymentManager implements AutoCloseable {

	/**
	 * Deployment strategies for different production scenarios
	 */
	public enum DeploymentStrategy {
		SINGLE_MODEL,           // Simple single model deployment
		ACTIVE_PASSIVE,         // Primary model with passive backup
		ACTIVE_ACTIVE,          // Multiple active models with load balancing
		CANARY_DEPLOYMENT,      // Gradual rollout with A/B testing
		BLUE_GREEN,             // Zero-downtime deployments
		AUTO_SCALING           // Dynamic scaling based on load
	}

	/**
	 * Health check strategies
	 */
	public enum HealthCheckStrategy {
		PING_ONLY,              // Simple availability check
		INFERENCE_TEST,         // Test with sample inference
		PERFORMANCE_MONITOR,    // Monitor response times and throughput
		COMPREHENSIVE          // Full health assessment
	}

	/**
	 * Deployment configuration
	 */
	public static class DeploymentConfig {
		public final DeploymentStrategy strategy;
		public final HealthCheckStrategy healthCheck;
		public final int minInstances;
		public final int maxInstances;
		public final double cpuThreshold;
		public final double memoryThreshold;
		public final long healthCheckIntervalMs;
		public final long scaleUpCooldownMs;
		public final long scaleDownCooldownMs;
		public final boolean enableCircuitBreaker;
		public final boolean enableMetrics;

		private DeploymentConfig(Builder builder) {
			this.strategy = builder.strategy;
			this.healthCheck = builder.healthCheck;
			this.minInstances = builder.minInstances;
			this.maxInstances = builder.maxInstances;
			this.cpuThreshold = builder.cpuThreshold;
			this.memoryThreshold = builder.memoryThreshold;
			this.healthCheckIntervalMs = builder.healthCheckIntervalMs;
			this.scaleUpCooldownMs = builder.scaleUpCooldownMs;
			this.scaleDownCooldownMs = builder.scaleDownCooldownMs;
			this.enableCircuitBreaker = builder.enableCircuitBreaker;
			this.enableMetrics = builder.enableMetrics;
		}

		public static class Builder {
			private DeploymentStrategy strategy = DeploymentStrategy.ACTIVE_ACTIVE;
			private HealthCheckStrategy healthCheck = HealthCheckStrategy.PERFORMANCE_MONITOR;
			private int minInstances = 1;
			private int maxInstances = 5;
			private double cpuThreshold = 0.8;
			private double memoryThreshold = 0.85;
			private long healthCheckIntervalMs = 30000;
			private long scaleUpCooldownMs = 300000; // 5 minutes
			private long scaleDownCooldownMs = 600000; // 10 minutes
			private boolean enableCircuitBreaker = true;
			private boolean enableMetrics = true;

			public Builder strategy(DeploymentStrategy strategy) { this.strategy = strategy; return this; }
			public Builder healthCheck(HealthCheckStrategy healthCheck) { this.healthCheck = healthCheck; return this; }
			public Builder minInstances(int min) { this.minInstances = min; return this; }
			public Builder maxInstances(int max) { this.maxInstances = max; return this; }
			public Builder cpuThreshold(double threshold) { this.cpuThreshold = threshold; return this; }
			public Builder memoryThreshold(double threshold) { this.memoryThreshold = threshold; return this; }
			public Builder healthCheckInterval(long ms) { this.healthCheckIntervalMs = ms; return this; }
			public Builder scaleUpCooldown(long ms) { this.scaleUpCooldownMs = ms; return this; }
			public Builder scaleDownCooldown(long ms) { this.scaleDownCooldownMs = ms; return this; }
			public Builder enableCircuitBreaker(boolean enable) { this.enableCircuitBreaker = enable; return this; }
			public Builder enableMetrics(boolean enable) { this.enableMetrics = enable; return this; }

			public DeploymentConfig build() {
				return new DeploymentConfig(this);
			}
		}
	}

	/**
	 * Circuit breaker for handling model failures gracefully
	 */
	public static class CircuitBreaker {
		public enum State { CLOSED, OPEN, HALF_OPEN }

		private final int failureThreshold;
		private final long timeoutMs;
		private final long retryTimeoutMs;

		private volatile State state = State.CLOSED;
		private final AtomicInteger failures = new AtomicInteger(0);
		private volatile long lastFailureTime = 0;
		private volatile long stateChangeTime = 0;

		public CircuitBreaker(int failureThreshold, long timeoutMs, long retryTimeoutMs) {
			this.failureThreshold = failureThreshold;
			this.timeoutMs = timeoutMs;
			this.retryTimeoutMs = retryTimeoutMs;
		}

		public boolean allowRequest() {
			switch (state) {
				case OPEN:
					if (System.currentTimeMillis() - stateChangeTime > retryTimeoutMs) {
						state = State.HALF_OPEN;
						return true;
					}
					return false;
				case HALF_OPEN:
					return true;
				case CLOSED:
				default:
					return true;
			}
		}

		public void recordSuccess() {
			failures.set(0);
			if (state == State.HALF_OPEN) {
				state = State.CLOSED;
				stateChangeTime = System.currentTimeMillis();
			}
		}

		public void recordFailure() {
			failures.incrementAndGet();
			lastFailureTime = System.currentTimeMillis();

			if (failures.get() >= failureThreshold) {
				state = State.OPEN;
				stateChangeTime = System.currentTimeMillis();
			}
		}

		public State getState() { return state; }
		public int getFailureCount() { return failures.get(); }
	}

	/**
	 * Auto-scaling manager
	 */
	public static class AutoScaler {
		private final DeploymentConfig config;
		private final MultiModelManager modelManager;
		private volatile long lastScaleUpTime = 0;
		private volatile long lastScaleDownTime = 0;
		private final AtomicInteger currentInstances = new AtomicInteger(0);

		public AutoScaler(DeploymentConfig config, MultiModelManager modelManager) {
			this.config = config;
			this.modelManager = modelManager;
			this.currentInstances.set(config.minInstances);
		}

		public boolean shouldScaleUp() {
			long now = System.currentTimeMillis();
			if (now - lastScaleUpTime < config.scaleUpCooldownMs) {
				return false;
			}

			if (currentInstances.get() >= config.maxInstances) {
				return false;
			}

			// Check if we need to scale up based on metrics
			Map<String, Object> metrics = modelManager.getMetrics();
			@SuppressWarnings("unchecked")
			Map<String, Object> modelStats = (Map<String, Object>) metrics.get("models");

			if (modelStats == null) return false;

			// Calculate average load across all models
			double totalLoad = 0;
			int modelCount = 0;

			for (Object stats : modelStats.values()) {
				if (stats instanceof Map) {
					@SuppressWarnings("unchecked")
					Map<String, Object> modelStat = (Map<String, Object>) stats;
					Object activeRequests = modelStat.get("activeRequests");
					if (activeRequests instanceof Number) {
						totalLoad += ((Number) activeRequests).doubleValue();
						modelCount++;
					}
				}
			}

			if (modelCount == 0) return false;

			double avgLoad = totalLoad / modelCount;
			return avgLoad > 0.8; // Scale up if average load > 80%
		}

		public boolean shouldScaleDown() {
			long now = System.currentTimeMillis();
			if (now - lastScaleDownTime < config.scaleDownCooldownMs) {
				return false;
			}

			if (currentInstances.get() <= config.minInstances) {
				return false;
			}

			// Check if we can scale down based on low load
			Map<String, Object> metrics = modelManager.getMetrics();
			@SuppressWarnings("unchecked")
			Map<String, Object> modelStats = (Map<String, Object>) metrics.get("models");

			if (modelStats == null) return false;

			double totalLoad = 0;
			int modelCount = 0;

			for (Object stats : modelStats.values()) {
				if (stats instanceof Map) {
					@SuppressWarnings("unchecked")
					Map<String, Object> modelStat = (Map<String, Object>) stats;
					Object activeRequests = modelStat.get("activeRequests");
					if (activeRequests instanceof Number) {
						totalLoad += ((Number) activeRequests).doubleValue();
						modelCount++;
					}
				}
			}

			if (modelCount == 0) return true;

			double avgLoad = totalLoad / modelCount;
			return avgLoad < 0.3; // Scale down if average load < 30%
		}

		public void scaleUp() {
			lastScaleUpTime = System.currentTimeMillis();
			int newCount = currentInstances.incrementAndGet();
			System.out.println("Scaling up to " + newCount + " instances");
		}

		public void scaleDown() {
			lastScaleDownTime = System.currentTimeMillis();
			int newCount = currentInstances.decrementAndGet();
			System.out.println("Scaling down to " + newCount + " instances");
		}

		public int getCurrentInstances() {
			return currentInstances.get();
		}
	}

	// Core components
	private final DeploymentConfig config;
	private final MultiModelManager modelManager;
	private final CircuitBreaker circuitBreaker;
	private final AutoScaler autoScaler;
	private final ScheduledExecutorService scheduler;
	private final AtomicBoolean running = new AtomicBoolean(false);

	// Metrics
	private final AtomicLong totalRequests = new AtomicLong(0);
	private final AtomicLong successfulRequests = new AtomicLong(0);
	private final AtomicLong failedRequests = new AtomicLong(0);
	private final AtomicLong totalResponseTime = new AtomicLong(0);

	public DeploymentManager(DeploymentConfig config, MultiModelManager modelManager) {
		this.config = config;
		this.modelManager = modelManager;

		this.circuitBreaker = config.enableCircuitBreaker ?
			new CircuitBreaker(5, 30000, 60000) : null;

		this.autoScaler = config.strategy == DeploymentStrategy.AUTO_SCALING ?
			new AutoScaler(config, modelManager) : null;

		this.scheduler = Executors.newScheduledThreadPool(3, r -> {
			Thread t = new Thread(r, "production-deployment-scheduler");
			t.setDaemon(true);
			return t;
		});
	}

	/**
	 * Start the production deployment
	 */
	public void start() {
		if (!running.compareAndSet(false, true)) {
			throw new IllegalStateException("Deployment manager already running");
		}

		System.out.println("Starting production deployment with strategy: " + config.strategy);

		// Start health monitoring
		scheduler.scheduleWithFixedDelay(this::performHealthCheck,
			config.healthCheckIntervalMs, config.healthCheckIntervalMs, TimeUnit.MILLISECONDS);

		// Start auto-scaling monitoring
		if (autoScaler != null) {
			scheduler.scheduleWithFixedDelay(this::performAutoScaling,
				60000, 60000, TimeUnit.MILLISECONDS); // Check every minute
		}

		// Start metrics collection
		if (config.enableMetrics) {
			scheduler.scheduleWithFixedDelay(this::collectMetrics,
				300000, 300000, TimeUnit.MILLISECONDS); // Collect every 5 minutes
		}
	}

	/**
	 * Execute request with error handling and monitoring
	 */
	public <T> CompletableFuture<T> executeRequest(
			MultiModelManager.RequestContext context,
			java.util.function.Function<LlamaModel, T> task) {

		totalRequests.incrementAndGet();

		// Check circuit breaker
		if (circuitBreaker != null && !circuitBreaker.allowRequest()) {
			CompletableFuture<T> failedFuture = new CompletableFuture<>();
			failedFuture.completeExceptionally(
				new RuntimeException("Circuit breaker is OPEN"));
			failedRequests.incrementAndGet();
			return failedFuture;
		}

		long startTime = System.currentTimeMillis();

		return modelManager.execute(context, task)
			.whenComplete((result, throwable) -> {
				long responseTime = System.currentTimeMillis() - startTime;
				totalResponseTime.addAndGet(responseTime);

				if (throwable == null) {
					successfulRequests.incrementAndGet();
					if (circuitBreaker != null) {
						circuitBreaker.recordSuccess();
					}
				} else {
					failedRequests.incrementAndGet();
					if (circuitBreaker != null) {
						circuitBreaker.recordFailure();
					}
				}
			});
	}

	/**
	 * Execute with fallback strategies for graceful degradation
	 */
	public <T> CompletableFuture<T> executeWithFallback(
			MultiModelManager.RequestContext context,
			java.util.function.Function<LlamaModel, T> task,
			java.util.function.Supplier<T> fallback) {

		return executeRequest(context, task)
			.exceptionally(throwable -> {
				System.err.println("Primary execution failed, using fallback: " + throwable.getMessage());
				try {
					return fallback.get();
				} catch (Exception e) {
					throw new RuntimeException("Both primary and fallback execution failed", e);
				}
			});
	}

	/**
	 * Get production metrics
	 */
	public Map<String, Object> getProductionMetrics() {
		Map<String, Object> metrics = new HashMap<>();

		// Request metrics
		long total = totalRequests.get();
		long successful = successfulRequests.get();
		long failed = failedRequests.get();

		metrics.put("totalRequests", total);
		metrics.put("successfulRequests", successful);
		metrics.put("failedRequests", failed);
		metrics.put("successRate", total > 0 ? (double) successful / total : 0.0);
		metrics.put("errorRate", total > 0 ? (double) failed / total : 0.0);
		metrics.put("averageResponseTime",
			successful > 0 ? (double) totalResponseTime.get() / successful : 0.0);

		// Circuit breaker metrics
		if (circuitBreaker != null) {
			Map<String, Object> cbMetrics = new HashMap<>();
			cbMetrics.put("state", circuitBreaker.getState().name());
			cbMetrics.put("failureCount", circuitBreaker.getFailureCount());
			metrics.put("circuitBreaker", cbMetrics);
		}

		// Auto-scaler metrics
		if (autoScaler != null) {
			Map<String, Object> scalerMetrics = new HashMap<>();
			scalerMetrics.put("currentInstances", autoScaler.getCurrentInstances());
			scalerMetrics.put("minInstances", config.minInstances);
			scalerMetrics.put("maxInstances", config.maxInstances);
			metrics.put("autoScaler", scalerMetrics);
		}

		// Deployment info
		Map<String, Object> deploymentInfo = new HashMap<>();
		deploymentInfo.put("strategy", config.strategy.name());
		deploymentInfo.put("healthCheckStrategy", config.healthCheck.name());
		deploymentInfo.put("running", running.get());
		metrics.put("deployment", deploymentInfo);

		// Include model manager metrics
		metrics.putAll(modelManager.getMetrics());

		return metrics;
	}

	@Override
	public void close() {
		if (!running.compareAndSet(true, false)) {
			return; // Already closed
		}

		System.out.println("Shutting down production deployment manager");

		scheduler.shutdown();
		try {
			if (!scheduler.awaitTermination(30, TimeUnit.SECONDS)) {
				scheduler.shutdownNow();
			}
		} catch (InterruptedException e) {
			scheduler.shutdownNow();
			Thread.currentThread().interrupt();
		}
	}

	// Private helper methods

	private void performHealthCheck() {
		try {
			switch (config.healthCheck) {
				case PING_ONLY:
					performBasicHealthCheck();
					break;
				case INFERENCE_TEST:
					performInferenceHealthCheck();
					break;
				case PERFORMANCE_MONITOR:
					performPerformanceHealthCheck();
					break;
				case COMPREHENSIVE:
					performHealthCheck();
					break;
			}
		} catch (Exception e) {
			System.err.println("Health check failed: " + e.getMessage());
		}
	}

	private void performBasicHealthCheck() {
		Map<String, Object> metrics = modelManager.getMetrics();
		@SuppressWarnings("unchecked")
		Map<String, Object> modelStats = (Map<String, Object>) metrics.get("models");

		if (modelStats == null || modelStats.isEmpty()) {
			System.err.println("No healthy models available");
		}
	}

	private void performInferenceHealthCheck() {
		MultiModelManager.RequestContext context = new MultiModelManager.RequestContext.Builder()
			.taskType("health_check")
			.timeout(5000) // 5 second timeout
			.build();

		try {
			CompletableFuture<String> healthTest = modelManager.execute(context,
				model -> "Health check successful");

			healthTest.get(10, TimeUnit.SECONDS);
		} catch (Exception e) {
			System.err.println("Inference health check failed: " + e.getMessage());
		}
	}

	private void performPerformanceHealthCheck() {
		Map<String, Object> metrics = getProductionMetrics();
		Double avgResponseTime = (Double) metrics.get("averageResponseTime");
		Double errorRate = (Double) metrics.get("errorRate");

		if (avgResponseTime != null && avgResponseTime > 10000) { // 10 seconds
			System.err.println("High average response time detected: " + avgResponseTime + "ms");
		}

		if (errorRate != null && errorRate > 0.1) { // 10% error rate
			System.err.println("High error rate detected: " + (errorRate * 100) + "%");
		}
	}


	private void performAutoScaling() {
		if (autoScaler == null) return;

		try {
			if (autoScaler.shouldScaleUp()) {
				autoScaler.scaleUp();
				// In a real implementation, this would trigger actual instance creation
			} else if (autoScaler.shouldScaleDown()) {
				autoScaler.scaleDown();
				// In a real implementation, this would trigger instance termination
			}
		} catch (Exception e) {
			System.err.println("Auto-scaling check failed: " + e.getMessage());
		}
	}

	private void collectMetrics() {
		if (!config.enableMetrics) return;

		try {
			Map<String, Object> metrics = getProductionMetrics();
			System.out.println("Production Metrics: " +
				"Requests: " + metrics.get("totalRequests") +
				", Success Rate: " + String.format("%.2f%%", (Double) metrics.get("successRate") * 100) +
				", Avg Response: " + String.format("%.1fms", (Double) metrics.get("averageResponseTime")));
		} catch (Exception e) {
			System.err.println("Metrics collection failed: " + e.getMessage());
		}
	}
}
