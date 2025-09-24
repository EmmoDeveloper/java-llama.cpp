package de.kherud.llama;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * System monitoring and metrics collection for llama.cpp operations.
 * Tracks performance, resource usage, and operational statistics.
 */
public class SystemMonitor {

	// Global performance counters
	private static final AtomicLong totalInferences = new AtomicLong(0);
	private static final AtomicLong totalTokensGenerated = new AtomicLong(0);
	private static final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
	private static final AtomicInteger activeModels = new AtomicInteger(0);

	// Per-model statistics
	private static final Map<String, ModelStats> modelStatistics = new ConcurrentHashMap<>();

	/**
	 * Statistics for individual models.
	 */
	public static class ModelStats {
		private final AtomicLong inferences = new AtomicLong(0);
		private final AtomicLong tokensGenerated = new AtomicLong(0);
		private final AtomicLong totalLatencyMs = new AtomicLong(0);
		private final AtomicLong errors = new AtomicLong(0);
		private volatile long lastUsed = System.currentTimeMillis();
		private volatile String lastError = null;

		public long getInferences() { return inferences.get(); }
		public long getTokensGenerated() { return tokensGenerated.get(); }
		public long getTotalLatencyMs() { return totalLatencyMs.get(); }
		public long getErrors() { return errors.get(); }
		public long getLastUsed() { return lastUsed; }
		public String getLastError() { return lastError; }

		public double getAverageLatencyMs() {
			long inf = inferences.get();
			return inf > 0 ? (double) totalLatencyMs.get() / inf : 0.0;
		}

		public double getTokensPerSecond() {
			long latency = totalLatencyMs.get();
			return latency > 0 ? (double) tokensGenerated.get() * 1000 / latency : 0.0;
		}

		void recordInference(int tokens, long latencyMs) {
			inferences.incrementAndGet();
			tokensGenerated.addAndGet(tokens);
			totalLatencyMs.addAndGet(latencyMs);
			lastUsed = System.currentTimeMillis();
		}

		void recordError(String error) {
			errors.incrementAndGet();
			lastError = error;
			lastUsed = System.currentTimeMillis();
		}
	}

	/**
	 * System-wide performance metrics.
	 */
	public static class SystemMetrics {
		private final long totalInferences;
		private final long totalTokens;
		private final long totalProcessingTime;
		private final int activeModels;
		private final double averageLatency;
		private final double tokensPerSecond;
		private final long uptime;

		private SystemMetrics(long totalInferences, long totalTokens, long totalProcessingTime,
							 int activeModels, long uptime) {
			this.totalInferences = totalInferences;
			this.totalTokens = totalTokens;
			this.totalProcessingTime = totalProcessingTime;
			this.activeModels = activeModels;
			this.uptime = uptime;
			this.averageLatency = totalInferences > 0 ? (double) totalProcessingTime / totalInferences : 0.0;
			this.tokensPerSecond = totalProcessingTime > 0 ? (double) totalTokens * 1000 / totalProcessingTime : 0.0;
		}

		public long getTotalInferences() { return totalInferences; }
		public long getTotalTokens() { return totalTokens; }
		public long getTotalProcessingTime() { return totalProcessingTime; }
		public int getActiveModels() { return activeModels; }
		public double getAverageLatency() { return averageLatency; }
		public double getTokensPerSecond() { return tokensPerSecond; }
		public long getUptime() { return uptime; }

		@Override
		public String toString() {
			return String.format(
				"SystemMetrics{inferences:%d, tokens:%d, avgLatency:%.1fms, tokensPerSec:%.1f, activeModels:%d, uptime:%ds}",
				totalInferences, totalTokens, averageLatency, tokensPerSecond, activeModels, uptime / 1000
			);
		}
	}

	private static final long startTime = System.currentTimeMillis();

	/**
	 * Record the start of a model inference session.
	 */
	public static void recordModelCreated(String modelId) {
		if (modelId != null) {
			modelStatistics.computeIfAbsent(modelId, k -> new ModelStats());
			activeModels.incrementAndGet();
		}
	}

	/**
	 * Record the end of a model inference session.
	 */
	public static void recordModelDestroyed(String modelId) {
		if (modelId != null && modelStatistics.containsKey(modelId)) {
			activeModels.decrementAndGet();
		}
	}

	/**
	 * Record a successful inference operation.
	 */
	public static void recordInference(String modelId, int tokensGenerated, long latencyMs) {
		if (modelId != null && tokensGenerated >= 0 && latencyMs >= 0) {
			// Update global counters
			totalInferences.incrementAndGet();
			totalTokensGenerated.addAndGet(tokensGenerated);
			totalProcessingTimeMs.addAndGet(latencyMs);

			// Update model-specific stats
			ModelStats stats = modelStatistics.computeIfAbsent(modelId, k -> new ModelStats());
			stats.recordInference(tokensGenerated, latencyMs);
		}
	}

	/**
	 * Record an error during inference.
	 */
	public static void recordError(String modelId, String error) {
		if (modelId != null && error != null) {
			ModelStats stats = modelStatistics.computeIfAbsent(modelId, k -> new ModelStats());
			stats.recordError(error);
		}
	}

	/**
	 * Get statistics for a specific model.
	 */
	public static ModelStats getModelStats(String modelId) {
		return modelId != null ? modelStatistics.get(modelId) : null;
	}

	/**
	 * Get system-wide performance metrics.
	 */
	public static SystemMetrics getSystemMetrics() {
		long uptime = System.currentTimeMillis() - startTime;
		return new SystemMetrics(
			totalInferences.get(),
			totalTokensGenerated.get(),
			totalProcessingTimeMs.get(),
			activeModels.get(),
			uptime
		);
	}

	/**
	 * Get a status report.
	 */
	public static String getStatusReport() {
		SystemMetrics metrics = getSystemMetrics();
		StringBuilder report = new StringBuilder();

		report.append("=== Production Monitor Status Report ===\n");
		report.append("System: ").append(metrics).append("\n");
		report.append("\nModel Statistics:\n");

		if (modelStatistics.isEmpty()) {
			report.append("  No models currently tracked\n");
		} else {
			for (Map.Entry<String, ModelStats> entry : modelStatistics.entrySet()) {
				String modelId = entry.getKey();
				ModelStats stats = entry.getValue();

				long timeSinceLastUse = System.currentTimeMillis() - stats.getLastUsed();
				report.append(String.format(
					"  %s: inferences:%d, tokens:%d, avgLatency:%.1fms, tokensPerSec:%.1f, errors:%d, lastUsed:%ds ago\n",
					modelId, stats.getInferences(), stats.getTokensGenerated(),
					stats.getAverageLatencyMs(), stats.getTokensPerSecond(),
					stats.getErrors(), timeSinceLastUse / 1000
				));

				if (stats.getLastError() != null) {
					report.append("    Last error: ").append(stats.getLastError()).append("\n");
				}
			}
		}

		return report.toString();
	}

	/**
	 * Reset all statistics (useful for testing).
	 */
	public static void reset() {
		totalInferences.set(0);
		totalTokensGenerated.set(0);
		totalProcessingTimeMs.set(0);
		activeModels.set(0);
		modelStatistics.clear();
	}

	/**
	 * Check if the system is healthy based on error rates.
	 */
	public static boolean isHealthy() {
		SystemMetrics metrics = getSystemMetrics();
		if (metrics.getTotalInferences() == 0) {
			return true; // No operations yet, consider healthy
		}

		// Calculate global error rate
		long totalErrors = modelStatistics.values().stream()
			.mapToLong(ModelStats::getErrors)
			.sum();

		double errorRate = (double) totalErrors / metrics.getTotalInferences();
		return errorRate < 0.05; // Less than 5% error rate considered healthy
	}

	/**
	 * Get a simple health status string.
	 */
	public static String getHealthStatus() {
		SystemMetrics metrics = getSystemMetrics();
		if (metrics.getTotalInferences() == 0) {
			return "READY";
		}

		return isHealthy() ? "HEALTHY" : "DEGRADED";
	}
}
