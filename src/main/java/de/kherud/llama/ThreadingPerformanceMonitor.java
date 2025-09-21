package de.kherud.llama;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Performance monitoring for C++ threading optimization.
 * Tracks performance metrics to help optimize thread allocation over time.
 */
public class ThreadingPerformanceMonitor {
	private static final System.Logger logger = System.getLogger(ThreadingPerformanceMonitor.class.getName());
	private static final Map<String, PerformanceMetrics> metrics = new ConcurrentHashMap<>();
	private static final AtomicLong totalOperations = new AtomicLong(0);
	private static volatile boolean monitoringEnabled = false;

	/**
	 * Enable performance monitoring.
	 */
	public static void enableMonitoring() {
		monitoringEnabled = true;
		logger.log(DEBUG, "📊 Threading performance monitoring enabled");
	}

	/**
	 * Disable performance monitoring.
	 */
	public static void disableMonitoring() {
		monitoringEnabled = false;
		logger.log(DEBUG, "📊 Threading performance monitoring disabled");
	}

	/**
	 * Record the start of an operation.
	 *
	 * @param operationType The type of operation (completion, embedding, etc.)
	 * @param threadCount The number of threads being used
	 * @return Operation ID for tracking
	 */
	public static long recordOperationStart(String operationType, int threadCount) {
		if (!monitoringEnabled) return -1;

		long operationId = System.nanoTime();
		String key = operationType + "_" + threadCount;

		PerformanceMetrics metric = metrics.computeIfAbsent(key, k -> new PerformanceMetrics());
		metric.operationsStarted.incrementAndGet();
		totalOperations.incrementAndGet();

		return operationId;
	}

	/**
	 * Record the completion of an operation.
	 *
	 * @param operationType The type of operation
	 * @param threadCount The number of threads used
	 * @param operationId The operation ID from recordOperationStart
	 * @param tokensProcessed Number of tokens processed
	 */
	public static void recordOperationComplete(String operationType, int threadCount, long operationId, int tokensProcessed) {
		if (!monitoringEnabled || operationId == -1) return;

		long duration = System.nanoTime() - operationId;
		String key = operationType + "_" + threadCount;

		PerformanceMetrics metric = metrics.get(key);
		if (metric != null) {
			metric.operationsCompleted.incrementAndGet();
			metric.totalDurationNs.addAndGet(duration);
			metric.totalTokensProcessed.addAndGet(tokensProcessed);
		}
	}

	/**
	 * Get performance statistics.
	 *
	 * @return Performance summary
	 */
	public static PerformanceSummary getPerformanceStats() {
		if (!monitoringEnabled || metrics.isEmpty()) {
			return new PerformanceSummary("No performance data available");
		}

		StringBuilder report = new StringBuilder();
		report.append("\n📊 Threading Performance Report:\n");
		report.append("=====================================\n");

		String bestConfig = null;
		double bestThroughput = 0;

		for (Map.Entry<String, PerformanceMetrics> entry : metrics.entrySet()) {
			String config = entry.getKey();
			PerformanceMetrics metric = entry.getValue();

			long completed = metric.operationsCompleted.get();
			if (completed == 0) continue;

			double avgLatencyMs = (metric.totalDurationNs.get() / completed) / 1_000_000.0;
			double tokensPerSecond = (metric.totalTokensProcessed.get() * 1_000_000_000.0) / metric.totalDurationNs.get();

			report.append(String.format("Config: %s\n", config));
			report.append(String.format("  Operations: %d completed\n", completed));
			report.append(String.format("  Avg Latency: %.2f ms\n", avgLatencyMs));
			report.append(String.format("  Throughput: %.2f tokens/sec\n", tokensPerSecond));
			report.append("  ---\n");

			if (tokensPerSecond > bestThroughput) {
				bestThroughput = tokensPerSecond;
				bestConfig = config;
			}
		}

		if (bestConfig != null) {
			report.append(String.format("🏆 Best performing config: %s (%.2f tokens/sec)\n", bestConfig, bestThroughput));
		}

		report.append(String.format("\nTotal operations monitored: %d\n", totalOperations.get()));

		return new PerformanceSummary(report.toString());
	}

	/**
	 * Clear all performance metrics.
	 */
	public static void clearMetrics() {
		metrics.clear();
		totalOperations.set(0);
		logger.log(DEBUG, "📊 Performance metrics cleared");
	}

	/**
	 * Get recommended thread count based on performance data.
	 *
	 * @param operationType The operation type
	 * @return Recommended thread count, or -1 if no data available
	 */
	public static int getRecommendedThreadCount(String operationType) {
		if (!monitoringEnabled || metrics.isEmpty()) {
			return -1;
		}

		int bestThreadCount = -1;
		double bestThroughput = 0;

		for (Map.Entry<String, PerformanceMetrics> entry : metrics.entrySet()) {
			String config = entry.getKey();
			if (!config.startsWith(operationType + "_")) continue;

			PerformanceMetrics metric = entry.getValue();
			long completed = metric.operationsCompleted.get();
			if (completed < 3) continue; // Need at least 3 samples

			double tokensPerSecond = (metric.totalTokensProcessed.get() * 1_000_000_000.0) / metric.totalDurationNs.get();

			if (tokensPerSecond > bestThroughput) {
				bestThroughput = tokensPerSecond;
				String threadStr = config.substring(config.lastIndexOf('_') + 1);
				try {
					bestThreadCount = Integer.parseInt(threadStr);
				} catch (NumberFormatException e) {
					// Ignore malformed config names
				}
			}
		}

		return bestThreadCount;
	}

	/**
	 * Performance metrics for a specific configuration.
	 */
	private static class PerformanceMetrics {
		final AtomicLong operationsStarted = new AtomicLong(0);
		final AtomicLong operationsCompleted = new AtomicLong(0);
		final AtomicLong totalDurationNs = new AtomicLong(0);
		final AtomicLong totalTokensProcessed = new AtomicLong(0);
	}

	/**
	 * Performance summary report.
	 */
	public static class PerformanceSummary {
		public final String report;

		PerformanceSummary(String report) {
			this.report = report;
		}

		@Override
		public String toString() {
			return report;
		}
	}
}
