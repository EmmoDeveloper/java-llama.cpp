package de.kherud.llama;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class SystemMonitorTest {
	private static final System.Logger logger = System.getLogger(SystemMonitorTest.class.getName());

	@Before
	public void setUp() {
		// Reset monitor state before each test
		SystemMonitor.reset();
	}

	@Test
	public void testBasicMetricsCollection() {
		logger.log(DEBUG, "\n=== Basic Metrics Collection Test ===");

		// Initially empty
		SystemMonitor.SystemMetrics initial = SystemMonitor.getSystemMetrics();
		Assert.assertEquals("No inferences initially", 0, initial.getTotalInferences());
		Assert.assertEquals("No tokens initially", 0, initial.getTotalTokens());
		Assert.assertEquals("No active models initially", 0, initial.getActiveModels());

		// Create a model and record activity
		SystemMonitor.recordModelCreated("test-model-1");
		Assert.assertEquals("One active model", 1, SystemMonitor.getSystemMetrics().getActiveModels());

		// Record some inferences
		SystemMonitor.recordInference("test-model-1", 50, 100); // 50 tokens in 100ms
		SystemMonitor.recordInference("test-model-1", 30, 80);  // 30 tokens in 80ms

		SystemMonitor.SystemMetrics metrics = SystemMonitor.getSystemMetrics();
		Assert.assertEquals("Two inferences recorded", 2, metrics.getTotalInferences());
		Assert.assertEquals("80 total tokens", 80, metrics.getTotalTokens());
		Assert.assertEquals("180ms total processing time", 180, metrics.getTotalProcessingTime());

		double expectedAvgLatency = 180.0 / 2; // 90ms average
		Assert.assertEquals("Average latency", expectedAvgLatency, metrics.getAverageLatency(), 0.1);

		double expectedTokensPerSec = 80.0 * 1000 / 180; // ~444 tokens/sec
		Assert.assertEquals("Tokens per second", expectedTokensPerSec, metrics.getTokensPerSecond(), 1.0);

		logger.log(DEBUG, "Final metrics: " + metrics);
		logger.log(DEBUG, "✅ Basic metrics collection test passed!");
	}

	@Test
	public void testModelSpecificStatistics() {
		logger.log(DEBUG, "\n=== Model-Specific Statistics Test ===");

		// Create multiple models
		SystemMonitor.recordModelCreated("model-fast");
		SystemMonitor.recordModelCreated("model-slow");

		// Record different performance patterns
		SystemMonitor.recordInference("model-fast", 10, 20); // Fast model: 500 tokens/sec
		SystemMonitor.recordInference("model-fast", 15, 30); // Fast model: 500 tokens/sec

		SystemMonitor.recordInference("model-slow", 20, 200); // Slow model: 100 tokens/sec
		SystemMonitor.recordInference("model-slow", 25, 250); // Slow model: 100 tokens/sec

		// Check model-specific stats
		SystemMonitor.ModelStats fastStats = SystemMonitor.getModelStats("model-fast");
		SystemMonitor.ModelStats slowStats = SystemMonitor.getModelStats("model-slow");

		Assert.assertNotNull("Fast model stats exist", fastStats);
		Assert.assertNotNull("Slow model stats exist", slowStats);

		Assert.assertEquals("Fast model inferences", 2, fastStats.getInferences());
		Assert.assertEquals("Fast model tokens", 25, fastStats.getTokensGenerated());
		Assert.assertEquals("Fast model avg latency", 25.0, fastStats.getAverageLatencyMs(), 0.1);

		Assert.assertEquals("Slow model inferences", 2, slowStats.getInferences());
		Assert.assertEquals("Slow model tokens", 45, slowStats.getTokensGenerated());
		Assert.assertEquals("Slow model avg latency", 225.0, slowStats.getAverageLatencyMs(), 0.1);

		// Verify fast model is indeed faster
		Assert.assertTrue("Fast model should have higher tokens/sec",
			fastStats.getTokensPerSecond() > slowStats.getTokensPerSecond());

		logger.log(DEBUG, "Fast model: " + fastStats.getTokensPerSecond() + " tokens/sec");
		logger.log(DEBUG, "Slow model: " + slowStats.getTokensPerSecond() + " tokens/sec");
		logger.log(DEBUG, "✅ Model-specific statistics test passed!");
	}

	@Test
	public void testErrorTracking() {
		logger.log(DEBUG, "\n=== Error Tracking Test ===");

		SystemMonitor.recordModelCreated("error-prone-model");

		// Record successful operations
		SystemMonitor.recordInference("error-prone-model", 10, 50);
		SystemMonitor.recordInference("error-prone-model", 12, 60);

		// Record errors
		SystemMonitor.recordError("error-prone-model", "Out of memory");
		SystemMonitor.recordError("error-prone-model", "Model file corrupted");

		SystemMonitor.ModelStats stats = SystemMonitor.getModelStats("error-prone-model");
		Assert.assertNotNull("Model stats exist", stats);
		Assert.assertEquals("Two errors recorded", 2, stats.getErrors());
		Assert.assertEquals("Last error message", "Model file corrupted", stats.getLastError());

		// System should be considered unhealthy with high error rate (2 errors / 2 successes = 50%)
		Assert.assertFalse("System should be unhealthy with high error rate", SystemMonitor.isHealthy());
		Assert.assertEquals("Health status should be DEGRADED", "DEGRADED", SystemMonitor.getHealthStatus());

		logger.log(DEBUG, "Health status: " + SystemMonitor.getHealthStatus());
		logger.log(DEBUG, "✅ Error tracking test passed!");
	}

	@Test
	public void testHealthySystem() {
		logger.log(DEBUG, "\n=== Healthy System Test ===");

		SystemMonitor.recordModelCreated("reliable-model");

		// Record many successful operations with very few errors
		for (int i = 0; i < 100; i++) {
			SystemMonitor.recordInference("reliable-model", 10 + i % 5, 50 + i % 20);
		}

		// Add a few errors (but less than 5% error rate)
		SystemMonitor.recordError("reliable-model", "Rare network timeout");
		SystemMonitor.recordError("reliable-model", "Temporary resource unavailable");

		// 2 errors out of 100 operations = 2% error rate, should be healthy
		Assert.assertTrue("System should be healthy with low error rate", SystemMonitor.isHealthy());
		Assert.assertEquals("Health status should be HEALTHY", "HEALTHY", SystemMonitor.getHealthStatus());

		SystemMonitor.SystemMetrics metrics = SystemMonitor.getSystemMetrics();
		Assert.assertEquals("100 inferences", 100, metrics.getTotalInferences());
		Assert.assertTrue("Should have generated many tokens", metrics.getTotalTokens() > 1000);

		logger.log(DEBUG, "Health status: " + SystemMonitor.getHealthStatus());
		logger.log(DEBUG, "System metrics: " + metrics);
		logger.log(DEBUG, "✅ Healthy system test passed!");
	}

	@Test
	public void testStatusReport() {
		logger.log(DEBUG, "\n=== Status Report Test ===");

		// Create a realistic scenario
		SystemMonitor.recordModelCreated("chatbot-model");
		SystemMonitor.recordModelCreated("code-completion-model");

		// Simulate chatbot activity
		SystemMonitor.recordInference("chatbot-model", 45, 200);
		SystemMonitor.recordInference("chatbot-model", 52, 230);
		SystemMonitor.recordInference("chatbot-model", 38, 180);

		// Simulate code completion activity
		SystemMonitor.recordInference("code-completion-model", 15, 50);
		SystemMonitor.recordInference("code-completion-model", 12, 40);
		SystemMonitor.recordError("code-completion-model", "Syntax error in prompt");

		// Generate and verify status report
		String report = SystemMonitor.getStatusReport();
		logger.log(DEBUG, "\nGenerated Status Report:");
		logger.log(DEBUG, report);

		Assert.assertTrue("Report contains system info", report.contains("System:"));
		Assert.assertTrue("Report contains model statistics", report.contains("Model Statistics:"));
		Assert.assertTrue("Report contains chatbot model", report.contains("chatbot-model"));
		Assert.assertTrue("Report contains code completion model", report.contains("code-completion-model"));
		Assert.assertTrue("Report contains error info", report.contains("Last error: Syntax error in prompt"));

		logger.log(DEBUG, "✅ Status report test passed!");
	}

	@Test
	public void testModelLifecycle() {
		logger.log(DEBUG, "\n=== Model Lifecycle Test ===");

		// Test model creation and destruction tracking
		Assert.assertEquals("Initially no active models", 0, SystemMonitor.getSystemMetrics().getActiveModels());

		SystemMonitor.recordModelCreated("temp-model-1");
		SystemMonitor.recordModelCreated("temp-model-2");
		SystemMonitor.recordModelCreated("temp-model-3");
		Assert.assertEquals("Three active models", 3, SystemMonitor.getSystemMetrics().getActiveModels());

		SystemMonitor.recordModelDestroyed("temp-model-1");
		Assert.assertEquals("Two active models after destruction", 2, SystemMonitor.getSystemMetrics().getActiveModels());

		SystemMonitor.recordModelDestroyed("temp-model-2");
		SystemMonitor.recordModelDestroyed("temp-model-3");
		Assert.assertEquals("No active models after all destroyed", 0, SystemMonitor.getSystemMetrics().getActiveModels());

		logger.log(DEBUG, "✅ Model lifecycle test passed!");
	}
}
