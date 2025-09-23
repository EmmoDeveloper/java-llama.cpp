package de.kherud.llama.benchmark;

import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;

/**
 * Unit tests for ServerBenchmark.
 */
public class ServerBenchmarkTest {

	@Rule
	public TemporaryFolder tempFolder = new TemporaryFolder();

	@Test
	public void testBenchmarkConfigBuilder() {
		ServerBenchmark.BenchmarkConfig config = new ServerBenchmark.BenchmarkConfig()
			.serverUrl("http://localhost:8080")
			.requests(50)
			.concurrency(2)
			.dataset("random")
			.promptLength(10, 20)
			.maxTokens(50)
			.timeout(Duration.ofMinutes(1))
			.verbose(true)
			.outputFile("results.json")
			.addGenerationParam("temperature", 0.8)
			.addGenerationParam("top_p", 0.9);

		Assert.assertNotNull(config);
	}

	@Test
	public void testRequestMetrics() {
		Instant start = Instant.now();
		Instant end = start.plusMillis(1000);

		ServerBenchmark.RequestMetrics metrics = new ServerBenchmark.RequestMetrics(
			"test-request", start, end, 10, 20, true, null
		);

		Assert.assertEquals("test-request", metrics.requestId);
		Assert.assertEquals(1000, metrics.durationMs);
		Assert.assertEquals(30, metrics.totalTokens);
		Assert.assertEquals(20.0, metrics.tokensPerSecond, 0.1);
		Assert.assertTrue(metrics.success);
		Assert.assertNull(metrics.error);
	}

	@Test
	public void testRequestMetricsFailure() {
		Instant start = Instant.now();
		Instant end = start.plusMillis(5000);

		ServerBenchmark.RequestMetrics metrics = new ServerBenchmark.RequestMetrics(
			"failed-request", start, end, 0, 0, false, "Timeout"
		);

		Assert.assertFalse(metrics.success);
		Assert.assertEquals("Timeout", metrics.error);
		Assert.assertEquals(0, metrics.totalTokens);
		Assert.assertEquals(0.0, metrics.tokensPerSecond, 0.0);
	}

	@Test
	public void testMetricsCollector() {
		ServerBenchmark.MetricsCollector collector = new ServerBenchmark.MetricsCollector();

		// Add successful request
		Instant now = Instant.now();
		collector.recordMetric(new ServerBenchmark.RequestMetrics(
			"req1", now, now.plusMillis(1000), 10, 20, true, null
		));

		// Add failed request
		collector.recordMetric(new ServerBenchmark.RequestMetrics(
			"req2", now, now.plusMillis(2000), 0, 0, false, "Error"
		));

		ServerBenchmark.BenchmarkResults results = collector.calculateResults();

		Assert.assertEquals(2, results.totalRequests);
		Assert.assertEquals(1, results.successfulRequests);
		Assert.assertEquals(1, results.failedRequests);
		Assert.assertEquals(50.0, results.successRate, 0.1);
		Assert.assertEquals(10, results.totalPromptTokens);
		Assert.assertEquals(20, results.totalCompletionTokens);
	}

	@Test
	public void testBenchmarkResultsFormatting() {
		ServerBenchmark.MetricsCollector collector = new ServerBenchmark.MetricsCollector();

		Instant base = Instant.now();
		for (int i = 0; i < 5; i++) {
			collector.recordMetric(new ServerBenchmark.RequestMetrics(
				"req" + i, base.plusMillis(i * 100), base.plusMillis(i * 100 + 200),
				10, 15, true, null
			));
		}

		ServerBenchmark.BenchmarkResults results = collector.calculateResults();
		String formatted = results.toFormattedString();

		Assert.assertTrue(formatted.contains("BENCHMARK RESULTS"));
		Assert.assertTrue(formatted.contains("Total Requests:        5"));
		Assert.assertTrue(formatted.contains("Successful:            5"));
		Assert.assertTrue(formatted.contains("P50:"));
		Assert.assertTrue(formatted.contains("P90:"));
		Assert.assertTrue(formatted.contains("P99:"));
	}

	@Test
	public void testBenchmarkResultsSaveToFile() throws IOException {
		ServerBenchmark.MetricsCollector collector = new ServerBenchmark.MetricsCollector();

		Instant now = Instant.now();
		collector.recordMetric(new ServerBenchmark.RequestMetrics(
			"req1", now, now.plusMillis(1000), 10, 20, true, null
		));

		ServerBenchmark.BenchmarkResults results = collector.calculateResults();

		Path outputFile = tempFolder.newFile("results.json").toPath();
		results.saveToFile(outputFile.toString());

		Assert.assertTrue(Files.exists(outputFile));
		String content = Files.readString(outputFile);
		Assert.assertTrue(content.contains("totalRequests"));
		Assert.assertTrue(content.contains("successfulRequests"));
		Assert.assertTrue(content.contains("requestsPerSecond"));
	}

	@Test
	public void testPercentileCalculation() {
		ServerBenchmark.MetricsCollector collector = new ServerBenchmark.MetricsCollector();

		Instant base = Instant.now();
		// Add requests with latencies: 100, 200, 300, 400, 500 ms
		for (int i = 1; i <= 5; i++) {
			collector.recordMetric(new ServerBenchmark.RequestMetrics(
				"req" + i, base, base.plusMillis(i * 100),
				10, 10, true, null
			));
		}

		ServerBenchmark.BenchmarkResults results = collector.calculateResults();

		Assert.assertEquals(300.0, results.p50LatencyMs, 0.1); // Median
		Assert.assertEquals(500.0, results.p90LatencyMs, 0.1); // 90th percentile
		Assert.assertEquals(500.0, results.p99LatencyMs, 0.1); // 99th percentile
		Assert.assertEquals(100.0, results.minLatencyMs, 0.1);
		Assert.assertEquals(500.0, results.maxLatencyMs, 0.1);
	}

	@Test
	public void testEmptyMetricsHandling() {
		ServerBenchmark.MetricsCollector collector = new ServerBenchmark.MetricsCollector();
		ServerBenchmark.BenchmarkResults results = collector.calculateResults();

		Assert.assertEquals(0, results.totalRequests);
		Assert.assertEquals(0, results.successfulRequests);
		Assert.assertEquals(0, results.failedRequests);
		Assert.assertEquals(0.0, results.successRate, 0.0);
		Assert.assertEquals(0.0, results.requestsPerSecond, 0.0);
		Assert.assertEquals(0.0, results.avgLatencyMs, 0.0);
	}

	@Test
	public void testTokenSpeedCalculation() {
		ServerBenchmark.MetricsCollector collector = new ServerBenchmark.MetricsCollector();

		Instant now = Instant.now();
		// 20 tokens in 1 second = 20 tokens/sec
		collector.recordMetric(new ServerBenchmark.RequestMetrics(
			"req1", now, now.plusMillis(1000), 10, 20, true, null
		));

		// 30 tokens in 2 seconds = 15 tokens/sec
		collector.recordMetric(new ServerBenchmark.RequestMetrics(
			"req2", now, now.plusMillis(2000), 10, 30, true, null
		));

		ServerBenchmark.BenchmarkResults results = collector.calculateResults();

		// Average should be (20 + 15) / 2 = 17.5
		Assert.assertEquals(17.5, results.avgTokensPerSecond, 0.1);
		Assert.assertEquals(50, results.totalCompletionTokens);
	}

	@Test
	public void testBenchmarkCreationWithInvalidServer() {
		ServerBenchmark.BenchmarkConfig config = new ServerBenchmark.BenchmarkConfig()
			.serverUrl("http://nonexistent:9999")
			.requests(1);

		try {
			ServerBenchmark benchmark = new ServerBenchmark(config);
			// Creation should succeed
			Assert.assertNotNull(benchmark);
		} catch (IOException e) {
			Assert.fail("Benchmark creation should not fail until run() is called");
		}
	}


	@Test
	public void testDatasetOptions() throws IOException {
		// Test random dataset
		ServerBenchmark.BenchmarkConfig config1 = new ServerBenchmark.BenchmarkConfig()
			.dataset("random")
			.requests(5);

		try {
			ServerBenchmark benchmark1 = new ServerBenchmark(config1);
			Assert.assertNotNull(benchmark1);
		} catch (IOException e) {
			Assert.fail("Random dataset should work");
		}

		// Test shakespeare dataset
		ServerBenchmark.BenchmarkConfig config2 = new ServerBenchmark.BenchmarkConfig()
			.dataset("shakespeare")
			.requests(3);

		try {
			ServerBenchmark benchmark2 = new ServerBenchmark(config2);
			Assert.assertNotNull(benchmark2);
		} catch (IOException e) {
			Assert.fail("Shakespeare dataset should work");
		}

		// Test file dataset with nonexistent file
		ServerBenchmark.BenchmarkConfig config3 = new ServerBenchmark.BenchmarkConfig()
			.dataset("nonexistent-file.txt")
			.requests(1);

		try {
			ServerBenchmark benchmark3 = new ServerBenchmark(config3);
			Assert.fail("Should fail with nonexistent file dataset");
		} catch (IOException e) {
			Assert.assertTrue(e.getMessage().contains("Prompt file not found"));
		}
	}

}
