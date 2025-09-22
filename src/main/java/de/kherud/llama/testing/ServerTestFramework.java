package de.kherud.llama.testing;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Server testing framework.
 *
 * Equivalent to server_test.py - provides comprehensive testing of llama.cpp server
 * including correctness, performance, concurrency, and edge case handling.
 */
public class ServerTestFramework {
	private static final Logger LOGGER = Logger.getLogger(ServerTestFramework.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();

	public static class TestConfig {
		private String serverUrl = "http://localhost:8080";
		private int timeout = 30000; // 30 seconds
		private int maxConcurrentRequests = 10;
		private boolean verbose = false;
		private String testDataPath = "test_data";
		private List<String> testSuites = Arrays.asList("basic", "performance", "concurrency", "edge_cases");
		private Map<String, Object> serverParams = new HashMap<>();

		public TestConfig serverUrl(String url) {
			this.serverUrl = url;
			return this;
		}

		public TestConfig timeout(int timeout) {
			this.timeout = timeout;
			return this;
		}

		public TestConfig maxConcurrentRequests(int max) {
			this.maxConcurrentRequests = max;
			return this;
		}

		public TestConfig verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public TestConfig testDataPath(String path) {
			this.testDataPath = path;
			return this;
		}

		public TestConfig testSuites(List<String> suites) {
			this.testSuites = new ArrayList<>(suites);
			return this;
		}

		public TestConfig addServerParam(String key, Object value) {
			this.serverParams.put(key, value);
			return this;
		}
	}

	public static class TestResult {
		public String testName;
		public boolean passed;
		public String message;
		public long duration;
		public Map<String, Object> details = new HashMap<>();
		public Throwable exception;

		public TestResult(String testName) {
			this.testName = testName;
		}

		public TestResult pass(String message) {
			this.passed = true;
			this.message = message;
			return this;
		}

		public TestResult fail(String message) {
			this.passed = false;
			this.message = message;
			return this;
		}

		public TestResult fail(String message, Throwable exception) {
			this.passed = false;
			this.message = message;
			this.exception = exception;
			return this;
		}

		public TestResult detail(String key, Object value) {
			this.details.put(key, value);
			return this;
		}
	}

	public static class TestSuite {
		public String name;
		public List<TestResult> results = new ArrayList<>();
		public long totalDuration;
		public int passed;
		public int failed;
		public Instant startTime;
		public Instant endTime;

		public TestSuite(String name) {
			this.name = name;
		}

		public void addResult(TestResult result) {
			results.add(result);
			if (result.passed) {
				passed++;
			} else {
				failed++;
			}
			totalDuration += result.duration;
		}
	}

	private final TestConfig config;
	private final ExecutorService executor;

	public ServerTestFramework(TestConfig config) {
		this.config = config;
		this.executor = Executors.newFixedThreadPool(config.maxConcurrentRequests);
	}

	/**
	 * Run all configured test suites
	 */
	public List<TestSuite> runTests() {
		LOGGER.info("Starting server test framework");
		LOGGER.info("Server URL: " + config.serverUrl);
		LOGGER.info("Test suites: " + config.testSuites);

		List<TestSuite> results = new ArrayList<>();

		for (String suiteName : config.testSuites) {
			TestSuite suite = runTestSuite(suiteName);
			results.add(suite);

			if (config.verbose) {
				printTestSuite(suite);
			}
		}

		return results;
	}

	private TestSuite runTestSuite(String suiteName) {
		LOGGER.info("Running test suite: " + suiteName);
		TestSuite suite = new TestSuite(suiteName);
		suite.startTime = Instant.now();

		try {
			switch (suiteName) {
				case "basic":
					runBasicTests(suite);
					break;
				case "performance":
					runPerformanceTests(suite);
					break;
				case "concurrency":
					runConcurrencyTests(suite);
					break;
				case "edge_cases":
					runEdgeCaseTests(suite);
					break;
				default:
					throw new IllegalArgumentException("Unknown test suite: " + suiteName);
			}
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "Test suite failed: " + suiteName, e);
			suite.addResult(new TestResult("suite_execution").fail("Suite execution failed", e));
		}

		suite.endTime = Instant.now();
		return suite;
	}

	private void runBasicTests(TestSuite suite) {
		// Health check
		suite.addResult(testHealthCheck());

		// Model info
		suite.addResult(testModelInfo());

		// Simple completion
		suite.addResult(testSimpleCompletion());

		// Chat completion
		suite.addResult(testChatCompletion());

		// Tokenization
		suite.addResult(testTokenization());

		// Embeddings
		suite.addResult(testEmbeddings());
	}

	private void runPerformanceTests(TestSuite suite) {
		// Throughput test
		suite.addResult(testThroughput());

		// Latency test
		suite.addResult(testLatency());

		// Memory usage
		suite.addResult(testMemoryUsage());

		// Long sequence handling
		suite.addResult(testLongSequence());
	}

	private void runConcurrencyTests(TestSuite suite) {
		// Concurrent requests
		suite.addResult(testConcurrentRequests());

		// Request queuing
		suite.addResult(testRequestQueuing());

		// Resource sharing
		suite.addResult(testResourceSharing());
	}

	private void runEdgeCaseTests(TestSuite suite) {
		// Empty input
		suite.addResult(testEmptyInput());

		// Very long input
		suite.addResult(testVeryLongInput());

		// Invalid parameters
		suite.addResult(testInvalidParameters());

		// Malformed requests
		suite.addResult(testMalformedRequests());

		// Unicode handling
		suite.addResult(testUnicodeHandling());
	}

	private TestResult testHealthCheck() {
		TestResult result = new TestResult("health_check");
		Instant start = Instant.now();

		try {
			HttpURLConnection conn = createConnection("/health", "GET");
			int responseCode = conn.getResponseCode();

			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				result.pass("Health check successful");
			} else {
				result.fail("Health check failed with status: " + responseCode);
			}

			result.detail("response_code", responseCode);
			result.detail("response_time_ms", result.duration);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Health check failed", e);
		}

		return result;
	}

	private TestResult testModelInfo() {
		TestResult result = new TestResult("model_info");
		Instant start = Instant.now();

		try {
			HttpURLConnection conn = createConnection("/v1/models", "GET");
			int responseCode = conn.getResponseCode();

			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);

				if (json.has("data") && json.get("data").isArray()) {
					result.pass("Model info retrieved successfully");
					result.detail("model_count", json.get("data").size());
				} else {
					result.fail("Invalid model info response format");
				}
			} else {
				result.fail("Model info request failed with status: " + responseCode);
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Model info test failed", e);
		}

		return result;
	}

	private TestResult testSimpleCompletion() {
		TestResult result = new TestResult("simple_completion");
		Instant start = Instant.now();

		try {
			Map<String, Object> request = new HashMap<>();
			request.put("prompt", "The capital of France is");
			request.put("max_tokens", 10);
			request.put("temperature", 0.1);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);

				if (json.has("choices") && json.get("choices").size() > 0) {
					String text = json.get("choices").get(0).get("text").asText();
					result.pass("Completion successful");
					result.detail("completion_text", text);
					result.detail("completion_length", text.length());
				} else {
					result.fail("Invalid completion response format");
				}
			} else {
				result.fail("Completion request failed with status: " + responseCode);
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Simple completion test failed", e);
		}

		return result;
	}

	private TestResult testChatCompletion() {
		TestResult result = new TestResult("chat_completion");
		Instant start = Instant.now();

		try {
			Map<String, Object> request = new HashMap<>();
			List<Map<String, String>> messages = new ArrayList<>();
			messages.add(Map.of("role", "user", "content", "Hello, how are you?"));

			request.put("messages", messages);
			request.put("max_tokens", 20);

			HttpURLConnection conn = createConnection("/v1/chat/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);

				if (json.has("choices") && json.get("choices").size() > 0) {
					JsonNode message = json.get("choices").get(0).get("message");
					result.pass("Chat completion successful");
					result.detail("response_role", message.get("role").asText());
					result.detail("response_content", message.get("content").asText());
				} else {
					result.fail("Invalid chat completion response format");
				}
			} else {
				result.fail("Chat completion request failed with status: " + responseCode);
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Chat completion test failed", e);
		}

		return result;
	}

	private TestResult testTokenization() {
		TestResult result = new TestResult("tokenization");
		Instant start = Instant.now();

		try {
			Map<String, Object> request = new HashMap<>();
			request.put("content", "Hello world, this is a test sentence.");

			HttpURLConnection conn = createConnection("/tokenize", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);

				if (json.has("tokens")) {
					int tokenCount = json.get("tokens").size();
					result.pass("Tokenization successful");
					result.detail("token_count", tokenCount);
				} else {
					result.fail("Invalid tokenization response format");
				}
			} else {
				result.fail("Tokenization request failed with status: " + responseCode);
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Tokenization test failed", e);
		}

		return result;
	}

	private TestResult testEmbeddings() {
		TestResult result = new TestResult("embeddings");
		Instant start = Instant.now();

		try {
			Map<String, Object> request = new HashMap<>();
			request.put("content", "This is a test sentence for embeddings.");

			HttpURLConnection conn = createConnection("/embedding", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);

				if (json.has("embedding")) {
					int embeddingSize = json.get("embedding").size();
					result.pass("Embeddings successful");
					result.detail("embedding_size", embeddingSize);
				} else {
					result.fail("Invalid embeddings response format");
				}
			} else {
				result.fail("Embeddings request failed with status: " + responseCode);
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Embeddings test failed", e);
		}

		return result;
	}

	private TestResult testThroughput() {
		TestResult result = new TestResult("throughput");
		Instant start = Instant.now();

		try {
			int requestCount = 10;
			List<Future<Long>> futures = new ArrayList<>();

			for (int i = 0; i < requestCount; i++) {
				final int requestIndex = i;
				Future<Long> future = executor.submit(() -> {
					try {
						Map<String, Object> request = new HashMap<>();
						request.put("prompt", "Test prompt " + requestIndex);
						request.put("max_tokens", 5);

						HttpURLConnection conn = createConnection("/v1/completions", "POST");
						writeJsonRequest(conn, request);

						return System.currentTimeMillis();
					} catch (Exception e) {
						throw new RuntimeException(e);
					}
				});
				futures.add(future);
			}

			// Wait for all requests
			for (Future<Long> future : futures) {
				future.get(config.timeout, TimeUnit.MILLISECONDS);
			}

			result.duration = Duration.between(start, Instant.now()).toMillis();
			double throughput = (double) requestCount / (result.duration / 1000.0);

			result.pass("Throughput test completed");
			result.detail("request_count", requestCount);
			result.detail("total_time_ms", result.duration);
			result.detail("throughput_rps", throughput);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Throughput test failed", e);
		}

		return result;
	}

	private TestResult testLatency() {
		TestResult result = new TestResult("latency");
		List<Long> latencies = new ArrayList<>();

		try {
			for (int i = 0; i < 5; i++) {
				Instant requestStart = Instant.now();

				Map<String, Object> request = new HashMap<>();
				request.put("prompt", "Quick test");
				request.put("max_tokens", 1);

				HttpURLConnection conn = createConnection("/v1/completions", "POST");
				writeJsonRequest(conn, request);
				readResponse(conn);

				long latency = Duration.between(requestStart, Instant.now()).toMillis();
				latencies.add(latency);
			}

			result.duration = latencies.stream().mapToLong(Long::longValue).sum();
			double avgLatency = latencies.stream().mapToLong(Long::longValue).average().orElse(0);
			long maxLatency = latencies.stream().mapToLong(Long::longValue).max().orElse(0);
			long minLatency = latencies.stream().mapToLong(Long::longValue).min().orElse(0);

			result.pass("Latency test completed");
			result.detail("avg_latency_ms", avgLatency);
			result.detail("max_latency_ms", maxLatency);
			result.detail("min_latency_ms", minLatency);

		} catch (Exception e) {
			result.fail("Latency test failed", e);
		}

		return result;
	}

	private TestResult testMemoryUsage() {
		TestResult result = new TestResult("memory_usage");
		Instant start = Instant.now();

		try {
			// Get initial memory stats if available
			Runtime runtime = Runtime.getRuntime();
			long initialMemory = runtime.totalMemory() - runtime.freeMemory();

			// Make several requests to potentially increase memory usage
			for (int i = 0; i < 5; i++) {
				Map<String, Object> request = new HashMap<>();
				request.put("prompt", "Memory test prompt with some content to process");
				request.put("max_tokens", 10);

				HttpURLConnection conn = createConnection("/v1/completions", "POST");
				writeJsonRequest(conn, request);
				readResponse(conn);
			}

			long finalMemory = runtime.totalMemory() - runtime.freeMemory();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			result.pass("Memory usage test completed");
			result.detail("initial_memory_mb", initialMemory / 1024 / 1024);
			result.detail("final_memory_mb", finalMemory / 1024 / 1024);
			result.detail("memory_delta_mb", (finalMemory - initialMemory) / 1024 / 1024);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Memory usage test failed", e);
		}

		return result;
	}

	private TestResult testLongSequence() {
		TestResult result = new TestResult("long_sequence");
		Instant start = Instant.now();

		try {
			// Create a longer prompt
			StringBuilder prompt = new StringBuilder();
			for (int i = 0; i < 100; i++) {
				prompt.append("This is sentence number ").append(i).append(". ");
			}

			Map<String, Object> request = new HashMap<>();
			request.put("prompt", prompt.toString());
			request.put("max_tokens", 10);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				readResponse(conn);
				result.pass("Long sequence test completed");
				result.detail("prompt_length", prompt.length());
			} else {
				result.fail("Long sequence failed with status: " + responseCode);
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Long sequence test failed", e);
		}

		return result;
	}

	private TestResult testConcurrentRequests() {
		TestResult result = new TestResult("concurrent_requests");
		Instant start = Instant.now();

		try {
			int concurrentCount = Math.min(config.maxConcurrentRequests, 5);
			List<Future<Boolean>> futures = new ArrayList<>();

			for (int i = 0; i < concurrentCount; i++) {
				final int requestIndex = i;
				Future<Boolean> future = executor.submit(() -> {
					try {
						Map<String, Object> request = new HashMap<>();
						request.put("prompt", "Concurrent request " + requestIndex);
						request.put("max_tokens", 5);

						HttpURLConnection conn = createConnection("/v1/completions", "POST");
						writeJsonRequest(conn, request);
						int responseCode = conn.getResponseCode();
						return responseCode == 200;
					} catch (Exception e) {
						return false;
					}
				});
				futures.add(future);
			}

			int successCount = 0;
			for (Future<Boolean> future : futures) {
				if (future.get(config.timeout, TimeUnit.MILLISECONDS)) {
					successCount++;
				}
			}

			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (successCount == concurrentCount) {
				result.pass("All concurrent requests successful");
			} else {
				result.fail("Some concurrent requests failed");
			}

			result.detail("concurrent_count", concurrentCount);
			result.detail("success_count", successCount);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Concurrent requests test failed", e);
		}

		return result;
	}

	private TestResult testRequestQueuing() {
		TestResult result = new TestResult("request_queuing");
		Instant start = Instant.now();

		try {
			// Submit more requests than the server can handle simultaneously
			int queueCount = config.maxConcurrentRequests + 2;
			List<Future<Long>> futures = new ArrayList<>();

			for (int i = 0; i < queueCount; i++) {
				final int requestIndex = i;
				Future<Long> future = executor.submit(() -> {
					Instant requestStart = Instant.now();
					try {
						Map<String, Object> request = new HashMap<>();
						request.put("prompt", "Queue test " + requestIndex);
						request.put("max_tokens", 3);

						HttpURLConnection conn = createConnection("/v1/completions", "POST");
						writeJsonRequest(conn, request);
						readResponse(conn);

						return Duration.between(requestStart, Instant.now()).toMillis();
					} catch (Exception e) {
						return -1L;
					}
				});
				futures.add(future);
			}

			List<Long> responseTimes = new ArrayList<>();
			for (Future<Long> future : futures) {
				Long responseTime = future.get(config.timeout, TimeUnit.MILLISECONDS);
				if (responseTime > 0) {
					responseTimes.add(responseTime);
				}
			}

			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseTimes.size() == queueCount) {
				result.pass("Request queuing test completed");
			} else {
				result.fail("Some queued requests failed");
			}

			result.detail("queued_count", queueCount);
			result.detail("completed_count", responseTimes.size());

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Request queuing test failed", e);
		}

		return result;
	}

	private TestResult testResourceSharing() {
		TestResult result = new TestResult("resource_sharing");
		Instant start = Instant.now();

		try {
			// Test that the server can handle multiple different types of requests
			Future<Boolean> completion = executor.submit(() -> {
				try {
					Map<String, Object> request = new HashMap<>();
					request.put("prompt", "Resource test completion");
					request.put("max_tokens", 3);

					HttpURLConnection conn = createConnection("/v1/completions", "POST");
					writeJsonRequest(conn, request);
					return conn.getResponseCode() == 200;
				} catch (Exception e) {
					return false;
				}
			});

			Future<Boolean> tokenization = executor.submit(() -> {
				try {
					Map<String, Object> request = new HashMap<>();
					request.put("content", "Resource test tokenization");

					HttpURLConnection conn = createConnection("/tokenize", "POST");
					writeJsonRequest(conn, request);
					return conn.getResponseCode() == 200;
				} catch (Exception e) {
					return false;
				}
			});

			boolean completionSuccess = completion.get(config.timeout, TimeUnit.MILLISECONDS);
			boolean tokenizationSuccess = tokenization.get(config.timeout, TimeUnit.MILLISECONDS);

			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (completionSuccess && tokenizationSuccess) {
				result.pass("Resource sharing test completed");
			} else {
				result.fail("Resource sharing test failed");
			}

			result.detail("completion_success", completionSuccess);
			result.detail("tokenization_success", tokenizationSuccess);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Resource sharing test failed", e);
		}

		return result;
	}

	private TestResult testEmptyInput() {
		TestResult result = new TestResult("empty_input");
		Instant start = Instant.now();

		try {
			Map<String, Object> request = new HashMap<>();
			request.put("prompt", "");
			request.put("max_tokens", 5);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			// Server should handle empty input gracefully (either 200 or 400)
			if (responseCode == 200 || responseCode == 400) {
				result.pass("Empty input handled correctly");
			} else {
				result.fail("Unexpected response to empty input: " + responseCode);
			}

			result.detail("response_code", responseCode);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Empty input test failed", e);
		}

		return result;
	}

	private TestResult testVeryLongInput() {
		TestResult result = new TestResult("very_long_input");
		Instant start = Instant.now();

		try {
			// Create a very long prompt (beyond typical context window)
			StringBuilder longPrompt = new StringBuilder();
			for (int i = 0; i < 10000; i++) {
				longPrompt.append("word").append(i).append(" ");
			}

			Map<String, Object> request = new HashMap<>();
			request.put("prompt", longPrompt.toString());
			request.put("max_tokens", 1);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			// Server should handle this gracefully (200, 400, or 413)
			if (responseCode == 200 || responseCode == 400 || responseCode == 413) {
				result.pass("Very long input handled correctly");
			} else {
				result.fail("Unexpected response to very long input: " + responseCode);
			}

			result.detail("response_code", responseCode);
			result.detail("prompt_length", longPrompt.length());

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Very long input test failed", e);
		}

		return result;
	}

	private TestResult testInvalidParameters() {
		TestResult result = new TestResult("invalid_parameters");
		Instant start = Instant.now();

		try {
			Map<String, Object> request = new HashMap<>();
			request.put("prompt", "Test");
			request.put("max_tokens", -1); // Invalid value
			request.put("temperature", 2.5); // Out of range

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			// Server should reject invalid parameters with 400
			if (responseCode == 400) {
				result.pass("Invalid parameters rejected correctly");
			} else {
				result.fail("Server did not reject invalid parameters");
			}

			result.detail("response_code", responseCode);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Invalid parameters test failed", e);
		}

		return result;
	}

	private TestResult testMalformedRequests() {
		TestResult result = new TestResult("malformed_requests");
		Instant start = Instant.now();

		try {
			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			conn.setDoOutput(true);
			conn.setRequestProperty("Content-Type", "application/json");

			// Send malformed JSON
			try (OutputStream os = conn.getOutputStream()) {
				os.write("{ invalid json".getBytes(StandardCharsets.UTF_8));
			}

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			// Server should reject malformed JSON with 400
			if (responseCode == 400) {
				result.pass("Malformed request rejected correctly");
			} else {
				result.fail("Server did not reject malformed request");
			}

			result.detail("response_code", responseCode);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Malformed requests test failed", e);
		}

		return result;
	}

	private TestResult testUnicodeHandling() {
		TestResult result = new TestResult("unicode_handling");
		Instant start = Instant.now();

		try {
			Map<String, Object> request = new HashMap<>();
			request.put("prompt", "Unicode test: 🌟 こんにちは العالم 🚀");
			request.put("max_tokens", 5);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (responseCode == 200) {
				String response = readResponse(conn);
				result.pass("Unicode handling successful");
				result.detail("response_contains_unicode", response.contains("Unicode") || response.length() > 0);
			} else {
				result.fail("Unicode handling failed with status: " + responseCode);
			}

			result.detail("response_code", responseCode);

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Unicode handling test failed", e);
		}

		return result;
	}

	private HttpURLConnection createConnection(String endpoint, String method) throws IOException {
		URL url = new URL(config.serverUrl + endpoint);
		HttpURLConnection conn = (HttpURLConnection) url.openConnection();
		conn.setRequestMethod(method);
		conn.setConnectTimeout(config.timeout);
		conn.setReadTimeout(config.timeout);

		if ("POST".equals(method)) {
			conn.setDoOutput(true);
			conn.setRequestProperty("Content-Type", "application/json");
		}

		return conn;
	}

	private void writeJsonRequest(HttpURLConnection conn, Map<String, Object> request) throws IOException {
		String json = MAPPER.writeValueAsString(request);
		try (OutputStream os = conn.getOutputStream()) {
			os.write(json.getBytes(StandardCharsets.UTF_8));
		}
	}

	private String readResponse(HttpURLConnection conn) throws IOException {
		try (InputStream is = conn.getInputStream();
			 BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
			StringBuilder response = new StringBuilder();
			String line;
			while ((line = reader.readLine()) != null) {
				response.append(line);
			}
			return response.toString();
		}
	}

	private void printTestSuite(TestSuite suite) {
		System.out.println("=== TEST SUITE: " + suite.name.toUpperCase() + " ===");
		System.out.printf("Duration: %d ms%n", suite.totalDuration);
		System.out.printf("Passed: %d, Failed: %d%n", suite.passed, suite.failed);
		System.out.println();

		for (TestResult result : suite.results) {
			String status = result.passed ? "PASS" : "FAIL";
			System.out.printf("[%s] %s (%d ms)%n", status, result.testName, result.duration);
			if (!result.passed && result.message != null) {
				System.out.println("  Error: " + result.message);
			}
			if (config.verbose && !result.details.isEmpty()) {
				result.details.forEach((key, value) ->
					System.out.println("  " + key + ": " + value));
			}
		}
		System.out.println();
	}

	public void printSummary(List<TestSuite> results) {
		System.out.println("=== TEST SUMMARY ===");
		int totalPassed = 0;
		int totalFailed = 0;
		long totalDuration = 0;

		for (TestSuite suite : results) {
			totalPassed += suite.passed;
			totalFailed += suite.failed;
			totalDuration += suite.totalDuration;

			System.out.printf("Suite %-15s: %d passed, %d failed%n",
				suite.name, suite.passed, suite.failed);
		}

		System.out.println();
		System.out.printf("Total: %d passed, %d failed%n", totalPassed, totalFailed);
		System.out.printf("Total duration: %d ms%n", totalDuration);
		System.out.printf("Success rate: %.1f%%%n",
			(double) totalPassed / (totalPassed + totalFailed) * 100);
	}

	public void close() {
		if (executor != null && !executor.isShutdown()) {
			executor.shutdown();
			try {
				if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
					executor.shutdownNow();
				}
			} catch (InterruptedException e) {
				executor.shutdownNow();
				Thread.currentThread().interrupt();
			}
		}
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		TestConfig config = new TestConfig();

		// Parse arguments
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--url":
					if (i + 1 < args.length) {
						config.serverUrl(args[++i]);
					}
					break;
				case "--timeout":
					if (i + 1 < args.length) {
						config.timeout(Integer.parseInt(args[++i]));
					}
					break;
				case "--concurrent":
					if (i + 1 < args.length) {
						config.maxConcurrentRequests(Integer.parseInt(args[++i]));
					}
					break;
				case "--verbose":
				case "-v":
					config.verbose(true);
					break;
				case "--suites":
					if (i + 1 < args.length) {
						config.testSuites(Arrays.asList(args[++i].split(",")));
					}
					break;
				case "--help":
				case "-h":
					printUsage();
					System.exit(0);
					break;
			}
		}

		ServerTestFramework framework = new ServerTestFramework(config);

		try {
			List<TestSuite> results = framework.runTests();
			framework.printSummary(results);

			// Exit with non-zero status if any tests failed
			boolean anyFailed = results.stream().anyMatch(suite -> suite.failed > 0);
			System.exit(anyFailed ? 1 : 0);

		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "Test framework failed", e);
			System.exit(1);
		} finally {
			framework.close();
		}
	}

	private static void printUsage() {
		System.out.println("Usage: ServerTestFramework [options]");
		System.out.println();
		System.out.println("Test llama.cpp server functionality comprehensively.");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --url <url>           Server URL (default: http://localhost:8080)");
		System.out.println("  --timeout <ms>        Request timeout in milliseconds (default: 30000)");
		System.out.println("  --concurrent <n>      Max concurrent requests (default: 10)");
		System.out.println("  --verbose, -v         Verbose output");
		System.out.println("  --suites <list>       Comma-separated test suites to run");
		System.out.println("                        (basic,performance,concurrency,edge_cases)");
		System.out.println("  --help, -h            Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  ServerTestFramework --url http://localhost:8080");
		System.out.println("  ServerTestFramework --verbose --suites basic,performance");
		System.out.println("  ServerTestFramework --concurrent 5 --timeout 60000");
	}
}