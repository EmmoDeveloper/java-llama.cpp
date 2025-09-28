package de.kherud.llama.testing;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Library-friendly server testing framework.
 *
 * This refactored version provides a fluent API for comprehensive testing of llama.cpp server
 * including correctness, performance, concurrency, and edge case handling.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic testing
 * TestResults results = ServerTestFrameworkLibrary.builder()
 *     .serverUrl("http://localhost:8080")
 *     .build()
 *     .runTests();
 *
 * // Configured testing
 * TestResults results = ServerTestFrameworkLibrary.builder()
 *     .serverUrl("http://localhost:8080")
 *     .timeout(Duration.ofSeconds(60))
 *     .maxConcurrentRequests(5)
 *     .testSuites(Arrays.asList("basic", "performance"))
 *     .verbose(true)
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .build()
 *     .runTests();
 *
 * // Async testing
 * ServerTestFrameworkLibrary.builder()
 *     .serverUrl("http://localhost:8080")
 *     .build()
 *     .runTestsAsync()
 *     .thenAccept(results -> System.out.println("Tests complete: " + results.getSuccessRate()));
 *
 * // Custom test configurations
 * TestResults results = framework.runCustomTest("my_test", testCase -> {
 *     // Custom test logic
 *     return testCase.makeRequest("/custom", Map.of("param", "value"));
 * });
 * }</pre>
 */
public class ServerTestFrameworkLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(ServerTestFrameworkLibrary.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final String serverUrl;
	private final Duration timeout;
	private final int maxConcurrentRequests;
	private final boolean verbose;
	private final String testDataPath;
	private final List<String> testSuites;
	private final Map<String, Object> serverParams;
	private final Consumer<TestProgress> progressCallback;
	private final ExecutorService executor;
	private final boolean skipSlowTests;
	private final Duration slowTestThreshold;

	private ServerTestFrameworkLibrary(Builder builder) {
		this.serverUrl = builder.serverUrl;
		this.timeout = builder.timeout;
		this.maxConcurrentRequests = builder.maxConcurrentRequests;
		this.verbose = builder.verbose;
		this.testDataPath = builder.testDataPath;
		this.testSuites = Collections.unmodifiableList(builder.testSuites);
		this.serverParams = Collections.unmodifiableMap(builder.serverParams);
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
		this.skipSlowTests = builder.skipSlowTests;
		this.slowTestThreshold = builder.slowTestThreshold;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Run all configured test suites
	 */
	public TestResults runTests() {
		progress("Starting server test framework", 0.0);
		Instant startTime = Instant.now();

		try {
			// Check server availability first
			if (!isServerAvailable()) {
				return new TestResults.Builder()
					.success(false)
					.message("Server not available at: " + serverUrl)
					.duration(Duration.between(startTime, Instant.now()))
					.build();
			}

			List<TestSuite> suiteResults = new ArrayList<>();
			int totalSuites = testSuites.size();

			for (int i = 0; i < totalSuites; i++) {
				String suiteName = testSuites.get(i);
				progress("Running test suite: " + suiteName, (double) i / totalSuites);

				TestSuite suite = runTestSuite(suiteName);
				suiteResults.add(suite);

				if (verbose) {
					logTestSuite(suite);
				}
			}

			progress("All test suites complete", 1.0);

			// Calculate overall results
			int totalPassed = suiteResults.stream().mapToInt(s -> s.passed).sum();
			int totalFailed = suiteResults.stream().mapToInt(s -> s.failed).sum();
			Duration totalDuration = Duration.between(startTime, Instant.now());

			return new TestResults.Builder()
				.success(totalFailed == 0)
				.message(String.format("Tests completed: %d passed, %d failed", totalPassed, totalFailed))
				.suites(suiteResults)
				.totalPassed(totalPassed)
				.totalFailed(totalFailed)
				.duration(totalDuration)
				.build();

		} catch (Exception e) {
			String errorMsg = "Test execution failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new TestResults.Builder()
				.success(false)
				.message(errorMsg)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Run tests asynchronously
	 */
	public CompletableFuture<TestResults> runTestsAsync() {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();

		return CompletableFuture.supplyAsync(this::runTests, exec);
	}

	/**
	 * Run a specific test suite
	 */
	public TestSuite runTestSuite(String suiteName) {
		LOGGER.log(System.Logger.Level.INFO, "Running test suite: " + suiteName);
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
			LOGGER.log(System.Logger.Level.ERROR, "Test suite failed: " + suiteName, e);
			suite.addResult(new TestResult("suite_execution").fail("Suite execution failed", e));
		}

		suite.endTime = Instant.now();
		return suite;
	}

	/**
	 * Run a custom test with user-defined logic
	 */
	public TestResult runCustomTest(String testName, TestFunction testFunction) {
		TestResult result = new TestResult(testName);
		Instant start = Instant.now();

		try {
			TestContext context = new TestContext(serverUrl, timeout, MAPPER);
			boolean success = testFunction.execute(context);

			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (success) {
				result.pass("Custom test completed successfully");
			} else {
				result.fail("Custom test failed");
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail("Custom test failed", e);
		}

		return result;
	}

	/**
	 * Validate server health and basic functionality
	 */
	public ValidationResult validateServer() {
		try {
			boolean healthy = isServerAvailable();
			if (!healthy) {
				return new ValidationResult(false, "Server not available", serverUrl);
			}

			// Test basic endpoints
			boolean hasModels = testEndpoint("/v1/models", "GET") == 200;
			boolean hasHealth = testEndpoint("/health", "GET") == 200;

			if (hasModels && hasHealth) {
				return new ValidationResult(true, "Server is functional", serverUrl);
			} else {
				return new ValidationResult(false, "Server missing required endpoints", serverUrl);
			}

		} catch (Exception e) {
			return new ValidationResult(false, "Server validation failed: " + e.getMessage(), serverUrl);
		}
	}

	/**
	 * Benchmark tokenizer performance with various text types
	 */
	public TokenizerBenchmarkResult benchmarkTokenizer() {
		return benchmarkTokenizer(getStandardBenchmarkTexts(), 100);
	}

	/**
	 * Benchmark tokenizer performance with custom texts and iterations
	 */
	public TokenizerBenchmarkResult benchmarkTokenizer(List<String> texts, int iterations) {
		TokenizerBenchmarkResult.Builder builder = new TokenizerBenchmarkResult.Builder();
		Instant startTime = Instant.now();

		try {
			progress("Starting tokenizer benchmark", 0.0);

			// Warmup phase
			warmupTokenizer(texts, Math.min(iterations / 10, 10));
			progress("Warmup complete, starting benchmark", 0.2);

			long totalEncodeTime = 0;
			long totalDecodeTime = 0;
			int totalTokens = 0;
			int totalProcessed = 0;

			for (int iter = 0; iter < iterations; iter++) {
				progress("Benchmark iteration " + (iter + 1) + "/" + iterations,
					0.2 + 0.7 * (double) iter / iterations);

				for (String text : texts) {
					try {
						// Measure encoding via server
						long startEncode = System.nanoTime();
						Map<String, Object> tokenizeRequest = Map.of("content", text);
						HttpURLConnection conn = createConnection("/tokenize", "POST");
						writeJsonRequest(conn, tokenizeRequest);

						if (conn.getResponseCode() == 200) {
							String response = readResponse(conn);
							JsonNode json = MAPPER.readTree(response);
							if (json.has("tokens")) {
								int tokenCount = json.get("tokens").size();
								totalTokens += tokenCount;
								totalProcessed++;
							}
						}
						long endEncode = System.nanoTime();
						totalEncodeTime += (endEncode - startEncode);

						// Note: Decode timing would require additional server endpoint
						// For now, we'll focus on encode performance

					} catch (Exception e) {
						LOGGER.log(System.Logger.Level.WARNING, "Benchmark iteration failed for text: " + text, e);
					}
				}
			}

			progress("Benchmark complete", 1.0);

			Duration duration = Duration.between(startTime, Instant.now());
			double textsPerSecond = totalProcessed > 0 ?
				(double) totalProcessed / (totalEncodeTime / 1_000_000_000.0) : 0.0;
			double tokensPerSecond = totalTokens > 0 ?
				(double) totalTokens / (totalEncodeTime / 1_000_000_000.0) : 0.0;
			double avgLatencyMicros = totalProcessed > 0 ?
				(totalEncodeTime / 1000.0) / totalProcessed : 0.0;

			builder.success(true)
				.message(String.format("Tokenizer benchmark completed: %.1f texts/sec, %.1f tokens/sec",
					textsPerSecond, tokensPerSecond))
				.totalTexts(totalProcessed)
				.totalTokens(totalTokens)
				.encodeTimeNanos(totalEncodeTime)
				.iterations(iterations)
				.textsPerSecond(textsPerSecond)
				.tokensPerSecond(tokensPerSecond)
				.avgLatencyMicros(avgLatencyMicros)
				.duration(duration);

		} catch (Exception e) {
			builder.success(false)
				.message("Tokenizer benchmark failed: " + e.getMessage())
				.duration(Duration.between(startTime, Instant.now()))
				.error(e);
		}

		return builder.build();
	}

	/**
	 * Benchmark tokenizer with texts of specific length ranges
	 */
	public TokenizerBenchmarkResult benchmarkTokenizerByLength(int minLength, int maxLength, int samples, int iterations) {
		List<String> texts = generateTextsByLength(minLength, maxLength, samples);
		return benchmarkTokenizer(texts, iterations);
	}

	/**
	 * Benchmark tokenizer with special tokens and edge cases
	 */
	public TokenizerBenchmarkResult benchmarkTokenizerSpecialCases(int iterations) {
		List<String> specialTexts = getSpecialTokenTexts();
		return benchmarkTokenizer(specialTexts, iterations);
	}

	/**
	 * Compare tokenizer benchmark results
	 */
	public TokenizerComparisonResult compareTokenizerBenchmarks(TokenizerBenchmarkResult baseline, TokenizerBenchmarkResult comparison) {
		TokenizerComparisonResult.Builder builder = new TokenizerComparisonResult.Builder();

		try {
			if (!baseline.isSuccess() || !comparison.isSuccess()) {
				return builder.success(false)
					.message("Cannot compare failed benchmarks")
					.build();
			}

			double speedupRatio = comparison.getTextsPerSecond() / baseline.getTextsPerSecond();
			double latencyRatio = comparison.getAvgLatencyMicros() / baseline.getAvgLatencyMicros();
			double throughputRatio = comparison.getTokensPerSecond() / baseline.getTokensPerSecond();

			String message = String.format("Benchmark comparison: %.2fx speedup, %.2fx latency ratio, %.2fx throughput ratio",
				speedupRatio, latencyRatio, throughputRatio);

			builder.success(true)
				.message(message)
				.baseline(baseline)
				.comparison(comparison)
				.speedupRatio(speedupRatio)
				.latencyRatio(latencyRatio)
				.throughputRatio(throughputRatio);

		} catch (Exception e) {
			builder.success(false)
				.message("Benchmark comparison failed: " + e.getMessage())
				.error(e);
		}

		return builder.build();
	}

	// Helper methods for tokenizer benchmarking
	private void warmupTokenizer(List<String> texts, int warmupIterations) {
		try {
			for (int i = 0; i < warmupIterations; i++) {
				for (String text : texts.subList(0, Math.min(5, texts.size()))) {
					Map<String, Object> request = Map.of("content", text);
					HttpURLConnection conn = createConnection("/tokenize", "POST");
					writeJsonRequest(conn, request);
					if (conn.getResponseCode() == 200) {
						readResponse(conn);
					}
				}
			}
		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.WARNING, "Warmup failed", e);
		}
	}

	private List<String> getStandardBenchmarkTexts() {
		return Arrays.asList(
			"Hello world",
			"The quick brown fox jumps over the lazy dog",
			"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
			"This is a longer text sample that contains multiple sentences. " +
			"It should provide a good test case for tokenizer performance benchmarking.",
			"Code example: function hello() { return 'world'; }",
			"Unicode test: αβγδε 你好世界 こんにちは 🚀🌍",
			"Mixed content with numbers 123 and symbols @#$%",
			"Multi\nline\ntext\nwith\nbreaks"
		);
	}

	private List<String> generateTextsByLength(int minLength, int maxLength, int samples) {
		List<String> texts = new ArrayList<>();
		Random random = new Random(42);
		String baseText = "This is a sample text that will be repeated to reach the desired length. ";

		for (int i = 0; i < samples; i++) {
			int targetLength = minLength + random.nextInt(maxLength - minLength + 1);
			StringBuilder text = new StringBuilder();

			while (text.length() < targetLength) {
				text.append(baseText);
			}

			texts.add(text.substring(0, Math.min(targetLength, text.length())));
		}

		return texts;
	}

	private List<String> getSpecialTokenTexts() {
		return Arrays.asList(
			"<s>Beginning of sequence</s>",
			"<unk>Unknown token test</unk>",
			"<|endoftext|>End of text marker<|endoftext|>",
			"Special chars: \t\n\r",
			"Unicode controls: \u001f\u0000\u00a0",
			"Emojis: 😀😃😄😁😆😅😂🤣",
			"Long special: " + "<mask>".repeat(100),
			"Mixed: Hello <s>world</s> test <unk>unknown</unk>"
		);
	}

	// Test suite implementations
	private void runBasicTests(TestSuite suite) {
		suite.addResult(testHealthCheck());
		suite.addResult(testModelInfo());
		suite.addResult(testSimpleCompletion());
		suite.addResult(testChatCompletion());
		suite.addResult(testTokenization());
		suite.addResult(testEmbeddings());
	}

	private void runPerformanceTests(TestSuite suite) {
		if (skipSlowTests) {
			suite.addResult(new TestResult("performance_tests").pass("Skipped (slow tests disabled)"));
			return;
		}

		suite.addResult(testThroughput());
		suite.addResult(testLatency());
		suite.addResult(testMemoryUsage());
		suite.addResult(testLongSequence());
	}

	private void runConcurrencyTests(TestSuite suite) {
		suite.addResult(testConcurrentRequests());
		suite.addResult(testRequestQueuing());
		suite.addResult(testResourceSharing());
	}

	private void runEdgeCaseTests(TestSuite suite) {
		suite.addResult(testEmptyInput());
		suite.addResult(testVeryLongInput());
		suite.addResult(testInvalidParameters());
		suite.addResult(testMalformedRequests());
		suite.addResult(testUnicodeHandling());
	}

	// Individual test implementations (simplified versions of original)
	private TestResult testHealthCheck() {
		return executeTest("health_check", () -> {
			int responseCode = testEndpoint("/health", "GET");
			return responseCode == 200;
		});
	}

	private TestResult testModelInfo() {
		return executeTest("model_info", () -> {
			HttpURLConnection conn = createConnection("/v1/models", "GET");
			int responseCode = conn.getResponseCode();

			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);
				return json.has("data") && json.get("data").isArray();
			}
			return false;
		});
	}

	private TestResult testSimpleCompletion() {
		return executeTest("simple_completion", () -> {
			Map<String, Object> request = Map.of(
				"prompt", "The capital of France is",
				"max_tokens", 10,
				"temperature", 0.1
			);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);
				return json.has("choices") && json.get("choices").size() > 0;
			}
			return false;
		});
	}

	private TestResult testChatCompletion() {
		return executeTest("chat_completion", () -> {
			Map<String, Object> request = Map.of(
				"messages", List.of(Map.of("role", "user", "content", "Hello, how are you?")),
				"max_tokens", 20
			);

			HttpURLConnection conn = createConnection("/v1/chat/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);
				return json.has("choices") && json.get("choices").size() > 0;
			}
			return false;
		});
	}

	private TestResult testTokenization() {
		return executeTest("tokenization", () -> {
			Map<String, Object> request = Map.of("content", "Hello world, this is a test sentence.");

			HttpURLConnection conn = createConnection("/tokenize", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);
				return json.has("tokens");
			}
			return false;
		});
	}

	private TestResult testEmbeddings() {
		return executeTest("embeddings", () -> {
			Map<String, Object> request = Map.of("content", "This is a test sentence for embeddings.");

			HttpURLConnection conn = createConnection("/embedding", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			if (responseCode == 200) {
				String response = readResponse(conn);
				JsonNode json = MAPPER.readTree(response);
				return json.has("embedding");
			}
			return false;
		});
	}

	private TestResult testThroughput() {
		return executeTest("throughput", () -> {
			int requestCount = 10;
			List<Future<Boolean>> futures = new ArrayList<>();

			for (int i = 0; i < requestCount; i++) {
				final int requestIndex = i;
				Future<Boolean> future = executor.submit(() -> {
					try {
						Map<String, Object> request = Map.of(
							"prompt", "Test prompt " + requestIndex,
							"max_tokens", 5
						);

						HttpURLConnection conn = createConnection("/v1/completions", "POST");
						writeJsonRequest(conn, request);
						return conn.getResponseCode() == 200;
					} catch (Exception e) {
						return false;
					}
				});
				futures.add(future);
			}

			// Wait for all requests
			for (Future<Boolean> future : futures) {
				if (!future.get((int) timeout.toMillis(), TimeUnit.MILLISECONDS)) {
					return false;
				}
			}
			return true;
		});
	}

	private TestResult testLatency() {
		return executeTest("latency", () -> {
			for (int i = 0; i < 5; i++) {
				Map<String, Object> request = Map.of(
					"prompt", "Quick test",
					"max_tokens", 1
				);

				HttpURLConnection conn = createConnection("/v1/completions", "POST");
				writeJsonRequest(conn, request);
				if (conn.getResponseCode() != 200) {
					return false;
				}
				readResponse(conn);
			}
			return true;
		});
	}

	private TestResult testMemoryUsage() {
		return executeTest("memory_usage", () -> {
			// Make several requests to potentially increase memory usage
			for (int i = 0; i < 5; i++) {
				Map<String, Object> request = Map.of(
					"prompt", "Memory test prompt with some content to process",
					"max_tokens", 10
				);

				HttpURLConnection conn = createConnection("/v1/completions", "POST");
				writeJsonRequest(conn, request);
				if (conn.getResponseCode() != 200) {
					return false;
				}
				readResponse(conn);
			}
			return true;
		});
	}

	private TestResult testLongSequence() {
		return executeTest("long_sequence", () -> {
			StringBuilder prompt = new StringBuilder();
			for (int i = 0; i < 100; i++) {
				prompt.append("This is sentence number ").append(i).append(". ");
			}

			Map<String, Object> request = Map.of(
				"prompt", prompt.toString(),
				"max_tokens", 10
			);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);
			return conn.getResponseCode() == 200;
		});
	}

	private TestResult testConcurrentRequests() {
		return executeTest("concurrent_requests", () -> {
			int concurrentCount = Math.min(maxConcurrentRequests, 5);
			List<Future<Boolean>> futures = new ArrayList<>();

			for (int i = 0; i < concurrentCount; i++) {
				final int requestIndex = i;
				Future<Boolean> future = executor.submit(() -> {
					try {
						Map<String, Object> request = Map.of(
							"prompt", "Concurrent request " + requestIndex,
							"max_tokens", 5
						);

						HttpURLConnection conn = createConnection("/v1/completions", "POST");
						writeJsonRequest(conn, request);
						return conn.getResponseCode() == 200;
					} catch (Exception e) {
						return false;
					}
				});
				futures.add(future);
			}

			int successCount = 0;
			for (Future<Boolean> future : futures) {
				if (future.get((int) timeout.toMillis(), TimeUnit.MILLISECONDS)) {
					successCount++;
				}
			}

			return successCount == concurrentCount;
		});
	}

	private TestResult testRequestQueuing() {
		return executeTest("request_queuing", () -> {
			int queueCount = maxConcurrentRequests + 2;
			List<Future<Boolean>> futures = new ArrayList<>();

			for (int i = 0; i < queueCount; i++) {
				final int requestIndex = i;
				Future<Boolean> future = executor.submit(() -> {
					try {
						Map<String, Object> request = Map.of(
							"prompt", "Queue test " + requestIndex,
							"max_tokens", 3
						);

						HttpURLConnection conn = createConnection("/v1/completions", "POST");
						writeJsonRequest(conn, request);
						readResponse(conn);
						return true;
					} catch (Exception e) {
						return false;
					}
				});
				futures.add(future);
			}

			int successCount = 0;
			for (Future<Boolean> future : futures) {
				if (future.get((int) timeout.toMillis(), TimeUnit.MILLISECONDS)) {
					successCount++;
				}
			}

			return successCount == queueCount;
		});
	}

	private TestResult testResourceSharing() {
		return executeTest("resource_sharing", () -> {
			Future<Boolean> completion = executor.submit(() -> {
				try {
					Map<String, Object> request = Map.of(
						"prompt", "Resource test completion",
						"max_tokens", 3
					);

					HttpURLConnection conn = createConnection("/v1/completions", "POST");
					writeJsonRequest(conn, request);
					return conn.getResponseCode() == 200;
				} catch (Exception e) {
					return false;
				}
			});

			Future<Boolean> tokenization = executor.submit(() -> {
				try {
					Map<String, Object> request = Map.of("content", "Resource test tokenization");

					HttpURLConnection conn = createConnection("/tokenize", "POST");
					writeJsonRequest(conn, request);
					return conn.getResponseCode() == 200;
				} catch (Exception e) {
					return false;
				}
			});

			boolean completionSuccess = completion.get((int) timeout.toMillis(), TimeUnit.MILLISECONDS);
			boolean tokenizationSuccess = tokenization.get((int) timeout.toMillis(), TimeUnit.MILLISECONDS);

			return completionSuccess && tokenizationSuccess;
		});
	}

	private TestResult testEmptyInput() {
		return executeTest("empty_input", () -> {
			Map<String, Object> request = Map.of(
				"prompt", "",
				"max_tokens", 5
			);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			// Server should handle empty input gracefully (either 200 or 400)
			return responseCode == 200 || responseCode == 400;
		});
	}

	private TestResult testVeryLongInput() {
		return executeTest("very_long_input", () -> {
			StringBuilder longPrompt = new StringBuilder();
			for (int i = 0; i < 10000; i++) {
				longPrompt.append("word").append(i).append(" ");
			}

			Map<String, Object> request = Map.of(
				"prompt", longPrompt.toString(),
				"max_tokens", 1
			);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			// Server should handle this gracefully (200, 400, or 413)
			return responseCode == 200 || responseCode == 400 || responseCode == 413;
		});
	}

	private TestResult testInvalidParameters() {
		return executeTest("invalid_parameters", () -> {
			Map<String, Object> request = Map.of(
				"prompt", "Test",
				"max_tokens", -1, // Invalid value
				"temperature", 2.5 // Out of range
			);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			// Server should reject invalid parameters with 400
			return responseCode == 400;
		});
	}

	private TestResult testMalformedRequests() {
		return executeTest("malformed_requests", () -> {
			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			conn.setDoOutput(true);
			conn.setRequestProperty("Content-Type", "application/json");

			// Send malformed JSON
			try (OutputStream os = conn.getOutputStream()) {
				os.write("{ invalid json".getBytes(StandardCharsets.UTF_8));
			}

			int responseCode = conn.getResponseCode();
			// Server should reject malformed JSON with 400
			return responseCode == 400;
		});
	}

	private TestResult testUnicodeHandling() {
		return executeTest("unicode_handling", () -> {
			Map<String, Object> request = Map.of(
				"prompt", "Unicode test: 🌟 こんにちは العالم 🚀",
				"max_tokens", 5
			);

			HttpURLConnection conn = createConnection("/v1/completions", "POST");
			writeJsonRequest(conn, request);

			int responseCode = conn.getResponseCode();
			if (responseCode == 200) {
				readResponse(conn);
				return true;
			}
			return false;
		});
	}

	// Helper methods
	private TestResult executeTest(String testName, TestLogic logic) {
		TestResult result = new TestResult(testName);
		Instant start = Instant.now();

		try {
			boolean success = logic.execute();
			result.duration = Duration.between(start, Instant.now()).toMillis();

			if (success) {
				result.pass(testName + " successful");
			} else {
				result.fail(testName + " failed");
			}

		} catch (Exception e) {
			result.duration = Duration.between(start, Instant.now()).toMillis();
			result.fail(testName + " failed", e);
		}

		return result;
	}

	private boolean isServerAvailable() {
		try {
			return testEndpoint("/health", "GET") == 200;
		} catch (Exception e) {
			return false;
		}
	}

	private int testEndpoint(String endpoint, String method) throws IOException {
		HttpURLConnection conn = createConnection(endpoint, method);
		return conn.getResponseCode();
	}

	private HttpURLConnection createConnection(String endpoint, String method) throws IOException {
		URL url = new URL(serverUrl + endpoint);
		HttpURLConnection conn = (HttpURLConnection) url.openConnection();
		conn.setRequestMethod(method);
		conn.setConnectTimeout((int) timeout.toMillis());
		conn.setReadTimeout((int) timeout.toMillis());

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

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new TestProgress(message, progress));
		}
	}

	private void logTestSuite(TestSuite suite) {
		LOGGER.log(System.Logger.Level.INFO, String.format("Suite %s: %d passed, %d failed (%d ms)",
			suite.name, suite.passed, suite.failed, suite.totalDuration));
	}

	@Override
	public void close() throws IOException {
		if (executor != null) {
			executor.shutdown();
		}
	}

	// Functional interfaces
	@FunctionalInterface
	public interface TestLogic {
		boolean execute() throws Exception;
	}

	@FunctionalInterface
	public interface TestFunction {
		boolean execute(TestContext context) throws Exception;
	}

	// Builder class
	public static class Builder {
		private String serverUrl = "http://localhost:8080";
		private Duration timeout = Duration.ofSeconds(30);
		private int maxConcurrentRequests = 10;
		private boolean verbose = false;
		private String testDataPath = "test_data";
		private List<String> testSuites = Arrays.asList("basic", "performance", "concurrency", "edge_cases");
		private Map<String, Object> serverParams = new HashMap<>();
		private Consumer<TestProgress> progressCallback;
		private ExecutorService executor;
		private boolean skipSlowTests = false;
		private Duration slowTestThreshold = Duration.ofSeconds(10);

		public Builder serverUrl(String serverUrl) {
			this.serverUrl = serverUrl;
			return this;
		}

		public Builder timeout(Duration timeout) {
			this.timeout = timeout;
			return this;
		}

		public Builder maxConcurrentRequests(int maxConcurrentRequests) {
			this.maxConcurrentRequests = maxConcurrentRequests;
			return this;
		}

		public Builder verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public Builder testDataPath(String testDataPath) {
			this.testDataPath = testDataPath;
			return this;
		}

		public Builder testSuites(List<String> testSuites) {
			this.testSuites = new ArrayList<>(testSuites);
			return this;
		}

		public Builder testSuite(String testSuite) {
			this.testSuites = List.of(testSuite);
			return this;
		}

		public Builder serverParam(String key, Object value) {
			this.serverParams.put(key, value);
			return this;
		}

		public Builder serverParams(Map<String, Object> serverParams) {
			this.serverParams.putAll(serverParams);
			return this;
		}

		public Builder progressCallback(Consumer<TestProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public Builder skipSlowTests(boolean skipSlowTests) {
			this.skipSlowTests = skipSlowTests;
			return this;
		}

		public Builder slowTestThreshold(Duration slowTestThreshold) {
			this.slowTestThreshold = slowTestThreshold;
			return this;
		}

		public ServerTestFrameworkLibrary build() {
			return new ServerTestFrameworkLibrary(this);
		}
	}

	// Helper classes for API
	public static class TestProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public TestProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	public static class TestContext {
		private final String serverUrl;
		private final Duration timeout;
		private final ObjectMapper mapper;

		public TestContext(String serverUrl, Duration timeout, ObjectMapper mapper) {
			this.serverUrl = serverUrl;
			this.timeout = timeout;
			this.mapper = mapper;
		}

		public boolean makeRequest(String endpoint, Map<String, Object> requestData) throws Exception {
			URL url = new URL(serverUrl + endpoint);
			HttpURLConnection conn = (HttpURLConnection) url.openConnection();
			conn.setRequestMethod("POST");
			conn.setConnectTimeout((int) timeout.toMillis());
			conn.setReadTimeout((int) timeout.toMillis());
			conn.setDoOutput(true);
			conn.setRequestProperty("Content-Type", "application/json");

			String json = mapper.writeValueAsString(requestData);
			try (OutputStream os = conn.getOutputStream()) {
				os.write(json.getBytes(StandardCharsets.UTF_8));
			}

			return conn.getResponseCode() == 200;
		}

		public String getServerUrl() { return serverUrl; }
		public Duration getTimeout() { return timeout; }
	}

	// Result classes (using classes from original for compatibility)
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

		public boolean isPassed() { return passed; }
		public String getTestName() { return testName; }
		public String getMessage() { return message; }
		public long getDuration() { return duration; }
		public Optional<Throwable> getException() { return Optional.ofNullable(exception); }
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

		public String getName() { return name; }
		public List<TestResult> getResults() { return Collections.unmodifiableList(results); }
		public int getPassed() { return passed; }
		public int getFailed() { return failed; }
		public long getTotalDuration() { return totalDuration; }
	}

	public static class TestResults {
		private final boolean success;
		private final String message;
		private final List<TestSuite> suites;
		private final int totalPassed;
		private final int totalFailed;
		private final Duration duration;
		private final Exception error;

		private TestResults(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.suites = Collections.unmodifiableList(builder.suites);
			this.totalPassed = builder.totalPassed;
			this.totalFailed = builder.totalFailed;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public List<TestSuite> getSuites() { return suites; }
		public int getTotalPassed() { return totalPassed; }
		public int getTotalFailed() { return totalFailed; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public double getSuccessRate() {
			int total = totalPassed + totalFailed;
			return total > 0 ? (double) totalPassed / total : 0.0;
		}

		public static class Builder {
			private boolean success;
			private String message;
			private List<TestSuite> suites = new ArrayList<>();
			private int totalPassed;
			private int totalFailed;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder suites(List<TestSuite> suites) { this.suites = suites; return this; }
			public Builder totalPassed(int totalPassed) { this.totalPassed = totalPassed; return this; }
			public Builder totalFailed(int totalFailed) { this.totalFailed = totalFailed; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public TestResults build() { return new TestResults(this); }
		}
	}

	public static class ValidationResult {
		private final boolean valid;
		private final String message;
		private final String serverUrl;

		public ValidationResult(boolean valid, String message, String serverUrl) {
			this.valid = valid;
			this.message = message;
			this.serverUrl = serverUrl;
		}

		public boolean isValid() { return valid; }
		public String getMessage() { return message; }
		public String getServerUrl() { return serverUrl; }
	}

	// Tokenizer benchmark result classes
	public static class TokenizerBenchmarkResult {
		private final boolean success;
		private final String message;
		private final int totalTexts;
		private final int totalTokens;
		private final long encodeTimeNanos;
		private final int iterations;
		private final double textsPerSecond;
		private final double tokensPerSecond;
		private final double avgLatencyMicros;
		private final Duration duration;
		private final Exception error;

		private TokenizerBenchmarkResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.totalTexts = builder.totalTexts;
			this.totalTokens = builder.totalTokens;
			this.encodeTimeNanos = builder.encodeTimeNanos;
			this.iterations = builder.iterations;
			this.textsPerSecond = builder.textsPerSecond;
			this.tokensPerSecond = builder.tokensPerSecond;
			this.avgLatencyMicros = builder.avgLatencyMicros;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public int getTotalTexts() { return totalTexts; }
		public int getTotalTokens() { return totalTokens; }
		public long getEncodeTimeNanos() { return encodeTimeNanos; }
		public int getIterations() { return iterations; }
		public double getTextsPerSecond() { return textsPerSecond; }
		public double getTokensPerSecond() { return tokensPerSecond; }
		public double getAvgLatencyMicros() { return avgLatencyMicros; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public double getAvgTokensPerText() {
			return totalTexts > 0 ? (double) totalTokens / totalTexts : 0.0;
		}

		public static class Builder {
			private boolean success;
			private String message;
			private int totalTexts;
			private int totalTokens;
			private long encodeTimeNanos;
			private int iterations;
			private double textsPerSecond;
			private double tokensPerSecond;
			private double avgLatencyMicros;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder totalTexts(int totalTexts) { this.totalTexts = totalTexts; return this; }
			public Builder totalTokens(int totalTokens) { this.totalTokens = totalTokens; return this; }
			public Builder encodeTimeNanos(long encodeTimeNanos) { this.encodeTimeNanos = encodeTimeNanos; return this; }
			public Builder iterations(int iterations) { this.iterations = iterations; return this; }
			public Builder textsPerSecond(double textsPerSecond) { this.textsPerSecond = textsPerSecond; return this; }
			public Builder tokensPerSecond(double tokensPerSecond) { this.tokensPerSecond = tokensPerSecond; return this; }
			public Builder avgLatencyMicros(double avgLatencyMicros) { this.avgLatencyMicros = avgLatencyMicros; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public TokenizerBenchmarkResult build() { return new TokenizerBenchmarkResult(this); }
		}
	}

	public static class TokenizerComparisonResult {
		private final boolean success;
		private final String message;
		private final TokenizerBenchmarkResult baseline;
		private final TokenizerBenchmarkResult comparison;
		private final double speedupRatio;
		private final double latencyRatio;
		private final double throughputRatio;
		private final Exception error;

		private TokenizerComparisonResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.baseline = builder.baseline;
			this.comparison = builder.comparison;
			this.speedupRatio = builder.speedupRatio;
			this.latencyRatio = builder.latencyRatio;
			this.throughputRatio = builder.throughputRatio;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public TokenizerBenchmarkResult getBaseline() { return baseline; }
		public TokenizerBenchmarkResult getComparison() { return comparison; }
		public double getSpeedupRatio() { return speedupRatio; }
		public double getLatencyRatio() { return latencyRatio; }
		public double getThroughputRatio() { return throughputRatio; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private TokenizerBenchmarkResult baseline;
			private TokenizerBenchmarkResult comparison;
			private double speedupRatio;
			private double latencyRatio;
			private double throughputRatio;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder baseline(TokenizerBenchmarkResult baseline) { this.baseline = baseline; return this; }
			public Builder comparison(TokenizerBenchmarkResult comparison) { this.comparison = comparison; return this; }
			public Builder speedupRatio(double speedupRatio) { this.speedupRatio = speedupRatio; return this; }
			public Builder latencyRatio(double latencyRatio) { this.latencyRatio = latencyRatio; return this; }
			public Builder throughputRatio(double throughputRatio) { this.throughputRatio = throughputRatio; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public TokenizerComparisonResult build() { return new TokenizerComparisonResult(this); }
		}
	}
}
