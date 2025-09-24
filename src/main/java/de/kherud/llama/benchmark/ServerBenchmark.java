package de.kherud.llama.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * Server benchmarking tool for llama.cpp server.
 *
 * Based on the Python server-bench.py, this tool measures:
 * - Throughput (requests per second)
 * - Latency (p50, p90, p99)
 * - Token generation speed
 * - Concurrent request handling
 * - Memory usage
 *
 * Supports various workload patterns and dataset sources.
 */
public class ServerBenchmark {
	private static final System.Logger LOGGER = System.getLogger(ServerBenchmark.class.getName());

	private final BenchmarkConfig config;
	private final HttpClient httpClient;
	private final ObjectMapper objectMapper;
	private final List<String> prompts;
	private final MetricsCollector metrics;

	public static class BenchmarkConfig {
		private String serverUrl = "http://localhost:8080";
		private int numRequests = 100;
		private int concurrentClients = 1;
		private String dataset = "random";
		private int promptMinLength = 100;
		private int promptMaxLength = 500;
		private int maxTokens = 100;
		private Duration timeout = Duration.ofMinutes(5);
		private boolean verbose = false;
		private String outputFile = null;
		private Map<String, Object> generationParams = new HashMap<>();

		public BenchmarkConfig serverUrl(String url) {
			this.serverUrl = url;
			return this;
		}

		public BenchmarkConfig requests(int num) {
			this.numRequests = num;
			return this;
		}

		public BenchmarkConfig concurrency(int num) {
			this.concurrentClients = num;
			return this;
		}

		public BenchmarkConfig dataset(String dataset) {
			this.dataset = dataset;
			return this;
		}

		public BenchmarkConfig promptLength(int min, int max) {
			this.promptMinLength = min;
			this.promptMaxLength = max;
			return this;
		}

		public BenchmarkConfig maxTokens(int tokens) {
			this.maxTokens = tokens;
			return this;
		}

		public BenchmarkConfig timeout(Duration timeout) {
			this.timeout = timeout;
			return this;
		}

		public BenchmarkConfig verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public BenchmarkConfig outputFile(String file) {
			this.outputFile = file;
			return this;
		}

		public BenchmarkConfig addGenerationParam(String key, Object value) {
			this.generationParams.put(key, value);
			return this;
		}
	}

	public static class RequestMetrics {
		public final String requestId;
		public final Instant startTime;
		public final Instant endTime;
		public final long durationMs;
		public final int promptTokens;
		public final int completionTokens;
		public final int totalTokens;
		public final double tokensPerSecond;
		public final boolean success;
		public final String error;

		RequestMetrics(String requestId, Instant start, Instant end,
		              int promptTokens, int completionTokens, boolean success, String error) {
			this.requestId = requestId;
			this.startTime = start;
			this.endTime = end;
			this.durationMs = Duration.between(start, end).toMillis();
			this.promptTokens = promptTokens;
			this.completionTokens = completionTokens;
			this.totalTokens = promptTokens + completionTokens;
			this.tokensPerSecond = completionTokens > 0 ?
				(completionTokens * 1000.0 / durationMs) : 0;
			this.success = success;
			this.error = error;
		}
	}

	public static class MetricsCollector {
		private final List<RequestMetrics> allMetrics = new CopyOnWriteArrayList<>();
		private final AtomicInteger successCount = new AtomicInteger();
		private final AtomicInteger failureCount = new AtomicInteger();
		private final AtomicLong totalPromptTokens = new AtomicLong();
		private final AtomicLong totalCompletionTokens = new AtomicLong();

		public void recordMetric(RequestMetrics metric) {
			allMetrics.add(metric);
			if (metric.success) {
				successCount.incrementAndGet();
				totalPromptTokens.addAndGet(metric.promptTokens);
				totalCompletionTokens.addAndGet(metric.completionTokens);
			} else {
				failureCount.incrementAndGet();
			}
		}

		public BenchmarkResults calculateResults() {
			return new BenchmarkResults(this);
		}
	}

	public static class BenchmarkResults {
		public final int totalRequests;
		public final int successfulRequests;
		public final int failedRequests;
		public final double successRate;
		public final long totalDurationMs;
		public final double requestsPerSecond;
		public final double avgLatencyMs;
		public final double p50LatencyMs;
		public final double p90LatencyMs;
		public final double p99LatencyMs;
		public final double minLatencyMs;
		public final double maxLatencyMs;
		public final long totalPromptTokens;
		public final long totalCompletionTokens;
		public final double avgTokensPerSecond;
		public final double totalTokensPerSecond;

		BenchmarkResults(MetricsCollector collector) {
			List<RequestMetrics> metrics = collector.allMetrics;
			this.totalRequests = metrics.size();
			this.successfulRequests = collector.successCount.get();
			this.failedRequests = collector.failureCount.get();
			this.successRate = totalRequests > 0 ?
				(100.0 * successfulRequests / totalRequests) : 0;

			// Calculate timing metrics only from successful requests
			List<Long> latencies = metrics.stream()
				.filter(m -> m.success)
				.map(m -> m.durationMs)
				.sorted()
				.collect(Collectors.toList());

			if (!latencies.isEmpty()) {
				// Overall duration from first start to last end
				Instant firstStart = metrics.stream()
					.map(m -> m.startTime)
					.min(Instant::compareTo)
					.orElse(Instant.now());
				Instant lastEnd = metrics.stream()
					.map(m -> m.endTime)
					.max(Instant::compareTo)
					.orElse(Instant.now());
				this.totalDurationMs = Duration.between(firstStart, lastEnd).toMillis();

				this.requestsPerSecond = totalDurationMs > 0 ?
					(totalRequests * 1000.0 / totalDurationMs) : 0;

				// Latency percentiles
				this.avgLatencyMs = latencies.stream()
					.mapToLong(Long::longValue)
					.average()
					.orElse(0);
				this.minLatencyMs = latencies.get(0);
				this.maxLatencyMs = latencies.get(latencies.size() - 1);
				this.p50LatencyMs = percentile(latencies, 50);
				this.p90LatencyMs = percentile(latencies, 90);
				this.p99LatencyMs = percentile(latencies, 99);
			} else {
				this.totalDurationMs = 0;
				this.requestsPerSecond = 0;
				this.avgLatencyMs = 0;
				this.minLatencyMs = 0;
				this.maxLatencyMs = 0;
				this.p50LatencyMs = 0;
				this.p90LatencyMs = 0;
				this.p99LatencyMs = 0;
			}

			// Token metrics
			this.totalPromptTokens = collector.totalPromptTokens.get();
			this.totalCompletionTokens = collector.totalCompletionTokens.get();

			List<Double> tokenSpeeds = metrics.stream()
				.filter(m -> m.success && m.tokensPerSecond > 0)
				.map(m -> m.tokensPerSecond)
				.collect(Collectors.toList());

			this.avgTokensPerSecond = tokenSpeeds.stream()
				.mapToDouble(Double::doubleValue)
				.average()
				.orElse(0);

			this.totalTokensPerSecond = totalDurationMs > 0 ?
				(totalCompletionTokens * 1000.0 / totalDurationMs) : 0;
		}

		private static double percentile(List<Long> sorted, int percentile) {
			if (sorted.isEmpty()) return 0;
			int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
			index = Math.max(0, Math.min(sorted.size() - 1, index));
			return sorted.get(index);
		}

		public String toFormattedString() {
			StringBuilder sb = new StringBuilder();
			sb.append("\n========== BENCHMARK RESULTS ==========\n");
			sb.append(String.format("Total Requests:        %d\n", totalRequests));
			sb.append(String.format("Successful:            %d (%.1f%%)\n", successfulRequests, successRate));
			sb.append(String.format("Failed:                %d\n", failedRequests));
			sb.append(String.format("Total Duration:        %.2f seconds\n", totalDurationMs / 1000.0));
			sb.append(String.format("Requests/Second:       %.2f\n", requestsPerSecond));
			sb.append("\n--- Latency (ms) ---\n");
			sb.append(String.format("Average:               %.2f\n", avgLatencyMs));
			sb.append(String.format("Min:                   %.2f\n", minLatencyMs));
			sb.append(String.format("Max:                   %.2f\n", maxLatencyMs));
			sb.append(String.format("P50:                   %.2f\n", p50LatencyMs));
			sb.append(String.format("P90:                   %.2f\n", p90LatencyMs));
			sb.append(String.format("P99:                   %.2f\n", p99LatencyMs));
			sb.append("\n--- Tokens ---\n");
			sb.append(String.format("Total Prompt:          %d\n", totalPromptTokens));
			sb.append(String.format("Total Completion:      %d\n", totalCompletionTokens));
			sb.append(String.format("Avg Tokens/Second:     %.2f\n", avgTokensPerSecond));
			sb.append(String.format("Total Tokens/Second:   %.2f\n", totalTokensPerSecond));
			sb.append("========================================\n");
			return sb.toString();
		}

		public void saveToFile(String filename) throws IOException {
			Map<String, Object> results = new HashMap<>();
			results.put("totalRequests", totalRequests);
			results.put("successfulRequests", successfulRequests);
			results.put("failedRequests", failedRequests);
			results.put("successRate", successRate);
			results.put("totalDurationMs", totalDurationMs);
			results.put("requestsPerSecond", requestsPerSecond);
			results.put("avgLatencyMs", avgLatencyMs);
			results.put("p50LatencyMs", p50LatencyMs);
			results.put("p90LatencyMs", p90LatencyMs);
			results.put("p99LatencyMs", p99LatencyMs);
			results.put("minLatencyMs", minLatencyMs);
			results.put("maxLatencyMs", maxLatencyMs);
			results.put("totalPromptTokens", totalPromptTokens);
			results.put("totalCompletionTokens", totalCompletionTokens);
			results.put("avgTokensPerSecond", avgTokensPerSecond);
			results.put("totalTokensPerSecond", totalTokensPerSecond);

			ObjectMapper mapper = new ObjectMapper();
			mapper.writerWithDefaultPrettyPrinter()
				.writeValue(new File(filename), results);
		}
	}

	public ServerBenchmark(BenchmarkConfig config) throws IOException {
		this.config = config;
		this.httpClient = HttpClient.newBuilder()
			.connectTimeout(Duration.ofSeconds(10))
			.build();
		this.objectMapper = new ObjectMapper();
		this.metrics = new MetricsCollector();
		this.prompts = loadPrompts();
	}

	private List<String> loadPrompts() throws IOException {
		List<String> promptList = new ArrayList<>();

		switch (config.dataset.toLowerCase()) {
			case "random":
				LOGGER.log(System.Logger.Level.INFO,"Generating random prompts");
				Random random = new Random();
				for (int i = 0; i < config.numRequests; i++) {
					int length = config.promptMinLength +
						random.nextInt(config.promptMaxLength - config.promptMinLength + 1);
					promptList.add(generateRandomPrompt(length));
				}
				break;

			case "shakespeare":
				// Sample Shakespeare text
				String shakespeare = "To be or not to be, that is the question. " +
					"Whether 'tis nobler in the mind to suffer " +
					"The slings and arrows of outrageous fortune, " +
					"Or to take arms against a sea of troubles.";
				for (int i = 0; i < config.numRequests; i++) {
					promptList.add(shakespeare);
				}
				break;

			default:
				// If it's not a predefined dataset, try to load it as a file
				Path promptFile = Paths.get(config.dataset);
				if (Files.exists(promptFile)) {
					promptList = Files.readAllLines(promptFile);
					LOGGER.log(System.Logger.Level.INFO,"Loaded " + promptList.size() + " prompts from file: " + config.dataset);
				} else {
					throw new IOException("Prompt file not found: " + config.dataset);
				}
				break;
		}

		// Ensure we have enough prompts
		while (promptList.size() < config.numRequests) {
			promptList.add(promptList.get(promptList.size() % Math.max(1, promptList.size())));
		}

		return promptList.subList(0, config.numRequests);
	}

	private String generateRandomPrompt(int wordCount) {
		String[] words = {
			"the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
			"it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
			"this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
			"or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
			"so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
		};

		Random random = new Random();
		StringBuilder prompt = new StringBuilder();
		for (int i = 0; i < wordCount; i++) {
			if (i > 0) prompt.append(" ");
			prompt.append(words[random.nextInt(words.length)]);
		}
		return prompt.toString();
	}

	public BenchmarkResults run() throws Exception {
		LOGGER.log(System.Logger.Level.INFO,"Starting benchmark with " + config.numRequests +
			" requests and " + config.concurrentClients + " concurrent clients");

		// Check server health
		checkServerHealth();

		// Create executor for concurrent requests
		ExecutorService executor = Executors.newFixedThreadPool(config.concurrentClients);
		List<Future<RequestMetrics>> futures = new ArrayList<>();

		// Submit all requests
		Instant benchmarkStart = Instant.now();
		for (int i = 0; i < config.numRequests; i++) {
			final int requestId = i;
			final String prompt = prompts.get(i);

			Future<RequestMetrics> future = executor.submit(() ->
				sendCompletionRequest(requestId, prompt));
			futures.add(future);

			// Small delay between submissions to avoid overwhelming
			if (config.concurrentClients > 1 && i % config.concurrentClients == 0) {
				Thread.sleep(10);
			}
		}

		// Collect results
		int completed = 0;
		for (Future<RequestMetrics> future : futures) {
			try {
				RequestMetrics metric = future.get(config.timeout.toMillis(), TimeUnit.MILLISECONDS);
				metrics.recordMetric(metric);
				completed++;

				if (config.verbose || completed % 10 == 0) {
					LOGGER.log(System.Logger.Level.INFO,"Progress: " + completed + "/" + config.numRequests + " requests completed");
				}
			} catch (TimeoutException e) {
				String requestId = "req-" + futures.indexOf(future);
				metrics.recordMetric(new RequestMetrics(
					requestId, benchmarkStart, Instant.now(), 0, 0, false, "Timeout"
				));
			} catch (Exception e) {
				String requestId = "req-" + futures.indexOf(future);
				metrics.recordMetric(new RequestMetrics(
					requestId, benchmarkStart, Instant.now(), 0, 0, false, e.getMessage()
				));
			}
		}

		executor.shutdown();
		executor.awaitTermination(1, TimeUnit.MINUTES);

		// Calculate and return results
		BenchmarkResults results = metrics.calculateResults();

		// Print results
		System.out.println(results.toFormattedString());

		// Save to file if specified
		if (config.outputFile != null) {
			results.saveToFile(config.outputFile);
			LOGGER.log(System.Logger.Level.INFO,"Results saved to: " + config.outputFile);
		}

		return results;
	}

	private void checkServerHealth() throws Exception {
		HttpRequest request = HttpRequest.newBuilder()
			.uri(URI.create(config.serverUrl + "/health"))
			.timeout(Duration.ofSeconds(5))
			.GET()
			.build();

		try {
			HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
			if (response.statusCode() == 200) {
				LOGGER.log(System.Logger.Level.INFO,"Server health check passed");
			} else {
				throw new IOException("Server health check failed with status: " + response.statusCode());
			}
		} catch (Exception e) {
			throw new IOException("Cannot connect to server at " + config.serverUrl, e);
		}
	}

	private RequestMetrics sendCompletionRequest(int requestId, String prompt) {
		String reqId = "req-" + requestId;
		Instant startTime = Instant.now();

		try {
			// Build request body
			Map<String, Object> requestBody = new HashMap<>();
			requestBody.put("prompt", prompt);
			requestBody.put("n_predict", config.maxTokens);
			requestBody.put("temperature", config.generationParams.getOrDefault("temperature", 0.7));
			requestBody.put("top_p", config.generationParams.getOrDefault("top_p", 0.9));
			requestBody.put("top_k", config.generationParams.getOrDefault("top_k", 40));
			requestBody.put("repeat_penalty", config.generationParams.getOrDefault("repeat_penalty", 1.1));
			requestBody.put("stream", false);

			// Add any additional generation parameters
			requestBody.putAll(config.generationParams);

			String jsonBody = objectMapper.writeValueAsString(requestBody);

			// Send request
			HttpRequest request = HttpRequest.newBuilder()
				.uri(URI.create(config.serverUrl + "/completion"))
				.timeout(config.timeout)
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(jsonBody))
				.build();

			HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
			Instant endTime = Instant.now();

			if (response.statusCode() == 200) {
				// Parse response
				JsonNode responseJson = objectMapper.readTree(response.body());

				int promptTokens = responseJson.path("tokens_evaluated").asInt(0);
				int completionTokens = responseJson.path("tokens_predicted").asInt(0);

				if (config.verbose) {
					String content = responseJson.path("content").asText();
					LOGGER.log(System.Logger.Level.INFO,"Request " + reqId + " completed: " +
						content.substring(0, Math.min(50, content.length())) + "...");
				}

				return new RequestMetrics(reqId, startTime, endTime,
					promptTokens, completionTokens, true, null);
			} else {
				return new RequestMetrics(reqId, startTime, endTime,
					0, 0, false, "HTTP " + response.statusCode());
			}
		} catch (Exception e) {
			Instant endTime = Instant.now();
			return new RequestMetrics(reqId, startTime, endTime,
				0, 0, false, e.getMessage());
		}
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(ServerBenchmark::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length == 0 || args[0].equals("--help")) {
			printUsage();
			return;
		}

		BenchmarkConfig config = new BenchmarkConfig();

		// Parse arguments
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--server":
					config.serverUrl(args[++i]);
					break;
				case "--requests":
					config.requests(Integer.parseInt(args[++i]));
					break;
				case "--concurrency":
					config.concurrency(Integer.parseInt(args[++i]));
					break;
				case "--dataset":
					config.dataset(args[++i]);
					break;
				case "--prompt-min":
					int min = Integer.parseInt(args[++i]);
					int max = config.promptMaxLength;
					config.promptLength(min, max);
					break;
				case "--prompt-max":
					min = config.promptMinLength;
					max = Integer.parseInt(args[++i]);
					config.promptLength(min, max);
					break;
				case "--max-tokens":
					config.maxTokens(Integer.parseInt(args[++i]));
					break;
				case "--timeout":
					config.timeout(Duration.ofSeconds(Integer.parseInt(args[++i])));
					break;
				case "--output":
					config.outputFile(args[++i]);
					break;
				case "--verbose":
					config.verbose(true);
					break;
				case "--temperature":
					config.addGenerationParam("temperature", Float.parseFloat(args[++i]));
					break;
				case "--top-p":
					config.addGenerationParam("top_p", Float.parseFloat(args[++i]));
					break;
				case "--top-k":
					config.addGenerationParam("top_k", Integer.parseInt(args[++i]));
					break;
			}
		}

		ServerBenchmark benchmark = new ServerBenchmark(config);
		BenchmarkResults results = benchmark.run();
	}

	private static void printUsage() {
		System.out.println("Usage: ServerBenchmark [options]");
		System.out.println("\nOptions:");
		System.out.println("  --server <url>        Server URL (default: http://localhost:8080)");
		System.out.println("  --requests <n>        Number of requests to send (default: 100)");
		System.out.println("  --concurrency <n>     Number of concurrent clients (default: 1)");
		System.out.println("  --dataset <name>      Dataset: random, file:<path>, shakespeare (default: random)");
		System.out.println("  --prompt-min <n>      Minimum prompt length in words (default: 100)");
		System.out.println("  --prompt-max <n>      Maximum prompt length in words (default: 500)");
		System.out.println("  --max-tokens <n>      Max tokens to generate (default: 100)");
		System.out.println("  --timeout <seconds>   Request timeout (default: 300)");
		System.out.println("  --output <file>       Save results to JSON file");
		System.out.println("  --verbose             Enable verbose output");
		System.out.println("\nGeneration parameters:");
		System.out.println("  --temperature <f>     Temperature (default: 0.7)");
		System.out.println("  --top-p <f>          Top-p sampling (default: 0.9)");
		System.out.println("  --top-k <n>          Top-k sampling (default: 40)");
		System.out.println("\nExample:");
		System.out.println("  ServerBenchmark --server http://localhost:8080 --requests 1000 --concurrency 10");
	}
}
