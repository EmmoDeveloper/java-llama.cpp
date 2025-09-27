package de.kherud.llama.testing;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Library-friendly embedding server HTTP client.
 *
 * This refactored version provides a fluent API for interacting with embedding servers
 * with builder pattern configuration, batch processing, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic embedding
 * EmbeddingResult result = ServerEmbeddingHttpClientLibrary.builder()
 *     .serverUrl("http://localhost:8080")
 *     .build()
 *     .getEmbedding("Hello world");
 *
 * // Batch embeddings
 * BatchEmbeddingResult result = ServerEmbeddingHttpClientLibrary.builder()
 *     .serverUrl("http://localhost:8080")
 *     .timeout(Duration.ofMinutes(10))
 *     .batchSize(100)
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .build()
 *     .getBatchEmbeddings(textList);
 *
 * // Async embeddings
 * ServerEmbeddingHttpClientLibrary.builder()
 *     .serverUrl("http://localhost:8080")
 *     .build()
 *     .getEmbeddingAsync("Hello world")
 *     .thenAccept(result -> System.out.println("Embedding size: " + result.getEmbedding().size()));
 *
 * // Compare embeddings
 * SimilarityResult similarity = client.compareEmbeddings("text1", "text2");
 * }</pre>
 */
public class ServerEmbeddingHttpClientLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(ServerEmbeddingHttpClientLibrary.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final String serverUrl;
	private final HttpClient httpClient;
	private final Duration timeout;
	private final int retryAttempts;
	private final Duration retryDelay;
	private final int batchSize;
	private final Consumer<EmbeddingProgress> progressCallback;
	private final ExecutorService executor;
	private final boolean enableMetrics;

	private ServerEmbeddingHttpClientLibrary(Builder builder) {
		this.serverUrl = builder.serverUrl.endsWith("/") ?
			builder.serverUrl.substring(0, builder.serverUrl.length() - 1) : builder.serverUrl;
		this.timeout = builder.timeout;
		this.retryAttempts = builder.retryAttempts;
		this.retryDelay = builder.retryDelay;
		this.batchSize = builder.batchSize;
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
		this.enableMetrics = builder.enableMetrics;

		this.httpClient = HttpClient.newBuilder()
			.connectTimeout(timeout)
			.build();
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Get embedding for a single text
	 */
	public EmbeddingResult getEmbedding(String text) {
		return getEmbedding(text, false);
	}

	/**
	 * Get embedding using simple endpoint
	 */
	public EmbeddingResult getSimpleEmbedding(String text) {
		return getEmbedding(text, true);
	}

	/**
	 * Get embeddings for multiple texts in batches
	 */
	public BatchEmbeddingResult getBatchEmbeddings(List<String> texts) {
		progress("Starting batch embedding", 0.0);
		Instant startTime = Instant.now();

		try {
			List<EmbeddingResult> results = new ArrayList<>();
			List<String> failedTexts = new ArrayList<>();
			int totalBatches = (int) Math.ceil((double) texts.size() / batchSize);

			for (int i = 0; i < totalBatches; i++) {
				int startIdx = i * batchSize;
				int endIdx = Math.min(startIdx + batchSize, texts.size());
				List<String> batch = texts.subList(startIdx, endIdx);

				progress("Processing batch " + (i + 1) + "/" + totalBatches,
					(double) i / totalBatches);

				try {
					List<EmbeddingResult> batchResults = processBatch(batch);
					results.addAll(batchResults);
				} catch (Exception e) {
					LOGGER.log(System.Logger.Level.WARNING, "Batch " + (i + 1) + " failed", e);
					failedTexts.addAll(batch);
				}
			}

			progress("Batch embedding complete", 1.0);

			Duration duration = Duration.between(startTime, Instant.now());
			boolean success = failedTexts.isEmpty();

			return new BatchEmbeddingResult.Builder()
				.success(success)
				.message(String.format("Processed %d texts, %d failed", texts.size(), failedTexts.size()))
				.results(results)
				.totalTexts(texts.size())
				.successfulTexts(results.size())
				.failedTexts(failedTexts.size())
				.duration(duration)
				.build();

		} catch (Exception e) {
			String errorMsg = "Batch embedding failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new BatchEmbeddingResult.Builder()
				.success(false)
				.message(errorMsg)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Get embedding asynchronously
	 */
	public CompletableFuture<EmbeddingResult> getEmbeddingAsync(String text) {
		return getEmbeddingAsync(text, false);
	}

	/**
	 * Get simple embedding asynchronously
	 */
	public CompletableFuture<EmbeddingResult> getSimpleEmbeddingAsync(String text) {
		return getEmbeddingAsync(text, true);
	}

	/**
	 * Get batch embeddings asynchronously
	 */
	public CompletableFuture<BatchEmbeddingResult> getBatchEmbeddingsAsync(List<String> texts) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();
		return CompletableFuture.supplyAsync(() -> getBatchEmbeddings(texts), exec);
	}

	/**
	 * Compare two texts using cosine similarity of their embeddings
	 */
	public SimilarityResult compareEmbeddings(String text1, String text2) {
		try {
			EmbeddingResult embedding1 = getEmbedding(text1);
			EmbeddingResult embedding2 = getEmbedding(text2);

			if (!embedding1.isSuccess() || !embedding2.isSuccess()) {
				return new SimilarityResult.Builder()
					.success(false)
					.message("Failed to get embeddings for comparison")
					.build();
			}

			double similarity = calculateCosineSimilarity(
				embedding1.getEmbedding(), embedding2.getEmbedding());

			return new SimilarityResult.Builder()
				.success(true)
				.message("Similarity calculated successfully")
				.text1(text1)
				.text2(text2)
				.similarity(similarity)
				.embedding1(embedding1.getEmbedding())
				.embedding2(embedding2.getEmbedding())
				.build();

		} catch (Exception e) {
			return new SimilarityResult.Builder()
				.success(false)
				.message("Similarity calculation failed: " + e.getMessage())
				.error(e)
				.build();
		}
	}

	/**
	 * Find most similar texts from a list
	 */
	public SimilaritySearchResult findMostSimilar(String queryText, List<String> candidateTexts, int topK) {
		try {
			EmbeddingResult queryEmbedding = getEmbedding(queryText);
			if (!queryEmbedding.isSuccess()) {
				return new SimilaritySearchResult.Builder()
					.success(false)
					.message("Failed to get query embedding")
					.build();
			}

			BatchEmbeddingResult candidateEmbeddings = getBatchEmbeddings(candidateTexts);
			if (!candidateEmbeddings.isSuccess()) {
				return new SimilaritySearchResult.Builder()
					.success(false)
					.message("Failed to get candidate embeddings")
					.build();
			}

			List<SimilarityMatch> matches = new ArrayList<>();
			List<Double> queryEmb = queryEmbedding.getEmbedding();

			for (int i = 0; i < candidateEmbeddings.getResults().size(); i++) {
				EmbeddingResult candEmb = candidateEmbeddings.getResults().get(i);
				if (candEmb.isSuccess()) {
					double similarity = calculateCosineSimilarity(queryEmb, candEmb.getEmbedding());
					matches.add(new SimilarityMatch(candidateTexts.get(i), similarity, i));
				}
			}

			// Sort by similarity (descending) and take top K
			matches.sort((a, b) -> Double.compare(b.getSimilarity(), a.getSimilarity()));
			List<SimilarityMatch> topMatches = matches.subList(0, Math.min(topK, matches.size()));

			return new SimilaritySearchResult.Builder()
				.success(true)
				.message(String.format("Found %d similar texts", topMatches.size()))
				.queryText(queryText)
				.topMatches(topMatches)
				.totalCandidates(candidateTexts.size())
				.build();

		} catch (Exception e) {
			return new SimilaritySearchResult.Builder()
				.success(false)
				.message("Similarity search failed: " + e.getMessage())
				.error(e)
				.build();
		}
	}

	/**
	 * Test server health and embedding functionality
	 */
	public HealthCheckResult healthCheck() {
		try {
			EmbeddingResult testResult = getEmbedding("Health check test");

			if (testResult.isSuccess()) {
				return new HealthCheckResult.Builder()
					.healthy(true)
					.message("Server is responding and embedding generation works")
					.serverUrl(serverUrl)
					.responseTime(testResult.getDuration())
					.embeddingDimension(testResult.getEmbedding().size())
					.build();
			} else {
				return new HealthCheckResult.Builder()
					.healthy(false)
					.message("Server responded but embedding generation failed")
					.serverUrl(serverUrl)
					.error(testResult.getError().orElse(null))
					.build();
			}

		} catch (Exception e) {
			return new HealthCheckResult.Builder()
				.healthy(false)
				.message("Server health check failed: " + e.getMessage())
				.serverUrl(serverUrl)
				.error(e)
				.build();
		}
	}

	// Private helper methods
	private EmbeddingResult getEmbedding(String text, boolean useSimpleEndpoint) {
		Instant startTime = Instant.now();

		for (int attempt = 0; attempt <= retryAttempts; attempt++) {
			try {
				if (useSimpleEndpoint) {
					return getSimpleEmbeddingInternal(text, startTime);
				} else {
					return getStandardEmbeddingInternal(text, startTime);
				}
			} catch (Exception e) {
				if (attempt == retryAttempts) {
					return new EmbeddingResult.Builder()
						.success(false)
						.message("Embedding request failed after " + retryAttempts + " attempts: " + e.getMessage())
						.text(text)
						.duration(Duration.between(startTime, Instant.now()))
						.error(e)
						.build();
				}

				try {
					Thread.sleep(retryDelay.toMillis());
				} catch (InterruptedException ie) {
					Thread.currentThread().interrupt();
					break;
				}
			}
		}

		return new EmbeddingResult.Builder()
			.success(false)
			.message("Embedding request interrupted")
			.text(text)
			.duration(Duration.between(startTime, Instant.now()))
			.build();
	}

	private EmbeddingResult getStandardEmbeddingInternal(String text, Instant startTime) throws IOException, InterruptedException {
		Map<String, Object> request = Map.of("input", text);
		String json = MAPPER.writeValueAsString(request);

		HttpRequest httpRequest = HttpRequest.newBuilder()
			.uri(URI.create(serverUrl + "/v1/embeddings"))
			.header("Content-Type", "application/json")
			.POST(HttpRequest.BodyPublishers.ofString(json))
			.timeout(timeout)
			.build();

		HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

		if (response.statusCode() != 200) {
			throw new IOException("Server returned status " + response.statusCode() + ": " + response.body());
		}

		Map<String, Object> responseData = MAPPER.readValue(response.body(), Map.class);
		List<Map<String, Object>> dataList = (List<Map<String, Object>>) responseData.get("data");
		List<Double> embedding = (List<Double>) dataList.get(0).get("embedding");

		Duration duration = Duration.between(startTime, Instant.now());

		return new EmbeddingResult.Builder()
			.success(true)
			.message("Embedding generated successfully")
			.text(text)
			.embedding(embedding)
			.duration(duration)
			.build();
	}

	private EmbeddingResult getSimpleEmbeddingInternal(String text, Instant startTime) throws IOException, InterruptedException {
		Map<String, String> request = Map.of("content", text);
		String json = MAPPER.writeValueAsString(request);

		HttpRequest httpRequest = HttpRequest.newBuilder()
			.uri(URI.create(serverUrl + "/embedding"))
			.header("Content-Type", "application/json")
			.POST(HttpRequest.BodyPublishers.ofString(json))
			.timeout(timeout)
			.build();

		HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

		if (response.statusCode() != 200) {
			throw new IOException("Server returned status " + response.statusCode() + ": " + response.body());
		}

		Map<String, Object> responseData = MAPPER.readValue(response.body(), Map.class);
		List<Double> embedding = (List<Double>) responseData.get("embedding");

		Duration duration = Duration.between(startTime, Instant.now());

		return new EmbeddingResult.Builder()
			.success(true)
			.message("Simple embedding generated successfully")
			.text(text)
			.embedding(embedding)
			.duration(duration)
			.build();
	}

	private CompletableFuture<EmbeddingResult> getEmbeddingAsync(String text, boolean useSimpleEndpoint) {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();
		return CompletableFuture.supplyAsync(() -> getEmbedding(text, useSimpleEndpoint), exec);
	}

	private List<EmbeddingResult> processBatch(List<String> batch) {
		List<EmbeddingResult> results = new ArrayList<>();
		for (String text : batch) {
			EmbeddingResult result = getEmbedding(text);
			results.add(result);
		}
		return results;
	}

	private double calculateCosineSimilarity(List<Double> vector1, List<Double> vector2) {
		if (vector1.size() != vector2.size()) {
			throw new IllegalArgumentException("Vectors must have the same dimension");
		}

		double dotProduct = 0.0;
		double norm1 = 0.0;
		double norm2 = 0.0;

		for (int i = 0; i < vector1.size(); i++) {
			double v1 = vector1.get(i);
			double v2 = vector2.get(i);
			dotProduct += v1 * v2;
			norm1 += v1 * v1;
			norm2 += v2 * v2;
		}

		return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new EmbeddingProgress(message, progress));
		}
	}

	@Override
	public void close() {
		if (executor != null) {
			executor.shutdown();
		}
	}

	// Builder class
	public static class Builder {
		private String serverUrl = "http://localhost:8080";
		private Duration timeout = Duration.ofMinutes(5);
		private int retryAttempts = 3;
		private Duration retryDelay = Duration.ofSeconds(1);
		private int batchSize = 50;
		private Consumer<EmbeddingProgress> progressCallback;
		private ExecutorService executor;
		private boolean enableMetrics = false;

		public Builder serverUrl(String serverUrl) {
			this.serverUrl = Objects.requireNonNull(serverUrl);
			return this;
		}

		public Builder timeout(Duration timeout) {
			this.timeout = timeout;
			return this;
		}

		public Builder retryAttempts(int retryAttempts) {
			this.retryAttempts = Math.max(0, retryAttempts);
			return this;
		}

		public Builder retryDelay(Duration retryDelay) {
			this.retryDelay = retryDelay;
			return this;
		}

		public Builder batchSize(int batchSize) {
			this.batchSize = Math.max(1, batchSize);
			return this;
		}

		public Builder progressCallback(Consumer<EmbeddingProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public Builder enableMetrics(boolean enableMetrics) {
			this.enableMetrics = enableMetrics;
			return this;
		}

		public ServerEmbeddingHttpClientLibrary build() {
			return new ServerEmbeddingHttpClientLibrary(this);
		}
	}

	// Progress tracking class
	public static class EmbeddingProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public EmbeddingProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	// Result classes
	public static class EmbeddingResult {
		private final boolean success;
		private final String message;
		private final String text;
		private final List<Double> embedding;
		private final Duration duration;
		private final Exception error;

		private EmbeddingResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.text = builder.text;
			this.embedding = builder.embedding != null ?
				Collections.unmodifiableList(builder.embedding) : Collections.emptyList();
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public String getText() { return text; }
		public List<Double> getEmbedding() { return embedding; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private String text;
			private List<Double> embedding;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder text(String text) { this.text = text; return this; }
			public Builder embedding(List<Double> embedding) { this.embedding = embedding; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public EmbeddingResult build() { return new EmbeddingResult(this); }
		}
	}

	public static class BatchEmbeddingResult {
		private final boolean success;
		private final String message;
		private final List<EmbeddingResult> results;
		private final int totalTexts;
		private final int successfulTexts;
		private final int failedTexts;
		private final Duration duration;
		private final Exception error;

		private BatchEmbeddingResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.results = Collections.unmodifiableList(builder.results);
			this.totalTexts = builder.totalTexts;
			this.successfulTexts = builder.successfulTexts;
			this.failedTexts = builder.failedTexts;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public List<EmbeddingResult> getResults() { return results; }
		public int getTotalTexts() { return totalTexts; }
		public int getSuccessfulTexts() { return successfulTexts; }
		public int getFailedTexts() { return failedTexts; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }
		public double getSuccessRate() { return totalTexts > 0 ? (double) successfulTexts / totalTexts : 0.0; }

		public static class Builder {
			private boolean success;
			private String message;
			private List<EmbeddingResult> results = new ArrayList<>();
			private int totalTexts;
			private int successfulTexts;
			private int failedTexts;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder results(List<EmbeddingResult> results) { this.results = results; return this; }
			public Builder totalTexts(int totalTexts) { this.totalTexts = totalTexts; return this; }
			public Builder successfulTexts(int successfulTexts) { this.successfulTexts = successfulTexts; return this; }
			public Builder failedTexts(int failedTexts) { this.failedTexts = failedTexts; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public BatchEmbeddingResult build() { return new BatchEmbeddingResult(this); }
		}
	}

	public static class SimilarityResult {
		private final boolean success;
		private final String message;
		private final String text1;
		private final String text2;
		private final double similarity;
		private final List<Double> embedding1;
		private final List<Double> embedding2;
		private final Exception error;

		private SimilarityResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.text1 = builder.text1;
			this.text2 = builder.text2;
			this.similarity = builder.similarity;
			this.embedding1 = builder.embedding1 != null ?
				Collections.unmodifiableList(builder.embedding1) : Collections.emptyList();
			this.embedding2 = builder.embedding2 != null ?
				Collections.unmodifiableList(builder.embedding2) : Collections.emptyList();
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public String getText1() { return text1; }
		public String getText2() { return text2; }
		public double getSimilarity() { return similarity; }
		public List<Double> getEmbedding1() { return embedding1; }
		public List<Double> getEmbedding2() { return embedding2; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private String text1;
			private String text2;
			private double similarity;
			private List<Double> embedding1;
			private List<Double> embedding2;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder text1(String text1) { this.text1 = text1; return this; }
			public Builder text2(String text2) { this.text2 = text2; return this; }
			public Builder similarity(double similarity) { this.similarity = similarity; return this; }
			public Builder embedding1(List<Double> embedding1) { this.embedding1 = embedding1; return this; }
			public Builder embedding2(List<Double> embedding2) { this.embedding2 = embedding2; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public SimilarityResult build() { return new SimilarityResult(this); }
		}
	}

	public static class SimilarityMatch {
		private final String text;
		private final double similarity;
		private final int originalIndex;

		public SimilarityMatch(String text, double similarity, int originalIndex) {
			this.text = text;
			this.similarity = similarity;
			this.originalIndex = originalIndex;
		}

		public String getText() { return text; }
		public double getSimilarity() { return similarity; }
		public int getOriginalIndex() { return originalIndex; }
	}

	public static class SimilaritySearchResult {
		private final boolean success;
		private final String message;
		private final String queryText;
		private final List<SimilarityMatch> topMatches;
		private final int totalCandidates;
		private final Exception error;

		private SimilaritySearchResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.queryText = builder.queryText;
			this.topMatches = Collections.unmodifiableList(builder.topMatches);
			this.totalCandidates = builder.totalCandidates;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public String getQueryText() { return queryText; }
		public List<SimilarityMatch> getTopMatches() { return topMatches; }
		public int getTotalCandidates() { return totalCandidates; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private String queryText;
			private List<SimilarityMatch> topMatches = new ArrayList<>();
			private int totalCandidates;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder queryText(String queryText) { this.queryText = queryText; return this; }
			public Builder topMatches(List<SimilarityMatch> topMatches) { this.topMatches = topMatches; return this; }
			public Builder totalCandidates(int totalCandidates) { this.totalCandidates = totalCandidates; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public SimilaritySearchResult build() { return new SimilaritySearchResult(this); }
		}
	}

	public static class HealthCheckResult {
		private final boolean healthy;
		private final String message;
		private final String serverUrl;
		private final Duration responseTime;
		private final int embeddingDimension;
		private final Exception error;

		private HealthCheckResult(Builder builder) {
			this.healthy = builder.healthy;
			this.message = builder.message;
			this.serverUrl = builder.serverUrl;
			this.responseTime = builder.responseTime;
			this.embeddingDimension = builder.embeddingDimension;
			this.error = builder.error;
		}

		public boolean isHealthy() { return healthy; }
		public String getMessage() { return message; }
		public String getServerUrl() { return serverUrl; }
		public Duration getResponseTime() { return responseTime; }
		public int getEmbeddingDimension() { return embeddingDimension; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean healthy;
			private String message;
			private String serverUrl;
			private Duration responseTime = Duration.ZERO;
			private int embeddingDimension;
			private Exception error;

			public Builder healthy(boolean healthy) { this.healthy = healthy; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder serverUrl(String serverUrl) { this.serverUrl = serverUrl; return this; }
			public Builder responseTime(Duration responseTime) { this.responseTime = responseTime; return this; }
			public Builder embeddingDimension(int embeddingDimension) { this.embeddingDimension = embeddingDimension; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public HealthCheckResult build() { return new HealthCheckResult(this); }
		}
	}
}