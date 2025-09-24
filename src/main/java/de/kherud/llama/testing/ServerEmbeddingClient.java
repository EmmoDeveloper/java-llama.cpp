package de.kherud.llama.testing;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.concurrent.CompletableFuture;

public abstract class ServerEmbeddingClient {

	public static class EmbeddingRequest {
		private Object input;
		@JsonProperty("encoding_format")
		private String encodingFormat;

		public EmbeddingRequest(String input) {
			this.input = input;
		}

		public EmbeddingRequest(List<String> input) {
			this.input = input;
		}

		public EmbeddingRequest(String[] input) {
			this.input = java.util.Arrays.asList(input);
		}

		public EmbeddingRequest encodingFormat(String format) {
			this.encodingFormat = format;
			return this;
		}

		public Object getInput() { return input; }
		public String getEncodingFormat() { return encodingFormat; }
	}

	public static class Usage {
		@JsonProperty("prompt_tokens")
		private int promptTokens;
		@JsonProperty("total_tokens")
		private int totalTokens;

		public int getPromptTokens() { return promptTokens; }
		public int getTotalTokens() { return totalTokens; }
	}

	public static class EmbeddingData {
		private Object embedding;
		private int index;
		private String object = "embedding";

		public Object getEmbedding() { return embedding; }
		public int getIndex() { return index; }
		public String getObject() { return object; }

		@SuppressWarnings("unchecked")
		public List<Double> getEmbeddingAsDoubles() {
			if (embedding instanceof List) {
				return (List<Double>) embedding;
			}
			throw new IllegalStateException("Embedding is not in double array format");
		}
	}

	public static class EmbeddingResponse {
		private String object = "list";
		private List<EmbeddingData> data;
		private String model;
		private Usage usage;

		public String getObject() { return object; }
		public List<EmbeddingData> getData() { return data; }
		public String getModel() { return model; }
		public Usage getUsage() { return usage; }
	}

	public static class SimpleEmbeddingRequest {
		private String content;

		public SimpleEmbeddingRequest(String content) {
			this.content = content;
		}

		public String getContent() { return content; }
	}

	public static class SimpleEmbeddingResponse {
		private List<Double> embedding;

		public List<Double> getEmbedding() { return embedding; }
	}

	public static class EmbeddingBenchmark {
		public final String text;
		public final int requestCount;
		public final long totalTimeMillis;
		public final double averageLatencyMillis;
		public final double requestsPerSecond;
		public final List<Double> embedding;

		public EmbeddingBenchmark(String text, int requestCount, long totalTimeMillis, List<Double> embedding) {
			this.text = text;
			this.requestCount = requestCount;
			this.totalTimeMillis = totalTimeMillis;
			this.averageLatencyMillis = (double) totalTimeMillis / requestCount;
			this.requestsPerSecond = requestCount / (totalTimeMillis / 1000.0);
			this.embedding = embedding;
		}

		@Override
		public String toString() {
			return String.format("EmbeddingBenchmark{text='%s', requests=%d, totalTime=%dms, " +
				"avgLatency=%.2fms, requestsPerSec=%.2f}",
				text.length() > 50 ? text.substring(0, 47) + "..." : text,
				requestCount, totalTimeMillis, averageLatencyMillis, requestsPerSecond);
		}
	}

	public abstract EmbeddingResponse getEmbeddings(String text) throws Exception;
	public abstract EmbeddingResponse getEmbeddings(List<String> texts) throws Exception;
	public abstract EmbeddingResponse getEmbeddings(String[] texts) throws Exception;
	public abstract EmbeddingResponse getEmbeddings(EmbeddingRequest request) throws Exception;
	public abstract SimpleEmbeddingResponse getSimpleEmbedding(String text) throws Exception;
	public abstract CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(String text);
	public abstract CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(List<String> texts);
	public abstract CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(EmbeddingRequest request);
	public abstract CompletableFuture<SimpleEmbeddingResponse> getSimpleEmbeddingAsync(String text);
	public abstract boolean isServerHealthy();
	public abstract void waitForServer(long maxWaitSeconds) throws InterruptedException;
	public abstract void close();

	public static double cosineSimilarity(List<Double> vec1, List<Double> vec2) {
		if (vec1.size() != vec2.size()) {
			throw new IllegalArgumentException("Vectors must have the same dimension");
		}

		double dotProduct = 0.0;
		double norm1 = 0.0;
		double norm2 = 0.0;

		for (int i = 0; i < vec1.size(); i++) {
			double v1 = vec1.get(i);
			double v2 = vec2.get(i);
			dotProduct += v1 * v2;
			norm1 += v1 * v1;
			norm2 += v2 * v2;
		}

		return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
	}

	public EmbeddingBenchmark benchmark(String text, int numRequests) throws Exception {
		java.util.logging.Logger.getLogger(getClass().getName()).info(
			String.format("Benchmarking embedding generation: %d requests for text length %d",
				numRequests, text.length()));

		// Warmup
		getEmbeddings(text);

		long startTime = System.currentTimeMillis();
		EmbeddingResponse lastResponse = null;

		for (int i = 0; i < numRequests; i++) {
			lastResponse = getEmbeddings(text);
		}

		long totalTime = System.currentTimeMillis() - startTime;
		List<Double> embedding = lastResponse.getData().get(0).getEmbeddingAsDoubles();

		EmbeddingBenchmark result = new EmbeddingBenchmark(text, numRequests, totalTime, embedding);
		java.util.logging.Logger.getLogger(getClass().getName()).info("Benchmark completed: " + result);
		return result;
	}

	public EmbeddingBenchmark benchmarkAsync(String text, int numRequests) throws InterruptedException {
		java.util.logging.Logger.getLogger(getClass().getName()).info(
			String.format("Benchmarking async embedding generation: %d requests for text length %d",
				numRequests, text.length()));

		// Warmup
		try {
			getEmbeddings(text);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		long startTime = System.currentTimeMillis();

		CompletableFuture<EmbeddingResponse>[] futures = new CompletableFuture[numRequests];
		for (int i = 0; i < numRequests; i++) {
			futures[i] = getEmbeddingsAsync(text);
		}

		CompletableFuture.allOf(futures).join();
		long totalTime = System.currentTimeMillis() - startTime;

		List<Double> embedding = futures[0].join().getData().get(0).getEmbeddingAsDoubles();

		EmbeddingBenchmark result = new EmbeddingBenchmark(text, numRequests, totalTime, embedding);
		java.util.logging.Logger.getLogger(getClass().getName()).info("Async benchmark completed: " + result);
		return result;
	}
}
