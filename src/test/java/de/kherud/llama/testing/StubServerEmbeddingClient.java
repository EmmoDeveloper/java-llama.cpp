package de.kherud.llama.testing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Stub implementation of ServerEmbeddingClient for testing.
 * Does not require a real server - generates fake but consistent embeddings.
 */
public class StubServerEmbeddingClient extends ServerEmbeddingClient {

	private boolean healthy;

	public StubServerEmbeddingClient() {
		this.healthy = true; // Stub is always "healthy"
	}

	public StubServerEmbeddingClient(boolean healthy) {
		this.healthy = healthy;
	}

	@Override
	public EmbeddingResponse getEmbeddings(String text) throws Exception {
		return createFakeResponse(Arrays.asList(text));
	}

	@Override
	public EmbeddingResponse getEmbeddings(List<String> texts) throws Exception {
		return createFakeResponse(texts);
	}

	@Override
	public EmbeddingResponse getEmbeddings(String[] texts) throws Exception {
		return createFakeResponse(Arrays.asList(texts));
	}

	@Override
	public EmbeddingResponse getEmbeddings(EmbeddingRequest request) throws Exception {
		Object input = request.getInput();
		if (input instanceof String) {
			return getEmbeddings((String) input);
		} else if (input instanceof List) {
			@SuppressWarnings("unchecked")
			List<String> texts = (List<String>) input;
			return getEmbeddings(texts);
		}
		throw new IllegalArgumentException("Unsupported input type");
	}

	@Override
	public SimpleEmbeddingResponse getSimpleEmbedding(String text) throws Exception {
		List<Double> embedding = generateFakeEmbedding(text);
		SimpleEmbeddingResponse response = new SimpleEmbeddingResponse();
		setEmbedding(response, embedding);
		return response;
	}

	@Override
	public CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(String text) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return getEmbeddings(text);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		});
	}

	@Override
	public CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(List<String> texts) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return getEmbeddings(texts);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		});
	}

	@Override
	public CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(EmbeddingRequest request) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return getEmbeddings(request);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		});
	}

	@Override
	public CompletableFuture<SimpleEmbeddingResponse> getSimpleEmbeddingAsync(String text) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return getSimpleEmbedding(text);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		});
	}

	@Override
	public boolean isServerHealthy() {
		return healthy;
	}

	@Override
	public void waitForServer(long maxWaitSeconds) throws InterruptedException {
		// Stub implementation - either return immediately (if healthy) or timeout
		if (!healthy) {
			Thread.sleep(maxWaitSeconds * 1000);
			throw new RuntimeException("Server timeout (stub)");
		}
	}

	@Override
	public void close() {
		// Nothing to close in stub
	}

	// Helper methods

	private EmbeddingResponse createFakeResponse(List<String> texts) {
		EmbeddingResponse response = new EmbeddingResponse();
		List<EmbeddingData> dataList = new ArrayList<>();

		for (int i = 0; i < texts.size(); i++) {
			EmbeddingData data = new EmbeddingData();
			List<Double> embedding = generateFakeEmbedding(texts.get(i));
			setEmbeddingData(data, embedding, i);
			dataList.add(data);
		}

		setResponseData(response, dataList);
		return response;
	}

	private List<Double> generateFakeEmbedding(String text) {
		// Generate a consistent fake embedding based on text hash
		int hash = text.hashCode();
		List<Double> embedding = new ArrayList<>();

		// Create a 384-dimensional embedding (common size)
		for (int i = 0; i < 384; i++) {
			// Use hash and index to create deterministic but varied values
			double value = Math.sin(hash + i * 0.1) * 0.5;
			embedding.add(value);
		}

		// Normalize to unit vector
		double norm = Math.sqrt(embedding.stream().mapToDouble(x -> x * x).sum());
		if (norm > 0) {
			embedding.replaceAll(x -> x / norm);
		}

		return embedding;
	}

	// Reflection helpers to set private fields in response objects

	private void setEmbeddingData(EmbeddingData data, List<Double> embedding, int index) {
		try {
			java.lang.reflect.Field embeddingField = EmbeddingData.class.getDeclaredField("embedding");
			embeddingField.setAccessible(true);
			embeddingField.set(data, embedding);

			java.lang.reflect.Field indexField = EmbeddingData.class.getDeclaredField("index");
			indexField.setAccessible(true);
			indexField.set(data, index);
		} catch (Exception e) {
			throw new RuntimeException("Failed to set embedding data", e);
		}
	}

	private void setResponseData(EmbeddingResponse response, List<EmbeddingData> dataList) {
		try {
			java.lang.reflect.Field dataField = EmbeddingResponse.class.getDeclaredField("data");
			dataField.setAccessible(true);
			dataField.set(response, dataList);

			java.lang.reflect.Field modelField = EmbeddingResponse.class.getDeclaredField("model");
			modelField.setAccessible(true);
			modelField.set(response, "stub-model");

			// Set fake usage
			Usage usage = new Usage();
			setUsage(usage, dataList.size() * 10, dataList.size() * 10);

			java.lang.reflect.Field usageField = EmbeddingResponse.class.getDeclaredField("usage");
			usageField.setAccessible(true);
			usageField.set(response, usage);
		} catch (Exception e) {
			throw new RuntimeException("Failed to set response data", e);
		}
	}

	private void setUsage(Usage usage, int promptTokens, int totalTokens) {
		try {
			java.lang.reflect.Field promptField = Usage.class.getDeclaredField("promptTokens");
			promptField.setAccessible(true);
			promptField.set(usage, promptTokens);

			java.lang.reflect.Field totalField = Usage.class.getDeclaredField("totalTokens");
			totalField.setAccessible(true);
			totalField.set(usage, totalTokens);
		} catch (Exception e) {
			throw new RuntimeException("Failed to set usage", e);
		}
	}

	private void setEmbedding(SimpleEmbeddingResponse response, List<Double> embedding) {
		try {
			java.lang.reflect.Field embeddingField = SimpleEmbeddingResponse.class.getDeclaredField("embedding");
			embeddingField.setAccessible(true);
			embeddingField.set(response, embedding);
		} catch (Exception e) {
			throw new RuntimeException("Failed to set simple embedding", e);
		}
	}
}