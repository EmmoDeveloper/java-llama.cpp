package de.kherud.llama.testing;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.logging.Logger;

public class ServerEmbeddingHttpClient extends ServerEmbeddingClient {

	private static final System.Logger LOGGER = System.getLogger(ServerEmbeddingHttpClient.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final String serverUrl;
	private final HttpClient httpClient;

	public ServerEmbeddingHttpClient(String serverUrl) {
		this.serverUrl = serverUrl.endsWith("/") ? serverUrl.substring(0, serverUrl.length() - 1) : serverUrl;
		this.httpClient = HttpClient.newBuilder()
			.connectTimeout(Duration.ofSeconds(30))
			.build();
	}

	@Override
	public EmbeddingResponse getEmbeddings(String text) throws IOException, InterruptedException {
		return getEmbeddings(new EmbeddingRequest(text));
	}

	@Override
	public EmbeddingResponse getEmbeddings(List<String> texts) throws IOException, InterruptedException {
		return getEmbeddings(new EmbeddingRequest(texts));
	}

	@Override
	public EmbeddingResponse getEmbeddings(String[] texts) throws IOException, InterruptedException {
		return getEmbeddings(new EmbeddingRequest(texts));
	}

	@Override
	public EmbeddingResponse getEmbeddings(EmbeddingRequest request) throws IOException, InterruptedException {
		String json = MAPPER.writeValueAsString(request);

		HttpRequest httpRequest = HttpRequest.newBuilder()
			.uri(URI.create(serverUrl + "/v1/embeddings"))
			.header("Content-Type", "application/json")
			.POST(HttpRequest.BodyPublishers.ofString(json))
			.timeout(Duration.ofMinutes(5))
			.build();

		HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

		if (response.statusCode() != 200) {
			throw new IOException("Server returned status " + response.statusCode() + ": " + response.body());
		}

		return MAPPER.readValue(response.body(), EmbeddingResponse.class);
	}

	@Override
	public SimpleEmbeddingResponse getSimpleEmbedding(String text) throws IOException, InterruptedException {
		SimpleEmbeddingRequest request = new SimpleEmbeddingRequest(text);
		String json = MAPPER.writeValueAsString(request);

		HttpRequest httpRequest = HttpRequest.newBuilder()
			.uri(URI.create(serverUrl + "/embedding"))
			.header("Content-Type", "application/json")
			.POST(HttpRequest.BodyPublishers.ofString(json))
			.timeout(Duration.ofMinutes(5))
			.build();

		HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

		if (response.statusCode() != 200) {
			throw new IOException("Server returned status " + response.statusCode() + ": " + response.body());
		}

		return MAPPER.readValue(response.body(), SimpleEmbeddingResponse.class);
	}

	@Override
	public CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(String text) {
		return getEmbeddingsAsync(new EmbeddingRequest(text));
	}

	@Override
	public CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(List<String> texts) {
		return getEmbeddingsAsync(new EmbeddingRequest(texts));
	}

	@Override
	public CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(EmbeddingRequest request) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return getEmbeddings(request);
			} catch (IOException | InterruptedException e) {
				throw new RuntimeException(e);
			}
		});
	}

	@Override
	public CompletableFuture<SimpleEmbeddingResponse> getSimpleEmbeddingAsync(String text) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return getSimpleEmbedding(text);
			} catch (IOException | InterruptedException e) {
				throw new RuntimeException(e);
			}
		});
	}

	@Override
	public boolean isServerHealthy() {
		try {
			HttpRequest request = HttpRequest.newBuilder()
				.uri(URI.create(serverUrl + "/health"))
				.timeout(Duration.ofSeconds(10))
				.GET()
				.build();

			HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
			return response.statusCode() == 200;
		} catch (Exception e) {
			return false;
		}
	}

	@Override
	public void waitForServer(long maxWaitSeconds) throws InterruptedException {
		long startTime = System.currentTimeMillis();
		long maxWaitMillis = maxWaitSeconds * 1000;

		while (System.currentTimeMillis() - startTime < maxWaitMillis) {
			if (isServerHealthy()) {
				LOGGER.log(System.Logger.Level.INFO,"Server is healthy and ready");
				return;
			}
			Thread.sleep(1000);
		}

		throw new RuntimeException("Server did not become healthy within " + maxWaitSeconds + " seconds");
	}

	@Override
	public void close() {
		// HttpClient doesn't need explicit closing in modern Java
	}

	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(ServerEmbeddingHttpClient::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length < 1) {
			throw new IllegalArgumentException("Usage: ServerEmbeddingHttpClient <server_url> [text]");
		}

		String serverUrl = args[0];
		String text = args.length > 1 ? args[1] : "This is a test sentence for embedding generation.";

		ServerEmbeddingHttpClient client = new ServerEmbeddingHttpClient(serverUrl);

		// Check server health
		if (!client.isServerHealthy()) {
			throw new RuntimeException("Server is not healthy at " + serverUrl);
		}

		try {
			System.out.println("Server is healthy, generating embeddings...");

			// Test single embedding
			EmbeddingResponse response = client.getEmbeddings(text);
			List<Double> embedding = response.getData().get(0).getEmbeddingAsDoubles();
			System.out.printf("Generated embedding with %d dimensions for text: %s%n",
				embedding.size(), text.length() > 50 ? text.substring(0, 47) + "..." : text);
			System.out.println("First 8 values: " + embedding.subList(0, Math.min(8, embedding.size())));
			System.out.println("Last 8 values: " + embedding.subList(Math.max(0, embedding.size() - 8), embedding.size()));

			// Test multiple embeddings
			List<String> texts = Arrays.asList(
				"Hello world",
				"This is a test",
				"Another test sentence",
				"Final example text"
			);

			EmbeddingResponse multiResponse = client.getEmbeddings(texts);
			System.out.printf("Generated %d embeddings for batch request%n", multiResponse.getData().size());

			// Calculate similarities
			if (multiResponse.getData().size() >= 2) {
				List<Double> emb1 = multiResponse.getData().get(0).getEmbeddingAsDoubles();
				List<Double> emb2 = multiResponse.getData().get(1).getEmbeddingAsDoubles();
				double similarity = cosineSimilarity(emb1, emb2);
				System.out.printf("Cosine similarity between first two texts: %.4f%n", similarity);
			}

			// Run benchmark
			EmbeddingBenchmark benchmark = client.benchmark(text, 10);
			System.out.println("Benchmark result: " + benchmark);

		} finally {
			client.close();
		}
	}
}
