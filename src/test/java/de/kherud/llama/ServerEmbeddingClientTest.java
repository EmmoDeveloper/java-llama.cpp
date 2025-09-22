package de.kherud.llama;

import de.kherud.llama.testing.ServerEmbeddingClient;
import de.kherud.llama.testing.ServerEmbeddingHttpClient;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class ServerEmbeddingClientTest {

	private ServerEmbeddingClient client;
	private boolean skipTests = true; // Skip by default since no server is running

	@Before
	public void setUp() {
		client = new ServerEmbeddingHttpClient("http://127.0.0.1:8080");

		// Only run tests if we can detect a running server
		skipTests = !client.isServerHealthy();
	}

	@After
	public void tearDown() {
		if (client != null) {
			client.close();
		}
	}

	@Test
	public void testClientCreation() {
		// This test always runs - just tests object creation
		assertNotNull(client);

		ServerEmbeddingClient client2 = new ServerEmbeddingHttpClient("http://localhost:8080/");
		assertNotNull(client2);
	}

	@Test
	public void testHealthCheck() {
		// Test health check functionality (may return false if no server)
		boolean isHealthy = client.isServerHealthy();
		// We can't assert true/false since no server may be running
		// Just ensure the method doesn't throw
	}

	@Test
	public void testSingleEmbedding() throws Exception {
		if (skipTests) {
			System.out.println("Skipping test - no server detected at http://127.0.0.1:8080");
			return;
		}

		String text = "This is a test sentence.";
		ServerEmbeddingClient.EmbeddingResponse response = client.getEmbeddings(text);

		assertNotNull(response);
		assertNotNull(response.getData());
		assertEquals(1, response.getData().size());

		ServerEmbeddingClient.EmbeddingData data = response.getData().get(0);
		assertNotNull(data.getEmbedding());
		assertEquals("embedding", data.getObject());

		List<Double> embedding = data.getEmbeddingAsDoubles();
		assertTrue("Embedding should have multiple dimensions", embedding.size() > 1);
	}

	@Test
	public void testMultipleEmbeddings() throws Exception {
		if (skipTests) {
			System.out.println("Skipping test - no server detected at http://127.0.0.1:8080");
			return;
		}

		List<String> texts = Arrays.asList(
			"First test sentence.",
			"Second test sentence.",
			"Third test sentence."
		);

		ServerEmbeddingClient.EmbeddingResponse response = client.getEmbeddings(texts);

		assertNotNull(response);
		assertNotNull(response.getData());
		assertEquals(3, response.getData().size());

		for (int i = 0; i < 3; i++) {
			ServerEmbeddingClient.EmbeddingData data = response.getData().get(i);
			assertNotNull(data.getEmbedding());
			assertEquals(i, data.getIndex());

			List<Double> embedding = data.getEmbeddingAsDoubles();
			assertTrue("Embedding should have multiple dimensions", embedding.size() > 1);
		}
	}

	@Test
	public void testArrayEmbeddings() throws Exception {
		if (skipTests) {
			System.out.println("Skipping test - no server detected at http://127.0.0.1:8080");
			return;
		}

		String[] texts = {
			"Array test one.",
			"Array test two."
		};

		ServerEmbeddingClient.EmbeddingResponse response = client.getEmbeddings(texts);

		assertNotNull(response);
		assertNotNull(response.getData());
		assertEquals(2, response.getData().size());
	}

	@Test
	public void testSimpleEmbedding() throws Exception {
		if (skipTests) {
			System.out.println("Skipping test - no server detected at http://127.0.0.1:8080");
			return;
		}

		String text = "Simple embedding test.";
		ServerEmbeddingClient.SimpleEmbeddingResponse response = client.getSimpleEmbedding(text);

		assertNotNull(response);
		assertNotNull(response.getEmbedding());
		assertTrue("Embedding should have multiple dimensions", response.getEmbedding().size() > 1);
	}

	@Test
	public void testCosineSimilarity() {
		// Test the utility function
		List<Double> vec1 = Arrays.asList(1.0, 0.0, 0.0);
		List<Double> vec2 = Arrays.asList(0.0, 1.0, 0.0);
		List<Double> vec3 = Arrays.asList(1.0, 0.0, 0.0);

		double similarity1 = ServerEmbeddingClient.cosineSimilarity(vec1, vec2);
		assertEquals(0.0, similarity1, 0.001);

		double similarity2 = ServerEmbeddingClient.cosineSimilarity(vec1, vec3);
		assertEquals(1.0, similarity2, 0.001);
	}

	@Test
	public void testAsyncEmbedding() throws Exception {
		if (skipTests) {
			System.out.println("Skipping test - no server detected at http://127.0.0.1:8080");
			return;
		}

		String text = "Async test sentence.";
		ServerEmbeddingClient.EmbeddingResponse response = client.getEmbeddingsAsync(text).join();

		assertNotNull(response);
		assertNotNull(response.getData());
		assertEquals(1, response.getData().size());
	}

	@Test
	public void testBenchmarkStructure() throws Exception {
		if (skipTests) {
			System.out.println("Skipping test - no server detected at http://127.0.0.1:8080");
			return;
		}

		String text = "Benchmark test.";
		ServerEmbeddingClient.EmbeddingBenchmark benchmark = client.benchmark(text, 2);

		assertNotNull(benchmark);
		assertEquals(text, benchmark.text);
		assertEquals(2, benchmark.requestCount);
		assertTrue(benchmark.totalTimeMillis >= 0);
		assertTrue(benchmark.averageLatencyMillis >= 0);
		assertTrue(benchmark.requestsPerSecond >= 0);
		assertNotNull(benchmark.embedding);
	}

	@Test
	public void testEmbeddingRequestBuilder() {
		// Test request building
		ServerEmbeddingClient.EmbeddingRequest request1 = new ServerEmbeddingClient.EmbeddingRequest("test");
		assertEquals("test", request1.getInput());
		assertNull(request1.getEncodingFormat());

		ServerEmbeddingClient.EmbeddingRequest request2 = new ServerEmbeddingClient.EmbeddingRequest("test")
			.encodingFormat("base64");
		assertEquals("test", request2.getInput());
		assertEquals("base64", request2.getEncodingFormat());

		List<String> texts = Arrays.asList("test1", "test2");
		ServerEmbeddingClient.EmbeddingRequest request3 = new ServerEmbeddingClient.EmbeddingRequest(texts);
		assertEquals(texts, request3.getInput());

		String[] textArray = {"test1", "test2"};
		ServerEmbeddingClient.EmbeddingRequest request4 = new ServerEmbeddingClient.EmbeddingRequest(textArray);
		assertTrue(request4.getInput() instanceof List);
	}

	@Test
	public void testErrorHandling() {
		// Test with invalid server URL
		ServerEmbeddingClient badClient = new ServerEmbeddingHttpClient("http://invalid-server:9999");
		assertFalse(badClient.isServerHealthy());

		try {
			badClient.getEmbeddings("test");
			fail("Should have thrown exception for invalid server");
		} catch (Exception e) {
			// Expected
		}
	}

	@Test
	public void testMainMethodHandling() {
		// Test that main method handles arguments appropriately
		try {
			ServerEmbeddingHttpClient.main(new String[]{});
			fail("Should exit with error for no arguments");
		} catch (Exception e) {
			// Expected - main method should exit or throw
		}
	}
}