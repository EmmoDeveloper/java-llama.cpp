package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.AfterClass;

public class EmbeddingTest {

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
		model = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
				.enableEmbedding()
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testBasicEmbedding() {
		System.out.println("\n=== Basic Embedding Test ===");

		float[] embedding = model.embed("Hello world");

		Assert.assertNotNull("Embedding should not be null", embedding);
		Assert.assertEquals("Should have 4096 dimensions", 4096, embedding.length);

		// Check that embeddings are not all zeros (dummy implementation check)
		boolean hasNonZeroValues = false;
		for (float value : embedding) {
			if (value != 0.0f) {
				hasNonZeroValues = true;
				break;
			}
		}
		Assert.assertTrue("Embedding should contain non-zero values", hasNonZeroValues);

		System.out.printf("Generated embedding with %d dimensions\n", embedding.length);
		System.out.printf("First few values: %.6f, %.6f, %.6f, %.6f\n",
			embedding[0], embedding[1], embedding[2], embedding[3]);

		System.out.println("✅ Basic embedding test passed!");
	}

	@Test
	public void testEmbeddingSimilarity() {
		System.out.println("\n=== Embedding Similarity Test ===");

		// Test similar texts should have similar embeddings
		float[] embedding1 = model.embed("Hello world");
		float[] embedding2 = model.embed("Hello world");
		float[] embedding3 = model.embed("Goodbye universe");

		Assert.assertEquals("Embeddings should have same dimensions",
			embedding1.length, embedding2.length);
		Assert.assertEquals("Embeddings should have same dimensions",
			embedding1.length, embedding3.length);

		// Calculate cosine similarities
		double similarity_same = cosineSimilarity(embedding1, embedding2);
		double similarity_different = cosineSimilarity(embedding1, embedding3);

		System.out.printf("Similarity (same text): %.6f\n", similarity_same);
		System.out.printf("Similarity (different text): %.6f\n", similarity_different);

		// Identical texts should have very high similarity (close to 1.0)
		Assert.assertTrue("Same text should have high similarity", similarity_same > 0.95);

		// Different texts should have lower similarity
		Assert.assertTrue("Different texts should have lower similarity",
			similarity_different < similarity_same);

		System.out.println("✅ Embedding similarity test passed!");
	}

	@Test
	public void testEmbeddingConsistency() {
		System.out.println("\n=== Embedding Consistency Test ===");

		// Test that same input produces consistent embeddings
		String testText = "The quick brown fox jumps over the lazy dog";

		float[] embedding1 = model.embed(testText);
		float[] embedding2 = model.embed(testText);
		float[] embedding3 = model.embed(testText);

		double similarity_1_2 = cosineSimilarity(embedding1, embedding2);
		double similarity_2_3 = cosineSimilarity(embedding2, embedding3);
		double similarity_1_3 = cosineSimilarity(embedding1, embedding3);

		System.out.printf("Consistency similarity 1-2: %.6f\n", similarity_1_2);
		System.out.printf("Consistency similarity 2-3: %.6f\n", similarity_2_3);
		System.out.printf("Consistency similarity 1-3: %.6f\n", similarity_1_3);

		// All should be identical (or very close due to floating point precision)
		Assert.assertTrue("Consistent embeddings should be identical", similarity_1_2 > 0.999);
		Assert.assertTrue("Consistent embeddings should be identical", similarity_2_3 > 0.999);
		Assert.assertTrue("Consistent embeddings should be identical", similarity_1_3 > 0.999);

		System.out.println("✅ Embedding consistency test passed!");
	}

	@Test
	public void testEmbeddingWithoutEmbeddingMode() {
		System.out.println("\n=== Test Without Embedding Mode ===");

		// Create model without embedding enabled
		// Note: NOT calling .enableEmbedding()

		try (LlamaModel nonEmbeddingModel = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
			// Note: NOT calling .enableEmbedding()
		)) {
			// This should throw an exception
			float[] embedding = nonEmbeddingModel.embed("Hello world");
			Assert.fail("Expected IllegalStateException when embedding is not enabled");
		} catch (IllegalStateException e) {
			System.out.println("✅ Correctly threw exception: " + e.getMessage());
			Assert.assertTrue("Exception message should mention embedding support",
				e.getMessage().contains("embedding support"));
		}

		System.out.println("✅ Non-embedding mode test passed!");
	}

	@Test
	public void testEmbeddingDifferentTexts() {
		System.out.println("\n=== Different Texts Embedding Test ===");

		String[] testTexts = {
			"Hello world",
			"Programming in Java",
			"Machine learning algorithms",
			"The weather is nice today",
			"def hello():\n    print('Hello')"
		};

		float[][] embeddings = new float[testTexts.length][];

		// Generate embeddings for all texts
		for (int i = 0; i < testTexts.length; i++) {
			embeddings[i] = model.embed(testTexts[i]);
			System.out.printf("Text %d: \"%s\" -> embedding[0-3]: %.4f, %.4f, %.4f, %.4f\n",
				i, testTexts[i],
				embeddings[i][0], embeddings[i][1],
				embeddings[i][2], embeddings[i][3]);
		}

		// Verify all embeddings are different
		for (int i = 0; i < embeddings.length; i++) {
			for (int j = i + 1; j < embeddings.length; j++) {
				double similarity = cosineSimilarity(embeddings[i], embeddings[j]);
				System.out.printf("Similarity between text %d and %d: %.4f\n", i, j, similarity);

				// Different texts should not be identical
				Assert.assertTrue(String.format("Text %d and %d should not be identical", i, j),
					similarity < 0.999);
			}
		}

		System.out.println("✅ Different texts embedding test passed!");
	}

	// Helper method to calculate cosine similarity between two embeddings
	private double cosineSimilarity(float[] vec1, float[] vec2) {
		Assert.assertEquals("Vectors must have same length", vec1.length, vec2.length);

		double dotProduct = 0.0;
		double norm1 = 0.0;
		double norm2 = 0.0;

		for (int i = 0; i < vec1.length; i++) {
			dotProduct += vec1[i] * vec2[i];
			norm1 += vec1[i] * vec1[i];
			norm2 += vec2[i] * vec2[i];
		}

		return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
	}
}
