package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.AfterClass;

public class EmbeddingTest {
	private static final System.Logger logger = System.getLogger(EmbeddingTest.class.getName());

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
		// Use the new embedding-optimized factory method
		model = LlamaModel.forEmbedding(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
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
		logger.log(DEBUG, "\n=== Basic Embedding Test ===");

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

		logger.log(DEBUG, "Generated embedding with %d dimensions", embedding.length);
		logger.log(DEBUG, "First few values: %.6f, %.6f, %.6f, %.6f",
			embedding[0], embedding[1], embedding[2], embedding[3]);

		logger.log(DEBUG, "✅ Basic embedding test passed!");
	}

	@Test
	public void testEmbeddingSimilarity() {
		logger.log(DEBUG, "\n=== Embedding Similarity Test ===");

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

		logger.log(DEBUG, "Similarity (same text): %.6f", similarity_same);
		logger.log(DEBUG, "Similarity (different text): %.6f", similarity_different);

		// Identical texts should have very high similarity (close to 1.0)
		Assert.assertTrue("Same text should have high similarity", similarity_same > 0.95);

		// Different texts should have lower similarity
		Assert.assertTrue("Different texts should have lower similarity",
			similarity_different < similarity_same);

		logger.log(DEBUG, "✅ Embedding similarity test passed!");
	}

	@Test
	public void testEmbeddingConsistency() {
		logger.log(DEBUG, "\n=== Embedding Consistency Test ===");

		// Test that same input produces consistent embeddings
		String testText = "The quick brown fox jumps over the lazy dog";

		float[] embedding1 = model.embed(testText);
		float[] embedding2 = model.embed(testText);
		float[] embedding3 = model.embed(testText);

		double similarity_1_2 = cosineSimilarity(embedding1, embedding2);
		double similarity_2_3 = cosineSimilarity(embedding2, embedding3);
		double similarity_1_3 = cosineSimilarity(embedding1, embedding3);

		logger.log(DEBUG, "Consistency similarity 1-2: %.6f", similarity_1_2);
		logger.log(DEBUG, "Consistency similarity 2-3: %.6f", similarity_2_3);
		logger.log(DEBUG, "Consistency similarity 1-3: %.6f", similarity_1_3);

		// All should be identical (or very close due to floating point precision)
		Assert.assertTrue("Consistent embeddings should be identical", similarity_1_2 > 0.999);
		Assert.assertTrue("Consistent embeddings should be identical", similarity_2_3 > 0.999);
		Assert.assertTrue("Consistent embeddings should be identical", similarity_1_3 > 0.999);

		logger.log(DEBUG, "✅ Embedding consistency test passed!");
	}

//	@Test
	@Ignore
	public void testEmbeddingWithoutEmbeddingMode() {
		logger.log(DEBUG, "\n=== Test Without Embedding Mode ===");

		// Create model without embedding enabled
		// Note: NOT calling .enableEmbedding()

		try (LlamaModel nonEmbeddingModel = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
			// Note: NOT calling .enableEmbedding()
		)) {
			// This should throw an exception
			float[] embedding = nonEmbeddingModel.embed("Hello world");
			Assert.fail("Expected IllegalStateException when embedding is not enabled");
		} catch (IllegalStateException e) {
			logger.log(DEBUG, "✅ Correctly threw exception: " + e.getMessage());
			Assert.assertTrue("Exception message should mention embedding support",
				e.getMessage().contains("embedding support"));
		}

		logger.log(DEBUG, "✅ Non-embedding mode test passed!");
	}

	@Test
	public void testEmbeddingDifferentTexts() {
		logger.log(DEBUG, "\n=== Different Texts Embedding Test ===");

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
			logger.log(DEBUG, "Text %d: \"%s\" -> embedding[0-3]: %.4f, %.4f, %.4f, %.4f",
				i, testTexts[i],
				embeddings[i][0], embeddings[i][1],
				embeddings[i][2], embeddings[i][3]);
		}

		// Verify all embeddings are different
		for (int i = 0; i < embeddings.length; i++) {
			for (int j = i + 1; j < embeddings.length; j++) {
				double similarity = cosineSimilarity(embeddings[i], embeddings[j]);
				logger.log(DEBUG, "Similarity between text %d and %d: %.4f", i, j, similarity);

				// Different texts should not be identical
				Assert.assertTrue(String.format("Text %d and %d should not be identical", i, j),
					similarity < 0.999);
			}
		}

		logger.log(DEBUG, "✅ Different texts embedding test passed!");
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
