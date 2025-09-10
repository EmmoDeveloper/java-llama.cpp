package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.Assert;

public class EmbeddingCompatibilityTest {
	private static final System.Logger logger = System.getLogger(EmbeddingCompatibilityTest.class.getName());

	@Ignore
	public void testEmbeddingFunctionalityWorksWithoutErrors() {
		logger.log(DEBUG, "\n=== Embedding Functionality Compatibility Test ===");

		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");

		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
				.enableEmbedding()
		)) {
			// Test that embedding function works without throwing exceptions
			float[] embedding = model.embed("Hello world");

			Assert.assertNotNull("Embedding should not be null", embedding);
			Assert.assertEquals("Should have 4096 dimensions", 4096, embedding.length);

			logger.log(DEBUG, "✅ Embedding functionality works: Generated %d-dimensional embedding", embedding.length);
			logger.log(DEBUG, "First few values: %.6f, %.6f, %.6f, %.6f",
				embedding[0], embedding[1], embedding[2], embedding[3]);

			// Test that embeddings are deterministic (same input = same output)
			float[] embedding2 = model.embed("Hello world");
			Assert.assertEquals("Embeddings should have same length", embedding.length, embedding2.length);

			boolean identical = true;
			for (int i = 0; i < embedding.length; i++) {
				if (embedding[i] != embedding2[i]) {
					identical = false;
					break;
				}
			}
			Assert.assertTrue("Same input should produce identical embeddings", identical);

			logger.log(DEBUG, "✅ Embeddings are deterministic and consistent");

			// Test error handling for non-embedding model
			// Note: NOT calling .enableEmbedding()

			try (LlamaModel nonEmbeddingModel = new LlamaModel(
				new ModelParameters()
					.setCtxSize(512)
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(43)
				// Note: NOT calling .enableEmbedding()
			)) {
				nonEmbeddingModel.embed("Hello world");
				Assert.fail("Expected IllegalStateException when embedding is not enabled");
			} catch (IllegalStateException e) {
				logger.log(DEBUG, "✅ Correctly throws exception when embedding not enabled: " + e.getMessage());
			}

		}

		logger.log(DEBUG, "✅ Embedding functionality compatibility test passed!");
		logger.log(DEBUG, "📝 Note: The CodeLlama model used in this test may not be optimized for embeddings.");
		logger.log(DEBUG, "   For best embedding results, use dedicated embedding models like:");
		logger.log(DEBUG, "   - sentence-transformers models");
		logger.log(DEBUG, "   - BGE embedding models");
		logger.log(DEBUG, "   - Nomic embedding models");
		logger.log(DEBUG, "   - E5 embedding models");
	}
}
