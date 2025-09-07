package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

public class EmbeddingCompatibilityTest {

	@Test
	public void testEmbeddingFunctionalityWorksWithoutErrors() {
		System.out.println("\n=== Embedding Functionality Compatibility Test ===");

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

			System.out.printf("✅ Embedding functionality works: Generated %d-dimensional embedding\n", embedding.length);
			System.out.printf("First few values: %.6f, %.6f, %.6f, %.6f\n",
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

			System.out.println("✅ Embeddings are deterministic and consistent");

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
				System.out.println("✅ Correctly throws exception when embedding not enabled: " + e.getMessage());
			}

		}

		System.out.println("✅ Embedding functionality compatibility test passed!");
		System.out.println();
		System.out.println("📝 Note: The CodeLlama model used in this test may not be optimized for embeddings.");
		System.out.println("   For best embedding results, use dedicated embedding models like:");
		System.out.println("   - sentence-transformers models");
		System.out.println("   - BGE embedding models");
		System.out.println("   - Nomic embedding models");
		System.out.println("   - E5 embedding models");
	}
}
