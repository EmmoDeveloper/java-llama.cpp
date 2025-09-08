package de.kherud.llama;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.AfterClass;
import java.util.List;

public class RerankingTest {

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
		// Use the new reranking-optimized factory method
		model = LlamaModel.forReranking(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
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
	public void testBasicReranking() {
		System.out.println("\n=== Basic Reranking Test ===");

		String query = "Machine learning is";
		String[] documents = {
			"A machine is a physical system that uses power to apply forces.",
			"Learning is the process of acquiring new understanding and knowledge.",
			"Machine learning is a field of study in artificial intelligence.",
			"Paris is the capital city of France and a major cultural center."
		};

		LlamaOutput output = model.rerank(query, documents);

		Assert.assertNotNull("Reranking output should not be null", output);
		Assert.assertNotNull("Probabilities should not be null", output.probabilities);
		Assert.assertEquals("Should have scores for all documents", 4, output.probabilities.size());
		Assert.assertTrue("Output should be final", output.stop);

		System.out.printf("Query: \"%s\"\n", query);
		System.out.println("Document scores:");

		float maxScore = Float.NEGATIVE_INFINITY;
		String bestMatch = null;

		for (String doc : documents) {
			Float score = output.probabilities.get(doc);
			Assert.assertNotNull(String.format("Score should exist for document: %s", doc), score);
			System.out.printf("  Score %.6f: %s\n", score, doc.substring(0, Math.min(50, doc.length())) + "...");

			if (score > maxScore) {
				maxScore = score;
				bestMatch = doc;
			}
		}

		System.out.printf("Best match: %s\n", bestMatch);

		// Verify that scores are different (not all the same)
		long uniqueScores = output.probabilities.values().stream()
			.mapToDouble(Float::doubleValue)
			.distinct()
			.count();
		Assert.assertTrue("Scores should vary between documents", uniqueScores > 1);

		System.out.println("✅ Basic reranking test passed!");
	}

	@Test
	public void testRerankingConsistency() {
		System.out.println("\n=== Reranking Consistency Test ===");

		String query = "Programming languages";
		String[] documents = {
			"Java is a popular programming language.",
			"Python is widely used for data science.",
			"JavaScript runs in web browsers."
		};

		// Run reranking multiple times
		LlamaOutput output1 = model.rerank(query, documents);
		LlamaOutput output2 = model.rerank(query, documents);

		System.out.println("Comparing scores across multiple runs:");

		for (String doc : documents) {
			Float score1 = output1.probabilities.get(doc);
			Float score2 = output2.probabilities.get(doc);

			System.out.printf("Document: %s\n", doc.substring(0, Math.min(40, doc.length())) + "...");
			System.out.printf("  Run 1: %.6f\n", score1);
			System.out.printf("  Run 2: %.6f\n", score2);
			System.out.printf("  Diff:  %.6f\n", Math.abs(score1 - score2));

			// Scores should be identical for same query-document pairs
			Assert.assertEquals("Reranking should be deterministic", score1, score2, 0.0001f);
		}

		System.out.println("✅ Reranking consistency test passed!");
	}

	@Test
	@Ignore
	public void testRerankingWithoutRerankingMode() {
		System.out.println("\n=== Test Without Reranking Mode ===");

		// Create model without reranking enabled
		// Note: NOT calling .enableReranking()

		try (LlamaModel nonRerankingModel = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
			// Note: NOT calling .enableReranking()
		)) {
			String query = "test query";
			String[] documents = {"document 1", "document 2"};

			// This should throw an exception
			LlamaOutput output = nonRerankingModel.rerank(query, documents);
			Assert.fail("Expected IllegalStateException when reranking is not enabled");
		} catch (IllegalStateException e) {
			System.out.println("✅ Correctly threw exception: " + e.getMessage());
			Assert.assertTrue("Exception message should mention reranking support",
				e.getMessage().contains("reranking support"));
		}

		System.out.println("✅ Non-reranking mode test passed!");
	}

	@Test
	public void testRerankingWithDifferentQueries() {
		System.out.println("\n=== Different Queries Test ===");

		String[] documents = {
			"The quick brown fox jumps over the lazy dog.",
			"Machine learning algorithms can analyze large datasets.",
			"Cooking pasta requires boiling water and adding salt.",
			"Space exploration has led to many technological advances."
		};

		String[] queries = {
			"animal behavior",
			"artificial intelligence",
			"food preparation",
			"space technology"
		};

		System.out.println("Testing how different queries rank the same documents:");

		for (String query : queries) {
			LlamaOutput output = model.rerank(query, documents);

			System.out.printf("\nQuery: \"%s\"\n", query);

			// Find the highest scoring document for this query
			float maxScore = Float.NEGATIVE_INFINITY;
			String bestDoc = null;
			int bestDocIndex = -1;

			for (int j = 0; j < documents.length; j++) {
				String doc = documents[j];
				Float score = output.probabilities.get(doc);
				System.out.printf("  [%d] %.6f: %s\n", j, score, doc.substring(0, Math.min(40, doc.length())) + "...");

				if (score > maxScore) {
					maxScore = score;
					bestDoc = doc;
					bestDocIndex = j;
				}
			}

			System.out.printf("Best match [%d]: %s\n", bestDocIndex, bestDoc);

			// Verify we have scores for all documents
			Assert.assertEquals("Should have scores for all documents", documents.length, output.probabilities.size());
		}

		System.out.println("✅ Different queries test passed!");
	}

	@Test
	public void testJavaApiIntegration() {
		System.out.println("\n=== Java API Integration Test ===");

		String query = "software development";
		String[] documents = {
			"Programming requires logical thinking and problem solving.",
			"Music composition involves creativity and technical skill.",
			"Software engineering is a systematic approach to development.",
			"Cooking is both an art and a science."
		};

		// Test the helper method that sorts results
		List<Pair<String, Float>> rankedResults = model.rerank(true, query, documents);

		Assert.assertNotNull("Ranked results should not be null", rankedResults);
		Assert.assertEquals("Should have results for all documents", documents.length, rankedResults.size());

		System.out.printf("Query: \"%s\"\n", query);
		System.out.println("Ranked results (highest to lowest score):");

		float previousScore = Float.MAX_VALUE;
		for (int i = 0; i < rankedResults.size(); i++) {
			Pair<String, Float> result = rankedResults.get(i);
			String doc = result.key();
			Float score = result.value();

			System.out.printf("%d. Score %.6f: %s\n", i + 1, score,
				doc.substring(0, Math.min(50, doc.length())) + "...");

			// Verify sorting is correct (scores should be in descending order)
			Assert.assertTrue(String.format("Results should be sorted in descending order (position %d)", i),
				score <= previousScore);
			previousScore = score;
		}

		// Test without sorting
		List<Pair<String, Float>> unsortedResults = model.rerank(false, query, documents);
		Assert.assertEquals("Should have same number of results", rankedResults.size(), unsortedResults.size());

		System.out.println("✅ Java API integration test passed!");
	}
}
