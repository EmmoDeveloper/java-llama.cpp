package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;
import java.util.Map;

public class ModelInformationTest {

	private LlamaModel createModel() {
		// Use the same model configuration as other tests
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);  // Use GPU acceleration
		return new LlamaModel(params);
	}

	@Test
	public void testModelParameterCount() throws Exception {
		try (LlamaModel model = createModel()) {
			long paramCount = model.getModelParameterCount();

			// Should be a positive number for any real model
			Assert.assertTrue("Parameter count should be positive", paramCount > 0);

			// CodeLlama 7B should have billions of parameters (around 7 billion)
			Assert.assertTrue("CodeLlama 7B should have at least 6 billion parameters", paramCount > 6_000_000_000L);
			Assert.assertTrue("CodeLlama 7B should have less than 10 billion parameters", paramCount < 10_000_000_000L);

			System.out.println("Model parameter count: " + String.format("%,d", paramCount));
		}
	}

	@Test
	public void testModelSize() throws Exception {
		try (LlamaModel model = createModel()) {
			long modelSize = model.getModelSize();

			// Should be a positive number for any real model
			Assert.assertTrue("Model size should be positive", modelSize > 0);

			// Q2_K quantized 7B model should be several GB but less than original
			Assert.assertTrue("Q2_K model should be at least 2GB", modelSize > 2_000_000_000L);
			Assert.assertTrue("Q2_K model should be less than 20GB", modelSize < 20_000_000_000L);

			System.out.println("Model size: " + String.format("%.2f GB", modelSize / 1_000_000_000.0));
		}
	}

	@Test
	public void testModelMetadata() throws Exception {
		try (LlamaModel model = createModel()) {
			// Test metadata count
			int metadataCount = model.getModelMetadataCount();
			Assert.assertTrue("Metadata count should be non-negative", metadataCount >= 0);
			System.out.println("Metadata entries: " + metadataCount);

			if (metadataCount > 0) {
				// Test individual metadata access
				String firstKey = model.getModelMetadataKey(0);
				String firstValue = model.getModelMetadataValueByIndex(0);

				Assert.assertNotNull("First metadata key should not be null", firstKey);
				Assert.assertNotNull("First metadata value should not be null", firstValue);

				System.out.println("First metadata: " + firstKey + " = " + firstValue);

				// Test metadata by key lookup
				if (!firstKey.isEmpty()) {
					String valueByKey = model.getModelMetadataValue(firstKey);
					Assert.assertEquals("Values should match when accessed by key vs index", firstValue, valueByKey);
				}
			}

			// Test convenience method for getting all metadata
			Map<String, String> allMetadata = model.getModelMetadata();
			Assert.assertNotNull("Metadata map should not be null", allMetadata);
			Assert.assertEquals("Metadata map size should match count", metadataCount, allMetadata.size());

			// Print some interesting metadata
			System.out.println("Model metadata:");
			allMetadata.entrySet().stream()
				.limit(10)  // Print first 10 entries
				.forEach(entry -> System.out.println("  " + entry.getKey() + " = " + entry.getValue()));
		}
	}

	@Test
	public void testVocabularyInfo() throws Exception {
		try (LlamaModel model = createModel()) {
			// Test vocabulary type
			int vocabType = model.getVocabularyType();
			Assert.assertTrue("Vocabulary type should be non-negative", vocabType >= 0);
			System.out.println("Vocabulary type: " + vocabType);

			// Test vocabulary size
			int vocabSize = model.getVocabularySize();
			Assert.assertTrue("Vocabulary size should be positive", vocabSize > 0);

			// CodeLlama models typically have vocab sizes around 32000
			Assert.assertTrue("Vocabulary should have at least 1000 tokens", vocabSize >= 1000);
			Assert.assertTrue("Vocabulary should have less than 100000 tokens", vocabSize < 100000);

			System.out.println("Vocabulary size: " + String.format("%,d", vocabSize));
		}
	}

	@Test
	public void testSpecialTokens() throws Exception {
		try (LlamaModel model = createModel()) {
			// Test special token retrieval
			int bosToken = model.getBosToken();
			int eosToken = model.getEosToken();
			int eotToken = model.getEotToken();
			int sepToken = model.getSepToken();
			int nlToken = model.getNlToken();
			int padToken = model.getPadToken();

			System.out.println("Special tokens:");
			System.out.println("  BOS (Beginning of Sequence): " + bosToken);
			System.out.println("  EOS (End of Sequence): " + eosToken);
			System.out.println("  EOT (End of Turn): " + eotToken);
			System.out.println("  SEP (Separator): " + sepToken);
			System.out.println("  NL (Newline): " + nlToken);
			System.out.println("  PAD (Padding): " + padToken);

			// Most models should have at least BOS and EOS tokens
			// (Some may return -1 if not available, which is valid)
			if (bosToken != -1) {
				Assert.assertTrue("BOS token should be non-negative", bosToken >= 0);
			}
			if (eosToken != -1) {
				Assert.assertTrue("EOS token should be non-negative", eosToken >= 0);
			}

			// If BOS and EOS are available, they should be different
			if (bosToken != -1 && eosToken != -1) {
				Assert.assertNotEquals("BOS and EOS tokens should be different", bosToken, eosToken);
			}
		}
	}

	@Test
	public void testTokenInformation() throws Exception {
		try (LlamaModel model = createModel()) {
			int vocabSize = model.getVocabularySize();

			// Test token information for first few tokens
			for (int i = 0; i < Math.min(10, vocabSize); i++) {
				String tokenText = model.getTokenText(i);
				float tokenScore = model.getTokenScore(i);
				int tokenAttrs = model.getTokenAttributes(i);

				Assert.assertNotNull("Token text should not be null", tokenText);
				// Token text can be empty for some special tokens

				// Token scores can be any float value including negative
				System.out.println("Token " + i + ": '" + tokenText + "' (score: " + tokenScore + ", attrs: " + tokenAttrs + ")");
			}
		}
	}

	@Test
	public void testTokenChecking() throws Exception {
		try (LlamaModel model = createModel()) {
			int bosToken = model.getBosToken();
			int eosToken = model.getEosToken();

			// Test EOG (End of Generation) checking
			if (eosToken != -1) {
				boolean isEogEos = model.isEogToken(eosToken);
				System.out.println("EOS token (" + eosToken + ") is EOG: " + isEogEos);
				// EOS tokens are typically EOG tokens, but not always
			}

			// Test control token checking
			if (bosToken != -1) {
				boolean isControlBos = model.isControlToken(bosToken);
				System.out.println("BOS token (" + bosToken + ") is control: " + isControlBos);
				// BOS tokens are typically control tokens
			}

			// Test with first few regular tokens
			int vocabSize = model.getVocabularySize();
			for (int i = 0; i < Math.min(5, vocabSize); i++) {
				boolean isEog = model.isEogToken(i);
				boolean isControl = model.isControlToken(i);
				String tokenText = model.getTokenText(i);

				System.out.println("Token " + i + " ('" + tokenText + "'): EOG=" + isEog + ", Control=" + isControl);
			}
		}
	}

	@Test
	public void testInvalidInputs() throws Exception {
		try (LlamaModel model = createModel()) {
			// Test negative metadata index
			try {
				model.getModelMetadataKey(-1);
				Assert.fail("Should throw IllegalArgumentException for negative index");
			} catch (IllegalArgumentException e) {
				System.out.println("Correctly caught negative metadata index");
			}

			// Test null metadata key
			try {
				model.getModelMetadataValue(null);
				Assert.fail("Should throw IllegalArgumentException for null key");
			} catch (IllegalArgumentException e) {
				System.out.println("Correctly caught null metadata key");
			}

			// Test negative token ID
			try {
				model.getTokenText(-1);
				Assert.fail("Should throw IllegalArgumentException for negative token ID");
			} catch (IllegalArgumentException e) {
				System.out.println("Correctly caught negative token ID");
			}

			// Test invalid token ID (beyond vocabulary)
			int vocabSize = model.getVocabularySize();
			try {
				model.getTokenText(vocabSize + 1000);
				Assert.fail("Should throw exception for token ID beyond vocabulary");
			} catch (IllegalArgumentException e) {
				System.out.println("Correctly caught invalid token ID: " + e.getMessage());
			}
		}
	}

	@Test
	public void testModelIntrospection() throws Exception {
		try (LlamaModel model = createModel()) {
			// Comprehensive model information display
			System.out.println("\n=== MODEL INTROSPECTION ===");

			// Basic model info
			long paramCount = model.getModelParameterCount();
			long modelSize = model.getModelSize();
			int vocabSize = model.getVocabularySize();
			int vocabType = model.getVocabularyType();

			System.out.println("Parameters: " + String.format("%,d", paramCount));
			System.out.println("Model size: " + String.format("%.2f GB", modelSize / 1_000_000_000.0));
			System.out.println("Vocabulary size: " + String.format("%,d", vocabSize));
			System.out.println("Vocabulary type: " + vocabType);

			// Special tokens
			System.out.println("\nSpecial tokens:");
			System.out.println("  BOS: " + model.getBosToken());
			System.out.println("  EOS: " + model.getEosToken());
			System.out.println("  EOT: " + model.getEotToken());
			System.out.println("  SEP: " + model.getSepToken());
			System.out.println("  NL: " + model.getNlToken());
			System.out.println("  PAD: " + model.getPadToken());

			// Model metadata highlights
			Map<String, String> metadata = model.getModelMetadata();
			System.out.println("\nModel metadata (" + metadata.size() + " entries):");
			metadata.entrySet().stream()
				.filter(entry -> {
					String key = entry.getKey().toLowerCase();
					return key.contains("name") || key.contains("version") ||
						   key.contains("arch") || key.contains("type") ||
						   key.contains("author") || key.contains("license");
				})
				.limit(5)
				.forEach(entry -> System.out.println("  " + entry.getKey() + " = " + entry.getValue()));
		}
	}

	@Test
	public void testTokenSampling() throws Exception {
		try (LlamaModel model = createModel()) {
			// Test that we can get information about tokens used in a simple string
			String testString = "Hello world";
			int[] tokens = model.encode(testString);

			System.out.println("Tokenization of '" + testString + "':");
			for (int i = 0; i < tokens.length && i < 10; i++) {  // Limit to first 10 tokens
				int tokenId = tokens[i];
				String tokenText = model.getTokenText(tokenId);
				float tokenScore = model.getTokenScore(tokenId);
				boolean isControl = model.isControlToken(tokenId);
				boolean isEog = model.isEogToken(tokenId);

				System.out.println("  Token " + tokenId + ": '" + tokenText + "' " +
					"(score: " + String.format("%.3f", tokenScore) +
					", control: " + isControl + ", eog: " + isEog + ")");
			}

			// Verify we can decode back
			String decoded = model.decode(tokens);
			Assert.assertTrue("Decoded string should contain original", decoded.contains("Hello"));
			Assert.assertTrue("Decoded string should contain original", decoded.contains("world"));
		}
	}

	@Test
	public void testLargeTokenIds() throws Exception {
		try (LlamaModel model = createModel()) {
			int vocabSize = model.getVocabularySize();

			// Test tokens near the end of the vocabulary
			if (vocabSize > 100) {
				int[] testTokens = {
					vocabSize - 1,    // Last token
					vocabSize - 10,   // Near end
					vocabSize / 2,    // Middle
					Math.min(1000, vocabSize - 1)  // Somewhere in middle range
				};

				for (int tokenId : testTokens) {
					if (tokenId >= 0 && tokenId < vocabSize) {
						try {
							String text = model.getTokenText(tokenId);
							float score = model.getTokenScore(tokenId);
							boolean isControl = model.isControlToken(tokenId);

							System.out.println("Token " + tokenId + ": '" + text + "' " +
								"(score: " + String.format("%.3f", score) + ", control: " + isControl + ")");
						} catch (Exception e) {
							System.out.println("Error accessing token " + tokenId + ": " + e.getMessage());
						}
					}
				}
			}
		}
	}
}
