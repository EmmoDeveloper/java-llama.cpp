package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.AfterClass;
import org.junit.Assume;
import java.io.File;

public class LoRAAdapterIntegrationTest {
	private static final System.Logger logger = System.getLogger(LoRAAdapterIntegrationTest.class.getName());

	private static LlamaModel model;
	private static final String LORA_TEST_PATH = "/work/java/java-llama.cpp/models/LoRA-Llama-3.1-8B-MultiReflection-f16.gguf";
	private static final String LORA_TEST_PATH_ALT = "/work/java/java-llama.cpp/models/lora-test.gguf";

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");

		// Try to find an available model (Llama 3.1 models are preferred for LoRA compatibility)
		String[] modelPaths = {
			"/work/java/java-llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",  // Matches our LoRA adapter
			"/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf",
			"/work/java/fork/java-llama.cpp/models/codellama-7b.Q2_K.gguf",
			"/work/java/ai-ide-jvm/jvm-ai-ide/src/main/resources/models/Qwen3-4B-Instruct-2507-F16.gguf"
		};

		String availableModel = null;
		for (String path : modelPaths) {
			if (new File(path).exists()) {
				availableModel = path;
				break;
			}
		}

		Assume.assumeNotNull("No test model found", availableModel);

		model = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel(availableModel)
				.setGpuLayers(10) // Reduced for compatibility
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testLoRAAdapterRealFileIntegration() {
		logger.log(DEBUG, "\n=== LoRA Adapter Real File Integration Test ===");

		// Check if we have a real LoRA file to test with (must be >1KB to be valid)
		String testLoRAPath = null;
		if (new File(LORA_TEST_PATH).exists() && new File(LORA_TEST_PATH).length() > 1024) {
			testLoRAPath = LORA_TEST_PATH;
		} else if (new File(LORA_TEST_PATH_ALT).exists() && new File(LORA_TEST_PATH_ALT).length() > 1024) {
			testLoRAPath = LORA_TEST_PATH_ALT;
		}

		if (testLoRAPath == null) {
			logger.log(DEBUG, "⚠️  No valid LoRA test file found - testing error handling only");

			// Test that our error handling works correctly for nonexistent file
			try {
				model.loadLoRAAdapter("/nonexistent/lora/file.safetensors");
				Assert.fail("Should throw exception for nonexistent file");
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ Correctly handled nonexistent file: " + e.getMessage());
			}

			// Test handling of invalid/dummy file if it exists
			if (new File(LORA_TEST_PATH_ALT).exists()) {
				try {
					model.loadLoRAAdapter(LORA_TEST_PATH_ALT);
					Assert.fail("Should throw exception for invalid LoRA file");
				} catch (LlamaException e) {
					logger.log(DEBUG, "✅ Correctly rejected invalid/dummy LoRA file: " + e.getMessage());
				}
			}

			logger.log(DEBUG, "✅ Error handling test passed (no valid LoRA file available)");
			return;
		}

		// Test with real LoRA file
		logger.log(DEBUG, "📁 Testing with LoRA file: " + testLoRAPath);

		try {
			// Load LoRA adapter
			long adapterHandle = model.loadLoRAAdapter(testLoRAPath);
			logger.log(DEBUG, "✅ Successfully loaded LoRA adapter, handle: " + adapterHandle);
			Assert.assertTrue("Adapter handle should be valid", adapterHandle != -1 && adapterHandle != 0);

			// Test setting adapter with different scales
			int result1 = model.setLoRAAdapter(adapterHandle, 1.0f);
			logger.log(DEBUG, "✅ Set LoRA adapter with scale 1.0: result = " + result1);

			int result2 = model.setLoRAAdapter(adapterHandle, 0.5f);
			logger.log(DEBUG, "✅ Set LoRA adapter with scale 0.5: result = " + result2);

			// Test default scale
			int result3 = model.setLoRAAdapter(adapterHandle);
			logger.log(DEBUG, "✅ Set LoRA adapter with default scale: result = " + result3);

			// Test removing adapter
			int removeResult = model.removeLoRAAdapter(adapterHandle);
			logger.log(DEBUG, "✅ Removed LoRA adapter: result = " + removeResult);

			// Test metadata access
			try {
				int metaCount = model.getLoRAAdapterMetadataCount(adapterHandle);
				logger.log(DEBUG, "📊 LoRA adapter metadata count: " + metaCount);

				if (metaCount > 0) {
					for (int i = 0; i < Math.min(metaCount, 5); i++) {
						String key = model.getLoRAAdapterMetadataKey(adapterHandle, i);
						String value = model.getLoRAAdapterMetadataValue(adapterHandle, i);
						logger.log(DEBUG, "📋 Metadata[%d]: %s = %s", i, key, value);
					}
				}
			} catch (LlamaException e) {
				logger.log(DEBUG, "ℹ️  Metadata access failed (expected after removal): " + e.getMessage());
			}

			// Test ALORA functionality
			try {
				long tokenCount = model.getAloraInvocationTokenCount(adapterHandle);
				logger.log(DEBUG, "🔢 ALORA token count: " + tokenCount);

				if (tokenCount > 0) {
					int[] tokens = model.getAloraInvocationTokens(adapterHandle);
					logger.log(DEBUG, "🎯 ALORA tokens (%d): [", tokens.length);
					for (int i = 0; i < Math.min(tokens.length, 10); i++) {
						logger.log(DEBUG, tokens[i]);
						if (i < Math.min(tokens.length, 10) - 1) logger.log(DEBUG, ", ");
					}
					if (tokens.length > 10) logger.log(DEBUG, "...");
					logger.log(DEBUG, "]");
				} else {
					logger.log(DEBUG, "ℹ️  Not an ALORA adapter (token count: 0)");
				}
			} catch (LlamaException e) {
				logger.log(DEBUG, "ℹ️  ALORA access failed (expected after removal): " + e.getMessage());
			}

			// Clean up
			model.freeLoRAAdapter(adapterHandle);
			logger.log(DEBUG, "🧹 Freed LoRA adapter");

		} catch (LlamaException e) {
			Assert.fail("Real LoRA integration test failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Real LoRA integration test passed!");
	}

	@Test
	public void testControlVectorIntegration() {
		logger.log(DEBUG, "\n=== Control Vector Integration Test ===");

		try {
			// Test control vector operations
			logger.log(DEBUG, "🧹 Clearing any existing control vectors");
			int clearResult = model.clearControlVector();
			logger.log(DEBUG, "✅ Initial clear result: " + clearResult);

			// Test with different sized control vectors
			float[][] testVectors = {
				{},  // Empty vector
				{0.1f, 0.2f, 0.3f, 0.4f},  // Small vector
				new float[1024],  // Medium vector
				new float[4096]   // Large vector (typical embedding size)
			};

			// Fill the larger vectors with test data
			for (int i = 0; i < testVectors[2].length; i++) {
				testVectors[2][i] = (float) Math.sin(i * 0.1) * 0.1f;
			}
			for (int i = 0; i < testVectors[3].length; i++) {
				testVectors[3][i] = (float) Math.cos(i * 0.05) * 0.2f;
			}

			for (float[] vector : testVectors) {
				logger.log(DEBUG, "🧪 Testing control vector with %d elements", vector.length);

				try {
					int result = model.applyControlVector(vector);
					logger.log(DEBUG, "✅ Applied control vector[%d]: result = %d", vector.length, result);
				} catch (LlamaException e) {
					logger.log(DEBUG, "ℹ️  Control vector[%d] failed (may be expected): %s", vector.length, e.getMessage());
				}
			}

			// Test null control vector (clear)
			int nullResult = model.applyControlVector(null);
			logger.log(DEBUG, "✅ Applied null control vector (clear): result = " + nullResult);

			// Final cleanup
			model.clearControlVector();
			logger.log(DEBUG, "🧹 Final control vector cleanup completed");

		} catch (Exception e) {
			Assert.fail("Control vector integration test failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Control vector integration test passed!");
	}

	@Test
	public void testLoRAWorkflowWithClearOperations() {
		logger.log(DEBUG, "\n=== LoRA Workflow with Clear Operations Test ===");

		try {
			// Test clear operations work correctly
			logger.log(DEBUG, "🧹 Step 1: Clear all adapters and control vectors");
			model.clearLoRAAdapters();
			model.clearControlVector();

			// Test multiple clear operations (should be safe)
			logger.log(DEBUG, "🧹 Step 2: Multiple clear operations");
			for (int i = 0; i < 3; i++) {
				model.clearLoRAAdapters();
				model.clearControlVector();
			}
			logger.log(DEBUG, "✅ Multiple clears completed without error");

			// Test that we can perform operations after clearing
			logger.log(DEBUG, "🔧 Step 3: Test operations after clearing");

			// Apply and clear control vector
			float[] testVector = {0.1f, 0.2f, 0.3f, 0.4f};
			try {
				int applyResult = model.applyControlVector(testVector);
				logger.log(DEBUG, "✅ Applied test control vector after clear: " + applyResult);

				int clearResult = model.clearControlVector();
				logger.log(DEBUG, "✅ Cleared control vector: " + clearResult);
			} catch (LlamaException e) {
				logger.log(DEBUG, "ℹ️  Control vector operations failed (may be model-dependent): " + e.getMessage());
			}

			// Final state verification
			logger.log(DEBUG, "🧹 Step 4: Final cleanup and verification");
			model.clearLoRAAdapters();
			model.clearControlVector();

			// Test that invalid operations still fail appropriately
			try {
				model.setLoRAAdapter(-1L, 1.0f);
				Assert.fail("Should fail with invalid adapter handle");
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ Invalid operations still fail correctly: " + e.getMessage());
			}

		} catch (Exception e) {
			Assert.fail("LoRA workflow test failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA workflow test passed!");
	}

	@Test
	public void testErrorHandlingEdgeCases() {
		logger.log(DEBUG, "\n=== Error Handling Edge Cases Test ===");

		// Test various invalid inputs
		String[] invalidPaths = {
			"",
			"   ",
			"/dev/null",
			"/invalid/path/file.txt",
			"not-a-real-file.safetensors",
			null
		};

		for (String path : invalidPaths) {
			try {
				if (path == null) {
					logger.log(DEBUG, "🧪 Testing null path");
					model.loadLoRAAdapter(null);
				} else {
					logger.log(DEBUG, "🧪 Testing invalid path: '%s'", path);
					model.loadLoRAAdapter(path);
				}
				Assert.fail("Should have thrown exception for invalid path: " + path);
			} catch (IllegalArgumentException | LlamaException e) {
				logger.log(DEBUG, "✅ Correctly handled invalid path '%s': %s",
					path == null ? "null" : path, e.getClass().getSimpleName());
			}
		}

		// Test invalid handles
		long[] invalidHandles = {-1L, 0L, Long.MAX_VALUE, Long.MIN_VALUE};

		for (long handle : invalidHandles) {
			logger.log(DEBUG, "🧪 Testing invalid handle: %d", handle);

			try {
				model.setLoRAAdapter(handle, 1.0f);
				Assert.fail("Should have thrown exception for invalid handle: " + handle);
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ setLoRAAdapter correctly handled invalid handle %d", handle);
			}

			try {
				model.getLoRAAdapterMetadataCount(handle);
				Assert.fail("Should have thrown exception for invalid handle: " + handle);
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ getLoRAAdapterMetadataCount correctly handled invalid handle %d", handle);
			}
		}

		logger.log(DEBUG, "✅ Error handling edge cases test passed!");
	}
}
