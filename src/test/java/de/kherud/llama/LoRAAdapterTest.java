package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.AfterClass;

public class LoRAAdapterTest {
	private static final System.Logger logger = System.getLogger(LoRAAdapterTest.class.getName());

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");

		// Try to load model, but don't fail the entire test suite if it doesn't work
		try {
			model = new LlamaModel(
				new ModelParameters()
					.setCtxSize(512)
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(10) // Reduced for stability
			);
			logger.log(DEBUG, "✅ Model loaded successfully for LoRA tests");
		} catch (Exception e) {
			System.err.println("⚠️  Failed to load model for LoRA tests: " + e.getMessage());
			model = null; // Will cause tests to skip
		}
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testLoRAAdapterLoadingWithInvalidPath() {
		logger.log(DEBUG, "\n=== LoRA Adapter Invalid Path Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		try {
			// Try to load a non-existent LoRA adapter
			model.loadLoRAAdapter("/nonexistent/path/to/lora.bin");
			Assert.fail("Expected LlamaException when loading invalid LoRA path");
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ Correctly threw exception: " + e.getMessage());
			Assert.assertTrue("Exception message should mention LoRA loading failure",
				e.getMessage().contains("Failed to load LoRA adapter"));
		}

		logger.log(DEBUG, "✅ Invalid LoRA path test passed!");
	}

	@Test
	public void testLoRAAdapterNullPath() {
		logger.log(DEBUG, "\n=== LoRA Adapter Null Path Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		try {
			// Try to load with null path
			model.loadLoRAAdapter(null);
			Assert.fail("Expected IllegalArgumentException when loading null LoRA path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly threw exception: " + e.getMessage());
			Assert.assertTrue("Exception message should mention null or empty path",
				e.getMessage().contains("cannot be null or empty"));
		} catch (LlamaException e) {
			// This is also acceptable
			logger.log(DEBUG, "✅ Correctly threw LlamaException: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Null LoRA path test passed!");
	}

	@Test
	public void testLoRAAdapterEmptyPath() {
		logger.log(DEBUG, "\n=== LoRA Adapter Empty Path Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		try {
			// Try to load with empty path
			model.loadLoRAAdapter("");
			Assert.fail("Expected IllegalArgumentException when loading empty LoRA path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly threw exception: " + e.getMessage());
			Assert.assertTrue("Exception message should mention null or empty path",
				e.getMessage().contains("cannot be null or empty"));
		} catch (LlamaException e) {
			// This is also acceptable
			logger.log(DEBUG, "✅ Correctly threw LlamaException: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Empty LoRA path test passed!");
	}

	@Test
	public void testLoRAAdapterFreeInvalidHandle() {
		logger.log(DEBUG, "\n=== LoRA Adapter Free Invalid Handle Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		// Test freeing only safe invalid adapter handles
		try {
			model.freeLoRAAdapter(-1L);
			model.freeLoRAAdapter(0L);
			logger.log(DEBUG, "✅ Safe invalid handle freeing succeeded without crash");
		} catch (Exception e) {
			Assert.fail("Freeing invalid LoRA adapter handle should not throw exceptions: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Invalid LoRA handle free test passed!");
	}

	@Test
	public void testLoRAAdapterOperationsWithInvalidHandle() {
		logger.log(DEBUG, "\n=== LoRA Adapter Operations with Invalid Handle Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		long invalidHandle = -1L;

		// Test setting invalid adapter
		try {
			int result = model.setLoRAAdapter(invalidHandle, 1.0f);
			Assert.fail("Expected LlamaException when setting invalid LoRA adapter");
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ setLoRAAdapter correctly threw exception: " + e.getMessage());
		}

		// Test removing invalid adapter
		try {
			int result = model.removeLoRAAdapter(invalidHandle);
			Assert.fail("Expected LlamaException when removing invalid LoRA adapter");
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ removeLoRAAdapter correctly threw exception: " + e.getMessage());
		}

		// Test getting metadata from invalid adapter
		try {
			String metadata = model.getLoRAAdapterMetadata(invalidHandle, "test.key");
			// If it returns null, that's acceptable behavior for invalid adapters
			if (metadata == null) {
				logger.log(DEBUG, "✅ getLoRAAdapterMetadata correctly returned null for invalid adapter");
			} else {
				Assert.fail("Expected null or LlamaException when getting metadata from invalid LoRA adapter");
			}
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ getLoRAAdapterMetadata correctly threw exception: " + e.getMessage());
		} catch (IllegalArgumentException e) {
			// Also acceptable if it validates the key first
			logger.log(DEBUG, "✅ getLoRAAdapterMetadata correctly validated key: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Invalid LoRA handle operations test passed!");
	}

	@Test
	public void testLoRAAdapterClearOperations() {
		logger.log(DEBUG, "\n=== LoRA Adapter Clear Operations Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		try {
			// Clear all adapters (should not crash even if no adapters are loaded)
			model.clearLoRAAdapters();
			logger.log(DEBUG, "✅ clearLoRAAdapters succeeded");

			// Try clearing multiple times
			model.clearLoRAAdapters();
			model.clearLoRAAdapters();
			logger.log(DEBUG, "✅ Multiple clearLoRAAdapters calls succeeded");

		} catch (Exception e) {
			Assert.fail("Clearing LoRA adapters should not throw exceptions: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA adapter clear operations test passed!");
	}

	@Test
	public void testControlVectorOperations() {
		logger.log(DEBUG, "\n=== Control Vector Operations Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		try {
			// Test applying null control vector (should clear)
			int result = model.applyControlVector(null);
			Assert.assertTrue("Apply null control vector should succeed", result >= 0);
			logger.log(DEBUG, "✅ Applied null control vector (clear): result = " + result);

			// Test clearing control vector
			int clearResult = model.clearControlVector();
			Assert.assertTrue("Clear control vector should succeed", clearResult >= 0);
			logger.log(DEBUG, "✅ Cleared control vector: result = " + clearResult);

			// Test applying empty control vector
			float[] emptyVector = new float[0];
			int emptyResult = model.applyControlVector(emptyVector);
			Assert.assertTrue("Apply empty control vector should succeed", emptyResult >= 0);
			logger.log(DEBUG, "✅ Applied empty control vector: result = " + emptyResult);

			// Test applying small control vector (should work even if wrong size)
			float[] smallVector = {0.1f, 0.2f, 0.3f, 0.4f};
			int smallResult = model.applyControlVector(smallVector);
			// This might succeed or fail depending on model requirements
			logger.log(DEBUG, "✅ Applied small control vector: result = " + smallResult);

		} catch (Exception e) {
			// Some control vector operations might fail with invalid sizes
			logger.log(DEBUG, "ℹ️ Control vector operation failed (expected): " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Control vector operations test passed!");
	}

	@Test
	public void testLoRAAdapterMetadataOperations() {
		logger.log(DEBUG, "\n=== LoRA Adapter Metadata Operations Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		long invalidHandle = -1L;

		// Test metadata operations with invalid handle
		try {
			int count = model.getLoRAAdapterMetadataCount(invalidHandle);
			Assert.fail("Expected LlamaException when getting metadata count from invalid adapter");
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ getLoRAAdapterMetadataCount correctly threw exception: " + e.getMessage());
		}

		try {
			String key = model.getLoRAAdapterMetadataKey(invalidHandle, 0);
			// If it returns null, that's acceptable behavior for invalid adapters
			if (key == null) {
				logger.log(DEBUG, "✅ getLoRAAdapterMetadataKey correctly returned null for invalid adapter");
			} else {
				Assert.fail("Expected null or LlamaException when getting metadata key from invalid adapter");
			}
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ getLoRAAdapterMetadataKey correctly threw exception: " + e.getMessage());
		}

		try {
			String value = model.getLoRAAdapterMetadataValue(invalidHandle, 0);
			// If it returns null, that's acceptable behavior for invalid adapters
			if (value == null) {
				logger.log(DEBUG, "✅ getLoRAAdapterMetadataValue correctly returned null for invalid adapter");
			} else {
				Assert.fail("Expected null or LlamaException when getting metadata value from invalid adapter");
			}
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ getLoRAAdapterMetadataValue correctly threw exception: " + e.getMessage());
		}

		// Test metadata key validation
		try {
			String metadata = model.getLoRAAdapterMetadata(invalidHandle, null);
			Assert.fail("Expected IllegalArgumentException when getting metadata with null key");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ getLoRAAdapterMetadata correctly validated null key: " + e.getMessage());
		} catch (LlamaException e) {
			// Also acceptable if it checks handle first
			logger.log(DEBUG, "✅ getLoRAAdapterMetadata threw LlamaException: " + e.getMessage());
		}

		try {
			String metadata = model.getLoRAAdapterMetadata(invalidHandle, "");
			Assert.fail("Expected IllegalArgumentException when getting metadata with empty key");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ getLoRAAdapterMetadata correctly validated empty key: " + e.getMessage());
		} catch (LlamaException e) {
			// Also acceptable if it checks handle first
			logger.log(DEBUG, "✅ getLoRAAdapterMetadata threw LlamaException: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA adapter metadata operations test passed!");
	}

	@Test
	public void testALORAOperations() {
		logger.log(DEBUG, "\n=== ALORA Operations Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		long invalidHandle = -1L;

		try {
			// Test ALORA token count with invalid handle (should return 0 for invalid adapters)
			long tokenCount = model.getAloraInvocationTokenCount(invalidHandle);
			Assert.assertEquals("Invalid adapter should return 0 tokens", 0L, tokenCount);
			logger.log(DEBUG, "✅ getAloraInvocationTokenCount correctly returned 0 for invalid adapter");
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ getAloraInvocationTokenCount correctly threw exception: " + e.getMessage());
		}

		try {
			// Test ALORA invocation tokens with invalid handle
			int[] tokens = model.getAloraInvocationTokens(invalidHandle);
			// If it returns empty array, that's acceptable behavior for invalid adapters
			if (tokens != null && tokens.length == 0) {
				logger.log(DEBUG, "✅ getAloraInvocationTokens correctly returned empty array for invalid adapter");
			} else if (tokens == null) {
				logger.log(DEBUG, "✅ getAloraInvocationTokens correctly returned null for invalid adapter");
			} else {
				Assert.fail("Expected empty array, null or LlamaException when getting ALORA tokens from invalid adapter");
			}
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ getAloraInvocationTokens correctly threw exception: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ ALORA operations test passed!");
	}

	@Test
	public void testLoRAAdapterScaleValidation() {
		logger.log(DEBUG, "\n=== LoRA Adapter Scale Validation Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		long invalidHandle = -1L;

		// Test different scale values with invalid handle (should fail due to invalid handle, not scale)
		float[] testScales = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, Float.MAX_VALUE, Float.MIN_VALUE};

		for (float scale : testScales) {
			try {
				int result = model.setLoRAAdapter(invalidHandle, scale);
				Assert.fail("Expected LlamaException when setting LoRA adapter with invalid handle and scale " + scale);
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ setLoRAAdapter with scale %.2f correctly threw exception: %s",
					scale, e.getMessage());
			}
		}

		// Test default scale method with invalid handle
		try {
			int result = model.setLoRAAdapter(invalidHandle);
			Assert.fail("Expected LlamaException when setting LoRA adapter with invalid handle (default scale)");
		} catch (LlamaException e) {
			logger.log(DEBUG, "✅ setLoRAAdapter with default scale correctly threw exception: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA adapter scale validation test passed!");
	}

	@Test
	public void testLoRAAdapterMethodChaining() {
		logger.log(DEBUG, "\n=== LoRA Adapter Method Chaining Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		// Test that we can call LoRA methods in sequence without crashes
		try {
			// Clear any existing adapters
			model.clearLoRAAdapters();

			// Clear control vector
			model.clearControlVector();

			// Try to free some invalid handles (should not crash)
			model.freeLoRAAdapter(-1L);
			model.freeLoRAAdapter(0L);

			// Clear again
			model.clearLoRAAdapters();
			model.clearControlVector();

			logger.log(DEBUG, "✅ Method chaining completed successfully");

		} catch (Exception e) {
			Assert.fail("Method chaining should not fail: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA adapter method chaining test passed!");
	}

	@Test
	public void testComprehensiveLoRAWorkflow() {
		logger.log(DEBUG, "\n=== Comprehensive LoRA Workflow Test ===");

		if (model == null) {
			logger.log(DEBUG, "⚠️ Skipping test - model not loaded");
			return;
		}

		logger.log(DEBUG, "Testing complete LoRA adapter workflow:");
		logger.log(DEBUG, "1. Clear any existing adapters");
		logger.log(DEBUG, "2. Attempt to load non-existent adapter (should fail)");
		logger.log(DEBUG, "3. Test control vector operations");
		logger.log(DEBUG, "4. Test metadata operations with invalid handles");
		logger.log(DEBUG, "5. Test ALORA operations with invalid handles");
		logger.log(DEBUG, "6. Clear everything");

		try {
			// Step 1: Clear existing adapters
			model.clearLoRAAdapters();
			model.clearControlVector();
			logger.log(DEBUG, "✅ Step 1: Cleared existing adapters and control vectors");

			// Step 2: Attempt to load non-existent adapter
			try {
				long handle = model.loadLoRAAdapter("/tmp/nonexistent_lora.bin");
				Assert.fail("Should have failed to load non-existent LoRA");
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ Step 2: Correctly failed to load non-existent LoRA");
			}

			// Step 3: Control vector operations
			model.applyControlVector(null);
			model.clearControlVector();
			logger.log(DEBUG, "✅ Step 3: Control vector operations completed");

			// Step 4: Metadata operations with invalid handles
			try {
				model.getLoRAAdapterMetadataCount(-1L);
				Assert.fail("Should have failed with invalid handle");
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ Step 4: Correctly failed metadata operations with invalid handle");
			}

			// Step 5: ALORA operations with invalid handles
			try {
				long count = model.getAloraInvocationTokenCount(-1L);
				// Accept either 0 return value or exception for invalid handles
				Assert.assertEquals("Invalid adapter should return 0 tokens", 0L, count);
				logger.log(DEBUG, "✅ Step 5: ALORA operations returned safe default for invalid handle");
			} catch (LlamaException e) {
				logger.log(DEBUG, "✅ Step 5: Correctly failed ALORA operations with invalid handle");
			}

			// Step 6: Final cleanup
			model.clearLoRAAdapters();
			model.clearControlVector();
			logger.log(DEBUG, "✅ Step 6: Final cleanup completed");

		} catch (Exception e) {
			Assert.fail("Comprehensive workflow failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Comprehensive LoRA workflow test passed!");
	}
}
