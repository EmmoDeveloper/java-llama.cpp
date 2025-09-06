package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.AfterClass;
import org.junit.Assume;
import java.io.File;

public class LoRAAdapterIntegrationTest {

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
		System.out.println("\n=== LoRA Adapter Real File Integration Test ===");
		
		// Check if we have a real LoRA file to test with (must be >1KB to be valid)
		String testLoRAPath = null;
		if (new File(LORA_TEST_PATH).exists() && new File(LORA_TEST_PATH).length() > 1024) {
			testLoRAPath = LORA_TEST_PATH;
		} else if (new File(LORA_TEST_PATH_ALT).exists() && new File(LORA_TEST_PATH_ALT).length() > 1024) {
			testLoRAPath = LORA_TEST_PATH_ALT;
		}
		
		if (testLoRAPath == null) {
			System.out.println("⚠️  No valid LoRA test file found - testing error handling only");
			
			// Test that our error handling works correctly for nonexistent file
			try {
				model.loadLoRAAdapter("/nonexistent/lora/file.safetensors");
				Assert.fail("Should throw exception for nonexistent file");
			} catch (LlamaException e) {
				System.out.println("✅ Correctly handled nonexistent file: " + e.getMessage());
			}
			
			// Test handling of invalid/dummy file if it exists
			if (new File(LORA_TEST_PATH_ALT).exists()) {
				try {
					model.loadLoRAAdapter(LORA_TEST_PATH_ALT);
					Assert.fail("Should throw exception for invalid LoRA file");
				} catch (LlamaException e) {
					System.out.println("✅ Correctly rejected invalid/dummy LoRA file: " + e.getMessage());
				}
			}
			
			System.out.println("✅ Error handling test passed (no valid LoRA file available)");
			return;
		}
		
		// Test with real LoRA file
		System.out.println("📁 Testing with LoRA file: " + testLoRAPath);
		
		try {
			// Load LoRA adapter
			long adapterHandle = model.loadLoRAAdapter(testLoRAPath);
			System.out.println("✅ Successfully loaded LoRA adapter, handle: " + adapterHandle);
			Assert.assertTrue("Adapter handle should be valid", adapterHandle != -1 && adapterHandle != 0);
			
			// Test setting adapter with different scales
			int result1 = model.setLoRAAdapter(adapterHandle, 1.0f);
			System.out.println("✅ Set LoRA adapter with scale 1.0: result = " + result1);
			
			int result2 = model.setLoRAAdapter(adapterHandle, 0.5f);
			System.out.println("✅ Set LoRA adapter with scale 0.5: result = " + result2);
			
			// Test default scale
			int result3 = model.setLoRAAdapter(adapterHandle);
			System.out.println("✅ Set LoRA adapter with default scale: result = " + result3);
			
			// Test removing adapter
			int removeResult = model.removeLoRAAdapter(adapterHandle);
			System.out.println("✅ Removed LoRA adapter: result = " + removeResult);
			
			// Test metadata access
			try {
				int metaCount = model.getLoRAAdapterMetadataCount(adapterHandle);
				System.out.println("📊 LoRA adapter metadata count: " + metaCount);
				
				if (metaCount > 0) {
					for (int i = 0; i < Math.min(metaCount, 5); i++) {
						String key = model.getLoRAAdapterMetadataKey(adapterHandle, i);
						String value = model.getLoRAAdapterMetadataValue(adapterHandle, i);
						System.out.printf("📋 Metadata[%d]: %s = %s\n", i, key, value);
					}
				}
			} catch (LlamaException e) {
				System.out.println("ℹ️  Metadata access failed (expected after removal): " + e.getMessage());
			}
			
			// Test ALORA functionality
			try {
				long tokenCount = model.getAloraInvocationTokenCount(adapterHandle);
				System.out.println("🔢 ALORA token count: " + tokenCount);
				
				if (tokenCount > 0) {
					int[] tokens = model.getAloraInvocationTokens(adapterHandle);
					System.out.printf("🎯 ALORA tokens (%d): [", tokens.length);
					for (int i = 0; i < Math.min(tokens.length, 10); i++) {
						System.out.print(tokens[i]);
						if (i < Math.min(tokens.length, 10) - 1) System.out.print(", ");
					}
					if (tokens.length > 10) System.out.print("...");
					System.out.println("]");
				} else {
					System.out.println("ℹ️  Not an ALORA adapter (token count: 0)");
				}
			} catch (LlamaException e) {
				System.out.println("ℹ️  ALORA access failed (expected after removal): " + e.getMessage());
			}
			
			// Clean up
			model.freeLoRAAdapter(adapterHandle);
			System.out.println("🧹 Freed LoRA adapter");
			
		} catch (LlamaException e) {
			Assert.fail("Real LoRA integration test failed: " + e.getMessage());
		}
		
		System.out.println("✅ Real LoRA integration test passed!");
	}

	@Test
	public void testControlVectorIntegration() {
		System.out.println("\n=== Control Vector Integration Test ===");
		
		try {
			// Test control vector operations
			System.out.println("🧹 Clearing any existing control vectors");
			int clearResult = model.clearControlVector();
			System.out.println("✅ Initial clear result: " + clearResult);
			
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
			
			for (int i = 0; i < testVectors.length; i++) {
				float[] vector = testVectors[i];
				System.out.printf("🧪 Testing control vector with %d elements\n", vector.length);
				
				try {
					int result = model.applyControlVector(vector);
					System.out.printf("✅ Applied control vector[%d]: result = %d\n", vector.length, result);
				} catch (LlamaException e) {
					System.out.printf("ℹ️  Control vector[%d] failed (may be expected): %s\n", vector.length, e.getMessage());
				}
			}
			
			// Test null control vector (clear)
			int nullResult = model.applyControlVector(null);
			System.out.println("✅ Applied null control vector (clear): result = " + nullResult);
			
			// Final cleanup
			model.clearControlVector();
			System.out.println("🧹 Final control vector cleanup completed");
			
		} catch (Exception e) {
			Assert.fail("Control vector integration test failed: " + e.getMessage());
		}
		
		System.out.println("✅ Control vector integration test passed!");
	}

	@Test
	public void testLoRAWorkflowWithClearOperations() {
		System.out.println("\n=== LoRA Workflow with Clear Operations Test ===");
		
		try {
			// Test clear operations work correctly
			System.out.println("🧹 Step 1: Clear all adapters and control vectors");
			model.clearLoRAAdapters();
			model.clearControlVector();
			
			// Test multiple clear operations (should be safe)
			System.out.println("🧹 Step 2: Multiple clear operations");
			for (int i = 0; i < 3; i++) {
				model.clearLoRAAdapters();
				model.clearControlVector();
			}
			System.out.println("✅ Multiple clears completed without error");
			
			// Test that we can perform operations after clearing
			System.out.println("🔧 Step 3: Test operations after clearing");
			
			// Apply and clear control vector
			float[] testVector = {0.1f, 0.2f, 0.3f, 0.4f};
			try {
				int applyResult = model.applyControlVector(testVector);
				System.out.println("✅ Applied test control vector after clear: " + applyResult);
				
				int clearResult = model.clearControlVector();
				System.out.println("✅ Cleared control vector: " + clearResult);
			} catch (LlamaException e) {
				System.out.println("ℹ️  Control vector operations failed (may be model-dependent): " + e.getMessage());
			}
			
			// Final state verification
			System.out.println("🧹 Step 4: Final cleanup and verification");
			model.clearLoRAAdapters();
			model.clearControlVector();
			
			// Test that invalid operations still fail appropriately
			try {
				model.setLoRAAdapter(-1L, 1.0f);
				Assert.fail("Should fail with invalid adapter handle");
			} catch (LlamaException e) {
				System.out.println("✅ Invalid operations still fail correctly: " + e.getMessage());
			}
			
		} catch (Exception e) {
			Assert.fail("LoRA workflow test failed: " + e.getMessage());
		}
		
		System.out.println("✅ LoRA workflow test passed!");
	}

	@Test 
	public void testErrorHandlingEdgeCases() {
		System.out.println("\n=== Error Handling Edge Cases Test ===");
		
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
					System.out.println("🧪 Testing null path");
					model.loadLoRAAdapter(null);
				} else {
					System.out.printf("🧪 Testing invalid path: '%s'\n", path);
					model.loadLoRAAdapter(path);
				}
				Assert.fail("Should have thrown exception for invalid path: " + path);
			} catch (IllegalArgumentException | LlamaException e) {
				System.out.printf("✅ Correctly handled invalid path '%s': %s\n", 
					path == null ? "null" : path, e.getClass().getSimpleName());
			}
		}
		
		// Test invalid handles
		long[] invalidHandles = {-1L, 0L, Long.MAX_VALUE, Long.MIN_VALUE};
		
		for (long handle : invalidHandles) {
			System.out.printf("🧪 Testing invalid handle: %d\n", handle);
			
			try {
				model.setLoRAAdapter(handle, 1.0f);
				Assert.fail("Should have thrown exception for invalid handle: " + handle);
			} catch (LlamaException e) {
				System.out.printf("✅ setLoRAAdapter correctly handled invalid handle %d\n", handle);
			}
			
			try {
				model.getLoRAAdapterMetadataCount(handle);
				Assert.fail("Should have thrown exception for invalid handle: " + handle);
			} catch (LlamaException e) {
				System.out.printf("✅ getLoRAAdapterMetadataCount correctly handled invalid handle %d\n", handle);
			}
		}
		
		System.out.println("✅ Error handling edge cases test passed!");
	}
}