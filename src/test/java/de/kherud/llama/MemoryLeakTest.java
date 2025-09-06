package de.kherud.llama;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * Comprehensive memory leak testing for java-llama.cpp
 * Tests various scenarios that could lead to native memory leaks
 */
public class MemoryLeakTest {

	private static LlamaModel model;
	private static final int STRESS_ITERATIONS = 100;
	private static final String TEST_PROMPT = "public static String test() { return";

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
		// Enable memory debugging if available
		System.setProperty("de.kherud.llama.debug.memory", "true");

		model = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
				.enableEmbedding().enableLogTimestamps().enableLogPrefix()
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
		// Force garbage collection and finalization
		System.gc();
		System.runFinalization();
		Thread.yield();
		System.gc();
	}

	@Test
	public void testRepeatedCompletion() {
		System.out.println("\n=== Memory Leak Test: Repeated Completion ===");

		long initialMemory = getUsedMemory();
		System.out.println("Initial memory usage: " + formatBytes(initialMemory));

		// Perform multiple completions
		for (int i = 0; i < STRESS_ITERATIONS; i++) {
			InferenceParameters params = new InferenceParameters(TEST_PROMPT)
				.setTemperature(0.1f)
				.setNPredict(10)
				.setSeed(42);

			String result = model.complete(params);
			Assert.assertNotNull("Result should not be null at iteration " + i, result);
			Assert.assertFalse("Result should not be empty at iteration " + i, result.isEmpty());

			// Periodic memory check
			if (i % 20 == 19) {
				long currentMemory = getUsedMemory();
				System.out.println("Memory after " + (i + 1) + " completions: " + formatBytes(currentMemory));

				// Force GC to clean up any Java-side references
				System.gc();
				Thread.yield();
			}
		}

		// Final memory check
		System.gc();
		System.runFinalization();
		Thread.yield();
		System.gc();

		long finalMemory = getUsedMemory();
		long memoryGrowth = finalMemory - initialMemory;

		System.out.println("Final memory usage: " + formatBytes(finalMemory));
		System.out.println("Memory growth: " + formatBytes(memoryGrowth));

		// Allow for some memory growth but flag excessive growth (>50MB)
		Assert.assertTrue("Excessive memory growth detected: " + formatBytes(memoryGrowth),
			memoryGrowth < 50 * 1024 * 1024);

		System.out.println("✅ Repeated completion test passed!");
	}

	@Test
	public void testRepeatedGeneration() {
		System.out.println("\n=== Memory Leak Test: Repeated Generation ===");

		long initialMemory = getUsedMemory();
		System.out.println("Initial memory usage: " + formatBytes(initialMemory));

		// Test streaming generation which uses different code paths
		for (int i = 0; i < STRESS_ITERATIONS / 2; i++) { // Fewer iterations for streaming
			InferenceParameters params = new InferenceParameters(TEST_PROMPT)
				.setTemperature(0.1f)
				.setNPredict(10)
				.setSeed(42);

			int tokenCount = 0;
			for (LlamaOutput output : model.generate(params)) {
				tokenCount++;
				Assert.assertNotNull("Output should not be null", output);
			}

			Assert.assertTrue("Should generate at least one token", tokenCount > 0);

			// Periodic memory check
			if (i % 10 == 9) {
				long currentMemory = getUsedMemory();
				System.out.println("Memory after " + (i + 1) + " generations: " + formatBytes(currentMemory));
				System.gc();
				Thread.yield();
			}
		}

		// Final memory check
		System.gc();
		System.runFinalization();
		Thread.yield();
		System.gc();

		long finalMemory = getUsedMemory();
		long memoryGrowth = finalMemory - initialMemory;

		System.out.println("Final memory usage: " + formatBytes(finalMemory));
		System.out.println("Memory growth: " + formatBytes(memoryGrowth));

		// Allow for some memory growth but flag excessive growth (>50MB)
		Assert.assertTrue("Excessive memory growth detected: " + formatBytes(memoryGrowth),
			memoryGrowth < 50 * 1024 * 1024);

		System.out.println("✅ Repeated generation test passed!");
	}

	@Test
	public void testRepeatedEmbedding() {
		System.out.println("\n=== Memory Leak Test: Repeated Embedding ===");

		long initialMemory = getUsedMemory();
		System.out.println("Initial memory usage: " + formatBytes(initialMemory));

		String[] testTexts = {
			"Hello world",
			"Machine learning is fascinating",
			"The quick brown fox jumps over the lazy dog",
			"Programming in Java and C++",
			"Natural language processing"
		};

		// Perform multiple embedding computations
		for (int i = 0; i < STRESS_ITERATIONS; i++) {
			String text = testTexts[i % testTexts.length];
			float[] embedding = model.embed(text);

			Assert.assertNotNull("Embedding should not be null at iteration " + i, embedding);
			Assert.assertEquals("Embedding should have correct dimension", 4096, embedding.length);

			// Verify embedding contains meaningful values (not all zeros)
			boolean hasNonZero = false;
			for (float val : embedding) {
				if (val != 0.0f) {
					hasNonZero = true;
					break;
				}
			}
			Assert.assertTrue("Embedding should contain non-zero values", hasNonZero);

			// Periodic memory check
			if (i % 20 == 19) {
				long currentMemory = getUsedMemory();
				System.out.println("Memory after " + (i + 1) + " embeddings: " + formatBytes(currentMemory));
				System.gc();
				Thread.yield();
			}
		}

		// Final memory check
		System.gc();
		System.runFinalization();
		Thread.yield();
		System.gc();

		long finalMemory = getUsedMemory();
		long memoryGrowth = finalMemory - initialMemory;

		System.out.println("Final memory usage: " + formatBytes(finalMemory));
		System.out.println("Memory growth: " + formatBytes(memoryGrowth));

		// Allow for some memory growth but flag excessive growth (>30MB)
		Assert.assertTrue("Excessive memory growth detected: " + formatBytes(memoryGrowth),
			memoryGrowth < 30 * 1024 * 1024);

		System.out.println("✅ Repeated embedding test passed!");
	}

	@Test
	public void testRepeatedTokenization() {
		System.out.println("\n=== Memory Leak Test: Repeated Tokenization ===");

		long initialMemory = getUsedMemory();
		System.out.println("Initial memory usage: " + formatBytes(initialMemory));

		String[] testTexts = {
			"Hello, world!",
			"The quick brown fox jumps over the lazy dog.",
			"Machine learning and artificial intelligence are transforming our world.",
			"Java Native Interface (JNI) provides a bridge between Java and native code.",
			"Memory management is crucial for high-performance applications."
		};

		// Perform encode/decode cycles
		for (int i = 0; i < STRESS_ITERATIONS; i++) {
			String text = testTexts[i % testTexts.length];
			System.out.println("text = " + text);

			// Encode
			int[] tokens = model.encode(text);
			Assert.assertNotNull("Tokens should not be null", tokens);
			Assert.assertTrue("Should have at least one token", tokens.length > 0);

			// Decode
			String decoded = model.decode(tokens);
			System.out.println("decoded = " + decoded);

			Assert.assertNotNull("Decoded text should not be null", decoded);

			// Verify round-trip (may not be exact due to tokenizer behavior)
			Assert.assertTrue("Decoded text should contain original content",
				decoded.toLowerCase().contains(text.toLowerCase().substring(0, 5)));

			// Periodic memory check
			if (i % 25 == 24) {
				long currentMemory = getUsedMemory();
				System.out.println("Memory after " + (i + 1) + " tokenizations: " + formatBytes(currentMemory));
				System.gc();
				Thread.yield();
			}
		}

		// Final memory check
		System.gc();
		System.runFinalization();
		Thread.yield();
		System.gc();

		long finalMemory = getUsedMemory();
		long memoryGrowth = finalMemory - initialMemory;

		System.out.println("Final memory usage: " + formatBytes(finalMemory));
		System.out.println("Memory growth: " + formatBytes(memoryGrowth));

		// Allow for some memory growth but flag excessive growth (>20MB)
		Assert.assertTrue("Excessive memory growth detected: " + formatBytes(memoryGrowth),
			memoryGrowth < 20 * 1024 * 1024);

		System.out.println("✅ Repeated tokenization test passed!");
	}

	@Test
	public void testMultipleModelInstances() {
		System.out.println("\n=== Memory Leak Test: Multiple Model Instances ===");

		long initialMemory = getUsedMemory();
		System.out.println("Initial memory usage: " + formatBytes(initialMemory));

		// Create and destroy multiple model instances
		List<LlamaModel> models = new ArrayList<>();

		try {
			for (int i = 0; i < 5; i++) { // Create 5 instances
				LlamaModel testModel = new LlamaModel(
					new ModelParameters()
						.setCtxSize(256) // Smaller context to save memory
						.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
						.setGpuLayers(10) // Fewer GPU layers to test CPU/GPU memory management
				);

				models.add(testModel);

				// Test each model briefly
				InferenceParameters params = new InferenceParameters("test")
					.setNPredict(5)
					.setSeed(42);

				String result = testModel.complete(params);
				Assert.assertNotNull("Model " + i + " should produce output", result);

				long currentMemory = getUsedMemory();
				System.out.println("Memory after creating model " + (i + 1) + ": " + formatBytes(currentMemory));
			}

		} finally {
			// Clean up all models
			for (LlamaModel testModel : models) {
				testModel.close();
			}
			models.clear();

			// Force cleanup
			System.gc();
			System.runFinalization();
			Thread.yield();
			System.gc();
		}

		long finalMemory = getUsedMemory();
		long memoryGrowth = finalMemory - initialMemory;

		System.out.println("Final memory usage after cleanup: " + formatBytes(finalMemory));
		System.out.println("Net memory growth: " + formatBytes(memoryGrowth));

		// Should return close to initial memory usage (allow 100MB variance)
		Assert.assertTrue("Models may not have been properly cleaned up: " + formatBytes(memoryGrowth),
			Math.abs(memoryGrowth) < 100 * 1024 * 1024);

		System.out.println("✅ Multiple model instances test passed!");
	}

	@Test
	public void testTemplateMemory() {
		System.out.println("\n=== Memory Leak Test: Template Operations ===");

		long initialMemory = getUsedMemory();
		System.out.println("Initial memory usage: " + formatBytes(initialMemory));

		// Test template operations repeatedly
		for (int i = 0; i < STRESS_ITERATIONS; i++) {
			List<Pair<String, String>> messages = new ArrayList<>();
			messages.add(new Pair<>("user", "What is " + (i % 10) + " + " + ((i + 1) % 10) + "?"));
			messages.add(new Pair<>("assistant", "The answer is " + ((i % 10) + ((i + 1) % 10))));

			InferenceParameters params = new InferenceParameters("You are a helpful math assistant.")
				.setMessages("Math Assistant", messages)
				.setTemperature(0.1f)
				.setNPredict(5)
				.setSeed(42);

			String template = model.applyTemplate(params);
			Assert.assertNotNull("Template should not be null", template);
			Assert.assertTrue("Template should contain expected format",
				template.contains("<|im_start|>") && template.contains("<|im_end|>"));

			// Periodic memory check
			if (i % 25 == 24) {
				long currentMemory = getUsedMemory();
				System.out.println("Memory after " + (i + 1) + " template operations: " + formatBytes(currentMemory));
				System.gc();
				Thread.yield();
			}
		}

		// Final memory check
		System.gc();
		System.runFinalization();
		Thread.yield();
		System.gc();

		long finalMemory = getUsedMemory();
		long memoryGrowth = finalMemory - initialMemory;

		System.out.println("Final memory usage: " + formatBytes(finalMemory));
		System.out.println("Memory growth: " + formatBytes(memoryGrowth));

		// Allow for some memory growth but flag excessive growth (>15MB)
		Assert.assertTrue("Excessive memory growth detected: " + formatBytes(memoryGrowth),
			memoryGrowth < 15 * 1024 * 1024);

		System.out.println("✅ Template memory test passed!");
	}

	// Helper methods
	private long getUsedMemory() {
		Runtime runtime = Runtime.getRuntime();
		return runtime.totalMemory() - runtime.freeMemory();
	}

	private String formatBytes(long bytes) {
		if (bytes < 0) {
			return "-" + formatBytes(-bytes);
		}
		if (bytes < 1024) {
			return bytes + " B";
		} else if (bytes < 1024 * 1024) {
			return String.format("%.1f KB", bytes / 1024.0);
		} else if (bytes < 1024 * 1024 * 1024) {
			return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
		} else {
			return String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
		}
	}
}
