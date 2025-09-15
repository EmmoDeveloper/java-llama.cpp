package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import static de.kherud.llama.MemoryLeakTest.formatBytes;
import static de.kherud.llama.MemoryLeakTest.getUsedMemory;

/**
 * Comprehensive memory leak testing for java-llama.cpp
 * Tests various scenarios that could lead to native memory leaks
 */
public class MemoryLeakEmbeddingTest {
	private static final System.Logger logger = System.getLogger(MemoryLeakEmbeddingTest.class.getName());

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
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
				.enableEmbedding()
				.enableLogTimestamps()
				.enableLogPrefix()
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
	public void testRepeatedEmbedding() {
		logger.log(DEBUG, "\n=== Memory Leak Test: Repeated Embedding ===");

		long initialMemory = getUsedMemory();
		logger.log(DEBUG, "Initial memory usage: " + formatBytes(initialMemory));

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
				logger.log(DEBUG, "Memory after " + (i + 1) + " embeddings: " + formatBytes(currentMemory));
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

		logger.log(DEBUG, "Final memory usage: " + formatBytes(finalMemory));
		logger.log(DEBUG, "Memory growth: " + formatBytes(memoryGrowth));

		// Allow for some memory growth but flag excessive growth (>30MB)
		Assert.assertTrue("Excessive memory growth detected: " + formatBytes(memoryGrowth),
			memoryGrowth < 30 * 1024 * 1024);

		logger.log(DEBUG, "✅ Repeated embedding test passed!");
	}
}
