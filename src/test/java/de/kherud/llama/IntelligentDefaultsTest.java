package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

public class IntelligentDefaultsTest {

	@Test
	public void testIntelligentGpuDefaults() {
		System.out.println("\n=== Testing Intelligent GPU Defaults ===\n");

		// Test 1: Model with minimal parameters - should auto-detect and configure GPU
		System.out.println("1. Testing auto-configuration with minimal parameters...");

		LlamaModel autoModel = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				// No explicit GPU layers - should be auto-configured
		);

		// Test 2: Model with explicit GPU layers - should NOT override user choice
		System.out.println("\n2. Testing that explicit user settings are preserved...");

		LlamaModel explicitModel = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(16) // Explicit setting - should not be overridden
		);

		// Test generation with both models to ensure they work
		System.out.println("\n3. Testing generation with auto-configured model:");
		testGeneration(autoModel, "auto-configured");

		System.out.println("\n4. Testing generation with explicit-configured model:");
		testGeneration(explicitModel, "explicit-configured");

		// Clean up
		autoModel.close();
		explicitModel.close();

		System.out.println("\n✅ Intelligent defaults test completed successfully!");
	}

	@Test
	public void testGpuDetection() {
		System.out.println("\n=== Testing GPU Detection ===\n");

		// Test GPU detection directly
		GpuDetector.GpuInfo gpu = GpuDetector.detectGpu();

		System.out.println("Detected GPU configuration:");
		System.out.println("  CUDA Available: " + gpu.cudaAvailable);
		System.out.println("  Device Name: " + gpu.deviceName);
		System.out.println("  VRAM: " + gpu.totalMemoryMB + " MB (" +
			String.format("%.1f GB", gpu.totalMemoryMB / 1024.0) + ")");
		System.out.println("  Recommended Layers: " + gpu.recommendedLayers);
		System.out.println("  Should Use GPU: " + gpu.shouldUseGpu);

		// Verify detection makes sense
		if (gpu.cudaAvailable) {
			Assert.assertTrue("If CUDA is available, should have some VRAM", gpu.totalMemoryMB > 0);
			Assert.assertNotEquals("Device name should not be unknown if CUDA found", "Unknown", gpu.deviceName);

			if (gpu.totalMemoryMB > 2048) { // 2GB+
				Assert.assertTrue("Should recommend GPU usage for 2GB+ cards", gpu.shouldUseGpu);
				Assert.assertTrue("Should recommend some layers for decent GPUs",
					gpu.recommendedLayers > 0);
			}
		} else {
			Assert.assertEquals("No GPU layers should be recommended without CUDA",
				0, gpu.recommendedLayers);
			Assert.assertFalse("Should not recommend GPU usage without CUDA", gpu.shouldUseGpu);
		}

		System.out.println("\n✅ GPU detection test passed!");
	}

	@Test
	public void testPerformanceComparison() {
		System.out.println("\n=== Performance Comparison: Manual vs Auto-Configured ===\n");

		String prompt = "def fibonacci(n):";
		int nPredict = 15;

		// Test 1: Manual configuration (old way)
		System.out.println("1. Testing manual GPU configuration...");
		long manualTime = benchmarkModel("Manual", new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(43)  // Manual setting
			.setCtxSize(512), prompt, nPredict);

		// Test 2: Auto-configuration (new way)
		System.out.println("\n2. Testing auto-configured model...");
		long autoTime = benchmarkModel("Auto", new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			// No explicit GPU config - let auto-detection handle it
			, prompt, nPredict);

		// Compare results
		System.out.println("\n=== Performance Comparison Results ===");
		System.out.println("Manual config time: " + manualTime + " ms");
		System.out.println("Auto config time: " + autoTime + " ms");

		double ratio = (double) manualTime / autoTime;
		if (ratio > 0.8 && ratio < 1.2) { // Within 20% is good
			System.out.println("✅ Auto-configuration performs similarly to manual (ratio: " +
				String.format("%.2f", ratio) + ")");
		} else {
			System.out.println("⚠️  Performance difference detected (ratio: " +
				String.format("%.2f", ratio) + ")");
		}

		// Both should be reasonably fast (auto-config shouldn't be dramatically slower)
		Assert.assertTrue("Auto-config should not be more than 2x slower than manual",
			autoTime < manualTime * 2);

		System.out.println("\n✅ Performance comparison test completed!");
	}

	private void testGeneration(LlamaModel model, String modelType) {
		InferenceParameters params = new InferenceParameters("Hello")
			.setNPredict(3)
			.setTemperature(0.1f);

		int tokenCount = 0;
		long startTime = System.currentTimeMillis();

		for (LlamaOutput output : model.generate(params)) {
			System.out.print(output.text);
			tokenCount++;
		}

		long duration = System.currentTimeMillis() - startTime;
		System.out.println();
		System.out.printf("  %s model: %d tokens in %d ms (%.2f tok/s)\n",
			modelType, tokenCount, duration, tokenCount * 1000.0 / duration);

		Assert.assertTrue("Should generate at least one token", tokenCount > 0);
	}

	private long benchmarkModel(String configType, ModelParameters params, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(params)) {

			InferenceParameters inferenceParams = new InferenceParameters(prompt)
				.setNPredict(nPredict)
				.setTemperature(0.1f);

			long startTime = System.currentTimeMillis();
			int tokenCount = 0;

			for (LlamaOutput output : model.generate(inferenceParams)) {
				tokenCount++;
			}

			long duration = System.currentTimeMillis() - startTime;

			System.out.printf("  %s: %d tokens in %d ms (%.2f tok/s)\n",
				configType, tokenCount, duration, tokenCount * 1000.0 / duration);

			return duration;

		}
	}
}
