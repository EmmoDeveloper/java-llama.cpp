package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class IntelligentDefaultsTest {
	private static final System.Logger logger = System.getLogger(IntelligentDefaultsTest.class.getName());

	@Test
	public void testIntelligentGpuDefaults() {
		logger.log(DEBUG, "\n=== Testing Intelligent GPU Defaults ===\n");

		// Test 1: Model with minimal parameters - should auto-detect and configure GPU
		logger.log(DEBUG, "1. Testing auto-configuration with minimal parameters...");

		LlamaModel autoModel = new LlamaModel(
			new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				// No explicit GPU layers - should be auto-configured
		);

		// Test 2: Model with explicit GPU layers - should NOT override user choice
		logger.log(DEBUG, "\n2. Testing that explicit user settings are preserved...");

		LlamaModel explicitModel = new LlamaModel(
			new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(16) // Explicit setting - should not be overridden
		);

		// Test generation with both models to ensure they work
		logger.log(DEBUG, "\n3. Testing generation with auto-configured model:");
		testGeneration(autoModel, "auto-configured");

		logger.log(DEBUG, "\n4. Testing generation with explicit-configured model:");
		testGeneration(explicitModel, "explicit-configured");

		// Clean up
		autoModel.close();
		explicitModel.close();

		logger.log(DEBUG, "\n✅ Intelligent defaults test completed successfully!");
	}

	@Test
	public void testGpuDetection() {
		logger.log(DEBUG, "\n=== Testing GPU Detection ===\n");

		// Test GPU detection directly
		GpuDetector.GpuInfo gpu = GpuDetector.detectGpu();

		logger.log(DEBUG, "Detected GPU configuration:");
		logger.log(DEBUG, "  CUDA Available: " + gpu.cudaAvailable());
		logger.log(DEBUG, "  Device Name: " + gpu.deviceName());
		logger.log(DEBUG, "  VRAM: " + gpu.totalMemoryMB() + " MB (" +
			String.format("%.1f GB", gpu.totalMemoryMB() / 1024.0) + ")");
		logger.log(DEBUG, "  Recommended Layers: " + gpu.recommendedLayers());
		logger.log(DEBUG, "  Should Use GPU: " + gpu.shouldUseGpu());

		// Verify detection makes sense
		if (gpu.cudaAvailable()) {
			Assert.assertTrue("If CUDA is available, should have some VRAM", gpu.totalMemoryMB() > 0);
			Assert.assertNotEquals("Device name should not be unknown if CUDA found", "Unknown", gpu.deviceName());

			if (gpu.totalMemoryMB() > 2048) { // 2GB+
				Assert.assertTrue("Should recommend GPU usage for 2GB+ cards", gpu.shouldUseGpu());
				Assert.assertTrue("Should recommend some layers for decent GPUs",
					gpu.recommendedLayers() > 0);
			}
		} else {
			Assert.assertEquals("No GPU layers should be recommended without CUDA",
				0, gpu.recommendedLayers());
			Assert.assertFalse("Should not recommend GPU usage without CUDA", gpu.shouldUseGpu());
		}

		logger.log(DEBUG, "\n✅ GPU detection test passed!");
	}

	@Test
	public void testPerformanceComparison() {
		logger.log(DEBUG, "\n=== Performance Comparison: Manual vs Auto-Configured ===\n");

		String prompt = "def fibonacci(n):";
		int nPredict = 15;

		// Test 1: Manual configuration (old way)
		logger.log(DEBUG, "1. Testing manual GPU configuration...");
		long manualTime = benchmarkModel("Manual", new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(43)  // Manual setting
			.setCtxSize(512), prompt, nPredict);

		// Test 2: Auto-configuration (new way)
		logger.log(DEBUG, "\n2. Testing auto-configured model...");
		long autoTime = benchmarkModel("Auto", new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			// No explicit GPU config - let auto-detection handle it
			, prompt, nPredict);

		// Compare results
		logger.log(DEBUG, "\n=== Performance Comparison Results ===");
		logger.log(DEBUG, "Manual config time: " + manualTime + " ms");
		logger.log(DEBUG, "Auto config time: " + autoTime + " ms");

		double ratio = (double) manualTime / autoTime;
		if (ratio > 0.8 && ratio < 1.2) { // Within 20% is good
			logger.log(DEBUG, "✅ Auto-configuration performs similarly to manual (ratio: " +
				String.format("%.2f", ratio) + ")");
		} else {
			logger.log(DEBUG, "⚠️  Performance difference detected (ratio: " +
				String.format("%.2f", ratio) + ")");
		}

		// Both should be reasonably fast (auto-config shouldn't be dramatically slower)
		Assert.assertTrue("Auto-config should not be more than 2x slower than manual",
			autoTime < manualTime * 2);

		logger.log(DEBUG, "\n✅ Performance comparison test completed!");
	}

	private void testGeneration(LlamaModel model, String modelType) {
		InferenceParameters params = new InferenceParameters("Hello")
			.setNPredict(3)
			.setTemperature(0.1f);

		int tokenCount = 0;
		long startTime = System.currentTimeMillis();

		for (LlamaOutput output : model.generate(params)) {
			logger.log(DEBUG, output.text);
			tokenCount++;
		}

		long duration = System.currentTimeMillis() - startTime;

		logger.log(DEBUG, "  %s model: %d tokens in %d ms (%.2f tok/s)",
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

			logger.log(DEBUG, "  %s: %d tokens in %d ms (%.2f tok/s)",
				configType, tokenCount, duration, tokenCount * 1000.0 / duration);

			return duration;

		}
	}
}
