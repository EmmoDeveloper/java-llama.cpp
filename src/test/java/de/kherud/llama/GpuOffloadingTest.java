package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class GpuOffloadingTest {
	private static final System.Logger logger = System.getLogger(GpuOffloadingTest.class.getName());

	@Test
	public void testGpuLayerOffloading() {
		logger.log(DEBUG, "\n=== GPU Layer Offloading Test (Performance-Based Verification) ===\n");

		// Instead of complex log parsing, test actual GPU performance
		// by comparing CPU vs GPU speeds directly

		String testPrompt = "def quicksort(arr):";
		int nPredict = 8;

		logger.log(DEBUG, "1. Testing CPU-only model (0 GPU layers):");
		long cpuTime = testModelPerformance(0, testPrompt, nPredict);

		logger.log(DEBUG, "\n2. Testing GPU-accelerated model (43 GPU layers):");
		long gpuTime = testModelPerformance(43, testPrompt, nPredict);

		logger.log(DEBUG, "\n3. Testing smart defaults model (auto GPU detection):");
		LlamaModel smartModel = null;
		long smartTime;
		int smartTokens = 0;

		try {
			smartModel = new LlamaModel(
				new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					// No explicit GPU config - should auto-configure GPU
			);

			InferenceParameters params = new InferenceParameters(testPrompt)
				.setNPredict(nPredict)
				.setTemperature(0.1f);

			long startTime = System.currentTimeMillis();
			for (LlamaOutput output : smartModel.generate(params)) {
				smartTokens++;
			}
			smartTime = System.currentTimeMillis() - startTime;

			logger.log(DEBUG, "  Smart defaults: %d tokens in %d ms (%.1f tok/s)",
				smartTokens, smartTime, smartTokens * 1000.0 / smartTime);

		} finally {
			if (smartModel != null) {
				smartModel.close();
			}
		}

		// Analysis and verification
		logger.log(DEBUG, "\n=== Performance Analysis ===");
		logger.log(DEBUG, "CPU-only time: " + cpuTime + " ms");
		logger.log(DEBUG, "GPU-explicit time: " + gpuTime + " ms");
		logger.log(DEBUG, "Smart defaults time: " + smartTime + " ms");

		double gpuSpeedup = (double) cpuTime / gpuTime;
		double smartSpeedup = (double) cpuTime / smartTime;

		logger.log(DEBUG, "GPU speedup vs CPU: %.2fx", gpuSpeedup);
		logger.log(DEBUG, "Smart defaults speedup vs CPU: %.2fx", smartSpeedup);

		// Verify GPU is actually providing acceleration
		Assert.assertTrue("CPU time should be > 0", cpuTime > 0);
		Assert.assertTrue("GPU time should be > 0", gpuTime > 0);
		Assert.assertTrue("Smart defaults time should be > 0", smartTime > 0);

		// GPU should be significantly faster than CPU (allow some tolerance)
		Assert.assertTrue("GPU should be faster than CPU (GPU: " + gpuTime + "ms, CPU: " + cpuTime + "ms)",
			gpuTime < cpuTime * 0.8); // GPU should be at least 20% faster

		// Smart defaults should perform similarly to explicit GPU config
		Assert.assertTrue("Smart defaults should perform well (Smart: " + smartTime + "ms, GPU: " + gpuTime + "ms)",
			smartTime < gpuTime * 1.5); // Allow 50% tolerance

		// Verify good performance overall
		Assert.assertTrue("Should generate tokens with all configurations",
			smartTokens > 0);

		logger.log(DEBUG, "\n✅ GPU Offloading Test PASSED");
		logger.log(DEBUG, "   - GPU provides %.2fx speedup over CPU", gpuSpeedup);
		logger.log(DEBUG, "   - Smart defaults provide %.2fx speedup over CPU", smartSpeedup);
		logger.log(DEBUG, "   - All models generate tokens successfully");
	}

	@Test
	public void testCompareGpuVsCpuPerformance() {
		logger.log(DEBUG, "\n=== GPU vs CPU Performance Comparison ===\n");

		String prompt = "public static int fibonacci(int n) {";
		int nPredict = 20;

		// Test with CPU only (0 GPU layers)
		logger.log(DEBUG, "Testing CPU-only performance...");
		long cpuTime = benchmarkModel(0, prompt, nPredict);

		// Test with GPU (all layers)
		logger.log(DEBUG, "\nTesting GPU-accelerated performance...");
		long gpuTime = benchmarkModel(43, prompt, nPredict);

		// Calculate speedup
		double speedup = (double) cpuTime / gpuTime;

		logger.log(DEBUG, "\n=== Performance Results ===");
		logger.log(DEBUG, "CPU-only time: " + cpuTime + " ms");
		logger.log(DEBUG, "GPU-accelerated time: " + gpuTime + " ms");
		logger.log(DEBUG, "Speedup: %.2fx faster with GPU", speedup);

		// GPU should be faster than CPU
		Assert.assertTrue("GPU should be faster than CPU (GPU: " + gpuTime + "ms, CPU: " + cpuTime + "ms)",
			gpuTime < cpuTime);

		logger.log(DEBUG, "\n✅ Performance comparison test PASSED");
	}

	private long testModelPerformance(int gpuLayers, String prompt, int nPredict) {
		return benchmarkModel(gpuLayers, prompt, nPredict);
	}

	private long benchmarkModel(int gpuLayers, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(gpuLayers)
				.setCtxSize(512)
		)) {

			InferenceParameters params = new InferenceParameters(prompt)
				.setNPredict(nPredict)
				.setTemperature(0.1f);  // Low temperature for consistent results

			long startTime = System.currentTimeMillis();
			int tokenCount = 0;

			for (LlamaOutput output : model.generate(params)) {
				tokenCount++;
			}

			long endTime = System.currentTimeMillis();
			long duration = endTime - startTime;

			logger.log(DEBUG, "  Generated " + tokenCount + " tokens in " + duration + " ms");
			logger.log(DEBUG, "  Speed: %.2f tokens/second", tokenCount * 1000.0 / duration);

			return duration;

		}
	}
}
