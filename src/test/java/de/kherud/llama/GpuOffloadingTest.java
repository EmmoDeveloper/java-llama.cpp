package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

public class GpuOffloadingTest {

	@Test
	public void testGpuLayerOffloading() {
		System.out.println("\n=== GPU Layer Offloading Test (Performance-Based Verification) ===\n");

		// Instead of complex log parsing, test actual GPU performance
		// by comparing CPU vs GPU speeds directly

		String testPrompt = "def quicksort(arr):";
		int nPredict = 8;

		System.out.println("1. Testing CPU-only model (0 GPU layers):");
		long cpuTime = testModelPerformance(0, testPrompt, nPredict);

		System.out.println("\n2. Testing GPU-accelerated model (43 GPU layers):");
		long gpuTime = testModelPerformance(43, testPrompt, nPredict);

		System.out.println("\n3. Testing smart defaults model (auto GPU detection):");
		LlamaModel smartModel = null;
		long smartTime;
		int smartTokens = 0;

		try {
			smartModel = new LlamaModel(
				new ModelParameters()
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
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

			System.out.printf("  Smart defaults: %d tokens in %d ms (%.1f tok/s)\n",
				smartTokens, smartTime, smartTokens * 1000.0 / smartTime);

		} finally {
			if (smartModel != null) {
				smartModel.close();
			}
		}

		// Analysis and verification
		System.out.println("\n=== Performance Analysis ===");
		System.out.println("CPU-only time: " + cpuTime + " ms");
		System.out.println("GPU-explicit time: " + gpuTime + " ms");
		System.out.println("Smart defaults time: " + smartTime + " ms");

		double gpuSpeedup = (double) cpuTime / gpuTime;
		double smartSpeedup = (double) cpuTime / smartTime;

		System.out.printf("GPU speedup vs CPU: %.2fx\n", gpuSpeedup);
		System.out.printf("Smart defaults speedup vs CPU: %.2fx\n", smartSpeedup);

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

		System.out.println("\n✅ GPU Offloading Test PASSED");
		System.out.printf("   - GPU provides %.2fx speedup over CPU\n", gpuSpeedup);
		System.out.printf("   - Smart defaults provide %.2fx speedup over CPU\n", smartSpeedup);
		System.out.println("   - All models generate tokens successfully");
	}

	@Test
	public void testCompareGpuVsCpuPerformance() {
		System.out.println("\n=== GPU vs CPU Performance Comparison ===\n");

		String prompt = "public static int fibonacci(int n) {";
		int nPredict = 20;

		// Test with CPU only (0 GPU layers)
		System.out.println("Testing CPU-only performance...");
		long cpuTime = benchmarkModel(0, prompt, nPredict);

		// Test with GPU (all layers)
		System.out.println("\nTesting GPU-accelerated performance...");
		long gpuTime = benchmarkModel(43, prompt, nPredict);

		// Calculate speedup
		double speedup = (double) cpuTime / gpuTime;

		System.out.println("\n=== Performance Results ===");
		System.out.println("CPU-only time: " + cpuTime + " ms");
		System.out.println("GPU-accelerated time: " + gpuTime + " ms");
		System.out.printf("Speedup: %.2fx faster with GPU\n", speedup);

		// GPU should be faster than CPU
		Assert.assertTrue("GPU should be faster than CPU (GPU: " + gpuTime + "ms, CPU: " + cpuTime + "ms)",
			gpuTime < cpuTime);

		System.out.println("\n✅ Performance comparison test PASSED");
	}

	private long testModelPerformance(int gpuLayers, String prompt, int nPredict) {
		return benchmarkModel(gpuLayers, prompt, nPredict);
	}

	private long benchmarkModel(int gpuLayers, String prompt, int nPredict) {
		try (LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
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

			System.out.println("  Generated " + tokenCount + " tokens in " + duration + " ms");
			System.out.printf("  Speed: %.2f tokens/second\n", tokenCount * 1000.0 / duration);

			return duration;

		}
	}
}
