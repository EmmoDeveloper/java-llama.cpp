package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;

public class SimpleGpuTest {
	private static final System.Logger logger = System.getLogger(SimpleGpuTest.class.getName());

	@Test
	public void testGpuOffloading() {
		logger.log(DEBUG, "\n=== Simple GPU Offloading Test ===\n");

		// Test 1: Model with 0 GPU layers (CPU only)
		logger.log(DEBUG, "1. Loading model with 0 GPU layers (CPU only)...");
		LlamaModel cpuModel = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(0)  // CPU only
				.setCtxSize(512)
		);

		// Test 2: Model with 43 GPU layers
		logger.log(DEBUG, "\n2. Loading model with 43 GPU layers...");
		LlamaModel gpuModel = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)  // GPU offloading
				.setCtxSize(512)
		);

		// Test generation with both models
		String prompt = "Hello";
		InferenceParameters params = new InferenceParameters(prompt)
			.setNPredict(5)
			.setTemperature(0.1f);

		logger.log(DEBUG, "\n3. Testing CPU model generation:");
		long cpuStart = System.currentTimeMillis();
		int cpuTokens = 0;
		for (LlamaOutput output : cpuModel.generate(params)) {
			logger.log(DEBUG, output.text);
			cpuTokens++;
		}
		long cpuTime = System.currentTimeMillis() - cpuStart;
		logger.log(DEBUG, "\n   CPU: " + cpuTokens + " tokens in " + cpuTime + " ms");

		logger.log(DEBUG, "\n4. Testing GPU model generation:");
		long gpuStart = System.currentTimeMillis();
		int gpuTokens = 0;
		for (LlamaOutput output : gpuModel.generate(params)) {
			logger.log(DEBUG, output.text);
			gpuTokens++;
		}
		long gpuTime = System.currentTimeMillis() - gpuStart;
		logger.log(DEBUG, "\n   GPU: " + gpuTokens + " tokens in " + gpuTime + " ms");

		// Compare performance
		logger.log(DEBUG, "\n=== Results ===");
		logger.log(DEBUG, "CPU time: " + cpuTime + " ms");
		logger.log(DEBUG, "GPU time: " + gpuTime + " ms");
		if (gpuTime < cpuTime) {
			double speedup = (double) cpuTime / gpuTime;
			logger.log(DEBUG, "GPU is %.2fx faster than CPU ✅", speedup);
		} else {
			logger.log(DEBUG, "GPU is not faster - may need investigation");
		}

		// Clean up
		cpuModel.close();
		gpuModel.close();

		// Assert that both models generated tokens
		Assert.assertTrue("CPU model should generate tokens", cpuTokens > 0);
		Assert.assertTrue("GPU model should generate tokens", gpuTokens > 0);

		logger.log(DEBUG, "\n✅ Test completed successfully!");
	}
}
