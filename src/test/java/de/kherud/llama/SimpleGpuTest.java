package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

public class SimpleGpuTest {

	@Test
	public void testGpuOffloading() {
		System.out.println("\n=== Simple GPU Offloading Test ===\n");
		
		// Test 1: Model with 0 GPU layers (CPU only)
		System.out.println("1. Loading model with 0 GPU layers (CPU only)...");
		LlamaModel cpuModel = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(0)  // CPU only
				.setCtxSize(512)
		);
		
		// Test 2: Model with 43 GPU layers
		System.out.println("\n2. Loading model with 43 GPU layers...");
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
		
		System.out.println("\n3. Testing CPU model generation:");
		long cpuStart = System.currentTimeMillis();
		int cpuTokens = 0;
		for (LlamaOutput output : cpuModel.generate(params)) {
			System.out.print(output.text);
			cpuTokens++;
		}
		long cpuTime = System.currentTimeMillis() - cpuStart;
		System.out.println("\n   CPU: " + cpuTokens + " tokens in " + cpuTime + " ms");
		
		System.out.println("\n4. Testing GPU model generation:");
		long gpuStart = System.currentTimeMillis();
		int gpuTokens = 0;
		for (LlamaOutput output : gpuModel.generate(params)) {
			System.out.print(output.text);
			gpuTokens++;
		}
		long gpuTime = System.currentTimeMillis() - gpuStart;
		System.out.println("\n   GPU: " + gpuTokens + " tokens in " + gpuTime + " ms");
		
		// Compare performance
		System.out.println("\n=== Results ===");
		System.out.println("CPU time: " + cpuTime + " ms");
		System.out.println("GPU time: " + gpuTime + " ms");
		if (gpuTime < cpuTime) {
			double speedup = (double) cpuTime / gpuTime;
			System.out.printf("GPU is %.2fx faster than CPU ✅\n", speedup);
		} else {
			System.out.println("GPU is not faster - may need investigation");
		}
		
		// Clean up
		cpuModel.close();
		gpuModel.close();
		
		// Assert that both models generated tokens
		Assert.assertTrue("CPU model should generate tokens", cpuTokens > 0);
		Assert.assertTrue("GPU model should generate tokens", gpuTokens > 0);
		
		System.out.println("\n✅ Test completed successfully!");
	}
}