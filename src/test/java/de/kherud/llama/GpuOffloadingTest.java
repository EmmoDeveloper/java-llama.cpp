package de.kherud.llama;

import de.kherud.llama.args.LogFormat;
import org.junit.Test;
import org.junit.Assert;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class GpuOffloadingTest {

	@Test
	public void testGpuLayerOffloading() {
		System.out.println("\n=== GPU Layer Offloading Test ===\n");
		
		// Track GPU offloading through log messages
		AtomicInteger gpuLayersOffloaded = new AtomicInteger(0);
		AtomicBoolean modelLoaded = new AtomicBoolean(false);
		
		// Set up logger to capture GPU offloading information
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> {
			// Print all messages for visibility
			System.out.println("[" + level + "] " + msg);
			
			// Check for GPU layer offloading messages
			if (msg.contains("offloaded") && msg.contains("layers to GPU")) {
				// Parse message like "offloaded 33/33 layers to GPU"
				String[] parts = msg.split(" ");
				for (int i = 0; i < parts.length - 1; i++) {
					if (parts[i].equals("offloaded")) {
						String layerInfo = parts[i + 1];
						if (layerInfo.contains("/")) {
							String offloaded = layerInfo.split("/")[0];
							gpuLayersOffloaded.set(Integer.parseInt(offloaded));
							System.out.println("\n✓ GPU OFFLOADING DETECTED: " + offloaded + " layers\n");
						}
						break;
					}
				}
			}
			
			// Check for CUDA device assignment
			if (msg.contains("assigned to device CUDA")) {
				System.out.println("✓ Layer assigned to CUDA device");
			}
			
			// Check for successful model loading
			if (msg.contains("model loaded successfully") || msg.contains("llama_model_load:") || msg.contains("model size")) {
				modelLoaded.set(true);
			}
		});
		
		LlamaModel model = null;
		try {
			System.out.println("Loading model with GPU layers...");
			
			// Create model with explicit GPU layer configuration
			model = new LlamaModel(
				new ModelParameters()
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(43)  // Request 43 layers to be offloaded to GPU
					.setCtxSize(512)
					.setVerbose()  // Enable verbose logging to see GPU offloading details
			);
			
			// Give the logger a moment to process messages
			Thread.sleep(100);
			
			System.out.println("\n=== Test Results ===");
			System.out.println("GPU Layers Requested: 43");
			System.out.println("GPU Layers Offloaded: " + gpuLayersOffloaded.get());
			
			// Test that GPU offloading actually happened
			Assert.assertTrue("Model should be loaded", modelLoaded.get() || model != null);
			Assert.assertTrue("At least some layers should be offloaded to GPU (found: " + gpuLayersOffloaded.get() + ")", 
				gpuLayersOffloaded.get() > 0);
			
			// Run a simple generation to verify the model works with GPU
			System.out.println("\nTesting generation with GPU-accelerated model...");
			InferenceParameters params = new InferenceParameters("Hello")
				.setNPredict(5);
			
			int tokenCount = 0;
			long startTime = System.currentTimeMillis();
			
			for (LlamaOutput output : model.generate(params)) {
				tokenCount++;
				System.out.print(output.text);
			}
			
			long endTime = System.currentTimeMillis();
			double tokensPerSecond = tokenCount * 1000.0 / (endTime - startTime);
			
			System.out.println("\n\nGeneration completed:");
			System.out.println("- Tokens generated: " + tokenCount);
			System.out.println("- Time taken: " + (endTime - startTime) + " ms");
			System.out.printf("- Speed: %.2f tokens/second\n", tokensPerSecond);
			
			Assert.assertTrue("Should generate at least one token", tokenCount > 0);
			
			System.out.println("\n✅ GPU Offloading Test PASSED");
			System.out.println("   - " + gpuLayersOffloaded.get() + " layers successfully offloaded to GPU");
			System.out.println("   - Model is generating tokens using GPU acceleration");
			
		} catch (Exception e) {
			System.err.println("Test failed with exception: " + e.getMessage());
			e.printStackTrace();
			Assert.fail("GPU offloading test failed: " + e.getMessage());
		} finally {
			if (model != null) {
				model.close();
			}
			// Reset logger
			LlamaModel.setLogger(null, null);
		}
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
	
	private long benchmarkModel(int gpuLayers, String prompt, int nPredict) {
		LlamaModel model = null;
		try {
			model = new LlamaModel(
				new ModelParameters()
					.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(gpuLayers)
					.setCtxSize(512)
			);
			
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
			
		} finally {
			if (model != null) {
				model.close();
			}
		}
	}
}