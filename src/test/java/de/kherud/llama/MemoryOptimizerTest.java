package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

public class MemoryOptimizerTest {

	@Test
	public void testMemoryOptimizationIntegration() {
		System.out.println("\n=== Memory Optimization Integration Test ===\n");
		
		System.out.println("🔍 Testing memory-optimized model creation:");
		
		// Test that memory optimization is applied automatically through smart defaults
		LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				// No explicit memory settings - should be auto-optimized
		);
		
		System.out.println("\n🧪 Testing generation with memory-optimized settings:");
		InferenceParameters params = new InferenceParameters("def hello_world():")
			.setNPredict(8)
			.setTemperature(0.1f);
		
		int tokenCount = 0;
		long startTime = System.currentTimeMillis();
		
		for (LlamaOutput output : model.generate(params)) {
			tokenCount++;
		}
		
		long duration = System.currentTimeMillis() - startTime;
		double tokensPerSecond = tokenCount * 1000.0 / duration;
		
		System.out.printf("Generated %d tokens in %d ms (%.2f tok/s)\n", 
			tokenCount, duration, tokensPerSecond);
		
		model.close();
		
		// Verify reasonable performance and functionality
		Assert.assertTrue("Should generate tokens", tokenCount > 0);
		Assert.assertTrue("Should have reasonable performance", tokensPerSecond > 5);
		
		System.out.println("\n✅ Memory optimization integration test completed!");
	}
	
	@Test
	public void testMemoryEfficientConfiguration() {
		System.out.println("\n=== Memory Efficient Configuration Test ===\n");
		
		// Test memory-efficient configuration creation
		System.out.println("🔧 Creating memory-efficient configuration:");
		ModelParameters memoryEfficientParams = MemoryOptimizer.createMemoryEfficientConfig(
			"/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf"
		);
		
		// Test the configuration works
		System.out.println("🧪 Testing memory-efficient model:");
		LlamaModel model = new LlamaModel(memoryEfficientParams);
		
		InferenceParameters params = new InferenceParameters("test")
			.setNPredict(5);
		
		int tokenCount = 0;
		for (LlamaOutput output : model.generate(params)) {
			tokenCount++;
		}
		
		model.close();
		
		System.out.printf("Memory-efficient model generated %d tokens\n", tokenCount);
		
		Assert.assertTrue("Memory-efficient model should work", tokenCount > 0);
		
		System.out.println("\n✅ Memory efficient configuration test completed!");
	}
	
	@Test
	public void testMemoryRecommendations() {
		System.out.println("\n=== Memory Recommendations Test ===\n");
		
		// Test memory recommendations output
		System.out.println("📊 Memory recommendations:");
		MemoryOptimizer.printMemoryRecommendations();
		
		// Test memory health check
		System.out.println("\n💊 Memory health check:");
		MemoryOptimizer.checkMemoryHealth();
		
		System.out.println("\n✅ Memory recommendations test completed!");
	}
}