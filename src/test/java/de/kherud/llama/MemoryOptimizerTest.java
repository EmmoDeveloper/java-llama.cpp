package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;

public class MemoryOptimizerTest {
	private static final System.Logger logger = System.getLogger(MemoryOptimizerTest.class.getName());

	@Test
	public void testMemoryOptimizationIntegration() {
		logger.log(DEBUG, "\n=== Memory Optimization Integration Test ===\n");

		logger.log(DEBUG, "🔍 Testing memory-optimized model creation:");

		// Test that memory optimization is applied automatically through smart defaults
		LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				// No explicit memory settings - should be auto-optimized
		);

		logger.log(DEBUG, "\n🧪 Testing generation with memory-optimized settings:");
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

		logger.log(DEBUG, "Generated %d tokens in %d ms (%.2f tok/s)",
			tokenCount, duration, tokensPerSecond);

		model.close();

		// Verify reasonable performance and functionality
		Assert.assertTrue("Should generate tokens", tokenCount > 0);
		Assert.assertTrue("Should have reasonable performance", tokensPerSecond > 5);

		logger.log(DEBUG, "\n✅ Memory optimization integration test completed!");
	}

	@Test
	public void testMemoryEfficientConfiguration() {
		logger.log(DEBUG, "\n=== Memory Efficient Configuration Test ===\n");

		// Test memory-efficient configuration creation
		logger.log(DEBUG, "🔧 Creating memory-efficient configuration:");
		ModelParameters memoryEfficientParams = MemoryOptimizer.createMemoryEfficientConfig(
			"models/codellama-7b.Q2_K.gguf"
		);

		// Test the configuration works
		logger.log(DEBUG, "🧪 Testing memory-efficient model:");
		LlamaModel model = new LlamaModel(memoryEfficientParams);

		InferenceParameters params = new InferenceParameters("test")
			.setNPredict(5);

		int tokenCount = 0;
		for (LlamaOutput output : model.generate(params)) {
			tokenCount++;
		}

		model.close();

		logger.log(DEBUG, "Memory-efficient model generated %d tokens", tokenCount);

		Assert.assertTrue("Memory-efficient model should work", tokenCount > 0);

		logger.log(DEBUG, "\n✅ Memory efficient configuration test completed!");
	}

	@Test
	public void testMemoryRecommendations() {
		logger.log(DEBUG, "\n=== Memory Recommendations Test ===\n");

		// Test memory recommendations output
		logger.log(DEBUG, "📊 Memory recommendations:");
		MemoryOptimizer.printMemoryRecommendations();

		// Test memory health check
		logger.log(DEBUG, "\n💊 Memory health check:");
		MemoryOptimizer.checkMemoryHealth();

		logger.log(DEBUG, "\n✅ Memory recommendations test completed!");
	}
}
