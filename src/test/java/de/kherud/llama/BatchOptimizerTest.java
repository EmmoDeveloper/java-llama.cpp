package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

public class BatchOptimizerTest {

	@Test
	public void testBatchOptimizationLogic() {
		System.out.println("\n=== Batch Optimization Logic Test ===\n");
		
		// Test GPU optimization for different context sizes
		System.out.println("🔍 Testing GPU batch optimization:");
		
		BatchOptimizer.BatchConfiguration small = BatchOptimizer.optimizeForGpu(512, true);
		System.out.println("Context 512: " + small);
		Assert.assertEquals("Small context should use 256 batch", 256, small.batchSize);
		
		BatchOptimizer.BatchConfiguration medium = BatchOptimizer.optimizeForGpu(1024, true);
		System.out.println("Context 1024: " + medium);
		Assert.assertEquals("Medium context should use 512 batch", 512, medium.batchSize);
		
		BatchOptimizer.BatchConfiguration large = BatchOptimizer.optimizeForGpu(2048, true);
		System.out.println("Context 2048: " + large);
		Assert.assertEquals("Large context should use 1024 batch", 1024, large.batchSize);
		
		// Test CPU optimization
		System.out.println("\n🔍 Testing CPU batch optimization:");
		BatchOptimizer.BatchConfiguration cpu = BatchOptimizer.optimizeForGpu(1024, false);
		System.out.println("CPU config: " + cpu);
		Assert.assertTrue("CPU should use smaller batches", cpu.batchSize <= 256);
		
		System.out.println("\n✅ Batch optimization logic test passed!");
	}
	
	@Test
	public void testSmartDefaultsWithBatchOptimization() {
		System.out.println("\n=== Smart Defaults with Batch Optimization ===\n");
		
		// Test that smart defaults now include batch optimization
		System.out.println("🚀 Creating model with smart defaults:");
		
		LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				// No explicit batch config - should be auto-optimized
		);
		
		// Test that model works with optimized batch settings
		System.out.println("\n🧪 Testing generation with optimized batch settings:");
		InferenceParameters params = new InferenceParameters("print('Hello')")
			.setNPredict(5)
			.setTemperature(0.1f);
		
		long startTime = System.currentTimeMillis();
		int tokenCount = 0;
		
		for (LlamaOutput output : model.generate(params)) {
			tokenCount++;
		}
		
		long duration = System.currentTimeMillis() - startTime;
		double tokensPerSecond = tokenCount * 1000.0 / duration;
		
		System.out.printf("Generated %d tokens in %d ms (%.2f tok/s)\n", 
			tokenCount, duration, tokensPerSecond);
		
		model.close();
		
		// Verify reasonable performance
		Assert.assertTrue("Should generate tokens", tokenCount > 0);
		Assert.assertTrue("Should have reasonable performance", tokensPerSecond > 5);
		
		System.out.println("\n✅ Smart defaults with batch optimization work correctly!");
	}
	
	@Test
	public void testUserConfigurationPreservation() {
		System.out.println("\n=== User Configuration Preservation Test ===\n");
		
		// Test that explicit user batch settings are preserved
		ModelParameters userParams = new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			.setBatchSize(128)  // Explicit user choice
			.setUbatchSize(64)  // Explicit user choice
			.setCtxSize(1024);
		
		// Apply smart defaults
		ModelParameters processed = SmartDefaults.apply(userParams);
		
		// Check that user settings were preserved
		String batchSize = processed.parameters.get("--batch-size");
		String ubatchSize = processed.parameters.get("--ubatch-size");
		
		Assert.assertEquals("User batch size should be preserved", "128", batchSize);
		Assert.assertEquals("User ubatch size should be preserved", "64", ubatchSize);
		
		System.out.println("✅ User-specified batch sizes: " + batchSize + "/" + ubatchSize);
		System.out.println("✅ User configuration preservation test passed!");
	}
	
	@Test
	public void testBatchValidation() {
		System.out.println("\n=== Batch Configuration Validation Test ===\n");
		
		// Test validation of invalid configurations
		System.out.println("🔍 Testing validation warnings:");
		
		// Test ubatch > batch warning
		ModelParameters invalidParams = new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			.setBatchSize(256)
			.setUbatchSize(512);  // Invalid: ubatch > batch
		
		System.out.println("Validating ubatch (512) > batch (256):");
		BatchOptimizer.validateBatchConfiguration(invalidParams);
		
		// Test very small batch size
		ModelParameters smallParams = new ModelParameters()
			.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
			.setBatchSize(32);  // Very small
		
		System.out.println("\nValidating very small batch size (32):");
		BatchOptimizer.validateBatchConfiguration(smallParams);
		
		System.out.println("\n✅ Batch validation test completed!");
	}
}