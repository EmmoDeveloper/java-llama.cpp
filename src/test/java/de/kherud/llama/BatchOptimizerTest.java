package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;

public class BatchOptimizerTest {
	private static final System.Logger logger = System.getLogger(BatchOptimizerTest.class.getName());

	@Test
	public void testBatchOptimizationLogic() {
		logger.log(DEBUG, "\n=== Batch Optimization Logic Test ===\n");

		// Test GPU optimization for different context sizes
		logger.log(DEBUG, "🔍 Testing GPU batch optimization:");

		BatchOptimizer.BatchConfiguration small = BatchOptimizer.optimizeForGpu(512, true);
		logger.log(DEBUG, "Context 512: " + small);
		Assert.assertEquals("Small context should use 256 batch", 256, small.batchSize);

		BatchOptimizer.BatchConfiguration medium = BatchOptimizer.optimizeForGpu(1024, true);
		logger.log(DEBUG, "Context 1024: " + medium);
		Assert.assertEquals("Medium context should use 512 batch", 512, medium.batchSize);

		BatchOptimizer.BatchConfiguration large = BatchOptimizer.optimizeForGpu(2048, true);
		logger.log(DEBUG, "Context 2048: " + large);
		Assert.assertEquals("Large context should use 1024 batch", 1024, large.batchSize);

		// Test CPU optimization
		logger.log(DEBUG, "\n🔍 Testing CPU batch optimization:");
		BatchOptimizer.BatchConfiguration cpu = BatchOptimizer.optimizeForGpu(1024, false);
		logger.log(DEBUG, "CPU config: " + cpu);
		Assert.assertTrue("CPU should use smaller batches", cpu.batchSize <= 256);

		logger.log(DEBUG, "\n✅ Batch optimization logic test passed!");
	}

	@Test
	public void testSmartDefaultsWithBatchOptimization() {
		logger.log(DEBUG, "\n=== Smart Defaults with Batch Optimization ===\n");

		// Test that smart defaults now include batch optimization
		logger.log(DEBUG, "🚀 Creating model with smart defaults:");

		LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				// No explicit batch config - should be auto-optimized
		);

		// Test that model works with optimized batch settings
		logger.log(DEBUG, "\n🧪 Testing generation with optimized batch settings:");
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

		logger.log(DEBUG, "Generated %d tokens in %d ms (%.2f tok/s)",
			tokenCount, duration, tokensPerSecond);

		model.close();

		// Verify reasonable performance
		Assert.assertTrue("Should generate tokens", tokenCount > 0);
		Assert.assertTrue("Should have reasonable performance", tokensPerSecond > 5);

		logger.log(DEBUG, "\n✅ Smart defaults with batch optimization work correctly!");
	}

	@Test
	public void testUserConfigurationPreservation() {
		logger.log(DEBUG, "\n=== User Configuration Preservation Test ===\n");

		// Test that explicit user batch settings are preserved
		ModelParameters userParams = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
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

		logger.log(DEBUG, "✅ User-specified batch sizes: " + batchSize + "/" + ubatchSize);
		logger.log(DEBUG, "✅ User configuration preservation test passed!");
	}

	@Test
	public void testBatchValidation() {
		logger.log(DEBUG, "\n=== Batch Configuration Validation Test ===\n");

		// Test validation of invalid configurations
		logger.log(DEBUG, "🔍 Testing validation warnings:");

		// Test ubatch > batch warning
		ModelParameters invalidParams = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setBatchSize(256)
			.setUbatchSize(512);  // Invalid: ubatch > batch

		logger.log(DEBUG, "Validating ubatch (512) > batch (256):");
		BatchOptimizer.validateBatchConfiguration(invalidParams);

		// Test very small batch size
		ModelParameters smallParams = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setBatchSize(32);  // Very small

		logger.log(DEBUG, "\nValidating very small batch size (32):");
		BatchOptimizer.validateBatchConfiguration(smallParams);

		logger.log(DEBUG, "\n✅ Batch validation test completed!");
	}
}
