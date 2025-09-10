package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.BeforeClass;
import org.junit.AfterClass;
import org.junit.Assert;

public class ThreadingControlTest {
	private static final System.Logger logger = System.getLogger(ThreadingControlTest.class.getName());

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");

		model = new LlamaModel(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(10)
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testOptimalConfiguration() {
		logger.log(DEBUG, "\n=== Threading Configuration Test ===");

		// Get optimal configuration for this system
		ThreadingManager.ThreadConfig optimal = ThreadingManager.getOptimalConfiguration();
		logger.log(DEBUG, "Optimal config: " + optimal);

		Assert.assertTrue("Generation threads should be positive", optimal.generationThreads() > 0);
		Assert.assertTrue("Batch threads should be positive", optimal.batchThreads() > 0);
		Assert.assertNotNull("Description should not be null", optimal.description());

		// Apply the configuration
		ThreadingManager.applyConfiguration(model, optimal);

		// Verify it was applied correctly
		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		logger.log(DEBUG, "Applied config: " + current);

		Assert.assertEquals("Generation threads should match", optimal.generationThreads(), current.generationThreads());
		Assert.assertEquals("Batch threads should match", optimal.batchThreads(), current.batchThreads());

		logger.log(DEBUG, "✅ Threading configuration test passed!");
	}

	@Test
	public void testSpecializedConfigurations() {
		logger.log(DEBUG, "\n=== Specialized Configuration Tests ===");

		// Test low latency configuration
		ThreadingManager.ThreadConfig lowLatency = ThreadingManager.forLowLatency();
		logger.log(DEBUG, "Low latency config: " + lowLatency);
		ThreadingManager.applyConfiguration(model, lowLatency);

		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("Low latency config applied", lowLatency.generationThreads(), current.generationThreads());

		// Test high throughput configuration
		ThreadingManager.ThreadConfig highThroughput = ThreadingManager.forHighThroughput();
		logger.log(DEBUG, "High throughput config: " + highThroughput);
		ThreadingManager.applyConfiguration(model, highThroughput);

		current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("High throughput config applied", highThroughput.generationThreads(), current.generationThreads());

		// Test memory efficient configuration
		ThreadingManager.ThreadConfig memoryEfficient = ThreadingManager.forMemoryEfficiency();
		logger.log(DEBUG, "Memory efficient config: " + memoryEfficient);
		ThreadingManager.applyConfiguration(model, memoryEfficient);

		current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("Memory efficient config applied", memoryEfficient.generationThreads(), current.generationThreads());

		logger.log(DEBUG, "✅ Specialized configuration test passed!");
	}

	@Test
	public void testCustomConfiguration() {
		logger.log(DEBUG, "\n=== Custom Configuration Test ===");

		// Create custom configuration
		ThreadingManager.ThreadConfig custom = ThreadingManager.createCustom(4, 6, true, "Custom test config");
		logger.log(DEBUG, "Custom config: " + custom);

		Assert.assertEquals("Custom generation threads", 4, custom.generationThreads());
		Assert.assertEquals("Custom batch threads", 6, custom.batchThreads());
		Assert.assertTrue("Custom NUMA setting", custom.numaOptimized());
		Assert.assertEquals("Custom description", "Custom test config", custom.description());

		// Apply custom configuration
		ThreadingManager.applyConfiguration(model, custom);

		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("Custom config applied correctly", 4, current.generationThreads());
		Assert.assertEquals("Custom batch config applied correctly", 6, current.batchThreads());

		logger.log(DEBUG, "✅ Custom configuration test passed!");
	}

	@Test
	public void testThreadingBoundaryConditions() {
		logger.log(DEBUG, "\n=== Threading Boundary Conditions Test ===");

		// Test minimum values (should be clamped to 1)
		ThreadingManager.ThreadConfig minimal = ThreadingManager.createCustom(0, -1);
		logger.log(DEBUG, "Minimal config (corrected): " + minimal);

		Assert.assertTrue("Generation threads should be at least 1", minimal.generationThreads() >= 1);
		Assert.assertTrue("Batch threads should be at least 1", minimal.batchThreads() >= 1);

		// Test very high values
		ThreadingManager.ThreadConfig maximal = ThreadingManager.createCustom(64, 128);
		logger.log(DEBUG, "Maximal config: " + maximal);

		Assert.assertEquals("High generation threads", 64, maximal.generationThreads());
		Assert.assertEquals("High batch threads", 128, maximal.batchThreads());

		// Apply and verify
		ThreadingManager.applyConfiguration(model, maximal);
		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("High config applied", 64, current.generationThreads());

		logger.log(DEBUG, "✅ Boundary conditions test passed!");
	}

	@Test
	public void testErrorHandling() {
		logger.log(DEBUG, "\n=== Threading Error Handling Test ===");

		try {
			ThreadingManager.applyConfiguration(null, ThreadingManager.getOptimalConfiguration());
			Assert.fail("Should throw IllegalArgumentException for null model");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly caught null model: " + e.getMessage());
		}

		try {
			ThreadingManager.applyConfiguration(model, null);
			Assert.fail("Should throw IllegalArgumentException for null config");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly caught null config: " + e.getMessage());
		}

		try {
			ThreadingManager.getCurrentConfiguration(null);
			Assert.fail("Should throw IllegalArgumentException for null model");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly caught null model in getCurrentConfiguration: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Error handling test passed!");
	}
}
