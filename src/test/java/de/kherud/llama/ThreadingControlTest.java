package de.kherud.llama;

import org.junit.Test;
import org.junit.BeforeClass;
import org.junit.AfterClass;
import org.junit.Assert;

public class ThreadingControlTest {
	
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
		System.out.println("\n=== Threading Configuration Test ===");
		
		// Get optimal configuration for this system
		ThreadingManager.ThreadConfig optimal = ThreadingManager.getOptimalConfiguration();
		System.out.println("Optimal config: " + optimal);
		
		Assert.assertTrue("Generation threads should be positive", optimal.getGenerationThreads() > 0);
		Assert.assertTrue("Batch threads should be positive", optimal.getBatchThreads() > 0);
		Assert.assertNotNull("Description should not be null", optimal.getDescription());
		
		// Apply the configuration
		ThreadingManager.applyConfiguration(model, optimal);
		
		// Verify it was applied correctly
		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		System.out.println("Applied config: " + current);
		
		Assert.assertEquals("Generation threads should match", optimal.getGenerationThreads(), current.getGenerationThreads());
		Assert.assertEquals("Batch threads should match", optimal.getBatchThreads(), current.getBatchThreads());
		
		System.out.println("✅ Threading configuration test passed!");
	}
	
	@Test
	public void testSpecializedConfigurations() {
		System.out.println("\n=== Specialized Configuration Tests ===");
		
		// Test low latency configuration
		ThreadingManager.ThreadConfig lowLatency = ThreadingManager.forLowLatency();
		System.out.println("Low latency config: " + lowLatency);
		ThreadingManager.applyConfiguration(model, lowLatency);
		
		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("Low latency config applied", lowLatency.getGenerationThreads(), current.getGenerationThreads());
		
		// Test high throughput configuration
		ThreadingManager.ThreadConfig highThroughput = ThreadingManager.forHighThroughput();
		System.out.println("High throughput config: " + highThroughput);
		ThreadingManager.applyConfiguration(model, highThroughput);
		
		current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("High throughput config applied", highThroughput.getGenerationThreads(), current.getGenerationThreads());
		
		// Test memory efficient configuration
		ThreadingManager.ThreadConfig memoryEfficient = ThreadingManager.forMemoryEfficiency();
		System.out.println("Memory efficient config: " + memoryEfficient);
		ThreadingManager.applyConfiguration(model, memoryEfficient);
		
		current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("Memory efficient config applied", memoryEfficient.getGenerationThreads(), current.getGenerationThreads());
		
		System.out.println("✅ Specialized configuration test passed!");
	}
	
	@Test
	public void testCustomConfiguration() {
		System.out.println("\n=== Custom Configuration Test ===");
		
		// Create custom configuration
		ThreadingManager.ThreadConfig custom = ThreadingManager.createCustom(4, 6, true, "Custom test config");
		System.out.println("Custom config: " + custom);
		
		Assert.assertEquals("Custom generation threads", 4, custom.getGenerationThreads());
		Assert.assertEquals("Custom batch threads", 6, custom.getBatchThreads());
		Assert.assertTrue("Custom NUMA setting", custom.isNumaOptimized());
		Assert.assertEquals("Custom description", "Custom test config", custom.getDescription());
		
		// Apply custom configuration
		ThreadingManager.applyConfiguration(model, custom);
		
		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("Custom config applied correctly", 4, current.getGenerationThreads());
		Assert.assertEquals("Custom batch config applied correctly", 6, current.getBatchThreads());
		
		System.out.println("✅ Custom configuration test passed!");
	}
	
	@Test
	public void testThreadingBoundaryConditions() {
		System.out.println("\n=== Threading Boundary Conditions Test ===");
		
		// Test minimum values (should be clamped to 1)
		ThreadingManager.ThreadConfig minimal = ThreadingManager.createCustom(0, -1);
		System.out.println("Minimal config (corrected): " + minimal);
		
		Assert.assertTrue("Generation threads should be at least 1", minimal.getGenerationThreads() >= 1);
		Assert.assertTrue("Batch threads should be at least 1", minimal.getBatchThreads() >= 1);
		
		// Test very high values
		ThreadingManager.ThreadConfig maximal = ThreadingManager.createCustom(64, 128);
		System.out.println("Maximal config: " + maximal);
		
		Assert.assertEquals("High generation threads", 64, maximal.getGenerationThreads());
		Assert.assertEquals("High batch threads", 128, maximal.getBatchThreads());
		
		// Apply and verify
		ThreadingManager.applyConfiguration(model, maximal);
		ThreadingManager.ThreadConfig current = ThreadingManager.getCurrentConfiguration(model);
		Assert.assertEquals("High config applied", 64, current.getGenerationThreads());
		
		System.out.println("✅ Boundary conditions test passed!");
	}
	
	@Test
	public void testErrorHandling() {
		System.out.println("\n=== Threading Error Handling Test ===");
		
		try {
			ThreadingManager.applyConfiguration(null, ThreadingManager.getOptimalConfiguration());
			Assert.fail("Should throw IllegalArgumentException for null model");
		} catch (IllegalArgumentException e) {
			System.out.println("✅ Correctly caught null model: " + e.getMessage());
		}
		
		try {
			ThreadingManager.applyConfiguration(model, null);
			Assert.fail("Should throw IllegalArgumentException for null config");
		} catch (IllegalArgumentException e) {
			System.out.println("✅ Correctly caught null config: " + e.getMessage());
		}
		
		try {
			ThreadingManager.getCurrentConfiguration(null);
			Assert.fail("Should throw IllegalArgumentException for null model");
		} catch (IllegalArgumentException e) {
			System.out.println("✅ Correctly caught null model in getCurrentConfiguration: " + e.getMessage());
		}
		
		System.out.println("✅ Error handling test passed!");
	}
}