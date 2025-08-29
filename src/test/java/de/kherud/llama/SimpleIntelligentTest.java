package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

public class SimpleIntelligentTest {

	@Test
	public void testBasicIntelligentDefaults() {
		System.out.println("\n=== Testing Basic Intelligent Defaults ===\n");
		
		System.out.println("Creating model with minimal parameters (should auto-configure GPU)...");
		
		// Create a model with minimal parameters - GPU should be auto-detected
		LlamaModel model = new LlamaModel(
			new ModelParameters()
				.setModel("/work/java/java-llama.cpp/models/codellama-7b.Q2_K.gguf")
				// No GPU layers specified - should be auto-configured
		);
		
		// Test that the model works by doing a simple generation
		System.out.println("Testing generation with auto-configured model:");
		InferenceParameters params = new InferenceParameters("print('hello')")
			.setNPredict(3)
			.setTemperature(0.1f);
		
		int tokenCount = 0;
		long startTime = System.currentTimeMillis();
		
		for (LlamaOutput output : model.generate(params)) {
			System.out.print(output.text);
			tokenCount++;
		}
		
		long duration = System.currentTimeMillis() - startTime;
		System.out.println();
		System.out.printf("Generated %d tokens in %d ms (%.2f tok/s)\n", 
			tokenCount, duration, tokenCount * 1000.0 / duration);
		
		// Clean up
		model.close();
		
		// Verify it worked
		Assert.assertTrue("Should generate at least one token", tokenCount > 0);
		Assert.assertTrue("Should complete in reasonable time", duration < 10000); // < 10 seconds
		
		System.out.println("\n✅ Basic intelligent defaults test passed!");
	}
}