package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

public class SimpleStateTest {

	@Test
	public void testBasicStateSize() throws Exception {
		System.out.println("=== Starting Simple State Test ===");
		try {
			ModelParameters params = new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(99);
			System.out.println("Creating model...");
			
			try (LlamaModel model = new LlamaModel(params)) {
				System.out.println("Model created successfully");
				
				// Try encoding some text first
				System.out.println("Encoding text...");
				int[] tokens = model.encode("The quick brown fox");
				System.out.println("Tokens encoded: " + (tokens != null ? tokens.length : "null"));
				
				if (tokens != null && tokens.length > 0) {
					// Now try to get state size
					System.out.println("Getting state size...");
					long stateSize = model.getModelStateSize();
					System.out.println("State size: " + stateSize);
					Assert.assertTrue("State size should be positive", stateSize > 0);
				} else {
					System.err.println("Encoding failed - tokens is null or empty");
				}
			}
		} catch (Exception e) {
			System.err.println("Exception occurred: " + e.getMessage());
			e.printStackTrace();
			throw e;
		}
		System.out.println("=== Test Complete ===");
	}
}