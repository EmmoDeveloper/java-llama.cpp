package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class SimpleStateTest {
	private static final System.Logger logger = System.getLogger(SimpleStateTest.class.getName());

	@Test
	public void testBasicStateSize() {
		logger.log(DEBUG, "=== Starting Simple State Test ===");
		try {
			ModelParameters params = new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(99);
			logger.log(DEBUG, "Creating model...");

			try (LlamaModel model = new LlamaModel(params)) {
				logger.log(DEBUG, "Model created successfully");

				// Try encoding some text first
				logger.log(DEBUG, "Encoding text...");
				int[] tokens = model.encode("The quick brown fox");
				logger.log(DEBUG, "Tokens encoded: " + (tokens != null ? tokens.length : "null"));

				if (tokens != null && tokens.length > 0) {
					// Now try to get state size
					logger.log(DEBUG, "Getting state size...");
					long stateSize = model.getModelStateSize();
					logger.log(DEBUG, "State size: " + stateSize);
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
		logger.log(DEBUG, "=== Test Complete ===");
	}
}
