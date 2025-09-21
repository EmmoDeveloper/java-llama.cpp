package de.kherud.llama;

import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class DebugSequenceTest {
	private static final System.Logger logger = System.getLogger(DebugSequenceTest.class.getName());

	@Test
	public void testSequenceDebug() {
		logger.log(DEBUG, "=== Debug Sequence State Test ===");
		try {
			ModelParameters params = new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(99);
			logger.log(DEBUG, "Creating model...");

			try (LlamaModel model = new LlamaModel(params)) {
				logger.log(DEBUG, "Model created successfully");

				// Try getting sequence state size for unused sequence
				logger.log(DEBUG, "Getting sequence state size for unused sequence 0...");
				try {
					long seqStateSize = model.getSequenceStateSize(0);
					logger.log(DEBUG, "SUCCESS: Unused sequence 0 state size: " + seqStateSize);
				} catch (Exception e) {
					logger.log(DEBUG, "FAILED: " + e.getMessage());
					e.printStackTrace();
				}

				// Try regular state size for comparison
				logger.log(DEBUG, "Getting regular state size...");
				try {
					// First encode something to create state
					int[] tokens = model.encode("test");
					logger.log(DEBUG, "Encoded " + tokens.length + " tokens");

					long stateSize = model.getModelStateSize();
					logger.log(DEBUG, "SUCCESS: Model state size: " + stateSize);
				} catch (Exception e) {
					logger.log(DEBUG, "FAILED: " + e.getMessage());
					e.printStackTrace();
				}

				// Now try sequence state after using the model
				logger.log(DEBUG, "Getting sequence state size after model usage...");
				try {
					long seqStateSize = model.getSequenceStateSize(0);
					logger.log(DEBUG, "SUCCESS: Used sequence 0 state size: " + seqStateSize);
				} catch (Exception e) {
					logger.log(DEBUG, "FAILED: " + e.getMessage());
					e.printStackTrace();
				}
			}
		} catch (Exception e) {
			System.err.println("Exception occurred: " + e.getMessage());
			e.printStackTrace();
			throw e;
		}
		logger.log(DEBUG, "=== Debug Complete ===");
	}
}
