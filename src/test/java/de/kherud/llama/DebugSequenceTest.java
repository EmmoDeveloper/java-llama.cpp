package de.kherud.llama;

import org.junit.Test;

public class DebugSequenceTest {

	@Test
	public void testSequenceDebug() {
		System.out.println("=== Debug Sequence State Test ===");
		try {
			ModelParameters params = new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(99);
			System.out.println("Creating model...");

			try (LlamaModel model = new LlamaModel(params)) {
				System.out.println("Model created successfully");

				// Try getting sequence state size for unused sequence
				System.out.println("Getting sequence state size for unused sequence 0...");
				try {
					long seqStateSize = model.getSequenceStateSize(0);
					System.out.println("SUCCESS: Unused sequence 0 state size: " + seqStateSize);
				} catch (Exception e) {
					System.out.println("FAILED: " + e.getMessage());
					e.printStackTrace();
				}

				// Try regular state size for comparison
				System.out.println("Getting regular state size...");
				try {
					// First encode something to create state
					int[] tokens = model.encode("test");
					System.out.println("Encoded " + tokens.length + " tokens");

					long stateSize = model.getModelStateSize();
					System.out.println("SUCCESS: Model state size: " + stateSize);
				} catch (Exception e) {
					System.out.println("FAILED: " + e.getMessage());
					e.printStackTrace();
				}

				// Now try sequence state after using the model
				System.out.println("Getting sequence state size after model usage...");
				try {
					long seqStateSize = model.getSequenceStateSize(0);
					System.out.println("SUCCESS: Used sequence 0 state size: " + seqStateSize);
				} catch (Exception e) {
					System.out.println("FAILED: " + e.getMessage());
					e.printStackTrace();
				}
			}
		} catch (Exception e) {
			System.err.println("Exception occurred: " + e.getMessage());
			e.printStackTrace();
			throw e;
		}
		System.out.println("=== Debug Complete ===");
	}
}
