package de.kherud.llama;

import org.junit.Test;

public class DebugBasicTest {

	@Test
	public void testBasicOperations() throws Exception {
		System.out.println("=== Debug Basic Operations ===");
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);

		try (LlamaModel model = new LlamaModel(params)) {
			System.out.println("Model created");

			// Test encode
			String testPrompt = "The quick brown fox";
			System.out.println("Testing encode with: '" + testPrompt + "'");
			int[] tokens = model.encode(testPrompt);
			System.out.println("Encode result: " + (tokens != null ? tokens.length + " tokens" : "null"));
			if (tokens != null) {
				System.out.println("First few tokens: " + java.util.Arrays.toString(java.util.Arrays.copyOf(tokens, Math.min(5, tokens.length))));
			}

			// Test getModelState
			System.out.println("Testing getModelState...");
			try {
				byte[] state = model.getModelState();
				System.out.println("getModelState result: " + (state != null ? state.length + " bytes" : "null"));
			} catch (Exception e) {
				System.out.println("getModelState failed: " + e.getMessage());
			}

			// Test complete
			System.out.println("Testing complete...");
			try {
				InferenceParameters inferParams = new InferenceParameters(testPrompt).setNPredict(1);
				String result = model.complete(inferParams);
				System.out.println("Complete result: " + (result != null ? "'" + result + "'" : "null"));
			} catch (Exception e) {
				System.out.println("Complete failed: " + e.getMessage());
			}
		}
		System.out.println("=== Debug Complete ===");
	}
}
