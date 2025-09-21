package de.kherud.llama;

import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class DebugBasicTest {
	private static final System.Logger logger = System.getLogger(DebugBasicTest.class.getName());

	@Test
	public void testBasicOperations() {
		logger.log(DEBUG, "=== Debug Basic Operations ===");
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);

		try (LlamaModel model = new LlamaModel(params)) {
			logger.log(DEBUG, "Model created");

			// Test encode
			String testPrompt = "The quick brown fox";
			logger.log(DEBUG, "Testing encode with: '" + testPrompt + "'");
			int[] tokens = model.encode(testPrompt);
			logger.log(DEBUG, "Encode result: " + (tokens != null ? tokens.length + " tokens" : "null"));
			if (tokens != null) {
				logger.log(DEBUG, "First few tokens: " + java.util.Arrays.toString(java.util.Arrays.copyOf(tokens, Math.min(5, tokens.length))));
			}

			// Test getModelState
			logger.log(DEBUG, "Testing getModelState...");
			try {
				byte[] state = model.getModelState();
				logger.log(DEBUG, "getModelState result: " + (state != null ? state.length + " bytes" : "null"));
			} catch (Exception e) {
				logger.log(DEBUG, "getModelState failed: " + e.getMessage());
			}

			// Test complete
			logger.log(DEBUG, "Testing complete...");
			try {
				InferenceParameters inferParams = new InferenceParameters(testPrompt).setNPredict(1);
				String result = model.complete(inferParams);
				logger.log(DEBUG, "Complete result: " + (result != null ? "'" + result + "'" : "null"));
			} catch (Exception e) {
				logger.log(DEBUG, "Complete failed: " + e.getMessage());
			}
		}
		logger.log(DEBUG, "=== Debug Complete ===");
	}
}
