package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

public class StatePersistenceTest {
	private static final System.Logger logger = System.getLogger(StatePersistenceTest.class.getName());

	private static final String TEST_PROMPT = "The quick brown fox";

	private LlamaModel createModel() {
		// Use the same model configuration as other tests
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);  // Use GPU acceleration
		return new LlamaModel(params);
	}

	@Test
	public void testGetStateSize() {
		try (LlamaModel model = createModel()) {
			// Generate some context first by encoding text
			model.encode(TEST_PROMPT);

			// Get state size - should be positive
			long stateSize = model.getModelStateSize();
			Assert.assertTrue("State size should be positive", stateSize > 0);

			logger.log(DEBUG, "Model state size: " + stateSize + " bytes");
		}
	}

	@Test
	public void testGetAndSetStateData() {
		try (LlamaModel model = createModel()) {
			// Generate some context
			int[] tokens = model.encode(TEST_PROMPT);
			Assert.assertNotNull("Tokens should not be null", tokens);
			Assert.assertTrue("Should have some tokens", tokens.length > 0);

			// Get current state
			byte[] stateData = model.getModelState();
			Assert.assertNotNull("State data should not be null", stateData);
			Assert.assertTrue("State data should not be empty", stateData.length > 0);

			logger.log(DEBUG, "Captured state data: " + stateData.length + " bytes");

			// Create a new model instance to restore state to
			ModelParameters params = new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(99);
			try (LlamaModel newModel = new LlamaModel(params)) {
				// Restore state to new model
				long loaded = newModel.setModelState(stateData);
				Assert.assertTrue("Should have loaded some bytes", loaded > 0);

				logger.log(DEBUG, "Loaded " + loaded + " bytes of state data");
			}
		}
	}

	@Test
	public void testSaveAndLoadStateFile() throws Exception {
		try (LlamaModel model = createModel()) {
			// Generate some context
			int[] originalTokens = model.encode(TEST_PROMPT);
			Assert.assertNotNull("Original tokens should not be null", originalTokens);

			// Create temporary file for state
			Path tempFile = Files.createTempFile("llama_state_test_", ".bin");
			String filePath = tempFile.toString();

			try {
				// Save current state with tokens
				model.saveState(filePath, originalTokens);

				// Verify file was created and has content
				File stateFile = new File(filePath);
				Assert.assertTrue("State file should exist", stateFile.exists());
				Assert.assertTrue("State file should not be empty", stateFile.length() > 0);

				logger.log(DEBUG, "Saved state file: " + stateFile.length() + " bytes");

				// Create new model and load state
				ModelParameters params = new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(99);
				try (LlamaModel newModel = new LlamaModel(params)) {
					int[] loadedTokens = newModel.loadState(filePath);

					Assert.assertNotNull("Loaded tokens should not be null", loadedTokens);
					Assert.assertArrayEquals("Loaded tokens should match original", originalTokens, loadedTokens);

					logger.log(DEBUG, "Successfully loaded state with " + loadedTokens.length + " tokens");
				}

			} finally {
				// Clean up temp file
				Files.deleteIfExists(tempFile);
			}
		}
	}

	@Test
	public void testSaveStateWithoutTokens() throws Exception {
		try (LlamaModel model = createModel()) {
			// Generate some context
			model.encode(TEST_PROMPT);

			// Create temporary file for state
			Path tempFile = Files.createTempFile("llama_state_no_tokens_", ".bin");
			String filePath = tempFile.toString();

			try {
				// Save state without tokens
				model.saveState(filePath);

				// Verify file was created
				File stateFile = new File(filePath);
				Assert.assertTrue("State file should exist", stateFile.exists());
				Assert.assertTrue("State file should not be empty", stateFile.length() > 0);

				// Load state from new model
				ModelParameters params = new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(99);
				try (LlamaModel newModel = new LlamaModel(params)) {
					int[] loadedTokens = newModel.loadState(filePath);

					// When no tokens were saved, should get empty array
					Assert.assertNotNull("Loaded tokens should not be null", loadedTokens);
					Assert.assertEquals("Should have no tokens when none were saved", 0, loadedTokens.length);

					logger.log(DEBUG, "Successfully loaded state without tokens");
				}

			} finally {
				Files.deleteIfExists(tempFile);
			}
		}
	}

	@Test
	public void testSequenceStateSize() {
		try (LlamaModel model = createModel()) {
			// Get sequence state size for unused sequence - may be small but non-zero
			long seqStateSize = model.getSequenceStateSize(0);
			Assert.assertTrue("Unused sequence should have minimal state size", seqStateSize >= 0);

			logger.log(DEBUG, "Sequence 0 state size (unused): " + seqStateSize + " bytes");

			// Generate context for sequence 0 by doing inference
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(1);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Now sequence should have state
			seqStateSize = model.getSequenceStateSize(0);
			Assert.assertTrue("Used sequence should have positive state size", seqStateSize > 0);

			logger.log(DEBUG, "Sequence 0 state size (after inference): " + seqStateSize + " bytes");
		}
	}

	@Test
	public void testSequenceStateData() {
		try (LlamaModel model = createModel()) {
			// Test unused sequence returns minimal state
			byte[] seqStateData = model.getSequenceState(0);
			Assert.assertNotNull("Sequence state data should not be null", seqStateData);

			logger.log(DEBUG, "Sequence 0 state data (unused): " + seqStateData.length + " bytes");

			// Generate context for sequence 0 by doing inference
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(1);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Now get sequence state data - should have content
			seqStateData = model.getSequenceState(0);
			Assert.assertNotNull("Sequence state data should not be null", seqStateData);
			Assert.assertTrue("Used sequence should have state data", seqStateData.length > 0);

			logger.log(DEBUG, "Sequence 0 state data (after inference): " + seqStateData.length + " bytes");

			// Create new model and restore sequence state
			ModelParameters newParams = new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(99);
			try (LlamaModel newModel = new LlamaModel(newParams)) {
				long loaded = newModel.setSequenceState(seqStateData, 0);
				Assert.assertTrue("Should have loaded sequence state", loaded > 0);

				logger.log(DEBUG, "Loaded " + loaded + " bytes of sequence state");
			}
		}
	}

	@Test
	public void testSaveAndLoadSequenceState() throws Exception {
		try (LlamaModel model = createModel()) {
			// Generate context for sequence 0 by doing inference
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(1);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Get tokens from the prompt
			int[] originalTokens = model.encode(TEST_PROMPT);

			// Create temporary file
			Path tempFile = Files.createTempFile("llama_seq_state_", ".bin");
			String filePath = tempFile.toString();

			try {
				// Save sequence state with tokens
				long saved = model.saveSequenceState(filePath, 0, originalTokens);
				Assert.assertTrue("Should have saved some bytes", saved > 0);

				// Verify file exists
				File stateFile = new File(filePath);
				Assert.assertTrue("Sequence state file should exist", stateFile.exists());

				logger.log(DEBUG, "Saved " + saved + " bytes of sequence state");

				// Create new model and load sequence state
				ModelParameters newParams = new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(99);
				try (LlamaModel newModel = new LlamaModel(newParams)) {
					int[] loadedTokens = newModel.loadSequenceState(filePath, 0);

					Assert.assertNotNull("Loaded tokens should not be null", loadedTokens);
					Assert.assertArrayEquals("Loaded tokens should match original", originalTokens, loadedTokens);

					logger.log(DEBUG, "Successfully loaded sequence state with " + loadedTokens.length + " tokens");
				}

			} finally {
				Files.deleteIfExists(tempFile);
			}
		}
	}

	@Test
	public void testInvalidArguments() {
		try (LlamaModel model = createModel()) {
			try {
				model.saveState(null);
				Assert.fail("Should throw IllegalArgumentException for null path");
			} catch (IllegalArgumentException e) {
				// Expected
			}

			try {
				model.saveState("");
				Assert.fail("Should throw IllegalArgumentException for empty path");
			} catch (IllegalArgumentException e) {
				// Expected
			}

			try {
				model.setModelState(null);
				Assert.fail("Should throw IllegalArgumentException for null state data");
			} catch (IllegalArgumentException e) {
				// Expected
			}

			try {
				model.setSequenceState(null, 0);
				Assert.fail("Should throw IllegalArgumentException for null sequence state data");
			} catch (IllegalArgumentException e) {
				// Expected
			}
		}
	}

	@Test
	public void testRoundTripStateConsistency() throws Exception {
		try (LlamaModel model = createModel()) {
			// Generate some context
			int[] originalTokens = model.encode(TEST_PROMPT);

			// Get original state size
			long originalStateSize = model.getModelStateSize();

			// Create temporary file
			Path tempFile = Files.createTempFile("llama_roundtrip_", ".bin");
			String filePath = tempFile.toString();

			try {
				// Save and reload state multiple times
				model.saveState(filePath, originalTokens);

				// Load into new model
				ModelParameters params = new ModelParameters()
					.setModel("models/codellama-7b.Q2_K.gguf")
					.setGpuLayers(99);
				try (LlamaModel newModel = new LlamaModel(params)) {
					int[] loadedTokens = newModel.loadState(filePath);

					// Verify tokens match
					Assert.assertArrayEquals("Round-trip tokens should match", originalTokens, loadedTokens);

					// Get state size from loaded model
					long newStateSize = newModel.getModelStateSize();
					Assert.assertEquals("State size should be consistent", originalStateSize, newStateSize);

					logger.log(DEBUG, "Round-trip test successful - state consistency maintained");
				}

			} finally {
				Files.deleteIfExists(tempFile);
			}
		}
	}
}
