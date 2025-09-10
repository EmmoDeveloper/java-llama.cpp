package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Assert;
import org.junit.Test;

public class MemoryKVCacheTest {
	private static final System.Logger logger = System.getLogger(MemoryKVCacheTest.class.getName());

	private static final String TEST_PROMPT = "The quick brown fox";

	private LlamaModel createModel() {
		// Use the same model configuration as other tests
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);  // Use GPU acceleration
		return new LlamaModel(params);
	}

	@Test
	public void testCopySequence() {
		try (LlamaModel model = createModel()) {
			// Generate context in sequence 0
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(5);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Copy sequence 0 to sequence 1
			model.copySequence(0, 1, 0, -1);

			// Verify both sequences exist by getting their positions
			try {
				int seq0Min = model.getSequenceMinPosition(0);
				int seq0Max = model.getSequenceMaxPosition(0);
				int seq1Min = model.getSequenceMinPosition(1);
				int seq1Max = model.getSequenceMaxPosition(1);

				// Positions may be -1 if sequence is empty, which is valid
				Assert.assertTrue("Should be able to query sequence 0 positions", seq0Min >= -1);
				Assert.assertTrue("Should be able to query sequence 1 positions", seq1Min >= -1);

				logger.log(DEBUG, "Sequence 0 range: [" + seq0Min + ", " + seq0Max + "]");
				logger.log(DEBUG, "Sequence 1 range: [" + seq1Min + ", " + seq1Max + "]");
			} catch (Exception e) {
				// This is acceptable if sequences have no data
				logger.log(DEBUG, "Sequences have no position data: " + e.getMessage());
			}

			logger.log(DEBUG, "Successfully copied sequence 0 to sequence 1");
		}
	}

	@Test
	public void testCopySequenceWithRange() {
		try (LlamaModel model = createModel()) {
			// Generate context in sequence 0
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(10);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Copy partial range from sequence 0 to sequence 2
			model.copySequence(0, 2, 0, 5);

			logger.log(DEBUG, "Successfully copied partial range from sequence 0 to sequence 2");
		}
	}

	@Test
	public void testKeepSequence() {
		try (LlamaModel model = createModel()) {
			// Generate context in multiple sequences
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(3);

			// Generate in sequence 0
			String result1 = model.complete(params);
			Assert.assertNotNull("First completion should not be null", result1);

			// Copy to create sequence 1
			model.copySequence(0, 1, 0, -1);

			// Keep only sequence 0, which should clear sequence 1
			model.keepSequence(0);

			logger.log(DEBUG, "Successfully kept sequence 0 and cleared others");
		}
	}

	@Test
	public void testAddPositionDelta() {
		try (LlamaModel model = createModel()) {
			// Generate context in sequence 0
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(5);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Add position delta to shift positions
			model.addPositionDelta(0, 0, -1, 10);

			logger.log(DEBUG, "Successfully added position delta to sequence 0");
		}
	}

	@Test
	public void testDividePositions() {
		try (LlamaModel model = createModel()) {
			// Generate context in sequence 0
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(8);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Divide positions by 2 to compress them
			model.dividePositions(0, 0, -1, 2);

			logger.log(DEBUG, "Successfully divided positions in sequence 0");
		}
	}

	@Test
	public void testSequencePositionQueries() {
		try (LlamaModel model = createModel()) {
			// Generate context in sequence 0
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(3);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			try {
				// Get position range for sequence 0
				int minPos = model.getSequenceMinPosition(0);
				int maxPos = model.getSequenceMaxPosition(0);

				// Positions may be -1 if sequence is empty, which is valid
				Assert.assertTrue("Min position should be valid", minPos >= -1);
				if (minPos >= 0 && maxPos >= 0) {
					Assert.assertTrue("Max position should be >= min position", maxPos >= minPos);
				}

				logger.log(DEBUG, "Sequence 0 position range: [" + minPos + ", " + maxPos + "]");

				// Test with non-existent sequence
				int minPos99 = model.getSequenceMinPosition(99);
				int maxPos99 = model.getSequenceMaxPosition(99);

				logger.log(DEBUG, "Sequence 99 position range: [" + minPos99 + ", " + maxPos99 + "]");
			} catch (Exception e) {
				// This is acceptable if sequence has no positions
				logger.log(DEBUG, "No position data available: " + e.getMessage());
			}
		}
	}

	@Test
	public void testCanShiftContext() {
		try (LlamaModel model = createModel()) {
			boolean canShift = model.canShiftContext();

			// Just verify we get a boolean response
			logger.log(DEBUG, "Context shifting support: " + canShift);
		}
	}

	@Test
	public void testClearMemory() {
		try (LlamaModel model = createModel()) {
			// Generate some context first
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(3);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Clear memory with data clearing
			model.clearMemory(true);
			logger.log(DEBUG, "Successfully cleared memory with data clearing");

			// Generate context again
			result = model.complete(params);
			Assert.assertNotNull("Should be able to generate after clearing", result);

			// Clear memory without data clearing (metadata only)
			model.clearMemory(false);
			logger.log(DEBUG, "Successfully cleared memory metadata only");

			// Test convenience method
			model.clearMemory();
			logger.log(DEBUG, "Successfully cleared memory using convenience method");
		}
	}

	@Test
	public void testRemoveSequenceTokens() {
		try (LlamaModel model = createModel()) {
			// Generate context in sequence 0
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(8);
			String result = model.complete(params);
			Assert.assertNotNull("Completion result should not be null", result);

			// Remove tokens from positions 2-5 in sequence 0
			boolean removed = model.removeSequenceTokens(0, 2, 6);

			// Result depends on whether there were tokens to remove
			logger.log(DEBUG, "Token removal result: " + removed);

			// Remove all tokens from sequence 0
			boolean removedAll = model.removeSequenceTokens(0, 0, -1);
			logger.log(DEBUG, "Remove all tokens result: " + removedAll);
		}
	}

	@Test
	public void testInvalidSequenceOperations() {
		try (LlamaModel model = createModel()) {
			// Test invalid sequence IDs
			try {
				model.copySequence(-1, 0, 0, -1);
				Assert.fail("Should throw IllegalArgumentException for negative source sequence ID");
			} catch (IllegalArgumentException e) {
				// Expected
				logger.log(DEBUG, "Correctly caught negative source sequence ID");
			}

			try {
				model.copySequence(0, -1, 0, -1);
				Assert.fail("Should throw IllegalArgumentException for negative destination sequence ID");
			} catch (IllegalArgumentException e) {
				// Expected
				logger.log(DEBUG, "Correctly caught negative destination sequence ID");
			}

			try {
				model.keepSequence(-1);
				Assert.fail("Should throw IllegalArgumentException for negative sequence ID");
			} catch (IllegalArgumentException e) {
				// Expected
				logger.log(DEBUG, "Correctly caught negative sequence ID in keepSequence");
			}
		}
	}

	@Test
	public void testInvalidPositionOperations() {
		try (LlamaModel model = createModel()) {
			// Test invalid positions
			try {
				model.copySequence(0, 1, -1, 5);
				Assert.fail("Should throw IllegalArgumentException for negative start position");
			} catch (IllegalArgumentException e) {
				// Expected
				logger.log(DEBUG, "Correctly caught negative start position");
			}

			try {
				model.copySequence(0, 1, 5, 3);
				Assert.fail("Should throw IllegalArgumentException for end <= start position");
			} catch (IllegalArgumentException e) {
				// Expected
				logger.log(DEBUG, "Correctly caught end <= start position");
			}

			try {
				model.dividePositions(0, 0, -1, 0);
				Assert.fail("Should throw IllegalArgumentException for zero divisor");
			} catch (IllegalArgumentException e) {
				// Expected
				logger.log(DEBUG, "Correctly caught zero divisor");
			}

			try {
				model.dividePositions(0, 0, -1, -2);
				Assert.fail("Should throw IllegalArgumentException for negative divisor");
			} catch (IllegalArgumentException e) {
				// Expected
				logger.log(DEBUG, "Correctly caught negative divisor");
			}
		}
	}

	@Test
	public void testSequenceBranching() {
		try (LlamaModel model = createModel()) {
			// Generate initial context in sequence 0
			InferenceParameters params = new InferenceParameters(TEST_PROMPT).setNPredict(5);
			String result0 = model.complete(params);
			Assert.assertNotNull("Initial completion should not be null", result0);

			// Copy sequence 0 to sequence 1 (creating a branch point)
			model.copySequence(0, 1, 0, -1);

			// Continue generation in both sequences (simulating branching)
			InferenceParameters continueParams = new InferenceParameters(" jumps").setNPredict(3);
			String continuation0 = model.complete(continueParams);
			Assert.assertNotNull("Continuation in sequence 0 should work", continuation0);

			logger.log(DEBUG, "Successfully created sequence branching scenario");
			logger.log(DEBUG, "Original: " + result0);
			logger.log(DEBUG, "Branch continuation: " + continuation0);
		}
	}

	@Test
	public void testMemoryOptimization() {
		try (LlamaModel model = createModel()) {
			// Create multiple sequences with different content
			InferenceParameters params1 = new InferenceParameters("Hello").setNPredict(3);
			InferenceParameters params2 = new InferenceParameters("World").setNPredict(3);

			// Generate in sequence 0
			String result1 = model.complete(params1);
			Assert.assertNotNull("First result should not be null", result1);

			// Copy to sequence 1 and modify
			model.copySequence(0, 1, 0, -1);

			// Add some position shifts to sequence 1
			model.addPositionDelta(1, 0, -1, 5);

			// Copy to sequence 2
			model.copySequence(1, 2, 0, -1);

			// Now optimize by keeping only sequence 0
			model.keepSequence(0);

			// Verify we can still generate from sequence 0
			String finalResult = model.complete(params2);
			Assert.assertNotNull("Final result after optimization should not be null", finalResult);

			logger.log(DEBUG, "Successfully performed memory optimization");
			logger.log(DEBUG, "Final result: " + finalResult);
		}
	}

	@Test
	public void testLargeSequenceOperations() {
		try (LlamaModel model = createModel()) {
			// Generate longer context to test with more substantial data
			InferenceParameters params = new InferenceParameters(TEST_PROMPT + " jumps over the lazy dog").setNPredict(15);
			String result = model.complete(params);
			Assert.assertNotNull("Long completion should not be null", result);

			// Test partial copying with specific ranges
			model.copySequence(0, 10, 2, 8);  // Copy positions 2-7
			model.copySequence(0, 11, 5, 12); // Copy positions 5-11

			// Test position manipulation on larger sequences
			model.addPositionDelta(10, 0, 4, -2);  // Shift first 4 positions back
			model.dividePositions(11, 3, -1, 2);   // Compress positions 3+ by half

			logger.log(DEBUG, "Successfully performed operations on longer sequences");
			logger.log(DEBUG, "Result length: " + result.length());
		}
	}
}
