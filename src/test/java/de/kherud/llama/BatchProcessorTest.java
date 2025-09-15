package de.kherud.llama;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import org.junit.Assert;

import static java.lang.System.Logger.Level.DEBUG;

public class BatchProcessorTest {

	private static final System.Logger logger = System.getLogger(BatchProcessorTest.class.getName());
	private LlamaModel model;

	@Before
	public void setUp() {
		ModelParameters params = new ModelParameters();
		params.setModel("models/codellama-7b.Q2_K.gguf");
		model = new LlamaModel(params);
	}

	@After
	public void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testBatchInitialization() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 1)) {
			Assert.assertNotNull("Batch processor should be initialized", batch);
		}
	}

	@Test
	public void testBatchInitializationWithEmbeddings() {
		try (BatchProcessor batch = new BatchProcessor(16, 4096, 2)) {
			Assert.assertNotNull("Batch processor should be initialized with embeddings", batch);
		}
	}

	@Test
	public void testTokenOperations() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 1)) {
			int[] tokens = {1, 2, 3, 4, 5};

			batch.setTokens(tokens);

			int[] retrievedTokens = batch.getTokens();
			Assert.assertNotNull("Retrieved tokens should not be null", retrievedTokens);
		}
	}

	@Test
	public void testPositionOperations() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 1)) {
			int[] positions = {0, 1, 2, 3, 4};

			batch.setPositions(positions);

			int[] retrievedPositions = batch.getPositions();
			Assert.assertNotNull("Retrieved positions should not be null", retrievedPositions);
		}
	}

	@Test
	public void testSequenceIdOperations() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 2)) {
			int[] sequenceIds = {0, 0, 1, 1, 1};

			batch.setSequenceIds(sequenceIds);

			int[] retrievedSequenceIds = batch.getSequenceIds();
			Assert.assertNotNull("Retrieved sequence IDs should not be null", retrievedSequenceIds);
		}
	}

	@Test
	public void testLogitFlagOperations() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 1)) {
			byte[] logitFlags = {1, 0, 1, 0, 1};

			batch.setLogitFlags(logitFlags);

			byte[] retrievedFlags = batch.getLogitFlags();
			Assert.assertNotNull("Retrieved logit flags should not be null", retrievedFlags);
		}
	}

	@Test
	public void testTokenCount() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 1)) {
			int count = batch.getTokenCount();
			Assert.assertTrue("Token count should be non-negative", count >= 0);
		}
	}

	@Ignore
	@Test
	public void testEncodeContext() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 1)) {
			int[] tokens = {1, 2, 3};
			int[] positions = {0, 1, 2};
			int[] sequenceIds = {0, 0, 0};
			byte[] logitFlags = {0, 0, 1};

			batch.setTokens(tokens);
			batch.setPositions(positions);
			batch.setSequenceIds(sequenceIds);
			batch.setLogitFlags(logitFlags);

			int result = batch.encodeContext(model);
			Assert.assertTrue("Encode result should be non-negative", result >= 0);

			logger.log(DEBUG, "Encode context result: " + result);
		}
	}

	@Ignore
	@Test
	public void testDecodeTokens() {
		try (BatchProcessor batch = new BatchProcessor(32, 0, 1)) {
			int[] tokens = {1, 2, 3};
			int[] positions = {0, 1, 2};
			int[] sequenceIds = {0, 0, 0};
			byte[] logitFlags = {0, 0, 1};

			batch.setTokens(tokens);
			batch.setPositions(positions);
			batch.setSequenceIds(sequenceIds);
			batch.setLogitFlags(logitFlags);

			int result = batch.decodeTokens(model);
			Assert.assertTrue("Decode result should be non-negative", result >= 0);

			logger.log(DEBUG, "Decode tokens result: " + result);
		}
	}

	@Ignore
	@Test
	public void testMultipleSequences() {
		try (BatchProcessor batch = new BatchProcessor(64, 0, 3)) {
			int[] tokens = {1, 2, 3, 4, 5, 6};
			int[] positions = {0, 1, 2, 0, 1, 2};
			int[] sequenceIds = {0, 0, 0, 1, 1, 1};
			byte[] logitFlags = {0, 0, 1, 0, 0, 1};

			batch.setTokens(tokens);
			batch.setPositions(positions);
			batch.setSequenceIds(sequenceIds);
			batch.setLogitFlags(logitFlags);

			int result = batch.decodeTokens(model);
			Assert.assertTrue("Multi-sequence decode should succeed", result >= 0);

			logger.log(DEBUG, "Multi-sequence decode result: " + result);
		}
	}

	@Test(expected = IllegalStateException.class)
	public void testClosedBatchThrowsException() {
		BatchProcessor batch = new BatchProcessor(32, 0, 1);
		batch.close();

		batch.getTokenCount();
	}

	@Test
	public void testAutoCloseableBehavior() {
		BatchProcessor batch = new BatchProcessor(32, 0, 1);
		Assert.assertNotNull("Batch should be created", batch);

		batch.close();

		try {
			batch.getTokenCount();
			Assert.fail("Should throw IllegalStateException after close");
		} catch (IllegalStateException e) {
			// Expected behavior
		}
	}
}
