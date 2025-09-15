package de.kherud.llama;

import org.junit.Before;
import org.junit.Test;
import org.junit.After;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import static org.junit.Assert.*;

public class LlamaTrainerTest {

	@Rule
	public TemporaryFolder tempFolder = new TemporaryFolder();

	private LlamaModel model;
	private File tempDataset;
	private File tempValidationDataset;
	private File tempCheckpoint;

	@Before
	public void setUp() throws IOException {
		// Initialize model for testing
		ModelParameters modelParams = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf");
		model = new LlamaModel(modelParams);

		// Create temporary dataset file
		tempDataset = tempFolder.newFile("train_dataset.txt");
		try (FileWriter writer = new FileWriter(tempDataset)) {
			writer.write("This is a training sample.\n");
			writer.write("Another training example.\n");
			writer.write("Yet another training line.\n");
			writer.write("Training data for fine-tuning.\n");
			writer.write("Sample text for model training.\n");
		}

		// Create temporary validation dataset
		tempValidationDataset = tempFolder.newFile("val_dataset.txt");
		try (FileWriter writer = new FileWriter(tempValidationDataset)) {
			writer.write("Validation sample one.\n");
			writer.write("Validation sample two.\n");
			writer.write("Validation sample three.\n");
		}

		// Set checkpoint path
		tempCheckpoint = new File(tempFolder.getRoot(), "checkpoint.bin");
	}

	@After
	public void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testGetDefaultParams() {
		LlamaTrainer.TrainingParams params = LlamaTrainer.getDefaultParams();

		assertNotNull(params);
		assertEquals(1, params.getEpochs());
		assertEquals(1e-4f, params.getLearningRate(), 1e-6);
		assertEquals(32, params.getBatchSize());
		assertEquals(1, params.getGradientAccumulationSteps());
		assertEquals(0.1f, params.getWarmupRatio(), 1e-6);
		assertEquals(0.01f, params.getWeightDecay(), 1e-6);
		assertEquals(1, params.getMaxGradientNorm());
		assertTrue(params.isUseAdamW());
		assertEquals(500, params.getSaveSteps());
		assertEquals(100, params.getEvalSteps());
	}

	@Test
	public void testTrainingParamsBuilder() {
		LlamaTrainer.TrainingParams params = LlamaTrainer.getDefaultParams()
				.setEpochs(5)
				.setLearningRate(0.001f)
				.setBatchSize(16)
				.setGradientAccumulationSteps(2)
				.setWarmupRatio(0.2f)
				.setWeightDecay(0.05f)
				.setMaxGradientNorm(2)
				.setUseAdamW(false)
				.setSaveSteps(1000)
				.setEvalSteps(200);

		assertEquals(5, params.getEpochs());
		assertEquals(0.001f, params.getLearningRate(), 1e-6);
		assertEquals(16, params.getBatchSize());
		assertEquals(2, params.getGradientAccumulationSteps());
		assertEquals(0.2f, params.getWarmupRatio(), 1e-6);
		assertEquals(0.05f, params.getWeightDecay(), 1e-6);
		assertEquals(2, params.getMaxGradientNorm());
		assertFalse(params.isUseAdamW());
		assertEquals(1000, params.getSaveSteps());
		assertEquals(200, params.getEvalSteps());
	}

	@Test
	public void testValidateDataset() {
		assertTrue(LlamaTrainer.validateDataset(tempDataset.getAbsolutePath()));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testValidateDatasetWithNullPath() {
		LlamaTrainer.validateDataset(null);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testValidateDatasetWithEmptyPath() {
		LlamaTrainer.validateDataset("");
	}

	@Test
	public void testValidateDatasetWithNonExistentFile() {
		assertFalse(LlamaTrainer.validateDataset("/non/existent/file.txt"));
	}

	@Test
	public void testPrepareTraining() {
		LlamaTrainer.TrainingParams params = LlamaTrainer.getDefaultParams();
		long trainingHandle = LlamaTrainer.prepareTraining(model, params);

		assertTrue(trainingHandle > 0);

		// Clean up
		LlamaTrainer.finishTraining(trainingHandle);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testPrepareTrainingWithNullModel() {
		LlamaTrainer.TrainingParams params = LlamaTrainer.getDefaultParams();
		LlamaTrainer.prepareTraining(null, params);
	}

	@Test
	public void testPrepareTrainingWithNullParams() {
		// Should use default params when null is passed
		long trainingHandle = LlamaTrainer.prepareTraining(model, null);
		assertTrue(trainingHandle > 0);

		// Clean up
		LlamaTrainer.finishTraining(trainingHandle);
	}

	@Test
	public void testTrainingLifecycle() {
		LlamaTrainer.TrainingParams params = LlamaTrainer.getDefaultParams()
				.setEpochs(2)
				.setLearningRate(0.0001f);

		// Prepare training
		long trainingHandle = LlamaTrainer.prepareTraining(model, params);
		assertTrue(trainingHandle > 0);

		// Create progress callback to track training
		StringBuilder progressLog = new StringBuilder();
		LlamaTrainer.TrainingCallback callback = new LlamaTrainer.TrainingCallback() {
			@Override
			public void onProgress(int epoch, int step, float loss, float learningRate) {
				progressLog.append(String.format("Epoch %d, Step %d, Loss %.4f, LR %.6f\n",
						epoch, step, loss, learningRate));
			}

			@Override
			public void onEvaluation(int epoch, float evalLoss, float evalAccuracy) {
				progressLog.append(String.format("Evaluation - Epoch %d, Loss %.4f, Accuracy %.3f\n",
						epoch, evalLoss, evalAccuracy));
			}

			@Override
			public void onCheckpointSave(int epoch, String checkpointPath) {
				progressLog.append(String.format("Checkpoint saved - Epoch %d, Path %s\n",
						epoch, checkpointPath));
			}
		};

		// Train one epoch
		LlamaTrainer.TrainingMetrics metrics = LlamaTrainer.trainEpoch(
				trainingHandle, tempDataset.getAbsolutePath(), callback);

		assertNotNull(metrics);
		assertTrue(metrics.getLoss() > 0);
		assertTrue(metrics.getLearningRate() > 0);
		assertTrue(metrics.getTotalSteps() > 0);
		assertTrue(metrics.getTrainingTime() >= 0);

		// Verify callback was called
		assertTrue(progressLog.length() > 0);
		assertTrue(progressLog.toString().contains("Epoch"));

		// Evaluate model
		LlamaTrainer.EvaluationMetrics evalMetrics = LlamaTrainer.evaluate(
				trainingHandle, tempValidationDataset.getAbsolutePath());

		assertNotNull(evalMetrics);
		assertTrue(evalMetrics.getLoss() > 0);
		assertTrue(evalMetrics.getAccuracy() >= 0 && evalMetrics.getAccuracy() <= 1);
		assertTrue(evalMetrics.getPerplexity() > 0);
		assertTrue(evalMetrics.getTotalSamples() > 0);

		// Save checkpoint
		LlamaTrainer.saveCheckpoint(trainingHandle, tempCheckpoint.getAbsolutePath());
		assertTrue(tempCheckpoint.exists());

		// Load checkpoint
		LlamaTrainer.loadCheckpoint(trainingHandle, tempCheckpoint.getAbsolutePath());

		// Finish training
		LlamaTrainer.finishTraining(trainingHandle);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testTrainEpochWithNullDatasetPath() {
		long trainingHandle = LlamaTrainer.prepareTraining(model, LlamaTrainer.getDefaultParams());

		try {
			LlamaTrainer.trainEpoch(trainingHandle, null, null);
		} finally {
			LlamaTrainer.finishTraining(trainingHandle);
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void testTrainEpochWithEmptyDatasetPath() {
		long trainingHandle = LlamaTrainer.prepareTraining(model, LlamaTrainer.getDefaultParams());

		try {
			LlamaTrainer.trainEpoch(trainingHandle, "", null);
		} finally {
			LlamaTrainer.finishTraining(trainingHandle);
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void testEvaluateWithNullDatasetPath() {
		long trainingHandle = LlamaTrainer.prepareTraining(model, LlamaTrainer.getDefaultParams());

		try {
			LlamaTrainer.evaluate(trainingHandle, null);
		} finally {
			LlamaTrainer.finishTraining(trainingHandle);
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSaveCheckpointWithNullPath() {
		long trainingHandle = LlamaTrainer.prepareTraining(model, LlamaTrainer.getDefaultParams());

		try {
			LlamaTrainer.saveCheckpoint(trainingHandle, null);
		} finally {
			LlamaTrainer.finishTraining(trainingHandle);
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void testLoadCheckpointWithNullPath() {
		long trainingHandle = LlamaTrainer.prepareTraining(model, LlamaTrainer.getDefaultParams());

		try {
			LlamaTrainer.loadCheckpoint(trainingHandle, null);
		} finally {
			LlamaTrainer.finishTraining(trainingHandle);
		}
	}

	@Test
	public void testTrainingMetricsToString() {
		LlamaTrainer.TrainingMetrics metrics = new LlamaTrainer.TrainingMetrics(
				2.5f, 0.0001f, 100, 5000L);

		String result = metrics.toString();
		assertTrue(result.contains("loss=2.5000"));
		assertTrue(result.contains("lr=0.000100"));
		assertTrue(result.contains("steps=100"));
		assertTrue(result.contains("time=5000ms"));
	}

	@Test
	public void testEvaluationMetricsToString() {
		LlamaTrainer.EvaluationMetrics metrics = new LlamaTrainer.EvaluationMetrics(
				1.8f, 0.85f, 6.05f, 50);

		String result = metrics.toString();
		assertTrue(result.contains("loss=1.8000"));
		assertTrue(result.contains("accuracy=0.850"));
		assertTrue(result.contains("perplexity=6.05"));
		assertTrue(result.contains("samples=50"));
	}

	@Test
	public void testTrainEpochWithoutCallback() {
		long trainingHandle = LlamaTrainer.prepareTraining(model, LlamaTrainer.getDefaultParams());

		// Should work without callback
		LlamaTrainer.TrainingMetrics metrics = LlamaTrainer.trainEpoch(
				trainingHandle, tempDataset.getAbsolutePath(), null);

		assertNotNull(metrics);
		assertTrue(metrics.getLoss() > 0);

		LlamaTrainer.finishTraining(trainingHandle);
	}

	@Test
	public void testMultipleTrainingEpochs() {
		LlamaTrainer.TrainingParams params = LlamaTrainer.getDefaultParams()
				.setEpochs(3)
				.setLearningRate(0.001f);

		long trainingHandle = LlamaTrainer.prepareTraining(model, params);

		float previousLR = params.getLearningRate();

		// Train multiple epochs and verify learning rate decay
		for (int epoch = 0; epoch < 3; epoch++) {
			LlamaTrainer.TrainingMetrics metrics = LlamaTrainer.trainEpoch(
					trainingHandle, tempDataset.getAbsolutePath(), null);

			// Loss should be positive
			assertTrue(metrics.getLoss() > 0);

			// Learning rate should decay (according to implementation: *= 0.995f)
			assertTrue(metrics.getLearningRate() <= previousLR);
			previousLR = metrics.getLearningRate();
		}

		LlamaTrainer.finishTraining(trainingHandle);
	}
}
