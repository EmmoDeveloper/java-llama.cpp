package de.kherud.llama;

/**
 * Training and optimization utilities for llama.cpp models.
 * Provides fine-tuning capabilities and training optimization functions.
 */
public class LlamaTrainer {

	static {
		LlamaLoader.initialize();
	}

	/**
	 * Training optimization parameters for fine-tuning operations.
	 */
	public static class TrainingParams {
		private int epochs = 1;
		private float learningRate = 1e-4f;
		private int batchSize = 32;
		private int gradientAccumulationSteps = 1;
		private float warmupRatio = 0.1f;
		private float weightDecay = 0.01f;
		private int maxGradientNorm = 1;
		private boolean useAdamW = true;
		private int saveSteps = 500;
		private int evalSteps = 100;

		public TrainingParams() {
		}

		public TrainingParams setEpochs(int epochs) {
			this.epochs = epochs;
			return this;
		}

		public TrainingParams setLearningRate(float learningRate) {
			this.learningRate = learningRate;
			return this;
		}

		public TrainingParams setBatchSize(int batchSize) {
			this.batchSize = batchSize;
			return this;
		}

		public TrainingParams setGradientAccumulationSteps(int steps) {
			this.gradientAccumulationSteps = steps;
			return this;
		}

		public TrainingParams setWarmupRatio(float warmupRatio) {
			this.warmupRatio = warmupRatio;
			return this;
		}

		public TrainingParams setWeightDecay(float weightDecay) {
			this.weightDecay = weightDecay;
			return this;
		}

		public TrainingParams setMaxGradientNorm(int maxGradientNorm) {
			this.maxGradientNorm = maxGradientNorm;
			return this;
		}

		public TrainingParams setUseAdamW(boolean useAdamW) {
			this.useAdamW = useAdamW;
			return this;
		}

		public TrainingParams setSaveSteps(int saveSteps) {
			this.saveSteps = saveSteps;
			return this;
		}

		public TrainingParams setEvalSteps(int evalSteps) {
			this.evalSteps = evalSteps;
			return this;
		}

		// Getters
		public int getEpochs() { return epochs; }
		public float getLearningRate() { return learningRate; }
		public int getBatchSize() { return batchSize; }
		public int getGradientAccumulationSteps() { return gradientAccumulationSteps; }
		public float getWarmupRatio() { return warmupRatio; }
		public float getWeightDecay() { return weightDecay; }
		public int getMaxGradientNorm() { return maxGradientNorm; }
		public boolean isUseAdamW() { return useAdamW; }
		public int getSaveSteps() { return saveSteps; }
		public int getEvalSteps() { return evalSteps; }
	}

	/**
	 * Training progress callback interface.
	 */
	public interface TrainingCallback {
		/**
		 * Called during training progress.
		 * @param epoch current epoch
		 * @param step current step
		 * @param loss current loss value
		 * @param learningRate current learning rate
		 */
		void onProgress(int epoch, int step, float loss, float learningRate);

		/**
		 * Called when evaluation is performed.
		 * @param epoch current epoch
		 * @param evalLoss evaluation loss
		 * @param evalAccuracy evaluation accuracy
		 */
		void onEvaluation(int epoch, float evalLoss, float evalAccuracy);

		/**
		 * Called when a checkpoint is saved.
		 * @param epoch current epoch
		 * @param checkpointPath path to saved checkpoint
		 */
		void onCheckpointSave(int epoch, String checkpointPath);
	}

	/**
	 * Initialize the training optimizer with default parameters.
	 * @return default training parameters
	 */
	public static TrainingParams getDefaultParams() {
		return new TrainingParams();
	}

	/**
	 * Validate training dataset format.
	 * @param datasetPath path to the training dataset
	 * @return true if dataset format is valid
	 */
	public static boolean validateDataset(String datasetPath) {
		if (datasetPath == null || datasetPath.trim().isEmpty()) {
			throw new IllegalArgumentException("Dataset path cannot be null or empty");
		}
		return validateDatasetNative(datasetPath);
	}

	/**
	 * Prepare model for fine-tuning training.
	 * @param model the model to prepare
	 * @param params training parameters
	 * @return training handle for the session
	 */
	public static long prepareTraining(LlamaModel model, TrainingParams params) {
		if (model == null) {
			throw new IllegalArgumentException("Model cannot be null");
		}
		if (params == null) {
			params = getDefaultParams();
		}
		return prepareTrainingNative(model, params);
	}

	/**
	 * Execute a single training epoch.
	 * @param trainingHandle training session handle
	 * @param datasetPath path to training dataset
	 * @param callback progress callback (optional)
	 * @return training metrics for this epoch
	 */
	public static TrainingMetrics trainEpoch(long trainingHandle, String datasetPath, TrainingCallback callback) {
		if (datasetPath == null || datasetPath.trim().isEmpty()) {
			throw new IllegalArgumentException("Dataset path cannot be null or empty");
		}
		return trainEpochNative(trainingHandle, datasetPath, callback);
	}

	/**
	 * Evaluate model performance on validation dataset.
	 * @param trainingHandle training session handle
	 * @param validationDatasetPath path to validation dataset
	 * @return evaluation metrics
	 */
	public static EvaluationMetrics evaluate(long trainingHandle, String validationDatasetPath) {
		if (validationDatasetPath == null || validationDatasetPath.trim().isEmpty()) {
			throw new IllegalArgumentException("Validation dataset path cannot be null or empty");
		}
		return evaluateNative(trainingHandle, validationDatasetPath);
	}

	/**
	 * Save training checkpoint.
	 * @param trainingHandle training session handle
	 * @param checkpointPath path to save checkpoint
	 */
	public static void saveCheckpoint(long trainingHandle, String checkpointPath) {
		if (checkpointPath == null || checkpointPath.trim().isEmpty()) {
			throw new IllegalArgumentException("Checkpoint path cannot be null or empty");
		}
		saveCheckpointNative(trainingHandle, checkpointPath);
	}

	/**
	 * Load training checkpoint.
	 * @param trainingHandle training session handle
	 * @param checkpointPath path to checkpoint file
	 */
	public static void loadCheckpoint(long trainingHandle, String checkpointPath) {
		if (checkpointPath == null || checkpointPath.trim().isEmpty()) {
			throw new IllegalArgumentException("Checkpoint path cannot be null or empty");
		}
		loadCheckpointNative(trainingHandle, checkpointPath);
	}

	/**
	 * Finalize training session and clean up resources.
	 * @param trainingHandle training session handle
	 */
	public static void finishTraining(long trainingHandle) {
		finishTrainingNative(trainingHandle);
	}

	/**
	 * Training metrics for a completed epoch.
	 */
	public static class TrainingMetrics {
		private final float loss;
		private final float learningRate;
		private final int totalSteps;
		private final long trainingTime;

		public TrainingMetrics(float loss, float learningRate, int totalSteps, long trainingTime) {
			this.loss = loss;
			this.learningRate = learningRate;
			this.totalSteps = totalSteps;
			this.trainingTime = trainingTime;
		}

		public float getLoss() { return loss; }
		public float getLearningRate() { return learningRate; }
		public int getTotalSteps() { return totalSteps; }
		public long getTrainingTime() { return trainingTime; }

		@Override
		public String toString() {
			return String.format("TrainingMetrics{loss=%.4f, lr=%.6f, steps=%d, time=%dms}",
								loss, learningRate, totalSteps, trainingTime);
		}
	}

	/**
	 * Evaluation metrics for model performance assessment.
	 */
	public static class EvaluationMetrics {
		private final float loss;
		private final float accuracy;
		private final float perplexity;
		private final int totalSamples;

		public EvaluationMetrics(float loss, float accuracy, float perplexity, int totalSamples) {
			this.loss = loss;
			this.accuracy = accuracy;
			this.perplexity = perplexity;
			this.totalSamples = totalSamples;
		}

		public float getLoss() { return loss; }
		public float getAccuracy() { return accuracy; }
		public float getPerplexity() { return perplexity; }
		public int getTotalSamples() { return totalSamples; }

		@Override
		public String toString() {
			return String.format("EvaluationMetrics{loss=%.4f, accuracy=%.3f, perplexity=%.2f, samples=%d}",
								loss, accuracy, perplexity, totalSamples);
		}
	}

	// Native method declarations
	private static native boolean validateDatasetNative(String datasetPath);
	private static native long prepareTrainingNative(LlamaModel model, TrainingParams params);
	private static native TrainingMetrics trainEpochNative(long trainingHandle, String datasetPath, TrainingCallback callback);
	private static native EvaluationMetrics evaluateNative(long trainingHandle, String validationDatasetPath);
	private static native void saveCheckpointNative(long trainingHandle, String checkpointPath);
	private static native void loadCheckpointNative(long trainingHandle, String checkpointPath);
	private static native void finishTrainingNative(long trainingHandle);
}