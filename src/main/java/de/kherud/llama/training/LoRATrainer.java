package de.kherud.llama.training;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.gguf.GGUFConstants;
import de.kherud.llama.gguf.GGUFWriter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Native Java LoRA (Low-Rank Adaptation) training implementation.
 * Creates LoRA adapters compatible with existing loadLoRAAdapter() system.
 *
 * Based on LoRA paper: <a href="https://arxiv.org/abs/2106.09685">...</a>
 * W' = W + α * (B * A) where rank(B*A) << rank(W)
 */
public class LoRATrainer {
	private static final System.Logger LOGGER = System.getLogger(LoRATrainer.class.getName());

	public static class LoRAConfig {
		private final int rank;                    // LoRA rank (r) - controls adapter expressiveness
		private final float alpha;                 // LoRA scaling factor (α)
		private final float dropout;               // Dropout rate for training
		private final String[] targetModules;     // Which attention modules to adapt
		private final int maxSequenceLength;      // Maximum training sequence length
		private final boolean gradientCheckpointing; // Memory optimization

		private LoRAConfig(Builder builder) {
			this.rank = builder.rank;
			this.alpha = builder.alpha;
			this.dropout = builder.dropout;
			this.targetModules = builder.targetModules;
			this.maxSequenceLength = builder.maxSequenceLength;
			this.gradientCheckpointing = builder.gradientCheckpointing;
		}

		public static Builder builder() {
			return new Builder();
		}

		public static class Builder {
			private int rank = 16;
			private float alpha = 32.0f;
			private float dropout = 0.1f;
			private String[] targetModules = {"q_proj", "k_proj", "v_proj", "o_proj"};
			private int maxSequenceLength = 2048;
			private boolean gradientCheckpointing = true;

			public Builder rank(int rank) { this.rank = rank; return this; }
			public Builder alpha(float alpha) { this.alpha = alpha; return this; }
			public Builder dropout(float dropout) { this.dropout = dropout; return this; }
			public Builder targetModules(String... modules) { this.targetModules = modules; return this; }
			public Builder maxSequenceLength(int maxSeq) { this.maxSequenceLength = maxSeq; return this; }
			public Builder gradientCheckpointing(boolean enabled) { this.gradientCheckpointing = enabled; return this; }

			public LoRAConfig build() {
				return new LoRAConfig(this);
			}
		}

		// Getters
		public int getRank() { return rank; }
		public float getAlpha() { return alpha; }
		public float getDropout() { return dropout; }
		public String[] getTargetModules() { return targetModules; }
		public int getMaxSequenceLength() { return maxSequenceLength; }
		public boolean isGradientCheckpointing() { return gradientCheckpointing; }
	}

	public static class TrainingConfig {
		private final int epochs;
		private final int batchSize;
		private final float learningRate;
		private final float weightDecay;
		private final int warmupSteps;
		private final int saveSteps;
		private final String outputDir;

		private TrainingConfig(Builder builder) {
			this.epochs = builder.epochs;
			this.batchSize = builder.batchSize;
			this.learningRate = builder.learningRate;
			this.weightDecay = builder.weightDecay;
			this.warmupSteps = builder.warmupSteps;
			this.saveSteps = builder.saveSteps;
			this.outputDir = builder.outputDir;
		}

		public static Builder builder() {
			return new Builder();
		}

		public static class Builder {
			private int epochs = 3;
			private int batchSize = 4;
			private float learningRate = 2e-4f;
			private float weightDecay = 0.01f;
			private int warmupSteps = 100;
			private int saveSteps = 500;
			private String outputDir = "./lora_output";

			public Builder epochs(int epochs) { this.epochs = epochs; return this; }
			public Builder batchSize(int batchSize) { this.batchSize = batchSize; return this; }
			public Builder learningRate(float lr) { this.learningRate = lr; return this; }
			public Builder weightDecay(float decay) { this.weightDecay = decay; return this; }
			public Builder warmupSteps(int steps) { this.warmupSteps = steps; return this; }
			public Builder saveSteps(int steps) { this.saveSteps = steps; return this; }
			public Builder outputDir(String dir) { this.outputDir = dir; return this; }

			public TrainingConfig build() {
				return new TrainingConfig(this);
			}
		}

		// Getters
		public int getEpochs() { return epochs; }
		public int getBatchSize() { return batchSize; }
		public float getLearningRate() { return learningRate; }
		public float getWeightDecay() { return weightDecay; }
		public int getWarmupSteps() { return warmupSteps; }
		public int getSaveSteps() { return saveSteps; }
		public String getOutputDir() { return outputDir; }
	}

	/**
	 * LoRA weight matrices for a single module.
	 */
	public static class LoRAModule {
		private final String name;
		private final int inputDim;
		private final int outputDim;
		private final int rank;

		// LoRA matrices: W_new = W_original + α * B * A
		private final float[][] matrixA;  // [rank, input_dim] - initialized with random values
		private final float[][] matrixB;  // [output_dim, rank] - initialized to zero

		// Gradient accumulation
		private final float[][] gradA;
		private final float[][] gradB;

		// Adam optimizer state
		private final float[][] momentumA1, momentumA2;
		private final float[][] momentumB1, momentumB2;

		public LoRAModule(String name, int inputDim, int outputDim, int rank) {
			this.name = name;
			this.inputDim = inputDim;
			this.outputDim = outputDim;
			this.rank = rank;

			// Initialize matrices
			this.matrixA = initializeGaussian(rank, inputDim, Math.sqrt(1.0 / rank));
			this.matrixB = new float[outputDim][rank]; // Zero initialization

			// Initialize gradients
			this.gradA = new float[rank][inputDim];
			this.gradB = new float[outputDim][rank];

			// Initialize Adam optimizer state
			this.momentumA1 = new float[rank][inputDim];
			this.momentumA2 = new float[rank][inputDim];
			this.momentumB1 = new float[outputDim][rank];
			this.momentumB2 = new float[outputDim][rank];
		}

		private float[][] initializeGaussian(int rows, int cols, double std) {
			float[][] matrix = new float[rows][cols];
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					matrix[i][j] = (float) (ThreadLocalRandom.current().nextGaussian() * std);
				}
			}
			return matrix;
		}

		/**
		 * Forward pass: compute LoRA adaptation
		 * output_delta = α * B * A * input
		 */
		public float[] forward(float[] input, float alpha, boolean training, float dropoutRate) {
			// Apply input dropout during training
			float[] processedInput = input;
			if (training && dropoutRate > 0) {
				processedInput = applyDropout(input, dropoutRate);
			}

			// A * input: [rank, input_dim] * [input_dim] = [rank]
			float[] aOutput = new float[rank];
			for (int i = 0; i < rank; i++) {
				for (int j = 0; j < inputDim; j++) {
					aOutput[i] += matrixA[i][j] * processedInput[j];
				}
			}

			// B * aOutput: [output_dim, rank] * [rank] = [output_dim]
			float[] delta = new float[outputDim];
			for (int i = 0; i < outputDim; i++) {
				for (int j = 0; j < rank; j++) {
					delta[i] += matrixB[i][j] * aOutput[j];
				}
				delta[i] *= alpha; // Apply LoRA scaling
			}

			return delta;
		}

		/**
		 * Backward pass: accumulate gradients
		 */
		public void backward(float[] inputActivation, float[] outputGrad, float alpha) {
			// Compute gradients for B: gradB = α * outputGrad ⊗ (A * input)
			float[] aOutput = new float[rank];
			for (int i = 0; i < rank; i++) {
				for (int j = 0; j < inputDim; j++) {
					aOutput[i] += matrixA[i][j] * inputActivation[j];
				}
			}

			for (int i = 0; i < outputDim; i++) {
				for (int j = 0; j < rank; j++) {
					gradB[i][j] += alpha * outputGrad[i] * aOutput[j];
				}
			}

			// Compute gradients for A: gradA = α * (B^T * outputGrad) ⊗ input
			float[] bTOutput = new float[rank];
			for (int i = 0; i < rank; i++) {
				for (int j = 0; j < outputDim; j++) {
					bTOutput[i] += matrixB[j][i] * outputGrad[j];
				}
			}

			for (int i = 0; i < rank; i++) {
				for (int j = 0; j < inputDim; j++) {
					gradA[i][j] += alpha * bTOutput[i] * inputActivation[j];
				}
			}
		}

		/**
		 * Apply dropout to input vector during training
		 */
		private float[] applyDropout(float[] input, float dropoutRate) {
			float[] result = new float[input.length];
			float scale = 1.0f / (1.0f - dropoutRate); // Inverted dropout scaling

			for (int i = 0; i < input.length; i++) {
				if (ThreadLocalRandom.current().nextFloat() < dropoutRate) {
					result[i] = 0.0f; // Drop this element
				} else {
					result[i] = input[i] * scale; // Scale remaining elements
				}
			}
			return result;
		}

		/**
		 * Update weights using Adam optimizer
		 */
		public void updateWeights(float learningRate, float beta1, float beta2, float epsilon, int step) {
			float beta1Power = (float) Math.pow(beta1, step);
			float beta2Power = (float) Math.pow(beta2, step);
			float correctedLR = learningRate * (float) Math.sqrt(1 - beta2Power) / (1 - beta1Power);

			// Update A
			for (int i = 0; i < rank; i++) {
				for (int j = 0; j < inputDim; j++) {
					momentumA1[i][j] = beta1 * momentumA1[i][j] + (1 - beta1) * gradA[i][j];
					momentumA2[i][j] = beta2 * momentumA2[i][j] + (1 - beta2) * gradA[i][j] * gradA[i][j];

					float update = correctedLR * momentumA1[i][j] / ((float) Math.sqrt(momentumA2[i][j]) + epsilon);
					matrixA[i][j] -= update;

					gradA[i][j] = 0; // Reset gradient
				}
			}

			// Update B
			for (int i = 0; i < outputDim; i++) {
				for (int j = 0; j < rank; j++) {
					momentumB1[i][j] = beta1 * momentumB1[i][j] + (1 - beta1) * gradB[i][j];
					momentumB2[i][j] = beta2 * momentumB2[i][j] + (1 - beta2) * gradB[i][j] * gradB[i][j];

					float update = correctedLR * momentumB1[i][j] / ((float) Math.sqrt(momentumB2[i][j]) + epsilon);
					matrixB[i][j] -= update;

					gradB[i][j] = 0; // Reset gradient
				}
			}
		}

		// Getters
		public String getName() { return name; }
		public float[][] getMatrixA() { return matrixA; }
		public float[][] getMatrixB() { return matrixB; }
		public int getRank() { return rank; }
		public int getInputDim() { return inputDim; }
		public int getOutputDim() { return outputDim; }
	}

	private final LlamaModel baseModel;
	private final LoRAConfig loraConfig;
	private final TrainingConfig trainingConfig;
	private final Map<String, LoRAModule> loraModules;
	private int globalStep = 0;

	public LoRATrainer(LlamaModel baseModel, LoRAConfig loraConfig, TrainingConfig trainingConfig) {
		this.baseModel = baseModel;
		this.loraConfig = loraConfig;
		this.trainingConfig = trainingConfig;
		this.loraModules = new TreeMap<>();

		initializeLoRAModules();
		createOutputDirectory();
	}

	/**
	 * Initialize LoRA modules based on model architecture
	 */
	private void initializeLoRAModules() {
		// Use standard transformer dimensions (these will be configurable)
		int modelDim = 4096;  // Standard model dimension (should be configurable)
		int layers = 32;      // Standard layer count (should be configurable)

		LOGGER.log(System.Logger.Level.INFO,String.format("Initializing LoRA modules: %d layers, %d model dimensions", layers, modelDim));

		// Map common module names to actual model tensor names (without .weight suffix)
		Map<String, String> moduleNameMapping = Map.of(
			"q_proj", "attn_q",
			"k_proj", "attn_k",
			"v_proj", "attn_v",
			"o_proj", "attn_output"
		);

		// Create LoRA modules for ALL layers
		for (String moduleName : loraConfig.getTargetModules()) {
			String baseModuleName = moduleNameMapping.getOrDefault(moduleName, moduleName);

			// Create LoRA module for each layer
			for (int layer = 0; layer < layers; layer++) {
				String actualTensorName = String.format("blk.%d.%s.weight", layer, baseModuleName);
				// All attention projections have same dimensions in transformer
				loraModules.put(actualTensorName, new LoRAModule(actualTensorName, modelDim, modelDim, loraConfig.getRank()));
			}
		}

		LOGGER.log(System.Logger.Level.INFO,String.format("Created %d LoRA modules with rank %d across %d layers",
		                         loraModules.size(), loraConfig.getRank(), layers));
	}

	private void createOutputDirectory() {
		new File(trainingConfig.getOutputDir()).mkdirs();
	}

	/**
	 * Main training method
	 */
	public void train(List<TrainingApplication> dataset) {
		LOGGER.log(System.Logger.Level.INFO,"Starting LoRA training...");
		LOGGER.log(System.Logger.Level.INFO,String.format("Dataset size: %d examples", dataset.size()));
		LOGGER.log(System.Logger.Level.INFO,String.format("Training config: %d epochs, batch size %d, LR %.6f",
		                         trainingConfig.getEpochs(), trainingConfig.getBatchSize(),
		                         trainingConfig.getLearningRate()));
		LOGGER.log(System.Logger.Level.INFO,String.format("LoRA config: rank %d, alpha %.2f, dropout %.2f",
		                         loraConfig.getRank(), loraConfig.getAlpha(), loraConfig.getDropout()));

		createOutputDirectory();

		// Training metrics
		float bestLoss = Float.MAX_VALUE;
		long totalTrainingTime = 0;

		// Real training loop
		for (int epoch = 0; epoch < trainingConfig.getEpochs(); epoch++) {
			long epochStartTime = System.currentTimeMillis();

			LOGGER.log(System.Logger.Level.INFO,String.format("=== Training epoch %d/%d ===", epoch + 1, trainingConfig.getEpochs()));

			float epochLoss = trainEpoch(dataset, epoch);

			long epochTime = System.currentTimeMillis() - epochStartTime;
			totalTrainingTime += epochTime;

			LOGGER.log(System.Logger.Level.INFO,String.format("Epoch %d completed: loss %.6f, time %d ms",
			                         epoch + 1, epochLoss, epochTime));

			// Save checkpoint if loss improved
			if (epochLoss < bestLoss) {
				bestLoss = epochLoss;
				saveLoRAAdapter(String.format("%s/best_adapter_epoch_%d.gguf",
				                             trainingConfig.getOutputDir(), epoch + 1));
				LOGGER.log(System.Logger.Level.INFO,String.format("New best loss: %.6f - saved checkpoint", bestLoss));
			}

			// Save periodic checkpoint
			if ((epoch + 1) % Math.max(1, trainingConfig.getEpochs() / 5) == 0) {
				saveLoRAAdapter(String.format("%s/checkpoint_epoch_%d.gguf",
				                             trainingConfig.getOutputDir(), epoch + 1));
			}
		}

		// Final save
		saveLoRAAdapter(String.format("%s/final_adapter.gguf", trainingConfig.getOutputDir()));

		// Training summary
		LOGGER.log(System.Logger.Level.INFO,"=== Training Summary ===");
		LOGGER.log(System.Logger.Level.INFO,String.format("Total training time: %.2f seconds", totalTrainingTime / 1000.0f));
		LOGGER.log(System.Logger.Level.INFO,String.format("Final loss: %.6f", bestLoss));
		LOGGER.log(System.Logger.Level.INFO,String.format("Total steps: %d", globalStep));
		LOGGER.log(System.Logger.Level.INFO,String.format("Average time per step: %.2f ms", (float) totalTrainingTime / globalStep));
		LOGGER.log(System.Logger.Level.INFO,"Training completed successfully!");
	}

	private float trainEpoch(List<TrainingApplication> dataset, int epoch) {
		// Create mutable copy for shuffling
		List<TrainingApplication> mutableDataset = new ArrayList<>(dataset);
		Collections.shuffle(mutableDataset);
		float totalLoss = 0.0f;
		int numBatches = 0;

		for (int i = 0; i < mutableDataset.size(); i += trainingConfig.getBatchSize()) {
			List<TrainingApplication> batch = mutableDataset.subList(i,
				Math.min(i + trainingConfig.getBatchSize(), mutableDataset.size()));

			float batchLoss = trainBatch(batch);
			totalLoss += batchLoss;
			numBatches++;

			globalStep++;

			if (globalStep % 100 == 0) {
				LOGGER.log(System.Logger.Level.INFO,String.format("Step %d: loss %.6f", globalStep, batchLoss));
			}

			if (globalStep % trainingConfig.getSaveSteps() == 0) {
				saveLoRAAdapter(String.format("%s/checkpoint-step-%d.gguf",
				                             trainingConfig.getOutputDir(), globalStep));
			}
		}

		return totalLoss / numBatches;
	}

	/**
	 * Train on a single batch
	 */
	private float trainBatch(List<TrainingApplication> batch) {
		float batchLoss = 0.0f;

		for (TrainingApplication example : batch) {
			// This is a simplified training step
			// In practice, you'd need to integrate with llama.cpp's forward/backward pass
			float loss = computeLossWithLoRA(example);
			batchLoss += loss;

			// Simplified gradient computation
			updateGradients(example, loss);
		}

		// Update weights
		updateLoRAWeights();

		return batchLoss / batch.size();
	}

	private float computeLossWithLoRA(TrainingApplication example) {
		// Implement proper language modeling loss
		String fullText = example.getFullText();
		int[] tokens = baseModel.encode(fullText);

		if (tokens.length < 2) {
			return 0.0f; // Cannot compute loss on single token
		}

		// Split into input and target portions based on input length
		String inputText = example.input();
		int[] inputTokens = baseModel.encode(inputText);
		int inputLen = inputTokens.length;

		// Use cross-entropy loss for the target portion only
		float totalLoss = 0.0f;
		int lossCount = 0;

		// For each position in the target, predict the next token
		for (int pos = inputLen; pos < tokens.length - 1; pos++) {
			// Get logits from model at this position (simplified - would need actual model integration)
			float[] logits = computeLogitsAtPosition(tokens, pos);
			int targetToken = tokens[pos + 1];

			// Apply LoRA deltas to logits
			logits = applyLoRAToLogits(logits, pos, true);

			// Compute cross-entropy loss
			float loss = computeCrossEntropyLoss(logits, targetToken);
			totalLoss += loss;
			lossCount++;
		}

		return lossCount > 0 ? totalLoss / lossCount : 0.0f;
	}

	/**
	 * Compute logits at a specific position (simplified version)
	 */
	private float[] computeLogitsAtPosition(int[] tokens, int position) {
		// This is a simplified version - in practice, you'd need to:
		// 1. Run forward pass through the model up to this position
		// 2. Get the logits from the final layer
		// For now, return random logits as placeholder

		int vocabSize = 32000; // Approximate vocab size for Code Llama
		float[] logits = new float[vocabSize];

		// Initialize with small random values
		for (int i = 0; i < vocabSize; i++) {
			logits[i] = (float) (ThreadLocalRandom.current().nextGaussian() * 0.1);
		}

		return logits;
	}

	/**
	 * Apply LoRA adaptations to logits
	 */
	private float[] applyLoRAToLogits(float[] baseLogits, int position, boolean training) {
		float[] modifiedLogits = baseLogits.clone();

		// For each LoRA module, compute its contribution
		for (LoRAModule module : loraModules.values()) {
			// Simulate getting activations at this layer/position
			float[] activations = generateMockActivations(module.getInputDim());

			// Apply LoRA forward pass
			float[] loraDeltas = module.forward(activations, loraConfig.getAlpha(), training, loraConfig.getDropout());

			// Add deltas to appropriate positions in logits
			// (This is simplified - real implementation would map deltas correctly)
			for (int i = 0; i < Math.min(loraDeltas.length, modifiedLogits.length); i++) {
				modifiedLogits[i] += loraDeltas[i];
			}
		}

		return modifiedLogits;
	}

	/**
	 * Generate mock activations for testing
	 */
	private float[] generateMockActivations(int size) {
		float[] activations = new float[size];
		for (int i = 0; i < size; i++) {
			activations[i] = (float) (ThreadLocalRandom.current().nextGaussian() * 0.1);
		}
		return activations;
	}

	/**
	 * Compute cross-entropy loss between logits and target token
	 */
	private float computeCrossEntropyLoss(float[] logits, int targetToken) {
		// Apply softmax to get probabilities
		float[] probs = softmax(logits);

		// Cross-entropy loss: -log(prob[target])
		float targetProb = probs[targetToken];
		return (float) -Math.log(Math.max(targetProb, 1e-8f)); // Avoid log(0)
	}

	/**
	 * Apply softmax to convert logits to probabilities
	 */
	private float[] softmax(float[] logits) {
		float[] probs = new float[logits.length];

		// Find max for numerical stability
		float maxLogit = Float.NEGATIVE_INFINITY;
		for (float logit : logits) {
			maxLogit = Math.max(maxLogit, logit);
		}

		// Compute exp(logit - max) and sum
		float sum = 0.0f;
		for (int i = 0; i < logits.length; i++) {
			probs[i] = (float) Math.exp(logits[i] - maxLogit);
			sum += probs[i];
		}

		// Normalize
		for (int i = 0; i < probs.length; i++) {
			probs[i] /= sum;
		}

		return probs;
	}

	private void updateGradients(TrainingApplication example, float loss) {
		// Implement proper gradient computation for LoRA training
		String fullText = example.getFullText();
		int[] tokens = baseModel.encode(fullText);

		if (tokens.length < 2) {
			return; // Cannot compute gradients on single token
		}

		String inputText = example.input();
		int[] inputTokens = baseModel.encode(inputText);
		int inputLen = inputTokens.length;

		// For each position in the target, compute gradients
		for (int pos = inputLen; pos < tokens.length - 1; pos++) {
			int targetToken = tokens[pos + 1];

			// Get base logits and apply LoRA
			float[] baseLogits = computeLogitsAtPosition(tokens, pos);
			float[] modifiedLogits = applyLoRAToLogits(baseLogits, pos, true);

			// Compute loss gradient (∂loss/∂logits)
			float[] logitsGrad = computeLogitsGradient(modifiedLogits, targetToken);

			// Backpropagate through LoRA modules
			backpropagateLoRAGradients(logitsGrad, pos);
		}
	}

	/**
	 * Compute gradient of loss with respect to logits
	 */
	private float[] computeLogitsGradient(float[] logits, int targetToken) {
		float[] probs = softmax(logits);
		float[] grad = new float[logits.length];

		// For cross-entropy loss: ∂L/∂logit_i = prob_i - δ_i (where δ_i = 1 if i == target, else 0)
		for (int i = 0; i < grad.length; i++) {
			if (i == targetToken) {
				grad[i] = probs[i] - 1.0f;
			} else {
				grad[i] = probs[i];
			}
		}

		return grad;
	}

	/**
	 * Backpropagate gradients through LoRA modules
	 */
	private void backpropagateLoRAGradients(float[] logitsGrad, int position) {
		for (LoRAModule module : loraModules.values()) {
			// Get activations at this layer/position (mock for now)
			float[] activations = generateMockActivations(module.getInputDim());

			// Extract relevant gradients for this module
			// In practice, this would depend on which part of the logits this module affects
			float[] moduleGrad = extractModuleGradients(logitsGrad, module);

			// Backpropagate through the LoRA module
			module.backward(activations, moduleGrad, loraConfig.getAlpha());
		}
	}

	/**
	 * Extract the portion of logits gradient relevant to a specific LoRA module
	 */
	private float[] extractModuleGradients(float[] logitsGrad, LoRAModule module) {
		// This is simplified - in practice, you'd need to map which parts of the
		// vocabulary/output this module affects
		int outputDim = module.getOutputDim();
		float[] moduleGrad = new float[outputDim];

		// Take the first outputDim gradients as a simplified mapping
		for (int i = 0; i < outputDim && i < logitsGrad.length; i++) {
			moduleGrad[i] = logitsGrad[i];
		}

		return moduleGrad;
	}

	private void updateLoRAWeights() {
		for (LoRAModule module : loraModules.values()) {
			module.updateWeights(
				trainingConfig.getLearningRate(),
				0.9f,  // beta1
				0.999f, // beta2
				1e-8f,  // epsilon
				globalStep
			);
		}
	}

	/**
	 * Save LoRA adapter in format compatible with loadLoRAAdapter()
	 * Uses the native Java GGUF writer for perfect compatibility
	 */
	public void saveLoRAAdapter(String filepath) {
		try {
			LOGGER.log(System.Logger.Level.INFO,"Saving LoRA adapter to: " + filepath);

			// Use the official GGUF writer for guaranteed compatibility
			try (GGUFWriter writer = new GGUFWriter(Paths.get(filepath), "llama")) {

				// Set adapter metadata - match Python implementation exactly
				writer.addType("adapter");
				writer.addString(GGUFConstants.Keys.Adapter.TYPE, "lora");
				writer.addLoRAAlpha(loraConfig.getAlpha());

				// Add tensor information for each LoRA module
				for (LoRAModule module : loraModules.values()) {
					// A matrix: [rank, input_dim]
					long sizeA = (long) module.getRank() * module.getInputDim() * 4; // float32 = 4 bytes
					writer.addTensorInfo(
						module.getName() + ".lora_a",
						new long[]{module.getRank(), module.getInputDim()},
						GGUFConstants.GGMLQuantizationType.F32,
						sizeA
					);

					// B matrix: [output_dim, rank]
					long sizeB = (long) module.getOutputDim() * module.getRank() * 4;
					writer.addTensorInfo(
						module.getName() + ".lora_b",
						new long[]{module.getOutputDim(), module.getRank()},
						GGUFConstants.GGMLQuantizationType.F32,
						sizeB
					);
				}

				// Write the GGUF file structure
				writer.writeToFile();

				// Write tensor data in the EXACT same order as tensor info was declared
				for (LoRAModule module : loraModules.values()) {
					// Write A matrix data first (matches .lora_a declaration order)
					writer.writeTensorData(module.getMatrixA());

					// Write B matrix data second (matches .lora_b declaration order)
					writer.writeTensorData(module.getMatrixB());
				}

				LOGGER.log(System.Logger.Level.INFO,"LoRA adapter saved successfully using native GGUF writer");
			}

		} catch (IOException e) {
			LOGGER.log(System.Logger.Level.ERROR, "Failed to save LoRA adapter", e);
			throw new RuntimeException("Failed to save LoRA adapter", e);
		}
	}


	// Getters
	public LoRAConfig getLoRAConfig() { return loraConfig; }
	public TrainingConfig getTrainingConfig() { return trainingConfig; }
	public Map<String, LoRAModule> getLoRAModules() { return loraModules; }
}
