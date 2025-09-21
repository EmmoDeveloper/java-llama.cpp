package de.kherud.llama.training;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Example demonstrating how to use the native Java LoRA training system.
 *
 * This replaces the broken native llama.cpp training with a pure Java implementation
 * that creates adapters compatible with the existing loadLoRAAdapter() system.
 */
public class LoRATrainingExample {
	private static final Logger LOGGER = Logger.getLogger(LoRATrainingExample.class.getName());

	public static void main(String[] args) {
		try {
			// Example 1: Train on Alpaca-style instruction dataset
			trainInstructionModel();

			// Example 2: Train on conversation dataset
			trainChatModel();

			// Example 3: Train on custom text completion
			trainCompletionModel();

		} catch (Exception e) {
			LOGGER.severe("Training failed: " + e.getMessage());
			e.printStackTrace();
		}
	}

	/**
	 * Example 1: Fine-tune for instruction following
	 */
	public static void trainInstructionModel() throws Exception {
		LOGGER.info("=== Training Instruction-Following Model ===");

		// Load base model
		ModelParameters params = new ModelParameters()
			.setModel("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
			.setCtxSize(2048)
			.setGpuLayers(40); // Use GPU acceleration

		try (LlamaModel model = new LlamaModel(params)) {

			// Configure LoRA
			LoRATrainer.LoRAConfig loraConfig = LoRATrainer.LoRAConfig.builder()
				.rank(16)                    // Moderate rank for good performance/memory balance
				.alpha(32.0f)               // 2x rank scaling
				.dropout(0.1f)              // Regularization
				.targetModules("q_proj", "k_proj", "v_proj", "o_proj") // Attention modules
				.maxSequenceLength(2048)
				.build();

			// Configure training
			LoRATrainer.TrainingConfig trainingConfig = LoRATrainer.TrainingConfig.builder()
				.epochs(3)
				.batchSize(4)               // Adjust based on VRAM (RTX 3080 = 16GB)
				.learningRate(2e-4f)        // Standard LoRA learning rate
				.warmupSteps(100)
				.saveSteps(500)
				.outputDir("./output/instruction_lora")
				.build();

			// Load training dataset
			List<TrainingApplication> dataset = DatasetProcessor.loadAlpacaDataset("datasets/alpaca_data.json");
			dataset = DatasetProcessor.filterByLength(dataset, 1800); // Leave room for response

			// Split train/validation
			Map<String, List<TrainingApplication>> split = DatasetProcessor.trainValidationSplit(dataset, 0.1f);

			// Create trainer and train
			LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
			trainer.train(split.get("train"));

			LOGGER.info("Instruction model training completed!");
			LOGGER.info("LoRA adapter saved to: ./output/instruction_lora/final-adapter.gguf");
			LOGGER.info("Load with: model.loadLoRAAdapter(\"./output/instruction_lora/final-adapter.gguf\")");
		}
	}

	/**
	 * Example 2: Fine-tune for chat conversations
	 */
	public static void trainChatModel() throws Exception {
		LOGGER.info("=== Training Chat Model ===");

		ModelParameters params = new ModelParameters()
			.setModel("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
			.setCtxSize(4096)  // Longer context for conversations
			.setGpuLayers(40);

		try (LlamaModel model = new LlamaModel(params)) {

			// LoRA config optimized for chat
			LoRATrainer.LoRAConfig loraConfig = LoRATrainer.LoRAConfig.builder()
				.rank(32)                   // Higher rank for conversational nuances
				.alpha(64.0f)
				.dropout(0.05f)             // Lower dropout for chat
				.targetModules("q_proj", "k_proj", "v_proj", "o_proj",
				              "gate_proj", "up_proj", "down_proj") // Include FFN
				.maxSequenceLength(4096)
				.build();

			LoRATrainer.TrainingConfig trainingConfig = LoRATrainer.TrainingConfig.builder()
				.epochs(2)                  // Fewer epochs for chat to avoid overfitting
				.batchSize(2)               // Smaller batch for longer sequences
				.learningRate(1e-4f)        // Lower LR for stability
				.outputDir("./output/chat_lora")
				.build();

			// Load conversation dataset
			List<TrainingApplication> dataset = DatasetProcessor.loadConversationDataset("datasets/conversations.json");
			dataset = DatasetProcessor.filterByLength(dataset, 3500);

			Map<String, List<TrainingApplication>> split = DatasetProcessor.trainValidationSplit(dataset, 0.15f);

			LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
			trainer.train(split.get("train"));

			LOGGER.info("Chat model training completed!");
		}
	}

	/**
	 * Example 3: Fine-tune for code completion
	 */
	public static void trainCompletionModel() throws Exception {
		LOGGER.info("=== Training Code Completion Model ===");

		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setCtxSize(8192)  // Long context for code
			.setGpuLayers(35);

		try (LlamaModel model = new LlamaModel(params)) {

			// LoRA config for code completion
			LoRATrainer.LoRAConfig loraConfig = LoRATrainer.LoRAConfig.builder()
				.rank(64)                   // High rank for code complexity
				.alpha(128.0f)
				.dropout(0.0f)              // No dropout for deterministic code
				.targetModules("q_proj", "k_proj", "v_proj", "o_proj")
				.maxSequenceLength(8192)
				.gradientCheckpointing(true) // Memory optimization for long sequences
				.build();

			LoRATrainer.TrainingConfig trainingConfig = LoRATrainer.TrainingConfig.builder()
				.epochs(1)                  // Single epoch for large code datasets
				.batchSize(1)               // Very long sequences
				.learningRate(5e-5f)        // Conservative LR for code
				.saveSteps(1000)
				.outputDir("./output/code_lora")
				.build();

			// Create code completion examples
			List<TrainingApplication> dataset = DatasetProcessor.loadTextDataset(
				"datasets/java_code.txt", 6000, 500);

			LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
			trainer.train(dataset);

			LOGGER.info("Code completion model training completed!");
		}
	}

	/**
	 * Example 4: Load and test trained LoRA adapter
	 */
	public static void testTrainedAdapter() throws Exception {
		LOGGER.info("=== Testing Trained LoRA Adapter ===");

		ModelParameters params = new ModelParameters()
			.setModel("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
			.setCtxSize(2048)
			.setGpuLayers(40);

		try (LlamaModel model = new LlamaModel(params)) {

			// Load the trained LoRA adapter
			long adapterHandle = model.loadLoRAAdapter("./output/instruction_lora/final-adapter.gguf");
			model.setLoRAAdapter(adapterHandle, 1.0f); // Full strength

			// Test the fine-tuned model
			String testPrompt = "Below is an instruction that describes a task. " +
				"Write a response that appropriately completes the request.\n\n" +
				"### Instruction:\nExplain quantum computing in simple terms.\n\n" +
				"### Response:\n";

			LOGGER.info("Testing with prompt: " + testPrompt);

			// Generate response using the LoRA-adapted model
			de.kherud.llama.InferenceParameters inferParams = new de.kherud.llama.InferenceParameters(testPrompt)
				.setNPredict(200)
				.setTemperature(0.7f);

			String response = model.complete(inferParams);
			LOGGER.info("LoRA-adapted response: " + response);

			// Clean up
			model.removeLoRAAdapter(adapterHandle);
			model.freeLoRAAdapter(adapterHandle);
		}
	}

	/**
	 * Example 5: Create training data from existing conversations
	 */
	public static void createTrainingDataset() {
		LOGGER.info("=== Creating Training Dataset ===");

		// Example of creating structured training data
		List<TrainingApplication> examples = List.of(
			TrainingApplication.instructionFormat(
				"Translate the following English text to French",
				"Hello, how are you today?",
				"Bonjour, comment allez-vous aujourd'hui?"
			),
			TrainingApplication.chatFormat(
				"You are a helpful coding assistant",
				"How do I reverse a string in Java?",
				"You can reverse a string in Java using StringBuilder:\n\n" +
				"```java\nString reversed = new StringBuilder(original).reverse().toString();\n```"
			),
			TrainingApplication.completionFormat(
				"The capital of France is",
				"Paris"
			)
		);

		try {
			DatasetProcessor.saveAsJsonl(examples, "datasets/custom_training.jsonl");
			LOGGER.info("Training dataset saved to datasets/custom_training.jsonl");
		} catch (Exception e) {
			LOGGER.severe("Failed to save dataset: " + e.getMessage());
		}
	}
}
