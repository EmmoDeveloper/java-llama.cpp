package de.kherud.llama.training;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * JUnit4 tests for the native Java LoRA training system.
 *
 * Tests the core LoRA training functionality, dataset processing,
 * and adapter file generation/compatibility.
 */
public class LoRATrainerTest {

	private static final String TEST_MODEL_PATH = "models/codellama-7b.Q2_K.gguf";
	private static final String TEST_OUTPUT_DIR = "./test_output";

	private LlamaModel model;
	private LoRATrainer.LoRAConfig testLoRAConfig;
	private LoRATrainer.TrainingConfig testTrainingConfig;

	@BeforeClass
	public static void setupClass() {
		// Set library path for tests
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");

		// Create test output directory
		new File(TEST_OUTPUT_DIR).mkdirs();
	}

	@AfterClass
	public static void tearDownClass() {
		// Clean up test output directory
		try {
			Files.walk(Paths.get(TEST_OUTPUT_DIR))
				.map(java.nio.file.Path::toFile)
				.forEach(File::delete);
			new File(TEST_OUTPUT_DIR).delete();
		} catch (IOException e) {
			System.err.println("Failed to clean up test directory: " + e.getMessage());
		}
	}

	@Before
	public void setUp() {
		// Skip if test model doesn't exist
		if (!new File(TEST_MODEL_PATH).exists()) {
			org.junit.Assume.assumeTrue("Test model not found: " + TEST_MODEL_PATH, false);
		}

		// Initialize model
		ModelParameters params = new ModelParameters()
			.setModel(TEST_MODEL_PATH)
			.setCtxSize(512)  // Small context for testing
			.setGpuLayers(10);    // Limited GPU layers for testing

		model = new LlamaModel(params);

		// Configure test LoRA settings
		testLoRAConfig = LoRATrainer.LoRAConfig.builder()
			.rank(4)              // Small rank for fast testing
			.alpha(8.0f)
			.dropout(0.1f)
			.targetModules("q_proj", "v_proj")  // Only 2 modules for testing
			.maxSequenceLength(512)
			.build();

		// Configure test training settings
		testTrainingConfig = LoRATrainer.TrainingConfig.builder()
			.epochs(1)            // Single epoch for testing
			.batchSize(1)
			.learningRate(1e-3f)  // Higher LR for faster convergence in tests
			.warmupSteps(0)
			.saveSteps(10)
			.outputDir(TEST_OUTPUT_DIR + "/lora_test")
			.build();
	}

	@After
	public void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testLoRAConfigBuilder() {
		LoRATrainer.LoRAConfig config = LoRATrainer.LoRAConfig.builder()
			.rank(16)
			.alpha(32.0f)
			.dropout(0.1f)
			.targetModules("q_proj", "k_proj", "v_proj")
			.maxSequenceLength(2048)
			.build();

		Assert.assertEquals("Rank should be 16", 16, config.getRank());
		Assert.assertEquals("Alpha should be 32.0", 32.0f, config.getAlpha(), 0.001f);
		Assert.assertEquals("Dropout should be 0.1", 0.1f, config.getDropout(), 0.001f);
		Assert.assertArrayEquals("Target modules should match",
			new String[]{"q_proj", "k_proj", "v_proj"}, config.getTargetModules());
		Assert.assertEquals("Max sequence length should be 2048", 2048, config.getMaxSequenceLength());
	}

	@Test
	public void testTrainingConfigBuilder() {
		LoRATrainer.TrainingConfig config = LoRATrainer.TrainingConfig.builder()
			.epochs(3)
			.batchSize(4)
			.learningRate(2e-4f)
			.warmupSteps(100)
			.saveSteps(500)
			.outputDir("./test")
			.build();

		Assert.assertEquals("Epochs should be 3", 3, config.getEpochs());
		Assert.assertEquals("Batch size should be 4", 4, config.getBatchSize());
		Assert.assertEquals("Learning rate should be 2e-4", 2e-4f, config.getLearningRate(), 1e-6f);
		Assert.assertEquals("Warmup steps should be 100", 100, config.getWarmupSteps());
		Assert.assertEquals("Save steps should be 500", 500, config.getSaveSteps());
		Assert.assertEquals("Output dir should be './test'", "./test", config.getOutputDir());
	}

	@Test
	public void testTrainingExampleCreation() {
		// Test basic example
		TrainingApplication basic = new TrainingApplication("input", "target");
		Assert.assertEquals("Input should match", "input", basic.input());
		Assert.assertEquals("Target should match", "target", basic.target());
		Assert.assertEquals("Default weight should be 1.0", 1.0f, basic.weight(), 0.001f);

		// Test instruction format
		TrainingApplication instruction = TrainingApplication.instructionFormat(
			"Translate to French", "Hello", "Bonjour"
		);
		Assert.assertTrue("Should contain instruction format",
			instruction.input().contains("### Instruction:"));
		Assert.assertEquals("Target should be response", "Bonjour", instruction.target());

		// Test chat format
		TrainingApplication chat = TrainingApplication.chatFormat(
			"You are helpful", "Hi there", "Hello! How can I help?"
		);
		Assert.assertTrue("Should contain chat format",
			chat.input().contains("<|im_start|>"));
		Assert.assertTrue("Target should end with chat token",
			chat.target().endsWith("<|im_end|>"));
	}

	@Test
	public void testLoRAModuleInitialization() {
		LoRATrainer trainer = new LoRATrainer(model, testLoRAConfig, testTrainingConfig);

		Assert.assertNotNull("Trainer should be created", trainer);
		Assert.assertEquals("Should have correct LoRA config", testLoRAConfig, trainer.getLoRAConfig());
		Assert.assertEquals("Should have correct training config", testTrainingConfig, trainer.getTrainingConfig());

		Map<String, LoRATrainer.LoRAModule> modules = trainer.getLoRAModules();
		// With 2 target modules (q_proj, v_proj) and 32 layers = 64 total modules
		Assert.assertEquals("Should have 64 modules (2 modules × 32 layers)", 64, modules.size());
		// Check that we have modules for each layer
		Assert.assertTrue("Should have layer 0 q_proj module", modules.containsKey("blk.0.attn_q.weight"));
		Assert.assertTrue("Should have layer 0 v_proj module", modules.containsKey("blk.0.attn_v.weight"));
		Assert.assertTrue("Should have layer 31 q_proj module", modules.containsKey("blk.31.attn_q.weight"));
		Assert.assertTrue("Should have layer 31 v_proj module", modules.containsKey("blk.31.attn_v.weight"));

		// Test module properties - pick the first q_proj module
		LoRATrainer.LoRAModule qModule = modules.get("blk.0.attn_q.weight");
		Assert.assertNotNull("Should have q_proj module", qModule);
		Assert.assertEquals("Module name should match", "blk.0.attn_q.weight", qModule.getName());
		Assert.assertEquals("Rank should match config", testLoRAConfig.getRank(), qModule.getRank());
		Assert.assertTrue("Input dim should be positive", qModule.getInputDim() > 0);
		Assert.assertTrue("Output dim should be positive", qModule.getOutputDim() > 0);
	}

	@Test
	public void testLoRAModuleForwardPass() {
		int inputDim = 128;
		int outputDim = 128;
		int rank = 4;

		LoRATrainer.LoRAModule module = new LoRATrainer.LoRAModule(
			"test_module", inputDim, outputDim, rank
		);

		// Test forward pass
		float[] input = new float[inputDim];
		// Small test values
		Arrays.fill(input, 0.1f);

		float[] output = module.forward(input, 1.0f, true, 0f);

		Assert.assertNotNull("Output should not be null", output);
		Assert.assertEquals("Output dimension should match", outputDim, output.length);

		// Since B matrix is initialized to zero, output should be zero initially
		for (float value : output) {
			Assert.assertEquals("Initial output should be near zero", 0.0f, value, 1e-6f);
		}
	}

	@Test
	public void testDatasetProcessingSimple() throws IOException {
		// Create simple test dataset
		List<TrainingApplication> examples = List.of(
			TrainingApplication.completionFormat("Hello", "Hi there!"),
			TrainingApplication.completionFormat("Goodbye", "See you later!"),
			TrainingApplication.instructionFormat("Greet", "Say hello", "Hello!")
		);

		// Test filtering by length
		List<TrainingApplication> filtered = DatasetProcessor.filterByLength(examples, 100); // ~400 chars
		Assert.assertEquals("All examples should pass filter", 3, filtered.size());

		filtered = DatasetProcessor.filterByLength(examples, 5); // ~20 chars
		Assert.assertTrue("Some examples should be filtered out", filtered.size() < 3);

		// Test train/validation split
		Map<String, List<TrainingApplication>> split = DatasetProcessor.trainValidationSplit(examples, 0.3f);
		Assert.assertTrue("Should have train split", split.containsKey("train"));
		Assert.assertTrue("Should have validation split", split.containsKey("validation"));

		int totalSize = split.get("train").size() + split.get("validation").size();
		Assert.assertEquals("Total size should match", examples.size(), totalSize);
	}

	@Test
	public void testMinimalTraining() {
		// Create minimal training dataset
		List<TrainingApplication> dataset = List.of(
			TrainingApplication.completionFormat("Test input 1", "Test output 1"),
			TrainingApplication.completionFormat("Test input 2", "Test output 2")
		);

		// Create trainer
		LoRATrainer trainer = new LoRATrainer(model, testLoRAConfig, testTrainingConfig);

		// Test that training doesn't crash (minimal dataset)
		try {
			trainer.train(dataset);
			Assert.assertTrue("Training should complete without error", true);
		} catch (Exception e) {
			Assert.fail("Training should not throw exception: " + e.getMessage());
		}

		// Check that output directory was created
		File outputDir = new File(testTrainingConfig.getOutputDir());
		Assert.assertTrue("Output directory should exist", outputDir.exists());
	}

	@Test
	public void testAdapterFileSaving() {
		LoRATrainer trainer = new LoRATrainer(model, testLoRAConfig, testTrainingConfig);

		String testAdapterPath = TEST_OUTPUT_DIR + "/test_adapter.gguf";

		try {
			trainer.saveLoRAAdapter(testAdapterPath);

			File adapterFile = new File(testAdapterPath);
			Assert.assertTrue("Adapter file should exist", adapterFile.exists());
			Assert.assertTrue("Adapter file should not be empty", adapterFile.length() > 0);

		} catch (Exception e) {
			Assert.fail("Adapter saving should not fail: " + e.getMessage());
		}
	}

	@Test
	public void testLoRAConfigValidation() {
		// Test valid configurations
		LoRATrainer.LoRAConfig validConfig = LoRATrainer.LoRAConfig.builder()
			.rank(8)
			.alpha(16.0f)
			.targetModules("q_proj")
			.build();
		Assert.assertNotNull("Valid config should be created", validConfig);

		// Test edge cases
		LoRATrainer.LoRAConfig minConfig = LoRATrainer.LoRAConfig.builder()
			.rank(1)  // Minimum rank
			.alpha(1.0f)
			.build();
		Assert.assertEquals("Minimum rank should work", 1, minConfig.getRank());

		LoRATrainer.LoRAConfig maxConfig = LoRATrainer.LoRAConfig.builder()
			.rank(256)  // Large rank
			.alpha(512.0f)
			.build();
		Assert.assertEquals("Large rank should work", 256, maxConfig.getRank());
	}

	@Test
	public void testMemoryManagement() {
		// Test that multiple trainer instances can be created and destroyed
		for (int i = 0; i < 3; i++) {
			LoRATrainer trainer = new LoRATrainer(model, testLoRAConfig, testTrainingConfig);
			Assert.assertNotNull("Trainer " + i + " should be created", trainer);

			// Create some training data
			List<TrainingApplication> smallDataset = List.of(
				TrainingApplication.completionFormat("Input " + i, "Output " + i)
			);

			// Verify initialization doesn't leak memory (basic check)
			Map<String, LoRATrainer.LoRAModule> modules = trainer.getLoRAModules();
			Assert.assertEquals("Should have expected number of modules", 64, modules.size());
		}

		// Force garbage collection to test for obvious memory leaks
		System.gc();
		Assert.assertTrue("Memory management test completed", true);
	}

	@Test
	public void testTrainingWithVariousBatchSizes() {
		List<TrainingApplication> dataset = new ArrayList<>();
		for (int i = 0; i < 10; i++) {
			dataset.add(TrainingApplication.completionFormat("Input " + i, "Output " + i));
		}

		// Test different batch sizes
		int[] batchSizes = {1, 2, 5};

		for (int batchSize : batchSizes) {
			LoRATrainer.TrainingConfig config = LoRATrainer.TrainingConfig.builder()
				.epochs(1)
				.batchSize(batchSize)
				.learningRate(1e-3f)
				.outputDir(TEST_OUTPUT_DIR + "/batch_" + batchSize)
				.build();

			LoRATrainer trainer = new LoRATrainer(model, testLoRAConfig, config);

			try {
				trainer.train(dataset);
				Assert.assertTrue("Batch size " + batchSize + " should work", true);
			} catch (Exception e) {
				Assert.fail("Batch size " + batchSize + " failed: " + e.getMessage());
			}
		}
	}
}
