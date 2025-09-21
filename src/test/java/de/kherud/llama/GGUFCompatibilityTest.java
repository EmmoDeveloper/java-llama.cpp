package de.kherud.llama;

import de.kherud.llama.training.LoRATrainer;
import de.kherud.llama.training.TrainingApplication;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.List;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class GGUFCompatibilityTest {
	private LlamaModel model;
	private final String modelPath = "models/codellama-7b.Q2_K.gguf";
	private final String testOutputDir = "/tmp/gguf_compatibility_test";
	private final String adapterPath = testOutputDir + "/test_adapter.gguf";

	@Before
	public void setUp() {
		File outputDir = new File(testOutputDir);
		if (!outputDir.exists()) {
			outputDir.mkdirs();
		}

		try {
			ModelParameters params = new ModelParameters()
				.setModel(modelPath)
				.setCtxSize(512)
				.setGpuLayers(15);
			model = new LlamaModel(params);
		} catch (Exception e) {
			System.err.println("Model loading failed: " + e.getMessage());
			throw new RuntimeException("Cannot run test without model", e);
		}
	}

	@After
	public void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testCompleteLoRAWorkflow() {
		System.out.println("=== GGUF Compatibility Test ===");

		LoRATrainer.LoRAConfig loraConfig = LoRATrainer.LoRAConfig.builder()
			.rank(4)
			.alpha(8.0f)
			.targetModules("q_proj")
			.build();

		LoRATrainer.TrainingConfig trainingConfig = LoRATrainer.TrainingConfig.builder()
			.epochs(1)
			.batchSize(1)
			.outputDir(testOutputDir)
			.build();

		System.out.println("✓ Configurations created");

		LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
		System.out.println("✓ LoRA trainer initialized with " + trainer.getLoRAModules().size() + " modules");

		List<TrainingApplication> dataset = List.of(
			TrainingApplication.completionFormat("def fibonacci(n):", "\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)")
		);

		try {
			trainer.train(dataset);
			System.out.println("✓ Training completed successfully");
		} catch (Exception e) {
			fail("Training failed: " + e.getMessage());
		}

		try {
			trainer.saveLoRAAdapter(adapterPath);
			System.out.println("✓ LoRA adapter saved to GGUF format");
		} catch (Exception e) {
			fail("GGUF save failed: " + e.getMessage());
		}

		File adapterFile = new File(adapterPath);
		assertTrue("Adapter file should exist", adapterFile.exists());
		assertTrue("Adapter file should not be empty", adapterFile.length() > 0);
		System.out.println("✓ Adapter file verified: " + adapterFile.length() + " bytes");

		try {
			long handle = model.loadLoRAAdapter(adapterPath);
			System.out.println("✓ LoRA adapter loaded successfully! Handle: " + handle);

			int result = model.setLoRAAdapter(handle, 1.0f);
			System.out.println("✓ LoRA adapter applied successfully! Result: " + result);

			model.removeLoRAAdapter(handle);
			model.freeLoRAAdapter(handle);
			System.out.println("✓ LoRA adapter cleaned up successfully");

		} catch (Exception e) {
			fail("Adapter loading/usage failed: " + e.getMessage());
		}

		System.out.println("\n🎉 Complete GGUF compatibility test passed!");
		System.out.println("   - Java LoRA training: ✓");
		System.out.println("   - GGUF file generation: ✓");
		System.out.println("   - Native adapter loading: ✓");
		System.out.println("   - Full integration: ✓");
	}

	@Test
	public void testGGUFFileStructure() {
		System.out.println("=== GGUF File Structure Verification ===");

		LoRATrainer.LoRAConfig loraConfig = LoRATrainer.LoRAConfig.builder()
			.rank(2)
			.alpha(4.0f)
			.targetModules("q_proj", "v_proj")
			.build();

		LoRATrainer.TrainingConfig trainingConfig = LoRATrainer.TrainingConfig.builder()
			.epochs(1)
			.batchSize(1)
			.outputDir(testOutputDir)
			.build();

		LoRATrainer trainer = new LoRATrainer(model, loraConfig, trainingConfig);
		String structureTestPath = testOutputDir + "/structure_test.gguf";

		try {
			trainer.saveLoRAAdapter(structureTestPath);
			System.out.println("✓ Multi-module adapter saved");

			File file = new File(structureTestPath);
			assertTrue("Structure test file should exist", file.exists());

			long fileSize = file.length();
			System.out.println("✓ File size: " + fileSize + " bytes");

			assertTrue("File should be reasonable size for 2-rank, 2-module LoRA", fileSize > 1000 && fileSize < 50000);

			long handle = model.loadLoRAAdapter(structureTestPath);
			System.out.println("✓ Multi-module adapter loads correctly! Handle: " + handle);

			model.freeLoRAAdapter(handle);
			System.out.println("✓ Multi-module cleanup successful");

		} catch (Exception e) {
			fail("Structure test failed: " + e.getMessage());
		}

		System.out.println("✓ GGUF structure verification passed");
	}
}
