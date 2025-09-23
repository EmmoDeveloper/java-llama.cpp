package de.kherud.llama.validation;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Test cases for ModelValidator utility.
 */
@Ignore
public class ModelValidatorTest {

	@Rule
	public TemporaryFolder tempDir = new TemporaryFolder();

	private Path testModel1;
	private Path testModel2;
	private ModelValidator validator;

	@Before
	public void setUp() throws IOException {
		testModel1 = tempDir.getRoot().toPath().resolve("model1.gguf");
		testModel2 = tempDir.getRoot().toPath().resolve("model2.gguf");
		createTestModel(testModel1);
		createTestModel(testModel2);

		ModelValidator.ValidationOptions options = new ModelValidator.ValidationOptions();
		validator = new ModelValidator(options);
	}

	@Test
	public void testValidatorCreation() {
		assertNotNull(validator);
	}

	@Test
	public void testValidateModel() {
		ModelValidator.ValidationResult result = validator.validateModel(testModel1);

		assertNotNull(result);
		assertTrue(result.isSuccess());
	}

	@Test
	public void testCompareModels() {
		ModelValidator.ValidationResult result = validator.compareModels(testModel1, testModel2);

		assertNotNull(result);
		// Should complete without throwing exception
	}

	@Test
	public void testBatchValidation() {
		List<Path> models = Arrays.asList(testModel1, testModel2);
		java.util.Map<Path, ModelValidator.ValidationResult> results = validator.validateModels(models);

		assertNotNull(results);
		assertEquals(2, results.size());

		for (ModelValidator.ValidationResult result : results.values()) {
			assertNotNull(result);
		}
	}

	@Test
	public void testValidationWithOptions() {
		ModelValidator.ValidationOptions options = new ModelValidator.ValidationOptions()
			.validateChecksums(false)
			.validateAccuracy(false)
			.verbose(true);

		ModelValidator customValidator = new ModelValidator(options);
		ModelValidator.ValidationResult result = customValidator.validateModel(testModel1);

		assertNotNull(result);
	}

	@Test
	public void testInvalidModelFile() {
		Path invalidModel = tempDir.getRoot().toPath().resolve("nonexistent.gguf");

		ModelValidator.ValidationResult result = validator.validateModel(invalidModel);
		assertNotNull(result);
		assertFalse(result.isSuccess());
	}

	/**
	 * Create a minimal test model file for testing
	 */
	private void createTestModel(Path modelPath) throws IOException {
		// Create a minimal GGUF-like test file
		byte[] testData = {
			'G', 'G', 'U', 'F', // Magic
			0x03, 0x00, 0x00, 0x00, // Version
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Tensor count
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Metadata count
			// Add some dummy metadata and tensor data
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
		};

		Files.write(modelPath, testData);
	}
}
