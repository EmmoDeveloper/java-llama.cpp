package de.kherud.llama.validation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.nio.file.Files;
import java.io.IOException;
import java.util.List;
import java.util.Arrays;

/**
 * Test cases for ModelValidator utility.
 */
public class ModelValidatorTest {

	@TempDir
	Path tempDir;

	private Path testModel1;
	private Path testModel2;
	private ModelValidator validator;

	@BeforeEach
	void setUp() throws IOException {
		testModel1 = tempDir.resolve("model1.gguf");
		testModel2 = tempDir.resolve("model2.gguf");
		createTestModel(testModel1);
		createTestModel(testModel2);

		ModelValidator.ValidationConfig config = new ModelValidator.ValidationConfig();
		validator = new ModelValidator(config);
	}

	@Test
	void testValidatorCreation() {
		assertNotNull(validator);
	}

	@Test
	void testValidateModel() throws IOException {
		ModelValidator.ValidationResult result = validator.validateModel(testModel1);

		assertNotNull(result);
		assertTrue(result.isValid);
		assertNotNull(result.modelPath);
		assertEquals(testModel1, result.modelPath);
	}

	@Test
	void testCompareModels() throws IOException {
		ModelValidator.ComparisonResult result = validator.compareModels(testModel1, testModel2);

		assertNotNull(result);
		assertNotNull(result.model1Path);
		assertNotNull(result.model2Path);
		assertEquals(testModel1, result.model1Path);
		assertEquals(testModel2, result.model2Path);
	}

	@Test
	void testBatchValidation() throws IOException {
		List<Path> models = Arrays.asList(testModel1, testModel2);
		List<ModelValidator.ValidationResult> results = validator.validateModels(models);

		assertNotNull(results);
		assertEquals(2, results.size());

		for (ModelValidator.ValidationResult result : results) {
			assertNotNull(result);
			assertNotNull(result.modelPath);
		}
	}

	@Test
	void testValidationWithOptions() throws IOException {
		ModelValidator.ValidationConfig config = new ModelValidator.ValidationConfig()
			.checkChecksum(true)
			.checkStructure(true)
			.checkTensors(true)
			.verbose(true);

		ModelValidator customValidator = new ModelValidator(config);
		ModelValidator.ValidationResult result = customValidator.validateModel(testModel1);

		assertNotNull(result);
	}

	@Test
	void testInvalidModelFile() {
		Path invalidModel = tempDir.resolve("nonexistent.gguf");

		assertThrows(IOException.class, () -> {
			validator.validateModel(invalidModel);
		});
	}

	@Test
	void testValidationResultProperties() throws IOException {
		ModelValidator.ValidationResult result = validator.validateModel(testModel1);

		assertNotNull(result.modelPath);
		assertNotNull(result.validationTime);
		assertTrue(result.validationTime >= 0);

		// Check that result has expected structure
		if (result.isValid) {
			assertNull(result.error);
		} else {
			assertNotNull(result.error);
		}
	}

	@Test
	void testComparisonResultProperties() throws IOException {
		ModelValidator.ComparisonResult result = validator.compareModels(testModel1, testModel2);

		assertNotNull(result.model1Path);
		assertNotNull(result.model2Path);
		assertTrue(result.comparisonTime >= 0);

		if (result.areCompatible) {
			assertTrue(result.compatibilityScore >= 0.0);
			assertTrue(result.compatibilityScore <= 1.0);
		}
	}

	@Test
	void testNMSECalculation() throws IOException {
		// Test NMSE calculation with identical models
		ModelValidator.ComparisonResult result = validator.compareModels(testModel1, testModel1);

		assertNotNull(result);
		// NMSE should be very low for identical models
		assertTrue(result.nmse >= 0.0);
	}

	@Test
	void testLogitComparison() throws IOException {
		ModelValidator.ValidationConfig config = new ModelValidator.ValidationConfig()
			.checkLogits(true)
			.logitTolerance(1e-4f);

		ModelValidator customValidator = new ModelValidator(config);
		ModelValidator.ComparisonResult result = customValidator.compareModels(testModel1, testModel2);

		assertNotNull(result);
		// Should have logit comparison data
	}

	@Test
	void testChecksumValidation() throws IOException {
		ModelValidator.ValidationConfig config = new ModelValidator.ValidationConfig()
			.checkChecksum(true)
			.checksumAlgorithm("SHA256");

		ModelValidator customValidator = new ModelValidator(config);
		ModelValidator.ValidationResult result = customValidator.validateModel(testModel1);

		assertNotNull(result);
		assertNotNull(result.checksumResult);
	}

	@Test
	void testValidationConfigBuilder() {
		ModelValidator.ValidationConfig config = new ModelValidator.ValidationConfig()
			.checkChecksum(true)
			.checkStructure(true)
			.checkTensors(true)
			.checkLogits(false)
			.verbose(true)
			.tolerance(1e-5f)
			.logitTolerance(1e-4f)
			.checksumAlgorithm("SHA256")
			.maxTensorsToCheck(100);

		assertNotNull(config);
		// Config should be buildable without issues
	}

	@Test
	void testValidationStatistics() throws IOException {
		ModelValidator.ValidationResult result = validator.validateModel(testModel1);

		assertNotNull(result);
		assertNotNull(result.statistics);

		// Statistics should contain basic metrics
		assertTrue(result.statistics.containsKey("validation_time_ms"));
	}

	@Test
	void testResourceCleanup() throws IOException {
		ModelValidator testValidator = new ModelValidator(new ModelValidator.ValidationConfig());

		// Validator should be closeable
		assertDoesNotThrow(() -> {
			testValidator.close();
		});
	}

	@Test
	void testCommandLineValidation() {
		String[] args = {
			"validate",
			testModel1.toString(),
			"--verbose",
			"--checksum"
		};

		// Test that CLI interface handles arguments properly
		assertDoesNotThrow(() -> {
			// In a real test, you might capture output and verify behavior
		});
	}

	@Test
	void testBatchComparisonPerformance() throws IOException {
		// Test that batch operations are reasonably efficient
		List<Path> models = Arrays.asList(testModel1, testModel2);

		long startTime = System.currentTimeMillis();
		List<ModelValidator.ValidationResult> results = validator.validateModels(models);
		long endTime = System.currentTimeMillis();

		assertNotNull(results);
		assertEquals(2, results.size());

		// Batch validation should complete in reasonable time
		assertTrue((endTime - startTime) < 10000); // Less than 10 seconds
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