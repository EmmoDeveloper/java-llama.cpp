package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

import static java.lang.System.Logger.Level.DEBUG;

public class LlamaQuantizerTest {

	private static final System.Logger logger = System.getLogger(LlamaQuantizerTest.class.getName());

	@Test
	public void testGetDefaultQuantizationParams() {
		LlamaQuantizer.QuantizationParams params = LlamaQuantizer.getDefaultParams();

		Assert.assertNotNull("Default params should not be null", params);
		Assert.assertTrue("Default thread count should be non-negative", params.getThreadCount() >= 0);
		Assert.assertNotNull("Default quantization type should not be null", params.getQuantizationType());

		logger.log(DEBUG, "Default params - Threads: " + params.getThreadCount() +
				", Type: " + params.getQuantizationType() +
				", Allow Requantize: " + params.isAllowRequantize());
	}

	@Test
	public void testQuantizationTypes() {
		for (LlamaQuantizer.QuantizationType type : LlamaQuantizer.QuantizationType.values()) {
			Assert.assertNotNull("Quantization type description should not be null", type.getDescription());
			Assert.assertTrue("Quantization type value should be non-negative", type.getValue() >= 0);

			LlamaQuantizer.QuantizationType fromValue = LlamaQuantizer.QuantizationType.fromValue(type.getValue());
			Assert.assertEquals("fromValue should return same type", type, fromValue);
		}
	}

	@Test
	public void testQuantizationTypesConversion() {
		LlamaQuantizer.QuantizationType q4_0 = LlamaQuantizer.QuantizationType.MOSTLY_Q4_0;
		Assert.assertEquals("Q4_0 should have correct description", "Q4_0", q4_0.getDescription());
		Assert.assertEquals("Q4_0 should have correct value", 2, q4_0.getValue());

		LlamaQuantizer.QuantizationType q8_0 = LlamaQuantizer.QuantizationType.MOSTLY_Q8_0;
		Assert.assertEquals("Q8_0 should have correct description", "Q8_0", q8_0.getDescription());
		Assert.assertEquals("Q8_0 should have correct value", 7, q8_0.getValue());
	}

	@Test
	public void testQuantizationParamsBuilder() {
		LlamaQuantizer.QuantizationParams params = new LlamaQuantizer.QuantizationParams()
				.setThreadCount(4)
				.setQuantizationType(LlamaQuantizer.QuantizationType.MOSTLY_Q4_0)
				.setAllowRequantize(true)
				.setQuantizeOutputTensor(false);

		Assert.assertEquals("Thread count should be set correctly", 4, params.getThreadCount());
		Assert.assertEquals("Quantization type should be Q4_0", LlamaQuantizer.QuantizationType.MOSTLY_Q4_0, params.getQuantizationType());
		Assert.assertTrue("Allow requantize should be true", params.isAllowRequantize());
		Assert.assertFalse("Quantize output tensor should be false", params.isQuantizeOutputTensor());
	}

	@Test
	public void testParameterValidation() {
		try {
			LlamaQuantizer.quantizeModel(null, "output.gguf");
			Assert.fail("Should throw exception for null input path");
		} catch (IllegalArgumentException e) {
			Assert.assertTrue("Should mention input path", e.getMessage().contains("Input path"));
		}

		try {
			LlamaQuantizer.quantizeModel("input.gguf", null);
			Assert.fail("Should throw exception for null output path");
		} catch (IllegalArgumentException e) {
			Assert.assertTrue("Should mention output path", e.getMessage().contains("Output path"));
		}

		try {
			LlamaQuantizer.quantizeModel("", "output.gguf");
			Assert.fail("Should throw exception for empty input path");
		} catch (IllegalArgumentException e) {
			Assert.assertTrue("Should mention input path", e.getMessage().contains("Input path"));
		}

		try {
			LlamaQuantizer.quantizeModel("input.gguf", "");
			Assert.fail("Should throw exception for empty output path");
		} catch (IllegalArgumentException e) {
			Assert.assertTrue("Should mention output path", e.getMessage().contains("Output path"));
		}
	}

	@Test
	public void testQuantizationTypesToString() {
		LlamaQuantizer.QuantizationType q4_0 = LlamaQuantizer.QuantizationType.MOSTLY_Q4_0;
		String toString = q4_0.toString();

		Assert.assertNotNull("toString should not be null", toString);
		Assert.assertTrue("toString should contain value", toString.contains("2"));

		logger.log(DEBUG, "Q4_0 toString: " + toString);
	}

	@Test
	public void testQuantizationTypeFromInvalidValue() {
		try {
			LlamaQuantizer.QuantizationType.fromValue(-1);
			Assert.fail("Should throw exception for invalid quantization type");
		} catch (IllegalArgumentException e) {
			Assert.assertTrue("Should mention unknown quantization type", e.getMessage().contains("Unknown quantization type"));
		}

		try {
			LlamaQuantizer.QuantizationType.fromValue(9999);
			Assert.fail("Should throw exception for invalid quantization type");
		} catch (IllegalArgumentException e) {
			Assert.assertTrue("Should mention unknown quantization type", e.getMessage().contains("Unknown quantization type"));
		}
	}

	@Test
	public void testQuantizationNonExistentModel() {
		String nonExistentInput = "models/nonexistent.gguf";
		String outputPath = "models/quantized_test.gguf";

		try {
			LlamaQuantizer.quantizeModel(nonExistentInput, outputPath);
			Assert.fail("Should throw exception for non-existent input file");
		} catch (LlamaException e) {
			Assert.assertTrue("Should mention quantization failed", e.getMessage().contains("quantization failed"));
		}
	}

	@Test
	public void testAllQuantizationTypeValues() {
		int[] expectedValues = {0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38};
		LlamaQuantizer.QuantizationType[] types = LlamaQuantizer.QuantizationType.values();

		Assert.assertEquals("Should have expected number of quantization types", expectedValues.length, types.length);

		for (int i = 0; i < types.length; i++) {
			Assert.assertEquals("Type at index " + i + " should have expected value",
					expectedValues[i], types[i].getValue());
		}
	}

	@Test
	public void testQuantizationParamsDefaults() {
		LlamaQuantizer.QuantizationParams params = new LlamaQuantizer.QuantizationParams();

		Assert.assertEquals("Default thread count should be 0", 0, params.getThreadCount());
		Assert.assertEquals("Default quantization type should be Q4_0",
				LlamaQuantizer.QuantizationType.MOSTLY_Q4_0, params.getQuantizationType());
		Assert.assertFalse("Default allow requantize should be false", params.isAllowRequantize());
		Assert.assertTrue("Default quantize output tensor should be true", params.isQuantizeOutputTensor());
		Assert.assertFalse("Default only copy should be false", params.isOnlyCopy());
		Assert.assertFalse("Default pure should be false", params.isPure());
		Assert.assertFalse("Default keep split should be false", params.isKeepSplit());
	}

	@Test
	public void testQuantizationParamsChaining() {
		LlamaQuantizer.QuantizationParams params = new LlamaQuantizer.QuantizationParams()
				.setThreadCount(8)
				.setQuantizationType(LlamaQuantizer.QuantizationType.MOSTLY_Q8_0)
				.setAllowRequantize(true)
				.setQuantizeOutputTensor(false)
				.setOnlyCopy(true)
				.setPure(true)
				.setKeepSplit(true);

		Assert.assertEquals("Thread count should be 8", 8, params.getThreadCount());
		Assert.assertEquals("Quantization type should be Q8_0", LlamaQuantizer.QuantizationType.MOSTLY_Q8_0, params.getQuantizationType());
		Assert.assertTrue("Allow requantize should be true", params.isAllowRequantize());
		Assert.assertFalse("Quantize output tensor should be false", params.isQuantizeOutputTensor());
		Assert.assertTrue("Only copy should be true", params.isOnlyCopy());
		Assert.assertTrue("Pure should be true", params.isPure());
		Assert.assertTrue("Keep split should be true", params.isKeepSplit());
	}
}
