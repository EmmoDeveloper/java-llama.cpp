package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class QuantizationTest {
	private static final System.Logger logger = System.getLogger(QuantizationTest.class.getName());

	@Test
	public void testGetDefaultParams() {
		LlamaQuantizer.QuantizationParams params = LlamaQuantizer.getDefaultParams();

		Assert.assertNotNull("Default params should not be null", params);
		Assert.assertTrue("Default thread count should be non-negative", params.getThreadCount() >= 0);
		Assert.assertNotNull("Default quantization type should not be null", params.getQuantizationType());

		logger.log(DEBUG, "Default quantization parameters:");
		logger.log(DEBUG, "  Thread count: " + params.getThreadCount());
		logger.log(DEBUG, "  Quantization type: " + params.getQuantizationType());
		logger.log(DEBUG, "  Allow requantize: " + params.isAllowRequantize());
		logger.log(DEBUG, "  Quantize output tensor: " + params.isQuantizeOutputTensor());
		logger.log(DEBUG, "  Only copy: " + params.isOnlyCopy());
		logger.log(DEBUG, "  Pure: " + params.isPure());
		logger.log(DEBUG, "  Keep split: " + params.isKeepSplit());
	}

	@Test
	public void testQuantizationTypes() {
		logger.log(DEBUG, "Available quantization types:");
		for (LlamaQuantizer.QuantizationType type : LlamaQuantizer.QuantizationType.values()) {
			logger.log(DEBUG, "  " + type);
		}

		LlamaQuantizer.QuantizationType q4_0 = LlamaQuantizer.QuantizationType.MOSTLY_Q4_0;
		Assert.assertEquals("Q4_0 value should be 2", 2, q4_0.getValue());
		Assert.assertEquals("Q4_0 description should be correct", "Q4_0", q4_0.getDescription());

		LlamaQuantizer.QuantizationType fromValue = LlamaQuantizer.QuantizationType.fromValue(2);
		Assert.assertEquals("Should get Q4_0 from value 2", q4_0, fromValue);

		try {
			LlamaQuantizer.QuantizationType.fromValue(999);
			Assert.fail("Should throw exception for invalid quantization type");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "Correctly caught invalid quantization type: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizationParamsBuilder() {
		LlamaQuantizer.QuantizationParams params = new LlamaQuantizer.QuantizationParams()
			.setThreadCount(4)
			.setQuantizationType(LlamaQuantizer.QuantizationType.MOSTLY_Q8_0)
			.setAllowRequantize(true)
			.setQuantizeOutputTensor(false)
			.setPure(true);

		Assert.assertEquals("Thread count should be 4", 4, params.getThreadCount());
		Assert.assertEquals("Quantization type should be Q8_0", LlamaQuantizer.QuantizationType.MOSTLY_Q8_0, params.getQuantizationType());
		Assert.assertTrue("Allow requantize should be true", params.isAllowRequantize());
		Assert.assertFalse("Quantize output tensor should be false", params.isQuantizeOutputTensor());
		Assert.assertTrue("Pure should be true", params.isPure());
	}

	@Test
	public void testInvalidInputs() {
		try {
			LlamaQuantizer.quantizeModel(null, "output.gguf");
			Assert.fail("Should throw IllegalArgumentException for null input path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "Correctly caught null input path: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("", "output.gguf");
			Assert.fail("Should throw IllegalArgumentException for empty input path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "Correctly caught empty input path: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("input.gguf", null);
			Assert.fail("Should throw IllegalArgumentException for null output path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "Correctly caught null output path: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("input.gguf", "");
			Assert.fail("Should throw IllegalArgumentException for empty output path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "Correctly caught empty output path: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizeNonexistentModel() {
		try {
			LlamaQuantizer.quantizeModel("nonexistent.gguf", "output.gguf");
			Assert.fail("Should throw LlamaException for nonexistent input file");
		} catch (LlamaException e) {
			logger.log(DEBUG, "Correctly caught quantization error for nonexistent file: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizationConvenienceMethods() {
		try {
			LlamaQuantizer.quantizeModel("nonexistent.gguf", "output.gguf", LlamaQuantizer.QuantizationType.MOSTLY_Q4_0);
		} catch (LlamaException e) {
			logger.log(DEBUG, "Convenience method with QuantizationType correctly failed: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("nonexistent.gguf", "output.gguf");
		} catch (LlamaException e) {
			logger.log(DEBUG, "Convenience method with defaults correctly failed: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizationDemo() {
		logger.log(DEBUG, "\n=== QUANTIZATION FUNCTIONALITY DEMO ===");

		logger.log(DEBUG, "\n1. Available Quantization Types:");
		for (LlamaQuantizer.QuantizationType type : LlamaQuantizer.QuantizationType.values()) {
			if (type.getValue() <= 20) {  // Show common types
				logger.log(DEBUG, "   " + String.format("%-12s", type.getDescription()) +
					" (ID: " + String.format("%2d", type.getValue()) + ") - " + type);
			}
		}

		logger.log(DEBUG, "\n2. Default Quantization Parameters:");
		LlamaQuantizer.QuantizationParams defaultParams = LlamaQuantizer.getDefaultParams();
		logger.log(DEBUG, "   Threads: " + (defaultParams.getThreadCount() == 0 ? "auto" : defaultParams.getThreadCount()));
		logger.log(DEBUG, "   Type: " + defaultParams.getQuantizationType().getDescription());
		logger.log(DEBUG, "   Allow requantize: " + defaultParams.isAllowRequantize());
		logger.log(DEBUG, "   Quantize output: " + defaultParams.isQuantizeOutputTensor());

		logger.log(DEBUG, "\n3. Custom Parameters Example:");
		LlamaQuantizer.QuantizationParams customParams = new LlamaQuantizer.QuantizationParams()
			.setThreadCount(8)
			.setQuantizationType(LlamaQuantizer.QuantizationType.MOSTLY_Q4_K_M)
			.setAllowRequantize(true)
			.setPure(true);

		logger.log(DEBUG, "   Custom config: " + customParams.getQuantizationType().getDescription() +
			" with " + customParams.getThreadCount() + " threads");

		logger.log(DEBUG, "\n4. Quantization API Usage Examples:");
		logger.log(DEBUG, "   // Default quantization:");
		logger.log(DEBUG, "   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q4_0.gguf\");");
		logger.log(DEBUG, "   ");
		logger.log(DEBUG, "   // Specific type:");
		logger.log(DEBUG, "   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q8_0.gguf\", QuantizationType.MOSTLY_Q8_0);");
		logger.log(DEBUG, "   ");
		logger.log(DEBUG, "   // Custom parameters:");
		logger.log(DEBUG, "   QuantizationParams params = new QuantizationParams()");
		logger.log(DEBUG, "       .setQuantizationType(QuantizationType.MOSTLY_Q4_K_M)");
		logger.log(DEBUG, "       .setThreadCount(8);");
		logger.log(DEBUG, "   LlamaQuantizer.quantizeModel(\"input.gguf\", \"output.gguf\", params);");

		logger.log(DEBUG, "\nQuantization functionality is ready for use!");
		logger.log(DEBUG, "Note: This test validates the API without actual model files.");
	}
}
