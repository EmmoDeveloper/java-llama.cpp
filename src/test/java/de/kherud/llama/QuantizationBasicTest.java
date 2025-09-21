package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class QuantizationBasicTest {
	private static final System.Logger logger = System.getLogger(QuantizationBasicTest.class.getName());

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
	public void testInvalidInputsJavaLevel() {
		try {
			LlamaQuantizer.quantizeModel(null, "output.gguf");
			Assert.fail("Should throw IllegalArgumentException for null input path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "Correctly caught null input path: " + e.getMessage());
		} catch (LlamaException e) {
			// This could happen if it tries to call native methods, also acceptable
			logger.log(DEBUG, "Caught at native level: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("", "output.gguf");
			Assert.fail("Should throw IllegalArgumentException for empty input path");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "Correctly caught empty input path: " + e.getMessage());
		} catch (LlamaException e) {
			// This could happen if it tries to call native methods, also acceptable
			logger.log(DEBUG, "Caught at native level: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizationApiDemo() {
		logger.log(DEBUG, "\n=== QUANTIZATION API DEMONSTRATION ===");

		logger.log(DEBUG, "\n1. Available Quantization Types (first 15):");
		LlamaQuantizer.QuantizationType[] types = LlamaQuantizer.QuantizationType.values();
		for (int i = 0; i < Math.min(15, types.length); i++) {
			LlamaQuantizer.QuantizationType type = types[i];
			logger.log(DEBUG, "   " + String.format("%-12s", type.getDescription()) +
				" (ID: " + String.format("%2d", type.getValue()) + ")");
		}

		logger.log(DEBUG, "\n2. Parameter Configuration:");
		LlamaQuantizer.QuantizationParams params = new LlamaQuantizer.QuantizationParams()
			.setThreadCount(8)
			.setQuantizationType(LlamaQuantizer.QuantizationType.MOSTLY_Q4_K_M)
			.setAllowRequantize(true);

		logger.log(DEBUG, "   Thread count: " + params.getThreadCount());
		logger.log(DEBUG, "   Quantization type: " + params.getQuantizationType().getDescription());
		logger.log(DEBUG, "   Allow requantize: " + params.isAllowRequantize());

		logger.log(DEBUG, "\n3. Usage Examples:");
		logger.log(DEBUG, "   // Quick quantization with default settings:");
		logger.log(DEBUG, "   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q4_0.gguf\");");
		logger.log(DEBUG, "   ");
		logger.log(DEBUG, "   // Specific quantization type:");
		logger.log(DEBUG, "   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q8_0.gguf\", MOSTLY_Q8_0);");
		logger.log(DEBUG, "   ");
		logger.log(DEBUG, "   // Custom parameters:");
		logger.log(DEBUG, "   QuantizationParams custom = new QuantizationParams()");
		logger.log(DEBUG, "       .setQuantizationType(MOSTLY_Q4_K_M)");
		logger.log(DEBUG, "       .setThreadCount(8)");
		logger.log(DEBUG, "       .setAllowRequantize(true);");
		logger.log(DEBUG, "   LlamaQuantizer.quantizeModel(\"input.gguf\", \"output.gguf\", custom);");

		logger.log(DEBUG, "\nQuantization API is ready for use!");
	}
}
