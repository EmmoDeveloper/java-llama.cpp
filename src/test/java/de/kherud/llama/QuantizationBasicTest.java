package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

public class QuantizationBasicTest {

	@Test
	public void testQuantizationTypes() {
		System.out.println("Available quantization types:");
		for (LlamaQuantizer.QuantizationType type : LlamaQuantizer.QuantizationType.values()) {
			System.out.println("  " + type);
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
			System.out.println("Correctly caught invalid quantization type: " + e.getMessage());
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
			System.out.println("Correctly caught null input path: " + e.getMessage());
		} catch (LlamaException e) {
			// This could happen if it tries to call native methods, also acceptable
			System.out.println("Caught at native level: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("", "output.gguf");
			Assert.fail("Should throw IllegalArgumentException for empty input path");
		} catch (IllegalArgumentException e) {
			System.out.println("Correctly caught empty input path: " + e.getMessage());
		} catch (LlamaException e) {
			// This could happen if it tries to call native methods, also acceptable
			System.out.println("Caught at native level: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizationApiDemo() {
		System.out.println("\n=== QUANTIZATION API DEMONSTRATION ===");

		System.out.println("\n1. Available Quantization Types (first 15):");
		LlamaQuantizer.QuantizationType[] types = LlamaQuantizer.QuantizationType.values();
		for (int i = 0; i < Math.min(15, types.length); i++) {
			LlamaQuantizer.QuantizationType type = types[i];
			System.out.println("   " + String.format("%-12s", type.getDescription()) + 
				" (ID: " + String.format("%2d", type.getValue()) + ")");
		}

		System.out.println("\n2. Parameter Configuration:");
		LlamaQuantizer.QuantizationParams params = new LlamaQuantizer.QuantizationParams()
			.setThreadCount(8)
			.setQuantizationType(LlamaQuantizer.QuantizationType.MOSTLY_Q4_K_M)
			.setAllowRequantize(true);
		
		System.out.println("   Thread count: " + params.getThreadCount());
		System.out.println("   Quantization type: " + params.getQuantizationType().getDescription());
		System.out.println("   Allow requantize: " + params.isAllowRequantize());

		System.out.println("\n3. Usage Examples:");
		System.out.println("   // Quick quantization with default settings:");
		System.out.println("   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q4_0.gguf\");");
		System.out.println("   ");
		System.out.println("   // Specific quantization type:");
		System.out.println("   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q8_0.gguf\", MOSTLY_Q8_0);");
		System.out.println("   ");
		System.out.println("   // Custom parameters:");
		System.out.println("   QuantizationParams custom = new QuantizationParams()");
		System.out.println("       .setQuantizationType(MOSTLY_Q4_K_M)");
		System.out.println("       .setThreadCount(8)");
		System.out.println("       .setAllowRequantize(true);");
		System.out.println("   LlamaQuantizer.quantizeModel(\"input.gguf\", \"output.gguf\", custom);");

		System.out.println("\nQuantization API is ready for use!");
	}
}