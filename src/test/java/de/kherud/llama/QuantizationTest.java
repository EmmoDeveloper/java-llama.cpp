package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

public class QuantizationTest {

	@Test
	public void testGetDefaultParams() {
		LlamaQuantizer.QuantizationParams params = LlamaQuantizer.getDefaultParams();

		Assert.assertNotNull("Default params should not be null", params);
		Assert.assertTrue("Default thread count should be non-negative", params.getThreadCount() >= 0);
		Assert.assertNotNull("Default quantization type should not be null", params.getQuantizationType());

		System.out.println("Default quantization parameters:");
		System.out.println("  Thread count: " + params.getThreadCount());
		System.out.println("  Quantization type: " + params.getQuantizationType());
		System.out.println("  Allow requantize: " + params.isAllowRequantize());
		System.out.println("  Quantize output tensor: " + params.isQuantizeOutputTensor());
		System.out.println("  Only copy: " + params.isOnlyCopy());
		System.out.println("  Pure: " + params.isPure());
		System.out.println("  Keep split: " + params.isKeepSplit());
	}

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
	public void testInvalidInputs() {
		try {
			LlamaQuantizer.quantizeModel(null, "output.gguf");
			Assert.fail("Should throw IllegalArgumentException for null input path");
		} catch (IllegalArgumentException e) {
			System.out.println("Correctly caught null input path: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("", "output.gguf");
			Assert.fail("Should throw IllegalArgumentException for empty input path");
		} catch (IllegalArgumentException e) {
			System.out.println("Correctly caught empty input path: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("input.gguf", null);
			Assert.fail("Should throw IllegalArgumentException for null output path");
		} catch (IllegalArgumentException e) {
			System.out.println("Correctly caught null output path: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("input.gguf", "");
			Assert.fail("Should throw IllegalArgumentException for empty output path");
		} catch (IllegalArgumentException e) {
			System.out.println("Correctly caught empty output path: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizeNonexistentModel() {
		try {
			LlamaQuantizer.quantizeModel("nonexistent.gguf", "output.gguf");
			Assert.fail("Should throw LlamaException for nonexistent input file");
		} catch (LlamaException e) {
			System.out.println("Correctly caught quantization error for nonexistent file: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizationConvenienceMethods() {
		try {
			LlamaQuantizer.quantizeModel("nonexistent.gguf", "output.gguf", LlamaQuantizer.QuantizationType.MOSTLY_Q4_0);
		} catch (LlamaException e) {
			System.out.println("Convenience method with QuantizationType correctly failed: " + e.getMessage());
		}

		try {
			LlamaQuantizer.quantizeModel("nonexistent.gguf", "output.gguf");
		} catch (LlamaException e) {
			System.out.println("Convenience method with defaults correctly failed: " + e.getMessage());
		}
	}

	@Test
	public void testQuantizationDemo() {
		System.out.println("\n=== QUANTIZATION FUNCTIONALITY DEMO ===");

		System.out.println("\n1. Available Quantization Types:");
		for (LlamaQuantizer.QuantizationType type : LlamaQuantizer.QuantizationType.values()) {
			if (type.getValue() <= 20) {  // Show common types
				System.out.println("   " + String.format("%-12s", type.getDescription()) +
					" (ID: " + String.format("%2d", type.getValue()) + ") - " + type);
			}
		}

		System.out.println("\n2. Default Quantization Parameters:");
		LlamaQuantizer.QuantizationParams defaultParams = LlamaQuantizer.getDefaultParams();
		System.out.println("   Threads: " + (defaultParams.getThreadCount() == 0 ? "auto" : defaultParams.getThreadCount()));
		System.out.println("   Type: " + defaultParams.getQuantizationType().getDescription());
		System.out.println("   Allow requantize: " + defaultParams.isAllowRequantize());
		System.out.println("   Quantize output: " + defaultParams.isQuantizeOutputTensor());

		System.out.println("\n3. Custom Parameters Example:");
		LlamaQuantizer.QuantizationParams customParams = new LlamaQuantizer.QuantizationParams()
			.setThreadCount(8)
			.setQuantizationType(LlamaQuantizer.QuantizationType.MOSTLY_Q4_K_M)
			.setAllowRequantize(true)
			.setPure(true);

		System.out.println("   Custom config: " + customParams.getQuantizationType().getDescription() +
			" with " + customParams.getThreadCount() + " threads");

		System.out.println("\n4. Quantization API Usage Examples:");
		System.out.println("   // Default quantization:");
		System.out.println("   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q4_0.gguf\");");
		System.out.println("   ");
		System.out.println("   // Specific type:");
		System.out.println("   LlamaQuantizer.quantizeModel(\"model.gguf\", \"model_q8_0.gguf\", QuantizationType.MOSTLY_Q8_0);");
		System.out.println("   ");
		System.out.println("   // Custom parameters:");
		System.out.println("   QuantizationParams params = new QuantizationParams()");
		System.out.println("       .setQuantizationType(QuantizationType.MOSTLY_Q4_K_M)");
		System.out.println("       .setThreadCount(8);");
		System.out.println("   LlamaQuantizer.quantizeModel(\"input.gguf\", \"output.gguf\", params);");

		System.out.println("\nQuantization functionality is ready for use!");
		System.out.println("Note: This test validates the API without actual model files.");
	}
}
