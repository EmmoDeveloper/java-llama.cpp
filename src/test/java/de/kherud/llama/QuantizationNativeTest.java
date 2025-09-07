package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

public class QuantizationNativeTest {

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
}
