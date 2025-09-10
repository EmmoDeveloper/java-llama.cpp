package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Assert;
import org.junit.Test;

public class QuantizationNativeTest {
	private static final System.Logger logger = System.getLogger(QuantizationNativeTest.class.getName());

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
}
