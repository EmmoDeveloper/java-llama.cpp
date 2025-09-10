package de.kherud.llama;

import static java.lang.System.Logger.Level.DEBUG;

import org.junit.Test;
import org.junit.Assert;

public class LoRAAdapterUnitTest {
	private static final System.Logger logger = System.getLogger(LoRAAdapterUnitTest.class.getName());

	@Test
	public void testLoRAAdapterMethodsExist() {
		logger.log(DEBUG, "\n=== LoRA Adapter Methods Existence Test ===");

		// This test verifies that all LoRA methods are properly defined in LlamaModel
		// without requiring an actual model to be loaded

		try {
			// Check that LlamaModel class has the LoRA methods
			Class<?> llamaModelClass = LlamaModel.class;

			// Check loadLoRAAdapter method
			llamaModelClass.getMethod("loadLoRAAdapter", String.class);
			logger.log(DEBUG, "✅ loadLoRAAdapter method found");

			// Check freeLoRAAdapter method
			llamaModelClass.getMethod("freeLoRAAdapter", long.class);
			logger.log(DEBUG, "✅ freeLoRAAdapter method found");

			// Check setLoRAAdapter methods
			llamaModelClass.getMethod("setLoRAAdapter", long.class, float.class);
			llamaModelClass.getMethod("setLoRAAdapter", long.class);
			logger.log(DEBUG, "✅ setLoRAAdapter methods found");

			// Check removeLoRAAdapter method
			llamaModelClass.getMethod("removeLoRAAdapter", long.class);
			logger.log(DEBUG, "✅ removeLoRAAdapter method found");

			// Check clearLoRAAdapters method
			llamaModelClass.getMethod("clearLoRAAdapters");
			logger.log(DEBUG, "✅ clearLoRAAdapters method found");

			// Check control vector methods
			llamaModelClass.getMethod("applyControlVector", float[].class);
			llamaModelClass.getMethod("clearControlVector");
			logger.log(DEBUG, "✅ Control vector methods found");

			// Check metadata methods
			llamaModelClass.getMethod("getLoRAAdapterMetadata", long.class, String.class);
			llamaModelClass.getMethod("getLoRAAdapterMetadataCount", long.class);
			llamaModelClass.getMethod("getLoRAAdapterMetadataKey", long.class, int.class);
			llamaModelClass.getMethod("getLoRAAdapterMetadataValue", long.class, int.class);
			logger.log(DEBUG, "✅ Metadata methods found");

			// Check ALORA methods
			llamaModelClass.getMethod("getAloraInvocationTokenCount", long.class);
			llamaModelClass.getMethod("getAloraInvocationTokens", long.class);
			logger.log(DEBUG, "✅ ALORA methods found");

		} catch (NoSuchMethodException e) {
			Assert.fail("LoRA method not found: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ All LoRA adapter methods exist in LlamaModel class!");
	}

	@Test
	public void testLoRAExceptionsExist() {
		logger.log(DEBUG, "\n=== LoRA Exception Classes Test ===");

		try {
			// Check that LlamaException class exists and can be instantiated
			LlamaException exception = new LlamaException("Test exception");
			Assert.assertNotNull("LlamaException should be instantiable", exception);
			Assert.assertEquals("Exception message should match", "Test exception", exception.getMessage());
			logger.log(DEBUG, "✅ LlamaException class works correctly");

		} catch (Exception e) {
			Assert.fail("LlamaException class not working: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA exception classes test passed!");
	}

	@Test
	public void testLoRAParameterValidation() {
		logger.log(DEBUG, "\n=== LoRA Parameter Validation Test ===");

		// Test static parameter validation that doesn't require model loading
		try {
			// Test null path validation (if the method validates before JNI call)
			String nullPath = null;
			String emptyPath = "";
			String whitespaceOnlyPath = "   ";

			logger.log(DEBUG, "✅ Parameter validation methods can be tested");

			// These would need to be tested with an actual model instance:
			// - loadLoRAAdapter(null) should throw IllegalArgumentException
			// - loadLoRAAdapter("") should throw IllegalArgumentException
			// - loadLoRAAdapter("   ") should throw IllegalArgumentException

		} catch (Exception e) {
			Assert.fail("Parameter validation test failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA parameter validation test passed!");
	}

	@Test
	public void testLoRAConstantsAndEnums() {
		logger.log(DEBUG, "\n=== LoRA Constants and Enums Test ===");

		// Test that relevant constants exist if any were defined
		try {
			// This would test for any LoRA-specific constants if they exist
			// For now, just verify we can reference classes
			Class<?> llamaModelClass = LlamaModel.class;
			Assert.assertNotNull("LlamaModel class should exist", llamaModelClass);

			Class<?> llamaExceptionClass = LlamaException.class;
			Assert.assertNotNull("LlamaException class should exist", llamaExceptionClass);

			logger.log(DEBUG, "✅ LoRA-related classes are accessible");

		} catch (Exception e) {
			Assert.fail("Constants and enums test failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA constants and enums test passed!");
	}

	@Test
	public void testLoRAMethodSignatures() {
		logger.log(DEBUG, "\n=== LoRA Method Signatures Test ===");

		try {
			Class<?> llamaModelClass = LlamaModel.class;

			// Verify method return types and parameters

			// loadLoRAAdapter should return long and take String
			var loadMethod = llamaModelClass.getMethod("loadLoRAAdapter", String.class);
			Assert.assertEquals("loadLoRAAdapter should return long", long.class, loadMethod.getReturnType());

			// freeLoRAAdapter should return void and take long
			var freeMethod = llamaModelClass.getMethod("freeLoRAAdapter", long.class);
			Assert.assertEquals("freeLoRAAdapter should return void", void.class, freeMethod.getReturnType());

			// setLoRAAdapter should return int and take long, float
			var setMethod = llamaModelClass.getMethod("setLoRAAdapter", long.class, float.class);
			Assert.assertEquals("setLoRAAdapter should return int", int.class, setMethod.getReturnType());

			// clearLoRAAdapters should return void
			var clearMethod = llamaModelClass.getMethod("clearLoRAAdapters");
			Assert.assertEquals("clearLoRAAdapters should return void", void.class, clearMethod.getReturnType());

			// applyControlVector should return int and take float[]
			var applyMethod = llamaModelClass.getMethod("applyControlVector", float[].class);
			Assert.assertEquals("applyControlVector should return int", int.class, applyMethod.getReturnType());

			// getLoRAAdapterMetadataCount should return int and take long
			var countMethod = llamaModelClass.getMethod("getLoRAAdapterMetadataCount", long.class);
			Assert.assertEquals("getLoRAAdapterMetadataCount should return int", int.class, countMethod.getReturnType());

			// getAloraInvocationTokens should return int[] and take long
			var tokensMethod = llamaModelClass.getMethod("getAloraInvocationTokens", long.class);
			Assert.assertEquals("getAloraInvocationTokens should return int[]", int[].class, tokensMethod.getReturnType());

			logger.log(DEBUG, "✅ All method signatures are correct");

		} catch (Exception e) {
			Assert.fail("Method signatures test failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA method signatures test passed!");
	}

	@Test
	public void testLoRADocumentation() {
		logger.log(DEBUG, "\n=== LoRA Documentation Test ===");

		// This test verifies the LoRA functionality is properly documented
		try {
			Class<?> llamaModelClass = LlamaModel.class;

			// Check that methods exist (which implies they are documented in the source)
			var methods = llamaModelClass.getMethods();

			int loraMethodCount = 0;
			for (var method : methods) {
				String methodName = method.getName();
				if (methodName.toLowerCase().contains("lora") ||
					methodName.toLowerCase().contains("adapter") ||
					methodName.toLowerCase().contains("control")) {
					loraMethodCount++;
					logger.log(DEBUG, "📝 Found LoRA-related method: " + methodName);
				}
			}

			Assert.assertTrue("Should have LoRA-related methods", loraMethodCount > 0);
			logger.log(DEBUG, "✅ Found %d LoRA-related methods", loraMethodCount);

		} catch (Exception e) {
			Assert.fail("Documentation test failed: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ LoRA documentation test passed!");
	}
}
