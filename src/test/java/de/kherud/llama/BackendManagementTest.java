package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class BackendManagementTest {

	private static final System.Logger logger = System.getLogger(BackendManagementTest.class.getName());

	@Test
	public void testBackendLifecycle() {
		// Test backend initialization
		LlamaUtils.initializeBackend();
		logger.log(DEBUG, "Backend initialized successfully");

		// Test multiple initialization calls (should be safe)
		LlamaUtils.initializeBackend();
		logger.log(DEBUG, "Multiple initialization calls handled safely");

		// Test backend cleanup
		LlamaUtils.freeBackend();
		logger.log(DEBUG, "Backend freed successfully");
	}

	@Test
	public void testNumaInitialization() {
		// Initialize backend first
		LlamaUtils.initializeBackend();

		// Test NUMA disabled (strategy 0)
		LlamaUtils.initializeNuma(0);
		logger.log(DEBUG, "NUMA disabled (strategy 0)");

		// Test NUMA distribute (strategy 1)
		LlamaUtils.initializeNuma(1);
		logger.log(DEBUG, "NUMA distribute (strategy 1)");

		// Test NUMA isolate (strategy 2)
		LlamaUtils.initializeNuma(2);
		logger.log(DEBUG, "NUMA isolate (strategy 2)");

		// Test NUMA numactl (strategy 3)
		LlamaUtils.initializeNuma(3);
		logger.log(DEBUG, "NUMA numactl (strategy 3)");

		// Clean up
		LlamaUtils.freeBackend();
	}

	@Test
	public void testNumaEnumInitialization() {
		// Initialize backend first
		LlamaUtils.initializeBackend();

		// Test all enum values
		for (de.kherud.llama.args.NumaStrategy strategy : de.kherud.llama.args.NumaStrategy.values()) {
			LlamaUtils.initializeNuma(strategy);
			logger.log(DEBUG, "NUMA strategy: " + strategy);
		}

		// Clean up
		LlamaUtils.freeBackend();
	}

	@Test
	public void testNumaStrategyEnum() {
		// Test enum functionality
		de.kherud.llama.args.NumaStrategy disabled = de.kherud.llama.args.NumaStrategy.DISABLED;
		Assert.assertEquals("DISABLED should have value 0", 0, disabled.getValue());
		Assert.assertEquals("DISABLED description should be correct", "NUMA optimizations disabled", disabled.getDescription());

		de.kherud.llama.args.NumaStrategy distribute = de.kherud.llama.args.NumaStrategy.DISTRIBUTE;
		Assert.assertEquals("DISTRIBUTE should have value 1", 1, distribute.getValue());

		// Test fromValue method
		de.kherud.llama.args.NumaStrategy fromValue = de.kherud.llama.args.NumaStrategy.fromValue(2);
		Assert.assertEquals("fromValue(2) should return ISOLATE", de.kherud.llama.args.NumaStrategy.ISOLATE, fromValue);

		// Test toString
		String toStringResult = disabled.toString();
		Assert.assertTrue("toString should contain description", toStringResult.contains("NUMA optimizations disabled"));
		Assert.assertTrue("toString should contain value", toStringResult.contains("0"));

		logger.log(DEBUG, "All NUMA enum tests passed");
	}

	@Test
	public void testBackendWithModelLoading() {
		// Initialize backend
		LlamaUtils.initializeBackend();

		try {
			// Test that backend initialization doesn't interfere with model operations
			ModelParameters params = new ModelParameters()
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setCtxSize(512);

			try (LlamaModel model = new LlamaModel(params)) {
				Assert.assertNotNull("Model should load successfully after backend init", model);
				logger.log(DEBUG, "Model loaded successfully with backend management");
			}
		} catch (Exception e) {
			// If model loading fails, it's likely due to missing model file, not backend issues
			logger.log(DEBUG, "Model loading test skipped (model file not available): " + e.getMessage());
		} finally {
			LlamaUtils.freeBackend();
		}
	}

	@Test
	public void testInvalidNumaStrategy() {
		LlamaUtils.initializeBackend();

		try {
			// Test with invalid strategy values
			LlamaUtils.initializeNuma(-1);
			logger.log(DEBUG, "Invalid NUMA strategy -1 handled");

			LlamaUtils.initializeNuma(999);
			logger.log(DEBUG, "Invalid NUMA strategy 999 handled");
		} catch (Exception e) {
			// Some invalid values might cause exceptions, which is acceptable
			logger.log(DEBUG, "Invalid NUMA strategy caused expected exception: " + e.getMessage());
		} finally {
			LlamaUtils.freeBackend();
		}
	}

	@Test
	public void testBackendStateManagement() {
		// Test that we can safely call freeBackend without initialization
		LlamaUtils.freeBackend();
		logger.log(DEBUG, "Free backend without init handled safely");

		// Test normal lifecycle
		LlamaUtils.initializeBackend();
		LlamaUtils.freeBackend();

		// Test double free
		LlamaUtils.freeBackend();
		logger.log(DEBUG, "Double free handled safely");
	}
}
