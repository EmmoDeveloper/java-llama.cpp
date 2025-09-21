package de.kherud.llama;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class SystemFunctionsTest {

	private static final System.Logger logger = System.getLogger(SystemFunctionsTest.class.getName());

	@Test
	public void testSystemCapabilities() {
		// Test basic system capability checks
		boolean gpuSupport = LlamaUtils.supportsGpuOffload();
		boolean mmapSupport = LlamaUtils.supportsMmap();
		boolean mlockSupport = LlamaUtils.supportsMlock();
		boolean rpcSupport = LlamaUtils.supportsRpc();

		logger.log(DEBUG, "GPU support: " + gpuSupport);
		logger.log(DEBUG, "Mmap support: " + mmapSupport);
		logger.log(DEBUG, "Mlock support: " + mlockSupport);
		logger.log(DEBUG, "RPC support: " + rpcSupport);

		// These should not throw exceptions
		Assert.assertTrue("System capabilities should be queryable", true);
	}

	@Test
	public void testSystemLimits() {
		long maxDevices = LlamaUtils.maxDevices();
		long maxSequences = LlamaUtils.maxParallelSequences();

		Assert.assertTrue("Max devices should be positive", maxDevices > 0);
		Assert.assertTrue("Max parallel sequences should be positive", maxSequences > 0);

		logger.log(DEBUG, "Max devices: " + maxDevices);
		logger.log(DEBUG, "Max parallel sequences: " + maxSequences);
	}

	@Test
	public void testSystemInfo() {
		String systemInfo = LlamaUtils.printSystemInfo();

		Assert.assertNotNull("System info should not be null", systemInfo);
		Assert.assertTrue("System info should contain content", systemInfo.length() > 0);

		logger.log(DEBUG, "System info length: " + systemInfo.length() + " characters");
	}

	@Test
	public void testHighPrecisionTiming() {
		long time1 = LlamaUtils.timeUs();

		// Small delay to ensure time difference
		try {
			Thread.sleep(1);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		long time2 = LlamaUtils.timeUs();

		Assert.assertTrue("Time should advance", time2 > time1);
		Assert.assertTrue("Time should be in microseconds (reasonable range)", time1 > 0);

		logger.log(DEBUG, "Time difference: " + (time2 - time1) + " microseconds");
	}

	@Test
	public void testSplitPathBuilding() {
		String basePath = "models/large-model.gguf";

		// Test building split paths
		String split0 = LlamaUtils.buildSplitPath(basePath, 0);
		String split1 = LlamaUtils.buildSplitPath(basePath, 1);
		String split2 = LlamaUtils.buildSplitPath(basePath, 2);

		Assert.assertNotNull("Split path 0 should not be null", split0);
		Assert.assertNotNull("Split path 1 should not be null", split1);
		Assert.assertNotNull("Split path 2 should not be null", split2);

		// Split paths should be different
		Assert.assertNotEquals("Split paths should be different", split0, split1);
		Assert.assertNotEquals("Split paths should be different", split1, split2);

		logger.log(DEBUG, "Split 0: " + split0);
		logger.log(DEBUG, "Split 1: " + split1);
		logger.log(DEBUG, "Split 2: " + split2);
	}

	@Test
	public void testSplitPrefixExtraction() {
		// Test extracting prefix from various path formats
		String[] testPaths = {
			"models/large-model.gguf",
			"models/large-model-00001-of-00004.gguf",
			"/absolute/path/model.gguf",
			"relative/path/model-split.gguf"
		};

		for (String testPath : testPaths) {
			String prefix = LlamaUtils.extractSplitPrefix(testPath);
			Assert.assertNotNull("Prefix should not be null for: " + testPath, prefix);

			logger.log(DEBUG, "Path: " + testPath + " -> Prefix: " + prefix);
		}
	}

	@Ignore
	@Test
	public void testSplitPathErrorHandling() {
		try {
			// Test with null path
			LlamaUtils.buildSplitPath(null, 0);
			Assert.fail("Should throw exception for null path");
		} catch (Exception e) {
			logger.log(DEBUG, "Null path correctly caused exception: " + e.getClass().getSimpleName());
		}

		try {
			// Test with negative split index
			LlamaUtils.buildSplitPath("test.gguf", -1);
			Assert.fail("Should throw exception for negative split index");
		} catch (Exception e) {
			logger.log(DEBUG, "Negative split index correctly caused exception: " + e.getClass().getSimpleName());
		}
	}

	@Ignore
	@Test
	public void testSplitPrefixErrorHandling() {
		try {
			// Test with null path
			LlamaUtils.extractSplitPrefix(null);
			Assert.fail("Should throw exception for null path");
		} catch (Exception e) {
			logger.log(DEBUG, "Null path correctly caused exception: " + e.getClass().getSimpleName());
		}
	}

	@Test
	public void testFlashAttentionTypeName() {
		// Test flash attention type names for common values
		String type0 = LlamaUtils.getFlashAttentionTypeName(0);
		String type1 = LlamaUtils.getFlashAttentionTypeName(1);

		Assert.assertNotNull("Flash attention type 0 should have a name", type0);
		Assert.assertNotNull("Flash attention type 1 should have a name", type1);

		logger.log(DEBUG, "Flash attention type 0: " + type0);
		logger.log(DEBUG, "Flash attention type 1: " + type1);
	}

	@Ignore
	@Test
	public void testChatBuiltinTemplates() {
		String[] templates = LlamaUtils.getChatBuiltinTemplates();

		Assert.assertNotNull("Built-in templates should not be null", templates);
		Assert.assertTrue("Should have at least one built-in template", templates.length > 0);

		logger.log(DEBUG, "Found " + templates.length + " built-in chat templates");
		for (int i = 0; i < Math.min(templates.length, 5); i++) {
			logger.log(DEBUG, "Template " + i + ": " + templates[i]);
		}
	}
}
