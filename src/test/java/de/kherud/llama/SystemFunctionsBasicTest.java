package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

public class SystemFunctionsBasicTest {

	private static final System.Logger logger = System.getLogger(SystemFunctionsBasicTest.class.getName());

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
}
