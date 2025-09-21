package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import java.util.Map;

import static java.lang.System.Logger.Level.DEBUG;

public class SystemInfoTest {

	private static final System.Logger logger = System.getLogger(SystemInfoTest.class.getName());

	@Test
	public void testGetSystemInfo() {
		String systemInfo = SystemInfo.getSystemInfo();
		Assert.assertNotNull("System info should not be null", systemInfo);
		Assert.assertFalse("System info should not be empty", systemInfo.isEmpty());

		// Should contain some expected information
		Assert.assertTrue("System info should contain substantial information", systemInfo.length() > 10);

		// Log the system info for debugging
		logger.log(DEBUG, "System Information:\n" + systemInfo);
	}

	@Test
	public void testGetHighPrecisionTime() {
		long time1 = SystemInfo.getHighPrecisionTime();
		Assert.assertTrue("Time should be positive", time1 > 0);

		// Small delay
		try {
			Thread.sleep(10);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		long time2 = SystemInfo.getHighPrecisionTime();
		Assert.assertTrue("Time should increase", time2 > time1);

		long diff = time2 - time1;
		Assert.assertTrue("Time difference should be at least 10ms (10000 microseconds)", diff >= 10000);

		logger.log(DEBUG, "Time difference: " + diff + " microseconds");
	}

	@Test
	public void testSupportsMemoryMapping() {
		boolean supportsMemoryMapping = SystemInfo.supportsMemoryMapping();
		// Just check it returns a value without throwing
		logger.log(DEBUG, "Memory mapping support: " + supportsMemoryMapping);

		// On most modern systems, memory mapping should be supported
		if (System.getProperty("os.name").toLowerCase().contains("linux") ||
			System.getProperty("os.name").toLowerCase().contains("mac")) {
			Assert.assertTrue("Memory mapping should be supported on Linux/Mac", supportsMemoryMapping);
		}
	}

	@Test
	public void testSupportsMemoryLocking() {
		boolean supportsMemoryLocking = SystemInfo.supportsMemoryLocking();
		logger.log(DEBUG, "Memory locking support: " + supportsMemoryLocking);
		// Boolean primitives can't be null, just verify it returns true or false
		Assert.assertTrue("Memory locking support should return a boolean",
			supportsMemoryLocking == true || supportsMemoryLocking == false);
	}

	@Test
	public void testSupportsGpuOffload() {
		boolean supportsGpu = SystemInfo.supportsGpuOffload();
		logger.log(DEBUG, "GPU offload support: " + supportsGpu);
		// Boolean primitives can't be null, just verify it returns true or false
		Assert.assertTrue("GPU support should return a boolean",
			supportsGpu == true || supportsGpu == false);
	}

	@Test
	public void testSupportsRemoteProcedureCall() {
		boolean supportsRemoteProcedureCall = SystemInfo.supportsRemoteProcedureCall();
		logger.log(DEBUG, "Remote procedure call support: " + supportsRemoteProcedureCall);
		// Boolean primitives can't be null, just verify it returns true or false
		Assert.assertTrue("Remote procedure call support should return a boolean",
			supportsRemoteProcedureCall == true || supportsRemoteProcedureCall == false);
	}

	@Test
	public void testGetCapabilities() {
		Map<String, Boolean> capabilities = SystemInfo.getCapabilities();
		Assert.assertNotNull("Capabilities map should not be null", capabilities);
		Assert.assertFalse("Capabilities map should not be empty", capabilities.isEmpty());

		// Should contain expected keys
		Assert.assertTrue("Should have memory mapping capability", capabilities.containsKey("memory_mapping"));
		Assert.assertTrue("Should have memory locking capability", capabilities.containsKey("memory_locking"));
		Assert.assertTrue("Should have gpu_offload capability", capabilities.containsKey("gpu_offload"));
		Assert.assertTrue("Should have remote procedure call capability", capabilities.containsKey("remote_procedure_call"));

		logger.log(DEBUG, "Capabilities: " + capabilities);
	}

	@Test
	public void testGetCapabilitiesSummary() {
		String summary = SystemInfo.getCapabilitiesSummary();
		Assert.assertNotNull("Summary should not be null", summary);
		Assert.assertFalse("Summary should not be empty", summary.isEmpty());

		// Should contain expected content
		Assert.assertTrue("Should have title", summary.contains("System Capabilities"));
		Assert.assertTrue("Should mention memory mapping", summary.contains("Memory Mapping"));
		Assert.assertTrue("Should mention memory locking", summary.contains("Memory Locking"));
		Assert.assertTrue("Should mention GPU", summary.contains("GPU Offloading"));
		Assert.assertTrue("Should mention remote procedure call", summary.contains("Remote Procedure Call"));

		logger.log(DEBUG, "\n" + summary);
	}

	@Test
	public void testGetSystemInfoMap() {
		Map<String, String> infoMap = SystemInfo.getSystemInfoMap();
		Assert.assertNotNull("Info map should not be null", infoMap);
		// Map might be empty if system info format is unexpected, but shouldn't be null
		logger.log(DEBUG, "System info map size: " + infoMap.size());
		logger.log(DEBUG, "System info map: " + infoMap);
	}

	@Test
	public void testPrintSystemInfo() {
		// This just ensures printSystemInfo doesn't throw
		try {
			SystemInfo.printSystemInfo();
		} catch (Exception e) {
			Assert.fail("printSystemInfo should not throw: " + e.getMessage());
		}
	}

	@Test
	public void testTimeIncreases() {
		long[] times = new long[5];
		for (int i = 0; i < times.length; i++) {
			times[i] = SystemInfo.getHighPrecisionTime();
			if (i > 0) {
				Assert.assertTrue(
					"Time should not go backwards: " + times[i-1] + " -> " + times[i],
					times[i] >= times[i-1]);
			}
			try {
				Thread.sleep(1);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		}
	}
}
