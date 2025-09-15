package de.kherud.llama;

import java.util.HashMap;
import java.util.Map;

public class SystemInfo {
	static { LlamaLoader.initialize(); }

	private SystemInfo() {}

	public static String getSystemInfo() {
		return getSystemInfoNative();
	}

	public static long getHighPrecisionTime() {
		return getHighPrecisionTimeNative();
	}

	public static boolean supportsMemoryMapping() {
		return supportsMemoryMappingNative();
	}

	public static boolean supportsMemoryLocking() {
		return supportsMemoryLockingNative();
	}

	public static boolean supportsGpuOffload() {
		return supportsGpuOffloadNative();
	}

	public static boolean supportsRemoteProcedureCall() {
		return supportsRemoteProcedureCallNative();
	}

	public static Map<String, Boolean> getCapabilities() {
		Map<String, Boolean> capabilities = new HashMap<>();
		capabilities.put("memory_mapping", supportsMemoryMapping());
		capabilities.put("memory_locking", supportsMemoryLocking());
		capabilities.put("gpu_offload", supportsGpuOffload());
		capabilities.put("remote_procedure_call", supportsRemoteProcedureCall());
		return capabilities;
	}

	public static void printSystemInfo() {
		System.out.println(getSystemInfo());
	}

	public static Map<String, String> getSystemInfoMap() {
		String systemInfo = getSystemInfo();
		Map<String, String> infoMap = new HashMap<>();

		String[] lines = systemInfo.split("\n");
		for (String line : lines) {
			line = line.trim();
			if (line.isEmpty()) continue;

			int colonIndex = line.indexOf(':');
			int equalsIndex = line.indexOf('=');

			if (colonIndex > 0) {
				String key = line.substring(0, colonIndex).trim();
				String value = line.substring(colonIndex + 1).trim();
				infoMap.put(key, value);
			} else if (equalsIndex > 0) {
				String key = line.substring(0, equalsIndex).trim();
				String value = line.substring(equalsIndex + 1).trim();
				infoMap.put(key, value);
			}
		}

		return infoMap;
	}

	public static String getCapabilitiesSummary() {
		return "System Capabilities:\n" +
			"├─ Memory Mapping: " + (supportsMemoryMapping() ? "✓" : "✗") + "\n" +
			"├─ Memory Locking: " + (supportsMemoryLocking() ? "✓" : "✗") + "\n" +
			"├─ GPU Offloading: " + (supportsGpuOffload() ? "✓" : "✗") + "\n" +
			"└─ Remote Procedure Call: " + (supportsRemoteProcedureCall() ? "✓" : "✗") + "\n";
	}

	private static native String getSystemInfoNative();
	private static native long getHighPrecisionTimeNative();
	private static native boolean supportsMemoryMappingNative();
	private static native boolean supportsMemoryLockingNative();
	private static native boolean supportsGpuOffloadNative();
	private static native boolean supportsRemoteProcedureCallNative();
}
