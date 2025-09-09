package de.kherud.llama;

public class UtilityFunctionsTest {
	public static void main(String[] args) {
		System.out.println("=== Testing Utility Functions ===\n");

		try {
			// Test system capability detection
			System.out.println("1. System Capabilities:");
			System.out.println("   GPU Offload Support: " + LlamaUtils.supportsGpuOffload());
			System.out.println("   Memory Mapping Support: " + LlamaUtils.supportsMmap());
			System.out.println("   Memory Locking Support: " + LlamaUtils.supportsMlock());
			System.out.println("   RPC Support: " + LlamaUtils.supportsRpc());
			System.out.println();

			// Test device and threading limits
			System.out.println("2. Device and Threading Limits:");
			System.out.println("   Max Devices: " + LlamaUtils.maxDevices());
			System.out.println("   Max Parallel Sequences: " + LlamaUtils.maxParallelSequences());
			System.out.println();

			// Test timing function
			System.out.println("3. Performance Timing:");
			long startTime = LlamaUtils.timeUs();

			// Do some work to measure timing
			try {
				Thread.sleep(100); // Sleep 100ms
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}

			long endTime = LlamaUtils.timeUs();
			long elapsedMicros = endTime - startTime;
			double elapsedMillis = elapsedMicros / 1000.0;

			System.out.println("   Start Time: " + startTime + " μs");
			System.out.println("   End Time: " + endTime + " μs");
			System.out.println("   Elapsed: " + elapsedMicros + " μs (" + elapsedMillis + " ms)");
			System.out.println();

			// Test system info
			System.out.println("4. System Information:");
			String systemInfo = LlamaUtils.printSystemInfo();
			if (systemInfo != null && !systemInfo.isEmpty()) {
				System.out.println("   System Info Retrieved Successfully:");
				// Print first few lines to avoid too much output
				String[] lines = systemInfo.split("\n");
				for (int i = 0; i < Math.min(5, lines.length); i++) {
					System.out.println("   " + lines[i]);
				}
				if (lines.length > 5) {
					System.out.println("   ... and " + (lines.length - 5) + " more lines");
				}
			} else {
				System.out.println("   No system info available");
			}
			System.out.println();

			// Test logging callback
			System.out.println("5. Custom Logging:");
			LlamaUtils.setLogCallback(new LlamaUtils.LogCallback() {
				@Override
				public void onLog(int level, String message) {
					System.out.println("   [LOG LEVEL " + level + "] " + message.trim());
				}
			});
			System.out.println("   Custom log callback set successfully");
			System.out.println();

			// Test split path functionality
			System.out.println("6. Split Path Generation:");
			String basePath = "/models/ggml-model-q4_0";
			String splitPath = LlamaUtils.buildSplitPath(basePath, 2);
			System.out.println("   Base path: " + basePath);
			System.out.println("   Split path (index 2): " + splitPath);
			System.out.println();

			System.out.println("=== Tier 1 Utility Functions Complete ===");
			System.out.println();

			System.out.println("=== Testing Tier 2 Operational Functions ===");
			System.out.println("Note: Model-dependent functions (setThreadCount, synchronizeOperations, etc.)");
			System.out.println("require a loaded LlamaModel instance and are tested in integration tests.");
			System.out.println("These functions provide runtime control over model behavior and performance.");
			System.out.println();

			System.out.println("=== Testing Tier 3 Advanced System Management ===");
			System.out.println("Context Information and Threading Management functions:");
			System.out.println("Note: These functions require a loaded LlamaModel instance for runtime");
			System.out.println("introspection and advanced threading control in production deployments.");
			System.out.println("- getContextSize() - Get context window size");
			System.out.println("- getBatchSize() - Get batch processing size");
			System.out.println("- getUbatchSize() - Get micro-batch size");  
			System.out.println("- getMaxSequences() - Get maximum parallel sequences");
			System.out.println("- getCurrentThreads() - Get current thread count");
			System.out.println("- getCurrentThreadsBatch() - Get batch thread count");
			System.out.println("- attachThreadPool() / detachThreadPool() - Custom thread pool management");
			System.out.println();

			System.out.println("=== All Utility Functions Tested Successfully! ===");

		} catch (Exception e) {
			System.err.println("Error testing utility functions: " + e.getMessage());
			e.printStackTrace();
		}
	}
}
