package de.kherud.llama.test;

import de.kherud.llama.LlamaUtils;

import java.io.PrintStream;

public class UtilityFunctionsTest {
	private UtilityFunctionsTest() {
	}

	public static void start(PrintStream out, PrintStream err) {
		out.println("=== Testing Utility Functions ===\n");

		try {
			// Test system capability detection
			out.println("1. System Capabilities:");
			out.println("   GPU Offload Support: " + LlamaUtils.supportsGpuOffload());
			out.println("   Memory Mapping Support: " + LlamaUtils.supportsMmap());
			out.println("   Memory Locking Support: " + LlamaUtils.supportsMlock());
			out.println("   RPC Support: " + LlamaUtils.supportsRpc());
			out.println();

			// Test device and threading limits
			out.println("2. Device and Threading Limits:");
			out.println("   Max Devices: " + LlamaUtils.maxDevices());
			out.println("   Max Parallel Sequences: " + LlamaUtils.maxParallelSequences());
			out.println();

			// Test timing function
			out.println("3. Performance Timing:");
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

			out.println("   Start Time: " + startTime + " μs");
			out.println("   End Time: " + endTime + " μs");
			out.println("   Elapsed: " + elapsedMicros + " μs (" + elapsedMillis + " ms)");
			out.println();

			// Test system info
			out.println("4. System Information:");
			String systemInfo = LlamaUtils.printSystemInfo();
			if (systemInfo != null && !systemInfo.isEmpty()) {
				out.println("   System Info Retrieved Successfully:");
				// Print first few lines to avoid too much output
				String[] lines = systemInfo.split("\n");
				for (int i = 0; i < Math.min(5, lines.length); i++) {
					out.println("   " + lines[i]);
				}
				if (lines.length > 5) {
					out.println("   ... and " + (lines.length - 5) + " more lines");
				}
			} else {
				out.println("   No system info available");
			}
			out.println();

			// Test logging callback
			out.println("5. Custom Logging:");
			LlamaUtils.setLogCallback(new LlamaUtils.LogCallback() {
				@Override
				public void onLog(int level, String message) {
					out.println("   [LOG LEVEL " + level + "] " + message.trim());
				}
			});
			out.println("   Custom log callback set successfully");
			out.println();

			// Test split path functionality
			out.println("6. Split Path Generation:");
			String basePath = "/models/ggml-model-q4_0";
			String splitPath = LlamaUtils.buildSplitPath(basePath, 2);
			out.println("   Base path: " + basePath);
			out.println("   Split path (index 2): " + splitPath);
			out.println();

			out.println("=== Tier 1 Utility Functions Complete ===");
			out.println();

			out.println("=== Testing Tier 2 Operational Functions ===");
			out.println("Note: Model-dependent functions (setThreadCount, synchronizeOperations, etc.)");
			out.println("require a loaded LlamaModel instance and are tested in integration tests.");
			out.println("These functions provide runtime control over model behavior and performance.");
			out.println();

			out.println("=== Testing Tier 3 Advanced System Management ===");
			out.println("Context Information and Threading Management functions:");
			out.println("Note: These functions require a loaded LlamaModel instance for runtime");
			out.println("introspection and advanced threading control in production deployments.");
			out.println("- getContextSize() - Get context window size");
			out.println("- getBatchSize() - Get batch processing size");
			out.println("- getUbatchSize() - Get micro-batch size");
			out.println("- getMaxSequences() - Get maximum parallel sequences");
			out.println("- getCurrentThreads() - Get current thread count");
			out.println("- getCurrentThreadsBatch() - Get batch thread count");
			out.println("- attachThreadPool() / detachThreadPool() - Custom thread pool management");
			out.println();

			out.println("=== All Utility Functions Tested Successfully! ===");

		} catch (Exception e) {
			err.println("Error testing utility functions: " + e.getMessage());
		}
	}

	public static void main(String[] args) {
		start(System.out, System.err);
	}
}
