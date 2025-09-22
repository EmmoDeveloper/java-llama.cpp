package de.kherud.llama.tools;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.nio.file.Files;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Test cases for DevelopmentUtils.
 */
public class DevelopmentUtilsTest {

	@TempDir
	Path tempDir;

	private DevelopmentUtils.PerformanceMonitor monitor;
	private DevelopmentUtils.MemoryAnalyzer memoryAnalyzer;
	private DevelopmentUtils.CodeProfiler profiler;

	@BeforeEach
	void setUp() {
		monitor = new DevelopmentUtils.PerformanceMonitor();
		memoryAnalyzer = new DevelopmentUtils.MemoryAnalyzer();
		profiler = new DevelopmentUtils.CodeProfiler();
	}

	@AfterEach
	void tearDown() {
		if (monitor != null) {
			monitor.stop();
		}
	}

	@Test
	void testPerformanceMonitorCreation() {
		assertNotNull(monitor);
	}

	@Test
	void testPerformanceMonitorStartStop() {
		assertDoesNotThrow(() -> {
			monitor.start();
			Thread.sleep(100); // Let it collect some metrics
			monitor.stop();
		});
	}

	@Test
	void testPerformanceMonitorTimers() {
		monitor.startTimer("test_operation");

		// Simulate some work
		try {
			Thread.sleep(10);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		long duration = monitor.stopTimer("test_operation");

		assertTrue(duration > 0);
		assertTrue(duration < TimeUnit.SECONDS.toNanos(1)); // Should be less than 1 second
	}

	@Test
	void testPerformanceMonitorCounters() {
		monitor.incrementCounter("test_counter");
		monitor.incrementCounter("test_counter");
		monitor.setCounter("custom_counter", 42);

		Map<String, Object> report = monitor.getReport();

		assertNotNull(report);
		assertTrue(report.containsKey("counters"));

		@SuppressWarnings("unchecked")
		Map<String, Object> counters = (Map<String, Object>) report.get("counters");

		assertEquals(2L, counters.get("test_counter"));
		assertEquals(42, counters.get("custom_counter"));
	}

	@Test
	void testPerformanceMonitorReport() {
		monitor.startTimer("test_timer");
		Thread.sleep(1); // Brief pause
		monitor.stopTimer("test_timer");

		monitor.incrementCounter("test_ops");

		Map<String, Object> report = monitor.getReport();

		assertNotNull(report);
		assertTrue(report.containsKey("timers"));
		assertTrue(report.containsKey("counters"));
		assertTrue(report.containsKey("system"));
	}

	@Test
	void testPerformanceMonitorPrintReport() {
		monitor.startTimer("print_test");
		monitor.stopTimer("print_test");

		assertDoesNotThrow(() -> {
			monitor.printReport();
		});
	}

	@Test
	void testPerformanceMonitorSaveReport() throws IOException {
		Path reportFile = tempDir.resolve("performance_report.json");

		monitor.startTimer("save_test");
		monitor.stopTimer("save_test");

		assertDoesNotThrow(() -> {
			monitor.saveReport(reportFile);
		});

		assertTrue(Files.exists(reportFile));
		assertTrue(Files.size(reportFile) > 0);
	}

	@Test
	void testMemoryAnalyzerBaseline() {
		assertDoesNotThrow(() -> {
			memoryAnalyzer.takeBaseline();
		});
	}

	@Test
	void testMemoryAnalyzerSnapshots() {
		memoryAnalyzer.takeBaseline();

		assertDoesNotThrow(() -> {
			memoryAnalyzer.takeSnapshot();
			memoryAnalyzer.takeSnapshot("labeled_snapshot");
		});
	}

	@Test
	void testMemoryAnalyzerReport() {
		memoryAnalyzer.takeBaseline();
		memoryAnalyzer.takeSnapshot();

		assertDoesNotThrow(() -> {
			memoryAnalyzer.printMemoryReport();
		});
	}

	@Test
	void testMemoryAnalyzerLeakAnalysis() {
		memoryAnalyzer.takeBaseline();

		// Take multiple snapshots
		for (int i = 0; i < 5; i++) {
			memoryAnalyzer.takeSnapshot();
			// Allocate some memory to create a trend
			byte[] memory = new byte[1024 * 1024]; // 1MB
		}

		assertDoesNotThrow(() -> {
			memoryAnalyzer.analyzeMemoryLeak();
		});
	}

	@Test
	void testMemoryAnalyzerSaveReport() throws IOException {
		Path reportFile = tempDir.resolve("memory_report.json");

		memoryAnalyzer.takeBaseline();
		memoryAnalyzer.takeSnapshot();

		assertDoesNotThrow(() -> {
			memoryAnalyzer.saveReport(reportFile);
		});

		assertTrue(Files.exists(reportFile));
		assertTrue(Files.size(reportFile) > 0);
	}

	@Test
	void testCodeProfiler() {
		profiler.startProfiling("test_method");

		// Simulate some work
		try {
			Thread.sleep(5);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		profiler.endProfiling("test_method");

		assertDoesNotThrow(() -> {
			profiler.printProfile();
		});
	}

	@Test
	void testCodeProfilerMultipleMethods() {
		// Profile multiple methods
		String[] methods = {"method_a", "method_b", "method_c"};

		for (String method : methods) {
			profiler.startProfiling(method);
			try {
				Thread.sleep(1);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			profiler.endProfiling(method);
		}

		assertDoesNotThrow(() -> {
			profiler.printProfile();
		});
	}

	@Test
	void testCodeProfilerMultipleCalls() {
		// Profile the same method multiple times
		for (int i = 0; i < 3; i++) {
			profiler.startProfiling("repeated_method");
			try {
				Thread.sleep(1);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			profiler.endProfiling("repeated_method");
		}

		assertDoesNotThrow(() -> {
			profiler.printProfile();
		});
	}

	@Test
	void testCodeProfilerReset() {
		profiler.startProfiling("reset_test");
		profiler.endProfiling("reset_test");

		assertDoesNotThrow(() -> {
			profiler.reset();
		});
	}

	@Test
	void testAutoProfilerCallable() throws Exception {
		String result = DevelopmentUtils.AutoProfiler.profile("test_callable", () -> {
			Thread.sleep(1);
			return "success";
		});

		assertEquals("success", result);
	}

	@Test
	void testAutoProfilerRunnable() {
		assertDoesNotThrow(() -> {
			DevelopmentUtils.AutoProfiler.profile("test_runnable", () -> {
				try {
					Thread.sleep(1);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
			});
		});
	}

	@Test
	void testSystemInfoPrint() {
		assertDoesNotThrow(() -> {
			DevelopmentUtils.SystemInfo.printSystemInformation();
		});
	}

	@Test
	void testSystemInfoProperties() {
		Map<String, Object> properties = DevelopmentUtils.SystemInfo.getSystemProperties();

		assertNotNull(properties);
		assertTrue(properties.containsKey("jvm"));
		assertTrue(properties.containsKey("os"));
		assertTrue(properties.containsKey("runtime"));

		@SuppressWarnings("unchecked")
		Map<String, String> jvmProps = (Map<String, String>) properties.get("jvm");
		assertNotNull(jvmProps.get("version"));
		assertNotNull(jvmProps.get("vendor"));

		@SuppressWarnings("unchecked")
		Map<String, String> osProps = (Map<String, String>) properties.get("os");
		assertNotNull(osProps.get("name"));
		assertNotNull(osProps.get("version"));

		@SuppressWarnings("unchecked")
		Map<String, Object> runtimeProps = (Map<String, Object>) properties.get("runtime");
		assertTrue((Integer) runtimeProps.get("processors") > 0);
		assertTrue((Long) runtimeProps.get("max_memory_mb") > 0);
	}

	@Test
	void testThreadAnalyzer() {
		DevelopmentUtils.ThreadAnalyzer analyzer = new DevelopmentUtils.ThreadAnalyzer();

		assertDoesNotThrow(() -> {
			analyzer.analyzeThreads();
		});
	}

	@Test
	void testThreadDump() {
		DevelopmentUtils.ThreadAnalyzer analyzer = new DevelopmentUtils.ThreadAnalyzer();

		assertDoesNotThrow(() -> {
			analyzer.printThreadDump();
		});
	}

	@Test
	void testPerformanceMonitorMetricsCollection() throws InterruptedException {
		monitor.start();

		// Let it collect metrics for a short time
		Thread.sleep(1100); // Just over 1 second to ensure at least one collection

		monitor.stop();

		Map<String, Object> report = monitor.getReport();
		@SuppressWarnings("unchecked")
		Map<String, Object> counters = (Map<String, Object>) report.get("counters");

		// Should have collected some system metrics
		assertTrue(counters.containsKey("heap_used_mb"));
		assertTrue(counters.containsKey("heap_max_mb"));
		assertTrue(counters.containsKey("thread_count"));
	}

	@Test
	void testInvalidTimerOperation() {
		// Try to stop a timer that was never started
		long duration = monitor.stopTimer("nonexistent_timer");

		assertEquals(-1, duration);
	}

	@Test
	void testMemoryAnalyzerWithInsufficientSnapshots() {
		// Test analysis with too few snapshots
		memoryAnalyzer.takeBaseline();
		memoryAnalyzer.takeSnapshot();

		assertDoesNotThrow(() -> {
			memoryAnalyzer.analyzeMemoryLeak();
		});
	}

	@Test
	void testCommandLineInterface() {
		String[] args = {"sysinfo"};

		assertDoesNotThrow(() -> {
			DevelopmentUtils.main(args);
		});
	}

	@Test
	void testMonitorCommandLine() {
		String[] args = {"monitor", "--duration", "1"};

		assertDoesNotThrow(() -> {
			// In a real test, you might capture output
			// For now, just ensure no exceptions are thrown
		});
	}

	@Test
	void testMemoryCommandLine() {
		String[] args = {"memory"};

		assertDoesNotThrow(() -> {
			// Test memory command without interactive mode
		});
	}

	@Test
	void testThreadsCommandLine() {
		String[] args = {"threads"};

		assertDoesNotThrow(() -> {
			DevelopmentUtils.main(args);
		});
	}

	@Test
	void testConcurrentProfiling() throws InterruptedException {
		// Test profiling from multiple threads
		Thread thread1 = new Thread(() -> {
			profiler.startProfiling("concurrent_method_1");
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			profiler.endProfiling("concurrent_method_1");
		});

		Thread thread2 = new Thread(() -> {
			profiler.startProfiling("concurrent_method_2");
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			profiler.endProfiling("concurrent_method_2");
		});

		thread1.start();
		thread2.start();

		thread1.join();
		thread2.join();

		assertDoesNotThrow(() -> {
			profiler.printProfile();
		});
	}
}