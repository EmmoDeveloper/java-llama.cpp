package de.kherud.llama.tools;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.lang.management.*;
import java.nio.file.*;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Development utilities and debugging tools.
 *
 * Equivalent to development and debugging scripts - provides performance monitoring,
 * memory analysis, profiling, and development workflow utilities.
 */
public class DevelopmentUtils {
	private static final Logger LOGGER = Logger.getLogger(DevelopmentUtils.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();
	private static final MemoryMXBean MEMORY_BEAN = ManagementFactory.getMemoryMXBean();
	private static final List<GarbageCollectorMXBean> GC_BEANS = ManagementFactory.getGarbageCollectorMXBeans();
	private static final ThreadMXBean THREAD_BEAN = ManagementFactory.getThreadMXBean();

	public static class PerformanceMonitor {
		private final Map<String, Long> timers = new ConcurrentHashMap<>();
		private final Map<String, List<Long>> measurements = new ConcurrentHashMap<>();
		private final Map<String, Object> counters = new ConcurrentHashMap<>();
		private boolean isRunning = false;
		private ScheduledExecutorService scheduler;

		public void start() {
			if (isRunning) return;

			isRunning = true;
			scheduler = Executors.newScheduledThreadPool(2);

			// Monitor system metrics every second
			scheduler.scheduleAtFixedRate(this::collectSystemMetrics, 0, 1, TimeUnit.SECONDS);

			LOGGER.info("Performance monitor started");
		}

		public void stop() {
			if (!isRunning) return;

			isRunning = false;
			if (scheduler != null) {
				scheduler.shutdown();
			}

			LOGGER.info("Performance monitor stopped");
		}

		public void startTimer(String name) {
			timers.put(name, System.nanoTime());
		}

		public long stopTimer(String name) {
			Long startTime = timers.remove(name);
			if (startTime == null) {
				return -1;
			}

			long duration = System.nanoTime() - startTime;
			measurements.computeIfAbsent(name, k -> new ArrayList<>()).add(duration);
			return duration;
		}

		public void incrementCounter(String name) {
			counters.merge(name, 1L, (old, val) -> ((Long) old) + 1);
		}

		public void setCounter(String name, Object value) {
			counters.put(name, value);
		}

		private void collectSystemMetrics() {
			// Memory usage
			MemoryUsage heapUsage = MEMORY_BEAN.getHeapMemoryUsage();
			setCounter("heap_used_mb", heapUsage.getUsed() / 1024 / 1024);
			setCounter("heap_max_mb", heapUsage.getMax() / 1024 / 1024);
			setCounter("heap_utilization", (double) heapUsage.getUsed() / heapUsage.getMax());

			MemoryUsage nonHeapUsage = MEMORY_BEAN.getNonHeapMemoryUsage();
			setCounter("non_heap_used_mb", nonHeapUsage.getUsed() / 1024 / 1024);

			// GC statistics
			long totalGcTime = 0;
			long totalGcCount = 0;
			for (GarbageCollectorMXBean gcBean : GC_BEANS) {
				totalGcTime += gcBean.getCollectionTime();
				totalGcCount += gcBean.getCollectionCount();
			}
			setCounter("gc_time_ms", totalGcTime);
			setCounter("gc_count", totalGcCount);

			// Thread information
			setCounter("thread_count", THREAD_BEAN.getThreadCount());
			setCounter("daemon_thread_count", THREAD_BEAN.getDaemonThreadCount());
		}

		public Map<String, Object> getReport() {
			Map<String, Object> report = new HashMap<>();

			// Timer statistics
			Map<String, Map<String, Object>> timerStats = new HashMap<>();
			for (Map.Entry<String, List<Long>> entry : measurements.entrySet()) {
				List<Long> values = entry.getValue();
				if (!values.isEmpty()) {
					Map<String, Object> stats = new HashMap<>();
					stats.put("count", values.size());
					stats.put("total_ns", values.stream().mapToLong(Long::longValue).sum());
					stats.put("avg_ns", values.stream().mapToLong(Long::longValue).average().orElse(0));
					stats.put("min_ns", values.stream().mapToLong(Long::longValue).min().orElse(0));
					stats.put("max_ns", values.stream().mapToLong(Long::longValue).max().orElse(0));
					timerStats.put(entry.getKey(), stats);
				}
			}
			report.put("timers", timerStats);

			// Current counters
			report.put("counters", new HashMap<>(counters));

			// System information
			Runtime runtime = Runtime.getRuntime();
			Map<String, Object> systemInfo = new HashMap<>();
			systemInfo.put("available_processors", runtime.availableProcessors());
			systemInfo.put("max_memory_mb", runtime.maxMemory() / 1024 / 1024);
			systemInfo.put("total_memory_mb", runtime.totalMemory() / 1024 / 1024);
			systemInfo.put("free_memory_mb", runtime.freeMemory() / 1024 / 1024);
			report.put("system", systemInfo);

			return report;
		}

		public void printReport() {
			Map<String, Object> report = getReport();
			System.out.println("=== PERFORMANCE REPORT ===");

			// Print timer statistics
			@SuppressWarnings("unchecked")
			Map<String, Map<String, Object>> timers = (Map<String, Map<String, Object>>) report.get("timers");
			if (!timers.isEmpty()) {
				System.out.println("\nTimers:");
				timers.forEach((name, stats) -> {
					double avgMs = ((Number) stats.get("avg_ns")).doubleValue() / 1_000_000;
					System.out.printf("  %-20s: %d calls, avg=%.2fms%n",
						name, stats.get("count"), avgMs);
				});
			}

			// Print counters
			@SuppressWarnings("unchecked")
			Map<String, Object> counters = (Map<String, Object>) report.get("counters");
			if (!counters.isEmpty()) {
				System.out.println("\nCounters:");
				counters.forEach((name, value) ->
					System.out.printf("  %-20s: %s%n", name, value));
			}

			// Print system info
			@SuppressWarnings("unchecked")
			Map<String, Object> system = (Map<String, Object>) report.get("system");
			System.out.println("\nSystem:");
			system.forEach((name, value) ->
				System.out.printf("  %-20s: %s%n", name, value));
		}

		public void saveReport(Path outputFile) throws IOException {
			Map<String, Object> report = getReport();
			report.put("timestamp", Instant.now().toString());
			report.put("report_type", "performance_monitor");

			try (FileWriter writer = new FileWriter(outputFile.toFile())) {
				MAPPER.writerWithDefaultPrettyPrinter().writeValue(writer, report);
			}

			LOGGER.info("Performance report saved to: " + outputFile);
		}
	}

	public static class MemoryAnalyzer {
		private static class MemorySnapshot {
			public final long timestamp;
			public final long heapUsed;
			public final long heapMax;
			public final long nonHeapUsed;
			public final long directMemory;
			public final Map<String, MemoryUsage> poolUsages;

			public MemorySnapshot() {
				this.timestamp = System.currentTimeMillis();
				MemoryUsage heap = MEMORY_BEAN.getHeapMemoryUsage();
				this.heapUsed = heap.getUsed();
				this.heapMax = heap.getMax();
				this.nonHeapUsed = MEMORY_BEAN.getNonHeapMemoryUsage().getUsed();

				// Estimate direct memory (simplified)
				this.directMemory = estimateDirectMemory();

				// Memory pool usages
				this.poolUsages = new HashMap<>();
				for (MemoryPoolMXBean poolBean : ManagementFactory.getMemoryPoolMXBeans()) {
					poolUsages.put(poolBean.getName(), poolBean.getUsage());
				}
			}

			private long estimateDirectMemory() {
				// This is a simplified estimation
				// In practice, you might use sun.misc.VM.maxDirectMemory() or similar
				return 0;
			}
		}

		private final List<MemorySnapshot> snapshots = new ArrayList<>();
		private MemorySnapshot baseline;

		public void takeBaseline() {
			baseline = new MemorySnapshot();
			snapshots.clear();
			LOGGER.info("Memory baseline captured");
		}

		public void takeSnapshot() {
			snapshots.add(new MemorySnapshot());
		}

		public void takeSnapshot(String label) {
			MemorySnapshot snapshot = new MemorySnapshot();
			// In a more complete implementation, you'd store the label
			snapshots.add(snapshot);
			LOGGER.info("Memory snapshot taken: " + label);
		}

		public void analyzeMemoryLeak() {
			if (snapshots.size() < 3) {
				System.out.println("Need at least 3 snapshots for leak analysis");
				return;
			}

			System.out.println("=== MEMORY LEAK ANALYSIS ===");

			// Analyze heap growth trend
			long[] heapUsages = snapshots.stream().mapToLong(s -> s.heapUsed).toArray();
			double heapGrowthRate = calculateGrowthRate(heapUsages);

			System.out.printf("Heap growth rate: %.2f MB/snapshot%n", heapGrowthRate / 1024 / 1024);

			// Analyze non-heap growth
			long[] nonHeapUsages = snapshots.stream().mapToLong(s -> s.nonHeapUsed).toArray();
			double nonHeapGrowthRate = calculateGrowthRate(nonHeapUsages);

			System.out.printf("Non-heap growth rate: %.2f MB/snapshot%n", nonHeapGrowthRate / 1024 / 1024);

			// Check for suspicious growth
			if (heapGrowthRate > 10 * 1024 * 1024) { // 10MB per snapshot
				System.out.println("⚠️  Potential heap memory leak detected!");
			}

			if (nonHeapGrowthRate > 5 * 1024 * 1024) { // 5MB per snapshot
				System.out.println("⚠️  Potential non-heap memory leak detected!");
			}

			// Analyze memory pools
			analyzeMemoryPools();
		}

		private double calculateGrowthRate(long[] values) {
			if (values.length < 2) return 0;

			double sum = 0;
			for (int i = 1; i < values.length; i++) {
				sum += values[i] - values[i - 1];
			}
			return sum / (values.length - 1);
		}

		private void analyzeMemoryPools() {
			System.out.println("\nMemory Pool Analysis:");

			if (baseline == null || snapshots.isEmpty()) {
				System.out.println("No baseline or snapshots available");
				return;
			}

			MemorySnapshot latest = snapshots.get(snapshots.size() - 1);

			for (String poolName : baseline.poolUsages.keySet()) {
				MemoryUsage baselineUsage = baseline.poolUsages.get(poolName);
				MemoryUsage latestUsage = latest.poolUsages.get(poolName);

				if (baselineUsage != null && latestUsage != null) {
					long growth = latestUsage.getUsed() - baselineUsage.getUsed();
					System.out.printf("  %-30s: %+.2f MB%n",
						poolName, growth / 1024.0 / 1024.0);
				}
			}
		}

		public void printMemoryReport() {
			if (snapshots.isEmpty()) {
				System.out.println("No memory snapshots available");
				return;
			}

			System.out.println("=== MEMORY ANALYSIS REPORT ===");

			MemorySnapshot latest = snapshots.get(snapshots.size() - 1);
			System.out.printf("Current heap usage: %.2f MB / %.2f MB (%.1f%%)%n",
				latest.heapUsed / 1024.0 / 1024.0,
				latest.heapMax / 1024.0 / 1024.0,
				(double) latest.heapUsed / latest.heapMax * 100);

			System.out.printf("Non-heap usage: %.2f MB%n",
				latest.nonHeapUsed / 1024.0 / 1024.0);

			if (baseline != null) {
				long heapGrowth = latest.heapUsed - baseline.heapUsed;
				long nonHeapGrowth = latest.nonHeapUsed - baseline.nonHeapUsed;

				System.out.printf("Heap growth since baseline: %+.2f MB%n",
					heapGrowth / 1024.0 / 1024.0);
				System.out.printf("Non-heap growth since baseline: %+.2f MB%n",
					nonHeapGrowth / 1024.0 / 1024.0);
			}

			System.out.printf("Total snapshots: %d%n", snapshots.size());
		}

		public void saveReport(Path outputFile) throws IOException {
			Map<String, Object> report = new HashMap<>();
			report.put("timestamp", Instant.now().toString());
			report.put("report_type", "memory_analysis");

			// Add snapshot data
			List<Map<String, Object>> snapshotData = new ArrayList<>();
			for (int i = 0; i < snapshots.size(); i++) {
				MemorySnapshot snapshot = snapshots.get(i);
				Map<String, Object> data = new HashMap<>();
				data.put("index", i);
				data.put("timestamp", snapshot.timestamp);
				data.put("heap_used_mb", snapshot.heapUsed / 1024 / 1024);
				data.put("heap_max_mb", snapshot.heapMax / 1024 / 1024);
				data.put("non_heap_used_mb", snapshot.nonHeapUsed / 1024 / 1024);
				snapshotData.add(data);
			}
			report.put("snapshots", snapshotData);

			// Add baseline if available
			if (baseline != null) {
				Map<String, Object> baselineData = new HashMap<>();
				baselineData.put("timestamp", baseline.timestamp);
				baselineData.put("heap_used_mb", baseline.heapUsed / 1024 / 1024);
				baselineData.put("heap_max_mb", baseline.heapMax / 1024 / 1024);
				baselineData.put("non_heap_used_mb", baseline.nonHeapUsed / 1024 / 1024);
				report.put("baseline", baselineData);
			}

			try (FileWriter writer = new FileWriter(outputFile.toFile())) {
				MAPPER.writerWithDefaultPrettyPrinter().writeValue(writer, report);
			}

			LOGGER.info("Memory analysis report saved to: " + outputFile);
		}
	}

	public static class ThreadAnalyzer {
		public static class CustomThreadInfo {
			public final long id;
			public final String name;
			public final Thread.State state;
			public final long cpuTime;
			public final long userTime;
			public final boolean isDeadlocked;

			public CustomThreadInfo(java.lang.management.ThreadInfo threadInfo) {
				this.id = threadInfo.getThreadId();
				this.name = threadInfo.getThreadName();
				this.state = threadInfo.getThreadState();
				this.cpuTime = THREAD_BEAN.getThreadCpuTime(id);
				this.userTime = THREAD_BEAN.getThreadUserTime(id);
				this.isDeadlocked = isInDeadlock(id);
			}

			private boolean isInDeadlock(long threadId) {
				long[] deadlockedThreads = THREAD_BEAN.findDeadlockedThreads();
				if (deadlockedThreads == null) return false;

				for (long id : deadlockedThreads) {
					if (id == threadId) return true;
				}
				return false;
			}
		}

		public void analyzeThreads() {
			System.out.println("=== THREAD ANALYSIS ===");

			ThreadInfo[] threadInfos = THREAD_BEAN.dumpAllThreads(false, false);

			// Group threads by state
			Map<Thread.State, List<ThreadInfo>> threadsByState = new HashMap<>();
			for (ThreadInfo info : threadInfos) {
				threadsByState.computeIfAbsent(info.getThreadState(), k -> new ArrayList<>()).add(info);
			}

			// Print summary
			System.out.printf("Total threads: %d%n", threadInfos.length);
			for (Thread.State state : Thread.State.values()) {
				List<ThreadInfo> threads = threadsByState.getOrDefault(state, Collections.emptyList());
				if (!threads.isEmpty()) {
					System.out.printf("  %-12s: %d threads%n", state, threads.size());
				}
			}

			// Check for deadlocks
			long[] deadlockedThreads = THREAD_BEAN.findDeadlockedThreads();
			if (deadlockedThreads != null && deadlockedThreads.length > 0) {
				System.out.println("\n⚠️  DEADLOCK DETECTED!");
				System.out.println("Deadlocked threads:");
				for (long threadId : deadlockedThreads) {
					ThreadInfo info = THREAD_BEAN.getThreadInfo(threadId);
					System.out.printf("  Thread %d: %s%n", threadId, info.getThreadName());
				}
			}

			// Find high CPU threads
			findHighCpuThreads(threadInfos);
		}

		private void findHighCpuThreads(ThreadInfo[] threadInfos) {
			System.out.println("\nHigh CPU Usage Threads:");

			Arrays.stream(threadInfos)
				.filter(info -> THREAD_BEAN.getThreadCpuTime(info.getThreadId()) > 0)
				.sorted((a, b) -> Long.compare(
					THREAD_BEAN.getThreadCpuTime(b.getThreadId()),
					THREAD_BEAN.getThreadCpuTime(a.getThreadId())))
				.limit(5)
				.forEach(info -> {
					long cpuTime = THREAD_BEAN.getThreadCpuTime(info.getThreadId());
					System.out.printf("  %-30s: %.2f ms CPU%n",
						info.getThreadName(), cpuTime / 1_000_000.0);
				});
		}

		public void printThreadDump() {
			System.out.println("=== THREAD DUMP ===");
			System.out.println("Timestamp: " + DateTimeFormatter.ISO_INSTANT.format(Instant.now()));
			System.out.println();

			ThreadInfo[] threadInfos = THREAD_BEAN.dumpAllThreads(true, true);

			for (ThreadInfo info : threadInfos) {
				System.out.printf("\"%s\" #%d %s%n",
					info.getThreadName(), info.getThreadId(), info.getThreadState());

				StackTraceElement[] stackTrace = info.getStackTrace();
				for (StackTraceElement element : stackTrace) {
					System.out.println("    at " + element);
				}

				if (info.getLockInfo() != null) {
					System.out.println("    - locked " + info.getLockInfo());
				}

				System.out.println();
			}
		}
	}

	public static class CodeProfiler {
		private static class ProfileData {
			public long totalTime;
			public long callCount;
			public long minTime = Long.MAX_VALUE;
			public long maxTime = Long.MIN_VALUE;

			public void addSample(long time) {
				totalTime += time;
				callCount++;
				minTime = Math.min(minTime, time);
				maxTime = Math.max(maxTime, time);
			}

			public double getAverageTime() {
				return callCount > 0 ? (double) totalTime / callCount : 0;
			}
		}

		private final Map<String, ProfileData> profiles = new ConcurrentHashMap<>();
		private final ThreadLocal<Map<String, Long>> startTimes = ThreadLocal.withInitial(HashMap::new);

		public void startProfiling(String method) {
			startTimes.get().put(method, System.nanoTime());
		}

		public void endProfiling(String method) {
			Long startTime = startTimes.get().remove(method);
			if (startTime != null) {
				long duration = System.nanoTime() - startTime;
				profiles.computeIfAbsent(method, k -> new ProfileData()).addSample(duration);
			}
		}

		public void printProfile() {
			System.out.println("=== CODE PROFILING REPORT ===");

			profiles.entrySet().stream()
				.sorted((a, b) -> Long.compare(b.getValue().totalTime, a.getValue().totalTime))
				.forEach(entry -> {
					String method = entry.getKey();
					ProfileData data = entry.getValue();

					System.out.printf("%-40s: %8d calls, avg=%8.2fms, total=%8.2fms%n",
						method, data.callCount,
						data.getAverageTime() / 1_000_000.0,
						data.totalTime / 1_000_000.0);
				});
		}

		public void reset() {
			profiles.clear();
			startTimes.get().clear();
		}
	}

	/**
	 * Utility methods for automated profiling
	 */
	public static class AutoProfiler {
		public static <T> T profile(String name, Callable<T> callable) throws Exception {
			long startTime = System.nanoTime();
			try {
				return callable.call();
			} finally {
				long duration = System.nanoTime() - startTime;
				LOGGER.info(String.format("Profile [%s]: %.2fms", name, duration / 1_000_000.0));
			}
		}

		public static void profile(String name, Runnable runnable) {
			long startTime = System.nanoTime();
			try {
				runnable.run();
			} finally {
				long duration = System.nanoTime() - startTime;
				LOGGER.info(String.format("Profile [%s]: %.2fms", name, duration / 1_000_000.0));
			}
		}
	}

	/**
	 * System information utilities
	 */
	public static class SystemInfo {
		public static void printSystemInformation() {
			System.out.println("=== SYSTEM INFORMATION ===");

			// JVM information
			System.out.println("JVM:");
			System.out.println("  Version: " + System.getProperty("java.version"));
			System.out.println("  Vendor: " + System.getProperty("java.vendor"));
			System.out.println("  Home: " + System.getProperty("java.home"));

			// OS information
			System.out.println("\nOperating System:");
			System.out.println("  Name: " + System.getProperty("os.name"));
			System.out.println("  Version: " + System.getProperty("os.version"));
			System.out.println("  Architecture: " + System.getProperty("os.arch"));

			// Runtime information
			Runtime runtime = Runtime.getRuntime();
			System.out.println("\nRuntime:");
			System.out.printf("  Processors: %d%n", runtime.availableProcessors());
			System.out.printf("  Max memory: %.2f MB%n", runtime.maxMemory() / 1024.0 / 1024.0);
			System.out.printf("  Total memory: %.2f MB%n", runtime.totalMemory() / 1024.0 / 1024.0);
			System.out.printf("  Free memory: %.2f MB%n", runtime.freeMemory() / 1024.0 / 1024.0);

			// Memory pools
			System.out.println("\nMemory Pools:");
			for (MemoryPoolMXBean poolBean : ManagementFactory.getMemoryPoolMXBeans()) {
				MemoryUsage usage = poolBean.getUsage();
				System.out.printf("  %-30s: %.2f MB%n",
					poolBean.getName(), usage.getUsed() / 1024.0 / 1024.0);
			}
		}

		public static Map<String, Object> getSystemProperties() {
			Map<String, Object> props = new HashMap<>();

			// JVM properties
			Map<String, String> jvmProps = new HashMap<>();
			jvmProps.put("version", System.getProperty("java.version"));
			jvmProps.put("vendor", System.getProperty("java.vendor"));
			jvmProps.put("home", System.getProperty("java.home"));
			props.put("jvm", jvmProps);

			// OS properties
			Map<String, String> osProps = new HashMap<>();
			osProps.put("name", System.getProperty("os.name"));
			osProps.put("version", System.getProperty("os.version"));
			osProps.put("arch", System.getProperty("os.arch"));
			props.put("os", osProps);

			// Runtime properties
			Runtime runtime = Runtime.getRuntime();
			Map<String, Object> runtimeProps = new HashMap<>();
			runtimeProps.put("processors", runtime.availableProcessors());
			runtimeProps.put("max_memory_mb", runtime.maxMemory() / 1024 / 1024);
			runtimeProps.put("total_memory_mb", runtime.totalMemory() / 1024 / 1024);
			runtimeProps.put("free_memory_mb", runtime.freeMemory() / 1024 / 1024);
			props.put("runtime", runtimeProps);

			return props;
		}
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		if (args.length < 1) {
			printUsage();
			System.exit(1);
		}

		try {
			String command = args[0];

			switch (command) {
				case "sysinfo":
					SystemInfo.printSystemInformation();
					break;
				case "monitor":
					handleMonitorCommand(args);
					break;
				case "memory":
					handleMemoryCommand(args);
					break;
				case "threads":
					handleThreadsCommand(args);
					break;
				case "threaddump":
					new ThreadAnalyzer().printThreadDump();
					break;
				default:
					System.err.println("Unknown command: " + command);
					printUsage();
					System.exit(1);
			}

		} catch (Exception e) {
			LOGGER.severe("Command failed: " + e.getMessage());
			e.printStackTrace();
			System.exit(1);
		}
	}

	private static void handleMonitorCommand(String[] args) throws Exception {
		int duration = 10; // seconds
		for (int i = 1; i < args.length; i++) {
			if ("--duration".equals(args[i]) && i + 1 < args.length) {
				duration = Integer.parseInt(args[++i]);
			}
		}

		PerformanceMonitor monitor = new PerformanceMonitor();
		monitor.start();

		System.out.println("Monitoring performance for " + duration + " seconds...");
		Thread.sleep(duration * 1000);

		monitor.stop();
		monitor.printReport();
	}

	private static void handleMemoryCommand(String[] args) throws Exception {
		MemoryAnalyzer analyzer = new MemoryAnalyzer();

		boolean interactive = Arrays.asList(args).contains("--interactive");

		if (interactive) {
			Scanner scanner = new Scanner(System.in);
			System.out.println("Interactive memory analysis mode");
			System.out.println("Commands: baseline, snapshot, analyze, report, quit");

			String command;
			while (!(command = scanner.nextLine().trim()).equals("quit")) {
				switch (command) {
					case "baseline":
						analyzer.takeBaseline();
						break;
					case "snapshot":
						analyzer.takeSnapshot();
						break;
					case "analyze":
						analyzer.analyzeMemoryLeak();
						break;
					case "report":
						analyzer.printMemoryReport();
						break;
					default:
						System.out.println("Unknown command: " + command);
				}
			}
		} else {
			analyzer.printMemoryReport();
		}
	}

	private static void handleThreadsCommand(String[] args) {
		new ThreadAnalyzer().analyzeThreads();
	}

	private static void printUsage() {
		System.out.println("Usage: DevelopmentUtils <command> [options]");
		System.out.println();
		System.out.println("Development utilities and debugging tools.");
		System.out.println();
		System.out.println("Commands:");
		System.out.println("  sysinfo                   Show system information");
		System.out.println("  monitor                   Performance monitoring");
		System.out.println("  memory                    Memory analysis");
		System.out.println("  threads                   Thread analysis");
		System.out.println("  threaddump                Generate thread dump");
		System.out.println();
		System.out.println("Monitor Options:");
		System.out.println("  --duration <seconds>      Monitoring duration (default: 10)");
		System.out.println();
		System.out.println("Memory Options:");
		System.out.println("  --interactive             Interactive memory analysis");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  DevelopmentUtils sysinfo");
		System.out.println("  DevelopmentUtils monitor --duration 30");
		System.out.println("  DevelopmentUtils memory --interactive");
		System.out.println("  DevelopmentUtils threads");
	}
}