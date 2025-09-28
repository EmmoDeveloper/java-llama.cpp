package de.kherud.llama.tools;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.lang.management.MemoryUsage;
import java.lang.management.ThreadInfo;
import java.lang.management.ThreadMXBean;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Library-friendly development utilities for performance monitoring and debugging.
 *
 * This refactored version provides a fluent API for development utilities,
 * with builder pattern configuration, event-driven monitoring, and structured results.
 *
 * Usage examples:
 * <pre>{@code
 * // System information
 * SystemReport sysInfo = LlamaDevelopmentUtilsLibrary.builder().build().getSystemInformation();
 * System.out.println("OS: " + sysInfo.getOsName());
 *
 * // Performance monitoring
 * LlamaDevelopmentUtilsLibrary utils = LlamaDevelopmentUtilsLibrary.builder()
 *     .monitoringDuration(Duration.ofMinutes(5))
 *     .build();
 *
 * PerformanceReport report = utils.monitorPerformance();
 * System.out.println("Peak memory: " + report.getPeakMemoryUsage());
 *
 * // Continuous monitoring with events
 * utils.startContinuousMonitoring(event -> {
 *     if (event instanceof MemoryEvent) {
 *         MemoryEvent memEvent = (MemoryEvent) event;
 *         System.out.println("Memory usage: " + memEvent.getUsedMemory());
 *     }
 * });
 *
 * // Memory analysis
 * MemoryReport memReport = utils.analyzeMemory();
 * memReport.getGarbageCollectors().forEach(gc ->
 *     System.out.println(gc.getName() + ": " + gc.getCollectionCount() + " collections"));
 *
 * // Thread analysis
 * ThreadReport threadReport = utils.analyzeThreads();
 * System.out.println("Thread count: " + threadReport.getThreadCount());
 * }</pre>
 */
public class LlamaDevelopmentUtilsLibrary implements AutoCloseable {
	private static final System.Logger logger = System.getLogger(LlamaDevelopmentUtilsLibrary.class.getName());

	// Configuration
	private final Duration monitoringDuration;
	private final Duration samplingInterval;
	private final boolean includeThreadDumps;
	private final boolean includeMemoryAnalysis;
	private final boolean includeSystemInfo;
	private final Path outputDirectory;

	// Monitoring state
	private volatile boolean monitoring = false;
	private ScheduledExecutorService monitoringExecutor;
	private final List<Consumer<MetricsEvent>> eventListeners = new CopyOnWriteArrayList<>();

	private LlamaDevelopmentUtilsLibrary(Builder builder) {
		this.monitoringDuration = builder.monitoringDuration;
		this.samplingInterval = builder.samplingInterval;
		this.includeThreadDumps = builder.includeThreadDumps;
		this.includeMemoryAnalysis = builder.includeMemoryAnalysis;
		this.includeSystemInfo = builder.includeSystemInfo;
		this.outputDirectory = builder.outputDirectory;
	}

	public static Builder builder() {
		return new Builder();
	}

	// System Information
	public SystemReport getSystemInformation() {
		SystemReport.Builder builder = new SystemReport.Builder();

		// OS Information
		builder.osName(System.getProperty("os.name"));
		builder.osVersion(System.getProperty("os.version"));
		builder.osArch(System.getProperty("os.arch"));

		// Java Information
		builder.javaVersion(System.getProperty("java.version"));
		builder.javaVendor(System.getProperty("java.vendor"));
		builder.javaHome(System.getProperty("java.home"));

		// Runtime Information
		Runtime runtime = Runtime.getRuntime();
		builder.availableProcessors(runtime.availableProcessors());
		builder.maxMemory(runtime.maxMemory());
		builder.totalMemory(runtime.totalMemory());
		builder.freeMemory(runtime.freeMemory());

		// Memory Management
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();

		builder.heapMemoryUsage(new MemoryInfo(
			heapUsage.getInit(),
			heapUsage.getUsed(),
			heapUsage.getCommitted(),
			heapUsage.getMax()
		));

		builder.nonHeapMemoryUsage(new MemoryInfo(
			nonHeapUsage.getInit(),
			nonHeapUsage.getUsed(),
			nonHeapUsage.getCommitted(),
			nonHeapUsage.getMax()
		));

		// Garbage Collectors
		List<GarbageCollectorInfo> gcInfos = new ArrayList<>();
		for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
			gcInfos.add(new GarbageCollectorInfo(
				gcBean.getName(),
				gcBean.getCollectionCount(),
				gcBean.getCollectionTime(),
				Arrays.asList(gcBean.getMemoryPoolNames())
			));
		}
		builder.garbageCollectors(gcInfos);

		// Thread Information
		ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
		builder.threadCount(threadBean.getThreadCount());
		builder.peakThreadCount(threadBean.getPeakThreadCount());
		builder.daemonThreadCount(threadBean.getDaemonThreadCount());

		return builder.build();
	}

	// Performance Monitoring
	public PerformanceReport monitorPerformance() {
		return monitorPerformance(null);
	}

	public PerformanceReport monitorPerformance(Consumer<MonitoringProgress> progressCallback) {
		PerformanceReport.Builder reportBuilder = new PerformanceReport.Builder();

		Instant startTime = Instant.now();
		reportBuilder.startTime(startTime);

		List<MemorySnapshot> memorySnapshots = new ArrayList<>();
		List<ThreadSnapshot> threadSnapshots = new ArrayList<>();
		List<GCSnapshot> gcSnapshots = new ArrayList<>();

		// Initial snapshot
		memorySnapshots.add(takeMemorySnapshot());
		threadSnapshots.add(takeThreadSnapshot());
		gcSnapshots.add(takeGCSnapshot());

		long totalSamples = monitoringDuration.toMillis() / samplingInterval.toMillis();
		long currentSample = 0;

		try {
			while (currentSample < totalSamples) {
				Thread.sleep(samplingInterval.toMillis());
				currentSample++;

				// Take snapshots
				memorySnapshots.add(takeMemorySnapshot());
				threadSnapshots.add(takeThreadSnapshot());
				gcSnapshots.add(takeGCSnapshot());

				// Progress callback
				if (progressCallback != null) {
					double progress = (double) currentSample / totalSamples;
					progressCallback.accept(new MonitoringProgress(
						"Monitoring performance", progress, currentSample, totalSamples));
				}
			}
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			logger.log(System.Logger.Level.WARNING, "Performance monitoring interrupted", e);
		}

		Instant endTime = Instant.now();
		reportBuilder.endTime(endTime);
		reportBuilder.duration(Duration.between(startTime, endTime));

		// Analyze snapshots
		analyzeMemorySnapshots(memorySnapshots, reportBuilder);
		analyzeThreadSnapshots(threadSnapshots, reportBuilder);
		analyzeGCSnapshots(gcSnapshots, reportBuilder);

		return reportBuilder.build();
	}

	public CompletableFuture<PerformanceReport> monitorPerformanceAsync() {
		return CompletableFuture.supplyAsync(this::monitorPerformance);
	}

	// Memory Analysis
	public MemoryReport analyzeMemory() {
		MemoryReport.Builder builder = new MemoryReport.Builder();

		// Current memory state
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();

		builder.heapMemory(new MemoryInfo(
			heapUsage.getInit(),
			heapUsage.getUsed(),
			heapUsage.getCommitted(),
			heapUsage.getMax()
		));

		builder.nonHeapMemory(new MemoryInfo(
			nonHeapUsage.getInit(),
			nonHeapUsage.getUsed(),
			nonHeapUsage.getCommitted(),
			nonHeapUsage.getMax()
		));

		// Memory pools
		List<MemoryPoolInfo> poolInfos = new ArrayList<>();
		for (MemoryPoolMXBean poolBean : ManagementFactory.getMemoryPoolMXBeans()) {
			MemoryUsage usage = poolBean.getUsage();
			MemoryUsage peakUsage = poolBean.getPeakUsage();

			poolInfos.add(new MemoryPoolInfo(
				poolBean.getName(),
				poolBean.getType(),
				usage != null ? new MemoryInfo(usage.getInit(), usage.getUsed(), usage.getCommitted(), usage.getMax()) : null,
				peakUsage != null ? new MemoryInfo(peakUsage.getInit(), peakUsage.getUsed(), peakUsage.getCommitted(), peakUsage.getMax()) : null
			));
		}
		builder.memoryPools(poolInfos);

		// Garbage collection info
		List<GarbageCollectorInfo> gcInfos = new ArrayList<>();
		for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
			gcInfos.add(new GarbageCollectorInfo(
				gcBean.getName(),
				gcBean.getCollectionCount(),
				gcBean.getCollectionTime(),
				Arrays.asList(gcBean.getMemoryPoolNames())
			));
		}
		builder.garbageCollectors(gcInfos);

		return builder.build();
	}

	// Thread Analysis
	public ThreadReport analyzeThreads() {
		ThreadReport.Builder builder = new ThreadReport.Builder();

		ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();

		// Basic thread counts
		builder.threadCount(threadBean.getThreadCount());
		builder.peakThreadCount(threadBean.getPeakThreadCount());
		builder.daemonThreadCount(threadBean.getDaemonThreadCount());
		builder.totalStartedThreadCount(threadBean.getTotalStartedThreadCount());

		// Thread details
		long[] threadIds = threadBean.getAllThreadIds();
		List<ThreadInfo> threadInfos = Arrays.asList(threadBean.getThreadInfo(threadIds, Integer.MAX_VALUE));

		Map<Thread.State, Integer> stateCount = new HashMap<>();
		List<ThreadDetailInfo> detailInfos = new ArrayList<>();

		for (ThreadInfo info : threadInfos) {
			if (info != null) {
				Thread.State state = info.getThreadState();
				stateCount.merge(state, 1, Integer::sum);

				detailInfos.add(new ThreadDetailInfo(
					info.getThreadId(),
					info.getThreadName(),
					state,
					info.getBlockedTime(),
					info.getBlockedCount(),
					info.getWaitedTime(),
					info.getWaitedCount(),
					info.isInNative(),
					info.isSuspended()
				));
			}
		}

		builder.threadStateDistribution(stateCount);
		builder.threadDetails(detailInfos);

		return builder.build();
	}

	// Continuous Monitoring
	public void startContinuousMonitoring(Consumer<MetricsEvent> eventCallback) {
		if (monitoring) {
			throw new IllegalStateException("Monitoring is already active");
		}

		monitoring = true;
		eventListeners.add(eventCallback);

		monitoringExecutor = Executors.newScheduledThreadPool(2, r -> {
			Thread t = new Thread(r, "LlamaDevUtils-Monitor");
			t.setDaemon(true);
			return t;
		});

		// Schedule memory monitoring
		monitoringExecutor.scheduleAtFixedRate(() -> {
			if (monitoring) {
				MemorySnapshot snapshot = takeMemorySnapshot();
				fireEvent(new MemoryEvent(snapshot));
			}
		}, 0, samplingInterval.toMillis(), TimeUnit.MILLISECONDS);

		// Schedule thread monitoring
		monitoringExecutor.scheduleAtFixedRate(() -> {
			if (monitoring) {
				ThreadSnapshot snapshot = takeThreadSnapshot();
				fireEvent(new ThreadEvent(snapshot));
			}
		}, 0, samplingInterval.toMillis() * 2, TimeUnit.MILLISECONDS); // Less frequent

		// Schedule GC monitoring
		monitoringExecutor.scheduleAtFixedRate(() -> {
			if (monitoring) {
				GCSnapshot snapshot = takeGCSnapshot();
				fireEvent(new GCEvent(snapshot));
			}
		}, 0, samplingInterval.toMillis() * 5, TimeUnit.MILLISECONDS); // Even less frequent
	}

	public void stopContinuousMonitoring() {
		monitoring = false;
		if (monitoringExecutor != null) {
			monitoringExecutor.shutdown();
			try {
				if (!monitoringExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
					monitoringExecutor.shutdownNow();
				}
			} catch (InterruptedException e) {
				monitoringExecutor.shutdownNow();
				Thread.currentThread().interrupt();
			}
		}
		eventListeners.clear();
	}

	// Export and Persistence
	public void exportReport(PerformanceReport report, Path outputPath) throws IOException {
		ObjectMapper mapper = new ObjectMapper();
		Map<String, Object> reportData = new HashMap<>();

		reportData.put("startTime", report.getStartTime().map(DateTimeFormatter.ISO_INSTANT::format).orElse(null));
		reportData.put("endTime", report.getEndTime().map(DateTimeFormatter.ISO_INSTANT::format).orElse(null));
		reportData.put("duration", report.getDuration().map(Duration::toString).orElse(null));
		reportData.put("peakMemoryUsage", report.getPeakMemoryUsage());
		reportData.put("averageMemoryUsage", report.getAverageMemoryUsage());
		reportData.put("peakThreadCount", report.getPeakThreadCount());
		reportData.put("averageThreadCount", report.getAverageThreadCount());

		Files.createDirectories(outputPath.getParent());
		mapper.writerWithDefaultPrettyPrinter().writeValue(outputPath.toFile(), reportData);
	}

	@Override
	public void close() {
		stopContinuousMonitoring();
	}

	// Helper methods
	private MemorySnapshot takeMemorySnapshot() {
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();

		return new MemorySnapshot(
			Instant.now(),
			heapUsage.getUsed(),
			heapUsage.getCommitted(),
			heapUsage.getMax(),
			nonHeapUsage.getUsed(),
			nonHeapUsage.getCommitted(),
			nonHeapUsage.getMax()
		);
	}

	private ThreadSnapshot takeThreadSnapshot() {
		ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
		return new ThreadSnapshot(
			Instant.now(),
			threadBean.getThreadCount(),
			threadBean.getDaemonThreadCount()
		);
	}

	private GCSnapshot takeGCSnapshot() {
		List<GCInfo> gcInfos = new ArrayList<>();
		for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
			gcInfos.add(new GCInfo(
				gcBean.getName(),
				gcBean.getCollectionCount(),
				gcBean.getCollectionTime()
			));
		}
		return new GCSnapshot(Instant.now(), gcInfos);
	}

	private void analyzeMemorySnapshots(List<MemorySnapshot> snapshots, PerformanceReport.Builder reportBuilder) {
		if (snapshots.isEmpty()) return;

		long peakHeapUsage = snapshots.stream().mapToLong(MemorySnapshot::getHeapUsed).max().orElse(0);
		double avgHeapUsage = snapshots.stream().mapToLong(MemorySnapshot::getHeapUsed).average().orElse(0);

		reportBuilder.peakMemoryUsage(peakHeapUsage);
		reportBuilder.averageMemoryUsage((long) avgHeapUsage);
	}

	private void analyzeThreadSnapshots(List<ThreadSnapshot> snapshots, PerformanceReport.Builder reportBuilder) {
		if (snapshots.isEmpty()) return;

		int peakThreadCount = snapshots.stream().mapToInt(ThreadSnapshot::getThreadCount).max().orElse(0);
		double avgThreadCount = snapshots.stream().mapToInt(ThreadSnapshot::getThreadCount).average().orElse(0);

		reportBuilder.peakThreadCount(peakThreadCount);
		reportBuilder.averageThreadCount((int) avgThreadCount);
	}

	private void analyzeGCSnapshots(List<GCSnapshot> snapshots, PerformanceReport.Builder reportBuilder) {
		if (snapshots.isEmpty()) return;

		// Calculate GC frequency and total time
		Map<String, Long> totalCollections = new HashMap<>();
		Map<String, Long> totalTime = new HashMap<>();

		for (GCSnapshot snapshot : snapshots) {
			for (GCInfo gcInfo : snapshot.getGcInfos()) {
				totalCollections.merge(gcInfo.getName(), gcInfo.getCollectionCount(), Math::max);
				totalTime.merge(gcInfo.getName(), gcInfo.getCollectionTime(), Math::max);
			}
		}

		reportBuilder.gcCollections(totalCollections);
		reportBuilder.gcTotalTime(totalTime);
	}

	private void fireEvent(MetricsEvent event) {
		for (Consumer<MetricsEvent> listener : eventListeners) {
			try {
				listener.accept(event);
			} catch (Exception e) {
				logger.log(System.Logger.Level.WARNING, "Error in event listener", e);
			}
		}
	}

	// Builder class
	public static class Builder {
		private Duration monitoringDuration = Duration.ofMinutes(1);
		private Duration samplingInterval = Duration.ofSeconds(1);
		private boolean includeThreadDumps = false;
		private boolean includeMemoryAnalysis = true;
		private boolean includeSystemInfo = true;
		private Path outputDirectory;

		public Builder monitoringDuration(Duration duration) {
			this.monitoringDuration = Objects.requireNonNull(duration);
			return this;
		}

		public Builder samplingInterval(Duration interval) {
			this.samplingInterval = Objects.requireNonNull(interval);
			return this;
		}

		public Builder includeThreadDumps(boolean include) {
			this.includeThreadDumps = include;
			return this;
		}

		public Builder includeMemoryAnalysis(boolean include) {
			this.includeMemoryAnalysis = include;
			return this;
		}

		public Builder includeSystemInfo(boolean include) {
			this.includeSystemInfo = include;
			return this;
		}

		public Builder outputDirectory(Path directory) {
			this.outputDirectory = directory;
			return this;
		}

		public LlamaDevelopmentUtilsLibrary build() {
			return new LlamaDevelopmentUtilsLibrary(this);
		}
	}

	// Data classes (simplified for space)
	public static class SystemReport {
		private final String osName, osVersion, osArch;
		private final String javaVersion, javaVendor, javaHome;
		private final int availableProcessors;
		private final long maxMemory, totalMemory, freeMemory;
		private final MemoryInfo heapMemoryUsage, nonHeapMemoryUsage;
		private final List<GarbageCollectorInfo> garbageCollectors;
		private final int threadCount, peakThreadCount, daemonThreadCount;

		private SystemReport(Builder builder) {
			this.osName = builder.osName;
			this.osVersion = builder.osVersion;
			this.osArch = builder.osArch;
			this.javaVersion = builder.javaVersion;
			this.javaVendor = builder.javaVendor;
			this.javaHome = builder.javaHome;
			this.availableProcessors = builder.availableProcessors;
			this.maxMemory = builder.maxMemory;
			this.totalMemory = builder.totalMemory;
			this.freeMemory = builder.freeMemory;
			this.heapMemoryUsage = builder.heapMemoryUsage;
			this.nonHeapMemoryUsage = builder.nonHeapMemoryUsage;
			this.garbageCollectors = Collections.unmodifiableList(builder.garbageCollectors);
			this.threadCount = builder.threadCount;
			this.peakThreadCount = builder.peakThreadCount;
			this.daemonThreadCount = builder.daemonThreadCount;
		}

		// Getters
		public String getOsName() { return osName; }
		public String getOsVersion() { return osVersion; }
		public String getOsArch() { return osArch; }
		public String getJavaVersion() { return javaVersion; }
		public String getJavaVendor() { return javaVendor; }
		public String getJavaHome() { return javaHome; }
		public int getAvailableProcessors() { return availableProcessors; }
		public long getMaxMemory() { return maxMemory; }
		public long getTotalMemory() { return totalMemory; }
		public long getFreeMemory() { return freeMemory; }
		public MemoryInfo getHeapMemoryUsage() { return heapMemoryUsage; }
		public MemoryInfo getNonHeapMemoryUsage() { return nonHeapMemoryUsage; }
		public List<GarbageCollectorInfo> getGarbageCollectors() { return garbageCollectors; }
		public int getThreadCount() { return threadCount; }
		public int getPeakThreadCount() { return peakThreadCount; }
		public int getDaemonThreadCount() { return daemonThreadCount; }

		public static class Builder {
			private String osName, osVersion, osArch;
			private String javaVersion, javaVendor, javaHome;
			private int availableProcessors;
			private long maxMemory, totalMemory, freeMemory;
			private MemoryInfo heapMemoryUsage, nonHeapMemoryUsage;
			private List<GarbageCollectorInfo> garbageCollectors = new ArrayList<>();
			private int threadCount, peakThreadCount, daemonThreadCount;

			public Builder osName(String name) { this.osName = name; return this; }
			public Builder osVersion(String version) { this.osVersion = version; return this; }
			public Builder osArch(String arch) { this.osArch = arch; return this; }
			public Builder javaVersion(String version) { this.javaVersion = version; return this; }
			public Builder javaVendor(String vendor) { this.javaVendor = vendor; return this; }
			public Builder javaHome(String home) { this.javaHome = home; return this; }
			public Builder availableProcessors(int count) { this.availableProcessors = count; return this; }
			public Builder maxMemory(long memory) { this.maxMemory = memory; return this; }
			public Builder totalMemory(long memory) { this.totalMemory = memory; return this; }
			public Builder freeMemory(long memory) { this.freeMemory = memory; return this; }
			public Builder heapMemoryUsage(MemoryInfo usage) { this.heapMemoryUsage = usage; return this; }
			public Builder nonHeapMemoryUsage(MemoryInfo usage) { this.nonHeapMemoryUsage = usage; return this; }
			public Builder garbageCollectors(List<GarbageCollectorInfo> gcs) { this.garbageCollectors = new ArrayList<>(gcs); return this; }
			public Builder threadCount(int count) { this.threadCount = count; return this; }
			public Builder peakThreadCount(int count) { this.peakThreadCount = count; return this; }
			public Builder daemonThreadCount(int count) { this.daemonThreadCount = count; return this; }

			public SystemReport build() { return new SystemReport(this); }
		}
	}

	// Additional data classes (simplified)
	public static class MemoryInfo {
		private final long init, used, committed, max;

		public MemoryInfo(long init, long used, long committed, long max) {
			this.init = init; this.used = used; this.committed = committed; this.max = max;
		}

		public long getInit() { return init; }
		public long getUsed() { return used; }
		public long getCommitted() { return committed; }
		public long getMax() { return max; }
	}

	public static class GarbageCollectorInfo {
		private final String name;
		private final long collectionCount, collectionTime;
		private final List<String> memoryPoolNames;

		public GarbageCollectorInfo(String name, long collectionCount, long collectionTime, List<String> memoryPoolNames) {
			this.name = name;
			this.collectionCount = collectionCount;
			this.collectionTime = collectionTime;
			this.memoryPoolNames = Collections.unmodifiableList(memoryPoolNames);
		}

		public String getName() { return name; }
		public long getCollectionCount() { return collectionCount; }
		public long getCollectionTime() { return collectionTime; }
		public List<String> getMemoryPoolNames() { return memoryPoolNames; }
	}

	// Additional classes for completeness (simplified)
	public static class PerformanceReport {
		private final Optional<Instant> startTime, endTime;
		private final Optional<Duration> duration;
		private final long peakMemoryUsage, averageMemoryUsage;
		private final int peakThreadCount, averageThreadCount;
		private final Map<String, Long> gcCollections, gcTotalTime;

		private PerformanceReport(Builder builder) {
			this.startTime = Optional.ofNullable(builder.startTime);
			this.endTime = Optional.ofNullable(builder.endTime);
			this.duration = Optional.ofNullable(builder.duration);
			this.peakMemoryUsage = builder.peakMemoryUsage;
			this.averageMemoryUsage = builder.averageMemoryUsage;
			this.peakThreadCount = builder.peakThreadCount;
			this.averageThreadCount = builder.averageThreadCount;
			this.gcCollections = Collections.unmodifiableMap(builder.gcCollections);
			this.gcTotalTime = Collections.unmodifiableMap(builder.gcTotalTime);
		}

		public Optional<Instant> getStartTime() { return startTime; }
		public Optional<Instant> getEndTime() { return endTime; }
		public Optional<Duration> getDuration() { return duration; }
		public long getPeakMemoryUsage() { return peakMemoryUsage; }
		public long getAverageMemoryUsage() { return averageMemoryUsage; }
		public int getPeakThreadCount() { return peakThreadCount; }
		public int getAverageThreadCount() { return averageThreadCount; }
		public Map<String, Long> getGcCollections() { return gcCollections; }
		public Map<String, Long> getGcTotalTime() { return gcTotalTime; }

		public static class Builder {
			private Instant startTime, endTime;
			private Duration duration;
			private long peakMemoryUsage, averageMemoryUsage;
			private int peakThreadCount, averageThreadCount;
			private Map<String, Long> gcCollections = new HashMap<>();
			private Map<String, Long> gcTotalTime = new HashMap<>();

			public Builder startTime(Instant time) { this.startTime = time; return this; }
			public Builder endTime(Instant time) { this.endTime = time; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder peakMemoryUsage(long usage) { this.peakMemoryUsage = usage; return this; }
			public Builder averageMemoryUsage(long usage) { this.averageMemoryUsage = usage; return this; }
			public Builder peakThreadCount(int count) { this.peakThreadCount = count; return this; }
			public Builder averageThreadCount(int count) { this.averageThreadCount = count; return this; }
			public Builder gcCollections(Map<String, Long> collections) { this.gcCollections = new HashMap<>(collections); return this; }
			public Builder gcTotalTime(Map<String, Long> times) { this.gcTotalTime = new HashMap<>(times); return this; }

			public PerformanceReport build() { return new PerformanceReport(this); }
		}
	}

	// Event classes and other supporting classes would be defined here
	// (Simplified for space - would include MemoryReport, ThreadReport, etc.)

	public interface MetricsEvent {
		Instant getTimestamp();
	}

	public static class MemoryEvent implements MetricsEvent {
		private final MemorySnapshot snapshot;
		public MemoryEvent(MemorySnapshot snapshot) { this.snapshot = snapshot; }
		public MemorySnapshot getSnapshot() { return snapshot; }
		public Instant getTimestamp() { return snapshot.getTimestamp(); }
		public long getUsedMemory() { return snapshot.getHeapUsed(); }
	}

	public static class ThreadEvent implements MetricsEvent {
		private final ThreadSnapshot snapshot;
		public ThreadEvent(ThreadSnapshot snapshot) { this.snapshot = snapshot; }
		public ThreadSnapshot getSnapshot() { return snapshot; }
		public Instant getTimestamp() { return snapshot.getTimestamp(); }
	}

	public static class GCEvent implements MetricsEvent {
		private final GCSnapshot snapshot;
		public GCEvent(GCSnapshot snapshot) { this.snapshot = snapshot; }
		public GCSnapshot getSnapshot() { return snapshot; }
		public Instant getTimestamp() { return snapshot.getTimestamp(); }
	}

	// Snapshot classes
	public static class MemorySnapshot {
		private final Instant timestamp;
		private final long heapUsed, heapCommitted, heapMax;
		private final long nonHeapUsed, nonHeapCommitted, nonHeapMax;

		public MemorySnapshot(Instant timestamp, long heapUsed, long heapCommitted, long heapMax,
				long nonHeapUsed, long nonHeapCommitted, long nonHeapMax) {
			this.timestamp = timestamp;
			this.heapUsed = heapUsed; this.heapCommitted = heapCommitted; this.heapMax = heapMax;
			this.nonHeapUsed = nonHeapUsed; this.nonHeapCommitted = nonHeapCommitted; this.nonHeapMax = nonHeapMax;
		}

		public Instant getTimestamp() { return timestamp; }
		public long getHeapUsed() { return heapUsed; }
		public long getHeapCommitted() { return heapCommitted; }
		public long getHeapMax() { return heapMax; }
		public long getNonHeapUsed() { return nonHeapUsed; }
		public long getNonHeapCommitted() { return nonHeapCommitted; }
		public long getNonHeapMax() { return nonHeapMax; }
	}

	public static class ThreadSnapshot {
		private final Instant timestamp;
		private final int threadCount, daemonThreadCount;

		public ThreadSnapshot(Instant timestamp, int threadCount, int daemonThreadCount) {
			this.timestamp = timestamp;
			this.threadCount = threadCount;
			this.daemonThreadCount = daemonThreadCount;
		}

		public Instant getTimestamp() { return timestamp; }
		public int getThreadCount() { return threadCount; }
		public int getDaemonThreadCount() { return daemonThreadCount; }
	}

	public static class GCSnapshot {
		private final Instant timestamp;
		private final List<GCInfo> gcInfos;

		public GCSnapshot(Instant timestamp, List<GCInfo> gcInfos) {
			this.timestamp = timestamp;
			this.gcInfos = Collections.unmodifiableList(gcInfos);
		}

		public Instant getTimestamp() { return timestamp; }
		public List<GCInfo> getGcInfos() { return gcInfos; }
	}

	public static class GCInfo {
		private final String name;
		private final long collectionCount, collectionTime;

		public GCInfo(String name, long collectionCount, long collectionTime) {
			this.name = name; this.collectionCount = collectionCount; this.collectionTime = collectionTime;
		}

		public String getName() { return name; }
		public long getCollectionCount() { return collectionCount; }
		public long getCollectionTime() { return collectionTime; }
	}

	// Additional placeholder classes
	public static class MemoryReport {
		public static class Builder {
			private MemoryInfo heapMemory, nonHeapMemory;
			private List<MemoryPoolInfo> memoryPools = new ArrayList<>();
			private List<GarbageCollectorInfo> garbageCollectors = new ArrayList<>();

			public Builder heapMemory(MemoryInfo info) { this.heapMemory = info; return this; }
			public Builder nonHeapMemory(MemoryInfo info) { this.nonHeapMemory = info; return this; }
			public Builder memoryPools(List<MemoryPoolInfo> pools) { this.memoryPools = new ArrayList<>(pools); return this; }
			public Builder garbageCollectors(List<GarbageCollectorInfo> gcs) { this.garbageCollectors = new ArrayList<>(gcs); return this; }
			public MemoryReport build() { return new MemoryReport(); }
		}
	}

	public static class ThreadReport {
		public static class Builder {
			private int threadCount, peakThreadCount, daemonThreadCount;
			private long totalStartedThreadCount;
			private Map<Thread.State, Integer> threadStateDistribution = new HashMap<>();
			private List<ThreadDetailInfo> threadDetails = new ArrayList<>();

			public Builder threadCount(int count) { this.threadCount = count; return this; }
			public Builder peakThreadCount(int count) { this.peakThreadCount = count; return this; }
			public Builder daemonThreadCount(int count) { this.daemonThreadCount = count; return this; }
			public Builder totalStartedThreadCount(long count) { this.totalStartedThreadCount = count; return this; }
			public Builder threadStateDistribution(Map<Thread.State, Integer> dist) { this.threadStateDistribution = new HashMap<>(dist); return this; }
			public Builder threadDetails(List<ThreadDetailInfo> details) { this.threadDetails = new ArrayList<>(details); return this; }
			public ThreadReport build() { return new ThreadReport(); }
		}
	}

	public static class MonitoringProgress {
		private final String message;
		private final double progress;
		private final long currentSample, totalSamples;

		public MonitoringProgress(String message, double progress, long currentSample, long totalSamples) {
			this.message = message; this.progress = progress; this.currentSample = currentSample; this.totalSamples = totalSamples;
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public long getCurrentSample() { return currentSample; }
		public long getTotalSamples() { return totalSamples; }
	}

	// Additional placeholder classes
	public static class MemoryPoolInfo {
		public MemoryPoolInfo(String name, MemoryType type, MemoryInfo usage, MemoryInfo peakUsage) {}
	}

	public static class ThreadDetailInfo {
		public ThreadDetailInfo(long id, String name, Thread.State state, long blockedTime, long blockedCount,
				long waitedTime, long waitedCount, boolean inNative, boolean suspended) {}
	}
}
