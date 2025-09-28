package de.kherud.llama;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Advanced resource management system for multi-model deployments.
 * Handles memory allocation, GPU resources, CPU scheduling, and automatic cleanup.
 */
public class ResourceManager implements AutoCloseable {

	/**
	 * Resource allocation strategies
	 */
	public enum AllocationStrategy {
		FAIR_SHARE,         // Equal resource distribution
		PRIORITY_BASED,     // Allocate based on model priority
		PERFORMANCE_BASED,  // Allocate based on model performance needs
		DYNAMIC,            // Dynamic allocation based on current load
		RESERVED            // Pre-allocated reserved resources
	}

	/**
	 * Resource types being managed
	 */
	public enum ResourceType {
		CPU_CORES,
		MEMORY_MB,
		GPU_MEMORY_MB,
		GPU_COMPUTE_UNITS,
		DISK_SPACE_MB,
		NETWORK_BANDWIDTH
	}

	/**
	 * Resource quota configuration
	 */
	public static class ResourceQuota {
		public final Map<ResourceType, Long> limits;
		public final Map<ResourceType, Long> requests;
		public final Map<ResourceType, Long> guaranteed;
		public final int priority;

		private ResourceQuota(Builder builder) {
			this.limits = new EnumMap<>(builder.limits);
			this.requests = new EnumMap<>(builder.requests);
			this.guaranteed = new EnumMap<>(builder.guaranteed);
			this.priority = builder.priority;
		}

		public static class Builder {
			private final Map<ResourceType, Long> limits = new EnumMap<>(ResourceType.class);
			private final Map<ResourceType, Long> requests = new EnumMap<>(ResourceType.class);
			private final Map<ResourceType, Long> guaranteed = new EnumMap<>(ResourceType.class);
			private int priority = 5;

			public Builder limit(ResourceType type, long value) {
				limits.put(type, value);
				return this;
			}

			public Builder request(ResourceType type, long value) {
				requests.put(type, value);
				return this;
			}

			public Builder guarantee(ResourceType type, long value) {
				guaranteed.put(type, value);
				return this;
			}

			public Builder priority(int priority) {
				this.priority = priority;
				return this;
			}

			public ResourceQuota build() {
				return new ResourceQuota(this);
			}
		}
	}

	/**
	 * Resource allocation tracking
	 */
	private static class ResourceAllocation {
		public final String modelId;
		public final ResourceQuota quota;
		public final Map<ResourceType, AtomicLong> allocated;
		public final Map<ResourceType, AtomicLong> used;
		public volatile long lastAccessTime;
		public volatile boolean active;

		public ResourceAllocation(String modelId, ResourceQuota quota) {
			this.modelId = modelId;
			this.quota = quota;
			this.allocated = new EnumMap<>(ResourceType.class);
			this.used = new EnumMap<>(ResourceType.class);
			this.lastAccessTime = System.currentTimeMillis();
			this.active = true;

			// Initialize counters
			for (ResourceType type : ResourceType.values()) {
				allocated.put(type, new AtomicLong(0));
				used.put(type, new AtomicLong(0));
			}
		}

		public void recordUsage(ResourceType type, long amount) {
			used.get(type).set(amount);
			lastAccessTime = System.currentTimeMillis();
		}

		public boolean isWithinLimits(ResourceType type, long additionalAmount) {
			long currentUsage = used.get(type).get();
			long limit = quota.limits.getOrDefault(type, Long.MAX_VALUE);
			return (currentUsage + additionalAmount) <= limit;
		}

		public double getUtilization(ResourceType type) {
			long currentUsage = used.get(type).get();
			long allocation = allocated.get(type).get();
			return allocation > 0 ? (double) currentUsage / allocation : 0.0;
		}
	}

	/**
	 * System resource monitor
	 */
	public static class SystemResourceMonitor {
		private final MemoryMXBean memoryBean;
		private final java.lang.management.OperatingSystemMXBean osBean;
		private final ScheduledExecutorService scheduler;

		public SystemResourceMonitor() {
			this.memoryBean = ManagementFactory.getMemoryMXBean();
			this.osBean = ManagementFactory.getOperatingSystemMXBean();
			this.scheduler = Executors.newScheduledThreadPool(1, r -> {
				Thread t = new Thread(r, "resource-monitor");
				t.setDaemon(true);
				return t;
			});
		}

		public SystemResources getCurrentResources() {
			long totalMemory = memoryBean.getHeapMemoryUsage().getMax() +
							   memoryBean.getNonHeapMemoryUsage().getMax();
			long usedMemory = memoryBean.getHeapMemoryUsage().getUsed() +
							  memoryBean.getNonHeapMemoryUsage().getUsed();

			double cpuUsage = getCpuUsage();
			int availableProcessors = osBean.getAvailableProcessors();

			// GPU information would need native calls - simplified here
			long totalGpuMemory = getGpuMemoryTotal();
			long usedGpuMemory = getGpuMemoryUsed();

			return new SystemResources(
				totalMemory / (1024 * 1024), // Convert to MB
				usedMemory / (1024 * 1024),
				totalGpuMemory,
				usedGpuMemory,
				availableProcessors,
				cpuUsage
			);
		}

		private long getGpuMemoryTotal() {
			// Simplified - would use native GPU memory queries
			return 8192; // 8GB default
		}

		private long getGpuMemoryUsed() {
			// Simplified - would use native GPU memory queries
			return 0;
		}

		private double getCpuUsage() {
			// Try to get process CPU load, fallback to system load if not available
			try {
				if (osBean instanceof com.sun.management.OperatingSystemMXBean) {
					com.sun.management.OperatingSystemMXBean sunBean =
						(com.sun.management.OperatingSystemMXBean) osBean;
					double cpuLoad = sunBean.getProcessCpuLoad();
					if (cpuLoad >= 0) {
						return cpuLoad;
					}
				}
			} catch (Exception e) {
				// Fallback to system load average
			}

			// Fallback to system load average
			double loadAverage = osBean.getSystemLoadAverage();
			if (loadAverage >= 0) {
				return Math.min(1.0, loadAverage / osBean.getAvailableProcessors());
			}

			// Final fallback
			return 0.0;
		}

		public void shutdown() {
			scheduler.shutdown();
		}
	}

	/**
	 * System resource snapshot
	 */
	public static class SystemResources {
		public final long totalMemoryMB;
		public final long usedMemoryMB;
		public final long totalGpuMemoryMB;
		public final long usedGpuMemoryMB;
		public final int availableCpuCores;
		public final double cpuUsage;

		public SystemResources(long totalMemoryMB, long usedMemoryMB,
							   long totalGpuMemoryMB, long usedGpuMemoryMB,
							   int availableCpuCores, double cpuUsage) {
			this.totalMemoryMB = totalMemoryMB;
			this.usedMemoryMB = usedMemoryMB;
			this.totalGpuMemoryMB = totalGpuMemoryMB;
			this.usedGpuMemoryMB = usedGpuMemoryMB;
			this.availableCpuCores = availableCpuCores;
			this.cpuUsage = cpuUsage;
		}

		public long getAvailableMemoryMB() {
			return totalMemoryMB - usedMemoryMB;
		}

		public long getAvailableGpuMemoryMB() {
			return totalGpuMemoryMB - usedGpuMemoryMB;
		}

		public double getMemoryUtilization() {
			return totalMemoryMB > 0 ? (double) usedMemoryMB / totalMemoryMB : 0.0;
		}

		public double getGpuMemoryUtilization() {
			return totalGpuMemoryMB > 0 ? (double) usedGpuMemoryMB / totalGpuMemoryMB : 0.0;
		}
	}

	// Core components
	private final AllocationStrategy strategy;
	private final Map<String, ResourceAllocation> allocations;
	private final SystemResourceMonitor systemMonitor;
	private final ScheduledExecutorService scheduler;
	private final ReentrantLock allocationLock;

	// Resource limits and thresholds
	private final Map<ResourceType, Long> globalLimits;
	private final double memoryThreshold;
	private final double gpuMemoryThreshold;
	private final long cleanupIntervalMs;

	// Metrics
	private final AtomicLong totalAllocations = new AtomicLong(0);
	private final AtomicLong activeAllocations = new AtomicLong(0);
	private final AtomicLong rejectedAllocations = new AtomicLong(0);

	public ResourceManager() {
		this(AllocationStrategy.DYNAMIC, 0.85, 0.90, 300000); // 5 minute cleanup interval
	}

	public ResourceManager(AllocationStrategy strategy, double memoryThreshold,
						   double gpuMemoryThreshold, long cleanupIntervalMs) {
		this.strategy = strategy;
		this.memoryThreshold = memoryThreshold;
		this.gpuMemoryThreshold = gpuMemoryThreshold;
		this.cleanupIntervalMs = cleanupIntervalMs;

		this.allocations = new ConcurrentHashMap<>();
		this.systemMonitor = new SystemResourceMonitor();
		this.allocationLock = new ReentrantLock();
		this.globalLimits = new EnumMap<>(ResourceType.class);

		// Initialize global limits based on system resources
		initializeGlobalLimits();

		// Start background tasks
		this.scheduler = Executors.newScheduledThreadPool(2, r -> {
			Thread t = new Thread(r, "resource-manager");
			t.setDaemon(true);
			return t;
		});

		startBackgroundTasks();
	}

	/**
	 * Allocate resources for a model
	 */
	public boolean allocateResources(String modelId, ResourceQuota quota) {
		allocationLock.lock();
		try {
			if (allocations.containsKey(modelId)) {
				throw new IllegalArgumentException("Resources already allocated for model: " + modelId);
			}

			// Check if allocation is feasible
			if (!canAllocate(quota)) {
				rejectedAllocations.incrementAndGet();
				return false;
			}

			// Create allocation
			ResourceAllocation allocation = new ResourceAllocation(modelId, quota);
			performAllocation(allocation);

			allocations.put(modelId, allocation);
			totalAllocations.incrementAndGet();
			activeAllocations.incrementAndGet();

			System.out.println("Allocated resources for model: " + modelId);
			return true;

		} finally {
			allocationLock.unlock();
		}
	}

	/**
	 * Update resource usage for a model
	 */
	public void updateResourceUsage(String modelId, ResourceType type, long amount) {
		ResourceAllocation allocation = allocations.get(modelId);
		if (allocation != null) {
			allocation.recordUsage(type, amount);
		}
	}

	/**
	 * Release resources for a model
	 */
	public void releaseResources(String modelId) {
		allocationLock.lock();
		try {
			ResourceAllocation allocation = allocations.remove(modelId);
			if (allocation != null) {
				allocation.active = false;
				activeAllocations.decrementAndGet();
				System.out.println("Released resources for model: " + modelId);
			}
		} finally {
			allocationLock.unlock();
		}
	}

	/**
	 * Get resource utilization metrics
	 */
	public Map<String, Object> getResourceMetrics() {
		SystemResources systemRes = systemMonitor.getCurrentResources();
		Map<String, Object> metrics = new HashMap<>();

		// System metrics
		metrics.put("systemMemoryUtilization", systemRes.getMemoryUtilization());
		metrics.put("systemGpuMemoryUtilization", systemRes.getGpuMemoryUtilization());
		metrics.put("systemCpuUsage", systemRes.cpuUsage);
		metrics.put("availableCpuCores", systemRes.availableCpuCores);

		// Allocation metrics
		metrics.put("totalAllocations", totalAllocations.get());
		metrics.put("activeAllocations", activeAllocations.get());
		metrics.put("rejectedAllocations", rejectedAllocations.get());

		// Per-model metrics
		Map<String, Object> modelMetrics = new HashMap<>();
		for (Map.Entry<String, ResourceAllocation> entry : allocations.entrySet()) {
			ResourceAllocation allocation = entry.getValue();
			Map<String, Object> modelData = new HashMap<>();

			// Utilization for each resource type
			Map<String, Double> utilization = new HashMap<>();
			for (ResourceType type : ResourceType.values()) {
				utilization.put(type.name().toLowerCase(), allocation.getUtilization(type));
			}
			modelData.put("utilization", utilization);

			// Usage amounts
			Map<String, Long> usage = new HashMap<>();
			for (ResourceType type : ResourceType.values()) {
				usage.put(type.name().toLowerCase(), allocation.used.get(type).get());
			}
			modelData.put("usage", usage);

			modelData.put("priority", allocation.quota.priority);
			modelData.put("active", allocation.active);
			modelData.put("lastAccessTime", allocation.lastAccessTime);

			modelMetrics.put(entry.getKey(), modelData);
		}
		metrics.put("models", modelMetrics);

		return metrics;
	}

	/**
	 * Get recommendations for resource optimization
	 */
	public List<String> getOptimizationRecommendations() {
		List<String> recommendations = new ArrayList<>();
		SystemResources systemRes = systemMonitor.getCurrentResources();

		// Memory recommendations
		if (systemRes.getMemoryUtilization() > memoryThreshold) {
			recommendations.add("High memory usage detected (" +
				String.format("%.1f%%", systemRes.getMemoryUtilization() * 100) +
				"). Consider reducing model cache sizes or freeing unused models.");
		}

		// GPU memory recommendations
		if (systemRes.getGpuMemoryUtilization() > gpuMemoryThreshold) {
			recommendations.add("High GPU memory usage detected (" +
				String.format("%.1f%%", systemRes.getGpuMemoryUtilization() * 100) +
				"). Consider reducing GPU layer allocation or using CPU fallback.");
		}

		// Underutilized models
		long currentTime = System.currentTimeMillis();
		for (ResourceAllocation allocation : allocations.values()) {
			if (allocation.active && (currentTime - allocation.lastAccessTime) > 1800000) { // 30 minutes
				recommendations.add("Model '" + allocation.modelId +
					"' has been inactive for over 30 minutes. Consider releasing resources.");
			}
		}

		// Resource imbalance
		long activeModels = activeAllocations.get();
		if (activeModels > 0) {
			double avgMemoryPerModel = systemRes.usedMemoryMB / (double) activeModels;
			if (avgMemoryPerModel > 2048) { // > 2GB per model
				recommendations.add("High average memory per model (" +
					String.format("%.0fMB", avgMemoryPerModel) +
					"). Consider using smaller models or quantization.");
			}
		}

		return recommendations;
	}

	/**
	 * Force cleanup of inactive resources
	 */
	public int forceCleanup() {
		int cleaned = 0;
		long currentTime = System.currentTimeMillis();
		List<String> toRemove = new ArrayList<>();

		allocationLock.lock();
		try {
			for (Map.Entry<String, ResourceAllocation> entry : allocations.entrySet()) {
				ResourceAllocation allocation = entry.getValue();
				if (!allocation.active ||
					(currentTime - allocation.lastAccessTime) > 3600000) { // 1 hour inactive
					toRemove.add(entry.getKey());
				}
			}

			for (String modelId : toRemove) {
				releaseResources(modelId);
				cleaned++;
			}
		} finally {
			allocationLock.unlock();
		}

		System.out.println("Force cleanup completed: " + cleaned + " models cleaned up");
		return cleaned;
	}

	@Override
	public void close() {
		System.out.println("Shutting down resource manager");

		scheduler.shutdown();
		systemMonitor.shutdown();

		try {
			if (!scheduler.awaitTermination(10, TimeUnit.SECONDS)) {
				scheduler.shutdownNow();
			}
		} catch (InterruptedException e) {
			scheduler.shutdownNow();
			Thread.currentThread().interrupt();
		}

		// Release all allocations
		allocationLock.lock();
		try {
			for (String modelId : new ArrayList<>(allocations.keySet())) {
				releaseResources(modelId);
			}
		} finally {
			allocationLock.unlock();
		}
	}

	// Private helper methods

	private void initializeGlobalLimits() {
		SystemResources systemRes = systemMonitor.getCurrentResources();

		// Set conservative global limits
		globalLimits.put(ResourceType.MEMORY_MB, (long) (systemRes.totalMemoryMB * 0.8));
		globalLimits.put(ResourceType.GPU_MEMORY_MB, (long) (systemRes.totalGpuMemoryMB * 0.8));
		globalLimits.put(ResourceType.CPU_CORES, (long) systemRes.availableCpuCores);
	}

	private boolean canAllocate(ResourceQuota quota) {
		SystemResources systemRes = systemMonitor.getCurrentResources();

		// Check memory constraints
		long requestedMemory = quota.requests.getOrDefault(ResourceType.MEMORY_MB, 0L);
		if (requestedMemory > 0 && systemRes.getAvailableMemoryMB() < requestedMemory) {
			return false;
		}

		// Check GPU memory constraints
		long requestedGpuMemory = quota.requests.getOrDefault(ResourceType.GPU_MEMORY_MB, 0L);
		if (requestedGpuMemory > 0 && systemRes.getAvailableGpuMemoryMB() < requestedGpuMemory) {
			return false;
		}

		// Check against global limits
		for (Map.Entry<ResourceType, Long> entry : quota.requests.entrySet()) {
			ResourceType type = entry.getKey();
			Long requested = entry.getValue();
			Long globalLimit = globalLimits.get(type);

			if (globalLimit != null && requested > globalLimit) {
				return false;
			}
		}

		return true;
	}

	private void performAllocation(ResourceAllocation allocation) {
		// Set allocated amounts based on strategy
		switch (strategy) {
			case FAIR_SHARE:
				performFairShareAllocation(allocation);
				break;
			case PRIORITY_BASED:
				performPriorityBasedAllocation(allocation);
				break;
			case PERFORMANCE_BASED:
				performPerformanceBasedAllocation(allocation);
				break;
			case DYNAMIC:
				performDynamicAllocation(allocation);
				break;
			case RESERVED:
				performReservedAllocation(allocation);
				break;
		}
	}

	private void performDynamicAllocation(ResourceAllocation allocation) {
		SystemResources systemRes = systemMonitor.getCurrentResources();

		// Allocate based on current system availability and requests
		for (Map.Entry<ResourceType, Long> entry : allocation.quota.requests.entrySet()) {
			ResourceType type = entry.getKey();
			Long requested = entry.getValue();

			switch (type) {
				case MEMORY_MB:
					long availableMemory = systemRes.getAvailableMemoryMB();
					long allocatedMemory = Math.min(requested, availableMemory / 2); // Conservative
					allocation.allocated.get(type).set(allocatedMemory);
					break;

				case GPU_MEMORY_MB:
					long availableGpuMemory = systemRes.getAvailableGpuMemoryMB();
					long allocatedGpuMemory = Math.min(requested, availableGpuMemory / 2);
					allocation.allocated.get(type).set(allocatedGpuMemory);
					break;

				case CPU_CORES:
					long allocatedCores = Math.min(requested, systemRes.availableCpuCores / 4);
					allocation.allocated.get(type).set(allocatedCores);
					break;

				default:
					allocation.allocated.get(type).set(requested);
					break;
			}
		}
	}

	private void performFairShareAllocation(ResourceAllocation allocation) {
		// Simple fair share - divide resources equally among active models
		int activeModelCount = Math.max(1, (int) activeAllocations.get() + 1);
		SystemResources systemRes = systemMonitor.getCurrentResources();

		allocation.allocated.get(ResourceType.MEMORY_MB)
			.set(systemRes.totalMemoryMB / activeModelCount / 2);
		allocation.allocated.get(ResourceType.GPU_MEMORY_MB)
			.set(systemRes.totalGpuMemoryMB / activeModelCount / 2);
		allocation.allocated.get(ResourceType.CPU_CORES)
			.set(Math.max(1, systemRes.availableCpuCores / activeModelCount));
	}

	private void performPriorityBasedAllocation(ResourceAllocation allocation) {
		// Allocate more resources to higher priority models
		double priorityMultiplier = allocation.quota.priority / 5.0; // Normalize around 5
		SystemResources systemRes = systemMonitor.getCurrentResources();

		long baseMemory = systemRes.totalMemoryMB / 4; // Base allocation
		long priorityMemory = (long) (baseMemory * priorityMultiplier);
		allocation.allocated.get(ResourceType.MEMORY_MB).set(Math.min(priorityMemory,
			systemRes.getAvailableMemoryMB()));
	}

	private void performPerformanceBasedAllocation(ResourceAllocation allocation) {
		// Allocate based on model performance requirements
		performDynamicAllocation(allocation); // Default to dynamic for now
	}

	private void performReservedAllocation(ResourceAllocation allocation) {
		// Use guaranteed amounts if specified, otherwise use requests
		for (ResourceType type : ResourceType.values()) {
			long reserved = allocation.quota.guaranteed.getOrDefault(type,
				allocation.quota.requests.getOrDefault(type, 0L));
			allocation.allocated.get(type).set(reserved);
		}
	}

	private void startBackgroundTasks() {
		// Resource monitoring and cleanup
		scheduler.scheduleWithFixedDelay(this::performResourceCleanup,
			cleanupIntervalMs, cleanupIntervalMs, TimeUnit.MILLISECONDS);

		// Resource optimization recommendations
		scheduler.scheduleWithFixedDelay(this::checkResourceOptimization,
			600000, 600000, TimeUnit.MILLISECONDS); // Every 10 minutes
	}

	private void performResourceCleanup() {
		try {
			List<String> recommendations = getOptimizationRecommendations();
			if (!recommendations.isEmpty()) {
				System.out.println("Resource optimization recommendations:");
				recommendations.forEach(rec -> System.out.println("  - " + rec));
			}

			// Auto cleanup if memory pressure is high
			SystemResources systemRes = systemMonitor.getCurrentResources();
			if (systemRes.getMemoryUtilization() > 0.9) { // 90% memory usage
				System.out.println("High memory pressure detected, performing cleanup");
				forceCleanup();
			}
		} catch (Exception e) {
			System.err.println("Resource cleanup error: " + e.getMessage());
		}
	}

	private void checkResourceOptimization() {
		try {
			Map<String, Object> metrics = getResourceMetrics();
			System.out.println("Resource Manager Status: " +
				"Active: " + metrics.get("activeAllocations") +
				", Memory: " + String.format("%.1f%%", (Double) metrics.get("systemMemoryUtilization") * 100) +
				", GPU: " + String.format("%.1f%%", (Double) metrics.get("systemGpuMemoryUtilization") * 100));
		} catch (Exception e) {
			System.err.println("Resource optimization check error: " + e.getMessage());
		}
	}
}
