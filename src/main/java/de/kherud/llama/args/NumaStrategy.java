package de.kherud.llama.args;

/**
 * NUMA (Non-Uniform Memory Access) optimization strategies for multi-socket systems.
 * These strategies control how memory and computation are distributed across NUMA nodes.
 */
public enum NumaStrategy {

	/**
	 * Disable NUMA optimizations (default).
	 * All operations use the default system memory allocation.
	 */
	DISABLED(0, "NUMA optimizations disabled"),

	/**
	 * Distribute memory and computation across all NUMA nodes.
	 * Provides good performance for parallel workloads.
	 */
	DISTRIBUTE(1, "Distribute across NUMA nodes"),

	/**
	 * Isolate computation to specific NUMA nodes.
	 * Provides better memory locality for single-threaded workloads.
	 */
	ISOLATE(2, "Isolate to specific NUMA nodes"),

	/**
	 * Use numactl-style NUMA control.
	 * Advanced NUMA management using system-level controls.
	 */
	NUMACTL(3, "Use numactl-style management");

	private final int value;
	private final String description;

	NumaStrategy(int value, String description) {
		this.value = value;
		this.description = description;
	}

	/**
	 * Get the integer value for this NUMA strategy.
	 * @return the strategy value used by llama.cpp
	 */
	public int getValue() {
		return value;
	}

	/**
	 * Get a human-readable description of this NUMA strategy.
	 * @return description of the strategy
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * Get a NUMA strategy from its integer value.
	 * @param value the strategy value
	 * @return the corresponding NumaStrategy
	 * @throws IllegalArgumentException if the value is invalid
	 */
	public static NumaStrategy fromValue(int value) {
		for (NumaStrategy strategy : values()) {
			if (strategy.value == value) {
				return strategy;
			}
		}
		throw new IllegalArgumentException("Unknown NUMA strategy value: " + value);
	}

	@Override
	public String toString() {
		return description + " (" + value + ")";
	}
}
