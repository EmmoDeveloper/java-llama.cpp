package de.kherud.llama;

/**
 * Utility class for creating and managing llama.cpp samplers.
 * 
 * This class provides static methods for creating samplers that don't require model context,
 * and utility methods for sampler management.
 */
public class LlamaSampler {
	
	// Load native library
	static {
		LlamaLoader.initialize();
	}
	
	// ========================================
	// Basic Samplers (No model required)
	// ========================================
	
	/**
	 * Create a greedy sampler that always picks the token with highest probability.
	 * @return Handle to the created sampler
	 */
	public static long createGreedy() {
		return createGreedySamplerNative();
	}
	
	/**
	 * Create a distribution sampler for random sampling.
	 * @param seed Random seed
	 * @return Handle to the created sampler
	 */
	public static long createDistribution(int seed) {
		return createDistributionSamplerNative(seed);
	}
	
	/**
	 * Create a Top-K sampler.
	 * @param k Number of top tokens to consider
	 * @return Handle to the created sampler
	 */
	public static long createTopK(int k) {
		return createTopKSamplerNative(k);
	}
	
	/**
	 * Create a Top-P (nucleus) sampler.
	 * @param p Probability threshold (0.0 to 1.0)
	 * @param minKeep Minimum number of tokens to keep
	 * @return Handle to the created sampler
	 */
	public static long createTopP(float p, int minKeep) {
		return createTopPSamplerNative(p, minKeep);
	}
	
	/**
	 * Create a Min-P sampler.
	 * @param p Minimum probability threshold
	 * @param minKeep Minimum number of tokens to keep
	 * @return Handle to the created sampler
	 */
	public static long createMinP(float p, int minKeep) {
		return createMinPSamplerNative(p, minKeep);
	}
	
	/**
	 * Create a temperature sampler.
	 * @param temperature Temperature value (higher = more random)
	 * @return Handle to the created sampler
	 */
	public static long createTemperature(float temperature) {
		return createTemperatureSamplerNative(temperature);
	}
	
	/**
	 * Create an extended temperature sampler.
	 * @param temperature Base temperature
	 * @param delta Temperature delta
	 * @param exponent Temperature exponent
	 * @return Handle to the created sampler
	 */
	public static long createExtendedTemperature(float temperature, float delta, float exponent) {
		return createExtendedTemperatureSamplerNative(temperature, delta, exponent);
	}
	
	/**
	 * Create a typical sampler.
	 * @param p Typical probability threshold
	 * @param minKeep Minimum number of tokens to keep
	 * @return Handle to the created sampler
	 */
	public static long createTypical(float p, int minKeep) {
		return createTypicalSamplerNative(p, minKeep);
	}
	
	/**
	 * Create an XTC (Exclude Top Choices) sampler.
	 * @param p Probability threshold
	 * @param threshold XTC threshold
	 * @param minKeep Minimum number of tokens to keep
	 * @param seed Random seed
	 * @return Handle to the created sampler
	 */
	public static long createXtc(float p, float threshold, int minKeep, int seed) {
		return createXtcSamplerNative(p, threshold, minKeep, seed);
	}
	
	/**
	 * Create a Mirostat v2 sampler.
	 * @param seed Random seed
	 * @param tau Target cross-entropy
	 * @param eta Learning rate
	 * @return Handle to the created sampler
	 */
	public static long createMirostatV2(int seed, float tau, float eta) {
		return createMirostatV2SamplerNative(seed, tau, eta);
	}
	
	/**
	 * Create a penalties sampler for repetition control.
	 * @param penaltyLastN Number of last tokens to consider
	 * @param penaltyRepeat Repetition penalty multiplier
	 * @param penaltyFreq Frequency penalty
	 * @param penaltyPresent Presence penalty
	 * @return Handle to the created sampler
	 */
	public static long createPenalties(int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent) {
		return createPenaltiesSamplerNative(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent);
	}
	
	// ========================================
	// Sampler Chain Management
	// ========================================
	
	/**
	 * Create a new sampler chain.
	 * @return Handle to the created sampler chain
	 */
	public static long createChain() {
		return createSamplerChainNative();
	}
	
	/**
	 * Add a sampler to a chain.
	 * @param chainHandle Handle to the sampler chain
	 * @param samplerHandle Handle to the sampler to add
	 */
	public static void addToChain(long chainHandle, long samplerHandle) {
		addToSamplerChainNative(chainHandle, samplerHandle);
	}
	
	/**
	 * Clone a sampler.
	 * @param samplerHandle Handle to the sampler to clone
	 * @return Handle to the cloned sampler
	 */
	public static long clone(long samplerHandle) {
		return cloneSamplerNative(samplerHandle);
	}
	
	/**
	 * Free a sampler and release its resources.
	 * @param samplerHandle Handle to the sampler to free
	 */
	public static void free(long samplerHandle) {
		freeSamplerNative(samplerHandle);
	}
	
	/**
	 * Get the name of a sampler.
	 * @param samplerHandle Handle to the sampler
	 * @return Name of the sampler
	 */
	public static String getName(long samplerHandle) {
		return getSamplerNameNative(samplerHandle);
	}
	
	/**
	 * Reset a sampler to its initial state.
	 * @param samplerHandle Handle to the sampler to reset
	 */
	public static void reset(long samplerHandle) {
		resetSamplerNative(samplerHandle);
	}
	
	/**
	 * Accept a token into the sampler (for stateful samplers).
	 * @param samplerHandle Handle to the sampler
	 * @param token Token to accept
	 */
	public static void acceptToken(long samplerHandle, int token) {
		acceptTokenNative(samplerHandle, token);
	}
	
	// ========================================
	// Native method declarations
	// ========================================
	
	private static native long createGreedySamplerNative();
	private static native long createDistributionSamplerNative(int seed);
	private static native long createTopKSamplerNative(int k);
	private static native long createTopPSamplerNative(float p, int minKeep);
	private static native long createMinPSamplerNative(float p, int minKeep);
	private static native long createTemperatureSamplerNative(float temperature);
	private static native long createExtendedTemperatureSamplerNative(float temperature, float delta, float exponent);
	private static native long createTypicalSamplerNative(float p, int minKeep);
	private static native long createXtcSamplerNative(float p, float threshold, int minKeep, int seed);
	private static native long createMirostatV2SamplerNative(int seed, float tau, float eta);
	private static native long createPenaltiesSamplerNative(int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent);
	
	private static native long createSamplerChainNative();
	private static native void addToSamplerChainNative(long chainHandle, long samplerHandle);
	private static native long cloneSamplerNative(long samplerHandle);
	private static native void freeSamplerNative(long samplerHandle);
	private static native String getSamplerNameNative(long samplerHandle);
	private static native void resetSamplerNative(long samplerHandle);
	private static native void acceptTokenNative(long samplerHandle, int token);
}