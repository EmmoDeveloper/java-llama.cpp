package de.kherud.llama;

import de.kherud.llama.args.LogFormat;
import org.jetbrains.annotations.Nullable;

import java.lang.annotation.Native;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * This class is a wrapper around the llama.cpp functionality.
 * Upon being created, it natively allocates memory for the model context.
 * Thus, this class is an {@link AutoCloseable}, in order to de-allocate the memory when it is no longer being needed.
 * <p>
 * The main functionality of this class is:
 * <ul>
 *     <li>Streaming answers (and probabilities) via {@link #generate(InferenceParameters)}</li>
 *     <li>Creating whole responses to prompts via {@link #complete(InferenceParameters)}</li>
 *     <li>Creating embeddings via {@link #embed(String)} (make sure to configure {@link ModelParameters#enableEmbedding()}</li>
 *     <li>Accessing the tokenizer via {@link #encode(String)} and {@link #decode(int[])}</li>
 * </ul>
 */
public class LlamaModel implements AutoCloseable {

	static {
		LlamaLoader.initialize();
	}

	@Native
	private long ctx;

	/**
	 * Load with the given {@link ModelParameters}. Make sure to either set
	 * <ul>
	 *     <li>{@link ModelParameters#setModel(String)}</li>
	 *     <li>{@link ModelParameters#setModelUrl(String)}</li>
	 *     <li>{@link ModelParameters#setHfRepo(String)}, {@link ModelParameters#setHfFile(String)}</li>
	 * </ul>
	 *
	 * @param parameters the set of options
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public LlamaModel(ModelParameters parameters) {
		// Apply smart defaults for better out-of-the-box experience
		ModelParameters optimizedParams = SmartDefaults.apply(parameters);
		loadModel(optimizedParams.toArray());
	}
	
	/**
	 * Create a model optimized for text completion and generation workloads.
	 * This applies completion-specific optimizations including threading, batch sizes, and continuous batching.
	 *
	 * @param parameters the base model parameters
	 * @return LlamaModel optimized for completion tasks
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public static LlamaModel forCompletion(ModelParameters parameters) {
		ModelParameters optimized = WorkloadOptimizer.optimizeForCompletion(parameters);
		return new LlamaModel(optimized);
	}
	
	/**
	 * Create a model optimized for embedding generation workloads.
	 * This applies embedding-specific optimizations including high-throughput threading and appropriate pooling.
	 *
	 * @param parameters the base model parameters  
	 * @return LlamaModel optimized for embedding tasks
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public static LlamaModel forEmbedding(ModelParameters parameters) {
		ModelParameters optimized = WorkloadOptimizer.optimizeForEmbedding(parameters);
		return new LlamaModel(optimized);
	}
	
	/**
	 * Create a model optimized for document reranking workloads.
	 * This applies reranking-specific optimizations including parallel processing and rank pooling.
	 *
	 * @param parameters the base model parameters
	 * @return LlamaModel optimized for reranking tasks
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public static LlamaModel forReranking(ModelParameters parameters) {
		ModelParameters optimized = WorkloadOptimizer.optimizeForReranking(parameters);
		return new LlamaModel(optimized);
	}

	/**
	 * Internal constructor for bypassing intelligent defaults (used by GPU detection)
	 */
	LlamaModel() {
		// Empty constructor for internal use
	}

	/**
	 * Internal method to load model directly without intelligent defaults (used by GPU detection)
	 */
	void loadModelDirect(String... parameters) {
		loadModel(parameters);
	}

	/**
	 * Generate and return a whole answer with custom parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @return an LLM response
	 */
	public String complete(InferenceParameters parameters) {
		parameters.setStream(false);
		int taskId = requestCompletion(parameters.toString());
		LlamaOutput output = receiveCompletion(taskId);
		return output.text;
	}

	/**
	 * Generate and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @return iterable LLM outputs
	 */
	public LlamaIterable generate(InferenceParameters parameters) {
		return () -> new LlamaIterator(this, parameters);
	}



	/**
	 * Get the embedding of a string. Note, that the prompt isn't preprocessed in any way, nothing like
	 * "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the string to embed
	 * @return an embedding float array
	 * @throws IllegalStateException if embedding mode was not activated (see {@link ModelParameters#enableEmbedding()})
	 */
	public  native float[] embed(String prompt);


	/**
	 * Tokenize a prompt given the native tokenizer
	 *
	 * @param prompt the prompt to tokenize
	 * @return an array of integers each representing a token id
	 */
	public native int[] encode(String prompt);

	/**
	 * Convert an array of token ids to its string representation
	 *
	 * @param tokens an array of tokens
	 * @return the token ids decoded to a string
	 */
	public String decode(int[] tokens) {
		byte[] bytes = decodeBytes(tokens);
		return new String(bytes, StandardCharsets.UTF_8);
	}

	/**
	 * Get the size in bytes needed to store the complete model state.
	 * This includes the KV cache, logits, and embeddings.
	 *
	 * @return size in bytes needed for state storage
	 * @throws LlamaException if state size cannot be determined
	 */
	public long getModelStateSize() throws LlamaException {
		long size = getStateSize();
		if (size < 0) {
			throw new LlamaException("Failed to get model state size");
		}
		return size;
	}

	/**
	 * Get the complete model state as a byte array.
	 * This includes the KV cache, logits, and embeddings.
	 *
	 * @return byte array containing the model state
	 * @throws LlamaException if state cannot be retrieved
	 */
	public byte[] getModelState() throws LlamaException {
		byte[] state = getStateData();
		if (state == null) {
			throw new LlamaException("Failed to get model state data");
		}
		return state;
	}

	/**
	 * Restore the model state from a byte array.
	 * This will restore the KV cache, logits, and embeddings.
	 *
	 * @param stateData byte array containing the model state
	 * @return number of bytes loaded from the state data
	 * @throws LlamaException if state cannot be restored
	 */
	public long setModelState(byte[] stateData) throws LlamaException {
		if (stateData == null) {
			throw new IllegalArgumentException("State data cannot be null");
		}
		long loaded = setStateData(stateData);
		if (loaded < 0) {
			throw new LlamaException("Failed to restore model state");
		}
		return loaded;
	}

	/**
	 * Save the complete model state and conversation tokens to a file.
	 * This creates a checkpoint that can be loaded later to resume the conversation.
	 *
	 * @param filePath path where the state file should be saved
	 * @param tokens conversation tokens to save with the state (can be null)
	 * @throws LlamaException if state cannot be saved
	 */
	public void saveState(String filePath, int[] tokens) throws LlamaException {
		if (filePath == null || filePath.trim().isEmpty()) {
			throw new IllegalArgumentException("File path cannot be null or empty");
		}
		boolean success = saveStateToFile(filePath, tokens);
		if (!success) {
			throw new LlamaException("Failed to save state to file: " + filePath);
		}
	}

	/**
	 * Save the complete model state to a file (without tokens).
	 *
	 * @param filePath path where the state file should be saved
	 * @throws LlamaException if state cannot be saved
	 */
	public void saveState(String filePath) throws LlamaException {
		saveState(filePath, null);
	}

	/**
	 * Load model state and conversation tokens from a file.
	 * This restores a previously saved checkpoint.
	 *
	 * @param filePath path to the state file
	 * @param maxTokens maximum number of tokens to load (use -1 for default)
	 * @return array of conversation tokens that were saved with the state
	 * @throws LlamaException if state cannot be loaded
	 */
	public int[] loadState(String filePath, int maxTokens) throws LlamaException {
		if (filePath == null || filePath.trim().isEmpty()) {
			throw new IllegalArgumentException("File path cannot be null or empty");
		}
		int[] tokens = loadStateFromFile(filePath, maxTokens);
		if (tokens == null) {
			throw new LlamaException("Failed to load state from file: " + filePath);
		}
		return tokens;
	}

	/**
	 * Load model state from a file with default token limit.
	 *
	 * @param filePath path to the state file
	 * @return array of conversation tokens that were saved with the state
	 * @throws LlamaException if state cannot be loaded
	 */
	public int[] loadState(String filePath) throws LlamaException {
		return loadState(filePath, -1);
	}

	/**
	 * Get the size needed to store state for a specific sequence.
	 *
	 * @param sequenceId the sequence identifier
	 * @return size in bytes needed for sequence state storage
	 * @throws LlamaException if sequence state size cannot be determined
	 */
	public long getSequenceStateSize(int sequenceId) throws LlamaException {
		long size = getSequenceStateSizeNative(sequenceId);
		if (size < 0) {
			throw new LlamaException("Failed to get sequence state size for sequence " + sequenceId);
		}
		return size;
	}

	/**
	 * Get the state for a specific sequence as a byte array.
	 *
	 * @param sequenceId the sequence identifier
	 * @return byte array containing the sequence state, empty array if sequence has no state
	 * @throws LlamaException if sequence state cannot be retrieved
	 */
	public byte[] getSequenceState(int sequenceId) throws LlamaException {
		// First check if the sequence has any state
		long stateSize = getSequenceStateSize(sequenceId);
		if (stateSize == 0) {
			// Return empty array for sequences with no state - this is valid
			return new byte[0];
		}

		byte[] state = getSequenceStateData(sequenceId);
		if (state == null) {
			throw new LlamaException("Failed to get sequence state for sequence " + sequenceId);
		}
		return state;
	}

	/**
	 * Restore state for a specific sequence from a byte array.
	 *
	 * @param stateData byte array containing the sequence state
	 * @param sequenceId target sequence identifier
	 * @return number of bytes loaded from the state data
	 * @throws LlamaException if sequence state cannot be restored
	 */
	public long setSequenceState(byte[] stateData, int sequenceId) throws LlamaException {
		if (stateData == null) {
			throw new IllegalArgumentException("State data cannot be null");
		}
		long loaded = setSequenceStateData(stateData, sequenceId);
		if (loaded < 0) {
			throw new LlamaException("Failed to restore sequence state for sequence " + sequenceId);
		}
		return loaded;
	}

	/**
	 * Save state for a specific sequence to a file.
	 *
	 * @param filePath path where the sequence state should be saved
	 * @param sequenceId the sequence identifier
	 * @param tokens tokens for this sequence (can be null)
	 * @return number of bytes saved
	 * @throws LlamaException if sequence state cannot be saved
	 */
	public long saveSequenceState(String filePath, int sequenceId, int[] tokens) throws LlamaException {
		if (filePath == null || filePath.trim().isEmpty()) {
			throw new IllegalArgumentException("File path cannot be null or empty");
		}
		long saved = saveSequenceToFile(filePath, sequenceId, tokens);
		if (saved < 0) {
			throw new LlamaException("Failed to save sequence state to file: " + filePath);
		}
		return saved;
	}

	/**
	 * Save state for a specific sequence to a file (without tokens).
	 *
	 * @param filePath path where the sequence state should be saved
	 * @param sequenceId the sequence identifier
	 * @return number of bytes saved
	 * @throws LlamaException if sequence state cannot be saved
	 */
	public long saveSequenceState(String filePath, int sequenceId) throws LlamaException {
		return saveSequenceState(filePath, sequenceId, null);
	}

	/**
	 * Load state for a specific sequence from a file.
	 *
	 * @param filePath path to the sequence state file
	 * @param sequenceId target sequence identifier
	 * @param maxTokens maximum number of tokens to load
	 * @return array of tokens for this sequence
	 * @throws LlamaException if sequence state cannot be loaded
	 */
	public int[] loadSequenceState(String filePath, int sequenceId, int maxTokens) throws LlamaException {
		if (filePath == null || filePath.trim().isEmpty()) {
			throw new IllegalArgumentException("File path cannot be null or empty");
		}
		int[] tokens = loadSequenceFromFile(filePath, sequenceId, maxTokens);
		if (tokens == null) {
			throw new LlamaException("Failed to load sequence state from file: " + filePath);
		}
		return tokens;
	}

	/**
	 * Load state for a specific sequence from a file with default token limit.
	 *
	 * @param filePath path to the sequence state file
	 * @param sequenceId target sequence identifier
	 * @return array of tokens for this sequence
	 * @throws LlamaException if sequence state cannot be loaded
	 */
	public int[] loadSequenceState(String filePath, int sequenceId) throws LlamaException {
		return loadSequenceState(filePath, sequenceId, -1);
	}

	// ===== ADVANCED SAMPLING METHODS =====

	/**
	 * Create a greedy sampler that always selects the highest probability token.
	 * Deterministic sampling strategy with no randomness.
	 *
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createGreedySampler() throws LlamaException {
		long handle = createGreedySamplerNative();
		if (handle == -1) {
			throw new LlamaException("Failed to create greedy sampler");
		}
		return handle;
	}

	/**
	 * Create a distribution sampler for probabilistic token selection.
	 *
	 * @param seed random seed for reproducibility
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createDistributionSampler(int seed) throws LlamaException {
		long handle = createDistributionSamplerNative(seed);
		if (handle == -1) {
			throw new LlamaException("Failed to create distribution sampler");
		}
		return handle;
	}

	/**
	 * Create a Top-K sampler that considers only the K most likely tokens.
	 *
	 * @param k number of top tokens to consider (must be positive)
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createTopKSampler(int k) throws LlamaException {
		if (k <= 0) {
			throw new IllegalArgumentException("Top-K value must be positive");
		}
		long handle = createTopKSamplerNative(k);
		if (handle == -1) {
			throw new LlamaException("Failed to create top-k sampler");
		}
		return handle;
	}

	/**
	 * Create a Top-P (nucleus) sampler that considers tokens within probability mass p.
	 *
	 * @param p probability mass threshold (0.0-1.0)
	 * @param minKeep minimum number of tokens to keep
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createTopPSampler(float p, int minKeep) throws LlamaException {
		if (p < 0.0f || p > 1.0f) {
			throw new IllegalArgumentException("Top-P value must be between 0.0 and 1.0");
		}
		if (minKeep < 0) {
			throw new IllegalArgumentException("Min keep must be non-negative");
		}
		long handle = createTopPSamplerNative(p, minKeep);
		if (handle == -1) {
			throw new LlamaException("Failed to create top-p sampler");
		}
		return handle;
	}

	/**
	 * Create a Min-P sampler that filters tokens below probability threshold.
	 *
	 * @param p minimum probability threshold (0.0-1.0)
	 * @param minKeep minimum number of tokens to keep
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createMinPSampler(float p, int minKeep) throws LlamaException {
		if (p < 0.0f || p > 1.0f) {
			throw new IllegalArgumentException("Min-P value must be between 0.0 and 1.0");
		}
		if (minKeep < 0) {
			throw new IllegalArgumentException("Min keep must be non-negative");
		}
		long handle = createMinPSamplerNative(p, minKeep);
		if (handle == -1) {
			throw new LlamaException("Failed to create min-p sampler");
		}
		return handle;
	}

	/**
	 * Create a temperature sampler that controls randomness in token selection.
	 * Higher temperature = more random, lower temperature = more deterministic.
	 *
	 * @param temperature sampling temperature (0.0+ recommended 0.1-2.0)
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createTemperatureSampler(float temperature) throws LlamaException {
		if (temperature < 0.0f) {
			throw new IllegalArgumentException("Temperature must be non-negative");
		}
		long handle = createTemperatureSamplerNative(temperature);
		if (handle == -1) {
			throw new LlamaException("Failed to create temperature sampler");
		}
		return handle;
	}

	/**
	 * Create an extended temperature sampler with additional parameters.
	 *
	 * @param temp base temperature
	 * @param delta temperature adjustment
	 * @param exponent temperature exponent
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createExtendedTemperatureSampler(float temp, float delta, float exponent) throws LlamaException {
		if (temp < 0.0f) {
			throw new IllegalArgumentException("Temperature must be non-negative");
		}
		long handle = createExtendedTemperatureSamplerNative(temp, delta, exponent);
		if (handle == -1) {
			throw new LlamaException("Failed to create extended temperature sampler");
		}
		return handle;
	}

	/**
	 * Create a typical sampler that selects tokens with typical entropy.
	 *
	 * @param p typical sampling threshold (0.0-1.0)
	 * @param minKeep minimum number of tokens to keep
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createTypicalSampler(float p, int minKeep) throws LlamaException {
		if (p < 0.0f || p > 1.0f) {
			throw new IllegalArgumentException("Typical sampling p value must be between 0.0 and 1.0");
		}
		if (minKeep < 0) {
			throw new IllegalArgumentException("Min keep must be non-negative");
		}
		long handle = createTypicalSamplerNative(p, minKeep);
		if (handle == -1) {
			throw new LlamaException("Failed to create typical sampler");
		}
		return handle;
	}

	/**
	 * Create an XTC (Exclude Top Choices) sampler.
	 *
	 * @param p XTC probability threshold (0.0-1.0)
	 * @param t XTC threshold value
	 * @param minKeep minimum number of tokens to keep
	 * @param seed random seed
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createXtcSampler(float p, float t, int minKeep, int seed) throws LlamaException {
		if (p < 0.0f || p > 1.0f) {
			throw new IllegalArgumentException("XTC p value must be between 0.0 and 1.0");
		}
		if (t < 0.0f) {
			throw new IllegalArgumentException("XTC threshold must be non-negative");
		}
		if (minKeep < 0) {
			throw new IllegalArgumentException("Min keep must be non-negative");
		}
		long handle = createXtcSamplerNative(p, t, minKeep, seed);
		if (handle == -1) {
			throw new LlamaException("Failed to create XTC sampler");
		}
		return handle;
	}

	/**
	 * Create a Mirostat sampler for dynamic temperature adjustment.
	 *
	 * @param nVocab vocabulary size
	 * @param seed random seed
	 * @param tau target surprise level
	 * @param eta learning rate
	 * @param m window size
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createMirostatSampler(int nVocab, int seed, float tau, float eta, int m) throws LlamaException {
		if (nVocab <= 0) {
			throw new IllegalArgumentException("Vocabulary size must be positive");
		}
		if (tau <= 0.0f) {
			throw new IllegalArgumentException("Mirostat tau must be positive");
		}
		if (eta <= 0.0f) {
			throw new IllegalArgumentException("Mirostat eta must be positive");
		}
		if (m <= 0) {
			throw new IllegalArgumentException("Mirostat m must be positive");
		}
		long handle = createMirostatSamplerNative(nVocab, seed, tau, eta, m);
		if (handle == -1) {
			throw new LlamaException("Failed to create Mirostat sampler");
		}
		return handle;
	}

	/**
	 * Create a Mirostat V2 sampler (simplified version).
	 *
	 * @param seed random seed
	 * @param tau target surprise level
	 * @param eta learning rate
	 * @return sampler handle
	 * @throws LlamaException if the sampler cannot be created
	 */
	public static long createMirostatV2Sampler(int seed, float tau, float eta) throws LlamaException {
		if (tau <= 0.0f) {
			throw new IllegalArgumentException("Mirostat V2 tau must be positive");
		}
		if (eta <= 0.0f) {
			throw new IllegalArgumentException("Mirostat V2 eta must be positive");
		}
		long handle = createMirostatV2SamplerNative(seed, tau, eta);
		if (handle == -1) {
			throw new LlamaException("Failed to create Mirostat V2 sampler");
		}
		return handle;
	}


	/**
	 * Create a sampler chain for combining multiple samplers.
	 *
	 * @return sampler chain handle
	 * @throws LlamaException if the chain cannot be created
	 */
	public static long createSamplerChain() throws LlamaException {
		long handle = createSamplerChainNative();
		if (handle == -1) {
			throw new LlamaException("Failed to create sampler chain");
		}
		return handle;
	}

	/**
	 * Add a sampler to a sampler chain.
	 *
	 * @param chainHandle handle to the sampler chain
	 * @param samplerHandle handle to the sampler to add
	 * @throws LlamaException if the sampler cannot be added
	 */
	public static void addToSamplerChain(long chainHandle, long samplerHandle) throws LlamaException {
		if (chainHandle <= 0 || samplerHandle <= 0) {
			throw new IllegalArgumentException("Invalid sampler handle");
		}
		addToSamplerChainNative(chainHandle, samplerHandle);
	}

	/**
	 * Free a sampler and release its resources.
	 *
	 * @param samplerHandle handle to the sampler
	 */
	public static void freeSampler(long samplerHandle) {
		if (samplerHandle > 0) {
			freeSamplerNative(samplerHandle);
		}
	}

	/**
	 * Sample a token using the specified sampler.
	 *
	 * @param samplerHandle handle to the sampler
	 * @return sampled token ID
	 * @throws LlamaException if sampling fails
	 */
	public int sampleToken(long samplerHandle) throws LlamaException {
		if (samplerHandle <= 0) {
			throw new IllegalArgumentException("Invalid sampler handle");
		}
		int token = sampleTokenNative(samplerHandle);
		if (token == -1) {
			throw new LlamaException("Failed to sample token");
		}
		return token;
	}

	/**
	 * Accept a token in the sampler (for grammar/sequence tracking).
	 *
	 * @param samplerHandle handle to the sampler
	 * @param token token to accept
	 */
	public static void acceptToken(long samplerHandle, int token) {
		if (samplerHandle > 0) {
			acceptTokenNative(samplerHandle, token);
		}
	}

	/**
	 * Reset a sampler to its initial state.
	 *
	 * @param samplerHandle handle to the sampler
	 */
	public static void resetSampler(long samplerHandle) {
		if (samplerHandle > 0) {
			resetSamplerNative(samplerHandle);
		}
	}

	/**
	 * Get the name of a sampler.
	 *
	 * @param samplerHandle handle to the sampler
	 * @return sampler name
	 * @throws LlamaException if the name cannot be retrieved
	 */
	public static String getSamplerName(long samplerHandle) throws LlamaException {
		if (samplerHandle <= 0) {
			throw new IllegalArgumentException("Invalid sampler handle");
		}
		String name = getSamplerNameNative(samplerHandle);
		if (name == null) {
			throw new LlamaException("Failed to get sampler name");
		}
		return name;
	}

	// ===== LORA ADAPTER METHODS =====

	/**
	 * Load a LoRA adapter from file.
	 * LoRA (Low-Rank Adaptation) allows fine-tuning models with minimal parameters.
	 *
	 * @param loraPath path to the LoRA adapter file
	 * @return handle to the loaded adapter
	 * @throws LlamaException if the adapter cannot be loaded
	 */
	public long loadLoRAAdapter(String loraPath) throws LlamaException {
		if (loraPath == null || loraPath.trim().isEmpty()) {
			throw new IllegalArgumentException("LoRA path cannot be null or empty");
		}
		long handle = loadLoRAAdapterNative(loraPath);
		if (handle == -1) {
			throw new LlamaException("Failed to load LoRA adapter from: " + loraPath);
		}
		return handle;
	}

	/**
	 * Free a LoRA adapter and release its resources.
	 *
	 * @param adapterHandle handle to the adapter
	 */
	public void freeLoRAAdapter(long adapterHandle) {
		if (adapterHandle != -1) {
			freeLoRAAdapterNative(adapterHandle);
		}
	}

	/**
	 * Apply a LoRA adapter to the current context with the specified scale.
	 *
	 * @param adapterHandle handle to the adapter
	 * @param scale scale factor for the adapter (typically 1.0)
	 * @return 0 on success, negative on error
	 * @throws LlamaException if the adapter cannot be applied
	 */
	public int setLoRAAdapter(long adapterHandle, float scale) throws LlamaException {
		int result = setLoRAAdapterNative(adapterHandle, scale);
		if (result < 0) {
			throw new LlamaException("Failed to apply LoRA adapter");
		}
		return result;
	}

	/**
	 * Apply a LoRA adapter with default scale of 1.0.
	 *
	 * @param adapterHandle handle to the adapter
	 * @return 0 on success, negative on error
	 * @throws LlamaException if the adapter cannot be applied
	 */
	public int setLoRAAdapter(long adapterHandle) throws LlamaException {
		return setLoRAAdapter(adapterHandle, 1.0f);
	}

	/**
	 * Remove a specific LoRA adapter from the context.
	 *
	 * @param adapterHandle handle to the adapter
	 * @return 0 on success, negative on error
	 * @throws LlamaException if the adapter cannot be removed
	 */
	public int removeLoRAAdapter(long adapterHandle) throws LlamaException {
		int result = removeLoRAAdapterNative(adapterHandle);
		if (result < 0) {
			throw new LlamaException("Failed to remove LoRA adapter");
		}
		return result;
	}

	/**
	 * Clear all LoRA adapters from the current context.
	 */
	public void clearLoRAAdapters() {
		clearLoRAAdaptersNative();
	}

	/**
	 * Apply a control vector to guide model behavior.
	 * Control vectors can steer generation towards specific styles or topics.
	 *
	 * @param controlVector float array containing the control vector, or null to clear
	 * @return 0 on success, negative on error
	 * @throws LlamaException if the control vector cannot be applied
	 */
	public int applyControlVector(float[] controlVector) throws LlamaException {
		int result = applyControlVectorNative(controlVector);
		if (result < 0) {
			throw new LlamaException("Failed to apply control vector");
		}
		return result;
	}

	/**
	 * Clear the current control vector.
	 *
	 * @return 0 on success, negative on error
	 * @throws LlamaException if the control vector cannot be cleared
	 */
	public int clearControlVector() throws LlamaException {
		return applyControlVector(null);
	}

	/**
	 * Get metadata value from a LoRA adapter.
	 *
	 * @param adapterHandle handle to the adapter
	 * @param key metadata key to retrieve
	 * @return metadata value, or null if not found
	 * @throws LlamaException if the adapter handle is invalid
	 */
	public String getLoRAAdapterMetadata(long adapterHandle, String key) throws LlamaException {
		if (key == null || key.trim().isEmpty()) {
			throw new IllegalArgumentException("Key cannot be null or empty");
		}
		String value = getLoRAAdapterMetadataNative(adapterHandle, key);
		return value;
	}

	/**
	 * Get the number of metadata entries in a LoRA adapter.
	 *
	 * @param adapterHandle handle to the adapter
	 * @return number of metadata entries
	 * @throws LlamaException if the adapter handle is invalid
	 */
	public int getLoRAAdapterMetadataCount(long adapterHandle) throws LlamaException {
		int count = getLoRAAdapterMetadataCountNative(adapterHandle);
		if (count < 0) {
			throw new LlamaException("Failed to get metadata count for adapter");
		}
		return count;
	}

	/**
	 * Get metadata key by index from a LoRA adapter.
	 *
	 * @param adapterHandle handle to the adapter
	 * @param index index of the metadata entry
	 * @return metadata key, or null if index is out of bounds
	 * @throws LlamaException if the adapter handle is invalid
	 */
	public String getLoRAAdapterMetadataKey(long adapterHandle, int index) throws LlamaException {
		return getLoRAAdapterMetadataKeyNative(adapterHandle, index);
	}

	/**
	 * Get metadata value by index from a LoRA adapter.
	 *
	 * @param adapterHandle handle to the adapter
	 * @param index index of the metadata entry
	 * @return metadata value, or null if index is out of bounds
	 * @throws LlamaException if the adapter handle is invalid
	 */
	public String getLoRAAdapterMetadataValue(long adapterHandle, int index) throws LlamaException {
		return getLoRAAdapterMetadataValueNative(adapterHandle, index);
	}

	/**
	 * Check if the LoRA adapter is an ALORA (Adaptive LoRA) and get invocation token count.
	 *
	 * @param adapterHandle handle to the adapter
	 * @return number of invocation tokens, or 0 if not ALORA
	 * @throws LlamaException if the adapter handle is invalid
	 */
	public long getAloraInvocationTokenCount(long adapterHandle) throws LlamaException {
		return getAloraInvocationTokenCountNative(adapterHandle);
	}

	/**
	 * Get ALORA invocation tokens if the adapter supports them.
	 *
	 * @param adapterHandle handle to the adapter
	 * @return array of invocation tokens, or empty array if not ALORA
	 * @throws LlamaException if the adapter handle is invalid
	 */
	public int[] getAloraInvocationTokens(long adapterHandle) throws LlamaException {
		int[] tokens = getAloraInvocationTokensNative(adapterHandle);
		return tokens != null ? tokens : new int[0];
	}

	// ===== MEMORY/KV CACHE MANAGEMENT =====

	/**
	 * Copy KV cache data from one sequence to another within a position range.
	 * Useful for branching conversations or saving checkpoints.
	 *
	 * @param srcSeqId source sequence ID to copy from
	 * @param dstSeqId destination sequence ID to copy to
	 * @param p0 start position (inclusive)
	 * @param p1 end position (exclusive, -1 for end of sequence)
	 * @throws LlamaException if the copy operation fails
	 */
	public void copySequence(int srcSeqId, int dstSeqId, int p0, int p1) throws LlamaException {
		if (srcSeqId < 0 || dstSeqId < 0) {
			throw new IllegalArgumentException("Sequence IDs must be non-negative");
		}
		if (p0 < 0) {
			throw new IllegalArgumentException("Start position must be non-negative");
		}
		if (p1 >= 0 && p1 <= p0) {
			throw new IllegalArgumentException("End position must be greater than start position");
		}
		copySequenceNative(srcSeqId, dstSeqId, p0, p1);
	}

	/**
	 * Mark a sequence to be kept in memory while clearing others.
	 * Useful for memory management when working with multiple sequences.
	 *
	 * @param seqId sequence ID to keep
	 * @throws LlamaException if the operation fails
	 */
	public void keepSequence(int seqId) throws LlamaException {
		if (seqId < 0) {
			throw new IllegalArgumentException("Sequence ID must be non-negative");
		}
		keepSequenceNative(seqId);
	}

	/**
	 * Add a position delta to all positions in a sequence within a range.
	 * Useful for shifting sequence positions after insertions.
	 *
	 * @param seqId sequence ID to modify
	 * @param p0 start position (inclusive)
	 * @param p1 end position (exclusive, -1 for end of sequence)
	 * @param delta position delta to add (can be negative)
	 * @throws LlamaException if the operation fails
	 */
	public void addPositionDelta(int seqId, int p0, int p1, int delta) throws LlamaException {
		if (seqId < 0) {
			throw new IllegalArgumentException("Sequence ID must be non-negative");
		}
		if (p0 < 0) {
			throw new IllegalArgumentException("Start position must be non-negative");
		}
		if (p1 >= 0 && p1 <= p0) {
			throw new IllegalArgumentException("End position must be greater than start position");
		}
		addPositionDeltaNative(seqId, p0, p1, delta);
	}

	/**
	 * Divide all positions in a sequence within a range by a divisor.
	 * Useful for position compression or scaling operations.
	 *
	 * @param seqId sequence ID to modify
	 * @param p0 start position (inclusive)
	 * @param p1 end position (exclusive, -1 for end of sequence)
	 * @param divisor divisor to divide positions by (must be positive)
	 * @throws LlamaException if the operation fails
	 */
	public void dividePositions(int seqId, int p0, int p1, int divisor) throws LlamaException {
		if (seqId < 0) {
			throw new IllegalArgumentException("Sequence ID must be non-negative");
		}
		if (p0 < 0) {
			throw new IllegalArgumentException("Start position must be non-negative");
		}
		if (p1 >= 0 && p1 <= p0) {
			throw new IllegalArgumentException("End position must be greater than start position");
		}
		if (divisor <= 0) {
			throw new IllegalArgumentException("Divisor must be positive");
		}
		dividePositionsNative(seqId, p0, p1, divisor);
	}

	/**
	 * Get the minimum position for a sequence in the KV cache.
	 *
	 * @param seqId sequence ID to query
	 * @return minimum position in the sequence
	 * @throws LlamaException if the operation fails
	 */
	public int getSequenceMinPosition(int seqId) throws LlamaException {
		if (seqId < 0) {
			throw new IllegalArgumentException("Sequence ID must be non-negative");
		}
		return getSequenceMinPositionNative(seqId);
	}

	/**
	 * Get the maximum position for a sequence in the KV cache.
	 *
	 * @param seqId sequence ID to query
	 * @return maximum position in the sequence
	 * @throws LlamaException if the operation fails
	 */
	public int getSequenceMaxPosition(int seqId) throws LlamaException {
		if (seqId < 0) {
			throw new IllegalArgumentException("Sequence ID must be non-negative");
		}
		return getSequenceMaxPositionNative(seqId);
	}

	/**
	 * Check if the memory system supports context shifting.
	 * Context shifting allows extending conversations beyond the context window.
	 *
	 * @return true if context shifting is supported
	 * @throws LlamaException if the capability cannot be determined
	 */
	public boolean canShiftContext() throws LlamaException {
		return canShiftContextNative();
	}

	/**
	 * Clear the KV cache memory with options for data clearing.
	 *
	 * @param clearData if true, clear the actual data; if false, only clear metadata
	 * @throws LlamaException if the operation fails
	 */
	public void clearMemory(boolean clearData) throws LlamaException {
		clearMemoryNative(clearData);
	}

	/**
	 * Clear the KV cache memory (both data and metadata).
	 *
	 * @throws LlamaException if the operation fails
	 */
	public void clearMemory() throws LlamaException {
		clearMemory(true);
	}

	/**
	 * Remove tokens from a sequence within a specific position range.
	 *
	 * @param seqId sequence ID to modify
	 * @param p0 start position (inclusive)
	 * @param p1 end position (exclusive, -1 for end of sequence)
	 * @return true if tokens were removed, false otherwise
	 * @throws LlamaException if the operation fails
	 */
	public boolean removeSequenceTokens(int seqId, int p0, int p1) throws LlamaException {
		if (seqId < 0) {
			throw new IllegalArgumentException("Sequence ID must be non-negative");
		}
		if (p0 < 0) {
			throw new IllegalArgumentException("Start position must be non-negative");
		}
		if (p1 >= 0 && p1 <= p0) {
			throw new IllegalArgumentException("End position must be greater than start position");
		}
		return removeSequenceTokensNative(seqId, p0, p1);
	}

	// ===== MODEL INFORMATION ACCESS =====

	/**
	 * Get the total number of parameters in the model.
	 * Useful for understanding model complexity and memory requirements.
	 *
	 * @return the total parameter count, or -1 if unavailable
	 * @throws LlamaException if the operation fails
	 */
	public long getModelParameterCount() throws LlamaException {
		return getModelParameterCountNative();
	}

	/**
	 * Get the total size of the model in bytes.
	 * Useful for understanding storage and memory requirements.
	 *
	 * @return the model size in bytes, or -1 if unavailable
	 * @throws LlamaException if the operation fails
	 */
	public long getModelSize() throws LlamaException {
		return getModelSizeNative();
	}

	/**
	 * Get the number of metadata entries in the model.
	 * Model metadata contains information about the model architecture, training, etc.
	 *
	 * @return the number of metadata entries
	 * @throws LlamaException if the operation fails
	 */
	public int getModelMetadataCount() throws LlamaException {
		return getModelMetadataCountNative();
	}

	/**
	 * Get a metadata key by its index.
	 *
	 * @param index the metadata entry index (0-based)
	 * @return the metadata key, or empty string if not found
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if index is negative
	 */
	public String getModelMetadataKey(int index) throws LlamaException {
		if (index < 0) {
			throw new IllegalArgumentException("Metadata index must be non-negative");
		}
		return getModelMetadataKeyByIndexNative(index);
	}

	/**
	 * Get a metadata value by its index.
	 *
	 * @param index the metadata entry index (0-based)
	 * @return the metadata value, or empty string if not found
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if index is negative
	 */
	public String getModelMetadataValueByIndex(int index) throws LlamaException {
		if (index < 0) {
			throw new IllegalArgumentException("Metadata index must be non-negative");
		}
		return getModelMetadataValueByIndexNative(index);
	}

	/**
	 * Get a metadata value by its key.
	 *
	 * @param key the metadata key to look up
	 * @return the metadata value, or empty string if not found
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if key is null
	 */
	public String getModelMetadataValue(String key) throws LlamaException {
		if (key == null) {
			throw new IllegalArgumentException("Metadata key cannot be null");
		}
		return getModelMetadataValueNative(key);
	}

	/**
	 * Get all model metadata as a Map for easy access.
	 * This is a convenience method that calls the individual metadata functions.
	 *
	 * @return a Map containing all metadata key-value pairs
	 * @throws LlamaException if the operation fails
	 */
	public Map<String, String> getModelMetadata() throws LlamaException {
		Map<String, String> metadata = new java.util.HashMap<>();
		int count = getModelMetadataCount();

		for (int i = 0; i < count; i++) {
			String key = getModelMetadataKey(i);
			String value = getModelMetadataValueByIndex(i);
			if (!key.isEmpty()) {
				metadata.put(key, value);
			}
		}

		return metadata;
	}

	// ===== VOCABULARY INFORMATION =====

	/**
	 * Get the vocabulary type used by this model.
	 * Different models use different tokenization approaches.
	 *
	 * @return the vocabulary type as an integer constant
	 * @throws LlamaException if the operation fails
	 */
	public int getVocabularyType() throws LlamaException {
		return getVocabularyTypeNative();
	}

	/**
	 * Get the total size of the vocabulary (number of tokens).
	 *
	 * @return the vocabulary size, or -1 if unavailable
	 * @throws LlamaException if the operation fails
	 */
	public int getVocabularySize() throws LlamaException {
		return getVocabularySizeNative();
	}

	/**
	 * Get the text representation of a token.
	 *
	 * @param tokenId the token ID to look up
	 * @return the text for this token, or empty string if not found
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if token ID is invalid
	 */
	public String getTokenText(int tokenId) throws LlamaException {
		if (tokenId < 0) {
			throw new IllegalArgumentException("Token ID must be non-negative");
		}
		int vocabSize = getVocabularySize();
		if (tokenId < vocabSize) {
			return getTokenTextNative(tokenId);
		}
		throw new IllegalArgumentException("Token ID larger than vocabulary size");
	}

	/**
	 * Get the score (probability) associated with a token.
	 * Scores are used during tokenization to determine the best token segmentation.
	 *
	 * @param tokenId the token ID to look up
	 * @return the token score, or 0.0 if unavailable
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if token ID is invalid
	 */
	public float getTokenScore(int tokenId) throws LlamaException {
		if (tokenId < 0) {
			throw new IllegalArgumentException("Token ID must be non-negative");
		}
		return getTokenScoreNative(tokenId);
	}

	/**
	 * Get the attributes/flags associated with a token.
	 * Token attributes provide information about token properties.
	 *
	 * @param tokenId the token ID to look up
	 * @return the token attributes as bit flags, or 0 if unavailable
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if token ID is invalid
	 */
	public int getTokenAttributes(int tokenId) throws LlamaException {
		if (tokenId < 0) {
			throw new IllegalArgumentException("Token ID must be non-negative");
		}
		return getTokenAttributesNative(tokenId);
	}

	// ===== SPECIAL TOKENS =====

	/**
	 * Get the Beginning-of-Sentence (BOS) token ID.
	 *
	 * @return the BOS token ID, or -1 if not available
	 * @throws LlamaException if the operation fails
	 */
	public int getBosToken() throws LlamaException {
		return getBosTokenNative();
	}

	/**
	 * Get the End-of-Sentence (EOS) token ID.
	 *
	 * @return the EOS token ID, or -1 if not available
	 * @throws LlamaException if the operation fails
	 */
	public int getEosToken() throws LlamaException {
		return getEosTokenNative();
	}

	/**
	 * Get the End-of-Turn (EOT) token ID.
	 *
	 * @return the EOT token ID, or -1 if not available
	 * @throws LlamaException if the operation fails
	 */
	public int getEotToken() throws LlamaException {
		return getEotTokenNative();
	}

	/**
	 * Get the Separator (SEP) token ID.
	 *
	 * @return the SEP token ID, or -1 if not available
	 * @throws LlamaException if the operation fails
	 */
	public int getSepToken() throws LlamaException {
		return getSepTokenNative();
	}

	/**
	 * Get the Newline (NL) token ID.
	 *
	 * @return the NL token ID, or -1 if not available
	 * @throws LlamaException if the operation fails
	 */
	public int getNlToken() throws LlamaException {
		return getNlTokenNative();
	}

	/**
	 * Get the Padding (PAD) token ID.
	 *
	 * @return the PAD token ID, or -1 if not available
	 * @throws LlamaException if the operation fails
	 */
	public int getPadToken() throws LlamaException {
		return getPadTokenNative();
	}

	/**
	 * Check if a token is an End-of-Generation token.
	 * EOG tokens signal the end of text generation.
	 *
	 * @param tokenId the token ID to check
	 * @return true if the token is an EOG token, false otherwise
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if token ID is invalid
	 */
	public boolean isEogToken(int tokenId) throws LlamaException {
		if (tokenId < 0) {
			throw new IllegalArgumentException("Token ID must be non-negative");
		}
		return isEogTokenNative(tokenId);
	}

	/**
	 * Check if a token is a control token.
	 * Control tokens have special meaning and are typically not displayed.
	 *
	 * @param tokenId the token ID to check
	 * @return true if the token is a control token, false otherwise
	 * @throws LlamaException if the operation fails
	 * @throws IllegalArgumentException if token ID is invalid
	 */
	public boolean isControlToken(int tokenId) throws LlamaException {
		if (tokenId < 0) {
			throw new IllegalArgumentException("Token ID must be non-negative");
		}
		return isControlTokenNative(tokenId);
	}

	/**
	 * Sets a callback for native llama.cpp log messages.
	 * Per default, log messages are written in JSON to stdout. Note, that in text mode the callback will be also
	 * invoked with log messages of the GGML backend, while JSON mode can only access request log messages.
	 * In JSON mode, GGML messages will still be written to stdout.
	 * To only change the log format but keep logging to stdout, the given callback can be <code>null</code>.
	 * To disable logging, pass an empty callback, i.e., <code>(level, msg) -> {}</code>.
	 *
	 * @param format the log format to use
	 * @param callback a method to call for log messages
	 */
	public static native void setLogger(LogFormat format, @Nullable BiConsumer<LogLevel, String> callback);

	@Override
	public void close() {
		delete();
	}

	// don't overload native methods since the C++ function names get nasty
	native int requestCompletion(String params) throws LlamaException;

	native LlamaOutput receiveCompletion(int taskId) throws LlamaException;

	native void cancelCompletion(int taskId);

	native byte[] decodeBytes(int[] tokens);

	private native void loadModel(String... parameters) throws LlamaException;

	private native void delete();

	native void releaseTask(int taskId);

	private static native byte[] jsonSchemaToGrammarBytes(String schema);

	// State persistence native methods
	private native long getStateSize();
	private native byte[] getStateData();
	private native long setStateData(byte[] stateData);
	private native boolean saveStateToFile(String path, int[] tokens);
	private native int[] loadStateFromFile(String path, int maxTokens);

	// Sequence-specific state persistence
	private native long getSequenceStateSizeNative(int sequenceId);
	private native byte[] getSequenceStateData(int sequenceId);
	private native long setSequenceStateData(byte[] stateData, int sequenceId);
	private native long saveSequenceToFile(String path, int sequenceId, int[] tokens);
	private native int[] loadSequenceFromFile(String path, int sequenceId, int maxTokens);

	// LoRA adapter native methods
	private native long loadLoRAAdapterNative(String loraPath);
	private native void freeLoRAAdapterNative(long adapterHandle);
	private native int setLoRAAdapterNative(long adapterHandle, float scale);
	private native int removeLoRAAdapterNative(long adapterHandle);
	private native void clearLoRAAdaptersNative();
	private native int applyControlVectorNative(float[] data);
	private native String getLoRAAdapterMetadataNative(long adapterHandle, String key);
	private native int getLoRAAdapterMetadataCountNative(long adapterHandle);
	private native String getLoRAAdapterMetadataKeyNative(long adapterHandle, int index);
	private native String getLoRAAdapterMetadataValueNative(long adapterHandle, int index);
	private native long getAloraInvocationTokenCountNative(long adapterHandle);
	private native int[] getAloraInvocationTokensNative(long adapterHandle);

	// ========================================
	// Model-dependent samplers (instance methods)
	// ========================================

	/**
	 * Create a DRY (Don't Repeat Yourself) sampler that requires model context.
	 * @param nCtxTrain Number of context tokens for training
	 * @param multiplier DRY multiplier
	 * @param base DRY base value
	 * @param allowedLength Allowed repetition length
	 * @param penaltyLastN Number of last tokens to penalize
	 * @param sequenceBreakers Token IDs that break sequences
	 * @return Handle to the created sampler
	 */
	public long createDrySampler(int nCtxTrain, float multiplier, float base,
			int allowedLength, int penaltyLastN, int[] sequenceBreakers) {
		return createDrySamplerNative(nCtxTrain, multiplier, base, allowedLength, penaltyLastN, sequenceBreakers);
	}

	/**
	 * Create a grammar sampler that constrains output to match a grammar.
	 * @param grammarStr GBNF grammar string
	 * @param rootRule Root rule name (can be null for "root")
	 * @return Handle to the created sampler
	 */
	public long createGrammarSampler(String grammarStr, String rootRule) {
		return createGrammarSamplerNative(grammarStr, rootRule);
	}

	/**
	 * Create a grammar sampler with default root rule.
	 * @param grammarStr GBNF grammar string
	 * @return Handle to the created sampler
	 */
	public long createGrammarSampler(String grammarStr) {
		return createGrammarSampler(grammarStr, null);
	}

	/**
	 * Create an infill sampler for code completion tasks.
	 * @return Handle to the created sampler
	 */
	public long createInfillSampler() {
		return createInfillSamplerNative();
	}


	/**
	 * Create a logit bias sampler for token probability manipulation.
	 * @param nVocab Vocabulary size
	 * @param biasTokens Array of token IDs to bias
	 * @param biasValues Array of bias values (same length as biasTokens)
	 * @return Handle to the created sampler
	 */
	public long createLogitBiasSampler(int nVocab, int[] biasTokens, float[] biasValues) {
		if (biasTokens.length != biasValues.length) {
			throw new IllegalArgumentException("biasTokens and biasValues must have the same length");
		}
		return createLogitBiasSamplerNative(nVocab, biasTokens.length, biasTokens, biasValues);
	}

	// Advanced sampling native methods (DEPRECATED - Use LlamaSampler class instead)
	private static native long createGreedySamplerNative();
	private static native long createDistributionSamplerNative(int seed);
	private static native long createTopKSamplerNative(int k);
	private static native long createTopPSamplerNative(float p, int minKeep);
	private static native long createMinPSamplerNative(float p, int minKeep);
	private static native long createTemperatureSamplerNative(float temperature);
	private static native long createExtendedTemperatureSamplerNative(float temp, float delta, float exponent);
	private static native long createTypicalSamplerNative(float p, int minKeep);
	private static native long createXtcSamplerNative(float p, float t, int minKeep, int seed);
	private static native long createTopNSigmaSamplerNative(float n);
	private static native long createMirostatSamplerNative(int nVocab, int seed, float tau, float eta, int m);
	private static native long createMirostatV2SamplerNative(int seed, float tau, float eta);
	private static native long createPenaltiesSamplerNative(int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent);
	private native long createDrySamplerNative(int nCtxTrain, float multiplier, float base, int allowedLength, int penaltyLastN, int[] sequenceBreakers);
	private static native long createLogitBiasSamplerNative(int nVocab, int nLogitBias, int[] biasTokens, float[] biasValues);
	private native long createGrammarSamplerNative(String grammarStr, String rootRule);
	private native long createInfillSamplerNative();
	private static native long createSamplerChainNative();
	private static native void addToSamplerChainNative(long chainHandle, long samplerHandle);
	private static native long cloneSamplerNative(long samplerHandle);
	private static native void freeSamplerNative(long samplerHandle);
	private native int sampleTokenNative(long samplerHandle);
	private static native void acceptTokenNative(long samplerHandle, int token);
	private static native void resetSamplerNative(long samplerHandle);
	private static native String getSamplerNameNative(long samplerHandle);

	// Memory/KV cache management native methods
	private native void copySequenceNative(int srcSeqId, int dstSeqId, int p0, int p1);
	private native void keepSequenceNative(int seqId);
	private native void addPositionDeltaNative(int seqId, int p0, int p1, int delta);
	private native void dividePositionsNative(int seqId, int p0, int p1, int divisor);
	private native int getSequenceMinPositionNative(int seqId);
	private native int getSequenceMaxPositionNative(int seqId);
	private native boolean canShiftContextNative();
	private native void clearMemoryNative(boolean clearData);
	private native boolean removeSequenceTokensNative(int seqId, int p0, int p1);

	// Model information native methods
	private native long getModelParameterCountNative();
	private native long getModelSizeNative();
	private native int getModelMetadataCountNative();
	private native String getModelMetadataKeyByIndexNative(int index);
	private native String getModelMetadataValueByIndexNative(int index);
	private native String getModelMetadataValueNative(String key);
	private native int getVocabularyTypeNative();
	private native int getVocabularySizeNative();
	private native String getTokenTextNative(int tokenId);
	private native float getTokenScoreNative(int tokenId);
	private native int getTokenAttributesNative(int tokenId);
	private native int getBosTokenNative();
	private native int getEosTokenNative();
	private native int getEotTokenNative();
	private native int getSepTokenNative();
	private native int getNlTokenNative();
	private native int getPadTokenNative();
	private native boolean isEogTokenNative(int tokenId);
	private native boolean isControlTokenNative(int tokenId);

	public static String jsonSchemaToGrammar(String schema) {
		return new String(jsonSchemaToGrammarBytes(schema), StandardCharsets.UTF_8);
	}

	public List<Pair<String, Float>> rerank(boolean reRank, String query, String ... documents) {
		LlamaOutput output = rerank(query, documents);

		Map<String, Float> scoredDocumentMap = output.probabilities;

		List<Pair<String, Float>> rankedDocuments = new ArrayList<>();

		if (reRank) {
			// Sort in descending order based on Float values
			scoredDocumentMap.entrySet()
				.stream()
				.sorted((a, b) -> Float.compare(b.getValue(), a.getValue())) // Descending order
				.forEach(entry -> rankedDocuments.add(new Pair<>(entry.getKey(), entry.getValue())));
		} else {
			// Copy without sorting
			scoredDocumentMap.forEach((key, value) -> rankedDocuments.add(new Pair<>(key, value)));
		}

		return rankedDocuments;
	}

	public native LlamaOutput rerank(String query, String... documents);

	public  String applyTemplate(InferenceParameters parameters) {
		return applyTemplate(parameters.toString());
	}
	public native String applyTemplate(String parametersJson);
}
