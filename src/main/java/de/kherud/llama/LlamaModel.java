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
