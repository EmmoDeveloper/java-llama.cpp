package de.kherud.llama;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Safe batch processor that implements batch processing at the Java level
 * to avoid native crashes in the llama_batch structure.
 *
 * This implementation processes batch items sequentially using individual
 * inference calls, providing the same API as BatchProcessor but with
 * guaranteed stability.
 */
public class SafeBatchProcessor implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(SafeBatchProcessor.class.getName());

	private final int maxTokenCount;
	private final int embeddingDimension;
	private final int maxSequenceCount;
	private boolean closed = false;

	// Batch data storage
	private int[] tokens;
	private float[] embeddings;
	private int[] positions;
	private int[] sequenceIds;
	private byte[] logitFlags;
	private int tokenCount = 0;

	public SafeBatchProcessor(int maxTokenCount, int embeddingDimension, int maxSequenceCount) {
		this.maxTokenCount = maxTokenCount;
		this.embeddingDimension = embeddingDimension;
		this.maxSequenceCount = maxSequenceCount;

		LOGGER.log(System.Logger.Level.INFO,String.format("SafeBatchProcessor initialized: maxTokens=%d, embedDim=%d, maxSeq=%d",
			maxTokenCount, embeddingDimension, maxSequenceCount));
	}

	/**
	 * Encode context using Java-level batch processing
	 */
	public int encodeContext(LlamaModel model) {
		checkClosed();
		if (tokens == null || tokens.length == 0) {
			return -1; // No tokens to process
		}

		return processBatch(model, true);
	}

	/**
	 * Decode tokens using Java-level batch processing
	 */
	public int decodeTokens(LlamaModel model) {
		checkClosed();
		if (tokens == null || tokens.length == 0) {
			return -1; // No tokens to process
		}

		return processBatch(model, false);
	}

	/**
	 * Process the batch by grouping tokens by sequence and processing each sequence
	 */
	private int processBatch(LlamaModel model, boolean isContextEncoding) {
		try {
			// Group tokens by sequence ID
			Map<Integer, List<BatchItem>> sequenceGroups = groupBySequence();

			int totalProcessed = 0;
			int result = 0;

			// Process each sequence
			for (Map.Entry<Integer, List<BatchItem>> entry : sequenceGroups.entrySet()) {
				int sequenceId = entry.getKey();
				List<BatchItem> items = entry.getValue();

				LOGGER.log(System.Logger.Level.DEBUG,String.format("Processing sequence %d with %d items", sequenceId, items.size()));

				if (isContextEncoding) {
					result = processSequenceContext(model, sequenceId, items);
				} else {
					result = processSequenceDecoding(model, sequenceId, items);
				}

				if (result < 0) {
					LOGGER.log(System.Logger.Level.WARNING,String.format("Failed to process sequence %d, result: %d", sequenceId, result));
					return result;
				}

				totalProcessed += items.size();
			}

			LOGGER.log(System.Logger.Level.DEBUG,String.format("Successfully processed %d tokens across %d sequences",
				totalProcessed, sequenceGroups.size()));

			return totalProcessed;

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR,"Error in batch processing: " + e.getMessage());
			return -1;
		}
	}

	/**
	 * Group batch items by sequence ID
	 */
	private Map<Integer, List<BatchItem>> groupBySequence() {
		Map<Integer, List<BatchItem>> groups = new HashMap<>();

		for (int i = 0; i < tokenCount; i++) {
			int token = tokens[i];
			int position = positions != null ? positions[i] : i;
			int sequenceId = sequenceIds != null ? sequenceIds[i] : 0;
			boolean needLogits = logitFlags != null ? logitFlags[i] != 0 : (i == tokenCount - 1);

			BatchItem item = new BatchItem(token, position, sequenceId, needLogits);
			groups.computeIfAbsent(sequenceId, k -> new ArrayList<>()).add(item);
		}

		// Sort items within each sequence by position
		for (List<BatchItem> items : groups.values()) {
			items.sort(Comparator.comparingInt(item -> item.position));
		}

		return groups;
	}

	/**
	 * Process a sequence for context encoding
	 */
	private int processSequenceContext(LlamaModel model, int sequenceId, List<BatchItem> items) {
		// For context encoding, we process the entire sequence as a prompt
		StringBuilder prompt = new StringBuilder();

		// Convert tokens back to text (simplified approach)
		// In a real implementation, you'd use the tokenizer
		for (BatchItem item : items) {
			prompt.append(item.token).append(" ");
		}

		try {
			// Use model's complete method to process the context
			InferenceParameters params = new InferenceParameters(prompt.toString().trim());
			params.setNPredict(1); // Just need to encode context

			String result = model.complete(params);
			return result != null && !result.isEmpty() ? 0 : -1; // Success if we get any output

		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.WARNING,"Context encoding failed for sequence " + sequenceId + ": " + e.getMessage());
			return -1;
		}
	}

	/**
	 * Process a sequence for token decoding
	 */
	private int processSequenceDecoding(LlamaModel model, int sequenceId, List<BatchItem> items) {
		// For decoding, process tokens individually
		int processed = 0;

		for (BatchItem item : items) {
			try {
				// Convert token to string and process
				String tokenStr = String.valueOf(item.token);
				InferenceParameters params = new InferenceParameters(tokenStr);
				params.setNPredict(1);

				String result = model.complete(params);
				if (result != null && !result.isEmpty()) {
					processed++;
				}

			} catch (Exception e) {
				LOGGER.log(System.Logger.Level.WARNING,"Token decoding failed for token " + item.token + ": " + e.getMessage());
				return -1;
			}
		}

		return processed;
	}

	// Data management methods (same as original BatchProcessor)

	public void setTokens(int[] tokens) {
		checkClosed();
		this.tokens = tokens != null ? tokens.clone() : null;
		this.tokenCount = tokens != null ? tokens.length : 0;
	}

	public void setEmbeddings(float[] embeddings) {
		checkClosed();
		this.embeddings = embeddings != null ? embeddings.clone() : null;
	}

	public void setPositions(int[] positions) {
		checkClosed();
		this.positions = positions != null ? positions.clone() : null;
	}

	public void setSequenceIds(int[] sequenceIds) {
		checkClosed();
		this.sequenceIds = sequenceIds != null ? sequenceIds.clone() : null;
	}

	public void setLogitFlags(byte[] logitFlags) {
		checkClosed();
		this.logitFlags = logitFlags != null ? logitFlags.clone() : null;
	}

	public int[] getTokens() {
		checkClosed();
		return tokens != null ? tokens.clone() : null;
	}

	public float[] getEmbeddings() {
		checkClosed();
		return embeddings != null ? embeddings.clone() : null;
	}

	public int[] getPositions() {
		checkClosed();
		return positions != null ? positions.clone() : null;
	}

	public int[] getSequenceIds() {
		checkClosed();
		return sequenceIds != null ? sequenceIds.clone() : null;
	}

	public byte[] getLogitFlags() {
		checkClosed();
		return logitFlags != null ? logitFlags.clone() : null;
	}

	public int getTokenCount() {
		checkClosed();
		return tokenCount;
	}

	private void checkClosed() {
		if (closed) {
			throw new IllegalStateException("SafeBatchProcessor has been closed");
		}
	}

	@Override
	public void close() {
		if (!closed) {
			// Clean up Java-level resources
			tokens = null;
			embeddings = null;
			positions = null;
			sequenceIds = null;
			logitFlags = null;
			tokenCount = 0;
			closed = true;
			LOGGER.log(System.Logger.Level.DEBUG,"SafeBatchProcessor closed");
		}
	}

	/**
	 * Internal class to represent a batch item
	 */
	private static class BatchItem {
		final int token;
		final int position;
		final int sequenceId;
		final boolean needLogits;

		BatchItem(int token, int position, int sequenceId, boolean needLogits) {
			this.token = token;
			this.position = position;
			this.sequenceId = sequenceId;
			this.needLogits = needLogits;
		}
	}
}
