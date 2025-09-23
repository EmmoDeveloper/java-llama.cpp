package de.kherud.llama;

import java.util.logging.Logger;

public class BatchProcessor implements AutoCloseable {
	private static final Logger LOGGER = Logger.getLogger(BatchProcessor.class.getName());
	private static final boolean USE_SAFE_FALLBACK = Boolean.parseBoolean(
		System.getProperty("llama.batch.safe_fallback", "true"));

	static { LlamaLoader.initialize(); }

	private final long batchHandle;
	private final SafeBatchProcessor safeFallback;
	private final boolean usingSafeFallback;
	private boolean closed = false;

	public BatchProcessor(int maxTokenCount, int embeddingDimension, int maxSequenceCount) {
		long tempBatchHandle = 0;
		SafeBatchProcessor tempSafeFallback = null;
		boolean tempUsingSafeFallback = true;

		if (USE_SAFE_FALLBACK) {
			// Use safe Java-level implementation
			tempSafeFallback = new SafeBatchProcessor(maxTokenCount, embeddingDimension, maxSequenceCount);
			LOGGER.info("Using SafeBatchProcessor fallback implementation");
		} else {
			// Try native implementation
			try {
				tempBatchHandle = initializeBatchNative(maxTokenCount, embeddingDimension, maxSequenceCount);
				if (tempBatchHandle == 0) {
					throw new LlamaException("Failed to initialize native batch processor");
				}
				tempUsingSafeFallback = false;
				LOGGER.info("Using native BatchProcessor implementation");
			} catch (Exception e) {
				LOGGER.warning("Native batch processing failed, falling back to safe implementation: " + e.getMessage());
				tempSafeFallback = new SafeBatchProcessor(maxTokenCount, embeddingDimension, maxSequenceCount);
			}
		}

		this.batchHandle = tempBatchHandle;
		this.safeFallback = tempSafeFallback;
		this.usingSafeFallback = tempUsingSafeFallback;
	}

	public int encodeContext(LlamaModel model) {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.encodeContext(model);
		}
		return encodeContextNative(model, batchHandle);
	}

	public int decodeTokens(LlamaModel model) {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.decodeTokens(model);
		}
		return decodeTokensNative(model, batchHandle);
	}

	public void setTokens(int[] tokens) {
		checkClosed();
		if (usingSafeFallback) {
			safeFallback.setTokens(tokens);
			return;
		}
		setBatchTokensNative(batchHandle, tokens);
	}

	public void setEmbeddings(float[] embeddings) {
		checkClosed();
		if (usingSafeFallback) {
			safeFallback.setEmbeddings(embeddings);
			return;
		}
		setBatchEmbeddingsNative(batchHandle, embeddings);
	}

	public void setPositions(int[] positions) {
		checkClosed();
		if (usingSafeFallback) {
			safeFallback.setPositions(positions);
			return;
		}
		setBatchPositionsNative(batchHandle, positions);
	}

	public void setSequenceIds(int[] sequenceIds) {
		checkClosed();
		if (usingSafeFallback) {
			safeFallback.setSequenceIds(sequenceIds);
			return;
		}
		setBatchSequenceIdsNative(batchHandle, sequenceIds);
	}

	public void setLogitFlags(byte[] logitFlags) {
		checkClosed();
		if (usingSafeFallback) {
			safeFallback.setLogitFlags(logitFlags);
			return;
		}
		setBatchLogitFlagsNative(batchHandle, logitFlags);
	}

	public int[] getTokens() {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.getTokens();
		}
		return getBatchTokensNative(batchHandle);
	}

	public float[] getEmbeddings() {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.getEmbeddings();
		}
		return getBatchEmbeddingsNative(batchHandle);
	}

	public int[] getPositions() {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.getPositions();
		}
		return getBatchPositionsNative(batchHandle);
	}

	public int[] getSequenceIds() {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.getSequenceIds();
		}
		return getBatchSequenceIdsNative(batchHandle);
	}

	public byte[] getLogitFlags() {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.getLogitFlags();
		}
		return getBatchLogitFlagsNative(batchHandle);
	}

	public int getTokenCount() {
		checkClosed();
		if (usingSafeFallback) {
			return safeFallback.getTokenCount();
		}
		return getBatchTokenCountNative(batchHandle);
	}

	private void checkClosed() {
		if (closed) {
			throw new IllegalStateException("BatchProcessor has been closed");
		}
	}

	@Override
	public void close() {
		if (!closed) {
			if (usingSafeFallback) {
				safeFallback.close();
			} else {
				freeBatchNative(batchHandle);
			}
			closed = true;
		}
	}

	private static native long initializeBatchNative(int maxTokenCount, int embeddingDimension, int maxSequenceCount);
	private static native void freeBatchNative(long batchHandle);
	private static native int encodeContextNative(LlamaModel model, long batchHandle);
	private static native int decodeTokensNative(LlamaModel model, long batchHandle);
	private static native void setBatchTokensNative(long batchHandle, int[] tokens);
	private static native void setBatchEmbeddingsNative(long batchHandle, float[] embeddings);
	private static native void setBatchPositionsNative(long batchHandle, int[] positions);
	private static native void setBatchSequenceIdsNative(long batchHandle, int[] sequenceIds);
	private static native void setBatchLogitFlagsNative(long batchHandle, byte[] logitFlags);
	private static native int[] getBatchTokensNative(long batchHandle);
	private static native float[] getBatchEmbeddingsNative(long batchHandle);
	private static native int[] getBatchPositionsNative(long batchHandle);
	private static native int[] getBatchSequenceIdsNative(long batchHandle);
	private static native byte[] getBatchLogitFlagsNative(long batchHandle);
	private static native int getBatchTokenCountNative(long batchHandle);
}