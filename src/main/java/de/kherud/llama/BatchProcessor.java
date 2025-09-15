package de.kherud.llama;

public class BatchProcessor implements AutoCloseable {
	static { LlamaLoader.initialize(); }

	private final long batchHandle;
	private boolean closed = false;

	public BatchProcessor(int maxTokenCount, int embeddingDimension, int maxSequenceCount) {
		this.batchHandle = initializeBatchNative(maxTokenCount, embeddingDimension, maxSequenceCount);
		if (this.batchHandle == 0) {
			throw new LlamaException("Failed to initialize batch processor");
		}
	}

	public int encodeContext(LlamaModel model) {
		checkClosed();
		return encodeContextNative(model, batchHandle);
	}

	public int decodeTokens(LlamaModel model) {
		checkClosed();
		return decodeTokensNative(model, batchHandle);
	}

	public void setTokens(int[] tokens) {
		checkClosed();
		setBatchTokensNative(batchHandle, tokens);
	}

	public void setEmbeddings(float[] embeddings) {
		checkClosed();
		setBatchEmbeddingsNative(batchHandle, embeddings);
	}

	public void setPositions(int[] positions) {
		checkClosed();
		setBatchPositionsNative(batchHandle, positions);
	}

	public void setSequenceIds(int[] sequenceIds) {
		checkClosed();
		setBatchSequenceIdsNative(batchHandle, sequenceIds);
	}

	public void setLogitFlags(byte[] logitFlags) {
		checkClosed();
		setBatchLogitFlagsNative(batchHandle, logitFlags);
	}

	public int[] getTokens() {
		checkClosed();
		return getBatchTokensNative(batchHandle);
	}

	public float[] getEmbeddings() {
		checkClosed();
		return getBatchEmbeddingsNative(batchHandle);
	}

	public int[] getPositions() {
		checkClosed();
		return getBatchPositionsNative(batchHandle);
	}

	public int[] getSequenceIds() {
		checkClosed();
		return getBatchSequenceIdsNative(batchHandle);
	}

	public byte[] getLogitFlags() {
		checkClosed();
		return getBatchLogitFlagsNative(batchHandle);
	}

	public int getTokenCount() {
		checkClosed();
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
			freeBatchNative(batchHandle);
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