package de.kherud.llama.gguf;

/**
 * Information about a tensor stored in a GGUF file.
 * Port of llama.cpp/gguf-py/gguf/gguf_writer.py TensorInfo
 */
public record TensorInfo(long[] shape, GGUFConstants.GGMLQuantizationType dtype, long nbytes) {
	public TensorInfo(long[] shape, GGUFConstants.GGMLQuantizationType dtype, long nbytes) {
		this.shape = shape.clone();
		this.dtype = dtype;
		this.nbytes = nbytes;
	}

	@Override
	public long[] shape() {
		return shape.clone();
	}

	/**
	 * Calculate total number of elements in the tensor
	 */
	public long getElementCount() {
		long count = 1;
		for (long dim : shape) {
			count *= dim;
		}
		return count;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("TensorInfo{shape=[");
		for (int i = 0; i < shape.length; i++) {
			if (i > 0) sb.append(", ");
			sb.append(shape[i]);
		}
		sb.append("], dtype=").append(dtype).append(", nbytes=").append(nbytes).append("}");
		return sb.toString();
	}
}
