package de.kherud.llama.gguf;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class GGUFReader implements AutoCloseable {

	private final RandomAccessFile file;
	private final ByteOrder byteOrder;
	private final int alignment;
	private final long dataOffset;
	private final Map<String, GGUFField> fields;
	private final List<GGUFTensor> tensors;

	public GGUFReader(Path path) throws IOException {
		this.file = new RandomAccessFile(path.toFile(), "r");
		this.fields = new LinkedHashMap<>();
		this.tensors = new ArrayList<>();

		try {
			long offset = 0;

			// Check GGUF magic
			int magic = readUInt32(offset, ByteOrder.LITTLE_ENDIAN);
			if (magic != GGUFConstants.GGUF_MAGIC) {
				throw new IOException("Invalid GGUF magic number: 0x" + Integer.toHexString(magic));
			}
			offset += 4;

			// Check version and endianness
			int version = readUInt32(offset, ByteOrder.LITTLE_ENDIAN);
			if ((version & 0xFFFF) == 0) {
				// Opposite byte order
				this.byteOrder = ByteOrder.BIG_ENDIAN;
				version = Integer.reverseBytes(version);
			} else {
				this.byteOrder = ByteOrder.LITTLE_ENDIAN;
			}

			if (version != GGUFConstants.GGUF_VERSION) {
				throw new IOException("Unsupported GGUF version: " + version);
			}
			offset += 4;

			// Read tensor count and metadata count
			long tensorCount = readUInt64(offset);
			offset += 8;
			long metadataCount = readUInt64(offset);
			offset += 8;

			// Read metadata
			offset = readMetadata(offset, metadataCount);

			// Read tensor info
			offset = readTensorInfo(offset, tensorCount);

			// Calculate alignment
			GGUFField alignmentField = fields.get("general.alignment");
			if (alignmentField != null && alignmentField.type == GGUFConstants.GGUFValueType.UINT32) {
				this.alignment = (Integer) alignmentField.value;
			} else {
				this.alignment = GGUFConstants.GGUF_DEFAULT_ALIGNMENT;
			}

			// Calculate data offset with alignment
			long padding = offset % alignment;
			if (padding != 0) {
				offset += alignment - padding;
			}
			this.dataOffset = offset;

			// Read tensor data
			readTensorData();

		} catch (Exception e) {
			file.close();
			throw e;
		}
	}

	private long readMetadata(long offset, long count) throws IOException {
		for (long i = 0; i < count; i++) {
			// Read key
			long keyLength = readUInt64(offset);
			offset += 8;
			String key = readString(offset, keyLength);
			offset += keyLength;

			// Read value type
			int valueType = readUInt32(offset);
			offset += 4;

			// Read value
			GGUFField field = readValue(offset, GGUFConstants.GGUFValueType.fromValue(valueType));
			field.name = key;
			fields.put(key, field);
			offset += field.size;
		}
		return offset;
	}

	private long readTensorInfo(long offset, long count) throws IOException {
		for (long i = 0; i < count; i++) {
			// Read tensor name
			long nameLength = readUInt64(offset);
			offset += 8;
			String name = readString(offset, nameLength);
			offset += nameLength;

			// Read number of dimensions
			int nDims = readUInt32(offset);
			offset += 4;

			// Read dimensions
			long[] shape = new long[nDims];
			for (int j = 0; j < nDims; j++) {
				shape[j] = readUInt64(offset);
				offset += 8;
			}

			// Read data type
			int dataType = readUInt32(offset);
			offset += 4;

			// Read tensor offset
			long tensorOffset = readUInt64(offset);
			offset += 8;

			// Calculate tensor size
			long nElements = 1;
			for (long dim : shape) {
				nElements *= dim;
			}

			GGUFTensor tensor = new GGUFTensor();
			tensor.name = name;
			tensor.shape = shape;
			tensor.type = GGUFConstants.GGMLQuantizationType.fromValue(dataType);
			tensor.nElements = nElements;
			tensor.dataOffset = dataOffset + tensorOffset;
			tensor.calculateSize();

			tensors.add(tensor);
		}
		return offset;
	}

	private void readTensorData() throws IOException {
		for (GGUFTensor tensor : tensors) {
			tensor.data = readTensorBytes(tensor.dataOffset, tensor.nBytes);
		}
	}

	private GGUFField readValue(long offset, GGUFConstants.GGUFValueType type) throws IOException {
		GGUFField field = new GGUFField();
		field.type = type;

		switch (type) {
			case UINT8:
				field.value = readUInt8(offset);
				field.size = 1;
				break;
			case INT8:
				field.value = readInt8(offset);
				field.size = 1;
				break;
			case UINT16:
				field.value = readUInt16(offset);
				field.size = 2;
				break;
			case INT16:
				field.value = readInt16(offset);
				field.size = 2;
				break;
			case UINT32:
				field.value = readUInt32(offset);
				field.size = 4;
				break;
			case INT32:
				field.value = readInt32(offset);
				field.size = 4;
				break;
			case UINT64:
				field.value = readUInt64(offset);
				field.size = 8;
				break;
			case INT64:
				field.value = readInt64(offset);
				field.size = 8;
				break;
			case FLOAT32:
				field.value = readFloat32(offset);
				field.size = 4;
				break;
			case FLOAT64:
				field.value = readFloat64(offset);
				field.size = 8;
				break;
			case BOOL:
				field.value = readUInt8(offset) != 0;
				field.size = 1;
				break;
			case STRING:
				long stringLength = readUInt64(offset);
				field.value = readString(offset + 8, stringLength);
				field.size = 8 + stringLength;
				break;
			case ARRAY:
				return readArray(offset);
			default:
				throw new IOException("Unsupported value type: " + type);
		}

		return field;
	}

	private GGUFField readArray(long offset) throws IOException {
		GGUFField field = new GGUFField();
		field.type = GGUFConstants.GGUFValueType.ARRAY;

		// Read array element type
		int elementType = readUInt32(offset);
		offset += 4;

		// Read array length
		long length = readUInt64(offset);
		offset += 8;

		GGUFConstants.GGUFValueType elemType = GGUFConstants.GGUFValueType.fromValue(elementType);
		List<Object> arrayValues = new ArrayList<>();

		long totalSize = 12; // 4 bytes for type + 8 bytes for length

		for (long i = 0; i < length; i++) {
			GGUFField element = readValue(offset, elemType);
			arrayValues.add(element.value);
			offset += element.size;
			totalSize += element.size;
		}

		field.value = arrayValues;
		field.size = totalSize;
		return field;
	}

	private byte readUInt8(long offset) throws IOException {
		file.seek(offset);
		return file.readByte();
	}

	private byte readInt8(long offset) throws IOException {
		file.seek(offset);
		return file.readByte();
	}

	private int readUInt16(long offset) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[2];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(byteOrder).getShort() & 0xFFFF;
	}

	private short readInt16(long offset) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[2];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(byteOrder).getShort();
	}

	private int readUInt32(long offset) throws IOException {
		return readUInt32(offset, byteOrder);
	}

	private int readUInt32(long offset, ByteOrder order) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[4];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(order).getInt();
	}

	private int readInt32(long offset) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[4];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(byteOrder).getInt();
	}

	private long readUInt64(long offset) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[8];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(byteOrder).getLong();
	}

	private long readInt64(long offset) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[8];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(byteOrder).getLong();
	}

	private float readFloat32(long offset) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[4];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(byteOrder).getFloat();
	}

	private double readFloat64(long offset) throws IOException {
		file.seek(offset);
		byte[] bytes = new byte[8];
		file.readFully(bytes);
		return ByteBuffer.wrap(bytes).order(byteOrder).getDouble();
	}

	private String readString(long offset, long length) throws IOException {
		if (length > Integer.MAX_VALUE) {
			throw new IOException("String too large: " + length + " bytes (max: " + Integer.MAX_VALUE + ")");
		}
		if (length < 0) {
			throw new IOException("Invalid string length: " + length);
		}
		file.seek(offset);
		byte[] bytes = new byte[(int) length];
		file.readFully(bytes);
		return new String(bytes, "UTF-8");
	}

	private byte[] readTensorBytes(long offset, long length) throws IOException {
		if (length > Integer.MAX_VALUE) {
			throw new IOException("Tensor too large: " + length + " bytes (max: " + Integer.MAX_VALUE + ")");
		}
		if (length < 0) {
			throw new IOException("Invalid tensor length: " + length);
		}
		file.seek(offset);
		byte[] bytes = new byte[(int) length];
		file.readFully(bytes);
		return bytes;
	}

	public GGUFField getField(String key) {
		return fields.get(key);
	}

	public GGUFTensor getTensor(int index) {
		return tensors.get(index);
	}

	public GGUFTensor getTensor(String name) {
		return tensors.stream()
			.filter(t -> t.name.equals(name))
			.findFirst()
			.orElse(null);
	}

	public Map<String, GGUFField> getFields() {
		return new LinkedHashMap<>(fields);
	}

	public List<GGUFTensor> getTensors() {
		return new ArrayList<>(tensors);
	}

	public int getTensorCount() {
		return tensors.size();
	}

	public int getFieldCount() {
		return fields.size();
	}

	public ByteOrder getByteOrder() {
		return byteOrder;
	}

	public int getAlignment() {
		return alignment;
	}

	public long getDataOffset() {
		return dataOffset;
	}

	@Override
	public void close() throws IOException {
		if (file != null) {
			file.close();
		}
	}

	public static class GGUFField {
		public String name;
		public GGUFConstants.GGUFValueType type;
		public Object value;
		public long size;

		@Override
		public String toString() {
			return String.format("GGUFField{name='%s', type=%s, value=%s}", name, type, value);
		}
	}

	public static class GGUFTensor {
		public String name;
		public long[] shape;
		public GGUFConstants.GGMLQuantizationType type;
		public long nElements;
		public long nBytes;
		public long dataOffset;
		public byte[] data;

		public void calculateSize() {
			Map<GGUFConstants.GGMLQuantizationType, Integer> quantSizes = Map.of(
				GGUFConstants.GGMLQuantizationType.F32, 4,
				GGUFConstants.GGMLQuantizationType.F16, 2,
				GGUFConstants.GGMLQuantizationType.Q4_0, 1,
				GGUFConstants.GGMLQuantizationType.Q4_1, 1,
				GGUFConstants.GGMLQuantizationType.Q5_0, 1,
				GGUFConstants.GGMLQuantizationType.Q5_1, 1,
				GGUFConstants.GGMLQuantizationType.Q8_0, 1,
				GGUFConstants.GGMLQuantizationType.Q8_1, 1
			);

			int typeSize = quantSizes.getOrDefault(type, 1);
			if (type == GGUFConstants.GGMLQuantizationType.F32 || type == GGUFConstants.GGMLQuantizationType.F16) {
				nBytes = nElements * typeSize;
			} else {
				// For quantized types, size calculation is more complex
				int blockSize = 32; // Most quantized types use 32-element blocks
				nBytes = (nElements + blockSize - 1) / blockSize * getBlockBytes(type);
			}
		}

		private int getBlockBytes(GGUFConstants.GGMLQuantizationType type) {
			switch (type) {
				case Q4_0: return 18; // 16 4-bit values + 2 bytes metadata
				case Q4_1: return 20; // 16 4-bit values + 4 bytes metadata
				case Q5_0: return 22; // 16 5-bit values + 6 bytes metadata
				case Q5_1: return 24; // 16 5-bit values + 8 bytes metadata
				case Q8_0: return 34; // 32 8-bit values + 2 bytes metadata
				case Q8_1: return 36; // 32 8-bit values + 4 bytes metadata
				default: return 1;
			}
		}

		@Override
		public String toString() {
			StringBuilder shapeStr = new StringBuilder("[");
			for (int i = 0; i < shape.length; i++) {
				if (i > 0) shapeStr.append(", ");
				shapeStr.append(shape[i]);
			}
			shapeStr.append("]");

			return String.format("GGUFTensor{name='%s', shape=%s, type=%s, elements=%d, bytes=%d}",
				name, shapeStr, type, nElements, nBytes);
		}
	}
}
