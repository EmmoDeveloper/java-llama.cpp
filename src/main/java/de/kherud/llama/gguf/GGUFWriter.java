package de.kherud.llama.gguf;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Java port of llama.cpp GGUF writer.
 * Writes GGUF format files with metadata and tensor information.
 *
 * Port of llama.cpp/gguf-py/gguf/gguf_writer.py
 */
public class GGUFWriter implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(GGUFWriter.class.getName());

	public enum WriterState {
		NO_FILE,
		EMPTY,
		HEADER,
		KV_DATA,
		TI_DATA,
		WEIGHTS
	}

	private BufferedOutputStream fout;
	private final Path path;
	private final String arch;
	private final GGUFConstants.GGUFEndian endianess;
	private final int dataAlignment;
	private final Map<String, TensorInfo> tensors;
	private final Map<String, GGUFValue> kvData;
	private WriterState state;
	private final boolean dryRun;
	private long currentPosition;

	public GGUFWriter(Path path, String arch) {
		this(path, arch, GGUFConstants.GGUFEndian.LITTLE, false);
	}

	public GGUFWriter(Path path, String arch, GGUFConstants.GGUFEndian endianess, boolean dryRun) {
		this.path = path;
		this.arch = arch;
		this.endianess = endianess;
		this.dataAlignment = GGUFConstants.DEFAULT_ALIGNMENT.value;
		this.tensors = new LinkedHashMap<>();
		this.kvData = new LinkedHashMap<>();
		this.dryRun = dryRun;
		this.state = WriterState.NO_FILE;
		this.currentPosition = 0;

		LOGGER.log(System.Logger.Level.INFO,"GGUF: This GGUF file is for " +
			(endianess == GGUFConstants.GGUFEndian.BIG ? "Big" : "Little") + " Endian only");

		// Add required architecture metadata
		addArchitecture();
	}

	/**
	 * Pack value according to GGUF format with proper endianness
	 */
	private byte[] pack(String format, Object value) {
		ByteBuffer buffer;

		switch (format) {
			case "B": // unsigned byte
				buffer = ByteBuffer.allocate(1);
				buffer.put(((Number) value).byteValue());
				return buffer.array();
			case "b": // signed byte
				buffer = ByteBuffer.allocate(1);
				buffer.put(((Number) value).byteValue());
				return buffer.array();
			case "H": // unsigned short
			case "h": // signed short
				buffer = ByteBuffer.allocate(2);
				buffer.order(endianess == GGUFConstants.GGUFEndian.LITTLE ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
				buffer.putShort(((Number) value).shortValue());
				return buffer.array();
			case "I": // unsigned int
			case "i": // signed int
				buffer = ByteBuffer.allocate(4);
				buffer.order(endianess == GGUFConstants.GGUFEndian.LITTLE ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
				buffer.putInt(((Number) value).intValue());
				return buffer.array();
			case "Q": // unsigned long long
			case "q": // signed long long
				buffer = ByteBuffer.allocate(8);
				buffer.order(endianess == GGUFConstants.GGUFEndian.LITTLE ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
				buffer.putLong(((Number) value).longValue());
				return buffer.array();
			case "f": // float
				buffer = ByteBuffer.allocate(4);
				buffer.order(endianess == GGUFConstants.GGUFEndian.LITTLE ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
				buffer.putFloat(((Number) value).floatValue());
				return buffer.array();
			case "d": // double
				buffer = ByteBuffer.allocate(8);
				buffer.order(endianess == GGUFConstants.GGUFEndian.LITTLE ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
				buffer.putDouble(((Number) value).doubleValue());
				return buffer.array();
			case "?": // boolean
				buffer = ByteBuffer.allocate(1);
				buffer.put((byte) (((Boolean) value) ? 1 : 0));
				return buffer.array();
			default:
				throw new IllegalArgumentException("Unsupported pack format: " + format);
		}
	}

	/**
	 * Pack a value with its type information
	 */
	private byte[] packValue(Object value, GGUFConstants.GGUFValueType type, boolean addType, GGUFConstants.GGUFValueType subType) throws IOException {
		ByteArrayOutputStream out = new ByteArrayOutputStream();

		if (addType) {
			out.write(pack("I", type.getValue()));
		}

		switch (type) {
			case UINT8:
				out.write(pack("B", value));
				break;
			case INT8:
				out.write(pack("b", value));
				break;
			case UINT16:
				out.write(pack("H", value));
				break;
			case INT16:
				out.write(pack("h", value));
				break;
			case UINT32:
				out.write(pack("I", value));
				break;
			case INT32:
				out.write(pack("i", value));
				break;
			case FLOAT32:
				out.write(pack("f", value));
				break;
			case UINT64:
				out.write(pack("Q", value));
				break;
			case INT64:
				out.write(pack("q", value));
				break;
			case FLOAT64:
				out.write(pack("d", value));
				break;
			case BOOL:
				out.write(pack("?", value));
				break;
			case STRING:
				byte[] stringBytes = ((String) value).getBytes(StandardCharsets.UTF_8);
				out.write(pack("Q", stringBytes.length));
				out.write(stringBytes);
				break;
			case ARRAY:
				@SuppressWarnings("unchecked")
				List<Object> array = (List<Object>) value;

				// Determine element type
				GGUFConstants.GGUFValueType elementType;
				if (subType != null) {
					elementType = subType;
				} else if (!array.isEmpty()) {
					elementType = GGUFValue.autoType(array.get(0)).getType();
				} else {
					elementType = GGUFConstants.GGUFValueType.STRING; // Default for empty arrays
				}

				out.write(pack("I", elementType.getValue()));
				out.write(pack("Q", array.size()));

				for (Object item : array) {
					out.write(packValue(item, elementType, false, null));
				}
				break;
			default:
				throw new IllegalArgumentException("Unsupported GGUF value type: " + type);
		}

		return out.toByteArray();
	}

	/**
	 * Calculate aligned padding size (ggml_pad equivalent)
	 */
	public static long ggmlPad(long size, int alignment) {
		return ((size + alignment - 1) / alignment) * alignment;
	}

	public void openOutputFile() throws IOException {
		if (state != WriterState.NO_FILE) {
			throw new IllegalStateException("Output file already open, got state: " + state);
		}

		if (!dryRun) {
			// Create parent directories if needed
			if (path.getParent() != null) {
				path.getParent().toFile().mkdirs();
			}

			fout = new BufferedOutputStream(new FileOutputStream(path.toFile()));
		}
		state = WriterState.EMPTY;
	}

	public void writeHeaderToFile() throws IOException {
		if (state != WriterState.EMPTY) {
			throw new IllegalStateException("Expected output file to be empty, got: " + state);
		}

		if (!dryRun) {
			// Write GGUF magic
			fout.write(pack("I", GGUFConstants.MAGIC.value));

			// Write version
			fout.write(pack("I", GGUFConstants.VERSION.value));

			// Write tensor count
			fout.write(pack("Q", (long) tensors.size()));

			// Write metadata count
			fout.write(pack("Q", (long) kvData.size()));

			fout.flush();
		}

		state = WriterState.HEADER;
	}

	public void writeKvDataToFile() throws IOException {
		if (state != WriterState.HEADER) {
			throw new IllegalStateException("Expected output file to contain header, got: " + state);
		}

		if (!dryRun) {
			for (Map.Entry<String, GGUFValue> entry : kvData.entrySet()) {
				String key = entry.getKey();
				GGUFValue val = entry.getValue();

				// Write key
				fout.write(packValue(key, GGUFConstants.GGUFValueType.STRING, false, null));

				// Write value with type
				fout.write(packValue(val.getValue(), val.getType(), true, val.getSubType()));
			}
			fout.flush();
		}

		state = WriterState.KV_DATA;
	}

	public void writeTensorInfoToFile() throws IOException {
		if (state != WriterState.KV_DATA) {
			throw new IllegalStateException("Expected output file to contain KV data, got: " + state);
		}

		if (!dryRun) {
			long offsetTensor = 0;

			for (Map.Entry<String, TensorInfo> entry : tensors.entrySet()) {
				String name = entry.getKey();
				TensorInfo ti = entry.getValue();

				// Write tensor name
				fout.write(packValue(name, GGUFConstants.GGUFValueType.STRING, false, null));

				// Write number of dimensions
				fout.write(pack("I", ti.shape().length));

				// Write dimensions in REVERSE order (critical!)
				long[] shape = ti.shape();
				for (int j = shape.length - 1; j >= 0; j--) {
					fout.write(pack("Q", shape[j]));
				}

				// Write data type
				fout.write(pack("I", ti.dtype().getValue()));

				// Write offset
				fout.write(pack("Q", offsetTensor));

				// Update offset for next tensor
				offsetTensor += ggmlPad(ti.nbytes(), dataAlignment);
			}

			fout.flush();
		}

		state = WriterState.TI_DATA;
	}

	public void writePadding(long currentPos) throws IOException {
		if (dryRun) return;

		long alignedPos = ggmlPad(currentPos, dataAlignment);
		long padding = alignedPos - currentPos;

		for (long i = 0; i < padding; i++) {
			fout.write(0);
		}
	}

	// Metadata methods
	public void addKeyValue(String key, Object value, GGUFConstants.GGUFValueType type) {
		addKeyValue(key, value, type, null);
	}

	public void addKeyValue(String key, Object value, GGUFConstants.GGUFValueType type, GGUFConstants.GGUFValueType subType) {
		if (kvData.containsKey(key)) {
			throw new IllegalArgumentException("Key already exists: " + key);
		}
		kvData.put(key, new GGUFValue(value, type, subType));
	}

	public void addString(String key, String value) {
		addKeyValue(key, value, GGUFConstants.GGUFValueType.STRING);
	}

	public void addFloat32(String key, float value) {
		addKeyValue(key, value, GGUFConstants.GGUFValueType.FLOAT32);
	}

	public void addInt32(String key, int value) {
		addKeyValue(key, value, GGUFConstants.GGUFValueType.INT32);
	}

	public void addUInt32(String key, int value) {
		addKeyValue(key, value, GGUFConstants.GGUFValueType.UINT32);
	}

	public void addBool(String key, boolean value) {
		addKeyValue(key, value, GGUFConstants.GGUFValueType.BOOL);
	}

	public void addArray(String key, List<?> value) {
		GGUFValue ggufValue = GGUFValue.autoType(value);
		kvData.put(key, ggufValue);
	}

	// Tensor methods
	public void addTensorInfo(String name, long[] shape, GGUFConstants.GGMLQuantizationType dtype, long nbytes) {
		if (tensors.containsKey(name)) {
			throw new IllegalArgumentException("Tensor already exists: " + name);
		}
		tensors.put(name, new TensorInfo(shape, dtype, nbytes));
	}

	// Standard metadata methods
	public void addType(String typeName) {
		addString(GGUFConstants.Keys.General.TYPE, typeName);
	}

	public void addArchitecture() {
		addString(GGUFConstants.Keys.General.ARCHITECTURE, arch);
	}

	public void addQuantizationVersion(int version) {
		addUInt32(GGUFConstants.Keys.General.QUANTIZATION_VERSION, version);
	}

	public void addName(String name) {
		addString(GGUFConstants.Keys.General.NAME, name);
	}

	public void addAuthor(String author) {
		addString(GGUFConstants.Keys.General.AUTHOR, author);
	}

	public void addDescription(String description) {
		addString(GGUFConstants.Keys.General.DESCRIPTION, description);
	}

	// Adapter-specific methods
	public void addLoRAAlpha(float alpha) {
		addFloat32(GGUFConstants.Keys.Adapter.LORA_ALPHA, alpha);
	}

	// Write complete file
	public void writeToFile() throws IOException {
		openOutputFile();
		writeHeaderToFile();
		writeKvDataToFile();
		writeTensorInfoToFile();

		if (!dryRun) {
			// Calculate exact current position for padding
			long currentPos = 0;

			// Header size
			currentPos += 4 + 4 + 8 + 8; // magic + version + tensor_count + kv_count

			// KV data size (exact calculation)
			for (Map.Entry<String, GGUFValue> entry : kvData.entrySet()) {
				try {
					// Calculate exact size of key
					byte[] keyData = packValue(entry.getKey(), GGUFConstants.GGUFValueType.STRING, false, null);
					currentPos += keyData.length;

					// Calculate exact size of value with type
					GGUFValue val = entry.getValue();
					byte[] valueData = packValue(val.getValue(), val.getType(), true, val.getSubType());
					currentPos += valueData.length;
				} catch (IOException e) {
					throw new RuntimeException("Failed to calculate KV data size", e);
				}
			}

			// Tensor info size (exact calculation)
			for (Map.Entry<String, TensorInfo> entry : tensors.entrySet()) {
				try {
					String name = entry.getKey();
					TensorInfo ti = entry.getValue();

					// Calculate exact size of tensor name
					byte[] nameData = packValue(name, GGUFConstants.GGUFValueType.STRING, false, null);
					currentPos += nameData.length;

					// Add sizes for tensor info fields
					currentPos += 4; // ndims
					currentPos += ti.shape().length * 8L; // dimensions
					currentPos += 4; // dtype
					currentPos += 8; // offset
				} catch (IOException e) {
					throw new RuntimeException("Failed to calculate tensor info size", e);
				}
			}

			// Write padding to align tensor data
			writePadding(currentPos);
		}

		state = WriterState.WEIGHTS;
	}

	public void writeTensorData(float[][] matrix) throws IOException {
		if (state != WriterState.WEIGHTS) {
			throw new IllegalStateException("Expected to be in WEIGHTS state for tensor data");
		}

		if (!dryRun) {
			// Write matrix data in row-major order with proper endianness
			for (float[] row : matrix) {
				for (float value : row) {
					fout.write(pack("f", value));
				}
			}
			fout.flush();
		}
	}

	public void flush() throws IOException {
		if (fout != null && !dryRun) {
			fout.flush();
		}
	}

	@Override
	public void close() throws IOException {
		if (fout != null && !dryRun) {
			fout.close();
		}
		state = WriterState.NO_FILE;
	}

	// Getters
	public WriterState getState() {
		return state;
	}

	public Map<String, TensorInfo> getTensors() {
		return Collections.unmodifiableMap(tensors);
	}

	public Map<String, GGUFValue> getKvData() {
		return Collections.unmodifiableMap(kvData);
	}
}
