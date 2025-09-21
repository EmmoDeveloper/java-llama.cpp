package de.kherud.llama.gguf;

import java.util.List;

/**
 * Typed value container for GGUF metadata.
 * Port of llama.cpp/gguf-py/gguf/gguf_writer.py GGUFValue
 */
public class GGUFValue {
	private final Object value;
	private final GGUFConstants.GGUFValueType type;
	private final GGUFConstants.GGUFValueType subType;

	public GGUFValue(Object value, GGUFConstants.GGUFValueType type) {
		this(value, type, null);
	}

	public GGUFValue(Object value, GGUFConstants.GGUFValueType type, GGUFConstants.GGUFValueType subType) {
		this.value = value;
		this.type = type;
		this.subType = subType;
	}

	public Object getValue() {
		return value;
	}

	public GGUFConstants.GGUFValueType getType() {
		return type;
	}

	public GGUFConstants.GGUFValueType getSubType() {
		return subType;
	}

	/**
	 * Automatically determine the appropriate GGUF type for a value
	 */
	public static GGUFValue autoType(Object value) {
		if (value instanceof String) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.STRING);
		} else if (value instanceof Boolean) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.BOOL);
		} else if (value instanceof Byte) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.INT8);
		} else if (value instanceof Short) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.INT16);
		} else if (value instanceof Integer) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.INT32);
		} else if (value instanceof Long) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.INT64);
		} else if (value instanceof Float) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.FLOAT32);
		} else if (value instanceof Double) {
			return new GGUFValue(value, GGUFConstants.GGUFValueType.FLOAT64);
		} else if (value instanceof List) {
			List<?> list = (List<?>) value;
			if (list.isEmpty()) {
				return new GGUFValue(value, GGUFConstants.GGUFValueType.ARRAY, GGUFConstants.GGUFValueType.STRING);
			} else {
				// Determine sub-type from first element
				Object first = list.get(0);
				GGUFConstants.GGUFValueType subType = autoType(first).getType();
				return new GGUFValue(value, GGUFConstants.GGUFValueType.ARRAY, subType);
			}
		} else {
			throw new IllegalArgumentException("Unsupported value type: " + value.getClass());
		}
	}

	@Override
	public String toString() {
		return String.format("GGUFValue{value=%s, type=%s, subType=%s}", value, type, subType);
	}
}