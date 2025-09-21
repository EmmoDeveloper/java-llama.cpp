package de.kherud.llama.gguf;

/**
 * GGUF format constants and metadata keys.
 * Port of llama.cpp/gguf-py/gguf/constants.py
 */
public final class GGUFConstants {

	// Core GGUF constants
	public static final int GGUF_MAGIC = 0x46554747;  // "GGUF"
	public static final int GGUF_VERSION = 3;
	public static final int GGUF_DEFAULT_ALIGNMENT = 32;
	public static final int GGML_QUANT_VERSION = 2;

	// Prevent instantiation
	private GGUFConstants() {}

	/**
	 * GGUF value types for metadata and tensor info
	 */
	public enum GGUFValueType {
		UINT8(0),
		INT8(1),
		UINT16(2),
		INT16(3),
		UINT32(4),
		INT32(5),
		FLOAT32(6),
		BOOL(7),
		STRING(8),
		ARRAY(9),
		UINT64(10),
		INT64(11),
		FLOAT64(12);

		private final int value;

		GGUFValueType(int value) {
			this.value = value;
		}

		public int getValue() {
			return value;
		}

		public static GGUFValueType fromValue(int value) {
			for (GGUFValueType type : values()) {
				if (type.value == value) {
					return type;
				}
			}
			throw new IllegalArgumentException("Unknown GGUFValueType: " + value);
		}
	}

	/**
	 * GGML quantization types for tensors
	 */
	public enum GGMLQuantizationType {
		F32(0),
		F16(1),
		Q4_0(2),
		Q4_1(3),
		Q5_0(6),
		Q5_1(7),
		Q8_0(8),
		Q8_1(9),
		Q2_K(10),
		Q3_K(11),
		Q4_K(12),
		Q5_K(13),
		Q6_K(14),
		Q8_K(15),
		IQ2_XXS(16),
		IQ2_XS(17),
		IQ3_XXS(18),
		IQ1_S(19),
		IQ4_NL(20),
		IQ3_S(21),
		IQ2_S(22),
		IQ4_XS(23),
		IQ1_M(24),
		BF16(25),
		IQ4_NM(26),
		IQ3_M(27),
		IQ2_M(28),
		TQ1_0(29),
		TQ2_0(30);

		private final int value;

		GGMLQuantizationType(int value) {
			this.value = value;
		}

		public int getValue() {
			return value;
		}

		public static GGMLQuantizationType fromValue(int value) {
			for (GGMLQuantizationType type : values()) {
				if (type.value == value) {
					return type;
				}
			}
			throw new IllegalArgumentException("Unknown GGMLQuantizationType: " + value);
		}
	}

	/**
	 * Endianness for GGUF files
	 */
	public enum GGUFEndian {
		LITTLE(0),
		BIG(1);

		private final int value;

		GGUFEndian(int value) {
			this.value = value;
		}

		public int getValue() {
			return value;
		}
	}

	/**
	 * Standard metadata keys for GGUF files
	 */
	public static final class Keys {

		public static final class General {
			public static final String TYPE = "general.type";
			public static final String ARCHITECTURE = "general.architecture";
			public static final String QUANTIZATION_VERSION = "general.quantization_version";
			public static final String ALIGNMENT = "general.alignment";
			public static final String FILE_TYPE = "general.file_type";
			public static final String NAME = "general.name";
			public static final String AUTHOR = "general.author";
			public static final String VERSION = "general.version";
			public static final String ORGANIZATION = "general.organization";
			public static final String FINETUNE = "general.finetune";
			public static final String BASENAME = "general.basename";
			public static final String DESCRIPTION = "general.description";
			public static final String QUANTIZED_BY = "general.quantized_by";
			public static final String SIZE_LABEL = "general.size_label";
			public static final String LICENSE = "general.license";
			public static final String LICENSE_NAME = "general.license.name";
			public static final String LICENSE_LINK = "general.license.link";
			public static final String URL = "general.url";
			public static final String DOI = "general.doi";
			public static final String UUID = "general.uuid";
			public static final String REPO_URL = "general.repo_url";
			public static final String SOURCE_URL = "general.source.url";
			public static final String SOURCE_DOI = "general.source.doi";
			public static final String SOURCE_UUID = "general.source.uuid";
			public static final String SOURCE_REPO_URL = "general.source.repo_url";
			public static final String TAGS = "general.tags";
			public static final String LANGUAGES = "general.languages";
		}

		public static final class Adapter {
			public static final String TYPE = "adapter.type";
			public static final String LORA_ALPHA = "adapter.lora.alpha";
		}

		public static final class LLM {
			public static final String VOCAB_SIZE = "{arch}.vocab_size";
			public static final String CONTEXT_LENGTH = "{arch}.context_length";
			public static final String EMBEDDING_LENGTH = "{arch}.embedding_length";
			public static final String BLOCK_COUNT = "{arch}.block_count";
			public static final String FEED_FORWARD_LENGTH = "{arch}.feed_forward_length";
			public static final String ATTENTION_HEAD_COUNT = "{arch}.attention.head_count";
			public static final String ATTENTION_HEAD_COUNT_KV = "{arch}.attention.head_count_kv";
			public static final String ATTENTION_LAYER_NORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon";
			public static final String ROPE_DIMENSION_COUNT = "{arch}.rope.dimension_count";
			public static final String ROPE_FREQ_BASE = "{arch}.rope.freq_base";
		}
	}
}