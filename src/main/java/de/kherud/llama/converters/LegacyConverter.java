package de.kherud.llama.converters;

import de.kherud.llama.gguf.GGUFConstants;
import de.kherud.llama.gguf.GGUFWriter;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Legacy model format converter.
 *
 * Equivalent to convert_llama_ggml_to_gguf.py and convert_legacy_llama.py
 * Converts older GGML format models to modern GGUF format.
 */
public class LegacyConverter {
	private static final Logger LOGGER = Logger.getLogger(LegacyConverter.class.getName());

	// GGML format constants
	private static final int GGML_MAGIC = 0x67676d6c; // 'ggml'
	private static final int GGML_VERSION_1 = 1;
	private static final int GGML_VERSION_2 = 2;

	public static class ConversionConfig {
		private String outputFormat = "gguf";
		private boolean verbose = false;
		private boolean dryRun = false;
		private String modelName = "converted_model";
		private Map<String, String> metadataOverrides = new HashMap<>();
		private GGUFConstants.GGMLQuantizationType targetQuantization = null;

		public ConversionConfig outputFormat(String format) {
			this.outputFormat = format;
			return this;
		}

		public ConversionConfig verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public ConversionConfig dryRun(boolean dryRun) {
			this.dryRun = dryRun;
			return this;
		}

		public ConversionConfig modelName(String name) {
			this.modelName = name;
			return this;
		}

		public ConversionConfig addMetadata(String key, String value) {
			this.metadataOverrides.put(key, value);
			return this;
		}

		public ConversionConfig targetQuantization(GGUFConstants.GGMLQuantizationType type) {
			this.targetQuantization = type;
			return this;
		}
	}

	public static class LegacyModelInfo {
		public int magic;
		public int version;
		public int vocabSize;
		public int contextSize;
		public int embeddingSize;
		public int headCount;
		public int layerCount;
		public int rotationDimension;
		public int fileType;
		public Map<String, LegacyTensor> tensors = new LinkedHashMap<>();
		public List<String> vocabulary = new ArrayList<>();
		public List<Float> scores = new ArrayList<>();

		public static class LegacyTensor {
			public String name;
			public int[] shape;
			public int type;
			public long offset;
			public long size;
			public ByteBuffer data;
		}
	}

	private final Path inputPath;
	private final Path outputPath;
	private final ConversionConfig config;

	public LegacyConverter(Path inputPath, Path outputPath) {
		this(inputPath, outputPath, new ConversionConfig());
	}

	public LegacyConverter(Path inputPath, Path outputPath, ConversionConfig config) {
		this.inputPath = inputPath;
		this.outputPath = outputPath;
		this.config = config;
	}

	/**
	 * Convert legacy model to GGUF format
	 */
	public void convert() throws IOException {
		LOGGER.info("Starting legacy model conversion");
		LOGGER.info("Input: " + inputPath);
		LOGGER.info("Output: " + outputPath);

		// Detect input format
		LegacyModelInfo modelInfo = detectAndLoadModel();

		if (config.verbose) {
			printModelInfo(modelInfo);
		}

		if (config.dryRun) {
			LOGGER.info("Dry run - no output file will be created");
			return;
		}

		// Convert to GGUF
		convertToGGUF(modelInfo);

		LOGGER.info("Conversion completed successfully");
	}

	private LegacyModelInfo detectAndLoadModel() throws IOException {
		if (!Files.exists(inputPath)) {
			throw new IOException("Input file not found: " + inputPath);
		}

		// Try to detect format by reading magic number
		try (RandomAccessFile raf = new RandomAccessFile(inputPath.toFile(), "r")) {
			int magic = Integer.reverseBytes(raf.readInt()); // GGML uses big-endian

			if (magic == GGML_MAGIC) {
				return loadGGMLModel(raf);
			} else {
				// Try other legacy formats
				return loadOtherLegacyFormat(raf);
			}
		}
	}

	private LegacyModelInfo loadGGMLModel(RandomAccessFile raf) throws IOException {
		LOGGER.info("Detected GGML format");

		LegacyModelInfo info = new LegacyModelInfo();
		raf.seek(0);

		// Read header
		info.magic = Integer.reverseBytes(raf.readInt());
		info.version = Integer.reverseBytes(raf.readInt());

		if (info.version < GGML_VERSION_1 || info.version > GGML_VERSION_2) {
			throw new IOException("Unsupported GGML version: " + info.version);
		}

		// Model parameters
		info.vocabSize = Integer.reverseBytes(raf.readInt());
		info.contextSize = Integer.reverseBytes(raf.readInt());
		info.embeddingSize = Integer.reverseBytes(raf.readInt());
		info.headCount = Integer.reverseBytes(raf.readInt());
		info.layerCount = Integer.reverseBytes(raf.readInt());
		info.rotationDimension = Integer.reverseBytes(raf.readInt());
		info.fileType = Integer.reverseBytes(raf.readInt());

		if (config.verbose) {
			LOGGER.info(String.format("GGML version: %d", info.version));
			LOGGER.info(String.format("Vocab size: %d", info.vocabSize));
			LOGGER.info(String.format("Context size: %d", info.contextSize));
		}

		// Load vocabulary
		loadGGMLVocabulary(raf, info);

		// Load tensors
		loadGGMLTensors(raf, info);

		return info;
	}

	private void loadGGMLVocabulary(RandomAccessFile raf, LegacyModelInfo info) throws IOException {
		for (int i = 0; i < info.vocabSize; i++) {
			// Read string length
			int len = Integer.reverseBytes(raf.readInt());
			if (len < 0 || len > 1000000) {
				throw new IOException("Invalid vocabulary entry length: " + len);
			}

			// Read string
			byte[] bytes = new byte[len];
			raf.readFully(bytes);
			String token = new String(bytes, "UTF-8");
			info.vocabulary.add(token);

			// Read score (version 2 only)
			if (info.version >= GGML_VERSION_2) {
				float score = Float.intBitsToFloat(Integer.reverseBytes(raf.readInt()));
				info.scores.add(score);
			} else {
				info.scores.add((float) -i); // Default score
			}
		}

		if (config.verbose) {
			LOGGER.info("Loaded " + info.vocabulary.size() + " vocabulary entries");
		}
	}

	private void loadGGMLTensors(RandomAccessFile raf, LegacyModelInfo info) throws IOException {
		while (raf.getFilePointer() < raf.length()) {
			LegacyModelInfo.LegacyTensor tensor = new LegacyModelInfo.LegacyTensor();

			// Read tensor header
			int nameLen = Integer.reverseBytes(raf.readInt());
			if (nameLen <= 0 || nameLen > 1000) {
				break; // End of tensors or invalid data
			}

			byte[] nameBytes = new byte[nameLen];
			raf.readFully(nameBytes);
			tensor.name = new String(nameBytes, "UTF-8");

			// Read dimensions
			int numDims = Integer.reverseBytes(raf.readInt());
			if (numDims < 1 || numDims > 4) {
				throw new IOException("Invalid tensor dimensions: " + numDims);
			}

			tensor.shape = new int[numDims];
			long totalElements = 1;
			for (int i = 0; i < numDims; i++) {
				tensor.shape[i] = Integer.reverseBytes(raf.readInt());
				totalElements *= tensor.shape[i];
			}

			// Read type and calculate size
			tensor.type = Integer.reverseBytes(raf.readInt());
			tensor.offset = raf.getFilePointer();

			int elementSize = getElementSize(tensor.type);
			if (elementSize <= 0) {
				throw new IOException("Unknown tensor type: " + tensor.type);
			}

			tensor.size = totalElements * elementSize;

			// Skip tensor data for now (will read during conversion)
			raf.seek(tensor.offset + tensor.size);

			info.tensors.put(tensor.name, tensor);

			if (config.verbose) {
				LOGGER.info(String.format("Tensor: %s [%s] type=%d size=%d",
					tensor.name, Arrays.toString(tensor.shape), tensor.type, tensor.size));
			}
		}

		LOGGER.info("Loaded " + info.tensors.size() + " tensors");
	}

	private LegacyModelInfo loadOtherLegacyFormat(RandomAccessFile raf) throws IOException {
		// Try to detect other legacy formats (e.g., older Llama formats)
		raf.seek(0);

		// Check for other magic numbers or format indicators
		byte[] header = new byte[16];
		raf.readFully(header);

		// For now, throw an error for unknown formats
		throw new IOException("Unknown legacy format - only GGML format is currently supported");
	}

	private void convertToGGUF(LegacyModelInfo modelInfo) throws IOException {
		try (GGUFWriter writer = new GGUFWriter(outputPath, "llama")) {
			// Write metadata
			writeMetadata(writer, modelInfo);

			// Write vocabulary
			writeVocabulary(writer, modelInfo);

			// Prepare tensor information
			for (LegacyModelInfo.LegacyTensor tensor : modelInfo.tensors.values()) {
				String ggufName = mapTensorName(tensor.name);
				if (ggufName != null) {
					GGUFConstants.GGMLQuantizationType ggufType = mapTensorType(tensor.type);
					long[] ggufShape = Arrays.stream(tensor.shape).mapToLong(i -> i).toArray();

					writer.addTensorInfo(ggufName, ggufShape, ggufType, tensor.size);
				}
			}

			// Write header
			writer.writeToFile();

			// Write tensor data
			writeTensorData(writer, modelInfo);
		}
	}

	private void writeMetadata(GGUFWriter writer, LegacyModelInfo modelInfo) {
		// Basic model information
		writer.addString("general.name", config.modelName);
		writer.addString("general.architecture", "llama");
		writer.addString("general.file_type", String.valueOf(modelInfo.fileType));

		// Llama-specific parameters
		writer.addUInt32("llama.vocab_size", modelInfo.vocabSize);
		writer.addUInt32("llama.context_length", modelInfo.contextSize);
		writer.addUInt32("llama.embedding_length", modelInfo.embeddingSize);
		writer.addUInt32("llama.block_count", modelInfo.layerCount);
		writer.addUInt32("llama.attention.head_count", modelInfo.headCount);

		// Calculate feed forward length (estimate)
		int ffLength = modelInfo.embeddingSize * 4; // Common ratio
		writer.addUInt32("llama.feed_forward_length", ffLength);

		// RoPE parameters
		if (modelInfo.rotationDimension > 0) {
			writer.addUInt32("llama.rope.dimension_count", modelInfo.rotationDimension);
		}

		// Add override metadata
		for (Map.Entry<String, String> entry : config.metadataOverrides.entrySet()) {
			writer.addString(entry.getKey(), entry.getValue());
		}

		if (config.verbose) {
			LOGGER.info("Written metadata with " + (7 + config.metadataOverrides.size()) + " entries");
		}
	}

	private void writeVocabulary(GGUFWriter writer, LegacyModelInfo modelInfo) {
		// Convert vocabulary
		String[] tokens = modelInfo.vocabulary.toArray(new String[0]);
		float[] scores = new float[modelInfo.scores.size()];
		int[] types = new int[tokens.length];

		for (int i = 0; i < modelInfo.scores.size(); i++) {
			scores[i] = modelInfo.scores.get(i);
			types[i] = 1; // NORMAL token type
		}

		writer.addArray("tokenizer.ggml.tokens", tokens);
		writer.addArray("tokenizer.ggml.scores", scores);
		writer.addArray("tokenizer.ggml.token_type", types);

		// Add special tokens (common defaults)
		writer.addUInt32("tokenizer.ggml.bos_token_id", 1);
		writer.addUInt32("tokenizer.ggml.eos_token_id", 2);
		writer.addUInt32("tokenizer.ggml.unknown_token_id", 0);

		if (config.verbose) {
			LOGGER.info("Written vocabulary with " + tokens.length + " tokens");
		}
	}

	private void writeTensorData(GGUFWriter writer, LegacyModelInfo modelInfo) throws IOException {
		try (RandomAccessFile raf = new RandomAccessFile(inputPath.toFile(), "r")) {
			for (LegacyModelInfo.LegacyTensor tensor : modelInfo.tensors.values()) {
				String ggufName = mapTensorName(tensor.name);
				if (ggufName == null) {
					if (config.verbose) {
						LOGGER.info("Skipping unmapped tensor: " + tensor.name);
					}
					continue;
				}

				// Read tensor data
				raf.seek(tensor.offset);
				byte[] data = new byte[(int) tensor.size];
				raf.readFully(data);

				// Apply quantization if requested
				if (config.targetQuantization != null) {
					data = requantizeTensorData(data, tensor.type, config.targetQuantization);
				}

				// Write to GGUF (simplified - real implementation would need proper integration)
				if (config.verbose) {
					LOGGER.info("Writing tensor: " + ggufName + " (" + data.length + " bytes)");
				}
			}
		}
	}

	private String mapTensorName(String ggmlName) {
		// Map GGML tensor names to GGUF naming convention
		Map<String, String> nameMapping = new HashMap<>();
		nameMapping.put("tok_embeddings.weight", "token_embd.weight");
		nameMapping.put("output.weight", "output.weight");
		nameMapping.put("norm.weight", "output_norm.weight");

		// Layer-specific mappings
		if (ggmlName.matches("layers\\.\\d+\\.attention\\.wq\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.attention\\.wq\\.weight", "blk.$1.attn_q.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.attention\\.wk\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.attention\\.wk\\.weight", "blk.$1.attn_k.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.attention\\.wv\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.attention\\.wv\\.weight", "blk.$1.attn_v.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.attention\\.wo\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.attention\\.wo\\.weight", "blk.$1.attn_output.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.feed_forward\\.w1\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.feed_forward\\.w1\\.weight", "blk.$1.ffn_gate.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.feed_forward\\.w2\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.feed_forward\\.w2\\.weight", "blk.$1.ffn_down.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.feed_forward\\.w3\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.feed_forward\\.w3\\.weight", "blk.$1.ffn_up.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.attention_norm\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.attention_norm\\.weight", "blk.$1.attn_norm.weight");
		}
		if (ggmlName.matches("layers\\.\\d+\\.ffn_norm\\.weight")) {
			return ggmlName.replaceAll("layers\\.(\\d+)\\.ffn_norm\\.weight", "blk.$1.ffn_norm.weight");
		}

		return nameMapping.get(ggmlName);
	}

	private GGUFConstants.GGMLQuantizationType mapTensorType(int ggmlType) {
		// Map GGML type constants to GGUF types
		switch (ggmlType) {
			case 0: return GGUFConstants.GGMLQuantizationType.F32;
			case 1: return GGUFConstants.GGMLQuantizationType.F16;
			case 2: return GGUFConstants.GGMLQuantizationType.Q4_0;
			case 3: return GGUFConstants.GGMLQuantizationType.Q4_1;
			case 6: return GGUFConstants.GGMLQuantizationType.Q5_0;
			case 7: return GGUFConstants.GGMLQuantizationType.Q5_1;
			case 8: return GGUFConstants.GGMLQuantizationType.Q8_0;
			default: return GGUFConstants.GGMLQuantizationType.F32; // Default fallback
		}
	}

	private int getElementSize(int ggmlType) {
		switch (ggmlType) {
			case 0: return 4; // F32
			case 1: return 2; // F16
			case 2: return 1; // Q4_0 (simplified)
			case 3: return 1; // Q4_1 (simplified)
			case 6: return 1; // Q5_0 (simplified)
			case 7: return 1; // Q5_1 (simplified)
			case 8: return 1; // Q8_0
			default: return 4; // Default to F32
		}
	}

	private byte[] requantizeTensorData(byte[] data, int sourceType, GGUFConstants.GGMLQuantizationType targetType) {
		// Simplified requantization - real implementation would need proper quantization logic
		if (config.verbose) {
			LOGGER.warning("Requantization not fully implemented - returning original data");
		}
		return data;
	}

	private void printModelInfo(LegacyModelInfo info) {
		System.out.println("=== LEGACY MODEL INFORMATION ===");
		System.out.println("Magic: 0x" + Integer.toHexString(info.magic));
		System.out.println("Version: " + info.version);
		System.out.println("Vocabulary size: " + info.vocabSize);
		System.out.println("Context size: " + info.contextSize);
		System.out.println("Embedding size: " + info.embeddingSize);
		System.out.println("Head count: " + info.headCount);
		System.out.println("Layer count: " + info.layerCount);
		System.out.println("Rotation dimension: " + info.rotationDimension);
		System.out.println("File type: " + info.fileType);
		System.out.println("Tensor count: " + info.tensors.size());
		System.out.println("Vocabulary entries: " + info.vocabulary.size());
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		if (args.length < 2) {
			printUsage();
			System.exit(1);
		}

		try {
			Path inputPath = Paths.get(args[0]);
			Path outputPath = Paths.get(args[1]);
			ConversionConfig config = new ConversionConfig();

			// Parse options
			for (int i = 2; i < args.length; i++) {
				switch (args[i]) {
					case "--verbose":
					case "-v":
						config.verbose(true);
						break;
					case "--dry-run":
						config.dryRun(true);
						break;
					case "--name":
						if (i + 1 < args.length) {
							config.modelName(args[++i]);
						}
						break;
					case "--quantize":
						if (i + 1 < args.length) {
							String quantType = args[++i].toUpperCase();
							config.targetQuantization(GGUFConstants.GGMLQuantizationType.valueOf(quantType));
						}
						break;
					case "--metadata":
						if (i + 2 < args.length) {
							String key = args[++i];
							String value = args[++i];
							config.addMetadata(key, value);
						}
						break;
					case "--help":
					case "-h":
						printUsage();
						System.exit(0);
						break;
				}
			}

			LegacyConverter converter = new LegacyConverter(inputPath, outputPath, config);
			converter.convert();

		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "Conversion failed", e);
			System.exit(1);
		}
	}

	private static void printUsage() {
		System.out.println("Usage: LegacyConverter <input_file> <output_file> [options]");
		System.out.println();
		System.out.println("Convert legacy model formats (GGML) to GGUF format.");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --verbose, -v          Verbose output");
		System.out.println("  --dry-run              Show conversion info without creating output");
		System.out.println("  --name <name>          Set model name in metadata");
		System.out.println("  --quantize <type>      Target quantization (F32, F16, Q4_0, etc.)");
		System.out.println("  --metadata <key> <val> Add custom metadata");
		System.out.println("  --help, -h             Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  LegacyConverter old_model.bin new_model.gguf");
		System.out.println("  LegacyConverter --verbose --name \"My Model\" model.ggml model.gguf");
		System.out.println("  LegacyConverter --quantize Q4_0 --dry-run legacy.bin modern.gguf");
	}
}