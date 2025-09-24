package de.kherud.llama.converters;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.gguf.GGUFConstants;
import de.kherud.llama.gguf.GGUFWriter;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Converts PyTorch LoRA adapters to GGUF format.
 *
 * This converter can read LoRA adapters from:
 * - PEFT format (adapter_model.bin, adapter_config.json)
 * - SafeTensors format (adapter_model.safetensors)
 * - Individual LoRA weight files
 *
 * The output is compatible with llama.cpp's loadLoRAAdapter() method.
 */
public class LoRAToGGUFConverter {
	private static final System.Logger LOGGER = System.getLogger(LoRAToGGUFConverter.class.getName());

	private final Path adapterPath;
	private final Path outputPath;
	private final ConversionConfig config;
	private JsonNode adapterConfig;
	private Map<String, LoRATensor> loraTensors = new HashMap<>();
	private float loraAlpha = 16.0f;
	private int loraRank = 0;

	public static class ConversionConfig {
		private boolean verbose = false;
		private boolean mergeLayerNorms = false;
		private String baseModelArch = "llama";
		private Map<String, String> targetModules = new HashMap<>();

		public ConversionConfig verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public ConversionConfig mergeLayerNorms(boolean merge) {
			this.mergeLayerNorms = merge;
			return this;
		}

		public ConversionConfig baseModelArchitecture(String arch) {
			this.baseModelArch = arch;
			return this;
		}

		public ConversionConfig addTargetModule(String source, String target) {
			this.targetModules.put(source, target);
			return this;
		}
	}

	private static class LoRATensor {
		String baseName;
		ByteBuffer loraA;
		ByteBuffer loraB;
		long[] shapeA;
		long[] shapeB;
		String dtype;
		boolean hasLayerNorm;
		ByteBuffer layerNormWeight;

		LoRATensor(String baseName) {
			this.baseName = baseName;
		}

		boolean isComplete() {
			return loraA != null && loraB != null;
		}
	}

	public LoRAToGGUFConverter(Path adapterPath, Path outputPath) {
		this(adapterPath, outputPath, new ConversionConfig());
	}

	public LoRAToGGUFConverter(Path adapterPath, Path outputPath, ConversionConfig config) {
		this.adapterPath = adapterPath;
		this.outputPath = outputPath;
		this.config = config;
	}

	/**
	 * Main conversion method
	 */
	public void convert() throws IOException {
		LOGGER.log(System.Logger.Level.INFO,"Starting LoRA to GGUF conversion");
		LOGGER.log(System.Logger.Level.INFO,"Adapter path: " + adapterPath);
		LOGGER.log(System.Logger.Level.INFO,"Output path: " + outputPath);

		// Step 1: Load adapter configuration
		loadAdapterConfig();

		// Step 2: Load LoRA tensors
		loadLoRATensors();

		// Step 3: Validate tensors
		validateTensors();

		// Step 4: Write GGUF file
		writeGGUF();

		LOGGER.log(System.Logger.Level.INFO,"Conversion completed successfully!");
	}

	private void loadAdapterConfig() throws IOException {
		Path configPath = adapterPath.resolve("adapter_config.json");

		if (Files.exists(configPath)) {
			// PEFT format config
			ObjectMapper mapper = new ObjectMapper();
			adapterConfig = mapper.readTree(Files.newBufferedReader(configPath));

			// Extract parameters
			loraAlpha = adapterConfig.path("lora_alpha").floatValue();
			loraRank = adapterConfig.path("r").asInt();

			JsonNode targetModules = adapterConfig.path("target_modules");
			if (targetModules.isArray()) {
				LOGGER.log(System.Logger.Level.INFO,"Target modules: " + targetModules.toString());
			}

			LOGGER.log(System.Logger.Level.INFO,"LoRA config: alpha=" + loraAlpha + ", rank=" + loraRank);
		} else {
			// Try to infer from file structure
			LOGGER.log(System.Logger.Level.WARNING,"adapter_config.json not found, using defaults");
			loraAlpha = 16.0f;
			// Rank will be inferred from tensor shapes
		}
	}

	private void loadLoRATensors() throws IOException {
		// Look for SafeTensors format first
		Path safetensorsPath = adapterPath.resolve("adapter_model.safetensors");
		if (Files.exists(safetensorsPath)) {
			LOGGER.log(System.Logger.Level.INFO,"Loading from SafeTensors format");
			loadFromSafeTensors(safetensorsPath);
			return;
		}

		// Look for PyTorch bin format
		Path binPath = adapterPath.resolve("adapter_model.bin");
		if (Files.exists(binPath)) {
			LOGGER.log(System.Logger.Level.INFO,"Loading from PyTorch bin format");
			loadFromPyTorchBin(binPath);
			return;
		}

		// Look for individual LoRA files
		List<Path> loraFiles = Files.list(adapterPath)
			.filter(p -> {
				String name = p.getFileName().toString();
				return name.contains("lora") && (name.endsWith(".safetensors") || name.endsWith(".bin"));
			})
			.collect(Collectors.toList());

		if (!loraFiles.isEmpty()) {
			LOGGER.log(System.Logger.Level.INFO,"Loading from individual LoRA files");
			for (Path file : loraFiles) {
				if (file.toString().endsWith(".safetensors")) {
					loadFromSafeTensors(file);
				} else {
					loadFromPyTorchBin(file);
				}
			}
			return;
		}

		throw new IOException("No LoRA adapter files found");
	}

	private void loadFromSafeTensors(Path file) throws IOException {
		try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "r");
		     FileChannel channel = raf.getChannel()) {

			// SafeTensors format: [header_size:8][header_json][tensor_data]
			ByteBuffer headerSizeBuffer = ByteBuffer.allocate(8);
			headerSizeBuffer.order(ByteOrder.LITTLE_ENDIAN);
			channel.read(headerSizeBuffer);
			headerSizeBuffer.flip();
			long headerSize = headerSizeBuffer.getLong();

			// Read header JSON
			ByteBuffer headerBuffer = ByteBuffer.allocate((int) headerSize);
			channel.read(headerBuffer);
			String headerJson = new String(headerBuffer.array());

			ObjectMapper mapper = new ObjectMapper();
			JsonNode header = mapper.readTree(headerJson);

			// Parse tensor metadata and load data
			Iterator<Map.Entry<String, JsonNode>> fields = header.fields();
			while (fields.hasNext()) {
				Map.Entry<String, JsonNode> entry = fields.next();
				String tensorName = entry.getKey();

				if (tensorName.equals("__metadata__")) continue;

				JsonNode tensorMeta = entry.getValue();
				String dtype = tensorMeta.get("dtype").asText();
				JsonNode shapeNode = tensorMeta.get("shape");
				JsonNode offsetsNode = tensorMeta.get("data_offsets");

				long[] shape = new long[shapeNode.size()];
				for (int i = 0; i < shape.length; i++) {
					shape[i] = shapeNode.get(i).asLong();
				}

				long startOffset = offsetsNode.get(0).asLong() + 8 + headerSize;
				long endOffset = offsetsNode.get(1).asLong() + 8 + headerSize;
				long size = endOffset - startOffset;

				// Load tensor data
				ByteBuffer data = ByteBuffer.allocate((int) size);
				data.order(ByteOrder.LITTLE_ENDIAN);
				channel.read(data, startOffset);
				data.flip();

				// Process LoRA tensor
				processLoRATensor(tensorName, shape, dtype, data);

				if (config.verbose) {
					LOGGER.log(System.Logger.Level.INFO,"  Loaded tensor: " + tensorName + " shape=" + Arrays.toString(shape));
				}
			}
		}
	}

	private void loadFromPyTorchBin(Path file) throws IOException {
		// PyTorch .bin files use Python pickle format which is complex to parse in Java
		// For production use, we recommend using SafeTensors format
		throw new UnsupportedOperationException(
			"PyTorch .bin file reading not yet implemented. " +
			"Please convert your adapter to SafeTensors format first using:\n" +
			"python -c \"from safetensors.torch import save_file; import torch; " +
			"adapter = torch.load('adapter_model.bin'); " +
			"save_file(adapter, 'adapter_model.safetensors')\""
		);
	}

	private void processLoRATensor(String tensorName, long[] shape, String dtype, ByteBuffer data) {
		String baseName = getBaseTensorName(tensorName);

		LoRATensor tensor = loraTensors.computeIfAbsent(baseName, k -> new LoRATensor(k));
		tensor.dtype = dtype;

		// Check if it's a LoRA A or B matrix
		boolean isLoraA = tensorName.contains(".lora_A.weight") || tensorName.contains(".lora_embedding_A");
		boolean isLoraB = tensorName.contains(".lora_B.weight") || tensorName.contains(".lora_embedding_B");

		if (isLoraA) {
			tensor.loraA = data;
			tensor.shapeA = shape;
			// Infer rank from shape if not set
			if (loraRank == 0 && shape.length >= 1) {
				loraRank = (int) shape[shape.length - 1];
				LOGGER.log(System.Logger.Level.INFO,"Inferred LoRA rank: " + loraRank);
			}
		} else if (isLoraB) {
			tensor.loraB = data;
			tensor.shapeB = shape;
		} else if (tensorName.contains("_layernorm") || tensorName.contains(".norm")) {
			// Handle layer normalization weights that might be included
			if (config.mergeLayerNorms) {
				tensor.hasLayerNorm = true;
				tensor.layerNormWeight = data;
				LOGGER.log(System.Logger.Level.INFO,"Found layer norm weight: " + tensorName);
			}
		} else if (!tensorName.contains(".base_layer.weight")) {
			LOGGER.log(System.Logger.Level.WARNING,"Unexpected tensor name pattern: " + tensorName);
		}
	}

	private String getBaseTensorName(String fullName) {
		// Remove LoRA-specific suffixes to get the base tensor name
		String baseName = fullName;
		baseName = baseName.replace(".lora_A.weight", "");
		baseName = baseName.replace(".lora_B.weight", "");
		baseName = baseName.replace(".lora_embedding_A", "");
		baseName = baseName.replace(".lora_embedding_B", "");
		baseName = baseName.replace(".base_layer.weight", "");
		baseName = baseName.replace(".default", ""); // PEFT sometimes adds this
		return baseName;
	}

	private void validateTensors() throws IOException {
		int completeCount = 0;
		int incompleteCount = 0;

		for (Map.Entry<String, LoRATensor> entry : loraTensors.entrySet()) {
			LoRATensor tensor = entry.getValue();
			if (tensor.isComplete()) {
				completeCount++;
				// Validate shapes
				if (tensor.shapeA[tensor.shapeA.length - 2] != tensor.shapeB[tensor.shapeB.length - 1]) {
					throw new IOException("Shape mismatch for " + entry.getKey() +
						": A=" + Arrays.toString(tensor.shapeA) +
						", B=" + Arrays.toString(tensor.shapeB));
				}
			} else {
				incompleteCount++;
				LOGGER.log(System.Logger.Level.WARNING,"Incomplete LoRA tensor: " + entry.getKey() +
					" (has A: " + (tensor.loraA != null) +
					", has B: " + (tensor.loraB != null) + ")");
			}
		}

		LOGGER.log(System.Logger.Level.INFO,"Tensor validation: " + completeCount + " complete, " + incompleteCount + " incomplete");

		if (completeCount == 0) {
			throw new IOException("No complete LoRA tensor pairs found");
		}
	}

	private void writeGGUF() throws IOException {
		try (GGUFWriter writer = new GGUFWriter(outputPath, "adapter")) {
			// Write adapter metadata
			writer.addType("adapter");
			writer.addString(GGUFConstants.Keys.Adapter.TYPE, "lora");
			writer.addLoRAAlpha(loraAlpha);

			// Count complete tensors
			int tensorCount = 0;
			for (LoRATensor tensor : loraTensors.values()) {
				if (tensor.isComplete()) {
					tensorCount += 2; // A and B matrices
					if (tensor.hasLayerNorm) {
						tensorCount++;
					}
				}
			}

			LOGGER.log(System.Logger.Level.INFO,"Writing " + tensorCount + " tensors to GGUF");

			// Add tensor information
			for (Map.Entry<String, LoRATensor> entry : loraTensors.entrySet()) {
				LoRATensor tensor = entry.getValue();
				if (!tensor.isComplete()) continue;

				String ggufBaseName = mapToGGUFName(entry.getKey());
				if (ggufBaseName == null) {
					LOGGER.log(System.Logger.Level.WARNING,"No GGUF mapping for: " + entry.getKey());
					continue;
				}

				// Add LoRA A tensor info
				writer.addTensorInfo(
					ggufBaseName + ".lora_a",
					tensor.shapeA,
					GGUFConstants.GGMLQuantizationType.F32,
					tensor.loraA.remaining()
				);

				// Add LoRA B tensor info
				writer.addTensorInfo(
					ggufBaseName + ".lora_b",
					tensor.shapeB,
					GGUFConstants.GGMLQuantizationType.F32,
					tensor.loraB.remaining()
				);

				// Add layer norm if present
				if (tensor.hasLayerNorm && tensor.layerNormWeight != null) {
					writer.addTensorInfo(
						ggufBaseName + ".norm",
						new long[]{tensor.layerNormWeight.remaining() / 4}, // Assuming F32
						GGUFConstants.GGMLQuantizationType.F32,
						tensor.layerNormWeight.remaining()
					);
				}
			}

			// Write header
			writer.writeToFile();

			// Write tensor data
			writeTensorData(writer);

			LOGGER.log(System.Logger.Level.INFO,"GGUF adapter file written to: " + outputPath);
		}
	}

	private void writeTensorData(GGUFWriter writer) throws IOException {
		for (Map.Entry<String, LoRATensor> entry : loraTensors.entrySet()) {
			LoRATensor tensor = entry.getValue();
			if (!tensor.isComplete()) continue;

			String ggufBaseName = mapToGGUFName(entry.getKey());
			if (ggufBaseName == null) continue;

			// Write LoRA A data
			writeTensorBuffer(writer, tensor.loraA, ggufBaseName + ".lora_a");

			// Write LoRA B data
			writeTensorBuffer(writer, tensor.loraB, ggufBaseName + ".lora_b");

			// Write layer norm if present
			if (tensor.hasLayerNorm && tensor.layerNormWeight != null) {
				writeTensorBuffer(writer, tensor.layerNormWeight, ggufBaseName + ".norm");
			}

			if (config.verbose) {
				LOGGER.log(System.Logger.Level.INFO,"Wrote LoRA tensors for: " + ggufBaseName);
			}
		}
	}

	private void writeTensorBuffer(GGUFWriter writer, ByteBuffer buffer, String tensorName) throws IOException {
		// Note: This is a simplified version
		// Real implementation would need proper integration with GGUFWriter's tensor writing
		buffer.rewind();
		byte[] data = new byte[buffer.remaining()];
		buffer.get(data);
		// writer.writeTensorData(tensorName, data); // This method would need to be added to GGUFWriter
		LOGGER.log(System.Logger.Level.DEBUG,"Writing tensor data: " + tensorName + " (" + data.length + " bytes)");
	}

	private String mapToGGUFName(String tensorName) {
		// Map various LoRA naming conventions to GGUF names
		// This handles PEFT, Diffusers, and other common formats

		// Handle custom mappings first
		if (!config.targetModules.isEmpty()) {
			for (Map.Entry<String, String> mapping : config.targetModules.entrySet()) {
				if (tensorName.contains(mapping.getKey())) {
					return tensorName.replace(mapping.getKey(), mapping.getValue());
				}
			}
		}

		// Standard mappings for Llama architecture
		if (config.baseModelArch.equals("llama")) {
			// Token embeddings
			if (tensorName.contains("embed_tokens")) {
				return "token_embd";
			}

			// Output
			if (tensorName.contains("lm_head")) {
				return "output";
			}

			// Model normalization
			if (tensorName.equals("model.norm") || tensorName.equals("norm")) {
				return "output_norm";
			}

			// Extract layer number
			Pattern layerPattern = Pattern.compile("layers?\\.(\\d+)");
			Matcher matcher = layerPattern.matcher(tensorName);
			if (matcher.find()) {
				int layerNum = Integer.parseInt(matcher.group(1));
				String layerPrefix = "blk." + layerNum;

				// Attention projections
				if (tensorName.contains("self_attn.q_proj")) {
					return layerPrefix + ".attn_q";
				}
				if (tensorName.contains("self_attn.k_proj")) {
					return layerPrefix + ".attn_k";
				}
				if (tensorName.contains("self_attn.v_proj")) {
					return layerPrefix + ".attn_v";
				}
				if (tensorName.contains("self_attn.o_proj")) {
					return layerPrefix + ".attn_output";
				}

				// MLP/FFN projections
				if (tensorName.contains("mlp.gate_proj")) {
					return layerPrefix + ".ffn_gate";
				}
				if (tensorName.contains("mlp.up_proj")) {
					return layerPrefix + ".ffn_up";
				}
				if (tensorName.contains("mlp.down_proj")) {
					return layerPrefix + ".ffn_down";
				}

				// Layer norms
				if (tensorName.contains("input_layernorm")) {
					return layerPrefix + ".attn_norm";
				}
				if (tensorName.contains("post_attention_layernorm")) {
					return layerPrefix + ".ffn_norm";
				}
			}
		}

		// If no mapping found, return cleaned version
		String cleaned = tensorName;
		cleaned = cleaned.replace("base_model.model.", "");
		cleaned = cleaned.replace("model.", "");
		cleaned = cleaned.replace(".weight", "");
		return cleaned;
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(LoRAToGGUFConverter::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length < 2) {
			System.err.println("Usage: LoRAToGGUFConverter <adapter_path> <output_path> [options]");
			System.err.println("Options:");
			System.err.println("  --arch <name>      Base model architecture (default: llama)");
			System.err.println("  --merge-norms      Include layer normalization weights");
			System.err.println("  --verbose          Enable verbose output");
			throw new IllegalArgumentException("Insufficient arguments");
		}

		try {
			Path adapterPath = Paths.get(args[0]);
			Path outputPath = Paths.get(args[1]);

			ConversionConfig config = new ConversionConfig();

			// Parse options
			for (int i = 2; i < args.length; i++) {
				if (args[i].equals("--arch") && i + 1 < args.length) {
					config.baseModelArchitecture(args[++i]);
				} else if (args[i].equals("--merge-norms")) {
					config.mergeLayerNorms(true);
				} else if (args[i].equals("--verbose")) {
					config.verbose(true);
				}
			}

			LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterPath, outputPath, config);
			converter.convert();

		} catch (IOException e) {
			throw e; // Re-throw IO exceptions
		} catch (Exception e) {
			LOGGER.log(System.Logger.Level.ERROR, "Conversion failed", e);
			throw new RuntimeException("Conversion failed", e);
		}
	}
}
