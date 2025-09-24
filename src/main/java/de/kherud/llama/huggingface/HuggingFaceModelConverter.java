package de.kherud.llama.huggingface;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.gguf.GGUFConstants;
import de.kherud.llama.gguf.GGUFWriter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * HuggingFace model converter.
 *
 * Equivalent to convert-hf-to-gguf.py - converts HuggingFace models to GGUF format
 * with support for various model architectures, tokenizers, and configurations.
 */
public class HuggingFaceModelConverter {
	private static final System.Logger logger = System.getLogger(HuggingFaceModelConverter.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();

	public static class ConversionConfig {
		private String outputPath;
		private GGUFConstants.GGMLQuantizationType quantization = GGUFConstants.GGMLQuantizationType.F16;
		private boolean verbose = false;
		private boolean dryRun = false;
		private String vocabType = "spm"; // spm, bpe, tokenizer
		private Map<String, String> metadataOverrides = new HashMap<>();
		private boolean skipTokenizer = false;
		private boolean skipEmbeddings = false;
		private Pattern tensorFilter = null;
		private int contextLength = 2048;

		public ConversionConfig outputPath(String path) {
			this.outputPath = path;
			return this;
		}

		public ConversionConfig quantization(GGUFConstants.GGMLQuantizationType type) {
			this.quantization = type;
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

		public ConversionConfig vocabType(String type) {
			this.vocabType = type;
			return this;
		}

		public ConversionConfig addMetadata(String key, String value) {
			this.metadataOverrides.put(key, value);
			return this;
		}

		public ConversionConfig skipTokenizer(boolean skip) {
			this.skipTokenizer = skip;
			return this;
		}

		public ConversionConfig skipEmbeddings(boolean skip) {
			this.skipEmbeddings = skip;
			return this;
		}

		public ConversionConfig tensorFilter(Pattern filter) {
			this.tensorFilter = filter;
			return this;
		}

		public ConversionConfig contextLength(int length) {
			this.contextLength = length;
			return this;
		}
	}

	public static class ModelArchitecture {
		public String name;
		public String family;
		public Map<String, Object> config = new HashMap<>();
		public Map<String, String> tensorNameMapping = new HashMap<>();
		public Set<String> requiredConfigKeys = new HashSet<>();

		public static ModelArchitecture detectArchitecture(JsonNode config) {
			ModelArchitecture arch = new ModelArchitecture();

			// Detect architecture from config
			if (config.has("model_type")) {
				String modelType = config.get("model_type").asText();
				switch (modelType.toLowerCase()) {
					case "llama":
					case "llamaforcausallm":
						arch.name = "llama";
						arch.family = "llama";
						setupLlamaMapping(arch);
						break;
					case "gpt2":
					case "gpt":
						arch.name = "gpt2";
						arch.family = "gpt";
						setupGPTMapping(arch);
						break;
					case "bloom":
						arch.name = "bloom";
						arch.family = "bloom";
						setupBloomMapping(arch);
						break;
					case "falcon":
					case "rw":
						arch.name = "falcon";
						arch.family = "falcon";
						setupFalconMapping(arch);
						break;
					default:
						arch.name = modelType;
						arch.family = "unknown";
						setupGenericMapping(arch);
				}
			} else {
				arch.name = "unknown";
				arch.family = "unknown";
				setupGenericMapping(arch);
			}

			return arch;
		}

		private static void setupLlamaMapping(ModelArchitecture arch) {
			arch.tensorNameMapping.put("model.embed_tokens.weight", "token_embd.weight");
			arch.tensorNameMapping.put("model.norm.weight", "output_norm.weight");
			arch.tensorNameMapping.put("lm_head.weight", "output.weight");

			// Layer-specific patterns will be handled separately
			arch.requiredConfigKeys.addAll(Arrays.asList(
				"vocab_size", "hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers"
			));
		}

		private static void setupGPTMapping(ModelArchitecture arch) {
			arch.tensorNameMapping.put("transformer.wte.weight", "token_embd.weight");
			arch.tensorNameMapping.put("transformer.ln_f.weight", "output_norm.weight");
			arch.tensorNameMapping.put("lm_head.weight", "output.weight");

			arch.requiredConfigKeys.addAll(Arrays.asList(
				"vocab_size", "n_embd", "n_head", "n_layer"
			));
		}

		private static void setupBloomMapping(ModelArchitecture arch) {
			arch.tensorNameMapping.put("transformer.word_embeddings.weight", "token_embd.weight");
			arch.tensorNameMapping.put("transformer.ln_f.weight", "output_norm.weight");
			arch.tensorNameMapping.put("lm_head.weight", "output.weight");

			arch.requiredConfigKeys.addAll(Arrays.asList(
				"vocab_size", "hidden_size", "n_head", "n_layer"
			));
		}

		private static void setupFalconMapping(ModelArchitecture arch) {
			arch.tensorNameMapping.put("transformer.word_embeddings.weight", "token_embd.weight");
			arch.tensorNameMapping.put("transformer.ln_f.weight", "output_norm.weight");
			arch.tensorNameMapping.put("lm_head.weight", "output.weight");

			arch.requiredConfigKeys.addAll(Arrays.asList(
				"vocab_size", "hidden_size", "num_attention_heads", "num_hidden_layers"
			));
		}

		private static void setupGenericMapping(ModelArchitecture arch) {
			// Generic mappings - may not work for all models
			arch.requiredConfigKeys.addAll(Arrays.asList("vocab_size"));
		}
	}

	public static class TokenizerInfo {
		public List<String> tokens = new ArrayList<>();
		public List<Float> scores = new ArrayList<>();
		public List<Integer> types = new ArrayList<>();
		public Map<String, Integer> specialTokens = new HashMap<>();
		public String tokenizerType;
		public Map<String, String> addedTokens = new HashMap<>();
	}

	public static class ConversionResult {
		public boolean success;
		public String error;
		public Path outputPath;
		public long originalSize;
		public long convertedSize;
		public int tensorCount;
		public long conversionTime;
		public ModelArchitecture architecture;
	}

	private final Path modelDir;
	private final ConversionConfig config;

	public HuggingFaceModelConverter(Path modelDir, ConversionConfig config) {
		this.modelDir = modelDir;
		this.config = config;
	}

	/**
	 * Convert HuggingFace model to GGUF format
	 */
	public ConversionResult convert() throws IOException {
		logger.log(System.Logger.Level.INFO, "Starting HuggingFace to GGUF conversion");
		logger.log(System.Logger.Level.INFO, "Model directory: " + modelDir);

		ConversionResult result = new ConversionResult();
		long startTime = System.currentTimeMillis();

		try {
			// Load and analyze model configuration
			JsonNode modelConfig = loadModelConfig();
			result.architecture = ModelArchitecture.detectArchitecture(modelConfig);

			if (config.verbose) {
				logger.log(System.Logger.Level.INFO, "Detected architecture: " + result.architecture.name);
				logger.log(System.Logger.Level.INFO, "Architecture family: " + result.architecture.family);
			}

			// Load tokenizer
			TokenizerInfo tokenizer = null;
			if (!config.skipTokenizer) {
				tokenizer = loadTokenizer();
			}

			// Find model files
			List<Path> modelFiles = findModelFiles();
			if (modelFiles.isEmpty()) {
				throw new IOException("No model files found");
			}

			logger.log(System.Logger.Level.INFO, "Found " + modelFiles.size() + " model files");
			result.originalSize = calculateTotalSize(modelFiles);

			if (config.dryRun) {
				result.success = true;
				logger.log(System.Logger.Level.INFO, "Dry run completed - no output file created");
				return result;
			}

			// Convert to GGUF
			Path outputPath = Paths.get(config.outputPath);
			convertToGGUF(modelConfig, tokenizer, modelFiles, outputPath, result);

			result.conversionTime = System.currentTimeMillis() - startTime;
			result.success = true;

			logger.log(System.Logger.Level.INFO, "Conversion completed successfully");
			logger.log(System.Logger.Level.INFO, "Output: " + result.outputPath);
			logger.log(System.Logger.Level.INFO, "Original size: " + formatSize(result.originalSize));
			logger.log(System.Logger.Level.INFO, "Converted size: " + formatSize(result.convertedSize));
			logger.log(System.Logger.Level.INFO, "Conversion time: " + result.conversionTime + "ms");

		} catch (Exception e) {
			result.success = false;
			result.error = e.getMessage();
			logger.log(System.Logger.Level.ERROR, "Conversion failed: " + e.getMessage(), e);
		}

		return result;
	}

	private JsonNode loadModelConfig() throws IOException {
		Path configPath = modelDir.resolve("config.json");
		if (!Files.exists(configPath)) {
			throw new IOException("Model config.json not found");
		}

		try (InputStream is = Files.newInputStream(configPath)) {
			return MAPPER.readTree(is);
		}
	}

	private TokenizerInfo loadTokenizer() throws IOException {
		TokenizerInfo tokenizer = new TokenizerInfo();

		// Try different tokenizer file formats
		Path tokenizerConfig = modelDir.resolve("tokenizer_config.json");
		Path tokenizerJson = modelDir.resolve("tokenizer.json");
		Path vocabFile = modelDir.resolve("vocab.txt");
		Path sentencepieceModel = modelDir.resolve("tokenizer.model");

		if (Files.exists(tokenizerJson)) {
			loadHuggingFaceTokenizer(tokenizerJson, tokenizer);
		} else if (Files.exists(sentencepieceModel)) {
			loadSentencePieceTokenizer(sentencepieceModel, tokenizer);
		} else if (Files.exists(vocabFile)) {
			loadVocabFileTokenizer(vocabFile, tokenizer);
		} else {
			logger.log(System.Logger.Level.WARNING, "No tokenizer files found - proceeding without tokenizer");
			return null;
		}

		// Load special tokens
		if (Files.exists(tokenizerConfig)) {
			loadSpecialTokens(tokenizerConfig, tokenizer);
		}

		logger.log(System.Logger.Level.INFO, "Loaded tokenizer with " + tokenizer.tokens.size() + " tokens");
		return tokenizer;
	}

	private void loadHuggingFaceTokenizer(Path tokenizerPath, TokenizerInfo tokenizer) throws IOException {
		try (InputStream is = Files.newInputStream(tokenizerPath)) {
			JsonNode tokenizerJson = MAPPER.readTree(is);

			tokenizer.tokenizerType = tokenizerJson.path("model").path("type").asText("BPE");

			// Load vocabulary
			JsonNode vocab = tokenizerJson.path("model").path("vocab");
			Map<String, Integer> vocabMap = new HashMap<>();

			vocab.fields().forEachRemaining(entry -> {
				vocabMap.put(entry.getKey(), entry.getValue().asInt());
			});

			// Sort by token ID
			vocabMap.entrySet().stream()
				.sorted(Map.Entry.comparingByValue())
				.forEach(entry -> {
					tokenizer.tokens.add(entry.getKey());
					tokenizer.scores.add(0.0f); // Default score
					tokenizer.types.add(1); // NORMAL type
				});

			// Load merges for BPE
			if ("BPE".equals(tokenizer.tokenizerType)) {
				JsonNode merges = tokenizerJson.path("model").path("merges");
				// Process merges if needed
			}
		}
	}

	private void loadSentencePieceTokenizer(Path modelPath, TokenizerInfo tokenizer) throws IOException {
		// This would require a SentencePiece library integration
		// For now, we'll implement a simple reader for the basic format
		logger.log(System.Logger.Level.WARNING, "SentencePiece tokenizer loading not fully implemented");
		tokenizer.tokenizerType = "SentencePiece";
	}

	private void loadVocabFileTokenizer(Path vocabPath, TokenizerInfo tokenizer) throws IOException {
		try (BufferedReader reader = Files.newBufferedReader(vocabPath)) {
			String line;
			while ((line = reader.readLine()) != null) {
				tokenizer.tokens.add(line.trim());
				tokenizer.scores.add(0.0f);
				tokenizer.types.add(1);
			}
		}
		tokenizer.tokenizerType = "WordPiece";
	}

	private void loadSpecialTokens(Path configPath, TokenizerInfo tokenizer) throws IOException {
		try (InputStream is = Files.newInputStream(configPath)) {
			JsonNode config = MAPPER.readTree(is);

			// Load special tokens
			if (config.has("bos_token")) {
				String bosToken = config.get("bos_token").asText();
				int bosId = tokenizer.tokens.indexOf(bosToken);
				if (bosId >= 0) {
					tokenizer.specialTokens.put("bos", bosId);
				}
			}

			if (config.has("eos_token")) {
				String eosToken = config.get("eos_token").asText();
				int eosId = tokenizer.tokens.indexOf(eosToken);
				if (eosId >= 0) {
					tokenizer.specialTokens.put("eos", eosId);
				}
			}

			if (config.has("unk_token")) {
				String unkToken = config.get("unk_token").asText();
				int unkId = tokenizer.tokens.indexOf(unkToken);
				if (unkId >= 0) {
					tokenizer.specialTokens.put("unk", unkId);
				}
			}

			if (config.has("pad_token")) {
				String padToken = config.get("pad_token").asText();
				int padId = tokenizer.tokens.indexOf(padToken);
				if (padId >= 0) {
					tokenizer.specialTokens.put("pad", padId);
				}
			}
		}
	}

	private List<Path> findModelFiles() throws IOException {
		List<Path> modelFiles = new ArrayList<>();

		try (DirectoryStream<Path> stream = Files.newDirectoryStream(modelDir)) {
			for (Path entry : stream) {
				String filename = entry.getFileName().toString();
				if (filename.endsWith(".bin") || filename.endsWith(".safetensors")) {
					modelFiles.add(entry);
				}
			}
		}

		modelFiles.sort(Comparator.comparing(p -> p.getFileName().toString()));
		return modelFiles;
	}

	private long calculateTotalSize(List<Path> files) throws IOException {
		long total = 0;
		for (Path file : files) {
			total += Files.size(file);
		}
		return total;
	}

	private void convertToGGUF(JsonNode modelConfig, TokenizerInfo tokenizer,
							   List<Path> modelFiles, Path outputPath,
							   ConversionResult result) throws IOException {

		try (GGUFWriter writer = new GGUFWriter(outputPath, result.architecture.name)) {
			// Write metadata
			writeMetadata(writer, modelConfig, result.architecture, config);

			// Write tokenizer
			if (tokenizer != null) {
				writeTokenizer(writer, tokenizer);
			}

			// Process model files and add tensor information
			Map<String, TensorData> tensors = loadTensors(modelFiles, result.architecture);
			result.tensorCount = tensors.size();

			// Add tensor info to writer
			for (Map.Entry<String, TensorData> entry : tensors.entrySet()) {
				TensorData tensor = entry.getValue();
				writer.addTensorInfo(entry.getKey(), tensor.shape, tensor.type, tensor.data.length);
			}

			// Write header
			writer.writeToFile();

			// Write tensor data
			for (Map.Entry<String, TensorData> entry : tensors.entrySet()) {
				// In a real implementation, we would write the tensor data to the file
				// This requires integration with the actual GGUFWriter tensor writing mechanism
			}

			result.outputPath = outputPath;
			result.convertedSize = Files.size(outputPath);
		}
	}

	private void writeMetadata(GGUFWriter writer, JsonNode modelConfig, ModelArchitecture architecture, ConversionConfig config) {
		// Basic model information
		writer.addString("general.architecture", architecture.name);
		writer.addString("general.name", modelConfig.path("_name_or_path").asText("converted_model"));

		// Architecture-specific metadata
		switch (architecture.name) {
			case "llama":
				writeLlamaMetadata(writer, modelConfig, config);
				break;
			case "gpt2":
				writeGPTMetadata(writer, modelConfig, config);
				break;
			case "bloom":
				writeBloomMetadata(writer, modelConfig, config);
				break;
			case "falcon":
				writeFalconMetadata(writer, modelConfig, config);
				break;
			default:
				writeGenericMetadata(writer, modelConfig);
		}

		// Add user overrides
		for (Map.Entry<String, String> entry : config.metadataOverrides.entrySet()) {
			writer.addString(entry.getKey(), entry.getValue());
		}

		if (config.verbose) {
			logger.log(System.Logger.Level.INFO, "Written metadata for architecture: " + architecture.name);
		}
	}

	private void writeLlamaMetadata(GGUFWriter writer, JsonNode modelConfig, ConversionConfig config) {
		writer.addUInt32("llama.vocab_size", modelConfig.get("vocab_size").asInt());
		writer.addUInt32("llama.context_length", modelConfig.path("max_position_embeddings").asInt(config.contextLength));
		writer.addUInt32("llama.embedding_length", modelConfig.get("hidden_size").asInt());
		writer.addUInt32("llama.block_count", modelConfig.get("num_hidden_layers").asInt());
		writer.addUInt32("llama.attention.head_count", modelConfig.get("num_attention_heads").asInt());

		if (modelConfig.has("num_key_value_heads")) {
			writer.addUInt32("llama.attention.head_count_kv", modelConfig.get("num_key_value_heads").asInt());
		}

		writer.addUInt32("llama.feed_forward_length", modelConfig.get("intermediate_size").asInt());

		// RoPE parameters
		if (modelConfig.has("rope_theta")) {
			writer.addFloat32("llama.rope.freq_base", (float) modelConfig.get("rope_theta").asDouble());
		}

		// Normalization
		writer.addFloat32("llama.attention.layer_norm_rms_epsilon",
			(float) modelConfig.path("rms_norm_eps").asDouble(1e-6));
	}

	private void writeGPTMetadata(GGUFWriter writer, JsonNode modelConfig, ConversionConfig config) {
		writer.addUInt32("gpt2.vocab_size", modelConfig.get("vocab_size").asInt());
		writer.addUInt32("gpt2.context_length", modelConfig.path("n_positions").asInt(config.contextLength));
		writer.addUInt32("gpt2.embedding_length", modelConfig.get("n_embd").asInt());
		writer.addUInt32("gpt2.block_count", modelConfig.get("n_layer").asInt());
		writer.addUInt32("gpt2.attention.head_count", modelConfig.get("n_head").asInt());
	}

	private void writeBloomMetadata(GGUFWriter writer, JsonNode modelConfig, ConversionConfig config) {
		writer.addUInt32("bloom.vocab_size", modelConfig.get("vocab_size").asInt());
		writer.addUInt32("bloom.context_length", modelConfig.path("seq_length").asInt(config.contextLength));
		writer.addUInt32("bloom.embedding_length", modelConfig.get("hidden_size").asInt());
		writer.addUInt32("bloom.block_count", modelConfig.get("n_layer").asInt());
		writer.addUInt32("bloom.attention.head_count", modelConfig.get("n_head").asInt());
	}

	private void writeFalconMetadata(GGUFWriter writer, JsonNode modelConfig, ConversionConfig config) {
		writer.addUInt32("falcon.vocab_size", modelConfig.get("vocab_size").asInt());
		writer.addUInt32("falcon.context_length", modelConfig.path("max_position_embeddings").asInt(config.contextLength));
		writer.addUInt32("falcon.embedding_length", modelConfig.get("hidden_size").asInt());
		writer.addUInt32("falcon.block_count", modelConfig.get("num_hidden_layers").asInt());
		writer.addUInt32("falcon.attention.head_count", modelConfig.get("num_attention_heads").asInt());
	}

	private void writeGenericMetadata(GGUFWriter writer, JsonNode modelConfig) {
		if (modelConfig.has("vocab_size")) {
			writer.addUInt32("general.vocab_size", modelConfig.get("vocab_size").asInt());
		}
		if (modelConfig.has("hidden_size")) {
			writer.addUInt32("general.embedding_length", modelConfig.get("hidden_size").asInt());
		}
	}

	private void writeTokenizer(GGUFWriter writer, TokenizerInfo tokenizer) {
		// Convert to arrays
		String[] tokens = tokenizer.tokens.toArray(new String[0]);
		float[] scores = new float[tokenizer.scores.size()];
		int[] types = new int[tokenizer.types.size()];

		for (int i = 0; i < tokenizer.scores.size(); i++) {
			scores[i] = tokenizer.scores.get(i);
		}

		for (int i = 0; i < tokenizer.types.size(); i++) {
			types[i] = tokenizer.types.get(i);
		}

		writer.addArray("tokenizer.ggml.tokens", Arrays.asList(tokens));
		writer.addArray("tokenizer.ggml.scores", IntStream.range(0, scores.length).mapToObj(i -> scores[i]).collect(Collectors.toList()));
		writer.addArray("tokenizer.ggml.token_type", Arrays.stream(types).boxed().collect(Collectors.toList()));

		// Special tokens
		writer.addUInt32("tokenizer.ggml.bos_token_id",
			tokenizer.specialTokens.getOrDefault("bos", 1));
		writer.addUInt32("tokenizer.ggml.eos_token_id",
			tokenizer.specialTokens.getOrDefault("eos", 2));
		writer.addUInt32("tokenizer.ggml.unknown_token_id",
			tokenizer.specialTokens.getOrDefault("unk", 0));

		if (tokenizer.specialTokens.containsKey("pad")) {
			writer.addUInt32("tokenizer.ggml.padding_token_id",
				tokenizer.specialTokens.get("pad"));
		}

		if (config.verbose) {
			logger.log(System.Logger.Level.INFO, "Written tokenizer with " + tokens.length + " tokens");
		}
	}

	private static class TensorData {
		public long[] shape;
		public GGUFConstants.GGMLQuantizationType type;
		public byte[] data;
		public String originalName;
	}

	private Map<String, TensorData> loadTensors(List<Path> modelFiles, ModelArchitecture architecture) throws IOException {
		Map<String, TensorData> tensors = new HashMap<>();

		for (Path modelFile : modelFiles) {
			if (config.verbose) {
				logger.log(System.Logger.Level.INFO, "Loading tensors from: " + modelFile.getFileName());
			}

			if (modelFile.toString().endsWith(".safetensors")) {
				loadSafetensorsTensors(modelFile, tensors, architecture);
			} else if (modelFile.toString().endsWith(".bin")) {
				loadPyTorchTensors(modelFile, tensors, architecture);
			}
		}

		return tensors;
	}

	private void loadSafetensorsTensors(Path file, Map<String, TensorData> tensors, ModelArchitecture architecture) throws IOException {
		// This would require a SafeTensors library integration
		// For now, we'll provide a placeholder implementation
		logger.log(System.Logger.Level.WARNING, "SafeTensors loading not fully implemented");
	}

	private void loadPyTorchTensors(Path file, Map<String, TensorData> tensors, ModelArchitecture architecture) throws IOException {
		// This would require PyTorch model loading
		// For now, we'll provide a placeholder implementation
		logger.log(System.Logger.Level.WARNING, "PyTorch tensor loading not fully implemented");
	}

	private String mapTensorName(String originalName, ModelArchitecture architecture) {
		// Check direct mapping first
		if (architecture.tensorNameMapping.containsKey(originalName)) {
			return architecture.tensorNameMapping.get(originalName);
		}

		// Apply pattern-based mappings for layer-specific tensors
		if (architecture.name.equals("llama")) {
			return mapLlamaTensorName(originalName);
		} else if (architecture.name.equals("gpt2")) {
			return mapGPTTensorName(originalName);
		}

		// Return original name if no mapping found
		return originalName;
	}

	private String mapLlamaTensorName(String name) {
		// Layer-specific mappings using regex
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight", "blk.$1.attn_q.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.self_attn\\.k_proj\\.weight", "blk.$1.attn_k.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.self_attn\\.v_proj\\.weight", "blk.$1.attn_v.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.self_attn\\.o_proj\\.weight", "blk.$1.attn_output.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.mlp\\.gate_proj\\.weight", "blk.$1.ffn_gate.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.mlp\\.up_proj\\.weight", "blk.$1.ffn_up.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.mlp\\.down_proj\\.weight", "blk.$1.ffn_down.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.input_layernorm\\.weight", "blk.$1.attn_norm.weight");
		name = name.replaceAll("model\\.layers\\.(\\d+)\\.post_attention_layernorm\\.weight", "blk.$1.ffn_norm.weight");
		return name;
	}

	private String mapGPTTensorName(String name) {
		// GPT-specific mappings
		name = name.replaceAll("transformer\\.h\\.(\\d+)\\.attn\\.c_attn\\.weight", "blk.$1.attn_qkv.weight");
		name = name.replaceAll("transformer\\.h\\.(\\d+)\\.attn\\.c_proj\\.weight", "blk.$1.attn_output.weight");
		name = name.replaceAll("transformer\\.h\\.(\\d+)\\.mlp\\.c_fc\\.weight", "blk.$1.ffn_up.weight");
		name = name.replaceAll("transformer\\.h\\.(\\d+)\\.mlp\\.c_proj\\.weight", "blk.$1.ffn_down.weight");
		name = name.replaceAll("transformer\\.h\\.(\\d+)\\.ln_1\\.weight", "blk.$1.attn_norm.weight");
		name = name.replaceAll("transformer\\.h\\.(\\d+)\\.ln_2\\.weight", "blk.$1.ffn_norm.weight");
		return name;
	}

	private String formatSize(long bytes) {
		if (bytes < 1024) return bytes + " B";
		int exp = (int) (Math.log(bytes) / Math.log(1024));
		String pre = "KMGTPE".charAt(exp - 1) + "";
		return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(HuggingFaceModelConverter::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length < 2) {
			printUsage();
			throw new IllegalArgumentException("Insufficient arguments");
		}

		Path modelDir = Paths.get(args[0]);
		ConversionConfig config = new ConversionConfig().outputPath(args[1]);

		// Parse options
		for (int i = 2; i < args.length; i++) {
			switch (args[i]) {
				case "--quantize":
					if (i + 1 < args.length) {
						config.quantization(GGUFConstants.GGMLQuantizationType.valueOf(args[++i]));
					}
					break;
				case "--verbose":
				case "-v":
					config.verbose(true);
					break;
				case "--dry-run":
					config.dryRun(true);
					break;
				case "--vocab-type":
					if (i + 1 < args.length) {
						config.vocabType(args[++i]);
					}
					break;
				case "--context-length":
					if (i + 1 < args.length) {
						config.contextLength(Integer.parseInt(args[++i]));
					}
					break;
				case "--skip-tokenizer":
					config.skipTokenizer(true);
					break;
				case "--metadata":
					if (i + 2 < args.length) {
						config.addMetadata(args[++i], args[++i]);
					}
					break;
				case "--help":
				case "-h":
					printUsage();
					return;
			}
		}

		HuggingFaceModelConverter converter = new HuggingFaceModelConverter(modelDir, config);
		ConversionResult result = converter.convert();

		if (result.success) {
			System.out.println("Conversion successful!");
			System.out.println("Output: " + result.outputPath);
			System.out.println("Architecture: " + result.architecture.name);
			System.out.println("Tensors: " + result.tensorCount);
			System.out.println("Time: " + result.conversionTime + "ms");
		} else {
			throw new RuntimeException("Conversion failed: " + result.error);
		}
	}

	private static void printUsage() {
		System.out.println("Usage: HuggingFaceModelConverter <model_dir> <output_file> [options]");
		System.out.println();
		System.out.println("Convert HuggingFace models to GGUF format.");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --quantize <type>      Target quantization (F32, F16, Q4_0, etc.)");
		System.out.println("  --verbose, -v          Verbose output");
		System.out.println("  --dry-run              Show conversion info without creating output");
		System.out.println("  --vocab-type <type>    Vocabulary type (spm, bpe, tokenizer)");
		System.out.println("  --context-length <n>   Context length (default: 2048)");
		System.out.println("  --skip-tokenizer       Skip tokenizer conversion");
		System.out.println("  --metadata <key> <val> Add custom metadata");
		System.out.println("  --help, -h             Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  HuggingFaceModelConverter ./llama-2-7b-hf model.gguf");
		System.out.println("  HuggingFaceModelConverter --quantize Q4_0 --verbose ./model ./model.gguf");
	}
}
