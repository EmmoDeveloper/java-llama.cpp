# Conversion API Reference

This document provides detailed API reference for the model and adapter conversion tools.

## HuggingFaceToGGUFConverter

### Constructor
```java
public HuggingFaceToGGUFConverter(Path modelPath, Path outputPath)
public HuggingFaceToGGUFConverter(Path modelPath, Path outputPath, ConversionConfig config)
```

### ConversionConfig
Builder pattern configuration for HuggingFace model conversion.

```java
ConversionConfig config = new ConversionConfig()
    .quantize(GGUFConstants.GGMLQuantizationType.Q4_0)
    .threads(8)
    .verbose(true)
    .addMetadata("custom.key", "value");
```

#### Methods
- `quantize(GGMLQuantizationType type)` - Set quantization type
- `threads(int count)` - Number of conversion threads
- `verbose(boolean verbose)` - Enable verbose logging
- `addMetadata(String key, String value)` - Add custom metadata

#### Supported Quantization Types
- `F32` - 32-bit float (no quantization)
- `F16` - 16-bit float
- `Q4_0`, `Q4_1` - 4-bit quantization
- `Q5_0`, `Q5_1` - 5-bit quantization
- `Q8_0` - 8-bit quantization
- `Q4_K_M`, `Q4_K_S` - Mixed 4-bit quantization
- `Q5_K_M`, `Q5_K_S` - Mixed 5-bit quantization

### Methods
```java
public void convert() throws IOException
```
Performs the model conversion. Throws `IOException` on conversion errors.

### Architecture Detection
Automatically detects and maps HuggingFace architectures:

| HuggingFace | GGUF Architecture |
|-------------|------------------|
| LlamaForCausalLM | llama |
| MistralForCausalLM | llama |
| GPT2LMHeadModel | gpt2 |
| BertModel | bert |
| FalconForCausalLM | falcon |

### Tensor Mapping (Llama Architecture)
```java
// Embeddings
"model.embed_tokens.weight" → "token_embd.weight"

// Attention
"model.layers.{N}.self_attn.q_proj.weight" → "blk.{N}.attn_q.weight"
"model.layers.{N}.self_attn.k_proj.weight" → "blk.{N}.attn_k.weight"
"model.layers.{N}.self_attn.v_proj.weight" → "blk.{N}.attn_v.weight"
"model.layers.{N}.self_attn.o_proj.weight" → "blk.{N}.attn_output.weight"

// Feed Forward
"model.layers.{N}.mlp.gate_proj.weight" → "blk.{N}.ffn_gate.weight"
"model.layers.{N}.mlp.up_proj.weight" → "blk.{N}.ffn_up.weight"
"model.layers.{N}.mlp.down_proj.weight" → "blk.{N}.ffn_down.weight"

// Normalization
"model.layers.{N}.input_layernorm.weight" → "blk.{N}.attn_norm.weight"
"model.layers.{N}.post_attention_layernorm.weight" → "blk.{N}.ffn_norm.weight"
"model.norm.weight" → "output_norm.weight"

// Output
"lm_head.weight" → "output.weight"
```

## LoRAToGGUFConverter

### Constructor
```java
public LoRAToGGUFConverter(Path adapterPath, Path outputPath)
public LoRAToGGUFConverter(Path adapterPath, Path outputPath, ConversionConfig config)
```

### ConversionConfig
Builder pattern configuration for LoRA adapter conversion.

```java
ConversionConfig config = new ConversionConfig()
    .verbose(true)
    .mergeLayerNorms(true)
    .baseModelArchitecture("llama")
    .addTargetModule("custom_proj", "blk.{N}.custom");
```

#### Methods
- `verbose(boolean verbose)` - Enable verbose logging
- `mergeLayerNorms(boolean merge)` - Include layer normalization weights
- `baseModelArchitecture(String arch)` - Set base model architecture
- `addTargetModule(String source, String target)` - Custom tensor name mapping

### Methods
```java
public void convert() throws IOException
```
Performs the LoRA adapter conversion.

### Supported Input Formats

#### PEFT Format
```
adapter_directory/
├── adapter_config.json    # LoRA configuration
├── adapter_model.safetensors  # LoRA weights
└── tokenizer_config.json  # Optional tokenizer config
```

#### Individual Files
```
adapter_directory/
├── q_proj.lora.safetensors
├── v_proj.lora.safetensors
└── adapter_config.json
```

### Configuration File Format
```json
{
  "peft_type": "LORA",
  "lora_alpha": 16.0,
  "r": 8,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

### LoRA Tensor Processing
The converter processes tensor pairs:
- `*.lora_A.weight` - Low-rank matrix A
- `*.lora_B.weight` - Low-rank matrix B
- Optional layer normalization weights

### Output Format
Creates GGUF files compatible with `loadLoRAAdapter()`:
```java
// Generated tensor names
"blk.{N}.attn_q.lora_a"
"blk.{N}.attn_q.lora_b"
```

## Error Handling

### Common Exceptions

#### HuggingFaceToGGUFConverter
```java
// Missing configuration
IOException: "config.json not found in model directory"

// Unsupported format
UnsupportedOperationException: "PyTorch .bin file conversion not yet implemented"

// Invalid model structure
IOException: "No model files found (looked for .safetensors, .bin, .pth)"
```

#### LoRAToGGUFConverter
```java
// Missing adapters
IOException: "No LoRA adapter files found"

// Shape mismatch
IOException: "Shape mismatch for tensor_name: A=[...], B=[...]"

// Incomplete tensors
IllegalStateException: "Incomplete LoRA tensor: has A: true, has B: false"
```

### Error Recovery
```java
try {
    converter.convert();
} catch (UnsupportedOperationException e) {
    if (e.getMessage().contains("PyTorch .bin")) {
        // Suggest SafeTensors conversion
        logger.warn("Convert to SafeTensors format first");
    }
} catch (IOException e) {
    // Handle file system errors
    logger.error("Conversion failed: " + e.getMessage());
}
```

## Integration Examples

### Convert HuggingFace Model
```java
public static void convertHuggingFaceModel(String modelDir, String outputFile) {
    try {
        HuggingFaceToGGUFConverter.ConversionConfig config =
            new HuggingFaceToGGUFConverter.ConversionConfig()
                .quantize(GGUFConstants.GGMLQuantizationType.Q4_K_M)
                .threads(Runtime.getRuntime().availableProcessors())
                .verbose(true);

        HuggingFaceToGGUFConverter converter = new HuggingFaceToGGUFConverter(
            Paths.get(modelDir), Paths.get(outputFile), config);

        converter.convert();
        System.out.println("Model converted successfully: " + outputFile);
    } catch (Exception e) {
        System.err.println("Conversion failed: " + e.getMessage());
    }
}
```

### Convert LoRA Adapter
```java
public static void convertLoRAAdapter(String adapterDir, String outputFile) {
    try {
        LoRAToGGUFConverter.ConversionConfig config =
            new LoRAToGGUFConverter.ConversionConfig()
                .baseModelArchitecture("llama")
                .mergeLayerNorms(true)
                .verbose(true);

        LoRAToGGUFConverter converter = new LoRAToGGUFConverter(
            Paths.get(adapterDir), Paths.get(outputFile), config);

        converter.convert();
        System.out.println("Adapter converted successfully: " + outputFile);
    } catch (Exception e) {
        System.err.println("Conversion failed: " + e.getMessage());
    }
}
```

### Use Converted Models
```java
// Load converted model
ModelParameters params = new ModelParameters()
    .setModel("converted_model.gguf")
    .setCtxSize(2048);

try (LlamaModel model = new LlamaModel(params)) {
    // Load converted adapter
    model.loadLoRAAdapter("converted_adapter.gguf");

    // Use for inference
    String response = model.complete("Hello, how are you?");
    System.out.println(response);
}
```

## Performance Considerations

### Memory Usage
- Large models require significant RAM during conversion
- Use quantization to reduce memory requirements
- Process sharded models efficiently

### Threading
- Optimal thread count: `Runtime.getRuntime().availableProcessors()`
- I/O bound operations benefit from fewer threads
- CPU-intensive quantization benefits from more threads

### Storage
- SafeTensors format is faster and safer than PyTorch .bin
- Quantized models are significantly smaller
- Temporary files may be created during conversion

### Optimization Tips
```java
// Optimal configuration for large models
ConversionConfig config = new ConversionConfig()
    .quantize(GGMLQuantizationType.Q4_K_M)  // Good compression/quality balance
    .threads(Math.min(8, Runtime.getRuntime().availableProcessors()))
    .verbose(false);  // Reduce logging overhead
```