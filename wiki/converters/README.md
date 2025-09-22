# Model Converters

This directory contains Java implementations for converting models and adapters from various formats to GGUF, eliminating the need for Python dependencies.

## Available Converters

### HuggingFaceToGGUFConverter

Converts HuggingFace models directly to GGUF format.

**Features:**
- Supports SafeTensors format (.safetensors)
- Automatic architecture detection (Llama, Mistral, GPT-2, BERT, Falcon)
- Multi-file sharded models
- Quantization during conversion
- Vocabulary and tokenizer conversion
- Metadata preservation

**Usage:**
```java
Path modelPath = Paths.get("/path/to/huggingface/model");
Path outputPath = Paths.get("output.gguf");

HuggingFaceToGGUFConverter.ConversionConfig config =
    new HuggingFaceToGGUFConverter.ConversionConfig()
        .quantize(GGUFConstants.GGMLQuantizationType.Q4_0)
        .threads(8)
        .verbose(true);

HuggingFaceToGGUFConverter converter =
    new HuggingFaceToGGUFConverter(modelPath, outputPath, config);
converter.convert();
```

**Command Line:**
```bash
java de.kherud.llama.converters.HuggingFaceToGGUFConverter \
    /path/to/model output.gguf \
    --quantize Q4_0 --threads 8 --verbose
```

### LoRAToGGUFConverter

Converts PyTorch LoRA adapters to GGUF format compatible with `loadLoRAAdapter()`.

**Features:**
- PEFT format support (adapter_model.safetensors, adapter_config.json)
- Individual LoRA weight files
- Layer normalization merging
- Automatic tensor name mapping
- Multiple architecture support

**Usage:**
```java
Path adapterPath = Paths.get("/path/to/lora/adapter");
Path outputPath = Paths.get("adapter.gguf");

LoRAToGGUFConverter.ConversionConfig config =
    new LoRAToGGUFConverter.ConversionConfig()
        .baseModelArchitecture("llama")
        .mergeLayerNorms(true)
        .verbose(true);

LoRAToGGUFConverter converter =
    new LoRAToGGUFConverter(adapterPath, outputPath, config);
converter.convert();
```

**Command Line:**
```bash
java de.kherud.llama.converters.LoRAToGGUFConverter \
    /path/to/adapter adapter.gguf \
    --arch llama --merge-norms --verbose
```

## Format Support

### Input Formats

| Format | HuggingFace Converter | LoRA Converter |
|--------|----------------------|----------------|
| SafeTensors (.safetensors) | ✅ | ✅ |
| PyTorch (.bin, .pth) | ❌ | ❌ |
| Individual files | N/A | ✅ |

**Note:** PyTorch .bin files are not supported. Convert to SafeTensors first:
```python
from safetensors.torch import save_file
import torch
model = torch.load('model.bin')
save_file(model, 'model.safetensors')
```

### Supported Architectures

| Architecture | HuggingFace | LoRA | GGUF Mapping |
|-------------|-------------|------|--------------|
| Llama/Llama2 | ✅ | ✅ | llama |
| Mistral | ✅ | ✅ | llama |
| GPT-2 | ✅ | ✅ | gpt2 |
| BERT | ✅ | ✅ | bert |
| Falcon | ✅ | ✅ | falcon |

## Integration with Training

The converters work seamlessly with the existing LoRA training system:

1. **Create LoRA adapter** using `LoRATrainer.java`
2. **Import external adapters** using `LoRAToGGUFConverter`
3. **Load and use** with `model.loadLoRAAdapter()`

Example workflow:
```java
// Convert external LoRA
LoRAToGGUFConverter converter = new LoRAToGGUFConverter(
    Paths.get("external_adapter"), Paths.get("converted.gguf"));
converter.convert();

// Load into model
LlamaModel model = new LlamaModel(params);
model.loadLoRAAdapter("converted.gguf");

// Use for inference
String response = model.complete("Hello");
```

## Tensor Name Mapping

The converters automatically map tensor names between formats:

### HuggingFace → GGUF (Llama)
- `model.embed_tokens.weight` → `token_embd.weight`
- `model.layers.{N}.self_attn.q_proj.weight` → `blk.{N}.attn_q.weight`
- `model.layers.{N}.mlp.gate_proj.weight` → `blk.{N}.ffn_gate.weight`
- `lm_head.weight` → `output.weight`

### LoRA → GGUF
- `q_proj.lora_A.weight` → `blk.{N}.attn_q.lora_a`
- `q_proj.lora_B.weight` → `blk.{N}.attn_q.lora_b`

## Performance Tips

1. **Use SafeTensors** for faster loading and better error handling
2. **Enable quantization** during conversion to reduce file size
3. **Use multiple threads** for large models
4. **Monitor memory usage** for very large models

## Error Handling

Common issues and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `config.json not found` | Missing model config | Ensure complete HuggingFace model directory |
| `No LoRA adapter files found` | Empty adapter directory | Check for .safetensors or .bin files |
| `Shape mismatch` | Incompatible LoRA tensors | Verify LoRA rank and target modules |
| `PyTorch .bin file` | Unsupported format | Convert to SafeTensors first |

## Advanced Usage

### Custom Tensor Mapping
```java
LoRAToGGUFConverter.ConversionConfig config =
    new LoRAToGGUFConverter.ConversionConfig()
        .addTargetModule("custom_layer", "blk.{N}.custom");
```

### Metadata Override
```java
HuggingFaceToGGUFConverter.ConversionConfig config =
    new HuggingFaceToGGUFConverter.ConversionConfig()
        .addMetadata("general.description", "Custom model");
```

### Quantization Options
```java
config.quantize(GGUFConstants.GGMLQuantizationType.Q4_K_M); // Mixed precision
config.quantize(GGUFConstants.GGMLQuantizationType.Q8_0);   // 8-bit quantization
config.quantize(GGUFConstants.GGMLQuantizationType.F16);    // Half precision
```