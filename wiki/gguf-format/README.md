# GGUF Format Implementation

GGUF (GPT-Generated Unified Format) is the standard file format for storing and exchanging machine learning models and adapters in the llama.cpp ecosystem.

## Format Overview

GGUF files contain:
- **Header**: Magic number, version, tensor count, metadata count
- **Metadata**: Key-value pairs describing the model/adapter
- **Tensor Information**: Names, shapes, data types, offsets
- **Tensor Data**: Raw binary tensor data

## File Structure

```
┌─────────────────┐
│ Header          │  Magic (4B) + Version (4B) + Counts (16B)
├─────────────────┤
│ Metadata        │  Key-value pairs (variable length)
├─────────────────┤
│ Tensor Info     │  Tensor descriptions (variable length)
├─────────────────┤
│ Padding         │  Alignment to 32-byte boundary
├─────────────────┤
│ Tensor Data     │  Raw binary data (variable length)
└─────────────────┘
```

## Java Implementation

Our implementation provides a native Java GGUF writer that generates files compatible with llama.cpp:

### Key Classes

- **`GGUFWriter`**: Main writer class for creating GGUF files
- **`GGUFConstants`**: Constants, enums, and metadata keys
- **`GGUFValue`**: Typed value container for metadata
- **`TensorInfo`**: Tensor description and metadata

### Basic Usage

```java
try (GGUFWriter writer = new GGUFWriter(Paths.get("adapter.gguf"), "llama")) {
    // Add metadata
    writer.addType("adapter");
    writer.addString("adapter.type", "lora");
    writer.addFloat32("adapter.lora.alpha", 16.0f);

    // Add tensor information
    writer.addTensorInfo("layer.lora_a",
                        new long[]{8, 4096},           // shape: [rank, input_dim]
                        GGMLQuantizationType.F32,      // data type
                        8 * 4096 * 4);                // size in bytes

    // Write structure
    writer.writeToFile();

    // Write tensor data
    writer.writeTensorData(matrixData);
}
```

## Metadata Specification

### Required Keys for LoRA Adapters

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `general.architecture` | String | Model architecture | `"llama"` |
| `general.type` | String | File type | `"adapter"` |
| `adapter.type` | String | Adapter type | `"lora"` |
| `adapter.lora.alpha` | Float32 | LoRA scaling factor | `16.0` |

### Optional Metadata

| Key | Type | Description |
|-----|------|-------------|
| `general.name` | String | Human-readable name |
| `general.author` | String | Creator information |
| `general.description` | String | Adapter description |
| `general.quantization_version` | UInt32 | Quantization version |

## Data Types

### Supported Value Types

```java
public enum GGUFValueType {
    UINT8(0),    UINT16(2),   UINT32(4),   UINT64(6),
    INT8(1),     INT16(3),    INT32(5),    INT64(7),
    FLOAT32(8),  FLOAT64(9),  BOOL(10),
    STRING(11),  ARRAY(12)
}
```

### Tensor Data Types

```java
public enum GGMLQuantizationType {
    F32(0),   F16(1),   Q4_0(2),  Q4_1(3),
    Q5_0(6),  Q5_1(7),  Q8_0(8),  Q8_1(9),
    Q2_K(10), Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14), Q8_K(15)
}
```

## Binary Format Details

### Header Layout

```c
struct gguf_header {
    uint32_t magic;           // "GGUF" (0x46554747)
    uint32_t version;         // Current version: 3
    uint64_t n_tensors;       // Number of tensors
    uint64_t n_kv;            // Number of metadata entries
};
```

### Endianness

- **Little Endian**: Default and recommended
- **Big Endian**: Supported but not commonly used
- **Detection**: Magic number byte order indicates endianness

### Alignment

- **Default Alignment**: 32 bytes
- **Tensor Data**: Must be aligned to specified boundary
- **Padding**: Filled with zeros

## Compatibility Features

### llama.cpp Integration

Our GGUF writer produces files that are:
- ✅ **Binary Compatible**: Identical format to reference implementation
- ✅ **Metadata Complete**: All required keys for LoRA adapters
- ✅ **Tensor Naming**: Correct naming conventions (`*.lora_a`, `*.lora_b`)
- ✅ **Data Layout**: Proper tensor ordering and alignment

### Verification

```java
// Generate adapter
trainer.saveLoRAAdapter("test_adapter.gguf");

// Load with llama.cpp
long handle = model.loadLoRAAdapter("test_adapter.gguf");
assert handle != 0; // Successfully loaded

model.setLoRAAdapter(handle, 1.0f);
// Adapter is now active and functional
```

## Advanced Features

### Multiple Tensors

```java
// Add multiple LoRA modules
for (LoRAModule module : modules) {
    // A matrix
    writer.addTensorInfo(
        module.getName() + ".lora_a",
        new long[]{module.getRank(), module.getInputDim()},
        GGMLQuantizationType.F32,
        module.getRank() * module.getInputDim() * 4
    );

    // B matrix
    writer.addTensorInfo(
        module.getName() + ".lora_b",
        new long[]{module.getOutputDim(), module.getRank()},
        GGMLQuantizationType.F32,
        module.getOutputDim() * module.getRank() * 4
    );
}
```

### Custom Metadata

```java
// Add custom application metadata
writer.addString("training.dataset", "alpaca-cleaned");
writer.addInt32("training.epochs", 3);
writer.addFloat32("training.learning_rate", 1e-4f);
writer.addArray("training.target_modules", List.of("q_proj", "v_proj"));
```

### Dry Run Mode

```java
// Test without writing files
GGUFWriter dryRunner = new GGUFWriter(path, arch, endianness, true);
dryRunner.addMetadata(...);
dryRunner.addTensorInfo(...);
dryRunner.writeToFile(); // No actual file I/O

// Get size estimates
long estimatedSize = dryRunner.getEstimatedSize();
```

## Error Handling

### Common Issues

1. **Duplicate Keys**: Metadata keys must be unique
2. **Missing Required Keys**: Architecture and type are mandatory
3. **Invalid Tensor Names**: Must follow naming conventions
4. **Alignment Issues**: Tensor data must be properly aligned

### Validation

```java
public void validateAdapter(Path adapterPath) throws IOException {
    // Check file exists and is readable
    if (!Files.isReadable(adapterPath)) {
        throw new IOException("Adapter file not readable: " + adapterPath);
    }

    // Check file size
    long size = Files.size(adapterPath);
    if (size < 1024) { // Minimum reasonable size
        throw new IOException("Adapter file too small: " + size + " bytes");
    }

    // Attempt to load with llama.cpp
    try {
        long handle = model.loadLoRAAdapter(adapterPath.toString());
        if (handle == 0) {
            throw new IOException("Failed to load adapter with llama.cpp");
        }
        model.freeLoRAAdapter(handle);
    } catch (Exception e) {
        throw new IOException("Adapter validation failed", e);
    }
}
```

## Performance Considerations

### Writing Performance

- **Buffered I/O**: Uses `BufferedOutputStream` for efficiency
- **Batch Writes**: Tensor data written in large chunks
- **Memory Usage**: Minimal memory overhead during writing

### File Sizes

Typical LoRA adapter sizes:
- **Rank 4**: ~1-5 MB
- **Rank 8**: ~5-20 MB
- **Rank 16**: ~20-80 MB
- **Rank 32**: ~80-320 MB

### Loading Performance

- **Memory Mapping**: llama.cpp uses mmap for efficient loading
- **Lazy Loading**: Tensors loaded on demand
- **Cache Friendly**: Aligned data layout for optimal access

## Debugging Tools

### Hex Dump Analysis

```bash
# Check GGUF magic and header
hexdump -C adapter.gguf | head -5

# Expected output:
# 00000000  47 47 55 46 03 00 00 00  02 00 00 00 00 00 00 00  |GGUF............|
#           ^^^^^^^^^^^ magic="GGUF" version=3
#                                     ^^^^^^^^^^^ tensor_count=2
```

### Metadata Inspection

```java
public void dumpMetadata(String ggufFile) {
    // Implementation would read and display all metadata
    // This is a conceptual example - actual implementation would require
    // a GGUF reader (not included in this project)
}
```

## Related Documentation

- [Writer Implementation](writer-implementation.md) - Detailed implementation guide
- [LoRA Integration](../lora-training/README.md) - How GGUF is used in LoRA training
- [API Reference](../api/gguf-api.md) - Complete API documentation

## Standards Compliance

- **GGUF Version**: 3 (current standard)
- **Magic Number**: `0x46554747` ("GGUF")
- **Metadata Format**: Compatible with gguf-py reference implementation
- **Tensor Layout**: Matches llama.cpp expectations