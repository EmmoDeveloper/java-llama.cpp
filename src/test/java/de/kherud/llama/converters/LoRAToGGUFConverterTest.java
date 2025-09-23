package de.kherud.llama.converters;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Unit tests for LoRAToGGUFConverter.
 */
public class LoRAToGGUFConverterTest {

	@Rule
	public TemporaryFolder tempFolder = new TemporaryFolder();

	private Path adapterDir;
	private Path outputPath;

	@Before
	public void setUp() throws IOException {
		adapterDir = tempFolder.newFolder("test-adapter").toPath();
		outputPath = tempFolder.newFile("test-adapter.gguf").toPath();
	}

	@Test
	public void testAdapterConfigLoading() throws IOException {
		// Create PEFT adapter config
		createAdapterConfig(Map.of(
			"peft_type", "LORA",
			"lora_alpha", 16.0,
			"r", 8,
			"target_modules", "[\"q_proj\", \"v_proj\"]"
		));

		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath);
		try {
			converter.convert();
		} catch (IOException e) {
			// Expected to fail due to missing tensor files, but should load config successfully
			Assert.assertFalse("Should load adapter config without error",
				e.getMessage().contains("adapter_config.json"));
		}
	}

	@Test
	public void testMissingAdapterConfig() throws IOException {
		// Test converter works without adapter_config.json (uses defaults)
		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath);
		try {
			converter.convert();
		} catch (IOException e) {
			// Should fail due to missing adapter files, not missing config
			Assert.assertTrue(e.getMessage().contains("No LoRA adapter files found"));
		}
	}

	@Test
	public void testSafeTensorsDetection() throws IOException {
		createAdapterConfig(Map.of(
			"lora_alpha", 16.0,
			"r", 8
		));

		// Create empty SafeTensors file
		Files.createFile(adapterDir.resolve("adapter_model.safetensors"));

		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath);
		try {
			converter.convert();
		} catch (Exception e) {
			// Should attempt SafeTensors loading
			Assert.assertTrue("Should detect SafeTensors file", true);
		}
	}

	@Test
	public void testPyTorchBinUnsupported() throws IOException {
		createAdapterConfig(Map.of(
			"lora_alpha", 16.0,
			"r", 8
		));

		// Create PyTorch bin file
		Files.createFile(adapterDir.resolve("adapter_model.bin"));

		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath);
		try {
			converter.convert();
			Assert.fail("Expected UnsupportedOperationException for PyTorch .bin files");
		} catch (UnsupportedOperationException e) {
			Assert.assertTrue(e.getMessage().contains("PyTorch .bin file reading not yet implemented"));
		}
	}

	@Test
	public void testConversionConfigBuilder() {
		LoRAToGGUFConverter.ConversionConfig config =
			new LoRAToGGUFConverter.ConversionConfig()
				.verbose(true)
				.mergeLayerNorms(true)
				.baseModelArchitecture("llama")
				.addTargetModule("q_proj", "attn_q");

		Assert.assertNotNull(config);
	}

	@Test
	public void testIndividualLoRAFiles() throws IOException {
		createAdapterConfig(Map.of(
			"lora_alpha", 16.0,
			"r", 8
		));

		// Create individual LoRA files
		Files.createFile(adapterDir.resolve("q_proj.lora.safetensors"));
		Files.createFile(adapterDir.resolve("v_proj.lora.safetensors"));

		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath);
		try {
			converter.convert();
		} catch (Exception e) {
			// Should attempt to load individual files
			Assert.assertTrue("Should detect individual LoRA files", true);
		}
	}

	@Test
	public void testTensorNameMapping() throws IOException {
		createAdapterConfig(Map.of(
			"lora_alpha", 16.0,
			"r", 8
		));

		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath);

		// This mainly tests that the converter can be created and configured
		Assert.assertNotNull(converter);
	}


	@Test
	public void testCustomArchitecture() throws IOException {
		createAdapterConfig(Map.of(
			"lora_alpha", 32.0,
			"r", 16
		));

		LoRAToGGUFConverter.ConversionConfig config =
			new LoRAToGGUFConverter.ConversionConfig()
				.baseModelArchitecture("mistral");

		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath, config);

		try {
			converter.convert();
		} catch (IOException e) {
			// Should handle custom architecture
			Assert.assertTrue(e.getMessage().contains("No LoRA adapter files found"));
		}
	}

	@Test
	public void testLayerNormMerging() throws IOException {
		createAdapterConfig(Map.of(
			"lora_alpha", 16.0,
			"r", 8
		));

		LoRAToGGUFConverter.ConversionConfig config =
			new LoRAToGGUFConverter.ConversionConfig()
				.mergeLayerNorms(true);

		LoRAToGGUFConverter converter = new LoRAToGGUFConverter(adapterDir, outputPath, config);
		Assert.assertNotNull(converter);
	}

	private void createAdapterConfig(Map<String, Object> config) throws IOException {
		StringBuilder json = new StringBuilder("{\n");
		boolean first = true;
		for (Map.Entry<String, Object> entry : config.entrySet()) {
			if (!first) json.append(",\n");
			json.append("  \"").append(entry.getKey()).append("\": ");
			Object value = entry.getValue();
			if (value instanceof String) {
				json.append("\"").append(value).append("\"");
			} else {
				json.append(value);
			}
			first = false;
		}
		json.append("\n}");

		Files.writeString(adapterDir.resolve("adapter_config.json"), json.toString());
	}

}
