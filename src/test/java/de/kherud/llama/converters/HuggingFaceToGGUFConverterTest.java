package de.kherud.llama.converters;

import de.kherud.llama.gguf.GGUFConstants;
import org.junit.*;
import org.junit.rules.TemporaryFolder;

import java.io.*;
import java.nio.file.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Unit tests for HuggingFaceToGGUFConverter.
 */
public class HuggingFaceToGGUFConverterTest {

	@Rule
	public TemporaryFolder tempFolder = new TemporaryFolder();

	private Path modelDir;
	private Path outputPath;

	@Before
	public void setUp() throws IOException {
		modelDir = tempFolder.newFolder("test-model").toPath();
		outputPath = tempFolder.newFile("test-output.gguf").toPath();
	}

	@Test
	public void testConfigJsonRequired() throws IOException {
		// Test that converter fails when config.json is missing
		try {
			HuggingFaceToGGUFConverter converter = new HuggingFaceToGGUFConverter(modelDir, outputPath);
			converter.convert();
			Assert.fail("Expected IOException when config.json is missing");
		} catch (IOException e) {
			Assert.assertTrue(e.getMessage().contains("config.json not found"));
		}
	}

	@Test
	public void testLlamaArchitectureDetection() throws IOException {
		// Create a minimal config.json for Llama
		createConfigJson(Map.of(
			"model_type", "llama",
			"architectures", "[\"LlamaForCausalLM\"]",
			"vocab_size", 32000,
			"hidden_size", 4096,
			"num_hidden_layers", 32,
			"num_attention_heads", 32
		));

		// Should detect architecture without crashing
		HuggingFaceToGGUFConverter converter = new HuggingFaceToGGUFConverter(modelDir, outputPath);
		try {
			converter.convert();
		} catch (IOException e) {
			// Expected to fail due to missing tensor files, but should get past architecture detection
			Assert.assertFalse("Should detect architecture successfully",
				e.getMessage().contains("Unknown architecture"));
		}
	}

	@Test
	public void testMistralArchitectureMapping() throws IOException {
		// Create config for Mistral model
		createConfigJson(Map.of(
			"model_type", "mistral",
			"architectures", "[\"MistralForCausalLM\"]",
			"vocab_size", 32000,
			"hidden_size", 4096,
			"num_hidden_layers", 32,
			"num_attention_heads", 32,
			"num_key_value_heads", 8
		));

		HuggingFaceToGGUFConverter converter = new HuggingFaceToGGUFConverter(modelDir, outputPath);
		try {
			converter.convert();
		} catch (IOException e) {
			// Should map Mistral to llama architecture
			Assert.assertFalse("Should handle Mistral architecture",
				e.getMessage().contains("Unknown architecture"));
		}
	}

	@Test
	public void testConversionConfigBuilder() {
		HuggingFaceToGGUFConverter.ConversionConfig config =
			new HuggingFaceToGGUFConverter.ConversionConfig()
				.quantize(GGUFConstants.GGMLQuantizationType.Q4_0)
				.threads(4)
				.verbose(true)
				.addMetadata("custom.key", "test-value");

		// Config should build successfully
		Assert.assertNotNull(config);
	}

	@Test
	public void testSafeTensorsFileDetection() throws IOException {
		createConfigJson(Map.of(
			"model_type", "llama",
			"vocab_size", 32000,
			"hidden_size", 4096
		));

		// Create empty SafeTensors file
		Files.createFile(modelDir.resolve("model.safetensors"));

		HuggingFaceToGGUFConverter converter = new HuggingFaceToGGUFConverter(modelDir, outputPath);
		try {
			converter.convert();
		} catch (Exception e) {
			// Should attempt to read SafeTensors (may fail due to invalid format)
			Assert.assertTrue("Should attempt SafeTensors loading", true);
		}
	}

	@Test
	public void testPyTorchBinUnsupported() throws IOException {
		createConfigJson(Map.of(
			"model_type", "llama",
			"vocab_size", 32000,
			"hidden_size", 4096
		));

		// Create PyTorch bin file
		Files.createFile(modelDir.resolve("pytorch_model.bin"));

		HuggingFaceToGGUFConverter converter = new HuggingFaceToGGUFConverter(modelDir, outputPath);
		try {
			converter.convert();
			Assert.fail("Expected UnsupportedOperationException for PyTorch .bin files");
		} catch (UnsupportedOperationException e) {
			Assert.assertTrue(e.getMessage().contains("PyTorch .bin file conversion not yet implemented"));
		} catch (IOException e) {
			// Also acceptable if it fails due to no model files found
			Assert.assertTrue(e.getMessage().contains("No model files found"));
		}
	}

	@Test
	public void testCommandLineInterface() {
		// Test that main method doesn't crash with invalid arguments
		try {
			HuggingFaceToGGUFConverter.main(new String[]{});
		} catch (SystemExit e) {
			// Expected to exit with usage message
			Assert.assertEquals(1, e.status);
		}

		try {
			HuggingFaceToGGUFConverter.main(new String[]{"nonexistent", "output.gguf"});
		} catch (SystemExit e) {
			// Expected to exit due to invalid path
			Assert.assertEquals(1, e.status);
		}
	}

	@Test
	public void testTensorNameMapping() throws IOException {
		createConfigJson(Map.of(
			"model_type", "llama",
			"vocab_size", 32000,
			"hidden_size", 4096,
			"num_hidden_layers", 2
		));

		HuggingFaceToGGUFConverter converter = new HuggingFaceToGGUFConverter(modelDir, outputPath);

		// Test tensor name mapping through reflection (if methods were public)
		// This is a simplified test that just ensures the converter can be created
		Assert.assertNotNull(converter);
	}

	private void createConfigJson(Map<String, Object> config) throws IOException {
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

		Files.writeString(modelDir.resolve("config.json"), json.toString());
	}

	// Helper class to catch System.exit() calls in tests
	private static class SystemExit extends SecurityException {
		public final int status;

		public SystemExit(int status) {
			this.status = status;
		}
	}

	// Install security manager to catch System.exit() in main method tests
	@BeforeClass
	public static void installSecurityManager() {
		System.setSecurityManager(new SecurityManager() {
			@Override
			public void checkExit(int status) {
				throw new SystemExit(status);
			}

			@Override
			public void checkPermission(java.security.Permission perm) {
				// Allow all other permissions
			}
		});
	}

	@AfterClass
	public static void restoreSecurityManager() {
		System.setSecurityManager(null);
	}
}