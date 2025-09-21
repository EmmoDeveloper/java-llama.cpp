package de.kherud.llama;

import de.kherud.llama.gguf.GGUFConstants;
import de.kherud.llama.gguf.GGUFReader;
import de.kherud.llama.gguf.GGUFWriter;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import static org.junit.Assert.*;

public class GGUFReaderTest {

	private Path testFile;

	@Before
	public void setUp() throws IOException {
		testFile = Files.createTempFile("test_adapter", ".gguf");
	}

	@After
	public void tearDown() throws IOException {
		if (Files.exists(testFile)) {
			Files.delete(testFile);
		}
	}

	@Test
	public void testReadWriteLoRAAdapter() throws IOException {
		// Create a test LoRA adapter file
		try (GGUFWriter writer = new GGUFWriter(testFile, "llama")) {
			// Add required metadata
			writer.addType("adapter");
			writer.addString(GGUFConstants.Keys.Adapter.TYPE, "lora");
			writer.addLoRAAlpha(16.0f);
			writer.addString("general.name", "test_adapter");
			writer.addString("general.description", "Test LoRA adapter for validation");

			// Add tensor information
			writer.addTensorInfo("layer.0.attention.wq.lora_a",
				new long[]{8, 4096},
				GGUFConstants.GGMLQuantizationType.F32,
				8 * 4096 * 4);

			writer.addTensorInfo("layer.0.attention.wq.lora_b",
				new long[]{4096, 8},
				GGUFConstants.GGMLQuantizationType.F32,
				4096 * 8 * 4);

			// Write structure
			writer.writeToFile();

			// Write tensor data
			float[][] tensorDataA = new float[8][4096];
			float[][] tensorDataB = new float[4096][8];

			// Fill with test pattern
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 4096; j++) {
					tensorDataA[i][j] = 0.1f; // A matrix initialized with small values
				}
			}

			for (int i = 0; i < 4096; i++) {
				for (int j = 0; j < 8; j++) {
					tensorDataB[i][j] = 0.0f; // B matrix starts at zero
				}
			}

			writer.writeTensorData(tensorDataA);
			writer.writeTensorData(tensorDataB);
		}

		// Now read it back and validate
		try (GGUFReader reader = new GGUFReader(testFile)) {
			// Validate metadata
			assertEquals(2, reader.getTensorCount());
			assertTrue(reader.getFieldCount() >= 4);

			// Check general fields
			GGUFReader.GGUFField typeField = reader.getField("general.type");
			assertNotNull("general.type field should exist", typeField);
			assertEquals("adapter", typeField.value);

			GGUFReader.GGUFField adapterTypeField = reader.getField(GGUFConstants.Keys.Adapter.TYPE);
			assertNotNull("adapter.type field should exist", adapterTypeField);
			assertEquals("lora", adapterTypeField.value);

			GGUFReader.GGUFField alphaField = reader.getField(GGUFConstants.Keys.Adapter.LORA_ALPHA);
			assertNotNull("adapter.lora.alpha field should exist", alphaField);
			assertEquals(16.0f, (Float) alphaField.value, 0.001f);

			GGUFReader.GGUFField nameField = reader.getField("general.name");
			assertNotNull("general.name field should exist", nameField);
			assertEquals("test_adapter", nameField.value);

			// Check tensors
			GGUFReader.GGUFTensor tensorA = reader.getTensor("layer.0.attention.wq.lora_a");
			assertNotNull("Tensor A should exist", tensorA);
			assertEquals("layer.0.attention.wq.lora_a", tensorA.name);
			assertArrayEquals(new long[]{4096, 8}, tensorA.shape);
			assertEquals(GGUFConstants.GGMLQuantizationType.F32, tensorA.type);
			assertEquals(8 * 4096, tensorA.nElements);
			assertEquals(8 * 4096 * 4, tensorA.nBytes);
			assertNotNull("Tensor A data should be loaded", tensorA.data);
			assertEquals(8 * 4096 * 4, tensorA.data.length);

			GGUFReader.GGUFTensor tensorB = reader.getTensor("layer.0.attention.wq.lora_b");
			assertNotNull("Tensor B should exist", tensorB);
			assertEquals("layer.0.attention.wq.lora_b", tensorB.name);
			assertArrayEquals(new long[]{8, 4096}, tensorB.shape);
			assertEquals(GGUFConstants.GGMLQuantizationType.F32, tensorB.type);
			assertEquals(4096 * 8, tensorB.nElements);
			assertEquals(4096 * 8 * 4, tensorB.nBytes);
			assertNotNull("Tensor B data should be loaded", tensorB.data);
			assertEquals(4096 * 8 * 4, tensorB.data.length);

			// Test getAllFields and getAllTensors
			Map<String, GGUFReader.GGUFField> allFields = reader.getFields();
			assertTrue("Should have at least 4 fields", allFields.size() >= 4);
			assertTrue("Should contain general.type", allFields.containsKey("general.type"));

			assertEquals(2, reader.getTensors().size());
		}
	}

	@Test
	public void testInvalidFile() {
		Path invalidFile = Paths.get("nonexistent.gguf");

		try (GGUFReader reader = new GGUFReader(invalidFile)) {
			fail("Should throw IOException for nonexistent file");
		} catch (IOException e) {
			// Expected
		}
	}

	@Test
	public void testCorruptedFile() throws IOException {
		// Create a file with invalid magic
		try (GGUFWriter writer = new GGUFWriter(testFile, "llama")) {
			// Don't write anything, just close to create empty file
		}

		// Overwrite with invalid data
		Files.write(testFile, new byte[]{1, 2, 3, 4});

		try (GGUFReader reader = new GGUFReader(testFile)) {
			fail("Should throw IOException for corrupted file");
		} catch (IOException e) {
			assertTrue("Should mention invalid magic", e.getMessage().contains("Invalid GGUF magic"));
		}
	}

	@Test
	public void testFileInspection() throws IOException {
		// Create minimal adapter
		try (GGUFWriter writer = new GGUFWriter(testFile, "llama")) {
			writer.addType("adapter");
			writer.addString(GGUFConstants.Keys.Adapter.TYPE, "lora");
			writer.addLoRAAlpha(8.0f);
			writer.writeToFile();
		}

		try (GGUFReader reader = new GGUFReader(testFile)) {
			// Test inspection methods
			System.out.println("File inspection:");
			System.out.println("Tensor count: " + reader.getTensorCount());
			System.out.println("Field count: " + reader.getFieldCount());
			System.out.println("Byte order: " + reader.getByteOrder());
			System.out.println("Alignment: " + reader.getAlignment());
			System.out.println("Data offset: " + reader.getDataOffset());

			System.out.println("\nFields:");
			for (Map.Entry<String, GGUFReader.GGUFField> entry : reader.getFields().entrySet()) {
				System.out.println("  " + entry.getKey() + ": " + entry.getValue());
			}

			System.out.println("\nTensors:");
			for (GGUFReader.GGUFTensor tensor : reader.getTensors()) {
				System.out.println("  " + tensor);
			}

			// Validate basic structure
			assertEquals(0, reader.getTensorCount());
			assertTrue(reader.getFieldCount() >= 3);
		}
	}
}