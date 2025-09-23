package de.kherud.llama.gguf;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertNotNull;

/**
 * Test cases for GGUFInspector utility.
 */
public class GGUFInspectorTest {

	@Rule
	public TemporaryFolder tempDir = new TemporaryFolder();

	private Path testGGUF;

	@Before
	public void setUp() throws IOException {
		testGGUF = tempDir.getRoot().toPath().resolve("test.gguf");
		createTestGGUF(testGGUF);
	}

	@Test
	public void testInspectorCreation() throws IOException {
		try (GGUFInspector inspector = new GGUFInspector(testGGUF)) {
			assertNotNull(inspector);
		}
	}

	@Test
	public void testInspection() throws IOException {
		try (GGUFInspector inspector = new GGUFInspector(testGGUF)) {
			GGUFInspector.InspectionResult result = inspector.inspect();
			assertNotNull(result);
		}
	}

	/**
	 * Create a minimal test GGUF file
	 */
	private void createTestGGUF(Path path) throws IOException {
		// Create a minimal GGUF file
		byte[] testData = {
			'G', 'G', 'U', 'F', // Magic
			0x03, 0x00, 0x00, 0x00, // Version 3 (now supported)
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Tensor count
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Metadata count
		};

		Files.write(path, testData);
	}
}
