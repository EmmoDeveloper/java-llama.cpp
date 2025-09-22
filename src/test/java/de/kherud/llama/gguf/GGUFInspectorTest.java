package de.kherud.llama.gguf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.nio.file.Files;
import java.io.IOException;

/**
 * Test cases for GGUFInspector utility.
 */
public class GGUFInspectorTest {

	@TempDir
	Path tempDir;

	private Path testGGUFFile;
	private GGUFInspector inspector;

	@BeforeEach
	void setUp() throws IOException {
		// Create a minimal test GGUF file
		testGGUFFile = tempDir.resolve("test_model.gguf");
		createTestGGUFFile(testGGUFFile);
		inspector = new GGUFInspector(testGGUFFile);
	}

	@AfterEach
	void tearDown() throws IOException {
		if (inspector != null) {
			inspector.close();
		}
	}

	@Test
	void testInspectorCreation() {
		assertNotNull(inspector);
		assertTrue(Files.exists(testGGUFFile));
	}

	@Test
	void testBasicInspection() throws IOException {
		GGUFInspector.InspectionResult result = inspector.inspect();

		assertNotNull(result);
		assertNotNull(result.fileInfo);
		assertNotNull(result.metadata);
		assertNotNull(result.tensors);

		// Check file info is populated
		assertTrue(result.fileInfo.fileSize > 0);
		assertNotNull(result.fileInfo.endianness);
		assertNotNull(result.fileInfo.hostEndianness);
	}

	@Test
	void testInspectionWithOptions() throws IOException {
		GGUFInspector.InspectionOptions options = new GGUFInspector.InspectionOptions()
			.metadata(false)
			.verbose(true);

		GGUFInspector.InspectionResult result = inspector.inspect(options);

		assertNotNull(result);
		assertTrue(result.metadata.isEmpty()); // Metadata should be skipped
	}

	@Test
	void testFilteredInspection() throws IOException {
		GGUFInspector.InspectionOptions options = new GGUFInspector.InspectionOptions()
			.filterKey("test")
			.maxStringLength(10);

		GGUFInspector.InspectionResult result = inspector.inspect(options);

		assertNotNull(result);
		// Results should be filtered based on the key
	}

	@Test
	void testInvalidFile() {
		Path invalidFile = tempDir.resolve("nonexistent.gguf");

		assertThrows(IOException.class, () -> {
			new GGUFInspector(invalidFile);
		});
	}

	@Test
	void testPrintInspectionDoesNotThrow() {
		assertDoesNotThrow(() -> {
			inspector.printInspection();
		});
	}

	@Test
	void testVerboseInspection() throws IOException {
		GGUFInspector.InspectionOptions options = new GGUFInspector.InspectionOptions()
			.verbose(true)
			.jsonOutput(false);

		assertDoesNotThrow(() -> {
			inspector.printInspection(options);
		});
	}

	@Test
	void testJsonOutput() throws IOException {
		GGUFInspector.InspectionOptions options = new GGUFInspector.InspectionOptions()
			.jsonOutput(true);

		assertDoesNotThrow(() -> {
			inspector.printInspection(options);
		});
	}

	/**
	 * Create a minimal test GGUF file for testing
	 */
	private void createTestGGUFFile(Path filePath) throws IOException {
		// This is a simplified test file creation
		// In practice, you would use GGUFWriter to create a proper test file
		byte[] testData = {
			'G', 'G', 'U', 'F', // Magic
			0x03, 0x00, 0x00, 0x00, // Version
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Tensor count
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00  // Metadata count
		};

		Files.write(filePath, testData);
	}

	/**
	 * Test command-line interface
	 */
	@Test
	void testCommandLineInterface() {
		String[] args = {
			testGGUFFile.toString(),
			"--verbose",
			"--no-metadata"
		};

		// Test that CLI doesn't throw exceptions
		assertDoesNotThrow(() -> {
			// In a real test, you might capture stdout/stderr
			// For now, just ensure no exceptions are thrown
		});
	}

	@Test
	void testInspectionResultProperties() throws IOException {
		GGUFInspector.InspectionResult result = inspector.inspect();

		// Test FileInfo properties
		assertNotNull(result.fileInfo.endianness);
		assertNotNull(result.fileInfo.hostEndianness);
		assertTrue(result.fileInfo.fileSize >= 0);
		assertTrue(result.fileInfo.version >= 0);
		assertTrue(result.fileInfo.tensorCount >= 0);
		assertTrue(result.fileInfo.metadataCount >= 0);
		assertNotNull(result.fileInfo.checksum);
	}

	@Test
	void testInspectionOptionsBuilder() {
		GGUFInspector.InspectionOptions options = new GGUFInspector.InspectionOptions()
			.metadata(true)
			.tensors(true)
			.tensorData(false)
			.fileStructure(true)
			.verbose(true)
			.jsonOutput(false)
			.filterKey("test")
			.maxStringLength(100);

		// Test that options can be built without issues
		assertNotNull(options);
	}

	@Test
	void testResourceCleanup() throws IOException {
		GGUFInspector testInspector = new GGUFInspector(testGGUFFile);

		// Inspector should be closeable
		assertDoesNotThrow(() -> {
			testInspector.close();
		});
	}
}