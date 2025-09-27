package de.kherud.llama.gguf;

import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Test suite for the library-friendly GGUFInspectorLibrary.
 * Demonstrates the improved API and validates functionality.
 */
public class GGUFInspectorLibraryTest {

	private Path testGgufFile;

	@Before
	public void setUp() {
		// Use a test GGUF file from test resources
		testGgufFile = Paths.get("src/test/resources/models/tinyllama-1.1b-chat-v0.3.Q2_K.gguf");

		// Skip test if file doesn't exist
		if (!testGgufFile.toFile().exists()) {
			System.err.println("Test GGUF file not found: " + testGgufFile);
			System.err.println("Skipping GGUFInspectorLibrary tests");
		}
	}

	@Test
	public void testBasicInspection() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			GGUFInspectorLibrary.InspectionResult result = inspector.inspect();

			assertNotNull("Inspection result should not be null", result);
			assertEquals("File path should match", testGgufFile, result.getFilePath());

			if (result.getFileInfo().isPresent()) {
				assertTrue("File size should be positive",
					result.getFileInfo().get().getFileSize() > 0);
			}

			// Should have some metadata and tensors
			assertFalse("Should have metadata", result.getMetadata().isEmpty());
			assertFalse("Should have tensors", result.getTensors().isEmpty());
		}
	}

	@Test
	public void testFluentConfiguration() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)
				.includeMetadata(true)
				.includeTensors(false)
				.verbose(true)
				.filterByKey("general")) {

			GGUFInspectorLibrary.InspectionResult result = inspector.inspect();

			assertNotNull("Result should not be null", result);
			assertFalse("Should have metadata", result.getMetadata().isEmpty());
			assertTrue("Should not have tensors", result.getTensors().isEmpty());

			// Check that only filtered metadata is present
			assertTrue("Should only have general metadata",
				result.getMetadata().keySet().stream()
					.allMatch(key -> key.contains("general")));
		}
	}

	@Test
	public void testStreamingMetadata() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			AtomicInteger count = new AtomicInteger(0);

			inspector.streamMetadata()
				.filter(entry -> entry.getKey().startsWith("general"))
				.forEach(entry -> {
					assertNotNull("Key should not be null", entry.getKey());
					assertNotNull("Value should not be null", entry.getValue());
					count.incrementAndGet();
				});

			assertTrue("Should find some general metadata", count.get() > 0);
		}
	}

	@Test
	public void testStreamingTensors() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			AtomicInteger count = new AtomicInteger(0);

			inspector.streamTensors()
				.limit(5) // Limit to avoid processing too many
				.forEach(tensor -> {
					assertNotNull("Tensor name should not be null", tensor.getName());
					assertTrue("Tensor shape should have dimensions", tensor.getShape().length > 0);
					count.incrementAndGet();
				});

			assertTrue("Should find some tensors", count.get() > 0);
		}
	}

	@Test
	public void testIndividualQueries() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			// Test metadata queries
			assertTrue("Should have metadata keys", !inspector.getMetadataKeys().isEmpty());

			// Test if specific metadata exists
			if (inspector.hasMetadata("general.name")) {
				assertTrue("Should get general.name value",
					inspector.getMetadata("general.name").isPresent());
			}

			// Test tensor queries
			assertTrue("Should have tensor names", !inspector.getTensorNames().isEmpty());

			// Test file info
			GGUFInspectorLibrary.FileInfo fileInfo = inspector.getFileInfo();
			assertTrue("File size should be positive", fileInfo.getFileSize() > 0);
			assertTrue("Should have metadata count", fileInfo.getMetadataCount() > 0);
			assertTrue("Should have tensor count", fileInfo.getTensorCount() > 0);
		}
	}

	@Test
	public void testValidation() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			GGUFInspectorLibrary.ValidationResult validation = inspector.validate();

			assertTrue("File should exist", validation.isFileExists());
			assertTrue("File should be readable", validation.isFileReadable());
			assertTrue("Header should be valid", validation.isHeaderValid());
			assertTrue("Overall validation should pass", validation.isValid());
			assertTrue("Should have positive metadata count", validation.getMetadataCount() > 0);
			assertTrue("Should have positive tensor count", validation.getTensorCount() > 0);
		}
	}

	@Test
	public void testAsyncInspection() throws IOException, InterruptedException, ExecutionException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			CompletableFuture<GGUFInspectorLibrary.InspectionResult> future = inspector.inspectAsync();

			GGUFInspectorLibrary.InspectionResult result = future.get();
			assertNotNull("Async result should not be null", result);
			assertFalse("Should have metadata", result.getMetadata().isEmpty());
		}
	}

	@Test
	public void testAsyncValidation() throws IOException, InterruptedException, ExecutionException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			CompletableFuture<GGUFInspectorLibrary.ValidationResult> future = inspector.validateAsync();

			GGUFInspectorLibrary.ValidationResult result = future.get();
			assertNotNull("Async validation result should not be null", result);
			assertTrue("Validation should pass", result.isValid());
		}
	}

	@Test
	public void testProgressCallback() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)) {
			AtomicInteger progressCallbacks = new AtomicInteger(0);

			inspector.inspectWithProgress(progress -> {
				assertNotNull("Progress message should not be null", progress.getMessage());
				assertTrue("Progress should be between 0 and 1",
					progress.getProgress() >= 0.0 && progress.getProgress() <= 1.0);
				progressCallbacks.incrementAndGet();
			});

			assertTrue("Should receive progress callbacks", progressCallbacks.get() > 0);
		}
	}

	@Test
	public void testPredicateFiltering() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		try (GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile)
				.filterByKeyPredicate(key -> key.startsWith("general") || key.startsWith("tokenizer"))) {

			GGUFInspectorLibrary.InspectionResult result = inspector.inspect();

			// All metadata keys should match the predicate
			assertTrue("All metadata should match predicate",
				result.getMetadata().keySet().stream()
					.allMatch(key -> key.startsWith("general") || key.startsWith("tokenizer")));
		}
	}

	@Test(expected = IOException.class)
	public void testNonExistentFile() throws IOException {
		Path nonExistent = Paths.get("/path/that/does/not/exist.gguf");
		GGUFInspectorLibrary.open(nonExistent);
	}

	@Test
	public void testBuilderPattern() throws IOException {
		if (!testGgufFile.toFile().exists()) return;

		// Test that builder methods return the same instance for chaining
		GGUFInspectorLibrary inspector = GGUFInspectorLibrary.open(testGgufFile);
		GGUFInspectorLibrary result = inspector
			.includeMetadata(true)
			.includeTensors(true)
			.verbose(false);

		assertSame("Builder methods should return same instance", inspector, result);
		inspector.close();
	}

	@Test
	public void testFileInfoBuilder() {
		GGUFInspectorLibrary.FileInfo fileInfo = new GGUFInspectorLibrary.FileInfo.Builder()
			.fileSize(1024)
			.metadataCount(10)
			.tensorCount(5)
			.checksum("abc123")
			.build();

		assertEquals("File size should match", 1024, fileInfo.getFileSize());
		assertEquals("Metadata count should match", 10, fileInfo.getMetadataCount());
		assertEquals("Tensor count should match", 5, fileInfo.getTensorCount());
		assertTrue("Checksum should be present", fileInfo.getChecksum().isPresent());
		assertEquals("Checksum should match", "abc123", fileInfo.getChecksum().get());
	}

	@Test
	public void testTensorInfoBuilder() {
		long[] shape = {128, 256};
		GGUFInspectorLibrary.TensorInfo tensorInfo = new GGUFInspectorLibrary.TensorInfo.Builder()
			.name("test.tensor")
			.type(1)
			.shape(shape)
			.offset(1024)
			.build();

		assertEquals("Name should match", "test.tensor", tensorInfo.getName());
		assertEquals("Type should match", 1, tensorInfo.getType());
		assertArrayEquals("Shape should match", shape, tensorInfo.getShape());
		assertEquals("Offset should match", 1024, tensorInfo.getOffset());

		// Test that shape is copied (defensive copy)
		long[] originalShape = tensorInfo.getShape();
		originalShape[0] = 999;
		assertNotEquals("Shape should be defensively copied", 999, tensorInfo.getShape()[0]);
	}
}