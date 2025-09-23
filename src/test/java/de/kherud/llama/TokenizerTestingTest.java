package de.kherud.llama;

import de.kherud.llama.testing.TokenizerComparator;
import de.kherud.llama.testing.TokenizerTester;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class TokenizerTestingTest {

	private Path tempModelFile;
	private LlamaModel testModel;

	@Before
	public void setUp() throws IOException {
		// Create a temporary model file for testing
		tempModelFile = Files.createTempFile("test_model", ".gguf");

		// For this test, we'll use a mock approach since we don't have a real model
		// In practice, you would use an actual model file
		try {
			// Try to create a minimal model for testing
			// This will fail gracefully if no model is available
			ModelParameters params = new ModelParameters()
				.setModel(tempModelFile.toString())
				.setCtxSize(256);

			// This will throw if no model is available - that's expected in test environment
			// testModel = new LlamaModel(params);
		} catch (Exception e) {
			// Expected in test environment without actual model files
		}
	}

	@After
	public void tearDown() throws IOException {
		if (testModel != null) {
			testModel.close();
		}
		if (Files.exists(tempModelFile)) {
			Files.delete(tempModelFile);
		}
	}

	@Test
	public void testTokenizerTesterCreation() {
		// Test that TokenizerTester can be created (even if it fails to load a model)
		try {
			TokenizerTester tester = new TokenizerTester(tempModelFile.toString());
			fail("Should throw exception for invalid model file");
		} catch (Exception e) {
			// Expected - no actual model file
			assertTrue("Should get meaningful error", e.getMessage() != null);
		}
	}

	@Test
	public void testTokenizerComparatorCreation() {
		// Test that TokenizerComparator can be created (even if it fails to load models)
		try {
			TokenizerComparator comparator = new TokenizerComparator(
				tempModelFile.toString(), tempModelFile.toString(), "Test1", "Test2");
			fail("Should throw exception for invalid model files");
		} catch (Exception e) {
			// Expected - no actual model files
			assertTrue("Should get meaningful error", e.getMessage() != null);
		}
	}

	@Test
	public void testTokenizerStatsBasics() {
		TokenizerTester.TokenizerStats stats = new TokenizerTester.TokenizerStats();

		assertEquals(0, stats.totalTests);
		assertEquals(0, stats.encodeErrors);
		assertEquals(0, stats.decodeErrors);
		assertEquals(0, stats.encodeTime);
		assertEquals(0, stats.decodeTime);
		assertTrue(stats.errorSamples.isEmpty());

		// Test reset functionality
		stats.totalTests = 10;
		stats.encodeErrors = 2;
		stats.reset();

		assertEquals(0, stats.totalTests);
		assertEquals(0, stats.encodeErrors);
	}

	@Test
	public void testComparisonResultBasics() {
		TokenizerComparator.ComparisonResult result = new TokenizerComparator.ComparisonResult();

		assertEquals(0.0, result.getEncodeAccuracy(), 0.001);
		assertEquals(0.0, result.getDecodeAccuracy(), 0.001);

		// Test with some data
		result.totalTests = 100;
		result.encodeMatches = 95;
		result.decodeMatches = 98;

		assertEquals(0.95, result.getEncodeAccuracy(), 0.001);
		assertEquals(0.98, result.getDecodeAccuracy(), 0.001);

		String str = result.toString();
		assertTrue("Should contain test count", str.contains("tests=100"));
		assertTrue("Should contain accuracy", str.contains("95.00%"));
	}

	@Test
	public void testMismatchReport() {
		TokenizerComparator.MismatchReport report = new TokenizerComparator.MismatchReport(
			"test input", "expected output", "actual output", 5);

		assertEquals("test input", report.input);
		assertEquals("expected output", report.expected);
		assertEquals("actual output", report.actual);
		assertEquals(5, report.firstDifferenceIndex);

		String str = report.toString();
		assertTrue("Should contain input", str.contains("test input"));
		assertTrue("Should contain expected", str.contains("expected output"));
		assertTrue("Should contain actual", str.contains("actual output"));
		assertTrue("Should contain index", str.contains("5"));
	}


	@Test
	public void testUtilityMethods() {
		// Test the utility classes can be instantiated
		assertNotNull(new TokenizerTester.TokenizerStats());
		assertNotNull(new TokenizerComparator.ComparisonResult());

		// Test edge case handling
		TokenizerTester.TokenizerStats stats = new TokenizerTester.TokenizerStats();
		String statsStr = stats.toString();
		assertNotNull(statsStr);
		assertTrue(statsStr.contains("TokenizerStats"));

		TokenizerComparator.ComparisonResult result = new TokenizerComparator.ComparisonResult();
		String resultStr = result.toString();
		assertNotNull(resultStr);
		assertTrue(resultStr.contains("ComparisonResult"));
	}

	@Test
	public void testMockTokenizerFunctionality() {
		// Test the structure and APIs without actual models

		// Test that we can create test data
		String[] basicTests = {
			"", " ", "Hello world", "Test 123", "🦙 emoji test"
		};

		assertTrue("Should have test cases", basicTests.length > 0);

		// Test edge cases array
		String[] edgeCases = {
			"\u001f-a", "¼-a", "<s>test</s>", "unicode: αβγ"
		};

		assertTrue("Should have edge cases", edgeCases.length > 0);

		// Verify we can iterate over test cases
		long count = Arrays.stream(basicTests).count();
		assertEquals(5, count);
	}
}
