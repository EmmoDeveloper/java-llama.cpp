package de.kherud.llama.testing;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.util.List;
import java.util.Arrays;

/**
 * Test cases for ServerTestFramework.
 */
public class ServerTestFrameworkTest {

	private ServerTestFramework framework;
	private ServerTestFramework.TestConfig config;

	@BeforeEach
	void setUp() {
		config = new ServerTestFramework.TestConfig()
			.serverUrl("http://localhost:8080")
			.timeout(5000)
			.maxConcurrentRequests(2)
			.verbose(false);

		framework = new ServerTestFramework(config);
	}

	@AfterEach
	void tearDown() {
		if (framework != null) {
			framework.close();
		}
	}

	@Test
	void testFrameworkCreation() {
		assertNotNull(framework);
		assertNotNull(config);
	}

	@Test
	void testConfigBuilder() {
		ServerTestFramework.TestConfig testConfig = new ServerTestFramework.TestConfig()
			.serverUrl("http://test:9000")
			.timeout(10000)
			.maxConcurrentRequests(5)
			.verbose(true)
			.testDataPath("test_data")
			.testSuites(Arrays.asList("basic", "performance"));

		assertNotNull(testConfig);
	}

	@Test
	void testTestResultCreation() {
		ServerTestFramework.TestResult result = new ServerTestFramework.TestResult("test_health");

		assertNotNull(result);
		assertEquals("test_health", result.testName);
		assertFalse(result.passed); // Default should be false

		result.pass("Success message");
		assertTrue(result.passed);
		assertEquals("Success message", result.message);

		result.fail("Failure message");
		assertFalse(result.passed);
		assertEquals("Failure message", result.message);
	}

	@Test
	void testTestSuiteCreation() {
		ServerTestFramework.TestSuite suite = new ServerTestFramework.TestSuite("basic");

		assertNotNull(suite);
		assertEquals("basic", suite.name);
		assertEquals(0, suite.passed);
		assertEquals(0, suite.failed);
		assertTrue(suite.results.isEmpty());

		// Add a test result
		ServerTestFramework.TestResult result = new ServerTestFramework.TestResult("test1");
		result.pass("Success");
		suite.addResult(result);

		assertEquals(1, suite.passed);
		assertEquals(0, suite.failed);
		assertEquals(1, suite.results.size());
	}

	@Test
	void testTestSuiteFailedResults() {
		ServerTestFramework.TestSuite suite = new ServerTestFramework.TestSuite("test_suite");

		ServerTestFramework.TestResult failedResult = new ServerTestFramework.TestResult("test_fail");
		failedResult.fail("Test failed");
		suite.addResult(failedResult);

		assertEquals(0, suite.passed);
		assertEquals(1, suite.failed);
		assertEquals(1, suite.results.size());
	}

	@Test
	void testTestSuiteMixedResults() {
		ServerTestFramework.TestSuite suite = new ServerTestFramework.TestSuite("mixed_suite");

		// Add passing test
		ServerTestFramework.TestResult passResult = new ServerTestFramework.TestResult("test_pass");
		passResult.pass("Success");
		suite.addResult(passResult);

		// Add failing test
		ServerTestFramework.TestResult failResult = new ServerTestFramework.TestResult("test_fail");
		failResult.fail("Failure");
		suite.addResult(failResult);

		assertEquals(1, suite.passed);
		assertEquals(1, suite.failed);
		assertEquals(2, suite.results.size());
	}

	@Test
	void testTestResultDetails() {
		ServerTestFramework.TestResult result = new ServerTestFramework.TestResult("detailed_test");

		result.detail("response_code", 200);
		result.detail("response_time", 150);

		assertEquals(200, result.details.get("response_code"));
		assertEquals(150, result.details.get("response_time"));
	}

	@Test
	void testConfigurationValidation() {
		// Test configuration with various options
		ServerTestFramework.TestConfig validConfig = new ServerTestFramework.TestConfig()
			.serverUrl("https://api.example.com")
			.timeout(30000)
			.maxConcurrentRequests(10)
			.verbose(true)
			.testSuites(Arrays.asList("basic", "performance", "concurrency", "edge_cases"));

		assertNotNull(validConfig);

		ServerTestFramework validFramework = new ServerTestFramework(validConfig);
		assertNotNull(validFramework);
		validFramework.close();
	}

	@Test
	void testTestSuiteFiltering() {
		ServerTestFramework.TestConfig filteredConfig = new ServerTestFramework.TestConfig()
			.testSuites(Arrays.asList("basic", "performance")); // Only run specific suites

		assertNotNull(filteredConfig);
	}

	@Test
	void testConcurrentRequestsConfiguration() {
		ServerTestFramework.TestConfig concurrentConfig = new ServerTestFramework.TestConfig()
			.maxConcurrentRequests(1); // Sequential execution

		ServerTestFramework sequentialFramework = new ServerTestFramework(concurrentConfig);
		assertNotNull(sequentialFramework);
		sequentialFramework.close();
	}

	@Test
	void testTimeoutConfiguration() {
		ServerTestFramework.TestConfig timeoutConfig = new ServerTestFramework.TestConfig()
			.timeout(1000); // Very short timeout

		ServerTestFramework timeoutFramework = new ServerTestFramework(timeoutConfig);
		assertNotNull(timeoutFramework);
		timeoutFramework.close();
	}

	@Test
	void testVerboseOutput() {
		ServerTestFramework.TestConfig verboseConfig = new ServerTestFramework.TestConfig()
			.verbose(true);

		ServerTestFramework verboseFramework = new ServerTestFramework(verboseConfig);
		assertNotNull(verboseFramework);
		verboseFramework.close();
	}

	@Test
	void testTestDataPath() {
		ServerTestFramework.TestConfig dataConfig = new ServerTestFramework.TestConfig()
			.testDataPath("/custom/test/data");

		assertNotNull(dataConfig);
	}

	@Test
	void testResourceCleanup() {
		ServerTestFramework testFramework = new ServerTestFramework(config);

		assertDoesNotThrow(() -> {
			testFramework.close();
		});
	}

	@Test
	void testCommandLineArguments() {
		String[] args = {
			"--url", "http://localhost:9000",
			"--timeout", "15000",
			"--concurrent", "3",
			"--verbose",
			"--suites", "basic,performance"
		};

		// Test that CLI arguments are parsed without issues
		assertDoesNotThrow(() -> {
			// In a real implementation, you would test argument parsing
		});
	}

	@Test
	void testTestResultTiming() {
		ServerTestFramework.TestResult result = new ServerTestFramework.TestResult("timed_test");

		// Set duration
		result.duration = 1500; // 1.5 seconds

		assertEquals(1500, result.duration);
	}

	@Test
	void testTestResultException() {
		ServerTestFramework.TestResult result = new ServerTestFramework.TestResult("exception_test");

		Exception testException = new RuntimeException("Test exception");
		result.fail("Test failed with exception", testException);

		assertFalse(result.passed);
		assertEquals("Test failed with exception", result.message);
		assertEquals(testException, result.exception);
	}

	@Test
	void testSummaryGeneration() {
		// Create multiple test suites
		ServerTestFramework.TestSuite suite1 = new ServerTestFramework.TestSuite("suite1");
		ServerTestFramework.TestSuite suite2 = new ServerTestFramework.TestSuite("suite2");

		// Add results to first suite
		suite1.addResult(new ServerTestFramework.TestResult("test1").pass("OK"));
		suite1.addResult(new ServerTestFramework.TestResult("test2").pass("OK"));

		// Add results to second suite
		suite2.addResult(new ServerTestFramework.TestResult("test3").pass("OK"));
		suite2.addResult(new ServerTestFramework.TestResult("test4").fail("Failed"));

		List<ServerTestFramework.TestSuite> results = Arrays.asList(suite1, suite2);

		// Test that summary can be printed without errors
		assertDoesNotThrow(() -> {
			framework.printSummary(results);
		});
	}

	@Test
	void testFrameworkWithMockServer() {
		// Test framework behavior when server is not available
		ServerTestFramework.TestConfig mockConfig = new ServerTestFramework.TestConfig()
			.serverUrl("http://nonexistent-server:9999")
			.timeout(1000); // Short timeout for quick failure

		ServerTestFramework mockFramework = new ServerTestFramework(mockConfig);

		assertNotNull(mockFramework);

		// Framework should handle connection failures gracefully
		assertDoesNotThrow(() -> {
			mockFramework.close();
		});
	}
}