package de.kherud.llama;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static java.lang.System.Logger.Level.DEBUG;

public class FunctionCallSystemTest {
	private static final System.Logger logger = System.getLogger(FunctionCallSystemTest.class.getName());
	private FunctionCallSystem functionSystem;

	// Sample tool class for testing
	public static class CalculatorTool {
		public int add(int a, int b) {
			return a + b;
		}

		public int multiply(int a, int b) {
			return a * b;
		}

		public String formatResult(int result, String operation) {
			return String.format("Result of %s: %d", operation, result);
		}

		public boolean isEven(int number) {
			return number % 2 == 0;
		}
	}

	public static class FileSystemTool {
		public List<String> listFiles(String directory) {
			return Arrays.asList("file1.txt", "file2.java", "readme.md");
		}

		public boolean createFile(String filename, String content) {
			// Simulate file creation
			return filename != null && !filename.isEmpty();
		}

		public String getFileInfo(String filename) {
			return "File: " + filename + ", Size: 1024 bytes, Modified: 2024-01-01";
		}
	}

	@Before
	public void setUp() {
		functionSystem = new FunctionCallSystem();
	}

	@Test
	public void testBasicFunctionRegistration() {
		logger.log(DEBUG, "\n=== Basic Function Registration Test ===");

		CalculatorTool calculator = new CalculatorTool();

		try {
			functionSystem.registerFunction("add", "Add two numbers",
				calculator.getClass().getMethod("add", int.class, int.class), calculator);

			Set<String> functionNames = functionSystem.getRegisteredFunctionNames();
			Assert.assertTrue("Function should be registered", functionNames.contains("add"));
			Assert.assertEquals("Should have one function", 1, functionNames.size());

			logger.log(DEBUG, "✅ Successfully registered function: add");

		} catch (Exception e) {
			Assert.fail("Function registration should not throw exception: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Basic function registration test passed!");
	}

	@Test
	public void testToolRegistration() {
		logger.log(DEBUG, "\n=== Tool Registration Test ===");

		CalculatorTool calculator = new CalculatorTool();
		functionSystem.registerTool(calculator, "calc");

		Set<String> functionNames = functionSystem.getRegisteredFunctionNames();

		// Should register all public methods (excluding Object methods)
		Assert.assertTrue("Should contain calc_add", functionNames.contains("calc_add"));
		Assert.assertTrue("Should contain calc_multiply", functionNames.contains("calc_multiply"));
		Assert.assertTrue("Should contain calc_formatResult", functionNames.contains("calc_formatResult"));
		Assert.assertTrue("Should contain calc_isEven", functionNames.contains("calc_isEven"));

		logger.log(DEBUG, "Registered functions: " + functionNames);
		logger.log(DEBUG, "✅ Tool registration test passed!");
	}

	@Test
	public void testFunctionExecution() {
		logger.log(DEBUG, "\n=== Function Execution Test ===");

		CalculatorTool calculator = new CalculatorTool();
		functionSystem.registerTool(calculator, "calc");

		FunctionCallSystem.FunctionCallConfig config = new FunctionCallSystem.FunctionCallConfig();

		// Test addition (using arg0, arg1 because reflection gives parameter names as argN)
		Map<String, Object> addParams = new HashMap<>();
		addParams.put("arg0", 15);
		addParams.put("arg1", 27);

		FunctionCallSystem.FunctionCallResult result = functionSystem.executeFunction("calc_add", addParams, config);

		Assert.assertTrue("Function call should succeed", result.isSuccess());
		Assert.assertEquals("Result should be 42", 42, result.getResult());
		Assert.assertTrue("Execution time should be positive", result.getExecutionTimeMs() >= 0);

		logger.log(DEBUG, "Add result: " + result);

		// Test string formatting
		Map<String, Object> formatParams = new HashMap<>();
		formatParams.put("arg0", 42);
		formatParams.put("arg1", "addition");

		FunctionCallSystem.FunctionCallResult formatResult = functionSystem.executeFunction("calc_formatResult", formatParams, config);

		Assert.assertTrue("Format function should succeed", formatResult.isSuccess());
		Assert.assertEquals("Formatted result", "Result of addition: 42", formatResult.getResult());

		logger.log(DEBUG, "Format result: " + formatResult);
		logger.log(DEBUG, "✅ Function execution test passed!");
	}

	@Test
	public void testFunctionCallParsing() {
		logger.log(DEBUG, "\n=== Function Call Parsing Test ===");

		// Test single function call parsing
		String llmOutput1 = "I need to call a function: {\"function\": \"calc_add\", \"parameters\": {\"a\": 10, \"b\": 5}}";
		List<Map<String, Object>> calls1 = functionSystem.parseFunctionCalls(llmOutput1);

		Assert.assertEquals("Should parse one function call", 1, calls1.size());
		Assert.assertEquals("Function name should be calc_add", "calc_add", calls1.get(0).get("function"));

		Map<String, Object> params1 = (Map<String, Object>) calls1.get(0).get("parameters");
		Assert.assertEquals("Parameter a should be 10", 10, params1.get("a"));
		Assert.assertEquals("Parameter b should be 5", 5, params1.get("b"));

		logger.log(DEBUG, "Parsed single call: " + calls1);

		// Test array of function calls parsing
		String llmOutput2 = "Here are multiple calls: [" +
			"{\"function\": \"calc_add\", \"parameters\": {\"a\": 1, \"b\": 2}}," +
			"{\"function\": \"calc_multiply\", \"parameters\": {\"a\": 3, \"b\": 4}}" +
			"]";
		List<Map<String, Object>> calls2 = functionSystem.parseFunctionCalls(llmOutput2);

		Assert.assertEquals("Should parse two function calls", 2, calls2.size());
		Assert.assertEquals("First function should be calc_add", "calc_add", calls2.get(0).get("function"));
		Assert.assertEquals("Second function should be calc_multiply", "calc_multiply", calls2.get(1).get("function"));

		logger.log(DEBUG, "Parsed multiple calls: " + calls2);
		logger.log(DEBUG, "✅ Function call parsing test passed!");
	}

	@Test
	public void testSchemaGeneration() {
		logger.log(DEBUG, "\n=== Schema Generation Test ===");

		CalculatorTool calculator = new CalculatorTool();
		FileSystemTool fileSystem = new FileSystemTool();

		functionSystem.registerTool(calculator, "calc");
		functionSystem.registerTool(fileSystem, "fs");

		String schema = functionSystem.generateFunctionSchema();

		Assert.assertNotNull("Schema should not be null", schema);
		Assert.assertTrue("Schema should contain functions", schema.contains("functions"));
		Assert.assertTrue("Schema should contain calc_add", schema.contains("calc_add"));
		Assert.assertTrue("Schema should contain fs_listFiles", schema.contains("fs_listFiles"));

		logger.log(DEBUG, "Generated schema:");
		logger.log(DEBUG, schema);
		logger.log(DEBUG, "✅ Schema generation test passed!");
	}

	@Test
	public void testParameterTypeConversion() {
		logger.log(DEBUG, "\n=== Parameter Type Conversion Test ===");

		CalculatorTool calculator = new CalculatorTool();
		functionSystem.registerTool(calculator, "calc");

		FunctionCallSystem.FunctionCallConfig config = new FunctionCallSystem.FunctionCallConfig();

		// Test string to int conversion
		Map<String, Object> params = new HashMap<>();
		params.put("arg0", "15");  // String instead of int
		params.put("arg1", "27");  // String instead of int

		FunctionCallSystem.FunctionCallResult result = functionSystem.executeFunction("calc_add", params, config);

		Assert.assertTrue("Function should succeed with type conversion", result.isSuccess());
		Assert.assertEquals("Result should be 42 after conversion", 42, result.getResult());

		logger.log(DEBUG, "Type conversion result: " + result);
		logger.log(DEBUG, "✅ Parameter type conversion test passed!");
	}

	@Test
	public void testErrorHandling() {
		logger.log(DEBUG, "\n=== Error Handling Test ===");

		FunctionCallSystem.FunctionCallConfig config = new FunctionCallSystem.FunctionCallConfig();

		// Test non-existent function
		Map<String, Object> params = new HashMap<>();
		params.put("x", 1);

		FunctionCallSystem.FunctionCallResult result1 = functionSystem.executeFunction("non_existent", params, config);

		Assert.assertFalse("Non-existent function should fail", result1.isSuccess());
		Assert.assertNotNull("Error message should be present", result1.getError());
		Assert.assertTrue("Error should mention function not found", result1.getError().contains("not found"));

		logger.log(DEBUG, "Non-existent function error: " + result1.getError());

		// Test invalid parameters
		CalculatorTool calculator = new CalculatorTool();
		functionSystem.registerTool(calculator, "calc");

		Map<String, Object> badParams = new HashMap<>();
		badParams.put("arg0", "not_a_number");
		badParams.put("arg1", 5);

		FunctionCallSystem.FunctionCallResult result2 = functionSystem.executeFunction("calc_add", badParams, config);

		Assert.assertFalse("Invalid parameter should fail", result2.isSuccess());
		Assert.assertNotNull("Error message should be present", result2.getError());

		logger.log(DEBUG, "Invalid parameter error: " + result2.getError());
		logger.log(DEBUG, "✅ Error handling test passed!");
	}

	@Test
	public void testMultipleTools() {
		logger.log(DEBUG, "\n=== Multiple Tools Test ===");

		CalculatorTool calculator = new CalculatorTool();
		FileSystemTool fileSystem = new FileSystemTool();

		functionSystem.registerTool(calculator, "calc");
		functionSystem.registerTool(fileSystem, "fs");

		FunctionCallSystem.FunctionCallConfig config = new FunctionCallSystem.FunctionCallConfig();

		// Test calculator function
		Map<String, Object> calcParams = new HashMap<>();
		calcParams.put("arg0", 6);
		calcParams.put("arg1", 7);

		FunctionCallSystem.FunctionCallResult calcResult = functionSystem.executeFunction("calc_multiply", calcParams, config);
		Assert.assertTrue("Calculator function should work", calcResult.isSuccess());
		Assert.assertEquals("6 * 7 = 42", 42, calcResult.getResult());

		// Test file system function
		Map<String, Object> fsParams = new HashMap<>();
		fsParams.put("arg0", "/home/user");

		FunctionCallSystem.FunctionCallResult fsResult = functionSystem.executeFunction("fs_listFiles", fsParams, config);
		Assert.assertTrue("File system function should work", fsResult.isSuccess());
		Assert.assertTrue("Should return a list", fsResult.getResult() instanceof List);

		List<String> files = (List<String>) fsResult.getResult();
		Assert.assertEquals("Should have 3 files", 3, files.size());

		logger.log(DEBUG, "Calculator result: " + calcResult.getResult());
		logger.log(DEBUG, "File system result: " + fsResult.getResult());
		logger.log(DEBUG, "✅ Multiple tools test passed!");
	}
}
