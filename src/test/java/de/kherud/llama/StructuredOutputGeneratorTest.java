package de.kherud.llama;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static java.lang.System.Logger.Level.DEBUG;

public class StructuredOutputGeneratorTest {
	private static final System.Logger logger = System.getLogger(StructuredOutputGeneratorTest.class.getName());

	private StructuredOutputGenerator generator;

	// Test data classes
	public static class PersonInfo {
		public String name;
		public int age;
		public String email;
		public List<String> skills;

		@StructuredOutputGenerator.Optional
		public String bio;

		@StructuredOutputGenerator.Description("Person's home address")
		public String address;
	}

	public static class TaskResponse {
		public String taskId;
		public String status;
		public String result;
		public Map<String, Object> metadata;
		public boolean completed;

		@StructuredOutputGenerator.Optional
		public String error;

		@StructuredOutputGenerator.Description("Execution time in milliseconds")
		public long executionTime;
	}

	public static class CodeAnalysis {
		public String language;
		public int linesOfCode;
		public List<String> functions;
		public List<String> imports;
		public Map<String, Integer> complexity;

		@StructuredOutputGenerator.Optional
		@StructuredOutputGenerator.Description("Additional notes about the code")
		public String notes;
	}

	@Before
	public void setUp() {
		generator = new StructuredOutputGenerator();
	}

	@Test
	public void testSchemaRegistration() {
		logger.log(DEBUG, "\n=== Schema Registration Test ===");

		generator.registerSchema("person", PersonInfo.class, "Person information schema", true);
		generator.registerSchema("task", TaskResponse.class, "Task response schema", false);

		Set<String> schemas = generator.getRegisteredSchemas();
		Assert.assertTrue("Should contain person schema", schemas.contains("person"));
		Assert.assertTrue("Should contain task schema", schemas.contains("task"));
		Assert.assertEquals("Should have 2 schemas", 2, schemas.size());

		logger.log(DEBUG, "Registered schemas: " + schemas);
		logger.log(DEBUG, "✅ Schema registration test passed!");
	}

	@Test
	public void testPromptGeneration() {
		logger.log(DEBUG, "\n=== Prompt Generation Test ===");

		generator.registerSchema("person", PersonInfo.class, "Person information schema", true);

		String prompt = generator.generatePromptInstructions("person");

		Assert.assertNotNull("Prompt should not be null", prompt);
		Assert.assertTrue("Prompt should contain JSON structure", prompt.contains("```json"));
		Assert.assertTrue("Prompt should contain field names", prompt.contains("\"name\""));
		Assert.assertTrue("Prompt should contain field names", prompt.contains("\"age\""));
		Assert.assertTrue("Prompt should contain field names", prompt.contains("\"skills\""));
		Assert.assertTrue("Prompt should mark required fields", prompt.contains("(REQUIRED)"));
		Assert.assertTrue("Prompt should contain strict mode warning",
			prompt.contains("IMPORTANT: Output MUST be valid JSON"));

		logger.log(DEBUG, "Generated prompt:");
		logger.log(DEBUG, prompt);
		logger.log(DEBUG, "✅ Prompt generation test passed!");
	}

	@Test
	public void testJsonExtraction() {
		logger.log(DEBUG, "\n=== JSON Extraction Test ===");

		// Test extraction from markdown code block
		String llmOutput1 = "Here's the person data:\n```json\n{\"name\": \"John\", \"age\": 30}\n```\nThat's the data.";
		String json1 = generator.extractJson(llmOutput1);

		Assert.assertNotNull("Should extract JSON from code block", json1);
		Assert.assertTrue("Should contain name field", json1.contains("\"name\""));
		Assert.assertTrue("Should contain John", json1.contains("\"John\""));

		logger.log(DEBUG, "Extracted from code block: " + json1);

		// Test extraction from raw JSON
		String llmOutput2 = "The result is {\"status\": \"success\", \"value\": 42} as expected.";
		String json2 = generator.extractJson(llmOutput2);

		Assert.assertNotNull("Should extract raw JSON", json2);
		Assert.assertTrue("Should contain status field", json2.contains("\"status\""));

		logger.log(DEBUG, "Extracted raw JSON: " + json2);

		// Test extraction of array
		String llmOutput3 = "Skills: [\"Java\", \"Python\", \"JavaScript\"]";
		String json3 = generator.extractJson(llmOutput3);

		Assert.assertNotNull("Should extract JSON array", json3);
		Assert.assertTrue("Should contain Java", json3.contains("\"Java\""));

		logger.log(DEBUG, "Extracted array: " + json3);
		logger.log(DEBUG, "✅ JSON extraction test passed!");
	}

	@Test
	public void testSuccessfulParsing() {
		logger.log(DEBUG, "\n=== Successful Parsing Test ===");

		generator.registerSchema("person", PersonInfo.class, "Person information schema", false);

		String llmOutput = "```json\n" +
			"{\n" +
			"  \"name\": \"Alice Smith\",\n" +
			"  \"age\": 28,\n" +
			"  \"email\": \"alice@example.com\",\n" +
			"  \"skills\": [\"Java\", \"Spring\", \"Docker\"],\n" +
			"  \"address\": \"123 Main St\"\n" +
			"}\n" +
			"```";

		StructuredOutputGenerator.ParseResult<PersonInfo> result =
			generator.parse(llmOutput, "person", PersonInfo.class);

		Assert.assertTrue("Parsing should succeed", result.success());
		Assert.assertNotNull("Should have value", result.value());

		PersonInfo person = result.value();
		Assert.assertEquals("Name should match", "Alice Smith", person.name);
		Assert.assertEquals("Age should match", 28, person.age);
		Assert.assertEquals("Email should match", "alice@example.com", person.email);
		Assert.assertEquals("Should have 3 skills", 3, person.skills.size());
		Assert.assertTrue("Should have Java skill", person.skills.contains("Java"));

		logger.log(DEBUG, "Parsed person: " + person.name + ", age: " + person.age);
		logger.log(DEBUG, "Skills: " + person.skills);
		logger.log(DEBUG, "✅ Successful parsing test passed!");
	}

	@Test
	public void testStrictModeValidation() {
		logger.log(DEBUG, "\n=== Strict Mode Validation Test ===");

		generator.registerSchema("task_strict", TaskResponse.class, "Task response schema", true);

		// Missing required field
		String llmOutput = "```json\n" +
			"{\n" +
			"  \"taskId\": \"task-123\",\n" +
			"  \"status\": \"completed\"\n" +
			"}\n" +
			"```";

		StructuredOutputGenerator.ParseResult<TaskResponse> result =
			generator.parse(llmOutput, "task_strict", TaskResponse.class);

		Assert.assertFalse("Parsing should fail in strict mode", result.success());
		Assert.assertFalse("Errors should not be empty", result.errors().isEmpty());

		boolean hasRequiredFieldError = result.errors().stream()
			.anyMatch(e -> e.contains("Missing required field"));
		Assert.assertTrue("Should have required field error", hasRequiredFieldError);

		logger.log(DEBUG, "Validation errors: " + result.errors());
		logger.log(DEBUG, "✅ Strict mode validation test passed!");
	}

	@Test
	public void testNonStrictMode() {
		logger.log(DEBUG, "\n=== Non-Strict Mode Test ===");

		generator.registerSchema("task_lenient", TaskResponse.class, "Task response schema", false);

		// Missing some required fields but non-strict mode
		String llmOutput = "```json\n" +
			"{\n" +
			"  \"taskId\": \"task-456\",\n" +
			"  \"status\": \"in_progress\",\n" +
			"  \"completed\": false\n" +
			"}\n" +
			"```";

		StructuredOutputGenerator.ParseResult<TaskResponse> result =
			generator.parse(llmOutput, "task_lenient", TaskResponse.class);

		Assert.assertTrue("Parsing should succeed in non-strict mode", result.success());
		Assert.assertNotNull("Should have value", result.value());

		TaskResponse task = result.value();
		Assert.assertEquals("Task ID should match", "task-456", task.taskId);
		Assert.assertEquals("Status should match", "in_progress", task.status);
		Assert.assertFalse("Completed should be false", task.completed);
		Assert.assertNull("Result should be null", task.result);

		// Check warnings in metadata
		List<String> warnings = (List<String>) result.metadata().get("validationWarnings");
		Assert.assertNotNull("Should have warnings", warnings);
		Assert.assertFalse("Warnings should not be empty", warnings.isEmpty());

		logger.log(DEBUG, "Parsed task: " + task.taskId + " - " + task.status);
		logger.log(DEBUG, "Validation warnings: " + warnings);
		logger.log(DEBUG, "✅ Non-strict mode test passed!");
	}

	@Test
	public void testComplexNestedStructure() {
		logger.log(DEBUG, "\n=== Complex Nested Structure Test ===");

		generator.registerSchema("code_analysis", CodeAnalysis.class, "Code analysis schema", false);

		String llmOutput = "Here's the analysis:\n" +
			"```json\n" +
			"{\n" +
			"  \"language\": \"Java\",\n" +
			"  \"linesOfCode\": 450,\n" +
			"  \"functions\": [\"main\", \"processData\", \"validateInput\", \"generateReport\"],\n" +
			"  \"imports\": [\"java.util.*\", \"java.io.*\", \"com.example.utils.*\"],\n" +
			"  \"complexity\": {\n" +
			"    \"cyclomatic\": 15,\n" +
			"    \"cognitive\": 22,\n" +
			"    \"halstead\": 180\n" +
			"  },\n" +
			"  \"notes\": \"Well-structured code with good separation of concerns\"\n" +
			"}\n" +
			"```";

		StructuredOutputGenerator.ParseResult<CodeAnalysis> result =
			generator.parse(llmOutput, "code_analysis", CodeAnalysis.class);

		Assert.assertTrue("Parsing should succeed", result.success());
		Assert.assertNotNull("Should have value", result.value());

		CodeAnalysis analysis = result.value();
		Assert.assertEquals("Language should match", "Java", analysis.language);
		Assert.assertEquals("Lines of code should match", 450, analysis.linesOfCode);
		Assert.assertEquals("Should have 4 functions", 4, analysis.functions.size());
		Assert.assertEquals("Should have 3 imports", 3, analysis.imports.size());
		Assert.assertNotNull("Should have complexity map", analysis.complexity);
		Assert.assertEquals("Cyclomatic complexity should match",
			Integer.valueOf(15), analysis.complexity.get("cyclomatic"));

		logger.log(DEBUG, "Language: " + analysis.language);
		logger.log(DEBUG, "Functions: " + analysis.functions);
		logger.log(DEBUG, "Complexity metrics: " + analysis.complexity);
		logger.log(DEBUG, "✅ Complex nested structure test passed!");
	}

	@Test
	public void testInvalidJsonHandling() {
		logger.log(DEBUG, "\n=== Invalid JSON Handling Test ===");

		generator.registerSchema("person", PersonInfo.class, "Person information schema", true);

		// Invalid JSON
		String llmOutput = "```json\n{\"name\": \"Bob\" \"age\": 25}\n```"; // Missing comma

		StructuredOutputGenerator.ParseResult<PersonInfo> result =
			generator.parse(llmOutput, "person", PersonInfo.class);

		Assert.assertFalse("Parsing should fail for invalid JSON", result.success());
		Assert.assertFalse("Should have errors", result.errors().isEmpty());

		boolean hasParsingError = result.errors().stream()
			.anyMatch(e -> e.contains("Parsing error"));
		Assert.assertTrue("Should have parsing error", hasParsingError);

		logger.log(DEBUG, "Error for invalid JSON: " + result.errors());

		// No JSON found
		String llmOutput2 = "This is just plain text without any JSON";

		StructuredOutputGenerator.ParseResult<PersonInfo> result2 =
			generator.parse(llmOutput2, "person", PersonInfo.class);

		Assert.assertFalse("Parsing should fail when no JSON found", result2.success());
		Assert.assertTrue("Should have no JSON error",
			result2.errors().stream().anyMatch(e -> e.contains("No valid JSON")));

		logger.log(DEBUG, "Error for no JSON: " + result2.errors());
		logger.log(DEBUG, "✅ Invalid JSON handling test passed!");
	}

	@Test
	public void testSchemaBuilder() {
		logger.log(DEBUG, "\n=== Schema Builder Test ===");

		StructuredOutputGenerator.OutputSchema schema = new StructuredOutputGenerator.SchemaBuilder()
			.name("custom_schema")
			.targetClass(PersonInfo.class)
			.description("Custom person schema")
			.strict(true)
			.build();

		Assert.assertNotNull("Schema should be created", schema);
		Assert.assertEquals("Name should match", "custom_schema", schema.getName());
		Assert.assertEquals("Target class should match", PersonInfo.class, schema.getTargetClass());
		Assert.assertEquals("Description should match", "Custom person schema", schema.getDescription());
		Assert.assertTrue("Should be strict", schema.isStrict());
		Assert.assertFalse("Should have fields", schema.getFields().isEmpty());

		logger.log(DEBUG, "Built schema: " + schema.getName());
		logger.log(DEBUG, "Fields: " + schema.getFields().keySet());
		logger.log(DEBUG, "✅ Schema builder test passed!");
	}
}
