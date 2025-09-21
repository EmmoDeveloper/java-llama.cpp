package de.kherud.llama.training;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * JUnit4 tests for dataset processing functionality.
 * Tests various dataset formats and processing utilities.
 */
public class DatasetProcessorTest {

	private static final String TEST_DIR = "./test_datasets";
	private static final ObjectMapper objectMapper = new ObjectMapper();

	@Before
	public void setUp() throws IOException {
		// Create test directory
		new File(TEST_DIR).mkdirs();
	}

	@After
	public void tearDown() throws IOException {
		// Clean up test files
		if (new File(TEST_DIR).exists()) {
			Files.walk(Paths.get(TEST_DIR))
				.map(java.nio.file.Path::toFile)
				.forEach(File::delete);
			new File(TEST_DIR).delete();
		}
	}

	@Test
	public void testTrainingExampleFormats() {
		// Test basic completion format
		TrainingApplication completion = TrainingApplication.completionFormat("Hello", "Hi there!");
		Assert.assertEquals("Input should match", "Hello", completion.input());
		Assert.assertEquals("Target should match", "Hi there!", completion.target());
		Assert.assertEquals("Full text should combine input and target", "HelloHi there!", completion.getFullText());

		// Test instruction format
		TrainingApplication instruction = TrainingApplication.instructionFormat(
			"Translate to French", "Hello world", "Bonjour le monde"
		);
		Assert.assertTrue("Should contain instruction format",
			instruction.input().contains("### Instruction:"));
		Assert.assertTrue("Should contain input section",
			instruction.input().contains("### Input:"));
		Assert.assertTrue("Should contain response prompt",
			instruction.input().contains("### Response:"));
		Assert.assertEquals("Target should be response", "Bonjour le monde", instruction.target());

		// Test chat format
		TrainingApplication chat = TrainingApplication.chatFormat(
			"You are a helpful assistant", "What is 2+2?", "2+2 equals 4."
		);
		Assert.assertTrue("Should contain system prompt",
			chat.input().contains("<|im_start|>system"));
		Assert.assertTrue("Should contain user message",
			chat.input().contains("<|im_start|>user"));
		Assert.assertTrue("Should contain assistant start",
			chat.input().contains("<|im_start|>assistant"));
		Assert.assertTrue("Target should end with end token",
			chat.target().endsWith("<|im_end|>"));
	}

	@Test
	public void testAlpacaDatasetLoading() throws IOException {
		// Create test Alpaca dataset
		String alpacaData = "[\n" +
			"  {\n" +
			"    \"instruction\": \"Translate the following to French\",\n" +
			"    \"input\": \"Hello world\",\n" +
			"    \"output\": \"Bonjour le monde\"\n" +
			"  },\n" +
			"  {\n" +
			"    \"instruction\": \"What is the capital of France?\",\n" +
			"    \"input\": \"\",\n" +
			"    \"output\": \"The capital of France is Paris.\"\n" +
			"  }\n" +
			"]";

		String filePath = TEST_DIR + "/alpaca_test.json";
		try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
			writer.write(alpacaData);
		}

		List<TrainingApplication> examples = DatasetProcessor.loadAlpacaDataset(filePath);

		Assert.assertEquals("Should load 2 examples", 2, examples.size());

		TrainingApplication first = examples.get(0);
		Assert.assertTrue("First example should contain instruction",
			first.input().contains("Translate the following to French"));
		Assert.assertEquals("First target should match", "Bonjour le monde", first.target());

		TrainingApplication second = examples.get(1);
		Assert.assertTrue("Second example should handle empty input",
			second.input().contains("What is the capital of France?"));
		Assert.assertEquals("Second target should match", "The capital of France is Paris.", second.target());
	}

	@Test
	public void testJsonlDatasetLoading() throws IOException {
		// Create test JSONL dataset
		String jsonlData =
			"{\"prompt\": \"What is 2+2?\", \"completion\": \"2+2 equals 4.\"}\n" +
			"{\"prompt\": \"Translate hello to Spanish\", \"completion\": \"Hola\"}\n" +
			"{\"prompt\": \"Name a color\", \"completion\": \"Blue\"}\n";

		String filePath = TEST_DIR + "/test.jsonl";
		try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
			writer.write(jsonlData);
		}

		List<TrainingApplication> examples = DatasetProcessor.loadJsonlDataset(filePath);

		Assert.assertEquals("Should load 3 examples", 3, examples.size());

		TrainingApplication first = examples.get(0);
		Assert.assertEquals("First prompt should match", "What is 2+2?", first.input());
		Assert.assertEquals("First completion should match", "2+2 equals 4.", first.target());

		TrainingApplication second = examples.get(1);
		Assert.assertEquals("Second prompt should match", "Translate hello to Spanish", second.input());
		Assert.assertEquals("Second completion should match", "Hola", second.target());
	}

	@Test
	public void testCsvDatasetLoading() throws IOException {
		// Create test CSV dataset
		String csvData =
			"prompt,completion,extra_column\n" +
			"\"What is the weather?\",\"It's sunny today\",ignored\n" +
			"\"How are you?\",\"I'm doing well\",also_ignored\n" +
			"\"What time is it?\",\"It's 3 PM\",more_data\n";

		String filePath = TEST_DIR + "/test.csv";
		try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
			writer.write(csvData);
		}

		List<TrainingApplication> examples = DatasetProcessor.loadCsvDataset(filePath);

		Assert.assertEquals("Should load 3 examples", 3, examples.size());

		TrainingApplication first = examples.get(0);
		Assert.assertEquals("First prompt should match", "What is the weather?", first.input());
		Assert.assertEquals("First completion should match", "It's sunny today", first.target());

		TrainingApplication third = examples.get(2);
		Assert.assertEquals("Third prompt should match", "What time is it?", third.input());
		Assert.assertEquals("Third completion should match", "It's 3 PM", third.target());
	}

	@Test
	public void testConversationDatasetLoading() throws IOException {
		// Create test conversation dataset
		String conversationData = "[\n" +
			"  {\n" +
			"    \"conversations\": [\n" +
			"      {\"from\": \"human\", \"value\": \"Hello there!\"},\n" +
			"      {\"from\": \"gpt\", \"value\": \"Hello! How can I help you today?\"}\n" +
			"    ]\n" +
			"  },\n" +
			"  {\n" +
			"    \"conversations\": [\n" +
			"      {\"from\": \"human\", \"value\": \"What is AI?\"},\n" +
			"      {\"from\": \"gpt\", \"value\": \"AI stands for Artificial Intelligence.\"}\n" +
			"    ]\n" +
			"  }\n" +
			"]";

		String filePath = TEST_DIR + "/conversations.json";
		try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
			writer.write(conversationData);
		}

		List<TrainingApplication> examples = DatasetProcessor.loadConversationDataset(filePath);

		Assert.assertEquals("Should load 2 examples", 2, examples.size());

		TrainingApplication first = examples.get(0);
		Assert.assertTrue("Should contain chat format", first.input().contains("<|im_start|>"));
		Assert.assertTrue("Should contain human message", first.input().contains("Hello there!"));
		Assert.assertTrue("Target should contain response",
			first.target().contains("Hello! How can I help you today?"));
	}

	@Test
	public void testTextDatasetLoading() throws IOException {
		// Create test text file
		String textContent =
			"This is the first sentence. This is the second sentence. " +
			"This is the third sentence. This is the fourth sentence. " +
			"This is the fifth sentence. This is the sixth sentence. " +
			"This is the seventh sentence. This is the eighth sentence.";

		String filePath = TEST_DIR + "/text_data.txt";
		try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
			writer.write(textContent);
		}

		List<TrainingApplication> examples = DatasetProcessor.loadTextDataset(filePath, 100, 20);

		Assert.assertTrue("Should create multiple examples", examples.size() > 0);

		for (TrainingApplication example : examples) {
			Assert.assertFalse("Input should not be empty", example.input().isEmpty());
			Assert.assertFalse("Target should not be empty", example.target().isEmpty());
			Assert.assertTrue("Combined length should be reasonable",
				example.getFullText().length() <= 120); // chunk size + some buffer
		}
	}

	@Test
	public void testDatasetFiltering() {
		// Create test examples of varying lengths
		List<TrainingApplication> examples = Arrays.asList(
			TrainingApplication.completionFormat("Short", "OK"),
			TrainingApplication.completionFormat("This is a medium length example", "This is also medium"),
			TrainingApplication.completionFormat(
				"This is a very long example that should be filtered out because it exceeds the token limit",
				"This is also a very long response that should cause this example to be filtered out"
			),
			TrainingApplication.completionFormat("Another short", "Yes")
		);

		// Filter with small token limit (approximate: 4 chars per token)
		List<TrainingApplication> filtered = DatasetProcessor.filterByLength(examples, 20); // ~80 chars

		Assert.assertTrue("Should filter out long examples", filtered.size() < examples.size());
		Assert.assertTrue("Should keep some examples", filtered.size() > 0);

		// All filtered examples should be within limit
		for (TrainingApplication example : filtered) {
			Assert.assertTrue("Filtered example should be within limit",
				example.getFullText().length() <= 80);
		}
	}

	@Test
	public void testTrainValidationSplit() {
		// Create test dataset
		List<TrainingApplication> examples = Arrays.asList(
			TrainingApplication.completionFormat("Input 1", "Output 1"),
			TrainingApplication.completionFormat("Input 2", "Output 2"),
			TrainingApplication.completionFormat("Input 3", "Output 3"),
			TrainingApplication.completionFormat("Input 4", "Output 4"),
			TrainingApplication.completionFormat("Input 5", "Output 5"),
			TrainingApplication.completionFormat("Input 6", "Output 6"),
			TrainingApplication.completionFormat("Input 7", "Output 7"),
			TrainingApplication.completionFormat("Input 8", "Output 8"),
			TrainingApplication.completionFormat("Input 9", "Output 9"),
			TrainingApplication.completionFormat("Input 10", "Output 10")
		);

		Map<String, List<TrainingApplication>> split = DatasetProcessor.trainValidationSplit(examples, 0.2f);

		Assert.assertTrue("Should have train split", split.containsKey("train"));
		Assert.assertTrue("Should have validation split", split.containsKey("validation"));

		int trainSize = split.get("train").size();
		int validationSize = split.get("validation").size();

		Assert.assertEquals("Total size should match", examples.size(), trainSize + validationSize);
		Assert.assertTrue("Train set should be larger", trainSize > validationSize);
		Assert.assertEquals("Validation should be ~20%", 2, validationSize); // 20% of 10
	}

	@Test
	public void testJsonlSaving() throws IOException {
		// Create test examples
		List<TrainingApplication> examples = Arrays.asList(
			TrainingApplication.completionFormat("Hello", "Hi there"),
			TrainingApplication.instructionFormat("Greet", "Say hello", "Hello!"),
			TrainingApplication.chatFormat("Be helpful", "How are you?", "I'm doing well!")
		);

		String outputPath = TEST_DIR + "/output.jsonl";
		DatasetProcessor.saveAsJsonl(examples, outputPath);

		// Verify file was created
		Assert.assertTrue("Output file should exist", new File(outputPath).exists());

		// Load back and verify
		List<TrainingApplication> loaded = DatasetProcessor.loadJsonlDataset(outputPath);
		Assert.assertEquals("Should load same number of examples", examples.size(), loaded.size());

		// Verify first example
		TrainingApplication first = loaded.get(0);
		Assert.assertEquals("First input should match", "Hello", first.input());
		Assert.assertEquals("First target should match", "Hi there", first.target());
	}

	@Test
	public void testInvalidDatasetHandling() throws IOException {
		// Test invalid JSON
		String invalidJson = "{ invalid json content }";
		String invalidPath = TEST_DIR + "/invalid.json";
		try (PrintWriter writer = new PrintWriter(new FileWriter(invalidPath))) {
			writer.write(invalidJson);
		}

		try {
			DatasetProcessor.loadAlpacaDataset(invalidPath);
			Assert.fail("Should throw exception for invalid JSON");
		} catch (Exception e) {
			Assert.assertTrue("Should be JSON parsing error",
				e.getMessage().contains("parsing") || e instanceof com.fasterxml.jackson.core.JsonParseException);
		}

		// Test missing file
		try {
			DatasetProcessor.loadJsonlDataset("/nonexistent/file.jsonl");
			Assert.fail("Should throw exception for missing file");
		} catch (IOException e) {
			Assert.assertTrue("Should be file not found error", true);
		}
	}

	@Test
	public void testEmptyDatasetHandling() throws IOException {
		// Test empty JSON array
		String emptyJsonPath = TEST_DIR + "/empty.json";
		try (PrintWriter writer = new PrintWriter(new FileWriter(emptyJsonPath))) {
			writer.write("[]");
		}

		List<TrainingApplication> examples = DatasetProcessor.loadAlpacaDataset(emptyJsonPath);
		Assert.assertEquals("Empty dataset should return empty list", 0, examples.size());

		// Test empty JSONL
		String emptyJsonlPath = TEST_DIR + "/empty.jsonl";
		try (PrintWriter writer = new PrintWriter(new FileWriter(emptyJsonlPath))) {
			writer.write("");
		}

		examples = DatasetProcessor.loadJsonlDataset(emptyJsonlPath);
		Assert.assertEquals("Empty JSONL should return empty list", 0, examples.size());
	}

	@Test
	public void testDatasetProcessingEdgeCases() {
		// Test filtering with zero token limit
		List<TrainingApplication> examples = Arrays.asList(
			TrainingApplication.completionFormat("Test", "Response")
		);

		List<TrainingApplication> filtered = DatasetProcessor.filterByLength(examples, 0);
		Assert.assertEquals("Zero token limit should filter everything", 0, filtered.size());

		// Test train/validation split with 100% validation
		Map<String, List<TrainingApplication>> split = DatasetProcessor.trainValidationSplit(examples, 1.0f);
		Assert.assertEquals("100% validation should leave no training data", 0, split.get("train").size());
		Assert.assertEquals("100% validation should put all in validation", 1, split.get("validation").size());

		// Test train/validation split with 0% validation
		split = DatasetProcessor.trainValidationSplit(examples, 0.0f);
		Assert.assertEquals("0% validation should put all in training", 1, split.get("train").size());
		Assert.assertEquals("0% validation should leave no validation data", 0, split.get("validation").size());
	}

	@Test
	public void testCsvParsingWithQuotes() throws IOException {
		// Test CSV with quotes and commas inside fields
		String complexCsv =
			"prompt,completion\n" +
			"\"What is the weather like today, specifically?\",\"It's sunny, warm, and pleasant\"\n" +
			"\"How do you say \"\"hello\"\" in French?\",\"You say \"\"bonjour\"\"\"\n";

		String filePath = TEST_DIR + "/complex.csv";
		try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
			writer.write(complexCsv);
		}

		List<TrainingApplication> examples = DatasetProcessor.loadCsvDataset(filePath);
		Assert.assertEquals("Should parse 2 examples", 2, examples.size());

		TrainingApplication first = examples.get(0);
		Assert.assertTrue("Should handle comma in field",
			first.input().contains("specifically?"));
		Assert.assertTrue("Should handle comma in response",
			first.target().contains("sunny, warm"));
	}
}
