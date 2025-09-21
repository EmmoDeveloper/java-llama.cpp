package de.kherud.llama;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static java.lang.System.Logger.Level.DEBUG;

public class InferencePatternsTest {
	private static final System.Logger logger = System.getLogger(InferencePatternsTest.class.getName());

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");

		model = new LlamaModel(
			new ModelParameters()
				.setCtxSize(1024)
				.setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(20)
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
		InferencePatterns.shutdown();
	}

	@Test
	public void testBatchInference() {
		logger.log(DEBUG, "\n=== Batch Inference Test ===");

		List<String> prompts = Arrays.asList(
			"def hello():",
			"class Calculator:",
			"import numpy as np"
		);

		InferenceParameters baseParams = new InferenceParameters("")
			.setNPredict(20)
			.setTemperature(0.1f);

		long startTime = System.currentTimeMillis();
		InferencePatterns.BatchResult result = InferencePatterns.batchInference(model, prompts, baseParams);
		long duration = System.currentTimeMillis() - startTime;

		Assert.assertEquals("Should have 3 inputs", 3, result.getInputs().size());
		Assert.assertEquals("Should have 3 outputs", 3, result.getOutputs().size());
		Assert.assertEquals("Should have 3 latencies", 3, result.getLatencies().size());

		logger.log(DEBUG, "Batch processing took: " + duration + "ms");
		logger.log(DEBUG, "Average latency per prompt: " + result.getAverageLatency() + "ms");
		logger.log(DEBUG, "Total tokens generated: " + result.getTotalTokensGenerated());

		for (int i = 0; i < prompts.size(); i++) {
			logger.log(DEBUG, "Input: '" + prompts.get(i) + "' -> Output: '" +
				result.getOutputs().get(i).substring(0, Math.min(50, result.getOutputs().get(i).length())) + "...'");
		}

		logger.log(DEBUG, "✅ Batch inference test passed!");
	}

	@Test
	public void testAsyncInference() throws Exception {
		logger.log(DEBUG, "\n=== Async Inference Test ===");

		String prompt = "def fibonacci(n):";
		InferenceParameters params = new InferenceParameters(prompt)
			.setNPredict(30)
			.setTemperature(0.2f);

		CompletableFuture<String> future = InferencePatterns.asyncInference(model, prompt, params);

		Assert.assertFalse("Future should not be done immediately", future.isDone());

		String result = future.get(30, TimeUnit.SECONDS);

		Assert.assertNotNull("Result should not be null", result);
		Assert.assertFalse("Result should not be empty", result.isEmpty());

		logger.log(DEBUG, "Async result: " + result.substring(0, Math.min(100, result.length())) + "...");
		logger.log(DEBUG, "✅ Async inference test passed!");
	}

	@Test
	public void testChainOfThoughtInference() {
		logger.log(DEBUG, "\n=== Chain of Thought Inference Test ===");

		String question = "What is 15 + 27?";
		InferencePatterns.ChainOfThoughtConfig config = new InferencePatterns.ChainOfThoughtConfig(
			"Let me solve this step by step:",
			"Therefore, the answer is:",
			50,
			20
		);

		String result = InferencePatterns.chainOfThoughtInference(model, question, config);

		Assert.assertNotNull("Result should not be null", result);
		Assert.assertTrue("Result should contain reasoning section", result.contains("Reasoning:"));
		Assert.assertTrue("Result should contain answer section", result.contains("Answer:"));

		logger.log(DEBUG, "Chain of thought result:");
		logger.log(DEBUG, result);
		logger.log(DEBUG, "✅ Chain of thought test passed!");
	}

	@Test
	public void testTemplateInference() {
		logger.log(DEBUG, "\n=== Template Inference Test ===");

		String template = "Write a {language} function called {function_name} that {description}:";
		Map<String, String> variables = new HashMap<>();
		variables.put("language", "Python");
		variables.put("function_name", "add_numbers");
		variables.put("description", "adds two numbers");

		InferenceParameters params = new InferenceParameters("")
			.setNPredict(25)
			.setTemperature(0.1f);

		String result = InferencePatterns.templateInference(model, template, variables, params);

		Assert.assertNotNull("Result should not be null", result);
		Assert.assertFalse("Result should not be empty", result.isEmpty());

		logger.log(DEBUG, "Template filled: " + template.replace("{language}", "Python")
			.replace("{function_name}", "add_numbers")
			.replace("{description}", "adds two numbers"));
		logger.log(DEBUG, "Generated code: " + result.substring(0, Math.min(150, result.length())) + "...");
		logger.log(DEBUG, "✅ Template inference test passed!");
	}

	@Test
	public void testConsensusInference() {
		logger.log(DEBUG, "\n=== Consensus Inference Test ===");

		String prompt = "Complete this function: def is_even(n):";
		InferenceParameters baseParams = new InferenceParameters(prompt)
			.setNPredict(15)
			.setTemperature(0.5f);

		String consensus = InferencePatterns.consensusInference(model, prompt, 3, baseParams);

		Assert.assertNotNull("Consensus result should not be null", consensus);
		Assert.assertFalse("Consensus result should not be empty", consensus.isEmpty());

		logger.log(DEBUG, "Consensus result: " + consensus.substring(0, Math.min(100, consensus.length())) + "...");
		logger.log(DEBUG, "✅ Consensus inference test passed!");
	}

	@Test
	public void testProgressiveRefinement() {
		logger.log(DEBUG, "\n=== Progressive Refinement Test ===");

		String initialPrompt = "Write a short Python comment explaining what this does: x = [i for i in range(10) if i % 2 == 0]";
		InferenceParameters params = new InferenceParameters(initialPrompt)
			.setNPredict(30)
			.setTemperature(0.3f);

		String refined = InferencePatterns.progressiveRefinement(model, initialPrompt, 2, params);

		Assert.assertNotNull("Refined result should not be null", refined);
		Assert.assertFalse("Refined result should not be empty", refined.isEmpty());

		logger.log(DEBUG, "Progressively refined result: " + refined);
		logger.log(DEBUG, "✅ Progressive refinement test passed!");
	}

	@Test
	public void testStructuredDataExtraction() {
		logger.log(DEBUG, "\n=== Structured Data Extraction Test ===");

		String text = """
			Name: John Smith
			Age: 30
			Occupation: Software Developer
			Location: San Francisco
			Skills: Java, Python, JavaScript
			""";

		List<String> fields = Arrays.asList("Name", "Age", "Occupation", "Skills");
		Map<String, String> extracted = InferencePatterns.extractStructuredData(text, fields);

		Assert.assertEquals("Should extract name", "John Smith", extracted.get("Name"));
		Assert.assertEquals("Should extract age", "30", extracted.get("Age"));
		Assert.assertEquals("Should extract occupation", "Software Developer", extracted.get("Occupation"));
		Assert.assertEquals("Should extract skills", "Java, Python, JavaScript", extracted.get("Skills"));

		logger.log(DEBUG, "Extracted data:");
		extracted.forEach((key, value) -> logger.log(DEBUG, "  " + key + ": " + value));
		logger.log(DEBUG, "✅ Structured data extraction test passed!");
	}

	@Test
	public void testErrorHandling() {
		logger.log(DEBUG, "\n=== Error Handling Test ===");

		// Test null model
		try {
			InferencePatterns.batchInference(null, Arrays.asList("test"), null);
			Assert.fail("Should throw IllegalArgumentException for null model");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly caught null model: " + e.getMessage());
		}

		// Test empty prompts
		try {
			InferencePatterns.batchInference(model, Collections.emptyList(), null);
			Assert.fail("Should throw IllegalArgumentException for empty prompts");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly caught empty prompts: " + e.getMessage());
		}

		// Test invalid consensus parameters
		try {
			InferencePatterns.consensusInference(model, "test", 1, null);
			Assert.fail("Should throw IllegalArgumentException for numResponses < 2");
		} catch (IllegalArgumentException e) {
			logger.log(DEBUG, "✅ Correctly caught invalid consensus params: " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Error handling test passed!");
	}
}
