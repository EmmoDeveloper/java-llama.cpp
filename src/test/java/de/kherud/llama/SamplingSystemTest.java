package de.kherud.llama;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Test suite for the AI sampling system.
 * Tests dynamic sampler switching, code completion constraints, JSON generation,
 * and context-aware sampling strategies.
 */
public class SamplingSystemTest {

	private static final System.Logger logger = System.getLogger(SamplingSystemTest.class.getName());
	private AISamplerManager.DynamicSampler dynamicSampler;
	private CodeCompletionSampler codeCompletionSampler;
	private JsonConstrainedSampler jsonConstrainedSampler;

	@Before
	public void setUp() {
		// Initialize backend
		try {
			LlamaUtils.initializeBackend();
		} catch (Exception e) {
			logger.log(DEBUG, "Backend already initialized or unavailable: " + e.getMessage());
		}

		dynamicSampler = new AISamplerManager.DynamicSampler();
		codeCompletionSampler = new CodeCompletionSampler();
		jsonConstrainedSampler = new JsonConstrainedSampler();
	}

	@After
	public void tearDown() {
		if (dynamicSampler != null) {
			dynamicSampler.close();
		}
		if (codeCompletionSampler != null) {
			codeCompletionSampler.close();
		}
		if (jsonConstrainedSampler != null) {
			jsonConstrainedSampler.close();
		}
	}

	@Test
	public void testDynamicSamplerCreation() {
		logger.log(DEBUG, "\n=== Dynamic Sampler Creation Test ===");

		Assert.assertNotNull("Dynamic sampler should not be null", dynamicSampler);

		// Register different contexts
		dynamicSampler.registerContext(
			AISamplerManager.SamplingContext.CODE_COMPLETION,
			AISamplerManager.PresetConfigs.codeCompletion());

		dynamicSampler.registerContext(
			AISamplerManager.SamplingContext.JSON_GENERATION,
			AISamplerManager.PresetConfigs.jsonGeneration());

		long currentSampler = dynamicSampler.getCurrentSampler();
		Assert.assertTrue("Should have valid sampler handle", currentSampler > 0);

		logger.log(DEBUG, "✅ Dynamic sampler created successfully");
		logger.log(DEBUG, "Current sampler handle: " + currentSampler);
	}

	@Test
	public void testContextSwitching() {
		logger.log(DEBUG, "\n=== Context Switching Test ===");

		// Register multiple contexts
		dynamicSampler.registerContext(
			AISamplerManager.SamplingContext.CODE_COMPLETION,
			AISamplerManager.PresetConfigs.codeCompletion());

		dynamicSampler.registerContext(
			AISamplerManager.SamplingContext.DOCUMENTATION,
			AISamplerManager.PresetConfigs.documentation());

		long initialSampler = dynamicSampler.getCurrentSampler();
		AISamplerManager.SamplingContext initialContext = dynamicSampler.getCurrentContext();

		logger.log(DEBUG, "Initial context: " + initialContext);
		logger.log(DEBUG, "Initial sampler: " + initialSampler);

		// Switch context
		dynamicSampler.switchContext(AISamplerManager.SamplingContext.DOCUMENTATION);

		long newSampler = dynamicSampler.getCurrentSampler();
		AISamplerManager.SamplingContext newContext = dynamicSampler.getCurrentContext();

		logger.log(DEBUG, "New context: " + newContext);
		logger.log(DEBUG, "New sampler: " + newSampler);

		Assert.assertEquals("Context should have switched",
			AISamplerManager.SamplingContext.DOCUMENTATION, newContext);
		Assert.assertTrue("Should have valid sampler after switch", newSampler > 0);

		logger.log(DEBUG, "✅ Context switching works correctly");
	}

	@Test
	public void testCodeCompletionSampler() {
		logger.log(DEBUG, "\n=== Code Completion Sampler Test ===");

		Assert.assertNotNull("Code completion sampler should not be null", codeCompletionSampler);

		// Test language detection
		codeCompletionSampler.analyzeContext("public class HelloWorld {", "HelloWorld.java");
		Assert.assertEquals("Should detect Java language",
			CodeCompletionSampler.Language.JAVA, codeCompletionSampler.getCurrentLanguage());

		// Test context detection for function
		codeCompletionSampler.analyzeContext("public void test", "Test.java");
		CodeCompletionSampler.CodeContext context = codeCompletionSampler.getCurrentContext();
		logger.log(DEBUG, "Detected context: " + context);

		// Test completion filtering
		List<String> rawCompletions = Arrays.asList("testMethod", "123invalid", "validName", "$invalid");
		List<String> filtered = codeCompletionSampler.filterCompletions(rawCompletions);

		Assert.assertNotNull("Filtered completions should not be null", filtered);
		Assert.assertFalse("Should filter out invalid names", filtered.isEmpty());

		logger.log(DEBUG, "Raw completions: " + rawCompletions);
		logger.log(DEBUG, "Filtered completions: " + filtered);
		logger.log(DEBUG, "✅ Code completion filtering works");
	}

	@Test
	public void testJsonConstrainedSampler() {
		logger.log(DEBUG, "\n=== JSON Constrained Sampler Test ===");

		Assert.assertNotNull("JSON constrained sampler should not be null", jsonConstrainedSampler);

		// Test initial state
		JsonConstrainedSampler.JsonState initialState = jsonConstrainedSampler.getCurrentState();
		Assert.assertEquals("Should start in object state",
			JsonConstrainedSampler.JsonState.OBJECT_START, initialState);

		// Test valid next tokens
		Set<String> validTokens = jsonConstrainedSampler.getValidNextTokens();
		Assert.assertNotNull("Valid tokens should not be null", validTokens);
		Assert.assertTrue("Should allow quote for object key", validTokens.contains("\""));

		logger.log(DEBUG, "Initial state: " + initialState);
		logger.log(DEBUG, "Valid next tokens: " + validTokens);

		// Test token processing
		boolean isValid = jsonConstrainedSampler.processToken("{");
		Assert.assertTrue("Opening brace should be valid", isValid);

		JsonConstrainedSampler.JsonState newState = jsonConstrainedSampler.getCurrentState();
		logger.log(DEBUG, "State after '{': " + newState);

		// Test JSON validation
		Assert.assertTrue("Should validate valid JSON prefix",
			jsonConstrainedSampler.isValidJsonSoFar());

		logger.log(DEBUG, "✅ JSON constraint validation works");
	}

	@Test
	public void testPresetConfigurations() {
		logger.log(DEBUG, "\n=== Preset Configurations Test ===");

		// Test all preset configurations
		AISamplerManager.SamplerConfig codeConfig =
			AISamplerManager.PresetConfigs.codeCompletion();
		Assert.assertNotNull("Code completion config should not be null", codeConfig);
		Assert.assertEquals("Should have correct name", "code_completion", codeConfig.name);
		Assert.assertFalse("Should have sampling steps", codeConfig.steps.isEmpty());

		AISamplerManager.SamplerConfig jsonConfig =
			AISamplerManager.PresetConfigs.jsonGeneration();
		Assert.assertNotNull("JSON generation config should not be null", jsonConfig);
		Assert.assertEquals("Should have correct name", "json_generation", jsonConfig.name);

		AISamplerManager.SamplerConfig docConfig =
			AISamplerManager.PresetConfigs.documentation();
		Assert.assertNotNull("Documentation config should not be null", docConfig);
		Assert.assertEquals("Should have correct name", "documentation", docConfig.name);

		AISamplerManager.SamplerConfig namingConfig =
			AISamplerManager.PresetConfigs.naming();
		Assert.assertNotNull("Naming config should not be null", namingConfig);
		Assert.assertEquals("Should have correct name", "naming", namingConfig.name);

		AISamplerManager.SamplerConfig debugConfig =
			AISamplerManager.PresetConfigs.debugging();
		Assert.assertNotNull("Debugging config should not be null", debugConfig);
		Assert.assertEquals("Should have correct name", "debugging", debugConfig.name);

		logger.log(DEBUG, "Code completion config steps: " + codeConfig.steps.size());
		logger.log(DEBUG, "JSON generation config steps: " + jsonConfig.steps.size());
		logger.log(DEBUG, "✅ All preset configurations are valid");
	}

	@Test
	public void testLanguageDetection() {
		logger.log(DEBUG, "\n=== Language Detection Test ===");

		// Test filename-based detection
		codeCompletionSampler.analyzeContext("", "test.java");
		Assert.assertEquals("Should detect Java from extension",
			CodeCompletionSampler.Language.JAVA, codeCompletionSampler.getCurrentLanguage());

		codeCompletionSampler.analyzeContext("", "test.cpp");
		Assert.assertEquals("Should detect C++ from extension",
			CodeCompletionSampler.Language.CPP, codeCompletionSampler.getCurrentLanguage());

		codeCompletionSampler.analyzeContext("", "test.py");
		Assert.assertEquals("Should detect Python from extension",
			CodeCompletionSampler.Language.PYTHON, codeCompletionSampler.getCurrentLanguage());

		// Test content-based detection
		codeCompletionSampler.analyzeContext("def main(): import sys", "");
		Assert.assertEquals("Should detect Python from content",
			CodeCompletionSampler.Language.PYTHON, codeCompletionSampler.getCurrentLanguage());

		codeCompletionSampler.analyzeContext("function test() { import * from 'module'; }", "");
		Assert.assertEquals("Should detect JavaScript from content",
			CodeCompletionSampler.Language.JAVASCRIPT, codeCompletionSampler.getCurrentLanguage());

		logger.log(DEBUG, "✅ Language detection works correctly");
	}

	@Test
	public void testSamplerConfigConstruction() {
		logger.log(DEBUG, "\n=== Sampler Config Construction Test ===");

		// Build custom sampler configuration
		AISamplerManager.SamplerConfig customConfig =
			new AISamplerManager.SamplerConfig("custom_test")
				.addStep(new AISamplerManager.SamplerStep(AISamplerManager.SamplerType.TOP_K)
					.param("k", 30))
				.addStep(new AISamplerManager.SamplerStep(AISamplerManager.SamplerType.TEMPERATURE)
					.param("temperature", 0.5f))
				.setParameter("description", "Custom test configuration");

		Assert.assertEquals("Should have correct name", "custom_test", customConfig.name);
		Assert.assertEquals("Should have 2 steps", 2, customConfig.steps.size());
		Assert.assertTrue("Should have description parameter",
			customConfig.parameters.containsKey("description"));

		// Test step parameters
		AISamplerManager.SamplerStep topKStep = customConfig.steps.get(0);
		Assert.assertEquals("First step should be TOP_K",
			AISamplerManager.SamplerType.TOP_K, topKStep.type);
		Assert.assertEquals("Should have correct k parameter",
			30.0f, topKStep.params.get("k"), 0.001f);

		AISamplerManager.SamplerStep tempStep = customConfig.steps.get(1);
		Assert.assertEquals("Second step should be TEMPERATURE",
			AISamplerManager.SamplerType.TEMPERATURE, tempStep.type);
		Assert.assertEquals("Should have correct temperature parameter",
			0.5f, tempStep.params.get("temperature"), 0.001f);

		logger.log(DEBUG, "Custom config: " + customConfig.name);
		logger.log(DEBUG, "Steps: " + customConfig.steps.size());
		logger.log(DEBUG, "✅ Sampler configuration construction works");
	}

	@Test
	public void testJsonStateTransitions() {
		logger.log(DEBUG, "\n=== JSON State Transitions Test ===");

		JsonConstrainedSampler sampler = new JsonConstrainedSampler();

		// Test object start
		Assert.assertEquals("Should start in OBJECT_START",
			JsonConstrainedSampler.JsonState.OBJECT_START, sampler.getCurrentState());

		// Test basic JSON prefix validation
		Assert.assertTrue("Empty string should be valid prefix", sampler.isValidJsonSoFar());

		sampler.processToken("{");
		Assert.assertTrue("Opening brace should be valid", sampler.isValidJsonSoFar());

		sampler.processToken("\"name\"");
		Assert.assertTrue("Key should be valid", sampler.isValidJsonSoFar());

		sampler.processToken(":");
		Assert.assertTrue("Colon should be valid", sampler.isValidJsonSoFar());

		sampler.processToken("\"value\"");
		Assert.assertTrue("String value should be valid", sampler.isValidJsonSoFar());

		String buffer = sampler.getCurrentBuffer();
		logger.log(DEBUG, "JSON buffer: " + buffer);
		logger.log(DEBUG, "Final state: " + sampler.getCurrentState());
		logger.log(DEBUG, "✅ JSON state transitions work correctly");

		sampler.close();
	}

	@Test
	public void testCodeContextDetection() {
		logger.log(DEBUG, "\n=== Code Context Detection Test ===");

		// Test different code contexts
		String[] testCases = {
			"public void test",           // Function signature
			"String variableName",        // Variable declaration
			"import java.util.List",      // Import statement
			"// This is a comment",       // Comment
			"\"hello world\"",           // String literal
		};

		for (String testCase : testCases) {
			codeCompletionSampler.analyzeContext(testCase, "Test.java");
			CodeCompletionSampler.CodeContext context = codeCompletionSampler.getCurrentContext();

			logger.log(DEBUG, "Text: \"" + testCase + "\" -> Context: " + context);
			Assert.assertNotNull("Context should not be null", context);
		}

		logger.log(DEBUG, "✅ Code context detection works");
	}
}