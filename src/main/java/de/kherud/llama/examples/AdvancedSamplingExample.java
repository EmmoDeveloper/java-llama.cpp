package de.kherud.llama.examples;

import de.kherud.llama.AdvancedSamplerManager;
import de.kherud.llama.CodeCompletionSampler;
import de.kherud.llama.JsonConstrainedSampler;

import java.util.Arrays;
import java.util.List;

/**
 * Example demonstrating the Advanced Sampling Ecosystem for AI IDE integration.
 * Shows how to use dynamic samplers, code completion sampling, and JSON constraints.
 */
public class AdvancedSamplingExample {

	public static void main(String[] args) {
		demonstrateDynamicSampling();
		demonstrateCodeCompletion();
		demonstrateJsonGeneration();
	}

	/**
	 * Demonstrates dynamic sampler switching based on context
	 */
	public static void demonstrateDynamicSampling() {
		System.out.println("=== Dynamic Sampling Demo ===");

		try (AdvancedSamplerManager.DynamicSampler dynamicSampler =
				new AdvancedSamplerManager.DynamicSampler()) {

			// Register different contexts with specialized samplers
			dynamicSampler.registerContext(
				AdvancedSamplerManager.SamplingContext.CODE_COMPLETION,
				AdvancedSamplerManager.PresetConfigs.codeCompletion());

			dynamicSampler.registerContext(
				AdvancedSamplerManager.SamplingContext.JSON_GENERATION,
				AdvancedSamplerManager.PresetConfigs.jsonGeneration());

			dynamicSampler.registerContext(
				AdvancedSamplerManager.SamplingContext.DOCUMENTATION,
				AdvancedSamplerManager.PresetConfigs.documentation());

			// Demonstrate context switching
			System.out.println("Initial context: " + dynamicSampler.getCurrentContext());
			System.out.println("Initial sampler: " + dynamicSampler.getCurrentSampler());

			dynamicSampler.switchContext(AdvancedSamplerManager.SamplingContext.JSON_GENERATION);
			System.out.println("Switched to JSON generation");
			System.out.println("New sampler: " + dynamicSampler.getCurrentSampler());

			dynamicSampler.switchContext(AdvancedSamplerManager.SamplingContext.DOCUMENTATION);
			System.out.println("Switched to documentation");
			System.out.println("New sampler: " + dynamicSampler.getCurrentSampler());

		} catch (Exception e) {
			System.err.println("Error in dynamic sampling: " + e.getMessage());
		}

		System.out.println();
	}

	/**
	 * Demonstrates AI IDE code completion with context awareness
	 */
	public static void demonstrateCodeCompletion() {
		System.out.println("=== Code Completion Sampling Demo ===");

		try (CodeCompletionSampler codeCompletion = new CodeCompletionSampler()) {

			// Demonstrate language detection
			codeCompletion.analyzeContext("public class HelloWorld {", "HelloWorld.java");
			System.out.println("Detected language: " + codeCompletion.getCurrentLanguage());
			System.out.println("Current context: " + codeCompletion.getCurrentContext());

			// Test different code contexts
			String[] testCases = {
				"public void method", // Function signature
				"String variable",    // Variable declaration
				"import java.util",   // Import statement
				"// Comment here"     // Documentation
			};

			for (String testCase : testCases) {
				codeCompletion.analyzeContext(testCase, "Test.java");
				System.out.println("\"" + testCase + "\" -> Context: " +
					codeCompletion.getCurrentContext());
			}

			// Demonstrate completion filtering
			List<String> rawCompletions = Arrays.asList(
				"methodName", "123invalid", "validName", "_privateField", "$invalid", "getName"
			);

			codeCompletion.analyzeContext("public String get", "Test.java");
			List<String> filtered = codeCompletion.filterCompletions(rawCompletions);

			System.out.println("Raw completions: " + rawCompletions);
			System.out.println("Filtered for function context: " + filtered);

		} catch (Exception e) {
			System.err.println("Error in code completion: " + e.getMessage());
		}

		System.out.println();
	}

	/**
	 * Demonstrates JSON constrained generation
	 */
	public static void demonstrateJsonGeneration() {
		System.out.println("=== JSON Constrained Generation Demo ===");

		try (JsonConstrainedSampler jsonSampler = new JsonConstrainedSampler()) {

			// Demonstrate JSON state tracking
			System.out.println("Initial state: " + jsonSampler.getCurrentState());
			System.out.println("Valid next tokens: " + jsonSampler.getValidNextTokens());

			// Simulate JSON generation
			String[] tokens = {"{", "\"name\"", ":", "\"John\"", ",", "\"age\"", ":", "25"};

			for (String token : tokens) {
				boolean valid = jsonSampler.processToken(token);
				System.out.println("Token: " + token +
					" -> State: " + jsonSampler.getCurrentState() +
					" (Valid: " + valid + ")");
			}

			System.out.println("Current JSON buffer: " + jsonSampler.getCurrentBuffer());
			System.out.println("Is valid JSON so far: " + jsonSampler.isValidJsonSoFar());

		} catch (Exception e) {
			System.err.println("Error in JSON generation: " + e.getMessage());
		}

		System.out.println();
	}

	/**
	 * Example of building custom sampler configurations
	 */
	public static AdvancedSamplerManager.SamplerConfig createCustomConfig() {
		return new AdvancedSamplerManager.SamplerConfig("custom_ai_ide")
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.PENALTIES)
				.param("last_n", 128)
				.param("repeat", 1.1f)
				.param("freq", 0.1f))
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.TOP_K)
				.param("k", 25))
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.TOP_P)
				.param("p", 0.85f))
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.TEMPERATURE)
				.param("temperature", 0.4f))
			.setParameter("description", "Custom AI IDE optimized sampling");
	}
}