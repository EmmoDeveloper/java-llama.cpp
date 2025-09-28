package de.kherud.llama;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * AI sampler management system for IDE integration.
 * Provides dynamic sampler switching, context-aware sampling, and specialized
 * sampling strategies for code completion, JSON generation, and other AI IDE tasks.
 */
public class AISamplerManager {

	/**
	 * Context types for different AI IDE scenarios
	 */
	public enum SamplingContext {
		CODE_COMPLETION,
		JSON_GENERATION,
		DOCUMENTATION,
		VARIABLE_NAME,
		FUNCTION_NAME,
		COMMENT_GENERATION,
		DEBUGGING,
		REFACTORING,
		GENERAL
	}

	/**
	 * Predefined sampler configurations for different contexts
	 */
	public static class SamplerConfig {
		public final String name;
		public final List<SamplerStep> steps;
		public final Map<String, Object> parameters;

		public SamplerConfig(String name) {
			this.name = name;
			this.steps = new ArrayList<>();
			this.parameters = new HashMap<>();
		}

		public SamplerConfig addStep(SamplerStep step) {
			steps.add(step);
			return this;
		}

		public SamplerConfig setParameter(String key, Object value) {
			parameters.put(key, value);
			return this;
		}
	}

	/**
	 * Individual sampler step in a chain
	 */
	public static class SamplerStep {
		public final SamplerType type;
		public final Map<String, Float> params;

		public SamplerStep(SamplerType type) {
			this.type = type;
			this.params = new HashMap<>();
		}

		public SamplerStep param(String key, float value) {
			params.put(key, value);
			return this;
		}
	}

	public enum SamplerType {
		TEMPERATURE,
		TOP_K,
		TOP_P,
		MIN_P,
		TYPICAL,
		XTC,
		PENALTIES,
		MIROSTAT_V2,
		GREEDY,
		DISTRIBUTION
	}

	/**
	 * Dynamic sampler that switches based on context
	 */
	public static class DynamicSampler implements AutoCloseable {
		private final Map<SamplingContext, Long> contextSamplers;
		private final Map<SamplingContext, SamplerConfig> configs;
		private SamplingContext currentContext;
		private long currentSamplerHandle;

		public DynamicSampler() {
			this.contextSamplers = new HashMap<>();
			this.configs = new HashMap<>();
			this.currentContext = SamplingContext.GENERAL;
		}

		public void registerContext(SamplingContext context, SamplerConfig config) {
			configs.put(context, config);
			long samplerHandle = buildSampler(config);
			contextSamplers.put(context, samplerHandle);

			if (currentSamplerHandle == 0) {
				currentSamplerHandle = samplerHandle;
			}
		}

		public void switchContext(SamplingContext newContext) {
			if (contextSamplers.containsKey(newContext)) {
				currentContext = newContext;
				currentSamplerHandle = contextSamplers.get(newContext);
			}
		}

		public long getCurrentSampler() {
			return currentSamplerHandle;
		}

		public SamplingContext getCurrentContext() {
			return currentContext;
		}

		@Override
		public void close() {
			for (long handle : contextSamplers.values()) {
				LlamaSampler.free(handle);
			}
			contextSamplers.clear();
		}

		private long buildSampler(SamplerConfig config) {
			long chain = LlamaSampler.createChain();

			for (SamplerStep step : config.steps) {
				long samplerHandle = createSamplerFromStep(step);
				LlamaSampler.addToChain(chain, samplerHandle);
			}

			return chain;
		}

		private long createSamplerFromStep(SamplerStep step) {
			switch (step.type) {
				case TEMPERATURE:
					return LlamaSampler.createTemperature(
						step.params.getOrDefault("temperature", 0.7f));

				case TOP_K:
					return LlamaSampler.createTopK(
						step.params.getOrDefault("k", 40f).intValue());

				case TOP_P:
					return LlamaSampler.createTopP(
						step.params.getOrDefault("p", 0.9f),
						step.params.getOrDefault("min_keep", 1f).intValue());

				case MIN_P:
					return LlamaSampler.createMinP(
						step.params.getOrDefault("p", 0.05f),
						step.params.getOrDefault("min_keep", 1f).intValue());

				case TYPICAL:
					return LlamaSampler.createTypical(
						step.params.getOrDefault("p", 0.95f),
						step.params.getOrDefault("min_keep", 1f).intValue());

				case XTC:
					return LlamaSampler.createXtc(
						step.params.getOrDefault("p", 0.5f),
						step.params.getOrDefault("threshold", 0.1f),
						step.params.getOrDefault("min_keep", 1f).intValue(),
						step.params.getOrDefault("seed", 42f).intValue());

				case PENALTIES:
					return LlamaSampler.createPenalties(
						step.params.getOrDefault("last_n", 64f).intValue(),
						step.params.getOrDefault("repeat", 1.1f),
						step.params.getOrDefault("freq", 0.0f),
						step.params.getOrDefault("present", 0.0f));

				case MIROSTAT_V2:
					return LlamaSampler.createMirostatV2(
						step.params.getOrDefault("seed", 42f).intValue(),
						step.params.getOrDefault("tau", 5.0f),
						step.params.getOrDefault("eta", 0.1f));

				case GREEDY:
					return LlamaSampler.createGreedy();

				case DISTRIBUTION:
					return LlamaSampler.createDistribution(
						step.params.getOrDefault("seed", 42f).intValue());

				default:
					throw new IllegalArgumentException("Unknown sampler type: " + step.type);
			}
		}
	}

	/**
	 * Context-aware sampler that automatically detects context from text
	 */
	public static class ContextAwareSampler {
		private final DynamicSampler dynamicSampler;
		private final List<ContextDetector> detectors;

		public ContextAwareSampler(DynamicSampler dynamicSampler) {
			this.dynamicSampler = dynamicSampler;
			this.detectors = createDefaultDetectors();
		}

		public void processText(String text) {
			for (ContextDetector detector : detectors) {
				if (detector.matches.test(text)) {
					dynamicSampler.switchContext(detector.context);
					break;
				}
			}
		}

		public long getCurrentSampler() {
			return dynamicSampler.getCurrentSampler();
		}

		private List<ContextDetector> createDefaultDetectors() {
			List<ContextDetector> detectors = new ArrayList<>();

			// Code completion context
			detectors.add(new ContextDetector(
				SamplingContext.CODE_COMPLETION,
				text -> text.matches(".*\\.(java|cpp|py|js|ts)$") ||
						text.contains("function ") ||
						text.contains("class ") ||
						text.contains("def ") ||
						text.contains("import ")
			));

			// JSON generation context
			detectors.add(new ContextDetector(
				SamplingContext.JSON_GENERATION,
				text -> text.contains("json") ||
						text.contains("{") ||
						text.contains("\"") && text.contains(":")
			));

			// Variable name context
			detectors.add(new ContextDetector(
				SamplingContext.VARIABLE_NAME,
				text -> text.matches(".*\\b(let|var|const|int|String|float|double)\\s+\\w*$")
			));

			// Function name context
			detectors.add(new ContextDetector(
				SamplingContext.FUNCTION_NAME,
				text -> text.matches(".*\\b(function|def|public|private|static)\\s+\\w*$")
			));

			// Documentation context
			detectors.add(new ContextDetector(
				SamplingContext.DOCUMENTATION,
				text -> text.contains("/**") ||
						text.contains("//") ||
						text.contains("# ") ||
						text.contains("doc:")
			));

			return detectors;
		}

		private static class ContextDetector {
			final SamplingContext context;
			final Predicate<String> matches;

			ContextDetector(SamplingContext context, Predicate<String> matches) {
				this.context = context;
				this.matches = matches;
			}
		}
	}

	/**
	 * Predefined configurations for different AI IDE scenarios
	 */
	public static class PresetConfigs {

		/**
		 * Precise sampling for code completion - favors correct syntax
		 */
		public static SamplerConfig codeCompletion() {
			return new SamplerConfig("code_completion")
				.addStep(new SamplerStep(SamplerType.PENALTIES)
					.param("last_n", 256)
					.param("repeat", 1.15f)
					.param("freq", 0.1f)
					.param("present", 0.0f))
				.addStep(new SamplerStep(SamplerType.TOP_K)
					.param("k", 20))
				.addStep(new SamplerStep(SamplerType.TOP_P)
					.param("p", 0.85f)
					.param("min_keep", 1))
				.addStep(new SamplerStep(SamplerType.TEMPERATURE)
					.param("temperature", 0.3f))
				.setParameter("description", "Precise sampling for code completion");
		}

		/**
		 * Structured sampling for JSON generation - ensures valid JSON
		 */
		public static SamplerConfig jsonGeneration() {
			return new SamplerConfig("json_generation")
				.addStep(new SamplerStep(SamplerType.PENALTIES)
					.param("last_n", 128)
					.param("repeat", 1.05f)
					.param("freq", 0.0f)
					.param("present", 0.0f))
				.addStep(new SamplerStep(SamplerType.TOP_P)
					.param("p", 0.9f)
					.param("min_keep", 1))
				.addStep(new SamplerStep(SamplerType.TEMPERATURE)
					.param("temperature", 0.2f))
				.setParameter("description", "Structured sampling for JSON generation");
		}

		/**
		 * Creative sampling for documentation - allows more variety
		 */
		public static SamplerConfig documentation() {
			return new SamplerConfig("documentation")
				.addStep(new SamplerStep(SamplerType.PENALTIES)
					.param("last_n", 128)
					.param("repeat", 1.1f)
					.param("freq", 0.05f)
					.param("present", 0.0f))
				.addStep(new SamplerStep(SamplerType.TOP_K)
					.param("k", 50))
				.addStep(new SamplerStep(SamplerType.TOP_P)
					.param("p", 0.95f)
					.param("min_keep", 2))
				.addStep(new SamplerStep(SamplerType.TEMPERATURE)
					.param("temperature", 0.6f))
				.setParameter("description", "Creative sampling for documentation");
		}

		/**
		 * Conservative sampling for variable/function names - highly deterministic
		 */
		public static SamplerConfig naming() {
			return new SamplerConfig("naming")
				.addStep(new SamplerStep(SamplerType.PENALTIES)
					.param("last_n", 64)
					.param("repeat", 1.2f)
					.param("freq", 0.2f)
					.param("present", 0.0f))
				.addStep(new SamplerStep(SamplerType.TOP_K)
					.param("k", 10))
				.addStep(new SamplerStep(SamplerType.TEMPERATURE)
					.param("temperature", 0.1f))
				.setParameter("description", "Conservative sampling for naming");
		}

		/**
		 * Balanced sampling for debugging and refactoring
		 */
		public static SamplerConfig debugging() {
			return new SamplerConfig("debugging")
				.addStep(new SamplerStep(SamplerType.PENALTIES)
					.param("last_n", 128)
					.param("repeat", 1.1f)
					.param("freq", 0.1f)
					.param("present", 0.0f))
				.addStep(new SamplerStep(SamplerType.TOP_K)
					.param("k", 30))
				.addStep(new SamplerStep(SamplerType.TOP_P)
					.param("p", 0.9f)
					.param("min_keep", 1))
				.addStep(new SamplerStep(SamplerType.TEMPERATURE)
					.param("temperature", 0.4f))
				.setParameter("description", "Balanced sampling for debugging");
		}
	}
}
