package de.kherud.llama.testing;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

public class TokenizerBenchmark {

	private static final System.Logger LOGGER = System.getLogger(TokenizerBenchmark.class.getName());

	private final LlamaModel model;
	private final String modelName;

	public TokenizerBenchmark(String modelPath, String modelName) {
		ModelParameters params = new ModelParameters()
			.setModel(modelPath)
			.setCtxSize(4096);
		this.model = new LlamaModel(params);
		this.modelName = modelName;
	}

	public TokenizerBenchmark(String modelPath) {
		this(modelPath, "Tokenizer");
	}

	public static class BenchmarkResult {
		public final String modelName;
		public final int totalTexts;
		public final int totalTokens;
		public final long encodeTimeNanos;
		public final long decodeTimeNanos;
		public final long totalTimeNanos;

		public BenchmarkResult(String modelName, int totalTexts, int totalTokens,
							  long encodeTimeNanos, long decodeTimeNanos) {
			this.modelName = modelName;
			this.totalTexts = totalTexts;
			this.totalTokens = totalTokens;
			this.encodeTimeNanos = encodeTimeNanos;
			this.decodeTimeNanos = decodeTimeNanos;
			this.totalTimeNanos = encodeTimeNanos + decodeTimeNanos;
		}

		public double getTextsPerSecond() {
			return (double) totalTexts / (totalTimeNanos / 1_000_000_000.0);
		}

		public double getTokensPerSecond() {
			return (double) totalTokens / (encodeTimeNanos / 1_000_000_000.0);
		}

		public double getEncodeLatencyMicros() {
			return totalTexts > 0 ? (encodeTimeNanos / 1000.0) / totalTexts : 0;
		}

		public double getDecodeLatencyMicros() {
			return totalTexts > 0 ? (decodeTimeNanos / 1000.0) / totalTexts : 0;
		}

		public double getAvgTokensPerText() {
			return totalTexts > 0 ? (double) totalTokens / totalTexts : 0;
		}

		@Override
		public String toString() {
			return String.format(
				"BenchmarkResult{model='%s', texts=%d, tokens=%d, " +
				"textsPerSec=%.1f, tokensPerSec=%.1f, " +
				"encodeLatency=%.2fμs, decodeLatency=%.2fμs, avgTokens=%.1f}",
				modelName, totalTexts, totalTokens,
				getTextsPerSecond(), getTokensPerSecond(),
				getEncodeLatencyMicros(), getDecodeLatencyMicros(), getAvgTokensPerText());
		}
	}

	public BenchmarkResult benchmark(String[] texts, int iterations) {
		return benchmark(Arrays.asList(texts), iterations);
	}

	public BenchmarkResult benchmark(List<String> texts, int iterations) {
		LOGGER.log(System.Logger.Level.INFO,String.format("Starting benchmark: %d texts, %d iterations", texts.size(), iterations));

		// Warmup
		warmup(texts, Math.min(iterations / 10, 10));

		long totalEncodeTime = 0;
		long totalDecodeTime = 0;
		int totalTokens = 0;
		int totalProcessed = 0;

		for (int iter = 0; iter < iterations; iter++) {
			for (String text : texts) {
				// Measure encoding
				long startEncode = System.nanoTime();
				int[] tokens = model.encode(text);
				long endEncode = System.nanoTime();
				totalEncodeTime += (endEncode - startEncode);

				// Measure decoding
				long startDecode = System.nanoTime();
				model.decode(tokens);
				long endDecode = System.nanoTime();
				totalDecodeTime += (endDecode - startDecode);

				totalTokens += tokens.length;
				totalProcessed++;
			}
		}

		BenchmarkResult result = new BenchmarkResult(
			modelName, totalProcessed, totalTokens, totalEncodeTime, totalDecodeTime);

		LOGGER.log(System.Logger.Level.INFO,"Benchmark completed: " + result);
		return result;
	}

	private void warmup(List<String> texts, int warmupIterations) {
		LOGGER.log(System.Logger.Level.INFO,"Warming up with " + warmupIterations + " iterations");
		for (int i = 0; i < warmupIterations; i++) {
			for (String text : texts) {
				int[] tokens = model.encode(text);
				model.decode(tokens);
			}
		}
	}

	public BenchmarkResult benchmarkStandardTexts(int iterations) {
		List<String> standardTexts = Arrays.asList(
			"Hello, world!",
			"This is a test sentence for tokenizer benchmarking.",
			"The quick brown fox jumps over the lazy dog.",
			"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
			"Natural language processing with large language models.",
			"Tokenization is the process of breaking text into smaller units.",
			"Machine learning algorithms can process and understand human language.",
			"Python, Java, and C++ are popular programming languages.",
			"The sun rises in the east and sets in the west.",
			"Technology continues to advance at an unprecedented pace.",
			"Artificial intelligence will transform many industries.",
			"Data science combines statistics, programming, and domain expertise.",
			"Cloud computing provides scalable infrastructure for modern applications.",
			"Open source software drives innovation across the technology industry.",
			"Cybersecurity is increasingly important in our connected world."
		);

		return benchmark(standardTexts, iterations);
	}

	public BenchmarkResult benchmarkByTextLength(int minLength, int maxLength, int samples, int iterations) {
		Random random = new Random(42);
		List<String> texts = new ArrayList<>();

		String baseText = "The quick brown fox jumps over the lazy dog. " +
						 "This pangram contains all letters of the alphabet. " +
						 "It is commonly used for testing typewriters and fonts. " +
						 "Lorem ipsum dolor sit amet consectetur adipiscing elit. ";

		for (int i = 0; i < samples; i++) {
			StringBuilder sb = new StringBuilder();
			int targetLength = random.nextInt(maxLength - minLength + 1) + minLength;

			while (sb.length() < targetLength) {
				sb.append(baseText);
			}

			texts.add(sb.substring(0, Math.min(targetLength, sb.length())));
		}

		LOGGER.log(System.Logger.Level.INFO,String.format("Benchmarking text lengths %d-%d chars with %d samples",
			minLength, maxLength, samples));

		return benchmark(texts, iterations);
	}

	public BenchmarkResult benchmarkSpecialTokens(int iterations) {
		List<String> specialTexts = Arrays.asList(
			"<s>Beginning of sequence</s>",
			"<|endoftext|>",
			"<unk>Unknown token</unk>",
			"<pad><pad><pad>",
			"Normal text with <special> tokens embedded",
			"Code: `function test() { return 'hello'; }`",
			"Math: ∑(i=1 to n) i² = n(n+1)(2n+1)/6",
			"Unicode: 你好世界 🌍 αβγ δεζ ηθι",
			"Mixed: English + 中文 + العربية + русский",
			"Emojis: 🚀🎉💻⭐🔥❤️🎯🌟",
			"Punctuation: !@#$%^&*()_+-=[]{}|;':\",./<>?",
			"Numbers: 1234567890 3.14159 1e-10 0xFF 0b1010",
			"URLs: https://example.com/path?query=value&param=123",
			"JSON: {\"key\": \"value\", \"array\": [1, 2, 3]}",
			"XML: <root><item id=\"1\">content</item></root>"
		);

		return benchmark(specialTexts, iterations);
	}

	public void runComprehensiveBenchmark() {
		LOGGER.log(System.Logger.Level.INFO,"Running comprehensive tokenizer benchmark for: " + modelName);

		// Standard texts
		BenchmarkResult standard = benchmarkStandardTexts(100);
		System.out.println("Standard texts: " + standard);

		// Short texts
		BenchmarkResult shortTexts = benchmarkByTextLength(1, 50, 50, 100);
		System.out.println("Short texts (1-50 chars): " + shortTexts);

		// Medium texts
		BenchmarkResult mediumTexts = benchmarkByTextLength(100, 500, 50, 50);
		System.out.println("Medium texts (100-500 chars): " + mediumTexts);

		// Long texts
		BenchmarkResult longTexts = benchmarkByTextLength(1000, 5000, 20, 20);
		System.out.println("Long texts (1000-5000 chars): " + longTexts);

		// Special tokens
		BenchmarkResult specialTokens = benchmarkSpecialTokens(100);
		System.out.println("Special tokens: " + specialTokens);

		// Summary
		System.out.println("\n=== COMPREHENSIVE BENCHMARK SUMMARY ===");
		System.out.printf("Model: %s%n", modelName);
		System.out.printf("%-20s: %8.1f texts/sec, %8.1f tokens/sec%n",
			"Standard", standard.getTextsPerSecond(), standard.getTokensPerSecond());
		System.out.printf("%-20s: %8.1f texts/sec, %8.1f tokens/sec%n",
			"Short", shortTexts.getTextsPerSecond(), shortTexts.getTokensPerSecond());
		System.out.printf("%-20s: %8.1f texts/sec, %8.1f tokens/sec%n",
			"Medium", mediumTexts.getTextsPerSecond(), mediumTexts.getTokensPerSecond());
		System.out.printf("%-20s: %8.1f texts/sec, %8.1f tokens/sec%n",
			"Long", longTexts.getTextsPerSecond(), longTexts.getTokensPerSecond());
		System.out.printf("%-20s: %8.1f texts/sec, %8.1f tokens/sec%n",
			"Special", specialTokens.getTextsPerSecond(), specialTokens.getTokensPerSecond());
	}

	public static void compareBenchmarks(BenchmarkResult baseline, BenchmarkResult comparison) {
		System.out.println("\n=== BENCHMARK COMPARISON ===");
		System.out.printf("Baseline (%s): %s%n", baseline.modelName, baseline);
		System.out.printf("Comparison (%s): %s%n", comparison.modelName, comparison);

		double speedupTexts = comparison.getTextsPerSecond() / baseline.getTextsPerSecond();
		double speedupTokens = comparison.getTokensPerSecond() / baseline.getTokensPerSecond();

		System.out.printf("Speedup: %.2fx texts/sec, %.2fx tokens/sec%n", speedupTexts, speedupTokens);

		if (speedupTexts > 1.1) {
			System.out.printf("%s is %.1f%% faster at processing texts%n",
				comparison.modelName, (speedupTexts - 1) * 100);
		} else if (speedupTexts < 0.9) {
			System.out.printf("%s is %.1f%% slower at processing texts%n",
				comparison.modelName, (1 - speedupTexts) * 100);
		} else {
			System.out.println("Performance is roughly equivalent");
		}
	}

	public void close() {
		if (model != null) {
			model.close();
		}
	}

	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(TokenizerBenchmark::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length < 1) {
			throw new IllegalArgumentException("Usage: TokenizerBenchmark <model_path> [model_name]");
		}

		String modelPath = args[0];
		String modelName = args.length > 1 ? args[1] : "Model";

		TokenizerBenchmark benchmark = new TokenizerBenchmark(modelPath, modelName);
		try {
			benchmark.runComprehensiveBenchmark();
		} finally {
			benchmark.close();
		}
	}
}
