package de.kherud.llama.testing;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

public class TokenizerComparator {

	private static final System.Logger LOGGER = System.getLogger(TokenizerComparator.class.getName());

	private final LlamaModel model1;
	private final LlamaModel model2;
	private final String model1Name;
	private final String model2Name;

	public TokenizerComparator(String model1Path, String model2Path) {
		this(model1Path, model2Path, "Model1", "Model2");
	}

	public TokenizerComparator(String model1Path, String model2Path, String model1Name, String model2Name) {
		// Validate model paths
		validateModelPath(model1Path, model1Name);
		validateModelPath(model2Path, model2Name);

		ModelParameters params1 = new ModelParameters()
			.setModel(model1Path)
			.setCtxSize(4096);
		ModelParameters params2 = new ModelParameters()
			.setModel(model2Path)
			.setCtxSize(4096);

		this.model1 = new LlamaModel(params1);
		this.model2 = new LlamaModel(params2);
		this.model1Name = model1Name;
		this.model2Name = model2Name;
	}

	private void validateModelPath(String modelPath, String modelName) {
		if (modelPath == null || modelPath.isEmpty()) {
			throw new IllegalArgumentException(modelName + " path cannot be null or empty");
		}

		java.io.File modelFile = new java.io.File(modelPath);
		if (!modelFile.exists()) {
			throw new IllegalArgumentException(modelName + " file does not exist: " + modelPath);
		}
		if (!modelFile.isFile()) {
			throw new IllegalArgumentException(modelName + " path is not a file: " + modelPath);
		}
		if (modelFile.length() < 1024) { // GGUF files are at least several KB
			throw new IllegalArgumentException(modelName + " file is too small to be valid: " + modelFile.length() + " bytes");
		}
	}

	public static class ComparisonResult {
		public int totalTests = 0;
		public int encodeMatches = 0;
		public int decodeMatches = 0;
		public final List<MismatchReport> encodeMismatches = new ArrayList<>();
		public final List<MismatchReport> decodeMismatches = new ArrayList<>();
		public long model1EncodeTime = 0;
		public long model2EncodeTime = 0;
		public long model1DecodeTime = 0;
		public long model2DecodeTime = 0;

		public double getEncodeAccuracy() {
			return totalTests > 0 ? (double) encodeMatches / totalTests : 0.0;
		}

		public double getDecodeAccuracy() {
			return totalTests > 0 ? (double) decodeMatches / totalTests : 0.0;
		}

		@Override
		public String toString() {
			return String.format(
				"ComparisonResult{tests=%d, encodeAcc=%.2f%%, decodeAcc=%.2f%%, " +
				"encodeMismatches=%d, decodeMismatches=%d}",
				totalTests, getEncodeAccuracy() * 100, getDecodeAccuracy() * 100,
				encodeMismatches.size(), decodeMismatches.size());
		}
	}

	public static class MismatchReport {
		public final String input;
		public final Object expected;
		public final Object actual;
		public final int firstDifferenceIndex;

		public MismatchReport(String input, Object expected, Object actual, int firstDifferenceIndex) {
			this.input = input;
			this.expected = expected;
			this.actual = actual;
			this.firstDifferenceIndex = firstDifferenceIndex;
		}

		@Override
		public String toString() {
			return String.format("Mismatch at index %d - Input: %s\nExpected: %s\nActual: %s",
				firstDifferenceIndex, escapeString(input), expected, actual);
		}

		private String escapeString(String str) {
			return str.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r");
		}
	}

	public ComparisonResult compareTokenizers(String[] testTexts) {
		return compareTokenizers(Arrays.stream(testTexts));
	}

	public ComparisonResult compareTokenizers(java.util.stream.Stream<String> testTexts) {
		ComparisonResult result = new ComparisonResult();

		testTexts.forEach(text -> {
			result.totalTests++;

			try {
				// Compare encoding
				long start = System.nanoTime();
				int[] tokens1 = model1.encode(text);
				result.model1EncodeTime += (System.nanoTime() - start) / 1_000_000;

				start = System.nanoTime();
				int[] tokens2 = model2.encode(text);
				result.model2EncodeTime += (System.nanoTime() - start) / 1_000_000;

				if (Arrays.equals(tokens1, tokens2)) {
					result.encodeMatches++;
				} else {
					int mismatchIndex = findFirstMismatch(tokens1, tokens2);
					result.encodeMismatches.add(new MismatchReport(
						text, Arrays.toString(tokens1), Arrays.toString(tokens2), mismatchIndex));
				}

				// Compare decoding (using tokens from model1)
				start = System.nanoTime();
				String decoded1 = model1.decode(tokens1);
				result.model1DecodeTime += (System.nanoTime() - start) / 1_000_000;

				start = System.nanoTime();
				String decoded2 = model2.decode(tokens1);
				result.model2DecodeTime += (System.nanoTime() - start) / 1_000_000;

				if (decoded1.equals(decoded2)) {
					result.decodeMatches++;
				} else {
					int mismatchIndex = findFirstMismatch(decoded1, decoded2);
					result.decodeMismatches.add(new MismatchReport(
						text, decoded1, decoded2, mismatchIndex));
				}

			} catch (Exception e) {
				LOGGER.log(System.Logger.Level.WARNING,"Error processing text: " + escapeString(text) + " - " + e.getMessage());
			}
		});

		return result;
	}

	private int findFirstMismatch(int[] arr1, int[] arr2) {
		int minLength = Math.min(arr1.length, arr2.length);
		for (int i = 0; i < minLength; i++) {
			if (arr1[i] != arr2[i]) {
				return i;
			}
		}
		return arr1.length == arr2.length ? -1 : minLength;
	}

	private int findFirstMismatch(String str1, String str2) {
		int minLength = Math.min(str1.length(), str2.length());
		for (int i = 0; i < minLength; i++) {
			if (str1.charAt(i) != str2.charAt(i)) {
				return i;
			}
		}
		return str1.length() == str2.length() ? -1 : minLength;
	}

	private String escapeString(String str) {
		return str.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r");
	}

	public ComparisonResult runStandardComparison() {
		LOGGER.log(System.Logger.Level.INFO,"Running standard tokenizer comparison...");

		List<String> testTexts = new ArrayList<>();

		// Basic tests
		testTexts.addAll(Arrays.asList(
			"", " ", "  ", "\t", "\n",
			"Hello world", " Hello world", "Hello, world!",
			"This is a test", "Testing 123",
			"🦙 llama.cpp tokenizer test",
			"Mixed English and 中文 text",
			"Code: int main() { return 0; }"
		));

		// Edge cases
		testTexts.addAll(Arrays.asList(
			"<s>", "</s>", "<unk>", "<pad>",
			"<s>Hello</s>", "<|endoftext|>",
			"\"quoted text\"", "'single quotes'",
			"Numbers: 123456789",
			"Special chars: !@#$%^&*()",
			"Unicode: αβγ δεζ ηθι"
		));

		// Random combinations
		Random random = new Random(42);
		String chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t";
		for (int i = 0; i < 100; i++) {
			StringBuilder sb = new StringBuilder();
			int length = random.nextInt(20) + 1;
			for (int j = 0; j < length; j++) {
				sb.append(chars.charAt(random.nextInt(chars.length())));
			}
			testTexts.add(sb.toString());
		}

		ComparisonResult result = compareTokenizers(testTexts.stream());

		LOGGER.log(System.Logger.Level.INFO,String.format("Comparison completed: %s", result));
		LOGGER.log(System.Logger.Level.INFO,String.format("%s encode time: %dms", model1Name, result.model1EncodeTime));
		LOGGER.log(System.Logger.Level.INFO,String.format("%s encode time: %dms", model2Name, result.model2EncodeTime));
		LOGGER.log(System.Logger.Level.INFO,String.format("%s decode time: %dms", model1Name, result.model1DecodeTime));
		LOGGER.log(System.Logger.Level.INFO,String.format("%s decode time: %dms", model2Name, result.model2DecodeTime));

		return result;
	}

	public void printDetailedReport(ComparisonResult result, int maxSamples) {
		System.out.println("\n=== TOKENIZER COMPARISON REPORT ===");
		System.out.println(result);
		System.out.println();

		if (!result.encodeMismatches.isEmpty()) {
			System.out.println("ENCODE MISMATCHES (showing up to " + maxSamples + "):");
			result.encodeMismatches.stream()
				.limit(maxSamples)
				.forEach(System.out::println);
			System.out.println();
		}

		if (!result.decodeMismatches.isEmpty()) {
			System.out.println("DECODE MISMATCHES (showing up to " + maxSamples + "):");
			result.decodeMismatches.stream()
				.limit(maxSamples)
				.forEach(System.out::println);
			System.out.println();
		}

		System.out.println("PERFORMANCE COMPARISON:");
		System.out.printf("%-20s: %6dms (encode), %6dms (decode)%n",
			model1Name, result.model1EncodeTime, result.model1DecodeTime);
		System.out.printf("%-20s: %6dms (encode), %6dms (decode)%n",
			model2Name, result.model2EncodeTime, result.model2DecodeTime);

		double encodeSpeedup = (double) result.model1EncodeTime / result.model2EncodeTime;
		double decodeSpeedup = (double) result.model1DecodeTime / result.model2DecodeTime;
		System.out.printf("Speedup (%s vs %s): %.2fx (encode), %.2fx (decode)%n",
			model2Name, model1Name, encodeSpeedup, decodeSpeedup);
	}

	public void close() {
		if (model1 != null) model1.close();
		if (model2 != null) model2.close();
	}

	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(TokenizerComparator::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length < 2 || args.length > 4) {
			System.err.println("Usage: TokenizerComparator <model1_path> <model2_path> [model1_name] [model2_name]");
			throw new IllegalArgumentException("Invalid number of arguments");
		}

		String model1Path = args[0];
		String model2Path = args[1];
		String model1Name = args.length > 2 ? args[2] : "Model1";
		String model2Name = args.length > 3 ? args[3] : "Model2";

		TokenizerComparator comparator = new TokenizerComparator(
			model1Path, model2Path, model1Name, model2Name);

		try {
			ComparisonResult result = comparator.runStandardComparison();
			comparator.printDetailedReport(result, 10);
		} finally {
			comparator.close();
		}
	}
}
