package de.kherud.llama.testing;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.logging.Logger;
import java.util.stream.Stream;

public class TokenizerTester {

	private static final Logger LOGGER = Logger.getLogger(TokenizerTester.class.getName());
	private static final int MAX_ERRORS = 10;

	private final LlamaModel model;
	private final TokenizerStats stats;

	public TokenizerTester(String modelPath) {
		ModelParameters params = new ModelParameters()
			.setModel(modelPath)
			.setCtxSize(4096);
		this.model = new LlamaModel(params);
		this.stats = new TokenizerStats();
	}

	public static class TokenizerStats {
		public int totalTests = 0;
		public int encodeErrors = 0;
		public int decodeErrors = 0;
		public long encodeTime = 0;
		public long decodeTime = 0;
		public final List<String> errorSamples = new ArrayList<>();

		public void reset() {
			totalTests = 0;
			encodeErrors = 0;
			decodeErrors = 0;
			encodeTime = 0;
			decodeTime = 0;
			errorSamples.clear();
		}

		@Override
		public String toString() {
			return String.format("TokenizerStats{tests=%d, encodeErrors=%d, decodeErrors=%d, " +
					"encodeTime=%dms, decodeTime=%dms}",
				totalTests, encodeErrors, decodeErrors, encodeTime, decodeTime);
		}
	}

	public void testBasicTokenization() {
		LOGGER.info("Testing basic tokenization...");
		String[] testCases = {
			"",
			" ",
			"  ",
			"   ",
			"\t",
			"\n",
			"\n\n",
			"\t\n",
			"Hello world",
			" Hello world",
			"Hello World",
			" Hello World!",
			"Hello, world!",
			" this is 🦙.cpp",
			"нещо на Български",
			"🚀 (normal) 😶‍🌫️ (multiple emojis) ✅",
			"Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
			"3",
			"33",
			"333",
			"33333",
			"333333333"
		};

		testTokenization("Basic Tests", Arrays.stream(testCases));
	}

	public void testEdgeCases() {
		LOGGER.info("Testing edge cases...");
		String[] edgeCases = {
			"\u001f-a",      // unicode control
			"¼-a",           // unicode digit
			"½-a",           // unicode digit
			"¾-a",           // unicode digit
			"a 〇b",         // unicode digit
			"Ⅵ-a",          // unicode digit
			"\uFEFF//",      // BOM
			"Cửa Việt",      // complex unicode
			"<s>a",          // special tokens
			"<unk><|endoftext|><s>",
			"a\na",          // newlines
			"\"`",           // quotes
			" \u2e4e",       // unicode punctuation
			"\n\u000b  ",    // mixed whitespace
			"a\u00a0\u00a0\u0000b", // non-breaking space + null
			"one <mask>",    // special tokens
			"a </s> b",      // tokens with spaces
			"a <mask> b",    // masked tokens
			"\u00a0aC",      // non-breaking space
			"\u2029 \ua3e4", // unicode separators
			"a ?",           // question mark
			"å",             // accented character
			"<s><s><unk><s>a<s>b<s>c<unk>d<unk></s>",
			"<s> <s> <unk><s>a<s>b<s>c<unk>d<unk></s>"
		};

		testTokenization("Edge Cases", Arrays.stream(edgeCases));
	}

	public void testAsciiCharacters() {
		LOGGER.info("Testing ASCII character combinations...");
		String[] whitespaces = {"", " ", "  "};

		Stream<String> asciiTests = generateAsciiCombinations(whitespaces, 500);
		testTokenization("ASCII Tests", asciiTests);
	}

	public void testUnicodeCharacters() {
		LOGGER.info("Testing Unicode characters...");
		Stream<String> unicodeTests = generateUnicodeTests(200);
		testTokenization("Unicode Tests", unicodeTests);
	}

	public void testRandomText() {
		LOGGER.info("Testing random text generation...");
		Stream<String> randomTests = generateRandomText(100);
		testTokenization("Random Tests", randomTests);
	}

	private void testTokenization(String testName, Stream<String> testCases) {
		AtomicInteger testCount = new AtomicInteger(0);
		AtomicInteger errors = new AtomicInteger(0);

		testCases.forEach(text -> {
			testCount.incrementAndGet();
			stats.totalTests++;

			try {
				// Test encode
				long start = System.nanoTime();
				int[] tokens = model.encode(text);
				stats.encodeTime += (System.nanoTime() - start) / 1_000_000;

				// Test decode
				start = System.nanoTime();
				String decoded = model.decode(tokens);
				stats.decodeTime += (System.nanoTime() - start) / 1_000_000;

				// Validate round-trip
				if (!validateRoundTrip(text, decoded)) {
					stats.decodeErrors++;
					errors.incrementAndGet();
					if (stats.errorSamples.size() < MAX_ERRORS) {
						stats.errorSamples.add(String.format(
							"Decode mismatch - Original: %s, Decoded: %s",
							escapeString(text), escapeString(decoded)));
					}
				}

			} catch (Exception e) {
				stats.encodeErrors++;
				errors.incrementAndGet();
				if (stats.errorSamples.size() < MAX_ERRORS) {
					stats.errorSamples.add(String.format(
						"Exception on text: %s - %s", escapeString(text), e.getMessage()));
				}
			}

			if (errors.get() >= MAX_ERRORS) {
				LOGGER.warning("Max errors reached for " + testName);
				return;
			}
		});

		LOGGER.info(String.format("%s completed: %d tests, %d errors",
			testName, testCount.get(), errors.get()));
	}

	private boolean validateRoundTrip(String original, String decoded) {
		if (original.equals(decoded)) {
			return true;
		}

		// Handle common tokenizer variations
		String normalized = decoded;

		// Remove potential BOS/EOS tokens from decoded text
		if (normalized.startsWith("<s>")) {
			normalized = normalized.substring(3);
		}
		if (normalized.endsWith("</s>")) {
			normalized = normalized.substring(0, normalized.length() - 4);
		}

		// Handle Unicode normalization differences
		return original.equals(normalized) ||
			   original.trim().equals(normalized.trim());
	}

	private String escapeString(String str) {
		if (str == null) return "null";
		return str.replace("\n", "\\n")
				 .replace("\t", "\\t")
				 .replace("\r", "\\r")
				 .replace("\"", "\\\"");
	}

	private Stream<String> generateAsciiCombinations(String[] whitespaces, int limit) {
		List<String> combinations = new ArrayList<>();
		Random random = new Random(42); // Fixed seed for reproducibility

		for (int i = 0; i < limit && i < 128; i++) {
			char c1 = (char) (i + 1);
			for (int j = 0; j < Math.min(10, 128 - i); j++) {
				char c2 = (char) (j + 1);
				for (String lstrip : whitespaces) {
					for (String rstrip : whitespaces) {
						combinations.add(lstrip + c1 + c2 + rstrip);
						combinations.add(lstrip + c1 + rstrip + c2);
						combinations.add(String.valueOf(c1) + lstrip + c2 + rstrip);
					}
				}
			}
		}

		Collections.shuffle(combinations, random);
		return combinations.stream().limit(limit);
	}

	private Stream<String> generateUnicodeTests(int count) {
		List<String> tests = new ArrayList<>();
		Random random = new Random(42);

		// Add common Unicode characters
		String[] unicodeRanges = {
			"áéíóúàèìòùâêîôûäëïöü",  // Latin extended
			"αβγδεζηθικλμνξοπρστυφχψω", // Greek
			"абвгдеёжзийклмнопрстуфхцчшщъыьэюя", // Cyrillic
			"你好世界中文测试",              // Chinese
			"こんにちは世界テスト",           // Japanese
			"안녕하세요세계테스트",           // Korean
			"🌍🎉💻🚀❤️🔥⭐🎯"           // Emojis
		};

		for (String range : unicodeRanges) {
			for (int i = 0; i < Math.min(count / unicodeRanges.length, range.length()); i++) {
				char c = range.charAt(i);
				tests.add(String.valueOf(c));
				tests.add(" " + c);
				tests.add(c + " ");
				tests.add("a" + c + "b");
			}
		}

		Collections.shuffle(tests, random);
		return tests.stream().limit(count);
	}

	private Stream<String> generateRandomText(int count) {
		List<String> texts = new ArrayList<>();
		Random random = new Random(42);

		String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n\t.,!?";
		String[] words = {"hello", "world", "test", "tokenizer", "llama", "cpp", "java", "unicode"};

		for (int i = 0; i < count; i++) {
			StringBuilder text = new StringBuilder();
			int length = random.nextInt(50) + 1;

			for (int j = 0; j < length; j++) {
				if (random.nextBoolean()) {
					// Add random character
					text.append(chars.charAt(random.nextInt(chars.length())));
				} else {
					// Add random word
					text.append(words[random.nextInt(words.length)]);
				}
			}

			texts.add(text.toString());
		}

		return texts.stream();
	}

	public void runAllTests() {
		LOGGER.info("Starting comprehensive tokenizer tests...");
		stats.reset();

		testBasicTokenization();
		testEdgeCases();
		testAsciiCharacters();
		testUnicodeCharacters();
		testRandomText();

		LOGGER.info("All tests completed: " + stats);

		if (!stats.errorSamples.isEmpty()) {
			LOGGER.warning("Error samples:");
			stats.errorSamples.forEach(LOGGER::warning);
		}
	}

	public TokenizerStats getStats() {
		return stats;
	}

	public void close() {
		if (model != null) {
			model.close();
		}
	}

	public static void main(String[] args) {
		if (args.length != 1) {
			System.err.println("Usage: TokenizerTester <model_path>");
			System.exit(1);
		}

		TokenizerTester tester = new TokenizerTester(args[0]);
		try {
			tester.runAllTests();
		} finally {
			tester.close();
		}
	}
}