package de.kherud.llama;

import de.kherud.llama.args.LogFormat;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.regex.Pattern;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * To run this test download the model from here <a href="https://huggingface.co/mradermacher/jina-reranker-v1-tiny-en-GGUF/tree/main">...</a>
 * remove .enableEmbedding() from model setup and add .enableReRanking() and then enable the test.
 */
public class LlamaModelTest {
	private static final System.Logger logger = System.getLogger(LlamaModelTest.class.getName());
	private static final String prefix = "public static String removeNonAscii(String s) {\n    // ";
	private static final String suffix = "\n    return result;\n}";
	private static final int nPredict = 10;

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
//		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> logger.log(DEBUG, level + ": " + msg));
		// Use the new completion-optimized factory method
		model = LlamaModel.forCompletion(
			new ModelParameters()
				.setCtxSize(512)
				.setModel("models/codellama-7b.Q2_K.gguf")
				//.setModelUrl("https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43)
				.enableEmbedding().enableLogTimestamps().enableLogPrefix()
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testGenerateAnswer() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters(prefix)
			.setTemperature(0.95f)
			.setStopStrings("*/", "//")
			.setNPredict(nPredict)
			.setTokenIdBias(logitBias);

		int generated = 0;
		for (LlamaOutput ignored : model.generate(params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testGenerateInfill() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters("")
			.setInputPrefix(prefix)
			.setInputSuffix(suffix )
			.setTemperature(0.95f)
			.setStopStrings("*/", "//")
			.setNPredict(nPredict)
			.setTokenIdBias(logitBias)
			.setSeed(42);

		int generated = 0;
		for (LlamaOutput ignored : model.generate(params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testGenerateGrammar() {
		InferenceParameters params = new InferenceParameters("")
			.setGrammar("root ::= (\"a\" | \"b\")+")
			.setNPredict(nPredict);
		StringBuilder sb = new StringBuilder();
		for (LlamaOutput output : model.generate(params)) {
			sb.append(output);
		}
		String output = sb.toString();

		Assert.assertTrue(output.matches("[ab]+"));
		int generated = model.encode(output).length;
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testCompleteAnswer() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters(prefix)
			.setTemperature(0.95f)
			.setStopStrings("*/", "//")
			.setNPredict(nPredict)
			.setTokenIdBias(logitBias)
			.setSeed(42);

		String output = model.complete(params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteInfillCustom() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters("")
			.setInputPrefix(prefix)
			.setInputSuffix(suffix)
			.setTemperature(0.95f)
			.setStopStrings("*/", "//")
			.setNPredict(nPredict)
			.setTokenIdBias(logitBias)
			.setSeed(42);

		String output = model.complete(params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		InferenceParameters params = new InferenceParameters("")
			.setGrammar("root ::= (\"a\" | \"b\")+")
			.setNPredict(nPredict);
		String output = model.complete(params);
		Assert.assertTrue(output + " doesn't match [ab]+", output.matches("[ab]+"));
		int generated = model.encode(output).length;
		Assert.assertTrue("generated count is: " + generated,  generated > 0 && generated <= nPredict + 1);

	}

	@Test
	public void testCancelGenerating() {
		InferenceParameters params = new InferenceParameters(prefix).setNPredict(nPredict);

		int generated = 0;
		LlamaIterator iterator = model.generate(params).iterator();
		while (iterator.hasNext()) {
			iterator.next();
			generated++;
			if (generated == 5) {
				iterator.cancel();
			}
		}
		Assert.assertEquals(5, generated);
	}

	@Test
	public void testEmbedding() {
		float[] embedding = model.embed(prefix);
		Assert.assertEquals(4096, embedding.length);
	}

	@Ignore
	public void testReRanking() {
		String query = "Machine learning is";
		String [] TEST_DOCUMENTS = new String[] {
			"A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
			"Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
			"Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
			"Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine."
		};
		LlamaOutput llamaOutput = model.rerank(query, TEST_DOCUMENTS[0], TEST_DOCUMENTS[1], TEST_DOCUMENTS[2], TEST_DOCUMENTS[3] );

		logger.log(DEBUG, llamaOutput);
	}

	@Test
	public void testTokenization() {
		String prompt = "Hello, world!";
		int[] encoded = model.encode(prompt);
		String decoded = model.decode(encoded);
		// the llama tokenizer behavior - check that we can encode and decode
		Assert.assertTrue("Decoded text should contain original prompt", decoded.contains("Hello") && decoded.contains("world"));
	}

	@Ignore
	public void testLogText() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
			.setNPredict(nPredict)
			.setSeed(42);
		model.complete(params);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			Assert.assertFalse(jsonPattern.matcher(message.text).matches());
		}
	}

	@Ignore
	public void testLogJSON() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.JSON, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
			.setNPredict(nPredict)
			.setSeed(42);
		model.complete(params);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			Assert.assertTrue(jsonPattern.matcher(message.text).matches());
		}
	}

	@Ignore
	@Test
	public void testLogStdout() {
		// Unfortunately, `printf` can't be easily re-directed to Java. This test only works manually, thus.
		InferenceParameters params = new InferenceParameters(prefix)
			.setNPredict(nPredict)
			.setSeed(42);

		logger.log(DEBUG, "########## Log Text ##########");
		LlamaModel.setLogger(LogFormat.TEXT, null);
		model.complete(params);

		logger.log(DEBUG, "########## Log JSON ##########");
		LlamaModel.setLogger(LogFormat.JSON, null);
		model.complete(params);

		logger.log(DEBUG, "########## Log None ##########");
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> {});
		model.complete(params);

		logger.log(DEBUG, "##############################");
	}

	private String completeAndReadStdOut() {
		PrintStream stdOut = System.out;
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		@SuppressWarnings("ImplicitDefaultCharsetUsage") PrintStream printStream = new PrintStream(outputStream);

		try (printStream) {
			System.setOut(printStream);
			InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
			model.complete(params);
		} finally {
			System.out.flush();
			System.setOut(stdOut);
		}

		return outputStream.toString();
	}

	private List<String> splitLines(String text) {
		List<String> lines = new ArrayList<>();

		Scanner scanner = new Scanner(text);
		while (scanner.hasNextLine()) {
			String line = scanner.nextLine();
			lines.add(line);
		}
		scanner.close();

		return lines;
	}

	private record LogMessage(LogLevel level, String text) {
	}

	@Test
	public void testTemplate() {

		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "What is the best book?"));
		userMessages.add(new Pair<>("assistant", "It depends on your interests. Do you like fiction or non-fiction?"));

		InferenceParameters params = new InferenceParameters("A book recommendation system.")
			.setMessages("Book", userMessages)
			.setTemperature(0.95f)
			.setStopStrings("*/", "//")
			.setNPredict(nPredict)
			.setSeed(42);
		String result = model.applyTemplate(params);
		Assert.assertEquals("<|im_start|>system\nBook<|im_end|>\n<|im_start|>user\nWhat is the best book?<|im_end|>\n<|im_start|>assistant\nIt depends on your interests. Do you like fiction or non-fiction?<|im_end|>\n<|im_start|>assistant\n", result);
	}
}
