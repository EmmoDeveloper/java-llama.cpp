package de.kherud.llama.training;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Processes various dataset formats for LoRA training.
 * Supports JSON, JSONL, CSV, and plain text formats.
 */
public class DatasetProcessor {
	private static final Logger LOGGER = Logger.getLogger(DatasetProcessor.class.getName());
	private static final ObjectMapper objectMapper = new ObjectMapper();

	/**
	 * Load dataset from JSON file (Alpaca format)
	 * Expected format: [{"instruction": "...", "input": "...", "output": "..."}]
	 */
	public static List<TrainingApplication> loadAlpacaDataset(String filepath) throws IOException {
		LOGGER.info("Loading Alpaca dataset from: " + filepath);
		List<TrainingApplication> examples = new ArrayList<>();

		JsonNode root = objectMapper.readTree(new File(filepath));
		if (!root.isArray()) {
			throw new IllegalArgumentException("Alpaca dataset must be a JSON array");
		}

		for (JsonNode item : root) {
			String instruction = item.path("instruction").asText();
			String input = item.path("input").asText("");
			String output = item.path("output").asText();

			if (instruction.isEmpty() || output.isEmpty()) {
				LOGGER.warning("Skipping invalid example: missing instruction or output");
				continue;
			}

			examples.add(TrainingApplication.instructionFormat(instruction, input, output));
		}

		LOGGER.info(String.format("Loaded %d examples from Alpaca dataset", examples.size()));
		return examples;
	}

	/**
	 * Load dataset from JSONL file (one JSON object per line)
	 * Expected format: {"prompt": "...", "completion": "..."}
	 */
	public static List<TrainingApplication> loadJsonlDataset(String filepath) throws IOException {
		LOGGER.info("Loading JSONL dataset from: " + filepath);
		List<TrainingApplication> examples = new ArrayList<>();

		try (BufferedReader reader = Files.newBufferedReader(Paths.get(filepath))) {
			String line;
			int lineNum = 0;
			while ((line = reader.readLine()) != null) {
				lineNum++;
				line = line.trim();
				if (line.isEmpty()) continue;

				try {
					JsonNode json = objectMapper.readTree(line);
					String prompt = json.path("prompt").asText();
					String completion = json.path("completion").asText();

					if (prompt.isEmpty() || completion.isEmpty()) {
						LOGGER.warning(String.format("Line %d: missing prompt or completion", lineNum));
						continue;
					}

					examples.add(TrainingApplication.completionFormat(prompt, completion));

				} catch (Exception e) {
					LOGGER.warning(String.format("Line %d: failed to parse JSON: %s", lineNum, e.getMessage()));
				}
			}
		}

		LOGGER.info(String.format("Loaded %d examples from JSONL dataset", examples.size()));
		return examples;
	}

	/**
	 * Load dataset from CSV file
	 * Expected columns: prompt, completion
	 */
	public static List<TrainingApplication> loadCsvDataset(String filepath) throws IOException {
		LOGGER.info("Loading CSV dataset from: " + filepath);
		List<TrainingApplication> examples = new ArrayList<>();

		try (BufferedReader reader = Files.newBufferedReader(Paths.get(filepath))) {
			String headerLine = reader.readLine();
			if (headerLine == null) {
				throw new IllegalArgumentException("CSV file is empty");
			}

			String[] headers = headerLine.split(",");
			int promptCol = -1, completionCol = -1;

			// Find column indices
			for (int i = 0; i < headers.length; i++) {
				String header = headers[i].trim().toLowerCase();
				if (header.equals("prompt") || header.equals("input")) {
					promptCol = i;
				} else if (header.equals("completion") || header.equals("output") || header.equals("response")) {
					completionCol = i;
				}
			}

			if (promptCol == -1 || completionCol == -1) {
				throw new IllegalArgumentException("CSV must have 'prompt' and 'completion' columns");
			}

			String line;
			int lineNum = 1;
			while ((line = reader.readLine()) != null) {
				lineNum++;
				String[] fields = parseCsvLine(line);

				if (fields.length <= Math.max(promptCol, completionCol)) {
					LOGGER.warning(String.format("Line %d: insufficient columns", lineNum));
					continue;
				}

				String prompt = fields[promptCol].trim();
				String completion = fields[completionCol].trim();

				if (!prompt.isEmpty() && !completion.isEmpty()) {
					examples.add(TrainingApplication.completionFormat(prompt, completion));
				}
			}
		}

		LOGGER.info(String.format("Loaded %d examples from CSV dataset", examples.size()));
		return examples;
	}

	/**
	 * Load conversation dataset from JSON
	 * Expected format: [{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]
	 */
	public static List<TrainingApplication> loadConversationDataset(String filepath) throws IOException {
		LOGGER.info("Loading conversation dataset from: " + filepath);
		List<TrainingApplication> examples = new ArrayList<>();

		JsonNode root = objectMapper.readTree(new File(filepath));
		if (!root.isArray()) {
			throw new IllegalArgumentException("Conversation dataset must be a JSON array");
		}

		for (JsonNode item : root) {
			JsonNode conversations = item.path("conversations");
			if (!conversations.isArray() || conversations.size() < 2) {
				continue;
			}

			// Find human-gpt pairs
			for (int i = 0; i < conversations.size() - 1; i++) {
				JsonNode human = conversations.get(i);
				JsonNode gpt = conversations.get(i + 1);

				if ("human".equals(human.path("from").asText()) &&
				    "gpt".equals(gpt.path("from").asText())) {

					String userMessage = human.path("value").asText();
					String assistantResponse = gpt.path("value").asText();

					if (!userMessage.isEmpty() && !assistantResponse.isEmpty()) {
						examples.add(TrainingApplication.chatFormat(null, userMessage, assistantResponse));
					}
				}
			}
		}

		LOGGER.info(String.format("Loaded %d examples from conversation dataset", examples.size()));
		return examples;
	}

	/**
	 * Create a simple text completion dataset from a single text file
	 * Splits the text into chunks for training
	 */
	public static List<TrainingApplication> loadTextDataset(String filepath, int chunkSize, int overlap) throws IOException {
		LOGGER.info("Loading text dataset from: " + filepath);
		List<TrainingApplication> examples = new ArrayList<>();

		String content = Files.readString(Paths.get(filepath));
		String[] sentences = content.split("[.!?]+");

		StringBuilder currentChunk = new StringBuilder();
		for (String sentence : sentences) {
			sentence = sentence.trim();
			if (sentence.isEmpty()) continue;

			if (currentChunk.length() + sentence.length() > chunkSize) {
				if (currentChunk.length() > 0) {
					String text = currentChunk.toString().trim();
					// Create input-output pairs for next token prediction
					if (text.length() > 50) {
						int splitPoint = text.length() * 2 / 3;
						String input = text.substring(0, splitPoint);
						String target = text.substring(splitPoint);
						examples.add(TrainingApplication.completionFormat(input, target));
					}

					// Keep overlap for context
					if (overlap > 0 && currentChunk.length() > overlap) {
						String overlapText = text.substring(text.length() - overlap);
						currentChunk = new StringBuilder(overlapText);
					} else {
						currentChunk = new StringBuilder();
					}
				}
			}

			currentChunk.append(sentence).append(". ");
		}

		// Handle remaining text
		if (currentChunk.length() > 50) {
			String text = currentChunk.toString().trim();
			int splitPoint = text.length() * 2 / 3;
			String input = text.substring(0, splitPoint);
			String target = text.substring(splitPoint);
			examples.add(TrainingApplication.completionFormat(input, target));
		}

		LOGGER.info(String.format("Created %d examples from text dataset", examples.size()));
		return examples;
	}

	/**
	 * Filter examples by length to ensure they fit within model's context window
	 */
	public static List<TrainingApplication> filterByLength(List<TrainingApplication> examples, int maxTokens) {
		// This is a rough approximation (4 chars ≈ 1 token)
		int maxChars = maxTokens * 4;

		List<TrainingApplication> filtered = new ArrayList<>();
		for (TrainingApplication example : examples) {
			if (example.getFullText().length() <= maxChars) {
				filtered.add(example);
			}
		}

		LOGGER.info(String.format("Filtered dataset: %d/%d examples within %d token limit",
		                         filtered.size(), examples.size(), maxTokens));
		return filtered;
	}

	/**
	 * Split dataset into train/validation sets
	 */
	public static Map<String, List<TrainingApplication>> trainValidationSplit(
			List<TrainingApplication> examples, float validationRatio) {

		List<TrainingApplication> mutableExamples = new ArrayList<>(examples);
		Collections.shuffle(mutableExamples);
		int splitIndex = (int) (mutableExamples.size() * (1 - validationRatio));

		Map<String, List<TrainingApplication>> split = new HashMap<>();
		split.put("train", new ArrayList<>(mutableExamples.subList(0, splitIndex)));
		split.put("validation", new ArrayList<>(mutableExamples.subList(splitIndex, mutableExamples.size())));

		LOGGER.info(String.format("Dataset split: %d train, %d validation",
		                         split.get("train").size(), split.get("validation").size()));
		return split;
	}

	/**
	 * Simple CSV line parser that handles quoted fields
	 */
	private static String[] parseCsvLine(String line) {
		List<String> fields = new ArrayList<>();
		StringBuilder current = new StringBuilder();
		boolean inQuotes = false;

		for (char c : line.toCharArray()) {
			if (c == '"') {
				inQuotes = !inQuotes;
			} else if (c == ',' && !inQuotes) {
				fields.add(current.toString());
				current = new StringBuilder();
			} else {
				current.append(c);
			}
		}

		fields.add(current.toString());
		return fields.toArray(new String[0]);
	}

	/**
	 * Save dataset to JSONL format for later use
	 */
	public static void saveAsJsonl(List<TrainingApplication> examples, String filepath) throws IOException {
		try (PrintWriter writer = new PrintWriter(new FileWriter(filepath))) {
			for (TrainingApplication example : examples) {
				Map<String, String> json = new HashMap<>();
				json.put("prompt", example.input());
				json.put("completion", example.target());
				if (example.instruction() != null) {
					json.put("instruction", example.instruction());
				}
				writer.println(objectMapper.writeValueAsString(json));
			}
		}
		LOGGER.info(String.format("Saved %d examples to %s", examples.size(), filepath));
	}
}
