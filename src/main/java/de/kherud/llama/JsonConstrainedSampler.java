package de.kherud.llama;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.*;
import java.util.regex.Pattern;

/**
 * JSON-constrained sampler that ensures generated tokens follow JSON syntax rules.
 * Provides context-aware sampling for JSON generation with syntax validation
 * and schema enforcement.
 */
public class JsonConstrainedSampler implements AutoCloseable {

	/**
	 * JSON parsing states for context-aware sampling
	 */
	public enum JsonState {
		OBJECT_START,      // After {
		OBJECT_KEY,        // Expecting key name
		OBJECT_COLON,      // After key, expecting :
		OBJECT_VALUE,      // After :, expecting value
		OBJECT_COMMA,      // After value, expecting , or }
		ARRAY_START,       // After [
		ARRAY_VALUE,       // Expecting array element
		ARRAY_COMMA,       // After element, expecting , or ]
		STRING_VALUE,      // Inside string literal
		NUMBER_VALUE,      // Building number
		BOOLEAN_VALUE,     // Building boolean
		NULL_VALUE,        // Building null
		COMPLETE          // JSON is complete
	}

	/**
	 * Schema constraint types for guided generation
	 */
	public enum SchemaConstraint {
		STRICT,           // Must match schema exactly
		FLEXIBLE,         // Allow additional properties
		PARTIAL           // Allow incomplete objects
	}

	private final AdvancedSamplerManager.DynamicSampler dynamicSampler;
	private final ObjectMapper objectMapper;
	private final Stack<JsonState> stateStack;
	private final Stack<String> contextStack;
	private JsonState currentState;
	private SchemaConstraint constraintMode;
	private JsonNode schemaNode;
	private StringBuilder currentBuffer;

	// JSON token patterns
	private static final Pattern OBJECT_START = Pattern.compile("\\s*\\{\\s*");
	private static final Pattern ARRAY_START = Pattern.compile("\\s*\\[\\s*");
	private static final Pattern STRING_PATTERN = Pattern.compile("\"([^\"\\\\]|\\\\.)*\"");
	private static final Pattern NUMBER_PATTERN = Pattern.compile("-?\\d+(\\.\\d+)?([eE][+-]?\\d+)?");
	private static final Pattern BOOLEAN_PATTERN = Pattern.compile("\\b(true|false)\\b");
	private static final Pattern NULL_PATTERN = Pattern.compile("\\bnull\\b");

	public JsonConstrainedSampler() {
		this(SchemaConstraint.FLEXIBLE, null);
	}

	public JsonConstrainedSampler(SchemaConstraint constraintMode, String jsonSchema) {
		this.dynamicSampler = new AdvancedSamplerManager.DynamicSampler();
		this.objectMapper = new ObjectMapper();
		this.stateStack = new Stack<>();
		this.contextStack = new Stack<>();
		this.currentState = JsonState.OBJECT_START;
		this.constraintMode = constraintMode;
		this.currentBuffer = new StringBuilder();

		if (jsonSchema != null && !jsonSchema.isEmpty()) {
			try {
				this.schemaNode = objectMapper.readTree(jsonSchema);
			} catch (Exception e) {
				// Fallback to no schema validation
				this.schemaNode = null;
			}
		}

		setupJsonSamplers();
	}

	/**
	 * Process generated token and update JSON state
	 */
	public boolean processToken(String token) {
		currentBuffer.append(token);
		updateJsonState(currentBuffer.toString());
		switchSamplerForState();

		// Return true if JSON is still valid
		return isValidJsonSoFar();
	}

	/**
	 * Get valid next tokens for current JSON state
	 */
	public Set<String> getValidNextTokens() {
		Set<String> validTokens = new HashSet<>();

		switch (currentState) {
			case OBJECT_START:
				validTokens.add("\"");  // Start key
				if (constraintMode == SchemaConstraint.PARTIAL) {
					validTokens.add("}");  // Allow empty object
				}
				break;

			case OBJECT_KEY:
				validTokens.addAll(getPossibleKeys());
				break;

			case OBJECT_COLON:
				validTokens.add(":");
				break;

			case OBJECT_VALUE:
				validTokens.addAll(getPossibleValues());
				break;

			case OBJECT_COMMA:
				validTokens.add(",");
				validTokens.add("}");
				break;

			case ARRAY_START:
				validTokens.addAll(getPossibleValues());
				if (constraintMode == SchemaConstraint.PARTIAL) {
					validTokens.add("]");  // Allow empty array
				}
				break;

			case ARRAY_VALUE:
				validTokens.addAll(getPossibleValues());
				break;

			case ARRAY_COMMA:
				validTokens.add(",");
				validTokens.add("]");
				break;

			default:
				// Allow continuation of current value
				validTokens.add("*");  // Wildcard for any continuation
				break;
		}

		return validTokens;
	}

	/**
	 * Check if current JSON buffer is valid so far
	 */
	public boolean isValidJsonSoFar() {
		String jsonText = currentBuffer.toString().trim();

		if (jsonText.isEmpty()) {
			return true;
		}

		// Try to parse as JSON - if it fails, check if it's a valid prefix
		try {
			objectMapper.readTree(jsonText);
			return true; // Valid complete JSON
		} catch (Exception e) {
			// Check if it's a valid JSON prefix
			return isValidJsonPrefix(jsonText);
		}
	}

	public long getCurrentSampler() {
		return dynamicSampler.getCurrentSampler();
	}

	public JsonState getCurrentState() {
		return currentState;
	}

	public String getCurrentBuffer() {
		return currentBuffer.toString();
	}

	public void reset() {
		stateStack.clear();
		contextStack.clear();
		currentState = JsonState.OBJECT_START;
		currentBuffer.setLength(0);
		switchSamplerForState();
	}

	@Override
	public void close() {
		dynamicSampler.close();
	}

	private void setupJsonSamplers() {
		// High precision sampler for JSON structure tokens
		AdvancedSamplerManager.SamplerConfig structureSampler = new AdvancedSamplerManager.SamplerConfig("json_structure")
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.TOP_K)
				.param("k", 5))
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.TEMPERATURE)
				.param("temperature", 0.1f));

		// Medium precision for keys and values
		AdvancedSamplerManager.SamplerConfig contentSampler = new AdvancedSamplerManager.SamplerConfig("json_content")
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.TOP_P)
				.param("p", 0.8f))
			.addStep(new AdvancedSamplerManager.SamplerStep(AdvancedSamplerManager.SamplerType.TEMPERATURE)
				.param("temperature", 0.3f));

		// Register samplers with different contexts
		dynamicSampler.registerContext(AdvancedSamplerManager.SamplingContext.JSON_GENERATION, structureSampler);
		dynamicSampler.registerContext(AdvancedSamplerManager.SamplingContext.GENERAL, contentSampler);
	}

	private void updateJsonState(String jsonText) {
		String trimmed = jsonText.trim();

		// State machine for JSON parsing
		if (trimmed.isEmpty()) {
			currentState = JsonState.OBJECT_START;
		} else if (trimmed.equals("{")) {
			pushState(JsonState.OBJECT_KEY);
		} else if (trimmed.equals("[")) {
			pushState(JsonState.ARRAY_VALUE);
		} else if (isInState(JsonState.OBJECT_KEY) && STRING_PATTERN.matcher(getLastToken(trimmed)).matches()) {
			currentState = JsonState.OBJECT_COLON;
		} else if (isInState(JsonState.OBJECT_COLON) && trimmed.endsWith(":")) {
			currentState = JsonState.OBJECT_VALUE;
		} else if (isInState(JsonState.OBJECT_VALUE) && isValueComplete(trimmed)) {
			currentState = JsonState.OBJECT_COMMA;
		} else if ((isInState(JsonState.OBJECT_COMMA) || isInState(JsonState.ARRAY_COMMA)) && trimmed.endsWith(",")) {
			if (isInObjectContext()) {
				currentState = JsonState.OBJECT_KEY;
			} else {
				currentState = JsonState.ARRAY_VALUE;
			}
		}
		// Add more state transitions as needed
	}

	private void switchSamplerForState() {
		switch (currentState) {
			case OBJECT_START:
			case OBJECT_COLON:
			case OBJECT_COMMA:
			case ARRAY_START:
			case ARRAY_COMMA:
				dynamicSampler.switchContext(AdvancedSamplerManager.SamplingContext.JSON_GENERATION);
				break;

			case OBJECT_KEY:
			case OBJECT_VALUE:
			case ARRAY_VALUE:
			default:
				dynamicSampler.switchContext(AdvancedSamplerManager.SamplingContext.GENERAL);
				break;
		}
	}

	private Set<String> getPossibleKeys() {
		Set<String> keys = new HashSet<>();

		if (schemaNode != null && schemaNode.has("properties")) {
			JsonNode properties = schemaNode.get("properties");
			properties.fieldNames().forEachRemaining(keys::add);
		} else {
			// Common JSON keys if no schema
			keys.addAll(Arrays.asList("name", "id", "value", "type", "data", "status", "message"));
		}

		return keys;
	}

	private Set<String> getPossibleValues() {
		Set<String> values = new HashSet<>();

		// JSON structural elements
		values.add("\"");  // String start
		values.add("{");   // Object start
		values.add("[");   // Array start
		values.add("true");
		values.add("false");
		values.add("null");

		// Numbers (start with digits or negative sign)
		for (int i = 0; i <= 9; i++) {
			values.add(String.valueOf(i));
		}
		values.add("-");

		return values;
	}

	private boolean isValidJsonPrefix(String jsonText) {
		// Basic checks for valid JSON prefixes
		int openBraces = 0;
		int openBrackets = 0;
		boolean inString = false;
		boolean escaped = false;

		for (char c : jsonText.toCharArray()) {
			if (escaped) {
				escaped = false;
				continue;
			}

			if (c == '\\' && inString) {
				escaped = true;
				continue;
			}

			if (c == '"' && !escaped) {
				inString = !inString;
				continue;
			}

			if (!inString) {
				if (c == '{') openBraces++;
				else if (c == '}') openBraces--;
				else if (c == '[') openBrackets++;
				else if (c == ']') openBrackets--;

				if (openBraces < 0 || openBrackets < 0) {
					return false; // Too many closing brackets
				}
			}
		}

		return true; // Valid prefix
	}

	private boolean isValueComplete(String jsonText) {
		String lastToken = getLastToken(jsonText);
		return STRING_PATTERN.matcher(lastToken).matches() ||
			   NUMBER_PATTERN.matcher(lastToken).matches() ||
			   BOOLEAN_PATTERN.matcher(lastToken).matches() ||
			   NULL_PATTERN.matcher(lastToken).matches() ||
			   lastToken.equals("}") ||
			   lastToken.equals("]");
	}

	private String getLastToken(String text) {
		String[] tokens = text.split("\\s+");
		return tokens.length > 0 ? tokens[tokens.length - 1] : "";
	}

	private void pushState(JsonState newState) {
		stateStack.push(currentState);
		currentState = newState;
	}

	private boolean isInState(JsonState state) {
		return currentState == state;
	}

	private boolean isInObjectContext() {
		return contextStack.contains("object") ||
			   currentState == JsonState.OBJECT_KEY ||
			   currentState == JsonState.OBJECT_VALUE ||
			   currentState == JsonState.OBJECT_COMMA;
	}
}