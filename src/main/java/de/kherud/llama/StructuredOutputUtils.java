package de.kherud.llama;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Static utility methods for structured output processing
 */
public final class StructuredOutputUtils {

	private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
	private static final Pattern FUNCTION_CALL_PATTERN = Pattern.compile("FUNCTION_CALL:\\s*(.+)", Pattern.CASE_INSENSITIVE);

	private StructuredOutputUtils() {
		// Utility class
	}

	/**
	 * Format prompt for JSON generation
	 */
	public static String formatJsonPrompt(String prompt, JsonSchema schema) {
		return prompt +
			"\n\nRespond with valid JSON that matches this schema:\n" +
			schema.toJsonString() +
			"\n\nJSON Response:\n";
	}

	/**
	 * Format prompt for function calling
	 */
	public static String formatFunctionPrompt(String prompt, List<FunctionSpec> functions) {
		String functionsDesc = functions.stream()
			.map(func -> "Function: " + func.name + "\n" +
						 "Description: " + func.description + "\n" +
						 "Parameters: " + func.parameters.toJsonString() + "\n")
			.reduce("", (a, b) -> a + "\n" + b);

		return prompt +
			"\n\nYou have access to the following functions:\n\n" +
			functionsDesc +
			"\nCall a function by responding with:\n" +
			"FUNCTION_CALL: {\"name\": \"function_name\", \"arguments\": {args}}\n\n";
	}

	/**
	 * Parse JSON from response
	 */
	public static JsonResult parseJsonResponse(String response, JsonSchema schema) {
		JsonResult result = new JsonResult();
		result.rawResponse = response;

		// Extract JSON from response
		String jsonStr = extractJson(response);
		if (jsonStr == null) {
			result.success = false;
			result.error = "No valid JSON found in response";
			return result;
		}

		try {
			// Parse JSON
			result.json = OBJECT_MAPPER.readTree(jsonStr);

			// Validate against schema
			List<String> errors = schema.validate(result.json);
			if (!errors.isEmpty()) {
				result.success = false;
				result.error = "Schema validation failed: " + String.join(", ", errors);
				result.validationErrors = errors;
			} else {
				result.success = true;
			}
		} catch (JsonProcessingException e) {
			result.success = false;
			result.error = "JSON parsing failed: " + e.getMessage();
		}

		return result;
	}

	/**
	 * Parse function call from response
	 */
	public static FunctionCallResult parseFunctionCall(String response, List<FunctionSpec> functions) {
		FunctionCallResult result = new FunctionCallResult();
		result.rawResponse = response;

		// Look for function call pattern
		Matcher matcher = FUNCTION_CALL_PATTERN.matcher(response);

		if (!matcher.find()) {
			result.success = false;
			result.error = "No function call found in response";
			return result;
		}

		String jsonStr = matcher.group(1).trim();

		try {
			JsonNode callNode = OBJECT_MAPPER.readTree(jsonStr);
			String functionName = callNode.get("name").asText();
			JsonNode arguments = callNode.get("arguments");

			// Find matching function
			FunctionSpec matchedFunc = functions.stream()
				.filter(func -> func.name.equals(functionName))
				.findFirst()
				.orElse(null);

			if (matchedFunc == null) {
				result.success = false;
				result.error = "Unknown function: " + functionName;
				return result;
			}

			// Validate arguments
			List<String> errors = matchedFunc.parameters.validate(arguments);
			if (!errors.isEmpty()) {
				result.success = false;
				result.error = "Invalid arguments: " + String.join(", ", errors);
				result.validationErrors = errors;
				return result;
			}

			result.success = true;
			result.functionName = functionName;
			result.arguments = arguments;

		} catch (JsonProcessingException e) {
			result.success = false;
			result.error = "Failed to parse function call: " + e.getMessage();
		}

		return result;
	}

	/**
	 * Extract JSON from text
	 */
	public static String extractJson(String text) {
		// Try to find JSON block
		int start = -1;
		int end = -1;
		int braceCount = 0;
		int bracketCount = 0;

		for (int i = 0; i < text.length(); i++) {
			char c = text.charAt(i);
			if (c == '{' || c == '[') {
				if (start == -1) {
					start = i;
				}
				if (c == '{') braceCount++;
				else bracketCount++;
			} else if (c == '}' || c == ']') {
				if (c == '}') braceCount--;
				else bracketCount--;

				if (braceCount == 0 && bracketCount == 0 && start != -1) {
					end = i + 1;
					break;
				}
			}
		}

		if (start != -1 && end != -1) {
			return text.substring(start, end);
		}

		return null;
	}

	/**
	 * JSON Schema definition
	 */
	public static class JsonSchema {
		private final Map<String, Object> schema;

		public JsonSchema() {
			this.schema = new HashMap<>();
			this.schema.put("type", "object");
			this.schema.put("properties", new HashMap<String, Object>());
		}

		public JsonSchema addProperty(String name, String type, boolean required) {
			@SuppressWarnings("unchecked")
			Map<String, Object> properties = (Map<String, Object>) schema.get("properties");
			Map<String, Object> prop = new HashMap<>();
			prop.put("type", type);
			return getJsonSchema(name, required, properties, prop);
		}

		private StructuredOutputUtils.JsonSchema getJsonSchema(String name, boolean required, Map<String, Object> properties, Map<String, Object> prop) {
			properties.put(name, prop);

			if (required) {
				@SuppressWarnings("unchecked")
				List<String> requiredList = (List<String>) schema.computeIfAbsent("required", k -> new ArrayList<>());
				requiredList.add(name);
			}

			return this;
		}

		public JsonSchema addProperty(String name, String type, String description, boolean required) {
			@SuppressWarnings("unchecked")
			Map<String, Object> properties = (Map<String, Object>) schema.get("properties");
			Map<String, Object> prop = new HashMap<>();
			prop.put("type", type);
			prop.put("description", description);
			return getJsonSchema(name, required, properties, prop);
		}

		public String toJsonString() {
			try {
				return OBJECT_MAPPER.writeValueAsString(schema);
			} catch (JsonProcessingException e) {
				return "{}";
			}
		}

		public List<String> validate(JsonNode json) {
			List<String> errors = new ArrayList<>();

			// Check required fields
			@SuppressWarnings("unchecked")
			List<String> required = (List<String>) schema.get("required");
			if (required != null) {
				for (String field : required) {
					if (!json.has(field)) {
						errors.add("Missing required field: " + field);
					}
				}
			}

			// Check types
			@SuppressWarnings("unchecked")
			Map<String, Object> properties = (Map<String, Object>) schema.get("properties");
			json.fields().forEachRemaining(entry -> {
				String field = entry.getKey();
				JsonNode value = entry.getValue();

				if (properties.containsKey(field)) {
					@SuppressWarnings("unchecked")
					Map<String, Object> prop = (Map<String, Object>) properties.get(field);
					String expectedType = (String) prop.get("type");

					if (!matchesType(value, expectedType)) {
						errors.add("Field " + field + " has wrong type. Expected: " + expectedType);
					}
				}
			});

			return errors;
		}

		private static boolean matchesType(JsonNode node, String type) {
			return switch (type) {
				case "string" -> node.isTextual();
				case "number" -> node.isNumber();
				case "integer" -> node.isIntegralNumber();
				case "boolean" -> node.isBoolean();
				case "array" -> node.isArray();
				case "object" -> node.isObject();
				default -> true;
			};
		}
	}

	/**
	 * Function specification
	 */
		public record FunctionSpec(String name, String description, JsonSchema parameters) {
	}

	/**
	 * JSON generation result
	 */
	public static class JsonResult {
		public boolean success;
		public String rawResponse;
		public JsonNode json;
		public String error;
		public List<String> validationErrors;
	}

	/**
	 * Function call result
	 */
	public static class FunctionCallResult {
		public boolean success;
		public String rawResponse;
		public String functionName;
		public JsonNode arguments;
		public String error;
		public List<String> validationErrors;
	}
}
