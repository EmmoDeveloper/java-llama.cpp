package de.kherud.llama;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;

/**
 * Structured output generation system for LLMs.
 * Enforces output schemas, validates responses, and extracts structured data.
 */
public class StructuredOutputGenerator {
	private final ObjectMapper objectMapper = new ObjectMapper();
	private final Map<String, OutputSchema> schemas = new HashMap<>();

	/**
	 * Output schema definition for structured generation
	 */
	public static class OutputSchema {
		private final String name;
		private final Class<?> targetClass;
		private final Map<String, FieldSchema> fields;
		private final String description;
		private final boolean strict;

		public OutputSchema(String name, Class<?> targetClass, String description, boolean strict) {
			this.name = name;
			this.targetClass = targetClass;
			this.description = description;
			this.strict = strict;
			this.fields = extractFieldSchemas(targetClass);
		}

		private static Map<String, FieldSchema> extractFieldSchemas(Class<?> clazz) {
			Map<String, FieldSchema> fieldMap = new HashMap<>();

			for (Field field : clazz.getDeclaredFields()) {
				field.setAccessible(true);
				FieldSchema fieldSchema = new FieldSchema(
					field.getName(),
					field.getType(),
					field.getGenericType(),
					!isOptional(field),
					getFieldDescription(field)
				);
				fieldMap.put(field.getName(), fieldSchema);
			}

			return fieldMap;
		}

		private static boolean isOptional(Field field) {
			// Check for Optional annotation or nullable types
			return field.isAnnotationPresent(Optional.class) ||
				field.getType().getSimpleName().equals("Optional");
		}

		private static String getFieldDescription(Field field) {
			if (field.isAnnotationPresent(Description.class)) {
				return field.getAnnotation(Description.class).value();
			}
			return "Field: " + field.getName();
		}

		public String getName() {
			return name;
		}

		public Class<?> getTargetClass() {
			return targetClass;
		}

		public Map<String, FieldSchema> getFields() {
			return fields;
		}

		public String getDescription() {
			return description;
		}

		public boolean isStrict() {
			return strict;
		}
	}

	/**
	 * Field schema for validation and type checking
	 */
	public record FieldSchema(String name, Class<?> type, Type genericType, boolean required, String description) {
	}

	/**
	 * Annotations for schema definition
	 */
	@Retention(RetentionPolicy.RUNTIME)
	public @interface Description {
		String value();
	}

	@Retention(RetentionPolicy.RUNTIME)
	public @interface Optional {
	}

	public @interface Pattern {
		String value();
	}

	/**
	 * Result of structured output parsing
	 */
	public record ParseResult<T>(T value, boolean success, List<String> errors, Map<String, Object> metadata) {
		public ParseResult(T value, boolean success, List<String> errors, Map<String, Object> metadata) {
			this.value = value;
			this.success = success;
			this.errors = errors != null ? errors : new ArrayList<>();
			this.metadata = metadata != null ? metadata : new HashMap<>();
		}

		public static <T> ParseResult<T> success(T value) {
			return new ParseResult<>(value, true, null, null);
		}

		public static <T> ParseResult<T> success(T value, Map<String, Object> metadata) {
			return new ParseResult<>(value, true, null, metadata);
		}

		public static <T> ParseResult<T> failure(List<String> errors) {
			return new ParseResult<>(null, false, errors, null);
		}
	}

	/**
	 * Register an output schema
	 */
	public void registerSchema(String name, Class<?> targetClass, String description, boolean strict) {
		OutputSchema schema = new OutputSchema(name, targetClass, description, strict);
		schemas.put(name, schema);
	}

	/**
	 * Generate prompt instructions for structured output
	 */
	public String generatePromptInstructions(String schemaName) {
		OutputSchema schema = schemas.get(schemaName);
		if (schema == null) {
			return "";
		}

		StringBuilder prompt = new StringBuilder();
		prompt.append("Generate output in the following JSON structure:\n");
		prompt.append("```json\n");
		prompt.append("{\n");

		for (Map.Entry<String, FieldSchema> entry : schema.getFields().entrySet()) {
			FieldSchema field = entry.getValue();
			prompt.append("  \"").append(field.name()).append("\": ");
			prompt.append(getTypeExample(field.type()));
			prompt.append(",  // ").append(field.description());
			if (field.required()) {
				prompt.append(" (REQUIRED)");
			}
			prompt.append("\n");
		}

		// Remove last comma
		if (!prompt.isEmpty() && prompt.charAt(prompt.length() - 2) == ',') {
			prompt.deleteCharAt(prompt.length() - 2);
		}

		prompt.append("}\n");
		prompt.append("```\n");

		if (schema.isStrict()) {
			prompt.append("IMPORTANT: Output MUST be valid JSON matching this exact structure.\n");
		}

		return prompt.toString();
	}

	/**
	 * Parse LLM output into structured format
	 */
	public <T> ParseResult<T> parse(String llmOutput, String schemaName, Class<T> targetClass) {
		OutputSchema schema = schemas.get(schemaName);
		if (schema == null) {
			return ParseResult.failure(List.of("Schema not found: " + schemaName));
		}

		// Extract JSON from output
		String jsonContent = extractJson(llmOutput);
		if (jsonContent == null) {
			return ParseResult.failure(List.of("No valid JSON found in output"));
		}

		try {
			// Parse JSON
			JsonNode rootNode = objectMapper.readTree(jsonContent);

			// Validate against schema
			List<String> validationErrors = validateAgainstSchema(rootNode, schema);
			if (!validationErrors.isEmpty() && schema.isStrict()) {
				return ParseResult.failure(validationErrors);
			}

			// Convert to target class
			T result = objectMapper.treeToValue(rootNode, targetClass);

			Map<String, Object> metadata = new HashMap<>();
			metadata.put("validationWarnings", validationErrors);
			metadata.put("rawJson", jsonContent);

			return ParseResult.success(result, metadata);

		} catch (Exception e) {
			return ParseResult.failure(List.of("Parsing error: " + e.getMessage()));
		}
	}

	/**
	 * Extract JSON from mixed text output
	 */
	public String extractJson(String text) {
		if (text == null || text.isEmpty()) {
			return null;
		}

		// Try to find JSON block in markdown code blocks
		java.util.regex.Pattern codeBlockPattern = java.util.regex.Pattern.compile("```(?:json)?\\s*\\n?([\\s\\S]*?)\\n?```");
		Matcher matcher = codeBlockPattern.matcher(text);
		if (matcher.find()) {
			return matcher.group(1).trim();
		}

		// Try to find raw JSON object or array
		int startObj = text.indexOf("{");
		int endObj = text.lastIndexOf("}");
		int startArr = text.indexOf("[");
		int endArr = text.lastIndexOf("]");

		String candidate;

		if (startObj >= 0 && endObj > startObj) {
			candidate = text.substring(startObj, endObj + 1);
			if (isValidJson(candidate)) {
				return candidate;
			}
		}

		if (startArr >= 0 && endArr > startArr && (startArr < startObj || startObj < 0)) {
			candidate = text.substring(startArr, endArr + 1);
			if (isValidJson(candidate)) {
				return candidate;
			}
		}

		return null;
	}

	/**
	 * Validate JSON string
	 */
	private boolean isValidJson(String json) {
		try {
			objectMapper.readTree(json);
			return true;
		} catch (Exception e) {
			return false;
		}
	}

	/**
	 * Validate JSON against schema
	 */
	private static List<String> validateAgainstSchema(JsonNode node, OutputSchema schema) {
		List<String> errors = new ArrayList<>();

		// Check required fields
		for (Map.Entry<String, FieldSchema> entry : schema.getFields().entrySet()) {
			String fieldName = entry.getKey();
			FieldSchema fieldSchema = entry.getValue();

			if (fieldSchema.required() && !node.has(fieldName)) {
				errors.add("Missing required field: " + fieldName);
			}
		}

		// Check field types
		Iterator<String> fieldNames = node.fieldNames();
		while (fieldNames.hasNext()) {
			String fieldName = fieldNames.next();
			JsonNode fieldNode = node.get(fieldName);
			FieldSchema fieldSchema = schema.getFields().get(fieldName);

			if (fieldSchema != null) {
				String typeError = validateFieldType(fieldNode, fieldSchema);
				if (typeError != null) {
					errors.add(typeError);
				}
			}
		}

		return errors;
	}

	/**
	 * Validate field type
	 */
	private static String validateFieldType(JsonNode node, FieldSchema schema) {
		Class<?> expectedType = schema.type();

		if (expectedType == String.class && !node.isTextual()) {
			return "Field " + schema.name() + " should be a string";
		}
		if ((expectedType == Integer.class || expectedType == int.class) && !node.isInt()) {
			return "Field " + schema.name() + " should be an integer";
		}
		if ((expectedType == Double.class || expectedType == double.class) && !node.isNumber()) {
			return "Field " + schema.name() + " should be a number";
		}
		if ((expectedType == Boolean.class || expectedType == boolean.class) && !node.isBoolean()) {
			return "Field " + schema.name() + " should be a boolean";
		}
		if (List.class.isAssignableFrom(expectedType) && !node.isArray()) {
			return "Field " + schema.name() + " should be an array";
		}
		if (Map.class.isAssignableFrom(expectedType) && !node.isObject()) {
			return "Field " + schema.name() + " should be an object";
		}

		return null;
	}

	/**
	 * Generate example value for type
	 */
	private static String getTypeExample(Class<?> type) {
		if (type == String.class) {
			return "\"string\"";
		} else if (type == Integer.class || type == int.class) {
			return "0";
		} else if (type == Double.class || type == double.class) {
			return "0.0";
		} else if (type == Boolean.class || type == boolean.class) {
			return "false";
		} else if (List.class.isAssignableFrom(type)) {
			return "[]";
		} else if (Map.class.isAssignableFrom(type)) {
			return "{}";
		} else {
			return "{}";
		}
	}

	/**
	 * Create schema from class with builder pattern
	 */
	public static class SchemaBuilder {
		private String name;
		private Class<?> targetClass;
		private String description = "";
		private boolean strict = false;

		public SchemaBuilder name(String name) {
			this.name = name;
			return this;
		}

		public SchemaBuilder targetClass(Class<?> targetClass) {
			this.targetClass = targetClass;
			return this;
		}

		public SchemaBuilder description(String description) {
			this.description = description;
			return this;
		}

		public SchemaBuilder strict(boolean strict) {
			this.strict = strict;
			return this;
		}

		public OutputSchema build() {
			if (name == null || targetClass == null) {
				throw new IllegalStateException("Name and targetClass are required");
			}
			return new OutputSchema(name, targetClass, description, strict);
		}
	}

	/**
	 * Get all registered schema names
	 */
	public Set<String> getRegisteredSchemas() {
		return new HashSet<>(schemas.keySet());
	}

	/**
	 * Clear all schemas
	 */
	public void clearSchemas() {
		schemas.clear();
	}
}
