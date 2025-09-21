package de.kherud.llama;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Advanced function calling system for LLM tool use and structured output.
 * Enables models to call Java methods, validate parameters, and handle structured responses.
 */
public class FunctionCallSystem {
	private static final System.Logger logger = System.getLogger(FunctionCallSystem.class.getName());
	private static final ObjectMapper objectMapper = new ObjectMapper();
	private final Map<String, RegisteredFunction> functions = new ConcurrentHashMap<>();
	private final Map<String, Object> toolInstances = new ConcurrentHashMap<>();

	/**
	 * Represents a registered function that can be called by the LLM.
	 */
	public static class RegisteredFunction {
		private final String name;
		private final String description;
		private final Method method;
		private final Object instance;
		private final List<FunctionParameter> parameters;
		private final Class<?> returnType;

		public RegisteredFunction(String name, String description, Method method, Object instance) {
			this.name = name;
			this.description = description;
			this.method = method;
			this.instance = instance;
			this.returnType = method.getReturnType();
			this.parameters = extractParameters(method);
		}

		private List<FunctionParameter> extractParameters(Method method) {
			List<FunctionParameter> params = new ArrayList<>();
			Parameter[] methodParams = method.getParameters();

			for (Parameter param : methodParams) {
				FunctionParameter fp = new FunctionParameter(
					param.getName(),
					param.getType().getSimpleName(),
					param.getType(),
					"Parameter: " + param.getName()
				);
				params.add(fp);
			}

			return params;
		}

		public String getName() { return name; }
		public String getDescription() { return description; }
		public Method getMethod() { return method; }
		public Object getInstance() { return instance; }
		public List<FunctionParameter> getParameters() { return parameters; }
		public Class<?> getReturnType() { return returnType; }
	}

	/**
	 * Represents a function parameter with type and validation information.
	 */
	public static class FunctionParameter {
		private final String name;
		private final String type;
		private final Class<?> javaType;
		private final String description;

		public FunctionParameter(String name, String type, Class<?> javaType, String description) {
			this.name = name;
			this.type = type;
			this.javaType = javaType;
			this.description = description;
		}

		public String getName() { return name; }
		public String getType() { return type; }
		public Class<?> getJavaType() { return javaType; }
		public String getDescription() { return description; }
	}

	/**
	 * Result of a function call execution.
	 */
	public static class FunctionCallResult {
		private final String functionName;
		private final Object result;
		private final boolean success;
		private final String error;
		private final long executionTimeMs;

		public FunctionCallResult(String functionName, Object result, boolean success, String error, long executionTimeMs) {
			this.functionName = functionName;
			this.result = result;
			this.success = success;
			this.error = error;
			this.executionTimeMs = executionTimeMs;
		}

		public static FunctionCallResult success(String functionName, Object result, long executionTimeMs) {
			return new FunctionCallResult(functionName, result, true, null, executionTimeMs);
		}

		public static FunctionCallResult error(String functionName, String error, long executionTimeMs) {
			return new FunctionCallResult(functionName, null, false, error, executionTimeMs);
		}

		public String getFunctionName() { return functionName; }
		public Object getResult() { return result; }
		public boolean isSuccess() { return success; }
		public String getError() { return error; }
		public long getExecutionTimeMs() { return executionTimeMs; }

		@Override
		public String toString() {
			if (success) {
				return String.format("FunctionCallResult{name='%s', result=%s, time=%dms}",
					functionName, result, executionTimeMs);
			} else {
				return String.format("FunctionCallResult{name='%s', error='%s', time=%dms}",
					functionName, error, executionTimeMs);
			}
		}
	}

	/**
	 * Configuration for function calling behavior.
	 */
	public static class FunctionCallConfig {
		private final boolean validateParameters;
		private final boolean logCalls;
		private final long timeoutMs;
		private final boolean strictMode;

		public FunctionCallConfig() {
			this(true, true, 30000, false);
		}

		public FunctionCallConfig(boolean validateParameters, boolean logCalls, long timeoutMs, boolean strictMode) {
			this.validateParameters = validateParameters;
			this.logCalls = logCalls;
			this.timeoutMs = timeoutMs;
			this.strictMode = strictMode;
		}

		public boolean isValidateParameters() { return validateParameters; }
		public boolean isLogCalls() { return logCalls; }
		public long getTimeoutMs() { return timeoutMs; }
		public boolean isStrictMode() { return strictMode; }
	}

	/**
	 * Register a function that can be called by the LLM.
	 */
	public void registerFunction(String name, String description, Method method, Object instance) {
		if (name == null || method == null) {
			throw new IllegalArgumentException("Name and method cannot be null");
		}

		RegisteredFunction function = new RegisteredFunction(name, description, method, instance);
		functions.put(name, function);

		if (instance != null) {
			toolInstances.put(instance.getClass().getSimpleName(), instance);
		}
	}

	/**
	 * Register all public methods of a tool class as callable functions.
	 */
	public void registerTool(Object toolInstance, String prefix) {
		if (toolInstance == null) {
			throw new IllegalArgumentException("Tool instance cannot be null");
		}

		Class<?> toolClass = toolInstance.getClass();
		Method[] methods = toolClass.getMethods();

		for (Method method : methods) {
			// Skip Object methods and synthetic methods
			if (method.getDeclaringClass() == Object.class || method.isSynthetic()) {
				continue;
			}

			String functionName = (prefix != null ? prefix + "_" : "") + method.getName();
			String description = "Tool method: " + method.getName() + " from " + toolClass.getSimpleName();

			registerFunction(functionName, description, method, toolInstance);
		}
	}

	/**
	 * Execute a function call from LLM output.
	 */
	public FunctionCallResult executeFunction(String functionName, Map<String, Object> parameters, FunctionCallConfig config) {
		long startTime = System.currentTimeMillis();

		if (config.isLogCalls()) {
			logger.log(DEBUG, "🔧 Executing function: " + functionName + " with parameters: " + parameters);
		}

		RegisteredFunction function = functions.get(functionName);
		if (function == null) {
			String error = "Function not found: " + functionName;
			return FunctionCallResult.error(functionName, error, System.currentTimeMillis() - startTime);
		}

		try {
			// Validate and convert parameters
			Object[] args = prepareArguments(function, parameters, config);

			// Execute the function
			Object result = function.getMethod().invoke(function.getInstance(), args);

			long executionTime = System.currentTimeMillis() - startTime;

			if (config.isLogCalls()) {
				logger.log(DEBUG, "✅ Function executed successfully: " + functionName + " -> " + result);
			}

			return FunctionCallResult.success(functionName, result, executionTime);

		} catch (Exception e) {
			long executionTime = System.currentTimeMillis() - startTime;
			String error = "Execution error: " + e.getMessage();

			if (config.isLogCalls()) {
				logger.log(DEBUG, "❌ Function execution failed: " + functionName + " -> " + error);
			}

			return FunctionCallResult.error(functionName, error, executionTime);
		}
	}

	/**
	 * Parse function calls from LLM JSON output.
	 */
	public List<Map<String, Object>> parseFunctionCalls(String llmOutput) {
		List<Map<String, Object>> calls = new ArrayList<>();

		try {
			// Try to find JSON function calls in the output
			if (llmOutput.contains("[") && llmOutput.contains("]")) {
				// Look for array first
				int arrayStart = llmOutput.indexOf("[");
				int arrayEnd = llmOutput.lastIndexOf("]") + 1;

				if (arrayStart < arrayEnd) {
					String jsonPortion = llmOutput.substring(arrayStart, arrayEnd);

					try {
						JsonNode jsonNode = objectMapper.readTree(jsonPortion);
						if (jsonNode.isArray()) {
							// Handle array of function calls
							for (JsonNode callNode : jsonNode) {
								if (callNode.has("function") && callNode.has("parameters")) {
									Map<String, Object> call = new HashMap<>();
									call.put("function", callNode.get("function").asText());
									call.put("parameters", objectMapper.convertValue(callNode.get("parameters"), new TypeReference<Map<String, Object>>() {}));
									calls.add(call);
								}
							}
							return calls; // Return early if array parsing succeeded
						}
					} catch (Exception e) {
						// Array parsing failed, try single object
					}
				}
			}

			// Try single object parsing
			if (llmOutput.contains("{") && llmOutput.contains("}")) {
				// Extract JSON portions
				int start = llmOutput.indexOf("{");
				int end = llmOutput.lastIndexOf("}") + 1;

				if (start < end) {
					String jsonPortion = llmOutput.substring(start, end);

					// Try to parse as single JSON object
					try {
						JsonNode jsonNode = objectMapper.readTree(jsonPortion);

						if (jsonNode.has("function") && jsonNode.has("parameters")) {
							// Handle single function call
							Map<String, Object> call = new HashMap<>();
							call.put("function", jsonNode.get("function").asText());
							call.put("parameters", objectMapper.convertValue(jsonNode.get("parameters"), new TypeReference<Map<String, Object>>() {}));
							calls.add(call);
						}

					} catch (Exception e) {
						// Parsing failed, return empty list
						System.err.println("Failed to parse JSON from: " + jsonPortion);
					}
				}
			}

		} catch (Exception e) {
			System.err.println("Failed to parse function calls from LLM output: " + e.getMessage());
		}

		return calls;
	}

	/**
	 * Generate JSON schema for all registered functions (for LLM prompting).
	 */
	public String generateFunctionSchema() {
		Map<String, Object> schema = new HashMap<>();
		schema.put("functions", new ArrayList<>());

		for (RegisteredFunction function : functions.values()) {
			Map<String, Object> functionSchema = new HashMap<>();
			functionSchema.put("name", function.getName());
			functionSchema.put("description", function.getDescription());

			Map<String, Object> parameters = new HashMap<>();
			parameters.put("type", "object");
			parameters.put("properties", new HashMap<String, Object>());
			parameters.put("required", new ArrayList<String>());

			for (FunctionParameter param : function.getParameters()) {
				Map<String, Object> paramSchema = new HashMap<>();
				paramSchema.put("type", mapJavaTypeToJsonType(param.getJavaType()));
				paramSchema.put("description", param.getDescription());

				((Map<String, Object>) parameters.get("properties")).put(param.getName(), paramSchema);
				((List<String>) parameters.get("required")).add(param.getName());
			}

			functionSchema.put("parameters", parameters);
			((List<Object>) schema.get("functions")).add(functionSchema);
		}

		try {
			return objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(schema);
		} catch (Exception e) {
			return "{}";
		}
	}

	/**
	 * Get all registered function names.
	 */
	public Set<String> getRegisteredFunctionNames() {
		return new HashSet<>(functions.keySet());
	}

	/**
	 * Clear all registered functions.
	 */
	public void clearFunctions() {
		functions.clear();
		toolInstances.clear();
	}

	private Object[] prepareArguments(RegisteredFunction function, Map<String, Object> parameters, FunctionCallConfig config) throws Exception {
		List<FunctionParameter> paramDefs = function.getParameters();
		Object[] args = new Object[paramDefs.size()];

		for (int i = 0; i < paramDefs.size(); i++) {
			FunctionParameter paramDef = paramDefs.get(i);
			Object value = parameters.get(paramDef.getName());

			if (value == null && config.isStrictMode()) {
				throw new IllegalArgumentException("Missing required parameter: " + paramDef.getName());
			}

			// Convert value to correct type
			args[i] = convertParameter(value, paramDef.getJavaType());
		}

		return args;
	}

	private Object convertParameter(Object value, Class<?> targetType) {
		if (value == null) {
			return null;
		}

		if (targetType.isAssignableFrom(value.getClass())) {
			return value;
		}

		// String conversions
		if (targetType == String.class) {
			return value.toString();
		}

		// Numeric conversions
		if (value instanceof Number) {
			Number num = (Number) value;
			if (targetType == int.class || targetType == Integer.class) {
				return num.intValue();
			} else if (targetType == long.class || targetType == Long.class) {
				return num.longValue();
			} else if (targetType == double.class || targetType == Double.class) {
				return num.doubleValue();
			} else if (targetType == float.class || targetType == Float.class) {
				return num.floatValue();
			}
		}

		// String to numeric conversions
		if (value instanceof String) {
			String str = (String) value;
			try {
				if (targetType == int.class || targetType == Integer.class) {
					return Integer.parseInt(str);
				} else if (targetType == long.class || targetType == Long.class) {
					return Long.parseLong(str);
				} else if (targetType == double.class || targetType == Double.class) {
					return Double.parseDouble(str);
				} else if (targetType == boolean.class || targetType == Boolean.class) {
					return Boolean.parseBoolean(str);
				}
			} catch (NumberFormatException e) {
				throw new IllegalArgumentException("Cannot convert '" + str + "' to " + targetType.getSimpleName());
			}
		}

		return value;
	}

	private String mapJavaTypeToJsonType(Class<?> javaType) {
		if (javaType == String.class) {
			return "string";
		} else if (javaType == int.class || javaType == Integer.class ||
				   javaType == long.class || javaType == Long.class) {
			return "integer";
		} else if (javaType == double.class || javaType == Double.class ||
				   javaType == float.class || javaType == Float.class) {
			return "number";
		} else if (javaType == boolean.class || javaType == Boolean.class) {
			return "boolean";
		} else if (javaType.isArray() || List.class.isAssignableFrom(javaType)) {
			return "array";
		} else {
			return "object";
		}
	}
}
