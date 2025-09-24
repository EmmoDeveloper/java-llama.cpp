package de.kherud.llama;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Specialized sampling system for AI IDE code completion.
 * Provides context-aware, syntax-aware, and intelligent sampling strategies
 * optimized for different code completion scenarios.
 */
public class CodeCompletionSampler implements AutoCloseable {

	/**
	 * Code completion contexts with specialized sampling behavior
	 */
	public enum CodeContext {
		FUNCTION_SIGNATURE,     // function declarations, parameter lists
		VARIABLE_DECLARATION,   // variable names and types
		IMPORT_STATEMENT,       // import/include statements
		STRING_LITERAL,         // inside string literals
		COMMENT_BLOCK,         // documentation and comments
		EXPRESSION,            // arithmetic, logical expressions
		CONTROL_FLOW,          // if/for/while statements
		CLASS_MEMBER,          // class fields and methods
		TYPE_ANNOTATION,       // type hints and generics
		GENERIC                // fallback for unknown contexts
	}

	/**
	 * Programming language specific patterns and rules
	 */
	public enum Language {
		JAVA("java", Arrays.asList(".java"),
			 Arrays.asList("public", "private", "class", "interface", "import")),
		CPP("cpp", Arrays.asList(".cpp", ".h", ".hpp"),
			Arrays.asList("#include", "class", "namespace", "public", "private")),
		PYTHON("python", Arrays.asList(".py"),
			   Arrays.asList("def", "class", "import", "from")),
		JAVASCRIPT("javascript", Arrays.asList(".js", ".ts"),
				   Arrays.asList("function", "class", "import", "export")),
		UNKNOWN("unknown", Arrays.asList(), Arrays.asList());

		public final String name;
		public final List<String> extensions;
		public final List<String> keywords;

		Language(String name, List<String> extensions, List<String> keywords) {
			this.name = name;
			this.extensions = extensions;
			this.keywords = keywords;
		}
	}

	private final AdvancedSamplerManager.DynamicSampler dynamicSampler;
	private final Map<CodeContext, ContextAnalyzer> analyzers;
	private final Map<Language, LanguageProfile> languageProfiles;
	private Language currentLanguage = Language.UNKNOWN;
	private CodeContext currentContext = CodeContext.GENERIC;

	public CodeCompletionSampler() {
		this.dynamicSampler = new AdvancedSamplerManager.DynamicSampler();
		this.analyzers = createContextAnalyzers();
		this.languageProfiles = createLanguageProfiles();
		setupDefaultSamplers();
	}

	/**
	 * Analyze code context and switch to appropriate sampler
	 */
	public void analyzeContext(String codeText, String filename) {
		// Detect language from filename or content
		Language detectedLanguage = detectLanguage(filename, codeText);
		if (detectedLanguage != currentLanguage) {
			currentLanguage = detectedLanguage;
		}

		// Analyze current context
		CodeContext detectedContext = detectContext(codeText);
		if (detectedContext != currentContext) {
			currentContext = detectedContext;
			switchToContext(detectedContext);
		}
	}

	public long getCurrentSampler() {
		return dynamicSampler.getCurrentSampler();
	}

	public CodeContext getCurrentContext() {
		return currentContext;
	}

	public Language getCurrentLanguage() {
		return currentLanguage;
	}

	/**
	 * Get completion suggestions with context-aware filtering
	 */
	public List<String> filterCompletions(List<String> rawCompletions) {
		ContextAnalyzer analyzer = analyzers.get(currentContext);
		if (analyzer != null) {
			return analyzer.filterCompletions.apply(rawCompletions, currentLanguage);
		}
		return rawCompletions;
	}

	@Override
	public void close() {
		dynamicSampler.close();
	}

	private Language detectLanguage(String filename, String codeText) {
		// First try filename extension
		if (filename != null && !filename.isEmpty()) {
			String extension = getFileExtension(filename);
			for (Language lang : Language.values()) {
				if (lang.extensions.contains(extension)) {
					return lang;
				}
			}
		}

		// Fallback to content analysis
		return detectLanguageFromContent(codeText);
	}

	private Language detectLanguageFromContent(String codeText) {
		if (codeText == null || codeText.isEmpty()) {
			return Language.UNKNOWN;
		}

		Map<Language, Integer> scores = new HashMap<>();
		for (Language lang : Language.values()) {
			if (lang == Language.UNKNOWN) continue;

			int score = 0;
			for (String keyword : lang.keywords) {
				if (codeText.contains(keyword)) {
					score++;
				}
			}
			scores.put(lang, score);
		}

		return scores.entrySet().stream()
			.max(Map.Entry.comparingByValue())
			.map(Map.Entry::getKey)
			.orElse(Language.UNKNOWN);
	}

	private String getFileExtension(String filename) {
		int lastDot = filename.lastIndexOf('.');
		return lastDot >= 0 ? filename.substring(lastDot) : "";
	}

	private CodeContext detectContext(String codeText) {
		for (Map.Entry<CodeContext, ContextAnalyzer> entry : analyzers.entrySet()) {
			if (entry.getValue().matches.test(codeText)) {
				return entry.getKey();
			}
		}
		return CodeContext.GENERIC;
	}

	private void switchToContext(CodeContext context) {
		dynamicSampler.switchContext(mapToSamplingContext(context));
	}

	private AdvancedSamplerManager.SamplingContext mapToSamplingContext(CodeContext context) {
		switch (context) {
			case FUNCTION_SIGNATURE:
			case VARIABLE_DECLARATION:
			case CLASS_MEMBER:
				return AdvancedSamplerManager.SamplingContext.FUNCTION_NAME;
			case IMPORT_STATEMENT:
				return AdvancedSamplerManager.SamplingContext.CODE_COMPLETION;
			case COMMENT_BLOCK:
				return AdvancedSamplerManager.SamplingContext.DOCUMENTATION;
			case STRING_LITERAL:
				return AdvancedSamplerManager.SamplingContext.GENERAL;
			default:
				return AdvancedSamplerManager.SamplingContext.CODE_COMPLETION;
		}
	}

	private void setupDefaultSamplers() {
		// Register specialized samplers for each context
		dynamicSampler.registerContext(
			AdvancedSamplerManager.SamplingContext.CODE_COMPLETION,
			AdvancedSamplerManager.PresetConfigs.codeCompletion());

		dynamicSampler.registerContext(
			AdvancedSamplerManager.SamplingContext.FUNCTION_NAME,
			AdvancedSamplerManager.PresetConfigs.naming());

		dynamicSampler.registerContext(
			AdvancedSamplerManager.SamplingContext.DOCUMENTATION,
			AdvancedSamplerManager.PresetConfigs.documentation());

		dynamicSampler.registerContext(
			AdvancedSamplerManager.SamplingContext.DEBUGGING,
			AdvancedSamplerManager.PresetConfigs.debugging());
	}

	private Map<CodeContext, ContextAnalyzer> createContextAnalyzers() {
		Map<CodeContext, ContextAnalyzer> analyzers = new HashMap<>();

		analyzers.put(CodeContext.FUNCTION_SIGNATURE, new ContextAnalyzer(
			text -> Pattern.compile("\\b(function|def|public|private|static)\\s+\\w*$").matcher(text).find(),
			(completions, lang) -> filterFunctionNames(completions, lang)
		));

		analyzers.put(CodeContext.VARIABLE_DECLARATION, new ContextAnalyzer(
			text -> Pattern.compile("\\b(let|var|const|int|String|float|double)\\s+\\w*$").matcher(text).find(),
			(completions, lang) -> filterVariableNames(completions, lang)
		));

		analyzers.put(CodeContext.IMPORT_STATEMENT, new ContextAnalyzer(
			text -> text.trim().startsWith("import") || text.trim().startsWith("#include") || text.trim().startsWith("from"),
			(completions, lang) -> filterImportStatements(completions, lang)
		));

		analyzers.put(CodeContext.COMMENT_BLOCK, new ContextAnalyzer(
			text -> text.contains("/**") || text.contains("//") || text.contains("#"),
			(completions, lang) -> filterDocumentation(completions, lang)
		));

		analyzers.put(CodeContext.STRING_LITERAL, new ContextAnalyzer(
			text -> isInsideStringLiteral(text),
			(completions, lang) -> completions // Allow any text inside strings
		));

		return analyzers;
	}

	private Map<Language, LanguageProfile> createLanguageProfiles() {
		Map<Language, LanguageProfile> profiles = new HashMap<>();

		profiles.put(Language.JAVA, new LanguageProfile(
			Arrays.asList("public", "private", "protected", "static", "final"),
			Arrays.asList("String", "int", "boolean", "List", "Map"),
			Arrays.asList("get", "set", "is", "has", "create", "build")
		));

		profiles.put(Language.CPP, new LanguageProfile(
			Arrays.asList("public", "private", "protected", "static", "const"),
			Arrays.asList("int", "string", "vector", "map", "bool"),
			Arrays.asList("get", "set", "is", "has", "make", "create")
		));

		profiles.put(Language.PYTHON, new LanguageProfile(
			Arrays.asList("def", "class", "self", "__init__", "return"),
			Arrays.asList("str", "int", "bool", "list", "dict"),
			Arrays.asList("get", "set", "is", "has", "make", "create")
		));

		return profiles;
	}

	private boolean isInsideStringLiteral(String text) {
		int doubleQuotes = text.length() - text.replace("\"", "").length();
		int singleQuotes = text.length() - text.replace("'", "").length();
		return (doubleQuotes % 2 == 1) || (singleQuotes % 2 == 1);
	}

	private List<String> filterFunctionNames(List<String> completions, Language lang) {
		LanguageProfile profile = languageProfiles.get(lang);
		if (profile == null) return completions;

		return completions.stream()
			.filter(completion -> isValidFunctionName(completion, profile))
			.sorted((a, b) -> scoreFunctionName(b, profile) - scoreFunctionName(a, profile))
			.collect(ArrayList::new, (list, item) -> list.add(item), ArrayList::addAll);
	}

	private List<String> filterVariableNames(List<String> completions, Language lang) {
		return completions.stream()
			.filter(completion -> isValidVariableName(completion, lang))
			.collect(ArrayList::new, (list, item) -> list.add(item), ArrayList::addAll);
	}

	private List<String> filterImportStatements(List<String> completions, Language lang) {
		return completions.stream()
			.filter(completion -> isValidImport(completion, lang))
			.collect(ArrayList::new, (list, item) -> list.add(item), ArrayList::addAll);
	}

	private List<String> filterDocumentation(List<String> completions, Language lang) {
		return completions; // Documentation can be more creative
	}

	private boolean isValidFunctionName(String name, LanguageProfile profile) {
		// Check if name follows naming conventions
		return name.matches("^[a-zA-Z][a-zA-Z0-9_]*$") &&
			   !name.matches("^\\d") &&
			   name.length() > 1;
	}

	private int scoreFunctionName(String name, LanguageProfile profile) {
		int score = 0;
		for (String commonPrefix : profile.commonFunctionPrefixes) {
			if (name.toLowerCase().startsWith(commonPrefix.toLowerCase())) {
				score += 10;
				break;
			}
		}
		return score;
	}

	private boolean isValidVariableName(String name, Language lang) {
		switch (lang) {
			case JAVA:
				return name.matches("^[a-z][a-zA-Z0-9]*$");
			case CPP:
				return name.matches("^[a-zA-Z_][a-zA-Z0-9_]*$");
			case PYTHON:
				return name.matches("^[a-z_][a-z0-9_]*$");
			default:
				return name.matches("^[a-zA-Z_][a-zA-Z0-9_]*$");
		}
	}

	private boolean isValidImport(String imp, Language lang) {
		// Basic validation for import statements
		return !imp.trim().isEmpty() &&
			   !imp.contains(" ") ||
			   imp.contains(".") ||
			   imp.contains("/");
	}

	/**
	 * Context analyzer for detecting and filtering completions
	 */
	private static class ContextAnalyzer {
		final java.util.function.Predicate<String> matches;
		final java.util.function.BiFunction<List<String>, Language, List<String>> filterCompletions;

		ContextAnalyzer(java.util.function.Predicate<String> matches,
						java.util.function.BiFunction<List<String>, Language, List<String>> filterCompletions) {
			this.matches = matches;
			this.filterCompletions = filterCompletions;
		}
	}

	/**
	 * Language-specific naming conventions and preferences
	 */
	private static class LanguageProfile {
		final List<String> keywords;
		final List<String> commonTypes;
		final List<String> commonFunctionPrefixes;

		LanguageProfile(List<String> keywords, List<String> commonTypes, List<String> commonFunctionPrefixes) {
			this.keywords = keywords;
			this.commonTypes = commonTypes;
			this.commonFunctionPrefixes = commonFunctionPrefixes;
		}
	}
}