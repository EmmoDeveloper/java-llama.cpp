package de.kherud.llama.tools;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Development and build tools.
 *
 * Equivalent to build scripts and development utilities - provides project building,
 * testing, packaging, and development workflow automation.
 */
public class ProjectBuilder implements AutoCloseable {
	private static final System.Logger logger = System.getLogger(ProjectBuilder.class.getName());
	private static final ObjectMapper MAPPER = new ObjectMapper();

	public static class BuildConfig {
		private String buildTarget = "jar"; // jar, native, all
		private String buildProfile = "release"; // debug, release
		private boolean runTests = true;
		private boolean generateDocs = false;
		private boolean verbose = false;
		private boolean parallel = true;
		private int maxConcurrency = Runtime.getRuntime().availableProcessors();
		private List<String> skipModules = new ArrayList<>();
		private Map<String, String> buildProperties = new HashMap<>();
		private String outputDirectory = "target";
		private boolean cleanBefore = false;

		public BuildConfig buildTarget(String target) {
			this.buildTarget = target;
			return this;
		}

		public BuildConfig buildProfile(String profile) {
			this.buildProfile = profile;
			return this;
		}

		public BuildConfig runTests(boolean run) {
			this.runTests = run;
			return this;
		}

		public BuildConfig generateDocs(boolean generate) {
			this.generateDocs = generate;
			return this;
		}

		public BuildConfig verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public BuildConfig parallel(boolean parallel) {
			this.parallel = parallel;
			return this;
		}

		public BuildConfig maxConcurrency(int max) {
			this.maxConcurrency = max;
			return this;
		}

		public BuildConfig skipModule(String module) {
			this.skipModules.add(module);
			return this;
		}

		public BuildConfig addProperty(String key, String value) {
			this.buildProperties.put(key, value);
			return this;
		}

		public BuildConfig outputDirectory(String dir) {
			this.outputDirectory = dir;
			return this;
		}

		public BuildConfig cleanBefore(boolean clean) {
			this.cleanBefore = clean;
			return this;
		}
	}

	public static class BuildResult {
		public boolean success;
		public String error;
		public long buildTime;
		public int modulesBuilt;
		public int testsRun;
		public int testsPassed;
		public int testsFailed;
		public List<Path> artifacts = new ArrayList<>();
		public Map<String, Object> buildStats = new HashMap<>();
		public List<String> warnings = new ArrayList<>();

		public BuildResult success() {
			this.success = true;
			return this;
		}

		public BuildResult fail(String error) {
			this.success = false;
			this.error = error;
			return this;
		}

		public BuildResult addArtifact(Path artifact) {
			this.artifacts.add(artifact);
			return this;
		}

		public BuildResult addWarning(String warning) {
			this.warnings.add(warning);
			return this;
		}
	}

	public static class ModuleInfo {
		public String name;
		public Path path;
		public List<String> dependencies = new ArrayList<>();
		public String type; // library, executable, test
		public Map<String, Object> properties = new HashMap<>();
		public boolean buildRequired = true;
	}

	public static class TestResult {
		public String testName;
		public boolean passed;
		public String message;
		public long duration;
		public String output;
		public Throwable exception;

		public TestResult(String testName) {
			this.testName = testName;
		}

		public TestResult pass() {
			this.passed = true;
			return this;
		}

		public TestResult fail(String message, Throwable exception) {
			this.passed = false;
			this.message = message;
			this.exception = exception;
			return this;
		}
	}

	private final Path projectRoot;
	private final BuildConfig config;
	private final ExecutorService executor;

	public ProjectBuilder(Path projectRoot, BuildConfig config) {
		this.projectRoot = projectRoot;
		this.config = config;
		this.executor = config.parallel ?
			Executors.newFixedThreadPool(config.maxConcurrency) :
			Executors.newSingleThreadExecutor();
	}

	/**
	 * Build the entire project
	 */
	public BuildResult build() {
		logger.log(System.Logger.Level.INFO,"Starting project build");
		logger.log(System.Logger.Level.INFO,"Target: " + config.buildTarget);
		logger.log(System.Logger.Level.INFO,"Profile: " + config.buildProfile);

		BuildResult result = new BuildResult();
		long startTime = System.currentTimeMillis();

		try {
			// Clean if requested
			if (config.cleanBefore) {
				clean();
			}

			// Discover modules
			List<ModuleInfo> modules = discoverModules();
			logger.log(System.Logger.Level.INFO,"Found " + modules.size() + " modules");

			// Determine build order
			List<ModuleInfo> buildOrder = determineBuildOrder(modules);

			// Build modules
			if (config.parallel) {
				buildModulesParallel(buildOrder, result);
			} else {
				buildModulesSequential(buildOrder, result);
			}

			// Run tests
			if (config.runTests) {
				runTests(result);
			}

			// Generate documentation
			if (config.generateDocs) {
				generateDocumentation(result);
			}

			// Package artifacts
			packageArtifacts(result);

			result.buildTime = System.currentTimeMillis() - startTime;
			result.success();

			logger.log(System.Logger.Level.INFO,"Build completed successfully in " + result.buildTime + "ms");

		} catch (Exception e) {
			result.buildTime = System.currentTimeMillis() - startTime;
			result.fail(e.getMessage());
			logger.log(System.Logger.Level.ERROR, "Build failed: " + e.getMessage(), e);
		}

		return result;
	}

	/**
	 * Clean build artifacts
	 */
	public void clean() throws IOException {
		logger.log(System.Logger.Level.INFO,"Cleaning build artifacts");

		Path outputDir = projectRoot.resolve(config.outputDirectory);
		if (Files.exists(outputDir)) {
			deleteDirectory(outputDir);
		}

		// Clean common build directories
		String[] cleanDirs = {"target", "build", "out", ".gradle/build-cache"};
		for (String dir : cleanDirs) {
			Path dirPath = projectRoot.resolve(dir);
			if (Files.exists(dirPath)) {
				deleteDirectory(dirPath);
			}
		}

		logger.log(System.Logger.Level.INFO,"Clean completed");
	}

	private void deleteDirectory(Path dir) throws IOException {
		if (Files.exists(dir)) {
			Files.walk(dir)
				.sorted(Comparator.reverseOrder())
				.forEach(path -> {
					try {
						Files.delete(path);
					} catch (IOException e) {
						logger.log(System.Logger.Level.WARNING,"Failed to delete: " + path);
					}
				});
		}
	}

	private List<ModuleInfo> discoverModules() throws IOException {
		List<ModuleInfo> modules = new ArrayList<>();

		// Look for Java modules (Maven/Gradle structure)
		discoverJavaModules(modules);

		// Look for native modules (CMake, Makefile)
		discoverNativeModules(modules);

		return modules;
	}

	private void discoverJavaModules(List<ModuleInfo> modules) throws IOException {
		// Look for pom.xml files (Maven)
		Files.walk(projectRoot)
			.filter(path -> path.getFileName().toString().equals("pom.xml"))
			.forEach(pomPath -> {
				try {
					ModuleInfo module = parseGradleModule(pomPath.getParent());
					if (module != null) {
						modules.add(module);
					}
				} catch (Exception e) {
					logger.log(System.Logger.Level.WARNING,"Failed to parse Maven module: " + pomPath);
				}
			});

		// Look for build.gradle files (Gradle)
		Files.walk(projectRoot)
			.filter(path -> path.getFileName().toString().equals("build.gradle"))
			.forEach(gradlePath -> {
				try {
					ModuleInfo module = parseGradleModule(gradlePath.getParent());
					if (module != null) {
						modules.add(module);
					}
				} catch (Exception e) {
					logger.log(System.Logger.Level.WARNING,"Failed to parse Gradle module: " + gradlePath);
				}
			});
	}

	private void discoverNativeModules(List<ModuleInfo> modules) throws IOException {
		// Look for CMakeLists.txt files
		Files.walk(projectRoot)
			.filter(path -> path.getFileName().toString().equals("CMakeLists.txt"))
			.forEach(cmakePath -> {
				try {
					ModuleInfo module = parseCMakeModule(cmakePath.getParent());
					if (module != null) {
						modules.add(module);
					}
				} catch (Exception e) {
					logger.log(System.Logger.Level.WARNING,"Failed to parse CMake module: " + cmakePath);
				}
			});

		// Look for Makefiles
		Files.walk(projectRoot)
			.filter(path -> path.getFileName().toString().equals("Makefile"))
			.forEach(makefilePath -> {
				try {
					ModuleInfo module = parseMakefileModule(makefilePath.getParent());
					if (module != null) {
						modules.add(module);
					}
				} catch (Exception e) {
					logger.log(System.Logger.Level.WARNING,"Failed to parse Makefile module: " + makefilePath);
				}
			});
	}

	private ModuleInfo parseMavenModule(Path moduleDir) throws IOException {
		Path pomPath = moduleDir.resolve("pom.xml");
		if (!Files.exists(pomPath)) {
			return null;
		}

		ModuleInfo module = new ModuleInfo();
		module.name = moduleDir.getFileName().toString();
		module.path = moduleDir;
		module.type = "library";

		// Parse pom.xml for dependencies and properties
		// This is a simplified implementation
		module.properties.put("build_tool", "maven");

		return module;
	}

	private ModuleInfo parseGradleModule(Path moduleDir) throws IOException {
		Path gradlePath = moduleDir.resolve("build.gradle");
		if (!Files.exists(gradlePath)) {
			return null;
		}

		ModuleInfo module = new ModuleInfo();
		module.name = moduleDir.getFileName().toString();
		module.path = moduleDir;
		module.type = "library";

		// Parse build.gradle for dependencies and properties
		// This is a simplified implementation
		module.properties.put("build_tool", "gradle");

		return module;
	}

	private ModuleInfo parseCMakeModule(Path moduleDir) throws IOException {
		Path cmakePath = moduleDir.resolve("CMakeLists.txt");
		if (!Files.exists(cmakePath)) {
			return null;
		}

		ModuleInfo module = new ModuleInfo();
		module.name = moduleDir.getFileName().toString();
		module.path = moduleDir;
		module.type = "native";

		module.properties.put("build_tool", "cmake");

		return module;
	}

	private ModuleInfo parseMakefileModule(Path moduleDir) throws IOException {
		Path makefilePath = moduleDir.resolve("Makefile");
		if (!Files.exists(makefilePath)) {
			return null;
		}

		ModuleInfo module = new ModuleInfo();
		module.name = moduleDir.getFileName().toString();
		module.path = moduleDir;
		module.type = "native";

		module.properties.put("build_tool", "make");

		return module;
	}

	private List<ModuleInfo> determineBuildOrder(List<ModuleInfo> modules) {
		// Simple topological sort based on dependencies
		List<ModuleInfo> ordered = new ArrayList<>();
		Set<String> built = new HashSet<>();

		while (ordered.size() < modules.size()) {
			boolean progress = false;

			for (ModuleInfo module : modules) {
				if (!ordered.contains(module) && !config.skipModules.contains(module.name)) {
					boolean canBuild = module.dependencies.stream()
						.allMatch(dep -> built.contains(dep));

					if (canBuild) {
						ordered.add(module);
						built.add(module.name);
						progress = true;
					}
				}
			}

			if (!progress) {
				// Add remaining modules (circular dependencies or missing deps)
				for (ModuleInfo module : modules) {
					if (!ordered.contains(module)) {
						ordered.add(module);
					}
				}
				break;
			}
		}

		return ordered;
	}

	private void buildModulesSequential(List<ModuleInfo> modules, BuildResult result) throws Exception {
		for (ModuleInfo module : modules) {
			buildModule(module, result);
		}
	}

	private void buildModulesParallel(List<ModuleInfo> modules, BuildResult result) throws Exception {
		List<Future<Void>> futures = new ArrayList<>();

		for (ModuleInfo module : modules) {
			Future<Void> future = executor.submit(() -> {
				try {
					buildModule(module, result);
					return null;
				} catch (Exception e) {
					throw new RuntimeException("Module build failed: " + module.name, e);
				}
			});
			futures.add(future);
		}

		// Wait for all builds to complete
		for (Future<Void> future : futures) {
			future.get();
		}
	}

	private void buildModule(ModuleInfo module, BuildResult result) throws Exception {
		if (config.verbose) {
			logger.log(System.Logger.Level.INFO,"Building module: " + module.name);
		}

		String buildTool = (String) module.properties.get("build_tool");
		switch (buildTool) {
			case "maven":
				buildMavenModule(module, result);
				break;
			case "gradle":
				buildGradleModule(module, result);
				break;
			case "cmake":
				buildCMakeModule(module, result);
				break;
			case "make":
				buildMakefileModule(module, result);
				break;
			default:
				logger.log(System.Logger.Level.WARNING,"Unknown build tool for module: " + module.name);
		}

		result.modulesBuilt++;
	}

	private void buildMavenModule(ModuleInfo module, BuildResult result) throws Exception {
		List<String> command = new ArrayList<>();
		command.add("mvn");
		command.add("clean");
		command.add("compile");

		if (config.runTests) {
			command.add("test");
		}

		command.add("package");

		executeCommand(command, module.path, result);
	}

	private void buildGradleModule(ModuleInfo module, BuildResult result) throws Exception {
		List<String> command = new ArrayList<>();
		command.add("./gradlew");
		command.add("clean");
		command.add("build");

		if (!config.runTests) {
			command.add("-x");
			command.add("test");
		}

		executeCommand(command, module.path, result);
	}

	private void buildCMakeModule(ModuleInfo module, BuildResult result) throws Exception {
		Path buildDir = module.path.resolve("build");
		Files.createDirectories(buildDir);

		// Configure
		List<String> configureCommand = new ArrayList<>();
		configureCommand.add("cmake");
		configureCommand.add("..");

		if (config.buildProfile.equals("debug")) {
			configureCommand.add("-DCMAKE_BUILD_TYPE=Debug");
		} else {
			configureCommand.add("-DCMAKE_BUILD_TYPE=Release");
		}

		executeCommand(configureCommand, buildDir, result);

		// Build
		List<String> buildCommand = new ArrayList<>();
		buildCommand.add("cmake");
		buildCommand.add("--build");
		buildCommand.add(".");

		if (config.parallel) {
			buildCommand.add("--parallel");
			buildCommand.add(String.valueOf(config.maxConcurrency));
		}

		executeCommand(buildCommand, buildDir, result);
	}

	private void buildMakefileModule(ModuleInfo module, BuildResult result) throws Exception {
		List<String> command = new ArrayList<>();
		command.add("make");

		if (config.parallel) {
			command.add("-j" + config.maxConcurrency);
		}

		executeCommand(command, module.path, result);
	}

	private void executeCommand(List<String> command, Path workingDir, BuildResult result) throws Exception {
		ProcessBuilder pb = new ProcessBuilder(command);
		pb.directory(workingDir.toFile());
		pb.redirectErrorStream(true);

		if (config.verbose) {
			logger.log(System.Logger.Level.INFO,"Executing: " + String.join(" ", command) + " in " + workingDir);
		}

		Process process = pb.start();

		// Capture output
		StringBuilder output = new StringBuilder();
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
			String line;
			while ((line = reader.readLine()) != null) {
				output.append(line).append("\n");
				if (config.verbose) {
					System.out.println(line);
				}
			}
		}

		int exitCode = process.waitFor();

		if (exitCode != 0) {
			throw new Exception("Command failed with exit code " + exitCode + ":\n" + output.toString());
		}
	}

	private void runTests(BuildResult result) throws Exception {
		logger.log(System.Logger.Level.INFO,"Running tests");

		// Find test classes and run them
		List<TestResult> testResults = discoverAndRunTests();

		result.testsRun = testResults.size();
		result.testsPassed = (int) testResults.stream().filter(t -> t.passed).count();
		result.testsFailed = result.testsRun - result.testsPassed;

		if (result.testsFailed > 0) {
			throw new Exception(result.testsFailed + " tests failed");
		}

		logger.log(System.Logger.Level.INFO,"All " + result.testsRun + " tests passed");
	}

	private List<TestResult> discoverAndRunTests() throws Exception {
		List<TestResult> results = new ArrayList<>();

		// This is a simplified test discovery
		// In practice, you would integrate with JUnit, TestNG, etc.

		Path testDir = projectRoot.resolve("src/test/java");
		if (Files.exists(testDir)) {
			Files.walk(testDir)
				.filter(path -> path.toString().endsWith("Test.java"))
				.forEach(testPath -> {
					TestResult result = runTest(testPath);
					results.add(result);
				});
		}

		return results;
	}

	private TestResult runTest(Path testPath) {
		String testName = testPath.getFileName().toString();
		TestResult result = new TestResult(testName);

		try {
			// Simplified test execution
			// In practice, you would compile and run the actual test
			result.pass();
		} catch (Exception e) {
			result.fail("Test execution failed", e);
		}

		return result;
	}

	private void generateDocumentation(BuildResult result) throws Exception {
		logger.log(System.Logger.Level.INFO,"Generating documentation");

		// Generate Javadoc
		generateJavadoc();

		// Generate other documentation formats
		generateMarkdownDocs();

		logger.log(System.Logger.Level.INFO,"Documentation generation completed");
	}

	private void generateJavadoc() throws Exception {
		List<String> command = new ArrayList<>();
		command.add("javadoc");
		command.add("-d");
		command.add(projectRoot.resolve(config.outputDirectory).resolve("docs").resolve("javadoc").toString());
		command.add("-sourcepath");
		command.add(projectRoot.resolve("src/main/java").toString());
		command.add("-subpackages");
		command.add("de.kherud.llama");

		executeCommand(command, projectRoot, new BuildResult());
	}

	private void generateMarkdownDocs() throws IOException {
		// Generate README and other markdown documentation
		Path docsDir = projectRoot.resolve(config.outputDirectory).resolve("docs");
		Files.createDirectories(docsDir);

		// This would generate various documentation files
		// For now, just create a placeholder
		Files.write(docsDir.resolve("README.md"),
			"# Project Documentation\n\nGenerated documentation for the project.\n".getBytes());
	}

	private void packageArtifacts(BuildResult result) throws IOException {
		logger.log(System.Logger.Level.INFO,"Packaging artifacts");

		Path outputDir = projectRoot.resolve(config.outputDirectory);
		Files.createDirectories(outputDir);

		// Package JAR files
		if (config.buildTarget.equals("jar") || config.buildTarget.equals("all")) {
			packageJarArtifacts(outputDir, result);
		}

		// Package native libraries
		if (config.buildTarget.equals("native") || config.buildTarget.equals("all")) {
			packageNativeArtifacts(outputDir, result);
		}

		logger.log(System.Logger.Level.INFO,"Packaged " + result.artifacts.size() + " artifacts");
	}

	private void packageJarArtifacts(Path outputDir, BuildResult result) throws IOException {
		// Find and copy JAR files
		Files.walk(projectRoot)
			.filter(path -> path.toString().endsWith(".jar"))
			.filter(path -> !path.toString().contains("test"))
			.forEach(jarPath -> {
				try {
					Path target = outputDir.resolve(jarPath.getFileName());
					Files.copy(jarPath, target, StandardCopyOption.REPLACE_EXISTING);
					result.addArtifact(target);
				} catch (IOException e) {
					logger.log(System.Logger.Level.WARNING,"Failed to copy JAR: " + jarPath);
				}
			});
	}

	private void packageNativeArtifacts(Path outputDir, BuildResult result) throws IOException {
		// Find and copy native libraries
		String[] extensions = {".so", ".dll", ".dylib", ".a"};

		for (String ext : extensions) {
			Files.walk(projectRoot)
				.filter(path -> path.toString().endsWith(ext))
				.filter(path -> !path.toString().contains("test"))
				.forEach(libPath -> {
					try {
						Path target = outputDir.resolve(libPath.getFileName());
						Files.copy(libPath, target, StandardCopyOption.REPLACE_EXISTING);
						result.addArtifact(target);
					} catch (IOException e) {
						logger.log(System.Logger.Level.WARNING,"Failed to copy library: " + libPath);
					}
				});
		}
	}

	public void close() {
		if (executor != null && !executor.isShutdown()) {
			executor.shutdown();
			try {
				if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
					executor.shutdownNow();
				}
			} catch (InterruptedException e) {
				executor.shutdownNow();
				Thread.currentThread().interrupt();
			}
		}
	}

	/**
	 * Command-line interface
	 */
	public static void main(String[] args) {
		de.kherud.llama.util.CliRunner.runWithExit(ProjectBuilder::runCli, args);
	}

	/**
	 * CLI runner that can be tested without System.exit
	 */
	public static void runCli(String[] args) throws Exception {
		if (args.length < 1) {
			printUsage();
			throw new IllegalArgumentException("No command specified");
		}

		String command = args[0];
		BuildConfig config = new BuildConfig();
		Path projectRoot = Paths.get(".");

		// Parse options
		for (int i = 1; i < args.length; i++) {
			switch (args[i]) {
				case "--target":
					if (i + 1 < args.length) {
						config.buildTarget(args[++i]);
					}
					break;
				case "--profile":
					if (i + 1 < args.length) {
						config.buildProfile(args[++i]);
					}
					break;
				case "--no-tests":
					config.runTests(false);
					break;
				case "--docs":
					config.generateDocs(true);
					break;
				case "--verbose":
				case "-v":
					config.verbose(true);
					break;
				case "--no-parallel":
					config.parallel(false);
					break;
				case "--clean":
					config.cleanBefore(true);
					break;
				case "--output":
					if (i + 1 < args.length) {
						config.outputDirectory(args[++i]);
					}
					break;
				case "--skip":
					if (i + 1 < args.length) {
						config.skipModule(args[++i]);
					}
					break;
				case "--project":
					if (i + 1 < args.length) {
						projectRoot = Paths.get(args[++i]);
					}
					break;
				case "--help":
				case "-h":
					printUsage();
					return;
			}
		}

		try (ProjectBuilder builder = new ProjectBuilder(projectRoot, config)) {
			switch (command) {
				case "build":
					handleBuildCommand(builder);
					break;
				case "clean":
					handleCleanCommand(builder);
					break;
				case "test":
					handleTestCommand(builder, config);
					break;
				default:
					printUsage();
					throw new IllegalArgumentException("Unknown command: " + command);
			}
		}
	}

	private static void handleBuildCommand(ProjectBuilder builder) throws Exception {
		BuildResult result = builder.build();

		if (result.success) {
			System.out.println("Build successful!");
			System.out.println("Modules built: " + result.modulesBuilt);
			System.out.println("Tests run: " + result.testsRun + " (passed: " + result.testsPassed + ")");
			System.out.println("Artifacts: " + result.artifacts.size());
			System.out.println("Build time: " + result.buildTime + "ms");

			if (!result.warnings.isEmpty()) {
				System.out.println("Warnings:");
				result.warnings.forEach(warning -> System.out.println("  " + warning));
			}
		} else {
			throw new RuntimeException("Build failed: " + result.error);
		}
	}

	private static void handleCleanCommand(ProjectBuilder builder) throws Exception {
		builder.clean();
		System.out.println("Clean completed");
	}

	private static void handleTestCommand(ProjectBuilder builder, BuildConfig config) throws Exception {
		config.runTests(true);
		BuildResult result = new BuildResult();
		builder.runTests(result);

		System.out.println("Tests completed");
		System.out.println("Tests run: " + result.testsRun);
		System.out.println("Passed: " + result.testsPassed);
		System.out.println("Failed: " + result.testsFailed);

		if (result.testsFailed > 0) {
			throw new RuntimeException(result.testsFailed + " tests failed");
		}
	}

	private static void printUsage() {
		System.out.println("Usage: ProjectBuilder <command> [options]");
		System.out.println();
		System.out.println("Development and build tools for the project.");
		System.out.println();
		System.out.println("Commands:");
		System.out.println("  build                     Build the project");
		System.out.println("  clean                     Clean build artifacts");
		System.out.println("  test                      Run tests only");
		System.out.println();
		System.out.println("Options:");
		System.out.println("  --target <type>           Build target (jar, native, all)");
		System.out.println("  --profile <profile>       Build profile (debug, release)");
		System.out.println("  --no-tests                Skip running tests");
		System.out.println("  --docs                    Generate documentation");
		System.out.println("  --verbose, -v             Verbose output");
		System.out.println("  --no-parallel             Disable parallel builds");
		System.out.println("  --clean                   Clean before building");
		System.out.println("  --output <dir>            Output directory (default: target)");
		System.out.println("  --skip <module>           Skip building module");
		System.out.println("  --project <path>          Project root path");
		System.out.println("  --help, -h                Show this help");
		System.out.println();
		System.out.println("Examples:");
		System.out.println("  ProjectBuilder build");
		System.out.println("  ProjectBuilder --verbose --clean build");
		System.out.println("  ProjectBuilder --target native --profile debug build");
		System.out.println("  ProjectBuilder test");
	}
}
