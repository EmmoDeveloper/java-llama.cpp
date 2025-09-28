package de.kherud.llama.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Library-friendly project builder and development tools.
 *
 * This refactored version provides a fluent API for project building,
 * with builder pattern configuration, progress callbacks, and async operations.
 *
 * Usage examples:
 * <pre>{@code
 * // Basic project build
 * BuildResult result = ProjectBuilderLibrary.builder()
 *     .projectPath(Paths.get("."))
 *     .build()
 *     .buildProject();
 *
 * // Configured build
 * BuildResult result = ProjectBuilderLibrary.builder()
 *     .projectPath(projectPath)
 *     .buildTarget("native")
 *     .buildProfile("release")
 *     .runTests(true)
 *     .generateDocs(false)
 *     .parallel(true)
 *     .maxConcurrency(8)
 *     .cleanBeforeBuild(true)
 *     .progressCallback(progress -> System.out.println(progress.getMessage()))
 *     .build()
 *     .buildProject();
 *
 * // Async build
 * ProjectBuilderLibrary.builder()
 *     .projectPath(projectPath)
 *     .build()
 *     .buildProjectAsync()
 *     .thenAccept(result -> System.out.println("Build complete: " + result.isSuccess()));
 *
 * // Test-only build
 * TestResult testResult = builder.runTests();
 *
 * // Clean project
 * builder.cleanProject();
 * }</pre>
 */
public class ProjectBuilderLibrary implements AutoCloseable {
	private static final System.Logger LOGGER = System.getLogger(ProjectBuilderLibrary.class.getName());

	private final Path projectPath;
	private final String buildTarget;
	private final String buildProfile;
	private final boolean runTests;
	private final boolean generateDocs;
	private final boolean verbose;
	private final boolean parallel;
	private final int maxConcurrency;
	private final List<String> skipModules;
	private final Map<String, String> buildProperties;
	private final String outputDirectory;
	private final boolean cleanBeforeBuild;
	private final Consumer<BuildProgress> progressCallback;
	private final ExecutorService executor;

	private ProjectBuilderLibrary(Builder builder) {
		this.projectPath = Objects.requireNonNull(builder.projectPath, "Project path cannot be null");
		this.buildTarget = builder.buildTarget;
		this.buildProfile = builder.buildProfile;
		this.runTests = builder.runTests;
		this.generateDocs = builder.generateDocs;
		this.verbose = builder.verbose;
		this.parallel = builder.parallel;
		this.maxConcurrency = builder.maxConcurrency;
		this.skipModules = Collections.unmodifiableList(builder.skipModules);
		this.buildProperties = Collections.unmodifiableMap(builder.buildProperties);
		this.outputDirectory = builder.outputDirectory;
		this.cleanBeforeBuild = builder.cleanBeforeBuild;
		this.progressCallback = builder.progressCallback;
		this.executor = builder.executor;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Build the project
	 */
	public BuildResult buildProject() throws IOException {
		validateProjectPath();

		progress("Starting project build", 0.0);
		Instant startTime = Instant.now();

		try {
			// Clean if requested
			if (cleanBeforeBuild) {
				progress("Cleaning project", 0.1);
				cleanProject();
			}

			// Build config for original builder
			ProjectBuilder.BuildConfig config = new ProjectBuilder.BuildConfig()
				.buildTarget(buildTarget)
				.buildProfile(buildProfile)
				.runTests(runTests)
				.generateDocs(generateDocs)
				.verbose(verbose)
				.parallel(parallel)
				.maxConcurrency(maxConcurrency)
				.outputDirectory(outputDirectory)
				.cleanBefore(cleanBeforeBuild);

			// Add skip modules
			for (String module : skipModules) {
				config.skipModule(module);
			}

			// Add build properties
			for (Map.Entry<String, String> entry : buildProperties.entrySet()) {
				config.addProperty(entry.getKey(), entry.getValue());
			}

			progress("Executing build", 0.3);

			// Use the original builder for the actual work
			try (ProjectBuilder projectBuilder = new ProjectBuilder(projectPath, config)) {
				ProjectBuilder.BuildResult originalResult = projectBuilder.build();

				progress("Build complete", 1.0);

				// Convert to our result format
				return new BuildResult.Builder()
					.success(originalResult.success)
					.message(originalResult.success ? "Build successful" : originalResult.error)
					.projectPath(projectPath)
					.buildTarget(buildTarget)
					.buildProfile(buildProfile)
					.modulesBuilt(originalResult.modulesBuilt)
					.testsRun(originalResult.testsRun)
					.testsPassed(originalResult.testsPassed)
					.testsFailed(originalResult.testsFailed)
					.artifacts(originalResult.artifacts)
					.warnings(originalResult.warnings)
					.duration(Duration.between(startTime, Instant.now()))
					.buildStats(originalResult.buildStats)
					.build();
			}

		} catch (Exception e) {
			String errorMsg = "Build failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new BuildResult.Builder()
				.success(false)
				.message(errorMsg)
				.projectPath(projectPath)
				.buildTarget(buildTarget)
				.buildProfile(buildProfile)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Build the project asynchronously
	 */
	public CompletableFuture<BuildResult> buildProjectAsync() {
		ExecutorService exec = executor != null ? executor : Executors.newSingleThreadExecutor();

		return CompletableFuture.supplyAsync(() -> {
			try {
				return buildProject();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}, exec);
	}

	/**
	 * Run tests only
	 */
	public TestResult runTests() throws IOException {
		validateProjectPath();

		progress("Running tests", 0.0);
		Instant startTime = Instant.now();

		try {
			ProjectBuilder.BuildConfig config = new ProjectBuilder.BuildConfig()
				.buildTarget("test")
				.runTests(true)
				.verbose(verbose)
				.parallel(parallel)
				.maxConcurrency(maxConcurrency);

			try (ProjectBuilder projectBuilder = new ProjectBuilder(projectPath, config)) {
				ProjectBuilder.BuildResult originalResult = projectBuilder.build();

				progress("Tests complete", 1.0);

				return new TestResult.Builder()
					.success(originalResult.success && originalResult.testsFailed == 0)
					.message(originalResult.success ? "Tests completed" : originalResult.error)
					.projectPath(projectPath)
					.testsRun(originalResult.testsRun)
					.testsPassed(originalResult.testsPassed)
					.testsFailed(originalResult.testsFailed)
					.duration(Duration.between(startTime, Instant.now()))
					.build();
			}

		} catch (Exception e) {
			String errorMsg = "Test execution failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new TestResult.Builder()
				.success(false)
				.message(errorMsg)
				.projectPath(projectPath)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Clean the project
	 */
	public CleanResult cleanProject() throws IOException {
		validateProjectPath();

		progress("Cleaning project", 0.0);
		Instant startTime = Instant.now();

		try {
			ProjectBuilder.BuildConfig config = new ProjectBuilder.BuildConfig()
				.outputDirectory(outputDirectory)
				.verbose(verbose);

			try (ProjectBuilder projectBuilder = new ProjectBuilder(projectPath, config)) {
				projectBuilder.clean();

				progress("Clean complete", 1.0);

				return new CleanResult.Builder()
					.success(true)
					.message("Project cleaned successfully")
					.projectPath(projectPath)
					.duration(Duration.between(startTime, Instant.now()))
					.build();
			}

		} catch (Exception e) {
			String errorMsg = "Clean failed: " + e.getMessage();
			LOGGER.log(System.Logger.Level.ERROR, errorMsg, e);

			return new CleanResult.Builder()
				.success(false)
				.message(errorMsg)
				.projectPath(projectPath)
				.duration(Duration.between(startTime, Instant.now()))
				.error(e)
				.build();
		}
	}

	/**
	 * Get project information
	 */
	public ProjectInfo getProjectInfo() throws IOException {
		validateProjectPath();

		// Analyze project structure
		List<ModuleInfo> modules = new ArrayList<>();
		Map<String, Object> projectProperties = new HashMap<>();

		// Basic project analysis
		boolean hasPomXml = Files.exists(projectPath.resolve("pom.xml"));
		boolean hasBuildGradle = Files.exists(projectPath.resolve("build.gradle"));
		boolean hasCmakeLists = Files.exists(projectPath.resolve("CMakeLists.txt"));
		boolean hasSrcDir = Files.exists(projectPath.resolve("src"));

		String projectType = "unknown";
		if (hasPomXml) projectType = "maven";
		else if (hasBuildGradle) projectType = "gradle";
		else if (hasCmakeLists) projectType = "cmake";

		projectProperties.put("projectType", projectType);
		projectProperties.put("hasPom", hasPomXml);
		projectProperties.put("hasGradle", hasBuildGradle);
		projectProperties.put("hasCMake", hasCmakeLists);
		projectProperties.put("hasSrc", hasSrcDir);

		return new ProjectInfo.Builder()
			.projectPath(projectPath)
			.projectType(projectType)
			.modules(modules)
			.properties(projectProperties)
			.build();
	}

	// Helper methods
	private void validateProjectPath() throws IOException {
		if (!Files.exists(projectPath)) {
			throw new IOException("Project path does not exist: " + projectPath);
		}
		if (!Files.isDirectory(projectPath)) {
			throw new IOException("Project path is not a directory: " + projectPath);
		}
	}

	private void progress(String message, double progress) {
		if (progressCallback != null) {
			progressCallback.accept(new BuildProgress(message, progress));
		}
	}

	@Override
	public void close() throws IOException {
		if (executor != null) {
			executor.shutdown();
		}
	}

	// Builder class
	public static class Builder {
		private Path projectPath;
		private String buildTarget = "jar";
		private String buildProfile = "release";
		private boolean runTests = true;
		private boolean generateDocs = false;
		private boolean verbose = false;
		private boolean parallel = true;
		private int maxConcurrency = Runtime.getRuntime().availableProcessors();
		private List<String> skipModules = new ArrayList<>();
		private Map<String, String> buildProperties = new HashMap<>();
		private String outputDirectory = "target";
		private boolean cleanBeforeBuild = false;
		private Consumer<BuildProgress> progressCallback;
		private ExecutorService executor;

		public Builder projectPath(Path projectPath) {
			this.projectPath = projectPath;
			return this;
		}

		public Builder buildTarget(String buildTarget) {
			this.buildTarget = buildTarget;
			return this;
		}

		public Builder buildProfile(String buildProfile) {
			this.buildProfile = buildProfile;
			return this;
		}

		public Builder runTests(boolean runTests) {
			this.runTests = runTests;
			return this;
		}

		public Builder generateDocs(boolean generateDocs) {
			this.generateDocs = generateDocs;
			return this;
		}

		public Builder verbose(boolean verbose) {
			this.verbose = verbose;
			return this;
		}

		public Builder parallel(boolean parallel) {
			this.parallel = parallel;
			return this;
		}

		public Builder maxConcurrency(int maxConcurrency) {
			this.maxConcurrency = Math.max(1, maxConcurrency);
			return this;
		}

		public Builder skipModule(String module) {
			this.skipModules.add(module);
			return this;
		}

		public Builder skipModules(List<String> modules) {
			this.skipModules.addAll(modules);
			return this;
		}

		public Builder buildProperty(String key, String value) {
			this.buildProperties.put(key, value);
			return this;
		}

		public Builder buildProperties(Map<String, String> properties) {
			this.buildProperties.putAll(properties);
			return this;
		}

		public Builder outputDirectory(String outputDirectory) {
			this.outputDirectory = outputDirectory;
			return this;
		}

		public Builder cleanBeforeBuild(boolean cleanBeforeBuild) {
			this.cleanBeforeBuild = cleanBeforeBuild;
			return this;
		}

		public Builder progressCallback(Consumer<BuildProgress> progressCallback) {
			this.progressCallback = progressCallback;
			return this;
		}

		public Builder executor(ExecutorService executor) {
			this.executor = executor;
			return this;
		}

		public ProjectBuilderLibrary build() {
			return new ProjectBuilderLibrary(this);
		}
	}

	// Progress tracking class
	public static class BuildProgress {
		private final String message;
		private final double progress;
		private final Instant timestamp;

		public BuildProgress(String message, double progress) {
			this.message = message;
			this.progress = Math.max(0.0, Math.min(1.0, progress));
			this.timestamp = Instant.now();
		}

		public String getMessage() { return message; }
		public double getProgress() { return progress; }
		public Instant getTimestamp() { return timestamp; }
	}

	// Result classes
	public static class BuildResult {
		private final boolean success;
		private final String message;
		private final Path projectPath;
		private final String buildTarget;
		private final String buildProfile;
		private final int modulesBuilt;
		private final int testsRun;
		private final int testsPassed;
		private final int testsFailed;
		private final List<Path> artifacts;
		private final List<String> warnings;
		private final Duration duration;
		private final Map<String, Object> buildStats;
		private final Exception error;

		private BuildResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.projectPath = builder.projectPath;
			this.buildTarget = builder.buildTarget;
			this.buildProfile = builder.buildProfile;
			this.modulesBuilt = builder.modulesBuilt;
			this.testsRun = builder.testsRun;
			this.testsPassed = builder.testsPassed;
			this.testsFailed = builder.testsFailed;
			this.artifacts = Collections.unmodifiableList(builder.artifacts);
			this.warnings = Collections.unmodifiableList(builder.warnings);
			this.duration = builder.duration;
			this.buildStats = Collections.unmodifiableMap(builder.buildStats);
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Path getProjectPath() { return projectPath; }
		public String getBuildTarget() { return buildTarget; }
		public String getBuildProfile() { return buildProfile; }
		public int getModulesBuilt() { return modulesBuilt; }
		public int getTestsRun() { return testsRun; }
		public int getTestsPassed() { return testsPassed; }
		public int getTestsFailed() { return testsFailed; }
		public List<Path> getArtifacts() { return artifacts; }
		public List<String> getWarnings() { return warnings; }
		public Duration getDuration() { return duration; }
		public Map<String, Object> getBuildStats() { return buildStats; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public boolean hasWarnings() { return !warnings.isEmpty(); }
		public boolean hasTestFailures() { return testsFailed > 0; }

		public static class Builder {
			private boolean success;
			private String message;
			private Path projectPath;
			private String buildTarget;
			private String buildProfile;
			private int modulesBuilt;
			private int testsRun;
			private int testsPassed;
			private int testsFailed;
			private List<Path> artifacts = new ArrayList<>();
			private List<String> warnings = new ArrayList<>();
			private Duration duration = Duration.ZERO;
			private Map<String, Object> buildStats = new HashMap<>();
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder projectPath(Path projectPath) { this.projectPath = projectPath; return this; }
			public Builder buildTarget(String buildTarget) { this.buildTarget = buildTarget; return this; }
			public Builder buildProfile(String buildProfile) { this.buildProfile = buildProfile; return this; }
			public Builder modulesBuilt(int modulesBuilt) { this.modulesBuilt = modulesBuilt; return this; }
			public Builder testsRun(int testsRun) { this.testsRun = testsRun; return this; }
			public Builder testsPassed(int testsPassed) { this.testsPassed = testsPassed; return this; }
			public Builder testsFailed(int testsFailed) { this.testsFailed = testsFailed; return this; }
			public Builder artifacts(List<Path> artifacts) { this.artifacts = artifacts; return this; }
			public Builder warnings(List<String> warnings) { this.warnings = warnings; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder buildStats(Map<String, Object> buildStats) { this.buildStats = buildStats; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public BuildResult build() { return new BuildResult(this); }
		}
	}

	public static class TestResult {
		private final boolean success;
		private final String message;
		private final Path projectPath;
		private final int testsRun;
		private final int testsPassed;
		private final int testsFailed;
		private final Duration duration;
		private final Exception error;

		private TestResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.projectPath = builder.projectPath;
			this.testsRun = builder.testsRun;
			this.testsPassed = builder.testsPassed;
			this.testsFailed = builder.testsFailed;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Path getProjectPath() { return projectPath; }
		public int getTestsRun() { return testsRun; }
		public int getTestsPassed() { return testsPassed; }
		public int getTestsFailed() { return testsFailed; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public double getSuccessRate() {
			return testsRun > 0 ? (double) testsPassed / testsRun : 0.0;
		}

		public static class Builder {
			private boolean success;
			private String message;
			private Path projectPath;
			private int testsRun;
			private int testsPassed;
			private int testsFailed;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder projectPath(Path projectPath) { this.projectPath = projectPath; return this; }
			public Builder testsRun(int testsRun) { this.testsRun = testsRun; return this; }
			public Builder testsPassed(int testsPassed) { this.testsPassed = testsPassed; return this; }
			public Builder testsFailed(int testsFailed) { this.testsFailed = testsFailed; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public TestResult build() { return new TestResult(this); }
		}
	}

	public static class CleanResult {
		private final boolean success;
		private final String message;
		private final Path projectPath;
		private final Duration duration;
		private final Exception error;

		private CleanResult(Builder builder) {
			this.success = builder.success;
			this.message = builder.message;
			this.projectPath = builder.projectPath;
			this.duration = builder.duration;
			this.error = builder.error;
		}

		public boolean isSuccess() { return success; }
		public String getMessage() { return message; }
		public Path getProjectPath() { return projectPath; }
		public Duration getDuration() { return duration; }
		public Optional<Exception> getError() { return Optional.ofNullable(error); }

		public static class Builder {
			private boolean success;
			private String message;
			private Path projectPath;
			private Duration duration = Duration.ZERO;
			private Exception error;

			public Builder success(boolean success) { this.success = success; return this; }
			public Builder message(String message) { this.message = message; return this; }
			public Builder projectPath(Path projectPath) { this.projectPath = projectPath; return this; }
			public Builder duration(Duration duration) { this.duration = duration; return this; }
			public Builder error(Exception error) { this.error = error; return this; }

			public CleanResult build() { return new CleanResult(this); }
		}
	}

	public static class ProjectInfo {
		private final Path projectPath;
		private final String projectType;
		private final List<ModuleInfo> modules;
		private final Map<String, Object> properties;

		private ProjectInfo(Builder builder) {
			this.projectPath = builder.projectPath;
			this.projectType = builder.projectType;
			this.modules = Collections.unmodifiableList(builder.modules);
			this.properties = Collections.unmodifiableMap(builder.properties);
		}

		public Path getProjectPath() { return projectPath; }
		public String getProjectType() { return projectType; }
		public List<ModuleInfo> getModules() { return modules; }
		public Map<String, Object> getProperties() { return properties; }

		public static class Builder {
			private Path projectPath;
			private String projectType;
			private List<ModuleInfo> modules = new ArrayList<>();
			private Map<String, Object> properties = new HashMap<>();

			public Builder projectPath(Path projectPath) { this.projectPath = projectPath; return this; }
			public Builder projectType(String projectType) { this.projectType = projectType; return this; }
			public Builder modules(List<ModuleInfo> modules) { this.modules = modules; return this; }
			public Builder properties(Map<String, Object> properties) { this.properties = properties; return this; }

			public ProjectInfo build() { return new ProjectInfo(this); }
		}
	}

	public static class ModuleInfo {
		private final String name;
		private final Path path;
		private final List<String> dependencies;
		private final String type;

		private ModuleInfo(Builder builder) {
			this.name = builder.name;
			this.path = builder.path;
			this.dependencies = Collections.unmodifiableList(builder.dependencies);
			this.type = builder.type;
		}

		public String getName() { return name; }
		public Path getPath() { return path; }
		public List<String> getDependencies() { return dependencies; }
		public String getType() { return type; }

		public static class Builder {
			private String name;
			private Path path;
			private List<String> dependencies = new ArrayList<>();
			private String type;

			public Builder name(String name) { this.name = name; return this; }
			public Builder path(Path path) { this.path = path; return this; }
			public Builder dependencies(List<String> dependencies) { this.dependencies = dependencies; return this; }
			public Builder type(String type) { this.type = type; return this; }

			public ModuleInfo build() { return new ModuleInfo(this); }
		}
	}
}
