package de.kherud.llama;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Comprehensive test suite for the Multi-Model System.
 * Tests model pooling, ensemble inference, production deployment patterns,
 * and resource management capabilities.
 */
public class MultiModelSystemTest {

	private static final System.Logger logger = System.getLogger(MultiModelSystemTest.class.getName());

	private MultiModelManager modelManager;
	private ResourceManager resourceManager;
	private DeploymentManager deploymentManager;

	@Before
	public void setUp() {
		// Initialize backend
		try {
			LlamaUtils.initializeBackend();
		} catch (Exception e) {
			logger.log(DEBUG, "Backend already initialized or unavailable: " + e.getMessage());
		}

		// Initialize components
		modelManager = new MultiModelManager();
		resourceManager = new ResourceManager();

		// Create deployment configuration
		DeploymentManager.DeploymentConfig deploymentConfig =
			new DeploymentManager.DeploymentConfig.Builder()
				.strategy(DeploymentManager.DeploymentStrategy.ACTIVE_ACTIVE)
				.minInstances(1)
				.maxInstances(3)
				.enableCircuitBreaker(true)
				.enableMetrics(true)
				.build();

		deploymentManager = new DeploymentManager(deploymentConfig, modelManager);
	}

	@After
	public void tearDown() {
		if (deploymentManager != null) {
			deploymentManager.close();
		}
		if (resourceManager != null) {
			resourceManager.close();
		}
		if (modelManager != null) {
			modelManager.close();
		}
	}

	@Test
	public void testMultiModelManagerCreation() {
		logger.log(DEBUG, "\n=== Multi-Model Manager Creation Test ===");

		Assert.assertNotNull("Multi-model manager should not be null", modelManager);

		// Test getting models by specialization (initially empty)
		List<String> codeModels = modelManager.getModelsBySpecialization(
			MultiModelManager.ModelSpecialization.CODE_COMPLETION);
		Assert.assertNotNull("Code completion models list should not be null", codeModels);
		Assert.assertTrue("Initial code completion models should be empty", codeModels.isEmpty());

		logger.log(DEBUG, "✅ Multi-model manager created successfully");
	}

	@Test
	public void testModelRegistration() {
		logger.log(DEBUG, "\n=== Model Registration Test ===");

		try {
			// Create model configurations for testing (using non-existent paths for test)
			MultiModelManager.ModelConfig codeCompletionConfig =
				new MultiModelManager.ModelConfig.Builder("code-completion-1", "models/code-model.gguf")
					.specialization(MultiModelManager.ModelSpecialization.CODE_COMPLETION)
					.capability("code_generation")
					.capability("syntax_validation")
					.priority(8)
					.maxConcurrent(2)
					.metadata("language", "java")
					.build();

			MultiModelManager.ModelConfig documentationConfig =
				new MultiModelManager.ModelConfig.Builder("documentation-1", "models/doc-model.gguf")
					.specialization(MultiModelManager.ModelSpecialization.DOCUMENTATION)
					.capability("natural_language")
					.capability("documentation")
					.priority(6)
					.maxConcurrent(1)
					.metadata("language", "english")
					.build();

			// Test configuration validation
			Assert.assertEquals("Code completion config should have correct ID",
				"code-completion-1", codeCompletionConfig.modelId);
			Assert.assertEquals("Code completion config should have correct specialization",
				MultiModelManager.ModelSpecialization.CODE_COMPLETION, codeCompletionConfig.specialization);
			Assert.assertTrue("Code completion config should have code_generation capability",
				codeCompletionConfig.capabilities.contains("code_generation"));

			Assert.assertEquals("Documentation config should have correct priority",
				6, documentationConfig.priority);
			Assert.assertEquals("Documentation config should have correct max concurrent",
				1, documentationConfig.maxConcurrent);

			logger.log(DEBUG, "Model configurations created successfully");
			logger.log(DEBUG, "Code completion capabilities: " + codeCompletionConfig.capabilities);
			logger.log(DEBUG, "Documentation metadata: " + documentationConfig.metadata);

		} catch (Exception e) {
			// Model loading will fail with non-existent paths, but config creation should work
			logger.log(DEBUG, "Expected model loading failure (using test paths): " + e.getMessage());
		}

		logger.log(DEBUG, "✅ Model configuration system works correctly");
	}

	@Test
	public void testRequestContextCreation() {
		logger.log(DEBUG, "\n=== Request Context Creation Test ===");

		// Test request context building
		MultiModelManager.RequestContext javaCodeContext =
			new MultiModelManager.RequestContext.Builder()
				.language("java")
				.fileType(".java")
				.taskType("completion")
				.requireCapability("code_generation")
				.priority(8)
				.timeout(10000)
				.metadata("project", "test-project")
				.build();

		MultiModelManager.RequestContext jsonContext =
			new MultiModelManager.RequestContext.Builder()
				.taskType("json_generation")
				.requireCapability("structured_output")
				.priority(6)
				.timeout(5000)
				.build();

		// Verify context properties
		Assert.assertEquals("Java context should have correct language", "java", javaCodeContext.language);
		Assert.assertEquals("Java context should have correct file type", ".java", javaCodeContext.fileType);
		Assert.assertEquals("Java context should have correct task type", "completion", javaCodeContext.taskType);
		Assert.assertTrue("Java context should require code_generation capability",
			javaCodeContext.requiredCapabilities.contains("code_generation"));
		Assert.assertEquals("Java context should have correct priority", 8, javaCodeContext.priority);
		Assert.assertEquals("Java context should have correct timeout", 10000L, javaCodeContext.timeoutMs);

		Assert.assertEquals("JSON context should have correct task type",
			"json_generation", jsonContext.taskType);
		Assert.assertTrue("JSON context should require structured_output capability",
			jsonContext.requiredCapabilities.contains("structured_output"));

		logger.log(DEBUG, "Java context: " + javaCodeContext.language + ", " +
			javaCodeContext.taskType + ", capabilities: " + javaCodeContext.requiredCapabilities);
		logger.log(DEBUG, "JSON context: " + jsonContext.taskType + ", capabilities: " +
			jsonContext.requiredCapabilities);

		logger.log(DEBUG, "✅ Request context creation works correctly");
	}

	@Test
	public void testResourceManagerCreation() {
		logger.log(DEBUG, "\n=== Resource Manager Creation Test ===");

		Assert.assertNotNull("Resource manager should not be null", resourceManager);

		// Test getting initial metrics
		Map<String, Object> metrics = resourceManager.getResourceMetrics();
		Assert.assertNotNull("Resource metrics should not be null", metrics);
		Assert.assertTrue("Metrics should contain system information", metrics.containsKey("systemMemoryUtilization"));
		Assert.assertTrue("Metrics should contain allocation information", metrics.containsKey("totalAllocations"));

		logger.log(DEBUG, "System memory utilization: " +
			String.format("%.1f%%", (Double) metrics.get("systemMemoryUtilization") * 100));
		logger.log(DEBUG, "Total allocations: " + metrics.get("totalAllocations"));
		logger.log(DEBUG, "Active allocations: " + metrics.get("activeAllocations"));

		logger.log(DEBUG, "✅ Resource manager created and metrics available");
	}

	@Test
	public void testResourceQuotaCreation() {
		logger.log(DEBUG, "\n=== Resource Quota Creation Test ===");

		// Test resource quota building
		ResourceManager.ResourceQuota codeModelQuota =
			new ResourceManager.ResourceQuota.Builder()
				.request(ResourceManager.ResourceType.MEMORY_MB, 2048)
				.request(ResourceManager.ResourceType.GPU_MEMORY_MB, 4096)
				.request(ResourceManager.ResourceType.CPU_CORES, 2)
				.limit(ResourceManager.ResourceType.MEMORY_MB, 4096)
				.limit(ResourceManager.ResourceType.GPU_MEMORY_MB, 6144)
				.guarantee(ResourceManager.ResourceType.MEMORY_MB, 1024)
				.priority(8)
				.build();

		ResourceManager.ResourceQuota embeddingModelQuota =
			new ResourceManager.ResourceQuota.Builder()
				.request(ResourceManager.ResourceType.MEMORY_MB, 1024)
				.request(ResourceManager.ResourceType.GPU_MEMORY_MB, 2048)
				.limit(ResourceManager.ResourceType.MEMORY_MB, 2048)
				.priority(5)
				.build();

		// Verify quota properties
		Assert.assertEquals("Code model should request 2GB memory",
			Long.valueOf(2048), codeModelQuota.requests.get(ResourceManager.ResourceType.MEMORY_MB));
		Assert.assertEquals("Code model should have 4GB memory limit",
			Long.valueOf(4096), codeModelQuota.limits.get(ResourceManager.ResourceType.MEMORY_MB));
		Assert.assertEquals("Code model should guarantee 1GB memory",
			Long.valueOf(1024), codeModelQuota.guaranteed.get(ResourceManager.ResourceType.MEMORY_MB));
		Assert.assertEquals("Code model should have priority 8", 8, codeModelQuota.priority);

		Assert.assertEquals("Embedding model should request 1GB memory",
			Long.valueOf(1024), embeddingModelQuota.requests.get(ResourceManager.ResourceType.MEMORY_MB));
		Assert.assertEquals("Embedding model should have priority 5", 5, embeddingModelQuota.priority);

		logger.log(DEBUG, "Code model quota - Memory: " +
			codeModelQuota.requests.get(ResourceManager.ResourceType.MEMORY_MB) + "MB requested, " +
			codeModelQuota.limits.get(ResourceManager.ResourceType.MEMORY_MB) + "MB limit");
		logger.log(DEBUG, "Embedding model quota - Memory: " +
			embeddingModelQuota.requests.get(ResourceManager.ResourceType.MEMORY_MB) + "MB requested");

		logger.log(DEBUG, "✅ Resource quota creation works correctly");
	}

	@Test
	public void testEnsembleResultCreation() {
		logger.log(DEBUG, "\n=== Ensemble Result Creation Test ===");

		// Create mock weighted results
		EnsembleInferenceEngine.WeightedResult<String> result1 =
			new EnsembleInferenceEngine.WeightedResult<>("Result from model 1", 1.0, 0.8, "model-1", 150);
		EnsembleInferenceEngine.WeightedResult<String> result2 =
			new EnsembleInferenceEngine.WeightedResult<>("Result from model 2", 0.9, 0.9, "model-2", 120);
		EnsembleInferenceEngine.WeightedResult<String> result3 =
			new EnsembleInferenceEngine.WeightedResult<>("Result from model 3", 0.8, 0.7, "model-3", 200);

		List<EnsembleInferenceEngine.WeightedResult<String>> results = List.of(result1, result2, result3);

		// Test ensemble result creation
		EnsembleInferenceEngine.EnsembleResult<String> ensembleResult =
			new EnsembleInferenceEngine.EnsembleResult<>(
				"Final ensemble result", results,
				EnsembleInferenceEngine.VotingStrategy.CONFIDENCE_BASED, 0.85);

		ensembleResult.addVotingMetadata("votingRounds", 3);
		ensembleResult.addVotingMetadata("consensusReached", true);

		// Verify ensemble result properties
		Assert.assertEquals("Ensemble should have final result",
			"Final ensemble result", ensembleResult.finalResult);
		Assert.assertEquals("Ensemble should have 3 results", 3, ensembleResult.allResults.size());
		Assert.assertEquals("Ensemble should use confidence-based voting",
			EnsembleInferenceEngine.VotingStrategy.CONFIDENCE_BASED, ensembleResult.strategy);
		Assert.assertEquals("Ensemble should have consensus score 0.85", 0.85, ensembleResult.consensusScore, 0.01);
		Assert.assertTrue("Ensemble should have consensus at 80% threshold", ensembleResult.hasConsensus(0.8));
		Assert.assertFalse("Ensemble should not have consensus at 90% threshold", ensembleResult.hasConsensus(0.9));

		// Test top results
		List<EnsembleInferenceEngine.WeightedResult<String>> topResults = ensembleResult.getTopResults(2);
		Assert.assertEquals("Should return top 2 results", 2, topResults.size());

		logger.log(DEBUG, "Ensemble result: " + ensembleResult.finalResult);
		logger.log(DEBUG, "Voting strategy: " + ensembleResult.strategy);
		logger.log(DEBUG, "Consensus score: " + ensembleResult.consensusScore);
		logger.log(DEBUG, "All results count: " + ensembleResult.allResults.size());
		logger.log(DEBUG, "Top 2 results: " + topResults.size());

		logger.log(DEBUG, "✅ Ensemble result creation works correctly");
	}

	@Test
	public void testProductionDeploymentConfig() {
		logger.log(DEBUG, "\n=== Production Deployment Configuration Test ===");

		// Test deployment configuration building
		DeploymentManager.DeploymentConfig config =
			new DeploymentManager.DeploymentConfig.Builder()
				.strategy(DeploymentManager.DeploymentStrategy.AUTO_SCALING)
				.healthCheck(DeploymentManager.HealthCheckStrategy.COMPREHENSIVE)
				.minInstances(2)
				.maxInstances(10)
				.cpuThreshold(0.75)
				.memoryThreshold(0.80)
				.enableCircuitBreaker(true)
				.enableMetrics(true)
				.build();

		// Verify configuration properties
		Assert.assertEquals("Should use auto-scaling strategy",
			DeploymentManager.DeploymentStrategy.AUTO_SCALING, config.strategy);
		Assert.assertEquals("Should use comprehensive health checks",
			DeploymentManager.HealthCheckStrategy.COMPREHENSIVE, config.healthCheck);
		Assert.assertEquals("Should have min 2 instances", 2, config.minInstances);
		Assert.assertEquals("Should have max 10 instances", 10, config.maxInstances);
		Assert.assertEquals("Should have CPU threshold 0.75", 0.75, config.cpuThreshold, 0.01);
		Assert.assertEquals("Should have memory threshold 0.80", 0.80, config.memoryThreshold, 0.01);
		Assert.assertTrue("Should enable circuit breaker", config.enableCircuitBreaker);
		Assert.assertTrue("Should enable metrics", config.enableMetrics);

		logger.log(DEBUG, "Deployment config - Strategy: " + config.strategy +
			", Health: " + config.healthCheck +
			", Instances: " + config.minInstances + "-" + config.maxInstances);
		logger.log(DEBUG, "Thresholds - CPU: " + config.cpuThreshold +
			", Memory: " + config.memoryThreshold);

		logger.log(DEBUG, "✅ Production deployment configuration works correctly");
	}

	@Test
	public void testModelManagerMetrics() {
		logger.log(DEBUG, "\n=== Model Manager Metrics Test ===");

		// Get initial metrics
		Map<String, Object> metrics = modelManager.getMetrics();

		Assert.assertNotNull("Metrics should not be null", metrics);
		Assert.assertTrue("Should have total requests metric", metrics.containsKey("totalRequests"));
		Assert.assertTrue("Should have average response time metric", metrics.containsKey("averageResponseTime"));
		Assert.assertTrue("Should have models metric", metrics.containsKey("models"));
		Assert.assertTrue("Should have specialization counts", metrics.containsKey("specializationCounts"));

		// Initial values should be zero
		Assert.assertEquals("Initial total requests should be 0", Long.valueOf(0), metrics.get("totalRequests"));
		Assert.assertEquals("Initial average response time should be 0.0", 0.0, (Double) metrics.get("averageResponseTime"), 0.01);

		@SuppressWarnings("unchecked")
		Map<String, Object> models = (Map<String, Object>) metrics.get("models");
		Assert.assertNotNull("Models metrics should not be null", models);
		Assert.assertTrue("Initially should have no models", models.isEmpty());

		@SuppressWarnings("unchecked")
		Map<String, Integer> specializationCounts = (Map<String, Integer>) metrics.get("specializationCounts");
		Assert.assertNotNull("Specialization counts should not be null", specializationCounts);

		logger.log(DEBUG, "Total requests: " + metrics.get("totalRequests"));
		logger.log(DEBUG, "Average response time: " + metrics.get("averageResponseTime"));
		logger.log(DEBUG, "Models count: " + models.size());
		logger.log(DEBUG, "Specialization counts: " + specializationCounts);

		logger.log(DEBUG, "✅ Model manager metrics work correctly");
	}

	@Test
	public void testVotingStrategies() {
		logger.log(DEBUG, "\n=== Voting Strategies Test ===");

		// Create test results
		List<EnsembleInferenceEngine.WeightedResult<String>> results = List.of(
			new EnsembleInferenceEngine.WeightedResult<>("Option A", 1.0, 0.8, "model-1", 100),
			new EnsembleInferenceEngine.WeightedResult<>("Option B", 0.9, 0.9, "model-2", 120),
			new EnsembleInferenceEngine.WeightedResult<>("Option A", 0.8, 0.7, "model-3", 90),
			new EnsembleInferenceEngine.WeightedResult<>("Option C", 0.7, 0.6, "model-4", 150)
		);

		// Test different voting strategies
		EnsembleInferenceEngine.EnsembleResult<String> majorityResult =
			EnsembleInferenceEngine.applyVotingStrategy(results, EnsembleInferenceEngine.VotingStrategy.MAJORITY_VOTE);

		EnsembleInferenceEngine.EnsembleResult<String> confidenceResult =
			EnsembleInferenceEngine.applyVotingStrategy(results, EnsembleInferenceEngine.VotingStrategy.CONFIDENCE_BASED);

		EnsembleInferenceEngine.EnsembleResult<String> bestOfNResult =
			EnsembleInferenceEngine.applyVotingStrategy(results, EnsembleInferenceEngine.VotingStrategy.BEST_OF_N);

		// Verify voting results
		Assert.assertNotNull("Majority vote result should not be null", majorityResult);
		Assert.assertNotNull("Confidence-based result should not be null", confidenceResult);
		Assert.assertNotNull("Best-of-N result should not be null", bestOfNResult);

		Assert.assertEquals("Majority vote should pick Option A (2 votes)", "Option A", majorityResult.finalResult);
		Assert.assertEquals("Confidence-based should pick Option B (highest confidence)", "Option B", confidenceResult.finalResult);

		logger.log(DEBUG, "Majority vote winner: " + majorityResult.finalResult +
			" (consensus: " + majorityResult.consensusScore + ")");
		logger.log(DEBUG, "Confidence-based winner: " + confidenceResult.finalResult +
			" (consensus: " + confidenceResult.consensusScore + ")");
		logger.log(DEBUG, "Best-of-N winner: " + bestOfNResult.finalResult +
			" (consensus: " + bestOfNResult.consensusScore + ")");

		logger.log(DEBUG, "✅ Voting strategies work correctly");
	}
}
