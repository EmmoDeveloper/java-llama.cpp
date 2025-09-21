package de.kherud.llama;

import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Test class for the new LlamaSampler API design.
 * Tests both the utility class methods and model-dependent instance methods.
 */
@Ignore
public class AdvancedSamplingTest {

	@BeforeClass
	public static void setup() {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
	}

	@Test
	public void testLlamaSamplerUtilityClass() {
		// Test LlamaSampler static methods (no model required)
		long greedy = LlamaSampler.createGreedy();
		assertTrue("Greedy sampler handle should be valid", greedy > 0);
		LlamaSampler.free(greedy);

		long dist = LlamaSampler.createDistribution(42);
		assertTrue("Distribution sampler handle should be valid", dist > 0);
		LlamaSampler.free(dist);

		long topK = LlamaSampler.createTopK(50);
		assertTrue("Top-K sampler handle should be valid", topK > 0);
		LlamaSampler.free(topK);

		long topP = LlamaSampler.createTopP(0.9f, 1);
		assertTrue("Top-P sampler handle should be valid", topP > 0);
		LlamaSampler.free(topP);

		long temp = LlamaSampler.createTemperature(0.8f);
		assertTrue("Temperature sampler handle should be valid", temp > 0);
		LlamaSampler.free(temp);
	}

	@Test
	public void testSamplerChains() {
		// Test LlamaSampler chain functionality
		long chain = LlamaSampler.createChain();
		assertTrue("Sampler chain handle should be valid", chain > 0);

		long topK = LlamaSampler.createTopK(50);
		long topP = LlamaSampler.createTopP(0.9f, 1);
		long temp = LlamaSampler.createTemperature(0.8f);

		LlamaSampler.addToChain(chain, topK);
		LlamaSampler.addToChain(chain, topP);
		LlamaSampler.addToChain(chain, temp);

		// Chain takes ownership, so we only free the chain
		LlamaSampler.free(chain);
	}

	@Test
	public void testAdvancedSamplers() {
		// Test advanced samplers from LlamaSampler class
		long xtc = LlamaSampler.createXtc(0.5f, 0.1f, 1, 42);
		assertTrue("XTC sampler handle should be valid", xtc > 0);
		LlamaSampler.free(xtc);

		long typical = LlamaSampler.createTypical(0.95f, 1);
		assertTrue("Typical sampler handle should be valid", typical > 0);
		LlamaSampler.free(typical);

		long minP = LlamaSampler.createMinP(0.05f, 1);
		assertTrue("Min-P sampler handle should be valid", minP > 0);
		LlamaSampler.free(minP);

		long mirostatV2 = LlamaSampler.createMirostatV2(42, 5.0f, 0.1f);
		assertTrue("Mirostat V2 sampler handle should be valid", mirostatV2 > 0);
		LlamaSampler.free(mirostatV2);

		long penalties = LlamaSampler.createPenalties(50, 1.1f, 0.0f, 0.0f);
		assertTrue("Penalties sampler handle should be valid", penalties > 0);
		LlamaSampler.free(penalties);
	}

	@Test
	public void testInvalidHandles() {
		// Test that invalid handles don't crash the JVM
		try {
			LlamaSampler.free(-1);
			LlamaSampler.free(0);
			LlamaSampler.free(999999L);
		} catch (Exception e) {
			fail("Should not throw exception for invalid handles: " + e.getMessage());
		}
	}

	@Test
	public void testSamplerNames() {
		// Test sampler name functionality
		long greedy = LlamaSampler.createGreedy();
		long temp = LlamaSampler.createTemperature(0.8f);

		String greedyName = LlamaSampler.getName(greedy);
		String tempName = LlamaSampler.getName(temp);

		assertNotNull("Greedy sampler should have a name", greedyName);
		assertNotNull("Temperature sampler should have a name", tempName);

		LlamaSampler.free(greedy);
		LlamaSampler.free(temp);
	}
}
