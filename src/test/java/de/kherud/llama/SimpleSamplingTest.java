package de.kherud.llama;

import org.junit.Test;

import static org.junit.Assert.*;

public class SimpleSamplingTest {

	@Test
	public void testCreateGreedySamplerOnly() throws Exception {
		System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
		
		try {
			long handle = LlamaModel.createGreedySampler();
			assertTrue("Greedy sampler handle should be valid", handle > 0);
			LlamaModel.freeSampler(handle);
		} catch (Exception e) {
			fail("Greedy sampler creation failed: " + e.getMessage());
		}
	}
}