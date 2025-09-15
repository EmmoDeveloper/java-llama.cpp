package de.kherud.llama;

import org.junit.Test;
import org.junit.Assert;

import static java.lang.System.Logger.Level.DEBUG;

public class SplitPathTest {

	private static final System.Logger logger = System.getLogger(SplitPathTest.class.getName());

	@Test
	public void testSplitPathBuilding() {
		String basePath = "models/large-model.gguf";

		// Test building split paths
		String split0 = LlamaUtils.buildSplitPath(basePath, 0);
		String split1 = LlamaUtils.buildSplitPath(basePath, 1);
		String split2 = LlamaUtils.buildSplitPath(basePath, 2);

		Assert.assertNotNull("Split path 0 should not be null", split0);
		Assert.assertNotNull("Split path 1 should not be null", split1);
		Assert.assertNotNull("Split path 2 should not be null", split2);

		// Split paths should be different
		Assert.assertNotEquals("Split paths should be different", split0, split1);
		Assert.assertNotEquals("Split paths should be different", split1, split2);

		logger.log(DEBUG, "Split 0: " + split0);
		logger.log(DEBUG, "Split 1: " + split1);
		logger.log(DEBUG, "Split 2: " + split2);
	}
}