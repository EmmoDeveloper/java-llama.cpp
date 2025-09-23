package de.kherud.llama.tools;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import static org.junit.Assert.fail;

/**
 * Test cases for DevelopmentUtils.
 */
public class DevelopmentUtilsTest {

	@Rule
	public TemporaryFolder tempDir = new TemporaryFolder();

	@Before
	public void setUp() {
		// Setup for tests
	}

	@Test
	public void testSystemInfo() {
		try {
			DevelopmentUtils.SystemInfo.printSystemInformation();
		} catch (Exception e) {
			fail("Should not throw exception: " + e.getMessage());
		}
	}

	@Test
	public void testCommandLineInterface() {
		String[] args = {"sysinfo"};

		try {
			DevelopmentUtils.main(args);
		} catch (Exception e) {
			fail("Should not throw exception: " + e.getMessage());
		}
	}
}
