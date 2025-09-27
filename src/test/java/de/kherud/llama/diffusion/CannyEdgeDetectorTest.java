package de.kherud.llama.diffusion;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Isolated test for Canny edge detection functionality.
 */
public class CannyEdgeDetectorTest {

	@Before
	public void setUp() {
		System.out.println("Setting up Canny edge detection test");
	}

	@Test
	public void testParameterValidation() {
		System.out.println("\n🔍 Testing Canny Parameter Validation Only");

		// Test parameter validation without calling native methods
		try {
			CannyEdgeDetector.validateParameters(-0.1f, 0.08f, 0.8f, 1.0f);
			fail("Should reject negative high threshold");
		} catch (IllegalArgumentException e) {
			assertTrue("Should mention high threshold", e.getMessage().contains("High threshold"));
		}

		try {
			CannyEdgeDetector.validateParameters(0.08f, 1.1f, 0.8f, 1.0f);
			fail("Should reject low threshold > 1.0");
		} catch (IllegalArgumentException e) {
			assertTrue("Should mention low threshold", e.getMessage().contains("Low threshold"));
		}

		try {
			CannyEdgeDetector.validateParameters(0.05f, 0.1f, 0.8f, 1.0f);
			fail("Should reject low > high threshold");
		} catch (IllegalArgumentException e) {
			assertTrue("Should mention threshold order", e.getMessage().contains("cannot be higher"));
		}

		// Valid parameters should not throw
		CannyEdgeDetector.validateParameters(0.08f, 0.05f, 0.8f, 1.0f);

		System.out.println("✅ Parameter validation passed");
	}

	@Test
	public void testConstantsAvailability() {
		System.out.println("\n🔍 Testing Canny Constants");

		// Test that constants are properly defined
		assertEquals("Default high threshold", 0.08f, CannyEdgeDetector.DEFAULT_HIGH_THRESHOLD, 0.001f);
		assertEquals("Default low threshold", 0.08f, CannyEdgeDetector.DEFAULT_LOW_THRESHOLD, 0.001f);
		assertEquals("Default weak value", 0.8f, CannyEdgeDetector.DEFAULT_WEAK, 0.001f);
		assertEquals("Default strong value", 1.0f, CannyEdgeDetector.DEFAULT_STRONG, 0.001f);
		assertFalse("Default inverse", CannyEdgeDetector.DEFAULT_INVERSE);

		System.out.println("✅ Constants test passed");
	}

	@Test
	public void testDisabledFunctionality() {
		System.out.println("\n🚫 Testing Disabled Canny Functionality");

		byte[] testImage = new byte[3 * 3 * 3]; // 3x3 RGB

		// Test that all methods throw UnsupportedOperationException
		try {
			CannyEdgeDetector.detectEdges(testImage, 3, 3, 3);
			fail("Should throw UnsupportedOperationException");
		} catch (UnsupportedOperationException e) {
			assertTrue("Should mention disabled", e.getMessage().contains("disabled"));
			assertTrue("Should mention library bug", e.getMessage().contains("bug"));
		}

		try {
			CannyEdgeDetector.detectEdges(testImage, 3, 3, 3, 0.1f, 0.05f);
			fail("Should throw UnsupportedOperationException");
		} catch (UnsupportedOperationException e) {
			assertTrue("Should mention disabled", e.getMessage().contains("disabled"));
		}

		try {
			CannyEdgeDetector.detectEdges(testImage, 3, 3, 3, 0.1f, 0.05f, 0.8f, 1.0f, false);
			fail("Should throw UnsupportedOperationException");
		} catch (UnsupportedOperationException e) {
			assertTrue("Should mention disabled", e.getMessage().contains("disabled"));
		}

		try {
			CannyEdgeDetector.detectEdgesCopy(testImage, 3, 3, 3);
			fail("Should throw UnsupportedOperationException");
		} catch (UnsupportedOperationException e) {
			assertTrue("Should mention disabled", e.getMessage().contains("disabled"));
		}

		System.out.println("✅ Disabled functionality test passed");
	}
}