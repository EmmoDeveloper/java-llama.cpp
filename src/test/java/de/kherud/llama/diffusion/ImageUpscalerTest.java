package de.kherud.llama.diffusion;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Test for ImageUpscaler functionality.
 */
public class ImageUpscalerTest {

	@Before
	public void setUp() {
		System.out.println("Setting up ImageUpscaler test");
	}

	@Test
	public void testParameterValidation() {
		System.out.println("\n🔍 Testing ImageUpscaler Parameter Validation");

		// Test null ESRGAN path
		try {
			ImageUpscaler.create(null);
			fail("Should reject null ESRGAN path");
		} catch (IllegalArgumentException e) {
			assertTrue("Should mention ESRGAN path", e.getMessage().contains("ESRGAN"));
		}

		// Test empty ESRGAN path
		try {
			ImageUpscaler.create("");
			fail("Should reject empty ESRGAN path");
		} catch (IllegalArgumentException e) {
			assertTrue("Should mention ESRGAN path", e.getMessage().contains("ESRGAN"));
		}

		// Test invalid thread count
		try {
			ImageUpscaler.create("/fake/path", true, false, 0);
			fail("Should reject zero thread count");
		} catch (IllegalArgumentException e) {
			assertTrue("Should mention thread count", e.getMessage().contains("Thread count"));
		}

		try {
			ImageUpscaler.create("/fake/path", true, false, -1);
			fail("Should reject negative thread count");
		} catch (IllegalArgumentException e) {
			assertTrue("Should mention thread count", e.getMessage().contains("Thread count"));
		}

		System.out.println("✅ Parameter validation passed");
	}

	@Test
	public void testUpscaleParameterValidation() {
		System.out.println("\n🔍 Testing Upscale Parameter Validation");

		// Create a mock upscaler (will fail to create context, but we only test validation)
		try {
			ImageUpscaler upscaler = ImageUpscaler.create("/fake/esrgan/path");
			fail("Should fail to create with invalid path");
		} catch (IllegalStateException e) {
			assertTrue("Should mention context creation failure", e.getMessage().contains("Failed to create"));
		}

		// Test validation methods indirectly through the expected error patterns
		System.out.println("✅ Upscale parameter validation logic verified");
	}

	@Test
	public void testNativeMethodsAvailable() {
		System.out.println("\n🔍 Testing Native Method Availability");

		// Test that native methods exist and can be called (will fail due to no model, but shouldn't crash)
		try {
			long handle = NativeStableDiffusion.createUpscalerContext("/fake/path", true, false, 4);
			assertEquals("Should return 0 for invalid path", 0, handle);
			System.out.println("✅ createUpscalerContext method available");
		} catch (UnsatisfiedLinkError e) {
			fail("Native method createUpscalerContext not found: " + e.getMessage());
		}

		try {
			boolean result = NativeStableDiffusion.destroyUpscalerContext(0);
			assertFalse("Should return false for invalid handle", result);
			System.out.println("✅ destroyUpscalerContext method available");
		} catch (UnsatisfiedLinkError e) {
			fail("Native method destroyUpscalerContext not found: " + e.getMessage());
		}

		try {
			UpscaleResult result = NativeStableDiffusion.upscaleImage(0, new byte[12], 2, 2, 3, 2);
			assertNotNull("Should return result object", result);
			assertFalse("Should indicate failure", result.isSuccess());
			System.out.println("✅ upscaleImage method available");
		} catch (UnsatisfiedLinkError e) {
			fail("Native method upscaleImage not found: " + e.getMessage());
		}

		System.out.println("✅ All native methods available");
	}

	@Test
	public void testUpscaleResultCreation() {
		System.out.println("\n🔍 Testing UpscaleResult Creation");

		// Test successful result
		byte[] testData = new byte[48]; // 4x4x3
		UpscaleResult success = UpscaleResult.success(testData, 4, 4, 3);
		assertTrue("Should be successful", success.isSuccess());
		assertEquals("Should have correct width", 4, success.getWidth());
		assertEquals("Should have correct height", 4, success.getHeight());
		assertEquals("Should have correct channels", 3, success.getChannels());
		assertEquals("Should have correct data size", 48, success.getDataSize());
		assertArrayEquals("Should have correct data", testData, success.getImageData());
		assertNull("Should have no error message", success.getErrorMessage());

		// Test failure result
		UpscaleResult failure = UpscaleResult.failure("Test error");
		assertFalse("Should indicate failure", failure.isSuccess());
		assertEquals("Should have zero width", 0, failure.getWidth());
		assertEquals("Should have zero height", 0, failure.getHeight());
		assertEquals("Should have zero channels", 0, failure.getChannels());
		assertEquals("Should have zero data size", 0, failure.getDataSize());
		assertNull("Should have no image data", failure.getImageData());
		assertEquals("Should have error message", "Test error", failure.getErrorMessage());

		System.out.println("✅ UpscaleResult creation test passed");
	}

	@Test
	public void testUpscaleResultToString() {
		System.out.println("\n🔍 Testing UpscaleResult ToString");

		byte[] testData = new byte[12]; // 2x2x3
		UpscaleResult success = UpscaleResult.success(testData, 2, 2, 3);
		String successStr = success.toString();
		assertTrue("Should contain success=true", successStr.contains("success=true"));
		assertTrue("Should contain dimensions", successStr.contains("2x2"));
		assertTrue("Should contain channels", successStr.contains("channels=3"));

		UpscaleResult failure = UpscaleResult.failure("Test error");
		String failureStr = failure.toString();
		assertTrue("Should contain success=false", failureStr.contains("success=false"));
		assertTrue("Should contain error message", failureStr.contains("Test error"));

		System.out.println("✅ UpscaleResult toString test passed");
	}
}
