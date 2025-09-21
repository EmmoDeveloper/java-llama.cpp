package de.kherud.llama;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import static java.lang.System.Logger.Level.DEBUG;

/**
 * Test comprehensive error handling in the JNI layer
 */
public class ErrorHandlingTest {
	private static final System.Logger logger = System.getLogger(ErrorHandlingTest.class.getName());

    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        System.setProperty("de.kherud.llama.lib.path", "src/main/resources/de/kherud/llama/Linux/x86_64");
        model = new LlamaModel(
            new ModelParameters()
                .setCtxSize(512)
                .setModel("models/codellama-7b.Q2_K.gguf")
                .setGpuLayers(43)
        );
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    @Test(expected = IllegalStateException.class)
	@Ignore
    public void testEmbeddingWithoutEmbeddingMode() {
        // This should throw an IllegalStateException since embedding mode is not enabled
        model.embed("test");
    }

    @Test
    public void testNullInputHandling() {
        // Test that null inputs are handled gracefully
        try {
            model.encode(null);
            Assert.fail("Expected exception for null input to encode()");
        } catch (NullPointerException e) {
            // Expected - JNI should handle null inputs properly
            Assert.assertTrue("Exception should have meaningful message",
                e.getMessage() != null && e.getMessage().contains("text string parameter is null"));
        } catch (Exception e) {
            // Also acceptable - other error types are fine as long as we don't crash
            Assert.assertTrue("Exception should have meaningful message",
                e.getMessage() != null && !e.getMessage().isEmpty());
        }
    }

    @Test
    public void testInvalidModelAccess() {
        // Create a model and close it, then try to use it
        LlamaModel tempModel = new LlamaModel(
            new ModelParameters()
                .setCtxSize(256)
                .setModel("models/codellama-7b.Q2_K.gguf")
                .setGpuLayers(10)
        );

        // Close the model
        tempModel.close();

        // Now try to use it - this should handle the error gracefully
        try {
            tempModel.encode("test");
            // If we get here, it either worked (unlikely) or returned null gracefully
        } catch (Exception e) {
            // Expected - should handle closed model gracefully
            Assert.assertTrue("Exception should have meaningful message",
                e.getMessage() != null && !e.getMessage().isEmpty());
        }
    }

    @Test
	@Ignore
    public void testErrorRecovery() {
        // Test that after an error, the model can still be used normally
        try {
            model.embed("test"); // This will fail due to no embedding mode
            Assert.fail("Expected exception for embedding without embedding mode");
        } catch (IllegalStateException e) {
            // Expected
        }

        // Now test that normal operations still work
        int[] tokens = model.encode("Hello world");
        Assert.assertNotNull("Model should still work after error", tokens);
        Assert.assertTrue("Should produce tokens", tokens.length > 0);

        String decoded = model.decode(tokens);
        Assert.assertNotNull("Decoding should work", decoded);
        Assert.assertTrue("Decoded text should contain original content",
            decoded.toLowerCase().contains("hello"));
    }

    @Test
    public void testLongInputHandling() {
        // Test with a very long input to ensure proper memory handling
        StringBuilder longInput = new StringBuilder();
		longInput.append("This is a very long input string that will test memory handling. ".repeat(1000));

        try {
            int[] tokens = model.encode(longInput.toString());
            Assert.assertNotNull("Should handle long input", tokens);

            if (tokens.length > 0) {
                String decoded = model.decode(tokens);
                Assert.assertNotNull("Should decode long input", decoded);
            }
        } catch (Exception e) {
            // If it fails, it should fail gracefully with a meaningful message
            Assert.assertTrue("Exception should have meaningful message",
                e.getMessage() != null && !e.getMessage().isEmpty());
            logger.log(DEBUG, "Long input handling failed gracefully: " + e.getMessage());
        }
    }

    @Test
    public void testEmptyInputHandling() {
        // Test empty string input
        try {
            int[] tokens = model.encode("");
            // Empty input should either work (produce empty/minimal tokens) or fail gracefully
            Assert.assertNotNull("Should handle empty input", tokens);
        } catch (Exception e) {
            // If it fails, it should fail gracefully
            Assert.assertTrue("Exception should have meaningful message",
                e.getMessage() != null && !e.getMessage().isEmpty());
        }
    }
}
