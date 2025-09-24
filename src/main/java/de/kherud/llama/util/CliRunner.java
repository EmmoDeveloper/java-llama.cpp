package de.kherud.llama.util;

/**
 * Utility class for running CLI applications with proper exit handling.
 * This allows main methods to throw exceptions instead of calling System.exit,
 * making them testable and usable as libraries.
 */
public class CliRunner {

	/**
	 * Interface for CLI applications that can throw exceptions.
	 */
	@FunctionalInterface
	public interface CliApplication {
		void run(String[] args) throws Exception;
	}

	/**
	 * Run a CLI application with proper exception handling and exit codes.
	 *
	 * @param app The application to run
	 * @param args Command line arguments
	 */
	public static void runWithExit(CliApplication app, String[] args) {
		try {
			app.run(args);
		} catch (IllegalArgumentException e) {
			// User error - invalid arguments
			System.err.println("Error: " + e.getMessage());
			System.exit(1);
		} catch (Exception e) {
			// Unexpected error
			System.err.println("Fatal error: " + e.getMessage());
			e.printStackTrace();
			System.exit(2);
		}
	}

	/**
	 * Run a CLI application without System.exit (for testing or embedding).
	 *
	 * @param app The application to run
	 * @param args Command line arguments
	 * @return 0 for success, non-zero for failure
	 */
	public static int runWithoutExit(CliApplication app, String[] args) {
		try {
			app.run(args);
			return 0;
		} catch (IllegalArgumentException e) {
			System.err.println("Error: " + e.getMessage());
			return 1;
		} catch (Exception e) {
			System.err.println("Fatal error: " + e.getMessage());
			return 2;
		}
	}
}