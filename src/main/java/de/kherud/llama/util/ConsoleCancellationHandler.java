package de.kherud.llama.util;

import de.kherud.llama.LlamaIterator;
import java.io.IOException;

/**
 * Utility class for handling ESC key cancellation in console applications.
 * Monitors keyboard input in a separate thread and cancels generation when ESC is pressed.
 */
public class ConsoleCancellationHandler {
	
	private static final int ESC_KEY = 27;
	private volatile boolean cancelled = false;
	private volatile boolean stopMonitoring = false;
	private Thread monitorThread;
	
	/**
	 * Start listening for ESC key press and cancel the iterator if detected.
	 * Works on systems where System.in.available() is supported.
	 * 
	 * @param iterator The LlamaIterator to cancel when ESC is pressed
	 * @return this handler for chaining
	 */
	public ConsoleCancellationHandler listen(LlamaIterator iterator) {
		if (iterator == null) {
			throw new IllegalArgumentException("Iterator cannot be null");
		}
		
		cancelled = false;
		stopMonitoring = false;
		monitorThread = new Thread(() -> {
			try {
				// Set terminal to raw mode if possible (Unix/Linux)
				configureTerminal(true);
				
				while (!stopMonitoring && !cancelled && iterator.hasNext()) {
					try {
						if (System.in.available() > 0) {
							int key = System.in.read();
							if (key == ESC_KEY) {
								System.err.println("\n[ESC pressed - Cancelling generation...]");
								iterator.cancel();
								cancelled = true;
								break;
							}
						}
					} catch (IOException e) {
						// System.in may not be available in test environments
						// Continue without cancellation support
					}
					Thread.sleep(50); // Check every 50ms
				}
			} catch (InterruptedException e) {
				// Thread was interrupted - monitoring is best effort
			} finally {
				// Restore terminal settings
				configureTerminal(false);
			}
		});
		
		monitorThread.setDaemon(true);
		monitorThread.setName("ESC-Monitor");
		monitorThread.start();
		
		return this;
	}
	
	/**
	 * Stop monitoring for ESC key.
	 */
	public void stop() {
		// Don't set cancelled to true here - that should only happen when ESC is pressed
		// We use a separate flag to stop the monitoring thread
		stopMonitoring = true;
		if (monitorThread != null && monitorThread.isAlive()) {
			monitorThread.interrupt();
		}
	}
	
	/**
	 * Check if generation was cancelled by ESC key.
	 * 
	 * @return true if ESC was pressed
	 */
	public boolean wasCancelled() {
		return cancelled;
	}
	
	/**
	 * Configure terminal for raw input (Unix/Linux only).
	 * On Windows, this has no effect.
	 * 
	 * @param raw true to enable raw mode, false to restore normal mode
	 */
	private void configureTerminal(boolean raw) {
		// Disable terminal configuration in test environments
		if (System.getProperty("java.awt.headless") != null || 
		    System.console() == null) {
			return;
		}
		
		if (System.getProperty("os.name").toLowerCase().contains("win")) {
			return; // Windows console handles this differently
		}
		
		try {
			if (raw) {
				// Put terminal in raw mode to capture ESC immediately
				Runtime.getRuntime().exec(new String[]{"sh", "-c", "stty -echo -icanon min 0 < /dev/tty"}).waitFor();
			} else {
				// Restore terminal to normal mode
				Runtime.getRuntime().exec(new String[]{"sh", "-c", "stty echo icanon < /dev/tty"}).waitFor();
			}
		} catch (IOException | InterruptedException e) {
			// Terminal configuration is best effort
		}
	}
	
	/**
	 * Convenience method to process an iterator with ESC cancellation support.
	 * 
	 * @param iterator The iterator to process
	 * @param processor Function to process each output
	 */
	public static void processWithCancellation(LlamaIterator iterator, OutputProcessor processor) {
		ConsoleCancellationHandler handler = new ConsoleCancellationHandler();
		handler.listen(iterator);
		
		try {
			while (iterator.hasNext()) {
				processor.process(iterator.next());
			}
		} finally {
			handler.stop();
			if (handler.wasCancelled()) {
				System.err.println("\nGeneration cancelled by user.");
			}
		}
	}
	
	/**
	 * Functional interface for processing LlamaOutput.
	 */
	@FunctionalInterface
	public interface OutputProcessor {
		void process(de.kherud.llama.LlamaOutput output);
	}
}