package de.kherud.llama;

import de.kherud.llama.util.ConsoleCancellationHandler;

/**
 * Example demonstrating ESC key cancellation during text generation.
 * Press ESC key during generation to cancel.
 */
public class CancellationExample {
	
	public static void main(String[] args) {
		ModelParameters params = new ModelParameters()
			.setModel("models/codellama-7b.Q2_K.gguf")
			.setGpuLayers(99);
		
		try (LlamaModel model = new LlamaModel(params)) {
			System.out.println("Model loaded. Starting generation...");
			System.out.println("Press ESC key to cancel generation at any time.\n");
			
			String prompt = "Write a detailed explanation of quantum computing:";
			InferenceParameters inferParams = new InferenceParameters(prompt)
				.setNPredict(500) // Generate up to 500 tokens
				.setStream(true);
			
			System.out.println("Prompt: " + prompt);
			System.out.println("Response:");
			System.out.println("─".repeat(50));
			
			// Method 1: Using the convenience method
			LlamaIterator iterator = model.generate(inferParams).iterator();
			ConsoleCancellationHandler.processWithCancellation(iterator, output -> {
				System.out.print(output.text);
				System.out.flush();
			});
			
			System.out.println("\n" + "─".repeat(50));
			
			// Method 2: Manual control for more flexibility
			System.out.println("\nStarting second generation (manual control)...");
			prompt = "List the top 10 programming languages:";
			inferParams = new InferenceParameters(prompt)
				.setNPredict(300)
				.setStream(true);
			
			System.out.println("Prompt: " + prompt);
			System.out.println("Response:");
			System.out.println("─".repeat(50));
			
			iterator = model.generate(inferParams).iterator();
			ConsoleCancellationHandler handler = new ConsoleCancellationHandler();
			handler.listen(iterator);
			
			try {
				while (iterator.hasNext()) {
					LlamaOutput output = iterator.next();
					System.out.print(output.text);
					System.out.flush();
					
					// You can check if cancelled and do custom handling
					if (handler.wasCancelled()) {
						System.err.println("\n\nCustom cancellation handling...");
						break;
					}
				}
			} finally {
				handler.stop();
			}
			
			System.out.println("\n" + "─".repeat(50));
			System.out.println("Generation complete.");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}