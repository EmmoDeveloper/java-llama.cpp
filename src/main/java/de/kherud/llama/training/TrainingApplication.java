package de.kherud.llama.training;

import java.util.Objects;

/**
 * Represents an application of training data for LoRA fine-tuning.
 * Contains input text and expected target/completion text with formatting logic.
 *
 * @param instruction Optional system instruction
 * @param weight      Optional example weight for loss
 */
public record TrainingApplication(String input, String target, String instruction, float weight) {
	public TrainingApplication(String input, String target) {
		this(input, target, null, 1.0f);
	}

	public TrainingApplication(String input, String target, String instruction) {
		this(input, target, instruction, 1.0f);
	}

	/**
	 * Create instruction-following format like Alpaca/Vicuna
	 */
	public static TrainingApplication instructionFormat(String instruction, String input, String response) {
		String formattedInput = String.format(
			"Below is an instruction that describes a task, paired with an input that provides further context. " +
				"Write a response that appropriately completes the request.\n\n" +
				"### Instruction:\n%s\n\n### Input:\n%s\n\n### Response:\n",
			instruction, input
		);
		return new TrainingApplication(formattedInput, response, instruction);
	}

	/**
	 * Create chat format like ChatML
	 */
	public static TrainingApplication chatFormat(String systemPrompt, String userMessage, String assistantResponse) {
		String formattedInput = String.format(
			"<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
			systemPrompt != null ? systemPrompt : "You are a helpful assistant.",
			userMessage
		);
		String formattedTarget = assistantResponse + "<|im_end|>";
		return new TrainingApplication(formattedInput, formattedTarget, systemPrompt);
	}

	/**
	 * Create simple completion format
	 */
	public static TrainingApplication completionFormat(String prompt, String completion) {
		return new TrainingApplication(prompt, completion);
	}

	/**
	 * Get the full training text (input + target) for tokenization
	 */
	public String getFullText() {
		return input + target;
	}

	/**
	 * Get formatted text with special tokens for training
	 */
	public String getFormattedForTraining() {
		// Add special tokens to mark input/target boundaries
		return "<|startoftext|>" + input + "<|startoftarget|>" + target + "<|endoftext|>";
	}


	@Override
	public String toString() {
		return String.format("TrainingApplication{input='%s...', target='%s...', weight=%.2f}",
			input.length() > 50 ? input.substring(0, 50) : input,
			target.length() > 50 ? target.substring(0, 50) : target,
			weight);
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (!(o instanceof TrainingApplication)) return false;
		TrainingApplication that = (TrainingApplication) o;
		return Float.compare(that.weight, weight) == 0 &&
			Objects.equals(input, that.input) &&
			Objects.equals(target, that.target) &&
			Objects.equals(instruction, that.instruction);
	}

}
