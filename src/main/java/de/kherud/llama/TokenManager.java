package de.kherud.llama;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utilities for token counting, budget management, and context window handling.
 */
public class TokenManager {

	private final LlamaModel model;
	private final long contextSize;
	private final TokenBudget defaultBudget;

	public TokenManager(LlamaModel model) {
		this.model = model;
		this.contextSize = (int) model.getContextSize();
		this.defaultBudget = new TokenBudget(contextSize);
	}

	public TokenManager(LlamaModel model, long maxContextSize) {
		this.model = model;
		this.contextSize = Math.min(maxContextSize, model.getContextSize());
		this.defaultBudget = new TokenBudget((int) Math.min(contextSize, Integer.MAX_VALUE));
	}

	/**
	 * Count tokens in text
	 */
	public int countTokens(String text) {
		if (text == null || text.isEmpty()) {
			return 0;
		}
		return model.encode(text).length;
	}

	/**
	 * Count tokens in multiple texts
	 */
	public int countTokens(List<String> texts) {
		int total = 0;
		for (String text : texts) {
			total += countTokens(text);
		}
		return total;
	}

	/**
	 * Count tokens for conversation messages
	 */
	public int countTokens(List<ConversationManager.Message> messages, boolean applyTemplate) {
		if (applyTemplate) {
			// For now, just use simple concatenation as we don't have direct template support
			// This could be enhanced when template support is added
			StringBuilder sb = new StringBuilder();
			for (ConversationManager.Message msg : messages) {
				sb.append(msg.role()).append(": ").append(msg.content()).append("\n");
			}
			return countTokens(sb.toString());
		} else {
			int total = 0;
			for (ConversationManager.Message msg : messages) {
				// Add role tokens (approximate)
				total += countTokens(msg.role() + ": ");
				total += countTokens(msg.content());
				total += 1; // newline
			}
			return total;
		}
	}

	/**
	 * Truncate text to fit token budget
	 */
	public String truncateToFit(String text, int maxTokens) {
		return truncateToFit(text, maxTokens, TruncationStrategy.END);
	}

	/**
	 * Truncate text with specific strategy
	 */
	public String truncateToFit(String text, int maxTokens, TruncationStrategy strategy) {
		int[] tokens = model.encode(text);

		if (tokens.length <= maxTokens) {
			return text;
		}

		int[] truncated;
		switch (strategy) {
			case START:
				truncated = Arrays.copyOfRange(tokens, tokens.length - maxTokens, tokens.length);
				break;
			case MIDDLE:
				int keepStart = maxTokens / 2;
				int keepEnd = maxTokens - keepStart;
				truncated = new int[maxTokens];
				System.arraycopy(tokens, 0, truncated, 0, keepStart);
				System.arraycopy(tokens, tokens.length - keepEnd, truncated, keepStart, keepEnd);
				break;
			/* END and */ default:
				truncated = Arrays.copyOf(tokens, maxTokens);
		}

		return model.decode(truncated);
	}

	/**
	 * Split text into chunks of maximum token size
	 */
	public List<String> splitIntoChunks(String text, int maxTokensPerChunk) {
		return splitIntoChunks(text, maxTokensPerChunk, 0);
	}

	/**
	 * Split text into chunks with overlap
	 */
	public List<String> splitIntoChunks(String text, int maxTokensPerChunk, int overlapTokens) {
		List<String> chunks = new ArrayList<>();
		int[] tokens = model.encode(text);

		if (tokens.length <= maxTokensPerChunk) {
			chunks.add(text);
			return chunks;
		}

		int step = Math.max(1, maxTokensPerChunk - overlapTokens);
		for (int i = 0; i < tokens.length; i += step) {
			int end = Math.min(i + maxTokensPerChunk, tokens.length);
			int[] chunkTokens = Arrays.copyOfRange(tokens, i, end);
			chunks.add(model.decode(chunkTokens));

			if (end >= tokens.length) {
				break;
			}
		}

		return chunks;
	}

	/**
	 * Create a new token budget
	 */
	public TokenBudget createBudget(int maxTokens) {
		return new TokenBudget((int) Math.min(maxTokens, contextSize));
	}

	/**
	 * Get the default budget
	 */
	public TokenBudget getDefaultBudget() {
		return defaultBudget;
	}

	/**
	 * Get model's context size
	 */
	public long getContextSize() {
		return contextSize;
	}

	/**
	 * Estimate if text fits within budget
	 */
	public boolean fitsInBudget(String text, long budgetTokens) {
		// Quick character-based estimation first
		if (text.length() / 4 > budgetTokens) {
			return false;
		}
		return countTokens(text) <= budgetTokens;
	}

	/**
	 * Token budget tracker
	 */
	public class TokenBudget {
		private final long maxTokens;
		private long usedTokens;
		private final List<String> components;

		public TokenBudget(long maxTokens) {
			this.maxTokens = maxTokens;
			this.usedTokens = 0;
			this.components = new ArrayList<>();
		}

		/**
		 * Reserve tokens for a component
		 */
		public boolean reserve(String component, long tokens) {
			if (usedTokens + tokens > maxTokens) {
				return false;
			}
			usedTokens += tokens;
			components.add(component + " (" + tokens + " tokens)");
			return true;
		}

		/**
		 * Reserve tokens for text
		 */
		public boolean reserveForText(String componentName, String text) {
			long tokens = countTokens(text);
			return reserve(componentName, tokens);
		}

		/**
		 * Get remaining tokens
		 */
		public long getRemaining() {
			return maxTokens - usedTokens;
		}

		/**
		 * Get used tokens
		 */
		public long getUsed() {
			return usedTokens;
		}

		/**
		 * Get maximum tokens
		 */
		public long getMax() {
			return maxTokens;
		}

		/**
		 * Check if budget has room for more tokens
		 */
		public boolean hasRoom(long tokens) {
			return usedTokens + tokens <= maxTokens;
		}

		/**
		 * Reset the budget
		 */
		public void reset() {
			usedTokens = 0;
			components.clear();
		}

		/**
		 * Get usage percentage
		 */
		public double getUsagePercent() {
			return (double) usedTokens / maxTokens * 100;
		}

		/**
		 * Get budget breakdown
		 */
		public String getBreakdown() {
			StringBuilder sb = new StringBuilder();
			sb.append("Token Budget: ").append(usedTokens).append("/").append(maxTokens);
			sb.append(" (").append(String.format("%.1f%%", getUsagePercent())).append(")\n");
			for (String component : components) {
				sb.append("  - ").append(component).append("\n");
			}
			return sb.toString();
		}
	}

	/**
	 * Truncation strategies
	 */
	public enum TruncationStrategy {
		START,   // Keep end of text
		END,     // Keep start of text
		MIDDLE   // Keep start and end, remove middle
	}

	/**
	 * Utility to find optimal token allocation for multiple components
	 */
	public static class TokenAllocator {
		private final long totalBudget;
		private final List<AllocationRequest> requests;

		public TokenAllocator(long totalBudget) {
			this.totalBudget = totalBudget;
			this.requests = new ArrayList<>();
		}

		public TokenAllocator addRequest(String name, long minTokens, long idealTokens, double priority) {
			requests.add(new AllocationRequest(name, minTokens, idealTokens, priority));
			return this;
		}

		public AllocationResult allocate() {
			AllocationResult result = new AllocationResult();

			// First pass: allocate minimum tokens
			long totalMin = 0;
			for (AllocationRequest req : requests) {
				totalMin += req.minTokens;
			}

			if (totalMin > totalBudget) {
				result.success = false;
				result.reason = "Minimum token requirements exceed budget";
				return result;
			}

			// Allocate minimums
			for (AllocationRequest req : requests) {
				result.allocations.put(req.name, req.minTokens);
			}

			// Second pass: distribute remaining tokens by priority
			long remaining = totalBudget - totalMin;
			double totalPriority = requests.stream().mapToDouble(r -> r.priority).sum();

			for (AllocationRequest req : requests) {
				long current = result.allocations.get(req.name);
				long ideal = req.idealTokens;
				long additional = Math.min(
					ideal - current,
					(long) (remaining * (req.priority / totalPriority))
				);
				result.allocations.put(req.name, current + additional);
			}

			result.success = true;
			result.totalAllocated = result.allocations.values().stream().mapToLong(Long::longValue).sum();
			return result;
		}

		private record AllocationRequest(String name, long minTokens, long idealTokens, double priority) {
		}

		public static class AllocationResult {
			public boolean success;
			public String reason;
			public final Map<String, Long> allocations = new HashMap<>();
			public long totalAllocated;
		}
	}
}
