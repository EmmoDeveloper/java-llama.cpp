package de.kherud.llama;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Multi-agent conversation orchestration system.
 * Manages complex multi-turn conversations between AI agents with coordination and workflow management.
 */
public class MultiAgentConversationSystem {

	private final Map<String, ConversationOrchestrator> activeConversations = new ConcurrentHashMap<>();
	private final Map<String, AgentProfile> agentProfiles = new ConcurrentHashMap<>();
	private final ExecutorService orchestratorExecutor = Executors.newCachedThreadPool();
	private final AgentCommunicationProtocol communicationProtocol;

	public MultiAgentConversationSystem(AgentCommunicationProtocol communicationProtocol) {
		this.communicationProtocol = communicationProtocol;
	}

	/**
	 * Agent profile with conversation-specific capabilities
	 */
	public static class AgentProfile {
		private final String agentId;
		private final String name;
		private final String role;
		private final String personality;
		private final Set<String> conversationCapabilities;
		private final Map<String, Object> systemPrompt;
		private final ConversationBehavior behavior;

		public AgentProfile(String agentId, String name, String role, String personality,
							Set<String> conversationCapabilities, Map<String, Object> systemPrompt,
							ConversationBehavior behavior) {
			this.agentId = agentId;
			this.name = name;
			this.role = role;
			this.personality = personality;
			this.conversationCapabilities = conversationCapabilities != null ? conversationCapabilities : new HashSet<>();
			this.systemPrompt = systemPrompt != null ? systemPrompt : new HashMap<>();
			this.behavior = behavior != null ? behavior : new ConversationBehavior();
		}

		public String getAgentId() { return agentId; }
		public String getName() { return name; }
		public String getRole() { return role; }
		public String getPersonality() { return personality; }
		public Set<String> getConversationCapabilities() { return conversationCapabilities; }
		public Map<String, Object> getSystemPrompt() { return systemPrompt; }
		public ConversationBehavior getBehavior() { return behavior; }
	}

	/**
	 * Agent behavior configuration for conversations
	 */
	public static class ConversationBehavior {
		private final boolean proactive;
		private final int maxTurnsBeforeSummary;
		private final float responseThreshold;
		private final Set<String> triggerKeywords;
		private final Map<String, Double> roleAffinities;

		public ConversationBehavior() {
			this(false, 10, 0.7f, new HashSet<>(), new HashMap<>());
		}

		public ConversationBehavior(boolean proactive, int maxTurnsBeforeSummary, float responseThreshold,
									Set<String> triggerKeywords, Map<String, Double> roleAffinities) {
			this.proactive = proactive;
			this.maxTurnsBeforeSummary = maxTurnsBeforeSummary;
			this.responseThreshold = responseThreshold;
			this.triggerKeywords = triggerKeywords != null ? triggerKeywords : new HashSet<>();
			this.roleAffinities = roleAffinities != null ? roleAffinities : new HashMap<>();
		}

		public boolean isProactive() { return proactive; }
		public int getMaxTurnsBeforeSummary() { return maxTurnsBeforeSummary; }
		public float getResponseThreshold() { return responseThreshold; }
		public Set<String> getTriggerKeywords() { return triggerKeywords; }
		public Map<String, Double> getRoleAffinities() { return roleAffinities; }
	}

	/**
	 * Conversation orchestrator manages a single multi-agent conversation
	 */
	public static class ConversationOrchestrator {
		private final String conversationId;
		private final Set<String> participantIds;
		private final Map<String, AgentProfile> participants;
		private final List<ConversationTurn> turns;
		private final ConversationFlow flow;
		private final Instant startTime;
		private final AtomicInteger turnCounter = new AtomicInteger(0);
		private volatile ConversationState state;
		private String currentSpeaker;
		private Map<String, Object> conversationContext;
		private final ExecutorService turnExecutor = Executors.newSingleThreadExecutor();

		public enum ConversationState {
			INITIALIZING,
			ACTIVE,
			WAITING_FOR_RESPONSE,
			SUMMARIZING,
			COMPLETED,
			FAILED,
			PAUSED
		}

		public ConversationOrchestrator(String conversationId, Set<String> participantIds,
										Map<String, AgentProfile> participants, ConversationFlow flow) {
			this.conversationId = conversationId;
			this.participantIds = new HashSet<>(participantIds);
			this.participants = new HashMap<>(participants);
			this.flow = flow;
			this.turns = new CopyOnWriteArrayList<>();
			this.startTime = Instant.now();
			this.state = ConversationState.INITIALIZING;
			this.conversationContext = new ConcurrentHashMap<>();
		}

		public CompletableFuture<ConversationTurn> processNextTurn() {
			return CompletableFuture.supplyAsync(() -> {
				try {
					if (state != ConversationState.ACTIVE) {
						throw new IllegalStateException("Conversation not in active state: " + state);
					}

					// Determine next speaker
					String nextSpeaker = flow.determineNextSpeaker(turns, participants, conversationContext);
					if (nextSpeaker == null) {
						state = ConversationState.COMPLETED;
						return null;
					}

					currentSpeaker = nextSpeaker;
					state = ConversationState.WAITING_FOR_RESPONSE;

					// Create turn context
					Map<String, Object> turnContext = buildTurnContext(nextSpeaker);

					ConversationTurn turn = new ConversationTurn(
						turnCounter.incrementAndGet(),
						conversationId,
						nextSpeaker,
						participants.get(nextSpeaker),
						turnContext,
						Instant.now()
					);

					turns.add(turn);
					return turn;

				} catch (Exception e) {
					state = ConversationState.FAILED;
					throw new RuntimeException("Failed to process conversation turn", e);
				}
			}, turnExecutor);
		}

		private Map<String, Object> buildTurnContext(String speakerId) {
			Map<String, Object> context = new HashMap<>(conversationContext);

			// Add conversation history
			context.put("conversationHistory", turns.stream()
				.map(turn -> Map.of(
					"speaker", turn.getSpeakerId(),
					"content", turn.getContent(),
					"timestamp", turn.getTimestamp()
				))
				.collect(Collectors.toList()));

			// Add participant information
			context.put("participants", participants.values().stream()
				.map(p -> Map.of(
					"id", p.getAgentId(),
					"name", p.getName(),
					"role", p.getRole()
				))
				.collect(Collectors.toList()));

			// Add speaker-specific context
			AgentProfile speaker = participants.get(speakerId);
			if (speaker != null) {
				context.put("speakerRole", speaker.getRole());
				context.put("speakerPersonality", speaker.getPersonality());
				context.putAll(speaker.getSystemPrompt());
			}

			return context;
		}

		public void completeTurn(int turnNumber, String content, Map<String, Object> metadata) {
			ConversationTurn turn = turns.stream()
				.filter(t -> t.getTurnNumber() == turnNumber)
				.findFirst()
				.orElseThrow(() -> new IllegalArgumentException("Turn not found: " + turnNumber));

			turn.complete(content, metadata);
			state = ConversationState.ACTIVE;

			// Update conversation context based on turn
			if (metadata != null) {
				conversationContext.putAll(metadata);
			}

			// Check if conversation should continue
			if (flow.shouldContinueConversation(turns, participants, conversationContext)) {
				// Continue to next turn
			} else {
				state = ConversationState.COMPLETED;
			}
		}

		// Getters
		public String getConversationId() { return conversationId; }
		public Set<String> getParticipantIds() { return participantIds; }
		public Map<String, AgentProfile> getParticipants() { return participants; }
		public List<ConversationTurn> getTurns() { return turns; }
		public ConversationFlow getFlow() { return flow; }
		public Instant getStartTime() { return startTime; }
		public ConversationState getState() { return state; }
		public String getCurrentSpeaker() { return currentSpeaker; }
		public Map<String, Object> getConversationContext() { return conversationContext; }

		public void setState(ConversationState state) { this.state = state; }
		public void setConversationContext(Map<String, Object> context) {
			this.conversationContext = new ConcurrentHashMap<>(context);
		}

		public void shutdown() {
			turnExecutor.shutdown();
		}
	}

	/**
	 * Represents a single turn in the conversation
	 */
	public static class ConversationTurn {
		private final int turnNumber;
		private final String conversationId;
		private final String speakerId;
		private final AgentProfile speakerProfile;
		private final Map<String, Object> turnContext;
		private final Instant timestamp;
		private String content;
		private Map<String, Object> metadata;
		private boolean completed = false;

		public ConversationTurn(int turnNumber, String conversationId, String speakerId,
								AgentProfile speakerProfile, Map<String, Object> turnContext,
								Instant timestamp) {
			this.turnNumber = turnNumber;
			this.conversationId = conversationId;
			this.speakerId = speakerId;
			this.speakerProfile = speakerProfile;
			this.turnContext = turnContext;
			this.timestamp = timestamp;
		}

		public void complete(String content, Map<String, Object> metadata) {
			this.content = content;
			this.metadata = metadata;
			this.completed = true;
		}

		// Getters
		public int getTurnNumber() { return turnNumber; }
		public String getConversationId() { return conversationId; }
		public String getSpeakerId() { return speakerId; }
		public AgentProfile getSpeakerProfile() { return speakerProfile; }
		public Map<String, Object> getTurnContext() { return turnContext; }
		public Instant getTimestamp() { return timestamp; }
		public String getContent() { return content; }
		public Map<String, Object> getMetadata() { return metadata; }
		public boolean isCompleted() { return completed; }
	}

	/**
	 * Conversation flow strategy interface
	 */
	public interface ConversationFlow {
		String determineNextSpeaker(List<ConversationTurn> turns, Map<String, AgentProfile> participants,
									Map<String, Object> context);
		boolean shouldContinueConversation(List<ConversationTurn> turns, Map<String, AgentProfile> participants,
										  Map<String, Object> context);
		Map<String, Object> generateSummary(List<ConversationTurn> turns, Map<String, AgentProfile> participants);
	}

	/**
	 * Round-robin conversation flow
	 */
	public static class RoundRobinFlow implements ConversationFlow {
		private final List<String> participantOrder;
		private final int maxTurns;

		public RoundRobinFlow(List<String> participantOrder, int maxTurns) {
			this.participantOrder = new ArrayList<>(participantOrder);
			this.maxTurns = maxTurns;
		}

		@Override
		public String determineNextSpeaker(List<ConversationTurn> turns, Map<String, AgentProfile> participants,
										   Map<String, Object> context) {
			if (turns.size() >= maxTurns) {
				return null; // End conversation
			}

			int nextIndex = turns.size() % participantOrder.size();
			return participantOrder.get(nextIndex);
		}

		@Override
		public boolean shouldContinueConversation(List<ConversationTurn> turns, Map<String, AgentProfile> participants,
												 Map<String, Object> context) {
			return turns.size() < maxTurns;
		}

		@Override
		public Map<String, Object> generateSummary(List<ConversationTurn> turns, Map<String, AgentProfile> participants) {
			Map<String, Object> summary = new HashMap<>();
			summary.put("totalTurns", turns.size());
			summary.put("participants", participants.keySet());
			summary.put("startTime", turns.isEmpty() ? null : turns.get(0).getTimestamp());
			summary.put("endTime", turns.isEmpty() ? null : turns.get(turns.size() - 1).getTimestamp());
			return summary;
		}
	}

	/**
	 * Dynamic conversation flow based on content analysis
	 */
	public static class DynamicFlow implements ConversationFlow {
		private final int maxTurns;
		private final Predicate<String> completionPredicate;
		private final BiFunction<List<ConversationTurn>, Map<String, AgentProfile>, String> speakerSelector;

		public DynamicFlow(int maxTurns, Predicate<String> completionPredicate,
						   BiFunction<List<ConversationTurn>, Map<String, AgentProfile>, String> speakerSelector) {
			this.maxTurns = maxTurns;
			this.completionPredicate = completionPredicate;
			this.speakerSelector = speakerSelector;
		}

		@Override
		public String determineNextSpeaker(List<ConversationTurn> turns, Map<String, AgentProfile> participants,
										   Map<String, Object> context) {
			if (turns.size() >= maxTurns) {
				return null;
			}

			// Check for completion based on last turn content
			if (!turns.isEmpty()) {
				String lastContent = turns.get(turns.size() - 1).getContent();
				if (lastContent != null && completionPredicate.test(lastContent)) {
					return null;
				}
			}

			return speakerSelector.apply(turns, participants);
		}

		@Override
		public boolean shouldContinueConversation(List<ConversationTurn> turns, Map<String, AgentProfile> participants,
												 Map<String, Object> context) {
			return turns.size() < maxTurns &&
				   (turns.isEmpty() || !completionPredicate.test(turns.get(turns.size() - 1).getContent()));
		}

		@Override
		public Map<String, Object> generateSummary(List<ConversationTurn> turns, Map<String, AgentProfile> participants) {
			Map<String, Object> summary = new HashMap<>();
			summary.put("totalTurns", turns.size());
			summary.put("participants", participants.keySet());
			summary.put("completedNaturally", turns.size() < maxTurns);

			// Analyze speaker participation
			Map<String, Long> speakerCounts = turns.stream()
				.collect(Collectors.groupingBy(ConversationTurn::getSpeakerId, Collectors.counting()));
			summary.put("speakerParticipation", speakerCounts);

			return summary;
		}
	}

	/**
	 * Register an agent profile for conversations
	 */
	public void registerAgentProfile(AgentProfile profile) {
		agentProfiles.put(profile.getAgentId(), profile);
	}

	/**
	 * Start a new multi-agent conversation
	 */
	public CompletableFuture<ConversationOrchestrator> startConversation(Set<String> participantIds,
																		ConversationFlow flow,
																		Map<String, Object> initialContext) {
		return CompletableFuture.supplyAsync(() -> {
			String conversationId = UUID.randomUUID().toString();

			// Validate participants
			Map<String, AgentProfile> participants = new HashMap<>();
			for (String participantId : participantIds) {
				AgentProfile profile = agentProfiles.get(participantId);
				if (profile == null) {
					throw new IllegalArgumentException("Agent profile not found: " + participantId);
				}
				participants.put(participantId, profile);
			}

			ConversationOrchestrator orchestrator = new ConversationOrchestrator(
				conversationId, participantIds, participants, flow
			);

			if (initialContext != null) {
				orchestrator.setConversationContext(initialContext);
			}

			orchestrator.setState(ConversationOrchestrator.ConversationState.ACTIVE);
			activeConversations.put(conversationId, orchestrator);

			return orchestrator;
		}, orchestratorExecutor);
	}

	/**
	 * Process a conversation turn
	 */
	public CompletableFuture<ConversationTurn> processConversationTurn(String conversationId) {
		ConversationOrchestrator orchestrator = activeConversations.get(conversationId);
		if (orchestrator == null) {
			return CompletableFuture.failedFuture(new IllegalArgumentException("Conversation not found: " + conversationId));
		}

		return orchestrator.processNextTurn();
	}

	/**
	 * Complete a conversation turn with agent response
	 */
	public void completeTurn(String conversationId, int turnNumber, String content, Map<String, Object> metadata) {
		ConversationOrchestrator orchestrator = activeConversations.get(conversationId);
		if (orchestrator == null) {
			throw new IllegalArgumentException("Conversation not found: " + conversationId);
		}

		orchestrator.completeTurn(turnNumber, content, metadata);

		// Remove if completed
		if (orchestrator.getState() == ConversationOrchestrator.ConversationState.COMPLETED ||
			orchestrator.getState() == ConversationOrchestrator.ConversationState.FAILED) {
			activeConversations.remove(conversationId);
			orchestrator.shutdown();
		}
	}

	/**
	 * Get conversation orchestrator
	 */
	public ConversationOrchestrator getConversation(String conversationId) {
		return activeConversations.get(conversationId);
	}

	/**
	 * Get all active conversation IDs
	 */
	public Set<String> getActiveConversationIds() {
		return new HashSet<>(activeConversations.keySet());
	}

	/**
	 * End a conversation
	 */
	public void endConversation(String conversationId) {
		ConversationOrchestrator orchestrator = activeConversations.remove(conversationId);
		if (orchestrator != null) {
			orchestrator.setState(ConversationOrchestrator.ConversationState.COMPLETED);
			orchestrator.shutdown();
		}
	}

	/**
	 * Get agent profile
	 */
	public AgentProfile getAgentProfile(String agentId) {
		return agentProfiles.get(agentId);
	}

	/**
	 * Get all registered agent profiles
	 */
	public Collection<AgentProfile> getAgentProfiles() {
		return new ArrayList<>(agentProfiles.values());
	}

	/**
	 * Shutdown the conversation system
	 */
	public void shutdown() {
		// End all active conversations
		for (ConversationOrchestrator orchestrator : activeConversations.values()) {
			orchestrator.setState(ConversationOrchestrator.ConversationState.COMPLETED);
			orchestrator.shutdown();
		}
		activeConversations.clear();

		orchestratorExecutor.shutdown();
		try {
			if (!orchestratorExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
				orchestratorExecutor.shutdownNow();
			}
		} catch (InterruptedException e) {
			orchestratorExecutor.shutdownNow();
			Thread.currentThread().interrupt();
		}
	}
}
