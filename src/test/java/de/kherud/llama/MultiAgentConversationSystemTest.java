package de.kherud.llama;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static java.lang.System.Logger.Level.DEBUG;

public class MultiAgentConversationSystemTest {
	private static final System.Logger logger = System.getLogger(MultiAgentConversationSystemTest.class.getName());

	private MultiAgentConversationSystem conversationSystem;
	private AgentCommunicationProtocol protocol;

	@Before
	public void setUp() {
		protocol = new AgentCommunicationProtocol();
		conversationSystem = new MultiAgentConversationSystem(protocol);
	}

	@After
	public void tearDown() {
		conversationSystem.shutdown();
		protocol.shutdown();
	}

	@Test
	public void testAgentProfileRegistration() {
		logger.log(DEBUG, "\n=== Agent Profile Registration Test ===");

		MultiAgentConversationSystem.AgentProfile profile = new MultiAgentConversationSystem.AgentProfile(
			"analyst_agent",
			"Data Analyst",
			"analyst",
			"analytical, detail-oriented, methodical",
			Set.of("data_analysis", "visualization", "reporting"),
			Map.of("systemPrompt", "You are a data analyst AI assistant"),
			new MultiAgentConversationSystem.ConversationBehavior()
		);

		conversationSystem.registerAgentProfile(profile);

		MultiAgentConversationSystem.AgentProfile retrieved = conversationSystem.getAgentProfile("analyst_agent");
		Assert.assertNotNull("Profile should be registered", retrieved);
		Assert.assertEquals("Name should match", "Data Analyst", retrieved.getName());
		Assert.assertEquals("Role should match", "analyst", retrieved.getRole());
		Assert.assertTrue("Should have data_analysis capability",
			retrieved.getConversationCapabilities().contains("data_analysis"));

		Collection<MultiAgentConversationSystem.AgentProfile> profiles = conversationSystem.getAgentProfiles();
		Assert.assertEquals("Should have 1 profile", 1, profiles.size());

		logger.log(DEBUG, "Registered profile: " + retrieved.getName() + " (" + retrieved.getRole() + ")");
		logger.log(DEBUG, "Capabilities: " + retrieved.getConversationCapabilities());
		logger.log(DEBUG, "✅ Agent profile registration test passed!");
	}

	@Test
	public void testRoundRobinConversation() throws Exception {
		logger.log(DEBUG, "\n=== Round Robin Conversation Test ===");

		// Register agent profiles
		MultiAgentConversationSystem.AgentProfile alice = new MultiAgentConversationSystem.AgentProfile(
			"alice", "Alice", "facilitator", "friendly, organized",
			Set.of("facilitation", "coordination"), Map.of(), new MultiAgentConversationSystem.ConversationBehavior()
		);
		MultiAgentConversationSystem.AgentProfile bob = new MultiAgentConversationSystem.AgentProfile(
			"bob", "Bob", "analyst", "logical, thorough",
			Set.of("analysis", "critique"), Map.of(), new MultiAgentConversationSystem.ConversationBehavior()
		);
		MultiAgentConversationSystem.AgentProfile charlie = new MultiAgentConversationSystem.AgentProfile(
			"charlie", "Charlie", "creative", "innovative, imaginative",
			Set.of("ideation", "brainstorming"), Map.of(), new MultiAgentConversationSystem.ConversationBehavior()
		);

		conversationSystem.registerAgentProfile(alice);
		conversationSystem.registerAgentProfile(bob);
		conversationSystem.registerAgentProfile(charlie);

		// Start round-robin conversation
		List<String> participantOrder = List.of("alice", "bob", "charlie");
		MultiAgentConversationSystem.RoundRobinFlow flow = new MultiAgentConversationSystem.RoundRobinFlow(participantOrder, 6);

		MultiAgentConversationSystem.ConversationOrchestrator orchestrator =
			conversationSystem.startConversation(Set.of("alice", "bob", "charlie"), flow,
				Map.of("topic", "project planning")).get(2, TimeUnit.SECONDS);

		Assert.assertNotNull("Orchestrator should be created", orchestrator);
		Assert.assertEquals("Should have 3 participants", 3, orchestrator.getParticipantIds().size());
		Assert.assertEquals("State should be ACTIVE",
			MultiAgentConversationSystem.ConversationOrchestrator.ConversationState.ACTIVE, orchestrator.getState());

		// Process several turns
		for (int i = 0; i < 3; i++) {
			MultiAgentConversationSystem.ConversationTurn turn =
				conversationSystem.processConversationTurn(orchestrator.getConversationId()).get(2, TimeUnit.SECONDS);
			Assert.assertNotNull("Turn should be created", turn);

			String expectedSpeaker = participantOrder.get(i);
			Assert.assertEquals("Speaker should follow round-robin order", expectedSpeaker, turn.getSpeakerId());

			// Complete the turn
			String content = "Response from " + turn.getSpeakerProfile().getName();
			conversationSystem.completeTurn(orchestrator.getConversationId(), turn.getTurnNumber(), content, null);
		}

		Assert.assertEquals("Should have 3 turns", 3, orchestrator.getTurns().size());

		logger.log(DEBUG, "Conversation ID: " + orchestrator.getConversationId());
		logger.log(DEBUG, "Participants: " + orchestrator.getParticipantIds());
		logger.log(DEBUG, "Turns completed: " + orchestrator.getTurns().size());
		logger.log(DEBUG, "✅ Round robin conversation test passed!");
	}

	@Test
	public void testDynamicConversationFlow() throws Exception {
		logger.log(DEBUG, "\n=== Dynamic Conversation Flow Test ===");

		// Register agents
		MultiAgentConversationSystem.AgentProfile researcher = new MultiAgentConversationSystem.AgentProfile(
			"researcher", "Dr. Smith", "researcher", "scientific, methodical",
			Set.of("research", "analysis"), Map.of(), new MultiAgentConversationSystem.ConversationBehavior()
		);
		MultiAgentConversationSystem.AgentProfile reviewer = new MultiAgentConversationSystem.AgentProfile(
			"reviewer", "Prof. Jones", "reviewer", "critical, thorough",
			Set.of("review", "validation"), Map.of(), new MultiAgentConversationSystem.ConversationBehavior()
		);

		conversationSystem.registerAgentProfile(researcher);
		conversationSystem.registerAgentProfile(reviewer);

		// Create dynamic flow with completion predicate
		MultiAgentConversationSystem.DynamicFlow flow = new MultiAgentConversationSystem.DynamicFlow(
			10,
			content -> content != null && content.toLowerCase().contains("conclusion"),
			(turns, participants) -> {
				if (turns.isEmpty() || !turns.get(turns.size() - 1).getSpeakerId().equals("researcher")) {
					return "researcher";
				} else {
					return "reviewer";
				}
			}
		);

		MultiAgentConversationSystem.ConversationOrchestrator orchestrator =
			conversationSystem.startConversation(Set.of("researcher", "reviewer"), flow,
				Map.of("research_topic", "AI safety")).get(2, TimeUnit.SECONDS);

		// Process turns until completion
		MultiAgentConversationSystem.ConversationTurn turn1 =
			conversationSystem.processConversationTurn(orchestrator.getConversationId()).get(2, TimeUnit.SECONDS);
		Assert.assertEquals("First speaker should be researcher", "researcher", turn1.getSpeakerId());
		conversationSystem.completeTurn(orchestrator.getConversationId(), turn1.getTurnNumber(),
			"Initial research findings on AI safety", null);

		MultiAgentConversationSystem.ConversationTurn turn2 =
			conversationSystem.processConversationTurn(orchestrator.getConversationId()).get(2, TimeUnit.SECONDS);
		Assert.assertEquals("Second speaker should be reviewer", "reviewer", turn2.getSpeakerId());
		conversationSystem.completeTurn(orchestrator.getConversationId(), turn2.getTurnNumber(),
			"Review complete. In conclusion, the research is solid.", null);

		// Next turn should not be created due to completion predicate
		try {
			MultiAgentConversationSystem.ConversationTurn turn3 =
				conversationSystem.processConversationTurn(orchestrator.getConversationId()).get(2, TimeUnit.SECONDS);
			Assert.assertNull("No more turns should be created after conclusion", turn3);
		} catch (Exception e) {
			// Expected when conversation is completed
		}

		Assert.assertEquals("Should have 2 turns", 2, orchestrator.getTurns().size());

		logger.log(DEBUG, "Dynamic flow completed with " + orchestrator.getTurns().size() + " turns");
		logger.log(DEBUG, "Final state: " + orchestrator.getState());
		logger.log(DEBUG, "✅ Dynamic conversation flow test passed!");
	}

	@Test
	public void testConversationContext() throws Exception {
		logger.log(DEBUG, "\n=== Conversation Context Test ===");

		MultiAgentConversationSystem.AgentProfile agent1 = new MultiAgentConversationSystem.AgentProfile(
			"context_agent1", "Agent One", "participant", "helpful",
			Set.of("participation"), Map.of("specialty", "mathematics"),
			new MultiAgentConversationSystem.ConversationBehavior()
		);

		conversationSystem.registerAgentProfile(agent1);

		MultiAgentConversationSystem.RoundRobinFlow flow = new MultiAgentConversationSystem.RoundRobinFlow(List.of("context_agent1"), 2);

		Map<String, Object> initialContext = Map.of(
			"problem", "solve quadratic equation",
			"difficulty", "intermediate"
		);

		MultiAgentConversationSystem.ConversationOrchestrator orchestrator =
			conversationSystem.startConversation(Set.of("context_agent1"), flow, initialContext).get(2, TimeUnit.SECONDS);

		// Check initial context is preserved
		Assert.assertEquals("Initial context should be preserved", "solve quadratic equation",
			orchestrator.getConversationContext().get("problem"));

		MultiAgentConversationSystem.ConversationTurn turn =
			conversationSystem.processConversationTurn(orchestrator.getConversationId()).get(2, TimeUnit.SECONDS);

		// Verify turn context includes conversation history and participant info
		Map<String, Object> turnContext = turn.getTurnContext();
		Assert.assertTrue("Turn context should include conversation history",
			turnContext.containsKey("conversationHistory"));
		Assert.assertTrue("Turn context should include participants",
			turnContext.containsKey("participants"));
		Assert.assertEquals("Turn context should include speaker role", "participant",
			turnContext.get("speakerRole"));

		// Complete turn with metadata
		Map<String, Object> turnMetadata = Map.of("solution", "x = 2, x = 3", "method", "factoring");
		conversationSystem.completeTurn(orchestrator.getConversationId(), turn.getTurnNumber(),
			"The solution is x = 2 and x = 3", turnMetadata);

		// Check metadata is added to conversation context
		Assert.assertEquals("Metadata should be added to context", "x = 2, x = 3",
			orchestrator.getConversationContext().get("solution"));

		logger.log(DEBUG, "Initial context: " + initialContext);
		logger.log(DEBUG, "Final context keys: " + orchestrator.getConversationContext().keySet());
		logger.log(DEBUG, "Turn metadata integrated: " + turnMetadata);
		logger.log(DEBUG, "✅ Conversation context test passed!");
	}

	@Test
	public void testConversationBehavior() {
		logger.log(DEBUG, "\n=== Conversation Behavior Test ===");

		MultiAgentConversationSystem.ConversationBehavior behavior = new MultiAgentConversationSystem.ConversationBehavior(
			true, // proactive
			5,    // maxTurnsBeforeSummary
			0.8f, // responseThreshold
			Set.of("urgent", "critical", "important"), // triggerKeywords
			Map.of("analyst", 0.9, "reviewer", 0.7) // roleAffinities
		);

		Assert.assertTrue("Should be proactive", behavior.isProactive());
		Assert.assertEquals("Max turns should be 5", 5, behavior.getMaxTurnsBeforeSummary());
		Assert.assertEquals("Response threshold should be 0.8", 0.8f, behavior.getResponseThreshold(), 0.001f);
		Assert.assertTrue("Should have trigger keyword", behavior.getTriggerKeywords().contains("urgent"));
		Assert.assertEquals("Analyst affinity should be 0.9", 0.9, behavior.getRoleAffinities().get("analyst"), 0.001);

		// Test default behavior
		MultiAgentConversationSystem.ConversationBehavior defaultBehavior = new MultiAgentConversationSystem.ConversationBehavior();
		Assert.assertFalse("Default should not be proactive", defaultBehavior.isProactive());
		Assert.assertEquals("Default max turns should be 10", 10, defaultBehavior.getMaxTurnsBeforeSummary());
		Assert.assertTrue("Default trigger keywords should be empty", defaultBehavior.getTriggerKeywords().isEmpty());

		logger.log(DEBUG, "Custom behavior - Proactive: " + behavior.isProactive());
		logger.log(DEBUG, "Custom behavior - Trigger keywords: " + behavior.getTriggerKeywords());
		logger.log(DEBUG, "Default behavior - Max turns: " + defaultBehavior.getMaxTurnsBeforeSummary());
		logger.log(DEBUG, "✅ Conversation behavior test passed!");
	}

	@Test
	public void testConversationManagement() throws Exception {
		logger.log(DEBUG, "\n=== Conversation Management Test ===");

		MultiAgentConversationSystem.AgentProfile testAgent = new MultiAgentConversationSystem.AgentProfile(
			"mgmt_agent", "Management Agent", "manager", "organized",
			Set.of("management"), Map.of(), new MultiAgentConversationSystem.ConversationBehavior()
		);

		conversationSystem.registerAgentProfile(testAgent);

		MultiAgentConversationSystem.RoundRobinFlow flow = new MultiAgentConversationSystem.RoundRobinFlow(List.of("mgmt_agent"), 3);

		// Start conversation
		MultiAgentConversationSystem.ConversationOrchestrator orchestrator =
			conversationSystem.startConversation(Set.of("mgmt_agent"), flow, Map.of()).get(2, TimeUnit.SECONDS);

		String conversationId = orchestrator.getConversationId();

		// Verify conversation is active
		Assert.assertTrue("Conversation should be in active list",
			conversationSystem.getActiveConversationIds().contains(conversationId));
		Assert.assertSame("Should get same orchestrator", orchestrator,
			conversationSystem.getConversation(conversationId));

		// End conversation manually
		conversationSystem.endConversation(conversationId);

		// Verify conversation is removed
		Assert.assertFalse("Conversation should be removed from active list",
			conversationSystem.getActiveConversationIds().contains(conversationId));
		Assert.assertEquals("State should be COMPLETED",
			MultiAgentConversationSystem.ConversationOrchestrator.ConversationState.COMPLETED,
			orchestrator.getState());

		logger.log(DEBUG, "Conversation lifecycle managed successfully");
		logger.log(DEBUG, "Active conversations after end: " + conversationSystem.getActiveConversationIds().size());
		logger.log(DEBUG, "✅ Conversation management test passed!");
	}

	@Test
	public void testFlowSummaryGeneration() {
		logger.log(DEBUG, "\n=== Flow Summary Generation Test ===");

		// Create mock turns
		MultiAgentConversationSystem.AgentProfile profile1 = new MultiAgentConversationSystem.AgentProfile(
			"agent1", "Agent One", "role1", "personality1", Set.of(), Map.of(),
			new MultiAgentConversationSystem.ConversationBehavior()
		);
		MultiAgentConversationSystem.AgentProfile profile2 = new MultiAgentConversationSystem.AgentProfile(
			"agent2", "Agent Two", "role2", "personality2", Set.of(), Map.of(),
			new MultiAgentConversationSystem.ConversationBehavior()
		);

		Map<String, MultiAgentConversationSystem.AgentProfile> participants = Map.of(
			"agent1", profile1,
			"agent2", profile2
		);

		List<MultiAgentConversationSystem.ConversationTurn> turns = Arrays.asList(
			new MultiAgentConversationSystem.ConversationTurn(1, "conv1", "agent1", profile1, Map.of(),
				java.time.Instant.now()),
			new MultiAgentConversationSystem.ConversationTurn(2, "conv1", "agent2", profile2, Map.of(),
				java.time.Instant.now()),
			new MultiAgentConversationSystem.ConversationTurn(3, "conv1", "agent1", profile1, Map.of(),
				java.time.Instant.now())
		);

		// Complete the turns
		turns.get(0).complete("First message", Map.of());
		turns.get(1).complete("Second message", Map.of());
		turns.get(2).complete("Third message", Map.of());

		// Test RoundRobinFlow summary
		MultiAgentConversationSystem.RoundRobinFlow roundRobinFlow =
			new MultiAgentConversationSystem.RoundRobinFlow(List.of("agent1", "agent2"), 10);

		Map<String, Object> roundRobinSummary = roundRobinFlow.generateSummary(turns, participants);
		Assert.assertEquals("Should have 3 total turns", 3, roundRobinSummary.get("totalTurns"));
		Assert.assertEquals("Should have 2 participants", 2, ((Set<?>)roundRobinSummary.get("participants")).size());

		// Test DynamicFlow summary
		MultiAgentConversationSystem.DynamicFlow dynamicFlow = new MultiAgentConversationSystem.DynamicFlow(
			10, content -> false, (t, p) -> "agent1"
		);

		Map<String, Object> dynamicSummary = dynamicFlow.generateSummary(turns, participants);
		Assert.assertEquals("Should have 3 total turns", 3, dynamicSummary.get("totalTurns"));
		Assert.assertTrue("Should have speaker participation data",
			dynamicSummary.containsKey("speakerParticipation"));

		@SuppressWarnings("unchecked")
		Map<String, Long> speakerCounts = (Map<String, Long>) dynamicSummary.get("speakerParticipation");
		Assert.assertEquals("Agent1 should have 2 turns", Long.valueOf(2), speakerCounts.get("agent1"));
		Assert.assertEquals("Agent2 should have 1 turn", Long.valueOf(1), speakerCounts.get("agent2"));

		logger.log(DEBUG, "RoundRobin summary: " + roundRobinSummary);
		logger.log(DEBUG, "Dynamic summary: " + dynamicSummary);
		logger.log(DEBUG, "✅ Flow summary generation test passed!");
	}
}
