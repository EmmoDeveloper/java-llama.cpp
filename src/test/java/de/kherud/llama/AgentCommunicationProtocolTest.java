package de.kherud.llama;

import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import org.junit.Assert;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

import static java.lang.System.Logger.Level.DEBUG;

public class AgentCommunicationProtocolTest {
	private static final System.Logger logger = System.getLogger(AgentCommunicationProtocolTest.class.getName());

	private AgentCommunicationProtocol protocol;

	@Before
	public void setUp() {
		protocol = new AgentCommunicationProtocol();
	}

	@After
	public void tearDown() {
		protocol.shutdown();
	}

	@Test
	public void testAgentRegistration() {
		logger.log(DEBUG, "\n=== Agent Registration Test ===");

		Set<String> capabilities1 = Set.of("search", "analyze", "report");
		Set<String> capabilities2 = Set.of("code", "test", "debug");

		protocol.registerAgent("agent1", "research", capabilities1);
		protocol.registerAgent("agent2", "developer", capabilities2);

		AgentCommunicationProtocol.AgentEndpoint agent1 = protocol.getAgent("agent1");
		AgentCommunicationProtocol.AgentEndpoint agent2 = protocol.getAgent("agent2");

		Assert.assertNotNull("Agent1 should be registered", agent1);
		Assert.assertNotNull("Agent2 should be registered", agent2);
		Assert.assertEquals("Agent1 type should match", "research", agent1.getAgentType());
		Assert.assertEquals("Agent2 type should match", "developer", agent2.getAgentType());
		Assert.assertTrue("Agent1 should have search capability",
			agent1.getCapabilities().contains("search"));
		Assert.assertTrue("Agent2 should have code capability",
			agent2.getCapabilities().contains("code"));

		Set<String> activeAgents = protocol.getActiveAgents();
		Assert.assertEquals("Should have 2 active agents", 2, activeAgents.size());

		logger.log(DEBUG, "Registered agents: " + activeAgents);
		logger.log(DEBUG, "Agent1 capabilities: " + agent1.getCapabilities());
		logger.log(DEBUG, "Agent2 capabilities: " + agent2.getCapabilities());
		logger.log(DEBUG, "✅ Agent registration test passed!");
	}

	@Test
	public void testMessageSending() throws Exception {
		logger.log(DEBUG, "\n=== Message Sending Test ===");

		protocol.registerAgent("sender", "producer", Set.of("send"));
		protocol.registerAgent("receiver", "consumer", Set.of("receive"));

		Map<String, Object> content = new HashMap<>();
		content.put("task", "analyze_data");
		content.put("priority", "high");

		AgentCommunicationProtocol.Message message = new AgentCommunicationProtocol.Message.Builder()
			.from("sender")
			.to("receiver")
			.type(AgentCommunicationProtocol.MessageType.REQUEST)
			.priority(AgentCommunicationProtocol.Priority.HIGH)
			.content(content)
			.build();

		CompletableFuture<Boolean> result = protocol.sendMessage(message);
		Assert.assertTrue("Message should be sent successfully", result.get(2, TimeUnit.SECONDS));

		// Check receiver inbox
		AgentCommunicationProtocol.AgentEndpoint receiver = protocol.getAgent("receiver");
		AgentCommunicationProtocol.Message received = receiver.receiveMessage(1, TimeUnit.SECONDS);

		Assert.assertNotNull("Message should be received", received);
		Assert.assertEquals("Sender should match", "sender", received.getSenderId());
		Assert.assertEquals("Task should match", "analyze_data", received.getContent().get("task"));
		Assert.assertEquals("Priority should be HIGH",
			AgentCommunicationProtocol.Priority.HIGH, received.getPriority());

		logger.log(DEBUG, "Message sent from: " + message.getSenderId());
		logger.log(DEBUG, "Message received by: " + message.getRecipientId());
		logger.log(DEBUG, "Message content: " + message.getContent());
		logger.log(DEBUG, "✅ Message sending test passed!");
	}

	@Test
	public void testMessageHandlers() throws Exception {
		logger.log(DEBUG, "\n=== Message Handlers Test ===");

		protocol.registerAgent("handler_agent", "processor", Set.of("handle"));

		AtomicReference<AgentCommunicationProtocol.Message> handledMessage = new AtomicReference<>();
		CountDownLatch latch = new CountDownLatch(1);

		// Register handler
		AgentCommunicationProtocol.AgentEndpoint agent = protocol.getAgent("handler_agent");
		agent.registerHandler(AgentCommunicationProtocol.MessageType.REQUEST, message -> {
			handledMessage.set(message);
			latch.countDown();
		});

		// Send message
		AgentCommunicationProtocol.Message message = new AgentCommunicationProtocol.Message.Builder()
			.from("sender")
			.to("handler_agent")
			.type(AgentCommunicationProtocol.MessageType.REQUEST)
			.addContent("action", "process")
			.build();

		protocol.sendMessage(message);

		Assert.assertTrue("Handler should be called", latch.await(2, TimeUnit.SECONDS));
		Assert.assertNotNull("Message should be handled", handledMessage.get());
		Assert.assertEquals("Action should match", "process",
			handledMessage.get().getContent().get("action"));

		logger.log(DEBUG, "Handler received message with action: " +
			handledMessage.get().getContent().get("action"));
		logger.log(DEBUG, "✅ Message handlers test passed!");
	}

	@Test
	public void testBroadcastMessage() throws Exception {
		logger.log(DEBUG, "\n=== Broadcast Message Test ===");

		// Register multiple agents
		protocol.registerAgent("broadcaster", "sender", Set.of("broadcast"));
		protocol.registerAgent("agent1", "receiver", Set.of("listen"));
		protocol.registerAgent("agent2", "receiver", Set.of("listen"));
		protocol.registerAgent("agent3", "receiver", Set.of("listen"));

		Map<String, Object> content = new HashMap<>();
		content.put("announcement", "System update");
		content.put("timestamp", System.currentTimeMillis());

		Set<String> recipients = Set.of("agent1", "agent2", "agent3");

		CompletableFuture<Map<String, Boolean>> results = protocol.broadcastMessage(
			"broadcaster", recipients, AgentCommunicationProtocol.MessageType.INFORM, content);

		Map<String, Boolean> deliveryResults = results.get(2, TimeUnit.SECONDS);

		Assert.assertEquals("Should have 3 delivery results", 3, deliveryResults.size());
		Assert.assertTrue("Agent1 should receive message", deliveryResults.get("agent1"));
		Assert.assertTrue("Agent2 should receive message", deliveryResults.get("agent2"));
		Assert.assertTrue("Agent3 should receive message", deliveryResults.get("agent3"));

		// Verify each agent received the message
		for (String agentId : recipients) {
			AgentCommunicationProtocol.AgentEndpoint agent = protocol.getAgent(agentId);
			AgentCommunicationProtocol.Message received = agent.receiveMessage(1, TimeUnit.SECONDS);
			Assert.assertNotNull("Agent " + agentId + " should receive message", received);
			Assert.assertEquals("Announcement should match", "System update",
				received.getContent().get("announcement"));
		}

		logger.log(DEBUG, "Broadcast sent to: " + recipients);
		logger.log(DEBUG, "Delivery results: " + deliveryResults);
		logger.log(DEBUG, "✅ Broadcast message test passed!");
	}

	@Test
	public void testConversationManagement() throws Exception {
		logger.log(DEBUG, "\n=== Conversation Management Test ===");

		protocol.registerAgent("agent1", "participant", Set.of("chat"));
		protocol.registerAgent("agent2", "participant", Set.of("chat"));
		protocol.registerAgent("agent3", "participant", Set.of("chat"));

		Set<String> participants = Set.of("agent1", "agent2", "agent3");
		AgentCommunicationProtocol.Conversation conversation = protocol.startConversation(participants);

		Assert.assertNotNull("Conversation should be created", conversation);
		Assert.assertEquals("Should have 3 participants", 3, conversation.getParticipants().size());
		Assert.assertEquals("State should be ACTIVE",
			AgentCommunicationProtocol.Conversation.ConversationState.ACTIVE,
			conversation.getState());

		// Send messages in conversation
		AgentCommunicationProtocol.Message msg1 = new AgentCommunicationProtocol.Message.Builder()
			.from("agent1")
			.to("agent2")
			.type(AgentCommunicationProtocol.MessageType.INFORM)
			.inConversation(conversation.getId())
			.addContent("text", "Hello from agent1")
			.build();

		AgentCommunicationProtocol.Message msg2 = new AgentCommunicationProtocol.Message.Builder()
			.from("agent2")
			.to("agent3")
			.type(AgentCommunicationProtocol.MessageType.RESPONSE)
			.inConversation(conversation.getId())
			.replyTo(msg1.getId())
			.addContent("text", "Response from agent2")
			.build();

		CompletableFuture<Boolean> send1 = protocol.sendMessage(msg1);
		CompletableFuture<Boolean> send2 = protocol.sendMessage(msg2);

		// Wait for messages to be sent
		Assert.assertTrue("First message should be sent", send1.get(2, TimeUnit.SECONDS));
		Assert.assertTrue("Second message should be sent", send2.get(2, TimeUnit.SECONDS));

		// Check conversation messages
		AgentCommunicationProtocol.Conversation retrievedConv =
			protocol.getConversation(conversation.getId());
		Assert.assertEquals("Should have 2 messages", 2, retrievedConv.getMessages().size());

		// End conversation
		protocol.endConversation(conversation.getId(),
			AgentCommunicationProtocol.Conversation.ConversationState.COMPLETED);
		Assert.assertEquals("State should be COMPLETED",
			AgentCommunicationProtocol.Conversation.ConversationState.COMPLETED,
			retrievedConv.getState());

		logger.log(DEBUG, "Conversation ID: " + conversation.getId());
		logger.log(DEBUG, "Participants: " + conversation.getParticipants());
		logger.log(DEBUG, "Messages exchanged: " + conversation.getMessages().size());
		logger.log(DEBUG, "✅ Conversation management test passed!");
	}

	@Test
	public void testCapabilityQuerying() {
		logger.log(DEBUG, "\n=== Capability Querying Test ===");

		protocol.registerAgent("search_agent", "searcher", Set.of("search", "index", "query"));
		protocol.registerAgent("analyze_agent", "analyzer", Set.of("analyze", "report"));
		protocol.registerAgent("code_agent", "coder", Set.of("code", "test", "debug"));
		protocol.registerAgent("multi_agent", "versatile", Set.of("search", "analyze", "code"));

		Set<String> searchAgents = protocol.findAgentsByCapability("search");
		Set<String> analyzeAgents = protocol.findAgentsByCapability("analyze");
		Set<String> codeAgents = protocol.findAgentsByCapability("code");

		Assert.assertEquals("Should find 2 agents with search capability", 2, searchAgents.size());
		Assert.assertTrue("Should include search_agent", searchAgents.contains("search_agent"));
		Assert.assertTrue("Should include multi_agent", searchAgents.contains("multi_agent"));

		Assert.assertEquals("Should find 2 agents with analyze capability", 2, analyzeAgents.size());
		Assert.assertEquals("Should find 2 agents with code capability", 2, codeAgents.size());

		logger.log(DEBUG, "Agents with 'search' capability: " + searchAgents);
		logger.log(DEBUG, "Agents with 'analyze' capability: " + analyzeAgents);
		logger.log(DEBUG, "Agents with 'code' capability: " + codeAgents);
		logger.log(DEBUG, "✅ Capability querying test passed!");
	}

	@Test
	public void testPriorityOrdering() throws Exception {
		logger.log(DEBUG, "\n=== Priority Ordering Test ===");

		protocol.registerAgent("priority_agent", "processor", Set.of("process"));

		// Send messages with different priorities
		AgentCommunicationProtocol.Message lowPriority = new AgentCommunicationProtocol.Message.Builder()
			.from("sender")
			.to("priority_agent")
			.type(AgentCommunicationProtocol.MessageType.INFORM)
			.priority(AgentCommunicationProtocol.Priority.LOW)
			.addContent("id", "low")
			.build();

		AgentCommunicationProtocol.Message highPriority = new AgentCommunicationProtocol.Message.Builder()
			.from("sender")
			.to("priority_agent")
			.type(AgentCommunicationProtocol.MessageType.REQUEST)
			.priority(AgentCommunicationProtocol.Priority.HIGH)
			.addContent("id", "high")
			.build();

		AgentCommunicationProtocol.Message criticalPriority = new AgentCommunicationProtocol.Message.Builder()
			.from("sender")
			.to("priority_agent")
			.type(AgentCommunicationProtocol.MessageType.REQUEST)
			.priority(AgentCommunicationProtocol.Priority.CRITICAL)
			.addContent("id", "critical")
			.build();

		AgentCommunicationProtocol.Message normalPriority = new AgentCommunicationProtocol.Message.Builder()
			.from("sender")
			.to("priority_agent")
			.type(AgentCommunicationProtocol.MessageType.INFORM)
			.priority(AgentCommunicationProtocol.Priority.NORMAL)
			.addContent("id", "normal")
			.build();

		// Send in random order
		protocol.sendMessage(lowPriority);
		protocol.sendMessage(highPriority);
		protocol.sendMessage(criticalPriority);
		protocol.sendMessage(normalPriority);

		Thread.sleep(100); // Allow messages to be delivered

		// Receive messages - should be in priority order
		AgentCommunicationProtocol.AgentEndpoint agent = protocol.getAgent("priority_agent");
		List<AgentCommunicationProtocol.Message> received = agent.receiveAllMessages();

		Assert.assertEquals("Should receive 4 messages", 4, received.size());
		Assert.assertEquals("First should be critical", "critical", received.get(0).getContent().get("id"));
		Assert.assertEquals("Second should be high", "high", received.get(1).getContent().get("id"));
		Assert.assertEquals("Third should be normal", "normal", received.get(2).getContent().get("id"));
		Assert.assertEquals("Fourth should be low", "low", received.get(3).getContent().get("id"));

		logger.log(DEBUG, "Message priority order:");
		for (AgentCommunicationProtocol.Message msg : received) {
			logger.log(DEBUG, "  - " + msg.getContent().get("id") + " (" + msg.getPriority() + ")");
		}
		logger.log(DEBUG, "✅ Priority ordering test passed!");
	}

	@Test
	public void testHeartbeatAndHealth() throws Exception {
		logger.log(DEBUG, "\n=== Heartbeat and Health Test ===");

		protocol.registerAgent("healthy_agent", "worker", Set.of("work"));

		// Initially healthy
		Assert.assertTrue("Agent should be healthy initially",
			protocol.isAgentHealthy("healthy_agent", 5));

		// Send heartbeat
		protocol.sendHeartbeat("healthy_agent");
		Thread.sleep(100);

		Assert.assertTrue("Agent should still be healthy",
			protocol.isAgentHealthy("healthy_agent", 1));

		// Simulate inactivity
		Thread.sleep(2000);

		Assert.assertFalse("Agent should be unhealthy after inactivity",
			protocol.isAgentHealthy("healthy_agent", 1));
		Assert.assertTrue("Agent should still be healthy with longer threshold",
			protocol.isAgentHealthy("healthy_agent", 10));

		// Send heartbeat to recover
		protocol.sendHeartbeat("healthy_agent");
		Assert.assertTrue("Agent should be healthy after heartbeat",
			protocol.isAgentHealthy("healthy_agent", 1));

		logger.log(DEBUG, "Health check with 1s threshold: " +
			protocol.isAgentHealthy("healthy_agent", 1));
		logger.log(DEBUG, "✅ Heartbeat and health test passed!");
	}

	@Test
	public void testAgentUnregistration() {
		logger.log(DEBUG, "\n=== Agent Unregistration Test ===");

		protocol.registerAgent("temp_agent", "temporary", Set.of("temp"));

		Assert.assertNotNull("Agent should exist", protocol.getAgent("temp_agent"));
		Assert.assertTrue("Agent should be active", protocol.getActiveAgents().contains("temp_agent"));

		protocol.unregisterAgent("temp_agent");

		Assert.assertNull("Agent should not exist after unregistration",
			protocol.getAgent("temp_agent"));
		Assert.assertFalse("Agent should not be in active list",
			protocol.getActiveAgents().contains("temp_agent"));

		logger.log(DEBUG, "Active agents after unregistration: " + protocol.getActiveAgents());
		logger.log(DEBUG, "✅ Agent unregistration test passed!");
	}
}
