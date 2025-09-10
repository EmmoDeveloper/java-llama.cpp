package de.kherud.llama;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.annotation.JsonIgnore;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Agent Communication Protocol (ACP) for multi-agent collaboration.
 * Provides message passing, coordination, and conversation management.
 */
public class AgentCommunicationProtocol {
	
	private final Map<String, AgentEndpoint> agents = new ConcurrentHashMap<>();
	private final Map<String, Conversation> conversations = new ConcurrentHashMap<>();
	private final ExecutorService messageExecutor = Executors.newCachedThreadPool();
	private final ObjectMapper objectMapper = new ObjectMapper();
	
	/**
	 * Message types for agent communication
	 */
	public enum MessageType {
		REQUEST,      // Request for action
		RESPONSE,     // Response to request
		INFORM,       // Information sharing
		QUERY,        // Query for information
		CONFIRM,      // Confirmation of action
		PROPOSE,      // Proposal for action
		ACCEPT,       // Accept proposal
		REJECT,       // Reject proposal
		SUBSCRIBE,    // Subscribe to updates
		NOTIFY,       // Notification/update
		ERROR,        // Error message
		HEARTBEAT     // Keep-alive signal
	}
	
	/**
	 * Message priority levels
	 */
	public enum Priority {
		LOW(0),
		NORMAL(1),
		HIGH(2),
		CRITICAL(3);
		
		private final int level;
		
		Priority(int level) {
			this.level = level;
		}
		
		public int getLevel() { return level; }
	}
	
	/**
	 * Agent message
	 */
	public static class Message {
		private final String id;
		private final String conversationId;
		private final String senderId;
		private final String recipientId;
		private final MessageType type;
		private final Priority priority;
		private final Map<String, Object> content;
		private final Instant timestamp;
		private final String inReplyTo;
		private final Map<String, String> metadata;
		
		public Message(String senderId, String recipientId, MessageType type, Map<String, Object> content) {
			this(UUID.randomUUID().toString(), null, senderId, recipientId, type, 
				Priority.NORMAL, content, null, new HashMap<>());
		}
		
		public Message(String id, String conversationId, String senderId, String recipientId,
					   MessageType type, Priority priority, Map<String, Object> content,
					   String inReplyTo, Map<String, String> metadata) {
			this.id = id;
			this.conversationId = conversationId;
			this.senderId = senderId;
			this.recipientId = recipientId;
			this.type = type;
			this.priority = priority;
			this.content = content != null ? content : new HashMap<>();
			this.timestamp = Instant.now();
			this.inReplyTo = inReplyTo;
			this.metadata = metadata != null ? metadata : new HashMap<>();
		}
		
		// Builder pattern for message construction
		public static class Builder {
			private String id = UUID.randomUUID().toString();
			private String conversationId;
			private String senderId;
			private String recipientId;
			private MessageType type = MessageType.INFORM;
			private Priority priority = Priority.NORMAL;
			private Map<String, Object> content = new HashMap<>();
			private String inReplyTo;
			private Map<String, String> metadata = new HashMap<>();
			
			public Builder from(String senderId) {
				this.senderId = senderId;
				return this;
			}
			
			public Builder to(String recipientId) {
				this.recipientId = recipientId;
				return this;
			}
			
			public Builder type(MessageType type) {
				this.type = type;
				return this;
			}
			
			public Builder priority(Priority priority) {
				this.priority = priority;
				return this;
			}
			
			public Builder content(Map<String, Object> content) {
				this.content = content;
				return this;
			}
			
			public Builder addContent(String key, Object value) {
				this.content.put(key, value);
				return this;
			}
			
			public Builder inConversation(String conversationId) {
				this.conversationId = conversationId;
				return this;
			}
			
			public Builder replyTo(String messageId) {
				this.inReplyTo = messageId;
				return this;
			}
			
			public Builder withMetadata(String key, String value) {
				this.metadata.put(key, value);
				return this;
			}
			
			public Message build() {
				if (senderId == null || recipientId == null) {
					throw new IllegalStateException("Sender and recipient are required");
				}
				return new Message(id, conversationId, senderId, recipientId, type, 
					priority, content, inReplyTo, metadata);
			}
		}
		
		// Getters
		public String getId() { return id; }
		public String getConversationId() { return conversationId; }
		public String getSenderId() { return senderId; }
		public String getRecipientId() { return recipientId; }
		public MessageType getType() { return type; }
		public Priority getPriority() { return priority; }
		public Map<String, Object> getContent() { return content; }
		public Instant getTimestamp() { return timestamp; }
		public String getInReplyTo() { return inReplyTo; }
		public Map<String, String> getMetadata() { return metadata; }
	}
	
	/**
	 * Agent endpoint for communication
	 */
	public static class AgentEndpoint {
		private final String agentId;
		private final String agentType;
		private final Set<String> capabilities;
		private final BlockingQueue<Message> inbox;
		private final Map<MessageType, Consumer<Message>> handlers;
		private volatile boolean active;
		private Instant lastHeartbeat;
		
		public AgentEndpoint(String agentId, String agentType, Set<String> capabilities) {
			this.agentId = agentId;
			this.agentType = agentType;
			this.capabilities = capabilities != null ? capabilities : new HashSet<>();
			this.inbox = new PriorityBlockingQueue<>(100, 
				Comparator.comparing((Message m) -> m.getPriority().getLevel()).reversed()
					.thenComparing(Message::getTimestamp));
			this.handlers = new ConcurrentHashMap<>();
			this.active = true;
			this.lastHeartbeat = Instant.now();
		}
		
		public void registerHandler(MessageType type, Consumer<Message> handler) {
			handlers.put(type, handler);
		}
		
		public void handleMessage(Message message) {
			Consumer<Message> handler = handlers.get(message.getType());
			if (handler != null) {
				handler.accept(message);
			} else {
				// Default handling - add to inbox
				try {
					inbox.offer(message, 1, TimeUnit.SECONDS);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
			}
		}
		
		public Message receiveMessage(long timeout, TimeUnit unit) throws InterruptedException {
			return inbox.poll(timeout, unit);
		}
		
		public List<Message> receiveAllMessages() {
			List<Message> messages = new ArrayList<>();
			inbox.drainTo(messages);
			return messages;
		}
		
		// Getters and setters
		public String getAgentId() { return agentId; }
		public String getAgentType() { return agentType; }
		public Set<String> getCapabilities() { return capabilities; }
		public boolean isActive() { return active; }
		public void setActive(boolean active) { this.active = active; }
		public Instant getLastHeartbeat() { return lastHeartbeat; }
		public void updateHeartbeat() { this.lastHeartbeat = Instant.now(); }
		
		@JsonIgnore
		public BlockingQueue<Message> getInbox() { return inbox; }
	}
	
	/**
	 * Conversation between agents
	 */
	public static class Conversation {
		private final String id;
		private final Set<String> participants;
		private final List<Message> messages;
		private final Instant startTime;
		private Instant lastActivity;
		private ConversationState state;
		private Map<String, Object> context;
		
		public enum ConversationState {
			ACTIVE,
			PAUSED,
			COMPLETED,
			FAILED
		}
		
		public Conversation(String id, Set<String> participants) {
			this.id = id;
			this.participants = new HashSet<>(participants);
			this.messages = new CopyOnWriteArrayList<>();
			this.startTime = Instant.now();
			this.lastActivity = startTime;
			this.state = ConversationState.ACTIVE;
			this.context = new ConcurrentHashMap<>();
		}
		
		public void addMessage(Message message) {
			messages.add(message);
			lastActivity = Instant.now();
		}
		
		public List<Message> getMessagesSince(Instant since) {
			return messages.stream()
				.filter(m -> m.getTimestamp().isAfter(since))
				.collect(Collectors.toList());
		}
		
		// Getters and setters
		public String getId() { return id; }
		public Set<String> getParticipants() { return participants; }
		public List<Message> getMessages() { return messages; }
		public Instant getStartTime() { return startTime; }
		public Instant getLastActivity() { return lastActivity; }
		public ConversationState getState() { return state; }
		public void setState(ConversationState state) { this.state = state; }
		public Map<String, Object> getContext() { return context; }
	}
	
	/**
	 * Register an agent endpoint
	 */
	public void registerAgent(String agentId, String agentType, Set<String> capabilities) {
		AgentEndpoint endpoint = new AgentEndpoint(agentId, agentType, capabilities);
		agents.put(agentId, endpoint);
	}
	
	/**
	 * Unregister an agent
	 */
	public void unregisterAgent(String agentId) {
		AgentEndpoint endpoint = agents.remove(agentId);
		if (endpoint != null) {
			endpoint.setActive(false);
		}
	}
	
	/**
	 * Send a message between agents
	 */
	public CompletableFuture<Boolean> sendMessage(Message message) {
		return CompletableFuture.supplyAsync(() -> {
			// Add to conversation if specified (do this first, before checking recipient)
			if (message.getConversationId() != null) {
				Conversation conversation = conversations.get(message.getConversationId());
				if (conversation != null) {
					conversation.addMessage(message);
				}
			}
			
			// Check and deliver to recipient
			AgentEndpoint recipient = agents.get(message.getRecipientId());
			if (recipient == null || !recipient.isActive()) {
				return false;
			}
			
			// Deliver message
			recipient.handleMessage(message);
			return true;
		}, messageExecutor);
	}
	
	/**
	 * Broadcast a message to multiple agents
	 */
	public CompletableFuture<Map<String, Boolean>> broadcastMessage(String senderId, 
			Set<String> recipientIds, MessageType type, Map<String, Object> content) {
		Map<String, CompletableFuture<Boolean>> futures = new HashMap<>();
		
		for (String recipientId : recipientIds) {
			Message message = new Message.Builder()
				.from(senderId)
				.to(recipientId)
				.type(type)
				.content(content)
				.build();
			futures.put(recipientId, sendMessage(message));
		}
		
		return CompletableFuture.allOf(futures.values().toArray(new CompletableFuture[0]))
			.thenApply(v -> futures.entrySet().stream()
				.collect(Collectors.toMap(
					Map.Entry::getKey,
					e -> e.getValue().join()
				)));
	}
	
	/**
	 * Start a new conversation
	 */
	public Conversation startConversation(Set<String> participants) {
		String conversationId = UUID.randomUUID().toString();
		Conversation conversation = new Conversation(conversationId, participants);
		conversations.put(conversationId, conversation);
		return conversation;
	}
	
	/**
	 * End a conversation
	 */
	public void endConversation(String conversationId, Conversation.ConversationState finalState) {
		Conversation conversation = conversations.get(conversationId);
		if (conversation != null) {
			conversation.setState(finalState);
		}
	}
	
	/**
	 * Query agents by capability
	 */
	public Set<String> findAgentsByCapability(String capability) {
		return agents.values().stream()
			.filter(agent -> agent.getCapabilities().contains(capability))
			.map(AgentEndpoint::getAgentId)
			.collect(Collectors.toSet());
	}
	
	/**
	 * Get agent endpoint
	 */
	public AgentEndpoint getAgent(String agentId) {
		return agents.get(agentId);
	}
	
	/**
	 * Get conversation
	 */
	public Conversation getConversation(String conversationId) {
		return conversations.get(conversationId);
	}
	
	/**
	 * Get all active agents
	 */
	public Set<String> getActiveAgents() {
		return agents.values().stream()
			.filter(AgentEndpoint::isActive)
			.map(AgentEndpoint::getAgentId)
			.collect(Collectors.toSet());
	}
	
	/**
	 * Send heartbeat for agent
	 */
	public void sendHeartbeat(String agentId) {
		AgentEndpoint agent = agents.get(agentId);
		if (agent != null) {
			agent.updateHeartbeat();
		}
	}
	
	/**
	 * Check agent health
	 */
	public boolean isAgentHealthy(String agentId, long maxInactivitySeconds) {
		AgentEndpoint agent = agents.get(agentId);
		if (agent == null || !agent.isActive()) {
			return false;
		}
		
		Instant threshold = Instant.now().minusSeconds(maxInactivitySeconds);
		return agent.getLastHeartbeat().isAfter(threshold);
	}
	
	/**
	 * Shutdown the protocol
	 */
	public void shutdown() {
		messageExecutor.shutdown();
		try {
			if (!messageExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
				messageExecutor.shutdownNow();
			}
		} catch (InterruptedException e) {
			messageExecutor.shutdownNow();
			Thread.currentThread().interrupt();
		}
	}
}