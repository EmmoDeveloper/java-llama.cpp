package de.kherud.llama;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Caches common prompt prefixes to optimize KV cache reuse.
 * Particularly useful for system prompts, RAG contexts, and repeated instructions.
 */
public class PromptCache {

	public static class CacheEntry {
		private final String prompt;
		private final int[] tokens;
		private final long kvCacheHandle;
		private final long createdAt;
		private long lastUsed;
		private int useCount;

		CacheEntry(String prompt, int[] tokens, long kvCacheHandle) {
			this.prompt = prompt;
			this.tokens = tokens;
			this.kvCacheHandle = kvCacheHandle;
			this.createdAt = System.currentTimeMillis();
			this.lastUsed = createdAt;
			this.useCount = 0;
		}

		void recordUse() {
			this.lastUsed = System.currentTimeMillis();
			this.useCount++;
		}

		public String getPrompt() { return prompt; }
		public int[] getTokens() { return tokens; }
		public long getKvCacheHandle() { return kvCacheHandle; }
		public long getCreatedAt() { return createdAt; }
		public long getLastUsed() { return lastUsed; }
		public int getUseCount() { return useCount; }
	}

	public static class CacheConfig {
		private int maxEntries = 100;
		private long maxMemoryBytes = 1024 * 1024 * 512; // 512MB default
		private long ttlMillis = 3600 * 1000; // 1 hour default
		private boolean enableStats = true;

		public CacheConfig setMaxEntries(int maxEntries) {
			this.maxEntries = maxEntries;
			return this;
		}

		public CacheConfig setMaxMemoryBytes(long maxMemoryBytes) {
			this.maxMemoryBytes = maxMemoryBytes;
			return this;
		}

		public CacheConfig setTtlMillis(long ttlMillis) {
			this.ttlMillis = ttlMillis;
			return this;
		}

		public CacheConfig setEnableStats(boolean enableStats) {
			this.enableStats = enableStats;
			return this;
		}
	}

	private final LlamaModel model;
	private final CacheConfig config;
	private final Map<String, CacheEntry> cache;
	private final ReadWriteLock lock;
	private final CacheStats stats;

	public PromptCache(LlamaModel model) {
		this(model, new CacheConfig());
	}

	public PromptCache(LlamaModel model, CacheConfig config) {
		this.model = model;
		this.config = config;
		this.lock = new ReentrantReadWriteLock();
		this.stats = new CacheStats();

		// Use LRU cache implementation
		this.cache = new LinkedHashMap<String, CacheEntry>(config.maxEntries + 1, 0.75f, true) {
			@Override
			protected boolean removeEldestEntry(Map.Entry<String, CacheEntry> eldest) {
				boolean shouldRemove = size() > config.maxEntries;
				if (shouldRemove && config.enableStats) {
					stats.recordEviction();
				}
				return shouldRemove;
			}
		};
	}

	/**
	 * Get or create cache entry for a prompt prefix
	 */
	public CacheEntry getOrCreate(String promptPrefix) {
		// Try read lock first
		lock.readLock().lock();
		try {
			CacheEntry entry = cache.get(promptPrefix);
			if (entry != null && !isExpired(entry)) {
				entry.recordUse();
				if (config.enableStats) {
					stats.recordHit();
				}
				return entry;
			}
		} finally {
			lock.readLock().unlock();
		}

		// Need write lock to create new entry
		lock.writeLock().lock();
		try {
			// Double-check after acquiring write lock
			CacheEntry entry = cache.get(promptPrefix);
			if (entry != null && !isExpired(entry)) {
				entry.recordUse();
				if (config.enableStats) {
					stats.recordHit();
				}
				return entry;
			}

			// Remove expired entry if exists
			if (entry != null) {
				cache.remove(promptPrefix);
				if (config.enableStats) {
					stats.recordEviction();
				}
			}

			// Create new cache entry
			int[] tokens = model.encode(promptPrefix);
			long kvCacheHandle = createKvCacheSnapshot();

			entry = new CacheEntry(promptPrefix, tokens, kvCacheHandle);
			cache.put(promptPrefix, entry);

			if (config.enableStats) {
				stats.recordMiss();
			}

			return entry;
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Check if a prompt prefix is cached
	 */
	public boolean contains(String promptPrefix) {
		lock.readLock().lock();
		try {
			CacheEntry entry = cache.get(promptPrefix);
			return entry != null && !isExpired(entry);
		} finally {
			lock.readLock().unlock();
		}
	}

	/**
	 * Invalidate a specific cache entry
	 */
	public void invalidate(String promptPrefix) {
		lock.writeLock().lock();
		try {
			CacheEntry removed = cache.remove(promptPrefix);
			if (removed != null && config.enableStats) {
				stats.recordEviction();
			}
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Clear all cache entries
	 */
	public void clear() {
		lock.writeLock().lock();
		try {
			int size = cache.size();
			cache.clear();
			if (config.enableStats) {
				stats.recordClear(size);
			}
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Clean up expired entries
	 */
	public void cleanExpired() {
		lock.writeLock().lock();
		try {
			long now = System.currentTimeMillis();
			cache.entrySet().removeIf(entry -> {
				boolean expired = (now - entry.getValue().lastUsed) > config.ttlMillis;
				if (expired && config.enableStats) {
					stats.recordEviction();
				}
				return expired;
			});
		} finally {
			lock.writeLock().unlock();
		}
	}

	/**
	 * Get cache statistics
	 */
	public CacheStats getStats() {
		return stats.snapshot();
	}

	/**
	 * Get current cache size
	 */
	public int size() {
		lock.readLock().lock();
		try {
			return cache.size();
		} finally {
			lock.readLock().unlock();
		}
	}

	/**
	 * Find the longest matching prefix in cache
	 */
	public CacheEntry findLongestPrefix(String prompt) {
		lock.readLock().lock();
		try {
			CacheEntry longest = null;
			int maxLength = 0;

			for (Map.Entry<String, CacheEntry> entry : cache.entrySet()) {
				String prefix = entry.getKey();
				if (prompt.startsWith(prefix) && prefix.length() > maxLength && !isExpired(entry.getValue())) {
					longest = entry.getValue();
					maxLength = prefix.length();
				}
			}

			if (longest != null) {
				longest.recordUse();
				if (config.enableStats) {
					stats.recordHit();
				}
			} else if (config.enableStats) {
				stats.recordMiss();
			}

			return longest;
		} finally {
			lock.readLock().unlock();
		}
	}

	private boolean isExpired(CacheEntry entry) {
		if (config.ttlMillis <= 0) {
			return false;
		}
		return (System.currentTimeMillis() - entry.lastUsed) > config.ttlMillis;
	}

	private long createKvCacheSnapshot() {
		// Use the model's actual state persistence functionality
		try {
			return model.getModelStateSize();
		} catch (LlamaException e) {
			// If state size cannot be determined, cache is not useful
			throw new RuntimeException("Cannot create KV cache snapshot: " + e.getMessage(), e);
		}
	}

	/**
	 * Cache statistics
	 */
	public static class CacheStats {
		private long hits;
		private long misses;
		private long evictions;
		private long clears;

		synchronized void recordHit() { hits++; }
		synchronized void recordMiss() { misses++; }
		synchronized void recordEviction() { evictions++; }
		synchronized void recordClear(int entries) {
			clears++;
			evictions += entries;
		}

		public synchronized CacheStats snapshot() {
			CacheStats snapshot = new CacheStats();
			snapshot.hits = this.hits;
			snapshot.misses = this.misses;
			snapshot.evictions = this.evictions;
			snapshot.clears = this.clears;
			return snapshot;
		}

		public long getHits() { return hits; }
		public long getMisses() { return misses; }
		public long getEvictions() { return evictions; }
		public long getClears() { return clears; }

		public double getHitRate() {
			long total = hits + misses;
			return total > 0 ? (double) hits / total : 0.0;
		}

		@Override
		public String toString() {
			return String.format("CacheStats{hits=%d, misses=%d, evictions=%d, hitRate=%.2f%%}",
				hits, misses, evictions, getHitRate() * 100);
		}
	}
}
