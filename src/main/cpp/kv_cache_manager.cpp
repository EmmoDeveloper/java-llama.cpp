#include "kv_cache_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "llama_server.h"

// Sequence copying and manipulation

void KVCacheManager::copySequence(JNIEnv* env, jobject obj, jint srcSeqId, jint dstSeqId, jint p0, jint p1) {
	JNI_TRY(env)
	
	validateSequenceId(env, srcSeqId);
	validateSequenceId(env, dstSeqId);
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return;
	}
	
	llama_memory_seq_cp(memory, static_cast<llama_seq_id>(srcSeqId), static_cast<llama_seq_id>(dstSeqId), 
	                   static_cast<llama_pos>(p0), static_cast<llama_pos>(p1));
	
	JNI_CATCH_RET(env, void())
}

void KVCacheManager::keepSequence(JNIEnv* env, jobject obj, jint seqId) {
	JNI_TRY(env)
	
	validateSequenceId(env, seqId);
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return;
	}
	
	llama_memory_seq_keep(memory, static_cast<llama_seq_id>(seqId));
	
	JNI_CATCH_RET(env, void())
}

void KVCacheManager::addPositionDelta(JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1, jint delta) {
	JNI_TRY(env)
	
	validateSequenceId(env, seqId);
	validatePosition(env, p0);
	if (p1 >= 0) {
		validatePosition(env, p1);
		if (p1 <= p0) {
			JNIErrorHandler::throw_runtime_exception(env, "End position must be greater than start position");
			return;
		}
	}
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return;
	}
	
	llama_memory_seq_add(memory, static_cast<llama_seq_id>(seqId), 
	                    static_cast<llama_pos>(p0), static_cast<llama_pos>(p1), 
	                    static_cast<llama_pos>(delta));
	
	JNI_CATCH_RET(env, void())
}

void KVCacheManager::dividePositions(JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1, jint divisor) {
	JNI_TRY(env)
	
	validateSequenceId(env, seqId);
	validatePosition(env, p0);
	if (p1 >= 0) {
		validatePosition(env, p1);
		if (p1 <= p0) {
			JNIErrorHandler::throw_runtime_exception(env, "End position must be greater than start position");
			return;
		}
	}
	if (divisor <= 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Divisor must be positive");
		return;
	}
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return;
	}
	
	llama_memory_seq_div(memory, static_cast<llama_seq_id>(seqId), 
	                    static_cast<llama_pos>(p0), static_cast<llama_pos>(p1), 
	                    divisor);
	
	JNI_CATCH_RET(env, void())
}

// Sequence position queries

jint KVCacheManager::getSequenceMinPosition(JNIEnv* env, jobject obj, jint seqId) {
	JNI_TRY(env)
	
	validateSequenceId(env, seqId);
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return -1;
	}
	
	llama_pos minPos = llama_memory_seq_pos_min(memory, static_cast<llama_seq_id>(seqId));
	return static_cast<jint>(minPos);
	
	JNI_CATCH_RET(env, -1)
}

jint KVCacheManager::getSequenceMaxPosition(JNIEnv* env, jobject obj, jint seqId) {
	JNI_TRY(env)
	
	validateSequenceId(env, seqId);
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return -1;
	}
	
	llama_pos maxPos = llama_memory_seq_pos_max(memory, static_cast<llama_seq_id>(seqId));
	return static_cast<jint>(maxPos);
	
	JNI_CATCH_RET(env, -1)
}

// Memory capabilities

jboolean KVCacheManager::canShiftContext(JNIEnv* env, jobject obj) {
	JNI_TRY(env)
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return JNI_FALSE;
	}
	
	bool canShift = llama_memory_can_shift(memory);
	return canShift ? JNI_TRUE : JNI_FALSE;
	
	JNI_CATCH_RET(env, JNI_FALSE)
}

// Memory clearing

void KVCacheManager::clearMemory(JNIEnv* env, jobject obj, jboolean clearData) {
	JNI_TRY(env)
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return;
	}
	
	llama_memory_clear(memory, clearData == JNI_TRUE);
	
	JNI_CATCH_RET(env, void())
}

// Sequence removal

jboolean KVCacheManager::removeSequenceTokens(JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1) {
	JNI_TRY(env)
	
	validateSequenceId(env, seqId);
	validatePosition(env, p0);
	if (p1 >= 0) {
		validatePosition(env, p1);
		if (p1 <= p0) {
			JNIErrorHandler::throw_runtime_exception(env, "End position must be greater than start position");
			return JNI_FALSE;
		}
	}
	
	llama_memory_t memory = getMemory(env, obj);
	if (!memory) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get memory context");
		return JNI_FALSE;
	}
	
	bool result = llama_memory_seq_rm(memory, static_cast<llama_seq_id>(seqId), 
	                                 static_cast<llama_pos>(p0), static_cast<llama_pos>(p1));
	return result ? JNI_TRUE : JNI_FALSE;
	
	JNI_CATCH_RET(env, JNI_FALSE)
}

// Helper methods

struct llama_context* KVCacheManager::getContext(JNIEnv* env, jobject obj) {
	jclass cls = env->GetObjectClass(obj);
	jfieldID fieldId = env->GetFieldID(cls, "ctx", "J");
	if (!fieldId) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get context field");
		return nullptr;
	}
	
	jlong ctxHandle = env->GetLongField(obj, fieldId);
	return reinterpret_cast<struct llama_context*>(ctxHandle);
}

llama_memory_t KVCacheManager::getMemory(JNIEnv* env, jobject obj) {
	struct llama_context* ctx = getContext(env, obj);
	if (!ctx) {
		return nullptr;
	}
	return llama_get_memory(ctx);
}

void KVCacheManager::validateSequenceId(JNIEnv* env, jint seqId) {
	if (seqId < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Sequence ID must be non-negative");
	}
}

void KVCacheManager::validatePosition(JNIEnv* env, jint position) {
	if (position < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Position must be non-negative");
	}
}