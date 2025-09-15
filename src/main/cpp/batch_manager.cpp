#include "batch_manager.h"
#include "jni_utils.h"
#include <memory>
#include <unordered_map>
#include <mutex>

static std::mutex batchMutex;
static std::unordered_map<jlong, std::unique_ptr<llama_batch>> batchRegistry;
static jlong nextBatchId = 1;

llama_batch* BatchManager::getBatch(jlong handle) {
	std::lock_guard<std::mutex> lock(batchMutex);
	auto it = batchRegistry.find(handle);
	return (it != batchRegistry.end()) ? it->second.get() : nullptr;
}

llama_context* BatchManager::getContext(JNIEnv* env, jobject modelObj) {
	jclass cls = env->GetObjectClass(modelObj);
	jfieldID contextField = env->GetFieldID(cls, "ctx", "J");
	if (!contextField) return nullptr;

	jlong contextHandle = env->GetLongField(modelObj, contextField);
	return reinterpret_cast<llama_context*>(contextHandle);
}

jlong BatchManager::initializeBatch(JNIEnv* env, jint tokenCount, jint embeddingSize, jint maxSequences) {
	llama_batch batch = llama_batch_init(tokenCount, embeddingSize, maxSequences);

	auto batchPtr = std::make_unique<llama_batch>(batch);

	std::lock_guard<std::mutex> lock(batchMutex);
	jlong batchId = nextBatchId++;
	batchRegistry[batchId] = std::move(batchPtr);

	return batchId;
}

void BatchManager::freeBatch(JNIEnv* env, jlong batchHandle) {
	std::lock_guard<std::mutex> lock(batchMutex);
	auto it = batchRegistry.find(batchHandle);
	if (it != batchRegistry.end()) {
		llama_batch_free(*it->second);
		batchRegistry.erase(it);
	}
}

jint BatchManager::encodeContext(JNIEnv* env, jobject modelObj, jlong batchHandle) {
	llama_context* ctx = getContext(env, modelObj);
	llama_batch* batch = getBatch(batchHandle);

	if (!ctx || !batch) {
		return -1;
	}

	return llama_encode(ctx, *batch);
}

jint BatchManager::decodeTokens(JNIEnv* env, jobject modelObj, jlong batchHandle) {
	llama_context* ctx = getContext(env, modelObj);
	llama_batch* batch = getBatch(batchHandle);

	if (!ctx || !batch) {
		return -1;
	}

	return llama_decode(ctx, *batch);
}

void BatchManager::setBatchTokens(JNIEnv* env, jlong batchHandle, jintArray tokens) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !tokens) return;

	jsize length = env->GetArrayLength(tokens);
	jint* tokenData = env->GetIntArrayElements(tokens, nullptr);

	if (tokenData) {
		batch->n_tokens = length;
		for (int i = 0; i < length; i++) {
			batch->token[i] = static_cast<llama_token>(tokenData[i]);
		}
		env->ReleaseIntArrayElements(tokens, tokenData, JNI_ABORT);
	}
}

void BatchManager::setBatchEmbeddings(JNIEnv* env, jlong batchHandle, jfloatArray embeddings) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !embeddings || !batch->embd) return;

	jsize length = env->GetArrayLength(embeddings);
	jfloat* embeddingData = env->GetFloatArrayElements(embeddings, nullptr);

	if (embeddingData) {
		for (int i = 0; i < length; i++) {
			batch->embd[i] = embeddingData[i];
		}
		env->ReleaseFloatArrayElements(embeddings, embeddingData, JNI_ABORT);
	}
}

void BatchManager::setBatchPositions(JNIEnv* env, jlong batchHandle, jintArray positions) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !positions) return;

	jsize length = env->GetArrayLength(positions);
	jint* positionData = env->GetIntArrayElements(positions, nullptr);

	if (positionData) {
		for (int i = 0; i < length && i < batch->n_tokens; i++) {
			batch->pos[i] = static_cast<llama_pos>(positionData[i]);
		}
		env->ReleaseIntArrayElements(positions, positionData, JNI_ABORT);
	}
}

void BatchManager::setBatchSequenceIds(JNIEnv* env, jlong batchHandle, jintArray sequenceIds) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !sequenceIds) return;

	jsize length = env->GetArrayLength(sequenceIds);
	jint* seqIdData = env->GetIntArrayElements(sequenceIds, nullptr);

	if (seqIdData) {
		for (int i = 0; i < length && i < batch->n_tokens; i++) {
			batch->n_seq_id[i] = 1;
			if (batch->seq_id && batch->seq_id[i]) {
				batch->seq_id[i][0] = static_cast<llama_seq_id>(seqIdData[i]);
			}
		}
		env->ReleaseIntArrayElements(sequenceIds, seqIdData, JNI_ABORT);
	}
}

void BatchManager::setBatchLogitFlags(JNIEnv* env, jlong batchHandle, jbyteArray logitFlags) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !logitFlags) return;

	jsize length = env->GetArrayLength(logitFlags);
	jbyte* flagData = env->GetByteArrayElements(logitFlags, nullptr);

	if (flagData) {
		for (int i = 0; i < length && i < batch->n_tokens; i++) {
			batch->logits[i] = static_cast<int8_t>(flagData[i]);
		}
		env->ReleaseByteArrayElements(logitFlags, flagData, JNI_ABORT);
	}
}

jintArray BatchManager::getBatchTokens(JNIEnv* env, jlong batchHandle) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !batch->token) return nullptr;

	jintArray result = env->NewIntArray(batch->n_tokens);
	if (result) {
		jint* resultData = new jint[batch->n_tokens];
		for (int i = 0; i < batch->n_tokens; i++) {
			resultData[i] = static_cast<jint>(batch->token[i]);
		}
		env->SetIntArrayRegion(result, 0, batch->n_tokens, resultData);
		delete[] resultData;
	}
	return result;
}

jfloatArray BatchManager::getBatchEmbeddings(JNIEnv* env, jlong batchHandle) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !batch->embd) return nullptr;

	// Note: We'd need to know the embedding dimension to return the correct size
	// For now, return nullptr to indicate embeddings are not available this way
	return nullptr;
}

jintArray BatchManager::getBatchPositions(JNIEnv* env, jlong batchHandle) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !batch->pos) return nullptr;

	jintArray result = env->NewIntArray(batch->n_tokens);
	if (result) {
		jint* resultData = new jint[batch->n_tokens];
		for (int i = 0; i < batch->n_tokens; i++) {
			resultData[i] = static_cast<jint>(batch->pos[i]);
		}
		env->SetIntArrayRegion(result, 0, batch->n_tokens, resultData);
		delete[] resultData;
	}
	return result;
}

jintArray BatchManager::getBatchSequenceIds(JNIEnv* env, jlong batchHandle) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !batch->seq_id) return nullptr;

	jintArray result = env->NewIntArray(batch->n_tokens);
	if (result) {
		jint* resultData = new jint[batch->n_tokens];
		for (int i = 0; i < batch->n_tokens; i++) {
			resultData[i] = (batch->seq_id[i] && batch->n_seq_id[i] > 0) ?
				static_cast<jint>(batch->seq_id[i][0]) : 0;
		}
		env->SetIntArrayRegion(result, 0, batch->n_tokens, resultData);
		delete[] resultData;
	}
	return result;
}

jbyteArray BatchManager::getBatchLogitFlags(JNIEnv* env, jlong batchHandle) {
	llama_batch* batch = getBatch(batchHandle);
	if (!batch || !batch->logits) return nullptr;

	jbyteArray result = env->NewByteArray(batch->n_tokens);
	if (result) {
		jbyte* resultData = new jbyte[batch->n_tokens];
		for (int i = 0; i < batch->n_tokens; i++) {
			resultData[i] = static_cast<jbyte>(batch->logits[i]);
		}
		env->SetByteArrayRegion(result, 0, batch->n_tokens, resultData);
		delete[] resultData;
	}
	return result;
}

jint BatchManager::getBatchTokenCount(JNIEnv* env, jlong batchHandle) {
	llama_batch* batch = getBatch(batchHandle);
	return batch ? static_cast<jint>(batch->n_tokens) : 0;
}