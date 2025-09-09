#include "model_info_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include <string>

// Model information functions

jlong ModelInfoManager::getModelParameterCount(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_model* model = getModel(env, obj);
	validateModel(env, model);

	uint64_t params = llama_model_n_params(model);
	return static_cast<jlong>(params);

	JNI_CATCH_RET(env, -1L)
}

jlong ModelInfoManager::getModelSize(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_model* model = getModel(env, obj);
	validateModel(env, model);

	uint64_t size = llama_model_size(model);
	return static_cast<jlong>(size);

	JNI_CATCH_RET(env, -1L)
}

jint ModelInfoManager::getModelMetadataCount(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_model* model = getModel(env, obj);
	validateModel(env, model);

	int32_t count = llama_model_meta_count(model);
	return static_cast<jint>(count);

	JNI_CATCH_RET(env, -1)
}

jstring ModelInfoManager::getModelMetadataKeyByIndex(JNIEnv* env, jobject obj, jint index) {
	JNI_TRY(env)

	const struct llama_model* model = getModel(env, obj);
	validateModel(env, model);

	if (index < 0) {
		JNIErrorHandler::throw_illegal_argument(env, "Metadata index must be non-negative");
		return nullptr;
	}

	// First get the required buffer size
	int32_t required_size = llama_model_meta_key_by_index(model, index, nullptr, 0);
	if (required_size <= 0) {
		return env->NewStringUTF("");  // Return empty string if no key found
	}

	// Allocate buffer and get the key
	std::vector<char> buffer(required_size);
	int32_t actual_size = llama_model_meta_key_by_index(model, index, buffer.data(), buffer.size());

	if (actual_size <= 0) {
		return env->NewStringUTF("");
	}

	return env->NewStringUTF(buffer.data());

	JNI_CATCH_RET(env, nullptr)
}

jstring ModelInfoManager::getModelMetadataValueByIndex(JNIEnv* env, jobject obj, jint index) {
	JNI_TRY(env)

	const struct llama_model* model = getModel(env, obj);
	validateModel(env, model);

	if (index < 0) {
		JNIErrorHandler::throw_illegal_argument(env, "Metadata index must be non-negative");
		return nullptr;
	}

	// First get the required buffer size
	int32_t required_size = llama_model_meta_val_str_by_index(model, index, nullptr, 0);
	if (required_size <= 0) {
		return env->NewStringUTF("");  // Return empty string if no value found
	}

	// Allocate buffer and get the value
	std::vector<char> buffer(required_size);
	int32_t actual_size = llama_model_meta_val_str_by_index(model, index, buffer.data(), buffer.size());

	if (actual_size <= 0) {
		return env->NewStringUTF("");
	}

	return env->NewStringUTF(buffer.data());

	JNI_CATCH_RET(env, nullptr)
}

jstring ModelInfoManager::getModelMetadataValue(JNIEnv* env, jobject obj, jstring key) {
	JNI_TRY(env)

	const struct llama_model* model = getModel(env, obj);
	validateModel(env, model);

	if (!key) {
		JNIErrorHandler::throw_illegal_argument(env, "Metadata key cannot be null");
		return nullptr;
	}

	const char* keyStr = env->GetStringUTFChars(key, nullptr);
	if (!keyStr) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get key string");
		return nullptr;
	}

	// First get the required buffer size
	int32_t required_size = llama_model_meta_val_str(model, keyStr, nullptr, 0);
	if (required_size <= 0) {
		env->ReleaseStringUTFChars(key, keyStr);
		return env->NewStringUTF("");  // Return empty string if no value found
	}

	// Allocate buffer and get the value
	std::vector<char> buffer(required_size);
	int32_t actual_size = llama_model_meta_val_str(model, keyStr, buffer.data(), buffer.size());

	env->ReleaseStringUTFChars(key, keyStr);

	if (actual_size <= 0) {
		return env->NewStringUTF("");
	}

	return env->NewStringUTF(buffer.data());

	JNI_CATCH_RET(env, nullptr)
}

// Vocabulary information functions

jint ModelInfoManager::getVocabularyType(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);

	enum llama_vocab_type type = llama_vocab_type(vocab);
	return static_cast<jint>(type);

	JNI_CATCH_RET(env, -1)
}

jint ModelInfoManager::getVocabularySize(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);

	int32_t size = llama_vocab_n_tokens(vocab);
	return static_cast<jint>(size);

	JNI_CATCH_RET(env, -1)
}

jstring ModelInfoManager::getTokenText(JNIEnv* env, jobject obj, jint token) {
	JNI_TRY(env)
	
	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);
	if (env->ExceptionCheck()) {
		return nullptr;
	}
	
	validateToken(env, vocab, token);
	if (env->ExceptionCheck()) {
		return nullptr;
	}
	
	const char* text = llama_vocab_get_text(vocab, static_cast<llama_token>(token));
	if (!text) {
		return env->NewStringUTF("");
	}
	
	return env->NewStringUTF(text);
	
	JNI_CATCH_RET(env, nullptr)
}

jfloat ModelInfoManager::getTokenScore(JNIEnv* env, jobject obj, jint token) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);
	if (env->ExceptionCheck()) {
		return 0.0f;
	}

	validateToken(env, vocab, token);
	if (env->ExceptionCheck()) {
		return 0.0f;
	}

	float score = llama_vocab_get_score(vocab, static_cast<llama_token>(token));
	return static_cast<jfloat>(score);

	JNI_CATCH_RET(env, 0.0f)
}

jint ModelInfoManager::getTokenAttributes(JNIEnv* env, jobject obj, jint token) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);
	if (env->ExceptionCheck()) {
		return 0;
	}

	validateToken(env, vocab, token);
	if (env->ExceptionCheck()) {
		return 0;
	}

	enum llama_token_attr attr = llama_vocab_get_attr(vocab, static_cast<llama_token>(token));
	return static_cast<jint>(attr);

	JNI_CATCH_RET(env, 0)
}

// Special token functions

jint ModelInfoManager::getBosToken(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	if (env->ExceptionCheck()) {
		return -1;
	}
	validateVocab(env, vocab);
	if (env->ExceptionCheck()) {
		return -1;
	}

	llama_token token = llama_vocab_bos(vocab);
	return static_cast<jint>(token);

	JNI_CATCH_RET(env, -1)
}

jint ModelInfoManager::getEosToken(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);

	llama_token token = llama_vocab_eos(vocab);
	return static_cast<jint>(token);

	JNI_CATCH_RET(env, -1)
}

jint ModelInfoManager::getEotToken(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);

	llama_token token = llama_vocab_eot(vocab);
	return static_cast<jint>(token);

	JNI_CATCH_RET(env, -1)
}

jint ModelInfoManager::getSepToken(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);

	llama_token token = llama_vocab_sep(vocab);
	return static_cast<jint>(token);

	JNI_CATCH_RET(env, -1)
}

jint ModelInfoManager::getNlToken(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);

	llama_token token = llama_vocab_nl(vocab);
	return static_cast<jint>(token);

	JNI_CATCH_RET(env, -1)
}

jint ModelInfoManager::getPadToken(JNIEnv* env, jobject obj) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);

	llama_token token = llama_vocab_pad(vocab);
	return static_cast<jint>(token);

	JNI_CATCH_RET(env, -1)
}

// Token checking functions

jboolean ModelInfoManager::isEogToken(JNIEnv* env, jobject obj, jint token) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);
	if (env->ExceptionCheck()) {
		return JNI_FALSE;
	}

	validateToken(env, vocab, token);
	if (env->ExceptionCheck()) {
		return JNI_FALSE;
	}

	bool isEog = llama_vocab_is_eog(vocab, static_cast<llama_token>(token));
	return isEog ? JNI_TRUE : JNI_FALSE;

	JNI_CATCH_RET(env, JNI_FALSE)
}

jboolean ModelInfoManager::isControlToken(JNIEnv* env, jobject obj, jint token) {
	JNI_TRY(env)

	const struct llama_vocab* vocab = getVocab(env, obj);
	validateVocab(env, vocab);
	if (env->ExceptionCheck()) {
		return JNI_FALSE;
	}

	validateToken(env, vocab, token);
	if (env->ExceptionCheck()) {
		return JNI_FALSE;
	}

	bool isControl = llama_vocab_is_control(vocab, static_cast<llama_token>(token));
	return isControl ? JNI_TRUE : JNI_FALSE;

	JNI_CATCH_RET(env, JNI_FALSE)
}

// Helper methods

const struct llama_model* ModelInfoManager::getModel(JNIEnv* env, jobject obj) {
	jclass cls = env->GetObjectClass(obj);
	jfieldID fieldId = env->GetFieldID(cls, "ctx", "J");
	if (!fieldId) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get context field");
		return nullptr;
	}

	jlong ctxHandle = env->GetLongField(obj, fieldId);
	struct llama_context* ctx = reinterpret_cast<struct llama_context*>(ctxHandle);
	if (!ctx) {
		JNIErrorHandler::throw_runtime_exception(env, "Context is null - model not properly loaded");
		return nullptr;
	}

	const struct llama_model* model = llama_get_model(ctx);
	if (!model) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get model from context");
		return nullptr;
	}

	return model;
}

const struct llama_vocab* ModelInfoManager::getVocab(JNIEnv* env, jobject obj) {
	const struct llama_model* model = getModel(env, obj);
	if (!model) {
		return nullptr;
	}
	return llama_model_get_vocab(model);
}

void ModelInfoManager::validateModel(JNIEnv* env, const struct llama_model* model) {
	if (!model) {
		JNIErrorHandler::throw_runtime_exception(env, "Model is null");
	}
}

void ModelInfoManager::validateVocab(JNIEnv* env, const struct llama_vocab* vocab) {
	if (!vocab) {
		JNIErrorHandler::throw_runtime_exception(env, "Vocabulary is null");
	}
}

void ModelInfoManager::validateToken(JNIEnv* env, const struct llama_vocab* vocab, jint token) {
	if (token < 0) {
		JNIErrorHandler::throw_illegal_argument(env, "Token ID must be non-negative");
		return;
	}

	int32_t vocab_size = llama_vocab_n_tokens(vocab);
	if (token >= vocab_size) {
		JNIErrorHandler::throw_illegal_argument(env, "Token ID exceeds vocabulary size");
	}
}
