#include "threading_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"

void ThreadingManager::setModelThreading(JNIEnv* env, jobject model, jint generationThreads, jint batchThreads) {
	JNI_TRY(env)
	
	struct llama_context* ctx = getContextFromModel(env, model);
	if (!ctx) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get context from model");
		return;
	}
	
	// Validate thread counts
	if (generationThreads < 1) {
		JNIErrorHandler::throw_illegal_argument(env, "Generation threads must be at least 1");
		return;
	}
	if (batchThreads < 1) {
		JNIErrorHandler::throw_illegal_argument(env, "Batch threads must be at least 1");
		return;
	}
	
	// Apply the threading configuration using llama.cpp's native function
	llama_set_n_threads(ctx, static_cast<int32_t>(generationThreads), static_cast<int32_t>(batchThreads));
	
	JNI_CATCH(env)
}

jintArray ThreadingManager::getModelThreading(JNIEnv* env, jobject model) {
	JNI_TRY(env)
	
	struct llama_context* ctx = getContextFromModel(env, model);
	if (!ctx) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get context from model");
		return nullptr;
	}
	
	// Get current thread counts using llama.cpp's native functions
	int32_t generationThreads = llama_n_threads(ctx);
	int32_t batchThreads = llama_n_threads_batch(ctx);
	
	// Create Java int array to return both values
	jintArray result = env->NewIntArray(2);
	if (!result) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create result array");
		return nullptr;
	}
	
	jint threads[2] = {static_cast<jint>(generationThreads), static_cast<jint>(batchThreads)};
	env->SetIntArrayRegion(result, 0, 2, threads);
	
	return result;
	
	JNI_CATCH_RET(env, nullptr)
}

struct llama_context* ThreadingManager::getContextFromModel(JNIEnv* env, jobject model) {
	if (!model) {
		return nullptr;
	}
	
	// Get the context handle from the Java model object
	jclass modelClass = env->GetObjectClass(model);
	jfieldID ctxField = env->GetFieldID(modelClass, "ctx", "J");
	if (!ctxField) {
		return nullptr;
	}
	
	jlong ctxHandle = env->GetLongField(model, ctxField);
	return reinterpret_cast<struct llama_context*>(ctxHandle);
}