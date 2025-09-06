#pragma once

#include <jni.h>
#include <string>
#include "llama.h"
#include "llama_server.h"

/**
 * Handles LoRA adapter operations for llama.cpp contexts.
 * Provides load/apply/remove functionality for LoRA adapters and control vectors.
 */
class LoRAAdapterManager {
public:
	/**
	 * Load a LoRA adapter from file.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param path_lora Path to the LoRA adapter file
	 * @return Java long handle to the adapter, or -1 on error
	 */
	static jlong loadAdapter(JNIEnv* env, jobject obj, jstring path_lora);

	/**
	 * Free a LoRA adapter.
	 * @param env JNI environment
	 * @param adapter_handle Handle to the adapter
	 */
	static void freeAdapter(JNIEnv* env, jlong adapter_handle);

	/**
	 * Apply a LoRA adapter to the context with given scale.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param adapter_handle Handle to the adapter
	 * @param scale Scale factor for the adapter (typically 1.0)
	 * @return 0 on success, negative on error
	 */
	static jint setAdapter(JNIEnv* env, jobject obj, jlong adapter_handle, jfloat scale);

	/**
	 * Remove a LoRA adapter from the context.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param adapter_handle Handle to the adapter
	 * @return 0 on success, negative on error
	 */
	static jint removeAdapter(JNIEnv* env, jobject obj, jlong adapter_handle);

	/**
	 * Clear all LoRA adapters from the context.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 */
	static void clearAdapters(JNIEnv* env, jobject obj);

	/**
	 * Apply a control vector to the context.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param data Float array containing control vector data (null to clear)
	 * @return 0 on success, negative on error
	 */
	static jint applyControlVector(JNIEnv* env, jobject obj, jfloatArray data);

	/**
	 * Get metadata value from adapter as string.
	 * @param env JNI environment
	 * @param adapter_handle Handle to the adapter
	 * @param key Metadata key to retrieve
	 * @return Java string with metadata value, or null if not found
	 */
	static jstring getAdapterMetaValue(JNIEnv* env, jlong adapter_handle, jstring key);

	/**
	 * Get number of metadata entries in adapter.
	 * @param env JNI environment
	 * @param adapter_handle Handle to the adapter
	 * @return Number of metadata entries, or -1 on error
	 */
	static jint getAdapterMetaCount(JNIEnv* env, jlong adapter_handle);

	/**
	 * Get metadata key by index.
	 * @param env JNI environment
	 * @param adapter_handle Handle to the adapter
	 * @param index Index of the metadata entry
	 * @return Java string with metadata key, or null if not found
	 */
	static jstring getAdapterMetaKeyByIndex(JNIEnv* env, jlong adapter_handle, jint index);

	/**
	 * Get metadata value by index.
	 * @param env JNI environment
	 * @param adapter_handle Handle to the adapter
	 * @param index Index of the metadata entry
	 * @return Java string with metadata value, or null if not found
	 */
	static jstring getAdapterMetaValueByIndex(JNIEnv* env, jlong adapter_handle, jint index);

	/**
	 * Check if adapter is an ALORA and get invocation token count.
	 * @param env JNI environment
	 * @param adapter_handle Handle to the adapter
	 * @return Number of invocation tokens, or 0 if not ALORA
	 */
	static jlong getAloraInvocationTokenCount(JNIEnv* env, jlong adapter_handle);

	/**
	 * Get ALORA invocation tokens.
	 * @param env JNI environment
	 * @param adapter_handle Handle to the adapter
	 * @return Java int array of invocation tokens, or null if not ALORA
	 */
	static jintArray getAloraInvocationTokens(JNIEnv* env, jlong adapter_handle);

private:
	/**
	 * Get server handle from Java object.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @return LlamaServer pointer, or nullptr if not found
	 */
	static LlamaServer* getServer(JNIEnv* env, jobject obj);

	/**
	 * Convert adapter handle to pointer.
	 * @param handle Java long handle
	 * @return llama_adapter_lora pointer, or nullptr if invalid
	 */
	static llama_adapter_lora* getAdapter(jlong handle);
};