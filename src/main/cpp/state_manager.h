#pragma once

#include <jni.h>
#include <string>
#include "llama.h"
#include "llama_server.h"

/**
 * Handles state persistence operations for llama.cpp contexts.
 * Provides save/load functionality for conversation state and KV cache.
 */
class StateManager {
public:
	/**
	 * Get the size needed to store the complete context state.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @return State size in bytes, or -1 on error
	 */
	static jlong getStateSize(JNIEnv* env, jobject obj);

	/**
	 * Get the context state as a byte array.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @return Java byte array containing state data, or nullptr on error
	 */
	static jbyteArray getStateData(JNIEnv* env, jobject obj);

	/**
	 * Set the context state from byte array data.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param state_data Byte array containing state data
	 * @return Number of bytes loaded, or -1 on error
	 */
	static jlong setStateData(JNIEnv* env, jobject obj, jbyteArray state_data);

	/**
	 * Save context state and tokens to file.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param path File path to save to
	 * @param tokens Token array to save with state
	 * @return true on success, false on error
	 */
	static jboolean saveStateToFile(JNIEnv* env, jobject obj, jstring path, jintArray tokens);

	/**
	 * Load context state and tokens from file.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param path File path to load from
	 * @param max_tokens Maximum number of tokens to load
	 * @return Java int array of loaded tokens, or nullptr on error
	 */
	static jintArray loadStateFromFile(JNIEnv* env, jobject obj, jstring path, jint max_tokens);

	/**
	 * Get the size needed to store a specific sequence state.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param seq_id Sequence ID
	 * @return Sequence state size in bytes, or -1 on error
	 */
	static jlong getSequenceStateSize(JNIEnv* env, jobject obj, jint seq_id);

	/**
	 * Get a specific sequence state as byte array.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param seq_id Sequence ID
	 * @return Java byte array containing sequence state, or nullptr on error
	 */
	static jbyteArray getSequenceStateData(JNIEnv* env, jobject obj, jint seq_id);

	/**
	 * Set a specific sequence state from byte array.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param state_data Byte array containing sequence state
	 * @param seq_id Target sequence ID
	 * @return Number of bytes loaded, or -1 on error
	 */
	static jlong setSequenceStateData(JNIEnv* env, jobject obj, jbyteArray state_data, jint seq_id);

	/**
	 * Save specific sequence state to file.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param path File path to save to
	 * @param seq_id Sequence ID to save
	 * @param tokens Token array for this sequence
	 * @return Number of bytes saved, or -1 on error
	 */
	static jlong saveSequenceToFile(JNIEnv* env, jobject obj, jstring path, jint seq_id, jintArray tokens);

	/**
	 * Load specific sequence state from file.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @param path File path to load from
	 * @param seq_id Target sequence ID
	 * @param max_tokens Maximum number of tokens to load
	 * @return Java int array of loaded tokens, or nullptr on error
	 */
	static jintArray loadSequenceFromFile(JNIEnv* env, jobject obj, jstring path, jint seq_id, jint max_tokens);

private:
	/**
	 * Get server handle from Java object.
	 * @param env JNI environment
	 * @param obj Java LlamaModel object
	 * @return LlamaServer pointer, or nullptr if not found
	 */
	static LlamaServer* getServer(JNIEnv* env, jobject obj);
};