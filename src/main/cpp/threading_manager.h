#ifndef THREADING_MANAGER_H
#define THREADING_MANAGER_H

#include <jni.h>
#include "llama.h"

/**
 * Threading Manager for fine-grained thread control in llama.cpp contexts.
 * Provides Java access to llama.cpp's thread management functions.
 */
class ThreadingManager {
public:
	// Apply threading configuration to a model context
	static void setModelThreading(JNIEnv* env, jobject model, jint generationThreads, jint batchThreads);
	
	// Get current threading configuration from a model context
	static jintArray getModelThreading(JNIEnv* env, jobject model);
	
private:
	// Helper method to get context from Java model object
	static struct llama_context* getContextFromModel(JNIEnv* env, jobject model);
};

#endif // THREADING_MANAGER_H