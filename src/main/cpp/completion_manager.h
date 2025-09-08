#ifndef COMPLETION_MANAGER_H
#define COMPLETION_MANAGER_H

#include <jni.h>
#include <string>
#include "llama.h"

class CompletionManager {
public:
	// Request a new completion
	static jint requestCompletion(JNIEnv* env, jobject obj, jstring params);
	
	// Receive completion results
	static jobject receiveCompletion(JNIEnv* env, jobject obj, jint id);
	
	// Cancel a completion
	static void cancelCompletion(JNIEnv* env, jobject obj, jint id);
	
	// Release a task
	static void releaseTask(JNIEnv* env, jobject obj, jint id);

private:
	// Helper functions for JSON parsing
	static int parseNPredict(const std::string& json);
	static std::string parsePrompt(const std::string& json);
	static std::string parseGrammar(const std::string& json);
	static std::string unescapeString(const std::string& str);
};

#endif // COMPLETION_MANAGER_H