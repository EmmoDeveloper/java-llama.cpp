#ifndef TEMPLATE_MANAGER_H
#define TEMPLATE_MANAGER_H

#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"

class TemplateManager {
public:
	// Apply chat template to messages
	static jstring applyTemplate(JNIEnv* env, jobject obj, jstring params);

private:
	// Helper functions for message parsing
	static std::vector<std::pair<std::string, std::string>> parseMessages(const std::string& json);
	static const char* getDefaultChatMLTemplate();
};

#endif // TEMPLATE_MANAGER_H