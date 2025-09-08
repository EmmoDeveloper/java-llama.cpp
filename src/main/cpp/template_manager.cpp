#include "template_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "llama_server.h"
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>

// These are defined in jllama.cpp but we need access to them
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_template_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jstring TemplateManager::applyTemplate(JNIEnv* env, jobject obj, jstring params) {
	JNI_TRY(env)
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_template_server(handle);
	if (!server) return nullptr;
	
	std::string param_str = JniUtils::jstring_to_string(env, params);
	
	// Parse JSON parameters to extract messages array
	std::vector<std::pair<std::string, std::string>> messages = parseMessages(param_str);
	
	// Get the model's chat template
	const char* tmpl = llama_model_chat_template(server->model, nullptr);
	if (!tmpl) {
		// Fallback to ChatML template if model doesn't have one
		tmpl = getDefaultChatMLTemplate();
	}
	
	// Create llama_chat_message array from parsed messages
	std::vector<llama_chat_message> chat_messages;
	
	// Add all messages from the JSON array
	// The messages vector already has stable string storage since it contains pairs of strings
	for (const auto& msg : messages) {
		llama_chat_message chat_msg;
		chat_msg.role = msg.first.c_str();
		chat_msg.content = msg.second.c_str();
		chat_messages.push_back(chat_msg);
	}
	
	// Apply the template
	std::string result_buffer;
	result_buffer.resize(8192); // Initial buffer size
	
	int32_t result_len = llama_chat_apply_template(
		tmpl,
		chat_messages.data(),
		chat_messages.size(),
		true,  // add_assistant (add generation prompt)
		&result_buffer[0],
		result_buffer.size()
	);
	
	if (result_len < 0) {
		// Buffer too small, resize and try again
		result_buffer.resize(-result_len);
		result_len = llama_chat_apply_template(
			tmpl,
			chat_messages.data(),
			chat_messages.size(),
			true,
			&result_buffer[0],
			result_buffer.size()
		);
	}
	
	if (result_len < 0) {
		JNIErrorHandler::throw_runtime_exception(env,
			"Failed to apply chat template");
		return nullptr;
	}
	
	result_buffer.resize(result_len);
	return JniUtils::string_to_jstring(env, result_buffer);
	
	JNI_CATCH_RET(env, nullptr)
}

// Helper functions
std::vector<std::pair<std::string, std::string>> TemplateManager::parseMessages(const std::string& param_str) {
	std::vector<std::pair<std::string, std::string>> messages;
	
	// Parse messages array - look for "messages" field
	size_t pos = param_str.find("\"messages\":");
	if (pos != std::string::npos) {
		// Find the opening bracket of the array
		size_t array_start = param_str.find('[', pos);
		if (array_start != std::string::npos) {
			size_t array_end = array_start + 1;
			int bracket_count = 1;
			
			// Find matching closing bracket
			while (array_end < param_str.length() && bracket_count > 0) {
				if (param_str[array_end] == '[') bracket_count++;
				else if (param_str[array_end] == ']') bracket_count--;
				array_end++;
			}
			
			// Parse messages within the array
			std::string messages_str = param_str.substr(array_start + 1, array_end - array_start - 2);
			
			// Simple parsing of role:content pairs
			size_t msg_pos = 0;
			while ((msg_pos = messages_str.find("\"role\":", msg_pos)) != std::string::npos) {
				// Extract role
				size_t role_start = messages_str.find('"', msg_pos + 7);
				if (role_start == std::string::npos) break;
				role_start++;
				size_t role_end = messages_str.find('"', role_start);
				if (role_end == std::string::npos) break;
				std::string role = messages_str.substr(role_start, role_end - role_start);
				
				// Extract content
				size_t content_pos = messages_str.find("\"content\":", role_end);
				if (content_pos == std::string::npos) break;
				size_t content_start = messages_str.find('"', content_pos + 10);
				if (content_start == std::string::npos) break;
				content_start++;
				size_t content_end = content_start;
				while (content_end < messages_str.length()) {
					if (messages_str[content_end] == '"' && (content_end == content_start || messages_str[content_end - 1] != '\\')) {
						break;
					}
					content_end++;
				}
				if (content_end >= messages_str.length()) break;
				std::string content = messages_str.substr(content_start, content_end - content_start);
				
				messages.emplace_back(role, content);
				msg_pos = content_end;
			}
		}
	}
	
	return messages;
}

const char* TemplateManager::getDefaultChatMLTemplate() {
	return "{% for message in messages %}"
		   "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
		   "{% endfor %}"
		   "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}";
}