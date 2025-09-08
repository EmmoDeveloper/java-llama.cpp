#include "completion_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "llama_server.h"
#include "pattern_preprocessor.h"
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>

// These are defined in jllama.cpp but we need access to them
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_completion_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

jint CompletionManager::requestCompletion(JNIEnv* env, jobject obj, jstring params) {
	JNI_TRY(env)
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_completion_server(handle);
	if (!server) return -1;
	
	std::string param_str = JniUtils::jstring_to_string(env, params);
	JNI_LOG_DEBUG("requestCompletion params: %s", param_str.c_str());
	
	// Parse JSON parameters
	int n_predict = parseNPredict(param_str);
	std::string prompt = parsePrompt(param_str);
	std::string grammar = parseGrammar(param_str);
	
	if (prompt.empty()) {
		prompt = "Hello";  // Fallback prompt
	}
	
	// Create task synchronously (with grammar if provided)
	auto task = std::make_unique<CompletionTask>(server->next_task_id++, prompt, n_predict, grammar);
	
	// Tokenize the actual prompt
	const llama_vocab* vocab = llama_model_get_vocab(server->model);
	std::vector<llama_token> tokens;
	tokens.resize(prompt.length() + 1);
	
	int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), 
								  tokens.data(), tokens.size(), true, false);
	
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
								  tokens.data(), tokens.size(), true, false);
	}
	
	if (n_tokens < 0) return -1;
	
	tokens.resize(n_tokens);
	task->prompt_tokens = tokens;
	task->state = TASK_STATE_PROCESSING_PROMPT;
	
	// Clear sequence 0 to ensure a clean state for this new request
	llama_memory_t memory = llama_get_memory(server->ctx);
	llama_memory_seq_rm(memory, 0, -1, -1);
	
	// Process prompt synchronously
	llama_batch batch = llama_batch_init(n_tokens, 0, 1);
	for (int i = 0; i < n_tokens; i++) {
		batch.token[i] = tokens[i];
		batch.pos[i] = i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = 0;
		batch.logits[i] = (i == n_tokens - 1); // Only compute logits for last token
	}
	batch.n_tokens = n_tokens;
	
	if (llama_decode(server->ctx, batch) != 0) {
		llama_batch_free(batch);
		return -1;
	}
	
	llama_batch_free(batch);
	task->current_pos = n_tokens;
	task->state = TASK_STATE_GENERATING;
	
	// Create grammar sampler if grammar is provided
	if (!grammar.empty()) {
		JNI_LOG_DEBUG("Creating grammar sampler with original grammar: '%s'", grammar.c_str());
		
		// Preprocess the regex pattern for llama.cpp's constraint system
		std::string processed_pattern = PatternPreprocessor::preprocess(grammar);
		JNI_LOG_DEBUG("Adapted pattern: '%s'", processed_pattern.c_str());
		
		// Create a grammar sampler
		const llama_vocab* vocab = llama_model_get_vocab(server->model);
		llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, processed_pattern.c_str(), "root");
		
		if (grammar_sampler) {
			// Create a sampler chain that includes the grammar
			llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
			llama_sampler* chain = llama_sampler_chain_init(chain_params);
			
			// Add grammar sampler first to constrain the output
			llama_sampler_chain_add(chain, grammar_sampler);
			// Then add greedy sampling
			llama_sampler_chain_add(chain, llama_sampler_init_greedy());
			
			task->task_sampler = chain;
			JNI_LOG_DEBUG("Grammar sampler created successfully");
		} else {
			JNI_LOG_ERROR("Failed to create grammar sampler - this is a hard error like in original llama.cpp");
			// Match original llama.cpp behavior: if grammar fails to parse, the entire request fails
			return -1;  // Return error to indicate grammar parsing failure
		}
	}
	
	int task_id = task->id;
	{
		std::lock_guard<std::mutex> tasks_lock(server->active_tasks_mutex);
		server->active_tasks[task_id] = std::move(task);
	}
	
	JNI_LOG_DEBUG("requestCompletion created task with id %d, prompt: '%s', grammar: '%s'", 
		   task_id, prompt.c_str(), grammar.c_str());
	return task_id;
	
	JNI_CATCH_RET(env, -1)
}

jobject CompletionManager::receiveCompletion(JNIEnv* env, jobject obj, jint id) {
	JNI_TRY(env)
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_completion_server(handle);
	if (!server) {
		JNI_LOG_DEBUG("receiveCompletion server is null for id %d", id);
		return nullptr;
	}
	
	// Find the task
	std::lock_guard<std::mutex> tasks_lock(server->active_tasks_mutex);
	auto task_it = server->active_tasks.find(id);
	if (task_it == server->active_tasks.end()) {
		JNI_LOG_DEBUG("receiveCompletion task not found for id %d", id);
		return nullptr;
	}
	
	CompletionTask* task = task_it->second.get();
	if (task->cancelled) return nullptr;
	
	// Check if we've reached the generation limit BEFORE generating a new token
	if (task->generated_tokens.size() >= task->n_predict) {
		// Return final result - we've reached the limit
		jclass output_class = env->FindClass("de/kherud/llama/LlamaOutput");
		if (!output_class) return nullptr;
		
		jmethodID constructor = env->GetMethodID(output_class, "<init>", "([BLjava/util/Map;Z)V");
		if (!constructor) return nullptr;
		
		// Return empty byte array to signal completion
		jbyteArray byte_array = env->NewByteArray(0);
		
		jclass hashmap_class = env->FindClass("java/util/HashMap");
		jmethodID hashmap_init = env->GetMethodID(hashmap_class, "<init>", "()V");
		jobject probabilities = env->NewObject(hashmap_class, hashmap_init);
		
		return env->NewObject(output_class, constructor, byte_array, probabilities, (jboolean)true);
	}
	
	// Generate one token synchronously (use task-specific sampler if available, e.g., for grammar)
	llama_sampler* sampler_to_use = task->task_sampler ? task->task_sampler : server->sampler;
	llama_token new_token = llama_sampler_sample(sampler_to_use, server->ctx, -1);
	
	// Accept the token to update grammar state if using task-specific sampler
	if (task->task_sampler) {
		llama_sampler_accept(task->task_sampler, new_token);
	}
	
	// Check for end of generation
	const llama_vocab* vocab = llama_model_get_vocab(server->model);
	if (llama_vocab_is_eog(vocab, new_token)) {
		jclass output_class = env->FindClass("de/kherud/llama/LlamaOutput");
		if (!output_class) return nullptr;
		
		jmethodID constructor = env->GetMethodID(output_class, "<init>", "([BLjava/util/Map;Z)V");
		if (!constructor) return nullptr;
		
		jbyteArray byte_array = env->NewByteArray(task->current_text.length());
		env->SetByteArrayRegion(byte_array, 0, task->current_text.length(), 
							   (jbyte*)task->current_text.data());
		
		jclass hashmap_class = env->FindClass("java/util/HashMap");
		jmethodID hashmap_init = env->GetMethodID(hashmap_class, "<init>", "()V");
		jobject probabilities = env->NewObject(hashmap_class, hashmap_init);
		
		return env->NewObject(output_class, constructor, byte_array, probabilities, (jboolean)true);
	}
	
	// Add token to generation
	task->generated_tokens.push_back(new_token);
	
	// Convert token to text
	char piece[256];
	int piece_len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
	if (piece_len > 0) {
		task->current_text.append(piece, piece_len);
		if (task->task_sampler) {
			JNI_LOG_DEBUG("Grammar generated token: %d -> '%.*s', total text: '%s'", 
				   new_token, piece_len, piece, task->current_text.c_str());
		}
	}
	
	// Process next token for context
	llama_batch batch = llama_batch_init(1, 0, 1);
	batch.token[0] = new_token;
	batch.pos[0] = task->current_pos++;
	batch.n_seq_id[0] = 1;
	batch.seq_id[0][0] = 0;
	batch.logits[0] = true;
	batch.n_tokens = 1;
	
	if (llama_decode(server->ctx, batch) != 0) {
		llama_batch_free(batch);
		return nullptr;
	}
	
	llama_batch_free(batch);
	
	// Return partial result (just the new token's text, not cumulative)
	jclass output_class = env->FindClass("de/kherud/llama/LlamaOutput");
	if (!output_class) return nullptr;
	
	jmethodID constructor = env->GetMethodID(output_class, "<init>", "([BLjava/util/Map;Z)V");
	if (!constructor) return nullptr;
	
	// Only return the new token's text piece, not the cumulative text
	jbyteArray byte_array = env->NewByteArray(piece_len);
	env->SetByteArrayRegion(byte_array, 0, piece_len, (jbyte*)piece);
	
	jclass hashmap_class = env->FindClass("java/util/HashMap");
	jmethodID hashmap_init = env->GetMethodID(hashmap_class, "<init>", "()V");
	jobject probabilities = env->NewObject(hashmap_class, hashmap_init);
	
	return env->NewObject(output_class, constructor, byte_array, probabilities, (jboolean)false);
	
	JNI_CATCH_RET(env, nullptr)
}

void CompletionManager::cancelCompletion(JNIEnv* env, jobject obj, jint id) {
	JNI_TRY(env)
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_completion_server(handle);
	if (!server) return;
	
	std::lock_guard<std::mutex> tasks_lock(server->active_tasks_mutex);
	auto task_it = server->active_tasks.find(id);
	if (task_it != server->active_tasks.end()) {
		task_it->second->cancelled = true;
	}
	
	JNI_CATCH(env)
}

void CompletionManager::releaseTask(JNIEnv* env, jobject obj, jint id) {
	JNI_TRY(env)
	
	jlong handle = env->GetLongField(obj, 
		env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
	LlamaServer* server = get_completion_server(handle);
	if (!server) return;
	
	// Clean up task
	{
		std::lock_guard<std::mutex> tasks_lock(server->active_tasks_mutex);
		server->active_tasks.erase(id);
	}
	{
		std::lock_guard<std::mutex> result_lock(server->result_mutex);
		server->task_results.erase(id);
	}
	
	JNI_CATCH(env)
}

// Helper functions for JSON parsing
int CompletionManager::parseNPredict(const std::string& json) {
	int n_predict = 10;  // Default value
	
	// Simple JSON parsing for n_predict - look for "n_predict":value pattern
	size_t pos = json.find("\"n_predict\":");
	if (pos != std::string::npos) {
		size_t start = json.find_first_of("0123456789", pos);
		if (start != std::string::npos) {
			size_t end = json.find_first_not_of("0123456789", start);
			if (end != std::string::npos) {
				std::string num_str = json.substr(start, end - start);
				n_predict = std::stoi(num_str);
			}
		}
	}
	
	return n_predict;
}

std::string CompletionManager::parsePrompt(const std::string& json) {
	std::string prompt;
	
	// Parse prompt - look for "prompt":"value" pattern
	size_t pos = json.find("\"prompt\":");
	if (pos != std::string::npos) {
		size_t start = json.find('"', pos + 9);  // Find opening quote after "prompt":
		if (start != std::string::npos) {
			start++; // Move past opening quote
			size_t end = start;
			// Find closing quote, handling escaped quotes
			while (end < json.length()) {
				if (json[end] == '"' && (end == start || json[end - 1] != '\\')) {
					break;
				}
				end++;
			}
			if (end < json.length()) {
				prompt = json.substr(start, end - start);
			}
		}
	}
	
	return prompt;
}

std::string CompletionManager::parseGrammar(const std::string& json) {
	std::string grammar;
	
	// Parse grammar - look for "grammar":"value" pattern
	size_t pos = json.find("\"grammar\":");
	if (pos != std::string::npos) {
		size_t start = json.find('"', pos + 10);  // Find opening quote after "grammar":
		if (start != std::string::npos) {
			start++; // Move past opening quote
			size_t end = start;
			// Find closing quote, handling escaped quotes and escaped backslashes
			while (end < json.length()) {
				if (json[end] == '"') {
					// Count preceding backslashes
					int backslash_count = 0;
					int check_pos = end - 1;
					while (check_pos >= (int)start && json[check_pos] == '\\') {
						backslash_count++;
						check_pos--;
					}
					// If even number of backslashes (or zero), the quote is not escaped
					if (backslash_count % 2 == 0) {
						break;
					}
				}
				end++;
			}
			if (end < json.length()) {
				grammar = json.substr(start, end - start);
				grammar = unescapeString(grammar);
			}
		}
	}
	
	return grammar;
}

std::string CompletionManager::unescapeString(const std::string& str) {
	std::string unescaped;
	for (size_t i = 0; i < str.length(); i++) {
		if (str[i] == '\\' && i + 1 < str.length()) {
			if (str[i + 1] == '"') {
				unescaped += '"';
				i++;
			} else if (str[i + 1] == '\\') {
				unescaped += '\\';
				i++;
			} else {
				unescaped += str[i];
			}
		} else {
			unescaped += str[i];
		}
	}
	return unescaped;
}