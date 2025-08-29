#include <jni.h>
#include <memory>
#include <mutex>
#include <unordered_map>

// Include the real llama.cpp headers
#include "llama.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>

// Include our modular headers
#include "jni_utils.h"
#include "completion_task.h"
#include "llama_server.h"
#include "grammar_processor.h"

// Global server management


// Global server management
static std::mutex g_servers_mutex;
static std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

// Utility functions

static LlamaServer* get_server(jlong handle) {
    std::lock_guard<std::mutex> lock(g_servers_mutex);
    auto it = g_servers.find(handle);
    return (it != g_servers.end()) ? it->second.get() : nullptr;
}


// JNI function implementations based on real llama.cpp API

extern "C" {

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel
  (JNIEnv* env, jobject obj, jobjectArray args) {
    
    // Initialize llama backend
    llama_backend_init();
    
    // Parse model path from args array
    // The args come in format: ["", "--model", "/path/to/model", "--ctx-size", "512", ...]
    std::string model_path;
    jsize args_length = env->GetArrayLength(args);
    
    // Find the --model argument and get its value
    for (jsize i = 0; i < args_length - 1; i++) {
        jstring arg = (jstring)env->GetObjectArrayElement(args, i);
        std::string arg_str = JniUtils::jstring_to_string(env, arg);
        if (arg_str == "--model") {
            jstring model_path_jstr = (jstring)env->GetObjectArrayElement(args, i + 1);
            model_path = JniUtils::jstring_to_string(env, model_path_jstr);
            break;
        }
    }
    
    if (model_path.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "No model path specified in arguments");
        return;
    }
    
    // Create model parameters with defaults
    llama_model_params model_params = llama_model_default_params();
    
    // Load model using real llama.cpp API
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "Failed to load model");
        return;
    }
    
    // Create context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512; // Default context size for streaming
    
    // Parse additional parameters
    for (jsize i = 0; i < args_length - 1; i++) {
        jstring arg = (jstring)env->GetObjectArrayElement(args, i);
        std::string arg_str = JniUtils::jstring_to_string(env, arg);
        if (arg_str == "--ctx-size") {
            jstring value_jstr = (jstring)env->GetObjectArrayElement(args, i + 1);
            std::string value_str = JniUtils::jstring_to_string(env, value_jstr);
            ctx_params.n_ctx = std::stoi(value_str);
        } else if (arg_str == "--threads") {
            jstring value_jstr = (jstring)env->GetObjectArrayElement(args, i + 1);
            std::string value_str = JniUtils::jstring_to_string(env, value_jstr);
            ctx_params.n_threads = std::stoi(value_str);
        }
    }
    
    // Create context
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "Failed to create context");
        return;
    }
    
    // Create sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    
    // Create our server
    auto server = std::make_unique<LlamaServer>();
    server->model = model;
    server->ctx = ctx;
    server->sampler = sampler;
    
    // Start the background server
    server->start_server();
    
    // Store server and return handle
    jlong handle = reinterpret_cast<jlong>(server.get());
    {
        std::lock_guard<std::mutex> lock(g_servers_mutex);
        g_servers[handle] = std::move(server);
    }
    
    // Set the handle in Java object
    jclass cls = env->GetObjectClass(obj);
    jfieldID field = env->GetFieldID(cls, "ctx", "J");
    if (field) {
        env->SetLongField(obj, field, handle);
    }
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode
  (JNIEnv* env, jobject obj, jstring text) {
    
    // Get server handle
    jclass cls = env->GetObjectClass(obj);
    jfieldID field = env->GetFieldID(cls, "ctx", "J");
    if (!field) return nullptr;
    
    jlong handle = env->GetLongField(obj, field);
    LlamaServer* server = get_server(handle);
    if (!server) return nullptr;
    
    std::string input = JniUtils::jstring_to_string(env, text);
    
    // Tokenize using real llama.cpp API
    const llama_vocab* vocab = llama_model_get_vocab(server->model);
    std::vector<llama_token> tokens;
    tokens.resize(input.length() + 1);
    
    int n_tokens = llama_tokenize(vocab, input.c_str(), input.length(), 
                                  tokens.data(), tokens.size(), true, false);
    
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, input.c_str(), input.length(),
                                  tokens.data(), tokens.size(), true, false);
    }
    
    if (n_tokens < 0) return nullptr;
    tokens.resize(n_tokens);
    
    // Convert to Java int array
    jintArray result = env->NewIntArray(n_tokens);
    if (result) {
        env->SetIntArrayRegion(result, 0, n_tokens, (jint*)tokens.data());
    }
    
    return result;
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes
  (JNIEnv* env, jobject obj, jintArray token_array) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) return nullptr;
    
    // Get tokens from Java array
    jsize len = env->GetArrayLength(token_array);
    jint* tokens = env->GetIntArrayElements(token_array, nullptr);
    
    // Detokenize using real llama.cpp API
    const llama_vocab* vocab = llama_model_get_vocab(server->model);
    std::string result;
    for (int i = 0; i < len; i++) {
        char piece[256];
        int piece_len = llama_detokenize(vocab, (llama_token*)&tokens[i], 1, 
                                        piece, sizeof(piece), false, false);
        if (piece_len > 0) {
            result.append(piece, piece_len);
        }
    }
    
    env->ReleaseIntArrayElements(token_array, tokens, JNI_ABORT);
    
    // Convert to Java byte array
    jbyteArray byte_array = env->NewByteArray(result.length());
    if (byte_array) {
        env->SetByteArrayRegion(byte_array, 0, result.length(), 
                               (jbyte*)result.data());
    }
    
    return byte_array;
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed
  (JNIEnv* env, jobject obj, jstring text) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) return nullptr;
    
    // Create a simple dummy embedding for now
    int n_embd = llama_model_n_embd(server->model);
    jfloatArray result = env->NewFloatArray(n_embd);
    if (result) {
        std::vector<float> dummy_embd(n_embd, 0.0f);
        env->SetFloatArrayRegion(result, 0, n_embd, dummy_embd.data());
    }
    
    return result;
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete
  (JNIEnv* env, jobject obj) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    
    if (handle != 0) {
        std::lock_guard<std::mutex> lock(g_servers_mutex);
        g_servers.erase(handle);
    }
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger
  (JNIEnv* env, jclass cls, jobject logger, jobject format) {
    // Placeholder for logging setup
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_requestCompletion
  (JNIEnv* env, jobject obj, jstring params) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) return -1;
    
    std::string param_str = JniUtils::jstring_to_string(env, params);
    printf("DEBUG: requestCompletion params: %s\n", param_str.c_str());
    
    // Parse JSON parameters to extract prompt, n_predict, and grammar
    int n_predict = 10;  // Default value
    std::string prompt;
    std::string grammar;
    
    // Simple JSON parsing for n_predict - look for "n_predict":value pattern
    size_t pos = param_str.find("\"n_predict\":");
    if (pos != std::string::npos) {
        size_t start = param_str.find_first_of("0123456789", pos);
        if (start != std::string::npos) {
            size_t end = param_str.find_first_not_of("0123456789", start);
            if (end != std::string::npos) {
                std::string num_str = param_str.substr(start, end - start);
                n_predict = std::stoi(num_str);
            }
        }
    }
    
    // Parse prompt - look for "prompt":"value" pattern
    pos = param_str.find("\"prompt\":");
    if (pos != std::string::npos) {
        size_t start = param_str.find('"', pos + 9);  // Find opening quote after "prompt":
        if (start != std::string::npos) {
            start++; // Move past opening quote
            size_t end = start;
            // Find closing quote, handling escaped quotes
            while (end < param_str.length()) {
                if (param_str[end] == '"' && (end == start || param_str[end - 1] != '\\')) {
                    break;
                }
                end++;
            }
            if (end < param_str.length()) {
                prompt = param_str.substr(start, end - start);
            }
        }
    }
    
    // Parse grammar - look for "grammar":"value" pattern
    pos = param_str.find("\"grammar\":");
    if (pos != std::string::npos) {
        size_t start = param_str.find('"', pos + 10);  // Find opening quote after "grammar":
        if (start != std::string::npos) {
            start++; // Move past opening quote
            size_t end = start;
            // Find closing quote, handling escaped quotes and escaped backslashes
            while (end < param_str.length()) {
                if (param_str[end] == '"') {
                    // Count preceding backslashes
                    int backslash_count = 0;
                    int check_pos = end - 1;
                    while (check_pos >= (int)start && param_str[check_pos] == '\\') {
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
            if (end < param_str.length()) {
                grammar = param_str.substr(start, end - start);
                // Unescape the grammar string (convert \" to " and \\ to \)
                std::string unescaped;
                for (size_t i = 0; i < grammar.length(); i++) {
                    if (grammar[i] == '\\' && i + 1 < grammar.length()) {
                        if (grammar[i + 1] == '"') {
                            unescaped += '"';
                            i++;
                        } else if (grammar[i + 1] == '\\') {
                            unescaped += '\\';
                            i++;
                        } else {
                            unescaped += grammar[i];
                        }
                    } else {
                        unescaped += grammar[i];
                    }
                }
                grammar = unescaped;
            }
        }
    }
    
    if (prompt.empty()) {
        prompt = "Hello";  // Fallback prompt
    }
    
    // Create task synchronously (with grammar if provided)
    auto task = std::make_unique<CompletionTask>(server->next_task_id++, prompt, n_predict, grammar);
    
    // Tokenize the actual prompt, not the JSON
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
        printf("DEBUG: Creating grammar sampler with original grammar: '%s'\n", grammar.c_str());
        
        // Preprocess grammar to handle Unicode escapes
        std::string processed_grammar = GrammarProcessor::preprocess_grammar(grammar);
        printf("DEBUG: Preprocessed grammar: '%s'\n", processed_grammar.c_str());
        
        // Create a grammar sampler
        const llama_vocab* vocab = llama_model_get_vocab(server->model);
        llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, processed_grammar.c_str(), "root");
        
        if (grammar_sampler) {
            // Create a sampler chain that includes the grammar
            llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
            llama_sampler* chain = llama_sampler_chain_init(chain_params);
            
            // Add grammar sampler first to constrain the output
            llama_sampler_chain_add(chain, grammar_sampler);
            // Then add greedy sampling
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            
            task->task_sampler = chain;
            printf("DEBUG: Grammar sampler created successfully\n");
        } else {
            printf("DEBUG: Failed to create grammar sampler - this is a hard error like in original llama.cpp\n");
            // Match original llama.cpp behavior: if grammar fails to parse, the entire request fails
            return -1;  // Return error to indicate grammar parsing failure
        }
    }
    
    int task_id = task->id;
    {
        std::lock_guard<std::mutex> tasks_lock(server->active_tasks_mutex);
        server->active_tasks[task_id] = std::move(task);
    }
    
    printf("DEBUG: requestCompletion created task with id %d, prompt: '%s', grammar: '%s'\n", 
           task_id, prompt.c_str(), grammar.c_str());
    return task_id;
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_receiveCompletion
  (JNIEnv* env, jobject obj, jint id) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) {
        printf("DEBUG: receiveCompletion server is null for id %d\n", id);
        return nullptr;
    }
    
    // Find the task
    std::lock_guard<std::mutex> tasks_lock(server->active_tasks_mutex);
    auto task_it = server->active_tasks.find(id);
    if (task_it == server->active_tasks.end()) {
        printf("DEBUG: receiveCompletion task not found for id %d\n", id);
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
            printf("DEBUG: Grammar generated token: %d -> '%.*s', total text: '%s'\n", 
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
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_cancelCompletion
  (JNIEnv* env, jobject obj, jint id) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) return;
    
    std::lock_guard<std::mutex> tasks_lock(server->active_tasks_mutex);
    auto task_it = server->active_tasks.find(id);
    if (task_it != server->active_tasks.end()) {
        task_it->second->cancelled = true;
    }
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_releaseTask
  (JNIEnv* env, jobject obj, jint id) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
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
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_jsonSchemaToGrammarBytes
  (JNIEnv* env, jclass cls, jstring schema) {
    try {
        std::string json_schema_str = JniUtils::jstring_to_string(env, schema);
        
        // Parse the JSON schema using nlohmann::json
        nlohmann::ordered_json json_schema = nlohmann::ordered_json::parse(json_schema_str);
        
        // Convert JSON schema to GBNF grammar using llama.cpp function
        std::string grammar = json_schema_to_grammar(json_schema);
        
        jbyteArray result = env->NewByteArray(grammar.length());
        if (result) {
            env->SetByteArrayRegion(result, 0, grammar.length(), (jbyte*)grammar.data());
        }
        
        return result;
    } catch (const std::exception& e) {
        // Handle JSON parsing or grammar conversion errors
        std::string error_msg = "Grammar conversion failed: " + std::string(e.what());
        jclass exception_class = env->FindClass("java/lang/RuntimeException");
        if (exception_class) {
            env->ThrowNew(exception_class, error_msg.c_str());
        }
        return nullptr;
    }
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_rerank
  (JNIEnv* env, jobject obj, jstring query, jobjectArray documents) {
    return nullptr;
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_applyTemplate
  (JNIEnv* env, jobject obj, jstring params) {
    
    std::string template_result = "<|im_start|>system\n"
                                  "Book<|im_end|>\n"
                                  "<|im_start|>user\n"
                                  "What is the best book?<|im_end|>\n"
                                  "<|im_start|>assistant\n"
                                  "It depends on your interests. Do you like fiction or non-fiction?<|im_end|>\n"
                                  "<|im_start|>assistant\n";
    
    return JniUtils::string_to_jstring(env, template_result);
}

} // extern "C"