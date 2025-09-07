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
#include "pattern_preprocessor.h"
#include "memory_manager.h"
#include "jni_logger.h"
#include "jni_error_handler.h"
#include "model_manager.h"
#include "tokenization_handler.h"
#include "state_manager.h"
#include "lora_adapter_manager.h"
#include "advanced_sampler_manager.h"
#include "kv_cache_manager.h"
#include "model_info_manager.h"
#include "quantization_manager.h"

// Global server management
std::mutex g_servers_mutex;
std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

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
    ModelManager::loadModel(env, obj, args);
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode
  (JNIEnv* env, jobject obj, jstring text) {
    return TokenizationHandler::encode(env, obj, text);
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes
  (JNIEnv* env, jobject obj, jintArray token_array) {
    return TokenizationHandler::decodeBytes(env, obj, token_array);
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed
  (JNIEnv* env, jobject obj, jstring text) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) return nullptr;
    
    // Check if embedding mode is enabled
    if (!server->embedding_mode) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), 
                     "Model was not loaded with embedding support (see ModelParameters#enableEmbedding())");
        return nullptr;
    }
    
    std::string input = JniUtils::jstring_to_string(env, text);
    
    // Tokenize the input text
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
    
    if (n_tokens < 0) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "Failed to tokenize input for embedding");
        return nullptr;
    }
    
    tokens.resize(n_tokens);
    
    // Clear previous memory (embeddings don't need persistent context)
    llama_memory_clear(llama_get_memory(server->ctx), true);
    
    // Use RAII for batch management to ensure cleanup
    BatchRAII batch_raii(n_tokens, 0, 1);
    llama_batch* batch = batch_raii.get();
    
    for (int i = 0; i < n_tokens; i++) {
        batch->token[i] = tokens[i];
        batch->pos[i] = i;
        batch->n_seq_id[i] = 1;
        batch->seq_id[i][0] = 0;
        batch->logits[i] = true; // We need embeddings for all tokens or just the last one
    }
    batch->n_tokens = n_tokens;
    
    // Process the batch to compute embeddings
    if (llama_decode(server->ctx, *batch) != 0) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "Failed to compute embeddings");
        return nullptr;
    }
    
    // Get embedding dimension
    int n_embd = llama_model_n_embd(server->model);
    
    // Get embeddings based on pooling type
    const float* embd = nullptr;
    enum llama_pooling_type pooling_type = llama_pooling_type(server->ctx);
    
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        // For models without pooling, get embedding from the last token
        embd = llama_get_embeddings_ith(server->ctx, n_tokens - 1);
    } else {
        // For models with pooling, get the sequence embedding
        embd = llama_get_embeddings_seq(server->ctx, 0);
    }
    
    // Batch will be automatically freed by RAII
    
    if (!embd) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "Failed to get embeddings from context");
        return nullptr;
    }
    
    // Create Java float array and copy embeddings
    jfloatArray result = env->NewFloatArray(n_embd);
    if (!result) {
        env->ThrowNew(env->FindClass("java/lang/OutOfMemoryError"), 
                     "Could not allocate embedding array");
        return nullptr;
    }
    
    env->SetFloatArrayRegion(result, 0, n_embd, embd);
    
    return result;
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete
  (JNIEnv* env, jobject obj) {
    ModelManager::deleteModel(env, obj);
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
    JNI_LOG_DEBUG("requestCompletion params: %s", param_str.c_str());
    
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
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_receiveCompletion
  (JNIEnv* env, jobject obj, jint id) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
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
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) return nullptr;
    
    // Check if reranking mode is enabled
    if (!server->reranking_mode) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), 
                     "Model was not loaded with reranking support (see ModelParameters#enableReranking())");
        return nullptr;
    }
    
    std::string query_str = JniUtils::jstring_to_string(env, query);
    
    // Get documents from Java array
    jsize num_documents = env->GetArrayLength(documents);
    if (num_documents == 0) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), 
                     "No documents provided for reranking");
        return nullptr;
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(server->model);
    
    // Tokenize query
    std::vector<llama_token> query_tokens;
    query_tokens.resize(query_str.length() + 1);
    
    int query_n_tokens = llama_tokenize(vocab, query_str.c_str(), query_str.length(),
                                       query_tokens.data(), query_tokens.size(), true, false);
    
    if (query_n_tokens < 0) {
        query_tokens.resize(-query_n_tokens);
        query_n_tokens = llama_tokenize(vocab, query_str.c_str(), query_str.length(),
                                       query_tokens.data(), query_tokens.size(), true, false);
    }
    
    if (query_n_tokens < 0) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "Failed to tokenize query for reranking");
        return nullptr;
    }
    
    query_tokens.resize(query_n_tokens);
    
    // Create LlamaOutput result object
    jclass output_class = env->FindClass("de/kherud/llama/LlamaOutput");
    if (!output_class) return nullptr;
    
    jmethodID constructor = env->GetMethodID(output_class, "<init>", "([BLjava/util/Map;Z)V");
    if (!constructor) return nullptr;
    
    // Create empty byte array (reranking doesn't return text content)
    jbyteArray byte_array = env->NewByteArray(0);
    
    // Create HashMap for probabilities (document -> score mapping)
    jclass hashmap_class = env->FindClass("java/util/HashMap");
    jmethodID hashmap_init = env->GetMethodID(hashmap_class, "<init>", "()V");
    jmethodID hashmap_put = env->GetMethodID(hashmap_class, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject probabilities = env->NewObject(hashmap_class, hashmap_init);
    
    // Process each document
    for (jsize i = 0; i < num_documents; i++) {
        jstring doc_jstr = (jstring)env->GetObjectArrayElement(documents, i);
        std::string doc_str = JniUtils::jstring_to_string(env, doc_jstr);
        
        // Tokenize document
        std::vector<llama_token> doc_tokens;
        doc_tokens.resize(doc_str.length() + 1);
        
        int doc_n_tokens = llama_tokenize(vocab, doc_str.c_str(), doc_str.length(),
                                         doc_tokens.data(), doc_tokens.size(), true, false);
        
        if (doc_n_tokens < 0) {
            doc_tokens.resize(-doc_n_tokens);
            doc_n_tokens = llama_tokenize(vocab, doc_str.c_str(), doc_str.length(),
                                         doc_tokens.data(), doc_tokens.size(), true, false);
        }
        
        if (doc_n_tokens < 0) {
            continue; // Skip this document
        }
        
        doc_tokens.resize(doc_n_tokens);
        
        // Format rerank task: [BOS]query[EOS][SEP]doc[EOS]
        std::vector<llama_token> rerank_tokens;
        rerank_tokens.reserve(query_n_tokens + doc_n_tokens + 4);
        
        // Add BOS if vocab has it
        llama_token bos_token = llama_vocab_bos(vocab);
        if (bos_token != LLAMA_TOKEN_NULL) {
            rerank_tokens.push_back(bos_token);
        }
        
        // Add query tokens
        rerank_tokens.insert(rerank_tokens.end(), query_tokens.begin(), query_tokens.end());
        
        // Add EOS token
        llama_token eos_token = llama_vocab_eos(vocab);
        if (eos_token != LLAMA_TOKEN_NULL) {
            rerank_tokens.push_back(eos_token);
        }
        
        // Add SEP token
        llama_token sep_token = llama_vocab_sep(vocab);
        if (sep_token != LLAMA_TOKEN_NULL) {
            rerank_tokens.push_back(sep_token);
        }
        
        // Add document tokens
        rerank_tokens.insert(rerank_tokens.end(), doc_tokens.begin(), doc_tokens.end());
        
        // Add final EOS token
        if (eos_token != LLAMA_TOKEN_NULL) {
            rerank_tokens.push_back(eos_token);
        }
        
        int total_tokens = rerank_tokens.size();
        
        // Clear previous memory for clean state
        llama_memory_clear(llama_get_memory(server->ctx), true);
        
        // Create batch for reranking computation
        llama_batch batch = llama_batch_init(total_tokens, 0, 1);
        for (int j = 0; j < total_tokens; j++) {
            batch.token[j] = rerank_tokens[j];
            batch.pos[j] = j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = true; // We need embeddings for reranking
        }
        batch.n_tokens = total_tokens;
        
        // Process the batch to compute reranking score
        if (llama_decode(server->ctx, batch) != 0) {
            llama_batch_free(batch);
            continue; // Skip this document
        }
        
        // Get embeddings for reranking score
        // For reranking models with RANK pooling, the score is typically in the first element
        enum llama_pooling_type pooling_type = llama_pooling_type(server->ctx);
        
        const float* embd = nullptr;
        if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
            // For reranking models, get the sequence embedding which contains the score
            embd = llama_get_embeddings_seq(server->ctx, 0);
        } else if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // Fallback: get embedding from last token
            embd = llama_get_embeddings_ith(server->ctx, total_tokens - 1);
        } else {
            // Other pooling types
            embd = llama_get_embeddings_seq(server->ctx, 0);
        }
        
        llama_batch_free(batch);
        
        float score = 0.0f;
        if (embd) {
            // For reranking, the score is typically the first element of the embedding
            score = embd[0];
        }
        
        // Add document and score to the result map
        jstring doc_key = env->NewStringUTF(doc_str.c_str());
        jclass float_class = env->FindClass("java/lang/Float");
        jmethodID float_constructor = env->GetMethodID(float_class, "<init>", "(F)V");
        jobject score_obj = env->NewObject(float_class, float_constructor, score);
        
        env->CallObjectMethod(probabilities, hashmap_put, doc_key, score_obj);
        
        env->DeleteLocalRef(doc_key);
        env->DeleteLocalRef(score_obj);
        env->DeleteLocalRef(doc_jstr);
    }
    
    return env->NewObject(output_class, constructor, byte_array, probabilities, (jboolean)true);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_applyTemplate
  (JNIEnv* env, jobject obj, jstring params) {
    
    jlong handle = env->GetLongField(obj, 
        env->GetFieldID(env->GetObjectClass(obj), "ctx", "J"));
    LlamaServer* server = get_server(handle);
    if (!server) return nullptr;
    
    std::string param_str = JniUtils::jstring_to_string(env, params);
    
    // Parse JSON parameters to extract messages array
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
    
    // Get the model's chat template
    const char* tmpl = llama_model_chat_template(server->model, nullptr);
    if (!tmpl) {
        // Fallback to ChatML template if model doesn't have one
        tmpl = "{% for message in messages %}"
               "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
               "{% endfor %}"
               "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}";
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
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), 
                     "Failed to apply chat template");
        return nullptr;
    }
    
    result_buffer.resize(result_len);
    return JniUtils::string_to_jstring(env, result_buffer);
}

// State persistence functions
JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getStateSize
  (JNIEnv* env, jobject obj) {
    return StateManager::getStateSize(env, obj);
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_getStateData
  (JNIEnv* env, jobject obj) {
    return StateManager::getStateData(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_setStateData
  (JNIEnv* env, jobject obj, jbyteArray state_data) {
    return StateManager::setStateData(env, obj, state_data);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_saveStateToFile
  (JNIEnv* env, jobject obj, jstring path, jintArray tokens) {
    return StateManager::saveStateToFile(env, obj, path, tokens);
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_loadStateFromFile
  (JNIEnv* env, jobject obj, jstring path, jint max_tokens) {
    return StateManager::loadStateFromFile(env, obj, path, max_tokens);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getSequenceStateSizeNative
  (JNIEnv* env, jobject obj, jint seq_id) {
    return StateManager::getSequenceStateSize(env, obj, seq_id);
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_getSequenceStateData
  (JNIEnv* env, jobject obj, jint seq_id) {
    return StateManager::getSequenceStateData(env, obj, seq_id);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_setSequenceStateData
  (JNIEnv* env, jobject obj, jbyteArray state_data, jint seq_id) {
    return StateManager::setSequenceStateData(env, obj, state_data, seq_id);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_saveSequenceToFile
  (JNIEnv* env, jobject obj, jstring path, jint seq_id, jintArray tokens) {
    return StateManager::saveSequenceToFile(env, obj, path, seq_id, tokens);
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_loadSequenceFromFile
  (JNIEnv* env, jobject obj, jstring path, jint seq_id, jint max_tokens) {
    return StateManager::loadSequenceFromFile(env, obj, path, seq_id, max_tokens);
}

// LoRA adapter functions
JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_loadLoRAAdapterNative
  (JNIEnv* env, jobject obj, jstring lora_path) {
    return LoRAAdapterManager::loadAdapter(env, obj, lora_path);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_freeLoRAAdapterNative
  (JNIEnv* env, jclass cls, jlong adapter_handle) {
    LoRAAdapterManager::freeAdapter(env, adapter_handle);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_setLoRAAdapterNative
  (JNIEnv* env, jobject obj, jlong adapter_handle, jfloat scale) {
    return LoRAAdapterManager::setAdapter(env, obj, adapter_handle, scale);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_removeLoRAAdapterNative
  (JNIEnv* env, jobject obj, jlong adapter_handle) {
    return LoRAAdapterManager::removeAdapter(env, obj, adapter_handle);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_clearLoRAAdaptersNative
  (JNIEnv* env, jobject obj) {
    LoRAAdapterManager::clearAdapters(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_applyControlVectorNative
  (JNIEnv* env, jobject obj, jfloatArray data) {
    return LoRAAdapterManager::applyControlVector(env, obj, data);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getLoRAAdapterMetadataNative
  (JNIEnv* env, jclass cls, jlong adapter_handle, jstring key) {
    return LoRAAdapterManager::getAdapterMetaValue(env, adapter_handle, key);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getLoRAAdapterMetadataCountNative
  (JNIEnv* env, jclass cls, jlong adapter_handle) {
    return LoRAAdapterManager::getAdapterMetaCount(env, adapter_handle);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getLoRAAdapterMetadataKeyNative
  (JNIEnv* env, jclass cls, jlong adapter_handle, jint index) {
    return LoRAAdapterManager::getAdapterMetaKeyByIndex(env, adapter_handle, index);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getLoRAAdapterMetadataValueNative
  (JNIEnv* env, jclass cls, jlong adapter_handle, jint index) {
    return LoRAAdapterManager::getAdapterMetaValueByIndex(env, adapter_handle, index);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getAloraInvocationTokenCountNative
  (JNIEnv* env, jclass cls, jlong adapter_handle) {
    return LoRAAdapterManager::getAloraInvocationTokenCount(env, adapter_handle);
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_getAloraInvocationTokensNative
  (JNIEnv* env, jclass cls, jlong adapter_handle) {
    return LoRAAdapterManager::getAloraInvocationTokens(env, adapter_handle);
}

// Advanced sampling functions
JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createGreedySamplerNative
  (JNIEnv* env, jclass cls) {
    return AdvancedSamplerManager::createGreedySampler(env);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createDistributionSamplerNative
  (JNIEnv* env, jclass cls, jint seed) {
    return AdvancedSamplerManager::createDistributionSampler(env, seed);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createTopKSamplerNative
  (JNIEnv* env, jclass cls, jint k) {
    return AdvancedSamplerManager::createTopKSampler(env, k);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createTopPSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jint minKeep) {
    return AdvancedSamplerManager::createTopPSampler(env, p, minKeep);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createMinPSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jint minKeep) {
    return AdvancedSamplerManager::createMinPSampler(env, p, minKeep);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createTemperatureSamplerNative
  (JNIEnv* env, jclass cls, jfloat temperature) {
    return AdvancedSamplerManager::createTemperatureSampler(env, temperature);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createExtendedTemperatureSamplerNative
  (JNIEnv* env, jclass cls, jfloat temp, jfloat delta, jfloat exponent) {
    return AdvancedSamplerManager::createExtendedTemperatureSampler(env, temp, delta, exponent);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createTypicalSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jint minKeep) {
    return AdvancedSamplerManager::createTypicalSampler(env, p, minKeep);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createXtcSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jfloat t, jint minKeep, jint seed) {
    return AdvancedSamplerManager::createXtcSampler(env, p, t, minKeep, seed);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createTopNSigmaSamplerNative
  (JNIEnv* env, jclass cls, jfloat n) {
    return AdvancedSamplerManager::createTopNSigmaSampler(env, n);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createMirostatSamplerNative
  (JNIEnv* env, jclass cls, jint nVocab, jint seed, jfloat tau, jfloat eta, jint m) {
    return AdvancedSamplerManager::createMirostatSampler(env, nVocab, seed, tau, eta, m);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createMirostatV2SamplerNative
  (JNIEnv* env, jclass cls, jint seed, jfloat tau, jfloat eta) {
    return AdvancedSamplerManager::createMirostatV2Sampler(env, seed, tau, eta);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createPenaltiesSamplerNative
  (JNIEnv* env, jclass cls, jint penaltyLastN, jfloat penaltyRepeat, jfloat penaltyFreq, jfloat penaltyPresent) {
    return AdvancedSamplerManager::createPenaltiesSampler(env, penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createDrySamplerNative
  (JNIEnv* env, jobject obj, jint nCtxTrain, jfloat multiplier, jfloat base, jint allowedLength, jint penaltyLastN, jintArray sequenceBreakers) {
    return AdvancedSamplerManager::createDrySampler(env, obj, nCtxTrain, multiplier, base, allowedLength, penaltyLastN, sequenceBreakers);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createLogitBiasSamplerNative
  (JNIEnv* env, jclass cls, jint nVocab, jint nLogitBias, jintArray biasTokens, jfloatArray biasValues) {
    return AdvancedSamplerManager::createLogitBiasSampler(env, nVocab, nLogitBias, biasTokens, biasValues);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createGrammarSamplerNative
  (JNIEnv* env, jobject obj, jstring grammarStr, jstring rootRule) {
    return AdvancedSamplerManager::createGrammarSampler(env, obj, grammarStr, rootRule);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createInfillSamplerNative
  (JNIEnv* env, jobject obj) {
    return AdvancedSamplerManager::createInfillSampler(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_createSamplerChainNative
  (JNIEnv* env, jclass cls) {
    return AdvancedSamplerManager::createSamplerChain(env);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_addToSamplerChainNative
  (JNIEnv* env, jclass cls, jlong chainHandle, jlong samplerHandle) {
    AdvancedSamplerManager::addToSamplerChain(env, chainHandle, samplerHandle);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_cloneSamplerNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    return AdvancedSamplerManager::cloneSampler(env, samplerHandle);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_freeSamplerNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    AdvancedSamplerManager::freeSampler(env, samplerHandle);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_sampleTokenNative
  (JNIEnv* env, jobject obj, jlong samplerHandle) {
    return AdvancedSamplerManager::sampleToken(env, obj, samplerHandle);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_acceptTokenNative
  (JNIEnv* env, jclass cls, jlong samplerHandle, jint token) {
    AdvancedSamplerManager::acceptToken(env, samplerHandle, token);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_resetSamplerNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    AdvancedSamplerManager::resetSampler(env, samplerHandle);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getSamplerNameNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    return AdvancedSamplerManager::getSamplerName(env, samplerHandle);
}

// JNI bindings for LlamaSampler class
JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createGreedySamplerNative
  (JNIEnv* env, jclass cls) {
    return AdvancedSamplerManager::createGreedySampler(env);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createDistributionSamplerNative
  (JNIEnv* env, jclass cls, jint seed) {
    return AdvancedSamplerManager::createDistributionSampler(env, seed);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createTopKSamplerNative
  (JNIEnv* env, jclass cls, jint k) {
    return AdvancedSamplerManager::createTopKSampler(env, k);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createTopPSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jint minKeep) {
    return AdvancedSamplerManager::createTopPSampler(env, p, minKeep);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createMinPSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jint minKeep) {
    return AdvancedSamplerManager::createMinPSampler(env, p, minKeep);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createTemperatureSamplerNative
  (JNIEnv* env, jclass cls, jfloat temperature) {
    return AdvancedSamplerManager::createTemperatureSampler(env, temperature);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createExtendedTemperatureSamplerNative
  (JNIEnv* env, jclass cls, jfloat temp, jfloat delta, jfloat exponent) {
    return AdvancedSamplerManager::createExtendedTemperatureSampler(env, temp, delta, exponent);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createTypicalSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jint minKeep) {
    return AdvancedSamplerManager::createTypicalSampler(env, p, minKeep);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createXtcSamplerNative
  (JNIEnv* env, jclass cls, jfloat p, jfloat t, jint minKeep, jint seed) {
    return AdvancedSamplerManager::createXtcSampler(env, p, t, minKeep, seed);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createMirostatV2SamplerNative
  (JNIEnv* env, jclass cls, jint seed, jfloat tau, jfloat eta) {
    return AdvancedSamplerManager::createMirostatV2Sampler(env, seed, tau, eta);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createPenaltiesSamplerNative
  (JNIEnv* env, jclass cls, jint penaltyLastN, jfloat penaltyRepeat, jfloat penaltyFreq, jfloat penaltyPresent) {
    return AdvancedSamplerManager::createPenaltiesSampler(env, penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_createSamplerChainNative
  (JNIEnv* env, jclass cls) {
    return AdvancedSamplerManager::createSamplerChain(env);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaSampler_addToSamplerChainNative
  (JNIEnv* env, jclass cls, jlong chainHandle, jlong samplerHandle) {
    AdvancedSamplerManager::addToSamplerChain(env, chainHandle, samplerHandle);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaSampler_cloneSamplerNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    return AdvancedSamplerManager::cloneSampler(env, samplerHandle);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaSampler_freeSamplerNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    AdvancedSamplerManager::freeSampler(env, samplerHandle);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaSampler_getSamplerNameNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    return AdvancedSamplerManager::getSamplerName(env, samplerHandle);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaSampler_resetSamplerNative
  (JNIEnv* env, jclass cls, jlong samplerHandle) {
    AdvancedSamplerManager::resetSampler(env, samplerHandle);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaSampler_acceptTokenNative
  (JNIEnv* env, jclass cls, jlong samplerHandle, jint token) {
    AdvancedSamplerManager::acceptToken(env, samplerHandle, token);
}

// KV Cache Management JNI bindings
JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_copySequenceNative
  (JNIEnv* env, jobject obj, jint srcSeqId, jint dstSeqId, jint p0, jint p1) {
    KVCacheManager::copySequence(env, obj, srcSeqId, dstSeqId, p0, p1);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_keepSequenceNative
  (JNIEnv* env, jobject obj, jint seqId) {
    KVCacheManager::keepSequence(env, obj, seqId);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_addPositionDeltaNative
  (JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1, jint delta) {
    KVCacheManager::addPositionDelta(env, obj, seqId, p0, p1, delta);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_dividePositionsNative
  (JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1, jint divisor) {
    KVCacheManager::dividePositions(env, obj, seqId, p0, p1, divisor);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getSequenceMinPositionNative
  (JNIEnv* env, jobject obj, jint seqId) {
    return KVCacheManager::getSequenceMinPosition(env, obj, seqId);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getSequenceMaxPositionNative
  (JNIEnv* env, jobject obj, jint seqId) {
    return KVCacheManager::getSequenceMaxPosition(env, obj, seqId);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_canShiftContextNative
  (JNIEnv* env, jobject obj) {
    return KVCacheManager::canShiftContext(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_clearMemoryNative
  (JNIEnv* env, jobject obj, jboolean clearData) {
    KVCacheManager::clearMemory(env, obj, clearData);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_removeSequenceTokensNative
  (JNIEnv* env, jobject obj, jint seqId, jint p0, jint p1) {
    return KVCacheManager::removeSequenceTokens(env, obj, seqId, p0, p1);
}

// Model Information JNI bindings
JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getModelParameterCountNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getModelParameterCount(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getModelSizeNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getModelSize(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getModelMetadataCountNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getModelMetadataCount(env, obj);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getModelMetadataKeyByIndexNative
  (JNIEnv* env, jobject obj, jint index) {
    return ModelInfoManager::getModelMetadataKeyByIndex(env, obj, index);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getModelMetadataValueByIndexNative
  (JNIEnv* env, jobject obj, jint index) {
    return ModelInfoManager::getModelMetadataValueByIndex(env, obj, index);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getModelMetadataValueNative
  (JNIEnv* env, jobject obj, jstring key) {
    return ModelInfoManager::getModelMetadataValue(env, obj, key);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getVocabularyTypeNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getVocabularyType(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getVocabularySizeNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getVocabularySize(env, obj);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getTokenTextNative
  (JNIEnv* env, jobject obj, jint token) {
    return ModelInfoManager::getTokenText(env, obj, token);
}

JNIEXPORT jfloat JNICALL Java_de_kherud_llama_LlamaModel_getTokenScoreNative
  (JNIEnv* env, jobject obj, jint token) {
    return ModelInfoManager::getTokenScore(env, obj, token);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getTokenAttributesNative
  (JNIEnv* env, jobject obj, jint token) {
    return ModelInfoManager::getTokenAttributes(env, obj, token);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getBosTokenNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getBosToken(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getEosTokenNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getEosToken(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getEotTokenNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getEotToken(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getSepTokenNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getSepToken(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getNlTokenNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getNlToken(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getPadTokenNative
  (JNIEnv* env, jobject obj) {
    return ModelInfoManager::getPadToken(env, obj);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_isEogTokenNative
  (JNIEnv* env, jobject obj, jint token) {
    return ModelInfoManager::isEogToken(env, obj, token);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_isControlTokenNative
  (JNIEnv* env, jobject obj, jint token) {
    return ModelInfoManager::isControlToken(env, obj, token);
}

// Quantization JNI bindings
JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaQuantizer_getDefaultQuantizationParamsNative
  (JNIEnv* env, jclass cls) {
    return QuantizationManager::getDefaultQuantizationParams(env);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaQuantizer_quantizeModelNative
  (JNIEnv* env, jclass cls, jstring inputPath, jstring outputPath, jobject params) {
    return QuantizationManager::quantizeModel(env, inputPath, outputPath, params);
}

} // extern "C"