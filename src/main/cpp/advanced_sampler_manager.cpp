#include "advanced_sampler_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "llama_server.h"
#include <mutex>
#include <unordered_map>
#include <memory>

// External global server management (defined in jllama.cpp)
extern std::mutex g_servers_mutex;
extern std::unordered_map<jlong, std::unique_ptr<LlamaServer>> g_servers;

static LlamaServer* get_server(jlong handle) {
	std::lock_guard<std::mutex> lock(g_servers_mutex);
	auto it = g_servers.find(handle);
	return (it != g_servers.end()) ? it->second.get() : nullptr;
}

// Static initialization to ensure backend is ready
static std::once_flag g_backend_init_flag;
static bool g_backend_initialized = false;
static void ensureBackendInitialized() {
	std::call_once(g_backend_init_flag, []() {
		// Only initialize if not already done elsewhere
		if (!g_backend_initialized) {
			llama_backend_init();
			g_backend_initialized = true;
		}
	});
}

// Basic samplers

jlong AdvancedSamplerManager::createGreedySampler(JNIEnv* env) {
	JNI_TRY(env)
	
	// Ensure backend is initialized
	ensureBackendInitialized();
	
	llama_sampler* sampler = llama_sampler_init_greedy();
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create greedy sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createDistributionSampler(JNIEnv* env, jint seed) {
	JNI_TRY(env)
	
	ensureBackendInitialized();
	llama_sampler* sampler = llama_sampler_init_dist(static_cast<uint32_t>(seed));
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create distribution sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

// Top-K and Top-P samplers

jlong AdvancedSamplerManager::createTopKSampler(JNIEnv* env, jint k) {
	JNI_TRY(env)
	
	ensureBackendInitialized();
	if (k <= 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Top-K value must be positive");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_top_k(static_cast<int32_t>(k));
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create top-k sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createTopPSampler(JNIEnv* env, jfloat p, jint minKeep) {
	JNI_TRY(env)
	
	ensureBackendInitialized();
	if (p < 0.0f || p > 1.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Top-P value must be between 0.0 and 1.0");
		return -1;
	}
	
	if (minKeep < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Min keep must be non-negative");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_top_p(p, static_cast<size_t>(minKeep));
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create top-p sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createMinPSampler(JNIEnv* env, jfloat p, jint minKeep) {
	JNI_TRY(env)
	
	if (p < 0.0f || p > 1.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Min-P value must be between 0.0 and 1.0");
		return -1;
	}
	
	if (minKeep < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Min keep must be non-negative");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_min_p(p, static_cast<size_t>(minKeep));
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create min-p sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

// Temperature samplers

jlong AdvancedSamplerManager::createTemperatureSampler(JNIEnv* env, jfloat temperature) {
	JNI_TRY(env)
	
	ensureBackendInitialized();
	if (temperature < 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Temperature must be non-negative");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_temp(temperature);
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create temperature sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createExtendedTemperatureSampler(JNIEnv* env, jfloat temp, jfloat delta, jfloat exponent) {
	JNI_TRY(env)
	
	if (temp < 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Temperature must be non-negative");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_temp_ext(temp, delta, exponent);
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create extended temperature sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

// Advanced samplers

jlong AdvancedSamplerManager::createTypicalSampler(JNIEnv* env, jfloat p, jint minKeep) {
	JNI_TRY(env)
	
	if (p < 0.0f || p > 1.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Typical sampling p value must be between 0.0 and 1.0");
		return -1;
	}
	
	if (minKeep < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Min keep must be non-negative");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_typical(p, static_cast<size_t>(minKeep));
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create typical sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createXtcSampler(JNIEnv* env, jfloat p, jfloat t, jint minKeep, jint seed) {
	JNI_TRY(env)
	
	if (p < 0.0f || p > 1.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "XTC p value must be between 0.0 and 1.0");
		return -1;
	}
	
	if (t < 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "XTC threshold must be non-negative");
		return -1;
	}
	
	if (minKeep < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Min keep must be non-negative");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_xtc(p, t, static_cast<size_t>(minKeep), static_cast<uint32_t>(seed));
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create XTC sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createTopNSigmaSampler(JNIEnv* env, jfloat n) {
	JNI_TRY(env)
	
	if (n <= 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Top-N Sigma value must be positive");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_top_n_sigma(n);
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create top-n sigma sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

// Mirostat samplers

jlong AdvancedSamplerManager::createMirostatSampler(JNIEnv* env, jint nVocab, jint seed, jfloat tau, jfloat eta, jint m) {
	JNI_TRY(env)
	
	if (nVocab <= 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Vocabulary size must be positive");
		return -1;
	}
	
	if (tau <= 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Mirostat tau must be positive");
		return -1;
	}
	
	if (eta <= 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Mirostat eta must be positive");
		return -1;
	}
	
	if (m <= 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Mirostat m must be positive");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_mirostat(
		static_cast<int32_t>(nVocab),
		static_cast<uint32_t>(seed),
		tau,
		eta,
		static_cast<int32_t>(m)
	);
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create Mirostat sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createMirostatV2Sampler(JNIEnv* env, jint seed, jfloat tau, jfloat eta) {
	JNI_TRY(env)
	
	if (tau <= 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Mirostat V2 tau must be positive");
		return -1;
	}
	
	if (eta <= 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "Mirostat V2 eta must be positive");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_mirostat_v2(
		static_cast<uint32_t>(seed),
		tau,
		eta
	);
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create Mirostat V2 sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

// Penalty and bias samplers

jlong AdvancedSamplerManager::createPenaltiesSampler(JNIEnv* env, jint penaltyLastN, 
		jfloat penaltyRepeat, jfloat penaltyFreq, jfloat penaltyPresent) {
	JNI_TRY(env)
	
	if (penaltyLastN < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Penalty last N must be non-negative");
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_penalties(
		static_cast<int32_t>(penaltyLastN),
		penaltyRepeat,
		penaltyFreq,
		penaltyPresent
	);
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create penalties sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createDrySampler(JNIEnv* env, jobject model, jint nCtxTrain, jfloat multiplier, jfloat base, 
		jint allowedLength, jint penaltyLastN, jintArray sequenceBreakers) {
	JNI_TRY(env)
	
	if (nCtxTrain <= 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Context train size must be positive");
		return -1;
	}
	
	if (multiplier <= 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "DRY multiplier must be positive");
		return -1;
	}
	
	if (base <= 0.0f) {
		JNIErrorHandler::throw_runtime_exception(env, "DRY base must be positive");
		return -1;
	}
	
	if (allowedLength < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "DRY allowed length must be non-negative");
		return -1;
	}
	
	if (penaltyLastN < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "DRY penalty last N must be non-negative");
		return -1;
	}
	
	const llama_vocab* vocab = getVocab(env, model);
	if (!vocab) {
		return -1;
	}
	
	// Handle sequence breakers array - convert to string array
	const char** breakers = nullptr;
	size_t n_breakers = 0;
	std::vector<std::string> breaker_strings;
	std::vector<const char*> breaker_ptrs;
	
	if (sequenceBreakers) {
		n_breakers = env->GetArrayLength(sequenceBreakers);
		if (n_breakers > 0) {
			jint* breaker_elements = env->GetIntArrayElements(sequenceBreakers, nullptr);
			if (!breaker_elements) {
				JNIErrorHandler::throw_runtime_exception(env, "Failed to get sequence breaker elements");
				return -1;
			}
			
			// Convert token IDs to strings (simplified - in practice you'd decode them)
			breaker_strings.reserve(n_breakers);
			breaker_ptrs.reserve(n_breakers);
			for (size_t i = 0; i < n_breakers; i++) {
				breaker_strings.push_back(std::to_string(breaker_elements[i]));
				breaker_ptrs.push_back(breaker_strings.back().c_str());
			}
			breakers = breaker_ptrs.data();
			
			env->ReleaseIntArrayElements(sequenceBreakers, breaker_elements, JNI_ABORT);
		}
	}
	
	llama_sampler* sampler = llama_sampler_init_dry(
		vocab,
		static_cast<int32_t>(nCtxTrain),
		multiplier,
		base,
		static_cast<int32_t>(allowedLength),
		static_cast<int32_t>(penaltyLastN),
		breakers,
		n_breakers
	);
	
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create DRY sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createLogitBiasSampler(JNIEnv* env, jint nVocab, jint nLogitBias, jintArray biasTokens, jfloatArray biasValues) {
	JNI_TRY(env)
	
	if (nVocab <= 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Vocabulary size must be positive");
		return -1;
	}
	
	if (nLogitBias < 0) {
		JNIErrorHandler::throw_runtime_exception(env, "Number of logit biases must be non-negative");
		return -1;
	}
	
	if (nLogitBias == 0) {
		// Create empty logit bias sampler
		llama_sampler* sampler = llama_sampler_init_logit_bias(static_cast<int32_t>(nVocab), 0, nullptr);
		if (!sampler) {
			JNIErrorHandler::throw_runtime_exception(env, "Failed to create empty logit bias sampler");
			return -1;
		}
		return reinterpret_cast<jlong>(sampler);
	}
	
	// Validate arrays
	if (!JNIErrorHandler::validate_array(env, biasTokens, "biasTokens", nLogitBias)) {
		return -1;
	}
	if (!JNIErrorHandler::validate_array(env, biasValues, "biasValues", nLogitBias)) {
		return -1;
	}
	
	// Get array elements
	jint* token_elements = env->GetIntArrayElements(biasTokens, nullptr);
	if (!token_elements) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get bias token elements");
		return -1;
	}
	
	jfloat* value_elements = env->GetFloatArrayElements(biasValues, nullptr);
	if (!value_elements) {
		env->ReleaseIntArrayElements(biasTokens, token_elements, JNI_ABORT);
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get bias value elements");
		return -1;
	}
	
	// Create array of llama_logit_bias structures
	std::vector<llama_logit_bias> bias_array(nLogitBias);
	for (int32_t i = 0; i < nLogitBias; i++) {
		bias_array[i].token = static_cast<llama_token>(token_elements[i]);
		bias_array[i].bias = value_elements[i];
	}
	
	llama_sampler* sampler = llama_sampler_init_logit_bias(
		static_cast<int32_t>(nVocab),
		static_cast<int32_t>(nLogitBias),
		bias_array.data()
	);
	
	// Release arrays
	env->ReleaseIntArrayElements(biasTokens, token_elements, JNI_ABORT);
	env->ReleaseFloatArrayElements(biasValues, value_elements, JNI_ABORT);
	
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create logit bias sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

// Grammar and pattern samplers

jlong AdvancedSamplerManager::createGrammarSampler(JNIEnv* env, jobject model, jstring grammarStr, jstring rootRule) {
	JNI_TRY(env)
	
	if (!JNIErrorHandler::validate_string(env, grammarStr, "grammarStr")) {
		return -1;
	}
	
	const llama_vocab* vocab = getVocab(env, model);
	if (!vocab) {
		return -1;
	}
	
	std::string grammar = JniUtils::jstring_to_string(env, grammarStr);
	std::string root = rootRule ? JniUtils::jstring_to_string(env, rootRule) : "root";
	
	llama_sampler* sampler = llama_sampler_init_grammar(vocab, grammar.c_str(), root.c_str());
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create grammar sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

jlong AdvancedSamplerManager::createInfillSampler(JNIEnv* env, jobject model) {
	JNI_TRY(env)
	
	const llama_vocab* vocab = getVocab(env, model);
	if (!vocab) {
		return -1;
	}
	
	llama_sampler* sampler = llama_sampler_init_infill(vocab);
	if (!sampler) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create infill sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(sampler);
	
	JNI_CATCH_RET(env, -1)
}

// Sampler chain management

jlong AdvancedSamplerManager::createSamplerChain(JNIEnv* env) {
	JNI_TRY(env)
	
	ensureBackendInitialized();
	llama_sampler_chain_params params = llama_sampler_chain_default_params();
	llama_sampler* chain = llama_sampler_chain_init(params);
	if (!chain) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to create sampler chain");
		return -1;
	}
	
	return reinterpret_cast<jlong>(chain);
	
	JNI_CATCH_RET(env, -1)
}

void AdvancedSamplerManager::addToSamplerChain(JNIEnv* env, jlong chainHandle, jlong samplerHandle) {
	JNI_TRY(env)
	
	if (!validateSamplerHandle(chainHandle) || !validateSamplerHandle(samplerHandle)) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid sampler handle");
		return;
	}
	
	llama_sampler* chain = reinterpret_cast<llama_sampler*>(chainHandle);
	llama_sampler* sampler = reinterpret_cast<llama_sampler*>(samplerHandle);
	
	llama_sampler_chain_add(chain, sampler);
	
	JNI_CATCH(env)
}

jlong AdvancedSamplerManager::cloneSampler(JNIEnv* env, jlong samplerHandle) {
	JNI_TRY(env)
	
	if (!validateSamplerHandle(samplerHandle)) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid sampler handle");
		return -1;
	}
	
	llama_sampler* original = reinterpret_cast<llama_sampler*>(samplerHandle);
	llama_sampler* cloned = llama_sampler_clone(original);
	if (!cloned) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to clone sampler");
		return -1;
	}
	
	return reinterpret_cast<jlong>(cloned);
	
	JNI_CATCH_RET(env, -1)
}

void AdvancedSamplerManager::freeSampler(JNIEnv* env, jlong samplerHandle) {
	JNI_TRY(env)
	
	// Early validation to prevent crashes
	if (samplerHandle <= 0 || samplerHandle == -1 || 
		samplerHandle < 0x1000 || samplerHandle >= 0x7FFFFFFFFFFFFFFFLL) {
		// Invalid handle, silently return without attempting to free
		return;
	}
	
	llama_sampler* sampler = reinterpret_cast<llama_sampler*>(samplerHandle);
	if (sampler) {
		llama_sampler_free(sampler);
	}
	
	JNI_CATCH(env)
}

// Sampling operations

jint AdvancedSamplerManager::sampleToken(JNIEnv* env, jobject obj, jlong samplerHandle) {
	JNI_TRY(env)
	
	if (!validateSamplerHandle(samplerHandle)) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid sampler handle");
		return -1;
	}
	
	llama_context* ctx = getContext(env, obj);
	if (!ctx) {
		return -1;
	}
	
	llama_sampler* sampler = reinterpret_cast<llama_sampler*>(samplerHandle);
	llama_token token = llama_sampler_sample(sampler, ctx, -1);
	
	return static_cast<jint>(token);
	
	JNI_CATCH_RET(env, -1)
}

void AdvancedSamplerManager::acceptToken(JNIEnv* env, jlong samplerHandle, jint token) {
	JNI_TRY(env)
	
	if (!validateSamplerHandle(samplerHandle)) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid sampler handle");
		return;
	}
	
	llama_sampler* sampler = reinterpret_cast<llama_sampler*>(samplerHandle);
	llama_sampler_accept(sampler, static_cast<llama_token>(token));
	
	JNI_CATCH(env)
}

void AdvancedSamplerManager::resetSampler(JNIEnv* env, jlong samplerHandle) {
	JNI_TRY(env)
	
	if (!validateSamplerHandle(samplerHandle)) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid sampler handle");
		return;
	}
	
	llama_sampler* sampler = reinterpret_cast<llama_sampler*>(samplerHandle);
	llama_sampler_reset(sampler);
	
	JNI_CATCH(env)
}

// Sampler configuration and info

jstring AdvancedSamplerManager::getSamplerName(JNIEnv* env, jlong samplerHandle) {
	JNI_TRY(env)
	
	if (!validateSamplerHandle(samplerHandle)) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid sampler handle");
		return nullptr;
	}
	
	llama_sampler* sampler = reinterpret_cast<llama_sampler*>(samplerHandle);
	const char* name = llama_sampler_name(sampler);
	
	if (!name) {
		return env->NewStringUTF("unknown");
	}
	
	return JniUtils::string_to_jstring(env, std::string(name));
	
	JNI_CATCH_RET(env, nullptr)
}

// Helper methods

struct llama_context* AdvancedSamplerManager::getContext(JNIEnv* env, jobject obj) {
	jclass cls = env->GetObjectClass(obj);
	if (!cls) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get object class");
		return nullptr;
	}
	
	jfieldID field = env->GetFieldID(cls, "ctx", "J");
	if (!field) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get ctx field");
		return nullptr;
	}
	
	jlong handle = env->GetLongField(obj, field);
	LlamaServer* server = get_server(handle);
	if (!server) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid server handle");
		return nullptr;
	}
	
	return server->ctx;
}

const struct llama_vocab* AdvancedSamplerManager::getVocab(JNIEnv* env, jobject obj) {
	jclass cls = env->GetObjectClass(obj);
	if (!cls) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get object class");
		return nullptr;
	}
	
	jfieldID field = env->GetFieldID(cls, "ctx", "J");
	if (!field) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get ctx field");
		return nullptr;
	}
	
	jlong handle = env->GetLongField(obj, field);
	LlamaServer* server = get_server(handle);
	if (!server || !server->model) {
		JNIErrorHandler::throw_runtime_exception(env, "Invalid server or model handle");
		return nullptr;
	}
	
	return llama_model_get_vocab(server->model);
}

bool AdvancedSamplerManager::validateSamplerHandle(jlong handle) {
	// Reject invalid handles immediately
	if (handle <= 0 || handle == -1) {
		return false;
	}
	
	// Additional safety check - reject handles that are clearly invalid pointers
	if (handle < 0x1000 || handle >= 0x7FFFFFFFFFFFFFFFLL) {
		return false;
	}
	
	return true;
}