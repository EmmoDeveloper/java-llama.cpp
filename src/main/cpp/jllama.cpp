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
#include "embedding_manager.h"
#include "completion_manager.h"
#include "template_manager.h"
#include "reranking_manager.h"
#include "schema_grammar_manager.h"
#include "model_loader_manager.h"
#include "utility_manager.h"

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

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_loadModelFromSplits
  (JNIEnv* env, jclass cls, jobjectArray paths, jobject params) {
    return ModelLoaderManager::loadModelFromSplits(env, cls, paths, params);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_saveModelToFile
  (JNIEnv* env, jobject obj, jstring path) {
    ModelLoaderManager::saveModelToFile(env, obj, path);
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
    return EmbeddingManager::createEmbedding(env, obj, text);
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_getAllEmbeddings
  (JNIEnv* env, jobject obj) {
    return EmbeddingManager::getAllEmbeddings(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setEmbeddingMode
  (JNIEnv* env, jobject obj, jboolean embeddings) {
    EmbeddingManager::setEmbeddingMode(env, obj, embeddings);
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
    return CompletionManager::requestCompletion(env, obj, params);
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_receiveCompletion
  (JNIEnv* env, jobject obj, jint id) {
    return CompletionManager::receiveCompletion(env, obj, id);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_cancelCompletion
  (JNIEnv* env, jobject obj, jint id) {
    CompletionManager::cancelCompletion(env, obj, id);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_releaseTask
  (JNIEnv* env, jobject obj, jint id) {
    CompletionManager::releaseTask(env, obj, id);
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_jsonSchemaToGrammarBytes
  (JNIEnv* env, jclass cls, jstring schema) {
    return SchemaGrammarManager::jsonSchemaToGrammarBytes(env, cls, schema);
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_rerank
  (JNIEnv* env, jobject obj, jstring query, jobjectArray documents) {
    return RerankingManager::rerank(env, obj, query, documents);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_applyTemplate
  (JNIEnv* env, jobject obj, jstring params) {
    return TemplateManager::applyTemplate(env, obj, params);
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

// Utility function JNI bindings
JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaUtils_supportsGpuOffloadNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::supportsGpuOffload(env, cls);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaUtils_supportsMmapNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::supportsMmap(env, cls);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaUtils_supportsMlockNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::supportsMlock(env, cls);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaUtils_supportsRpcNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::supportsRpc(env, cls);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaUtils_maxDevicesNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::maxDevices(env, cls);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaUtils_maxParallelSequencesNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::maxParallelSequences(env, cls);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaUtils_printSystemInfoNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::printSystemInfo(env, cls);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaUtils_timeUsNative
  (JNIEnv* env, jclass cls) {
    return UtilityManager::timeUs(env, cls);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaUtils_setLogCallbackNative
  (JNIEnv* env, jclass cls, jobject callback) {
    UtilityManager::setLogCallback(env, cls, callback);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setAbortCallbackNative
  (JNIEnv* env, jobject obj, jobject callback) {
    UtilityManager::setAbortCallback(env, obj, callback);
}

// Tier 2 Utility function JNI bindings
JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setThreadCountNative
  (JNIEnv* env, jobject obj, jint threads) {
    UtilityManager::setThreadCount(env, obj, threads);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_synchronizeOperationsNative
  (JNIEnv* env, jobject obj) {
    UtilityManager::synchronizeOperations(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setEmbeddingModeNative
  (JNIEnv* env, jobject obj, jboolean embeddings) {
    UtilityManager::setEmbeddingMode(env, obj, embeddings);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setCausalAttentionNative
  (JNIEnv* env, jobject obj, jboolean causal) {
    UtilityManager::setCausalAttention(env, obj, causal);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaUtils_splitPathNative
  (JNIEnv* env, jclass cls, jstring path, jint split) {
    return UtilityManager::splitPath(env, cls, path, split);
}

// ===== TIER 3: ADVANCED SYSTEM MANAGEMENT & PERFORMANCE =====

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getContextSizeNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getContextSize(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getBatchSizeNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getBatchSize(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getUbatchSizeNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getUbatchSize(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getMaxSequencesNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getMaxSequences(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getCurrentThreadsNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getCurrentThreads(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getCurrentThreadsBatchNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getCurrentThreadsBatch(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_attachThreadPoolNative
  (JNIEnv* env, jobject obj, jlong threadpool, jlong threadpool_batch) {
    UtilityManager::attachThreadPool(env, obj, threadpool, threadpool_batch);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_detachThreadPoolNative
  (JNIEnv* env, jobject obj) {
    UtilityManager::detachThreadPool(env, obj);
}

// ===== TIER 4: PERFORMANCE MONITORING & MODEL ARCHITECTURE =====

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getPerformanceDataNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getPerformanceData(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_printPerformanceDataNative
  (JNIEnv* env, jobject obj) {
    UtilityManager::printPerformanceData(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_resetPerformanceDataNative
  (JNIEnv* env, jobject obj) {
    UtilityManager::resetPerformanceData(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getModelLayerCountNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getModelLayerCount(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getModelTrainingContextSizeNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getModelTrainingContextSize(env, obj);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_hasEncoderNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::hasEncoder(env, obj);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_hasDecoderNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::hasDecoder(env, obj);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_getRopeTypeNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getRopeType(env, obj);
}

JNIEXPORT jfloat JNICALL Java_de_kherud_llama_LlamaModel_getRopeFrequencyScaleNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getRopeFrequencyScale(env, obj);
}

// Tier 5: Advanced model introspection & resource control

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getModelEmbeddingDimensionNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getModelEmbeddingDimension(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getModelAttentionHeadsNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getModelAttentionHeads(env, obj);
}

JNIEXPORT jlong JNICALL Java_de_kherud_llama_LlamaModel_getModelKeyValueHeadsNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getModelKeyValueHeads(env, obj);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_isRecurrentModelNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::isRecurrentModel(env, obj);
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_isDiffusionModelNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::isDiffusionModel(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setWarmupModeNative
  (JNIEnv* env, jobject obj, jboolean warmup) {
    UtilityManager::setWarmupMode(env, obj, warmup);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getFlashAttentionTypeNative
  (JNIEnv* env, jobject obj) {
    return UtilityManager::getFlashAttentionType(env, obj);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaUtils_initializeBackendNative
  (JNIEnv* env, jclass cls) {
    UtilityManager::initializeBackend(env, cls);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaUtils_freeBackendNative
  (JNIEnv* env, jclass cls) {
    UtilityManager::freeBackend(env, cls);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaUtils_initializeNumaNative
  (JNIEnv* env, jclass cls, jint strategy) {
    UtilityManager::initializeNuma(env, cls, strategy);
}

} // extern "C"