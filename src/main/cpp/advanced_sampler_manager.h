#ifndef ADVANCED_SAMPLER_MANAGER_H
#define ADVANCED_SAMPLER_MANAGER_H

#include <jni.h>
#include "llama.h"

/**
 * Advanced Sampler Manager for comprehensive sampling strategies.
 * Provides Java JNI bindings for all llama.cpp sampler functions.
 */
class AdvancedSamplerManager {
public:
	// Basic samplers
	static jlong createGreedySampler(JNIEnv* env);
	static jlong createDistributionSampler(JNIEnv* env, jint seed);
	
	// Top-K and Top-P samplers
	static jlong createTopKSampler(JNIEnv* env, jint k);
	static jlong createTopPSampler(JNIEnv* env, jfloat p, jint minKeep);
	static jlong createMinPSampler(JNIEnv* env, jfloat p, jint minKeep);
	
	// Temperature samplers
	static jlong createTemperatureSampler(JNIEnv* env, jfloat temperature);
	static jlong createExtendedTemperatureSampler(JNIEnv* env, jfloat temp, jfloat delta, jfloat exponent);
	
	// Advanced samplers
	static jlong createTypicalSampler(JNIEnv* env, jfloat p, jint minKeep);
	static jlong createXtcSampler(JNIEnv* env, jfloat p, jfloat t, jint minKeep, jint seed);
	static jlong createTopNSigmaSampler(JNIEnv* env, jfloat n);
	
	// Mirostat samplers
	static jlong createMirostatSampler(JNIEnv* env, jint nVocab, jint seed, jfloat tau, jfloat eta, jint m);
	static jlong createMirostatV2Sampler(JNIEnv* env, jint seed, jfloat tau, jfloat eta);
	
	// Penalty and bias samplers
	static jlong createPenaltiesSampler(JNIEnv* env, jint penaltyLastN, 
		jfloat penaltyRepeat, jfloat penaltyFreq, jfloat penaltyPresent);
	static jlong createDrySampler(JNIEnv* env, jobject model, jint nCtxTrain, jfloat multiplier, jfloat base, 
		jint allowedLength, jint penaltyLastN, jintArray sequenceBreakers);
	static jlong createLogitBiasSampler(JNIEnv* env, jint nVocab, jint nLogitBias, jintArray biasTokens, jfloatArray biasValues);
	
	// Grammar and pattern samplers
	static jlong createGrammarSampler(JNIEnv* env, jobject model, jstring grammarStr, jstring rootRule);
	static jlong createInfillSampler(JNIEnv* env, jobject model);
	
	// Sampler chain management
	static jlong createSamplerChain(JNIEnv* env);
	static void addToSamplerChain(JNIEnv* env, jlong chainHandle, jlong samplerHandle);
	static jlong cloneSampler(JNIEnv* env, jlong samplerHandle);
	static void freeSampler(JNIEnv* env, jlong samplerHandle);
	
	// Sampling operations
	static jint sampleToken(JNIEnv* env, jobject obj, jlong samplerHandle);
	static void acceptToken(JNIEnv* env, jlong samplerHandle, jint token);
	static void resetSampler(JNIEnv* env, jlong samplerHandle);
	
	// Sampler configuration and info
	static jstring getSamplerName(JNIEnv* env, jlong samplerHandle);
	
private:
	// Helper methods
	static struct llama_context* getContext(JNIEnv* env, jobject obj);
	static const struct llama_vocab* getVocab(JNIEnv* env, jobject obj);
	static bool validateSamplerHandle(jlong handle);
};

#endif // ADVANCED_SAMPLER_MANAGER_H