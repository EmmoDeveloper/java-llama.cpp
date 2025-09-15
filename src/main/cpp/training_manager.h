#pragma once

#include <jni.h>
#include "llama.h"
#include <string>
#include <vector>

struct TrainingSession {
    llama_model* model;
    llama_context* ctx;
    void* optimizer_ctx;
    int current_epoch;
    float current_learning_rate;
    bool is_active;

    TrainingSession() : model(nullptr), ctx(nullptr), optimizer_ctx(nullptr),
                       current_epoch(0), current_learning_rate(0.0f), is_active(false) {}
};

class TrainingManager {
public:
    // Training lifecycle
    static jboolean validateDataset(JNIEnv* env, jclass cls, jstring datasetPath);
    static jlong prepareTraining(JNIEnv* env, jclass cls, jobject model, jobject params);
    static jobject trainEpoch(JNIEnv* env, jclass cls, jlong trainingHandle, jstring datasetPath, jobject callback);
    static jobject evaluate(JNIEnv* env, jclass cls, jlong trainingHandle, jstring validationDatasetPath);
    static void saveCheckpoint(JNIEnv* env, jclass cls, jlong trainingHandle, jstring checkpointPath);
    static void loadCheckpoint(JNIEnv* env, jclass cls, jlong trainingHandle, jstring checkpointPath);
    static void finishTraining(JNIEnv* env, jclass cls, jlong trainingHandle);

private:
    static TrainingSession* getTrainingSession(jlong handle);
    static void cleanupTrainingSession(TrainingSession* session);
    static jobject createTrainingMetrics(JNIEnv* env, float loss, float learningRate, int totalSteps, long trainingTime);
    static jobject createEvaluationMetrics(JNIEnv* env, float loss, float accuracy, float perplexity, int totalSamples);
    static void invokeProgressCallback(JNIEnv* env, jobject callback, int epoch, int step, float loss, float learningRate);
    static bool loadDataset(const std::string& datasetPath, std::vector<std::string>& samples);
    static float computeLoss(const std::vector<float>& predictions, const std::vector<float>& targets);
    static float computeAccuracy(const std::vector<float>& predictions, const std::vector<float>& targets);
    static float computePerplexity(float loss);
};