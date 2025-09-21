#pragma once

#include <jni.h>
#include "llama.h"
#include "ggml-opt.h"
#include "common.h"
#include <string>
#include <vector>

struct TrainingSession {
    const llama_model* model;
    llama_context* ctx;
    ggml_opt_context_t opt_ctx;
    ggml_opt_dataset_t dataset;
    llama_opt_params opt_params;
    lr_opt learning_rate_config;
    int current_epoch;
    int total_epochs;
    float current_learning_rate;
    bool is_active;
    std::vector<llama_token> tokens;

    TrainingSession() : model(nullptr), ctx(nullptr), opt_ctx(nullptr), dataset(nullptr),
                       current_epoch(0), total_epochs(1), current_learning_rate(0.0001f), is_active(false) {
        // Initialize learning rate configuration
        learning_rate_config.lr0 = 0.0001f;
        learning_rate_config.lr_min = -1;
        learning_rate_config.decay_epochs = -1;
        learning_rate_config.scale_epoch = 0;
        learning_rate_config.wd = 0.01f;
        learning_rate_config.epochs = 1;
        learning_rate_config.epoch = 0;
    }
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