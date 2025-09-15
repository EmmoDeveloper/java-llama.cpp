#include "training_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

static std::mutex trainingMutex;
static std::unordered_map<jlong, std::unique_ptr<TrainingSession>> trainingSessions;
static jlong nextTrainingId = 1;

// Helper function to get context from model object
static llama_context* getContext(JNIEnv* env, jobject obj) {
	jclass cls = env->GetObjectClass(obj);
	jfieldID fieldId = env->GetFieldID(cls, "ctx", "J");
	if (!fieldId) {
		JNIErrorHandler::throw_runtime_exception(env, "Failed to get context field");
		return nullptr;
	}

	jlong ctxHandle = env->GetLongField(obj, fieldId);
	return reinterpret_cast<llama_context*>(ctxHandle);
}

TrainingSession* TrainingManager::getTrainingSession(jlong handle) {
    std::lock_guard<std::mutex> lock(trainingMutex);
    auto it = trainingSessions.find(handle);
    return (it != trainingSessions.end()) ? it->second.get() : nullptr;
}

void TrainingManager::cleanupTrainingSession(TrainingSession* session) {
    if (session) {
        session->is_active = false;
        // Note: We don't free model/context as they're managed externally
        session->model = nullptr;
        session->ctx = nullptr;
        session->optimizer_ctx = nullptr;
    }
}

jboolean TrainingManager::validateDataset(JNIEnv* env, jclass cls, jstring datasetPath) {
    JNI_TRY(env)

    if (!datasetPath) {
        JNIErrorHandler::throw_illegal_argument(env, "Dataset path cannot be null");
        return JNI_FALSE;
    }

    std::string pathStr = JniUtils::jstring_to_string(env, datasetPath);

    // Basic file existence and format validation
    std::ifstream file(pathStr);
    if (!file.is_open()) {
        return JNI_FALSE;
    }

    // Check if file has content and basic format
    std::string line;
    int lineCount = 0;
    while (std::getline(file, line) && lineCount < 10) {
        if (line.length() > 0) {
            lineCount++;
        }
    }
    file.close();

    // Dataset should have at least some lines
    return (lineCount > 0) ? JNI_TRUE : JNI_FALSE;

    JNI_CATCH_RET(env, JNI_FALSE)
}

jlong TrainingManager::prepareTraining(JNIEnv* env, jclass cls, jobject model, jobject params) {
    JNI_TRY(env)

    if (!model) {
        JNIErrorHandler::throw_illegal_argument(env, "Model cannot be null");
        return -1;
    }

    // Get model context
    jclass modelClass = env->GetObjectClass(model);
    jfieldID ctxField = env->GetFieldID(modelClass, "ctx", "J");
    if (!ctxField) {
        JNIErrorHandler::throw_illegal_state(env, "Cannot access model context");
        return -1;
    }

    jlong contextHandle = env->GetLongField(model, ctxField);
    llama_context* ctx = reinterpret_cast<llama_context*>(contextHandle);
    if (!ctx) {
        JNIErrorHandler::throw_illegal_state(env, "Invalid model context");
        return -1;
    }

    // Create training session
    auto session = std::make_unique<TrainingSession>();
    session->ctx = ctx;
    session->model = const_cast<llama_model*>(llama_get_model(ctx));
    session->current_epoch = 0;
    session->current_learning_rate = 1e-4f; // Default learning rate
    session->is_active = true;

    // Extract parameters if provided
    if (params) {
        jclass paramsClass = env->GetObjectClass(params);

        // Get learning rate
        jmethodID getLearningRateMethod = env->GetMethodID(paramsClass, "getLearningRate", "()F");
        if (getLearningRateMethod) {
            session->current_learning_rate = env->CallFloatMethod(params, getLearningRateMethod);
        }
    }

    // Store session and return handle
    std::lock_guard<std::mutex> lock(trainingMutex);
    jlong trainingId = nextTrainingId++;
    trainingSessions[trainingId] = std::move(session);

    return trainingId;

    JNI_CATCH_RET(env, -1)
}

jobject TrainingManager::trainEpoch(JNIEnv* env, jclass cls, jlong trainingHandle, jstring datasetPath, jobject callback) {
    JNI_TRY(env)

    TrainingSession* session = getTrainingSession(trainingHandle);
    if (!session || !session->is_active) {
        JNIErrorHandler::throw_illegal_state(env, "Invalid training session");
        return nullptr;
    }

    if (!datasetPath) {
        JNIErrorHandler::throw_illegal_argument(env, "Dataset path cannot be null");
        return nullptr;
    }

    std::string pathStr = JniUtils::jstring_to_string(env, datasetPath);

    // Load dataset
    std::vector<std::string> samples;
    if (!loadDataset(pathStr, samples)) {
        JNIErrorHandler::throw_runtime_exception(env, "Failed to load dataset");
        return nullptr;
    }

    if (samples.empty()) {
        JNIErrorHandler::throw_illegal_argument(env, "Dataset is empty");
        return nullptr;
    }

    auto startTime = std::chrono::steady_clock::now();

    // Simulate training process
    int totalSteps = samples.size();
    float totalLoss = 0.0f;

    for (int step = 0; step < totalSteps; step++) {
        // Simulate processing each sample
        // In a real implementation, this would:
        // 1. Tokenize the sample
        // 2. Forward pass through model
        // 3. Compute loss
        // 4. Backward pass and gradient computation
        // 5. Update parameters

        // For now, simulate with dummy loss calculation
        float stepLoss = 2.5f * exp(-0.1f * (session->current_epoch + (float)step / totalSteps));
        totalLoss += stepLoss;

        // Call progress callback every 10 steps
        if (callback && (step % 10 == 0 || step == totalSteps - 1)) {
            invokeProgressCallback(env, callback, session->current_epoch, step, stepLoss, session->current_learning_rate);
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    long trainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    float avgLoss = totalLoss / totalSteps;
    session->current_epoch++;

    // Decay learning rate slightly
    session->current_learning_rate *= 0.995f;

    return createTrainingMetrics(env, avgLoss, session->current_learning_rate, totalSteps, trainingTime);

    JNI_CATCH_RET(env, nullptr)
}

jobject TrainingManager::evaluate(JNIEnv* env, jclass cls, jlong trainingHandle, jstring validationDatasetPath) {
    JNI_TRY(env)

    TrainingSession* session = getTrainingSession(trainingHandle);
    if (!session || !session->is_active) {
        JNIErrorHandler::throw_illegal_state(env, "Invalid training session");
        return nullptr;
    }

    if (!validationDatasetPath) {
        JNIErrorHandler::throw_illegal_argument(env, "Validation dataset path cannot be null");
        return nullptr;
    }

    std::string pathStr = JniUtils::jstring_to_string(env, validationDatasetPath);

    // Load validation dataset
    std::vector<std::string> samples;
    if (!loadDataset(pathStr, samples)) {
        JNIErrorHandler::throw_runtime_exception(env, "Failed to load validation dataset");
        return nullptr;
    }

    if (samples.empty()) {
        JNIErrorHandler::throw_illegal_argument(env, "Validation dataset is empty");
        return nullptr;
    }

    // Simulate evaluation
    int totalSamples = samples.size();
    float totalLoss = 0.0f;
    int correctPredictions = 0;

    for (int i = 0; i < totalSamples; i++) {
        // Simulate evaluation of each sample
        float sampleLoss = 1.8f * exp(-0.05f * session->current_epoch);
        totalLoss += sampleLoss;

        // Simulate accuracy (improves with training)
        float accuracy_prob = 0.6f + 0.3f * (1.0f - exp(-0.1f * session->current_epoch));
        if ((float)rand() / RAND_MAX < accuracy_prob) {
            correctPredictions++;
        }
    }

    float avgLoss = totalLoss / totalSamples;
    float accuracy = (float)correctPredictions / totalSamples;
    float perplexity = computePerplexity(avgLoss);

    return createEvaluationMetrics(env, avgLoss, accuracy, perplexity, totalSamples);

    JNI_CATCH_RET(env, nullptr)
}

void TrainingManager::saveCheckpoint(JNIEnv* env, jclass cls, jlong trainingHandle, jstring checkpointPath) {
    JNI_TRY(env)

    TrainingSession* session = getTrainingSession(trainingHandle);
    if (!session || !session->is_active) {
        JNIErrorHandler::throw_illegal_state(env, "Invalid training session");
        return;
    }

    if (!checkpointPath) {
        JNIErrorHandler::throw_illegal_argument(env, "Checkpoint path cannot be null");
        return;
    }

    std::string pathStr = JniUtils::jstring_to_string(env, checkpointPath);

    // Save checkpoint (simplified implementation)
    std::ofstream checkpoint(pathStr);
    if (!checkpoint.is_open()) {
        JNIErrorHandler::throw_runtime_exception(env, "Failed to create checkpoint file");
        return;
    }

    checkpoint << "epoch=" << session->current_epoch << std::endl;
    checkpoint << "learning_rate=" << session->current_learning_rate << std::endl;
    checkpoint << "is_active=" << (session->is_active ? "true" : "false") << std::endl;

    checkpoint.close();

    JNI_CATCH_RET(env, )
}

void TrainingManager::loadCheckpoint(JNIEnv* env, jclass cls, jlong trainingHandle, jstring checkpointPath) {
    JNI_TRY(env)

    TrainingSession* session = getTrainingSession(trainingHandle);
    if (!session) {
        JNIErrorHandler::throw_illegal_state(env, "Invalid training session");
        return;
    }

    if (!checkpointPath) {
        JNIErrorHandler::throw_illegal_argument(env, "Checkpoint path cannot be null");
        return;
    }

    std::string pathStr = JniUtils::jstring_to_string(env, checkpointPath);

    // Load checkpoint (simplified implementation)
    std::ifstream checkpoint(pathStr);
    if (!checkpoint.is_open()) {
        JNIErrorHandler::throw_runtime_exception(env, "Failed to open checkpoint file");
        return;
    }

    std::string line;
    while (std::getline(checkpoint, line)) {
        if (line.find("epoch=") == 0) {
            session->current_epoch = std::stoi(line.substr(6));
        } else if (line.find("learning_rate=") == 0) {
            session->current_learning_rate = std::stof(line.substr(14));
        } else if (line.find("is_active=") == 0) {
            session->is_active = (line.substr(10) == "true");
        }
    }

    checkpoint.close();

    JNI_CATCH_RET(env, )
}

void TrainingManager::finishTraining(JNIEnv* env, jclass cls, jlong trainingHandle) {
    JNI_TRY(env)

    std::lock_guard<std::mutex> lock(trainingMutex);
    auto it = trainingSessions.find(trainingHandle);
    if (it != trainingSessions.end()) {
        cleanupTrainingSession(it->second.get());
        trainingSessions.erase(it);
    }

    JNI_CATCH_RET(env, )
}

// Helper methods

jobject TrainingManager::createTrainingMetrics(JNIEnv* env, float loss, float learningRate, int totalSteps, long trainingTime) {
    jclass metricsClass = env->FindClass("de/kherud/llama/LlamaTrainer$TrainingMetrics");
    if (!metricsClass) return nullptr;

    jmethodID constructor = env->GetMethodID(metricsClass, "<init>", "(FFIJ)V");
    if (!constructor) return nullptr;

    return env->NewObject(metricsClass, constructor, loss, learningRate, totalSteps, trainingTime);
}

jobject TrainingManager::createEvaluationMetrics(JNIEnv* env, float loss, float accuracy, float perplexity, int totalSamples) {
    jclass metricsClass = env->FindClass("de/kherud/llama/LlamaTrainer$EvaluationMetrics");
    if (!metricsClass) return nullptr;

    jmethodID constructor = env->GetMethodID(metricsClass, "<init>", "(FFFI)V");
    if (!constructor) return nullptr;

    return env->NewObject(metricsClass, constructor, loss, accuracy, perplexity, totalSamples);
}

void TrainingManager::invokeProgressCallback(JNIEnv* env, jobject callback, int epoch, int step, float loss, float learningRate) {
    if (!callback) return;

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onProgressMethod = env->GetMethodID(callbackClass, "onProgress", "(IIFF)V");
    if (onProgressMethod) {
        env->CallVoidMethod(callback, onProgressMethod, epoch, step, loss, learningRate);
    }
}

bool TrainingManager::loadDataset(const std::string& datasetPath, std::vector<std::string>& samples) {
    std::ifstream file(datasetPath);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            samples.push_back(line);
        }
    }

    file.close();
    return true;
}

float TrainingManager::computeLoss(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size()) {
        return 0.0f;
    }

    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
    }

    return loss / predictions.size();
}

float TrainingManager::computeAccuracy(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size()) {
        return 0.0f;
    }

    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (std::round(predictions[i]) == std::round(targets[i])) {
            correct++;
        }
    }

    return (float)correct / predictions.size();
}

float TrainingManager::computePerplexity(float loss) {
    return std::exp(loss);
}