#include "stable_diffusion_manager.h"
#include "jni_utils.h"
#include "jni_logger.h"
#include "jni_error_handler.h"

// Include stable-diffusion.cpp headers
#include "stable-diffusion.h"

#include <chrono>
#include <cstring>
#include <algorithm>
#include <vector>
#include <fstream>

// Thread-local error storage
thread_local std::string StableDiffusionManager::last_error_;

StableDiffusionManager& StableDiffusionManager::getInstance() {
	static StableDiffusionManager instance;
	return instance;
}

StableDiffusionManager::~StableDiffusionManager() {
	std::lock_guard<std::mutex> lock(contexts_mutex_);
	contexts_.clear();
}

jlong StableDiffusionManager::createContext(const std::string& model_path,
											const std::string& clip_l_path,
											const std::string& clip_g_path,
											const std::string& t5xxl_path,
											bool keep_clip_on_cpu) {
	try {
		// Validate model file exists and is readable
		std::ifstream file(model_path, std::ios::binary);
		if (!file.good()) {
			last_error_ = "Cannot access model file: " + model_path;
			printf("[SD_DEBUG] File access failed: %s\n", model_path.c_str());
			fflush(stdout);
			return 0;
		}
		file.close();

		// Initialize stable diffusion context parameters
		sd_ctx_params_t ctx_params;
		sd_ctx_params_init(&ctx_params);

		// Set model paths
		ctx_params.model_path = model_path.c_str();
		if (!clip_l_path.empty()) {
			ctx_params.clip_l_path = clip_l_path.c_str();
		}
		if (!clip_g_path.empty()) {
			ctx_params.clip_g_path = clip_g_path.c_str();
		}
		if (!t5xxl_path.empty()) {
			ctx_params.t5xxl_path = t5xxl_path.c_str();
		}

		// Configure performance settings
		ctx_params.keep_clip_on_cpu = keep_clip_on_cpu;
		ctx_params.n_threads = get_num_physical_cores();
		ctx_params.wtype = SD_TYPE_COUNT; // Auto-detect weight type from model
		ctx_params.rng_type = STD_DEFAULT_RNG;

		// Log detailed parameters before context creation
		printf("[SD_DEBUG] Creating SD context: model=%s, wtype=%d, threads=%d, keep_clip_cpu=%s\n",
			model_path.c_str(), ctx_params.wtype, ctx_params.n_threads,
			ctx_params.keep_clip_on_cpu ? "true" : "false");
		fflush(stdout);

		// Create the context
		sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
		if (!ctx) {
			// Try to get more specific error information
			std::string detailed_error = "Failed to create stable diffusion context for model: " + model_path;
			detailed_error += " (wtype=" + std::to_string(ctx_params.wtype) +
							  ", threads=" + std::to_string(ctx_params.n_threads) + ")";

			printf("[SD_DEBUG] Context creation failed: %s\n", detailed_error.c_str());
			fflush(stdout);
			last_error_ = detailed_error;
			return 0;
		}

		// Store context with unique handle
		std::lock_guard<std::mutex> lock(contexts_mutex_);
		jlong handle = next_handle_++;
		contexts_[handle] = std::make_unique<ContextData>(ctx);
		contexts_[handle]->model_path = model_path;

		JNILogger::log(JNILogger::Level::INFO,
			"Created stable diffusion context %ld for model: %s", handle, model_path.c_str());

		return handle;

	} catch (const std::exception& e) {
		last_error_ = "Exception creating stable diffusion context: " + std::string(e.what());
		return 0;
	}
}

jlong StableDiffusionManager::createContextWithControlNet(const std::string& model_path,
														  const std::string& clip_l_path,
														  const std::string& clip_g_path,
														  const std::string& t5xxl_path,
														  const std::string& control_net_path,
														  bool keep_clip_on_cpu,
														  bool keep_control_net_on_cpu) {
	try {
		// Validate model file exists and is readable
		std::ifstream file(model_path, std::ios::binary);
		if (!file.good()) {
			last_error_ = "Cannot access model file: " + model_path;
			printf("[SD_DEBUG] File access failed: %s\n", model_path.c_str());
			fflush(stdout);
			return 0;
		}
		file.close();

		// Initialize stable diffusion context parameters
		sd_ctx_params_t ctx_params;
		sd_ctx_params_init(&ctx_params);

		// Set model paths
		ctx_params.model_path = model_path.c_str();
		if (!clip_l_path.empty()) {
			ctx_params.clip_l_path = clip_l_path.c_str();
		}
		if (!clip_g_path.empty()) {
			ctx_params.clip_g_path = clip_g_path.c_str();
		}
		if (!t5xxl_path.empty()) {
			ctx_params.t5xxl_path = t5xxl_path.c_str();
		}

		// Set ControlNet path
		if (!control_net_path.empty()) {
			ctx_params.control_net_path = control_net_path.c_str();
		}

		// Set other parameters
		ctx_params.wtype = SD_TYPE_COUNT;  // Auto-detect weight type
		ctx_params.n_threads = 8;
		ctx_params.keep_clip_on_cpu = keep_clip_on_cpu;
		ctx_params.keep_control_net_on_cpu = keep_control_net_on_cpu;

		printf("[SD_DEBUG] JNI createContextWithControlNet called!\n");
		printf("[SD_DEBUG] Creating SD context with ControlNet: model=%s, controlnet=%s, wtype=%d, threads=%d, keep_clip_cpu=%s, keep_controlnet_cpu=%s\n",
			model_path.c_str(), control_net_path.c_str(), ctx_params.wtype, ctx_params.n_threads,
			keep_clip_on_cpu ? "true" : "false", keep_control_net_on_cpu ? "true" : "false");
		fflush(stdout);

		// Create the context
		sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
		if (!ctx) {
			// Try to get more specific error information
			std::string detailed_error = "Failed to create stable diffusion context with ControlNet for model: " + model_path;
			detailed_error += " (wtype=" + std::to_string(ctx_params.wtype) +
							  ", threads=" + std::to_string(ctx_params.n_threads) +
							  ", controlnet=" + control_net_path + ")";

			printf("[SD_DEBUG] Context creation with ControlNet failed: %s\n", detailed_error.c_str());
			fflush(stdout);
			last_error_ = detailed_error;
			return 0;
		}

		// Store context with unique handle
		std::lock_guard<std::mutex> lock(contexts_mutex_);
		jlong handle = next_handle_++;
		contexts_[handle] = std::make_unique<ContextData>(ctx);
		contexts_[handle]->model_path = model_path;

		JNILogger::log(JNILogger::Level::INFO,
			"Created stable diffusion context with ControlNet %ld for model: %s", handle, model_path.c_str());

		return handle;

	} catch (const std::exception& e) {
		last_error_ = "Exception creating stable diffusion context with ControlNet: " + std::string(e.what());
		return 0;
	}
}

bool StableDiffusionManager::destroyContext(jlong handle) {
	std::lock_guard<std::mutex> lock(contexts_mutex_);
	auto it = contexts_.find(handle);
	if (it != contexts_.end()) {
		JNILogger::log(JNILogger::Level::INFO, "Destroying stable diffusion context %ld", handle);
		contexts_.erase(it);
		return true;
	}
	return false;
}

StableDiffusionManager::GenerationResult StableDiffusionManager::generateImage(
	jlong handle, const GenerationParams& params) {

	GenerationResult result;
	auto start_time = std::chrono::high_resolution_clock::now();

	try {
		// Get context
		std::lock_guard<std::mutex> lock(contexts_mutex_);
		auto it = contexts_.find(handle);
		if (it == contexts_.end()) {
			result.error_message = "Invalid stable diffusion context handle: " + std::to_string(handle);
			return result;
		}

		sd_ctx_t* ctx = it->second->context.get();
		if (!ctx) {
			result.error_message = "Stable diffusion context is null";
			return result;
		}

		// Initialize image generation parameters
		sd_img_gen_params_t gen_params;
		sd_img_gen_params_init(&gen_params);

		// Set generation parameters
		gen_params.prompt = params.prompt.c_str();
		gen_params.negative_prompt = params.negative_prompt.c_str();
		gen_params.width = params.width;
		gen_params.height = params.height;
		gen_params.seed = params.seed > 0 ? params.seed : -1;
		gen_params.batch_count = 1;
		gen_params.clip_skip = -1; // Use default

		// Configure sampling parameters
		sd_sample_params_init(&gen_params.sample_params);
		gen_params.sample_params.sample_steps = params.steps;
		gen_params.sample_params.sample_method = static_cast<sample_method_t>(params.sample_method);
		gen_params.sample_params.scheduler = DEFAULT;

		// Configure guidance parameters
		gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;
		gen_params.sample_params.guidance.slg.scale = params.slg_scale;

		// Configure ControlNet if provided
		sd_image_t control_image = {};
		if (!params.control_image_data.empty()) {
			control_image.width = params.control_image_width;
			control_image.height = params.control_image_height;
			control_image.channel = params.control_image_channels;
			control_image.data = const_cast<uint8_t*>(params.control_image_data.data());
			gen_params.control_image = control_image;
			gen_params.control_strength = params.control_strength;
			JNILogger::log(JNILogger::Level::INFO, "Using ControlNet: %dx%d, strength=%.2f",
						   control_image.width, control_image.height, params.control_strength);
		}

		// Configure img2img if provided
		sd_image_t init_image = {};
		if (!params.init_image_data.empty()) {
			init_image.width = params.init_image_width;
			init_image.height = params.init_image_height;
			init_image.channel = params.init_image_channels;
			init_image.data = const_cast<uint8_t*>(params.init_image_data.data());
			gen_params.init_image = init_image;
			gen_params.strength = params.strength;
			JNILogger::log(JNILogger::Level::INFO, "Using img2img: %dx%d, strength=%.2f",
						   init_image.width, init_image.height, params.strength);
		}

		// Configure inpainting mask if provided
		sd_image_t mask_image = {};
		if (!params.mask_image_data.empty()) {
			// Inpainting requires both an init image and a mask
			if (params.init_image_data.empty()) {
				result.error_message = "Inpainting requires an init image along with the mask";
				JNILogger::log(JNILogger::Level::ERROR, "Inpainting mask provided without init image");
				return result;
			}

			// Check if this is an SD3 model (SD3 doesn't support inpainting)
			// SD3 models typically have "sd3" in the model path
			std::string model_path = it->second->model_path;
			std::transform(model_path.begin(), model_path.end(), model_path.begin(), ::tolower);
			if (model_path.find("sd3") != std::string::npos ||
				model_path.find("sd_3") != std::string::npos ||
				model_path.find("stable-diffusion-3") != std::string::npos) {
				result.error_message = "SD3 models do not support inpainting. Use SD1.5-inpaint, SD2-inpaint, or SDXL-inpaint models instead.";
				JNILogger::log(JNILogger::Level::ERROR, "Attempted to use inpainting with SD3 model: %s", model_path.c_str());
				return result;
			}

			mask_image.width = params.mask_image_width;
			mask_image.height = params.mask_image_height;
			mask_image.channel = params.mask_image_channels;
			mask_image.data = const_cast<uint8_t*>(params.mask_image_data.data());
			gen_params.mask_image = mask_image;
			JNILogger::log(JNILogger::Level::INFO, "Using inpainting: %dx%d, channels=%d",
						   mask_image.width, mask_image.height, mask_image.channel);
		}

		JNILogger::log(JNILogger::Level::INFO,
			"Generating image %dx%d, steps=%d, cfg=%.1f, slg=%.1f, controlNet=%s, img2img=%s, inpainting=%s, prompt='%s'",
			params.width, params.height, params.steps, params.cfg_scale, params.slg_scale,
			!params.control_image_data.empty() ? "yes" : "no",
			!params.init_image_data.empty() ? "yes" : "no",
			!params.mask_image_data.empty() ? "yes" : "no",
			params.prompt.c_str());

		// Generate the image
		sd_image_t* image = generate_image(ctx, &gen_params);

		if (!image) {
			result.error_message = "Image generation failed - generate_image returned null";
			return result;
		}

		if (!image->data) {
			result.error_message = "Image generation failed - image data is null";
			return result;
		}

		// Calculate generation time
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

		// Store result
		result.success = true;
		result.image = std::unique_ptr<sd_image_t>(image);
		result.width = image->width;
		result.height = image->height;
		result.generation_time = duration.count() / 1000.0f;

		JNILogger::log(JNILogger::Level::INFO,
			"Image generated successfully in %.2f seconds (%dx%d, %d channels)",
			result.generation_time, result.width, result.height, image->channel);

		return result;

	} catch (const std::exception& e) {
		result.error_message = "Exception during image generation: " + std::string(e.what());
		return result;
	}
}

bool StableDiffusionManager::isValidImageFormat(const std::string& path) {
	const std::string ext = path.substr(path.find_last_of('.') + 1);
	return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "bmp" || ext == "tga";
}

std::string StableDiffusionManager::getErrorMessage() {
	return last_error_;
}

// JNI implementations

extern "C" {

JNIEXPORT jlong JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_createContext(
	JNIEnv* env, jclass clazz, jstring model_path, jstring clip_l_path,
	jstring clip_g_path, jstring t5xxl_path, jboolean keep_clip_on_cpu) {

	printf("[SD_DEBUG] JNI createContext called!\n");
	fflush(stdout);

	try {
		std::string model_path_str = JniUtils::jstring_to_string(env, model_path);
		std::string clip_l_str = clip_l_path ? JniUtils::jstring_to_string(env, clip_l_path) : "";
		std::string clip_g_str = clip_g_path ? JniUtils::jstring_to_string(env, clip_g_path) : "";
		std::string t5xxl_str = t5xxl_path ? JniUtils::jstring_to_string(env, t5xxl_path) : "";

		return StableDiffusionManager::getInstance().createContext(
			model_path_str, clip_l_str, clip_g_str, t5xxl_str, keep_clip_on_cpu);

	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException",
			"Failed to create stable diffusion context: " + std::string(e.what()));
		return 0;
	}
}

JNIEXPORT jboolean JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_destroyContext(
	JNIEnv* env, jclass clazz, jlong handle) {

	try {
		return StableDiffusionManager::getInstance().destroyContext(handle);
	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException",
			"Failed to destroy stable diffusion context: " + std::string(e.what()));
		return false;
	}
}

JNIEXPORT jobject JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_generateImage(
	JNIEnv* env, jclass clazz, jlong handle, jstring prompt,
	jstring negative_prompt, jint width, jint height, jint steps,
	jfloat cfg_scale, jfloat slg_scale, jint seed, jint sample_method,
	jboolean clip_on_cpu) {

	try {
		StableDiffusionManager::GenerationParams params;
		params.prompt = JniUtils::jstring_to_string(env, prompt);
		params.negative_prompt = negative_prompt ? JniUtils::jstring_to_string(env, negative_prompt) : "";
		params.width = width;
		params.height = height;
		params.steps = steps;
		params.cfg_scale = cfg_scale;
		params.slg_scale = slg_scale;
		params.seed = seed;
		params.sample_method = sample_method;
		params.clip_on_cpu = clip_on_cpu;

		auto result = StableDiffusionManager::getInstance().generateImage(handle, params);

		// Create Java result object
		jclass resultClass = env->FindClass("de/kherud/llama/diffusion/StableDiffusionResult");
		if (!resultClass) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/ClassNotFoundException",
				"Could not find StableDiffusionResult class");
			return nullptr;
		}

		jmethodID constructor = env->GetMethodID(resultClass, "<init>", "(ZLjava/lang/String;[BIIF)V");
		if (!constructor) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/NoSuchMethodException",
				"Could not find StableDiffusionResult constructor");
			return nullptr;
		}

		jstring errorMsg = result.error_message.empty() ? nullptr : env->NewStringUTF(result.error_message.c_str());
		jbyteArray imageData = nullptr;

		if (result.success && result.image) {
			sd_image_t* img = result.image.get();
			size_t data_size = img->width * img->height * img->channel;
			imageData = env->NewByteArray(data_size);
			if (imageData) {
				env->SetByteArrayRegion(imageData, 0, data_size, reinterpret_cast<const jbyte*>(img->data));
			}
		}

		return env->NewObject(resultClass, constructor,
			static_cast<jboolean>(result.success),
			errorMsg,
			imageData,
			static_cast<jint>(result.width),
			static_cast<jint>(result.height),
			static_cast<jfloat>(result.generation_time));

	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException",
			"Failed to generate image: " + std::string(e.what()));
		return nullptr;
	}
}

JNIEXPORT jstring JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_getSystemInfo(
	JNIEnv* env, jclass clazz) {

	try {
		const char* info = sd_get_system_info();
		return env->NewStringUTF(info ? info : "System info not available");
	} catch (const std::exception& e) {
		return env->NewStringUTF("Error getting system info");
	}
}

JNIEXPORT jstring JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_getLastError(
	JNIEnv* env, jclass clazz) {

	try {
		std::string error = StableDiffusionManager::getErrorMessage();
		return env->NewStringUTF(error.c_str());
	} catch (const std::exception& e) {
		return env->NewStringUTF("Error getting last error message");
	}
}

JNIEXPORT jlong JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_createContextWithControlNet(
	JNIEnv* env, jclass clazz, jstring model_path, jstring clip_l_path,
	jstring clip_g_path, jstring t5xxl_path, jstring control_net_path,
	jboolean keep_clip_on_cpu, jboolean keep_control_net_on_cpu) {

	try {
		std::string model_path_str = JniUtils::jstring_to_string(env, model_path);
		std::string clip_l_path_str = clip_l_path ? JniUtils::jstring_to_string(env, clip_l_path) : "";
		std::string clip_g_path_str = clip_g_path ? JniUtils::jstring_to_string(env, clip_g_path) : "";
		std::string t5xxl_path_str = t5xxl_path ? JniUtils::jstring_to_string(env, t5xxl_path) : "";
		std::string control_net_path_str = control_net_path ? JniUtils::jstring_to_string(env, control_net_path) : "";

		return StableDiffusionManager::getInstance().createContextWithControlNet(
			model_path_str, clip_l_path_str, clip_g_path_str, t5xxl_path_str,
			control_net_path_str, static_cast<bool>(keep_clip_on_cpu),
			static_cast<bool>(keep_control_net_on_cpu)
		);
	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException", e.what());
		return 0;
	}
}

JNIEXPORT jobject JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_generateImageAdvanced(
	JNIEnv* env, jclass clazz, jlong handle, jstring prompt,
	jstring negative_prompt, jint width, jint height, jint steps,
	jfloat cfg_scale, jfloat slg_scale, jint seed, jint sample_method,
	jboolean clip_on_cpu, jbyteArray control_image, jint control_image_width,
	jint control_image_height, jint control_image_channels, jfloat control_strength,
	jbyteArray init_image, jint init_image_width, jint init_image_height,
	jint init_image_channels, jfloat strength, jbyteArray mask_image,
	jint mask_image_width, jint mask_image_height, jint mask_image_channels) {

	try {
		// Extract parameters
		StableDiffusionManager::GenerationParams params;
		params.prompt = JniUtils::jstring_to_string(env, prompt);
		params.negative_prompt = negative_prompt ? JniUtils::jstring_to_string(env, negative_prompt) : "";
		params.width = width;
		params.height = height;
		params.steps = steps;
		params.cfg_scale = cfg_scale;
		params.slg_scale = slg_scale;
		params.seed = seed;
		params.sample_method = sample_method;
		params.clip_on_cpu = static_cast<bool>(clip_on_cpu);

		// Extract ControlNet image data if provided
		if (control_image && control_image_width > 0 && control_image_height > 0) {
			jsize control_len = env->GetArrayLength(control_image);
			jbyte* control_data = env->GetByteArrayElements(control_image, nullptr);
			if (control_data && control_len > 0) {
				params.control_image_data.assign(
					reinterpret_cast<uint8_t*>(control_data),
					reinterpret_cast<uint8_t*>(control_data) + control_len
				);
				params.control_image_width = control_image_width;
				params.control_image_height = control_image_height;
				params.control_image_channels = control_image_channels;
				params.control_strength = control_strength;
				env->ReleaseByteArrayElements(control_image, control_data, JNI_ABORT);
			}
		}

		// Extract init image data if provided
		if (init_image && init_image_width > 0 && init_image_height > 0) {
			jsize init_len = env->GetArrayLength(init_image);
			jbyte* init_data = env->GetByteArrayElements(init_image, nullptr);
			if (init_data && init_len > 0) {
				params.init_image_data.assign(
					reinterpret_cast<uint8_t*>(init_data),
					reinterpret_cast<uint8_t*>(init_data) + init_len
				);
				params.init_image_width = init_image_width;
				params.init_image_height = init_image_height;
				params.init_image_channels = init_image_channels;
				params.strength = strength;
				env->ReleaseByteArrayElements(init_image, init_data, JNI_ABORT);
			}
		}

		// Handle mask image for inpainting
		if (mask_image != nullptr) {
			jsize mask_len = env->GetArrayLength(mask_image);
			if (mask_len > 0) {
				jbyte* mask_data = env->GetByteArrayElements(mask_image, nullptr);
				params.mask_image_data.assign(
					reinterpret_cast<uint8_t*>(mask_data),
					reinterpret_cast<uint8_t*>(mask_data) + mask_len
				);
				params.mask_image_width = mask_image_width;
				params.mask_image_height = mask_image_height;
				params.mask_image_channels = mask_image_channels;
				env->ReleaseByteArrayElements(mask_image, mask_data, JNI_ABORT);
			}
		}

		// Generate image
		auto result = StableDiffusionManager::getInstance().generateImage(handle, params);

		// Create Java result object
		jclass resultClass = env->FindClass("de/kherud/llama/diffusion/StableDiffusionResult");
		if (!resultClass) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/ClassNotFoundException", "Could not find StableDiffusionResult class");
			return nullptr;
		}

		jmethodID constructor = env->GetMethodID(resultClass, "<init>", "(ZLjava/lang/String;[BIIF)V");
		if (!constructor) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/NoSuchMethodException", "Could not find StableDiffusionResult constructor");
			return nullptr;
		}

		jstring errorMessage = result.success ? nullptr : env->NewStringUTF(result.error_message.c_str());
		jbyteArray imageData = nullptr;

		if (result.success && result.image && result.image->data) {
			// Convert image data to byte array
			size_t dataSize = result.width * result.height * result.image->channel;
			imageData = env->NewByteArray(static_cast<jsize>(dataSize));
			env->SetByteArrayRegion(imageData, 0, static_cast<jsize>(dataSize),
									reinterpret_cast<const jbyte*>(result.image->data));
		}

		return env->NewObject(resultClass, constructor,
							  static_cast<jboolean>(result.success),
							  errorMessage,
							  imageData,
							  static_cast<jint>(result.width),
							  static_cast<jint>(result.height),
							  static_cast<jfloat>(result.generation_time));

	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException", e.what());
		return nullptr;
	}
}

JNIEXPORT jboolean JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_preprocessCanny(
	JNIEnv* env, jclass clazz,
	jbyteArray imageData, jint width, jint height, jint channels,
	jfloat highThreshold, jfloat lowThreshold,
	jfloat weak, jfloat strong, jboolean inverse) {

	jbyte* dataPtr = nullptr;
	uint8_t* mallocedData = nullptr;

	try {
		if (!imageData) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/IllegalArgumentException",
												  "Image data cannot be null");
			return JNI_FALSE;
		}

		if (width <= 0 || height <= 0 || channels <= 0) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/IllegalArgumentException",
												  "Invalid image dimensions");
			return JNI_FALSE;
		}

		// Get image data from Java byte array
		jsize dataLength = env->GetArrayLength(imageData);
		dataPtr = env->GetByteArrayElements(imageData, nullptr);

		if (!dataPtr) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException",
												  "Failed to get image data");
			return JNI_FALSE;
		}

		// Verify data size matches dimensions
		jsize expectedSize = width * height * channels;
		if (dataLength != expectedSize) {
			env->ReleaseByteArrayElements(imageData, dataPtr, JNI_ABORT);
			JNIErrorHandler::throw_java_exception(env, "java/lang/IllegalArgumentException",
												  "Image data size does not match dimensions");
			return JNI_FALSE;
		}

		// Create sd_image_t struct with copied data
		// preprocess_canny will free() the image.data, so we need to malloc a copy
		sd_image_t image;
		image.width = static_cast<uint32_t>(width);
		image.height = static_cast<uint32_t>(height);
		image.channel = static_cast<uint32_t>(channels);
		mallocedData = static_cast<uint8_t*>(malloc(dataLength));
		image.data = mallocedData;

		if (!image.data) {
			env->ReleaseByteArrayElements(imageData, dataPtr, JNI_ABORT);
			JNIErrorHandler::throw_java_exception(env, "java/lang/OutOfMemoryError",
												  "Failed to allocate memory for image copy");
			return JNI_FALSE;
		}

		// Copy data from Java array to malloc'd buffer
		memcpy(image.data, dataPtr, dataLength);

		// Apply Canny edge detection (this will free image.data and allocate new data)
		// After this call, mallocedData is invalid (freed by preprocess_canny)
		mallocedData = nullptr; // Mark as freed to prevent double free in exception handler

		bool success = preprocess_canny(image,
										static_cast<float>(highThreshold),
										static_cast<float>(lowThreshold),
										static_cast<float>(weak),
										static_cast<float>(strong),
										static_cast<bool>(inverse));

		// Copy processed data back to Java array
		if (success && image.data) {
			env->SetByteArrayRegion(imageData, 0, dataLength,
									reinterpret_cast<const jbyte*>(image.data));
			free(image.data); // Free the data allocated by preprocess_canny
		}

		// Release the array
		env->ReleaseByteArrayElements(imageData, dataPtr, success ? 0 : JNI_ABORT);

		return success ? JNI_TRUE : JNI_FALSE;

	} catch (const std::exception& e) {
		// Clean up any allocated memory
		if (mallocedData) {
			free(mallocedData);
		}
		if (dataPtr) {
			env->ReleaseByteArrayElements(imageData, dataPtr, JNI_ABORT);
		}
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException", e.what());
		return JNI_FALSE;
	}
}

// Helper function to create UpscaleResult objects
namespace {
	jobject create_upscale_result(JNIEnv* env, bool success, jbyteArray image_data,
								  int width, int height, int channels, const char* error_message) {
		jclass resultClass = env->FindClass("de/kherud/llama/diffusion/UpscaleResult");
		if (!resultClass) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/ClassNotFoundException", "Could not find UpscaleResult class");
			return nullptr;
		}

		if (success) {
			jmethodID successMethod = env->GetStaticMethodID(resultClass, "success", "([BIII)Lde/kherud/llama/diffusion/UpscaleResult;");
			if (!successMethod) {
				JNIErrorHandler::throw_java_exception(env, "java/lang/NoSuchMethodException", "Could not find UpscaleResult.success method");
				return nullptr;
			}
			return env->CallStaticObjectMethod(resultClass, successMethod, image_data, width, height, channels);
		} else {
			jmethodID failureMethod = env->GetStaticMethodID(resultClass, "failure", "(Ljava/lang/String;)Lde/kherud/llama/diffusion/UpscaleResult;");
			if (!failureMethod) {
				JNIErrorHandler::throw_java_exception(env, "java/lang/NoSuchMethodException", "Could not find UpscaleResult.failure method");
				return nullptr;
			}
			jstring errorString = env->NewStringUTF(error_message ? error_message : "Unknown error");
			return env->CallStaticObjectMethod(resultClass, failureMethod, errorString);
		}
	}
}

// Image upscaling functions
JNIEXPORT jlong JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_createUpscalerContext(
	JNIEnv* env, jclass clazz, jstring esrgan_path, jboolean offload_to_cpu,
	jboolean direct, jint threads) {

	try {
		if (!esrgan_path) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/IllegalArgumentException", "ESRGAN path cannot be null");
			return 0;
		}

		const char* esrgan_path_cstr = env->GetStringUTFChars(esrgan_path, nullptr);
		if (!esrgan_path_cstr) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/OutOfMemoryError", "Failed to get ESRGAN path string");
			return 0;
		}

		// Create upscaler context using stable-diffusion.cpp API
		upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(
			esrgan_path_cstr,
			offload_to_cpu == JNI_TRUE,
			direct == JNI_TRUE,
			static_cast<int>(threads)
		);

		env->ReleaseStringUTFChars(esrgan_path, esrgan_path_cstr);

		if (!upscaler_ctx) {
			JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException",
				"Failed to create upscaler context");
			return 0;
		}

		return reinterpret_cast<jlong>(upscaler_ctx);

	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException", e.what());
		return 0;
	}
}

JNIEXPORT jboolean JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_destroyUpscalerContext(
	JNIEnv* env, jclass clazz, jlong handle) {

	try {
		if (handle == 0) {
			return JNI_FALSE;
		}

		upscaler_ctx_t* upscaler_ctx = reinterpret_cast<upscaler_ctx_t*>(handle);
		free_upscaler_ctx(upscaler_ctx);

		return JNI_TRUE;

	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException", e.what());
		return JNI_FALSE;
	}
}

JNIEXPORT jobject JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_upscaleImage(
	JNIEnv* env, jclass clazz, jlong handle, jbyteArray image_data,
	jint width, jint height, jint channels, jint upscale_factor) {

	try {
		if (handle == 0) {
			return create_upscale_result(env, false, nullptr, 0, 0, 0, "Invalid upscaler handle");
		}

		if (!image_data) {
			return create_upscale_result(env, false, nullptr, 0, 0, 0, "Image data cannot be null");
		}

		upscaler_ctx_t* upscaler_ctx = reinterpret_cast<upscaler_ctx_t*>(handle);

		// Get image data from Java array
		jsize data_length = env->GetArrayLength(image_data);
		jbyte* data_ptr = env->GetByteArrayElements(image_data, nullptr);

		if (!data_ptr) {
			return create_upscale_result(env, false, nullptr, 0, 0, 0, "Failed to get image data");
		}

		// Create sd_image_t structure
		sd_image_t input_image;
		input_image.width = static_cast<uint32_t>(width);
		input_image.height = static_cast<uint32_t>(height);
		input_image.channel = static_cast<uint32_t>(channels);
		input_image.data = reinterpret_cast<uint8_t*>(data_ptr);

		// Perform upscaling
		sd_image_t result_image = upscale(upscaler_ctx, input_image, static_cast<uint32_t>(upscale_factor));

		// Release input data
		env->ReleaseByteArrayElements(image_data, data_ptr, JNI_ABORT);

		if (!result_image.data) {
			return create_upscale_result(env, false, nullptr, 0, 0, 0, "Upscaling failed");
		}

		// Copy result data to Java array
		int result_width = static_cast<int>(result_image.width);
		int result_height = static_cast<int>(result_image.height);
		int result_channels = static_cast<int>(result_image.channel);
		int result_size = result_width * result_height * result_channels;

		jbyteArray result_array = env->NewByteArray(result_size);
		if (!result_array) {
			free(result_image.data);
			return create_upscale_result(env, false, nullptr, 0, 0, 0, "Failed to create result array");
		}

		env->SetByteArrayRegion(result_array, 0, result_size, reinterpret_cast<jbyte*>(result_image.data));

		// Free the result image data (allocated by stable-diffusion.cpp)
		free(result_image.data);

		return create_upscale_result(env, true, result_array, result_width, result_height, result_channels, nullptr);

	} catch (const std::exception& e) {
		JNIErrorHandler::throw_java_exception(env, "java/lang/RuntimeException", e.what());
		return nullptr;
	}
}

}