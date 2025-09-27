#ifndef STABLE_DIFFUSION_MANAGER_H
#define STABLE_DIFFUSION_MANAGER_H

#include <jni.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include "stable-diffusion.h"

/**
 * Manager for Stable Diffusion contexts and operations.
 * Provides native integration between Java and stable-diffusion.cpp.
 */
class StableDiffusionManager {
public:
	struct GenerationParams {
		std::string prompt;
		std::string negative_prompt;
		int width = 768;
		int height = 768;
		int steps = 30;
		float cfg_scale = 7.0f;
		float slg_scale = 2.5f;
		int seed = -1;
		int sample_method = 1; // EULER
		bool clip_on_cpu = true;

		// ControlNet parameters
		std::vector<uint8_t> control_image_data;
		int control_image_width = 0;
		int control_image_height = 0;
		int control_image_channels = 3;
		float control_strength = 0.9f;

		// Image-to-image parameters
		std::vector<uint8_t> init_image_data;
		int init_image_width = 0;
		int init_image_height = 0;
		int init_image_channels = 3;
		float strength = 0.8f;

		// Inpainting parameters
		std::vector<uint8_t> mask_image_data;
		int mask_image_width = 0;
		int mask_image_height = 0;
		int mask_image_channels = 1;
	};

	struct GenerationResult {
		bool success = false;
		std::string error_message;
		std::unique_ptr<sd_image_t> image;
		int width = 0;
		int height = 0;
		float generation_time = 0.0f;
	};

	static StableDiffusionManager& getInstance();

	// Context management
	jlong createContext(const std::string& model_path,
						const std::string& clip_l_path = "",
						const std::string& clip_g_path = "",
						const std::string& t5xxl_path = "",
						bool keep_clip_on_cpu = true);

	jlong createContextWithControlNet(const std::string& model_path,
									  const std::string& clip_l_path = "",
									  const std::string& clip_g_path = "",
									  const std::string& t5xxl_path = "",
									  const std::string& control_net_path = "",
									  bool keep_clip_on_cpu = true,
									  bool keep_control_net_on_cpu = false);

	bool destroyContext(jlong handle);

	// Image generation
	GenerationResult generateImage(jlong handle, const GenerationParams& params);

	// Utility functions
	static bool isValidImageFormat(const std::string& path);
	static std::string getErrorMessage();

private:
	StableDiffusionManager() = default;
	~StableDiffusionManager();

	struct ContextData {
		std::unique_ptr<sd_ctx_t, void(*)(sd_ctx_t*)> context;
		std::string model_path;

		ContextData(sd_ctx_t* ctx) : context(ctx, [](sd_ctx_t* p) {
			if (p) free_sd_ctx(p);
		}) {}
	};

	std::mutex contexts_mutex_;
	std::unordered_map<jlong, std::unique_ptr<ContextData>> contexts_;
	jlong next_handle_ = 1;

	static thread_local std::string last_error_;
};

// JNI function declarations
extern "C" {

// Context management
JNIEXPORT jlong JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_createContext(
	JNIEnv* env, jclass clazz, jstring model_path, jstring clip_l_path,
	jstring clip_g_path, jstring t5xxl_path, jboolean keep_clip_on_cpu);

JNIEXPORT jlong JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_createContextWithControlNet(
	JNIEnv* env, jclass clazz, jstring model_path, jstring clip_l_path,
	jstring clip_g_path, jstring t5xxl_path, jstring control_net_path,
	jboolean keep_clip_on_cpu, jboolean keep_control_net_on_cpu);

JNIEXPORT jboolean JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_destroyContext(
	JNIEnv* env, jclass clazz, jlong handle);

// Image generation
JNIEXPORT jobject JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_generateImage(
	JNIEnv* env, jclass clazz, jlong handle, jstring prompt,
	jstring negative_prompt, jint width, jint height, jint steps,
	jfloat cfg_scale, jfloat slg_scale, jint seed, jint sample_method,
	jboolean clip_on_cpu);

JNIEXPORT jobject JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_generateImageAdvanced(
	JNIEnv* env, jclass clazz, jlong handle, jstring prompt,
	jstring negative_prompt, jint width, jint height, jint steps,
	jfloat cfg_scale, jfloat slg_scale, jint seed, jint sample_method,
	jboolean clip_on_cpu, jbyteArray control_image, jint control_image_width,
	jint control_image_height, jint control_image_channels, jfloat control_strength,
	jbyteArray init_image, jint init_image_width, jint init_image_height,
	jint init_image_channels, jfloat strength, jbyteArray mask_image,
	jint mask_image_width, jint mask_image_height, jint mask_image_channels);

// Utility functions
JNIEXPORT jstring JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_getSystemInfo(
	JNIEnv* env, jclass clazz);

JNIEXPORT jstring JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_getLastError(
	JNIEnv* env, jclass clazz);

// Preprocessing utilities
JNIEXPORT jboolean JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_preprocessCanny(
	JNIEnv* env, jclass clazz,
	jbyteArray imageData, jint width, jint height, jint channels,
	jfloat highThreshold, jfloat lowThreshold,
	jfloat weak, jfloat strong, jboolean inverse);

// Image upscaling
JNIEXPORT jlong JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_createUpscalerContext(
	JNIEnv* env, jclass clazz, jstring esrgan_path, jboolean offload_to_cpu,
	jboolean direct, jint threads);

JNIEXPORT jboolean JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_destroyUpscalerContext(
	JNIEnv* env, jclass clazz, jlong handle);

JNIEXPORT jobject JNICALL
Java_de_kherud_llama_diffusion_NativeStableDiffusion_upscaleImage(
	JNIEnv* env, jclass clazz, jlong handle, jbyteArray image_data,
	jint width, jint height, jint channels, jint upscale_factor);

}

#endif // STABLE_DIFFUSION_MANAGER_H