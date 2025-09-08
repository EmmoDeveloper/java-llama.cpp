#include "schema_grammar_manager.h"
#include "jni_utils.h"
#include "jni_error_handler.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <string>

jbyteArray SchemaGrammarManager::jsonSchemaToGrammarBytes(JNIEnv* env, jclass cls, jstring schema) {
	JNI_TRY(env)
	
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
	
	JNI_CATCH_RET(env, nullptr)
}