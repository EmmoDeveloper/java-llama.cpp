#ifndef SCHEMA_GRAMMAR_MANAGER_H
#define SCHEMA_GRAMMAR_MANAGER_H

#include <jni.h>
#include <string>

class SchemaGrammarManager {
public:
	// Convert JSON schema to GBNF grammar
	static jbyteArray jsonSchemaToGrammarBytes(JNIEnv* env, jclass cls, jstring schema);
};

#endif // SCHEMA_GRAMMAR_MANAGER_H