#pragma once

#include <string>
#include <regex>

class PatternPreprocessor {
public:
	// Preprocess regex patterns for llama.cpp's constraint system
	static std::string preprocess(const std::string& pattern);
	
private:
	// Convert Unicode escape sequences (\uXXXX) to UTF-8
	static std::string process_unicode_escapes(const std::string& input);
	
	// Convert hex escape sequences (\xXX) to actual characters
	static std::string process_hex_escapes(const std::string& input);
	
	// Convert negated character classes [^...] to positive ones
	static std::string process_negated_char_classes(const std::string& input);
	
	// Helper to convert a codepoint to UTF-8
	static std::string codepoint_to_utf8(unsigned int codepoint);
};