#pragma once

#include <string>
#include <set>

class GrammarProcessor {
public:
	static std::string preprocess_grammar(const std::string& grammar);
	static std::string negate_character_class(const std::string& char_class);
	static std::set<char> extract_characters(const std::string& inside_brackets);
	static std::string unicode_to_utf8(unsigned int codepoint);
};