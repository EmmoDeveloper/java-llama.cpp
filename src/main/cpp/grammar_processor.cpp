#include "grammar_processor.h"
#include <sstream>
#include <iostream>
#include <cstdio>

std::string GrammarProcessor::preprocess_grammar(const std::string& grammar) {
	std::string result;
	result.reserve(grammar.size() * 2); // Extra space for replacements
	
	for (size_t i = 0; i < grammar.size(); ++i) {
		// Check for negative character class [^...]
		if (i + 2 < grammar.size() && grammar[i] == '[' && grammar[i + 1] == '^') {
			// Find the closing ]
			size_t j = i + 2;
			int bracket_depth = 1;
			while (j < grammar.size() && bracket_depth > 0) {
				if (grammar[j] == '\\' && j + 1 < grammar.size()) {
					j += 2; // Skip escaped character
				} else if (grammar[j] == '[') {
					bracket_depth++;
					j++;
				} else if (grammar[j] == ']') {
					bracket_depth--;
					j++;
				} else {
					j++;
				}
			}
			
			if (bracket_depth == 0) {
				// Found matching closing bracket
				std::string negated_chars = grammar.substr(i + 2, j - i - 3);
				std::string positive_chars = negate_character_class(negated_chars);
				result += "[" + positive_chars + "]";
				i = j - 1; // Skip to end of character class
				continue;
			}
		}
		
		// Handle Unicode escapes
		if (i + 5 < grammar.size() && grammar.substr(i, 2) == "\\u") {
			std::string hex_str = grammar.substr(i + 2, 4);
			bool is_hex = true;
			for (char c : hex_str) {
				if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
					is_hex = false;
					break;
				}
			}
			
			if (is_hex) {
				unsigned int codepoint = 0;
				sscanf(hex_str.c_str(), "%x", &codepoint);
				
				// Handle specific problematic Unicode characters
				if (codepoint == 0x2028 || codepoint == 0x2029) {
					// These are LINE SEPARATOR and PARAGRAPH SEPARATOR
					// Skip them entirely as they're rarely used and cause grammar parsing issues
					i += 5; // Skip the \uXXXX sequence
					continue;
				} else if (codepoint <= 0x7F) {
					// ASCII range - convert directly
					result += static_cast<char>(codepoint);
				} else if (codepoint <= 0x7FF) {
					// Two-byte UTF-8 sequence
					result += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
					result += static_cast<char>(0x80 | (codepoint & 0x3F));
				} else if (codepoint <= 0xFFFF) {
					// Three-byte UTF-8 sequence
					result += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
					result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
					result += static_cast<char>(0x80 | (codepoint & 0x3F));
				}
				
				i += 5; // Skip the \uXXXX sequence
				continue;
			}
		}
		
		// Handle hex escapes
		if (i + 3 < grammar.size() && grammar.substr(i, 2) == "\\x") {
			std::string hex_str = grammar.substr(i + 2, 2);
			bool is_hex = true;
			for (char c : hex_str) {
				if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
					is_hex = false;
					break;
				}
			}
			
			if (is_hex) {
				unsigned int value = 0;
				sscanf(hex_str.c_str(), "%x", &value);
				
				// Handle specific problematic hex escapes
				if (value == 0x0b || value == 0x0c || value == 0x85) {
					// These are VT, FF, and NEL characters that cause parsing issues
					// Skip them entirely
					i += 3; // Skip the \xXX sequence
					continue;
				} else {
					result += static_cast<char>(value);
					i += 3; // Skip the \xXX sequence
					continue;
				}
			}
		}
		
		// Regular character, copy as is
		result += grammar[i];
	}
	
	return result;
}

std::string GrammarProcessor::negate_character_class(const std::string& chars) {
	// Build a set of excluded characters
	std::set<unsigned char> excluded;
	
	for (size_t i = 0; i < chars.length(); ++i) {
		if (chars[i] == '\\' && i + 1 < chars.length()) {
			char next = chars[i + 1];
			switch (next) {
				case 'r': excluded.insert('\r'); i++; break;
				case 'n': excluded.insert('\n'); i++; break;
				case 't': excluded.insert('\t'); i++; break;
				case '\\': excluded.insert('\\'); i++; break;
				case '"': excluded.insert('"'); i++; break;
				case 'x':
					if (i + 3 < chars.length()) {
						// Parse hex escape
						unsigned int value = 0;
						if (sscanf(chars.c_str() + i + 2, "%2x", &value) == 1) {
							excluded.insert(static_cast<unsigned char>(value));
							i += 3;
						}
					}
					break;
				default:
					excluded.insert(next);
					i++;
					break;
			}
		} else if (i + 2 < chars.length() && chars[i + 1] == '-') {
			// Range like a-z
			unsigned char start = chars[i];
			unsigned char end = chars[i + 2];
			for (unsigned char c = start; c <= end; c++) {
				excluded.insert(c);
			}
			i += 2;
		} else {
			excluded.insert(chars[i]);
		}
	}
	
	// Build positive character class with all printable ASCII except excluded
	std::string positive;
	
	for (unsigned char c = 0x20; c <= 0x7E; c++) {
		if (excluded.find(c) == excluded.end()) {
			// Character not excluded, add to positive set
			if (c == '\\') {
				positive += "\\\\"; // Escape backslash
			} else if (c == ']') {
				positive += "\\]"; // Escape closing bracket
			} else if (c == '-') {
				positive += "\\-"; // Escape dash
			} else if (c == '^') {
				positive += "\\^"; // Escape caret
			} else {
				positive += c;
			}
		}
	}
	
	return positive;
}

std::set<char> GrammarProcessor::extract_characters(const std::string& inside_brackets) {
	std::set<char> chars;
	
	for (size_t i = 0; i < inside_brackets.length(); ++i) {
		if (inside_brackets[i] == '\\' && i + 1 < inside_brackets.length()) {
			// Escaped character
			char next = inside_brackets[i + 1];
			switch (next) {
				case 'r': chars.insert('\r'); break;
				case 'n': chars.insert('\n'); break;
				case 't': chars.insert('\t'); break;
				case '\\': chars.insert('\\'); break;
				case '"': chars.insert('"'); break;
				default: chars.insert(next); break;
			}
			i++; // Skip next character
		} else if (i + 2 < inside_brackets.length() && inside_brackets[i + 1] == '-') {
			// Range like a-z
			char start = inside_brackets[i];
			char end = inside_brackets[i + 2];
			for (char c = start; c <= end; c++) {
				chars.insert(c);
			}
			i += 2; // Skip range
		} else {
			chars.insert(inside_brackets[i]);
		}
	}
	
	return chars;
}

std::string GrammarProcessor::unicode_to_utf8(unsigned int codepoint) {
	std::string result;
	
	if (codepoint <= 0x7F) {
		result += static_cast<char>(codepoint);
	} else if (codepoint <= 0x7FF) {
		result += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	} else if (codepoint <= 0xFFFF) {
		result += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
		result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	}
	
	return result;
}