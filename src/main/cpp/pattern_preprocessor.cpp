#include "pattern_preprocessor.h"
#include <sstream>
#include <set>
#include <iomanip>
#include <functional>

std::string PatternPreprocessor::preprocess(const std::string& pattern) {
	std::string result = pattern;
	
	// Process in order: Unicode escapes, hex escapes, then negated character classes
	result = process_unicode_escapes(result);
	result = process_hex_escapes(result);
	result = process_negated_char_classes(result);
	
	return result;
}

std::string PatternPreprocessor::process_unicode_escapes(const std::string& input) {
	static const std::regex unicode_pattern(R"(\\u([0-9a-fA-F]{4}))");
	
	// Use regex_replace with callback via iterator
	std::string result;
	auto callback = [](const std::smatch& match) -> std::string {
		unsigned int codepoint = std::stoul(match[1].str(), nullptr, 16);
		
		// Skip problematic Unicode characters
		if (codepoint == 0x2028 || codepoint == 0x2029) {
			return "";  // LINE SEPARATOR and PARAGRAPH SEPARATOR - skip them
		}
		
		return PatternPreprocessor::codepoint_to_utf8(codepoint);
	};
	
	// Manual replacement using regex_iterator
	auto begin = std::sregex_iterator(input.begin(), input.end(), unicode_pattern);
	auto end = std::sregex_iterator();
	
	size_t last_pos = 0;
	for (auto it = begin; it != end; ++it) {
		result.append(input, last_pos, it->position() - last_pos);
		result.append(callback(*it));
		last_pos = it->position() + it->length();
	}
	result.append(input, last_pos);
	
	return result;
}

std::string PatternPreprocessor::process_hex_escapes(const std::string& input) {
	static const std::regex hex_pattern(R"(\\x([0-9a-fA-F]{2}))");
	
	// Use regex_replace with callback via iterator
	std::string result;
	auto callback = [](const std::smatch& match) -> std::string {
		unsigned int value = std::stoul(match[1].str(), nullptr, 16);
		
		// Skip problematic control characters
		if (value == 0x0b || value == 0x0c || value == 0x85) {
			return "";  // VT, FF, and NEL - skip them
		}
		
		return std::string(1, static_cast<char>(value));
	};
	
	// Manual replacement using regex_iterator
	auto begin = std::sregex_iterator(input.begin(), input.end(), hex_pattern);
	auto end = std::sregex_iterator();
	
	size_t last_pos = 0;
	for (auto it = begin; it != end; ++it) {
		result.append(input, last_pos, it->position() - last_pos);
		result.append(callback(*it));
		last_pos = it->position() + it->length();
	}
	result.append(input, last_pos);
	
	return result;
}

std::string PatternPreprocessor::process_negated_char_classes(const std::string& input) {
	// Complex regex to match [^...] with proper escape handling
	static const std::regex negated_class_pattern(R"(\[\^((?:[^\]\\]|\\.)*)?\])");
	
	std::string result;
	auto begin = std::sregex_iterator(input.begin(), input.end(), negated_class_pattern);
	auto end = std::sregex_iterator();
	
	size_t last_pos = 0;
	for (auto it = begin; it != end; ++it) {
		result.append(input, last_pos, it->position() - last_pos);
		
		// Extract and process the negated character set
		std::string negated_chars = (*it)[1].str();
		std::set<unsigned char> excluded;
		
		// Parse escape sequences and ranges
		for (size_t i = 0; i < negated_chars.length(); ++i) {
			if (negated_chars[i] == '\\' && i + 1 < negated_chars.length()) {
				char next = negated_chars[i + 1];
				switch (next) {
					case 'r': excluded.insert('\r'); i++; break;
					case 'n': excluded.insert('\n'); i++; break;
					case 't': excluded.insert('\t'); i++; break;
					case '\\': excluded.insert('\\'); i++; break;
					case ']': excluded.insert(']'); i++; break;
					case 'd': // \d = [0-9]
						for (char c = '0'; c <= '9'; ++c) excluded.insert(c);
						i++;
						break;
					case 's': // \s = whitespace
						excluded.insert(' ');
						excluded.insert('\t');
						excluded.insert('\n');
						excluded.insert('\r');
						excluded.insert('\f');
						i++;
						break;
					case 'w': // \w = [a-zA-Z0-9_]
						for (char c = 'a'; c <= 'z'; ++c) excluded.insert(c);
						for (char c = 'A'; c <= 'Z'; ++c) excluded.insert(c);
						for (char c = '0'; c <= '9'; ++c) excluded.insert(c);
						excluded.insert('_');
						i++;
						break;
					default:
						excluded.insert(next);
						i++;
						break;
				}
			} else if (i + 2 < negated_chars.length() && negated_chars[i + 1] == '-') {
				// Character range
				unsigned char start = negated_chars[i];
				unsigned char end = negated_chars[i + 2];
				for (unsigned char c = start; c <= end; ++c) {
					excluded.insert(c);
				}
				i += 2;
			} else {
				excluded.insert(negated_chars[i]);
			}
		}
		
		// Build positive character class
		result += '[';
		for (unsigned char c = 0; c < 255; ++c) {
			if (excluded.find(c) == excluded.end()) {
				if (c == '\\' || c == ']' || c == '-' || c == '^') {
					result += '\\';
					result += c;
				} else if (c < 0x20 || c == 0x7F) {
					// Control characters - use hex escape
					std::stringstream ss;
					ss << "\\x" << std::hex << std::setw(2) << std::setfill('0') << (int)c;
					result += ss.str();
				} else {
					result += c;
				}
			}
		}
		result += ']';
		
		last_pos = it->position() + it->length();
	}
	result.append(input, last_pos);
	
	return result;
}

std::string PatternPreprocessor::codepoint_to_utf8(unsigned int codepoint) {
	std::string result;
	
	if (codepoint <= 0x7F) {
		// Single-byte UTF-8
		result += static_cast<char>(codepoint);
	} else if (codepoint <= 0x7FF) {
		// Two-byte UTF-8
		result += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	} else if (codepoint <= 0xFFFF) {
		// Three-byte UTF-8
		result += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
		result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	} else if (codepoint <= 0x10FFFF) {
		// Four-byte UTF-8
		result += static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07));
		result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
		result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	}
	
	return result;
}