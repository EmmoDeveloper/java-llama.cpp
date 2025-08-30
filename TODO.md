# Full enhanced implementation of GBNF

GBNF (GGML [Backus-Naur Form]) is an extension of the traditional Backus-Naur Form, specifically designed for use with Large Language Models (LLMs).

The current architecture is too limited.

## Proposed Architecture
[Grammar String or File (.gbnf)]
↓
[Custom Parser + PCRE2]
↓
[Unicode Lexer (ICU)]
↓
[Token Masking Engine]
↓
[LLM Sampling (with SentencePiece tokens)]

## 🧱 Enhanced Library Stack
|    | Library type          | Name                                       | Description                                                                               | Source                                        | URL                                      | Documentation                                              |
|----|-----------------------|--------------------------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------------|------------------------------------------|------------------------------------------------------------|
| 1. | Unicode Handling      | ICU (International Components for Unicode) | Full-featured Unicode toolkit: normalization, collation, regex, locale data               | GitHub: unicode-org/icu ・ Official Site       | https://github.com/unicode-org/icu       | https://icu.unicode.org/                                   |
| 2. | Regex Engine          | PCRE2                                      | Perl-compatible regex engine with full Unicode support, negation, lookahead, named groups | GitHub: PCRE2Project/pcre2 ・ Docs & Guide     | https://github.com/PCRE2Project/pcre2    | https://pcre2project.github.io/pcre2/                      |                         |
|    |                       | jpcre2 (C++ wrapper)                       | C++ wrapper for PCRE2 with object-oriented API                                            | GitHub: jpcre2/jpcre2                         | https://github.com/jpcre2/jpcre2         |                                                            |
| 3. | Grammar Parsing       | Ragel                                      | State machine compiler for regex/BNF → C/C++ code; ideal for lexers and protocol parsers  | GitHub: adrian-thurston/ragel ・ Official Site | https://github.com/adrian-thurston/ragel | https://www.colm.net/open-source/ragel/                    |
|    |                       | ANTLR (C++ target)                         | Powerful parser generator; supports listener/visitor patterns                             | GitHub: antlr/antlr4 ・ C++ Target Guide       | https://github.com/antlr/antlr4          | https://github.com/antlr/antlr4/blob/dev/doc/cpp-target.md |
| 4. | Tokenizer Integration | SentencePiece                              | Language-independent subword tokenizer (BPE, unigram); C++ native                         | GitHub: google/sentencepiece ・ Paper & Docs   | https://github.com/google/sentencepiece  | https://github.com/google/sentencepiece#documentation      |

