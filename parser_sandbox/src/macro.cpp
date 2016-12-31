#include <cstring>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/lex.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"

#include "macro.hpp"

namespace occa {
  //---[ Part ]-------------------------
  macroPart_t::macroPart_t(const int info_) :
    info(info_) {}

  macroPart_t::macroPart_t(const std::string &str_) :
    info(macroInfo::string),
    str(str_) {}
  //====================================

  //---[ Macro ]-------------------------
  const std::string macro_t::VA_ARGS = "__VA_ARGS__";

  macro_t::macro_t() {}

  macro_t::macro_t(const char *c) {
    load(c, strlen(c));
  }

  macro_t::macro_t(const char *c, const int chars) {
    load(c, chars);
  }

  macro_t::macro_t(const std::string &s) {
    load(s.c_str(), s.size());
  }

  void macro_t::load(const std::string &s) {
    load(s.c_str(), s.size());
  }

  void macro_t::load(const char *c_, const int chars) {
    std::string chunk(c_, chars);
    const char *c = chunk.c_str();
    clear();

    lex::skipWhitespace(c);
    if (!lex::isAlpha(*c) && *c != '_') {
      return;
    }
    loadName(c);
    if (*c != '(') {
      parts.push_back(chunk.substr(c - chunk.c_str()));
      return;
    }
    macroPartVector_t argNames;
    // Skip '('
    loadArgs(++c, argNames);
    argc = (int) argNames.size();

    // Check and remove ... from arguments
    for (int i = 0; i < argc; ++i) {
      if (argNames[i].str == "...") {
        OCCA_ERROR("Variable arguments (...) must be the last argument",
                   i == (argc - 1));
        hasVarArgs = true;
        argNames.pop_back();
        --argc;
      }
    }

    // Skip ')'
    setParts(++c, argNames);
  }

  void macro_t::loadName(const char *&c) {
    static std::string delimiters;
    if (delimiters.size() == 0) {
      delimiters = lex::whitespaceChars;
      delimiters += '(';
    }
    const char *nameStart = c;
    lex::skipTo(c, delimiters);
    name = std::string(nameStart, c - nameStart);
    lex::skipWhitespace(c);
  }

  void macro_t::loadArgs(const char *&c, macroPartVector_t &argNames, const bool keepWhitespace) const {
    static std::string delimiters(",)");

    lex::skipWhitespace(c);
    const char *argsStart = c;
    lex::skipTo(c, ')');
    const char *argsEnd = c;
    OCCA_ERROR("Missing closing \")\"",
               *argsEnd == ')');

    c = argsStart;
    while(c < argsEnd) {
      c += (*c == ',');
      const char *start = c;
      lex::skipWhitespace(c);
      const char *argStart = c;
      lex::skipTo(c, delimiters);
      const char *argEnd = c;

      macroPart_t arg;
      arg.str = std::string(argStart, argEnd - argStart);
      if (keepWhitespace) {
        if (lex::isWhitespace(*start)) {
          arg.info |= macroInfo::hasLeftSpace;
        }
        if (lex::isWhitespace(*(argEnd - 1))) {
          arg.info |= macroInfo::hasRightSpace;
        }
      }
      argNames.push_back(arg);
    }
    c = argsEnd;
    lex::skipWhitespace(c);
  }

  void macro_t::setParts(const char *&c, macroPartVector_t &argNames) {
    static std::string delimiters;
    // Setup word delimeters [a-zA-Z0-9]
    if (delimiters.size() == 0) {
      int pos = 0;
      delimiters.resize(26 + 26 + 10 + 1);
      for (char c_ = 'a'; c_ <= 'z'; ++c_) {
        delimiters[pos++] = c_;
        delimiters[pos++] = ('A' + c_ - 'a');
      }
      for (char c_ = '0'; c_ <= '9'; ++c_) {
        delimiters[pos++] = c_;
      }
      delimiters[pos++] = '_';
    }

    lex::skipWhitespace(c);
    const char *cStart = c;
    while (*c != '\0') {
      lex::skipTo(c, delimiters);
      const char *partStart = c;
      lex::skipFrom(c, delimiters);

      const int partSize = (c - partStart);
      macroPart_t part;
      // Iterate over argument names if part starts with [a-zA-Z0-9]
      if ((*partStart < '0') || ('9' < *partStart)) {
        for (int i = 0; i < argc; ++i) {
          const std::string &argName = argNames[i].str;
          if ((partSize != (int) argName.size()) ||
              strncmp(argName.c_str(), partStart, partSize)) {
            continue;
          }
          // Setup macro part
          part.argPos = i;
          part.info   = macroInfo::arg;
        }
        if (hasVarArgs                         &&
            (part.info == 0)                   &&
            (partSize == (int) VA_ARGS.size()) &&
            !strncmp(VA_ARGS.c_str(), partStart, partSize)) {
          part.argPos = -1;
          part.info = (macroInfo::arg | macroInfo::variadic);
        }
      }
      // Add lazy string part if argument was found
      if (part.info) {
        if (cStart < partStart) {
          const int strChars = (partStart - cStart);
          std::string str(cStart, strChars);

          // Change arguemnt type if needed
          if (str[strChars - 1] == '#') {
            if ((2 <= strChars) &&
                str[strChars - 2] == '#') {
              part.info |= macroInfo::concat;
              str = str.substr(0, strChars - 2);
            } else {
              part.info |= macroInfo::stringify;
              str = str.substr(0, strChars - 1);
            }
          }
          // Ignore strings only made with # or ##
          if (str.size()) {
            parts.push_back(str);
          }
        }
        // Push back argument part
        parts.push_back(part);
          // Update the lazy string start
        lex::skipWhitespace(c);
        cStart = c;
      }
    }
    if (cStart < c) {
      parts.push_back(std::string(cStart, c - cStart));
    }
  }

  void macro_t::clear() {
    name = "";

    argc = 0;
    parts.clear();

    definedLine   = -1;
    undefinedLine = -1;
  }

  std::string macro_t::expand() const {
    const int partCount = (int) parts.size();
    if (partCount == 0) {
      return "";
    }
    if (argc == 0) {
      return parts[0].str;
    }
    return expand(macroPartVector_t());
  }

  std::string macro_t::expand(const char *c, const int chars) const {
    return expand(std::string(c, chars));
  }

  std::string macro_t::expand(const std::string &s) const {
    const char *c = s.c_str();
    macroPartVector_t args;
    loadArgs(c, args, true);
    return expand(args);
  }

  std::string macro_t::expand(const macroPartVector_t &args) const {
    const int partCount = (int) parts.size();
    if (partCount == 0) {
      return "";
    }

    const int inputArgc = (int) args.size();
    std::string ret;

    for (int i = 0; i < partCount; ++i) {
      const macroPart_t &part = parts[i];
      const size_t startRetSize = ret.size();

      if (part.info & macroInfo::string) {
        ret += part.str;
      } else if (part.info & macroInfo::arg) {
        std::string argStr;
        if (part.info & macroInfo::variadic) {
          for (int j = argc; j < inputArgc; ++j) {
            // Only add a space if there is there is an argument
            if ((argc < j) &&
                ((args[j].info & macroInfo::hasSpace) ||
                 args[j].str.size())) {
              argStr += ' ';
            }
            argStr += args[j].str;
            // ##__VA_ARGS__ doesn't print trailing,
            if ((j < (inputArgc - 1)) &&
                ((j < (inputArgc - 2))            ||
                 !(part.info & macroInfo::concat) ||
                 (0 < args[j + 1].str.size()))) {
                argStr += ',';
            }
          }
        } else if (part.argPos < inputArgc) {
          argStr = args[part.argPos].str;
        }

        // Modify argStr based on stringify, concat, and spaces
        if (part.info & macroInfo::stringify) {
          ret += '"';
          ret += argStr;
          ret += '"';
        } else if (part.info & macroInfo::concat) {
          ret += argStr;
        } else {
          if ((part.argPos < 0) || (inputArgc <= part.argPos)) {
            ret += argStr;
          } else {
            const macroPart_t arg = args[part.argPos];
            if (arg.info & macroInfo::hasLeftSpace) {
              ret += ' ';
            }
            ret += argStr;
            if (arg.info & macroInfo::hasRightSpace) {
              ret += ' ';
            }
          }
        }
      }
      // Separate inputs with spaces
      if ((i < (partCount - 1))                    &&
          !(parts[i + 1].info & macroInfo::concat) &&
          (ret.size() != startRetSize)) {
        ret += ' ';
      }
    }

    return ret;
  }
  //====================================
}