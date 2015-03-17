#ifndef OCCA_PARSER_MAGIC_HEADER
#define OCCA_PARSER_MAGIC_HEADER

#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    class parserBase;

    class ctInfo {
    public:
      bool hasConstValue;
      opHolder constValue;

      expNode minBound, maxBound;
      expNode stride;

      std::vector<expNode> reads;
      std::vector<expNode> writes;
    };

    typedef std::map<varInfo*, ctInfo> ctMap_t;
    typedef ctMap_t::iterator          ctMapIterator;
    typedef ctMap_t::const_iterator    cCtMapIterator;

    class magician {
    public:
      parserBase &parser;
      statement &globalScope;
      varUsedMap_t &varUpdateMap;
      varUsedMap_t &varUsedMap;

      ctMap_t ctMap;

      magician(parserBase &parser_);

      static void castMagicOn(parserBase &parser_);

      void castMagic();
      void castMagicOn(statement &kernel);
    };
  };
};

#endif

/*
  1: x = 1;
  2: y = x + 2;
  3: x = z - w;
  4: x = y / z;

  Flow:
    1 -> 2: y uses x
    2 -> 4: x uses y

  Anti:
    2 -> 3: x needs to wait for y to use x

  Out :
    1 -> 3: x is updated
    3 -> 4: x is updated

  In  :
    3 -> 4: z is used
 */