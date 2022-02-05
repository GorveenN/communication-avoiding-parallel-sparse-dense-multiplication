#ifndef __PARSER_H__
#define __PARSER_H__

#include <matrix.h>

class Parser {
   public:
    static matrix::Sparse parse(std::ifstream& stream);
};

#endif  // __PARSER_H__
