#pragma once

#include <tuple>
#include <queue>
//#include "utils.cpp"
using namespace std;


typedef float valType; // type for the values in the docs and queries arrays
//typedef uint8_t embeddngDimType; // embedding dimension is 128  
//typedef uint8_t numTermsType; // 32 terms for queries, max 180 terms for docs   
//typedef uint16_t numQueriesType; // 6980 queries   
typedef uint32_t numDocsType; // max is 4,294,967,295, should be fine...
typedef uint64_t numVectorsType;  
//typedef uint16_t vpqType; // number of values per query is 32 * 128 = 4096   
//typedef uint16_t vpdType; // number of values per document is 150 * 128 = 19200    
typedef uint64_t globalIdxType; 

