#include <vector>
#include <boost/regex.hpp>

#ifndef CSV_PARSE
#define CSV_PARSE

// used to split the file in lines
const boost::regex linesregx("\\r\\n|\\n\\r|\\n|\\r");
 
// used to split each line to tokens, assuming ',' as column separator
const boost::regex fieldsregx(",(?=(?:[^\"]*\"[^\"]*\")*(?![^\"]*\"))");
 
typedef std::vector<std::string> Row;

std::vector<Row> parse(const char*, unsigned int);






#endif