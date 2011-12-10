/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "makros.hh"
#include "application.hh"
#include <fstream>
#include "stringprocessing.hh"

int main(int argc, char** argv) {

  if (argc == 1 || (argc == 2 && strings_equal(argv[1],"-h"))) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << "-i <file> : list of sentences " << std::endl
              << "-o <file> : output for extracted vocabulary" << std::endl;
  }

  const int nParams = 2;
  ParamDescr  params[nParams] = {{"-i",mandInFilename,0,""},{"-o",mandOutFilename,0,""}};

  Application app(argc,argv,params,nParams);

  std::ifstream raw_stream(app.getParam("-i").c_str());
  std::ofstream out_stream(app.getParam("-o").c_str());

  std::string s;

  std::set<std::string> vocabulary;

  while (raw_stream >> s) {

    vocabulary.insert(s);
  }

  //empty word has index 0
  out_stream << "%NULL" << std::endl;
  for (std::set<std::string>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++) {
    out_stream << *it << std::endl;
  }

}
