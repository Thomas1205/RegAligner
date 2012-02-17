/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "makros.hh"
#include "application.hh"
#include <fstream>
#include "stringprocessing.hh"
#include "fileio.hh"

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

int main(int argc, char** argv) {

  if (argc == 1 || (argc == 2 && strings_equal(argv[1],"-h"))) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << "-i <file> : list of sentences " << std::endl
              << "[-i2 <file>] : further sentences (e.g. a dev set)" << std::endl
              << "-o <file> : output for extracted vocabulary" << std::endl;
  }

  const int nParams = 3;
  ParamDescr  params[nParams] = {{"-i",mandInFilename,0,""},{"-o",mandOutFilename,0,""},
                                 {"-i2",optInFilename,0,""}};

  Application app(argc,argv,params,nParams);

  //std::ifstream raw_stream(app.getParam("-i").c_str());

  std::istream* raw_stream;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(app.getParam("-i"))) {
    raw_stream = new igzstream(app.getParam("-i").c_str());
  }
  else {
    raw_stream = new std::ifstream(app.getParam("-i").c_str());
  }
#else
  raw_stream = new std::ifstream(app.getParam("-i").c_str())
#endif

  std::string s;

  std::set<std::string> vocabulary;

  while ((*raw_stream) >> s) {

    vocabulary.insert(s);
  }
  delete raw_stream;

  if (app.is_set("-i2")) {

#ifdef HAS_GZSTREAM
    if (is_gzip_file(app.getParam("-i2"))) {
      raw_stream = new igzstream(app.getParam("-i2").c_str());
    }
    else {
      raw_stream = new std::ifstream(app.getParam("-i2").c_str());
    }
#else
    raw_stream = new std::ifstream(app.getParam("-i2").c_str());
#endif

    while ((*raw_stream) >> s) {
      
      vocabulary.insert(s);
    }
    delete raw_stream;
  } 

  std::ostream* out_stream;

#ifdef HAS_GZSTREAM
  if (string_ends_with(app.getParam("-o"),".gz")) {
    out_stream = new ogzstream(app.getParam("-o").c_str());
  }
  else {
#else
  if (true) {
#endif
    out_stream = new std::ofstream(app.getParam("-o").c_str());
  }
  
  //empty word has index 0
  (*out_stream) << "%NULL" << std::endl;
  for (std::set<std::string>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++) {
    (*out_stream) << *it << std::endl;
  }

  delete out_stream;
}
