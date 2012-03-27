/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "makros.hh"
#include "application.hh"
#include "stringprocessing.hh"
#include "fileio.hh"
#include <fstream>
#include <string>
#include <vector>

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

int main(int argc, char** argv) {

  if (argc == 1 || strings_equal(argv[1],"-h")) {

    std::cerr << "USAGE: " << argv[0] << " -i <input file> -voc <vocabulary file> -o <output file (indices)>" << std::endl;
    exit(0);
  }

  const int nParams = 3;
  ParamDescr  params[nParams] = {{"-i",mandInFilename,0,""},{"-voc",mandInFilename,0,""},
                                 {"-o",mandOutFilename,0,""}};

  Application app(argc,argv,params,nParams);

  std::map<std::string,uint> vocabulary;

  std::istream* voc_stream;

#ifdef HAS_GZSTREAM
  if (is_gzip_file(app.getParam("-voc"))) {
    voc_stream = new igzstream(app.getParam("-voc").c_str());
  }
  else {
    voc_stream = new std::ifstream(app.getParam("-voc").c_str());
  }
#else
  voc_stream = new std::ifstream(app.getParam("-voc").c_str())
#endif

  std::string word;
  uint nWords = 0;
  while ((*voc_stream) >> word) {

    vocabulary[word] = nWords;
    nWords++;
  }
  delete voc_stream;
  
  std::istream* plain_stream;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(app.getParam("-i"))) {
    plain_stream = new igzstream(app.getParam("-i").c_str());
  }
  else {
    plain_stream = new std::ifstream(app.getParam("-i").c_str());
  }
#else
  plain_stream = new std::ifstream(app.getParam("-i").c_str())
#endif

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

  std::vector<std::string> tokens;

  bool oov_words=false;

  char cline[65536];
  while (plain_stream->getline(cline,65536)) {

    std::string line = cline;
    tokenize(line,tokens,' ');

    for (uint i=0; i < tokens.size(); i++) {

      std::map<std::string,uint>::iterator it = vocabulary.find(tokens[i]);
      if (it != vocabulary.end())
        (*out_stream) << it->second;
      else {
        oov_words = true;
        (*out_stream) << "OOV[" << tokens[i] << "]";
      }
      if (i+1 < tokens.size())
        (*out_stream) << " ";
    }

    (*out_stream) << std::endl;
  }

  if (oov_words)
    std::cerr << "WARNING: there are OOV words" << std::endl;

  delete plain_stream;
  delete out_stream;
}
