/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "makros.hh"
#include "application.hh"
#include "stringprocessing.hh"
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {

  const int nParams = 3;
  ParamDescr  params[nParams] = {{"-i",mandInFilename,0,""},{"-voc",mandInFilename,0,""},
                                 {"-o",mandOutFilename,0,""}};

  Application app(argc,argv,params,nParams);

  std::map<std::string,uint> vocabulary;

  std::ifstream voc_stream(app.getParam("-voc").c_str());
  std::string word;
  uint nWords = 0;
  while (voc_stream >> word) {

    vocabulary[word] = nWords;
    nWords++;
  }
  voc_stream.close();
  
  std::ifstream plain_stream(app.getParam("-i").c_str());
  std::ofstream out_stream(app.getParam("-o").c_str());

  std::vector<std::string> tokens;

  bool oov_words=false;

  char cline[65536];
  while (plain_stream.getline(cline,65536)) {

    std::string line = cline;
    tokenize(line,tokens,' ');

    for (uint i=0; i < tokens.size(); i++) {

      std::map<std::string,uint>::iterator it = vocabulary.find(tokens[i]);
      if (it != vocabulary.end())
        out_stream << it->second;
      else {
        oov_words = true;
        out_stream << "OOV[" << tokens[i] << "]";
      }
      if (i+1 < tokens.size())
        out_stream << " ";
    }

    out_stream << std::endl;
  }

  if (oov_words)
    std::cerr << "WARNING: there are OOV words" << std::endl;
}
