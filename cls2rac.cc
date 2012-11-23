/**** written by Thomas Schoenemann at the University of Düsseldorf, Germany, September 2012 ******/
/**** This is a small program to transform MKCLS's word class files into the RegAligner format ******/


#include "makros.hh"
#include "application.hh"
#include "stringprocessing.hh"
#include "storage1D.hh"
#include <fstream>
#include <string>
#include <vector>

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

int main(int argc, char** argv) {

  if (argc == 1 || (argc == 2 && strings_equal(argv[1],"-h"))) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << "-c <file with classes as output by MKCLS>" << std::endl
              << "-voc <vocabulary indices>" << std::endl
              << "-o <output file>" << std::endl
	      << std::endl;
      
    exit(1);
  }

  const int nParams = 3;
  ParamDescr  params[nParams] = {{"-c",mandInFilename,0,""},{"-voc",mandInFilename,0,""},{"-o",mandOutFilename,0,""}};

  Application app(argc,argv,params,nParams);

  std::ifstream vin(app.getParam("-voc").c_str());
  
  std::map<std::string,uint> voc_index;
  uint next_idx = 0;

  std::string s;
  while (vin >> s) {
    voc_index[s] = next_idx;
    next_idx++;
  }
  vin.close();
 
  Storage1D<uint> word_class(voc_index.size(),0);
 
  uint widx;
  std::ifstream classin(app.getParam("-c").c_str());
  while (classin >> s >> widx) {

    std::map<std::string,uint>::iterator it = voc_index.find(s);
    if (it == voc_index.end()) {
      INTERNAL_ERROR << " word does not occur in vocabulary file. Exiting..." << std::endl;
      exit(1);
    }

    word_class[it->second] = widx;
  }
  classin.close();

  std::ofstream out(app.getParam("-o").c_str());
  for (uint w=0; w < word_class.size(); w++)
    out << word_class[w] << std::endl;
}
