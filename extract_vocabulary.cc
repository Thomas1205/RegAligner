/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "makros.hh"
#include "application.hh"
#include <fstream>
#include "stringprocessing.hh"
#include "fileio.hh"

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

int main(int argc, char** argv)
{

  if (argc == 1 || (argc == 2 && strings_equal(argv[1],"-h"))) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << "-i <file> : list of sentences " << std::endl
              << "[-i2 <file>] : further sentences (e.g. a dev set)" << std::endl
              << "-o <file> : output for extracted vocabulary" << std::endl
              << "[-one-extra-class]: group all rare words together, not by length" << std::endl
              << "[-max-rare <uint>]: limit for counting as rare" << std::endl
              << "[-statistics]: print statistics" << std::endl;
    exit(0);
  }

  const int nParams = 6;
  ParamDescr  params[nParams] = {{"-i",mandInFilename,0,""},{"-o",mandOutFilename,0,""},
    {"-i2",optInFilename,0,""},{"-statistics",flag,0,""},
    {"-one-extra-class",flag,0,""},{"-max-rare",optWithValue,1,"2"}
  };

  Application app(argc,argv,params,nParams);

  bool one_extra_class = app.is_set("-one-extra-class");
  uint max_rare = convert<uint>(app.getParam("-max-rare"));

  std::istream* raw_stream;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(app.getParam("-i"))) {
    raw_stream = new igzstream(app.getParam("-i").c_str());
  }
  else {
    raw_stream = new std::ifstream(app.getParam("-i").c_str());
  }
#else
  raw_stream = new std::ifstream(app.getParam("-i").c_str());
#endif

  std::string s;

  std::set<std::string> vocabulary;

  std::map<std::string,uint> word_count;

  while ((*raw_stream) >> s) {

    word_count[s]++;
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

      if (app.is_set("-statistics"))
        word_count[s]++;
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

  uint nSingletons = 0;

  //empty word has index 0
  (*out_stream) << "%NULL" << std::endl;
  for (std::set<std::string>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++) {
    (*out_stream) << *it << std::endl;

    if (app.is_set("-statistics") && word_count[*it] == 1)
      nSingletons++;
  }

  if (app.is_set("-statistics"))
    std::cerr << vocabulary.size() << " words, " << nSingletons << " singletons" << std::endl;

  delete out_stream;

  std::string filename = app.getParam("-o") + ".fert_classes";
  std::ofstream classesout(filename.c_str());
  classesout << "0" << std::endl; //empty word
  const uint nWords = vocabulary.size()+1;
  uint idx = 1;
  for (std::set<std::string>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++) {
    if (word_count[*it] <= max_rare) {
      if (one_extra_class)
        classesout << (nWords+1) << std::endl;
      else
        classesout << (nWords+it->size()) << std::endl;
    }
    else
      classesout << idx << std::endl;
    idx++;
  }
}
