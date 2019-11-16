/**** written by Thomas Schoenemann as a private person, November 2012 ***/

#include "makros.hh"
#include "application.hh"
#include <fstream>
#include "stringprocessing.hh"
#include "alignment_error_rate.hh"
#include "fileio.hh"
#include "training_common.hh"
#include "corpusio.hh"

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include "stl_out.hh"

int main(int argc, char** argv)
{

  if (argc == 1 || (argc == 2 && strings_equal(argv[1], "-h"))) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << " -s <file> : source file (coded as indices)" << std::endl
              << " -t <file> : target file (coded as indices)" << std::endl
              << " [-ds <file>] : additional source file (word indices) " << std::endl
              << " [-dt <file>] : additional target file (word indices) " << std::endl
              << " -o <out file>" << std::endl;
  }

  const int nParams = 5;
  ParamDescr params[nParams] = {
    {"-s", mandInFilename, 0, ""}, {"-t", mandInFilename, 0, ""},
    {"-ds", optInFilename, 0, ""}, {"-dt", optInFilename, 0, ""},
    {"-o", mandOutFilename, 0, ""}
  };

  Application app(argc, argv, params, nParams);

  NamedStorage1D<Math1D::Vector<uint> > source_sentence(MAKENAME(source_sentence));
  NamedStorage1D<Math1D::Vector<uint> > target_sentence(MAKENAME(target_sentence));

  NamedStorage1D<Math1D::Vector<uint> > dev_source_sentence(MAKENAME(dev_source_sentence));
  NamedStorage1D<Math1D::Vector<uint> > dev_target_sentence(MAKENAME(dev_target_sentence));

  read_monolingual_corpus(app.getParam("-s"), source_sentence);
  read_monolingual_corpus(app.getParam("-t"), target_sentence);

  if (app.is_set("-ds") != app.is_set("-dt")) {
    std::cerr << "WARNING: you need to specify both -ds and -dt . Ignoring additional sentences" << std::endl;
  }

  bool dev_present = app.is_set("-ds") && app.is_set("-dt");

  if (dev_present) {

    if (app.getParam("-ds") == app.getParam("-dt")) {

      std::cerr << "WARNING: dev-files for source and target sentences are identical!" << std::endl;
    }

    if (app.getParam("-s") == app.getParam("-ds")) {
      std::cerr << "WARNING: same file for source part of main corpus and development corpus" << std::endl;
    }
    if (app.getParam("-t") == app.getParam("-dt")) {
      std::cerr << "WARNING: same file for target part of main corpus and development corpus" << std::endl;
    }

    read_monolingual_corpus(app.getParam("-ds"), dev_source_sentence);
    read_monolingual_corpus(app.getParam("-dt"), dev_target_sentence);
  }

  for (size_t s = 0; s < source_sentence.size(); s++) {

    uint curJ = source_sentence[s].size();
    uint curI = target_sentence[s].size();

    if (9 * curJ < curI || 9 * curI < curJ) {

      std::cerr << "WARNING: GIZA++ would ignore sentence pair #" << (s + 1) << ": J=" << curJ << ", I=" << curI << std::endl;
    }
  }

  uint nSourceWords = 0;
  uint nTargetWords = 0;

  for (size_t s = 0; s < source_sentence.size(); s++) {

    nSourceWords = std::max(nSourceWords, source_sentence[s].max() + 1);
    nTargetWords = std::max(nTargetWords, target_sentence[s].max() + 1);
  }

  for (size_t s = 0; s < dev_source_sentence.size(); s++) {

    nSourceWords = std::max(nSourceWords, dev_source_sentence[s].max() + 1);
    nTargetWords = std::max(nTargetWords, dev_target_sentence[s].max() + 1);
  }

  CooccuringWordsType wcooc(MAKENAME(wcooc));

  std::cerr << "finding cooccuring words" << std::endl;
  find_cooccuring_words(source_sentence, target_sentence, dev_source_sentence,
                        dev_target_sentence, nSourceWords, nTargetWords, wcooc);

  std::string filename = app.getParam("-o");

  std::ostream* out;
#ifdef HAS_GZSTREAM
  if (string_ends_with(filename, ".gz")) {
    out = new ogzstream(filename.c_str());
  }
  else {
    out = new std::ofstream(filename.c_str());
  }
#else
  out = new std::ofstream(filename.c_str());
#endif

  for (uint t = 1; t < wcooc.size(); t++) {

    for (uint k = 0; k < wcooc[t].size(); k++) {
      (*out) << wcooc[t][k] << " ";
    }
    (*out) << std::endl;
  }

  delete out;
}
