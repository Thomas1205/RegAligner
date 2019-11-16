/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** extended by Thomas Schoenemann at Lund University, Sweden, and University of Pisa, Italy, 2010-2011 ***/
/*** additions at the University of Düsseldorf, Germany, 2012 ***/

#include "corpusio.hh"
#include "stringprocessing.hh"
#include <fstream>
#include <map>

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

void read_vocabulary(std::string filename, std::vector<std::string>& voc_list)
{

#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  bool zipped = is_gzip_file(filename);

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else
  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  voc_list.clear();

  std::string word;
  while ((*instream) >> word) {
    voc_list.push_back(word);
  }
}

void read_monolingual_corpus(std::string filename, Storage1D<Math1D::Vector<uint> >& sentence_list)
{

  bool zipped = is_gzip_file(filename);

#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else

  if (zipped) {
    INTERNAL_ERROR << "zipped file input, but support for gz is not enabled" << std::endl;
    exit(1);
  }

  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  std::vector<std::vector<uint> > slist;

  char cline[65536];
  std::string line;

  std::vector<std::string> tokens;

  while(instream->getline(cline,65536)) {

    slist.push_back(std::vector<uint>());

    line = cline;
    tokenize(line,tokens,' ');

    std::vector<uint>& cur_line = slist.back();

    for (uint k=0; k < tokens.size(); k++) {
      if (tokens[k].size() > 3 && tokens[k].substr(0,3) == "OOV") {
        TODO("handling of OOVs");
      }
      else {
        cur_line.push_back(convert<uint>(tokens[k]));
      }
    }
  }

  sentence_list.resize_dirty(slist.size());
  for (uint s=0; s < slist.size(); s++) {
    sentence_list[s].resize_dirty(slist[s].size());
    for (uint k=0; k < slist[s].size(); k++)
      sentence_list[s][k] = slist[s][k];
  }
}

void read_monolingual_corpus(std::string filename, NestedStorage1D<uint,uint>& sentence_list)
{

  bool zipped = is_gzip_file(filename);


#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else

  if (zipped) {
    INTERNAL_ERROR << "zipped file input, but support for gz is not enabled" << std::endl;
    exit(1);
  }

  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  std::vector<std::vector<uint> > slist;

  char cline[65536];
  std::string line;

  std::vector<std::string> tokens;

  while(instream->getline(cline,65536)) {

    slist.push_back(std::vector<uint>());

    line = cline;
    tokenize(line,tokens,' ');

    std::vector<uint>& cur_line = slist.back();

    for (uint k=0; k < tokens.size(); k++) {
      if (tokens[k].size() > 3 && tokens[k].substr(0,3) == "OOV") {
        TODO("handling of OOVs");
      }
      else {
        cur_line.push_back(convert<uint>(tokens[k]));
      }
    }
  }

  sentence_list = slist;
}

void read_monolingual_corpus(std::string filename, Storage1D<Storage1D<std::string> >& sentence_list)
{


  bool zipped = is_gzip_file(filename);

#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else

  if (zipped) {
    INTERNAL_ERROR << "zipped file input, but support for gz is not enabled" << std::endl;
    exit(1);
  }

  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  std::vector<std::vector<std::string> > slist;

  char cline[65536];
  std::string line;

  while(instream->getline(cline,65536)) {

    slist.push_back(std::vector<std::string>());
    std::vector<std::string>& cur_line = slist.back();

    line = cline;
    tokenize(line,cur_line,' ');
  }

  sentence_list.resize_dirty(slist.size());
  for (uint s=0; s < slist.size(); s++) {
    sentence_list[s].resize_dirty(slist[s].size());
    for (uint k=0; k < slist[s].size(); k++)
      sentence_list[s][k] = slist[s][k];
  }
}

bool read_next_monolingual_sentence(std::istream& file, Storage1D<std::string>& sentence)
{

  char cline[65536];
  std::string line;
  std::vector<std::string> cur_line;

  file.getline(cline,65536);
  bool success = !(file.eof() || file.fail());
  if (!success)
    return false;

  line = cline;
  tokenize(line,cur_line,' ');

  if (cur_line.size() == 0) {
    std::cerr << "WARNING: empty line: " << line << std::endl;
    //exit(0);
  }

  sentence.resize(cur_line.size());
  for (uint k=0; k < cur_line.size(); k++)
    sentence[k] = cur_line[k];

  return true;
}

void read_idx_dict(std::string filename, SingleWordDictionary& dict, CooccuringWordsType& cooc)
{

  bool zipped = is_gzip_file(filename);

#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else

  if (zipped) {
    INTERNAL_ERROR << "zipped file input, but support for gz is not enabled" << std::endl;
    exit(1);
  }

  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  uint nTargetWords = dict.size();
  assert(cooc.size() == nTargetWords);

  char cline[65536];
  std::string line;
  std::vector<std::string> tokens;

  uint last_tidx = MAX_UINT;

  std::vector<uint> cur_cooc;
  std::vector<double> cur_dict;

  while(instream->getline(cline,65536)) {

    line = cline;
    tokenize(line,tokens,' ');
    assert(tokens.size() == 3);

    uint tidx = convert<uint>(tokens[0]);
    uint sidx = convert<uint>(tokens[1]);
    double prob = convert<double>(tokens[2]);

    if (tidx != last_tidx) {
      if (last_tidx < nTargetWords) {
        dict[last_tidx].resize_dirty(cur_cooc.size());
        cooc[last_tidx].resize_dirty(cur_cooc.size());

        for (uint k=0; k < cur_cooc.size(); k++) {
          cooc[last_tidx][k] = cur_cooc[k];
          dict[last_tidx][k] = cur_dict[k];
        }
      }
      cur_cooc.clear();
      cur_dict.clear();
    }
    cur_cooc.push_back(sidx);
    cur_dict.push_back(prob);

    last_tidx = tidx;
  }

  //write last entries
  dict[last_tidx].resize_dirty(cur_cooc.size());
  cooc[last_tidx].resize_dirty(cur_cooc.size());

  for (uint k=0; k < cur_cooc.size(); k++) {
    cooc[last_tidx][k] = cur_cooc[k];
    dict[last_tidx][k] = cur_dict[k];
  }

}


void read_prior_dict(std::string filename, std::set<std::pair<uint, uint> >& known_pairs, bool invert)
{


  bool zipped = is_gzip_file(filename);

#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else

  if (zipped) {
    INTERNAL_ERROR << "zipped file input, but support for gz is not enabled" << std::endl;
    exit(1);
  }

  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  uint i1,i2;

  while ((*instream) >> i1 >> i2) {
    if (invert)
      std::swap(i1,i2);

    known_pairs.insert(std::make_pair(i1,i2));
  }
}

void read_word_classes(std::string filename, Storage1D<WordClassType>& word_class)
{

  bool zipped = is_gzip_file(filename);

#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else

  if (zipped) {
    INTERNAL_ERROR << "zipped file input, but support for gz is not enabled" << std::endl;
    exit(1);
  }

  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  WordClassType next_class;
  uint next_word = 0;

  std::map<uint,uint> class_count;

  while ((*instream) >> next_class) {
    if (next_word >= word_class.size()) {
      INTERNAL_ERROR << " more word classes given than there are words. Exiting.." << std::endl;
      exit(1);
    }

    if (instream->bad() || instream->fail()) {
      std::cerr << "ERROR: could not read word classes. Please check the file. Exiting.." << std::endl;
      exit(1);
    }

    class_count[next_class]++;
    word_class[next_word] = next_class;
    next_word++;
  }

  //pack the words densely, i.e. remove empty classes
  uint next_idx=0;
  std::map<uint,uint> dense_class_idx;
  for (std::map<uint,uint>::const_iterator it = class_count.begin(); it != class_count.end(); it++) {
    dense_class_idx[it->first] = next_idx;
    next_idx++;
  }
  for (uint i=0; i < word_class.size(); i++)
    word_class[i] = dense_class_idx[word_class[i]];

  if (!instream->eof() && (instream->bad() || instream->fail()) ) {
    std::cerr << "ERROR: could not read word classes. Please check the file. Exiting.." << std::endl;
    exit(1);
  }

  if (next_word != word_class.size()) {
    std::cerr << "WARNING: less word classes given than there are words. Filling with standard values.." << std::endl;
  }
}


void read_word_classes(std::string filename, Storage1D<uint>& word_class)
{


  bool zipped = is_gzip_file(filename);

#ifdef HAS_GZSTREAM
  std::ifstream infile;
  igzstream gzin;

  if (zipped)
    gzin.open(filename.c_str());
  else {
    infile.open(filename.c_str());
  }

  std::istream* instream = (zipped) ? static_cast<std::istream*>(&gzin) : &infile;
#else

  if (zipped) {
    INTERNAL_ERROR << "zipped file input, but support for gz is not enabled" << std::endl;
    exit(1);
  }

  std::ifstream infile(filename.c_str());

  std::istream* instream = &infile;
#endif

  uint next_class;
  uint next_word = 0;

  std::map<uint,uint> class_count;

  while ((*instream) >> next_class) {
    if (next_word >= word_class.size()) {
      INTERNAL_ERROR << " more word classes given than there are words. Exiting.." << std::endl;
      exit(1);
    }

    if (instream->bad() || instream->fail()) {
      std::cerr << "ERROR: could not read word classes. Please check the file. Exiting.." << std::endl;
      exit(1);
    }

    class_count[next_class]++;
    word_class[next_word] = next_class;
    next_word++;
  }

  //pack the words densely, i.e. remove empty classes
  uint next_idx=0;
  std::map<uint,uint> dense_class_idx;
  for (std::map<uint,uint>::const_iterator it = class_count.begin(); it != class_count.end(); it++) {
    dense_class_idx[it->first] = next_idx;
    next_idx++;
  }
  for (uint i=0; i < word_class.size(); i++)
    word_class[i] = dense_class_idx[word_class[i]];

  std::cerr << next_idx << " word classes" << std::endl;

  if (!instream->eof() && (instream->bad() || instream->fail()) ) {
    std::cerr << "ERROR: could not read word classes. Please check the file. Exiting.." << std::endl;
    exit(1);
  }

  if (next_word != word_class.size()) {
    std::cerr << "WARNING: less word classes given than there are words. Filling with standard values.." << std::endl;
  }
}
