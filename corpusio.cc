/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "corpusio.hh"
#include "stringprocessing.hh"
#include <fstream>

void read_vocabulary(std::string filename, std::vector<std::string>& voc_list) {

  std::ifstream instream(filename.c_str());

  voc_list.clear();

  std::string word;
  while (instream >> word) {
    voc_list.push_back(word);
  }
}

void read_monolingual_corpus(std::string filename, Storage1D<Storage1D<uint> > & sentence_list) {

  std::ifstream instream(filename.c_str());

  std::vector<std::vector<uint> > slist;
  
  char cline[65536];
  std::string line;

  std::vector<std::string> tokens;

  while(instream.getline(cline,65536)) {

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

void read_monolingual_corpus(std::string filename, Storage1D<Storage1D<std::string> > & sentence_list) {

  std::ifstream instream(filename.c_str());

  std::vector<std::vector<std::string> > slist;
  
  char cline[65536];
  std::string line;

  while(instream.getline(cline,65536)) {

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

bool read_next_monolingual_sentence(std::istream& file, Storage1D<std::string>& sentence) {

  char cline[65536];
  std::string line;
  std::vector<std::string> cur_line;

  bool success = file.getline(cline,65536);
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

void read_idx_dict(std::string filename, SingleWordDictionary& dict, CooccuringWordsType& cooc) {

  std::ifstream instream(filename.c_str());

  uint nTargetWords = dict.size();
  assert(cooc.size() == nTargetWords);

  char cline[65536];
  std::string line;
  std::vector<std::string> tokens;

  uint last_tidx = MAX_UINT;

  std::vector<uint> cur_cooc;
  std::vector<double> cur_dict;

  while(instream.getline(cline,65536)) {

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


void read_prior_dict(std::string filename, std::set<std::pair<uint, uint> >& known_pairs, bool invert) {


  std::ifstream infile(filename.c_str());

  uint i1,i2;

  while (infile >> i1 >> i2) {
    if (invert)
      std::swap(i1,i2);

    known_pairs.insert(std::make_pair(i1,i2));
  }


}
