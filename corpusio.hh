/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** additions at the University of Düsseldorf, Germany, 2012 ***/

#ifndef CORPUSIO_HH
#define CORPUSIO_HH

#include "vector.hh"
#include <vector>
#include <set>
#include "mttypes.hh"
#include <iostream>
#include "nested_storage1D.hh"

void read_vocabulary(std::string filename, std::vector<std::string>& voc_list);

void read_monolingual_corpus(std::string filename, Storage1D<Math1D::Vector<uint> > & sentence_list);

void read_monolingual_corpus(std::string filename, NestedStorage1D<uint,uint> & sentence_list);

void read_monolingual_corpus(std::string filename, Storage1D<Storage1D<std::string> > & sentence_list);

//returns true if the file contained another line
bool read_next_monolingual_sentence(std::istream& file, Storage1D<std::string>& sentence);

void read_idx_dict(std::string filename, SingleWordDictionary& dict, CooccuringWordsType& cooc);

struct PriorPair {
	
  PriorPair() {}	
	
  PriorPair(uint s, uint t, float m) : s_idx_(s), t_idx_(t), multiplicator_(m) {}  
  
  uint s_idx_ = MAX_UINT;
  uint t_idx_ = MAX_UINT;
  float multiplicator_ = 0.0;
};

bool operator<(const PriorPair& p1, const PriorPair& p2);

void read_prior_dict(std::string filename, std::set<std::pair<uint, uint> >& known_pairs, bool invert = false);

void read_prior_dict(std::string filename, std::set<PriorPair>& known_pairs, bool invert = false);

void read_word_classes(std::string filename, Storage1D<WordClassType>& word_class);

void read_word_classes(std::string filename, Storage1D<uint>& word_class);

#endif
