/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** additions at the University of Düsseldorf, Germany, 2012 ***/

#ifndef CORPUSIO_HH
#define CORPUSIO_HH

#include "vector.hh"
#include <vector>
#include <set>
#include "mttypes.hh"
#include <iostream>

void read_vocabulary(std::string filename, std::vector<std::string>& voc_list);

void read_monolingual_corpus(std::string filename, Storage1D<Storage1D<uint> > & sentence_list);

void read_monolingual_corpus(std::string filename, Storage1D<Storage1D<std::string> > & sentence_list);

//returns true if the file contained another line
bool read_next_monolingual_sentence(std::istream& file, Storage1D<std::string>& sentence);

void read_idx_dict(std::string filename, SingleWordDictionary& dict, CooccuringWordsType& cooc);

void read_prior_dict(std::string filename, std::set<std::pair<uint, uint> >& known_pairs, bool invert = false);

void read_word_classes(std::string filename, Storage1D<WordClassType>& word_class);

#endif
