/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#ifndef TRAINING_COMMON_HH
#define TRAINING_COMMON_HH

#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

void find_cooccuring_words(const Storage1D<Storage1D<uint> >& source, 
                           const Storage1D<Storage1D<uint> >& target,
                           uint nSourceWords, uint nTargetWords,
                           CooccuringWordsType& cooc);

void find_cooccuring_words(const Storage1D<Storage1D<uint> >& source, 
                           const Storage1D<Storage1D<uint> >& target,
                           const Storage1D<Storage1D<uint> >& additional_source, 
                           const Storage1D<Storage1D<uint> >& additional_target,
                           uint nSourceWords, uint nTargetWords,
                           CooccuringWordsType& cooc);

bool read_cooccuring_words_structure(std::string filename, uint nSourceWords, uint nTargetWords,
                                     CooccuringWordsType& cooc);


void find_cooc_monolingual_pairs(const Storage1D<Storage1D<uint> >& sentence,
                                 uint voc_size, Storage1D<Storage1D<uint> >& cooc);

void monolingual_pairs_cooc_count(const Storage1D<Storage1D<uint> >& sentence,
                                  const Storage1D<Storage1D<uint> >&t_cooc, Storage1D<Storage1D<uint> >&t_cooc_count);


void find_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
                                             const Storage1D<Storage1D<uint> >& target,
                                             std::map<std::pair<uint,uint>, std::set<uint> >& cooc);

void find_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
                                             const Storage1D<Storage1D<uint> >& target,
                                             uint nSourceWords, uint nTargetWords,
                                             Storage1D<Storage1D<std::pair<uint,Storage1D<uint> > > >& cooc);

void find_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
                                             const Storage1D<Storage1D<uint> >& target,
                                             const Storage1D<Storage1D<uint> >& target_cooc,
                                             Storage1D<Storage1D<Storage1D<uint> > >& st_cooc);


void count_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
                                              const Storage1D<Storage1D<uint> >& target,
                                              std::map<std::pair<uint,uint>, std::map<uint,uint> >& cooc_count);


void count_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
                                              const Storage1D<Storage1D<uint> >& target,
                                              uint nSourceWords, uint nTargetWords,
                                              Storage1D<Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > > >& cooc);



void find_cooccuring_lengths(const Storage1D<Storage1D<uint> >& source, 
                             const Storage1D<Storage1D<uint> >& target,
                             CooccuringLengthsType& cooc);

void generate_wordlookup(const Storage1D<Storage1D<uint> >& source, 
                         const Storage1D<Storage1D<uint> >& target,
                         const CooccuringWordsType& cooc,
                         Storage1D<Math2D::Matrix<uint> >& slookup);

#endif
