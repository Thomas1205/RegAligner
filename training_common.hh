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
                         const CooccuringWordsType& cooc, uint nSourceWords,
                         LookupTable& slookup, uint max_size = MAX_UINT);

const SingleLookupTable& get_wordlookup(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                        const CooccuringWordsType& cooc, uint nSourceWords,
                                        const SingleLookupTable& lookup, SingleLookupTable& aux);

double prob_penalty(double x, double beta);

double prob_pen_prime(double x, double beta);

void update_dict_from_counts(const SingleWordDictionary& fdict_count, 
			     const floatSingleWordDictionary& prior_weight,
			     double dict_weight_sum, uint iter, 
			     bool smoothed_l0, double l0_beta,
			     uint nDictStepIter, SingleWordDictionary& dict,
			     double min_prob = 0.0);

void dict_m_step(const SingleWordDictionary& fdict_count, 
                 const floatSingleWordDictionary& prior_weight,
                 SingleWordDictionary& dict, double alpha, uint nIter = 100,
                 bool smoothed_l0 = false, double l0_beta = 1.0);

void single_dict_m_step(const Math1D::Vector<double>& fdict_count, 
                        const Math1D::Vector<float>& prior_weight,
                        Math1D::Vector<double>& dict, double alpha, uint nIter,
                        bool smoothed_l0, double l0_beta);

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, 
                                 const Math1D::Vector<float>& prior_weight,
                                 const Math1D::Vector<double>& dict, bool smoothed_l0, double l0_beta);


//for IBM-4/5 (i.e. no alignments to NULL considered)
void par2nonpar_start_prob(const Math1D::Vector<double>& sentence_start_parameters,
			   Storage1D<Math1D::Vector<double> >& sentence_start_prob);

void start_prob_m_step(const Storage1D<Math1D::Vector<double> >& start_count, Math1D::Vector<double>& param);


#endif
