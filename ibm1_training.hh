/*** written by Thomas Schoenemann as a private person without employment, October 2009 
 *** and later by Thomas Schoenemann as employee of Lund University, 2010 ***/


#ifndef IBM1_TRAINING_HH
#define IBM1_TRAINING_HH


#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>


void train_ibm1(const Storage1D<Storage1D<uint> >& source, 
                const Storage1D<Math2D::Matrix<uint> >& slookup,
                const Storage1D<Storage1D<uint> >& target,
                const CooccuringWordsType& cooc, 
                uint nSourceWords, uint nTargetWords,
                SingleWordDictionary& dict,
                uint nIterations,
                std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_ref_alignments,
                std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_ref_alignments,
                const floatSingleWordDictionary& prior_weight,
                bool smoothed_l0 = false, double l0_beta = 1.0);

void train_ibm1_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source, 
                               const Storage1D<Math2D::Matrix<uint> >& slookup,
                               const Storage1D<Storage1D<uint> >& target,
                               const CooccuringWordsType& cooc, 
                               uint nSourceWords, uint nTargetWords,
                               SingleWordDictionary& dict,
                               uint nIterations,
                               std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_ref_alignments,
                               std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_ref_alignments,
                               const floatSingleWordDictionary& prior_weight,
                               bool smoothed_l0 = false, double l0_beta = 1.0);

void dict_m_step(const SingleWordDictionary& fdict_count, 
                 const floatSingleWordDictionary& prior_weight,
                 SingleWordDictionary& dict, double alpha, uint nIter = 100,
                 bool smoothed_l0 = false, double l0_beta = 1.0);



void ibm1_viterbi_training(const Storage1D<Storage1D<uint> >& source, 
                           const Storage1D<Math2D::Matrix<uint> >& slookup,
                           const Storage1D<Storage1D<uint> >& target,
                           const CooccuringWordsType& cooc, 
                           uint nSourceWords, uint nTargetWords,
                           SingleWordDictionary& dict,
                           uint nIterations,
                           std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_ref_alignments,
                           std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_ref_alignments,
                           const floatSingleWordDictionary& prior_weight);

double prob_penalty(double x, double beta);

double prob_pen_prime(double x, double beta);


#endif
