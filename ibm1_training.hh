/*** written by Thomas Schoenemann. Started as a private person without employment, October 2009 
 *** continued at Lund University, Sweden, 2010, as a private person, and at the University of Düsseldorf, Germany, 2012 ***/


#ifndef IBM1_TRAINING_HH
#define IBM1_TRAINING_HH


#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>


class IBM1Options {
public:

  IBM1Options(uint nSourceWords,uint nTargetWords,
              std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
              std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments);

  uint nIterations_;

  bool smoothed_l0_;
  double l0_beta_;

  bool print_energy_;

  uint nSourceWords_; 
  uint nTargetWords_;

  uint dict_m_step_iter_;

  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments_;
  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments_;
};


void train_ibm1(const Storage1D<Storage1D<uint> >& source, 
                const LookupTable& slookup,
                const Storage1D<Storage1D<uint> >& target,
                const CooccuringWordsType& cooc, 
                SingleWordDictionary& dict,
                const floatSingleWordDictionary& prior_weight,
                IBM1Options& options);

void train_ibm1_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source, 
                               const LookupTable& slookup,
                               const Storage1D<Storage1D<uint> >& target,
                               const CooccuringWordsType& cooc, 
                               SingleWordDictionary& dict,
                               const floatSingleWordDictionary& prior_weight,
                               IBM1Options& options);

void ibm1_viterbi_training(const Storage1D<Storage1D<uint> >& source, 
                           const LookupTable& slookup,
                           const Storage1D<Storage1D<uint> >& target,
                           const CooccuringWordsType& cooc, 
                           SingleWordDictionary& dict,
                           const floatSingleWordDictionary& prior_weight,
                           IBM1Options& options,
			   const Math1D::Vector<double>& log_table);


#endif
