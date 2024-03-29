/*** written by Thomas Schoenemann. Started as a private person, October 2009
 *** continued at Lund University, Sweden, 2010, as a private person, at the University of Düsseldorf, Germany, 2012 ,
 *** and since as a private person ***/

#ifndef IBM1_TRAINING_HH
#define IBM1_TRAINING_HH

#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

class IBM1Options {
public:

  IBM1Options(uint nSourceWords, uint nTargetWords, RefAlignmentStructure& sure_ref_alignments,
              RefAlignmentStructure& possible_ref_alignments);

  uint nIterations_;

  bool smoothed_l0_;
  bool uniform_dict_init_ = true;
  double l0_beta_;
  double p0_ = -1.0;
  double gd_stepsize_ = 1.0;

  bool print_energy_;

  uint nSourceWords_;
  uint nTargetWords_;

  uint dict_m_step_iter_ = 45;

  bool unconstrained_m_step_ = false;

  RefAlignmentStructure& sure_ref_alignments_;
  RefAlignmentStructure& possible_ref_alignments_;
};

void train_ibm1(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                const CooccuringWordsType& cooc, SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                const IBM1Options& options);

void train_ibm1_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                               const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& cooc,
                               SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                               const IBM1Options& options);

void train_ibm1_lbfgs_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                  const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& cooc,
                                  SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                  const IBM1Options& options, uint L = 5);

void train_ibm1_projected_lbfgs_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
    const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& cooc,
    SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
    const IBM1Options& options, uint L = 5);


void ibm1_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                           const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& cooc,
                           SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                           const IBM1Options& options, const Math1D::Vector<double>& xlogx_table);

#endif
