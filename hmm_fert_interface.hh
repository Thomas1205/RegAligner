/**** written by Thomas Schoenemann as a private person, since October 2017 ****/

#ifndef HMM_FERT_INTERFACE
#define HMM_FERT_INTERFACE

#include "hmmc_training.hh"
#include "singleword_fertility_training.hh"

class HmmFertInterface:public FertilityModelTrainerBase {
public:

  HmmFertInterface(const HmmWrapperWithClasses& wrapper, const Storage1D<Math1D::Vector<uint> >& source_sentence,
                   const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target_sentence,
                   const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                   const std::map<uint,std::set<std::pair< AlignBaseType,AlignBaseType > > >& possible_ref_alignments,
                   SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords,
                   uint nTargetWords, uint fertility_limit = 10000);

  virtual std::string model_name() const
  {
    return "HMM";
  };

  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
      Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double >& swap_prob,
      Math1D::Vector<AlignBaseType>& alignment) const;

  const HmmWrapperWithClasses& hmm_wrapper()
  {
    return hmm_wrapper_;
  }

  virtual long double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment, AlignmentSetConstraints* constraints = 0);

  virtual void compute_approximate_jmarginals(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg,
      bool& converged) const;

protected:

  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                          const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment);

  const HmmWrapperWithClasses& hmm_wrapper_;
};

#endif
