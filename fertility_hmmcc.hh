/***** written by Thomas Schoenemann, October 2022 *******/

#ifndef FERTILITY_HMMCC_HH
#define FERTILITY_HMMCC_HH

#include "singleword_fertility_training.hh"
#include "hmm_fert_interface.hh"

class FertilityHMMTrainerDoubleClass:public FertilityModelTrainer {
public:

  FertilityHMMTrainerDoubleClass(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                                 const Storage1D<Math1D::Vector<uint> >& target_sentence,
                                 const Math1D::Vector<WordClassType>& source_class, const Math1D::Vector<WordClassType>& target_class,
                                 const RefAlignmentStructure& sure_ref_alignments, const RefAlignmentStructure& possible_ref_alignments,
                                 SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class,
                                 const Math1D::Vector<double>& source_fert, uint zero_offset,
                                 uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
                                 const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
                                 const HmmOptions& options, const FertModelOptions& fert_options,
                                 const Storage2D<Math1D::Vector<double> >& dist_params, Math2D::Matrix<double> dist_grouping_param,
                                 bool no_factorial = false);

  virtual std::string model_name() const override;

  //training without constraints on uncovered positions.
  //This is based on the EM-algorithm, where the E-step uses heuristics
  void train_em(uint nIter, FertilityModelTrainerBase* prev_model = 0, const HmmWrapperBase* passed_wrapper = 0);

  //unconstrained Viterbi training
  void train_viterbi(uint nIter, FertilityModelTrainerBase* prev_model = 0, const HmmWrapperBase* passed_wrapper = 0);

  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup, uint& nIter,
      Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double>& swap_prob,
      Math1D::Vector<AlignBaseType>& alignment) const override;

  virtual long double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      Math1D::Vector<AlignBaseType>& alignment, AlignmentSetConstraints* constraints = 0) override;

  //no need for a target class dimension, tclass is determined by i_prev
  virtual long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                     const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment) const override;

  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                          const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment) override;

  void init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper,
                           const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                           bool clear_prev = true, bool count_collection = false, bool viterbi = false);

protected:

  const HmmOptions& options_;
  uint zero_offset_;
  Storage2D<Math1D::Vector<double> > dist_params_;
  Math2D::Matrix<double> dist_grouping_param_;
  Math1D::Vector<double> source_fert_;

  const Math1D::Vector<WordClassType>& source_class_;
  const Math1D::Vector<WordClassType>& target_class_;

  bool compute_all_ = false;
};

#endif