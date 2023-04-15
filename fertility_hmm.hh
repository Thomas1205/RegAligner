/***** written by Thomas Schoenemann as a private person, since August 2018 ****/

#ifndef FERTILITY_HMM_HH
#define FERTILITY_HMM_HH

#include "singleword_fertility_training.hh"
#include "hmm_fert_interface.hh"

class FertilityHMMTrainer:public FertilityModelTrainer {
public:

  FertilityHMMTrainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                      const Storage1D<Math1D::Vector<uint> >& target_sentence, const Math1D::Vector<WordClassType>& target_class,
                      const RefAlignmentStructure& sure_ref_alignments, const RefAlignmentStructure& possible_ref_alignments,
                      SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class,
                      uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
                      const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
                      const HmmOptions& options, const FertModelOptions& fert_options, bool no_factorial = false);

  virtual std::string model_name() const override;

  void init_from_hmm(HmmFertInterfaceTargetClasses& wrapper, const Math2D::Matrix<double>& dist_params, const Math1D::Vector<double>& dist_grouping_param,
                     bool clear_prev = true, bool count_collection = false, bool viterbi = false);

  void init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper,
                           const Math2D::Matrix<double>& dist_params, const Math1D::Vector<double>& dist_grouping_param,
                           bool clear_prev = true, bool count_collection = false, bool viterbi = false);

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

  //returns the logarithm of the (approximated) normalization constant
  virtual double compute_approximate_marginals(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg, Math2D::Matrix<double>& i_marg,
      double hc_mass, bool& converged) const override;

protected:

  double compute_approximate_marginals(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                       Math1D::Vector<AlignBaseType>& alignment, const Storage1D<Math1D::Vector<double> >& fert_prob,
                                       const Math3D::Tensor<double>& align_prob, double p_zero, Math2D::Matrix<double>& j_marg, Math2D::Matrix<double>& i_marg,
                                       Math2D::Matrix<double>& align_expectation, double hc_mass, bool& converged) const;

  void compute_dist_param_gradient(const Math2D::Matrix<double>& dist_params, const Math1D::Vector<double>& dist_grouping_param,
                                   const FullHMMAlignmentModelSingleClass& align_grad, Math2D::Matrix<double>& distort_param_grad,
                                   Math1D::Vector<double>& dist_grouping_grad, uint zero_offset) const;

  long double hmm_prob(const Storage1D<uint>& target, const Math1D::Vector<AlignBaseType>& alignment) const;

  virtual long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                     const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment) const override;

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                             const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment, double p_zero,
                             const Math3D::Tensor<double>& distort_prob, const Storage1D<Math1D::Vector<double> >& fertility_prob,
                             bool with_dict = true) const;

  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                          const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment) override;

  void par2nonpar();

  void par2nonpar(const Math2D::Matrix<double> dist_params, const Math1D::Vector<double>& dist_grouping_param,
                  FullHMMAlignmentModelSingleClass& align_model) const;

  const HmmOptions& options_;
  FullHMMAlignmentModelSingleClass align_model_;
  Math2D::Matrix<double> dist_params_;
  Math1D::Vector<double> dist_grouping_param_;

  const Math1D::Vector<WordClassType>& target_class_;
};

#endif
