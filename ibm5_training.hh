/**** written by Thomas Schoenemann as a private person, since February 2013 ****/

#ifndef IBM5_TRAINING_HH
#define IBM5_TRAINING_HH

#include "ibm4_training.hh"

class IBM5Trainer : public FertilityModelTrainer {
public:

  IBM5Trainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
              const Storage1D<Math1D::Vector<uint> >& target_sentence,
              const RefAlignmentStructure& sure_ref_alignments, const RefAlignmentStructure& possible_ref_alignments,
              SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class, uint nSourceWords,
              uint nTargetWords, const floatSingleWordDictionary& prior_weight,
              const Storage1D<WordClassType>& source_class, const Storage1D<WordClassType>& target_class,
              const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
              const FertModelOptions& options);

  virtual std::string model_name() const override;

  void init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper,
                           bool clear_prev = true, bool count_collection = false, bool viterbi = false);

  void train_em(uint nIter, FertilityModelTrainerBase* prev_model = 0, const HmmWrapperBase* passed_wrapper = 0);

  void train_viterbi(uint nIter, FertilityModelTrainerBase* prev_model = 0, const HmmWrapperBase* passed_wrapper = 0);

  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
      Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double>& swap_prob,
      Math1D::Vector<AlignBaseType>& alignment) const override;

protected:

  virtual long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                     const Math1D::Vector<ushort>& alignment) const override;

  //NOTE: the vectors need to be sorted
  long double distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                              const Storage1D<std::vector<ushort> >& aligned_source_words) const;

  void par2nonpar_inter_distortion();

  double inter_distortion_diffpar_m_step_energy(const Math2D::Matrix<double>& single_diff_count, const Math3D::Tensor<double>& diff_span_count,
												uint sclass, const Math2D::Matrix<double>& param, double grouping_prob) const;

  double inter_distortion_pospar_m_step_energy(const Math3D::Tensor<double>& single_pos_count, const Math3D::Tensor<double>& pos_span_count,
											   uint sclass, uint prev_center, const Math3D::Tensor<double>& param, double grouping_prob) const;

  void inter_distortion_diffpar_m_step(const Math2D::Matrix<double>& single_diff_count, const Math3D::Tensor<double>& diff_span_count,
									   uint sclass);

  void inter_distortion_pospar_m_step(const Math3D::Tensor<double>& single_pos_count, const Math3D::Tensor<double>& pos_span_count,
									  uint sclass, uint prev_center);

  void inter_distortion_diffpar_m_step_unconstrained(const Math2D::Matrix<double>& single_diff_count,
      const Math3D::Tensor<double>& diff_span_count, uint sclass, uint L = 5);

  void par2nonpar_intra_distortion();

  double intra_distortion_m_step_energy(const Math2D::Matrix<double>& single_diff_count,
                                        const Math2D::Matrix<double>& diff_span_count, uint sclass,
                                        const Math2D::Matrix<double>& param) const;

  void intra_distortion_m_step(const Math2D::Matrix<double>& single_diff_count,
                               const Math2D::Matrix<double>& diff_span_count, uint sclass);

  void intra_distortion_m_step_unconstrained(const Math2D::Matrix<double>& single_diff_count, const Math2D::Matrix<double>& diff_span_count, uint sclass, uint L = 5);

  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                          Math1D::Vector<AlignBaseType>& alignment) override;

  //dependence on the previous center (yDim) and source word classes (zDim)
  Storage1D<Math3D::Tensor<double> > inter_distortion_prob_;
  Math2D::Matrix<double> inter_distortion_param_;
  Math3D::Tensor<double> inter_dist_pospar_param_;

  uint displacement_offset_;

  //dependence on source word classes (yDim)
  Storage1D<Math2D::Matrix<double> > intra_distortion_prob_;
  Math2D::Matrix<double> intra_distortion_param_;

  Math1D::NamedVector<double> sentence_start_parameters_;
  Storage1D<Math1D::Vector<double> > sentence_start_prob_;

  IBM23ParametricMode distortion_type_;
  //bool nonpar_distortion_;
 
  bool use_sentence_start_prob_;
  bool uniform_intra_prob_;

  IBM4CeptStartMode cept_start_mode_;
  IBM4IntraDistMode intra_dist_mode_;
  
  Math1D::Vector<double> inter_grouping_prob_; //relevant when inter_dist_grouping_ isn't Off
  Math2D::Matrix<double> inter_pospar_grouping_prob_;

  Storage1D<WordClassType> source_class_;
  Storage1D<WordClassType> target_class_;

  uint nSourceClasses_;
  uint nTargetClasses_;

  bool deficient_;
  DistortionGroupingMode inter_dist_grouping_;
  //bool inter_dist_grouping_ = false; // currently only for the inter probs, not the intra probs
  int dist_grouping_limit_ = 5; 

  uint dist_m_step_iter_ = 400;
  uint start_m_step_iter_ = 100;
};

#endif
