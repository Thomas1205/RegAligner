/** author: Thomas Schoenemann. This file was generated from singleword_fertility_training while
    Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012. It was subsequently
    modified and extended, both at the University of Düsseldorf and since as a private person. ***/

#ifndef IBM3_TRAINING_HH
#define IBM3_TRAINING_HH

#include "singleword_fertility_training.hh"
#include "hmm_fert_interface.hh"
#include "trimatrix.hh"
#include "swb_alignment_constraints.hh"

class CompactAlignedSourceWords;
class CountStructure;

class IBM3Trainer : public FertilityModelTrainer, protected IBMConstraintStates {
public:

  IBM3Trainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
              const Storage1D<Math1D::Vector<uint> >& target_sentence, const Math1D::Vector<WordClassType>& target_class,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
              SingleWordDictionary& dict, const CooccuringWordsType& wcooc,
              const Math1D::Vector<uint>& tfert_class, uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
              const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
              const FertModelOptions& options, bool extra_deficient);

  virtual std::string model_name() const;

  void init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperWithClasses* passed_wrapper,
                           bool clear_prev = true, bool count_collection = false, bool viterbi = false);

  //training without constraints on uncovered positions.
  //This is based on the EM-algorithm, where the E-step uses heuristics
  void train_em(uint nIter, FertilityModelTrainerBase* prev_model = 0, const HmmWrapperWithClasses* passed_wrapper = 0);

  //unconstrained Viterbi training
  void train_viterbi(uint nIter, const AlignmentSetConstraints& align_constraints,
                     FertilityModelTrainerBase* prev_model = 0, const HmmWrapperWithClasses* passed_wrapper = 0);

  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
      Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double>& swap_prob,
      Math1D::Vector<AlignBaseType>& alignment) const;

  virtual long double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment);

protected:

  double compute_approximate_marginals(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                       const Storage1D<Math1D::Vector<double> >& fert_prob, const ReducedIBM3ClassDistortionModel& distort_prob,
                                       double p_zero, Math2D::Matrix<double>& j_marg, Math2D::Matrix<double>& i_marg,
                                       double hc_mass, bool& converged) const;

  void compute_dist_param_gradient(const ReducedIBM3ClassDistortionModel& distort_grad, const Math3D::Tensor<double>& distort_param,
                                   Math3D::Tensor<double>& distort_param_grad) const;

  void update_distortion_probs(const ReducedIBM3ClassDistortionModel& fdistort_count, ReducedIBM3ClassDistortionModel& fnondef_distort_count,
                               Storage1D<std::map<CompactAlignedSourceWords,CountStructure> >& refined_nondef_aligned_words_count);

  void compute_alignment_via_msg_passing(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                         Math1D::Vector<AlignBaseType>& alignment);

  void par2nonpar_distortion(ReducedIBM3ClassDistortionModel& prob);

  void par2nonpar_distortion(const Math3D::Tensor<double>& param, ReducedIBM3ClassDistortionModel& prob) const;

  double par_distortion_m_step_energy(const Math1D::Vector<double>& fsingleton_count, const Math1D::Vector<double>& fspan_count,
                                      const Math1D::Vector<double>& param) const;

  double diffpar_distortion_m_step_energy(const Math3D::Tensor<double>& fsingleton_count, const Math3D::Tensor<double>& fspan_count,
                                          const Math3D::Tensor<double>& param, uint c) const;

  void par_distortion_m_step(const Math3D::Tensor<double>& fsingleton_count, const Math3D::Tensor<double>& fspan_count, uint i, uint c);

  void diffpar_distortion_m_step(const Math3D::Tensor<double>& fsingleton_count, const Math3D::Tensor<double>& fspan_count, uint c);

  void par_distortion_m_step_unconstrained(const Math3D::Tensor<double>& fsingleton_count,
      const Math3D::Tensor<double>& fspan_count, uint i, uint c, uint L = 5);

  //compact form
  double nondeficient_m_step_energy(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
                                    const std::vector<double>& sum_pos_count, const Math3D::Tensor<double>& param, uint i, uint c) const;

  //compact form
  double nondeficient_diffpar_m_step_energy(const Math3D::Tensor<double>& fsingleton_count,
      const Storage1D<std::vector<Math1D::Vector<ushort,uchar> > >& open_pos, const Storage1D<std::vector<double> >& sum_pos_count,
      const Math3D::Tensor<double>& param) const;

  //compact form with interpolation
  double nondeficient_m_step_energy(const double* single_pos_count, const uint J, const std::vector<double>& sum_pos_count,
                                    const double* param1, const Math1D::Vector<double>& param2,
                                    const Math1D::Vector<double>& sum1, const Math1D::Vector<double>& sum2, double lambda) const;

  //compact form
  double nondeficient_m_step_core(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
                                  const std::vector<double>& sum_pos_count, Math3D::Tensor<double>& param, uint i, uint c,
                                  double start_energy, bool quiet = false);

  double nondeficient_m_step_unconstrained_core(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
      const std::vector<double>& sum_pos_count, Math3D::Tensor<double>& param, uint i, uint c,
      double start_energy, bool quiet = false, uint L = 5);

  //compact form
  double nondeficient_m_step(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
                             const std::vector<double>& sum_pos_count, uint i, uint c, double start_energy);

  //compact form
  double nondeficient_diffpar_m_step(const Math3D::Tensor<double>& fsingleton_count,
                                     const Storage1D<std::vector<Math1D::Vector<ushort,uchar> > >& open_pos,
                                     const Storage1D<std::vector<double> >& sum_pos_count, double start_energy);

  double nondeficient_m_step_unconstrained(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
      const std::vector<double>& sum_pos_count, uint i, double start_energy, bool quiet = false, uint L = 5);

  //compact form
  double nondeficient_m_step_with_interpolation_core(const Math3D::Tensor<double>& single_pos_count,
      const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, const std::vector<double>& weight,
      Math3D::Tensor<double>& param, uint i, uint c, double start_energy, bool quiet = false);

  //compact form
  double nondeficient_m_step_with_interpolation(const Math3D::Tensor<double>& single_pos_count,
      const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, const std::vector<double>& count,
      uint i, uint c, double start_energy);

  double nondeficient_m_step_with_interpolation(const Math3D::Tensor<double>& single_pos_count,
      const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, const std::vector<double>& weight,
      uint i, uint c, uint J, double start_energy);

  //compact form for the nonparametric setting
  // make sure that you pass the count corresponding to J
  double nondeficient_m_step(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
                             const std::vector<double>& sum_pos_count, uint i, uint c, uint J, double start_energy);

  long double alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const;

  virtual long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                     const Math1D::Vector<AlignBaseType>& alignment) const;

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                             const Math1D::Vector<AlignBaseType>& alignment, const Math3D::Tensor<double>* distort_prob) const;

  long double nondeficient_alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const;

  long double nondeficient_alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                          const Math1D::Vector<AlignBaseType>& alignment) const;

  //this EXCLUDES the empty word
  long double nondeficient_distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) const;

  long double nondeficient_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                        uint& nIter, Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob,
                                        Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment) const;

  //@param time_limit: maximum amount of seconds spent in the ILP-solver.
  //          values <= 0 indicate that no time limit is set
  //@returns the energy (log prob) of the viterbi alignment
  double compute_viterbi_alignment_ilp(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                       Math1D::Vector<AlignBaseType>& alignment, double time_limit = -1.0);

  void itg_traceback(const NamedStorage2D<Math2D::TriMatrix<uint> >& trace,
                     const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup, uint J,
                     uint j, uint i, uint ii, Math1D::Vector<AlignBaseType>& alignment) const;

  long double compute_itg_viterbi_alignment_noemptyword(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      Math1D::Vector<AlignBaseType>& alignment, uint ext_level = 0, int max_mid_dev = 10000,
      uint level3_maxlength = 8) const;

  long double ibmconstrained_viterbi_subprob_noemptyword(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      uint j1, uint j2, uint i1, uint i2, Math2D::Matrix<long double>* scoremat = 0) const;

  long double compute_ibmconstrained_viterbi_alignment_noemptyword(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      Math1D::Vector<AlignBaseType>& alignment, const Math3D::Tensor<double>* distort_prob = 0) const;

  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                          Math1D::Vector<AlignBaseType>& alignment);

  const Math1D::Vector<WordClassType>& target_class_;

  ReducedIBM3ClassDistortionModel distortion_prob_;
  Math3D::Tensor<double> distortion_param_;

  IBM23ParametricMode par_mode_;
  bool extra_deficiency_;

  IlpMode viterbi_ilp_mode_;
  bool utmost_ilp_precision_;
  bool nondeficient_;
};

void add_nondef_count_compact(const Storage1D<std::vector<uchar> >& aligned_source_words, uint i, uint c, uint J, uint maxJ,
                              double count, std::map<Math1D::Vector<uchar,uchar>,double >& count_map,
                              Math3D::Tensor<double>& par_count);

//for diffpar
void add_nondef_count_compact_diffpar(const Storage1D<std::vector<uchar> >& aligned_source_words,
                                      const Storage1D<WordClassType>& tclass, uint J, uint offset, double count,
                                      Storage1D<std::map<Math1D::Vector<ushort,uchar>, double> >& count_map, Math3D::Tensor<double>& par_count);

#endif
