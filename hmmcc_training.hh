/**** written by Thomas Schoenemann, September 2022 ****/

#ifndef HMMCC_TRAINING
#define HMMCC_TRAINING

#include "hmm_training.hh"

class HmmWrapperDoubleClasses : public HmmWrapperBase {
public:

  HmmWrapperDoubleClasses(const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                          const Math1D::Vector<double>& source_fert, const InitialAlignmentProbability& initial_prob,
                          const Storage1D<WordClassType>& source_class, const Storage1D<WordClassType>& target_class,
                          const HmmOptions& hmm_options, uint zero_offset);

  virtual long double compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
      const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
      Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode = false, bool verbose = false,
      double min_dict_entry = 1e-15) const;

  virtual void compute_ehmm_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
      const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
      std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
      double threshold = 0.25) const;

  virtual void fill_dist_params(Math1D::Vector<double>& hmm_dist_params, double& hmm_dist_grouping_param) const;

  virtual void fill_dist_params(uint nTargetClasses, Math2D::Matrix<double>& hmmc_dist_params, Math1D::Vector<double>& hmmc_dist_grouping_param) const;

  virtual void fill_dist_params(uint nSourceClasses, uint nTargetClasses,
                                Storage2D<Math1D::Vector<double> >& hmmcc_dist_params, Math2D::Matrix<double>& hmmcc_dist_grouping_param) const;

  void set_zero_offset(uint offs)
  {
    zero_offset_ = offs;
  }

  const Storage2D<Math1D::Vector<double> >& dist_params_;
  const Math2D::Matrix<double>& dist_grouping_param_;
  const Math1D::Vector<double>& source_fert_;
  uint zero_offset_;

  const InitialAlignmentProbability& initial_prob_;
  const Storage1D<WordClassType>& source_class_;
  const Storage1D<WordClassType>& target_class_;
};

long double hmmcc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                 const Storage1D<uint>& target, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                 const SingleWordDictionary& dict, const Storage2D<Math2D::Matrix<double> >& align_model,
                                 const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                 bool with_dict = false);

//no need for a target class dimension, class is detirmened by i_prev
long double hmmcc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                 const Storage1D<uint>& target, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                 const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
                                 const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                 bool with_dict = false);

//no need for a target class dimension, class is detirmened by i_prev
long double hmmcc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                 const Storage1D<uint>& target, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                 const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
                                 const Math1D::Vector<double>& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                 bool with_dict = false);


void par2nonpar_hmmcc_alignment_model(Math1D::Vector<uint>& sclass, Math1D::Vector<uint>& tclass, const Math1D::Vector<double>& source_fert,
                                      const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                                      Storage2D<Math2D::Matrix<double> >& align_model, HmmAlignProbType align_type, bool deficient, int redpar_limit,
                                      uint zero_offset);

//no need for a target class dimension, class is detirmened by i_prev
void par2nonpar_hmmcc_alignment_model(Math1D::Vector<uint>& sclass, Math1D::Vector<uint>& tclass, const Math1D::Vector<double>& source_fert,
                                      const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                                      Math3D::Tensor<double>& align_model, HmmAlignProbType align_type, bool deficient, int redpar_limit,
                                      uint zero_offset);

void train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                        const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                        const Math1D::Vector<WordClassType>& source_class, const Math1D::Vector<WordClassType>& target_class,
                        Storage2D<Math1D::Vector<double> >& dist_params, Math2D::Matrix<double>& dist_grouping_param,
                        Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                        SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight, const HmmOptions& options, uint maxAllI);

void viterbi_train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                const Math1D::Vector<WordClassType>& source_class, const Math1D::Vector<WordClassType>& target_class,
                                Storage2D<Math1D::Vector<double> >& dist_params, Math2D::Matrix<double>& dist_grouping_param,
                                Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                                SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                const HmmOptions& options, const Math1D::Vector<double>& xlogx_table, uint maxAllI);

#endif