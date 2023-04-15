/**** written by Thomas Schoenemann as a private person, since 2019 ****/

#ifndef HMMC_TRAINING_HH
#define HMMC_TRAINING_HH

#include "hmm_training.hh"

class HmmWrapperWithTargetClasses : public HmmWrapperBase {
public:

  HmmWrapperWithTargetClasses(const FullHMMAlignmentModelSingleClass& align_model, const FullHMMAlignmentModelSingleClass& dev_align_model,
                              const InitialAlignmentProbability& initial_prob,
                              const Math2D::Matrix<double>& hmmc_dist_params, const Math1D::Vector<double>& hmmc_dist_grouping_param,
                              const Storage1D<WordClassType>& target_class,
                              const HmmOptions& hmm_options);

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

  const Math2D::Matrix<double>& dist_params_;
  const Math1D::Vector<double>& dist_grouping_param_;

  const FullHMMAlignmentModelSingleClass& align_model_;
  const FullHMMAlignmentModelSingleClass& dev_align_model_;
  const InitialAlignmentProbability& initial_prob_;
  const Storage1D<WordClassType>& target_class_;
};

long double hmmc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                const Storage1D<uint>& target, const Storage1D<uint>& tclass,
                                const SingleWordDictionary& dict, const FullHMMAlignmentModelSingleClass& align_model,
                                const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                bool with_dict = false);

long double hmmc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                const Storage1D<uint>& target, const Storage1D<uint>& tclass,
                                const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_model,
                                const Math1D::Vector<double>& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                bool with_dict = false);

void train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                        const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                        const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                        Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                        Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                        const floatSingleWordDictionary& prior_weight, const HmmOptions& options, uint maxAllI);

void train_extended_hmm_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                       const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                       const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                                       Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                                       Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob,
                                       Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                                       const floatSingleWordDictionary& prior_weight, const HmmOptions& options, uint maxAllI);

void viterbi_train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                                Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                                Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob,
                                Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                                const floatSingleWordDictionary& prior_weight, const HmmOptions& options,
                                const Math1D::Vector<double>& xlogx_table, uint maxAllI);

void par2nonpar_hmm_alignment_model(const Math2D::Matrix<double>& dist_params, const uint zero_offset,
                                    const Math1D::Vector<double>& dist_grouping_param, const Math1D::Vector<double>& source_fert,
                                    HmmAlignProbType align_type, bool deficient, int redpar_limit,
                                    FullHMMAlignmentModelSingleClass& align_model);

void par2nonpar_hmm_alignment_model(const Math2D::Matrix<double>& dist_params, const uint zero_offset,
                                    const Math1D::Vector<double>& dist_grouping_param, const Math1D::Vector<double>& source_fert,
                                    const Math1D::Vector<uint>& tclass, const HmmOptions& options, Math2D::Matrix<double>& align_model);

void par2nonpar_hmm_alignment_model(const Math1D::Vector<uint>& tclass, const FullHMMAlignmentModelSingleClass& nonpar_model,
                                    const HmmOptions& options, Math2D::Matrix<double>& align_model);

void par2nonpar_hmm_alignment_model(const Math1D::Vector<uint>& tclass, const Math3D::Tensor<double>& nonpar_model,
                                    const HmmOptions& options, Math2D::Matrix<double>& align_model);

void ehmm_m_step(const FullHMMAlignmentModelSingleClass& facount, Math2D::Matrix<double>& dist_params, uint zero_offset,
                 uint nIter, Math1D::Vector<double>& grouping_param, bool deficient, int redpar_limit, double gd_stepsize = 1.0, bool quiet = true);

#endif
