/**** written by Thomas Schoenemann as a private person, since 2019 ****/

#ifndef HMMC_TRAINING_HH
#define HMMC_TRAINING_HH

#include "hmm_training.hh"

class HmmWrapperWithClasses {
public:

  HmmWrapperWithClasses(const FullHMMAlignmentModelSingleClass& align_model,
                        const InitialAlignmentProbability& initial_prob,
                        const Storage1D<WordClassType>& target_class,
                        const HmmOptions& hmm_options);

  const FullHMMAlignmentModelSingleClass& align_model_;
  const InitialAlignmentProbability& initial_prob_;
  const HmmOptions& hmm_options_;
  const Storage1D<WordClassType>& target_class_;
};

long double hmm_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                               const Storage1D<uint>& target, const Storage1D<uint>& tclass,
                               const SingleWordDictionary& dict, const FullHMMAlignmentModelSingleClass& align_model,
                               const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                               bool with_dict = false);

void train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                        const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                        const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                        Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                        Math1D::Vector<double>& source_fert,InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                        const floatSingleWordDictionary& prior_weight, const HmmOptions& options);

void train_extended_hmm_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                       const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                       const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                                       Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                                       Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob,
                                       Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                                       const floatSingleWordDictionary& prior_weight, const HmmOptions& options);

void viterbi_train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                                Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                                Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob,
                                Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                                const floatSingleWordDictionary& prior_weight, const HmmOptions& options,
                                const Math1D::Vector<double>& xlogx_table);

void par2nonpar_hmm_alignment_model(const Math2D::Matrix<double>& dist_params, const uint zero_offset,
                                    const Math1D::Vector<double>& dist_grouping_param, const Math1D::Vector<double>& source_fert,
                                    HmmAlignProbType align_type, bool deficient, int redpar_limit,
                                    FullHMMAlignmentModelSingleClass& align_model);

void ehmm_m_step(const FullHMMAlignmentModelSingleClass& facount, Math2D::Matrix<double>& dist_params, uint zero_offset,
                 uint nIter, Math1D::Vector<double>& grouping_param, bool deficient, int redpar_limit);

#endif
