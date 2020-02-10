/*** written by Thomas Schoenemann. Started as a private person, October 2009
 *** continued at Lund University, Sweden, 2010, as a private person, at the University of Düsseldorf, Germany, 2012
 *** and since as a private person ***/

#ifndef HMM_TRAINING_HH
#define HMM_TRAINING_HH

#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

class HmmOptions {
public:

  HmmOptions(uint nSourceWords, uint nTargetWords, const ReducedIBM2ClassAlignmentModel& ibm2_alignment_model, const Math1D::Vector<WordClassType>& ibm2_sclass,
             RefAlignmentStructure& sure_ref_alignments, RefAlignmentStructure& possible_ref_alignments);

  uint nIterations_;
  HmmInitProbType init_type_;
  HmmAlignProbType align_type_;

  uint redpar_limit_ = 5;
  bool start_empty_word_ = false;
  bool smoothed_l0_ = true;
  bool deficient_ = false;
  bool fix_p0_ = false;
  double l0_beta_ = -0.1;

  double ibm1_p0_ = -1.0;

  bool print_energy_ = true;

  uint nSourceWords_;
  uint nTargetWords_;

  uint init_m_step_iter_ = 100;
  uint align_m_step_iter_ = 250;
  uint dict_m_step_iter_ = 45;

  TransferMode transfer_mode_ = TransferNo;

  MStepSolveMode msolve_mode_;
  ProjectionMode projection_mode_ = Simplex;

  const ReducedIBM2ClassAlignmentModel& ibm2_alignment_model_;
  const Math1D::Vector<WordClassType>& ibm2_sclass_;

  RefAlignmentStructure& sure_ref_alignments_;
  RefAlignmentStructure& possible_ref_alignments_;
};

class HmmWrapper {
public:

  HmmWrapper(const FullHMMAlignmentModel& align_model,
             const InitialAlignmentProbability& initial_prob,
             const HmmOptions& hmm_options);

  const FullHMMAlignmentModel& align_model_;
  const InitialAlignmentProbability& initial_prob_;
  const HmmOptions& hmm_options_;
};

long double hmm_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup, const Storage1D<uint>& target,
                               const SingleWordDictionary& dict, const FullHMMAlignmentModel& align_model,
                               const InitialAlignmentProbability& initial_prob, const Storage1D < AlignBaseType >& alignment,
                               bool with_dict = false);

void external2internal_hmm_alignment(const Storage1D<AlignBaseType>& ext_alignment, uint curI,
                                     const HmmOptions& options, Storage1D<AlignBaseType>& int_alignment);

void train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const CooccuringWordsType& wcooc, FullHMMAlignmentModel& align_model, Math1D::Vector<double>& dist_params,
                        double& dist_grouping_param, Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params, SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                        const HmmOptions& options);

void train_extended_hmm_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                       const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                       FullHMMAlignmentModel& align_model, Math1D::Vector < double >& dist_params,
                                       double& dist_grouping_param, Math1D::Vector<double>& source_fert,
                                       InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                                       SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight, const HmmOptions& options);

void viterbi_train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                FullHMMAlignmentModel& align_model, Math1D::Vector<double>& dist_params,
                                double& dist_grouping_param, Math1D::Vector<double>& source_fert,
                                InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                                SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                const HmmOptions& options, const Math1D::Vector<double>& xlogx_table);

void par2nonpar_hmm_init_model(const Math1D::Vector<double>& init_params, const Math1D::Vector<double>& source_fert,
                               HmmInitProbType init_type, InitialAlignmentProbability& initial_prob,
                               bool start_empty_word = false, bool fix_p0 = false);

void par2nonpar_hmm_alignment_model(const Math1D::Vector<double>& dist_params, const uint zero_offset,
                                    const double dist_grouping_param, const Math1D::Vector<double>& source_fert,
                                    HmmAlignProbType align_type, bool deficient, FullHMMAlignmentModelNoClasses& align_model, int redpar_limit);

double ehmm_m_step_energy(const Math1D::Vector<double>& singleton_count, double grouping_count, const Math2D::Matrix<double>& span_count,
                          const Math1D::Vector<double>& dist_params, uint zero_offset, double grouping_param, int redpar_limit);

void ehmm_m_step(const FullHMMAlignmentModelNoClasses& facount, Math1D::Vector<double>& dist_params, uint zero_offset,
                 uint nIter, double& grouping_param, bool deficient, int redpar_limit, ProjectionMode projection_mode = Simplex);

void ehmm_m_step_unconstrained(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params,
                               uint zero_offset, uint nIter, double& grouping_param, bool deficient, int redpar_limit);

void ehmm_m_step_unconstrained_LBFGS(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset, uint nIter,
                                     double& grouping_param, uint L, bool deficient, int redpar_limit);

void ehmm_init_m_step(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter, ProjectionMode projection_mode = Simplex);

#endif
