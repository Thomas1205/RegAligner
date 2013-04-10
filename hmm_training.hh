/*** written by Thomas Schoenemann. Started as a private person without employment, October 2009 
 *** continued at Lund University, Sweden, 2010, as a private person, and at the University of Düsseldorf, Germany, 2012 ***/


#ifndef HMM_TRAINING_HH
#define HMM_TRAINING_HH


#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

enum IBM1TransferMode {IBM1TransferNo, IBM1TransferViterbi, IBM1TransferPosterior, IBM1TransferInvalid};

class HmmOptions {
public:

  HmmOptions(uint nSourceWords,uint nTargetWords,
             std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
             std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments);

  uint nIterations_;
  HmmInitProbType init_type_;
  HmmAlignProbType align_type_;

  bool start_empty_word_;
  bool smoothed_l0_;
  double l0_beta_;

  bool print_energy_;

  uint nSourceWords_; 
  uint nTargetWords_;

  uint init_m_step_iter_;
  uint align_m_step_iter_;
  uint dict_m_step_iter_;

  IBM1TransferMode transfer_mode_;

  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments_;
  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments_;
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

long double hmm_alignment_prob(const Storage1D<uint>& source, 
                               const SingleLookupTable& slookup,
                               const Storage1D<uint>& target,
                               const SingleWordDictionary& dict,
                               const FullHMMAlignmentModel& align_model,
                               const InitialAlignmentProbability& initial_prob,
                               const Storage1D<AlignBaseType>& alignment, bool with_dict = false);

void external2internal_hmm_alignment(const Storage1D<AlignBaseType>& ext_alignment, uint curI,
				     const HmmOptions& options, Storage1D<AlignBaseType>& int_alignment);

void train_extended_hmm(const Storage1D<Storage1D<uint> >& source,
                        const LookupTable& slookup,
                        const Storage1D<Storage1D<uint> >& target,
                        const CooccuringWordsType& wcooc,
                        FullHMMAlignmentModel& align_model,
                        Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                        Math1D::Vector<double>& source_fert,
                        InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params,
                        SingleWordDictionary& dict,
                        const floatSingleWordDictionary& prior_weight, 
                        HmmOptions& options);


void train_extended_hmm_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source,
                                       const LookupTable& slookup,
                                       const Storage1D<Storage1D<uint> >& target,
                                       const CooccuringWordsType& wcooc,
                                       FullHMMAlignmentModel& align_model,
                                       Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                       Math1D::Vector<double>& source_fert,
                                       InitialAlignmentProbability& initial_prob,
                                       Math1D::Vector<double>& init_params,
                                       SingleWordDictionary& dict,
                                       const floatSingleWordDictionary& prior_weight,
                                       HmmOptions& options);


void viterbi_train_extended_hmm(const Storage1D<Storage1D<uint> >& source,
                                const LookupTable& slookup,
                                const Storage1D<Storage1D<uint> >& target,
                                const CooccuringWordsType& wcooc,
                                FullHMMAlignmentModel& align_model,
                                Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                Math1D::Vector<double>& source_fert,
                                InitialAlignmentProbability& initial_prob, 
                                Math1D::Vector<double>& init_params,
                                SingleWordDictionary& dict, 
                                const floatSingleWordDictionary& prior_weight,
                                bool deficient_parametric, HmmOptions& options,
				const Math1D::Vector<double>& log_table);

void par2nonpar_hmm_init_model(const Math1D::Vector<double>& init_params, const Math1D::Vector<double>& source_fert,
                               HmmInitProbType init_type, InitialAlignmentProbability& initial_prob, bool start_empty_word = false);

void par2nonpar_hmm_alignment_model(const Math1D::Vector<double>& dist_params, const uint zero_offset,
                                    const double dist_grouping_param, const Math1D::Vector<double>& source_fert,
                                    HmmAlignProbType align_type, FullHMMAlignmentModel& align_model);

void ehmm_m_step(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset,
                 uint nIter, double& grouping_param);

void ehmm_init_m_step(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter);

#endif
