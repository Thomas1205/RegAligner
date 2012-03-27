/*** written by Thomas Schoenemann as a private person without employment, October 2009 
 *** and later by Thomas Schoenemann as an employee of Lund University, 2010 ***/


#ifndef HMM_TRAINING_HH
#define HMM_TRAINING_HH


#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>



void train_extended_hmm(const Storage1D<Storage1D<uint> >& source,
                        const Storage1D<Math2D::Matrix<uint> >& slookup,
                        const Storage1D<Storage1D<uint> >& target,
                        const CooccuringWordsType& wcooc,
                        uint nSourceWords, uint nTargetWords,
                        FullHMMAlignmentModel& align_model,
                        Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                        Math1D::Vector<double>& source_fert,
                        InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params,
                        SingleWordDictionary& dict,
                        uint nIterations, HmmInitProbType init_type, HmmAlignProbType align_type,
                        std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                        std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                        const floatSingleWordDictionary& prior_weight, 
                        bool smoothed_l0 = false, double l0_beta = 1.0);


void train_extended_hmm_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source,
                                       const Storage1D<Math2D::Matrix<uint> >& slookup,
                                       const Storage1D<Storage1D<uint> >& target,
                                       const CooccuringWordsType& wcooc,
                                       uint nSourceWords, uint nTargetWords,
                                       FullHMMAlignmentModel& align_model,
                                       Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                       Math1D::Vector<double>& source_fert,
                                       InitialAlignmentProbability& initial_prob,
                                       Math1D::Vector<double>& init_params,
                                       SingleWordDictionary& dict,
                                       uint nIterations, HmmInitProbType init_type, HmmAlignProbType align_type,
                                       std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                                       std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                                       const floatSingleWordDictionary& prior_weight,
                                       bool smoothed_l0 = false, double l0_beta = 1.0);


void viterbi_train_extended_hmm(const Storage1D<Storage1D<uint> >& source,
                                const Storage1D<Math2D::Matrix<uint> >& slookup,
                                const Storage1D<Storage1D<uint> >& target,
                                const CooccuringWordsType& wcooc,
                                uint nSourceWords, uint nTargetWords,
                                FullHMMAlignmentModel& align_model,
                                Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                Math1D::Vector<double>& source_fert,
                                InitialAlignmentProbability& initial_prob,
                                SingleWordDictionary& dict, uint nIterations, 
                                HmmInitProbType init_type, HmmAlignProbType align_type, bool deficient_parametric,
                                std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                                std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                                const floatSingleWordDictionary& prior_weight);

void par2nonpar_hmm_init_model(const Math1D::Vector<double>& init_params, const Math1D::Vector<double>& source_fert,
                               HmmInitProbType init_type, InitialAlignmentProbability& initial_prob);

void par2nonpar_hmm_alignment_model(const Math1D::Vector<double>& dist_params, const uint zero_offset,
                                    const double dist_grouping_param, const Math1D::Vector<double>& source_fert,
                                    HmmAlignProbType align_type, FullHMMAlignmentModel& align_model);

void symtrain_extended_hmm(const Storage1D<Storage1D<uint> >& source,
                           const Storage1D<Math2D::Matrix<uint> >& slookup,
                           const Storage1D<Math2D::Matrix<uint> >& tlookup,
                           const Storage1D<Storage1D<uint> >& target,
                           uint nSourceWords, uint nTargetWords,
                           FullHMMAlignmentModel& s2t_align_model, FullHMMAlignmentModel& t2s_align_model,
                           InitialAlignmentProbability& s2t_initial_prob, InitialAlignmentProbability& t2s_initial_prob,
                           SingleWordDictionary& s2t_dict, SingleWordDictionary& t2s_dict,
                           uint nIterations, double gamma,
                           std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                           std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments);


#endif
