/*** written by Thomas Schoenemann as a private person, October 2009
 *** and later by Thomas Schoenemann as employee of Lund University, 2010 and since as a private person ***/

#ifndef IBM2_TRAINING_HH
#define IBM2_TRAINING_HH

#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

void train_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, uint nSourceWords, uint nTargetWords,
                IBM2AlignmentModel& alignment_model, SingleWordDictionary& dict, uint nIterations,
                std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments);

void train_reduced_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, uint nSourceWords, uint nTargetWords,
                        ReducedIBM2AlignmentModel& alignment_model, SingleWordDictionary& dict, uint nIterations,
                        std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                        std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                        const floatSingleWordDictionary& prior_weight, double l0_beta, bool smoothed_l0, uint dict_m_step_iter);

void ibm2_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                           const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, uint nSourceWords, uint nTargetWords,
                           ReducedIBM2AlignmentModel& alignment_model, SingleWordDictionary& dict, uint nIterations,
                           std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                           std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                           const floatSingleWordDictionary& prior_weight);

void symtrain_reduced_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const LookupTable& tlookup,
                           const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& s2t_wcooc, const CooccuringWordsType& t2s_wcooc,
                           const CooccuringLengthsType& s2t_lcooc, const CooccuringLengthsType& t2s_lcooc, uint nSourceWords, uint nTargetWords,
                           ReducedIBM2AlignmentModel& s2t_alignment_model, ReducedIBM2AlignmentModel& t2s_alignment_model,
                           SingleWordDictionary& s2t_dict, SingleWordDictionary& t2s_dict, uint nIterations, double gamma,
                           std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                           std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                           bool diff_of_logs);

#endif
