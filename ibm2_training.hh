/*** written by Thomas Schoenemann as a private person, October 2009
 *** and later by Thomas Schoenemann as employee of Lund University, 2010 and since as a private person ***/

#ifndef IBM2_TRAINING_HH
#define IBM2_TRAINING_HH

#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

class IBM2Options {
public:

  IBM2Options(uint nSourceWords, uint nTargetWords, RefAlignmentStructure& sure_ref_alignments,
              RefAlignmentStructure& possible_ref_alignments);

  uint nIterations_;

  IBM23ParametricMode ibm2_mode_ = IBM23ParByPosition;

  bool smoothed_l0_;
  double l0_beta_;
  double p0_ = 0.02;
  double gd_stepsize_ = 1.0;

  bool print_energy_;

  uint nSourceWords_;
  uint nTargetWords_;

  double ibm1_p0_ = -1.0;

  TransferMode transfer_mode_ = TransferNo;

  uint dict_m_step_iter_ = 45;
  uint align_m_step_iter_ = 400;

  bool deficient_ = false;
  bool unconstrained_m_step_ = false;

  RefAlignmentStructure& sure_ref_alignments_;
  RefAlignmentStructure& possible_ref_alignments_;
};

//train IBM-2 with the original proposal [Brown et al.] to condition on both J and I
void train_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, uint nSourceWords, uint nTargetWords,
                IBM2AlignmentModel& alignment_model, SingleWordDictionary& dict, uint nIterations,
                std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                const floatSingleWordDictionary& prior_weight, double l0_beta, bool smoothed_l0, uint dict_m_step_iter);

void par2nonpar_reduced_ibm2alignment_model(const Math3D::Tensor<double>& align_param, const Math1D::Vector<double>& source_fert,
    ReducedIBM2ClassAlignmentModel& alignment_model, IBM23ParametricMode par_mode, uint offset,
    bool deficient = false);

void train_reduced_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2ClassAlignmentModel& alignment_model,
                        Math3D::Tensor<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                        const Math1D::Vector<WordClassType>& sclass, const IBM2Options& options, const floatSingleWordDictionary& prior_weight);

void train_reduced_ibm2_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                                       const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2ClassAlignmentModel& alignment_model,
                                       Math3D::Tensor<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                                       const Math1D::Vector<WordClassType>& sclass, const IBM2Options& options, const floatSingleWordDictionary& prior_weight);

void reduced_ibm2_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                                   const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2ClassAlignmentModel& alignment_model,
                                   Math3D::Tensor<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                                   const Math1D::Vector<WordClassType>& sclass, const IBM2Options& options, const floatSingleWordDictionary& prior_weight,
                                   const Math1D::Vector<double>& xlogx_table);

void symtrain_reduced_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const LookupTable& tlookup,
                           const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& s2t_wcooc, const CooccuringWordsType& t2s_wcooc,
                           const CooccuringLengthsType& s2t_lcooc, const CooccuringLengthsType& t2s_lcooc, uint nSourceWords, uint nTargetWords,
                           ReducedIBM2AlignmentModel& s2t_alignment_model, ReducedIBM2AlignmentModel& t2s_alignment_model,
                           SingleWordDictionary& s2t_dict, SingleWordDictionary& t2s_dict, uint nIterations, double gamma,
                           std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                           std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                           bool diff_of_logs);

#endif
