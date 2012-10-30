/*** ported here from singleword_fertility_training ****/
/** author: Thomas Schoenemann. This file was generated while Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012 ***/

#ifndef IBM3_TRAINING_HH
#define IBM3_TRAINING_HH


#include "singleword_fertility_training.hh"
#include "hmm_training.hh"

class IBM4Trainer;

class IBM3Trainer : public FertilityModelTrainer {
public:
  
  IBM3Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
              const Storage1D<Math2D::Matrix<uint> >& slookup,
              const Storage1D<Storage1D<uint> >& target_sentence,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
              SingleWordDictionary& dict,
              const CooccuringWordsType& wcooc,
              uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
              bool parametric_distortion = false,
              bool och_ney_empty_word = true,
              bool viterbi_ilp = false, double l0_fertpen = 0.0,
              bool smoothed_l0 = false, double l0_beta = 1.0);
  
  void init_from_hmm(const FullHMMAlignmentModel& align_model,
                     const InitialAlignmentProbability& initial_prob, const HmmOptions& hmm_options);

  //training without constraints on maximal fertility or uncovered positions.
  //This is based on the EM-algorithm, where the E-step uses heuristics
  void train_unconstrained(uint nIter);

  //unconstrained Viterbi training
  void train_viterbi(uint nIter, bool use_ilp = false);

  //training for IBM reordering constraints. This is done exactly
  void train_with_ibm_constraints(uint nIter, uint maxFertility, uint nMaxSkips = 4, bool verbose = false);

  void train_with_itg_constraints(uint nIter, bool extended_reordering = false, bool verbose = false);

  void update_alignments_unconstrained();

  double p_zero() const;

  void fix_p0(double p0);

  double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                    const Math2D::Matrix<uint>& lookup,
                                    Math1D::Vector<AlignBaseType>& alignment, bool ilp=false);

  // <code> start_alignment </code> is used as initialization for hillclimbing and later modified
  // the extracted alignment is written to <code> postdec_alignment </code>
  void compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
					  const Math2D::Matrix<uint>& lookup,
					  Math1D::Vector<AlignBaseType>& start_alignment,
					  std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
					  double threshold = 0.25);

  void release_memory();

  void write_postdec_alignments(const std::string filename, double thresh);
  
protected:
  
  friend class IBM4Trainer;

  void par2nonpar_distortion(ReducedIBM3DistortionModel& prob);

  double par_distortion_m_step_energy(const ReducedIBM3DistortionModel& fdistort_count,
                                      const Math2D::Matrix<double>& param, uint i);

  double par_distortion_m_step_energy(const ReducedIBM3DistortionModel& fdistort_count,
                                      const Math1D::Vector<double>& param, uint i);

  void par_distortion_m_step(const ReducedIBM3DistortionModel& fdistort_count, uint i);

  long double alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const;

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                             const Math2D::Matrix<uint>& lookup, const Math1D::Vector<AlignBaseType>& alignment) const;


  //improves the currently best known alignment using hill climbing and
  // returns the probability of the resulting alignment
  long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                               const Math2D::Matrix<uint>& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
                                               Math2D::Matrix<long double>& expansion_prob,
                                               Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment);


  long double compute_itg_viterbi_alignment_noemptyword(uint s, bool extended_reordering = false);

  //@param time_limit: maximum amount of seconds spent in the ILP-solver.
  //          values <= 0 indicate that no time limit is set
  long double compute_viterbi_alignment_ilp(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                            const Math2D::Matrix<uint>& lookup, uint max_fertility,
                                            Math1D::Vector<AlignBaseType>& alignment, double time_limit = -1.0);

  void itg_traceback(uint s, const NamedStorage1D<Math3D::NamedTensor<uint> >& trace, uint J, uint j, uint i, uint ii);

  long double compute_ibmconstrained_viterbi_alignment_noemptyword(uint s, uint maxFertility, uint nMaxSkips);

  ReducedIBM3DistortionModel distortion_prob_;
  Math2D::Matrix<double> distortion_param_;

  double p_zero_;
  double p_nonzero_;

  bool och_ney_empty_word_;
  const floatSingleWordDictionary& prior_weight_;
  double l0_fertpen_;
  bool parametric_distortion_;
  bool viterbi_ilp_;
  bool smoothed_l0_;
  double l0_beta_;

  bool fix_p0_;
};



#endif
