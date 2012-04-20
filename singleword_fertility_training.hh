/*** written by Thomas Schoenemann as a private person without employment, November 2009 ***/


#ifndef SINGLEWORD_FERTILITY_TRAINING_HH
#define SINGLEWORD_FERTILITY_TRAINING_HH

#include "mttypes.hh"
#include "vector.hh"
#include "tensor.hh"

#include <map>
#include <set>

class FertilityModelTrainer {
public:

  FertilityModelTrainer(const Storage1D<Storage1D<uint> >& source_sentence,
                        const Storage1D<Math2D::Matrix<uint> >& slookup,
                        const Storage1D<Storage1D<uint> >& target_sentence,
                        SingleWordDictionary& dict,
                        const CooccuringWordsType& wcooc,
                        uint nSourceWords, uint nTargetWords,
                        const std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_ref_alignments,
                        const std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_ref_alignments
                        );

  void write_alignments(const std::string filename) const;

  double AER();

  double AER(const Storage1D<Math1D::Vector<ushort> >& alignments);

  double f_measure(double alpha = 0.1);

  double DAE_S();

  const NamedStorage1D<Math1D::Vector<double> >& fertility_prob() const;

  const NamedStorage1D<Math1D::Vector<ushort> >& best_alignments() const;

protected:

  void print_uncovered_set(uint state) const;

  uint nUncoveredPositions(uint state) const;

  void compute_uncovered_sets(uint nMaxSkips = 4);

  void cover(uint level);

  void visualize_set_graph(std::string filename);

  void compute_coverage_states();

  Math2D::NamedMatrix<ushort> uncovered_set_;
  
  //the first entry denotes the predecessor state, the second the source position covered in the transition
  NamedStorage1D<Math2D::Matrix<uint> > predecessor_sets_;

  //tells for each state how many uncovered positions are in that state
  Math1D::NamedVector<ushort> nUncoveredPositions_; 
  Math1D::NamedVector<ushort> j_before_end_skips_;

  //first_set_[i] marks the first row of <code> uncovered_sets_ </code> where the
  // position i appears
  Math1D::NamedVector<uint> first_set_;
  uint next_set_idx_; // used during the computation of uncovered sets

  //each row is a coverage state, where the first index denotes the number of uncovered set and the
  //second the maximum covered source position
  Math2D::NamedMatrix<uint> coverage_state_;
  Math1D::NamedVector<uint> first_state_;
  NamedStorage1D<Math2D::Matrix<uint> > predecessor_coverage_states_;
  

  const Storage1D<Storage1D<uint> >& source_sentence_;
  const Storage1D<Math2D::Matrix<uint> >& slookup_;
  const Storage1D<Storage1D<uint> >& target_sentence_;

  const CooccuringWordsType& wcooc_;
  SingleWordDictionary& dict_;

  uint nSourceWords_;
  uint nTargetWords_;

  uint maxJ_;
  uint maxI_;

  NamedStorage1D<Math1D::Vector<double> > fertility_prob_;

  NamedStorage1D<Math1D::Vector<ushort> > best_known_alignment_;

  std::map<uint,std::set<std::pair<ushort,ushort> > > sure_ref_alignments_;
  std::map<uint,std::set<std::pair<ushort,ushort> > > possible_ref_alignments_;
};

class IBM4Trainer;

class IBM3Trainer : public FertilityModelTrainer {
public:
  
  IBM3Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
              const Storage1D<Math2D::Matrix<uint> >& slookup,
              const Storage1D<Storage1D<uint> >& target_sentence,
              const std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_ref_alignments,
              SingleWordDictionary& dict,
              const CooccuringWordsType& wcooc,
              uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
              bool parametric_distortion = false,
              bool och_ney_empty_word = true,
              bool viterbi_ilp = false, double l0_fertpen = 0.0,
              bool smoothed_l0 = false, double l0_beta = 1.0);
  
  void init_from_hmm(const FullHMMAlignmentModel& align_model,
                     const InitialAlignmentProbability& initial_prob, HmmAlignProbType align_type);

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

  double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                    const Math2D::Matrix<uint>& lookup,
                                    Math1D::Vector<ushort>& alignment, bool ilp=false);

  void release_memory();
  
protected:
  
  friend class IBM4Trainer;

  void par2nonpar_distortion(ReducedIBM3DistortionModel& prob);

  double par_distortion_m_step_energy(const ReducedIBM3DistortionModel& fdistort_count,
                                      const Math2D::Matrix<double>& param);

  void par_distortion_m_step(const ReducedIBM3DistortionModel& fdistort_count);

  long double alignment_prob(uint s, const Math1D::Vector<ushort>& alignment) const;

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                             const Math2D::Matrix<uint>& lookup, const Math1D::Vector<ushort>& alignment) const;

  //improves the currently best known alignment using hill climbing and
  // returns the probability of the resulting alignment
  long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                               const Math2D::Matrix<uint>& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
                                               Math2D::Matrix<long double>& expansion_prob,
                                               Math2D::Matrix<long double>& swap_prob, Math1D::Vector<ushort>& alignment);

  long double compute_itg_viterbi_alignment_noemptyword(uint s, bool extended_reordering = false);

  //@param time_limit: maximum amount of seconds spent in the ILP-solver.
  //          values <= 0 indicate that no time limit is set
  long double compute_viterbi_alignment_ilp(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                            const Math2D::Matrix<uint>& lookup, uint max_fertility,
                                            Math1D::Vector<ushort>& alignment, double time_limit = -1.0);

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
};

enum IBM4CeptStartMode { IBM4CENTER, IBM4FIRST, IBM4LAST, IBM4UNIFORM };

class IBM4Trainer : public FertilityModelTrainer {
public: 

  IBM4Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
              const Storage1D<Math2D::Matrix<uint> >& slookup,
              const Storage1D<Storage1D<uint> >& target_sentence,
              const std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_ref_alignments,
              SingleWordDictionary& dict,
              const CooccuringWordsType& wcooc,
              uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
              bool och_ney_empty_word = false,
              bool use_sentence_start_prob = false,
              bool no_factorial = true,
              IBM4CeptStartMode cept_start_mode = IBM4CENTER,
              bool smoothed_l0 = false, double l0_beta = 1.0);


  void init_from_ibm3(IBM3Trainer& ibm3trainer, bool clear_ibm3 = true, 
		      bool count_collection = false, bool viterbi = false);

  //training without constraints on maximal fertility or uncovered positions.
  //This is based on the EM-algorithm where the E-step uses heuristics
  void train_unconstrained(uint nIter, IBM3Trainer* ibm3 = 0);

  //unconstrained Viterbi training
  void train_viterbi(uint nIter, IBM3Trainer* ibm3 = 0);

  void update_alignments_unconstrained();

  double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                    const Math2D::Matrix<uint>& lookup,
                                    Math1D::Vector<ushort>& alignment);

protected:

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                             const Math2D::Matrix<uint>& lookup, const Math1D::Vector<ushort>& alignment) const;

  void print_alignment_prob_factors(const Storage1D<uint>& source, const Storage1D<uint>& target, 
				    const Math2D::Matrix<uint>& lookup, const Math1D::Vector<ushort>& alignment) const;

  long double alignment_prob(uint s, const Math1D::Vector<ushort>& alignment) const;

  long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                               const Math2D::Matrix<uint>& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
                                               Math2D::Matrix<long double>& expansion_prob,
                                               Math2D::Matrix<long double>& swap_prob, Math1D::Vector<ushort>& alignment);

  void par2nonpar_inter_distortion();

  void par2nonpar_intra_distortion();

  void par2nonpar_start_prob();

  double inter_distortion_m_step_energy(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                                        const Math3D::Tensor<double>& inter_param, uint class1 = 0, uint class2 = 0);

  double intra_distortion_m_step_energy(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                                        const Math2D::Matrix<double>& intra_param, uint word_class = 0);

  void inter_distortion_m_step(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                               uint class1 = 0, uint class2 = 0);

  void intra_distortion_m_step(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                               uint word_class = 0);

  double start_prob_m_step_energy(const Storage1D<Math1D::Vector<double> >& start_count, Math1D::Vector<double>& param);

  void start_prob_m_step(const Storage1D<Math1D::Vector<double> >& start_count);


  //indexed by (target word class idx, source word class idx, displacement)
  IBM4CeptStartModel cept_start_prob_; //note: displacements of 0 are possible here (the center of a cept need not be an aligned word)
  //indexed by (source word class, displacement)
  IBM4WithinCeptModel within_cept_prob_; //note: displacements of 0 are impossible

  Math1D::NamedVector<double> sentence_start_parameters_;
  Storage1D<Math1D::Vector<double> > sentence_start_prob_;

  Storage1D<Storage2D<Math2D::Matrix<double> > > inter_distortion_prob_;
  Storage1D<Math3D::Tensor<double> > intra_distortion_prob_;

  uint displacement_offset_;

  double p_zero_;
  double p_nonzero_;

  bool och_ney_empty_word_;
  IBM4CeptStartMode cept_start_mode_;
  bool use_sentence_start_prob_;
  bool no_factorial_;
  const floatSingleWordDictionary& prior_weight_;
  bool smoothed_l0_;
  double l0_beta_;
};


#endif
