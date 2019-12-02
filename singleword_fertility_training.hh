/*** written by Thomas Schoenemann. Started as a private person without employment, November 2009 ***/
/*** continued at Lund University, Sweden, January 2010 - March 2011, as a private person, ***/
/*** at the University of Düsseldorf, Germany, January - April 2012 and since as a private person ***/

#ifndef SINGLEWORD_FERTILITY_TRAINING_HH
#define SINGLEWORD_FERTILITY_TRAINING_HH

#include "mttypes.hh"
#include "vector.hh"
#include "tensor.hh"

#include "hmm_training.hh"
#include "combinatoric.hh"

#include <map>
#include <set>

enum HillclimbingMode { HillclimbingReuse, HillclimbingRestart, HillclimbingReinit };

struct FertModelOptions {

  double p0_ = -1.0;

  FertNullModel empty_word_model_ = FertNullOchNey;
  bool smoothed_l0_ = false;
  double l0_beta_ = 0.0;
  double l0_fertpen_ = 0.0;
  double min_nondef_count_ = 1e-6;

  uint nMaxHCIter_ = 150;
  uint dict_m_step_iter_ = 45;
  uint fert_m_step_iter_ = 250;
  uint dist_m_step_iter_ = 400;
  uint nondef_dist34_m_step_iter_ = 250;

  HillclimbingMode hillclimb_mode_ = HillclimbingReuse;

  bool deficient_ = false; //turn IBM-5 with parametric distortion into a deficient model (no normalization)
  bool nondeficient_ = false; //turn IBM-3/4 into nondeficient models by local normalization

  MStepSolveMode msolve_mode_ = MSSolvePGD;

  IlpMode viterbi_ilp_mode_ = IlpOff;
  bool utmost_ilp_precision_ = false;

  IBM23ParametricMode par_mode_ = IBM23ParByPosition;

  IBM4CeptStartMode cept_start_mode_ = IBM4FIRST; //CENTER is the original of Brown et. al.
  IBM4InterDistMode inter_dist_mode_ = IBM4InterDistModePrevious; // previous is the original of Brown et. al.
  IBM4IntraDistMode intra_dist_mode_ = IBM4IntraDistModeSource; // source is the original of Brown et al.

  bool uniform_sentence_start_prob_ = false;
  bool reduce_deficiency_ = false;
  bool uniform_intra_prob_ = false;
  bool ibm5_nonpar_distortion_ = true;
};

/*abstract*/ class FertilityModelTrainerBase {
public:

  FertilityModelTrainerBase(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                            const Storage1D<Math1D::Vector<uint> >& target_sentence,
                            const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                            const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                            SingleWordDictionary& dict, const CooccuringWordsType& wcooc,
                            uint nSourceWords, uint nTargetWords, uint fertility_limit = 10000);

  virtual std::string model_name() const = 0;

  void write_alignments(const std::string filename) const;

  const NamedStorage1D<Math1D::Vector<AlignBaseType> >& best_alignments() const;

  void write_postdec_alignments(const std::string filename, double thresh);

  const Storage1D<Math1D::Vector<AlignBaseType> >& update_alignments_unconstrained(bool inform = true, const HmmWrapperWithClasses* wrapper = 0);

  virtual void set_fertility_limit(uint new_limit);

  virtual void set_rare_fertility_limit(uint new_limit, uint max_count = 2);

  //improves the passed alignment using hill climbing and
  // returns the probability of the resulting alignment
  //@param fertility: this vector must be of size I+1, where I is the length of target. It is then filled with the fertilities
  //      of the return <code> alignment </code>
  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      uint& nIter, Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double >& swap_prob,
      Math1D::Vector<AlignBaseType>& alignment) const = 0;

  virtual long double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
      const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment);

  // <code> start_alignment </code> is used as initialization for hillclimbing and later modified
  // the extracted alignment is written to <code> postdec_alignment </code>
  virtual void compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      Math1D::Vector<AlignBaseType>& start_alignment, std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
      double threshold = 0.25);

  void compute_approximate_jmarginals(const Math1D::Vector<AlignBaseType>& alignment, const Math2D::Matrix<long double>& expansion_prob,
                                      const Math2D::Matrix<long double>& swap_prob, const long double sentence_prob,
                                      Math2D::Matrix<double>& j_marg) const;

  //don't even need the swap matrix
  void compute_approximate_imarginals(const Math1D::Vector<AlignBaseType>& alignment, const Math1D::Vector<uint>& fertility,
                                      const Math2D::Matrix<long double>& expansion_prob, const long double sentence_prob,
                                      Math2D::Matrix<double>& i_marg) const;

  virtual void compute_approximate_jmarginals(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg, bool& converged) const
  {
    Math2D::Matrix<double> i_marg;
    compute_approximate_marginals(source, target, lookup, alignment, j_marg, i_marg, 1.0, converged);
  }

  //compute marginals needed for the IBM-3//returns the logarithm of the (approximated) normalization constant
  virtual double compute_approximate_marginals(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
      Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg,
      Math2D::Matrix<double >& i_marg, double hc_mass, bool& converged) const;

  virtual void release_memory();

  double AER();

  double AER(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments);

  double f_measure(double alpha = 0.1);

  double f_measure(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments, double alpha = 0.1);

  double DAE_S();

  double DAE_S(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments);

  void set_hmm_alignments(const HmmWrapper& hmm_wrapper);

  void set_hmm_alignments(const HmmWrapperWithClasses& hmmc_wrapper);

protected:

  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                          Math1D::Vector<AlignBaseType>& alignment) = 0;

  //converts the passed alignment so that it satisfies the constraints on the fertilities, including the one for the empty word.
  // returns if the alignment was changed
  bool make_alignment_feasible(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                               Math1D::Vector<AlignBaseType>& alignment) const;

  void compute_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                 Math1D::Vector<AlignBaseType>& alignment, double threshold,
                                 std::set<std::pair<AlignBaseType,AlignBaseType > >& postdec_alignment);

  const Storage1D<Math1D::Vector<uint> >& source_sentence_;
  const LookupTable& slookup_;
  const Storage1D<Math1D::Vector<uint> >& target_sentence_;

  const CooccuringWordsType& wcooc_;
  SingleWordDictionary& dict_;

  uint nSourceWords_;
  uint nTargetWords_;

  uint maxJ_;
  uint maxI_;

  Math1D::Vector<ushort> fertility_limit_;

  NamedStorage1D<Math1D::Vector<AlignBaseType> > best_known_alignment_;

  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType > > > sure_ref_alignments_;
  std::map<uint,std::set<std::pair < AlignBaseType,AlignBaseType > > > possible_ref_alignments_;
};

/*abstract*/ class FertilityModelTrainer : public FertilityModelTrainerBase {
public:

  FertilityModelTrainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                        const Storage1D<Math1D::Vector<uint> >& target_sentence, SingleWordDictionary& dict,
                        const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class,
                        uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
                        FertNullModel empty_word_model, bool smoothed_l0, double l0_beta, double l0_fertpen, bool no_factorial,
                        const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                        const std::map<uint,std::set<std::pair< AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                        const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table, uint fertility_limit = 10000,
                        MStepSolveMode msolve_mode = MSSolvePGD, HillclimbingMode hillclimb_mode = HillclimbingReuse);

  FertilityModelTrainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                        const Storage1D<Math1D::Vector<uint> >& target_sentence, SingleWordDictionary& dict,
                        const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class,
                        uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
                        const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                        const std::map<uint,std::set<std::pair< AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                        const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
                        const FertModelOptions& options, bool no_factorial, uint fertility_limit = 10000);

  double p_zero() const;

  void fix_p0(double p0);

  void set_hc_iteration_limit(uint new_limit);

  virtual void set_fertility_limit(uint new_limit);

  virtual void set_rare_fertility_limit(uint new_limit, uint max_count = 2);

  void PostdecEval(double& aer, double& f_measure, double& daes, double threshold = 0.25, double alpha = 0.1);

  const NamedStorage1D<Math1D::Vector<double> >& fertility_prob() const;

  void write_fertilities(std::string filename);

  virtual void release_memory();

protected:

  long double alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const;

  virtual long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
                                     const Math1D::Vector<AlignBaseType>& alignment) const = 0;

  double regularity_term() const;

  double exact_l0_reg_term(const Storage1D<Math1D::Vector<double> >& fwcount, const Storage1D<Math1D::Vector<double> >& ffert_count) const;

  void update_fertility_prob(const Storage1D<Math1D::Vector<double> >& ffert_count, double min_prob = 1e-8, bool with_regularity = true);

  inline void update_fertility_counts(const Storage1D<uint>& target, const Math1D::Vector<AlignBaseType>& best_alignment,
                                      const Math1D::NamedVector<uint>& fertility,
                                      const Math2D::NamedMatrix<long double>& expansion_move_prob,
                                      const long double sentence_prob, const long double inv_sentence_prob,
                                      Storage1D<Math1D::Vector<double> >& ffert_count);

  inline void update_dict_counts(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
                                 const SingleLookupTable& cur_lookup, const Math1D::Vector<AlignBaseType>& best_alignment,
                                 const Math2D::NamedMatrix<long double>& expansion_move_prob,
                                 const Math2D::NamedMatrix<long double>& swap_move_prob,
                                 const long double sentence_prob, const long double inv_sentence_prob,
                                 Storage1D<Math1D::Vector<double> >& fwcount);

  inline void update_zero_counts(const Math1D::Vector<AlignBaseType>& best_alignment, const Math1D::NamedVector<uint>& fertility,
                                 const Math2D::NamedMatrix<long double>& expansion_move_prob, const long double swap_sum,
                                 const long double best_prob, const long double sentence_prob, const long double inv_sentence_prob,
                                 double& fzero_count, double& fnonzero_count);

  inline long double swap_mass(const Math2D::Matrix<long double>& swap_move_prob) const;

  inline double common_icm_change(const Math1D::Vector<uint>& cur_fertilities, const double log_pzero, const double log_pnonzero,
                                  const Math1D::NamedVector<uint>& dict_sum, const Math1D::Vector<double>& cur_dictcount,
                                  const Math1D::Vector<double>& hyp_dictcount, const Math1D::Vector<float>& cur_prior,
                                  const Math1D::Vector<float>& hyp_prior, const Math1D::Vector<double>& cur_fert_count,
                                  const Math1D::Vector<double>& hyp_fert_count, const Storage1D<Math1D::Vector<double> >& ffertclass_count,
                                  const uint cur_target_word, const uint hyp_target_word, const uint cur_idx, const uint hyp_idx,
                                  const uint cur_aj, const uint hyp_aj, const uint curJ) const;

  inline void common_icm_count_change(Math1D::NamedVector<uint>& dict_sum, Math1D::Vector<double>& cur_dictcount,
                                      Math1D::Vector<double>& hyp_dictcount, Math1D::Vector<double>& cur_fert_count,
                                      Math1D::Vector<double>& hyp_fert_count, Storage1D<Math1D::Vector<double> >& ffertclass_count,
                                      const uint cur_word, const uint new_target_word, const uint cur_idx, const uint hyp_idx,
                                      const uint cur_aj, const uint hyp_aj, Math1D::Vector<uint>& cur_fertilities);

  void common_prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                         const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment);

  void init_fertilities(FertilityModelTrainerBase* prev_model, double count_weight = 0.95);

  inline void update_nullpow(uint fert0, uint fert1) const;

  inline void update_nullpow(uint fert0, uint fert1, const double p_zero, const double p_nonzero) const;

  inline void compute_null_prob(uint curJ, double p_zero, double p_nonzero,
                                Math1D::Vector<long double>& null_prob, Math1D::Vector<double>& null_theta) const;

  inline void compute_null_prob(uint curJ, double p_zero, double p_nonzero, Math1D::Vector<long double>& null_prob) const;

  inline void compute_null_theta(uint curJ, double p_zero, double p_nonzero, Math1D::Vector<double>& null_theta) const;

  inline void compute_null_theta(uint curJ, double p_zero, double p_nonzero, Math1D::Vector<float>& null_theta) const;

  double p_zero_;
  double p_nonzero_;

  bool fix_p0_;

  uint iter_offs_;

  FertNullModel empty_word_model_;
  bool smoothed_l0_;
  double l0_beta_;
  double l0_fertpen_;

  bool no_factorial_;

  uint nMaxHCIter_;
  uint dict_m_step_iter_ = 45;
  uint fert_m_step_iter_ = 250;

  MStepSolveMode msolve_mode_;
  HillclimbingMode hillclimb_mode_;

  const floatSingleWordDictionary& prior_weight_;
  bool prior_weight_active_ = false;

  NamedStorage1D<Math1D::Vector<double> > fertility_prob_;

  const Math1D::Vector<uint>& tfert_class_;
  Math1D::Vector<uint> tfert_class_count_;
  uint nTFertClasses_;
  bool fertprob_sharing_;

  static Math1D::NamedVector<long double> ld_fac_; //precomputation of factorials
  static Storage1D<Math1D::Vector<long double> > choose_factor_;
  static Storage1D<Math1D::Vector<long double> > och_ney_factor_;

  const Math1D::Vector<double>& log_table_;
  const Math1D::Vector<double>& xlogx_table_;

  mutable Math1D::Vector<long double> p_zero_pow_;
  mutable Math1D::Vector<long double> p_nonzero_pow_;
};

/*************** definition of inline functions ************/

inline void FertilityModelTrainer::update_fertility_counts(const Storage1D<uint>& target, const Math1D::Vector<AlignBaseType>& best_alignment,
    const Math1D::NamedVector<uint>& fertility, const Math2D::NamedMatrix<long double>& expansion_move_prob,
    const long double sentence_prob, const long double inv_sentence_prob, Storage1D<Math1D::Vector<double> >& ffert_count)
{
  uint curJ = best_alignment.size();
  uint curI = target.size();

  assert(fertility.size() == curI + 1);

  //this passage exploits that entries of expansion_move_prob are 0.0 if they refer to an unchanged alignment

  for (uint i = 1; i <= curI; i++) {

    const uint t_idx = target[i - 1];

    Math1D::Vector<double>& cur_fert_count = ffert_count[t_idx];

    const uint cur_fert = fertility[i];

    long double addon = sentence_prob;
    for (uint j = 0; j < curJ; j++) {
      if (best_alignment[j] == i) {
        for (uint ii = 0; ii <= curI; ii++)
          addon -= expansion_move_prob(j, ii);
      }
      else
        addon -= expansion_move_prob(j, i);
    }
    addon *= inv_sentence_prob;

    double daddon = (double)addon;
    if (!(daddon > 0.0)) {
      std::
      cerr << "STRANGE: fractional weight " << daddon <<
           " for sentence pair with " << curJ << " source words and " << curI <<
           " target words" << std::endl;
      std::cerr << "sentence prob: " << sentence_prob << std::endl;
      std::cerr << "" << std::endl;
    }

    cur_fert_count[cur_fert] += addon;

    //NOTE: swap moves do not change the fertilities
    if (cur_fert > 0) {
      long double alt_addon = 0.0;
      for (uint j = 0; j < curJ; j++) {
        if (best_alignment[j] == i) {
          for (uint ii = 0; ii <= curI; ii++) {
            alt_addon += expansion_move_prob(j, ii);
          }
        }
      }

      cur_fert_count[cur_fert - 1] += inv_sentence_prob * alt_addon;
    }
    //check for the unlikely event that all source words in best_alignment align to i
    if (cur_fert + 1 < fertility_prob_[t_idx].size()) {

      long double alt_addon = 0.0;
      for (uint j = 0; j < curJ; j++) {
        alt_addon += expansion_move_prob(j, i);
      }

      cur_fert_count[cur_fert + 1] += inv_sentence_prob * alt_addon;
    }
  }
}

inline void FertilityModelTrainer::update_dict_counts(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
    const SingleLookupTable& cur_lookup, const Math1D::Vector<AlignBaseType>& best_alignment,
    const Math2D::NamedMatrix<long double>& expansion_move_prob, const Math2D::NamedMatrix<long double>& swap_move_prob,
    const long double sentence_prob, const long double inv_sentence_prob, Storage1D<Math1D::Vector<double> >& fwcount)
{
  const uint curJ = cur_source.size();
  const uint curI = cur_target.size();

  for (uint j = 0; j < curJ; j++) {

    const uint s_idx = cur_source[j];
    const uint cur_aj = best_alignment[j];

    //this passage exploits that entries of expansion_move_prob and swap_move_prob are 0.0 if they refer to an unchanged alignment

    long double addon = sentence_prob;
    for (uint i = 0; i <= curI; i++)
      addon -= expansion_move_prob(j, i);
    for (uint jj = 0; jj < curJ; jj++)
      addon -= swap_move_prob(j, jj);   //exploits that swap_move_prob is a symmetric matrix

    addon *= inv_sentence_prob;
    if (cur_aj != 0) {
      fwcount[cur_target[cur_aj - 1]][cur_lookup(j, cur_aj - 1)] += addon;
    }
    else {
      fwcount[0][s_idx - 1] += addon;
    }

    for (uint i = 0; i <= curI; i++) {

      if (i != cur_aj) {

        const uint t_idx = (i == 0) ? 0 : cur_target[i - 1];

        long double addon = expansion_move_prob(j, i);
        for (uint jj = 0; jj < curJ; jj++) {
          if (best_alignment[jj] == i)
            addon += swap_move_prob(j, jj);
        }
        addon *= inv_sentence_prob;

        if (i != 0) {
          fwcount[t_idx][cur_lookup(j, i - 1)] += addon;
        }
        else {
          fwcount[0][s_idx - 1] += addon;
        }
      }
    }
  }
}

inline void FertilityModelTrainer::update_zero_counts(const Math1D::Vector<AlignBaseType>& best_alignment,
    const Math1D::NamedVector<uint>& fertility, const Math2D::NamedMatrix<long double>& expansion_move_prob,
    const long double swap_sum, const long double best_prob, const long double sentence_prob, const long double inv_sentence_prob,
    double& fzero_count, double& fnonzero_count)
{
  const uint curJ = expansion_move_prob.xDim();
  const uint curI = expansion_move_prob.yDim() - 1;

  assert(curJ == best_alignment.size());
  assert(fertility.size() == curI + 1);

  double cur_zero_weight = best_prob;
  cur_zero_weight += swap_sum;
  for (uint j = 0; j < curJ; j++) {
    if (best_alignment[j] != 0) {
      for (uint i = 1; i <= curI; i++)
        cur_zero_weight += expansion_move_prob(j, i);
    }
  }

  cur_zero_weight *= inv_sentence_prob;

  assert(!isnan(cur_zero_weight));
  assert(!isinf(cur_zero_weight));

  fzero_count += cur_zero_weight * (fertility[0]);
  fnonzero_count += cur_zero_weight * (curJ - 2 * fertility[0]);

  if (curJ >= 2 * (fertility[0] + 1)) {
    long double inc_zero_weight = 0.0;
    for (uint j = 0; j < curJ; j++)
      inc_zero_weight += expansion_move_prob(j, 0);

    inc_zero_weight *= inv_sentence_prob;
    fzero_count += inc_zero_weight * (fertility[0] + 1);
    fnonzero_count += inc_zero_weight * (curJ - 2 * (fertility[0] + 1));

    assert(!isnan(inc_zero_weight));
    assert(!isinf(inc_zero_weight));
  }

  if (fertility[0] > 1) {
    long double dec_zero_weight = 0.0;
    for (uint j = 0; j < curJ; j++) {
      if (best_alignment[j] == 0) {
        for (uint i = 1; i <= curI; i++)
          dec_zero_weight += expansion_move_prob(j, i);
      }
    }

    dec_zero_weight *= inv_sentence_prob;

    fzero_count += dec_zero_weight * (fertility[0] - 1);
    fnonzero_count += dec_zero_weight * (curJ - 2 * (fertility[0] - 1));

    assert(!isnan(dec_zero_weight));
    assert(!isinf(dec_zero_weight));
  }
  //DEBUG
  if (isnan(fzero_count) || isnan(fnonzero_count)
      || isinf(fzero_count) || isinf(fnonzero_count)) {

    std::
    cerr << "zero counts: " << fzero_count << ", " << fnonzero_count <<
         std::endl;
    std::cerr << "J=" << curJ << ", I=" << curI << std::endl;
    std::cerr << "sentence weight: " << sentence_prob << std::endl;
    exit(1);
  }
  //END_DEBUG
}

inline long double FertilityModelTrainer::swap_mass(const Math2D::Matrix<long double>& swap_move_prob) const
{
  //return 0.5 * swap_move_prob.sum();

  const uint J = swap_move_prob.xDim();
  assert(J == swap_move_prob.yDim());

  double sum = 0.0;
  for (uint j1 = 0; j1 < J - 1; j1++)
    for (uint j2 = j1 + 1; j2 < J; j2++)
      sum += swap_move_prob(j1, j2);

  return sum;
}

inline double FertilityModelTrainer::common_icm_change(const Math1D::Vector<uint>& cur_fertilities, const double log_pzero, const double log_pnonzero,
    const Math1D::NamedVector<uint>& dict_sum, const Math1D::Vector<double>& cur_dictcount, const Math1D::Vector<double>& hyp_dictcount,
    const Math1D::Vector<float>& cur_prior, const Math1D::Vector<float>& hyp_prior, const Math1D::Vector<double>& cur_fert_count,
    const Math1D::Vector<double>& hyp_fert_count, const Storage1D<Math1D::Vector<double> >& ffertclass_count,
    const uint cur_word, const uint new_target_word, const uint cur_idx, const uint hyp_idx,
    const uint cur_aj, const uint hyp_aj, const uint curJ) const
{
  double change = 0.0;

  assert(log_table_[0] == 0.0);

  const uint cur_fert = cur_fertilities[cur_aj];
  const uint cur_hyp_fert = cur_fertilities[hyp_aj];

  if (!no_factorial_) {
    if (cur_word != 0) {
      //transition from -log(ld_fac[cur_fert]) to -log(ld_fac[cur_fert-1])
      change += log_table_[cur_fert];
    }
    if (new_target_word != 0) {
      //transition from -log(ld_fac[cur_hyp_fert]) to -log(ld_fac[cur_hyp_fert+1])
      change -= log_table_[cur_hyp_fert + 1];
    }
  }

  if (cur_word != new_target_word) {

    uint cur_dictsum = dict_sum[cur_word];

    if (dict_sum[new_target_word] > 0) {
      //exploit log(1) = 0
      change -= xlogx_table_[dict_sum[new_target_word]];
      change += xlogx_table_[dict_sum[new_target_word] + 1];
    }

    //prior_weight is always relevant
    if (hyp_dictcount[hyp_idx] > 0) {
      //exploit log(1) = 0
      change += xlogx_table_[hyp_dictcount[hyp_idx]];
      change -= xlogx_table_[hyp_dictcount[hyp_idx] + 1];
    }
    else
      change += hyp_prior[hyp_idx];

    if (cur_dictsum > 1) {
      //exploit log(1) = 0
      change -= xlogx_table_[cur_dictsum];
      change += xlogx_table_[cur_dictsum - 1];
    }

    //prior_weight is always relevant
    if (cur_dictcount[cur_idx] > 1) {
      //exploit log(1) = 0
      change -= -xlogx_table_[cur_dictcount[cur_idx]];
      change += -xlogx_table_[cur_dictcount[cur_idx] - 1];
    }
    else
      change -= cur_prior[cur_idx];

    /***** fertilities for the (easy) case where the old and the new word differ ****/

    //note: currently not updating f_zero / f_nonzero
    if (cur_aj == 0) {

      const uint zero_fert = cur_fert;
      const uint new_zero_fert = zero_fert - 1;

      //changes regarding ldchoose()
      change -= log_table_[curJ - new_zero_fert];
      change += log_table_[zero_fert];  // - - = +
      change += log_table_[curJ - 2 * zero_fert + 1] + log_table_[curJ - 2 * new_zero_fert];

      change += log_pzero;      // - -  = +

      if (empty_word_model_ == FertNullOchNey) {
        change -= log_table_[curJ] - log_table_[zero_fert];
      }
      //FertNullIntra has to be handled by the model itself

      change -= 2.0 * log_pnonzero;
    }
    else {

      if (!fertprob_sharing_) {

        const int c = cur_fert_count[cur_fert];
        if (c > 1) {
          //exploit log(1) = 0
          change -= -xlogx_table_[c];
          change += -xlogx_table_[c - 1];
        }

        const int c2 = cur_fert_count[cur_fert - 1];

        if (c2 > 0) {
          //exploit log(1) = 0
          change -= -xlogx_table_[c2];
          change += -xlogx_table_[c2 + 1];
        }
      }
    }

    if (hyp_aj == 0) {

      const uint zero_fert = cur_hyp_fert;
      const uint new_zero_fert = zero_fert + 1;

      //changes regarding ldchoose()
      change += log_table_[curJ - zero_fert];   // - -  = +
      change -= log_table_[curJ - 2 * zero_fert] + log_table_[curJ - 2 * zero_fert - 1];
      change += log_table_[new_zero_fert];

      change += 2.0 * log_pnonzero;

      change -= log_pzero;

      if (empty_word_model_ == FertNullOchNey) {
        change += log_table_[curJ] - log_table_[new_zero_fert];
      }
      //FertNullIntra has to be handled by the model itself
    }
    else {

      if (!fertprob_sharing_) {

        const int c = hyp_fert_count[cur_hyp_fert];
        if (c > 1) {
          //exploit log(1) = 0
          change -= -xlogx_table_[c];
          change += -xlogx_table_[c - 1];
        }
        else
          change -= l0_fertpen_;        //remove penalty when the count becomes 0

        const int c2 = hyp_fert_count[cur_hyp_fert + 1];
        if (c2 > 0) {
          //exploit log(1) = 0
          change -= -xlogx_table_[c2];
          change += -xlogx_table_[c2 + 1];
        }
        else
          change += l0_fertpen_;        //introduce penalty when the count becomes nonzero
      }
    }
  }
  else {
    //the old and the new word are the same.
    //No dictionary terms affected, but the fertilities are tricky in this case

    assert(cur_aj != 0);
    assert(hyp_aj != 0);

    const Math1D::Vector<double>& cur_count = cur_fert_count;
    Math1D::Vector<double> new_count = cur_count;
    new_count[cur_fert]--;
    new_count[cur_fert - 1]++;
    new_count[cur_hyp_fert]--;
    new_count[cur_hyp_fert + 1]++;

    for (uint k = 0; k < cur_count.size(); k++) {
      if (cur_count[k] != new_count[k]) {

        if (!fertprob_sharing_) {
          change += xlogx_table_[cur_count[k]]; // - - = +
          change -= xlogx_table_[new_count[k]];

          if (cur_count[k] == 0)
            change += l0_fertpen_;      //count becomes nonzero
          if (new_count[k] == 0)
            change -= l0_fertpen_;
        }
      }
    }
  }

  if (fertprob_sharing_) {

    uint cur_class = (cur_aj != 0) ? tfert_class_[cur_word] : MAX_UINT;
    uint new_class = (hyp_aj != 0) ? tfert_class_[new_target_word] : MAX_UINT;

    if (cur_class != new_class) {

      //easy case
      if (cur_class != MAX_UINT) {

        const int c = ffertclass_count[cur_class][cur_fert];
        assert(c > 0);
        if (c > 1) {
          //exploit log(1) = 0
          change -= -c * log_table_[c];
          change += -(c - 1) * log_table_[c - 1];
        }
        else
          change -= l0_fertpen_ * tfert_class_count_[cur_class];        //count becomes 0

        const int c2 = ffertclass_count[cur_class][cur_fert - 1];
        if (c2 > 0) {
          //exploit log(1) = 0
          change -= -c2 * log_table_[c2];
          change += -(c2 + 1) * log_table_[c2 + 1];
        }
        else
          change += l0_fertpen_ * tfert_class_count_[cur_class];        //count becomes non-zero
      }

      if (new_class != MAX_UINT) {

        const int c = ffertclass_count[new_class][cur_hyp_fert];
        assert(c > 0);
        if (c > 1) {
          //exploit log(1) = 0
          change -= -c * log_table_[c];
          change += -(c - 1) * log_table_[c - 1];
        }
        else
          change -= l0_fertpen_ * tfert_class_count_[new_class];        //count becomes 0

        const int c2 = ffertclass_count[new_class][cur_hyp_fert + 1];
        if (c2 > 0) {
          //exploit log(1) = 0
          change -= -c2 * log_table_[c2];
          change += -(c2 + 1) * log_table_[c2 + 1];
        }
        else
          change += l0_fertpen_ * tfert_class_count_[new_class];        //count becomes non-zero
      }
    }
    else {

      //the fertilities are tricky in this case

      const Math1D::Vector<double>& cur_count = ffertclass_count[cur_class];
      Math1D::Vector<double> new_count = cur_count;
      new_count[cur_fert]--;
      new_count[cur_fert - 1]++;
      new_count[cur_hyp_fert]--;
      new_count[cur_hyp_fert + 1]++;

      for (uint k = 0; k < cur_count.size(); k++) {
        if (cur_count[k] != new_count[k]) {

          change += cur_count[k] * log_table_[cur_count[k]];    // - - = +
          change -= new_count[k] * log_table_[new_count[k]];

          if (cur_count[k] == 0)
            change += l0_fertpen_ * tfert_class_count_[cur_class];      //count becomes nonzero
          if (new_count[k] == 0)
            change -= l0_fertpen_ * tfert_class_count_[cur_class];      //count becomes 0
        }
      }
    }
  }

  return change;
}

inline void FertilityModelTrainer::common_icm_count_change(Math1D::NamedVector<uint>& dict_sum, Math1D::Vector<double>& cur_dictcount,
    Math1D::Vector<double>& hyp_dictcount, Math1D::Vector<double>& cur_fert_count, Math1D::Vector<double>& hyp_fert_count,
    Storage1D<Math1D::Vector<double> >& ffertclass_count, const uint cur_word, const uint new_target_word,
    const uint cur_idx, const uint hyp_idx, const uint cur_aj, const uint hyp_aj, Math1D::Vector<uint>& cur_fertilities)
{
  //std::cerr << "A" << std::endl;

  //dict
  cur_dictcount[cur_idx] -= 1;
  hyp_dictcount[hyp_idx] += 1;
  dict_sum[cur_word] -= 1;
  dict_sum[new_target_word] += 1;

  //std::cerr << "B" << std::endl;

  //fert
  if (cur_word != 0) {
    const uint prev_fert = cur_fertilities[cur_aj];
    assert(prev_fert != 0);
    cur_fert_count[prev_fert] -= 1;
    cur_fert_count[prev_fert - 1] += 1;

    if (fertprob_sharing_) {
      const uint c = tfert_class_[cur_word];
      ffertclass_count[c][prev_fert] -= 1;
      ffertclass_count[c][prev_fert - 1] += 1;
    }
  }
  if (new_target_word != 0) {
    const uint prev_fert = cur_fertilities[hyp_aj];
    hyp_fert_count[prev_fert] -= 1;
    hyp_fert_count[prev_fert + 1] += 1;

    if (fertprob_sharing_) {
      const uint c = tfert_class_[new_target_word];
      ffertclass_count[c][prev_fert] -= 1;
      ffertclass_count[c][prev_fert + 1] += 1;
    }
  }
  //std::cerr << "C" << std::endl;

  cur_fertilities[cur_aj]--;
  cur_fertilities[hyp_aj]++;
}

inline void FertilityModelTrainer::update_nullpow(uint fert0, uint fert1) const
{
  update_nullpow(fert0, fert1, p_zero_, p_nonzero_);
}

inline void FertilityModelTrainer::update_nullpow(uint fert0, uint fert1, const double p_zero, const double p_nonzero) const
{
  bool update = false;
  if (fert0 >= p_zero_pow_.size()) {
    p_zero_pow_.resize(fert0 + 1);
    update = true;
  }
  if (fert1 >= p_nonzero_pow_.size()) {
    p_nonzero_pow_.resize(fert1 + 1);
    update = true;
  }

  if (p_zero_pow_[1] != p_zero)
    update = true;
  if (p_nonzero_pow_[1] != p_nonzero)
    update = true;

  if (update) {
    //std::cerr << "------ update of p0 pow table" << std::endl;

    p_zero_pow_[0] = 1.0;
    p_zero_pow_[1] = p_zero;
    for (uint c = 2; c < p_zero_pow_.size(); c++)
      p_zero_pow_[c] = p_zero_pow_[c - 1] * p_zero;

    p_nonzero_pow_[0] = 1.0;
    p_nonzero_pow_[1] = p_nonzero;
    for (uint c = 2; c < p_nonzero_pow_.size(); c++)
      p_nonzero_pow_[c] = p_nonzero_pow_[c - 1] * p_nonzero;
  }
}

inline void FertilityModelTrainer::compute_null_prob(uint curJ, double p_zero, double p_nonzero,
    Math1D::Vector<long double>& null_prob, Math1D::Vector<double>& null_theta) const
{
  null_prob.resize_dirty(curJ + 1);
  null_theta.resize_dirty(curJ + 1);
  null_prob.set_constant(0.0);
  null_theta.set_constant(-1e7);

  long double running_product = 1.0;

  update_nullpow(0, curJ, p_zero, p_nonzero);

  const Math1D::Vector<long double>& choose_factor = choose_factor_[curJ];

  for (uint c = 0; 2 * c <= curJ; c++) {

    long double base_prob = choose_factor[c];

    if (c > 0 && empty_word_model_ == FertNullOchNey)
      base_prob *= och_ney_factor_[curJ][c];
    else if (c > 0 && empty_word_model_ == FertNullIntra)
      base_prob /= (double)curJ;        //the rest of the model depends on the NULL-aligned positions
    base_prob *= running_product;
    running_product *= p_zero;

    base_prob *= p_nonzero_pow_[curJ - 2 * c];

    null_prob[c] = std::max<long double>(1e-300, base_prob);
    null_theta[c] = logl(null_prob[c]);
  }
}

inline void FertilityModelTrainer::compute_null_prob(uint curJ, double p_zero, double p_nonzero, Math1D::Vector<long double>& null_prob) const
{
  null_prob.resize_dirty(curJ + 1);
  null_prob.set_constant(0.0);

  const Math1D::Vector<long double>& choose_factor = choose_factor_[curJ];

  long double running_product = 1.0;

  update_nullpow(0, curJ, p_zero, p_nonzero);

  for (uint c = 0; 2 * c <= curJ; c++) {

    long double base_prob = choose_factor[c];

    if (c > 0 && empty_word_model_ == FertNullOchNey)
      base_prob *= och_ney_factor_[curJ][c];
    else if (c > 0 && empty_word_model_ == FertNullIntra)
      base_prob /= (double)curJ;        //the rest of the model depends on the NULL-aligned positions
    base_prob *= running_product;
    running_product *= p_zero;

    base_prob *= p_nonzero_pow_[curJ - 2 * c];

    null_prob[c] = std::max<long double>(1e-300, base_prob);
  }
}

inline void FertilityModelTrainer::compute_null_theta(uint curJ, double p_zero, double p_nonzero,
    Math1D::Vector<double>& null_theta) const
{
  null_theta.resize_dirty(curJ + 1);
  null_theta.set_constant(-1e7);

  const Math1D::Vector<long double>& choose_factor = choose_factor_[curJ];

  long double running_product = 1.0;

  update_nullpow(0, curJ, p_zero, p_nonzero);

  for (uint c = 0; 2 * c <= curJ; c++) {

    long double base_prob = choose_factor[c];

    if (c > 0 && empty_word_model_ == FertNullOchNey)
      base_prob *= och_ney_factor_[curJ][c];
    else if (c > 0 && empty_word_model_ == FertNullIntra)
      base_prob /= (double)curJ;        //the rest of the model depends on the NULL-aligned positions
    base_prob *= running_product;
    running_product *= p_zero;

    base_prob *= p_nonzero_pow_[curJ - 2 * c];

    null_theta[c] = logl(std::max<long double>(1e-300, base_prob));
  }
}

inline void FertilityModelTrainer::compute_null_theta(uint curJ, double p_zero, double p_nonzero, Math1D::Vector<float>& null_theta) const
{
  null_theta.resize_dirty(curJ + 1);
  null_theta.set_constant(-1e7);

  const Math1D::Vector<long double>& choose_factor = choose_factor_[curJ];

  long double running_product = 1.0;

  update_nullpow(0, curJ, p_zero, p_nonzero);

  for (uint c = 0; 2 * c <= curJ; c++) {

    long double base_prob = choose_factor[c];

    if (c > 0 && empty_word_model_ == FertNullOchNey)
      base_prob *= och_ney_factor_[curJ][c];
    else if (c > 0 && empty_word_model_ == FertNullIntra)
      base_prob /= (double)curJ;        //the rest of the model depends on the NULL-aligned positions
    base_prob *= running_product;
    running_product *= p_zero;

    base_prob *= p_nonzero_pow_[curJ - 2 * c];

    null_theta[c] = logl(std::max<long double>(1e-300, base_prob));
  }
}

#endif
