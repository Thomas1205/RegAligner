/*** written by Thomas Schoenemann. Started as a private person without employment, November 2009 ***/
/*** continued at Lund University, Sweden, January 2010 - March 2011, as a private person, ***/
/*** at the University of Düsseldorf, Germany, January - September 2012 and since as a private person ***/

#include "singleword_fertility_training.hh"
#include "combinatoric.hh"
#include "alignment_error_rate.hh"
#include "timing.hh"
#include "alignment_computation.hh"
#include "training_common.hh"   // for get_wordlookup()

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"
#include "stl_util.hh"

/********************** implementation of FertilityModelTrainer *******************************/

FertilityModelTrainerBase::FertilityModelTrainerBase(const Storage1D<Math1D::Vector<uint> >& source_sentence,
    const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target_sentence,
    const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
    const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
    SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords, uint nTargetWords,
    uint fertility_limit)
  :source_sentence_(source_sentence), slookup_(slookup), target_sentence_(target_sentence),
   wcooc_(wcooc), dict_(dict), nSourceWords_(nSourceWords),
   nTargetWords_(nTargetWords),
   best_known_alignment_(MAKENAME(best_known_alignment_)),
   sure_ref_alignments_(sure_ref_alignments), possible_ref_alignments_(possible_ref_alignments)
{
  maxJ_ = 0;
  maxI_ = 0;

  for (size_t s = 0; s < source_sentence.size(); s++) {

    const uint curJ = source_sentence[s].size();
    const uint curI = target_sentence[s].size();

    maxJ_ = std::max(maxJ_, curJ);
    maxI_ = std::max(maxI_, curI);
  }

  fertility_limit_.resize(nTargetWords, fertility_limit);

  best_known_alignment_.resize(source_sentence.size());
  for (size_t s = 0; s < source_sentence.size(); s++)
    best_known_alignment_[s].resize(source_sentence[s].size(), 0);
}

const NamedStorage1D<Math1D::Vector<AlignBaseType> >& FertilityModelTrainerBase::best_alignments() const
{
  return best_known_alignment_;
}

/*virtual*/ void FertilityModelTrainerBase::set_fertility_limit(uint new_limit)
{
  fertility_limit_.set_constant(new_limit);

  for (uint i = 1; i < nTargetWords_; i++)
    fertility_limit_[i] = std::min < uint > (new_limit, fertility_limit_[i]);
}

/*virtual*/ void FertilityModelTrainerBase::set_rare_fertility_limit(uint new_limit, uint max_count)
{
  Math1D::Vector<uint> count(nTargetWords_, 0);
  for (uint s = 0; s < target_sentence_.size(); s++) {

    const Storage1D<uint>& cur_target = target_sentence_[s];
    for (uint i = 0; i < cur_target.size(); i++)
      count[cur_target[i]]++;
  }

  for (uint i = 1; i < nTargetWords_; i++) {
    if (count[i] <= max_count) {
      fertility_limit_[i] = std::min < uint > (new_limit, fertility_limit_[i]);
    }
  }
}

void FertilityModelTrainerBase::write_alignments(const std::string filename) const
{
  std::ostream* out;

#ifdef HAS_GZSTREAM
  if (string_ends_with(filename, ".gz")) {
    out = new ogzstream(filename.c_str());
  }
  else {
    out = new std::ofstream(filename.c_str());
  }
#else
  out = new std::ofstream(filename.c_str());
#endif

  for (size_t s = 0; s < source_sentence_.size(); s++) {

    const uint curJ = source_sentence_[s].size();

    for (uint j = 0; j < curJ; j++) {
      if (best_known_alignment_[s][j] > 0)
        (*out) << (best_known_alignment_[s][j] - 1) << " " << j << " ";
    }

    (*out) << std::endl;
  }

  delete out;
}

void FertilityModelTrainerBase::write_postdec_alignments(const std::string filename, double thresh)
{
  std::ostream* out;

#ifdef HAS_GZSTREAM
  if (string_ends_with(filename, ".gz")) {
    out = new ogzstream(filename.c_str());
  }
  else {
    out = new std::ofstream(filename.c_str());
  }
#else
  out = new std::ofstream(filename.c_str());
#endif

  for (uint s = 0; s < source_sentence_.size(); s++) {

    Math1D::Vector<AlignBaseType> viterbi_alignment =  best_known_alignment_[s];
    std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;

    SingleLookupTable aux_lookup;

    const SingleLookupTable& cur_lookup =
      get_wordlookup(source_sentence_[s], target_sentence_[s], wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    compute_external_postdec_alignment(source_sentence_[s], target_sentence_[s],
                                       cur_lookup, viterbi_alignment, postdec_alignment, thresh);

    for (std::set<std::pair<AlignBaseType,AlignBaseType> >::iterator it = postdec_alignment.begin();
         it != postdec_alignment.end(); it++) {

      (*out) << (it->second - 1) << " " << (it->first - 1) << " ";
    }
    (*out) << std::endl;

  }

  delete out;
}

const Storage1D<Math1D::Vector<AlignBaseType> >& FertilityModelTrainerBase::update_alignments_unconstrained(bool inform,
    const HmmWrapperWithClasses* wrapper)
{
  if (wrapper != 0)
    set_hmm_alignments(*wrapper);

  Math2D::NamedMatrix<long double> expansion_prob(MAKENAME(expansion_prob));
  Math2D::NamedMatrix<long double> swap_prob(MAKENAME(swap_prob));

  for (size_t s = 0; s < source_sentence_.size(); s++) {

    //std::cerr << "s: " << s << std::endl;

    const uint curI = target_sentence_[s].size();
    Math1D::NamedVector < uint > fertility(curI + 1, 0, MAKENAME(fertility));

    SingleLookupTable aux_lookup;
    const SingleLookupTable& cur_lookup = get_wordlookup(source_sentence_[s], target_sentence_[s], wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    uint nIter = 0;
    update_alignment_by_hillclimbing(source_sentence_[s], target_sentence_[s], cur_lookup, nIter, fertility,
                                     expansion_prob, swap_prob, best_known_alignment_[s]);
  }

  if (inform && possible_ref_alignments_.size() > 0) {

    std::cerr << "#### AER after alignment update for " << model_name() << ": " << AER() << std::endl;
    std::cerr << "#### fmeasure after alignment update for " << model_name() << ": " << f_measure() << std::endl;
    std::cerr << "#### DAE/S after alignment update for " << model_name() << ": " << DAE_S() << std::endl;
  }

  return best_known_alignment_;
}

bool FertilityModelTrainerBase::make_alignment_feasible(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment) const
{
  const uint J = source.size();
  const uint I = target.size();

  Math1D::Vector<uint> fertility(I + 1, 0);

  for (uint j = 0; j < J; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  }

  bool changed = false;

  bool have_warned = false;

  if (2 * fertility[0] > J) {

    std::vector<std::pair<double,AlignBaseType> > priority;
    for (uint j = 0; j < J; j++) {

      if (alignment[j] == 0) {
        priority.push_back(std::make_pair(dict_[0][source[j] - 1], j));
      }
    }

    vec_sort(priority);
    assert(priority.size() < 2 || priority[0].first <= priority[1].first);

    for (uint k = 0; 2 * fertility[0] > J; k++) {

      uint j = priority[k].second;

      uint best_i = 0;
      double best = -1.0;
      for (uint i = 1; i <= I; i++) {

        if (fertility[i] >= fertility_limit_[target[i - 1]])
          continue;

        double hyp = dict_[target[i - 1]][lookup(j, i - 1)];
        if (hyp > best) {

          best = hyp;
          best_i = i;
        }
      }

      if (best_i == 0) {

        if (!have_warned) {
          std::
          cerr <<
               "WARNING: the given sentence pair cannot be explained by IBM-3/4/5 with the given fertility limits. J="
               << J << ", I=" << I << std::endl;
          have_warned = true;
        }

        best_i = 1;
        alignment[j] = 1;
        fertility[1]++;

        // for (uint k=fertility_limit_[target[0]]+1; k <= fertility[1]; k++) //ensure that all ways to reduce the fertility are given sufficient probability
        //   fertility_prob_[target[0]][k] = 1e-8;
      }
      else {
        alignment[j] = best_i;
        fertility[best_i]++;
      }
      fertility[0]--;

      changed = true;

      // if (dict_[target[best_i-1]][lookup(j,best_i)] < 0.001) {

      //   dict_[target[best_i-1]] *= 0.999;
      //   dict_[target[best_i-1]][lookup(j,0)] += 0.001;
      // }
    }
  }

  for (uint i = 1; i <= I; i++) {

    if (fertility[i] > fertility_limit_[target[i - 1]]) {

      std::vector<std::pair<double,AlignBaseType> > priority;
      for (uint j = 0; j < J; j++) {

        if (alignment[j] == i) {
          priority.push_back(std::make_pair(dict_[target[i - 1]][lookup(j, i - 1)], j));
        }
      }

      vec_sort(priority);
      assert(priority.size() < 2 || priority[0].first <= priority[1].first);

      for (uint k = 0; fertility[i] > fertility_limit_[target[i - 1]]; k++) {

        uint j = priority[k].second;

        uint best_i = i;
        double best = -1.0;
        for (uint ii = 1; ii <= I; ii++) {

          if (ii == i || fertility[ii] >= fertility_limit_[target[ii - 1]])
            continue;

          double hyp = dict_[target[ii - 1]][lookup(j, ii - 1)];
          if (hyp > best) {

            best = hyp;
            best_i = ii;
          }
        }

        if (best_i == i) {
          //check empty word

          if (2 * fertility[0] + 2 <= J)
            best_i = 0;
        }

        if (best_i == i) {
          if (!have_warned) {
            std::cerr << "WARNING: the given sentence pair cannot be explained by IBM-3/4/5 with the given fertility limits."
                      << std::endl;
            have_warned = true;
          }
          // for (uint k=fertility_limit_[target[i-1]]+1; k <= fertility[i]; k++) //ensure that all ways to reduce the fertility are given sufficient probability
          //   fertility_prob_[target[i-1]][k] = 1e-8;
          break;                //no use resolving the remaining words, it's not possible
        }
        else {
          changed = true;

          alignment[j] = best_i;

          fertility[i]--;
          fertility[best_i]++;
        }

        const uint dict_num = (best_i == 0) ? 0 : target[best_i - 1];
        const uint dict_idx = (best_i == 0) ? source[j] - 1 : lookup(j, best_i - 1);

        if (dict_[dict_num][dict_idx] < 0.001) {

          dict_[dict_num] *= 0.999;
          dict_[dict_num][dict_idx] += 0.001;
        }
      }
    }
  }

  return changed;
}

/* virtual*/ long double FertilityModelTrainerBase::compute_external_alignment(const Storage1D<uint>& source,
    const Storage1D<uint>& target, const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();
  const uint I = target.size();

  //create matrices
  Math2D::Matrix<long double> expansion_prob(J, I + 1);
  Math2D::Matrix<long double> swap_prob(J, J);

  Math1D::Vector<uint> fertility(I + 1, 0);

  uint nIter;

  return update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility, expansion_prob, swap_prob, alignment);
}

// <code> start_alignment </code> is used as initialization for hillclimbing and later modified
// the extracted alignment is written to <code> postdec_alignment </code>
void FertilityModelTrainerBase::compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& start_alignment,
    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment, double threshold)
{
  prepare_external_alignment(source, target, lookup, start_alignment);
  compute_postdec_alignment(source, target, lookup, start_alignment, threshold, postdec_alignment);
}

void FertilityModelTrainerBase::compute_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment,
    double threshold, std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment)
{
  //std::cerr << "compute_postdec_alignment" << std::endl;

  postdec_alignment.clear();

  Math2D::Matrix<double> marg;
  bool converged;
  compute_approximate_jmarginals(source, target, lookup, alignment, marg, converged);

  /*** compute marginals and threshold ***/
  for (uint j = 0; j < marg.yDim(); j++) {

    for (uint i = 1; i < marg.xDim(); i++) {

      if (marg(i, j) >= threshold) {
        postdec_alignment.insert(std::make_pair(j + 1, i));
      }
    }
  }
}

void FertilityModelTrainerBase::compute_approximate_jmarginals(const Math1D::Vector<AlignBaseType>& alignment,
    const Math2D::Matrix<long double>& expansion_move_prob, const Math2D::Matrix<long double>& swap_move_prob,
    const long double sentence_prob, Math2D::Matrix<double>& j_marg) const
{
  //j_marg is double, so we need to divide by sentence_prob in place

  const uint curJ = alignment.size();
  const uint curI = expansion_move_prob.yDim() - 1;

  j_marg.resize(curI + 1, curJ);
  j_marg.set_constant(0.0);

  for (uint j = 0; j < curJ; j++) {

    const uint aj = alignment[j];

    long double addon = sentence_prob;
    for (uint i = 0; i <= curI; i++)
      addon -= expansion_move_prob(j, i);
    for (uint jj = 0; jj < curJ; jj++)
      addon -= swap_move_prob(j, jj);

    j_marg(aj, j) += addon / sentence_prob;

    for (uint i = 0; i <= curI; i++)
      j_marg(i, j) += expansion_move_prob(j, i) / sentence_prob;

    for (uint jj = j + 1; jj < curJ; jj++) {

      const double sprob = swap_move_prob(j, jj);

      if (sprob > 0.0) {

        const double cur_prob = sprob / sentence_prob;

        j_marg(alignment[jj], j) += cur_prob;
        j_marg(alignment[j], jj) += cur_prob;
      }
    }
  }

  for (uint j = 0; j < curJ; j++) {
    const double check_sum = j_marg.row_sum(j);
    assert(check_sum >= 0.99 && check_sum < 1.01);
  }
}

void FertilityModelTrainerBase::compute_approximate_imarginals(const Math1D::Vector<AlignBaseType>& alignment,
    const Math1D::Vector<uint>& fertility, const Math2D::Matrix<long double>& expansion_move_prob,
    const long double sentence_prob, Math2D::Matrix<double>& i_marg) const
{
  //i_marg is double, so we need to divide by sentence_prob in place

  const uint curJ = alignment.size();
  const uint curI = expansion_move_prob.yDim() - 1;

  i_marg.resize(curJ + 1, curI + 1);
  i_marg.set_constant(0.0);

  for (uint i = 0; i <= curI; i++) {

    //std::cerr << "i: " << i << std::endl;

    const uint cur_fert = fertility[i];

    long double addon = sentence_prob;
    for (uint j = 0; j < curJ; j++) {
      if (alignment[j] == i) {
        for (uint ii = 0; ii <= curI; ii++)
          addon -= expansion_move_prob(j, ii);
      }
      else
        addon -= expansion_move_prob(j, i);
    }

    i_marg(cur_fert, i) += addon / sentence_prob;

    //NOTE: swap moves do not change the fertilities
    if (cur_fert > 0) {
      long double alt_addon = 0.0;
      for (uint j = 0; j < curJ; j++) {
        if (alignment[j] == i) {
          for (uint ii = 0; ii <= curI; ii++) {
            alt_addon += expansion_move_prob(j, ii);
          }
        }
      }

      i_marg(cur_fert - 1, i) += alt_addon / sentence_prob;
    }

    if (cur_fert + 1 <= curJ)
      i_marg(cur_fert + 1, i) += expansion_move_prob.row_sum(i) / sentence_prob;

    const double check_sum = i_marg.row_sum(i);
    assert(check_sum >= 0.99 && check_sum < 1.01);
  }
}

//returns the logarithm of the (approximated) normalization constant
/*virtual*/ double FertilityModelTrainerBase::compute_approximate_marginals(const Storage1D<uint>& source,
    const Storage1D<uint>& target, const SingleLookupTable& lookup,
    Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg,
    Math2D::Matrix<double >& i_marg, double hc_mass, bool& converged) const
{
  //NOTE: estep_mode_ is ignored, we use hillclimbing in all cases!

  converged = true;

  //std::cerr << "compute_approximate_marginals" << std::endl;

  Math2D::Matrix<long double> expansion_move_prob;
  Math2D::Matrix<long double> swap_move_prob;

  SingleLookupTable aux_lookup;

  const SingleLookupTable& cur_lookup = get_wordlookup(source, target, wcooc_, nSourceWords_, lookup, aux_lookup);

  Math1D::Vector<uint> fertility(target.size() + 1);

  uint nIter = 0;
  long double best_prob = update_alignment_by_hillclimbing(source, target, cur_lookup, nIter,
                          fertility, expansion_move_prob, swap_move_prob, alignment);

  const long double expansion_prob = expansion_move_prob.sum();
  const long double swap_prob = 0.5 * swap_move_prob.sum();

  const long double sentence_prob = best_prob + expansion_prob + swap_prob;

  compute_approximate_jmarginals(alignment, expansion_move_prob, swap_move_prob, sentence_prob, j_marg);
  compute_approximate_imarginals(alignment, fertility, expansion_move_prob, sentence_prob, i_marg);

  return logl(sentence_prob);
}

/*virtual*/ void FertilityModelTrainerBase::release_memory()
{
  best_known_alignment_.resize(0);
  fertility_limit_.resize(0);
}

double FertilityModelTrainerBase::AER()
{
  double sum_aer = 0.0;
  uint nContributors = 0;

  for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments_.begin();
       it != possible_ref_alignments_.end(); it++) {

    uint s = it->first - 1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add alignment error rate
    sum_aer +=::AER(best_known_alignment_[s], sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1]);
  }

  sum_aer *= 100.0 / nContributors;
  return sum_aer;
}

double FertilityModelTrainerBase::AER(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments)
{
  double sum_aer = 0.0;
  uint nContributors = 0;

  for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments_.begin();
       it != possible_ref_alignments_.end(); it++) {

    uint s = it->first - 1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add alignment error rate
    sum_aer +=::AER(alignments[s], sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1]);
  }

  sum_aer *= 100.0 / nContributors;
  return sum_aer;
}

double FertilityModelTrainerBase::f_measure(double alpha)
{
  double sum_fmeasure = 0.0;
  uint nContributors = 0;

  for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments_.begin();
       it != possible_ref_alignments_.end(); it++) {

    uint s = it->first - 1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add f-measure

    // std::cerr << "s: " << s << ", " << ::f_measure(uint_alignment,sure_ref_alignments_[s+1],possible_ref_alignments_[s+1], alpha) << std::endl;
    // std::cerr << "precision: " << ::precision(uint_alignment,sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]) << std::endl;
    // std::cerr << "recall: " << ::recall(uint_alignment,sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]) << std::endl;
    // std::cerr << "alpha: " << alpha << std::endl;
    // std::cerr << "sure alignments: " << sure_ref_alignments_[s+1] << std::endl;
    // std::cerr << "possible alignments: " << possible_ref_alignments_[s+1] << std::endl;
    // std::cerr << "computed alignment: " << uint_alignment << std::endl;

    sum_fmeasure +=::f_measure(best_known_alignment_[s], sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1], alpha);
  }

  sum_fmeasure /= nContributors;
  return sum_fmeasure;
}

double FertilityModelTrainerBase::f_measure(const Storage1D<Math1D::Vector<AlignBaseType> >& alignment, double alpha)
{
  double sum_fmeasure = 0.0;
  uint nContributors = 0;

  for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments_.begin();
       it != possible_ref_alignments_.end(); it++) {

    uint s = it->first - 1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add f-measure

    sum_fmeasure +=::f_measure(alignment[s], sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1], alpha);
  }

  sum_fmeasure /= nContributors;
  return sum_fmeasure;
}

double FertilityModelTrainerBase::DAE_S()
{
  double sum_errors = 0.0;
  uint nContributors = 0;

  for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments_.begin();
       it != possible_ref_alignments_.end(); it++) {

    uint s = it->first - 1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add DAE/S
    sum_errors +=::nDefiniteAlignmentErrors(best_known_alignment_[s], sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1]);
  }

  sum_errors /= nContributors;
  return sum_errors;
}

double FertilityModelTrainerBase::DAE_S(const Storage1D<Math1D::Vector<AlignBaseType> >& alignment)
{
  double sum_errors = 0.0;
  uint nContributors = 0;

  for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments_.begin();
       it != possible_ref_alignments_.end(); it++) {

    uint s = it->first - 1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add DAE/S
    sum_errors +=::nDefiniteAlignmentErrors(alignment[s], sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1]);
  }

  sum_errors /= nContributors;
  return sum_errors;
}

void FertilityModelTrainerBase::set_hmm_alignments(const HmmWrapper& hmm_wrapper)
{
  SingleLookupTable aux_lookup;

  for (uint s = 0; s < source_sentence_.size(); s++) {

    const Storage1D<uint>& cur_source = source_sentence_[s];
    const Storage1D<uint>& cur_target = target_sentence_[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    const uint curI = cur_target.size();

    compute_ehmm_viterbi_alignment(cur_source, cur_lookup, cur_target, dict_, hmm_wrapper.align_model_[curI - 1],
                                   hmm_wrapper.initial_prob_[curI - 1], best_known_alignment_[s],
                                   hmm_wrapper.hmm_options_, false, false);

    make_alignment_feasible(cur_source, cur_target, cur_lookup, best_known_alignment_[s]);
  }
}

void FertilityModelTrainerBase::set_hmm_alignments(const HmmWrapperWithClasses& hmmc_wrapper)
{
  SingleLookupTable aux_lookup;

  for (uint s = 0; s < source_sentence_.size(); s++) {

    const Storage1D<uint>& cur_source = source_sentence_[s];
    const Storage1D<uint>& cur_target = target_sentence_[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    const uint curI = cur_target.size();

    Storage1D < uint > tclass(curI);
    for (uint i = 0; i < curI; i++)
      tclass[i] = hmmc_wrapper.target_class_[cur_target[i]];

    compute_ehmm_viterbi_alignment(cur_source, cur_lookup, cur_target, tclass, dict_, hmmc_wrapper.align_model_[curI - 1],
                                   hmmc_wrapper.initial_prob_[curI - 1], best_known_alignment_[s],
                                   hmmc_wrapper.hmm_options_, false, false);

    make_alignment_feasible(cur_source, cur_target, cur_lookup, best_known_alignment_[s]);
  }
}

/********************** implementation of FertilityModelTrainer *******************************/

/*static*/ Math1D::NamedVector<long double> FertilityModelTrainer::ld_fac_(MAKENAME(ld_fac_));
/*static*/ Storage1D<Math1D::Vector<long double> > FertilityModelTrainer::choose_factor_;
/*static*/ Storage1D<Math1D::Vector<long double> > FertilityModelTrainer::och_ney_factor_;

FertilityModelTrainer::FertilityModelTrainer(const Storage1D<Math1D::Vector<uint> >& source_sentence,
    const Storage1D<Math2D::Matrix<uint, ushort> >& slookup, const Storage1D<Math1D::Vector<uint> >& target_sentence,
    SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class, uint nSourceWords, uint nTargetWords,
    const floatSingleWordDictionary& prior_weight, FertNullModel empty_word_model, bool smoothed_l0, double l0_beta, double l0_fertpen,
    bool no_factorial, const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
    const std::map<uint, std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
    const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
    uint fertility_limit, MStepSolveMode msolve_mode, HillclimbingMode hillclimb_mode)
  : FertilityModelTrainerBase(source_sentence, slookup, target_sentence, sure_ref_alignments,
                              possible_ref_alignments, dict, wcooc, nSourceWords, nTargetWords,
                              fertility_limit), iter_offs_(0), empty_word_model_(empty_word_model),
    smoothed_l0_(smoothed_l0), l0_beta_(l0_beta), l0_fertpen_(l0_fertpen),
    no_factorial_(no_factorial), nMaxHCIter_(150), msolve_mode_(msolve_mode), hillclimb_mode_(hillclimb_mode),
    prior_weight_(prior_weight), fertility_prob_(nTargetWords, MAKENAME(fertility_prob_)),
    tfert_class_(tfert_class), log_table_(log_table), xlogx_table_(xlogx_table),
    p_zero_pow_(maxJ_ / 2 + 1, -1.0), p_nonzero_pow_(maxJ_ + 1, -1.0)
{
  assert(tfert_class.size() == nTargetWords);
  nTFertClasses_ = tfert_class.max() + 1;
  tfert_class_count_.resize(nTFertClasses_, 0);

  for (uint i = 0; i < dict_.size(); i++)
    for (uint k = 0; k < dict_[i].size(); k++)
      dict_[i][k] = std::max(dict_[i][k],fert_min_dict_entry);

  for (uint i = 1; i < nTargetWords; i++)
    tfert_class_count_[tfert_class[i]]++;

  fertprob_sharing_ = (tfert_class_count_.max() > 1);

#ifndef NDEBUG
  for (uint i = 0; i < prior_weight_.size(); i++) {
    if (prior_weight_[i].size() > 0)
      assert(prior_weight_[i].min() == 0.0);
  }
#endif

  Math1D::Vector<uint> max_fertility(nTargetWords, 0);

  p_zero_ = 0.02;
  p_nonzero_ = 0.98;

  fix_p0_ = false;

  for (size_t s = 0; s < source_sentence.size(); s++) {

    const uint curJ = source_sentence[s].size();
    const uint curI = target_sentence[s].size();

    if (max_fertility[0] < curJ)
      max_fertility[0] = curJ;

    for (uint i = 0; i < curI; i++) {

      const uint t_idx = target_sentence[s][i];

      if (max_fertility[t_idx] < curJ)
        max_fertility[t_idx] = curJ;
    }
  }

  if (ld_fac_.size() == 0) {
    ld_fac_.resize(maxJ_ + 1);
    ld_fac_[0] = 1.0;
    for (uint c = 1; c < ld_fac_.size(); c++)
      ld_fac_[c] = ld_fac_[c - 1] * c;
  }
  if (choose_factor_.size() == 0) {
    choose_factor_.resize(maxJ_ + 1);
    for (uint J = 1; J <= maxJ_; J++) {
      choose_factor_[J].resize(J / 2 + 1);
      for (uint c = 0; c <= J / 2; c++)
        choose_factor_[J][c] = ldchoose(J - c, c, ld_fac_);
    }
  }
  if (och_ney_factor_.size() == 0) {
    och_ney_factor_.resize(maxJ_ + 1);
    for (uint J = 1; J <= maxJ_; J++) {
      och_ney_factor_[J].resize(J / 2 + 1);
      och_ney_factor_[J][0] = 1.0;
      for (uint c = 1; c <= J / 2; c++)
        och_ney_factor_[J][c] = och_ney_factor_[J][c - 1] * (double)c / ((double)J);
    }
  }

  for (uint i = 0; i < nTargetWords; i++) {
    fertility_prob_[i].resize(max_fertility[i] + 1, 1.0 / (max_fertility[i] + 1));
  }

  prior_weight_active_ = false;
  for (uint i = 0; i < prior_weight_.size(); i++) {
    if (prior_weight_[i].max_abs() != 0.0) {
      prior_weight_active_ = true;
      break;
    }
  }
}

FertilityModelTrainer::FertilityModelTrainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
    const Storage1D<Math1D::Vector<uint> >& target_sentence, SingleWordDictionary& dict,
    const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class,
    uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
    const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
    const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
    const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
    const FertModelOptions& options, bool no_factorial, uint fertility_limit)
  : FertilityModelTrainerBase(source_sentence, slookup, target_sentence, sure_ref_alignments,
                              possible_ref_alignments, dict, wcooc, nSourceWords, nTargetWords,
                              fertility_limit), iter_offs_(0), empty_word_model_(options.empty_word_model_),
    smoothed_l0_(options.l0_beta_ != 0.0), l0_beta_(options.l0_beta_), l0_fertpen_(options.l0_fertpen_), no_factorial_(no_factorial),
    nMaxHCIter_(options.nMaxHCIter_), dict_m_step_iter_(options.dict_m_step_iter_), msolve_mode_(options.msolve_mode_),
    hillclimb_mode_(options.hillclimb_mode_), prior_weight_(prior_weight), fertility_prob_(nTargetWords, MAKENAME(fertility_prob_)),
    tfert_class_(tfert_class), log_table_(log_table), xlogx_table_(xlogx_table), p_zero_pow_(maxJ_ / 2 + 1, -1.0), p_nonzero_pow_(maxJ_ + 1, -1.0)
{
  if (options.p0_ >= 0.0) {
    p_zero_ = std::min(0.95,options.p0_);
    p_nonzero_ = 1.0 - p_zero_;
    fix_p0_ = true;
  }
  else {
    p_zero_ = 0.02;
    p_nonzero_ = 0.98;
    fix_p0_ = false;
  }

  assert(tfert_class.size() == nTargetWords);
  nTFertClasses_ = tfert_class.max() + 1;
  tfert_class_count_.resize(nTFertClasses_, 0);

  for (uint i = 0; i < dict_.size(); i++)
    for (uint k = 0; k < dict_[i].size(); k++)
      dict_[i][k] = std::max(dict_[i][k],fert_min_dict_entry);

  for (uint i = 1; i < nTargetWords; i++)
    tfert_class_count_[tfert_class[i]]++;

  fertprob_sharing_ = (tfert_class_count_.max() > 1);

#ifndef NDEBUG
  for (uint i = 0; i < prior_weight_.size(); i++) {
    if (prior_weight_[i].size() > 0)
      assert(prior_weight_[i].min() == 0.0);
  }
#endif

  Math1D::Vector<uint> max_fertility(nTargetWords, 0);

  for (size_t s = 0; s < source_sentence.size(); s++) {

    const uint curJ = source_sentence[s].size();
    const uint curI = target_sentence[s].size();

    if (max_fertility[0] < curJ)
      max_fertility[0] = curJ;

    for (uint i = 0; i < curI; i++) {

      const uint t_idx = target_sentence[s][i];

      if (max_fertility[t_idx] < curJ)
        max_fertility[t_idx] = curJ;
    }
  }

  if (ld_fac_.size() == 0) {
    ld_fac_.resize(maxJ_ + 1);
    ld_fac_[0] = 1.0;
    for (uint c = 1; c < ld_fac_.size(); c++)
      ld_fac_[c] = ld_fac_[c - 1] * c;
  }
  if (choose_factor_.size() == 0) {
    choose_factor_.resize(maxJ_ + 1);
    for (uint J = 1; J <= maxJ_; J++) {
      choose_factor_[J].resize(J / 2 + 1);
      for (uint c = 0; c <= J / 2; c++)
        choose_factor_[J][c] = ldchoose(J - c, c, ld_fac_);
    }
  }
  if (och_ney_factor_.size() == 0) {
    och_ney_factor_.resize(maxJ_ + 1);
    for (uint J = 1; J <= maxJ_; J++) {
      och_ney_factor_[J].resize(J / 2 + 1);
      och_ney_factor_[J][0] = 1.0;
      for (uint c = 1; c <= J / 2; c++)
        och_ney_factor_[J][c] = och_ney_factor_[J][c - 1] * (double)c / ((double)J);
    }
  }

  for (uint i = 0; i < nTargetWords; i++) {
    fertility_prob_[i].resize(max_fertility[i] + 1, 1.0 / (max_fertility[i] + 1));
  }

  prior_weight_active_ = false;
  for (uint i = 0; i < prior_weight_.size(); i++) {
    if (prior_weight_[i].max_abs() != 0.0) {
      prior_weight_active_ = true;
      break;
    }
  }
}

void FertilityModelTrainer::fix_p0(double p0)
{
  p_zero_ = std::min(0.95,p0);
  p_nonzero_ = 1.0 - p_zero_;
  fix_p0_ = true;
}

double FertilityModelTrainer::p_zero() const
{
  return p_zero_;
}

/*virtual*/ void FertilityModelTrainer::release_memory()
{
  best_known_alignment_.resize(0);
  fertility_limit_.resize(0);
  fertility_prob_.resize(0);
}

long double FertilityModelTrainer::alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const
{
  SingleLookupTable aux_lookup;

  const SingleLookupTable& lookup = get_wordlookup(source_sentence_[s], target_sentence_[s], wcooc_, nSourceWords_, slookup_[s], aux_lookup);

  return alignment_prob(source_sentence_[s], target_sentence_[s], lookup, alignment);
}

double FertilityModelTrainer::regularity_term() const
{
  if (!prior_weight_active_)
    return 0.0;

  double reg_term = 0.0;
  for (uint i = 0; i < dict_.size(); i++)
    for (uint k = 0; k < dict_[i].size(); k++) {
      if (smoothed_l0_)
        reg_term += prior_weight_[i][k] * prob_penalty(dict_[i][k], l0_beta_);
      else
        reg_term += prior_weight_[i][k] * dict_[i][k];
    }

  return reg_term;
}

const NamedStorage1D<Math1D::Vector<double> >& FertilityModelTrainer::fertility_prob() const
{
  return fertility_prob_;
}

void FertilityModelTrainer::write_fertilities(std::string filename)
{
  std::ofstream out(filename.c_str());

  for (uint k = 0; k < fertility_prob_.size(); k++) {

    for (uint l = 0; l < fertility_prob_[k].size(); l++)
      out << fertility_prob_[k][l] << " ";

    out << std::endl;
  }
}

void FertilityModelTrainer::set_hc_iteration_limit(uint new_limit)
{
  nMaxHCIter_ = new_limit;
}

/*virtual*/ void FertilityModelTrainer::set_fertility_limit(uint new_limit)
{
  fertility_limit_.set_constant(new_limit);

  for (uint i = 1; i < nTargetWords_; i++) {
    fertility_limit_[i] = std::min<uint>(new_limit, fertility_prob_[i].size() - 1);

    fertility_prob_[i].set_constant(0.0);
    for (uint c = 0; c <= fertility_limit_[i]; c++)
      fertility_prob_[i][c] = 1.0 / (fertility_limit_[i] + 1);
  }
}

/*virtual*/ void FertilityModelTrainer::set_rare_fertility_limit(uint new_limit, uint max_count)
{
  Math1D::Vector<uint> count(nTargetWords_, 0);
  for (uint s = 0; s < target_sentence_.size(); s++) {

    const Storage1D<uint>& cur_target = target_sentence_[s];
    for (uint i = 0; i < cur_target.size(); i++)
      count[cur_target[i]]++;
  }

  for (uint i = 1; i < nTargetWords_; i++) {
    if (count[i] <= max_count) {
      fertility_limit_[i] = std::min<uint> (new_limit, fertility_prob_[i].size() - 1);
      fertility_prob_[i].set_constant(0.0);
      for (uint c = 0; c <= fertility_limit_[i]; c++)
        fertility_prob_[i][c] = 1.0 / (fertility_limit_[i] + 1);
    }
  }
}

void FertilityModelTrainer::PostdecEval(double& aer, double& f_measure, double& daes, double threshold, double alpha)
{
  aer = 0.0;
  f_measure = 0.0;
  daes = 0.0;

  uint nContributors = 0;

  SingleLookupTable aux_lookup;

  uint nIter = 0;

  for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments_.begin();
       it != possible_ref_alignments_.end(); it++) {

    uint s = it->first - 1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;

    const SingleLookupTable& cur_lookup = get_wordlookup(source_sentence_[s], target_sentence_[s], wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    Math1D::Vector<AlignBaseType> alignment = FertilityModelTrainer::best_known_alignment_[s];

    std::set<std::pair<AlignBaseType,AlignBaseType > >postdec_alignment;
    compute_postdec_alignment(source_sentence_[s], target_sentence_[s], cur_lookup, alignment, threshold, postdec_alignment);

    aer +=::AER(postdec_alignment, sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1]);
    f_measure +=::f_measure(postdec_alignment, sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1], alpha);
    daes +=::nDefiniteAlignmentErrors(postdec_alignment, sure_ref_alignments_[s + 1], possible_ref_alignments_[s + 1]);
  }

  aer /= nContributors;
  f_measure /= nContributors;
  daes /= nContributors;
}

void FertilityModelTrainer::update_fertility_prob(const Storage1D<Math1D::Vector<double> >& ffert_count,
    double min_prob, bool with_regularity)
{
  if (fertprob_sharing_) {

    Storage1D<Math1D::Vector<double> > class_count(nTFertClasses_);
    Math1D::Vector<uint> prob_num(nTFertClasses_);
    for (uint i = 1; i < ffert_count.size(); i++) {
      const Math1D::Vector<double> cur_ffert_count = ffert_count[i];
      const uint size = cur_ffert_count.size();

      const uint c = tfert_class_[i];
      Math1D::Vector<double>& cur_class_count = class_count[c];

      if (cur_class_count.size() < size) {
        cur_class_count.resize(size, 0.0);
        prob_num[c] = i;
      }
      for (uint k = 0; k < size; k++)
        cur_class_count[k] += cur_ffert_count[k];
    }

    for (uint i = 0; i < nTFertClasses_; i++) {
      if (!with_regularity || l0_fertpen_ == 0.0)
        class_count[i] *= 1.0 / class_count[i].sum();
      else {
        Math1D::Vector<double>& cur_class_count = class_count[i];
        Math1D::Vector<double>& cur_prob = fertility_prob_[prob_num[i]];
        Math1D::Vector<float> prior_weight(cur_class_count.size(), l0_fertpen_ * tfert_class_count_[i]);
        single_dict_m_step(cur_class_count, prior_weight, cur_prob, 1.0, fert_m_step_iter_, true, l0_beta_, false);
        cur_class_count = cur_prob;
      }
    }

    for (uint i = 1; i < ffert_count.size(); i++) {
      Math1D::Vector<double>& cur_fert_prob = fertility_prob_[i];
      const Math1D::Vector<double>& cur_class_count = class_count[tfert_class_[i]];
      for (uint f = 0; f < cur_fert_prob.size(); f++) {
        const double real_min_prob = (f <= fertility_limit_[i]) ? min_prob : 1e-15;
        cur_fert_prob[f] = std::max(real_min_prob, cur_class_count[f]);
      }
    }
  }
  else {
    for (uint i = 1; i < ffert_count.size(); i++) {

      //std::cerr << "i: " << i << std::endl;

      const Math1D::Vector<double>& cur_count = ffert_count[i];
      const double sum = cur_count.sum();

      if (sum > 1e-305) {

        Math1D::Vector<double>& cur_fert_prob = fertility_prob_[i];

        if (!with_regularity || l0_fertpen_ == 0.0) {

          assert(sum > 0.0);
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));

          for (uint f = 0; f < cur_fert_prob.size(); f++) {
            const double real_min_prob = (f <= fertility_limit_[i]) ? min_prob : 1e-15;
            cur_fert_prob[f] = std::max(real_min_prob, inv_sum * cur_count[f]);
          }
        }
        else {
          Math1D::Vector<float> prior_weight(cur_count.size(), l0_fertpen_);
          single_dict_m_step(cur_count, prior_weight, cur_fert_prob, 1.0, 250, true, l0_beta_, false);
        }
      }
      else {
        //std::cerr << "WARNING: did not update fertility count because sum was " << sum << std::endl;
      }
    }
  }
}

void FertilityModelTrainer::common_prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  const uint J = source.size();
  const uint I = target.size();

  assert(lookup.xDim() == J && lookup.yDim() == I);

  if (alignment.size() != J)
    alignment.resize(J, 1);

  Math1D::Vector<uint> fertility(I + 1, 0);

  for (uint j = 0; j < J; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  }

  if (fertility[0] > 0 && p_zero_ < 1e-12)
    p_zero_ = 1e-12;

  make_alignment_feasible(source, target, lookup, alignment);

  /*** check if fertility tables are large enough ***/
  for (uint i = 0; i < I; i++) {

    if (fertility_prob_[target[i]].size() < J + 1)
      fertility_prob_[target[i]].resize(J + 1, 1e-15);

    if (fertility_prob_[target[i]][fertility[i + 1]] < 1e-15)
      fertility_prob_[target[i]][fertility[i + 1]] = 1e-15;

    if (fertility_prob_[target[i]].sum() < 0.5)
      fertility_prob_[target[i]].set_constant(1.0 / fertility_prob_[target[i]].size());

    if (fertility_prob_[target[i]][fertility[i + 1]] < 1e-8)
      fertility_prob_[target[i]][fertility[i + 1]] = 1e-8;
  }

  /*** check if a source word does not have a translation (with non-zero prob.) ***/
  for (uint j = 0; j < J; j++) {
    uint src_idx = source[j];

    double sum = dict_[0][src_idx - 1];
    for (uint i = 0; i < I; i++)
      sum += dict_[target[i]][lookup(j, i)];

    if (sum < 1e-100) {
      for (uint i = 0; i < I; i++)
        dict_[target[i]][lookup(j, i)] = 1e-15;
    }

    uint aj = alignment[j];
    if (aj == 0) {
      if (dict_[0][src_idx - 1] < 1e-20)
        dict_[0][src_idx - 1] = 1e-20;
    }
    else {
      if (dict_[target[aj - 1]][lookup(j, aj - 1)] < 1e-20)
        dict_[target[aj - 1]][lookup(j, aj - 1)] = 1e-20;
    }
  }

  if (J >= choose_factor_.size() || choose_factor_[J].size() == 0) {
    choose_factor_.resize(J + 1);
    choose_factor_[J].resize(J / 2 + 1);
    for (uint c = 0; c <= J / 2; c++)
      choose_factor_[J][c] = ldchoose(J - c, c, ld_fac_);
  }

  if (J >= och_ney_factor_.size() || och_ney_factor_[J].size() == 0) {
    och_ney_factor_.resize(J + 1);
    och_ney_factor_[J].resize(J / 2 + 1);
    och_ney_factor_[J][0] = 1.0;
    for (uint c = 1; c <= J / 2; c++)
      och_ney_factor_[J][c] = och_ney_factor_[J][c - 1] * (double)c / ((double)J);
  }

}

void FertilityModelTrainer::init_fertilities(FertilityModelTrainerBase* prev_model, double count_weight)
{
  SingleLookupTable aux_lookup;

  NamedStorage1D<Math1D::Vector<double> > fert_count(nTargetWords_,  MAKENAME(fert_count));
  for (uint i = 0; i < nTargetWords_; i++) {
    fert_count[i].resize(fertility_prob_[i].size(), 0.0);
  }

  for (size_t s = 0; s < source_sentence_.size(); s++) {

    //std::cerr << "s: " << s << std::endl;

    const Storage1D<uint>& cur_source = source_sentence_[s];
    const Storage1D<uint>& cur_target = target_sentence_[s];

    const uint curI = cur_target.size();
    const uint curJ = cur_source.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

    if (prev_model != 0) {
      prev_model->compute_external_alignment(cur_source, cur_target, cur_lookup, cur_alignment);
      make_alignment_feasible(cur_source, cur_target, cur_lookup, cur_alignment);
    }

    Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

    for (uint j = 0; j < curJ; j++) {
      const uint aj = cur_alignment[j];
      fertility[aj]++;
    }

    fert_count[0][fertility[0]]++;

    for (uint i = 0; i < curI; i++) {
      const uint t_idx = target_sentence_[s][i];

      fert_count[t_idx][fertility[i + 1]]++;
    }
  }

  //init fertility prob. by weighted combination of uniform distribution
  // and counts from Viterbi alignments

  const double uni_weight = 1.0 - count_weight;

  if (fertprob_sharing_) {

    update_fertility_prob(fert_count);

    for (uint i = 1; i < nTargetWords_; i++) {

      const uint max_fert = fertility_prob_[i].size();
      const double uni_contrib = uni_weight / std::min<ushort>(max_fert, fertility_limit_[i]);

      for (uint f = 0; f < max_fert; f++) {

        if (f <= fertility_limit_[i])
          fertility_prob_[i][f] = uni_contrib + count_weight * fertility_prob_[i][f];
        else
          fertility_prob_[i][f] = 1e-8;
      }
    }
  }
  else {

    for (uint i = 1; i < nTargetWords_; i++) {

      const Math1D::Vector<double>& cur_fert_count = fert_count[i];

      const uint max_fert = cur_fert_count.size();

      double fc_sum = cur_fert_count.sum();

      if (fc_sum > 1e-300) {

        const double inv_fc_sum = 1.0 / fc_sum;

        const double uni_contrib = uni_weight / std::min < ushort > (max_fert, fertility_limit_[i]);
        for (uint f = 0; f < max_fert; f++) {

          if (f <= fertility_limit_[i])
            fertility_prob_[i][f] = uni_contrib + count_weight * inv_fc_sum * cur_fert_count[f];
          else
            fertility_prob_[i][f] = count_weight * inv_fc_sum * cur_fert_count[f];
        }
      }
      else {
        //this can happen if there is a development corpus
        for (uint f = 0; f < max_fert; f++) {

          if (f <= fertility_limit_[i])
            fertility_prob_[i][f] = 1.0 / std::min < ushort > (fertility_limit_[i] + 1, max_fert);
          else
            fertility_prob_[i][f] = 1e-8;
        }
      }
    }
  }
}
