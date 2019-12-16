/*** written by Thomas Schoenemann. Started as a private person, October 2009
 *** continued at Lund University, Sweden, 2010, as a private person, and at the University of Düsseldorf, Germany, 2012,
 *** and since as a private person ***/

#include "ibm1_training.hh"

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "matrix.hh"

#include "training_common.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"
#include "storage_util.hh"

#include "projection.hh"

#ifdef HAS_CBC
#include "sparse_matrix_description.hh"
#include "OsiClpSolverInterface.hpp"
#include "CbcModel.hpp"
#include "CglGomory/CglGomory.hpp"
#endif

IBM1Options::IBM1Options(uint nSourceWords, uint nTargetWords, RefAlignmentStructure& sure_ref_alignments,
                         RefAlignmentStructure& possible_ref_alignments):
  nIterations_(5), smoothed_l0_(false), l0_beta_(1.0), print_energy_(true),
  nSourceWords_(nSourceWords), nTargetWords_(nTargetWords), dict_m_step_iter_(45),
  sure_ref_alignments_(sure_ref_alignments), possible_ref_alignments_(possible_ref_alignments)
{
}

double ibm1_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                       const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords)
{
  //std::cerr << "calculating IBM-1 perplexity" << std::endl;

  double sum = 0.0;

  SingleLookupTable aux_lookup;

  const size_t nSentences = target.size();
  assert(slookup.size() == nSentences);

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    sum += curJ * std::log(curI+1);

    for (uint j = 0; j < curJ; j++) {

      double cur_sum = dict[0][cur_source[j] - 1];      // handles empty word

      for (uint i = 0; i < curI; i++) {
        cur_sum += dict[cur_target[i]][cur_lookup(j, i)];
      }

      sum -= std::log(cur_sum);
    }
  }

  return sum / nSentences;
}

double ibm1_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                   const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords,
                   const floatSingleWordDictionary& prior_weight, bool smoothed_l0, double l0_beta, double dict_weight_sum)
{
  double energy = 0.0;

  if (dict_weight_sum != 0.0) {
    for (uint i = 0; i < dict.size(); i++) {

      const Math1D::Vector<double>& cur_dict = dict[i];
      const Math1D::Vector<float>& cur_prior = prior_weight[i];

      const uint size = cur_dict.size();

      if (smoothed_l0) {
        for (uint k = 0; k < size; k++)
          energy += cur_prior[k] * prob_penalty(cur_dict[k], l0_beta);
      }
      else {
        for (uint k = 0; k < size; k++)
          energy += cur_prior[k] * cur_dict[k];
      }
    }
  }

  energy += ibm1_perplexity(source, slookup, target, dict, wcooc, nSourceWords);

  return energy;
}

void train_ibm1(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                const CooccuringWordsType& wcooc, SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                const IBM1Options& options)
{
  const uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  assert(wcooc.size() == options.nTargetWords_);
  dict.resize_dirty(options.nTargetWords_);

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  const uint nSourceWords = options.nSourceWords_;

  SingleLookupTable aux_lookup;

  //prepare dictionary
  for (uint i = 0; i < options.nTargetWords_; i++) {

    const uint size = (i == 0) ? options.nSourceWords_ - 1 : wcooc[i].size();
    if (size == 0) {
      std::cerr << "WARNING: dict-size for t-word " << i << " is zero" << std::endl;
    }

    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double)size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

#if 1
  for (uint i = 1; i < options.nTargetWords_; i++) {
    dict[i].set_constant(0.0);
  }
  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    for (uint i = 0; i < curI; i++) {
      uint tidx = cur_target[i];
      for (uint j = 0; j < curJ; j++) {

        dict[tidx][cur_lookup(j, i)] += 1.0;
      }
    }
  }

  for (uint i = 1; i < options.nTargetWords_; i++) {
    double sum = dict[i].sum();
    if (sum > 1e-305)
      dict[i] *= 1.0 / sum;
  }
#endif

  //fractional counts used for EM-iterations
  NamedStorage1D<Math1D::Vector<double> > fcount(options.nTargetWords_, MAKENAME(fcount));
  for (uint i = 0; i < options.nTargetWords_; i++) {
    fcount[i].resize(dict[i].size());
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM-1 EM-iteration #" << iter << std::endl;

    /*** a) compute fractional counts ***/

    for (uint i = 0; i < options.nTargetWords_; i++) {
      fcount[i].set_constant(0.0);
    }

    for (size_t s = 0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        double coeff = dict[0][s_idx - 1];      // entry for empty word (the emtpy word is not listed, hence s_idx-1)
        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          coeff += dict[t_idx][cur_lookup(j, i)];
        }
        coeff = 1.0 / coeff;

        assert(!isnan(coeff));

        fcount[0][s_idx - 1] += coeff * dict[0][s_idx - 1];
        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          const uint k = cur_lookup(j, i);
          fcount[t_idx][k] += coeff * dict[t_idx][k];
        }
      }
    } //loop over sentences finished

    std::cerr << "updating dict from counts" << std::endl;

    /*** update dict from counts ***/

    update_dict_from_counts(fcount, prior_weight, dict_weight_sum, smoothed_l0, l0_beta, options.dict_m_step_iter_, dict,
                            ibm1_min_dict_entry, options.unconstrained_m_step_);

    if (options.print_energy_) {
      std::cerr << "IBM-1 energy after iteration #" << iter << ": "
                << ibm1_energy(source, slookup, target, dict, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum) << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm1_viterbi_alignment(source[s], cur_lookup, target[s], dict, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm1_postdec_alignment(source[s], cur_lookup, target[s], dict, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### IBM-1 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;
      std::cerr << "#### IBM-1 Postdec-AER after iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Postdec-fmeasure after iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Postdec-DAE/S after iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  }  //end for (iter)
}

void train_ibm1_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                               const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                               SingleWordDictionary& dict,      //uint nIter,
                               const floatSingleWordDictionary& prior_weight, const IBM1Options& options)
{
  const uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  assert(wcooc.size() == options.nTargetWords_);
  dict.resize_dirty(options.nTargetWords_);

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  //prepare dictionary
  for (uint i = 0; i < options.nTargetWords_; i++) {

    const uint size = (i == 0) ? options.nSourceWords_ - 1 : wcooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double)size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

  Math1D::Vector<double> slack_vector(options.nTargetWords_, 0.0);

  SingleLookupTable aux_lookup;

  const uint nSourceWords = options.nSourceWords_;

#if 1
  for (uint i = 1; i < options.nTargetWords_; i++) {
    dict[i].set_constant(0.0);
  }

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    for (uint i = 0; i < curI; i++) {
      uint tidx = cur_target[i];
      for (uint j = 0; j < curJ; j++) {
        dict[tidx][cur_lookup(j, i)] += 1.0;
      }
    }
  }

  for (uint i = 1; i < options.nTargetWords_; i++) {
    double sum = dict[i].sum();
    if (sum > 1e-305)
      dict[i] *= 1.0 / sum;
  }
#endif

  double energy = ibm1_energy(source, slookup, target, dict, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum);

  std::cerr << "initial energy: " << energy << std::endl;

  SingleWordDictionary new_dict(options.nTargetWords_, MAKENAME(new_dict));
  SingleWordDictionary hyp_dict(options.nTargetWords_, MAKENAME(hyp_dict));

  for (uint i = 0; i < options.nTargetWords_; i++) {

    const uint size = dict[i].size();
    new_dict[i].resize_dirty(size);
    hyp_dict[i].resize_dirty(size);
  }

  Math1D::Vector<double> new_slack_vector(options.nTargetWords_, 0.0);

  //double alpha = 0.0001; //0.01;
  //double alpha = 0.5; //0.1; // 0.0001;
  //double alpha = 1.0;

  double alpha = 100.0;

  double line_reduction_factor = 0.5;
  //double line_reduction_factor = 0.75;

  uint nSuccessiveReductions = 0;

  double best_lower_bound = -1e300;

  SingleWordDictionary dict_grad(options.nTargetWords_, MAKENAME(dict_grad));
  SingleWordDictionary old_dict = dict;

  for (uint i = 0; i < options.nTargetWords_; i++) {

    const uint size = dict[i].size();
    dict_grad[i].resize_dirty(size);
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM-1 gradient descent iteration #" << iter << std::endl;

    /**** calcuate gradients ****/

    //SPG--START
    SingleWordDictionary old_dict_grad;
    if (iter > 1)
      old_dict_grad = dict_grad;
    //SPG--END

    for (uint i = 0; i < options.nTargetWords_; i++) {
      dict_grad[i].set_constant(0.0);
    }

    for (size_t s = 0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      for (uint j = 0; j < curJ; j++) {

        uint s_idx = cur_source[j];

        double sum = dict[0][s_idx - 1];

        for (uint i = 0; i < curI; i++)
          sum += dict[cur_target[i]][cur_lookup(j, i)];

        const double cur_grad = -1.0 / sum;

        dict_grad[0][s_idx - 1] += cur_grad;
        for (uint i = 0; i < curI; i++)
          dict_grad[cur_target[i]][cur_lookup(j, i)] += cur_grad;
      }
    }

    for (uint i = 0; i < options.nTargetWords_; i++) {
      dict_grad[i] *= 1.0 / nSentences;
    }

    if (dict_weight_sum != 0.0) {
      for (uint i = 0; i < options.nTargetWords_; i++) {

        const Math1D::Vector<double>& cur_dict = dict[i];
        const Math1D::Vector<float>& cur_prior = prior_weight[i];
        Math1D::Vector<double>& cur_dict_grad = dict_grad[i];
        const uint size = cur_dict.size();

        for (uint k = 0; k < size; k++) {
          if (smoothed_l0)
            cur_dict_grad[k] += cur_prior[k] * prob_pen_prime(cur_dict[k], l0_beta);
          else
            cur_dict_grad[k] += cur_prior[k];
        }
      }
    }

    /**** compute lower bound ****/

    //lower bounds can only be derived for convex functions. L0 isn't convex!!!
    if (dict_weight_sum == 0.0 || !smoothed_l0) {
      double lower_bound = energy;
      for (uint i = 0; i < options.nTargetWords_; i++) {

        const Math1D::Vector<double>& cur_dict_grad = dict_grad[i];

        if (dict_weight_sum != 0.0) //consider slack gradient of 0
          lower_bound += std::min(0.0, cur_dict_grad.min());
        else
          lower_bound += cur_dict_grad.min();
        lower_bound -= cur_dict_grad % dict[i];
      }

      best_lower_bound = std::max(best_lower_bound, lower_bound);

      std::cerr << "lower bound: " << lower_bound << ", best known: " << best_lower_bound << std::endl;
    }

    //SPG--START
#if 0
    if (iter > 1) {
      double alpha_num = 0.0;
      double alpha_denom = 0.0;

      for (uint i = 0; i < options.nTargetWords_; i++) {

        for (uint k = 0; k < dict[i].size(); k++) {
          double pdiff = dict[i][k] - old_dict[i][k];
          double gdiff = dict_grad[i][k] - old_dict_grad[i][k];

          alpha_num += pdiff * pdiff;
          alpha_denom += pdiff * gdiff;
        }
      }

      assert(alpha_denom > 1e-305);
      alpha = alpha_num / alpha_denom;
      std::cerr << "org alpha: " << alpha << std::endl;
      if (alpha < 1e-4)
        alpha = 1e-4;
      if (alpha > 1e6)
        alpha = 1e6;
      std::cerr << "next alpha: " << alpha << ", denom: " << alpha_denom << std::endl;
    }
#endif
    //SPG--END

    /**** move in gradient direction ****/
    double real_alpha = alpha;
#if 0
    double sqr_grad_norm = 0.0;
    for (uint i = 0; i < options.nTargetWords_; i++)
      sqr_grad_norm += dict_grad[i].sqr_norm();
    real_alpha /= sqrt(sqr_grad_norm);

    // double highest_norm = 0.0;
    // for (uint i=0; i < options.nTargetWords_; i++)
    //   highest_norm = std::max(highest_norm,dict_grad[i].sqr_norm());
    // real_alpha /= sqrt(highest_norm);
#endif

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //for (uint k = 0; k < dict[i].size(); k++)
      //  new_dict[i][k] = dict[i][k] - real_alpha * dict_grad[i][k];
      Math1D::go_in_neg_direction(new_dict[i], dict[i], dict_grad[i], real_alpha);
    }

    if (dict_weight_sum != 0.0)
      new_slack_vector = slack_vector;

    /**** reproject on the simplices [Michelot 1986] ****/
    for (uint i = 0; i < options.nTargetWords_; i++) {

      const uint nCurWords = new_dict[i].size();

      if (dict_weight_sum != 0.0)
        projection_on_simplex_with_slack(new_dict[i].direct_access(), slack_vector[i], nCurWords, ibm1_min_dict_entry);
      else
        projection_on_simplex(new_dict[i].direct_access(), nCurWords, ibm1_min_dict_entry);
    }

    //SPG--START
    old_dict = dict;
    //SPG--END

    /**** find a suitable step size ****/

    double lambda = 1.0;
    double best_lambda = 1.0;

    double hyp_energy = 1e300;

    uint nInnerIter = 0;

    bool decreasing = true;

    while (hyp_energy > energy || decreasing) {

      nInnerIter++;

      if (hyp_energy <= 0.95 * energy)
        break;

      if (hyp_energy < 0.99 * energy && nInnerIter > 3)
        break;

      lambda *= line_reduction_factor;

      const double neg_lambda = 1.0 - lambda;

      for (uint i = 0; i < options.nTargetWords_; i++) {

        //for (uint k = 0; k < dict[i].size(); k++)
        //  hyp_dict[i][k] = neg_lambda * dict[i][k] + lambda * new_dict[i][k];

        assert(dict[i].size() == hyp_dict[i].size());
        Math1D::assign_weighted_combination(hyp_dict[i], neg_lambda, dict[i], lambda, new_dict[i]);
      }

      double new_energy = ibm1_energy(source, slookup, target, hyp_dict, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum);

      std::cerr << "new hyp: " << new_energy << ", previous: " << hyp_energy << std::endl;

      if (new_energy < hyp_energy) {
        hyp_energy = new_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;
    }

    //EXPERIMENTAL
#if 0
    if (nInnerIter > 4)
      alpha *= 1.5;
#endif
    //END_EXPERIMENTAL

    if (nInnerIter > 4) {
      nSuccessiveReductions++;
    }
    else {
      nSuccessiveReductions = 0;
    }

    if (nSuccessiveReductions > 15) {
      line_reduction_factor *= 0.9;
      nSuccessiveReductions = 0;
    }
    //     std::cerr << "alpha: " << alpha << std::endl;

    const double neg_best_lambda = 1.0 - best_lambda;

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //for (uint k = 0; k < dict[i].size(); k++)
      //  dict[i][k] = neg_best_lambda * dict[i][k] + best_lambda * new_dict[i][k];

      Math1D::assign_weighted_combination(dict[i], neg_best_lambda, dict[i], best_lambda, new_dict[i]);
    }
    
    if (dict_weight_sum > 0.0)
      Math1D::assign_weighted_combination(slack_vector, neg_best_lambda, slack_vector, best_lambda, new_slack_vector);

#ifndef NDEBUG
    double check_energy = ibm1_energy(source, slookup, target, dict, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum);
    assert(fabs(check_energy - hyp_energy) < 0.0025);
#endif

    energy = hyp_energy;

    //     if (best_lambda == 1.0)
    //       alpha *= 1.5;
    //     else
    //       alpha *= 0.75;

    if (options.print_energy_)
      std::cerr << "energy: " << energy << std::endl;

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm1_viterbi_alignment(source[s], cur_lookup, target[s], dict, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm1_postdec_alignment(source[s], cur_lookup, target[s], dict, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### IBM-1 Viterbi-AER after gd-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Viterbi-fmeasure after gd-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Viterbi-DAE/S after gd-iteration #" << iter << ": " << nErrors << std::endl;
      std::cerr << "#### IBM-1 Postdec-AER after gd-iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Postdec-fmeasure after gd-iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Postdec-DAE/S after gd-iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }

    std::cerr << "slack sum: " << slack_vector.sum() << std::endl;
  } //end for (iter)

  //std::cerr << "slack sum: " << slack_vector.sum() << std::endl;
}

double compute_ibm1_lbfgs_gradient(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                   const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                   const SingleWordDictionary& dict, const SingleWordDictionary& dict_param,
                                   const floatSingleWordDictionary& prior_weight, const IBM1Options& options,
                                   SingleWordDictionary& dict_param_grad, bool smoothed_l0, double l0_beta, const double dict_weight_sum)
{
  //WARNING: regularity terms are so far ignored

  const uint nSentences = source.size();
  const uint nSourceWords = options.nSourceWords_;

  SingleLookupTable aux_lookup;

  Math1D::Vector<double> param_scale(dict.size(), 0.0);
  for (uint k = 0; k < dict.size(); k++) {
    double sum = 0.0;
    for (uint i = 0; i < dict[k].size(); i++)
      sum += dict_param[k][i] * dict_param[k][i];
    param_scale[k] = sum;
  }

  SingleWordDictionary dict_grad = dict_param_grad;

  // for (size_t s=0; s < nSentences; s++) {

  //   const Storage1D<uint>& cur_source = source[s];
  //   const Storage1D<uint>& cur_target = target[s];

  //   const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);

  //   const uint nCurSourceWords = cur_source.size();
  //   const uint nCurTargetWords = cur_target.size();

  //   sum += nCurSourceWords*std::log(nCurTargetWords);

  //   for (uint j=0; j < nCurSourceWords; j++) {

  //     double cur_sum = dict[0][cur_source[j]-1]; // handles empty word

  //     for (uint i=0; i < nCurTargetWords; i++) {
  //       cur_sum += dict[cur_target[i]][cur_lookup(j,i)];
  //     }

  //     sum -= std::log(cur_sum);
  //   }
  // }

  // return sum / nSentences;

  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_grad[i].set_constant(0.0);
    dict_param_grad[i].set_constant(0.0);
  }

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    for (uint j = 0; j < curJ; j++) {

      double cur_sum = dict[0][cur_source[j] - 1];      // handles empty word

      for (uint i = 0; i < curI; i++) {
        cur_sum += dict[cur_target[i]][cur_lookup(j, i)];
      }

      const double grad_weight = -1.0 / cur_sum;

      //empty word
      dict_grad[0][cur_source[j] - 1] += grad_weight;

      //real words
      for (uint i = 0; i < curI; i++) {
        dict_grad[cur_target[i]][cur_lookup(j, i)] += grad_weight;
      }
    }
  }

  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_grad[i] *= 1.0 / nSentences;
  }

  //regularity terms
  if (dict_weight_sum != 0.0) {

    for (uint i = 0; i < options.nTargetWords_; i++) {

      const uint size = dict[i].size();

      for (uint k = 0; k < size; k++) {
        if (smoothed_l0)
          dict_grad[i][k] += prior_weight[i][k] * prob_pen_prime(dict[i][k], l0_beta);
        else
          dict_grad[i][k] += prior_weight[i][k];
      }
    }
  }

  double lower_bound_share = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {

    //if (regularity_weight != 0.0)
    if (false)
      lower_bound_share += std::min(0.0, dict_grad[i].min());
    else
      lower_bound_share += dict_grad[i].min();
    lower_bound_share -= dict_grad[i] % dict[i];
  }


  for (uint i = 0; i < options.nTargetWords_; i++) {

    //this is rather complicated: need the quotient rule of derivatives
    // but there should be plenty of room for optimization

    for (uint k = 0; k < dict[i].size(); k++) {

      const double weight = dict_grad[i][k];
      dict_param_grad[i][k] += weight * 2.0 *   //dict_param[i][k] * //mult is done below
                               (1.0 - dict[i][k]);   // / param_scale[i]; //division is done below

      for (uint kk = 0; kk < dict[i].size(); kk++) {
        if (kk != k)
          dict_param_grad[i][kk] -= weight * 2.0 * dict[i][k] /* *dict_param[i][kk] / param_scale[i] */ ;       //mult is done below
      }
    }

    for (uint k = 0; k < dict[i].size(); k++) {
      dict_param_grad[i][k] *= dict_param[i][k] / param_scale[i];
    }
  }

  return lower_bound_share;
}

void train_ibm1_lbfgs_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                  const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                  SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                  const IBM1Options& options, uint L)
{
  //NOTE: so far we do not consider slack variables

  const uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  assert(wcooc.size() == options.nTargetWords_);
  dict.resize_dirty(options.nTargetWords_);

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  //prepare dictionary
  for (uint i = 0; i < options.nTargetWords_; i++) {
    const uint size = (i == 0) ? options.nSourceWords_ - 1 : wcooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double)size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

  Math1D::Vector<double> slack_vector(options.nTargetWords_, 0.0);

  SingleLookupTable aux_lookup;

  const uint nSourceWords = options.nSourceWords_;

#if 1
  for (uint i = 1; i < options.nTargetWords_; i++) {
    dict[i].set_constant(0.0);
  }

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    for (uint i = 0; i < curI; i++) {
      uint tidx = cur_target[i];
      for (uint j = 0; j < curJ; j++) {

        dict[tidx][cur_lookup(j, i)] += 1.0;
      }
    }
  }

  for (uint i = 1; i < options.nTargetWords_; i++) {
    double sum = dict[i].sum();
    if (sum > 1e-305)
      dict[i] *= 1.0 / sum;
  }
#endif

  double line_reduction_factor = 0.5;

  double energy = ibm1_energy(source, slookup, target, dict, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum);

  std::cerr << "initial energy: " << energy << std::endl;

  //we convert this CONSTRAINED PROBLEM to an unconstrained one
  //NOTE: we probably lose convexity with this transformation

  SingleWordDictionary param = dict;
  for (uint i = 0; i < param.size(); i++)
    for (uint k = 0; k < param[i].size(); k++)
      param[i][k] = sqrt(dict[i][k]);

  SingleWordDictionary dict_param_grad = dict;
  SingleWordDictionary hyp_dict = dict;
  SingleWordDictionary search_direction = dict;

  Storage1D<SingleWordDictionary> step(L);
  Storage1D<SingleWordDictionary> grad_diff(L);

  for (uint l = 0; l < L; l++) {
    step[l] = dict;
    grad_diff[l] = dict;
  }

  Math1D::Vector<double> rho(L);
  Math1D::Vector<double> alpha(L);

  double best_lower_bound = -1e300;

  uint last_restart = 1;
  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM-1 L-BGFS iteration #" << iter << std::endl;

    /*** update the gradient ***/
    double lower_bound = energy + compute_ibm1_lbfgs_gradient(source, slookup, target, wcooc, dict, param, prior_weight, options,
                         dict_param_grad, smoothed_l0, l0_beta, dict_weight_sum);
    best_lower_bound = std::max(best_lower_bound, lower_bound);

    std::cerr << "lower bound: " << lower_bound << ", best lower bound: " << best_lower_bound << std::endl;

    double sqr_grad_norm = 0.0;
    for (uint i = 0; i < dict_param_grad.size(); i++)
      sqr_grad_norm += dict_param_grad[i].sqr_norm();

    std::cerr << "squared gradient norm: " << sqr_grad_norm << std::endl;

    /** 2a finish setting the latest grad difference and compute the corresponding rho **/
    double latest_inv_rho = 0.0;

    if (iter > last_restart) {
      SingleWordDictionary& cur_grad_diff = grad_diff[(iter - 1) % L];
      const SingleWordDictionary& cur_step = step[(iter - 1) % L];

      for (uint i = 0; i < cur_grad_diff.size(); i++) {

        negate(cur_grad_diff[i]);       //was set to the previous gradient in the last loop iteration
        cur_grad_diff[i] += dict_param_grad[i];
        latest_inv_rho += cur_grad_diff[i] % cur_step[i];
      }

      //std::cerr << "new inv rho: " <<  latest_inv_rho << std::endl;

      rho[(iter - 1) % L] = 1.0 / latest_inv_rho;
      if (latest_inv_rho < 1e-305) {

        std::cerr << "RESTART, inv_rho = " << latest_inv_rho << std::endl;
        last_restart = iter;
      }
    }

    double initial_scale = 1.0 / sqrt(sqr_grad_norm);
    if (iter > last_restart) {

      const SingleWordDictionary& cur_grad_diff = grad_diff[(iter - 1) % L];
      double sqr_norm = 0.0;
      for (uint i = 0; i < cur_grad_diff.size(); i++)
        sqr_norm += cur_grad_diff[i].sqr_norm();

      initial_scale = latest_inv_rho / sqr_norm;
    }

    /*** compute search direction ***/

    search_direction = dict_param_grad;
    for (uint i = 0; i < search_direction.size(); i++)
      negate(search_direction[i]);

    /** 1 forward multiplications **/
    for (int i = iter - 1; i >= std::max < int >(last_restart, int (iter) - L); i--) {

      const SingleWordDictionary& cur_step = step[i % L];

      double cur_alpha = 0.0;
      for (uint k = 0; k < search_direction.size(); k++)
        cur_alpha += cur_step[k] % search_direction[k];

      cur_alpha *= rho[i % L];

      const SingleWordDictionary& cur_grad_diff = grad_diff[i % L];

      for (uint k = 0; k < search_direction.size(); k++)
        search_direction[k].add_vector_multiple(cur_grad_diff[k], -cur_alpha);

      alpha[i % L] = cur_alpha;
    }

    /** 2 apply initial matrix **/
    for (uint i = 0; i < search_direction.size(); i++)
      search_direction[i] *= initial_scale;

    /** 3 backward multiplications **/
    for (int i = std::max < int >(last_restart, int (iter) - L); i < int (iter); i++) {

      const SingleWordDictionary& cur_grad_diff = grad_diff[i % L];

      double beta = 0.0;
      for (uint k = 0; k < search_direction.size(); k++)
        beta += search_direction[k] % cur_grad_diff[k];

      beta *= rho[i % L];

      double gamma = alpha[i % L] - beta;

      const SingleWordDictionary& cur_step = step[i % L];

      for (uint k = 0; k < search_direction.size(); k++)
        search_direction[k].add_vector_multiple(cur_step[k], gamma);
    }

    /*** search for a good step size ***/
    double best_energy = 1e300;

    uint nInnerIter = 0;

    bool decreasing = true;
    double lambda = 1.0;
    double best_lambda = 1.0;

    while (best_energy > energy || decreasing) {

      nInnerIter++;

      if (best_energy <= 0.95 * energy)
        break;

      if (best_energy < 0.99 * energy && nInnerIter > 3)
        break;

      for (uint i = 0; i < options.nTargetWords_; i++) {

        double scale = 0.0;
        for (uint k = 0; k < dict[i].size(); k++) {
          hyp_dict[i][k] = param[i][k] + lambda * search_direction[i][k];
          hyp_dict[i][k] = hyp_dict[i][k] * hyp_dict[i][k];
          scale += hyp_dict[i][k];
        }

        for (uint k = 0; k < dict[i].size(); k++)
          hyp_dict[i][k] = std::max(ibm1_min_dict_entry, hyp_dict[i][k] / scale);
      }

      double new_energy = ibm1_energy(source, slookup, target, hyp_dict, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum);

      std::cerr << "new hyp: " << new_energy << ", previous: " << best_energy << std::endl;

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      lambda *= line_reduction_factor;
    }

    /*** update the point and some auxiliary variables ***/

    SingleWordDictionary& new_step = step[iter % L];
    SingleWordDictionary& new_grad_diff = grad_diff[iter % L];

    for (uint i = 0; i < dict.size(); i++) {

      new_grad_diff[i] = dict_param_grad[i];
      new_step[i] = search_direction[i];
      new_step[i] *= best_lambda;

      // for (uint k=0; k < dict[i].size(); k++) {

      //   new_step[i][k] = best_lambda * search_direction[i][k];
      // }

      param[i] += new_step[i];
      double scale = 0.0;
      for (uint k = 0; k < dict[i].size(); k++) {
        dict[i][k] = param[i][k] * param[i][k];
        scale += dict[i][k];
      }
      for (uint k = 0; k < dict[i].size(); k++) {
        dict[i][k] = std::max(ibm1_min_dict_entry, dict[i][k] / scale);
      }
    }

    energy = best_energy;

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm1_viterbi_alignment(source[s], cur_lookup, target[s], dict, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm1_postdec_alignment(source[s], cur_lookup, target[s], dict, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### IBM-1 Viterbi-AER after lbfgs-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Viterbi-fmeasure after lbfgs-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Viterbi-DAE/S after lbfgs-iteration #" << iter << ": " << nErrors << std::endl;
      std::cerr << "#### IBM-1 Postdec-AER after lbfgs-iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Postdec-fmeasure after lbfgs-iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Postdec-DAE/S after lbfgs-iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  } //end for iter
}


#ifdef HAS_CBC
void viterbi_move(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                  const Storage1D<Math1D::Vector<uint> >& target, uint first_sent, uint last_sent, double dict_regularity,
                  Storage1D<Math1D::Vector<AlignBaseType> >& viterbi_alignment, NamedStorage1D<Math1D::Vector<double> >& dcount)
{
  Storage1D<Math1D::Vector<double> > min_count = dcount;
  Storage1D<Math1D::Vector<double> > max_count = dcount;

  Storage1D<uint> min_tcount(dcount.size());
  Storage1D<uint> max_tcount(dcount.size());
  for (uint i = 0; i < dcount.size(); i++) {
    min_tcount[i] = min_count[i].sum();
    max_tcount[i] = min_tcount[i];
  }

  size_t nSentences = source.size();

  uint nVars = 0;
  uint nConstraints = 0;
  uint nEntries = 0;

  std::map<std::pair<uint,uint>,uint> word_pair_idx;
  uint next_idx = 0;

  std::map<uint,uint> target_idx;
  target_idx[0] = 0;
  uint next_tidx = 1;

  Math1D::Vector<uint> sent_var_base(last_sent - first_sent + 1);
  Math1D::Vector<uint> sent_con_base(last_sent - first_sent + 1);

  for (size_t s = first_sent; s < nSentences && s <= last_sent; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    uint curJ = cur_source.size();
    uint curI = cur_target.size();

    sent_var_base[s - first_sent] = nVars;
    sent_con_base[s - first_sent] = nConstraints;

    nVars += curJ * (curI + 1);
    nConstraints += curJ;
    nEntries += 4 * curJ * (curI + 1);

    for (uint i = 0; i <= curI; i++) {
      if (i > 0 && target_idx.find(cur_target[i - 1]) == target_idx.end()) {

        target_idx[cur_target[i - 1]] = next_tidx;
        next_tidx++;
      }

      for (uint j = 0; j < curJ; j++) {

        std::pair<uint,uint> p;
        if (i == 0) {
          p = std::make_pair(0, cur_source[j] - 1);
        }
        else {
          p = std::make_pair(cur_target[i - 1], slookup[s] (j, i - 1));
        }
        if (word_pair_idx.find(p) == word_pair_idx.end()) {
          word_pair_idx[p] = next_idx;
          next_idx++;
        }

        if (viterbi_alignment[s][j] == i) {
          if (i == 0) {
            min_count[0][cur_source[j] - 1]--;
            min_tcount[0]--;
          }
          else {
            min_count[cur_target[i - 1]][slookup[s] (j, i - 1)]--;
            min_tcount[cur_target[i - 1]]--;
          }
        }
        else {
          if (i == 0) {
            max_count[0][cur_source[j] - 1]++;
            max_tcount[0]++;
          }
          else {
            max_count[cur_target[i - 1]][slookup[s] (j, i - 1)]++;
            max_tcount[cur_target[i - 1]]++;
          }
        }
      }
    }
  }

  uint nAlignmentConstraints = nConstraints;

  uint wp_con_base = nConstraints;

  nConstraints += 2 * word_pair_idx.size();

  Math1D::Vector<uint> wp_var_base(next_idx);

  for (std::map<std::pair<uint,uint>,uint>::iterator it = word_pair_idx.begin(); it != word_pair_idx.end(); it++) {

    uint first = it->first.first;
    uint second = it->first.second;

    uint minc = min_count[first][second];
    uint maxc = max_count[first][second];

    assert(minc < maxc);

    wp_var_base[it->second] = nVars;

    nVars += (maxc - minc) + 1;
    nEntries += 2 * ((maxc - minc) + 1);
  }

  uint w_con_base = nConstraints;

  nConstraints += 2 * target_idx.size();
  Math1D::Vector<uint> w_var_base(next_tidx);

  for (std::map<uint,uint >::iterator it = target_idx.begin(); it != target_idx.end(); it++) {

    w_var_base[it->second] = nVars;

    uint minc = min_tcount[it->first];
    uint maxc = max_tcount[it->first];

    nVars += (maxc - minc) + 1;
    nEntries += 2 * ((maxc - minc) + 1);
  }

  Math1D::Vector<double> var_lb(nVars, 0.0);
  Math1D::Vector<double> var_ub(nVars, 1.0);
  Math1D::Vector<double> cost(nVars, 0.0);

  /*** code cost entries ***/
  for (std::map<std::pair<uint,uint>,uint>::iterator it = word_pair_idx.begin(); it != word_pair_idx.end(); it++) {

    uint first = it->first.first;
    uint second = it->first.second;

    uint minc = min_count[first][second];
    uint maxc = max_count[first][second];

    uint idx = it->second;

    for (uint v = 0; v <= maxc - minc; v++) {
      double cur_cost = 0.0;
      double c = v + minc;
      if (c > 0.0)
        cur_cost = -c * std::log(c);

      if (minc == 0 && v > 0)
        cur_cost += dict_regularity;

      cost[wp_var_base[idx] + v] = cur_cost;
    }
  }

  for (std::map<uint,uint>::iterator it = target_idx.begin(); it != target_idx.end(); it++) {

    uint idx = it->second;

    uint minc = min_tcount[it->first];
    uint maxc = max_tcount[it->first];

    for (uint v = 0; v <= maxc - minc; v++) {
      double cur_cost = 0.0;
      double c = v + minc;
      if (c > 0.0)
        cur_cost = c * std::log(c);

      cost[w_var_base[idx] + v] = cur_cost;
    }
  }

  SparseMatrixDescription<double> lp_descr(nEntries, nConstraints, nVars);

  Math1D::Vector<double> rhs_lower(nConstraints, 0.0);
  Math1D::Vector<double> rhs_upper(nConstraints, 0.0);

  for (uint c = 0; c < nAlignmentConstraints; c++) {
    rhs_lower[c] = 1.0;
    rhs_upper[c] = 1.0;
  }

  std::cerr << "A" << std::endl;

  /**** handle alignment variables ****/
  for (size_t s = first_sent; s < nSentences && s <= last_sent; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    uint curJ = cur_source.size();
    uint curI = cur_target.size();

    uint cur_con_base = sent_con_base[s - first_sent];
    uint cur_var_base = sent_var_base[s - first_sent];

    for (uint j = 0; j < curJ; j++) {
      for (uint i = 0; i <= curI; i++) {
        uint var_idx = cur_var_base + j * (curI + 1) + i;

        lp_descr.add_entry(cur_con_base + j, var_idx, 1.0);

        uint tw = (i == 0) ? 0 : cur_target[i - 1];
        lp_descr.add_entry(w_con_base + 2 * target_idx[tw], var_idx, -1.0);

        std::pair < uint, uint > p;
        if (i == 0) {
          p = std::make_pair(0, cur_source[j] - 1);
        }
        else {
          p = std::make_pair(cur_target[i - 1], slookup[s] (j, i - 1));
        }
        lp_descr.add_entry(wp_con_base + 2 * word_pair_idx[p], var_idx, -1.0);
      }
    }
  }

  std::cerr << "B" << std::endl;

  /**** now add count variables ***/
  for (std::map<std::pair<uint,uint>,uint >::iterator it = word_pair_idx.begin(); it != word_pair_idx.end(); it++) {

    uint first = it->first.first;
    uint second = it->first.second;

    uint minc = min_count[first][second];
    uint maxc = max_count[first][second];

    assert(minc < maxc);

    rhs_lower[wp_con_base + 2 * it->second + 1] = 1.0;
    rhs_upper[wp_con_base + 2 * it->second + 1] = 1.0;

    for (uint c = 0; c <= maxc - minc; c++) {

      if (c > 0)
        lp_descr.add_entry(wp_con_base + 2 * it->second, wp_var_base[it->second] + c, c);
      lp_descr.add_entry(wp_con_base + 2 * it->second + 1, wp_var_base[it->second] + c, 1.0);
    }
  }

  std::cerr << "C" << std::endl;

  for (std::map<uint,uint >::iterator it = target_idx.begin(); it != target_idx.end(); it++) {

    rhs_lower[w_con_base + 2 * it->second + 1] = 1.0;
    rhs_upper[w_con_base + 2 * it->second + 1] = 1.0;

    uint minc = min_tcount[it->first];
    uint maxc = max_tcount[it->first];

    for (uint c = 0; c <= maxc - minc; c++) {

      if (c > 0)
        lp_descr.add_entry(w_con_base + 2 * it->second, w_var_base[it->second] + c, c);
      lp_descr.add_entry(w_con_base + 2 * it->second + 1, w_var_base[it->second] + c, 1.0);
    }
  }

  std::cerr << "D" << std::endl;

  CoinPackedMatrix coinMatrix(false, (int*)lp_descr.row_indices(),
                              (int*)lp_descr.col_indices(), lp_descr.value(), lp_descr.nEntries());

  OsiClpSolverInterface clp_interface;
  clp_interface.setLogLevel(0);

  clp_interface.loadProblem(coinMatrix, var_lb.direct_access(), var_ub.direct_access(), cost.direct_access(),
                            rhs_lower.direct_access(), rhs_upper.direct_access());

  for (uint v = 0; v < nVars; v++)
    clp_interface.setInteger(v);

  CbcModel cbcobj(clp_interface);

  CglGomory gomory_cut;
  gomory_cut.setLimit(200);
  gomory_cut.setLimitAtRoot(200);
  gomory_cut.setAway(0.05);
  gomory_cut.setAwayAtRoot(0.05);
  cbcobj.addCutGenerator(&gomory_cut, 0, "Gomory Cut");

  /**** construct initial solution *****/
  Math1D::Vector<uint> init_wp_count(next_idx, 0);
  Math1D::Vector<uint> init_w_count(next_tidx, 0);
  Math1D::Vector<double>initial_solution(nVars, 0.0);

  for (size_t s = first_sent; s < nSentences && s <= last_sent; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    uint curJ = cur_source.size();
    uint curI = cur_target.size();

    for (uint j = 0; j < curJ; j++) {

      ushort aj = viterbi_alignment[s][j];

      initial_solution[sent_var_base[s - first_sent] + j * (curI + 1) + aj] = 1.0;

      std::pair < uint, uint > p;
      if (aj == 0) {
        p = std::make_pair(0, cur_source[j] - 1);
      }
      else {
        p = std::make_pair(cur_target[aj - 1], slookup[s] (j, aj - 1));
      }

      init_wp_count[word_pair_idx[p]]++;

      if (aj == 0) {
        init_w_count[0]++;
      }
      else {
        init_w_count[target_idx[cur_target[aj - 1]]]++;
      }
    }
  }

  for (uint k = 0; k < next_idx; k++)
    initial_solution[wp_var_base[k] + init_wp_count[k]] = 1.0;

  for (uint k = 0; k < next_tidx; k++)
    initial_solution[w_var_base[k] + init_w_count[k]] = 1.0;

  double start_energy = 0.0;
  for (uint v = 0; v < nVars; v++)
    start_energy += initial_solution[v] * cost[v];

  cbcobj.setBestSolution(initial_solution.direct_access(), nVars, start_energy, true);

  cbcobj.branchAndBound();

  std::cerr << "start energy: " << start_energy << std::endl;

  //TODO: process the result

}
#endif

void ibm1_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                           const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                           SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                           const IBM1Options& options, const Math1D::Vector<double>& xlogx_table)
{
  const uint nIter = options.nIterations_;

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  //prepare dictionary
  dict.resize(options.nTargetWords_);
  for (uint i = 0; i < options.nTargetWords_; i++) {

    const uint size = (i == 0) ? options.nSourceWords_ - 1 : wcooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double)size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  SingleLookupTable aux_lookup;

  const uint nSourceWords = options.nSourceWords_;

#if 1
  for (uint i = 1; i < options.nTargetWords_; i++) {
    dict[i].set_constant(0.0);
  }
  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    for (uint i = 0; i < curI; i++) {
      uint tidx = cur_target[i];
      for (uint j = 0; j < curJ; j++) {

        dict[tidx][cur_lookup(j, i)] += 1.0;
      }
    }
  }

  for (uint i = 1; i < options.nTargetWords_; i++) {
    dict[i] *= 1.0 / dict[i].sum();
  }
#endif

  //counts of words
  NamedStorage1D<Math1D::Vector<uint> > dcount(options.nTargetWords_, MAKENAME(dcount));

  for (uint i = 0; i < options.nTargetWords_; i++) {
    dcount[i].resize(dict[i].size());
    dcount[i].set_constant(0);
  }

  double energy_offset = 0.0;
  for (size_t s = 0; s < nSentences; s++) {

    const Math1D::Vector<uint>& cur_source = source[s];
    const Math1D::Vector<uint>& cur_target = target[s];

    viterbi_alignment[s].resize(cur_source.size());

    const uint nCurSourceWords = cur_source.size();
    const uint nCurTargetWords = cur_target.size();

    energy_offset += nCurSourceWords * std::log(nCurTargetWords + 1.0);
  }

  double last_energy = 1e300;

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM-1 Viterbi iteration #" << iter << std::endl;

    for (uint i = 0; i < options.nTargetWords_; i++) {
      dcount[i].set_constant(0);
    }

    double sum = 0.0;

    for (size_t s = 0; s < nSentences; s++) {

      const Math1D::Vector<uint>& cur_source = source[s];
      const Math1D::Vector<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      for (uint j = 0; j < curJ; j++) {
        const uint s_idx = cur_source[j];

        double min = 1e50;
        uint arg_min = MAX_UINT;

        if (iter == 1) {

          min = -std::log(dict[0][s_idx - 1]);
          arg_min = 0;

          for (uint i = 0; i < curI; i++) {

            double hyp = -std::log(dict[cur_target[i]][cur_lookup(j, i)]);

            //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;

            if (hyp < min) {
              min = hyp;
              arg_min = i + 1;
            }
          }
        }
        else {

          if (dict[0][s_idx - 1] == 0.0) {
            min = 1e20;
          }
          else {
            min = -std::log(dict[0][s_idx - 1]);
          }
          arg_min = 0;

          for (uint i = 0; i < curI; i++) {

            const uint ti = cur_target[i];
            double hyp;

            if (dict[ti][cur_lookup(j, i)] == 0.0) {
              hyp = 1e20;
            }
            else {
              hyp = -std::log(dict[ti][cur_lookup(j, i)]);
            }

            //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;

            if (hyp < min) {
              min = hyp;
              arg_min = i + 1;
            }
          }
        }

        //std::cerr << "arg_min: " << arg_min << std::endl;

        sum += min;

        cur_alignment[j] = arg_min;

        if (arg_min == 0)
          dcount[0][s_idx - 1]++;
        else
          dcount[cur_target[arg_min - 1]][cur_lookup(j, arg_min - 1)]++;
      }
    }

    //exit(1);

    sum += energy_offset;
    //std::cerr << "sum: " << sum << std::endl;

    /*** ICM phase ***/

    uint nSwitches = 0;

    Math1D::Vector<uint> dict_sum(dcount.size());
    for (uint k = 0; k < dcount.size(); k++)
      dict_sum[k] = dcount[k].sum();

    for (size_t s = 0; s < nSentences; s++) {

      if ((s % 12500) == 0)
        std::cerr << "s: " << s << std::endl;

      const Math1D::Vector<uint>& cur_source = source[s];
      const Math1D::Vector<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      for (uint j = 0; j < curJ; j++) {

        const ushort cur_aj = cur_alignment[j];
        ushort new_aj = cur_aj;

        const uint cur_target_word = (cur_aj == 0) ? 0 : cur_target[cur_aj - 1];
        const uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);
        const uint cur_dictsum = dict_sum[cur_target_word];
        Math1D::Vector<uint>& cur_dictcount = dcount[cur_target_word];

        //NOTE: IBM-1 has no change when the target words are identical
        double common_change = 0.0;
        if (cur_dictsum > 1) {
          //exploit that log(1) = 0
          common_change -= xlogx_table[cur_dictsum];
          common_change += xlogx_table[cur_dictsum - 1];
        }

        if (cur_dictcount[cur_idx] > 1) {
          //exploit that log(1) = 0
          common_change -= -xlogx_table[cur_dictcount[cur_idx]];
          common_change += -xlogx_table[cur_dictcount[cur_idx] - 1];
        }
        else
          common_change -= prior_weight[cur_target_word][cur_idx];

        double best_change = 1e300;

        for (uint i = 0; i <= curI; i++) {

          const uint new_target_word = (i == 0) ? 0 : cur_target[i - 1];

          //NOTE: IBM-1 scores don't change when the two words in question are identical
          if (cur_target_word != new_target_word) {

            const uint hyp_idx = (i == 0) ? cur_source[j] - 1 : cur_lookup(j, i - 1);

            double change = common_change;

            if (dict_sum[new_target_word] > 0) {
              //exploit that log(1) = 0
              change -= xlogx_table[dict_sum[new_target_word]];
              change += xlogx_table[dict_sum[new_target_word] + 1];
            }

            if (dcount[new_target_word][hyp_idx] > 0) {
              //exploit that log(1) = 0
              change -= -xlogx_table[dcount[new_target_word][hyp_idx]];
              change += -xlogx_table[dcount[new_target_word][hyp_idx] + 1];
            }
            else
              change += prior_weight[new_target_word][hyp_idx];

            assert(!isnan(change));

            if (change < best_change) {

              best_change = change;
              new_aj = i;
            }
          }
        }

        if (best_change < -1e-2 && new_aj != cur_aj) {

          nSwitches++;

          const uint new_target_word = (new_aj == 0) ? 0 : cur_target[new_aj - 1];
          const uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);
          Math1D::Vector<uint>& hyp_dictcount = dcount[new_target_word];
          const uint hyp_idx = (new_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, new_aj - 1);

          cur_alignment[j] = new_aj;
          cur_dictcount[cur_idx] -= 1;
          hyp_dictcount[hyp_idx] += 1;
          dict_sum[cur_target_word] -= 1;
          dict_sum[new_target_word] += 1;
        }
      }
    }

    std::cerr << nSwitches << " switches in ICM" << std::endl;

    // Math1D::Vector<uint> count_count(6,0);

    // for (uint i=0; i < options.nTargetWords_; i++) {
    //   for (uint k=0; k < dcount[i].size(); k++) {
    //     if (dcount[i][k] < count_count.size())
    //       count_count[dcount[i][k]]++;
    //   }
    // }

    // std::cerr << "count count (lower end): " << count_count << std::endl;

    /*** recompute the dictionary ***/
    double energy = energy_offset;

    double sum_sum = 0.0;

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //std::cerr << "i: " << i << std::endl;

      const double sum = dcount[i].sum();

      sum_sum += sum;

      if (sum > 1e-307) {

        energy += sum * std::log(sum);

        const double inv_sum = 1.0 / sum;
        assert(!isnan(inv_sum));

        for (uint k = 0; k < dcount[i].size(); k++) {
          dict[i][k] = std::max(ibm1_min_dict_entry, dcount[i][k] * inv_sum);

          if (dcount[i][k] > 0) {
            energy -= dcount[i][k] * std::log(dcount[i][k]);
            energy += prior_weight[i][k];
          }
        }
      }
      else {
        dict[i].set_constant(0.0);
        //std::cerr << "WARNING : did not update dictionary entries because sum is " << sum << std::endl;
      }
    }

    //std::cerr << "number of total alignments: " << sum_sum << std::endl;
    //std::cerr.precision(10);

    if (dict_weight_sum > 0.0) {
      for (uint i = 0; i < dcount.size(); i++)
        for (uint k = 0; k < dcount[i].size(); k++)
          if (dcount[i][k] > 0)
            energy += prior_weight[i][k];
    }

    //we need to divide as we are truly minimizing the perplexity WITHOUT division plus the l0-term
    energy /= nSentences;

    if (options.print_energy_) {
      std::cerr << "energy: " << energy << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        nContributors++;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment[s], cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment[s], cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment[s], cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM-1 Viterbi-AER after Viterbi-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Viterbi-fmeasure after Viterbi-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Viterbi-DAE/S after Viterbi-iteration #" << iter << ": " << nErrors << std::endl;

      if (nSwitches == 0 && fabs(last_energy - energy) < 1e-4) {
        std::cerr << "LOCAL MINIMUM => break." << std::endl;
        break;
      }

      last_energy = energy;
    }
  }

#if 0
  uint granularity = 10;
  for (uint k = 0; k < source.size(); k += granularity) {

    std::cerr << "++++++++++++++  ILP-move for k=" << k << std::endl;

    viterbi_move(source, slookup, target, k, k + (granularity - 1), dict_penalty, viterbi_alignment, dcount);
  }
#endif
}
