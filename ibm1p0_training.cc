/**** Written by Thomas Schoenemann as a private person, November 2019 ****/

#include "ibm1p0_training.hh"
#include "training_common.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"
#include "projection.hh"

double ibm1p0_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                         const SingleWordDictionary& dict, const double p0, const CooccuringWordsType& wcooc, uint nSourceWords)
{
  //std::cerr << "calculating IBM-1 perplexity" << std::endl;

  const double p1 = 1.0 - p0;
  assert(p0 >= 0.0 && p1 <= 1.0);

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

    const double w1 = p1 / curI;

    for (uint j = 0; j < curJ; j++) {
      double cur_sum = p0 * dict[0][cur_source[j] - 1];      // handles empty word

      for (uint i = 0; i < curI; i++) {
        cur_sum += w1 * dict[cur_target[i]][cur_lookup(j, i)];
      }

      sum -= std::log(cur_sum);
    }
  }

  return sum / nSentences;
}

double ibm1p0_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                     const SingleWordDictionary& dict, const double p0, const CooccuringWordsType& wcooc, uint nSourceWords,
                     const floatSingleWordDictionary& prior_weight, bool smoothed_l0, double l0_beta, double dict_weight_sum)
{
  double energy = 0.0;

  if (dict_weight_sum != 0.0) {

    assert(!smoothed_l0 || l0_beta > 0.0);

    energy = dict_reg_term(dict, prior_weight, l0_beta);

    // for (uint i = 0; i < dict.size(); i++) {

    // const Math1D::Vector<double>& cur_dict = dict[i];
    // const Math1D::Vector<float>& cur_prior = prior_weight[i];

    // const uint size = cur_dict.size();

    // if (smoothed_l0) {
    // for (uint k = 0; k < size; k++)
    // energy += cur_prior[k] * prob_penalty(cur_dict[k], l0_beta);
    // }
    // else {
    // for (uint k = 0; k < size; k++)
    // energy += cur_prior[k] * cur_dict[k];
    // }
    // }
  }

  energy += ibm1p0_perplexity(source, slookup, target, dict, p0, wcooc, nSourceWords);

  return energy;
}

void train_ibm1p0(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                  const CooccuringWordsType& wcooc, SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                  const IBM1Options& options)
{
  const double p0 = options.p0_;
  const double p1 = 1.0 - p0;

  assert(p0 >= 0.0 && p1 <= 1.0);

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

  if (!options.uniform_dict_init_) {
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
  }

  //fractional counts used for EM-iterations
  NamedStorage1D<Math1D::Vector<double> > fcount(options.nTargetWords_, MAKENAME(fcount));
  for (uint i = 0; i < options.nTargetWords_; i++) {
    fcount[i].resize(dict[i].size());
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM-1-p0 EM-iteration #" << iter << std::endl;

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

      const double w1 = p1 / curI;

      for (uint j = 0; j < curJ; j++) {
        const uint s_idx = cur_source[j];

        double coeff = p0 * dict[0][s_idx - 1];      // entry for empty word (the emtpy word is not listed, hence s_idx-1)
        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          coeff += w1 * dict[t_idx][cur_lookup(j, i)];
        }
        coeff = 1.0 / coeff;

        assert(!isnan(coeff));

        fcount[0][s_idx - 1] += coeff * p0 * dict[0][s_idx - 1];
        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          const uint k = cur_lookup(j, i);
          fcount[t_idx][k] += coeff * w1 * dict[t_idx][k];
        }
      }

    } //loop over sentences finished

    std::cerr << "updating dict from counts" << std::endl;

    /*** update dict from counts ***/

    update_dict_from_counts(fcount, prior_weight, nSentences, dict_weight_sum, smoothed_l0, l0_beta, options.dict_m_step_iter_, dict,
                            ibm1_min_dict_entry, options.unconstrained_m_step_, options.gd_stepsize_);

    if (options.print_energy_) {
      std::cerr << "IBM-1-p0 energy after iteration #" << iter << ": "
                << ibm1p0_energy(source, slookup, target, dict, p0, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum) << std::endl;
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
        const uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm1p0_viterbi_alignment(source[s], cur_lookup, target[s], dict, p0, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm1p0_postdec_alignment(source[s], cur_lookup, target[s], dict, p0, postdec_alignment);

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

      std::cerr << "#### IBM-1-p0 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1-p0 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1-p0 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;
      std::cerr << "#### IBM-1-p0 Postdec-AER after iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### IBM-1-p0 Postdec-fmeasure after iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### IBM-1-p0 Postdec-DAE/S after iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  } //end for iter
}


void train_ibm1p0_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                 const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                 SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                 const IBM1Options& options)
{
  const double p0 = options.p0_;
  const double p1 = 1.0 - p0;

  assert(p0 >= 0.0 && p1 <= 1.0);

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

  if (!options.uniform_dict_init_) {
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
  }

  double energy = ibm1p0_energy(source, slookup, target, dict, p0, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum);

  std::cerr << "initial energy: " << energy << std::endl;

  SingleWordDictionary new_dict(options.nTargetWords_, MAKENAME(new_dict));
  SingleWordDictionary hyp_dict(options.nTargetWords_, MAKENAME(hyp_dict));

  for (uint i = 0; i < options.nTargetWords_; i++) {

    const uint size = dict[i].size();
    new_dict[i].resize_dirty(size);
    hyp_dict[i].resize_dirty(size);
  }

  Math1D::Vector<double> new_slack_vector(options.nTargetWords_, 0.0);

  double alpha = options.gd_stepsize_;

  double line_reduction_factor = 0.1;

  uint nSuccessiveReductions = 0;

  double best_lower_bound = -1e300;

  SingleWordDictionary dict_grad(options.nTargetWords_, MAKENAME(dict_grad));
  SingleWordDictionary old_dict = dict;

  for (uint i = 0; i < options.nTargetWords_; i++) {

    const uint size = dict[i].size();
    dict_grad[i].resize_dirty(size);
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM-1-p0 gradient descent iteration #" << iter << std::endl;

    /**** calcuate gradients ****/

    for (uint i = 0; i < options.nTargetWords_; i++) {
      dict_grad[i].set_constant(0.0);
    }

    for (size_t s = 0; s < nSentences; s++) {
      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const double w1 = p1 / curI;

      for (uint j = 0; j < curJ; j++) {

        uint s_idx = cur_source[j];

        double sum = p0 * dict[0][s_idx - 1];

        for (uint i = 0; i < curI; i++)
          sum += w1 * dict[cur_target[i]][cur_lookup(j, i)];

        const double cur_grad = -1.0 / sum;

        dict_grad[0][s_idx - 1] += cur_grad;
        for (uint i = 0; i < curI; i++)
          dict_grad[cur_target[i]][cur_lookup(j, i)] += cur_grad;
      }
    }

    for (uint i = 0; i < options.nTargetWords_; i++) {
      dict_grad[i] *= 1.0 / nSentences;
    }

    for (uint i = 0; i < options.nTargetWords_; i++) {
      const uint size = dict[i].size();

      for (uint k = 0; k < size; k++) {
        if (smoothed_l0)
          dict_grad[i][k] += prior_weight[i][k] * prob_pen_prime(dict[i][k], l0_beta);
        else
          dict_grad[i][k] += prior_weight[i][k];
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

    /**** move in gradient direction ****/

    double real_alpha = alpha;
#if 1
    double sqr_grad_norm = 0.0;
    for (uint i = 0; i < options.nTargetWords_; i++)
      sqr_grad_norm += dict_grad[i].sqr_norm();
    real_alpha /= sqrt(sqr_grad_norm);
#endif

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //for (uint k = 0; k < dict[i].size(); k++)
      //  new_dict[i][k] = dict[i][k] - real_alpha * dict_grad[i][k];
      Math1D::go_in_neg_direction(new_dict[i], dict[i], dict_grad[i], real_alpha);
    }

    if (dict_weight_sum != 0.0)
      new_slack_vector = slack_vector;

    /**** reproject on the simplices [Michelot 1986]****/
    for (uint i = 0; i < options.nTargetWords_; i++) {

      const uint nCurWords = new_dict[i].size();

      if (dict_weight_sum != 0)
        projection_on_simplex_with_slack(new_dict[i].direct_access(), slack_vector[i], nCurWords, ibm1_min_dict_entry);
      else
        projection_on_simplex(new_dict[i].direct_access(), nCurWords, ibm1_min_dict_entry);
    }

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

        Math1D::assign_weighted_combination(hyp_dict[i], neg_lambda, dict[i], lambda, new_dict[i]);
      }

      double new_energy = ibm1p0_energy(source, slookup, target, hyp_dict, p0, wcooc, nSourceWords, prior_weight, smoothed_l0, l0_beta, dict_weight_sum);

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

    energy = hyp_energy;

    if (options.print_energy_)
      std::cerr << "IBM-1-p0 energy after gd-iteration #" << iter << ": " << energy << std::endl;

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
        compute_ibm1p0_viterbi_alignment(source[s], cur_lookup, target[s], dict, p0, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm1p0_postdec_alignment(source[s], cur_lookup, target[s], dict, p0, postdec_alignment);

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

      std::cerr << "#### IBM-1-p0 Viterbi-AER after gd-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1-p0 Viterbi-fmeasure after gd-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1-p0 Viterbi-DAE/S after gd-iteration #" << iter << ": " << nErrors << std::endl;
      std::cerr << "#### IBM-1-p0 Postdec-AER after gd-iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### IBM-1-p0 Postdec-fmeasure after gd-iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### IBM-1-p0 Postdec-DAE/S after gd-iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  } // end for iter
}


void ibm1p0_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                             const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                             SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                             const IBM1Options& options, const Math1D::Vector<double>& xlogx_table)
{
  const double p0 = options.p0_;
  const double p1 = 1.0 - p0;

  assert(p0 >= 0.0 && p1 <= 1.0);

  const uint nIter = options.nIterations_;

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  //prepare dictionary
  dict.resize(options.nTargetWords_);
  for (uint i = 0; i < options.nTargetWords_; i++) {
    const uint size = (i == 0) ? options.nSourceWords_ - 1 : wcooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double)size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

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

    std::cerr << "starting IBM-1-p0 Viterbi iteration #" << iter << std::endl;

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

      const double w1 = p1 / curI;

      for (uint j = 0; j < curJ; j++) {
        const uint s_idx = cur_source[j];

        double cur_max = p0 * dict[0][s_idx - 1];
        uint arg_max = 0;

        for (uint i = 0; i < curI; i++) {
          const double hyp = w1 * dict[cur_target[i]][cur_lookup(j, i)];
          if (hyp > cur_max) {
            cur_max = hyp;
            arg_max = i+1;
          }
        }

        cur_alignment[j] = arg_max;
        if (arg_max == 0)
          dcount[0][s_idx-1]++;
        else
          dcount[cur_target[arg_max-1]][cur_lookup(j, arg_max-1)]++;
        sum -= std::log(cur_max);
      }
    }

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

      const double w1 = p1 / curI;

      for (uint j = 0; j < curJ; j++) {
        //std::cerr << "j: " << j << std::endl;

        const uint s_idx = cur_source[j];

        const ushort cur_aj = cur_alignment[j];
        ushort new_aj = cur_aj;

        const uint cur_target_word = (cur_aj == 0) ? 0 : cur_target[cur_aj - 1];
        const uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);
        const uint cur_dictsum = dict_sum[cur_target_word];
        Math1D::Vector<uint>& cur_dictcount = dcount[cur_target_word];

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

        if (cur_aj > 0)
          common_change -= -std::log(w1);
        else
          common_change -= -std::log(p0);

        double best_change = 1e300;

        for (uint i = 0; i <= curI; i++) {
          //std::cerr << "i: " << i << std::endl;

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

            if (i > 0)
              change += -std::log(w1);
            else
              change += -std::log(p0);

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
          const uint cur_idx = (cur_aj == 0) ? s_idx - 1 : cur_lookup(j, cur_aj - 1);
          Math1D::Vector<uint>& hyp_dictcount = dcount[new_target_word];
          const uint hyp_idx = (new_aj == 0) ? s_idx - 1 : cur_lookup(j, new_aj - 1);

          assert(cur_dictcount[cur_idx] > 0);
          assert(dict_sum[cur_target_word] > 0);

          cur_alignment[j] = new_aj;
          cur_dictcount[cur_idx] -= 1;
          hyp_dictcount[hyp_idx] += 1;
          dict_sum[cur_target_word] -= 1;
          dict_sum[new_target_word] += 1;
        }
      }
    }

    std::cerr << nSwitches << " switches in ICM" << std::endl;

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

    if (dict_weight_sum != 0.0) {
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

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment[s], cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment[s], cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment[s], cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM-1-p0 Viterbi-AER after Viterbi-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1-p0 Viterbi-fmeasure after Viterbi-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1-p0 Viterbi-DAE/S after Viterbi-iteration #" << iter << ": " << nErrors << std::endl;

      if (nSwitches == 0 && fabs(last_energy - energy) < 1e-4) {
        std::cerr << "LOCAL MINIMUM => break." << std::endl;
        break;
      }

      last_energy = energy;
    }
  }
}

