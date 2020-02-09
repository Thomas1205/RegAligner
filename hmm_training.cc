/*** written by Thomas Schoenemann as a private person without employment, October 2009
 *** later as an employee of Lund University, 2010 - Mar. 2011
 *** later as a private person, and finally at the University of Düsseldorf, Germany, January - September 2012,
 *** and since as a private person ***/

#include "hmm_training.hh"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "matrix.hh"

#include "training_common.hh"
#include "ibm1_training.hh"
#include "hmm_forward_backward.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"

#include "projection.hh"
#include "stl_out.hh"
#include "storage_util.hh"

HmmOptions::HmmOptions(uint nSourceWords, uint nTargetWords, const ReducedIBM2ClassAlignmentModel& ibm2_alignment_model,
                       const Math1D::Vector<WordClassType>& ibm2_sclass, RefAlignmentStructure& sure_ref_alignments,
                       RefAlignmentStructure& possible_ref_alignments):
  nIterations_(5), init_type_(HmmInitPar), align_type_(HmmAlignProbReducedpar), redpar_limit_(5), start_empty_word_(false), smoothed_l0_(false),
  deficient_(false), fix_p0_(false), l0_beta_(1.0), print_energy_(true), nSourceWords_(nSourceWords), nTargetWords_(nTargetWords),
  init_m_step_iter_(1000), align_m_step_iter_(1000), dict_m_step_iter_(45), transfer_mode_(TransferNo), msolve_mode_(MSSolvePGD),
  ibm2_alignment_model_(ibm2_alignment_model), ibm2_sclass_(ibm2_sclass), sure_ref_alignments_(sure_ref_alignments),
  possible_ref_alignments_(possible_ref_alignments)
{
}

HmmWrapper::HmmWrapper(const FullHMMAlignmentModel& align_model, const InitialAlignmentProbability& initial_prob,
                       const HmmOptions& hmm_options)
  : align_model_(align_model), initial_prob_(initial_prob), hmm_options_(hmm_options)
{
}

long double hmm_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup, const Storage1D<uint>& target,
                               const SingleWordDictionary& dict, const FullHMMAlignmentModel& align_model,
                               const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                               bool with_dict)
{
  const uint I = target.size();
  const uint J = source.size();

  const Math2D::Matrix<double>& cur_align_model = align_model[I - 1];
  const Math1D::Vector<double>& cur_initial_prob = initial_prob[I - 1];

  //std::cerr << "J: " << J << ", I: " << I << std::endl;
  //std::cerr << "alignment: " << alignment << std::endl;

  assert(J == alignment.size());

  long double prob = (alignment[0] == 2 * I) ? cur_initial_prob[I] : cur_initial_prob[alignment[0]];

  if (with_dict) {
    for (uint j = 0; j < J; j++) {
      const uint aj = alignment[j];
      if (aj < I)
        prob *= dict[target[aj]][slookup(j, aj)];
      else
        prob *= dict[0][source[j] - 1];
    }
  }

  for (uint j = 1; j < alignment.size(); j++) {
    uint prev_aj = alignment[j - 1];
    if (prev_aj >= I)
      prev_aj -= I;

    const uint aj = alignment[j];
    if (aj >= I)
      assert(aj == prev_aj + I);

    if (prev_aj == I)
      prob *= cur_initial_prob[std::min(I, aj)];
    else
      prob *= cur_align_model(std::min(I, aj), prev_aj);
  }

  return prob;
}

void external2internal_hmm_alignment(const Storage1D<AlignBaseType>& ext_alignment, uint curI, const HmmOptions& options,
                                     Storage1D<AlignBaseType>& int_alignment)
{
  int_alignment.resize(ext_alignment.size());

  int jj = -1;

  for (uint j = 0; j < ext_alignment.size(); j++) {

    uint ext_aj = ext_alignment[j];

    if (ext_aj > 0) {
      int_alignment[j] = ext_aj - 1;
      jj = j;
    }
    else {

      if (jj >= 0) {
        int_alignment[j] = ext_alignment[jj] - 1 + curI;
      }
      else
        int_alignment[j] = (options.start_empty_word_) ? 2 * curI : curI;
    }
  }
}

double extended_hmm_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                               const Storage1D<Math1D::Vector<uint> >& target, const FullHMMAlignmentModel& align_model,
                               const InitialAlignmentProbability& initial_prob,
                               const SingleWordDictionary& dict, const CooccuringWordsType& wcooc,
                               uint nSourceWords, const HmmOptions& options)
{
  HmmAlignProbType align_type = options.align_type_;
  bool start_empty_word = options.start_empty_word_;

  //std::cerr << "start empty word: " << start_empty_word << std::endl;

  double sum = 0.0;

  const size_t nSentences = target.size();

  SingleLookupTable aux_lookup;

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    const uint curI = cur_target.size();

    const Math2D::Matrix<double>& cur_align_model = align_model[curI - 1];

    sum -= calculate_hmm_forward_log_sum(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                                         initial_prob[curI - 1], align_type, start_empty_word, options.redpar_limit_);
  }

  return sum / nSentences;
}

double extended_hmm_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                           const FullHMMAlignmentModel& align_model, const InitialAlignmentProbability& initial_prob, const SingleWordDictionary& dict,
                           const CooccuringWordsType& wcooc, uint nSourceWords, const floatSingleWordDictionary& prior_weight,
                           const HmmOptions& options, double dict_weight_sum)
{
  double energy = 0.0;

  if (dict_weight_sum != 0.0) {

    for (uint i = 0; i < dict.size(); i++) {

      const Math1D::Vector<double>& cur_dict = dict[i];
      const Math1D::Vector<float>& cur_prior = prior_weight[i];

      const uint size = cur_dict.size();

      if (options.smoothed_l0_) {
        for (uint k = 0; k < size; k++)
          energy += cur_prior[k] * prob_penalty(cur_dict[k], options.l0_beta_);
      }
      else {
        for (uint k = 0; k < size; k++)
          energy += cur_prior[k] * cur_dict[k];
      }
    }
  }

  //std::cerr << "before adding perplexity: " << energy << std::endl;

  energy += extended_hmm_perplexity(source, slookup, target, align_model, initial_prob, dict, wcooc, nSourceWords, options);

  return energy;
}

double ehmm_m_step_energy(const FullHMMAlignmentModel& facount, const Math1D::Vector<double>& dist_params,
                          uint zero_offset, double grouping_param, int redpar_limit)
{
  double energy = 0.0;

  //std::cerr << "grouping_param: " << grouping_param << std::endl;

  for (uint I = 1; I <= facount.size(); I++) {

    if (facount[I - 1].size() > 0) {

      for (int i = 0; i < (int)I; i++) {

        double non_zero_sum = 0.0;

        if (grouping_param < 0.0) {

          for (uint ii = 0; ii < I; ii++)
            non_zero_sum += dist_params[zero_offset + ii - i];

          for (int ii = 0; ii < (int)I; ii++) {

            const double cur_count = facount[I - 1] (ii, i);

            //NOTE: division by non_zero_sum gives a constant due to log laws
            energy -= cur_count * std::log(dist_params[zero_offset + ii - i] / non_zero_sum);
          }
        }
        else {

          double grouping_norm = std::max(0, i - redpar_limit);
          grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

          //double check = 0.0;

          for (int ii = 0; ii < (int)I; ii++) {
            if (abs(ii - i) <= redpar_limit)
              non_zero_sum += dist_params[zero_offset + ii - i];
            else {
              non_zero_sum += grouping_param / grouping_norm;
              //check ++;
            }
          }

          //assert(check == grouping_norm);

          for (int ii = 0; ii < (int)I; ii++) {

            double cur_count = facount[I - 1] (ii, i);
            double cur_param;

            if (abs(ii - i) > redpar_limit)
              cur_param = grouping_param / grouping_norm;
            else
              cur_param = dist_params[zero_offset + ii - i];

            //NOTE: division by non_zero_sum gives a constant due to log laws
            //   same for division by grouping_norm
            energy -= cur_count * std::log(cur_param / non_zero_sum);
          }
        }
      }
    }
    //std::cerr << "intermediate energy: " << energy << std::endl;
  }

  return energy;
}

//new compact variant
double ehmm_m_step_energy(const Math1D::Vector<double>& singleton_count, double grouping_count,
                          const Math2D::Matrix<double>& span_count, const Math1D::Vector<double>& dist_params,
                          uint zero_offset, double grouping_param, int redpar_limit)
{
  //NOTE: we could exploit here that span_count will only be nonzero if zero_offset lies in the span

  double energy = 0.0;

  if (grouping_param < 0.0) {

    //singleton terms
    for (uint d = 0; d < singleton_count.size(); d++)
      energy -= singleton_count[d] * std::log(std::max(hmm_min_param_entry, dist_params[d]));

    //normalization terms
    Math1D::Vector<double> init_sum(zero_offset + 1);
    init_sum[zero_offset] = 0.0;
    init_sum[zero_offset - 1] = dist_params[zero_offset - 1];
    for (int s = zero_offset - 2; s >= 0; s--)
      init_sum[s] = init_sum[s + 1] + dist_params[s];

    for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

      double param_sum = init_sum[span_start];  //dist_params.range_sum(span_start,zero_offset);
      for (uint span_end = zero_offset;
           span_end < zero_offset + span_count.yDim(); span_end++) {

        param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

        const double cur_count = span_count(span_start, span_end - zero_offset);
        if (cur_count != 0.0)
          energy += cur_count * std::log(param_sum);
      }
    }

  }
  else {

    uint first_diff = zero_offset - redpar_limit;
    uint last_diff = zero_offset + redpar_limit;

    for (uint d = first_diff; d <= last_diff; d++)
      energy -= singleton_count[d] * std::log(std::max(hmm_min_param_entry, dist_params[d]));

    //NOTE: because we do not divide grouping_param by the various numbers of affected positions
    //   this energy will differ from the inefficient version by a constant
    energy -= grouping_count * std::log(std::max(hmm_min_param_entry, grouping_param));

    //normalization terms
    for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

      for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

        //there should be plenty of room for speed-ups here

        const double cur_count = span_count(span_start, span_end - zero_offset);
        if (cur_count != 0.0) {

          double param_sum = 0.0;
          if (span_start < first_diff || span_end > last_diff)
            param_sum = grouping_param;

          for (uint d = std::max(first_diff, span_start);
               d <= std::min(span_end, last_diff); d++)
            param_sum += std::max(hmm_min_param_entry, dist_params[d]);

          energy += cur_count * std::log(param_sum);
        }
      }
    }
  }

  return energy;
}

void noncompact_ehmm_m_step(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset, uint nIter,
                            double& grouping_param, bool deficient, int redpar_limit)
{
  std::cerr.precision(8);

  //std::cerr << "init params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;

  bool norm_constraint = true;

  if (grouping_param < 0.0) {
    projection_on_simplex(dist_params.direct_access(), dist_params.size(), hmm_min_param_entry);
  }
  else {
    projection_on_simplex_with_slack(dist_params.direct_access() + zero_offset - redpar_limit, grouping_param, 2 * redpar_limit + 1, hmm_min_param_entry);
    grouping_param = std::max(hmm_min_param_entry, grouping_param);
  }

  //std::cerr << "init params after projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param after projection: " << grouping_param << std::endl;

  Math1D::Vector<double> dist_grad = dist_params;
  Math1D::Vector<double> new_dist_params = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  double grouping_grad = 0.0;
  double new_grouping_param = grouping_param;
  double hyp_grouping_param = grouping_param;

  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(facount, dist_params, zero_offset, grouping_param, redpar_limit);

  double line_reduction_factor = 0.5;

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  //test if normalized counts give a better starting point
  {

    Math1D::Vector<double> dist_count(dist_params.size(), 0.0);
    double dist_grouping_count = (grouping_param < 0.0) ? -1.0 : 0.0;

    for (uint I = 1; I <= facount.size(); I++) {

      if (facount[I - 1].xDim() != 0) {

        for (int i = 0; i < (int)I; i++) {

          for (int ii = 0; ii < (int)I; ii++) {
            if (grouping_param < 0.0 || abs(ii - i) <= redpar_limit)
              dist_count[zero_offset + ii - i] += facount[I - 1](ii, i);
            else {
              //don't divide by grouping norm, the deficient problem doesn't need it:
              //  due to log laws we get additive constants
              dist_grouping_count += facount[I - 1](ii, i);
            }
          }
        }
      }
    }

    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += dist_count[zero_offset + k];
      norm += dist_grouping_count;

      if (norm > 1e-305) {

        for (uint k = 0; k < dist_count.size(); k++)
          dist_count[k] = std::max(hmm_min_param_entry, dist_count[k] / norm);

        dist_grouping_count = std::max(hmm_min_param_entry, dist_grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(facount, dist_count, zero_offset, dist_grouping_count, redpar_limit);

        //std::cerr << "hyp energy: " << hyp_energy << std::endl;

        if (hyp_energy < energy) {

          dist_params = dist_count;
          grouping_param = dist_grouping_count;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = dist_count.sum();

      if (sum > 1e-305) {
        for (uint k = 0; k < dist_count.size(); k++)
          dist_count[k] = std::max(hmm_min_param_entry, dist_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(facount, dist_count, zero_offset, grouping_param, redpar_limit);

        if (!deficient)
          std::cerr << "hyp energy: " << hyp_energy << std::endl;

        if (hyp_energy < energy) {
          dist_params = dist_count;
          energy = hyp_energy;
        }
      }

    }
  }

  if (deficient)
    return;

  std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  //double alpha  = 0.0001;
  //double alpha  = 0.001;
  double alpha = 100.0;

  for (uint iter = 1; iter <= nIter; iter++) {

    if ((iter % 5) == 0)
      std::cerr << "m-step gd-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;

    //calculate gradient
    for (uint I = 1; I <= facount.size(); I++) {

      if (facount[I - 1].size() > 0) {

        for (int i = 0; i < (int)I; i++) {

          double grouping_norm = std::max(0, i - redpar_limit);
          grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

          double non_zero_sum = 0.0;
          for (int ii = 0; ii < (int)I; ii++) {
            if (grouping_param < 0.0 || abs(i - ii) <= redpar_limit)
              non_zero_sum += dist_params[zero_offset + ii - i];
            else
              non_zero_sum += grouping_param / grouping_norm;
          }
          //if (grouping_param >= 0.0 && grouping_norm > 0.0)
          //  non_zero_sum += grouping_param;

          double count_sum = 0.0;
          for (int ii = 0; ii < (int)I; ii++) {
            count_sum += facount[I - 1] (ii, i);
          }

          for (int ii = 0; ii < (int)I; ii++) {
            double cur_param = dist_params[zero_offset + ii - i];

            double cur_count = facount[I - 1] (ii, i);

            if (grouping_param < 0.0) {
              dist_grad[zero_offset + ii - i] -= cur_count / cur_param;
            }
            else {

              if (abs(ii - i) > redpar_limit) {
                //NOTE: -std::log( param / norm) = -std::log(param) + std::log(norm)
                // => grouping_norm does NOT enter here
                grouping_grad -= cur_count / grouping_param;
              }
              else {
                dist_grad[zero_offset + ii - i] -= cur_count / cur_param;
              }
            }
          }

          for (int ii = 0; ii < (int)I; ii++) {
            if (grouping_param < 0.0 || abs(ii - i) <= redpar_limit)
              dist_grad[zero_offset + ii - i] += count_sum / non_zero_sum;
            // else
            //   m_grouping_grad += count_sum / (non_zero_sum * grouping_norm);
          }

          if (grouping_param >= 0.0 && grouping_norm > 0.0)
            grouping_grad += count_sum / non_zero_sum;
        }
      }
    }

    //go in gradient direction

    double real_alpha = alpha;

    //TRIAL
    // double sqr_grad_norm  = 0.0;
    // for (uint k=0; k < dist_params.size(); k++)
    //   sqr_grad_norm += dist_grad.direct_access(k) * dist_grad.direct_access(k);
    // sqr_grad_norm += grouping_grad * grouping_grad;

    // real_alpha /= sqrt(sqr_grad_norm);
    //END_TRIAL

    for (uint k = start_idx; k <= end_idx; k++)
      new_dist_params.direct_access(k) = dist_params.direct_access(k) - real_alpha * dist_grad.direct_access(k);

    new_grouping_param = grouping_param - real_alpha * grouping_grad;

    //std::cerr << "new params before projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param before projection: " << new_grouping_param << std::endl;

    for (uint k = start_idx; k <= end_idx; k++) {
      if (new_dist_params[k] >= 1e75)
        new_dist_params[k] = 9e74;
      else if (new_dist_params[k] <= -1e75)
        new_dist_params[k] = -9e74;
    }
    if (new_grouping_param >= 1e75)
      new_grouping_param = 9e74;
    else if (new_grouping_param <= -1e75)
      new_grouping_param = -9e74;

    if (norm_constraint) {
      // reproject
      if (grouping_param < 0.0) {
        projection_on_simplex(new_dist_params.direct_access(), dist_params.size(), hmm_min_param_entry);
      }
      else {
        projection_on_simplex_with_slack(new_dist_params.direct_access() + start_idx, new_grouping_param, 2 * redpar_limit + 1, hmm_min_param_entry);
        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
      }
    }
    else {
      //projection on the positive orthant, followed by renormalization
      //(justified by scale invariance with positive scaling factors)
      // may be faster than the simplex projection

      uint nNeg = 0; //DEBUG

      double sum = 0.0;
      for (uint k = start_idx; k <= end_idx; k++) {

        //DEBUG
        if (new_dist_params[k] < 0.0)
          nNeg++;
        //END_DEBUG

        new_dist_params[k] = std::max(hmm_min_param_entry, new_dist_params[k]);
        sum += new_dist_params[k];
      }
      if (grouping_param >= 0.0) {

        //DEBUG
        if (new_grouping_param < 0.0)
          nNeg++;
        //END_DEBUG

        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
        sum += new_grouping_param;
      }
      //DEBUG
      //std::cerr << "sum: " << sum << ", " << nNeg << " were negative" << std::endl;
      //END_DEBUG

      //projection done, now renormalize to keep the probability constraint
      double inv_sum = 1.0 / sum;
      for (uint k = start_idx; k <= end_idx; k++) {
        new_dist_params[k] = std::max(hmm_min_param_entry,inv_sum * new_dist_params[k]);
      }
      new_grouping_param = std::max(hmm_min_param_entry,inv_sum * new_grouping_param);
    }

    //std::cerr << "new params after projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param after projection: " << new_grouping_param << std::endl;

    //find step-size

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint k = start_idx; k <= end_idx; k++)
        hyp_dist_params.direct_access(k) = std::max(hmm_min_param_entry, lambda * new_dist_params.direct_access(k) + neg_lambda * dist_params.direct_access(k));

      if (grouping_param >= 0.0)
        hyp_grouping_param = std::max(hmm_min_param_entry, lambda * new_grouping_param + neg_lambda * grouping_param);

      const double new_energy = ehmm_m_step_energy(facount, hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "hyp energy: " << new_energy << std::endl;
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    // if (nIter > 4)
    //   alpha *= 1.5;

    //DEBUG
    // if (best_lambda == 1.0)
    //   std::cerr << "!!!TAKING FULL STEP!!!" << std::endl;
    //END_DEBUG

    if (nIter > 15 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = start_idx; k <= end_idx; k++)
      dist_params.direct_access(k) = std::max(hmm_min_param_entry, neg_best_lambda * dist_params.direct_access(k) +
                                              best_lambda * new_dist_params.direct_access(k));

    if (grouping_param >= 0.0)
      grouping_param = std::max(hmm_min_param_entry, best_lambda * new_grouping_param + neg_best_lambda * grouping_param);

    //std::cerr << "updated params: " << dist_params << std::endl;
    //std::cerr << "updated grouping param: " << grouping_param << std::endl;
  }
}

void ehmm_m_step(const FullHMMAlignmentModelNoClasses& facount, Math1D::Vector<double>& dist_params, uint zero_offset,
                 uint nIter, double& grouping_param, bool deficient, int redpar_limit, ProjectionMode projection_mode)
{
  bool norm_constraint = true;

  // 1. collect compact counts from facount

  Math1D::Vector<double> singleton_count(dist_params.size(), 0.0);
  double grouping_count = 0.0;
  Math2D::Matrix<double> span_count(zero_offset + 1, dist_params.size() - zero_offset, 0.0);

  //const uint maxI = dist_params.size() / 2 + 1;

  for (int I = 1; I <= int (facount.size()); I++) {

    //assert(I <= maxI);

    if (facount[I - 1].xDim() != 0) {

      for (int i = 0; i < I; i++) {

        double count_sum = 0.0;
        for (int i_next = 0; i_next < I; i_next++) {

          const double cur_count = facount[I - 1](i_next, i);
          if (grouping_param < 0.0 || abs(i_next - i) <= redpar_limit) {
            singleton_count[zero_offset + i_next - i] += cur_count;
            count_sum += cur_count;
          }
          else {
            double grouping_norm = std::max(0, i - redpar_limit);
            grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));
            grouping_count += cur_count / grouping_norm;
            count_sum += cur_count / grouping_norm;
          }
        }
        span_count(zero_offset - i, I - 1 - i) += count_sum;
      }
    }
  }

  // 2. preparations and testing of starting points

  std::cerr.precision(8);

  //std::cerr << "init params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;


  if (projection_mode == Simplex) {
    if (grouping_param < 0.0) {
      projection_on_simplex(dist_params.direct_access(), dist_params.size(), hmm_min_param_entry);
    }
    else {
      projection_on_simplex_with_slack(dist_params.direct_access() + zero_offset - redpar_limit, grouping_param, 2 * redpar_limit + 1, hmm_min_param_entry);
      grouping_param = std::max(hmm_min_param_entry, grouping_param);
    }
  }
  else {

    //projection on the positive orthant, followed by renormalization
    //(justified by scale invariance with positive scaling factors)
    // may be faster than the simplex projection

    double sum = 0.0;
    for (uint k = start_idx; k <= end_idx; k++) {

      dist_params[k] = std::max(hmm_min_param_entry, dist_params[k]);
      sum += dist_params[k];
    }
    if (grouping_param >= 0.0) {

      grouping_param = std::max(hmm_min_param_entry, grouping_param);
      sum += grouping_param;
    }

    double inv_sum = 1.0 / sum;
    for (uint k = start_idx; k <= end_idx; k++) {
      dist_params[k] = std::max(hmm_min_param_entry, inv_sum * dist_params[k]);
    }
    grouping_param = std::max(hmm_min_param_entry, inv_sum * grouping_param);
  }

  //std::cerr << "init params after projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param after projection: " << grouping_param << std::endl;

  Math1D::Vector<double> dist_grad = dist_params;
  Math1D::Vector<double> new_dist_params = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  double grouping_grad = 0.0;
  double new_grouping_param = grouping_param;
  double hyp_grouping_param = grouping_param;

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, dist_params,
                  zero_offset, grouping_param, redpar_limit);

  //test if normalized counts give a better starting point
  {
    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += singleton_count[zero_offset + k];
      norm += grouping_count;

      if (norm > 1e-305) {

        for (int k = zero_offset-redpar_limit; k <= zero_offset+redpar_limit; k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / norm);

        double hyp_grouping_param = std::max(hmm_min_param_entry, grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {

          dist_params = hyp_dist_params;
          grouping_param = hyp_grouping_param;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = singleton_count.sum();

      if (sum > 1e-305) {
        for (uint k = 0; k < hyp_dist_params.size(); k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {
          dist_params = hyp_dist_params;
          energy = hyp_energy;
        }
      }
    }
  }

  if (deficient)
    return;

  // 3. main loop
  std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  //double alpha  = 0.0001;
  //double alpha  = 0.001;
  double alpha = 100.0;

  double line_reduction_factor = 0.5;

  for (uint iter = 1; iter <= nIter; iter++) {

    if ((iter % 5) == 0)
      std::cerr << "m-step gd-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;

    // a) calculate gradient
    if (grouping_param < 0.0) {
      //fully parametric

      //singleton terms
      for (uint d = 0; d < singleton_count.size(); d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        Math1D::Vector<double> addon(dist_params.size(), 0.0);

        double param_sum = dist_params.range_sum(span_start, zero_offset);
        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            addon[span_end] = cur_count / param_sum;
            // double addon = cur_count / param_sum;

            // for (uint d=span_start; d <= span_end; d++)
            //   dist_grad[d] += addon;
          }
        }

        double sum_addon = 0.0;
        for (int d = zero_offset + span_count.yDim() - 1; d >= int (zero_offset); d--) {
          sum_addon += addon[d];
          dist_grad[d] += sum_addon;
        }
        for (int d = zero_offset - 1; d >= int (span_start); d--)
          dist_grad[d] += sum_addon;
      }
    }
    else {
      //reduced parametric

      uint first_diff = zero_offset - redpar_limit;
      uint last_diff = zero_offset + redpar_limit;

      for (uint d = first_diff; d <= last_diff; d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //NOTE: because we do not divide grouping_param by the various numbers of affected positions
      //   this energy will differ from the inefficient version by a constant
      grouping_grad -= grouping_count / std::max(hmm_min_param_entry, grouping_param);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          //there should be plenty of room for speed-ups here

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            double param_sum = 0.0;
            if (span_start < first_diff || span_end > last_diff)
              param_sum = grouping_param;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              param_sum += std::max(hmm_min_param_entry, dist_params[d]);

            double addon = cur_count / param_sum;
            if (span_start < first_diff || span_end > last_diff)
              grouping_grad += addon;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              dist_grad[d] += addon;
          }
        }
      }
    }

    // b) go in gradient direction

    double real_alpha = alpha;

    //TRIAL
    // double sqr_grad_norm  = 0.0;
    // for (uint k=0; k < dist_params.size(); k++)
    //   sqr_grad_norm += dist_grad.direct_access(k) * dist_grad.direct_access(k);
    // sqr_grad_norm += grouping_grad * grouping_grad;

    // real_alpha /= sqrt(sqr_grad_norm);
    //END_TRIAL

    for (uint k = start_idx; k <= end_idx; k++)
      new_dist_params.direct_access(k) = dist_params.direct_access(k) - real_alpha * dist_grad.direct_access(k);

    new_grouping_param = grouping_param - real_alpha * grouping_grad;

    //std::cerr << "new params before projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param before projection: " << new_grouping_param << std::endl;

    // c) projection

    for (uint k = start_idx; k <= end_idx; k++) {
      if (new_dist_params[k] >= 1e75)
        new_dist_params[k] = 9e74;
      else if (new_dist_params[k] <= -1e75)
        new_dist_params[k] = -9e74;
    }
    if (new_grouping_param >= 1e75)
      new_grouping_param = 9e74;
    else if (new_grouping_param <= -1e75)
      new_grouping_param = -9e74;

    if (norm_constraint) {

      if (grouping_param < 0.0) {
        projection_on_simplex(new_dist_params.direct_access(), dist_params.size(), hmm_min_param_entry);
      }
      else {
        projection_on_simplex_with_slack(new_dist_params.direct_access() + start_idx, new_grouping_param,
                                         2 * redpar_limit + 1, hmm_min_param_entry);
        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
      }
    }
    else {

      //projection on the positive orthant, followed by renormalization
      //(justified by scale invariance with positive scaling factors)
      // may be faster than the simplex projection

      uint nNeg = 0; //DEBUG

      double sum = 0.0;
      for (uint k = start_idx; k <= end_idx; k++) {

        //DEBUG
        if (new_dist_params[k] < 0.0)
          nNeg++;
        //END_DEBUG

        new_dist_params[k] = std::max(hmm_min_param_entry, new_dist_params[k]);
        sum += new_dist_params[k];
      }
      if (grouping_param >= 0.0) {

        //DEBUG
        if (new_grouping_param < 0.0)
          nNeg++;
        //END_DEBUG

        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
        sum += new_grouping_param;
      }
      //DEBUG
      //std::cerr << "sum: " << sum << ", " << nNeg << " were negative" << std::endl;
      //END_DEBUG

      //projection done, now renormalize to keep the probability constraint
      double inv_sum = 1.0 / sum;
      for (uint k = start_idx; k <= end_idx; k++) {
        new_dist_params[k] = std::max(hmm_min_param_entry,inv_sum * new_dist_params[k]);
      }
      new_grouping_param = std::max(hmm_min_param_entry,inv_sum * new_grouping_param);
    }

    //std::cerr << "new params after projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param after projection: " << new_grouping_param << std::endl;

    // d) find step-size

    new_grouping_param = std::max(new_grouping_param, hmm_min_param_entry);

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint k = start_idx; k <= end_idx; k++)
        hyp_dist_params.direct_access(k) = std::max(hmm_min_param_entry,lambda * new_dist_params.direct_access(k) +
                                           neg_lambda * dist_params.direct_access(k));

      if (grouping_param >= 0.0)
        hyp_grouping_param = std::max(hmm_min_param_entry, lambda * new_grouping_param + neg_lambda * grouping_param);

      double new_energy = ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                                             hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "hyp energy: " << new_energy << std::endl;
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    // if (nIter > 4)
    //   alpha *= 1.5;

    //DEBUG
    // if (best_lambda == 1.0)
    //   std::cerr << "!!!TAKING FULL STEP!!!" << std::endl;
    //END_DEBUG

    if (nIter > 15 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = start_idx; k <= end_idx; k++)
      dist_params.direct_access(k) = std::max(hmm_min_param_entry, neg_best_lambda * dist_params.direct_access(k) +
                                              best_lambda * new_dist_params.direct_access(k));

    if (grouping_param >= 0.0)
      grouping_param = std::max(hmm_min_param_entry, best_lambda * new_grouping_param + neg_best_lambda * grouping_param);

#ifndef NDEBUG
    double sum = dist_params.sum();
    if (grouping_param >= 0.0)
      sum += grouping_param;

    assert(sum >= 0.99 && sum <= 1.01);
#endif

    //std::cerr << "updated params: " << dist_params << std::endl;
    //std::cerr << "updated grouping param: " << grouping_param << std::endl;
  }
}

//@returns the denominator of the renormalization expression
inline double unconstrained2constrained_m_step_point(const Math1D::Vector<double>& param, uint start_idx, uint end_idx, bool redpar,
    Math1D::Vector<double>& dist_prob, double& grouping_prob, int redpar_limit)
{

  const uint nParams = param.size();

  double sum = 0.0;

  if (!redpar) {

    for (uint k = 0; k < nParams; k++) {
      double x = param[k];
      dist_prob[k] = x * x;
      sum += x * x;
    }

    assert(sum > 1e-305);
    double inv_sum = 1.0 / sum;
    for (uint k = 0; k < nParams; k++) {
      dist_prob[k] = std::max(hmm_min_param_entry, dist_prob[k] * inv_sum);
    }
  }
  else {

    for (uint k = start_idx; k <= end_idx; k++) {
      double x = param[k - start_idx];
      dist_prob[k] = x * x;
      sum += x * x;
    }
    double x = param[2 * redpar_limit + 1];
    grouping_prob = x * x;
    sum += x * x;

    //std::cerr << "sum: " << sum << std::endl;

    assert(sum > 1e-305);
    double inv_sum = 1.0 / sum;
    for (uint k = start_idx; k <= end_idx; k++) {
      dist_prob[k] = std::max(hmm_min_param_entry, dist_prob[k] * inv_sum);
    }
    grouping_prob = std::max(hmm_min_param_entry, grouping_prob * inv_sum);
  }

  return sum;
}

void ehmm_m_step_unconstrained(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params,
                               uint zero_offset, uint nIter, double& grouping_param, bool deficient, int redpar_limit)
{
  //in this formulation we use parameters p=x^2 to get an unconstrained formulation
  // here we use nonlinear conjugate gradients  and as a special case gradient descent

  std::cerr.precision(8);

  //std::cerr << "init params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;

  const uint nParams = (grouping_param < 0.0) ? dist_params.size() : 2 * redpar_limit + 2;

  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);
  Math1D::Vector<double> prev_work_grad(nParams, 0.0);
  Math1D::Vector<double> hyp_work_param(nParams);

  Math1D::Vector<double> dist_grad = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  double grouping_grad = 0.0;
  double hyp_grouping_param = grouping_param;

  // collect compact counts from facount

  Math1D::Vector<double> singleton_count(dist_params.size(), 0.0);
  double grouping_count = 0.0;
  Math2D::Matrix<double> span_count(zero_offset + 1, dist_params.size() - zero_offset, 0.0);

  for (int I = 1; I <= int (facount.size()); I++) {

    if (facount[I - 1].xDim() != 0) {

      for (int i = 0; i < I; i++) {

        double count_sum = 0.0;
        for (int i_next = 0; i_next < I; i_next++) {

          const double cur_count = facount[I - 1](i_next, i);
          if (grouping_param < 0.0 || abs(i_next - i) <= redpar_limit) {
            singleton_count[zero_offset + i_next - i] += cur_count;
            count_sum += cur_count;
          }
          else {
            double grouping_norm = std::max(0, i - redpar_limit);
            grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));
            grouping_count += cur_count / grouping_norm;
            count_sum += cur_count / grouping_norm;
          }
        }
        span_count(zero_offset - i, I - 1 - i) += count_sum;
      }
    }
  }

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, dist_params,
                  zero_offset, grouping_param, redpar_limit);

  //test if normalized counts give a better starting point
  {
    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += singleton_count[zero_offset + k];
      norm += grouping_count;

      if (norm > 1e-305) {

        for (int k = zero_offset-redpar_limit; k <= zero_offset+redpar_limit; k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / norm);

        double hyp_grouping_param = std::max(hmm_min_param_entry, grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {

          dist_params = hyp_dist_params;
          grouping_param = hyp_grouping_param;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = singleton_count.sum();

      if (sum > 1e-305) {
        for (uint k = 0; k < hyp_dist_params.size(); k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {
          dist_params = hyp_dist_params;
          energy = hyp_energy;
        }
      }
    }
  }

  if (deficient)
    return;

  //std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  double alpha = 1.0;           //modified below, dependent on the gradient norm

  //extract working params from the current probabilities (the probabilities are the squared working params)
  if (grouping_param < 0.0) {
    for (uint k = 0; k < nParams; k++)
      work_param[k] = sqrt(dist_params[k]);
  }
  else {
    for (uint k = start_idx; k <= end_idx; k++) {

      work_param[k - start_idx] = sqrt(dist_params[k]);
      work_param[2 * redpar_limit + 1] = sqrt(grouping_param);
    }
  }

  double prev_grad_norm = 0.0;

  double line_reduction_factor = 0.5;

  bool restart = false;

  for (uint iter = 1; iter <= nIter; iter++) {

    //if ((iter % 5) == 0)
    std::cerr << "unconstrained m-step gd/nlcg-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;
    work_grad.set_constant(0.0);

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    if (grouping_param < 0.0) {
      //fully parametric

      //singleton terms
      for (uint d = 0; d < singleton_count.size(); d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        Math1D::Vector<double> addon(dist_params.size(), 0.0);

        Math1D::Vector<double> init_sum(zero_offset + 1);
        init_sum[zero_offset] = 0.0;
        init_sum[zero_offset - 1] = dist_params[zero_offset - 1];
        for (int s = zero_offset - 2; s >= 0; s--)
          init_sum[s] = init_sum[s + 1] + dist_params[s];

        double param_sum = init_sum[span_start];        //dist_params.range_sum(span_start,zero_offset);
        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            addon[span_end] = cur_count / param_sum;
            // double addon = cur_count / param_sum;

            // for (uint d=span_start; d <= span_end; d++)
            //   dist_grad[d] += addon;
          }
        }

        double sum_addon = 0.0;
        for (int d = zero_offset + span_count.yDim() - 1; d >= int (zero_offset); d--) {
          sum_addon += addon[d];
          dist_grad[d] += sum_addon;
        }
        for (int d = zero_offset - 1; d >= int (span_start); d--)
          dist_grad[d] += sum_addon;
      }
    }
    else {
      //reduced parametric

      uint first_diff = zero_offset - redpar_limit;
      uint last_diff = zero_offset + redpar_limit;

      for (uint d = first_diff; d <= last_diff; d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //NOTE: because we do not divide grouping_param by the various numbers of affected positions
      //   this energy will differ from the inefficient version by a constant
      grouping_grad -= grouping_count / std::max(hmm_min_param_entry, grouping_param);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          //there should be plenty of room for speed-ups here

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            double param_sum = 0.0;
            if (span_start < first_diff || span_end > last_diff)
              param_sum = grouping_param;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              param_sum += std::max(hmm_min_param_entry, dist_params[d]);

            double addon = cur_count / param_sum;
            if (span_start < first_diff || span_end > last_diff)
              grouping_grad += addon;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              dist_grad[d] += addon;
          }
        }
      }
    }

    // b) now calculate the gradient for the actual parameters, store in work_grad

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    if (grouping_param < 0.0) {

      double coeff_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        const double wp = work_param[k];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;

        work_grad[k] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }
    else {

      double coeff_sum = 0.0;

      for (uint k = start_idx; k <= end_idx; k++) {
        const double wp = work_param[k - start_idx];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;
        work_grad[k - start_idx] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      const double wp = work_param[2 * redpar_limit + 1];
      const double grad = grouping_grad;
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[2 * redpar_limit + 1] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //        work_grad[kk] -= coeff * work_param[kk];

      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }

    double new_grad_norm = work_grad.sqr_norm();

    //in NLCG mode: modify the search direction
    if (iter == 1 || restart) {
      //if (true) {
      std::cerr << "RESTART" << std::endl;
      search_direction = work_grad;
      negate(search_direction);
    }
    else {
      double beta_fr = new_grad_norm / prev_grad_norm;  //Fletcher-Reeves variant

      double numerator = 0.0;
      double beta_hs_denom = 0.0;
      for (uint k = 0; k < nParams; k++) {

        const double diff = work_grad[k] - prev_work_grad[k];
        numerator += diff * work_grad[k];
        beta_hs_denom += diff * search_direction[k];
      }

      double beta_hs = numerator / beta_hs_denom;
      //beta_hs = std::max(0.0,beta_hs);
      if (beta_hs < -beta_fr)
        beta_hs = -beta_fr;
      if (beta_hs > beta_fr)
        beta_hs = beta_fr;

      double beta = beta_hs;

      for (uint k = 0; k < nParams; k++) {
        search_direction[k] = -work_grad[k] + beta * search_direction[k];
      }
    }

    alpha = 10.0 / sqrt(search_direction.sqr_norm());

    // c) line search

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double best_alpha = alpha;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      if (nIter > 1)
        alpha *= line_reduction_factor;

      for (uint k = 0; k < nParams; k++)
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];

      //calculate corresponding probability distribution (square the params and renormalize)
      unconstrained2constrained_m_step_point(hyp_work_param, start_idx, end_idx, grouping_param >= 0,
                                             hyp_dist_params, hyp_grouping_param, redpar_limit);

      double new_energy = ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                                             hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "alpha: " << alpha << ", hyp energy: " << new_energy << std::endl;
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    //d) go to the determined point

    if (best_energy >= energy - 1e-4) {
      if (!restart)
        restart = true;
      else {
        std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy) << std::endl;
        std::cerr << "last squared gradient norm: " << new_grad_norm << std::endl;
        break;
      }
    }
    else
      restart = false;

    if (best_energy < energy) {
      energy = best_energy;

      for (uint k = 0; k < nParams; k++)
        work_param[k] += best_alpha * search_direction[k];

      //calculate corresponding probability distribution (square the params and renormalize)

      unconstrained2constrained_m_step_point(work_param, start_idx, end_idx, grouping_param >= 0, dist_params,
                                             grouping_param, redpar_limit);

      double sum = dist_params.sum();
      if (grouping_param >= 0.0)
        sum += grouping_param;

      assert(sum >= 0.99 && sum <= 1.01);
    }

    prev_work_grad = work_grad; //could be more efficient
    prev_grad_norm = new_grad_norm;
  }

}

void ehmm_m_step_unconstrained_LBFGS(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset, uint nIter,
                                     double& grouping_param, uint L, bool deficient, int redpar_limit)
{
  //in this formulation we use parameters p=x^2 to get an unconstrained formulation

  std::cerr.precision(8);

  //std::cerr << "init params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;

  const uint nParams = (grouping_param < 0.0) ? dist_params.size() : 2 * redpar_limit + 2;

  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);
  Math1D::Vector<double> hyp_work_param(nParams);

  Math1D::Vector<double> dist_grad = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  double grouping_grad = 0.0;
  double hyp_grouping_param = grouping_param;

  // collect compact counts from facount

  Math1D::Vector<double> singleton_count(dist_params.size(), 0.0);
  double grouping_count = 0.0;
  Math2D::Matrix<double> span_count(zero_offset + 1, dist_params.size() - zero_offset, 0.0);

  for (int I = 1; I <= int (facount.size()); I++) {

    if (facount[I - 1].xDim() != 0) {

      for (int i = 0; i < I; i++) {

        double count_sum = 0.0;
        for (int i_next = 0; i_next < I; i_next++) {

          double cur_count = facount[I - 1] (i_next, i);
          if (grouping_param < 0.0 || abs(i_next - i) <= redpar_limit) {
            singleton_count[zero_offset + i_next - i] += cur_count;
            count_sum += cur_count;
          }
          else {
            double grouping_norm = std::max(0, i - redpar_limit);
            grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));
            grouping_count += cur_count / grouping_norm;
            count_sum += cur_count / grouping_norm;
          }
        }
        span_count(zero_offset - i, I - 1 - i) += count_sum;
      }
    }
  }

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, dist_params,
                  zero_offset, grouping_param, redpar_limit);

  //test if normalized counts give a better starting point
  {
    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += singleton_count[zero_offset + k];
      norm += grouping_count;

      if (norm > 1e-305) {

        for (int k = zero_offset-redpar_limit; k <= zero_offset+redpar_limit; k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / norm);

        double hyp_grouping_param = std::max(hmm_min_param_entry, grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, hyp_dist_params, zero_offset,
                            hyp_grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {

          dist_params = hyp_dist_params;
          grouping_param = hyp_grouping_param;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = singleton_count.sum();

      if (sum > 1e-305) {
        for (uint k = 0; k < hyp_dist_params.size(); k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, hyp_dist_params, zero_offset,
                            grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {
          dist_params = hyp_dist_params;
          energy = hyp_energy;
        }
      }
    }
  }

  if (deficient)
    return;

  //std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  //extract working params from the current probabilities (the probabilities are the squared working params)
  if (grouping_param < 0.0) {
    for (uint k = 0; k < nParams; k++)
      work_param[k] = sqrt(dist_params[k]);
  }
  else {
    for (uint k = start_idx; k <= end_idx; k++) {

      work_param[k - start_idx] = sqrt(dist_params[k]);
      work_param[2 * redpar_limit + 1] = sqrt(grouping_param);
    }
  }

  double scale = 1.0;

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint iter = 1; iter <= nIter; iter++) {

    if ((iter % 5) == 0)
      std::cerr << "unconstrained m-step L-BFGS(" << L << ")-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    if (grouping_param < 0.0) {
      //fully parametric

      //singleton terms
      for (uint d = 0; d < singleton_count.size(); d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        Math1D::Vector<double> addon(dist_params.size(), 0.0);

        Math1D::Vector<double> init_sum(zero_offset + 1);
        init_sum[zero_offset] = 0.0;
        init_sum[zero_offset - 1] = dist_params[zero_offset - 1];
        for (int s = zero_offset - 2; s >= 0; s--)
          init_sum[s] = init_sum[s + 1] + dist_params[s];

        double param_sum = init_sum[span_start];        //dist_params.range_sum(span_start,zero_offset);
        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            addon[span_end] = cur_count / param_sum;
            // double addon = cur_count / param_sum;

            // for (uint d=span_start; d <= span_end; d++)
            //   dist_grad[d] += addon;
          }
        }

        double sum_addon = 0.0;
        for (int d = zero_offset + span_count.yDim() - 1; d >= int (zero_offset); d--) {
          sum_addon += addon[d];
          dist_grad[d] += sum_addon;
        }
        for (int d = zero_offset - 1; d >= int (span_start); d--)
          dist_grad[d] += sum_addon;
      }
    }
    else {
      //reduced parametric

      uint first_diff = zero_offset - redpar_limit;
      uint last_diff = zero_offset + redpar_limit;

      for (uint d = first_diff; d <= last_diff; d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //NOTE: because we do not divide grouping_param by the various numbers of affected positions
      //   this energy will differ from the inefficient version by a constant
      grouping_grad -= grouping_count / std::max(hmm_min_param_entry, grouping_param);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          //there should be plenty of room for speed-ups here

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            double param_sum = 0.0;
            if (span_start < first_diff || span_end > last_diff)
              param_sum = grouping_param;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              param_sum += std::max(hmm_min_param_entry, dist_params[d]);

            double addon = cur_count / param_sum;
            if (span_start < first_diff || span_end > last_diff)
              grouping_grad += addon;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              dist_grad[d] += addon;
          }
        }
      }
    }

    // b) now calculate the gradient for the actual parameters, store in work_grad

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = scale; //work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    std::cerr << "scale: " << denom << std::endl;

    if (grouping_param < 0.0) {

      double coeff_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        const double wp = work_param[k];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;

        work_grad[k] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }
    else {

      double coeff_sum = 0.0;

      for (uint k = start_idx; k <= end_idx; k++) {
        const double wp = work_param[k - start_idx];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;
        work_grad[k - start_idx] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      const double wp = work_param[2 * redpar_limit + 1];
      const double grad = grouping_grad;
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[2 * redpar_limit + 1] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //        work_grad[kk] -= coeff * work_param[kk];

      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }

    double new_grad_norm = work_grad.sqr_norm();

    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < nParams; k++) {

        //cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
        cur_grad_diff[k] += work_grad[k];
        cur_rho += cur_grad_diff[k] * cur_step[k];
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      if (cur_curv <= 0) {
        //this can happen as our function is not convex and we do not enforce part 2 of the Wolfe conditions
        // (this cannot be done by backtracking line search, see Algorithm 3.5 in [Nocedal & Wright])
        // Our solution is to simply restart L-BFGS now
        start_iter = iter;
      }

      rho[cur_l] = 1.0 / cur_rho;
    }

    search_direction = work_grad;

    if (iter > start_iter) {

      Math1D::Vector<double> alpha(L);

      const int cur_first_iter = std::max<int>(start_iter, iter - L);

      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter - 1; prev_iter >= cur_first_iter; prev_iter--) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double cur_alpha = search_direction % cur_step;
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        search_direction.add_vector_multiple(cur_grad_diff, -cur_alpha);
        // for (uint k=0; k < nParams; k++)
        //   search_direction[k] -= cur_alpha * cur_grad_diff[k];
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = search_direction % cur_grad_diff;
        beta *= rho[prev_l];

        const double gamma = alpha[prev_l] - beta;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] += cur_step[k] * gamma;
        }
      }

    }
    else {
      search_direction *= 1.0 / sqrt(search_direction.sqr_norm());
    }

    negate(search_direction);

    // d) line search

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double alpha = 1.0;
    double best_alpha = alpha;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      if (nIter > 1)
        alpha *= line_reduction_factor;

      for (uint k = 0; k < nParams; k++)
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];

      //calculate corresponding probability distribution (square the params and renormalize)

      unconstrained2constrained_m_step_point(hyp_work_param, start_idx, end_idx, (grouping_param >= 0),
                                             hyp_dist_params, hyp_grouping_param, redpar_limit);

      double new_energy = ehmm_m_step_energy(singleton_count, grouping_count, span_count, hyp_dist_params, zero_offset,
                                             hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "lambda: " << lambda << ", hyp energy: " << new_energy << std::endl;
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point

    if (best_energy >= energy - 1e-4) {
      std::
      cerr << "CUTOFF after " << iter << " iterations, last gain: " <<
           (energy - best_energy) << std::endl;
      std::cerr << "last squared gradient norm: " << new_grad_norm << std::endl;
      break;
    }

    energy = best_energy;

    uint cur_l = (iter % L);

    Math1D::Vector<double>& cur_step = step[cur_l];
    Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

    for (uint k = 0; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    //calculate corresponding probability distribution (square the params and renormalize)
    scale = unconstrained2constrained_m_step_point(work_param, start_idx, end_idx, grouping_param >= 0, dist_params, grouping_param, redpar_limit);

    double sum = dist_params.sum();
    if (grouping_param >= 0.0)
      sum += grouping_param;

    assert(sum >= 0.99 && sum <= 1.01);
  }
}

double ehmm_init_m_step_energy(const InitialAlignmentProbability& init_acount, const Math1D::Vector<double>& init_params)
{
  double energy = 0.0;

  for (uint I = 0; I < init_acount.size(); I++) {

    if (init_acount[I].size() > 0) {

      double non_zero_sum = 0.0;
      for (uint i = 0; i <= I; i++)
        non_zero_sum += init_params[i];

      for (uint i = 0; i <= I; i++) {

        energy -= init_acount[I][i] * std::log(init_params[i] / non_zero_sum);
      }
    }
  }

  return energy;
}

void ehmm_init_m_step(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter,
                      ProjectionMode projection_mode)
{
  projection_on_simplex(init_params.direct_access(), init_params.size(), hmm_min_param_entry);

  Math1D::Vector<double> m_init_grad = init_params;
  Math1D::Vector<double> new_init_params = init_params;
  Math1D::Vector<double> hyp_init_params = init_params;

  double energy = ehmm_init_m_step_energy(init_acount, init_params);

  {
    //try to find a better starting point
    Math1D::Vector<double> init_count(init_params.size(), 0.0);

    for (uint I = 1; I <= init_acount.size(); I++) {

      if (init_acount[I - 1].size() != 0) {
        for (uint i = 0; i < I; i++) {
          init_count[i] += init_acount[I - 1][i];
        }
      }
    }

    const double sum = init_count.sum();

    for (uint k=0; k < init_count.size(); k++)
      init_count[k] = std::max(hmm_min_param_entry, init_count[k] / sum);

    double hyp_energy = ehmm_init_m_step_energy(init_acount, init_count);

    if (hyp_energy < energy) {
      init_params = init_count;
      energy = hyp_energy;
    }
  }

  std::cerr << "start m-energy: " << energy << std::endl;

  for (uint iter = 1; iter <= nIter; iter++) {

    m_init_grad.set_constant(0.0);

    //calculate gradient
    for (uint I = 0; I < init_acount.size(); I++) {

      if (init_acount[I].size() > 0) {

        double non_zero_sum = 0.0;
        for (uint i = 0; i <= I; i++)
          non_zero_sum += init_params[i];

        double count_sum = 0.0;
        for (uint i = 0; i <= I; i++) {
          count_sum += init_acount[I][i];

          double cur_param = init_params[i];

          m_init_grad[i] -= init_acount[I][i] / cur_param;
        }

        for (uint i = 0; i <= I; i++) {
          m_init_grad[i] += count_sum / non_zero_sum;
        }
      }
    }

    //std::cerr << "gradient calculated" << std::endl;

    double alpha = 0.0001;

    for (uint k = 0; k < init_params.size(); k++) {
      new_init_params.direct_access(k) = init_params.direct_access(k) - alpha * m_init_grad.direct_access(k);
      assert(!isnan(new_init_params[k]));
    }

    //std::cerr << "projecting " << new_init_params << std::endl;

    // reproject
    for (uint k = 0; k < init_params.size(); k++) {
      if (new_init_params[k] >= 1e75)
        new_init_params[k] = 9e74;
      if (new_init_params[k] <= -1e75)
        new_init_params[k] = -9e74;
    }

    if (projection_mode == Simplex)
      projection_on_simplex(new_init_params.direct_access(), init_params.size(), hmm_min_param_entry);
    else {

      double sum = 0;
      for (uint k = 0; k < init_params.size(); k++) {
        new_init_params[k] = std::max(hmm_min_param_entry, new_init_params[k]);
        sum += new_init_params[k];
      }

      //projection on orthant done, now renormalize to keep the probability constraint
      new_init_params *= 1.0 / sum;
    }

    //std::cerr << "projection done" << std::endl;

    //find step-size
    double best_energy = 1e300;

    double lambda = 1.0;
    double line_reduction_factor = 0.5;
    double best_lambda = lambda;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint k = 0; k < init_params.size(); k++)
        hyp_init_params.direct_access(k) = lambda * new_init_params.direct_access(k) + neg_lambda * init_params.direct_access(k);

      double hyp_energy = ehmm_init_m_step_energy(init_acount, hyp_init_params);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = 0; k < init_params.size(); k++)
      init_params.direct_access(k) = neg_best_lambda * init_params.direct_access(k) + best_lambda * new_init_params.direct_access(k);

    if (nIter > 15 || best_lambda < 1e-12 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF" << std::endl;
      break;
    }

    energy = best_energy;

    if ((iter % 5) == 0)
      std::cerr << "init m-step gd-iter #" << iter << ", energy: " << energy << std::endl;
  }
}

void par2nonpar_hmm_init_model(const Math1D::Vector<double>& init_params, const Math1D::Vector<double>& source_fert, HmmInitProbType init_type,
                               InitialAlignmentProbability& initial_prob, bool start_empty_word, bool fix_p0)
{
  for (uint I = 1; I <= initial_prob.size(); I++) {

    Math1D::Vector<double>& cur_initial_prob = initial_prob[I-1];

    if (cur_initial_prob.size() > 0) {

      if (init_type == HmmInitPar) {

        const double norm = init_params.range_sum(0, I);
        const double factor = source_fert[1] / norm;

        for (uint i = 0; i < I; i++) {
          cur_initial_prob[i] = std::max(hmm_min_param_entry, factor * init_params[i]);
          assert(!isnan(initial_prob[I - 1][i]));
        }
        if (!start_empty_word) {
          for (uint i = I; i < 2 * I; i++) {
            cur_initial_prob[i] = source_fert[0] / I;
            assert(!isnan(initial_prob[I - 1][i]));
          }
        }
        else
          cur_initial_prob[I] = source_fert[0];
      }
      else if (init_type == HmmInitFix) {
        cur_initial_prob.set_constant(1.0 / cur_initial_prob.size());
      }
      else if (init_type == HmmInitFix2) {
        const double p1 = (fix_p0) ? source_fert[1] : 0.98;
        const double p0 = (fix_p0) ? source_fert[0] : 0.02;

        cur_initial_prob.range_set_constant(p1 / I, 0, I);
        cur_initial_prob.range_set_constant(p0 / (initial_prob[I - 1].size() - I), I, initial_prob[I - 1].size() - I);
      }

      double sum = cur_initial_prob.sum();
      assert(sum >= 0.99 && sum <= 1.01);
    }
  }
}

void par2nonpar_hmm_alignment_model(const Math1D::Vector<double>& dist_params, const uint zero_offset, const double dist_grouping_param,
                                    const Math1D::Vector<double>& source_fert, HmmAlignProbType align_type, bool deficient,
                                    FullHMMAlignmentModelNoClasses& align_model, int redpar_limit)
{
  for (uint I = 1; I <= align_model.size(); I++) {

    Math2D::Matrix<double>& cur_align_model = align_model[I-1];

    if (cur_align_model.size() > 0) {

      for (int i = 0; i < (int)I; i++) {

        double grouping_norm = std::max(0, i - redpar_limit);
        grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

        double non_zero_sum = 0.0;
        for (int ii = 0; ii < (int)I; ii++) {
          if (align_type != HmmAlignProbReducedpar || abs(ii - i) <= redpar_limit)
            non_zero_sum += dist_params[zero_offset + ii - i];
        }

        if (align_type == HmmAlignProbReducedpar && grouping_norm > 0.0) {
          non_zero_sum += dist_grouping_param;
        }

        assert(non_zero_sum > 1e-305);
        const double inv_sum = (deficient) ? 1.0 : 1.0 / non_zero_sum;

        for (int ii = 0; ii < (int)I; ii++) {
          if (align_type == HmmAlignProbReducedpar && abs(ii - i) > redpar_limit) {
            assert(!isnan(grouping_norm));
            assert(grouping_norm > 0.0);
            cur_align_model(ii, i) = std::max(hmm_min_param_entry, source_fert[1] * inv_sum * dist_grouping_param / grouping_norm);
          }
          else {
            assert(dist_params[zero_offset + ii - i] > 0.0);
            cur_align_model(ii, i) = std::max(hmm_min_param_entry, source_fert[1] * inv_sum * dist_params[zero_offset + ii - i]);
          }
          assert(!isnan(cur_align_model(ii, i)));
          assert(cur_align_model(ii, i) >= 0.0);
        }
        cur_align_model(I, i) = source_fert[0];
        assert(!isnan(cur_align_model(I, i)));
        assert(cur_align_model(I, i) >= 0.0);

        double sum = cur_align_model.row_sum(i);
        assert(sum >= 0.99 && sum <= 1.01);
      }
    }
  }
}

void init_hmm_from_prev(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, FullHMMAlignmentModel& align_model,
                        Math1D::Vector<double>& dist_params, double& dist_grouping_param, Math1D::Vector<double>& source_fert,
                        InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params, const HmmOptions& options,
                        TransferMode transfer_mode = TransferViterbi)
{
  dist_grouping_param = -1.0;

  HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;

  if (init_type >= HmmInitInvalid) {

    INTERNAL_ERROR << "invalid type for HMM initial alignment model" << std::endl;
    exit(1);
  }
  if (align_type >= HmmAlignProbInvalid) {

    INTERNAL_ERROR << "invalid type for HMM alignment model" << std::endl;
    exit(1);
  }

  SingleLookupTable aux_lookup;

  const uint nSentences = source.size();

  std::set<uint> seenIs;

  uint maxI = (align_type == HmmAlignProbReducedpar) ? redpar_limit + 1 : 1;
  for (size_t s = 0; s < nSentences; s++) {
    const uint curI = target[s].size();

    seenIs.insert(curI);
    maxI = std::max(maxI, curI);
  }

  uint zero_offset = maxI - 1;

  dist_params.resize(2 * maxI - 1, 1.0 / (2 * maxI - 1)); //even for nonpar, we will use this as initialization

  if (align_type == HmmAlignProbReducedpar) {

    dist_grouping_param = 0.2;
    dist_params.set_constant(0.0);
    const double val = 0.8 / (2 * redpar_limit + 1);
    for (int k = -redpar_limit; k <= redpar_limit; k++)
      dist_params[zero_offset + k] = 0.8 / val;
  }

  if (source_fert.size() != 2 || !options.fix_p0_) {
    source_fert.resize(2);  //even for nonpar, we will use this as initialization
    source_fert[0] = 0.02;
    source_fert[1] = 0.98;
  }

  if (init_type == HmmInitPar) {
    init_params.resize(maxI);
    init_params.set_constant(1.0 / maxI);
  }

  align_model.resize_dirty(maxI);       //note: access using I-1
  initial_prob.resize(maxI);

  for (std::set<uint>::const_iterator it = seenIs.begin(); it != seenIs.end(); it++) {
    const uint I = *it;

    //std::cerr << "I: " << I << std::endl;

    //x = new index, y = given index
    align_model[I - 1].resize_dirty(I + 1, I);  //because of empty words
    align_model[I - 1].set_constant(1.0 / (I + 1));

    if (align_type != HmmAlignProbNonpar) {
      for (uint i = 0; i < I; i++) {
        for (uint ii = 0; ii < I; ii++) {
          align_model[I - 1](ii, i) = source_fert[1] / I;
        }
        align_model[I - 1](I, i) = source_fert[0];
      }
    }

    if (!start_empty_word)
      initial_prob[I - 1].resize_dirty(2 * I);
    else
      initial_prob[I - 1].resize_dirty(I + 1);

    //std::cerr << "size: " << initial_prob[I-1].size() << std::endl;

    if (init_type != HmmInitPar) {
      if (init_type == HmmInitFix2) {

        const double p1 = (options.fix_p0_) ? source_fert[1] : 0.98;
        const double p0 = (options.fix_p0_) ? source_fert[0] : 0.02;

        initial_prob[I - 1].range_set_constant(p1 / I, 0, I);
        initial_prob[I - 1].range_set_constant(p0 / (initial_prob[I - 1].size() - I), I, initial_prob[I - 1].size() - I);
      }
      else
        initial_prob[I - 1].set_constant(1.0 / initial_prob[I - 1].size());
    }
    else {
      //we be set later
      initial_prob[I - 1].range_set_constant(source_fert[1] / I, 0, I);
      initial_prob[I - 1].range_set_constant(source_fert[0] / (initial_prob[I - 1].size() - I), I, initial_prob[I - 1].size() - I);
    }

    //std::cerr << "set initial prob: " << initial_prob[I-1] << std::endl;

    assert(fabs(initial_prob[I - 1].sum() - 1.0) < 1e-5);
  }

  if (transfer_mode != TransferNo) {

    const double ibm1_p0 = options.ibm1_p0_;

    if (align_type == HmmAlignProbReducedpar)
      dist_grouping_param = 0.0;
    dist_params.set_constant(0.0);

    for (uint s = 0; s < source.size(); s++) {

      const Math1D::Vector<uint>& cur_source = source[s];
      const Math1D::Vector<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      if (transfer_mode == TransferViterbi) {

        Storage1D<AlignBaseType> viterbi_alignment(curJ, 0);

        if (options.ibm2_alignment_model_.size() > curI && options.ibm2_alignment_model_[curI].size() > 0)
          compute_ibm2_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, options.ibm2_alignment_model_[curI],
                                         options.ibm2_sclass_, viterbi_alignment);
        else if (ibm1_p0 >= 0.0 && ibm1_p0 < 1.0)
          compute_ibm1p0_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, ibm1_p0, viterbi_alignment);
        else
          compute_ibm1_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, viterbi_alignment);

        for (uint j = 1; j < curJ; j++) {

          int prev_aj = viterbi_alignment[j - 1];
          int cur_aj = viterbi_alignment[j];

          if (prev_aj != 0 && cur_aj != 0) {
            int diff = cur_aj - prev_aj;

            if (abs(diff) <= redpar_limit || align_type != HmmAlignProbReducedpar)
              dist_params[zero_offset + diff] += 1.0;
            else
              dist_grouping_param += 1.0;
          }
        }
      }
      else {
        assert(transfer_mode == TransferPosterior);

        if (options.ibm2_alignment_model_.size() > curI && options.ibm2_alignment_model_[curI].size() > 0) {

          const Math3D::Tensor<double>& cur_align_prob = options.ibm2_alignment_model_[curI];
          const Math1D::Vector<WordClassType>& sclass = options.ibm2_sclass_;

          Math1D::Vector<double> prev_marg(curI+1);
          Math1D::Vector<double> cur_marg(curI+1);

          for (uint j = 1; j < curJ; j++) {

            //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
            //  We could cheat if we only want training/word alignment. But we just take the previous word
            const uint c = sclass[cur_source[j - 1]];
            const uint c_prev = (j-1 == 0) ? 0 : sclass[cur_source[j - 2]];

            const uint j_prev = j - 1;

            cur_marg[0] = cur_align_prob(0, j, c) * dict[0][cur_source[j] - 1];
            for (uint i = 0; i < curI; i++)
              cur_marg[i + 1] = cur_align_prob(i + 1, j, c) * dict[cur_target[i]][cur_lookup(j, i)];

            cur_marg *= 1.0 / cur_marg.sum();

            prev_marg[0] = cur_align_prob(0, j_prev, c_prev) * dict[0][cur_source[j - 1] - 1];
            for (uint i = 0; i < curI; i++)
              prev_marg[i + 1] = cur_align_prob(i + 1, j_prev, c_prev) * dict[cur_target[i]][cur_lookup(j - 1, i)];

            prev_marg *= 1.0 / prev_marg.sum();

            for (int i1 = 0; i1 < int(curI); i1++) {
              for (int i2 = 0; i2 < int(curI); i2++) {

                const double marg = prev_marg[i1 + 1] * cur_marg[i2 + 1];

                int diff = i2 - i1;
                if (abs(diff) <= redpar_limit || align_type != HmmAlignProbReducedpar)
                  dist_params[zero_offset + diff] += marg;
                else
                  dist_grouping_param += marg;
              }
            }
          }
        }
        else {

          double w0 = 1.0;
          double w1 = 1.0;

          if (ibm1_p0 >= 0.0 && ibm1_p0 < 1.0) {
            w0 = ibm1_p0;
            w1 = (1.0 - ibm1_p0) / curI;
          }

          for (uint j = 1; j < curJ; j++) {

            double sum = w0 * dict[0][cur_source[j] - 1];
            for (uint i = 0; i < curI; i++)
              sum += w1 * dict[cur_target[i]][cur_lookup(j, i)];

            double prev_sum = w0 * dict[0][cur_source[j - 1] - 1];
            for (uint i = 0; i < curI; i++)
              prev_sum += w1 * dict[cur_target[i]][cur_lookup(j - 1, i)];

            for (int i1 = 0; i1 < int(curI); i1++) {

              const double i1_weight = w1 * (dict[cur_target[i1]][cur_lookup(j - 1, i1)] / prev_sum);

              for (int i2 = 0; i2 < int(curI); i2++) {

                const double marg = w1 * (dict[cur_target[i2]][cur_lookup(j, i2)] / sum) * i1_weight;

                int diff = i2 - i1;
                if (abs(diff) <= redpar_limit || align_type != HmmAlignProbReducedpar)
                  dist_params[zero_offset + diff] += marg;
                else
                  dist_grouping_param += marg;
              }
            }
          }
        }
      }
    }
    double sum = dist_params.sum();
    if (align_type == HmmAlignProbReducedpar)
      sum += dist_grouping_param;
    dist_params *= 1.0 / sum;
    if (align_type == HmmAlignProbReducedpar) {
      dist_grouping_param *= 1.0 / sum;

      for (int k = -redpar_limit; k <= redpar_limit; k++)
        dist_params[zero_offset + k] = 0.75 * dist_params[maxI + k] + 0.25 * 0.8 / (2 * redpar_limit + 1);

      dist_grouping_param = 0.75 * dist_grouping_param + 0.25 * 0.2;
    }
    else {
      for (uint k = 0; k < dist_params.size(); k++) {
        dist_params[k] = 0.75 * dist_params[k] + 0.25 / dist_params.size();
      }
    }
  }

  HmmAlignProbType align_mode = align_type;
  if (align_mode == HmmAlignProbNonpar || align_mode == HmmAlignProbNonpar2)
    align_mode = HmmAlignProbFullpar;

  par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param,
                                 source_fert, align_mode, options.deficient_, align_model, redpar_limit);

  //std::cerr << "calling par2nonpar_hmm_init_model" << std::endl;

  if (init_type == HmmInitPar)
    par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word, options.fix_p0_);

  //std::cerr << "leaving init" << std::endl;
}

void train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const CooccuringWordsType& wcooc, FullHMMAlignmentModel& align_model, Math1D::Vector<double>& dist_params,
                        double& dist_grouping_param, Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params, SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                        const HmmOptions& options)
{
  std::cerr << "starting Extended HMM EM-training" << std::endl;

  uint nIterations = options.nIterations_;
  HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dictionary is assumed to be initialized

  SingleLookupTable aux_lookup;

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  const uint nSourceWords = options.nSourceWords_;
  const uint start_addon = (start_empty_word) ? 1 : 0;

  SingleWordDictionary fwcount(MAKENAME(fwcount));
  fwcount = dict;

  init_hmm_from_prev(source, slookup, target, dict, wcooc, align_model, dist_params, dist_grouping_param, source_fert,
                     initial_prob, init_params, options, options.transfer_mode_);

  const uint maxI = align_model.size();
  const uint zero_offset = maxI - 1;

  Math1D::Vector<double> source_fert_count(2);

  InitialAlignmentProbability ficount(maxI, MAKENAME(ficount));
  ficount = initial_prob;

  Math1D::Vector<double> fsentence_start_count(maxI);
  Math1D::Vector<double> fstart_span_count(maxI);

  FullHMMAlignmentModel facount(align_model.size(), MAKENAME(facount));
  for (uint I = 1; I <= facount.size(); I++) {
    if (align_model[I - 1].size() > 0)
      facount[I - 1].resize_dirty(I + 1, I);
  }

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting EHMM iteration #" << iter << std::endl;

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i = 0; i < options.nTargetWords_; i++) {
      fwcount[i].set_constant(0.0);
    }

    for (uint I = 1; I <= maxI; I++) {
      facount[I - 1].set_constant(0.0);
      ficount[I - 1].set_constant(0.0);
    }

    source_fert_count.set_constant(0.0);

    //these two are calculated from ficount after the loop over the sentences:
    fsentence_start_count.set_constant(0.0);
    fstart_span_count.set_constant(0.0);

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      //std::cerr << "J = " << curJ << ", curI = " << curI << std::endl;

      const Math2D::Matrix<double>& cur_align_model = align_model[curI - 1];
      const Math1D::Vector<double>& cur_init_prob = initial_prob[curI - 1];

      Math2D::Matrix<double>& cur_facount = facount[curI - 1];
      Math1D::Vector<double>& cur_ficount = ficount[curI - 1];

      Math2D::Matrix<double> cur_dict(curJ,curI+1);
      compute_dictmat(cur_source, cur_lookup, cur_target, dict, cur_dict);

      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2 * curI + start_addon, curJ, MAKENAME(forward));

      calculate_hmm_forward(cur_dict, cur_align_model, cur_init_prob, align_type, start_empty_word, forward, redpar_limit);

      //std::cerr << "forward calculated" << std::endl;

      const uint start_s_idx = cur_source[0];

      const long double sentence_prob = forward.row_sum(curJ - 1);
      assert(forward.min() >= 0.0);

      prev_perplexity -= std::log(sentence_prob);

      if (!(sentence_prob > 0.0)) {
        //if (true) {
        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair " << s << " with I=" << curI << ", J= " << curJ << std::endl;

        //DEBUG
        //exit(1);
        //END_DEBUG
      }
      assert(sentence_prob > 0.0);

      Math2D::NamedMatrix<long double> backward(2 * curI + start_addon, curJ, MAKENAME(backward));
      calculate_hmm_backward(cur_dict, cur_align_model, cur_init_prob, align_type, start_empty_word, backward, true, redpar_limit);

      const long double bwd_sentence_prob = backward.row_sum(0);
      assert(backward.min() >= 0.0);

      long double fwd_bwd_ratio = sentence_prob / bwd_sentence_prob;

      if (fwd_bwd_ratio < 0.999 || fwd_bwd_ratio > 1.001) {

        std::cerr << "fwd_bwd_ratio of " << fwd_bwd_ratio << " for sentence pair " << s << " with I=" << curI << ", J= " << curJ << std::endl;
      }

      assert(fwd_bwd_ratio < 1.001);
      assert(fwd_bwd_ratio > 0.999);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      /**** update counts ****/

      //NOTE: it might be faster to compute forward and backward with double precision and rescaling.
      //  There are then two options:
      //   1. keep track of scaling factors (as long double) and convert everything when needed in the count collection
      //   2. work with the rescaled values and normalize appropriately

      //start of sentence
      for (uint i = 0; i < curI; i++) {
        uint t_idx = cur_target[i];

        const double coeff = inv_sentence_prob * backward(i, 0);
        fwcount[t_idx][cur_lookup(0, i)] += coeff;

        assert(!isnan(coeff));

        cur_ficount[i] += coeff;
      }
      if (!start_empty_word) {
        double dict_count_sum = 0.0;
        for (uint i = 0; i < curI; i++) {
          const double coeff = inv_sentence_prob * backward(i + curI, 0);
          dict_count_sum += coeff;

          assert(!isnan(coeff));

          cur_ficount[i + curI] += coeff;
        }
        fwcount[0][start_s_idx - 1] += dict_count_sum;
      }
      else {
        //when using a separate start empty word: cover the case where the entire sentence aligns to NULL.
        // for long sentences the probability should be negligible

        const double addon = inv_sentence_prob * backward(2 * curI, 0);
        fwcount[0][start_s_idx - 1] += addon;
        cur_ficount[curI] += addon;
      }

      //mid-sentence
      for (uint j = 1; j < curJ; j++) {

        //std::cerr << "j: " << j << std::endl;

        const uint s_idx = cur_source[j];
        const uint j_prev = j - 1;

        //real positions
        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];

          const double dict_entry = cur_dict(j, i); //dict[t_idx][cur_lookup(j, i)];

          if (dict_entry > 1e-305) {
            fwcount[t_idx][cur_lookup(j, i)] += forward(i, j) * backward(i, j) * inv_sentence_prob / dict_entry;

            const long double bw = backward(i, j) * inv_sentence_prob;

            for (uint i_prev = 0; i_prev < curI; i_prev++) {
              long double addon = bw * cur_align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + curI, j_prev));
              assert(!isnan(addon));
              cur_facount(i, i_prev) += addon;
            }

            //start empty word
            if (start_empty_word) {
              long double addon = bw * cur_init_prob[i] * forward(2 * curI, j_prev);
              cur_ficount[i] += addon;
            }
          }
        }

        //empty words
        double dict_count_sum = 0.0;
        for (uint i = curI; i < 2 * curI; i++) {

          //combining j and j_prev does not contain a dict probability twice
          const long double bw = backward(i, j) * inv_sentence_prob;
          const long double addon = bw * cur_align_model(curI, i - curI) * (forward(i, j_prev) + forward(i - curI, j_prev));

          assert(!isnan(addon));

          dict_count_sum += addon;
          cur_facount(curI, i - curI) += addon;
        }
        fwcount[0][s_idx - 1] += dict_count_sum;

        //start empty word
        if (start_empty_word) {

          //combining j and j_prev does not contain a dict probability twice
          const long double bw = backward(2 * curI, j) * inv_sentence_prob;

          long double addon = bw * forward(2 * curI, j_prev) * cur_init_prob[curI];
          fwcount[0][s_idx - 1] += addon;
          cur_ficount[curI] += addon;
        }
      }
    } // loop over sentences finished

    //finish counts (align already done)

    if (init_type == HmmInitPar) {

      for (uint I = 1; I <= maxI; I++) {

        if (initial_prob[I - 1].size() != 0) {
          for (uint i = 0; i < I; i++) {

            const double cur_count = ficount[I - 1][i];
            source_fert_count[1] += cur_count;
            fsentence_start_count[i] += cur_count;
            fstart_span_count[I - 1] += cur_count;
          }
          for (uint i = I; i < ficount[I - 1].size(); i++) {
            source_fert_count[0] += ficount[I - 1][i];
          }
        }
      }
    }

    if (align_type != HmmAlignProbNonpar) {

      //compute the expectations of the parameters from the expectations of the models

      for (uint I = 1; I <= maxI; I++) {

        if (align_model[I - 1].xDim() != 0) {

          for (int i = 0; i < (int)I; i++) {

            for (int ii = 0; ii < (int)I; ii++) {
              source_fert_count[1] += facount[I - 1](ii, i);
            }
            source_fert_count[0] += facount[I - 1](I, i);
          }
        }
      }
    }

    double energy = prev_perplexity / nSentences;

    if (dict_weight_sum != 0.0) {
      energy += dict_reg_term(dict, prior_weight, options.l0_beta_);
    }

    std::cerr << "energy after iteration #" << (iter - 1) << ": " <<  energy << std::endl;
    std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    if (source_fert.size() > 0 && source_fert_count.sum() > 0.0 && !options.fix_p0_) {

      for (uint i = 0; i < 2; i++) {
        source_fert[i] = source_fert_count[i] / source_fert_count.sum();
        assert(!isnan(source_fert[i]));
      }

      if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {
        std::cerr << "new probability for zero alignments: " << source_fert[0] << std::endl;
      }
    }

    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {
      //std::cerr << "dist_params: " << dist_params << std::endl;

      //call m-step
      if (options.msolve_mode_ == MSSolvePGD)
        ehmm_m_step(facount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param, options.deficient_, redpar_limit,
                    options.projection_mode_);
      else if (options.msolve_mode_ == MSSolveGD)
        ehmm_m_step_unconstrained(facount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param,
                                  options.deficient_, redpar_limit);
      else
        ehmm_m_step_unconstrained_LBFGS(facount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param, 5,
                                        options.deficient_, redpar_limit);

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, options.deficient_, align_model, redpar_limit);
    }

    if (init_type == HmmInitPar) {

      if (options.msolve_mode_ == MSSolvePGD)
        start_prob_m_step(fsentence_start_count, fstart_span_count, init_params, options.init_m_step_iter_);
      else
        start_prob_m_step_unconstrained(fsentence_start_count,fstart_span_count, init_params, options.init_m_step_iter_);

      par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word, options.fix_p0_);
    }

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts
    update_dict_from_counts(fwcount, prior_weight, nSentences, dict_weight_sum, options.smoothed_l0_, options.l0_beta_,
                            options.dict_m_step_iter_, dict, hmm_min_dict_entry, options.msolve_mode_ != MSSolvePGD);

    //compute new nonparametric probabilities from normalized fractional counts
    for (uint I = 1; I <= maxI; I++) {

      if (align_model[I - 1].xDim() != 0) {

        if (init_type == HmmInitNonpar) {
          double inv_norm = 1.0 / ficount[I - 1].sum();
          for (uint i = 0; i < initial_prob[I - 1].size(); i++)
            initial_prob[I - 1][i] = std::max(hmm_min_param_entry, inv_norm * ficount[I - 1][i]);
        }

        if (align_type == HmmAlignProbNonpar) {

          for (uint i = 0; i < I; i++) {

            double sum = facount[I - 1].row_sum(i);

            if (sum >= 1e-300) {

              assert(!isnan(sum));
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));

              for (uint i_next = 0; i_next <= I; i_next++) {
                align_model[I - 1](i_next, i) = std::max(hmm_min_param_entry, inv_sum * facount[I - 1] (i_next, i));
                assert(!isnan(align_model[I - 1](i_next, i)));
              }
            }
          }
        }
        else if (align_type == HmmAlignProbNonpar2) {

          for (uint i = 0; i < I; i++) {

            align_model[I - 1] (I, i) = source_fert[0];

            double sum = facount[I - 1].row_sum(i) - facount[I - 1] (I, i);

            if (sum >= 1e-300) {

              assert(!isnan(sum));
              const double inv_sum = source_fert[1] / sum;
              assert(!isnan(inv_sum));

              for (uint i_next = 0; i_next < I; i_next++) {
                align_model[I - 1] (i_next, i) = std::max(hmm_min_param_entry, inv_sum * facount[I - 1](i_next, i));
                assert(!isnan(align_model[I - 1](i_next, i)));
              }
            }
          }
        }
      }
    }

    /************* compute alignment error rate ****************/
    if (options.print_energy_) {
      std::cerr << "#### EHMM energy after iteration # " << iter << ": "
                << extended_hmm_energy(source, slookup, target, align_model, initial_prob, dict, wcooc, nSourceWords,
                                       prior_weight, options, dict_weight_sum)
                << std::endl;
    }

    if (!options.possible_ref_alignments_.empty()) {

      //std::cerr << "computing error rates" << std::endl;

      double sum_aer = 0.0;
      double sum_marg_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        //std::cerr << "s: " << s << std::endl;

        if (s >= nSentences)
          break;

        nContributors++;
        //compute viterbi alignment

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        Math1D::Vector<AlignBaseType> viterbi_alignment;
        const uint curI = target[s].size();

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        //std::cerr << "computing alignment" << std::endl;

        compute_ehmm_viterbi_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                       initial_prob[curI - 1],viterbi_alignment, options);

        //std::cerr << "alignment: " << viterbi_alignment << std::endl;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        Storage1D<AlignBaseType> marg_alignment;
        compute_ehmm_optmarginal_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                           initial_prob[curI - 1], start_empty_word, marg_alignment);

        //std::cerr << "marg_alignment: " << marg_alignment << std::endl;

        sum_marg_aer += AER(marg_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ehmm_postdec_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                       initial_prob[curI - 1], options, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_marg_aer *= 100.0 / nContributors;
      nErrors /= nContributors;
      sum_fmeasure /= nContributors;
      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### EHMM Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "---- EHMM Marginal-AER : " << sum_marg_aer << " %" << std::endl;
      std::cerr << "#### EHMM Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;
      std::cerr << "#### EHMM Postdec-AER after iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### EHMM Postdec-fmeasure after iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### EHMM Postdec-DAE/S after iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  }  //end for (iter)
}

void train_extended_hmm_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                       const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                       FullHMMAlignmentModel& align_model, Math1D::Vector<double>& dist_params,
                                       double& dist_grouping_param, Math1D::Vector<double>& source_fert,
                                       InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                                       SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight, const HmmOptions& options)
{
  std::cerr << "starting Extended HMM GD-training" << std::endl;

  uint nIterations = options.nIterations_;
  HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;

  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dicitionary is assumed to be initialized

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  const uint nSourceWords = options.nSourceWords_;

  SingleLookupTable aux_lookup;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  init_hmm_from_prev(source, slookup, target, dict, wcooc, align_model, dist_params, dist_grouping_param, source_fert,
                     initial_prob, init_params, options, options.transfer_mode_);

  const uint maxI = align_model.size();
  const uint zero_offset = maxI - 1;
  const uint start_addon = (start_empty_word) ? 1 : 0;

  double dist_grouping_grad = dist_grouping_param;
  double new_dist_grouping_param = dist_grouping_param;
  double hyp_dist_grouping_param = dist_grouping_param;

  Math1D::NamedVector<double> dist_grad(MAKENAME(dist_grad));
  dist_grad = dist_params;

  Math1D::NamedVector<double> new_dist_params(MAKENAME(new_dist_params));
  new_dist_params = dist_params;
  Math1D::NamedVector<double> hyp_dist_params(MAKENAME(hyp_dist_params));
  hyp_dist_params = dist_params;

  Math1D::NamedVector<double> init_param_grad(MAKENAME(init_param_grad));
  init_param_grad = init_params;
  Math1D::NamedVector<double> new_init_params(MAKENAME(new_init_params));
  new_init_params = init_params;
  Math1D::NamedVector<double> hyp_init_params(MAKENAME(hyp_init_params));
  hyp_init_params = init_params;

  Math1D::Vector<double> source_fert_grad = source_fert;
  Math1D::Vector<double> new_source_fert = source_fert;
  Math1D::Vector<double> hyp_source_fert = source_fert;

  InitialAlignmentProbability init_grad(maxI, MAKENAME(init_grad));

  Storage1D<Math1D::Vector<double> > dict_grad(options.nTargetWords_);
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_grad[i].resize(dict[i].size());
  }

  FullHMMAlignmentModel align_grad(maxI, MAKENAME(align_grad));
  for (uint I = 1; I <= maxI; I++) {
    if (align_model[I - 1].size() > 0) {
      align_grad[I - 1].resize_dirty(I + 1, I);
      init_grad[I - 1].resize_dirty(initial_prob[I - 1].size());
    }
  }

  InitialAlignmentProbability new_init_prob(maxI, MAKENAME(new_init_prob));
  InitialAlignmentProbability hyp_init_prob(maxI, MAKENAME(hyp_init_prob));
  new_init_prob = initial_prob;
  hyp_init_prob = initial_prob;

  SingleWordDictionary new_dict_prob(options.nTargetWords_, MAKENAME(new_dict_prob));
  SingleWordDictionary hyp_dict_prob(options.nTargetWords_, MAKENAME(hyp_dict_prob));

  for (uint i = 0; i < options.nTargetWords_; i++) {
    new_dict_prob[i].resize(dict[i].size());
    hyp_dict_prob[i].resize(dict[i].size());
  }

  FullHMMAlignmentModel new_align_prob(maxI, MAKENAME(new_align_prob));
  FullHMMAlignmentModel hyp_align_prob(maxI, MAKENAME(hyp_align_prob));

  for (uint I = 1; I <= maxI; I++) {
    if (align_grad[I - 1].size() > 0) {
      new_align_prob[I - 1].resize_dirty(I + 1, I);
      hyp_align_prob[I - 1].resize_dirty(I + 1, I);
    }
  }

  Math1D::Vector<double> slack_vector(options.nTargetWords_, 0.0);
  Math1D::Vector<double> new_slack_vector(options.nTargetWords_, 0.0);

  for (uint i = 0; i < options.nTargetWords_; i++) {
    double slack_val = 1.0 - dict[i].sum();
    slack_vector[i] = slack_val;
    new_slack_vector[i] = slack_val;
  }

  double energy = extended_hmm_energy(source, slookup, target, align_model, initial_prob, dict, wcooc, nSourceWords, prior_weight, options, dict_weight_sum);

  double line_reduction_factor = 0.5;

  uint nSuccessiveReductions = 0;

  std::cerr << "start energy: " << energy << std::endl;

  //double alpha = 0.005; //0.002;
  //double alpha = 0.001; //(align_type == HmmAlignProbReducedpar) ? 0.0025 : 0.001;
  double alpha = 50.0;

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting EHMM gd-iter #" << iter << std::endl;
    std::cerr << "alpha: " << alpha << std::endl;

    //set counts to 0
    for (uint i = 0; i < options.nTargetWords_; i++) {
      dict_grad[i].set_constant(0.0);
    }

    for (uint I = 1; I <= maxI; I++) {
      align_grad[I - 1].set_constant(0.0);
      init_grad[I - 1].set_constant(0.0);
    }

    if (align_type != HmmAlignProbNonpar) {

      dist_grad.set_constant(0.0);
      source_fert_grad.set_constant(0.0);
    }

    dist_grouping_grad = 0.0;

    /******** 1. calculate gradients **********/

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      const Math2D::Matrix<double>& cur_align_model = align_model[curI - 1];
      const Math1D::Vector<double>& cur_init_prob = initial_prob[curI - 1];

      Math2D::Matrix<double> cur_dict(curJ,curI+1);
      compute_dictmat(cur_source, cur_lookup, cur_target, dict, cur_dict);

      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2 * curI + start_addon, curJ, MAKENAME(forward));
      Math2D::NamedMatrix<long double> backward(2 * curI + start_addon, curJ,MAKENAME(backward));

      calculate_hmm_forward(cur_dict, cur_align_model, cur_init_prob, align_type, start_empty_word, forward, redpar_limit);

      const uint start_s_idx = cur_source[0];

      const long double sentence_prob = forward.row_sum(curJ - 1);
      assert(forward.min() >= 0.0);

      if (!(sentence_prob > 0.0)) {
        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair " << s << " with I=" << curI << ", J= " << curJ << std::endl;
      }
      assert(sentence_prob > 0.0);

      calculate_hmm_backward(cur_dict, cur_align_model, cur_init_prob, align_type, start_empty_word, backward, true, redpar_limit);
      assert(backward.min() >= 0.0);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      Math2D::Matrix<double>& cur_align_grad = align_grad[curI - 1];
      Math1D::Vector<double>& cur_init_grad = init_grad[curI - 1];

      /**** update gradients ****/

      //start of sentence
      for (uint i = 0; i < curI; i++) {
        const uint t_idx = cur_target[i];

        const double coeff = inv_sentence_prob * backward(i, 0);
        const double cur_dict_entry = cur_dict(0, i); //dict[t_idx][cur_lookup(0, i)];

        if (cur_dict_entry > 1e-300) {

          double addon = coeff / cur_dict_entry;
          dict_grad[t_idx][cur_lookup(0, i)] -= addon;
        }

        if (cur_init_prob[i] > 1e-300) {
          cur_init_grad[i] -= coeff; // division by param deferred until after the loop
          assert(!isnan(cur_init_grad[i]));
        }
      }

      const double cur_dict_entry = cur_dict(0, curI); //dict[0][start_s_idx - 1];
      if (cur_dict_entry > 1e-300) {
        if (start_empty_word) {
          const double coeff = inv_sentence_prob * backward(2 * curI, 0);

          dict_grad[0][start_s_idx - 1] -= coeff / std::max(hmm_min_dict_entry, cur_dict_entry);
          cur_init_grad[curI] -= coeff; // division by param deferred until after the loop
        }
        else {
          for (uint i = 0; i < curI; i++) {

            const long double coeff = inv_sentence_prob * backward(i + curI, 0);

            dict_grad[0][start_s_idx - 1] -= coeff / std::max(hmm_min_dict_entry, cur_dict_entry);

            if (initial_prob[curI - 1][i + curI] > 1e-300) {
              cur_init_grad[i + curI] -= coeff / std::max(hmm_min_param_entry, cur_init_prob[i + curI]);
              assert(!isnan(cur_init_grad[i + curI]));
            }
          }
        }
      }

      //mid-sentence
      for (uint j = 1; j < curJ; j++) {
        const uint s_idx = cur_source[j];
        const uint j_prev = j - 1;

        //real positions
        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];

          const double cur_dict_entry = cur_dict(j,i); //dict[t_idx][cur_lookup(j, i)];

          if (cur_dict_entry > 1e-70) {

            double dict_addon = forward(i, j) * backward(i, j) / (sentence_prob * std::max(1e-30, cur_dict_entry * cur_dict_entry));

            dict_grad[t_idx][cur_lookup(j, i)] -= dict_addon;

            const long double bw = backward(i, j) / sentence_prob;

            for (uint i_prev = 0; i_prev < curI; i_prev++) {

              const double addon = bw * (forward(i_prev, j_prev) + forward(i_prev + curI, j_prev));
              cur_align_grad(i, i_prev) -= addon; //division by prob deferred
            }
          }
        }

        //empty words
        const double cur_dict_entry = cur_dict(j,curI); //dict[0][s_idx - 1];

        if (cur_dict_entry > 1e-70) {

          for (uint i = curI; i < 2 * curI; i++) {

            const long double bw = backward(i, j) * inv_sentence_prob;
            const long double shared = bw * forward(i, j) / cur_dict_entry;

            const double addon = shared / cur_dict_entry;
            dict_grad[0][s_idx - 1] -= addon;

            //combining j and j_prev doesn't
            const double align_addon = shared;
            cur_align_grad(curI, i - curI) -= align_addon; //division by prob deferred
          }

          if (start_empty_word) {

            const long double bw = backward(2 * curI, j) * inv_sentence_prob;
            const long double shared = bw * forward(2 * curI, j) / cur_dict_entry;

            dict_grad[0][s_idx - 1] -= shared / cur_dict_entry;
            cur_init_grad[curI] -= shared; // division by param deferred until after the loop

            for (uint i = 0; i < curI; i++)
              cur_init_grad[curI] -= shared; // division by param deferred until after the loop
          }
        }
      }
    }  //end for (s)

    //std::cerr << "loop over s finished" << std::endl;

    for (uint i = 0; i < options.nTargetWords_; i++) {
      dict_grad[i] *= 1.0 / nSentences;
    }
    for (uint I = 1; I <= maxI; I++) {

      init_grad[I - 1] *= 1.0 / nSentences;

      for (uint k=0; k < init_grad[I - 1].size(); k++)
        init_grad[I - 1].direct_access(k) /= initial_prob[I - 1].direct_access(k);
    }
    for (uint I = 1; I <= maxI; I++) {
      align_grad[I - 1] *= 1.0 / nSentences;

      //finally include division
      for (uint k=0; k < align_grad[I - 1].size(); k++) {
        align_grad[I - 1].direct_access(k) /= std::max(hmm_min_param_entry, align_model[I - 1].direct_access(k));
      }
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

    for (uint I = 1; I <= maxI; I++) {

      if (init_grad[I - 1].size() > 0) {

        if (init_type == HmmInitPar) {

          for (uint i = 0; i < I; i++)
            source_fert_grad[1] += init_grad[I - 1][i] * (initial_prob[I - 1][i] / source_fert[1]);

          if (!start_empty_word)
            source_fert_grad[0] += init_grad[I - 1].range_sum(I, 2 * I) / I;
          else
            source_fert_grad[0] += init_grad[I - 1][I];

          const double sum = init_params.range_sum(0, I);

          for (uint i = 0; i < I; i++) {

            const double cur_grad = init_grad[I - 1][i] * source_fert[1] / (sum * sum);

            for (uint ii = 0; ii < I; ii++) {

              if (ii == i)
                init_param_grad[ii] += cur_grad * (sum - init_params[i]);
              else
                init_param_grad[ii] -= cur_grad * init_params[i];
            }
          }
        }
      }
    }

    if (align_type != HmmAlignProbNonpar) {

      for (uint I = 1; I <= maxI; I++) {

        if (align_grad[I - 1].size() > 0) {

          for (int i = 0; i < (int)I; i++) {

            for (uint ii = 0; ii < I; ii++)
              source_fert_grad[1] += align_grad[I - 1] (ii, i) * (align_model[I - 1] (ii, i) / source_fert[1]);
            source_fert_grad[0] += align_grad[I - 1] (I, i) * (align_model[I - 1] (I, i) / source_fert[0]);

            double non_zero_sum = 0.0;

            if (align_type == HmmAlignProbFullpar) {

              if (options.deficient_) {
                for (uint ii = 0; ii < I; ii++)
                  dist_grad[zero_offset + ii - i] += source_fert[1] * align_grad[I - 1] (ii, i);
              }
              else {

                for (uint ii = 0; ii < I; ii++) {
                  non_zero_sum += dist_params[zero_offset + ii - i];
                }

                //std::cerr << "I: " << I << ", i: " << i << ", non_zero_sum: " << non_zero_sum << std::endl;

                const double factor = source_fert[1] / (non_zero_sum * non_zero_sum);

                assert(!isnan(factor));

                for (uint ii = 0; ii < I; ii++) {
                  //NOTE: align_grad has already a negative sign

                  const double cur_grad = align_grad[I - 1] (ii, i);

                  dist_grad[zero_offset + ii - i] += cur_grad * factor * non_zero_sum;
                  for (uint iii = 0; iii < I; iii++)
                    dist_grad[zero_offset + iii - i] -= cur_grad * factor * dist_params[zero_offset + ii - i];
                }
              }
            }
            else {

              double grouping_norm = std::max(0, i - redpar_limit);
              grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

              if (options.deficient_) {
                for (int ii = 0; ii < (int)I; ii++) {
                  //NOTE: align_grad has already a negative sign
                  if (abs(ii - i) <= redpar_limit) {
                    dist_grad[zero_offset + ii - i] += source_fert[1] * align_grad[I - 1] (ii, i);
                  }
                  else {
                    dist_grouping_grad += source_fert[1] * align_grad[I - 1] (ii, i) / grouping_norm;
                  }
                }
              }
              else {

                if (grouping_norm > 0.0)
                  non_zero_sum += dist_grouping_param;

                for (int ii = 0; ii < (int)I; ii++) {

                  if (abs(ii - i) <= redpar_limit)
                    non_zero_sum += dist_params[zero_offset + ii - i];
                }

                const double factor = source_fert[1] / (non_zero_sum * non_zero_sum);

                assert(!isnan(factor));

                for (int ii = 0; ii < (int)I; ii++) {

                  const double cur_grad = align_grad[I - 1] (ii, i);

                  //NOTE: align_grad has already a negative sign
                  if (abs(ii - i) <= redpar_limit) {

                    dist_grad[zero_offset + ii - i] += cur_grad * factor * non_zero_sum;
                    for (int iii = 0; iii < (int)I; iii++) {
                      if (abs(iii - i) <= redpar_limit)
                        dist_grad[zero_offset + iii - i] -= cur_grad * factor * dist_params[zero_offset + ii - i];
                      else
                        dist_grouping_grad -= cur_grad * factor * dist_params[zero_offset + ii - i] / grouping_norm;
                    }
                  }
                  else {

                    dist_grouping_grad += cur_grad * factor * (non_zero_sum - dist_grouping_param) / grouping_norm;

                    for (int iii = 0; iii < (int)I; iii++) {
                      if (abs(iii - i) <= redpar_limit) {
                        assert(iii != ii);
                        dist_grad[zero_offset + iii - i] -= cur_grad * factor * dist_grouping_param / grouping_norm;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    /******** 2. move in gradient direction *********/

    //std::cerr << "move" << std::endl;

    double real_alpha = alpha;

    double sqr_grad_norm = 0.0;

    if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar)
      sqr_grad_norm += source_fert_grad.sqr_norm();
    if (init_type == HmmInitPar)
      sqr_grad_norm += init_param_grad.sqr_norm();
    if (align_type == HmmAlignProbFullpar) {
      sqr_grad_norm += dist_grad.sqr_norm();
    }
    else if (align_type == HmmAlignProbReducedpar) {
      sqr_grad_norm += dist_grad.sqr_norm() + dist_grouping_grad * dist_grouping_grad;
    }
    for (uint i = 0; i < options.nTargetWords_; i++)
      sqr_grad_norm += dict_grad[i].sqr_norm();

    real_alpha /= sqrt(sqr_grad_norm);

    //std::cerr << "A" << std::endl;

    if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {

      if (!options.fix_p0_) {
        for (uint i = 0; i < 2; i++)
          new_source_fert[i] = source_fert[i] - real_alpha * source_fert_grad[i];

        projection_on_simplex(new_source_fert.direct_access(), 2);
      }
    }
    //std::cerr << "B" << std::endl;

    if (init_type == HmmInitPar) {

      //for (uint k = 0; k < init_params.size(); k++)
      //  new_init_params[k] = init_params[k] - real_alpha * init_param_grad[k];
      Math1D::go_in_neg_direction(new_init_params, init_params, init_param_grad, real_alpha);

      projection_on_simplex(new_init_params.direct_access(), new_init_params.size(), hmm_min_param_entry);
    }
    //std::cerr << "C" << std::endl;

    if (align_type == HmmAlignProbFullpar) {

      //for (uint k = 0; k < dist_params.size(); k++)
      //  new_dist_params[k] = dist_params[k] - real_alpha * dist_grad[k];
      Math1D::go_in_neg_direction(new_dist_params, dist_params, dist_grad, real_alpha);

      projection_on_simplex(new_dist_params.direct_access(), new_dist_params.size(), hmm_min_param_entry);
    }
    else if (align_type == HmmAlignProbReducedpar) {

      //for (uint k = 0; k < dist_params.size(); k++)
      //  new_dist_params[k] = dist_params[k] - real_alpha * dist_grad[k];
      Math1D::go_in_neg_direction(new_dist_params, dist_params, dist_grad, real_alpha);

      new_dist_grouping_param = dist_grouping_param - real_alpha * dist_grouping_grad;

      assert(new_dist_params.size() >= 2 * redpar_limit + 1);

      projection_on_simplex_with_slack(new_dist_params.direct_access() + zero_offset - redpar_limit, new_dist_grouping_param,
                                       2 * redpar_limit + 1, hmm_min_param_entry);
    }
    //std::cerr << "D" << std::endl;

    //std::cerr << "E" << std::endl;

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //for (uint k = 0; k < dict[i].size(); k++)
      //  new_dict_prob[i][k] = dict[i][k] - real_alpha * dict_grad[i][k];

      Math1D::go_in_neg_direction(new_dict_prob[i], dict[i], dict_grad[i], real_alpha);
    }

    //std::cerr << "params: " << new_init_params << std::endl;

    for (uint I = 1; I <= maxI; I++) {

      if (init_grad[I - 1].size() > 0) {

        if (init_type == HmmInitPar) {

          double sum = new_init_params.range_sum(0, I);

          if (sum > 1e-305) {
            for (uint k = 0; k < I; k++) {
              new_init_prob[I - 1][k] = new_source_fert[1] * new_init_params[k] / sum;
              assert(!isnan(new_init_prob[I - 1][k]));
            }
          }
          else
            new_init_prob[I - 1].set_constant(new_source_fert[1] / I);

          if (start_empty_word)
            new_init_prob[I - 1][I] = new_source_fert[0];
          else {
            for (uint k = I; k < 2 * I; k++) {
              new_init_prob[I - 1][k] = new_source_fert[0] / I;
              assert(!isnan(new_init_prob[I - 1][k]));
            }
          }
        }
        else if (init_type != HmmInitFix && init_type == HmmInitFix2) {
          //for (uint k = 0; k < initial_prob[I - 1].size(); k++)
          //  new_init_prob[I - 1][k] = initial_prob[I - 1][k] - alpha * init_grad[I - 1][k];

          if (new_init_prob[I - 1].size() > 0)
            Math1D::go_in_neg_direction(new_init_prob[I - 1], initial_prob[I - 1], init_grad[I -1], alpha);
        }

        if (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2) {
          //for (uint k = 0; k < align_model[I - 1].size(); k++)
          //  new_align_prob[I - 1].direct_access(k) = align_model[I - 1].direct_access(k) - alpha * align_grad[I - 1].direct_access(k);

          if (align_model[I - 1].size() > 0)
            Math2D::go_in_neg_direction(new_align_prob[I - 1], align_model[I - 1], align_grad[I - 1], alpha);
        }
      }
    }

    if (dict_weight_sum > 0.0)
      new_slack_vector = slack_vector;

    /******** 3. reproject on the simplices [Michelot 1986] *********/

    //std::cerr << "reproject" << std::endl;

    for (uint i = 0; i < options.nTargetWords_; i++) {

      for (uint k = 0; k < new_dict_prob[i].size(); k++) {
        assert(!isnan(new_dict_prob[i][k]));
        if (new_dict_prob[i][k] < -1e75)
          new_dict_prob[i][k] = -9e74;
        if (new_dict_prob[i][k] > 1e75)
          new_dict_prob[i][k] = 9e74;
      }

      projection_on_simplex_with_slack(new_dict_prob[i].direct_access(), new_slack_vector[i], new_dict_prob[i].size(), hmm_min_dict_entry);
    }

    //std::cerr << "new params: " << new_dist_params << std::endl;

    //std::cerr << "F" << std::endl;

    for (uint I = 1; I <= maxI; I++) {

      if (initial_prob[I - 1].size() > 0) {

        if (init_type == HmmInitNonpar) {
          for (uint k = 0; k < new_init_prob[I - 1].size(); k++) {
            if (new_init_prob[I - 1][k] <= -1e75)
              new_init_prob[I - 1][k] = -9e74;
            if (new_init_prob[I - 1][k] >= 1e75)
              new_init_prob[I - 1][k] = 9e74;

            if (!(fabs(new_init_prob[I - 1][k]) < 1e75))
              std::cerr << "prob: " << new_init_prob[I - 1][k] << std::endl;

            assert(fabs(new_init_prob[I - 1][k]) < 1e75);
          }

          projection_on_simplex(new_init_prob[I - 1].direct_access(), new_init_prob[I - 1].size(), hmm_min_param_entry);
        }

        if (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2) {

          for (uint k = 0; k < new_align_prob[I - 1].size(); k++) {
            if (new_align_prob[I - 1].direct_access(k) <= -1e75)
              new_align_prob[I - 1].direct_access(k) = -9e74;
            if (new_align_prob[I - 1].direct_access(k) >= 1e75)
              new_align_prob[I - 1].direct_access(k) = 9e74;

            assert(fabs(new_align_prob[I - 1].direct_access(k)) < 1e75);
          }

          for (uint y = 0; y < align_model[I - 1].yDim(); y++) {

            const uint dim = (align_type == HmmAlignProbNonpar) ? align_model[I - 1].xDim() : align_model[I - 1].xDim() - 1;
            projection_on_simplex(new_align_prob[I - 1].row_ptr(y), dim, hmm_min_param_entry);
          }
        }
      }
    }

    /******** 4. find a suitable stepsize *********/

    //std::cerr << "G" << std::endl;

    //std::cerr << "previous source fert prob: " << source_fert << std::endl;
    //std::cerr << "new source fert prob: " << new_source_fert << std::endl;

    //evaluating the full step-size is usually a waste of running time
    double hyp_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = 1.0;

    bool decreasing = true;

    uint nInnerIter = 0;

    while (hyp_energy > energy || decreasing) {

      nInnerIter++;

      lambda *= line_reduction_factor;

      const double neg_lambda = 1.0 - lambda;

      for (uint i = 0; i < options.nTargetWords_; i++) {

        //for (uint k = 0; k < dict[i].size(); k++)
        //  hyp_dict_prob[i][k] = lambda * new_dict_prob[i][k] + neg_lambda * dict[i][k];
        Math1D::assign_weighted_combination(hyp_dict_prob[i], neg_lambda, dict[i], lambda, new_dict_prob[i]);
      }

      if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {

        for (uint i = 0; i < 2; i++)
          hyp_source_fert[i] = lambda * new_source_fert[i] + neg_lambda * source_fert[i];
      }

      if (init_type == HmmInitPar) {
        //for (uint k = 0; k < hyp_init_params.size(); k++)
        //  hyp_init_params[k] = lambda * new_init_params[k] + neg_lambda * init_params[k];
        Math1D::assign_weighted_combination(hyp_init_params, lambda, new_init_params, neg_lambda, init_params);

        par2nonpar_hmm_init_model(hyp_init_params, hyp_source_fert, init_type, hyp_init_prob, start_empty_word, options.fix_p0_);
      }
      else if (init_type == HmmInitNonpar) {

        for (uint I = 1; I <= maxI; I++) {

          //for (uint k = 0; k < initial_prob[I - 1].size(); k++)
          //  hyp_init_prob[I - 1][k] = lambda * new_init_prob[I - 1][k] + neg_lambda * initial_prob[I - 1][k];

          if (hyp_init_prob[I - 1].size() > 0)
            Math1D::assign_weighted_combination(hyp_init_prob[I - 1], lambda, new_init_prob[I - 1], neg_lambda, initial_prob[I - 1]);
        }
      }

      if (align_type == HmmAlignProbReducedpar)
        hyp_dist_grouping_param = lambda * new_dist_grouping_param + neg_lambda * dist_grouping_param;

      if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {

        //for (uint k = 0; k < dist_params.size(); k++)
        //  hyp_dist_params[k] = lambda * new_dist_params[k] + neg_lambda * dist_params[k];
        Math1D::assign_weighted_combination(hyp_dist_params, lambda, new_dist_params, neg_lambda, dist_params);

        par2nonpar_hmm_alignment_model(hyp_dist_params, zero_offset, hyp_dist_grouping_param, hyp_source_fert,
                                       align_type, options.deficient_, hyp_align_prob, redpar_limit);

        //std::cerr << "hyp_dist_params: " << hyp_dist_params << std::endl;
        //std::cerr << "hyp source fert: " << hyp_source_fert << std::endl;
      }
      else {

        //combination for Nonpar2 is also correct

        for (uint I = 1; I <= maxI; I++) {
          //for (uint k = 0; k < hyp_align_prob[I - 1].size(); k++)
          //  hyp_align_prob[I - 1].direct_access(k) = lambda * new_align_prob[I - 1].direct_access(k)
          //      + neg_lambda * align_model[I - 1].direct_access(k);

          if (hyp_align_prob[I - 1].size() > 0)
            Math2D::assign_weighted_combination(hyp_align_prob[I - 1], lambda, new_align_prob[I - 1], neg_lambda, align_model[I - 1]);
        }
      }

      double new_energy = extended_hmm_energy(source, slookup, target, hyp_align_prob,
                                              hyp_init_prob, hyp_dict_prob, wcooc, nSourceWords, prior_weight, options, dict_weight_sum);

      std::cerr << "new: " << new_energy << ", prev: " << hyp_energy << std::endl;

      //double prev_energy = hyp_energy;

      if (new_energy < hyp_energy) {

        hyp_energy = new_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      //       if (new_energy > 0.99*prev_energy && hyp_energy < energy)
      //        break;
    }

    //EXPERIMENTAL
    if (nInnerIter > 4)
      alpha *= 1.5;
    //END_EXPERIMENTAL

    if (nInnerIter > 3) {
      nSuccessiveReductions++;
    }
    else {
      nSuccessiveReductions = 0;
    }

    if (nSuccessiveReductions > 3) {
      line_reduction_factor *= 0.9;
      nSuccessiveReductions = 0;
    }

    if (nInnerIter > 5) {
      line_reduction_factor *= 0.75;
      nSuccessiveReductions = 0;
    }

    energy = hyp_energy;

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //for (uint k = 0; k < dict[i].size(); k++)
      //  dict[i][k] = best_lambda * new_dict_prob[i][k] + neg_best_lambda * dict[i][k];

      Math1D::assign_weighted_combination(dict[i], neg_best_lambda, dict[i], best_lambda, new_dict_prob[i]);
    }
    if (dict_weight_sum > 0.0)
      Math1D::assign_weighted_combination(slack_vector, neg_best_lambda, slack_vector, best_lambda, new_slack_vector);

    if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {

      for (uint i = 0; i < 2; i++)
        source_fert[i] = best_lambda * new_source_fert[i] + neg_best_lambda * source_fert[i];
    }

    if (init_type == HmmInitPar) {

      //for (uint k = 0; k < init_params.size(); k++)
      //  init_params[k] = best_lambda * new_init_params[k] + neg_best_lambda * init_params[k];
      Math1D::assign_weighted_combination(init_params, best_lambda, new_init_params, neg_best_lambda, init_params);

      par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word, options.fix_p0_);
    }
    else if (init_type == HmmInitNonpar) {
      for (uint I = 1; I <= maxI; I++) {

        //for (uint k = 0; k < initial_prob[I - 1].size(); k++)
        //  initial_prob[I - 1][k] = best_lambda * new_init_prob[I - 1][k] + neg_best_lambda * initial_prob[I - 1][k];

        if (initial_prob[I - 1].size() > 0)
          Math1D::assign_weighted_combination(initial_prob[I - 1], best_lambda, new_init_prob[I - 1], neg_best_lambda, initial_prob[I - 1]);
      }
    }

    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {

      //for (uint k = 0; k < dist_params.size(); k++)
      //  dist_params[k] = best_lambda * new_dist_params[k] + neg_best_lambda * dist_params[k];
      Math1D::assign_weighted_combination(dist_params, best_lambda, new_dist_params, neg_best_lambda, dist_params);

      if (align_type == HmmAlignProbReducedpar)
        dist_grouping_param = best_lambda * new_dist_grouping_param + neg_best_lambda * dist_grouping_param;

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, options.deficient_, align_model, redpar_limit);
    }
    else {
      for (uint I = 1; I <= maxI; I++) {

        //for (uint k = 0; k < align_model[I - 1].size(); k++)
        //  align_model[I - 1].direct_access(k) = best_lambda * new_align_prob[I - 1].direct_access(k)
        //                                        + neg_best_lambda * align_model[I - 1].direct_access(k);

        if (align_model[I - 1].size() > 0)
          Math2D::assign_weighted_combination(align_model[I - 1], best_lambda, new_align_prob[I - 1], neg_best_lambda, align_model[I - 1]);
      }
    }

    if (options.print_energy_) {
      std::cerr << "slack-sum: " << slack_vector.sum() << std::endl;
      std::cerr << "#### EHMM energy after gd-iteration # " << iter << ": " << "energy: " << energy << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double sum_marg_aer = 0.0;
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
        //compute viterbi alignment

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        Storage1D<AlignBaseType> viterbi_alignment;
        const uint curI = target[s].size();

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        compute_ehmm_viterbi_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                       initial_prob[curI - 1], viterbi_alignment, options);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        Storage1D < AlignBaseType > marg_alignment;
        compute_ehmm_optmarginal_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                           initial_prob[curI - 1], start_empty_word, marg_alignment);

        sum_marg_aer += AER(marg_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ehmm_postdec_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                       initial_prob[curI - 1], options, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_marg_aer *= 100.0 / nContributors;
      nErrors /= nContributors;
      sum_fmeasure /= nContributors;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### EHMM Viterbi-AER after gd-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "---- EHMM Marginal-AER : " << sum_marg_aer << " %" << std::endl;
      std::cerr << "#### EHMM Viterbi-fmeasure after gd-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM Viterbi-DAE/S after gd-iteration #" << iter << ": " << nErrors << std::endl;

      std::cerr << "#### EHMM Postdec-AER after gd-iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### EHMM Postdec-fmeasure after gd-iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### EHMM Postdec-DAE/S after gd-iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  } // end  for (iter)
}

void viterbi_train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                FullHMMAlignmentModel& align_model, Math1D::Vector<double>& dist_params,
                                double& dist_grouping_param, Math1D::Vector<double>& source_fert,
                                InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                                SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                const HmmOptions& options, const Math1D::Vector<double>& xlogx_table)
{
  std::cerr << "starting Viterbi Training for Extended HMM" << std::endl;

  uint nIterations = options.nIterations_;
  HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  bool deficient_parametric = (options.deficient_ && align_type == HmmAlignProbFullpar);
  const int redpar_limit = options.redpar_limit_;

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  const uint nSourceWords = options.nSourceWords_;

  SingleLookupTable aux_lookup;

  for (size_t s = 0; s < nSentences; s++) {

    viterbi_alignment[s].resize(source[s].size());
  }
  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dictionary is assumed to be initialized

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  //std::cerr << "init_type: " << init_type << std::endl;

  init_hmm_from_prev(source, slookup, target, dict, wcooc, align_model, dist_params, dist_grouping_param, source_fert,
                     initial_prob, init_params, options, options.transfer_mode_);

  const uint maxI = align_model.size();
  const uint zero_offset = maxI - 1;

  Math1D::Vector<double> source_fert_count(2, 0.0);

  NamedStorage1D<Math1D::Vector<double> > dcount(options.nTargetWords_, MAKENAME(dcount));
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dcount[i].resize(dict[i].size());
  }

  InitialAlignmentProbability icount(maxI, MAKENAME(icount));
  icount = initial_prob;

  Math1D::Vector<double> fsentence_start_count(maxI);
  Math1D::Vector<double> fstart_span_count(maxI);

  FullHMMAlignmentModel acount(maxI, MAKENAME(acount));
  for (uint I = 1; I <= maxI; I++) {
    if (align_model[I - 1].size() > 0)
      acount[I - 1].resize_dirty(I + 1, I);
  }

  Math1D::NamedVector<double> dist_count(MAKENAME(dist_count));       //used in deficient mode
  dist_count = dist_params;

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting Viterbi-EHMM iteration #" << iter << std::endl;

    //DEBUG
    //for (uint I=0; I < 10; I++)
    //  std::cerr << "init prob: " << initial_prob[I] << std::endl;
    //END_DEBUG

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i = 0; i < options.nTargetWords_; i++) {
      dcount[i].set_constant(0);
    }

    for (uint I = 1; I <= maxI; I++) {
      acount[I - 1].set_constant(0.0);
      icount[I - 1].set_constant(0.0);
    }

    source_fert_count.set_constant(0.0);
    dist_count.set_constant(0.0);

    fsentence_start_count.set_constant(0.0);
    fstart_span_count.set_constant(0.0);

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "sentence #" << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      Storage1D<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      const Math2D::Matrix<double>& cur_align_model = align_model[curI - 1];
      Math2D::Matrix<double>& cur_facount = acount[curI - 1];

      //std::cerr << " passing initial prob " << initial_prob[curI-1] << std::endl;

      long double prob = compute_ehmm_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model,
                         initial_prob[curI - 1], cur_alignment, options, true, false, 0.0);

      prev_perplexity -= logl(prob);

      if (!(prob > 0.0)) {

        std::cerr << "sentence_prob " << prob << " for sentence pair " << s << " with I=" << curI << ", J= " << curJ << std::endl;
      }
      assert(prob > 0.0);

      /**** update counts ****/
      for (uint j = 0; j < curJ; j++) {

        const ushort aj = cur_alignment[j];
        if (aj >= curI) {
          dcount[0][cur_source[j] - 1] += 1;
        }
        else {
          dcount[cur_target[aj]][cur_lookup(j, aj)] += 1;
        }

        if (j == 0) {
          if (!start_empty_word)
            icount[curI - 1][aj] += 1.0;
          else {
            if (aj < curI)
              icount[curI - 1][aj] += 1.0;
            else {
              assert(aj == 2 * curI);
              icount[curI - 1][curI] += 1.0;
            }
          }
        }
        else {

          const ushort prev_aj = cur_alignment[j - 1];

          if (prev_aj == 2 * curI) {
            assert(start_empty_word);
            if (aj == prev_aj)
              icount[curI - 1][curI] += 1.0;
            else
              icount[curI - 1][aj] += 1.0;
          }
          else if (prev_aj >= curI) {

            if (aj >= curI) {
              cur_facount(curI, prev_aj - curI) += 1.0;
            }
            else {
              cur_facount(aj, prev_aj - curI) += 1.0;

              if (deficient_parametric) {

                uint diff = zero_offset - (prev_aj - curI);
                diff += (aj >= curI) ? aj - curI : aj;

                dist_count[diff] += 1.0;
              }
            }
          }
          else {
            if (aj >= curI) {
              cur_facount(curI, prev_aj) += 1.0;
            }
            else {
              cur_facount(aj, prev_aj) += 1.0;

              if (deficient_parametric) {

                uint diff = zero_offset - prev_aj;
                diff += (aj >= curI) ? aj - curI : aj;

                dist_count[diff] += 1.0;
              }
            }
          }
        }
      }
    }  // loop over sentences finished

    //finish counts

    if (init_type == HmmInitPar) {

      for (uint I = 1; I <= maxI; I++) {

        if (initial_prob[I - 1].size() != 0) {
          for (uint i = 0; i < I; i++) {

            const double cur_count = icount[I - 1][i];
            source_fert_count[1] += cur_count;
            fsentence_start_count[i] += cur_count;
            fstart_span_count[I - 1] += cur_count;
          }
          for (uint i = I; i < icount[I - 1].size(); i++) {
            source_fert_count[0] += icount[I - 1][i];
          }
        }
      }
    }

    if (align_type != HmmAlignProbNonpar) {

      //compute the expectations of the parameters from the expectations of the models

      for (uint I = 1; I <= maxI; I++) {

        if (align_model[I - 1].xDim() != 0) {

          for (int i = 0; i < (int)I; i++) {

            for (int ii = 0; ii < (int)I; ii++) {
              source_fert_count[1] += acount[I - 1] (ii, i);
            }
            source_fert_count[0] += acount[I - 1] (I, i);
          }
        }
      }
    }

    //include the dict_regularity term in the output energy
    if (options.print_energy_ ) {
      double energy = prev_perplexity;
      if (dict_weight_sum > 0.0) {
        for (uint i = 0; i < dcount.size(); i++)
          for (uint k = 0; k < dcount[i].size(); k++)
            if (dcount[i][k] > 0)
              //we need to divide as we are truly minimizing the perplexity WITHOUT division plus the l0-term
              energy += prior_weight[i][k];
      }

      std::cerr << "energy after iteration #" << (iter - 1) << ": " << (energy / nSentences) << std::endl;
    }
    //std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    //no need to update dict here

    double sfsum = source_fert_count.sum();
    if (sfsum > 1e-305 && !options.fix_p0_) {
      for (uint k = 0; k < 2; k++)
        source_fert[k] = source_fert_count[k] / sfsum;
    }

    if (init_type == HmmInitPar) {

      start_prob_m_step(fsentence_start_count, fstart_span_count, init_params, options.init_m_step_iter_);
      par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word, options.fix_p0_);
    }
    else if (init_type == HmmInitNonpar) {

      for (uint I = 1; I <= maxI; I++) {
        if (icount[I - 1].size() > 0) {
          double sum = icount[I - 1].sum();
          if (sum > 1e-305) {
            for (uint k = 0; k < initial_prob[I - 1].size(); k++)
              initial_prob[I - 1][k] = std::max(hmm_min_param_entry, icount[I - 1][k] / sum);
          }
        }
      }
    }

    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {

      //also applies to deficient_parametric

      //call m-step
      //noncompact_ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param, options.deficient, redpar_limit);
      ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_,
                  dist_grouping_param, options.deficient_, redpar_limit, options.projection_mode_);

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, options.deficient_, align_model, redpar_limit);
    }
    else {

      if (align_type == HmmAlignProbNonpar) {

        for (uint I = 1; I <= maxI; I++) {

          if (align_model[I - 1].xDim() != 0) {

            for (uint i = 0; i < I; i++) {

              double sum = acount[I - 1].row_sum(i);

              if (sum >= 1e-300) {

                const double inv_sum = 1.0 / sum;
                assert(!isnan(inv_sum));

                for (uint i_next = 0; i_next <= I; i_next++) {
                  align_model[I - 1](i_next, i) = std::max(hmm_min_param_entry, inv_sum * acount[I - 1](i_next, i));
                }
              }
            }
          }
        }
      }
      else if (align_type == HmmAlignProbNonpar2) {

        for (uint I = 1; I <= maxI; I++) {

          if (align_model[I - 1].xDim() != 0) {

            for (uint i = 0; i < I; i++) {

              double sum = acount[I - 1].row_sum(i) - acount[I - 1] (I, i);

              align_model[I - 1] (I, i) = source_fert[0];

              if (sum >= 1e-300) {

                const double inv_sum = 1.0 / sum;
                assert(!isnan(inv_sum));

                for (uint i_next = 0; i_next < I; i_next++) {
                  align_model[I - 1](i_next, i) = std::max(hmm_min_param_entry, inv_sum * source_fert[1] * acount[I - 1](i_next, i));
                  //          if (isnan(align_model[I-1](i_next,i)))
                  //            std::cerr << "nan: " << inv_sum << " * " << acount[I-1](i_next,i) << std::endl;

                  //          assert(!isnan(align_model[I-1](i_next,i)));
                }
              }
              else {
                for (uint i_next = 0; i_next < I; i_next++) {
                  align_model[I - 1](i_next, i) = source_fert[1] / double (I);
                }
              }
            }
          }
        }
      }
    }

    std::cerr << "source_fert_count before ICM: " << source_fert_count << std::endl;

#if 1
    std::cerr << "starting ICM stage" << std::endl;

    /**** ICM stage ****/

    uint nSwitches = 0;

    int dist_count_sum = dist_count.sum();

    Math1D::Vector<uint> dict_sum(dcount.size());
    for (uint k = 0; k < dcount.size(); k++)
      dict_sum[k] = dcount[k].sum();

    for (size_t s = 0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const ushort curJ = cur_source.size();
      const ushort curI = cur_target.size();
      const ushort nLabels = (start_empty_word) ? 2 * curI + 1 : 2 * curI;

      Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      const Math2D::Matrix<double>& cur_align_model = align_model[curI - 1];
      const Math1D::Vector<double>& cur_initial_prob = initial_prob[curI - 1];
      Math2D::Matrix<double>& cur_acount = acount[curI - 1];
      Math1D::Vector<double>& cur_icount = icount[curI - 1];

      //std::cerr << "s: " << s << ", I = " << curI << ", J = " << curJ << std::endl;

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      for (uint j = 0; j < curJ; j++) {

        //std::cerr << "j: " << j << std::endl;

        const ushort cur_aj = cur_alignment[j];
        ushort new_aj = cur_aj; // is argbest_change

        uint effective_cur_aj = cur_aj;
        if (effective_cur_aj >= curI)
          effective_cur_aj -= curI;

        uint effective_prev_aj = MAX_UINT;
        if (j > 0) {
          effective_prev_aj = cur_alignment[j - 1];
          if (effective_prev_aj >= curI)
            effective_prev_aj -= curI;
        }

        uint effective_next_aj = MAX_UINT;
        if (j + 1 < curJ) {
          for (uint jj = j + 1; jj < curJ; jj++) {
            if (cur_alignment[jj] < curI) {
              effective_next_aj = cur_alignment[jj];
              break;
            }
          }
        }
        //std::cerr << "cur_aj: " << cur_aj << std::endl;

        double best_change = 1e300;

        const uint cur_target_word = (cur_aj >= curI) ? 0 : cur_target[cur_aj];
        const uint cur_idx = (cur_aj >= curI) ? cur_source[j] - 1 : cur_lookup(j, cur_aj);

        Math1D::Vector<double>& cur_dictcount = dcount[cur_target_word];
        const uint cur_dictsum = dict_sum[cur_target_word];

        double common_change = 0.0;     //NOTE: only included if the target word is different
        if (cur_dictsum > 1) {
          //exploit log(1) = 0
          common_change -= xlogx_table[cur_dictsum];
          common_change += xlogx_table[cur_dictsum - 1];
        }

        //prior_weight is always relevant
        if (cur_dictcount[cur_idx] > 1) {
          //exploit log(1) = 0
          common_change -= -xlogx_table[cur_dictcount[cur_idx]];
          common_change += -xlogx_table[cur_dictcount[cur_idx] - 1];
        }
        else
          common_change -= prior_weight[cur_target_word][cur_idx];

        for (ushort i = 0; i < nLabels; i++) {

          //std::cerr << "i: " << i << std::endl;

          if (i == cur_aj)
            continue;
          if (start_empty_word && j == 0 && i >= curI && i < 2 * curI)
            continue;

          if (j > 0 && i >= curI) {

            ushort prev_aj = cur_alignment[j - 1];
            if (i != prev_aj && i != prev_aj + curI)
              continue;
            if (i == 2 * curI && prev_aj != i)  //need this!
              continue;
          }

          bool fix = true;

          uint effective_i = i;
          if (effective_i >= curI)
            effective_i -= curI;

          if (j + 1 < curJ) {

            uint next_aj = cur_alignment[j + 1];

            //TODO: handle these special cases (some distributions are doubly affected)
            if (align_type == HmmAlignProbNonpar
                || align_type == HmmAlignProbNonpar2) {

              fix = false;

              if (align_type == HmmAlignProbNonpar2
                  && std::max(cur_aj, i) >= curI)
                fix = true;

              if (next_aj >= curI)
                fix = true;
              if (effective_prev_aj == effective_cur_aj)
                fix = true;
              if (effective_prev_aj == effective_i)
                fix = true;
            }
            else if (deficient_parametric) {

              fix = (effective_i == curI || effective_cur_aj == curI);

              if (j > 0 && effective_prev_aj == curI)
                fix = true;

              if (next_aj >= curI)
                fix = true;
              if (effective_cur_aj - effective_prev_aj ==
                  next_aj - effective_cur_aj)
                fix = true;
              if (effective_cur_aj - effective_prev_aj == next_aj - effective_i)
                fix = true;
              if (effective_cur_aj - effective_i == next_aj - effective_cur_aj)
                fix = true;
              if (effective_cur_aj - effective_i == next_aj - effective_i)
                fix = true;
            }
          }

          const uint new_target_word = (i >= curI) ? 0 : cur_target[i];
          const uint hyp_idx = (i >= curI) ? source[s][j] - 1 : cur_lookup(j, i);

          double change = 0.0;

          if (cur_target_word != new_target_word) {

            if (dict_sum[new_target_word] > 0) {
              //exploit log(1) = 0
              change -= xlogx_table[dict_sum[new_target_word]];
              change += xlogx_table[dict_sum[new_target_word] + 1];
            }

            //prior_weight is always relevant
            if (dcount[new_target_word][hyp_idx] > 0) {
              //exploit log(1) = 0
              change -= -xlogx_table[dcount[new_target_word][hyp_idx]];
              change += -xlogx_table[dcount[new_target_word][hyp_idx] + 1];
            }
            else
              change += prior_weight[new_target_word][hyp_idx];

            change += common_change;
          }

          //a) regarding preceeding pos
          if (j == 0) {

            if (!start_empty_word && init_type == HmmInitNonpar) {
              assert(icount[curI - 1][cur_aj] > 0);

              if (cur_icount[i] > 0) {
                //exploit log(1) = 0
                change -= -xlogx_table[cur_icount[i]];
                change += -xlogx_table[cur_icount[i] + 1];
              }

              if (cur_icount[cur_aj] > 1) {
                //exploit log(1) = 0
                change -= -xlogx_table[cur_icount[cur_aj]];
                change += -xlogx_table[cur_icount[cur_aj] - 1];
              }
            }
            else if (init_type != HmmInitFix) {

              if (cur_aj != 2 * curI)
                change += std::log(cur_initial_prob[cur_aj]);
              else
                change += std::log(cur_initial_prob[curI]);
              if (i != 2 * curI)
                change -= std::log(cur_initial_prob[i]);
              else
                change -= std::log(cur_initial_prob[curI]);
            }
          }
          else {
            // j > 0

            //note: the total sum of counts for effective_prev_aj stays constant in this operation
            if (!fix && (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2)) {

              if (effective_prev_aj < curI) {
                const int cur_c = cur_acount(std::min<ushort>(curI, cur_aj), effective_prev_aj);
                const int new_c = cur_acount(std::min<ushort>(curI, i), effective_prev_aj);
                assert(cur_c > 0);

                if (cur_c > 1) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_c];
                  change += -xlogx_table[cur_c - 1];
                }

                if (new_c > 0) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[new_c];
                  change += -xlogx_table[new_c + 1];
                }
              }
              else {
                change -= -std::log(cur_initial_prob[std::min<ushort>(curI, cur_aj)]);
                change += -std::log(cur_initial_prob[std::min<ushort>(curI, i)]);
              }
            }
            else {              //parametric model

              if (!fix && deficient_parametric) {
                if (cur_aj < curI) {

                  const int cur_c = dist_count[zero_offset + cur_aj - effective_prev_aj];

                  if (cur_c > 1) {
                    //exploit that log(1) = 0
                    change -= -xlogx_table[cur_c];
                    change += -xlogx_table[cur_c - 1];
                  }
                }
              }
              else {
                if (effective_prev_aj < curI)
                  change -= -std::log(cur_align_model(std::min < ushort > (curI, cur_aj), effective_prev_aj));
                else
                  change -= -std::log(cur_initial_prob[effective_cur_aj]);
              }

              if (!fix && deficient_parametric) {

                if (i < curI) {

                  const int cur_c = dist_count[zero_offset + i - effective_prev_aj];
                  if (cur_c > 0) {
                    //exploit that log(1) = 0
                    change -= -xlogx_table[cur_c];
                    change += -xlogx_table[cur_c + 1];
                  }
                }
              }
              else {
                if (effective_prev_aj < curI)
                  change += -std::log(cur_align_model(std::min(curI, i), effective_prev_aj));
                else
                  change += -std::log(cur_initial_prob[effective_i]);
              }

              //source fertility counts
              if (!fix && deficient_parametric) {
                if (cur_aj < curI && i >= curI) {

                  if (!options.fix_p0_) {
                    const int cur_c0 = source_fert_count[0];
                    const int cur_c1 = source_fert_count[1];

                    if (cur_c0 > 0) {
                      //exploit that log(1) = 0
                      change -= -xlogx_table[cur_c0];
                      change += -xlogx_table[cur_c0 + 1];
                    }
                    if (cur_c1 > 1) {
                      //exploit log(1) = 0
                      change -= -xlogx_table[cur_c1];
                      change += -xlogx_table[cur_c1 - 1];
                    }
                  }

                  change -= xlogx_table[dist_count_sum];
                  change += xlogx_table[dist_count_sum - 1];
                }
                else if (cur_aj >= curI && i < curI) {

                  if (!options.fix_p0_) {
                    const int cur_c0 = source_fert_count[0];
                    const int cur_c1 = source_fert_count[1];

                    if (cur_c1 > 0) {
                      //exploit log(1) = 0
                      change += -xlogx_table[cur_c1 + 1];
                      change -= -xlogx_table[cur_c1];
                    }
                    if (cur_c0 > 1) {
                      //exploit log(1) = 0
                      change -= -xlogx_table[cur_c0];
                      change += -xlogx_table[cur_c0 - 1];
                    }
                  }

                  change -= xlogx_table[dist_count_sum];
                  change += xlogx_table[dist_count_sum + 1];
                }
              }
            }
          }

          assert(!isnan(change));

          //std::cerr << "b)" << std::endl;

          //b) regarding succeeding pos
          if (j + 1 < curJ && effective_cur_aj != effective_i) {

            if (!fix && (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2)) {

              const uint limit = (align_type == HmmAlignProbNonpar) ? curI : curI - 1;

              if (effective_cur_aj < curI) {
                int total_cur_count = 0;
                for (uint k = 0; k <= limit; k++)
                  total_cur_count += cur_acount(k, effective_cur_aj);

                assert(total_cur_count > 0);
                assert(cur_acount(effective_next_aj, effective_cur_aj) > 0);

                if (total_cur_count > 1) {
                  //exploit log(1) = 0
                  change -= xlogx_table[total_cur_count];
                  change += xlogx_table[total_cur_count - 1];
                }

                if (cur_acount(effective_next_aj, effective_cur_aj) > 1) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_acount(effective_next_aj, effective_cur_aj)];
                  change += -xlogx_table[cur_acount(effective_next_aj, effective_cur_aj) - 1];
                }
              }
              else {
                change -= -std::log(initial_prob[curI - 1][curI]);
              }

              if (effective_i < curI) {
                int total_new_count = 0;
                for (uint k = 0; k <= limit; k++)
                  total_new_count += cur_acount(k, effective_i);

                if (total_new_count > 0) {
                  //exploit log(1) = 0
                  change -= xlogx_table[total_new_count];
                  change += xlogx_table[total_new_count + 1];
                }
                if (cur_acount(effective_next_aj, effective_i) > 0) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_acount(effective_next_aj, effective_i)];
                  change += -xlogx_table[cur_acount(effective_next_aj, effective_i) + 1];
                }
              }
              else {

                change += -std::log(initial_prob[curI - 1][curI]);
              }

              assert(!isnan(change));
            }
            else {
              //parametric model

              if (!fix && deficient_parametric) {
                assert(j + 1 < curJ);
                assert(effective_i < curI && effective_cur_aj < curI);

                if (cur_alignment[j + 1] < curI) {

                  const int cur_c = dist_count[zero_offset + effective_next_aj - effective_cur_aj];
                  const int new_c = dist_count[zero_offset + effective_next_aj - effective_i];

                  if (new_c > 0) {
                    //exploit log(1) = 0
                    change -= -xlogx_table[new_c];
                    change += -xlogx_table[new_c + 1];
                  }

                  if (cur_c > 1) {
                    //exploit log(1) = 0
                    change -= -xlogx_table[cur_c];
                    change += -xlogx_table[cur_c - 1];
                  }
                }
                else {
                  std::cerr << "WARNING: this case should not occur" << std::endl;
                }
              }
              else {
                if (effective_next_aj != MAX_UINT) {
                  if (effective_cur_aj != curI) {
                    change -= -std::log(cur_align_model(effective_next_aj, effective_cur_aj));
                    if (effective_i != curI)
                      change += -std::log(cur_align_model(effective_next_aj, effective_i));
                    else
                      change += -std::log(initial_prob[curI - 1][effective_next_aj]);
                  }
                  else {
                    change -= -std::log(initial_prob[curI - 1][curI]);
                    change += -std::log(cur_align_model(effective_next_aj, effective_i));
                  }
                }
                else if (start_empty_word && j + 1 < curJ
                         && effective_prev_aj == curI
                         && effective_cur_aj == curI) {
                  assert(effective_i != curI);
                  change -= -std::log(initial_prob[curI - 1][curI]);
                  change += -std::log(initial_prob[curI - 1][effective_i]);
                }
              }
            }
          }

          assert(!isnan(change));

          if (change < best_change) {

            best_change = change;
            new_aj = i;
          }
        }

        //std::cerr << "base alignment: " << cur_alignment << std::endl;
        //std::cerr << "best_change: " << best_change << std::endl;

        if (best_change < -0.01 && new_aj != cur_aj) {

          nSwitches++;

          const uint new_target_word = (new_aj >= curI) ? 0 : cur_target[new_aj];
          const uint hyp_idx = (new_aj >= curI) ? cur_source[j] - 1 : cur_lookup(j, new_aj);

          cur_alignment[j] = new_aj;

          if (j > 0 && new_aj >= curI)
            assert(new_aj == cur_alignment[j - 1] || new_aj - curI == cur_alignment[j - 1]);

          ushort effective_new_aj = new_aj;
          if (effective_new_aj >= curI)
            effective_new_aj -= curI;

          bool future_handled = false;

          if (j + 1 < curJ) {

            ushort next_aj = cur_alignment[j + 1];

            //std::cerr << "next_aj: " << next_aj << std::endl;

            if (next_aj >= curI) {

              ushort new_next_aj = (new_aj < curI) ? new_aj + curI : new_aj;

              //std::cerr << "new_next_aj: " << new_next_aj << std::endl;

              if (new_next_aj != next_aj) {

                //std::cerr << "setting future handled" << std::endl;
                future_handled = true;

                if (cur_aj != 2 * curI) {
                  cur_acount(curI, effective_cur_aj)--;
                  assert(cur_acount(curI, effective_cur_aj) >= 0);
                }
                else
                  cur_icount[curI]--;
                if (new_aj != 2 * curI)
                  cur_acount(curI, effective_new_aj)++;
                else
                  cur_icount[curI]++;

                uint jj = j + 1;
                for (; jj < curJ; jj++) {
                  if (cur_alignment[jj] == next_aj) {
                    cur_alignment[jj] = new_next_aj;
                    if (jj > j + 1) {
                      if (next_aj != 2 * curI) {
                        cur_acount(curI, next_aj - curI)--;
                        assert(cur_acount(curI, next_aj - curI) >= 0);
                      }
                      else
                        cur_icount[curI]--;
                      if (new_next_aj != 2 * curI)
                        cur_acount(curI, new_next_aj - curI)++;
                      else
                        cur_icount[curI]++;
                    }
                  }
                  else
                    break;
                }
                if (jj < curJ) {
                  assert(cur_alignment[jj] < curI);
                  if (next_aj != 2 * curI) {
                    cur_acount(cur_alignment[jj], next_aj - curI)--;
                    assert(cur_acount(cur_alignment[jj], next_aj - curI) >= 0);
                  }
                  else
                    cur_icount[curI]--;
                  if (new_next_aj != 2 * curI)
                    cur_acount(cur_alignment[jj], new_next_aj - curI)++;
                  else
                    cur_icount[curI]++;
                  if (deficient_parametric) {
                    if (effective_cur_aj < curI) {
                      dist_count[zero_offset + cur_alignment[jj] - effective_cur_aj]--;
                      dist_count_sum--;
                    }
                    if (effective_new_aj < curI) {
                      dist_count[zero_offset + cur_alignment[jj] - effective_new_aj]++;
                      dist_count_sum++;
                    }
                  }
                }
              }
            }
          }
          //recompute the stored values for the two affected words
          if (cur_target_word != new_target_word) {

            Math1D::Vector<double>& hyp_dictcount = dcount[new_target_word];

            cur_dictcount[cur_idx] -= 1;
            hyp_dictcount[hyp_idx] += 1;
            dict_sum[cur_target_word] -= 1;
            dict_sum[new_target_word] += 1;
          }

          /****** change alignment counts *****/

          //std::cerr << "change alignment counts, j=" << j << ", cur_aj: " << cur_aj << ", new_aj: " << new_aj << std::endl;

          //a) dependency to preceeding pos
          if (j == 0) {

            //if (init_type != HmmInitFix) {
            if (true) {
              if (cur_aj != 2 * curI) {
                assert(cur_icount[cur_aj] > 0);
                cur_icount[cur_aj]--;
              }
              else {
                assert(cur_icount[curI] > 0);
                cur_icount[curI]--;
              }

              if (new_aj != 2 * curI)
                cur_icount[new_aj]++;
              else
                cur_icount[curI]++;
            }

            if (init_type == HmmInitPar) {

              if (cur_aj < curI && new_aj >= curI) {
                source_fert_count[1]--;
                source_fert_count[0]++;
              }
              else if (cur_aj >= curI && new_aj < curI) {
                source_fert_count[0]--;
                source_fert_count[1]++;
              }
            }
          }
          else {

            if (effective_prev_aj != curI) {
              cur_acount(std::min<ushort>(curI, cur_aj), effective_prev_aj)--;
              assert(cur_acount(std::min<ushort >(curI, cur_aj), effective_prev_aj) >= 0);
              cur_acount(std::min<ushort>(curI, new_aj), effective_prev_aj)++;
              assert(cur_acount(std::min<ushort>(curI, new_aj), effective_prev_aj) >= 0);
            }
            else {
              cur_icount[std::min<ushort>(curI, cur_aj)]--;
              cur_icount[std::min<ushort>(curI, new_aj)]++;
            }

            if (deficient_parametric && effective_prev_aj < curI) {

              if (cur_aj < curI) {
                dist_count[zero_offset + cur_aj - effective_prev_aj]--;
                dist_count_sum--;
              }
              if (new_aj < curI) {
                dist_count[zero_offset + new_aj - effective_prev_aj]++;
                dist_count_sum++;
              }
            }

            if (align_type != HmmAlignProbNonpar) {
              if (cur_aj < curI && new_aj >= curI) {
                source_fert_count[1]--;
                source_fert_count[0]++;
              }
              else if (cur_aj >= curI && new_aj < curI) {
                source_fert_count[0]--;
                source_fert_count[1]++;
              }
            }
          }

          //std::cerr << "b), future handled: " << future_handled << std::endl;

          //b) dependency to succceeding pos
          if (j + 1 < curJ && !future_handled) {

            ushort next_aj = cur_alignment[j + 1];
            ushort effective_cur_aj = cur_aj;
            if (effective_cur_aj >= curI)
              effective_cur_aj -= curI;
            ushort effective_new_aj = new_aj;
            if (effective_new_aj >= curI)
              effective_new_aj -= curI;

            if (effective_cur_aj < curI) {
              cur_acount(std::min(curI, next_aj), effective_cur_aj)--;
              assert(cur_acount(std::min(curI, next_aj), effective_cur_aj) >= 0);
            }
            else
              cur_icount[std::min(curI, next_aj)]--;
            if (effective_new_aj < curI)
              cur_acount(std::min(curI, next_aj), effective_new_aj)++;
            else
              cur_icount[std::min(curI, next_aj)]++;

            if (deficient_parametric) {
              if (next_aj < curI) {
                if (effective_cur_aj < curI) {
                  dist_count[zero_offset + next_aj - effective_cur_aj]--;
                  dist_count_sum--;
                }
                if (effective_new_aj < curI) {
                  dist_count[zero_offset + next_aj - effective_new_aj]++;
                  dist_count_sum++;
                }
              }
            }
          }
        }
      }

      // std::cerr << "new prob: " << hmm_alignment_prob(cur_source,cur_lookup,cur_target,dict,
      //                                                      align_model, initial_prob, cur_alignment, true) << std::endl;
    }

    std::cerr << nSwitches << " switches in ICM" << std::endl;

#ifndef NDEBUG
    //DEBUG
    if ((init_type == HmmInitFix || init_type == HmmInitFix2)
        && !start_empty_word) {
      Math1D::Vector<double> check_source_fert_count(2, 0.0);

      Storage1D<Math1D::Vector<double> > check_dcount(options.nTargetWords_);
      for (uint i = 0; i < options.nTargetWords_; i++) {
        check_dcount[i].resize(dict[i].size(), 0);
      }

      FullHMMAlignmentModel check_acount(maxI, MAKENAME(acount));
      for (uint I = 1; I <= maxI; I++) {
        check_acount[I - 1].resize(align_model[I - 1].xDim(), align_model[I - 1].yDim(), 0.0);
      }

      for (size_t s = 0; s < nSentences; s++) {

        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];
        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

        const uint curJ = cur_source.size();
        const uint curI = cur_target.size();

        Math2D::Matrix<double>& cur_facount = check_acount[curI - 1];

        /**** update counts ****/
        for (uint j = 0; j < curJ; j++) {

          ushort aj = viterbi_alignment[s][j];
          if (aj >= curI) {
            check_dcount[0][cur_source[j] - 1] += 1;
            if (align_type != HmmAlignProbNonpar && j > 0)
              check_source_fert_count[0] += 1.0;
          }
          else {
            check_dcount[cur_target[aj]][cur_lookup(j, aj)] += 1;
            if (align_type != HmmAlignProbNonpar && j > 0)
              check_source_fert_count[1] += 1.0;
          }

          if (j == 0) {
          }
          else {

            ushort prev_aj = viterbi_alignment[s][j - 1];

            if (prev_aj >= curI) {

              if (aj >= curI) {
                cur_facount(curI, prev_aj - curI) += 1.0;
              }
              else {
                cur_facount(aj, prev_aj - curI) += 1.0;
              }
            }
            else {
              if (aj >= curI) {
                cur_facount(curI, prev_aj) += 1.0;
              }
              else {
                cur_facount(aj, prev_aj) += 1.0;
              }
            }
          }
        }
      }

      assert(check_dcount == dcount);
      for (uint I = 0; I < acount.size(); I++) {

        if (check_acount[I] != acount[I]) {
          std::cerr << "I: " << I << std::endl;
          std::cerr << "should be: " << check_acount[I] << std::endl;
          std::cerr << "is: " << acount[I] << std::endl;
        }

        assert(check_acount[I] == acount[I]);
      }
      //assert(check_acount == acount);
      std::cerr << "should be: " << check_source_fert_count << std::endl;
      std::cerr << "is: " << source_fert_count << std::endl;

      assert(check_source_fert_count == source_fert_count);
    }
    //END_DEBUG
#endif

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts
    update_dict_from_counts(dcount, prior_weight, nSentences, 0.0, false, 0.0, 0, dict, hmm_min_dict_entry);

    //the changes in source-fert-counts are SO FAR NOT accounted for in the ICM hyp score calculations
    // nevertheless, updating them according to the new counts cannot worsen the energy
    sfsum = source_fert_count.sum();
    if (sfsum > 1e-305 && !options.fix_p0_) {
      for (uint k = 0; k < 2; k++)
        source_fert[k] = source_fert_count[k] / sfsum;
      std::cerr << "new source-fert: " << source_fert << std::endl;
    }

    if (init_type == HmmInitPar) {

      fsentence_start_count.set_constant(0.0);
      fstart_span_count.set_constant(0.0);

      for (uint I = 1; I <= maxI; I++) {

        if (initial_prob[I - 1].size() != 0) {
          for (uint i = 0; i < I; i++) {

            const double cur_count = icount[I - 1][i];
            fsentence_start_count[i] += cur_count;
            fstart_span_count[I - 1] += cur_count;
          }
        }
      }

      start_prob_m_step(fsentence_start_count, fstart_span_count, init_params, options.init_m_step_iter_);
      par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word, options.fix_p0_);
    }

    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {

      //also applies to deficient_parametric

      //noncompact_ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param, options.deficient_, redpar_limit);
      ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_,
                  dist_grouping_param, options.deficient_, redpar_limit, options.projection_mode_);

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, options.deficient_, align_model, redpar_limit);
    }
    //compute new alignment probabilities from normalized fractional counts
    for (uint I = 1; I <= maxI; I++) {

      if (acount[I - 1].xDim() != 0) {

        if (init_type == HmmInitNonpar) {
          double inv_norm = 1.0 / icount[I - 1].sum();
          assert(!isnan(inv_norm));
          for (uint i = 0; i < initial_prob[I - 1].size(); i++)
            initial_prob[I - 1][i] = std::max(hmm_min_param_entry, inv_norm * icount[I - 1][i]);
        }

        if (align_type == HmmAlignProbNonpar) {

          for (uint i = 0; i < I; i++) {

            double sum = acount[I - 1].row_sum(i);

            if (sum >= 1e-300) {

              assert(!isnan(sum));
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));

              for (uint i_next = 0; i_next <= I; i_next++) {
                align_model[I - 1] (i_next, i) = std::max(hmm_min_param_entry, inv_sum * acount[I - 1] (i_next, i));
                //            if (isnan(align_model[I-1](i_next,i)))
                //              std::cerr << "nan: " << inv_sum << " * " << acount[I-1](i_next,i) << std::endl;

                //            assert(!isnan(align_model[I-1](i_next,i)));
              }
            }
          }
        }
        else if (align_type == HmmAlignProbNonpar2) {

          //changes in these counts are so far not included in the ICM score calculation.
          // nevertheless, updating the probabilities according to the new counts cannor worsen the energy

          for (uint i = 0; i < I; i++) {

            double sum = acount[I - 1].row_sum(i) - acount[I - 1] (I, i);

            align_model[I - 1] (I, i) = source_fert[0];

            if (sum >= 1e-300) {

              assert(!isnan(sum));
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));

              for (uint i_next = 0; i_next < I; i_next++) {
                align_model[I - 1](i_next, i) = std::max(hmm_min_param_entry, inv_sum * source_fert[1] * acount[I - 1] (i_next, i));
                //            if (isnan(align_model[I-1](i_next,i)))
                //              std::cerr << "nan: " << inv_sum << " * " << acount[I-1](i_next,i) << std::endl;

                //            assert(!isnan(align_model[I-1](i_next,i)));
              }
            }
            else {
              for (uint i_next = 0; i_next < I; i_next++) {
                align_model[I - 1](i_next, i) = source_fert[1] / double (I);
              }
            }
          }
        }
      }
    }

    double energy = 0.0;
    for (uint s = 0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      //std::cerr << "J: " << cur_source.size() << ", I: " << cur_target.size() << std::endl;
      //std::cerr << "alignment: " << cur_alignment << std::endl;

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      energy -= logl(hmm_alignment_prob(cur_source, cur_lookup, cur_target, dict, align_model, initial_prob, cur_alignment, true));
    }

    if (dict_weight_sum > 0.0) {
      for (uint i = 0; i < dcount.size(); i++)
        for (uint k = 0; k < dcount[i].size(); k++)
          if (dcount[i][k] > 0)
            energy += prior_weight[i][k];
    }

    std::cerr << "energy after ICM and all updates: " << (energy / nSentences) << std::endl;
#else
    //compute new dict from normalized fractional counts
    update_dict_from_counts(dcount, prior_weight, nSentences, 0.0, false, 0.0, 0, dict, hmm_min_dict_entry);
#endif

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_marg_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;
        //compute viterbi alignment

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        Storage1D<AlignBaseType> viterbi_alignment;
        const uint curI = target[s].size();

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        compute_ehmm_viterbi_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                       initial_prob[curI - 1], viterbi_alignment, options, false, false, 0.0);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        Storage1D<AlignBaseType> marg_alignment;

        compute_ehmm_optmarginal_alignment(source[s], cur_lookup, target[s], dict, align_model[curI - 1],
                                           initial_prob[curI - 1], start_empty_word, marg_alignment);

        sum_marg_aer += AER(marg_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_marg_aer *= 100.0 / nContributors;
      nErrors /= nContributors;
      sum_fmeasure /= nContributors;

      std::cerr << "#### EHMM Viterbi-AER after Viterbi-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "---- EHMM Marginal-AER : " << sum_marg_aer << " %" << std::endl;
      std::cerr << "#### EHMM Viterbi-fmeasure after Viterbi-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM Viterbi-DAE/S after Viterbi-iteration #" << iter << ": " << nErrors << std::endl;
    }
  } //end for (iter)
}
