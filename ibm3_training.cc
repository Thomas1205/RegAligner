/** author: Thomas Schoenemann. This file was generated from singleword_fertility_training while
    Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012. It was subsequently
    modified and extended, both at the University of Düsseldorf and since as a private person. ***/

#include "ibm3_training.hh"
#include "timing.hh"
#include "projection.hh"
#include "training_common.hh"   // for get_wordlookup() and dictionary m-step
#include "stl_util.hh"
#include "storage_stl_interface.hh"
#include "storage_util.hh"

#ifdef HAS_CBC
#include "sparse_matrix_description.hh"
#include "ClpSimplex.hpp"
#include "CbcModel.hpp"
#include "OsiClpSolverInterface.hpp"

#include "CglGomory/CglGomory.hpp"
#include "CglProbing/CglProbing.hpp"
#include "CglRedSplit/CglRedSplit.hpp"
#include "CglTwomir/CglTwomir.hpp"
#include "CglMixedIntegerRounding/CglMixedIntegerRounding.hpp"
#include "CglMixedIntegerRounding2/CglMixedIntegerRounding2.hpp"
#include "CglOddHole/CglOddHole.hpp"
#include "CglLandP/CglLandP.hpp"
#include "CglClique/CglClique.hpp"
#include "CglStored.hpp"

#include "CbcHeuristic.hpp"

#include "count_cut_generator.hh"
#endif

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"

/************************** implementation of IBM3Trainer *********************/

IBM3Trainer::IBM3Trainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                         const Storage1D<Math1D::Vector<uint> >& target_sentence, const Math1D::Vector<WordClassType>& target_class,
                         const RefAlignmentStructure& sure_ref_alignments, const RefAlignmentStructure& possible_ref_alignments,
                         SingleWordDictionary& dict, const CooccuringWordsType& wcooc,
                         const Math1D::Vector<uint>& tfert_class, uint nSourceWords, uint nTargetWords,
                         const floatSingleWordDictionary& prior_weight,
                         const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
                         const FertModelOptions& options)
  : FertilityModelTrainer(source_sentence, slookup, target_sentence, dict, wcooc, tfert_class, nSourceWords, nTargetWords, prior_weight,
                          sure_ref_alignments, possible_ref_alignments, log_table, xlogx_table, options, false),
    target_class_(target_class), distortion_prob_(MAKENAME(distortion_prob_)), min_nondef_count_(options.min_nondef_count_),
    dist_m_step_iter_(options.dist_m_step_iter_), nondef_dist_m_step_iter_(options.nondef_dist34_m_step_iter_),
    par_mode_(options.par_mode_), extra_deficiency_(options.ibm3_extra_deficient_),
    viterbi_ilp_mode_(options.viterbi_ilp_mode_), utmost_ilp_precision_(options.utmost_ilp_precision_),    
    nondeficient_(options.nondeficient_)
{
#ifndef HAS_CBC
  viterbi_ilp_mode_ = IlpOff;
#endif

  if (nondeficient_) {
    empty_word_model_ = FertNullNondeficient;  
  }

  uint nClasses = target_class_.max()+1;

  distortion_prob_.resize(maxJ_);
  if (par_mode_ != IBM23ParByDifference)
    distortion_param_.resize(maxJ_, maxI_, nClasses, 1.0 / maxJ_);
  else
    distortion_param_.resize(maxJ_ + maxI_ - 1, 1, nClasses, 1.0 / (maxJ_ + maxI_ - 1));

  Math1D::Vector<uint> max_I(maxJ_, 0);

  for (size_t s = 0; s < source_sentence_.size(); s++) {

    const uint curI = target_sentence_[s].size();
    const uint curJ = source_sentence_[s].size();

    if (curI > max_I[curJ - 1])
      max_I[curJ - 1] = curI;
  }

  for (uint J = 0; J < maxJ_; J++) {
    distortion_prob_[J].resize_dirty(J + 1, max_I[J], nClasses);
    distortion_prob_[J].set_constant(1.0 / (J + 1));
  }
}

/*virtual*/ std::string IBM3Trainer::model_name() const
{
  return "IBM-3";
}

void IBM3Trainer::init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperWithClasses* passed_wrapper, bool clear_prev,
                                      bool count_collection, bool viterbi)
{
  std::cerr << "initializing IBM-3 from " << prev_model->model_name() << std::endl;

  FertilityModelTrainer* fert_model = dynamic_cast<FertilityModelTrainer*>(prev_model);

  if (count_collection) {

    best_known_alignment_ = prev_model->best_alignments();

    AlignmentSetConstraints align_constraints;

    if (viterbi)
      train_viterbi(1, align_constraints, prev_model, passed_wrapper);
    else
      train_em(1, prev_model, passed_wrapper);

    iter_offs_ = 1;
  }
  else {

    best_known_alignment_ = prev_model->update_alignments_unconstrained(true, passed_wrapper);

    SingleLookupTable aux_lookup;

    if (fert_model == 0) {
      init_fertilities(0); //alignments were already updated and set
    }
    else {

      for (uint k = 1; k < fertility_prob_.size(); k++) {
        fertility_prob_[k] = fert_model->fertility_prob()[k];

        //EXPERIMENTAL
        for (uint l = 0; l < fertility_prob_[k].size(); l++) {
          if (l <= fertility_limit_[k])
            fertility_prob_[k][l] = 0.95 * std::max(fert_min_param_entry,fertility_prob_[k][l])
                                    + 0.05 / std::min<uint>(fertility_prob_[k].size(), fertility_limit_[k] + 1);
          else
            fertility_prob_[k][l] = 0.95 * fertility_prob_[k][l];
        }
        //END_EXPERIMENTAL
      }
    }

    //std::cerr << "BB" << std::endl;

    ReducedIBM3ClassDistortionModel fdcount(distortion_prob_.size(), MAKENAME(fdcount));
    for (uint J = 0; J < distortion_prob_.size(); J++) {
      fdcount[J].resize(J + 1, distortion_prob_[J].yDim(), distortion_prob_[J].zDim(), 0.0);
    }

    if (!fix_p0_) {
      p_zero_ = 0.0;
      p_nonzero_ = 0.0;
    }

    if (!viterbi)
      std::cerr << "initializing distortion prob. by marginals of the previous model" << std::endl;

    //init distortion probabilities using forward-backward
    for (size_t s = 0; s < source_sentence_.size(); s++) {

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      if (viterbi) {

        for (uint j = 0; j < best_known_alignment_[s].size(); j++) {

          uint aj = best_known_alignment_[s][j];

          if (!fix_p0_) {

            if (aj == 0)
              p_zero_ += 1.0;
            else
              p_nonzero_ += 1.0;
          }

          if (aj != 0)
            fdcount[curJ - 1](j, aj - 1, target_class_[cur_target[aj-1]]) += 1.0;
        }
      }
      else {

        Math2D::Matrix<double> marg(curI + 1, curJ, 0.0);
        bool converged;
        prev_model->compute_approximate_jmarginals(cur_source, cur_target, cur_lookup, best_known_alignment_[s], marg, converged);

        for (uint i = 0; i < curI; i++) {

          for (uint j = 0; j < curJ; j++) {

            const double contrib = marg(i + 1, j);

            fdcount[curJ - 1](j, i, target_class_[cur_target[i]]) += contrib;
            if (!fix_p0_)
              p_nonzero_ += contrib;
          }
        }

        if (!fix_p0_) {
          for (uint j = 0; j < curJ; j++)
            p_zero_ += marg(0, j);
        }
      }

      //std::cerr << "CC" << std::endl;
    }

    if (!fix_p0_) {
      double p_norm = p_zero_ + p_nonzero_;
      assert(p_norm != 0.0);
      p_zero_ = p_zero_/p_norm;
      p_nonzero_ = p_nonzero_/p_norm;

      //TRIAL
      // p0 of IBM-3 and IBM-4 is different from p0 for the HMM: if 50% of the words align to zero, p0 is 100% for IBM3/4
      p_zero_ = std::min(2 * p_zero_, 0.75);
      p_nonzero_ = 1.0 - p_zero_;
      //END_TRIAL

      p_zero_ = std::max(p_zero_,fert_min_p0);
      p_nonzero_ = std::max(p_nonzero_,fert_min_p0);
    }

    std::cerr << "initial value of p_zero: " << p_zero_ << std::endl;

    if (par_mode_ != IBM23Nonpar)
      distortion_param_.set_constant(0.0);

    for (uint J = 0; J < distortion_prob_.size(); J++) {

      //std::cerr << "J:" << J << std::endl;

      for (uint i = 0; i < distortion_prob_[J].yDim(); i++) {

        for (uint c = 0; c < distortion_prob_[J].zDim(); c++) {

          double sum = 0.0;
          for (uint j=0; j < fdcount[J].xDim(); j++)
            sum += fdcount[J](j,i,c);
          assert(!isnan(sum));

          if (sum > 1e-305) {
            double inv_sum = 1.0 / sum;

            assert(!isnan(inv_sum));

            for (uint j = 0; j < J + 1; j++) {

              if (par_mode_ == IBM23ParByPosition)
                distortion_param_(j, i, c) += fdcount[J](j, i, c);
              else if (par_mode_ == IBM23ParByDifference)
                distortion_param_(maxI_ - 1 + j - i, 0, c) += fdcount[J](j, i, c);
              else {
                if (viterbi)
                  distortion_prob_[J](j, i, c) = 0.8 * std::max(fert_min_param_entry, inv_sum * fdcount[J](j, i, c)) + 0.2 / (J + 1);
                else
                  distortion_prob_[J](j, i, c) = 0.9 * std::max(fert_min_param_entry, inv_sum * fdcount[J](j, i, c)) + 0.1 / (J + 1);
              }
            }
          }
          else if (par_mode_ == IBM23Nonpar) {
            for (uint j = 0; j < J + 1; j++)
              distortion_prob_[J](j, i, c) = 1.0 / (J + 1);
          }
        }
      }
    }

    if (par_mode_ != IBM23Nonpar) {

      const double w1 = (viterbi) ? 0.9 : 0.95;
      const double w2 = 1.0 - w1;

      for (uint i = 0; i < distortion_param_.yDim(); i++) {

        for (uint c = 0; c < distortion_param_.zDim(); c++) {

          double sum = 0.0;
          for (uint j = 0; j < distortion_param_.xDim(); j++)
            sum += distortion_param_(j, i, c);

          if (sum < 1e-305) {
            for (uint j = 0; j < distortion_param_.xDim(); j++)
              distortion_param_(j, i, c) = 1.0 / distortion_param_.xDim();
          }
          else {
            for (uint j = 0; j < distortion_param_.xDim(); j++) {
              distortion_param_(j, i, c) = w1 * std::max(fert_min_param_entry, distortion_param_(j, i, c) / sum) + w2 / distortion_param_.xDim();
            }
          }
        }
      }

      //std::cerr << "calling par2nonpar" << std::endl;
      par2nonpar_distortion(distortion_prob_);
    }
    assert(distortion_param_.min() > 0.0);
  }

  //std::cerr << "calling clear_prev" << std::endl;
  if (clear_prev)
    prev_model->release_memory();
}

void IBM3Trainer::par2nonpar_distortion(ReducedIBM3ClassDistortionModel& prob)
{
  for (uint J = 0; J < prob.size(); J++) {

    //std::cerr << "J: " << J << std::endl;
    //std::cerr << "xDim: " << prob[J].xDim() << std::endl;

    if (prob[J].size() == 0)
      continue;

    assert(prob[J].xDim() == J + 1);
    assert(distortion_param_.xDim() >= J + 1);

    for (uint i = 0; i < prob[J].yDim(); i++) {

      for (uint c = 0; c < prob[J].zDim(); c++) {

        double sum = 0.0;

        if (par_mode_ == IBM23ParByPosition) {

          if (extra_deficiency_)
            sum = 1.0;
          else {
            for (uint j = 0; j <= J; j++)
              sum += distortion_param_(j, i, c);
          }

          assert(!isnan(sum));

          if (sum > 1e-305) {
            double inv_sum = 1.0 / sum;

            assert(!isnan(inv_sum));

            for (uint j = 0; j <= J; j++)
              prob[J](j, i, c) = std::max(fert_min_param_entry, inv_sum * distortion_param_(j, i, c));
          }
        }
        else {

          if (extra_deficiency_)
            sum = 1.0;
          else {
            for (uint j = 0; j <= J; j++)
              sum += distortion_param_(maxI_ - 1 + j - i, 0, c);
          }

          assert(!isnan(sum));

          if (sum > 1e-305) {
            double inv_sum = 1.0 / sum;

            assert(!isnan(inv_sum));

            for (uint j = 0; j <= J; j++)
              prob[J](j, i, c) = std::max(fert_min_param_entry, inv_sum * distortion_param_(maxI_ - 1 + j - i, 0, c));
          }
        }
      }
    }

    assert(prob[J].min() > 0.0);
  }

  //std::cerr << "leaving par2nonpar" << std::endl;
}

void IBM3Trainer::par2nonpar_distortion(const Math3D::Tensor<double>& param, ReducedIBM3ClassDistortionModel& prob) const
{
  for (uint J = 0; J < prob.size(); J++) {

    assert(param.xDim() >= J + 1);
    assert(prob[J].xDim() == J + 1);

    for (uint i = 0; i < prob[J].yDim(); i++) {

      for (uint c = 0; c < prob[J].zDim(); c++) {

        double sum = 0.0;

        if (par_mode_ == IBM23ParByPosition) {

          if (extra_deficiency_)
            sum = 1.0;
          else {
            for (uint j = 0; j <= J; j++)
              sum += param(j, i, c);
          }

          assert(!isnan(sum));

          if (sum > 1e-305) {
            double inv_sum = 1.0 / sum;

            assert(!isnan(inv_sum));

            for (uint j = 0; j <= J; j++)
              prob[J](j, i, c) = std::max(fert_min_param_entry, inv_sum * param(j, i, c));
          }
        }
        else {

          if (extra_deficiency_)
            sum = 1.0;
          else {
            for (uint j = 0; j <= J; j++)
              sum += param(maxI_ - 1 + j - i, 0, c);
          }

          if (sum > 1e-305) {
            double inv_sum = 1.0 / sum;

            assert(!isnan(inv_sum));

            for (uint j = 0; j <= J; j++)
              prob[J](j, i, c) = std::max(fert_min_param_entry, inv_sum * param(maxI_ - 1 + j - i, 0, c));
          }
        }
      }
    }
  }
}

void IBM3Trainer::nonpar2par_distortion()
{
  distortion_param_.set_constant(0.0);

  for (uint J = 0; J < maxJ_; J++) {

    for (uint c = 0; c < distortion_prob_[J].zDim(); c++) {
      for (uint i = 0; i < distortion_prob_[J].yDim(); i++) {

        for (uint j = 0; j < J + 1; j++)
          distortion_param_(j, i, c) += distortion_prob_[J](j, i, c);
      }
    }
  }

  for (uint c = 0; c < distortion_param_.zDim(); c++) {
    for (uint i = 0; i < distortion_param_.yDim(); i++) {

      double sum = 0.0;
      for (uint j = 0; j < distortion_param_.xDim(); j++)
        sum += distortion_param_(j, i, c);

      if (sum > 1e-305) {

        for (uint j = 0; j < distortion_param_.xDim(); j++)
          distortion_param_(j, i, c) /= sum;
      }
      else
        for (uint j = 0; j < distortion_param_.xDim(); j++)
          distortion_param_(j, i, c) = 1.0 / distortion_param_.xDim();
    }
  }
}

double IBM3Trainer::par_distortion_m_step_energy(const Math1D::Vector<double>& fsingleton_count, const Math1D::Vector<double>& fspan_count,
                                                 const Math1D::Vector<double>& param) const
{
  double energy = 0.0;

  for (uint j = 0; j < fsingleton_count.size(); j++)
    energy -= fsingleton_count[j] * std::log(param[j]);

  double sum = 0.0;

  for (uint J = 0; J < fspan_count.size(); J++) {
    sum += param[J];

    double count = fspan_count[J];
    if (count != 0.0)
      energy += count * std::log(sum);
  }

  return energy;
}


double IBM3Trainer::diffpar_distortion_m_step_energy(const Math3D::Tensor<double>& fsingleton_count, const Math3D::Tensor<double>& fspan_count,
    const Math3D::Tensor<double>& param, uint c) const
{
  double energy = 0.0;

  for (uint k = 0; k < fsingleton_count.xDim(); k++)
    energy -= fsingleton_count(k, 0, c) * std::log(param(k, 0, c));

  for (uint k_start = 0; k_start < fspan_count.xDim(); k_start++) {

    double sum = 0.0;
    for (uint k_end = k_start; k_end < fspan_count.yDim(); k_end++) {

      sum += param(k_end, 0, c);

      const double count = fspan_count(k_start, k_end, c);
      if (count != 0.0) {

        double sum = 0.0;
        for (uint k=k_start; k <= k_end; k++)
          sum += param(k, 0, c);

        energy += count * std::log(sum);
      }
    }
  }

  return energy;
}

void IBM3Trainer::par_distortion_m_step(const Math3D::Tensor<double>& fsingleton_count, const Math3D::Tensor<double>& fspan_count, uint i, uint c,
                                        ProjectionMode projection_mode)
{
  assert(fsingleton_count.xDim() == maxJ_);
  assert(fspan_count.xDim() == maxJ_);

  Math1D::Vector<double> single_vec(maxJ_);
  fsingleton_count.get_x(i, c, single_vec);

  Math1D::Vector<double> span_vec(maxJ_);
  fspan_count.get_x(i, c, span_vec);

  double alpha = 0.1;

  Math1D::Vector<double> cur_param(maxJ_);
  for (uint j = 0; j < maxJ_; j++)
    cur_param[j] = std::max(1e-8, distortion_param_(j, i, c));

  double energy = par_distortion_m_step_energy(single_vec, span_vec, cur_param);

  Math1D::Vector<double> distortion_grad(maxJ_, 0.0);
  Math1D::Vector<double> new_distortion_param(maxJ_, 0.0);
  Math1D::Vector<double> hyp_distortion_param(maxJ_, 0.0);

  double line_reduction_factor = 0.35;

  //try if normalized counts give a better starting point
  {
    const double sum = single_vec.sum();

    for (uint j = 0; j < maxJ_; j++)
      hyp_distortion_param[j] = std::max(fert_min_param_entry, single_vec[j] / sum);

    double hyp_energy = par_distortion_m_step_energy(single_vec, span_vec, hyp_distortion_param);

    if (hyp_energy < energy || extra_deficiency_) {

      energy = hyp_energy;
      cur_param = hyp_distortion_param;
    }
  }

  if (extra_deficiency_) {
    distortion_param_.set_x(i, c, cur_param);
    return;
  }

  const uint nClasses = distortion_param_.zDim();

  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {

    //distortion_grad.set_constant(0.0);

    /*** compute gradient ***/

    for (uint j = 0; j < maxJ_; j++)
      distortion_grad[j] = -(single_vec[j] / std::max(fert_min_param_entry, cur_param[j]));

    double sum = 0.0;

    Math1D::Vector<double> addon(maxJ_, 0.0);

    for (uint J = 0; J < maxJ_; J++) {
      sum += std::max(fert_min_param_entry, cur_param[J]);

      double count = span_vec[J];
      if (count != 0.0) {
        addon[J] = count / sum;
        // double addon = count / sum;

        // for (uint j=0; j <= J; j++)
        //   distortion_grad[j] += addon;
      }
    }

    double sum_addon = 0.0;
    for (int j = maxJ_ - 1; j >= 0; j--) {
      sum_addon += addon[j];
      distortion_grad[j] += sum_addon;
    }

    /*** go in neg. gradient direction and reproject ***/
    if (projection_mode == Simplex) {

      //for (uint j = 0; j < maxJ_; j++)
      //  new_distortion_param[j] = cur_param[j] - alpha * distortion_grad[j];      
      Math1D::go_in_neg_direction(new_distortion_param, cur_param, distortion_grad, alpha);

      projection_on_simplex(new_distortion_param.direct_access(), maxJ_, fert_min_param_entry);
    }
    else {

      //orthant projection followed by renormalization
      // (justified by the scale invariance of the objective)
      // may be faster than simplex

      double sum = 0.0;
      for (uint j = 0; j < maxJ_; j++) {
        new_distortion_param[j] = std::max(fert_min_param_entry, cur_param[j] - alpha * distortion_grad[j]);
        sum += new_distortion_param[j];
      }

      new_distortion_param *= 1.0 / sum;
    }

    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      //for (uint j = 0; j < maxJ_; j++)
      //  hyp_distortion_param[j] = lambda * new_distortion_param[j] + neg_lambda * cur_param[j];
      Math1D::assign_weighted_combination(hyp_distortion_param, lambda, new_distortion_param, neg_lambda, cur_param);

      double hyp_energy = par_distortion_m_step_energy(single_vec, span_vec, hyp_distortion_param);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;

    if (best_energy >= energy) {
      if (nClasses <= 5)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    //EXPERIMENTAL
    // if (nIter > 4)
    //   alpha *= 1.5;
    //END_EXPERIMENTAL

    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint j = 0; j < maxJ_; j++)
    //  cur_param[j] = best_lambda * new_distortion_param[j] + neg_best_lambda * cur_param[j];
    Math1D::assign_weighted_combination(cur_param, best_lambda, new_distortion_param, neg_best_lambda, cur_param);

    energy = best_energy;
  }

  if (nClasses <= 5)
    std::cerr << "final m-step energy: " << energy << std::endl;

  distortion_param_.set_x(i, c, cur_param);
}

void IBM3Trainer::diffpar_distortion_m_step(const Math3D::Tensor<double>& fsingleton_count, const Math3D::Tensor<double>& fspan_count, uint c)
{
  //std::cerr << "diffpar_distortion_m_step" << std::endl;

  const uint xDim = distortion_param_.xDim();
  const uint yDim = distortion_param_.yDim();
  const uint zDim = distortion_param_.zDim();

  double energy = diffpar_distortion_m_step_energy(fsingleton_count, fspan_count, distortion_param_, c);

  const uint nClasses = distortion_param_.zDim();

  if (nClasses == 1)
    std::cerr << "diffpar m-step start energy : " << energy << std::endl;

  Math1D::Vector<double> distortion_grad(xDim);
  Math1D::Vector<double> new_distortion_param(xDim);
  Math3D::Tensor<double> hyp_distortion_param(xDim, distortion_param_.yDim(), distortion_param_.zDim());

  //try if normalized counts give a better starting point
  {
    double sum = fsingleton_count.sum_x(0, c);

    for (uint j = 0; j < xDim; j++)
      hyp_distortion_param(j, 0, c) = std::max(fert_min_param_entry, fsingleton_count(j, 0, c) / sum);

    double hyp_energy = diffpar_distortion_m_step_energy(fsingleton_count, fspan_count, hyp_distortion_param, c);

    if (nClasses == 1)
      std::cerr << "diffpar m-step normalized count energy : " << hyp_energy << std::endl;

    if (hyp_energy < energy || extra_deficiency_) {

      if (nClasses == 1)
        std::cerr << "switching to normalized counts" << std::endl;

      energy = hyp_energy;
      distortion_param_ = hyp_distortion_param;
    }
  }

  if (extra_deficiency_)
    return;

  Math1D::Vector<double> cur_param(xDim);
  distortion_param_.get_x(0, c, cur_param);

  double alpha = 0.1;
  double line_reduction_factor = 0.35;

  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {
    
    if (nClasses == 1 && (iter % 10) == 0)
      std::cerr << "diffpar m-step energy for iter " << iter << " : " << energy << std::endl;

    //distortion_grad.set_constant(0.0);

    /*** compute gradient ***/
    for (uint k = 0; k < distortion_grad.size(); k++)
      distortion_grad[k] = -(fsingleton_count(k, 0, c) / std::max(fert_min_param_entry, distortion_param_(k, 0, c)));

    for (uint k_start = 0; k_start < fspan_count.xDim(); k_start++) {

      double sum = 0.0;
      for (uint k_end = k_start; k_end < fspan_count.yDim(); k_end++) {

        sum += distortion_param_(k_end, 0, c);
        const double count = fspan_count(k_start, k_end, c);

        if (count != 0.0) {
          double sum = 0.0;
          for (uint k=k_start; k <= k_end; k++)
            sum += std::max(1e-15,distortion_param_(k, 0, c));

          for (uint k = k_start; k <= k_end; k++)
            distortion_grad[k] += count / sum;
        }
      }
    }

    /*** go in negative gradient direction and reproject ***/
    for (uint k = 0; k < distortion_grad.size(); k++)
      new_distortion_param[k] = cur_param[k] - alpha * distortion_grad[k];

    projection_on_simplex(new_distortion_param.direct_access(), distortion_grad.size(), fert_min_param_entry);

    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      for (uint k = 0; k < hyp_distortion_param.xDim(); k++)
        hyp_distortion_param(k, 0, c) = lambda * new_distortion_param[k] + neg_lambda * cur_param[k];

      double hyp_energy = diffpar_distortion_m_step_energy(fsingleton_count, fspan_count, hyp_distortion_param, c);

      //std::cerr << "hyp_energy: " << hyp_energy << std::endl;

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;

    if (best_energy >= energy) {
      if (nClasses <= 5)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    const double neg_best_lambda = 1.0 - best_lambda;

    for (uint j = 0; j < hyp_distortion_param.xDim(); j++)
      cur_param[j] = best_lambda * new_distortion_param[j] + neg_best_lambda * distortion_param_(j, 0, c);

    energy = best_energy;
  }
  
  distortion_param_.set_x(0, c, cur_param);
}

void IBM3Trainer::par_distortion_m_step_unconstrained(const Math3D::Tensor<double>& fsingleton_count, const Math3D::Tensor<double>& fspan_count,
    uint i, uint c, uint L)
{
  //in this formulation we use parameters p=x^2 to get an unconstrained formulation
  // here we use L-BFGS

  assert(fsingleton_count.xDim() == maxJ_);
  assert(fspan_count.xDim() == maxJ_);

  Math1D::Vector<double> single_vec(maxJ_);
  fsingleton_count.get_x(i, c, single_vec);

  Math1D::Vector<double> span_vec(maxJ_);
  fspan_count.get_x(i, c, span_vec);

  double alpha = 0.1;

  Math1D::Vector<double> cur_param(maxJ_);
  for (uint j = 0; j < maxJ_; j++)
    cur_param[j] = std::max(fert_min_param_entry, distortion_param_(j, i, c));

  const uint nClasses = distortion_param_.zDim();

  double start_energy = par_distortion_m_step_energy(single_vec, span_vec, cur_param);

  Math1D::Vector<double> distortion_grad(maxJ_);
  Math1D::Vector<double> hyp_distortion_param(maxJ_, 0.0);
  Math1D::Vector<double> work_param(maxJ_);
  Math1D::Vector<double> hyp_work_param(maxJ_);
  Math1D::Vector<double> work_grad(maxJ_);
  Math1D::Vector<double> search_direction(maxJ_);

  double line_reduction_factor = 0.35;

  double energy = start_energy;

  //try if normalized counts give a better starting point
  {
    double sum = single_vec.sum();

    for (uint j = 0; j < maxJ_; j++)
      hyp_distortion_param[j] = std::max(fert_min_param_entry, single_vec[j] / sum);

    double hyp_energy = par_distortion_m_step_energy(single_vec, span_vec, hyp_distortion_param);

    if (hyp_energy < energy) {

      energy = hyp_energy;
      cur_param = hyp_distortion_param;
    }
  }

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(maxJ_);
    step[k].resize(maxJ_);
  }

  //extract working params from the current probabilities (the probabilities are the squared working params)
  for (uint k = 0; k < maxJ_; k++)
    work_param[k] = sqrt(cur_param[k]);

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  double scale = 1.0;

  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {
    //distortion_grad.set_constant(0.0);
    work_grad.set_constant(0.0);

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    for (uint j = 0; j < maxJ_; j++)
      distortion_grad[j] = -single_vec[j] / std::max(fert_min_param_entry, cur_param[j]);

    double sum = 0.0;

    Math1D::Vector<double> addon(maxJ_, 0.0);

    for (uint J = 0; J < maxJ_; J++) {
      sum += std::max(1e-8, cur_param[J]);

      double count = span_vec[J];
      if (count != 0.0) {
        addon[J] = count / sum;
        // double addon = count / sum;

        // for (uint j=0; j <= J; j++)
        //   distortion_grad[j] += addon;
      }
    }

    double sum_addon = 0.0;
    for (int j = maxJ_ - 1; j >= 0; j--) {
      sum_addon += addon[j];
      distortion_grad[j] += sum_addon;
    }

    // b) now calculate the gradient for the actual parameters

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = scale; //work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    double coeff_sum = 0.0;

    for (uint k = 0; k < maxJ_; k++) {
      const double wp = work_param[k];
      const double grad = distortion_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 0; kk < maxJ_; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

    // c) determine the search direction
    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < maxJ_; k++) {

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

        // for (uint k=0; k < maxJ_; k++)
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

        search_direction.add_vector_multiple(cur_step, gamma);
        // for (uint k=0; k < maxJ_; k++)
        //   search_direction[k] += cur_step[k] * gamma;
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

      scale = 0.0;
      for (uint k = 0; k < maxJ_; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
        scale += hyp_work_param[k] * hyp_work_param[k];
      }

      for (uint k = 0; k < maxJ_; k++)
        hyp_distortion_param[k] = std::max(fert_min_param_entry, hyp_work_param[k] * hyp_work_param[k] / scale);

      double hyp_energy = par_distortion_m_step_energy(single_vec, span_vec, hyp_distortion_param);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point
    if (best_energy >= energy - 1e-4) {
      if (nClasses <= 5) {
        std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy) << std::endl;
        std::cerr << "last squared gradient norm: " << work_grad.sqr_norm() << std::endl;
      }
      break;
    }

    energy = best_energy;

    uint cur_l = (iter % L);

    Math1D::Vector<double>& cur_step = step[cur_l];
    Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

    double new_sum = 0.0;

    for (uint k = 0; k < maxJ_; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      new_sum += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    for (uint k = 0; k < maxJ_; k++)
      cur_param[k] = std::max(fert_min_param_entry, work_param[k] * work_param[k] / new_sum);
  }

  if (nClasses <= 5)
    std::cerr << "final m-step energy: " << energy << std::endl;

  for (uint x = 0; x < maxJ_; x++)
    distortion_param_(x, i, c) = cur_param[x];
}

//compact form
double IBM3Trainer::nondeficient_m_step_energy(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
    const std::vector<double>& sum_pos_count, const Math3D::Tensor<double>& param, uint i, uint c) const
{
  assert(open_pos.size() == sum_pos_count.size());
  assert(single_pos_count.xDim() == param.xDim());
  assert(i < param.yDim());
  assert(c < param.zDim());

  double energy = 0.0;

  Math1D::Vector<double> cur_param(param.xDim());
  param.get_x(i, c, cur_param);

  //part 1: singleton terms
  for (uint j = 0; j < single_pos_count.xDim(); j++) {
    //std::cerr << "j: " << j << std::endl;
    energy -= single_pos_count(j, i, c) * std::log(cur_param[j]);
    //std::cerr << "subtracting " << single_pos_count(j,i) << "* std::log(" << cur_param[j] << std::endl;
    assert(!isnan(energy));
  }

  //part 2: normalization terms

  //DEBUG
  //double singleton_share = 0.0;
  //END_DEBUG

  for (uint k = 0; k < open_pos.size(); k++) {

    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];
    const uchar size = open_positions.size();

    assert(size > 0);
    // if (size == 0)
    //   continue;

    const double weight = sum_pos_count[k];

    double sum = 0.0;
    for (uint k = 0; k < size; k++) {
      sum += cur_param[open_positions[k]];
    }

    //DEBUG
    //if (size == 1)
    //  singleton_share += weight * std::log(sum);
    //END_DEBUG

    energy += weight * std::log(sum);

    //DEBUG
    // if (isnan(energy)) {
    //   std::cerr << "sum: " << sum << std::endl;
    //   std::cerr << "open_positions: ";
    //   for (uint i=0; i < open_positions.size(); i++)
    //  std::cerr << open_positions[i] << " ";
    //   std::cerr << std::endl;
    // }
    assert(!isnan(energy));
    //END_DEBUG
  }

  //DEBUG
  //std::cerr << "singleton share on the normalization part: " << singleton_share << std::endl;
  //END_DEBUG

  return energy;
}

double IBM3Trainer::nondeficient_m_step_energy(const Math1D::Vector<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
                                               const std::vector<double>& sum_pos_count, const Math1D::Vector<double>& param) const 
{                                                
  double energy = 0.0;

  //part 1: singleton terms
  for (uint j = 0; j < single_pos_count.size(); j++) {
    energy -= single_pos_count[j] * std::log(param[j]);
    assert(!isnan(energy));
  }

  //part 2: normalization terms
  for (uint k = 0; k < open_pos.size(); k++) {

    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];
    const uchar size = open_positions.size();

    assert(size > 0);

    const double weight = sum_pos_count[k];

    double sum = 0.0;
    for (uint k = 0; k < size; k++) {
      sum += param[open_positions[k]];
    }

    energy += weight * std::log(sum);
  }

  return energy;                                                 
}

//compact form
double IBM3Trainer::nondeficient_diffpar_m_step_energy(const Math3D::Tensor<double>& fsingleton_count,
    const Storage1D<std::vector<Math1D::Vector<ushort,uchar> > >& open_pos,
    const Storage1D<std::vector<double> >& sum_pos_count, const Math3D::Tensor<double>& param) const
{
  double energy = 0.0;

  //part 1: singleton terms
  for (uint c = 0; c < fsingleton_count.zDim(); c++)
    for (uint k = 0; k < fsingleton_count.xDim(); k++)
      energy -= fsingleton_count(k, 0, c) * std::log(param(k, 0, c));

  //part 2: sum terms
  for (uint c = 0; c < open_pos.size(); c++) {
    for (uint k = 0; k < open_pos[c].size(); k++) {

      const Math1D::Vector<ushort,uchar>& open_positions = open_pos[c][k];
      const uchar size = open_positions.size();

      assert(size > 0);

      const double weight = sum_pos_count[c][k];

      double sum = 0.0;
      for (uint l = 0; l < size; l++) {
        sum += param(open_positions[l], 0, c);
      }
      energy += weight * std::log(sum);
      assert(!isnan(energy));
    }
  }

  return energy;
}

//compact form with interpolation
double IBM3Trainer::nondeficient_m_step_energy(const double* single_pos_count, const uint J, const std::vector<double>& sum_pos_count,
    const double* param1, const Math1D::Vector<double>& param2, const Math1D::Vector<double>& sum1,
    const Math1D::Vector<double>& sum2, double lambda) const
{
  const double neg_lambda = 1.0 - lambda;

  double energy = 0.0;

  //part 1: singleton terms
  for (uint j = 0; j < J; j++) {

    double cur_param = lambda * param2[j] + neg_lambda * param1[j];

    energy -= single_pos_count[j] * std::log(cur_param);
  }

  //part 2: normalization terms

  for (uint k = 0; k < sum_pos_count.size(); k++) {

    const double weight = sum_pos_count[k];

    double sum = lambda * sum2[k] + neg_lambda * sum1[k];

    energy += weight * std::log(sum);
  }

  return energy;
}

//compact form
double IBM3Trainer::nondeficient_m_step_core(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
    const std::vector<double>& sum_pos_count, Math3D::Tensor<double>& param, uint i, uint c,
    double start_energy, bool quiet)
{
  const uint xDim = param.xDim();

  assert(single_pos_count.xDim() == xDim);

  double alpha = 0.01;

  double energy = start_energy;
  assert(!isnan(start_energy));

  if (distortion_param_.zDim() > 1)
    quiet = true;

  if (!quiet)
    std::cerr << "start energy: " << energy << std::endl;

  Math1D::Vector<double> distortion_grad(xDim);
  Math1D::Vector<double> hyp_distortion_param(xDim);
  Math1D::Vector<double> new_distortion_param(xDim);

  Math1D::Vector<double> cur_param(xDim);
  param.get_x(i, c, cur_param);
  
  Math1D::Vector<double> cur_single_pos_count(xDim);
  single_pos_count.get_x(i, c, cur_single_pos_count);

  //test if normalizing the passed singleton count gives a better starting point
  const double norm = cur_single_pos_count.sum();

  if (norm > 1e-305) {

    for (uint j = 0; j < xDim; j++)
      hyp_distortion_param[j] = std::max(fert_min_param_entry, cur_single_pos_count[j] / norm);

    double hyp_energy = nondeficient_m_step_energy(cur_single_pos_count, open_pos, sum_pos_count, hyp_distortion_param);

    if (hyp_energy < energy) {

      cur_param = hyp_distortion_param;

      if (!quiet)
        std::cerr << "switching to passed normalized singleton count ---> " << hyp_energy << std::endl;

      energy = hyp_energy;
    }
  }

  double line_reduction_factor = 0.35;

  std::clock_t tStart = std::clock();

  for (uint iter = 1; iter <= nondef_dist_m_step_iter_; iter++) {

    if (!quiet && (iter % 50) == 0) {
      std::cerr << "m-step iter # " << iter << ", energy: " << energy << std::endl;

      std::clock_t tInter = std::clock();
      std::cerr << "spent " << diff_seconds(tInter, tStart) << " seconds so far" << std::endl;
    }

    //distortion_grad.set_constant(0.0);

    /*** compute gradient ***/

    //part 1: singleton terms
    for (uint j = 0; j < xDim; j++) {

      const double weight = cur_single_pos_count[j];
      distortion_grad[j] = -weight / cur_param[j];
    }

    //part 2: normalization terms
    for (uint k = 0; k < open_pos.size(); k++) {

      const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];
      const uchar size = open_positions.size();
      const double weight = sum_pos_count[k];

      double sum = 0.0;
      for (uint k = 0; k < size; k++)
        sum += cur_param[open_positions[k]];
      sum = std::max(sum, fert_min_param_entry);

      const double addon = weight / sum;

      for (uint k = 0; k < open_positions.size(); k++)
        distortion_grad[open_positions[k]] += addon;
    }

    /*** go in neg. gradient direction and reproject ***/

    //for (uint j = 0; j < xDim; j++)
    //  new_distortion_param[j] = cur_param[j] - alpha * distortion_grad[j];
    Math1D::go_in_neg_direction(new_distortion_param, cur_param, distortion_grad, alpha);

    projection_on_simplex(new_distortion_param.direct_access(), xDim, fert_min_param_entry);

    /*** find appropriate step-size ***/

    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;
      const double neg_lambda = 1.0 - lambda;

      //for (uint j = 0; j < xDim; j++)
      //  hyp_distortion_param[j] = lambda * new_distortion_param[j] + neg_lambda * cur_param[j];
      Math1D::assign_weighted_combination(hyp_distortion_param, lambda, new_distortion_param, neg_lambda, cur_param); 

      double hyp_energy = nondeficient_m_step_energy(cur_single_pos_count, open_pos, sum_pos_count, hyp_distortion_param);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;

    if (best_energy >= energy) {
      if (!quiet)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint j = 0; j < xDim; j++)
    //  cur_param[j] = std::max(fert_min_param_entry, best_lambda * new_distortion_param[j] + neg_best_lambda * cur_param[j]);
    Math1D::assign_weighted_combination(cur_param, best_lambda, new_distortion_param, neg_best_lambda, cur_param); 

    energy = best_energy;
  }

  param.set_x(i, c, cur_param);

  return energy;
}

//compact form
double IBM3Trainer::nondeficient_diffpar_m_step(const Math3D::Tensor<double>& fsingleton_count,
    const Storage1D<std::vector<Math1D::Vector<ushort,uchar> > >& open_pos,
    const Storage1D<std::vector<double> >& sum_pos_count, double start_energy)
{
  const uint nClasses = distortion_param_.zDim();

  if (nClasses == 1)
    std::cerr << "nondeficient_diffpar_m_step, start energy: " << start_energy << std::endl;

  double energy = start_energy;

  const uint xDim = distortion_param_.xDim();
  const uint zDim = distortion_param_.zDim();

  Math3D::Tensor<double> hyp_param = fsingleton_count;
  Math2D::Matrix<double> new_param(xDim,zDim);
  Math2D::Matrix<double> gradient(xDim,zDim);

  {
    //test normalized counts

    for (uint c = 0; c < zDim; c++) {

      const double sum = hyp_param.sum_x(0, c);

      if (sum > 1-305) {
        for (uint k = 0; k < xDim; k++)
          hyp_param(k, 0, c) = std::max(hyp_param(k, 0, c) / sum, fert_min_param_entry);
      }
    }

    double hyp_energy = nondeficient_diffpar_m_step_energy(fsingleton_count, open_pos, sum_pos_count, hyp_param);

    if (nClasses == 1)
      std::cerr << "energy for normalized counts: " << hyp_energy << std::endl;

    if (hyp_energy < energy) {

      if (nClasses == 1)
        std::cerr << "switching to normalized counts" << std::endl;

      energy = hyp_energy;
      distortion_param_ = hyp_param;
    }
  }

  double line_reduction_factor = 0.35;
  double alpha = 0.01;

  std::clock_t tStart = std::clock();

  for (uint iter = 1; iter <= nondef_dist_m_step_iter_; iter++) {

    if (nClasses == 1 && (iter % 5) == 0) {
      std::cerr << "iteration " << iter << ", energy: " << energy << std::endl;
    }

    /*** compute gradient ***/
    for (uint c = 0; c < zDim; c++)
      for (uint k = 0; k < xDim; k++)
        gradient(k, c) = -fsingleton_count(k, 0, c) / distortion_param_(k, 0, c);

    for (uint c = 0; c < zDim; c++) {
      for (uint k = 0; k < open_pos[c].size(); k++) {
        const Math1D::Vector<ushort,uchar>& cur_open = open_pos[c][k];
        double sum = 0.0;
        for (uint l = 0; l < cur_open.size(); l++)
          sum += distortion_param_(cur_open[l], 0, c);
        const double grad = sum_pos_count[c][k] / sum;
        for (uint l = 0; l < cur_open.size(); l++)
          gradient(cur_open[l], c) += grad;
      }
    }

    /*** go in gradient direction and reproject ***/
    for (uint c = 0; c < zDim; c++) {
      for (uint k = 0; k < xDim; k++)
        new_param(k, c) = distortion_param_(k, 0, c) - alpha * gradient(k, c);

      projection_on_simplex(new_param.direct_access() + c*xDim, xDim, fert_min_param_entry);
    }

    /*** find appropriate step-size ***/

    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      for (uint k = 0; k < hyp_param.size(); k++)
        hyp_param.direct_access(k) = lambda * new_param.direct_access(k) + neg_lambda * distortion_param_.direct_access(k);

      double hyp_energy = nondeficient_diffpar_m_step_energy(fsingleton_count, open_pos, sum_pos_count, hyp_param);

      //std::cerr << "lambda: " << lambda << ", hyp_energy: " << hyp_energy << std::endl;

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;

    if (best_energy >= energy) {
      if (nClasses == 1)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = 0; k < distortion_param_.size(); k++)
      distortion_param_.direct_access(k) = best_lambda * new_param.direct_access(k) + neg_best_lambda * distortion_param_.direct_access(k);

    energy = best_energy;
  }

  return energy;
}

double IBM3Trainer::nondeficient_m_step_unconstrained_core(const Math3D::Tensor<double>& single_pos_count,
    const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
    const std::vector<double>& sum_pos_count, Math3D::Tensor<double>& param,
    uint i, uint c, double start_energy, bool quiet, uint L)
{
  const uint nClasses = param.zDim();
  if (nClasses > 1)
    quiet = true;

  const uint xDim = param.xDim();

  assert(single_pos_count.xDim() == xDim);

  double energy = start_energy;
  assert(!isnan(start_energy));

  if (!quiet)
    std::cerr << "start energy: " << energy << std::endl;

  Math1D::Vector<double> distortion_grad(xDim);
  Math1D::Vector<double> work_param(xDim);
  Math1D::Vector<double> hyp_work_param(xDim);
  Math1D::Vector<double> work_grad(xDim);
  Math1D::Vector<double> search_direction(xDim);
  Math3D::Tensor<double> hyp_distortion_param = param;

  //test if normalizing the passed singleton count gives a better starting point
  const double norm = single_pos_count.sum_x(i, c);

  if (norm > 1e-305) {

    for (uint j = 0; j < xDim; j++)
      hyp_distortion_param(j, i, c) = std::max(fert_min_param_entry, single_pos_count(j, i, c) / norm);

    double hyp_energy = nondeficient_m_step_energy(single_pos_count, open_pos, sum_pos_count, hyp_distortion_param, i, c);

    if (hyp_energy < energy) {

      for (uint j = 0; j < xDim; j++)
        param(j, i, c) = hyp_distortion_param(j, i, c);

      if (!quiet)
        std::cerr << "switching to passed normalized singleton count ---> " << hyp_energy << std::endl;

      energy = hyp_energy;
    }
  }

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(xDim);
    step[k].resize(xDim);
  }

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  std::clock_t tStart = std::clock();

  for (uint k = 0; k < xDim; k++)
    work_param[k] = sqrt(param(k, i, c));

  double scale = 1.0;

  Math1D::Vector<double> cur_param(xDim);
  param.get_x(i, c, cur_param);

  for (uint iter = 1; iter <= nondef_dist_m_step_iter_; iter++) {

    if (!quiet && (iter % 50) == 0) {
      std::cerr << "L-BFGS m-step iter # " << iter << ", energy: " << energy << std::endl;

      std::clock_t tInter = std::clock();
      std::cerr << "spent " << diff_seconds(tInter,tStart) << " seconds so far" << std::endl;
    }
    // a) calculate gradient w.r.t. the probabilities, not the parameters

    distortion_grad.set_constant(0.0);

    //part 1: singleton terms
    for (uint j = 0; j < xDim; j++) {

      double weight = single_pos_count(j, i, c);

      distortion_grad[j] -= weight / cur_param[j];
    }

    //part 2: normalization terms
    for (uint k = 0; k < open_pos.size(); k++) {

      const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];
      const uchar size = open_positions.size();
      const double weight = sum_pos_count[k];

      double sum = 0.0;
      for (uint k = 0; k < size; k++)
        sum += cur_param[open_positions[k]];
      sum = std::max(sum, fert_min_param_entry);

      const double addon = weight / sum;

      for (uint k = 0; k < open_positions.size(); k++)
        distortion_grad[open_positions[k]] += addon;
    }

    // b) now calculate the gradient for the actual parameters

    if (!quiet)
      std::cerr << "sqr sum: " << scale << std::endl;

    const double denom = scale; //work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    double coeff_sum = 0.0;

    for (uint k = 0; k < xDim; k++) {
      const double wp = work_param[k];
      const double grad = distortion_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 0; kk < xDim; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < xDim; k++) {

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

      const int cur_first_iter = std::max < int >(start_iter, iter - L);

      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter - 1; prev_iter >= cur_first_iter; prev_iter--) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double cur_alpha = search_direction % cur_step;
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        search_direction.add_vector_multiple(cur_grad_diff, -cur_alpha);
        // for (uint k=0; k < xDim; k++)
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

        search_direction.add_vector_multiple(cur_step, gamma);
        // for (uint k=0; k < xDim; k++)
        //   search_direction[k] += cur_step[k] * gamma;
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

      double sqr_sum = 0.0;

      for (uint k = 0; k < xDim; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_work_param[k] * hyp_work_param[k];
      }

      for (uint k = 0; k < xDim; k++)
        hyp_distortion_param(k, i, c) = std::max(fert_min_param_entry, hyp_work_param[k] * hyp_work_param[k] / sqr_sum);

      double hyp_energy = nondeficient_m_step_energy(single_pos_count, open_pos, sum_pos_count, hyp_distortion_param, i, c);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point

    if (best_energy >= energy - 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy)
                << ", final energy: " << energy << std::endl;
      std::cerr << "last squared gradient norm: " << work_grad.sqr_norm() << std::endl;
      break;
    }

    energy = best_energy;

    uint cur_l = (iter % L);

    Math1D::Vector<double>& cur_step = step[cur_l];
    Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

    scale = 0.0;
    for (uint k = 0; k < xDim; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      scale += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    for (uint k = 0; k < xDim; k++)
      cur_param[k] = std::max(fert_min_param_entry, work_param[k] * work_param[k] / scale);
  }

  param.set_x(i, c, cur_param);
  
  return energy;
}

//compact form
double IBM3Trainer::nondeficient_m_step(const Math3D::Tensor<double>& single_pos_count,
                                        const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
                                        const std::vector<double>& sum_pos_count, uint i, uint c, double start_energy)
{
  assert(open_pos.size() == sum_pos_count.size());
  assert(!isnan(start_energy));

  return nondeficient_m_step_core(single_pos_count, open_pos, sum_pos_count, distortion_param_, i, c, start_energy);
}

//compact form for the non-parametric setting
double IBM3Trainer::nondeficient_m_step(const Math3D::Tensor<double>& single_pos_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
                                        const std::vector<double>& sum_pos_count, uint i, uint c, uint J, double start_energy)
{
  assert(single_pos_count.xDim() == J);
  assert(open_pos.size() == sum_pos_count.size());
  assert(!isnan(start_energy));

  return nondeficient_m_step_core(single_pos_count, open_pos, sum_pos_count, distortion_prob_[J - 1], i, c, start_energy, true);
}

//compact form
double IBM3Trainer::nondeficient_m_step_with_interpolation_core(const Math3D::Tensor<double>& single_pos_count,
    const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
    const std::vector<double>& weight, Math3D::Tensor<double>& param,
    uint i, uint c, double start_energy, bool quiet)
{
  assert(!isnan(start_energy));

  const uint nClasses = param.zDim();
  const uint J = param.xDim();

  if (nClasses > 1)
    quiet = true;

  for (uint j = 0; j < J; j++)
    param(j, i, c) = std::max(fert_min_param_entry, param(j, i, c));

  //test if normalizing the passed singleton count gives a better starting point
  Math3D::Tensor<double> hyp_distortion_param(J, param.yDim(), param.zDim());

  double norm = 0.0;
  for (uint j = 0; j < J; j++)
    norm += single_pos_count(j, i, c);

  if (norm > 1e-305) {

    for (uint j = 0; j < J; j++)
      hyp_distortion_param(j, i, c) = std::max(fert_min_param_entry, single_pos_count(j, i, c) / norm);

    double hyp_energy = nondeficient_m_step_energy(single_pos_count, open_pos, weight, hyp_distortion_param, i, c);

    if (hyp_energy < start_energy) {

      for (uint j = 0; j < J; j++)
        param(j, i, c) = hyp_distortion_param(j, i, c);

      if (!quiet)
        std::cerr << "switching to passed normalized singleton count ---> " << hyp_energy << std::endl;

      start_energy = hyp_energy;
    }
  }

  Math1D::Vector<double> cur_param(J);
  Math1D::Vector<double> cur_single_count(J);
  param.get_x(i, c, cur_param);
  single_pos_count.get_x(i, c, cur_single_count);

  Math1D::Vector<double> sum(weight.size());
  Math1D::Vector<double> new_sum(weight.size(), 0.0);

  for (uint k = 0; k < weight.size(); k++) {

    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];

    //DEBUG
    // if (open_positions.size() <= 1) {
    //   INTERNAL_ERROR << " too few positions listed: " << open_positions.size() << std::endl;
    // }
    //END_DEBUG

    assert(open_positions.size() > 0);

    double cur_sum = 0.0;
    for (uchar l = 0; l < open_positions.size(); l++) {
      cur_sum += cur_param[open_positions[l]];
    }
    sum[k] = cur_sum;
  }

  double alpha = 0.01;          //0.1;

  double energy = start_energy;
  assert(!isnan(start_energy));

  if (!quiet)
    std::cerr << "start energy: " << energy << std::endl;

  Math1D::Vector<double> distortion_grad(J);
  Math1D::Vector<double> new_distortion_param(J);

  double line_reduction_factor = 0.35;

  std::clock_t tStart = std::clock();

  double save_energy = energy;

  for (uint iter = 1; iter <= nondef_dist_m_step_iter_; iter++) {

    if ((iter % 50) == 0) {
      if (!quiet) {
        std::cerr << "m-step iter # " << iter << ", energy: " << energy << std::endl;

        std::clock_t tInter = std::clock();
        std::cerr << "spent " << diff_seconds(tInter,tStart) << " seconds so far" << std::endl;
      }

      if (save_energy - energy < 0.35)
        break;
      if (iter >= 100 && save_energy - energy < 0.75)
        break;

      save_energy = energy;
    }

    distortion_grad.set_constant(0.0);

    /*** compute gradient ***/

    //a) singleton terms
    for (uint j = 0; j < single_pos_count.xDim(); j++) {

      const double weight = single_pos_count(j, i, c);
      distortion_grad[j] -= weight / cur_param[j];
    }

    //b) normalization terms
    int k = -1;
    for (std::vector<Math1D::Vector<uchar,uchar> >::const_iterator it = open_pos.begin(); it != open_pos.end(); it++) {

      k++;

      const Math1D::Vector<uchar,uchar>& open_positions = *it;
      const double cur_weight = weight[k];

      double cur_sum = sum[k];

      const double addon = cur_weight / cur_sum;

      for (uint k = 0; k < open_positions.size(); k++) {
        assert(open_positions[k] < J);
        distortion_grad[open_positions[k]] += addon;
      }
    }

    /*** go in neg. gradient direction and reproject ***/

    for (uint j = 0; j < J; j++) {
      new_distortion_param[j] = cur_param[j] - alpha * distortion_grad[j];

      if (new_distortion_param[j] < -1e75) {
        std::cerr << "fixing numerical instability" << std::endl;
        new_distortion_param[j] = -1e-75;
      }
      if (new_distortion_param[j] > 1e75) {
        std::cerr << "fixing numerical instability" << std::endl;
        new_distortion_param[j] = 1e-75;
      }
    }

    projection_on_simplex(new_distortion_param.direct_access(), new_distortion_param.size(), fert_min_param_entry);

    for (uint k = 0; k < weight.size(); k++) {

      const Math1D::Vector<uchar, uchar>& open_positions = open_pos[k];

      double cur_new_sum = 0.0;
      for (uchar l = 0; l < open_positions.size(); l++)
        cur_new_sum += new_distortion_param[open_positions[l]];
      new_sum[k] = cur_new_sum;
    }

    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;

      double hyp_energy = nondeficient_m_step_energy(cur_single_count.direct_access(), J, weight, cur_param.direct_access(),
                          new_distortion_param, sum, new_sum, lambda);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;

    if (best_energy >= energy) {
      if (!quiet)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    //EXPERIMENTAL
    // if (nIter > 4)
    //   alpha *= 1.5;
    //END_EXPERIMENTAL

    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint j = 0; j < J; j++)
    //  cur_param[j] = best_lambda * new_distortion_param[j] + neg_best_lambda * cur_param[j];
    Math1D::assign_weighted_combination(cur_param, best_lambda, new_distortion_param, neg_best_lambda, cur_param);

    energy = best_energy;

    //for (uint k = 0; k < weight.size(); k++) {
    //  sum[k] = neg_best_lambda * sum[k] + best_lambda * new_sum[k];
    //}
    Math1D::assign_weighted_combination(sum, neg_best_lambda, sum, best_lambda, new_sum);
  }

  param.set_x(i, c, cur_param);

  return energy;
}

//compact form
double IBM3Trainer::nondeficient_m_step_with_interpolation(const Math3D::Tensor<double>& single_pos_count,
    const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
    const std::vector<double>& weight, uint i, uint c, double start_energy)
{
  assert(distortion_param_.xDim() == maxJ_);
  assert(i < distortion_param_.yDim());

  return nondeficient_m_step_with_interpolation_core(single_pos_count, open_pos, weight, distortion_param_, i, c, start_energy);
}

//compact form for the nonparametric setting
// make sure that you pass the count corresponding to J
double IBM3Trainer::nondeficient_m_step_with_interpolation(const Math3D::Tensor<double>& single_pos_count,
    const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, const std::vector<double>& weight,
    uint i, uint c, uint J, double start_energy)
{
  assert(open_pos.size() == weight.size());
  assert(single_pos_count.xDim() == J);
  assert(distortion_prob_[J - 1].xDim() == J);

  return nondeficient_m_step_with_interpolation_core(single_pos_count, open_pos, weight, distortion_prob_[J - 1], i, c, start_energy, true);
}

/*virtual*/ long double IBM3Trainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& cur_lookup,
    const Math1D::Vector<AlignBaseType>& alignment) const
{
  return alignment_prob(source,target,cur_lookup,alignment,0);
}

long double IBM3Trainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& cur_lookup,
                                        const Math1D::Vector<AlignBaseType>& alignment, const Math3D::Tensor<double>* distort_prob) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Math3D::Tensor<double>& cur_distort_prob = (distort_prob != 0) ? *distort_prob : distortion_prob_[curJ - 1];

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  }

  //std::cerr << "ap: fertility: " << fertility << std::endl;

  if (curJ < 2 * fertility[0])
    return 0.0;

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = target[i - 1];
    prob *= fertility_prob_[t_idx][fertility[i]];
    if (!no_factorial_)
      prob *= ld_fac_[fertility[i]];    //ldfac(fertility[i]);

    //     std::cerr << "fertility_factor(" << i << "): "
    //        << (ldfac(fertility[i]) * fertility_prob_[t_idx][fertility[i]])
    //        << std::endl;
  }

  for (uint j = 0; j < curJ; j++) {

    uint s_idx = source[j];
    uint aj = alignment[j];

    if (aj == 0)
      prob *= dict_[0][s_idx - 1];
    else {
      uint t_idx = target[aj - 1];
      prob *= dict_[t_idx][cur_lookup(j, aj - 1)] * cur_distort_prob(j, aj - 1, target_class_[target[aj -1]]);
      //       std::cerr << "dict-factor(" << j << "): "
      //                << dict_[t_idx][cur_lookup(j,aj-1)] << std::endl;
      //       std::cerr << "distort-factor(" << j << "): "
      //                << cur_distort_prob(j,aj-1) << std::endl;
    }
  }

  //std::cerr << "ap before empty word: " << prob << std::endl;

  //handle empty word
  const uint zero_fert = fertility[0];
  assert(zero_fert <= 2 * curJ);
  prob *= choose_factor_[curJ][zero_fert];
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  prob *= p_zero_pow_[zero_fert];
  prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  if (empty_word_model_ != FertNullNondeficient)
    prob *= och_ney_factor_[curJ][zero_fert];

  return prob;
}

long double IBM3Trainer::nondeficient_alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const
{
  SingleLookupTable aux_lookup;

  const SingleLookupTable& lookup = get_wordlookup(source_sentence_[s], target_sentence_[s], wcooc_, nSourceWords_, slookup_[s], aux_lookup);

  return nondeficient_alignment_prob(source_sentence_[s], target_sentence_[s], lookup, alignment);
}

long double IBM3Trainer::nondeficient_alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& cur_lookup, const Math1D::Vector<AlignBaseType>& alignment) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

  Storage1D<std::vector<AlignBaseType> > aligned_source_words(curI + 1);    //words are listed in ascending order

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    aligned_source_words[aj].push_back(j);
  }

  //std::cerr << "ap: fertility: " << fertility << std::endl;

  if (curJ < 2 * fertility[0])
    return 0.0;

  for (uint i = 1; i <= curI; i++) {
    const uint t_idx = target[i - 1];
    //NOTE: no factorial here
    prob *= fertility_prob_[t_idx][fertility[i]];
  }
  for (uint j = 0; j < curJ; j++) {

    const uint s_idx = source[j];
    const uint aj = alignment[j];

    if (aj == 0)
      prob *= dict_[0][s_idx - 1];
    else {
      uint t_idx = target[aj - 1];
      prob *= dict_[t_idx][cur_lookup(j, aj - 1)];
    }
  }

  prob *= nondeficient_distortion_prob(source, target, aligned_source_words);

  //std::cerr << "ap before empty word: " << prob << std::endl;

  //handle empty word
  const uint zero_fert = fertility[0];
  assert(zero_fert <= 2 * curJ);
  prob *= choose_factor_[curJ][zero_fert];      //ldchoose(curJ-fertility[0],fertility[0]);
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  prob *= p_zero_pow_[zero_fert];
  prob *= p_nonzero_pow_[curJ - 2 * zero_fert];
  // for (uint k=1; k <= zero_fert; k++)
  //   prob *= p_zero_;
  // for (uint k=1; k <= curJ-2*zero_fert; k++)
  //   prob *= p_nonzero_;

  return prob;
}

long double IBM3Trainer::nondeficient_distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) const
{
  //std::cerr << "start ndist" << std::endl;

//#define nondefver1

  const uint curJ = source.size();
  const uint curI = target.size();

  const Math3D::Tensor<double>& cur_distort_prob = distortion_prob_[curJ - 1];

  long double prob = 1.0;

  Storage1D<bool> filled(curJ, false);

  //std::cerr << "################# I=" << curI << ", J=" << curJ << std::endl;

  uint first_open = 0;
  uint last_open = curJ - 1;

  for (uint i = 1; i <= curI; i++) {

    //std::cerr << "i: " << i << std::endl;

    const std::vector<AlignBaseType>& cur_aligned = aligned_source_words[i];

    if (cur_aligned.size() == 0)
      continue;

    const uint c = target_class_[target[i-1]];

    //std::cerr << "**** i: " << i << std::endl;
    //std::cerr << "cur_aligned: " << cur_aligned << std::endl;

    /*** a) head word ***/
    const uint first = cur_aligned[0];
    assert(first < curJ);

    uint nToRemove = cur_aligned.size() - 1;

    double sum = 0.0;

#ifdef nondefver1
    std::cerr << "XXXXXXX" << std::endl;
    //this version is orders of magnitude slower
    std::vector<uint> open_pos;
    open_pos.reserve(curJ);
    for (uint jj = first_open; jj <= last_open; jj++) {
      if (!filled[jj])
        open_pos.push_back(jj);
    }

    first_open = open_pos[0];
    last_open = open_pos[open_pos.size()-nToRemove-1];

    for (uint k = 0; k < open_pos.size()-nToRemove; k++) {
      sum += cur_distort_prob(open_pos[k], i-1, c);
      assert(cur_distort_prob(open_pos[k], i-1, c) > 0.0);
    }
#else

    while (filled[first_open])
      first_open++;
    while (filled[last_open])
      last_open--;

    for (uint jj = first_open; jj <= last_open; jj++) {
      if (!filled[jj]) {
        //std::cerr << "old1: adding pos " << jj << std::endl;
        sum += cur_distort_prob(jj, i-1, c);
      }
    }

    if (nToRemove > 0) {
      //remove the aligned_source_words[i].size()-1 last terms as otherwise the given fertility cannot be fulfilled
      uint nRemoved = 0;

      //std::cerr << "1: removing " << nToRemove << " positions" << std::endl;

      for (int jj = curJ - 1; jj >= 0; jj--) {

        if (!filled[jj]) {
          sum -= cur_distort_prob(jj, i-1, c);
          nRemoved++;
          if (nRemoved == nToRemove)
            break;
        }
      }

      assert(nRemoved == nToRemove);
    }
#endif

    // if (!(sum >= cur_param[first] - 1e-4)) {
    // std::cerr << "J: " << curJ << ", I: " << curI << std::endl;
    // std::cerr << "**** i: " << i << std::endl;
    // std::cerr << "cur_aligned: " << cur_aligned << std::endl;
    // std::cerr << "sum: " << sum << std::endl;
    // std::cerr << "param[first]: " << cur_param[first] << std::endl;
    // std::cerr << "filled: " << filled << std::endl;
    // }

    // assert(sum >= cur_param[first] - 1e-4);

    sum = std::max(sum,cur_distort_prob(first, i-1, c));
    prob *= cur_distort_prob(first, i-1, c) / sum;

    assert(!filled[first]);
    filled[first] = true;
#ifdef nondefver1
    open_pos.remove(first);
#endif

    /*** b) remaining words ***/
    for (uint k = 1; k < cur_aligned.size(); k++) {

      nToRemove--;
      const uint j = cur_aligned[k];
      assert(j < curJ);

      double sum = 0.0;

#ifdef nondefver1
      //this version is orders of magnitude slower
      last_open = open_pos[open_pos.size()-nToRemove-1];

      for (uint k = 0; k < open_pos.size() - nToRemove; k++)
        sum += cur_distort_prob(open_pos[k], i-1, c);
#else
      for (uint jj = cur_aligned[k - 1] + 1; jj <= last_open; jj++) {
        if (!filled[jj]) {
          //std::cerr << "old2: adding pos " << jj << std::endl;
          sum += cur_distort_prob(jj, i-1, c);
        }
      }

      if (nToRemove > 0) {
        //remove the aligned_source_words[i].size()-1 last terms as otherwise the given fertility cannot be fulfilled
        uint nRemoved = 0;
        for (int jj = curJ - 1; jj >= 0; jj--) {

          if (!filled[jj]) {
            sum -= cur_distort_prob(jj, i-1, c);
            nRemoved++;
            if (nRemoved == nToRemove)
              break;
          }
        }

        assert(nRemoved == nToRemove);
      }
#endif

      sum = std::max(sum,cur_distort_prob(j, i-1, c));
      prob *= cur_distort_prob(j, i-1, c) / sum;

      //std::cerr << "removing pos " << j << std::endl;
      assert(!filled[j]);
      filled[j] = true;
#ifdef nondefver1
      open_pos.remove(j);
#endif
    }
  }

  //std::cerr << "end ndist" << std::endl;

  return prob;
}

long double IBM3Trainer::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
    Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double>& swap_prob,
    Math1D::Vector<AlignBaseType>& alignment) const
{
  if (nondeficient_) {
    return nondeficient_hillclimbing(source, target, lookup, nIter, fertility, expansion_prob, swap_prob, alignment);
  }
  //std::cerr << "*************** hillclimb()" << std::endl;
  //std::cerr << "start alignment: " << alignment << std::endl;

  double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Math3D::Tensor<double>& cur_distort_prob = distortion_prob_[curJ - 1];

  Math2D::Matrix<double> jcost(curJ,curI+1);
  for (uint aj=1; aj <= curI; aj++) {

    const uint taj = target[aj-1];
    const Math1D::Vector<double>& cur_dict = dict_[taj];
    for (uint j=0; j < curJ; j++)
      jcost(j,aj) = cur_dict[lookup(j, aj - 1)] * cur_distort_prob(j, aj - 1, target_class_[taj]);
  }
  for (uint j=0; j < curJ; j++)
    jcost(j,0) = dict_[0][source[j] - 1];


  assert(cur_distort_prob.size() > 0);

  /**** calculate probability of so far best known alignment *****/
  long double base_prob = 1.0;

  fertility.resize_dirty(curI+1);
  fertility.set_constant(0);

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;

    base_prob *= jcost(j,aj);
    assert(jcost(j,aj) > 0.0);
  }

  assert(2 * fertility[0] <= curJ);
  assert(!isnan(base_prob));

  //handle fertility prob
  for (uint i = 0; i < curI; i++) {

    //std::cerr << "i: " << i << std::endl;

    const uint fert = fertility[i + 1];
    const uint t_idx = target[i];

    //std::cerr << "t_idx: " << t_idx << std::endl;

    if (!(fertility_prob_[t_idx][fert] > 0.0)) {

      std::cerr << "fert_prob[" << t_idx << "][" << fert << "]: " << fertility_prob_[t_idx][fert] << std::endl;
      //std::cerr << "target sentence #" << s << ", word #" << i << std::endl;
      std::cerr << "alignment: " << alignment << std::endl;
    }

    assert(fertility_prob_[t_idx][fert] > 0.0);

    base_prob *= fertility_prob_[t_idx][fert];
    if (!no_factorial_)
      base_prob *= ld_fac_[fert];
  }

  assert(base_prob > 0.0);
  assert(!isnan(base_prob));

  //std::cerr << "base prob before empty word: " << base_prob << std::endl;

  //handle fertility of empty word
  uint zero_fert = fertility[0];
  if (curJ < 2 * zero_fert) {
    std::cerr << "WARNING: alignment startpoint for HC violates the assumption that less words "
              << " are aligned to NULL than to a real word" << std::endl;
  }
  else {

    base_prob *= choose_factor_[curJ][zero_fert];
    update_nullpow(zero_fert, curJ - 2 * zero_fert);
    base_prob *= p_zero_pow_[zero_fert];
    base_prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

    if (empty_word_model_ != FertNullOchNey)
      base_prob *= och_ney_factor_[curJ][zero_fert];
  }

  if (isinf(base_prob) || isnan(base_prob) || base_prob <= 0.0) {

    INTERNAL_ERROR << "base prob in hillclimbing is " << base_prob <<
                   ", alignment " << alignment << std::endl;
    exit(1);
  }
#ifndef NDEBUG
  long double check = alignment_prob(source, target, lookup, alignment);

  long double check_ratio = base_prob / check;

  if (!(check_ratio > 0.999 && check_ratio < 1.001)) {
    std::cerr << "alignment: " << alignment << std::endl;
    std::cerr << "fertility: " << fertility << std::endl;
    std::cerr << "base_prob: " << base_prob << std::endl;
    std::cerr << "check: " << check << std::endl;

    std::cerr << "ratio: " << check_ratio << std::endl;
  }

  if (base_prob > 1e-300 || check > 1e-300)
    assert(check_ratio > 0.999 && check_ratio < 1.001);

  assert(!isnan(base_prob));
#endif

  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);
  //swap_prob.set_constant(0.0);
  //expansion_prob.set_constant(0.0);

  uint count_iter = 0;

  bool have_warned_for_empty_word = false;

  Math1D::Vector<long double> fert_increase_factor(curI + 1);
  Math1D::Vector<long double> fert_decrease_factor(curI + 1);

  for (uint i = 1; i <= curI; i++) {

    uint t_idx = target[i - 1];
    uint cur_fert = fertility[i];

    assert(fertility_prob_[t_idx][cur_fert] > 0.0);

    if (cur_fert > 0) {
      fert_decrease_factor[i] = ((long double)fertility_prob_[t_idx][cur_fert - 1]) / fertility_prob_[t_idx][cur_fert];

      if (!no_factorial_)
        fert_decrease_factor[i] /= cur_fert;
    }
    else
      fert_decrease_factor[i] = 0.0;

    if (cur_fert + 1 < fertility_prob_[t_idx].size()
        && cur_fert + 1 <= fertility_limit_[t_idx]) {
      fert_increase_factor[i] = ((long double)fertility_prob_[t_idx][cur_fert + 1]) / fertility_prob_[t_idx][cur_fert];

      if (!no_factorial_)
        fert_increase_factor[i] *= cur_fert + 1;
    }
    else
      fert_increase_factor[i] = 0.0;
  }

  fert_decrease_factor[0] = 0.0;
  fert_increase_factor[0] = 0.0;

  while (true) {

    count_iter++;
    nIter++;

    //std::cerr << "****************** starting new hillclimb iteration, current best prob: " << base_prob << std::endl;

    long double best_prob = base_prob;
    bool best_change_is_move = false;
    uint best_move_j = MAX_UINT;
    uint best_move_aj = MAX_UINT;
    uint best_swap_j1 = MAX_UINT;
    uint best_swap_j2 = MAX_UINT;

    /**** scan neighboring alignments and keep track of the best one that is better
     ****  than the current alignment  ****/

    //a) expansion moves

    //std::cerr << "considering moves" << std::endl;

    if (fert_increase_factor[0] == 0.0 && p_zero_ > 0.0) {
      if (curJ >= 2 * (zero_fert + 1)) {

        fert_increase_factor[0] = (curJ - 2 * zero_fert) * (curJ - 2 * zero_fert - 1) * p_zero_
                                  / ((curJ - zero_fert) * (zero_fert + 1) * p_nonzero_ * p_nonzero_);

#ifndef NDEBUG
        long double old_const = choose_factor_[curJ][zero_fert + 1] * p_zero_
                                / (choose_factor_[curJ][zero_fert] * p_nonzero_ * p_nonzero_);

        double ratio = fert_increase_factor[0] / old_const;

        assert(ratio >= 0.975 && ratio <= 1.05);
#endif

        if (empty_word_model_ != FertNullNondeficient) {
          fert_increase_factor[0] *= (zero_fert + 1) / ((long double)curJ);
        }
      }
      else {
        if (curJ > 3 && !have_warned_for_empty_word) {
          std::cerr << "WARNING: reached limit of allowed number of zero-aligned words, "
                    << "J=" << curJ << ", zero_fert =" << zero_fert << std::endl;
          have_warned_for_empty_word = true;
        }
      }
    }

    if (fert_decrease_factor[0] == 0.0 && zero_fert > 0) {

      fert_decrease_factor[0] = (curJ - zero_fert + 1) * zero_fert * p_nonzero_ * p_nonzero_
                                / ((curJ - 2 * zero_fert + 1) * (curJ - 2 * zero_fert + 2) * p_zero_);

#ifndef NDEBUG
      long double old_const = choose_factor_[curJ][zero_fert - 1] * p_nonzero_ * p_nonzero_
                              / (choose_factor_[curJ][zero_fert] * p_zero_);

      double ratio = fert_decrease_factor[0] / old_const;

      assert(ratio >= 0.975 && ratio <= 1.05);
#endif

      if (empty_word_model_ != FertNullNondeficient) {
        fert_decrease_factor[0] *= curJ / ((long double)zero_fert);
      }
    }

    for (uint j = 0; j < curJ; j++) {

      const uint aj = alignment[j];
      assert(fertility[aj] > 0);
      expansion_prob(j, aj) = 0.0;

      //std::cerr << "j: " << j << ", aj: " << aj << std::endl;

      assert(jcost(j,aj) > 1e-305);
      const long double mod_base_prob = base_prob * fert_decrease_factor[aj] / jcost(j, aj);

#ifndef NDEBUG
      if (isnan(mod_base_prob)) {

        std::cerr << "base prob: " << base_prob << std::endl;
        std::cerr << "aj: " << aj << std::endl;

        if (aj > 0) {

          const uint t_idx = target[aj - 1];

          std::cerr << " mult by " << fert_decrease_factor[aj] << " / " << dict_[t_idx][lookup(j, aj - 1)] << std::endl;
          std::cerr << " div by " << cur_distort_prob(j, aj - 1, target_class_[t_idx]) << std::endl;

          uint cur_fert = fertility[aj];

          std::cerr << "fert dec. factor = " << ((long double) fertility_prob_[t_idx][cur_fert - 1])
                    << " / (" << cur_fert << " * " << fertility_prob_[t_idx][cur_fert] << ")" << std::endl;
        }
      }
#endif

      assert(!isnan(mod_base_prob));

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        //      std::cerr << "examining move " << j << " -> " << cand_aj << " (instead of " << aj << ") in iteration #"
        //                << count_iter << std::endl;
        //      std::cerr << "cand_aj has then fertility " << (fertility[cand_aj]+1) << std::endl;
        //      std::cerr << "current aj reduces its fertility from " << fertility[aj] << " to " << (fertility[aj]-1)
        //                << std::endl;

        //      if (cand_aj != 0 && cand_aj != aj) {
        //        std::cerr << "previous fert prob of candidate: "
        //                  << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]] << std::endl;
        //        std::cerr << "new fert prob of candidate: "
        //                  << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]+1] << std::endl;
        //      }
        //      if (aj != 0) {
        //        std::cerr << "previous fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]] << std::endl;
        //        std::cerr << "new fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]-1] << std::endl;
        //      }

        //      if (aj != 0)
        //        std::cerr << "prev distort prob: " << cur_distort_prob(j,aj-1) << std::endl;
        //      if (cand_aj != 0)
        //        std::cerr << "new distort prob: " << cur_distort_prob(j,cand_aj-1) << std::endl;

        if (cand_aj != aj) {

          if (fert_increase_factor[0] == 0.0) {
            expansion_prob(j, cand_aj) = 0.0;
            continue;
          }

          const long double hyp_prob = mod_base_prob * jcost(j,cand_aj) * fert_increase_factor[cand_aj];

          //std::cerr << "hyp_prob: " << hyp_prob << std::endl;

#ifndef NDEBUG
          if (isnan(hyp_prob)) {
            INTERNAL_ERROR << " nan in move " << j << " -> " << cand_aj << " . Exiting." << std::endl;

            std::cerr << "mod_base_prob: " << mod_base_prob << std::endl;
            if (cand_aj != 0) {

              const uint t_idx = target[cand_aj - 1];

              std::cerr << "dict-factor: " << dict_[t_idx][lookup(j, cand_aj - 1)] << std::endl;
              std::cerr << "fert-factor: " << fert_increase_factor[cand_aj] << std::endl;
              std::cerr << "distort-factor: " << cur_distort_prob(j, cand_aj - 1, target_class_[t_idx]) << std::endl;

              //std::cerr << "distort table: " << cur_distort_prob << std::endl;
            }
            exit(1);
          }
#endif

          assert(!isnan(hyp_prob));

#ifndef NDEBUG
          if (hyp_prob > 0.0) {
            Math1D::Vector<AlignBaseType> cand_alignment = alignment;
            cand_alignment[j] = cand_aj;

            double check_ratio = hyp_prob / alignment_prob(source,target,lookup,cand_alignment);
            assert(check_ratio >= 0.99 && check_ratio < 1.01);
          }
#endif
          assert(!isnan(hyp_prob));

          expansion_prob(j, cand_aj) = hyp_prob;

          if (hyp_prob > best_prob) {
            //std::cerr << "improvement of " << (hyp_prob - best_prob) << std::endl;

            best_prob = hyp_prob;
            best_change_is_move = true;
            best_move_j = j;
            best_move_aj = cand_aj;
          }
        }
      }
    }

    //     if (improvement) {
    //       std::cerr << "expansion improvement for sentence pair #" << s << std::endl;
    //     }

    //std::cerr << "improvements for moves: " << improvement << std::endl;
    //std::cerr << "considering swaps" << std::endl;

    //b) swap_moves (NOTE that swaps do not affect the fertilities)
    for (uint j1 = 0; j1 < curJ; j1++) {

      //std::cerr << "j1: " << j1 << std::endl;

      const uint aj1 = alignment[j1];

      const long double hyp_prob_common = base_prob / jcost(j1,aj1);

      for (uint j2 = j1 + 1; j2 < curJ; j2++) {

        //std::cerr << "j2: " << j2 << std::endl;

        const uint aj2 = alignment[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1, j2) = 0.0;
        }
        else {

          const long double hyp_prob = hyp_prob_common * jcost(j2,aj1) * jcost(j1,aj2) / jcost(j2,aj2);

#ifndef NDEBUG
          if (hyp_prob > 0.0) {
            Math1D::Vector<AlignBaseType> cand_alignment = alignment;
            cand_alignment[j1] = aj2;
            cand_alignment[j2] = aj1;

            double check_ratio = hyp_prob / alignment_prob(source,target,lookup,cand_alignment);
            assert(check_ratio > 0.999 && check_ratio < 1.001);
          }
#endif
          assert(!isnan(hyp_prob));

          swap_prob(j1, j2) = hyp_prob;

          if (hyp_prob > best_prob) {

            best_change_is_move = false;
            best_prob = hyp_prob;
            best_swap_j1 = j1;
            best_swap_j2 = j2;
          }
        }
      }
    }

    //std::cerr << "[s=" << s << "] swaps done, improvement: " << improvement << std::endl;

    if (best_prob < improvement_factor * base_prob || count_iter > nMaxHCIter_) {
      if (count_iter > nMaxHCIter_)
        std::cerr << "HC Iteration limit reached" << std::endl;
      break;
    }
    //update alignment
    if (best_change_is_move) {
      const uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      //std::cerr << "moving source pos " << best_move_j << " from " << cur_aj << " to " << best_move_aj << std::endl;

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;

      if (cur_aj * best_move_aj == 0) {
        //signal recomputation
        zero_fert = fertility[0];
        fert_decrease_factor[0] = 0.0;
        fert_increase_factor[0] = 0.0;
      }

      if (cur_aj != 0) {

        const uint t_idx = target[cur_aj - 1];
        const uint cur_fert = fertility[cur_aj];

        if (cur_fert > 0) {
          fert_decrease_factor[cur_aj] = ((long double)fertility_prob_[t_idx][cur_fert - 1]) / fertility_prob_[t_idx][cur_fert];

          if (!no_factorial_)
            fert_decrease_factor[cur_aj] /= cur_fert;
        }
        else
          fert_decrease_factor[cur_aj] = 0.0;

        fert_increase_factor[cur_aj] = ((long double)fertility_prob_[t_idx][cur_fert + 1]) / fertility_prob_[t_idx][cur_fert];

        if (!no_factorial_)
          fert_increase_factor[cur_aj] *= cur_fert + 1;
      }

      if (best_move_aj != 0) {

        const uint t_idx = target[best_move_aj - 1];
        const uint cur_fert = fertility[best_move_aj];

        fert_decrease_factor[best_move_aj] = ((long double)fertility_prob_[t_idx][cur_fert - 1]) / fertility_prob_[t_idx][cur_fert];

        if (!no_factorial_)
          fert_decrease_factor[best_move_aj] /= cur_fert;

        if (cur_fert + 1 < fertility_prob_[t_idx].size() && cur_fert + 1 <= fertility_limit_[t_idx]) {

          fert_increase_factor[best_move_aj] = ((long double)fertility_prob_[t_idx][cur_fert + 1]) / fertility_prob_[t_idx][cur_fert];

          if (!no_factorial_)
            fert_increase_factor[best_move_aj] *= cur_fert + 1;
        }
        else
          fert_increase_factor[best_move_aj] = 0.0;
      }
    }
    else {
      //std::cerr << "swapping: j1=" << best_swap_j1 << std::endl;
      //std::cerr << "swapping: j2=" << best_swap_j2 << std::endl;

      uint cur_aj1 = alignment[best_swap_j1];
      uint cur_aj2 = alignment[best_swap_j2];

      //std::cerr << "old aj1: " << cur_aj1 << std::endl;
      //std::cerr << "old aj2: " << cur_aj2 << std::endl;

      assert(cur_aj1 != cur_aj2);

      alignment[best_swap_j1] = cur_aj2;
      alignment[best_swap_j2] = cur_aj1;
    }

    //std::cerr << "probability improved from " << base_prob << " to " << best_prob << std::endl;

#ifndef NDEBUG
    long double check_ratio = best_prob / alignment_prob(source, target, lookup, alignment);
    if (best_prob > 1e-300 && !(check_ratio > 0.995 && check_ratio < 1.005)) {

      std::cerr << "hc iter " << count_iter << std::endl;
      std::cerr << "alignment: " << alignment << std::endl;
      std::cerr << "fertility: " << fertility << std::endl;

      std::cerr << "no factorial: " << no_factorial_ << std::endl;

      if (best_change_is_move) {
        std::cerr << "moved j=" << best_move_j << " -> aj=" << best_move_aj << std::endl;
      }
      else {
        std::cerr << "swapped j1=" << best_swap_j1 << " and j2=" << best_swap_j2 << std::endl;
        std::cerr << "now aligned to " << alignment[best_swap_j1] << " and " << alignment[best_swap_j2] << std::endl;
      }

      std::cerr << "probability improved from " << base_prob << " to " << best_prob << std::endl;
      std::cerr << "check prob: " << alignment_prob(source, target, lookup, alignment) << std::endl;
      std::cerr << "check_ratio: " << check_ratio << std::endl;
    }
    //std::cerr << "check_ratio: " << check_ratio << std::endl;
    if (best_prob > 1e-275)
      assert(check_ratio > 0.995 && check_ratio < 1.005);
#endif

    base_prob = best_prob;
  }

  //std::cerr << "leaving hillclimb" << std::endl;

  assert(!isnan(base_prob));

  assert(2 * fertility[0] <= curJ);

  //symmetrize swap_prob
  for (uint j1 = 0; j1 < curJ; j1++) {

    swap_prob(j1, j1) = 0.0;

    for (uint j2 = j1 + 1; j2 < curJ; j2++) {

      swap_prob(j2, j1) = swap_prob(j1, j2);
    }
  }

  return base_prob;
}

long double IBM3Trainer::nondeficient_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
    Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double>& swap_prob,
    Math1D::Vector<AlignBaseType>& alignment) const
{
  //std::cerr << "nondef hc" << std::endl;

  /**** calculate probability of the passed alignment *****/

  double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size();

  Storage1D<std::vector<AlignBaseType> > aligned_source_words(curI + 1);
  fertility.resize_dirty(curI+1);
  fertility.set_constant(0);

  Math2D::Matrix<double> dict(curJ,curI+1);
  compute_dictmat_fertform(source, lookup, target, dict_, dict);

  for (uint j = 0; j < curJ; j++) {

    const uint aj = alignment[j];
    aligned_source_words[aj].push_back(j);
    fertility[aj]++;
  }

  long double base_distortion_prob = nondeficient_distortion_prob(source, target, aligned_source_words);
  long double base_prob = base_distortion_prob;

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = target[i - 1];
    //NOTE: no factorial here
    base_prob *= fertility_prob_[t_idx][fertility[i]];
  }
  for (uint j = 0; j < curJ; j++) {

    uint aj = alignment[j];
    base_prob *= dict(j,aj);
  }

  uint zero_fert = fertility[0];
  base_prob *= choose_factor_[curJ][zero_fert];
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  base_prob *= p_zero_pow_[zero_fert];
  base_prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  //DEBUG
#ifndef NDEBUG
  long double check_prob = nondeficient_alignment_prob(source, target, lookup, alignment);
  double check_ratio = base_prob / check_prob;
  assert(check_ratio >= 0.99 && check_ratio <= 1.01);
#endif
  //END_DEBUG

  uint count_iter = 0;

  Storage1D<std::vector<AlignBaseType> > hyp_aligned_source_words = aligned_source_words;

  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);
  //swap_prob.set_constant(0.0);
  //expansion_prob.set_constant(0.0);

  long double empty_word_increase_const = 0.0;
  long double empty_word_decrease_const = 0.0;

  while (true) {

    count_iter++;
    nIter++;

    //std::cerr << "****************** starting new nondef hc iteration, current best prob: " << base_prob << std::endl;

    if (empty_word_increase_const == 0.0 && p_zero_ > 0.0 && curJ >= 2 * (zero_fert + 1)) {

      empty_word_increase_const = (curJ - 2 * zero_fert) * (curJ - 2 * zero_fert - 1) * p_zero_
                                  / ((curJ - zero_fert) * (zero_fert + 1) * p_nonzero_ * p_nonzero_);
    }

    if (empty_word_decrease_const == 0.0 && zero_fert > 0) {

      empty_word_decrease_const = (curJ - zero_fert + 1) * zero_fert * p_nonzero_ * p_nonzero_
                                  / ((curJ - 2 * zero_fert + 1) * (curJ - 2 * zero_fert + 2) * p_zero_);

#ifndef NDEBUG
      long double old_const = choose_factor_[curJ][zero_fert - 1] * p_nonzero_ * p_nonzero_ / (choose_factor_[curJ][zero_fert] * p_zero_);
      long double ratio = empty_word_decrease_const / old_const;

      assert(ratio >= 0.975 && ratio <= 1.05);
#endif
    }

    long double best_prob = base_prob;
    bool best_change_is_move = false;
    uint best_move_j = MAX_UINT;
    uint best_move_aj = MAX_UINT;
    uint best_swap_j1 = MAX_UINT;
    uint best_swap_j2 = MAX_UINT;

    /**** scan neighboring alignments and keep track of the best one that is better
     ****  than the current alignment  ****/

    /**** expansion moves ****/

    for (uint j = 0; j < curJ; j++) {

      //std::cerr << "j: " << j << std::endl;

      //const uint s_idx = source[j];

      const uint aj = alignment[j];
      expansion_prob(j, aj) = 0.0;

      const double old_dict_prob = dict(j,aj);
      const long double leaving_prob_common = base_distortion_prob * old_dict_prob;

      vec_erase<AlignBaseType>(hyp_aligned_source_words[aj],j);

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        if (aj == cand_aj) {
          expansion_prob(j, cand_aj) = 0.0;
          continue;
        }

        if (cand_aj > 0) { //better to check this before computing distortion probs
          if ((fertility[cand_aj] + 1) > fertility_limit_[target[cand_aj - 1]]) {
            expansion_prob(j, cand_aj) = 0.0;
            continue;
          }
        }
        if (cand_aj == 0 && 2 * fertility[0] + 2 > curJ) {  //better to check this before computing distortion probs
          expansion_prob(j, cand_aj) = 0.0;
          continue;
        }

        const double new_dict_prob = dict(j,cand_aj);

        if (new_dict_prob < 1e-8)
          expansion_prob(j, cand_aj) = 0.0;
        else {
          hyp_aligned_source_words[cand_aj].push_back(j);
          vec_sort(hyp_aligned_source_words[cand_aj]);

          long double leaving_prob = leaving_prob_common;
          long double incoming_prob = nondeficient_distortion_prob(source, target, hyp_aligned_source_words) * new_dict_prob;

          if (aj > 0) {
            uint tidx = target[aj - 1];
            leaving_prob *= fertility_prob_[tidx][fertility[aj]];
            incoming_prob *= fertility_prob_[tidx][fertility[aj] - 1];
          }
          else {

            //compute null-fert-model (null-fert decreases by 1)

            incoming_prob *= empty_word_decrease_const;
          }

          if (cand_aj > 0) {
            uint tidx = target[cand_aj - 1];
            leaving_prob *= fertility_prob_[tidx][fertility[cand_aj]];
            if (fertility[cand_aj] + 1 <= fertility_limit_[target[cand_aj - 1]])
              incoming_prob *= fertility_prob_[tidx][fertility[cand_aj] + 1];
            else
              incoming_prob = 0.0;
          }
          else {

            //compute null-fert-model (zero-fert goes up by 1)

            incoming_prob *= empty_word_increase_const; //will be 0.0 if no more words can be aligned to the empty word
          }

          long double incremental_cand_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
          if (incremental_cand_prob > 0.0) {
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j] = cand_aj;

            long double cand_prob = nondeficient_alignment_prob(source,target,lookup,hyp_alignment);

            long double ratio = incremental_cand_prob / cand_prob;

            //   if (! (ratio >= 0.99 && ratio <= 1.01)) {
            //     std::cerr << "j: " << j << ", aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
            //     std::cerr << "incremental: " << incremental_cand_prob << ", standalone: " << cand_prob << std::endl;
            //   }
            assert(ratio >= 0.99 && ratio <= 1.01);
          }
#endif

          expansion_prob(j, cand_aj) = incremental_cand_prob;

          if (incremental_cand_prob > best_prob) {
            best_change_is_move = true;
            best_prob = incremental_cand_prob;
            best_move_j = j;
            best_move_aj = cand_aj;
          }
          //restore for the next iteration
          hyp_aligned_source_words[cand_aj] = aligned_source_words[cand_aj];
        }
      }

      hyp_aligned_source_words[aj] = aligned_source_words[aj];
    }

    /**** swap moves ****/
    for (uint j1 = 0; j1 < curJ; j1++) {

      //std::cerr << "j1: " << j1 << std::endl;

      const uint aj1 = alignment[j1];

      const long double common_prob = base_prob / (base_distortion_prob * dict(j1,aj1));

      for (uint j2 = j1 + 1; j2 < curJ; j2++) {

        //std::cerr << "j2: " << j2 << std::endl;

        const uint aj2 = alignment[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1, j2) = 0.0;
        }
        else {

          vec_replace<AlignBaseType>(hyp_aligned_source_words[aj1],j1,j2);
          vec_replace<AlignBaseType>(hyp_aligned_source_words[aj2],j2,j1);

          vec_sort(hyp_aligned_source_words[aj1]);
          vec_sort(hyp_aligned_source_words[aj2]);

          //long double incremental_prob = base_prob / base_distortion_prob *
          //                               nondeficient_distortion_prob(source, target, hyp_aligned_source_words);

          long double incremental_prob = common_prob * nondeficient_distortion_prob(source, target, hyp_aligned_source_words);


          incremental_prob *= dict(j2,aj1); // / dict(j1,aj1);
          incremental_prob *= dict(j1,aj2) / dict(j2,aj2);

#ifndef NDEBUG
          if (incremental_prob > 0.0) {
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            std::swap(hyp_alignment[j1],hyp_alignment[j2]);
            long double cand_prob = nondeficient_alignment_prob(source,target,lookup,hyp_alignment);

            double ratio = cand_prob / incremental_prob;
            assert(ratio > 0.99 && ratio < 1.01);
          }
#endif

          swap_prob(j1, j2) = incremental_prob;

          if (incremental_prob > best_prob) {
            best_change_is_move = false;
            best_prob = incremental_prob;
            best_swap_j1 = j1;
            best_swap_j2 = j2;
          }
          //restore for the next iteration
          hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
          hyp_aligned_source_words[aj2] = aligned_source_words[aj2];
        }
      }
    }

    /**** update to best alignment ****/

    if (best_prob < improvement_factor * base_prob || count_iter > nMaxHCIter_) {
      if (count_iter > nMaxHCIter_)
        std::cerr << "HC Iteration limit reached" << std::endl;
      break;
    }
    //update alignment
    if (best_change_is_move) {
      const uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      //std::cerr << "moving source pos" << best_move_j << " from " << cur_aj << " to " << best_move_aj << std::endl;

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;

      if (cur_aj * best_move_aj == 0) {
        //signal recomputation
        zero_fert = fertility[0];
        empty_word_increase_const = 0.0;
        empty_word_decrease_const = 0.0;
      }

      vec_erase<AlignBaseType>(aligned_source_words[cur_aj], best_move_j);
      aligned_source_words[best_move_aj].push_back(best_move_j);
      vec_sort(aligned_source_words[best_move_aj]);

      hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
      hyp_aligned_source_words[best_move_aj] = aligned_source_words[best_move_aj];
    }
    else {
      //std::cerr << "swapping: j1=" << best_swap_j1 << std::endl;
      //std::cerr << "swapping: j2=" << best_swap_j2 << std::endl;

      uint cur_aj1 = alignment[best_swap_j1];
      uint cur_aj2 = alignment[best_swap_j2];

      //std::cerr << "old aj1: " << cur_aj1 << std::endl;
      //std::cerr << "old aj2: " << cur_aj2 << std::endl;

      assert(cur_aj1 != cur_aj2);

      alignment[best_swap_j1] = cur_aj2;
      alignment[best_swap_j2] = cur_aj1;

      vec_replace<AlignBaseType>(aligned_source_words[cur_aj1],best_swap_j1,best_swap_j2);
      vec_replace<AlignBaseType>(aligned_source_words[cur_aj2],best_swap_j2,best_swap_j1);

      vec_sort(aligned_source_words[cur_aj1]);
      vec_sort(aligned_source_words[cur_aj2]);

      hyp_aligned_source_words[cur_aj1] = aligned_source_words[cur_aj1];
      hyp_aligned_source_words[cur_aj2] = aligned_source_words[cur_aj2];
    }

    base_prob = best_prob;
    base_distortion_prob = nondeficient_distortion_prob(source, target, aligned_source_words);
  }

  //symmetrize swap_prob
  for (uint j1 = 0; j1 < curJ; j1++) {

    swap_prob(j1, j1) = 0.0;

    for (uint j2 = j1 + 1; j2 < curJ; j2++) {

      swap_prob(j2, j1) = swap_prob(j1, j2);
    }
  }

  return base_prob;
}

void IBM3Trainer::compute_dist_param_gradient(const ReducedIBM3ClassDistortionModel& distort_grad, const Math3D::Tensor<double>& distort_param,
    Math3D::Tensor<double>& distort_param_grad) const
{
  distort_param_grad.resize(distortion_param_.xDim(), distortion_param_.yDim(), distortion_param_.zDim());
  distort_param_grad.set_constant(0.0);

  for (uint J = 0; J < distort_grad.size(); J++) {

    for (uint i = 0; i < distort_grad[J].yDim(); i++) {

      for (uint c = 0; c < distort_grad[J].zDim(); c++) {

        double param_sum = 0.0;
        double product_sum = 0.0;

        if (par_mode_ == IBM23ParByPosition) {

          for (uint j = 0; j < distort_grad[J].xDim(); j++) {

            //general coefficient: distort_grad[J](j,i) = distort_param(j,i) /  \sum_j' distort_param(j',i)

            //quotient rule: u(x) = distort_param(j,i), v(x) = \sum_j' distort_param(j',i)
            // numerator regarding distort_param(j,i): u'(x)v(x) - v'(x) u(x) = (\sum_j distort_param(j,i)) - distort_param(j,i) = v(x) - u(x)
            // numerator regarding distort_param(j',i): u'(x)v(x) - v'(x) u(x) = -distort_param(j,i) for j'!=j
            // hence, the numerator for all terms has a final -distort_param(j,i) component
            // denominator: [v(x)]² = param_sum²

            //gradient for this:
            //  distort_grad[J](j,i) * (param_sum - distort_param(j,i)) / param_sum² for j
            //  distort_grad[J](j,i) * -distort_param(j,i) / param_sum² for j'!=j
            // both have distort_grad[J](j,i) *-distort_param(j,i) / param_sum² in common,
            //      for j there is an additional distort_grad[J](j,i)/param_sum term

            // combining all js:
            // -summing the common component gives  - (\sum j' grad_j' * param_j') / param_sum²
            // -for each j additionally distort_grad[J](j,i) * param_sum / param_sum² = grad_j / param_sum

            param_sum += distort_param(j, i, c);
            product_sum += distort_grad[J](j, i, c) * distort_param(j, i, c);
          }

          param_sum = std::max(fert_min_param_entry, param_sum);
          const double combined = -product_sum / (param_sum * param_sum);

          for (uint j = 0; j < distort_grad[J].xDim(); j++) {
            distort_param_grad(j, i, c) += combined  //combined term
                                           + (distort_grad[J](j, i, c) / param_sum);   //term for j
          }
        }
        else {

          for (uint j = 0; j < distort_grad[J].xDim(); j++) {
            param_sum += distort_param(maxI_ - 1 + j - i, 0, c);
            product_sum += distort_grad[J](j, i, c) * distort_param(maxI_ - 1 + j - i, 0, c);
          }

          param_sum = std::max(fert_min_param_entry, param_sum);
          const double combined = -product_sum / (param_sum * param_sum);

          for (uint j = 0; j < distort_grad[J].xDim(); j++) {
            distort_param_grad(maxI_ - 1 + j - i, 0, c) += combined  //combined term
                + (distort_grad[J](j, i, c) / param_sum);   //term for j
          }
        }
      }
    }
  }
}

/* virtual */
void IBM3Trainer::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  common_prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();
  const uint I = target.size();

  /*** check if respective distortion table is present. If not, create one from the parameters ***/

  if (distortion_param_.xDim() < J)
    distortion_param_.resize(J, distortion_param_.yDim(), distortion_param_.zDim(), 1e-8);
  if (distortion_param_.yDim() < I)
    distortion_param_.resize(distortion_param_.xDim(), I, distortion_param_.zDim(), 1e-8);

  if (distortion_prob_.size() < J)
    distortion_prob_.resize(J);

  if (distortion_prob_[J - 1].yDim() < I) {

    //std::cerr << "extending" << std::endl;

    assert(distortion_prob_[J - 1].xDim() == J);
    uint oldYDim = distortion_prob_[J - 1].yDim();

    ReducedIBM3ClassDistortionModel temp_prob(J, MAKENAME(temp_prob));
    temp_prob[J - 1].resize(J, I, distortion_param_.zDim(), 1e-8);

    par2nonpar_distortion(temp_prob);

    distortion_prob_[J - 1].resize(J, I, distortion_param_.zDim());

    for (uint i = oldYDim; i < I; i++) {
      for (uint j = 0; j < J; j++) {
        for (uint c = 0; c < distortion_param_.zDim(); c++) {
          distortion_prob_[J - 1](j, i, c) = temp_prob[J - 1](j, i, c);
        }
      }
    }
  }
}

/*virtual */
long double IBM3Trainer::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();
  const uint I = target.size();

  bool use_ilp = (viterbi_ilp_mode_ != IlpOff && !nondeficient_);

#ifndef HAS_CBC
  use_ilp = false;
#endif

  //create matrices
  Math2D::Matrix<long double> expansion_prob(J, I + 1);
  Math2D::Matrix<long double> swap_prob(J, J);

  Math1D::Vector<uint> fertility(I + 1, 0);

  uint nIter;

  long double hc_prob = update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility, expansion_prob, swap_prob, alignment);

  if (use_ilp)
    return expl(-compute_viterbi_alignment_ilp(source, target, lookup, alignment));
  else
    return hc_prob;
}

class CountStructure {
public:

  CountStructure() : main_count_(0.0)
  {
  }

  void unpack_compact(uint i, uint c, uint J, uint maxJ, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
                      std::map<Math1D::Vector<uchar,uchar>,double>& nondef_count, Math3D::Tensor<double>& par_count) const;

  //diffpar
  void unpack_compact(uint offset, uint J, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
                      const Storage1D<WordClassType>& tclass, Storage1D<std::map<Math1D::Vector<ushort, uchar>, double> >& nondef_count,
                      Math3D::Tensor<double>& par_count) const;

  double main_count_;

  std::map<uchar,std::map<uchar,double> > exp_count_;
  std::map<uchar,std::map<uchar,double> > swap_count_;
};

void CountStructure::unpack_compact(uint i, uint c, uint J, uint maxJ, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
                                    std::map<Math1D::Vector<uchar,uchar>,double>& nondef_count, Math3D::Tensor<double>& par_count) const
{

  Math1D::Vector<uchar> main_alignment(J, 255);
  for (uint ii = 0; ii < main_aligned_source_words.size(); ii++) {
    for (uint k = 0; k < main_aligned_source_words[ii].size(); k++) {
      uint j = main_aligned_source_words[ii][k];
      main_alignment[j] = ii;
    }
  }

  //it is faster to accumulate an intermediate map first
  std::map<Math1D::Vector<uchar,uchar>,double> temp_count;

  const uint base_fert = main_aligned_source_words[i + 1].size();

  Storage1D<std::vector<uchar> > hyp_aligned_source_words = main_aligned_source_words;

  if (base_fert >= 1) {
    //consider main alignment, most expansions and all swaps

    double main_count = main_count_;

    //expansions
    for (std::map<uchar,std::map<uchar,double> >::const_iterator it = exp_count_.begin(); it != exp_count_.end(); it++) {

      uint j = it->first;
      uint cur_aj = main_alignment[j];

      vec_erase<uchar>(hyp_aligned_source_words[cur_aj], j);

      const std::map<uchar,double>& inner_exp = it->second;
      for (std::map<uchar,double>::const_iterator inner_it = inner_exp.begin(); inner_it != inner_exp.end(); inner_it++) {
        uint new_aj = inner_it->first;
        double count = inner_it->second;

        bool not_null = (cur_aj != 0 && new_aj != 0);

        if (not_null && ((cur_aj - 1 > i && new_aj - 1 > i) || (cur_aj - 1 < i && new_aj - 1 < i)))
          main_count += count;
        else if (base_fert > 1 || cur_aj != i + 1) {

          hyp_aligned_source_words[new_aj].push_back(j);
          //NOTE: sorting is only needed if new_aj == i+1
          if (new_aj == i + 1)
            vec_sort(hyp_aligned_source_words[new_aj]);

          //call routine to add counts for hyp_aligned_source_words
          add_nondef_count_compact(hyp_aligned_source_words, i, c, J, maxJ, count, temp_count, par_count);

          //restore
          hyp_aligned_source_words[new_aj] = main_aligned_source_words[new_aj];
        }
      }
      hyp_aligned_source_words[cur_aj] = main_aligned_source_words[cur_aj];
    }

    //swaps
    for (std::map<uchar,std::map<uchar,double> >::const_iterator it = swap_count_.begin(); it != swap_count_.end(); it++) {

      uint j1 = it->first;
      uint cur_aj1 = main_alignment[j1];

      const std::map<uchar,double>& inner_swap = it->second;
      for (std::map<uchar,double>::const_iterator inner_it = inner_swap.begin(); inner_it != inner_swap.end(); inner_it++) {
        uint j2 = inner_it->first;
        uint cur_aj2 = main_alignment[j2];
        double count = inner_it->second;

        bool not_null = (cur_aj1 != 0 && cur_aj2 != 0);
        if (not_null && ((cur_aj1 - 1 > i && cur_aj2 - 1 > i) || (cur_aj1 - 1 < i && cur_aj2 - 1 < i))) {
          main_count += count;
        }
        else {

          vec_replace<uchar>(hyp_aligned_source_words[cur_aj1], j1, j2);
          if (cur_aj1 == i + 1)
            vec_sort(hyp_aligned_source_words[cur_aj1]);

          vec_replace<uchar>(hyp_aligned_source_words[cur_aj2], j2, j1);
          if (cur_aj2 == i + 1)
            vec_sort(hyp_aligned_source_words[cur_aj2]);

          //call routine to add counts for hyp_aligned_source_words
          add_nondef_count_compact(hyp_aligned_source_words, i, c, J, maxJ, count, temp_count, par_count);

          //restore
          hyp_aligned_source_words[cur_aj1] = main_aligned_source_words[cur_aj1];
          hyp_aligned_source_words[cur_aj2] = main_aligned_source_words[cur_aj2];
        }
      }
    }

    //now handle main alignment
    add_nondef_count_compact(main_aligned_source_words, i, c, J, maxJ, main_count, temp_count, par_count);
  }
  else {
    //consider only expansions to i+1 here

    //expansions
    for (std::map<uchar,std::map<uchar,double> >::const_iterator it = exp_count_.begin(); it != exp_count_.end(); it++) {

      uint j = it->first;
      uint cur_aj = main_alignment[j];
      assert(cur_aj != i + 1);

      const std::map<uchar,double>& inner_exp = it->second;

      std::map<uchar,double>::const_iterator inner_it = inner_exp.find(i + 1);
      if (inner_it != inner_exp.end()) {

        double count = inner_it->second;

        assert(hyp_aligned_source_words[i + 1].empty());

        vec_erase(hyp_aligned_source_words[cur_aj], (uchar) j);
        hyp_aligned_source_words[i + 1].push_back(j);   //no need to sort, the array was empty

        //call routine to add counts for hyp_aligned_source_words
        add_nondef_count_compact(hyp_aligned_source_words, i, c, J, maxJ, count, temp_count, par_count);

        //restore
        hyp_aligned_source_words[cur_aj] = main_aligned_source_words[cur_aj];
        hyp_aligned_source_words[i + 1].clear();
      }
    }
  }

  for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = temp_count.begin(); it != temp_count.end(); it++)
    nondef_count[it->first] += it->second;
}


//diffpar
void CountStructure::unpack_compact(uint offset, uint J, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
                                    const Storage1D<WordClassType>& tclass, Storage1D<std::map<Math1D::Vector<ushort, uchar>, double> >& nondef_count,
                                    Math3D::Tensor<double>& par_count) const
{
  Math1D::Vector<uchar> main_alignment(J, 255);
  for (uint ii = 0; ii < main_aligned_source_words.size(); ii++) {
    for (uint k = 0; k < main_aligned_source_words[ii].size(); k++) {
      uint j = main_aligned_source_words[ii][k];
      main_alignment[j] = ii;
    }
  }

  //it is faster to accumulate an intermediate map first
  Storage1D<std::map<Math1D::Vector<ushort,uchar>,double> > temp_count(nondef_count.size());

  add_nondef_count_compact_diffpar(main_aligned_source_words, tclass, J, offset, main_count_, temp_count, par_count);

  Storage1D<std::vector<uchar> > hyp_aligned_source_words = main_aligned_source_words;

  //handle expansions
  for (std::map<uchar,std::map<uchar,double> >::const_iterator it = exp_count_.begin(); it != exp_count_.end(); it++) {

    uint j = it->first;
    uint cur_aj = main_alignment[j];

    vec_erase(hyp_aligned_source_words[cur_aj], (uchar) j);

    const std::map<uchar,double>& inner_exp = it->second;
    for (std::map<uchar,double>::const_iterator inner_it = inner_exp.begin(); inner_it != inner_exp.end(); inner_it++) {
      uint new_aj = inner_it->first;
      double count = inner_it->second;

      hyp_aligned_source_words[new_aj].push_back(j);
      vec_sort(hyp_aligned_source_words[new_aj]);

      //call routine to add counts for hyp_aligned_source_words
      add_nondef_count_compact_diffpar(hyp_aligned_source_words, tclass, J, offset, count, temp_count, par_count);
      hyp_aligned_source_words[new_aj] = main_aligned_source_words[new_aj];
    }

    hyp_aligned_source_words[cur_aj] = main_aligned_source_words[cur_aj];
  }

  //handle swaps
  for (std::map<uchar,std::map<uchar,double> >::const_iterator it = swap_count_.begin(); it != swap_count_.end(); it++) {

    uint j1 = it->first;
    uint cur_aj1 = main_alignment[j1];

    const std::map<uchar,double>& inner_swap = it->second;
    for (std::map<uchar,double>::const_iterator inner_it = inner_swap.begin(); inner_it != inner_swap.end(); inner_it++) {
      uint j2 = inner_it->first;
      uint cur_aj2 = main_alignment[j2];
      double count = inner_it->second;

      vec_replace(hyp_aligned_source_words[cur_aj1], (uchar) j1, (uchar) j2);
      vec_sort(hyp_aligned_source_words[cur_aj1]);

      vec_replace(hyp_aligned_source_words[cur_aj2], (uchar) j2, (uchar) j1);
      vec_sort(hyp_aligned_source_words[cur_aj2]);

      //call routine to add counts for hyp_aligned_source_words
      add_nondef_count_compact_diffpar(hyp_aligned_source_words, tclass, J, offset, count, temp_count, par_count);

      //restore
      hyp_aligned_source_words[cur_aj1] = main_aligned_source_words[cur_aj1];
      hyp_aligned_source_words[cur_aj2] = main_aligned_source_words[cur_aj2];
    }
  }

  for (uint c=0; c < temp_count.size(); c++)
    for (std::map<Math1D::Vector<ushort,uchar>, double>::const_iterator it = temp_count[c].begin(); it != temp_count[c].end(); it++)
      nondef_count[c][it->first] += it->second;
}

class CompactAlignedSourceWords {
public:

  CompactAlignedSourceWords(const Storage1D<std::vector<uchar> >& aligned_source_words, const Storage1D<WordClassType>& tclass)
    : tclass_(tclass)
  {
    start_.resize(aligned_source_words.size() - 1);

    //we don't copy the empty word as we do not need it

    uint nPos = 0;
    for (uint i = 1; i < aligned_source_words.size(); i++)
      nPos += aligned_source_words[i].size();

    pos_.resize(nPos);
    uint next_pos = 0;

    for (uint i = 1; i < aligned_source_words.size(); i++) {
      start_[i - 1] = next_pos;
      for (uint k = 0; k < aligned_source_words[i].size(); k++) {
        pos_[next_pos] = aligned_source_words[i][k];
        next_pos++;
      }
    }
    assert(next_pos == nPos);
  }

  void get_noncompact_form(Storage1D<std::vector<uchar> >& aligned_source_words, uint J) const
  {
    aligned_source_words.resize(size() + 1);
    aligned_source_words[0].clear();

    Storage1D<bool> covered(J, false);
    for (uint i = 0; i < size(); i++) {
      aligned_source_words[i + 1].clear();
      for (uint k = start(i); k < end(i); k++) {
        uint j = pos_[k];
        aligned_source_words[i + 1].push_back(j);
        covered[j] = true;
      }
    }
    for (uint j = 0; j < J; j++) {
      if (!covered[j])
        aligned_source_words[0].push_back(j);
    }
  }

  uchar start(uchar i) const
  {
    return start_[i];
  }

  uchar end(uchar i) const
  {
    if (i + 1 < start_.size())
      return start_[i + 1];
    else
      return pos_.size();
  }

  uchar size() const
  {
    return start_.size();
  }

  Math1D::Vector<uchar,uchar> pos_;
  Math1D::Vector<uchar,uchar> start_;
  const Storage1D<WordClassType> tclass_;
};

//this operator only exists in order to make the class usable in std::map.
//Hence, it is optimized for speed, not for a sensible order
bool operator<(const CompactAlignedSourceWords& v1, const CompactAlignedSourceWords& v2)
{
  if (v1.tclass_.size() != v2.tclass_.size())
    return (v1.tclass_.size() < v2.tclass_.size());

  if (v1.pos_.size() != v2.pos_.size())
    return (v1.pos_.size() < v2.pos_.size());

  if (v1.start_.size() != v2.start_.size())
    return (v1.start_.size() < v2.start_.size());

  for (uint k = 0; k < v1.tclass_.size(); k++) {
    if (v1.tclass_[k] != v2.tclass_[k])
      return (v1.tclass_[k] < v2.tclass_[k]);
  }

  for (uchar k = 0; k < v1.pos_.size(); k++) {
    if (v1.pos_[k] != v2.pos_[k])
      return (v1.pos_[k] < v2.pos_[k]);
  }

  for (uchar k = 0; k < v1.start_.size(); k++) {
    if (v1.start_[k] != v2.start_[k])
      return (v1.start_[k] < v2.start_[k]);
  }

  return false;
}

void IBM3Trainer::update_distortion_probs(const ReducedIBM3ClassDistortionModel& fdistort_count, ReducedIBM3ClassDistortionModel& fnondef_distort_count,
    Storage1D<std::map<CompactAlignedSourceWords,CountStructure> >& refined_nondef_aligned_words_count)
{
  //std::cerr << "update_distortion_probs" << std::endl;
  const uint nClasses = distortion_param_.zDim();

  if (par_mode_ != IBM23Nonpar) {

    Math3D::Tensor<double> fpar_singleton_count(distortion_param_.xDim(), distortion_param_.yDim(), nClasses, 0.0);
    Math3D::Tensor<double> fpar_span_count(distortion_param_.xDim(), distortion_param_.yDim(), nClasses, 0.0);

    if (par_mode_ == IBM23ParByPosition) {

      for (uint J = 0; J < fdistort_count.size(); J++) {

        //std::cerr << "J: " << J << std::endl;

        for (uint c = 0; c < nClasses; c++) {
          for (uint j = 0; j < fdistort_count[J].xDim(); j++) {
            for (uint i = 0; i < fdistort_count[J].yDim(); i++) {
              const double count = fdistort_count[J](j, i, c);
              fpar_singleton_count(j, i, c) += count;
              fpar_span_count(J, i, c) += count;
            }
          }
        }
      }
    }
    else {

      assert(par_mode_ == IBM23ParByDifference);

      const uint zero_offset = maxI_ - 1;
      fpar_span_count.resize(maxI_, distortion_param_.xDim(), nClasses, 0.0);

      for (uint J = 0; J < fdistort_count.size(); J++) {

        //std::cerr << "J: " << J << std::endl;

        for (uint c = 0; c < nClasses; c++) {
          for (uint j = 0; j < fdistort_count[J].xDim(); j++) {
            for (uint i = 0; i < fdistort_count[J].yDim(); i++) {
              const double count = fdistort_count[J](j, i, c);
              fpar_singleton_count(zero_offset + j - i, 0, c) += count;
              fpar_span_count(zero_offset - i, zero_offset + fdistort_count[J].xDim() - 1 - i, c) += count;
            }
          }
        }
      }
    }

    if (nondeficient_) {

      Math3D::Tensor<double> fnondef_par_distort_count(distortion_param_.xDim(), distortion_param_.yDim(), nClasses, 0.0);
      Math3D::Tensor<double> hyp_distort_param(distortion_param_.xDim(), distortion_param_.yDim(), nClasses, 0.0);

      for (uint J = 0; J < fdistort_count.size(); J++) {

        for (uint c = 0; c < nClasses; c++) {
          for (uint j = 0; j < fdistort_count[J].xDim(); j++) {
            for (uint i = 0; i < fdistort_count[J].yDim(); i++) {
              if (par_mode_ == IBM23ParByPosition)
                fnondef_par_distort_count(j, i, c) += fnondef_distort_count[J](j, i, c);
              else
                fnondef_par_distort_count(maxI_ - 1 + j - i, 0, c) += fnondef_distort_count[J](j, i, c);
            }
          }
        }
      }

      //NOTE: fnondef_par_distort_count is still modified when calling unpack_compact below (removal of singleton terms)

      const uint xDim = fpar_singleton_count.xDim();

      for (uint c = 0; c < nClasses; c++) {
        for (uint i = 0; i < fpar_singleton_count.yDim(); i++) {

          const double sum = fpar_singleton_count.sum_x(i, c);

          if (sum > 1e-305) {
            for (uint j = 0; j < xDim; j++)
              hyp_distort_param(j, i, c) = std::max(fert_min_param_entry, fpar_singleton_count(j, i, c) / sum);
          }
          else {
            for (uint j = 0; j < xDim; j++)
              hyp_distort_param(j, i, c) = distortion_param_(j, i, c);
          }
        }
      }

      if (par_mode_ == IBM23ParByDifference) {

        Storage1D<std::map<Math1D::Vector<ushort,uchar>,double> > cur_nondef_count(nClasses);

        for (uint J = 1; J <= refined_nondef_aligned_words_count.size(); J++) {

          //std::cerr << "J: " << J << std::endl;

          //unpack the compactly stored counts

          for (std::map<CompactAlignedSourceWords,CountStructure>::const_iterator it = refined_nondef_aligned_words_count[J - 1].begin();
               it != refined_nondef_aligned_words_count[J - 1].end(); it++) {

            const CompactAlignedSourceWords& main_compact_aligned_source_words = it->first;

            Storage1D<std::vector<uchar> > main_aligned_source_words;
            main_compact_aligned_source_words.get_noncompact_form(main_aligned_source_words, J);
            const Storage1D<WordClassType>& tclass = main_compact_aligned_source_words.tclass_;

            const CountStructure& cur_count_struct = it->second;
            cur_count_struct.unpack_compact(maxI_ - 1, J, main_aligned_source_words, tclass, cur_nondef_count, fnondef_par_distort_count);
          }
        }

        Storage1D<std::vector<Math1D::Vector<ushort,uchar> > > open_pos(nClasses);
        Storage1D<std::vector<double> > sum_pos_count(nClasses);
        for (uint c=0; c < nClasses; c++) {
          assign(open_pos[c], sum_pos_count[c], cur_nondef_count[c]);
          cur_nondef_count[c].clear();
        }

        //std::cerr << "copied to vectors" << std::endl;

        assert(distortion_param_.min() > 0.0);
        double start_energy = nondeficient_diffpar_m_step_energy(fnondef_par_distort_count, open_pos, sum_pos_count, distortion_param_);

        std::cerr << "old energy: " << start_energy << std::endl;

        assert(hyp_distort_param.min() > 0.0);
        double hyp_energy = nondeficient_diffpar_m_step_energy(fnondef_par_distort_count, open_pos, sum_pos_count, hyp_distort_param);

        std::cerr << "energy for normalized full counts: " << hyp_energy << std::endl;

        if (hyp_energy < start_energy) {

          start_energy = hyp_energy;
          distortion_param_ = hyp_distort_param;
        }

        nondeficient_diffpar_m_step(fnondef_par_distort_count, open_pos, sum_pos_count, start_energy);
      }
      else {

        for (uint i = 0; i < maxI_; i++) {

          //std::cerr << "i: " << i << std::endl;

          Storage1D<std::map<Math1D::Vector<uchar,uchar>,double> > cur_nondef_count(nClasses);

          for (uint J = 1; J <= refined_nondef_aligned_words_count.size(); J++) {

            //unpack the compactly stored counts

            for (std::map<CompactAlignedSourceWords,CountStructure>::const_iterator it = refined_nondef_aligned_words_count[J - 1].begin();
                 it != refined_nondef_aligned_words_count[J - 1].end(); it++) {

              const CompactAlignedSourceWords& main_compact_aligned_source_words = it->first;

              if (main_compact_aligned_source_words.size() <= i)
                continue;

              Storage1D<std::vector<uchar> > main_aligned_source_words;
              main_compact_aligned_source_words.get_noncompact_form(main_aligned_source_words, J);
              const Storage1D<WordClassType>& tclass = main_compact_aligned_source_words.tclass_;

              const CountStructure& cur_count_struct = it->second;
              const uint c = main_compact_aligned_source_words.tclass_[i];
              cur_count_struct.unpack_compact(i, c, J, maxJ_, main_aligned_source_words, cur_nondef_count[c], fnondef_par_distort_count);
            }
          }

          for (uint c = 0; c < hyp_distort_param.zDim(); c++) {

            if (cur_nondef_count[c].empty())
              continue;

            //copy map->vector for faster access
            std::vector<Math1D::Vector<uchar,uchar> > open_pos;
            std::vector<double> vec_nondef_count;
            assign(open_pos, vec_nondef_count, cur_nondef_count[c]);

            cur_nondef_count[c].clear();

            std::cerr << "calling  m-step for i=" << i << std::endl;

            assert(distortion_param_.min() > 0.0);
            double start_energy = nondeficient_m_step_energy(fnondef_par_distort_count, open_pos, vec_nondef_count, distortion_param_, i, c);

            assert(hyp_distort_param.min() > 0.0);
            double hyp_energy = nondeficient_m_step_energy(fnondef_par_distort_count, open_pos, vec_nondef_count, hyp_distort_param, i, c);

            assert(!isnan(start_energy));
            assert(!isnan(hyp_energy));

            if (hyp_energy < start_energy) {

              for (uint x = 0; x < hyp_distort_param.xDim(); x++) {
                distortion_param_(x, i, c) = hyp_distort_param(x, i, c);
              }

              std::cerr << "switching to normalized counts: " << start_energy << " -> " << hyp_energy << std::endl;

              start_energy = hyp_energy;
            }
            //nondeficient_m_step_with_interpolation(fnondef_par_distort_count,open_pos,vec_nondef_count,i,c,start_energy);
            nondeficient_m_step(fnondef_par_distort_count, open_pos, vec_nondef_count, i, c, start_energy);
          }
        }
      }
    }
    else {
      //deficient

      if (par_mode_ == IBM23ParByPosition) {

        for (uint i = 0; i < maxI_; i++) {
          std::cerr << "m-step for i=" << i << std::endl;

          for (uint c = 0; c < nClasses; c++) {

            if (msolve_mode_ == MSSolvePGD)
              par_distortion_m_step(fpar_singleton_count, fpar_span_count, i, c);
            else
              par_distortion_m_step_unconstrained(fpar_singleton_count,fpar_span_count, i, c);
          }
        }
      }
      else {

        assert(par_mode_ == IBM23ParByDifference);

        for (uint c = 0; c < nClasses; c++)
          diffpar_distortion_m_step(fpar_singleton_count, fpar_span_count, c);
      }

      par2nonpar_distortion(distortion_prob_);
    }
  }
  else {
    //nonparametric distortion model

    if (nondeficient_) {

      ReducedIBM3ClassDistortionModel hyp_distort_prob(MAKENAME(hyp_distort_prob));
      hyp_distort_prob = fdistort_count;
      for (uint J = 0; J < fdistort_count.size(); J++) {

        for (uint c = 0; c < nClasses; c++) {
          for (uint i = 0; i < hyp_distort_prob[J].yDim(); i++) {

            const double sum = fdistort_count[J].sum_x(i, c);

            if (sum > 1e-305) {
              for (uint j = 0; j < hyp_distort_prob[J].xDim(); j++)
                hyp_distort_prob[J](j, i, c) = std::max(1e-15, hyp_distort_prob[J](j, i, c) / sum);
            }
            else {
              for (uint j = 0; j < hyp_distort_prob[J].xDim(); j++)
                hyp_distort_prob[J](j, i, c) = distortion_prob_[J](j, i, c);
            }
          }
        }
      }

      for (uint i = 0; i < maxI_; i++) {

        std::cerr << "calling nondeficient m-steps for i=" << i << std::endl;

        for (uint J = 1; J <= refined_nondef_aligned_words_count.size(); J++) {

          //unpack the compactly stored counts

          Storage1D<std::map<Math1D::Vector<uchar,uchar>,double> > cur_nondef_count(nClasses);

          for (std::map<CompactAlignedSourceWords,CountStructure>::const_iterator it = refined_nondef_aligned_words_count[J - 1].begin();
               it != refined_nondef_aligned_words_count[J - 1].end(); it++) {

            const CompactAlignedSourceWords& main_compact_aligned_source_words = it->first;

            if (main_compact_aligned_source_words.size() <= i)
              continue;

            const CountStructure& cur_count_struct = it->second;

            Storage1D<std::vector<uchar> > main_aligned_source_words;
            main_compact_aligned_source_words.get_noncompact_form(main_aligned_source_words, J);

            const uint c = main_compact_aligned_source_words.tclass_[i];

            cur_count_struct.unpack_compact(i, c, J, J, main_aligned_source_words, cur_nondef_count[c], fnondef_distort_count[J - 1]);
          }

          for (uint c = 0; c < nClasses; c++) {

            if (cur_nondef_count[c].empty())
              continue;

            //copy map->vector for faster access
            std::vector<Math1D::Vector<uchar,uchar> > open_pos;
            std::vector<double> vec_nondef_count;
            assign(open_pos, vec_nondef_count, cur_nondef_count[c]);

            cur_nondef_count[c].clear();

            double start_energy = nondeficient_m_step_energy(fnondef_distort_count[J - 1], open_pos, vec_nondef_count, distortion_prob_[J - 1], i, c);
            const double hyp_energy = nondeficient_m_step_energy(fnondef_distort_count[J - 1], open_pos, vec_nondef_count, hyp_distort_prob[J - 1], i, c);

            assert(!isnan(start_energy));
            assert(!isnan(hyp_energy));

            //std::cerr << "J: " << J << ", prev energy: " << start_energy << ", hyp energy: " << hyp_energy << std::endl;

            if (hyp_energy < start_energy) {

              //std::cerr << "switching to normalized counts: " << start_energy << " -> " << hyp_energy << std::endl;

              start_energy = hyp_energy;

              for (uint j = 0; j < distortion_prob_[J - 1].xDim(); j++)
                distortion_prob_[J - 1](j, i, c) = hyp_distort_prob[J - 1](j, i, c);
            }
            //nondeficient_m_step(vec_nondef_count,i,J,start_energy);
            //nondeficient_m_step(fnondef_distort_count[J-1], open_pos, vec_nondef_count, i, J, start_energy);
            nondeficient_m_step_with_interpolation(fnondef_distort_count[J - 1], open_pos, vec_nondef_count, i, c, J, start_energy);
          }
        }
      }
    }
    else {
      //deficient mode

      for (uint J = 0; J < distortion_prob_.size(); J++) {

        if (distortion_prob_[J].size() == 0)
          continue;

        //       std::cerr << "J:" << J << std::endl;
        //       std::cerr << "distort_count: " << fdistort_count[J] << std::endl;
        for (uint c = 0; c < nClasses; c++) {
          for (uint i = 0; i < distortion_prob_[J].yDim(); i++) {

            const double sum = fdistort_count[J].sum_x(i, c);

            if (sum > 1e-307) {
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));

              for (uint j = 0; j < J + 1; j++) {
                distortion_prob_[J](j, i, c) = std::max(fert_min_param_entry, inv_sum * fdistort_count[J](j, i, c));
                if (isnan(distortion_prob_[J](j, i, c))) {
                  std::cerr << "sum: " << sum << std::endl;
                  std::cerr << "set to " << inv_sum << " * " << fdistort_count[J](j, i, c) << " = "
                            << (inv_sum * fdistort_count[J](j, i, c)) << std::endl;
                }
                assert(!isnan(fdistort_count[J](j, i, c)));
                assert(!isnan(distortion_prob_[J](j, i, c)));
              }
            }
            else {
              std::cerr << "WARNING: did not update distortion count because sum was " << sum << std::endl;
            }
          }
        }
      }
    }
  }
}

void IBM3Trainer::train_em(uint nIter, FertilityModelTrainerBase* prev_model, const HmmWrapperWithClasses* passed_wrapper)
{
  std::cerr << "starting IBM-3 training without constraints" << std::endl;

  const uint nSentences = source_sentence_.size();

  double max_perplexity = 0.0;
  double approx_sum_perplexity = 0.0;
  double viterbi_max_perplexity = 0.0;

  double max_ratio = 1.0;
  double min_ratio = 1.0;

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment;
  if (viterbi_ilp_mode_ != IlpOff)
    viterbi_alignment.resize(source_sentence_.size());

  double dict_weight_sum = (prior_weight_active_) ? 1.0 : 0.0; //only used as a flag

  SingleLookupTable aux_lookup;

  if (par_mode_ != IBM23Nonpar)
    par2nonpar_distortion(distortion_prob_);

  ReducedIBM3ClassDistortionModel fdistort_count(distortion_prob_.size(), MAKENAME(fdistort_count));

  //used only in nondeficient mode. will only collect counts > min_nondef_count_
  ReducedIBM3ClassDistortionModel fnondef_distort_count(distortion_prob_.size(), MAKENAME(fnondef_distort_count));

  for (uint J = 0; J < distortion_prob_.size(); J++) {
    fdistort_count[J].resize_dirty(distortion_prob_[J].xDim(), distortion_prob_[J].yDim(), distortion_prob_[J].zDim());
    fnondef_distort_count[J].resize_dirty(distortion_prob_[J].xDim(), distortion_prob_[J].yDim(), distortion_prob_[J].zDim());
  }

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords, MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  double fzero_count;
  double fnonzero_count;

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    std::cerr << "******* IBM-3 EM-iteration #" << iter << std::endl;

    if (passed_wrapper != 0 && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    uint nViterbiBetter = 0;
    uint nViterbiWorse = 0;

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint J = 0; J < distortion_prob_.size(); J++) {
      fdistort_count[J].set_constant(0.0);
      fnondef_distort_count[J].set_constant(0.0);
    }
    //fnondef_distort_count.set_constant(0.0);

    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    //DEBUG
    // std::cerr << "WARNING: setting uniform fertility probabilities" << std::endl;
    // for (uint i=1; i < nTargetWords; i++) {
    //   fertility_prob_[i].set_constant(1.0 / std::max<uint>(fertility_prob_[i].size(),fertility_limit_[i]+1));
    // }
    //END_DEBUG

    Storage1D<std::map<CompactAlignedSourceWords,CountStructure> > refined_nondef_aligned_words_count(maxJ_);

    max_perplexity = 0.0;
    approx_sum_perplexity = 0.0;

    std::clock_t tStartLoop = std::clock();

    for (size_t s = 0; s < source_sentence_.size(); s++) {

      // if (s > 25)
      //        exit(0);

      if ((s % 10000) == 0)
        //if ((s% 100) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      Math3D::Tensor<double>& cur_distort_count = fdistort_count[curJ - 1];
      Math3D::Tensor<double>& cur_nondef_distort_count = fnondef_distort_count[curJ - 1];

      Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      long double best_prob;

      if (prev_model != 0) {

        best_prob = prev_model->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, cur_alignment);
      }
      else {

        best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, cur_alignment);

        assert(!isnan(best_prob));

        //std::cerr << "back from hillclimbing" << std::endl;

        if (viterbi_ilp_mode_ != IlpOff && !nondeficient_) {

          viterbi_alignment[s] = cur_alignment;
          if (curJ == 1)
            viterbi_max_perplexity -= logl(best_prob);  //for curJ==1 hillclimbing finds the optimum
          else {

            double viterbi_energy = compute_viterbi_alignment_ilp(cur_source, cur_target, cur_lookup, viterbi_alignment[s]);

            viterbi_max_perplexity += viterbi_energy;

            bool alignments_equal = (cur_alignment == viterbi_alignment[s]);

            if (!alignments_equal) {

              long double viterbi_prob = expl(-viterbi_energy);

              double ratio = viterbi_prob / best_prob;

              if (ratio > 1.01) {
                nViterbiBetter++;

                // std::cerr << "pair #" << s << std::endl;
                // std::cerr << "ilp prob:          " << viterbi_prob << std::endl;
                // std::cerr << "ilp alignment: " << viterbi_alignment[s] << std::endl;

                // std::cerr << "hillclimbing prob: " << best_prob << std::endl;
                // std::cerr << "hc. alignment: " << cur_alignment << std::endl;
              }
              else if (ratio < 0.99) {
                nViterbiWorse++;

                std::cerr << "pair #" << s << ": WORSE!!!!" << std::endl;
                std::cerr << "ilp prob:          " << viterbi_prob << std::endl;
                std::cerr << "ilp alignment: " << viterbi_alignment[s] << std::endl;

                std::cerr << "hillclimbing prob: " << best_prob << std::endl;
                std::cerr << "hc. alignment: " << cur_alignment << std::endl;
              }
              if (ratio > max_ratio) {
                max_ratio = ratio;

                std::cerr << "pair #" << s << std::endl;
                std::cerr << "ilp prob:          " << viterbi_prob << std::endl;
                std::cerr << "ilp alignment: " << viterbi_alignment[s] << std::endl;

                std::cerr << "hillclimbing prob: " << best_prob << std::endl;
                std::cerr << "hc. alignment: " << cur_alignment << std::endl;
              }
              if (ratio < min_ratio) {
                min_ratio = ratio;

                std::cerr << "pair #" << s << std::endl;
                std::cerr << "ilp prob:          " << viterbi_prob << std::endl;
                std::cerr << "ilp alignment: " << viterbi_alignment[s] << std::endl;

                std::cerr << "hillclimbing prob: " << best_prob << std::endl;
                std::cerr << "hc. alignment: " << cur_alignment << std::endl;
              }
            }

            if (viterbi_ilp_mode_ == IlpCenter) {
              cur_alignment = viterbi_alignment[s];
              best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                          expansion_move_prob, swap_move_prob, cur_alignment);
            }
          }
        }
      }

      max_perplexity -= std::log(best_prob);

      const long double expansion_prob = expansion_move_prob.sum();
      const long double swap_prob = swap_mass(swap_move_prob);

      const long double sentence_prob = best_prob + expansion_prob + swap_prob;

      approx_sum_perplexity -= std::log(sentence_prob);

      Math2D::Matrix<double> j_marg;
      Math2D::Matrix<double> i_marg;

      FertilityModelTrainer::compute_approximate_jmarginals(cur_alignment, expansion_move_prob, swap_move_prob, sentence_prob, j_marg);
      compute_approximate_imarginals(cur_alignment, fertility, expansion_move_prob, sentence_prob, i_marg);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      if (isnan(inv_sentence_prob)) {

        std::cerr << "best prob: " << best_prob << std::endl;
        std::cerr << "swap prob: " << swap_prob << std::endl;
        std::cerr << "expansion prob: " << expansion_prob << std::endl;
        exit(1);
      }

      assert(!isnan(inv_sentence_prob));

      //std::cerr << "updating counts " << std::endl;

      for (uint j = 0; j <= curJ / 2; j++) {

        fzero_count += j * i_marg(j, 0);
        fnonzero_count += (curJ - 2 * j) * i_marg(j, 0);
      }

      //increase counts for dictionary and distortion

      double nondef_relevant_sum = 0.0;
      if (nondeficient_) {
        nondef_relevant_sum = best_prob;

        for (uint j = 0; j < curJ; j++) {
          for (uint aj = 0; aj <= curI; aj++) {

            const double contrib = expansion_move_prob(j, aj) * inv_sentence_prob;

            if (contrib > min_nondef_count_) {
              nondef_relevant_sum += expansion_move_prob(j, aj);
            }
          }
        }

        for (uint j1 = 0; j1 < curJ - 1; j1++) {
          for (uint j2 = j1 + 1; j2 < curJ; j2++) {

            const double contrib = swap_move_prob(j1, j2) * inv_sentence_prob;

            if (contrib > min_nondef_count_) {
              nondef_relevant_sum += swap_move_prob(j1, j2);
            }
          }
        }
      }

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        for (uint i = 0; i <= curI; i++) {

          const double marg = j_marg(i, j);

          if (i == 0)
            fwcount[0][s_idx - 1] += marg;
          else {
            const uint ti = cur_target[i - 1];
            fwcount[ti][cur_lookup(j, i - 1)] += marg;
            cur_distort_count(j, i - 1, target_class_[ti]) += marg;
          }
        }
        if (nondeficient_) {

          const uint cur_aj = cur_alignment[j];

          if (cur_aj != 0) {

            long double nondef_addon = nondef_relevant_sum;
            for (uint i = 0; i <= curI; i++) {
              if (expansion_move_prob(j, i) * inv_sentence_prob > min_nondef_count_)
                nondef_addon -= expansion_move_prob(j, i);
            }
            for (uint jj = 0; jj < curJ; jj++) {
              if (swap_move_prob(j, jj) * inv_sentence_prob > min_nondef_count_)
                nondef_addon -= swap_move_prob(j, jj);
            }

            //for numerical stability:
            nondef_addon = std::max<long double>(0.0, nondef_addon);

            cur_nondef_distort_count(j, cur_aj - 1, target_class_[cur_target[cur_aj - 1]]) += nondef_addon * inv_sentence_prob;
          }

          for (uint i = 0; i <= curI; i++) {

            if (i != cur_aj) {

              if (i != 0) {

                long double nondef_addon = 0.0;

                if (expansion_move_prob(j, i) * inv_sentence_prob > min_nondef_count_)
                  nondef_addon += expansion_move_prob(j, i);

                for (uint jj = 0; jj < curJ; jj++) {
                  if (cur_alignment[jj] == i) {
                    if (swap_move_prob(j, jj) * inv_sentence_prob > min_nondef_count_)
                      nondef_addon += swap_move_prob(j, jj);
                  }
                }
                cur_nondef_distort_count(j, i - 1, target_class_[cur_target[i - 1]]) += nondef_addon * inv_sentence_prob;
              }
            }
          }
        }
      }

      if (nondeficient_) {

        //a) best known alignment (=mode)
        Storage1D<std::vector<uchar> > aligned_source_words(curI + 1);
        for (uint j = 0; j < curJ; j++)
          aligned_source_words[cur_alignment[j]].push_back(j);

        Storage1D<WordClassType> tclass(curI);
        for (uint i = 0; i < curI; i++)
          tclass[i] = target_class_[cur_target[i]];

        CountStructure& cur_refined_count = refined_nondef_aligned_words_count[curJ - 1][CompactAlignedSourceWords(aligned_source_words,tclass)];

        const double mode_contrib = best_prob / sentence_prob;
        cur_refined_count.main_count_ += mode_contrib;

        Storage1D<std::vector<uchar> > hyp_aligned_source_words = aligned_source_words;

        //b) expansions
        for (uint jj = 0; jj < curJ; jj++) {

          const uint cur_aj = cur_alignment[jj];

          std::vector<uchar>& cur_hyp_aligned_source_words = hyp_aligned_source_words[cur_aj];

          vec_erase(cur_hyp_aligned_source_words, (uchar) jj);

          for (uint aj = 0; aj <= curI; aj++) {

            const double contrib = expansion_move_prob(jj, aj) * inv_sentence_prob;

            if (contrib > min_nondef_count_) {

              assert(aj != cur_aj);

              cur_hyp_aligned_source_words.push_back(jj);
              vec_sort(cur_hyp_aligned_source_words);

              cur_refined_count.exp_count_[jj][aj] += contrib;

              cur_hyp_aligned_source_words = aligned_source_words[aj];
            }
          }

          hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
        }

        //c) swaps
        for (uint j1 = 0; j1 < curJ - 1; j1++) {

          const uint aj1 = cur_alignment[j1];

          for (uint j2 = j1 + 1; j2 < curJ; j2++) {

            const double contrib = swap_move_prob(j1, j2) * inv_sentence_prob;

            if (contrib > min_nondef_count_) {

              const uint aj2 = cur_alignment[j2];

              vec_replace<uchar>(hyp_aligned_source_words[aj1], j1, j2);
              vec_sort(hyp_aligned_source_words[aj1]);
              vec_replace<uchar>(hyp_aligned_source_words[aj2], j2, j1);
              vec_sort(hyp_aligned_source_words[aj2]);

              cur_refined_count.swap_count_[j1][j2] += contrib;

              hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
              hyp_aligned_source_words[aj2] = aligned_source_words[aj2];
            }
          }
        }
      }

      //update fertility counts
      for (uint i = 1; i <= curI; i++) {

        const uint t_idx = cur_target[i - 1];

        Math1D::Vector<double>& cur_fert_count = ffert_count[t_idx];

        for (uint c = 0; c <= std::min<ushort>(curJ, fertility_limit_[t_idx]); c++)
          cur_fert_count[c] += i_marg(c, i);
      }

      assert(!isnan(fzero_count));
      assert(!isnan(fnonzero_count));
    }  //loop over sentences finished

    std::clock_t tEndLoop = std::clock();

    std::cerr << "loop over sentences took " << diff_seconds(tEndLoop, tStartLoop) << " seconds." << std::endl;

    const double reg_term = regularity_term();  //we need the reg-term BEFORE the parameter update

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
      p_nonzero_ = fnonzero_count / fsum;

      std::cerr << "new p_zero: " << p_zero_ << std::endl;
    }
    //DEBUG
    uint nZeroAlignments = 0;
    uint nAlignments = 0;
    for (size_t s = 0; s < source_sentence_.size(); s++) {

      nAlignments += source_sentence_[s].size();

      for (uint j = 0; j < source_sentence_[s].size(); j++) {
        if (best_known_alignment_[s][j] == 0)
          nZeroAlignments++;
      }
    }
    std::cerr << "percentage of zero-aligned words: "
              << (((double)nZeroAlignments) / ((double)nAlignments)) << std::endl;
    //END_DEBUG

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, nSentences, dict_weight_sum, smoothed_l0_, l0_beta_, dict_m_step_iter_, dict_,
                            fert_min_dict_entry, msolve_mode_ != MSSolvePGD);

    //update distortion prob from counts
    update_distortion_probs(fdistort_count, fnondef_distort_count, refined_nondef_aligned_words_count);

    update_fertility_prob(ffert_count, fert_min_param_entry);

    max_perplexity /= source_sentence_.size();
    approx_sum_perplexity /= source_sentence_.size();
    viterbi_max_perplexity /= source_sentence_.size();

    max_perplexity += reg_term;
    approx_sum_perplexity += reg_term;
    viterbi_max_perplexity += reg_term;

    std::string transfer = (prev_model != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "IBM-3 max-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << max_perplexity << std::endl;
    std::cerr << "IBM-3 approx-sum-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << approx_sum_perplexity << std::endl;

    if (viterbi_ilp_mode_ != IlpOff && !nondeficient_ && prev_model == 0) {

      if (viterbi_ilp_mode_ == IlpComputeOnly) {
        std::cerr << "IBM-3 viterbi max-perplex-energy in between iterations #" << (iter - 1) << " and "
                  << iter << transfer << ": " << viterbi_max_perplexity << std::endl;
      }

      std::cerr << "Viterbi-ILP better in " << nViterbiBetter << ", worse in " << nViterbiWorse << " cases." << std::endl;

      std::cerr << "max-ratio: " << max_ratio << std::endl;
      //std::cerr << "inv min-ratio: " << (1.0 / min_ratio) << std::endl;
    }

    if (possible_ref_alignments_.size() > 0) {

      std::cerr << "#### IBM3-AER in between iterations #" << (iter - 1) << " and " << iter
                << transfer << ": " << FertilityModelTrainerBase::AER() << std::endl;
      std::cerr << "#### IBM3-fmeasure in between iterations #" << (iter - 1) << " and " << iter
                << transfer << ": " << FertilityModelTrainerBase::f_measure() << std::endl;
      std::cerr << "#### IBM3-DAE/S in between iterations #" << (iter - 1) << " and " << iter
                << transfer << ": " << FertilityModelTrainerBase::DAE_S() << std::endl;

      if (viterbi_ilp_mode_ != IlpOff) {
        std::cerr << "#### IBM3-AER for Viterbi in between iterations #" << (iter - 1)
                  << " and " << iter << transfer << ": " << FertilityModelTrainerBase::AER(viterbi_alignment) << std::endl;
        std::cerr << "#### IBM3-fmeasure for Viterbi in between iterations #" << (iter - 1) << " and " << iter
                  << transfer << ": " << FertilityModelTrainerBase::f_measure(viterbi_alignment) << std::endl;
        std::cerr << "#### IBM3-DAE/S for Viterbi in between iterations #" << (iter - 1) << " and " << iter << transfer << ": "
                  << FertilityModelTrainerBase::DAE_S(viterbi_alignment) << std::endl;
      }

      double postdec_aer;
      double postdec_fmeasure;
      double postdec_daes;
      PostdecEval(postdec_aer, postdec_fmeasure, postdec_daes, 0.25);
      std::cerr << "#### IBM3-Postdec-AER in between iterations #" << (iter - 1) << " and " << iter
                << transfer << ": " << postdec_aer << std::endl;
      std::cerr << "#### IBM3-Postdec-fmeasure in between iterations #" << (iter - 1) << " and " << iter
                << transfer << ": " << postdec_fmeasure << std::endl;
      std::cerr << "#### IBM3-Postdec-DAE/S in between iterations #" << (iter - 1) << " and " << iter
                << transfer << ": " << postdec_daes << std::endl;
    }

    std::cerr << (((double)sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" << std::endl;
  }

  if (par_mode_ == IBM23Nonpar) {

    //we compute this so that we can use it for computation of external alignments
    nonpar2par_distortion();
  }

  iter_offs_ = iter - 1;
}

void IBM3Trainer::train_viterbi(uint nIter, const AlignmentSetConstraints& align_constraints,
                                FertilityModelTrainerBase* prev_model, const HmmWrapperWithClasses* passed_wrapper)
{
  const uint nSentences = source_sentence_.size();

  std::cerr << "starting IBM-3 training without constraints" << std::endl;

  const uint max_mid_dev = align_constraints.itg_max_mid_dev_;
  const uint itg_ext_level = align_constraints.itg_extension_level_;
  const uint nMaxSkips = std::min(maxJ_-1,align_constraints.nMaxSkips_);
  const uint level3_maxlength = align_constraints.itg_level3_maxlength_;

  double max_perplexity = 0.0;

  ReducedIBM3ClassDistortionModel fdistort_count(distortion_prob_.size(), MAKENAME(fdistort_count));

  for (uint J = 0; J < distortion_prob_.size(); J++) {
    fdistort_count[J].resize_dirty(distortion_prob_[J].xDim(), distortion_prob_[J].yDim(), distortion_prob_[J].zDim());
  }

  SingleLookupTable aux_lookup;

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords, MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));
  NamedStorage1D<Math1D::Vector<double> > ffertclass_count(MAKENAME(ffert_count));

  SingleLookupTable aux;

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  if (fertprob_sharing_) {
    ffertclass_count.resize(nTFertClasses_);
    for (uint i = 1; i < nTargetWords; i++) {
      const uint c = tfert_class_[i];
      if (fertility_prob_[i].size() > ffertclass_count[c].size())
        ffertclass_count[c].resize_dirty(fertility_prob_[i].size());
    }
  }

  long double fzero_count;
  long double fnonzero_count;

  if (log_table_.size() < nSentences || xlogx_table_.size() < nSentences) {
    EXIT("passed log table is not large enough.");
  }

  if (align_constraints.align_set_ != AllAlignments) {
    if (nondeficient_) {
      nondeficient_ = false; //not supported with alignment constraints
    }

    if (align_constraints.align_set_ == IBMSkipAlignments || align_constraints.itg_extension_level_ >= 4) {

      uint maxJ = (align_constraints.align_set_ == IBMSkipAlignments) ? maxJ_ : 15;

      compute_uncovered_sets(maxJ,nMaxSkips);
      compute_coverage_states(maxJ);

      predecessor_sets_.resize(0);
      first_set_.resize(0);
    }
  }

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    std::cerr << "******* IBM-3 Viterbi-iteration #" << iter << std::endl;

    if (passed_wrapper != 0 && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint J = 0; J < fdistort_count.size(); J++) {
      fdistort_count[J].set_constant(0.0);
    }
    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0);
      ffert_count[i].set_constant(0.0);
    }

    max_perplexity = 0.0;

    std::clock_t tStartMainLoop = std::clock();

    for (size_t s = 0; s < nSentences; s++) {

      //if ((s% 1250) == 0)
      if ((s % 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_best_known_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      Math3D::Tensor<double>& cur_distort_count = fdistort_count[curJ - 1];

      Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

      //these will not actually be used, but need to be passed to the hillclimbing routine
      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      long double best_prob;

      if (prev_model != 0) {

        //we can skip calling hillclimbing here - we don't look at the neighbors anyway
        //but note that we have to deal with alignments that have 2*fertility[0] > curJ

        best_prob = prev_model->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, cur_best_known_alignment);

        make_alignment_feasible(cur_source, cur_target, cur_lookup, cur_best_known_alignment);

        for (uint j = 0; j < curJ; j++)
          fertility[cur_best_known_alignment[j]]++;

        max_perplexity -= logl(best_prob);

        //NOTE: to be 100% proper we should recalculate the prob of the alignment if it was made feasible
        //(would need to convert the alignment to internal mode first). But this only affects the energy printout at the end of
        // the iteration
      }
      else {

        if (align_constraints.align_set_ != AllAlignments) {

          long double prob = 0.0;

          if (align_constraints.align_set_ == IBMSkipAlignments) {

            prob = compute_ibmconstrained_viterbi_alignment_noemptyword(cur_source, cur_target, cur_lookup, cur_best_known_alignment);
            prob *= pow(p_nonzero_, source_sentence_[s].size());
            assert(alignment_satisfies_ibm_nonull(cur_best_known_alignment, nMaxSkips));
          }
          else if (align_constraints.align_set_ == ITGAlignments) {

            //std::cerr << "s: " << s << std::endl;

            prob = compute_itg_viterbi_alignment_noemptyword(cur_source, cur_target, cur_lookup, cur_best_known_alignment, itg_ext_level, max_mid_dev);
            prob *= pow(p_nonzero_, source_sentence_[s].size());
            assert(alignment_satisfies_itg_nonull(cur_best_known_alignment,curI,itg_ext_level,max_mid_dev,nMaxSkips,level3_maxlength));
          }

          //std::cerr << "probability " << prob << std::endl;
          //std::cerr << "generated alignment: " << cur_best_known_alignment << std::endl;

          max_perplexity -= logl(prob);

          long double check_prob = FertilityModelTrainer::alignment_prob(s, cur_best_known_alignment);
          double check_ratio = prob / check_prob;
          //std::cerr << "check prob: " << check_prob << std::endl;
          //std::cerr << "check_ratio: " << check_ratio << std::endl;
          assert(check_ratio > 0.999 && check_ratio < 1.001);

          //compute fertilities
          for (uint j=0; j < curJ; j++)
            fertility[cur_best_known_alignment[j]]++;
        }
        else {

          //std::cerr << "calling hillclimbing" << std::endl;

          best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                      expansion_move_prob, swap_move_prob, cur_best_known_alignment);

          //std::cerr << "back from hillclimbing" << std::endl;

          if (viterbi_ilp_mode_ != IlpOff && !nondeficient_ && curJ > 1) {        //for curJ == 1 hillclimbing finds the optimum

            //NOTE: we take the alignment for both modes (compute only and center)
            double energy = compute_viterbi_alignment_ilp(cur_source, cur_target, cur_lookup, best_known_alignment_[s]);
            max_perplexity += energy;

            fertility.set_constant(0);
            for (uint j = 0; j < curJ; j++)
              fertility[cur_best_known_alignment[j]]++;
          }
          else
            max_perplexity -= logl(best_prob);
        }
      }

      assert(2 * fertility[0] <= curJ);

      //std::cerr << "updating counts " << std::endl;

      fzero_count += fertility[0];
      fnonzero_count += curJ - 2 * fertility[0];

      //increase counts for dictionary and distortion
      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = cur_best_known_alignment[j];

        if (cur_aj != 0) {
          const uint t_idx = cur_target[cur_aj - 1];
          fwcount[t_idx][cur_lookup(j, cur_aj - 1)] += 1;
          cur_distort_count(j, cur_aj - 1, target_class_[t_idx]) += 1.0;

          assert(!isnan(cur_distort_count(j, cur_aj - 1, target_class_[t_idx])));
        }
        else {
          fwcount[0][s_idx - 1] += 1;
        }
      }

      //update fertility counts
      for (uint i = 1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i - 1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }

      //       std::cerr << "fzero_count: " << fzero_count << std::endl;
      //       std::cerr << "fnonzero_count: " << fnonzero_count << std::endl;

      assert(!isnan(fzero_count));
      assert(!isnan(fnonzero_count));
    } //loop over sentences finished

    std::clock_t tEndMainLoop = std::clock();

    std::cerr << " main loop took " << (diff_seconds(tEndMainLoop, tStartMainLoop) / 60.0) << " minutes." << std::endl;

    max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
    max_perplexity /= source_sentence_.size();

    std::cerr << (((double)sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" << std::endl;

    std::string transfer = (prev_model != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "IBM-3 energy in between iterations #" << (iter - 1) << " and "
              << iter << transfer << ": " << max_perplexity << std::endl;

    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
      p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);
    }

    update_fertility_prob(ffert_count, fert_min_param_entry, false);

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, nSentences, 0.0, false, 0.0, 0, dict_, fert_min_dict_entry);

    std::cerr << "new p0: " << p_zero_ << std::endl;

    /*** ICM stage ***/

    if (prev_model == 0) {
      //no use doing ICM in a transfer iteration.
      //in nondeficient mode, ICM does well at decreasing the energy, but it heavily aligns to the rare words

      if (fertprob_sharing_) {

        for (uint c = 0; c < ffertclass_count.size(); c++)
          ffertclass_count[c].set_constant(0.0);

        for (uint i = 1; i < nTargetWords; i++) {
          uint c = tfert_class_[i];
          for (uint k = 0; k < ffert_count[i].size(); k++)
            ffertclass_count[c][k] += ffert_count[i][k];
        }
      }

      const double log_pzero = std::log(p_zero_);
      const double log_pnonzero = std::log(p_nonzero_);

      Math1D::NamedVector<uint> dict_sum(fwcount.size(), MAKENAME(dict_sum));
      for (uint k = 0; k < fwcount.size(); k++)
        dict_sum[k] = fwcount[k].sum();

      uint nSwitches = 0;

      for (size_t s = 0; s < nSentences; s++) {

        if ((s % 10000) == 0)
          std::cerr << "ICM, sentence pair #" << s << std::endl;

        const Storage1D<uint>& cur_source = source_sentence_[s];
        const Storage1D<uint>& cur_target = target_sentence_[s];
        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

        Math1D::Vector<AlignBaseType>& cur_best_known_alignment = best_known_alignment_[s];

        const uint curI = cur_target.size();
        const uint curJ = cur_source.size();

        Math1D::Vector<uint> cur_fertilities(curI + 1, 0);
        for (uint j = 0; j < curJ; j++) {

          const uint cur_aj = cur_best_known_alignment[j];
          cur_fertilities[cur_aj]++;
        }

        Storage1D<std::vector<AlignBaseType> > hyp_aligned_source_words(curI + 1);
        if (nondeficient_) {
          for (uint j = 0; j < curJ; j++) {

            const uint cur_aj = cur_best_known_alignment[j];
            hyp_aligned_source_words[cur_aj].push_back(j);
          }
        }

        const Math3D::Tensor<double>& cur_distort_prob = distortion_prob_[curJ - 1];

        Math3D::Tensor<double>& cur_distort_count = fdistort_count[curJ - 1];

        double cur_dist_log = 0.0;
        double hyp_dist_log = 0.0;

        if (nondeficient_)
          cur_dist_log = logl(nondeficient_distortion_prob(cur_source, cur_target, hyp_aligned_source_words));

        for (uint j = 0; j < curJ; j++) {

          //std::cerr << "hyp_aligned: " << hyp_aligned_source_words << std::endl;

          const uint cur_aj = cur_best_known_alignment[j];
          const uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj - 1];
          const uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);
          Math1D::Vector<double>& cur_dictcount = fwcount[cur_word];
          Math1D::Vector<double>& cur_fert_count = ffert_count[cur_word];
          const Math1D::Vector<float>& cur_prior = prior_weight_[cur_word];
          std::vector<AlignBaseType>& cur_hyp_aligned_source_words = hyp_aligned_source_words[cur_aj];
          if (nondeficient_)
            vec_erase(cur_hyp_aligned_source_words, (AlignBaseType) j);

          double best_change = 0.0;
          uint new_aj = cur_aj;

          for (uint i = 0; i <= curI; i++) {

            /**** dict ***/
            //std::cerr << "i: " << i << ", cur_aj: " << cur_aj << std::endl;

            bool allowed = (cur_aj != i && (i != 0 || 2 * cur_fertilities[0] + 2 <= curJ));

            if (i != 0 && (cur_fertilities[i] + 1) > fertility_limit_[cur_word])
              allowed = false;

            if (align_constraints.align_set_ != AllAlignments) {

              if (align_constraints.align_set_ == ITGAlignments) {
                if (i == 0)
                  allowed = false;
                if (allowed) {
                  Math1D::Vector<AlignBaseType> hyp_alignment = cur_best_known_alignment;
                  hyp_alignment[j] = i;
                  allowed = alignment_satisfies_itg_nonull(hyp_alignment,curI,itg_ext_level,max_mid_dev,nMaxSkips,level3_maxlength);
                }
              }
              else {
                if (i == 0)
                  allowed = false;
                if (allowed) {
                  Math1D::Vector<AlignBaseType> hyp_alignment = cur_best_known_alignment;
                  hyp_alignment[j] = i;
                  allowed = alignment_satisfies_ibm_nonull(hyp_alignment,nMaxSkips);
                }
              }
            }

            if (allowed) {

              const uint new_target_word = (i == 0) ? 0 : cur_target[i - 1];
              const Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
              const uint hyp_idx = (i == 0) ? cur_source[j] - 1 : cur_lookup(j, i - 1);
              const Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

              double change = common_icm_change(cur_fertilities, log_pzero, log_pnonzero, dict_sum, cur_dictcount, hyp_dictcount,
                                                cur_prior, prior_weight_[new_target_word],  cur_fert_count, hyp_fert_count,
                                                ffertclass_count, cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, i, curJ);

              //std::cerr << "dist" << std::endl;

              /***** distortion ****/
              if (!nondeficient_) {

                if (cur_aj != 0) {

                  const uint aj_class = target_class_[cur_target[cur_aj - 1]];

                  if (par_mode_ != IBM23Nonpar) {

                    change -= -std::log(cur_distort_prob(j, cur_aj - 1, aj_class));
                  }
                  else {

                    const uint c = cur_distort_count(j, cur_aj - 1, aj_class);

                    if (c > 1) {
                      //exploit log(1) = 0
                      change -= -xlogx_table_[c];
                      change += -xlogx_table_[c - 1];
                    }

                    //room for speed-ups here
                    int total_c = 0;
                    for (uint jj = 0; jj < cur_distort_count.xDim(); jj++)
                      total_c += cur_distort_count(jj, cur_aj - 1, aj_class);

                    if (total_c > 0) {
                      //exploit log(1) = 0
                      change -= xlogx_table_[total_c];
                      change += xlogx_table_[total_c - 1];
                    }
                  }
                }
                if (i != 0) {

                  const uint i_class = target_class_[cur_target[i - 1]];

                  if (par_mode_ != IBM23Nonpar) {

                    change += -std::log(cur_distort_prob(j, i - 1, i_class));
                  }
                  else {

                    int c = cur_distort_count(j, i - 1, i_class);        //must be signed (negation below)!!
                    if (c > 0) {
                      //exploit log(1) = 0
                      change -= -xlogx_table_[c];
                      change += -xlogx_table_[c + 1];
                    }

                    //room for speed-ups here
                    int total_c = 0;
                    for (uint jj = 0; jj < cur_distort_count.xDim(); jj++)
                      total_c += cur_distort_count(jj, i - 1, i_class);

                    if (total_c > 0) {
                      //exploit log(1) = 0
                      change -= xlogx_table_[total_c];
                      change += xlogx_table_[total_c + 1];
                    }
                  }
                }

              }
              else {

                hyp_aligned_source_words[i].push_back(j);
                vec_sort(hyp_aligned_source_words[i]);

                hyp_dist_log = logl(nondeficient_distortion_prob(cur_source, cur_target, hyp_aligned_source_words));

                change += cur_dist_log; // - - = +
                change -= hyp_dist_log;

                //restore
                vec_erase<AlignBaseType>(hyp_aligned_source_words[i],j);
              }

              if (change < best_change) {
                best_change = change;
                new_aj = i;
              }
            }
          }

          if (best_change < -0.01) {

            //std::cerr << "changing!!" << std::endl;
            assert(cur_aj != new_aj);

            cur_best_known_alignment[j] = new_aj;
            nSwitches++;

            const uint new_target_word = (new_aj == 0) ? 0 : cur_target[new_aj - 1];
            Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
            const uint hyp_idx = (new_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, new_aj - 1);
            Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

            common_icm_count_change(dict_sum, cur_dictcount, hyp_dictcount, cur_fert_count, hyp_fert_count,
                                    ffertclass_count, cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, new_aj, cur_fertilities);

            //fzero_count += fertility[0];
            //fnonzero_count += curJ - 2 * fertility[0];

            //distort
            if (cur_aj != 0)
              cur_distort_count(j, cur_aj - 1, target_class_[cur_target[cur_aj - 1]])--;
            else {
              fnonzero_count += 2.0;
              fzero_count--;
            }

            if (new_aj != 0)
              cur_distort_count(j, new_aj - 1, target_class_[cur_target[new_aj - 1]])++;
            else {
              fnonzero_count -= 2.0;
              fzero_count++;
            }

            //std::cerr << "D" << std::endl;

            if (nondeficient_) {
              hyp_aligned_source_words[new_aj].push_back(j);
              vec_sort(hyp_aligned_source_words[new_aj]);
              cur_dist_log = logl(nondeficient_distortion_prob(cur_source, cur_target, hyp_aligned_source_words));
            }
          }
          else if (nondeficient_) {
            cur_hyp_aligned_source_words.push_back(j);
            vec_sort(cur_hyp_aligned_source_words);
          }
        }
      } //ICM-loop over sentences finished

      std::cerr << nSwitches << " changes in ICM stage" << std::endl;

      //DEBUG
#ifndef NDEBUG
      uint nZeroAlignments = 0;
      uint nAlignments = 0;
      for (size_t s = 0; s < source_sentence_.size(); s++) {

        nAlignments += source_sentence_[s].size();

        for (uint j = 0; j < source_sentence_[s].size(); j++) {
          if (best_known_alignment_[s][j] == 0)
            nZeroAlignments++;
        }
      }
      std::cerr << "percentage of zero-aligned words: " << (((double)nZeroAlignments) / ((double)nAlignments)) << std::endl;
#endif
      //END_DEBUG

      if (!fix_p0_) {
        double fsum = fzero_count + fnonzero_count;
        p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
        p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);
      }

      //update dictionary
      update_dict_from_counts(fwcount, prior_weight_, nSentences, 0.0, false, 0.0, 0, dict_, fert_min_dict_entry);

      //update distortion prob from counts
      Storage1D<std::map<CompactAlignedSourceWords,CountStructure> > refined_nondef_aligned_words_count(maxJ_);
      ReducedIBM3ClassDistortionModel fnondef_distort_count(distortion_prob_.size(), MAKENAME(fnondef_distort_count));

      if (nondeficient_) {

        fnondef_distort_count = fdistort_count;       //same for viterbi

        for (uint s = 0; s < source_sentence_.size(); s++) {

          const uint curJ = source_sentence_[s].size();
          const uint curI = target_sentence_[s].size();

          Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

          //a) best known alignment (=mode)
          Storage1D<std::vector<uchar> > aligned_source_words(curI + 1);
          Storage1D<WordClassType> tclass(curI);
          for (uint i = 0; i < curI; i++)
            tclass[i] = target_class_[target_sentence_[s][i]];
          for (uint j = 0; j < curJ; j++)
            aligned_source_words[cur_alignment[j]].push_back(j);

          refined_nondef_aligned_words_count[curJ - 1][CompactAlignedSourceWords(aligned_source_words,tclass)].main_count_ += 1.0;
        }
      }

      update_distortion_probs(fdistort_count, fnondef_distort_count, refined_nondef_aligned_words_count);
      
      update_fertility_prob(ffert_count, fert_min_param_entry, false);

      max_perplexity = 0.0;
      for (size_t s = 0; s < source_sentence_.size(); s++) {
        if (nondeficient_)
          max_perplexity -= logl(nondeficient_alignment_prob(s, best_known_alignment_[s]));
        else
          max_perplexity -= logl(FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]));
      }

      max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
      max_perplexity /= source_sentence_.size();

      std::cerr << "IBM-3 energy after iteration #" << iter << transfer << ": " << max_perplexity << std::endl;
    }

    if (possible_ref_alignments_.size() > 0) {

      std::cerr << "#### IBM-3-AER after iteration #" << iter << transfer << ": " << FertilityModelTrainerBase::AER() << std::endl;
      std::cerr << "#### IBM-3-fmeasure after iteration #" << iter << transfer << ": " << FertilityModelTrainerBase::f_measure() << std::endl;
      std::cerr << "#### IBM-3-DAE/S after iteration #" << iter << transfer << ": " << FertilityModelTrainerBase::DAE_S() << std::endl;
    }
  }

  if (par_mode_ == IBM23Nonpar) {

    //we compute this so that we can use it for computation of external alignments
    nonpar2par_distortion();
  }

  iter_offs_ = iter - 1;
}

#ifdef HAS_CBC
class IBM3IPHeuristic : public CbcHeuristic {
public:

  IBM3IPHeuristic(CbcModel& model, uint I, uint J, uint nNullFertVars, uint nFertilityVarsPerWord);

  IBM3IPHeuristic(const CbcHeuristic& heuristic, CbcModel& model,
                  uint I, uint J, uint nNullFertVars, uint nFertilityVarsPerWord);

  virtual CbcHeuristic* clone() const;

  virtual void resetModel(CbcModel* model);

  virtual int solution(double& objectiveValue, double* newSolution);

protected:

  uint I_;
  uint J_;
  uint nNullFertVars_;
  uint nFertilityVarsPerWord_;
};

IBM3IPHeuristic::IBM3IPHeuristic(CbcModel& model, uint I, uint J, uint nNullFertVars, uint nFertilityVarsPerWord)
  : CbcHeuristic(model), I_(I), J_(J), nNullFertVars_(nNullFertVars), nFertilityVarsPerWord_(nFertilityVarsPerWord)
{
}

IBM3IPHeuristic::IBM3IPHeuristic(const CbcHeuristic& heuristic, CbcModel& /*model */,
                                 uint I, uint J, uint nNullFertVars, uint nFertilityVarsPerWord)
  : CbcHeuristic(heuristic), I_(I), J_(J), nNullFertVars_(nNullFertVars), nFertilityVarsPerWord_(nFertilityVarsPerWord)
{
}

/*virtual*/ CbcHeuristic* IBM3IPHeuristic::clone() const
{
  return new IBM3IPHeuristic(*this, *model_, I_, J_, nNullFertVars_, nFertilityVarsPerWord_);
}

/*virtual*/ void IBM3IPHeuristic::resetModel(CbcModel* /*model */ )
{
  TODO("resetModel");
}

/*virtual*/ int IBM3IPHeuristic::solution(double& objectiveValue, double* newSolution)
{
  //uint nVars = (I_+1)*J_ + nNullFertVars_ + I_*nFertilityVarsPerWord_;

  const OsiSolverInterface* solver = model_->solver();
  const double* cur_solution = solver->getColSolution();
  const double* cost = solver->getObjCoefficients();

  const uint nVars = solver->getNumCols();

  const uint fert_var_offs = J_ * (I_ + 1) + nNullFertVars_;

  Math1D::NamedVector<uint> alignment(J_, MAKENAME(alignment));
  Math1D::NamedVector<uint> fert_count(I_ + 1, 0, MAKENAME(fert_count));

  for (uint j = 0; j < J_; j++) {

    double max_var = 0.0;
    uint aj = MAX_UINT;

    for (uint i = 0; i <= I_; i++) {

      double var_val = cur_solution[j * (I_ + 1) + i];

      if (var_val > max_var) {
        max_var = var_val;
        aj = i;
      }
    }

    alignment[j] = aj;
    fert_count[aj]++;
  }

#if 1
  if (fert_var_offs + (I_ - 1) * nFertilityVarsPerWord_ + fert_count[I_] >=
      nVars)
    return 0;

  double init_energy = 0.0;
  for (uint j = 0; j < J_; j++)
    init_energy += cost[j * (I_ + 1) + alignment[j]];
  init_energy += cost[J_ * (I_ + 1) + fert_count[0]];
  for (uint i = 1; i <= I_; i++) {
    const uint idx = fert_var_offs + (i - 1) * nFertilityVarsPerWord_ + fert_count[i];
    assert(idx < nVars);
    init_energy += cost[idx];
  }

  if (init_energy > 100.0 * objectiveValue)
    return 0;

  //greedy hillclimbing
  for (uint iter = 1; iter <= 50; iter++) {

    bool changed = false;

    // expansion moves
    for (uint j = 0; j < J_; j++) {

      const uint cur_aj = alignment[j];
      const uint j_offs = j * (I_ + 1);

      double min_cost_addon = 1e300;
      uint arg_min = MAX_UINT;

      if (cur_aj != 0 && fert_count[0] + 1 < nNullFertVars_) {

        const uint idx = J_ * (I_ + 1) + fert_count[0];
        const double addon = cost[j_offs] + cost[idx + 1] - cost[idx];
        if (addon < min_cost_addon) {
          min_cost_addon = addon;
          arg_min = 0;
        }
      }

      for (uint cand_aj = 1; cand_aj <= I_; cand_aj++) {

        if (cand_aj == cur_aj)
          continue;
        if (fert_count[cand_aj] + 1 >= nFertilityVarsPerWord_)
          continue;

        const uint idx =
          fert_var_offs + (cand_aj - 1) * nFertilityVarsPerWord_ +
          fert_count[cand_aj];
        if (idx + 1 >= nVars)
          continue;

        const double addon = cost[j_offs + cand_aj] + cost[idx + 1] - cost[idx];
        if (addon < min_cost_addon) {
          min_cost_addon = addon;
          arg_min = cand_aj;
        }
      }

      double change = min_cost_addon - cost[j_offs + cur_aj];
      const uint idx = ((cur_aj == 0) ? J_ * (I_ + 1) : fert_var_offs + (cur_aj - 1) * nFertilityVarsPerWord_) + fert_count[cur_aj];
      change += cost[idx - 1] - cost[idx];

      if (change < -0.01) {
        fert_count[cur_aj]--;
        fert_count[arg_min]++;
        alignment[j] = arg_min;
        changed = true;
      }
    }

#if 1
    //swap moves
    for (uint j1 = 0; j1 < J_ - 1; j1++) {

      const uint cur_aj1 = alignment[j1];

      double min_cost_addon = 1e300;
      uint arg_min = MAX_UINT;

      for (uint j2 = j1 + 1; j2 < J_; j2++) {

        const uint cur_aj2 = alignment[j2];

        if (cur_aj1 == cur_aj2)
          continue;

        const double addon = cost[j1 * (I_ + 1) + cur_aj2] + cost[j2 * (I_ + 1) + cur_aj1];
        if (addon < min_cost_addon) {
          min_cost_addon = addon;
          arg_min = j2;
        }
      }

      if (arg_min == MAX_UINT)
        continue;

      double change = min_cost_addon - cost[j1 * (I_ + 1) + cur_aj1] - cost[arg_min * (I_ + 1) + alignment[arg_min]];

      if (change < -0.01) {
        std::swap(alignment[j1], alignment[arg_min]);
        changed = true;
      }
    }
#endif

    if (!changed)
      break;
  }
#endif

  std::fill_n(newSolution, nVars, 0.0);

  double new_energy = 0.0;

  for (uint j = 0; j < J_; j++) {
    newSolution[j * (I_ + 1) + alignment[j]] = 1.0;
    new_energy += cost[j * (I_ + 1) + alignment[j]];
  }

  newSolution[J_ * (I_ + 1) + fert_count[0]] = 1.0;
  new_energy += cost[J_ * (I_ + 1) + fert_count[0]];

  for (uint i = 1; i <= I_; i++) {
    newSolution[fert_var_offs + (i - 1) * nFertilityVarsPerWord_ +
                              fert_count[i]] = 1.0;
    new_energy +=
      cost[fert_var_offs + (i - 1) * nFertilityVarsPerWord_ + fert_count[i]];
  }

  if (new_energy < objectiveValue) {
    return 1;
  }
  //std::cerr << "new integer energy " << new_energy << ",  previous " << objectiveValue << std::endl;

  return 0;
}

class IBM3ColCutGenerator:public CglCutGenerator {
public:

  IBM3ColCutGenerator(uint curI, uint curJ, uint nNullFertVars, uint nFertVarsPerWord)
    :curI_(curI), curJ_(curJ), nNullFertVars_(nNullFertVars), nFertVarsPerWord_(nFertVarsPerWord)
  {
  }

  virtual void generateCuts(const OsiSolverInterface& si, OsiCuts& cs, const CglTreeInfo info = CglTreeInfo())
  {
    //std::cerr << "++++++ pass: " << info.pass << std::endl;

    //if (info.level == 0 || info.pass > 0) //this make sense once per node and not at the root node
    //  return;

    //if (info.level > 0)
    //  std::cerr << "level: " << info.level << std::endl;

    const uint nSolverVars = si.getNumCols();
    const double* colLower = si.getColLower();
    const double* colUpper = si.getColUpper();
    const double* cost = si.getObjCoefficients();

    double upper_bound;
    si.getDblParam(OsiDualObjectiveLimit, upper_bound);
    //std::cerr << "#################################upper_bound: " << upper_bound << std::endl;

    const uint null_fert_var_offs = (curI_ + 1) * curJ_;
    const uint fert_var_offs = null_fert_var_offs + nNullFertVars_;

    double fix_cost = 0.0;

    Math1D::Vector<uint> fixed_alignment(curJ_, MAX_UINT);
    Math1D::Vector<uint> fert_lower_limit(curI_ + 1, 0);
    Math1D::Vector<uint> fert_upper_limit(curI_ + 1, 0);

    std::vector<int> var_idx; //all derived column cuts set variables to 0

    //1. scan for fixed vars
    //a) alignment vars
    for (uint j = 0; j < curJ_; j++) {
      for (uint l = 0; l <= curI_; l++) {

        if (colLower[j * (curI_ + 1) + l] >= 0.99) {
          fixed_alignment[j] = l;
          fix_cost += cost[j * (curI_ + 1) + l];
          fert_lower_limit[l]++;
        }
      }
    }

    for (uint f = 0; f < fert_lower_limit[0]; f++) {
      if (colUpper[null_fert_var_offs + f] > 0.01)
        var_idx.push_back(null_fert_var_offs + f);
    }

    for (uint i = 1; i <= curI_; i++) {
      for (uint f = 0; f < fert_lower_limit[i]; f++) {

        const uint idx = fert_var_offs + (i - 1) * nFertVarsPerWord_ + f;
        if (colUpper[idx] > 0.01)
          var_idx.push_back(idx);
      }
    }

    //b) null fertility vars
    for (uint c = 0; c < nNullFertVars_; c++) {

      const uint idx = null_fert_var_offs + c;

      if (colLower[idx] >= 0.99) {

        if (fert_lower_limit[0] > c)
          return;

        fert_lower_limit[0] = c;
        fert_upper_limit[0] = c;
        break;
      }
      else if (colUpper[idx] < 0.01) {
        if (fert_lower_limit[0] == c)
          fert_lower_limit[0]++;
      }
      else
        fert_upper_limit[0] = c;
    }

    //c) other fertility vars
    for (uint i = 0; i < curI_; i++) {

      for (uint c = 0; c < nFertVarsPerWord_; c++) {

        const uint idx = fert_var_offs + i * nFertVarsPerWord_ + c;
        if (idx >= nSolverVars)
          break;

        if (colLower[idx] >= 0.99) {
          fert_lower_limit[i + 1] = c;
          fert_upper_limit[i + 1] = c;
          break;
        }
        else if (colUpper[idx] < 0.01) {
          if (fert_lower_limit[i + 1] == c)
            fert_lower_limit[i + 1]++;
        }
        else
          fert_upper_limit[i + 1] = c;
      }
    }

    //2. exclusion strategy 1

    //2.1 prepare bounds

    Math1D::NamedVector<double> jcost_lower_bound(curJ_, 0.0, MAKENAME(jcost_lower_bound));
    Math1D::NamedVector<double> icost_lower_bound(curI_ + 1, 0.0, MAKENAME(icost_lower_bound));

    for (uint j = 0; j < curJ_; j++) {
      if (fixed_alignment[j] == MAX_UINT) {

        double min_cost = 1e300;

        for (uint l = 0; l <= curI_; l++) {
          if (colUpper[j * (curI_ + 1) + l] > 0.01 && fert_upper_limit[l] > 0) {
            min_cost = std::min(min_cost, cost[j * (curI_ + 1) + l]);
          }
        }

        jcost_lower_bound[j] = min_cost;
      }
    }

    double min_cost = 1e300;
    for (uint c = fert_lower_limit[0]; c <= fert_upper_limit[0]; c++) {
      if (colUpper[null_fert_var_offs + c] > 0.01) {
        min_cost = std::min(min_cost, cost[null_fert_var_offs + c]);
      }
    }
    icost_lower_bound[0] = min_cost;

    for (uint i = 0; i < curI_; i++) {

      double min_cost = 1e300;
      for (uint c = fert_lower_limit[i + 1]; c <= fert_upper_limit[i + 1]; c++) {

        const uint idx = fert_var_offs + i * nFertVarsPerWord_ + c;
        if (idx >= nSolverVars)
          break;

        if (colUpper[idx] > 0.01) {
          min_cost = std::min(min_cost, cost[idx]);
        }
      }
      icost_lower_bound[i + 1] = min_cost;
    }

    //2.2 actual exclusion stage

    const double loose_lower_bound = fix_cost + jcost_lower_bound.sum() + icost_lower_bound.sum();
    const double loose_gap = upper_bound - loose_lower_bound + 0.01;

    for (uint c = fert_lower_limit[0]; c <= fert_upper_limit[0]; c++) {
      const uint idx = null_fert_var_offs + c;
      if (colUpper[idx] < 0.99)
        continue;
      if (cost[null_fert_var_offs + c] - icost_lower_bound[0] > loose_gap) {
        var_idx.push_back(idx);
      }
    }

    for (uint i = 0; i < curI_; i++) {

      for (uint c = fert_lower_limit[i + 1]; c <= fert_upper_limit[i + 1]; c++) {
        const uint idx = fert_var_offs + i * nFertVarsPerWord_ + c;
        if (idx >= nSolverVars)
          break;

        if (colUpper[idx] < 0.99)
          continue;
        if (cost[idx] - icost_lower_bound[i + 1] > loose_gap) {
          var_idx.push_back(idx);
        }
      }
    }

    //3. exclusion strategy 2

    //3.1. prepare bound

    Math1D::Vector<double> ifert_fwd[2];
    ifert_fwd[0].resize(curJ_ + 1, 1e50);
    ifert_fwd[1].resize(curJ_ + 1, 1e50);

    uint cur_idx = 0;
    for (uint f = fert_lower_limit[1]; f <= fert_upper_limit[1]; f++)
      ifert_fwd[0][f] = cost[fert_var_offs + f];

    for (uint i = 1; i < curI_; i++) {

      const Math1D::Vector<double>& prev_fwd = ifert_fwd[cur_idx];
      cur_idx = 1 - cur_idx;

      Math1D::Vector<double>& cur_fwd = ifert_fwd[cur_idx];

      const uint idx_base = fert_var_offs + i * nFertVarsPerWord_;
      const uint limit = std::min(curJ_, (i + 1) * nFertVarsPerWord_);

      for (uint j = 0; j <= limit; j++) {

        double opt_cost = prev_fwd[j] + cost[idx_base + fert_lower_limit[i + 1]];
        for (uint f = fert_lower_limit[i + 1] + 1; f <= std::min(j, fert_upper_limit[i + 1]); f++)
          opt_cost = std::min(opt_cost, prev_fwd[j - f] + cost[idx_base + f]);

        cur_fwd[j] = opt_cost;
      }
    }

    double fert_lower_bound = 1e50;
    const Math1D::Vector<double>& last_fwd = ifert_fwd[cur_idx];
    for (uint f = fert_lower_limit[0]; f <= fert_upper_limit[0]; f++) {
      fert_lower_bound = std::min(fert_lower_bound, last_fwd[curJ_ - f] + cost[null_fert_var_offs + f]);
    }

    const double lower_bound = fix_cost + jcost_lower_bound.sum() + fert_lower_bound;
    const double gap = upper_bound - lower_bound + 0.01;

    //std::cerr << "----cut gap: " << gap << std::endl;

    //3.2 actual exclusion stage

    for (uint j = 0; j < curJ_; j++) {
      if (fixed_alignment[j] == MAX_UINT) {

        for (uint l = 0; l <= curI_; l++) {
          const uint idx = j * (curI_ + 1) + l;
          if (colUpper[idx] < 0.99)
            continue;
          if (cost[idx] - jcost_lower_bound[j] > gap
              || fert_upper_limit[l] == 0) {
            var_idx.push_back(idx);
          }
        }
      }
    }

    //4. exclusion strategy 2

    //4.1. prepare bound

    Math2D::Matrix<double> approx_icost(curI_ + 1, std::max(nNullFertVars_, nFertVarsPerWord_));

    for (uint i = 0; i <= curI_; i++) {

      std::vector<double> values;
      values.reserve(curJ_);

      for (uint j = 0; j < curJ_; j++) {
        if (colUpper[j * (curI_ + 1) + i] > 0.99)
          values.push_back(cost[j * (curI_ + 1) + i]);
      }

      const uint limit = (i == 0) ? nNullFertVars_ : nFertVarsPerWord_;
      //partial sort would suffice
      vec_sort(values);

      for (uint j = 1; j < std::min < uint > (values.size(), limit); j++)
        values[j] += values[j - 1];

      const uint base_idx = (i == 0) ? null_fert_var_offs : fert_var_offs + (i - 1) * nFertVarsPerWord_;

      approx_icost(i, 0) = (fert_lower_limit[i] == 0) ? cost[base_idx] : 1e50;

      for (uint f = 1; f < limit; f++) {

        if (f - 1 >= values.size() || fert_lower_limit[i] > f || fert_upper_limit[i] < f) {
          approx_icost(i, f) = 1e50;
        }
        else {
          approx_icost(i, f) = cost[base_idx + f] + values[f - 1];
        }
      }
    }

    //fertility exclusion based on forward-backward
    Math2D::NamedMatrix<double> fert_fwd(curJ_ + 1, curI_, 1e50, MAKENAME(fert_fwd));

    for (uint f = fert_lower_limit[0]; f <= fert_upper_limit[0]; f++) {
      fert_fwd(f, 0) = approx_icost(0, f);
    }

    for (uint i = 1; i < curI_; i++) {

      for (uint j = 0; j <= curJ_; j++) {

        double opt_cost = fert_fwd(j, i - 1) + approx_icost(i, fert_lower_limit[i]);

        for (uint f = fert_lower_limit[i] + 1; f <= std::min(j, fert_upper_limit[i]); f++)
          opt_cost = std::min(opt_cost, fert_fwd(j - f, i - 1) + approx_icost(i, f));

        fert_fwd(j, i) = opt_cost;
      }
    }

    Math2D::NamedMatrix<double> fert_bwd(curJ_ + 1, curI_, 1e50, MAKENAME(fert_bwd));
    for (uint f = fert_lower_limit[curI_]; f <= fert_upper_limit[curI_]; f++)
      fert_bwd(f, curI_ - 1) = approx_icost(curI_, f);
    for (uint i = curI_ - 1; i >= 1; i--) {

      for (uint j = 0; j <= curJ_; j++) {

        double opt_cost = fert_bwd(j, i) + approx_icost(i, fert_lower_limit[i]);

        for (uint f = fert_lower_limit[i] + 1; f <= std::min(j, fert_upper_limit[i]); f++)
          opt_cost = std::min(opt_cost, fert_bwd(j - f, i) + approx_icost(i, f));

        fert_bwd(j, i - 1) = opt_cost;
      }
    }

    //4.2 actual exclusion stage

    //exclude fert vars by combining forward and backward

    //CAUTION: CBC does not accept indices to be listed twice!

    const double upper_tol = upper_bound + 0.01;

    for (uint f = fert_lower_limit[0]; f <= fert_upper_limit[0]; f++) {

      const uint idx = null_fert_var_offs + f;

      if (colUpper[idx] > 0.01 && fert_bwd(curJ_ - f, 0) + approx_icost(0, f) > upper_tol) {
        if (vec_find(var_idx, (int)idx) == var_idx.end())
          var_idx.push_back(idx);
      }
    }

    for (uint i = 1; i < curI_; i++) {

      for (uint f = fert_lower_limit[i]; f <= fert_upper_limit[i]; f++) {

        const uint idx = fert_var_offs + (i - 1) * nFertVarsPerWord_ + f;

        if (colUpper[idx] < 0.01)
          continue;

        if (approx_icost(i, f) >= 1e50) {
          if (vec_find(var_idx, (int)idx) == var_idx.end())
            var_idx.push_back(idx);
        }
        else {

          double opt_cost = fert_fwd(0, i - 1) + fert_bwd(curJ_ - f, i);
          for (uint j = 1; j <= curJ_ - f; j++)
            opt_cost = std::min(opt_cost, fert_fwd(j, i - 1) + fert_bwd(curJ_ - f - j, i));

          opt_cost += approx_icost(i, f);

          if (opt_cost > upper_tol) {
            if (vec_find(var_idx, (int)idx) == var_idx.end())
              var_idx.push_back(idx);
          }
        }
      }
    }

    for (uint f = fert_lower_limit[curI_]; f <= fert_upper_limit[curI_]; f++) {

      const uint idx = fert_var_offs + (curI_ - 1) * nFertVarsPerWord_ + f;

      if (colUpper[idx] < 0.01)
        continue;

      if (fert_fwd(curJ_ - f, curI_ - 1) + approx_icost(curI_, f) > upper_tol) {
        if (vec_find(var_idx, (int)idx) == var_idx.end())
          var_idx.push_back(idx);
      }
    }

    if (var_idx.size() > 0) {

      //std::cerr << " adding " << var_idx.size() << " column cuts, " << nSolverVars << " vars in all" << std::endl;

      std::vector<double> var_ub(var_idx.size(), 0.0);

      OsiColCut newCut;
      newCut.setUbs(var_idx.size(), var_idx.data(), var_ub.data());

      cs.insert(newCut);
    }
  }

  virtual CglCutGenerator* clone() const
  {
    return new IBM3ColCutGenerator(curI_, curJ_, nNullFertVars_, nFertVarsPerWord_);
  }

protected:

  uint curI_;
  uint curJ_;
  uint nNullFertVars_;
  uint nFertVarsPerWord_;
};

#endif

double IBM3Trainer::compute_viterbi_alignment_ilp(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
    const SingleLookupTable& cur_lookup, Math1D::Vector<AlignBaseType>& alignment, double time_limit)
{

#ifdef HAS_CBC
  const uint curI = cur_target.size();
  const uint curJ = cur_source.size();
  const Math3D::Tensor<double>& cur_distort_prob = distortion_prob_[curJ - 1];

  uint max_fertility = 1;
  for (uint i = 0; i < curI; i++) {
    max_fertility = std::max<uint>(max_fertility, fertility_limit_[cur_target[i]]);
  }

  uint nNullFertVars = curJ / 2 + 1;
  uint nFertVarsPerWord = std::min(curJ, max_fertility) + 1;

  //std::cerr << "computing lp-relax for sentence pair, I= " << curI << ", J= " << curJ << std::endl;
  //std::cerr << "max_fertility: " << max_fertility << std::endl;

  uint nVars = (curI + 1) * curJ        //alignment variables
               + nNullFertVars + curI * nFertVarsPerWord;        //fertility variables

  uint null_fert_var_offs = (curI + 1) * curJ;
  uint fert_var_offs = null_fert_var_offs + nNullFertVars;
  //std::cerr << "fert_var_offs: " << fert_var_offs << std::endl;

  uint nConstraints = curJ      // alignment variables must sum to 1 for each source word
                      + curI + 1                //fertility variables must sum to 1 for each target word including the empty word
                      + curI + 1;               //fertility variables must be consistent with alignment variables

  uint fert_con_offs = curJ;
  uint consistency_con_offs = fert_con_offs + curI + 1;

  Math1D::NamedVector<double> cost(nVars, 0.0, MAKENAME(cost));

  Math1D::NamedVector<double> var_lb(nVars, 0.0, MAKENAME(var_lb));
  Math1D::NamedVector<double> var_ub(nVars, 1.0, MAKENAME(var_ub));

  Math1D::NamedVector<double> jcost_lower_bound(curJ, MAKENAME(jcost_lower_bound));
  Math1D::NamedVector<double> icost_lower_bound(curI + 1, MAKENAME(icost_lower_bound));

  //code cost entries for alignment variables
  for (uint j = 0; j < curJ; j++) {

    //std::cerr << "j: " << j << std::endl;

    uint s_idx = cur_source[j];
    const uint cur_offs = j * (curI + 1);

    double min_cost = 1e50;

    for (uint i = 0; i <= curI; i++) {

      if (i == 0) {

        cost[cur_offs] = (p_zero_ > 0.0) ? -std::log(dict_[0][s_idx - 1]) : 1e30;        //distortion is here handled by the fertilities
        assert(!isnan(cost[cur_offs]));
        assert(cost[cur_offs] >= 0.0);
      }
      else {
        const uint ti = cur_target[i - 1];

        cost[cur_offs + i] = -std::log(dict_[ti][cur_lookup(j, i - 1)])
                             - std::log(cur_distort_prob(j, i - 1, target_class_[cur_target[i - 1]]));
        assert(!isnan(cost[cur_offs + i]));
        assert(cost[cur_offs + 1] >= 0.0);
      }

      min_cost = std::min(min_cost, cost[cur_offs + i]);
    }

    jcost_lower_bound[j] = min_cost;
  }

  //code cost entries for the fertility variables of the empty word
  double min_empty_fert_cost = 1e50;

  Math1D::Vector<double> null_theta(curJ + 1, 0.0);
  if (p_zero_ > 0.0)
    compute_null_theta(curJ, p_zero_, p_nonzero_, null_theta);

  for (uint fert = 0; fert < nNullFertVars; fert++) {

    //std::cerr << "fert: " << fert << std::endl;

    if (curJ - fert >= fert) {

      cost[null_fert_var_offs + fert] = -null_theta[fert];
      min_empty_fert_cost = std::min(min_empty_fert_cost, cost[null_fert_var_offs + fert]);
    }
    else {
      cost[null_fert_var_offs + fert] = 1e10;
      var_ub[null_fert_var_offs + fert] = 0.0;
    }
  }
  icost_lower_bound[0] = min_empty_fert_cost;

  //code cost entries for the fertility variables of the real words
  for (uint i = 0; i < curI; i++) {

    //std::cerr << "i: " << i << std::endl;

    const uint ti = cur_target[i];

    double min_cost = 1e50;

    for (uint fert = 0; fert < nFertVarsPerWord; fert++) {

      uint idx = fert_var_offs + i * nFertVarsPerWord + fert;

      if (fert > fertility_limit_[ti]) {
        var_ub[idx] = 0.0;
        cost[idx] = 1e10;
      }
      else {

        double prob = fertility_prob_[ti][fert];
        if (prob > 1e-75) {
          if (!no_factorial_)
            prob *= ld_fac_[fert];
          cost[idx] = -logl(prob);
          assert(!isnan(cost[idx]));
        }
        else
          cost[idx] = 1e10;

        min_cost = std::min(min_cost, cost[idx]);
      }
    }

    icost_lower_bound[i + 1] = min_cost;
  }

  //std::cerr << "A" << std::endl;

  Math1D::Vector<double> ifert_fwd[2];
  ifert_fwd[0].resize(curJ + 1, 1e50);
  ifert_fwd[1].resize(curJ + 1, 1e50);

  uint cur_idx = 0;
  for (uint f = 0; f < nFertVarsPerWord; f++)
    ifert_fwd[0][f] = cost[fert_var_offs + f];

  for (uint i = 1; i < curI; i++) {

    const Math1D::Vector<double>& prev_fwd = ifert_fwd[cur_idx];
    cur_idx = 1 - cur_idx;

    Math1D::Vector<double>& cur_fwd = ifert_fwd[cur_idx];

    const uint idx_base = fert_var_offs + i * nFertVarsPerWord;
    const uint limit = std::min(curJ, (i + 1) * nFertVarsPerWord);

    for (uint j = 0; j <= limit; j++) {

      double opt_cost = prev_fwd[j] + cost[idx_base];
      for (uint f = 1; f < std::min(j + 1, nFertVarsPerWord); f++)
        opt_cost = std::min(opt_cost, prev_fwd[j - f] + cost[idx_base + f]);

      cur_fwd[j] = opt_cost;
    }
  }

  double fert_lower_bound = 1e50;
  const Math1D::Vector<double>& last_fwd = ifert_fwd[cur_idx];
  for (uint f = 0; f < nNullFertVars; f++) {
    fert_lower_bound = std::min(fert_lower_bound, last_fwd[curJ - f] + cost[null_fert_var_offs + f]);
  }

  //std::cerr << "B" << std::endl;

  uint nHighCost = 0;

  //const double upper_bound = -logl(alignment_prob(cur_source,cur_target,cur_lookup,alignment));
  double upper_bound = 0.0;
  Math1D::Vector<uint> fert_count(curI + 1, 0);
  for (uint j = 0; j < curJ; j++) {

    uint aj = alignment[j];
    fert_count[aj]++;
    upper_bound += cost[j * (curI + 1) + aj];
  }
  upper_bound += cost[null_fert_var_offs + fert_count[0]];
  for (uint i = 1; i <= curI; i++)
    upper_bound += cost[fert_var_offs + (i - 1) * nFertVarsPerWord + fert_count[i]];

  const double jcost_lower_sum = jcost_lower_bound.sum();
  const double lower_bound = jcost_lower_sum + fert_lower_bound;
  const double loose_lower_bound = jcost_lower_sum + icost_lower_bound.sum();

  //std::cerr << "lower bound: " << lower_bound << " = " << jcost_lower_bound.sum() << " + "
  //    << ifert_cost(curJ,curI) << std::endl;

  const double loose_gap = upper_bound - loose_lower_bound + 0.01;
  const double gap = upper_bound - lower_bound + 0.01;
  //std::cerr << "loose_gap: " << loose_gap << ", gap: " << gap << std::endl;
  assert(gap <= loose_gap + 1e-8);

  //std::cerr << "++apriori gap: " << gap << std::endl;

  for (uint j = 0; j < curJ; j++) {

    for (uint aj = 0; aj <= curI; aj++) {
      if (cost[j * (curI + 1) + aj] - jcost_lower_bound[j] >= gap) {

        var_ub[j * (curI + 1) + aj] = 0.0;
        nHighCost++;
      }
    }
  }

  //std::cerr << nHighCost << " of " << (curJ*(curI+1)) << " alignment vars could be excluded" << std::endl;

  Math1D::Vector<double> ibound2(curI + 1);
  Math2D::Matrix<double> approx_icost(curI + 1, std::max(nNullFertVars, nFertVarsPerWord));

  for (uint i = 0; i <= curI; i++) {

    std::vector<double> values;
    values.reserve(curJ);

    for (uint j = 0; j < curJ; j++) {
      if (var_ub[j * (curI + 1) + i] > 0.99)
        values.push_back(cost[j * (curI + 1) + i]);
    }

    const uint limit = (i == 0) ? nNullFertVars : nFertVarsPerWord;
    //partial sort would suffice
    vec_sort(values);

    for (uint j = 1; j < std::min < uint > (values.size(), limit); j++)
      values[j] += values[j - 1];

    const uint base_idx = (i == 0) ? null_fert_var_offs : fert_var_offs + (i - 1) * nFertVarsPerWord;

    approx_icost(i, 0) = cost[base_idx];

    double min_cost = approx_icost(i, 0);

    for (uint f = 1; f < limit; f++) {

      if (f - 1 >= values.size()) {
        approx_icost(i, f) = 1e50;
        var_ub[base_idx + f] = 0.0;
        nHighCost++;
      }
      else {
        approx_icost(i, f) = cost[base_idx + f] + values[f - 1];
        min_cost = std::min(min_cost, approx_icost(i, f));
      }
    }
    ibound2[i] = min_cost;
  }

  const double loose_bound2 = ibound2.sum();
  const double loose_gap2 = upper_bound - loose_bound2 + 0.01;

  //std::cerr << "C" << std::endl;
  for (uint f = 0; f < nNullFertVars; f++) {

    const uint idx = null_fert_var_offs + f;

    if (cost[idx] - icost_lower_bound[0] >= loose_gap
        || approx_icost(0, f) - ibound2[0] >= loose_gap2) {
      var_ub[idx] = 0.0;
      approx_icost(0, f) = 1e50;
      nHighCost++;
    }
  }

  for (uint i = 1; i <= curI; i++) {

    for (uint f = 0; f < nFertVarsPerWord; f++) {

      const uint idx = fert_var_offs + (i - 1) * nFertVarsPerWord + f;

      if (cost[idx] - icost_lower_bound[i] >= loose_gap
          || approx_icost(i, f) - ibound2[i] >= loose_gap2) {
        var_ub[idx] = 0.0;
        approx_icost(i, f) = 1e50;
        nHighCost++;
      }
    }
  }

  //fertility exclusion based on forward-backward
  Math2D::NamedMatrix<double> fert_fwd(curJ + 1, curI, 1e50, MAKENAME(fert_fwd));

  for (uint f = 0; f < nNullFertVars; f++) {
    fert_fwd(f, 0) = approx_icost(0, f);
  }

  for (uint i = 1; i < curI; i++) {

    for (uint j = 0; j <= curJ; j++) {

      double opt_cost = fert_fwd(j, i - 1) + approx_icost(i, 0);

      for (uint f = 1; f < std::min(j + 1, nFertVarsPerWord); f++)
        opt_cost = std::min(opt_cost, fert_fwd(j - f, i - 1) + approx_icost(i, f));

      fert_fwd(j, i) = opt_cost;
    }
  }

  Math2D::NamedMatrix<double> fert_bwd(curJ + 1, curI, 1e50, MAKENAME(fert_bwd));
  for (uint f = 0; f < nFertVarsPerWord; f++)
    fert_bwd(f, curI - 1) = approx_icost(curI, f);
  for (uint i = curI - 1; i >= 1; i--) {

    for (uint j = 0; j <= curJ; j++) {

      double opt_cost = fert_bwd(j, i) + approx_icost(i, 0);

      for (uint f = 1; f < std::min(j + 1, nFertVarsPerWord); f++)
        opt_cost = std::min(opt_cost, fert_bwd(j - f, i) + approx_icost(i, f));

      fert_bwd(j, i - 1) = opt_cost;
    }
  }

  //exclude fert vars by combining forward and backward
  const double upper_tol = upper_bound + 0.01;

  for (uint f = 0; f < nNullFertVars; f++) {

    if (fert_bwd(curJ - f, 0) + approx_icost(0, f) > upper_tol) {
      var_ub[null_fert_var_offs + f] = 0.0;
      nHighCost++;
    }
  }

  for (uint i = 1; i < curI; i++) {

    for (uint f = 0; f < nFertVarsPerWord; f++) {

      if (var_ub[fert_var_offs + (i - 1) * nFertVarsPerWord + f] < 0.01)
        continue;

      double opt_cost = fert_fwd(0, i - 1) + fert_bwd(curJ - f, i);
      for (uint j = 1; j <= curJ - f; j++)
        opt_cost = std::min(opt_cost, fert_fwd(j, i - 1) + fert_bwd(curJ - f - j, i));

      opt_cost += approx_icost(i, f);

      if (opt_cost > upper_tol) {
        var_ub[fert_var_offs + (i - 1) * nFertVarsPerWord + f] = 0.0;
        nHighCost++;
      }
    }
  }

  for (uint f = 0; f < nFertVarsPerWord; f++) {

    double opt_cost = fert_fwd(curJ - f, curI - 1) + approx_icost(curI, f);

    if (opt_cost > upper_tol) {
      var_ub[fert_var_offs + (curI - 1) * nFertVarsPerWord + f] = 0.0;
      nHighCost++;
    }
  }

  for (uint v = 0; v < nVars; v++) {

    assert(!isnan(cost[v]));

    if (cost[v] > 1e10) {
      nHighCost++;
      cost[v] = 1e10;
      var_ub[v] = 0.0;
    }
  }

  //if (nHighCost > 0) //std::cerr << "WARNING: dampened " << nHighCost << " cost entries" << std::endl;

  //   std::cerr << "highest cost: " << cost.max() << std::endl;

  Math1D::NamedVector<double> rhs(nConstraints, 1.0, MAKENAME(rhs));

  for (uint c = consistency_con_offs; c < nConstraints; c++)
    rhs[c] = 0.0;

  //code matrix constraints
  uint nMatrixEntries = (curI + 1) * curJ       // for the alignment unity constraints
                        + nNullFertVars + 1 + curI * (nFertVarsPerWord + 1)       // for the fertility unity constraints
                        + nNullFertVars + curJ + curI * (nFertVarsPerWord - 1 + curJ);    // for the consistency constraints

  SparseMatrixDescription<double> lp_descr(nMatrixEntries, nConstraints, nVars);

  //code unity constraints for alignment variables
  for (uint j = 0; j < curJ; j++) {

    for (uint v = j * (curI + 1); v < (j + 1) * (curI + 1); v++) {
      if (var_ub[v] > 0.0)
        lp_descr.add_entry(j, v, 1.0);
    }
  }

  //code unity constraints for fertility variables
  for (uint fert = 0; fert < nNullFertVars; fert++) {
    if (var_ub[null_fert_var_offs + fert] > 0.0)
      lp_descr.add_entry(fert_con_offs, null_fert_var_offs + fert, 1.0);
  }

  for (uint i = 1; i <= curI; i++) {

    for (uint fert = 0; fert < nFertVarsPerWord; fert++) {
      const uint col = fert_var_offs + (i - 1) * nFertVarsPerWord + fert;

      if (var_ub[col] > 0.0) {
        lp_descr.add_entry(fert_con_offs + i, col, 1.0);
      }
    }
  }

  Math1D::Vector<uint> maxFert(curI + 1, 0);

  //code consistency constraints
  {
    uint nOpen = 0;
    for (uint j = 0; j < curJ; j++) {

      const uint col = j * (curI + 1);

      if (var_ub[col] > 0.0) {
        lp_descr.add_entry(consistency_con_offs, col, -1.0);
        nOpen++;
      }
    }

    for (uint fert = 1; fert < std::min(nOpen + 1, nNullFertVars); fert++) {

      const uint col = null_fert_var_offs + fert;
      if (var_ub[col] > 0.0) {
        lp_descr.add_entry(consistency_con_offs, col, fert);
        maxFert[0] = fert;
      }
    }
    for (uint f = nOpen + 1; f < nNullFertVars; f++)
      var_ub[null_fert_var_offs + f] = 0.0;
  }

  for (uint i = 1; i <= curI; i++) {

    const uint row = consistency_con_offs + i;

    uint nOpen = 0;
    for (uint j = 0; j < curJ; j++) {

      const uint col = j * (curI + 1) + i;

      if (var_ub[col] > 0.0) {

        lp_descr.add_entry(row, col, -1.0);
        nOpen++;
      }
    }

    for (uint fert = 1; fert < std::min(nOpen + 1, nFertVarsPerWord); fert++) {

      const uint col = fert_var_offs + (i - 1) * nFertVarsPerWord + fert;
      if (var_ub[col] > 0.0) {
        lp_descr.add_entry(row, col, fert);
        maxFert[i] = fert;
      }
    }
    for (uint f = nOpen + 1; f < nFertVarsPerWord; f++)
      var_ub[fert_var_offs + (i - 1) * nFertVarsPerWord + f] = 0.0;
  }

  Math1D::Vector<uint> row_start(nConstraints + 1);  //Not useful for Osi load without CoinPackedMatrix, that requires column-sort
  lp_descr.sort_by_row(row_start, true);

  OsiClpSolverInterface clp_interface;

  clp_interface.setLogLevel(0);
  clp_interface.messageHandler()->setLogLevel(0);

  //clp_interface.setDblParam(OsiDualTolerance,1e-10);
  //clp_interface.setDblParam(OsiPrimalTolerance,1e-10);

  CoinPackedMatrix coinMatrix(false, (int*)lp_descr.row_indices(), (int*)lp_descr.col_indices(), lp_descr.value(), lp_descr.nEntries());

  //this doesn't work either (gives an invalid free):
  // CoinPackedMatrix* m = &coinMatrix;
  // double* collb = var_lb.direct_access();
  // double* colub = var_ub.direct_access();
  // double* obj = cost.direct_access();
  // double* rowlb = rhs.direct_access();
  // double* rowub = rhs.direct_access();
  // clp_interface.assignProblem(m, collb, colub, obj, rowlb, rowub);

  clp_interface.loadProblem(coinMatrix, var_lb.direct_access(), var_ub.direct_access(), cost.direct_access(), rhs.direct_access(), rhs.direct_access());

  const uint nSolverVars = clp_interface.getNumCols();

  //std::cerr << "nVars: " << nVars << ", solver knows " << nSolverVars << " vars" << std::endl;

  for (uint v = 0; v < nSolverVars; v++) {
    clp_interface.setInteger(v);
  }

  //std::clock_t tStartCLP, tEndCLP;

  //DEBUG
  // for (uint j=0; j < curJ; j++)
  //   clp_interface.setColLower(j*(curI+1)+alignment[j],1.0);

  // clp_interface.setHintParam(OsiDoPresolveInResolve,true,OsiHintDo);
  // clp_interface.resolve();
  // clp_interface.setHintParam(OsiDoPresolveInResolve,false,OsiHintDo);

  // for (uint j=0; j < curJ; j++)
  //   clp_interface.setColLower(j*(curI+1)+alignment[j],0.0);
  // //std::cerr << "calling second solve!" << std::endl;
  // clp_interface.initialSolve();
  //END_DEBUG

  //tStartCLP = std::clock();

  int error = 0;
  //presolving deteriorates performance
  //clp_interface.setHintParam(OsiDoPresolveInResolve,true,OsiHintDo);
  clp_interface.resolve();
  //clp_interface.setHintParam(OsiDoPresolveInResolve,false,OsiHintDo);
  //clp_interface.initialSolve();
  error = 1 - clp_interface.isProvenOptimal();

  if (error) {
    INTERNAL_ERROR << "solving the LP-relaxation failed. Exiting..." << std::endl;
    exit(1);
  }
  //tEndCLP = std::clock();
  //std::cerr << "CLP-time: " << diff_seconds(tEndCLP,tStartCLP) << " seconds. " << std::endl;

  const double* solution = clp_interface.getColSolution();

  uint nNonIntegral = 0;
  uint nNonIntegralFert = 0;

  for (uint v = 0; v < nSolverVars; v++) {

    double var_val = solution[v];

    if (var_val > 0.01 && var_val < 0.99) {
      nNonIntegral++;

      if (v >= fert_var_offs)
        nNonIntegralFert++;
    }
  }

  //std::cerr << nNonIntegral << " non-integral variables (" << (nNonIntegral - nNonIntegralFert)
  //    << "/" << nNonIntegralFert << ")" << std::endl;

  // double temp;
  // clp_interface.getDblParam(OsiPrimalTolerance,temp);
  // std::cerr << "standard Clp primal tolerance: " << temp << std::endl;
  // clp_interface.getDblParam(OsiDualTolerance,temp);
  // std::cerr << "standard Clp dual tolerance: " << temp << std::endl;

  CbcModel cbc_model(clp_interface);

  if (nNonIntegral > 0) {

    //std::cerr << "lp-relax for sentence pair #" << s << ", I= " << curI << ", J= " << curJ << std::endl;

    //no point identifying SOS because there is next to no branching
    //clp_interface.findIntegersAndSOS(false);
    clp_interface.setupForRepeatedUse();

    // std::cerr << "standard CBC gap: " << cbc_model.getDblParam(CbcModel::CbcAllowableGap) << std::endl;
    // std::cerr << "standard CBC integer tolerance: " << cbc_model.getDblParam(CbcModel::CbcIntegerTolerance) << std::endl;
    // std::cerr << "standard CBC smallest change: " << cbc_model.getDblParam(CbcModel::CbcSmallestChange) << std::endl;
    // std::cerr << "standard CBC sum change: " << cbc_model.getDblParam(CbcModel::CbcSumChange) << std::endl;
    // std::cerr << "standard CBC small change: " << cbc_model.getDblParam(CbcModel::CbcSmallChange) << std::endl;

    if (utmost_ilp_precision_) {
      cbc_model.setDblParam(CbcModel::CbcAllowableGap, 0.0);
      cbc_model.setDblParam(CbcModel::CbcIntegerTolerance, 0.0);
    }

    cbc_model.messageHandler()->setLogLevel(0);
    cbc_model.setLogLevel(0);

    if (time_limit > 0.0) {
      cbc_model.setMaximumSeconds(time_limit);
      std::cerr << "time limit: " << time_limit << std::endl;
    }

    CglGomory gomory_cut;
    gomory_cut.setLimit(500);
    gomory_cut.setAway(0.01);
    gomory_cut.setLimitAtRoot(500);
    gomory_cut.setAwayAtRoot(0.01);
    cbc_model.addCutGenerator(&gomory_cut, 0, "Gomory Cut");

    // CglProbing probing_cut;
    // probing_cut.setUsingObjective(true);
    // probing_cut.setMaxPass(10);
    // probing_cut.setMaxPassRoot(50);
    // probing_cut.setMaxProbe(100);
    // probing_cut.setMaxProbeRoot(500);
    // probing_cut.setMaxLook(150);
    // probing_cut.setMaxLookRoot(1500);
    //cbc_model.addCutGenerator(&probing_cut,0,"Probing Cut");

    //CglRedSplit redsplit_cut;
    //redsplit_cut.setLimit(1500);
    //cbc_model.addCutGenerator(&redsplit_cut,0,"RedSplit Cut");

    //CglMixedIntegerRounding mi1_cut;
    //cbc_model.addCutGenerator(&mi1_cut,0,"Mixed Integer Cut 1");

    //CglMixedIntegerRounding2 mi2_cut;
    //cbc_model.addCutGenerator(&mi2_cut,0,"Mixed Integer 2");

    //CglTwomir twomir_cut;
    //cbc_model.addCutGenerator(&twomir_cut,0,"Twomir Cut");

    //CglLandP landp_cut;
    //cbc_model.addCutGenerator(&landp_cut,0,"LandP Cut");

    //CglOddHole oddhole_cut;
    //cbc_model.addCutGenerator(&oddhole_cut,0,"OddHole Cut");

    //CglClique clique_cut;
    //cbc_model.addCutGenerator(&clique_cut,0,"Clique Cut");

    //CglStored stored_cut;
    //cbc_model.addCutGenerator(&stored_cut,0,"Stored Cut");

    Math2D::Matrix<uint> active_rows(curI + 1, 3, 0);

    active_rows(0, 0) = consistency_con_offs;
    active_rows(0, 1) = null_fert_var_offs;
    active_rows(0, 2) = null_fert_var_offs + maxFert[0];

    for (uint i = 1; i <= curI; i++) {
      active_rows(i, 0) = consistency_con_offs + 1;
      active_rows(i, 1) = fert_var_offs + (i - 1) * nFertVarsPerWord;
      active_rows(i, 2) =
        fert_var_offs + (i - 1) * nFertVarsPerWord + maxFert[i];
    }

    CountCutGenerator cc_cut(lp_descr, row_start, active_rows, -1.0);
    cbc_model.addCutGenerator(&cc_cut, 0, "Count Cut");

    CountColCutGenerator ccc_cut(lp_descr, row_start, active_rows, -1.0);
    cbc_model.addCutGenerator(&ccc_cut, 0, "Count Col Cut");

    IBM3ColCutGenerator exclusion_cut(curI, curJ, nNullFertVars, nFertVarsPerWord);
    cbc_model.addCutGenerator(&exclusion_cut, 0, "Exclusion Cut");

    IBM3IPHeuristic ibm3_heuristic(cbc_model, curI, curJ, nNullFertVars, nFertVarsPerWord);
    ibm3_heuristic.setWhereFrom(63);
    cbc_model.addHeuristic(&ibm3_heuristic, "IBM3 Heuristic");

    /*** set initial upper bound given by alignment ****/
    {
      Math1D::Vector<double> best_sol(nSolverVars, 0.0);

      //NOTE: the variable exclusion stage cannot have excluded variables used for the upper bound since it is based on that bound

      for (uint j = 0; j < curJ; j++) {

        uint aj = alignment[j];
        best_sol[j * (curI + 1) + aj] = 1.0;
        assert(var_ub[j * (curI + 1) + aj] == 1.0);
      }
      best_sol[null_fert_var_offs + fert_count[0]] = 1.0;
      assert(var_ub[null_fert_var_offs + fert_count[0]] == 1.0);
      for (uint i = 1; i <= curI; i++) {
        best_sol[fert_var_offs + (i - 1) * nFertVarsPerWord + fert_count[i]] = 1.0;
        assert(var_ub[fert_var_offs + (i - 1) * nFertVarsPerWord + fert_count[i]] == 1.0);
        assert(fert_var_offs + (i - 1) * nFertVarsPerWord + fert_count[i] < nSolverVars);
      }

      cbc_model.setBestSolution(best_sol.direct_access(), nSolverVars, upper_bound, true);
    }

    //std::cerr << "+++++ obj value2: " << cbc_model.getMinimizationObjValue() << std::endl;

    cbc_model.branchAndBound();

    //we find that with Gomory Cuts usually just the root node is evaluated, i.e. no branching
    uint nNodesUsed = cbc_model.getNodeCount2();
    if (nNodesUsed > 0)
      std::cerr << "++++ needed " << nNodesUsed << " nodes" << std::endl;

    solution = cbc_model.bestSolution();

    if (!cbc_model.isProvenOptimal()) {

      std::cerr << "ERROR: the optimal solution could not be found. Exiting..." << std::endl;
      exit(1);
    }
    if (cbc_model.isProvenInfeasible()) {

      std::cerr << "ERROR: problem marked as infeasible. Exiting..." << std::endl;
      exit(1);
    }
    if (cbc_model.isAbandoned()) {

      std::cerr << "ERROR: problem was abandoned. Exiting..." << std::endl;
      exit(1);
    }
  }

  double energy = 0.0;

  alignment.resize(curJ);
  //alignment.set_constant(10*curI); //DEBUG

  uint nNonIntegralVars = 0;

  Math1D::Vector<uint> fert(curI + 1, 0);

  for (uint j = 0; j < curJ; j++) {

    double max_val = 0.0;
    uint arg_max = MAX_UINT;

    for (uint i = 0; i <= curI; i++) {

      if (j * (curI + 1) + i >= nSolverVars)
        continue;

      double val = solution[j * (curI + 1) + i];

      if (val > 0.01 && val < 0.99)
        nNonIntegralVars++;

      if (val > max_val) {

        //if (arg_max != MAX_UINT)
        //  std::cerr << "error: two values!" << std::endl;

        //std::cerr << "var " << (j*curI+i) << " is " << val << std::endl;

        max_val = val;
        arg_max = i;
      }
    }

    assert(arg_max != MAX_UINT);
    alignment[j] = arg_max;
    energy += cost[j * (curI + 1) + arg_max];
    fert[arg_max]++;
  }

  energy += cost[null_fert_var_offs + fert[0]];
  for (uint i = 0; i < curI; i++)
    energy += cost[fert_var_offs + i * nFertVarsPerWord + fert[i + 1]];

  // for (uint v=null_fert_var_offs; v < nSolverVars; v++) {
  //   energy += cost[v] * solution[v];
  // }

  //DEBUG
  //double sum = std::accumulate(solution,solution+nSolverVars,0.0);
  //assert(  fabs(sum - curJ-curI-1) < 0.01 );
  //END_DEBUG

  //std::cerr << "fert: " << fert << std::endl;
  //if (solution[null_fert_var_offs + fert[0]] < 0.95)
  //  std::cerr << "error for NULL: " << solution[null_fert_var_offs + fert[0]] << std::endl;
  // for (uint i=1; i <= curI; i++) {
  //   if (solution[fert_var_offs + (i-1)*nFertVarsPerWord + fert[i]] < 0.95) {
  //     std::cerr << "error for " << i << ": " << solution[fert_var_offs + (i-1)*nFertVarsPerWord + fert[i]] << std::endl;
  //     std::cerr << "all fert vars: " << std::endl;
  //     for (uint f=0; f < nFertVarsPerWord; f++)
  //    std::cerr << (fert_var_offs + (i-1)*nFertVarsPerWord + f) << ", "
  //              << solution[fert_var_offs + (i-1)*nFertVarsPerWord + f] << ", ub "
  //              << var_ub[fert_var_offs + (i-1)*nFertVarsPerWord + f] << std::endl;
  //   }
  // }

  // std::cerr << nNonIntegralVars << " non-integral variables after branch and cut" << std::endl;

#ifndef NDEBUG
  long double clp_prob = expl(-energy);
  long double actual_prob = alignment_prob(cur_source, cur_target, cur_lookup, alignment);
  double ratio = clp_prob / actual_prob;
#endif

  // std::cerr << "clp alignment: " << alignment << std::endl;
  // std::cerr << "clp-prob:    " << clp_prob << std::endl;
  // std::cerr << "actual prob:      " << actual_prob << std::endl;

  assert(ratio >= 0.999 && ratio <= 1.001);

  return energy;
#else
  return -logl(alignment_prob(cur_source, cur_target, cur_lookup, alignment));
#endif
}

long double IBM3Trainer::compute_itg_viterbi_alignment_noemptyword(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
    const SingleLookupTable& cur_lookup, Math1D::Vector<AlignBaseType>& alignment,
    uint ext_level, int max_mid_dev, uint level3_maxlength) const
{
  //std::cerr << "******** compute_itg_viterbi_alignment_noemptyword() J: " << cur_source.size() << ", I: "
  //          << cur_target.size() << " **********" << std::endl;

  const int curI = cur_target.size();
  const int curJ = cur_source.size();
  const Math3D::Tensor<double>& cur_distort_prob = distortion_prob_[curJ - 1];

  assert(cur_distort_prob.min() > 0.0);

  ushort max_fert = 1;
  for (uint i=0; i < curI; i++)
    max_fert = std::max(max_fert, fertility_limit_[cur_target[i]]);

  NamedStorage2D<Math2D::TriMatrix<long double> > score(curJ + 1, curJ, MAKENAME(score));
  NamedStorage2D<Math2D::TriMatrix<uint> > trace(curJ + 1, curJ, MAKENAME(trace));

  NamedStorage2D<Math2D::Matrix<long double> > level3_score;
  if (ext_level >= 4) {
    level3_score.resize(curJ-1,curI-1);
    for (int j=0; j < curJ-1; j++) {
      for (int i=0; i < curI-1; i++) {
        ibmconstrained_viterbi_subprob_noemptyword(cur_source, cur_target, cur_lookup, j, std::min<int>(j+level3_maxlength-1,curJ-1),
            i, std::min<int>(i+level3_maxlength-1,curI-1), &level3_score(j,i));
      }
    }
  }

  for (uint J = 1; J <= curJ; J++) {

    for (uint j = 0; j <= curJ-J; j++) {
      score(J, j).resize(curI, 0.0);
      trace(J, j).resize(curI, MAX_UINT);
    }
  }

  Math2D::Matrix<double> jcost(curJ,curI);
  for (uint i=0; i < curI; i++) {
    const uint ti = cur_target[i];
    for (uint j=0; j < curJ; j++) {
      jcost(j,i) = cur_distort_prob(j, i, target_class_[ti]) * dict_[ti][cur_lookup(j, i)];
      //assert(jcost(j,i) > 0.0);
    }
  }

  //base case 1: each j aligned to only one target word

  //without NULL, the remaining curJ-1 words together have to take at least div target words
  const uint div = (curJ-1) / max_fert + std::min(1,(curJ-1) % max_fert);
  const int max_span = (curJ-1 > 0) ? std::max<int>(1,int(curI-div)) : curI;

  for (uint j = 0; j < curJ; j++) {

    //std::cerr << "j: " << j << std::endl;

    Math2D::TriMatrix<long double>& score1 = score(1, j);
    Math2D::TriMatrix<uint>& trace1 = trace(1, j);

    for (uint i = 0; i < curI; i++) {

      const uint ti = cur_target[i];

      score1(i,i) = jcost(j, i) * fertility_prob_[ti][1];
      trace1(i,i) = i;
    }

    for (uint I=1; I < max_span; I++) {

      for (uint i=0; i < curI-I; i++) {

        long double hyp1 = score1(i,i+I-1) * fertility_prob_[cur_target[i+I]][0];
        long double hyp2 = fertility_prob_[cur_target[i]][0] * score1(i+1,i+I);

        if (hyp1 > hyp2) {
          score1(i,i+I) = hyp1;
          trace1(i,i+I) = trace1(i,i+I-1);
        }
        else {
          score1(i,i+I) = hyp2;
          trace1(i,i+I) = trace1(i+1,i+I);
        }
      }
    }
  }

  //base case 2: several adjacent js aligned to a single i
  for (int i=0; i < curI; i++) {

    //std::cerr << "i: " << i << std::endl;

    const uint ti = cur_target[i];

    for (int j=0; j < curJ-1; j++) {

      long double prob = jcost(j,i);

      for (int jj=j+1; jj < std::min(j+fertility_limit_[ti]+1,curJ); jj++) {

        const uint J = jj-j+1;

        prob *= jcost(jj,i);
        long double cur_prob = prob * fertility_prob_[ti][J];
        if (!no_factorial_)
          cur_prob *= ld_fac_[J];
        score(J,j)(i,i) = cur_prob;
      }
    }
  }

  const double J2fac = (no_factorial_) ? 1.0 : ld_fac_[2];
  const double J3fac = (no_factorial_) ? 1.0 : ld_fac_[3];
  const double J4fac = (no_factorial_) ? 1.0 : ld_fac_[4];

  for (int J = 2; J <= curJ; J++) {

    //if (curJ == 62)
    //  std::cerr << "J: " << J << std::endl;

    //without NULL, the remaining curJ-J words together have to take a least div target words
    //const uint max_span = curI;
    const int div = (curJ-J) / max_fert + std::min(1,(curJ-J) % max_fert);
    const int max_span = (curJ-J > 0) ? std::max(1,curI-div) : curI;

    const uint idiv = J / max_fert + std::min(1, J % max_fert);
    //const uint Istart = (J == curJ) ? curI : 1;
    const int Istart = (J == curJ) ? std::max<uint>(2,curI) : std::max<uint>(2,idiv);

    for (int I = Istart; I <= max_span; I++) {

      //if (curJ == 62)
      //  std::cerr << "I: " << I << std::endl;

      for (int i = 0; i < (curI - (I - 1)); i++) {

        const int ii = i + I - 1;

        //if (curJ == 62) {
        //  std::cerr << "i: " << i << ", ii: " << ii << std::endl;
        //}

        assert(ii < curI);
        assert(ii >= i);

        const uint ti = cur_target[i];
        const uint tii = cur_target[ii];

        const Math1D::Vector<double>& fert_prob_ti = fertility_prob_[ti];
        const Math1D::Vector<double>& fert_prob_tii = fertility_prob_[tii];

        const int imid_point = i + (I / 2);

        const int i_lower = std::max(i,imid_point-max_mid_dev);
        const int i_upper = std::min(ii,imid_point+max_mid_dev+1);

        const uint ti_limit = fertility_limit_[ti];
        const uint tii_limit = fertility_limit_[tii];
        const uint limit_sum = ti_limit + tii_limit;

        //if (curJ == 62) {
        //  std::cerr << "i_lower: " << i_lower << std::endl;
        //  std::cerr << "i_upper: " << i_upper << std::endl;
        //}

        for (int j = 0; j < (curJ - (J - 1)); j++) {

          //if (curJ == 62)
          //  std::cerr << "j: " << j << std::endl;

          const int jj = j + J - 1;
          assert(jj < curJ);
          assert(jj >= j);

          long double best_prob = 0.0;
          uint trace_entry = MAX_UINT;

          if (ext_level >= 1 && I <= 10 && J <= std::min(10,max_fert+1) && J > 2) {

            //if (curJ == 62)
            //  std::cerr << "ext1" << std::endl;

            long double between_prob = 1.0;
            for (uint iii = i + 1; iii <= ii - 1; iii++)
              between_prob *= fertility_prob_[cur_target[iii]][0];

            if (limit_sum >= J && ti_limit >= 2) {

              //if (curJ == 62)
              //  std::cerr << "ext1.1" << std::endl;

              const uint maxCard = J-2;
              Math1D::Vector<double> score[2];
              score[0].resize(maxCard+1,0.0);
              score[1].resize(maxCard+1,0.0);

              Storage2D<bool> trace(maxCard+1,J-2,false);

              score[0][0] = jcost(j+1,i);
              score[0][1] = jcost(j+1,ii);
              trace(1,0) = true;

              uint k=0;
              for (uint jjj = j+2; jjj < jj; jjj++) {

                const Math1D::Vector<double>& prev_score = score[k];
                k = 1-k;
                Math1D::Vector<double>& cur_score = score[k];

                const double hypi_cost = jcost(jjj,i);
                const double hypii_cost = jcost(jjj,ii);

                cur_score[0] = prev_score[0] * hypi_cost;
                for (uint c=1; c < cur_score.size(); c++) {
                  const double scorei = prev_score[c] * hypi_cost;
                  const double scoreii = prev_score[c-1] * hypii_cost;

                  if (scorei > scoreii) {
                    cur_score[c] = scorei;
                  }
                  else {
                    cur_score[c] = scoreii;
                    trace(c,jjj-j-1) = true;
                  }
                }
              }

              const Math1D::Vector<double>& last_score = score[k];

              long double best_ext_prob = 0.0;
              uint best_c = 0;
              //card 0 is not an extension!
              for (uint c = 1; c <= std::min<uint>(maxCard,tii_limit); c++) {
                if (J - c > ti_limit)
                  continue;

                const long double hyp_score = last_score[c] * fert_prob_ti[J-c] * fert_prob_tii[c];

                if (best_ext_prob < hyp_score) {
                  best_ext_prob = hyp_score;
                  best_c = c;
                }
              }

              //if (curJ == 62)
              //  std::cerr << "best_ext_prob: " << best_ext_prob << std::endl;

              long double hyp_prob = best_ext_prob * jcost(j,i) * jcost(jj,i);
              if (!no_factorial_) {
                hyp_prob *= ld_fac_[J-best_c];
                hyp_prob *= ld_fac_[best_c];
              }
              hyp_prob *= between_prob;

              if (hyp_prob > best_prob) {

                std::set<uint> iiset;
                uint c = best_c;
                int jjj = J-3;

                while (jjj >= 0) {

                  if (trace(c,jjj)) {
                    c--;
                    iiset.insert(jjj+1);
                  }
                  jjj--;
                }

                //DEBUG
                // Storage1D<uint> sub_source(J);
                // for (uint k=0; k < J; k++)
                // sub_source[k] = cur_source[j+k];

                // Storage1D<uint> sub_target(I);
                // for (uint k=0; k < I; k++)
                // sub_target[k] = cur_target[i+k];

                // Math1D::Vector<AlignBaseType> sub_alignment(J,1);
                // for (std::set<uint>::const_iterator it = iiset.begin(); it != iiset.end(); it++)
                // sub_alignment[*it] = I;

                // SingleLookupTable sub_lookup(J,I);
                // Math2D::Matrix<double> sub_distort_prob(J,I);
                // for (uint k1 = 0; k1 < J; k1++) {
                // for (uint k2 = 0; k2 < I; k2++) {
                // sub_lookup(k1,k2) = cur_lookup(j+k1,i+k2);
                // sub_distort_prob(k1,k2) = cur_distort_prob(j+k1,i+k2);
                // }
                // }

                // long double check_prob = alignment_prob(sub_source, sub_target, sub_lookup, sub_alignment, &sub_distort_prob)  / std::pow(p_nonzero_,J);
                // double check_ratio = hyp_prob / check_prob;
                // std::cerr << "hyp: " << hyp_prob << std::endl;
                // std::cerr << "check: " << check_prob << std::endl;
                // std::cerr << "best_c: " << best_c << std::endl;
                // std::cerr << "sub_alignment: " << sub_alignment << std::endl;
                // assert(check_ratio >= 0.99 && check_ratio <= 1.01);
                //END_DEBUG

                best_prob = hyp_prob;
                trace_entry = 0xC0000000;
                uint base = 1;
                for (uint l = 0; l < J; l++) {
                  if (iiset.find(l) != iiset.end())
                    trace_entry += base;
                  base *= 2;
                }
              }
            }

            if (limit_sum >= J && tii_limit >= 2) {

              //if (curJ == 62)
              //  std::cerr << "ext1.2" << std::endl;

              const uint maxCard = J-2;
              Math1D::Vector<double> score[2];
              score[0].resize(maxCard+1,0.0);
              score[1].resize(maxCard+1,0.0);

              Storage2D<bool> trace(maxCard+1,J-2,false);

              score[0][0] = jcost(j+1,ii);
              score[0][1] = jcost(j+1,i);
              trace(1,0) = true;

              uint k=0;
              for (uint jjj = j+2; jjj < jj; jjj++) {

                const Math1D::Vector<double>& prev_score = score[k];
                k = 1-k;
                Math1D::Vector<double>& cur_score = score[k];

                const double hypi_cost = jcost(jjj,i);
                const double hypii_cost = jcost(jjj,ii);

                cur_score[0] = prev_score[0] * hypii_cost;
                for (uint c=1; c < cur_score.size(); c++) {
                  const double scorei = prev_score[c-1] * hypi_cost;
                  const double scoreii = prev_score[c] * hypii_cost;

                  if (scorei > scoreii) {
                    cur_score[c] = scorei;
                    trace(c,jjj-j-1) = true;
                  }
                  else {
                    cur_score[c] = scoreii;
                  }
                }
              }

              const Math1D::Vector<double>& last_score = score[k];

              long double best_ext_prob = 0.0;
              uint best_c = 0;
              //card 0 is not an extension!
              for (uint c = 1; c <= std::min<uint>(maxCard,ti_limit); c++) {
                if (J - c > tii_limit)
                  continue;

                const long double hyp_score = last_score[c] * fert_prob_ti[c] * fert_prob_tii[J-c];

                if (best_ext_prob < hyp_score) {
                  best_ext_prob = hyp_score;
                  best_c = c;
                }
              }

              //if (curJ == 62)
              //  std::cerr << "best_ext_prob: " << best_ext_prob << std::endl;

              long double hyp_prob = best_ext_prob * jcost(j,ii) * jcost(jj,ii);
              if (!no_factorial_) {
                hyp_prob *= ld_fac_[J-best_c];
                hyp_prob *= ld_fac_[best_c];
              }
              hyp_prob *= between_prob;

              if (hyp_prob > best_prob) {

                std::set<uint> iset;
                uint c = best_c;
                int jjj = J-3;

                while (jjj >= 0) {

                  if (trace(c,jjj)) {
                    c--;
                    iset.insert(jjj+1);
                  }
                  jjj--;
                }

                best_prob = hyp_prob;
                trace_entry = 0xC0000000;
                uint base = 1;
                for (uint l = 0; l < J; l++) {
                  if (iset.find(l) == iset.end())
                    trace_entry += base;
                  base *= 2;
                }
              }
            }

          } //end of extended reordering level 1

          if (ext_level >= 2 && I <= 10 && J == 4 && std::min(ti_limit,tii_limit) >= 2) {

            //if (curJ == 62)
            //  std::cerr << "ext2.1" << std::endl;

            long double common_prod = J2fac*J2fac * fert_prob_ti[2] * fert_prob_tii[2];
            for (uint iii = i + 1; iii <= ii - 1; iii++)
              common_prod *= fertility_prob_[cur_target[iii]][0];

            // try alignment i - ii - i - ii
            {
              long double hyp_prob = common_prod * jcost(j,i) * jcost(j+1,ii) * jcost(j+2,i) * jcost(j+3,ii);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;

                trace_entry = 0xC0000000 + 2 + 8;
              }
            }

            // try alignment ii - i - ii - i
            {
              long double hyp_prob = common_prod * jcost(j,ii) * jcost(j+1,i) * jcost(j+2,ii) * jcost(j+3,i);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;

                trace_entry = 0xC0000000 + 1 + 4;
              }
            }
          }

          if (ext_level >= 2 && J >= 4 && I >= 3) { //J = 3 is covered by ext_level 1

            //if (curJ == 62)
            //  std::cerr << "ext2.2" << std::endl;

            const Math2D::TriMatrix<long double>& sub_score = score(J-2,j+1);

            //std::cerr << "check 1" << std::endl;

            //check alignment i - sub(i+1,ii) - i
            if (ti_limit >= 2) {
              long double hyp_prob = J2fac * fert_prob_ti[2] * jcost(j,i) * jcost(jj,i) * sub_score(i+1,ii);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000000;
              }
            }

            //std::cerr << "check 2" << std::endl;

            //check alignment ii - sub(i,ii-1) - ii
            if (tii_limit >= 2) {
              long double hyp_prob = J2fac * fert_prob_tii[2] * jcost(j,ii) * jcost(jj,ii) * sub_score(i,ii-1);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000001;
              }
            }
          }

          if (ext_level >= 2 && J == 4 && I == 4) {

            //if (curJ == 62)
            //  std::cerr << "ext2.3" << std::endl;

            const uint ti1 = cur_target[i+1];
            const uint ti2 = cur_target[i+2];

            const long double fertprob_base = fert_prob_ti[1] * fertility_prob_[ti1][1] * fertility_prob_[ti2][1] * fert_prob_tii[1];

            //the two alignments ITG doesn't cover
            {
              long double hyp_prob = fertprob_base * jcost(j,i+1) * jcost(j+1,ii) * jcost(j+2,i) * jcost(j+3,i+2);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000002;
              }
            }

            {
              long double hyp_prob = fertprob_base * jcost(j,i+2) * jcost(j+1,i) * jcost(j+2,ii) * jcost(j+3,i+1);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000003;
              }
            }
          }

          if (ext_level >= 3 && J >= 5 && I >= 3) { //J = 4 is covered by ext_level 1

            //if (curJ == 62)
            //  std::cerr << "ext3.1" << std::endl;

            const Math2D::TriMatrix<long double>& sub_score1 = score(J-3,j+2);

            //check alignment i - i - sub(i+1,ii) - i
            if (ti_limit >= 3) {
              long double hyp_prob = J3fac * fert_prob_ti[3] * jcost(j,i) * jcost(j+1,i) * jcost(jj,i) * sub_score1(i+1,ii);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000004;
              }
            }

            //check alignment ii - ii - sub(i,ii-1) - ii
            if (tii_limit >= 3) {
              long double hyp_prob = J3fac * fert_prob_tii[3] * jcost(j,ii) * jcost(j+1,ii) * jcost(jj,ii) * sub_score1(i,ii-1);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000005;
              }
            }

            const Math2D::TriMatrix<long double>& sub_score2 = score(J-3,j+1);

            //check alignment i - sub(i+1,ii) - i - i
            if (ti_limit >= 3) {
              long double hyp_prob = J3fac * fert_prob_ti[3] * jcost(j,i) * jcost(jj-1,i) * jcost(jj,i) * sub_score2(i+1,ii);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000006;
              }
            }

            //check alignment ii - sub(i,ii-1) - ii - ii
            if (tii_limit >= 3) {
              long double hyp_prob = J3fac * fert_prob_tii[3] * jcost(j,ii) * jcost(jj-1,ii) * jcost(jj,ii) * sub_score2(i,ii-1);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000007;
              }
            }
          }

          if (ext_level >= 3 && J >= 6 && I >= 3) { //J = 5 is covered by ext_level 1

            //if (curJ == 62)
            //  std::cerr << "ext3.2" << std::endl;

            const Math2D::TriMatrix<long double>& sub_score = score(J-4,j+2);

            //only the two equal splits are tried
            if (ti_limit >= 4) {

              long double hyp_prob = J4fac * fert_prob_ti[4] * jcost(j,i) * jcost(j+1,i) * jcost(jj-1,i) * jcost(jj,i)
                                     * sub_score(i+1,ii);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000008;
              }
            }

            if (tii_limit >= 4) {

              long double hyp_prob = J4fac * fert_prob_tii[4] * jcost(j,ii) * jcost(j+1,ii) * jcost(jj-1,ii) * jcost(jj,ii)
                                     * sub_score(i,ii-1);

              if (hyp_prob > best_prob) {
                best_prob = hyp_prob;
                trace_entry = 0xD0000009;
              }
            }
          }

          if (ext_level >= 4 && J >= 3 && J <= level3_maxlength && I >= 3 && I <= level3_maxlength && std::max(J,I) > 3) {

            //if (curJ == 62)
            //  std::cerr << "ext4" << std::endl;

            //std::cerr << "j: " << j << ", jj: " << jj << ", i: " << i << ", ii: " << ii << std::endl;

            //long double hyp_prob = ibmconstrained_viterbi_subprob_noemptyword(cur_source, cur_target, cur_lookup, j, jj, i, ii);
            long double hyp_prob = level3_score(j,i)(jj-j-1,ii-i-1);

            //DEBUG
            // Storage1D<uint> mini_source(J);
            // Storage1D<uint> mini_target(I);

            // for (uint k=0; k < J; k++)
            // mini_source[k] = cur_source[j+k];
            // for (uint k=0; k < I; k++)
            // mini_target[k] = cur_target[i+k];

            // SingleLookupTable mini_lookup(J,I);
            // Math2D::Matrix<double> distort_prob(J,I);
            // for (uint kj=0; kj < J; kj++) {
            // for (uint ki=0; ki < I; ki++) {
            // mini_lookup(kj,ki) = cur_lookup(kj+j,ki+i);
            // distort_prob(kj,ki) = cur_distort_prob(kj+j,ki+i);
            // }
            // }

            // Math1D::Vector<AlignBaseType> check_alignment(J,0);

            // long double check_prob = compute_ibmconstrained_viterbi_alignment_noemptyword(mini_source, mini_target, mini_lookup,
            // check_alignment, &distort_prob);

            // //std::cerr << "hyp:   " << hyp_prob << std::endl;
            // //std::cerr << "check: " << check_prob << std::endl;

            // assert(check_prob == hyp_prob);
            //END_DEBUG

            if (hyp_prob > best_prob) {
              best_prob = hyp_prob;

              trace_entry = 0xE0000000;
            }
          }

          //if (curJ == 62)
          //  std::cerr << "extensions finished" << std::endl;

          //1.) consider extending the target interval by a zero-fertility word
          const long double left_extend_prob = score(J, j)(i + 1, ii) * fert_prob_ti[0];
          if (left_extend_prob > best_prob) {
            best_prob = left_extend_prob;
            trace_entry = MAX_UINT - 1;
          }
          const long double right_extend_prob = score(J, j)(i, ii - 1) * fert_prob_tii[0];
          if (right_extend_prob > best_prob) {
            best_prob = right_extend_prob;
            trace_entry = MAX_UINT - 2;
          }


          //2.) consider splitting both source and target interval

          const int jmid_point = j + (J / 2);

          const int j_lower = std::max(j,jmid_point-max_mid_dev);
          const int j_upper = std::min(jj,jmid_point+max_mid_dev+1);

          for (int split_j = j_lower; split_j < j_upper; split_j++) {

            //partitioning into [j,split_j] and [split_j+1,jj]

            const uint J1 = split_j - j + 1;
            const uint J2 = jj - split_j;
            assert(J1 >= 1 && J1 < J);
            assert(J2 >= 1 && J2 < J);
            assert(J1 + J2 == J);

            const Math2D::TriMatrix<long double>& score1 = score(J1,j);
            const Math2D::TriMatrix<long double>& score2 = score(J2,split_j+1);

            for (int split_i = i_lower; split_i < i_upper; split_i++) {

              //partitioning into [i,split_i] and [split_i+1,ii]

              //              const uint I1 = split_i - i + 1;
              //              const uint I2 = ii - split_i;
              //              assert(I1 >= 1 && I1 < I);
              //              assert(I2 >= 1 && I2 < I);

              //NOTE: hyp_monotone_prob covers more than just monotone alignments as the subscores can come from non-monotone ones
              const long double hyp_monotone_prob = score1(i, split_i) * score2(split_i + 1, ii);

              if (hyp_monotone_prob > best_prob) {
                best_prob = hyp_monotone_prob;
                trace_entry = 2 * (split_j * curI + split_i);
              }

              const long double hyp_invert_prob = score2(i, split_i) * score1(split_i + 1, ii);

              if (hyp_invert_prob > best_prob) {
                best_prob = hyp_invert_prob;
                trace_entry = 2 * (split_j * curI + split_i) + 1;
              }
            }
          }

          score(J, j)(i, ii) = best_prob;
          trace(J, j)(i, ii) = trace_entry;
        }
      }
    }
  }

  //std::cerr << "calling traceback" << std::endl;

  alignment.resize(curJ);
  alignment.set_constant(0);
  itg_traceback(trace, cur_source, cur_target, cur_lookup, curJ, 0, 0, curI - 1, alignment);

  //std::cerr << "computed alignment: " << alignment << std::endl;

  return score(curJ, 0)(0, curI - 1);
}

void IBM3Trainer::itg_traceback(const NamedStorage2D<Math2D::TriMatrix<uint> >& trace,
                                const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup, uint J,
                                uint j, uint i, uint ii, Math1D::Vector<AlignBaseType>& alignment) const
{
  //std::cerr << "****itg_traceback(" << J << "," << j << "," << i << "," << ii << ")" << std::endl;

  uint trace_entry = trace(J, j)(i, ii);

  //std::cerr << "trace_entry: " << trace_entry << std::endl;

  if (J == 1) {
    alignment[j] = trace_entry + 1;
  }
  else if (i == ii) {
    assert(trace_entry == MAX_UINT);

    for (uint jj = j; jj < j + J; jj++)
      alignment[jj] = i + 1;
  }
  else if (trace_entry == MAX_UINT - 1) {
    itg_traceback(trace, source, target, lookup, J, j, i + 1, ii, alignment);
  }
  else if (trace_entry == MAX_UINT - 2) {
    itg_traceback(trace, source, target, lookup, J, j, i, ii - 1, alignment);
  }
  else if (trace_entry == 0xE0000000) {

    //std::cerr << "IBM traceback for " << J << "," << j << "," << i << "," << ii << std::endl;

    const uint I = ii-i+1;

    Storage1D<uint> mini_source(J);
    Storage1D<uint> mini_target(I);

    for (uint k=0; k < J; k++)
      mini_source[k] = source[j+k];
    for (uint k=0; k < I; k++)
      mini_target[k] = target[i+k];

    SingleLookupTable mini_lookup(J,I);
    const Math3D::Tensor<double>& cur_distort_prob = distortion_prob_[source.size()-1];
    Math3D::Tensor<double> distort_prob(J,I,cur_distort_prob.zDim());
    for (uint kj=0; kj < J; kj++) {
      for (uint ki=0; ki < I; ki++) {
        mini_lookup(kj, ki) = lookup(kj+j, ki+i);
        for (uint c = 0; c < cur_distort_prob.zDim(); c++)
          distort_prob(kj, ki, c) = cur_distort_prob(kj+j, ki+i, c);
      }
    }

    Math1D::Vector<AlignBaseType> mini_alignment(J,0);
    compute_ibmconstrained_viterbi_alignment_noemptyword(mini_source, mini_target, mini_lookup, mini_alignment, &distort_prob);

    for (uint k=0; k < J; k++) {
      alignment[j+k] = mini_alignment[k] + i;
    }
  }
  else if (trace_entry >= 0xD0000000) {

    assert(J >= 4);

    if (trace_entry == 0xD0000000) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside case 1" << std::endl;

      alignment[j] = i+1;
      alignment[j+J-1] = i+1;
      itg_traceback(trace, source, target, lookup, J-2, j+1, i+1, ii, alignment);
    }
    else if (trace_entry == 0xD0000001) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside case 2" << std::endl;

      alignment[j] = ii+1;
      alignment[j+J-1] = ii+1;
      itg_traceback(trace, source, target, lookup, J-2, j+1, i, ii-1, alignment);
    }
    else if (trace_entry == 0xD0000002) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): non-ITG case 1" << std::endl;

      assert(J == 4 && ii == i+3);
      alignment[j] = i+1 + 1;
      alignment[j+1] = ii + 1;
      alignment[j+2] = i + 1;
      alignment[j+3] = i+2 + 1;
    }
    else if (trace_entry == 0xD0000003) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): non-ITG case 2" << std::endl;

      assert(J == 4 && ii == i+3);
      alignment[j] = i+2 + 1;
      alignment[j+1] = i + 1;
      alignment[j+2] = ii + 1;
      alignment[j+3] = i+1 + 1;
    }
    else if (trace_entry == 0xD0000004) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside3 case a" << std::endl;

      alignment[j] = i+1;
      alignment[j+1] = i+1;
      alignment[j+J-1] = i+1;
      itg_traceback(trace, source, target, lookup, J-3, j+2, i+1, ii, alignment);
    }
    else if (trace_entry == 0xD0000005) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside3 case b" << std::endl;

      alignment[j] = ii+1;
      alignment[j+1] = ii+1;
      alignment[j+J-1] = ii+1;
      itg_traceback(trace, source, target, lookup, J-3, j+2, i, ii-1, alignment);
    }
    else if (trace_entry == 0xD0000006) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside3 case c" << std::endl;

      alignment[j] = i+1;
      alignment[j+J-2] = i+1;
      alignment[j+J-1] = i+1;
      itg_traceback(trace, source, target, lookup, J-3, j+1, i+1, ii, alignment);
    }
    else if (trace_entry == 0xD0000007) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside3 case d" << std::endl;

      alignment[j] = ii+1;
      alignment[j+J-2] = ii+1;
      alignment[j+J-1] = ii+1;
      itg_traceback(trace, source, target, lookup, J-3, j+1, i, ii-1, alignment);
    }
    else if (trace_entry == 0xD0000008) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside4 case 1" << std::endl;

      alignment[j] = i+1;
      alignment[j+1] = i+1;
      alignment[j+J-2] = i+1;
      alignment[j+J-1] = i+1;
      itg_traceback(trace, source, target, lookup, J-4, j+2, i+1, ii, alignment);
    }
    else if (trace_entry == 0xD0000009) {

      //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): inside4 case 2" << std::endl;

      alignment[j] = ii+1;
      alignment[j+1] = ii+1;
      alignment[j+J-2] = ii+1;
      alignment[j+J-1] = ii+1;
      itg_traceback(trace, source, target, lookup, J-4, j+2, i, ii-1, alignment);
    }
  }
  else if (trace_entry >= 0xC0000000) {

    //std::cerr << "traceback(" << J << "," << j << "," << i << "," << ii << "): ext-level 1" << std::endl;

    uint temp = trace_entry & 0x3FFFFFF;
    for (uint k = 0; k < J; k++) {

      uint bit = (temp % 2);
      //std::cerr << "k: " << k << ", bit: " << bit << std::endl;

      alignment[j + k] = (bit == 1) ? (ii + 1) : (i + 1);
      temp /= 2;
    }
  }
  else {

    bool reverse = ((trace_entry % 2) == 1);
    trace_entry /= 2;

    uint split_i = trace_entry % trace(J, j).dim();
    uint split_j = trace_entry / trace(J, j).dim();

    //std::cerr << "split_i: " << split_i << ", split_j: " << split_j << ", reverse: " << reverse << std::endl;

    assert(split_i < trace(J, j).dim());
    assert(split_j < trace.yDim());

    const uint J1 = split_j - j + 1;
    const uint J2 = J - J1;

    if (!reverse) {
      itg_traceback(trace, source, target, lookup, J1, j, i, split_i, alignment);
      itg_traceback(trace, source, target, lookup, J2, split_j + 1, split_i + 1, ii, alignment);
    }
    else {
      itg_traceback(trace, source, target, lookup, J2, split_j + 1, i, split_i, alignment);
      itg_traceback(trace, source, target, lookup, J1, j, split_i + 1, ii, alignment);
    }
  }

  //std::cerr << "leaving itg_traceback" << std::endl;
}

long double IBM3Trainer::compute_ibmconstrained_viterbi_alignment_noemptyword(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
    const SingleLookupTable& cur_lookup, Math1D::Vector<AlignBaseType>& alignment,
    const Math3D::Tensor<double>* distort_prob) const
{
  const uint nMaxSkips = uncovered_set_.xDim();
  assert(uncovered_set_.yDim() > 0);

  //std::cerr << "******** compute_ibmconstrained_viterbi_alignment_noemptyword(" << nMaxSkips << ") **********" << std::endl;

  //convention here: target positions start at 0, source positions start at 1
  // (so we can express that no source position was covered yet)

  //NOTE: some alignments are covered multiple times as target words with fertility > 1 can also list source words in nonmotone order
  //   when skiped positions are filled.
  // This is not problematic for maximizing, but for summing a more refined solution would be needed

  const uint curI = cur_target.size();
  const uint curJ = cur_source.size();
  const Math3D::Tensor<double>& cur_distort_prob =  (distort_prob != 0) ? *distort_prob : distortion_prob_[curJ - 1];

  const uint nStates = first_state_[curJ] + 1; //last state is completely uncovered source (and some unaligned targets)

  //std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
  //std::cerr << nStates << " active states" << std::endl;

  NamedStorage1D<Math2D::NamedMatrix<long double> > score(curI, MAKENAME(score));

  NamedStorage1D<Math2D::NamedMatrix<uchar> > state_trace(curI, MAKENAME(state_trace));
  NamedStorage1D<Math1D::NamedVector<uchar> > fert_trace(curI, MAKENAME(fert_trace));

  Math1D::NamedVector<long double> best_prev_score(nStates, 0.0, MAKENAME(best_prev_score));

  Math1D::NamedVector<long double> translation_cost(curJ, MAKENAME(translation_cost));

  const uint t_start = cur_target[0];

  score[0].set_name(MAKENAME(score[0]));
  score[0].resize(nStates-1, std::min<uint>(curJ,fertility_limit_[t_start]) + 1, 0.0);

  const uint start_allfert_max_reachable_j = std::min<uint>(curJ, nMaxSkips + score[0].yDim()-1); //1-based
  const uint start_allfert_max_reachable_state = first_state_[start_allfert_max_reachable_j] - 1; //this is 0-based, uncovered state not reachable

  fert_trace[0].set_name("fert_trace[0]");
  fert_trace[0].resize(start_allfert_max_reachable_state + 1, 255);
  state_trace[0].set_name("state_trace[0]");
  state_trace[0].resize(start_allfert_max_reachable_state + 1, std::min<uint>(curJ,fertility_limit_[t_start]) );

  //initialization for fertility 0 is done directly for best_prev_score below

  //initialize for fertility 1
  for (std::set<uint>::const_iterator it = start_states_.begin(); it != start_states_.end(); it++) {

    uint state = *it;
    if (state >= nStates-1)
      break;

    uint max_covered_j = coverage_state_(1,state);
    score[0](state, 1) = dict_[t_start][cur_lookup(max_covered_j, 0)] * cur_distort_prob(max_covered_j, 0, target_class_[t_start]);
    state_trace[0](state, 0) = max_covered_j;
    //std::cerr << "score " << score[0](state, 1) << " = " << dict_[t_start][cur_lookup(max_covered_j, 0)]
    //          << " * " << cur_distort_prob(max_covered_j, 0) << std::endl;
  }

  //initialize for fertility 2
  for (uint fert = 2; fert <= std::min<uint>(curJ,fertility_limit_[t_start]); fert++) {

    //std::cerr << "fert: " << fert << std::endl;

    const uint curfert_max_reachable_j = std::min(curJ, nMaxSkips + fert); //this is 1-based
    const uint curfert_max_reachable_state = first_state_[curfert_max_reachable_j] - 1; //this is 0-based

    for (uint state = 0; state <= curfert_max_reachable_state; state++) {

      //std::cerr << "state: " << state << std::endl;

      assert(coverage_state_(1, state) < curJ);

      long double best_score = 0.0;
      uchar trace_entry = 255;

      const uint nPredecessors = predecessor_coverage_states_[state].yDim();
      assert(nPredecessors < 255);

      for (uint p = 0; p < nPredecessors; p++) {

        //std::cerr << "p: " << p << std::endl;

        const uint prev_state = predecessor_coverage_states_[state](0, p);
        const uint cover_j = predecessor_coverage_states_[state](1, p); //0-based!

        //std::cerr << "prev_state: " << prev_state << ", cover_j: " << cover_j << std::endl;

        assert(cover_j < curJ);

        const long double hyp_score = score[0](prev_state, fert - 1) * dict_[t_start][cur_lookup(cover_j, 0)] *
                                      cur_distort_prob(cover_j, 0, target_class_[t_start]);

        if (hyp_score > best_score) {
          best_score = hyp_score;
          trace_entry = p;
        }
      }

      score[0](state, fert) = best_score;
      state_trace[0](state, fert - 1) = trace_entry;
    }
  }

  //finally include fertility probabilities
  for (uint fert = 0; fert <= std::min<uint>(curJ,fertility_limit_[t_start]); fert++) {

    //std::cerr << "fert: " << fert << std::endl;

    long double fert_factor = (fertility_prob_[t_start].size() >  fert) ? fertility_prob_[t_start][fert] : 0.0;
    if (fert > 1 && !no_factorial_)
      fert_factor *= ld_fac_[fert];

    if (fert == 0)
      best_prev_score[nStates-1] = fert_factor;
    else {
      for (uint state = 0; state <= start_allfert_max_reachable_state; state++)
        score[0](state, fert) *= fert_factor;
    }
  }

  //std::cerr << "score[0](0,1): " << score[0](0,1) << std::endl;

  //compute fert_trace and best_prev_score
  for (uint state = 0; state <= start_allfert_max_reachable_state; state++) {

    long double best_score = 0.0;
    uchar best_fert = 255;

    for (uint fert = 0; fert <= std::min<uint>(curJ,fertility_limit_[t_start]); fert++) {
      const long double cur_score = score[0](state, fert);

      if (cur_score > best_score) {
        best_score = cur_score;
        best_fert = fert;
      }
    }

    best_prev_score[state] = best_score;
    fert_trace[0][state] = best_fert;
  }
  //nothing to do for all uncovered state

  /**** now proceeed with the remainder of the sentence ****/

  //std::cerr << "initial best_prev_score: " << best_prev_score << std::endl;

  for (uint i = 1; i < curI; i++) {
    //std::cerr << "********* i: " << i << " ***************" << std::endl;

    const uint ti = cur_target[i];
    const uint curMaxFertility = std::min<uint>(curJ,fertility_limit_[ti]);

    const Math1D::Vector<double>& cur_dict = dict_[ti];

    for (uint j = 0; j < curJ; j++)
      translation_cost[j] = cur_dict[cur_lookup(j, i)] * cur_distort_prob(j, i, target_class_[cur_target[i]]);

    const uint allfert_max_reachable_j = std::min(curJ, nMaxSkips + (i + 1) * curMaxFertility); //1-based
    const uint fertone_max_reachable_j = std::min(curJ, nMaxSkips + i * curMaxFertility + 1); //1-based
    const uint prevword_max_reachable_j = std::min(curJ, nMaxSkips + i * curMaxFertility); //1-based

    const uint prevword_max_reachable_state = first_state_[prevword_max_reachable_j] - 1; //this is 0-based
    const uint allfert_max_reachable_state = first_state_[allfert_max_reachable_j] - 1; //this is 0-based
    const uint fertone_max_reachable_state = first_state_[fertone_max_reachable_j] - 1; //this is 0-based

    fert_trace[i].set_name("fert_trace[" + toString(i) + "]");
    fert_trace[i].resize(allfert_max_reachable_state + 1, 255);

    state_trace[i].set_name("state_trace[" + toString(i) + "]");
    state_trace[i].resize(allfert_max_reachable_state + 1, curMaxFertility, 255);

    score[i - 1].resize(0, 0);
    score[i].set_name("score[" + toString(i) + "]");

    Math2D::NamedMatrix<long double>& cur_score = score[i];
    cur_score.resize(nStates-1, curMaxFertility + 1, 0.0);

    Math2D::NamedMatrix<uchar>& cur_state_trace = state_trace[i];

    //     std::cerr << "fert 0" << std::endl;
    //std::cerr << "prevword_max_reachable_state: " << prevword_max_reachable_state << std::endl;
    //std::cerr << "allfert_max_reachable_state: " << allfert_max_reachable_state << std::endl;
    //std::cerr << "fertone_max_reachable_state: " << fertone_max_reachable_state << std::endl;

    //fertility 0
    for (uint state = 0; state <= prevword_max_reachable_state; state++) {
      cur_score(state, 0) = best_prev_score[state];
    }
    //nothing to do for all uncovered state (best_prev_score stays)

    //std::cerr << " fert 1" << std::endl;

    //fertility 1
    for (uint state = 0; state <= fertone_max_reachable_state; state++) {

      //if (coverage_state_(0, state) == 0) {
      //  std::cerr << "state: " << state << std::endl;
      //}

      assert(coverage_state_(1, state) < curJ);

      long double best_score = 0.0;
      uchar trace_entry = 255;

      const uint nPredecessors = predecessor_coverage_states_[state].yDim();
      assert(nPredecessors < 255);

      for (uint p = 0; p < nPredecessors; p++) {

        const uint prev_state = predecessor_coverage_states_[state](0, p);
        const uint cover_j = predecessor_coverage_states_[state](1, p);

        //if (coverage_state_(0, state) == 0) {
        //  std::cerr << "predecessor " << p << ", prev_state = " << prev_state << ", cover_j: " << cover_j << std::endl;
        //  std::cerr << "hyp_score = " << best_prev_score[prev_state] << " * " << translation_cost[cover_j] << std::endl;
        //}

        assert(cover_j < curJ);

        const long double hyp_score = best_prev_score[prev_state] * translation_cost[cover_j];

        if (hyp_score > best_score) {
          best_score = hyp_score;
          trace_entry = p;
        }
      }

      //transition from uncovered state
      if (state < first_state_[nMaxSkips+1] && start_states_.find(state) != start_states_.end()) {
        const long double hyp_score = best_prev_score[nStates-1] * translation_cost[coverage_state_(1,state)];
        if (hyp_score > best_score) {
          best_score = hyp_score;
          trace_entry = 255;
        }
      }

      cur_score(state, 1) = best_score;
      cur_state_trace(state, 0) = trace_entry;
    }

    //fertility > 1
    for (uint fert = 2; fert <= curMaxFertility; fert++) {

      const uint curfert_max_reachable_j = std::min(curJ, nMaxSkips + i * curMaxFertility + fert); //this is 1-based
      const uint curfert_max_reachable_state = first_state_[curfert_max_reachable_j] - 1; //this is 0-based

      //std::cerr << "fert: " << fert << std::endl;

      for (uint state = 0; state <= curfert_max_reachable_state; state++) {

        assert(coverage_state_(1, state) <= curJ);

        long double best_score = 0.0;
        uchar trace_entry = 255;

        const uint nPredecessors = predecessor_coverage_states_[state].yDim();
        assert(nPredecessors < 255);

        for (uint p = 0; p < nPredecessors; p++) {

          const uint prev_state = predecessor_coverage_states_[state](0, p);
          const uint cover_j = predecessor_coverage_states_[state](1, p); //0-based!

          assert(cover_j < curJ);

          const long double hyp_score = cur_score(prev_state, fert - 1) * translation_cost[cover_j];

          if (hyp_score > best_score) {
            best_score = hyp_score;
            trace_entry = p;
          }
        }

        cur_score(state, fert) = best_score;
        cur_state_trace(state, fert - 1) = trace_entry;
      }
    }

    //std::cerr << "including fertility probs" << std::endl;

    //include fertility probabilities
    for (uint fert = 0; fert <= curMaxFertility; fert++) {
      long double fert_factor = (fertility_prob_[ti].size() > fert) ? fertility_prob_[ti][fert] : 0.0;
      if (fert > 1 && !no_factorial_)
        fert_factor *= ld_fac_[fert];

      if (fert == 0)
        best_prev_score[nStates-1] *= fert_factor;
      for (uint state = 0; state <= allfert_max_reachable_state; state++) {
        cur_score(state, fert) *= fert_factor;
        // if (fert == 1 && coverage_state_(0, state) == 0) {
        // std::cerr << "cur_score(#" << state << " = no skips, " << coverage_state_(1, state) << ", 1): "
        // << cur_score(state, fert) << std::endl;
        // }
      }
    }

    //DEBUG
    //best_prev_score.set_constant(0.0);
    //END_DEBUG

    //compute fert_trace and best_prev_score
    for (uint state = 0; state <= allfert_max_reachable_state; state++) {

      long double best_score = 0.0;
      uchar best_fert = 255;

      for (uint fert = 0; fert <= curMaxFertility; fert++) {
        const long double cand_score = cur_score(state, fert);

        if (cand_score > best_score) {
          best_score = cand_score;
          best_fert = fert;
        }
      }

      best_prev_score[state] = best_score;
      fert_trace[i][state] = best_fert;
    }
    //nothing to do for all uncovered state

    //std::cerr << "new best_prev_score: " << best_prev_score << std::endl;
  }

  const uint end_state = first_state_[curJ-1];
  assert(coverage_state_(0, end_state) == 0);
  assert(coverage_state_(1, end_state) == curJ-1);
  const long double best_prob = best_prev_score[end_state];
  const ushort best_end_fert = fert_trace[curI - 1][end_state];

  /**** traceback ****/
  alignment.resize(curJ);
  alignment.set_constant(0);

  uint fert = best_end_fert;
  uint i = curI - 1;
  uint state = end_state;

  while (true) {

    if (state == nStates-1) {
      //std::cerr << "reached all unaligned state" << std::endl;
      //all previous target words are unaligned
      break;
    }

    // std::cerr << "**** traceback: i=" << i << ", fert=" << fert << ", state #" << state
    // << ", uncovered set=";
    // print_uncovered_set(coverage_state_(0,state));
    // std::cerr << " ; max_covered_j=" << coverage_state_(1,state) << std::endl;

    //std::cerr << "score: " << score[i](state,fert) << std::endl;

    //default values apply to the case with fertility 0
    uint prev_state = state;

    if (fert > 0) {

      const uchar transition = state_trace[i](state, fert - 1);

      if (i > 0 || fert > 1) {

        prev_state = (transition == 255) ? nStates-1 : predecessor_coverage_states_[state](0, transition);
        if (prev_state == nStates-1) {
          //std::cerr << "previous state is all unaligned" << std::endl;
          alignment[coverage_state_(1,state)] = i + 1;
          break;
        }

        uint covered_j = predecessor_coverage_states_[state](1, transition);
        alignment[covered_j] = i + 1;

        //std::cerr << "transition: " << uint(transition) << ", covered_j: " << covered_j << ", prev_state: " << prev_state << std::endl;
      }
      if (i == 0 && fert == 1)
        alignment[transition] = i + 1;
    }

    if (i == 0 && fert <= 1)
      break;

    //default value applies to the case with fertility > 1

    if (fert <= 1) {
      i--;
      fert = fert_trace[i][prev_state];
    }
    else
      fert--;

    state = prev_state;
  }

  //std::cerr << "computed alignment: " << best_known_alignment_[s] << std::endl;
  //std::cerr << "computed best prob (without null fert): " << best_prob << std::endl;

  assert(alignment.min() > 0);

  //DEBUG
  // if (curJ == curI) {

  // long double mono_prob = 1.0;
  // for (uint j=0; j < curJ; j++) {
  // mono_prob *= dict_[cur_target[j]][cur_lookup(j, j)] * cur_distort_prob(j, j) *  fertility_prob_[cur_target[j]][1];
  // //std::cerr << "mono prob after j=" << j << ": " << mono_prob << std::endl;
  // }

  // //std::cerr << "prob of monotone alignment: " << mono_prob << std::endl;

  // assert(best_prob > 0.99 * mono_prob);
  // }
  //END_DEBUG

  return best_prob;
}

long double IBM3Trainer::ibmconstrained_viterbi_subprob_noemptyword(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
    const SingleLookupTable& cur_lookup, uint j1, uint j2, uint i1, uint i2,
    Math2D::Matrix<long double>* scoremat) const
{
  assert(j1 < j2);
  assert(i1 < i2);
  assert(j2 < cur_source.size());
  assert(i2 < cur_target.size());

  const uint nMaxSkips = uncovered_set_.xDim();
  assert(uncovered_set_.yDim() > 0);

  const uint curI = i2-i1+1;
  const uint curJ = j2-j1+1;
  const Math3D::Tensor<double>& cur_distort_prob = distortion_prob_[cur_source.size() - 1];

  if (scoremat != 0) {
    scoremat->resize(curJ-1,curI-1); // length 1 is not to be reported
    scoremat->set_constant(0.0);
  }

  const uint nStates = first_state_[curJ] + 1; //last state is completely uncovered source (and some unaligned targets)

  Storage1D<Math2D::Matrix<long double> > score(curI);

  Math1D::NamedVector<long double> best_prev_score(nStates, 0.0, MAKENAME(best_prev_score));

  Math1D::Vector<long double> translation_cost(curJ+1);

  const uint t_start = cur_target[i1];

  score[0].resize(nStates-1, std::min<uint>(curJ,fertility_limit_[t_start]) + 1, 0.0);

  const uint start_allfert_max_reachable_j = std::min<uint>(curJ, nMaxSkips + score[0].yDim()-1); //1-based
  const uint start_allfert_max_reachable_state = first_state_[start_allfert_max_reachable_j] - 1; //this is 0-based, uncovered state not reachable

  //initialization for fertility 0 is done directly for best_prev_score below

  //initialize for fertility 1
  for (std::set<uint>::const_iterator it = start_states_.begin(); it != start_states_.end(); it++) {

    uint state = *it;
    if (state >= nStates-1)
      break;

    uint max_covered_j = coverage_state_(1,state) + j1;
    score[0](state, 1) = dict_[t_start][cur_lookup(max_covered_j, i1)] * cur_distort_prob(max_covered_j, i1, target_class_[t_start]);
    //std::cerr << "score " << score[0](state, 1) << " = " << dict_[t_start][cur_lookup(max_covered_j, i1, target_class_[t_start])]
    //          << " * " << cur_distort_prob(max_covered_j, i1) << std::endl;
  }

  //initialize for fertility 2
  for (uint fert = 2; fert <= std::min<uint>(curJ,fertility_limit_[t_start]); fert++) {

    //std::cerr << "fert: " << fert << std::endl;

    const uint curfert_max_reachable_j = std::min(curJ, nMaxSkips + fert); //this is 1-based
    const uint curfert_max_reachable_state = first_state_[curfert_max_reachable_j] - 1; //this is 0-based

    for (uint state = 0; state <= curfert_max_reachable_state; state++) {

      //std::cerr << "state: " << state << std::endl;

      assert(coverage_state_(1, state) < cur_source.size());

      long double best_score = 0.0;

      const uint nPredecessors = predecessor_coverage_states_[state].yDim();
      assert(nPredecessors < 255);

      for (uint p = 0; p < nPredecessors; p++) {

        //std::cerr << "p: " << p << std::endl;

        const uint prev_state = predecessor_coverage_states_[state](0, p);
        const uint cover_j = predecessor_coverage_states_[state](1, p) + j1; //0-based!

        //std::cerr << "prev_state: " << prev_state << ", cover_j: " << cover_j << std::endl;

        assert(cover_j < cur_source.size());

        const long double hyp_score = score[0](prev_state, fert - 1) * dict_[t_start][cur_lookup(cover_j, i1)] *
                                      cur_distort_prob(cover_j, i1, target_class_[t_start]);

        if (hyp_score > best_score) {
          best_score = hyp_score;
        }
      }

      score[0](state, fert) = best_score;
    }
  }

  //finally include fertility probabilities
  for (uint fert = 0; fert <= std::min<uint>(curJ,fertility_limit_[t_start]); fert++) {

    //std::cerr << "fert: " << fert << std::endl;

    long double fert_factor = (fertility_prob_[t_start].size() > fert) ? fertility_prob_[t_start][fert] : 0.0;
    if (fert > 1 && !no_factorial_)
      fert_factor *= ld_fac_[fert];

    if (fert == 0)
      best_prev_score[nStates-1] = fert_factor;
    else {
      for (uint state = 0; state <= start_allfert_max_reachable_state; state++)
        score[0](state, fert) *= fert_factor;
    }
  }

  //compute best_prev_score
  for (uint state = 0; state <= start_allfert_max_reachable_state; state++) {

    long double best_score = 0.0;

    for (uint fert = 0; fert <= std::min<uint>(curJ,fertility_limit_[t_start]); fert++) {

      best_score = std::max(best_score, score[0](state, fert));
    }

    best_prev_score[state] = best_score;
  }
  //nothing to do for all uncovered state

  /**** now proceeed with the remainder of the sentence ****/

  //std::cerr << "initial best_prev_score: " << best_prev_score << std::endl;

  for (uint i = 1; i < curI; i++) {
    //std::cerr << "********* i: " << i << " ***************" << std::endl;

    const uint ti = cur_target[i1+i];
    const uint curMaxFertility = std::min<uint>(curJ,fertility_limit_[ti]);

    const Math1D::Vector<double>& cur_dict = dict_[ti];

    for (uint j = 0; j < curJ; j++)
      translation_cost[j] = cur_dict[cur_lookup(j+j1, i1+i)] * cur_distort_prob(j+j1, i1+i, target_class_[ti]);

    const uint allfert_max_reachable_j = std::min(curJ, nMaxSkips + (i + 1) * curMaxFertility); //1-based
    const uint fertone_max_reachable_j = std::min(curJ, nMaxSkips + i * curMaxFertility + 1); //1-based
    const uint prevword_max_reachable_j = std::min(curJ, nMaxSkips + i * curMaxFertility); //1-based

    const uint prevword_max_reachable_state = first_state_[prevword_max_reachable_j] - 1; //this is 0-based
    const uint allfert_max_reachable_state = first_state_[allfert_max_reachable_j] - 1; //this is 0-based
    const uint fertone_max_reachable_state = first_state_[fertone_max_reachable_j] - 1; //this is 0-based

    score[i - 1].resize(0, 0);

    Math2D::Matrix<long double>& cur_score = score[i];
    cur_score.resize(nStates-1, curMaxFertility + 1, 0.0);

    //     std::cerr << "fert 0" << std::endl;
    //std::cerr << "prevword_max_reachable_state: " << prevword_max_reachable_state << std::endl;
    //std::cerr << "allfert_max_reachable_state: " << allfert_max_reachable_state << std::endl;
    //std::cerr << "fertone_max_reachable_state: " << fertone_max_reachable_state << std::endl;

    //fertility 0
    for (uint state = 0; state <= prevword_max_reachable_state; state++) {
      cur_score(state, 0) = best_prev_score[state];
    }
    //nothing to do for all uncovered state (best_prev_score stays)

    //std::cerr << " fert 1" << std::endl;

    //fertility 1
    for (uint state = 0; state <= fertone_max_reachable_state; state++) {

      //if (coverage_state_(0, state) == 0) {
      //  std::cerr << "state: " << state << std::endl;
      //}

      assert(coverage_state_(1, state) < cur_source.size());

      long double best_score = 0.0;

      const uint nPredecessors = predecessor_coverage_states_[state].yDim();
      assert(nPredecessors < 255);

      for (uint p = 0; p < nPredecessors; p++) {

        const uint prev_state = predecessor_coverage_states_[state](0, p);
        const uint cover_j = predecessor_coverage_states_[state](1, p);

        //if (coverage_state_(0, state) == 0) {
        //  std::cerr << "predecessor " << p << ", prev_state = " << prev_state << ", cover_j: " << cover_j << std::endl;
        //  std::cerr << "hyp_score = " << best_prev_score[prev_state] << " * " << translation_cost[cover_j] << std::endl;
        //}

        assert(cover_j < cur_source.size());

        const long double hyp_score = best_prev_score[prev_state] * translation_cost[cover_j];

        if (hyp_score > best_score) {
          best_score = hyp_score;
        }
      }

      //transition from uncovered state
      if (state < first_state_[nMaxSkips+1] && start_states_.find(state) != start_states_.end()) {
        const long double hyp_score = best_prev_score[nStates-1] * translation_cost[coverage_state_(1,state)];
        if (hyp_score > best_score) {
          best_score = hyp_score;
        }
      }

      cur_score(state, 1) = best_score;
    }

    //fertility > 1
    for (uint fert = 2; fert <= curMaxFertility; fert++) {

      const uint curfert_max_reachable_j = std::min(curJ, nMaxSkips + i * curMaxFertility + fert); //this is 1-based
      const uint curfert_max_reachable_state = first_state_[curfert_max_reachable_j] - 1; //this is 0-based

      //std::cerr << "fert: " << fert << std::endl;

      for (uint state = 0; state <= curfert_max_reachable_state; state++) {

        assert(coverage_state_(1, state) <= cur_source.size());

        long double best_score = 0.0;

        const uint nPredecessors = predecessor_coverage_states_[state].yDim();
        assert(nPredecessors < 255);

        for (uint p = 0; p < nPredecessors; p++) {

          const uint prev_state = predecessor_coverage_states_[state](0, p);
          const uint cover_j = predecessor_coverage_states_[state](1, p); //0-based!

          assert(cover_j < cur_source.size());

          const long double hyp_score = cur_score(prev_state, fert - 1) * translation_cost[cover_j];

          if (hyp_score > best_score) {
            best_score = hyp_score;
          }
        }

        cur_score(state, fert) = best_score;
      }
    }

    //include fertility probabilities
    for (uint fert = 0; fert <= curMaxFertility; fert++) {
      long double fert_factor = (fertility_prob_[ti].size() > fert) ? fertility_prob_[ti][fert] : 0.0;
      if (fert > 1 && !no_factorial_)
        fert_factor *= ld_fac_[fert];

      if (fert == 0)
        best_prev_score[nStates-1] *= fert_factor;
      for (uint state = 0; state <= allfert_max_reachable_state; state++) {
        cur_score(state, fert) *= fert_factor;
      }
    }

    //compute best_prev_score
    for (uint state = 0; state <= allfert_max_reachable_state; state++) {

      long double best_score = 0.0;

      for (uint fert = 0; fert <= curMaxFertility; fert++) {
        best_score = std::max(best_score,cur_score(state, fert));
      }

      best_prev_score[state] = best_score;
    }
    //nothing to do for all uncovered state

    if (scoremat != 0) {
      for (uint j=1; j < curJ; j++) {
        (*scoremat)(j-1,i-1) = best_prev_score[first_state_[j]];
      }
    }

    //std::cerr << "new best_prev_score: " << best_prev_score << std::endl;
  }

  const uint end_state = first_state_[curJ-1];
  assert(coverage_state_(0, end_state) == 0);
  assert(coverage_state_(1, end_state) == curJ-1);
  return best_prev_score[end_state];
}

void add_nondef_count_compact(const Storage1D<std::vector<uchar> >& aligned_source_words, uint i, uint c, uint J, uint maxJ,
                              double count, std::map<Math1D::Vector<uchar,uchar>, double>& count_map, Math3D::Tensor<double>& par_count)
{
  //note: i starts at zero, but the index for aligned_source_words starts at 1

  const std::vector<uchar>& cur_aligned = aligned_source_words[i + 1];

#ifndef NDEBUG
  if (cur_aligned.empty()) {
    std::cerr << "WARNING: useless call!" << std::endl;
    return;
  }
#endif

  Storage1D<bool> fixed(J, false);
  for (uint ii = 0; ii < i; ii++) {
    const std::vector<uchar>& words = aligned_source_words[ii + 1];
    for (uint k = 0; k < words.size(); k++)
      fixed[words[k]] = true;
  }

  uint prev_j = MAX_UINT;

  for (uint k = 0; k < cur_aligned.size(); k++) {

    uint j = cur_aligned[k];

    uint nFollowing = cur_aligned.size() - k - 1;

    const uint start = (prev_j == MAX_UINT) ? 0 : prev_j + 1;

    std::vector<uchar> possible;
    possible.reserve(J);
    for (uint jj = start; jj < J; jj++) {
      if (!fixed[jj]) {
        possible.push_back(jj);
      }
    }

    if (nFollowing > 0)
      possible.resize(possible.size() - nFollowing);

    //NOTE: presently we have to include entries with only one position.
    // Preferably, we should remove such positions from fpar_distort_count

    assert(possible.size() > 0);

    if (possible.size() == 1) {
      //if (false) {
      //terms with only one position always yield probability one.
      //in order to skip them we must remove their contribution from par_count

      //NOTE: in practice we DO get different results when not including these terms.
      // Even the initial energies are different (usually higher). As far as I can tell this must be due to numerical error accumulation
      par_count(possible[0], i, c) -= count;
    }
    else if (possible.size() < maxJ) {          //the normalization term can be skipped if ALL positions are free
      Math1D::Vector<uchar,uchar> vec_possible(possible.size());
      assign(vec_possible, possible);

      assert(vec_possible.size() > 0);

      count_map[vec_possible] += count;

      //DEBUG
      for (std::map<Math1D::Vector<uchar,uchar>,double >::iterator it = count_map.begin(); it != count_map.end(); it++)
        assert(it->first.size() > 0);
      //END_DEBUG

      fixed[j] = true;
      prev_j = j;
    }
  }
}

//for diffpar
void add_nondef_count_compact_diffpar(const Storage1D<std::vector<uchar> >& aligned_source_words,
                                      const Storage1D<WordClassType>& tclass, uint J, uint offset, double count,
                                      Storage1D<std::map<Math1D::Vector<ushort,uchar>, double> >& count_map, Math3D::Tensor<double>& par_count)
{
  Storage1D<bool> fixed(J, false);

  for (uint i = 0; i < aligned_source_words.size() - 1; i++) {

    const uint c = tclass[i];
    std::map<Math1D::Vector<ushort,uchar>, double>& cur_count_map = count_map[c];

    const std::vector<uchar>& cur_aligned = aligned_source_words[i + 1];

    uint prev_j = MAX_UINT;

    for (uint k = 0; k < cur_aligned.size(); k++) {

      uint j = cur_aligned[k];

      uint nFollowing = cur_aligned.size() - k - 1;

      const uint start = (prev_j == MAX_UINT) ? 0 : prev_j + 1;

      std::vector<ushort> possible;
      possible.reserve(J);
      for (uint jj = start; jj < J; jj++) {
        if (!fixed[jj]) {
          possible.push_back(offset + jj - i);
        }
      }

      if (nFollowing > 0)
        possible.resize(possible.size() - nFollowing);

      //NOTE: presently we have to include entries with only one position.
      // Preferably, we should remove such positions from fpar_distort_count

      assert(possible.size() > 0);

      if (possible.size() == 1) {
        //if (false) {
        //terms with only one position always yield probability one.
        //in order to skip them we must remove their contribution from par_count

        //NOTE: in practice we DO get different results when not including these terms.
        // Even the initial energies are different (usually higher). As far as I can tell this must be due to numerical error accumulation
        par_count(possible[0], 0, c) -= count;
      }
      else {
        Math1D::Vector<ushort,uchar> vec_possible(possible.size());
        assign(vec_possible, possible);

        assert(vec_possible.size() > 0);

        cur_count_map[vec_possible] += count;

        //DEBUG
        for (std::map<Math1D::Vector<ushort,uchar>,double>::const_iterator it = cur_count_map.begin(); it != cur_count_map.end(); it++)
          assert(it->first.size() > 0);
        //END_DEBUG

        fixed[j] = true;
        prev_j = j;
      }
    }
  }
}
