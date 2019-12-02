/*** ported here from singleword_fertility_training ****/
/** author: Thomas Schoenemann. This file was generated while Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012
 ** and greatly extended since as a private person ***/

#include "ibm4_training.hh"

#include "timing.hh"
#include "projection.hh"
#include "training_common.hh"   // for get_wordlookup(), dictionary and start-prob m-step
#include "stl_util.hh"
#include "storage_stl_interface.hh"
#include "storage_util.hh"

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"

IBM4CacheStruct::IBM4CacheStruct(uchar j, WordClassType sc, WordClassType tc):j_(j), sclass_(sc), tclass_(tc)
{
}

bool operator<(const IBM4CacheStruct& c1, const IBM4CacheStruct& c2)
{
  if (c1.j_ != c2.j_)
    return (c1.j_ < c2.j_);
  if (c1.sclass_ != c2.sclass_)
    return (c1.sclass_ < c2.sclass_);

  return c1.tclass_ < c2.tclass_;
}

IBM4Trainer::IBM4Trainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                         const Storage1D<Math1D::Vector<uint> >& target_sentence,
                         const std::map<uint, std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                         const std::map<uint, std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                         SingleWordDictionary& dict, const CooccuringWordsType& wcooc,
                         const Math1D::Vector<uint>& tfert_class, uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
                         const Storage1D<WordClassType>& source_class, const Storage1D<WordClassType>& target_class,
                         const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
                         const FertModelOptions& options, bool no_factorial)
  : FertilityModelTrainer(source_sentence, slookup, target_sentence, dict, wcooc, tfert_class, nSourceWords, nTargetWords, prior_weight,
                          sure_ref_alignments, possible_ref_alignments, log_table, xlogx_table, options, no_factorial),
    cept_start_prob_(MAKENAME(cept_start_prob_)), within_cept_prob_(MAKENAME(within_cept_prob_)),
    sentence_start_parameters_(MAKENAME(sentence_start_parameters_)), source_class_(source_class), target_class_(target_class),
    cept_start_mode_(options.cept_start_mode_), inter_dist_mode_(options.inter_dist_mode_),
    intra_dist_mode_(options.intra_dist_mode_), use_sentence_start_prob_(!options.uniform_sentence_start_prob_),
    reduce_deficiency_(options.reduce_deficiency_), uniform_intra_prob_(options.uniform_intra_prob_), min_nondef_count_(options.min_nondef_count_),
    dist_m_step_iter_(options.dist_m_step_iter_), nondef_dist_m_step_iter_(options.nondef_dist34_m_step_iter_),
    nondeficient_(options.nondeficient_), storage_limit_(12)
{
  const uint nDisplacements = 2 * maxJ_ - 1;
  displacement_offset_ = maxJ_ - 1;

  inter_distortion_cache_.resize(maxJ_ + 1);

  uint max_source_class = 0;
  uint min_source_class = MAX_UINT;
  for (uint j = 1; j < source_class_.size(); j++) {
    max_source_class = std::max<uint>(max_source_class, source_class_[j]);
    min_source_class = std::min<uint>(min_source_class, source_class_[j]);
  }
  if (min_source_class > 0) {
    for (uint j = 1; j < source_class_.size(); j++)
      source_class_[j] -= min_source_class;
    max_source_class -= min_source_class;
  }

  uint max_target_class = 0;
  uint min_target_class = MAX_UINT;
  for (uint i = 1; i < target_class_.size(); i++) {
    max_target_class = std::max<uint>(max_target_class, target_class_[i]);
    min_target_class = std::min<uint>(min_target_class, target_class_[i]);
  }
  if (min_target_class > 0) {
    for (uint i = 1; i < target_class_.size(); i++)
      target_class_[i] -= min_target_class;
    max_target_class -= min_target_class;
  }

  nSourceClasses_ = max_source_class + 1;
  nTargetClasses_ = max_target_class + 1;

  cept_start_prob_.resize(nSourceClasses_, nTargetClasses_, 2 * maxJ_ - 1);
  if (intra_dist_mode_ == IBM4IntraDistModeTarget)
    within_cept_prob_.resize(nTargetClasses_, maxJ_);
  else
    within_cept_prob_.resize(nSourceClasses_, maxJ_);

  cept_start_prob_.set_constant(1.0 / nDisplacements);

  within_cept_prob_.set_constant(1.0 / (maxJ_ - 1));
  for (uint x = 0; x < within_cept_prob_.xDim(); x++)
    within_cept_prob_(x, 0) = 0.0;

  if (use_sentence_start_prob_) {
    sentence_start_parameters_.resize(maxJ_, 1.0 / maxJ_);
    sentence_start_prob_.resize(maxJ_ + 1);
  }

  std::set<uint> seenJs;
  for (uint s = 0; s < source_sentence.size(); s++)
    seenJs.insert(source_sentence[s].size());

  inter_distortion_prob_.resize(maxJ_ + 1);
  intra_distortion_prob_.resize(maxJ_ + 1);

  if (use_sentence_start_prob_) {
    for (uint J = 1; J <= maxJ_; J++) {
      if (seenJs.find(J) != seenJs.end()) {

        sentence_start_prob_[J].resize(J, 0.0);

        for (uint j = 0; j < J; j++)
          sentence_start_prob_[J][j] = sentence_start_parameters_[j];
      }
    }
  }

  null_intra_prob_.resize(maxJ_, 1.0 / (maxJ_ - 1));    //entry 0 will not be accessed

  for (uint s = 0; s < source_sentence_.size(); s++) {

    const uint curJ = source_sentence_[s].size();
    const uint curI = target_sentence_[s].size();

    uint max_t = 0;

    for (uint i = 0; i < curI; i++) {
      const uint tclass = target_class_[target_sentence_[s][i]];
      max_t = std::max(max_t, tclass);
    }

    uint max_s = 0;

    for (uint j = 0; j < curJ; j++) {
      const uint sclass = source_class_[source_sentence_[s][j]];
      max_s = std::max(max_s, sclass);
    }

    if (reduce_deficiency_ && !nondeficient_) {
      if (inter_distortion_prob_[curJ].xDim() < max_s + 1 || inter_distortion_prob_[curJ].yDim() < max_t + 1)
        inter_distortion_prob_[curJ].resize(std::max<uint>(inter_distortion_prob_[curJ].xDim(), max_s + 1),
                                            std::max<uint>(inter_distortion_prob_[curJ].yDim(), max_t + 1));
    }

    if (intra_dist_mode_ == IBM4IntraDistModeTarget) {

      if (intra_distortion_prob_[curJ].xDim() < max_t + 1)
        intra_distortion_prob_[curJ].resize_dirty(max_t + 1, curJ, curJ);
    }
    else {

      if (intra_distortion_prob_[curJ].xDim() < max_s + 1)
        intra_distortion_prob_[curJ].resize_dirty(max_s + 1, curJ, curJ);
    }
  }

  for (std::set<uint>::const_iterator it = seenJs.begin(); it != seenJs.end(); it++) {
    const uint J = *it;

    for (int j1 = 0; j1 < (int)J; j1++) {
      for (int j2 = j1 + 1; j2 < (int)J; j2++) {

        for (uint y = 0; y < intra_distortion_prob_[J].xDim(); y++) {
          intra_distortion_prob_[J](y, j2, j1) = within_cept_prob_(y, j2 - j1);
        }
      }
    }

    if (reduce_deficiency_) {

      if (J <= storage_limit_ || nSourceClasses_ * nTargetClasses_ <= 10) {

        for (uint sclass = 0; sclass < inter_distortion_prob_[J].xDim(); sclass++) {
          for (uint tclass = 0; tclass < inter_distortion_prob_[J].yDim(); tclass++) {

            if (inter_distortion_prob_[J](sclass, tclass).size() == 0) {
              inter_distortion_prob_[J](sclass, tclass).resize(J, J);
            }
          }
        }
      }
    }
  }

  //EXPERIMENTAL - expand SOME of the inter distortion matrices
  if (reduce_deficiency_ && !nondeficient_  && nSourceClasses_ * nTargetClasses_ > 10) {

    Storage1D<Math2D::Matrix<uint> > combi_count(inter_distortion_prob_.size());

    for (uint J = 1; J < combi_count.size(); J++) {

      combi_count[J].resize(nSourceClasses_, nTargetClasses_, 0);
    }

    for (uint s = 0; s < source_sentence_.size(); s++) {

      const uint curJ = source_sentence_[s].size();
      const uint curI = target_sentence_[s].size();

      for (uint i = 0; i < curI; i++) {
        const uint tclass = target_class_[target_sentence_[s][i]];

        for (uint j = 0; j < curJ; j++) {
          const uint sclass = source_class_[source_sentence_[s][j]];

          combi_count[curJ] (sclass, tclass)++;
        }
      }
    }

    uint nExpanded = 0;

    for (uint J = storage_limit_ + 1; J < combi_count.size(); J++) {

      for (uint x = 0; x < combi_count[J].xDim(); x++) {
        for (uint y = 0; y < combi_count[J].yDim(); y++) {

          if (combi_count[J](x, y) >= 1.2 * J * J) {

            nExpanded++;

            if (inter_distortion_prob_[J].xDim() <= x || inter_distortion_prob_[J].yDim() <= y)
              inter_distortion_prob_[J].resize(nSourceClasses_, nTargetClasses_);

            inter_distortion_prob_[J](x, y).resize(J, J, 0.0);
          }
        }
      }
    }

    std::cerr << "expanded " << nExpanded << " inter probs." << std::endl;
  }
//END_EXPERIMENTAL

  if (reduce_deficiency_)
    par2nonpar_inter_distortion();
}

/*virtual*/ std::string IBM4Trainer::model_name() const
{
  return "IBM-4";
}

/*virtual*/ void IBM4Trainer::release_memory()
{
  FertilityModelTrainer::release_memory();
  sentence_start_prob_.resize(0);

  inter_distortion_prob_.resize(0);
  intra_distortion_prob_.resize(0);

  inter_distortion_cache_.resize(0);
}

const Math1D::Vector<double>& IBM4Trainer::sentence_start_parameters() const
{
  return sentence_start_parameters_;
}

double IBM4Trainer::inter_distortion_prob(int j, int j_prev, uint sclass, uint tclass, uint J) const
{
  if (!reduce_deficiency_)
    return cept_start_prob_(sclass, tclass, j - j_prev + displacement_offset_);

  assert(inter_distortion_prob_[J].xDim() >= sclass && inter_distortion_prob_[J].yDim() >= tclass);

  if (inter_distortion_prob_[J].size() > 0 && inter_distortion_prob_[J] (sclass, tclass).size() > 0)
    return inter_distortion_prob_[J] (sclass, tclass) (j, j_prev);

  IBM4CacheStruct cs(j, sclass, tclass);

  const std::map<IBM4CacheStruct,float>::const_iterator it =  inter_distortion_cache_[J][j_prev].find(cs);

  if (it == inter_distortion_cache_[J][j_prev].end()) {

    double sum = 0.0;

    for (int jj = 0; jj < int (J); jj++) {
      sum += cept_start_prob_(sclass, tclass, jj - j_prev + displacement_offset_);
      assert(!isnan(sum));
    }

    float prob = std::max(1e-8, cept_start_prob_(sclass, tclass, j - j_prev + displacement_offset_) / sum);
    inter_distortion_cache_[J][j_prev][cs] = prob;
    return prob;
  }
  else
    return it->second;
}

void IBM4Trainer::par2nonpar_inter_distortion()
{
  for (int J = 1; J <= (int)maxJ_; J++) {

    if (inter_distortion_prob_[J].size() > 0) {

      for (uint x = 0; x < inter_distortion_prob_[J].xDim(); x++) {
        for (uint y = 0; y < inter_distortion_prob_[J].yDim(); y++) {

          if (inter_distortion_prob_[J] (x, y).size() > 0) {

            assert(inter_distortion_prob_[J] (x, y).xDim() == uint(J)
                   && inter_distortion_prob_[J] (x, y).yDim() == uint(J));

            if (reduce_deficiency_) {

              for (int j1 = 0; j1 < J; j1++) {

                double sum = 0.0;

                for (int j2 = 0; j2 < J; j2++) {
                  sum += cept_start_prob_(x, y, j2 - j1 + displacement_offset_);
                  assert(!isnan(sum));
                }

                if (sum > 1e-305) {
                  for (int j2 = 0; j2 < J; j2++) {
                    inter_distortion_prob_[J] (x, y) (j2, j1) =
                      std::max(fert_min_param_entry, cept_start_prob_(x, y, j2 - j1 + displacement_offset_) / sum);
                  }
                }
                else if (j1 > 0) {
                  //std::cerr << "WARNING: sum too small for inter prob " << j1 << ", not updating." << std::endl;
                }
              }
            }
            else {
              for (int j1 = 0; j1 < J; j1++)
                for (int j2 = 0; j2 < J; j2++)
                  inter_distortion_prob_[J] (x, y) (j2, j1) = cept_start_prob_(x, y, j2 - j1 + displacement_offset_);
            }
          }
        }
      }
    }
  }
}

void IBM4Trainer::par2nonpar_inter_distortion(int J, uint sclass, uint tclass)
{
  if (inter_distortion_prob_[J].xDim() <= sclass || inter_distortion_prob_[J].xDim() <= tclass)
    inter_distortion_prob_[J].resize(std::max<uint>(inter_distortion_prob_[J].xDim(), sclass + 1),
                                     std::max<uint>(inter_distortion_prob_[J].yDim(), tclass + 1));
  if (inter_distortion_prob_[J] (sclass, tclass).size() == 0)
    inter_distortion_prob_[J] (sclass, tclass).resize(J, J, 1.0 / J);

  if (reduce_deficiency_) {

    for (int j1 = 0; j1 < J; j1++) {

      double sum = 0.0;

      for (int j2 = 0; j2 < J; j2++) {
        sum += cept_start_prob_(sclass, tclass, j2 - j1 + displacement_offset_);
        assert(!isnan(sum));
      }

      if (sum > 1e-305) {
        for (int j2 = 0; j2 < J; j2++) {
          inter_distortion_prob_[J](sclass, tclass) (j2, j1) =
            std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, j2 - j1 + displacement_offset_) / sum);
        }
      }
      else if (j1 > 0) {
        //std::cerr << "WARNING: sum too small for inter prob " << j1 << ", not updating." << std::endl;
      }
    }
  }
  else {
    for (int j1 = 0; j1 < J; j1++)
      for (int j2 = 0; j2 < J; j2++)
        inter_distortion_prob_[J](sclass, tclass) (j2, j1) = cept_start_prob_(sclass, tclass, j2 - j1 + displacement_offset_);
  }
}

void IBM4Trainer::par2nonpar_intra_distortion()
{
  for (int J = 1; J <= (int)maxJ_; J++) {

    if (intra_distortion_prob_[J].size() > 0) {

      for (uint x = 0; x < intra_distortion_prob_[J].xDim(); x++) {

        if (reduce_deficiency_) {

          for (int j1 = 0; j1 < J - 1; j1++) {

            double sum = 0.0;

            for (int j2 = j1 + 1; j2 < J; j2++) {
              sum += within_cept_prob_(x, j2 - j1);
            }

            if (sum > 1e-305) {
              for (int j2 = j1 + 1; j2 < J; j2++) {
                intra_distortion_prob_[J] (x, j2, j1) = std::max(fert_min_param_entry, within_cept_prob_(x, j2 - j1) / sum);
              }
            }
            else {
              std::cerr << "WARNING: sum too small for intra prob " << j1 << ", J=" << J << ", not updating." << std::endl;
            }
          }
        }
        else {
          for (int j1 = 0; j1 < J - 1; j1++)
            for (int j2 = j1 + 1; j2 < J; j2++)
              intra_distortion_prob_[J](x, j2, j1) = within_cept_prob_(x, j2 - j1);
        }
      }
    }
  }
}

//new variant
double IBM4Trainer::inter_distortion_m_step_energy(const IBM4CeptStartModel& singleton_count,
    const Math2D::Matrix<double>& inter_span_count, const Math3D::Tensor<double>& inter_param, uint sclass, uint tclass) const
{
  // we exploit here that span_count will only be nonzero if zero_offset lies in the span

  double energy = 0.0;

  uint nParams = inter_param.zDim();

  for (uint diff = 0; diff < nParams; diff++)
    energy -= singleton_count(sclass, tclass, diff) * std::log(std::max(fert_min_param_entry, inter_param(sclass, tclass, diff)));

  for (uint diff_start = 0; diff_start < inter_span_count.xDim(); diff_start++) {

    double param_sum = 0.0;

    // for (uint diff_end = diff_start; diff_end < nParams; diff_end++) {

    //   param_sum += std::max(1e-15,inter_param(sclass,tclass,diff_end));

    //   const double count = inter_span_count(diff_start,diff_end);

    //   if (count != 0.0) {
    //  assert(diff_start <= uint(displacement_offset_) && diff_end >= uint(displacement_offset_));
    //  energy += count * std::log(param_sum);
    //   }
    // }

    for (uint d = diff_start; d < uint(displacement_offset_); d++)
      param_sum += std::max(fert_min_param_entry, inter_param(sclass, tclass, d));

    for (uint diff_end = displacement_offset_; diff_end < nParams; diff_end++) {

      param_sum += std::max(fert_min_param_entry, inter_param(sclass, tclass, diff_end));

      const double count = inter_span_count(diff_start, diff_end - displacement_offset_);

      if (count != 0.0) {
        energy += count * std::log(param_sum);
      }
    }
  }

  return energy;
}

//new variant
double IBM4Trainer::intra_distortion_m_step_energy(const IBM4WithinCeptModel& singleton_count,
    const Math2D::Matrix<double>& intra_span_count, const Math2D::Matrix<double>& intra_param, uint word_class) const
{
  double energy = 0.0;

  const uint nParams = intra_param.yDim();

  for (uint diff = 1; diff < nParams; diff++)
    energy -= singleton_count(word_class, diff) * std::log(std::max(fert_min_param_entry, intra_param(word_class, diff)));

  double param_sum = 0.0;
  for (uint diff = 1; diff < nParams; diff++) {

    param_sum += std::max(fert_min_param_entry, intra_param(word_class, diff));
    energy += intra_span_count(word_class, diff) * std::log(param_sum);
  }

  return energy;
}

//new variant
void IBM4Trainer::inter_distortion_m_step(const IBM4CeptStartModel& singleton_count, const Math2D::Matrix<double>& inter_span_count, uint sclass, uint tclass)
{
  const uint nParams = cept_start_prob_.zDim();

  Math3D::Tensor<double> hyp_ceptstart_prob = cept_start_prob_;
  Math1D::Vector<double> ceptstart_grad(nParams);
  Math1D::Vector<double> new_ceptstart_prob(nParams);

  double alpha = 0.01;

  for (uint k = 0; k < nParams; k++)
    cept_start_prob_(sclass, tclass, k) = std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, k));

  double energy = inter_distortion_m_step_energy(singleton_count, inter_span_count, cept_start_prob_, sclass, tclass);

  //test if normalizing the singleton count gives a better starting point
  {
    double sum = 0.0;

    for (uint k = 0; k < nParams; k++)
      sum += singleton_count(sclass, tclass, k);

    if (sum > 1e-305) {

      for (uint k = 0; k < nParams; k++)
        hyp_ceptstart_prob(sclass, tclass, k) = std::max(fert_min_param_entry, singleton_count(sclass, tclass, k) / sum);

      double hyp_energy = inter_distortion_m_step_energy(singleton_count, inter_span_count, hyp_ceptstart_prob, sclass, tclass);

      if (hyp_energy < energy) {

        for (uint k = 0; k < nParams; k++)
          cept_start_prob_(sclass, tclass, k) = hyp_ceptstart_prob(sclass, tclass, k);

        energy = hyp_energy;
      }
    }
  }

  if (nSourceClasses_ * nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  uint maxIter = dist_m_step_iter_;

  for (uint iter = 1; iter <= maxIter; iter++) {

    ceptstart_grad.set_constant(0.0);

    //compute gradient

    for (uint diff = 0; diff < nParams; diff++)
      ceptstart_grad[diff] -= singleton_count(sclass, tclass, diff) / std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, diff));

    for (uint diff_start = 0; diff_start < inter_span_count.xDim(); diff_start++) {

      Math1D::Vector<double> addon(nParams, 0.0);

      double param_sum = 0.0;

      // for (uint diff_end = diff_start; diff_end < nParams; diff_end++) {

      //        param_sum += std::max(1e-15,cept_start_prob_(sclass,tclass,diff_end));

      //        const double count = inter_span_count(diff_start,diff_end);

      //        if (count != 0.0) {
      //          addon[diff_end] = count / param_sum;
      //          // double addon = count / param_sum;

      //          // for (uint diff=diff_start; diff <= diff_end; diff++)
      //          //   ceptstart_grad[diff] += addon;
      //        }
      // }

      // double addon_sum = 0.0;
      // for (int diff = nParams-1; diff >= int(diff_start); diff--) {
      //        addon_sum += addon[diff];
      //        ceptstart_grad[diff] += addon_sum;
      // }

      for (uint d = diff_start; d < uint(displacement_offset_); d++)
        param_sum += std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, d));

      for (uint diff_end = displacement_offset_; diff_end < nParams; diff_end++) {

        param_sum += std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, diff_end));

        const double count = inter_span_count(diff_start, diff_end - displacement_offset_);

        if (count != 0.0) {

          addon[diff_end] = count / param_sum;
          // double addon = count / param_sum;

          // for (uint diff=diff_start; diff <= diff_end; diff++)
          //   ceptstart_grad[diff] += addon;
        }
      }

      double addon_sum = 0.0;
      for (int diff = nParams - 1; diff >= displacement_offset_; diff--) {
        addon_sum += addon[diff];
        ceptstart_grad[diff] += addon_sum;
      }
      for (int diff = diff_start; diff < displacement_offset_; diff++)
        ceptstart_grad[diff] += addon_sum;
    }

    //go in neg. gradient direction
    for (uint k = 0; k < nParams; k++)
      new_ceptstart_prob[k] = cept_start_prob_(sclass, tclass, k) - alpha * ceptstart_grad[k];

    //reproject
    projection_on_simplex(new_ceptstart_prob.direct_access(), cept_start_prob_.zDim(), fert_min_param_entry);

    double best_energy = 1e300;
    bool decreasing = true;

    double lambda = 1.0;
    double best_lambda = 1.0;

    uint nIter = 0;

    while (best_energy > energy || decreasing) {

      nIter++;

      lambda *= 0.5;
      double neg_lambda = 1.0 - lambda;

      for (uint k = 0; k < cept_start_prob_.zDim(); k++)
        hyp_ceptstart_prob(sclass, tclass, k) = neg_lambda * cept_start_prob_(sclass, tclass, k) + lambda * new_ceptstart_prob[k];

      double hyp_energy = inter_distortion_m_step_energy(singleton_count, inter_span_count, hyp_ceptstart_prob, sclass, tclass);

      if (hyp_energy < best_energy) {

        decreasing = true;
        best_lambda = lambda;
        best_energy = hyp_energy;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * best_energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }

    if (best_energy >= energy) {
      if (nSourceClasses_ * nTargetClasses_ <= 4)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = 0; k < cept_start_prob_.zDim(); k++)
      cept_start_prob_(sclass, tclass, k) = neg_best_lambda * cept_start_prob_(sclass, tclass, k) + best_lambda * new_ceptstart_prob[k];

    energy = best_energy;

    if ((nSourceClasses_ * nTargetClasses_ <= 4) && (iter % 5) == 0)
      std::cerr << "iteration " << iter << ", inter energy: " << energy << std::endl;
  }
}

void IBM4Trainer::inter_distortion_m_step_unconstrained(const IBM4CeptStartModel& singleton_count, const Math2D::Matrix<double>& inter_span_count,
    uint sclass, uint tclass, uint L)
{
  const uint nParams = cept_start_prob_.zDim();

  Math3D::Tensor<double> hyp_ceptstart_prob = cept_start_prob_;
  Math1D::Vector<double> ceptstart_grad(nParams);
  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> hyp_work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);

  for (uint k = 0; k < nParams; k++)
    cept_start_prob_(sclass, tclass, k) = std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, k));

  double energy = inter_distortion_m_step_energy(singleton_count, inter_span_count, cept_start_prob_, sclass, tclass);

  //test if normalizing the singleton count gives a better starting point
  {
    double sum = 0.0;

    for (uint k = 0; k < nParams; k++)
      sum += singleton_count(sclass, tclass, k);

    if (sum > 1e-305) {

      for (uint k = 0; k < nParams; k++)
        hyp_ceptstart_prob(sclass, tclass, k) = std::max(fert_min_param_entry, singleton_count(sclass, tclass, k) / sum);

      double hyp_energy = inter_distortion_m_step_energy(singleton_count, inter_span_count, hyp_ceptstart_prob, sclass, tclass);

      if (hyp_energy < energy) {

        for (uint k = 0; k < nParams; k++)
          cept_start_prob_(sclass, tclass, k) =  hyp_ceptstart_prob(sclass, tclass, k);

        energy = hyp_energy;
      }
    }
  }

  if (nSourceClasses_ * nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  uint start_iter = 1; //changed whenever the curvature condition is violated

  double line_reduction_factor = 0.75;

  for (uint k = 0; k < nParams; k++)
    work_param[k] = sqrt(cept_start_prob_(sclass, tclass, k));

  double scale = 1.0;

  uint maxIter = dist_m_step_iter_;

  for (uint iter = 1; iter <= maxIter; iter++) {

    ceptstart_grad.set_constant(0.0);
    work_grad.set_constant(0.0);

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    for (uint diff = 0; diff < nParams; diff++)
      ceptstart_grad[diff] -= singleton_count(sclass, tclass, diff) / std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, diff));

    for (uint diff_start = 0; diff_start < inter_span_count.xDim(); diff_start++) {

      Math1D::Vector<double> addon(nParams, 0.0);

      double param_sum = 0.0;

      // for (uint diff_end = diff_start; diff_end < nParams; diff_end++) {

      //        param_sum += std::max(1e-15,cept_start_prob_(sclass,tclass,diff_end));

      //        const double count = inter_span_count(diff_start,diff_end);

      //        if (count != 0.0) {
      //          addon[diff_end] = count / param_sum;
      //          // double addon = count / param_sum;

      //          // for (uint diff=diff_start; diff <= diff_end; diff++)
      //          //   ceptstart_grad[diff] += addon;
      //        }
      // }

      // double addon_sum = 0.0;
      // for (int diff = nParams-1; diff >= int(diff_start); diff--) {
      //        addon_sum += addon[diff];
      //        ceptstart_grad[diff] += addon_sum;
      // }

      for (uint d = diff_start; d < uint(displacement_offset_); d++)
        param_sum += std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, d));

      for (uint diff_end = displacement_offset_; diff_end < nParams; diff_end++) {

        param_sum += std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, diff_end));

        const double count = inter_span_count(diff_start, diff_end - displacement_offset_);

        if (count != 0.0) {

          addon[diff_end] = count / param_sum;
          // double addon = count / param_sum;

          // for (uint diff=diff_start; diff <= diff_end; diff++)
          //   ceptstart_grad[diff] += addon;
        }
      }

      double addon_sum = 0.0;
      for (int diff = nParams - 1; diff >= displacement_offset_; diff--) {
        addon_sum += addon[diff];
        ceptstart_grad[diff] += addon_sum;
      }
      for (int diff = diff_start; diff < displacement_offset_; diff++)
        ceptstart_grad[diff] += addon_sum;
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

    for (uint k = 0; k < nParams; k++) {
      const double wp = work_param[k];
      const double grad = ceptstart_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 0; kk < nParams; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

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

        search_direction.add_vector_multiple(cur_step, gamma);
        // for (uint k=0; k < nParams; k++)
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

      for (uint k = 0; k < nParams; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_work_param[k] * hyp_work_param[k];
      }

      for (uint k = 0; k < nParams; k++)
        hyp_ceptstart_prob(sclass, tclass, k) = std::max(fert_min_param_entry, hyp_work_param[k] * hyp_work_param[k] / sqr_sum);

      double hyp_energy = inter_distortion_m_step_energy(singleton_count, inter_span_count, hyp_ceptstart_prob, sclass, tclass);

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
    for (uint k = 0; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      scale += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    for (uint k = 0; k < nParams; k++)
      cept_start_prob_(sclass, tclass, k) = std::max(fert_min_param_entry, work_param[k] * work_param[k] / scale);
  }

  for (uint k = 0; k < nParams; k++)
    cept_start_prob_(sclass, tclass, k) = std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, k));
}

//new variant
void IBM4Trainer::intra_distortion_m_step(const IBM4WithinCeptModel& singleton_count, const Math2D::Matrix<double>& intra_span_count, uint word_class)
{
  const uint nParams = within_cept_prob_.yDim();

  Math2D::Matrix<double> hyp_within_cept_prob = within_cept_prob_;
  Math1D::Vector<double> within_cept_grad(nParams);
  Math1D::Vector<double> new_within_cept_prob(nParams);

  double alpha = 0.01;

  for (uint d = 1; d < nParams; d++)
    within_cept_prob_(word_class, d) = std::max(fert_min_param_entry, within_cept_prob_(word_class, d));

  double energy = intra_distortion_m_step_energy(singleton_count, intra_span_count, within_cept_prob_, word_class);

  //check if normalizing the singleton count gives a lower energy
  {
    double sum = 0.0;
    for (uint d = 1; d < nParams; d++)
      sum += singleton_count(word_class, d);

    for (uint d = 1; d < nParams; d++)
      hyp_within_cept_prob(word_class, d) = std::max(fert_min_param_entry, singleton_count(word_class, d) / sum);

    double hyp_energy = intra_distortion_m_step_energy(singleton_count, intra_span_count, hyp_within_cept_prob, word_class);

    if (hyp_energy < energy) {

      for (uint d = 1; d < nParams; d++)
        within_cept_prob_(word_class, d) = hyp_within_cept_prob(word_class, d);

      energy = hyp_energy;
    }
  }

  if (nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  const uint maxIter = dist_m_step_iter_;

  for (uint iter = 1; iter <= maxIter; iter++) {

    within_cept_grad.set_constant(0.0);

    //calculate gradient
    for (uint diff = 1; diff < nParams; diff++)
      within_cept_grad[diff] -= singleton_count(word_class, diff) / std::max(fert_min_param_entry, within_cept_prob_(word_class, diff));

    Math1D::Vector<double> addon(nParams);

    double param_sum = 0.0;
    for (uint diff = 1; diff < nParams; diff++) {

      param_sum += std::max(fert_min_param_entry, within_cept_prob_(word_class, diff));
      addon[diff] = intra_span_count(word_class, diff) / param_sum;
      // double addon = intra_span_count(word_class,diff) / param_sum;

      // for (uint d=1; d <= diff; d++)
      //        within_cept_grad[d] += addon;
    }

    double sum_addon = 0.0;
    for (uint diff = nParams - 1; diff >= 1; diff--) {
      sum_addon += addon[diff];
      within_cept_grad[diff] += sum_addon;
    }

    //go in neg. gradient direction
    for (uint k = 1; k < within_cept_prob_.yDim(); k++) {

      new_within_cept_prob[k] =
        within_cept_prob_(word_class, k) - alpha * within_cept_grad[k];

      if (fabs(new_within_cept_prob[k]) > 1e75)
        std::cerr << "error: abnormally large number: " << new_within_cept_prob[k] << std::endl;
    }

    //reproject
    projection_on_simplex(new_within_cept_prob.direct_access() + 1, new_within_cept_prob.size() - 1, fert_min_param_entry); //the entry for 0 is always 0!

    double best_energy = 1e300;
    bool decreasing = true;

    double lambda = 1.0;
    double best_lambda = 1.0;

    uint nIter = 0;

    while (best_energy > energy || decreasing) {

      nIter++;

      lambda *= 0.5;
      double neg_lambda = 1.0 - lambda;

      for (uint k = 0; k < within_cept_prob_.yDim(); k++)
        hyp_within_cept_prob(word_class, k) = neg_lambda * within_cept_prob_(word_class, k) + lambda * new_within_cept_prob[k];

      double hyp_energy = intra_distortion_m_step_energy(singleton_count, intra_span_count, hyp_within_cept_prob, word_class);

      if (hyp_energy < best_energy) {

        decreasing = true;
        best_lambda = lambda;
        best_energy = hyp_energy;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }

    if (best_energy >= energy) {
      if (nTargetClasses_ <= 4)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = 0; k < within_cept_prob_.yDim(); k++)
      within_cept_prob_(word_class, k) = neg_best_lambda * within_cept_prob_(word_class, k) + best_lambda * new_within_cept_prob[k];

    energy = best_energy;

    if ((nTargetClasses_ <= 4) && (iter % 5) == 0)
      std::cerr << "iteration " << iter << ", intra energy: " << energy << std::endl;
  }
}

void IBM4Trainer::intra_distortion_m_step_unconstrained(const IBM4WithinCeptModel& singleton_count, const Math2D::Matrix<double>& intra_span_count,
    uint word_class, uint L)
{
  const uint nParams = within_cept_prob_.yDim();

  Math2D::Matrix<double> hyp_within_cept_prob = within_cept_prob_;
  Math1D::Vector<double> within_cept_grad(nParams);
  Math1D::Vector<double> work_param(nParams, 0.0);
  Math1D::Vector<double> hyp_work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams, 0.0);

  for (uint d = 1; d < nParams; d++)
    within_cept_prob_(word_class, d) = std::max(fert_min_param_entry, within_cept_prob_(word_class, d));

  double energy =
    intra_distortion_m_step_energy(singleton_count, intra_span_count, within_cept_prob_, word_class);

  //check if normalizing the singleton count gives a lower energy
  {
    double sum = 0.0;
    for (uint d = 1; d < nParams; d++)
      sum += singleton_count(word_class, d);

    for (uint d = 1; d < nParams; d++)
      hyp_within_cept_prob(word_class, d) = std::max(fert_min_param_entry, singleton_count(word_class, d) / sum);

    double hyp_energy = intra_distortion_m_step_energy(singleton_count, intra_span_count, hyp_within_cept_prob, word_class);

    if (hyp_energy < energy) {

      for (uint d = 1; d < nParams; d++)
        within_cept_prob_(word_class, d) = hyp_within_cept_prob(word_class, d);

      energy = hyp_energy;
    }
  }

  if (nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  double line_reduction_factor = 0.75;

  for (uint k = 1; k < nParams; k++)
    work_param[k] = sqrt(within_cept_prob_(word_class, k));

  double scale = 1.0;

  uint maxIter = dist_m_step_iter_;

  for (uint iter = 1; iter <= maxIter; iter++) {

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    within_cept_grad.set_constant(0.0);

    for (uint diff = 1; diff < nParams; diff++)
      within_cept_grad[diff] -= singleton_count(word_class, diff) / std::max(fert_min_param_entry, within_cept_prob_(word_class, diff));

    Math1D::Vector<double> addon(nParams);

    double param_sum = 0.0;
    for (uint diff = 1; diff < nParams; diff++) {

      param_sum += std::max(fert_min_param_entry, within_cept_prob_(word_class, diff));
      addon[diff] = intra_span_count(word_class, diff) / param_sum;
      // double addon = intra_span_count(word_class,diff) / param_sum;

      // for (uint d=1; d <= diff; d++)
      //        within_cept_grad[d] += addon;
    }

    double sum_addon = 0.0;
    for (uint diff = nParams - 1; diff >= 1; diff--) {
      sum_addon += addon[diff];
      within_cept_grad[diff] += sum_addon;
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

    for (uint k = 1; k < nParams; k++) {
      const double wp = work_param[k];
      const double grad = within_cept_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 1; kk < nParams; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 1; k < nParams; k++) {

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

        double cur_alpha = 0.0;
        for (uint k = 1; k < nParams; k++) {
          cur_alpha += search_direction[k] * cur_step[k];
        }
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        for (uint k = 1; k < nParams; k++)
          search_direction[k] -= cur_alpha * cur_grad_diff[k];
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = 0.0;
        for (uint k = 1; k < nParams; k++) {
          beta += search_direction[k] * cur_grad_diff[k];
        }
        beta *= rho[prev_l];

        const double gamma = alpha[prev_l] - beta;

        for (uint k = 1; k < nParams; k++) {
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

      for (uint k = 1; k < nParams; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
      }

      const double sum = hyp_work_param.sqr_norm();
      for (uint d = 1; d < nParams; d++)
        hyp_within_cept_prob(word_class, d) = std::max(fert_min_param_entry, hyp_work_param[d] * hyp_work_param[d] / sum);

      double hyp_energy = intra_distortion_m_step_energy(singleton_count, intra_span_count, hyp_within_cept_prob, word_class);

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
    for (uint k = 1; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      scale += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    for (uint k = 1; k < nParams; k++) {
      within_cept_prob_(word_class, k) = std::max(fert_min_param_entry, work_param[k] * work_param[k] / scale);
    }
  }
}

void IBM4Trainer::init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperWithClasses* passed_wrapper, bool clear_prev,
                                      bool collect_counts, bool viterbi)
{
  std::cerr << "******** initializing IBM-4 from " << prev_model->model_name() << " *******" << std::endl;

  if (collect_counts) {

    best_known_alignment_ = prev_model->best_alignments();

    if (viterbi) {
      train_viterbi(1, prev_model, passed_wrapper);
    }
    else {
      train_em(1, prev_model, passed_wrapper);
    }

    iter_offs_ = 1;
  }
  else {

    best_known_alignment_ = prev_model->update_alignments_unconstrained(true, passed_wrapper);

    FertilityModelTrainer* fert_model = dynamic_cast < FertilityModelTrainer* >(prev_model);

    if (!fix_p0_ && fert_model != 0) {
      p_zero_ = fert_model->p_zero();
      p_nonzero_ = 1.0 - p_zero_;
    }

    if (fert_model == 0) {
      init_fertilities(0); //alignments were already updated an set
    }
    else {

      fertility_prob_.resize(fert_model->fertility_prob().size());

      for (uint k = 1; k < fertility_prob_.size(); k++) {
        fertility_prob_[k] = fert_model->fertility_prob()[k];

        //EXPERIMENTAL
        for (uint l = 0; l < fertility_prob_[k].size(); l++) {
          if (l <= fertility_limit_[k])
            fertility_prob_[k][l] =
              0.95 * std::max(fert_min_param_entry, fertility_prob_[k][l]) + 0.05 / std::min<uint>(fertility_prob_[k].size(), fertility_limit_[k]);
          else
            fertility_prob_[k][l] = 0.95 * fertility_prob_[k][l];
        }
        //END_EXPERIMENTAL
      }
    }

    //init distortion models from best known alignments
    cept_start_prob_.set_constant(0.0);
    if (!uniform_intra_prob_)
      within_cept_prob_.set_constant(0.0);
    if (use_sentence_start_prob_)
      sentence_start_parameters_.set_constant(0.0);

    for (size_t s = 0; s < source_sentence_.size(); s++) {

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];

      const Math1D::Vector < AlignBaseType >& cur_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      NamedStorage1D<std::vector<int> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));

      for (uint j = 0; j < curJ; j++) {
        const uint aj = cur_alignment[j];
        aligned_source_words[aj].push_back(j);
      }

      int prev_center = -100;
      int prev_cept = -1;

      for (uint i = 1; i <= curI; i++) {

        if (!aligned_source_words[i].empty()) {

          const int first_j = aligned_source_words[i][0];

          //collect counts for the head model
          if (prev_center >= 0) {
            const uint sclass = source_class_[cur_source[first_j]];
            const uint tclass =
              (inter_dist_mode_ == IBM4InterDistModePrevious) ? target_class_[cur_target[prev_cept - 1]] : target_class_[cur_target[i - 1]];

            int diff = first_j - prev_center;
            diff += displacement_offset_;
            cept_start_prob_(sclass, tclass, diff) += 1.0;
          }
          else if (use_sentence_start_prob_)
            sentence_start_parameters_[first_j] += 1.0;

          //collect counts for the within-cept model
          int prev_j = first_j;

          if (!uniform_intra_prob_) {
            for (uint k = 1; k < aligned_source_words[i].size(); k++) {

              const int cur_j = aligned_source_words[i][k];

              const uint tclass = target_class_[cur_target[i - 1]];
              const uint sclass = source_class_[cur_source[cur_j]];

              int diff = cur_j - prev_j;
              if (intra_dist_mode_ == IBM4IntraDistModeTarget)
                within_cept_prob_(tclass, diff) += 1.0;
              else
                within_cept_prob_(sclass, diff) += 1.0;

              prev_j = cur_j;
            }
          }

          //update prev_center
          switch (cept_start_mode_) {
          case IBM4CENTER: {
            double sum_j = vec_sum(aligned_source_words[i]);
            prev_center = (int)round(sum_j / aligned_source_words[i].size());
            break;
          }
          case IBM4FIRST:
            prev_center = first_j;
            break;
          case IBM4LAST: {
            prev_center = prev_j;     //was set to the last pos in the above loop
          }
          break;
          case IBM4UNIFORM:
            prev_center = first_j;      //will not be used
            break;
          }
          prev_cept = i;
        }
      }
    }

    //now that all counts are collected, initialize the distributions

    //a) cept start
    for (uint x = 0; x < cept_start_prob_.xDim(); x++) {
      for (uint y = 0; y < cept_start_prob_.yDim(); y++) {

        double sum = 0.0;
        for (uint d = 0; d < cept_start_prob_.zDim(); d++)
          sum += cept_start_prob_(x, y, d);

        if (sum > 1e-300) {
          const double count_factor = 0.9 / sum;
          const double uniform_share = 0.1 / cept_start_prob_.zDim();

          for (uint d = 0; d < cept_start_prob_.zDim(); d++)
            cept_start_prob_(x, y, d) = count_factor * cept_start_prob_(x, y, d) + uniform_share;
        }
        else {
          //this combination did not occur in the viterbi alignments
          //but it may still be possible in the data
          for (uint d = 0; d < cept_start_prob_.zDim(); d++)
            cept_start_prob_(x, y, d) = 1.0 / cept_start_prob_.zDim();
        }
      }
    }

    par2nonpar_inter_distortion();

    if (!uniform_intra_prob_) {
      //b) within-cept
      for (uint x = 0; x < within_cept_prob_.xDim(); x++) {

        double sum = 0.0;
        for (uint d = 0; d < within_cept_prob_.yDim(); d++)
          sum += within_cept_prob_(x, d);

        if (sum > 1e-300) {
          const double count_factor = 0.9 / sum;
          const double uniform_share = 0.1 / (within_cept_prob_.yDim() - 1);

          for (uint d = 0; d < within_cept_prob_.yDim(); d++) {
            if (d == 0) {
              //zero-displacements are impossible within cepts
              within_cept_prob_(x, d) = 0.0;
            }
            else
              within_cept_prob_(x, d) = count_factor * within_cept_prob_(x, d) + uniform_share;
          }
        }
        else {
          for (uint d = 0; d < within_cept_prob_.yDim(); d++) {
            if (d == 0) {
              //zero-displacements are impossible within cepts
              within_cept_prob_(x, d) = 0.0;
            }
            else
              within_cept_prob_(x, d) = 1.0 / (within_cept_prob_.yDim() - 1);
          }
        }
      }

      par2nonpar_intra_distortion();
    }

    //c) sentence start prob
    if (use_sentence_start_prob_) {

      double sum = sentence_start_parameters_.sum();
      for (uint k = 0; k < sentence_start_parameters_.size(); k++)
        sentence_start_parameters_[k] *= std::max(fert_min_param_entry, sentence_start_parameters_[k] / sum);

      par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
      //par2nonpar_start_prob();
    }
  }

  //DEBUG
#ifndef NDEBUG
  std::cerr << "checking" << std::endl;

  for (size_t s = 0; s < source_sentence_.size(); s++) {

    long double align_prob = FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]);

    if (isinf(align_prob) || isnan(align_prob) || align_prob == 0.0) {

      std::cerr << "ERROR: initial align-prob for sentence " << s << " has prob " << align_prob << std::endl;
      exit(1);
    }
  }
#endif
  //END_DEBUG

  if (clear_prev)
    prev_model->release_memory();
}

/*virtual*/ long double IBM4Trainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
    const Math1D::Vector<AlignBaseType>& alignment) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));
  NamedStorage1D<std::vector<int> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));

  const Math3D::Tensor<float>& cur_intra_distortion_prob = intra_distortion_prob_[curJ];

  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    aligned_source_words[aj].push_back(j);
    fertility[aj]++;

    if (aj == 0) {
      prob *= dict_[0][source[j] - 1];
      //DEBUG
      if (isnan(prob))
        std::cerr << "prob nan after empty word dict prob" << std::endl;
      //END_DEBUG
    }
  }

  if (curJ < 2 * fertility[0])
    return 0.0;

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = target[i - 1];
    prob *= fertility_prob_[t_idx][fertility[i]];
    if (!no_factorial_)
      prob *= ld_fac_[fertility[i]];
  }

  //DEBUG
  if (isnan(prob))
    std::cerr << "prob nan after fertility probs" << std::endl;
  //END_DEBUG

  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;
  int prev_cept = -1;

  for (uint i = 1; i <= curI; i++) {

    if (fertility[i] > 0) {

      const std::vector<int>& cur_aligned_source_words = aligned_source_words[i];

      const uint ti = target[i - 1];
      uint tclass = target_class_[ti];

      const Math1D::Vector<double>& cur_dict = dict_[ti];

      const int first_j = cur_aligned_source_words[0];

      prob *= cur_dict[lookup(first_j, i - 1)];

      //handle the head of the cept
      if (prev_cept_center != -1) {

        //DEBUG
        if (isnan(prob))
          std::cerr << "prob nan after dict-prob, pc != -1, i=" << i << std::
                    endl;
        //END_DEBUG

        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[first_j]];

          if (inter_dist_mode_ == IBM4InterDistModePrevious)
            tclass = target_class_[target[prev_cept - 1]];

          prob *= inter_distortion_prob(first_j, prev_cept_center, sclass, tclass, curJ);

          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after inter-distort prob, i=" << i << std::endl;
          //END_DEBUG
        }
        else
          prob /= curJ;
      }
      else {

        if (use_sentence_start_prob_) {
          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after dict-prob, pc == -1, i=" << i << std::endl;
          //END_DEBUG

          prob *= cur_sentence_start_prob[first_j];

          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after sent start prob, pc == -1, i=" << i << std::endl;
          //END_DEBUG
        }
        else {

          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after dict-prob, pc == -1, i=" << i << std::endl;
          //END_DEBUG

          prob *= 1.0 / curJ;
        }
      }

      //handle the body of the cept
      int prev_j = first_j;
      for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

        const int cur_j = cur_aligned_source_words[k];

        const uint cur_class =
          (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[source[cur_j]] : target_class_[target[i - 1]];

        prob *= cur_dict[lookup(cur_j, i - 1)] * cur_intra_distortion_prob(cur_class, cur_j, prev_j);

        //std::cerr << "ap: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

        //DEBUG
        if (isnan(prob))
          std::cerr << "prob nan after combined body-prob, i=" << i << std::endl;
        //END_DEBUG

        prev_j = cur_j;
      }

      //compute the center of this cept and store the result in prev_cept_center

      switch (cept_start_mode_) {
      case IBM4CENTER: {
        double sum = vec_sum(cur_aligned_source_words);
        prev_cept_center = (int)round(sum / fertility[i]);
        break;
      }
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST: {
        prev_cept_center = prev_j;    //was set to the last pos in the above loop
        break;
      }
      case IBM4UNIFORM:
        prev_cept_center = first_j;     //will not be used
        break;
      default:
        assert(false);
      }

      prev_cept = i;

      assert(prev_cept_center >= 0);
    }
  }

  //handle empty word -- dictionary probs were handled above
  assert(fertility[0] <= 2 * curJ);

  const uint zero_fert = fertility[0];
  prob *= choose_factor_[curJ][zero_fert];
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  prob *= p_zero_pow_[zero_fert];
  prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  if (empty_word_model_ == FertNullOchNey) {
    prob *= och_ney_factor_[curJ][zero_fert];
  }
  else if (empty_word_model_ == FertNullIntra) {
    if (zero_fert > 0) {
      prob /= (double)curJ;

      const std::vector<int>& cur_aligned_source_words = aligned_source_words[0];

      for (uint k = 1; k < zero_fert; k++) {
        const int cur = cur_aligned_source_words[k];
        const int prev = cur_aligned_source_words[k - 1];
        prob *= null_intra_prob_[cur - prev];
      }
    }
  }

  return prob;
}

long double IBM4Trainer::nondeficient_alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& cur_lookup,
    const Math1D::Vector<AlignBaseType>& alignment) const
{
  //this is exactly like for the IBM-3 (the difference is in the subroutine nondeficient_distortion_prob())

  long double prob = 1.0;

  const Storage1D<uint>& cur_source = source;
  const Storage1D<uint>& cur_target = target;

  const uint curI = cur_target.size();
  const uint curJ = cur_source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

  Storage1D<std::vector<AlignBaseType> > aligned_source_words(curI + 1);    //words are listed in ascending order

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    aligned_source_words[aj].push_back(j);
  }

  //std::cerr << "ap: fertility: " << fertility << std::endl;

  const uint zero_fert = fertility[0];
  if (curJ < 2 * zero_fert)
    return 0.0;

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = cur_target[i - 1];
    //NOTE: no factorial here
    prob *= fertility_prob_[t_idx][fertility[i]];
  }
  for (uint j = 0; j < curJ; j++) {

    uint s_idx = cur_source[j];
    uint aj = alignment[j];

    if (aj == 0)
      prob *= dict_[0][s_idx - 1];
    else {
      uint t_idx = cur_target[aj - 1];
      prob *= dict_[t_idx][cur_lookup(j, aj - 1)];
    }
  }

  prob *= nondeficient_distortion_prob(source, target, aligned_source_words);

  //std::cerr << "ap before empty word: " << prob << std::endl;

  //handle empty word
  assert(zero_fert <= 2 * curJ);

  prob *= choose_factor_[curJ][zero_fert];
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  prob *= p_zero_pow_[zero_fert];
  prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  return prob;
}

long double IBM4Trainer::distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const Math1D::Vector<AlignBaseType>& alignment) const
{
  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  NamedStorage1D<std::vector<AlignBaseType> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    aligned_source_words[aj].push_back(j);
  }

  return distortion_prob(source, target, aligned_source_words);
}

//NOTE: the vectors need to be sorted
long double IBM4Trainer::distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Math3D::Tensor<float>& cur_intra_distortion_prob = intra_distortion_prob_[curJ];

  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  if (curJ < 2 * aligned_source_words[0].size())
    return 0.0;

  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;
  int prev_cept = -1;

  for (uint i = 1; i <= curI; i++) {

    const std::vector<AlignBaseType>& cur_aligned_source_words = aligned_source_words[i];

    if (cur_aligned_source_words.size() > 0) {

      const uint ti = target[i - 1];
      uint tclass = target_class_[ti];

      const int first_j = cur_aligned_source_words[0];

      //handle the head of the cept
      if (prev_cept_center != -1) {

        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[first_j]];

          if (inter_dist_mode_ == IBM4InterDistModePrevious)
            tclass = target_class_[target[prev_cept - 1]];

          prob *= inter_distortion_prob(first_j, prev_cept_center, sclass, tclass, curJ);
        }
        else
          prob /= curJ;
      }
      else {
        if (use_sentence_start_prob_) {
          prob *= cur_sentence_start_prob[first_j];
        }
        else {
          prob *= 1.0 / curJ;
        }
      }

      //handle the body of the cept
      int prev_j = first_j;
      for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

        const int cur_j = cur_aligned_source_words[k];

        const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[source[cur_j]] : target_class_[target[i - 1]];

        prob *= cur_intra_distortion_prob(cur_class, cur_j, prev_j);

        //std::cerr << "dp: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

        prev_j = cur_j;
      }

      switch (cept_start_mode_) {
      case IBM4CENTER: {

        //compute the center of this cept and store the result in prev_cept_center
        double sum = vec_sum(cur_aligned_source_words);
        prev_cept_center = (int)round(sum / cur_aligned_source_words.size());
        break;
      }
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST:
        prev_cept_center = prev_j;      //was set to the last position in the above loop
        break;
      case IBM4UNIFORM:
        prev_cept_center = first_j;
        break;
      default:
        assert(false);
      }

      prev_cept = i;
      assert(prev_cept_center >= 0);
    }
  }

  return prob;
}

long double IBM4Trainer::null_distortion_prob(const std::vector<AlignBaseType>& null_aligned_source_words, uint curJ) const
{
  long double prob = 1.0;
  const uint zero_fert = null_aligned_source_words.size();
  if (zero_fert > 0) {

    if (empty_word_model_ == FertNullOchNey)
      prob = och_ney_factor_[curJ][zero_fert];
    else if (empty_word_model_ == FertNullIntra) {
      prob /= (double)curJ;
      for (uint k = 1; k < zero_fert; k++) {
        const uint cur = null_aligned_source_words[k];
        const uint prev = null_aligned_source_words[k - 1];
        prob *= null_intra_prob_[cur - prev];
      }
    }
  }
  return prob;
}

double IBM4Trainer::intra_distortion_prob(const std::vector<AlignBaseType>& aligned_source_words, const Math1D::Vector<uint>& sclass, uint tclass) const
{
  double prob = 1.0;

  if (aligned_source_words.size() > 1) {

    const Math3D::Tensor<float>& cur_intra_distortion_prob = intra_distortion_prob_[sclass.size()];

    int prev_j = aligned_source_words[0];
    for (uint k = 1; k < aligned_source_words.size(); k++) {

      const int cur_j = aligned_source_words[k];

      const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[cur_j] : tclass;

      prob *= cur_intra_distortion_prob(cur_class, cur_j, prev_j);

      //std::cerr << "dp: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

      prev_j = cur_j;
    }
  }

  return prob;
}

//NOTE: the vectors need to be sorted
long double IBM4Trainer::inter_distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  if (curJ < 2 * aligned_source_words[0].size())
    return 0.0;

  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;
  int prev_cept = -1;

  for (uint i = 1; i <= curI; i++) {

    const std::vector<AlignBaseType>& cur_aligned_source_words = aligned_source_words[i];

    if (cur_aligned_source_words.size() > 0) {

      const uint ti = target[i - 1];
      uint tclass = target_class_[ti];

      const int first_j = cur_aligned_source_words[0];

      //handle the head of the cept
      if (prev_cept_center != -1) {

        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[first_j]];

          if (inter_dist_mode_ == IBM4InterDistModePrevious)
            tclass = target_class_[target[prev_cept - 1]];

          prob *= inter_distortion_prob(first_j, prev_cept_center, sclass, tclass, curJ);
        }
        else
          prob /= curJ;
      }
      else {
        if (use_sentence_start_prob_) {
          prob *= cur_sentence_start_prob[first_j];
        }
        else {
          prob *= 1.0 / curJ;
        }
      }

      switch (cept_start_mode_) {
      case IBM4CENTER: {

        //compute the center of this cept and store the result in prev_cept_center
        double sum = vec_sum(cur_aligned_source_words);
        prev_cept_center = (int)round(sum / cur_aligned_source_words.size());
        break;
      }
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST:
        prev_cept_center = cur_aligned_source_words.back();
        break;
      case IBM4UNIFORM:
        prev_cept_center = first_j;
        break;
      default:
        assert(false);
      }

      prev_cept = i;
      assert(prev_cept_center >= 0);
    }
  }

  return prob;
}

//NOTE: the vectors need to be sorted
long double IBM4Trainer::nondeficient_distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  if (curJ < 2 * aligned_source_words[0].size())
    return 0.0;

  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;
  int prev_cept = -1;

  Storage1D<bool> fixed(curJ, false);

  for (uint i = 1; i <= curI; i++) {

    const std::vector<AlignBaseType>& cur_aligned_source_words = aligned_source_words[i];

    if (cur_aligned_source_words.size() > 0) {

      const uint ti = target[i - 1];
      uint tclass = target_class_[ti];

      const int first_j = cur_aligned_source_words[0];

      uint nToRemove = cur_aligned_source_words.size() - 1;

      //handle the head of the cept
      if (prev_cept_center != -1) {

        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[first_j]];

          if (inter_dist_mode_ == IBM4InterDistModePrevious)
            tclass = target_class_[target[prev_cept - 1]];

          double num = cept_start_prob_(sclass, tclass, first_j - prev_cept_center + displacement_offset_);
          double denom = 0.0;

#if 0
          //this version is several orders of magnitude slower
          std::vector<uint> open_pos;
          open_pos.reserve(curJ);
          for (int j = 0; j < int (curJ); j++) {
            if (!fixed[j])
              open_pos.push_back(j);
          }
          if (nToRemove > 0)
            open_pos.resize(open_pos.size() - nToRemove);

          for (uint k = 0; k < open_pos.size(); k++)
            denom += cept_start_prob_(sclass, tclass, open_pos[k] - prev_cept_center + displacement_offset_);
#else
          for (int j = 0; j < int (curJ); j++) {
            if (!fixed[j])
              denom += cept_start_prob_(sclass, tclass, j - prev_cept_center + displacement_offset_);
          }

          if (nToRemove > 0) {
            uint nRemoved = 0;
            for (int jj = curJ - 1; jj >= 0; jj--) {
              if (!fixed[jj]) {
                denom -= cept_start_prob_(sclass, tclass, jj - prev_cept_center + displacement_offset_);
                nRemoved++;
                if (nRemoved == nToRemove)
                  break;
              }
            }
            assert(nRemoved == nToRemove);
          }
#endif
          prob *= num / denom;
        }
        else
          prob /= curJ;
      }
      else {
        if (use_sentence_start_prob_) {
          prob *= cur_sentence_start_prob[first_j];
        }
        else {
          prob *= 1.0 / curJ;
        }
      }
      fixed[first_j] = true;

      //handle the body of the cept
      int prev_j = first_j;
      for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

        nToRemove--;

        const int cur_j = cur_aligned_source_words[k];

        const uint cur_class =
          (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[source[cur_j]] : target_class_[target[i - 1]];

        double num = within_cept_prob_(cur_class, cur_j - prev_j);

        double denom = 0.0;
#if 0
        //this version is several orders of magnitude slower
        std::vector<uint> open_pos;
        open_pos.reserve(curJ);
        for (int j = prev_j + 1; j < int (curJ); j++) {
          if (!fixed[j])
            open_pos.push_back(j);
        }
        if (nToRemove > 0)
          open_pos.resize(open_pos.size() - nToRemove);

        for (uint k = 0; k < open_pos.size(); k++)
          denom += within_cept_prob_(cur_class, open_pos[k] - prev_j);
#else

        for (uint j = prev_j + 1; j < curJ; j++) {
          denom += within_cept_prob_(cur_class, j - prev_j);
        }

        if (nToRemove > 0) {

          uint nRemoved = 0;
          for (int jj = curJ - 1; jj >= 0; jj--) {
            if (!fixed[jj]) {
              denom -= within_cept_prob_(cur_class, jj - prev_j);
              nRemoved++;
              if (nRemoved == nToRemove)
                break;
            }
          }
          assert(nRemoved == nToRemove);
        }
#endif

        prob *= num / denom;

        //std::cerr << "dp: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

        fixed[cur_j] = true;

        prev_j = cur_j;
      }

      switch (cept_start_mode_) {
      case IBM4CENTER: {

        //compute the center of this cept and store the result in prev_cept_center
        double sum = vec_sum(cur_aligned_source_words);
        // double sum = 0.0;
        // for (uint k=0; k < cur_aligned_source_words.size(); k++) {
        //   sum += cur_aligned_source_words[k];
        // }

        prev_cept_center = (int)round(sum / cur_aligned_source_words.size());
        break;
      }
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST:
        prev_cept_center = prev_j;      //was set to the last position in the above loop
        break;
      case IBM4UNIFORM:
        prev_cept_center = first_j;     //will not be used
        break;
      default:
        assert(false);
      }

      prev_cept = i;
      assert(prev_cept_center >= 0);
    }
  }

  return prob;
}

void IBM4Trainer::print_alignment_prob_factors(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& cur_lookup, const Math1D::Vector<AlignBaseType>& alignment) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));
  NamedStorage1D<std::set<int> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));

  const Math3D::Tensor<float>& cur_intra_distortion_prob = intra_distortion_prob_[curJ];
  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    aligned_source_words[aj].insert(j);
    fertility[aj]++;

    if (aj == 0) {
      prob *= dict_[0][source[j] - 1];

      std::cerr << "mult by dict-prob for empty word, factor: " << dict_[0][source[j] - 1]
                << ", result: " << prob << std::endl;
    }
  }

  if (curJ < 2 * fertility[0]) {

    std::cerr << "ERROR: too many zero-aligned words, returning 0.0" << std::endl;
    return;
  }

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = target[i - 1];
    prob *= fertility_prob_[t_idx][fertility[i]];

    std::cerr << "mult by fert-prob " << fertility_prob_[t_idx][fertility[i]] << ", result: " << prob << std::endl;

    if (!no_factorial_) {
      prob *= ld_fac_[fertility[i]];

      std::cerr << "mult by factorial " << ld_fac_[fertility[i]] << ", result: " << prob << std::endl;
    }
  }

  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;
  int prev_cept = -1;

  for (uint i = 1; i <= curI; i++) {

    if (fertility[i] > 0) {
      const uint ti = target[i - 1];
      uint tclass = target_class_[ti];

      const int first_j = *aligned_source_words[i].begin();

      //handle the head of the cept
      if (prev_cept_center != -1) {

        const int first_j = *aligned_source_words[i].begin();
        prob *= dict_[ti][cur_lookup(first_j, i - 1)];

        std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(first_j, i - 1)]
                  << ", result: " << prob << std::endl;

        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[first_j]];

          if (inter_dist_mode_ == IBM4InterDistModePrevious)
            tclass = target_class_[target[prev_cept - 1]];

          prob *= inter_distortion_prob(first_j, prev_cept_center, sclass, tclass, curJ);

          std::cerr << "mult by distortion-prob " << inter_distortion_prob(first_j, prev_cept_center, sclass, tclass, curJ)
                    << ", result: " << prob << std::endl;

        }
        else {
          prob /= curJ;

          std::cerr << "div by " << curJ << ", result: " << prob << std::endl;
        }
      }
      else {
        if (use_sentence_start_prob_) {
          prob *= dict_[ti][cur_lookup(first_j, i - 1)];

          std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(first_j, i - 1)]
                    << ", result: " << prob << std::endl;

          prob *= cur_sentence_start_prob[first_j];

          std::cerr << "mult by start prob " << cur_sentence_start_prob[first_j] << ", result: " << prob << std::endl;
        }
        else {
          prob *= dict_[ti][cur_lookup(first_j, i - 1)];

          std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(first_j, i - 1)] << ", result: " << prob << std::endl;

          prob *= 1.0 / curJ;

          std::cerr << "div by " << curJ << ", result: " << prob << std::endl;
        }
      }

      //handle the body of the cept
      int prev_j = first_j;
      std::set<int>::const_iterator ait = aligned_source_words[i].begin();
      for (++ait; ait != aligned_source_words[i].end(); ait++) {

        const int cur_j = *ait;

        const uint cur_class =
          (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[source[cur_j]] : target_class_[target[i - 1]];

        prob *= dict_[ti][cur_lookup(cur_j, i - 1)] * cur_intra_distortion_prob(cur_class, cur_j, prev_j);

        std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(cur_j, i - 1)]
                  << " and distortion-prob " << cur_intra_distortion_prob(cur_class, cur_j, prev_j)
                  << ", result: " << prob << ", target index: " << ti << ", source index: " << cur_lookup(cur_j, i - 1) << std::endl;

        prev_j = cur_j;
      }

      //compute the center of this cept and store the result in prev_cept_center

      switch (cept_start_mode_) {
      case IBM4CENTER: {
        double sum = set_sum(aligned_source_words[i]);
        prev_cept_center = (int)round(sum / fertility[i]);
        break;
      }
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST: {
        prev_cept_center = prev_j;    //was set to the last position in the above loop
        break;
      }
      case IBM4UNIFORM:
        prev_cept_center = first_j;     //will not be used
        break;
      default:
        assert(false);
      }

      prev_cept = i;
      assert(prev_cept_center >= 0);
    }
  }

  //handle empty word
  assert(fertility[0] <= 2 * curJ);

  //dictionary probs were handled above

  prob *= choose_factor_[curJ][fertility[0]];

  std::cerr << "mult by ldchoose " << choose_factor_[curJ][fertility[0]] << ", result: " << prob << std::endl;

  for (uint k = 1; k <= fertility[0]; k++) {
    prob *= p_zero_;

    std::cerr << "mult by p0 " << p_zero_ << ", result: " << prob << std::endl;
  }
  for (uint k = 1; k <= curJ - 2 * fertility[0]; k++) {
    prob *= p_nonzero_;

    std::cerr << "mult by p1 " << p_nonzero_ << ", result: " << prob << std::endl;
  }

  if (empty_word_model_ == FertNullOchNey) {

    for (uint k = 1; k <= fertility[0]; k++) {
      prob *= ((long double)k) / curJ;

      std::cerr << "mult by k/curJ = " << (((long double)k) / curJ) << ", result: " << prob << std::endl;
    }
  }
  else if (empty_word_model_ == FertNullIntra) {
    TODO("");
  }
}

//compact form
double IBM4Trainer::nondeficient_inter_m_step_energy(const IBM4CeptStartModel& singleton_count, const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
    const std::vector<double>& norm_weight, const IBM4CeptStartModel& param, uint sclass, uint tclass) const
{
  double energy = 0.0;

  assert(singleton_count.zDim() == param.zDim());

  // a) singleton terms
  for (uint diff = 0; diff < singleton_count.zDim(); diff++)
    energy -= singleton_count(sclass, tclass, diff) * std::log(param(sclass, tclass, diff));

  // b) normalization terms
  //NOTE: here we don't need to consider the displacement offset, it is included in the values in count already

  for (uint k = 0; k < norm_weight.size(); k++) {

    const Math1D::Vector<uchar,uchar>& open_diffs = open_pos[k];

    assert(open_diffs.size() > 0);
    if (open_diffs.size() == 0)
      continue;

    double weight = norm_weight[k];

    double sum = 0.0;
    for (uchar i = 0; i < open_diffs.size(); i++)
      sum += param(sclass, tclass, open_diffs[i]);

    energy += weight * std::log(sum);
  }

  return energy;
}

//compact form with interpolation
double IBM4Trainer::nondeficient_inter_m_step_energy(const Math1D::Vector<double>& singleton_count, const std::vector<double>& norm_weight,
    const Math1D::Vector<double>& param1, const Math1D::Vector<double>& param2, const Math1D::Vector<double>& sum1, const Math1D::Vector<double>& sum2,
    double lambda) const
{
  const double neg_lambda = 1.0 - lambda;

  double energy = 0.0;

  // a) singleton terms
  for (uint diff = 0; diff < singleton_count.size(); diff++) {

    const double cur_param = lambda * param2[diff] + neg_lambda * param1[diff];
    energy -= singleton_count[diff] * std::log(cur_param);
  }

  // b) normalization terms
  //NOTE: here we don't need to consider the displacement offset, it is included in the values in count already

  for (uint k = 0; k < norm_weight.size(); k++) {

    const double weight = norm_weight[k];

    double sum = lambda * sum2[k] + neg_lambda * sum1[k];

    energy += weight * std::log(sum);
  }

  return energy;
}

//compact form
void IBM4Trainer::nondeficient_inter_m_step_with_interpolation(const IBM4CeptStartModel& singleton_count,
    const std::vector<Math1D::Vector<uchar,uchar> >& open_diff, const std::vector<double>& norm_weight,
    uint sclass, uint tclass, double start_energy)
{
  const uint nParams = singleton_count.zDim();

  Math1D::Vector<double> relevant_singleton_count(nParams);
  for (uint diff = 0; diff < nParams; diff++)
    relevant_singleton_count[diff] = singleton_count(sclass, tclass, diff);

  double energy = start_energy;

  if (nSourceClasses_ * nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Math1D::Vector<double> cur_cept_start_prob(nParams);
  Math1D::Vector<double> gradient(nParams);

  Math1D::Vector<double> new_cept_start_prob(nParams);

  //test if normalizing the passed singleton counts gives a better starting point

  double rel_sum = relevant_singleton_count.sum();
  if (rel_sum > 1e-305) {

    IBM4CeptStartModel hyp_cept_start_prob(nSourceClasses_, nTargetClasses_, singleton_count.zDim(), MAKENAME(hyp_cept_start_prob));

    for (uint diff = 0; diff < nParams; diff++)
      hyp_cept_start_prob(sclass, tclass, diff) = std::max(fert_min_param_entry, relevant_singleton_count[diff] / rel_sum);

    double hyp_energy = nondeficient_inter_m_step_energy(singleton_count, open_diff,
                        norm_weight, hyp_cept_start_prob, sclass, tclass);

    if (hyp_energy < energy) {

      for (uint diff = 0; diff < nParams; diff++)
        cept_start_prob_(sclass, tclass, diff) = hyp_cept_start_prob(sclass, tclass, diff);

      if (nSourceClasses_ * nTargetClasses_ <= 4)
        std::cerr << "switching to passed normalized singleton count ---> " << hyp_energy << std::endl;

      energy = hyp_energy;
    }
  }

  double save_energy = energy;

  double alpha = 0.01;
  double line_reduction_factor = 0.35;

  for (uint k = 0; k < nParams; k++) {
    cur_cept_start_prob[k] = std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, k));
  }

  Math1D::Vector<double> sum(norm_weight.size());
  Math1D::Vector<double> new_sum(norm_weight.size(), 0.0);

  for (uint k = 0; k < norm_weight.size(); k++) {

    double cur_sum = 0.0;

    const Math1D::Vector<uchar,uchar>& open_diffs = open_diff[k];

    for (uchar i = 0; i < open_diffs.size(); i++)
      cur_sum += cur_cept_start_prob[open_diffs[i]];

    sum[k] = cur_sum;
  }

  for (uint iter = 1; iter <= nondef_dist_m_step_iter_; iter++) {

    if ((iter % 50) == 0) {
      if (nSourceClasses_ * nTargetClasses_ <= 4)
        std::cerr << "inter energy after iter #" << iter << ": " << energy << std::endl;

      if (save_energy - energy < 0.15)
        break;
      if (iter >= 100 && save_energy - energy < 0.5)
        break;

      save_energy = energy;
    }

    gradient.set_constant(0.0);

    /*** compute the gradient ***/

    // a) singleton terms
    for (uint diff = 0; diff < nParams; diff++) {

      const double cur_param = cur_cept_start_prob[diff];
      assert(cur_param >= fert_min_param_entry);
      gradient[diff] -= relevant_singleton_count[diff] / cur_param;
    }

    //b) normalization terms
    for (uint k = 0; k < norm_weight.size(); k++) {

      const Math1D::Vector<uchar,uchar>& open_diffs = open_diff[k];
      double weight = norm_weight[k];

      const uint size = open_diffs.size();

      const double addon = weight / sum[k];
      for (uchar i = 0; i < size; i++)
        gradient[open_diffs[i]] += addon;
    }

    /*** go in neg. gradient direction ***/
    Math1D::Vector<double> temp(nParams);

    for (uint i = 0; i < nParams; i++)
      new_cept_start_prob[i] = cur_cept_start_prob[i] - alpha * gradient[i];

    /*** reproject ***/
    projection_on_simplex(new_cept_start_prob.direct_access(), nParams, fert_min_param_entry);

    for (uint k = 0; k < norm_weight.size(); k++) {

      double cur_sum = 0.0;

      const Math1D::Vector<uchar,uchar>& open_diffs = open_diff[k];

      for (uchar i = 0; i < open_diffs.size(); i++)
        cur_sum += new_cept_start_prob[open_diffs[i]];

      new_sum[k] = cur_sum;
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

      double hyp_energy = nondeficient_inter_m_step_energy(relevant_singleton_count, norm_weight, cur_cept_start_prob,
                          new_cept_start_prob, sum, new_sum, lambda);

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
      if (nSourceClasses_ * nTargetClasses_ <= 4)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = 0; k < nParams; k++)
      cur_cept_start_prob[k] = std::max(fert_min_param_entry, best_lambda * new_cept_start_prob[k] + neg_best_lambda * cur_cept_start_prob[k]);

    for (uint k = 0; k < norm_weight.size(); k++) {

      sum[k] = best_lambda * new_sum[k] + neg_best_lambda * sum[k];
    }

    energy = best_energy;
  }

  //copy back
  for (uint k = 0; k < nParams; k++) {

    cept_start_prob_(sclass, tclass, k) = cur_cept_start_prob[k];
    assert(cept_start_prob_(sclass, tclass, k) >= fert_min_param_entry);
  }

}

void IBM4Trainer::nondeficient_inter_m_step_unconstrained(const IBM4CeptStartModel& singleton_count,
    const std::vector<Math1D::Vector<uchar,uchar> >& open_diff,
    const std::vector<double>& norm_weight, uint sclass, uint tclass, double start_energy, uint L)
{
  const uint nParams = singleton_count.zDim();

  Math1D::Vector<double> relevant_singleton_count(nParams);
  for (uint diff = 0; diff < nParams; diff++)
    relevant_singleton_count[diff] = singleton_count(sclass, tclass, diff);

  double energy = start_energy;

  if (nSourceClasses_ * nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Math1D::Vector<double> cur_cept_start_prob(nParams);
  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> hyp_work_param(nParams);
  Math1D::Vector<double> gradient(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);

  IBM4CeptStartModel hyp_cept_start_prob(nSourceClasses_, nTargetClasses_, nParams, MAKENAME(hyp_cept_start_prob));
  //test if normalizing the passed singleton counts gives a better starting point

  double rel_sum = relevant_singleton_count.sum();
  if (rel_sum > 1e-305) {

    for (uint diff = 0; diff < nParams; diff++)
      hyp_cept_start_prob(sclass, tclass, diff) = std::max(fert_min_param_entry, relevant_singleton_count[diff] / rel_sum);

    double hyp_energy = nondeficient_inter_m_step_energy(singleton_count, open_diff, norm_weight, hyp_cept_start_prob, sclass, tclass);

    if (hyp_energy < energy) {

      for (uint diff = 0; diff < nParams; diff++)
        cept_start_prob_(sclass, tclass, diff) = hyp_cept_start_prob(sclass, tclass, diff);

      if (nSourceClasses_ * nTargetClasses_ <= 4)
        std::cerr << "switching to passed normalized singleton count ---> " << hyp_energy << std::endl;

      energy = hyp_energy;
    }
  }

  for (uint k = 0; k < nParams; k++) {
    cur_cept_start_prob[k] = std::max(fert_min_param_entry, cept_start_prob_(sclass, tclass, k));
    work_param[k] = sqrt(cur_cept_start_prob[k]);
  }

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint iter = 1; iter <= nondef_dist_m_step_iter_; iter++) {

    if ((iter % 50) == 0) {
      if (nSourceClasses_ * nTargetClasses_ <= 4)
        std::cerr << "L-BFGS inter energy after iter #" << iter << ": " << energy << std::endl;
    }
    // a) calculate gradient w.r.t. the probabilities, not the parameters

    gradient.set_constant(0.0);

    // a1) singleton terms
    for (uint diff = 0; diff < nParams; diff++) {

      const double cur_param = cur_cept_start_prob[diff];
      assert(cur_param >= fert_min_param_entry);
      gradient[diff] -= relevant_singleton_count[diff] / cur_param;
    }

    // a2) normalization terms
    for (uint k = 0; k < norm_weight.size(); k++) {

      const Math1D::Vector<uchar,uchar>& open_diffs = open_diff[k];
      double weight = norm_weight[k];

      const uint size = open_diffs.size();

      double sum = 0.0;
      for (uchar i = 0; i < size; i++)
        sum += cur_cept_start_prob[open_diffs[i]];

      const double addon = weight / sum;
      for (uchar i = 0; i < size; i++)
        gradient[open_diffs[i]] += addon;
    }

    // b) now calculate the gradient for the actual parameters

    double scale = 0.0;
    for (uint k = 0; k < nParams; k++)
      scale += work_param[k] * work_param[k];

    std::cerr << "sqr sum: " << scale << std::endl;

    //above, we have calculated the gradient for a point scaled by 1.0 / scale.
    // by design of the gradient, we have to multiply with scale now to get the gradient we want

    for (uint k = 0; k < nParams; k++)
      work_grad[k] = scale * 2.0 * work_param[k] * gradient[k];

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

        double cur_alpha = 0.0;
        for (uint k = 0; k < nParams; k++) {
          cur_alpha += search_direction[k] * cur_step[k];
        }
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] -= cur_alpha * cur_grad_diff[k];
        }
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = 0.0;
        for (uint k = 0; k < nParams; k++) {
          beta += search_direction[k] * cur_grad_diff[k];
        }
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

      double sqr_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_work_param[k] * hyp_work_param[k];
      }

      for (uint k = 0; k < nParams; k++)
        hyp_cept_start_prob(sclass, tclass, k) = std::max(fert_min_param_entry, hyp_work_param[k] * hyp_work_param[k] / sqr_sum);

      double hyp_energy = nondeficient_inter_m_step_energy(singleton_count, open_diff,
                          norm_weight, hyp_cept_start_prob, sclass, tclass);

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

    double sqr_sum = 0.0;
    for (uint k = 0; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      sqr_sum += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    for (uint k = 0; k < nParams; k++)
      cur_cept_start_prob[k] = std::max(fert_min_param_entry, work_param[k] * work_param[k] / sqr_sum);
  }

  //copy back
  for (uint k = 0; k < nParams; k++) {

    cept_start_prob_(sclass, tclass, k) = cur_cept_start_prob[k];
    assert(cept_start_prob_(sclass, tclass, k) >= fert_min_param_entry);
  }
}

//compact form
double IBM4Trainer::nondeficient_intra_m_step_energy(const IBM4WithinCeptModel& singleton_count,
    const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count,
    const IBM4WithinCeptModel& param, uint sclass) const
{
  double energy = 0.0;

  // a) singleton terms
  for (uint k = 1; k < singleton_count.yDim(); k++) {

    const double cur_param = param(sclass, k);
    assert(cur_param >= fert_min_param_entry);

    energy -= singleton_count(sclass, k) * std::log(cur_param);
  }

  // b) normalization terms
  for (uint k = 0; k < count.size(); k++) {

    const Math1D::Vector<uchar,uchar>& open_diffs = count[k].first;
    if (open_diffs.size() == 0)
      continue;

    const double weight = count[k].second;

    double sum = 0.0;
    for (uchar i = 0; i < open_diffs.size(); i++)
      sum += param(sclass, open_diffs[i]);

    energy += weight * std::log(sum);
  }

  return energy;
}

//compact form
void IBM4Trainer::nondeficient_intra_m_step(const IBM4WithinCeptModel& singleton_count,
    const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count, uint sclass)
{
  const uint nParams = within_cept_prob_.yDim();

  for (uint k = 1; k < nParams; k++)
    within_cept_prob_(sclass, k) = std::max(fert_min_param_entry, within_cept_prob_(sclass, k));

  double energy = nondeficient_intra_m_step_energy(singleton_count, count, within_cept_prob_, sclass);

  if (nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Math1D::Vector<double> gradient(within_cept_prob_.yDim());

  Math1D::Vector<double> new_within_cept_prob(within_cept_prob_.yDim());
  IBM4WithinCeptModel hyp_within_cept_prob = within_cept_prob_;

  //test if normalizing the passed singleton count gives a better starting point
  double rel_sum = 0.0;
  for (uint k = 1; k < within_cept_prob_.yDim(); k++)
    rel_sum += singleton_count(sclass, k);

  if (rel_sum > 1e-305) {

    for (uint k = 1; k < within_cept_prob_.yDim(); k++)
      hyp_within_cept_prob(sclass, k) = std::max(fert_min_param_entry, singleton_count(sclass, k) / rel_sum);

    double hyp_energy = nondeficient_intra_m_step_energy(singleton_count, count, hyp_within_cept_prob, sclass);

    if (hyp_energy < energy) {

      for (uint k = 1; k < within_cept_prob_.yDim(); k++)
        within_cept_prob_(sclass, k) = hyp_within_cept_prob(sclass, k);

      if (nTargetClasses_ <= 4)
        std::cerr << "switching to passed normalized count ----> " << hyp_energy  << std::endl;

      energy = hyp_energy;
    }
  }

  double save_energy = energy;

  double alpha = 0.01;
  double line_reduction_factor = 0.35;

  for (uint iter = 1; iter <= 250 /*400 */ ; iter++) {

    gradient.set_constant(0.0);

    if ((iter % 50) == 0) {
      if (nTargetClasses_ <= 4)
        std::cerr << "L-BFGS intra energy after iter #" << iter << ": " << energy << std::endl;

      if (save_energy - energy < 0.15)
        break;
      if (iter >= 100 && save_energy - energy < 0.5)
        break;

      save_energy = energy;
    }

    /*** compute the gradient ***/

    // a) singleton terms
    for (uint k = 1; k < nParams; k++)
      gradient[k] -= singleton_count(sclass, k) / within_cept_prob_(sclass, k);

    // b) normalization terms
    for (uint k = 0; k < count.size(); k++) {

      const Math1D::Vector<uchar,uchar>& open_diffs = count[k].first;
      double weight = count[k].second;

      double sum = 0.0;
      for (uchar i = 0; i < open_diffs.size(); i++)
        sum += within_cept_prob_(sclass, open_diffs[i]);

      const double addon = weight / sum;
      for (uchar i = 0; i < open_diffs.size(); i++)
        gradient[open_diffs[i]] += addon;
    }

    /*** go in neg. gradient direction ***/
    for (uint i = 1; i < nParams; i++)
      new_within_cept_prob[i] = within_cept_prob_(sclass, i) - alpha * gradient[i];

    /*** reproject ***/
    projection_on_simplex(new_within_cept_prob.direct_access() + 1, gradient.size() - 1, fert_min_param_entry);

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

      for (uint k = 1; k < nParams; k++)
        hyp_within_cept_prob(sclass, k) =
          std::max(fert_min_param_entry, lambda * new_within_cept_prob[k] + neg_lambda * within_cept_prob_(sclass, k));

      double hyp_energy = nondeficient_intra_m_step_energy(singleton_count, count, hyp_within_cept_prob, sclass);

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
      if (nTargetClasses_ <= 4)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = 1; k < nParams; k++)
      within_cept_prob_(sclass, k) =
        std::max(fert_min_param_entry, best_lambda * new_within_cept_prob[k] + neg_best_lambda * within_cept_prob_(sclass, k));

    energy = best_energy;
  }

  //DEBUG
  for (uint k = 1; k < nParams; k++)
    assert(within_cept_prob_(sclass, k) >= fert_min_param_entry);
  //END_DEBUG
}

void IBM4Trainer::nondeficient_intra_m_step_unconstrained(const IBM4WithinCeptModel& singleton_count,
    const std::vector<std::pair<Math1D::Vector<uchar,uchar>, double> >& count, uint sclass, uint L)
{
  const uint nParams = within_cept_prob_.yDim();

  for (uint k = 1; k < within_cept_prob_.yDim(); k++)
    within_cept_prob_(sclass, k) = std::max(fert_min_param_entry, within_cept_prob_(sclass, k));

  double energy = nondeficient_intra_m_step_energy(singleton_count, count, within_cept_prob_, sclass);

  if (nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Math1D::Vector<double> gradient(nParams);
  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> hyp_work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);

  IBM4WithinCeptModel hyp_within_cept_prob = within_cept_prob_;

  //test if normalizing the passed singleton count gives a better starting point
  double rel_sum = 0.0;
  for (uint k = 1; k < within_cept_prob_.yDim(); k++)
    rel_sum += singleton_count(sclass, k);

  if (rel_sum > 1e-305) {

    for (uint k = 1; k < nParams; k++)
      hyp_within_cept_prob(sclass, k) = std::max(fert_min_param_entry, singleton_count(sclass, k) / rel_sum);

    double hyp_energy = nondeficient_intra_m_step_energy(singleton_count, count, hyp_within_cept_prob, sclass);

    if (hyp_energy < energy) {

      for (uint k = 1; k < within_cept_prob_.yDim(); k++)
        within_cept_prob_(sclass, k) = hyp_within_cept_prob(sclass, k);

      if (nTargetClasses_ <= 4)
        std::cerr << "switching to passed normalized count ----> " << hyp_energy << std::endl;

      energy = hyp_energy;
    }
  }

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint k = 1; k < nParams; k++)
    work_param[k] = sqrt(within_cept_prob_(sclass, k));

  for (uint iter = 1; iter <= nondef_dist_m_step_iter_; iter++) {

    if ((iter % 50) == 0) {
      if (nTargetClasses_ <= 4)
        std::cerr << "intra energy after iter #" << iter << ": " << energy << std::endl;
    }
    // a) calculate gradient w.r.t. the probabilities, not the parameters

    gradient.set_constant(0.0);

    // a) singleton terms
    for (uint k = 1; k < nParams; k++)
      gradient[k] -= singleton_count(sclass, k) / within_cept_prob_(sclass, k);

    // b) normalization terms
    for (uint k = 0; k < count.size(); k++) {

      const Math1D::Vector<uchar,uchar>& open_diffs = count[k].first;
      double weight = count[k].second;

      double sum = 0.0;
      for (uchar i = 0; i < open_diffs.size(); i++)
        sum += within_cept_prob_(sclass, open_diffs[i]);

      const double addon = weight / sum;
      for (uchar i = 0; i < open_diffs.size(); i++)
        gradient[open_diffs[i]] += addon;
    }

    // b) now calculate the gradient for the actual parameters

    double scale = 0.0;
    for (uint k = 0; k < nParams; k++)
      scale += work_param[k] * work_param[k];

    std::cerr << "sqr sum: " << scale << std::endl;

    //above, we have calculated the gradient for a point scaled by 1.0 / scale.
    // by design of the gradient, we have to multiply with scale now to get the gradient we want

    for (uint k = 0; k < nParams; k++)
      work_grad[k] = scale * 2.0 * work_param[k] * gradient[k];

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

        double cur_alpha = 0.0;
        for (uint k = 0; k < nParams; k++) {
          cur_alpha += search_direction[k] * cur_step[k];
        }
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] -= cur_alpha * cur_grad_diff[k];
        }
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = 0.0;
        for (uint k = 0; k < nParams; k++) {
          beta += search_direction[k] * cur_grad_diff[k];
        }
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

      double sqr_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_work_param[k] * hyp_work_param[k];
      }

      for (uint k = 1; k < nParams; k++)
        hyp_within_cept_prob(sclass, k) = std::max(fert_min_param_entry, hyp_work_param[k] * hyp_work_param[k] / sqr_sum);

      double hyp_energy = nondeficient_intra_m_step_energy(singleton_count, count, hyp_within_cept_prob, sclass);

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

    double sqr_sum = 0.0;
    for (uint k = 1; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      sqr_sum += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    for (uint k = 1; k < nParams; k++)
      within_cept_prob_(sclass, k) = std::max(fert_min_param_entry,work_param[k] * work_param[k] / sqr_sum);
  }
}

long double IBM4Trainer::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
    uint& nIter, Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob,
    Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment) const
{
  if (nondeficient_) {
    return nondeficient_hillclimbing(source, target, lookup, nIter, fertility, expansion_prob, swap_prob, alignment);
  }

  const double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size();

  Math1D::Vector<uint> sclass(curJ);
  Math1D::Vector<uint> tclass(curI);

  for (uint j = 0; j < curJ; j++)
    sclass[j] = source_class_[source[j]];
  for (uint i = 0; i < curI; i++)
    tclass[i] = target_class_[target[i]];

  //std::cerr << "*************** hillclimb: J = " << curJ << ", I=" << curI << std::endl;
  //std::cerr << "start alignment: " << alignment << std::endl;

  fertility.resize(curI + 1);

  Math2D::Matrix<double> dict(curJ,curI+1);
  compute_dictmat_fertform(source, lookup, target, dict_, dict);

  long double base_prob = alignment_prob(source, target, lookup, alignment);

  //DEBUG
  if (isnan(base_prob) || isinf(base_prob) || base_prob <= 0.0) {
    std::cerr << "ERROR: base_prob in hillclimbing is " << base_prob << std::endl;
    print_alignment_prob_factors(source, target, lookup, alignment);
    exit(1);
  }
  //END_DEBUG

  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);
  //swap_prob.set_constant(0.0);
  //expansion_prob.set_constant(0.0);

  uint count_iter = 0;

  const Math3D::Tensor<float>& cur_intra_distortion_prob = intra_distortion_prob_[curJ];
  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  //source words are listed in ascending order
  NamedStorage1D<std::vector<AlignBaseType> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));

  fertility.set_constant(0);
  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    aligned_source_words[aj].push_back(j);
  }

  while (true) {

    const long double base_inter_distortion_prob = inter_distortion_prob(source, target, aligned_source_words);
    const long double base_null_distortion_prob = null_distortion_prob(aligned_source_words[0], curJ);

    Math1D::NamedVector<uint> prev_cept(curI + 1, MAX_UINT, MAKENAME(prev_cept));
    Math1D::NamedVector<uint> next_cept(curI + 1, MAX_UINT, MAKENAME(next_cept));
    Math1D::NamedVector<uint> cept_center(curI + 1, MAX_UINT, MAKENAME(cept_center));

    uint prev_i = MAX_UINT;
    for (uint i = 1; i <= curI; i++) {

      if (fertility[i] > 0) {

        const std::vector<AlignBaseType>& cur_aligned_source_words = aligned_source_words[i];

        prev_cept[i] = prev_i;
        if (prev_i != MAX_UINT)
          next_cept[prev_i] = i;

        switch (cept_start_mode_) {
        case IBM4CENTER: {
          double sum_j = vec_sum(cur_aligned_source_words);
          cept_center[i] = (int)round(sum_j / cur_aligned_source_words.size());
          break;
        }
        case IBM4FIRST:
          cept_center[i] = cur_aligned_source_words[0];
          break;
        case IBM4LAST:
          cept_center[i] = cur_aligned_source_words.back();
          break;
        case IBM4UNIFORM:
          break;
        default:
          assert(false);
        }

        prev_i = i;
      }
    }

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

    //std::clock_t tStartExp,tEndExp;
    //tStartExp = std::clock();

    //a) expansion moves

    NamedStorage1D<std::vector<ushort> > hyp_aligned_source_words(MAKENAME(hyp_aligned_source_words));
    hyp_aligned_source_words = aligned_source_words;

    for (uint j = 0; j < curJ; j++) {

      const uint aj = alignment[j];
      const std::vector<ushort>& aligned_aj = aligned_source_words[aj];

      const uint aj_fert = fertility[aj];
      assert(aj_fert > 0);
      expansion_prob(j, aj) = 0.0;

      //const uint s_idx = source[j];
      const uint j_class = sclass[j];

      const uint prev_ti = (aj == 0) ? 0 : target[aj - 1];
      const Math1D::Vector<double>& fert_prob_prev_ti = fertility_prob_[prev_ti];
      const uint prev_ti_class = (aj == 0) ? MAX_UINT : tclass[aj - 1];
      const double old_dict_prob = dict(j,aj);

      long double leaving_prob_common = old_dict_prob;
      long double incoming_prob_common = 1.0;
      if (aj != 0) {
        leaving_prob_common *= fert_prob_prev_ti[aj_fert];
        if (!no_factorial_)
          leaving_prob_common *= aj_fert;
        incoming_prob_common = fert_prob_prev_ti[aj_fert - 1];
      }

      const uint prev_aj_fert = aj_fert;

      const uint prev_i = prev_cept[aj];
      const uint next_i = next_cept[aj];

      //std::cerr << "j: " << j << ", aj: " << aj << std::endl;

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        if (cand_aj != aj) {

          //std::cerr << "cand_aj: " << cand_aj << std::endl;

          long double hyp_prob = 0.0;

          bool incremental_calculation = false;

          const uint new_ti = (cand_aj == 0) ? 0 : target[cand_aj - 1];
          const Math1D::Vector<double>& fert_prob_new_ti = fertility_prob_[new_ti];
          const uint new_ti_class = (cand_aj == 0) ? MAX_UINT : tclass[cand_aj - 1];
          const double new_dict_prob = dict(j,cand_aj);
          const uint cand_aj_fert = fertility[cand_aj];

#if 0
          //EXPERIMENTAL (prune constellations with very unlikely translation probs.)
          if (new_dict_prob < 1e-10) {
            expansion_prob(j, cand_aj) = 0.0;
            continue;
          }
          //END_EXPERIMENTAL
#endif

          if (cand_aj != 0 && (cand_aj_fert + 1) > fertility_limit_[new_ti]) {

            expansion_prob(j, cand_aj) = 0.0;
            continue;
          }
          else if (cand_aj == 0 && curJ < 2 * fertility[0] + 2) {

            expansion_prob(j, cand_aj) = 0.0;
            continue;
          }

          const std::vector<ushort>& aligned_cand_aj = aligned_source_words[cand_aj];

          long double incoming_prob = new_dict_prob * incoming_prob_common;
          long double leaving_prob = leaving_prob_common;

          if (cand_aj != 0) {
            incoming_prob *= fert_prob_new_ti[cand_aj_fert + 1];
            if (!no_factorial_)
              incoming_prob *= cand_aj_fert + 1;
            leaving_prob *= fert_prob_new_ti[cand_aj_fert];
          }

          assert(leaving_prob > 0.0);

          if (aj != 0 && cand_aj != 0) {

            if (next_i != MAX_UINT && (((prev_i != MAX_UINT && cand_aj < prev_i) || (prev_i == MAX_UINT && cand_aj > next_i))
                                       || ((next_i != MAX_UINT && cand_aj > next_i) || (next_i == MAX_UINT && prev_i != MAX_UINT && cand_aj < prev_i)))) {

              incremental_calculation = true;

              /***************************** 1. changes regarding aj ******************************/
              if (prev_aj_fert > 1) {
                //the cept aj remains

                uint jnum;
                for (jnum = 0; jnum < prev_aj_fert; jnum++) {
                  if (aligned_aj[jnum] == j)
                    break;
                }

                //std::cerr << "jnum: " << jnum << std::endl;

                assert(jnum < aligned_aj.size());

                //calculate new center of aj
                uint new_aj_center = MAX_UINT;
                switch (cept_start_mode_) {
                case IBM4CENTER: {
                  double sum_j = vec_sum(aligned_aj) - aligned_aj[jnum];
                  new_aj_center = (int)round(sum_j / (aligned_aj.size() - 1));
                  //std::cerr << "new_aj_center: " << new_aj_center << std::endl;
                  break;
                }
                case IBM4FIRST: {
                  if (jnum == 0)
                    new_aj_center = aligned_aj[1];
                  else {
                    new_aj_center = aligned_aj[0];
                    assert(new_aj_center == cept_center[aj]);
                  }
                  break;
                }
                case IBM4LAST: {
                  if (jnum + 1 == prev_aj_fert)
                    new_aj_center = aligned_aj[prev_aj_fert - 2];
                  else {
                    new_aj_center = aligned_aj[prev_aj_fert - 1];
                    assert(new_aj_center == cept_center[aj]);
                  }
                  break;
                }
                case IBM4UNIFORM:
                  break;
                default:
                  assert(false);
                }

                //std::cerr << "old aj center: " << cept_center[aj] << std::endl;
                //std::cerr << "new_aj_center: " << new_aj_center << std::endl;
                //std::cerr << "prev_i: " << prev_i << ", next_i: " << next_i << std::endl;

                //re-calculate the transition aj -> next_i
                if (next_i != MAX_UINT && new_aj_center != cept_center[aj]) {
                  const uint j0 = aligned_source_words[next_i][0];
                  const uint sc = sclass[j0];
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? prev_ti_class : tclass[next_i - 1];

                  leaving_prob *= inter_distortion_prob(j0, cept_center[aj], sc, tc, curJ);

                  assert(leaving_prob > 0.0);

                  incoming_prob *= inter_distortion_prob(j0, new_aj_center, sc, tc, curJ);
                }

                if (jnum == 0) {
                  //the transition prev_i -> aj is affected

                  const uint j1 = aligned_aj[1];

                  if (prev_i != MAX_UINT) {
                    const uint old_sclass = j_class;
                    const uint j1 = aligned_aj[1];
                    const uint new_sclass = sclass[j1];

                    const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_i - 1] : prev_ti_class;

                    leaving_prob *= inter_distortion_prob(j, cept_center[prev_i], old_sclass, tc, curJ);
                    incoming_prob *= inter_distortion_prob(j1, cept_center[prev_i], new_sclass, tc, curJ);
                  }
                  else if (use_sentence_start_prob_) {
                    leaving_prob *= cur_sentence_start_prob[j];
                    incoming_prob *= cur_sentence_start_prob[aligned_aj[1]];
                  }

                  assert(leaving_prob > 0.0);

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[j1] : prev_ti_class;

                  leaving_prob *= cur_intra_distortion_prob(cur_class, j1, j);

                  assert(leaving_prob > 0.0);
                }
                else {

                  const uint j_prev = aligned_aj[jnum - 1];
                  const uint j_cur = aligned_aj[jnum];

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[j_cur] : prev_ti_class;

                  leaving_prob *= cur_intra_distortion_prob(cur_class, j_cur, j_prev);

                  assert(leaving_prob > 0.0);

                  if (jnum + 1 < prev_aj_fert) {

                    const uint j_next = aligned_aj[jnum + 1];

                    const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[aligned_aj[jnum + 1]] : prev_ti_class;

                    leaving_prob *= cur_intra_distortion_prob(cur_class, j_next, j_cur);

                    assert(leaving_prob > 0.0);

                    incoming_prob *= cur_intra_distortion_prob(cur_class, j_next, j_prev);
                  }
                }

              }
              else {
                //the cept aj vanishes

                //erase the transitions prev_i -> aj    and    aj -> next_i
                if (prev_i != MAX_UINT) {

                  const uint sc = j_class;
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_i - 1] : prev_ti_class;

                  leaving_prob *= inter_distortion_prob(j, cept_center[prev_i], sc, tc, curJ);

                  assert(leaving_prob > 0.0);
                }
                else if (use_sentence_start_prob_) {
                  leaving_prob *= cur_sentence_start_prob[j];

                  assert(leaving_prob > 0.0);
                }

                const uint j0 = aligned_source_words[next_i][0];

                if (next_i != MAX_UINT) {
                  const uint sc = sclass[j0];
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? prev_ti_class : tclass[next_i - 1];

                  leaving_prob *= inter_distortion_prob(j0, j, sc, tc, curJ);

                  assert(leaving_prob > 0.0);
                }
                //introduce the transition prev_i -> next_i
                if (prev_i != MAX_UINT) {
                  if (next_i != MAX_UINT) {
                    const uint sc = sclass[j0];
                    const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_i - 1] : tclass[next_i - 1];

                    incoming_prob *= inter_distortion_prob(j0, cept_center[prev_i], sc, tc, curJ);
                  }
                }
                else if (use_sentence_start_prob_)
                  incoming_prob *= cur_sentence_start_prob[j0];
              }

              /********************** 2. changes regarding cand_aj **********************/
              uint cand_prev_i = MAX_UINT;
              for (uint k = cand_aj - 1; k > 0; k--) {
                if (fertility[k] > 0) {
                  cand_prev_i = k;
                  break;
                }
              }
              uint cand_next_i = MAX_UINT;
              for (uint k = cand_aj + 1; k <= curI; k++) {
                if (fertility[k] > 0) {
                  cand_next_i = k;
                  break;
                }
              }

              //std::cerr << "cand_prev_i: " << cand_prev_i << std::endl;
              //std::cerr << "cand_next_i: " << cand_next_i << std::endl;

              if (fertility[cand_aj] > 0) {
                //the cept cand_aj was already there

                uint insert_pos = 0;
                for (; insert_pos < fertility[cand_aj] && aligned_cand_aj[insert_pos] < j; insert_pos++) {
                  //empty body
                }

                //std::cerr << "insert_pos: " << insert_pos << std::endl;

                if (insert_pos == 0) {

                  if (cand_prev_i == MAX_UINT) {

                    const uint j0 = aligned_cand_aj[0];

                    if (use_sentence_start_prob_) {
                      leaving_prob *= cur_sentence_start_prob[j0];
                      incoming_prob *= cur_sentence_start_prob[j];

                      assert(leaving_prob > 0.0);
                    }

                    const uint cur_class =
                      (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[j0] : new_ti_class;

                    incoming_prob *= cur_intra_distortion_prob(cur_class, j0, j);
                  }
                  else {

                    const uint j0 = aligned_cand_aj[0];

                    const uint old_sclass = sclass[j0];
                    const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[cand_prev_i - 1] : new_ti_class;

                    leaving_prob *= inter_distortion_prob(j0, cept_center[cand_prev_i], old_sclass, tc, curJ);

                    assert(leaving_prob > 0.0);

                    const uint new_sclass = j_class;

                    incoming_prob *= inter_distortion_prob(j, cept_center[cand_prev_i], new_sclass, tc, curJ);

                    const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? old_sclass : new_ti_class;

                    incoming_prob *= cur_intra_distortion_prob(cur_class, j0, j);
                  }
                }
                else if (insert_pos < fertility[cand_aj]) {

                  const uint j_prev = aligned_cand_aj[insert_pos - 1];
                  const uint j_cur = aligned_cand_aj[insert_pos];

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[j_cur] : new_ti_class;

                  leaving_prob *= cur_intra_distortion_prob(cur_class, j_cur, j_prev);

                  assert(leaving_prob > 0.0);

                  const uint new_sclass = (intra_dist_mode_ == IBM4IntraDistModeSource) ? j_class : cur_class;

                  incoming_prob *= cur_intra_distortion_prob(new_sclass, j, j_prev);
                  incoming_prob *= cur_intra_distortion_prob(cur_class, j_cur, j);
                }
                else {
                  //insert at the end
                  assert(insert_pos == fertility[cand_aj]);

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? j_class : new_ti_class;

                  incoming_prob *= cur_intra_distortion_prob(cur_class, j, aligned_cand_aj[insert_pos - 1]);

                  //std::cerr << "including prob. " << cur_intra_distortion_prob(tclass,j,aligned_cand_aj[insert_pos-1])
                  //   << std::endl;
                }

                if (cand_next_i != MAX_UINT) {
                  //calculate new center of cand_aj

                  uint new_cand_aj_center = MAX_UINT;
                  switch (cept_start_mode_) {
                  case IBM4CENTER: {
                    double sum_j = j + vec_sum(aligned_cand_aj);
                    new_cand_aj_center = (int)round(sum_j / (fertility[cand_aj] + 1));
                    break;
                  }
                  case IBM4FIRST: {
                    if (insert_pos == 0)
                      new_cand_aj_center = j;
                    else
                      new_cand_aj_center = cept_center[cand_aj];
                    break;
                  }
                  case IBM4LAST: {
                    if (insert_pos >= fertility[cand_aj])
                      new_cand_aj_center = j;
                    else
                      new_cand_aj_center = cept_center[cand_aj];
                    break;
                  }
                  case IBM4UNIFORM:
                    break;
                  default:
                    assert(false);
                  }             //end of switch-statement

                  if (new_cand_aj_center != cept_center[cand_aj]
                      && cept_center[cand_aj] != new_cand_aj_center) {
                    const uint j0 = aligned_source_words[cand_next_i][0];
                    const uint sc = sclass[j0];
                    const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? new_ti_class : tclass[cand_next_i - 1];

                    leaving_prob *= inter_distortion_prob(j0, cept_center[cand_aj], sc, tc, curJ);

                    assert(leaving_prob > 0.0);

                    incoming_prob *= inter_distortion_prob(j0, new_cand_aj_center, sc, tc, curJ);
                  }
                }
              }
              else {
                //the cept cand_aj is newly created

                //erase the transition cand_prev_i -> cand_next_i (if existent)
                if (cand_prev_i != MAX_UINT && cand_next_i != MAX_UINT) {

                  const uint j0 = aligned_source_words[cand_next_i][0];
                  const uint sc = sclass[j0];
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[cand_prev_i - 1] : tclass[cand_next_i - 1];

                  leaving_prob *= inter_distortion_prob(j0, cept_center[cand_prev_i], sc, tc, curJ);

                  assert(leaving_prob > 0.0);
                }
                else if (cand_prev_i == MAX_UINT) {

                  assert(cand_next_i != MAX_UINT);
                  if (use_sentence_start_prob_)
                    leaving_prob *= cur_sentence_start_prob[aligned_source_words[cand_next_i][0]];

                  assert(leaving_prob > 0.0);
                }
                else {
                  //nothing to do here
                }

                //introduce the transitions cand_prev_i -> cand_aj    and   cand_aj -> cand_next_i
                if (cand_prev_i != MAX_UINT) {
                  const uint sc = j_class;
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[cand_prev_i - 1] : new_ti_class;

                  incoming_prob *= inter_distortion_prob(j, cept_center[cand_prev_i], sc, tc, curJ);
                }
                else
                  incoming_prob *= cur_sentence_start_prob[j];

                if (cand_next_i != MAX_UINT) {
                  const uint j0 = aligned_source_words[cand_next_i][0];
                  const uint sc = sclass[j0];
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? new_ti_class : tclass[cand_next_i - 1];

                  incoming_prob *= inter_distortion_prob(j0, j, sc, tc, curJ);
                }
              }

              assert(leaving_prob > 0.0);

              hyp_prob = base_prob * incoming_prob / leaving_prob;

              if (isnan(hyp_prob)) {

                std::cerr << "######in exp. move for j=" << j << " cur aj: " << aj << ", candidate: " << cand_aj << std::endl;
                std::cerr << "hyp_prob: " << hyp_prob << std::endl;
                std::cerr << "base: " << base_prob << ", incoming: " << incoming_prob << ", leaving: " << leaving_prob << std::endl;
                std::cerr << "hc iter: " << count_iter << std::endl;
                exit(1);
              }
#ifndef NDEBUG
              //DEBUG
              Math1D::Vector < AlignBaseType > hyp_alignment = alignment;
              hyp_alignment[j] = cand_aj;
              long double check_prob = alignment_prob(source, target, lookup, hyp_alignment);

              if (check_prob > 0.0) {

                long double check_ratio = hyp_prob / check_prob;

                if (!(check_ratio > 0.99 && check_ratio < 1.01)) {

                  std::cerr << "****************************************************************" << std::endl;
                  std::cerr << "expansion: moving j=" << j << " to cand_aj=" << cand_aj << " from aj=" << aj << std::endl;
                  std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
                  std::cerr << "base alignment: " << alignment << std::endl;
                  std::cerr << "actual prob: " << check_prob << std::endl;
                  std::cerr << "incremental hyp_prob: " << hyp_prob << std::endl;
                  std::cerr << "(base_prob: " << base_prob << ")" << std::endl;
                  std::cerr << "################################################################" << std::endl;
                }

                if (check_prob > 1e-15 * base_prob)
                  assert(check_ratio > 0.99 && check_ratio < 1.01);
              }
              //END_DEBUG
#endif
            }
          }                     //end -- if (aj != 0 && cand_aj != 0)
          else if (cand_aj == 0) {
            //NOTE: this time we also handle the cases where next_i == MAX_UINT or where prev_i == MAX_UINT

            incremental_calculation = true;

            assert(aj != 0);

            const uint prev_zero_fert = fertility[0];
            const uint new_zero_fert = prev_zero_fert + 1;

            if (curJ < 2 * new_zero_fert) {
              hyp_prob = 0.0;
            }
            else {

              incoming_prob *= (curJ - 2 * prev_zero_fert) * (curJ - 2 * prev_zero_fert - 1) * p_zero_;
              leaving_prob *= ((curJ - prev_zero_fert) * new_zero_fert * p_nonzero_ * p_nonzero_);

              if (empty_word_model_ == FertNullOchNey)
                incoming_prob *= (prev_zero_fert + 1) / ((long double)curJ);
              else if (empty_word_model_ == FertNullIntra) {

                std::vector<AlignBaseType> hyp_null_words = aligned_source_words[0];
                hyp_null_words.push_back(j);
                vec_sort(hyp_null_words);

                incoming_prob *= null_distortion_prob(hyp_null_words, curJ);
                leaving_prob *= base_null_distortion_prob;
              }

              if (prev_aj_fert > 1) {
                //the cept aj remains

                uint jnum;
                for (jnum = 0; jnum < prev_aj_fert; jnum++) {
                  if (aligned_aj[jnum] == j)
                    break;
                }

                //std::cerr << "jnum: " << jnum << std::endl;

                assert(jnum < aligned_aj.size());

                if (next_i != MAX_UINT) {
                  //calculate new center of aj
                  uint new_aj_center = MAX_UINT;
                  switch (cept_start_mode_) {
                  case IBM4CENTER: {
                    double sum_j = vec_sum(aligned_aj) - aligned_aj[jnum];
                    new_aj_center = (uint) round(sum_j / (aligned_aj.size() - 1));
                    break;
                  }
                  case IBM4FIRST: {
                    if (jnum == 0)
                      new_aj_center = aligned_aj[1];
                    else {
                      new_aj_center = aligned_aj[0];
                      assert(new_aj_center == cept_center[aj]);
                    }
                    break;
                  }
                  case IBM4LAST: {
                    if (jnum + 1 == prev_aj_fert)
                      new_aj_center = aligned_aj[prev_aj_fert - 2];
                    else {
                      new_aj_center = aligned_aj[prev_aj_fert - 1];
                      assert(new_aj_center == cept_center[aj]);
                    }
                    break;
                  }
                  case IBM4UNIFORM:
                    break;
                  default:
                    assert(false);
                  }

                  //            std::cerr << "old aj center: " << cept_center[aj] << std::endl;
                  //            std::cerr << "new_aj_center: " << new_aj_center << std::endl;
                  //            std::cerr << "prev_i: " << prev_i << ", next_i: " << next_i << std::endl;

                  //re-calculate the transition aj -> next_i
                  if (cept_center[aj] != new_aj_center) {
                    const uint sc = source_class_[source[aligned_source_words[next_i][0]]];
                    const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? prev_ti_class : tclass[next_i - 1];

                    leaving_prob *= inter_distortion_prob(aligned_source_words[next_i][0], cept_center[aj], sc, tc, curJ);
                    incoming_prob *= inter_distortion_prob(aligned_source_words[next_i][0], new_aj_center, sc, tc, curJ);
                  }
                }

                if (jnum == 0) {
                  //the transition prev_i -> aj is affected

                  if (prev_i != MAX_UINT) {
                    const uint old_sclass = j_class;

                    const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_i - 1] : prev_ti_class;
                    leaving_prob *= inter_distortion_prob(j, cept_center[prev_i], old_sclass, tc, curJ);
                    const uint new_sclass = sclass[aligned_aj[1]];
                    incoming_prob *= inter_distortion_prob(aligned_aj[1], cept_center[prev_i], new_sclass, tc, curJ);
                  }
                  else if (use_sentence_start_prob_) {
                    leaving_prob *= sentence_start_prob_[curJ][j];
                    incoming_prob *= sentence_start_prob_[curJ][aligned_aj[1]];
                  }

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[aligned_aj[1]] :  prev_ti_class;

                  leaving_prob *= cur_intra_distortion_prob(cur_class, aligned_aj[1], j);
                }
                else {

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[aligned_aj[jnum]] : prev_ti_class;

                  leaving_prob *= cur_intra_distortion_prob(cur_class, aligned_aj[jnum], aligned_aj[jnum - 1]);

                  if (jnum + 1 < prev_aj_fert) {

                    const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[aligned_aj[jnum + 1]] : prev_ti_class;

                    leaving_prob *= cur_intra_distortion_prob(cur_class, aligned_aj[jnum + 1], aligned_aj[jnum]);
                    incoming_prob *= cur_intra_distortion_prob(cur_class, aligned_aj[jnum + 1], aligned_aj[jnum - 1]);
                  }
                }
              }
              else {
                //the cept aj vanishes

                //erase the transitions prev_i -> aj    and    aj -> next_i
                if (prev_i != MAX_UINT) {
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_i - 1] : prev_ti_class;
                  const uint sc = j_class;

                  leaving_prob *= inter_distortion_prob(j, cept_center[prev_i], sc, tc, curJ);
                }
                else if (use_sentence_start_prob_) {
                  leaving_prob *= sentence_start_prob_[curJ][j];
                }

                if (next_i != MAX_UINT) {
                  const uint sc = sclass[aligned_source_words[next_i][0]];
                  const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? prev_ti_class : tclass[next_i - 1];

                  leaving_prob *= inter_distortion_prob(aligned_source_words[next_i][0], j, sc, tc, curJ);

                  //introduce the transition prev_i -> next_i
                  if (prev_i != MAX_UINT) {

                    const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_i - 1] : tclass[next_i - 1];
                    incoming_prob *= inter_distortion_prob(aligned_source_words[next_i][0], cept_center[prev_i], sc, tc, curJ);
                  }
                  else if (use_sentence_start_prob_) {
                    incoming_prob *= sentence_start_prob_[curJ][aligned_source_words[next_i][0]];
                  }
                }
              }

              assert(leaving_prob > 0.0);

              hyp_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
              //DEBUG
              Math1D::Vector < AlignBaseType > hyp_alignment = alignment;
              hyp_alignment[j] = cand_aj;
              long double check_prob = alignment_prob(source, target, lookup, hyp_alignment);

              if (check_prob != 0.0) {

                long double check_ratio = hyp_prob / check_prob;

                if (!(check_ratio > 0.99 && check_ratio < 1.01)) {
                  //if (true) {

                  std::cerr << "incremental prob: " << hyp_prob << std::endl;
                  std::cerr << "actual prob: " << check_prob << std::endl;

                  std::cerr << ", J=" << curJ << ", I=" << curI << std::endl;
                  std::cerr << "base alignment: " << alignment << std::endl;
                  std::cerr << "moving source word " << j << " from " << alignment[j] << " to 0" << std::endl;
                }

                if (check_prob > 1e-12 * best_prob)
                  assert(check_ratio > 0.99 && check_ratio < 1.01);
              }
              //END_DEBUG
#endif
            }
          }

          if (!incremental_calculation) {

            vec_erase(hyp_aligned_source_words[aj], (ushort) j);

            hyp_aligned_source_words[cand_aj].push_back(j);
            vec_sort(hyp_aligned_source_words[cand_aj]);

            hyp_prob = base_prob * inter_distortion_prob(source, target, hyp_aligned_source_words) / base_inter_distortion_prob;
            if (aj != 0) {
              hyp_prob *= intra_distortion_prob(hyp_aligned_source_words[aj], sclass, tclass[aj - 1])
                          / intra_distortion_prob(aligned_source_words[aj], sclass, tclass[aj - 1]);
            }
            if (cand_aj != 0) {
              hyp_prob *= intra_distortion_prob(hyp_aligned_source_words[cand_aj], sclass, tclass[cand_aj - 1])
                          / intra_distortion_prob(aligned_source_words[cand_aj], sclass, tclass[cand_aj - 1]);
            }

            assert(cand_aj != 0);       //since we handle that case above

            uint prev_zero_fert = fertility[0];
            uint new_zero_fert = prev_zero_fert;

            if (aj == 0) {
              new_zero_fert--;
            }
            else if (cand_aj == 0) {
              new_zero_fert++;
            }

            if (prev_zero_fert < new_zero_fert) {

              incoming_prob *= (curJ - 2 * prev_zero_fert) * (curJ - 2 * prev_zero_fert - 1) * p_zero_;
              leaving_prob *= ((curJ - prev_zero_fert) * new_zero_fert * p_nonzero_ * p_nonzero_);

              if (empty_word_model_ == FertNullOchNey)
                incoming_prob *= (prev_zero_fert + 1) / ((long double)curJ);
              else if (empty_word_model_ == FertNullIntra) {

                std::vector<AlignBaseType> hyp_null_words = aligned_source_words[0];
                hyp_null_words.push_back(j);
                vec_sort(hyp_null_words);

                incoming_prob *= null_distortion_prob(hyp_null_words, curJ);
                leaving_prob *= base_null_distortion_prob;
              }
            }
            else if (prev_zero_fert > new_zero_fert) {

              incoming_prob *= (curJ - new_zero_fert) * prev_zero_fert * p_nonzero_ * p_nonzero_;
              leaving_prob *= ((curJ - 2 * prev_zero_fert + 1) * (curJ - 2 * new_zero_fert) * p_zero_);

              if (empty_word_model_ == FertNullOchNey)
                incoming_prob *= curJ / ((long double)prev_zero_fert);
              else if (empty_word_model_ == FertNullIntra) {

                std::vector<AlignBaseType> hyp_null_words = aligned_source_words[0];
                vec_erase(hyp_null_words, (AlignBaseType) j);

                incoming_prob *= null_distortion_prob(hyp_null_words, curJ);
                leaving_prob *= base_null_distortion_prob;
              }
            }

            hyp_prob *= incoming_prob / leaving_prob;

            //restore for next loop execution
            hyp_aligned_source_words[aj] = aligned_aj;
            hyp_aligned_source_words[cand_aj] = aligned_cand_aj;

            //DEBUG
            if (isnan(hyp_prob)) {
              std::cerr << "incoming: " << incoming_prob << std::endl;
              std::cerr << "leaving: " << leaving_prob << std::endl;
            }
            //END_DEBUG

#ifndef NDEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j] = cand_aj;

            long double check = alignment_prob(source, target, lookup, hyp_alignment);

            if (check > 1e-250) {

              if (!(check / hyp_prob <= 1.005 && check / hyp_prob >= 0.995)) {
                std::cerr << "j: " << j << ", aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
                std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
                std::cerr << "base alignment: " << alignment << std::endl;
                std::cerr << "no_factorial_: " << no_factorial_ << std::endl;
                std::cerr << "prev_zero_fert: " << prev_zero_fert << ", new_zero_fert: " << new_zero_fert << std::endl;
              }

              assert(check / hyp_prob <= 1.005);
              assert(check / hyp_prob >= 0.995);
            }
            else if (check > 0.0) {

              if (!(check / hyp_prob <= 1.5 && check / hyp_prob >= 0.666)) {
                std::cerr << "aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
                std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
              }

              assert(check / hyp_prob <= 1.5);
              assert(check / hyp_prob >= 0.666);

            }
            else
              assert(hyp_prob == 0.0);
#endif
          }

          expansion_prob(j, cand_aj) = hyp_prob;

          if (isnan(expansion_prob(j, cand_aj))
              || isinf(expansion_prob(j, cand_aj))) {

            std::cerr << "nan/inf in exp. move for j=" << j << ", " << aj << " -> " << cand_aj << ": " << expansion_prob(j,  cand_aj) << std::endl;
            std::cerr << "current alignment: " << aj << std::endl;
            std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
            std::cerr << "incremental calculation: " << incremental_calculation << std::endl;

            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j] = cand_aj;
            std::cerr << "prob. of start alignment: " << alignment_prob(source, target, lookup, alignment) << std::endl;
            std::cerr << "check prob: " << alignment_prob(source, target, lookup, hyp_alignment) << std::endl;

            print_alignment_prob_factors(source, target, lookup, alignment);
          }

          assert(!isnan(expansion_prob(j, cand_aj)));
          assert(!isinf(expansion_prob(j, cand_aj)));

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

    //tEndExp = std::clock();
    //if (curJ >= 45 && curI >= 45)
    //std::cerr << "pair #" << s << ": spent " << diff_seconds(tEndExp,tStartExp) << " seconds on expansion moves" << std::endl;

    //std::clock_t tStartSwap,tEndSwap;
    //tStartSwap = std::clock();

    //for now, to be sure:
    //hyp_aligned_source_words = aligned_source_words;

    //std::cerr << "starting with swap moves" << std::endl;

    //b) swap moves
    for (uint j1 = 0; j1 < curJ; j1++) {

      //std::cerr << "j1: " << j1 << std::endl;

      const uint aj1 = alignment[j1];

      const long double leaving_prob_common = dict(j1,aj1);

      const std::vector<ushort>& aligned_aj1 = aligned_source_words[aj1];

      for (uint j2 = j1 + 1; j2 < curJ; j2++) {

        //std::cerr << "j2: " << j2 << std::endl;

        const uint aj2 = alignment[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1, j2) = 0.0;
        }
        else {

#if 0
          //EXPERIMENTAL (prune constellations with very unlikely translation probs.)
          if (dict(j2,aj1) < 1e-10 || dict(j1,aj2) < 1e-10) {
            swap_prob(j1, j2) = 0.0;
            continue;
          }
          //END_EXPERIMENTAL
#endif

          long double hyp_prob = 0.0;

          uint temp_aj1 = aj1;
          uint temp_aj2 = aj2;
          uint temp_j1 = j1;
          uint temp_j2 = j2;
          if (aj1 > aj2) {
            temp_aj1 = aj2;
            temp_aj2 = aj1;
            temp_j1 = j2;
            temp_j2 = j1;
          }

          long double incoming_prob = dict(j2,aj1) * dict(j1,aj2);
          long double leaving_prob = leaving_prob_common * dict(j2,aj2);

          if (empty_word_model_ == FertNullIntra && (aj1 * aj2) == 0) { //aj1 == 0 || aj2 == 0 more compactly written

            std::vector<AlignBaseType> hyp_null_words = aligned_source_words[0];
            if (aj1 == 0)
              vec_replace<AlignBaseType>(hyp_null_words, j1, j2);
            else
              vec_replace<AlignBaseType>(hyp_null_words, j2, j1);
            vec_sort(hyp_null_words);

            incoming_prob *= null_distortion_prob(hyp_null_words, curJ);
            leaving_prob *= base_null_distortion_prob;
          }

          assert(leaving_prob > 0.0);

          const uint temp_j1_class = sclass[temp_j1];
          const uint temp_j2_class = sclass[temp_j2];
          const uint temp_taj1_class = (temp_aj1 == 0) ? MAX_UINT : tclass[temp_aj1 - 1];
          const uint temp_taj2_class = (temp_aj2 == 0) ? MAX_UINT : tclass[temp_aj2 - 1];

          if (aj1 != 0 && aj2 != 0 && cept_start_mode_ != IBM4UNIFORM &&  aligned_source_words[aj1].size() == 1
              && aligned_source_words[aj2].size() == 1) {
            //both affected cepts are one-word cepts

            //std::cerr << "case 1" << std::endl;

            // 1. entering cept temp_aj1
            if (prev_cept[temp_aj1] != MAX_UINT) {
              const uint old_sclass = temp_j1_class;
              const uint new_sclass = temp_j2_class;
              const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_cept[temp_aj1] - 1] : temp_taj1_class;

              leaving_prob *= inter_distortion_prob(temp_j1, cept_center[prev_cept[temp_aj1]], old_sclass, tc, curJ);
              incoming_prob *= inter_distortion_prob(temp_j2, cept_center[prev_cept[temp_aj1]], new_sclass, tc, curJ);
            }
            else if (use_sentence_start_prob_) {
              leaving_prob *= sentence_start_prob_[curJ][temp_j1];
              incoming_prob *= sentence_start_prob_[curJ][temp_j2];
            }
            // 2. leaving cept temp_aj1 and entering cept temp_aj2
            if (prev_cept[temp_aj2] != temp_aj1) {

              //std::cerr << "A" << std::endl;

              //a) leaving cept aj1
              const uint next_i = next_cept[temp_aj1];
              if (next_i != MAX_UINT) {

                const uint sc = sclass[aligned_source_words[next_i][0]];
                const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? temp_taj1_class : tclass[next_i - 1];

                leaving_prob *= inter_distortion_prob(aligned_source_words[next_i][0], temp_j1, sc, tc, curJ);
                incoming_prob *= inter_distortion_prob(aligned_source_words[next_i][0], temp_j2, sc, tc, curJ);
              }
              //b) entering cept temp_aj2
              const uint sclass1 = temp_j1_class;
              const uint sclass2 = temp_j2_class;
              const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_cept[temp_aj2] - 1] : temp_taj2_class;

              leaving_prob *= inter_distortion_prob(temp_j2, cept_center[prev_cept[temp_aj2]], sclass2, tc, curJ);
              incoming_prob *= inter_distortion_prob(temp_j1, cept_center[prev_cept[temp_aj2]], sclass1, tc, curJ);
            }
            else {
              //leaving cept temp_aj1 is simultaneously entering cept temp_aj2
              //NOTE: the aligned target word is here temp_aj2-1 in both the incoming and the leaving term

              //std::cerr << "B" << std::endl;

              const uint sclass1 = temp_j1_class;
              const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? temp_taj1_class : temp_taj2_class;
              const uint sclass2 = temp_j2_class;

              leaving_prob *= inter_distortion_prob(temp_j2, temp_j1, sclass2, tc, curJ);
              incoming_prob *= inter_distortion_prob(temp_j1, temp_j2, sclass1, tc, curJ);
            }

            // 3. leaving cept temp_aj2
            if (next_cept[temp_aj2] != MAX_UINT) {

              const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? temp_taj2_class : tclass[next_cept[temp_aj2] - 1];
              const uint sc = sclass[aligned_source_words[next_cept[temp_aj2]][0]];

              leaving_prob *= inter_distortion_prob(aligned_source_words[next_cept[temp_aj2]][0], temp_j2, sc, tc, curJ);
              incoming_prob *= inter_distortion_prob(aligned_source_words[next_cept[temp_aj2]][0], temp_j1, sc, tc, curJ);
            }

            hyp_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
            //DEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j1] = aj2;
            hyp_alignment[j2] = aj1;
            long double check_prob = alignment_prob(source, target, lookup, hyp_alignment);

            if (check_prob > 0.0) {

              long double check_ratio = check_prob / hyp_prob;

              if (!(check_ratio > 0.99 && check_ratio < 1.01)) {

                std::cerr << "******* swapping " << j1 << "->" << aj1 << " and " << j2 << "->" << aj2 << std::endl;
                std::cerr << " curJ: " << curJ << ", curI: " << curI << std::endl;
                std::cerr << "base alignment: " << alignment << std::endl;
                std::cerr << "actual prob: " << check_prob << std::endl;
                std::cerr << "incremental_hyp_prob: " << hyp_prob << std::endl;
              }

              assert(check_ratio > 0.99 && check_ratio < 1.01);
            }
            //END_DEBUG
#endif
          }
          else if (aj1 != 0 && aj2 != 0 && prev_cept[aj1] != aj2 && prev_cept[aj2] != aj1) {

            //std::cerr << "case 2" << std::endl;

            //std::cerr << "A1" << std::endl;

            uint old_j1_num = MAX_UINT;
            for (uint k = 0; k < fertility[temp_aj1]; k++) {
              if (aligned_source_words[temp_aj1][k] == temp_j1) {
                old_j1_num = k;
                break;
              }
            }
            assert(old_j1_num != MAX_UINT);

            uint old_j2_num = MAX_UINT;
            for (uint k = 0; k < fertility[temp_aj2]; k++) {
              if (aligned_source_words[temp_aj2][k] == temp_j2) {
                old_j2_num = k;
                break;
              }
            }
            assert(old_j2_num != MAX_UINT);

            const std::vector<ushort>& aligned_temp_aj1 = aligned_source_words[temp_aj1];

            std::vector<ushort> new_temp_aj1_aligned_source_words = aligned_temp_aj1;
            new_temp_aj1_aligned_source_words[old_j1_num] = temp_j2;
            vec_sort(new_temp_aj1_aligned_source_words);

            //std::cerr << "B1" << std::endl;

            uint new_temp_aj1_center = MAX_UINT;
            switch (cept_start_mode_) {
            case IBM4CENTER: {
              double sum_j = vec_sum(new_temp_aj1_aligned_source_words);
              new_temp_aj1_center = (uint) round(sum_j / fertility[temp_aj1]);
              break;
            }
            case IBM4FIRST: {
              new_temp_aj1_center = new_temp_aj1_aligned_source_words[0];
              break;
            }
            case IBM4LAST: {
              new_temp_aj1_center = new_temp_aj1_aligned_source_words.back();
              break;
            }
            case IBM4UNIFORM: {
              break;
            }
            default:
              assert(false);
            }

            //std::cerr << "C1" << std::endl;

            const int old_head1 = aligned_temp_aj1[0];
            const int new_head1 = new_temp_aj1_aligned_source_words[0];

            //std::cerr << "D1" << std::endl;

            if (old_head1 != new_head1) {
              if (prev_cept[temp_aj1] != MAX_UINT) {
                const uint old_sclass = source_class_[source[old_head1]];
                const uint new_sclass = source_class_[source[new_head1]];
                const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_cept[temp_aj1] - 1] : temp_taj1_class;

                leaving_prob *= inter_distortion_prob(old_head1, cept_center[prev_cept[temp_aj1]], old_sclass, tc, curJ);
                incoming_prob *= inter_distortion_prob(new_head1, cept_center[prev_cept[temp_aj1]], new_sclass, tc, curJ);
              }
              else if (use_sentence_start_prob_) {
                leaving_prob *= sentence_start_prob_[curJ][old_head1];
                incoming_prob *= sentence_start_prob_[curJ][new_head1];
              }
            }
            //std::cerr << "E1" << std::endl;

            for (uint k = 1; k < fertility[temp_aj1]; k++) {
              const uint old_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[aligned_temp_aj1[k]] : temp_taj1_class;
              const uint new_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass[new_temp_aj1_aligned_source_words[k]] : temp_taj1_class;
              leaving_prob *= cur_intra_distortion_prob(old_class, aligned_temp_aj1[k], aligned_temp_aj1[k - 1]);
              incoming_prob *= cur_intra_distortion_prob(new_class, new_temp_aj1_aligned_source_words[k],
                               new_temp_aj1_aligned_source_words[k - 1]);
            }

            //std::cerr << "F1" << std::endl;

            //transition to next cept
            if (next_cept[temp_aj1] != MAX_UINT) {
              const int next_head = aligned_source_words[next_cept[temp_aj1]][0];

              const uint sc = sclass[next_head];
              const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? temp_taj1_class : tclass[next_cept[temp_aj1] - 1];

              leaving_prob *= inter_distortion_prob(next_head, cept_center[temp_aj1], sc, tc, curJ);
              incoming_prob *= inter_distortion_prob(next_head, new_temp_aj1_center, sc, tc, curJ);
            }
            //std::cerr << "G1" << std::endl;

            const std::vector<ushort>& aligned_temp_aj2 = aligned_source_words[temp_aj2];

            std::vector<ushort> new_temp_aj2_aligned_source_words = aligned_temp_aj2;
            new_temp_aj2_aligned_source_words[old_j2_num] = temp_j1;
            vec_sort(new_temp_aj2_aligned_source_words);

            //std::cerr << "H1" << std::endl;

            uint new_temp_aj2_center = MAX_UINT;
            switch (cept_start_mode_) {
            case IBM4CENTER: {
              double sum_j = vec_sum(new_temp_aj2_aligned_source_words);
              new_temp_aj2_center = (uint) round(sum_j / fertility[temp_aj2]);
              break;
            }
            case IBM4FIRST: {
              new_temp_aj2_center = new_temp_aj2_aligned_source_words[0];
              break;
            }
            case IBM4LAST: {
              new_temp_aj2_center = new_temp_aj2_aligned_source_words.back();
              break;
            }
            case IBM4UNIFORM: {
              break;
            }
            default:
              assert(false);
            }

            const int old_head2 = aligned_source_words[temp_aj2][0];
            const int new_head2 = new_temp_aj2_aligned_source_words[0];

            //std::cerr << "I1" << std::endl;

            if (old_head2 != new_head2) {
              if (prev_cept[temp_aj2] != MAX_UINT) {
                const uint old_sclass = source_class_[source[old_head2]];
                const uint new_sclass = source_class_[source[new_head2]];
                const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? tclass[prev_cept[temp_aj2] - 1] : temp_taj2_class;

                leaving_prob *= inter_distortion_prob(old_head2, cept_center[prev_cept[temp_aj2]], old_sclass, tc, curJ);
                incoming_prob *= inter_distortion_prob(new_head2, cept_center[prev_cept[temp_aj2]], new_sclass, tc, curJ);
              }
              else if (use_sentence_start_prob_) {
                leaving_prob *= sentence_start_prob_[curJ][old_head2];
                incoming_prob *= sentence_start_prob_[curJ][new_head2];
              }
            }
            //std::cerr << "J1" << std::endl;

            for (uint k = 1; k < fertility[temp_aj2]; k++) {
              const uint old_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ?
                                     source_class_[source[aligned_temp_aj2[k]]] : temp_taj2_class;
              const uint new_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ?
                                     source_class_[source[new_temp_aj2_aligned_source_words[k]]] : temp_taj2_class;

              leaving_prob *= cur_intra_distortion_prob(old_class, aligned_temp_aj2[k], aligned_temp_aj2[k - 1]);
              incoming_prob *= cur_intra_distortion_prob(new_class, new_temp_aj2_aligned_source_words[k],
                               new_temp_aj2_aligned_source_words[k - 1]);
            }

            //std::cerr << "K1" << std::endl;

            //transition to next cept
            if (next_cept[temp_aj2] != MAX_UINT
                && cept_center[temp_aj2] != new_temp_aj2_center) {
              const int next_head = aligned_source_words[next_cept[temp_aj2]][0];

              const uint sc = sclass[next_head];
              const uint tc = (inter_dist_mode_ == IBM4InterDistModePrevious) ? temp_taj2_class : tclass[next_cept[temp_aj2] - 1];

              leaving_prob *= inter_distortion_prob(next_head, cept_center[temp_aj2], sc, tc, curJ);
              incoming_prob *= inter_distortion_prob(next_head, new_temp_aj2_center, sc, tc, curJ);
            }
            //std::cerr << "M1" << std::endl;

            hyp_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
            //DEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j1] = aj2;
            hyp_alignment[j2] = aj1;
            long double check_prob = alignment_prob(source, target, lookup, hyp_alignment);

            if (check_prob > 0.0) {

              long double check_ratio = check_prob / hyp_prob;

              if (!(check_ratio > 0.99 && check_ratio < 1.01)) {

                std::cerr << "******* swapping " << j1 << "->" << aj1 << " and " << j2 << "->" << aj2 << std::endl;
                std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
                std::cerr << "base alignment: " << alignment << std::endl;
                std::cerr << "actual prob: " << check_prob << std::endl;
                std::cerr << "incremental_hyp_prob: " << hyp_prob << std::endl;
                std::cerr << "(base prob: " << base_prob << ")" << std::endl;
              }

              if (check_prob > 1e-12 * base_prob)
                assert(check_ratio > 0.99 && check_ratio < 1.01);
            }
            //END_DEBUG
#endif

          }
          else {

            //std::cerr << "case 3" << std::endl;

            vec_replace<ushort>(hyp_aligned_source_words[aj1], j1, j2);
            vec_replace<ushort>(hyp_aligned_source_words[aj2], j2, j1);
            vec_sort(hyp_aligned_source_words[aj1]);
            vec_sort(hyp_aligned_source_words[aj2]);

            hyp_prob = base_prob * inter_distortion_prob(source, target, hyp_aligned_source_words)
                       / base_inter_distortion_prob;
            if (aj1 != 0) {
              hyp_prob *= intra_distortion_prob(hyp_aligned_source_words[aj1], sclass, tclass[aj1 - 1])
                          / intra_distortion_prob(aligned_source_words[aj1], sclass, tclass[aj1 - 1]);
            }
            if (aj2 != 0) {
              hyp_prob *= intra_distortion_prob(hyp_aligned_source_words[aj2], sclass, tclass[aj2 - 1])
                          / intra_distortion_prob(aligned_source_words[aj2], sclass, tclass[aj2 - 1]);
            }

            hyp_prob *= incoming_prob / leaving_prob;

            //restore for next loop execution:
            hyp_aligned_source_words[aj1] = aligned_aj1;
            hyp_aligned_source_words[aj2] = aligned_source_words[aj2];

#ifndef NDEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            std::swap(hyp_alignment[j1], hyp_alignment[j2]);

            long double check = alignment_prob(source, target, lookup, hyp_alignment);

            if (check > 1e-250) {

              if (!(check / hyp_prob <= 1.005 && check / hyp_prob >= 0.995)) {
                std::cerr << "aj1: " << aj1 << ", aj2: " << aj2 << std::endl;
                std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
              }

              assert(check / hyp_prob <= 1.005);
              assert(check / hyp_prob >= 0.995);
            }
            else if (check > 0.0) {

              if (!(check / hyp_prob <= 1.5 && check / hyp_prob >= 0.666)) {
                std::cerr << "aj1: " << aj1 << ", aj2: " << aj2 << std::endl;
                std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
              }

              assert(check / hyp_prob <= 1.5);
              assert(check / hyp_prob >= 0.666);

            }
            else
              assert(hyp_prob == 0.0);
#endif
          }                     //end of case 3

          //DEBUG
          if (isnan(hyp_prob) || isinf(hyp_prob)) {
            std::cerr << "ERROR: swap prob in hillclimbing is " << hyp_prob << std::endl;
            Math1D::Vector < AlignBaseType > temp_alignment = alignment;
            std::swap(temp_alignment[j1], temp_alignment[j2]);
            print_alignment_prob_factors(source, target, lookup, temp_alignment);
            exit(1);
          }
          //END_DEBUG

          assert(!isnan(hyp_prob));

          swap_prob(j1, j2) = hyp_prob;

          if (hyp_prob > best_prob) {

            best_change_is_move = false;
            best_prob = hyp_prob;
            best_swap_j1 = j1;
            best_swap_j2 = j2;
          }
        }

        assert(!isnan(swap_prob(j1, j2)));
        assert(!isinf(swap_prob(j1, j2)));
      }
    }

    //tEndSwap = std::clock();
    // if (curJ >= 45 && curI >= 45)
    //   std::cerr << "pair #" << s << ": spent " << diff_seconds(tEndSwap,tStartSwap)
    //          << " seconds on swap moves" << std::endl;

    //update alignment if a better one was found
    if (best_prob < improvement_factor * base_prob || count_iter > nMaxHCIter_) {
      if (count_iter > nMaxHCIter_)
        std::cerr << "HC Iteration limit reached" << std::endl;
      break;
    }


    //update alignment
    if (best_change_is_move) {
      uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      //std::cerr << "moving source pos" << best_move_j << " from " << cur_aj << " to " << best_move_aj << std::endl;

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;

      aligned_source_words[best_move_aj].push_back(best_move_j);
      vec_sort(aligned_source_words[best_move_aj]);
      vec_erase(aligned_source_words[cur_aj], (ushort) best_move_j);
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

      //NOTE: the fertilities are not affected here
      for (uint k = 0; k < aligned_source_words[cur_aj2].size(); k++) {
        if (aligned_source_words[cur_aj2][k] == best_swap_j2) {
          aligned_source_words[cur_aj2][k] = best_swap_j1;
          break;
        }
      }
      for (uint k = 0; k < aligned_source_words[cur_aj1].size(); k++) {
        if (aligned_source_words[cur_aj1][k] == best_swap_j1) {
          aligned_source_words[cur_aj1][k] = best_swap_j2;
          break;
        }
      }

      vec_sort(aligned_source_words[cur_aj1]);
      vec_sort(aligned_source_words[cur_aj2]);
    }

    //std::cerr << "probability improved from " << base_prob << " to " << best_prob << std::endl;
    base_prob = best_prob;

#ifndef NDEBUG
    double check_ratio = alignment_prob(source, target, lookup, alignment) / base_prob;

    if (base_prob > 1e-250) {

      if (!(check_ratio >= 0.99 && check_ratio <= 1.01)) {
        std::cerr << "check: " << alignment_prob(source, target, lookup, alignment) << std::endl;;
      }

      assert(check_ratio >= 0.99 && check_ratio <= 1.01);
    }
#endif
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

long double IBM4Trainer::nondeficient_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
    Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double>& swap_prob,
    Math1D::Vector<AlignBaseType>& alignment) const
{
  //this is just like for the IBM-3, only a different distortion routine is called

  //std::cerr << "nondef hc" << std::endl;

  /**** calculate probability of the passed alignment *****/

  double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size();

  Storage1D<std::vector<ushort> > aligned_source_words(curI + 1);

  fertility.set_constant(0);

  for (uint j = 0; j < curJ; j++) {

    const uint aj = alignment[j];
    aligned_source_words[aj].push_back(j);
    fertility[aj]++;
  }

  Math2D::Matrix<double> dict(curJ,curI+1);
  compute_dictmat_fertform(source, lookup, target, dict_, dict);

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
  // long double check_prob = nondeficient_alignment_prob(source,target,lookup,alignment);
  // double check_ratio = base_prob / check_prob;
  // assert(check_ratio >= 0.99 && check_ratio <= 1.01);
  //END_DEBUG

  uint count_iter = 0;

  Storage1D<std::vector<ushort> > hyp_aligned_source_words = aligned_source_words;

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

    if (empty_word_increase_const == 0.0 && curJ >= 2 * (zero_fert + 1)) {

      empty_word_increase_const = (curJ - 2 * zero_fert) * (curJ - 2 * zero_fert -  1) * p_zero_
                                  / ((curJ - zero_fert) * (zero_fert + 1) * p_nonzero_ * p_nonzero_);
    }

    if (empty_word_decrease_const == 0.0 && zero_fert > 0) {

      empty_word_decrease_const = (curJ - zero_fert + 1) * zero_fert * p_nonzero_ * p_nonzero_ /
                                  ((curJ - 2 * zero_fert + 1) * (curJ - 2 * zero_fert + 2) * p_zero_);
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
      const std::vector<ushort>& aligned_aj = aligned_source_words[aj];
      expansion_prob(j, aj) = 0.0;

      vec_erase<ushort>(hyp_aligned_source_words[aj], j);

      const double old_dict_prob = dict(j,aj);
      const long double leaving_prob_common = base_distortion_prob * old_dict_prob;

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        if (aj == cand_aj) {
          expansion_prob(j, cand_aj) = 0.0;
          continue;
        }
        if (cand_aj > 0) {  //better to check this before computing distortion probs
          if ((fertility[cand_aj] + 1) > fertility_limit_[target[cand_aj - 1]]) {
            expansion_prob(j, cand_aj) = 0.0;
            continue;
          }
        }
        if (cand_aj == 0 && 2 * fertility[0] + 2 > curJ) { //better to check this before computing distortion probs
          expansion_prob(j, cand_aj) = 0.0;
          continue;
        }

        const std::vector<ushort>& aligned_cand_aj = aligned_source_words[cand_aj];

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
            incoming_prob *= fertility_prob_[tidx][fertility[cand_aj] + 1];
          }
          else {
            if (curJ < 2 * fertility[0] + 2)
              incoming_prob = 0.0;
            else {

              //compute null-fert-model (zero-fert goes up by 1)

              incoming_prob *= empty_word_increase_const;
            }
          }

          long double incremental_cand_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
          Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
          hyp_alignment[j] = cand_aj;

          long double cand_prob = nondeficient_alignment_prob(source,target,lookup,hyp_alignment);

          if (cand_prob > 1e-250) {

            double ratio = incremental_cand_prob / cand_prob;
            if (! (ratio >= 0.99 && ratio <= 1.01)) {
              std::cerr << "j: " << j << ", aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
              std::cerr << "incremental: " << incremental_cand_prob << ", standalone: " << cand_prob << std::endl;
            }
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
          hyp_aligned_source_words[cand_aj] = aligned_cand_aj;
        }
      }

      hyp_aligned_source_words[aj] = aligned_aj;
    }

    /**** swap moves ****/
    for (uint j1 = 0; j1 < curJ; j1++) {

      //std::cerr << "j1: " << j1 << std::endl;

      const uint aj1 = alignment[j1];
      //const uint s_j1 = source[j1];
      const std::vector<ushort>& aligned_aj1 = aligned_source_words[aj1];

      const long double common_prob = base_prob / (base_distortion_prob * dict(j1,aj1));

      for (uint j2 = j1 + 1; j2 < curJ; j2++) {

        //std::cerr << "j2: " << j2 << std::endl;

        const uint aj2 = alignment[j2];
        //const uint s_j2 = source[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1, j2) = 0.0;
        }
        else {

          vec_replace<ushort>(hyp_aligned_source_words[aj1],j1,j2);
          vec_replace<ushort>(hyp_aligned_source_words[aj2],j2,j1);

          vec_sort(hyp_aligned_source_words[aj1]);
          vec_sort(hyp_aligned_source_words[aj2]);

          //long double incremental_prob = base_prob / base_distortion_prob *
          //                               nondeficient_distortion_prob(source, target, hyp_aligned_source_words);

          long double incremental_prob = common_prob * nondeficient_distortion_prob(source, target, hyp_aligned_source_words);

          incremental_prob *= dict(j2,aj1); // / dict(j1,aj1);
          incremental_prob *= dict(j1,aj2) / dict(j2,aj2);

#ifndef NDEBUG
          Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
          std::swap(hyp_alignment[j1],hyp_alignment[j2]);

          long double cand_prob = nondeficient_alignment_prob(source,target,lookup,hyp_alignment);

          if (cand_prob > 1e-250) {

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
          hyp_aligned_source_words[aj1] = aligned_aj1;
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

      vec_erase<ushort>(aligned_source_words[cur_aj],best_move_j);
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

      vec_replace<ushort>(aligned_source_words[cur_aj1],best_swap_j1,best_swap_j2);
      vec_replace<ushort>(aligned_source_words[cur_aj2],best_swap_j2,best_swap_j1);

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

/* virtual */
void IBM4Trainer::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  common_prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();

  /*** check if respective distortion table is present. If not, create one from the parameters ***/

  int oldJ = (cept_start_prob_.zDim() + 1) / 2;

  bool update = false;

  if (oldJ < int (J)) {
    update = true;

    inter_distortion_cache_.resize(J + 1);

    //inter params
    IBM4CeptStartModel new_param(cept_start_prob_.xDim(), cept_start_prob_.yDim(), 2 * J - 1, 1e-8, MAKENAME(new_param));
    uint new_zero_offset = J - 1;
    for (int j = -int (maxJ_) + 1; j <= int (maxJ_) - 1; j++) {

      for (uint w1 = 0; w1 < cept_start_prob_.xDim(); w1++)
        for (uint w2 = 0; w2 < cept_start_prob_.yDim(); w2++)
          new_param(w1, w2, new_zero_offset + j) = cept_start_prob_(w1, w2, displacement_offset_ + j);

    }
    cept_start_prob_ = new_param;

    //intra params

    IBM4WithinCeptModel new_wi_model(within_cept_prob_.xDim(), J, 1e-8,  MAKENAME(new_wi_model));

    for (uint c = 0; c < new_wi_model.xDim(); c++) {

      for (uint k = 0; k < within_cept_prob_.yDim(); k++)
        new_wi_model(c, k) = within_cept_prob_(c, k);
    }

    within_cept_prob_ = new_wi_model;

    displacement_offset_ = new_zero_offset;

    maxJ_ = J;

    sentence_start_parameters_.resize(J, 0.0);
  }

  if (inter_distortion_prob_.size() <= J) {
    update = true;
    inter_distortion_prob_.resize(J + 1);
  }
  if (intra_distortion_prob_.size() <= J) {
    update = true;
    intra_distortion_prob_.resize(J + 1);
  }

  uint max_s = 0;
  uint max_t = 0;
  for (uint s = 0; s < source.size(); s++) {
    max_s = std::max<uint>(max_s, source_class_[source[s]]);
  }
  for (uint t = 0; t < target.size(); t++) {
    max_t = std::max<uint>(max_t, target_class_[target[t]]);
  }

  inter_distortion_prob_[J].resize(std::max<uint>(inter_distortion_prob_[J].xDim(), max_s + 1),
                                   std::max<uint>(inter_distortion_prob_[J].yDim(), max_t + 1));

  uint dim = (intra_dist_mode_ == IBM4IntraDistModeTarget) ? max_t + 1 : max_s + 1;

  if (intra_distortion_prob_[J].xDim() <= dim) {
    update = true;
    intra_distortion_prob_[J].resize(dim, J, J);
  }

  if (use_sentence_start_prob_) {

    if (sentence_start_prob_.size() <= J) {
      update = true;
      sentence_start_prob_.resize(J + 1);
    }

    if (sentence_start_prob_[J].size() < J) {
      update = true;
      sentence_start_prob_[J].resize(J);
    }
  }

  if (update) {
    par2nonpar_inter_distortion();

    par2nonpar_intra_distortion();
    if (use_sentence_start_prob_) {
      par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
    }
  }
}

DistortCount::DistortCount(uchar J, uchar j, uchar j_prev)
  : J_(J), j_(j), j_prev_(j_prev)
{
}

bool operator<(const DistortCount& d1, const DistortCount& d2)
{
  if (d1.J_ != d2.J_)
    return (d1.J_ < d2.J_);
  if (d1.j_ != d2.j_)
    return (d1.j_ < d2.j_);
  return (d1.j_prev_ < d2.j_prev_);
}

void IBM4Trainer::train_em(uint nIter, FertilityModelTrainerBase* fert_trainer, const HmmWrapperWithClasses* passed_wrapper)
{
  //solutions with binary variables are not available for IBM-4

  std::cerr << "starting IBM-4 training without constraints";
  if (fert_trainer != 0)
    std::cerr << " (init from " << fert_trainer->model_name() << ") ";
  std::cerr << std::endl;

  double max_perplexity = 0.0;
  double approx_sum_perplexity = 0.0;

  IBM4CeptStartModel fceptstart_count(cept_start_prob_.xDim(), cept_start_prob_.yDim(), 2 * maxJ_ - 1, MAKENAME(fceptstart_count));
  IBM4WithinCeptModel fwithincept_count(within_cept_prob_.xDim(), within_cept_prob_.yDim(), MAKENAME(fwithincept_count));

  //new variant
  Storage2D<Math2D::Matrix<double> > finter_par_span_count;
  Math2D::Matrix<double> fintra_par_span_count;
  Math1D::Vector<double> fsentence_start_count(maxJ_);
  Math1D::Vector<double> fstart_span_count(maxJ_);
  if (reduce_deficiency_) {
    finter_par_span_count.resize(nSourceClasses_, nTargetClasses_);
    for (uint s = 0; s < nSourceClasses_; s++) {
      for (uint t = 0; t < nTargetClasses_; t++) {
        //finter_par_span_count(s,t).resize(displacement_offset_+1,cept_start_prob_.zDim());

        //we exploit that a span always starts below (or at) the zero offset and always ends above (or at) the zero offset
        // => remove superfluous parts of the matrix
        // => for the second index you have to add displacement_offset_ to get the true index
        finter_par_span_count(s, t).resize(displacement_offset_ + 1, displacement_offset_ + 1);
      }
    }

    fintra_par_span_count.resize(within_cept_prob_.xDim(), maxJ_ + 1);
  }

  SingleLookupTable aux_lookup;

  double dict_weight_sum = (prior_weight_active_) ? 1.0 : 0.0; //only used as a flag

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,  MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));

  Math1D::Vector<double> fnull_intra_count(null_intra_prob_.size());

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  double fzero_count;
  double fnonzero_count;

  double hillclimbtime = 0.0;
  double countcollecttime = 0.0;

  const uint out_frequency = (source_sentence_.size() <= 50000) ? 100 : 10000;

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    //NOTE: in the presence of many word classes using these counts is a waste of memory.
    // It would be more prudent to keep track of the counts for every sentence (using the CountStructure from the IBM3,
    //  or maybe with maps replaced by vectors)
    // and to then filter out the relevant counts for the current combination. Only, that's much more complex to implement,
    // so it may take a while (or never be done)
    Storage2D<std::map<Math1D::Vector<uchar,uchar>,double> > nondef_cept_start_count(nSourceClasses_, nTargetClasses_);
    Storage1D<std::map<Math1D::Vector<uchar,uchar>,double> > nondef_within_cept_count(nTargetClasses_);

    //try to collect counts in temporary maps for faster access
    Storage2D<std::map<Math1D::Vector<uchar,uchar>,double> > temp_cept_start_count(nSourceClasses_, nTargetClasses_);
    Storage1D<std::map<Math1D::Vector<uchar,uchar>,double> > temp_within_cept_count(nTargetClasses_);

    //this count is almost like fceptstart_count, but only includes the terms > min_nondef_count_ and also no terms
    // where only one position is available
    IBM4CeptStartModel fnondef_ceptstart_singleton_count(cept_start_prob_.xDim(), cept_start_prob_.yDim(), 2 * maxJ_ - 1, 0.0, MAKENAME(fnondef_ceptstart_singleton_count));

    //same for this count and fnondef_withincept_count
    IBM4WithinCeptModel fnondef_withincept_singleton_count(within_cept_prob_.xDim(), within_cept_prob_.yDim(), 0.0, MAKENAME(fnondef_withincept_singleton_count));

    std::cerr << "******* IBM-4 EM-iteration " << iter << std::endl;

    if (passed_wrapper != 0
        && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    //DEBUG
    // std::cerr << "WARNING: setting uniform fertility probabilities" << std::endl;
    // for (uint i=1; i < nTargetWords; i++) {
    //   fertility_prob_[i].set_constant(1.0 / std::max<uint>(fertility_prob_[i].size(),fertility_limit_[i]+1));
    // }
    //END_DEBUG

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    fceptstart_count.set_constant(0.0);
    fwithincept_count.set_constant(0.0);

    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    fnull_intra_count.set_constant(0.0);

    for (uint s = 0; s < finter_par_span_count.xDim(); s++)
      for (uint t = 0; t < finter_par_span_count.yDim(); t++)
        finter_par_span_count(s, t).set_constant(0.0);

    fintra_par_span_count.set_constant(0.0);

    fsentence_start_count.set_constant(0.0);
    fstart_span_count.set_constant(0.0);

    max_perplexity = 0.0;
    approx_sum_perplexity = 0.0;

    for (size_t s = 0; s < source_sentence_.size(); s++) {

      if ((s % out_frequency) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      //std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;

      Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      std::clock_t tHillclimbStart, tHillclimbEnd;
      tHillclimbStart = std::clock();

      long double best_prob = 0.0;

      //std::cerr << "calling hillclimbing" << std::endl;

      //in iter 1 this is guaranteed
      //if (iter > 1 && hillclimb_mode_ == HillclimbingRestart)
      //  best_known_alignment_[s] = initial_alignment[s];

      if (fert_trainer != 0 && iter == 1) {

        best_prob = fert_trainer->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter,
                    fertility, expansion_move_prob, swap_move_prob, cur_alignment);
      }
      else {

        best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, cur_alignment);
      }
      max_perplexity -= std::log(best_prob);

      //std::cerr << "back from hillclimbing" << std::endl;

      tHillclimbEnd = std::clock();

      hillclimbtime += diff_seconds(tHillclimbEnd, tHillclimbStart);

      const long double expansion_prob = expansion_move_prob.sum();
      const long double swap_prob = swap_mass(swap_move_prob);

      const long double sentence_prob = best_prob + expansion_prob + swap_prob;

      Math2D::Matrix<double> j_marg;
      Math2D::Matrix<double> i_marg;

      FertilityModelTrainer::compute_approximate_jmarginals(cur_alignment, expansion_move_prob, swap_move_prob, sentence_prob, j_marg);
      compute_approximate_imarginals(cur_alignment, fertility, expansion_move_prob, sentence_prob, i_marg);

      //std::cerr << "sentence_prob: " << sentence_prob << std::endl;
      //std::cerr << "best prob: " << best_prob << std::endl;
      //std::cerr << "expansion prob: " << expansion_prob << std::endl;
      //std::cerr << "swap prob: " << swap_prob << std::endl;

      approx_sum_perplexity -= std::log(sentence_prob);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      NamedStorage1D<std::vector<int> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));
      for (uint j = 0; j < curJ; j++) {
        const uint cur_aj = cur_alignment[j];
        aligned_source_words[cur_aj].push_back(j);
      }

      /**** update empty word counts *****/
      for (uint j = 0; j <= curJ / 2; j++) {

        fzero_count += j * i_marg(j, 0);
        fnonzero_count += (curJ - 2 * j) * i_marg(j, 0);
      }
      // update_zero_counts(cur_alignment, fertility,
      //                         expansion_move_prob, swap_prob, best_prob,
      //                         sentence_prob, inv_sentence_prob,
      //                         fzero_count, fnonzero_count);

      if (empty_word_model_ == FertNullIntra) {


        const std::vector<int>& null_words = aligned_source_words[0];
        assert(null_words.size() == fertility[0]);

        const uint zero_fert = fertility[0];
        if (zero_fert > 1) {

          long double main_prob = sentence_prob - expansion_move_prob.row_sum(0);
          for (uint j1 = 0; j1 < curJ - 1; j1++) {
            for (uint j2 = j1 + 1; j2 < curJ; j2++) {
              if (cur_alignment[j1] * cur_alignment[j2] == 0) //short for ||
                main_prob -= swap_move_prob(j1, j2);
            }
          }
          for (uint j=0; j < curJ; j++) {
            if (cur_alignment[j] == 0) {
              for (uint i=1; i <= curI; i++)
                main_prob -= expansion_move_prob(j,i);
            }
          }

          assert(main_prob > 0.0);

          const double addon = main_prob / sentence_prob;
          for (uint k = 1; k < zero_fert; k++) {
            int cur = null_words[k];
            int prev = null_words[k - 1];
            assert(cur > prev);
            fnull_intra_count[cur - prev] += addon;
          }

          //swaps involving NULL
          for (int j1 = 0; j1 < curJ - 1; j1++) {
            for (int j2 = j1 + 1; j2 < curJ; j2++) {

              if (cur_alignment[j1] * cur_alignment[j2] == 0) { // short for ||

                const long double sprob = swap_move_prob(j1, j2);
                if (sprob > 0.0) {

                  const double prob = sprob / sentence_prob;

                  std::vector<int> hyp_null_words = null_words;

                  if (cur_alignment[j1] == 0)
                    vec_replace(hyp_null_words, j1, j2);
                  else
                    vec_replace(hyp_null_words, j2, j1);
                  vec_sort(hyp_null_words);

                  for (uint k = 1; k < zero_fert; k++) {
                    const int cur = hyp_null_words[k];
                    const int prev = hyp_null_words[k - 1];
                    assert(cur > prev);
                    fnull_intra_count[cur - prev] += prob;
                  }
                }
              }
            }
          }
        }
        if (zero_fert > 2) {
          //expansions away from NULL

          for (uint j=0; j < curJ; j++) {
            if (cur_alignment[j] == 0) {
              long double sum = 0.0;
              for (uint i=1; i <= curI; i++)
                sum += expansion_move_prob(j,i);

              const double prob = sum / sentence_prob;
              if (prob > 0.0) {

                std::vector<int> hyp_null_words = null_words;
                vec_erase<int>(hyp_null_words,j);

                for (uint k = 1; k < zero_fert - 1; k++) {
                  const int cur = hyp_null_words[k];
                  const int prev = hyp_null_words[k - 1];
                  assert(cur > prev);
                  fnull_intra_count[cur - prev] += prob;
                }
              }
            }
          }
        }
        if (zero_fert > 0) {
          //expansions to NULL

          for (uint j = 0; j < curJ; j++) {

            const long double eprob = expansion_move_prob(j, 0);
            if (eprob > 0.0) {

              const double prob = eprob / sentence_prob;

              std::vector<int> hyp_null_words = null_words;
              hyp_null_words.push_back(j);
              vec_sort(hyp_null_words);

              for (uint k = 1; k < zero_fert + 1; k++) {
                const int cur = hyp_null_words[k];
                const int prev = hyp_null_words[k - 1];
                assert(cur > prev);
                fnull_intra_count[cur - prev] += prob;
              }
            }
          }
        }
      }

      /**** update fertility counts *****/
      for (uint i = 1; i <= curI; i++) {

        const uint t_idx = cur_target[i - 1];

        Math1D::Vector<double>& cur_fert_count = ffert_count[t_idx];

        const double check_sum = i_marg.row_sum(i);
        assert(check_sum >= 0.99 && check_sum <= 1.01);

        for (uint c = 0; c <= std::min<ushort>(curJ, fertility_limit_[t_idx]); c++)
          cur_fert_count[c] += i_marg(c, i);
      }
      //update_fertility_counts(cur_target, cur_alignment, fertility,
      //                        expansion_move_prob, sentence_prob, inv_sentence_prob, ffert_count);

      /**** update dictionary counts *****/
      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        const double check_sum = j_marg.row_sum(j);
        assert(check_sum >= 0.99 && check_sum <= 1.01);

        for (uint i = 0; i <= curI; i++) {

          const double marg = j_marg(i, j);

          if (i == 0)
            fwcount[0][s_idx - 1] += marg;
          else {
            fwcount[cur_target[i - 1]][cur_lookup(j, i - 1)] += marg;
          }
        }
      }
      //update_dict_counts(cur_source, cur_target, cur_lookup, cur_alignment,
      //                   expansion_move_prob, swap_move_prob, sentence_prob, inv_sentence_prob,fwcount);

      std::clock_t tCountCollectStart, tCountCollectEnd;
      tCountCollectStart = std::clock();

      /**** update distortion counts *****/
      //std::cerr << "update of distortion counts" << std::endl;

      fstart_span_count[curJ - 1] += 1.0;

      // 1. handle viterbi alignment

      //std::cerr << "a) viterbi" << std::endl;

      int cur_prev_cept = -1;
      int prev_cept_center = -1;
      for (uint i = 1; i <= curI; i++) {

        uint tclass = target_class_[cur_target[i - 1]];

        const double cur_prob = inv_sentence_prob * best_prob;

        if (fertility[i] > 0) {

          const std::vector<int>& cur_aligned_source_words = aligned_source_words[i];
          assert(cur_aligned_source_words.size() == fertility[i]);

          const uint first_j = cur_aligned_source_words[0];

          //a) update head prob
          if (cur_prev_cept >= 0) {

            const uint sclass = source_class_[cur_source[first_j]];

            if (inter_dist_mode_ == IBM4InterDistModePrevious)
              tclass = target_class_[cur_target[cur_prev_cept - 1]];

            int diff = first_j - prev_cept_center;
            diff += displacement_offset_;

            fceptstart_count(sclass, tclass, diff) += cur_prob;

            if (!nondeficient_ && reduce_deficiency_) {

              //new variant
              uint diff_start = displacement_offset_ - prev_cept_center;
              uint diff_end = diff_start + curJ - 1;
              finter_par_span_count(sclass, tclass)(diff_start, diff_end - displacement_offset_) += cur_prob;
            }
          }
          else if (use_sentence_start_prob_) {

            fsentence_start_count[first_j] += cur_prob;
          }
          //b) update within-cept prob
          int prev_aligned_j = first_j;

          for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

            const int cur_j = cur_aligned_source_words[k];

            const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[cur_source[cur_j]] : target_class_[cur_target[i - 1]];

            int diff = cur_j - prev_aligned_j;
            fwithincept_count(cur_class, diff) += cur_prob;

            if (reduce_deficiency_) {
              fintra_par_span_count(cur_class, curJ - prev_aligned_j) += cur_prob;
            }

            prev_aligned_j = cur_j;
          }

          switch (cept_start_mode_) {
          case IBM4CENTER: {
            prev_cept_center = (int)round(((double)vec_sum(cur_aligned_source_words)) / fertility[i]);
            break;
          }
          case IBM4FIRST:
            prev_cept_center = cur_aligned_source_words[0];
            break;
          case IBM4LAST: {
            prev_cept_center = cur_aligned_source_words.back();
            break;
          }
          case IBM4UNIFORM:
            prev_cept_center = cur_aligned_source_words[0];
            break;
          default:
            assert(false);
          }

          cur_prev_cept = i;
        }
      }

      //std::cerr << "b) expansion" << std::endl;

      const long double thresh = best_prob * 1e-11;

      // 2. handle expansion moves
      NamedStorage1D<std::vector<int> > exp_aligned_source_words(MAKENAME(exp_aligned_source_words));
      exp_aligned_source_words = aligned_source_words;

      for (uint exp_j = 0; exp_j < curJ; exp_j++) {

        const uint cur_aj = cur_alignment[exp_j];

        vec_erase(exp_aligned_source_words[cur_aj], (int)exp_j);

        for (uint exp_i = 0; exp_i <= curI; exp_i++) {

          const long double eprob = expansion_move_prob(exp_j, exp_i);

          if (eprob > thresh) {

            const double cur_prob = inv_sentence_prob * eprob;

            //modify
            exp_aligned_source_words[exp_i].push_back(exp_j);
            vec_sort(exp_aligned_source_words[exp_i]);

            int prev_center = -100;
            int prev_cept = -1;

            for (uint i = 1; i <= curI; i++) {

              const std::vector<int>& cur_aligned_source_words = exp_aligned_source_words[i];

              if (cur_aligned_source_words.size() > 0) {

                uint tclass = target_class_[cur_target[i - 1]];
                const int first_j = cur_aligned_source_words[0];

                //collect counts for the head model
                if (prev_center >= 0) {

                  const uint sclass = source_class_[cur_source[first_j]];

                  if (inter_dist_mode_ == IBM4InterDistModePrevious)
                    tclass = target_class_[cur_target[prev_cept - 1]];

                  int diff = first_j - prev_center;
                  diff += displacement_offset_;
                  fceptstart_count(sclass, tclass, diff) += cur_prob;

                  if (!nondeficient_ && reduce_deficiency_) {

                    //new variant
                    uint diff_start = displacement_offset_ - prev_center;
                    uint diff_end = diff_start + curJ - 1;
                    finter_par_span_count(sclass, tclass)(diff_start, diff_end - displacement_offset_) += cur_prob;
                  }
                }
                else if (use_sentence_start_prob_) {

                  fsentence_start_count[first_j] += cur_prob;
                }
                //collect counts for the within-cept model
                int prev_j = first_j;

                for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

                  const int cur_j = cur_aligned_source_words[k];

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ?
                                         source_class_[cur_source[cur_j]] : target_class_[cur_target[i - 1]];

                  int diff = cur_j - prev_j;
                  fwithincept_count(cur_class, diff) += cur_prob;

                  if (reduce_deficiency_) {
                    fintra_par_span_count(cur_class, curJ - prev_j) += cur_prob;
                  }

                  prev_j = cur_j;
                }

                //update prev_center
                switch (cept_start_mode_) {
                case IBM4CENTER:
                  prev_center = (int)round(((double)vec_sum(cur_aligned_source_words)) / cur_aligned_source_words.size());
                  break;
                case IBM4FIRST:
                  prev_center = first_j;
                  break;
                case IBM4LAST: {
                  prev_center = prev_j;     //prev_j was set to the last pos in the above loop
                  break;
                }
                case IBM4UNIFORM:
                  prev_center = first_j;      //will not be used
                  break;
                default:
                  assert(false);
                }

                prev_cept = i;
              }
            }

            //restore
            exp_aligned_source_words[exp_i] = aligned_source_words[exp_i];
          }
        }

        exp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
      }

      //std::cerr << "c) swap" << std::endl;

      //3. handle swap moves
      NamedStorage1D<std::vector<int> > swap_aligned_source_words(MAKENAME(swap_aligned_source_words));
      swap_aligned_source_words = aligned_source_words;

      for (uint swap_j1 = 0; swap_j1 < curJ; swap_j1++) {

        const uint aj1 = cur_alignment[swap_j1];

        for (uint swap_j2 = 0; swap_j2 < curJ; swap_j2++) {

          const long double sprob = swap_move_prob(swap_j1, swap_j2);

          if (sprob > thresh) {

            const double cur_prob = inv_sentence_prob * sprob;

            const uint aj2 = cur_alignment[swap_j2];

            //modify
            vec_replace<int>(swap_aligned_source_words[aj1], swap_j1, swap_j2);
            vec_replace<int>(swap_aligned_source_words[aj2], swap_j2, swap_j1);
            vec_sort(swap_aligned_source_words[aj1]);
            vec_sort(swap_aligned_source_words[aj2]);

            int prev_center = -100;
            int prev_cept = -1;

            for (uint i = 1; i <= curI; i++) {

              const std::vector<int>& cur_aligned_source_words = swap_aligned_source_words[i];
              assert(cur_aligned_source_words.size() == fertility[i]);

              if (cur_aligned_source_words.size() > 0) {

                uint tclass = target_class_[cur_target[i - 1]];

                const int first_j = cur_aligned_source_words[0];

                //collect counts for the head model
                if (prev_center >= 0) {

                  const uint sclass = source_class_[cur_source[first_j]];

                  if (inter_dist_mode_ == IBM4InterDistModePrevious)
                    tclass = target_class_[cur_target[prev_cept - 1]];

                  int diff = first_j - prev_center;
                  diff += displacement_offset_;
                  fceptstart_count(sclass, tclass, diff) += cur_prob;

                  if (!nondeficient_ && reduce_deficiency_) {

                    //new variant
                    uint diff_start = displacement_offset_ - prev_center;
                    uint diff_end = diff_start + curJ - 1;
                    finter_par_span_count(sclass, tclass)(diff_start, diff_end - displacement_offset_) += cur_prob;
                  }
                }
                else if (use_sentence_start_prob_) {

                  fsentence_start_count[first_j] += cur_prob;
                }
                //collect counts for the within-cept model
                int prev_j = first_j;

                for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

                  const int cur_j = cur_aligned_source_words[k];

                  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ?
                                         source_class_[cur_source[cur_j]]: target_class_[cur_target[i - 1]];

                  int diff = cur_j - prev_j;
                  fwithincept_count(cur_class, diff) += cur_prob;

                  if (reduce_deficiency_) {
                    fintra_par_span_count(cur_class, curJ - prev_j) += cur_prob;
                  }

                  prev_j = cur_j;
                }

                //update prev_center
                switch (cept_start_mode_) {
                case IBM4CENTER:
                  prev_center = (int)round(((double)vec_sum(cur_aligned_source_words)) / cur_aligned_source_words.size());
                  break;
                case IBM4FIRST:
                  prev_center = first_j;
                  break;
                case IBM4LAST:
                  prev_center = prev_j;       //prev_j was set to the last pos in the above loop
                  break;
                case IBM4UNIFORM:
                  prev_center = first_j;      //will not be used
                  break;
                default:
                  assert(false);
                }

                prev_cept = i;
              }
            }

            //restore
            swap_aligned_source_words[aj1] = aligned_source_words[aj1];
            swap_aligned_source_words[aj2] = aligned_source_words[aj2];
          }
        }
      }

      if (nondeficient_) {

        //try to collect counts in temporary maps for faster access
        for (uint c1 = 0; c1 < nSourceClasses_; c1++)
          for (uint c2 = 0; c2 < nTargetClasses_; c2++)
            temp_cept_start_count(c1, c2).clear();
        for (uint c = 0; c < nTargetClasses_; c++)
          temp_within_cept_count[c].clear();

        //a) best known alignment (=mode)
        const double mode_contrib = inv_sentence_prob * best_prob;

        int prev_cept_center = -1;
        int prev_cept = -1;

        Storage1D<bool> fixed(curJ, false);

        for (uint i = 1; i <= curI; i++) {

          const std::vector<int>& cur_aligned_source_words = aligned_source_words[i];

          if (cur_aligned_source_words.size() > 0) {

            const uint ti = cur_target[i - 1];
            uint tclass = target_class_[ti];

            const int first_j = cur_aligned_source_words[0];

            uint nToRemove = cur_aligned_source_words.size() - 1;

            //handle the head of the cept
            if (prev_cept_center != -1 && cept_start_mode_ != IBM4UNIFORM) {

              const uint sclass = source_class_[cur_source[first_j]];

              if (inter_dist_mode_ == IBM4InterDistModePrevious)
                tclass = target_class_[cur_target[prev_cept - 1]];

              std::vector<uchar> possible_diffs;

              for (int j = 0; j < int (curJ); j++) {
                if (!fixed[j]) {
                  possible_diffs.push_back(j - prev_cept_center + displacement_offset_);
                }
              }

              if (nToRemove > 0) {
                possible_diffs.resize(possible_diffs.size() - nToRemove);
              }

              if (possible_diffs.size() > 1) {  //no use storing cases where only one pos. is available

                Math1D::Vector<uchar,uchar> vec_possible_diffs(possible_diffs.size());
                assign(vec_possible_diffs, possible_diffs);

                temp_cept_start_count(sclass, tclass)[vec_possible_diffs] += mode_contrib;
                fnondef_ceptstart_singleton_count(sclass, tclass, first_j - prev_cept_center + displacement_offset_) += mode_contrib;
              }
            }
            fixed[first_j] = true;

            //handle the body of the cept
            int prev_j = first_j;
            if (!uniform_intra_prob_) {
              for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

                nToRemove--;

                std::vector<uchar> possible_diffs;

                const int cur_j = cur_aligned_source_words[k];

                const uint cur_class =
                  (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[cur_source[cur_j]] : target_class_[cur_target[i - 1]];

                for (int j = prev_j + 1; j < int (curJ); j++) {
                  possible_diffs.push_back(j - prev_j);
                }

                if (nToRemove > 0) {
                  possible_diffs.resize(possible_diffs.size() - nToRemove);
                }

                if (possible_diffs.size() > 1) {  //no use storing cases where only one pos. is available

                  Math1D::Vector<uchar,uchar> vec_possible_diffs(possible_diffs.size());
                  assign(vec_possible_diffs, possible_diffs);

                  temp_within_cept_count[cur_class][vec_possible_diffs] += mode_contrib;
                  fnondef_withincept_singleton_count(cur_class, cur_j - prev_j) += mode_contrib;
                }
                //std::cerr << "dp: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

                fixed[cur_j] = true;

                prev_j = cur_j;
              }
            }
            else
              prev_j = cur_aligned_source_words.back();

            switch (cept_start_mode_) {
            case IBM4CENTER: {

              //compute the center of this cept and store the result in prev_cept_center
              double sum = vec_sum(cur_aligned_source_words);
              prev_cept_center = (int)round(sum / cur_aligned_source_words.size());
              break;
            }
            case IBM4FIRST:
              prev_cept_center = first_j;
              break;
            case IBM4LAST: {
              prev_cept_center = prev_j;      //was set to the last pos in the above llop
              break;
            }
            case IBM4UNIFORM:
              prev_cept_center = first_j;       //will not be used
              break;
            default:
              assert(false);
            }

            prev_cept = i;
            assert(prev_cept_center >= 0);
          }
        }

        Storage1D<std::vector<int> > hyp_aligned_source_words = aligned_source_words;

        //b) expansion moves
        for (uint jj = 0; jj < curJ; jj++) {

          uint cur_aj = cur_alignment[jj];

          for (uint aj = 0; aj <= curI; aj++) {

            const double contrib = expansion_move_prob(jj, aj) / sentence_prob;

            if (contrib > min_nondef_count_) {

              assert(aj != cur_aj);

              vec_erase<int>(hyp_aligned_source_words[cur_aj], jj);
              hyp_aligned_source_words[aj].push_back(jj);
              vec_sort(hyp_aligned_source_words[aj]);

              int prev_cept_center = -1;
              int prev_cept = -1;

              Storage1D<bool> fixed(curJ, false);

              for (uint i = 1; i <= curI; i++) {

                const std::vector<int> cur_hyp_aligned_source_words = hyp_aligned_source_words[i];

                if (cur_hyp_aligned_source_words.size() > 0) {

                  const uint ti = cur_target[i - 1];
                  uint tclass = target_class_[ti];

                  const int first_j = cur_hyp_aligned_source_words[0];

                  uint nToRemove = cur_hyp_aligned_source_words.size() - 1;

                  //handle the head of the cept
                  if (prev_cept_center != -1 && cept_start_mode_ != IBM4UNIFORM) {

                    const uint sclass = source_class_[cur_source[first_j]];

                    if (inter_dist_mode_ == IBM4InterDistModePrevious)
                      tclass = target_class_[cur_target[prev_cept - 1]];

                    std::vector<uchar> possible_diffs;

                    for (int j = 0; j < int (curJ); j++) {
                      if (!fixed[j]) {
                        possible_diffs.push_back(j - prev_cept_center + displacement_offset_);
                      }
                    }

                    if (nToRemove > 0) {
                      possible_diffs.resize(possible_diffs.size() - nToRemove);
                    }

                    if (possible_diffs.size() > 1) {  //no use storing cases where only one pos. is available

                      Math1D::Vector<uchar,uchar> vec_possible_diffs(possible_diffs.size());
                      assign(vec_possible_diffs, possible_diffs);

                      temp_cept_start_count(sclass, tclass)[vec_possible_diffs] += contrib;
                      fnondef_ceptstart_singleton_count(sclass, tclass, first_j - prev_cept_center + displacement_offset_) += contrib;
                    }
                  }
                  fixed[first_j] = true;

                  //handle the body of the cept
                  int prev_j = first_j;
                  if (!uniform_intra_prob_) {
                    for (uint k = 1; k < cur_hyp_aligned_source_words.size(); k++) {

                      nToRemove--;

                      std::vector<uchar> possible_diffs;

                      const int cur_j = cur_hyp_aligned_source_words[k];

                      const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[cur_source[cur_j]]
                                             : target_class_[cur_target[i - 1]];

                      for (int j = prev_j + 1; j < int (curJ); j++) {
                        possible_diffs.push_back(j - prev_j);
                      }

                      if (nToRemove > 0) {
                        possible_diffs.resize(possible_diffs.size() - nToRemove);
                      }

                      if (possible_diffs.size() > 1) {    //no use storing cases where only one pos. is available

                        Math1D::Vector<uchar,uchar> vec_possible_diffs(possible_diffs.size());
                        assign(vec_possible_diffs, possible_diffs);

                        temp_within_cept_count[cur_class][vec_possible_diffs] += contrib;
                        fnondef_withincept_singleton_count(cur_class, cur_j - prev_j) += contrib;
                      }
                      //std::cerr << "dp: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

                      fixed[cur_j] = true;

                      prev_j = cur_j;
                    }
                  }
                  else
                    prev_j = cur_hyp_aligned_source_words.back();

                  switch (cept_start_mode_) {
                  case IBM4CENTER: {

                    //compute the center of this cept and store the result in prev_cept_center
                    double sum = vec_sum(cur_hyp_aligned_source_words);

                    prev_cept_center = (int)round(sum / cur_hyp_aligned_source_words.size());
                    break;
                  }
                  case IBM4FIRST:
                    prev_cept_center = first_j;
                    break;
                  case IBM4LAST:
                    prev_cept_center = prev_j;  //was set to the last pos in the above loop
                    break;
                  case IBM4UNIFORM:
                    prev_cept_center = first_j;
                    break;
                  default:
                    assert(false);
                  }

                  prev_cept = i;
                  assert(prev_cept_center >= 0);
                }
              }

              hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
              hyp_aligned_source_words[aj] = aligned_source_words[aj];
            }
          }
        }

        //c) swap moves
        for (uint j1 = 0; j1 < curJ - 1; j1++) {

          const uint aj1 = cur_alignment[j1];

          for (uint j2 = j1 + 1; j2 < curJ; j2++) {

            const double contrib = swap_move_prob(j1, j2) / sentence_prob;

            if (contrib > min_nondef_count_) {

              const uint aj2 = cur_alignment[j2];

              vec_replace<int>(hyp_aligned_source_words[aj1], j1, j2);
              vec_sort(hyp_aligned_source_words[aj1]);
              vec_replace<int >(hyp_aligned_source_words[aj2], j2, j1);
              vec_sort(hyp_aligned_source_words[aj2]);

              int prev_cept_center = -1;
              int prev_cept = -1;

              Storage1D<bool> fixed(curJ, false);

              for (uint i = 1; i <= curI; i++) {

                const std::vector<int>& cur_hyp_aligned_source_words = hyp_aligned_source_words[i];

                if (cur_hyp_aligned_source_words.size() > 0) {

                  const uint ti = cur_target[i - 1];
                  uint tclass = target_class_[ti];

                  const int first_j = cur_hyp_aligned_source_words[0];

                  uint nToRemove = cur_hyp_aligned_source_words.size() - 1;

                  //handle the head of the cept
                  if (prev_cept_center != -1 && cept_start_mode_ != IBM4UNIFORM) {

                    const uint sclass = source_class_[cur_source[first_j]];

                    if (inter_dist_mode_ == IBM4InterDistModePrevious)
                      tclass = target_class_[cur_target[prev_cept - 1]];

                    std::vector<uchar> possible_diffs;

                    for (int j = 0; j < int (curJ); j++) {
                      if (!fixed[j]) {
                        possible_diffs.push_back(j - prev_cept_center + displacement_offset_);
                      }
                    }

                    if (nToRemove > 0) {
                      possible_diffs.resize(possible_diffs.size() - nToRemove);
                    }

                    if (possible_diffs.size() > 1) {  //no use storing cases where only one pos. is available

                      Math1D::Vector<uchar,uchar> vec_possible_diffs(possible_diffs.size());
                      assign(vec_possible_diffs, possible_diffs);

                      temp_cept_start_count(sclass, tclass)[vec_possible_diffs] += contrib;
                      fnondef_ceptstart_singleton_count(sclass, tclass, first_j - prev_cept_center + displacement_offset_) += contrib;
                    }
                  }
                  fixed[first_j] = true;

                  //handle the body of the cept
                  int prev_j = first_j;
                  if (!uniform_intra_prob_) {
                    for (uint k = 1; k < hyp_aligned_source_words[i].size(); k++) {

                      nToRemove--;

                      std::vector<uchar> possible_diffs;

                      const int cur_j = cur_hyp_aligned_source_words[k];

                      const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ?
                                             source_class_[cur_source[cur_j]] : target_class_[cur_target[i - 1]];

                      for (int j = prev_j + 1; j < int (curJ); j++) {
                        possible_diffs.push_back(j - prev_j);
                      }

                      if (nToRemove > 0) {
                        possible_diffs.resize(possible_diffs.size() - nToRemove);
                      }

                      if (possible_diffs.size() > 1) {    //no use storing cases where only one pos. is available

                        Math1D::Vector<uchar,uchar> vec_possible_diffs(possible_diffs.size());
                        assign(vec_possible_diffs, possible_diffs);

                        temp_within_cept_count[cur_class][vec_possible_diffs] += contrib;
                        fnondef_withincept_singleton_count(cur_class, cur_j - prev_j) += contrib;
                      }
                      //std::cerr << "dp: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

                      fixed[cur_j] = true;

                      prev_j = cur_j;
                    }
                  }
                  else
                    prev_j = cur_hyp_aligned_source_words.back();


                  switch (cept_start_mode_) {
                  case IBM4CENTER: {

                    //compute the center of this cept and store the result in prev_cept_center
                    double sum = vec_sum(cur_hyp_aligned_source_words);

                    prev_cept_center = (int)round(sum / cur_hyp_aligned_source_words.size());
                    break;
                  }
                  case IBM4FIRST:
                    prev_cept_center = first_j;
                    break;
                  case IBM4LAST:
                    prev_cept_center = prev_j;  //was set to the last pos in the above loop
                    break;
                  case IBM4UNIFORM:
                    prev_cept_center = first_j;
                    break;
                  default:
                    assert(false);
                  }

                  prev_cept = i;
                  assert(prev_cept_center >= 0);
                }
              }

              hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
              hyp_aligned_source_words[aj2] = aligned_source_words[aj2];
            }
          }
        }

        for (uint c1 = 0; c1 < nSourceClasses_; c1++) {
          for (uint c2 = 0; c2 < nTargetClasses_; c2++) {
            for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = temp_cept_start_count(c1, c2).begin();
                 it != temp_cept_start_count(c1, c2).end(); it++) {
              nondef_cept_start_count(c1, c2)[it->first] += it->second;
            }
          }
        }
        for (uint c = 0; c < nTargetClasses_; c++) {
          for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = temp_within_cept_count[c].begin();
               it != temp_within_cept_count[c].end(); it++)
            nondef_within_cept_count[c][it->first] += it->second;
        }

      } //end -- if (nondeficient_)

      tCountCollectEnd = std::clock();
      countcollecttime += diff_seconds(tCountCollectEnd, tCountCollectStart);

      //clean up cache
      for (uint j = 0; j < inter_distortion_cache_[curJ].size(); j++)
        inter_distortion_cache_[curJ][j].clear();

    } //loop over sentences finished

    double reg_term = regularity_term();        //we need the reg-term before the parameter update!

    /***** update probability models from counts *******/

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      std::cerr << "zero counts: " << fzero_count << ", " << fnonzero_count << std::endl;
      const double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max(fert_min_p0,fzero_count / fsum);
      p_nonzero_ = std::max(fert_min_p0,fnonzero_count / fsum);
    }

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    if (empty_word_model_ == FertNullIntra) {

      double sum = fnull_intra_count.sum();
      if (sum > 1e-300) {

        for (uint k=0; k < null_intra_prob_.size(); k++)
          null_intra_prob_[k] = std::max(fert_min_param_entry,fnull_intra_count[k] / sum);
      }
    }

    assert(!isnan(p_zero_));
    assert(!isnan(p_nonzero_));

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
    update_dict_from_counts(fwcount, prior_weight_, dict_weight_sum, smoothed_l0_, l0_beta_, dict_m_step_iter_, dict_, fert_min_dict_entry,
                            msolve_mode_ != MSSolvePGD);

    //update fertility probabilities
    update_fertility_prob(ffert_count, fert_min_param_entry);

    //update distortion probabilities

    //std::cerr << "A" << std::endl;

    //a) inter distortion
    if (cept_start_mode_ != IBM4UNIFORM) {
      if (nondeficient_) {

        IBM4CeptStartModel hyp_cept_start_prob = cept_start_prob_;

        for (uint x = 0; x < cept_start_prob_.xDim(); x++) {
          std::cerr << "nondeficient inter-m-step(" << x << ",*)" << std::endl;
          for (uint y = 0; y < cept_start_prob_.yDim(); y++) {

            std::map<Math1D::Vector<uchar,uchar>,double>& cur_map = nondef_cept_start_count(x, y);

            std::vector<Math1D::Vector<uchar,uchar> > open_pos(cur_map.size());
            std::vector<double> weight(cur_map.size());

            uint k = 0;
            for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = cur_map.begin(); it != cur_map.end(); it++) {
              open_pos[k] = it->first;
              weight[k] = it->second;

              k++;
            }

            cur_map.clear();

            double cur_energy = nondeficient_inter_m_step_energy(fnondef_ceptstart_singleton_count, open_pos, weight, cept_start_prob_, x, y);

            double sum = 0.0;
            for (uint k = 0; k < cept_start_prob_.zDim(); k++) {
              sum += fceptstart_count(x, y, k);
            }

            if (sum > 1e-300) {

              for (uint k = 0; k < cept_start_prob_.zDim(); k++) {
                hyp_cept_start_prob(x, y, k) = std::max(fert_min_param_entry, fceptstart_count(x, y, k) / sum);
              }

              const double hyp_energy = nondeficient_inter_m_step_energy(fnondef_ceptstart_singleton_count, open_pos, weight, hyp_cept_start_prob, x, y);

              if (hyp_energy < cur_energy) {
                for (uint k = 0; k < cept_start_prob_.zDim(); k++)
                  cept_start_prob_(x, y, k) = hyp_cept_start_prob(x, y, k);
                cur_energy = hyp_energy;
              }
            }
            if (msolve_mode_ == MSSolvePGD)
              nondeficient_inter_m_step_with_interpolation(fnondef_ceptstart_singleton_count,open_pos,weight,x,y,cur_energy);
            else
              nondeficient_inter_m_step_unconstrained(fnondef_ceptstart_singleton_count, open_pos, weight, x, y, cur_energy);
          }
        }
      }
      else {

        for (uint x = 0; x < cept_start_prob_.xDim(); x++) {

          std::cerr << "inter-m-step(" << x << ",*)" << std::endl;

          for (uint y = 0; y < cept_start_prob_.yDim(); y++) {

            if (!reduce_deficiency_) {

              double sum = 0.0;
              for (uint d = 0; d < cept_start_prob_.zDim(); d++)
                sum += fceptstart_count(x, y, d);

              if (sum > 1e-305) {

                const double inv_sum = 1.0 / sum;

                for (uint d = 0; d < cept_start_prob_.zDim(); d++)
                  cept_start_prob_(x, y, d) = std::max(fert_min_param_entry, inv_sum * fceptstart_count(x, y, d));
              }
            }
            else {
              if (msolve_mode_ == MSSolvePGD)
                inter_distortion_m_step(fceptstart_count,finter_par_span_count(x,y),x,y);
              else
                inter_distortion_m_step_unconstrained(fceptstart_count, finter_par_span_count(x, y), x, y);
            }
          }
        }

        par2nonpar_inter_distortion();
      }
    }

    //std::cerr << "B" << std::endl;

    //b) within-cept
    if (!uniform_intra_prob_) {
      if (nondeficient_) {

        IBM4WithinCeptModel hyp_withincept_prob = within_cept_prob_;

        for (uint x = 0; x < within_cept_prob_.xDim(); x++) {
          std::cerr << "calling nondeficient intra-m-step(" << x << ")" << std::endl;

          std::map<Math1D::Vector<uchar,uchar>,double>& cur_map = nondef_within_cept_count[x];
          std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> > count(cur_map.size());

          uint k = 0;
          for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = cur_map.begin(); it != cur_map.end(); it++) {
            //std::pair<Math1D::Vector<uchar,uchar>,double> new_pair;

            count[k] = *it;
            k++;
          }

          cur_map.clear();

          double sum = 0.0;
          for (uint d = 0; d < within_cept_prob_.yDim(); d++)
            sum += fwithincept_count(x, d);

          if (sum > 1e-305) {

            const double inv_sum = 1.0 / sum;
            for (uint d = 0; d < within_cept_prob_.yDim(); d++)
              hyp_withincept_prob(x, d) = std::max(fert_min_param_entry, inv_sum * fwithincept_count(x, d));

            double cur_energy = nondeficient_intra_m_step_energy(fnondef_withincept_singleton_count, count, within_cept_prob_, x);
            double hyp_energy = nondeficient_intra_m_step_energy(fnondef_withincept_singleton_count, count, hyp_withincept_prob, x);

            if (hyp_energy < cur_energy) {
              for (uint d = 0; d < within_cept_prob_.yDim(); d++)
                within_cept_prob_(x, d) = hyp_withincept_prob(x, d);
            }
          }
          if (msolve_mode_ == MSSolvePGD)
            nondeficient_intra_m_step(fnondef_withincept_singleton_count,count,x);
          else
            nondeficient_intra_m_step_unconstrained(fnondef_withincept_singleton_count, count, x);
        }
      }
      else {

        for (uint x = 0; x < within_cept_prob_.xDim(); x++) {

          std::cerr << "intra-m-step(" << x << ")" << std::endl;

          if (!reduce_deficiency_) {

            double sum = 0.0;
            for (uint d = 0; d < within_cept_prob_.yDim(); d++)
              sum += fwithincept_count(x, d);

            if (sum > 1e-305) {

              const double inv_sum = 1.0 / sum;

              for (uint d = 0; d < within_cept_prob_.yDim(); d++)
                within_cept_prob_(x, d) = std::max(inv_sum * fwithincept_count(x, d), fert_min_param_entry);
            }
          }
          else {
            if (msolve_mode_ == MSSolvePGD)
              intra_distortion_m_step(fwithincept_count,fintra_par_span_count,x);
            else
              intra_distortion_m_step_unconstrained(fwithincept_count, fintra_par_span_count, x);
          }
        }

        par2nonpar_intra_distortion();
      }
    }

    //std::cerr << "C" << std::endl;

    //c) sentence start prob
    if (use_sentence_start_prob_) {

      if (msolve_mode_ == MSSolvePGD)
        start_prob_m_step(fsentence_start_count, fstart_span_count, sentence_start_parameters_);
      else
        start_prob_m_step_unconstrained(fsentence_start_count, fstart_span_count, sentence_start_parameters_);
      par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
    }

    max_perplexity += reg_term;
    approx_sum_perplexity += reg_term;

    max_perplexity /= source_sentence_.size();
    approx_sum_perplexity /= source_sentence_.size();

    std::string transfer = (fert_trainer != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "IBM-4 max-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << max_perplexity << std::endl;
    std::cerr << "IBM-4 approx-sum-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << approx_sum_perplexity << std::endl;


    if (possible_ref_alignments_.size() > 0) {

      std::cerr << "#### IBM-4-AER in between iterations #" << (iter - 1) << " and "
                << iter << transfer << ": " << FertilityModelTrainerBase::AER() << std::endl;
      std::cerr << "#### IBM-4-fmeasure in between iterations #" << (iter - 1)
                << " and " << iter << transfer << ": " << FertilityModelTrainerBase::f_measure() << std::endl;
      std::cerr << "#### IBM-4-DAE/S in between iterations #" << (iter - 1) << " and "
                << iter << transfer << ": " << FertilityModelTrainerBase::DAE_S() << std::endl;

      double postdec_aer;
      double postdec_fmeasure;
      double postdec_daes;
      PostdecEval(postdec_aer, postdec_fmeasure, postdec_daes, 0.25);
      std::cerr << "#### IBM-4-Postdec-AER in between iterations #" << (iter - 1)
                << " and " << iter << transfer << ": " << postdec_aer << std::endl;
      std::cerr << "#### IBM-4-Postdec-fmeasure in between iterations #" << (iter - 1)
                << " and " << iter << transfer << ": " << postdec_fmeasure << std::endl;
      std::cerr << "#### IBM-4-Postdec-DAE/S in between iterations #" << (iter - 1)
                << " and " << iter << transfer << ": " << postdec_daes << std::endl;
    }

    std::cerr << (((double)sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" << std::endl;
  }

  std::cerr << "spent " << hillclimbtime << " seconds on IBM-4-hillclimbing" << std::endl;
  std::cerr << "spent " << countcollecttime << " seconds on IBM-4-distortion count collection" << std::endl;

  iter_offs_ = iter - 1;
}

void IBM4Trainer::train_viterbi(uint nIter, FertilityModelTrainerBase* fert_trainer, const HmmWrapperWithClasses* passed_wrapper)
{
  const uint nSentences = source_sentence_.size();

  std::cerr << "starting IBM-4 Viterbi training without constraints";
  if (fert_trainer != 0)
    std::cerr << " (init from " << fert_trainer->model_name() << ") ";
  std::cerr << std::endl;

  double max_perplexity = 0.0;

  IBM4CeptStartModel fceptstart_count(cept_start_prob_.xDim(), cept_start_prob_.yDim(), 2 * maxJ_ - 1, MAKENAME(fceptstart_count));
  IBM4WithinCeptModel fwithincept_count(within_cept_prob_.xDim(), within_cept_prob_.yDim(), MAKENAME(fwithincept_count));

  if (log_table_.size() < nSentences) {
    EXIT("passed log table is not large enough.");
  }

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords, MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));
  NamedStorage1D<Math1D::Vector<double> > ffertclass_count(MAKENAME(ffert_count));

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  if (fertprob_sharing_) {
    ffertclass_count.resize(nTFertClasses_);
    for (uint i = 1; i < nTargetWords; i++) {
      uint c = tfert_class_[i];
      if (fertility_prob_[i].size() > ffertclass_count[c].size())
        ffertclass_count[c].resize_dirty(fertility_prob_[i].size());
    }
  }

  long double fzero_count;
  long double fnonzero_count;

  Math1D::Vector<double> fnull_intra_count(null_intra_prob_.size());

  //new variant
  Storage2D<Math2D::Matrix<double> > finter_par_span_count;
  Math2D::Matrix<double> fintra_par_span_count;
  Math1D::Vector<double> fsentence_start_count(maxJ_);
  Math1D::Vector<double> fstart_span_count(maxJ_);
  if (reduce_deficiency_) {
    finter_par_span_count.resize(nSourceClasses_, nTargetClasses_);
    for (uint s = 0; s < nSourceClasses_; s++) {
      for (uint t = 0; t < nTargetClasses_; t++) {
        //finter_par_span_count(s,t).resize(displacement_offset_+1,cept_start_prob_.zDim());

        //we exploit that a span always starts below (or at) the zero offset and always ends above (or at) the zero offset
        // => remove superfluous parts of the matrix
        // => for the second index you have to add displacement_offset_ to get the true index
        finter_par_span_count(s, t).resize(displacement_offset_ + 1, displacement_offset_ + 1);
      }
    }

    fintra_par_span_count.resize(within_cept_prob_.xDim(), maxJ_ + 1);
  }

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    std::cerr << "******* IBM-4 Viterbi-iteration #" << iter << std::endl;

    if (passed_wrapper != 0
        && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    fnull_intra_count.set_constant(0.0);

    fceptstart_count.set_constant(0.0);
    fwithincept_count.set_constant(0.0);

    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    if (fertprob_sharing_) {
      for (uint c = 0; c < ffertclass_count.size(); c++)
        ffertclass_count[c].set_constant(0.0);
    }

    fsentence_start_count.set_constant(0.0);
    fstart_span_count.set_constant(0.0);

    Storage2D<std::map<Math1D::Vector<uchar,uchar>,uint> > nondef_cept_start_count(nSourceClasses_, nTargetClasses_);
    Storage1D<std::map<Math1D::Vector<uchar,uchar>,uint> >nondef_within_cept_count(nTargetClasses_);

    //this count is almost like fceptstart_count, but includes no terms where only one position is available
    IBM4CeptStartModel fnondef_ceptstart_singleton_count(cept_start_prob_.xDim(), cept_start_prob_.yDim(), 2 * maxJ_ - 1, 0.0,
        MAKENAME(fnondef_ceptstart_singleton_count));

    //same for this count and fnondef_withincept_count
    IBM4WithinCeptModel fnondef_withincept_singleton_count(within_cept_prob_.xDim(), within_cept_prob_.yDim(), 0.0,
        MAKENAME(fnondef_withincept_singleton_count));

    for (uint s = 0; s < finter_par_span_count.xDim(); s++)
      for (uint t = 0; t < finter_par_span_count.yDim(); t++)
        finter_par_span_count(s, t).set_constant(0.0);

    fintra_par_span_count.set_constant(0.0);

    SingleLookupTable aux_lookup;

    max_perplexity = 0.0;

    for (size_t s = 0; s < nSentences; s++) {

      //DEBUG
      uint prev_sum_iter = sum_iter;
      //END_DEBUG

      if ((s % 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

      //these will not actually be used, but need to be passed to the hillclimbing routine
      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      //std::clock_t tHillclimbStart, tHillclimbEnd;
      //tHillclimbStart = std::clock();

      long double best_prob = 0.0;

      if (fert_trainer != 0 && iter == 1) {

        best_prob = fert_trainer->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter,
                    fertility, expansion_move_prob, swap_move_prob, best_known_alignment_[s]);

        //DEBUG
#ifndef NDEBUG
        long double align_prob = FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]);

        if (isinf(align_prob) || isnan(align_prob) || align_prob == 0.0) {

          std::cerr << "ERROR: after hillclimbing: align-prob for sentence " << s << " has prob " << align_prob << std::endl;

          print_alignment_prob_factors(source_sentence_[s], target_sentence_[s], slookup_[s], best_known_alignment_[s]);

          exit(1);
        }
#endif
        //END_DEBUG
      }
      else {

        best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, best_known_alignment_[s]);
      }

      max_perplexity -= logl(best_prob);

      //DEBUG
      if (isinf(max_perplexity)) {

        std::cerr << "ERROR: inf after sentence  " << s << ", last alignment prob: " << best_prob << std::endl;
        std::cerr << "J = " << curJ << ", I = " << curI << std::endl;
        std::cerr << "preceding number of hillclimbing iterations: " << (sum_iter - prev_sum_iter) << std::endl;

        exit(1);
      }
      //END_DEBUG

      //tHillclimbEnd = std::clock();

      /**** update empty word counts *****/

      fzero_count += fertility[0];
      fnonzero_count += curJ - 2 * fertility[0];

      /**** update fertility counts *****/
      for (uint i = 1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i - 1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }

      /**** update dictionary counts *****/
      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = best_known_alignment_[s][j];

        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj - 1]][cur_lookup(j, cur_aj - 1)] += 1.0;
        }
        else {
          fwcount[0][s_idx - 1] += 1.0;
        }
      }

      /**** update distortion counts *****/
      NamedStorage1D<std::vector<ushort> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));

      for (uint j = 0; j < curJ; j++) {
        const uint cur_aj = best_known_alignment_[s][j];
        aligned_source_words[cur_aj].push_back(j);
      }

      if (empty_word_model_ == FertNullIntra) {

        const std::vector<ushort>& null_words = aligned_source_words[0];
        if (null_words.size() > 1) {
          for (uint k = 1; k < null_words.size(); k++) {
            const ushort cur = null_words[k];
            const ushort prev = null_words[k - 1];
            fnull_intra_count[cur - prev] += 1.0;
          }
        }
      }
      // handle viterbi alignment
      int cur_prev_cept = -1;
      uint prev_cept_center = MAX_UINT;
      for (uint i = 1; i <= curI; i++) {

        uint tclass = target_class_[cur_target[i - 1]];

        if (fertility[i] > 0) {

          const std::vector<ushort>& cur_aligned_source_words = aligned_source_words[i];

          //a) update head prob
          if (cur_prev_cept >= 0 && cept_start_mode_ != IBM4UNIFORM) {

            const uint sclass = source_class_[cur_source[cur_aligned_source_words[0]]];

            if (inter_dist_mode_ == IBM4InterDistModePrevious)
              tclass = target_class_[cur_target[cur_prev_cept - 1]];

            int diff = cur_aligned_source_words[0] - prev_cept_center;
            diff += displacement_offset_;

            fceptstart_count(sclass, tclass, diff) += 1.0;

            if (!nondeficient_ && reduce_deficiency_) {

              //new variant
              uint diff_start = displacement_offset_ - prev_cept_center;
              uint diff_end = diff_start + curJ - 1;
              finter_par_span_count(sclass, tclass)(diff_start, diff_end - displacement_offset_) += 1.0;
            }
          }
          else if (use_sentence_start_prob_) {
            fsentence_start_count[cur_aligned_source_words[0]] += 1.0;
            fstart_span_count[curJ - 1] += 1.0;
          }
          //b) update within-cept prob
          int prev_aligned_j = cur_aligned_source_words[0];

          for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

            const int cur_j = cur_aligned_source_words[k];

            const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[cur_source[cur_j]] : target_class_[cur_target[i - 1]];

            int diff = cur_j - prev_aligned_j;
            fwithincept_count(cur_class, diff) += 1.0;

            if (reduce_deficiency_) {
              fintra_par_span_count(cur_class, curJ - prev_aligned_j) += 1.0;
            }

            prev_aligned_j = cur_j;
          }

          switch (cept_start_mode_) {
          case IBM4CENTER: {

            double sum = vec_sum(cur_aligned_source_words);
            prev_cept_center = (uint) round(sum / fertility[i]);
            break;
          }
          case IBM4FIRST:
            prev_cept_center = cur_aligned_source_words[0];
            break;
          case IBM4LAST:
            prev_cept_center = cur_aligned_source_words.back();
            break;
          case IBM4UNIFORM:
            prev_cept_center = cur_aligned_source_words[0];     //will not be used
            break;
          default:
            assert(false);
          }

          cur_prev_cept = i;
        }
      }

      if (nondeficient_) {

        //a) best known alignment (=mode)
        int prev_cept_center = -1;
        int prev_cept = -1;

        Storage1D<bool> fixed(curJ, false);

        for (uint i = 1; i <= curI; i++) {

          const std::vector<ushort>& cur_aligned_source_words = aligned_source_words[i];

          if (cur_aligned_source_words.size() > 0) {

            const uint ti = cur_target[i - 1];
            uint tclass = target_class_[ti];

            const int first_j = cur_aligned_source_words[0];

            uint nToRemove = cur_aligned_source_words.size() - 1;

            //handle the head of the cept
            if (prev_cept_center != -1 && cept_start_mode_ != IBM4UNIFORM) {

              const uint sclass = source_class_[cur_source[first_j]];

              if (inter_dist_mode_ == IBM4InterDistModePrevious)
                tclass = target_class_[cur_target[prev_cept - 1]];

              std::vector<uchar> possible_diffs;

              for (int j = 0; j < int (curJ); j++) {
                if (!fixed[j]) {
                  possible_diffs.push_back(j - prev_cept_center + displacement_offset_);
                }
              }

              if (nToRemove > 0) {
                possible_diffs.resize(possible_diffs.size() - nToRemove);
              }

              if (possible_diffs.size() > 1) {  //no use storing cases where only one pos. is available

                Math1D::Vector<uchar,uchar> vec_possible_diffs(possible_diffs.size());
                for (uint k = 0; k < possible_diffs.size(); k++)
                  vec_possible_diffs[k] = possible_diffs[k];

                fnondef_ceptstart_singleton_count(sclass, tclass, first_j - prev_cept_center + displacement_offset_) += 1.0;

                nondef_cept_start_count(sclass, tclass)[vec_possible_diffs] += 1.0;
              }
            }
            fixed[first_j] = true;

            //handle the body of the cept
            int prev_j = first_j;
            if (!uniform_intra_prob_) {
              for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

                nToRemove--;

                std::vector<uchar> possible;

                const int cur_j = cur_aligned_source_words[k];

                const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[cur_source[cur_j]]
                                       : target_class_[cur_target[i - 1]];

                for (int j = prev_j + 1; j < int (curJ); j++) {
                  possible.push_back(j - prev_j);
                }

                if (nToRemove > 0) {
                  possible.resize(possible.size() - nToRemove);
                }

                if (possible.size() > 1) {        //no use storing cases where only one pos. is available

                  Math1D::Vector<uchar,uchar> vec_possible(possible.size());
                  for (uint k = 0; k < possible.size(); k++)
                    vec_possible[k] = possible[k];

                  fnondef_withincept_singleton_count(cur_class, cur_j - prev_j) += 1.0;
                  nondef_within_cept_count[cur_class][vec_possible] += 1.0;
                }
                //std::cerr << "dp: tclass " << tclass << ", prob: " << cur_intra_distortion_prob(tclass,cur_j,prev_j) << std::endl;

                fixed[cur_j] = true;

                prev_j = cur_j;
              }
            }
            else
              prev_j = cur_aligned_source_words.back();

            switch (cept_start_mode_) {
            case IBM4CENTER: {

              //compute the center of this cept and store the result in prev_cept_center
              double sum = vec_sum(cur_aligned_source_words);
              prev_cept_center = (int)round(sum / cur_aligned_source_words.size());
              break;
            }
            case IBM4FIRST:
              prev_cept_center = first_j;
              break;
            case IBM4LAST: {
              prev_cept_center = prev_j;      //was set to the last pos in the above llop
              break;
            }
            case IBM4UNIFORM:
              prev_cept_center = first_j;       //will not be used
              break;
            default:
              assert(false);
            }

            prev_cept = i;
            assert(prev_cept_center >= 0);
          }
        }
      }
      //clean up cache
      for (uint j = 0; j < inter_distortion_cache_[curJ].size(); j++)
        inter_distortion_cache_[curJ][j].clear();

    }                           // loop over sentences finished

    /***** update probability models from counts *******/

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
      p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);
    }

    if (empty_word_model_ == FertNullIntra) {

      double sum = fnull_intra_count.sum();
      if (sum > 1e-300) {

        for (uint k=0; k < null_intra_prob_.size(); k++)
          null_intra_prob_[k] = std::max(fert_min_param_entry,fnull_intra_count[k] / sum);
      }
    }

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

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
    update_dict_from_counts(fwcount, prior_weight_, 0.0, false, 0.0, 0, dict_, fert_min_dict_entry);

    //update fertility probabilities
    update_fertility_prob(ffert_count, fert_min_param_entry, false); //needed at least with fert-prob-sharing

    //update distortion probabilities

    //a) cept-start (inter distortion)
    if (cept_start_mode_ != IBM4UNIFORM) {
      if (!nondeficient_) {

        IBM4CeptStartModel hyp_cept_start_prob;
        if (reduce_deficiency_)
          hyp_cept_start_prob = cept_start_prob_;

        for (uint x = 0; x < cept_start_prob_.xDim(); x++) {

          std::cerr << "inter m-step(" << x << ",*)" << std::endl;

          for (uint y = 0; y < cept_start_prob_.yDim(); y++) {

            if (!reduce_deficiency_) {

              double sum = 0.0;
              for (uint d = 0; d < cept_start_prob_.zDim(); d++)
                sum += fceptstart_count(x, y, d);

              if (sum > 1e-305) {

                const double inv_sum = 1.0 / sum;

                for (uint d = 0; d < cept_start_prob_.zDim(); d++)
                  cept_start_prob_(x, y, d) = std::max(fert_min_param_entry, inv_sum * fceptstart_count(x, y, d));
              }
            }
            else {
              inter_distortion_m_step(fceptstart_count, finter_par_span_count(x, y), x, y);
            }
          }
        }

        par2nonpar_inter_distortion();
      }
      else {

        for (uint x = 0; x < cept_start_prob_.xDim(); x++) {
          std::cerr << "nondeficient inter-m-step(" << x << ",*)" << std::endl;
          for (uint y = 0; y < cept_start_prob_.yDim(); y++) {

            std::map<Math1D::Vector<uchar,uchar>,uint>& cur_map = nondef_cept_start_count(x, y);

            std::vector<Math1D::Vector<uchar,uchar> > open_diff(cur_map.size());
            std::vector<double> weight(cur_map.size());

            uint k = 0;
            for (std::map<Math1D::Vector<uchar,uchar>,uint>::const_iterator it = cur_map.begin(); it != cur_map.end(); it++) {
              open_diff[k] = it->first;
              weight[k] = it->second;
              k++;
            }

            cur_map.clear();

            double cur_energy = nondeficient_inter_m_step_energy(fnondef_ceptstart_singleton_count, open_diff, weight, cept_start_prob_, x, y);

            double sum = 0.0;
            for (uint k = 0; k < cept_start_prob_.zDim(); k++) {
              sum += fceptstart_count(x, y, k);
            }

            if (sum > 1e-300) {

              IBM4CeptStartModel hyp_cept_start_prob = cept_start_prob_;

              for (uint k = 0; k < cept_start_prob_.zDim(); k++)
                hyp_cept_start_prob(x, y, k) = std::max(fert_min_param_entry, fceptstart_count(x, y, k) / sum);

              double hyp_energy = nondeficient_inter_m_step_energy(fnondef_ceptstart_singleton_count, open_diff, weight, hyp_cept_start_prob, x, y);

              if (hyp_energy < cur_energy) {
                cept_start_prob_ = hyp_cept_start_prob;
                cur_energy = hyp_energy;
              }
            }

            nondeficient_inter_m_step_with_interpolation(fnondef_ceptstart_singleton_count, open_diff, weight, x, y, cur_energy);
          }
        }
      }
    }

    //b) within-cept (intra distortion)
    if (!uniform_intra_prob_) {
      if (!nondeficient_) {

        IBM4WithinCeptModel hyp_withincept_prob;
        if (reduce_deficiency_)
          hyp_withincept_prob = within_cept_prob_;

        for (uint x = 0; x < within_cept_prob_.xDim(); x++) {

          std::cerr << "intra-m-step(" << x << ")" << std::endl;

          if (!reduce_deficiency_) {

            double sum = 0.0;
            for (uint d = 0; d < within_cept_prob_.yDim(); d++)
              sum += fwithincept_count(x, d);

            if (sum > 1e-305) {

              const double inv_sum = 1.0 / sum;

              for (uint d = 0; d < within_cept_prob_.yDim(); d++)
                within_cept_prob_(x, d) = std::max(inv_sum * fwithincept_count(x, d), fert_min_param_entry);
            }
          }
          else {
            intra_distortion_m_step(fwithincept_count, fintra_par_span_count, x);
          }
        }

        par2nonpar_intra_distortion();
      }
      else {

        IBM4WithinCeptModel hyp_withincept_prob = within_cept_prob_;

        for (uint x = 0; x < within_cept_prob_.xDim(); x++) {
          std::cerr << "calling nondeficient intra-m-step(" << x << ")" << std::endl;

          std::map<Math1D::Vector<uchar,uchar>,uint>& cur_map = nondef_within_cept_count[x];

          std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> > count(cur_map.size());

          uint k = 0;
          for (std::map<Math1D::Vector<uchar,uchar>,uint >::const_iterator it = cur_map.begin(); it != cur_map.end(); it++) {

            count[k] = *it;
            k++;
          }

          cur_map.clear();

          double sum = 0.0;
          for (uint d = 0; d < within_cept_prob_.yDim(); d++)
            sum += fwithincept_count(x, d);

          if (sum > 1e-305) {

            const double inv_sum = 1.0 / sum;
            for (uint d = 0; d < within_cept_prob_.yDim(); d++)
              hyp_withincept_prob(x, d) = std::max(fert_min_param_entry, inv_sum * fwithincept_count(x, d));

            double cur_energy = nondeficient_intra_m_step_energy(fnondef_withincept_singleton_count, count, within_cept_prob_, x);
            double hyp_energy = nondeficient_intra_m_step_energy(fnondef_withincept_singleton_count, count, hyp_withincept_prob, x);

            if (hyp_energy < cur_energy) {
              for (uint d = 0; d < within_cept_prob_.yDim(); d++)
                within_cept_prob_(x, d) = hyp_withincept_prob(x, d);

              cur_energy = hyp_energy;
            }
          }

          nondeficient_intra_m_step(fnondef_withincept_singleton_count, count, x);
        }
      }
    }

    //c) sentence start prob
    if (use_sentence_start_prob_) {

      if (msolve_mode_ == MSSolvePGD)
        start_prob_m_step(fsentence_start_count, fstart_span_count, sentence_start_parameters_);
      else
        start_prob_m_step_unconstrained(fsentence_start_count, fstart_span_count, sentence_start_parameters_);

      par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
    }
    //DEBUG
#ifndef NDEBUG
    for (size_t s = 0; s < source_sentence_.size(); s++) {

      long double align_prob = FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]);

      if (isinf(align_prob) || isnan(align_prob) || align_prob == 0.0) {

        std::cerr << "ERROR: after parameter update: align-prob for sentence " << s << " has prob " << align_prob << std::endl;

        const SingleLookupTable& cur_lookup = get_wordlookup(source_sentence_[s], target_sentence_[s], wcooc_, nSourceWords_, slookup_[s], aux_lookup);

        print_alignment_prob_factors(source_sentence_[s], target_sentence_[s], cur_lookup, best_known_alignment_[s]);

        exit(1);
      }
    }
#endif
    //END_DEBUG

    max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
    max_perplexity /= source_sentence_.size();

    std::cerr << (((double)sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" << std::endl;

    std::string transfer = (fert_trainer != 0 && iter == 1) ? " (transfer) " : "";
    std::cerr << "IBM-4 max-perplex-energy in between iterations #" << (iter - 1) << " and " << iter << transfer << ": " << max_perplexity << std::endl;

    //ICM STAGE
    if (fert_trainer == 0) {
      //no point doing ICM in a transfer iteration
      //in nondeficient mode, ICM does well at decreasing the energy, but it heavily aligns to the rare words

      std::cerr << "starting ICM" << std::endl;

      const double log_pzero = std::log(p_zero_);
      const double log_pnonzero = std::log(p_nonzero_);

      Math1D::NamedVector<uint> dict_sum(fwcount.size(), MAKENAME(dict_sum));
      for (uint k = 0; k < fwcount.size(); k++)
        dict_sum[k] = fwcount[k].sum();

      if (fertprob_sharing_) {

        for (uint i = 1; i < nTargetWords; i++) {
          uint c = tfert_class_[i];
          for (uint k = 0; k < ffert_count[i].size(); k++)
            ffertclass_count[c][k] += ffert_count[i][k];
        }
      }

      uint nSwitches = 0;

      for (size_t s = 0; s < nSentences; s++) {

        if ((s % 10000) == 0)
          std::cerr << "sentence pair #" << s << std::endl;

        const Storage1D<uint>& cur_source = source_sentence_[s];
        const Storage1D<uint>& cur_target = target_sentence_[s];
        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

        Math1D::Vector<AlignBaseType>& cur_best_known_alignment = best_known_alignment_[s];

        const uint curI = cur_target.size();
        const uint curJ = cur_source.size();

        Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

        for (uint j = 0; j < curJ; j++)
          fertility[cur_best_known_alignment[j]]++;

        NamedStorage1D<std::vector<ushort> > hyp_aligned_source_words(curI + 1, MAKENAME(hyp_aligned_source_words));

        for (uint j = 0; j < curJ; j++) {

          uint aj = cur_best_known_alignment[j];
          hyp_aligned_source_words[aj].push_back(j);
        }

        long double cur_neglog_distort_prob = (nondeficient_) ? -logl(nondeficient_distortion_prob(cur_source, cur_target, hyp_aligned_source_words))
                                              : -logl(distortion_prob(cur_source, cur_target, hyp_aligned_source_words));

        long double cur_neglog_null_intra_prob = 0.0;
        if (empty_word_model_ == FertNullIntra) {
          cur_neglog_null_intra_prob = -logl(null_distortion_prob(hyp_aligned_source_words[0], curJ));
          assert(null_distortion_prob(hyp_aligned_source_words[0], curJ) > 0.0);
        }

        for (uint j = 0; j < curJ; j++) {

          const uint cur_aj = best_known_alignment_[s][j];
          const uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj - 1];
          const uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);
          Math1D::Vector<double>& cur_fert_count = ffert_count[cur_word];
          Math1D::Vector<double>& cur_dictcount = fwcount[cur_word];
          std::vector<ushort>& cur_hyp_aligned_source_words = hyp_aligned_source_words[cur_aj];

          double best_change = 0.0;
          uint new_aj = cur_aj;

          for (uint i = 0; i <= curI; i++) {

            //const uint cur_aj = best_known_alignment_[s][j];
            //const uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];

            /**** dict ***/
            //std::cerr << "i: " << i << ", cur_aj: " << cur_aj << std::endl;

            bool allowed = (cur_aj != i && (i != 0 || 2 * fertility[0] + 2 <= curJ));

            if (i != 0
                && (fertility[i] + 1) > fertility_limit_[cur_target[i - 1]])
              allowed = false;

            if (allowed) {

              const uint new_target_word = (i == 0) ? 0 : cur_target[i - 1];
              const Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

              vec_erase(cur_hyp_aligned_source_words, (ushort) j);
              hyp_aligned_source_words[i].push_back(j);
              vec_sort(hyp_aligned_source_words[i]);

              //std::cerr << "cur_word: " << cur_word << std::endl;
              //std::cerr << "new_word: " << new_target_word << std::endl;

              double change = 0.0;

              const Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
              const uint hyp_idx = (i == 0) ? cur_source[j] - 1 : cur_lookup(j, i - 1);

              change += common_icm_change(fertility, log_pzero, log_pnonzero, dict_sum, cur_dictcount, hyp_dictcount,
                                          prior_weight_[cur_word], prior_weight_[new_target_word], cur_fert_count, hyp_fert_count,
                                          ffertclass_count, cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, i, curJ);

              //std::cerr << "dist" << std::endl;

              /***** distortion ****/
              if (empty_word_model_ == FertNullIntra) {

                if (cur_aj == 0 || i == 0) {
                  change -= cur_neglog_null_intra_prob;
                  change += -logl(null_distortion_prob(hyp_aligned_source_words[0], curJ));
                }
              }

              change -= cur_neglog_distort_prob;

              const long double hyp_neglog_distort_prob = (nondeficient_) ?
                  -logl(nondeficient_distortion_prob(cur_source, cur_target, hyp_aligned_source_words)) :
                  -logl(distortion_prob(cur_source, cur_target, hyp_aligned_source_words));

              change += hyp_neglog_distort_prob;

              if (change < best_change) {
                best_change = change;
                new_aj = i;
              }
              //rollback the changes
              vec_erase(hyp_aligned_source_words[i], (ushort) j);
              cur_hyp_aligned_source_words.push_back(j);
              vec_sort(cur_hyp_aligned_source_words);
            }
          }

          if (best_change < -0.01) {

            //std::cerr << "changing!!" << std::endl;

            cur_best_known_alignment[j] = new_aj;
            nSwitches++;

            const uint new_target_word = (new_aj == 0) ? 0 : cur_target[new_aj - 1];
            Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
            const uint hyp_idx = (new_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, new_aj - 1);
            Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

            common_icm_count_change(dict_sum, cur_dictcount, hyp_dictcount, cur_fert_count, hyp_fert_count,
                                    ffertclass_count, cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, new_aj, fertility);

            if (cur_aj == 0) {
              fnonzero_count += 2.0;
              fzero_count--;
            }

            if (new_aj == 0) {
              fnonzero_count -= 2.0;
              fzero_count++;
            }

            vec_erase(cur_hyp_aligned_source_words, (ushort) j);
            hyp_aligned_source_words[new_aj].push_back(j);
            vec_sort(hyp_aligned_source_words[new_aj]);

            cur_neglog_distort_prob = (nondeficient_) ?
                                      -logl(nondeficient_distortion_prob(cur_source, cur_target, hyp_aligned_source_words)) :
                                      -logl(distortion_prob(cur_source, cur_target, hyp_aligned_source_words));

            if (empty_word_model_ == FertNullIntra) {
              if (cur_aj == 0 || new_aj == 0) {
                cur_neglog_null_intra_prob = -logl(null_distortion_prob(hyp_aligned_source_words[0], curJ));
                assert(null_distortion_prob(hyp_aligned_source_words[0], curJ) > 0.0);
              }
            }
          }
        }

        //clean up cache
        for (uint j = 0; j < inter_distortion_cache_[curJ].size(); j++)
          inter_distortion_cache_[curJ][j].clear();

      } //ICM-loop over sentences finished

      std::cerr << nSwitches << " changes in ICM stage" << std::endl;

      if (!fix_p0_) {
        double fsum = fzero_count + fnonzero_count;
        p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
        p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);
      }

      //update dictionary
      update_dict_from_counts(fwcount, prior_weight_, 0.0, false, 0.0, 0, dict_, fert_min_dict_entry);

      //update fertility probabilities
      update_fertility_prob(ffert_count, fert_min_param_entry);

      //TODO: think about updating distortion here as well (will have to recollect counts from the best known alignments)

      max_perplexity = 0.0;
      for (uint s = 0; s < source_sentence_.size(); s++) {
        max_perplexity -= logl(FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]));
      }

      max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
      max_perplexity /= source_sentence_.size();

      std::cerr << "IBM-4 max-perplex-energy after iteration #" << iter << transfer << ": " << max_perplexity << std::endl;
    }

    if (possible_ref_alignments_.size() > 0) {

      std::cerr << "#### IBM-4-AER after iteration #" << iter << transfer << ": " << FertilityModelTrainerBase::AER() << std::endl;
      std::cerr << "#### IBM-4-fmeasure after iterations #" << iter << transfer  << ": " << FertilityModelTrainerBase::f_measure() << std::endl;
      std::cerr << "#### IBM-4-DAE/S after iteration #" << iter << transfer << ": " << FertilityModelTrainerBase::DAE_S() << std::endl;
    }
  }

  iter_offs_ = iter - 1;
}
