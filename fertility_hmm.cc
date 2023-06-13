/***** written by Thomas Schoenemann as a private person, since August 2018 ****/

#include "fertility_hmm.hh"
#include "training_common.hh"   // for get_wordlookup() and dictionary m-step
#include "projection.hh"


//DEBUG
bool __debug = false;
//END_DEBUG

FertilityHMMTrainer::FertilityHMMTrainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
    const Storage1D<Math1D::Vector<uint> >& target_sentence, const Math1D::Vector<WordClassType>& target_class,
    const RefAlignmentStructure& sure_ref_alignments, const RefAlignmentStructure& possible_ref_alignments,
    SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class,
    uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
    const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
    const HmmOptions& options, const FertModelOptions& fert_options, bool no_factorial)
  : FertilityModelTrainer(source_sentence, slookup, target_sentence, dict, wcooc, tfert_class, nSourceWords, nTargetWords, prior_weight,
                          sure_ref_alignments, possible_ref_alignments, log_table, xlogx_table, fert_options, no_factorial),
    options_(options), dist_grouping_param_(1.0), target_class_(target_class)
{
  const uint nClasses = target_class_.max() + 1;
  dist_params_.resize(2 * maxI_ - 1, nClasses, 1.0);
  dist_grouping_param_.resize(nClasses);

  std::set<uint> seenIs;

  const size_t nSentences = source_sentence_.size();
  for (size_t s = 0; s < nSentences; s++) {

    const uint curI = target_sentence_[s].size();
    seenIs.insert(curI);
  }

  align_model_.resize_dirty(maxI_);     //note: access using I-1

  for (std::set<uint>::const_iterator it = seenIs.begin();
       it != seenIs.end(); it++) {
    const uint I = *it;

    //x = new index, y = given index
    align_model_[I - 1].resize_dirty(I, I, nClasses);   //no empty words!
    align_model_[I - 1].set_constant(1.0 / I);
  }
}

/*virtual*/ std::string FertilityHMMTrainer::model_name() const
{
  return "FertilityHMM-SingleClass";
}

void FertilityHMMTrainer::init_from_hmm(HmmFertInterfaceTargetClasses& prev_model, const Math2D::Matrix<double>& dist_params,
                                        const Math1D::Vector<double>& dist_grouping_param,
                                        bool clear_prev, bool count_collection, bool /*viterbi */ )
{
  std::cerr << "initializing Fertility-HMM from HMM" << std::endl;

  const HmmWrapperWithTargetClasses& hmm_wrapper = prev_model.hmm_wrapper();
  set_hmm_alignments(hmm_wrapper);

  //std::cerr << "done setting hmm alignments" << std::endl;

  assert(dist_params.xDim() == dist_params_.xDim());
  if (dist_params.yDim() == 1 && dist_params_.yDim() > 1) {
    for (uint c = 0; c < dist_params_.yDim(); c++)
      for (uint k = 0; k < dist_params_.xDim(); k++)
        dist_params_(k, c) = dist_params(k, 0);
    dist_grouping_param_.set_constant(dist_grouping_param[0]);
  }
  else if (dist_params.yDim() > 1 && dist_params_.yDim() == 1) {
    dist_params_.set_constant(0.0);
    dist_grouping_param_.set_constant(0.0);
    for (uint c = 0; c < dist_params_.yDim(); c++) {
      for (uint k = 0; k < dist_params_.xDim(); k++) {
        dist_params_(k, 0) += dist_params(k, c);
      }
      dist_grouping_param_[0] += dist_grouping_param[c];
    }
    dist_params_ *= 1.0 / dist_params.yDim();
    dist_grouping_param_[0] *=  1.0 / dist_grouping_param.size();
  }
  else {
    assert(dist_params.yDim() == dist_params_.yDim());
    dist_params_ = dist_params;
    dist_grouping_param_ = dist_grouping_param;
  }

  /*** init hmm parameters ***/
  
  par2nonpar();

  std::cerr << "done initializing the alignment model" << std::endl;

  /*** leave p_zero_ as is ***/

  if (count_collection) {

    train_em(1, &prev_model);

    iter_offs_ = 1;
  }
  else {

    /*** init alignments and fertility parameters ***/

    init_fertilities(0);
  }

  if (clear_prev)
    prev_model.release_memory();
}

void FertilityHMMTrainer::init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper,
    const Math2D::Matrix<double>& dist_params, const Math1D::Vector<double>& dist_grouping_param, bool clear_prev,
    bool count_collection, bool viterbi)
{
  std::cerr << "******** initializing Fertility-HMM from " << prev_model->model_name() << " *******" << std::endl;

  //std::cerr << "done setting hmm alignments" << std::endl;

  assert(dist_params.xDim() == dist_params_.xDim());
  assert(dist_params.yDim() == dist_grouping_param.size());
  if (dist_params.yDim() == 1 && dist_params_.yDim() > 1) {
    for (uint c = 0; c < dist_params_.yDim(); c++)
      for (uint k = 0; k < dist_params_.xDim(); k++)
        dist_params_(k, c) = dist_params(k, 0);
    dist_grouping_param_.set_constant(dist_grouping_param[0]);
  }
  else if (dist_params.yDim() > 1 && dist_params_.yDim() == 1) {
    dist_params_.set_constant(0.0);
    dist_grouping_param_.set_constant(0.0);
    for (uint c = 0; c < dist_params_.yDim(); c++) {
      for (uint k = 0; k < dist_params_.xDim(); k++) {
        dist_params_(k, 0) += dist_params(k, c);
      }
      dist_grouping_param_[0] += dist_grouping_param[c];
    }
    dist_params_ *= 1.0 / dist_params.yDim();
    dist_grouping_param_[0] *=  1.0 / dist_grouping_param.size();
  }
  else {
    //std::cerr << "passed dim: " << dist_params.yDim() << ", should be: " << dist_params_.yDim() << std::endl;
    assert(dist_params.yDim() == dist_params_.yDim());
    dist_params_ = dist_params;
    dist_grouping_param_ = dist_grouping_param;
  }

  /*** init hmm parameters ***/

  par2nonpar();

  std::cerr << "done initializing the alignment model" << std::endl;

  /*** leave p_zero_ as is ***/

  if (count_collection) {

    best_known_alignment_ = prev_model->best_alignments();

    train_em(1, prev_model);

    iter_offs_ = 1;
  }
  else {

    best_known_alignment_ = prev_model->update_alignments_unconstrained(true, passed_wrapper);

    const FertilityModelTrainer* fert_model = dynamic_cast<const FertilityModelTrainer*>(prev_model);

    if (fert_model == 0) {
      init_fertilities(0);      //alignments were already updated an set
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
  }

  if (clear_prev)
    prev_model->release_memory();
}

//training without constraints on uncovered positions.
//This is based on the EM-algorithm, where the E-step uses heuristics
void FertilityHMMTrainer::train_em(uint nIter, FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper)
{
  const size_t nSentences = source_sentence_.size();

  std::cerr << "starting Fertility-HMM training without constraints" << std::endl;

  double max_perplexity = 0.0;
  double approx_sum_perplexity = 0.0;

  double dict_weight_sum = (prior_weight_active_) ? 1.0 : 0.0; //only used as a flag

  SingleLookupTable aux_lookup;

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords, MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  FullHMMAlignmentModelSingleClass falign_count = align_model_;

  double fzero_count;
  double fnonzero_count;

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    std::cerr << "******* Fertility-HMM EM-iteration #" << iter << std::endl;

    if (passed_wrapper != 0
        && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    for (uint I = 0; I < falign_count.size(); I++)
      falign_count[I].set_constant(0.0);

    max_perplexity = 0.0;
    approx_sum_perplexity = 0.0;

    uint nNotConverged = 0;

    for (size_t s = 0; s < nSentences; s++) {

      if ((s % 10000) == 0)
        //if ((s% 100) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      Math1D::NamedVector<uint>fertility(curI + 1, 0, MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      long double best_prob;

      if (prev_model != 0) {

        best_prob = prev_model->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter,
                    fertility, expansion_move_prob, swap_move_prob, cur_alignment);
      }
      else {

        best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, cur_alignment);
      }

      max_perplexity -= std::log(best_prob);

      const long double expansion_prob = expansion_move_prob.sum();
      const long double swap_prob = swap_mass(swap_move_prob);

      const long double sentence_prob = best_prob + expansion_prob + swap_prob;

      approx_sum_perplexity -= std::log(sentence_prob);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      if (isnan(inv_sentence_prob)) {

        std::cerr << "best prob: " << best_prob << std::endl;
        std::cerr << "swap prob: " << swap_prob << std::endl;
        std::cerr << "expansion prob: " << expansion_prob << std::endl;
        exit(1);
      }

      assert(!isnan(inv_sentence_prob));

      Math2D::Matrix<double> j_marg;
      Math2D::Matrix<double> i_marg;

      compute_approximate_jmarginals(cur_alignment, expansion_move_prob, swap_move_prob, sentence_prob, j_marg);
      compute_approximate_imarginals(cur_alignment, fertility, expansion_move_prob, sentence_prob, i_marg);
      
      //std::cerr << "updating counts" << std::endl;

      //update zero counts
      for (uint c = 0; c <= curJ / 2; c++) {

        fzero_count += 2 * c * i_marg(c, 0);
        fnonzero_count += ((curJ - 2 * c) + curJ - c) * i_marg(c, 0);
      }
      for (uint j = 0; j < curJ; j++) {
        fzero_count += j_marg(0, j);
        for (uint i=1; i <= curI; i++)
          fnonzero_count += j_marg(i, j);
      }

      //update fertility counts
      for (uint i = 1; i <= curI; i++) {

        const uint t_idx = cur_target[i - 1];

        Math1D::Vector<double>& cur_fert_count = ffert_count[t_idx];

        for (uint c = 0; c <= std::min<ushort>(curJ, fertility_limit_[t_idx]); c++)
          cur_fert_count[c] += i_marg(c, i);
      }

      //update dict counts
      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        for (uint i = 0; i <= curI; i++) {

          const double marg = j_marg(i, j);

          if (i == 0)
            fwcount[0][s_idx - 1] += marg;
          else
            fwcount[cur_target[i - 1]][cur_lookup(j, i - 1)] += marg;
        }
      }

      //update alignment counts

      Math3D::Tensor<double>& cur_align_count = falign_count[curI - 1];

      const double main_prob = best_prob * inv_sentence_prob;
      uint prev_aj = MAX_UINT;
      for (uint j = 0; j < curJ; j++) {
        const uint aj = cur_alignment[j];

        if (aj != 0) {
          if (prev_aj != MAX_UINT)
            cur_align_count(aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]]) += main_prob;
          prev_aj = aj;
        }
      }

      Math1D::Vector<AlignBaseType> work_alignment = cur_alignment;

      for (uint j = 0; j < curJ; j++) {
        for (uint i = 0; i <= curI; i++) {

          const double prob = expansion_move_prob(j, i) * inv_sentence_prob;
          if (prob > 0.0) {

            work_alignment[j] = i;
            uint prev_aj = MAX_UINT;
            for (uint j = 0; j < curJ; j++) {
              const uint aj = work_alignment[j];

              if (aj != 0) {
                if (prev_aj != MAX_UINT)
                  cur_align_count(aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]]) += prob;
                 prev_aj = aj;
              }
            }
          }
        }
        work_alignment[j] = cur_alignment[j];
      }

      for (uint j1 = 0; j1 < curJ - 1; j1++) {
        for (uint j2 = j1 + 1; j2 < curJ; j2++) {

          const double prob = swap_move_prob(j1, j2) * inv_sentence_prob;
          if (prob > 0.0) {

            std::swap(work_alignment[j1], work_alignment[j2]);
            uint prev_aj = MAX_UINT;
            for (uint j = 0; j < curJ; j++) {
              const uint aj = work_alignment[j];

              if (aj != 0) {
                if (prev_aj != MAX_UINT)
                  cur_align_count(aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]]) += prob;

                prev_aj = aj;
              }
            }

            std::swap(work_alignment[j1], work_alignment[j2]);
          }
        }
      }
    }

    // print-outs

    const double reg_term = regularity_term();

    max_perplexity /= source_sentence_.size();
    approx_sum_perplexity /= source_sentence_.size();

    max_perplexity += reg_term;
    approx_sum_perplexity += reg_term;

    std::string transfer = (prev_model != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "Fertility-HMM max-perplex-energy in between iterations #" <<  (iter - 1)
              << " and " << iter << transfer << ": " << max_perplexity << std::endl;
    std::cerr << "Fertility-HMM approx-sum-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << approx_sum_perplexity << std::endl;

    std::cerr << (((double)sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" << std::endl;

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
              << (((double) nZeroAlignments) / ((double) nAlignments)) << std::endl;
    //END_DEBUG

    /**** model update ****/

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, nSentences, dict_weight_sum, smoothed_l0_, l0_beta_, dict_m_step_iter_, dict_, fert_min_dict_entry,
                            msolve_mode_ != MSSolvePGD, gd_stepsize_);

    if (true) {

      //update p_zero_ and p_nonzero_
      if (!fix_p0_ /*&& mstep_mode_ == FertMStepCountsOnly */ ) {
        double fsum = fzero_count + fnonzero_count;
        p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
        p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);

        std::cerr << "new p_zero: " << p_zero_ << std::endl;
      }
      //update fertilities
      update_fertility_prob(ffert_count, 1e-8);

      //update alignment model
      HmmAlignProbType align_type = options_.align_type_;

      if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {
        //std::cerr << "dist_params: " << dist_params << std::endl;

        //call m-step
        ehmm_m_step(falign_count, dist_params_, maxI_ - 1, options_.align_m_step_iter_, dist_grouping_param_,
                    options_.deficient_, options_.redpar_limit_);
        par2nonpar();
      }
      else {

        for (uint I = 1; I <= maxI_; I++) {

          if (align_model_[I - 1].xDim() != 0) {

            for (uint c = 0; c < align_model_[I - 1].zDim(); c++) {
              for (uint i = 0; i < I; i++) {

                const double sum = falign_count[I - 1].sum_x(i, c);

                if (sum >= 1e-300) {

                  const double inv_sum = 1.0 / sum;
                  assert(!isnan(inv_sum));

                  for (uint i_next = 0; i_next < I; i_next++) {
                    align_model_[I - 1](i_next, i, c) = std::max(fert_min_param_entry, inv_sum * falign_count[I - 1](i_next, i, c));
                    assert(!isnan(align_model_[I - 1](i_next, i, c)));
                  }
                }
              }
            }
          }
        }
      }
    }

    printEval(iter, transfer, "EM");
  }

  iter_offs_ = iter - 1;
}

//unconstrained Viterbi training
void FertilityHMMTrainer::train_viterbi(uint nIter, FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper)
{
  const size_t nSentences = source_sentence_.size();

  SingleLookupTable aux_lookup;

  const uint nTargetWords = dict_.size();

  HmmAlignProbType align_type = options_.align_type_;

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

  FullHMMAlignmentModelSingleClass falign_count = align_model_;

  double fzero_count;
  double fnonzero_count;

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    std::cerr << "******* Fertility-HMM Viterbi-iteration #" << iter << std::endl;

    if (passed_wrapper != 0
        && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    if (fertprob_sharing_) {
      for (uint c = 0; c < ffertclass_count.size(); c++)
        ffertclass_count[c].set_constant(0.0);
    }

    for (uint I = 0; I < falign_count.size(); I++)
      falign_count[I].set_constant(0.0);

    double max_perplexity = 0.0;

    uint nNotConverged = 0;

    for (size_t s = 0; s < nSentences; s++) {

      if ((s % 10000) == 0)
        //if ((s% 100) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      long double best_prob;

      if (prev_model != 0) {

        best_prob = prev_model->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter,
                    fertility, expansion_move_prob, swap_move_prob, cur_alignment);
      }
      else {

        best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, cur_alignment);
      }

      max_perplexity -= logl(best_prob);

      //update counts

      fzero_count += 2*fertility[0];
      fnonzero_count += (curJ - 2) * fertility[0] + (curJ - fertility[0]);

      //increase counts for dictionary
      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = cur_alignment[j];

        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj - 1]][cur_lookup(j, cur_aj - 1)] += 1.0;
        }
        else {
          fwcount[0][s_idx - 1] += 1.0;
        }
      }

      //update fertility counts
      for (uint i = 1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i - 1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }

      //update distortion counts
      Math3D::Tensor<double>& cur_align_count = falign_count[curI - 1];

      uint prev_aj = MAX_UINT;
      for (uint j = 0; j < curJ; j++) {
        const uint cur_aj = cur_alignment[j];
        if (cur_aj != 0) {
          if (prev_aj != MAX_UINT) {
            cur_align_count(cur_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]]) += 1.0;
          }
          prev_aj = cur_aj;
        }
      }
    }  //loop over sentences finished

    max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
    max_perplexity /= source_sentence_.size();

    std::string transfer = (prev_model != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "Fertility-HMM energy after in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << max_perplexity << std::endl;

    //update parameters

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, nSentences, 0.0, smoothed_l0_, l0_beta_, 0, dict_, fert_min_dict_entry, false, gd_stepsize_);

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
      p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);

      std::cerr << "new p_zero: " << p_zero_ << std::endl;
    }

    //update fertilities
    update_fertility_prob(ffert_count, 0.0, false);

    //update alignment model
    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {
      //std::cerr << "dist_params: " << dist_params << std::endl;

      //call m-step
      ehmm_m_step(falign_count, dist_params_, maxI_ - 1, options_.align_m_step_iter_, dist_grouping_param_,
                  options_.deficient_, options_.redpar_limit_);
      par2nonpar();
    }
    else {

      for (uint I = 1; I <= maxI_; I++) {

        if (align_model_[I - 1].xDim() != 0) {

          for (uint c = 0; c < align_model_[I - 1].zDim(); c++) {
            for (uint i = 0; i < I; i++) {

              const double sum = falign_count[I - 1].sum_x(i, c);

              if (sum >= 1e-300) {

                const double inv_sum = 1.0 / sum;
                assert(!isnan(inv_sum));

                for (uint i_next = 0; i_next < I; i_next++) {
                  align_model_[I - 1](i_next, i, c) = std::max(fert_min_param_entry, inv_sum * falign_count[I - 1](i_next, i, c));
                  assert(!isnan(align_model_[I - 1](i_next, i, c)));
                }
              }
            }
          }
        }
      }
    }

    /*** ICM stage ***/

    if (prev_model == 0) {
      //no use doing ICM in a transfer iteration.
      //in nondeficient mode, ICM does well at decreasing the energy, but it heavily aligns to the rare words

      if (fertprob_sharing_) {

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
          //if ((s% 100) == 0)
          std::cerr << "ICM, sentence pair #" << s << std::endl;

        const Storage1D<uint>& cur_source = source_sentence_[s];
        const Storage1D<uint>& cur_target = target_sentence_[s];
        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

        Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

        const uint curI = cur_target.size();
        const uint curJ = cur_source.size();

        const Math3D::Tensor<double>& cur_align_prob = align_model_[curI - 1];
        Math3D::Tensor<double>& cur_align_count = falign_count[curI - 1];

        Math1D::NamedVector<uint> cur_fertilities(curI + 1, 0, MAKENAME(fertility));

        for (uint j = 0; j < curJ; j++)
          cur_fertilities[cur_alignment[j]]++;

        uint prev_aj = MAX_UINT;

        for (uint j = 0; j < curJ; j++) {

          uint next_aj = MAX_UINT;
          uint jj = j + 1;
          for (; jj < curJ; jj++) {

            if (cur_alignment[jj] != 0) {
              next_aj = cur_alignment[jj];
              break;
            }
          }

          const uint cur_aj = cur_alignment[j];
          const uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj - 1];
          const uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);
          Math1D::Vector<double>& cur_dictcount = fwcount[cur_word];
          Math1D::Vector<double>& cur_fert_count = ffert_count[cur_word];

          double best_change = 0.0;
          uint new_aj = cur_aj;

          for (uint i = 0; i <= curI; i++) {

            //const uint cur_aj = cur_alignment[j];
            //const uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];

            /**** dict ***/
            //std::cerr << "i: " << i << ", cur_aj: " << cur_aj << std::endl;

            bool allowed = (cur_aj != i && (i != 0 || 2 * cur_fertilities[0] + 2 <= curJ));

            if (i != 0 && (cur_fertilities[i] + 1) > fertility_limit_[cur_target[i-1]])
              allowed = false;

            if (allowed) {

              const uint new_target_word = (i == 0) ? 0 : cur_target[i - 1];
              const Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
              const uint hyp_idx = (i == 0) ? cur_source[j] - 1 : cur_lookup(j, i - 1);
              const Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

              double change = common_icm_change(cur_fertilities, log_pzero, log_pnonzero, dict_sum, cur_dictcount, hyp_dictcount,
                                                prior_weight_[cur_word], prior_weight_[new_target_word], cur_fert_count, hyp_fert_count,
                                                ffertclass_count, cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, i, curJ);

              if (cur_aj != 0) {
                if (prev_aj != MAX_UINT)
                  change -=  -std::log(cur_align_prob(cur_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])) - log_pnonzero;
                if (next_aj != MAX_UINT)
                  change -=  -std::log(cur_align_prob(next_aj - 1, cur_aj - 1, target_class_[cur_target[cur_aj - 1]])) - log_pnonzero;
              }
              else {
                change -= -log_pzero;
                if (prev_aj != MAX_UINT && next_aj != MAX_UINT)
                  change -=  -std::log(cur_align_prob(next_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])) - log_pnonzero;
              }

              if (i != 0) {
                if (prev_aj != MAX_UINT)
                  change +=  -std::log(cur_align_prob(i - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])) - log_pnonzero;
                if (next_aj != MAX_UINT)
                  change +=  -std::log(cur_align_prob(next_aj - 1, i - 1, target_class_[cur_target[i - 1]])) - log_pnonzero;
              }
              else {
                change += -log_pzero;
                if (prev_aj != MAX_UINT && next_aj != MAX_UINT)
                  change +=  -std::log(cur_align_prob(next_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])) - log_pnonzero;
              }

              if (change < best_change) {
                best_change = change;
                new_aj = i;
              }
            }
          }

          if (best_change < -0.01) {

            cur_alignment[j] = new_aj;
            nSwitches++;

            const uint new_target_word = (new_aj == 0) ? 0 : cur_target[new_aj - 1];
            Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
            const uint hyp_idx = (new_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, new_aj - 1);
            Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

            common_icm_count_change(dict_sum, cur_dictcount, hyp_dictcount, cur_fert_count, hyp_fert_count,
                                    ffertclass_count, cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, new_aj, cur_fertilities);

            if (cur_aj == 0) {
              fnonzero_count += 2.0 + 1.0;
              fzero_count -= 2.0;
            }

            if (new_aj == 0) {
              fnonzero_count -= 2.0 + 1.0;
              fzero_count += 2.0;
            }

            if (cur_aj != 0) {
              if (prev_aj != MAX_UINT)
                cur_align_count(cur_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])--;
              if (next_aj != MAX_UINT)
                cur_align_count(next_aj - 1, cur_aj - 1, target_class_[cur_target[cur_aj - 1]])--;
            }
            else {
              if (prev_aj != MAX_UINT && next_aj != MAX_UINT)
                cur_align_count(next_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])--;
            }

            if (new_aj != 0) {
              if (prev_aj != MAX_UINT)
                cur_align_count(new_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])++;
              if (next_aj != MAX_UINT)
                cur_align_count(next_aj - 1, new_aj - 1, target_class_[cur_target[new_aj - 1]])++;
            }
            else {
              if (prev_aj != MAX_UINT && next_aj != MAX_UINT)
                cur_align_count(next_aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]])++;
            }
          }

          if (cur_alignment[j] != 0)
            prev_aj = cur_alignment[j];
        }
      }

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
      std::cerr << "percentage of zero-aligned words: "
                << (((double)nZeroAlignments) / ((double)nAlignments)) << std::endl;
#endif
      //END_DEBUG

      //update probs

      //update p_zero_ and p_nonzero_
      if (!fix_p0_ ) {
        double fsum = fzero_count + fnonzero_count;
        p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
        p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);

        std::cerr << "new p_zero: " << p_zero_ << std::endl;
      }

      //update dictionary
      update_dict_from_counts(fwcount, prior_weight_, nSentences, 0.0, smoothed_l0_, l0_beta_, 0, dict_, fert_min_dict_entry, false, gd_stepsize_);

      //update fertilities
      update_fertility_prob(ffert_count, 0.0, false);

      //update alignment model
      if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {
        //std::cerr << "dist_params: " << dist_params << std::endl;

        //call m-step
        ehmm_m_step(falign_count, dist_params_, maxI_ - 1, options_.align_m_step_iter_, dist_grouping_param_,
                    options_.deficient_, options_.redpar_limit_);
        par2nonpar();
      }
      else {

        for (uint I = 1; I <= maxI_; I++) {

          if (align_model_[I - 1].xDim() != 0) {

            for (uint c = 0; c < align_model_[I - 1].zDim(); c++) {
              for (uint i = 0; i < I; i++) {

                const double sum = falign_count[I - 1].sum_x(i, c);

                if (sum >= 1e-300) {

                  assert(!isnan(sum));
                  const double inv_sum = 1.0 / sum;
                  assert(!isnan(inv_sum));

                  for (uint i_next = 0; i_next < I; i_next++) {
                    align_model_[I - 1](i_next, i, c) = std::max(fert_min_param_entry, inv_sum * falign_count[I - 1](i_next, i, c));
                    assert(!isnan(align_model_[I - 1](i_next, i, c)));
                  }
                }
              }
            }
          }
        }
      }

      max_perplexity = 0.0;
      for (size_t s = 0; s < source_sentence_.size(); s++) {
        max_perplexity -= logl(FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]));
      }

      max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
      max_perplexity /= source_sentence_.size();

      std::cerr << "Fertility-HMM energy after iteration #" << iter << transfer << ": " << max_perplexity << std::endl;
    }

    if (possible_ref_alignments_.size() > 0) {

      std::cerr << "#### Fertility-HMM-AER after iteration #" << iter << transfer << ": " << AER() << std::endl;
      std::cerr << "#### Fertility-HMM-fmeasure after iteration #" << iter << transfer << ": " << f_measure() << std::endl;
      std::cerr << "#### Fertility-HMM-DAE/S after iteration #" << iter << transfer << ": " << DAE_S() << std::endl;
    }
  }

  iter_offs_ = iter - 1;
}

/*virtual*/
long double FertilityHMMTrainer::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
    Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double>& swap_prob,
    Math1D::Vector<AlignBaseType>& alignment) const
{
  //std::cerr << "******update_alignment_by_hillclimbing" << std::endl;

  double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size();

  Math1D::Vector<uint> tclass(curI);

  for (uint i = 0; i < curI; i++)
    tclass[i] = target_class_[target[i]];

  /**** calculate probability of so far best known alignment *****/
  long double base_prob = alignment_prob(source, target, lookup, alignment);
  assert(base_prob > 0.0);

  Math2D::Matrix<double> dict(curJ,curI+1);
  compute_dictmat_fertform(source, lookup, target, dict_, dict);
  //include the p_zero terms of the alignment prob
  for (uint j=0; j < curJ; j++)
    dict(j,0) *= p_zero_;
  for (uint i=1; i <= curI; i++)
    for (uint j=0; j < curJ; j++)
      dict(j,i) *= p_nonzero_;

  fertility.set_constant(0);

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  }

  uint zero_fert = fertility[0];
  if (curJ < 2 * zero_fert) {
    std::cerr << "WARNING: alignment startpoint for HC violates the assumption that less words "
              << " are aligned to NULL than to a real word" << std::endl;

    return 0.0;
  }

  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);

  uint count_iter = 0;

  bool have_warned_for_empty_word = false;

  const Math3D::Tensor<double>& big_align_model = align_model_[curI - 1];
  Math2D::Matrix<double> align_model(big_align_model.xDim(), big_align_model.yDim());
  for (uint i = 0; i < align_model.xDim(); i++)
    for (uint i_prev = 0; i_prev < align_model.yDim(); i_prev++)
      align_model(i, i_prev) = big_align_model(i, i_prev, tclass[i_prev]);

  Math1D::Vector<long double> fert_increase_factor(curI + 1, 0.0);
  Math1D::Vector<long double> fert_decrease_factor(curI + 1, 0.0);

  fert_decrease_factor[0] = 0.0;
  fert_increase_factor[0] = 0.0;

  while (true) {

    count_iter++;
    nIter++;

    //std::cerr << "****************** starting new hillclimb iteration, current best prob: " << base_prob << std::endl;
    //std::cerr << "alignment: " << alignment << std::endl;

    long double best_prob = base_prob;
    bool best_change_is_move = false;
    uint best_move_j = MAX_UINT;
    uint best_move_aj = MAX_UINT;
    uint best_swap_j1 = MAX_UINT;
    uint best_swap_j2 = MAX_UINT;

    Math1D::Vector<int> prev_j(curJ, -1);
    Math1D::Vector<uint> prev_aj(curJ, MAX_UINT);
    Math1D::Vector<uint> next_aj(curJ, MAX_UINT);

    uint paj = MAX_UINT;
    uint pj = 0;
    for (uint j = 0; j < curJ; j++) {
      const uint cur_aj = alignment[j];
      if (paj != MAX_UINT) {
        prev_aj[j] = paj;
        prev_j[j] = pj;
      }
      if (cur_aj != 0) {
        for (uint jj = pj; jj < j; jj++)
          next_aj[jj] = cur_aj - 1;
        paj = cur_aj - 1;
        pj = j;
      }
    }

    /**** scan neighboring alignments and keep track of the best one that is better
     ****  than the current alignment  ****/

    for (uint i = 1; i <= curI; i++) {

      uint t_idx = target[i - 1];
      uint cur_fert = fertility[i];

      assert(fertility_prob_[t_idx][cur_fert] > 0.0);

      if (cur_fert > 0) {
        fert_decrease_factor[i] = ((long double) fertility_prob_[t_idx][cur_fert - 1]) / fertility_prob_[t_idx][cur_fert];

        if (!no_factorial_)
          fert_decrease_factor[i] /= cur_fert;
      }
      else
        fert_decrease_factor[i] = 0.0;

      if (cur_fert + 1 < fertility_prob_[t_idx].size() && cur_fert + 1 <= fertility_limit_[t_idx]) {
        fert_increase_factor[i] = ((long double) fertility_prob_[t_idx][cur_fert + 1]) / fertility_prob_[t_idx][cur_fert];

        if (!no_factorial_)
          fert_increase_factor[i] *= cur_fert + 1;
      }
      else
        fert_increase_factor[i] = 0.0;
    }

    //a) expansion moves

    if (fert_increase_factor[0] == 0.0 && p_zero_ > 0.0) {
      if (curJ >= 2 * (zero_fert + 1)) {

        fert_increase_factor[0] = (curJ - 2 * zero_fert) * (curJ - 2 * zero_fert - 1) * p_zero_
                                  / ((curJ - zero_fert) * (zero_fert + 1) * p_nonzero_ * p_nonzero_);

#ifndef NDEBUG
        long double old_const = ldchoose(curJ - zero_fert - 1,  zero_fert + 1) * p_zero_
                                / (ldchoose(curJ - zero_fert, zero_fert) * p_nonzero_ * p_nonzero_);

        long double ratio = fert_increase_factor[0] / old_const;

        assert(ratio >= 0.975 && ratio <= 1.05);
#endif

        if (empty_word_model_ == FertNullOchNey) {
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
      long double old_const = ldchoose(curJ - zero_fert + 1, zero_fert - 1) * p_nonzero_ * p_nonzero_
                              / (ldchoose(curJ - zero_fert, zero_fert) * p_zero_);

      long double ratio = fert_decrease_factor[0] / old_const;

      assert(ratio >= 0.975 && ratio <= 1.05);
#endif

      if (empty_word_model_ == FertNullOchNey) {
        fert_decrease_factor[0] *= curJ / ((long double)zero_fert);
      }
    }

    for (uint j = 0; j < curJ; j++) {

      const uint aj = alignment[j];
      const uint paj = prev_aj[j];
      const uint naj = next_aj[j];

      //const uint aj_class = (aj != 0) ? tclass[aj - 1] : MAX_UINT;
      //const uint paj_class = (paj != MAX_UINT) ? tclass[paj] : MAX_UINT;

      assert(fertility[aj] > 0);
      expansion_prob(j, aj) = 0.0;

      long double mod_base_prob = base_prob;
      mod_base_prob *= fert_decrease_factor[aj] / dict(j,aj);

      if (aj > 0) {

        if (paj != MAX_UINT)
          mod_base_prob /= align_model(aj - 1, paj);
        if (naj != MAX_UINT)
          mod_base_prob /= align_model(naj, aj - 1);
      }
      else {

        if (paj != MAX_UINT && naj != MAX_UINT)
          mod_base_prob /= align_model(naj, paj);
      }

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        //std::cerr << "examining move " << j << " -> " << cand_aj << " (instead of " << aj << ") in iteration #"
        //        << count_iter << std::endl;
        // std::cerr << "cand_aj has then fertility " << (fertility[cand_aj]+1) << std::endl;
        // std::cerr << "current aj reduces its fertility from " << fertility[aj] << " to " << (fertility[aj]-1)
        //        << std::endl;

        // if (cand_aj != 0 && cand_aj != aj) {
        //   std::cerr << "previous fert prob of candidate: "
        //          << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]] << std::endl;
        //   std::cerr << "new fert prob of candidate: "
        //          << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]+1] << std::endl;
        // }
        // if (aj != 0) {
        //   std::cerr << "previous fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]] << std::endl;
        //   std::cerr << "new fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]-1] << std::endl;
        // }

        if (cand_aj != aj) {

          if (fert_increase_factor[cand_aj] == 0.0) {
            expansion_prob(j, cand_aj) = 0.0;
            continue;
          }

          long double hyp_prob = mod_base_prob;
          hyp_prob *= dict(j,cand_aj) * fert_increase_factor[cand_aj];

          //alignment: entering
          if (cand_aj != 0) {
            if (paj != MAX_UINT)
              hyp_prob *= align_model(cand_aj - 1, paj);
            if (naj != MAX_UINT)
              hyp_prob *= align_model(naj, cand_aj - 1);
          }
          else if (paj != MAX_UINT && naj != MAX_UINT)
            hyp_prob *= align_model(naj, paj);

#ifndef NDEBUG
          Math1D::Vector<ushort> cand_alignment = alignment;
          cand_alignment[j] = cand_aj;

          //std::cerr << "hyp_prob: " << hyp_prob << std::endl;
          //std::cerr << "correct: " << alignment_prob(source,target,lookup,cand_alignment) << std::endl;

          long double check_prob = alignment_prob(source, target, lookup, cand_alignment);

          double check_ratio = hyp_prob / check_prob;
          assert(check_prob == 0.0 || (check_ratio >= 0.99 && check_ratio <= 1.01));
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

    //b) swap_moves (NOTE that swaps do not affect the fertilities)
    for (uint j1 = 0; j1 < curJ; j1++) {

      //std::cerr << "j1: " << j1 << std::endl;

      const uint aj1 = alignment[j1];

      const uint t_idx1 = (aj1 > 0) ? target[aj1 - 1] : MAX_UINT;

      const uint prev_aj1 = prev_aj[j1];
      const uint next_aj1 = next_aj[j1];

      //const uint paj1_class = (prev_aj1 != MAX_UINT) ? tclass[prev_aj1] : 0;
      //const uint aj1_class = (aj1 != 0) ? tclass[aj1 - 1] : MAX_UINT;

      const long double mod_base_prob = base_prob / dict(j1,aj1);

      double align_outgoing_prob = 1.0;
      if (prev_aj1 != MAX_UINT) {
        if (aj1 == 0) {
          if (next_aj1 != MAX_UINT)
            align_outgoing_prob = align_model(next_aj1, prev_aj1);
        }
        else
          align_outgoing_prob = align_model(aj1 - 1, prev_aj1);
      }

      for (uint j2 = j1 + 1; j2 < curJ; j2++) {

        //std::cerr << "j2: " << j2 << std::endl;

        const uint aj2 = alignment[j2];
        const uint s_j2 = source[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1, j2) = 0.0;
        }
        else {

          const uint prev_aj2 = prev_aj[j2];
          const uint next_aj2 = next_aj[j2];

          //const uint paj2_class = (prev_aj2 != MAX_UINT) ? tclass[prev_aj2] : MAX_UINT;
          //const uint aj2_class = (aj2 != 0) ? tclass[aj2 - 1] : MAX_UINT;

          long double hyp_prob = mod_base_prob * dict(j2,aj1) * dict(j1,aj2) / dict(j2,aj2);

          if (aj1 == 0) {

            if (prev_j[j2] > (int)j1) {

              //DEBUG
              // if (prev_aj2 == MAX_UINT) {

              //        std::cerr << "swap " << j1 << "->" << aj1 << " <-> " << j2 << "->" << aj2 << std::endl;

              //        std::cerr << "alignment:      " << alignment << std::endl;
              //        std::cerr << "prev_aj1: " << prev_aj1 << ", next_aj1: " << next_aj1 << std::endl;
              //        std::cerr << "prev_aj2: " << prev_aj2 << ", next_aj2: " << next_aj2 << std::endl;

              //        std::cerr << "prev_j[j2]: " << prev_j[j2] << std::endl;
              // }
              //END_DEBUG

              assert(next_aj1 != MAX_UINT);
              assert(prev_aj2 != MAX_UINT);

              hyp_prob *= align_model(next_aj1, aj2 - 1);

              if (prev_aj1 != MAX_UINT)
                hyp_prob *= align_model(aj2 - 1, prev_aj1) / align_outgoing_prob;      //align_model(next_aj1,prev_aj1,paj1_class);

              hyp_prob /= align_model(aj2 - 1, prev_aj2);

              if (next_aj2 != MAX_UINT)
                hyp_prob *= align_model(next_aj2, prev_aj2) / align_model(next_aj2, aj2 - 1);
            }
          }
          else if (aj2 == 0) {

            if (prev_j[j2] > (int)j1) {

              assert(prev_aj2 != MAX_UINT);
              assert(next_aj1 != MAX_UINT);

              hyp_prob /= align_model(next_aj1, aj1 - 1);

              if (prev_aj1 != MAX_UINT)
                hyp_prob *= align_model(next_aj1, prev_aj1) / align_outgoing_prob;      //align_model(aj1-1,prev_aj1,paj1_class);

              hyp_prob *= align_model(aj1 - 1, prev_aj2);

              if (next_aj2 != MAX_UINT)
                hyp_prob *= align_model(next_aj2, aj1 - 1) / align_model(next_aj2, prev_aj2);
            }
          }
          else {

            if (prev_aj1 != MAX_UINT)
              hyp_prob *= align_model(aj2 - 1, prev_aj1) / align_outgoing_prob;        //align_model(aj1-1,prev_aj1,paj1_class);

            if (prev_j[j2] != j1) {
              hyp_prob *= align_model(next_aj1, aj2 - 1) / align_model(next_aj1, aj1 - 1);
              hyp_prob *= align_model(aj1 - 1, prev_aj2) / align_model(aj2 - 1, prev_aj2);
            }
            else {
              hyp_prob *= align_model(aj1 - 1, aj2 - 1) / align_model(aj2 - 1, aj1 - 1);
            }

            if (next_aj2 != MAX_UINT)
              hyp_prob *= align_model(next_aj2, aj1 - 1) / align_model(next_aj2, aj2 - 1);
          }

#ifndef NDEBUG
          Math1D::Vector<ushort> cand_alignment = alignment;

          cand_alignment[j1] = aj2;
          cand_alignment[j2] = aj1;

          long double check_prob = alignment_prob(source, target, lookup, cand_alignment);
          double check_ratio = hyp_prob / check_prob;

          if (check_prob > 0.0 && !(check_ratio >= 0.99 && check_ratio <= 1.01)) {
            std::cerr << "swap " << j1 << "->" << aj1 << " <-> " << j2 << "->" << aj2 << std::endl;

            std::cerr << "alignment:      " << alignment << std::endl;
            std::cerr << "cand_alignment: " << cand_alignment << std::endl;

            std::cerr << "check_ratio: " << check_ratio << std::endl;
            std::cerr << "prev_aj1: " << prev_aj1 << ", next_aj1: " << next_aj1 << std::endl;
          }

          assert(check_prob == 0.0 || check_ratio >= 0.99 && check_ratio <= 1.01);
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

    if (best_prob < improvement_factor * base_prob || count_iter > nMaxHCIter_) {
      if (count_iter > nMaxHCIter_)
        std::cerr << "HC Iteration limit reached" << std::endl;
      break;
    }
    //update alignment
    if (best_change_is_move) {
      uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      //std::cerr << "moving source pos " << best_move_j << " from " << cur_aj << " to " << best_move_aj << std::endl;

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;

      if (cur_aj * best_move_aj == 0) {
        //signal recomputation
        zero_fert = fertility[0];
        fert_increase_factor[0] = 0.0;
        fert_decrease_factor[0] = 0.0;
      }
    }
    else {
      //std::cerr << "swapping: j1=" << best_swap_j1 << std::endl;
      //std::cerr << "swapping: j2=" << best_swap_j2 << std::endl;

      std::swap(alignment[best_swap_j1], alignment[best_swap_j2]);
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

  assert(!isnan(base_prob));

  assert(2 * fertility[0] <= curJ);

  //symmetrize swap_prob
  symmetrize_swapmat(swap_prob, curJ);

  //std::cerr << "******leaving update_alignment_by_hillclimbing" << std::endl;
  return base_prob;
}

/*virtual*/ long double FertilityHMMTrainer::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment,
    AlignmentSetConstraints* constraints)
{
  prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();
  const uint I = target.size();

  uint nIter = 0;
  Math2D::Matrix<long double> expansion_prob(J, I + 1);
  Math2D::Matrix<long double> swap_prob(J, J);
  Math1D::Vector<uint> fertility(I + 1, 0);
  long double hc_prob = update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility, expansion_prob, swap_prob, alignment);

  return hc_prob;
}

long double FertilityHMMTrainer::hmm_prob(const Storage1D<uint>& target, const Math1D::Vector<AlignBaseType>& alignment) const
{
  const uint I = target.size();

  const uint curJ = alignment.size();
  const Math3D::Tensor<double>& cur_align_prob = align_model_[I - 1];

  long double prob = 1.0;

  uint prev_aj = MAX_UINT;
  for (uint j = 0; j < curJ; j++) {

    const uint aj = alignment[j];
    if (aj != 0) {

      if (prev_aj != MAX_UINT)
        prob *= cur_align_prob(aj - 1, prev_aj - 1, target_class_[target[prev_aj - 1]]);

      prev_aj = aj;
    }
    if (aj == 0)
      prob *= p_zero_;
    else
      prob *= p_nonzero_;
  }

  return prob;
}

/*virtual*/ long double FertilityHMMTrainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Math3D::Tensor<double>& cur_align_prob = align_model_[curI - 1];

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    //we find that we need to include p_zero in the alignment prob as well. Otherwise p_zero grows large in training
    if (aj == 0)
      prob *= p_zero_;
    else
      prob *= p_nonzero_;
  }

  //std::cerr << "ap: fertility: " << fertility << std::endl;

  const uint zero_fert = fertility[0];
  if (curJ < 2 * zero_fert)
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

  uint prev_aj = MAX_UINT;
  for (uint j = 0; j < curJ; j++) {

    uint s_idx = source[j];
    uint aj = alignment[j];

    if (aj == 0)
      prob *= dict_[0][s_idx - 1];
    else {
      uint t_idx = target[aj - 1];
      prob *= dict_[t_idx][lookup(j, aj - 1)];

      if (prev_aj != MAX_UINT)
        prob *= cur_align_prob(aj - 1, prev_aj - 1, target_class_[target[prev_aj - 1]]);

      prev_aj = aj;
    }
  }

  //std::cerr << "ap before empty word: " << prob << std::endl;

  //handle empty word
  assert(zero_fert <= 2 * curJ);

  prob *= choose_factor_[curJ][zero_fert];
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  prob *= p_zero_pow_[zero_fert];
  prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  if (empty_word_model_ == FertNullOchNey) {

    prob *= och_ney_factor_[curJ][zero_fert];
  }

  return prob;
}

long double FertilityHMMTrainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment, double p_zero,
    const Math3D::Tensor<double>& cur_align_prob, const Storage1D<Math1D::Vector<double> >& fertility_prob,
    bool with_dict) const
{
  //std::cerr << "******* align_prob for alignment " << alignment << std::endl;
  long double prob = 1.0;
  const double p_nonzero = 1.0 - p_zero;

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);
  assert(alignment.max() <= curI);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    //we find that we need to include p_zero in the alignment prob as well. Otherwise p_zero grows large in training
    if (aj == 0)
      prob *= p_zero;
    else
      prob *= p_nonzero;
  }

  //std::cerr << "ap: fertility: " << fertility << std::endl;

  const uint zero_fert = fertility[0];
  if (curJ < 2 * zero_fert)
    return 0.0;

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = target[i - 1];
    prob *= fertility_prob[t_idx][fertility[i]];
    if (!no_factorial_)
      prob *= ld_fac_[fertility[i]];    //ldfac(fertility[i]);

    //     std::cerr << "fertility_factor(" << i << "): "
    //        << (ldfac(fertility[i]) * fertility_prob_[t_idx][fertility[i]])
    //        << std::endl;
  }

  uint prev_aj = MAX_UINT;
  for (uint j = 0; j < curJ; j++) {

    uint s_idx = source[j];
    uint aj = alignment[j];

    if (aj == 0) {
      if (with_dict)
        prob *= dict_[0][s_idx - 1];
    }
    else {
      uint t_idx = target[aj - 1];
      if (with_dict)
        prob *= dict_[t_idx][lookup(j, aj - 1)];

      if (prev_aj != MAX_UINT)
        prob *= cur_align_prob(aj - 1, prev_aj - 1, target_class_[target[prev_aj - 1]]);

      prev_aj = aj;
    }
  }

  //std::cerr << "ap before empty word: " << prob << std::endl;

  //handle empty word
  assert(zero_fert <= 2 * curJ);

  prob *= choose_factor_[curJ][zero_fert];
  update_nullpow(zero_fert, curJ - 2 * zero_fert, p_zero, p_nonzero);
  prob *= p_zero_pow_[zero_fert];
  prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  if (empty_word_model_ == FertNullOchNey) {

    prob *= och_ney_factor_[curJ][zero_fert];
  }

  return prob;
}

/*virtual*/ void FertilityHMMTrainer::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  common_prepare_external_alignment(source, target, lookup, alignment);

  const uint I = target.size();
  const uint nClasses = dist_params_.yDim();

  if (dist_params_.xDim() < 2 * I - 1) {
    uint prev_maxI = dist_params_.size() / 2 + 1;
    const uint old_zero_offs = prev_maxI-1;
    const uint new_zero_offs = I - 1;
    Math2D::Matrix<double> new_params(2 * I - 1, nClasses, 1e-8);
    for (uint c = 0; c < nClasses; c++)
      for (int i = -prev_maxI - 1; i < prev_maxI; i++)
        new_params(new_zero_offs + i, c) = dist_params_(old_zero_offs + i, c);

    dist_params_ = new_params;
  }
  if (align_model_.size() < I) {
    align_model_.resize(I);
  }

  if (align_model_[I - 1].size() == 0) {
    align_model_[I - 1].resize(I, I, nClasses, 1.0 / I);
    int zero_offset = dist_params_.size() / 2;
    if (options_.align_type_ == HmmAlignProbNonpar || options_.align_type_ == HmmAlignProbNonpar2) {
      //compute params
      dist_params_.set_constant(0.0);
      for (uint I = 1; I <= align_model_.size(); I++) {
        for (uint c = 0; c < align_model_[I-1].zDim(); c++) {
          for (uint i = 0; i < align_model_[I-1].xDim(); i++)
            for (uint i_given = 0; i_given < align_model_[I-1].yDim(); i_given++)
              dist_params_(zero_offset + i - i_given,c) += align_model_[I-1](i,i_given,c);
        }
      }
      for (uint c = 0; c < align_model_[I-1].zDim(); c++) {
        for (uint i_given = 0; i_given < align_model_[I-1].yDim(); i_given++) {
          double sum = 0.0;
          for (uint i = 0; i < align_model_[I-1].xDim(); i++) {
            align_model_[I-1](i,i_given,c) = dist_params_(zero_offset + i - i_given,c);
            sum += align_model_[I-1](i,i_given,c);
          }
          if (sum > 0.0) {
            for (uint i = 0; i < align_model_[I-1].xDim(); i++)
              align_model_[I-1](i,i_given,c) /= sum;
          }
        }
      }
    }
    else {
      par2nonpar();
    }
  }
}

void FertilityHMMTrainer::par2nonpar()
{
  const HmmAlignProbType align_type = options_.align_type_;
  const uint zero_offset = dist_params_.xDim() / 2;     //maxI_-1;
  const int rlimit = options_.redpar_limit_;
  bool deficient = options_.deficient_;

  for (uint I = 1; I <= align_model_.size(); I++) {

    //std::cerr << "I: " << I << std::endl;

    if (align_model_[I - 1].size() > 0) {

      for (uint c = 0; c < align_model_[I - 1].zDim(); c++) {

        //std::cerr << "c: " << c << std::endl;

        for (int i = 0; i < (int)I; i++) {

          double grouping_norm = std::max(0, i - rlimit);
          grouping_norm += std::max(0, int (I) - 1 - (i + rlimit));

          double non_zero_sum = 0.0;
          for (int ii = 0; ii < (int)I; ii++) {
            if (align_type != HmmAlignProbReducedpar || abs(ii - i) <= rlimit)
              non_zero_sum += dist_params_(zero_offset + ii - i, c);
          }

          if (align_type == HmmAlignProbReducedpar && grouping_norm > 0.0) {
            non_zero_sum += dist_grouping_param_[c];
          }

          assert(non_zero_sum > 1e-305);
          const double inv_sum = (deficient) ? 1.0 : 1.0 / non_zero_sum;

          for (int ii = 0; ii < (int)I; ii++) {
            if (align_type == HmmAlignProbReducedpar && abs(ii - i) > rlimit) {
              assert(!isnan(grouping_norm));
              assert(grouping_norm > 0.0);
              align_model_[I - 1](ii, i, c) = std::max(fert_min_param_entry, inv_sum * dist_grouping_param_[c] / grouping_norm);
            }
            else {
              assert(dist_params_(zero_offset + ii - i, c) >= 0);
              align_model_[I - 1](ii, i, c) = std::max(fert_min_param_entry, inv_sum * dist_params_(zero_offset + ii - i, c));
            }
            assert(!isnan(align_model_[I - 1](ii, i, c)));
            assert(align_model_[I - 1](ii, i, c) >= 0.0);
          }
        }
      }
    }
  }
}

void FertilityHMMTrainer::par2nonpar(const Math2D::Matrix<double> dist_params, const Math1D::Vector<double>& dist_grouping_param,
                                     FullHMMAlignmentModelSingleClass& align_model) const
{
  const HmmAlignProbType align_type = options_.align_type_;
  const int redpar_limit = options_.redpar_limit_;
  const bool deficient = options_.deficient_;

  assert(dist_params.size() == dist_params_.size());
  const uint zero_offset = dist_params_.xDim() / 2;     //maxI_-1;

  align_model.resize(align_model_.size());
  for (uint I = 1; I <= align_model.size(); I++) {

    //std::cerr << "I: " << I << std::endl;

    if (align_model_[I - 1].size() > 0) {

      align_model[I - 1].resize(I, I, align_model_[I - 1].zDim());

      for (uint c = 0; c < align_model[I - 1].zDim(); c++) {

        //std::cerr << "c: " << c << std::endl;

        for (int i = 0; i < (int)I; i++) {

          double grouping_norm = std::max(0, i - redpar_limit);
          grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

          double non_zero_sum = 0.0;
          for (int ii = 0; ii < (int)I; ii++) {
            if (align_type != HmmAlignProbReducedpar || abs(ii - i) <= redpar_limit)
              non_zero_sum += dist_params(zero_offset + ii - i, c);
          }

          if (align_type == HmmAlignProbReducedpar && grouping_norm > 0.0) {
            non_zero_sum += dist_grouping_param[c];
          }

          assert(non_zero_sum > 1e-305);
          const double inv_sum = (deficient) ? 1.0 : 1.0 / non_zero_sum;

          for (int ii = 0; ii < (int)I; ii++) {
            if (align_type == HmmAlignProbReducedpar && abs(ii - i) > redpar_limit) {
              assert(!isnan(grouping_norm));
              assert(grouping_norm > 0.0);
              assert(dist_grouping_param[c] >= fert_min_param_entry);
              align_model[I - 1](ii, i, c) = std::max(fert_min_param_entry, inv_sum * dist_grouping_param[c] / grouping_norm);
            }
            else {
              assert(dist_params(zero_offset + ii - i, c) >= fert_min_param_entry);
              align_model[I - 1](ii, i, c) = std::max(fert_min_param_entry, inv_sum * dist_params(zero_offset + ii - i, c));
            }
            assert(!isnan(align_model[I - 1](ii, i, c)));
            assert(align_model[I - 1](ii, i, c) >= fert_min_param_entry);
          }
        }
      }
    }
  }
}

/*virtual*/ double FertilityHMMTrainer::compute_approximate_marginals(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment,
    Math2D::Matrix<double>& j_marg, Math2D::Matrix<double>& i_marg, double hc_mass,
    bool& converged) const
{
  return FertilityModelTrainer::compute_approximate_marginals(source, target, lookup, alignment, j_marg, i_marg, hc_mass, converged);
}

void FertilityHMMTrainer::compute_dist_param_gradient(const Math2D::Matrix<double>& dist_params, const Math1D::Vector<double>& dist_grouping_param,
    const FullHMMAlignmentModelSingleClass& align_grad, Math2D::Matrix<double>& distort_param_grad,
    Math1D::Vector<double>& dist_grouping_grad, uint zero_offset) const
{
  const HmmAlignProbType align_type = options_.align_type_;

  if (align_type == HmmAlignProbFullpar || align_type == HmmAlignProbReducedpar) {

    distort_param_grad.resize_dirty(dist_params.xDim(), dist_params.yDim());
    dist_grouping_grad.resize_dirty(dist_grouping_param.size());

    distort_param_grad.set_constant(0.0);
    dist_grouping_grad.set_constant(0.0);

    const int redpar_limit = options_.redpar_limit_;

    for (uint I = 1; I <= align_grad.size(); I++) {

      if (align_grad[I - 1].size() > 0) {

        for (int i = 0; i < (int)I; i++) {
          for (int ii = 0; ii < (int)I; ii++) {

            for (uint c = 0; c < distort_param_grad.yDim(); c++) {

              double non_zero_sum = 0.0;

              if (align_type == HmmAlignProbFullpar) {

                if (options_.deficient_) {
                  for (uint ii = 0; ii < I; ii++)
                    distort_param_grad(zero_offset + ii - i, c) += align_grad[I - 1] (ii, i, c);
                }
                else {

                  for (uint ii = 0; ii < I; ii++) {
                    non_zero_sum += dist_params(zero_offset + ii - i, c);
                  }

                  //std::cerr << "I: " << I << ", i: " << i << ", non_zero_sum: " << non_zero_sum << std::endl;

                  const double factor = 1.0 / (non_zero_sum * non_zero_sum);

                  assert(!isnan(factor));

                  for (uint ii = 0; ii < I; ii++) {
                    //NOTE: align_grad has already a negative sign

                    const double cur_grad = align_grad[I - 1] (ii, i, c);

                    distort_param_grad(zero_offset + ii - i, c) += cur_grad * factor * non_zero_sum;
                    for (uint iii = 0; iii < I; iii++)
                      distort_param_grad(zero_offset + iii - i, c) -= cur_grad * factor * dist_params(zero_offset + ii - i,c);
                  }
                }
              }
              else {

                double grouping_norm = std::max(0, i - redpar_limit);
                grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

                if (options_.deficient_) {
                  for (int ii = 0; ii < (int)I; ii++) {
                    //NOTE: align_grad has already a negative sign
                    if (abs(ii - i) <= redpar_limit) {
                      distort_param_grad(zero_offset + ii - i, c) += align_grad[I - 1] (ii, i, c);
                    }
                    else {
                      dist_grouping_grad[c] += align_grad[I - 1] (ii, i, c) / grouping_norm;
                    }
                  }
                }
                else {

                  if (grouping_norm > 0.0)
                    non_zero_sum += dist_grouping_param[c];

                  for (int ii = 0; ii < (int)I; ii++) {

                    if (abs(ii - i) <= redpar_limit)
                      non_zero_sum += dist_params(zero_offset + ii - i, c);
                  }

                  const double factor = 1.0 / (non_zero_sum * non_zero_sum);

                  assert(!isnan(factor));

                  for (int ii = 0; ii < (int)I; ii++) {

                    const double cur_grad = align_grad[I - 1] (ii, i, c);

                    //NOTE: align_grad has already a negative sign
                    if (abs(ii - i) <= redpar_limit) {

                      distort_param_grad(zero_offset + ii - i, c) += cur_grad * factor * non_zero_sum;
                      for (int iii = 0; iii < (int)I; iii++) {
                        if (abs(iii - i) <= redpar_limit)
                          distort_param_grad(zero_offset + iii - i, c) -= cur_grad * factor * dist_params(zero_offset + ii - i, c);
                        else
                          dist_grouping_grad[c] -= cur_grad * factor * dist_params(zero_offset + ii - i, c) / grouping_norm;
                      }
                    }
                    else {

                      dist_grouping_grad[c] += cur_grad * factor * (non_zero_sum - dist_grouping_param[c]) / grouping_norm;

                      for (int iii = 0; iii < (int)I; iii++) {
                        if (abs(iii - i) <= redpar_limit) {
                          assert(iii != ii);
                          distort_param_grad(zero_offset + iii - i, c) -= cur_grad * factor * dist_grouping_param[c] / grouping_norm;
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
    }
  }

}