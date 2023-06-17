/****** written by Thomas Schoenemann, October 2022 ****/

#include "training_common.hh"
#include "fertility_hmmcc.hh"
#include "hmm_training.hh"
#include "stl_util.hh"
#include "conditional_m_steps.hh"

FertilityHMMTrainerDoubleClass::FertilityHMMTrainerDoubleClass(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
    const Storage1D<Math1D::Vector<uint> >& target_sentence,
    const Math1D::Vector<WordClassType>& source_class, const Math1D::Vector<WordClassType>& target_class,
    const RefAlignmentStructure& sure_ref_alignments, const RefAlignmentStructure& possible_ref_alignments,
    SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class,
    const Math1D::Vector<double>& source_fert, uint zero_offset,
    uint nSourceWords, uint nTargetWords, const floatSingleWordDictionary& prior_weight,
    const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
    const HmmOptions& options, const FertModelOptions& fert_options,
    const Storage2D<Math1D::Vector<double> >& dist_params, Math2D::Matrix<double> dist_grouping_param,
    bool no_factorial)
  : FertilityModelTrainer(source_sentence, slookup, target_sentence, dict, wcooc, tfert_class, nSourceWords, nTargetWords, prior_weight,
                          sure_ref_alignments, possible_ref_alignments, log_table, xlogx_table, fert_options, no_factorial),
    options_(options), zero_offset_(zero_offset), dist_params_(dist_params), dist_grouping_param_(dist_grouping_param), source_fert_(source_fert),
    source_class_(source_class), target_class_(target_class)
{}

/*virtual*/
std::string FertilityHMMTrainerDoubleClass::model_name() const
{
  return "FertilityHMM-DoubleClass";
}

/*virtual*/
long double FertilityHMMTrainerDoubleClass::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  Math1D::Vector<uint> sclass(curJ);
  for (uint j = 0; j < curJ; j++)
    sclass[j] = source_class_[source[j]];

  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = target_class_[target[i]];

  //Storage2D<Math2D::Matrix<double> > cur_align_model;
  //par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_, dist_params_, dist_grouping_param_, cur_align_model, options_.align_type_,
  //								   options_.deficient_, options_.redpar_limit_, zero_offset_);


  //no need for a target class dimension, class is determined by i_prev
  Math3D::Tensor<double> cur_align_model;
  par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_, dist_params_, dist_grouping_param_, cur_align_model, options_.align_type_,
                                   options_.deficient_, options_.redpar_limit_, zero_offset_);

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
        prob *= cur_align_model(sclass[j-1], aj - 1, prev_aj - 1);

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

void FertilityHMMTrainerDoubleClass::init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper,
    const Storage2D<Math1D::Vector<double> >& dist_params,
    const Math2D::Matrix<double>& dist_grouping_param,
    bool clear_prev, bool count_collection, bool viterbi)
{
  std::cerr << "******** initializing Fertility-HMM-DoubleClass from " << prev_model->model_name() << " *******" << std::endl;

  dist_grouping_param_ = dist_grouping_param;
  dist_params_ = dist_params;

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

/*virtual*/
long double FertilityHMMTrainerDoubleClass::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, uint& nIter,
    Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob,
    Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment) const
{
  //std::cerr << "******update_alignment_by_hillclimbing" << std::endl;

  double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size();

  Math1D::Vector<uint> sclass(curJ);
  for (uint j = 0; j < curJ; j++)
    sclass[j] = source_class_[source[j]];

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

  //Storage2D<Math2D::Matrix<double> > align_model;
  Math3D::Tensor<double> align_model; //no need for a target class dimension, class is detirmened by i_prev
  par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_, dist_params_, dist_grouping_param_, align_model, options_.align_type_,
                                   options_.deficient_, options_.redpar_limit_, zero_offset_);

  assert(alignment.size() == curJ);


  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);

  uint count_iter = 0;

  bool have_warned_for_empty_word = false;

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
      const uint sc = (j > 0) ? sclass[j-1] : MAX_UINT;

      uint next_j = j+1;
      while (next_j < curJ && alignment[next_j] == 0)
        next_j++;
      const uint scnext = sclass[next_j - 1];

      //std::cerr << "---- expansions for j: " << j << ", alignment: " << alignment << ", aj: " << aj << std::endl;
      //std::cerr << "next_j: " << next_j << ", next_aj: " << naj << std::endl;

      if (next_j < curJ)
        assert(next_aj[j] == alignment[next_j]-1);

      assert(fertility[aj] > 0);
      expansion_prob(j, aj) = 0.0;

      long double mod_base_prob = base_prob;
      mod_base_prob *= fert_decrease_factor[aj] / dict(j,aj);

      if (aj > 0) {

        if (paj != MAX_UINT)
          mod_base_prob /= align_model(sc, aj - 1, paj);
        if (naj != MAX_UINT)
          mod_base_prob /= align_model(scnext, naj, aj - 1);
      }
      else {

        if (paj != MAX_UINT && naj != MAX_UINT)
          mod_base_prob /= align_model(scnext, naj, paj);
      }

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

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
              hyp_prob *= align_model(sc, cand_aj - 1, paj);
            if (naj != MAX_UINT)
              hyp_prob *= align_model(scnext, naj, cand_aj - 1);
          }
          else if (paj != MAX_UINT && naj != MAX_UINT)
            hyp_prob *= align_model(scnext, naj, paj);

//#ifndef NDEBUG
#if 0
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

      //std::cerr << "---- swaps for j1: " << j1 << std::endl;

      const uint aj1 = alignment[j1];
      const uint sc1 = (j1 == 0) ? MAX_UINT : sclass[j1 - 1];

      //std::cerr << "aj1: " << aj1 << std::endl;

      const uint t_idx1 = (aj1 > 0) ? target[aj1 - 1] : MAX_UINT;

      const uint prev_aj1 = prev_aj[j1];
      const uint next_aj1 = next_aj[j1];

      //const uint paj1_class = (prev_aj1 != MAX_UINT) ? tclass[prev_aj1] : MAX_UINT;
      //const uint aj1_class = (aj1 != 0) ? tclass[aj1 - 1] : MAX_UINT;

      uint next_j1 = j1 + 1;
      while (next_j1 < curJ && alignment[next_j1] == 0)
        next_j1++;

      uint sc2 = (next_j1 < curJ) ? sclass[next_j1 - 1] : MAX_UINT;

      const long double mod_base_prob = base_prob / dict(j1,aj1);

      long double align_outgoing_prob = 1.0;
      if (prev_aj1 != MAX_UINT) {
        if (aj1 == 0) {
          if (next_aj1 != MAX_UINT)
            align_outgoing_prob = align_model(sc2, next_aj1, prev_aj1); //j1 was aligned to NULL
        }
        else
          align_outgoing_prob = align_model(sc1, aj1 - 1, prev_aj1); //j1 was aligned to aj1
      }

      for (uint j2 = j1 + 1; j2 < curJ; j2++) {

        //std::cerr << "j2: " << j2 << std::endl;

        const uint aj2 = alignment[j2];
        const uint s_j2 = source[j2];
        const uint sc3 = sclass[j2 - 1];

        //std::cerr << "aj2: " << aj2 << std::endl;

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1, j2) = 0.0;
        }
        else {

          const uint prev_aj2 = prev_aj[j2];
          const uint next_aj2 = next_aj[j2];

          //const uint paj2_class = (prev_aj2 != MAX_UINT) ? tclass[prev_aj2] : MAX_UINT;
          //const uint aj2_class = (aj2 != 0) ? tclass[aj2 - 1] : MAX_UINT;

          uint next_j2 = j2 + 1;
          while (next_j2 < curJ && alignment[next_j2] == 0)
            next_j2++;
          uint sc4 = (next_j2 < curJ) ? sclass[next_j2 - 1] : MAX_UINT;

          long double hyp_prob = mod_base_prob * dict(j2,aj1) * dict(j1,aj2) / dict(j2,aj2);

          if (aj1 == 0) {

            if (prev_j[j2] > (int) j1) {

              //DEBUG
              // if (prev_aj2 == MAX_UINT) {

              // std::cerr << "swap " << j1 << "->" << aj1 << " <-> " << j2 << "->" << aj2 << std::endl;
              // std::cerr << "alignment:      " << alignment << std::endl;
              // std::cerr << "prev_aj1: " << prev_aj1 << ", next_aj1: " << next_aj1 << std::endl;
              // std::cerr << "prev_aj2: " << prev_aj2 << ", next_aj2: " << next_aj2 << std::endl;

              // std::cerr << "prev_j[j2]: " << prev_j[j2] << std::endl;
              // }
              //END_DEBUG

              assert(next_aj1 != MAX_UINT);
              assert(prev_aj2 != MAX_UINT);

              hyp_prob *= align_model(sc2, next_aj1, aj2 - 1); //j1 becomes aligned to aj2

              if (prev_aj1 != MAX_UINT) //j1 becomes aligned to aj2
                hyp_prob *= align_model(sc1, aj2 - 1, prev_aj1) / align_outgoing_prob;      //align_model(next_aj1,prev_aj1,paj1_class);

              hyp_prob /= align_model(sc3, aj2 - 1, prev_aj2); //j2 was aligned to aj2

              if (next_aj2 != MAX_UINT)
                hyp_prob *= align_model(sc4, next_aj2, prev_aj2) / align_model(sc4, next_aj2, aj2 - 1);
            }
            else {

              //j1 becomes aligned to aj2, j2 becomes aligned to NULL

              // std::cerr << "swap " << j1 << "->" << aj1 << " <-> " << j2 << "->" << aj2 << std::endl;
              // std::cerr << "alignment:      " << alignment << std::endl;
              // std::cerr << "prev_j[j1]: " << prev_j[j1] << ", prev_j[j2]: " << prev_j[j2] << std::endl;
              // std::cerr << "prev_aj1: " << prev_aj1 << ", next_aj1: " << next_aj1 << std::endl;
              // std::cerr << "prev_aj2: " << prev_aj2 << ", next_aj2: " << next_aj2 << std::endl;
              // std::cerr << "next_j1: " << next_j1 << ", next_j2: " << next_j2 << std::endl;

              assert(prev_j[j2] == prev_j[j1]);
              assert(next_aj1 == aj2 - 1);
              assert(sc2 == sc3);

              if (prev_aj1 != MAX_UINT) {
                //j2 becomes aligned to NULL
                hyp_prob /= align_model(sc3, aj2 - 1, prev_aj1);

                //j1 decomes aligned to aj2
                if (sc1 != MAX_UINT)
                  hyp_prob *= align_model(sc1, aj2 - 1, prev_aj1);
              }
            }
          }
          else if (aj2 == 0) {

            //std::cerr << "prev_j2: " << prev_j[j2] << ", j1: " << j1 << std::endl;

            if (prev_j[j2] > (int) j1) {

              assert(prev_aj2 != MAX_UINT);
              assert(next_aj1 != MAX_UINT);

              hyp_prob /= align_model(sc2, next_aj1, aj1 - 1); // next_j1 was aligned to next_aj1

              if (prev_aj1 != MAX_UINT) //j1 becomes aligned to NULL
                hyp_prob *= align_model(sc2, next_aj1, prev_aj1) / align_outgoing_prob;      //align_model(aj1-1,prev_aj1,paj1_class);

              hyp_prob *= align_model(sc3, aj1 - 1, prev_aj2); //aj1 becomes aligned to j2

              if (next_aj2 != MAX_UINT)
                hyp_prob *= align_model(sc4, next_aj2, aj1 - 1) / align_model(sc4, next_aj2, prev_aj2);
            }
            else {
              assert(prev_j[j2] == j1);
              //j1 becomes aligned to NULL, j2 becomes aligned to aj1
              if (prev_aj1 != MAX_UINT) {
                hyp_prob /= align_model(sc1, aj1 - 1, prev_aj1);
                hyp_prob *= align_model(sc3, aj1 - 1, prev_aj1);
              }
            }
          }
          else {

            //std::cerr << "both alignments not null" << std::endl;

            if (prev_aj1 != MAX_UINT)
              hyp_prob *= align_model(sc1, aj2 - 1, prev_aj1) / align_outgoing_prob;        //align_model(aj1-1,prev_aj1,paj1_class);

            if (prev_j[j2] != j1) {
              //std::cerr << "direct link" << std::endl;
              hyp_prob *= align_model(sc2, next_aj1, aj2 - 1) / align_model(sc2, next_aj1, aj1 - 1);
              hyp_prob *= align_model(sc3, aj1 - 1, prev_aj2) / align_model(sc3, aj2 - 1, prev_aj2);
            }
            else {
              hyp_prob *= align_model(sc2, aj1 - 1, aj2 - 1) / align_model(sc2, aj2 - 1, aj1 - 1);
            }

            if (next_aj2 != MAX_UINT)
              hyp_prob *= align_model(sc4, next_aj2, aj1 - 1) / align_model(sc4, next_aj2, aj2 - 1);
          }

//#ifndef NDEBUG
#if 0
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

/*virtual*/
long double FertilityHMMTrainerDoubleClass::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
    Math1D::Vector<AlignBaseType>& alignment, AlignmentSetConstraints* constraints)
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

/*virtual*/
void FertilityHMMTrainerDoubleClass::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  common_prepare_external_alignment(source, target, lookup, alignment);

  const uint I = target.size();
  const uint nSourceClasses = dist_params_.xDim();
  const uint nTargetClasses = dist_params_.yDim();

  std::set<uint> source_classes;
  for (uint j = 0; j < source.size(); j++)
    source_classes.insert(source_class_[source[j]]);
  std::set<uint> target_classes;
  for (uint i = 0; i < target.size(); i++)
    target_classes.insert(target_class_[target[i]]);

  for (uint sc = 0; sc < nSourceClasses; sc++) {
    for (uint tc = 0; tc < nTargetClasses; tc++) {
      if (contains(source_classes, sc) && contains(target_classes,tc) && dist_params_(sc,tc).size() < 2 * I - 1) {
        Math1D::Vector<double> new_params(2 * I - 1, 1e-8);
        const uint new_zero_offs = I-1;
        uint prev_maxI = dist_params_(sc,tc).size() / 2;
        for (int i = -prev_maxI - 1; i < prev_maxI; i++)
          new_params[new_zero_offs + i] = dist_params_(sc,tc)[zero_offset_ + i];
        dist_params_(sc,tc) = std::move(new_params);
        zero_offset_ = I;
      }
    }
  }
}

//training without constraints on uncovered positions.
//This is based on the EM-algorithm, where the E-step uses heuristics
void FertilityHMMTrainerDoubleClass::train_em(uint nIter, FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper)
{
  const size_t nSentences = source_sentence_.size();
  const uint nSourceClasses = source_class_.max() + 1;
  const uint nTargetClasses = target_class_.max() + 1;

  std::cerr << "starting Fertility-HMM-DoubleClass training without constraints" << std::endl;

  double max_perplexity = 0.0;
  double approx_sum_perplexity = 0.0;
  double variational_perplexity = 0.0;

  double dict_weight_sum = (prior_weight_active_) ? 1.0 : 0.0; //only used as a flag

  uint maxI = 0;
  for (size_t s = 0; s < nSentences; s++)
    maxI = std::max<uint>(maxI, target_sentence_[s].size());

  SingleLookupTable aux_lookup;

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords, MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  Storage2D<Math1D::Vector<double> > fsingle_align_count(nSourceClasses,nTargetClasses);
  Math2D::Matrix<double> fgrouping_count(nSourceClasses,nTargetClasses);
  Storage2D<Math2D::Matrix<double> > fspan_align_count(nSourceClasses,nTargetClasses);
  for (uint sc = 0; sc < nSourceClasses; sc++) {
    for (uint tc = 0; tc < nTargetClasses; tc++) {
      if (dist_params_(sc, tc).size() > 0) {
        fsingle_align_count(sc,tc).resize_dirty(2*maxI-1);
        fspan_align_count(sc,tc).resize_dirty(maxI+1,maxI);
        assert(dist_params_(sc,tc).size() == 2*maxI-1);
      }
    }
  }

  double fzero_count;
  double fnonzero_count;

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    uint nNotConverged = 0;

    std::cerr << "******* Fertility-HMM-DoubleClass EM-iteration #" << iter << std::endl;

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

    for (uint x = 0; x < fsingle_align_count.xDim(); x++) {
      for (uint y = 0; y < fsingle_align_count.yDim(); y++) {
        fsingle_align_count(x,y).set_constant(0.0);
        fspan_align_count(x,y).set_constant(0.0);
      }
    }

    fgrouping_count.set_constant(0.0);

    max_perplexity = 0.0;
    approx_sum_perplexity = 0.0;


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
      Math3D::Tensor<double> align_expectation;

      compute_approximate_jmarginals(cur_alignment, expansion_move_prob, swap_move_prob, sentence_prob, j_marg);
      compute_approximate_imarginals(cur_alignment, fertility, expansion_move_prob, sentence_prob, i_marg);
      
      // update zero counts
      for (uint c = 0; c <= curJ / 2; c++) {

        fzero_count += 2 * c * i_marg(c, 0);
        fnonzero_count += (curJ - 2 * c + curJ - c) * i_marg(c, 0);
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

      Math1D::Vector<uint> sclass(curJ);
      for (uint j = 0; j < curJ; j++)
        sclass[j] = source_class_[cur_source[j]];

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class_[cur_target[i]];

      if (true) {
        long double main_prob = best_prob * inv_sentence_prob;
        uint prev_aj = MAX_UINT;
        for (uint j = 0; j < curJ; j++) {
          const uint aj = cur_alignment[j];

          //std::cerr << "j: " << j << ", aj: " << aj << std::endl;

          if (aj != 0) {
            if (prev_aj != MAX_UINT) {
              const uint tcpaj = tclass[prev_aj - 1];
              const uint sc = sclass[j-1];
              //cur_align_count(sclass[j-1],tclass[prev_aj - 1])(aj - 1, prev_aj - 1) += main_prob;
              int diff = (int) (aj-1) - (int) (prev_aj-1);
              if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                fsingle_align_count(sc,tcpaj)[zero_offset_ + diff] += main_prob;
              else
                fgrouping_count(sc,tcpaj) += main_prob;
              fspan_align_count(sc,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1) += main_prob;
            }
            prev_aj = aj;
          }
        }

        Math1D::Vector<AlignBaseType> work_alignment = cur_alignment;

        for (uint j = 0; j < curJ; j++) {

          //std::cerr << "--- expansions for j=" << j << std::endl;
          for (uint i = 0; i <= curI; i++) {

            const long double prob = expansion_move_prob(j, i) * inv_sentence_prob;
            if (prob > 0.0) {

              work_alignment[j] = i;
              uint prev_aj = MAX_UINT;
              for (uint j = 0; j < curJ; j++) {
                const uint aj = work_alignment[j];

                if (aj != 0) {
                  if (prev_aj != MAX_UINT) {
                    const uint tcpaj = tclass[prev_aj - 1];
                    const uint sc = sclass[j-1];
                    //cur_align_count(aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]]) += prob;
                    int diff = (int) (aj - 1) - (int) (prev_aj - 1);
                    if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                      fsingle_align_count(sc,tcpaj)[zero_offset_ + diff] += prob;
                    else
                      fgrouping_count(sc,tcpaj) += prob;
                    fspan_align_count(sc,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1) += prob;
                  }
                  prev_aj = aj;
                }
              }
            }
          }
          work_alignment[j] = cur_alignment[j];
        }

        for (uint j1 = 0; j1 < curJ - 1; j1++) {
          for (uint j2 = j1 + 1; j2 < curJ; j2++) {

            const long double prob = swap_move_prob(j1, j2) * inv_sentence_prob;
            if (prob > 0.0) {

              std::swap(work_alignment[j1], work_alignment[j2]);

              uint prev_aj = MAX_UINT;
              for (uint j = 0; j < curJ; j++) {
                const uint aj = work_alignment[j];

                if (aj != 0) {
                  if (prev_aj != MAX_UINT) {
                    assert(prev_aj > 0);
                    const uint tcpaj = tclass[prev_aj - 1];
                    const uint sc = sclass[j-1];
                    //cur_align_count(aj - 1, prev_aj - 1, target_class_[cur_target[prev_aj - 1]]) += prob;
                    int diff = (int) (aj - 1) - (int) (prev_aj - 1);
                    if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                      fsingle_align_count(sc,tcpaj)[zero_offset_ + diff] += prob;
                    else
                      fgrouping_count(sc,tcpaj) += prob;
                    fspan_align_count(sc,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1) += prob;
                  }
                  prev_aj = aj;
                }
              }

              std::swap(work_alignment[j1], work_alignment[j2]);
            }
          }
        }
      }

    } //loop over sentences finished

    // print-outs

    const double reg_term = regularity_term();

    max_perplexity /= source_sentence_.size();
    approx_sum_perplexity /= source_sentence_.size();

    max_perplexity += reg_term;
    approx_sum_perplexity += reg_term;

    std::string transfer = (prev_model != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "Fertility-HMM-DoubleClass max-perplex-energy in between iterations #" <<  (iter - 1)
              << " and " << iter << transfer << ": " << max_perplexity << std::endl;
    std::cerr << "Fertility-HMM-DoubleClass approx-sum-perplex-energy in between iterations #" << (iter - 1)
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

    //update p_zero_ and p_nonzero_
    if (!fix_p0_ /*&& mstep_mode_ == FertMStepCountsOnly */ ) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
      p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);

      std::cerr << "new p_zero: " << p_zero_ << std::endl;
    }
    //update fertilities
    update_fertility_prob(ffert_count, 1e-8);

    //update distortion params
    for (uint sc = 0; sc < nSourceClasses; sc++) {

      std::cerr << "calling m-steps for classes " << sc << ",*" << std::endl;
      for (uint tc = 0; tc < nTargetClasses; tc++) {

        if (dist_params_(sc, tc).size() == 0)
          continue;

        ehmm_m_step(fsingle_align_count(sc,tc), fgrouping_count(sc,tc), fspan_align_count(sc,tc), dist_params_(sc,tc), zero_offset_,
                    dist_grouping_param_(sc,tc), options_.deficient_, options_.redpar_limit_, options_.align_m_step_iter_, options_.gd_stepsize_, true);
      }
    }

    printEval(iter, transfer, "EM");
  } //loop over iter finished

  iter_offs_ = iter - 1;
}

//unconstrained Viterbi training
void FertilityHMMTrainerDoubleClass::train_viterbi(uint nIter, FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper)
{
  const size_t nSentences = source_sentence_.size();
  const uint nSourceClasses = source_class_.max() + 1;
  const uint nTargetClasses = target_class_.max() + 1;

  uint maxI = 0;
  for (size_t s = 0; s < nSentences; s++)
    maxI = std::max<uint>(maxI, target_sentence_[s].size());

  SingleLookupTable aux_lookup;

  const uint nTargetWords = dict_.size();

  HmmAlignProbType align_type = options_.align_type_;
  double dict_weight_sum = (prior_weight_active_) ? 1.0 : 0.0; //only used as a flag


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

  Storage2D<Math1D::Vector<double> > fsingle_align_count(nSourceClasses,nTargetClasses);
  Math2D::Matrix<double> fgrouping_count(nSourceClasses,nTargetClasses);
  Storage2D<Math2D::Matrix<double> > fspan_align_count(nSourceClasses,nTargetClasses);
  for (uint sc = 0; sc < nSourceClasses; sc++) {
    for (uint tc = 0; tc < nTargetClasses; tc++) {
      if (dist_params_(sc, tc).size() > 0) {
        fsingle_align_count(sc,tc).resize_dirty(2*maxI-1);
        fspan_align_count(sc,tc).resize_dirty(maxI+1,maxI);
        assert(dist_params_(sc,tc).size() == 2*maxI-1);
      }
    }
  }

  double fzero_count;
  double fnonzero_count;

  double max_perplexity = 0.0;

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    uint nNotConverged = 0;
    std::string transfer = (prev_model != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "******* Fertility-HMM-DoubleClass Viterbi-iteration #" << iter << std::endl;

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

    for (uint x = 0; x < fsingle_align_count.xDim(); x++) {
      for (uint y = 0; y < fsingle_align_count.yDim(); y++) {
        fsingle_align_count(x,y).set_constant(0.0);
        fspan_align_count(x,y).set_constant(0.0);
      }
    }

    fgrouping_count.set_constant(0.0);

    max_perplexity = 0.0;

    for (size_t s = 0; s < nSentences; s++) {

      //if ((s % 10000) == 0)
      if ((s% 100) == 0)
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

      //std::cerr << "prob: " << best_prob << std::endl;
      max_perplexity -= std::log(best_prob);

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
      Math1D::Vector<uint> sclass(curJ);
      for (uint j = 0; j < curJ; j++)
        sclass[j] = source_class_[cur_source[j]];

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class_[cur_target[i]];

      uint prev_aj = MAX_UINT;
      for (uint j = 0; j < curJ; j++) {
        const uint aj = cur_alignment[j];

        //std::cerr << "j: " << j << ", aj: " << aj << std::endl;

        if (aj != 0) {
          if (prev_aj != MAX_UINT) {
            assert(prev_aj > 0);
            const uint tcpaj = tclass[prev_aj - 1];
            const uint sc = sclass[j-1];
            //cur_align_count(sclass[j-1],tclass[prev_aj - 1])(aj - 1, prev_aj - 1) += 1.0;
            int diff = (int) (aj-1) - (int) (prev_aj-1);
            if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
              fsingle_align_count(sc,tcpaj)[zero_offset_ + diff] += 1.0;
            else
              fgrouping_count(sc,tcpaj) += 1.0;
            fspan_align_count(sc,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1) += 1.0;
          }
          prev_aj = aj;
        }
      }
    } // loop over sentences finished

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, nSentences, dict_weight_sum, smoothed_l0_, l0_beta_, dict_m_step_iter_, dict_, fert_min_dict_entry,
                            msolve_mode_ != MSSolvePGD, gd_stepsize_);

    //update p_zero_ and p_nonzero_
    if (!fix_p0_ /*&& mstep_mode_ == FertMStepCountsOnly */ ) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
      p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);

      std::cerr << "new p_zero: " << p_zero_ << std::endl;
    }
    //update fertilities
    update_fertility_prob(ffert_count, 1e-8);

    //update distortion params
    for (uint sc = 0; sc < nSourceClasses; sc++) {

      std::cerr << "calling m-steps for classes " << sc << ",*" << std::endl;
      for (uint tc = 0; tc < nTargetClasses; tc++) {

        if (dist_params_(sc, tc).size() == 0)
          continue;

        ehmm_m_step(fsingle_align_count(sc,tc), fgrouping_count(sc,tc), fspan_align_count(sc,tc), dist_params_(sc,tc), zero_offset_,
                    dist_grouping_param_(sc,tc), options_.deficient_, options_.redpar_limit_, options_.align_m_step_iter_, options_.gd_stepsize_, true);
      }
    }

    //start ICM stage
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

        //if ((s % 10000) == 0)
        if ((s% 100) == 0)
          std::cerr << "sentence pair #" << s << std::endl;

        const Storage1D<uint>& cur_source = source_sentence_[s];
        const Storage1D<uint>& cur_target = target_sentence_[s];
        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

        Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

        const uint curI = cur_target.size();
        const uint curJ = cur_source.size();

        //std::cerr << "I: " << curI << ", J: " << curJ << std::endl;

        Math1D::NamedVector<uint> cur_fertilities(curI + 1, 0, MAKENAME(fertility));
        for (uint j = 0; j < curJ; j++)
          cur_fertilities[cur_alignment[j]]++;


        Math1D::Vector<uint> sclass(curJ);
        for (uint j = 0; j < curJ; j++)
          sclass[j] = source_class_[cur_source[j]];

        Math1D::Vector<uint> tclass(curI);
        for (uint i = 0; i < curI; i++)
          tclass[i] = target_class_[cur_target[i]];

        Math3D::Tensor<double> cur_align_prob; //no need for a target class dimension, class is detirmened by i_prev
        par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_, dist_params_, dist_grouping_param_, cur_align_prob, options_.align_type_,
                                         options_.deficient_, options_.redpar_limit_, zero_offset_);

        uint prev_aj = MAX_UINT;

        for (uint j = 0; j < curJ; j++) {

          //std::cerr << "j: " << j << std::endl;

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
          const uint sc = (j > 0) ? sclass[j-1] : MAX_UINT;
          const uint scjj = (jj < curJ) ? sclass[jj-1] : MAX_UINT;
          const uint tcpaj = (prev_aj <= curI && prev_aj > 0) ? tclass[prev_aj-1] : MAX_UINT;
          Math1D::Vector<double>& cur_dictcount = fwcount[cur_word];
          Math1D::Vector<double>& cur_fert_count = ffert_count[cur_word];

          double best_change = 0.0;
          uint new_aj = cur_aj;

          //std::cerr << "cur aj: " << cur_aj << std::endl;

          for (uint i = 0; i <= curI; i++) {

            //std::cerr << "i: " << i << std::endl;

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
                  change -=  -std::log(cur_align_prob(sc, cur_aj - 1, prev_aj - 1)) - log_pnonzero;
                if (next_aj != MAX_UINT)
                  change -=  -std::log(cur_align_prob(scjj, next_aj - 1, cur_aj - 1)) - log_pnonzero;
              }
              else {
                change -= -log_pzero;
                if (prev_aj != MAX_UINT && next_aj != MAX_UINT)
                  change -=  -std::log(cur_align_prob(scjj, next_aj - 1, prev_aj - 1)) - log_pnonzero;
              }

              if (i != 0) {
                if (prev_aj != MAX_UINT)
                  change +=  -std::log(cur_align_prob(sc, i - 1, prev_aj - 1)) - log_pnonzero;
                if (next_aj != MAX_UINT)
                  change +=  -std::log(cur_align_prob(scjj, next_aj - 1, i - 1)) - log_pnonzero;
              }
              else {
                change += -log_pzero;
                if (prev_aj != MAX_UINT && next_aj != MAX_UINT)
                  change +=  -std::log(cur_align_prob(scjj, next_aj - 1, prev_aj - 1)) - log_pnonzero;
              }

              if (change < best_change) {
                best_change = change;
                new_aj = i;
              }
            }
          } // end for i

          if (best_change < -0.01) {

            //std::cerr << "best change: " << best_change << ", new aj: " << new_aj << std::endl;
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

            // //cur_align_count(sclass[j-1],tclass[prev_aj - 1])(aj - 1, prev_aj - 1) += 1.0;
            // int diff = (int) (aj-1) - (int) (prev_aj-1);
            // if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
            // fsingle_align_count(sclass[j-1],tclass[prev_aj-1])[zero_offset_ + diff] += 1.0;
            // else
            // fgrouping_count(sclass[j-1],tclass[prev_aj-1]) += 1.0;
            // fspan_align_count(sclass[j-1],tclass[prev_aj-1])(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1) += 1.0;

            if (cur_aj != 0) {
              if (prev_aj != MAX_UINT) {
                int diff = (int) cur_aj - 1 - ((int) prev_aj - 1);
                const uint tcpaj = tclass[prev_aj - 1];
                if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                  fsingle_align_count(sc, tcpaj)[zero_offset_ + diff]--;
                else
                  fgrouping_count(sc,tcpaj)--;
                fspan_align_count(sc,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1)--;
              }
              if (next_aj != MAX_UINT) {
                int diff = (int) next_aj - 1 - ((int) cur_aj - 1);
                const uint tc = tclass[cur_aj - 1];
                if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                  fsingle_align_count(scjj, tc)[zero_offset_ + diff]--;
                else
                  fgrouping_count(scjj,tc)--;
                fspan_align_count(scjj,tc)(zero_offset_ - (cur_aj-1), curI - (cur_aj - 1) - 1)--;
              }
            }
            else {
              if (prev_aj != MAX_UINT && next_aj != MAX_UINT) {
                int diff = (int) next_aj - 1 -  (int) (prev_aj - 1);
                const uint tcpaj = tclass[prev_aj - 1];
                if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                  fsingle_align_count(scjj, tcpaj)[zero_offset_ + diff]--;
                else
                  fgrouping_count(scjj,tcpaj)--;
                fspan_align_count(scjj,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1)--;
              }
            }

            if (new_aj != 0) {
              if (prev_aj != MAX_UINT) {
                const int diff = (int) new_aj - 1 - ((int) (prev_aj - 1));
                const uint tcpaj = tclass[prev_aj - 1];
                if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                  fsingle_align_count(sc, tclass[prev_aj - 1])[zero_offset_ + diff]++;
                else
                  fgrouping_count(sc,tcpaj)++;
                fspan_align_count(sc,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1)++;
              }
              if (next_aj != MAX_UINT) {
                const int diff = (int) next_aj - 1 - ((int) new_aj - 1);
                const uint tc = tclass[new_aj - 1];
                if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                  fsingle_align_count(scjj, tc)[zero_offset_ + diff]++;
                else
                  fgrouping_count(scjj,tc)++;
                fspan_align_count(scjj,tc)(zero_offset_ - (new_aj-1), curI - (new_aj - 1) - 1)++;
              }
            }
            else {
              if (prev_aj != MAX_UINT && next_aj != MAX_UINT) {
                const int diff = (int) next_aj - 1 - ((int) prev_aj - 1);
                const uint tcpaj = tclass[prev_aj - 1];
                if (options_.align_type_ == HmmAlignProbFullpar  || abs(diff) <= options_.redpar_limit_)
                  fsingle_align_count(scjj, tcpaj)[zero_offset_ + diff]++;
                else
                  fgrouping_count(scjj,tcpaj)++;
                fspan_align_count(scjj,tcpaj)(zero_offset_ - (prev_aj-1), curI - (prev_aj - 1) - 1)++;
              }
            }
          }

          if (cur_alignment[j] != 0)
            prev_aj = cur_alignment[j];

        } // end for j
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

      //update dictionary
      update_dict_from_counts(fwcount, prior_weight_, nSentences, dict_weight_sum, smoothed_l0_, l0_beta_, dict_m_step_iter_, dict_, fert_min_dict_entry,
                              msolve_mode_ != MSSolvePGD, gd_stepsize_);

      //update p_zero_ and p_nonzero_
      if (!fix_p0_ /*&& mstep_mode_ == FertMStepCountsOnly */ ) {
        double fsum = fzero_count + fnonzero_count;
        p_zero_ = std::max<double>(fert_min_p0, fzero_count / fsum);
        p_nonzero_ = std::max<double>(fert_min_p0, fnonzero_count / fsum);

        std::cerr << "new p_zero: " << p_zero_ << std::endl;
      }
      //update fertilities
      update_fertility_prob(ffert_count, 1e-8);

      //update distortion params
      for (uint sc = 0; sc < nSourceClasses; sc++) {

        std::cerr << "calling m-steps for classes " << sc << ",*" << std::endl;
        for (uint tc = 0; tc < nTargetClasses; tc++) {

          if (dist_params_(sc, tc).size() == 0)
            continue;

          ehmm_m_step(fsingle_align_count(sc,tc), fgrouping_count(sc,tc), fspan_align_count(sc,tc), dist_params_(sc,tc), zero_offset_,
                      dist_grouping_param_(sc,tc), options_.deficient_, options_.redpar_limit_, options_.align_m_step_iter_, options_.gd_stepsize_, true);
        }
      }

      max_perplexity = 0.0;
      for (size_t s = 0; s < source_sentence_.size(); s++) {
        max_perplexity -= logl(FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]));
      }

      max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
      max_perplexity /= source_sentence_.size();

      std::cerr << "Fertility-HMM-DoubleClass energy after iteration #" << iter << transfer << ": " << max_perplexity << std::endl;

    } // end if prev_model != 0

    if (possible_ref_alignments_.size() > 0) {

      std::cerr << "#### Fertility-HMM-DoubleClass-AER after iteration #" << iter << transfer << ": " << AER() << std::endl;
      std::cerr << "#### Fertility-HMM-DoubleClass-fmeasure after iteration #" << iter << transfer << ": " << f_measure() << std::endl;
      std::cerr << "#### Fertility-HMM-DoubleClass-DAE/S after iteration #" << iter << transfer << ": " << DAE_S() << std::endl;
    }

  } //loop over iter finished

  iter_offs_ = iter - 1;
}