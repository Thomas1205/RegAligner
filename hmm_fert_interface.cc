/**** written by Thomas Schoenemann as a private person, since October 2017 ****/

#include "hmm_fert_interface.hh"
#include "hmm_forward_backward.hh"
#include "alignment_computation.hh"

#include "training_common.hh"   // for get_wordlookup() and dictionary m-step

HmmFertInterface::HmmFertInterface(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                                   const Storage1D<Math1D::Vector<uint> >& target_sentence,
                                   const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                                   const std::map<uint,std::set<std::pair< AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                                   SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords, uint nTargetWords, uint fertility_limit)
  : FertilityModelTrainerBase(source_sentence, slookup, target_sentence, sure_ref_alignments, possible_ref_alignments, dict,
                              wcooc, nSourceWords, nTargetWords, fertility_limit) {}

HmmFertInterfaceTargetClasses::HmmFertInterfaceTargetClasses(const HmmWrapperWithTargetClasses& wrapper, const Storage1D<Math1D::Vector<uint> >& source_sentence,
    const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target_sentence, const std::map < uint,
    std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
    const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
    SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords, uint nTargetWords,
    uint fertility_limit)
  : HmmFertInterface(source_sentence, slookup, target_sentence, sure_ref_alignments, possible_ref_alignments, dict,
                     wcooc, nSourceWords, nTargetWords, fertility_limit), hmm_wrapper_(wrapper)
{
  SingleLookupTable aux_lookup;

  uint sum_iter = 0;

  for (uint s = 0; s < source_sentence_.size(); s++) {

    const Storage1D<uint>& cur_source = source_sentence_[s];
    const Storage1D<uint>& cur_target = target_sentence_[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    const uint curI = cur_target.size();
    const uint curJ = cur_source.size();

    Math1D::Vector<uint> fertility(curI + 1, 0);
    Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
    Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

    update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility, expansion_move_prob,
                                     swap_move_prob, best_known_alignment_[s]);
  }
}

/*virtual*/
long double HmmFertInterfaceTargetClasses::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
    Math1D::Vector<AlignBaseType>& alignment, AlignmentSetConstraints* constraints)
{
  const uint curI = target.size();

  if ((curI - 1) >= hmm_wrapper_.align_model_.size()
      || hmm_wrapper_.align_model_[curI - 1].xDim() == 0)
    prepare_external_alignment(source, target, lookup, alignment);

  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  assert(hmm_wrapper_.initial_prob_.size() > 0);

  Math2D::Matrix<double> slim_align_model(curI+1, curI);
  par2nonpar_hmm_alignment_model(tclass, hmm_wrapper_.align_model_[curI - 1], hmm_wrapper_.hmm_options_, slim_align_model);

  long double prob = compute_ehmmc_viterbi_alignment(source, lookup, target, tclass, dict_, slim_align_model, hmm_wrapper_.initial_prob_[curI - 1],
                     alignment, hmm_wrapper_.hmm_options_);

  return prob;
}

/*virtual*/
void HmmFertInterfaceTargetClasses::compute_approximate_jmarginals(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
    Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg, bool& converged) const
{
  converged = true;

  const uint curJ = source.size();
  const uint curI = target.size();

  j_marg.resize(curI + 1, curJ);

  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  //const Math3D::Tensor<double>& cur_align_model = hmm_wrapper_.align_model_[curI - 1];

  Math2D::Matrix<double> cur_align_model(curI+1, curI);
  par2nonpar_hmm_alignment_model(tclass, hmm_wrapper_.align_model_, hmm_wrapper_.hmm_options_, cur_align_model);

  Math2D::NamedMatrix<long double> forward(2 * curI, curJ, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2 * curI, curJ, MAKENAME(backward));

  Math1D::Vector<double> empty_vec;
  const Math1D::Vector<double>& init_prob = (hmm_wrapper_.initial_prob_.size() > 0) ? hmm_wrapper_.initial_prob_[curI - 1] : empty_vec;



  Math2D::Matrix<double> cur_dict(curJ,curI+1);
  compute_dictmat(source, lookup, target, dict_, cur_dict);

  calculate_hmmc_forward(tclass, cur_dict, cur_align_model, init_prob, hmm_wrapper_.hmm_options_, forward);
  calculate_hmmc_backward(tclass, cur_dict, cur_align_model, init_prob, hmm_wrapper_.hmm_options_, backward, false);

  long double sentence_prob = 0.0;
  for (uint i = 0; i < forward.xDim(); i++)
    sentence_prob += forward(i, curJ - 1);
  long double inv_sentence_prob = 1.0 / sentence_prob;

  for (uint i = 0; i < curI; i++) {
    const uint t_idx = target[i];

    for (uint j = 0; j < curJ; j++) {

      if (dict_[t_idx][lookup(j, i)] > 1e-305) {
        const double contrib = inv_sentence_prob * forward(i, j) * backward(i, j) / dict_[t_idx][lookup(j, i)];

        j_marg(i + 1, j) = contrib;
      }
      else
        j_marg(i + 1, j) = 0.0;
    }
  }

  for (uint j = 0; j < curJ; j++) {

    const uint s_idx = source[j];

    if (dict_[0][s_idx - 1] > 1e-305) {

      long double contrib = 0.0;
      for (uint i = curI; i < forward.xDim(); i++)
        contrib += forward(i, j) * backward(i, j);

      j_marg(0, j) = inv_sentence_prob * contrib / dict_[0][s_idx - 1];
    }
    else
      j_marg(0, j) = 0.0;
  }
}

/*virtual*/
long double HmmFertInterfaceTargetClasses::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
    uint& nIter, Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob,
    Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment) const
{
  const uint curJ = source.size();
  const uint curI = target.size();

  //std::cerr << "---update_alignment_by_hillclimbing target classes, J: " << curJ << ", I: " << curI << std::endl;

  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  Math1D::Vector<AlignBaseType> internal_hyp_alignment(curJ);

  Math2D::Matrix<double> slim_align_model(curI+1, curI);
  par2nonpar_hmm_alignment_model(tclass, hmm_wrapper_.align_model_[curI - 1], hmm_wrapper_.hmm_options_, slim_align_model);

  //note: here the index 0 will represent alignments to 0. we will have to convert to the proper representations
  // when calling routines below
  assert(hmm_wrapper_.initial_prob_.size() > 0);
  long double best_prob = compute_ehmmc_viterbi_alignment(source, lookup, target, tclass, dict_, slim_align_model,
                          hmm_wrapper_.initial_prob_[curI - 1], alignment, hmm_wrapper_.hmm_options_);
  assert(best_prob > 0.0);

  bool changed = make_alignment_feasible(source, target, lookup, alignment);
  if (changed) {

    external2internal_hmm_alignment(alignment, curI, hmm_wrapper_.hmm_options_, internal_hyp_alignment);

    best_prob = hmmc_alignment_prob(source, lookup, target, tclass, dict_, slim_align_model,
                                    hmm_wrapper_.initial_prob_[curI-1], internal_hyp_alignment, true);
    assert(best_prob > 0.0);
  }

  //need this after make_alignment_feasible
  fertility.resize(curI + 1);
  fertility.set_constant(0);

  for (uint j = 0; j < curJ; j++) {
    fertility[alignment[j]]++;
  }

  //NOTE: lots of room for speed-ups here -> switch to incremental calculation!

  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);
  swap_prob.set_constant(0.0);
  expansion_prob.set_constant(0.0);

  //std::cerr << "J: " << curJ << ", I: " << curI << std::endl;
  //std::cerr << "base alignment: " << alignment << std::endl;

  Math1D::Vector<AlignBaseType> hyp_alignment = alignment;

  //a) expansion moves
  for (uint j = 0; j < curJ; j++) {

    //std::cerr << "j: " << j << std::endl;

    const uint cur_aj = alignment[j];

    for (uint i = 0; i <= curI; i++) {

      //std::cerr << "i: " << i << std::endl;

      if (i == 0 && 2 * fertility[0] + 2 > curJ)
        continue;

      if (i != cur_aj
          && (i == 0 || fertility[i] < fertility_limit_[target[i - 1]])) {

        hyp_alignment[j] = i;

        //now convert to internal mode
        external2internal_hmm_alignment(hyp_alignment, curI, hmm_wrapper_.hmm_options_, internal_hyp_alignment);

        expansion_prob(j, i) = hmmc_alignment_prob(source, lookup, target, tclass, dict_, slim_align_model,
                               hmm_wrapper_.initial_prob_[curI - 1], internal_hyp_alignment, true);
      }
    }

    //restore for next iteration
    hyp_alignment[j] = cur_aj;
  }

  //b) swap moves
  for (uint j1 = 0; j1 < curJ - 1; j1++) {

    //std::cerr << "j1: " << j1 << std::endl;

    const uint cur_aj1 = alignment[j1];

    for (uint j2 = j1 + 1; j2 < curJ; j2++) {

      //std::cerr << "j2: " << j2 << std::endl;

      const uint cur_aj2 = alignment[j2];

      if (cur_aj1 != cur_aj2) {

        std::swap(hyp_alignment[j1], hyp_alignment[j2]);

        //now convert to internal mode
        external2internal_hmm_alignment(hyp_alignment, curI, hmm_wrapper_.hmm_options_, internal_hyp_alignment);

        long double cur_swap_prob = hmmc_alignment_prob(source, lookup, target, tclass, dict_, slim_align_model,
                                    hmm_wrapper_.initial_prob_[curI - 1], internal_hyp_alignment, true);

        swap_prob(j1, j2) = cur_swap_prob;
        swap_prob(j2, j1) = cur_swap_prob;

        //reverse for next iteration
        std::swap(hyp_alignment[j1], hyp_alignment[j2]);
      }
    }
  }

  return best_prob;
}

/*virtual*/ void HmmFertInterfaceTargetClasses::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  //this is just an interface class. we would need the parameters to do this properly
}

/**************************************************************************/

HmmFertInterfaceDoubleClasses::HmmFertInterfaceDoubleClasses(const HmmWrapperDoubleClasses& wrapper, const Storage1D<Math1D::Vector<uint> >& source_sentence,
    const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target_sentence,
    const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
    const std::map<uint,std::set<std::pair< AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
    SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords,
    uint nTargetWords, uint fertility_limit)
  : HmmFertInterface(source_sentence, slookup, target_sentence, sure_ref_alignments, possible_ref_alignments, dict,
                     wcooc, nSourceWords, nTargetWords, fertility_limit), hmm_wrapper_(wrapper)
{
  SingleLookupTable aux_lookup;

  uint sum_iter = 0;

  for (uint s = 0; s < source_sentence_.size(); s++) {

    const Storage1D<uint>& cur_source = source_sentence_[s];
    const Storage1D<uint>& cur_target = target_sentence_[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

    const uint curI = cur_target.size();
    const uint curJ = cur_source.size();

    Math1D::Vector<uint> fertility(curI + 1, 0);
    Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
    Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

    update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility, expansion_move_prob,
                                     swap_move_prob, best_known_alignment_[s]);
  }
}

/*virtual*/ long double HmmFertInterfaceDoubleClasses::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
    Math2D::Matrix<long double>& expansion_prob, Math2D::Matrix<long double >& swap_prob,
    Math1D::Vector<AlignBaseType>& alignment) const
{
  const uint curJ = source.size();
  const uint curI = target.size();

  //std::cerr << "---update_alignment_by_hillclimbing double classes, J: " << curJ << ", I: " << curI << std::endl;
  //std::cerr << "J: " << curJ << ", I: " << curI << std::endl;

  Math1D::Vector<uint> sclass(curJ);
  for (uint j = 0; j < curJ; j++)
    sclass[j] = hmm_wrapper_.source_class_[source[j]];

  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  Math1D::Vector<AlignBaseType> internal_hyp_alignment(curJ);

  //std::cerr << "calling par2nonpar_hmmcc_alignment_model" << std::endl;
  Math3D::Tensor<double> cur_align_model;
  par2nonpar_hmmcc_alignment_model(sclass, tclass, hmm_wrapper_.source_fert_, hmm_wrapper_.dist_params_, hmm_wrapper_.dist_grouping_param_,
                                   cur_align_model, hmm_wrapper_.hmm_options_.align_type_, hmm_wrapper_.hmm_options_.deficient_,
                                   hmm_wrapper_.hmm_options_.redpar_limit_, hmm_wrapper_.zero_offset_);


  //note: here the index 0 will represent alignments to 0. we will have to convert to the proper representations
  // when calling routines below
  assert(hmm_wrapper_.initial_prob_.size() > 0);
  //std::cerr << "calling compute_ehmm_viterbi_alignment" << std::endl;
  long double best_prob = compute_ehmmcc_viterbi_alignment(source, lookup, target, sclass, tclass, dict_, cur_align_model,
                          hmm_wrapper_.initial_prob_[curI - 1], alignment, hmm_wrapper_.hmm_options_);
  assert(best_prob > 0.0);

  //std::cerr << "calling make_alignment_feasible" << std::endl;
  bool changed = make_alignment_feasible(source, target, lookup, alignment);
  if (changed) {

    external2internal_hmm_alignment(alignment, curI, hmm_wrapper_.hmm_options_, internal_hyp_alignment);

    best_prob = hmmcc_alignment_prob(source, lookup, target, sclass, tclass, dict_, cur_align_model,
                                     hmm_wrapper_.initial_prob_, internal_hyp_alignment, true);
    assert(best_prob > 0.0);
  }

  //need this after make_alignment_feasible
  fertility.resize(curI + 1);
  fertility.set_constant(0);

  for (uint j = 0; j < curJ; j++) {
    fertility[alignment[j]]++;
  }

  //NOTE: lots of room for speed-ups here -> switch to incremental calculation!

  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);
  swap_prob.set_constant(0.0);
  expansion_prob.set_constant(0.0);

  //std::cerr << "J: " << curJ << ", I: " << curI << std::endl;
  //std::cerr << "base alignment: " << alignment << std::endl;

  Math1D::Vector<AlignBaseType> hyp_alignment = alignment;

  //a) expansion moves
  for (uint j = 0; j < curJ; j++) {

    //std::cerr << "j: " << j << std::endl;

    const uint cur_aj = alignment[j];

    for (uint i = 0; i <= curI; i++) {

      //std::cerr << "i: " << i << std::endl;

      if (i == 0 && 2 * fertility[0] + 2 > curJ)
        continue;

      if (i != cur_aj
          && (i == 0 || fertility[i] < fertility_limit_[target[i - 1]])) {

        hyp_alignment[j] = i;

        //now convert to internal mode
        external2internal_hmm_alignment(hyp_alignment, curI, hmm_wrapper_.hmm_options_, internal_hyp_alignment);

        expansion_prob(j, i) = hmmcc_alignment_prob(source, lookup, target, sclass, tclass, dict_, cur_align_model,
                               hmm_wrapper_.initial_prob_, internal_hyp_alignment, true);
      }
    }

    //restore for next iteration
    hyp_alignment[j] = cur_aj;
  }

  //b) swap moves
  for (uint j1 = 0; j1 < curJ - 1; j1++) {

    //std::cerr << "j1: " << j1 << std::endl;

    const uint cur_aj1 = alignment[j1];

    for (uint j2 = j1 + 1; j2 < curJ; j2++) {

      //std::cerr << "j2: " << j2 << std::endl;

      const uint cur_aj2 = alignment[j2];

      if (cur_aj1 != cur_aj2) {

        std::swap(hyp_alignment[j1], hyp_alignment[j2]);

        //now convert to internal mode
        external2internal_hmm_alignment(hyp_alignment, curI, hmm_wrapper_.hmm_options_, internal_hyp_alignment);

        long double cur_swap_prob = hmmcc_alignment_prob(source, lookup, target, sclass, tclass, dict_, cur_align_model,
                                    hmm_wrapper_.initial_prob_, internal_hyp_alignment, true);

        swap_prob(j1, j2) = cur_swap_prob;
        swap_prob(j2, j1) = cur_swap_prob;

        //reverse for next iteration
        std::swap(hyp_alignment[j1], hyp_alignment[j2]);
      }
    }
  }

  return best_prob;
}


/*virtual*/ long double HmmFertInterfaceDoubleClasses::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment,
    AlignmentSetConstraints* constraints)
{
  const uint curI = target.size();
  const uint curJ = source.size();

  prepare_external_alignment(source, target, lookup, alignment);

  Math1D::Vector<uint> sclass(curJ);
  for (uint j = 0; j < curJ; j++)
    sclass[j] = hmm_wrapper_.source_class_[source[j]];

  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  assert(hmm_wrapper_.initial_prob_.size() > 0);

  Math3D::Tensor<double> cur_align_model;
  par2nonpar_hmmcc_alignment_model(sclass, tclass, hmm_wrapper_.source_fert_, hmm_wrapper_.dist_params_, hmm_wrapper_.dist_grouping_param_,
                                   cur_align_model, hmm_wrapper_.hmm_options_.align_type_, hmm_wrapper_.hmm_options_.deficient_,
                                   hmm_wrapper_.hmm_options_.redpar_limit_, hmm_wrapper_.zero_offset_);


  long double prob = compute_ehmmcc_viterbi_alignment(source, lookup, target, sclass, tclass, dict_, cur_align_model,
                     hmm_wrapper_.initial_prob_[curI - 1], alignment, hmm_wrapper_.hmm_options_);

  return prob;
}

/*virtual*/ void HmmFertInterfaceDoubleClasses::compute_approximate_jmarginals(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg,
    bool& converged) const
{
  converged = true;

  const uint curJ = source.size();
  const uint curI = target.size();

  j_marg.resize(curI + 1, curJ);

  Math2D::NamedMatrix<long double> forward(2 * curI, curJ, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2 * curI, curJ, MAKENAME(backward));

  Math1D::Vector<double> empty_vec;
  const Math1D::Vector<double>& init_prob = (hmm_wrapper_.initial_prob_.size() > 0) ? hmm_wrapper_.initial_prob_[curI - 1] : empty_vec;

  Math1D::Vector<uint> sclass(curJ);
  for (uint j = 0; j < curJ; j++)
    sclass[j] = hmm_wrapper_.source_class_[source[j]];

  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  Math3D::Tensor<double> cur_align_model(sclass.max()+1, curI+1, curI);
  par2nonpar_hmmcc_alignment_model(sclass, tclass, hmm_wrapper_.source_fert_, hmm_wrapper_.dist_params_, hmm_wrapper_.dist_grouping_param_,
                                   cur_align_model, hmm_wrapper_.hmm_options_.align_type_, hmm_wrapper_.hmm_options_.deficient_,
                                   hmm_wrapper_.hmm_options_.redpar_limit_, hmm_wrapper_.zero_offset_);

  Math2D::Matrix<double> cur_dict(curJ,curI+1);
  compute_dictmat(source, lookup, target, dict_, cur_dict);

  calculate_hmmcc_forward(sclass, tclass, cur_dict, cur_align_model, init_prob, hmm_wrapper_.hmm_options_, forward);
  calculate_hmmcc_backward(sclass, tclass, cur_dict, cur_align_model, init_prob, hmm_wrapper_.hmm_options_, backward, false);

  long double sentence_prob = 0.0;
  for (uint i = 0; i < forward.xDim(); i++)
    sentence_prob += forward(i, curJ - 1);
  long double inv_sentence_prob = 1.0 / sentence_prob;

  for (uint i = 0; i < curI; i++) {
    const uint t_idx = target[i];

    for (uint j = 0; j < curJ; j++) {

      if (dict_[t_idx][lookup(j, i)] > 1e-305) {
        const double contrib = inv_sentence_prob * forward(i, j) * backward(i, j) / dict_[t_idx][lookup(j, i)];

        j_marg(i + 1, j) = contrib;
      }
      else
        j_marg(i + 1, j) = 0.0;
    }
  }

  for (uint j = 0; j < curJ; j++) {

    const uint s_idx = source[j];

    if (dict_[0][s_idx - 1] > 1e-305) {

      long double contrib = 0.0;
      for (uint i = curI; i < forward.xDim(); i++)
        contrib += forward(i, j) * backward(i, j);

      j_marg(0, j) = inv_sentence_prob * contrib / dict_[0][s_idx - 1];
    }
    else
      j_marg(0, j) = 0.0;
  }
}

// /*virtual*/ double HmmFertInterfaceDoubleClasses::compute_approximate_marginals(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
// Math1D::Vector<AlignBaseType>& alignment, Math2D::Matrix<double>& j_marg, Math2D::Matrix<double>& i_marg,
// double hc_mass, bool& converged) const
// {
// TODO("");
// }

/*virtual*/ void HmmFertInterfaceDoubleClasses::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  //this is just an interface class. we would need the parameters to do this properly
}
