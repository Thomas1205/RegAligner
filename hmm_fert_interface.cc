/**** written by Thomas Schoenemann as a private person, since October 2017 ****/

#include "hmm_fert_interface.hh"
#include "hmm_forward_backward.hh"
#include "alignment_computation.hh"

#include "training_common.hh"   // for get_wordlookup() and dictionary m-step

HmmFertInterface::HmmFertInterface(const HmmWrapperWithClasses& wrapper, const Storage1D<Math1D::Vector<uint> >& source_sentence,
                                   const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target_sentence, const std::map < uint,
                                   std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                                   const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                                   SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords, uint nTargetWords,
                                   uint fertility_limit)
  : FertilityModelTrainerBase(source_sentence, slookup, target_sentence, sure_ref_alignments, possible_ref_alignments, dict,
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

    Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));
    Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
    Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

    update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility, expansion_move_prob,
                                     swap_move_prob, best_known_alignment_[s]);
  }
}

/*virtual*/
long double HmmFertInterface::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  const uint curI = target.size();

  if ((curI - 1) >= hmm_wrapper_.align_model_.size()
      || hmm_wrapper_.align_model_[curI - 1].xDim() == 0)
    prepare_external_alignment(source, target, lookup, alignment);

  Storage1D < uint > tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  long double prob = 0.0;

  assert(hmm_wrapper_.initial_prob_.size() > 0);

  prob = compute_ehmm_viterbi_alignment(source, lookup, target, tclass, dict_,
                                        hmm_wrapper_.align_model_[curI - 1], hmm_wrapper_.initial_prob_[curI - 1],
                                        alignment, hmm_wrapper_.hmm_options_);

  return prob;
}

/*virtual*/
void HmmFertInterface::compute_approximate_jmarginals(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment,
    Math2D::Matrix<double>& j_marg, bool& converged) const
{
  converged = true;

  const uint curJ = source.size();
  const uint curI = target.size();

  j_marg.resize(curI + 1, curJ);

  const Math3D::Tensor<double>& cur_align_model = hmm_wrapper_.align_model_[curI - 1];

  Math2D::NamedMatrix<long double> forward(2 * curI, curJ, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2 * curI, curJ, MAKENAME(backward));

  const HmmAlignProbType align_type = hmm_wrapper_.hmm_options_.align_type_;
  const bool start_empty_word = hmm_wrapper_.hmm_options_.start_empty_word_;
  const int redpar_limit = hmm_wrapper_.hmm_options_.redpar_limit_;

  Math1D::Vector<double> empty_vec;
  const Math1D::Vector<double>& init_prob = (hmm_wrapper_.initial_prob_.size() > 0)? hmm_wrapper_.initial_prob_[curI - 1] : empty_vec;

  Storage1D<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  Math2D::Matrix<double> cur_dict(curJ,curI+1);
  compute_dictmat(source, lookup, target, dict_, cur_dict);

  calculate_hmm_forward(tclass, cur_dict, cur_align_model, init_prob, align_type, start_empty_word, forward, redpar_limit);
  calculate_hmm_backward(tclass, cur_dict, cur_align_model, init_prob, align_type, start_empty_word, backward, false, redpar_limit);

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
long double HmmFertInterface::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob,
    Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment) const
{

  const uint curJ = source.size();
  const uint curI = target.size();

  Storage1D<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = hmm_wrapper_.target_class_[target[i]];

  Math1D::Vector<AlignBaseType> internal_hyp_alignment(curJ);

  //note: here the index 0 will represent alignments to 0. we will have to convert to the proper representations
  // when calling routines below
  assert(hmm_wrapper_.initial_prob_.size() > 0);
  long double best_prob = compute_ehmm_viterbi_alignment(source, lookup, target, tclass, dict_, hmm_wrapper_.align_model_[curI - 1],
                                                         hmm_wrapper_.initial_prob_[curI - 1], alignment, hmm_wrapper_.hmm_options_);

  bool changed = make_alignment_feasible(source, target, lookup, alignment);
  if (changed) {

    external2internal_hmm_alignment(alignment, curI, hmm_wrapper_.hmm_options_, internal_hyp_alignment);

    best_prob = hmm_alignment_prob(source, lookup, target, tclass, dict_, hmm_wrapper_.align_model_,
                                   hmm_wrapper_.initial_prob_, internal_hyp_alignment, true);
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

        expansion_prob(j, i) = hmm_alignment_prob(source, lookup, target, tclass, dict_, hmm_wrapper_.align_model_,
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

        long double cur_swap_prob = hmm_alignment_prob(source, lookup, target, tclass, dict_, hmm_wrapper_.align_model_, hmm_wrapper_.initial_prob_,
                                                       internal_hyp_alignment, true);

        swap_prob(j1, j2) = cur_swap_prob;
        swap_prob(j2, j1) = cur_swap_prob;

        //reverse for next iteration
        std::swap(hyp_alignment[j1], hyp_alignment[j2]);
      }
    }
  }

  return best_prob;
}

/*virtual*/ void HmmFertInterface::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{

  //this is just an interface class. we would need the parameters to do this properly
}
