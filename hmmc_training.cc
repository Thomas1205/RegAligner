/**** written by Thomas Schoenemann as a private person, since 2019 ****/

#include "hmmc_training.hh"

#include "training_common.hh"
#include "hmm_forward_backward.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"
#include "conditional_m_steps.hh"

#include "projection.hh"
#include "stl_out.hh"
#include "storage_util.hh"

HmmWrapperWithTargetClasses::HmmWrapperWithTargetClasses(const FullHMMAlignmentModelSingleClass& align_model,
    const FullHMMAlignmentModelSingleClass& dev_align_model,
    const InitialAlignmentProbability& initial_prob,
    const Math2D::Matrix<double>& hmmc_dist_params, const Math1D::Vector<double>& hmmc_dist_grouping_param,
    const Storage1D<WordClassType>& target_class, const HmmOptions& hmm_options)
  : HmmWrapperBase(hmm_options), dist_params_(hmmc_dist_params), dist_grouping_param_(hmmc_dist_grouping_param),
    align_model_(align_model), dev_align_model_(dev_align_model), initial_prob_(initial_prob), target_class_(target_class)
{
}

/*virtual*/ long double HmmWrapperWithTargetClasses::compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
    Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode, bool verbose,
    double min_dict_entry) const
{
  const uint curI = target_sentence.size();
  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = target_class_[target_sentence[i]];

  const Math3D::Tensor<double>& cur_align_model = (align_model_[curI - 1].size() > 0) ? align_model_[curI - 1] : dev_align_model_[curI-1];

  Math2D::Matrix<double> slim_align_model(curI+1, curI);
  par2nonpar_hmm_alignment_model(tclass, cur_align_model, hmm_options_, slim_align_model);

  return ::compute_ehmmc_viterbi_alignment(source_sentence, slookup, target_sentence, tclass, dict, slim_align_model,
         initial_prob_[curI - 1], viterbi_alignment, hmm_options_, false, false);
}

/*virtual*/ void HmmWrapperWithTargetClasses::compute_ehmm_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
    double threshold) const
{
  const uint curI = target_sentence.size();
  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = target_class_[target_sentence[i]];

  const Math3D::Tensor<double>& cur_align_model = (align_model_[curI - 1].size() > 0) ? align_model_[curI - 1] : dev_align_model_[curI-1];

  Math2D::Matrix<double> slim_align_model(curI+1, curI);
  par2nonpar_hmm_alignment_model(tclass, cur_align_model, hmm_options_, slim_align_model);


  ::compute_ehmmc_postdec_alignment(source_sentence, slookup, target_sentence, tclass, dict, slim_align_model,
                                    initial_prob_[curI-1], hmm_options_, postdec_alignment, threshold);

}

/*virtual*/ void HmmWrapperWithTargetClasses::fill_dist_params(Math1D::Vector<double>& hmm_dist_params, double& hmm_dist_grouping_param) const
{
  hmm_dist_params.resize_dirty(dist_params_.xDim());
  hmm_dist_params.set_constant(0.0);
  hmm_dist_grouping_param = 0.0;

  for (uint tc = 0; tc < dist_params_.yDim(); tc++) {
    for (uint k = 0; k < dist_params_.xDim(); k++)
      hmm_dist_params[k] += dist_params_(k,tc);
    hmm_dist_grouping_param += dist_grouping_param_[tc];
  }

  double sum = hmm_dist_params.sum() + hmm_dist_grouping_param;
  if (sum > 0.0) {
    hmm_dist_params *= 1.0 / sum;
    hmm_dist_grouping_param *= 1.0 / sum;
  }
}

/*virtual*/ void HmmWrapperWithTargetClasses::fill_dist_params(uint nTargetClasses, Math2D::Matrix<double>& hmmc_dist_params,
    Math1D::Vector<double>& hmmc_dist_grouping_param) const
{
  if (&hmmc_dist_params != &dist_params_) {
    hmmc_dist_params = dist_params_;
    hmmc_dist_grouping_param = dist_grouping_param_;
  }
}

/*virtual*/ void HmmWrapperWithTargetClasses::fill_dist_params(uint nSourceClasses, uint nTargetClasses,
    Storage2D<Math1D::Vector<double> >& hmmcc_dist_params,
    Math2D::Matrix<double>& hmmcc_dist_grouping_param) const
{
  hmmcc_dist_params.resize(nSourceClasses,nTargetClasses);
  hmmcc_dist_grouping_param.resize_dirty(dist_grouping_param_.size(),nTargetClasses);
  for (uint tc = 0; tc < nTargetClasses; tc++) {
    //std::cerr << "tc: " << tc << std::endl;
    for (uint k = 0; k < dist_grouping_param_.size(); k++) {
      hmmcc_dist_grouping_param(k, tc) = dist_grouping_param_[k];
    }
  }
  for (uint sc = 0; sc < nSourceClasses; sc++) {
    for (uint tc = 0; tc < nTargetClasses; tc++) {
      //std::cerr << "sc: " << sc << ", tc: " << tc << std::endl;
      if (hmmcc_dist_params(sc, tc).size() == 0)
        continue;
      hmmcc_dist_params(sc, tc).resize_dirty(dist_params_.xDim());
      for (uint k = 0; k < dist_params_.xDim(); k++)
        hmmcc_dist_params(sc, tc)[k] += dist_params_(k,tc);
    }
  }
}

long double hmmc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup, const Storage1D<uint>& target,
                                const Storage1D<uint>& tclass, const SingleWordDictionary& dict, const FullHMMAlignmentModelSingleClass& align_model,
                                const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                bool with_dict)
{
  const uint I = target.size();
  const uint J = source.size();

  const Math3D::Tensor<double>& cur_align_model = align_model[I - 1];
  const Math1D::Vector<double>& cur_initial_prob = initial_prob[I - 1];

  //std::cerr << "J: " << J << ", I: " << I << std::endl;
  //std::cerr << "alignment: " << alignment << std::endl;

  assert(J == alignment.size());

  long double prob = (alignment[0] == 2 * I) ? cur_initial_prob[I] : cur_initial_prob[alignment[0]];
  //assert(prob > 0.0);

  if (with_dict) {
    for (uint j = 0; j < J; j++) {
      const uint aj = alignment[j];
      if (aj < I)
        prob *= dict[target[aj]][slookup(j, aj)];
      else
        prob *= dict[0][source[j] - 1];
    }
  }
  //assert(prob > 0.0);

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
      prob *= cur_align_model(std::min(I, aj), prev_aj, tclass[prev_aj]);

    //assert(prob > 0.0);
  }

  return prob;
}

long double hmmc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                const Storage1D<uint>& target, const Storage1D<uint>& tclass,
                                const SingleWordDictionary& dict, const Math2D::Matrix<double>& cur_align_model,
                                const Math1D::Vector<double>& cur_initial_prob, const Storage1D<AlignBaseType>& alignment,
                                bool with_dict)
{
  const uint I = target.size();
  const uint J = source.size();

  //std::cerr << "J: " << J << ", I: " << I << std::endl;
  //std::cerr << "alignment: " << alignment << std::endl;

  assert(J == alignment.size());

  long double prob = (alignment[0] == 2 * I) ? cur_initial_prob[I] : cur_initial_prob[alignment[0]];
  //assert(prob > 0.0);

  if (with_dict) {
    for (uint j = 0; j < J; j++) {
      const uint aj = alignment[j];
      if (aj < I)
        prob *= dict[target[aj]][slookup(j, aj)];
      else
        prob *= dict[0][source[j] - 1];
    }
  }
  //assert(prob > 0.0);

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

    //assert(prob > 0.0);
  }

  return prob;
}

void init_hmm_from_prev(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<WordClassType>& target_class,
                        FullHMMAlignmentModelSingleClass& align_model, Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                        Math1D::Vector<double>& source_fert_prob, InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                        const HmmOptions& options, TransferMode transfer_mode = TransferViterbi, uint maxAllI = 0)
{
  const uint nClasses = target_class.max() + 1;

  dist_grouping_param.resize(nClasses);
  dist_grouping_param.set_constant(-1.0);

  const HmmInitProbType init_type = options.init_type_;
  const HmmAlignProbType align_type = options.align_type_;
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

  const size_t nSentences = source.size();

  std::set<uint> seenIs;

  uint maxI = (align_type == HmmAlignProbReducedpar) ? redpar_limit + 1 : 1;
  for (size_t s = 0; s < nSentences; s++) {
    const uint curI = target[s].size();

    seenIs.insert(curI);
    maxI = std::max(maxI, curI);
  }

  maxI = std::max(maxI, maxAllI);
  uint zero_offset = maxI - 1;

  dist_params.resize(2 * maxI - 1, nClasses, 1.0 / (2 * maxI - 1));     //even for nonpar, we will use this as initialization

  if (align_type == HmmAlignProbReducedpar) {
    dist_grouping_param.set_constant(0.2);
    dist_params.set_constant(0.0);

    const double val = 0.8 / (2 * redpar_limit + 1);
    for (uint c = 0; c < nClasses; c++)
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        dist_params(zero_offset + k, c) = val;
  }
  else
    dist_grouping_param.set_constant(-1.0);

  if (source_fert_prob.size() == 0 || !options.fix_p0_) {
    source_fert_prob.resize(2);        //even for nonpar, we will use this as initialization
    source_fert_prob[0] = 0.02;
    source_fert_prob[1] = 0.98;
  }

  if (init_type == HmmInitPar) {
    init_params.resize(maxI);
    init_params.set_constant(1.0 / maxI);
  }

  align_model.resize_dirty(maxI);       //note: access using I-1
  initial_prob.resize(maxI);

  for (std::set<uint>::const_iterator it = seenIs.begin(); it != seenIs.end(); it++) {
    const uint I = *it;

    //x = new index, y = given index
    align_model[I - 1].resize_dirty(I + 1, I, nClasses);        //because of empty words
    align_model[I - 1].set_constant(1.0 / (I + 1));

    if (align_type != HmmAlignProbNonpar) {
      for (uint c = 0; c < nClasses; c++) {
        for (uint i = 0; i < I; i++) {
          for (uint ii = 0; ii < I; ii++) {
            align_model[I - 1](ii, i, c) = source_fert_prob[1] / I;
          }
          align_model[I - 1](I, i, c) = source_fert_prob[0];
        }
      }
    }

    if (!start_empty_word)
      initial_prob[I - 1].resize_dirty(2 * I);
    else
      initial_prob[I - 1].resize_dirty(I + 1);
    if (init_type != HmmInitPar) {
      if (init_type == HmmInitFix2) {

        const double p1 = (options.fix_p0_) ? source_fert_prob[1] : 0.98;
        const double p0 = (options.fix_p0_) ? source_fert_prob[0] : 0.02;

        initial_prob[I - 1].range_set_constant(p1 / I, 0, I);
        initial_prob[I - 1].range_set_constant(p0 / (initial_prob[I - 1].size() - I), I, initial_prob[I - 1].size() - I);
      }
      else
        initial_prob[I - 1].set_constant(1.0 / initial_prob[I - 1].size());
    }
    else {
      for (uint i = 0; i < I; i++)
        initial_prob[I - 1][i] = source_fert_prob[1] / I;
      if (!start_empty_word) {
        for (uint i = I; i < 2 * I; i++)
          initial_prob[I - 1][i] = source_fert_prob[0] / I;
      }
      else
        initial_prob[I - 1][I] = source_fert_prob[0];
    }
  }

  if (transfer_mode != TransferNo) {

    const double ibm1_p0 = options.ibm1_p0_;

    if (align_type == HmmAlignProbReducedpar)
      dist_grouping_param.set_constant(0.0);
    dist_params.set_constant(0.0);

    for (size_t s = 0; s < nSentences; s++) {

      const Math1D::Vector<uint>& cur_source = source[s];
      const Math1D::Vector<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      if (transfer_mode == TransferViterbi) {

        Storage1D<AlignBaseType> viterbi_alignment(source[s].size(), 0);

        if (options.ibm2_alignment_model_.size() > curI && options.ibm2_alignment_model_[curI].size() > 0)
          compute_ibm2_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, options.ibm2_alignment_model_[curI], options.ibm2_sclass_, viterbi_alignment);
        else if (ibm1_p0 >= 0.0 && ibm1_p0 < 1.0)
          compute_ibm1p0_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, ibm1_p0, viterbi_alignment);
        else
          compute_ibm1_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, viterbi_alignment);

        for (uint j = 1; j < source[s].size(); j++) {

          int prev_aj = viterbi_alignment[j - 1];
          int cur_aj = viterbi_alignment[j];

          if (prev_aj != 0 && cur_aj != 0) {
            int diff = cur_aj - prev_aj;

            if (abs(diff) <= redpar_limit || align_type != HmmAlignProbReducedpar)
              dist_params(zero_offset + diff, target_class[cur_target[prev_aj - 1]]) += 1.0;
            else
              dist_grouping_param[target_class[cur_target[prev_aj - 1]]] += 1.0;
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
                  dist_params(zero_offset + diff, target_class[cur_target[i1]]) += marg;
                else
                  dist_grouping_param[target_class[cur_target[i1]]] += marg;
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
                  dist_params(zero_offset + diff, target_class[cur_target[i1]]) +=  marg;
                else
                  dist_grouping_param[target_class[cur_target[i1]]] += marg;
              }
            }
          }
        }
      }
    }

    for (uint c = 0; c < nClasses; c++) {

      double sum = 0.0;
      for (uint i = 0; i < dist_params.xDim(); i++)
        sum += dist_params(i, c);
      if (align_type == HmmAlignProbReducedpar)
        sum += dist_grouping_param[c];

      //std::cerr << "c: " << c << ", sum: " << sum << std::endl;

      if (sum > 1e-300) {

	    for (uint i = 0; i < dist_params.xDim(); i++)
          dist_params(i, c) *= 1.0 / sum;

        if (align_type == HmmAlignProbReducedpar) {
          dist_grouping_param[c] *= 1.0 / sum;

          for (int k = -redpar_limit; k <= redpar_limit; k++) {
            dist_params(zero_offset + k, c) = 0.75 * dist_params(zero_offset + k, c) + 0.25 * 0.8 / (2 * redpar_limit + 1);
			assert(dist_params(zero_offset + k, c) >= hmm_min_param_entry);
		  }
          dist_grouping_param[c] = 0.75 * dist_grouping_param[c] + 0.25 * 0.2;
          assert(dist_grouping_param[c] >= hmm_min_param_entry);
		}
        else {
          for (uint k = 0; k < dist_params.size(); k++) {
            dist_params(k, c) = 0.75 * dist_params(k, c) + 0.25 / dist_params.xDim();
          }
        }
      }
      else {

		if (align_type == HmmAlignProbReducedpar) {
          dist_grouping_param[c] = 0.2;
          for (uint k = 0; k < dist_params.xDim(); k++)
            dist_params(k, c) = 0.0;

          const double val = 0.8 / (2 * redpar_limit + 1);
          for (int k = -redpar_limit; k <= redpar_limit; k++)
            dist_params(zero_offset + k, c) = val;
		}
		else {
		  dist_params.set_constant(1.0 / dist_params.xDim());
		}
	  }
    }
  }

  HmmAlignProbType align_mode = align_type;
  if (align_mode == HmmAlignProbNonpar || align_mode == HmmAlignProbNonpar2)
    align_mode = HmmAlignProbFullpar;
  par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert_prob, align_mode, options.deficient_, redpar_limit, align_model);

  if (init_type != HmmInitNonpar)
    par2nonpar_hmm_init_model(init_params, source_fert_prob, init_type, initial_prob, start_empty_word, options.fix_p0_);

  //std::cerr << "leaving init" << std::endl;
}

double extended_hmm_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                               const Storage1D<Math1D::Vector<uint> >& target, const Storage1D<WordClassType>& target_class,
                               const FullHMMAlignmentModelSingleClass& align_model, const InitialAlignmentProbability& initial_prob,
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

    const Math1D::Vector<uint>& cur_source = source[s];
    const Math1D::Vector<uint>& cur_target = target[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    const uint curI = cur_target.size();

    //const Math3D::Tensor<double>& cur_align_model = align_model[curI - 1];

    Math1D::Vector<uint> tclass(curI);
    for (uint i = 0; i < curI; i++)
      tclass[i] = target_class[cur_target[i]];

    Math2D::Matrix<double> cur_align_model(curI+1, curI);
    par2nonpar_hmm_alignment_model(tclass, align_model, options, cur_align_model);

    if (start_empty_word)
      sum -= calculate_sehmmc_forward_log_sum(cur_source, cur_target, cur_lookup, tclass, dict, cur_align_model, initial_prob[curI - 1]);
    else
      sum -= calculate_hmmc_forward_log_sum(cur_source, cur_target, cur_lookup, tclass, dict, cur_align_model, initial_prob[curI - 1]);
  }

  return sum / nSentences;
}

double extended_hmm_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                           const Storage1D<WordClassType>& tclass, const FullHMMAlignmentModelSingleClass& align_model,
                           const InitialAlignmentProbability& initial_prob, const SingleWordDictionary& dict,
                           const CooccuringWordsType& wcooc, uint nSourceWords, const floatSingleWordDictionary& prior_weight,
                           const HmmOptions& options, const double dict_weight_sum)
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

  energy += extended_hmm_perplexity(source, slookup, target, tclass, align_model, initial_prob, dict, wcooc, nSourceWords, options);

  return energy;
}

void ehmm_m_step(const FullHMMAlignmentModelSingleClass& facount, Math2D::Matrix<double>& dist_params, uint zero_offset,
                 uint nIter, Math1D::Vector<double>& grouping_param, bool deficient, int redpar_limit, double gd_stepsize, bool quiet)
{
  const uint nClasses = dist_params.yDim();

  bool norm_constraint = true;

  for (uint c = 0; c < nClasses; c++) {

    std::cerr << "class " << c << std::endl;

    Math1D::Vector<double> singleton_count(dist_params.xDim(), 0.0);
    double grouping_count = 0.0;
    Math2D::Matrix<double> span_count(zero_offset + 1, dist_params.xDim() - zero_offset, 0.0);

    for (int I = 1; I <= int (facount.size()); I++) {

      if (facount[I - 1].xDim() != 0) {

        for (int i = 0; i < I; i++) {

          double count_sum = 0.0;
          for (int i_next = 0; i_next < I; i_next++) {

            double cur_count = facount[I - 1](i_next, i, c);
            if (grouping_param[c] < 0.0 || abs(i_next - i) <= redpar_limit) {
              singleton_count[zero_offset + i_next - i] += cur_count;
            }
            else {
              //don't divide by grouping norm, the deficient problem doesn't need it:
              //  due to log laws we get additive constants
              grouping_count += cur_count;
            }
            count_sum += cur_count;
          }
          span_count(zero_offset - i, I - 1 - i) += count_sum;
        }
      }
    }

    if (singleton_count.sum() < 1e-250)
      continue;

    std::cerr.precision(8);

    //std::cerr << "init params before projection: " << dist_params << std::endl;
    //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

    const uint start_idx = (grouping_param[c] < 0.0) ? 0 : zero_offset - redpar_limit;
    const uint end_idx = (grouping_param[c] < 0.0) ? dist_params.xDim() - 1 : zero_offset + redpar_limit;

    Math1D::Vector<double> cur_dist_params(dist_params.xDim());
    dist_params.get_row(c, cur_dist_params);

    if (grouping_param[c] < 0.0) {
      projection_on_simplex(cur_dist_params, hmm_min_param_entry);
    }
    else {
      projection_on_simplex_with_slack(cur_dist_params.direct_access() + zero_offset - redpar_limit, grouping_param[c], 2 * redpar_limit + 1, hmm_min_param_entry);
      grouping_param[c] = std::max(hmm_min_param_entry, grouping_param[c]);
    }

    Math1D::Vector<double> dist_grad(dist_params.xDim());
    Math1D::Vector<double> new_dist_params(dist_params.xDim());
    Math1D::Vector<double> hyp_dist_params(dist_params.xDim());

    double grouping_grad = 0.0;
    double new_grouping_param = grouping_param[c];
    double hyp_grouping_param = grouping_param[c];

    //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

    double energy = (deficient) ? 0.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, cur_dist_params,
                    zero_offset, grouping_param[c], redpar_limit);

    //test if normalized counts give a better starting point
    {
      Math1D::Vector<double> hyp_dist_param(singleton_count.size());

      if (grouping_param[c] >= 0.0) {
        //reduced parametric

        double norm = 0.0;
        for (int k = -redpar_limit; k <= redpar_limit; k++)
          norm += singleton_count[zero_offset + k];
        norm += grouping_count;

        if (norm > 1e-305) {

          for (uint k = 0; k < hyp_dist_param.size(); k++)
            hyp_dist_param[k] = std::max(hmm_min_param_entry, singleton_count[k] / norm);

          double hyp_grouping_param = std::max(hmm_min_param_entry, grouping_count / norm);

          double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                              hyp_dist_param, zero_offset, hyp_grouping_param, redpar_limit);

          if (!deficient && !quiet) {
            std::cerr << "start energy: " << energy << std::endl;
            std::cerr << "hyp energy: " << hyp_energy << std::endl;
          }

          if (hyp_energy < energy) {

            cur_dist_params = hyp_dist_param;
            grouping_param[c] = hyp_grouping_param;
            energy = hyp_energy;
          }
        }
      }
      else {
        //fully parametric

        double sum = singleton_count.sum();

        if (sum > 1e-305) {
          for (uint k = 0; k < hyp_dist_param.size(); k++)
            hyp_dist_param[k] = std::max(hmm_min_param_entry, singleton_count[k] / sum);

          double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                              hyp_dist_param, zero_offset, grouping_param[c], redpar_limit);

          if (!deficient) {
            std::cerr << "start energy: " << energy << std::endl;
            std::cerr << "hyp energy: " << hyp_energy << std::endl;
          }

          if (hyp_energy < energy) {
            cur_dist_params = hyp_dist_param;
            energy = hyp_energy;
          }
        }
      }
    }

    if (deficient) {

      dist_params.set_row(c, cur_dist_params);
      continue;
    }

    // 3. main loop
    if (!quiet)
      std::cerr << "start m-energy: " << energy << std::endl;

    assert(grouping_param[c] < 0.0 || grouping_param[c] >= hmm_min_param_entry);

    //double alpha  = 0.0001;
    //double alpha  = 0.001;
    double alpha = gd_stepsize;

    double line_reduction_factor = 0.5;

    for (uint iter = 1; iter <= nIter; iter++) {

      if (!quiet && (iter % 5) == 0)
        std::cerr << "m-step gd-iter #" << iter << ", cur energy: " << energy << std::endl;

      dist_grad.set_constant(0.0);
      grouping_grad = 0.0;

      // a) calculate gradient
      if (grouping_param[c] < 0.0) {
        //fully parametric

        //singleton terms
        for (uint d = 0; d < singleton_count.size(); d++)
          dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, cur_dist_params[d]);

        //normalization terms
        for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

          Math1D::Vector<double> addon(cur_dist_params.size(), 0.0);

          Math1D::Vector<double> init_sum(zero_offset + 1);
          init_sum[zero_offset] = 0.0;
          init_sum[zero_offset - 1] = cur_dist_params[zero_offset - 1];
          for (int s = zero_offset - 2; s >= 0; s--)
            init_sum[s] = init_sum[s + 1] + cur_dist_params[s];

          double param_sum = init_sum[span_start];      //cur_dist_params.range_sum(span_start,zero_offset);
          for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

            param_sum += std::max(hmm_min_param_entry, cur_dist_params[span_end]);

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
          dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, cur_dist_params[d]);

        //NOTE: because we do not divide grouping_param by the various numbers of affected positions
        //   this energy will differ from the inefficient version by a constant
        grouping_grad -= grouping_count / std::max(hmm_min_param_entry, grouping_param[c]);

        //normalization terms
        for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

          for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

            //there should be plenty of room for speed-ups here

            const double cur_count = span_count(span_start, span_end - zero_offset);
            if (cur_count != 0.0) {

              double param_sum = 0.0;
              if (span_start < first_diff || span_end > last_diff)
                param_sum = grouping_param[c];

              for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
                param_sum += std::max(hmm_min_param_entry, cur_dist_params[d]);

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
      double sqr_grad_norm  = dist_grad.sqr_norm();
      sqr_grad_norm += grouping_grad * grouping_grad;

      real_alpha /= sqrt(sqr_grad_norm);
      if (!quiet)
        std::cerr << "stepsize: " << real_alpha << ", sqr_norm " << sqr_grad_norm << std::endl;
      if (sqr_grad_norm < 1e-5) {
        if (!quiet)
          std::cerr << "CUTOFF after " << iter << " iterations because of squared gradient norm near 0: " << sqr_grad_norm << std::endl;
        return;
      }
      //END_TRIAL

      for (uint k = start_idx; k <= end_idx; k++)
        new_dist_params.direct_access(k) = cur_dist_params.direct_access(k) - real_alpha * dist_grad.direct_access(k);
      //Math1D::go_in_neg_direction(new_dist_params, cur_dist_params, dist_grad, real_alpha);

      new_grouping_param = grouping_param[c] - real_alpha * grouping_grad;

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

        if (grouping_param[c] < 0.0) {
          projection_on_simplex(new_dist_params, hmm_min_param_entry);
        }
        else {
          projection_on_simplex_with_slack(new_dist_params.direct_access() + start_idx, new_grouping_param, 2 * redpar_limit + 1, hmm_min_param_entry);
          new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
        }
      }
      else {

        //projection on the positive orthant, followed by re-normalization (allowed because the energy is scale invariant)

        double sum = 0.0;
        for (uint k = start_idx; k <= end_idx; k++) {
          new_dist_params[k] = std::max(hmm_min_param_entry, new_dist_params[k]);
          sum += new_dist_params[k];
        }

        assert(sum > 1e-12);
        for (uint k = start_idx; k <= end_idx; k++)
          new_dist_params[k] = std::max(hmm_min_param_entry, new_dist_params[k] / sum);
      }

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
          hyp_dist_params.direct_access(k) = std::max(hmm_min_param_entry, lambda * new_dist_params.direct_access(k) +
                                             neg_lambda * cur_dist_params.direct_access(k));
        //Math1D::assign_weighted_combination(hyp_dist_params, lambda, new_dist_params, neg_lambda, cur_dist_params);

        if (grouping_param[c] >= 0.0)
          hyp_grouping_param = std::max(hmm_min_param_entry, lambda * new_grouping_param + neg_lambda * grouping_param[c]);

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

        if (!quiet)
          std::cerr << "hyp energy: " << new_energy << std::endl;
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
        if (!quiet)
          std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
        break;
      }

      energy = best_energy;

      //set new values
      double neg_best_lambda = 1.0 - best_lambda;

      for (uint k = start_idx; k <= end_idx; k++)
        cur_dist_params.direct_access(k) = std::max(hmm_min_param_entry, neg_best_lambda * cur_dist_params.direct_access(k) +
                                           best_lambda * new_dist_params.direct_access(k));
      //Math1D::assign_weighted_combination(cur_dist_params, best_lambda, new_dist_params, neg_best_lambda, cur_dist_params);

      if (grouping_param[c] >= 0.0)
        grouping_param[c] = std::max(hmm_min_param_entry, best_lambda * new_grouping_param + neg_best_lambda * grouping_param[c]);

      //std::cerr << "updated params: " << dist_params << std::endl;
      //std::cerr << "updated grouping param: " << grouping_param << std::endl;
    }

    dist_params.set_row(c, cur_dist_params);
  }
}

void par2nonpar_hmm_alignment_model(const Math2D::Matrix<double>& dist_params, const uint zero_offset,
                                    const Math1D::Vector<double>& dist_grouping_param, const Math1D::Vector<double>& source_fert_prob,
                                    HmmAlignProbType align_type, bool deficient, int redpar_limit,
                                    FullHMMAlignmentModelSingleClass& align_model)
{
  const uint nClasses = dist_params.yDim();
  assert(nClasses == dist_grouping_param.size());

  for (uint I = 1; I <= align_model.size(); I++) {

    Math3D::Tensor<double>& cur_align_model = align_model[I - 1];

    if (cur_align_model.size() > 0) {

      assert(cur_align_model.zDim() == nClasses);

      for (uint c = 0; c < nClasses; c++) {

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

          if (non_zero_sum > 1e-305) {
            const double inv_sum = (deficient) ? 1.0 : 1.0 / non_zero_sum;

            for (int ii = 0; ii < (int)I; ii++) {
              if (align_type == HmmAlignProbReducedpar && abs(ii - i) > redpar_limit) {
                assert(!isnan(grouping_norm));
                assert(grouping_norm > 0.0);
                cur_align_model(ii, i, c) = std::max(hmm_min_param_entry, source_fert_prob[1] * inv_sum * dist_grouping_param[c] / grouping_norm);
              }
              else {
                assert(dist_params(zero_offset + ii - i, c) >= 0);
                cur_align_model(ii, i, c) = std::max(hmm_min_param_entry, source_fert_prob[1] * inv_sum * dist_params(zero_offset + ii - i, c));
              }
              assert(!isnan(cur_align_model(ii, i, c)));
              assert(cur_align_model(ii, i, c) >= 0.0);
            }
            cur_align_model(I, i, c) = source_fert_prob[0];
            assert(!isnan(align_model[I - 1](I, i, c)));
            assert(align_model[I - 1](I, i, c) >= 0.0);

            const double sum = cur_align_model.sum_x(i, c);
            if (!deficient)
              assert(sum >= 0.99 && sum <= 1.01);
          }
        }
      }
    }
  }
}


void par2nonpar_hmm_alignment_model(const Math2D::Matrix<double>& dist_params, const uint zero_offset,
                                    const Math1D::Vector<double>& dist_grouping_param, const Math1D::Vector<double>& source_fert_prob,
                                    const Math1D::Vector<uint>& tclass, const HmmOptions& options, Math2D::Matrix<double>& cur_align_model)
{
  const int I = tclass.size();
  const int redpar_limit = options.redpar_limit_;
  const bool deficient = options.deficient_;
  cur_align_model.resize_dirty(I+1, I);

  for (int i = 0; i < I; i++) {

    const int c = tclass[i];

    double grouping_norm = std::max(0, i - redpar_limit);
    grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

    double non_zero_sum = 0.0;
    for (int ii = 0; ii < (int)I; ii++) {
      if (options.align_type_ != HmmAlignProbReducedpar || abs(ii - i) <= redpar_limit)
        non_zero_sum += dist_params(zero_offset + ii - i, c);
    }

    if (options.align_type_ == HmmAlignProbReducedpar && grouping_norm > 0.0) {
      non_zero_sum += dist_grouping_param[c];
    }

    if (non_zero_sum > 1e-305) {
      const double inv_sum = (deficient) ? 1.0 : 1.0 / non_zero_sum;

      for (int ii = 0; ii < (int)I; ii++) {
        if (options.align_type_ == HmmAlignProbReducedpar && abs(ii - i) > redpar_limit) {
          assert(!isnan(grouping_norm));
          assert(grouping_norm > 0.0);
          cur_align_model(ii, i) = std::max(hmm_min_param_entry, source_fert_prob[1] * inv_sum * dist_grouping_param[c] / grouping_norm);
        }
        else {
          assert(dist_params(zero_offset + ii - i, c) >= 0);
          cur_align_model(ii, i) = std::max(hmm_min_param_entry, source_fert_prob[1] * inv_sum * dist_params(zero_offset + ii - i, c));
        }
        assert(!isnan(cur_align_model(ii, i)));
        assert(cur_align_model(ii, i) >= 0.0);
      }
      cur_align_model(I, i) = source_fert_prob[0];
      assert(!isnan(cur_align_model(I, i)));
      assert(cur_align_model(I, i) >= 0.0);

      const double sum = cur_align_model.row_sum(i);
      if (!deficient)
        assert(sum >= 0.99 && sum <= 1.01);
    }
  }
}

void par2nonpar_hmm_alignment_model(const Math1D::Vector<uint>& tclass, const FullHMMAlignmentModelSingleClass& nonpar_model,
                                    const HmmOptions& options, Math2D::Matrix<double>& align_model)
{
  const int I = tclass.size();

  align_model.resize_dirty(I + 1, I);

  const Math3D::Tensor<double>& ref_align_model = nonpar_model[I - 1];
  assert(ref_align_model.size() > 0);

  for (int i_prev = 0; i_prev < align_model.yDim(); i_prev++)
    for (int i = 0; i < align_model.xDim(); i++)
      align_model(i, i_prev) = ref_align_model(i, i_prev, tclass[i_prev]);
}

void par2nonpar_hmm_alignment_model(const Math1D::Vector<uint>& tclass, const Math3D::Tensor<double>& ref_align_model,
                                    const HmmOptions& options, Math2D::Matrix<double>& align_model)
{
  const int I = tclass.size();

  align_model.resize_dirty(I + 1, I);

  for (int i_prev = 0; i_prev < align_model.yDim(); i_prev++)
    for (int i = 0; i < align_model.xDim(); i++)
      align_model(i, i_prev) = ref_align_model(i, i_prev, tclass[i_prev]);
}

void train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const CooccuringWordsType& wcooc, const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                        Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param, Math1D::Vector<double>& source_fert_prob,
                        InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                        const floatSingleWordDictionary& prior_weight, const HmmOptions& options, uint maxAllI)
{
  std::cerr << "starting Extended HMM-SingleClass EM-training" << std::endl;

  const uint nClasses = target_class.max() + 1;

  uint nIterations = options.nIterations_;
  const HmmInitProbType init_type = options.init_type_;
  const HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;
  const uint start_addon = (start_empty_word) ? 1 : 0;

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

  SingleWordDictionary fwcount(MAKENAME(fwcount));
  fwcount = dict;

  init_hmm_from_prev(source, slookup, target, dict, wcooc, target_class, align_model, dist_params, dist_grouping_param, source_fert_prob,
                     initial_prob, init_params, options, options.transfer_mode_, maxAllI);

  const uint maxI = maxAllI; //align_model.size();
  const uint zero_offset = maxI - 1;

  Math1D::Vector<double> source_fert_count(2);

  InitialAlignmentProbability ficount(maxI, MAKENAME(ficount));
  ficount = initial_prob;

  Math1D::Vector<double> fsentence_start_count(maxI);
  Math1D::Vector<double> fstart_span_count(maxI);

  FullHMMAlignmentModelSingleClass facount(align_model.size(), MAKENAME(facount));
  for (uint I = 1; I <= facount.size(); I++) {
    if (align_model[I - 1].size() > 0)
      facount[I - 1].resize_dirty(I + 1, I, nClasses);
  }

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting EHMM-SingleClass iteration #" << iter << std::endl;

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

      const Math1D::Vector<uint>& cur_source = source[s];
      const Math1D::Vector<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      //std::cerr << "J = " << curJ << ", curI = " << curI << std::endl;

      //const Math3D::Tensor<double>& cur_align_model = align_model[curI - 1];

      Math2D::Matrix<double> cur_align_model(curI+1, curI);
      par2nonpar_hmm_alignment_model(tclass, align_model, options, cur_align_model);

      const Math1D::Vector<double>& cur_init_prob = initial_prob[curI - 1];

      Math3D::Tensor<double>& cur_facount = facount[curI - 1];
      Math1D::Vector<double>& cur_ficount = ficount[curI - 1];

      Math2D::Matrix<double> cur_dict(curJ,curI+1);
      compute_dictmat(cur_source, cur_lookup, cur_target, dict, cur_dict);

      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2 * curI + start_addon, curJ, MAKENAME(forward));
      calculate_hmmc_forward(tclass, cur_dict, cur_align_model, cur_init_prob, options, forward);

      //std::cerr << "forward calculated" << std::endl;

      const uint start_s_idx = cur_source[0];

      const long double sentence_prob = forward.row_sum(curJ - 1);
      assert(forward.min() >= 0.0);

      prev_perplexity -= logl(sentence_prob);

      if (!(sentence_prob > 0.0)) {
        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair " << s << " with I=" << curI << ", J= " << curJ << std::endl;
      }
      assert(sentence_prob > 0.0);

      Math2D::NamedMatrix<long double> backward(2 * curI + start_addon, curJ, MAKENAME(backward));
      calculate_hmmc_backward(tclass, cur_dict, cur_align_model, cur_init_prob, options, backward, true);

      const long double bwd_sentence_prob = backward.row_sum(0);
      assert(backward.min() >= 0.0);

      const long double fwd_bwd_ratio = sentence_prob / bwd_sentence_prob;

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

          const double dict_entry = cur_dict(j,i); //dict[t_idx][cur_lookup(j, i)];

          if (dict_entry > 1e-305) {
            fwcount[t_idx][cur_lookup(j, i)] += forward(i, j) * backward(i, j) * inv_sentence_prob / dict_entry;

            const long double bw = backward(i, j) * inv_sentence_prob;

            for (uint i_prev = 0; i_prev < curI; i_prev++) {
              long double addon = bw * cur_align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + curI, j_prev));
              assert(!isnan(addon));
              cur_facount(i, i_prev, tclass[i_prev]) += addon;
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

          //combining j and j_prev doesn't require to divide by the dict prob
          const long double bw = backward(i, j) * inv_sentence_prob;
          long double addon = bw * cur_align_model(curI, i - curI) * (forward(i, j_prev) + forward(i - curI, j_prev));

          assert(!isnan(addon));

          dict_count_sum += addon;
          cur_facount(curI, i - curI, tclass[i - curI]) += addon;
        }
        fwcount[0][s_idx - 1] += dict_count_sum;

        //start empty word
        if (start_empty_word) {

          //combining j and j_prev doesn't require to divide by the dict prob
          const long double bw = backward(2 * curI, j) * inv_sentence_prob;

          long double addon = bw * forward(2 * curI, j_prev) * cur_init_prob[curI];
          fwcount[0][s_idx - 1] += addon;
          cur_ficount[curI] += addon;
        }
      }
    }   //loop over sentences finished

    //finish counts

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

          for (uint c = 0; c < nClasses; c++) {
            for (int i = 0; i < (int)I; i++) {

              for (int ii = 0; ii < (int)I; ii++) {
                source_fert_count[1] += facount[I - 1](ii, i, c);
              }
              source_fert_count[0] += facount[I - 1](I, i, c);
            }
          }
        }
      }
    }

    double energy = prev_perplexity / nSentences;

    if (dict_weight_sum != 0.0) {
      energy += dict_reg_term(dict, prior_weight, options.l0_beta_);
    }

    std::cerr << "energy after iteration #" << (iter - 1) << ": " << energy << std::endl;
    std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    const double sfsum = source_fert_count.sum();
    if (source_fert_prob.size() > 0 && sfsum > 0.0 && !options.fix_p0_) {

      for (uint i = 0; i < 2; i++) {
        source_fert_prob[i] = source_fert_count[i] / sfsum;
        assert(!isnan(source_fert_prob[i]));
      }

      if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {
        std::cerr << "new probability for zero alignments: " << source_fert_prob[0] << std::endl;
      }
    }

    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {
      //std::cerr << "dist_params: " << dist_params << std::endl;

      //call m-step (no alternatives for grouped HMM implemented so far)
      ehmm_m_step(facount, dist_params, zero_offset, options.align_m_step_iter_,
                  dist_grouping_param, options.deficient_, options.redpar_limit_, options.gd_stepsize_);

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert_prob,
                                     align_type, options.deficient_, redpar_limit, align_model);
    }

    std::cerr << "calling start m step" << std::endl;

    if (init_type == HmmInitPar) {

      if (options.msolve_mode_ == MSSolvePGD)
        start_prob_m_step(fsentence_start_count,fstart_span_count, init_params, options.init_m_step_iter_, options.gd_stepsize_);
      else
        start_prob_m_step_unconstrained(fsentence_start_count, fstart_span_count, init_params, options.init_m_step_iter_);

      par2nonpar_hmm_init_model(init_params, source_fert_prob, init_type, initial_prob, start_empty_word, options.fix_p0_);
    }

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts

    update_dict_from_counts(fwcount, prior_weight, nSentences, dict_weight_sum, options.smoothed_l0_, options.l0_beta_,
                            options.dict_m_step_iter_, dict, hmm_min_dict_entry, options.msolve_mode_ != MSSolvePGD, options.gd_stepsize_);

    //compute new nonparametric probabilities from normalized fractional counts

    for (uint I = 1; I <= maxI; I++) {

      if (align_model[I - 1].xDim() != 0) {

        if (init_type == HmmInitNonpar) {
          double inv_norm = 1.0 / ficount[I - 1].sum();
          for (uint i = 0; i < initial_prob[I - 1].size(); i++)
            initial_prob[I - 1][i] = std::max(hmm_min_param_entry, inv_norm * ficount[I - 1][i]);
        }

        if (align_type == HmmAlignProbNonpar) {

          for (uint c = 0; c < nClasses; c++) {

            for (uint i = 0; i < I; i++) {

              double sum = 0.0;
              for (uint i_next = 0; i_next <= I; i_next++)
                sum += facount[I - 1](i_next, i, c);

              if (sum >= 1e-300) {

                assert(!isnan(sum));
                const double inv_sum = 1.0 / sum;
                assert(!isnan(inv_sum));

                for (uint i_next = 0; i_next <= I; i_next++) {
                  align_model[I - 1](i_next, i, c) = std::max(hmm_min_param_entry, inv_sum * facount[I - 1](i_next, i, c));
                  assert(!isnan(align_model[I - 1](i_next, i, c)));
                }
              }
            }
          }
        }
        else if (align_type == HmmAlignProbNonpar2) {

          for (uint c = 0; c < nClasses; c++) {

            for (uint i = 0; i < I; i++) {

              align_model[I - 1](I, i, c) = source_fert_prob[0];

              double sum = 0.0;
              for (uint i_next = 0; i_next < I; i_next++)
                sum += facount[I - 1](i_next, i, c);

              if (sum >= 1e-300) {

                assert(!isnan(sum));
                const double inv_sum = source_fert_prob[1] / sum;
                assert(!isnan(inv_sum));

                for (uint i_next = 0; i_next < I; i_next++) {
                  align_model[I - 1](i_next, i, c) = std::max(hmm_min_param_entry, inv_sum * facount[I - 1](i_next, i, c));
                  assert(!isnan(align_model[I - 1](i_next, i, c)));
                }
              }
            }
          }
        }
      }
    }

    if (options.print_energy_) {
      std::cerr << "#### EHMM energy after iteration #" << iter << ": "
                << extended_hmm_energy(source, slookup, target, target_class, align_model, initial_prob, dict, wcooc,
                                       nSourceWords, prior_weight, options, dict_weight_sum)
                << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      //std::cerr << "computing error rates" << std::endl;

      uint nContributors = 0;

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double sum_daes = 0.0;

      double sum_marg_aer = 0.0;
      double sum_marg_fmeasure = 0.0;
      double sum_marg_daes = 0.0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
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

        Math1D::Vector<uint> tclass(curI);
        for (uint i = 0; i < curI; i++)
          tclass[i] = target_class[target[s][i]];

        Math2D::Matrix<double> cur_align_model(curI+1, curI);
        par2nonpar_hmm_alignment_model(tclass, align_model, options, cur_align_model);

        //std::cerr << "computing alignment" << std::endl;

        compute_ehmmc_viterbi_alignment(source[s], cur_lookup, target[s], tclass, dict, cur_align_model,
                                        initial_prob[curI - 1], viterbi_alignment, options);

        //std::cerr << "alignment: " << viterbi_alignment << std::endl;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        sum_daes += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        Storage1D<AlignBaseType> marg_alignment;
        compute_ehmmc_optmarginal_alignment(source[s], cur_lookup, target[s], tclass, dict, cur_align_model,
                                            initial_prob[curI - 1], options, marg_alignment);

        //std::cerr << "marg_alignment: " << marg_alignment << std::endl;
        sum_marg_aer += AER(marg_alignment, cur_sure, cur_possible);
        sum_marg_fmeasure += f_measure(marg_alignment, cur_sure, cur_possible);
        sum_marg_daes += nDefiniteAlignmentErrors(marg_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ehmmc_postdec_alignment(source[s], cur_lookup, target[s], tclass, dict, cur_align_model,
                                        initial_prob[curI - 1], options, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_daes /= nContributors;
      sum_fmeasure /= nContributors;

      sum_marg_aer *= 100.0 / nContributors;
      sum_marg_fmeasure /= nContributors;
      sum_marg_daes /= nContributors;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### EHMM-SingleClass Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### EHMM-SingleClass Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM-SingleClass Viterbi-DAE/S after iteration #" << iter << ": " << sum_daes << std::endl;

      std::cerr << "---- EHMM-SingleClass OptMarg-AER after iteration #" << iter << ": " << sum_marg_aer << " %" << std::endl;
      std::cerr << "---- EHMM-SingleClass OptMarg-fmeasure after iteration #" << iter << ": " << sum_marg_fmeasure << std::endl;
      std::cerr << "---- EHMM-SingleClass OptMarg-DAE/S after iteration #" << iter << ": " << sum_marg_daes << std::endl;

      std::cerr << "#### EHMM-SingleClass Postdec-AER after iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### EHMM-SingleClass Postdec-fmeasure after iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### EHMM-SingleClass Postdec-DAE/S after iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  }
}

void train_extended_hmm_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                       const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                       const Math1D::Vector<WordClassType>& target_class,
                                       FullHMMAlignmentModelSingleClass& align_model, Math2D::Matrix<double>& dist_params,
                                       Math1D::Vector<double>& dist_grouping_param, Math1D::Vector<double>& source_fert_prob,
                                       InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                                       SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight, const HmmOptions& options, uint maxAllI)
{
  std::cerr << "starting Extended HMM-Singleclass GD-training" << std::endl;

  uint nIterations = options.nIterations_;
  const HmmInitProbType init_type = options.init_type_;
  const HmmAlignProbType align_type = options.align_type_;
  const bool smoothed_l0 = options.smoothed_l0_;
  const double l0_beta = options.l0_beta_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;

  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dicitionary is assumed to be initialized

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  const uint nSourceWords = options.nSourceWords_;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  SingleLookupTable aux_lookup;

  init_hmm_from_prev(source, slookup, target, dict, wcooc, target_class, align_model, dist_params, dist_grouping_param, source_fert_prob,
                     initial_prob, init_params, options, options.transfer_mode_, maxAllI);

  //std::cerr << "after init" << std::endl;

  const uint maxI = maxAllI; //align_model.size();
  const uint zero_offset = maxI - 1;
  const uint start_addon = (start_empty_word) ? 1 : 0;

  Math1D::Vector<double> dist_grouping_grad = dist_grouping_param;
  Math1D::Vector<double> new_dist_grouping_param = dist_grouping_param;
  Math1D::Vector<double> hyp_dist_grouping_param = dist_grouping_param;

  Math2D::NamedMatrix<double> dist_grad(MAKENAME(dist_grad));
  dist_grad = dist_params;

  Math2D::NamedMatrix<double> new_dist_params(MAKENAME(new_dist_params));
  new_dist_params = dist_params;
  Math2D::NamedMatrix<double> hyp_dist_params(MAKENAME(hyp_dist_params));
  hyp_dist_params = dist_params;

  Math1D::NamedVector<double> init_param_grad(MAKENAME(init_param_grad));
  init_param_grad = init_params;
  Math1D::NamedVector<double> new_init_params(MAKENAME(new_init_params));
  new_init_params = init_params;
  Math1D::NamedVector<double> hyp_init_params(MAKENAME(hyp_init_params));
  hyp_init_params = init_params;

  Math1D::Vector<double> source_fert_grad = source_fert_prob;
  Math1D::Vector<double> new_source_fert = source_fert_prob;
  Math1D::Vector<double> hyp_source_fert = source_fert_prob;

  InitialAlignmentProbability init_grad(maxI, MAKENAME(init_grad));

  Storage1D<Math1D::Vector<double> > dict_grad(options.nTargetWords_);
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_grad[i].resize(dict[i].size());
  }

  FullHMMAlignmentModelSingleClass align_grad(maxI, MAKENAME(align_grad));
  for (uint I = 1; I <= maxI; I++) {
    if (align_model[I - 1].size() > 0) {
      align_grad[I - 1].resize_dirty(I + 1, I, align_model[I - 1].zDim());
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

  FullHMMAlignmentModelSingleClass new_align_prob(maxI, MAKENAME(new_align_prob));
  FullHMMAlignmentModelSingleClass hyp_align_prob(maxI, MAKENAME(hyp_align_prob));

  for (uint I = 1; I <= maxI; I++) {
    if (align_grad[I - 1].size() > 0) {
      new_align_prob[I - 1].resize_dirty(I + 1, I, align_grad[I - 1].zDim());
      hyp_align_prob[I - 1].resize_dirty(I + 1, I, align_grad[I - 1].zDim());
    }
  }

  Math1D::Vector<double> slack_vector(options.nTargetWords_, 0.0);
  Math1D::Vector<double> new_slack_vector(options.nTargetWords_, 0.0);

  for (uint i = 0; i < options.nTargetWords_; i++) {
    double slack_val = 1.0 - dict[i].sum();
    slack_vector[i] = slack_val;
    new_slack_vector[i] = slack_val;
  }

  double energy = extended_hmm_energy(source, slookup, target, target_class, align_model,
                                      initial_prob, dict, wcooc, nSourceWords, prior_weight, options, dict_weight_sum);

  double line_reduction_factor = 0.5;

  uint nSuccessiveReductions = 0;

  std::cerr << "start energy: " << energy << std::endl;

  //double alpha = 0.005; //0.002;
  //double alpha = 0.001; //(align_type == HmmAlignProbReducedpar) ? 0.0025 : 0.001;
  double alpha = options.gd_stepsize_;

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting EHMM-Singleclass gd-iter #" << iter << std::endl;
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

    dist_grouping_grad.set_constant(0.0);

    /******** 1. calculate gradients **********/

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      //const Math3D::Tensor<double>& cur_align_model = align_model[curI - 1];
      const Math1D::Vector<double>& cur_init_prob = initial_prob[curI - 1];

      Math2D::Matrix<double> cur_align_model(curI+1, curI);
      par2nonpar_hmm_alignment_model(tclass, align_model, options, cur_align_model);

      Math2D::Matrix<double> cur_dict(curJ,curI+1);
      compute_dictmat(cur_source, cur_lookup, cur_target, dict, cur_dict);

      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2 * curI + start_addon, curJ, MAKENAME(forward));
      Math2D::NamedMatrix<long double> backward(2 * curI + start_addon, curJ, MAKENAME(backward));

      calculate_hmmc_forward(tclass, cur_dict, cur_align_model, cur_init_prob, options, forward);

      const uint start_s_idx = cur_source[0];

      const long double sentence_prob = forward.row_sum(curJ - 1);
      assert(forward.min() >= 0.0);

      if (!(sentence_prob > 0.0)) {

        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair "
                  << s << " with I=" << curI << ", J= " << curJ << std::endl;

      }
      assert(sentence_prob > 0.0);

      calculate_hmmc_backward(tclass, cur_dict, cur_align_model, cur_init_prob, options, backward, true);
      assert(backward.min() >= 0.0);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      Math3D::Tensor<double>& cur_align_grad = align_grad[curI - 1];
      Math1D::Vector<double>& cur_init_grad = init_grad[curI - 1];

      /**** update gradients ****/

      //start of sentence
      for (uint i = 0; i < curI; i++) {
        const uint t_idx = cur_target[i];

        const double coeff = inv_sentence_prob * backward(i, 0);

        const double cur_dict_entry = cur_dict(0,i); //dict[t_idx][cur_lookup(0, i)];

        if (cur_dict_entry > 1e-300) {

          double addon = coeff / cur_dict_entry;
          dict_grad[t_idx][cur_lookup(0, i)] -= addon;
        }

        if (cur_init_prob[i] > 1e-300) {
          cur_init_grad[i] -= coeff; //division by prob is deferred to after the loop
          assert(!isnan(cur_init_grad[i]));
        }
      }

      const double cur_dict_entry = cur_dict(0,curI); //dict[0][start_s_idx - 1];
      if (cur_dict_entry > 1e-300) {
        if (start_empty_word) {
          const long double coeff = inv_sentence_prob * backward(2 * curI, 0);

          dict_grad[0][start_s_idx - 1] -= coeff / std::max(hmm_min_param_entry, cur_dict_entry);
          cur_init_grad[curI] -= coeff; //division by prob is deferred to after the loop
        }
        else {
          for (uint i = 0; i < curI; i++) {

            const long double coeff = inv_sentence_prob * backward(i + curI, 0);

            dict_grad[0][start_s_idx - 1] -= coeff / std::max(hmm_min_dict_entry, cur_dict_entry);

            if (initial_prob[curI - 1][i + curI] > 1e-300) {
              cur_init_grad[i + curI] -= coeff; //division by prob is deferred to after the loop
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

            double dict_addon = forward(i, j) * backward(i, j)
                                / (sentence_prob * std::max(1e-30, cur_dict_entry * cur_dict_entry));

            dict_grad[t_idx][cur_lookup(j, i)] -= dict_addon;

            const long double bw = backward(i, j) / sentence_prob;

            uint i_prev;
            double addon;

            for (i_prev = 0; i_prev < curI; i_prev++) {

              addon = bw * (forward(i_prev, j_prev) + forward(i_prev + curI, j_prev));
              cur_align_grad(i, i_prev, tclass[i_prev]) -= addon; //division by prob is deferred
            }
          }
        }

        //empty words
        const double cur_dict_entry = cur_dict(j,curI); //dict[0][s_idx - 1];
        const double cur_dict_denom = std::max(1e-30, cur_dict_entry * cur_dict_entry);

        if (cur_dict_entry > 1e-70) {

          for (uint i = curI; i < 2 * curI; i++) {

            const long double bw = backward(i, j) * inv_sentence_prob;

            double addon = bw * forward(i, j) / cur_dict_denom;
            dict_grad[0][s_idx - 1] -= addon;

            const long double align_addon = bw * (forward(i, j_prev) + forward(i - curI, j_prev));
            cur_align_grad(curI, i - curI, tclass[i - curI]) -= align_addon; //division by prob is deferred
          }

          if (start_empty_word) {

            const long double bw = backward(2 * curI, j) * inv_sentence_prob;

            dict_grad[0][s_idx - 1] -= bw * forward(2 * curI, j) / std::max(1e-30, cur_dict_entry * cur_dict_entry);
            cur_init_grad[curI] -= bw * forward(2 * curI, j_prev); //division by prob is deferred to after the loop

            for (uint i = 0; i < curI; i++) {
              double addon = backward(i, j) * forward(2 * curI, j_prev) * inv_sentence_prob;
              cur_init_grad[curI] -= addon; //division by prob is deferred to after the loop
            }
          }
        }
      }
    }  //loop over sentences finished

    //std::cerr << "loop over sentences finished" << std::endl;

    for (uint i = 0; i < options.nTargetWords_; i++) {
      dict_grad[i] *= 1.0 / nSentences;
    }
    for (uint I = 1; I <= maxI; I++) {
      init_grad[I - 1] *= 1.0 / nSentences;

      //finally include divisions
      for (uint k=0; k < init_grad[I - 1].size(); k++)
        init_grad[I - 1].direct_access(k) /= initial_prob[I - 1].direct_access(k);
      assert(!isnan(init_grad[I - 1].sum()));
	}
    for (uint I = 1; I <= maxI; I++) {
      align_grad[I - 1] *= 1.0 / nSentences;

      //finally include division
      for (uint k=0; k < align_grad[I - 1].size(); k++) {
        align_grad[I - 1].direct_access(k) /= std::max(hmm_min_param_entry, align_model[I - 1].direct_access(k));
      }
	  assert(!isnan(align_grad[I - 1].sum()));
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
            source_fert_grad[1] += init_grad[I - 1][i] * (initial_prob[I - 1][i] / source_fert_prob[1]);

          if (!start_empty_word)
            source_fert_grad[0] += init_grad[I - 1].range_sum(I, 2 * I) / I;
          else
            source_fert_grad[0] += init_grad[I - 1][I];

          const double sum = init_params.range_sum(0, I);

          for (uint i = 0; i < I; i++) {

            const double cur_grad =
              init_grad[I - 1][i] * source_fert_prob[1] / (sum * sum);

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

	  assert(!isnan(dist_params.sum()));

      for (uint I = 1; I <= maxI; I++) {
		  
		//std::cerr << "I: " << I << std::endl;

        if (align_grad[I - 1].size() > 0) {

          for (uint c = 0; c < align_grad[I - 1].zDim(); c++) {
            for (int i = 0; i < (int)I; i++) {

              for (uint ii = 0; ii < I; ii++)
                source_fert_grad[1] += align_grad[I - 1](ii, i, c) * (align_model[I - 1](ii, i, c) / source_fert_prob[1]);
              source_fert_grad[0] += align_grad[I - 1](I, i, c) * (align_model[I - 1](I, i, c) / source_fert_prob[0]);

              double non_zero_sum = 0.0;

              if (align_type == HmmAlignProbFullpar) {

                if (options.deficient_) {
                  for (uint ii = 0; ii < I; ii++)
                    dist_grad(zero_offset + ii - i, c) += source_fert_prob[1] * align_grad[I - 1](ii, i, c);
                }
                else {

                  for (uint ii = 0; ii < I; ii++) {
                    non_zero_sum += dist_params(zero_offset + ii - i, c);
                  }

                  //std::cerr << "I: " << I << ", i: " << i << ", non_zero_sum: " << non_zero_sum << std::endl;

                  const double factor = source_fert_prob[1] / (non_zero_sum * non_zero_sum);

                  assert(!isnan(factor));

                  for (uint ii = 0; ii < I; ii++) {
                    //NOTE: align_grad has already a negative sign

                    const double cur_grad = align_grad[I - 1] (ii, i, c);

                    dist_grad(zero_offset + ii - i, c) += cur_grad * factor * non_zero_sum;
                    for (uint iii = 0; iii < I; iii++)
                      dist_grad(zero_offset + iii - i, c) -= cur_grad * factor * dist_params(zero_offset + ii - i, c);
                  }
                }
              }
              else {

                double grouping_norm = std::max(0, i - redpar_limit);
                grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

				// double grouping_norm = 0.0;
				// for (int ii = 0; ii < (int) I; ii++) {
				  // if (abs(ii-i) > redpar_limit)
					// grouping_norm += 1.0;
				// }

                if (options.deficient_) {
                  for (int ii = 0; ii < (int)I; ii++) {
                    //NOTE: align_grad has already a negative sign
                    if (abs(ii - i) <= redpar_limit) {
                      dist_grad(zero_offset + ii - i, c) += source_fert_prob[1] * align_grad[I - 1] (ii, i, c);
                    }
                    else {
                      dist_grouping_grad[c] += source_fert_prob[1] * align_grad[I - 1] (ii, i, c) / grouping_norm;
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

				  //std::cerr << "source_fert_prob: " << source_fert_prob << std::endl;
				  //std::cerr << "non_zero_sum: " << non_zero_sum << std::endl;
                  const double factor = source_fert_prob[1] / (non_zero_sum * non_zero_sum);

                  assert(!isnan(factor));

                  for (int ii = 0; ii < (int)I; ii++) {

                    const double cur_grad = align_grad[I - 1](ii, i, c);
					assert(!isnan(cur_grad));

                    //NOTE: align_grad has already a negative sign
                    if (abs(ii - i) <= redpar_limit) {

                      dist_grad(zero_offset + ii - i, c) += cur_grad * factor * non_zero_sum;
                      for (int iii = 0; iii < (int)I; iii++) {
                        if (abs(iii - i) <= redpar_limit) {
                          dist_grad(zero_offset + iii - i, c) -= cur_grad * factor * dist_params(zero_offset + ii - i, c);
                          //std::cerr << "cur_grad: " << cur_grad << ", factor: " << factor << ", dist_param: " 
						  //	      << dist_params(zero_offset + ii - i, c) << std::endl;
						  assert(!isnan(dist_grad(zero_offset + iii - i, c)));
						}
						else {
                          dist_grouping_grad[c] -= cur_grad * factor * dist_params(zero_offset + ii - i, c) / grouping_norm;
						  assert(grouping_norm > 1e-300);
						  assert(!isnan(dist_grouping_grad[c]));
						}
					  }
					  //assert(!isnan(dist_grad.sum()));
  					  //assert(!isnan(dist_grouping_grad.sum()));
                    }
                    else {

					  assert(grouping_norm > 1e-300);
                      dist_grouping_grad[c] += cur_grad * factor * (non_zero_sum - dist_grouping_param[c]) / grouping_norm;

                      for (int iii = 0; iii < (int)I; iii++) {
                        if (abs(iii - i) <= redpar_limit) {
                          assert(iii != ii);
                          dist_grad(zero_offset + iii - i, c) -= cur_grad * factor * dist_grouping_param[c] / grouping_norm;
                        }
                      }
					  //assert(!isnan(dist_grad.sum()));
  					  //assert(!isnan(dist_grouping_grad.sum()));
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
    assert(!isnan(sqr_grad_norm));
    if (init_type == HmmInitPar)
      sqr_grad_norm += init_param_grad.sqr_norm();
    assert(!isnan(sqr_grad_norm));
    if (align_type == HmmAlignProbFullpar) {
      sqr_grad_norm += dist_grad.sqr_norm();
    }
    else if (align_type == HmmAlignProbReducedpar) {
      sqr_grad_norm += dist_grad.sqr_norm() + dist_grouping_grad.sqr_norm();
    }
    assert(!isnan(sqr_grad_norm));
    for (uint i = 0; i < options.nTargetWords_; i++)
      sqr_grad_norm += dict_grad[i].sqr_norm();
    assert(!isnan(sqr_grad_norm));

    real_alpha /= sqrt(sqr_grad_norm);

    //std::cerr << "A, sqr_grad_norm: " << sqr_grad_norm << ", real_alpha: " << real_alpha  << std::endl;

    if (!options.fix_p0_) {
      if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {

        for (uint i = 0; i < 2; i++)
          new_source_fert[i] = source_fert_prob[i] - real_alpha * source_fert_grad[i];

        projection_on_simplex(new_source_fert, 1e-12);
      }
    }
    //std::cerr << "B" << std::endl;

    if (init_type == HmmInitPar) {

      //for (uint k = 0; k < init_params.size(); k++)
      //  new_init_params[k] = init_params[k] - real_alpha * init_param_grad[k];
      Math1D::go_in_neg_direction(new_init_params, init_params, init_param_grad, real_alpha);

      projection_on_simplex(new_init_params, hmm_min_param_entry);
    }
    //std::cerr << "C" << std::endl;

    if (align_type == HmmAlignProbFullpar) {

      Math2D::go_in_neg_direction(new_dist_params, dist_params, dist_grad, real_alpha);

      for (uint c = 0; c < dist_params.yDim(); c++) {
        //for (uint k = 0; k < dist_params.xDim(); k++)
        //  new_dist_params(k, c) = dist_params(k, c) - real_alpha * dist_grad(k, c);

        projection_on_simplex(new_dist_params.row_ptr(c), new_dist_params.xDim(), hmm_min_param_entry);
      }
    }
    else if (align_type == HmmAlignProbReducedpar) {

      Math2D::go_in_neg_direction(new_dist_params, dist_params, dist_grad, real_alpha);

      for (uint c = 0; c < dist_params.yDim(); c++) {
        //for (uint k = 0; k < dist_params.xDim(); k++)
        //  new_dist_params(k, c) = dist_params(k, c) - real_alpha * dist_grad(k, c);

        new_dist_grouping_param[c] = dist_grouping_param[c] - real_alpha * dist_grouping_grad[c];

        assert(new_dist_params.size() >= 2 * redpar_limit + 1);

        projection_on_simplex_with_slack(new_dist_params.row_ptr(c) + zero_offset - redpar_limit,
                                         new_dist_grouping_param[c], 2 * redpar_limit + 1, hmm_min_param_entry);
      }
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
          //  new_init_prob[I - 1][k] =  initial_prob[I - 1][k] - real_alpha * init_grad[I - 1][k];

          if (new_init_prob[I - 1].size() > 0)
            Math1D::go_in_neg_direction(new_init_prob[I - 1], initial_prob[I - 1], init_grad[I -1], real_alpha);
        }

        if (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2) {
          //for (uint k = 0; k < align_model[I - 1].size(); k++)
          //  new_align_prob[I - 1].direct_access(k) = align_model[I - 1].direct_access(k) - real_alpha * align_grad[I - 1].direct_access(k);

          if (align_model[I - 1].size() > 0)
            Math3D::go_in_neg_direction(new_align_prob[I - 1], align_model[I - 1], align_grad[I - 1], real_alpha);
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

      projection_on_simplex_with_slack(new_dict_prob[i], new_slack_vector[i], hmm_min_dict_entry);
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

          projection_on_simplex(new_init_prob[I - 1], hmm_min_param_entry);
        }

        if (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2) {

          for (uint k = 0; k < new_align_prob[I - 1].size(); k++) {
            if (new_align_prob[I - 1].direct_access(k) <= -1e75)
              new_align_prob[I - 1].direct_access(k) = -9e74;
            if (new_align_prob[I - 1].direct_access(k) >= 1e75)
              new_align_prob[I - 1].direct_access(k) = 9e74;

            assert(fabs(new_align_prob[I - 1].direct_access(k)) < 1e75);
          }

          const uint dim = (align_type == HmmAlignProbNonpar) ? align_model[I - 1].xDim() : align_model[I - 1].xDim() - 1;

          Math1D::Vector<double> temp(dim);

          for (uint c = 0; c < align_model[I - 1].zDim(); c++) {
            for (uint y = 0; y < align_model[I - 1].yDim(); y++) {

              align_model[I - 1].get_x(y, c, temp);
              projection_on_simplex(temp, hmm_min_param_entry);
              align_model[I - 1].set_x(y, c, temp);
            }
          }
        }
      }
    }

    /******** 4. find a suitable stepsize *********/

    //std::cerr << "G" << std::endl;

    //std::cerr << "previous source fert prob: " << source_fert_prob << std::endl;
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
          hyp_source_fert[i] = lambda * new_source_fert[i] + neg_lambda * source_fert_prob[i];
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

      if (align_type == HmmAlignProbReducedpar) {
        //for (uint c = 0; c < hyp_dist_grouping_param.size(); c++)
        //  hyp_dist_grouping_param[c] = lambda * new_dist_grouping_param[c] + neg_lambda * dist_grouping_param[c];

        Math1D::assign_weighted_combination(hyp_dist_grouping_param, lambda, new_dist_grouping_param, neg_lambda, dist_grouping_param);
      }

      if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {

        //for (uint k = 0; k < dist_params.size(); k++)
        //  hyp_dist_params.direct_access(k) = lambda * new_dist_params.direct_access(k) + neg_lambda * dist_params.direct_access(k);
        Math2D::assign_weighted_combination(hyp_dist_params, lambda, new_dist_params, neg_lambda, dist_params);

        par2nonpar_hmm_alignment_model(hyp_dist_params, zero_offset, hyp_dist_grouping_param, hyp_source_fert,
                                       align_type, options.deficient_, redpar_limit, hyp_align_prob);

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
            Math3D::assign_weighted_combination(hyp_align_prob[I - 1], lambda, new_align_prob[I - 1], neg_lambda, align_model[I - 1]);
        }
      }

      double new_energy = extended_hmm_energy(source, slookup, target, target_class, hyp_align_prob,
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
        source_fert_prob[i] = best_lambda * new_source_fert[i] + neg_best_lambda * source_fert_prob[i];
    }

    if (init_type == HmmInitPar) {

      //for (uint k = 0; k < init_params.size(); k++)
      //  init_params[k] = best_lambda * new_init_params[k] + neg_best_lambda * init_params[k];
      Math1D::assign_weighted_combination(init_params, best_lambda, new_init_params, neg_best_lambda, init_params);

      par2nonpar_hmm_init_model(init_params, source_fert_prob, init_type, initial_prob, start_empty_word, options.fix_p0_);
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
      //  dist_params.direct_access(k) = best_lambda * new_dist_params.direct_access(k) + neg_best_lambda * dist_params.direct_access(k);
      Math2D::assign_weighted_combination(dist_params, best_lambda, new_dist_params, neg_best_lambda, dist_params);

      if (align_type == HmmAlignProbReducedpar) {
        //for (uint c = 0; c < dist_grouping_param.size(); c++)
        //  dist_grouping_param[c] = best_lambda * new_dist_grouping_param[c] + neg_best_lambda * dist_grouping_param[c];

        Math1D::assign_weighted_combination(dist_grouping_param, best_lambda, new_dist_grouping_param, neg_best_lambda, dist_grouping_param);
      }

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert_prob,
                                     align_type, options.deficient_, redpar_limit, align_model);
    }
    else {
      for (uint I = 1; I <= maxI; I++) {
        //for (uint k = 0; k < align_model[I - 1].size(); k++)
        //  align_model[I - 1].direct_access(k) = best_lambda * new_align_prob[I - 1].direct_access(k)
        //                                        + neg_best_lambda * align_model[I - 1].direct_access(k);

        if (align_model[I - 1].size() > 0)
          Math3D::assign_weighted_combination(align_model[I - 1], best_lambda, new_align_prob[I - 1], neg_best_lambda, align_model[I - 1]);
      }
    }

    if (options.print_energy_) {
      std::cerr << "slack-sum: " << slack_vector.sum() << std::endl;
      std::cerr << "#### EHMM energy after gd-iteration #" << iter << ": " << energy << std::endl;
    }


    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      //std::cerr << "computing error rates" << std::endl;

      uint nContributors = 0;

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double sum_daes = 0.0;

      double sum_marg_aer = 0.0;
      double sum_marg_fmeasure = 0.0;
      double sum_marg_daes = 0.0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
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

        Math1D::Vector<uint> tclass(curI);
        for (uint i = 0; i < curI; i++)
          tclass[i] = target_class[target[s][i]];

        Math2D::Matrix<double> cur_align_model(curI+1, curI);
        par2nonpar_hmm_alignment_model(tclass, align_model, options, cur_align_model);


        //std::cerr << "computing alignment" << std::endl;

        compute_ehmmc_viterbi_alignment(source[s], cur_lookup, target[s], tclass, dict, cur_align_model,
                                        initial_prob[curI - 1], viterbi_alignment, options);

        //std::cerr << "alignment: " << viterbi_alignment << std::endl;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        sum_daes += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        Storage1D<AlignBaseType> marg_alignment;
        compute_ehmmc_optmarginal_alignment(source[s], cur_lookup, target[s], tclass, dict, cur_align_model,
                                            initial_prob[curI - 1], options, marg_alignment);

        //std::cerr << "marg_alignment: " << marg_alignment << std::endl;
        sum_marg_aer += AER(marg_alignment, cur_sure, cur_possible);
        sum_marg_fmeasure += f_measure(marg_alignment, cur_sure, cur_possible);
        sum_marg_daes += nDefiniteAlignmentErrors(marg_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ehmmc_postdec_alignment(source[s], cur_lookup, target[s], tclass, dict, cur_align_model,
                                        initial_prob[curI - 1], options, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_daes /= nContributors;
      sum_fmeasure /= nContributors;

      sum_marg_aer *= 100.0 / nContributors;
      sum_marg_fmeasure /= nContributors;
      sum_marg_daes /= nContributors;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### EHMM-SingleClass Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### EHMM-SingleClass Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM-SingleClass Viterbi-DAE/S after iteration #" << iter << ": " << sum_daes << std::endl;

      std::cerr << "---- EHMM-SingleClass OptMarg-AER after iteration #" << iter << ": " << sum_marg_aer << " %" << std::endl;
      std::cerr << "---- EHMM-SingleClass OptMarg-fmeasure after iteration #" << iter << ": " << sum_marg_fmeasure << std::endl;
      std::cerr << "---- EHMM-SingleClass OptMarg-DAE/S after iteration #" << iter << ": " << sum_marg_daes << std::endl;

      std::cerr << "#### EHMM-SingleClass Postdec-AER after iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### EHMM-SingleClass Postdec-fmeasure after iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### EHMM-SingleClass Postdec-DAE/S after iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  }
}

struct NonparHMMCCountStruct  {

  NonparHMMCCountStruct() : cur_(0), last_(0), target_class_(0) {}

  NonparHMMCCountStruct(int cur, int last, int tc) : cur_(cur), last_(last), target_class_(tc) {}

  int cur_;
  int last_;
  int target_class_;
};

bool operator<(const NonparHMMCCountStruct& first, const NonparHMMCCountStruct& second)
{
  if (first.cur_ != second.cur_)
    return (first.cur_ < second.cur_);
  if (first.last_ != second.last_)
    return (first.last_ < second.last_);
  return (first.target_class_ < second.target_class_);
}

bool operator==(const NonparHMMCCountStruct& first, const NonparHMMCCountStruct& second)
{
  return ((first.cur_ == second.cur_) && (first.last_ == second.last_) && (first.target_class_==second.target_class_) );
}


void viterbi_train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                const Math1D::Vector<WordClassType>& target_class, FullHMMAlignmentModelSingleClass& align_model,
                                Math2D::Matrix<double>& dist_params, Math1D::Vector<double>& dist_grouping_param,
                                Math1D::Vector<double>& source_fert_prob, InitialAlignmentProbability& initial_prob,
                                Math1D::Vector<double>& init_params, SingleWordDictionary& dict,
                                const floatSingleWordDictionary& prior_weight, const HmmOptions& options,
                                const Math1D::Vector<double>& xlogx_table, uint maxAllI)
{
  std::cerr << "starting Viterbi Training for Extended HMM-SingleClass" << std::endl;

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

  //std::cerr << "init_type: " << init_type << std::endl;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  init_hmm_from_prev(source, slookup, target, dict, wcooc, target_class, align_model, dist_params, dist_grouping_param, source_fert_prob,
                     initial_prob, init_params, options, options.transfer_mode_, maxAllI);

  const uint maxI = maxAllI; //align_model.size();
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

  FullHMMAlignmentModelSingleClass acount(maxI, MAKENAME(acount));
  for (uint I = 1; I <= maxI; I++) {
    if (align_model[I - 1].size() > 0)
      acount[I - 1].resize_dirty(I + 1, I, align_model[I - 1].zDim());
  }

  Math2D::NamedMatrix<double> dist_count(MAKENAME(dist_count));       //used in deficient mode
  dist_count = dist_params;

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting Viterbi-EHMM-SingleClass iteration #" << iter << std::endl;

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i = 0; i < options.nTargetWords_; i++) {
      dcount[i].set_constant(0);
    }

    for (uint I = 0; I < maxI; I++) {
      acount[I].set_constant(0.0);
      icount[I].set_constant(0.0);
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

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      //const Math3D::Tensor<double>& cur_align_model = align_model[curI - 1];
      Math3D::Tensor<double>& cur_facount = acount[curI - 1];

      Math2D::Matrix<double> cur_align_model(curI+1, curI);
      par2nonpar_hmm_alignment_model(tclass, align_model, options, cur_align_model);

      //std::cerr << " passing initial prob " << initial_prob[curI-1] << std::endl;

      long double prob = compute_ehmmc_viterbi_alignment(cur_source, cur_lookup, cur_target, tclass, dict, cur_align_model,
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
              cur_facount(curI, prev_aj - curI, tclass[prev_aj - curI]) += 1.0;
            }
            else {
              cur_facount(aj, prev_aj - curI, tclass[prev_aj - curI]) += 1.0;

              if (deficient_parametric) {

                uint diff = zero_offset - (prev_aj - curI);
                diff += (aj >= curI) ? aj - curI : aj;

                dist_count(diff, tclass[prev_aj - curI]) += 1.0;
              }
            }
          }
          else { //prev_aj is target pos
            if (aj >= curI) {
              cur_facount(curI, prev_aj, tclass[prev_aj]) += 1.0;
            }
            else {
              cur_facount(aj, prev_aj, tclass[prev_aj]) += 1.0;

              if (deficient_parametric) {

                uint diff = zero_offset - prev_aj;
                diff += (aj >= curI) ? aj - curI : aj;

                dist_count(diff, tclass[prev_aj]) += 1.0;
              }
            }
          }
        }
      }
    }

    //finish counts (align already done)

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

          for (uint c = 0; c < acount[I - 1].zDim(); c++) {
            for (int i = 0; i < (int)I; i++) {

              for (int ii = 0; ii < (int)I; ii++) {
                source_fert_count[1] += acount[I - 1] (ii, i, c);
              }
              source_fert_count[0] += acount[I - 1] (I, i, c);
            }
          }
        }
      }
    }
    //include the dict_regularity term in the output energy
    if (options.print_energy_) {
      double energy = prev_perplexity;
      for (uint i = 0; i < dcount.size(); i++)
        for (uint k = 0; k < dcount[i].size(); k++)
          if (dcount[i][k] > 0)
            //we need to divide as we are truly minimizing the perplexity WITHOUT division plus the l0-term
            energy += prior_weight[i][k];

      std::cerr << "energy after iteration #" << (iter - 1) << ": " << (energy / nSentences) << std::endl;
    }
    //std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    //no need to update dict here

    double sfsum = source_fert_count.sum();
    if (sfsum > 1e-305 && !options.fix_p0_) {
      for (uint k = 0; k < 2; k++)
        source_fert_prob[k] = source_fert_count[k] / source_fert_count.sum();
    }

    if (init_type == HmmInitPar) {

      start_prob_m_step(fsentence_start_count, fstart_span_count, init_params, options.init_m_step_iter_, options.gd_stepsize_);

      par2nonpar_hmm_init_model(init_params, source_fert_prob, init_type, initial_prob, start_empty_word, options.fix_p0_);
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

      ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_,
                  dist_grouping_param, options.deficient_, options.redpar_limit_);

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert_prob, align_type, options.deficient_,
                                     redpar_limit, align_model);
    }
    else {

      if (align_type == HmmAlignProbNonpar) {

        for (uint I = 1; I <= maxI; I++) {

          if (align_model[I - 1].xDim() != 0) {

            for (uint c = 0; c < acount[I - 1].zDim(); c++) {
              for (uint i = 0; i < I; i++) {

                double sum = 0.0;
                for (uint i_next = 0; i_next <= I; i_next++)
                  sum += acount[I - 1](i_next, i, c);

                if (sum >= 1e-300) {

                  const double inv_sum = 1.0 / sum;
                  assert(!isnan(inv_sum));

                  for (uint i_next = 0; i_next <= I; i_next++) {
                    align_model[I - 1](i_next, i, c) = std::max(hmm_min_param_entry, inv_sum * acount[I - 1](i_next, i, c));
                  }
                }
              }
            }
          }
        }
      }
      else if (align_type == HmmAlignProbNonpar2) {

        for (uint I = 1; I <= maxI; I++) {

          if (align_model[I - 1].xDim() != 0) {

            for (uint c = 0; c < acount[I - 1].zDim(); c++) {
              for (uint i = 0; i < I; i++) {

                double sum = 0.0;
                for (uint i_next = 0; i_next < I; i_next++)
                  sum += acount[I - 1](i_next, i, c);

                align_model[I - 1](I, i, c) = source_fert_prob[0];

                if (sum >= 1e-300) {

                  const double inv_sum = 1.0 / sum;
                  assert(!isnan(inv_sum));

                  for (uint i_next = 0; i_next < I; i_next++) {
                    align_model[I - 1](i_next, i, c) = std::max(hmm_min_param_entry, inv_sum * source_fert_prob[1] * acount[I - 1](i_next, i, c));
                  }
                }
                else {
                  for (uint i_next = 0; i_next < I; i_next++) {
                    align_model[I - 1](i_next, i, c) = source_fert_prob[1] / double (I);
                  }
                }
              }
            }
          }
        }
      }
    }

#if 1
    std::cerr << "starting ICM stage" << std::endl;

    /**** ICM stage ****/

    uint nSwitches = 0;

    Math1D::Vector<int> dist_count_sum(dist_count.yDim());
    for (uint c = 0; c < dist_count.yDim(); c++)
      dist_count_sum[c] = dist_count.row_sum(c);

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

      const Math3D::Tensor<double>& cur_align_model = align_model[curI - 1];
      const Math1D::Vector<double>& cur_initial_prob = initial_prob[curI - 1];
      Math3D::Tensor<double>& cur_acount = acount[curI - 1];
      Math1D::Vector<double>& cur_icount = icount[curI - 1];

      if ((s % 1250) == 0)
        std::cerr << "##################s: " << s << ", I = " << curI << ", J = " << curJ << std::endl;

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      Storage1D<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      const bool implicit = (deficient_parametric &&
                             (align_type == HmmAlignProbFullpar || (align_type == HmmAlignProbReducedpar && curI <= redpar_limit)));

      for (uint j = 0; j < curJ; j++) {

        // if (s == 2914) {
        // std::cerr << "###j: " << j << std::endl;
        // std::cerr << "J=" << curJ << ", I=" << curI << std::endl;
        // }

        const ushort cur_aj = cur_alignment[j];
        ushort new_aj = cur_aj; // is argbest_change
        ushort prev_aj = (j > 0) ? cur_alignment[j - 1] : MAX_UINT;

        uint effective_cur_aj = cur_aj;
        if (effective_cur_aj >= curI)
          effective_cur_aj -= curI;

        uint effective_prev_aj = MAX_UINT;
        if (j > 0) {
          effective_prev_aj = cur_alignment[j - 1];
          if (effective_prev_aj >= curI)
            effective_prev_aj -= curI;
        }
        //std::cerr << "effective_prev_aj: " << effective_prev_aj << std::endl;

        uint effective_next_aj = MAX_UINT;
        uint jj = j+1;
        for (; jj < curJ; jj++) {
          if (cur_alignment[jj] < curI) {
            effective_next_aj = cur_alignment[jj];
            break;
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

        if (cur_dictcount[cur_idx] > 1) {
          //exploit log(1) = 0
          common_change -= -xlogx_table[cur_dictcount[cur_idx]];
          common_change += -xlogx_table[cur_dictcount[cur_idx] - 1];
        }
        else
          common_change -= prior_weight[cur_target_word][cur_idx];

        for (ushort i = 0; i < nLabels; i++) {

          //if (s == 2914)
          //std::cerr << "##i: " << i << std::endl;

          if (i == cur_aj)
            continue;
          if (start_empty_word && j == 0 && i >= curI && i < 2 * curI)
            continue;
          if (!start_empty_word && j == 0 && i > curI)
            continue;

          if (j > 0 && i >= curI) {

            //ushort prev_aj = cur_alignment[j - 1];
            if (i != prev_aj && i != prev_aj + curI)
              continue;
            if (i == 2 * curI && prev_aj != i)  //need this!
              continue;
          }

          bool fix = true;

          uint effective_i = i;
          if (effective_i >= curI)
            effective_i -= curI;

          // if (j + 1 < curJ) {

          // uint next_aj = cur_alignment[j + 1];

          // //TODO: handle these special cases (some distributions are doubly affected)
          // if (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2) {

          // fix = false;

          // if (align_type == HmmAlignProbNonpar2 && std::max(cur_aj, i) >= curI)
          // fix = true;

          // if (next_aj >= curI)
          // fix = true;
          // if (effective_prev_aj == effective_cur_aj)
          // fix = true;
          // if (effective_prev_aj == effective_i)
          // fix = true;
          // }
          // else if (deficient_parametric) {

          // fix = (effective_i == curI || effective_cur_aj == curI);
          // if (align_type == HmmAlignProbReducedpar)
          // fix = true; //grouping count is hard to handle

          // if (j > 0 && effective_prev_aj == curI)
          // fix = true;

          // if (next_aj >= curI)
          // fix = true;
          // if (effective_cur_aj - effective_prev_aj == next_aj - effective_cur_aj)
          // fix = true;
          // if (effective_cur_aj - effective_prev_aj == next_aj - effective_i)
          // fix = true;
          // if (effective_cur_aj - effective_i == next_aj - effective_cur_aj)
          // fix = true;
          // if (effective_cur_aj - effective_i == next_aj - effective_i)
          // fix = true;
          // }
          // }

          const uint new_target_word = (i >= curI) ? 0 : cur_target[i];
          const uint hyp_idx = (i >= curI) ? source[s][j] - 1 : cur_lookup(j, i);

          double change = 0.0;

          if (cur_target_word != new_target_word) {

            if (dict_sum[new_target_word] > 0) {
              //exploit log(1) = 0
              change -= xlogx_table[dict_sum[new_target_word]];
              change += xlogx_table[dict_sum[new_target_word] + 1];
            }

            if (dcount[new_target_word][hyp_idx] > 0) {
              //exploit log(1) = 0
              change -= -xlogx_table[dcount[new_target_word][hyp_idx]];
              change += -xlogx_table[dcount[new_target_word][hyp_idx] + 1];
            }
            else
              change += prior_weight[new_target_word][hyp_idx];

            change += common_change;
          }
          //if (s == 2914)
          //std::cerr << "a)" << std::endl;

          std::map<NonparHMMCCountStruct, int> nonpar_count_change;
          std::map<std::pair<int,int>, int> nonpar_total_change;

          std::map<std::pair<int,int>, int> dist_count_change;
          Math1D::Vector<int> total_dist_count_sum_change(dist_count.yDim(),0);

          std::map<int, int> icount_change;
          int total_icount_change = 0;

          //a) changes regarding preceeding pos and alignments to NULL
          if (j == 0) {

            if (!start_empty_word && init_type == HmmInitNonpar) {
              assert(cur_icount[cur_aj] > 0);

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
            else if (start_empty_word && init_type == HmmInitNonpar) {
              icount_change[std::min<ushort>(curI,cur_aj)]--;
              icount_change[std::min<ushort>(curI,i)]++;
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

            const uint tc = (effective_prev_aj < curI) ? tclass[effective_prev_aj] : MAX_UINT;

            //note: the total sum of counts for effective_prev_aj stays constant in this operation
            if (align_type == HmmAlignProbNonpar) {
              if (effective_prev_aj < curI) {
                nonpar_count_change[NonparHMMCCountStruct(std::min<ushort>(curI, cur_aj), effective_prev_aj, tc)]--;
                nonpar_count_change[NonparHMMCCountStruct(std::min<ushort>(curI, i), effective_prev_aj, tc)]++;

                assert(cur_acount(std::min<ushort>(curI, cur_aj), effective_prev_aj, tc)
                       +  nonpar_count_change[NonparHMMCCountStruct(std::min<ushort>(curI, cur_aj), effective_prev_aj, tc)] >= 0);
                assert(cur_acount(std::min<ushort>(curI, i), effective_prev_aj, tc)
                       + nonpar_count_change[NonparHMMCCountStruct(std::min<ushort>(curI, i), effective_prev_aj, tc)] >= 0);
              }
              else {
                if (init_type == HmmInitNonpar) {
                  icount_change[std::min<ushort>(curI, cur_aj)]--;
                  icount_change[std::min<ushort>(curI, i)]++;
                }
                else {
                  change -= -std::log(cur_initial_prob[std::min<ushort>(curI, cur_aj)]);
                  change += -std::log(cur_initial_prob[std::min<ushort>(curI, i)]);
                }
              }
            }
            else if (align_type == HmmAlignProbNonpar2) {
              if (effective_prev_aj < curI) {
                if (cur_aj >= curI) {
                  change -= -std::log(source_fert_prob[0]);
                  change += -std::log(source_fert_prob[1]);
                  assert(i < curI);
                  nonpar_count_change[NonparHMMCCountStruct(i,effective_prev_aj,tc)]++;
                  nonpar_total_change[std::make_pair(effective_prev_aj, tc)]++;
                }
                else if (i >= curI) {
                  change -= -std::log(source_fert_prob[1]);
                  change += -std::log(source_fert_prob[0]);
                  assert(cur_aj < curI);
                  nonpar_count_change[NonparHMMCCountStruct(cur_aj,effective_prev_aj, tc)]--;
                  nonpar_total_change[std::make_pair(effective_prev_aj, tc)]--;
                }
                else {
                  nonpar_count_change[NonparHMMCCountStruct(cur_aj, effective_prev_aj, tc)]--;
                  nonpar_count_change[NonparHMMCCountStruct(i, effective_prev_aj, tc)]++;
                }
              }
              else {
                if (init_type == HmmInitNonpar) {
                  icount_change[std::min<ushort>(curI, cur_aj)]--;
                  icount_change[std::min<ushort>(curI, i)]++;
                }
                else {
                  change -= -std::log(cur_initial_prob[std::min<ushort>(curI, cur_aj)]);
                  change += -std::log(cur_initial_prob[std::min<ushort>(curI, i)]);
                }
              }
            }
            // if (!fix && (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2)) {

            // if (effective_prev_aj < curI) {
            // const int cur_c = cur_acount(std::min<ushort>(curI, cur_aj), effective_prev_aj, tc);
            // const int new_c = cur_acount(std::min<ushort>(curI, i), effective_prev_aj, tc);
            // assert(cur_c > 0);

            // if (cur_c > 1) {
            // //exploit log(1) = 0
            // change -= -xlogx_table[cur_c];
            // change += -xlogx_table[cur_c - 1];
            // }

            // if (new_c > 0) {
            // //exploit log(1) = 0
            // change -= -xlogx_table[new_c];
            // change += -xlogx_table[new_c + 1];
            // }
            // }
            // else {
            // change -= -std::log(cur_initial_prob[std::min<ushort>(curI, cur_aj)]);
            // change += -std::log(cur_initial_prob[std::min<ushort>(curI, i)]);
            // }
            // }
            else {  //parametric model

              if (implicit) {
                if (effective_prev_aj < curI) {
                  if (cur_aj < curI) {
                    const uint tc = tclass[effective_prev_aj];
                    dist_count_change[std::make_pair(zero_offset + cur_aj - effective_prev_aj, tc)]--;
                    total_dist_count_sum_change[tc]--;
                  }
                  else {
                    change -= -std::log(source_fert_prob[0]);
                  }
                  if (i < curI) {
                    const uint tc = tclass[effective_prev_aj];
                    dist_count_change[std::make_pair(zero_offset + i - effective_prev_aj, tc)]++;
                    total_dist_count_sum_change[tc]++;
                  }
                  else {
                    change += -std::log(source_fert_prob[0]);
                  }
                }
                else {
                  change -= -std::log(cur_initial_prob[std::min<ushort>(curI, cur_aj)]);
                  change += -std::log(cur_initial_prob[std::min<ushort>(curI, i)]);
                }
              }
              else {

                if (!fix && deficient_parametric) {
                  if (cur_aj < curI) {

                    const int cur_c = dist_count(zero_offset + cur_aj - effective_prev_aj, tc);

                    if (cur_c > 1) {
                      //exploit that log(1) = 0
                      change -= -xlogx_table[cur_c];
                      change += -xlogx_table[cur_c - 1];
                    }
                  }
                }
                else {
                  if (effective_prev_aj < curI)
                    change -= -std::log(cur_align_model(std::min<ushort>(curI, cur_aj), effective_prev_aj, tc));
                  else
                    change -= -std::log(cur_initial_prob[effective_cur_aj]);
                }

                if (!fix && deficient_parametric) {

                  if (i < curI) {

                    const int cur_c = dist_count(zero_offset + i - effective_prev_aj, tc);
                    if (cur_c > 0) {
                      //exploit that log(1) = 0
                      change -= -xlogx_table[cur_c];
                      change += -xlogx_table[cur_c + 1];
                    }
                  }
                }
                else {
                  if (effective_prev_aj < curI)
                    change += -std::log(cur_align_model(std::min(curI, i), effective_prev_aj, tc));
                  else
                    change += -std::log(cur_initial_prob[effective_i]);
                }
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

                  change -= xlogx_table[dist_count_sum[tc]];
                  change += xlogx_table[dist_count_sum[tc] - 1];
                }
                else if (cur_aj >= curI && i < curI) {

                  int cur_c0 = source_fert_count[0];
                  int cur_c1 = source_fert_count[1];

                  if (!options.fix_p0_) {
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

                  change -= xlogx_table[dist_count_sum[tc]];
                  change += xlogx_table[dist_count_sum[tc] + 1];
                }
              }
            }
          }

          assert(!isnan(change));

          //if (s == 2914)
          //std::cerr << "b), effective_cur_aj: " << effective_cur_aj << ", effective_i: " << effective_i << std::endl;

          //b) regarding succeeding pos
          if (j + 1 < curJ && effective_cur_aj != effective_i) {

            const uint tc_old = (effective_cur_aj < curI) ? tclass[effective_cur_aj] : MAX_UINT;
            const uint tc_new = (effective_i < curI) ? tclass[effective_i] : MAX_UINT;
            //if (s == 2914)
            //  std::cerr << "tc_old: " << tc_old << ", tc_new: " << tc_new << std::endl;

            if (align_type == HmmAlignProbNonpar) {
              if (effective_next_aj < curI) {
                if (effective_cur_aj < curI) {
                  nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_cur_aj,tc_old)]--;
                  nonpar_total_change[std::make_pair(effective_cur_aj,tc_old)]--;
                  assert(cur_acount(effective_next_aj,effective_cur_aj,tc_old)
                         + nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_cur_aj,tc_old)] >= 0);
                }
                else {
                  if (init_type == HmmInitNonpar) {
                    icount_change[effective_next_aj]--;
                    total_icount_change--;
                  }
                  else
                    change -= -std::log(cur_initial_prob[effective_next_aj]);
                }

                if (effective_i < curI) {

                  nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_i,tc_new)]++;
                  nonpar_total_change[std::make_pair(effective_i,tc_new)]++;
                  assert(cur_acount(effective_next_aj,effective_i,tc_new)
                         + nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_i,tc_new)] >= 0);
                }
                else {
                  if (init_type == HmmInitNonpar) {
                    icount_change[effective_next_aj]++;
                    total_icount_change++;
                  }
                  else
                    change += -std::log(cur_initial_prob[effective_next_aj]);
                }
              }

              if (jj > j+1) {
                const uint ajjj = cur_alignment[j+1];
                assert(ajjj >= curI);
                assert(ajjj == effective_cur_aj + curI);
                const uint effective_ajjj = ajjj - curI;

                if (ajjj == 2*curI) {
                  for (uint jjj=j+1; jjj < jj; jjj++) {
                    assert(cur_alignment[jjj] == ajjj);
                    if (init_type == HmmInitNonpar) {
                      icount_change[curI]--;
                      total_icount_change--;
                    }
                    else
                      change -= -std::log(cur_initial_prob[curI]);
                  }
                }
                else {
                  for (uint jjj=j+1; jjj < jj; jjj++) {
                    nonpar_count_change[NonparHMMCCountStruct(curI,effective_ajjj,tclass[effective_ajjj])]--;
                    nonpar_total_change[std::make_pair(effective_ajjj,tclass[effective_ajjj])]--;
                  }
                }

                if (i == 2*curI) {
                  for (uint jjj=j+1; jjj < jj; jjj++) {
                    if (init_type == HmmInitNonpar) {
                      icount_change[curI]++;
                      total_icount_change++;
                    }
                    else
                      change += -std::log(cur_initial_prob[curI]);
                  }
                }
                else {
                  for (uint jjj=j+1; jjj < jj; jjj++) {
                    nonpar_count_change[NonparHMMCCountStruct(curI,effective_i,tclass[effective_i])]++;
                    nonpar_total_change[std::make_pair(effective_i,tclass[effective_i])]++;
                  }
                }
              }
            }
            else if (align_type == HmmAlignProbNonpar2) {

              if (effective_next_aj < curI) {
                if (cur_aj >= curI) {

                  if (effective_cur_aj < curI) {
                    change -= -std::log(source_fert_prob[0]);
                    nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_cur_aj,tc_old)]--;
                    nonpar_total_change[std::make_pair(effective_cur_aj,tc_old)]--;
                  }
                  else
                    change -= -std::log(cur_initial_prob[effective_next_aj]);

                  assert(effective_i < curI);
                  if (effective_i < curI) {
                    change += -std::log(source_fert_prob[1]);
                    nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_i,tc_new)]++;
                    nonpar_total_change[std::make_pair(effective_i,tc_new)]++;
                  }
                  else
                    change += -std::log(cur_initial_prob[effective_next_aj]);
                }
                else if (i >= curI) {

                  change -= -std::log(source_fert_prob[1]);
                  change += -std::log(source_fert_prob[0]);
                  assert(effective_cur_aj < curI);
                  if (effective_cur_aj < curI) {
                    change -= -std::log(source_fert_prob[1]);
                    nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_cur_aj,tc_old)]--;
                    nonpar_total_change[std::make_pair(effective_cur_aj,tc_old)]--;
                  }
                  if (effective_i < curI) {
                    change += -std::log(source_fert_prob[0]);
                    nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_i,tc_new)]++;
                    nonpar_total_change[std::make_pair(effective_i,tc_new)]++;
                  }
                  else
                    change += -std::log(cur_initial_prob[effective_next_aj]);
                }
                else {
                  assert(cur_aj < curI && i < curI);

                  nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_cur_aj,tc_old)]--;
                  nonpar_total_change[std::make_pair(effective_cur_aj,tc_old)]--;
                  nonpar_count_change[NonparHMMCCountStruct(effective_next_aj,effective_i,tc_new)]++;
                  nonpar_total_change[std::make_pair(effective_i,tc_new)]++;
                }
              }

              if (jj > j+1 && init_type == HmmInitNonpar && cur_alignment[j+1] == 2*curI) {

                for (uint jjj=j+1; jjj < jj; jjj++) {
                  change -= -std::log(cur_initial_prob[curI]);
                  change += -std::log(source_fert_prob[0]);
                }
              }
              else if (jj > j+1 && init_type == HmmInitNonpar && i == 2*curI) {

                for (uint jjj=j+1; jjj < jj; jjj++) {
                  change -= -std::log(source_fert_prob[0]);
                  change += -std::log(cur_initial_prob[curI]);
                }
              }
            }
            // if (!fix && (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2)) {

            // const uint limit = (align_type == HmmAlignProbNonpar) ? curI : curI - 1;

            // if (effective_cur_aj < curI) {
            // int total_cur_count = 0;
            // const uint c = tclass[effective_cur_aj];
            // for (uint k = 0; k <= limit; k++)
            // total_cur_count += cur_acount(k, effective_cur_aj, c);

            // assert(total_cur_count > 0);
            // assert(cur_acount(effective_next_aj, effective_cur_aj, c) > 0);

            // if (total_cur_count > 1) {
            // //exploit log(1) = 0
            // change -= xlogx_table[total_cur_count];
            // change += xlogx_table[total_cur_count - 1];
            // }

            // if (cur_acount(effective_next_aj, effective_cur_aj, c) > 1) {
            // //exploit log(1) = 0
            // change -= -xlogx_table[cur_acount(effective_next_aj, effective_cur_aj, c)];
            // change += -xlogx_table[cur_acount(effective_next_aj, effective_cur_aj, c) - 1];
            // }
            // }
            // else {

            // change -= -std::log(initial_prob[curI - 1][curI]);
            // }

            // if (effective_i < curI) {
            // int total_new_count = 0;
            // const uint c = tclass[effective_i];
            // for (uint k = 0; k <= limit; k++)
            // total_new_count += cur_acount(k, effective_i, c);

            // if (total_new_count > 0) {
            // //exploit log(1) = 0
            // change -= xlogx_table[total_new_count];
            // change += xlogx_table[total_new_count + 1];
            // }
            // if (cur_acount(effective_next_aj, effective_i, c) > 0) {
            // //exploit log(1) = 0
            // change -= -xlogx_table[cur_acount(effective_next_aj, effective_i, c)];
            // change += -xlogx_table[cur_acount(effective_next_aj, effective_i, c) + 1];
            // }
            // }
            // else {

            // change += -std::log(initial_prob[curI - 1][curI]);
            // }

            // assert(!isnan(change));
            // }
            else {
              //parametric model

              if (implicit) {
                if (effective_next_aj < curI) {
                  if (effective_cur_aj < curI) {
                    const uint tc = tclass[effective_cur_aj];
                    dist_count_change[std::make_pair(zero_offset + effective_next_aj - effective_cur_aj, tc)]--;
                    total_dist_count_sum_change[tc]--;
                  }
                  else {
                    assert(effective_prev_aj >= curI);
                  }
                  if (effective_i != curI) {
                    const uint tc = tclass[effective_i];
                    dist_count_change[std::make_pair(zero_offset + effective_next_aj - effective_i, tc)]++;
                    total_dist_count_sum_change[tc]++;
                  }
                  else {
                    assert(effective_prev_aj >= curI);
                  }
                }
              }
              else if (!fix && deficient_parametric) {
                assert(j + 1 < curJ);
                assert(effective_i < curI && effective_cur_aj < curI);

                if (cur_alignment[j + 1] < curI) {

                  const int cur_c = dist_count(zero_offset + effective_next_aj - effective_cur_aj, tclass[effective_cur_aj]);
                  const int new_c = dist_count(zero_offset + effective_next_aj - effective_i, tclass[effective_i]);

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
                    change -= -std::log(cur_align_model(effective_next_aj, effective_cur_aj, tclass[effective_cur_aj]));
                    if (effective_i != curI)
                      change += -std::log(cur_align_model(effective_next_aj, effective_i, tclass[effective_i]));
                    else
                      change += -std::log(cur_initial_prob[effective_next_aj]);
                  }
                  else {
                    change -= -std::log(cur_initial_prob[effective_next_aj]);
                    change += -std::log(cur_align_model(effective_next_aj, effective_i, tclass[effective_i]));
                  }
                }
              }
            }
          }

          //if (s == 2914)
          //std::cerr << "c)" << std::endl;
          for (std::map<NonparHMMCCountStruct,int>::const_iterator it = nonpar_count_change.begin();
               it != nonpar_count_change.end(); it++) {

            const int cur = it->first.cur_;
            const int given = it->first.last_;
            const int c = it->first.target_class_;
            const int diff = it->second;

            int cur_c = cur_acount(cur,given,c);
            assert(cur_c + diff >= 0);
            change -= -xlogx_table[cur_c];
            change += -xlogx_table[cur_c + diff];
          }
          for (std::map<std::pair<int,int>, int>::const_iterator it = nonpar_total_change.begin(); it != nonpar_total_change.end(); it++) {
            const int given = it->first.first;
            const int c = it->first.second;
            const int diff = it->second;

            const uint limit = (align_type == HmmAlignProbNonpar) ? curI : curI - 1;

            int total_new_count = 0;
            for (uint k = 0; k <= limit; k++)
              total_new_count += cur_acount(k, given, c);
            assert(total_new_count + diff >= 0);

            change -= xlogx_table[total_new_count];
            change += xlogx_table[total_new_count + diff];
          }
          for (std::map<std::pair<int,int>, int>::const_iterator it = dist_count_change.begin(); it != dist_count_change.end(); it++) {
            const int offs = it->first.first;
            const int c = it->first.second;
            const int diff = it->second;
            //std::cerr << "diff: " << diff << std::endl;

            const int cur_c = dist_count(offs,c);
            change -= -xlogx_table[cur_c];
            assert(cur_c + diff >= 0);
            change += -xlogx_table[cur_c + diff];
          }
          for (uint c=0; c < total_dist_count_sum_change.size(); c++) {
            const int diff = total_dist_count_sum_change[c];
            if (diff != 0)
              change += xlogx_table[dist_count_sum[c]+diff] - xlogx_table[dist_count_sum[c]];
          }

          for (std::map<int, int>::const_iterator it = icount_change.begin(); it != icount_change.end(); it++) {
            int pos = it->first;
            int diff = it->second;
            //if (s == 3901)
            //std::cerr << "pos: " << pos << ", diff: " << diff << ", cur count: " << cur_icount[pos] << std::endl;
            change -= -xlogx_table[cur_icount[pos]];
            assert(cur_icount[pos] + diff >= 0);
            change += -xlogx_table[cur_icount[pos]+diff];
          }
          if (total_icount_change != 0) {
            int sum = cur_icount.sum();
            change -= xlogx_table[sum];
            assert(sum + total_icount_change >= 0);
            change += xlogx_table[sum+total_icount_change];
          }

          if (change < best_change) {

            best_change = change;
            new_aj = i;
          }
        }

        // if (s == 2914) {
        // std::cerr << "base alignment: " << cur_alignment << std::endl;
        // std::cerr << "best_change: " << best_change << ", argbest: " << new_aj << std::endl;
        // }

        if (best_change < -0.01 && new_aj != cur_aj) {

          // if (s == 2914) {
          // std::cerr << "base alignment: " << cur_alignment << std::endl;
          // std::cerr << "switching a" << j << " from " << cur_aj << " to " << new_aj << std::endl;
          // std::cerr << "I: " << curI << ", J: " << curJ << std::endl;
          // std::cerr << "initial source_fert_count: " << source_fert_count << std::endl;
          // }
          nSwitches++;

          const uint new_target_word = (new_aj >= curI) ? 0 : cur_target[new_aj];
          const uint hyp_idx = (new_aj >= curI) ? cur_source[j] - 1 : cur_lookup(j, new_aj);

          cur_alignment[j] = new_aj;

          if (j > 0 && new_aj >= curI)
            assert(new_aj == cur_alignment[j - 1] || new_aj - curI == cur_alignment[j - 1]);

          ushort effective_new_aj = new_aj;
          if (effective_new_aj >= curI)
            effective_new_aj -= curI;

          if (j + 1 < curJ) {

            ushort next_aj = cur_alignment[j + 1];

            //if (s == 2914)
            //std::cerr << "next_aj: " << next_aj << std::endl;

            if (next_aj >= curI) {

              ushort new_next_aj = (new_aj < curI) ? new_aj + curI : new_aj;

              //if (s == 2914)
              //std::cerr << "new_next_aj: " << new_next_aj << std::endl;

              if (new_next_aj != next_aj) {

                for (uint jjj = j+1; jjj < jj; jjj++) {
                  assert(cur_alignment[jjj] == next_aj);
                  cur_alignment[jjj] = new_next_aj;
                  //if (s == 2914)
                  //std::cerr << "jjj: " << jjj << std::endl;
                  if (next_aj != 2 * curI) {
                    //if (s == 2914)
                    //std::cerr << "decrease " << curI << ", " << (next_aj-curI) << ", " << tclass[next_aj-curI] << std::endl;
                    cur_acount(curI, next_aj - curI, tclass[next_aj - curI])--;
                    assert(cur_acount(curI, next_aj - curI, tclass[next_aj - curI]) >= 0);
                    if (align_type != HmmAlignProbNonpar)
                      source_fert_count[0]--;
                  }
                  else {
                    cur_icount[curI]--;
                    if (init_type == HmmInitPar)
                      source_fert_count[0]--;
                  }
                  if (new_next_aj != 2 * curI) {
                    //if (s == 2914)
                    //std::cerr << "increase " << curI << ", " << (new_next_aj - curI) << ", " << tclass[new_next_aj - curI] << std::endl;
                    cur_acount(curI, new_next_aj - curI, tclass[new_next_aj - curI])++;
                    if (align_type != HmmAlignProbNonpar)
                      source_fert_count[0]++;
                  }
                  else {
                    cur_icount[curI]++;
                    if (init_type == HmmInitPar)
                      source_fert_count[0]++;
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

            if (s == 2914)
              std::cerr << "A" << std::endl;

            if (cur_aj != 2 * curI) {
              assert(cur_icount[cur_aj] > 0);
              cur_icount[cur_aj]--;
              if (init_type == HmmInitPar) {
                if (cur_aj < curI)
                  source_fert_count[1]--;
                else
                  source_fert_count[0]--;
              }
            }
            else {
              assert(cur_icount[curI] > 0);
              cur_icount[curI]--;
              if (init_type == HmmInitPar)
                source_fert_count[0]--;
            }

            if (new_aj != 2 * curI) {
              cur_icount[new_aj]++;
              if (init_type == HmmInitPar) {
                if (new_aj < curI)
                  source_fert_count[1]++;
                else
                  source_fert_count[0]++;
              }
            }
            else {
              cur_icount[curI]++;
              if (init_type == HmmInitPar)
                source_fert_count[0]++;
            }

            // if (init_type == HmmInitPar) {

            // if (s == 2914)
            // //std::cerr << "B" << std::endl;
            // if (cur_aj < curI && new_aj >= curI) {
            // source_fert_count[1]--;
            // source_fert_count[0]++;
            // }
            // else if (cur_aj >= curI && new_aj < curI) {
            // source_fert_count[0]--;
            // source_fert_count[1]++;
            // }
            // }
          }
          else { // j > 0

            if (effective_prev_aj != curI) {
              // if (s == 2914) {
              // std::cerr << "C" << std::endl;
              // std::cerr << "decrease " << std::min<ushort>(curI, cur_aj) << ", " << effective_prev_aj << ", " << tclass[effective_prev_aj] << std::endl;
              // }
              cur_acount(std::min<ushort>(curI, cur_aj), effective_prev_aj, tclass[effective_prev_aj])--;
              assert(cur_acount(std::min<ushort>(curI, cur_aj), effective_prev_aj, tclass[effective_prev_aj]) >= 0);
              //if (s == 2914)
              //std::cerr << "increase " << std::min<ushort>(curI, new_aj) << ", " << effective_prev_aj << ", " << tclass[effective_prev_aj] << std::endl;
              cur_acount(std::min<ushort>(curI, new_aj), effective_prev_aj, tclass[effective_prev_aj])++;
              assert(cur_acount(std::min<ushort>(curI, new_aj), effective_prev_aj, tclass[effective_prev_aj]) >= 0);

              if (align_type != HmmAlignProbNonpar) {
                if (cur_aj < curI && new_aj >= curI) {
                  source_fert_count[1]--;
                  source_fert_count[0]++;
                  //if (s == 291496 || s == 29143296 || s == 2914)
                  //std::cerr << "C1 increase sf0, decrease sf1: " << source_fert_count << std::endl;
                }
                else if (cur_aj >= curI && new_aj < curI) {
                  source_fert_count[0]--;
                  source_fert_count[1]++;
                  //if (s == 291496 || s == 29143296 || s == 2914)
                  //std::cerr << "C2 decrease sf0, increase sf1: " << source_fert_count << std::endl;
                }
              }
            }
            else {
              //if (s == 2914)
              //std::cerr << "D" << std::endl;
              cur_icount[std::min<ushort>(curI, cur_aj)]--;
              cur_icount[std::min<ushort>(curI, new_aj)]++;

              if (init_type == HmmInitPar && std::max(cur_aj,new_aj) >= curI) {
                if (cur_aj >= curI) {
                  assert(new_aj < curI);
                  source_fert_count[0]--;
                  source_fert_count[1]++;
                }
                else {
                  assert(new_aj >= curI);
                  source_fert_count[1]--;
                  source_fert_count[0]++;
                }
              }
            }

            if (deficient_parametric && effective_prev_aj < curI) {

              //if (s == 2914)
              //std::cerr << "E" << std::endl;
              if (cur_aj < curI) {
                dist_count(zero_offset + cur_aj - effective_prev_aj, tclass[effective_prev_aj])--;
                dist_count_sum[tclass[effective_prev_aj]]--;
              }
              if (new_aj < curI) {
                dist_count(zero_offset + new_aj - effective_prev_aj, tclass[effective_prev_aj])++;
                dist_count_sum[tclass[effective_prev_aj]]++;
              }
            }
          }

          //if (s == 2914)
          //std::cerr << "b) " << std::endl;

          //b) dependency to succceeding pos
          if (jj < curJ) {

            //if (s == 2914)
            //std::cerr << "G" << std::endl;
            ushort next_aj = cur_alignment[jj];
            assert(next_aj < curI);
            ushort effective_cur_aj = cur_aj;
            if (effective_cur_aj >= curI)
              effective_cur_aj -= curI;
            ushort effective_new_aj = new_aj;
            if (effective_new_aj >= curI)
              effective_new_aj -= curI;

            //leaving
            if (effective_cur_aj < curI) {
              // if (s == 2914) {
              // std::cerr << "next_aj: " << next_aj << ", effective_cur_aj: " << effective_cur_aj << std::endl;
              // std::cerr << "decrease " << std::min(curI, next_aj) << ", " << effective_cur_aj << ", " << tclass[effective_cur_aj] << std::endl;
              // }
              cur_acount(std::min(curI, next_aj), effective_cur_aj, tclass[effective_cur_aj])--;
              assert(cur_acount(std::min(curI, next_aj), effective_cur_aj, tclass[effective_cur_aj]) >= 0);
              if (align_type != HmmAlignProbNonpar) {
                if (next_aj >= curI)
                  source_fert_count[0]--;
                else
                  source_fert_count[1]--;
              }
            }
            else {
              cur_icount[std::min(curI, next_aj)]--;
              if (init_type == HmmInitPar) {
                if (next_aj < curI)
                  source_fert_count[1]--;
                else
                  source_fert_count[0]--;
                //if (s == 291496 || s == 29143296)
                //std::cerr << "changed source fert count: " << source_fert_count << std::endl;
              }
            }

            //entering
            if (effective_new_aj < curI) {
              //if (s == 2914)
              //std::cerr << "increase " << std::min(curI, next_aj) << ", " << effective_new_aj << ", " << tclass[effective_new_aj] << std::endl;
              cur_acount(std::min(curI, next_aj), effective_new_aj, tclass[effective_new_aj])++;
              if (align_type != HmmAlignProbNonpar) {
                if (next_aj >= curI)
                  source_fert_count[0]++;
                else
                  source_fert_count[1]++;
              }
            }
            else {
              cur_icount[std::min(curI, next_aj)]++;
              if (init_type == HmmInitPar) {
                if (next_aj < curI)
                  source_fert_count[1]++;
                else
                  source_fert_count[0]++;
              }
            }

            if (deficient_parametric) {

              //if (s == 2914)
              //std::cerr << "H, effective_cur_aj: " << effective_cur_aj << ", effective_new_aj: " << effective_new_aj << std::endl;
              if (next_aj < curI) {
                if (effective_cur_aj < curI) {
                  dist_count(zero_offset + next_aj - effective_cur_aj, tclass[effective_cur_aj])--;
                  dist_count_sum[tclass[effective_cur_aj]]--;
                }
                if (effective_new_aj < curI) {
                  dist_count(zero_offset + next_aj - effective_new_aj, tclass[effective_new_aj])++;
                  dist_count_sum[tclass[effective_new_aj]]++;
                }
              }

              //if (s == 2914)
              //std::cerr << "after H" << std::endl;
            }
          }
//#if 0
#ifndef NDEBUG
          if ((s%750) == 0) {

            //std::cerr << "checks for sentence #" << s << std::endl;

            //check counts
            Math1D::Vector<double> check_source_fert_count(2, 0.0);

            Storage1D<Math1D::Vector<double> > check_dcount(options.nTargetWords_);
            for (uint i = 0; i < options.nTargetWords_; i++) {
              check_dcount[i].resize(dict[i].size(),0.0);
            }

            InitialAlignmentProbability check_icount(maxI, MAKENAME(check_icount));
            for (uint I = 0; I < maxI; I++) {
              check_icount[I].resize_dirty(icount[I].size());
              check_icount[I].set_constant(0.0);
            }

            FullHMMAlignmentModelSingleClass check_acount(maxI, MAKENAME(check_acount));
            for (uint I = 1; I <= maxI; I++) {
              if (align_model[I - 1].size() > 0) {
                check_acount[I - 1].resize_dirty(I + 1, I, align_model[I - 1].zDim());
                check_acount[I - 1].set_constant(0.0);
              }
            }

            Math2D::Matrix<double> check_dist_count(dist_params.dims(),0.0);       //used in deficient mode

            Math1D::Vector<double> check_fsentence_start_count(maxI,0.0);
            Math1D::Vector<double> check_fstart_span_count(maxI,0.0);

            for (size_t s = 0; s < nSentences; s++) {

              //std::cerr << "sentence #" << s << std::endl;

              const Storage1D<uint>& cur_source = source[s];
              const Storage1D<uint>& cur_target = target[s];
              const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

              Storage1D<AlignBaseType>& cur_alignment = viterbi_alignment[s];

              const uint curJ = cur_source.size();
              const uint curI = cur_target.size();

              Math1D::Vector<uint> tclass(curI);
              for (uint i = 0; i < curI; i++)
                tclass[i] = target_class[cur_target[i]];

              Math3D::Tensor<double>& cur_facount = check_acount[curI - 1];

              /**** update counts ****/

              for (uint j = 0; j < curJ; j++) {

                const ushort aj = cur_alignment[j];
                if (aj >= curI)
                  check_dcount[0][cur_source[j] - 1] += 1;
                else
                  check_dcount[cur_target[aj]][cur_lookup(j, aj)] += 1;

                if (j == 0) {
                  if (!start_empty_word)
                    check_icount[curI - 1][aj] += 1.0;
                  else {
                    if (aj < curI)
                      check_icount[curI - 1][aj] += 1.0;
                    else {
                      assert(aj == 2 * curI);
                      check_icount[curI - 1][curI] += 1.0;
                    }
                  }
                }
                else {

                  const ushort prev_aj = cur_alignment[j - 1];

                  if (prev_aj == 2 * curI) {
                    assert(start_empty_word);
                    if (aj == prev_aj)
                      check_icount[curI - 1][curI] += 1.0;
                    else
                      check_icount[curI - 1][aj] += 1.0;
                  }
                  else if (prev_aj >= curI) {

                    if (aj >= curI)
                      cur_facount(curI, prev_aj - curI, tclass[prev_aj - curI]) += 1.0;
                    else {
                      cur_facount(aj, prev_aj - curI, tclass[prev_aj - curI]) += 1.0;

                      if (deficient_parametric) {

                        uint diff = zero_offset - (prev_aj - curI);
                        diff += (aj >= curI) ? aj - curI : aj;

                        check_dist_count(diff, tclass[prev_aj - curI]) += 1.0;
                      }
                    }
                  }
                  else { //prev_aj is target pos
                    if (aj >= curI)
                      cur_facount(curI, prev_aj, tclass[prev_aj]) += 1.0;
                    else {
                      cur_facount(aj, prev_aj, tclass[prev_aj]) += 1.0;

                      if (deficient_parametric) {

                        uint diff = zero_offset - prev_aj;
                        diff += (aj >= curI) ? aj - curI : aj;

                        check_dist_count(diff, tclass[prev_aj]) += 1.0;
                      }
                    }
                  }
                }
              }
            }

            if (init_type == HmmInitPar) {

              for (uint I = 1; I <= maxI; I++) {

                if (initial_prob[I - 1].size() != 0) {
                  for (uint i = 0; i < I; i++) {

                    const double cur_count = check_icount[I - 1][i];
                    check_source_fert_count[1] += cur_count;
                    check_fsentence_start_count[i] += cur_count;
                    check_fstart_span_count[I - 1] += cur_count;
                  }
                  for (uint i = I; i < icount[I - 1].size(); i++) {
                    check_source_fert_count[0] += check_icount[I - 1][i];
                  }
                }
              }
            }

            if (align_type != HmmAlignProbNonpar) {

              //compute the expectations of the parameters from the expectations of the models

              for (uint I = 1; I <= maxI; I++) {

                if (align_model[I - 1].xDim() != 0) {

                  for (uint c = 0; c < acount[I - 1].zDim(); c++) {
                    for (int i = 0; i < (int)I; i++) {

                      for (int ii = 0; ii < (int)I; ii++) {
                        check_source_fert_count[1] += check_acount[I - 1] (ii, i, c);
                      }
                      check_source_fert_count[0] += check_acount[I - 1] (I, i, c);
                    }
                  }
                }
              }
            }

            if (check_source_fert_count != source_fert_count) {
              std::cerr << "sfc should be " << check_source_fert_count << ", is " << source_fert_count << std::endl;
            }
            assert(check_source_fert_count == source_fert_count);

            for (uint i = 0; i < options.nTargetWords_; i++) {
              assert(check_dcount[i] == dcount[i]);
            }

            for (uint I = 0; I < maxI; I++) {
              assert(check_icount[I] == icount[I]);
            }

            for (uint I = 1; I <= maxI; I++) {
              for (uint x=0; x < check_acount[I-1].xDim(); x++) {
                for (uint y=0; y < check_acount[I-1].yDim(); y++) {
                  for (uint z=0; z < check_acount[I-1].zDim(); z++) {
                    if (!(check_acount[I-1](x,y,z) == acount[I-1](x,y,z))) {
                      std::cerr << "I: " << I << ", x: " << x << ", y: " << y << ", z: " << z << std::endl;
                      std::cerr << "should be: " << check_acount[I-1](x,y,z) << std::endl;
                      std::cerr << "is:        " << acount[I-1](x,y,z) << std::endl;
                    }
                  }
                }
              }
              assert(check_acount[I-1] == acount[I-1]);
            }

            if (align_type == HmmAlignProbFullpar && deficient_parametric)
              assert(check_dist_count == dist_count);
          }
#endif
        }
      }
    }

    std::cerr << nSwitches << " switches in ICM" << std::endl;

    update_dict_from_counts(dcount, prior_weight, nSentences, 0.0, false, 0.0, 0, dict, hmm_min_dict_entry,
                            options.msolve_mode_ != MSSolvePGD, options.gd_stepsize_);

    sfsum = source_fert_count.sum();
    if (sfsum > 1e-305 && !options.fix_p0_) {
      for (uint k = 0; k < 2; k++)
        source_fert_prob[k] = source_fert_count[k] / sfsum;
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

      start_prob_m_step(fsentence_start_count, fstart_span_count, init_params, options.init_m_step_iter_, options.gd_stepsize_);

      par2nonpar_hmm_init_model(init_params, source_fert_prob, init_type, initial_prob, start_empty_word, options.fix_p0_);
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

      ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_,
                  dist_grouping_param, options.deficient_, options.redpar_limit_);

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert_prob,
                                     align_type, options.deficient_, redpar_limit, align_model);
    }
    else {

      if (align_type == HmmAlignProbNonpar) {

        for (uint I = 1; I <= maxI; I++) {

          if (align_model[I - 1].xDim() != 0) {

            for (uint c = 0; c < acount[I - 1].zDim(); c++) {
              for (uint i = 0; i < I; i++) {

                double sum = 0.0;
                for (uint i_next = 0; i_next <= I; i_next++)
                  sum += acount[I - 1](i_next, i, c);

                if (sum >= 1e-300) {

                  const double inv_sum = 1.0 / sum;
                  assert(!isnan(inv_sum));

                  for (uint i_next = 0; i_next <= I; i_next++) {
                    align_model[I - 1](i_next, i, c) = std::max(hmm_min_param_entry, inv_sum * acount[I - 1] (i_next, i, c));
                  }
                }
              }
            }
          }
        }
      }
      else if (align_type == HmmAlignProbNonpar2) {

        for (uint I = 1; I <= maxI; I++) {

          if (align_model[I - 1].xDim() != 0) {

            for (uint c = 0; c < acount[I - 1].zDim(); c++) {
              for (uint i = 0; i < I; i++) {

                double sum = 0.0;
                for (uint i_next = 0; i_next < I; i_next++)
                  sum += acount[I - 1] (i_next, i, c);

                align_model[I - 1](I, i, c) = source_fert_prob[0];

                if (sum >= 1e-300) {

                  const double inv_sum = 1.0 / sum;
                  assert(!isnan(inv_sum));

                  for (uint i_next = 0; i_next < I; i_next++) {
                    align_model[I - 1](i_next, i, c) = std::max(hmm_min_param_entry, inv_sum * source_fert_prob[1] * acount[I - 1](i_next, i, c));
                  }
                }
                else {
                  for (uint i_next = 0; i_next < I; i_next++) {
                    align_model[I - 1](i_next, i, c) = source_fert_prob[1] / double (I);
                  }
                }
              }
            }
          }
        }
      }
    }

    double energy = 0.0;
    for (size_t s = 0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const uint curI = cur_target.size();

      const Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      //std::cerr << "s: " << s << ", J: " << cur_source.size() << ", I: " << cur_target.size() << std::endl;
      //std::cerr << "alignment: " << cur_alignment << std::endl;

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      Storage1D<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      //assert(hmmc_alignment_prob(cur_source,cur_lookup,cur_target,tclass,dict,
      //                        align_model, initial_prob, cur_alignment, true) > 0.0);

      energy -= logl(hmmc_alignment_prob(cur_source, cur_lookup, cur_target, tclass, dict, align_model, initial_prob, cur_alignment, true));
    }

    if (dict_weight_sum > 0.0) {
      for (uint i = 0; i < dcount.size(); i++)
        for (uint k = 0; k < dcount[i].size(); k++)
          if (dcount[i][k] > 0)
            energy += prior_weight[i][k];
    }

    std::cerr << "energy after ICM and all updates: " << (energy / nSentences) << std::endl;
#else
    update_dict_from_counts(dcount, prior_weight, nSentences, 0.0, false, 0.0, 0, dict, hmm_min_dict_entry,
                            options.msolve_mode_ != MSSolvePGD, options.gd_stepsize_);
#endif

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      //std::cerr << "computing error rates" << std::endl;

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double sum_daes = 0.0;
      uint nContributors = 0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
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

        Math1D::Vector<uint> tclass(curI);
        for (uint i = 0; i < curI; i++)
          tclass[i] = target_class[target[s][i]];

        Math2D::Matrix<double> cur_align_model(curI+1, curI);
        par2nonpar_hmm_alignment_model(tclass, align_model, options, cur_align_model);

        //std::cerr << "computing alignment" << std::endl;

        compute_ehmmc_viterbi_alignment(source[s], cur_lookup, target[s], tclass, dict, cur_align_model,
                                        initial_prob[curI - 1], viterbi_alignment, options);

        //std::cerr << "alignment: " << viterbi_alignment << std::endl;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        sum_daes += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_daes /= nContributors;
      sum_fmeasure /= nContributors;

      std::cerr << "#### EHMM-SingleClass Viterbi-AER after Viterbi-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### EHMM-SingleClass Viterbi-fmeasure after Viterbi-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM-SingleClass Viterbi-DAE/S after Viterbi-iteration #" << iter << ": " << sum_daes << std::endl;
    }
  }
}
