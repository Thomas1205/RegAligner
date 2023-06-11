/**** written by Thomas Schoenemann, September 2022 ****/

#include "hmmcc_training.hh"
#include "alignment_computation.hh"
#include "alignment_error_rate.hh"
#include "hmm_forward_backward.hh"
#include "conditional_m_steps.hh"

HmmWrapperDoubleClasses::HmmWrapperDoubleClasses(const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
    const Math1D::Vector<double>& source_fert, const InitialAlignmentProbability& initial_prob,
    const Storage1D<WordClassType>& source_class, const Storage1D<WordClassType>& target_class,
    const HmmOptions& hmm_options, uint zero_offset)
  : HmmWrapperBase(hmm_options), dist_params_(dist_params), dist_grouping_param_(dist_grouping_param), source_fert_(source_fert),
    initial_prob_(initial_prob), source_class_(source_class), target_class_(target_class), zero_offset_(zero_offset) {}


/*virtual*/ long double HmmWrapperDoubleClasses::compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
    Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode, bool verbose,
    double min_dict_entry) const
{
  const uint curJ = source_sentence.size();
  Math1D::Vector<uint> sclass(curJ);
  for (uint j = 0; j < curJ; j++)
    sclass[j] = source_class_[source_sentence[j]];

  const uint curI = target_sentence.size();
  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = target_class_[target_sentence[i]];

  Math3D::Tensor<double> cur_align_model;
  par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_, dist_params_, dist_grouping_param_, cur_align_model, hmm_options_.align_type_,
                                   hmm_options_.deficient_, hmm_options_.redpar_limit_, zero_offset_);

  return ::compute_ehmmcc_viterbi_alignment(source_sentence, slookup, target_sentence, sclass, tclass, dict, cur_align_model, initial_prob_[curI-1],
         viterbi_alignment, hmm_options_, internal_mode, verbose, min_dict_entry);
}

/*virtual*/ void HmmWrapperDoubleClasses::compute_ehmm_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
    double threshold) const
{
  postdec_alignment.clear();

  const uint curJ = source_sentence.size();
  Math1D::Vector<uint> sclass(curJ);
  for (uint j = 0; j < curJ; j++)
    sclass[j] = source_class_[source_sentence[j]];

  const uint curI = target_sentence.size();
  Math1D::Vector<uint> tclass(curI);
  for (uint i = 0; i < curI; i++)
    tclass[i] = target_class_[target_sentence[i]];

  Math3D::Tensor<double> cur_align_model;
  par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_, dist_params_, dist_grouping_param_, cur_align_model, hmm_options_.align_type_,
                                   hmm_options_.deficient_,  hmm_options_.redpar_limit_, zero_offset_);

  ::compute_ehmmcc_postdec_alignment(source_sentence, slookup, target_sentence, sclass, tclass, dict, cur_align_model, initial_prob_[curI-1], hmm_options_,
                                     postdec_alignment, threshold);
}

/*virtual*/ void HmmWrapperDoubleClasses::fill_dist_params(Math1D::Vector<double>& hmm_dist_params, double& hmm_dist_grouping_param) const
{
  //std::cerr << "A, set size " << dist_params_(0,0).size() << std::endl;
  hmm_dist_params.resize_dirty(dist_params_(0,0).size());
  hmm_dist_params.set_constant(0.0);
  hmm_dist_grouping_param = 0.0;

  for (uint sc = 0; sc < dist_params_.xDim(); sc++) {
    for (uint tc = 0; tc < dist_params_.yDim(); tc++) {
      if (dist_params_(sc,tc).size() > 0)
        hmm_dist_params += dist_params_(sc,tc);
      hmm_dist_grouping_param += dist_grouping_param_(sc,tc);
    }
  }

  double sum = hmm_dist_params.sum() + hmm_dist_grouping_param;
  if (sum > 0.0) {
    hmm_dist_params *= 1.0 / sum;
    hmm_dist_grouping_param *= 1.0 / sum;
  }
}

/*virtual*/ void HmmWrapperDoubleClasses::fill_dist_params(uint nTargetClasses, Math2D::Matrix<double>& hmmc_dist_params, Math1D::Vector<double>& hmmc_dist_grouping_param) const
{
  //std::cerr << "B" << std::endl;
  hmmc_dist_params.resize_dirty(dist_params_(0,0).size(),dist_params_.yDim());
  hmmc_dist_params.set_constant(0.0);
  hmmc_dist_grouping_param.resize_dirty(dist_grouping_param_.xDim());
  hmmc_dist_grouping_param.set_constant(0.0);

  for (uint sc = 0; sc < dist_params_.xDim(); sc++) {
    for (uint tc = 0; tc < dist_grouping_param_.xDim(); tc++) {
      for (uint k = 0; k < dist_params_(sc,tc).size(); k++) {
        hmmc_dist_params(k, tc) += dist_params_(sc, tc)[k];
      }
      hmmc_dist_grouping_param[tc] += dist_grouping_param_(sc,tc);
    }
  }

  for (uint tc = 0; tc < dist_grouping_param_.xDim(); tc++) {
    double sum = hmmc_dist_grouping_param[tc];
    for (uint k=0; k < hmmc_dist_params.xDim(); k++)
      sum += hmmc_dist_params(k,tc);

    if (sum > 0.0) {
      double inv_sum = 1.0 / sum;
      for (uint k=0; k < hmmc_dist_params.xDim(); k++)
        hmmc_dist_params(k,tc) *= inv_sum;
      hmmc_dist_grouping_param[tc] *= inv_sum;
    }
  }

}

/*virtual*/ void HmmWrapperDoubleClasses::fill_dist_params(uint nSourceClasses, uint nTargetClasses,
    Storage2D<Math1D::Vector<double> >& hmmcc_dist_params,
    Math2D::Matrix<double>& hmmcc_dist_grouping_param) const
{
  if (&dist_params_ != &hmmcc_dist_params) {
    hmmcc_dist_params = dist_params_;
    hmmcc_dist_grouping_param = dist_grouping_param_;
  }
}


long double hmmcc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                 const Storage1D<uint>& target, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                 const SingleWordDictionary& dict, const Storage2D<Math2D::Matrix<double> >& align_model,
                                 const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                 bool with_dict)
{
  const uint I = target.size();
  const uint J = source.size();

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

    const uint sc = sclass[j-1];

    uint prev_aj = alignment[j - 1];
    if (prev_aj >= I)
      prev_aj -= I;

    const uint aj = alignment[j];
    if (aj >= I)
      assert(aj == prev_aj + I);

    if (prev_aj == I)
      prob *= cur_initial_prob[std::min(I, aj)];
    else
      prob *= align_model(sc,tclass[prev_aj])(std::min(I, aj), prev_aj);

    //assert(prob > 0.0);
  }

  return prob;
}

long double hmmcc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                 const Storage1D<uint>& target, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                 const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
                                 const InitialAlignmentProbability& initial_prob, const Storage1D<AlignBaseType>& alignment,
                                 bool with_dict)
{
  const uint I = target.size();
  const uint J = source.size();

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

    const uint sc = sclass[j-1];

    uint prev_aj = alignment[j - 1];
    if (prev_aj >= I)
      prev_aj -= I;

    const uint aj = alignment[j];
    if (aj >= I)
      assert(aj == prev_aj + I);

    if (prev_aj == I)
      prob *= cur_initial_prob[std::min(I, aj)];
    else
      prob *= align_model(sc,std::min(I, aj), prev_aj);

    //assert(prob > 0.0);
  }

  return prob;
}

//no need for a target class dimension, class is detirmened by i_prev
long double hmmcc_alignment_prob(const Storage1D<uint>& source, const SingleLookupTable& slookup,
                                 const Storage1D<uint>& target, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                 const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
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

    const uint sc = sclass[j-1];

    uint prev_aj = alignment[j - 1];
    if (prev_aj >= I)
      prev_aj -= I;

    const uint aj = alignment[j];
    if (aj >= I)
      assert(aj == prev_aj + I);

    if (prev_aj == I)
      prob *= cur_initial_prob[std::min(I, aj)];
    else
      prob *= align_model(sc,std::min(I, aj), prev_aj);

    //assert(prob > 0.0);
  }

  return prob;
}

void par2nonpar_hmmcc_alignment_model(Math1D::Vector<uint>& sclass, Math1D::Vector<uint>& tclass, const Math1D::Vector<double>& source_fert,
                                      const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                                      Storage2D<Math2D::Matrix<double> >& align_model, HmmAlignProbType align_type, bool deficient, int redpar_limit,
                                      uint zero_offset)
{

  align_model.resize(sclass.max()+1,tclass.max()+1);
  const int I = tclass.size();

  for(uint j = 0; j < sclass.size(); j++) {

    const uint sc = sclass[j];

    for (uint i = 0; i < tclass.size(); i++) {

      const uint tc = tclass[i];

      Math2D::Matrix<double>& cur_align_model = align_model(sc,tc);
      const Math1D::Vector<double>& cur_dist_params = dist_params(sc,tc);
      cur_align_model.resize(I+1,I);

      double grouping_norm = std::max<int>(0, i - redpar_limit);
      grouping_norm += std::max<int>(0, int (I) - 1 - (i + redpar_limit));

      double non_zero_sum = 0.0;
      for (int ii = 0; ii < (int)I; ii++) {
        if (align_type != HmmAlignProbReducedpar || abs(ii - i) <= redpar_limit)
          non_zero_sum += cur_dist_params[zero_offset + ii - i];
      }

      if (align_type == HmmAlignProbReducedpar && grouping_norm > 0.0) {
        non_zero_sum += dist_grouping_param(sc,tc);
      }

      if (non_zero_sum > 1e-305) {
        const double inv_sum = (deficient) ? 1.0 : 1.0 / non_zero_sum;

        for (int ii = 0; ii < (int)I; ii++) {
          if (align_type == HmmAlignProbReducedpar && abs(ii - i) > redpar_limit) {
            assert(!isnan(grouping_norm));
            assert(grouping_norm > 0.0);
            cur_align_model(ii, i) = std::max(hmm_min_param_entry, source_fert[1] * inv_sum * dist_grouping_param(sc,tc) / grouping_norm);
          }
          else {
            assert(cur_dist_params[zero_offset + ii - i] >= 0);
            cur_align_model(ii, i) = std::max(hmm_min_param_entry, source_fert[1] * inv_sum * cur_dist_params[zero_offset + ii - i]);
          }
          assert(!isnan(cur_align_model(ii, i)));
          assert(cur_align_model(ii, i) >= 0.0);
        }
        cur_align_model(I, i) = source_fert[0];
        assert(!isnan(cur_align_model(I, i)));
        assert(cur_align_model(I, i) >= 0.0);

#ifndef NDEBUG
        const double sum = cur_align_model.row_sum(i);
        if (!deficient)
          assert(sum >= 0.99 && sum <= 1.01);
#endif
      }
    }
  }
}

//no need for a target class dimension, class is detirmened by i_prev
void par2nonpar_hmmcc_alignment_model(Math1D::Vector<uint>& sclass, Math1D::Vector<uint>& tclass, const Math1D::Vector<double>& source_fert,
                                      const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                                      Math3D::Tensor<double>& align_model, HmmAlignProbType align_type, bool deficient, int redpar_limit,
                                      uint zero_offset)
{
  const int J = sclass.size();
  const int I = tclass.size();
  const uint nSourceClasses = sclass.max() + 1;

  align_model.resize_dirty(nSourceClasses, I+1, I);

  for(uint j = 0; j < sclass.size(); j++) {

    const uint sc = sclass[j];

    for (uint i_prev = 0; i_prev < tclass.size(); i_prev++) {

      const uint tc = tclass[i_prev];

      const Math1D::Vector<double>& cur_dist_params = dist_params(sc,tc);

      double grouping_norm = std::max<int>(0, i_prev - redpar_limit);
      grouping_norm += std::max<int>(0, int (I) - 1 - (i_prev + redpar_limit));

      double non_zero_sum = 0.0;
      for (int ii = 0; ii < (int)I; ii++) {
        if (align_type != HmmAlignProbReducedpar || abs(ii - i_prev) <= redpar_limit)
          non_zero_sum += cur_dist_params[zero_offset + ii - i_prev];
      }

      if (align_type == HmmAlignProbReducedpar && grouping_norm > 0.0) {
        non_zero_sum += dist_grouping_param(sc, tc);
      }

      if (non_zero_sum > 1e-305) {
        const double inv_sum = (deficient) ? 1.0 : 1.0 / non_zero_sum;

        for (int ii = 0; ii < (int) I; ii++) {
          if (align_type == HmmAlignProbReducedpar && abs(ii - i_prev) > redpar_limit) {
            assert(!isnan(grouping_norm));
            assert(grouping_norm > 0.0);
            align_model(sc, ii, i_prev) = std::max(hmm_min_param_entry, source_fert[1] * inv_sum * dist_grouping_param(sc,tc) / grouping_norm);
          }
          else {
            assert(cur_dist_params[zero_offset + ii - i_prev] >= 0);
            align_model(sc, ii, i_prev) = std::max(hmm_min_param_entry, source_fert[1] * inv_sum * cur_dist_params[zero_offset + ii - i_prev]);
          }
          assert(!isnan(align_model(sc, ii, i_prev)));
          assert(align_model(sc, ii, i_prev) >= 0.0);
        }
        align_model(sc, I, i_prev) = source_fert[0];
        assert(!isnan(align_model(sc, I, i_prev)));
        assert(align_model(sc, I, i_prev) >= 0.0);

#ifndef NDEBUG
        double sum = 0.0;
        for (int ii = 0; ii < align_model.yDim(); ii++)
          sum +=  align_model(sc, ii, i_prev);
        if (!deficient)
          assert(sum >= 0.99 && sum <= 1.01);
#endif
      }
    }
  }
}

void init_hmm_from_prev(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const SingleWordDictionary& dict, const CooccuringWordsType& wcooc,
                        const Math1D::Vector<WordClassType>& source_class, const Math1D::Vector<WordClassType>& target_class,
                        Storage2D<Math1D::Vector<double> >& dist_params, Math2D::Matrix<double>& dist_grouping_param,
                        Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                        const HmmOptions& options, uint zero_offset, TransferMode transfer_mode = TransferViterbi, uint maxAllI = 0)
{
  const uint nTargetClasses = target_class.max() + 1;
  const uint nSourceClasses = source_class.max() + 1;
  const size_t nSentences = source.size();

  SingleLookupTable aux_lookup;

  std::set<uint> seenIs; //for initial_prob
  uint maxI = 0;
  for (size_t i = 0; i < nSentences; i++) {
    maxI = std::max<uint>(maxI, target[i].size());
    seenIs.insert(target[i].size());
  }
  maxI = std::max(maxI, maxAllI);

  dist_grouping_param.resize(nSourceClasses, nTargetClasses);
  dist_params.resize(nSourceClasses,nTargetClasses);
  for (uint sc = 0; sc < nSourceClasses; sc++) {
    for (uint tc = 0; tc < nTargetClasses; tc++) {
      dist_params(sc,tc).resize(2*maxI-1);
      dist_params(sc,tc).set_constant(0.0);
    }
  }
  dist_grouping_param.set_constant(0.0);

  const HmmInitProbType init_type = options.init_type_;
  const HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;

  if (init_type == HmmInitPar) {
    init_params.resize(maxI);
    init_params.set_constant(1.0 / maxI);
  }

  if (init_type >= HmmInitInvalid) {
    INTERNAL_ERROR << "invalid type for HMM initial alignment model" << std::endl;
    exit(1);
  }
  if (align_type >= HmmAlignProbInvalid) {
    INTERNAL_ERROR << "invalid type for HMM alignment model" << std::endl;
    exit(1);
  }

  if (align_type == HmmAlignProbReducedpar) {
    dist_grouping_param.set_constant(0.2);

    const double val = 0.8 / (2 * redpar_limit + 1);
    for (uint sc = 0; sc < nSourceClasses; sc++) {
      for (uint tc = 0; tc < nTargetClasses; tc++) {
        if (dist_params(sc,tc).size() > 0) {
          dist_params(sc,tc).set_constant(0.0);
          for (int k = -redpar_limit; k <= redpar_limit; k++) {
            dist_params(sc,tc)[zero_offset + k] = val;
          }
        }
      }
    }
  }
  else {
    dist_grouping_param.set_constant(-1.0);
    for (uint sc = 0; sc < nSourceClasses; sc++)
      for (uint tc = 0; tc < nTargetClasses; tc++)
        dist_params(sc,tc).set_constant(1.0 / dist_params(sc,tc).size());
  }


  initial_prob.resize(maxI);
  for (std::set<uint>::const_iterator it = seenIs.begin(); it != seenIs.end(); it++) {
    const uint I = *it;
    if (!start_empty_word)
      initial_prob[I - 1].resize_dirty(2 * I);
    else
      initial_prob[I - 1].resize_dirty(I + 1);
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
      for (uint i = 0; i < I; i++)
        initial_prob[I - 1][i] = source_fert[1] / I;
      if (!start_empty_word) {
        for (uint i = I; i < 2 * I; i++)
          initial_prob[I - 1][i] = source_fert[0] / I;
      }
      else
        initial_prob[I - 1][I] = source_fert[0];
    }
  }

  if (transfer_mode != TransferNo) {

	//std::cerr << "here" << std::endl;
    const double ibm1_p0 = options.ibm1_p0_;

    if (align_type == HmmAlignProbReducedpar)
      dist_grouping_param.set_constant(0.0);
    for (uint sc = 0; sc < nSourceClasses; sc++) {
      for (uint tc = 0; tc < nTargetClasses; tc++) {
        dist_params(sc, tc).set_constant(0.0);
      }
    }

    for (size_t s = 0; s < nSentences; s++) {

	  //std::cerr << "s: " << s << std::endl;
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

		  //std::cerr << "j: " << j << std::endl;
          int prev_aj = viterbi_alignment[j - 1];
          int cur_aj = viterbi_alignment[j];

          if (prev_aj != 0 && cur_aj != 0) {
            int diff = cur_aj - prev_aj;

            if (abs(diff) <= redpar_limit || align_type != HmmAlignProbReducedpar) {
              //dist_params(zero_offset + diff, target_class[cur_target[prev_aj - 1]]) += 1.0;
              dist_params(source_class[cur_source[j-1]],target_class[cur_target[prev_aj - 1]])[zero_offset + diff]++;
            }
            else {
              //dist_grouping_param[target_class[cur_target[prev_aj - 1]]] += 1.0;
              dist_grouping_param(source_class[cur_source[j-1]],target_class[cur_target[prev_aj - 1]])++;
            }
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
                  dist_params(c, target_class[cur_target[i1]])[zero_offset + diff] += marg;
                else
                  dist_grouping_param(c, target_class[cur_target[i1]]) += marg;
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
                if (abs(diff) <= redpar_limit || align_type != HmmAlignProbReducedpar) {
                  dist_params(source_class[cur_source[j-1]],target_class[cur_target[i1]])[zero_offset + diff]+=marg;
                  ; //dist_params[zero_offset + diff] += marg;
                }
                else {
                  dist_grouping_param(source_class[cur_source[j-1]],target_class[cur_target[i1]]) += marg;
                  ; //dist_grouping_param += marg;
                }
              }
            }
          }
        }
      }
    }

    for (uint sc = 0; sc < nSourceClasses; sc++) {
      for (uint tc = 0; tc < nTargetClasses; tc++) {

        double sum = 0.0;
        for (uint i = 0; i < dist_params.xDim(); i++)
          sum += dist_params(sc,tc).sum();
        if (align_type == HmmAlignProbReducedpar)
          sum += dist_grouping_param(sc,tc);

        if (sum > 1e-300) {
          dist_params(sc, tc) *= 1.0 / sum;
          if (align_type == HmmAlignProbReducedpar) {
            dist_grouping_param(sc, tc) *= 1.0 / sum;
            for (int k = -redpar_limit; k <= redpar_limit; k++)
              dist_params(sc, tc)[zero_offset + k] = 0.75 * dist_params(sc, tc)[zero_offset + k] + 0.25 * 0.8 / (2 * redpar_limit + 1);
            dist_grouping_param(sc, tc) = 0.75 * dist_grouping_param(sc, tc) + 0.25 * 0.2;
          }
          else {
            for (uint k = 0; k < dist_params(sc, tc).size(); k++) {
              dist_params(sc, tc)[k] = 0.75 * dist_params(sc, tc)[k] + 0.25 / dist_params(sc, tc).size();
            }
          }
        }
        else {
		  if (align_type == HmmAlignProbReducedpar) {
            dist_grouping_param(sc, tc) = 0.2;
            dist_params(sc, tc).set_constant(0.0);

            const double val = 0.8 / (2 * redpar_limit + 1);
            for (int k = -redpar_limit; k <= redpar_limit; k++)
              dist_params(sc, tc)[zero_offset + k] = val;
		  }
		  else {
			dist_params(sc,tc).set_constant(1.0 / dist_params(sc,tc).size());
		  }
		}
      }
    }
  }


  if (init_type != HmmInitNonpar)
    par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word, options.fix_p0_);
}

double extended_hmm_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                               const Storage1D<WordClassType>& source_class, const Storage1D<WordClassType>& target_class,
                               const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                               const InitialAlignmentProbability& initial_prob, const Math1D::Vector<double>& source_fert,
                               const SingleWordDictionary& dict, const CooccuringWordsType& wcooc,
                               uint nSourceWords, const HmmOptions& options, uint zero_offset)
{
  HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  const bool deficient = options.deficient_;
  const int redpar_limit = options.redpar_limit_;

  //std::cerr << "start empty word: " << start_empty_word << std::endl;

  double sum = 0.0;

  const size_t nSentences = target.size();

  SingleLookupTable aux_lookup;

  for (size_t s = 0; s < nSentences; s++) {

    const Math1D::Vector<uint>& cur_source = source[s];
    const Math1D::Vector<uint>& cur_target = target[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    const uint curI = cur_target.size();
    const uint curJ = cur_source.size();

    Math1D::Vector<uint> tclass(curI);
    for (uint i = 0; i < curI; i++)
      tclass[i] = target_class[cur_target[i]];
    Math1D::Vector<uint> sclass(curJ);
    for (uint j = 0; j < curJ; j++)
      sclass[j] = source_class[cur_source[j]];

    Math3D::Tensor<double> cur_align_model;
    par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
                                     redpar_limit, zero_offset);

    if (start_empty_word)
      sum -= calculate_sehmmcc_forward_log_sum(cur_source, cur_target, cur_lookup, sclass, tclass, dict, cur_align_model, initial_prob[curI - 1]);
    else
      sum -= calculate_hmmcc_forward_log_sum(cur_source, cur_target, cur_lookup, sclass, tclass, dict, cur_align_model, initial_prob[curI - 1]);
  }

  return sum / nSentences;
}

double extended_hmm_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                           const Storage1D<WordClassType>& sclass, const Storage1D<WordClassType>& tclass,
                           const Storage2D<Math1D::Vector<double> >& dist_params, const Math2D::Matrix<double>& dist_grouping_param,
                           const InitialAlignmentProbability& initial_prob, const SingleWordDictionary& dict, const Math1D::Vector<double>& source_fert,
                           const CooccuringWordsType& wcooc, uint nSourceWords, const floatSingleWordDictionary& prior_weight,
                           const HmmOptions& options, const double dict_weight_sum, uint zero_offset)
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

  energy += extended_hmm_perplexity(source, slookup, target, sclass, tclass, dist_params, dist_grouping_param, initial_prob, source_fert, dict,
                                    wcooc, nSourceWords, options, zero_offset);

  return energy;
}

void train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                        const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                        const Math1D::Vector<WordClassType>& source_class, const Math1D::Vector<WordClassType>& target_class,
                        Storage2D<Math1D::Vector<double> >& dist_params, Math2D::Matrix<double>& dist_grouping_param,
                        Math1D::Vector<double>& source_fert, InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                        SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight, const HmmOptions& options, uint maxAllI)
{
  std::cerr << "starting Extended HMM-DoubleClass EM-training" << std::endl;

  const uint nTargetClasses = target_class.max() + 1;
  const uint nSourceClasses = source_class.max() + 1;

  uint nIterations = options.nIterations_;
  const HmmInitProbType init_type = options.init_type_;
  const HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;
  const uint start_addon = (start_empty_word) ? 1 : 0;
  const bool deficient = options.deficient_;

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

  //uint maxI = 0;
  //for (size_t i = 0; i < nSentences; i++)
  //  maxI = std::max<uint>(maxI, target[i].size());

  uint maxI = maxAllI;
  const uint zero_offset = maxI - 1;

  init_hmm_from_prev(source, slookup, target, dict, wcooc, source_class, target_class, dist_params, dist_grouping_param, source_fert,
                     initial_prob, init_params, options, zero_offset, options.transfer_mode_, maxAllI);

  Math1D::Vector<double> source_fert_count(2);

  InitialAlignmentProbability ficount(maxI, MAKENAME(ficount));
  ficount = initial_prob;

  Math1D::Vector<double> fsentence_start_count(maxI);
  Math1D::Vector<double> fstart_span_count(maxI);

  Storage2D<Math1D::Vector<double> > fsingle_align_count(nSourceClasses,nTargetClasses);
  Math2D::Matrix<double> fgrouping_count(nSourceClasses,nTargetClasses);
  Storage2D<Math2D::Matrix<double> > fspan_align_count(nSourceClasses,nTargetClasses);
  for (uint sc = 0; sc < nSourceClasses; sc++) {
    for (uint tc = 0; tc < nTargetClasses; tc++) {
      fsingle_align_count(sc,tc).resize_dirty(2*maxI-1);
      fspan_align_count(sc,tc).resize_dirty(zero_offset+1,maxI);
    }
  }

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "Starting EHMM-DoubleClass iteration #" << iter << std::endl;

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i = 0; i < options.nTargetWords_; i++) {
      fwcount[i].set_constant(0.0);
    }

    for (uint sc = 0; sc < nSourceClasses; sc++) {
      for (uint tc = 0; tc < nTargetClasses; tc++) {
        fsingle_align_count(sc,tc).set_constant(0.0);
        fspan_align_count(sc,tc).set_constant(0.0);
      }
    }
    fgrouping_count.set_constant(0.0);

    for (uint I = 1; I <= maxI; I++) {
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

      Math1D::Vector<uint> sclass(curJ);
      for (uint j = 0; j < curJ; j++)
        sclass[j] = source_class[cur_source[j]];

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      //std::cerr << "J = " << curJ << ", curI = " << curI << std::endl;

      // Storage2D<Math2D::Matrix<double> > cur_align_model(nSourceClasses,nTargetClasses);
      // par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
      // redpar_limit, zero_offset);

      Math3D::Tensor<double> cur_align_model(nSourceClasses, curI+1, curI);
      par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
                                       redpar_limit, zero_offset);


      const Math1D::Vector<double>& cur_init_prob = initial_prob[curI - 1];
      Math1D::Vector<double>& cur_ficount = ficount[curI - 1];

      Math2D::Matrix<double> cur_dict(curJ,curI+1);
      compute_dictmat(cur_source, cur_lookup, cur_target, dict, cur_dict);

      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2 * curI + start_addon, curJ, MAKENAME(forward));
      calculate_hmmcc_forward(sclass, tclass, cur_dict, cur_align_model, cur_init_prob, options, forward);

      const uint start_s_idx = cur_source[0];

      const long double sentence_prob = forward.row_sum(curJ - 1);
      assert(forward.min() >= 0.0);

      prev_perplexity -= std::log(sentence_prob);

      if (!(sentence_prob > 0.0)) {
        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair " << s << " with I=" << curI << ", J= " << curJ << std::endl;
      }
      assert(sentence_prob > 0.0);

      Math2D::NamedMatrix<long double> backward(2 * curI + start_addon, curJ, MAKENAME(backward));
      calculate_hmmcc_backward(sclass, tclass, cur_dict, cur_align_model, cur_init_prob, options, backward, true);

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
        const uint sc = sclass[j_prev];

        //real positions
        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];

          const double dict_entry = cur_dict(j,i); //dict[t_idx][cur_lookup(j, i)];

          if (dict_entry > 1e-305) {
            fwcount[t_idx][cur_lookup(j, i)] += forward(i, j) * backward(i, j) * inv_sentence_prob / dict_entry;

            const long double bw = backward(i, j) * inv_sentence_prob;

            for (uint i_prev = 0; i_prev < curI; i_prev++) {
              const uint tc = tclass[i_prev];
              long double addon = bw * cur_align_model(sc, i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + curI, j_prev));
              assert(!isnan(addon));
              //cur_facount(i, i_prev, tclass[i_prev]) += addon;

              if (align_type == HmmAlignProbReducedpar && abs( (int) i - (int) i_prev) > redpar_limit)
                fgrouping_count(sc, tc) += addon;
              else
                fsingle_align_count(sc, tc)[zero_offset + i - i_prev] += addon;
              fspan_align_count(sc,tc)(zero_offset - i_prev, curI - i_prev - 1) += addon;
              source_fert_count[1] += addon;
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
          long double addon = bw * cur_align_model(sc, curI, i - curI) * (forward(i, j_prev) + forward(i - curI, j_prev));

          assert(!isnan(addon));

          dict_count_sum += addon;
          //cur_facount(curI, i - curI, tclass[i - curI]) += addon;
          source_fert_count[0] += addon;
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

    } //loop over sentences finished

    double energy = prev_perplexity / nSentences;

    if (dict_weight_sum != 0.0) {
      energy += dict_reg_term(dict, prior_weight, options.l0_beta_);
    }

    std::cerr << "energy after iteration #" << (iter - 1) << ": " << energy << std::endl;
    std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

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

    const double sfsum = source_fert_count.sum();
    if (source_fert.size() > 0 && sfsum > 0.0 && !options.fix_p0_) {

      for (uint i = 0; i < 2; i++) {
        source_fert[i] = source_fert_count[i] / sfsum;
        assert(!isnan(source_fert[i]));
      }

      if (init_type == HmmInitPar) {
        std::cerr << "new probability for zero alignments: " << source_fert[0] << std::endl;
      }
    }

    for (uint sc = 0; sc < nSourceClasses; sc++) {

      std::cerr << "calling m-steps for classes " << sc << ",*" << std::endl;
      for (uint tc = 0; tc < nTargetClasses; tc++) {

        if (dist_params(sc, tc).size() == 0)
          continue;

        ehmm_m_step(fsingle_align_count(sc,tc), fgrouping_count(sc,tc), fspan_align_count(sc,tc), dist_params(sc,tc), zero_offset, dist_grouping_param(sc,tc),
                    deficient, redpar_limit, options.align_m_step_iter_, options.gd_stepsize_, true);
      }
    }

    std::cerr << "calling start m step" << std::endl;

    if (init_type == HmmInitPar) {

      if (options.msolve_mode_ == MSSolvePGD)
        start_prob_m_step(fsentence_start_count,fstart_span_count, init_params, options.init_m_step_iter_, options.gd_stepsize_);
      else
        start_prob_m_step_unconstrained(fsentence_start_count, fstart_span_count, init_params, options.init_m_step_iter_);

      par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word, options.fix_p0_);
    }

    if (init_type == HmmInitNonpar) {
      for (uint I = 1; I <= maxI; I++) {
        double inv_norm = 1.0 / ficount[I - 1].sum();
        for (uint i = 0; i < initial_prob[I - 1].size(); i++)
          initial_prob[I - 1][i] = std::max(hmm_min_param_entry, inv_norm * ficount[I - 1][i]);
      }
    }

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts

    update_dict_from_counts(fwcount, prior_weight, nSentences, dict_weight_sum, options.smoothed_l0_, options.l0_beta_,
                            options.dict_m_step_iter_, dict, hmm_min_dict_entry, options.msolve_mode_ != MSSolvePGD, options.gd_stepsize_);


    // if (options.print_energy_) {
    // std::cerr << "#### EHMM energy after iteration #" << iter << ": "
    // << extended_hmm_energy(source, slookup, target, target_class, align_model, initial_prob, dict, wcooc,
    // nSourceWords, prior_weight, options, dict_weight_sum)
    // << std::endl;
    // }

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
        const uint curJ = source[s].size();

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        Math1D::Vector<uint> sclass(curJ);
        for (uint j = 0; j < curJ; j++)
          sclass[j] = source_class[source[s][j]];
        Math1D::Vector<uint> tclass(curI);
        for (uint i = 0; i < curI; i++)
          tclass[i] = target_class[target[s][i]];

        //std::cerr << "computing alignment" << std::endl;

        Math3D::Tensor<double> cur_align_model;
        par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
                                         redpar_limit, zero_offset);

        compute_ehmmcc_viterbi_alignment(source[s], cur_lookup, target[s], sclass, tclass, dict, cur_align_model,
                                         initial_prob[curI - 1], viterbi_alignment, options);

        //std::cerr << "alignment: " << viterbi_alignment << std::endl;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        sum_daes += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        Storage1D<AlignBaseType> marg_alignment;
        compute_ehmmcc_optmarginal_alignment(source[s], cur_lookup, target[s], sclass, tclass, dict, cur_align_model,
                                             initial_prob[curI - 1], options, marg_alignment);

        //std::cerr << "marg_alignment: " << marg_alignment << std::endl;
        sum_marg_aer += AER(marg_alignment, cur_sure, cur_possible);
        sum_marg_fmeasure += f_measure(marg_alignment, cur_sure, cur_possible);
        sum_marg_daes += nDefiniteAlignmentErrors(marg_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ehmmcc_postdec_alignment(source[s], cur_lookup, target[s], sclass, tclass, dict, cur_align_model,
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

      std::cerr << "#### EHMM-DoubleClass Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### EHMM-DoubleClass Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM-DoubleClass Viterbi-DAE/S after iteration #" << iter << ": " << sum_daes << std::endl;

      std::cerr << "---- EHMM-DoubleClass OptMarg-AER after iteration #" << iter << ": " << sum_marg_aer << " %" << std::endl;
      std::cerr << "---- EHMM-DoubleClass OptMarg-fmeasure after iteration #" << iter << ": " << sum_marg_fmeasure << std::endl;
      std::cerr << "---- EHMM-DoubleClass OptMarg-DAE/S after iteration #" << iter << ": " << sum_marg_daes << std::endl;

      std::cerr << "#### EHMM-DoubleClass Postdec-AER after iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### EHMM-DoubleClass Postdec-fmeasure after iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### EHMM-DoubleClass Postdec-DAE/S after iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  } // loop over iter finished

}

struct HMMParCountStructure {
  int diff_;
  int sc_;
  int tc_;

  HMMParCountStructure(int d, int s, int t) : diff_(d), sc_(s), tc_(t) {}
};

bool operator<(const HMMParCountStructure& s1, const HMMParCountStructure& s2)
{
  if (s1.diff_ != s2.diff_)
    return (s1.diff_ < s2.diff_);
  if (s1.sc_ != s2.sc_)
    return (s1.sc_ < s2.sc_);
  return (s1.tc_ < s2.tc_);
}

void viterbi_train_extended_hmm(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& wcooc,
                                const Math1D::Vector<WordClassType>& source_class, const Math1D::Vector<WordClassType>& target_class,
                                Storage2D<Math1D::Vector<double> >& dist_params, Math2D::Matrix<double>& dist_grouping_param,
                                Math1D::Vector<double>& source_fert_prob, InitialAlignmentProbability& initial_prob, Math1D::Vector<double>& init_params,
                                SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                const HmmOptions& options, const Math1D::Vector<double>& xlogx_table, uint maxAllI)
{
  std::cerr << "starting Extended HMM-DoubleClass Viterbi-training" << std::endl;

  uint nIterations = options.nIterations_;
  const HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  const bool start_empty_word = options.start_empty_word_;
  const int redpar_limit = options.redpar_limit_;
  const uint start_addon = (start_empty_word) ? 1 : 0;
  const bool deficient = options.deficient_;

  if (align_type == HmmAlignProbNonpar || align_type == HmmAlignProbNonpar2) {
	TODO("nonpar");
  }

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
  const uint nSourceClasses = source_class.max() + 1;
  const uint nTargetClasses = target_class.max() + 1;

  SingleWordDictionary dcount(MAKENAME(dcount));
  dcount = dict;

  //uint maxI = 0;
  //for (size_t i = 0; i < nSentences; i++)
  //  maxI = std::max<uint>(maxI, target[i].size());

  uint maxI = maxAllI;
  const uint zero_offset = maxI - 1;

  std::cerr << "calling init" << std::endl;
  init_hmm_from_prev(source, slookup, target, dict, wcooc, source_class, target_class, dist_params, dist_grouping_param, source_fert_prob,
                     initial_prob, init_params, options, zero_offset, options.transfer_mode_, maxAllI);
  std::cerr << "back from init" << std::endl;

  Math1D::Vector<double> source_fert_count(2);

  InitialAlignmentProbability icount(maxI, MAKENAME(ficount));
  icount = initial_prob;

  Math1D::Vector<double> sentence_start_count(maxI);
  Math1D::Vector<double> start_span_count(maxI);

  Storage2D<Math1D::Vector<double> > single_align_count(nSourceClasses,nTargetClasses);
  Math2D::Matrix<double> grouping_count(nSourceClasses,nTargetClasses,0.0);
  Storage2D<Math2D::Matrix<double> > span_align_count(nSourceClasses,nTargetClasses);
  for (uint sc = 0; sc < nSourceClasses; sc++) {
    for (uint tc = 0; tc < nTargetClasses; tc++) {
      single_align_count(sc,tc).resize_dirty(2*maxI-1);
      span_align_count(sc,tc).resize_dirty(maxI+1,maxI+1);
    }
  }

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "Starting EHMM-DoubleClass Viterbi-iteration #" << iter << std::endl;

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i = 0; i < options.nTargetWords_; i++) {
      dcount[i].set_constant(0.0);
    }

    for (uint sc = 0; sc < nSourceClasses; sc++) {
      for (uint tc = 0; tc < nTargetClasses; tc++) {
        single_align_count(sc,tc).set_constant(0.0);
        span_align_count(sc,tc).set_constant(0.0);
      }
    }
    grouping_count.set_constant(0.0);

    for (uint I = 1; I <= maxI; I++) {
      icount[I - 1].set_constant(0.0);
    }

    source_fert_count.set_constant(0.0);

    //these two are calculated from ficount after the loop over the sentences:
    sentence_start_count.set_constant(0.0);
    start_span_count.set_constant(0.0);

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Math1D::Vector<uint>& cur_source = source[s];
      const Math1D::Vector<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      Math1D::Vector<uint> sclass(curJ);
      for (uint j = 0; j < curJ; j++)
        sclass[j] = source_class[cur_source[j]];

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      //std::cerr << "J = " << curJ << ", curI = " << curI << std::endl;

      Math3D::Tensor<double> cur_align_model;
      par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_prob, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
                                       redpar_limit, zero_offset);

      const Math1D::Vector<double>& cur_init_prob = initial_prob[curI - 1];
      Math1D::Vector<double>& cur_icount = icount[curI - 1];

      Storage1D<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      long double prob = compute_ehmmcc_viterbi_alignment(cur_source, cur_lookup, cur_target, sclass, tclass, dict, cur_align_model,
                         cur_init_prob, cur_alignment, options, true, false, 0.0);

      prev_perplexity -= logl(prob);

      if (!(prob > 0.0)) {
        //if (true) {

        std::cerr << "sentence_prob " << prob << " for sentence pair " << s << " with I=" << curI << ", J= " << curJ << std::endl;
      }
      assert(prob > 0.0);

      /**** update counts ****/
      for (uint j = 0; j < curJ; j++) {

        //std::cerr << "j: " << j << std::endl;
        const ushort aj = cur_alignment[j];
        const uint sc = (j > 0) ? sclass[j-1] : MAX_UINT;

        if (aj >= curI) {
          dcount[0][cur_source[j] - 1] += 1;
          //if (j > 0 || init_type == HmmInitPar)
          //  source_fert_count[0] += 1;
        }
        else {
          dcount[cur_target[aj]][cur_lookup(j, aj)] += 1;
          //if (j > 0 || init_type == HmmInitPar)
          //  source_fert_count[1] += 1;
        }

        if (j == 0) {
          if (!start_empty_word) {

            icount[curI - 1][aj] += 1.0;
            if (init_type == HmmInitPar) {
              if (aj >= curI) {
                source_fert_count[0]++;
              }
              else {
                source_fert_count[1]++;
              }
            }
          }
          else {

            if (aj < curI) {
              icount[curI - 1][aj] += 1.0;
              if (init_type == HmmInitPar)
                source_fert_count[1]++;
            }
            else {
              assert(aj == 2 * curI);
              icount[curI - 1][curI] += 1.0;
              if (init_type == HmmInitPar) {
                source_fert_count[0]++;
              }
            }
          }
        }
        else {

          const ushort prev_aj = cur_alignment[j - 1];
          //std::cerr << "prev_aj: " << prev_aj << std::endl;

          if (prev_aj == 2 * curI) {
            assert(start_empty_word);
            if (aj == prev_aj) {
              icount[curI - 1][curI] += 1.0;
              if (init_type == HmmInitPar) {
                source_fert_count[0]++;
              }
            }
            else {
              assert(aj < curI);
              icount[curI - 1][aj] += 1.0;
              if (init_type == HmmInitPar) {
                source_fert_count[1]++;
              }
            }
          }
          else if (prev_aj >= curI) {

            const uint tc = tclass[prev_aj - curI];
            //std::cerr << "here, tc: " << tc << std::endl;

            if (aj >= curI) {
              //cur_facount(curI, prev_aj - curI) += 1.0;
              source_fert_count[0]++;
            }
            else {
              //cur_facount(aj, prev_aj - curI) += 1.0;

              int diff = (int) aj - (int) (prev_aj - curI);

              if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                grouping_count(sc, tc) += 1;
              else
                single_align_count(sc, tc)[zero_offset + diff] += 1;
              span_align_count(sc, tc)(zero_offset - (prev_aj - curI), curI - (prev_aj - curI) - 1) += 1;
              source_fert_count[1]++;
            }
          }
          else { //prev_aj < curI

            const uint tc = tclass[prev_aj];
            if (aj >= curI) {
              //cur_facount(curI, prev_aj) += 1.0;
              source_fert_count[0]++;
            }
            else {
              //cur_facount(aj, prev_aj) += 1.0;

              int diff = (int) aj - (int) prev_aj;

              if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                grouping_count(sc, tc) += 1;
              else
                single_align_count(sc, tc)[zero_offset + diff] += 1;
              span_align_count(sc, tc)(zero_offset - prev_aj, curI - prev_aj - 1) += 1;
              source_fert_count[1]++;
            }
          }
        }
      }
    } // loop over sentences finished

    //finish counts

    if (init_type == HmmInitPar) {

      for (uint I = 1; I <= maxI; I++) {

        if (initial_prob[I - 1].size() != 0) {
          for (uint i = 0; i < I; i++) {

            const double cur_count = icount[I - 1][i];
            //source_fert_count[1] += cur_count; //was increased above
            sentence_start_count[i] += cur_count;
            start_span_count[I - 1] += cur_count;
          }
          for (uint i = I; i < icount[I - 1].size(); i++) {
            //source_fert_count[0] += icount[I - 1][i]; //was increased above
          }
        }
      }
    }

    update_dict_from_counts(dcount, prior_weight, nSentences, 0.0, false, 0.0, 0, dict, hmm_min_dict_entry,
                            options.msolve_mode_ != MSSolvePGD, options.gd_stepsize_);

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

    double sfsum = source_fert_count.sum();
    if (sfsum > 1e-305 && !options.fix_p0_) {
      for (uint k = 0; k < 2; k++)
        source_fert_prob[k] = source_fert_count[k] / sfsum;
    }

    if (init_type == HmmInitPar) {
      start_prob_m_step(sentence_start_count, start_span_count, init_params, options.init_m_step_iter_, options.gd_stepsize_);
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

    for (uint sc = 0; sc < nSourceClasses; sc++) {

      std::cerr << "calling m-steps for classes " << sc << ",*" << std::endl;
      for (uint tc = 0; tc < nTargetClasses; tc++) {

        //std::cerr << "tc: " << tc << std::endl;
        //std::cerr << "size: " << single_align_count(sc, tc).size() << std::endl;
        if (dist_params(sc, tc).size() == 0)
          continue;

        //std::cerr << "calling" << std::endl;
        ehmm_m_step(single_align_count(sc,tc), grouping_count(sc,tc), span_align_count(sc,tc), dist_params(sc,tc), zero_offset, dist_grouping_param(sc,tc),
                    deficient, redpar_limit, options.align_m_step_iter_, options.gd_stepsize_, true);
      }
    }

    std::cerr << "source_fert_count before ICM: " << source_fert_count << std::endl;

#if 1
    std::cerr << "starting ICM stage" << std::endl;

    /**** ICM stage ****/

    uint nSwitches = 0;

    Math2D::Matrix<double> dist_count_sum(nSourceClasses, nTargetClasses);
    for (uint sc = 0; sc < nSourceClasses; sc++)
      for (uint tc = 0; tc < nTargetClasses; tc++)
        dist_count_sum(sc, tc) = single_align_count(sc, tc).sum() + grouping_count(sc, tc);

    Math1D::Vector<uint> dict_sum(dcount.size());
    for (uint k = 0; k < dcount.size(); k++)
      dict_sum[k] = dcount[k].sum();

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "###############s: " << s << ", J; " << source[s].size() << ", I: " << target[s].size() << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const ushort curJ = cur_source.size();
      const ushort curI = cur_target.size();
      const ushort nLabels = (start_empty_word) ? 2 * curI + 1 : 2 * curI;

      Math1D::Vector<uint> sclass(curJ);
      for (uint j = 0; j < curJ; j++)
        sclass[j] = source_class[cur_source[j]];

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      Storage2D<Math2D::Matrix<double> > cur_align_model(nSourceClasses,nTargetClasses);
      par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_prob, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
                                       redpar_limit, zero_offset);

      const Math1D::Vector<double>& cur_initial_prob = initial_prob[curI - 1];
      Math1D::Vector<double>& cur_icount = icount[curI - 1];

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const bool implicit = (deficient &&
                             (align_type == HmmAlignProbFullpar || (align_type == HmmAlignProbReducedpar && curI <= redpar_limit)));

      for (uint j = 0; j < curJ; j++) {

        if (s == 100000)
          std::cerr << "###j: " << j << std::endl;
        //std::cerr << "source_fert_count: " << source_fert_count << std::endl;
        const uint sc1 = (j > 0) ? sclass[j - 1] : MAX_UINT;

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

        const uint tc1 = (effective_prev_aj < curI) ? tclass[effective_prev_aj] : MAX_UINT;
        const uint tc2_old = (effective_cur_aj < curI) ? tclass[effective_cur_aj] : MAX_UINT;

        uint effective_next_aj = MAX_UINT;
        uint jj = j + 1;
        for (; jj < curJ; jj++) {
          if (cur_alignment[jj] < curI) {
            effective_next_aj = cur_alignment[jj];
            break;
          }
        }

        const uint sc2 = (jj < curJ) ? sclass[jj-1] : MAX_UINT;
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

          if (s == 100000)
            std::cerr << "##i: " << i << std::endl;

          if (i == cur_aj)
            continue;
          if (start_empty_word && j == 0 && i >= curI && i < 2 * curI)
            continue;
          if (!start_empty_word && j == 0 && i > curI)
            continue;

          if (s == 100000) {
            std::cerr << "cur_aj: " << cur_aj << std::endl;
            std::cerr << "base alignment: " << cur_alignment << std::endl;
          }

          ushort prev_aj = (j > 0) ? cur_alignment[j - 1] : MAX_UINT;

          if (j > 0 && i >= curI) {

            if (i != prev_aj && i != prev_aj + curI)
              continue;
            if (i == 2 * curI && prev_aj != i)  //need this!
              continue;
          }

          bool fix = true;

          uint effective_i = i;
          if (effective_i >= curI)
            effective_i -= curI;

          const uint tc2_new = (effective_i < curI) ? tclass[effective_i] : MAX_UINT;

          // if (j + 1 < curJ) {

          // const uint next_aj = cur_alignment[j + 1];

          // if (deficient) {

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

          std::map<HMMParCountStructure, int> dist_count_change;
          std::map<std::pair<int,int>, int> total_dist_count_change;
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

          //a) changes regarding preceeding pos and alignments to NULL
          //std::cerr << "a) terms regarding preceeding pos, effective_prev_aj: " << effective_prev_aj << std::endl;
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

            if (true) { //parametric model

              //std::cerr << "leaving terms" << std::endl;
              if (implicit) {
                if (effective_prev_aj < curI) {

                  if (cur_aj < curI) {
                    const uint tc = tclass[effective_prev_aj];
                    const uint sc = sclass[j-1];
                    dist_count_change[HMMParCountStructure(zero_offset + cur_aj - effective_prev_aj,sc,tc)]--;
                    total_dist_count_change[std::make_pair(sc,tc)]--;
                  }
                  else {
                    if (cur_aj < 2*curI || init_type == HmmInitPar)
                      change -= -std::log(source_fert_prob[0]);
                    else
                      change += -std::log(cur_initial_prob[curI]);
                  }
                }
                else {
                  change -= -std::log(cur_initial_prob[std::min<ushort>(curI, cur_aj)]);
                }
              }
              else if (!fix && deficient) {
                if (cur_aj < curI) {

                  int diff = cur_aj - effective_prev_aj;
                  int cur_c = 0;
                  if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                    cur_c = grouping_count(sc1, tc1);
                  else
                    cur_c = single_align_count(sc1, tc1)[zero_offset + diff];

                  if (cur_c > 1) {
                    //exploit that log(1) = 0
                    change -= -xlogx_table[cur_c];
                    change += -xlogx_table[cur_c - 1];
                  }
                }
              }
              else {
                if (effective_prev_aj < curI)
                  change -= -std::log(cur_align_model(sc1, tc1)(std::min<ushort>(curI, cur_aj), effective_prev_aj));
                else
                  change -= -std::log(cur_initial_prob[effective_cur_aj]);
              }

              //std::cerr << "entering terms" << std::endl;
              if (implicit) {
                if (effective_prev_aj < curI) {
                  if (i < curI) {
                    const uint tc = tclass[effective_prev_aj];
                    const uint sc = sclass[j-1];
                    dist_count_change[HMMParCountStructure(zero_offset + i - effective_prev_aj,sc,tc)]++;
                    total_dist_count_change[std::make_pair(sc,tc)]++;
                  }
                  else {
                    if (i < 2*curI || init_type == HmmInitPar)
                      change += -std::log(source_fert_prob[0]);
                    else
                      change += -std::log(cur_initial_prob[curI]);
                  }
                }
                else {
                  change += -std::log(cur_initial_prob[std::min<ushort>(curI, i)]);
                }
              }
              else if (!fix && deficient) {

                if (i < curI) {

                  int diff = i - effective_prev_aj;
                  int cur_c = 0;
                  if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                    cur_c = grouping_count(sc1, tc1);
                  else
                    cur_c = single_align_count(sc1, tc1)[zero_offset + diff];
                  if (cur_c > 0) {
                    //exploit that log(1) = 0
                    change -= -xlogx_table[cur_c];
                    change += -xlogx_table[cur_c + 1];
                  }
                }
              }
              else {
                if (effective_prev_aj < curI)
                  change += -std::log(cur_align_model(sc1, tc1)(std::min(curI, i), effective_prev_aj));
                else
                  change += -std::log(cur_initial_prob[effective_i]);
              }

              //source fertility counts
              if (!fix && deficient) {
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

                  change -= xlogx_table[dist_count_sum(sc1, tc1)];
                  change += xlogx_table[dist_count_sum(sc1, tc1) - 1];
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

                  change -= xlogx_table[dist_count_sum(sc1, tc1)];
                  change += xlogx_table[dist_count_sum(sc1, tc1) + 1];
                }
              }
            }
          }

          assert(!isnan(change));

          //std::cerr << "b) terms regarding succceeding pos" << std::endl;
          //std::cerr << "jj: " << jj << ", effective_next_aj: " << effective_next_aj << std::endl;

          //b) changes regarding succeeding pos
          if (j+1 < curJ && effective_cur_aj != effective_i) {

            if (true) {
              //parametric model

              if (implicit) {

                const uint sc = sclass[jj-1];

                if (effective_cur_aj < curI) {
                  if (effective_next_aj < curI) {
                    const uint tc1 = tclass[effective_cur_aj];
                    dist_count_change[HMMParCountStructure(zero_offset + effective_next_aj - effective_cur_aj,sc,tc1)]--;
                    total_dist_count_change[std::make_pair(sc,tc1)]--;
                  }
                }
                else {
                  change -= -std::log(cur_initial_prob[curI]);
                }

                if (effective_i < curI) {
                  if (effective_next_aj < curI) {
                    const uint tc2 = tclass[effective_i];
                    dist_count_change[HMMParCountStructure(zero_offset + effective_next_aj - effective_i,sc,tc2)]++;
                    total_dist_count_change[std::make_pair(sc,tc2)]++;
                  }
                }
                else {
                  change += -std::log(cur_initial_prob[curI]);
                }
              }
              else if (!fix && deficient) {
                assert(j + 1 < curJ);
                assert(effective_i < curI && effective_cur_aj < curI);

                if (cur_alignment[j + 1] < curI) {

                  const int diff1 = effective_next_aj - effective_cur_aj;
                  int cur_c = 0;
                  if (align_type == HmmAlignProbReducedpar && abs(diff1) > redpar_limit)
                    cur_c = grouping_count(sc2, tc2_old);
                  else
                    cur_c = single_align_count(sc2, tc2_old)[zero_offset + diff1];

                  int diff2 = effective_next_aj - effective_i;
                  int new_c = 0;
                  if (align_type == HmmAlignProbReducedpar && abs(diff2) > redpar_limit)
                    new_c = grouping_count(sc2, tc2_new);
                  else
                    new_c = single_align_count(sc2, tc2_new)[zero_offset + diff2];

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
              else { //non-deficient

                if (effective_next_aj < curI) {
                  if (effective_cur_aj != curI) {
                    change -= -std::log(cur_align_model(sc2, tc2_old)(effective_next_aj, effective_cur_aj));
                    if (effective_i != curI)
                      change += -std::log(cur_align_model(sc2, tc2_new)(effective_next_aj, effective_i));
                    else
                      change += -std::log(cur_initial_prob[effective_next_aj]);
                  }
                  else {
                    change -= -std::log(cur_initial_prob[effective_next_aj]);
                    change += -std::log(cur_align_model(sc2, tc2_new)(effective_next_aj, effective_i));
                  }
                }
              }
            }
          }

          for (std::map<HMMParCountStructure,int>::const_iterator it = dist_count_change.begin(); it != dist_count_change.end(); it++) {
            const int diff = it->second;
            if (diff == 0)
              continue;
            const int offs = it->first.diff_;
            const int sc = it->first.sc_;
            const int tc = it->first.tc_;

            int count = single_align_count(sc, tc)[offs];
            change -= -xlogx_table[count];
            assert(count+diff >= 0);
            change += -xlogx_table[count+diff];
          }
          for (std::map<std::pair<int,int>, int>::const_iterator it = total_dist_count_change.begin(); it != total_dist_count_change.end(); it++) {
            const int diff = it->second;
            if (diff == 0)
              continue;
            const int sc = it->first.first;
            const int tc = it->first.second;
            const int count = dist_count_sum(sc,tc);
            change -= xlogx_table[count];
            assert(count+diff >= 0);
            change += xlogx_table[count+diff];
          }

          assert(!isnan(change));

          if (change < best_change) {

            best_change = change;
            new_aj = i;
          }
        }

        //if (true) {
        if (s == 100000) {
          std::cerr << "base alignment: " << cur_alignment << std::endl;
          std::cerr << "best_change: " << best_change << std::endl;
        }

        if (best_change < -0.01 && new_aj != cur_aj) {

          if (s == 100000) {
            //if (true) {
            std::cerr << "base alignment: " << cur_alignment << std::endl;
            std::cerr << "switching a" << j << " from " << cur_aj << " to " << new_aj << std::endl;
            std::cerr << "initial source_fert_count: " << source_fert_count << std::endl;
            std::cerr << "initial icount: " << cur_icount << std::endl;
          }

          nSwitches++;

          const uint new_target_word = (new_aj >= curI) ? 0 : cur_target[new_aj];
          const uint hyp_idx = (new_aj >= curI) ? cur_source[j] - 1 : cur_lookup(j, new_aj);

          ushort effective_new_aj = new_aj;
          if (effective_new_aj >= curI)
            effective_new_aj -= curI;

          cur_alignment[j] = new_aj;

          if (j > 0 && new_aj >= curI)
            assert(new_aj == cur_alignment[j - 1] || new_aj - curI == cur_alignment[j - 1]);

          const uint tc2_old = (effective_cur_aj < curI) ? tclass[effective_cur_aj] : MAX_UINT;
          const uint tc2_new = (effective_new_aj < curI) ? tclass[effective_new_aj] : MAX_UINT;

          // if (j > 0 || init_type == HmmInitPar) {
          // if (cur_aj >= curI) {
          // assert(new_aj < curI);
          // source_fert_count[0]--;
          // source_fert_count[1]++;
          // if (s == 100000)
          // std::cerr << "A, source_fert_count: " << source_fert_count << std::endl;
          // }
          // else if (new_aj >= curI) {
          // assert(cur_aj < curI);
          // source_fert_count[0]++;
          // source_fert_count[1]--;
          // if (s == 100000)
          // std::cerr << "B, source_fert_count: " << source_fert_count << std::endl;
          // }
          // }

          if (j + 1 < curJ) {

            const ushort next_aj = cur_alignment[j + 1];

            if (s == 100000)
              std::cerr << "next_aj: " << next_aj << std::endl;

            if (next_aj >= curI) {

              ushort new_next_aj = (new_aj < curI) ? new_aj + curI : new_aj;

              if (s == 100000)
                std::cerr << "new_next_aj: " << new_next_aj << std::endl;

              if (new_next_aj != next_aj) {

                for (uint jjj = j+1; jjj < jj; jjj++) {
                  assert(cur_alignment[jjj] == next_aj);
                  if (next_aj == 2*curI) {
                    cur_icount[curI]--;
                    //cur_acount(curI,effective_new_aj, tclass[effective_new_aj])++;

                    //source_fert_count[0] remains active exactly once if init is par
                    if (init_type == HmmInitPar)
                      source_fert_count[0]--;
                    if (s == 100000)
                      std::cerr << "C1, jjj: " << jjj << ", icount: " << cur_icount << ", sfc: " << source_fert_count << std::endl;
                  }
                  else {
                    //cur_acount(curI, next_aj - curI, tclass[next_aj - curI])--;
                    source_fert_count[0]--;
                  }
                  cur_alignment[jjj] = new_next_aj;

                  if (new_next_aj == 2*curI) {
                    cur_icount[curI]++;
                    if (init_type == HmmInitPar)
                      source_fert_count[0]++;
                    if (s == 100000)
                      std::cerr << "C2 increase icount " << curI << ", new source_fert_count: " << source_fert_count << std::endl;
                  }
                  else {
                    //cur_acount(curI,effective_new_aj)++;
                    source_fert_count[0]++;
                    if (s == 100000)
                      std::cerr << "D increase acount " << curI << ", " << effective_new_aj
                                << ", new source_fert_count " << source_fert_count<< std::endl;
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

            if (cur_aj != 2 * curI) {
              assert(cur_icount[cur_aj] > 0);
              cur_icount[cur_aj]--;
              if (init_type == HmmInitPar) {
                if (cur_aj < curI)
                  source_fert_count[1]--;
                else
                  source_fert_count[0]--;
              }
              if (s == 100000)
                std::cerr << "E, icount: " << cur_icount << ", sfc: " << source_fert_count << std::endl;
            }
            else {
              assert(cur_icount[curI] > 0);
              cur_icount[curI]--;
              if (init_type == HmmInitPar)
                source_fert_count[0]--;
              if (s == 100000)
                std::cerr << "F, icount: " << cur_icount << ", sfc: " << source_fert_count << std::endl;
            }

            if (new_aj != 2 * curI) {
              cur_icount[new_aj]++;
              if (init_type == HmmInitPar) {
                if (new_aj < curI)
                  source_fert_count[1]++;
                else
                  source_fert_count[0]++;
              }
              if (s == 100000)
                std::cerr << "G" << std::endl;
            }
            else {
              cur_icount[curI]++;
              if (init_type == HmmInitPar)
                source_fert_count[0]++;
              if (s == 100000)
                std::cerr << "H, icount: " << cur_icount << ", sfc: " << source_fert_count << std::endl;
            }
          }
          else { //there is a previous source word

            if (s == 100000)
              std::cerr << "effective_prev_aj: " << effective_prev_aj << std::endl;
            if (effective_prev_aj != curI) {

              //leaving
              if (cur_aj < curI) {
                const int diff = (int) cur_aj - (int) effective_prev_aj;
                if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                  grouping_count(sc1, tc1)--;
                else
                  single_align_count(sc1, tc1)[zero_offset + diff]--;
                span_align_count(sc1, tc1)(zero_offset - effective_prev_aj, curI - effective_prev_aj - 1)--;
                if (s == 100000)
                  std::cerr << "I" << std::endl;
              }

              //entering
              if (new_aj < curI) {
                const int diff = (int) new_aj - (int) effective_prev_aj;
                if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                  grouping_count(sc1, tc1)++;
                else
                  single_align_count(sc1, tc1)[zero_offset + diff]++;
                span_align_count(sc1, tc1)(zero_offset - effective_prev_aj, curI - effective_prev_aj - 1)++;
                if (s == 100000)
                  std::cerr << "J" << std::endl;
              }


              if (align_type != HmmAlignProbNonpar) {
                if (cur_aj < curI && new_aj >= curI) {
                  if (s == 100000)
                    std::cerr << "N increase sf0, decrease sf1" << std::endl;
                  source_fert_count[1]--;
                  source_fert_count[0]++;
                }
                else if (cur_aj >= curI && new_aj < curI) {
                  //if (s == 10000096 || s == 1000003296 || s == 9)
                  //std::cerr << "O decrease sf0, increase sf1" << std::endl;
                  source_fert_count[0]--;
                  source_fert_count[1]++;
                }
              }
            }
            else {
              if (s == 100000)
                std::cerr << "K" << std::endl;
              cur_icount[std::min<ushort>(curI, cur_aj)]--;
              cur_icount[std::min<ushort>(curI, new_aj)]++;

              if (init_type == HmmInitPar && std::max(cur_aj,new_aj) >= curI) {
                if (cur_aj >= curI) {
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

            if (deficient && effective_prev_aj < curI) {

              if (cur_aj < curI) {
                //single_align_count and grouping_count were handled above already
                dist_count_sum(sc1,tc1)--;
              }
              if (new_aj < curI) {
                //single_align_count and grouping_count were handled above already
                dist_count_sum(sc1,tc1)++;
              }
            }
          }

          if (s == 100000)
            std::cerr << "b), jj: " << jj << std::endl;

          //b) dependency to succceeding pos
          if (jj < curJ) {

            const ushort next_aj = cur_alignment[jj];
            assert(next_aj < curI);
            if (s == 100000)
              std::cerr << "next_aj: " << next_aj << std::endl;
            assert(next_aj < curI);
            ushort effective_cur_aj = cur_aj;
            if (effective_cur_aj >= curI)
              effective_cur_aj -= curI;
            ushort effective_new_aj = new_aj;
            if (effective_new_aj >= curI)
              effective_new_aj -= curI;

            if (s == 100000)
              std::cerr << "leaving, effective_cur_aj: " << effective_cur_aj << std::endl;
            //leaving
            if (effective_cur_aj < curI) {
              if (next_aj < curI) {
                const int diff = (int) next_aj - (int) effective_cur_aj;
                if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                  grouping_count(sc2, tc2_old)--;
                else
                  single_align_count(sc2,tc2_old)[zero_offset + diff]--;
                span_align_count(sc2,tc2_old)(zero_offset - effective_cur_aj, curI - effective_cur_aj - 1)--;
                source_fert_count[1]--;
                if (s == 100000)
                  std::cerr << "changed source fert count: " << source_fert_count << std::endl;

                if (s == 100000)
                  std::cerr << "L" << std::endl;
              }
              else
                source_fert_count[0]--;
            }
            else {
              cur_icount[std::min(curI, next_aj)]--;
              if (init_type == HmmInitPar) {
                if (next_aj < curI)
                  source_fert_count[1]--;
                else
                  source_fert_count[0]--;
              }
              if (s == 100000)
                std::cerr << "M, icount: " << cur_icount << ", sfc: " << source_fert_count << std::endl;
            }

            if (s == 100000)
              std::cerr << "entering, effective_new_aj: " << effective_new_aj << std::endl;
            //entering
            if (effective_new_aj < curI) {
              assert(next_aj < curI);
              if (next_aj < curI) {
                const int diff = (int) next_aj - (int) effective_new_aj;
                if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                  grouping_count(sc2, tc2_new)++;
                else
                  single_align_count(sc2,tc2_new)[zero_offset + diff]++;
                span_align_count(sc2,tc2_new)(zero_offset - effective_new_aj, curI - effective_new_aj-1)++;
                source_fert_count[1]++;

                if (s == 100000)
                  std::cerr << "N, sfc: " << source_fert_count << std::endl;
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
              if (s == 100000)
                std::cerr << "O, icount: " << cur_icount << std::endl;
            }

            if (deficient) {
              if (next_aj < curI) {
                if (effective_cur_aj < curI) {
                  //single_align_count and grouping_count were handled above already
                  dist_count_sum(sc2,tc2_old)--;
                }
                if (effective_new_aj < curI) {
                  //single_align_count and grouping_count were handled above alreadynew_aj]++;
                  dist_count_sum(sc2,tc2_new)++;
                }
              }
            }
          }
//#if 0
#ifndef NDEBUG
          //check counts
          if ((s%750) == 0) { // || (s >= 100 && s <= 200)) {

            std::cerr << "checks for sentence #" << s << std::endl;

            SingleWordDictionary check_dcount(MAKENAME(check_dcount));
            check_dcount = dict;
            for (uint i=0; i < check_dcount.size(); i++)
              check_dcount[i].set_constant(0.0);

            Math1D::Vector<double> check_source_fert_count(2,0.0);;

            InitialAlignmentProbability check_icount(maxI, MAKENAME(ficount));
            check_icount = initial_prob;
            for (uint I = 0; I < maxI; I++)
              check_icount[I].set_constant(0.0);

            Math1D::Vector<double> check_sentence_start_count(maxI,0.0);
            Math1D::Vector<double> check_start_span_count(maxI,0.0);

            Storage2D<Math1D::Vector<double> > check_single_align_count(nSourceClasses,nTargetClasses);
            Math2D::Matrix<double> check_grouping_count(nSourceClasses,nTargetClasses,0.0);
            Storage2D<Math2D::Matrix<double> > check_span_align_count(nSourceClasses,nTargetClasses);
            for (uint sc = 0; sc < nSourceClasses; sc++) {
              for (uint tc = 0; tc < nTargetClasses; tc++) {
                check_single_align_count(sc,tc).resize(2*maxI-1,0.0);
                check_span_align_count(sc,tc).resize(maxI+1,maxI+1,0.0);
              }
            }

            uint nSourceFert1 = 0;

            for (size_t s = 0; s < nSentences; s++) {

              const Math1D::Vector<uint>& cur_source = source[s];
              const Math1D::Vector<uint>& cur_target = target[s];
              const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

              const uint curJ = cur_source.size();
              const uint curI = cur_target.size();

              Math1D::Vector<uint> sclass(curJ);
              for (uint j = 0; j < curJ; j++)
                sclass[j] = source_class[cur_source[j]];

              Math1D::Vector<uint> tclass(curI);
              for (uint i = 0; i < curI; i++)
                tclass[i] = target_class[cur_target[i]];

              const Math1D::Vector<double>& cur_init_prob = initial_prob[curI - 1];
              Math1D::Vector<double>& cur_icount = icount[curI - 1];

              Storage1D<AlignBaseType>& cur_alignment = viterbi_alignment[s];

              //if (s == 100000)
              //std::cerr << "----cur_alignment: " << cur_alignment << std::endl;

              /**** update counts ****/
              for (uint j = 0; j < curJ; j++) {

                //if (s == 100000)
                //std::cerr << "---j: " << j << ", aj: " << cur_alignment[j] << std::endl;
                const ushort aj = cur_alignment[j];
                const uint sc = (j > 0) ? sclass[j-1] : MAX_UINT;

                if (aj >= curI) {
                  check_dcount[0][cur_source[j] - 1] += 1;
                }
                else {
                  check_dcount[cur_target[aj]][cur_lookup(j, aj)] += 1;
                }

                if (j == 0) {
                  if (!start_empty_word) {
                    check_icount[curI - 1][aj] += 1.0;
                    if (init_type == HmmInitPar) {
                      if (aj >= curI) {
                        check_source_fert_count[0]++;
                        //if (s == 100000) {
                        //std::cerr << "---inc sfc0" << std::endl;
                        ///}
                      }
                      else {
                        check_source_fert_count[1]++;
                        // if (s == 100000) {
                        // nSourceFert1++;
                        // std::cerr << "--- inc sfc1" << std::endl;
                        // }
                      }
                    }
                  }
                  else {
                    //if (s == 100000)
                    //std::cerr << "--- j=0, start_empty_word" << std::endl;
                    if (aj < curI) {
                      check_icount[curI - 1][aj] += 1.0;
                      if (init_type == HmmInitPar) {
                        check_source_fert_count[1]++;
                        //if (s == 100000) {
                        //nSourceFert1++;
                        //std::cerr << "--- inc sfc1" << std::endl;
                        //}
                      }
                    }
                    else {
                      assert(aj == 2 * curI);
                      check_icount[curI - 1][curI] += 1.0;
                      if (init_type == HmmInitPar) {
                        check_source_fert_count[0]++;
                      }
                    }
                  }
                }
                else { // j > 0

                  //if (s == 100000)
                  //std::cerr << "---- j: " << j << ", prev_aj: " << cur_alignment[j-1] << std::endl;
                  const ushort prev_aj = cur_alignment[j - 1];

                  if (prev_aj == 2 * curI) {
                    assert(start_empty_word);
                    if (aj == prev_aj) {
                      check_icount[curI - 1][curI] += 1.0;
                      if (init_type == HmmInitPar) {
                        check_source_fert_count[0]++;
                      }
                    }
                    else {
                      assert(aj < curI);
                      check_icount[curI - 1][aj] += 1.0;
                      if (init_type == HmmInitPar) {
                        check_source_fert_count[1]++;
                        if (s == 100000) {
                          nSourceFert1++;
                          //std::cerr << "--- prev_aj = 2*I, inc sfc1" << std::endl;
                        }
                      }
                    }
                  }
                  else if (prev_aj >= curI) {

                    const uint tc = tclass[prev_aj - curI];

                    //if (s == 100000)
                    //std::cerr << "---prev_aj = " << prev_aj << " >= I, aj: " << aj << std::endl;

                    if (aj >= curI) {
                      check_source_fert_count[0]++;
                      //if (s == 100000)
                      //std::cerr << "aj > I, inc sfc0" << std::endl;
                    }
                    else {
                      //cur_facount(aj, prev_aj - curI) += 1.0;

                      int diff = (int) aj - (int) (prev_aj - curI);

                      if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                        check_grouping_count(sc, tc) += 1;
                      else
                        check_single_align_count(sc, tc)[zero_offset + diff] += 1;
                      check_span_align_count(sc, tc)(zero_offset - (prev_aj - curI), curI - (prev_aj - curI) - 1) += 1;
                      check_source_fert_count[1]++;
                      // if (s == 100000) {
                      // std::cerr << "---inc sfc1" << std::endl;
                      // nSourceFert1++;
                      // }
                    }
                  }
                  else { //prev_aj < curI

                    //if (s == 100000)
                    //std::cerr << "---prev_aj = " << prev_aj << " < I, aj: " << aj << std::endl;

                    const uint tc = tclass[prev_aj];
                    if (aj >= curI) {
                      check_source_fert_count[0]++;
                      //if (s == 100000)
                      //std::cerr << "--- inc sfc0" << std::endl;
                    }
                    else {
                      //cur_facount(aj, prev_aj) += 1.0;

                      int diff = (int) aj - (int) prev_aj;

                      if (align_type == HmmAlignProbReducedpar && abs(diff) > redpar_limit)
                        check_grouping_count(sc, tc) += 1;
                      else
                        check_single_align_count(sc, tc)[zero_offset + diff] += 1;
                      check_span_align_count(sc, tc)(zero_offset - prev_aj, curI - prev_aj - 1) += 1;
                      check_source_fert_count[1]++;
                      //if (s == 100000) {
                      //std::cerr << "---inc sfc1" << std::endl;
                      //nSourceFert1++;
                      //}
                    }
                  }
                }
              }
            }

            //if (s == 100000)
            //std::cerr << "---nSourceFert1 for sentence 196: " << nSourceFert1 << std::endl;


            //now check
            if (!(grouping_count == check_grouping_count)) {
              std::cerr << "grouping count should be " << check_grouping_count << ", is " << grouping_count << std::endl;
            }
            assert(grouping_count == check_grouping_count);
            assert(single_align_count == check_single_align_count);
            if (!(span_align_count == check_span_align_count)) {
              for (uint x=0; x < span_align_count.xDim(); x++) {
                for (uint y=0; y < span_align_count.yDim(); y++) {
                  if (span_align_count(x,y) != check_span_align_count(x,y)) {
                    std::cerr << "span align count for x=" << x << " and y=" << y << " should be "
                              << check_span_align_count(x,y) << ", is " << span_align_count(x,y) << std::endl;
                  }
                }
              }
            }
            assert(span_align_count == check_span_align_count);

            if (check_source_fert_count != source_fert_count) {
              std::cerr << "source fert count should be " << check_source_fert_count << ", is " << source_fert_count << std::endl;
            }
            assert(check_source_fert_count == source_fert_count);

            for (uint i = 0; i < options.nTargetWords_; i++) {
              assert(check_dcount[i] == dcount[i]);
            }

            for (uint I = 0; I < check_icount.size(); I++) {
              if (check_icount[I] != icount[I]) {
                std::cerr << "icount[" << I << "] should be " << std::endl
                          << check_icount[I] << ", is " << std::endl << icount[I] << std::endl;
              }
            }
            assert(check_icount == icount);
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
      start_prob_m_step(sentence_start_count, start_span_count, init_params, options.init_m_step_iter_, options.gd_stepsize_);
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

    for (uint sc = 0; sc < nSourceClasses; sc++) {

      std::cerr << "calling m-steps for classes " << sc << ",*" << std::endl;
      for (uint tc = 0; tc < nTargetClasses; tc++) {

        if (dist_params(sc, tc).size() == 0)
          continue;

        ehmm_m_step(single_align_count(sc,tc), grouping_count(sc,tc), span_align_count(sc,tc), dist_params(sc,tc), zero_offset, dist_grouping_param(sc,tc),
                    deficient, redpar_limit, options.align_m_step_iter_, options.gd_stepsize_, true);
      }
    }

    double energy = 0.0;
    for (size_t s = 0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      const ushort curJ = cur_source.size();
      const ushort curI = cur_target.size();
      const ushort nLabels = (start_empty_word) ? 2 * curI + 1 : 2 * curI;

      Math1D::Vector<uint> sclass(curJ);
      for (uint j = 0; j < curJ; j++)
        sclass[j] = source_class[cur_source[j]];

      Math1D::Vector<uint> tclass(curI);
      for (uint i = 0; i < curI; i++)
        tclass[i] = target_class[cur_target[i]];

      Storage2D<Math2D::Matrix<double> > cur_align_model(nSourceClasses,nTargetClasses);
      par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_prob, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
                                       redpar_limit, zero_offset);

      //std::cerr << "J: " << cur_source.size() << ", I: " << cur_target.size() << std::endl;
      //std::cerr << "alignment: " << cur_alignment << std::endl;

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      energy -= logl(hmmcc_alignment_prob(cur_source, cur_lookup, cur_target, sclass, tclass, dict, cur_align_model, initial_prob, cur_alignment, true));
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
    update_dict_from_counts(dcount, prior_weight, nSentences, 0.0, false, 0.0, 0, dict, hmm_min_dict_entry,
                            options.msolve_mode_ != MSSolvePGD, options.gd_stepsize_);
#endif

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double sum_daes = 0.0;
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

        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];

        const ushort curJ = cur_source.size();
        const ushort curI = cur_target.size();
        const ushort nLabels = (start_empty_word) ? 2 * curI + 1 : 2 * curI;

        Math1D::Vector<uint> sclass(curJ);
        for (uint j = 0; j < curJ; j++)
          sclass[j] = source_class[cur_source[j]];

        Math1D::Vector<uint> tclass(curI);
        for (uint i = 0; i < curI; i++)
          tclass[i] = target_class[cur_target[i]];

        Math3D::Tensor<double> cur_align_model;
        par2nonpar_hmmcc_alignment_model(sclass, tclass, source_fert_prob, dist_params, dist_grouping_param, cur_align_model, align_type, deficient,
                                         redpar_limit, zero_offset);

        Storage1D<AlignBaseType> viterbi_alignment;

        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

        compute_ehmmcc_viterbi_alignment(cur_source, cur_lookup, cur_target, sclass, tclass, dict, cur_align_model,
                                         initial_prob[curI - 1], viterbi_alignment, options, false, false, 0.0);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        sum_daes += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_daes /= nContributors;
      sum_fmeasure /= nContributors;

      std::cerr << "#### EHMM-DoubleClass Viterbi-AER after Viterbi-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### EHMM-DoubleClass Viterbi-fmeasure after Viterbi-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM-DoubleClass Viterbi-DAE/S after Viterbi-iteration #" << iter << ": " << sum_daes << std::endl;
    }


  }	//end for iter
}