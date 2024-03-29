/*** written by Thomas Schoenemann, initially as a private person, October 2009 ***
 *** refined at Lund University, Sweden, January 2010 - March 2011 ***
 *** and at the University of Düsseldorf, January 2012 - May 2012
 *** and since as a private person ***/

#include "alignment_computation.hh"
#include "hmm_forward_backward.hh"
#include "routines.hh"


/********************************** IBM-1 ******************************/

void compute_ibm1_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, Storage1D<AlignBaseType>& viterbi_alignment)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  viterbi_alignment.resize_dirty(J);

  for (uint j = 0; j < J; j++) {

    double max_prob = dict[0][source_sentence[j] - 1];
    AlignBaseType arg_max = 0;
    for (uint i = 0; i < I; i++) {

      double cur_prob = dict[target_sentence[i]][slookup(j, i)];

      if (cur_prob > max_prob) {
        max_prob = cur_prob;
        arg_max = i + 1;
      }
    }

    viterbi_alignment[j] = arg_max;
  }
}

//posterior decoding for IBM-1
void compute_ibm1_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                    double threshold)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  postdec_alignment.clear();

  for (uint j = 0; j < J; j++) {

    //std::cerr << "j:" << j << std::endl;

    double sum = dict[0][source_sentence[j] - 1];
    //std::cerr << "null-prob: " << dict[0][source_sentence[j]-1] << std::endl;
    for (uint i = 0; i < I; i++) {

      //std::cerr << i << "-prob: " << dict[target_sentence[i]][slookup(j,i)] << std::endl;

      sum += std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]);
    }

    assert(sum > 1e-305);

    for (uint i = 0; i < I; i++) {

      double cur_prob = std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]) / sum;

      if (cur_prob >= threshold) {
        postdec_alignment.insert(std::make_pair(j + 1, i + 1));
      }
    }
  }
}

void compute_ibm1p0_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                      const SingleWordDictionary& dict, double p0, Storage1D<AlignBaseType>& viterbi_alignment)
{
  assert(p0 >= 0.0 && p0 <= 1.0);

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  const double w1 = (1.0 - p0) / I;

  viterbi_alignment.resize_dirty(J);

  for (uint j = 0; j < J; j++) {

    double max_prob = p0 * dict[0][source_sentence[j] - 1];
    AlignBaseType arg_max = 0;
    for (uint i = 0; i < I; i++) {

      double cur_prob = w1 * dict[target_sentence[i]][slookup(j, i)];

      if (cur_prob > max_prob) {
        max_prob = cur_prob;
        arg_max = i + 1;
      }
    }

    viterbi_alignment[j] = arg_max;
  }
}

//posterior decoding for IBM-1
void compute_ibm1p0_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                      const SingleWordDictionary& dict, const double p0,
                                      std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment, double threshold)
{
  assert(p0 >= 0.0 && p0 <= 1.0);

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  const double w1 = (1.0 - p0) / I;

  postdec_alignment.clear();

  for (uint j = 0; j < J; j++) {

    //std::cerr << "j:" << j << std::endl;

    double sum = p0 * dict[0][source_sentence[j] - 1];
    //std::cerr << "null-prob: " << dict[0][source_sentence[j]-1] << std::endl;
    for (uint i = 0; i < I; i++) {

      //std::cerr << i << "-prob: " << dict[target_sentence[i]][slookup(j,i)] << std::endl;

      sum += w1 * std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]);
    }

    assert(sum > 1e-305);

    for (uint i = 0; i < I; i++) {

      double cur_prob = w1 * std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]) / sum;

      if (cur_prob >= threshold) {
        postdec_alignment.insert(std::make_pair(j + 1, i + 1));
      }
    }
  }
}

/****************************** IBM-2 ******************************/

void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
                                    Storage1D<AlignBaseType>& viterbi_alignment)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  viterbi_alignment.resize_dirty(J);

  for (uint j = 0; j < J; j++) {

    double max_prob = dict[0][source_sentence[j] - 1] * align_prob(0, j);
    AlignBaseType arg_max = 0;
    for (uint i = 0; i < I; i++) {

      double cur_prob = dict[target_sentence[i]][slookup(j, i)] * align_prob(i + 1, j);

      if (cur_prob > max_prob) {
        max_prob = cur_prob;
        arg_max = i + 1;
      }
    }

    viterbi_alignment[j] = arg_max;
  }
}

void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
                                    const Storage1D<WordClassType>& sclass, Storage1D<AlignBaseType>& viterbi_alignment)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  viterbi_alignment.resize_dirty(J);

  for (uint j = 0; j < J; j++) {

    //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
    //  We could cheat if we only want training/word alignment. But we just take the previous word
    const uint c = (j == 0) ? 0 : sclass[source_sentence[j-1]];

    double max_prob = dict[0][source_sentence[j] - 1] * align_prob(0, j, c);
    AlignBaseType arg_max = 0;
    for (uint i = 0; i < I; i++) {

      double cur_prob = dict[target_sentence[i]][slookup(j, i)] * align_prob(i + 1, j, c);

      if (cur_prob > max_prob) {
        max_prob = cur_prob;
        arg_max = i + 1;
      }
    }

    viterbi_alignment[j] = arg_max;
  }
}

void compute_ibm2_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
                                    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                    double threshold)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  postdec_alignment.clear();

  for (uint j = 0; j < J; j++) {

    double sum = dict[0][source_sentence[j] - 1] * align_prob(0, j);    //CHECK: no alignment prob for alignments to 0??

    for (uint i = 0; i < I; i++)
      sum += std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]) * align_prob(i + 1, j);

    assert(sum > 1e-305);

    for (uint i = 0; i < I; i++) {

      double marg = std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]) * align_prob(i + 1, j) / sum;

      if (marg >= threshold) {

        postdec_alignment.insert(std::make_pair(j + 1, i + 1));
      }
    }
  }
}

void compute_ibm2_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
                                    const Storage1D<WordClassType>& sclass, std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                    double threshold)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  postdec_alignment.clear();

  for (uint j = 0; j < J; j++) {

    //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
    //  We could cheat if we only want training/word alignment. But we just take the previous word
    const uint c = (j == 0) ? 0 : sclass[source_sentence[j-1]];

    double sum = dict[0][source_sentence[j] - 1] * align_prob(0, j, c);    //CHECK: no alignment prob for alignments to 0??

    for (uint i = 0; i < I; i++)
      sum += std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]) * align_prob(i + 1, j, c);

    assert(sum > 1e-305);

    for (uint i = 0; i < I; i++) {

      double marg = std::max(1e-15, dict[target_sentence[i]][slookup(j, i)]) * align_prob(i + 1, j, c) / sum;

      if (marg >= threshold) {

        postdec_alignment.insert(std::make_pair(j + 1, i + 1));
      }
    }
  }
}

/******************************************* HMM *********************************/

long double compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    Storage1D<AlignBaseType>& viterbi_alignment, const HmmOptions& hmm_options,
    bool internal_mode, bool verbose, double min_dict_entry)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  Math2D::Matrix<double> dicttab(J,I+1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  for (uint i=0; i < I; i++) {
    const Math1D::Vector<double>& cur_dict = dict[target_sentence[i]];
    for (uint j=0; j < J; j++)
      dicttab(j,i) = cur_dict[slookup(j,i)];
  }
  for (uint j=0; j < J; j++)
    dicttab(j,I) = dict[0][source_sentence[j]-1];

  if (hmm_options.start_empty_word_ && hmm_options.align_type_ == HmmAlignProbReducedpar)
    return compute_sehmm_viterbi_alignment_with_tricks(dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose,
           min_dict_entry, hmm_options.redpar_limit_);
  else if (hmm_options.start_empty_word_)
    return compute_sehmm_viterbi_alignment(dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose, min_dict_entry);
  else if (hmm_options.align_type_ == HmmAlignProbReducedpar)
    return compute_ehmm_viterbi_alignment_with_tricks(dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose,
           min_dict_entry, hmm_options.redpar_limit_);
  else
    return compute_ehmm_viterbi_alignment(dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose, min_dict_entry);
}

long double compute_ehmm_viterbi_alignment(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode, bool verbose, double min_dict_entry)
{
  //const uint J = source_sentence.size();
  //const uint I = target_sentence.size();

  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);
  assert(align_prob.yDim() == I);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I);

  Math2D::NamedMatrix<uint> traceback(2 * I, J, MAKENAME(traceback));

  uint cur_idx = 0;

  const double start_null_dict_entry = std::max(min_dict_entry, dict(0,I) /*dict[0][source_sentence[0] - 1]*/);

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0,i)/* dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = start_null_dict_entry * initial_prob[i];

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {
    // if (verbose)
    //   std::cerr << "j: " << j << std::endl;

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j,I) /*dict[0][source_sentence[j] - 1]*/);

    for (uint i = 0; i < I; i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      for (uint i_prev = 0; i_prev < I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev - I);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }

      //       if (arg_max == MAX_UINT) {
      //        std::cerr << "ERROR: j=" << j << ", J=" << J << ", I=" << I << std::endl;
      //       }

      //       assert(arg_max != MAX_UINT);

      double dict_entry = std::max(min_dict_entry, dict(j,i) /*dict[target_sentence[i]][slookup(j, i)]*/);

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }

      cur_score[i] = max_score * null_dict_entry * align_prob(I, i - I);
      traceback(i, j) = arg_max;
    }

    // if (J == 25 && I == 15)
    if (verbose)
      std::cerr << "j=" << j << ", cur_score: " << cur_score << std::endl;

    //DEBUG
    //     if (J == 27 && I == 36) {
    //       std::cerr << "j= " << j << ", score: " << cur_score << std::endl;
    //       std::cerr << "zero-dict-prob: " << dict[0][source_sentence[j]-1] << std::endl;
    //     }

    //     if (isnan(cur_score.max())) {
    //       std::cerr << "j= " << j << ", nan occurs: " << cur_score << std::endl;
    //     }
    //     if (cur_score.max()<= 0.0) {
    //       std::cerr << "j= " << j << ", all zero: " << cur_score << std::endl;
    //     }
    //END_DEBUG

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/
  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  // for (uint i = 0; i < 2 * I; i++) {
  // if (cur_score[i] > max_score) {

  // max_score = cur_score[i];
  // arg_max = i;
  // }
  // }

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  if (verbose && !internal_mode) {

    for (uint j = 1; j < J; j++) {

      if (viterbi_alignment[j] > 0 && viterbi_alignment[j - 1] > 0)
        std::cerr << "p(" << viterbi_alignment[j] << "|" << viterbi_alignment[j - 1]
                  << "): " << align_prob(viterbi_alignment[j] - 1, viterbi_alignment[j - 1] - 1) << std::endl;
    }
  }

  long double prob = ((long double) max_score) * correction_factor;
  return prob;
}

//with classes
long double compute_ehmmc_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& tclass, const SingleWordDictionary& dict,
    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    Storage1D<AlignBaseType>& viterbi_alignment, const HmmOptions& hmm_options, bool internal_mode, bool verbose,
    double min_dict_entry)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  Math2D::Matrix<double> dicttab(J,I+1);
  for (uint i=0; i < I; i++) {
    const Math1D::Vector<double>& cur_dict = dict[target_sentence[i]];
    for (uint j=0; j < J; j++)
      dicttab(j,i) = cur_dict[slookup(j,i)];
  }
  for (uint j=0; j < J; j++)
    dicttab(j,I) = dict[0][source_sentence[j]-1];

  if (hmm_options.start_empty_word_)
    return compute_sehmmc_viterbi_alignment(tclass, dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose, min_dict_entry);
  else
    return compute_ehmmc_viterbi_alignment(tclass, dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose, min_dict_entry);
}


long double compute_ehmmcc_viterbi_alignment(const Storage1D<uint>& sclass, const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict,
    const Math3D::Tensor<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode, bool verbose, double min_dict_entry)
{
  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I);

  Math2D::NamedMatrix<uint> traceback(2 * I, J, MAKENAME(traceback));

  uint cur_idx = 0;

  const double start_null_dict_entry = std::max(min_dict_entry, dict(0,I)); // dict[0][source_sentence[0] - 1]);

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0,i) /*dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = start_null_dict_entry * initial_prob[i];

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {
    // if (verbose)
    //   std::cerr << "j: " << j << std::endl;

    const uint sc = sclass[j-1];

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j,I));

    for (uint i = 0; i < I; i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      for (uint i_prev = 0; i_prev < I; i_prev++) {
        const uint c = tclass[i_prev];
        double hyp_score = prev_score[i_prev] * align_prob(sc, i, i_prev);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
        const uint c = tclass[i_prev - I];
        double hyp_score = prev_score[i_prev] * align_prob(sc, i, i_prev - I);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }

      //       if (arg_max == MAX_UINT) {
      //        std::cerr << "ERROR: j=" << j << ", J=" << J << ", I=" << I << std::endl;
      //       }

      //       assert(arg_max != MAX_UINT);

      double dict_entry = std::max(min_dict_entry, dict(j,i));

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }

      const uint c = tclass[i - I];
      cur_score[i] = max_score * null_dict_entry * align_prob(sc, I, i - I);
      traceback(i, j) = arg_max;
    }

    if (verbose)
      std::cerr << "j=" << j << ", cur_score: " << cur_score << std::endl;

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/

  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  // for (uint i = 0; i < 2 * I; i++) {
  // if (cur_score[i] > max_score) {

  // max_score = cur_score[i];
  // arg_max = i;
  // }
  // }

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    //std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  long double prob = ((long double)max_score) * correction_factor;
  return prob;
}


long double compute_sehmmcc_viterbi_alignment(const Storage1D<uint>& sclass, const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict,
    const Math3D::Tensor<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode, bool verbose, double min_dict_entry)
{
  //const uint J = source_sentence.size();
  //const uint I = target_sentence.size();

  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I + 1);

  Math2D::NamedMatrix<uint> traceback(2 * I + 1, J, MAKENAME(traceback));

  uint cur_idx = 0;
  //uint last_idx = 1;

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0,i) /*dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = 0.0;
  score[0][2 * I] = initial_prob[I] * std::max(min_dict_entry, dict(0,I)); //dict[0][source_sentence[0] - 1]);

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {
    // if (verbose)
    //   std::cerr << "j: " << j << std::endl;

    const uint sc = sclass[j-1];

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j,I)); // dict[0][source_sentence[j] - 1]);

    for (uint i = 0; i < I; i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      for (uint i_prev = 0; i_prev < I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(sc, i, i_prev);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(sc, i, i_prev - I);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      //initial empty word
      {
        double hyp_score = prev_score[2 * I] * initial_prob[i];
        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = 2 * I;
        }
      }

      //       if (arg_max == MAX_UINT) {
      //        std::cerr << "ERROR: j=" << j << ", J=" << J << ", I=" << I << std::endl;
      //       }

      //       assert(arg_max != MAX_UINT);

      double dict_entry = std::max(min_dict_entry, dict(j,i)); // dict[target_sentence[i]][slookup(j, i)]);

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }
      //double dict_entry = std::max(min_dict_entry,dict[0][source_sentence[j]-1]);

      cur_score[i] = max_score * null_dict_entry * align_prob(sc, I, i - I);
      traceback(i, j) = arg_max;
    }
    //initial empty word
    {
      cur_score[2 * I] = prev_score[2 * I] * initial_prob[I] * std::max(1e-15, dict(j,I)); //dict[0][source_sentence[j] - 1]);
      traceback(2 * I, j) = 2 * I;
    }

    if (verbose)
      std::cerr << "j=" << j << ", cur_score: " << cur_score << std::endl;

    //DEBUG
    //     if (J == 27 && I == 36) {
    //       std::cerr << "j= " << j << ", score: " << cur_score << std::endl;
    //       std::cerr << "zero-dict-prob: " << dict[0][source_sentence[j]-1] << std::endl;
    //     }

    //     if (isnan(cur_score.max())) {
    //       std::cerr << "j= " << j << ", nan occurs: " << cur_score << std::endl;
    //     }
    //     if (cur_score.max()<= 0.0) {
    //       std::cerr << "j= " << j << ", all zero: " << cur_score << std::endl;
    //     }
    //END_DEBUG

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/

  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  // for (uint i = 0; i <= 2 * I; i++) {
  // if (cur_score[i] > max_score) {

  // max_score = cur_score[i];
  // arg_max = i;
  // }
  // }

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    //std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  // if (verbose && !internal_mode) {

  // for (uint j = 1; j < J; j++) {

  // if (viterbi_alignment[j] > 0 && viterbi_alignment[j - 1] > 0)
  // std::cerr << "p(" << viterbi_alignment[j] << "|" << viterbi_alignment[j - 1]
  // << "): " << align_prob(viterbi_alignment[j] - 1, viterbi_alignment[j - 1] - 1,
  // tclass[viterbi_alignment[j - 1] - 1]) << std::endl;
  // }
  // }

  long double prob = ((long double)max_score) * correction_factor;
  return prob;
}


long double compute_ehmmcc_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    const HmmOptions& hmm_options, bool internal_mode, bool verbose, double min_dict_entry)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  Math2D::Matrix<double> dicttab(J,I+1);
  for (uint i=0; i < I; i++) {
    const Math1D::Vector<double>& cur_dict = dict[target_sentence[i]];
    for (uint j=0; j < J; j++)
      dicttab(j,i) = cur_dict[slookup(j,i)];
  }
  for (uint j=0; j < J; j++)
    dicttab(j,I) = dict[0][source_sentence[j]-1];

  if (hmm_options.start_empty_word_)
    return compute_sehmmcc_viterbi_alignment(sclass, tclass, dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose, min_dict_entry);
  else
    return compute_ehmmcc_viterbi_alignment(sclass, tclass, dicttab, align_prob, initial_prob, viterbi_alignment, internal_mode, verbose, min_dict_entry);
}

long double compute_ehmm_viterbi_alignment_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode, bool verbose,double min_dict_entry, int redpar_limit)
{
  //const uint J = source_sentence.size();
  //const uint I = target_sentence.size();

  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);
  assert(align_prob.yDim() == I);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I);

  Math2D::NamedMatrix<uint> traceback(2 * I, J, MAKENAME(traceback));

  uint cur_idx = 0;

  const double start_null_dict_entry = std::max(min_dict_entry, dict(0,I) /*dict[0][source_sentence[0] - 1]*/);

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0,i) /*dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = start_null_dict_entry * initial_prob[i];

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  //std::cerr << "initial prob: " << initial_prob << std::endl;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j, I) /*dict[0][source_sentence[j] - 1]*/);

    //find best prev
    int arg_best_prev = -1;
    double best_prev = 0.0;

    for (uint i_prev = 0; i_prev < 2 * I; i_prev++) {
      if (prev_score[i_prev] > best_prev) {
        best_prev = prev_score[i_prev];
        arg_best_prev = i_prev;
      }
    }

    assert(arg_best_prev >= 0);

    int effective_best_prev = arg_best_prev;
    if (effective_best_prev >= int (I))
      effective_best_prev -= I;

    //regular alignments
    for (int i = 0; i < int (I); i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      if (abs(effective_best_prev - i) > redpar_limit) {
        //in this case we only need to visit the positions inside the window

        arg_max = arg_best_prev;
        max_score = best_prev * align_prob(i, effective_best_prev);

        for (int i_prev = std::max<int>(0, i - redpar_limit); i_prev <= std::min<int>(int (I) - 1, i + redpar_limit); i_prev++) {

          double hyp_score = std::max(prev_score[i_prev], prev_score[i_prev + I]) * align_prob(i, i_prev);

          if (hyp_score > max_score) {
            max_score = hyp_score;
            arg_max = i_prev;
            if (prev_score[i_prev] < prev_score[i_prev + I])
              arg_max += I;
          }
        }
      }
      else {
        //regular case, visit all positions

        for (uint i_prev = 0; i_prev < I; i_prev++) {
          double hyp_score = prev_score[i_prev] * align_prob(i, i_prev);

          if (hyp_score > max_score) {
            max_score = hyp_score;
            arg_max = i_prev;
          }
        }
        for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
          double hyp_score = prev_score[i_prev] * align_prob(i, i_prev - I);

          if (hyp_score > max_score) {
            max_score = hyp_score;
            arg_max = i_prev;
          }
        }
      }
      double dict_entry = std::max(min_dict_entry, dict(j,i) /*dict[target_sentence[i]][slookup(j, i)]*/);

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }

    //null alignments
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }
      //double dict_entry = std::max(min_dict_entry,dict[0][source_sentence[j]-1]);

      cur_score[i] = max_score * null_dict_entry * align_prob(I, i - I);
      traceback(i, j) = arg_max;
    }

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/

  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  // for (uint i = 0; i < 2 * I; i++) {
  // if (cur_score[i] > max_score) {

  // max_score = cur_score[i];
  // arg_max = i;
  // }
  // }

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }

    double cm = 0.0;
    size_t cam = MAX_UINT;
    Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), cm, cam);

    std::cerr << "old gives: " << cm << ", " << cam << std::endl;
    std::cerr << "new gives: " << max_score << ", " << arg_max << std::endl;
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  if (verbose && !internal_mode) {

    for (uint j = 1; j < J; j++) {

      if (viterbi_alignment[j] > 0 && viterbi_alignment[j - 1] > 0)
        std::cerr << "p(" << viterbi_alignment[j] << "|" << viterbi_alignment[j - 1]
                  << "): " << align_prob(viterbi_alignment[j] - 1, viterbi_alignment[j - 1] - 1) << std::endl;
    }
  }

  long double prob = ((long double)max_score) * correction_factor;
  return prob;
}

long double compute_ehmmc_viterbi_alignment(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict,
    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode, bool verbose, double min_dict_entry)
{
  //const uint J = source_sentence.size();
  //const uint I = target_sentence.size();

  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);
  assert(align_prob.yDim() == I);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I);

  Math2D::NamedMatrix<uint> traceback(2 * I, J, MAKENAME(traceback));

  uint cur_idx = 0;

  const double start_null_dict_entry = std::max(min_dict_entry, dict(0,I)); // dict[0][source_sentence[0] - 1]);

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0,i) /*dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = start_null_dict_entry * initial_prob[i];

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {
    // if (verbose)
    //   std::cerr << "j: " << j << std::endl;

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j,I)); // dict[0][source_sentence[j] - 1]);

    for (uint i = 0; i < I; i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      for (uint i_prev = 0; i_prev < I; i_prev++) {
        const uint c = tclass[i_prev];
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
        const uint c = tclass[i_prev - I];
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev - I);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }

      //       if (arg_max == MAX_UINT) {
      //        std::cerr << "ERROR: j=" << j << ", J=" << J << ", I=" << I << std::endl;
      //       }

      //       assert(arg_max != MAX_UINT);

      double dict_entry = std::max(min_dict_entry, dict(j,i)); //dict[target_sentence[i]][slookup(j, i)]);

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }

      const uint c = tclass[i - I];
      cur_score[i] = max_score * null_dict_entry * align_prob(I, i - I);
      traceback(i, j) = arg_max;
    }

    if (verbose)
      std::cerr << "j=" << j << ", cur_score: " << cur_score << std::endl;

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/

  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  // for (uint i = 0; i < 2 * I; i++) {
  // if (cur_score[i] > max_score) {

  // max_score = cur_score[i];
  // arg_max = i;
  // }
  // }

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    //std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  long double prob = ((long double)max_score) * correction_factor;
  return prob;
}

long double compute_sehmm_viterbi_alignment(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode,
    bool verbose, double min_dict_entry)
{
  //const uint J = source_sentence.size();
  //const uint I = target_sentence.size();

  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);
  assert(align_prob.yDim() == I);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I + 1);

  Math2D::NamedMatrix<uint> traceback(2 * I + 1, J, MAKENAME(traceback));

  uint cur_idx = 0;
  //uint last_idx = 1;

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0,i) /*dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = 0.0;
  score[0][2 * I] = initial_prob[I] * std::max(min_dict_entry, dict(0,I) /*dict[0][source_sentence[0] - 1]*/);

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {
    // if (verbose)
    //   std::cerr << "j: " << j << std::endl;

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j,I) /*dict[0][source_sentence[j] - 1]*/);

    for (uint i = 0; i < I; i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      for (uint i_prev = 0; i_prev < I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev - I);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      //initial empty word
      {
        double hyp_score = prev_score[2 * I] * initial_prob[i];
        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = 2 * I;
        }
      }

      //       if (arg_max == MAX_UINT) {
      //        std::cerr << "ERROR: j=" << j << ", J=" << J << ", I=" << I << std::endl;
      //       }

      //       assert(arg_max != MAX_UINT);

      double dict_entry = std::max(min_dict_entry, dict(j,i) /*dict[target_sentence[i]][slookup(j, i)]*/);

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }
      //double dict_entry = std::max(min_dict_entry,dict[0][source_sentence[j]-1]);

      cur_score[i] = max_score * null_dict_entry * align_prob(I, i - I);
      traceback(i, j) = arg_max;
    }
    //initial empty word
    {
      cur_score[2 * I] = prev_score[2 * I] * initial_prob[I] * null_dict_entry;
      traceback(2 * I, j) = 2 * I;
    }

    if (verbose)
      std::cerr << "j=" << j << ", cur_score: " << cur_score << std::endl;

    //DEBUG
    //     if (J == 27 && I == 36) {
    //       std::cerr << "j= " << j << ", score: " << cur_score << std::endl;
    //       std::cerr << "zero-dict-prob: " << dict[0][source_sentence[j]-1] << std::endl;
    //     }

    //     if (isnan(cur_score.max())) {
    //       std::cerr << "j= " << j << ", nan occurs: " << cur_score << std::endl;
    //     }
    //     if (cur_score.max()<= 0.0) {
    //       std::cerr << "j= " << j << ", all zero: " << cur_score << std::endl;
    //     }
    //END_DEBUG

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/

  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  // for (uint i = 0; i <= 2 * I; i++) {
  // if (cur_score[i] > max_score) {

  // max_score = cur_score[i];
  // arg_max = i;
  // }
  // }

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
    exit(1);
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  if (verbose && !internal_mode) {

    for (uint j = 1; j < J; j++) {

      if (viterbi_alignment[j] > 0 && viterbi_alignment[j - 1] > 0)
        std::cerr << "p(" << viterbi_alignment[j] << "|" << viterbi_alignment[j - 1]
                  << "): " << align_prob(viterbi_alignment[j] - 1, viterbi_alignment[j - 1] - 1) << std::endl;
    }
  }

  long double prob = ((long double)max_score) * correction_factor;
  return prob;
}


long double compute_sehmm_viterbi_alignment_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode, bool verbose, double min_dict_entry, int redpar_limit)
{
  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);
  assert(align_prob.yDim() == I);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I + 1);

  Math2D::NamedMatrix<uint> traceback(2 * I + 1, J, MAKENAME(traceback));

  uint cur_idx = 0;

  const double start_null_dict_entry = std::max(min_dict_entry, dict(0, I) /*dict[0][source_sentence[0] - 1]*/);

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0, i) /*dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = 0.0;
  score[0][2*I] = start_null_dict_entry * initial_prob[I];

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  //std::cerr << "initial prob: " << initial_prob << std::endl;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j, I) /*dict[0][source_sentence[j] - 1]*/);

    //find best prev
    int arg_best_prev = -1;
    double best_prev = 0.0;

    for (uint i_prev = 0; i_prev < 2 * I; i_prev++) {
      if (prev_score[i_prev] > best_prev) {
        best_prev = prev_score[i_prev];
        arg_best_prev = i_prev;
      }
    }

    assert(arg_best_prev >= 0);

    int effective_best_prev = arg_best_prev;
    if (effective_best_prev >= int (I))
      effective_best_prev -= I;

    //regular alignments
    for (int i = 0; i < int (I); i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      if (abs(effective_best_prev - i) > redpar_limit) {
        //in this case we only need to visit the positions inside the window

        arg_max = arg_best_prev;
        max_score = best_prev * align_prob(i, effective_best_prev);

        for (int i_prev = std::max<int>(0, i - redpar_limit); i_prev <= std::min<int>(int (I) - 1, i + redpar_limit); i_prev++) {

          double hyp_score = std::max(prev_score[i_prev], prev_score[i_prev + I]) * align_prob(i, i_prev);

          if (hyp_score > max_score) {
            max_score = hyp_score;
            arg_max = i_prev;
            if (prev_score[i_prev] < prev_score[i_prev + I])
              arg_max += I;
          }
        }
      }
      else {
        //regular case, visit all positions

        for (uint i_prev = 0; i_prev < I; i_prev++) {
          double hyp_score = prev_score[i_prev] * align_prob(i, i_prev);

          if (hyp_score > max_score) {
            max_score = hyp_score;
            arg_max = i_prev;
          }
        }
        for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
          double hyp_score = prev_score[i_prev] * align_prob(i, i_prev - I);

          if (hyp_score > max_score) {
            max_score = hyp_score;
            arg_max = i_prev;
          }
        }
      }
      //initial empty word
      {
        double hyp_score = prev_score[2 * I] * initial_prob[i];
        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = 2 * I;
        }
      }

      double dict_entry = std::max(min_dict_entry, dict(j,i) /*dict[target_sentence[i]][slookup(j, i)]*/);

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }

    //null alignments
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }
      //double dict_entry = std::max(min_dict_entry,dict[0][source_sentence[j]-1]);

      cur_score[i] = max_score * null_dict_entry * align_prob(I, i - I);
      traceback(i, j) = arg_max;
    }

    //initial empty word
    {
      cur_score[2 * I] = prev_score[2 * I] * initial_prob[I] * null_dict_entry;
      traceback(2 * I, j) = 2 * I;
    }

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/

  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  if (verbose && !internal_mode) {

    for (uint j = 1; j < J; j++) {

      if (viterbi_alignment[j] > 0 && viterbi_alignment[j - 1] > 0)
        std::cerr << "p(" << viterbi_alignment[j] << "|" << viterbi_alignment[j - 1]
                  << "): " << align_prob(viterbi_alignment[j] - 1, viterbi_alignment[j - 1] - 1) << std::endl;
    }
  }

  long double prob = ((long double)max_score) * correction_factor;
  return prob;
}

long double compute_sehmmc_viterbi_alignment(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict,
    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    Storage1D<AlignBaseType>& viterbi_alignment, bool internal_mode, bool verbose, double min_dict_entry)
{
  //const uint J = source_sentence.size();
  //const uint I = target_sentence.size();

  const uint J = dict.xDim();
  const uint I = dict.yDim()-1;

  viterbi_alignment.resize_dirty(J);
  assert(align_prob.yDim() == I);

  Math1D::Vector<double> score[2];
  for (uint k = 0; k < 2; k++)
    score[k].resize_dirty(2 * I + 1);

  Math2D::NamedMatrix<uint> traceback(2 * I + 1, J, MAKENAME(traceback));

  uint cur_idx = 0;
  //uint last_idx = 1;

  for (uint i = 0; i < I; i++) {
    score[0][i] = std::max(min_dict_entry, dict(0,i) /*dict[target_sentence[i]][slookup(0, i)]*/) * initial_prob[i];
  }
  for (uint i = I; i < 2 * I; i++)
    score[0][i] = 0.0;
  score[0][2 * I] = initial_prob[I] * std::max(min_dict_entry, dict(0,I)); //dict[0][source_sentence[0] - 1]);

  //to keep the numbers inside double precision
  double cur_max = score[0].max();
  long double correction_factor = cur_max;
  score[0] *= 1.0 / cur_max;

  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;

  for (uint j = 1; j < J; j++) {
    // if (verbose)
    //   std::cerr << "j: " << j << std::endl;

    const Math1D::Vector<double>& prev_score = score[cur_idx];
    cur_idx = 1 - cur_idx;
    Math1D::Vector<double>& cur_score = score[cur_idx];

    const double null_dict_entry = std::max(min_dict_entry, dict(j,I)); // dict[0][source_sentence[j] - 1]);

    for (uint i = 0; i < I; i++) {

      double max_score = 0.0;
      uint arg_max = MAX_UINT;

      for (uint i_prev = 0; i_prev < I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      for (uint i_prev = I; i_prev < 2 * I; i_prev++) {
        double hyp_score = prev_score[i_prev] * align_prob(i, i_prev - I);

        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = i_prev;
        }
      }
      //initial empty word
      {
        double hyp_score = prev_score[2 * I] * initial_prob[i];
        if (hyp_score > max_score) {
          max_score = hyp_score;
          arg_max = 2 * I;
        }
      }

      //       if (arg_max == MAX_UINT) {
      //        std::cerr << "ERROR: j=" << j << ", J=" << J << ", I=" << I << std::endl;
      //       }

      //       assert(arg_max != MAX_UINT);

      double dict_entry = std::max(min_dict_entry, dict(j,i)); // dict[target_sentence[i]][slookup(j, i)]);

      cur_score[i] = max_score * dict_entry;
      traceback(i, j) = arg_max;
    }
    for (uint i = I; i < 2 * I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i - I];
      if (hyp_score > max_score) {
        max_score = hyp_score;
        arg_max = i - I;
      }
      //double dict_entry = std::max(min_dict_entry,dict[0][source_sentence[j]-1]);

      cur_score[i] = max_score * null_dict_entry * align_prob(I, i - I);
      traceback(i, j) = arg_max;
    }
    //initial empty word
    {
      cur_score[2 * I] = prev_score[2 * I] * initial_prob[I] * std::max(1e-15, dict(j,I)); //dict[0][source_sentence[j] - 1]);
      traceback(2 * I, j) = 2 * I;
    }

    if (verbose)
      std::cerr << "j=" << j << ", cur_score: " << cur_score << std::endl;

    //DEBUG
    //     if (J == 27 && I == 36) {
    //       std::cerr << "j= " << j << ", score: " << cur_score << std::endl;
    //       std::cerr << "zero-dict-prob: " << dict[0][source_sentence[j]-1] << std::endl;
    //     }

    //     if (isnan(cur_score.max())) {
    //       std::cerr << "j= " << j << ", nan occurs: " << cur_score << std::endl;
    //     }
    //     if (cur_score.max()<= 0.0) {
    //       std::cerr << "j= " << j << ", all zero: " << cur_score << std::endl;
    //     }
    //END_DEBUG

    //to keep the numbers inside double precision
    double cur_max = cur_score.max();
    correction_factor *= cur_max;
    cur_score *= 1.0 / cur_max;
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/

  const Math1D::Vector<double>& cur_score = score[cur_idx];

  double max_score = 0.0;
  size_t arg_max = MAX_UINT;

  Routines::find_max_and_argmax(cur_score.direct_access(), cur_score.size(), max_score, arg_max);

  // for (uint i = 0; i <= 2 * I; i++) {
  // if (cur_score[i] > max_score) {

  // max_score = cur_score[i];
  // arg_max = i;
  // }
  // }

  if (arg_max >= MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    //std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
  }

  assert(arg_max < MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  for (int j = J - 1; j >= 0; j--) {

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max + 1) : 0;

    arg_max = traceback(arg_max, j);
  }

  if (verbose && !internal_mode) {

    for (uint j = 1; j < J; j++) {

      if (viterbi_alignment[j] > 0 && viterbi_alignment[j - 1] > 0)
        std::cerr << "p(" << viterbi_alignment[j] << "|" << viterbi_alignment[j - 1]
                  << "): " << align_prob(viterbi_alignment[j] - 1, viterbi_alignment[j - 1] - 1) << std::endl;
    }
  }

  long double prob = ((long double)max_score) * correction_factor;
  return prob;
}

/***** HMM OptMarg ******/

void compute_ehmm_optmarginal_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                        const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
                                        const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
                                        const HmmOptions& options, Storage1D<AlignBaseType>& optmarginal_alignment)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  optmarginal_alignment.resize_dirty(J);

  uint nLabels = (options.start_empty_word_) ? 2 * I + 1 : 2 * I;

  Math2D::Matrix<double> dicttab(J, I + 1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  Math2D::NamedMatrix<long double> forward(nLabels, J, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(nLabels, J, MAKENAME(backward));

  calculate_hmm_forward(dicttab, align_prob, initial_prob, options, forward);
  calculate_hmm_backward(dicttab, align_prob, initial_prob, options, backward, false);

  for (uint j = 0; j < J; j++) {

    const uint s_idx = source_sentence[j];

    long double max_marginal = 0.0;
    AlignBaseType arg_max = I + 2;

    for (uint i = 0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double hyp_marginal = 0.0;

      if (dict[t_idx][slookup(j, i)] > 0.0) {
        hyp_marginal = forward(i, j) * backward(i, j) / dict[t_idx][slookup(j, i)];
      }

      if (hyp_marginal > max_marginal) {

        max_marginal = hyp_marginal;
        arg_max = i;
      }
    }

    for (uint i = I; i < nLabels; i++) {

      long double hyp_marginal = 0.0;

      if (dict[0][s_idx - 1] > 0.0) {
        hyp_marginal = forward(i, j) * backward(i, j) / dict[0][s_idx - 1];
      }

      if (hyp_marginal > max_marginal) {

        max_marginal = hyp_marginal;
        arg_max = i;
      }
    }

    assert(arg_max <= 2 * I + 1);

    if (arg_max < I)
      optmarginal_alignment[j] = arg_max + 1;
    else
      optmarginal_alignment[j] = 0;
  }
}

void compute_ehmmc_optmarginal_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& tclass, const SingleWordDictionary& dict,
    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    const HmmOptions& options, Storage1D<AlignBaseType>& optmarginal_alignment)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  optmarginal_alignment.resize_dirty(J);

  uint nLabels = (options.start_empty_word_) ? 2 * I + 1 : 2 * I;

  Math2D::Matrix<double> dicttab(J, I + 1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  Math2D::NamedMatrix<long double> forward(nLabels, J, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(nLabels, J, MAKENAME(backward));

  calculate_hmmc_forward(tclass, dicttab, align_prob, initial_prob, options, forward);
  calculate_hmmc_backward(tclass, dicttab, align_prob, initial_prob, options, backward, false);


  //calculate_hmmc_forward(source_sentence, target_sentence, slookup, tclass,  dict, align_prob, initial_prob,
  //                      HmmAlignProbFullpar, start_empty_word, forward, 10000);

  //calculate_hmmc_backward(source_sentence, target_sentence, slookup, tclass, dict, align_prob, initial_prob,
  //                        HmmAlignProbFullpar, start_empty_word, backward, false, 10000);

  for (uint j = 0; j < J; j++) {

    const uint s_idx = source_sentence[j];

    long double max_marginal = 0.0;
    AlignBaseType arg_max = I + 2;

    for (uint i = 0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double hyp_marginal = 0.0;

      if (dict[t_idx][slookup(j, i)] > 0.0) {
        hyp_marginal = forward(i, j) * backward(i, j) / dicttab(j, i); // dict[t_idx][slookup(j, i)];
      }

      if (hyp_marginal > max_marginal) {

        max_marginal = hyp_marginal;
        arg_max = i;
      }
    }

    for (uint i = I; i < nLabels; i++) {

      long double hyp_marginal = 0.0;

      if (dict[0][s_idx - 1] > 0.0) {
        hyp_marginal = forward(i, j) * backward(i, j) / dicttab(j, I); // dict[0][s_idx - 1];
      }

      if (hyp_marginal > max_marginal) {

        max_marginal = hyp_marginal;
        arg_max = i;
      }
    }

    assert(arg_max <= 2 * I + 1);

    if (arg_max < I)
      optmarginal_alignment[j] = arg_max + 1;
    else
      optmarginal_alignment[j] = 0;
  }
}

void compute_ehmmcc_optmarginal_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, const HmmOptions& options,
    Storage1D<AlignBaseType>& optmarginal_alignment)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();
  bool start_empty_word = options.start_empty_word_;

  optmarginal_alignment.resize_dirty(J);

  uint nLabels = (start_empty_word) ? 2 * I + 1 : 2 * I;

  Math2D::NamedMatrix<long double> forward(nLabels, J, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(nLabels, J, MAKENAME(backward));

  Math2D::Matrix<double> dicttab(J,I+1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  calculate_hmmcc_forward(sclass, tclass,  dicttab, align_prob, initial_prob, options, forward);
  calculate_hmmcc_backward(sclass, tclass,  dicttab, align_prob, initial_prob, options, backward, false);

  for (uint j = 0; j < J; j++) {

    const uint s_idx = source_sentence[j];

    long double max_marginal = 0.0;
    AlignBaseType arg_max = I + 2;

    for (uint i = 0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double hyp_marginal = 0.0;

      if (dict[t_idx][slookup(j, i)] > 0.0) {
        hyp_marginal = forward(i, j) * backward(i, j) / dicttab(j, i); //  dict[t_idx][slookup(j, i)];
      }

      if (hyp_marginal > max_marginal) {

        max_marginal = hyp_marginal;
        arg_max = i;
      }
    }

    for (uint i = I; i < nLabels; i++) {

      long double hyp_marginal = 0.0;

      if (dict[0][s_idx - 1] > 0.0) {
        hyp_marginal = forward(i, j) * backward(i, j) / dicttab(j, I); // dict[0][s_idx - 1];
      }

      if (hyp_marginal > max_marginal) {

        max_marginal = hyp_marginal;
        arg_max = i;
      }
    }

    assert(arg_max <= 2 * I + 1);

    if (arg_max < I)
      optmarginal_alignment[j] = arg_max + 1;
    else
      optmarginal_alignment[j] = 0;
  }
}

void compute_ehmm_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
                                    const Math1D::Vector<double>& initial_prob, const HmmOptions& options,
                                    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment, double threshold)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  postdec_alignment.clear();

  const uint addon = (options.start_empty_word_) ? 1 : 0;

  Math2D::Matrix<double> dicttab(J, I + 1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  Math2D::NamedMatrix<long double> forward(2 * I + addon, J, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2 * I + addon, J, MAKENAME(backward));

  calculate_hmm_forward(dicttab, align_prob, initial_prob, options, forward);
  calculate_hmm_backward(dicttab, align_prob, initial_prob, options, backward, false);

  long double sent_prob = 0.0;
  for (uint i = 0; i < forward.xDim(); i++)
    sent_prob += forward(i, J - 1);

  long double inv_sent_prob = 1.0 / sent_prob;

  for (uint j = 0; j < J; j++) {

    for (uint i = 0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double marginal = 0.0;

      if (dict[t_idx][slookup(j, i)] > 1e-75) {
        marginal = forward(i, j) * backward(i, j) * inv_sent_prob / dict[t_idx][slookup(j, i)];
      }

      if (marginal >= threshold) {

        postdec_alignment.insert(std::make_pair(j + 1, i + 1));
      }
    }
  }

}

void compute_ehmmc_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                     const Storage1D<uint>& target_sentence, const Storage1D<uint>& tclass,
                                     const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
                                     const Math1D::Vector<double>& initial_prob, const HmmOptions& options,
                                     std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment, double threshold)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  const uint addon = (options.start_empty_word_) ? 1 : 0;

  postdec_alignment.clear();

  Math2D::Matrix<double> dicttab(J, I + 1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  Math2D::NamedMatrix<long double> forward(2 * I + addon, J, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2 * I + addon, J, MAKENAME(backward));

  calculate_hmmc_forward(tclass, dicttab, align_prob, initial_prob, options, forward);
  //calculate_hmmc_forward(source_sentence, target_sentence, slookup, tclass,
  //                       dict, align_prob, initial_prob, options, forward);


  calculate_hmmc_backward(tclass, dicttab, align_prob, initial_prob, options, backward, false);
  //calculate_hmmc_backward(source_sentence, target_sentence, slookup, tclass,
  //                        dict, align_prob, initial_prob, options, backward, false);

  long double sent_prob = 0.0;
  for (uint i = 0; i < forward.xDim(); i++)
    sent_prob += forward(i, J - 1);

  long double inv_sent_prob = 1.0 / sent_prob;

  for (uint j = 0; j < J; j++) {

    for (uint i = 0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double marginal = 0.0;

      if (dict[t_idx][slookup(j, i)] > 1e-75) {
        marginal = forward(i, j) * backward(i, j) * inv_sent_prob / dicttab(j, i); // dict[t_idx][slookup(j, i)];
      }

      if (marginal >= threshold) {

        postdec_alignment.insert(std::make_pair(j + 1, i + 1));
      }
    }
  }
}

void compute_ehmmcc_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                      const Storage1D<uint>& target_sentence, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                      const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
                                      const Math1D::Vector<double>& initial_prob, const HmmOptions& options,
                                      std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                      double threshold)
{

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  const uint addon = (options.start_empty_word_) ? 1 : 0;

  postdec_alignment.clear();

  Math2D::NamedMatrix<long double> forward(2 * I + addon, J, MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2 * I + addon, J, MAKENAME(backward));

  Math2D::Matrix<double> dicttab(J, I + 1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  calculate_hmmcc_forward(sclass, tclass, dicttab, align_prob, initial_prob, options, forward);
  calculate_hmmcc_backward(sclass, tclass, dicttab, align_prob, initial_prob, options, backward, false);

  long double sent_prob = 0.0;
  for (uint i = 0; i < forward.xDim(); i++)
    sent_prob += forward(i, J - 1);

  long double inv_sent_prob = 1.0 / sent_prob;

  for (uint j = 0; j < J; j++) {

    for (uint i = 0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double marginal = 0.0;

      if (dict[t_idx][slookup(j, i)] > 1e-75) {
        marginal = forward(i, j) * backward(i, j) * inv_sent_prob / dicttab(j, i); // dict[t_idx][slookup(j, i)];
      }

      if (marginal >= threshold) {

        postdec_alignment.insert(std::make_pair(j + 1, i + 1));
      }
    }
  }
}
