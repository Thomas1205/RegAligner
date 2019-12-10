/*** first version written by Thomas Schoenemann as a private person, November 2009 ***/
/*** refined at Lund University, Sweden, 2010-2011, at the University of Düsseldorf, Germany, 2012, and since as a private person ***/

#ifndef HMM_FORWARD_BACKWARD
#define HMM_FORWARD_BACKWARD

#include "mttypes.hh"
#include "matrix.hh"
#include "training_common.hh"

/******************************* interface routines *****************************/

//forward

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                           const SingleLookupTable& slookup, const SingleWordDictionary& dict,
                           const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit);

template<typename T>
void calculate_hmm_forward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit);

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                           const SingleLookupTable& slookup, const Storage1D<uint>& tclass, const SingleWordDictionary& dict,
                           const Math3D::Tensor<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit);

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict,
                           const Math3D::Tensor<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit);


//forward sum

double calculate_hmm_forward_log_sum(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                                     const SingleLookupTable& slookup, const SingleWordDictionary& dict,
                                     const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                                     const HmmAlignProbType align_type, bool start_empty_word, int redpar_limit);

//backward

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                            const SingleLookupTable& slookup, const SingleWordDictionary& dict,
                            const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                            const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& backward,
                            bool include_start_alignment, int redpar_limit);

template<typename T>
void calculate_hmm_backward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                            const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& backward,
                            bool include_start_alignment, int redpar_limit);

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                            const SingleLookupTable& slookup, const Storage1D<uint>& tclass,
                            const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
                            const Math1D::Vector<double>& start_prob, const HmmAlignProbType align_type, bool start_empty_word,
                            Math2D::Matrix<T>& backward, bool include_start_alignment, int redpar_limit);

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                            const Math1D::Vector<double>& start_prob, const HmmAlignProbType align_type, bool start_empty_word,
                            Math2D::Matrix<T>& backward, bool include_start_alignment, int redpar_limit);


/******************************* implementation routines **************************/

/*** forward routines ***/

template<typename T>
void calculate_hmm_forward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                           const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward);

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict,
                           const Math3D::Tensor<double>& align_model, const Math1D::Vector<double>& start_prob,
                           Math2D::Matrix<T>& forward);

/** this exploits the special structure of reduced parametric models.
    Make sure that you are using such a model **/
template<typename T>
void calculate_hmm_forward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                                       Math2D::Matrix<T>& forward, int redpar_limit);

// void calculate_scaled_hmm_forward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
// const SingleLookupTable& slookup, const SingleWordDictionary& dict,
// const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
// Math2D::Matrix<double>& forward, Math1D::Vector<long double>& scale);

template<typename T>
void calculate_sehmm_forward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                             const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward);

template<typename T>
void calculate_sehmm_forward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                             const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward);

/** this exploits the special structure of reduced parametric models.
    Make sure that you are using such a model **/
template<typename T>
void calculate_sehmm_forward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, 
                                         const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward, int redpar_limit);

/*** forward sum routines ***/

double calculate_hmm_forward_log_sum(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob);

double calculate_hmm_forward_log_sum(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                                     const SingleLookupTable& slookup, const Storage1D<uint>& tclass,
                                     const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
                                     const Math1D::Vector<double>& start_prob);

double calculate_hmm_forward_log_sum_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                                                 const Math1D::Vector<double>& start_prob, int redpar_limit);

double calculate_sehmm_forward_log_sum(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob);

double calculate_sehmm_forward_log_sum_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, 
                                                   const Math1D::Vector<double>& start_prob, int redpar_limit);

double calculate_sehmm_forward_log_sum(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                                       const SingleLookupTable& slookup, const Storage1D<uint>& tclass,
                                       const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
                                       const Math1D::Vector<double>& start_prob);

/*** backward routines ***/

template<typename T>
void calculate_hmm_backward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                            const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward, bool include_start_alignment = true);

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                            const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward,
                            bool include_start_alignment = true);

// void calculate_scaled_hmm_backward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
// const SingleLookupTable& slookup, const SingleWordDictionary& dict,
// const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
// Math2D::Matrix<double>& backward, Math1D::Vector<long double>& scale);

template<typename T>
void calculate_sehmm_backward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                              Math2D::Matrix<T>& backward, bool include_start_alignment = true);

template<typename T>
void calculate_sehmm_backward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                              const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward, bool include_start_alignment = true);

template<typename T>
void calculate_hmm_backward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                                        Math2D::Matrix<T>& backward, bool include_start_alignment, int redpar_limit);

template<typename T>
void calculate_sehmm_backward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                                          Math2D::Matrix<T>& backward, bool include_start_alignment, int redpar_limit);

/************ implementation **********/

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence, const SingleLookupTable& slookup,
                           const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  Math2D::Matrix<double> dicttab(J, I + 1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  calculate_hmm_forward(dicttab, align_model, start_prob, align_type, start_empty_word, forward, redpar_limit);
}

template<typename T>
void calculate_hmm_forward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit)
{
  if (start_empty_word && align_type == HmmAlignProbReducedpar)
    calculate_sehmm_forward_with_tricks(dict, align_model, start_prob, forward, redpar_limit);
  else if (start_empty_word)
    calculate_sehmm_forward(dict, align_model, start_prob, forward);
  else if (align_type == HmmAlignProbReducedpar)
    calculate_hmm_forward_with_tricks(dict, align_model, start_prob, forward, redpar_limit);
  else
    calculate_hmm_forward(dict, align_model, start_prob, forward);
}

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                           const SingleLookupTable& slookup, const Storage1D<uint>& tclass, const SingleWordDictionary& dict,
                           const Math3D::Tensor<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit)
{

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  Math2D::Matrix<double> dicttab(J,I+1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  calculate_hmm_forward(tclass, dicttab, align_model, start_prob, align_type, start_empty_word, forward, redpar_limit);
}

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict,
                           const Math3D::Tensor<double>& align_model, const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& forward, int redpar_limit)
{

  if (start_empty_word)
    calculate_sehmm_forward(tclass, dict, align_model, start_prob, forward);
  else
    calculate_hmm_forward(tclass, dict, align_model, start_prob, forward);
}

template<typename T>
void calculate_hmm_forward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                           const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward)
{
  const uint I = dict.yDim()-1;
  const uint J = dict.xDim();

  assert(forward.xDim() >= 2 * I);
  assert(forward.yDim() >= J);

  for (uint i = 0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,i);
  }

  for (uint i = I; i < 2 * I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,I);
  }

  for (uint j = 1; j < J; j++) {
    const uint j_prev = j - 1;

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;

      for (uint i_prev = 0; i_prev < I; i_prev++)
        sum += align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));

      forward(i, j) = sum * dict(j,i);
    }

    const T cur_emptyword_prob = dict(j,I);

    for (uint i = I; i < 2 * I; i++) {

      const T sum = align_model(I, i - I) * (forward(i, j_prev) + forward(i - I, j_prev));

      forward(i, j) = sum * cur_emptyword_prob;
    }
  }
}

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                           const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward)
{
  //const uint I = target.size();
  //const uint J = source.size();

  const uint I = dict.yDim()-1;
  const uint J = dict.xDim();

  assert(forward.xDim() >= 2 * I);
  assert(forward.yDim() >= J);

  //const uint start_s_idx = source[0];
  for (uint i = 0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,i); //dict[target[i]][slookup(0, i)];
  }

  for (uint i = I; i < 2 * I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,I); //dict[0][start_s_idx - 1];
  }

  for (uint j = 1; j < J; j++) {
    const uint j_prev = j - 1;
    //const uint s_idx = source[j];

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;

      for (uint i_prev = 0; i_prev < I; i_prev++)
        sum += align_model(i, i_prev, tclass[i_prev]) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));

      forward(i, j) = sum * dict(j,i); //dict[target[i]][slookup(j, i)];
    }

    const T cur_emptyword_prob = dict(j,I); //dict[0][s_idx - 1];

    for (uint i = I; i < 2 * I; i++) {

      const T sum = align_model(I, i - I, tclass[i - I]) * (forward(i, j_prev) + forward(i - I, j_prev));

      forward(i, j) = sum * cur_emptyword_prob;
    }
  }
}

template<typename T>
void calculate_sehmm_forward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                             const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward)
{
  const uint I = dict.yDim() - 1;
  const uint J = dict.xDim();

  forward.resize(2 * I + 1, J);

  for (uint i = 0; i < I; i++) {
    const double start_align_prob = start_prob[i];
    forward(i, 0) = start_align_prob * dict(0, i);
  }

  for (uint i = I; i < 2 * I; i++) {
    forward(i, 0) = 0.0;
  }
  //initial empty word
  forward(2 * I, 0) = start_prob[I] * dict(0, I);

  for (uint j = 1; j < J; j++) {
    const uint j_prev = j - 1;

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;

      for (uint i_prev = 0; i_prev < I; i_prev++)
        sum += align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));

      sum += forward(2 * I, j_prev) * start_prob[i];

      forward(i, j) = sum * dict(j,i);

      assert(!isnan(forward(i, j)));
    }

    const T cur_emptyword_prob = dict(j, I);

    for (uint i = I; i < 2 * I; i++) {

      const T sum = align_model(I, i - I) * (forward(i, j_prev) + forward(i - I, j_prev));
      forward(i, j) = sum * cur_emptyword_prob;
    }

    //initial empty word
    forward(2 * I, j) = forward(2 * I, j_prev) * start_prob[I] * cur_emptyword_prob;
  }
}

template<typename T>
void calculate_sehmm_forward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                             const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward)
{
  //const uint I = target.size();
  //const uint J = source.size();

  const uint I = dict.yDim()-1;
  const uint J = dict.xDim();

  forward.resize(2 * I + 1, J);

  //const uint start_s_idx = source[0];
  for (uint i = 0; i < I; i++) {
    const double start_align_prob = start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,i); //dict[target[i]][slookup(0, i)];
  }

  for (uint i = I; i < 2 * I; i++) {
    forward(i, 0) = 0.0;
  }
  //initial empty word
  forward(2 * I, 0) = start_prob[I] * dict(0,I); //dict[0][start_s_idx - 1];

  for (uint j = 1; j < J; j++) {
    const uint j_prev = j - 1;
    //const uint s_idx = source[j];

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;

      for (uint i_prev = 0; i_prev < I; i_prev++)
        sum += align_model(i, i_prev, tclass[i_prev]) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));

      sum += forward(2 * I, j_prev) * start_prob[i];

      forward(i, j) = sum * dict(j,i); //dict[target[i]][slookup(j, i)];

      assert(!isnan(forward(i, j)));
    }

    T cur_emptyword_prob = dict(j,I); //dict[0][s_idx - 1];

    for (uint i = I; i < 2 * I; i++) {

      T sum = align_model(I, i - I, tclass[i - I]) * (forward(i, j_prev) + forward(i - I, j_prev));

      forward(i, j) = sum * cur_emptyword_prob;
    }

    //initial empty word
    forward(2 * I, j) = forward(2 * I, j_prev) * start_prob[I] * cur_emptyword_prob;
  }
}

template<typename T>
void calculate_hmm_forward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                                       const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward, int redpar_limit)
{
  const int I = dict.yDim()-1;
  const int J = dict.xDim();

  assert(int (forward.xDim()) >= 2 * I);
  assert(int (forward.yDim()) >= J);

  for (int i = 0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,i);
  }

  for (int i = I; i < 2 * I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,I);
  }

  Math1D::Vector<double> long_dist_align_prob(I, 0.0);
  for (int i = 0; i < I; i++) {

    if (i + redpar_limit + 1 < I)
      long_dist_align_prob[i] = align_model(i + redpar_limit + 1, i);
    else if (i - redpar_limit > 0)
      long_dist_align_prob[i] = align_model(i - redpar_limit - 1, i);
  }

  for (int j = 1; j < J; j++) {
    const int j_prev = j - 1;

    //NOTE: we are exploiting here that p(i|i_prev) does not depend on i for
    //   the considered i_prev's. But it DOES depend on i_prev

#if 0
    T prev_sum = 0.0;
    for (int i_prev = 0; i_prev < I; i_prev++) {

      prev_sum += (forward(i_prev, j_prev) + forward(i_prev + I, j_prev)) * long_dist_align_prob[i_prev];
    }

    for (int i = 0; i < I; i++) {

      T sum = 0.0;
      T prev_distant_sum = prev_sum;

      for (int i_prev = std::max(0, i - redpar_limit); i_prev <= std::min(I - 1, i + redpar_limit); i_prev++) {
        sum += align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));
        prev_distant_sum -= (forward(i_prev, j_prev) + forward(i_prev + I, j_prev)) * long_dist_align_prob[i_prev];
      }
      sum += prev_distant_sum;

      forward(i, j) = sum * dict(j,i);
    }
#else

    T prev_distant_sum = 0.0;
    for (int i_prev = redpar_limit + 1; i_prev < I; i_prev++)
      prev_distant_sum += (forward(i_prev, j_prev) +   forward(i_prev + I, j_prev)) * long_dist_align_prob[i_prev];

    {
      //i=0
      T sum = prev_distant_sum;

      for (int i_prev = 0; i_prev <= std::min(I - 1, redpar_limit); i_prev++) {
        sum += align_model(0, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));
      }

      forward(0, j) = sum * dict(j,0);
    }

    for (int i = 1; i < I; i++) {

      if (i + redpar_limit < I)
        prev_distant_sum -= (forward(i + redpar_limit, j_prev) + forward(i + redpar_limit + I, j_prev)) * long_dist_align_prob[i + redpar_limit];
      if (i - redpar_limit > 0)
        prev_distant_sum += (forward(i - redpar_limit - 1, j_prev) + forward(i - redpar_limit - 1 + I, j_prev)) * long_dist_align_prob[i - redpar_limit - 1];

      T sum = prev_distant_sum;

      for (int i_prev = std::max(0, i - redpar_limit); i_prev <= std::min(I - 1, i + redpar_limit); i_prev++) {
        sum += align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));
      }

      //stability issues may arise here:
      if (sum <= 0.0) {
        sum = 0.0;
        for (int i_prev = 0; i_prev < I; i_prev++)
          sum += align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));
      }

      forward(i, j) = sum * dict(j,i);
    }
#endif

    const T cur_emptyword_prob = dict(j,I);

    for (int i = I; i < 2 * I; i++) {

      const T sum = align_model(I, i - I) * (forward(i, j_prev) + forward(i - I, j_prev));

      forward(i, j) = sum * cur_emptyword_prob;
    }
  }
}

/** this exploits the special structure of reduced parametric models.
    Make sure that you are using such a model **/
template<typename T>
void calculate_sehmm_forward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, 
                                         const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& forward, int redpar_limit) 
{
  
  const int I = dict.yDim() - 1;
  const int J = dict.xDim();

  Math1D::Vector<double> long_dist_align_prob(I, 0.0);
  for (int i = 0; i < I; i++) {

    if (i + redpar_limit + 1 < I)
      long_dist_align_prob[i] = align_model(i + redpar_limit + 1, i);
    else if (i - redpar_limit > 0)
      long_dist_align_prob[i] = align_model(i - redpar_limit - 1, i);
  }

  forward.resize(2 * I + 1, J);

  for (uint i = 0; i < I; i++) {
    const double start_align_prob = start_prob[i];
    forward(i, 0) = start_align_prob * dict(0,i);
  }

  for (uint i = I; i < 2 * I; i++) {
    forward(i, 0) = 0.0;
  }
  //initial empty word
  forward(2 * I, 0) = start_prob[I] * dict(0,I);

  for (uint j = 1; j < J; j++) {
    const uint j_prev = j - 1;

    T prev_distant_sum = 0.0;
    for (int i_prev = redpar_limit + 1; i_prev < I; i_prev++)
      prev_distant_sum += (forward(i_prev, j_prev) +   forward(i_prev + I, j_prev)) * long_dist_align_prob[i_prev];

    {
      //i=0
      T sum = prev_distant_sum;

      for (int i_prev = 0; i_prev <= std::min(I - 1, redpar_limit); i_prev++) {
        sum += align_model(0, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));
      }
      sum += start_prob[0] * forward(2 * I, j_prev);

      forward(0, j) = sum * dict(j,0);
    }

    for (int i = 1; i < I; i++) {

      if (i + redpar_limit < I)
        prev_distant_sum -= (forward(i + redpar_limit, j_prev) + forward(i + redpar_limit + I, j_prev)) * long_dist_align_prob[i + redpar_limit];
      if (i - redpar_limit > 0)
        prev_distant_sum += (forward(i - redpar_limit - 1, j_prev) + forward(i - redpar_limit - 1 + I, j_prev)) * long_dist_align_prob[i - redpar_limit - 1];

      T sum = prev_distant_sum;

      for (int i_prev = std::max(0, i - redpar_limit); i_prev <= std::min(I - 1, i + redpar_limit); i_prev++) {
        sum += align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));
      }
      sum += start_prob[i] * forward(2 * I, j_prev);

      //stability issues may arise here:
      if (sum <= 0.0) {
        sum = 0.0;
        for (int i_prev = 0; i_prev < I; i_prev++)
          sum += align_model(i, i_prev) * (forward(i_prev, j_prev) + forward(i_prev + I, j_prev));
        sum += start_prob[i] * forward(2 * I, j_prev);
      }

      forward(i, j) = sum * dict(j,i);
    }

    const T cur_emptyword_prob = dict(j, I);

    for (uint i = I; i < 2 * I; i++) {

      T sum = align_model(I, i - I) * (forward(i, j_prev) + forward(i - I, j_prev));

      forward(i, j) = sum * cur_emptyword_prob;
    }

    //initial empty word
    forward(2 * I, j) = forward(2 * I, j_prev) * start_prob[I] * cur_emptyword_prob;
  }
}

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence, const SingleLookupTable& slookup,
                            const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                            const HmmAlignProbType align_type, bool start_empty_word,  Math2D::Matrix<T>& backward, bool include_start_alignment,
                            int redpar_limit)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  Math2D::Matrix<double> dicttab(J, I + 1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  calculate_hmm_backward(dicttab, align_model, start_prob, align_type, start_empty_word, backward, include_start_alignment, redpar_limit);
}

template<typename T>
void calculate_hmm_backward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                            const HmmAlignProbType align_type, bool start_empty_word, Math2D::Matrix<T>& backward,
                            bool include_start_alignment, int redpar_limit)
{
  if (start_empty_word && align_type == HmmAlignProbReducedpar)
    calculate_sehmm_backward_with_tricks(dict, align_model, start_prob, backward, include_start_alignment, redpar_limit);
  else if (start_empty_word)
    calculate_sehmm_backward(dict, align_model, start_prob, backward, include_start_alignment);
  else if (align_type == HmmAlignProbReducedpar)
    calculate_hmm_backward_with_tricks(dict, align_model, start_prob, backward, include_start_alignment, redpar_limit);
  else
    calculate_hmm_backward(dict, align_model, start_prob, backward, include_start_alignment);
}

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source_sentence, const Storage1D<uint>& target_sentence,
                            const SingleLookupTable& slookup, const Storage1D<uint>& tclass,
                            const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_model,
                            const Math1D::Vector<double>& start_prob, const HmmAlignProbType align_type, bool start_empty_word,
                            Math2D::Matrix<T>& backward, bool include_start_alignment, int redpar_limit)
{
  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  Math2D::Matrix<double> dicttab(J,I+1);
  compute_dictmat(source_sentence, slookup, target_sentence, dict, dicttab);

  calculate_hmm_backward(tclass, dicttab, align_model, start_prob, align_type, start_empty_word, backward, include_start_alignment, redpar_limit);
}

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                            const Math1D::Vector<double>& start_prob, const HmmAlignProbType align_type, bool start_empty_word,
                            Math2D::Matrix<T>& backward, bool include_start_alignment, int redpar_limit)
{
  if (start_empty_word)
    calculate_sehmm_backward(tclass, dict, align_model, start_prob, backward, include_start_alignment);
  else
    calculate_hmm_backward(tclass, dict, align_model, start_prob, backward, include_start_alignment);
}

template<typename T>
void calculate_hmm_backward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                            const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward, bool include_start_alignment)
{
  const uint I = dict.yDim()-1;
  const uint J = dict.xDim();

  assert(backward.xDim() >= 2 * I);
  assert(backward.yDim() >= J);

  for (uint i = 0; i < I; i++)
    backward(i, J - 1) = dict(J-1,i);
  for (uint i = I; i < 2 * I; i++)
    backward(i, J - 1) = dict(J-1,I);

  for (int j = J - 2; j >= 0; j--) {
    const uint j_next = j + 1;

    const T cur_emptyword_prob = dict(j,I);

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next, j_next) * align_model(i_next, i);
      sum += backward(i + I, j_next) * align_model(I, i);

      backward(i, j) = sum * dict(j,i);
      backward(i + I, j) = sum * cur_emptyword_prob;
    }
  }

  if (include_start_alignment) {
    for (uint i = 0; i < 2 * I; i++) {

      const T start_align_prob =
        (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
      backward(i, 0) *= start_align_prob;
    }
  }
}

template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                            const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward, bool include_start_alignment)
{
  //const uint I = target.size();
  //const uint J = source.size();

  const int I = dict.yDim()-1;
  const int J = dict.xDim();

  assert(backward.xDim() >= 2 * I);
  assert(backward.yDim() >= J);

  //const uint end_s_idx = source[J - 1];

  for (uint i = 0; i < I; i++)
    backward(i, J - 1) = dict(J - 1, i); //dict[target[i]][slookup(J - 1, i)];
  for (uint i = I; i < 2 * I; i++)
    backward(i, J - 1) = dict(J - 1, I); //dict[0][end_s_idx - 1];

  for (int j = J - 2; j >= 0; j--) {
    //const uint s_idx = source[j];
    const uint j_next = j + 1;

    const T cur_emptyword_prob = dict(j, I); //dict[0][s_idx - 1];

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next, j_next) * align_model(i_next, i, tclass[i]);
      sum += backward(i + I, j_next) * align_model(I, i, tclass[i]);

      backward(i, j) = sum * dict(j,i); //dict[target[i]][slookup(j, i)];

      backward(i + I, j) = sum * cur_emptyword_prob;
    }
  }

  if (include_start_alignment) {
    for (uint i = 0; i < 2 * I; i++) {

      const T start_align_prob =
        (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
      backward(i, 0) *= start_align_prob;
    }
  }
}

template<typename T>
void calculate_sehmm_backward(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                              const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward, bool include_start_alignment)
{
  const uint I = dict.yDim()-1;
  const uint J = dict.xDim();

  backward.resize(2 * I + 1, J);

  for (uint i = 0; i < I; i++)
    backward(i, J - 1) = dict(J - 1, i);
  for (uint i = I; i <= 2 * I; i++)
    backward(i, J - 1) = dict(J - 1, I);

  for (int j = J - 2; j >= 0; j--) {
    const uint j_next = j + 1;

    const T cur_emptyword_prob = dict(j, I);

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next, j_next) * align_model(i_next, i);
      sum += backward(i + I, j_next) * align_model(I, i);

      backward(i, j) = sum * dict(j, i);
      backward(i + I, j) = sum * cur_emptyword_prob;
    }

    //start empty word
    {
      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next, j_next) * start_prob[i_next];
      sum += backward(2 * I, j_next) * start_prob[I];

      backward(2 * I, j) = sum * cur_emptyword_prob;
    }
  }

  if (include_start_alignment) {
    for (uint i = 0; i < I; i++)
      backward(i, 0) *= start_prob[i];
    for (uint i=I; i < 2*I; i++)
      backward(i, 0) = 0.0;

    backward(2 * I, 0) *= start_prob[I];
  }
}

template<typename T>
void calculate_sehmm_backward(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math3D::Tensor<double>& align_model,
                              const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward, bool include_start_alignment)
{
  const int I = dict.yDim()-1;
  const int J = dict.xDim();

  backward.resize(2 * I + 1, J);

  for (uint i = 0; i < I; i++)
    backward(i, J - 1) = dict(J - 1, i);
  for (uint i = I; i <= 2 * I; i++)
    backward(i, J - 1) = dict(J - 1, I);

  for (int j = J - 2; j >= 0; j--) {
    const uint j_next = j + 1;

    const T cur_emptyword_prob = dict(j, I);

    for (uint i = 0; i < I; i++) {

      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next, j_next) * align_model(i_next, i, tclass[i]);
      sum += backward(i + I, j_next) * align_model(I, i, tclass[i]);

      backward(i, j) = sum * dict(j, i);

      backward(i + I, j) = sum * cur_emptyword_prob;
    }

    //start empty word
    {
      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next, j_next) * start_prob[i_next];
      sum += backward(2 * I, j_next) * start_prob[I];

      backward(2 * I, j) = sum * cur_emptyword_prob;
    }
  }

  if (include_start_alignment) {

    for (uint i = 0; i < I; i++)
      backward(i, 0) *= start_prob[i];
    for (uint i=I; i < 2*I; i++)
      backward(i, 0) = 0.0;

    backward(2 * I, 0) *= start_prob[I];
  }
}

template<typename T>
void calculate_hmm_backward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model,
                                        const Math1D::Vector<double>& start_prob, Math2D::Matrix<T>& backward,
                                        bool include_start_alignment, int redpar_limit)
{
  const int I = dict.yDim()-1;
  const int J = dict.xDim();

  assert(int (backward.xDim()) >= 2 * I);
  assert(int (backward.yDim()) >= J);

  Math1D::Vector<double> long_dist_align_prob(I, 0.0);

  for (int i = 0; i < I; i++) {
    if (i + redpar_limit + 1 < I)
      long_dist_align_prob[i] = align_model(i + redpar_limit + 1, i);
    else if (i - redpar_limit > 0)
      long_dist_align_prob[i] = align_model(i - redpar_limit - 1, i);
  }

  for (int i = 0; i < I; i++)
    backward(i, J - 1) = dict(J-1,i);
  for (int i = I; i < 2 * I; i++)
    backward(i, J - 1) = dict(J-1,I);

  for (int j = J - 2; j >= 0; j--) {
    const uint j_next = j + 1;

    const T cur_emptyword_prob = dict(j,I);

#if 0
    T next_sum = 0.0;
    for (int i_next = 0; i_next < I; i_next++)
      next_sum += backward(i_next, j_next);

    for (int i = 0; i < I; i++) {

      T next_distant_sum = next_sum;

      T sum = 0.0;
      for (int i_next = std::max(0, i - redpar_limit);
           i_next <= std::min(I - 1, i + redpar_limit); i_next++) {
        sum += backward(i_next, j_next) * align_model(i_next, i);
        next_distant_sum -= backward(i_next, j_next);
      }
      sum += next_distant_sum * long_dist_align_prob[i];
      sum += backward(i + I, j_next) * align_model(I, i);

      backward(i, j) = sum * dict(j,i);

      backward(i + I, j) = sum * cur_emptyword_prob;
    }
#else

    T next_distant_sum = 0.0;

    for (int i_next = redpar_limit + 1; i_next < I; i_next++)
      next_distant_sum += backward(i_next, j_next);

    {
      // i= 0

      T sum = next_distant_sum * long_dist_align_prob[0];
      for (int i_next = 0; i_next <= std::min(I - 1, redpar_limit); i_next++) {
        sum += backward(i_next, j_next) * align_model(i_next, 0);
      }

      sum += backward(I, j_next) * align_model(I, 0);

      backward(0, j) = sum * dict(j,0);
      backward(I, j) = sum * cur_emptyword_prob;
    }

    for (int i = 1; i < I; i++) {

      if (i + redpar_limit < I)
        next_distant_sum -= backward(i + redpar_limit, j_next);
      if (i - redpar_limit > 0)
        next_distant_sum += backward(i - redpar_limit - 1, j_next);

      T sum = next_distant_sum * long_dist_align_prob[i];

      for (int i_next = std::max(0, i - redpar_limit); i_next <= std::min(I - 1, i + redpar_limit); i_next++) {
        sum += backward(i_next, j_next) * align_model(i_next, i);
      }
      sum += backward(i + I, j_next) * align_model(I, i);

      //stability issues may arise here:
      if (sum <= 0.0) {
        sum = backward(i + I, j_next) * align_model(I, i);
        for (int i_next = 0; i_next < I; i_next++)
          sum += backward(i_next, j_next) * align_model(i_next, i);
      }

      backward(i, j) = sum * dict(j,i);
      backward(i + I, j) = sum * cur_emptyword_prob;
    }
#endif
  }

  if (include_start_alignment) {
    for (int i = 0; i < 2 * I; i++) {

      const T start_align_prob = (start_prob.size() == 0) ? 1.0 / (2 * I) : start_prob[i];
      backward(i, 0) *= start_align_prob;
    }
  }
}

template<typename T>
void calculate_sehmm_backward_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_model, const Math1D::Vector<double>& start_prob,
                                          Math2D::Matrix<T>& backward, bool include_start_alignment, int redpar_limit)
{ 
  const int I = dict.yDim()-1;
  const int J = dict.xDim();

  Math1D::Vector<double> long_dist_align_prob(I, 0.0);

  for (int i = 0; i < I; i++) {
    if (i + redpar_limit + 1 < I)
      long_dist_align_prob[i] = align_model(i + redpar_limit + 1, i);
    else if (i - redpar_limit > 0)
      long_dist_align_prob[i] = align_model(i - redpar_limit - 1, i);
  }

  backward.resize(2 * I + 1, J);

  for (uint i = 0; i < I; i++)
    backward(i, J - 1) = dict(J - 1, i);
  for (uint i = I; i <= 2 * I; i++)
    backward(i, J - 1) = dict(J - 1, I);

  for (int j = J - 2; j >= 0; j--) {
    const uint j_next = j + 1;

    const T cur_emptyword_prob = dict(j, I);

    T next_distant_sum = 0.0;

    for (int i_next = redpar_limit + 1; i_next < I; i_next++)
      next_distant_sum += backward(i_next, j_next);

    {
      // i= 0

      T sum = next_distant_sum * long_dist_align_prob[0];
      for (int i_next = 0; i_next <= std::min(I - 1, redpar_limit); i_next++) {
        sum += backward(i_next, j_next) * align_model(i_next, 0);
      }

      sum += backward(I, j_next) * align_model(I, 0);

      backward(0, j) = sum * dict(j,0);
      backward(I, j) = sum * cur_emptyword_prob;
    }

    for (int i = 1; i < I; i++) {

      if (i + redpar_limit < I)
        next_distant_sum -= backward(i + redpar_limit, j_next);
      if (i - redpar_limit > 0)
        next_distant_sum += backward(i - redpar_limit - 1, j_next);

      T sum = next_distant_sum * long_dist_align_prob[i];

      for (int i_next = std::max(0, i - redpar_limit); i_next <= std::min(I - 1, i + redpar_limit); i_next++) {
        sum += backward(i_next, j_next) * align_model(i_next, i);
      }
      sum += backward(i + I, j_next) * align_model(I, i);

      //stability issues may arise here:
      if (sum <= 0.0) {
        sum = backward(i + I, j_next) * align_model(I, i);
        for (int i_next = 0; i_next < I; i_next++)
          sum += backward(i_next, j_next) * align_model(i_next, i);
      }

      backward(i, j) = sum * dict(j,i);
      backward(i + I, j) = sum * cur_emptyword_prob;
    }

    //start empty word
    {
      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next, j_next) * start_prob[i_next];
      sum += backward(2 * I, j_next) * start_prob[I];

      backward(2 * I, j) = sum * cur_emptyword_prob;
    }
  }

  if (include_start_alignment) {

    for (uint i = 0; i < I; i++)
      backward(i, 0) *= start_prob[i];
    for (uint i=I; i < 2*I; i++)
      backward(i, 0) = 0.0;

    backward(2 * I, 0) *= start_prob[I];
  }
}

#endif
