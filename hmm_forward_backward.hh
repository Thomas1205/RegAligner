/*** first version written by Thomas Schoenemann as a private person without employment, November 2009 ***/
/*** refined at Lund University, Sweden, 2010-2011 and at the University of Düsseldorf, Germany, 2012 ***/

#ifndef HMM_FORWARD_BACKWARD
#define HMM_FORWARD_BACKWARD

#include "mttypes.hh"
#include "matrix.hh"

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source_sentence,
                           const Storage1D<uint>& target_sentence,
                           const SingleLookupTable& slookup,
                           const SingleWordDictionary& dict,
                           const Math2D::Matrix<double>& align_model,
                           const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type,
                           Math2D::Matrix<T>& forward);


template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source_sentence,
                           const Storage1D<uint>& target_sentence,
                           const SingleLookupTable& slookup,
                           const SingleWordDictionary& dict,
                           const Math2D::Matrix<double>& align_model,
                           const Math1D::Vector<double>& start_prob,
                           Math2D::Matrix<T>& forward);


/** this exploits the special structure of reduced parametric models. 
    Make sure that you are using such a model **/
template<typename T>
void calculate_hmm_forward_with_tricks(const Storage1D<uint>& source_sentence,
				       const Storage1D<uint>& target_sentence,
				       const SingleLookupTable& slookup,
				       const SingleWordDictionary& dict,
				       const Math2D::Matrix<double>& align_model,
				       const Math1D::Vector<double>& start_prob,
				       Math2D::Matrix<T>& forward);


template<typename T>
void calculate_sehmm_forward(const Storage1D<uint>& source_sentence,
                             const Storage1D<uint>& target_sentence,
                             const SingleLookupTable& slookup,
                             const SingleWordDictionary& dict,
                             const Math2D::Matrix<double>& align_model,
                             const Math1D::Vector<double>& start_prob,
                             Math2D::Matrix<T>& forward);


template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source_sentence,
                            const Storage1D<uint>& target_sentence,
                            const SingleLookupTable& slookup,
                            const SingleWordDictionary& dict,
                            const Math2D::Matrix<double>& align_model,
                            const Math1D::Vector<double>& start_prob,
                            const HmmAlignProbType align_type,
                            Math2D::Matrix<T>& backward,
                            bool include_start_alignment = true);



template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source_sentence,
                            const Storage1D<uint>& target_sentence,
                            const SingleLookupTable& slookup,
                            const SingleWordDictionary& dict,
                            const Math2D::Matrix<double>& align_model,
                            const Math1D::Vector<double>& start_prob,
                            Math2D::Matrix<T>& backward,
                            bool include_start_alignment = true);


template<typename T>
void calculate_sehmm_backward(const Storage1D<uint>& source_sentence,
                              const Storage1D<uint>& target_sentence,
                              const SingleLookupTable& slookup,
                              const SingleWordDictionary& dict,
                              const Math2D::Matrix<double>& align_model,
                              const Math1D::Vector<double>& start_prob,
                              Math2D::Matrix<T>& backward,
                              bool include_start_alignment = true);


template<typename T>
void calculate_hmm_backward_with_tricks(const Storage1D<uint>& source_sentence,
					const Storage1D<uint>& target_sentence,
					const SingleLookupTable& slookup,
					const SingleWordDictionary& dict,
					const Math2D::Matrix<double>& align_model,
					const Math1D::Vector<double>& start_prob,
					Math2D::Matrix<T>& backward,
					bool include_start_alignment = true);


/************ implementation **********/

template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source_sentence,
                           const Storage1D<uint>& target_sentence,
                           const SingleLookupTable& slookup,
                           const SingleWordDictionary& dict,
                           const Math2D::Matrix<double>& align_model,
                           const Math1D::Vector<double>& start_prob,
                           const HmmAlignProbType align_type,
                           Math2D::Matrix<T>& forward) {


  if (align_type == HmmAlignProbReducedpar)
    calculate_hmm_forward_with_tricks(source_sentence, target_sentence, slookup, dict, align_model,
                                      start_prob, forward);
  else
    calculate_hmm_forward(source_sentence, target_sentence, slookup, dict, align_model, start_prob, forward);
}


template<typename T>
void calculate_hmm_forward(const Storage1D<uint>& source,
                           const Storage1D<uint>& target,
                           const SingleLookupTable& slookup,
                           const SingleWordDictionary& dict,
                           const Math2D::Matrix<double>& align_model,
                           const Math1D::Vector<double>& start_prob,
                           Math2D::Matrix<T>& forward) {

  const uint I = target.size();
  const uint J = source.size();

  assert(forward.xDim() >= 2*I);
  assert(forward.yDim() >= J);

  const uint start_s_idx = source[0];
  for (uint i=0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward(i,0) = start_align_prob * dict[target[i]][slookup(0,i)];
  }

  for (uint i=I; i < 2*I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward(i,0) = start_align_prob * dict[0][start_s_idx-1];
  }
  
  for (uint j=1; j < J; j++) {
    const uint j_prev = j-1;
    const uint s_idx = source[j];
    
    for (uint i=0; i < I; i++) {
      
      T sum = 0.0;

      for (uint i_prev=0; i_prev < I; i_prev++)
        sum += align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));

      forward(i,j) = sum * dict[target[i]][slookup(j,i)];
    }
    
    T cur_emptyword_prob = dict[0][s_idx-1];
    
    for (uint i=I; i < 2*I; i++) {
      
      T sum = align_model(I,i-I) * ( forward(i,j_prev) + forward(i-I,j_prev) );

      forward(i,j) = sum * cur_emptyword_prob;
    }
  }
}

template<typename T>
void calculate_sehmm_forward(const Storage1D<uint>& source,
                             const Storage1D<uint>& target,
                             const SingleLookupTable& slookup,
                             const SingleWordDictionary& dict,
                             const Math2D::Matrix<double>& align_model,
                             const Math1D::Vector<double>& start_prob,
                             Math2D::Matrix<T>& forward) {

  const uint I = target.size();
  const uint J = source.size();

  forward.resize(2*I+1,J);

  const uint start_s_idx = source[0];
  for (uint i=0; i < I; i++) {
    const double start_align_prob = start_prob[i];
    forward(i,0) = start_align_prob * dict[target[i]][slookup(0,i)];
  }

  for (uint i=I; i < 2*I; i++) {
    forward(i,0) = 0.0;
  }
  //initial empty word
  forward(2*I,0) = start_prob[I] * dict[0][start_s_idx-1];
  
  for (uint j=1; j < J; j++) {
    const uint j_prev = j-1;
    const uint s_idx = source[j];

    for (uint i=0; i < I; i++) {
      
      T sum = 0.0;

      for (uint i_prev=0; i_prev < I; i_prev++)
        sum += align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));

      sum += forward(2*I,j_prev) * start_prob[i];

      forward(i,j) = sum * dict[target[i]][slookup(j,i)];

      assert(!isnan(forward(i,j)));
    }

    T cur_emptyword_prob = dict[0][s_idx-1];
    
    for (uint i=I; i < 2*I; i++) {
      
      T sum = align_model(I,i-I) * ( forward(i,j_prev) + forward(i-I,j_prev) );

      forward(i,j) = sum * cur_emptyword_prob;
    }

    //initial empty word
    forward(2*I,j) = forward(2*I,j_prev) * start_prob[I] * cur_emptyword_prob;
  }  

}


template<typename T>
void calculate_hmm_forward_with_tricks(const Storage1D<uint>& source,
				       const Storage1D<uint>& target,
				       const SingleLookupTable& slookup,
				       const SingleWordDictionary& dict,
				       const Math2D::Matrix<double>& align_model,
				       const Math1D::Vector<double>& start_prob,
				       Math2D::Matrix<T>& forward) {

  const int I = target.size();
  const int J = source.size();

  assert(int(forward.xDim()) >= 2*I);
  assert(int(forward.yDim()) >= J);

  const uint start_s_idx = source[0];
  for (int i=0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward(i,0) = start_align_prob * dict[target[i]][slookup(0,i)];
  }

  for (int i=I; i < 2*I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward(i,0) = start_align_prob * dict[0][start_s_idx-1];
  }
  
  Math1D::Vector<double> long_dist_align_prob(I,0.0);
  for (int i=0; i < I; i++) {

    if (i + 6 < I)
      long_dist_align_prob[i] = align_model(i+6,i);
    else if (i - 6 >= 0)
      long_dist_align_prob[i] = align_model(i-6,i);
  }


  for (int j=1; j < J; j++) {
    const int j_prev = j-1;
    const uint s_idx = source[j];

    //NOTE: we are exploiting here that p(i|i_prev) does not depend on i for
    //   the considered i_prev's. But it DOES depend on i_prev

#if 0
    T prev_sum = 0.0;
    for (int i_prev=0; i_prev < I; i_prev++) {

      prev_sum += (forward(i_prev,j_prev) + forward(i_prev+I,j_prev)) * long_dist_align_prob[i_prev];
    }

    for (int i=0; i < I; i++) {
      
      T sum = 0.0;
      T prev_distant_sum = prev_sum;

      for (int i_prev= std::max(0,i-5); i_prev <= std::min(I-1,i+5); i_prev++) {
        sum += align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));
	prev_distant_sum -= (forward(i_prev,j_prev) + forward(i_prev+I,j_prev))
	  * long_dist_align_prob[i_prev];
      }
      sum += prev_distant_sum;

      forward(i,j) = sum * dict[target[i]][slookup(j,i)];
    }
#else

    T prev_distant_sum = 0.0;
    for (int i_prev = 6; i_prev < I; i_prev++)
      prev_distant_sum += (forward(i_prev,j_prev) + forward(i_prev+I,j_prev)) * long_dist_align_prob[i_prev];

    {
      //i=0
      T sum = 0.0;

      for (int i_prev= 0; i_prev <= std::min(I-1,5); i_prev++) {
        sum += align_model(0,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));
      }
      sum += prev_distant_sum;

      forward(0,j) = sum * dict[target[0]][slookup(j,0)];
    }

    for (int i=1; i < I; i++) {

      if (i+5 < I)
	prev_distant_sum -= (forward(i+5,j_prev) + forward(i+5+I,j_prev)) * long_dist_align_prob[i+5];
      if (i-6 >= 0)
	prev_distant_sum += (forward(i-6,j_prev) + forward(i-6+I,j_prev)) * long_dist_align_prob[i-6];

      T sum = 0.0;

      for (int i_prev= std::max(0,i-5); i_prev <= std::min(I-1,i+5); i_prev++) {
        sum += align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));
      }
      sum += prev_distant_sum;

      //stability issues may arise here:
      if (sum <= 0.0) {
	sum = 0.0;
	for (int i_prev= std::max(0,i-5); i_prev <= std::min(I-1,i+5); i_prev++) 
	  sum += align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));
      }

      forward(i,j) = sum * dict[target[i]][slookup(j,i)];
    }
#endif

    
    T cur_emptyword_prob = dict[0][s_idx-1];
    
    for (int i=I; i < 2*I; i++) {
      
      T sum = align_model(I,i-I) * ( forward(i,j_prev) + forward(i-I,j_prev) );

      forward(i,j) = sum * cur_emptyword_prob;
    }
  }
}



template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source_sentence,
                            const Storage1D<uint>& target_sentence,
                            const SingleLookupTable& slookup,
                            const SingleWordDictionary& dict,
                            const Math2D::Matrix<double>& align_model,
                            const Math1D::Vector<double>& start_prob,
                            const HmmAlignProbType align_type,
                            Math2D::Matrix<T>& backward,
                            bool include_start_alignment) {

  if (align_type == HmmAlignProbReducedpar)
    calculate_hmm_backward_with_tricks(source_sentence, target_sentence, slookup, dict, align_model,
                                       start_prob, backward, include_start_alignment);
  else
    calculate_hmm_backward(source_sentence, target_sentence, slookup, dict, align_model,
                           start_prob, backward, include_start_alignment);      
}


template<typename T>
void calculate_hmm_backward(const Storage1D<uint>& source,
                            const Storage1D<uint>& target,
                            const SingleLookupTable& slookup,
                            const SingleWordDictionary& dict,
                            const Math2D::Matrix<double>& align_model,
                            const Math1D::Vector<double>& start_prob,
                            Math2D::Matrix<T>& backward,
                            bool include_start_alignment) {

  const uint I = target.size();
  const uint J = source.size();

  assert(backward.xDim() >= 2*I);
  assert(backward.yDim() >= J);

  const uint end_s_idx = source[J-1];
  
  for (uint i=0; i < I; i++)
    backward(i,J-1) = dict[target[i]][slookup(J-1,i)];
  for (uint i=I; i < 2*I; i++)
    backward(i,J-1) = dict[0][end_s_idx-1];
      
  for (int j=J-2; j >= 0; j--) {
    const uint s_idx = source[j];
    const uint j_next = j+1;
    
    const T cur_emptyword_prob = dict[0][s_idx-1];

    for (uint i=0; i < I; i++) {

      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next,j_next) * align_model(i_next,i);
      sum += backward(i+I,j_next) * align_model(I,i);

      backward(i,j) = sum * dict[target[i]][slookup(j,i)];

      backward(i+I,j) = sum * cur_emptyword_prob;
    }
  }

  if (include_start_alignment) {
    for (uint i=0; i < 2*I; i++) {

      const T start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
      backward(i,0) *= start_align_prob;
    }
  }
}


template<typename T>
void calculate_sehmm_backward(const Storage1D<uint>& source,
                              const Storage1D<uint>& target,
                              const SingleLookupTable& slookup,
                              const SingleWordDictionary& dict,
                              const Math2D::Matrix<double>& align_model,
                              const Math1D::Vector<double>& start_prob,
                              Math2D::Matrix<T>& backward,
                              bool include_start_alignment) {


  const uint I = target.size();
  const uint J = source.size();

  backward.resize(2*I+1,J);

  const uint end_s_idx = source[J-1];
  
  for (uint i=0; i < I; i++)
    backward(i,J-1) = dict[target[i]][slookup(J-1,i)];
  for (uint i=I; i <= 2*I; i++)
    backward(i,J-1) = dict[0][end_s_idx-1];
      
  for (int j=J-2; j >= 0; j--) {
    const uint s_idx = source[j];
    const uint j_next = j+1;
    
    const T cur_emptyword_prob = dict[0][s_idx-1];

    for (uint i=0; i < I; i++) {

      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next,j_next) * align_model(i_next,i);
      sum += backward(i+I,j_next) * align_model(I,i);

      backward(i,j) = sum * dict[target[i]][slookup(j,i)];

      backward(i+I,j) = sum * cur_emptyword_prob;
    }

    //start empty word
    {
      T sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next,j_next) * start_prob[i_next];
      sum += backward(2*I,j_next) * start_prob[I];

      backward(2*I,j) = sum * cur_emptyword_prob;
    }
  }

  if (include_start_alignment) {
    for (uint i=0; i < I; i++) {

      const T start_align_prob = start_prob[i];
      backward(i,0) *= start_align_prob;
      backward(i+I,0) = 0.0;
    }
    
    backward(2*I,0) *= start_prob[I];
  }

}

template<typename T>
void calculate_hmm_backward_with_tricks(const Storage1D<uint>& source,
					const Storage1D<uint>& target,
					const SingleLookupTable& slookup,
					const SingleWordDictionary& dict,
					const Math2D::Matrix<double>& align_model,
					const Math1D::Vector<double>& start_prob,
					Math2D::Matrix<T>& backward,
					bool include_start_alignment) {

  
  const int I = target.size();
  const int J = source.size();

  assert(int(backward.xDim()) >= 2*I);
  assert(int(backward.yDim()) >= J);

  const uint end_s_idx = source[J-1];

  Math1D::Vector<double> long_dist_align_prob(I,0.0);

  for (int i=0; i < I; i++) {
    if (i+6 < I)
      long_dist_align_prob[i] = align_model(i+6,i);
    else if (i-6 >= 0)
      long_dist_align_prob[i] = align_model(i-6,i);
  }
  
  for (int i=0; i < I; i++)
    backward(i,J-1) = dict[target[i]][slookup(J-1,i)];
  for (int i=I; i < 2*I; i++)
    backward(i,J-1) = dict[0][end_s_idx-1];
  
  for (int j=J-2; j >= 0; j--) {
    const uint s_idx = source[j];
    const uint j_next = j+1;

    const T cur_emptyword_prob = dict[0][s_idx-1];

#if 0    
    T next_sum = 0.0;
    for (int i_next = 0; i_next < I; i_next++)
      next_sum += backward(i_next,j_next);

    for (int i=0; i < I; i++) {
      
      T next_distant_sum = next_sum;
      
      T sum = 0.0;
      for (int i_next = std::max(0,i-5); i_next <= std::min(I-1,i+5); i_next++) {
        sum += backward(i_next,j_next) * align_model(i_next,i);
	next_distant_sum -= backward(i_next,j_next);
      }
      sum += next_distant_sum * long_dist_align_prob[i];
      sum += backward(i+I,j_next) * align_model(I,i);
      
      backward(i,j) = sum * dict[target[i]][slookup(j,i)];

      backward(i+I,j) = sum * cur_emptyword_prob;
    }
#else

    T next_distant_sum = 0.0;

    for (int i_next = 6; i_next < I; i_next++)
      next_distant_sum += backward(i_next,j_next);

    {
      // i= 0

      T sum = 0.0;
      for (int i_next = 0; i_next <= std::min(I-1,5); i_next++) {
	sum += backward(i_next,j_next) * align_model(i_next,0);
      }

      sum += next_distant_sum * long_dist_align_prob[0];
      sum += backward(I,j_next) * align_model(I,0);

      backward(0,j) = sum * dict[target[0]][slookup(j,0)];

      backward(I,j) = sum * cur_emptyword_prob;      
    }

    for (int i=1; i < I; i++) {

      if (i+5 < I)
	next_distant_sum -= backward(i+5,j_next);
      if (i-6 >= 0)
	next_distant_sum += backward(i-6,j_next);

      T sum = 0.0;
      
      for (int i_next = std::max(0,i-5); i_next <= std::min(I-1,i+5); i_next++) {
        sum += backward(i_next,j_next) * align_model(i_next,i);
      }
      sum += next_distant_sum * long_dist_align_prob[i];
      sum += backward(i+I,j_next) * align_model(I,i);

      //stability issues may arise here:
      if (sum <= 0.0) {
	sum = 0.0;
	for (int i_next = std::max(0,i-5); i_next <= std::min(I-1,i+5); i_next++) 
	  sum += backward(i_next,j_next) * align_model(i_next,i);
      }      
      
      backward(i,j) = sum * dict[target[i]][slookup(j,i)];

      backward(i+I,j) = sum * cur_emptyword_prob;
    }
#endif
  }

  if (include_start_alignment) {
    for (int i=0; i < 2*I; i++) {

      const T start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
      backward(i,0) *= start_align_prob;
    }
  }

}

#endif
