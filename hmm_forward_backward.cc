/*** written by Thomas Schoenemann as a private person without employment, November 2009 ***/

#include "hmm_forward_backward.hh"

void calculate_hmm_forward(const Storage1D<uint>& source,
			   const Storage1D<uint>& target,
			   const Math2D::Matrix<uint>& slookup,
			   const SingleWordDictionary& dict,
			   const Math2D::Matrix<double>& align_model,
			   const Math1D::Vector<double>& start_prob,
			   Math2D::Matrix<long double>& forward) {

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
      
      long double sum = 0.0;

      for (uint i_prev=0; i_prev < I; i_prev++)
	sum += align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));

      forward(i,j) = sum * dict[target[i]][slookup(j,i)];
    }
    
    long double cur_emptyword_prob = dict[0][s_idx-1];
    
    for (uint i=I; i < 2*I; i++) {
      
      long double sum = align_model(I,i-I) * ( forward(i,j_prev) + forward(i-I,j_prev) );

      forward(i,j) = sum * cur_emptyword_prob;
    }
  }
}


void calculate_hmm_backward(const Storage1D<uint>& source,
			    const Storage1D<uint>& target,
			    const Math2D::Matrix<uint>& slookup,
			    const SingleWordDictionary& dict,
			    const Math2D::Matrix<double>& align_model,
			    const Math1D::Vector<double>& start_prob,
			    Math2D::Matrix<long double>& backward,
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

    for (uint i=0; i < I; i++) {

      long double sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
	sum += backward(i_next,j_next) * align_model(i_next,i);
      sum += backward(i+I,j_next) * align_model(I,i);

      backward(i,j) = sum * dict[target[i]][slookup(j,i)];
    }
    
    long double cur_emptyword_prob = dict[0][s_idx-1];

    for (uint i=I; i < 2*I; i++) {

      long double sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
	sum += backward(i_next,j_next) * align_model(i_next,i-I);
      sum += backward(i,j_next) * align_model(I,i-I);
	  
      backward(i,j) = sum * cur_emptyword_prob;
    }
  }


  if (include_start_alignment) {
    for (uint i=0; i < 2*I; i++) {

      const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
      backward(i,0) *= start_align_prob;
    }
  }
}

