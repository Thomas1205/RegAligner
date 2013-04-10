/*** written by Thomas Schoenemann as a private person without employment, March 2013 ****/

#include "hmm_forward_backward.hh"


void calculate_scaled_hmm_forward(const Storage1D<uint>& source, const Storage1D<uint>& target,
				  const SingleLookupTable& slookup, const SingleWordDictionary& dict,
				  const Math2D::Matrix<double>& align_model,
				  const Math1D::Vector<double>& start_prob,
				  Math2D::Matrix<double>& forward, Math1D::Vector<long double>& scale) {


  const uint I = target.size();
  const uint J = source.size();

  assert(forward.xDim() >= 2*I);
  assert(forward.yDim() >= J);
  assert(scale.size() >= J);

  const uint start_s_idx = source[0];
  double max_entry = 0.0;
  for (uint i=0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward(i,0) = start_align_prob * dict[target[i]][slookup(0,i)];

    if (forward(i,0) > max_entry)
      max_entry = forward(i,0);
  }

  for (uint i=I; i < 2*I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward(i,0) = start_align_prob * dict[0][start_s_idx-1];

    if (forward(i,0) > max_entry)
      max_entry = forward(i,0);
  }

  for (uint i=0; i < 2*I; i++)
    forward(i,0) /= max_entry;
  scale[0] = max_entry;

  
  for (uint j=1; j < J; j++) {
    const uint j_prev = j-1;
    const uint s_idx = source[j];
    
    double max_entry = 0.0;

    for (uint i=0; i < I; i++) {
      
      double sum = 0.0;

      for (uint i_prev=0; i_prev < I; i_prev++)
        sum += align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+I,j_prev));

      forward(i,j) = sum * dict[target[i]][slookup(j,i)];

      if (forward(i,j) > max_entry)
	max_entry = forward(i,j);
    }
    
    const double cur_emptyword_prob = dict[0][s_idx-1];
    
    for (uint i=I; i < 2*I; i++) {
      
      const double sum = align_model(I,i-I) * ( forward(i,j_prev) + forward(i-I,j_prev) );

      forward(i,j) = sum * cur_emptyword_prob;

      if (forward(i,j) > max_entry)
	max_entry = forward(i,j);
    }

    for (uint i=0; i < 2*I; i++)
      forward(i,0) /= max_entry;
    scale[j] =  scale[j-1] * max_entry;
  }
}


double calculate_hmm_forward_log_sum(const Storage1D<uint>& source_sentence,
				     const Storage1D<uint>& target_sentence,
				     const SingleLookupTable& slookup,
				     const SingleWordDictionary& dict,
				     const Math2D::Matrix<double>& align_model,
				     const Math1D::Vector<double>& start_prob,
				     const HmmAlignProbType align_type) {

  if (align_type == HmmAlignProbReducedpar)
    return calculate_hmm_forward_log_sum_with_tricks(source_sentence,target_sentence,slookup,dict,align_model,start_prob);
  else
    return calculate_hmm_forward_log_sum(source_sentence,target_sentence,slookup,dict,align_model,start_prob);
}


double calculate_hmm_forward_log_sum(const Storage1D<uint>& source,
				     const Storage1D<uint>& target,
				     const SingleLookupTable& slookup,
				     const SingleWordDictionary& dict,
				     const Math2D::Matrix<double>& align_model,
				     const Math1D::Vector<double>& start_prob) {
  
  const uint I = target.size();
  const uint J = source.size();

  Math1D::Vector<double> forward[2];
  forward[0].resize(2*I);
  forward[1].resize(2*I);

  double log_sum = 0.0;

  /*** init ***/
  const uint start_s_idx = source[0];
  for (uint i=0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward[0][i] = start_align_prob * dict[target[i]][slookup(0,i)];
  }

  for (uint i=I; i < 2*I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward[0][i] = start_align_prob * dict[0][start_s_idx-1];
  }

  double scale = forward[0].max();
  log_sum = std::log(scale);

  forward[0] *= 1.0 / scale;

  /*** proceed ***/

  uint cur_idx = 0;

  for (uint j=1; j < J; j++) {

    const Math1D::Vector<double>& prev_forward = forward[cur_idx];

    cur_idx = 1-cur_idx;
    
    Math1D::Vector<double>& cur_forward = forward[cur_idx];
    
    const uint s_idx = source[j];
    
    for (uint i=0; i < I; i++) {
      
      double sum = 0.0;

      for (uint i_prev=0; i_prev < I; i_prev++)
        sum += align_model(i,i_prev) * (prev_forward[i_prev] + prev_forward[i_prev+I]);

      cur_forward[i] = sum * dict[target[i]][slookup(j,i)];
    }
    
    const double cur_emptyword_prob = dict[0][s_idx-1];
    
    for (uint i=I; i < 2*I; i++) {
      
      const double sum = align_model(I,i-I) * ( prev_forward[i] + prev_forward[i-I] );

      cur_forward[i] = sum * cur_emptyword_prob;
    }

    double scale = cur_forward.max();
    log_sum += std::log(scale);

    cur_forward *= 1.0 / scale;
  }

  return log_sum + std::log(forward[cur_idx].sum());
}


double calculate_hmm_forward_log_sum_with_tricks(const Storage1D<uint>& source,
						 const Storage1D<uint>& target,
						 const SingleLookupTable& slookup,
						 const SingleWordDictionary& dict,
						 const Math2D::Matrix<double>& align_model,
						 const Math1D::Vector<double>& start_prob) {


  const int I = target.size();
  const int J = source.size();

  Math1D::Vector<double> long_dist_align_prob(I,0.0);
  for (int i=0; i < I; i++) {

    if (i + 6 < I)
      long_dist_align_prob[i] = align_model(i+6,i);
    else if (i - 6 >= 0)
      long_dist_align_prob[i] = align_model(i-6,i);
  }

  Math1D::Vector<double> forward[2];
  forward[0].resize(2*I);
  forward[1].resize(2*I);

  double log_sum = 0.0;

  /*** init ***/
  const uint start_s_idx = source[0];
  for (int i=0; i < I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward[0][i] = start_align_prob * dict[target[i]][slookup(0,i)];
  }

  for (int i=I; i < 2*I; i++) {
    const double start_align_prob = (start_prob.size() == 0) ? 1.0 / (2*I) : start_prob[i];
    forward[0][i] = start_align_prob * dict[0][start_s_idx-1];
  }

  double scale = forward[0].max();
  log_sum = std::log(scale);

  forward[0] *= 1.0 / scale;

  /*** proceed ***/

  uint cur_idx = 0;

  for (int j=1; j < J; j++) {

    const Math1D::Vector<double>& prev_forward = forward[cur_idx];

    cur_idx = 1-cur_idx;
    
    Math1D::Vector<double>& cur_forward = forward[cur_idx];

    const uint s_idx = source[j];

    //NOTE: we are exploiting here that p(i|i_prev) does not depend on i for
    //   the considered i_prev's. But it DOES depend on i_prev
    
    /*** 1. real words ***/
    
    double prev_distant_sum = 0.0;
    for (int i_prev = 6; i_prev < I; i_prev++)
      prev_distant_sum += (prev_forward[i_prev] + prev_forward[i_prev+I]) * long_dist_align_prob[i_prev];

    {
      //i=0
      double sum = 0.0;

      for (int i_prev= 0; i_prev <= std::min(I-1,5); i_prev++) {
        sum += align_model(0,i_prev) * (prev_forward[i_prev] + prev_forward[i_prev+I]);
      }
      sum += prev_distant_sum;

      cur_forward[0] = sum * dict[target[0]][slookup(j,0)];
    }

    for (int i=1; i < I; i++) {

      if (i+5 < I)
	prev_distant_sum -= (prev_forward[i+5] + prev_forward[i+5+I]) * long_dist_align_prob[i+5];
      if (i-6 >= 0)
	prev_distant_sum += (prev_forward[i-6] + prev_forward[i-6+I]) * long_dist_align_prob[i-6];

      double sum = 0.0;

      for (int i_prev= std::max(0,i-5); i_prev <= std::min(I-1,i+5); i_prev++) {
        sum += align_model(i,i_prev) * (prev_forward[i_prev] + prev_forward[i_prev+I]);
      }
      sum += prev_distant_sum;

      //stability issues may arise here:
      if (sum <= 0.0) {
	sum = 0.0;
	for (int i_prev= std::max(0,i-5); i_prev <= std::min(I-1,i+5); i_prev++) 
	  sum += align_model(i,i_prev) * (prev_forward[i_prev] + prev_forward[i_prev+I]);
      }

      cur_forward[i] = sum * dict[target[i]][slookup(j,i)];
    }


    /*** 2. empty words ***/

    const double cur_emptyword_prob = dict[0][s_idx-1];
    
    for (int i=I; i < 2*I; i++) {
      
      const double sum = align_model(I,i-I) * ( prev_forward[i] + prev_forward[i-I] );

      cur_forward[i] = sum * cur_emptyword_prob;
    }


    double scale = cur_forward.max();
    log_sum += std::log(scale);

    cur_forward *= 1.0 / scale;
  }

  return log_sum + std::log(forward[cur_idx].sum());
}


void calculate_scaled_hmm_backward(const Storage1D<uint>& source, const Storage1D<uint>& target,
				   const SingleLookupTable& slookup,
				   const SingleWordDictionary& dict,
				   const Math2D::Matrix<double>& align_model,
				   const Math1D::Vector<double>& /*start_prob*/,
				   Math2D::Matrix<double>& backward, Math1D::Vector<long double>& scale) {


  const uint I = target.size();
  const uint J = source.size();

  assert(backward.xDim() >= 2*I);
  assert(backward.yDim() >= J);
  assert(scale.size() >= J);

  const uint end_s_idx = source[J-1];

  double max_entry = 0.0;
  
  for (uint i=0; i < I; i++) {
    backward(i,J-1) = dict[target[i]][slookup(J-1,i)];

    if (max_entry < backward(i,J-1))
      max_entry = backward(i,J-1);
  }
  for (uint i=I; i < 2*I; i++) {
    backward(i,J-1) = dict[0][end_s_idx-1];

    if (max_entry < backward(i,J-1))
      max_entry = backward(i,J-1);
  }      

  scale[J-1] = max_entry;
  for (uint i=0; i < 2*I; i++)
    backward(i,J-1) /= max_entry;

  for (int j=J-2; j >= 0; j--) {
    const uint s_idx = source[j];
    const uint j_next = j+1;
    
    const double cur_emptyword_prob = dict[0][s_idx-1];

    double max_entry = 0.0;
    
    for (uint i=0; i < I; i++) {

      double sum = 0.0;
      for (uint i_next = 0; i_next < I; i_next++)
        sum += backward(i_next,j_next) * align_model(i_next,i);
      sum += backward(i+I,j_next) * align_model(I,i);

      backward(i,j) = sum * dict[target[i]][slookup(j,i)];

      if (backward(i,j) > max_entry)
	max_entry = backward(i,j);

      backward(i+I,j) = sum * cur_emptyword_prob;

      if (backward(i+I,j) > max_entry)
	max_entry = backward(i+I,j);
    }

    for (uint i=0; i < 2*I; i++)
      backward(i,j) /= max_entry;

    scale[j] = scale[j+1] *  max_entry;
  }
}
