/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "alignment_computation.hh"
#include "hmm_forward_backward.hh"

void compute_ibm1_viterbi_alignment(const Storage1D<uint>& source_sentence,
				    const Math2D::Matrix<uint>& slookup,
				    const Storage1D<uint>& target_sentence,
				    const SingleWordDictionary& dict,
				    Storage1D<uint>& viterbi_alignment) {


  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  viterbi_alignment.resize_dirty(J);

  for (uint j=0; j < J; j++) {

    double max_prob = dict[0][source_sentence[j]-1];
    uint arg_max = 0;
    for (uint i=0; i < I; i++) {

      double cur_prob = dict[target_sentence[i]][slookup(j,i)];
      
      if (cur_prob > max_prob) {
	max_prob = cur_prob;
	arg_max = i+1;
      }
    }

    viterbi_alignment[j] = arg_max;
  }
}

void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence,
				    const Math2D::Matrix<uint>& slookup,
				    const Storage1D<uint>& target_sentence,
				    const SingleWordDictionary& dict,
				    const Math2D::Matrix<double>& align_prob,
				    Storage1D<uint>& viterbi_alignment) {

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  viterbi_alignment.resize_dirty(J);

  for (uint j=0; j < J; j++) {

    double max_prob = dict[0][source_sentence[j]-1];
    uint arg_max = 0;
    for (uint i=0; i < I; i++) {

      double cur_prob = dict[target_sentence[i]][slookup(j,i)] * align_prob(j,i);
      
      if (cur_prob > max_prob) {
	max_prob = cur_prob;
	arg_max = i+1;
      }
    }

    viterbi_alignment[j] = arg_max;
  }

}

void compute_fullhmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
				       const Math2D::Matrix<uint>& slookup,
				       const Storage1D<uint>& target_sentence,
				       const SingleWordDictionary& dict,
				       const Math2D::Matrix<double>& align_prob,
				       Storage1D<uint>& viterbi_alignment) {

  const uint I = target_sentence.size();

  Math1D::Vector<double> start_prob(2*I, 1.0 / (2*I));
  compute_ehmm_viterbi_alignment(source_sentence,slookup,target_sentence,dict,align_prob,start_prob,
				 viterbi_alignment);  
}


double compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
				      const Math2D::Matrix<uint>& slookup,
				      const Storage1D<uint>& target_sentence,
				      const SingleWordDictionary& dict,
				      const Math2D::Matrix<double>& align_prob,
				      const Math1D::Vector<double>& initial_prob,
				      Storage1D<uint>& viterbi_alignment, bool internal_mode) {

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  viterbi_alignment.resize_dirty(J);
  assert(align_prob.yDim() == I);

  Math1D::Vector<double> score[2];
  for (uint k=0; k < 2; k++)
    score[k].resize_dirty(2*I);

  Math2D::NamedMatrix<ushort> traceback(2*I,J,MAKENAME(traceback));

  uint cur_idx = 0;
  uint last_idx = 1;

  for (uint i=0; i < I; i++) {
    score[0][i] = dict[target_sentence[i]][slookup(0,i)] * initial_prob[i];
  }
  for (uint i=I; i < 2*I; i++)
    score[0][i] = dict[0][source_sentence[0]-1] * initial_prob[i];

  //to keep the numbers inside double precision
  double correction_factor = score[0].max();
  score[0] *= 1.0 / score[0].max();


  for (uint j=1; j < J; j++) {
    //std::cerr << "j: " << j << std::endl;

    cur_idx = j % 2;
    last_idx = 1 - cur_idx;

    Math1D::Vector<double>& cur_score = score[cur_idx];
    Math1D::Vector<double>& prev_score = score[last_idx];

    for (uint i=0; i < I; i++) {
    
      double max_score = 0.0;
      uint arg_max = MAX_UINT;
      
      for (uint i_prev = 0; i_prev < I; i_prev++) {
	double hyp_score = prev_score[i_prev] * align_prob(i,i_prev);

	if (hyp_score > max_score) {
	  max_score = hyp_score;
	  arg_max = i_prev;
	}
      }
      for (uint i_prev = I; i_prev < 2*I; i_prev++) {
	double hyp_score = prev_score[i_prev] * align_prob(i,i_prev-I);

	if (hyp_score > max_score) {
	  max_score = hyp_score;
	  arg_max = i_prev;
	}
      }

//       if (arg_max == MAX_UINT) {
// 	std::cerr << "ERROR: j=" << j << ", J=" << J << ", I=" << I << std::endl;
//       }

//       assert(arg_max != MAX_UINT);

      cur_score[i] = max_score * dict[target_sentence[i]][slookup(j,i)];
      traceback(i,j) = arg_max;
    }
    for (uint i=I; i < 2*I; i++) {

      double max_score = prev_score[i];
      uint arg_max = i;

      double hyp_score = prev_score[i-I];
      if (hyp_score > max_score) {
	max_score = hyp_score;
	arg_max = i-I;
      }

      cur_score[i] = max_score * dict[0][source_sentence[j]-1] * align_prob(I,i-I);
      traceback(i,j) = arg_max;
    }

    //to keep the numbers inside double precision    
    correction_factor *= cur_score.max();
    cur_score *= 1.0 / cur_score.max();
  }

  /*** now extract Viterbi alignment from the score and the traceback matrix ***/
  double max_score = 0.0;
  uint arg_max = MAX_UINT;

  //std::cerr << "finding max" << std::endl;

  Math1D::Vector<double>& cur_score = score[cur_idx];

  for (uint i=0; i < 2*I; i++) {
    if (cur_score[i] > max_score) {

      max_score = cur_score[i];
      arg_max = i;
    }
  }

  if (arg_max == MAX_UINT) {

    std::cerr << "error: no maximizer for J= " << J << ", I= " << I << std::endl;
    std::cerr << "end-score: " << cur_score << std::endl;
    std::cerr << "align_model: " << align_prob << std::endl;
    std::cerr << "initial_prob: " << initial_prob << std::endl;
    exit(1);
  }

  assert(arg_max != MAX_UINT);
  //std::cerr << "traceback" << std::endl;

  if (internal_mode)
    viterbi_alignment[J-1] = arg_max;
  else
    viterbi_alignment[J-1] = (arg_max < I) ? (arg_max+1) : 0;

  for (int j=J-2; j >= 0; j--) {
    arg_max = traceback(arg_max,j+1);

    assert(arg_max != MAX_UINT);

    if (internal_mode)
      viterbi_alignment[j] = arg_max;
    else
      viterbi_alignment[j] = (arg_max < I) ? (arg_max+1): 0;
  }

  double prob =  max_score * correction_factor;
  return prob;
}


void compute_ehmm_optmarginal_alignment(const Storage1D<uint>& source_sentence,
					const Math2D::Matrix<uint>& slookup,
					const Storage1D<uint>& target_sentence,
					const SingleWordDictionary& dict,
					const Math2D::Matrix<double>& align_prob,
					const Math1D::Vector<double>& initial_prob,
					Storage1D<uint>& optmarginal_alignment) {

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  optmarginal_alignment.resize_dirty(J);

  Math2D::NamedMatrix<long double> forward(2*I,J,MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2*I,J,MAKENAME(backward));

  calculate_hmm_forward(source_sentence, target_sentence, slookup, dict, align_prob,
			initial_prob, forward);

  calculate_hmm_backward(source_sentence, target_sentence, slookup, dict, align_prob,
			 initial_prob, backward);

  for (uint j=0; j < J; j++) {

    const uint s_idx = source_sentence[j];

    long double max_marginal = 0.0;
    uint arg_max = MAX_UINT;

    for (uint i=0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double hyp_marginal = 0.0;

      if (dict[t_idx][slookup(j,i)] > 0.0) {
	hyp_marginal = forward(i,j) * backward(i,j) / dict[t_idx][slookup(j,i)];
      }

      if (hyp_marginal > max_marginal) {

	max_marginal = hyp_marginal;
	arg_max = i;
      }
    }

    for (uint i=I; i < 2*I; i++) {

      long double hyp_marginal = 0.0;

      if (dict[0][s_idx-1] > 0.0) {
	hyp_marginal = forward(i,j) * backward(i,j) / dict[0][s_idx-1];
      }

      if (hyp_marginal > max_marginal) {

	max_marginal = hyp_marginal;
	arg_max = i;
      }      
    }

    assert(arg_max != MAX_UINT);

    if (arg_max < I)
      optmarginal_alignment[j] = arg_max + 1;
    else
      optmarginal_alignment[j] = 0;
  }

}

