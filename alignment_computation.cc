/*** written by Thomas Schoenemann, initially as a private person without employment, October 2009 ***
 *** refined at Lund University, Sweden, January 2010 - March 2011 ***
 *** and at the University of Düsseldorf, January 2012 - May 2012 ***/

#include "alignment_computation.hh"
#include "hmm_forward_backward.hh"

void compute_ibm1_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    Storage1D<ushort>& viterbi_alignment) {


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

//posterior decoding for IBM-1
void compute_ibm1_postdec_alignment(const Storage1D<uint>& source_sentence,
				    const Math2D::Matrix<uint>& slookup,
				    const Storage1D<uint>& target_sentence,
				    const SingleWordDictionary& dict,
				    std::set<std::pair<ushort,ushort> >& postdec_alignment,
				    double threshold) {


  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  postdec_alignment.clear();

  for (uint j=0; j < J; j++) {

    //std::cerr << "j:" << j << std::endl;

    double sum = dict[0][source_sentence[j]-1];
    //std::cerr << "null-prob: " << dict[0][source_sentence[j]-1] << std::endl;
    for (uint i=0; i < I; i++) {

      //std::cerr << i << "-prob: " << dict[target_sentence[i]][slookup(j,i)] << std::endl;

      sum += std::max(1e-15,dict[target_sentence[i]][slookup(j,i)]);
    }

    assert(sum > 1e-305);

    for (uint i=0; i < I; i++) {

      double cur_prob = std::max(1e-15,dict[target_sentence[i]][slookup(j,i)]) / sum;
      
      if (cur_prob >= threshold) {
	postdec_alignment.insert(std::make_pair(j+1,i+1));
      }
    }
  }

}


void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    const Math2D::Matrix<double>& align_prob,
                                    Storage1D<ushort>& viterbi_alignment) {

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  viterbi_alignment.resize_dirty(J);

  for (uint j=0; j < J; j++) {

    double max_prob = dict[0][source_sentence[j]-1] * align_prob(j,0);
    uint arg_max = 0;
    for (uint i=0; i < I; i++) {

      double cur_prob = dict[target_sentence[i]][slookup(j,i)] * align_prob(j,i+1);
      
      if (cur_prob > max_prob) {
        max_prob = cur_prob;
        arg_max = i+1;
      }
    }

    viterbi_alignment[j] = arg_max;
  }

}

void compute_ibm2_postdec_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    const Math2D::Matrix<double>& align_prob,
				    std::set<std::pair<ushort,ushort> >& postdec_alignment,
				    double threshold) {

  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  postdec_alignment.clear();

  for (uint j=0; j < J; j++) {

    double sum = dict[0][source_sentence[j]-1] * align_prob(j,0); //CHECK: no alignment prob for alignments to 0??

    for (uint i=0; i < I; i++) 
      sum += std::max(1e-15,dict[target_sentence[i]][slookup(j,i)]) * align_prob(j,i+1);

    assert(sum > 1e-305);

    for (uint i=0; i < I; i++) {

      double marg = std::max(1e-15,dict[target_sentence[i]][slookup(j,i)]) * align_prob(j,i+1) / sum;

      if (marg >= threshold) {

	postdec_alignment.insert(std::make_pair(j+1,i+1));
      }
    }

  }
}

void compute_fullhmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                       const Math2D::Matrix<uint>& slookup,
                                       const Storage1D<uint>& target_sentence,
                                       const SingleWordDictionary& dict,
                                       const Math2D::Matrix<double>& align_prob,
                                       Storage1D<ushort>& viterbi_alignment) {

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
                                      Storage1D<ushort>& viterbi_alignment, bool internal_mode,
				      bool verbose) {

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
    score[0][i] = std::max(1e-15,dict[target_sentence[i]][slookup(0,i)]) * initial_prob[i];
  }
  for (uint i=I; i < 2*I; i++)
    score[0][i] = std::max(1e-15,dict[0][source_sentence[0]-1]) * initial_prob[i];

  //to keep the numbers inside double precision
  double correction_factor = score[0].max();
  score[0] *= 1.0 / score[0].max();

  //   if (J == 27 && I == 36)
  if (verbose)
    std::cerr << "initial score: " << score[0] << std::endl;


  for (uint j=1; j < J; j++) {
    // if (verbose)
    //   std::cerr << "j: " << j << std::endl;

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

      double dict_entry = std::max(1e-15,dict[target_sentence[i]][slookup(j,i)]);

      cur_score[i] = max_score * dict_entry;
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

      double dict_entry = std::max(1e-15,dict[0][source_sentence[j]-1]);

      cur_score[i] = max_score * dict_entry * align_prob(I,i-I);
      traceback(i,j) = arg_max;
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

    // for (uint i=0; i < I; i++) {
    //   std::cerr << "dict for target word #" << i << dict[target_sentence[i]] << std::endl;
    // }
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

  if (verbose && !internal_mode) {

    for (uint j=1; j < J; j++) {

      if (viterbi_alignment[j] > 0 && viterbi_alignment[j-1] > 0)
	std::cerr << "p(" << viterbi_alignment[j] << "|" << viterbi_alignment[j-1] << "): "
		  << align_prob(viterbi_alignment[j]-1,viterbi_alignment[j-1]-1) << std::endl;
    }
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
                                        Storage1D<ushort>& optmarginal_alignment) {

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
    ushort arg_max = MAX_USHORT;

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


void compute_ehmm_postdec_alignment(const Storage1D<uint>& source_sentence,
				    const Math2D::Matrix<uint>& slookup,
				    const Storage1D<uint>& target_sentence,
				    const SingleWordDictionary& dict,
				    const Math2D::Matrix<double>& align_prob,
				    const Math1D::Vector<double>& initial_prob,
				    std::set<std::pair<ushort,ushort> >& postdec_alignment,
				    double threshold) {


  const uint J = source_sentence.size();
  const uint I = target_sentence.size();

  postdec_alignment.clear();

  Math2D::NamedMatrix<long double> forward(2*I,J,MAKENAME(forward));
  Math2D::NamedMatrix<long double> backward(2*I,J,MAKENAME(backward));

  calculate_hmm_forward(source_sentence, target_sentence, slookup, dict, align_prob,
                        initial_prob, forward);

  calculate_hmm_backward(source_sentence, target_sentence, slookup, dict, align_prob,
                         initial_prob, backward);

  long double sent_prob = 0.0;
  for (uint i=0; i < 2*I; i++)
    sent_prob += forward(i,J-1);

  long double inv_sent_prob = 1.0 / sent_prob;

  for (uint j=0; j < J; j++) {

    for (uint i=0; i < I; i++) {

      const uint t_idx = target_sentence[i];

      long double marginal = 0.0;

      if (dict[t_idx][slookup(j,i)] > 1e-75) {
        marginal = forward(i,j) * backward(i,j) * inv_sent_prob / dict[t_idx][slookup(j,i)];
      }

      if (marginal >= threshold) {

	postdec_alignment.insert(std::make_pair(j+1,i+1));
      }
    }
  }

}
