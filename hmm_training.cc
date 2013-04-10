/*** written by Thomas Schoenemann as a private person without employment, October 2009 
 *** later as an employee of Lund University, 2010 - Mar. 2011
 *** later as a private person, and finally at the University of Düsseldorf, Germany, January - September 2012  ***/

#include "hmm_training.hh"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "matrix.hh"

#include "training_common.hh"
#include "ibm1_training.hh"
#include "hmm_forward_backward.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"

#include "projection.hh"
#include "stl_out.hh"


HmmOptions::HmmOptions(uint nSourceWords,uint nTargetWords,
                       std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                       std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments) :
  nIterations_(5), init_type_(HmmInitPar), align_type_(HmmAlignProbReducedpar), start_empty_word_(false), smoothed_l0_(false),
  l0_beta_(1.0), print_energy_(true), nSourceWords_(nSourceWords), nTargetWords_(nTargetWords), 
  init_m_step_iter_(1000), align_m_step_iter_(1000), dict_m_step_iter_(45), transfer_mode_(IBM1TransferNo),
  sure_ref_alignments_(sure_ref_alignments), possible_ref_alignments_(possible_ref_alignments){}


HmmWrapper::HmmWrapper(const FullHMMAlignmentModel& align_model,
		       const InitialAlignmentProbability& initial_prob,
		       const HmmOptions& hmm_options) 
  : align_model_(align_model), initial_prob_(initial_prob), hmm_options_(hmm_options)  {}


long double hmm_alignment_prob(const Storage1D<uint>& source, 
                               const SingleLookupTable& slookup,
                               const Storage1D<uint>& target,
                               const SingleWordDictionary& dict,
                               const FullHMMAlignmentModel& align_model,
                               const InitialAlignmentProbability& initial_prob,
                               const Storage1D<AlignBaseType>& alignment, bool with_dict) {

  const uint I = target.size();
  const uint J = source.size();

  assert(J == alignment.size());

  long double prob = (alignment[0] == 2*I) ? initial_prob[I-1][I] : initial_prob[I-1][alignment[0]];

  if (with_dict) {
    for (uint j=0; j < J; j++) {
      uint aj = alignment[j];
      if (aj < I)
        prob *= dict[target[aj]][slookup(j,aj)];
      else
        prob *= dict[0][source[j]-1];
    }
  }

  for (uint j=1; j < alignment.size(); j++) {
    uint prev_aj = alignment[j-1];
    if (prev_aj >= I)
      prev_aj -= I;

    uint aj = alignment[j];
    if (aj >= I)
      assert(aj == prev_aj+I);
    
    prob *= align_model[I-1](std::min(I,aj),prev_aj);
  }

  return prob;
}

void external2internal_hmm_alignment(const Storage1D<AlignBaseType>& ext_alignment, uint curI,
				     const HmmOptions& options, Storage1D<AlignBaseType>& int_alignment) {


  int_alignment.resize(ext_alignment.size());

  int jj=-1;

  for (uint j=0; j < ext_alignment.size(); j++) {

    uint ext_aj = ext_alignment[j];
    
    if (ext_aj > 0) {
      int_alignment[j] = ext_aj-1;
      jj = j;
    }
    else {

      if (jj >= 0) {
	int_alignment[j] = ext_alignment[jj]-1 + curI;
      }
      else
	int_alignment[j] = (options.start_empty_word_) ? 2*curI : curI;
    }
  }
}


double extended_hmm_perplexity(const Storage1D<Storage1D<uint> >& source, 
                               const LookupTable& slookup,
                               const Storage1D<Storage1D<uint> >& target,
                               const FullHMMAlignmentModel& align_model,
                               const InitialAlignmentProbability& initial_prob,
                               const SingleWordDictionary& dict,
                               const CooccuringWordsType& wcooc, uint nSourceWords,
			       HmmAlignProbType align_type, bool start_empty_word) {

  double sum = 0.0;

  const size_t nSentences = target.size();

  SingleLookupTable aux_lookup;
  
  for (size_t s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);

    const uint curI = cur_target.size();
    
    const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];


    if (start_empty_word) {

      /**** calculate forward ********/

      const uint curJ = cur_source.size();

      Math2D::NamedMatrix<double> forward(2*curI,curJ,MAKENAME(forward));


      calculate_sehmm_forward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                              initial_prob[curI-1], forward);

      double sentence_prob = 0.0;
      for (uint i=0; i < forward.xDim(); i++) {
	
	// 	if (!(forward(i,curJ-1) >= 0.0)) {
	
	// 	  std::cerr << "s=" << s << ", I=" << curI << ", i= " << i << ", value " << forward(i,curJ-1) << std::endl;
	// 	}
	
	assert(forward(i,curJ-1) >= 0.0);
	sentence_prob += forward(i,curJ-1);
      }
      
      if (sentence_prob > 1e-300)
	sum -= std::log(sentence_prob);
      else
	sum -= std::log(1e-300);
    }
    else {
    
      sum -= calculate_hmm_forward_log_sum(cur_source, cur_target, cur_lookup, dict, cur_align_model,
					   initial_prob[curI-1], align_type);
    }
  }


  return sum / nSentences;
}


double extended_hmm_energy(const Storage1D<Storage1D<uint> >& source, 
                           const LookupTable& slookup,
                           const Storage1D<Storage1D<uint> >& target,
                           const FullHMMAlignmentModel& align_model,
                           const InitialAlignmentProbability& initial_prob,
                           const SingleWordDictionary& dict,
                           const CooccuringWordsType& wcooc, uint nSourceWords,
                           const floatSingleWordDictionary& prior_weight,
			   HmmAlignProbType align_type, bool start_empty_word,
			   bool smoothed_l0, double l0_beta) {
  
  double energy = 0.0;

  for (uint i=0; i < dict.size(); i++)
    for (uint k=0; k < dict[i].size(); k++) {
      if (smoothed_l0)
	energy += prior_weight[i][k] * prob_penalty(dict[i][k],l0_beta);
      else
	energy += prior_weight[i][k] * dict[i][k];
    }


  energy /= source.size();

  energy += extended_hmm_perplexity(source,slookup,target,align_model,initial_prob,dict,wcooc,nSourceWords,align_type,start_empty_word);

  return energy;
}

double ehmm_m_step_energy(const FullHMMAlignmentModel& facount, const Math1D::Vector<double>& dist_params, 
                          uint zero_offset, double grouping_param = -1.0) {

  double energy = 0.0;

  for (uint I=1; I <= facount.size(); I++) {

    if (facount[I-1].size() > 0) {
      
      for (int i=0; i < (int) I; i++) {

        double non_zero_sum = 0.0;

        if (grouping_param < 0.0) {
	
          for (uint ii=0; ii < I; ii++)
            non_zero_sum += std::max(1e-15,dist_params[zero_offset + ii - i]);
	  
          for (int ii=0; ii < (int) I; ii++) {
	    
            const double cur_count = facount[I-1](ii,i);
	    
            energy -= cur_count * std::log( std::max(1e-15,dist_params[zero_offset + ii - i]) / non_zero_sum);
          }
        }
        else {

          double grouping_norm = std::max(0,i-5);
          grouping_norm += std::max(0,int(I)-1-(i+5));

	  
          for (int ii=0; ii < (int) I; ii++) {
            if (abs(ii-i) <= 5) 
              non_zero_sum += std::max(1e-15,dist_params[zero_offset + ii - i]);
            else {
              non_zero_sum += grouping_param / grouping_norm;
            }
          }

          for (int ii=0; ii < (int) I; ii++) {
	    
            double cur_count = facount[I-1](ii,i);
            double cur_param = std::max(1e-15,dist_params[zero_offset + ii - i]);

            if (abs(ii-i) > 5) {
              cur_param = std::max(1e-15,grouping_param / grouping_norm);
            }

            energy -= cur_count * std::log( cur_param / non_zero_sum);
          }
        }
      }
    }
  }

  return energy;
}

void ehmm_m_step(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset,
                 uint nIter, double& grouping_param) {

  if (grouping_param < 0.0)
    projection_on_simplex(dist_params.direct_access(),dist_params.size());
  else {
    projection_on_simplex_with_slack(dist_params.direct_access() + zero_offset - 5, grouping_param, 11); 
  }
  
  Math1D::Vector<double> m_dist_grad = dist_params;
  Math1D::Vector<double> new_dist_params = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  double m_grouping_grad = 0.0;
  double new_grouping_param = grouping_param;
  double hyp_grouping_param = grouping_param;

  double energy = ehmm_m_step_energy(facount, dist_params, zero_offset, grouping_param);

  assert(grouping_param < 0.0 || grouping_param >= 1e-15);

  double alpha = 100.0;

  for (uint iter = 1; iter <= nIter; iter++) {

    if ((iter % 5) == 0)
      std::cerr << "m-step gd-iter #" << iter << ", cur energy: " << energy << std::endl;

    m_dist_grad.set_constant(0.0);
    m_grouping_grad = 0.0;

    //calculate gradient
    for (uint I=1; I <= facount.size(); I++) {

      if (facount[I-1].size() > 0) {

        for (int i=0; i < (int) I; i++) {

          double grouping_norm = std::max(0,i-5);
          grouping_norm += std::max(0,int(I)-1-(i+5));
	    
          double non_zero_sum = 0.0;
          for (int ii=0; ii < (int) I; ii++) {
            if (grouping_param < 0.0 || abs(i-ii) <= 5)
              non_zero_sum +=  std::max(1e-15,dist_params[zero_offset + ii - i]);
            else
              non_zero_sum += std::max(1e-15,grouping_param / grouping_norm);
          }
	  
          double count_sum = 0.0;
          for (int ii=0; ii < (int) I; ii++) {
            count_sum += facount[I-1](ii,i);
          }
	   
          for (int ii=0; ii < (int) I; ii++) {
            double cur_param = std::max(1e-15,dist_params[zero_offset + ii - i]);

            double cur_count = facount[I-1](ii,i);
	    
            if (grouping_param < 0.0) {
              m_dist_grad[zero_offset + ii-i] -= cur_count / cur_param;
            }
            else {
	
              if (abs(ii-i) > 5) {
		//NOTE: -std::log( param / norm) = -std::log(param) + std::log(norm)
		// => grouping_norm does NOT enter here
                m_grouping_grad -= cur_count / grouping_param;  
              }
              else {
                m_dist_grad[zero_offset + ii-i] -= cur_count / cur_param;
              }
            }
          }

          for (int ii=0; ii < (int) I; ii++) {
            if (grouping_param < 0.0 || abs(ii-i) <= 5)
              m_dist_grad[zero_offset + ii-i] += count_sum / non_zero_sum;
          }

	  if (grouping_param >= 0.0 && grouping_norm > 0.0)
	    m_grouping_grad += count_sum / non_zero_sum;
        }
      }
    }

    //go in gradient direction

    double real_alpha = alpha;

    //TRIAL
    // double sqr_grad_norm  = 0.0;
    // for (uint k=0; k < dist_params.size(); k++)
    //   sqr_grad_norm += m_dist_grad.direct_access(k) * m_dist_grad.direct_access(k);
    // sqr_grad_norm += m_grouping_grad * m_grouping_grad;

    // real_alpha /= sqrt(sqr_grad_norm);
    //END_TRIAL


    for (uint k=0; k < dist_params.size(); k++)
      new_dist_params.direct_access(k) = dist_params.direct_access(k) - real_alpha * m_dist_grad.direct_access(k);

    new_grouping_param = grouping_param - real_alpha * m_grouping_grad;

    for (uint k=0; k < dist_params.size(); k++) {
      if (dist_params[k] >= 1e75)
	dist_params[k] = 9e74;
      else if (dist_params[k] <= -1e75)
	dist_params[k] = -9e74;
    }
    if (new_grouping_param >= 1e75)
      new_grouping_param = 9e74;
    else if (new_grouping_param <= -1e75)
      new_grouping_param = -9e74;

    // reproject
    if (grouping_param < 0.0)  {   
      //projection_on_simplex(new_dist_params.direct_access(),dist_params.size());
      fast_projection_on_simplex(new_dist_params.direct_access(),dist_params.size());
    }
    else {
      //projection_on_simplex_with_slack(new_dist_params.direct_access()+zero_offset-5,new_grouping_param,11);
      fast_projection_on_simplex_with_slack(new_dist_params.direct_access()+zero_offset-5,new_grouping_param,11);
    }

    //find step-size

    new_grouping_param = std::max(new_grouping_param,1e-15);

    double best_energy = 1e300; 
    
    double lambda = 1.0;
    double line_reduction_factor = 0.5;
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

      for (uint k=0; k < dist_params.size(); k++)
        hyp_dist_params.direct_access(k) = lambda * new_dist_params.direct_access(k) + neg_lambda * dist_params.direct_access(k);

      if (grouping_param >= 0.0)
	hyp_grouping_param = std::max(1e-15,lambda * new_grouping_param + neg_lambda * grouping_param);

      double new_energy = ehmm_m_step_energy(facount, hyp_dist_params, zero_offset, hyp_grouping_param);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    // if (nIter > 4)
    //   alpha *= 1.5;

    if (nIter > 15 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }
      
    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k=0; k < dist_params.size(); k++)
      dist_params.direct_access(k) = neg_best_lambda * dist_params.direct_access(k) + best_lambda * new_dist_params.direct_access(k);

    if (grouping_param >= 0.0) 
      grouping_param = std::max(1e-15,best_lambda * new_grouping_param + neg_best_lambda * grouping_param);
  }

}


double ehmm_init_m_step_energy(const InitialAlignmentProbability& init_acount, const Math1D::Vector<double>& init_params) {

  double energy = 0.0;

  for (uint I=0; I < init_acount.size(); I++) {

    if (init_acount[I].size() > 0) {

      double non_zero_sum = 0.0;
      for (uint i=0; i < I; i++)
        non_zero_sum += std::max(1e-15,init_params[i]);
      
      for (uint i=0; i < I; i++) {
	
        energy -= init_acount[I][i] * std::log( std::max(1e-15,init_params[i]) / non_zero_sum);
      }
    }
  }

  return energy;
}


void ehmm_init_m_step(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter) {

  projection_on_simplex(init_params.direct_access(), init_params.size());

  Math1D::Vector<double> m_init_grad = init_params;
  Math1D::Vector<double> new_init_params = init_params;
  Math1D::Vector<double> hyp_init_params = init_params;

  double energy = ehmm_init_m_step_energy(init_acount, init_params);

  std::cerr << "start m-energy: " << energy << std::endl;

  for (uint iter = 1; iter <= nIter; iter++) {

    m_init_grad.set_constant(0.0);

    //calculate gradient
    for (uint I=0; I < init_acount.size(); I++) {

      if (init_acount[I].size() > 0) {

        double non_zero_sum = 0.0;
        for (uint i=0; i <= I; i++)
          non_zero_sum += std::max(1e-15,init_params[i]);
	
        double count_sum = 0.0;
        for (uint i=0; i <= I; i++) {
          count_sum += init_acount[I][i];

          double cur_param = std::max(1e-15,init_params[i]);
	  
          m_init_grad[i] -= init_acount[I][i] / cur_param;
        }
	
        for (uint i=0; i <= I; i++) {
          m_init_grad[i] += count_sum / non_zero_sum;
        }
      }
    }    

    double alpha  = 0.0001;

    for (uint k=0; k < init_params.size(); k++) {
      new_init_params.direct_access(k) = init_params.direct_access(k) - alpha * m_init_grad.direct_access(k);
      assert(!isnan(new_init_params[k]));
    }

    // reproject
    for (uint k=0; k < init_params.size(); k++) {
      if (new_init_params[k] >= 1e75)
	new_init_params[k] = 9e74;
      if (new_init_params[k] <= -1e75)
	new_init_params[k] = -9e74;
    }
    projection_on_simplex(new_init_params.direct_access(),init_params.size());

    //find step-size
    double best_energy = 1e300; 

    double lambda = 1.0;
    double line_reduction_factor = 0.5;
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

      for (uint k=0; k < init_params.size(); k++)
        hyp_init_params.direct_access(k) = lambda * new_init_params.direct_access(k) + neg_lambda * init_params.direct_access(k);
      
      double hyp_energy = ehmm_init_m_step_energy(init_acount, hyp_init_params);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k=0; k < init_params.size(); k++)
      init_params.direct_access(k) = neg_best_lambda * init_params.direct_access(k) + best_lambda * new_init_params.direct_access(k);

    if (nIter > 15 ||  best_lambda < 1e-12 || fabs(energy - best_energy) < 1e-4 ) {
      std::cerr << "CUTOFF" << std::endl;
      break;
    } 

    energy = best_energy;

    if ((iter % 5) == 0)
      std::cerr << "init m-step gd-iter #" << iter << ", energy: " << energy << std::endl;
  }
}

void par2nonpar_hmm_init_model(const Math1D::Vector<double>& init_params, const Math1D::Vector<double>& source_fert,
                               HmmInitProbType init_type, InitialAlignmentProbability& initial_prob, bool start_empty_word) {

  for (uint I=1; I <= initial_prob.size(); I++) {

    if (initial_prob[I-1].size() > 0) {

      if (init_type == HmmInitPar) {
 
        double norm = 0.0;
        for (uint i=0; i < I; i++)
          norm += init_params[i];
	
        double inv_norm = 1.0 / norm;
        for (uint i=0; i < I; i++) {
          initial_prob[I-1][i] = std::max(1e-8,source_fert[1] * inv_norm * init_params[i]);
	  assert(!isnan(initial_prob[I-1][i]));
	}
        if (!start_empty_word) {
          for (uint i=I; i < 2*I; i++) {
            initial_prob[I-1][i] = source_fert[0] / I;
            assert(!isnan(initial_prob[I-1][i]));
          }
        }
        else
          initial_prob[I-1][I] = source_fert[0] / I;
      }
      else if (init_type == HmmInitFix) {
        if (!start_empty_word) 
          initial_prob[I-1].set_constant(0.5/I);
        else {
          initial_prob[I-1].set_constant(1.0/(I+1));
        }
      }
      else if (init_type == HmmInitFix2) {
        for (uint i=0; i < I; i++) 
          initial_prob[I-1][i] = 0.98 / I;
        if (!start_empty_word) {
          for (uint i=I; i < 2*I; i++) 
            initial_prob[I-1][i] = 0.02 / I;
        }
        else {
          initial_prob[I-1][I] = 0.02;
        }
      }
    }
  }

}


void par2nonpar_hmm_alignment_model(const Math1D::Vector<double>& dist_params, const uint zero_offset,
                                    const double dist_grouping_param, const Math1D::Vector<double>& source_fert,
                                    HmmAlignProbType align_type, FullHMMAlignmentModel& align_model) {

  for (uint I=1; I <= align_model.size(); I++) {
    
    if (align_model[I-1].size() > 0) {

      for (int i=0; i < (int) I; i++) {

        double grouping_norm = std::max(0,i-5);
        grouping_norm += std::max(0,int(I)-1-(i+5));

        double non_zero_sum = 0.0;
        for (int ii=0; ii < (int) I; ii++) {
          if (align_type != HmmAlignProbReducedpar || abs(ii-i) <= 5)
            non_zero_sum += dist_params[zero_offset + ii - i];
        }

        if (align_type == HmmAlignProbReducedpar && grouping_norm > 0.0) {
          non_zero_sum += dist_grouping_param;
        }

	assert(non_zero_sum > 1e-305);
        double inv_sum = 1.0 / non_zero_sum;

        for (int ii=0; ii < (int) I; ii++) {
          if (align_type == HmmAlignProbReducedpar && abs(ii-i) > 5) {
	    assert(!isnan(grouping_norm));
	    assert(grouping_norm > 0.0);
            align_model[I-1](ii,i) = std::max(1e-8,source_fert[1] * inv_sum * dist_grouping_param / grouping_norm);
          }
          else {
	    assert(dist_params[zero_offset + ii - i] >= 0);
            align_model[I-1](ii,i) = std::max(1e-8,source_fert[1] * inv_sum * dist_params[zero_offset + ii - i]);
	  }
	  assert(!isnan(align_model[I-1](ii,i)));
	  assert(align_model[I-1](ii,i) >= 0.0);
        }
        align_model[I-1](I,i) = source_fert[0];
	assert(!isnan(align_model[I-1](I,i)));
	assert(align_model[I-1](I,i) >= 0.0);
      }
    }
  }
}


void init_hmm_from_ibm1(const Storage1D<Storage1D<uint> >& source, 
                        const LookupTable& slookup,
                        const Storage1D<Storage1D<uint> >& target,
                        const SingleWordDictionary& dict,
                        const CooccuringWordsType& wcooc,
                        FullHMMAlignmentModel& align_model,
                        Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                        Math1D::Vector<double>& source_fert,
                        InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params,
                        const HmmOptions& options,
                        IBM1TransferMode transfer_mode = IBM1TransferViterbi) {

  dist_grouping_param = -1.0;

  HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  bool start_empty_word = options.start_empty_word_;

  if (init_type >= HmmInitInvalid) {
    
    INTERNAL_ERROR << "invalid type for HMM initial alignment model" << std::endl;
    exit(1);
  }
  if (align_type >= HmmAlignProbInvalid) {

    INTERNAL_ERROR << "invalid type for HMM alignment model" << std::endl;
    exit(1);
  }

  SingleLookupTable aux_lookup;

  const uint nSentences = source.size();

  std::set<uint> seenIs;

  uint maxI = 6;
  uint maxJ = 0;
  for (size_t s=0; s < nSentences; s++) {
    const uint curI = target[s].size();
    const uint curJ = source[s].size();

    seenIs.insert(curI);

    if (curI > maxI)
      maxI = curI;
    if (curJ > maxJ)
      maxJ = curJ;
  }

  
  uint zero_offset = maxI-1;
  
  dist_params.resize(2*maxI-1, 1.0 / (2*maxI-1)); //even for nonpar, we will use this as initialization

  if (align_type == HmmAlignProbReducedpar) {
    dist_grouping_param = 0.2;
    dist_params.set_constant(0.0);
    for (int k=-5; k <= 5; k++) 
      dist_params[zero_offset+k] = 0.8 / 11.0;
  }

  source_fert.resize(2); //even for nonpar, we will use this as initialization
  source_fert[0] = 0.02;
  source_fert[1] = 0.98;

  if (init_type == HmmInitPar) {
    init_params.resize(maxI);
    init_params.set_constant(1.0 / maxI);
  }

  align_model.resize_dirty(maxI); //note: access using I-1
  initial_prob.resize(maxI);

  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      //x = new index, y = given index
      align_model[I-1].resize_dirty(I+1,I); //because of empty words
      align_model[I-1].set_constant(1.0 / (I+1));

      if (align_type != HmmAlignProbNonpar) {
        for (uint i=0; i < I; i++) {
          for (uint ii=0; ii < I; ii++) {
            align_model[I-1](ii,i) = source_fert[1] / I;
          }
          align_model[I-1](I,i) = source_fert[0];
        }
      }

      if (!start_empty_word)
        initial_prob[I-1].resize_dirty(2*I);
      else
        initial_prob[I-1].resize_dirty(I+1);
      if (init_type != HmmInitPar) {
        if (init_type == HmmInitFix2) {
          for (uint i=0; i < I; i++) 
            initial_prob[I-1][i] = 0.98 / I;
          for (uint i=I; i < initial_prob[I-1].size(); i++)
            initial_prob[I-1][i] = 0.02 / (initial_prob[I-1].size()-I);
        }
        else
          initial_prob[I-1].set_constant(1.0 / initial_prob[I-1].size());
      }
      else {
        for (uint i=0; i < I; i++)
          initial_prob[I-1][i] = source_fert[1] / I;
        if (!start_empty_word) {
          for (uint i=I; i < 2*I; i++)
            initial_prob[I-1][i] = source_fert[0] / I;
        }
        else
          initial_prob[I-1][I] = source_fert[0];
      }
    }
  }

  if (transfer_mode != IBM1TransferNo) {
    if (align_type == HmmAlignProbReducedpar)
      dist_grouping_param = 0.0;
    dist_params.set_constant(0.0);
    for (uint s=0; s < source.size(); s++) {

      const SingleLookupTable& cur_lookup = get_wordlookup(source[s],target[s],wcooc,options.nSourceWords_,slookup[s],aux_lookup);

      if (transfer_mode == IBM1TransferViterbi) {

        Storage1D<AlignBaseType> viterbi_alignment(source[s].size(),0);
        
        compute_ibm1_viterbi_alignment(source[s],cur_lookup,target[s], dict, viterbi_alignment);
        
        for (uint j=1; j < source[s].size(); j++) {
          
          int prev_aj = viterbi_alignment[j-1];
          int cur_aj = viterbi_alignment[j];
          
          if (prev_aj != 0 && cur_aj != 0) {
            int diff = cur_aj - prev_aj;
            
            if (abs(diff) <= 5 || align_type != HmmAlignProbReducedpar)
              dist_params[zero_offset+diff] += 1.0;
            else
              dist_grouping_param += 1.0;
          }
        }
      }
      else {
        assert(transfer_mode == IBM1TransferPosterior);
        
        for (uint j=1; j < source[s].size(); j++) {

          double sum = dict[0][source[s][j]-1];
          for (uint i=0; i < target[s].size(); i++)
            sum += dict[target[s][i]][cur_lookup(j,i)];

          double prev_sum = dict[0][source[s][j-1]-1];
          for (uint i=0; i < target[s].size(); i++)
            prev_sum += dict[target[s][i]][cur_lookup(j-1,i)];

          for (int i1 = 0; i1 < int(target[s].size()); i1++) {

            const double i1_weight = (dict[target[s][i1]][cur_lookup(j-1,i1)] / prev_sum);

            for (int i2 = 0; i2 < int(target[s].size()); i2++) {

              const double marg = (dict[target[s][i2]][cur_lookup(j,i2)] / sum) * i1_weight;

              int diff = i2 - i1;
              if (abs(diff) <= 5 || align_type != HmmAlignProbReducedpar)
                dist_params[zero_offset+diff] += marg;
              else
                dist_grouping_param += marg;
            }
          }
        }        
      }
    }
    double sum = dist_params.sum();
    if (align_type == HmmAlignProbReducedpar)
      sum += dist_grouping_param;
    dist_params *= 1.0 / sum;
    if (align_type == HmmAlignProbReducedpar) {
      dist_grouping_param *= 1.0 / sum;
  
      for (int k=-5; k <= 5; k++) 
        dist_params[zero_offset+k] = 0.75 * dist_params[maxI+k] + 0.25 * 0.8 / 11.0;
    
      dist_grouping_param = 0.75 * dist_grouping_param + 0.25 * 0.2;
    }
    else {
      for (uint k=0; k < dist_params.size(); k++) {
        dist_params[k] = 0.75 * dist_params[k] + 0.25 / dist_params.size();
      }
    }
  }

  HmmAlignProbType align_mode = align_type;
  if (align_mode == HmmAlignProbNonpar || align_mode == HmmAlignProbNonpar2)
    align_mode = HmmAlignProbFullpar;
  par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                 align_mode, align_model);

  if (init_type != HmmInitNonpar)
    par2nonpar_hmm_init_model(init_params,  source_fert, init_type,  initial_prob, start_empty_word);
}

void train_extended_hmm(const Storage1D<Storage1D<uint> >& source, 
                        const LookupTable& slookup,
                        const Storage1D<Storage1D<uint> >& target,
                        const CooccuringWordsType& wcooc,
                        FullHMMAlignmentModel& align_model,
                        Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                        Math1D::Vector<double>& source_fert,
                        InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params,
                        SingleWordDictionary& dict,
                        const floatSingleWordDictionary& prior_weight,
                        HmmOptions& options) {

  std::cerr << "starting Extended HMM EM-training" << std::endl;

  uint nIterations = options.nIterations_;
  HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  bool start_empty_word = options.start_empty_word_;

  double dict_weight_sum = 0.0;
  for (uint i=0; i < options.nTargetWords_; i++) {
    dict_weight_sum += fabs(prior_weight[i].sum());
  }

  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dictionary is assumed to be initialized

  SingleLookupTable aux_lookup;

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  const uint nSourceWords = options.nSourceWords_;

  std::set<uint> seenIs;

  uint maxI = 5;
  uint maxJ = 0;
  for (size_t s=0; s < nSentences; s++) {
    const uint curI = target[s].size();
    const uint curJ = source[s].size();

    seenIs.insert(curI);

    if (curI > maxI)
      maxI = curI;
    if (curJ > maxJ)
      maxJ = curJ;
  }

  std::cerr << "maxJ: " << maxJ << ", maxI: " << maxI << std::endl;

  SingleWordDictionary fwcount(MAKENAME(fwcount));
  fwcount = dict;


  init_hmm_from_ibm1(source, slookup, target, dict, wcooc, align_model, dist_params, dist_grouping_param,
                     source_fert, initial_prob, init_params, options, options.transfer_mode_);
  
  
  Math1D::NamedVector<double> dist_count(MAKENAME(dist_count));
  dist_count = dist_params;

  double dist_grouping_count = dist_grouping_param;
  
  Math1D::Vector<double> source_fert_count = source_fert;
  
  Math1D::NamedVector<double> init_count(MAKENAME(init_count));
  init_count = init_params;
  
  uint zero_offset = maxI-1;
  
  InitialAlignmentProbability ficount(maxI,MAKENAME(ficount));
  ficount = initial_prob;
  
  
  FullHMMAlignmentModel facount(maxI,MAKENAME(facount));
  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      facount[I-1].resize_dirty(I+1,I);
    }
  }

  for (uint iter = 1; iter <= nIterations; iter++) {
    
    std::cerr << "starting EHMM iteration #" << iter << std::endl;

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i=0; i < options.nTargetWords_; i++) {
      fwcount[i].set_constant(0.0);
    }

    for (uint I = 1; I <= maxI; I++) {
      facount[I-1].set_constant(0.0);
      ficount[I-1].set_constant(0.0);
    }

    if (align_type != HmmAlignProbNonpar) {
      dist_count.set_constant(0.0);
      source_fert_count.set_constant(0.0);
      dist_grouping_count = 0.0;
    }

    if (init_type == HmmInitPar) {
      init_count.set_constant(0.0);
    }

    for (size_t s=0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);
      
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
      Math2D::Matrix<double>& cur_facount = facount[curI-1];
      
      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2*curI,curJ,MAKENAME(forward));

      if (start_empty_word) {

        calculate_sehmm_forward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                                initial_prob[curI-1], forward);
      }
      else {
        calculate_hmm_forward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                              initial_prob[curI-1], align_type, forward);
      }

      const uint start_s_idx = cur_source[0];

      long double sentence_prob = 0.0;
      for (uint i=0; i < forward.xDim(); i++) {

	//DEBUG
	if (!(forward(i,curJ-1) >= 0.0)) {
	  
	  std::cerr << "s=" << s << ", I=" << curI << ", i= " << i << ", value " << forward(i,curJ-1) << std::endl;

	  for (uint k=0; k < initial_prob[curI-1].size(); k++) {
	    double p = initial_prob[curI-1][k];
	    if (!(p >= 0))
	      std::cerr << "initial prob[" << k << "]: " << p << std::endl;
	  }

	  for (uint x=0; x < cur_align_model.xDim(); x++) {
	    for (uint y=0; y < cur_align_model.yDim(); y++) {
	      double p = cur_align_model(x,y);
	      if (!(p >= 0))
		std::cerr << "align model(" << x << "," << y << "): " << p << std::endl;
	    }
	  }

	  for (uint j=0; j < curJ; j++) {

	    double p = dict[0][cur_source[j]-1];
	    if (!(p >= 0))
	      std::cerr << "null-prob for source word " << j << ": " << p << std::endl;

	    for (uint i=0; i < curI; i++) {
	      
	      p = dict[cur_target[i]][cur_lookup(j,i)];
	      if (!(p >= 0)) {
		std::cerr << "dict-prob for source word " << j << " and target word " << i 
			  << ": " << p << std::endl;
	      }
	    } 
	  }


	  exit(1);
	}
	//END_DEBUG

        assert(forward(i,curJ-1) >= 0.0);
        sentence_prob += forward(i,curJ-1);
      }

      prev_perplexity -= std::log(sentence_prob);
      
      if (! (sentence_prob > 0.0)) {
        //if (true) {
        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair " << s << " with I=" << curI
                  << ", J= " << curJ << std::endl;

	//DEBUG
	//exit(1);
	//END_DEBUG
      }
      assert(sentence_prob > 0.0);
      
      Math2D::NamedMatrix<long double> backward(2*curI,curJ,MAKENAME(backward));

      if (start_empty_word) {

        calculate_sehmm_backward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                                 initial_prob[curI-1], backward, true);
      }
      else {

        calculate_hmm_backward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                               initial_prob[curI-1], align_type, backward, true);
      }

      long double bwd_sentence_prob = 0.0;
      for (uint i=0; i < backward.xDim(); i++)
        bwd_sentence_prob += backward(i,0);

      long double fwd_bwd_ratio = sentence_prob / bwd_sentence_prob;

      if (fwd_bwd_ratio < 0.999 || fwd_bwd_ratio > 1.001) {
	
        std::cerr << "fwd_bwd_ratio of " << fwd_bwd_ratio << " for sentence pair " << s << " with I=" << curI
                  << ", J= " << curJ << std::endl;
      }

      assert(fwd_bwd_ratio < 1.001);
      assert(fwd_bwd_ratio > 0.999);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      /**** update counts ****/

      //NOTE: it might be faster to compute forward and backward with double precision and rescaling.
      //  There are then two options:
      //   1. keep track of scaling factors (as long double) and convert everything when needed in the count collection
      //   2. work with the rescaled values and normlize appropriately


      //start of sentence
      for (uint i=0; i < curI; i++) {
        uint t_idx = cur_target[i];

        const double coeff = inv_sentence_prob * backward(i,0);
        fwcount[t_idx][cur_lookup(0,i)] += coeff;

	assert(!isnan(coeff));

        ficount[curI-1][i] += coeff;
      }
      if (!start_empty_word) {
        for (uint i=0; i < curI; i++) {
          const double coeff = inv_sentence_prob * backward(i+curI,0);
          fwcount[0][start_s_idx-1] += coeff;
          
          assert(!isnan(coeff));
          
          ficount[curI-1][i+curI] += coeff;
        }
      }
      else
        ficount[curI-1][curI] += inv_sentence_prob * backward(2*curI,0);

      //mid-sentence
      for (uint j=1; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint j_prev = j -1;

        //real positions
        for (uint i=0; i < curI; i++) {
          const uint t_idx = cur_target[i];


          if (dict[t_idx][cur_lookup(j,i)] > 1e-305) {
            fwcount[t_idx][cur_lookup(j,i)] += forward(i,j)*backward(i,j)*inv_sentence_prob / dict[t_idx][cur_lookup(j,i)];

            const long double bw = backward(i,j) * inv_sentence_prob;	  

            uint i_prev;
            long double addon;
	    
            for (i_prev = 0; i_prev < curI; i_prev++) {
              addon = bw * cur_align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+curI,j_prev));
	      assert(!isnan(addon));
              cur_facount(i,i_prev) += addon;
            }

            //start empty word
            if (start_empty_word) {
              addon = bw * initial_prob[curI-1][i] * forward(2*curI,j_prev);
              ficount[curI-1][i] += addon;
            }
          }
        }

        //empty words
        for (uint i=curI; i < 2*curI; i++) {

          const long double bw = backward(i,j) * inv_sentence_prob;
          long double addon = bw * cur_align_model(curI,i-curI) * 
            (forward(i,j_prev) + forward(i-curI,j_prev));

	  assert(!isnan(addon));

          fwcount[0][s_idx-1] += addon;  
          cur_facount(curI,i-curI) += addon;
        }

        //start empty word
        if (start_empty_word) {

          const long double bw = backward(2*curI,j) * inv_sentence_prob;
          
          long double addon = bw * forward(2*curI,j_prev) * initial_prob[curI-1][curI];
          fwcount[0][s_idx-1] += addon;            
          ficount[curI-1][curI] += addon;
        }
      }
    } // loop over sentences finished

    prev_perplexity /= nSentences;
    std::cerr << "perplexity after iteration #" << (iter-1) << ": " << prev_perplexity << std::endl;
    std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    if (align_type != HmmAlignProbNonpar) {

      //compute the expectations of the parameters from the expectations of the models

      for (uint I=1; I <= maxI; I++) {

        if (align_model[I-1].xDim() != 0) {

          for (int i=0; i < (int) I; i++) {

            for (int ii=0; ii < (int) I; ii++) {
              source_fert_count[1] += facount[I-1](ii,i);
	      if (align_type != HmmAlignProbReducedpar || abs(ii-i) <= 5)
		dist_count[zero_offset + ii - i] += facount[I-1](ii,i);
	      else {
                double grouping_norm = std::max(0,i-5);
                grouping_norm += std::max(0,int(I)-1-(i+5));

                assert(grouping_norm > 1e-305);

                dist_grouping_count += facount[I-1](ii,i) / grouping_norm;
              }
            }
            source_fert_count[0] += facount[I-1](I,i);
          }
        }
      }
    }


    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {

      double cur_energy = ehmm_m_step_energy(facount,dist_params,zero_offset,dist_grouping_param);
        
      std::cerr << "cur energy: " << cur_energy << std::endl;

      if (align_type == HmmAlignProbFullpar) {

        dist_count *= 1.0 / dist_count.sum();

        double hyp_energy = ehmm_m_step_energy(facount,dist_count,zero_offset,dist_grouping_param);        

        if (hyp_energy < cur_energy)
          dist_params = dist_count;
      }
      else if (align_type == HmmAlignProbReducedpar) {

        double norm = 0.0;
        for (int k = -5; k <= 5; k++)
          norm += dist_count[zero_offset + k];
        norm += dist_grouping_count;
        
        dist_count *= 1.0 / norm;
        dist_grouping_count *= 1.0 / norm;

        double hyp_energy = ehmm_m_step_energy(facount,dist_count,zero_offset,dist_grouping_count);

        if (hyp_energy < cur_energy) {

          dist_params = dist_count;
          dist_grouping_param = dist_grouping_count;
        }
      }

      //call m-step
      ehmm_m_step(facount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param);
    }

    if (init_type == HmmInitPar) {

      for (uint I=1; I <= maxI; I++) {
	
        if (initial_prob[I-1].size() != 0) {
          for (uint i=0; i < I; i++) {
            source_fert_count[1] += ficount[I-1][i];
            init_count[i] += ficount[I-1][i];
          }
          for (uint i=I; i < ficount[I-1].size(); i++) {
            source_fert_count[0] += ficount[I-1][i];
          }
        }
      }

      double cur_energy = ehmm_init_m_step_energy(ficount,init_params);

      init_count *= 1.0 / init_count.sum();

      double hyp_energy = ehmm_init_m_step_energy(ficount,init_count);

      if (hyp_energy < cur_energy)
        init_params = init_count;

      ehmm_init_m_step(ficount, init_params, options.init_m_step_iter_);
    }

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts

    double nonnullcount = 0.0;
    for (uint k=1; k < fwcount.size(); k++)
      nonnullcount += fwcount[k].sum();

    update_dict_from_counts(fwcount, prior_weight, dict_weight_sum, iter, 
			    options.smoothed_l0_, options.l0_beta_, options.dict_m_step_iter_, dict, 1e-6);

    if (source_fert.size() > 0 && source_fert_count.size() > 0) {

      for (uint i=0; i < 2; i++) {
        source_fert[i] = source_fert_count[i] / source_fert_count.sum();
	assert(!isnan(source_fert[i]));
      }

      if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {
        std::cerr << "new probability for zero alignments: " << source_fert[0] << std::endl; 
      }
    }

    //compute new alignment probabilities from normalized fractional counts

    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {
      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, align_model);
    }

    if (init_type == HmmInitPar) {
      par2nonpar_hmm_init_model(init_params,  source_fert, init_type,  initial_prob, start_empty_word);
    }

    for (uint I=1; I <= maxI; I++) {

      if (align_model[I-1].xDim() != 0) {

        if (init_type == HmmInitNonpar) {
          double inv_norm = 1.0 / ficount[I-1].sum();
          for (uint i=0; i < initial_prob[I-1].size(); i++)
            initial_prob[I-1][i] = std::max(1e-8,inv_norm * ficount[I-1][i]); 
        }
	
        if (align_type == HmmAlignProbNonpar) {

          for (uint i=0; i < I; i++) {
	    
            double sum = 0.0;
            for (uint i_next = 0; i_next <= I; i_next++)
              sum += facount[I-1](i_next,i);
	    
            if (sum >= 1e-300) {
	      
              assert(!isnan(sum));
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));
	      
              for (uint i_next = 0; i_next <= I; i_next++) {
                align_model[I-1](i_next,i) = std::max(1e-8,inv_sum *facount[I-1](i_next,i));
		assert(!isnan(align_model[I-1](i_next,i)));
              }
            }
          }
        }
        else if (align_type == HmmAlignProbNonpar2) {

          for (uint i=0; i < I; i++) {
            
            align_model[I-1](I,i) = source_fert[0];
          
            double sum = 0.0;
            for (uint i_next = 0; i_next < I; i_next++)
              sum += facount[I-1](i_next,i);
          
            if (sum >= 1e-300) {
            
              assert(!isnan(sum));
              const double inv_sum = source_fert[1] / sum;
              assert(!isnan(inv_sum));
              
              for (uint i_next = 0; i_next < I; i_next++) {
                align_model[I-1](i_next,i) = std::max(1e-8,inv_sum * facount[I-1](i_next,i));
                assert(!isnan(align_model[I-1](i_next,i)));
              }
            }
          }

        }
      }
    }


    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      
      double sum_aer = 0.0;
      double sum_marg_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = options.possible_ref_alignments_.begin();
          it != options.possible_ref_alignments_.end(); it ++) {

        uint s = it->first-1;

	if (s >= nSentences)
	  break;

        nContributors++;
        //compute viterbi alignment
        
        Math1D::Vector<AlignBaseType> viterbi_alignment;
        const uint curI = target[s].size();

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s],target[s],wcooc,nSourceWords,slookup[s],aux_lookup);

        if (start_empty_word) 
          compute_sehmm_viterbi_alignment(source[s],cur_lookup, target[s], 
                                          dict, align_model[curI-1], initial_prob[curI-1],
                                          viterbi_alignment);
        else
          compute_ehmm_viterbi_alignment(source[s],cur_lookup, target[s], 
                                         dict, align_model[curI-1], initial_prob[curI-1],
                                         viterbi_alignment,align_type);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        sum_fmeasure += f_measure(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        
        
        Storage1D<AlignBaseType> marg_alignment;
        
        if (!start_empty_word) {
          compute_ehmm_optmarginal_alignment(source[s],cur_lookup, target[s], 
                                             dict, align_model[curI-1], initial_prob[curI-1],
                                             marg_alignment);
          
          sum_marg_aer += AER(marg_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        }
      }

      sum_aer *= 100.0 / nContributors;
      sum_marg_aer *= 100.0 / nContributors;
      nErrors /= nContributors;
      sum_fmeasure /= nContributors;

      if (options.print_energy_) {
        std::cerr << "#### EHMM energy after iteration # " << iter << ": " 
                  <<  extended_hmm_energy(source, slookup, target, align_model, initial_prob, 
                                          dict, wcooc, nSourceWords, prior_weight, align_type, 
                                          start_empty_word, options.smoothed_l0_, options.l0_beta_) 
                  << std::endl;
      }
      std::cerr << "#### EHMM Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      if (!start_empty_word)
        std::cerr << "---- EHMM Marginal-AER : " << sum_marg_aer << " %" << std::endl;
      std::cerr << "#### EHMM Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;

    }
  } //end for (iter)
}



void train_extended_hmm_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source,
                                       const LookupTable& slookup,
                                       const Storage1D<Storage1D<uint> >& target,
                                       const CooccuringWordsType& wcooc,
                                       FullHMMAlignmentModel& align_model,
                                       Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                       Math1D::Vector<double>& source_fert,
                                       InitialAlignmentProbability& initial_prob,
                                       Math1D::Vector<double>& init_params,
                                       SingleWordDictionary& dict,
                                       const floatSingleWordDictionary& prior_weight,
                                       HmmOptions& options) {

  std::cerr << "starting Extended HMM GD-training" << std::endl;

  uint nIterations = options.nIterations_;
  HmmInitProbType init_type = options.init_type_;
  HmmAlignProbType align_type = options.align_type_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;
  bool start_empty_word = false;

  if (align_type == HmmAlignProbNonpar2)  {
    INTERNAL_ERROR << " gradient descent does not (yet?) support Nonpar2. Exiting" << std::endl;
    exit(1);
  }

  if (options.start_empty_word_) {
    std::cerr << "WARNING: gradient descent does not support a start empty word. Ignoring this setting." << std::endl;
    options.start_empty_word_ = false;
  }

  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dicitionary is assumed to be initialized

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  const uint nSourceWords = options.nSourceWords_;

  SingleLookupTable aux_lookup;

  std::set<uint> seenIs;

  uint maxI = 5;
  uint maxJ = 0;
  for (size_t s=0; s < nSentences; s++) {
    const uint curI = target[s].size();
    const uint curJ = source[s].size();

    seenIs.insert(curI);

    if (curI > maxI)
      maxI = curI;
    if (curJ > maxJ)
      maxJ = curJ;
  }

  uint zero_offset = maxI-1;

  init_hmm_from_ibm1(source, slookup, target, dict, wcooc, align_model, dist_params, dist_grouping_param,
                     source_fert, initial_prob, init_params, options, options.transfer_mode_);


  double dist_grouping_grad = dist_grouping_param;
  double new_dist_grouping_param = dist_grouping_param;
  double hyp_dist_grouping_param = dist_grouping_param;

  Math1D::NamedVector<double> dist_grad(MAKENAME(dist_grad));
  dist_grad = dist_params;

  Math1D::NamedVector<double> new_dist_params(MAKENAME(new_dist_params));
  new_dist_params = dist_params;
  Math1D::NamedVector<double> hyp_dist_params(MAKENAME(hyp_dist_params));
  hyp_dist_params = dist_params;

  Math1D::NamedVector<double> init_param_grad(MAKENAME(init_param_grad));
  init_param_grad = init_params;
  Math1D::NamedVector<double> new_init_params(MAKENAME(new_init_params));
  new_init_params = init_params;
  Math1D::NamedVector<double> hyp_init_params(MAKENAME(hyp_init_params));
  hyp_init_params = init_params;

  Math1D::Vector<double> source_fert_grad = source_fert;
  Math1D::Vector<double> new_source_fert = source_fert;  
  Math1D::Vector<double> hyp_source_fert = source_fert;


  InitialAlignmentProbability init_grad(maxI,MAKENAME(init_grad));
  
  Storage1D<Math1D::Vector<double> > dict_grad(options.nTargetWords_);
  for (uint i=0; i < options.nTargetWords_; i++) {
    dict_grad[i].resize(dict[i].size());
  }
  
  FullHMMAlignmentModel align_grad(maxI,MAKENAME(align_grad));
  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      align_grad[I-1].resize_dirty(I+1,I);
    }
    init_grad[I-1].resize_dirty(2*I);
  }

  InitialAlignmentProbability new_init_prob(maxI,MAKENAME(new_init_prob));
  InitialAlignmentProbability hyp_init_prob(maxI,MAKENAME(hyp_init_prob));
  
  SingleWordDictionary new_dict_prob(options.nTargetWords_,MAKENAME(new_dict_prob));
  SingleWordDictionary hyp_dict_prob(options.nTargetWords_,MAKENAME(hyp_dict_prob));

  for (uint i=0; i < options.nTargetWords_; i++) {
    new_dict_prob[i].resize(dict[i].size());
    hyp_dict_prob[i].resize(dict[i].size());
  }
  
  FullHMMAlignmentModel new_align_prob(maxI,MAKENAME(new_align_prob));
  FullHMMAlignmentModel hyp_align_prob(maxI,MAKENAME(hyp_align_prob));

  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      new_align_prob[I-1].resize_dirty(I+1,I);
      hyp_align_prob[I-1].resize_dirty(I+1,I);
    }
    new_init_prob[I-1].resize_dirty(2*I);
    hyp_init_prob[I-1].resize_dirty(2*I);
  }

  Math1D::Vector<double> slack_vector(options.nTargetWords_,0.0);
  Math1D::Vector<double> new_slack_vector(options.nTargetWords_,0.0);

  for (uint i=0; i < options.nTargetWords_; i++) {
    double slack_val = 1.0 - dict[i].sum();
    slack_vector[i] = slack_val;
    new_slack_vector[i] = slack_val;
  }

  double energy = extended_hmm_energy(source, slookup, target, align_model, initial_prob, 
				      dict, wcooc, nSourceWords, prior_weight, align_type, 
                                      start_empty_word, smoothed_l0, l0_beta);


  double line_reduction_factor = 0.5;

  uint nSuccessiveReductions = 0;

  std::cerr << "start energy: " << energy << std::endl;

  double alpha = 50.0;

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting EHMM gd-iter #" << iter << std::endl;
    std::cerr << "alpha: " << alpha << std::endl;

    //set counts to 0
    for (uint i=0; i < options.nTargetWords_; i++) {
      dict_grad[i].set_constant(0.0);
    }

    for (uint I = 1; I <= maxI; I++) {
      align_grad[I-1].set_constant(0.0);
      init_grad[I-1].set_constant(0.0);
    }
    
    if (align_type != HmmAlignProbNonpar) {
      
      dist_grad.set_constant(0.0);
      source_fert_grad.set_constant(0.0);
    }

    dist_grouping_grad = 0.0;

    /******** 1. calculate gradients **********/

    for (size_t s=0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);
      
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      
      const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
      Math2D::Matrix<double>& cur_align_grad = align_grad[curI-1];
      
      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2*curI,curJ,MAKENAME(forward));

      calculate_hmm_forward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                            initial_prob[curI-1], align_type, forward);

      const uint start_s_idx = cur_source[0];

      long double sentence_prob = 0.0;
      for (uint i=0; i < 2*curI; i++) {

        assert(forward(i,curJ-1) >= 0.0);
        sentence_prob += forward(i,curJ-1);
      }
      
      if (! (sentence_prob > 0.0)) {

        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair " << s << " with I=" << curI
                  << ", J= " << curJ << std::endl;

      }
      assert(sentence_prob > 0.0);
      
      Math2D::NamedMatrix<long double> backward(2*curI,curJ,MAKENAME(backward));

      calculate_hmm_backward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
                             initial_prob[curI-1], align_type, backward, true);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      /**** update gradients ****/

      //start of sentence
      for (uint i=0; i < curI; i++) {
        const uint t_idx = cur_target[i];

        const long double coeff = inv_sentence_prob * backward(i,0);

        const double cur_dict_entry = dict[t_idx][cur_lookup(0,i)];

        if (cur_dict_entry > 1e-300) {

          double addon = coeff / cur_dict_entry;
          if (smoothed_l0)
            addon *= prob_pen_prime(cur_dict_entry, l0_beta);
          dict_grad[t_idx][cur_lookup(0,i)] -= addon;
        }

        if (initial_prob[curI-1][i] > 1e-300) {
          init_grad[curI-1][i] -= coeff / std::max(1e-15,initial_prob[curI-1][i]);
	  assert(!isnan(init_grad[curI-1][i]));
	}
      }
      for (uint i=0; i < curI; i++) {

        const long double coeff = inv_sentence_prob * backward(i+curI,0);

        const double cur_dict_entry = dict[0][start_s_idx-1];

        if (cur_dict_entry > 1e-300) {

          double addon = coeff / std::max(1e-15,cur_dict_entry);
          if (smoothed_l0)
            addon *= prob_pen_prime(cur_dict_entry, l0_beta);

          dict_grad[0][start_s_idx-1] -= addon;
        }

        if (initial_prob[curI-1][i+curI] > 1e-300) {
          init_grad[curI-1][i+curI] -= coeff / std::max(1e-15,initial_prob[curI-1][i+curI]);
	  assert(!isnan(init_grad[curI-1][i+curI]));
	}
      }

      //mid-sentence
      for (uint j=1; j < curJ; j++) {
        const uint s_idx = cur_source[j];
        const uint j_prev = j -1;

        //real positions
        for (uint i=0; i < curI; i++) {
          const uint t_idx = cur_target[i];

          const double cur_dict_entry = dict[t_idx][cur_lookup(j,i)];

          if (cur_dict_entry > 1e-70) {

            double dict_addon = forward(i,j)*backward(i,j) 
              / (sentence_prob * std::max(1e-30,cur_dict_entry * cur_dict_entry));

            if (smoothed_l0)
              dict_addon *= prob_pen_prime(cur_dict_entry, l0_beta);

            dict_grad[t_idx][cur_lookup(j,i)] -= dict_addon;

            const long double bw = backward(i,j) / sentence_prob;

            uint i_prev;
            double addon;
	    
            double fert_addon = 0.0;

            for (i_prev = 0; i_prev < curI; i_prev++) {

              addon = bw * (forward(i_prev,j_prev) + forward(i_prev+curI,j_prev));
              fert_addon += addon * cur_align_model(i,i_prev);
              cur_align_grad(i,i_prev) -= addon;
            }

            if (align_type != HmmAlignProbNonpar) {
              source_fert_grad[1] -= fert_addon / source_fert[1];
            }
          }
        }

        //empty words
        for (uint i=curI; i < 2*curI; i++) {

          const long double bw = backward(i,j) * inv_sentence_prob;

          const double cur_dict_entry = dict[0][s_idx-1];

          if (cur_dict_entry > 1e-70) {

            double addon = bw*forward(i,j) / std::max(1e-15,cur_dict_entry * cur_dict_entry);

            if (smoothed_l0)
              addon *= prob_pen_prime(cur_dict_entry, l0_beta);

            dict_grad[0][s_idx-1] -= addon;
          }

          const long double align_addon = bw * (forward(i,j_prev) + forward(i-curI,j_prev));
          if (align_type != HmmAlignProbNonpar) {
            source_fert_grad[0] -= align_addon;
          }
          else {
            cur_align_grad(curI,i-curI) -= align_addon;
          }
        }
      }
     
    } //end for (s)

    if (init_type == HmmInitPar) {      
      
      for (uint I = 1; I <= maxI; I++) {

        if (seenIs.find(I) != seenIs.end()) {

          double sum = 0.0;
          for (uint i=0; i < I; i++)
            sum += init_params[i];

          for (uint i=0; i < I; i++) {

            double cur_grad = init_grad[I-1][i];
            source_fert_grad[1] += cur_grad;

            const double factor = source_fert[1] / (sum * sum);

            for (uint ii=0; ii < I; ii++) {
	      
              if (ii == i)
                init_param_grad[ii] += cur_grad * factor * (sum - init_params[i]);
              else
                init_param_grad[ii] -= cur_grad * factor * init_params[i];
            }
          }

          for (uint i=I; i < 2*I; i++) 
            source_fert_grad[0] += init_grad[I-1][i] / I;
        }
      }
    }

    if (align_type != HmmAlignProbNonpar) {
      for (uint I = 1; I <= maxI; I++) {

        if (seenIs.find(I) != seenIs.end()) {
          for (int i=0; i < (int) I; i++) {	   

            double non_zero_sum = 0.0;
	    
            if (align_type == HmmAlignProbFullpar) {
              double non_zero_sum = 0.0;
	    
              for (uint ii=0; ii < I; ii++) {
                non_zero_sum += dist_params[zero_offset + ii - i];
              }
	      
              double factor = source_fert[1] / (non_zero_sum * non_zero_sum);
	      
              assert(!isnan(factor));
	      
              for (uint ii=0; ii < I; ii++) {
                //NOTE: align_grad has already a negative sign
                dist_grad[zero_offset + ii - i] += align_grad[I-1](ii,i) * factor * (non_zero_sum - dist_params[zero_offset + ii - i]);
                for (uint iii=0; iii < I; iii++) {
                  if (iii != ii)
                    dist_grad[zero_offset + iii - i] -= align_grad[I-1](ii,i) * factor * dist_params[zero_offset + ii - i];
                }
              }
            }
            else {

              double grouping_norm = std::max(0,i-5);
              grouping_norm += std::max(0,int(I)-1-(i+5));
	      
              if (grouping_norm > 0.0)
                non_zero_sum += dist_grouping_param;
	      	      
              for (int ii=0; ii < (int) I; ii++) {
		
                if (abs(ii-i) <= 5)
                  non_zero_sum += dist_params[zero_offset + ii - i];
              }

              double factor = source_fert[1] / (non_zero_sum * non_zero_sum);
	      
              assert(!isnan(factor));

              for (int ii=0; ii < (int) I; ii++) {
                //NOTE: align_grad has already a negative sign
                if (abs(ii-i) <= 5) {
                  dist_grad[zero_offset + ii - i] += align_grad[I-1](ii,i) * factor * (non_zero_sum - dist_params[zero_offset + ii - i]);

                  for (int iii=0; iii < (int) I; iii++) {
                    if (iii != ii) {
                      if (abs(iii-i) <= 5)
                        dist_grad[zero_offset + iii - i] -= align_grad[I-1](ii,i) * factor * dist_params[zero_offset + ii - i];
                      else
                        dist_grouping_grad -= align_grad[I-1](ii,i) * factor * dist_params[zero_offset + ii - i] / grouping_norm;
                    }
                  }
                }
                else {

                  dist_grouping_grad += align_grad[I-1](ii,i) * factor * (non_zero_sum - dist_grouping_param) / grouping_norm;
		  
                  for (int iii=0; iii < (int) I; iii++) {
                    if (iii != ii) {
                      if (abs(iii-i) <= 5)
                        dist_grad[zero_offset + iii - i] -= align_grad[I-1](ii,i) * factor * dist_grouping_param / grouping_norm;
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

    double real_alpha = alpha;

    double sqr_grad_norm = 0.0;
    
    if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) 
      sqr_grad_norm += source_fert_grad.sqr_norm();
    if (init_type == HmmInitPar) 
      sqr_grad_norm += init_param_grad.sqr_norm();
    if (align_type == HmmAlignProbFullpar) {
      sqr_grad_norm += dist_grad.sqr_norm();
    }
    else if (align_type == HmmAlignProbReducedpar) {
      sqr_grad_norm += dist_grad.sqr_norm() + dist_grouping_grad*dist_grouping_grad;
    }
    for (uint i=0; i < options.nTargetWords_; i++) 
      sqr_grad_norm += dict_grad[i].sqr_norm();

    real_alpha /= sqrt(sqr_grad_norm);

    if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {
      
      for (uint i=0; i < 2; i++)
        new_source_fert[i] = source_fert[i] - real_alpha * source_fert_grad[i];

      projection_on_simplex(new_source_fert.direct_access(), 2);
    }

    if (init_type == HmmInitPar) {

      for (uint k=0; k < init_params.size(); k++)
        new_init_params[k] = init_params[k] - real_alpha * init_param_grad[k];

      projection_on_simplex(new_init_params.direct_access(), new_init_params.size());
    }

    if (align_type == HmmAlignProbFullpar) {

      for (uint k=0; k < dist_params.size(); k++)
        new_dist_params[k] = dist_params[k] - real_alpha * dist_grad[k];

      projection_on_simplex(new_dist_params.direct_access(), new_dist_params.size());
    }
    else if (align_type == HmmAlignProbReducedpar) {

      for (uint k=0; k < dist_params.size(); k++)
        new_dist_params[k] = dist_params[k] - real_alpha * dist_grad[k];

      new_dist_grouping_param = dist_grouping_param - real_alpha * dist_grouping_grad;

      assert(new_dist_params.size() >= 11);

      projection_on_simplex_with_slack(new_dist_params.direct_access()+zero_offset-5,new_dist_grouping_param,11);
    }

    for (uint i=0; i < options.nTargetWords_; i++) {

      for (uint k=0; k < dict[i].size(); k++) 
        new_dict_prob[i][k] = dict[i][k] - real_alpha * dict_grad[i][k];
    }

    for (uint I = 1; I <= maxI; I++) {

      if (seenIs.find(I) != seenIs.end()) {

        if (init_type == HmmInitPar) {

          double sum = 0;
          for (uint k=0; k < I; k++)
            sum += new_init_params[k];

	  if (sum > 1e-305) {
	    for (uint k=0; k < I; k++) {
	      new_init_prob[I-1][k] = new_source_fert[1] * new_init_params[k] / sum;
	      assert(!isnan(new_init_prob[I-1][k]));
	    }
	  }
	  else
	    new_init_prob[I-1].set_constant(new_source_fert[1]/I);

          for (uint k=I; k < 2*I; k++) {
            new_init_prob[I-1][k] = new_source_fert[0] / I;
	    assert(!isnan(new_init_prob[I-1][k]));
	  }
        }
        else {
          for (uint k=0; k < initial_prob[I-1].size(); k++) {
	  
            if (init_type == HmmInitFix)
              new_init_prob[I-1][k] = initial_prob[I-1][k];
            else if (init_type == HmmInitNonpar)
              new_init_prob[I-1][k] = initial_prob[I-1][k] - alpha * init_grad[I-1][k];
          }
        }

        if (align_type == HmmAlignProbNonpar) {
          for (uint x=0; x < align_model[I-1].xDim(); x++) 
            for (uint y=0; y < align_model[I-1].yDim(); y++) 
              new_align_prob[I-1](x,y) = align_model[I-1](x,y) - alpha * align_grad[I-1](x,y);
        }
      }
    }

    new_slack_vector = slack_vector;

    /******** 3. reproject on the simplices [Michelot 1986] *********/

    for (uint i=0; i < options.nTargetWords_; i++) {

      for (uint k=0; k < new_dict_prob[i].size(); k++) {
	assert(!isnan(new_dict_prob[i][k]));
	if (new_dict_prob[i][k] < -1e75)
	  new_dict_prob[i][k] = -9e74;
	if (new_dict_prob[i][k] > 1e75)
	  new_dict_prob[i][k] = 9e74;
      }

      projection_on_simplex_with_slack(new_dict_prob[i].direct_access(),new_slack_vector[i],new_dict_prob[i].size());
    }

    for (uint I = 1; I <= maxI; I++) {

      if (init_type == HmmInitNonpar) {
	for (uint k=0; k < new_init_prob[I-1].size(); k++) {
	  if (new_init_prob[I-1][k] <= -1e75)
	    new_init_prob[I-1][k] = -9e74;
	  if (new_init_prob[I-1][k] >= 1e75)
	    new_init_prob[I-1][k] = 9e74;
	
	  if (! (fabs(new_init_prob[I-1][k]) < 1e75) )
	    std::cerr << "prob: " << new_init_prob[I-1][k] << std::endl;
	  
	  assert(fabs(new_init_prob[I-1][k]) < 1e75);
	}
      }

      projection_on_simplex(new_init_prob[I-1].direct_access(),new_init_prob[I-1].size());

      if (align_type == HmmAlignProbNonpar) {

	for (uint k=0; k < new_align_prob[I-1].size(); k++) {
	  if (new_align_prob[I-1].direct_access(k) <= -1e75)
	    new_align_prob[I-1].direct_access(k) = -9e74;
	  if (new_align_prob[I-1].direct_access(k) >= 1e75)
	    new_align_prob[I-1].direct_access(k) = 9e74;

	  assert(fabs(new_align_prob[I-1].direct_access(k)) < 1e75);
	}

        for (uint y=0; y < align_model[I-1].yDim(); y++) {
	 	  
          projection_on_simplex(new_align_prob[I-1].direct_access() + y*align_model[I-1].xDim(),
                                align_model[I-1].xDim());
        }
      }
    }

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
      
      for (uint i=0; i < options.nTargetWords_; i++) {
	
        for (uint k=0; k < dict[i].size(); k++) 
          hyp_dict_prob[i][k] = lambda * new_dict_prob[i][k] + neg_lambda * dict[i][k];
      }

      if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {

        for (uint i=0; i < 2; i++)
          hyp_source_fert[i] = lambda * new_source_fert[i] + neg_lambda * source_fert[i];
      }

      if (align_type == HmmAlignProbReducedpar)
        hyp_dist_grouping_param = lambda * new_dist_grouping_param + neg_lambda * dist_grouping_param;

      if (init_type == HmmInitPar) {
        for (uint k=0; k < hyp_init_params.size(); k++)
          hyp_init_params[k] = lambda * new_init_params[k] + neg_lambda * init_params[k];

        par2nonpar_hmm_init_model(hyp_init_params,  hyp_source_fert, init_type,  hyp_init_prob);
      }

      for (uint I = 1; I <= maxI; I++) {

        if (init_type != HmmInitPar) {

          for (uint k=0; k < initial_prob[I-1].size(); k++) {
	  
            hyp_init_prob[I-1][k] = lambda * new_init_prob[I-1][k] + neg_lambda * initial_prob[I-1][k];
          }
        }
      }

      if (align_type != HmmAlignProbNonpar) {

        for (uint k=0; k < dist_params.size(); k++)
          hyp_dist_params[k] = lambda * new_dist_params[k] + neg_lambda * dist_params[k];
	
        par2nonpar_hmm_alignment_model(hyp_dist_params, zero_offset, hyp_dist_grouping_param, hyp_source_fert,
                                       align_type, hyp_align_prob);
      }
      else {

        for (uint I = 1; I <= maxI; I++) {
	  
          for (uint x=0; x < align_model[I-1].xDim(); x++) 
            for (uint y=0; y < align_model[I-1].yDim(); y++) 
              hyp_align_prob[I-1](x,y) = lambda * new_align_prob[I-1](x,y) + neg_lambda * align_model[I-1](x,y);
        }
      }

      double new_energy = extended_hmm_energy(source, slookup, target, hyp_align_prob, 
                                              hyp_init_prob, hyp_dict_prob, wcooc, nSourceWords, prior_weight, 
					      align_type, start_empty_word, smoothed_l0, l0_beta);   

      std::cerr << "new: " << new_energy << ", prev: " << hyp_energy << std::endl;

      if (new_energy < hyp_energy) {

        hyp_energy = new_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;
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

    for (uint i=0; i < options.nTargetWords_; i++) {
      
      slack_vector[i] = best_lambda * new_slack_vector[i] + neg_best_lambda * slack_vector[i];

      for (uint k=0; k < dict[i].size(); k++) 
        dict[i][k] = best_lambda * new_dict_prob[i][k] + neg_best_lambda * dict[i][k];
    }

    if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {
      
      for (uint i=0; i < 2; i++)
        source_fert[i] = best_lambda * new_source_fert[i] + neg_best_lambda * source_fert[i];
    }

    if (init_type == HmmInitPar) {
      
      for (uint k=0; k < init_params.size(); k++)
        init_params[k] = best_lambda * new_init_params[k] + neg_best_lambda * init_params[k];

      par2nonpar_hmm_init_model(init_params, source_fert, init_type,  initial_prob);

    }
    else {
      for (uint I = 1; I <= maxI; I++) {
	
        for (uint k=0; k < initial_prob[I-1].size(); k++) {
	
          initial_prob[I-1][k] = best_lambda * new_init_prob[I-1][k] + neg_best_lambda * initial_prob[I-1][k];
        }
      }
    }

    if (align_type != HmmAlignProbNonpar) {
      
      for (uint k=0; k < dist_params.size(); k++)
        dist_params[k] = best_lambda * new_dist_params[k] + neg_best_lambda * dist_params[k];

      if (align_type == HmmAlignProbReducedpar)
        dist_grouping_param = best_lambda * new_dist_grouping_param + neg_best_lambda * dist_grouping_param;

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, align_model);
    }
    else {
      for (uint I = 1; I <= maxI; I++) {
      
        for (uint x=0; x < align_model[I-1].xDim(); x++) 
          for (uint y=0; y < align_model[I-1].yDim(); y++) 
            align_model[I-1](x,y) = best_lambda * new_align_prob[I-1](x,y) + neg_best_lambda * align_model[I-1](x,y);
      }
    }

    if (options.print_energy_) {
      std::cerr << "slack-sum: " << slack_vector.sum() << std::endl;
      std::cerr << "energy: " << energy << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double sum_marg_aer = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;


      for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = options.possible_ref_alignments_.begin();
          it != options.possible_ref_alignments_.end(); it ++) {

        uint s = it->first-1;

	if (s >= nSentences)
	  break;

        nContributors++;
        //compute viterbi alignment

        Storage1D<AlignBaseType> viterbi_alignment;
        const uint curI = target[s].size();

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s],target[s],wcooc,nSourceWords,slookup[s],aux_lookup);
	
        compute_ehmm_viterbi_alignment(source[s],cur_lookup, target[s], 
                                       dict, align_model[curI-1], initial_prob[curI-1],
                                       viterbi_alignment,align_type);
        
        //add alignment error rate
        sum_aer += AER(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        sum_fmeasure += f_measure(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        
        Storage1D<AlignBaseType> marg_alignment;
	  
        compute_ehmm_optmarginal_alignment(source[s],cur_lookup, target[s], 
                                           dict, align_model[curI-1], initial_prob[curI-1],
                                           marg_alignment);
        
        sum_marg_aer += AER(marg_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_marg_aer *= 100.0 / nContributors;
      nErrors /= nContributors;
      sum_fmeasure /= nContributors;
      
      std::cerr << "#### EHMM Viterbi-AER after gd-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "---- EHMM Marginal-AER : " << sum_marg_aer << " %" << std::endl;
      std::cerr << "#### EHMM Viterbi-fmeasure after gd-iteration #" << iter << ": " << sum_fmeasure << std::endl;      
      std::cerr << "#### EHMM Viterbi-DAE/S after gd-iteration #" << iter << ": " << nErrors << std::endl;      
    }
  } // end  for (iter)
}


void viterbi_train_extended_hmm(const Storage1D<Storage1D<uint> >& source,
                                const LookupTable& slookup,
                                const Storage1D<Storage1D<uint> >& target,
                                const CooccuringWordsType& wcooc,
                                FullHMMAlignmentModel& align_model,
                                Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                Math1D::Vector<double>& source_fert,
                                InitialAlignmentProbability& initial_prob,
                                Math1D::Vector<double>& init_params,
                                SingleWordDictionary& dict, 
                                const floatSingleWordDictionary& prior_weight,
                                bool deficient_parametric, HmmOptions& options,
				const Math1D::Vector<double>& log_table) {

  std::cerr << "starting Viterbi Training for Extended HMM" << std::endl;

  uint nIterations = options.nIterations_;
  HmmInitProbType init_type = options.init_type_; 
  HmmAlignProbType align_type = options.align_type_;
  bool start_empty_word = options.start_empty_word_;

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  const uint nSourceWords = options.nSourceWords_;

  SingleLookupTable aux_lookup;

  for (size_t s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    viterbi_alignment[s].resize(cur_source.size());
  }
  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dictionary is assumed to be initialized

  std::set<uint> seenIs;

  uint maxI = 0;
  uint maxJ = 0;
  for (size_t s=0; s < nSentences; s++) {
    const uint curI = target[s].size();
    const uint curJ = source[s].size();

    seenIs.insert(curI);

    if (curI > maxI)
      maxI = curI;
    if (curJ > maxJ)
      maxJ = curJ;
  }
  
  init_hmm_from_ibm1(source, slookup, target, dict, wcooc, align_model, dist_params, dist_grouping_param,
                     source_fert, initial_prob, init_params, options, options.transfer_mode_);

  uint zero_offset = maxI-1;

  Math1D::Vector<double> source_fert_count(2,0.0);

  NamedStorage1D<Math1D::Vector<double> > dcount(options.nTargetWords_,MAKENAME(dcount));
  for (uint i=0; i < options.nTargetWords_; i++) {
    dcount[i].resize(dict[i].size());
  }

  InitialAlignmentProbability icount(maxI,MAKENAME(icount));
  icount = initial_prob;

  Math1D::NamedVector<double> init_count(MAKENAME(init_count) );
  init_count = init_params;

  FullHMMAlignmentModel acount(maxI,MAKENAME(acount));
  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      acount[I-1].resize_dirty(I+1,I);
    }
  }

  Math1D::NamedVector<double> dist_count(MAKENAME(dist_count));
  dist_count = dist_params;

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting Viterbi-EHMM iteration #" << iter << std::endl;

    //DEBUG
    // for (uint I=0; I < 10; I++)
    //   std::cerr << "init prob: " << initial_prob[I] << std::endl;
    //END_DEBUG

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i=0; i < options.nTargetWords_; i++) {
      dcount[i].set_constant(0);
    }

    for (uint I = 1; I <= maxI; I++) {
      acount[I-1].set_constant(0.0);
      icount[I-1].set_constant(0.0);
    }

    source_fert_count.set_constant(0.0);
    init_count.set_constant(0.0);
    dist_count.set_constant(0.0);      

    for (size_t s=0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);
            

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      
      const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
      Math2D::Matrix<double>& cur_facount = acount[curI-1];


      long double prob;

      if (start_empty_word) {
        prob = compute_sehmm_viterbi_alignment(cur_source,cur_lookup, cur_target, 
                                               dict, cur_align_model, initial_prob[curI-1],
                                               viterbi_alignment[s], true, false, 0.0);
      }
      else {
        prob = compute_ehmm_viterbi_alignment(cur_source,cur_lookup, cur_target, 
                                              dict, cur_align_model, initial_prob[curI-1],
                                              viterbi_alignment[s], align_type, true, false, 0.0);
      }

      prev_perplexity -= std::log(prob);
      
      if (! (prob > 0.0)) {

        std::cerr << "sentence_prob " << prob << " for sentence pair " << s << " with I=" << curI
                  << ", J= " << curJ << std::endl;
      }
      assert(prob > 0.0);

      /**** update counts ****/      
      for (uint j=0; j < curJ; j++) {

        ushort aj = viterbi_alignment[s][j];
        if (aj >= curI) {
          dcount[0][cur_source[j]-1] += 1;
          if (align_type != HmmAlignProbNonpar && j > 0) 
            source_fert_count[0] += 1.0;
        }
        else {
          dcount[cur_target[aj]][cur_lookup(j,aj)] += 1;
          if (align_type != HmmAlignProbNonpar && j > 0) 
            source_fert_count[1] += 1.0;
        }

        if (j == 0) {
          if (!start_empty_word)
            icount[curI-1][aj] += 1.0;
          else {
            if (aj < curI)
              icount[curI-1][aj] += 1.0;
            else {
              assert( aj == 2*curI);
              icount[curI-1][curI] += 1.0;
            }
          }
        }
        else {

          ushort prev_aj = viterbi_alignment[s][j-1];
	  
          if (prev_aj == 2*curI) {
            assert(start_empty_word);
            if (aj == prev_aj)
              icount[curI-1][curI] += 1.0;
            else
              icount[curI-1][aj] += 1.0;
          }
          else if (prev_aj >= curI) {

            if (aj >= curI) {
              cur_facount(curI,prev_aj-curI) += 1.0;
            }
            else {
              cur_facount(aj,prev_aj-curI) += 1.0;
	    
              if (deficient_parametric) {

                uint diff = zero_offset - (prev_aj - curI);
                diff += (aj >= curI) ?  aj - curI : aj;

                dist_count[diff] += 1.0;
              }
            }
          }
          else {
            if (aj >= curI) {
              cur_facount(curI,prev_aj) += 1.0;
            }
            else {
              cur_facount(aj,prev_aj) += 1.0;
	    

              if (deficient_parametric) {
		
                uint diff = zero_offset - prev_aj;
                diff += (aj >= curI) ?  aj - curI : aj;
		
                dist_count[diff] += 1.0;
              }
            }
          }
        }
      }
    } // loop over sentences finished

    prev_perplexity /= nSentences;

    //include the dict_regularity term in the output energy
    if (options.print_energy_) {
      double energy = prev_perplexity;
      for (uint i=0; i < dcount.size(); i++)
        for (uint k=0; k < dcount[i].size(); k++)
          if (dcount[i][k] > 0)
            //we need to divide as we are truly minimizing the perplexity WITHOUT division plus the l0-term
            energy += prior_weight[i][k] / nSentences; 
      
      //std::cerr << "perplexity after iteration #" << (iter-1) <<": " << prev_perplexity << std::endl;
      std::cerr << "energy after iteration #" << (iter-1) <<": " << energy << std::endl;
    }

    //std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    if (init_type == HmmInitPar) {

      for (uint I=1; I <= maxI; I++) {
	
        if (initial_prob[I-1].size() != 0) {
          for (uint i=0; i < I; i++) {
            source_fert_count[1] += icount[I-1][i];
            init_count[i] += icount[I-1][i];
          }
          for (uint i=I; i < icount[I-1].size(); i++) {
            source_fert_count[0] += icount[I-1][i];
          }
        }
      }

      double cur_energy = ehmm_init_m_step_energy(icount,init_params);

      init_count *= 1.0 / init_count.sum();

      double hyp_energy = ehmm_init_m_step_energy(icount,init_count);

      if (hyp_energy < cur_energy)
        init_params = init_count;


      ehmm_init_m_step(icount, init_params, options.init_m_step_iter_);
      //par2nonpar can only be called after source_fert has been updated
    }
    else if (init_type == HmmInitNonpar) {

      for (uint I=1; I <= maxI; I++) {
        if (icount[I-1].size() > 0) {
          double sum = icount[I-1].sum();
          if (sum > 1e-305) {
            initial_prob[I-1] = icount[I-1];
            initial_prob[I-1] *= 1.0 / sum;
          }
        }
      }
    }

    if (source_fert_count.sum() > 1e-305) {
      for (uint k=0; k < 2; k++)
        source_fert[k] = source_fert_count[k] / source_fert_count.sum();
    }

    if (!deficient_parametric && align_type != HmmAlignProbNonpar) {

      double dist_grouping_count = 0.0;


      for (uint I=1; I <= maxI; I++) {
        
        if (align_model[I-1].xDim() != 0) {
          
          for (int i=0; i < (int) I; i++) {
            
            for (int ii=0; ii < (int) I; ii++) {
              dist_count[zero_offset + ii - i] += acount[I-1](ii,i);
		
              if (align_type == HmmAlignProbReducedpar && abs(ii-i) > 5) {
                double grouping_norm = std::max(0,i-5);
                grouping_norm += std::max(0,int(I)-1-(i+5));
		
                assert(grouping_norm > 0.0);
                  
                dist_grouping_count += acount[I-1](ii,i) / grouping_norm;
              }
            }
          }
        }
      }
      
      double cur_energy = ehmm_m_step_energy(acount,dist_params,zero_offset,dist_grouping_param);

      std::cerr << "cur_energy: " << cur_energy << std::endl;

      if (align_type == HmmAlignProbFullpar) {

        dist_count *= 1.0 / dist_count.sum();

        double hyp_energy = ehmm_m_step_energy(acount,dist_count,zero_offset,dist_grouping_param);        

        std::cerr << "hyp_energy: " << hyp_energy << std::endl;

        if (hyp_energy < cur_energy)
          dist_params = dist_count;
      }
      else if (align_type == HmmAlignProbReducedpar) {

        double norm = 0.0;
        for (int k = -5; k <= 5; k++)
          norm += dist_count[zero_offset + k];
        norm += dist_grouping_count;
        
        dist_count *= 1.0 / norm;
        dist_grouping_count *= 1.0 / norm;

        double hyp_energy = ehmm_m_step_energy(acount,dist_count,zero_offset,dist_grouping_count);

        std::cerr << "hyp_energy: " << hyp_energy << std::endl;

        if (hyp_energy < cur_energy) {

          dist_params = dist_count;
          dist_grouping_param = dist_grouping_count;
        }
      }

      //call m-step
      ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param);

      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, align_model);
    }


    if (init_type == HmmInitPar)
      par2nonpar_hmm_init_model(init_params, source_fert, init_type, initial_prob, start_empty_word);

    std::cerr << "source_fert_count before ICM: " << source_fert_count << std::endl;

#if 1
    std::cerr << "starting ICM stage" << std::endl;

    /**** ICM stage ****/

    uint nSwitches = 0;

    int dist_count_sum = dist_count.sum();

    Math1D::Vector<uint> dict_sum(dcount.size());
    for (uint k=0; k < dcount.size(); k++)
      dict_sum[k] = dcount[k].sum();

    for (size_t s=0; s < nSentences; s++) {

      const uint curJ = source[s].size();
      const uint curI = target[s].size();

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
      
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);
    
      Math2D::Matrix<double>& cur_acount = acount[curI-1];

      for (uint j=0; j < curJ; j++) {

        ushort cur_aj = cur_alignment[j];
        ushort new_aj = cur_aj;

        double best_change = 1e300;

        uint cur_target_word = (cur_aj >= curI) ? 0 : cur_target[cur_aj];
        uint cur_dict_num = cur_target_word;
        uint cur_idx = (cur_aj >= curI) ? cur_source[j]-1 : cur_lookup(j,cur_aj);
	
        Math1D::Vector<double>& cur_dictcount = dcount[cur_dict_num]; 
        uint cur_dictsum = dict_sum[cur_dict_num]; 

        if (cur_aj == 2*curI)
          continue;
        if (j > 0 && cur_alignment[j-1] == 2*curI)
          continue;
        
        for (uint i=0; i < 2*curI; i++) {

          if (i == cur_aj)
            continue;
	  
          bool allowed = true;
          bool careful = false;

          if (start_empty_word && i >= curI)
            allowed = false;

          if (j > 0 && i >= curI) {

            ushort prev_aj = cur_alignment[j-1];
	    
            if (i != prev_aj && i != prev_aj+curI)
              continue;
          }

          if (j > 0 && j+1 < curJ) {
            int effective_prev_aj = cur_alignment[j-1];
            if (effective_prev_aj >= (int) curI)
              effective_prev_aj -= curI;

            int effective_cur_aj = cur_aj;
            if (effective_cur_aj >= (int) curI)
              effective_cur_aj -= curI;
	    
            int effective_i = 0;
            if (effective_i >= (int) curI)
              effective_i -= curI;

            int next_aj = cur_alignment[j+1];
            if (next_aj >= (int) curI)
              allowed = false;

	    
            //TODO: handle these special cases (some distributions are doubly affected)
            if (align_type == HmmAlignProbNonpar) {
              if (effective_prev_aj == effective_cur_aj)
                allowed = false;
              if (effective_prev_aj == effective_i)
                allowed = false;
            }
            else {
              if ( effective_cur_aj - effective_prev_aj == next_aj - effective_cur_aj)
                careful = true;
              if ( effective_cur_aj - effective_prev_aj == next_aj - effective_i)
                careful = true;
              if ( effective_cur_aj - effective_i == next_aj - effective_cur_aj)
                careful = true;
              if ( effective_cur_aj - effective_i == next_aj - effective_i)
                careful = true;
            }
          }
          if (j+1 < curJ && cur_alignment[j+1] >= curI)
            allowed = false; //in this case many more distributions /successive positions could be affected

          if (allowed) {	    

            uint new_target_word = (i >= curI) ? 0 : cur_target[i];

            uint effective_i = i;
            if (effective_i >= curI)
              effective_i -= curI;

            uint effective_cur_aj = cur_aj;
            if (effective_cur_aj >= curI)
              effective_cur_aj -= curI;

            uint hyp_dict_num = (i >= curI) ? 0 : cur_target[i];
	    
            uint hyp_idx = (i >= curI) ? cur_source[j]-1 : cur_lookup(j,i);
	    
            double change = 0.0;

            //a) regarding preceeding pos
            if (j == 0) {

              if (init_type == HmmInitNonpar) {
                assert(icount[curI-1][cur_aj] > 0);
		
                change -= -icount[curI-1][cur_aj] * log_table[icount[curI-1][cur_aj]];
                if (icount[curI-1][i] > 0)
                  change -= -icount[curI-1][i] * log_table[icount[curI-1][i]];
		
                if (icount[curI-1][cur_aj] > 1)
                  change += -(icount[curI-1][cur_aj]-1) * log_table[icount[curI-1][cur_aj]-1];
                change += -(icount[curI-1][i]+1) * log_table[icount[curI-1][i]+1];

              }
              else if (init_type != HmmInitFix) {
                
                change += std::log(initial_prob[curI-1][std::min<uint>(cur_aj,curI)]);
                change -= std::log(initial_prob[curI-1][std::min<uint>(i,curI)]);
              }
            }
            else {
              // j > 0

              //note: the total sum of counts for prev_aj stays constant in this operation
              uint prev_aj = cur_alignment[j-1];
              if (prev_aj >= curI)
                prev_aj -= curI;

              if (align_type == HmmAlignProbNonpar) {
                int cur_c = cur_acount(std::min<ushort>(curI,cur_aj),prev_aj);
                int new_c = cur_acount(std::min<ushort>(curI,i),prev_aj);
                assert(cur_c > 0);
	      
                change -= -cur_c * std::log(cur_c);
                if (new_c > 0)
                  change -= -new_c*std::log(new_c);
	      
                if (cur_c > 1)
                  change += -(cur_c-1) * std::log(cur_c-1);
                change += -(new_c+1) * std::log(new_c+1);
              }
              else if (align_type == HmmAlignProbNonpar2) {
                
                //incremental calculation is quite complicated in this case:
                // if you switch between zero and non-zero alignments you also need to renormalize
                // => we do not update the alignment parameters in ICM

                change -= -std::log(cur_align_model(std::min<ushort>(curI,cur_aj),prev_aj));
                change += -std::log(cur_align_model(std::min<ushort>(curI,i),prev_aj));
              }
              else { //parametric model
		
                if (deficient_parametric) {
                  if (cur_aj < curI) {
		  
                    int cur_c = dist_count[zero_offset + cur_aj - prev_aj];

                    if (careful) {
                      change -= -std::log( double(cur_c) / double(dist_count_sum) );
                    }
                    else {
                      change -= -cur_c * std::log(cur_c);
                      if (cur_c > 1)
                        change += -(cur_c-1) * std::log(cur_c-1);
                    }
                  }
                }
                else {
                  change -= -std::log(cur_align_model(std::min<ushort>(curI,cur_aj),prev_aj));
                }


                if (deficient_parametric) {

                  if (i < curI) {
		  
                    int cur_c = dist_count[zero_offset + i - prev_aj];
		    
                    if (careful) {
                      if (cur_c == 0)
                        change += 1e20;
                      else
                        change += -std::log( double(cur_c) / double(dist_count_sum) );
                    }
                    else {
                      if (cur_c > 0)
                        change -= -cur_c * std::log(cur_c);
                      change += -(cur_c +1)* std::log(cur_c+1);
                    }
                  }
                }
                else {
                  change += -std::log(cur_align_model(std::min(curI,i),prev_aj));
                }

                //source fertility counts
                if (deficient_parametric && !careful) {
                  if (cur_aj < curI && i >= curI) {

                    int cur_c0 = source_fert_count[0];
                    int cur_c1 = source_fert_count[1];
		    
                    if (cur_c0 > 0)
                      change -= -(cur_c0) * std::log(cur_c0);
                    change -= -(cur_c1) * std::log(cur_c1);
		    
                    change -= dist_count_sum * std::log(dist_count_sum);

                    if (cur_c1 > 1)
                      change += -(cur_c1-1) * std::log(cur_c1-1);
                    change += -(cur_c0+1) * std::log(cur_c0+1);
		    
                    change += (dist_count_sum-1) * std::log(dist_count_sum-1);
                  }
                  else if (cur_aj >= curI && i < curI) {
		    
                    int cur_c0 = source_fert_count[0];
                    int cur_c1 = source_fert_count[1];
		    
                    change -= -(cur_c0) * std::log(cur_c0);
                    if (cur_c1 > 0)
                      change -= -(cur_c1) * std::log(cur_c1);
                    change -= dist_count_sum * std::log(dist_count_sum);
		    
                    if (cur_c0 > 1)
                      change += -(cur_c0-1) * std::log(cur_c0-1);
                    change += -(cur_c1+1) * std::log(cur_c1+1);
                    change += (dist_count_sum+1) * std::log(dist_count_sum+1);
                  }
                }
              }
            }

            assert(!isnan(change));

            uint effective_new_aj = i;
            if (effective_new_aj >= curI)
              effective_new_aj -= curI;

            //b) regarding succeeding pos
            if (j+1 < curJ && effective_cur_aj != effective_new_aj) {

              uint next_aj = std::min<ushort>(curI,cur_alignment[j+1]);
              assert(next_aj < curI);
              
              if (align_type == HmmAlignProbNonpar) {
                int total_cur_count = 0;
                for (uint k=0; k <= curI; k++)
                  total_cur_count += cur_acount(k,effective_cur_aj);

                assert(total_cur_count > 0);
                assert(cur_acount(next_aj,effective_cur_aj) > 0);
		
                int total_new_count = 0;
                for (uint k=0; k <= curI; k++)
                  total_new_count += cur_acount(k,effective_new_aj);
		
                change -= total_cur_count * std::log(total_cur_count);
                change -= -(cur_acount(next_aj,effective_cur_aj)) * std::log(cur_acount(next_aj,effective_cur_aj));
		
                if (total_new_count > 0) 
                  change -= total_new_count * std::log(total_new_count);
                if (cur_acount(next_aj,effective_new_aj) > 0)
                  change -= -(cur_acount(next_aj,effective_new_aj)) * std::log(cur_acount(next_aj,effective_new_aj));

                assert(!isnan(change));
		
                if (total_cur_count > 1) 
                  change += (total_cur_count-1) * std::log(total_cur_count-1);
                if (cur_acount(next_aj,effective_cur_aj) > 1)
                  change += -(cur_acount(next_aj,effective_cur_aj)-1) * std::log(cur_acount(next_aj,effective_cur_aj)-1);
	      
                change += (total_new_count+1) * std::log(total_new_count+1);
                change += -(cur_acount(next_aj,effective_new_aj)+1) * std::log(cur_acount(next_aj,effective_new_aj)+1);
              }
              else {
                //parametric model

                if (deficient_parametric) {
                  if (next_aj < curI) {
		    
                    int cur_c = dist_count[zero_offset + next_aj - effective_cur_aj];
                    int new_c = dist_count[zero_offset + next_aj - effective_new_aj];
		    
                    if (careful) {
                      change -= -std::log( double(cur_c) / double(dist_count_sum) );
                      change += -std::log( double(new_c) / double(dist_count_sum) );
                    }
                    else {
                      change -= -(cur_c)* std::log(cur_c);
                      if (new_c > 0)
                        change -= -(new_c)* std::log(new_c);
		      
                      if (cur_c > 1)
                        change += -(cur_c - 1)* std::log(cur_c-1);
                      change += -(new_c+1)* std::log(new_c+1);
                    }
                  }
                  else {
                    std::cerr << "WARNING: this case should not occur" << std::endl;
                  }
                }
                else {

                  change -= -std::log(cur_align_model(std::min(next_aj,curI),effective_cur_aj));
                  change += -std::log(cur_align_model(std::min(next_aj,curI),effective_new_aj));
                }
              }
            }

            assert(!isnan(change));

            uint prev_word = cur_target_word; 
            uint new_word =  (new_aj >= curI) ? 0 : cur_target[new_aj];

            if (prev_word != new_word) {

              if (dict_sum[new_target_word] > 0)
                change -= double(dict_sum[new_target_word]) * log_table[dict_sum[new_target_word]];
              change += double(dict_sum[new_target_word]+1.0) * log_table[dict_sum[new_target_word]+1];

              if (dcount[new_target_word][hyp_idx] > 0)
                change -= double(dcount[new_target_word][hyp_idx]) * 
                  (-log_table[dcount[new_target_word][hyp_idx]]);
              else
                change += prior_weight[hyp_dict_num][hyp_idx]; 

              change += double(dcount[new_target_word][hyp_idx]+1) * 
                (-log_table[dcount[new_target_word][hyp_idx]+1]);

              change -= double(cur_dictsum) * log_table[cur_dictsum];
              if (cur_dictsum > 1)
                change += double(cur_dictsum-1) * log_table[cur_dictsum-1];

              change -= - double(cur_dictcount[cur_idx]) * log_table[cur_dictcount[cur_idx]];

              if (cur_dictcount[cur_idx] > 1) {
                change += double(cur_dictcount[cur_idx]-1) * (-log_table[cur_dictcount[cur_idx]-1]);
              }
              else
                change -= prior_weight[cur_dict_num][cur_idx];
            }

            assert(!isnan(change));

            if (change < best_change) {

              best_change = change;
              new_aj = i;
            }
          }
          else if (align_type != HmmAlignProbNonpar && !start_empty_word
                   && init_type != HmmInitNonpar) {

            //TODO: make this ready for start-empty-word-mode

            uint new_target_word = (i >= curI) ? 0 : cur_target[i];

            uint effective_i = i;
            if (effective_i >= curI)
              effective_i -= curI;

            uint effective_cur_aj = cur_aj;
            if (effective_cur_aj >= curI)
              effective_cur_aj -= curI;

            uint hyp_dict_num = (i >= curI) ? 0 : cur_target[i];
	    
            uint hyp_idx = (i >= curI) ? source[s][j]-1 : cur_lookup(j,i);
	    
            double change = 0.0;

            uint prev_word = cur_target_word; 
            uint new_word =  (new_aj >= curI) ? 0 : cur_target[new_aj];

            if (prev_word != new_word) {

              if (dict_sum[new_target_word] > 0)
                change -= double(dict_sum[new_target_word]) * log_table[ dict_sum[new_target_word] ];
              change += double(dict_sum[new_target_word]+1.0) * log_table[ dict_sum[new_target_word]+1 ];

              if (dcount[new_target_word][hyp_idx] > 0)
                change -= double(dcount[new_target_word][hyp_idx]) * 
                  (-log_table[dcount[new_target_word][hyp_idx]]);
              else
                change += prior_weight[hyp_dict_num][hyp_idx]; 

              change += double(dcount[new_target_word][hyp_idx]+1) * 
                (-log_table[dcount[new_target_word][hyp_idx]+1]);

              change -= double(cur_dictsum) * log_table[cur_dictsum];
              if (cur_dictsum > 1)
                change += double(cur_dictsum-1) * log_table[cur_dictsum-1];
	      
              change -= - double(cur_dictcount[cur_idx]) * log_table[cur_dictcount[cur_idx]];

              if (cur_dictcount[cur_idx] > 1) {
                change += double(cur_dictcount[cur_idx]-1) * (-log_table[cur_dictcount[cur_idx]-1]);
              }
              else
                change -= prior_weight[cur_dict_num][cur_idx];
            }

            change -= -std::log(hmm_alignment_prob(cur_source,cur_lookup,cur_target, 
                                                   dict, //will not be used
                                                   align_model, initial_prob, cur_alignment, false));
            
            Math1D::Vector<AlignBaseType> hyp_alignment = cur_alignment;
            hyp_alignment[j] = i;
            if (j > 0 && i >= curI)
              assert(i == hyp_alignment[j-1] || i-curI == hyp_alignment[j-1]);

            if (j+1 < curJ) {

              uint next_aj = hyp_alignment[j+1];
              if (next_aj >= curI) {

                uint new_next_aj = (i < curI) ? i+curI : i;
                
                if (new_next_aj != next_aj) {
                  for (uint jj=j+1; jj < curJ; jj++) {
                    if (hyp_alignment[jj] == next_aj)
                      hyp_alignment[jj] = new_next_aj;
                    else
                      break;
                  }
                }
              }
            }

            change += -std::log(hmm_alignment_prob(cur_source,cur_lookup,cur_target, 
                                                   dict, //will not be used
                                                   align_model, initial_prob, hyp_alignment, false));
            
            if (change < best_change) {

              best_change = change;
              new_aj = i;
            }
          }
        }
	      
        if (best_change < -0.01 && new_aj != cur_aj) {

          nSwitches++;
	  
          uint hyp_idx = (new_aj >= curI) ? cur_source[j]-1 : cur_lookup(j,new_aj);

          cur_alignment[j] = new_aj;
          
          if (j > 0 && new_aj >= curI)
            assert(new_aj == cur_alignment[j-1] || new_aj - curI == cur_alignment[j-1]);

          uint effective_cur_aj = cur_aj;
          if (effective_cur_aj >= curI)
            effective_cur_aj -= curI;

          uint effective_new_aj = new_aj;
          if (effective_new_aj >= curI)
            effective_new_aj -= curI;
          
          bool future_handled = false;

          if (j+1 < curJ) {

            uint next_aj = cur_alignment[j+1];
            if (next_aj >= curI) {
              
              uint new_next_aj = (new_aj < curI) ? new_aj+curI : new_aj;
              
              if (new_next_aj != next_aj) {

                future_handled = true;

                cur_acount(curI,effective_cur_aj)--;
                cur_acount(curI,effective_new_aj)++;

                uint jj=j+1;
                for (; jj < curJ; jj++) {
                  if (cur_alignment[jj] == next_aj) {
                    cur_alignment[jj] = new_next_aj;
                    if (jj  > j+1) {
                      cur_acount(curI,next_aj-curI)--;
                      cur_acount(curI,new_next_aj-curI)++;
                    }
                  }
                  else
                    break;
                }
                if (jj < curJ) {
                  assert(cur_alignment[jj] < curI);
                  cur_acount(std::min<uint>(curI,cur_alignment[jj]),next_aj-curI)--;
                  cur_acount(std::min<uint>(curI,cur_alignment[jj]),new_next_aj-curI)++;                  
                }
              }
            }
          }


          uint prev_word = cur_target_word; 
          uint new_word = (new_aj >= curI) ? 0 : cur_target[new_aj];
	      
          //recompute the stored values for the two affected words
          if (prev_word != new_word) {

            uint hyp_dict_num = (new_aj >= curI) ? 0 : cur_target[new_aj];
            Math1D::Vector<double>& hyp_dictcount = dcount[hyp_dict_num];
	    
            cur_dictcount[cur_idx] -= 1;
            hyp_dictcount[hyp_idx] += 1;
            dict_sum[cur_dict_num] -= 1;
            dict_sum[hyp_dict_num] += 1;
          }

          /****** change alignment counts *****/
	      
          //a) dependency to preceeding pos
          if (j == 0) {
	    
            //if (init_type != HmmInitFix) {
            if (true) {
              assert(icount[curI-1][cur_aj] > 0);
	      
              icount[curI-1][cur_aj]--;
              icount[curI-1][new_aj]++;
            }

            if (init_type == HmmInitPar) {

              if (cur_aj < curI && new_aj >= curI) {
                source_fert_count[1]--;
                source_fert_count[0]++;
              }
              else if (cur_aj >= curI && new_aj < curI) {
                source_fert_count[0]--;
                source_fert_count[1]++;
              }
            }
          }
          else {

            uint prev_aj = cur_alignment[j-1];
            if (prev_aj >= curI)
              prev_aj -= curI;
	    
            assert( cur_acount(std::min<ushort>(curI,cur_aj),prev_aj) > 0);
	      
            cur_acount(std::min<ushort>(curI,cur_aj),prev_aj)--;
            cur_acount(std::min<ushort>(curI,new_aj),prev_aj)++;
	    
            if (deficient_parametric) {

              if (cur_aj < curI) {
                dist_count[zero_offset + cur_aj - prev_aj]--;
                dist_count_sum--;
              }
              if (new_aj < curI) {
                dist_count[zero_offset + new_aj - prev_aj]++;
                dist_count_sum++;
              }
            }
	      
            if (cur_aj < curI && new_aj >= curI) {
              source_fert_count[1]--;
              source_fert_count[0]++;
            }
            else if (cur_aj >= curI && new_aj < curI) {
              source_fert_count[0]--;
              source_fert_count[1]++;
            }
          }
	      
          //b) dependency to succceeding pos
          if (j+1 < curJ && !future_handled) {

            uint next_aj = cur_alignment[j+1];
            uint effective_cur_aj = cur_aj;
            if (effective_cur_aj >= curI)
              effective_cur_aj -= curI;
            uint effective_new_aj = new_aj;
            if (effective_new_aj >= curI)
              effective_new_aj -= curI;
	    
            assert( cur_acount(std::min(curI,next_aj),effective_cur_aj) > 0);
		  
            cur_acount(std::min(curI,next_aj),effective_cur_aj)--;
            cur_acount(std::min(curI,next_aj),effective_new_aj)++;
	    
            if (deficient_parametric) {
              if (next_aj < curI) {
                dist_count[zero_offset + next_aj - effective_cur_aj]--;
                dist_count[zero_offset + next_aj - effective_new_aj]++;
              }
            }
          }
        }
      }
    }
    
    std::cerr << nSwitches << " switches in ICM" << std::endl;

#ifndef NDEBUG
    //DEBUG
    if (init_type == HmmInitFix || init_type == HmmInitFix2) {
      Math1D::Vector<double> check_source_fert_count(2,0.0);
    
      Storage1D<Math1D::Vector<double> > check_dcount(options.nTargetWords_);
      for (uint i=0; i < options.nTargetWords_; i++) {
        check_dcount[i].resize(dict[i].size(),0);
      }
      
      FullHMMAlignmentModel check_acount(maxI,MAKENAME(acount));
      for (uint I = 1; I <= maxI; I++) {
        if (seenIs.find(I) != seenIs.end()) {
          check_acount[I-1].resize(I+1,I,0.0);
        }
      }
      
      for (size_t s=0; s < nSentences; s++) {
        
        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];
        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);


        const uint curJ = cur_source.size();
        const uint curI = cur_target.size();
        
        Math2D::Matrix<double>& cur_facount = check_acount[curI-1];
        
        
        /**** update counts ****/      
        for (uint j=0; j < curJ; j++) {

          ushort aj = viterbi_alignment[s][j];
          if (aj >= curI) {
            check_dcount[0][cur_source[j]-1] += 1;
            if (align_type != HmmAlignProbNonpar && j > 0) 
              check_source_fert_count[0] += 1.0;
          }
          else {
            check_dcount[cur_target[aj]][cur_lookup(j,aj)] += 1;
            if (align_type != HmmAlignProbNonpar && j > 0) 
              check_source_fert_count[1] += 1.0;
          }
          
          if (j == 0) {
          }
          else {
            
            ushort prev_aj = viterbi_alignment[s][j-1];
            
            if (prev_aj >= curI) {
              
              if (aj >= curI) {
                cur_facount(curI,prev_aj-curI) += 1.0;
              }
              else {
                cur_facount(aj,prev_aj-curI) += 1.0;
              }
            }
            else {
              if (aj >= curI) {
                cur_facount(curI,prev_aj) += 1.0;
              }
              else {
                cur_facount(aj,prev_aj) += 1.0;
              }
            }
          }
        }
      }

      assert(check_dcount == dcount);
      for (uint I=0; I < acount.size(); I++) {
        //std::cerr << "I: " << I << std::endl; 
        
        if (check_acount[I] != acount[I]) {
          std::cerr << "should be: " << check_acount[I] << std::endl;
          std::cerr << "is: " << acount[I] << std::endl;
        }

        assert(check_acount[I] == acount[I]);
      }
      //assert(check_acount == acount);
      std::cerr << "should be: " << check_source_fert_count << std::endl;
      std::cerr << "is: " << source_fert_count << std::endl;
      
      assert(check_source_fert_count == source_fert_count);
    }
    //END_DEBUG
#endif

#endif

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts
    update_dict_from_counts(dcount, prior_weight, 0.0, iter,  false, 0.0, 0, dict);

    if (init_type == HmmInitPar) {
      
      init_count.set_constant(0.0);

      for (uint I=1; I <= maxI; I++) {
        
        if (initial_prob[I-1].size() != 0) {
          for (uint i=0; i < I; i++) {
            init_count[i] += icount[I-1][i];
          }
        }
      }
      
      ehmm_init_m_step(icount, init_params, options.init_m_step_iter_);
    }

    if (align_type != HmmAlignProbNonpar && align_type != HmmAlignProbNonpar2) {


      if (deficient_parametric) {
	
        double fdist_count_sum = dist_count.sum();
        assert(fdist_count_sum == dist_count_sum);
        for (uint k=0; k < dist_count.size(); k++)
          dist_params[k] = dist_count[k] / fdist_count_sum;
      }
      else {

        ehmm_m_step(acount, dist_params, zero_offset, options.align_m_step_iter_, dist_grouping_param);
        //par2nonpar can only be called when source_fert has been updated (may still get some counts from the init model!)
      }
    }


    //the changes in source-fert-counts are SO FAR NOT accounted for in the ICM hyp score calculations
    // nevertheless, updating them according to the new counts cannot worsen the energy
    if (source_fert_count.sum() > 1e-305) {
      for (uint k=0; k < 2; k++)
        source_fert[k] = source_fert_count[k] / source_fert_count.sum();
      std::cerr << "new source-fert: " << source_fert << std::endl;
    }

    //compute new alignment probabilities from normalized fractional counts
    for (uint I=1; I <= maxI; I++) {

      if (acount[I-1].xDim() != 0) {

        if (init_type == HmmInitNonpar) {
          double inv_norm = 1.0 / icount[I-1].sum();
          assert(!isnan(inv_norm));
          for (uint i=0; i < initial_prob[I-1].size(); i++)
            initial_prob[I-1][i] = inv_norm * icount[I-1][i]; 
        }

        if (align_type == HmmAlignProbNonpar) {

          for (uint i=0; i < I; i++) {
	    
            double sum = 0.0;
            for (uint i_next = 0; i_next <= I; i_next++)
              sum += acount[I-1](i_next,i);
	    
            if (sum >= 1e-300) {
	      
              assert(!isnan(sum));
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));
	      
              for (uint i_next = 0; i_next <= I; i_next++) {
                align_model[I-1](i_next,i) = inv_sum *acount[I-1](i_next,i);
                // 	      if (isnan(align_model[I-1](i_next,i)))
                // 		std::cerr << "nan: " << inv_sum << " * " << acount[I-1](i_next,i) << std::endl;
		
                // 	      assert(!isnan(align_model[I-1](i_next,i)));
              }
            }
          }
        }
        else if (align_type == HmmAlignProbNonpar2) {

          //changes in these counts are so far not included in the ICM score calculation.
          // nevertheless, updating the probabilities according to the new counts cannor worsen the energy

          for (uint i=0; i < I; i++) {
	    
            double sum = 0.0;
            for (uint i_next = 0; i_next < I; i_next++)
              sum += acount[I-1](i_next,i);

            align_model[I-1](I,i) = source_fert[0];

            if (sum >= 1e-300) {
	      
              assert(!isnan(sum));
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));

              for (uint i_next = 0; i_next < I; i_next++) {
                align_model[I-1](i_next,i) = inv_sum * source_fert[1] * acount[I-1](i_next,i);
                // 	      if (isnan(align_model[I-1](i_next,i)))
                // 		std::cerr << "nan: " << inv_sum << " * " << acount[I-1](i_next,i) << std::endl;
		
                // 	      assert(!isnan(align_model[I-1](i_next,i)));
              }
            }
            else {
              for (uint i_next = 0; i_next < I; i_next++) {
                align_model[I-1](i_next,i) = source_fert[1] / double(I);
              }
            }               
          }          
        }
        else if (deficient_parametric) {

          for (uint i=0; i < I; i++) {	    

            double factor = 1.0;

            //NOTE: we do NOT normalize here (to get a computationally tractable model)
            for (uint ii=0; ii < I; ii++)
              align_model[I-1](ii,i) = source_fert[1] * factor * dist_params[zero_offset + ii - i];
            align_model[I-1](I,i) = source_fert[0];
          }
        }
      }
    }


    if (init_type == HmmInitPar)
      //can only call this AFTER source_fert has been updated
      par2nonpar_hmm_init_model(init_params, source_fert, init_type,  initial_prob, start_empty_word);

    if (align_type != HmmAlignProbNonpar) {
      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, align_model);
    }


    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      
      double sum_aer = 0.0;
      double sum_marg_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = options.possible_ref_alignments_.begin();
          it != options.possible_ref_alignments_.end(); it ++) {

        uint s = it->first-1;

	if (s >= nSentences)
	  break;

        nContributors++;
        //compute viterbi alignment

        Storage1D<AlignBaseType> viterbi_alignment;
        const uint curI = target[s].size();

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s],target[s],wcooc,nSourceWords,slookup[s],aux_lookup);
	
        if (start_empty_word)
          compute_sehmm_viterbi_alignment(source[s],cur_lookup, target[s], 
                                          dict, align_model[curI-1], initial_prob[curI-1],
                                          viterbi_alignment, false, false, 0.0);
        else
          compute_ehmm_viterbi_alignment(source[s],cur_lookup, target[s], 
                                         dict, align_model[curI-1], initial_prob[curI-1],
                                         viterbi_alignment, align_type, false, false, 0.0);
        
        //add alignment error rate
        sum_aer += AER(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        sum_fmeasure += f_measure(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        
        if (!start_empty_word) {
          Storage1D<AlignBaseType> marg_alignment;
	  
          compute_ehmm_optmarginal_alignment(source[s],cur_lookup, target[s], 
                                             dict, align_model[curI-1], initial_prob[curI-1],
                                             marg_alignment);
          
          sum_marg_aer += AER(marg_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        }
      }

      sum_aer *= 100.0 / nContributors;
      sum_marg_aer *= 100.0 / nContributors;
      nErrors /= nContributors;
      sum_fmeasure /= nContributors;
      
      std::cerr << "#### EHMM Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "---- EHMM Marginal-AER : " << sum_marg_aer << " %" << std::endl;
      std::cerr << "#### EHMM Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;

    }
  } //end for (iter)

}


