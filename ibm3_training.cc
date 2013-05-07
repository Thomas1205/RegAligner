/** author: Thomas Schoenemann. This file was generated from singleword_fertility_training while 
    Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012. It was subsequently 
    modified and extended, both at the University of Düsseldorf and in his free time. ***/

#include "ibm3_training.hh"
#include "combinatoric.hh"
#include "alignment_computation.hh"
#include "hmm_forward_backward.hh"
#include "timing.hh"
#include "projection.hh"
#include "training_common.hh" // for get_wordlookup() and dictionary m-step 
#include "stl_util.hh"
#include "storage_stl_interface.hh"


#ifdef HAS_CBC
#include "sparse_matrix_description.hh"
#include "ClpSimplex.hpp"
#include "CbcModel.hpp"
#include "OsiClpSolverInterface.hpp"

#include "CglGomory/CglGomory.hpp"
#include "CglProbing/CglProbing.hpp"
#include "CglRedSplit/CglRedSplit.hpp"
#include "CglTwomir/CglTwomir.hpp"
#include "CglMixedIntegerRounding/CglMixedIntegerRounding.hpp"
#include "CglMixedIntegerRounding2/CglMixedIntegerRounding2.hpp"
#include "CglOddHole/CglOddHole.hpp"
#include "CglLandP/CglLandP.hpp"
#include "CglClique/CglClique.hpp"
#include "CglStored.hpp"

#include "CbcHeuristic.hpp"
#endif

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"


/************************** implementation of IBM3Trainer *********************/

IBM3Trainer::IBM3Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
			 const LookupTable& slookup,
                         const Storage1D<Storage1D<uint> >& target_sentence,
                         const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                         const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                         SingleWordDictionary& dict,
                         const CooccuringWordsType& wcooc,
                         uint nSourceWords, uint nTargetWords,
                         const floatSingleWordDictionary& prior_weight,
			 const Math1D::Vector<double>& log_table,
                         bool parametric_distortion, bool och_ney_empty_word, 
                         bool nondeficient, 
                         bool viterbi_ilp, 
                         double l0_fertpen, bool smoothed_l0, double l0_beta)
: FertilityModelTrainer(source_sentence,slookup,target_sentence,dict,wcooc,nSourceWords,nTargetWords,
			prior_weight, och_ney_empty_word, smoothed_l0, l0_beta, l0_fertpen,
			sure_ref_alignments,possible_ref_alignments,log_table),
  distortion_prob_(MAKENAME(distortion_prob_)), parametric_distortion_(parametric_distortion), viterbi_ilp_(viterbi_ilp),
  nondeficient_(nondeficient)
{

#ifndef HAS_CBC
  viterbi_ilp_ = false;
#endif

  
  distortion_prob_.resize(maxJ_);
  distortion_param_.resize(maxJ_,maxI_,1.0 / maxJ_);

  Math1D::Vector<uint> max_I(maxJ_,0);

  for (size_t s=0; s < source_sentence_.size(); s++) {

    const uint curI = target_sentence_[s].size();
    const uint curJ = source_sentence_[s].size();

    if (curI > max_I[curJ-1])
      max_I[curJ-1] = curI;
  }

  for (uint J=0; J < maxJ_; J++) {
    distortion_prob_[J].resize_dirty(J+1,max_I[J]);
    distortion_prob_[J].set_constant(1.0 / (J+1));
  }

  denom_cache_.resize(maxI_+1); 
}

/*virtual*/ std::string IBM3Trainer::model_name() const {

  return "IBM-3";
}

void IBM3Trainer::init_from_hmm(HmmWrapper& hmm_wrapper,
				bool count_collection,	bool viterbi) {


  const HmmAlignProbType align_type = hmm_wrapper.hmm_options_.align_type_;
  const bool start_empty_word = hmm_wrapper.hmm_options_.start_empty_word_;

  std::cerr << "initializing IBM-3 from HMM" << std::endl;

  if (count_collection) {

    if (viterbi)
      train_viterbi(1,&hmm_wrapper);
    else
      train_unconstrained(1,&hmm_wrapper);

    iter_offs_ = 1;
  }
  else {

    NamedStorage1D<Math1D::Vector<double> > fert_count(nTargetWords_,MAKENAME(fert_count));
    for (uint i=0; i < nTargetWords_; i++) {
      fert_count[i].resize(fertility_prob_[i].size(),0.0);
    }
    
    SingleLookupTable aux_lookup;
    
    for (size_t s=0; s < source_sentence_.size(); s++) {

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      
      const uint curI = target_sentence_[s].size();
      const uint curJ = source_sentence_[s].size();
      
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
							   nSourceWords_,slookup_[s],aux_lookup);
      
      
      Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      if (hmm_wrapper.initial_prob_.size() == 0) 
	compute_fullhmm_viterbi_alignment(cur_source, cur_lookup, cur_target, 
					  dict_, hmm_wrapper.align_model_[curI-1], cur_alignment);
      else {
	if (start_empty_word) {
	  compute_sehmm_viterbi_alignment(cur_source, cur_lookup, cur_target, 
					  dict_, hmm_wrapper.align_model_[curI-1], hmm_wrapper.initial_prob_[curI-1],
					  cur_alignment);
	}
	else {
	  
	  compute_ehmm_viterbi_alignment(cur_source, cur_lookup, cur_target, 
					 dict_, hmm_wrapper.align_model_[curI-1], hmm_wrapper.initial_prob_[curI-1],
					 cur_alignment,align_type);
	}
      }
      
      Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));
      
      for (uint j=0; j < curJ; j++) {
	const uint aj = cur_alignment[j];
	fertility[aj]++;
      }

      make_alignment_feasible(cur_source, cur_target, cur_lookup, best_known_alignment_[s]);

      fert_count[0][fertility[0]]++;
      
      for (uint i=0; i < curI; i++) {
	const uint t_idx = target_sentence_[s][i];
	
	fert_count[t_idx][fertility[i+1]]++;
      }
    }

    //init fertility prob. by weighted combination of uniform distribution 
    // and counts from Viterbi alignments
  
    const double count_weight = 0.95;    
    const double uni_weight = 1.0 - count_weight;

    for (uint i=0; i < nTargetWords_; i++) {

      const Math1D::Vector<double>& cur_fert_count = fert_count[i];
      
      const uint max_fert = cur_fert_count.size();
      
      double fc_sum = cur_fert_count.sum();

      if (fc_sum > 1e-300) {
	
	const double inv_fc_sum = 1.0 / fc_sum;
	
	const double uni_contrib = uni_weight / std::min(max_fert,fertility_limit_);
	for (uint f=0; f < max_fert; f++) {
	  
	  if (f <= fertility_limit_)
	    fertility_prob_[i][f] = uni_contrib + count_weight * inv_fc_sum * cur_fert_count[f];
	  else
	    fertility_prob_[i][f] = count_weight * inv_fc_sum * cur_fert_count[f];
	}
      }
      else {
	//this can happen if there is a development corpus
	for (uint f=0; f < max_fert; f++) {
	  
	  if (f <= fertility_limit_)
	    fertility_prob_[i][f] = 1.0 / std::min(fertility_limit_+1,max_fert);
	  else
	    fertility_prob_[i][f] = 1e-8;
	}
      }
    }

    
    if (!viterbi)
      std::cerr << "initializing distortion prob. by forward-backward HMM" << std::endl;
    
    ReducedIBM3DistortionModel fdcount(distortion_prob_.size(),MAKENAME(fdcount));
    for (uint J=0; J < distortion_prob_.size(); J++) {
      fdcount[J].resize(J+1,distortion_prob_[J].yDim(),0.0);
    }
    
    if (!fix_p0_) {
      p_zero_ = 0.0;
      p_nonzero_ = 0.0;
    }
    
    Math1D::Vector<double> empty_vec;

    //init distortion probabilities using forward-backward
    for (size_t s=0; s < source_sentence_.size(); s++) {
      
      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
							   nSourceWords_,slookup_[s],aux_lookup);
    
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      
      Math2D::NamedMatrix<long double> forward(2*curI,curJ,MAKENAME(forward));
      Math2D::NamedMatrix<long double> backward(2*curI,curJ,MAKENAME(backward));

      if (!viterbi) {
	
	const Math1D::Vector<double> init_prob = (hmm_wrapper.initial_prob_.size() > 0) 
	  ? hmm_wrapper.initial_prob_[curI-1] : empty_vec;
      
	if (start_empty_word) {
	  
	  calculate_sehmm_forward(cur_source, cur_target, cur_lookup, dict_,
				  hmm_wrapper.align_model_[curI-1], init_prob, forward);
	  
	}
	else {
	  
	  calculate_hmm_forward(cur_source, cur_target, cur_lookup, dict_,
				hmm_wrapper.align_model_[curI-1], init_prob, align_type, forward);
	}
	
	if (start_empty_word) {
	  
	  calculate_sehmm_backward(cur_source, cur_target, cur_lookup, dict_,
				   hmm_wrapper.align_model_[curI-1], init_prob, backward, false);
	}
	else  {
	  
	  calculate_hmm_backward(cur_source, cur_target, cur_lookup, dict_,
				 hmm_wrapper.align_model_[curI-1], init_prob, align_type, backward, false);
	}
      }
      
      // extract fractional counts
      long double sentence_prob = 0.0;
      if (!viterbi) {
	for (uint i=0; i < forward.xDim(); i++)
	  sentence_prob += forward(i,curJ-1);
      }
      long double inv_sentence_prob = (viterbi) ? 1.0 : 1.0 / sentence_prob;
      
      assert(!isnan(inv_sentence_prob));
      
      if (viterbi) {
	
	for (uint j=0; j < best_known_alignment_[s].size(); j++) {
	  
	  uint aj = best_known_alignment_[s][j];
        
	  if (!fix_p0_) {
	    
	    if (aj == 0)
	      p_zero_ += 1.0;
	    else 
	      p_nonzero_ += 1.0;
	  }

	  if (aj != 0)
	    fdcount[curJ-1](j,aj-1) += 1.0;
	}
      }
      else {
	
	for (uint i=0; i < curI; i++) {
	  const uint t_idx = cur_target[i];
	  
	  for (uint j=0; j < curJ; j++) {
	    
	    if (dict_[t_idx][cur_lookup(j,i)] > 1e-305) {
	      double contrib = inv_sentence_prob * forward(i,j) * backward(i,j) 
		/ dict_[t_idx][cur_lookup(j,i)];
	      
	      fdcount[curJ-1](j,i) += contrib;
	      if (!fix_p0_)
		p_nonzero_ += contrib;
	    }
	  }
	}
	for (uint i=curI; i < forward.xDim(); i++) {
	  for (uint j=0; j < curJ; j++) {
	    const uint s_idx = cur_source[j];
	    
	    if (dict_[0][s_idx-1] > 1e-305) {
	      double contrib = inv_sentence_prob * forward(i,j) * backward(i,j) 
		/ dict_[0][s_idx-1];
	      
	      if (!fix_p0_)     
		p_zero_ += contrib;
	    }
	  }
	}
      }
    }

    if (!fix_p0_) {
      double p_norm = p_zero_ + p_nonzero_;
      assert(p_norm != 0.0);
      p_zero_ /= p_norm;
      p_nonzero_ /= p_norm;

      //TRIAL
      // p0 of IBM-3 and IBM-4 is different from p0 for the HMM: if 50% of the words align to zero, p0 is 100% for IBM3/4
      p_zero_ = std::min(2*p_zero_,0.75);
      p_nonzero_ = 1.0 - p_zero_;
      //END_TRIAL
    }
    
    std::cerr << "initial value of p_zero: " << p_zero_ << std::endl;

    if (parametric_distortion_)
      distortion_param_.set_constant(0.0);
    
    for (uint J=0; J < distortion_prob_.size(); J++) {
      
      for (uint i=0; i < distortion_prob_[J].yDim(); i++) {
	
	double sum = 0.0;
	for (uint j=0; j < J+1; j++)
	  sum += fdcount[J](j,i);
	
	assert(!isnan(sum));
      
	if (sum > 1e-305) {
	  double inv_sum = 1.0 / sum;
        
	  assert(!isnan(inv_sum));
        
	  for (uint j=0; j < J+1; j++) {
	    
	    if (parametric_distortion_)
	      distortion_param_(j,i) += fdcount[J](j,i);
	    else {
	      if (viterbi)
		distortion_prob_[J](j,i) = 0.8 * inv_sum * fdcount[J](j,i) + 0.2 / (J+1);
	      else
		distortion_prob_[J](j,i) = 0.9 * inv_sum * fdcount[J](j,i) + 0.1 / (J+1);
	    }
	  }
	}
	else if (!parametric_distortion_) {
	  for (uint j=0; j < J+1; j++)
	    distortion_prob_[J](j,i) = 1.0 / (J+1);
	}
      }
    }

    if (parametric_distortion_) {
      for (uint i=0; i < distortion_param_.yDim(); i++) {
	
	double sum = 0.0;
	for (uint j=0; j < distortion_param_.xDim(); j++)
	  sum += distortion_param_(j,i);
	
	if (sum < 1e-305) {
	  for (uint j=0; j < distortion_param_.xDim(); j++)
	    distortion_param_(j,i) = 1.0 / distortion_param_.xDim();
	}
	else {
	  for (uint j=0; j < distortion_param_.xDim(); j++) { 
	    if (viterbi)
	      distortion_param_(j,i) = 0.9 * distortion_param_(j,i) / sum + 0.1 / distortion_param_.xDim();
	    else
	      distortion_param_(j,i) = 0.95 * distortion_param_(j,i) / sum + 0.05 / distortion_param_.xDim();
	  }
	}
      }
      par2nonpar_distortion(distortion_prob_);
    }   
  }
}

void IBM3Trainer::par2nonpar_distortion(ReducedIBM3DistortionModel& prob) {

  for (uint J=0; J < prob.size(); J++) {

    for (uint i=0; i < prob[J].yDim(); i++) {

      assert(distortion_param_.xDim() >= J+1);
      assert(prob[J].xDim() == J+1);

      double sum = 0.0;
      for (uint j=0; j < J+1; j++)
        sum += distortion_param_(j,i);
      
      assert(!isnan(sum));
      
      if (sum > 1e-305) {
        double inv_sum = 1.0 / sum;

        assert(!isnan(inv_sum));

        for (uint j=0; j < J+1; j++)
          prob[J](j,i) = std::max(1e-8,inv_sum * distortion_param_(j,i));
      }
    }
  }
}

double IBM3Trainer::par_distortion_m_step_energy(const ReducedIBM3DistortionModel& fdistort_count,
                                                 const Math2D::Matrix<double>& param, uint i) {

  Math1D::Vector<double> vec(param.xDim());
  for (uint x=0; x < vec.size(); x++)
    vec[x] = param(x,i);

  return par_distortion_m_step_energy(fdistort_count,vec,i);
}

double IBM3Trainer::par_distortion_m_step_energy(const ReducedIBM3DistortionModel& fdistort_count,
                                                 const Math1D::Vector<double>& param, uint i) {

 double energy = 0.0;

  for (uint J=0; J < maxJ_; J++) {

    const Math2D::Matrix<double>& cur_distort_count = fdistort_count[J];

    if (cur_distort_count.yDim() > i) {

      double sum = 0.0;
      
      for (uint j=0; j < cur_distort_count.xDim(); j++) {
        sum += param[j];
      }
      
      for (uint j=0; j < cur_distort_count.xDim(); j++) {
        energy -= cur_distort_count(j,i) * std::log( param[j] / sum);
      }
    }
  }

  return energy;
}

void IBM3Trainer::par_distortion_m_step(const ReducedIBM3DistortionModel& fdistort_count, uint i, double start_energy) {


  double alpha = 0.1;

  Math1D::Vector<double> start_param(maxJ_);
  for (uint j=0; j < maxJ_; j++)
    start_param[j] = distortion_param_(j,i);

  double energy = start_energy; 

  Math1D::Vector<double> distortion_grad(maxJ_,0.0);
  Math1D::Vector<double> new_distortion_param(maxJ_,0.0);
  Math1D::Vector<double> hyp_distortion_param(maxJ_,0.0);

  double line_reduction_factor = 0.35;

  for (uint iter = 1; iter <= 400; iter++) {

    distortion_grad.set_constant(0.0);

    /*** compute gradient ***/
    for (uint J=0; J < maxJ_; J++) {

      const Math2D::Matrix<double>& cur_distort_count = fdistort_count[J];

      if (cur_distort_count.yDim() > i) {

        double sum = 0.0;
        double count_sum =0.0;
        
        for (uint j=0; j < cur_distort_count.xDim(); j++) {
          sum += std::max(1e-15,distortion_param_(j,i));
          count_sum += cur_distort_count(j,i);
        }
        
        for (uint j=0; j < cur_distort_count.xDim(); j++) {
          
          const double cur_param = std::max(1e-15,distortion_param_(j,i));
          
          distortion_grad[j] += count_sum / sum;
          distortion_grad[j] -= cur_distort_count(j,i) / cur_param;
        }
      }
    }

    /*** go on neg. gradient direction and reproject ***/
    for (uint j=0; j < distortion_param_.xDim(); j++) 
      new_distortion_param[j] = distortion_param_(j,i) - alpha * distortion_grad[j];

    projection_on_simplex(new_distortion_param.direct_access(),maxJ_);


    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      for (uint j=0; j < distortion_param_.xDim(); j++)   
        hyp_distortion_param[j] = lambda * new_distortion_param[j]
          + neg_lambda * distortion_param_(j,i);

      double hyp_energy = par_distortion_m_step_energy(fdistort_count, hyp_distortion_param, i);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
	break;
    }


    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    //EXPERIMENTAL
    // if (nIter > 4)
    //   alpha *= 1.5;
    //END_EXPERIMENTAL

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint j=0; j < distortion_param_.xDim(); j++)   
      distortion_param_(j,i) = best_lambda * new_distortion_param[j]
        + neg_best_lambda * distortion_param_(j,i);

    energy = best_energy;
  }
}

double IBM3Trainer::nondeficient_m_step_energy(const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count,
                                               const Math2D::Matrix<double>& param, uint i) {

  double energy = 0.0;

  const double* cur_param = param.direct_access() + i*param.xDim();
  
  for (std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >::const_iterator it = count.begin();
       it != count.end(); it++) {
    
    const Math1D::Vector<uchar,uchar>& open_positions = it->first;
    const uchar size = open_positions.size();
    const uchar filled_pos = open_positions[size-1];
    const double weight = it->second;
    
    double sum = 0.0;
    for (uint k=0; k < size; k++)
      sum += cur_param[ open_positions[k] ]; 
    
    energy -= weight * std::log(cur_param[filled_pos]/sum);
  }
  
  return energy;
}

double IBM3Trainer::nondeficient_m_step_energy(const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
					       const std::vector<Math1D::Vector<double,uchar> >& count,
					       const Math2D::Matrix<double>& param, uint i) {



  double energy = 0.0;

  const double* cur_param = param.direct_access() + i*param.xDim();

  assert(open_pos.size() == count.size());

  for (uint l=0; l < open_pos.size(); l++) {

    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[l];
    const uchar size = open_positions.size();

    double sum = 0.0;
    for (uint k=0; k < size; k++)
      sum += cur_param[ open_positions[k] ]; 
    
    const Math1D::Vector<double,uchar>& weight = count[l];

    assert(weight.size() <= size); //tail 0s have been removed

    for (uint k=0; k < weight.size(); k++) {

      double cur_weight = weight[k];

      if (cur_weight == 0.0)
	continue;

      energy -= cur_weight * std::log(cur_param[open_positions[k]]/sum);
    }
  }

  return energy;  
}


double IBM3Trainer::nondeficient_m_step_energy(const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, 
                                               const std::vector<double>& count,
                                               const Math2D::Matrix<double>& param, uint i) {

  double energy = 0.0;

  const double* cur_param = param.direct_access() + i*param.xDim();
  
  for (uint k=0; k < open_pos.size(); k++) {
    
    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];
    const uchar size = open_positions.size();
    const uchar filled_pos = open_positions[size-1];
    const double weight = count[k];
    
    double sum = 0.0;
    for (uint k=0; k < size; k++)
      sum += cur_param[ open_positions[k] ]; 
    
    energy -= weight * std::log(cur_param[filled_pos]/sum);
  }
  
  return energy;
}


//with interpolation
double IBM3Trainer::nondeficient_m_step_energy(const std::vector<double>& count, const Storage1D<uchar>& filled_position,
                                               const Math2D::Matrix<double>& param1, const Math2D::Matrix<double>& param2,
                                               const Math1D::Vector<double>& sum1, const Math1D::Vector<double>& sum2, 
                                               uint i, double lambda) {

  assert(lambda >= 0.0 && lambda <= 1.0);

  const double neg_lambda = 1.0 - lambda;

  double energy = 0.0;

  const double* cur_param1 = param1.direct_access() + i*param1.xDim();
  const double* cur_param2 = param2.direct_access() + i*param2.xDim();
  
  for (uint k=0; k < count.size(); k++) {

    const uchar filled_pos = filled_position[k];
    const double weight = count[k];

    double sum = lambda * sum2[k] + neg_lambda * sum1[k];
    
    double param = lambda * cur_param2[filled_pos] + neg_lambda * cur_param1[filled_pos];

    energy -= weight * std::log(param/sum);
  }

  return energy;
}

//with interpolation
double IBM3Trainer::nondeficient_m_step_energy(const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
					       const std::vector<Math1D::Vector<double,uchar> >& count,
					       const Math2D::Matrix<double>& param1, const Math2D::Matrix<double>& param2,
					       const Math1D::Vector<double>& sum1, const Math1D::Vector<double>& sum2, 
					       uint i, double lambda) {

  assert(lambda >= 0.0 && lambda <= 1.0);

  const double neg_lambda = 1.0 - lambda;

  double energy = 0.0;
  
  const double* cur_param1 = param1.direct_access() + i*param1.xDim();
  const double* cur_param2 = param2.direct_access() + i*param2.xDim();

  assert(open_pos.size() == count.size());

  for (uint l=0; l < open_pos.size(); l++) {

    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[l];

    double sum = lambda * sum2[l] + neg_lambda * sum1[l];

    const Math1D::Vector<double,uchar>& weight = count[l];

    assert(weight.size() <= open_positions.size()); //tail 0s have been removed

    for (uint k=0; k < weight.size(); k++) {

      double cur_weight = weight[k];

      if (cur_weight == 0.0)
	continue;

      double param = lambda * cur_param2[open_positions[k]] + neg_lambda * cur_param1[open_positions[k]];

      energy -= cur_weight * std::log(param/sum);
    }
  }

  return energy;
}

// for the nonparametric model
double IBM3Trainer::nondeficient_m_step_energy(const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count,
                                               const Storage1D<Math2D::Matrix<double> >& param, uint i, uint J) {

  double energy = 0.0;

  const double* cur_param = param[J-1].direct_access() + i*param[J-1].xDim();
  
  for (std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >::const_iterator it = count.begin();
       it != count.end(); it++) {
    
    const Math1D::Vector<uchar,uchar>& open_positions = it->first;
    const uchar size = open_positions.size();
    const uchar filled_pos = open_positions[size-1];
    const double weight = it->second;
    
    double sum = 0.0;
    for (uint k=0; k < size; k++)
      sum += cur_param[ open_positions[k] ]; 
    
    energy -= weight * std::log(cur_param[filled_pos]/sum);
  }
  
  return energy;
}

double IBM3Trainer::nondeficient_m_step_with_interpolation(const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, 
                                                           const std::vector<double>& weight,
                                                           uint i, double start_energy) {

  assert(distortion_param_.xDim() == maxJ_);
  assert(i < distortion_param_.yDim());

  Math1D::Vector<double> sum(weight.size());
  Math1D::Vector<double> new_sum(weight.size(),0.0);

  Storage1D<uchar> filled_position(weight.size());

  for (uint k=0; k < weight.size(); k++) {

    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k]; 

    assert(open_positions.size() > 0);
    
    double cur_sum = 0.0;
    for (uchar l=0; l < open_positions.size(); l++) {
      cur_sum += std::max(1e-15,distortion_param_(open_positions[l],i));
    }
    sum[k] = cur_sum;

    filled_position[k] = open_positions[open_positions.size()-1];
  }

  double alpha = 0.01; //0.1;
  
  double energy = start_energy; 

  std::cerr << "start energy: " << energy << std::endl;
  
  Math1D::Vector<double> distortion_grad(distortion_param_.xDim(),0.0);
  Math2D::Matrix<double> new_distortion_param = distortion_param_;

  assert(distortion_param_.xDim() == maxJ_);
  
  double line_reduction_factor = 0.35;
  
  std::clock_t tStart = std::clock();

  const double* cur_param = distortion_param_.direct_access() + i*distortion_param_.xDim();

  double save_energy = energy;

  for (uint iter = 1; iter <= 250 /*400*/; iter++) {
    
    if ((iter % 50) == 0) {
      std::cerr << "m-step iter # " << iter << ", energy: " << energy << std::endl;

      std::clock_t tInter = std::clock();
      std::cerr << "spent " << diff_seconds(tInter,tStart) << " seconds so far" << std::endl;

      if (save_energy - energy < 0.35)
        break;
      if (iter >= 100 && save_energy - energy < 0.75)
        break;
      

      save_energy = energy;
    }

    distortion_grad.set_constant(0.0);

    /*** compute gradient ***/
    
    int k=-1;
    for (std::vector<Math1D::Vector<uchar,uchar> >::const_iterator it = open_pos.begin();
         it != open_pos.end(); it++) {

      k++;

      const Math1D::Vector<uchar,uchar>& open_positions = *it;
      const double cur_weight = weight[k];

      const uchar filled_pos = filled_position[k];
      
      double cur_sum = std::max(1e-15,sum[k]);
      
      distortion_grad[filled_pos] -= cur_weight / std::max(cur_param[filled_pos],1e-15);
      
      const double addon = cur_weight / cur_sum;
      
      for (uint k=0; k < open_positions.size(); k++) {
        assert(open_positions[k] < distortion_param_.xDim());
        distortion_grad[open_positions[k]] += addon;
      }
    }
  
    /*** go in neg. gradient direction and reproject ***/

    for (uint j=0; j < maxJ_; j++) {
      new_distortion_param(j,i) = distortion_param_(j,i) - alpha * distortion_grad[j];

      if (new_distortion_param(j,i) < -1e75) {
        std::cerr << "fixing numerical instability" << std::endl;
        new_distortion_param(j,i) = -1e-75;
      }
      if (new_distortion_param(j,i) > 1e75) {
        std::cerr << "fixing numerical instability" << std::endl;
        new_distortion_param(j,i) = 1e-75;
      }
    }

    projection_on_simplex(new_distortion_param.direct_access()+i*distortion_param_.xDim(), maxJ_);

    for (uint j=0; j < maxJ_; j++)
      new_distortion_param(j,i) = std::max(1e-15,new_distortion_param(j,i));

    /*** find appropriate step-size ***/
    for (uint k=0; k < weight.size(); k++) {
      
      const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];
      
      double cur_new_sum = 0.0;
      for (uchar l=0; l < open_positions.size(); l++)
        cur_new_sum += new_distortion_param(open_positions[l],i);
      new_sum[k] = cur_new_sum;
    }


    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;

      double hyp_energy = nondeficient_m_step_energy(weight, filled_position, distortion_param_,new_distortion_param,
                                                     sum,new_sum, i, lambda);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
	break;
    }

    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }


    if (nIter > 6)
      line_reduction_factor *= 0.9;

    //EXPERIMENTAL
    // if (nIter > 4)
    //   alpha *= 1.5;
    //END_EXPERIMENTAL

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint j=0; j < maxJ_; j++)
      distortion_param_(j,i) = best_lambda * new_distortion_param(j,i) + neg_best_lambda * distortion_param_(j,i);

    energy = best_energy;

    for (uint k=0; k < weight.size(); k++) {
      sum[k] = neg_best_lambda * sum[k] + best_lambda * new_sum[k];
    }
  }

  //FOR_SAFETY
  for (uint j=0; j < maxJ_; j++)
    distortion_param_(j,i) = std::max(distortion_param_(j,i),1e-15);
  //END_FOR_SAFETY

  return energy;
}

double IBM3Trainer::nondeficient_m_step_with_interpolation(const std::vector<Math1D::Vector<uchar,uchar> >& open_pos,
							   const std::vector<Math1D::Vector<double,uchar> >& count, 
							   uint i, double start_energy) {


  assert(distortion_param_.xDim() == maxJ_);
  assert(i < distortion_param_.yDim());

  Math1D::Vector<double> sum(count.size());
  Math1D::Vector<double> new_sum(count.size(),0.0);

  for (uint k=0; k < count.size(); k++) {

    const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k]; 

    assert(open_positions.size() > 0);
    
    double cur_sum = 0.0;
    for (uchar l=0; l < open_positions.size(); l++) {
      cur_sum += std::max(1e-15,distortion_param_(open_positions[l],i));
    }
    sum[k] = cur_sum;
  }

  double alpha = 0.01; //0.1;
  
  double energy = start_energy; 

  std::cerr << "start energy: " << energy << std::endl;


  Math1D::Vector<double> distortion_grad(distortion_param_.xDim(),0.0);
  Math2D::Matrix<double> new_distortion_param = distortion_param_;
  
  double line_reduction_factor = 0.35;
  
  std::clock_t tStart = std::clock();

  for (uint iter = 1; iter <= 250; iter++) {
    
    if ((iter % 50) == 0) {
      std::cerr << "m-step iter # " << iter << ", energy: " << energy << std::endl;

      std::clock_t tInter = std::clock();
      std::cerr << "spent " << diff_seconds(tInter,tStart) << " seconds so far" << std::endl;
    }

    distortion_grad.set_constant(0.0);
    
    /*** compute gradient ***/

    const double* cur_param = distortion_param_.direct_access() + i*distortion_param_.xDim();

    for (uint l=0; l < open_pos.size(); l++) {
      
      const Math1D::Vector<uchar,uchar>& open_positions = open_pos[l];

      const Math1D::Vector<double,uchar>& weight = count[l];
      
      double cur_sum = sum[l];
      cur_sum = std::max(cur_sum,1e-15);

      double weight_sum = 0.0;

      assert(weight.size() <= open_positions.size()); //tail 0s have been removed

      for (uint k=0; k < weight.size(); k++) {

	const double cur_weight = weight[k];

	const uint cur_pos = open_positions[k];

	if (cur_weight == 0.0)
	  continue;

	weight_sum += cur_weight;

	distortion_grad[cur_pos] -= cur_weight / std::max(cur_param[cur_pos],1e-15);
      }

      const double addon = weight_sum / cur_sum;
      
      for (uint k=0; k < open_positions.size(); k++)
        distortion_grad[open_positions[k]] += addon;
    }
  
    /*** go in neg. gradient direction and reproject ***/

    for (uint j=0; j < maxJ_; j++)
      new_distortion_param(j,i) = distortion_param_(j,i) - alpha * distortion_grad[j];

    projection_on_simplex(new_distortion_param.direct_access()+i*distortion_param_.xDim(), maxJ_);

    for (uint k=0; k < count.size(); k++) {
      
      const Math1D::Vector<uchar,uchar>& open_positions = open_pos[k];
      
      double cur_new_sum = 0.0;
      for (uchar l=0; l < open_positions.size(); l++)
        cur_new_sum += new_distortion_param(open_positions[l],i);
      new_sum[k] = cur_new_sum;
    }


    /*** find appropriate step-size ***/

    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;


      double hyp_energy = nondeficient_m_step_energy(open_pos, count, distortion_param_,new_distortion_param,
						     sum,new_sum,i,lambda);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
	break;
    }

    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint j=0; j < maxJ_; j++)
      distortion_param_(j,i) = std::max(1e-8, best_lambda * new_distortion_param(j,i) + neg_best_lambda * distortion_param_(j,i));

    energy = best_energy;

    for (uint k=0; k < count.size(); k++) {
      sum[k] = neg_best_lambda * sum[k] + best_lambda * new_sum[k];
    }
  }


 //FOR_SAFETY
  for (uint j=0; j < maxJ_; j++)
    distortion_param_(j,i) = std::max(distortion_param_(j,i),1e-15);
  //END_FOR_SAFETY

  return energy;
}

//for the nonparametric model
double IBM3Trainer::nondeficient_m_step(const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count, uint i, uint J, 
					double start_energy) {

  double alpha = 0.1;
  
  double energy = start_energy; 

  const uint xDim = distortion_prob_[J-1].xDim();
  assert(xDim == J);
  
  Math1D::Vector<double> distortion_grad(xDim,0.0);
  ReducedIBM3DistortionModel new_distortion_prob = distortion_prob_;
  ReducedIBM3DistortionModel hyp_distortion_prob = distortion_prob_;

  Math2D::Matrix<double>& relevant_distortion_prob = distortion_prob_[J-1];
  Math2D::Matrix<double>& relevant_new_distort_prob = new_distortion_prob[J-1];
  Math2D::Matrix<double>& relevant_hyp_distort_prob = hyp_distortion_prob[J-1];

  
  double line_reduction_factor = 0.35;
  
  for (uint iter = 1; iter <= 250 /*400*/; iter++) {

    distortion_grad.set_constant(0.0);
    
    /*** compute gradient ***/

    const double* cur_param = relevant_distortion_prob.direct_access() + i*xDim;

    for (std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >::const_iterator it = count.begin();
         it != count.end(); it++) {
      
      const Math1D::Vector<uchar,uchar>& open_positions = it->first;
      const uchar size = open_positions.size();
      const uchar filled_pos = open_positions[size-1];
      const double weight = it->second;
      
      double sum = 0.0;
      for (uint k=0; k < size; k++)
        sum += cur_param[ open_positions[k] ]; 
      sum = std::max(sum,1e-15);
      
      distortion_grad[filled_pos] -= weight / std::max(cur_param[filled_pos],1e-15);
      
      const double addon = weight / sum;
      
      for (uint k=0; k < open_positions.size(); k++)
        distortion_grad[open_positions[k]] += addon;
    }
  
    /*** go in neg. gradient direction and reproject ***/

    Math1D::Vector<double> temp(xDim);
    
    for (uint j=0; j < xDim; j++)
      temp[j] = cur_param[j] - alpha * distortion_grad[j];
    
    projection_on_simplex(temp.direct_access(), xDim);
    
    for (uint j=0; j < xDim; j++)
      relevant_new_distort_prob(j,i) = temp[j];

    /*** find appropriate step-size ***/

    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      for (uint x=0; x < xDim; x++)
        relevant_hyp_distort_prob(x,i) = lambda * relevant_new_distort_prob(x,i) + neg_lambda * cur_param[x];

      double hyp_energy = nondeficient_m_step_energy(count, hyp_distortion_prob,i,J);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;
      
      if (nIter > 15 && lambda < 1e-12)
	break;
    }

    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nIter > 6)
      line_reduction_factor *= 0.9;

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint j=0; j < xDim; j++)
      relevant_distortion_prob(j,i) = best_lambda * relevant_new_distort_prob(j,i)
        + neg_best_lambda * relevant_distortion_prob(j,i);

    energy = best_energy;
  }

  for (uint j=0; j < xDim; j++) {
    assert(!isnan(relevant_distortion_prob(j,i)));
    assert(!isinf(relevant_distortion_prob(j,i)));           
    relevant_distortion_prob(j,i) = std::max(1e-8,relevant_distortion_prob(j,i));
  }

  return energy;
}

long double IBM3Trainer::alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const {

  SingleLookupTable aux_lookup;
  
  const SingleLookupTable& lookup = get_wordlookup(source_sentence_[s],target_sentence_[s],wcooc_,
                                                   nSourceWords_,slookup_[s],aux_lookup);

  return alignment_prob(source_sentence_[s],target_sentence_[s],lookup,alignment);
}

long double IBM3Trainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                        const SingleLookupTable& cur_lookup, const Math1D::Vector<AlignBaseType>& alignment) const {

  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Math2D::Matrix<double>& cur_distort_prob = distortion_prob_[curJ-1];

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));
  
  for (uint j=0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  }

  if (curJ < 2*fertility[0])
    return 0.0;

  for (uint i=1; i <= curI; i++) {
    uint t_idx = target[i-1];
    prob *= fertility_prob_[t_idx][fertility[i]];
    prob *= ld_fac_[fertility[i]];
  }

  for (uint j=0; j < curJ; j++) {
    
    uint s_idx = source[j];
    uint aj = alignment[j];
    
    if (aj == 0)
      prob *= dict_[0][s_idx-1];
    else {
      uint t_idx = target[aj-1];
      prob *= dict_[t_idx][cur_lookup(j,aj-1)] * cur_distort_prob(j,aj-1);
    }
  }

  //handle empty word
  assert(fertility[0] <= 2*curJ);
  
  prob *= ldchoose(curJ-fertility[0],fertility[0]);
  for (uint k=1; k <= fertility[0]; k++)
    prob *= p_zero_;
  for (uint k=1; k <= curJ-2*fertility[0]; k++)
    prob *= p_nonzero_;

  if (och_ney_empty_word_) {

    for (uint k=1; k<= fertility[0]; k++)
      prob *= ((long double) k) / curJ;
  }

  return prob;
}

long double IBM3Trainer::nondeficient_alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const {

  SingleLookupTable aux_lookup;
  
  const SingleLookupTable& lookup = get_wordlookup(source_sentence_[s],target_sentence_[s],wcooc_,
                                                   nSourceWords_,slookup_[s],aux_lookup);

  return nondeficient_alignment_prob(source_sentence_[s],target_sentence_[s],lookup,alignment);
}

long double IBM3Trainer::nondeficient_alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                                     const SingleLookupTable& cur_lookup, 
                                                     const Math1D::Vector<AlignBaseType>& alignment) const {


  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));

  Storage1D<std::vector<AlignBaseType> > aligned_source_words(curI+1); //words are listed in ascending order
  
  for (uint j=0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    aligned_source_words[aj].push_back(j);
  }
  
  if (curJ < 2*fertility[0])
    return 0.0;

  for (uint i=1; i <= curI; i++) {
    uint t_idx = target[i-1];
    //NOTE: no factorial here 
    prob *= fertility_prob_[t_idx][fertility[i]];
  }
  for (uint j=0; j < curJ; j++) {
    
    uint s_idx = source[j];
    uint aj = alignment[j];
    
    if (aj == 0)
      prob *= dict_[0][s_idx-1];
    else {
      uint t_idx = target[aj-1];
      prob *= dict_[t_idx][cur_lookup(j,aj-1)]; 
    }
  }

  prob *= nondeficient_distortion_prob(source,target,aligned_source_words);

  //handle empty word
  assert(fertility[0] <= 2*curJ);
  
  prob *= ldchoose(curJ-fertility[0],fertility[0]);
  for (uint k=1; k <= fertility[0]; k++)
    prob *= p_zero_;
  for (uint k=1; k <= curJ-2*fertility[0]; k++)
    prob *= p_nonzero_;

  return prob;  
}

long double IBM3Trainer::nondeficient_distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                                      const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) const {


  const uint curJ = source.size();
  const uint curI = target.size();

  const Math2D::Matrix<double>& cur_distort_prob = distortion_prob_[curJ-1];

  long double prob = 1.0;

  Storage1D<bool> filled(curJ,false);

  //std::cerr << "################# I=" << curI << ", J=" << curJ << std::endl;

  for (uint i=1; i <= curI; i++) {

    if (aligned_source_words[i].size() == 0)
      continue;

    const double* cur_param = cur_distort_prob.direct_access() + (i-1) * cur_distort_prob.xDim();

    /*** a) head word ***/
    uint first = aligned_source_words[i][0];

    double sum = 0.0;

    for (uint jj=0; jj < curJ; jj++) {
      if (!filled[jj]) {
        sum += cur_param[jj];
      } 
    }

    uint nToRemove = aligned_source_words[i].size()-1;

    if (nToRemove > 0) {
      //remove the aligned_source_words[i].size()-1 last terms as otherwise the given fertility cannot be fulfilled
      uint nRemoved = 0;

      for (int jj = curJ-1; jj >= 0; jj--) {

        if (!filled[jj]) {
          sum -= cur_param[jj]; 
          nRemoved++;
          if (nRemoved == nToRemove)
            break;
        }
      }

      assert(nRemoved == nToRemove);
    }
    
    assert(sum >= cur_param[first]-1e-4);
    
    prob *= cur_param[first] / sum;

    filled[first] = true;

    /*** b) remaining words ***/
    for (uint k=1; k < aligned_source_words[i].size(); k++) {

      nToRemove--;
      uint j = aligned_source_words[i][k];

      double sum = 0.0;
      for (uint jj=aligned_source_words[i][k-1]+1; jj < curJ; jj++) {
        if (!filled[jj]) {
          sum += cur_param[jj];
        } 
      }

      if (nToRemove > 0) {
        //remove the aligned_source_words[i].size()-1 last terms as otherwise the given fertility cannot be fulfilled
        uint nRemoved = 0;
        for (int jj = curJ-1; jj >= 0; jj--) {
          
          if (!filled[jj]) {
            sum -= cur_param[jj]; 
            nRemoved++;
            if (nRemoved == nToRemove)
              break;
          }
        }

        assert(nRemoved == nToRemove);
      }
      
      prob *= cur_param[j] / sum;

      filled[j] = true;
    }
  }

  return prob;
}

long double IBM3Trainer::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                                          const SingleLookupTable& lookup,
                                                          uint& nIter, Math1D::Vector<uint>& fertility,
                                                          Math2D::Matrix<long double>& expansion_prob,
                                                          Math2D::Matrix<long double>& swap_prob, 
                                                          Math1D::Vector<AlignBaseType>& alignment) {

  if (nondeficient_) {

    return nondeficient_hillclimbing(source,target,lookup,nIter,fertility,expansion_prob,swap_prob,alignment);
  }

  double improvement_factor = 1.001;
  
  const uint curI = target.size();
  const uint curJ = source.size();

  const Math2D::Matrix<double>& cur_distort_prob = distortion_prob_[curJ-1];
  
  assert(cur_distort_prob.size() > 0);
  
  /**** calculate probability of so far best known alignment *****/
  long double base_prob = 1.0;
  
  fertility.set_constant(0);

  for (uint j=0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  
    //handle lexicon prob and distortion prob for aj != 0
    if (aj == 0) {
      base_prob *= dict_[0][source[j]-1];
      assert(dict_[0][source[j]-1] > 0.0);
    }
    else {
      base_prob *= cur_distort_prob(j,aj-1);
      base_prob *= dict_[target[aj-1]][lookup(j,aj-1)];

      assert(cur_distort_prob(j,aj-1) > 0.0);
      assert(dict_[target[aj-1]][lookup(j,aj-1)] > 0.0);
    }
  }

  assert(2*fertility[0] <= curJ);
  assert(!isnan(base_prob));

  //handle fertility prob
  for (uint i=0; i < curI; i++) {

    const uint fert = fertility[i+1];
    const uint t_idx = target[i];

    if (!(fertility_prob_[t_idx][fert] > 0.0)) {

      std::cerr << "fert_prob[" << t_idx << "][" << fert << "]: " << fertility_prob_[t_idx][fert] << std::endl;
      //std::cerr << "target sentence #" << s << ", word #" << i << std::endl;
      std::cerr << "alignment: " << alignment << std::endl;
    }

    assert(fertility_prob_[t_idx][fert] > 0.0);

    base_prob *= fertility_prob_[t_idx][fert];
    base_prob *= ld_fac_[fert]; 
  }

  assert(base_prob > 0.0);
  assert(!isnan(base_prob));

  //handle fertility of empty word
  uint zero_fert = fertility[0];
  if (curJ < 2*zero_fert) {
    std::cerr << "WARNING: alignment startpoint for HC violates the assumption that less words "
              << " are aligned to NULL than to a real word" << std::endl;
  }
  else {

    assert(zero_fert <= 15);
    base_prob *= ldchoose(curJ-zero_fert,zero_fert);
    for (uint k=1; k <= zero_fert; k++)
      base_prob *= p_zero_;
    for (uint k=1; k <= curJ-2*zero_fert; k++) {
      base_prob *= p_nonzero_;
    }

    if (och_ney_empty_word_) {

      for (uint k=1; k<= fertility[0]; k++)
        base_prob *= ((long double) k) / curJ;
    }
  }

  long double check = alignment_prob(source,target,lookup,alignment);

  long double check_ratio = base_prob / check;

  if (!(check_ratio > 0.999 && check_ratio < 1.001)) {
    std::cerr << "alignment: " << alignment << std::endl;
    std::cerr << "fertility: " << fertility << std::endl;
    std::cerr << "base_prob: " << base_prob << std::endl;
    std::cerr << "check: " << check << std::endl;
  
    std::cerr << "ratio: " << check_ratio << std::endl;
  }

  if (base_prob > 1e-300 || check > 1e-300)
    assert(check_ratio > 0.999 && check_ratio < 1.001);

  assert(!isnan(base_prob));


  swap_prob.resize(curJ,curJ);
  expansion_prob.resize(curJ,curI+1);
  //swap_prob.set_constant(0.0);
  //expansion_prob.set_constant(0.0);

  uint count_iter = 0;

  while (true) {    

    count_iter++;
    nIter++;

    if (count_iter > 50)
      break;

    //std::cerr << "****************** starting new hillclimb iteration, current best prob: " << base_prob << std::endl;

    bool improvement = false;

    long double best_prob = base_prob;
    bool best_change_is_move = false;
    uint best_move_j = MAX_UINT;
    uint best_move_aj = MAX_UINT;
    uint best_swap_j1 = MAX_UINT;
    uint best_swap_j2 = MAX_UINT;

    /**** scan neighboring alignments and keep track of the best one that is better 
     ****  than the current alignment  ****/

    Math1D::Vector<long double> fert_increase_factor(curI+1);
    Math1D::Vector<long double> fert_decrease_factor(curI+1);

    for (uint i=1; i <= curI; i++) {

      uint t_idx = target[i-1];
      uint cur_fert = fertility[i];

      assert(fertility_prob_[t_idx][cur_fert] > 0.0);

      if (cur_fert > 0) {
        fert_decrease_factor[i] = ((long double) fertility_prob_[t_idx][cur_fert-1]) 
          / fertility_prob_[t_idx][cur_fert];

	fert_decrease_factor[i] /= cur_fert;
      }
      else
        fert_decrease_factor[i] = 0.0;

      if (cur_fert+1 < fertility_prob_[t_idx].size() && cur_fert+1 <= fertility_limit_) {
        fert_increase_factor[i] = ((long double) fertility_prob_[t_idx][cur_fert+1])
          / fertility_prob_[t_idx][cur_fert];

	fert_increase_factor[i] *= cur_fert+1;
      }
      else
        fert_increase_factor[i] = 0.0;
    }
    
    //a) expansion moves

    long double empty_word_increase_const = 0.0;
    if (curJ >= 2*(zero_fert+1)) {
      empty_word_increase_const = ldchoose(curJ-zero_fert-1,zero_fert+1) * p_zero_ 
        / (ldchoose(curJ-zero_fert,zero_fert) * p_nonzero_ * p_nonzero_);

      if (och_ney_empty_word_) {
        empty_word_increase_const *= (zero_fert+1) / ((long double) curJ);
      }
    }
    else {
      if (curJ > 3) {
        std::cerr << "WARNING: reached limit of allowed number of zero-aligned words, " 
                  << "J=" << curJ << ", zero_fert =" << zero_fert << std::endl;
      }
    }

    long double empty_word_decrease_const = 0.0;
    if (zero_fert > 0) {
      empty_word_decrease_const = ldchoose(curJ-zero_fert+1,zero_fert-1) * p_nonzero_ * p_nonzero_ 
        / (ldchoose(curJ-zero_fert,zero_fert) * p_zero_);

      if (och_ney_empty_word_) {
        empty_word_decrease_const *= curJ / ((long double) zero_fert);
      }
    }

    for (uint j=0; j < curJ; j++) {

      const uint s_idx = source[j];

      const uint aj = alignment[j];
      assert(fertility[aj] > 0);
      expansion_prob(j,aj) = 0.0;
      
      long double mod_base_prob = base_prob;

      if (aj > 0) {
        const uint t_idx = target[aj-1];

        if (dict_[t_idx][lookup(j,aj-1)] * cur_distort_prob(j,aj-1) > 1e-305) {
          mod_base_prob *= fert_decrease_factor[aj] / dict_[t_idx][lookup(j,aj-1)];	
          mod_base_prob /= cur_distort_prob(j,aj-1);
        }
        else
          mod_base_prob = 0.0;
      }
      else {
        if (dict_[0][s_idx-1] > 1e-305)
          mod_base_prob *= empty_word_decrease_const / dict_[0][s_idx-1];
        else
          mod_base_prob = 0.0;
      }

      if (isnan(mod_base_prob)) {

        std::cerr << "base prob: " << base_prob << std::endl;
        std::cerr << "aj: " << aj << std::endl;

        if (aj > 0) {

          const uint t_idx = target[aj-1];

          std::cerr << " mult by " << fert_decrease_factor[aj] << " / " << dict_[t_idx][lookup(j,aj-1)] << std::endl;
          std::cerr << " div by " << cur_distort_prob(j,aj-1) << std::endl;

          uint cur_fert = fertility[aj];

          std::cerr << "fert dec. factor = " << ((long double) fertility_prob_[t_idx][cur_fert-1])  << " / ("
                    << cur_fert  << " * " << fertility_prob_[t_idx][cur_fert] << ")" << std::endl;
        }
      }

      assert(!isnan(mod_base_prob));
      
      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        // 	std::cerr << "examining move " << j << " -> " << cand_aj << " (instead of " << aj << ") in iteration #" 
        // 		  << count_iter << std::endl;
        // 	std::cerr << "cand_aj has then fertility " << (fertility[cand_aj]+1) << std::endl;
        // 	std::cerr << "current aj reduces its fertility from " << fertility[aj] << " to " << (fertility[aj]-1)
        // 		  << std::endl;

        // 	if (cand_aj != 0 && cand_aj != aj) {
        // 	  std::cerr << "previous fert prob of candidate: " 
        // 		    << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]] << std::endl;
        // 	  std::cerr << "new fert prob of candidate: " 
        // 		    << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]+1] << std::endl;
        // 	}
        // 	if (aj != 0) {
        // 	  std::cerr << "previous fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]] << std::endl;
        // 	  std::cerr << "new fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]-1] << std::endl;
        // 	}

        // 	if (aj != 0) 
        // 	  std::cerr << "prev distort prob: " << cur_distort_prob(j,aj-1) << std::endl;
        // 	if (cand_aj != 0)
        // 	  std::cerr << "new distort prob: " << cur_distort_prob(j,cand_aj-1) << std::endl;
	
        if (cand_aj != aj) {
	  
          long double hyp_prob = mod_base_prob;


          if (cand_aj != 0) {

	    if ((fertility[cand_aj]+1) > fertility_limit_) {
	      expansion_prob(j,cand_aj) = 0.0;
	      continue;
	    }

	    const uint t_idx = target[cand_aj-1];
	    
	    hyp_prob *= dict_[t_idx][lookup(j,cand_aj-1)];
	    hyp_prob *= fert_increase_factor[cand_aj]; //will be 0 if the fertility limit is already reached
	    hyp_prob *= cur_distort_prob(j,cand_aj-1);
          }
          else {
	    if (2*fertility[0]+2 > curJ) {
	      expansion_prob(j,cand_aj) = 0.0;
	      continue;
	    } 

            hyp_prob *= empty_word_increase_const * dict_[0][s_idx-1];
          }

          if (isnan(hyp_prob)) {
            INTERNAL_ERROR << " nan in move " << j << " -> " << cand_aj << " . Exiting." << std::endl;

            std::cerr << "mod_base_prob: " << mod_base_prob << std::endl;
            if (cand_aj != 0) {

              const uint t_idx = target[cand_aj-1];

              std::cerr << "dict-factor: " << dict_[t_idx][lookup(j,cand_aj-1)] << std::endl;
              std::cerr << "fert-factor: " << fert_increase_factor[cand_aj] << std::endl;
              std::cerr << "distort-factor: " << cur_distort_prob(j,cand_aj-1) << std::endl;

              std::cerr << "distort table: " << cur_distort_prob << std::endl;
            }
            exit(1);
          }

          assert(!isnan(hyp_prob));

          //DEBUG
          // 	  Math1D::Vector<uint> cand_alignment = alignment[s];
          // 	  cand_alignment[j] = cand_aj;
	
          // 	  if (hyp_prob > 0.0) {
          // 	    double check_ratio = hyp_prob / alignment_prob(s,cand_alignment);
          // 	    if (hyp_prob > best_prob* 0.0000001) {

          // 	      if (! (check_ratio > 0.999 && check_ratio < 1.001 ) ) {
          // 		std::cerr << "check for sentence pair #" << s << " with " << curJ << " source words and "
          // 			  << curI << " target words" << std::endl;

          // 		std::cerr << "j: " << j << ", aj: " << aj << std::endl;
		
          // 		std::cerr << "examining move " << j << " -> " << cand_aj << " (instead of " << aj << ") in iteration #" 
          // 			  << count_iter << std::endl;
          // 		std::cerr << "cand_aj has then fertility " << (fertility[cand_aj]+1) << std::endl;
          // 		std::cerr << "current aj reduces its fertility from " << fertility[aj] << " to " << (fertility[aj]-1)
          // 			  << std::endl;
		
          // 		if (cand_aj != 0 && cand_aj != aj) {
          // 		  std::cerr << "previous fert prob of candidate: " 
          // 			    << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]] << std::endl;
          // 		  std::cerr << "new fert prob of candidate: " 
          // 			    << fertility_prob_[target[cand_aj-1]][fertility[cand_aj]+1] << std::endl;
          // 		}
          // 		if (aj != 0) {
          // 		  std::cerr << "previous fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]] 
          // 			    << std::endl;
          // 		  std::cerr << "new fert. prob of aj: " << fertility_prob_[target[aj-1]][fertility[aj]-1] 
          // 			    << std::endl;
          // 		}

          // 		if (aj != 0) 
          // 		  std::cerr << "prev distort prob: " << cur_distort_prob(j,aj-1) << std::endl;
          // 		if (cand_aj != 0)
          // 		  std::cerr << "new distort prob: " << cur_distort_prob(j,cand_aj-1) << std::endl;

          // 		std::cerr << "hyp prob: " << hyp_prob << std::endl;
          // 		std::cerr << "check: " << alignment_prob(s,cand_alignment) << std::endl;
          // 		std::cerr << "best prob: " << best_prob << std::endl;
          // 	      }
	      
          // 	      assert(check_ratio > 0.999 && check_ratio < 1.001);
          // 	    }
          // 	    else if (check_ratio < 0.999 || check_ratio > 1.001) {
          // #if 0
          // 	      std::cerr << "WARNING: ratio " << check_ratio << " for very unlikely alignment" << std::endl; 
          // 	      std::cerr << "   hyp prob: " << hyp_prob << std::endl;
          // 	      std::cerr << "   check: " << alignment_prob(s,cand_alignment) << std::endl;
          // 	      std::cerr << "   best prob: " << best_prob << std::endl;
          // #endif
          // 	    }
          // 	  }
          //END_DEBUG

          assert(!isnan(hyp_prob));

          expansion_prob(j,cand_aj) = hyp_prob;
	  
          if (hyp_prob > improvement_factor*best_prob) {

            best_prob = hyp_prob;
            improvement = true;
            best_change_is_move = true;
            best_move_j = j;
            best_move_aj = cand_aj;
          }	  
        }
      }
    }

    //b) swap_moves (NOTE that swaps do not affect the fertilities)
    for (uint j1=0; j1 < curJ; j1++) {
      
      const uint aj1 = alignment[j1];
      const uint s_j1 = source[j1];

      for (uint j2 = j1+1; j2 < curJ; j2++) {

        const uint aj2 = alignment[j2];
        const uint s_j2 = source[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1,j2) = 0.0;
        }
        else {

          long double hyp_prob = base_prob;
          if (aj1 != 0) {
            const uint t_idx = target[aj1-1];
            hyp_prob *= dict_[t_idx][lookup(j2,aj1-1)] * cur_distort_prob(j2,aj1-1)
              / (dict_[t_idx][lookup(j1,aj1-1)] * cur_distort_prob(j1,aj1-1) ) ;
          }
          else
            hyp_prob *= dict_[0][s_j2-1] / dict_[0][s_j1-1];

          if (aj2 != 0) {
            const uint t_idx = target[aj2-1];
            hyp_prob *= dict_[t_idx][lookup(j1,aj2-1)] * cur_distort_prob(j1,aj2-1)
              / (dict_[t_idx][lookup(j2,aj2-1)] * cur_distort_prob(j2,aj2-1) );
          }
          else {
            hyp_prob *= dict_[0][s_j1-1] / dict_[0][s_j2-1];
          }

          //DEBUG
          // 	  Math1D::Vector<uint> cand_alignment = alignment;
          // 	  cand_alignment[j1] = aj2;
          // 	  cand_alignment[j2] = aj1;
	  
          // 	  double check_ratio = hyp_prob / alignment_prob(s,cand_alignment);
          // 	  if (hyp_prob > best_prob* 0.0000001) 
          // 	    assert(check_ratio > 0.999 && check_ratio < 1.001);
          //END_DEBUG

          assert(!isnan(hyp_prob));

          swap_prob(j1,j2) = hyp_prob;

          if (hyp_prob > improvement_factor*best_prob) {
	    
            improvement = true;
            best_change_is_move = false;
            best_prob = hyp_prob;
            best_swap_j1 = j1;
            best_swap_j2 = j2;
          }
        }
      }
    }

    if (!improvement)
      break;

    //update alignment
    if (best_change_is_move) {
      uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      //std::cerr << "moving source pos" << best_move_j << " from " << cur_aj << " to " << best_move_aj << std::endl;

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;
      zero_fert = fertility[0];
    }
    else {
      //std::cerr << "swapping: j1=" << best_swap_j1 << std::endl;
      //std::cerr << "swapping: j2=" << best_swap_j2 << std::endl;

      uint cur_aj1 = alignment[best_swap_j1];
      uint cur_aj2 = alignment[best_swap_j2];

      assert(cur_aj1 != cur_aj2);
      
      alignment[best_swap_j1] = cur_aj2;
      alignment[best_swap_j2] = cur_aj1;
    }

#ifndef NDEBUG
    long double check_ratio = best_prob / alignment_prob(source,target,lookup,alignment);
    if (best_prob > 1e-300 && !(check_ratio > 0.995 && check_ratio < 1.005)) {

      std::cerr << "hc iter " << nIter << std::endl;

      if (best_change_is_move) {
        std::cerr << "moved j=" << best_move_j << " -> aj=" << best_move_aj << std::endl; 
      }
      else {
        std::cerr << "swapped j1=" << best_swap_j1 << " and j2=" << best_swap_j2 << std::endl;
        std::cerr << "now aligned to " << alignment[best_swap_j1] << " and " << alignment[best_swap_j2] << std::endl;
      }

      std::cerr << "probability improved from " << base_prob << " to " << best_prob << std::endl;
      std::cerr << "check_ratio: " << check_ratio << std::endl;
    }
    
    if (best_prob > 1e-275)
      assert(check_ratio > 0.995 && check_ratio < 1.005);
#endif

    base_prob = best_prob;
  }

  assert(!isnan(base_prob));

  assert(2*fertility[0] <= curJ);


  //symmetrize swap_prob
  for (uint j1=0; j1 < curJ; j1++) {

    swap_prob(j1,j1) = 0.0;
    
    for (uint j2 = j1+1; j2 < curJ; j2++) {
      
      swap_prob(j2,j1) = swap_prob(j1,j2);
    }
  }

  return base_prob;
}

long double IBM3Trainer::nondeficient_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                                   const SingleLookupTable& lookup, 
                                                   uint& nIter, Math1D::Vector<uint>& fertility,
                                                   Math2D::Matrix<long double>& expansion_prob,
                                                   Math2D::Matrix<long double>& swap_prob, 
                                                   Math1D::Vector<AlignBaseType>& alignment) {


  /**** calculate probability of the passed alignment *****/

  double improvement_factor = 1.001;
  
  const uint curI = target.size();
  const uint curJ = source.size();

  Storage1D<std::vector<AlignBaseType> > aligned_source_words(curI+1);

  fertility.set_constant(0);

  for (uint j=0; j < curJ; j++) {

    const uint aj = alignment[j];

    aligned_source_words[aj].push_back(j);

    fertility[aj]++;
  }

  long double base_distortion_prob = nondeficient_distortion_prob(source,target,aligned_source_words);
  long double base_prob = base_distortion_prob;

  for (uint i=1; i <= curI; i++) {
    uint t_idx = target[i-1];
    //NOTE: no factorial here 
    base_prob *= fertility_prob_[t_idx][fertility[i]];
  }
  for (uint j=0; j < curJ; j++) {
    
    uint s_idx = source[j];
    uint aj = alignment[j];
    
    if (aj == 0)
      base_prob *= dict_[0][s_idx-1];
    else {
      uint t_idx = target[aj-1];
      base_prob *= dict_[t_idx][lookup(j,aj-1)]; 
    }
  }

  base_prob *= ldchoose(curJ-fertility[0],fertility[0]);
  for (uint k=1; k <= fertility[0]; k++)
    base_prob *= p_zero_;
  for (uint k=1; k <= curJ-2*fertility[0]; k++)
    base_prob *= p_nonzero_;

  //DEBUG
  // double check_prob = nondeficient_alignment_prob(source,target,lookup,alignment);
  // double check_ratio = base_prob / check_prob;
  // assert(check_ratio >= 0.99 && check_ratio <= 1.01);
  //END_DEBUG


  uint count_iter = 0;

  //Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
  Storage1D<std::vector<AlignBaseType> > hyp_aligned_source_words = aligned_source_words;

  swap_prob.resize(curJ,curJ);
  expansion_prob.resize(curJ,curI+1);
  //swap_prob.set_constant(0.0);
  //expansion_prob.set_constant(0.0);

  while (true) {    

    count_iter++;
    nIter++;

    if (count_iter > 50)
      break;

    //std::cerr << "****************** starting new nondef hc iteration, current best prob: " << base_prob << std::endl;

    const uint zero_fert = fertility[0];

    long double empty_word_increase_const = 0.0;
    if (curJ >= 2*(zero_fert+1)) {
      empty_word_increase_const = ldchoose(curJ-zero_fert-1,zero_fert+1) * p_zero_ 
        / (ldchoose(curJ-zero_fert,zero_fert) * p_nonzero_ * p_nonzero_);
    }

    long double empty_word_decrease_const = 0.0;
    if (zero_fert > 0) {
      empty_word_decrease_const = ldchoose(curJ-zero_fert+1,zero_fert-1) * p_nonzero_ * p_nonzero_ 
        / (ldchoose(curJ-zero_fert,zero_fert) * p_zero_);
    }


    bool improvement = false;

    long double best_prob = base_prob;
    bool best_change_is_move = false;
    uint best_move_j = MAX_UINT;
    uint best_move_aj = MAX_UINT;
    uint best_swap_j1 = MAX_UINT;
    uint best_swap_j2 = MAX_UINT;

    /**** scan neighboring alignments and keep track of the best one that is better 
     ****  than the current alignment  ****/

    //Math1D::Vector<AlignBaseType> hyp_alignment = alignment;

    /**** expansion moves ****/
    
    for (uint j=0; j < curJ; j++) {

      const uint s_idx = source[j];

      const uint aj = alignment[j];
      expansion_prob(j,aj) = 0.0;

      const double old_dict_prob = (aj == 0) ? dict_[0][s_idx-1] : dict_[target[aj-1]][lookup(j,aj-1)];

      hyp_aligned_source_words[aj].erase(std::find(hyp_aligned_source_words[aj].begin(),hyp_aligned_source_words[aj].end(),j));

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        if (aj == cand_aj) {
          expansion_prob(j,cand_aj) = 0.0;
          continue;
        }
	if (cand_aj > 0) {
	  if ((fertility[cand_aj]+1) > fertility_limit_) { //better to check this before computing distortion probs
	    expansion_prob(j,cand_aj) = 0.0;
	    continue;
	  }
	}
	if (cand_aj == 0 && 2*fertility[0]+2 > curJ) { //better to check this before computing distortion probs
	  expansion_prob(j,cand_aj) = 0.0;
	  continue;
	}

        const double new_dict_prob = (cand_aj == 0) ? dict_[0][s_idx-1] : dict_[target[cand_aj-1]][lookup(j,cand_aj-1)];

        if (new_dict_prob < 1e-8)
          expansion_prob(j,cand_aj) = 0.0;
        else {
          hyp_aligned_source_words[cand_aj].push_back(j);
	  vec_sort(hyp_aligned_source_words[cand_aj]);

          long double leaving_prob = base_distortion_prob * old_dict_prob;
          long double incoming_prob = nondeficient_distortion_prob(source,target,hyp_aligned_source_words)
            * new_dict_prob;

          if (aj > 0) {
            uint tidx = target[aj-1];
            leaving_prob *= fertility_prob_[tidx][fertility[aj]];
            incoming_prob *= fertility_prob_[tidx][fertility[aj]-1];
          }
          else {
            
            //compute null-fert-model (null-fert decreases by 1)

            incoming_prob *= empty_word_decrease_const;
          }

          if (cand_aj > 0) {
            uint tidx = target[cand_aj-1];
            leaving_prob *= fertility_prob_[tidx][fertility[cand_aj]];
	    if (fertility[cand_aj]+1 <= fertility_limit_) 
	      incoming_prob *= fertility_prob_[tidx][fertility[cand_aj]+1]; 
	    else
	      incoming_prob = 0.0;
          }
          else {
            if (curJ < 2*fertility[0]+2)
              incoming_prob = 0.0;
            else {
              
              //compute null-fert-model (zero-fert goes up by 1)              
              incoming_prob *= empty_word_increase_const; //will be 0.0 if no more words can be aligned to the empty word
            }
          }

          long double incremental_cand_prob = base_prob * incoming_prob / leaving_prob;

          //DEBUG
	  //Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
          //hyp_alignment[j] = cand_aj;

          // long double cand_prob = nondeficient_alignment_prob(source,target,lookup,hyp_alignment);

          // long double ratio = incremental_cand_prob / cand_prob;

          // if (cand_prob > 1e-250) {
          //   if (! (ratio >= 0.99 && ratio <= 1.01)) {
          //     std::cerr << "j: " << j << ", aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
          //     std::cerr << "incremental: " << incremental_cand_prob << ", standalone: " << cand_prob << std::endl; 
          //   }
          //   assert(ratio >= 0.99 && ratio <= 1.01);
          // }
          //END_DEBUG

          expansion_prob(j,cand_aj) = incremental_cand_prob;

          if (incremental_cand_prob > improvement_factor * best_prob) {
            improvement = true;
            best_change_is_move = true;
            best_prob = incremental_cand_prob;
            best_move_j = j;
            best_move_aj = cand_aj;
          }

          //restore for the next iteration
          hyp_aligned_source_words[cand_aj] = aligned_source_words[cand_aj];
        }
      }

      hyp_aligned_source_words[aj] = aligned_source_words[aj];	  
    }

    /**** swap moves ****/
    for (uint j1=0; j1 < curJ; j1++) {

      const uint aj1 = alignment[j1];
      const uint s_j1 = source[j1];

      for (uint j2 = j1+1; j2 < curJ; j2++) {

        const uint aj2 = alignment[j2];
        const uint s_j2 = source[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1,j2) = 0.0;
        }
        else {
          
          for (uint k=0; k < hyp_aligned_source_words[aj2].size(); k++) {
            if (hyp_aligned_source_words[aj2][k] == j2) {
              hyp_aligned_source_words[aj2][k] = j1;
              break;
            }
          }
          for (uint k=0; k < hyp_aligned_source_words[aj1].size(); k++) {
            if (hyp_aligned_source_words[aj1][k] == j1) {
              hyp_aligned_source_words[aj1][k] = j2;
              break;
            }
          }

	  vec_sort(hyp_aligned_source_words[aj1]);
	  vec_sort(hyp_aligned_source_words[aj2]);          

          long double incremental_prob = base_prob / base_distortion_prob * 
            nondeficient_distortion_prob(source,target,hyp_aligned_source_words);

          if (aj1 != 0) {
            const uint t_idx = target[aj1-1];
            incremental_prob *= dict_[t_idx][lookup(j2,aj1-1)] 
              / dict_[t_idx][lookup(j1,aj1-1)] ;
          }
          else
            incremental_prob *= dict_[0][s_j2-1] / dict_[0][s_j1-1];

          if (aj2 != 0) {
            const uint t_idx = target[aj2-1];
            incremental_prob *= dict_[t_idx][lookup(j1,aj2-1)] 
              / dict_[t_idx][lookup(j2,aj2-1)] ;
          }
          else {
            incremental_prob *= dict_[0][s_j1-1] / dict_[0][s_j2-1];
          }

          //DEBUG
	  //Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
          //std::swap(hyp_alignment[j1],hyp_alignment[j2]);
          // long double cand_prob = nondeficient_alignment_prob(source,target,lookup,hyp_alignment);          

          // if (cand_prob > 1e-250) {

          //   double ratio = cand_prob / incremental_prob;
          //   assert(ratio > 0.99 && ratio < 1.01);
          // }
          //END_DEBUG

          
          swap_prob(j1,j2) = incremental_prob;

          if (incremental_prob > improvement_factor * best_prob) {
            improvement = true;
            best_change_is_move = false;
            best_prob = incremental_prob;
            best_swap_j1 = j1;
            best_swap_j2 = j2;
          }

          //restore for the next iteration
          //std::swap(hyp_alignment[j1],hyp_alignment[j2]);
          hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
          hyp_aligned_source_words[aj2] = aligned_source_words[aj2];
        }
      }
    }


    /**** update to best alignment ****/

    if (!improvement)
      break;

    //update alignment
    if (best_change_is_move) {
      uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      //std::cerr << "moving source pos" << best_move_j << " from " << cur_aj << " to " << best_move_aj << std::endl;

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;

      aligned_source_words[cur_aj].erase(std::find(aligned_source_words[cur_aj].begin(),aligned_source_words[cur_aj].end(),best_move_j));
      aligned_source_words[best_move_aj].push_back(best_move_j);
      vec_sort(aligned_source_words[best_move_aj]);

      hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
      hyp_aligned_source_words[best_move_aj] = aligned_source_words[best_move_aj];
    }
    else {
      //std::cerr << "swapping: j1=" << best_swap_j1 << std::endl;
      //std::cerr << "swapping: j2=" << best_swap_j2 << std::endl;

      uint cur_aj1 = alignment[best_swap_j1];
      uint cur_aj2 = alignment[best_swap_j2];

      assert(cur_aj1 != cur_aj2);
      
      alignment[best_swap_j1] = cur_aj2;
      alignment[best_swap_j2] = cur_aj1;

      for (uint k=0; k < aligned_source_words[cur_aj2].size(); k++) {
	if (aligned_source_words[cur_aj2][k] == best_swap_j2) {
	  aligned_source_words[cur_aj2][k] = best_swap_j1;
          break;
        }
      }
      for (uint k=0; k < aligned_source_words[cur_aj1].size(); k++) {
	if (aligned_source_words[cur_aj1][k] == best_swap_j1) {
	  aligned_source_words[cur_aj1][k] = best_swap_j2;
          break;
        }
      }

      vec_sort(aligned_source_words[cur_aj1]);
      vec_sort(aligned_source_words[cur_aj2]);

      hyp_aligned_source_words[cur_aj1] = aligned_source_words[cur_aj1];
      hyp_aligned_source_words[cur_aj2] = aligned_source_words[cur_aj2];
    }

    base_prob = best_prob;
    base_distortion_prob = nondeficient_distortion_prob(source,target,aligned_source_words);
  }


  //symmetrize swap_prob
  for (uint j1=0; j1 < curJ; j1++) {

    swap_prob(j1,j1) = 0.0;
    
    for (uint j2 = j1+1; j2 < curJ; j2++) {
      
      swap_prob(j2,j1) = swap_prob(j1,j2);
    }
  }


  return base_prob;
}


/* virtual */
void IBM3Trainer::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
					     const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment) {

  
  common_prepare_external_alignment(source,target,lookup,alignment);
  
  const uint J = source.size();
  const uint I = target.size();

  /*** check if respective distortion table is present. If not, create one from the parameters ***/

  if (parametric_distortion_) {
    if (distortion_param_.xDim() < J)
      distortion_param_.resize(J,distortion_param_.yDim(),1e-8);
    if (distortion_param_.yDim() < I)
      distortion_param_.resize(distortion_param_.xDim(),I,1e-8);
  }

  if (distortion_prob_.size() < J)
    distortion_prob_.resize(J);

  if (distortion_prob_[J-1].yDim() < I) {

    uint oldXDim = distortion_prob_[J-1].xDim();
    uint oldYDim = distortion_prob_[J-1].yDim();

    ReducedIBM3DistortionModel temp_prob(J,MAKENAME(temp_prob));
    temp_prob[J-1].resize(std::max<size_t>(J,distortion_prob_[J-1].xDim()), I, 1e-8 );

    if (parametric_distortion_)
      par2nonpar_distortion(temp_prob);

    distortion_prob_[J-1].resize(std::max(J,oldXDim),std::max(I,oldYDim));
 
    for (uint j=0; j < std::max(J,oldXDim); j++) {
      for (uint i=0; i  < std::max(I,oldYDim); i++) {

        if (j >= oldXDim || i >= oldYDim)
          distortion_prob_[J-1](j,i) = temp_prob[J-1](j,i);
      }
    }
  }
}

/*virtual */
long double IBM3Trainer::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
						    const SingleLookupTable& lookup,
						    Math1D::Vector<AlignBaseType>& alignment) {


  prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();
  const uint I = target.size();

  bool use_ilp = viterbi_ilp_;

#ifndef HAS_CBC
  use_ilp = false;
#endif

  //create matrices
  Math2D::Matrix<long double> expansion_prob(J,I+1);
  Math2D::Matrix<long double> swap_prob(J,J);

  Math1D::Vector<uint> fertility(I+1,0);
  
  uint nIter;

  long double hc_prob =  update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility,
							  expansion_prob, swap_prob, alignment);
  
  if (use_ilp) {
    
    return compute_viterbi_alignment_ilp(source, target, lookup, alignment);
  }
  else
    return hc_prob;
}


// <code> start_alignment </code> is used as initialization for hillclimbing and later modified
// the extracted alignment is written to <code> postdec_alignment </code>
/* virtual */
void IBM3Trainer::compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
						     const SingleLookupTable& lookup,
						     Math1D::Vector<AlignBaseType>& alignment,
						     std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
						     double threshold) {

  postdec_alignment.clear();

  prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();
  const uint I = target.size();

  //create matrices
  Math2D::Matrix<long double> expansion_move_prob(J,I+1);
  Math2D::Matrix<long double> swap_move_prob(J,J);

  Math1D::Vector<uint> fertility(I+1,0);
  
  uint nIter;
  
  long double best_prob;
  
  best_prob = update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility,
						expansion_move_prob, swap_move_prob, alignment);

  compute_postdec_alignment(alignment, best_prob, expansion_move_prob, swap_move_prob, threshold, postdec_alignment);
}


class CountStructure {
public:

  CountStructure() : main_count_(0.0) {}

  void unpack(uint i, uint J, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
	      std::map<Math1D::Vector<uchar,uchar>,double>& nondef_count) const;

  void unpack(uint i, uint J, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
	      std::map<Math1D::Vector<uchar,uchar>,Math1D::Vector<double,uchar> >& nondef_count) const;


  double main_count_;

  std::map<uchar,std::map<uchar,double> > exp_count_;
  std::map<uchar,std::map<uchar,double> > swap_count_;
};

void CountStructure::unpack(uint i, uint J, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
			    std::map<Math1D::Vector<uchar,uchar>,double>& nondef_count) const {

  
  Math1D::Vector<uchar> main_alignment(J,255);
  for (uint ii=0; ii < main_aligned_source_words.size(); ii++) {
    for (uint k=0; k < main_aligned_source_words[ii].size(); k++) {
      uint j = main_aligned_source_words[ii][k];
      main_alignment[j] = ii;
    }
  }
  
  const uint base_fert = main_aligned_source_words[i+1].size();

  Storage1D<std::vector<uchar> > hyp_aligned_source_words = main_aligned_source_words;
  
  if (base_fert >= 1) {
    //consider main alignment, most expansions and all swaps
    
    double main_count = main_count_; 

    //expansions
    for (std::map<uchar, std::map<uchar, double> >::const_iterator it = exp_count_.begin();
	 it != exp_count_.end(); it++) {

      uint j=it->first;
      uint cur_aj = main_alignment[j];
      
      hyp_aligned_source_words[cur_aj].erase(std::find(hyp_aligned_source_words[cur_aj].begin(),
						       hyp_aligned_source_words[cur_aj].end(),j));

      
      const std::map<uchar, double>& inner_exp = it->second;
      for (std::map<uchar, double>::const_iterator inner_it = inner_exp.begin(); inner_it != inner_exp.end(); inner_it++) {
	uint new_aj = inner_it->first;
	double count = inner_it->second;
	
	bool not_null = (cur_aj != 0 && new_aj != 0);

	if (not_null && ((cur_aj-1 > i && new_aj-1 > i) || (cur_aj-1 < i && new_aj-1 < i)) )
	  main_count += count;
	else if (base_fert > 1 || cur_aj != i+1) {
	  
	  hyp_aligned_source_words[new_aj].push_back(j);
	  //NOTE: sorting is only needed if new_aj == i+1
	  if (new_aj == i+1)
	    vec_sort(hyp_aligned_source_words[new_aj]);
	  
	  //call routine to add counts for hyp_aligned_source_words
	  add_nondef_count(hyp_aligned_source_words, i, J, nondef_count, count); 
	  
	  //restore
	  hyp_aligned_source_words[new_aj] = main_aligned_source_words[new_aj];
	}
      }
      hyp_aligned_source_words[cur_aj] = main_aligned_source_words[cur_aj];
    }
    
    //swaps
    for (std::map<uchar, std::map<uchar, double> >::const_iterator it = swap_count_.begin();
	 it != swap_count_.end(); it++) {
      
      uint j1=it->first;
      uint cur_aj1 = main_alignment[j1];
      
      const std::map<uchar, double>& inner_swap = it->second;
      for (std::map<uchar, double>::const_iterator inner_it = inner_swap.begin(); inner_it != inner_swap.end(); inner_it++) {
	uint j2 = inner_it->first;
	uint cur_aj2 = main_alignment[j2];
	double count = inner_it->second;
        
	bool not_null = (cur_aj1 != 0 && cur_aj2 != 0);
	if (not_null && ( (cur_aj1 -1 > i && cur_aj2-1 > i) || (cur_aj1 -1 < i && cur_aj2-1 < i) )) {
	  main_count += count;
	}
	else {

	  hyp_aligned_source_words[cur_aj1].erase(std::find(hyp_aligned_source_words[cur_aj1].begin(),
							    hyp_aligned_source_words[cur_aj1].end(),j1));
	  hyp_aligned_source_words[cur_aj1].push_back(j2);
	  //NOTE: sorting is only needed if cur_aj1==i+1
	  if (cur_aj1 == i+1)
	    vec_sort(hyp_aligned_source_words[cur_aj1]);
	  
	  hyp_aligned_source_words[cur_aj2].erase(std::find(hyp_aligned_source_words[cur_aj2].begin(),
							    hyp_aligned_source_words[cur_aj2].end(),j2));
	  hyp_aligned_source_words[cur_aj2].push_back(j1);
	  //NOTE: sorting is probably only needed if cur_aj2==i+1
	  if (cur_aj2 == i+1)
	    vec_sort(hyp_aligned_source_words[cur_aj2]);
	  
	  //call routine to add counts for hyp_aligned_source_words
	  add_nondef_count(hyp_aligned_source_words, i, J, nondef_count, count); 
	  
	  //restore
	  hyp_aligned_source_words[cur_aj1] = main_aligned_source_words[cur_aj1];
	  hyp_aligned_source_words[cur_aj2] = main_aligned_source_words[cur_aj2];
	}
      }
    }

    //now handle main alignment
    add_nondef_count(main_aligned_source_words, i, J, nondef_count, main_count); 
  }
  else {
    //consider only expansions to i+1 here

    //expansions
    for (std::map<uchar, std::map<uchar, double> >::const_iterator it = exp_count_.begin();
	 it != exp_count_.end(); it++) {
      
      uint j=it->first;
      uint cur_aj = main_alignment[j];
      assert(cur_aj != i+1);
      
      const std::map<uchar, double>& inner_exp = it->second;
      
      std::map<uchar, double>::const_iterator inner_it = inner_exp.find(i+1);
      if (inner_it != inner_exp.end()) {
	
	double count = inner_it->second;

	assert(hyp_aligned_source_words[i+1].empty());
		    
	hyp_aligned_source_words[cur_aj].erase(std::find(hyp_aligned_source_words[cur_aj].begin(),
							 hyp_aligned_source_words[cur_aj].end(),j));
	hyp_aligned_source_words[i+1].push_back(j); //no need to sort, the array was empty
	
	//call routine to add counts for hyp_aligned_source_words
	add_nondef_count(hyp_aligned_source_words, i, J, nondef_count, count); 
	
	//restore
	hyp_aligned_source_words[cur_aj] = main_aligned_source_words[cur_aj];
	hyp_aligned_source_words[i+1].clear();
      }
    }
  }

}

void CountStructure::unpack(uint i, uint J, const Storage1D<std::vector<uchar> >& main_aligned_source_words,
			    std::map<Math1D::Vector<uchar,uchar>,Math1D::Vector<double,uchar> >& nondef_count) const {


  Math1D::Vector<uchar> main_alignment(J,255);
  for (uint ii=0; ii < main_aligned_source_words.size(); ii++) {
    for (uint k=0; k < main_aligned_source_words[ii].size(); k++) {
      uint j = main_aligned_source_words[ii][k];
      main_alignment[j] = ii;
    }
  }
  
  const uint base_fert = main_aligned_source_words[i+1].size();

  Storage1D<std::vector<uchar> > hyp_aligned_source_words = main_aligned_source_words;
  
  if (base_fert >= 1) {
    //consider main alignment, most expansions and all swaps
    
    double main_count = main_count_; 

    //expansions
    for (std::map<uchar, std::map<uchar, double> >::const_iterator it = exp_count_.begin();
	 it != exp_count_.end(); it++) {

      uint j=it->first;
      uint cur_aj = main_alignment[j];
      
      hyp_aligned_source_words[cur_aj].erase(std::find(hyp_aligned_source_words[cur_aj].begin(),
						       hyp_aligned_source_words[cur_aj].end(),j));

      
      const std::map<uchar, double>& inner_exp = it->second;
      for (std::map<uchar, double>::const_iterator inner_it = inner_exp.begin(); inner_it != inner_exp.end(); inner_it++) {
	uint new_aj = inner_it->first;
	double count = inner_it->second;
	
	bool not_null = (cur_aj != 0 && new_aj != 0);

	if (not_null && ((cur_aj-1 > i && new_aj-1 > i) || (cur_aj-1 < i && new_aj-1 < i)) )
	  main_count += count;
	else if (base_fert > 1 || cur_aj != i+1) {
	  
	  hyp_aligned_source_words[new_aj].push_back(j);
	  //NOTE: sorting is only needed if new_aj == i+1
	  if (new_aj == i+1)
	    vec_sort(hyp_aligned_source_words[new_aj]);
	  
	  //call routine to add counts for hyp_aligned_source_words
	  add_nondef_count(hyp_aligned_source_words, i, J, nondef_count, count); 
	  
	  //restore
	  hyp_aligned_source_words[new_aj] = main_aligned_source_words[new_aj];
	}
      }
      hyp_aligned_source_words[cur_aj] = main_aligned_source_words[cur_aj];
    }
    
    //swaps
    for (std::map<uchar, std::map<uchar, double> >::const_iterator it = swap_count_.begin();
	 it != swap_count_.end(); it++) {
      
      uint j1=it->first;
      uint cur_aj1 = main_alignment[j1];
      
      const std::map<uchar, double>& inner_swap = it->second;
      for (std::map<uchar, double>::const_iterator inner_it = inner_swap.begin(); inner_it != inner_swap.end(); inner_it++) {
	uint j2 = inner_it->first;
	uint cur_aj2 = main_alignment[j2];
	double count = inner_it->second;
        
	bool not_null = (cur_aj1 != 0 && cur_aj2 != 0);
	if (not_null && ( (cur_aj1 -1 > i && cur_aj2-1 > i) || (cur_aj1 -1 < i && cur_aj2-1 < i) )) {
	  main_count += count;
	}
	else {

	  hyp_aligned_source_words[cur_aj1].erase(std::find(hyp_aligned_source_words[cur_aj1].begin(),
							    hyp_aligned_source_words[cur_aj1].end(),j1));
	  hyp_aligned_source_words[cur_aj1].push_back(j2);
	  //NOTE: sorting is only needed if cur_aj1==i+1
	  if (cur_aj1 == i+1)
	    vec_sort(hyp_aligned_source_words[cur_aj1]);
	  
	  hyp_aligned_source_words[cur_aj2].erase(std::find(hyp_aligned_source_words[cur_aj2].begin(),
							    hyp_aligned_source_words[cur_aj2].end(),j2));
	  hyp_aligned_source_words[cur_aj2].push_back(j1);
	  //NOTE: sorting is probably only needed if cur_aj2==i+1
	  if (cur_aj2 == i+1)
	    vec_sort(hyp_aligned_source_words[cur_aj2]);
	  
	  //call routine to add counts for hyp_aligned_source_words
	  add_nondef_count(hyp_aligned_source_words, i, J, nondef_count, count); 
	  
	  //restore
	  hyp_aligned_source_words[cur_aj1] = main_aligned_source_words[cur_aj1];
	  hyp_aligned_source_words[cur_aj2] = main_aligned_source_words[cur_aj2];
	}
      }
    }

    //now handle main alignment
    add_nondef_count(main_aligned_source_words, i, J, nondef_count, main_count); 
  }
  else {
    //consider only expansions to i+1 here

    //expansions
    for (std::map<uchar, std::map<uchar, double> >::const_iterator it = exp_count_.begin();
	 it != exp_count_.end(); it++) {
      
      uint j=it->first;
      uint cur_aj = main_alignment[j];
      assert(cur_aj != i+1);
      
      const std::map<uchar, double>& inner_exp = it->second;
      
      std::map<uchar, double>::const_iterator inner_it = inner_exp.find(i+1);
      if (inner_it != inner_exp.end()) {
	
	double count = inner_it->second;

	assert(hyp_aligned_source_words[i+1].empty());
		    
	hyp_aligned_source_words[cur_aj].erase(std::find(hyp_aligned_source_words[cur_aj].begin(),
							 hyp_aligned_source_words[cur_aj].end(),j));
	hyp_aligned_source_words[i+1].push_back(j); //no need to sort, the array was empty
	
	//call routine to add counts for hyp_aligned_source_words
	add_nondef_count(hyp_aligned_source_words, i, J, nondef_count, count); 
	
	//restore
	hyp_aligned_source_words[cur_aj] = main_aligned_source_words[cur_aj];
	hyp_aligned_source_words[i+1].clear();
      }
    }
  }
}


class CompactAlignedSourceWords {
public:

  CompactAlignedSourceWords(const Storage1D<std::vector<uchar> >& aligned_source_words) {

    start_.resize(aligned_source_words.size()-1);

    //we don't copy the empty word as we do not need it

    uint nPos = 0;
    for (uint i=1; i < aligned_source_words.size(); i++)
      nPos += aligned_source_words[i].size();

    pos_.resize(nPos);
    uint next_pos=0;

    for (uint i=1; i < aligned_source_words.size(); i++) {
      start_[i-1] = next_pos;
      for (uint k=0; k < aligned_source_words[i].size(); k++) {
        pos_[next_pos] = aligned_source_words[i][k];
        next_pos++;
      }
    }
    assert(next_pos == nPos);
  }

  void get_noncompact_form(Storage1D<std::vector<uchar> >& aligned_source_words, uint J) const {

    aligned_source_words.resize(size()+1);
    aligned_source_words[0].clear();

    Storage1D<bool> covered(J,false);
    for (uint i=0; i < size(); i++) {
      aligned_source_words[i+1].clear();
      for (uint k=start(i); k < end(i); k++) {
        uint j = pos_[k];
        aligned_source_words[i+1].push_back(j);
        covered[j] = true;
      }
    }

    for (uint j=0; j < J; j++) {
      if (!covered[j])
        aligned_source_words[0].push_back(j);
    }
  }

  uchar start(uchar i) const {
    return start_[i];
  }

  uchar end(uchar i) const {
    if (i+1 < start_.size())
      return start_[i+1];
    else
      return pos_.size();
  }

  uchar size() const {
    return start_.size();
  }

  Math1D::Vector<uchar,uchar> pos_;

  Math1D::Vector<uchar,uchar> start_;
};


//this operator only exists in order to make the class usable in std::map.
//Hence, it is optimized for speed, not for a sensible order
bool operator<(const CompactAlignedSourceWords& v1, const CompactAlignedSourceWords& v2) {

  if (v1.pos_.size() != v2.pos_.size())
    return (v1.pos_.size() < v2.pos_.size());

  if (v1.start_.size() != v2.start_.size())
    return (v1.start_.size() < v2.start_.size());

  for (uchar k=0; k < v1.pos_.size(); k++) {
    if (v1.pos_[k] != v2.pos_[k])
      return (v1.pos_[k] < v2.pos_[k]);
  }

  for (uchar k=0; k < v1.start_.size(); k++) {
    if (v1.start_[k] != v2.start_[k])
      return (v1.start_[k] < v2.start_[k]);
  }  

  return false;
}

void IBM3Trainer::train_unconstrained(uint nIter, HmmWrapper* wrapper) {

  std::cerr << "starting IBM-3 training without constraints" << std::endl;

  double max_perplexity = 0.0;
  double approx_sum_perplexity = 0.0;
  double viterbi_max_perplexity = 0.0;

  double max_ratio = 1.0;
  double min_ratio = 1.0;

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment;
  if (viterbi_ilp_)
    viterbi_alignment.resize(source_sentence_.size());

  double dict_weight_sum = 0.0;
  for (uint i=0; i < nTargetWords_; i++) {
    dict_weight_sum += fabs(prior_weight_[i].sum());
  }

  SingleLookupTable aux_lookup;

  if (parametric_distortion_)
    par2nonpar_distortion(distortion_prob_);

  ReducedIBM3DistortionModel fdistort_count(distortion_prob_.size(),MAKENAME(fdistort_count));
  for (uint J=0; J < fdistort_count.size(); J++) {
    fdistort_count[J].resize_dirty(distortion_prob_[J].xDim(), distortion_prob_[J].yDim());
  }

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords,MAKENAME(ffert_count));

  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  double fzero_count;
  double fnonzero_count;

  uint iter;
  for (iter=1+iter_offs_; iter <= nIter+iter_offs_; iter++) {

    std::cerr << "******* IBM-3 EM-iteration #" << iter << std::endl;

    uint nViterbiBetter = 0;
    uint nViterbiWorse = 0;
  
    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint J=0; J < fdistort_count.size(); J++) {
      fdistort_count[J].set_constant(0.0);
    }
    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }


    Storage1D<std::map<CompactAlignedSourceWords, CountStructure> > refined_nondef_aligned_words_count(maxJ_);


    max_perplexity = 0.0;
    approx_sum_perplexity = 0.0;

    std::clock_t tStartLoop = std::clock();

    for (size_t s=0; s < source_sentence_.size(); s++) {
      
      if ((s% 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;
      
      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
                                                           nSourceWords_,slookup_[s],aux_lookup);

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      Math2D::Matrix<double>& cur_distort_count = fdistort_count[curJ-1];

      Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ,curJ,MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ,curI+1,MAKENAME(expansion_move_prob));
      
      long double best_prob;

      if (wrapper != 0) {
	
	best_prob = simulate_hmm_hillclimbing(cur_source, cur_target, cur_lookup, *wrapper,
					      fertility, expansion_move_prob, swap_move_prob, best_known_alignment_[s]);
      }
      else {
	best_prob = update_alignment_by_hillclimbing(cur_source,cur_target,cur_lookup,sum_iter,fertility,
						     expansion_move_prob,swap_move_prob,best_known_alignment_[s]);
      }      

      assert(!isnan(best_prob));


      long double viterbi_prob = 0.0;
      if (viterbi_ilp_) {
        viterbi_alignment[s] = best_known_alignment_[s];
        viterbi_prob = compute_viterbi_alignment_ilp(cur_source,cur_target,cur_lookup,viterbi_alignment[s]);
        //OPTIONAL
        //best_known_alignment_[s] = viterbi_alignment[s];
        //best_prob = update_alignment_by_hillclimbing(s,sum_iter,fertility,
        // 						   expansion_move_prob,swap_move_prob);
        //END_OPTIONAL
        viterbi_max_perplexity -= std::log(viterbi_prob);

        bool alignments_equal = true;
        for (uint j=0; j < curJ; j++) {
	  
          if (best_known_alignment_[s][j] != viterbi_alignment[s][j])
            alignments_equal = false;
        }

        if (!alignments_equal) {
	  
          double ratio = viterbi_prob / best_prob;
	  
          if (ratio > 1.01) {
            nViterbiBetter++;
          }
          else if (ratio < 0.99) {
            nViterbiWorse++;
	    
            std::cerr << "pair #" << s << ": WORSE!!!!" << std::endl;
            std::cerr << "ilp prob:          " << viterbi_prob << std::endl;
            std::cerr << "ilp alignment: " << viterbi_alignment[s] << std::endl;
	    
            std::cerr << "hillclimbing prob: " << best_prob << std::endl;
            std::cerr << "hc. alignment: " << best_known_alignment_[s] << std::endl;
          }
          if (ratio > max_ratio) {
            max_ratio = ratio;
	    
            std::cerr << "pair #" << s << std::endl;
            std::cerr << "ilp prob:          " << viterbi_prob << std::endl;
            std::cerr << "ilp alignment: " << viterbi_alignment[s] << std::endl;
	    
            std::cerr << "hillclimbing prob: " << best_prob << std::endl;
            std::cerr << "hc. alignment: " << best_known_alignment_[s] << std::endl;
          }
          if (ratio < min_ratio) {
            min_ratio = ratio;
	
            std::cerr << "pair #" << s << std::endl;
            std::cerr << "ilp prob:          " << viterbi_prob << std::endl;
            std::cerr << "ilp alignment: " << viterbi_alignment[s] << std::endl;
	    
            std::cerr << "hillclimbing prob: " << best_prob << std::endl;
            std::cerr << "hc. alignment: " << best_known_alignment_[s] << std::endl;
          }
        }
      }

      max_perplexity -= std::log(best_prob);
      
      const long double expansion_prob = expansion_move_prob.sum();
      const long double swap_prob =  swap_mass(swap_move_prob); //0.5 * swap_move_prob.sum();

      const long double sentence_prob = best_prob + expansion_prob +  swap_prob;

      approx_sum_perplexity -= std::log(sentence_prob);

      Math2D::Matrix<double> j_marg;
      Math2D::Matrix<double> i_marg;

      const long double inv_sentence_prob = 1.0 / sentence_prob;
     
      if (isnan(inv_sentence_prob)) {

	std::cerr << "best prob: " << best_prob << std::endl;
	std::cerr << "swap prob: " << swap_prob << std::endl;
	std::cerr << "expansion prob: " << expansion_prob << std::endl;
      }

      assert(!isnan(inv_sentence_prob));

      update_zero_counts(best_known_alignment_[s], fertility,
			 expansion_move_prob, swap_prob, best_prob,
			 sentence_prob, inv_sentence_prob,
			 fzero_count, fnonzero_count);

      //increase counts for dictionary and distortion
      for (uint j=0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

	const uint cur_aj = best_known_alignment_[s][j];
	
	long double addon = sentence_prob;
	for (uint i=0; i <= curI; i++) 
	  addon -= expansion_move_prob(j,i);
	for (uint jj=0; jj < curJ; jj++)
	  addon -= swap_move_prob(j,jj);
	
	addon *= inv_sentence_prob;
	
	assert(!isnan(addon));
	
	if (cur_aj != 0) {
	  fwcount[cur_target[cur_aj-1]][cur_lookup(j,cur_aj-1)] += addon;
	  cur_distort_count(j,cur_aj-1) += addon;
	  assert(!isnan(cur_distort_count(j,cur_aj-1)));
	}
	else {
	  fwcount[0][s_idx-1] += addon;
	}

	for (uint i=0; i <= curI; i++) {

	  if (i != cur_aj) {
	    
	    long double addon = expansion_move_prob(j,i);
	    for (uint jj=0; jj < curJ; jj++) {
	      if (best_known_alignment_[s][jj] == i)
		addon += swap_move_prob(j,jj);
	    }
	    addon *= inv_sentence_prob;

	    assert(!isnan(addon));
	
	    if (i!=0) {
	      fwcount[cur_target[i-1]][cur_lookup(j,i-1)] += addon;
	      cur_distort_count(j,i-1) += addon;
	      assert(!isnan(cur_distort_count(j,i-1)));
	    }
	    else {
	      fwcount[0][s_idx-1] += addon;
	    }
          }
        }
      }

      if (nondeficient_) {

        //a) best known alignment (=mode)
        Storage1D<std::vector<uchar> > aligned_source_words(curI+1);
        for (uint j=0; j < curJ; j++)
          aligned_source_words[best_known_alignment_[s][j]].push_back(j);

        CountStructure& cur_refined_count = refined_nondef_aligned_words_count[curJ-1][CompactAlignedSourceWords(aligned_source_words)];

        const double mode_contrib = best_prob / sentence_prob;
	cur_refined_count.main_count_ += mode_contrib;
        
        Storage1D<std::vector<uchar> > hyp_aligned_source_words = aligned_source_words;

        //b) expansions
        for (uint jj=0; jj < curJ; jj++) {

          uint cur_aj = best_known_alignment_[s][jj];
	  
	  hyp_aligned_source_words[cur_aj].erase(std::find(hyp_aligned_source_words[cur_aj].begin(),
							   hyp_aligned_source_words[cur_aj].end(),jj));
          
          for (uint aj = 0; aj <= curI; aj++) {

            const double contrib = expansion_move_prob(jj,aj) / sentence_prob;
 
            if (contrib > 1e-6) {
              
              assert(aj != cur_aj);

              hyp_aligned_source_words[aj].push_back(jj);
	      vec_sort(hyp_aligned_source_words[aj]);

              cur_refined_count.exp_count_[jj][aj] += contrib;

              hyp_aligned_source_words[aj] = aligned_source_words[aj];
            }
          }

	  hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
        }

        //c) swaps
        for (uint j1=0; j1 < curJ-1; j1++) {

          uint aj1 = best_known_alignment_[s][j1];

          for (uint j2=j1+1; j2 < curJ; j2++) {

            const double contrib = swap_move_prob(j1,j2) / sentence_prob;

            if (contrib > 1e-6) {

              uint aj2 = best_known_alignment_[s][j2];

              hyp_aligned_source_words[aj1].erase(std::find(hyp_aligned_source_words[aj1].begin(),
                                                            hyp_aligned_source_words[aj1].end(),j1));
              hyp_aligned_source_words[aj1].push_back(j2);
	      vec_sort(hyp_aligned_source_words[aj1]);
              hyp_aligned_source_words[aj2].erase(std::find(hyp_aligned_source_words[aj2].begin(),
                                                            hyp_aligned_source_words[aj2].end(),j2));
              hyp_aligned_source_words[aj2].push_back(j1);
	      vec_sort(hyp_aligned_source_words[aj2]);

              cur_refined_count.swap_count_[j1][j2] += contrib;
              
              hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
              hyp_aligned_source_words[aj2] = aligned_source_words[aj2];
            }
          }
        }
      }

      //update fertility counts

      update_fertility_counts(cur_target, best_known_alignment_[s], fertility,
			      expansion_move_prob, sentence_prob, inv_sentence_prob, ffert_count);	
      
      assert(!isnan(fzero_count));
      assert(!isnan(fnonzero_count));

      for (uint i=0; i < denom_cache_.size(); i++)
        denom_cache_[i].clear(); 

    } //loop over sentences finished

    std::clock_t tEndLoop = std::clock();
    
    std::cerr << "loop over sentences took " << diff_seconds(tEndLoop,tStartLoop) << " seconds." << std::endl;

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(1e-8,fzero_count / fsum);
      p_nonzero_ = fnonzero_count / fsum;
    }

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    //DEBUG
    uint nZeroAlignments = 0;
    uint nAlignments = 0;
    for (size_t s=0; s < source_sentence_.size(); s++) {

      nAlignments += source_sentence_[s].size();

      for (uint j=0; j < source_sentence_[s].size(); j++) {
        if (best_known_alignment_[s][j] == 0)
          nZeroAlignments++;
      }
    }
    std::cerr << "percentage of zero-aligned words: " 
              << (((double) nZeroAlignments) / ((double) nAlignments)) << std::endl;
    //END_DEBUG

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, dict_weight_sum, iter, 
			    smoothed_l0_, l0_beta_, 45, dict_, 1e-6);


    //update distortion prob from counts
    if (parametric_distortion_) {

      assert(!nondeficient_ || estep_mode_ == FertEStepHillclimbing);
      
      Math2D::Matrix<double> fpar_distort_count(distortion_param_.xDim(),distortion_param_.yDim(),0.0);

      for (uint J=0; J < fdistort_count.size(); J++) {

        for (uint x=0; x < fdistort_count[J].xDim(); x++)
          for (uint y=0; y < fdistort_count[J].yDim(); y++)
            fpar_distort_count(x,y) += fdistort_count[J](x,y);
      }

      for (uint y=0; y < fpar_distort_count.yDim(); y++) {
        
        double sum =  0.0;
        for (uint x = 0; x < fpar_distort_count.xDim(); x++)
          sum += fpar_distort_count(x,y);

        if (sum > 1e-305) {
          for (uint x = 0; x < fpar_distort_count.xDim(); x++)
            fpar_distort_count(x,y) /= sum;
        }
      }

      if (nondeficient_) {

        for (uint i=0; i < maxI_; i++) {


          std::map<Math1D::Vector<uchar,uchar>,double> cur_nondef_count;
	  //std::map<Math1D::Vector<uchar,uchar>,Math1D::Vector<double,uchar> > cur_nondef_count;

          for (uint J=1; J <= refined_nondef_aligned_words_count.size(); J++) {

	    //unpack the compactly stored counts

            for (std::map<CompactAlignedSourceWords, CountStructure>::const_iterator it = refined_nondef_aligned_words_count[J-1].begin();
                 it != refined_nondef_aligned_words_count[J-1].end(); it++) {
              
              const CompactAlignedSourceWords& main_compact_aligned_source_words = it->first;

              if (main_compact_aligned_source_words.size() <= i)
                continue;

              Storage1D<std::vector<uchar> > main_aligned_source_words;
              main_compact_aligned_source_words.get_noncompact_form(main_aligned_source_words,J);


              const CountStructure& cur_count_struct = it->second;
	      cur_count_struct.unpack(i,J,main_aligned_source_words,cur_nondef_count);
            }
          }
     
          if (cur_nondef_count.empty())
            continue;

	  //copy map->vector for faster access
          std::vector<Math1D::Vector<uchar,uchar> > open_pos;
	  //std::vector<Math1D::Vector<double,uchar> > vec_count;
          std::vector<double> vec_nondef_count;
	  

          vec_nondef_count.reserve(cur_nondef_count.size());
          //vec_count.reserve(cur_nondef_count.size());
          open_pos.reserve(cur_nondef_count.size());

          for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = cur_nondef_count.begin(); 
               it != cur_nondef_count.end(); it++) {
            
            open_pos.push_back(it->first);
            vec_nondef_count.push_back(it->second);
          }
          // for (std::map<Math1D::Vector<uchar,uchar>,Math1D::Vector<double,uchar> >::const_iterator it = cur_nondef_count.begin(); 
          //      it != cur_nondef_count.end(); it++) {
            
          //   open_pos.push_back(it->first);
          //   vec_count.push_back(it->second);
	    
	  //   //truncate the counts: remove tail 0s
	  //   Math1D::Vector<double,uchar>& cur_count = vec_count.back();
	  //   int s;
	  //   for (s=cur_count.size()-1; s > 0; s--) {
	  //     if (cur_count[s] != 0.0)
	  // 	break;
	  //   }

	  //   if (s != cur_count.size()-1)
	  //     cur_count.resize(s+1);
          // }

          cur_nondef_count.clear();

          std::cerr << "calling  m-step for i=" << i << std::endl;

          double start_energy = nondeficient_m_step_energy(open_pos, vec_nondef_count, distortion_param_,i);

          double hyp_energy = nondeficient_m_step_energy(open_pos, vec_nondef_count, fpar_distort_count,i);

          //double start_energy = nondeficient_m_step_energy(open_pos, vec_count, distortion_param_,i);

          //double hyp_energy = nondeficient_m_step_energy(open_pos, vec_count, fpar_distort_count,i);

          if (hyp_energy < start_energy) {

            for (uint x = 0; x < fpar_distort_count.xDim(); x++) {
              distortion_param_(x,i) = fpar_distort_count(x,i);
            }

            std::cerr << "switching to normalized counts: " << start_energy << " -> " << hyp_energy << std::endl;

            start_energy = hyp_energy;
          }

          nondeficient_m_step_with_interpolation(open_pos,vec_nondef_count,i,start_energy);
          //nondeficient_m_step_with_interpolation(open_pos,vec_count,i,start_energy);
        }
      } 
      else {

        for (uint i=0; i < maxI_; i++) {
          std::cerr << "m-step for i=" << i << std::endl;

          double start_energy = par_distortion_m_step_energy(fdistort_count,distortion_param_,i);

          double hyp_energy = par_distortion_m_step_energy(fdistort_count,fpar_distort_count,i);

          if (hyp_energy < start_energy) {

            for (uint x = 0; x < fpar_distort_count.xDim(); x++) {
              distortion_param_(x,i) = fpar_distort_count(x,i);
            }

	    start_energy = hyp_energy;
          }

          par_distortion_m_step(fdistort_count,i,start_energy);
        }
        par2nonpar_distortion(distortion_prob_);
      }
    } 
    else {
      //nonparametric distortion model

      if (nondeficient_) {

        //NOTE: in the nonparametric setting we could skip entries where ALL positions are free. 
        //  Implementing this is TODO

        ReducedIBM3DistortionModel normalized_distort_count(MAKENAME(normalized_distort_count));
        normalized_distort_count = fdistort_count;
        for (uint J=0; J < fdistort_count.size(); J++) {

	  for (uint y=0; y < normalized_distort_count[J].yDim(); y++) {

	    double sum = 0.0;
	    for (uint x=0; x < normalized_distort_count[J].xDim(); x++) 
	      sum += normalized_distort_count[J](x,y);

	    if (sum > 1e-305) {
	      for (uint x=0; x < normalized_distort_count[J].xDim(); x++) 
		normalized_distort_count[J](x,y) = normalized_distort_count[J](x,y) / sum;
	    }
	    else {
	      for (uint x=0; x < normalized_distort_count[J].xDim(); x++) 
		normalized_distort_count[J](x,y) = 1.0 / normalized_distort_count[J].xDim();
	    }
	  }
        }


        for (uint i=0; i < maxI_; i++) {

          std::cerr << "calling nondeficient m-steps for i=" << i << std::endl;

          for (uint J=1; J <= refined_nondef_aligned_words_count.size(); J++) {

	    //unpack the compactly stored counts

            std::map<Math1D::Vector<uchar,uchar>,double> cur_nondef_count;

            for (std::map<CompactAlignedSourceWords, CountStructure>::const_iterator it = refined_nondef_aligned_words_count[J-1].begin();
                 it != refined_nondef_aligned_words_count[J-1].end(); it++) {
              
              const CompactAlignedSourceWords& main_compact_aligned_source_words = it->first;
	      
              if (main_compact_aligned_source_words.size() <= i)
                continue;

              const CountStructure& cur_count_struct = it->second;

	      
              Storage1D<std::vector<uchar> > main_aligned_source_words;
              main_compact_aligned_source_words.get_noncompact_form(main_aligned_source_words,J);


	      cur_count_struct.unpack(i,J,main_aligned_source_words,cur_nondef_count);
	    }
         
	    if (cur_nondef_count.empty())
	      continue;

	    //copy map->vector for faster access
            std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> > vec_nondef_count;	    
	    vec_nondef_count.reserve(cur_nondef_count.size());
	    for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = cur_nondef_count.begin(); 
		 it != cur_nondef_count.end(); it++) {
	      
	      vec_nondef_count.push_back(*it);
	    }
	    cur_nondef_count.clear();
	    
            double start_energy = nondeficient_m_step_energy(vec_nondef_count,distortion_prob_,i,J);

            double hyp_energy = nondeficient_m_step_energy(vec_nondef_count,normalized_distort_count,i,J);

            if (hyp_energy < start_energy) {

              start_energy = hyp_energy;

	      for (uint j=0; j < distortion_prob_[J-1].xDim(); j++)
		distortion_prob_[J-1](j,i) = normalized_distort_count[J-1](j,i);
            }

            nondeficient_m_step(vec_nondef_count,i,J,start_energy);
          }
        }
      }
      else {
	//deficient mode

        for (uint J=0; J < distortion_prob_.size(); J++) {
	
          for (uint i=0; i < distortion_prob_[J].yDim(); i++) {
	  
            double sum = 0.0;
            for (uint j=0; j < J+1; j++)
              sum += fdistort_count[J](j,i);
            
            if (sum > 1e-307) {
              const double inv_sum = 1.0 / sum;
              assert(!isnan(inv_sum));
              
              for (uint j=0; j < J+1; j++) {
                distortion_prob_[J](j,i) = std::max(1e-8,inv_sum * fdistort_count[J](j,i));
                if (isnan(distortion_prob_[J](j,i))) {
                  std::cerr << "sum: " << sum << std::endl;
                  std::cerr << "set to " << inv_sum << " * " << fdistort_count[J](j,i) << " = "
                            << (inv_sum * fdistort_count[J](j,i)) << std::endl;
                }
                assert(!isnan(fdistort_count[J](j,i)));
                assert(!isnan(distortion_prob_[J](j,i)));
              }
            }
            else {
              std::cerr << "WARNING: did not update distortion count because sum was " << sum << std::endl;
            }
          }
        }
      }
    }


    update_fertility_prob(ffert_count,1e-8);

    double reg_term = 0.0;
    for (uint i=0; i < dict_.size(); i++)
      for (uint k=0; k < dict_[i].size(); k++) {
	if (smoothed_l0_)
	  reg_term += prior_weight_[i][k] * prob_penalty(dict_[i][k],l0_beta_);
	else
	  reg_term += prior_weight_[i][k] * dict_[i][k];
      }
    
    max_perplexity += reg_term;
    approx_sum_perplexity += reg_term;
    viterbi_max_perplexity += reg_term;

    max_perplexity /= source_sentence_.size();
    approx_sum_perplexity /= source_sentence_.size();
    viterbi_max_perplexity /= source_sentence_.size();

    std::string transfer = (wrapper != 0 && iter == 1) ? " (transfer) " : ""; 

    std::cerr << "IBM-3 max-perplex-energy in between iterations #" << (iter-1) << " and " << iter << transfer << ": "
              << max_perplexity << std::endl;
    std::cerr << "IBM-3 approx-sum-perplex-energy in between iterations #" << (iter-1) << " and " << iter << transfer << ": "
              << approx_sum_perplexity << std::endl;


    if (viterbi_ilp_) {
      std::cerr << "IBM-3 viterbi max-perplex-energy in between iterations #" << (iter-1) << " and " << iter << transfer << ": "
                << viterbi_max_perplexity << std::endl;

      std::cerr << "Viterbi-ILP better in " << nViterbiBetter << ", worse in " << nViterbiWorse << " cases." << std::endl;
      
      std::cerr << "max-ratio: " << max_ratio << std::endl;
    }

    if (possible_ref_alignments_.size() > 0) {
      
      std::cerr << "#### IBM3-AER in between iterations #" << (iter-1) << " and " << iter << transfer << ": " << AER() << std::endl;
      
      if (viterbi_ilp_) {
	std::cerr << "#### IBM3-AER for Viterbi in between iterations #" << (iter-1) << " and " << iter << transfer << ": " 
		  << AER(viterbi_alignment) << std::endl;
      }
      std::cerr << "#### IBM3-fmeasure in between iterations #" << (iter-1) << " and " << iter << transfer << ": " << f_measure() << std::endl;
      std::cerr << "#### IBM3-DAE/S in between iterations #" << (iter-1) << " and " << iter << transfer << ": " 
                << DAE_S() << std::endl;
    }

    std::cerr << (((double) sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" 
              << std::endl; 
  }

  if (!parametric_distortion_) {

    //we compute this so that we can use it for computation of external alignments

    distortion_param_.set_constant(0.0);

    for (uint J=0; J < maxJ_; J++) {

      for (uint i=0; i < distortion_prob_[J].yDim(); i++) {

        for (uint j=0; j < J+1; j++)
          distortion_param_(j,i) += distortion_prob_[J](j,i);
      }
    }

    for (uint i=0; i < distortion_param_.yDim(); i++) {

      double sum = 0.0;
      for (uint j=0; j < distortion_param_.xDim(); j++)
        sum += distortion_param_(j,i);

      if (sum > 1e-305) {

        for (uint j=0; j < distortion_param_.xDim(); j++)
          distortion_param_(j,i) /= sum;
      }
      else
        for (uint j=0; j < distortion_param_.xDim(); j++)
          distortion_param_(j,i) = 1.0 / distortion_param_.xDim();
    }
  }

  iter_offs_ = iter-1;
}

void IBM3Trainer::train_viterbi(uint nIter, HmmWrapper* wrapper, bool use_ilp) {

  const uint nSentences = source_sentence_.size();

  std::cerr << "starting IBM-3 training without constraints" << std::endl;

  double max_perplexity = 0.0;

  ReducedIBM3DistortionModel fdistort_count(distortion_prob_.size(),MAKENAME(fdistort_count));
  for (uint J=0; J < fdistort_count.size(); J++) {
    fdistort_count[J].resize_dirty(distortion_prob_[J].xDim(), distortion_prob_[J].yDim());
  }

  SingleLookupTable aux_lookup;

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords,MAKENAME(ffert_count));

  SingleLookupTable aux;

  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  long double fzero_count;
  long double fnonzero_count;

  if (log_table_.size() < nSentences) {
    EXIT("passed log table is not large enough.");
  }

  Storage1D<Storage1D<Storage1D<uchar> > > best_aligned_source_words;
  if (nondeficient_) {
    best_aligned_source_words.resize(nSentences);

    for (uint s=0; s < nSentences; s++) 
      best_aligned_source_words[s].resize(target_sentence_[s].size()); //we don't need to store entries for the empty word
  }

  uint iter;
  for (iter=1+iter_offs_; iter <= nIter+iter_offs_; iter++) {

    std::cerr << "******* IBM-3 Viterbi-iteration #" << iter << std::endl;

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint J=0; J < fdistort_count.size(); J++) {
      fdistort_count[J].set_constant(0.0);
    }
    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0);
      ffert_count[i].set_constant(0.0);
    }

    max_perplexity = 0.0;

    for (size_t s=0; s < nSentences; s++) {

      if ((s% 10000) == 0)
	std::cerr << "sentence pair #" << s << std::endl;
      
      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
                                                           nSourceWords_,slookup_[s],aux_lookup);
      
      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      Math2D::Matrix<double>& cur_distort_count = fdistort_count[curJ-1];

      Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));

      //these will not actually be used, but need to be passed to the hillclimbing routine
      Math2D::NamedMatrix<long double> swap_move_prob(curJ,curJ,MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ,curI+1,MAKENAME(expansion_move_prob));
      
      long double best_prob;

      if (wrapper != 0) {

	//we can skip calling hillclimbing here - we don't look at the neighbors anyway
	//but note that we have to deal with alignments that have 2*fertility[0] > curJ

	best_prob = compute_ehmm_viterbi_alignment(cur_source,cur_lookup,cur_target,dict_,
						   wrapper->align_model_[curI-1], wrapper->initial_prob_[curI-1],
						   best_known_alignment_[s], wrapper->hmm_options_.align_type_,
						   false, false);

	make_alignment_feasible(cur_source, cur_target, cur_lookup, best_known_alignment_[s]);

	//NOTE: to be 100% proper we should recalculate the prob of the alignment if fertility[0] was corrected
	//(would need to convert the alignment to internal mode first). But this only affects the energy printout at the end of 
	// the iteration
      }
      else {

	best_prob = update_alignment_by_hillclimbing(cur_source,cur_target,cur_lookup,sum_iter,fertility,
						     expansion_move_prob,swap_move_prob,best_known_alignment_[s]);
      }

#ifdef HAS_CBC
      if (use_ilp) {

        Math1D::Vector<AlignBaseType> alignment = best_known_alignment_[s];
        compute_viterbi_alignment_ilp(cur_source, cur_target, cur_lookup, alignment, 0.25);

        if (alignment_prob(s,alignment) > 1e-300) {

          best_known_alignment_[s] = alignment;
	  
          best_prob = update_alignment_by_hillclimbing(cur_source,cur_target,cur_lookup,sum_iter,fertility,
                                                       expansion_move_prob,swap_move_prob,best_known_alignment_[s]);
        }
      }
#endif

      assert(2*fertility[0] <= curJ);


      max_perplexity -= std::log(best_prob);
      

      fzero_count += fertility[0];
      fnonzero_count += curJ - 2*fertility[0];

      //increase counts for dictionary and distortion
      for (uint j=0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = best_known_alignment_[s][j];

        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj-1]][cur_lookup(j,cur_aj-1)] += 1;
          cur_distort_count(j,cur_aj-1) += 1.0;
          assert(!isnan(cur_distort_count(j,cur_aj-1)));
        }
        else {
          fwcount[0][s_idx-1] += 1;
        }
      }

      if (nondeficient_) {

	Storage1D<std::vector<uchar> > aligned_source_words(curI);
	for (uint j=0; j < curJ; j++) {

	  const uint cur_aj = best_known_alignment_[s][j];
	  if (cur_aj > 0)
	    aligned_source_words[cur_aj-1].push_back(j);
	}

	for (uint i=0; i < curI; i++) {
	  best_aligned_source_words[s][i].resize_dirty(aligned_source_words[i].size());

	  for (uint k=0; k < aligned_source_words[i].size(); k++)
	    best_aligned_source_words[s][i][k] = aligned_source_words[i][k];
	}
      }

      //update fertility counts
      for (uint i=1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i-1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }

      assert(!isnan(fzero_count));
      assert(!isnan(fnonzero_count));
    } //loop over sentences finished

    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max<double>(1e-15,fzero_count / fsum);
      p_nonzero_ = std::max<double>(1e-15,fnonzero_count / fsum);
    }

    std::cerr << "new p0: " << p_zero_ << std::endl;
    
    /*** ICM stage ***/
    if (wrapper == 0) { //no use doing ICM in a transfer iteration. 

      Math1D::NamedVector<uint> dict_sum(fwcount.size(),MAKENAME(dict_sum));
      for (uint k=0; k < fwcount.size(); k++)
	dict_sum[k] = fwcount[k].sum();
      
      uint nSwitches = 0;
      
      for (size_t s=0; s < nSentences; s++) {
	
	if ((s% 10000) == 0)
	  std::cerr << "ICM, sentence pair #" << s << std::endl;
	
	const Storage1D<uint>& cur_source = source_sentence_[s];
	const Storage1D<uint>& cur_target = target_sentence_[s];
	const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
							     nSourceWords_,slookup_[s],aux_lookup);
	
	Math1D::Vector<AlignBaseType>& cur_best_known_alignment = best_known_alignment_[s];
	
	const uint curI = cur_target.size();
	const uint curJ = cur_source.size();
	
	Math1D::Vector<uint> cur_fertilities(curI+1,0);
	for (uint j=0; j < curJ; j++) {
	  
	  uint cur_aj = cur_best_known_alignment[j];
	  cur_fertilities[cur_aj]++;
	}
	
	Storage1D<std::vector<AlignBaseType> > aligned_source_words(curI+1);
	if (nondeficient_) {
	  for (uint j=0; j < curJ; j++) {
	  
	    uint cur_aj = cur_best_known_alignment[j];
	    aligned_source_words[cur_aj].push_back(j);
	  }
	}

	const Math2D::Matrix<double>& cur_distort_prob = distortion_prob_[curJ-1];
	
	Math2D::Matrix<double>& cur_distort_count = fdistort_count[curJ-1];

	double cur_dist_log = 0.0;
	double hyp_dist_log = 0.0;

	if (nondeficient_)
	  cur_dist_log = std::log(nondeficient_distortion_prob(cur_source, cur_target, aligned_source_words));

	Storage1D<std::vector<AlignBaseType> > hyp_aligned_source_words = aligned_source_words;
	
	for (uint j=0; j < curJ; j++) {

	  for (uint i = 0; i <= curI; i++) {
	    
	    uint cur_aj = cur_best_known_alignment[j];
	    uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];
	    
	    /**** dict ***/
	    
	    bool allowed = (cur_aj != i && (i != 0 || 2*cur_fertilities[0]+2 <= curJ));
	    
	    if (i != 0 && (cur_fertilities[i]+1) > fertility_limit_)
	      allowed = false;
	    
	    if (allowed) {

	      if (!nondeficient_) {
		hyp_aligned_source_words[cur_aj].erase(std::find(hyp_aligned_source_words[cur_aj].begin(),
								 hyp_aligned_source_words[cur_aj].end(),j));
		hyp_aligned_source_words[i].push_back(j);
		vec_sort(hyp_aligned_source_words[i]);
	      }
  
	      uint new_target_word = (i == 0) ? 0 : cur_target[i-1];
	      
	      double change = 0.0;
	      
	      Math1D::Vector<double>& cur_dictcount = fwcount[cur_word]; 
	      Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word]; 
	    
	    
	      if (cur_word != new_target_word) {

		uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);
		
		uint cur_dictsum = dict_sum[cur_word];
		
		uint hyp_idx = (i == 0) ? cur_source[j]-1 : cur_lookup(j,i-1);
		
		if (dict_sum[new_target_word] > 0)
		  change -= double(dict_sum[new_target_word]) * log_table_[ dict_sum[new_target_word] ];
		change += double(dict_sum[new_target_word]+1.0) * log_table_[ dict_sum[new_target_word]+1 ];

		if (fwcount[new_target_word][hyp_idx] > 0)
		  change += double(fwcount[new_target_word][hyp_idx]) * 
		    log_table_[fwcount[new_target_word][hyp_idx]];
		else
		  change += prior_weight_[new_target_word][hyp_idx]; 

		change -= double(fwcount[new_target_word][hyp_idx]+1) * 
		  log_table_[fwcount[new_target_word][hyp_idx]+1];

		change -= double(cur_dictsum) * std::log(cur_dictsum);
		if (cur_dictsum > 1)
		  change += double(cur_dictsum-1) * log_table_[cur_dictsum-1];

		change -= - double(cur_dictcount[cur_idx]) * log_table_[cur_dictcount[cur_idx]];

		
		if (cur_dictcount[cur_idx] > 1) {
		  change += double(cur_dictcount[cur_idx]-1) * (-log_table_[cur_dictcount[cur_idx]-1]);
		}
		else
		  change -= prior_weight_[cur_word][cur_idx];
		
		
		/***** fertilities (only affected if the old and new target word differ) ****/
		
		//note: currently not updating f_zero / f_nonzero
		if (cur_aj == 0) {
		  
		  uint zero_fert = cur_fertilities[0];
		  
		  change -= - std::log(ldchoose(curJ-zero_fert,zero_fert));
		  change -= -std::log(p_zero_);
		  
		  if (och_ney_empty_word_) {
		    change -= -std::log(((long double) zero_fert) / curJ);
		  }
		
		  uint new_zero_fert = zero_fert-1;
		  change += - std::log(ldchoose(curJ-new_zero_fert,new_zero_fert));
		  change += 2.0*(-std::log(p_nonzero_));
		  
		  
		  if (och_ney_empty_word_) {

		  }
		}
		else {

		  change -= - log_table_[cur_fertilities[cur_aj]]; 
		  
		  double c = ffert_count[cur_word][cur_fertilities[cur_aj]];
		  change -= -c * log_table_[c];
		  if (c > 1)
		    change += -(c-1) * log_table_[c-1];
		
		  double c2 = ffert_count[cur_word][cur_fertilities[cur_aj]-1];
		  
		  if (c2 > 0)
		    change -= -c2 * log_table_[c2];
		  change += -(c2+1) * log_table_[c2+1];
		}
		
		if (i == 0) {
		  
		  uint zero_fert = cur_fertilities[0];
		  
		  change -= -std::log(ldchoose(curJ-zero_fert,zero_fert));
		  change -= 2.0*(-std::log(p_nonzero_));
		  
		  if (och_ney_empty_word_) {

		  }
		  
		  uint new_zero_fert = zero_fert+1;
		  change += - std::log(ldchoose(curJ-new_zero_fert,new_zero_fert));
		  change += -std::log(p_zero_);
		  
		  if (och_ney_empty_word_) {
		    change += -std::log(((long double) new_zero_fert) / curJ);
		  }
		}
		else {

		  change += - log_table_[cur_fertilities[i]+1]; 
		  
		  double c = ffert_count[new_target_word][cur_fertilities[i]];
		  change -= -c * log_table_[c]; 
		  if (c > 1)
		    change += -(c-1) * log_table_[c-1];
		  else
		    change -= l0_fertpen_;
		  
		  double c2 = ffert_count[new_target_word][cur_fertilities[i]+1];
		  if (c2 > 0)
		    change -= -c2 * log_table_[c2];
		  else
		    change += l0_fertpen_;
		  change += -(c2+1) * log_table_[c2+1];
		}
	      }
	      
	      /***** distortion ****/
	      if (!nondeficient_) {
		if (cur_aj != 0) {
		
		  if (parametric_distortion_) {
		    
		    change -= -std::log(cur_distort_prob(j,cur_aj-1));
		  }
		  else {
		    
		    uint c = cur_distort_count(j,cur_aj-1);
		    
		    change -= -c * log_table_[c];
		    if (c > 1)
		      change += -(c-1) * log_table_[c-1];
		  }
		}
		if (i != 0) {
		  
		  if (parametric_distortion_) {
		    
		    change += -std::log(cur_distort_prob(j,i-1));
		  }
		  else {
		    
		    uint c = cur_distort_count(j,i-1);
		    if (c > 0)
		      change -= -c * log_table_[c];
		    change += -(c+1) * log_table_[c+1];
		  }
		}
	      }	
	      else {

		hyp_dist_log = std::log(nondeficient_distortion_prob(cur_source, cur_target, hyp_aligned_source_words));


		change += cur_dist_log; // - - = +
		change -= hyp_dist_log;
	      }
	      
	      if (change < -0.01) {
		
		cur_best_known_alignment[j] = i;
		nSwitches++;
		
		uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);
		
		uint hyp_idx = (i == 0) ? cur_source[j]-1 : cur_lookup(j,i-1);
		
		//dict
		cur_dictcount[cur_idx] -= 1;
		hyp_dictcount[hyp_idx] += 1;
		dict_sum[cur_word] -= 1;
		dict_sum[new_target_word] += 1;
		
		//fert
		if (cur_word != 0) {
		  uint prev_fert = cur_fertilities[cur_aj];
		  assert(prev_fert != 0);
		  ffert_count[cur_word][prev_fert] -= 1;
		  ffert_count[cur_word][prev_fert-1] += 1;
		}
		if (new_target_word != 0) {
		  uint prev_fert = cur_fertilities[i];
		  ffert_count[new_target_word][prev_fert] -= 1;
		  ffert_count[new_target_word][prev_fert+1] += 1;
		}
		
		cur_fertilities[cur_aj]--;
		cur_fertilities[i]++;
		
		//distort
		if (cur_aj != 0)
		  cur_distort_count(j,cur_aj-1)--;
		if (i != 0)
		  cur_distort_count(j,i-1)++;
		
		if (nondeficient_) {

		  aligned_source_words[i] = hyp_aligned_source_words[i];
		  aligned_source_words[cur_aj] = hyp_aligned_source_words[cur_aj];

		  cur_dist_log = hyp_dist_log;
		}
	      }
	      else if (nondeficient_) {

		hyp_aligned_source_words[i] = aligned_source_words[i];
		hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
	      }
	    }
	  }
	}
	
	if (nondeficient_) {

	  for (uint i=0; i < curI; i++) {
	    best_aligned_source_words[s][i].resize_dirty(aligned_source_words[i+1].size());

	    for (uint k=0; k < aligned_source_words[i+1].size(); k++)
	      best_aligned_source_words[s][i][k] = aligned_source_words[i+1][k];
	  }
	}


      } //ICM-loop over sentences finished
      
      std::cerr << nSwitches << " changes in ICM stage" << std::endl;
    }

    //DEBUG
#ifndef NDEBUG
    uint nZeroAlignments = 0;
    uint nAlignments = 0;
    for (size_t s=0; s < source_sentence_.size(); s++) {

      nAlignments += source_sentence_[s].size();

      for (uint j=0; j < source_sentence_[s].size(); j++) {
        if (best_known_alignment_[s][j] == 0)
          nZeroAlignments++;
      }
    }
    std::cerr << "percentage of zero-aligned words: " 
              << (((double) nZeroAlignments) / ((double) nAlignments)) << std::endl;
#endif
    //END_DEBUG

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, 0.0, iter,  false, 0.0, 0, dict_);

    if (parametric_distortion_) {

      Math2D::Matrix<double> fpar_distort_count(distortion_param_.xDim(),distortion_param_.yDim(),0.0);

      for (uint J=0; J < fdistort_count.size(); J++) {

        for (uint x=0; x < fdistort_count[J].xDim(); x++)
          for (uint y=0; y < fdistort_count[J].yDim(); y++)
            fpar_distort_count(x,y) += fdistort_count[J](x,y);
      }

      for (uint y=0; y < fpar_distort_count.yDim(); y++) {
        
        double sum =  0.0;
        for (uint x = 0; x < fpar_distort_count.xDim(); x++)
          sum += fpar_distort_count(x,y);

        if (sum > 1e-305) {
          for (uint x = 0; x < fpar_distort_count.xDim(); x++)
            fpar_distort_count(x,y) /= sum;
        }
      }


      if (!nondeficient_) {

	for (uint i=0; i < maxI_; i++) {
	  std::cerr << "m-step for i=" << i << std::endl;

          double start_energy = par_distortion_m_step_energy(fdistort_count,distortion_param_,i);

          double hyp_energy = par_distortion_m_step_energy(fdistort_count,fpar_distort_count,i);

          if (hyp_energy < start_energy) {

            for (uint x = 0; x < fpar_distort_count.xDim(); x++) {
              distortion_param_(x,i) = fpar_distort_count(x,i);
            }

	    start_energy = hyp_energy;
          }

	  par_distortion_m_step(fdistort_count,i,start_energy);
	}
	par2nonpar_distortion(distortion_prob_);
      }
      else {

	for (uint i=0; i < maxI_; i++) {

	  std::map<Math1D::Vector<uchar,uchar>,double> cur_nondef_count;

	  for (uint s=0; s < nSentences; s++) {

	    const uint curJ = source_sentence_[s].size();
	    const uint curI = target_sentence_[s].size();

	    const Storage1D<Storage1D<uchar> > cur_best_aligned_source_words = best_aligned_source_words[s];

	    if (i < curI && cur_best_aligned_source_words[i].size() > 0) {

	      Storage1D<bool> fixed(curJ,false); //NOTE: for speed we could keep the fixed vectors in memory outside the loop over i
	      
	      for (uint ii=0; ii < i; ii++) {
		for (uint k=0; k < cur_best_aligned_source_words[ii].size(); k++)
		  fixed[cur_best_aligned_source_words[ii][k]] = true;
	      }

	      //1. head position
	      uint head = cur_best_aligned_source_words[i][0];

	      {
		std::vector<uchar> open_pos;
		uint head_pos = MAX_UINT;

		for (uint j=0; j < curJ; j++) {
		  if (!fixed[j]) {
		    if (j == head)
		      head_pos = open_pos.size();
		    open_pos.push_back(j);
		  }
		}

		if (head_pos+1 != open_pos.size())
		  std::swap(open_pos[head_pos],open_pos[open_pos.size()-1]);

		Math1D::Vector<uchar,uchar> vec_open_pos(open_pos.size());
		for (uint k=0; k < open_pos.size(); k++)
		  vec_open_pos[k] = open_pos[k];

		cur_nondef_count[vec_open_pos] += 1.0;

		fixed[head] = true; //only relevant if we store the fixed vectors
	      }
		
	      //2. remaining positions
	      uint prev = head;
	      for (uint kk=1; kk < cur_best_aligned_source_words[i].size(); kk++) {

		uint cur = cur_best_aligned_source_words[i][kk];

		std::vector<uchar> open_pos;
		uint cur_pos = MAX_UINT;
		
		for (uint j=prev+1; j < curJ; j++) {
		  if (!fixed[j]) {
		    if (j == cur)
		      cur_pos = open_pos.size();
		    open_pos.push_back(j);
		  }
		}

		if (cur_pos+1 != open_pos.size())
		  std::swap(open_pos[cur_pos],open_pos[open_pos.size()-1]);

		Math1D::Vector<uchar,uchar> vec_open_pos(open_pos.size());
		for (uint k=0; k < open_pos.size(); k++)
		  vec_open_pos[k] = open_pos[k];

		cur_nondef_count[vec_open_pos] += 1.0;
				
		fixed[cur] = true; //only relevant if we store the fixed vectors
		prev = cur;
	      }

	    }
	  }


          if (cur_nondef_count.empty())
            continue;

	  //copy map->vector for faster access
          std::vector<Math1D::Vector<uchar,uchar> > open_pos;
          std::vector<double> vec_nondef_count;

          vec_nondef_count.reserve(cur_nondef_count.size());
          open_pos.reserve(cur_nondef_count.size());
          for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = cur_nondef_count.begin(); 
               it != cur_nondef_count.end(); it++) {
            
            open_pos.push_back(it->first);
            vec_nondef_count.push_back(it->second);
          }
          cur_nondef_count.clear();

          std::cerr << "calling  m-step for i=" << i << std::endl;

          double start_energy = nondeficient_m_step_energy(open_pos, vec_nondef_count, distortion_param_,i);

          double hyp_energy = nondeficient_m_step_energy(open_pos, vec_nondef_count, fpar_distort_count,i);

          if (hyp_energy < start_energy) {

            for (uint x = 0; x < fpar_distort_count.xDim(); x++) {
              distortion_param_(x,i) = fpar_distort_count(x,i);
            }

            std::cerr << "switching to normalized counts: " << start_energy << " -> " << hyp_energy << std::endl;

            start_energy = hyp_energy;
          }

          nondeficient_m_step_with_interpolation(open_pos,vec_nondef_count,i,start_energy);

	}
      }
    }
    else {
      //nonparametric distortion

      if (!nondeficient_) {

	for (uint J=0; J < distortion_prob_.size(); J++) {
	
	  for (uint i=0; i < distortion_prob_[J].yDim(); i++) {
	    
	    double sum = 0.0;
	    for (uint j=0; j < J+1; j++)
	      sum += fdistort_count[J](j,i);
	    
	    if (sum > 1e-305) {
	      const double inv_sum = 1.0 / sum;
	      assert(!isnan(inv_sum));
	      
	      for (uint j=0; j < J+1; j++) {
		distortion_prob_[J](j,i) = std::max(1e-8,inv_sum * fdistort_count[J](j,i));
		if (isnan(distortion_prob_[J](j,i))) {
		  std::cerr << "sum: " << sum << std::endl;
		  std::cerr << "set to " << inv_sum << " * " << fdistort_count[J](j,i) << " = "
			    << (inv_sum * fdistort_count[J](j,i)) << std::endl;
		}
		assert(!isnan(fdistort_count[J](j,i)));
		assert(!isnan(distortion_prob_[J](j,i)));
	      }
	    }
	    else {
	      //std::cerr << "WARNING: did not update distortion count because sum was " << sum << std::endl;
	    }
	  }
	}
      }
      else {

        ReducedIBM3DistortionModel normalized_distort_count(MAKENAME(normalized_distort_count));
        normalized_distort_count = fdistort_count;
        for (uint J=0; J < fdistort_count.size(); J++) {

	  for (uint y=0; y < normalized_distort_count[J].yDim(); y++) {

	    double sum = 0.0;
	    for (uint x=0; x < normalized_distort_count[J].xDim(); x++) 
	      sum += normalized_distort_count[J](x,y);

	    if (sum > 1e-305) {
	      for (uint x=0; x < normalized_distort_count[J].xDim(); x++) 
		normalized_distort_count[J](x,y) = normalized_distort_count[J](x,y) / sum;
	    }
	    else {
	      for (uint x=0; x < normalized_distort_count[J].xDim(); x++) 
		normalized_distort_count[J](x,y) = 1.0 / normalized_distort_count[J].xDim();
	    }
	  }
        }


        for (uint i=0; i < maxI_; i++) {

          std::cerr << "calling nondeficient m-steps for i=" << i << std::endl;

          for (uint J=1; J <= fdistort_count.size(); J++) {
	    
	    if (fdistort_count[J-1].size() == 0)
	      continue;

	    std::map<Math1D::Vector<uchar,uchar>,double> cur_nondef_count;

	    for (uint s=0; s < nSentences; s++) {

	      if (source_sentence_[s].size() != J)
		continue;

	      const uint curI = target_sentence_[s].size();
	      const Storage1D<Storage1D<uchar> > cur_best_aligned_source_words = best_aligned_source_words[s];

	      if (i < curI && cur_best_aligned_source_words[i].size() > 0) {

		//just like in the parametric case
		
		Storage1D<bool> fixed(J,false); //NOTE: for speed we could keep the fixed vectors in memory outside the loop over i
	      
		for (uint ii=0; ii < i; ii++) {
		  for (uint k=0; k < cur_best_aligned_source_words[ii].size(); k++)
		    fixed[cur_best_aligned_source_words[ii][k]] = true;
		}

		//1. head position
		uint head = cur_best_aligned_source_words[i][0];
		
		{
		  std::vector<uchar> open_pos;
		  uint head_pos = MAX_UINT;

		  for (uint j=0; j < J; j++) {
		    if (!fixed[j]) {
		      if (j == head)
			head_pos = open_pos.size();
		      open_pos.push_back(j);
		    }
		  }

		  if (head_pos+1 != open_pos.size())
		    std::swap(open_pos[head_pos],open_pos[open_pos.size()-1]);

		  Math1D::Vector<uchar,uchar> vec_open_pos(open_pos.size());
		  for (uint k=0; k < open_pos.size(); k++)
		    vec_open_pos[k] = open_pos[k];

		  cur_nondef_count[vec_open_pos] += 1.0;

		  fixed[head] = true; //only relevant if we store the fixed vectors
		}
		
		//2. remaining positions
		uint prev = head;
		for (uint kk=1; kk < cur_best_aligned_source_words[i].size(); kk++) {

		  uint cur = cur_best_aligned_source_words[i][kk];

		  std::vector<uchar> open_pos;
		  uint cur_pos = MAX_UINT;
		
		  for (uint j=prev+1; j < J; j++) {
		    if (!fixed[j]) {
		      if (j == cur)
			cur_pos = open_pos.size();
		      open_pos.push_back(j);
		    }
		  }

		  if (cur_pos+1 != open_pos.size())
		    std::swap(open_pos[cur_pos],open_pos[open_pos.size()-1]);

		  Math1D::Vector<uchar,uchar> vec_open_pos(open_pos.size());
		  for (uint k=0; k < open_pos.size(); k++)
		    vec_open_pos[k] = open_pos[k];

		  cur_nondef_count[vec_open_pos] += 1.0;
				
		  fixed[cur] = true; //only relevant if we store the fixed vectors
		  prev = cur;
		}

	      }
	    }
 
	    
	    if (cur_nondef_count.empty())
	      continue;

	    //copy map->vector for faster access
            std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> > vec_nondef_count;	    
	    vec_nondef_count.reserve(cur_nondef_count.size());
	    for (std::map<Math1D::Vector<uchar,uchar>,double>::const_iterator it = cur_nondef_count.begin(); 
		 it != cur_nondef_count.end(); it++) {
	      
	      vec_nondef_count.push_back(*it);
	    }
	    cur_nondef_count.clear();
	    
            double start_energy = nondeficient_m_step_energy(vec_nondef_count,distortion_prob_,i,J);

            double hyp_energy = nondeficient_m_step_energy(vec_nondef_count,normalized_distort_count,i,J);

            if (hyp_energy < start_energy) {

              start_energy = hyp_energy;

	      for (uint j=0; j < distortion_prob_[J-1].xDim(); j++)
		distortion_prob_[J-1](j,i) = normalized_distort_count[J-1](j,i);
            }

            nondeficient_m_step(vec_nondef_count,i,J,start_energy);
	  }
	}

      }
    }
    

    update_fertility_prob(ffert_count,0.0);

    max_perplexity = 0.0;
    for (size_t s=0; s < source_sentence_.size(); s++) {
      if (nondeficient_)
	max_perplexity -= std::log(nondeficient_alignment_prob(s,best_known_alignment_[s]));
      else
	max_perplexity -= std::log(alignment_prob(s,best_known_alignment_[s]));
    }

    for (uint i=0; i < fwcount.size(); i++)
      for (uint k=0; k < fwcount[i].size(); k++)
        if (fwcount[i][k] > 0)
          max_perplexity += prior_weight_[i][k];

    max_perplexity /= source_sentence_.size();

    std::string transfer = (wrapper != 0 && iter == 1) ? " (transfer) " : ""; 

    std::cerr << "IBM-3 energy after iteration #" << iter << transfer << ": "
              << max_perplexity << std::endl;


    if (possible_ref_alignments_.size() > 0) {
      
      std::cerr << "#### IBM-3-AER in between iterations #" << (iter-1) << " and " << iter << transfer 
		<< ": " << AER() << std::endl;
      std::cerr << "#### IBM-3-fmeasure in between iterations #" << (iter-1) << " and " << iter << transfer
		<< ": " << f_measure() << std::endl;
      std::cerr << "#### IBM-3-DAE/S in between iterations #" << (iter-1) << " and " << iter << transfer << ": " 
                << DAE_S() << std::endl;
    }

    std::cerr << (((double) sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" 
              << std::endl; 
  }

  if (!parametric_distortion_) {

    //we compute this so that we can use it for computation of external alignments

    distortion_param_.set_constant(0.0);

    for (uint J=0; J < maxJ_; J++) {

      for (uint i=0; i < distortion_prob_[J].yDim(); i++) {

        for (uint j=0; j < J+1; j++)
          distortion_param_(j,i) += distortion_prob_[J](j,i);
      }
    }

    for (uint i=0; i < distortion_param_.yDim(); i++) {

      double sum = 0.0;
      for (uint j=0; j < distortion_param_.xDim(); j++)
        sum += distortion_param_(j,i);

      if (sum > 1e-305) {

        for (uint j=0; j < distortion_param_.xDim(); j++)
          distortion_param_(j,i) /= sum;
      }
      else
        for (uint j=0; j < distortion_param_.xDim(); j++)
          distortion_param_(j,i) = 1.0 / distortion_param_.xDim();
    }
  }

  iter_offs_ = iter-1;
}


long double IBM3Trainer::compute_itg_viterbi_alignment_noemptyword(uint s, bool extended_reordering) {

  const Storage1D<uint>& cur_source = source_sentence_[s];
  const Storage1D<uint>& cur_target = target_sentence_[s];


  std::cerr << "******** compute_itg_viterbi_alignment_noemptyword(" << s 
            << ") J: " << cur_source.size() << ", I: " << cur_target.size() << " **********" << std::endl;

  SingleLookupTable aux_lookup;

  const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
                                                       nSourceWords_,slookup_[s],aux_lookup);  
  
  const uint curI = cur_target.size();
  const uint curJ = cur_source.size();
  const Math2D::Matrix<double>& cur_distort_prob = distortion_prob_[curJ-1];
  
  Math2D::NamedMatrix<long double> score0(curI,curI,0.0,MAKENAME(score0));

  for (uint i=0; i < curI; i++) {
    long double prob = fertility_prob_[cur_target[i]][0];
    score0(i,i) = prob;
    for (uint ii=i+1; ii < curI; ii++) {
      prob *= fertility_prob_[cur_target[ii]][0];
      score0(i,ii) = prob;
    }
  }

  NamedStorage1D<Math3D::NamedTensor<long double> > score(curJ+1,MAKENAME(score));
  NamedStorage1D<Math3D::NamedTensor<uint> > trace(curJ+1,MAKENAME(trace));

  score[1].set_name("score[1]");
  score[1].resize(curJ,curI,curI,0.0);
  trace[1].set_name("trace[1]");
  trace[1].resize(curJ,curI,curI,MAX_UINT);

  Math3D::NamedTensor<long double>& score1 = score[1];
  Math3D::NamedTensor<uint>& trace1 = trace[1];

  for (uint j=0; j < curJ; j++) {
    
    for (uint i=0; i < curI; i++) {
      for (uint ii=i; ii < curI; ii++) {

        const long double zero_prob = score0(i,ii);

        long double best_prob = 0.0;
	
        for (uint iii=i; iii <= ii; iii++) {

          const uint t_idx = cur_target[iii];

          long double hyp_prob = fertility_prob_[t_idx][1]
            * cur_distort_prob(j,iii) * dict_[t_idx][cur_lookup(j,iii)] / fertility_prob_[t_idx][0];

          if (hyp_prob > best_prob) {
            best_prob = hyp_prob;
            trace1(j,i,ii) = iii;
          }
        }

        score1(j,i,ii) = zero_prob * best_prob;
      }
    }
  }

  for (uint J=2; J <= curJ; J++) {

    score[J].set_name("score[" + toString(J) + "]");
    score[J].resize(curJ,curI,curI,0.0);
    trace[J].set_name("trace[" + toString(J) + "]");
    trace[J].resize(curJ,curI,curI,MAX_UINT);

    const long double Jfac = ld_fac_[J];

    Math3D::NamedTensor<uint>& traceJ = trace[J];
    Math3D::NamedTensor<long double>& scoreJ = score[J];
    
    for (uint I=1; I <= curI; I++) {

      for (uint i=0; i < (curI-(I-1)); i++) {

        const uint ii=i+I-1;
        assert(ii < curI);
        assert(ii >= i);

        const uint ti = cur_target[i];
        const uint tii = cur_target[ii];

        for (uint j=0; j < (curJ-(J-1)); j++) {
	  
          const uint jj = j + J -1; 
          assert(jj < curJ);
          assert(jj >= j);
	  
          long double best_prob = 0.0;
          uint trace_entry = MAX_UINT;

          if (I==1) {
            best_prob = 1.0;
            for (uint jjj = j; jjj <= jj; jjj++)
              best_prob *= dict_[ti][cur_lookup(jjj,i)] * cur_distort_prob(jjj,i);
            best_prob *= fertility_prob_[ti][J] * Jfac;
          }
          else {

            if (extended_reordering && I <= 10 && J <= 10) {

              long double base_prob;
              base_prob = ldfac(J-1) * fertility_prob_[ti][J-1] * fertility_prob_[tii][1];
              for (uint iii=i+1; iii <= ii-1; iii++)
                base_prob *= fertility_prob_[cur_target[iii]][0];
	      
              for (uint k=1; k < J-1; k++) {
		
                long double hyp_prob = base_prob;
                for (uint l=0; l < J; l++) {
		  
                  if (l==k)
                    hyp_prob *= dict_[tii][cur_lookup(j+l,ii)] * cur_distort_prob(j+l,ii);
                  else
                    hyp_prob *= dict_[ti][cur_lookup(j+l,i)] * cur_distort_prob(j+l,i);
		  
                }
                if (hyp_prob > best_prob) {
                  best_prob = hyp_prob;
		    
                  trace_entry = 0xC0000000;
                  uint base = 1;
                  for (uint l=0; l < J; l++) {
                    if (l==k)
                      trace_entry += base;
                    base *= 2;
                  }
                }
              }
		
              base_prob = ld_fac_[J-1] * fertility_prob_[tii][J-1] * fertility_prob_[ti][1];
              for (uint iii=i+1; iii <= ii-1; iii++)
                base_prob *= fertility_prob_[cur_target[iii]][0];
		
              for (uint k=1; k < J-1; k++) {
		  
                long double hyp_prob = base_prob;
                for (uint l=0; l < J; l++) {
		    
                  if (l==k)
                    hyp_prob *= dict_[ti][cur_lookup(j+l,i)] * cur_distort_prob(j+l,i);
                  else
                    hyp_prob *= dict_[tii][cur_lookup(j+l,ii)] * cur_distort_prob(j+l,ii);
		    
                }
                if (hyp_prob > best_prob) {
                  best_prob = hyp_prob;
		  
                  trace_entry = 0xC0000000;
                  uint base = 1;
                  for (uint l=0; l < J; l++) {
                    if (l!=k)
                      trace_entry += base;
                    base *= 2;
                  }
                }
              }
            } //end of extended reordering

            //1.) consider extending the target interval by a zero-fertility word
            const double left_extend_prob  = scoreJ(j,i+1,ii) * fertility_prob_[ti][0];
            if (left_extend_prob > best_prob) {
              best_prob = left_extend_prob;
              trace_entry = MAX_UINT - 1;
            }
            const double right_extend_prob = scoreJ(j,i,ii-1) * fertility_prob_[tii][0];
            if (right_extend_prob > best_prob) {
              best_prob = right_extend_prob;
              trace_entry = MAX_UINT - 2;
            }

            //2.) consider splitting both source and target interval
	    
            for (uint split_j = j; split_j < jj; split_j++) {
	      
              //partitioning into [j,split_j] and [split_j+1,jj]
	      
              const uint J1 = split_j - j + 1;
              const uint J2 = jj - split_j;
              assert(J1 >= 1 && J1 < J);
              assert(J2 >= 1 && J2 < J);
              assert(J1 + J2 == J);
              const uint split_j_p1 = split_j + 1;
	      
              const Math3D::Tensor<long double>& score_J1 = score[J1];
              const Math3D::Tensor<long double>& score_J2 = score[J2];
	      
              for (uint split_i = i; split_i < ii; split_i++) {
		
                //partitioning into [i,split_i] and [split_i+1,ii]
		
                // 		const uint I1 = split_i - i + 1;
                // 		const uint I2 = ii - split_i;
                // 		assert(I1 >= 1 && I1 < I);
                // 		assert(I2 >= 1 && I2 < I);

                const long double hyp_monotone_prob = score_J1(j,i,split_i) * score_J2(split_j_p1,split_i+1,ii);

                if (hyp_monotone_prob > best_prob) {
                  best_prob = hyp_monotone_prob;
                  trace_entry = 2*(split_j * curI + split_i);
                }
		
                const long double hyp_invert_prob = score_J2(split_j_p1,i,split_i) * score_J1(j,split_i+1,ii);

                if (hyp_invert_prob > best_prob) {
                  best_prob = hyp_invert_prob;
                  trace_entry = 2*(split_j * curI + split_i) + 1;
                }
              }
            }
          }

          scoreJ(j,i,ii) = best_prob;
          traceJ(j,i,ii) = trace_entry;
        }
      }
    }
  }

  best_known_alignment_[s].set_constant(0);
  itg_traceback(s,trace,curJ,0,0,curI-1);

  return score[curJ](0,0,curI-1);
}

void IBM3Trainer::itg_traceback(uint s, const NamedStorage1D<Math3D::NamedTensor<uint> >& trace, 
                                uint J, uint j, uint i, uint ii) {


  uint trace_entry = trace[J](j,i,ii);

  if (J == 1) {
    best_known_alignment_[s][j] = trace_entry+1;
  }
  else if (i==ii) {
    assert(trace_entry == MAX_UINT);
    
    for (uint jj=j; jj < j+J; jj++)
      best_known_alignment_[s][jj] = i+1;
  }
  else if (trace_entry == MAX_UINT-1) {
    itg_traceback(s,trace,J,j,i+1,ii);
  }
  else if (trace_entry == MAX_UINT-2) {
    itg_traceback(s,trace,J,j,i,ii-1);
  }
  else if (trace_entry >= 0xC0000000) {

    uint temp = trace_entry & 0x3FFFFFF;
    for (uint k=0; k < J; k++) {

      uint bit = (temp % 2);
      best_known_alignment_[s][j+k] = (bit == 1) ? (ii+1) : (i+1);
      temp /= 2;
    }
  }
  else {
    
    bool reverse = ((trace_entry % 2) == 1);
    trace_entry /= 2;

    uint split_i = trace_entry % target_sentence_[s].size();
    uint split_j = trace_entry / target_sentence_[s].size();

    assert(split_i < target_sentence_[s].size());
    assert(split_j < source_sentence_[s].size());
    
    const uint J1 = split_j - j + 1;
    const uint J2 = J - J1;

    if (!reverse) {
      itg_traceback(s,trace,J1,j,i,split_i);
      itg_traceback(s,trace,J2,split_j+1,split_i+1,ii);
    }
    else {
      itg_traceback(s,trace,J2,split_j+1,i,split_i);
      itg_traceback(s,trace,J1,j,split_i+1,ii);
    }
  }

}

void IBM3Trainer::train_with_itg_constraints(uint nIter, bool extended_reordering, bool verbose) {


  ReducedIBM3DistortionModel fdistort_count(distortion_prob_.size(),MAKENAME(fdistort_count));
  for (uint J=0; J < fdistort_count.size(); J++) {
    fdistort_count[J].resize_dirty(distortion_prob_[J].xDim(), distortion_prob_[J].yDim());
  }

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords,MAKENAME(ffert_count));

  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  long double fzero_count;
  long double fnonzero_count;

  SingleLookupTable aux_lookup;
    
  for (uint iter=1+iter_offs_; iter <= nIter+iter_offs_; iter++) {

    std::cerr << "******* IBM-3 ITG-iteration #" << iter << std::endl;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    for (uint J=0; J < distortion_prob_.size(); J++) {
      fdistort_count[J].set_constant(0.0);
    }

    double max_perplexity = 0.0;

    uint nBetter = 0;
    uint nEqual = 0;

    for (size_t s=0; s < source_sentence_.size(); s++) {

      long double hillclimbprob = alignment_prob(s,best_known_alignment_[s]);

      long double prob = compute_itg_viterbi_alignment_noemptyword(s,extended_reordering);

      long double actual_prob = prob * pow(p_nonzero_,source_sentence_[s].size());

      max_perplexity -= std::log(actual_prob);

      if (actual_prob < 1e-305)
        continue;
      
      if (verbose) {
        long double check_prob = alignment_prob(s,best_known_alignment_[s]);
        long double check_ratio = actual_prob / check_prob;
	
        if (check_prob == hillclimbprob)
          nEqual++;
        else if (check_prob > hillclimbprob)
          nBetter++;
	
        assert(check_ratio >= 0.999 && check_ratio < 1.001);
      }

      const Storage1D<uint>&  cur_source = source_sentence_[s];
      const Storage1D<uint>&  cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
                                                           nSourceWords_,slookup_[s],aux_lookup);

      const uint curJ = source_sentence_[s].size();
      const uint curI = target_sentence_[s].size();
 
      Math2D::Matrix<double>& cur_distort_count = fdistort_count[curJ-1];

      Math1D::Vector<uint> fertility(curI+1,0);
      for (uint j=0; j < curJ; j++) {
        fertility[best_known_alignment_[s][j]]++;
      }

      //currently implementing Viterbi training

      double cur_zero_weight = 1.0;
      
      fzero_count += cur_zero_weight * (fertility[0]);
      fnonzero_count += cur_zero_weight * (curJ - 2*fertility[0]);

      //increase counts for dictionary and distortion
      for (uint j=0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = best_known_alignment_[s][j];

        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj-1]][cur_lookup(j,cur_aj-1)] += 1;
          cur_distort_count(j,cur_aj-1) += 1.0;
          assert(!isnan(cur_distort_count(j,cur_aj-1)));
        }
        else {
          fwcount[0][s_idx-1] += 1;
        }
      }

      //update fertility counts
      for (uint i=1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i-1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }      
    }
    
    max_perplexity /= source_sentence_.size();

    //update p_zero_ and p_nonzero_
    double fsum = fzero_count + fnonzero_count;
    p_zero_ = fzero_count / fsum;
    p_nonzero_ = fnonzero_count / fsum;

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, 0.0, iter,  false, 0.0, 0, dict_, 1e-6);

    //update distortion prob from counts
    for (uint J=0; J < distortion_prob_.size(); J++) {

      for (uint i=0; i < distortion_prob_[J].yDim(); i++) {

        double sum = 0.0;
        for (uint j=0; j < J+1; j++)
          sum += fdistort_count[J](j,i);

        if (sum > 1e-305) {
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint j=0; j < J+1; j++) {
            distortion_prob_[J](j,i) = inv_sum * fdistort_count[J](j,i);
            if (isnan(distortion_prob_[J](j,i))) {
              std::cerr << "sum: " << sum << std::endl;
              std::cerr << "set to " << inv_sum << " * " << fdistort_count[J](j,i) << " = "
                        << (inv_sum * fdistort_count[J](j,i)) << std::endl;
            }
            assert(!isnan(fdistort_count[J](j,i)));
            assert(!isnan(distortion_prob_[J](j,i)));
          }
        }
        else {
          //std::cerr << "WARNING: did not update distortion count because sum was " << sum << std::endl;
        }
      }
    }

    update_fertility_prob(ffert_count);


    if (possible_ref_alignments_.size() > 0) {
      
      std::cerr << "#### IBM3-AER in between iterations #" << (iter-1) << " and " << iter << ": " << AER() << std::endl;
      std::cerr << "#### IBM3-fmeasure in between iterations #" << (iter-1) << " and " << iter << ": " << f_measure() << std::endl;
      std::cerr << "#### IBM3-DAE/S in between iterations #" << (iter-1) << " and " << iter << ": " 
                << DAE_S() << std::endl;
    }


    std::cerr << "max-perplexility after iteration #" << (iter - 1) << ": " << max_perplexity << std::endl;
    if (verbose) {
      std::cerr << "itg-constraints are eqaul to hillclimbing in " << nEqual << " cases" << std::endl;
      std::cerr << "itg-constraints are better than hillclimbing in " << nBetter << " cases" << std::endl;
    }

  }
}

long double IBM3Trainer::compute_ibmconstrained_viterbi_alignment_noemptyword(uint s, uint maxFertility, 
                                                                              uint nMaxSkips) {

  std::cerr << "******** compute_ibmconstrained_viterbi_alignment_noemptyword2(" << s << ") **********" << std::endl;

  assert(maxFertility >= 1);

  //convention here: target positions start at 0, source positions start at 1 
  // (so we can express that no source position was covered yet)

  SingleLookupTable aux_lookup;

  const Storage1D<uint>& cur_source = source_sentence_[s];
  const Storage1D<uint>& cur_target = target_sentence_[s];
  const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
                                                       nSourceWords_,slookup_[s],aux_lookup);
  

  const uint curI = cur_target.size();
  const uint curJ = cur_source.size();
  const Math2D::Matrix<double>& cur_distort_prob = distortion_prob_[curJ-1];

  const uint nStates = first_state_[curJ+1];

  std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
  std::cerr << nStates << " active states" << std::endl;

  maxFertility = std::min(maxFertility,curJ);

  NamedStorage1D<Math2D::NamedMatrix<long double> > score(curI,MAKENAME(score));

  NamedStorage1D<Math2D::NamedMatrix<uchar> > state_trace(curI,MAKENAME(state_trace));
  NamedStorage1D<Math1D::NamedVector<uchar> > fert_trace(curI,MAKENAME(fert_trace));

  Math1D::NamedVector<long double> best_prev_score(nStates,0.0,MAKENAME(best_prev_score));

  score[0].set_name(MAKENAME(score[0]));
  score[0].resize(nStates,maxFertility+1,0.0);

  const uint start_allfert_max_reachable_j  = std::min(curJ, nMaxSkips + maxFertility);
  const uint start_allfert_max_reachable_state = first_state_[start_allfert_max_reachable_j+1]-1;

  fert_trace[0].set_name("fert_trace[0]");
  fert_trace[0].resize(start_allfert_max_reachable_state+1/*,255*/);
  state_trace[0].set_name("state_trace[0]");
  state_trace[0].resize(start_allfert_max_reachable_state+1,maxFertility/*,255*/);

  const uint t_start = cur_target[0];

  //initialize fertility 0
  score[0](0,0) = 1.0;

  //initialize for fertility 1
  for (uint state = 0; state <= start_allfert_max_reachable_state; state++) {

    const uint set_idx = coverage_state_(0,state);
    const uint maxUncoveredPos = (set_idx == 0) ? 0 : uncovered_set_(nMaxSkips-1,set_idx);
    const uint max_covered_j = coverage_state_(1,state);

    assert(max_covered_j <= curJ);
    if (max_covered_j <= nMaxSkips+1 && max_covered_j == maxUncoveredPos+1) {
	
      //check if all positions until max_covered_j are skipped
      const uint nCurUncoveredPositions = nUncoveredPositions_[set_idx];
      
      //TODO: pre-generate a list of start states
      bool consecutive = true;
      for (uint k=1; k <= nCurUncoveredPositions; k++) {
	
        if (uncovered_set_(nMaxSkips-nCurUncoveredPositions+(k-1),set_idx) != k) {
          consecutive = false;
          break;
        }
      }
      
      if (consecutive)
        score[0](state,1) = dict_[t_start][cur_lookup(max_covered_j-1,0)] 
          * cur_distort_prob(max_covered_j-1,0);
    }
  }
 
  //initialize for fertility 2
  for (uint fert=2; fert <= maxFertility; fert++) {

    const uint curfert_max_reachable_j  = std::min(curJ, nMaxSkips + fert);
    const uint curfert_max_reachable_state = first_state_[curfert_max_reachable_j+1]-1;

    for (uint state = 0; state <= curfert_max_reachable_state; state++) {

      assert(coverage_state_(1,state) <= curJ);
	
      long double best_score = 0.0;
      uchar trace_entry = 255;
      
      const uint nPredecessors = predecessor_coverage_states_[state].yDim();
      assert(nPredecessors < 255);
      
      for (uint p=0; p < nPredecessors; p++) {
	
        const uint prev_state = predecessor_coverage_states_[state](0,p);
        const uint cover_j = predecessor_coverage_states_[state](1,p);
	
        assert(cover_j <= curJ);
	
        const long double hyp_score = score[0](prev_state,fert-1) 
          * dict_[t_start][cur_lookup(cover_j-1,0)] * cur_distort_prob(cover_j-1,0);
	
        if (hyp_score > best_score) {
          best_score = hyp_score;
          trace_entry = p;
        }
      }
      
      score[0](state,fert) = best_score;
      state_trace[0](state,fert-1) = trace_entry;
    }
  }

  //finally include fertility probabilities
  for (uint fert = 0; fert <= maxFertility; fert++) {
    long double fert_factor = (fertility_prob_[t_start].size() > fert) ? fertility_prob_[t_start][fert] : 0.0;
    if (fert > 1)
      fert_factor *= ld_fac_[fert];

    for (uint state = 0; state <= start_allfert_max_reachable_state; state++) 
      score[0](state,fert) *= fert_factor;
  }

  //compute fert_trace and best_prev_score
  for (uint state = 0; state <= start_allfert_max_reachable_state; state++) {

    long double best_score = 0.0;
    uchar best_fert = 255;
    
    for (uint fert=0; fert <= maxFertility; fert++) {
      const long double cur_score = score[0](state,fert);

      if (cur_score > best_score) {
        best_score = cur_score;
        best_fert = fert;
      }
    }

    best_prev_score[state] = best_score;
    fert_trace[0][state] = best_fert;
  }

  /**** now proceeed with the remainder of the sentence ****/

  for (uint i=1; i < curI; i++) {
    std::cerr << "********* i: " << i << " ***************" << std::endl;

    const uint ti = cur_target[i];
    const Math1D::Vector<double>& cur_dict = dict_[ti];

    Math1D::NamedVector<long double> translation_cost(curJ+1,MAKENAME(translation_cost));
    translation_cost[0] = 0.0; //we do not allow an empty word here
    for (uint j=1; j <= curJ; j++) {
      translation_cost[j] = cur_dict[cur_lookup(j-1,i)] * cur_distort_prob(j-1,i);
    }

    const uint allfert_max_reachable_j  = std::min(curJ, nMaxSkips + (i+1)*maxFertility);
    const uint fertone_max_reachable_j  = std::min(curJ, nMaxSkips + i*maxFertility+1);
    const uint prevword_max_reachable_j = std::min(curJ, nMaxSkips + i*maxFertility);
    
    const uint prevword_max_reachable_state = first_state_[prevword_max_reachable_j+1]-1;
    const uint allfert_max_reachable_state = first_state_[allfert_max_reachable_j+1]-1;
    const uint fertone_max_reachable_state = first_state_[fertone_max_reachable_j+1]-1;

    fert_trace[i].set_name("fert_trace["+toString(i)+"]");
    fert_trace[i].resize(allfert_max_reachable_state+1,255);

    state_trace[i].set_name("state_trace["+toString(i)+"]");
    state_trace[i].resize(allfert_max_reachable_state+1,maxFertility,255);

    score[i-1].resize(0,0);
    score[i].set_name("score["+toString(i)+"]");
    
    Math2D::NamedMatrix<long double>& cur_score = score[i];
    cur_score.resize(nStates,maxFertility+1,0.0);

    Math2D::NamedMatrix<uchar>& cur_state_trace = state_trace[i];    

    //fertility 0
    for (uint state=0; state <= prevword_max_reachable_state; state++) {
      cur_score(state,0) = best_prev_score[state];
    }

    //std::cerr << " fert 1" << std::endl; 

    //fertility 1
    for (uint state=0; state <= fertone_max_reachable_state; state++) {

      assert(coverage_state_(1,state) <= curJ);

      long double best_score = 0.0;
      uchar trace_entry = 255;
	
      const uint nPredecessors = predecessor_coverage_states_[state].yDim();
      assert(nPredecessors < 255);	
      
      for (uint p=0; p < nPredecessors; p++) {
	
        const uint prev_state = predecessor_coverage_states_[state](0,p);
        const uint cover_j = predecessor_coverage_states_[state](1,p);
	
        assert(cover_j <= curJ);
	
        const long double hyp_score = best_prev_score[prev_state] * translation_cost[cover_j];
	
        if (hyp_score > best_score) {
          best_score = hyp_score;
          trace_entry = p;
        }
      }
      
      cur_score(state,1) = best_score;
      cur_state_trace(state,0) = trace_entry;
    }

    //fertility > 1
    for (uint fert=2; fert <= maxFertility; fert++) {

      const uint curfert_max_reachable_j  = std::min(curJ, nMaxSkips + i*maxFertility + fert);
      const uint curfert_max_reachable_state = first_state_[curfert_max_reachable_j+1]-1;

      for (uint state=0; state <= curfert_max_reachable_state; state++) {

        assert(coverage_state_(1,state) <= curJ);

        long double best_score = 0.0;
        uchar trace_entry = 255;
	  
        const uint nPredecessors = predecessor_coverage_states_[state].yDim();
        assert(nPredecessors < 255);
	
        for (uint p=0; p < nPredecessors; p++) {
	  
          const uint prev_state = predecessor_coverage_states_[state](0,p);
          const uint cover_j = predecessor_coverage_states_[state](1,p);
	  
          assert(cover_j <= curJ);
  
          const long double hyp_score = cur_score(prev_state,fert-1) * translation_cost[cover_j];
	  
          if (hyp_score > best_score) {
            best_score = hyp_score;
            trace_entry = p;
          }
        }
	
        cur_score(state,fert) = best_score;
        cur_state_trace(state,fert-1) = trace_entry;
      }
    }

    //std::cerr << "including fertility probs" << std::endl;

    //include fertility probabilities
    for (uint fert = 0; fert <= maxFertility; fert++) {
      long double fert_factor = (fertility_prob_[ti].size() > fert) ? fertility_prob_[ti][fert] : 0.0;
      if (fert > 1)
        fert_factor *= ld_fac_[fert];
      
      for (uint state=0; state <= allfert_max_reachable_state; state++) 
        cur_score(state,fert) *= fert_factor;
    }


    //compute fert_trace and best_prev_score
    for (uint state=0; state <= allfert_max_reachable_state; state++) {

      long double best_score = 0.0;
      uchar best_fert = 255;
      
      for (uint fert=0; fert <= maxFertility; fert++) {
        const long double cand_score = cur_score(state,fert);
	
        if (cand_score > best_score) {
          best_score = cand_score;
          best_fert = fert;
        }
      }
      
      best_prev_score[state] = best_score;
      fert_trace[i][state] = best_fert;
    }
  }

  uint end_state = first_state_[curJ];
  assert(coverage_state_(0,end_state) == 0);
  assert(coverage_state_(1,end_state) == curJ);
  long double best_score = best_prev_score[end_state];
  ushort best_end_fert = fert_trace[curI-1][end_state];

  /**** traceback ****/
  best_known_alignment_[s].set_constant(0);

  uint fert = best_end_fert;
  uint i = curI-1;
  uint state = end_state;

  while (true) {

    //default values apply to the case with fertility 0
    uint prev_i = i-1;
    uint prev_state = state;

    if (fert > 0) {
      
      uint covered_j = coverage_state_(1,state);

      if (i > 0 || fert > 1) {
        const uchar transition = state_trace[i](state,fert-1);
        assert(transition != 255);
	
        prev_state = predecessor_coverage_states_[state](0,transition);
        covered_j = predecessor_coverage_states_[state](1,transition);
      }

      best_known_alignment_[s][covered_j-1] = i+1;
    
      if (fert > 1)
        prev_i = i;
    }

    if (i == 0 && fert <= 1)
      break;

    //default value applies to the case with fertility > 1
    uint prev_fert = fert-1;    

    if (fert <= 1)
      prev_fert = fert_trace[prev_i][prev_state];

    fert = prev_fert;
    i = prev_i;
    state = prev_state;
  }

  return best_score;
}

#ifdef HAS_CBC
class IBM3IPHeuristic : public CbcHeuristic {
public:

  IBM3IPHeuristic(CbcModel& model, uint I, uint J, uint nFertilityVarsPerWord);

  IBM3IPHeuristic(const CbcHeuristic& heuristic, CbcModel& model, 
                  uint I, uint J, uint nFertilityVarsPerWord);


  virtual CbcHeuristic* clone() const;

  virtual void resetModel(CbcModel* model);

  virtual int solution(double& objectiveValue, double* newSolution);

protected:

  uint I_;
  uint J_;
  uint nFertilityVarsPerWord_;
};


IBM3IPHeuristic::IBM3IPHeuristic(CbcModel& model, uint I, uint J, uint nFertilityVarsPerWord) :
  CbcHeuristic(model), I_(I), J_(J), nFertilityVarsPerWord_(nFertilityVarsPerWord)
{}

IBM3IPHeuristic::IBM3IPHeuristic(const CbcHeuristic& heuristic, CbcModel& /*model*/, uint I, uint J, uint nFertilityVarsPerWord) :
  CbcHeuristic(heuristic), I_(I), J_(J), nFertilityVarsPerWord_(nFertilityVarsPerWord)
{}

/*virtual*/ CbcHeuristic* IBM3IPHeuristic::clone() const {
  return new IBM3IPHeuristic(*this,*model_,I_,J_,nFertilityVarsPerWord_);
}

/*virtual*/ void IBM3IPHeuristic::resetModel(CbcModel* /*model*/) {
  TODO("resetModel");
}

/*virtual*/ int IBM3IPHeuristic::solution(double& objectiveValue, double* newSolution) {

  uint nVars = (I_+1)*J_ + (I_+1)*nFertilityVarsPerWord_;

  for (uint v=0; v < nVars; v++) {
    newSolution[v] = 0.0;
  }

  OsiSolverInterface* solver = model_->solver();
  const double* cur_solution = solver->getColSolution();

  Math1D::NamedVector<uint> fert_count(I_+1,0,MAKENAME(fert_count));

  for (uint j=0; j < J_; j++) {

    double max_var = 0.0;
    uint aj = MAX_UINT;

    for (uint i=0; i <= I_; i++) {

      double var_val = cur_solution[j*(I_+1)+i];

      if (var_val > max_var) {
        max_var = var_val;
        aj = i;
      }
    }

    newSolution[j*(I_+1)+aj] = 1.0;

    fert_count[aj]++;
  }
  
  uint fert_var_offs = J_*(I_+1);

  for (uint i=0; i <= I_; i++) {

    newSolution[fert_var_offs + i*nFertilityVarsPerWord_ + fert_count[i]] = 1.0;
  }


  uint return_code = 0;

  const double* objective = solver->getObjCoefficients();

  double new_energy = 0.0;
  for (uint v=0; v < nVars; v++)
    new_energy += newSolution[v] * objective[v];


  if (new_energy < objectiveValue) {

    return_code = 1;
  }

  return return_code;
}
#endif

long double IBM3Trainer::compute_viterbi_alignment_ilp(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                                       const SingleLookupTable& cur_lookup, 
                                                       Math1D::Vector<AlignBaseType>& alignment, double time_limit) {

#ifdef HAS_CBC
  const Storage1D<uint>& cur_source = source;
  const Storage1D<uint>& cur_target = target;
  
  const uint curI = cur_target.size();
  const uint curJ = cur_source.size();
  const Math2D::Matrix<double>& cur_distort_prob = distortion_prob_[curJ-1];

  uint nFertVarsPerWord = std::min(curJ,fertility_limit_) + 1;

  uint nVars = (curI+1)*curJ  //alignment variables
    + (curI+1)*nFertVarsPerWord; //fertility variables
  
  uint fert_var_offs = (curI+1)*curJ;

  uint nConstraints = curJ // alignment variables must sum to 1 for each source word
    + curI + 1 //fertility variables must sum to 1 for each target word including the empty word
    + curI + 1; //fertility variables must be consistent with alignment variables

  uint fert_con_offs = curJ;
  uint consistency_con_offs = fert_con_offs + curI + 1;
  
  Math1D::NamedVector<double> cost(nVars,0.0,MAKENAME(cost));

  Math1D::NamedVector<double> var_lb(nVars,0.0,MAKENAME(var_lb));
  Math1D::NamedVector<double> var_ub(nVars,1.0,MAKENAME(var_ub));

  Math1D::NamedVector<double> jcost_lower_bound(curJ,MAKENAME(jcost_lower_bound));
  Math1D::NamedVector<double> icost_lower_bound(curI+1,MAKENAME(icost_lower_bound));

  //code cost entries for alignment variables
  for (uint j=0; j < curJ; j++) {

    uint s_idx = cur_source[j];
    const uint cur_offs = j*(curI+1);

    double min_cost = 1e50;

    for (uint i=0; i <= curI; i++) {

      if (i == 0) {
	
        cost[cur_offs] = - std::log(dict_[0][s_idx-1]); //distortion is here handled by the fertilities
        assert(!isnan(cost[cur_offs]));
	
      }
      else {
        const uint ti = cur_target[i-1];
	
        cost[cur_offs + i] = -std::log(dict_[ti][cur_lookup(j,i-1) ]) - std::log( cur_distort_prob(j,i-1) );
        assert(!isnan(cost[cur_offs+i]));
      }

      if (cost[cur_offs + i] < min_cost)
        min_cost = cost[cur_offs + i];
    }

    jcost_lower_bound[j] = min_cost;
  }
  
  //code cost entries for the fertility variables of the empty word
  double min_empty_fert_cost = 1e50;
  for (uint fert=0; fert < nFertVarsPerWord; fert++) {

    if (curJ-fert >= fert) {
      long double prob = 1.0;
      
      prob *= ldchoose(curJ-fert,fert);
      for (uint k=1; k <= fert; k++)
        prob *= p_zero_;
      for (uint k=1; k <= curJ-2*fert; k++)
        prob *= p_nonzero_;
    
      if (och_ney_empty_word_) {
	
        for (uint k=1; k<= fert; k++)
          prob *= ((long double) k) / curJ;
      }
      
      if (prob >= 1e-300) {
        cost[fert_var_offs + fert] = - logl( prob );
        assert(!isnan(cost[fert_var_offs+fert]));
      }
      else {
        cost[fert_var_offs + fert] = 1e10;
      }

      if (cost[fert_var_offs + fert] < min_empty_fert_cost) {
        min_empty_fert_cost = cost[fert_var_offs + fert];
      }
    }
    else {      
      cost[fert_var_offs + fert] = 1e10;
      var_ub[fert_var_offs + fert] = 0.0;
    }
  }
  icost_lower_bound[0] = min_empty_fert_cost;

  //code cost entries for the fertility variables of the real words
  for (uint i=0; i < curI; i++) {

    const uint ti = cur_target[i];

    double min_cost = 1e50;

    for (uint fert=0; fert < nFertVarsPerWord; fert++) {

      uint idx = fert_var_offs + (i+1)*nFertVarsPerWord + fert;

      if (fertility_prob_[ti][fert] > 1e-75) {
        cost[idx] = -logl( ld_fac_[fert] * fertility_prob_[ti][fert]  );
        assert(!isnan(cost[fert_var_offs + (i+1)*nFertVarsPerWord + fert]));
      }
      else 
        cost[idx] = 1e10;

      if (cost[idx] < min_cost)
        min_cost = cost[idx];
    }

    icost_lower_bound[i+1] = min_cost;
  }

  Math2D::NamedMatrix<double> ifert_cost(curJ+1,curI+1,1e50,MAKENAME(ifert_cost));
  
  for (uint f=0; f < nFertVarsPerWord; f++) {
    ifert_cost(f,0) = cost[fert_var_offs + f];
  }

  for (uint i = 1; i <= curI; i++) {

    for (uint j=0; j <= curJ; j++) {

      double opt_cost = ifert_cost(j,i-1) + cost[fert_var_offs+i*nFertVarsPerWord];

      for (uint f=1; f < std::min(j+1,nFertVarsPerWord); f++) {
	
        double hyp_cost = ifert_cost(j-f,i-1) + cost[fert_var_offs+i*nFertVarsPerWord + f];

        if (hyp_cost < opt_cost) {
          opt_cost = hyp_cost;
        }
      }

      ifert_cost(j,i) = opt_cost;
    }
  }

  uint nHighCost = 0;

  double upper_bound = -logl(alignment_prob(cur_source,cur_target,cur_lookup,alignment));
  double lower_bound = jcost_lower_bound.sum() + ifert_cost(curJ,curI);
  
  double loose_lower_bound = jcost_lower_bound.sum() + icost_lower_bound.sum();
  
  double gap = upper_bound - lower_bound;
  double loose_gap = upper_bound - loose_lower_bound;

  double loose_bound2 = 0.0;
  Math1D::Vector<double> values(curJ);
  Math1D::Vector<double> ibound2(curI+1);
  Math2D::Matrix<double> approx_icost(curI+1,nFertVarsPerWord);

  for (uint i=0; i <= curI; i++) {

    for (uint j=0; j < curJ; j++)
      values[j] = cost[j*(curI+1)+i] - jcost_lower_bound[j];

    std::sort(values.direct_access(),values.direct_access()+curJ);

    for (uint j=1; j < curJ; j++)
      values[j] += values[j-1];
   
    approx_icost(i,0) = cost[fert_var_offs+i*nFertVarsPerWord];

    double min_cost = approx_icost(i,0);

    for (uint f=1; f < nFertVarsPerWord; f++) {

      if (i == 0)
        approx_icost(i,f) = cost[fert_var_offs+f] + values[f-1];
      else
        approx_icost(i,f) = cost[fert_var_offs + nFertVarsPerWord + (i-1)*nFertVarsPerWord + f] + values[f-1];

      if (approx_icost(i,f) < min_cost)
        min_cost = approx_icost(i,f);
    }
    ibound2[i] = min_cost;

    loose_bound2 += min_cost;
  }

  double loose_gap2 = upper_bound + 0.1 - loose_bound2;  

#if 1
  for (uint j=0; j < curJ; j++) {
    
    for (uint aj=0; aj <= curI; aj++) {
      if (cost[j*(curI+1) + aj] >= gap+jcost_lower_bound[j]+0.1) {

        var_ub[j*(curI+1) + aj] = 0.0;
        nHighCost++;
      }
    }
  }

  for (uint i=0; i <= curI; i++) {

    for (uint f=0; f < nFertVarsPerWord; f++) {
      
      if (cost[fert_var_offs+i*nFertVarsPerWord + f] >= icost_lower_bound[i] + loose_gap + 0.1) {
        var_ub[fert_var_offs+i*nFertVarsPerWord + f] = 0.0;
        nHighCost++;
      }
      else if (approx_icost(i,f) >= loose_gap2 + ibound2[i]) {
        var_ub[fert_var_offs+i*nFertVarsPerWord + f] = 0.0;
        nHighCost++;
      }
    }
  }
#endif


  for (uint v=0; v < nVars; v++) {

    assert(!isnan(cost[v]));

    if (cost[v] > 1e10) {
      nHighCost++;
      cost[v] = 1e10;
      var_ub[v] = 0.0;
    }
  }

  Math1D::NamedVector<double> rhs(nConstraints,1.0,MAKENAME(rhs));
  
  for (uint c=consistency_con_offs; c < nConstraints; c++)
    rhs[c] = 0.0;

  //code matrix constraints
  uint nMatrixEntries = (curI+1)*curJ // for the alignment unity constraints
    + (curI+1)*(nFertVarsPerWord+1) // for the fertility unity constraints
    + (curI+1)*(nFertVarsPerWord-1+curJ); // for the consistency constraints

  SparseMatrixDescription<double> lp_descr(nMatrixEntries, nConstraints, nVars);

  //code unity constraints for alignment variables
  for (uint j=0; j < curJ; j++) {
  
    for (uint v= j*(curI+1); v < (j+1)*(curI+1); v++) {
      if (var_ub[v] > 0.0)
        lp_descr.add_entry(j, v, 1.0);
    }
  }


  //code unity constraints for fertility variables
  for (uint i=0; i <= curI; i++) {

    for (uint fert=0; fert < nFertVarsPerWord; fert++) {

      lp_descr.add_entry(fert_con_offs + i, fert_var_offs + i*nFertVarsPerWord + fert, 1.0 );
    }
  }

  //code consistency constraints
  for (uint i=0; i <= curI; i++) {

    uint row = consistency_con_offs + i;

    for (uint fert=1; fert < nFertVarsPerWord; fert++) {

      uint col = fert_var_offs + i*nFertVarsPerWord + fert;

      lp_descr.add_entry(row, col, fert);

    }

    for (uint j=0; j < curJ; j++) {

      uint col = j*(curI+1) + i;

      if (var_ub[col] > 0.0)
        lp_descr.add_entry(row, col, -1.0);
    }
  }

  CoinPackedMatrix coinMatrix(false,(int*) lp_descr.row_indices(),(int*) lp_descr.col_indices(),
                              lp_descr.value(),lp_descr.nEntries());

  OsiClpSolverInterface clp_interface;

  clp_interface.setLogLevel(0);
  clp_interface.messageHandler()->setLogLevel(0);

  clp_interface.loadProblem (coinMatrix, var_lb.direct_access(), var_ub.direct_access(),   
                             cost.direct_access(), rhs.direct_access(), rhs.direct_access());

  for (uint v=0; v < nVars; v++) {
    clp_interface.setInteger(v);
  }

  //std::clock_t tStartCLP, tEndCLP;
  
  //tStartCLP = std::clock();

  int error = 0; 
  clp_interface.resolve();
  error =  1 - clp_interface.isProvenOptimal();

  if (error) {
    INTERNAL_ERROR << "solving the LP-relaxation failed. Exiting..." << std::endl;
    exit(1);
  }

  //tEndCLP = std::clock();
  //std::cerr << "CLP-time: " << diff_seconds(tEndCLP,tStartCLP) << " seconds. " << std::endl;

  const double* lp_solution = clp_interface.getColSolution(); 
  long double energy = 0.0;

  uint nNonIntegral = 0;
  uint nNonIntegralFert = 0;

  for (uint v=0; v < nVars; v++) {

    double var_val = lp_solution[v];

    if (var_val > 0.01 && var_val < 0.99) {
      nNonIntegral++;

      if (v >= fert_var_offs)
        nNonIntegralFert++;
    }

    energy += cost[v] * var_val;
  }

  const double* solution = lp_solution;

  CbcModel cbc_model(clp_interface);

  if (nNonIntegral > 0) {

    clp_interface.setupForRepeatedUse();
    
    cbc_model.messageHandler()->setLogLevel(0);
    cbc_model.setLogLevel(0);

    if (time_limit > 0.0)
      cbc_model.setMaximumSeconds(time_limit);

    CglGomory gomory_cut;
    gomory_cut.setLimit(500);
    gomory_cut.setAway(0.01);
    gomory_cut.setLimitAtRoot(500);
    gomory_cut.setAwayAtRoot(0.01);
    cbc_model.addCutGenerator(&gomory_cut,0,"Gomory Cut");

    CglProbing probing_cut;
    probing_cut.setUsingObjective(true);
    probing_cut.setMaxPass(10);
    probing_cut.setMaxPassRoot(50);
    probing_cut.setMaxProbe(100);
    probing_cut.setMaxProbeRoot(500);
    probing_cut.setMaxLook(150);
    probing_cut.setMaxLookRoot(1500);
    //cbc_model.addCutGenerator(&probing_cut,0,"Probing Cut");
    
    CglRedSplit redsplit_cut;
    redsplit_cut.setLimit(1500);
    //cbc_model.addCutGenerator(&redsplit_cut,0,"RedSplit Cut");

    //CglMixedIntegerRounding mi1_cut;
    //cbc_model.addCutGenerator(&mi1_cut,0,"Mixed Integer Cut 1");

    CglMixedIntegerRounding2 mi2_cut;
    //cbc_model.addCutGenerator(&mi2_cut,0,"Mixed Integer 2");

    CglTwomir twomir_cut;
    //cbc_model.addCutGenerator(&twomir_cut,0,"Twomir Cut");

    CglLandP landp_cut;
    //cbc_model.addCutGenerator(&landp_cut,0,"LandP Cut");

    CglOddHole oddhole_cut;
    //cbc_model.addCutGenerator(&oddhole_cut,0,"OddHole Cut");

    //CglClique clique_cut;
    //cbc_model.addCutGenerator(&clique_cut,0,"Clique Cut");

    //CglStored stored_cut;
    //cbc_model.addCutGenerator(&stored_cut,0,"Stored Cut");
    
    IBM3IPHeuristic  ibm3_heuristic(cbc_model, curI, curJ, nFertVarsPerWord);
    ibm3_heuristic.setWhereFrom(63);
    cbc_model.addHeuristic(&ibm3_heuristic,"IBM3 Heuristic");

    /*** set initial upper bound given by best_known_alignment_[s] ****/
    Math1D::Vector<double> best_sol(nVars,0.0);
    Math1D::Vector<uint> fert_count(curI+1,0);
    
    for (uint j=0; j < curJ; j++) {
      
      uint aj = alignment[j];
      fert_count[aj]++;
      best_sol[ j*(curI+1)+aj] = 1.0;
    } 
    for (uint i=0; i <= curI; i++) {
      best_sol[fert_var_offs + i*nFertVarsPerWord + fert_count[i] ] = 1.0;
    }
    
    double temp_energy = 0.0;
    for (uint v=0; v < nVars; v++)
      temp_energy += cost[v]*best_sol[v];

    cbc_model.setBestSolution(best_sol.direct_access(),nVars,temp_energy,true);

    cbc_model.branchAndBound();

    const double* cbc_solution = cbc_model.bestSolution();

    if (!cbc_model.isProvenOptimal()) {

      std::cerr << "ERROR: the optimal solution could not be found. Exiting..." << std::endl;
      exit(1);
    }
    if (cbc_model.isProvenInfeasible()) {

      std::cerr << "ERROR: problem marked as infeasible. Exiting..." << std::endl;
      exit(1);
    }
    if (cbc_model.isAbandoned()) {

      std::cerr << "ERROR: problem was abandoned. Exiting..." << std::endl;
      exit(1);
    }

    energy = 0.0;

    for (uint v=0; v < nVars; v++) {
      
      double var_val = cbc_solution[v];
      
      energy += cost[v] * var_val;
    }

    solution = cbc_solution;
  }

  alignment.resize(curJ);

  uint nNonIntegralVars = 0;

  Math1D::Vector<uint> fert(curI+1,0);

  for (uint j=0; j < curJ; j++) {

    double max_val = 0.0;
    uint arg_max = MAX_UINT;

    for (uint i=0; i <= curI; i++) {

      double val = solution[j*(curI+1)+i];
      
      if (val > 0.01 && val < 0.99)
        nNonIntegralVars++;

      if (val > max_val) {

        max_val = val;
        arg_max = i;
      }
    }

    alignment[j] = arg_max;
    fert[arg_max]++;
  }
  
  long double clp_prob =  expl(-energy);
  long double actual_prob = alignment_prob(cur_source,cur_target,cur_lookup,alignment);

  return actual_prob;
#else
  return alignment_prob(source,target,cur_lookup,alignment);
#endif
}

void IBM3Trainer::train_with_ibm_constraints(uint nIter, uint maxFertility, uint nMaxSkips, bool verbose) {


  ReducedIBM3DistortionModel fdistort_count(distortion_prob_.size(),MAKENAME(fdistort_count));
  for (uint J=0; J < fdistort_count.size(); J++) {
    fdistort_count[J].resize_dirty(distortion_prob_[J].xDim(), distortion_prob_[J].yDim());
  }

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords,MAKENAME(ffert_count));

  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  long double fzero_count;
  long double fnonzero_count;

  SingleLookupTable aux_lookup;

  compute_uncovered_sets(nMaxSkips); 
  compute_coverage_states();

  for (uint iter=1+iter_offs_; iter <= nIter+iter_offs_; iter++) {

    std::cerr << "******* IBM-3 IBM-constraint-iteration #" << iter << std::endl;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    for (uint J=0; J < distortion_prob_.size(); J++) {
      fdistort_count[J].set_constant(0.0);
    }

    double max_perplexity = 0.0;

    uint nBetter = 0;
    uint nEqual = 0;

    for (size_t s=0; s < source_sentence_.size(); s++) {

      long double prev_prob = (verbose) ? alignment_prob(s,best_known_alignment_[s]) : 0.0;

      long double prob = compute_ibmconstrained_viterbi_alignment_noemptyword(s,maxFertility,nMaxSkips);
      prob *= pow(p_nonzero_,source_sentence_[s].size());
      std::cerr << "probability " << prob << std::endl;
      std::cerr << "generated alignment: " << best_known_alignment_[s] << std::endl;

      max_perplexity -= std::log(prob);

      if (prob < 1e-305)
        continue;

      long double check_prob = alignment_prob(s,best_known_alignment_[s]);
      double check_ratio = prob / check_prob;
      std::cerr << "check_ratio: " << check_ratio << std::endl;
      assert(check_ratio > 0.999 && check_ratio < 1.001);

      if (verbose) {
        if (prev_prob == check_prob)
          nEqual++;
        else if (prev_prob < check_prob)
          nBetter++;
      }

      const Storage1D<uint>&  cur_source = source_sentence_[s];
      const Storage1D<uint>&  cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc_,
                                                           nSourceWords_,slookup_[s],aux_lookup);


      const uint curJ = source_sentence_[s].size();
      const uint curI = target_sentence_[s].size();
 
      Math2D::Matrix<double>& cur_distort_count = fdistort_count[curJ-1];

      Math1D::Vector<uint> fertility(curI+1,0);
      for (uint j=0; j < curJ; j++) {
        fertility[best_known_alignment_[s][j]]++;
      }

      //currently implementing Viterbi training

      double cur_zero_weight = 1.0;
      
      fzero_count += cur_zero_weight * (fertility[0]);
      fnonzero_count += cur_zero_weight * (curJ - 2*fertility[0]);

      //increase counts for dictionary and distortion
      for (uint j=0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = best_known_alignment_[s][j];

        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj-1]][cur_lookup(j,cur_aj-1)] += 1;
          cur_distort_count(j,cur_aj-1) += 1.0;
          assert(!isnan(cur_distort_count(j,cur_aj-1)));
        }
        else {
          fwcount[0][s_idx-1] += 1;
        }
      }

      //update fertility counts
      for (uint i=1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i-1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }
    }


    //update p_zero_ and p_nonzero_
    double fsum = fzero_count + fnonzero_count;
    p_zero_ = fzero_count / fsum;
    p_nonzero_ = fnonzero_count / fsum;

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, 0.0, iter,  false, 0.0, 0, dict_, 1e-6);

    //update distortion prob from counts
    for (uint J=0; J < distortion_prob_.size(); J++) {

      for (uint i=0; i < distortion_prob_[J].yDim(); i++) {

        double sum = 0.0;
        for (uint j=0; j < J+1; j++)
          sum += fdistort_count[J](j,i);

        if (sum > 1e-305) {
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint j=0; j < J+1; j++) {
            distortion_prob_[J](j,i) = inv_sum * fdistort_count[J](j,i);
            if (isnan(distortion_prob_[J](j,i))) {
              std::cerr << "sum: " << sum << std::endl;
              std::cerr << "set to " << inv_sum << " * " << fdistort_count[J](j,i) << " = "
                        << (inv_sum * fdistort_count[J](j,i)) << std::endl;
            }
            assert(!isnan(fdistort_count[J](j,i)));
            assert(!isnan(distortion_prob_[J](j,i)));
          }
        }
        else {
          //std::cerr << "WARNING: did not update distortion count because sum was " << sum << std::endl;
        }
      }
    }

    update_fertility_prob(ffert_count);

    if (possible_ref_alignments_.size() > 0) {
      
      std::cerr << "#### IBM3-AER in between iterations #" << (iter-1) << " and " << iter << ": " << AER() << std::endl;
      std::cerr << "#### IBM3-fmeasure in between iterations #" << (iter-1) << " and " << iter << ": " << f_measure() << std::endl;
      std::cerr << "#### IBM3-DAE/S in between iterations #" << (iter-1) << " and " << iter << ": " 
                << DAE_S() << std::endl;
    }

    if (verbose) {
      std::cerr << "ibm-constraints are eqaul to hillclimbing in " << nEqual << " cases" << std::endl;
      std::cerr << "ibm-constraints are better than hillclimbing in " << nBetter << " cases" << std::endl;
    }    

    max_perplexity /= source_sentence_.size();

    std::cerr << "max-fertility after iteration #" << (iter - 1) << ": " << max_perplexity << std::endl;
  }
}


void add_nondef_count(const Storage1D<std::vector<uchar> >& aligned_source_words, uint i, uint J,
		      std::map<Math1D::Vector<uchar,uchar>,double>& count_map, double count) {

  //note: i starts at zero, but the index for aligned_source_words starts at 1
  
  if (aligned_source_words[i+1].empty()) {
    std::cerr << "WARNING: useless call!" << std::endl;
    return;
  }

  Storage1D<bool> fixed(J,false);
  for (uint ii=0; ii < i; ii++) {
    const std::vector<uchar>& words = aligned_source_words[ii+1];
    for (uint k=0; k < words.size(); k++)
      fixed[words[k]] = true;
  }

  uint prev_j = MAX_UINT;
   
  for (uint k=0; k < aligned_source_words[i+1].size(); k++) {
    
    uint j = aligned_source_words[i+1][k];

    uint nFollowing = aligned_source_words[i+1].size() - k - 1;

    uint pos = MAX_UINT;

    const uint start = (prev_j == MAX_UINT) ? 0 : prev_j+1;

    std::vector<uchar> possible;
    possible.reserve(J);
    for (uint jj=start; jj < J; jj++) {
      if (!fixed[jj]) {
        if (jj == j)
          pos = possible.size();
        possible.push_back(jj);
      }
    }

    assert(pos != MAX_UINT);
    
    if (nFollowing > 0)
      possible.resize(possible.size()-nFollowing);

    if (possible.size() > 1) { //no use adding counts when there is only one position left

      if (pos != possible.size()-1) // the chosen position must be the last listed one
        std::swap(possible[pos],possible[possible.size()-1]);

      Math1D::Vector<uchar,uchar> vec_possible(possible.size());
      assign(vec_possible,possible);
      
      count_map[vec_possible] += count;
    }

    fixed[j] = true;
    prev_j = j;
  }
}




void add_nondef_count(const Storage1D<std::vector<uchar> >& aligned_source_words, uint i, uint J,
		      std::map<Math1D::Vector<uchar,uchar>,Math1D::Vector<double,uchar> >& count_map, double count) {

  //note: i starts at zero, but the index for aligned_source_words starts at 1
  
  if (aligned_source_words[i+1].empty()) {
    std::cerr << "WARNING: useless call!" << std::endl;
    return;
  }

  Storage1D<bool> fixed(J,false);
  for (uint ii=0; ii < i; ii++) {
    const std::vector<uchar>& words = aligned_source_words[ii+1];
    for (uint k=0; k < words.size(); k++)
      fixed[words[k]] = true;
  }

  uint prev_j = MAX_UINT;
   
  for (uint k=0; k < aligned_source_words[i+1].size(); k++) {
    
    uint j = aligned_source_words[i+1][k];

    uint nFollowing = aligned_source_words[i+1].size() - k - 1;

    const uint start = (prev_j == MAX_UINT) ? 0 : prev_j+1;

    uint pos = MAX_UINT;
    
    std::vector<uchar> possible;
    possible.reserve(J);
    for (uint jj=start; jj < J; jj++) {
      if (!fixed[jj]) {
        if (jj == j)
          pos = possible.size();
        possible.push_back(jj);
      }
    }
    
    if (nFollowing > 0)
      possible.resize(possible.size()-nFollowing);

    if (possible.size() > 1) { //no use adding counts when there is only one position left

      Math1D::Vector<uchar,uchar> vec_possible(possible.size());
      assign(vec_possible,possible);

      std::map<Math1D::Vector<uchar,uchar>,Math1D::Vector<double,uchar> >::iterator it = count_map.find(vec_possible);

      if (it != count_map.end())
	it->second[pos] += count;
      else {
	//avoid double access to count_map

	Math1D::Vector<double,uchar> temp(vec_possible.size(),0.0);
	temp[pos] = count;

	count_map[vec_possible] = temp;
      }
    }

    fixed[j] = true;
    prev_j = j;
  }
}
