/*** written by Thomas Schoenemann as a private person without employment, October 2009 
 *** later as an employee of Lund University, 2010 - Mar. 2011
 *** and later as a private person ***/

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

double extended_hmm_perplexity(const Storage1D<Storage1D<uint> >& source, 
                               const Storage1D<Math2D::Matrix<uint> >& slookup,
                               const Storage1D<Storage1D<uint> >& target,
                               const FullHMMAlignmentModel& align_model,
                               const InitialAlignmentProbability& initial_prob,
                               const SingleWordDictionary& dict,
			       HmmAlignProbType align_type = HmmAlignProbNonpar) {

  double sum = 0.0;

  const size_t nSentences = target.size();
  
  for (size_t s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const Math2D::Matrix<uint>& cur_lookup = slookup[s];
    
    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();
    
    const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
    
    /**** calculate forward ********/

    Math2D::NamedMatrix<double> forward(2*curI,curJ,MAKENAME(forward));

    if (align_type == HmmAlignProbReducedpar) {

      calculate_hmm_forward_with_tricks(cur_source, cur_target, cur_lookup, dict, cur_align_model,
					initial_prob[curI-1], forward);
    }
    else {

      calculate_hmm_forward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
			    initial_prob[curI-1], forward);
    }

    double sentence_prob = 0.0;
    for (uint i=0; i < 2*curI; i++) {
      
      assert(forward(i,curJ-1) >= 0.0);
      sentence_prob += forward(i,curJ-1);
    }

    if (sentence_prob > 1e-300)
      sum -= std::log(sentence_prob);
    else
      sum -= std::log(1e-300);
  }

  return sum / nSentences;
}


double extended_hmm_energy(const Storage1D<Storage1D<uint> >& source, 
                           const Storage1D<Math2D::Matrix<uint> >& slookup,
                           const Storage1D<Storage1D<uint> >& target,
                           const FullHMMAlignmentModel& align_model,
                           const InitialAlignmentProbability& initial_prob,
                           const SingleWordDictionary& dict,
                           const floatSingleWordDictionary& prior_weight,
			   HmmAlignProbType align_type = HmmAlignProbNonpar) {
  
  double energy = 0.0;

  for (uint i=0; i < dict.size(); i++)
    for (uint k=0; k < dict[i].size(); k++) {
      energy += prior_weight[i][k] * dict[i][k];
    }

  energy /= source.size();

  energy += extended_hmm_perplexity(source,slookup,target,align_model,initial_prob,dict,align_type);

  return energy;
}

double ehmm_m_step_energy(const FullHMMAlignmentModel& facount, const Math1D::Vector<double>& dist_params, 
                          uint zero_offset, double grouping_param = -1.0) {

  double energy = 0.0;

  //std::cerr << "grouping_param: " << grouping_param << std::endl;
  
  for (uint I=1; I <= facount.size(); I++) {

    if (facount[I-1].size() > 0) {
      
      for (int i=0; i < (int) I; i++) {

        double non_zero_sum = 0.0;

        if (grouping_param < 0.0) {
	
          for (uint ii=0; ii < I; ii++)
            non_zero_sum += dist_params[zero_offset + ii - i];
	  
          for (int ii=0; ii < (int) I; ii++) {
	    
            const double cur_count = facount[I-1](ii,i);
	    
            if (dist_params[zero_offset + ii - i] > 0.0)
              energy -= cur_count * std::log( dist_params[zero_offset + ii - i] / non_zero_sum);
            else
              energy += cur_count * 1200.0;
          }
        }
        else {

          double grouping_norm = std::max(0,i-5);
          grouping_norm += std::max(0,int(I)-1-(i+5));

          double check = 0.0;
	  
          for (int ii=0; ii < (int) I; ii++) {
            if (abs(ii-i) <= 5) 
              non_zero_sum += dist_params[zero_offset + ii - i];
            else {
              non_zero_sum += grouping_param / grouping_norm;
              check ++;
            }
          }

          assert(check == grouping_norm);

          for (int ii=0; ii < (int) I; ii++) {
	    
            double cur_count = facount[I-1](ii,i);
            double cur_param = dist_params[zero_offset + ii - i];

            if (abs(ii-i) > 5) {
              cur_param = grouping_param / grouping_norm;
            }

            if (cur_param > 0.0)
              energy -= cur_count * std::log( cur_param / non_zero_sum);
            else
              energy += cur_count * 1200.0;
          }
        }
      }
    }

    //std::cerr << "intermediate energy: " << energy << std::endl;
  }

  return energy;
}

void ehmm_m_step(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset,
                 uint nIter, double& grouping_param) {

  //std::cerr << "init params before projection: " << dist_params << std::endl;

  if (grouping_param < 0.0)
    projection_on_simplex(dist_params.direct_access(),dist_params.size());
  else
    projection_on_simplex_with_slack(dist_params.direct_access() + zero_offset - 5, grouping_param, 11); 

  //std::cerr << "init params after projection: " << dist_params << std::endl;
  
  Math1D::Vector<double> m_dist_grad = dist_params;
  Math1D::Vector<double> new_dist_params = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  double m_grouping_grad = 0.0;
  double new_grouping_param = 0.0;
  double hyp_grouping_param = 0.0;

  double energy = ehmm_m_step_energy(facount, dist_params, zero_offset, grouping_param);

  std::cerr << "start m-energy: " << energy << std::endl;

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "m-step gd-iter #" << iter << std::endl;

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
              non_zero_sum += dist_params[zero_offset + ii - i];
            else
              non_zero_sum += grouping_param / grouping_norm;
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
            else
              m_grouping_grad += count_sum / (non_zero_sum * grouping_norm);
          }
        }
      }
    }

    //std::cerr << "gradient: " << m_dist_grad << std::endl;

    //go in gradient direction
    double alpha  = 0.001;

    for (uint k=0; k < dist_params.size(); k++)
      new_dist_params.direct_access(k) = dist_params.direct_access(k) - alpha * m_dist_grad.direct_access(k);

    new_grouping_param = grouping_param - alpha * m_grouping_grad;

    // reproject
    if (grouping_param < 0.0)      
      projection_on_simplex(new_dist_params.direct_access(),dist_params.size());
    else
      projection_on_simplex_with_slack(new_dist_params.direct_access()+zero_offset-5,new_grouping_param,11);

    //find step-size

    double best_energy = ehmm_m_step_energy(facount, new_dist_params, zero_offset, new_grouping_param);
    
    double lambda = 1.0;
    double line_reduction_factor = 0.5;
    double best_lambda = lambda;

    uint nIter = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        std::cerr << "CUTOFF" << std::endl;
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint k=0; k < dist_params.size(); k++)
        hyp_dist_params.direct_access(k) = lambda * new_dist_params.direct_access(k) + neg_lambda * dist_params.direct_access(k);
      
      hyp_grouping_param = lambda * new_grouping_param + neg_lambda * grouping_param;

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

    if (nIter > 15) 
      break;
      
    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k=0; k < dist_params.size(); k++)
      dist_params.direct_access(k) = neg_best_lambda * dist_params.direct_access(k) + best_lambda * new_dist_params.direct_access(k);

    if (grouping_param >= 0.0) 
      grouping_param = best_lambda * new_grouping_param + neg_best_lambda * grouping_param;
  }  

}


double ehmm_init_m_step_energy(const InitialAlignmentProbability& init_acount, const Math1D::Vector<double>& init_params) {

  double energy = 0.0;

  for (uint I=0; I < init_acount.size(); I++) {

    if (init_acount[I].size() > 0) {

      double non_zero_sum = 0.0;
      for (uint i=0; i < I; i++)
        non_zero_sum += init_params[i];
      
      for (uint i=0; i < I; i++) {
	
        if (init_params[i] > 0.0)
          energy -= init_acount[I][i] * std::log( init_params[i] / non_zero_sum);
        else
          energy += init_acount[I][i] * 1200.0;
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

    std::cerr << "init m-step gd-iter #" << iter << std::endl;

    m_init_grad.set_constant(0.0);

    //calculate gradient
    for (uint I=0; I < init_acount.size(); I++) {

      if (init_acount[I].size() > 0) {

        double non_zero_sum = 0.0;
        for (uint i=0; i <= I; i++)
          non_zero_sum += init_params[i];
	
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
    projection_on_simplex(new_init_params.direct_access(),init_params.size());

    //find step-size
    double hyp_energy = ehmm_init_m_step_energy(init_acount, new_init_params);

    double lambda = 1.0;
    double line_reduction_factor = 0.5;

    while (hyp_energy > energy) {

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint k=0; k < init_params.size(); k++)
        hyp_init_params.direct_access(k) = lambda * new_init_params.direct_access(k) + neg_lambda * init_params.direct_access(k);
      
      hyp_energy = ehmm_init_m_step_energy(init_acount, hyp_init_params);

    }

    energy = hyp_energy;

    //set new values
    double best_lambda = lambda;
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k=0; k < init_params.size(); k++)
      init_params.direct_access(k) = neg_best_lambda * init_params.direct_access(k) + best_lambda * new_init_params.direct_access(k);
  }
}

void par2nonpar_hmm_init_model(const Math1D::Vector<double>& init_params, const Math1D::Vector<double>& source_fert,
                               HmmInitProbType init_type, InitialAlignmentProbability& initial_prob) {

  for (uint I=1; I <= initial_prob.size(); I++) {

    if (initial_prob[I-1].size() > 0) {

      if (init_type == HmmInitPar) {
 
        double norm = 0.0;
        for (uint i=0; i < I; i++)
          norm += init_params[i];
	
        double inv_norm = 1.0 / norm;
        for (uint i=0; i < I; i++)
          initial_prob[I-1][i] = source_fert[1] * inv_norm * init_params[i];
        for (uint i=I; i < 2*I; i++)
          initial_prob[I-1][i] = source_fert[0] / I;
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
	
        double inv_sum = 1.0 / non_zero_sum;

        for (int ii=0; ii < (int) I; ii++) {
          if (align_type == HmmAlignProbReducedpar && abs(ii-i) > 5) {
            align_model[I-1](ii,i) = source_fert[1] * inv_sum * dist_grouping_param / grouping_norm;
          }
          else
            align_model[I-1](ii,i) = source_fert[1] * inv_sum * dist_params[zero_offset + ii - i];
        }
        align_model[I-1](I,i) = source_fert[0];
      }
    }
  }
}

void train_extended_hmm(const Storage1D<Storage1D<uint> >& source, 
                        const Storage1D<Math2D::Matrix<uint> >& slookup,
                        const Storage1D<Storage1D<uint> >& target,
                        const CooccuringWordsType& wcooc,
                        uint nSourceWords, uint nTargetWords,
                        FullHMMAlignmentModel& align_model,
                        Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                        Math1D::Vector<double>& source_fert,
                        InitialAlignmentProbability& initial_prob,
                        Math1D::Vector<double>& init_params,
                        SingleWordDictionary& dict,
                        uint nIterations, HmmInitProbType init_type, HmmAlignProbType align_type,
                        std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                        std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                        const floatSingleWordDictionary& prior_weight,
                        bool smoothed_l0, double l0_beta) {

  std::cerr << "starting Extended HMM EM-training" << std::endl;

  dist_grouping_param = -1.0;

  double dict_weight_sum = 0.0;
  for (uint i=0; i < nTargetWords; i++) {
    dict_weight_sum += fabs(prior_weight[i].sum());
  }

  if (init_type >= HmmInitInvalid) {
    
    INTERNAL_ERROR << "invalid type for HMM initial alignment model" << std::endl;
    exit(1);
  }
  if (align_type >= HmmAlignProbInvalid) {

    INTERNAL_ERROR << "invalid type for HMM alignment model" << std::endl;
    exit(1);
  }

  assert(wcooc.size() == nTargetWords);
  //NOTE: the dictionary is assumed to be initialized

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

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

  Math1D::NamedVector<double> dist_count(0,MAKENAME(dist_count));

  double dist_grouping_count = -1.0;

  Math1D::Vector<double> source_fert_count(0);
  
  uint zero_offset = maxI-1;
  
  if (align_type != HmmAlignProbNonpar) {
    dist_params.resize(2*maxI-1, 1.0 / (2*maxI-1));
    dist_count.resize(2*maxI-1,0.0);

    if (align_type == HmmAlignProbReducedpar) {
      dist_grouping_param = 1.0;
    }
  }

  if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {
    source_fert.resize(2);
    source_fert[0] = 0.02;
    source_fert[1] = 0.98;
    source_fert_count.resize(2,0.0);
  }

  Math1D::NamedVector<double> init_count(0, MAKENAME(init_count) );
  if (init_type == HmmInitPar) {
    init_params.resize(maxI,1.0 / maxI);
    init_count.resize(maxI,0.0);
  }

  align_model.resize_dirty(maxI); //note: access using I-1
  initial_prob.resize(maxI);
  InitialAlignmentProbability ficount(maxI,MAKENAME(ficount));

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

      initial_prob[I-1].resize_dirty(2*I);
      if (init_type != HmmInitPar) 
        initial_prob[I-1].set_constant(0.5 / I);
      else {
        for (uint i=0; i < I; i++)
          initial_prob[I-1][i] = source_fert[1] / I;
        for (uint i=I; i < 2*I; i++)
          initial_prob[I-1][i] = source_fert[0] / I;
      }
      ficount[I-1].resize_dirty(2*I);
    }
  }

  Math1D::Vector<double> empty_count(maxI,0.0);
  Math1D::Vector<double> real_count(maxI,0.0);

  //if (align_type == HmmAlignProbNonpar) {
  if (false) {
    //transfer IBM1 => HMM
    //note that it is not possible to estimate conditional alignment probabilities from IBM1 
    // due to its independence assumption. Instead we estimate the probabilities for the empty word 
    for (size_t s=0; s < nSentences; s++) {
      
      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      
      for (uint j=0; j < curJ; j++) {
	
        uint s_idx = cur_source[j];
        double empty_part = dict[0][s_idx-1];
        double real_part = 0.0;
	
        assert(empty_part >= 0.0);
        assert(real_part >= 0.0);
	
        for (uint i=0; i < curI; i++)
          real_part += dict[cur_target[i]][slookup[s](j,i)];
	
        double sum = empty_part + real_part;
        empty_count[curI-1] += empty_part / sum;
        real_count[curI-1]  += real_part / sum;
      }
    }
  
    for (uint I = 1; I <= maxI; I++) {
      if (seenIs.find(I) != seenIs.end()) {
	
        double norm = empty_count[I-1] + real_count[I-1];
        for (uint i=0; i < I; i++) {
          align_model[I-1](I,i) = empty_count[I-1] / norm;
          for (uint ii=0; ii < I; ii++)
            align_model[I-1](ii,i) = real_count[I-1] / (norm*I);
        }
      }
    }
  }
  
  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,MAKENAME(fwcount));
  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(wcooc[i].size());
  }
  
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
    for (uint i=0; i < nTargetWords; i++) {
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
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];
      
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      //std::cerr << "J = " << curJ << ", curI = " << curI << std::endl;
      
      const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
      Math2D::Matrix<double>& cur_facount = facount[curI-1];
      
      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2*curI,curJ,MAKENAME(forward));

      if (align_type == HmmAlignProbReducedpar) {

       	calculate_hmm_forward_with_tricks(cur_source, cur_target, cur_lookup, dict, cur_align_model,
       					  initial_prob[curI-1], forward);
      }
      else {

	calculate_hmm_forward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
			      initial_prob[curI-1], forward);
      }

      const uint start_s_idx = cur_source[0];

      long double sentence_prob = 0.0;
      for (uint i=0; i < 2*curI; i++) {

        assert(forward(i,curJ-1) >= 0.0);
        sentence_prob += forward(i,curJ-1);
      }

      prev_perplexity -= std::log(sentence_prob);
      
      if (! (sentence_prob > 0.0)) {

        std::cerr << "sentence_prob " << sentence_prob << " for sentence pair " << s << " with I=" << curI
                  << ", J= " << curJ << std::endl;

      }
      assert(sentence_prob > 0.0);
      
      Math2D::NamedMatrix<long double> backward(2*curI,curJ,MAKENAME(backward));

      if (align_type == HmmAlignProbReducedpar) {
  
	calculate_hmm_backward_with_tricks(cur_source, cur_target, cur_lookup, dict, cur_align_model,
					   initial_prob[curI-1], backward);
	
      }
      else {

	calculate_hmm_backward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
			       initial_prob[curI-1], backward, true);
      }

      long double bwd_sentence_prob = 0.0;
      for (uint i=0; i < 2*curI; i++)
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
      //start of sentence
      for (uint i=0; i < curI; i++) {
        uint t_idx = cur_target[i];

        double coeff = inv_sentence_prob * backward(i,0);
        fwcount[t_idx][cur_lookup(0,i)] += coeff;

        ficount[curI-1][i] += coeff;
      }
      for (uint i=0; i < curI; i++) {
        double coeff = inv_sentence_prob * backward(i+curI,0);
        fwcount[0][start_s_idx-1] += coeff;

        ficount[curI-1][i+curI] += coeff;
      }

      //mid-sentence
      for (uint j=1; j < curJ; j++) {
        const uint s_idx = cur_source[j];
        const uint j_prev = j -1;

        //real positions
        for (uint i=0; i < curI; i++) {
          const uint t_idx = cur_target[i];


          if (dict[t_idx][cur_lookup(j,i)] > 0.0) {
            fwcount[t_idx][cur_lookup(j,i)] += forward(i,j)*backward(i,j) / (sentence_prob * dict[t_idx][cur_lookup(j,i)]);

            long double bw = backward(i,j) / sentence_prob;	  

            uint i_prev;
            double addon;
	    
            for (i_prev = 0; i_prev < curI; i_prev++) {
              addon = bw * cur_align_model(i,i_prev) * (forward(i_prev,j_prev) + forward(i_prev+curI,j_prev));
              cur_facount(i,i_prev) += addon;
            }
          }
        }

        //empty words
        for (uint i=curI; i < 2*curI; i++) {

          long double bw = backward(i,j) * inv_sentence_prob;
          long double addon = bw * cur_align_model(curI,i-curI) * 
            (forward(i,j_prev) + forward(i-curI,j_prev));
          fwcount[0][s_idx-1] += addon;  
          cur_facount(curI,i-curI) += addon;
        }
      }
    }

    prev_perplexity /= nSentences;
    std::cerr << "perplexity after iteration #" << (iter-1) <<": " << prev_perplexity << std::endl;
    std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    if (align_type != HmmAlignProbNonpar) {

      //compute the expectations of the parameters from the expectations of the models

      for (uint I=1; I <= maxI; I++) {

        if (align_model[I-1].xDim() != 0) {

          for (int i=0; i < (int) I; i++) {

            for (int ii=0; ii < (int) I; ii++) {
              source_fert_count[1] += facount[I-1](ii,i);
              dist_count[zero_offset + ii - i] += facount[I-1](ii,i);

              if (align_type == HmmAlignProbReducedpar && abs(ii-i) > 5) {
                double grouping_norm = std::max(0,i-5);
                grouping_norm += std::max(0,int(I)-1-(i+5));

                assert(grouping_norm > 0.0);

                dist_grouping_count += facount[I-1](ii,i) / grouping_norm;
              }
            }
            source_fert_count[0] += facount[I-1](I,i);
          }
        }
      }

      if (align_type == HmmAlignProbFullpar) {

        if (iter <= 10) {
          dist_params = dist_count;
          dist_params *= 1.0 / dist_params.sum();
        }

        ehmm_m_step(facount, dist_params, zero_offset, 50, dist_grouping_param);
      }
      else if (align_type == HmmAlignProbReducedpar) {

        if (iter <= 2) {

          double norm = 0.0;
          for (int k = -5; k <= 5; k++)
            norm += dist_count[zero_offset + k];
          norm += dist_grouping_count;

          dist_params = dist_count;
          dist_params *= 1.0 / norm;
          dist_grouping_param = dist_grouping_count / norm;
        }

        //call m-step
        ehmm_m_step(facount, dist_params, zero_offset, 25, dist_grouping_param);
      }
    }

    if (init_type == HmmInitPar) {

      for (uint I=1; I <= maxI; I++) {
	
        if (align_model[I-1].xDim() != 0) {
          for (uint i=0; i < I; i++) {
            source_fert_count[1] += initial_prob[I-1][i];
            init_count[i] += initial_prob[I-1][i];
          }
          for (uint i=I; i < 2*I; i++) {
            source_fert_count[0] += initial_prob[I-1][i];
          }
        }
      }

      if (iter <= 2) {
        init_params = init_count;
        init_params *= 1.0 / init_params.sum();
      }

      ehmm_init_m_step(ficount, init_params, 25);
    }

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts

    std::cerr << "null-dict-count: " << fwcount[0].sum() << std::endl;
    double nonnullcount = 0.0;
    for (uint k=1; k < fwcount.size(); k++)
      nonnullcount += fwcount[k].sum();
    std::cerr << "non-null-dict-count: " << nonnullcount << std::endl;


    if (dict_weight_sum > 0.0) {

      //if (iter <= 10) {
      if (true) {
        for (uint i=0; i < nTargetWords; i++) {
	  
          double prev_sum = dict[i].sum();
          double sum = fwcount[i].sum();
          if (sum >= 1e-50) {
	    
            const double inv_sum = 1.0 / fwcount[i].sum();
	    
            if (isnan(inv_sum)) {
              std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
              std::cerr << "sum = " << fwcount[i].sum() << std::endl;
              std::cerr << "number of cooccuring source words: " << fwcount[i].size() << std::endl;
            }
	    
            assert(!isnan(inv_sum));
	    
            for (uint k=0; k < fwcount[i].size(); k++) {
              dict[i][k] = fwcount[i][k] * prev_sum * inv_sum;
            }
          }
        }
      }

      double alpha = 1.0;

      dict_m_step(fwcount, prior_weight, dict, alpha, 45, smoothed_l0, l0_beta);
    }
    else {

      for (uint i=0; i < nTargetWords; i++) {
	
        double sum = fwcount[i].sum();
        if (sum <= 1e-150) {
	  if (dict[i].size() > 0)
	    std::cerr << "WARNING: sum of dictionary counts is almost 0" << std::endl;
        }
        else {
          const double inv_sum = 1.0 / fwcount[i].sum();
	  
          if (isnan(inv_sum)) {
            std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
            std::cerr << "sum = " << fwcount[i].sum() << std::endl;
            std::cerr << "number of cooccuring source words: " << fwcount[i].size() << std::endl;
          }
	  
          assert(!isnan(inv_sum));
	
          for (uint k=0; k < fwcount[i].size(); k++) {
            dict[i][k] = fwcount[i][k] * inv_sum;
          }
        }
      }
    }


    if (align_type != HmmAlignProbNonpar) {

      for (uint i=0; i < 2; i++)
        source_fert[i] = source_fert_count[i] / source_fert_count.sum();
    }

    //compute new alignment probabilities from normalized fractional counts

    if (align_type != HmmAlignProbNonpar) {
      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, align_model);
    }

    if (init_type != HmmInitNonpar) {
      par2nonpar_hmm_init_model(init_params,  source_fert, init_type,  initial_prob);
    }

    for (uint I=1; I <= maxI; I++) {

      if (align_model[I-1].xDim() != 0) {

        if (init_type == HmmInitNonpar) {
          double inv_norm = 1.0 / ficount[I-1].sum();
          for (uint i=0; i < 2*I; i++)
            initial_prob[I-1][i] = inv_norm * ficount[I-1][i]; 
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
                align_model[I-1](i_next,i) = inv_sum *facount[I-1](i_next,i);
              }
            }
          }
        }
      }
    }


    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_marg_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (size_t s=0; s < nSentences; s++) {

        if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

          nContributors++;
          //compute viterbi alignment

          Math1D::Vector<uint> viterbi_alignment;
          const uint curI = target[s].size();
	  
          compute_ehmm_viterbi_alignment(source[s],slookup[s], target[s], 
                                         dict, align_model[curI-1], initial_prob[curI-1],
                                         viterbi_alignment);

          //std::cerr << "alignment: " << viterbi_alignment << std::endl;

          //add alignment error rate
          sum_aer += AER(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          sum_fmeasure += f_measure(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          nErrors += nDefiniteAlignmentErrors(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);


          Storage1D<uint> marg_alignment;
	  
          compute_ehmm_optmarginal_alignment(source[s],slookup[s], target[s], 
                                             dict, align_model[curI-1], initial_prob[curI-1],
                                             marg_alignment);

          sum_marg_aer += AER(marg_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
        }
      }

      sum_aer *= 100.0 / nContributors;
      sum_marg_aer *= 100.0 / nContributors;
      nErrors /= nContributors;
      sum_fmeasure /= nContributors;
      
      std::cerr << "#### EHMM energy after iteration # " << iter << ": " 
                <<  extended_hmm_energy(source, slookup, target, align_model, initial_prob, dict, prior_weight, align_type) 
		<< std::endl;
      std::cerr << "#### EHMM Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "---- EHMM Marginal-AER : " << sum_marg_aer << " %" << std::endl;
      std::cerr << "#### EHMM Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### EHMM Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;

    }
  } //end for (iter)
}


void train_extended_hmm_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source,
                                       const Storage1D<Math2D::Matrix<uint> >& slookup,
                                       const Storage1D<Storage1D<uint> >& target,
                                       const CooccuringWordsType& wcooc,
                                       uint nSourceWords, uint nTargetWords,
                                       FullHMMAlignmentModel& align_model,
                                       Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                       Math1D::Vector<double>& source_fert,
                                       InitialAlignmentProbability& initial_prob,
                                       Math1D::Vector<double>& init_params,
                                       SingleWordDictionary& dict,
                                       uint nIterations, HmmInitProbType init_type, HmmAlignProbType align_type,
                                       std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                                       std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                                       const floatSingleWordDictionary& prior_weight,
                                       bool smoothed_l0, double l0_beta) {

  std::cerr << "starting Extended HMM GD-training" << std::endl;

  dist_grouping_param = -1.0;

  if (init_type >= HmmInitInvalid) {
    
    INTERNAL_ERROR << "invalid type for HMM initial alignment model" << std::endl;
    exit(1);
  }

  if (align_type >= HmmAlignProbInvalid) {

    INTERNAL_ERROR << "invalid type for HMM alignment model" << std::endl;
    exit(1);
  }

  assert(wcooc.size() == nTargetWords);
  //NOTE: the dicitionary is assumed to be initialized

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

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

  align_model.resize_dirty(maxI); //note: access using I-1
  initial_prob.resize(maxI);

  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      //x = new index, y = given index
      align_model[I-1].resize_dirty(I+1,I); //because of empty words
      align_model[I-1].set_constant(1.0 / (I+1));

      initial_prob[I-1].resize_dirty(2*I);
      initial_prob[I-1].set_constant(0.5 / I);
    }
  }

  Math1D::Vector<double> empty_count(maxI,0.0);
  Math1D::Vector<double> real_count(maxI,0.0);

  double dist_grouping_grad = 0.0;
  double new_dist_grouping_param = 0.0;
  double hyp_dist_grouping_param = 0.0;

  Math1D::NamedVector<double> dist_grad(0,MAKENAME(dist_grad));
  Math1D::NamedVector<double> new_dist_params(0,MAKENAME(new_dist_params));
  Math1D::NamedVector<double> hyp_dist_params(0,MAKENAME(hyp_dist_params));

  Math1D::NamedVector<double> init_param_grad(0,MAKENAME(init_param_grad));
  Math1D::NamedVector<double> new_init_params(0,MAKENAME(new_init_params));
  Math1D::NamedVector<double> hyp_init_params(0,MAKENAME(hyp_init_params));

  Math1D::Vector<double> source_fert_grad(0);
  Math1D::Vector<double> new_source_fert(0);  
  Math1D::Vector<double> hyp_source_fert(0);

  uint zero_offset = maxI-1;

  if (init_type == HmmInitPar) {

    init_params.resize(maxI,1.0 / maxI);
    init_param_grad = init_params;
    new_init_params = init_params;
    hyp_init_params = init_params;
  }
  if (init_type == HmmInitPar || align_type != HmmAlignProbNonpar) {

    source_fert.resize(2);
    source_fert[0] = 0.02;
    source_fert[1] = 0.98;
    source_fert_grad.resize(2,0.0);
    new_source_fert.resize(2,0.0);
    hyp_source_fert.resize(2,0.0);
  }

  
  if (align_type != HmmAlignProbNonpar) {
    dist_params.resize(2*maxI-1, 1.0 / (2*maxI-1));
    dist_grad.resize(2*maxI-1,0.0);
    new_dist_params.resize(2*maxI-1, 0.0);
    hyp_dist_params.resize(2*maxI-1, 0.0);

    if (align_type == HmmAlignProbReducedpar) {
      dist_grouping_param = 0.2;
      dist_params.set_constant(0.8 / 11.0);
    }

    par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                   align_type, align_model);
  }

  if (init_type == HmmInitPar) {

    par2nonpar_hmm_init_model(init_params,  source_fert, init_type,  initial_prob);
  }

#if 0
  if (align_type == HmmAlignProbNonpar) {
    //transfer IBM1 => HMM
    //note that it is not possible to estimate conditional alignment probabilities from IBM1 
    // due to its independence assumption. Instead we estimate the probabilities for the empty word 
    for (size_t s=0; s < nSentences; s++) {
      
      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      
      for (uint j=0; j < curJ; j++) {
	
        uint s_idx = cur_source[j];
        double empty_part = dict[0][s_idx-1];
        double real_part = 0.0;
	
        assert(empty_part >= 0.0);
        assert(real_part >= 0.0);
	
        for (uint i=0; i < curI; i++)
          real_part += dict[cur_target[i]][slookup[s](j,i)];
	
        double sum = empty_part + real_part;
        empty_count[curI-1] += empty_part / sum;
        real_count[curI-1]  += real_part / sum;
      }
    }
    
    for (uint I = 1; I <= maxI; I++) {
      if (seenIs.find(I) != seenIs.end()) {
	
        double norm = empty_count[I-1] + real_count[I-1];
        for (uint i=0; i < I; i++) {
          align_model[I-1](I,i) = empty_count[I-1] / norm;
          for (uint ii=0; ii < I; ii++)
            align_model[I-1](ii,i) = real_count[I-1] / (norm*I);
        }
      }
    }
  }
#endif

  InitialAlignmentProbability init_grad(maxI,MAKENAME(init_grad));
  
  Storage1D<Math1D::Vector<double> > dict_grad(nTargetWords);
  for (uint i=0; i < nTargetWords; i++) {
    dict_grad[i].resize(wcooc[i].size());
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
  
  SingleWordDictionary new_dict_prob(nTargetWords,MAKENAME(new_dict_prob));
  SingleWordDictionary hyp_dict_prob(nTargetWords,MAKENAME(hyp_dict_prob));

  for (uint i=0; i < nTargetWords; i++) {
    new_dict_prob[i].resize(wcooc[i].size());
    hyp_dict_prob[i].resize(wcooc[i].size());
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

  Math1D::Vector<double> slack_vector(nTargetWords,0.0);
  Math1D::Vector<double> new_slack_vector(nTargetWords,0.0);

  for (uint i=0; i < nTargetWords; i++) {
    double slack_val = 1.0 - dict[i].sum();
    slack_vector[i] = slack_val;
    new_slack_vector[i] = slack_val;
  }

  double energy = extended_hmm_energy(source, slookup, target, align_model, initial_prob, dict, prior_weight, align_type);


  double line_reduction_factor = 0.5;

  uint nSuccessiveReductions = 0;

  std::cerr << "start energy: " << energy << std::endl;

  double alpha = 0.001; 

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting EHMM gd-iter #" << iter << std::endl;
    std::cerr << "alpha: " << alpha << std::endl;

    //set counts to 0
    for (uint i=0; i < nTargetWords; i++) {
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

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];
      
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      
      const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
      const Math2D::Matrix<double>& cur_align_grad = align_grad[curI-1];
      
      /**** Baum-Welch traininig: start with calculating forward and backward ********/

      Math2D::NamedMatrix<long double> forward(2*curI,curJ,MAKENAME(forward));

      if (align_type == HmmAlignProbReducedpar) {

       	calculate_hmm_forward_with_tricks(cur_source, cur_target, cur_lookup, dict, cur_align_model,
       					  initial_prob[curI-1], forward);
      }
      else {

	calculate_hmm_forward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
			      initial_prob[curI-1], forward);
      }
      
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

      if (align_type == HmmAlignProbReducedpar) {
  
	calculate_hmm_backward_with_tricks(cur_source, cur_target, cur_lookup, dict, cur_align_model,
					   initial_prob[curI-1], backward);
	
      }
      else {

	calculate_hmm_backward(cur_source, cur_target, cur_lookup, dict, cur_align_model,
			       initial_prob[curI-1], backward, true);
      }

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      /**** update gradients ****/

      //start of sentence
      for (uint i=0; i < curI; i++) {
        const uint t_idx = cur_target[i];

        const double coeff = inv_sentence_prob * backward(i,0);

        const double cur_dict_entry = dict[t_idx][cur_lookup(0,i)];

        if (cur_dict_entry > 1e-300) {

          double addon = coeff / cur_dict_entry;
          if (smoothed_l0)
            addon *= prob_pen_prime(cur_dict_entry, l0_beta);
          dict_grad[t_idx][cur_lookup(0,i)] -= addon;
        }

        if (initial_prob[curI-1][i] > 1e-300)
          init_grad[curI-1][i] -= coeff / initial_prob[curI-1][i];
      }
      for (uint i=0; i < curI; i++) {

        const double coeff = inv_sentence_prob * backward(i+curI,0);

        const double cur_dict_entry = dict[0][start_s_idx-1];

        if (cur_dict_entry > 1e-300) {

          double addon = coeff / cur_dict_entry;
          if (smoothed_l0)
            addon *= prob_pen_prime(cur_dict_entry, l0_beta);

          dict_grad[0][start_s_idx-1] -= addon;
        }

        if (initial_prob[curI-1][i+curI] > 1e-300)
          init_grad[curI-1][i+curI] -= coeff / initial_prob[curI-1][i+curI];
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
              / (sentence_prob * cur_dict_entry * cur_dict_entry);

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

            double addon = bw*forward(i,j) / (cur_dict_entry * cur_dict_entry);

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
	      
              //std::cerr << "I: " << I << ", i: " << i << ", non_zero_sum: " << non_zero_sum << std::endl;
	      
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

    if (align_type != HmmAlignProbNonpar || init_type == HmmInitPar) {
      
      for (uint i=0; i < 2; i++)
        new_source_fert[i] = source_fert[i] - alpha * source_fert_grad[i];

      projection_on_simplex(new_source_fert.direct_access(), 2);
    }

    if (init_type == HmmInitPar) {

      for (uint k=0; k < init_params.size(); k++)
        new_init_params[k] = init_params[k] - alpha * init_param_grad[k];

      projection_on_simplex(new_init_params.direct_access(), new_init_params.size());
    }

    if (align_type == HmmAlignProbFullpar) {

      for (uint k=0; k < dist_params.size(); k++)
        new_dist_params[k] = dist_params[k] - alpha * dist_grad[k];

      projection_on_simplex(new_dist_params.direct_access(), new_dist_params.size());
    }
    else if (align_type == HmmAlignProbReducedpar) {

      for (uint k=0; k < dist_params.size(); k++)
        new_dist_params[k] = dist_params[k] - alpha * dist_grad[k];

      new_dist_grouping_param = dist_grouping_param - alpha * dist_grouping_grad;

      assert(new_dist_params.size() >= 11);

      projection_on_simplex_with_slack(new_dist_params.direct_access()+zero_offset-5,new_dist_grouping_param,11);
    }

    for (uint i=0; i < nTargetWords; i++) {
      projection_on_simplex_with_slack(new_dict_prob[i].direct_access(),new_slack_vector[i],new_dict_prob[i].size());
    }

    for (uint i=0; i < nTargetWords; i++) {

      for (uint k=0; k < dict[i].size(); k++) 
        new_dict_prob[i][k] = dict[i][k] - alpha * dict_grad[i][k];
    }

    for (uint I = 1; I <= maxI; I++) {

      if (seenIs.find(I) != seenIs.end()) {

        if (init_type == HmmInitPar) {

          double sum = 0;
          for (uint k=0; k < I; k++)
            sum += new_init_params[k];

          for (uint k=0; k < I; k++)
            new_init_prob[I-1][k] = new_source_fert[1] * new_init_params[k] / sum;
          for (uint k=I; k < 2*I; k++)
            new_init_prob[I-1][k] = new_source_fert[0] / I;
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

    for (uint i=0; i < nTargetWords; i++) {
      projection_on_simplex_with_slack(new_dict_prob[i].direct_access(),new_slack_vector[i],new_dict_prob[i].size());
    }

    for (uint I = 1; I <= maxI; I++) {
      projection_on_simplex(new_init_prob[I-1].direct_access(),new_init_prob[I-1].size());

      if (align_type == HmmAlignProbNonpar) {
        for (uint y=0; y < align_model[I-1].yDim(); y++) {
	  
          projection_on_simplex(new_align_prob[I-1].direct_access() + y*align_model[I-1].xDim(),
                                align_model[I-1].xDim());
        }
      }
    }

    double hyp_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = 1.0;

    bool decreasing = true;

    uint nInnerIter = 0;

    while (hyp_energy > energy || decreasing) {

      nInnerIter++;

      lambda *= line_reduction_factor;

      const double neg_lambda = 1.0 - lambda;
      
      for (uint i=0; i < nTargetWords; i++) {
	
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
                                              hyp_init_prob, hyp_dict_prob, prior_weight);   

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

    for (uint i=0; i < nTargetWords; i++) {
      
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

    std::cerr << "slack-sum: " << slack_vector.sum() << std::endl;
    std::cerr << "energy: " << energy << std::endl;

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double sum_marg_aer = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (size_t s=0; s < nSentences; s++) {

        if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

          nContributors++;
          //compute viterbi alignment

          Storage1D<uint> viterbi_alignment;
          const uint curI = target[s].size();
	  
          compute_ehmm_viterbi_alignment(source[s],slookup[s], target[s], 
                                         dict, align_model[curI-1], initial_prob[curI-1],
                                         viterbi_alignment);

          //add alignment error rate
          sum_aer += AER(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          sum_fmeasure += f_measure(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          nErrors += nDefiniteAlignmentErrors(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);

          Storage1D<uint> marg_alignment;
	  
          compute_ehmm_optmarginal_alignment(source[s],slookup[s], target[s], 
                                             dict, align_model[curI-1], initial_prob[curI-1],
                                             marg_alignment);

          sum_marg_aer += AER(marg_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
        }
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
                                const Storage1D<Math2D::Matrix<uint> >& slookup,
                                const Storage1D<Storage1D<uint> >& target,
                                const CooccuringWordsType& wcooc,
                                uint nSourceWords, uint nTargetWords,
                                FullHMMAlignmentModel& align_model,
                                Math1D::Vector<double>& dist_params, double& dist_grouping_param,
                                Math1D::Vector<double>& source_fert,
                                InitialAlignmentProbability& initial_prob,
                                SingleWordDictionary& dict, uint nIterations, 
                                HmmInitProbType init_type, HmmAlignProbType align_type, bool deficient_parametric,
                                std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                                std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                                const floatSingleWordDictionary& prior_weight) {

  std::cerr << "starting Viterbi Training for Extended HMM" << std::endl;

  if (init_type >= HmmInitInvalid) {
    
    INTERNAL_ERROR << "invalid type for HMM initial alignment model" << std::endl;
    exit(1);
  }

  if (init_type == HmmInitPar) {
    TODO("HmmInitPar");
  }

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  Storage1D<Math1D::Vector<uint> > viterbi_alignment(source.size());

  for (size_t s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    viterbi_alignment[s].resize(cur_source.size());
  }
  assert(wcooc.size() == nTargetWords);
  //NOTE: the dictionary is assumed to be initialized

  Math1D::Vector<double> source_fert_count(2,0.0);
  source_fert.resize(2);
  source_fert[0] = 0.02;
  source_fert[1] = 0.98;

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

  align_model.resize_dirty(maxI); //note: access using I-1
  initial_prob.resize(maxI);
  InitialAlignmentProbability icount(maxI,MAKENAME(icount));

  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      //x = new index, y = given index

      align_model[I-1].resize_dirty(I+1,I); //because of empty words
 
      if (align_type == HmmAlignProbNonpar) {
        align_model[I-1].set_constant(1.0 / (I+1));
      }
      else {
        align_model[I-1].set_constant(source_fert[1] / (I));
        for (uint i=0; i < I; i++)
          align_model[I-1](I,i) = source_fert[0];
      }

      initial_prob[I-1].resize_dirty(2*I);
      initial_prob[I-1].set_constant(0.5 / I);
      icount[I-1].resize_dirty(2*I);
    }
  }

#if 0
  Math1D::Vector<double> empty_count(maxI,0.0);
  Math1D::Vector<double> real_count(maxI,0.0);

  //transfer IBM1 => HMM
  //note that it is not possible to estimate conditional alignment probabilities from IBM1 
  // due to its independence assumption. Instead we estimate the probabilities for the empty word 
  for (size_t s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    
    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();
    
    for (uint j=0; j < curJ; j++) {

      uint s_idx = cur_source[j];
      double empty_part = dict[0][s_idx-1];
      double real_part = 0.0;

      assert(empty_part >= 0.0);
      assert(real_part >= 0.0);

      for (uint i=0; i < curI; i++)
        real_part += dict[cur_target[i]][slookup[s](j,i)];
      
      double sum = empty_part + real_part;
      empty_count[curI-1] += empty_part / sum;
      real_count[curI-1]  += real_part / sum;
    }
  }
  
  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {

      double norm = empty_count[I-1] + real_count[I-1];
      for (uint i=0; i < I; i++) {
        align_model[I-1](I,i) = empty_count[I-1] / norm;
        for (uint ii=0; ii < I; ii++)
          align_model[I-1](ii,i) = real_count[I-1] / (norm*I);
      }
    }
  }
#endif  

  Storage1D<Math1D::Vector<uint> > dcount(nTargetWords);
  for (uint i=0; i < nTargetWords; i++) {
    dcount[i].resize(wcooc[i].size());
  }
  
  FullHMMAlignmentModel acount(maxI,MAKENAME(acount));
  for (uint I = 1; I <= maxI; I++) {
    if (seenIs.find(I) != seenIs.end()) {
      acount[I-1].resize_dirty(I+1,I);
    }
  }

  Math1D::NamedVector<double> dist_count(0,MAKENAME(dist_count));

  dist_grouping_param = (align_type == HmmAlignProbReducedpar) ? 0.2 : -1.0; 

  uint zero_offset = maxI-1;
  
  if (align_type != HmmAlignProbNonpar) {
    dist_params.resize(2*maxI-1, 1.0 / (2*maxI-1));
    dist_count.resize(2*maxI-1,0.0);

    if (align_type == HmmAlignProbReducedpar) {
      
      dist_params.set_constant(0.0);
      for (int k=-5; k <= 5; k++)
        dist_params[zero_offset+k] = 0.8 / 11.0;
    }

    for (uint I = 1; I <= maxI; I++) {
      if (seenIs.find(I) != seenIs.end()) {
        align_model[I-1].resize_dirty(I+1,I); //because of empty words
        align_model[I-1].set_constant(source_fert[1] / I);
	  
        for (uint y=0; y < align_model[I-1].yDim(); y++)
          align_model[I-1](I,y) = source_fert[0];
      }
    }
  }

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting Viterbi-EHMM iteration #" << iter << std::endl;

    double prev_perplexity = 0.0;

    //set counts to 0
    for (uint i=0; i < nTargetWords; i++) {
      dcount[i].set_constant(0);
    }

    for (uint I = 1; I <= maxI; I++) {
      acount[I-1].set_constant(0.0);
      icount[I-1].set_constant(0.0);
    }

    if (align_type != HmmAlignProbNonpar) {
      source_fert_count.set_constant(0.0);
      dist_count.set_constant(0.0);      
    }

    for (size_t s=0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];
      
      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      
      const Math2D::Matrix<double>& cur_align_model = align_model[curI-1];
      Math2D::Matrix<double>& cur_facount = acount[curI-1];


      double prob = compute_ehmm_viterbi_alignment(cur_source,cur_lookup, cur_target, 
                                                   dict, cur_align_model, initial_prob[curI-1],
                                                   viterbi_alignment[s], true);

      prev_perplexity -= std::log(prob);
      
      if (! (prob > 0.0)) {

        std::cerr << "sentence_prob " << prob << " for sentence pair " << s << " with I=" << curI
                  << ", J= " << curJ << std::endl;
      }
      assert(prob > 0.0);

      /**** update counts ****/      
      for (uint j=0; j < curJ; j++) {

        uint aj = viterbi_alignment[s][j];
        if (aj >= curI) {
          dcount[0][cur_source[j]-1] += 1;
          if (align_type != HmmAlignProbNonpar) 
            source_fert_count[0] += 1.0;
        }
        else {
          dcount[cur_target[aj]][cur_lookup(j,aj)] += 1;
          if (align_type != HmmAlignProbNonpar) 
            source_fert_count[1] += 1.0;
        }

        if (j == 0)
          icount[curI-1][aj] += 1.0;
        else {

          uint prev_aj = viterbi_alignment[s][j-1];
	  
          if (prev_aj >= curI) {

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
    }

    prev_perplexity /= nSentences;

    //include the dict_regularity term in the output energy
    double energy = prev_perplexity;
    for (uint i=0; i < dcount.size(); i++)
      for (uint k=0; k < dcount[i].size(); k++)
        if (dcount[i][k] > 0)
          energy += prior_weight[i][k] / nSentences;

    //std::cerr << "perplexity after iteration #" << (iter-1) <<": " << prev_perplexity << std::endl;
    std::cerr << "energy after iteration #" << (iter-1) <<": " << energy << std::endl;
    
    //std::cerr << "computing alignment and dictionary probabilities from normalized counts" << std::endl;

    if (!deficient_parametric && align_type != HmmAlignProbNonpar) {

      for (uint k=0; k < 2; k++)
        source_fert[k] = source_fert_count[k] / source_fert_count.sum();

      if (iter == 1) {

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

        //init from counts
        if (align_type == HmmAlignProbFullpar) {

          dist_params = dist_count;
          dist_params *= 1.0 / dist_params.sum();
        }
        else if (align_type == HmmAlignProbReducedpar) {


          double norm = 0.0;
          for (int k = -5; k <= 5; k++)
            norm += dist_count[zero_offset + k];
          norm += dist_grouping_count;

          dist_params = dist_count;
          dist_params *= 1.0 / norm;
          dist_grouping_param = dist_grouping_count / norm;
        }
      }

      ehmm_m_step(acount, dist_params, zero_offset, 80, dist_grouping_param);
	
      par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                     align_type, align_model);
    }

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
      
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];
      Math2D::Matrix<double>& cur_acount = acount[curI-1];

      for (uint j=0; j < curJ; j++) {

        //std::cerr << "j: " << j << std::endl;
	
        uint cur_aj = viterbi_alignment[s][j];
        uint new_aj = cur_aj;

        double best_change = 1e300;

        uint cur_target_word = (cur_aj >= curI) ? 0 : target[s][cur_aj];
        uint cur_dict_num = (cur_aj >= curI) ? 0 : target[s][cur_aj];
        uint cur_idx = (cur_aj >= curI) ? source[s][j]-1 : cur_lookup(j,cur_aj);
	
        Math1D::Vector<uint>& cur_dictcount = dcount[cur_dict_num]; 
        uint cur_dictsum = dict_sum[cur_dict_num]; 

        for (uint i=0; i < 2*curI; i++) {
	  
          bool allowed = true;
          bool careful = false;
          if (i == cur_aj)
            allowed = false;

          if (j > 0 && i >= curI) {

            uint prev_aj = viterbi_alignment[s][j-1];
	    
            if (i != prev_aj && i != prev_aj+curI)
              allowed = false;
          }

          if (j > 0 && j+1 < curJ) {
            int effective_prev_aj = viterbi_alignment[s][j-1];
            if (effective_prev_aj >= (int) curI)
              effective_prev_aj -= curI;

            int effective_cur_aj = cur_aj;
            if (effective_cur_aj >= (int) curI)
              effective_cur_aj -= curI;
	    
            int effective_i = 0;
            if (effective_i >= (int) curI)
              effective_i -= curI;

            int next_aj = viterbi_alignment[s][j+1];
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
          if (j+1 < curJ && viterbi_alignment[s][j+1] >= curI)
            allowed = false; //in this case many more distributions /successive positions could be affected

          if (allowed) {	    

            uint new_target_word = (i >= curI) ? 0 : target[s][i];

            uint effective_i = i;
            if (effective_i >= curI)
              effective_i -= curI;

            uint effective_cur_aj = cur_aj;
            if (effective_cur_aj >= curI)
              effective_cur_aj -= curI;

            uint hyp_dict_num = (i >= curI) ? 0 : target[s][i];
	    
            uint hyp_idx = (i >= curI) ? source[s][j]-1 : cur_lookup(j,i);
	    
            double change = 0.0;

            //a) regarding preceeding pos
            if (j == 0) {

              if (init_type != HmmInitFix) {
                assert(icount[curI-1][cur_aj] > 0);
		
                change -= -icount[curI-1][cur_aj] * std::log(icount[curI-1][cur_aj]);
                if (icount[curI-1][i] > 0)
                  change -= -icount[curI-1][i] * std::log(icount[curI-1][i]);
		
                if (icount[curI-1][cur_aj] > 1)
                  change += -(icount[curI-1][cur_aj]-1) * std::log(icount[curI-1][cur_aj]-1);
                change += -(icount[curI-1][i]+1) * std::log(icount[curI-1][i]+1);
              }
            }
            else {
              // j > 0

              //note: the total sum of counts for prev_aj stays constant in this operation
              uint prev_aj = viterbi_alignment[s][j-1];
              if (prev_aj >= curI)
                prev_aj -= curI;

              if (align_type == HmmAlignProbNonpar) {
                int cur_c = cur_acount(std::min(curI,cur_aj),prev_aj);
                int new_c = cur_acount(std::min(curI,i),prev_aj);
                assert(cur_c > 0);
	      
                change -= -cur_c * std::log(cur_c);
                if (new_c > 0)
                  change -= -new_c*std::log(new_c);
	      
                if (cur_c > 1)
                  change += -(cur_c-1) * std::log(cur_c-1);
                change += -(new_c+1) * std::log(new_c+1);
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
                  change -= -std::log(align_model[curI-1](std::min(curI,cur_aj),prev_aj));
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
                  change += -std::log(align_model[curI-1](std::min(curI,i),prev_aj));
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

              uint next_aj = std::min(curI,viterbi_alignment[s][j+1]);

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

                  change -= -std::log(align_model[curI-1](std::min(next_aj,curI),effective_cur_aj));
                  change += -std::log(align_model[curI-1](std::min(next_aj,curI),effective_new_aj));
                }
              }
            }

            assert(!isnan(change));

            uint prev_word = cur_target_word; 
            uint new_word =  (new_aj >= curI) ? 0 : target[s][new_aj];

            if (prev_word != new_word) {

              if (dict_sum[new_target_word] > 0)
                change -= double(dict_sum[new_target_word]) * std::log( dict_sum[new_target_word] );
              change += double(dict_sum[new_target_word]+1.0) * std::log( dict_sum[new_target_word]+1.0 );

              if (dcount[new_target_word][hyp_idx] > 0)
                change -= double(dcount[new_target_word][hyp_idx]) * 
                  (-std::log(dcount[new_target_word][hyp_idx]));
              else
                change += prior_weight[hyp_dict_num][hyp_idx]; 

              change += double(dcount[new_target_word][hyp_idx]+1) * 
                (-std::log(dcount[new_target_word][hyp_idx]+1.0));

              change -= double(cur_dictsum) * std::log(cur_dictsum);
              if (cur_dictsum > 1)
                change += double(cur_dictsum-1) * std::log(cur_dictsum-1.0);
	      
              change -= - double(cur_dictcount[cur_idx]) * std::log(cur_dictcount[cur_idx]);
	      
              if (cur_dictcount[cur_idx] > 1) {
                change += double(cur_dictcount[cur_idx]-1) * (-std::log(cur_dictcount[cur_idx]-1));
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
        }
	      
        if (best_change < -0.01 && new_aj != cur_aj) {

          nSwitches++;
	  
          uint hyp_idx = (new_aj >= curI) ? source[s][j]-1 : cur_lookup(j,new_aj);

          viterbi_alignment[s][j] = new_aj;
	      
          uint prev_word = cur_target_word; 
          uint new_word = (new_aj >= curI) ? 0 : target[s][new_aj];
	      
          //recompute the stored values for the two affected words
          if (prev_word != new_word) {

            uint hyp_dict_num = (new_aj >= curI) ? 0 : target[s][new_aj];
            Math1D::Vector<uint>& hyp_dictcount = dcount[hyp_dict_num];
	    
            cur_dictcount[cur_idx] -= 1;
            hyp_dictcount[hyp_idx] += 1;
            dict_sum[cur_dict_num] -= 1;
            dict_sum[hyp_dict_num] += 1;
          }

          /****** change alignment counts *****/
	      
          //a) dependency to preceeding pos
          if (j == 0) {
	    
            if (init_type != HmmInitFix) {
              assert(icount[curI-1][cur_aj] > 0);
	      
              icount[curI-1][cur_aj]--;
              icount[curI-1][new_aj]++;
            }
          }
          else {

            uint prev_aj = viterbi_alignment[s][j-1];
            if (prev_aj >= curI)
              prev_aj -= curI;
	    
            assert( cur_acount(std::min(curI,cur_aj),prev_aj) > 0);
	      
            cur_acount(std::min(curI,cur_aj),prev_aj)--;
            cur_acount(std::min(curI,new_aj),prev_aj)++;
	    
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
          if (j+1 < curJ) {

            uint next_aj = viterbi_alignment[s][j+1];
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

#endif

    /***** compute alignment and dictionary probabilities from normalized counts ******/

    //compute new dict from normalized fractional counts
    for (uint i=0; i < nTargetWords; i++) {

      double sum = dcount[i].sum();
      if (sum <= 1e-250) {
        //std::cerr << "WARNING: sum of dictionary counts is almost 0" << std::endl;
        dict[i].set_constant(0.0);
      }
      else {
        const double inv_sum = 1.0 / sum;
      
        if (isnan(inv_sum)) {
          std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
          std::cerr << "sum = " << dcount[i].sum() << std::endl;
          std::cerr << "number of cooccuring source words: " << dcount[i].size() << std::endl;
        }
	
        assert(!isnan(inv_sum));
	
        for (uint k=0; k < dcount[i].size(); k++) {
          dict[i][k] = dcount[i][k] * inv_sum;
        }
      }
    }

    if (align_type != HmmAlignProbNonpar) {

      for (uint k=0; k < 2; k++)
        source_fert[k] = source_fert_count[k] / source_fert_count.sum();

      if (deficient_parametric) {
	
        double fdist_count_sum = dist_count.sum();
        assert(fdist_count_sum == dist_count_sum);
        for (uint k=0; k < dist_count.size(); k++)
          dist_params[k] = dist_count[k] / fdist_count_sum;
      }
      else {

        ehmm_m_step(acount, dist_params, zero_offset, 80, dist_grouping_param);
	
        par2nonpar_hmm_alignment_model(dist_params, zero_offset, dist_grouping_param, source_fert,
                                       align_type, align_model);
      }
    }

    std::cerr << "new source-fert: " << source_fert << std::endl;

    //compute new alignment probabilities from normalized fractional counts
    for (uint I=1; I <= maxI; I++) {

      if (acount[I-1].xDim() != 0) {

        if (init_type != HmmInitFix) {
          double inv_norm = 1.0 / icount[I-1].sum();
          assert(!isnan(inv_norm));
          for (uint i=0; i < 2*I; i++)
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

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_marg_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (size_t s=0; s < nSentences; s++) {

        if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

          nContributors++;
          //compute viterbi alignment

          Storage1D<uint> viterbi_alignment;
          const uint curI = target[s].size();
	  
          compute_ehmm_viterbi_alignment(source[s],slookup[s], target[s], 
                                         dict, align_model[curI-1], initial_prob[curI-1],
                                         viterbi_alignment);

          //add alignment error rate
          sum_aer += AER(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          sum_fmeasure += f_measure(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          nErrors += nDefiniteAlignmentErrors(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);


          Storage1D<uint> marg_alignment;
	  
          compute_ehmm_optmarginal_alignment(source[s],slookup[s], target[s], 
                                             dict, align_model[curI-1], initial_prob[curI-1],
                                             marg_alignment);

          sum_marg_aer += AER(marg_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
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

