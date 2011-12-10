/*** written by Thomas Schoenemann as a private person without employment, October 2009 
 *** and later by Thomas Schoenemann as employee of Lund University, 2010 ***/


#include "ibm1_training.hh"


#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "matrix.hh"

#include "training_common.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"

#include "projection.hh"


double prob_penalty(double x) {

  //return 1-pow(1-x,2);
  return x;
}

double prob_pen_prime(double x) {
  
  //return 2*(1-x);
  return 1.0;
}

double ibm1_perplexity( const Storage1D<Storage1D<uint> >& source,
                        const Storage1D<Math2D::Matrix<uint> >& slookup,
                        const Storage1D< Storage1D<uint> >& target,
                        const SingleWordDictionary& dict) {

  //std::cerr << "calculating IBM 1 perplexity" << std::endl;

  double sum = 0.0;

  const size_t nSentences = target.size();
  assert(slookup.size() == nSentences);

  uint nActualSentences = nSentences;

  for (size_t s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const Math2D::Matrix<uint>& cur_lookup = slookup[s]; 
    
    const uint nCurSourceWords = cur_source.size();
    const uint nCurTargetWords = cur_target.size();

    sum += nCurSourceWords*std::log(nCurTargetWords);
    
    for (uint j=0; j < nCurSourceWords; j++) {

      double cur_sum = dict[0][cur_source[j]-1]; // handles empty word
      
      for (uint i=0; i < nCurTargetWords; i++) {
        cur_sum += dict[cur_target[i]][cur_lookup(j,i)];
      }
      
      sum -= std::log(cur_sum);
    }
  }

  return sum / nActualSentences;
}

double ibm1_energy( const Storage1D<Storage1D<uint> >& source,
                    const Storage1D<Math2D::Matrix<uint> >& slookup,
                    const Storage1D< Storage1D<uint> >& target,
                    const SingleWordDictionary& dict,
                    const floatSingleWordDictionary& prior_weight) {

  double energy = 0.0; 
    
  for (uint i=0; i < dict.size(); i++) {

    const uint size = dict[i].size();
    
    for (uint k=0; k < size; k++) {
      energy += prior_weight[i][k] * prob_penalty(dict[i][k]);
    }
  }

  energy /= target.size(); //since the perplexity is also divided by that amount
  
  energy += ibm1_perplexity(source, slookup, target, dict);

  return energy;
}

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, 
                                 const Math1D::Vector<float>& prior_weight,
                                 const Math1D::Vector<double>& dict) {


  double energy = 0.0;

  for (uint k=0; k < dict.size(); k++) {
    energy += prior_weight[k] * dict[k];

    if (dict[k] > 1e-300)
      energy -= fdict_count[k] * std::log(dict[k]);
    else
      energy += fdict_count[k] * 15000.0;
  }

  return energy;
}

void single_dict_m_step(const Math1D::Vector<double>& fdict_count, 
                        const Math1D::Vector<float>& prior_weight,
                        Math1D::Vector<double>& dict, double alpha, uint nIter) {

  if (prior_weight.max_abs() == 0.0) {
    
    const double sum = fdict_count.sum();

    if (sum > 1e-305) {
      for (uint k=0; k < prior_weight.size(); k++) {
        dict[k] = fdict_count[k] / sum;
      }
    }

    return;
  }


  double energy = single_dict_m_step_energy(fdict_count,prior_weight,dict);

  Math1D::Vector<double> dict_grad = dict;
  Math1D::Vector<double> hyp_dict = dict;
  Math1D::Vector<double> new_dict = dict;

  double slack_entry = 1.0 - dict.sum();
  double new_slack_entry = slack_entry;

  double line_reduction_factor = 0.5;

  for (uint iter=1; iter <= nIter; iter++) {

    //set gradient to 0 and recalculate
    for (uint k=0; k < prior_weight.size(); k++) {
      double cur_dict_entry = std::max(1e-15, dict[k]);

      dict_grad[k] = prior_weight[k] - fdict_count[k] / cur_dict_entry;
    }

    //go in neg. gradient direction
    for (uint k=0; k < prior_weight.size(); k++) {
	
      new_dict[k] = dict[k] - alpha * dict_grad[k];
    }
    
    new_slack_entry = slack_entry;
    
    //reproject
    projection_on_simplex_with_slack(new_dict.direct_access(), new_slack_entry, new_dict.size());

    
    double hyp_energy = 1e300; //dict_m_step_energy(fdict_count,prior_weight,new_dict);
    
    double lambda  = 1.0;
    double best_lambda = lambda;

    bool decreasing = true;

    uint nTries = 0;

    while (decreasing || hyp_energy > energy) {

      nTries++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;
      
      for (uint k=0; k < prior_weight.size(); k++) {      
        hyp_dict[k] = lambda * new_dict[k] + neg_lambda * dict[k];
      }
      
      double new_energy = single_dict_m_step_energy(fdict_count,prior_weight,hyp_dict);
      //std::cerr << "lambda = " << lambda << ", hyp_energy = " << new_energy << std::endl;

      if (new_energy < hyp_energy) {
        hyp_energy = new_energy;
        decreasing = true;

        best_lambda = lambda;
      }
      else
        decreasing = false;      

      if (hyp_energy <= 0.95*energy)
        break;
      if (nTries >= 4 && hyp_energy <= 0.99*energy)
        break;

      if (nTries >= 18)
        break;

      //       if (iter > 2 && nTries >= 15)
      // 	break;

      //       if (nTries > 35)
      // 	break;
    }

    if (hyp_energy > energy) {
      break;
    }

    if (nTries > 12)
      line_reduction_factor *= 0.5;
    else if (nTries > 4)
      line_reduction_factor *= 0.8;
    
 
    double best_energy = hyp_energy;
    double neg_best_lambda = 1.0 - best_lambda;
    
    energy = best_energy;

    for (uint k=0; k < prior_weight.size(); k++) {      
      dict[k] = best_lambda * new_dict[k] + neg_best_lambda * dict[k];
    }

    slack_entry = best_lambda * new_slack_entry + neg_best_lambda * slack_entry;

    if (best_lambda < 1e-8)
      break;
  }  
}


//NOTE: the function to be minimized can be decomposed over the target words
void dict_m_step(const SingleWordDictionary& fdict_count, 
                 const floatSingleWordDictionary& prior_weight,
                 SingleWordDictionary& dict, double alpha, uint nIter) {

  for (uint k=0; k < dict.size(); k++)
    single_dict_m_step(fdict_count[k],prior_weight[k],dict[k],alpha,nIter);    
}



void train_ibm1(const Storage1D<Storage1D<uint> >& source, 
                const Storage1D<Math2D::Matrix<uint> >& slookup,
                const Storage1D<Storage1D<uint> >& target, 
                const CooccuringWordsType& cooc,
                uint nSourceWords, uint nTargetWords,
                SingleWordDictionary& dict,
                uint nIter,
                std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                const floatSingleWordDictionary& prior_weight) {
  
  assert(cooc.size() == nTargetWords);
  dict.resize_dirty(nTargetWords);

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  double dict_weight_sum = 0.0;
  for (uint i=0; i < nTargetWords; i++) {
    dict_weight_sum += fabs(prior_weight[i].sum());
  }

  //prepare dictionary
  for (uint i=0; i < nTargetWords; i++) {
    
    const uint size = cooc[i].size();
    if (size == 0) {
      std::cerr << "WARNING: dict-size for t-word " << i << " is zero" << std::endl;
    }

    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double) size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

#if 0
  for (size_t s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint nCurSourceWords = cur_source.size();
    const uint nCurTargetWords = cur_target.size();

    for (uint i=0; i < nCurTargetWords; i++) {
      uint tidx = cur_target[i];
      for (uint j=0; j < nCurSourceWords; j++) {
	
        dict[tidx][slookup[s](j,i)] += 1.0;
      }
    }
  }

  for (uint i=1; i < nTargetWords; i++) {
    dict[i] *= 1.0 / dict[i].sum();
  }
#endif
      
  //fractional counts used for EM-iterations
  NamedStorage1D<Math1D::Vector<double> > fcount(nTargetWords,MAKENAME(fcount));
  for (uint i=0; i < nTargetWords; i++) {
    fcount[i].resize(cooc[i].size());
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM 1 iteration #" << iter << std::endl;

    /*** a) compute fractional counts ***/
    
    for (uint i=0; i < nTargetWords; i++) {
      fcount[i].set_constant(0.0);
    }

    for (size_t s=0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint nCurSourceWords = cur_source.size();
      const uint nCurTargetWords = cur_target.size();
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];

      if (nCurSourceWords == 0)
        std::cerr << "WARNING: empty source sentence #" << s << std::endl;
      if (nCurTargetWords == 0)
        std::cerr << "WARNING: empty target sentence #" << s << std::endl;
      
      for (uint j=0; j < nCurSourceWords; j++) {
	
        const uint s_idx = source[s][j];

        double coeff = dict[0][s_idx-1]; // entry for empty word (the emtpy word is not listed, hence s_idx-1)
        for (uint i=0; i < nCurTargetWords; i++) {
          const uint t_idx = cur_target[i];
          coeff += dict[t_idx][cur_lookup(j,i)];
        }
        coeff = 1.0 / coeff;

        assert(!isnan(coeff));

        fcount[0][s_idx-1] += coeff * dict[0][s_idx-1];
        for (uint i=0; i < nCurTargetWords; i++) {
          const uint t_idx = cur_target[i];
          const uint k = cur_lookup(j,i);
          fcount[t_idx][k] += coeff * dict[t_idx][k];
        }
      }
    }

    std::cerr << "updating dict from counts" << std::endl;

    /*** update dict from counts ***/
    if (dict_weight_sum > 0.0) {
     
      for (uint i=0; i < nTargetWords; i++) {
	
        const double sum = fcount[i].sum();
        const double prev_sum = dict[i].sum();

        if (sum > 1e-307) {
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint k=0; k < fcount[i].size(); k++) {
            dict[i][k] = fcount[i][k] * prev_sum * inv_sum;
          }
        }
      }

      double alpha = 100.0;
      if (iter > 2)
        alpha = 1.0;
      if (iter > 5)
        alpha = 0.1;

      dict_m_step(fcount, prior_weight, dict, alpha, 45);
    }
    else {

      for (uint i=0; i < nTargetWords; i++) {
	
        const double sum = fcount[i].sum();
        if (sum > 1e-307) {
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint k=0; k < fcount[i].size(); k++) {
            dict[i][k] = fcount[i][k] * inv_sum;
          }
        }
        else {
          std::cerr << "WARNING : did not update dictionary entries for target word #" << i
                    << " because sum is " << sum << "( dict-size = " << dict[i].size() << " )" << std::endl;
        }
      }
    }

    std::cerr << "IBM 1 energy after iteration #" << iter << ": " 
              << ibm1_energy(source,slookup,target,dict,prior_weight) << std::endl;

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (size_t s=0; s < nSentences; s++) {

        if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

          nContributors++;

          //compute viterbi alignment
          Storage1D<uint> viterbi_alignment;
          compute_ibm1_viterbi_alignment(source[s], slookup[s], target[s], dict, viterbi_alignment);
  
          //add alignment error rate
          sum_aer += AER(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          sum_fmeasure += f_measure(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          nErrors += nDefiniteAlignmentErrors(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
        }
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM1 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM1 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM1 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;
    }


  } //end for-loop (iter)

}

void train_ibm1_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source, 
                               const Storage1D<Math2D::Matrix<uint> >& slookup,
                               const Storage1D<Storage1D<uint> >& target,
                               const CooccuringWordsType& cooc, 
                               uint nSourceWords, uint nTargetWords,
                               SingleWordDictionary& dict, uint nIter,
                               std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                               std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                               const floatSingleWordDictionary& prior_weight) {

  assert(cooc.size() == nTargetWords);
  dict.resize_dirty(nTargetWords);

  const size_t nSentences = source.size();
  assert(nSentences == target.size());
  
  //prepare dictionary
  for (uint i=0; i < nTargetWords; i++) {
    
    const uint size = cooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double) size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

  Math1D::Vector<double> slack_vector(nTargetWords,0.0);
  
#if 1
  for (uint i=1; i < nTargetWords; i++) {    
    dict[i].set_constant(0.0);
  }

  for (size_t s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    
    const uint nCurSourceWords = cur_source.size();
    const uint nCurTargetWords = cur_target.size();
    
    for (uint i=0; i < nCurTargetWords; i++) {
      uint tidx = cur_target[i];
      for (uint j=0; j < nCurSourceWords; j++) {
	
        dict[tidx][slookup[s](j,i)] += 1.0;
      }
    }
  }

  for (uint i=1; i < nTargetWords; i++) {
    dict[i] *= 1.0 / dict[i].sum();
  }
#endif

  double energy = ibm1_energy(source,slookup,target,dict,prior_weight);

  std::cerr << "initial energy: " << energy  << std::endl;
  
  SingleWordDictionary new_dict(nTargetWords,MAKENAME(new_dict));
  SingleWordDictionary hyp_dict(nTargetWords,MAKENAME(hyp_dict));
  
  for (uint i=0; i < nTargetWords; i++) {
    
    const uint size = cooc[i].size();
    new_dict[i].resize_dirty(size);
    hyp_dict[i].resize_dirty(size);
  }
  
  Math1D::Vector<double> new_slack_vector(nTargetWords,0.0);  

  double alpha = 0.5; //0.1; // 0.0001;
  //double alpha = 1.0;

  double line_reduction_factor = 0.5;

  uint nSuccessiveReductions = 0;

  for (uint iter = 1; iter <= nIter; iter++) {
    
    std::cerr << "starting IBM-1 gradient descent iteration #" << iter << std::endl;
    
    /***** calcuate gradients ****/
    
    SingleWordDictionary dict_grad(nTargetWords,MAKENAME(dict_grad));
    
    for (uint i=0; i < nTargetWords; i++) {
      
      const uint size = cooc[i].size();
      dict_grad[i].resize_dirty(size);
      dict_grad[i].set_constant(0.0);
    }
    
    for (size_t s=0; s < nSentences; s++) {
      
      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];
      
      for (uint j=0; j < curJ; j++) {
	
        uint s_idx = cur_source[j];

        double sum = dict[0][s_idx-1];
	
        for (uint i=0; i < curI; i++) 
          sum += dict[cur_target[i]][cur_lookup(j,i)];
	
        double cur_grad = -1.0 / sum;

        dict_grad[0][s_idx-1] += cur_grad;
        for (uint i=0; i < curI; i++) 
          dict_grad[cur_target[i]][cur_lookup(j,i)] += cur_grad;
      }
    }

    for (uint i=0; i < nTargetWords; i++) {
	
      const uint size = cooc[i].size();
      
      for (uint k=0; k < size; k++) {
        dict_grad[i][k] += prior_weight[i][k] * prob_pen_prime(dict[i][k]);
      }
    }

    /**** move in gradient direction ****/
    
    for (uint i=0; i < nTargetWords; i++) {

      for (uint k=0; k < dict[i].size(); k++) 
        new_dict[i][k] = dict[i][k] - alpha * dict_grad[i][k];
    }
    
    new_slack_vector = slack_vector;

    /**** reproject on the simplices [Michelot 1986]****/
    for (uint i=0; i < nTargetWords; i++) {

      const uint nCurWords = new_dict[i].size();

      projection_on_simplex_with_slack(new_dict[i].direct_access(),slack_vector[i],nCurWords);
    }
    
    double lambda = 1.0;
    double best_lambda = 1.0;

    double hyp_energy = ibm1_energy(source,slookup,target,new_dict,prior_weight);

    uint nInnerIter = 0;

    bool decreasing = true;

    while (hyp_energy > energy || decreasing) {

      nInnerIter++;

      if (hyp_energy <= 0.95*energy)
        break;

      if (hyp_energy < 0.99*energy && nInnerIter > 3)
        break;

      lambda *= line_reduction_factor;

      double inv_lambda = 1.0 - lambda;
      
      for (uint i=0; i < nTargetWords; i++) {
	
        for (uint k=0; k < dict[i].size(); k++) 
          hyp_dict[i][k] = inv_lambda * dict[i][k] + lambda * new_dict[i][k];
      }
      
      double new_energy = ibm1_energy(source,slookup,target,hyp_dict,prior_weight); 

      std::cerr << "new hyp: " << new_energy << ", previous: " << hyp_energy << std::endl;
      
      if (new_energy < hyp_energy) {
        hyp_energy = new_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;
    }

    if (nInnerIter > 4) {
      nSuccessiveReductions++;
    }
    else {
      nSuccessiveReductions = 0;
    }    

    if (nSuccessiveReductions > 15) {
      line_reduction_factor *= 0.9;
      nSuccessiveReductions = 0;
    }

    //     std::cerr << "alpha: " << alpha << std::endl;

    for (uint i=0; i < nTargetWords; i++) {

      double inv_lambda = 1.0 - best_lambda;
      
      for (uint k=0; k < dict[i].size(); k++) 
        dict[i][k] = inv_lambda * dict[i][k] + best_lambda * new_dict[i][k];

      slack_vector[i] = inv_lambda * slack_vector[i] + best_lambda * new_slack_vector[i];
    }

    double check_energy = ibm1_energy(source,slookup,target,dict,prior_weight);

    assert(fabs(check_energy - hyp_energy) < 0.0025);

    energy = hyp_energy;

    //     if (best_lambda == 1.0)
    //       alpha *= 1.5;
    //     else
    //       alpha *= 0.75;

    std::cerr << "energy: " << energy << std::endl;


    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (size_t s=0; s < nSentences; s++) {

        if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

          nContributors++;

          //compute viterbi alignment
          Storage1D<uint> viterbi_alignment;
          compute_ibm1_viterbi_alignment(source[s], slookup[s], target[s], dict, viterbi_alignment);
  
          //add alignment error rate
          sum_aer += AER(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          sum_fmeasure += f_measure(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          nErrors += nDefiniteAlignmentErrors(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
        }
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM1 Viterbi-AER after gd-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM1 Viterbi-fmeasure after gd-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM1 Viterbi-DAE/S after gd-iteration #" << iter << ": " << nErrors << std::endl;
    }

    std::cerr << "slack sum: " << slack_vector.sum() << std::endl;

  } //end for (iter)

  //std::cerr << "slack sum: " << slack_vector.sum() << std::endl;

}

void ibm1_viterbi_training(const Storage1D<Storage1D<uint> >& source, 
                           const Storage1D<Math2D::Matrix<uint> >& slookup,
                           const Storage1D<Storage1D<uint> >& target,
                           const CooccuringWordsType& cooc, 
                           uint nSourceWords, uint nTargetWords,
                           SingleWordDictionary& dict, uint nIterations,
                           std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
                           std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
                           const floatSingleWordDictionary& prior_weight) {

  Storage1D<Math1D::Vector<uint> > viterbi_alignment(source.size());

  const size_t nSentences = source.size();
  assert(nSentences == target.size());
  
  //prepare dictionary
  dict.resize(nTargetWords);
  for (uint i=0; i < nTargetWords; i++) {
    
    const size_t size = cooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double) size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

#if 1
  for (uint i=1; i < nTargetWords; i++) {
    dict[i].set_constant(0.0);
  }
  for (size_t s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint nCurSourceWords = cur_source.size();
    const uint nCurTargetWords = cur_target.size();

    for (uint i=0; i < nCurTargetWords; i++) {
      uint tidx = cur_target[i];
      for (uint j=0; j < nCurSourceWords; j++) {
	
        dict[tidx][slookup[s](j,i)] += 1.0;
      }
    }
  }

  for (uint i=1; i < nTargetWords; i++) {
    dict[i] *= 1.0 / dict[i].sum();
  }
#endif

  //counts of words
  NamedStorage1D<Math1D::Vector<uint> > dcount(nTargetWords,MAKENAME(dcount));

  for (uint i=0; i < nTargetWords; i++) {
    dcount[i].resize(cooc[i].size());
    dcount[i].set_constant(0);
  }

  double energy_offset = 0.0;
  for (size_t s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    
    viterbi_alignment[s].resize(cur_source.size());

    const uint nCurSourceWords = cur_source.size();
    const uint nCurTargetWords = cur_target.size();

    energy_offset += nCurSourceWords * std::log(nCurTargetWords + 1.0);
  }

  double last_energy = 1e300;

  for (uint iter = 1; iter <= nIterations; iter++) {

    for (uint i=0; i < nTargetWords; i++) {      
      dcount[i].set_constant(0);
    }

    double sum = 0.0;

    for (size_t s=0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint nCurSourceWords = cur_source.size();
      const uint nCurTargetWords = cur_target.size();
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];

      for (uint j=0; j < nCurSourceWords; j++) {
	
        const uint s_idx = source[s][j];

        double min = 1e50;
        uint arg_min = MAX_UINT;

        if (iter == 1) {

          min = -std::log(dict[0][s_idx-1]);
          arg_min = 0;

          for (uint i=0; i < nCurTargetWords; i++) {

            double hyp = -std::log(dict[cur_target[i]][cur_lookup(j,i)]);

            //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;
	    
            if (hyp < min) {
              min = hyp;
              arg_min = i+1;
            }
          }
        }
        else {
	  
          if (dict[0][s_idx-1] == 0.0) {
	
            min = 1e20;
          }
          else {
	    
            min = -std::log(dict[0][s_idx-1]);
          }
          arg_min = 0;
	  
          for (uint i=0; i < nCurTargetWords; i++) {
	    
            double hyp;
	    
            if (dict[cur_target[i]][cur_lookup(j,i)] == 0.0) {
	      
              hyp = 1e20;
            }
            else {
	      
              hyp = -std::log( dict[cur_target[i]][cur_lookup(j,i)]);
            }
	    
            //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;

            if (hyp < min) {
              min = hyp;
              arg_min = i+1;
            }
	    
          }
        }

        //std::cerr << "arg_min: " << arg_min << std::endl;

        sum += min;
	
        viterbi_alignment[s][j] = arg_min;

        if (arg_min == 0)
          dcount[0][s_idx-1]++;
        else
          dcount[cur_target[arg_min-1]][cur_lookup(j,arg_min-1)]++;
      }
    }
   
    //exit(1);

    sum += energy_offset;
    //std::cerr << "sum: " << sum << std::endl;


    /*** ICM phase ***/

    uint nSwitches = 0;

    if (true) {

      Math1D::Vector<uint> dict_sum(dcount.size());
      for (uint k=0; k < dcount.size(); k++)
       	dict_sum[k] = dcount[k].sum();
      

      for (size_t s=0; s < nSentences; s++) {

        if ((s%12500) == 0)
          std::cerr << "s: " << s << std::endl;

        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];

        const uint curJ = cur_source.size();
        const uint curI = cur_target.size();
	
        const Math2D::Matrix<uint>& cur_lookup = slookup[s];
	
        for (uint j=0; j < curJ; j++) {
	  
          uint cur_aj = viterbi_alignment[s][j];
          uint new_aj = cur_aj;

          uint cur_dict_num = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];
          uint cur_target_word = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];
          uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);

          double best_change = 1e300;

          for (uint i=0; i <= curI; i++) {
	    
            uint new_target_word = (i == 0) ? 0 : cur_target[i-1];

            if (cur_target_word != new_target_word) {
	      
              uint hyp_idx = (i == 0) ? cur_source[j]-1 : cur_lookup(j,i-1);

              uint hyp_dict_num = (i == 0) ? 0 : cur_target[i-1];
	      
              double change = 0.0;

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
	      
              assert(!isnan(change));
	      
              if (change < best_change) {

                best_change = change;
                new_aj = i;
              }
            }
          }

          Math1D::Vector<uint>& cur_dictcount = dcount[cur_dict_num]; 
          uint cur_dictsum = dict_sum[cur_dict_num]; 

          best_change -= double(cur_dictsum) * std::log(cur_dictsum);
          if (cur_dictsum > 1)
            best_change += double(cur_dictsum-1) * std::log(cur_dictsum-1.0);

          best_change -= - double(cur_dictcount[cur_idx]) * std::log(cur_dictcount[cur_idx]);

          if (cur_dictcount[cur_idx] > 1) {
            best_change += double(cur_dictcount[cur_idx]-1) * (-std::log(cur_dictcount[cur_idx]-1));
          }
          else
            best_change -= prior_weight[cur_dict_num][cur_idx];

          if (best_change < -1e-2 && new_aj != cur_aj) {

            nSwitches++;
	    
            uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);
            uint hyp_dict_num = (new_aj == 0) ? 0 : cur_target[new_aj-1];
	    
            Math1D::Vector<uint>& hyp_dictcount = dcount[hyp_dict_num];
	    
            uint hyp_idx = (new_aj == 0) ? cur_source[j]-1 : cur_lookup(j,new_aj-1);

            viterbi_alignment[s][j] = new_aj;
            cur_dictcount[cur_idx] -= 1;
            hyp_dictcount[hyp_idx] += 1;
            dict_sum[cur_dict_num] -= 1;
            dict_sum[hyp_dict_num] += 1;
          }
        }
      }
      
      std::cerr << nSwitches << " switches in ICM" << std::endl;
    }

    // Math1D::Vector<uint> count_count(6,0);

    // for (uint i=0; i < nTargetWords; i++) {      
    //   for (uint k=0; k < dcount[i].size(); k++) {
    // 	if (dcount[i][k] < count_count.size())
    // 	  count_count[dcount[i][k]]++;
    //   }
    // }

    // std::cerr << "count count (lower end): " << count_count << std::endl;

    /*** recompute the dictionary ***/
    double energy = energy_offset;

    double sum_sum = 0.0;

    for (uint i=0; i < nTargetWords; i++) {

      //std::cerr << "i: " << i << std::endl;

      const double sum = dcount[i].sum();

      sum_sum += sum;

      if (sum > 1e-307) {

        energy += sum * std::log(sum);

        const double inv_sum = 1.0 / sum;
        assert(!isnan(inv_sum));
	
        for (uint k=0; k < dcount[i].size(); k++) {
          dict[i][k] = dcount[i][k] * inv_sum;

          if (dcount[i][k] > 0) {
            energy -= dcount[i][k] * std::log(dcount[i][k]);
            energy += prior_weight[i][k]; 
          }
        }
      }
      else {
        dict[i].set_constant(0.0);
        //std::cerr << "WARNING : did not update dictionary entries because sum is " << sum << std::endl;
      }
    }
    
    //std::cerr << "number of total alignments: " << sum_sum << std::endl;
    //std::cerr.precision(10);
    std::cerr << "energy: " << (energy / nSentences) << std::endl;

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (size_t s=0; s < nSentences; s++) {

        if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

          nContributors++;

          //add alignment error rate
          sum_aer += AER(viterbi_alignment[s],sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          sum_fmeasure += f_measure(viterbi_alignment[s],sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
          nErrors += nDefiniteAlignmentErrors(viterbi_alignment[s],sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
        }
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM1 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM1 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM1 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;

      if (nSwitches == 0 && fabs(last_energy-energy) < 1e-4) {
        std::cerr << "LOCAL MINIMUM => break." << std::endl;
        break;
      }

      last_energy = energy;
    }
  }
}



