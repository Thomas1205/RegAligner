/*** written by Thomas Schoenemann. Started as a private person without employment, October 2009 
 *** continued at Lund University, Sweden, 2010, as a private person, and at the University of Düsseldorf, Germany, 2012 ***/

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

#ifdef HAS_CBC
#include "sparse_matrix_description.hh"
#include "OsiClpSolverInterface.hpp"
#include "CbcModel.hpp"
#include "CglGomory/CglGomory.hpp"
#endif


IBM1Options::IBM1Options(uint nSourceWords,uint nTargetWords,
                         std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                         std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments) :
  nIterations_(5), smoothed_l0_(false), l0_beta_(1.0), print_energy_(true), 
  nSourceWords_(nSourceWords), nTargetWords_(nTargetWords), dict_m_step_iter_(45),
  sure_ref_alignments_(sure_ref_alignments), possible_ref_alignments_(possible_ref_alignments) {}


double ibm1_perplexity( const Storage1D<Storage1D<uint> >& source,
                        const LookupTable& slookup,
                        const Storage1D< Storage1D<uint> >& target,
                        const SingleWordDictionary& dict,
                        const CooccuringWordsType& wcooc, uint nSourceWords) {

  double sum = 0.0;

  SingleLookupTable aux_lookup;

  const size_t nSentences = target.size();
  assert(slookup.size() == nSentences);

  size_t nActualSentences = nSentences;

  for (size_t s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);


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
                    const LookupTable& slookup,
                    const Storage1D< Storage1D<uint> >& target,
                    const SingleWordDictionary& dict,
                    const CooccuringWordsType& wcooc, uint nSourceWords,
                    const floatSingleWordDictionary& prior_weight,
                    bool smoothed_l0 = false, double l0_beta = 1.0) {

  double energy = 0.0; 

  for (uint i=0; i < dict.size(); i++) {

    const uint size = dict[i].size();
    
    for (uint k=0; k < size; k++) {
      if (smoothed_l0)
        energy += prior_weight[i][k] * prob_penalty(dict[i][k],l0_beta);
      else
        energy += prior_weight[i][k] * dict[i][k];
    }
  }

  energy /= target.size(); //since the perplexity is also divided by that amount
  
  energy += ibm1_perplexity(source, slookup, target, dict, wcooc, nSourceWords);

  return energy;
}

void train_ibm1(const Storage1D<Storage1D<uint> >& source, 
                const LookupTable& slookup,
                const Storage1D<Storage1D<uint> >& target, 
                const CooccuringWordsType& wcooc,
                SingleWordDictionary& dict,
                const floatSingleWordDictionary& prior_weight, 
                IBM1Options& options) {
  
  uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  assert(wcooc.size() == options.nTargetWords_);
  dict.resize_dirty(options.nTargetWords_);

  const size_t nSentences = source.size();
  assert(nSentences == target.size());
  
  double dict_weight_sum = 0.0;
  for (uint i=0; i < options.nTargetWords_; i++) {
    dict_weight_sum += fabs(prior_weight[i].sum());
  }

  const uint nSourceWords = options.nSourceWords_;

  SingleLookupTable aux_lookup;

  //prepare dictionary
  for (uint i=0; i < options.nTargetWords_; i++) {
    
    const uint size = (i == 0) ? options.nSourceWords_-1 : wcooc[i].size();
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

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);

    for (uint i=0; i < curI; i++) {
      uint tidx = cur_target[i];
      for (uint j=0; j < curJ; j++) {
	
        dict[tidx][cur_lookup(j,i)] += 1.0;
      }
    }
  }

  for (uint i=1; i < nTargetWords; i++) {
    double sum = dict[i].sum();
    if (sum > 1e-305)
      dict[i] *= 1.0 / sum;
  }
#endif
      
  //fractional counts used for EM-iterations
  NamedStorage1D<Math1D::Vector<double> > fcount(options.nTargetWords_,MAKENAME(fcount));
  for (uint i=0; i < options.nTargetWords_; i++) {
    fcount[i].resize(dict[i].size());
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting IBM-1 EM-iteration #" << iter << std::endl;

    /*** a) compute fractional counts ***/
    
    for (uint i=0; i < options.nTargetWords_; i++) {
      fcount[i].set_constant(0.0);
    }

    for (size_t s=0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint nCurSourceWords = cur_source.size();
      const uint nCurTargetWords = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);

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
    } //loop over sentences finished

    std::cerr << "updating dict from counts" << std::endl;

    /*** update dict from counts ***/

    update_dict_from_counts(fcount, prior_weight, dict_weight_sum, iter, 
			    smoothed_l0, l0_beta, options.dict_m_step_iter_, dict);

    if (options.print_energy_) {
      std::cerr << "IBM-1 energy after iteration #" << iter << ": " 
                << ibm1_energy(source,slookup,target,dict,wcooc,nSourceWords,prior_weight,smoothed_l0,l0_beta) << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = options.possible_ref_alignments_.begin();
          it != options.possible_ref_alignments_.end(); it ++) {

        uint s = it->first-1;

	if (s >= nSentences)
	  break;

        nContributors++;

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s],target[s],wcooc,nSourceWords,slookup[s],aux_lookup);

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm1_viterbi_alignment(source[s], cur_lookup, target[s], dict, viterbi_alignment);
        
        //add alignment error rate
        sum_aer += AER(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        sum_fmeasure += f_measure(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM-1 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;
    }


  } //end for (iter)

}

void train_ibm1_gd_stepcontrol(const Storage1D<Storage1D<uint> >& source, 
                               const LookupTable& slookup,
                               const Storage1D<Storage1D<uint> >& target,
                               const CooccuringWordsType& wcooc, 
                               SingleWordDictionary& dict, //uint nIter,
                               const floatSingleWordDictionary& prior_weight, 
                               IBM1Options& options) {

  uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  assert(wcooc.size() == options.nTargetWords_);
  dict.resize_dirty(options.nTargetWords_);

  const size_t nSentences = source.size();
  assert(nSentences == target.size());
  
  //prepare dictionary
  for (uint i=0; i < options.nTargetWords_; i++) {
    
    const uint size = (i == 0) ? options.nSourceWords_-1 : wcooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double) size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

  Math1D::Vector<double> slack_vector(options.nTargetWords_,0.0);

  SingleLookupTable aux_lookup;

  const uint nSourceWords = options.nSourceWords_;
  
#if 1
  for (uint i=1; i < options.nTargetWords_; i++) {    
    dict[i].set_constant(0.0);
  }

  for (size_t s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    
    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);
    
    for (uint i=0; i < curI; i++) {
      uint tidx = cur_target[i];
      for (uint j=0; j < curJ; j++) {
	
        dict[tidx][cur_lookup(j,i)] += 1.0;
      }
    }
  }

  for (uint i=1; i < options.nTargetWords_; i++) {
    double sum = dict[i].sum();
    if (sum > 1e-305)
      dict[i] *= 1.0 / sum;
  }
#endif

  double energy = ibm1_energy(source,slookup,target,dict,wcooc,nSourceWords,prior_weight,smoothed_l0,l0_beta);

  std::cerr << "initial energy: " << energy  << std::endl;
  
  SingleWordDictionary new_dict(options.nTargetWords_,MAKENAME(new_dict));
  SingleWordDictionary hyp_dict(options.nTargetWords_,MAKENAME(hyp_dict));
  
  for (uint i=0; i < options.nTargetWords_; i++) {
    
    const uint size = dict[i].size();
    new_dict[i].resize_dirty(size);
    hyp_dict[i].resize_dirty(size);
  }
  
  Math1D::Vector<double> new_slack_vector(options.nTargetWords_,0.0);  

  double alpha = 100.0;

  double line_reduction_factor = 0.5;

  uint nSuccessiveReductions = 0;

  double best_lower_bound = -1e300;

  SingleWordDictionary dict_grad(options.nTargetWords_,MAKENAME(dict_grad));

  for (uint i=0; i < options.nTargetWords_; i++) {
      
    const uint size = dict[i].size();
    dict_grad[i].resize_dirty(size);
  }

  for (uint iter = 1; iter <= nIter; iter++) {
    
    std::cerr << "starting IBM-1 gradient descent iteration #" << iter << std::endl;
    
    /***** calcuate gradients ****/
    
    for (uint i=0; i < options.nTargetWords_; i++) {
      dict_grad[i].set_constant(0.0);
    }
    
    for (size_t s=0; s < nSentences; s++) {
      
      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);

      
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

    for (uint i=0; i < options.nTargetWords_; i++) {
	
      const uint size = dict[i].size();
      
      for (uint k=0; k < size; k++) {
        if (smoothed_l0)
          dict_grad[i][k] += prior_weight[i][k] * prob_pen_prime(dict[i][k],l0_beta);
        else 
          dict_grad[i][k] += prior_weight[i][k];
      }
    }

    /*** compute lower bound ****/
    
    double lower_bound = energy;
    for (uint i=0; i < options.nTargetWords_; i++) {

      //if (regularity_weight != 0.0)
      if (true)
        lower_bound += std::min(0.0,dict_grad[i].min());
      else
        lower_bound += dict_grad[i].min();
      lower_bound -= dict_grad[i] % dict[i];
    }

    best_lower_bound = std::max(best_lower_bound, lower_bound);
    
    std::cerr << "lower bound: " << lower_bound << ", best known: " << best_lower_bound << std::endl;

    /**** move in gradient direction ****/
    double real_alpha = alpha;

    double sqr_grad_norm = 0.0;
    for (uint i=0; i < options.nTargetWords_; i++)
      sqr_grad_norm +=  dict_grad[i].sqr_norm();
    real_alpha /= sqrt(sqr_grad_norm); 

    // double highest_norm = 0.0;
    // for (uint i=0; i < options.nTargetWords_; i++)
    //   highest_norm = std::max(highest_norm,dict_grad[i].sqr_norm());
    // real_alpha /= sqrt(highest_norm);

    for (uint i=0; i < options.nTargetWords_; i++) {

      for (uint k=0; k < dict[i].size(); k++) 
        new_dict[i][k] = dict[i][k] - real_alpha * dict_grad[i][k];
    }
    
    //if (regularity_weight != 0.0)
    if (true)
      new_slack_vector = slack_vector;

    /**** reproject on the simplices [Michelot 1986]****/
    for (uint i=0; i < options.nTargetWords_; i++) {

      const uint nCurWords = new_dict[i].size();

      //if (regularity_weight != 0.0)
      if (true)
        projection_on_simplex_with_slack(new_dict[i].direct_access(),slack_vector[i],nCurWords);
      else
        projection_on_simplex(new_dict[i].direct_access(),nCurWords);

    }

    double lambda = 1.0;
    double best_lambda = 1.0;

    double hyp_energy = 1e300; 

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
      
      for (uint i=0; i < options.nTargetWords_; i++) {
	
        for (uint k=0; k < dict[i].size(); k++) 
          hyp_dict[i][k] = inv_lambda * dict[i][k] + lambda * new_dict[i][k];
      }
      
      double new_energy = ibm1_energy(source,slookup,target,hyp_dict,wcooc,nSourceWords,prior_weight,smoothed_l0,l0_beta); 

      std::cerr << "new hyp: " << new_energy << ", previous: " << hyp_energy << std::endl;
      
      if (new_energy < hyp_energy) {
        hyp_energy = new_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;
    }

    //EXPERIMENTAL
#if 0
    if (nInnerIter > 4)
      alpha *= 1.5;
#endif
    //END_EXPERIMENTAL

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

    for (uint i=0; i < options.nTargetWords_; i++) {

      double inv_lambda = 1.0 - best_lambda;
      
      for (uint k=0; k < dict[i].size(); k++) {
	const double cur_new_dict = inv_lambda * dict[i][k] + best_lambda * new_dict[i][k];

        dict[i][k] = cur_new_dict;
      }

      //if (regularity_weight > 0.0)
      if (true)
        slack_vector[i] = inv_lambda * slack_vector[i] + best_lambda * new_slack_vector[i];
    }

#ifndef NDEBUG
    double check_energy = ibm1_energy(source,slookup,target,dict,wcooc,nSourceWords,prior_weight,smoothed_l0,l0_beta);
    assert(fabs(check_energy - hyp_energy) < 0.0025);
#endif
    
    energy = hyp_energy;

    if (options.print_energy_)
      std::cerr << "energy: " << energy << std::endl;


    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = options.possible_ref_alignments_.begin();
          it != options.possible_ref_alignments_.end(); it ++) {

        uint s = it->first-1;

	if (s >= nSentences)
	  break;

        nContributors++;

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s],target[s],wcooc,nSourceWords,slookup[s],aux_lookup);

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm1_viterbi_alignment(source[s], cur_lookup, target[s], dict, viterbi_alignment);
        
        //add alignment error rate
        sum_aer += AER(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        sum_fmeasure += f_measure(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment,options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM-1 Viterbi-AER after gd-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Viterbi-fmeasure after gd-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Viterbi-DAE/S after gd-iteration #" << iter << ": " << nErrors << std::endl;
    }

    std::cerr << "slack sum: " << slack_vector.sum() << std::endl;

  } //end for (iter)
}


void ibm1_viterbi_training(const Storage1D<Storage1D<uint> >& source, 
                           const LookupTable& slookup,
                           const Storage1D<Storage1D<uint> >& target,
                           const CooccuringWordsType& wcooc, 
                           SingleWordDictionary& dict,
                           const floatSingleWordDictionary& prior_weight,
                           IBM1Options& options,
			   const Math1D::Vector<double>& log_table) {

  uint nIterations = options.nIterations_;

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  const size_t nSentences = source.size();
  assert(nSentences == target.size());
  
  //prepare dictionary
  dict.resize(options.nTargetWords_);
  for (uint i=0; i < options.nTargetWords_; i++) {
    
    const uint size = (i == 0) ? options.nSourceWords_-1 : wcooc[i].size();
    dict[i].resize_dirty(size);
    dict[i].set_constant(1.0 / ((double) size));
  }
  dict[0].set_constant(1.0 / dict[0].size());

  SingleLookupTable aux_lookup;

  const uint nSourceWords = options.nSourceWords_;

#if 1
  for (uint i=1; i < options.nTargetWords_; i++) {
    dict[i].set_constant(0.0);
  }
  for (size_t s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);

    for (uint i=0; i < curI; i++) {
      uint tidx = cur_target[i];
      for (uint j=0; j < curJ; j++) {
	
        dict[tidx][cur_lookup(j,i)] += 1.0;
      }
    }
  }

  for (uint i=1; i < options.nTargetWords_; i++) {
    dict[i] *= 1.0 / dict[i].sum();
  }
#endif

  //counts of words
  NamedStorage1D<Math1D::Vector<uint> > dcount(options.nTargetWords_,MAKENAME(dcount));

  for (uint i=0; i < options.nTargetWords_; i++) {
    dcount[i].resize(dict[i].size());
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

    std::cerr << "starting IBM-1 Viterbi iteration #" << iter << std::endl;

    for (uint i=0; i < options.nTargetWords_; i++) {      
      dcount[i].set_constant(0);
    }

    double sum = 0.0;

    for (size_t s=0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint nCurSourceWords = cur_source.size();
      const uint nCurTargetWords = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);

      for (uint j=0; j < nCurSourceWords; j++) {
	
        const uint s_idx = source[s][j];

        double min = 1e50;
        uint arg_min = MAX_UINT;

        if (iter == 1) {

          min = -std::log(dict[0][s_idx-1]);
          arg_min = 0;

          for (uint i=0; i < nCurTargetWords; i++) {

            double hyp = -std::log(dict[cur_target[i]][cur_lookup(j,i)]);
	    
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

            if (hyp < min) {
              min = hyp;
              arg_min = i+1;
            }
	    
          }
        }

        sum += min;
	
        viterbi_alignment[s][j] = arg_min;

        if (arg_min == 0)
          dcount[0][s_idx-1]++;
        else
          dcount[cur_target[arg_min-1]][cur_lookup(j,arg_min-1)]++;
      }
    }
   
    sum += energy_offset;


    /*** ICM phase ***/

    uint nSwitches = 0;

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
      
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source,cur_target,wcooc,nSourceWords,slookup[s],aux_lookup);
	
      Math1D::Vector<AlignBaseType>& cur_alignment = viterbi_alignment[s];

      for (uint j=0; j < curJ; j++) {
	
	ushort cur_aj = cur_alignment[j];
	ushort new_aj = cur_aj;
	
	uint cur_dict_num = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];
	uint cur_target_word = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];
	uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);

	double best_change = 1e300;

	for (uint i=0; i <= curI; i++) {
	  
	  uint new_target_word = (i == 0) ? 0 : cur_target[i-1];
	  
	  //NOTE: IBM-1 scores don't change when the two words in question are identical
	  if (cur_target_word != new_target_word) {
	    
	    uint hyp_idx = (i == 0) ? cur_source[j]-1 : cur_lookup(j,i-1);
	    
	    uint hyp_dict_num = (i == 0) ? 0 : cur_target[i-1];
	    
	    double change = 0.0;
	    
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
	    
	    assert(!isnan(change));
	    
	    if (change < best_change) {
	      
	      best_change = change;
	      new_aj = i;
	    }
	  }
	}
	
	Math1D::Vector<uint>& cur_dictcount = dcount[cur_dict_num]; 
	uint cur_dictsum = dict_sum[cur_dict_num]; 
	
	best_change -= double(cur_dictsum) * log_table[cur_dictsum];
	if (cur_dictsum > 1)
	  best_change += double(cur_dictsum-1) * log_table[cur_dictsum-1];
	
	best_change -= - double(cur_dictcount[cur_idx]) * log_table[cur_dictcount[cur_idx]];


	if (cur_dictcount[cur_idx] > 1) {
	  best_change += double(cur_dictcount[cur_idx]-1) * (-log_table[cur_dictcount[cur_idx]-1]);
	}
	else
	  best_change -= prior_weight[cur_dict_num][cur_idx];
	
	if (best_change < -1e-2 && new_aj != cur_aj) {
	  
	  nSwitches++;
	  
	  uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);
	  uint hyp_dict_num = (new_aj == 0) ? 0 : cur_target[new_aj-1];
	  
	  Math1D::Vector<uint>& hyp_dictcount = dcount[hyp_dict_num];
	  
	  uint hyp_idx = (new_aj == 0) ? cur_source[j]-1 : cur_lookup(j,new_aj-1);

	  cur_alignment[j] = new_aj;
	  cur_dictcount[cur_idx] -= 1;
	  hyp_dictcount[hyp_idx] += 1;
	  dict_sum[cur_dict_num] -= 1;
	  dict_sum[hyp_dict_num] += 1;
	}
      }
    }
    
    std::cerr << nSwitches << " switches in ICM" << std::endl;

    // Math1D::Vector<uint> count_count(6,0);

    // for (uint i=0; i < options.nTargetWords_; i++) {      
    //   for (uint k=0; k < dcount[i].size(); k++) {
    //     if (dcount[i][k] < count_count.size())
    //       count_count[dcount[i][k]]++;
    //   }
    // }

    // std::cerr << "count count (lower end): " << count_count << std::endl;

    /*** recompute the dictionary ***/
    double energy = energy_offset;

    double sum_sum = 0.0;

    for (uint i=0; i < options.nTargetWords_; i++) {

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

    energy /= nSentences;
    for (uint i=0; i < dcount.size(); i++)
      for (uint k=0; k < dcount[i].size(); k++)
        if (dcount[i][k] > 0)
          //we need to divide as we are truly minimizing the perplexity WITHOUT division plus the l0-term
          energy += prior_weight[i][k] / nSentences; 


    if (options.print_energy_) {
      std::cerr << "energy: " << energy << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = options.possible_ref_alignments_.begin();
          it != options.possible_ref_alignments_.end(); it ++) {

        uint s = it->first-1;

	if (s >= nSentences)
	  break;

        nContributors++;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment[s],options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        sum_fmeasure += f_measure(viterbi_alignment[s],options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment[s],options.sure_ref_alignments_[s+1],options.possible_ref_alignments_[s+1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM-1 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM-1 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM-1 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;

      if (nSwitches == 0 && fabs(last_energy-energy) < 1e-4) {
        std::cerr << "LOCAL MINIMUM => break." << std::endl;
        break;
      }

      last_energy = energy;
    }
  }
}

