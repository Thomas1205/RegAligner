/*** written by Thomas Schoenemann. Started as a private person without employment, November 2009 ***/
/*** continued at Lund University, Sweden, January 2010 - March 2011, as a private person and ***/
/*** at the University of Düsseldorf, Germany, January - September 2012 ***/

#include "singleword_fertility_training.hh"
#include "combinatoric.hh"
#include "alignment_error_rate.hh"
#include "timing.hh"
#include "alignment_computation.hh"
#include "training_common.hh" // for get_wordlookup()

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"
#include "stl_util.hh"

/********************** implementation of FertilityModelTrainer *******************************/

FertilityModelTrainer::FertilityModelTrainer(const Storage1D<Storage1D<uint> >& source_sentence,
                                             const Storage1D<Math2D::Matrix<uint, ushort> >& slookup,
                                             const Storage1D<Storage1D<uint> >& target_sentence,
                                             SingleWordDictionary& dict,
                                             const CooccuringWordsType& wcooc,
                                             uint nSourceWords, uint nTargetWords,
					     const floatSingleWordDictionary& prior_weight,
					     bool och_ney_empty_word, bool smoothed_l0,
					     double l0_beta, double l0_fertpen,
                                             const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                                             const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
					     const Math1D::Vector<double>& log_table, uint fertility_limit) :
  uncovered_set_(MAKENAME(uncovered_sets_)), predecessor_sets_(MAKENAME(predecessor_sets_)), 
  nUncoveredPositions_(MAKENAME(nUncoveredPositions_)), j_before_end_skips_(MAKENAME(j_before_end_skips_)),
  first_set_(MAKENAME(first_set_)), next_set_idx_(0), coverage_state_(MAKENAME(coverage_state_)),
  first_state_(MAKENAME(first_state_)), predecessor_coverage_states_(MAKENAME(predecessor_coverage_states_)),
  source_sentence_(source_sentence), slookup_(slookup), target_sentence_(target_sentence), 
  wcooc_(wcooc), dict_(dict), nSourceWords_(nSourceWords), nTargetWords_(nTargetWords), iter_offs_(0),
  och_ney_empty_word_(och_ney_empty_word), smoothed_l0_(smoothed_l0), l0_beta_(l0_beta), l0_fertpen_(l0_fertpen),
  prior_weight_(prior_weight), fertility_prob_(nTargetWords,MAKENAME(fertility_prob_)), 
  best_known_alignment_(MAKENAME(best_known_alignment_)),
  sure_ref_alignments_(sure_ref_alignments), possible_ref_alignments_(possible_ref_alignments), log_table_(log_table)
{

  Math1D::Vector<uint> max_fertility(nTargetWords,0);

  maxJ_ = 0;
  maxI_ = 0;
  fertility_limit_ = fertility_limit;

  p_zero_ = 0.02;
  p_nonzero_ = 0.98;

  fix_p0_ = false;
 
  for (size_t s=0; s < source_sentence.size(); s++) {

    const uint curJ = source_sentence[s].size();
    const uint curI = target_sentence[s].size();

    if (maxJ_ < curJ)
      maxJ_ = curJ;
    if (maxI_ < curI)
      maxI_ = curI;

    if (max_fertility[0] < curJ)
      max_fertility[0] = curJ;

    for (uint i = 0; i < curI; i++) {

      const uint t_idx = target_sentence[s][i];

      if (max_fertility[t_idx] < curJ)
        max_fertility[t_idx] = curJ;
    }
  }

  ld_fac_.resize(maxJ_+1);
  ld_fac_[0] = 1.0;
  for (uint c=1; c < ld_fac_.size(); c++)
    ld_fac_[c] = ld_fac_[c-1] * c;
  
  for (uint i=0; i < nTargetWords; i++) {
    fertility_prob_[i].resize_dirty(max_fertility[i]+1);
    fertility_prob_[i].set_constant(1.0 / (max_fertility[i]+1));
  }

  best_known_alignment_.resize(source_sentence.size());
  for (size_t s=0; s < source_sentence.size(); s++)
    best_known_alignment_[s].resize(source_sentence[s].size(),0);

  //compute_uncovered_sets(3);
}

void FertilityModelTrainer::fix_p0(double p0) {
  p_zero_ = p0;
  p_nonzero_ = 1.0 - p0;
  fix_p0_ = true;
}

double FertilityModelTrainer::p_zero() const {
  return p_zero_;
}

void FertilityModelTrainer::release_memory() {

  best_known_alignment_.resize(0);
  fertility_prob_.resize(0);
}

const NamedStorage1D<Math1D::Vector<double> >& FertilityModelTrainer::fertility_prob() const {
  return fertility_prob_;
}

void FertilityModelTrainer::write_fertilities(std::string filename) {

  std::ofstream out(filename.c_str());

  for (uint k=0; k < fertility_prob_.size(); k++) {
    
    for (uint l=0; l < fertility_prob_[k].size(); l++)
      out << fertility_prob_[k][l] << " ";

    out << std::endl;
  }
}

const NamedStorage1D<Math1D::Vector<AlignBaseType> >& FertilityModelTrainer::best_alignments() const {
  return best_known_alignment_;
}

void FertilityModelTrainer::set_fertility_limit(uint new_limit) {
  fertility_limit_ = new_limit;
}

double FertilityModelTrainer::AER() {

  double sum_aer = 0.0;
  uint nContributors = 0;
  

  for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = possible_ref_alignments_.begin();
      it != possible_ref_alignments_.end(); it ++) {
    
    uint s = it->first-1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add alignment error rate
    sum_aer += ::AER(best_known_alignment_[s],sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]);
  }
  
  sum_aer *= 100.0 / nContributors;
  return sum_aer;
}

double FertilityModelTrainer::AER(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments) {

  double sum_aer = 0.0;
  uint nContributors = 0;

  for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = possible_ref_alignments_.begin();
      it != possible_ref_alignments_.end(); it ++) {
    
    uint s = it->first-1;

    if (s >= source_sentence_.size())
      break;
  
    nContributors++;
    //add alignment error rate
    sum_aer += ::AER(alignments[s],sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]);
  }
  
  sum_aer *= 100.0 / nContributors;
  return sum_aer;
}

double FertilityModelTrainer::f_measure(double alpha) {

  double sum_fmeasure = 0.0;
  uint nContributors = 0;
  
  for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = possible_ref_alignments_.begin();
      it != possible_ref_alignments_.end(); it ++) {
    
    uint s = it->first-1;

    if (s >= source_sentence_.size())
      break;
      
    nContributors++;
    //add f-measure
    
    // std::cerr << "s: " << s << ", " << ::f_measure(uint_alignment,sure_ref_alignments_[s+1],possible_ref_alignments_[s+1], alpha) << std::endl;
    // std::cerr << "precision: " << ::precision(uint_alignment,sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]) << std::endl;
    // std::cerr << "recall: " << ::recall(uint_alignment,sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]) << std::endl;
    // std::cerr << "alpha: " << alpha << std::endl;
    // std::cerr << "sure alignments: " << sure_ref_alignments_[s+1] << std::endl;
    // std::cerr << "possible alignments: " << possible_ref_alignments_[s+1] << std::endl;
    // std::cerr << "computed alignment: " << uint_alignment << std::endl;
    
    sum_fmeasure += ::f_measure(best_known_alignment_[s],sure_ref_alignments_[s+1],possible_ref_alignments_[s+1], alpha);
  }
  
  sum_fmeasure /= nContributors;
  return sum_fmeasure;
}

double FertilityModelTrainer::f_measure(const Storage1D<Math1D::Vector<AlignBaseType> >& alignment, double alpha) {

  double sum_fmeasure = 0.0;
  uint nContributors = 0;
  
  for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = possible_ref_alignments_.begin();
      it != possible_ref_alignments_.end(); it ++) {
    
    uint s = it->first-1;

    if (s >= source_sentence_.size())
      break;
      
    nContributors++;
    //add f-measure
    
    sum_fmeasure += ::f_measure(alignment[s],sure_ref_alignments_[s+1],possible_ref_alignments_[s+1], alpha);
  }
  
  sum_fmeasure /= nContributors;
  return sum_fmeasure;
}

double FertilityModelTrainer::DAE_S() {

  double sum_errors = 0.0;
  uint nContributors = 0;
  
  for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = possible_ref_alignments_.begin();
      it != possible_ref_alignments_.end(); it ++) {
    
    uint s = it->first-1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add DAE/S
    sum_errors += ::nDefiniteAlignmentErrors(best_known_alignment_[s],sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]);
  }
  
  sum_errors /= nContributors;
  return sum_errors;
}

double FertilityModelTrainer::DAE_S(const Storage1D<Math1D::Vector<AlignBaseType> >& alignment) {

  double sum_errors = 0.0;
  uint nContributors = 0;
  
  for(std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::iterator it = possible_ref_alignments_.begin();
      it != possible_ref_alignments_.end(); it ++) {
    
    uint s = it->first-1;

    if (s >= source_sentence_.size())
      break;

    nContributors++;
    //add DAE/S
    sum_errors += ::nDefiniteAlignmentErrors(alignment[s],sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]);
  }
  
  sum_errors /= nContributors;
  return sum_errors;
}

long double FertilityModelTrainer::simulate_hmm_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
							     const SingleLookupTable& lookup,const HmmWrapper& hmm_wrapper,
							     Math1D::Vector<uint>& fertility, 
							     Math2D::Matrix<long double>& expansion_prob,
							     Math2D::Matrix<long double>& swap_prob, 
							     Math1D::Vector<AlignBaseType>& alignment) {

  const uint curJ = source.size();
  const uint curI = target.size();

  //note: here the index 0 will represent alignments to 0. we will have to convert to the proper representations 
  // when calling routines below
  long double best_prob = compute_ehmm_viterbi_alignment(source,lookup,target,dict_,hmm_wrapper.align_model_[curI-1],
							 hmm_wrapper.initial_prob_[curI-1],alignment,
							 hmm_wrapper.hmm_options_.align_type_,false,false);

  

  fertility.resize(curI+1);
  fertility.set_constant(0);

  for (uint j=0; j < curJ; j++) {
    fertility[alignment[j]]++;
  }

  bool changed = make_alignment_feasible(source, target, lookup, alignment);

  Math1D::Vector<AlignBaseType> internal_hyp_alignment(curJ);

  if (changed) {

    external2internal_hmm_alignment(alignment, curI, hmm_wrapper.hmm_options_, internal_hyp_alignment);

    best_prob = hmm_alignment_prob(source,lookup,target,dict_,hmm_wrapper.align_model_,
				   hmm_wrapper.initial_prob_, internal_hyp_alignment, true);
  }

  //NOTE: lots of room for speed-ups here!

  swap_prob.resize(curJ,curJ);
  expansion_prob.resize(curJ,curI+1);
  swap_prob.set_constant(0.0);
  expansion_prob.set_constant(0.0);

  //std::cerr << "J: " << curJ << ", I: " << curI << std::endl;
  //std::cerr << "base alignment: " << alignment << std::endl;

  Math1D::Vector<AlignBaseType> hyp_alignment = alignment;

  //a) expansion moves
  for (uint j=0; j < curJ; j++) {
    
    //std::cerr << "j: " << j << std::endl;

    const uint cur_aj = alignment[j];

    for (uint i=0; i <= curI; i++) {

      //std::cerr << "i: " << i << std::endl;

      if (i == 0 && 2*fertility[0]+2 > curJ)
	continue;

      if (i != cur_aj) {

	hyp_alignment[j] = i;

	//now convert to internal mode
	external2internal_hmm_alignment(hyp_alignment, curI, hmm_wrapper.hmm_options_, internal_hyp_alignment);
	
	expansion_prob(j,i) = hmm_alignment_prob(source,lookup,target,dict_,hmm_wrapper.align_model_,
						 hmm_wrapper.initial_prob_, internal_hyp_alignment, true);
      }
    }

    //restore for next iteration
    hyp_alignment[j] = cur_aj;
  }

  //b) swap moves
  for (uint j1=0; j1 < curJ-1; j1++) {

    //std::cerr << "j1: " << j1 << std::endl;

    const uint cur_aj1 = alignment[j1];

    for (uint j2=j1+1; j2 < curJ; j2++) {

      //std::cerr << "j2: " << j2 << std::endl;

      const uint cur_aj2 = alignment[j2];

      if (cur_aj1 != cur_aj2) {

	std::swap(hyp_alignment[j1],hyp_alignment[j2]);

	//now convert to internal mode
	external2internal_hmm_alignment(hyp_alignment, curI, hmm_wrapper.hmm_options_, internal_hyp_alignment);
	
	double cur_swap_prob = hmm_alignment_prob(source,lookup,target,dict_,hmm_wrapper.align_model_,
						  hmm_wrapper.initial_prob_, internal_hyp_alignment, true);

	swap_prob(j1,j2) = cur_swap_prob;
	swap_prob(j2,j1) = cur_swap_prob;

	//reverse for next iteration
	std::swap(hyp_alignment[j1],hyp_alignment[j2]);
      }
    }
  }

  return best_prob;
}


void FertilityModelTrainer::print_uncovered_set(uint state) const {

  for (uint k=0; k < uncovered_set_.xDim(); k++) {

    if (uncovered_set_(k,state) == MAX_USHORT)
      std::cerr << "-";
    else
      std::cerr << uncovered_set_(k,state);
    std::cerr << ",";
  }
}

uint FertilityModelTrainer::nUncoveredPositions(uint state) const {

  uint result = uncovered_set_.xDim();

  for (uint k=0; k < uncovered_set_.xDim(); k++) {
    if (uncovered_set_(k,state) == MAX_USHORT)
      result--;
    else
      break;
  }

  return result;
}

void FertilityModelTrainer::cover(uint level) {

  //  std::cerr << "*****cover(" << level << ")" << std::endl;

  if (level == 0) {
    next_set_idx_++;
    return;
  }

  const uint ref_set_idx = next_set_idx_;

  next_set_idx_++; //to account for sets which are not fully filled
  assert(next_set_idx_ <= uncovered_set_.yDim());

  const uint ref_j = uncovered_set_(level,ref_set_idx);
  //std::cerr << "ref_j: " << ref_j << std::endl;
  //std::cerr << "ref_line: ";
  //   for (uint k=0; k < uncovered_set_.xDim(); k++) {

  //     if (uncovered_set_(k,ref_set_idx) == MAX_USHORT)
  //       std::cerr << "-";
  //     else
  //       std::cerr << uncovered_set_(k,ref_set_idx);
  //     std::cerr << ",";
  //   }
  //   std::cerr << std::endl;
  

  for (uint j=1; j < ref_j; j++) {
    
    //std::cerr << "j: " << j << std::endl;

    assert(next_set_idx_ <= uncovered_set_.yDim());
    
    for (uint k=level; k < uncovered_set_.xDim(); k++)
      uncovered_set_(k,next_set_idx_) = uncovered_set_(k,ref_set_idx);

    uncovered_set_(level-1,next_set_idx_) = j;
    
    cover(level-1);
  }
}

void FertilityModelTrainer::compute_uncovered_sets(uint nMaxSkips) {

  uint nSets = choose(maxJ_,nMaxSkips);
  for (int k= nMaxSkips-1; k >= 0; k--)
    nSets += choose(maxJ_,k);
  std::cerr << nSets << " sets of uncovered positions" << std::endl;

  uncovered_set_.resize_dirty(nMaxSkips,nSets);
  uncovered_set_.set_constant(MAX_USHORT);
  nUncoveredPositions_.resize_dirty(nSets);
  first_set_.resize_dirty(maxJ_+2);
  
  next_set_idx_ = 1; //the first set contains no uncovered positions at all

  for (uint j=1; j <= maxJ_; j++) {
    
    first_set_[j] = next_set_idx_;
    uncovered_set_(nMaxSkips-1,next_set_idx_) = j;
   
    cover(nMaxSkips-1);
  }
  first_set_[maxJ_+1] = next_set_idx_;

  assert(nSets == next_set_idx_);

  std::cerr << next_set_idx_ << " states." << std::endl;

  predecessor_sets_.resize(nSets);
  j_before_end_skips_.resize(nSets);

  for (uint state=0; state < nSets; state++) {
    nUncoveredPositions_[state] = nUncoveredPositions(state);

    uint j_before_end_skips = (state == 0) ? 0 : (uncovered_set_(nMaxSkips-1,state) - 1);
    for (int k=nMaxSkips-2; k >= ((int) (nMaxSkips-nUncoveredPositions_[state])); k--) {
      
      if (uncovered_set_(k,state)+1 == uncovered_set_(k+1,state)) {
        j_before_end_skips = uncovered_set_(k,state)-1;
      }
      else
        break;
    }

    j_before_end_skips_[state] = j_before_end_skips;
  }

  uint nMaxPredecessors = 0;
  
  for (uint state = 0; state < next_set_idx_; state++) {

    std::vector<std::pair<ushort,ushort> > cur_predecessor_sets;

    //     std::cerr << "processing state ";
    //     for (uint k=0; k < nMaxSkips; k++) {

    //       if (uncovered_set_(k,state) == MAX_USHORT)
    // 	std::cerr << "-";
    //       else
    // 	std::cerr << uncovered_set_(k,state);
    //       std::cerr << ",";
    //     }
    //     std::cerr << std::endl;

    //uint maxUncoveredPos = uncovered_set_(nMaxSkips-1,state);

    //NOTE: a state is always its own predecessor state; to save memory we omit the entry
    bool limit_state = (uncovered_set_(0,state) != MAX_USHORT);
    //uint prev_candidate;

    if (limit_state) {
      //       for (uint k=1; k < nMaxSkips; k++)
      // 	assert(uncovered_set_(k,state) != MAX_USHORT);

      //predecessor states can only be states with less entries

      uint nConsecutiveEndSkips = 1;
      for (int k=nMaxSkips-2; k >= 0; k--) {
	
        if (uncovered_set_(k,state) == uncovered_set_(k+1,state) - 1)
          nConsecutiveEndSkips++;
        else
          break;
      }
      const uint nPrevSkips = nMaxSkips-nConsecutiveEndSkips;
      const uint highestUncoveredPos = uncovered_set_(nMaxSkips-1,state);

      assert(nMaxSkips >= 2); //TODO: handle the cases of nMaxSkips = 1 or 0

      if (nConsecutiveEndSkips == nMaxSkips)
        cur_predecessor_sets.push_back(std::make_pair(0,highestUncoveredPos+1));
      else {

        const uint skip_before_end_skips = uncovered_set_(nMaxSkips-nConsecutiveEndSkips-1,state);
	
        for (uint prev_candidate = first_set_[skip_before_end_skips]; 
             prev_candidate < first_set_[skip_before_end_skips+1]; prev_candidate++) {

          if (nUncoveredPositions_[prev_candidate] == nPrevSkips) {

            bool is_predecessor = true;
	    
            for (uint k=0; k < nPrevSkips; k++) {
              if (uncovered_set_(k+nConsecutiveEndSkips,prev_candidate) != uncovered_set_(k,state)) {
                is_predecessor = false;
                break;
              }
            }

            if (is_predecessor) {
              cur_predecessor_sets.push_back(std::make_pair(prev_candidate,highestUncoveredPos+1));

              break;
            }
	    
          }
        }
      }

      // #if 0
      //       assert(nMaxSkips >= 2); //TODO: handle the cases of nMaxSkips = 1 or 0

      //       const uint highestUncoveredPos = uncovered_set_(nMaxSkips-1,state);
      //       const uint secondHighestUncoveredPos = uncovered_set_(nMaxSkips-2,state);

      //       bool is_predecessor;
      //       for (prev_candidate = 0; prev_candidate < first_set_[secondHighestUncoveredPos+1]; prev_candidate++) {

      // 	is_predecessor = true;
      // 	if (uncovered_set_(0,prev_candidate) != MAX_USHORT)
      // 	  is_predecessor = false;
      // 	else {
      // 	  const uint nCandidateSkips = nUncoveredPositions_[prev_candidate];
      // 	  const uint nNewSkips = nMaxSkips-nCandidateSkips;
	  
      // 	  if (nNewSkips != nConsecutiveEndSkips)
      // 	    is_predecessor = false;
      // 	  else {
      // 	    for (uint k=0; k < nCandidateSkips; k++) {
      // 	      if (uncovered_set_(k+nNewSkips,prev_candidate) != uncovered_set_(k,state)) {
      // 		is_predecessor = false;
      // 		break;
      // 	      }
      // 	    }
      // 	  }
      // 	}

      // 	if (is_predecessor) {
      // 	  cur_predecessor_sets.push_back(std::make_pair(prev_candidate,highestUncoveredPos+1));
      // 	}
      //       }
      // #endif
    }
    else {

      //predecessor entries can be states with less entries 
      // or states with more entries

      const uint nUncoveredPositions = nUncoveredPositions_[state];

      uint nConsecutiveEndSkips = (state == 0) ? 0 : 1;
      for (int k=nMaxSkips-2; k >= ((int) (nMaxSkips-nUncoveredPositions)); k--) {
	
        if (uncovered_set_(k,state) == uncovered_set_(k+1,state) - 1)
          nConsecutiveEndSkips++;
        else
          break;
      }
      const uint nPrevSkips = nUncoveredPositions -nConsecutiveEndSkips; 
      const uint highestUncoveredPos = uncovered_set_(nMaxSkips-1,state);
      
      //a) find states with less entries
      if (nUncoveredPositions == nConsecutiveEndSkips) {
        if (state != 0)
          cur_predecessor_sets.push_back(std::make_pair(0,highestUncoveredPos+1));
      }
      else {

        assert(state != 0);

        const uint skip_before_end_skips = uncovered_set_(nMaxSkips-nConsecutiveEndSkips-1,state);
	
        for (uint prev_candidate = first_set_[skip_before_end_skips]; 
             prev_candidate < first_set_[skip_before_end_skips+1]; prev_candidate++) {

          if (nUncoveredPositions_[prev_candidate] == nPrevSkips) {

            bool is_predecessor = true;
	    
            for (uint k=nMaxSkips-nUncoveredPositions; 
                 k < nMaxSkips - nUncoveredPositions + nPrevSkips; k++) {
              if (uncovered_set_(k+nConsecutiveEndSkips,prev_candidate) != uncovered_set_(k,state)) {
                is_predecessor = false;
                break;
              }
            }

            if (is_predecessor) {
              cur_predecessor_sets.push_back(std::make_pair(prev_candidate,highestUncoveredPos+1));
              break;
            }
	    
          }
        }

        // #if 0
        // 	bool match;
	
        // 	for (prev_candidate = 0; prev_candidate < first_set_[secondHighestUncoveredPos+1]; prev_candidate++) {

        // 	  if (nUncoveredPositions_[prev_candidate] == nPrevSkips) {

        // 	    //the candidate set has exactly one entry less
        // 	    //now check if the sets match when the highest position is removed from the 

        // 	    match = true;
        // 	    for (uint k=nMaxSkips-nPrevSkips; k < nMaxSkips; k++) {
        // 	      if (uncovered_set_(k-nConsecutiveEndSkips,state) != 
        // 		  uncovered_set_(k,prev_candidate)) {
        // 		match = false;
        // 		break;
        // 	      }
        // 	    }

        // 	    if (match)
        // 	      cur_predecessor_sets.push_back(std::make_pair(prev_candidate,highestUncoveredPos+1));
        // 	  }
        // 	}
        // #endif	
      }

      // #if 0
      //       //b) find states with exactly one entry more
      //       for (prev_candidate = 1; prev_candidate < next_set_idx_; prev_candidate++) {

      // 	if (nUncoveredPositions_[prev_candidate] == nUncoveredPositions+1) {

      // 	  uint nContained = 0;
      // 	  uint not_contained_pos = MAX_UINT;
      // 	  bool contained;

      // 	  uint k,l;

      // 	  for (k= nMaxSkips-nUncoveredPositions-1; k < nMaxSkips; k++) {
	    
      // 	    const uint entry = uncovered_set_(k,prev_candidate);
	    
      // 	    contained = false;
      // 	    for (l=nMaxSkips-nUncoveredPositions; l < nMaxSkips; l++) {
      // 	      if (entry == uncovered_set_(l,state)) {
      // 		contained = true;
      // 		break;
      // 	      }
      // 	    }

      // 	    if (contained) {
      // 	      nContained++;
      // 	    }
      // 	    else
      // 	      not_contained_pos = entry;
      // 	  }
	
      // 	  if (nContained == nUncoveredPositions) {
      // 	    cur_predecessor_sets.push_back(std::make_pair(prev_candidate,not_contained_pos));
      // 	  }
      // 	}
      //       }
      // #endif
    }
    
    const uint nCurPredecessors = cur_predecessor_sets.size();
    predecessor_sets_[state].resize(2,nCurPredecessors);
    uint k;
    for (k=0; k < nCurPredecessors; k++) {
      predecessor_sets_[state](0,k) = cur_predecessor_sets[k].first;
      predecessor_sets_[state](1,k) = cur_predecessor_sets[k].second;
    }

    nMaxPredecessors = std::max(nMaxPredecessors,nCurPredecessors);
  }

  for (uint state = 1; state < nSets; state++) {

    //find successors of the states
    const uint nUncoveredPositions = nUncoveredPositions_[state];
    if (nUncoveredPositions == 1) {

      uint nPrevPredecessors = predecessor_sets_[0].yDim();
      predecessor_sets_[0].resize(2,nPrevPredecessors+1);
      predecessor_sets_[0](0,nPrevPredecessors) = state;
      predecessor_sets_[0](1,nPrevPredecessors) = uncovered_set_(nMaxSkips-1,state);
    }
    else {
      Math1D::NamedVector<uint> cur_uncovered(nUncoveredPositions,MAKENAME(cur_uncovered));
      for (uint k=0; k < nUncoveredPositions; k++) 
        cur_uncovered[k] = uncovered_set_(nMaxSkips-nUncoveredPositions+k,state);
      
      for (uint erase_pos = 0; erase_pos < nUncoveredPositions; erase_pos++) {
        Math1D::NamedVector<uint> succ_uncovered(nUncoveredPositions-1,MAKENAME(succ_uncovered));

        //std::cerr << "A" << std::endl;

        uint l=0;
        for (uint k=0; k < nUncoveredPositions; k++) {
          if (k != erase_pos) {
            succ_uncovered[l] = cur_uncovered[k];
            l++;
          }
        }

        //std::cerr << "B" << std::endl;

        const uint last_uncovered_pos = succ_uncovered[nUncoveredPositions-2];

        for (uint succ_candidate = first_set_[last_uncovered_pos]; 
             succ_candidate < first_set_[last_uncovered_pos+1]; succ_candidate++) {

          if (nUncoveredPositions_[succ_candidate] == nUncoveredPositions-1) {
            bool match = true;
            for (uint l=0; l < nUncoveredPositions-1; l++) {

              if (uncovered_set_(nMaxSkips-nUncoveredPositions+1+l,succ_candidate) != succ_uncovered[l]) {
                match = false;
                break;
              }
            }

            if (match) {

              uint nPrevPredecessors = predecessor_sets_[succ_candidate].yDim();
              predecessor_sets_[succ_candidate].resize(2,nPrevPredecessors+1);
              predecessor_sets_[succ_candidate](0,nPrevPredecessors) = state;
              predecessor_sets_[succ_candidate](1,nPrevPredecessors) = cur_uncovered[erase_pos];

              break;
            }
          }
        }
      }
    }
  }


  std::cerr << "each state has at most " << nMaxPredecessors << " predecessor states" << std::endl;

  uint nTransitions = 0;
  for (uint s=0; s < nSets; s++)
    nTransitions += predecessor_sets_[s].yDim();

  std::cerr << nTransitions << " transitions" << std::endl;

  //visualize_set_graph("stategraph.dot");
}

void FertilityModelTrainer::visualize_set_graph(std::string filename) {

  std::ofstream dotstream(filename.c_str());
  
  dotstream << "digraph corpus {" << std::endl
            << "node [fontsize=\"6\",height=\".1\",width=\".1\"];" << std::endl;
  dotstream << "ratio=compress" << std::endl;
  dotstream << "page=\"8.5,11\"" << std::endl;
  
  for (uint state=0; state < uncovered_set_.yDim(); state++) {

    dotstream << "state" << state << " [shape=record,label=\"";
    for (uint k=0; k < uncovered_set_.xDim(); k++) {

      if (uncovered_set_(k,state) == MAX_USHORT)
        dotstream << "-";
      else
        dotstream << uncovered_set_(k,state);

      if (k+1 < uncovered_set_.xDim())
        dotstream << "|";
    }
    dotstream << "\"]" << std::endl;
  }

  for (uint state = 0; state < uncovered_set_.yDim(); state++) {

    for (uint k=0; k < predecessor_sets_[state].yDim(); k++) 
      dotstream << "state" << predecessor_sets_[state](0,k) << " -> state" << state << std::endl; 
  }

  dotstream << "}" << std::endl;
  dotstream.close();
}

void FertilityModelTrainer::compute_coverage_states() {

  const uint nMaxSkips = uncovered_set_.xDim();

  uint nStates = maxJ_+1; //states for set #0
  for (uint k=1; k < uncovered_set_.yDim(); k++) {

    const uint highest_uncovered_pos = uncovered_set_(nMaxSkips-1,k);
    nStates += maxJ_ - highest_uncovered_pos;
  }

  coverage_state_.resize(2,nStates);

  for (uint l=0; l <= maxJ_; l++) {
    coverage_state_(0,l) = 0;
    coverage_state_(1,l) = l;
  }
  
  const uint nUncoveredSets = uncovered_set_.yDim();
  first_state_.resize(maxJ_+2);

  Math2D::NamedMatrix<uint> cov_state_num(uncovered_set_.yDim(),maxJ_+1,MAX_UINT,MAKENAME(cov_state_num));

  uint cur_state = 0;
  for (uint j=0; j <= maxJ_; j++) {

    first_state_[j] = cur_state;
    coverage_state_(0,cur_state) = 0;
    coverage_state_(1,cur_state) = j;
    cov_state_num(0,j) = cur_state;

    cur_state++;
    
    for (uint k=1; k < nUncoveredSets; k++) {
      
      const uint highest_uncovered_pos = uncovered_set_(nMaxSkips-1,k);
      if (highest_uncovered_pos < j) {

        coverage_state_(0,cur_state) = k;
        coverage_state_(1,cur_state) = j;
        cov_state_num(k,j) = cur_state;
        cur_state++;
      }
    }
  }
  first_state_[maxJ_+1] = cur_state;

  std::cerr << nStates << " coverage states" << std::endl;
  
  assert(cur_state == nStates);

  /*** now compute predecessor states ****/
  predecessor_coverage_states_.resize(nStates);

  for (uint state_num = 0; state_num < nStates; state_num++) {

    //std::cerr << "state #" << state_num << std::endl;

    std::vector<std::pair<ushort,ushort> > cur_predecessor_states;

    const uint highest_covered_source_pos = coverage_state_(1,state_num);
    const uint uncovered_set_idx = coverage_state_(0,state_num);
    const uint highest_uncovered_source_pos = uncovered_set_(nMaxSkips-1,uncovered_set_idx);

    if (highest_uncovered_source_pos == MAX_USHORT) {
      //the set of uncovered positions is empty

      assert(uncovered_set_idx == 0);
      
      if (highest_covered_source_pos > 0) { //otherwise there are no predecessor states
	
        //a) handle transition where the uncovered set is kept
        assert(state_num > 0);
        const uint prev_state = cov_state_num(uncovered_set_idx,highest_covered_source_pos-1);

        assert(coverage_state_(1,prev_state) == highest_covered_source_pos-1);
        assert(coverage_state_(0,prev_state) == uncovered_set_idx);
        cur_predecessor_states.push_back(std::make_pair(prev_state,highest_covered_source_pos));

        //b) handle transitions where the uncovered set is changed
        const uint nPredecessorSets = predecessor_sets_[uncovered_set_idx].yDim();

        for (uint p=0; p < nPredecessorSets; p++) {

          const uint covered_source_pos = predecessor_sets_[uncovered_set_idx](1,p);
          if (covered_source_pos < highest_covered_source_pos) {
            const uint predecessor_set = predecessor_sets_[uncovered_set_idx](0,p);

            //find the index of the predecessor state
            const uint prev_idx = cov_state_num(predecessor_set,highest_covered_source_pos);

            assert(prev_idx < first_state_[highest_covered_source_pos+1]);
            assert(coverage_state_(1,prev_idx) == highest_covered_source_pos);

            cur_predecessor_states.push_back(std::make_pair(prev_idx,covered_source_pos));
          }
        }
      }
      else
        assert(state_num == 0);
    }
    else {
      assert(highest_uncovered_source_pos < highest_covered_source_pos);
      
      //a) handle transition where the uncovered set is kept
      if (highest_covered_source_pos > highest_uncovered_source_pos+1) {

        assert(state_num > 0);
        const uint prev_state = cov_state_num(uncovered_set_idx,highest_covered_source_pos-1);

        assert(coverage_state_(1,prev_state) == highest_covered_source_pos-1);
        assert(coverage_state_(0,prev_state) == uncovered_set_idx);	
        cur_predecessor_states.push_back(std::make_pair(prev_state,highest_covered_source_pos));	
      }

      //b) handle transitions where the uncovered set is changed
      const uint nPredecessorSets = predecessor_sets_[uncovered_set_idx].yDim();
      
      //       std::cerr << "examining state (";
      //       print_uncovered_set(uncovered_set_idx);
      //       std::cerr << " ; " << highest_covered_source_pos << " )" << std::endl;

      for (uint p=0; p < nPredecessorSets; p++) {
	
        const uint covered_source_pos = predecessor_sets_[uncovered_set_idx](1,p);
        if (covered_source_pos <= highest_covered_source_pos) {
          const uint predecessor_set = predecessor_sets_[uncovered_set_idx](0,p);

          // 	  std::cerr << "predecessor set ";
          // 	  print_uncovered_set(predecessor_set);
          // 	  std::cerr << std::endl;
	  
          uint prev_highest_covered = highest_covered_source_pos;
          if (covered_source_pos == highest_covered_source_pos) {
            if (nUncoveredPositions_[predecessor_set] < nUncoveredPositions_[uncovered_set_idx])
              prev_highest_covered = j_before_end_skips_[uncovered_set_idx];
            else
              //in this case there is no valid transition
              prev_highest_covered = MAX_UINT;
          }
          else if (nUncoveredPositions_[predecessor_set] < nUncoveredPositions_[uncovered_set_idx]) {
            //if a new position is skipped, the highest covered source pos. must be the one after the skip
            prev_highest_covered = MAX_UINT;
          }

          if (prev_highest_covered != MAX_UINT) {
            // 	    std::cerr << "prev_highest_covered: " << prev_highest_covered << std::endl;
	    
            //find the index of the predecessor state
            const uint prev_idx = cov_state_num(predecessor_set,prev_highest_covered);

            assert(prev_idx < first_state_[prev_highest_covered+1]);
            assert(coverage_state_(1,prev_idx) == prev_highest_covered);
	    
            cur_predecessor_states.push_back(std::make_pair(prev_idx,covered_source_pos));
          }
        }
      }
    }
    
    /*** copy cur_predecessor_states to predecessor_covered_sets_[state_num] ***/
    predecessor_coverage_states_[state_num].resize(2,cur_predecessor_states.size());
    for (uint k=0; k < cur_predecessor_states.size(); k++) {
      predecessor_coverage_states_[state_num](0,k) = cur_predecessor_states[k].first;
      predecessor_coverage_states_[state_num](1,k) = cur_predecessor_states[k].second;
    }
    
  }
}

void FertilityModelTrainer::update_alignments_unconstrained() {

  Math2D::NamedMatrix<long double> expansion_prob(MAKENAME(expansion_prob));
  Math2D::NamedMatrix<long double> swap_prob(MAKENAME(swap_prob));

  for (size_t s=0; s < source_sentence_.size(); s++) {

    const uint curI = target_sentence_[s].size();
    Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));
    
    SingleLookupTable aux_lookup;
    const SingleLookupTable& cur_lookup = get_wordlookup(source_sentence_[s],target_sentence_[s],wcooc_,
                                                         nSourceWords_,slookup_[s],aux_lookup);

    uint nIter=0;
    update_alignment_by_hillclimbing(source_sentence_[s], target_sentence_[s], cur_lookup,nIter,fertility,
				     expansion_prob,swap_prob, best_known_alignment_[s]);
  }

  if (possible_ref_alignments_.size() > 0) {
      
    std::cerr << "#### AER after alignment update: " << AER() << std::endl;
    std::cerr << "#### fmeasure after alignment update: " << f_measure() << std::endl;
    std::cerr << "#### DAE/S after alignment update: " << DAE_S() << std::endl;
  }


}

void FertilityModelTrainer::compute_postdec_alignment(const Math1D::Vector<AlignBaseType>& alignment,
						      double best_prob, const Math2D::Matrix<long double>& expansion_move_prob,
						      const Math2D::Matrix<long double>& swap_move_prob, double threshold,
						      std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment) {


  const uint J = alignment.size();
  const uint I = expansion_move_prob.yDim()-1;

  const long double expansion_prob = expansion_move_prob.sum();
  const long double swap_prob =  0.5 * swap_move_prob.sum();
  
  const long double sentence_prob = best_prob + expansion_prob +  swap_prob;
  
  /**** calculate sums ***/
  Math2D::Matrix<long double> marg(J,I+1,0.0);

  for (uint j=0; j < J; j++) {
    marg(j, alignment[j]) += best_prob;

    for (uint i=0; i <= I; i++) {
      const long double cur_prob = expansion_move_prob(j,i);

      if (cur_prob > 0.0) {
	marg(j,i) += cur_prob;
	for (uint jj=0; jj < J; jj++) {
	  if (jj != j) {
	    marg(jj,alignment[jj]) += cur_prob;
	  }
	}
      }
    }
    for (uint jj=j+1; jj < J; jj++) {

      const long double cur_prob = swap_move_prob(j,jj);

      if (cur_prob > 0.0) {
	marg(j,alignment[jj]) += cur_prob;
	marg(jj,alignment[j]) += cur_prob;
	
	for (uint jjj=0; jjj < J; jjj++)
	  if (jjj != j && jjj != jj)
	    marg(jjj,alignment[jjj]) += cur_prob;
      }
    }
  }

  /*** compute marginals and threshold ***/
  for (uint j=0; j < J; j++) {

    //DEBUG
#ifndef NDEBUG
    long double check = 0.0;
    for (uint i=0; i <= I; i++)
      check += marg(j,i);
    long double ratio = sentence_prob/check;
    if ( ! (ratio >= 0.99 && ratio <= 1.01) ) {
      
      std::cerr << "sentence_prob: " << sentence_prob << ", check: " << check << std::endl;
      std::cerr << "J: " << J << ", I: " << I << ", j: " << j << std::endl;
      std::cerr << "best prob: " << best_prob << std::endl;
      std::cerr << "expansion prob: " << expansion_move_prob << std::endl;
      std::cerr << "swap prob: " << swap_move_prob << std::endl;
      std::cerr << "marg: " << marg << std::endl;
    }
    assert( ratio >= 0.99);
    assert( ratio <= 1.01);
#endif
    //END_DEBUG

    for (uint i=1; i <= I; i++) {

      long double cur_marg = marg(j,i) / sentence_prob;

      if (cur_marg >= threshold) {
	postdec_alignment.insert(std::make_pair(j+1,i));
      }
    }
  }

}


void FertilityModelTrainer::write_alignments(const std::string filename) const {

  std::ostream* out;

#ifdef HAS_GZSTREAM
  if (string_ends_with(filename,".gz")) {
    out = new ogzstream(filename.c_str());
  }
  else {
    out = new std::ofstream(filename.c_str());
  }
#else
  out = new std::ofstream(filename.c_str());
#endif

  for (size_t s=0; s < source_sentence_.size(); s++) {

    const uint curJ = source_sentence_[s].size();

    for (uint j=0; j < curJ; j++) { 
      if (best_known_alignment_[s][j] > 0)
    	(*out) << (best_known_alignment_[s][j]-1) << " " << j << " ";
    }

    (*out) << std::endl;
  }

  delete out;
}

void FertilityModelTrainer::write_postdec_alignments(const std::string filename, double thresh) {

    std::ostream* out;

#ifdef HAS_GZSTREAM
  if (string_ends_with(filename,".gz")) {
    out = new ogzstream(filename.c_str());
  }
  else {
    out = new std::ofstream(filename.c_str());
  }
#else
  out = new std::ofstream(filename.c_str());
#endif


  for (uint s=0; s < source_sentence_.size(); s++) {
    
    Math1D::Vector<AlignBaseType> viterbi_alignment = best_known_alignment_[s];
    std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
  
    SingleLookupTable aux_lookup;

    const SingleLookupTable& cur_lookup = get_wordlookup(source_sentence_[s],target_sentence_[s],wcooc_,
                                                         nSourceWords_,slookup_[s],aux_lookup);

    compute_external_postdec_alignment(source_sentence_[s], target_sentence_[s], cur_lookup,
				       viterbi_alignment, postdec_alignment, thresh);

    for(std::set<std::pair<AlignBaseType,AlignBaseType> >::iterator it = postdec_alignment.begin(); 
	it != postdec_alignment.end(); it++) {
      
      (*out) << (it->second-1) << " " << (it->first-1) << " ";
    }
    (*out) << std::endl;
    
  }
}

void FertilityModelTrainer::update_fertility_prob(const Storage1D<Math1D::Vector<double> >& ffert_count, double min_prob) {


  for (uint i=1; i < ffert_count.size(); i++) {

    //std::cerr << "i: " << i << std::endl;

    const double sum = ffert_count[i].sum();

    if (sum > 1e-305) {
      
      if (fertility_prob_[i].size() > 0) {
	assert(sum > 0.0);     
	const double inv_sum = 1.0 / sum;
	assert(!isnan(inv_sum));
	
	for (uint f=0; f < fertility_prob_[i].size(); f++) {
	  const double real_min_prob = (f <= fertility_limit_) ? min_prob : 1e-15;
	  fertility_prob_[i][f] = std::max(real_min_prob,inv_sum * ffert_count[i][f]);
	}
      }
      else {
	//std::cerr << "WARNING: target word #" << i << " does not occur" << std::endl;
      }
    }
    else {
      //std::cerr << "WARNING: did not update fertility count because sum was " << sum << std::endl;
    }
  }
} 


bool FertilityModelTrainer::make_alignment_feasible(const Storage1D<uint>& source, const Storage1D<uint>& target,
						    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment) {

 const uint J = source.size();
 const uint I = target.size();

  Math1D::Vector<uint> fertility(I+1,0);

  for (uint j=0; j < J; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  }

  bool changed = false;

  if (2*fertility[0] > J) {
	
    std::vector<std::pair<double,AlignBaseType> > priority;
    for (uint j=0; j < J; j++) {
      
      if (alignment[j] == 0) {
	priority.push_back(std::make_pair(dict_[0][source[j]-1],j));
      }
    }
    
    vec_sort(priority);
    assert(priority.size() < 2 || priority[0].first <= priority[1].first);
    
    for (uint k=0; 2*fertility[0] > J; k++) {
      
      uint j = priority[k].second;
      
      uint best_i = 0;
      double best = -1.0;
      for (uint i=1; i <= I; i++) {
	
	if (fertility[i] >= fertility_limit_)
	  continue;
	
	double hyp = dict_[target[i-1]][lookup(j,i-1)];
	if (hyp > best) {
	  
	  best = hyp;
	  best_i = i;
	}	    
      }
	  
      if (best_i == 0) {
	std::cerr << "WARNING: the given external sentence pair cannot be explained by IBM-3/4/5 with the given fertility limits." 
		       << std::endl;

	best_i = 1;
	alignment[j] = 1;
	fertility[1]++;

	fertility_prob_[target[0]][fertility[1]] = 1e-8;
      }
      else {
	alignment[j] = best_i;
	fertility[best_i]++;
      }
      fertility[0]--;
      
      changed = true;
	  
      if (dict_[target[best_i-1]][lookup(j,best_i)] < 0.001) {
	    
	dict_[target[best_i-1]] *= 0.999;
	dict_[target[best_i-1]][lookup(j,0)] += 0.001;
      } 
    }
  }

  if (fertility_limit_ < J) {
    for (uint i=1; i <= I; i++) {
    
      if (fertility[i] > fertility_limit_) {
	
	std::vector<std::pair<double,AlignBaseType> > priority;
	for (uint j=0; j < J; j++) {
	  
	  if (alignment[j] == i) {
	    priority.push_back(std::make_pair(dict_[target[i-1]][lookup(j,i-1)],j));
	  }
	}
	
	vec_sort(priority);
	assert(priority.size() < 2 || priority[0].first <= priority[1].first);
	
	for (uint k=0; fertility[i] > fertility_limit_; k++) {
	  
	  uint j = priority[k].second;
	  
	  uint best_i = i;
	  double best = -1.0;
	  for (uint ii=1; ii <= I; ii++) {
	    
	    if (ii == i || fertility[ii] >= fertility_limit_)
	      continue;
	    
	    double hyp = dict_[target[ii-1]][lookup(j,ii-1)];
	    if (hyp > best) {
	      
	      best = hyp;
	      best_i = ii;
	    }
	  }

	  if (best_i == i) {
	    //check empty word

	    if (2*fertility[0]+2 <= J)
	      best_i = 0;
	  }
	    
	  if (best_i == i) {
	    std::cerr << "WARNING: the given external sentence pair cannot be explained by IBM-3/4/5 with the given fertility limits." 
		      << std::endl;
	    
	    fertility_prob_[target[i-1]][fertility[i]] = 1e-8;
	    break; //no use resolving the remaining words, it's not possible
	  }
	  else {
	    changed = true;

	    alignment[j] = best_i;
	    
	    fertility[i]--;
	    fertility[best_i]++;
	  }	

	  const uint dict_num = (best_i == 0) ? 0 : target[best_i-1];
	  const uint dict_idx = (best_i == 0) ? source[j]-1 : lookup(j,best_i-1);
	  
	  if (dict_[dict_num][dict_idx] < 0.001) {
	    
	    dict_[dict_num] *= 0.999;
	    dict_[dict_num][dict_idx] += 0.001;
	  } 
	}
      }
    }
  }

  return changed;
}


void FertilityModelTrainer::common_prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
							      const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment) {



 const uint J = source.size();
 const uint I = target.size();

  assert(lookup.xDim() == J && lookup.yDim() == I);

  if (alignment.size() != J)
    alignment.resize(J,1);

  Math1D::Vector<uint> fertility(I+1,0);

    for (uint j=0; j < J; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
  }

  if (fertility[0] > 0 && p_zero_ < 1e-12)
    p_zero_ = 1e-12;
  

  make_alignment_feasible(source, target, lookup, alignment);

  /*** check if fertility tables are large enough ***/
  for (uint i=0; i < I; i++) {

    if (fertility_prob_[target[i]].size() < J+1)
      fertility_prob_[target[i]].resize(J+1,1e-15);

    if (fertility_prob_[target[i]][fertility[i+1]] < 1e-15)
      fertility_prob_[target[i]][fertility[i+1]] = 1e-15;

    if (fertility_prob_[target[i]].sum() < 0.5)
      fertility_prob_[target[i]].set_constant(1.0 / fertility_prob_[target[i]].size());

    if (fertility_prob_[target[i]][fertility[i+1]] < 1e-8)
      fertility_prob_[target[i]][fertility[i+1]] = 1e-8;
  }

  /*** check if a source word does not have a translation (with non-zero prob.) ***/
  for (uint j=0; j < J; j++) {
    uint src_idx = source[j];

    double sum = dict_[0][src_idx-1];
    for (uint i=0; i < I; i++)
      sum += dict_[target[i]][lookup(j,i)];

    if (sum < 1e-100) {
      for (uint i=0; i < I; i++)
        dict_[target[i]][lookup(j,i)] = 1e-15;
    }

    uint aj = alignment[j];
    if (aj == 0) {
      if (dict_[0][src_idx-1] < 1e-20)
        dict_[0][src_idx-1] = 1e-20;
    }
    else {
      if (dict_[target[aj-1]][lookup(j,aj-1)] < 1e-20)
        dict_[target[aj-1]][lookup(j,aj-1)] = 1e-20;
    }
  }
}


long double FertilityModelTrainer::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
							      const SingleLookupTable& lookup,
							      Math1D::Vector<AlignBaseType>& alignment) {


  prepare_external_alignment(source, target, lookup, alignment);

  const uint J = source.size();
  const uint I = target.size();

  //create matrices
  Math2D::Matrix<long double> expansion_prob(J,I+1);
  Math2D::Matrix<long double> swap_prob(J,J);

  Math1D::Vector<uint> fertility(I+1,0);
  
  uint nIter;

  return update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility,
					  expansion_prob, swap_prob, alignment);
}

// <code> start_alignment </code> is used as initialization for hillclimbing and later modified
// the extracted alignment is written to <code> postdec_alignment </code>
void FertilityModelTrainer::compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
							       const SingleLookupTable& lookup,
							       Math1D::Vector<AlignBaseType>& start_alignment,
							       std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
							       double threshold) {

  prepare_external_alignment(source, target, lookup, start_alignment);

  const uint J = source.size();
  const uint I = target.size();

  //create matrices
  Math2D::Matrix<long double> expansion_prob(J,I+1);
  Math2D::Matrix<long double> swap_prob(J,J);

  Math1D::Vector<uint> fertility(I+1,0);
  
  uint nIter;

  long double best_prob = update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility,
							   expansion_prob, swap_prob, start_alignment);

  compute_postdec_alignment(start_alignment, best_prob, expansion_prob, swap_prob, threshold, postdec_alignment);
}


