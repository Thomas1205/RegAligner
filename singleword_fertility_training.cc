/*** written by Thomas Schoenemann. Started as a private person without employment, November 2009 ***/
/*** continued at Lund University, Sweden, January 2010 - March 2011, as a private person and ***/
/*** at the University of Düsseldorf, Germany, January - September 2012 ***/

#include "singleword_fertility_training.hh"
#include "combinatoric.hh"
#include "alignment_error_rate.hh"
#include "timing.hh"
#include "alignment_computation.hh"

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"

/************* implementation of FertilityModelTrainer *******************************/

FertilityModelTrainer::FertilityModelTrainer(const Storage1D<Storage1D<uint> >& source_sentence,
                                             const Storage1D<Math2D::Matrix<uint, ushort> >& slookup,
                                             const Storage1D<Storage1D<uint> >& target_sentence,
                                             SingleWordDictionary& dict,
                                             const CooccuringWordsType& wcooc,
                                             uint nSourceWords, uint nTargetWords,
                                             const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                                             const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
					     uint fertility_limit) :
  uncovered_set_(MAKENAME(uncovered_sets_)), predecessor_sets_(MAKENAME(predecessor_sets_)), 
  nUncoveredPositions_(MAKENAME(nUncoveredPositions_)), j_before_end_skips_(MAKENAME(j_before_end_skips_)),
  first_set_(MAKENAME(first_set_)), next_set_idx_(0), coverage_state_(MAKENAME(coverage_state_)),
  first_state_(MAKENAME(first_state_)), predecessor_coverage_states_(MAKENAME(predecessor_coverage_states_)),
  source_sentence_(source_sentence), slookup_(slookup), target_sentence_(target_sentence), 
  wcooc_(wcooc), dict_(dict), nSourceWords_(nSourceWords), nTargetWords_(nTargetWords), iter_offs_(0),
  fertility_prob_(nTargetWords,MAKENAME(fertility_prob_)), 
  best_known_alignment_(MAKENAME(best_known_alignment_)),
  sure_ref_alignments_(sure_ref_alignments), possible_ref_alignments_(possible_ref_alignments)
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

    nContributors++;
    //add DAE/S
    sum_errors += ::nDefiniteAlignmentErrors(alignment[s],sure_ref_alignments_[s+1],possible_ref_alignments_[s+1]);
  }
  
  sum_errors /= nContributors;
  return sum_errors;
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
      const double cur_prob = expansion_move_prob(j,i);

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

      const double cur_prob = swap_move_prob(j,jj);

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

