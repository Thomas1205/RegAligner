/*** written by Thomas Schoenemann. Started as a private person without employment, November 2009 ***/
/*** continued at Lund University, Sweden, January 2010 - March 2011, as a private person and ***/
/*** at the University of Düsseldorf, Germany, January - April 2012 ***/

#ifndef SINGLEWORD_FERTILITY_TRAINING_HH
#define SINGLEWORD_FERTILITY_TRAINING_HH

#include "mttypes.hh"
#include "vector.hh"
#include "tensor.hh"

#include "hmm_training.hh"

#include <map>
#include <set>

/*abstract*/ class FertilityModelTrainer {
public:

  FertilityModelTrainer(const Storage1D<Storage1D<uint> >& source_sentence,
                        const LookupTable& slookup,
                        const Storage1D<Storage1D<uint> >& target_sentence,
                        SingleWordDictionary& dict,
                        const CooccuringWordsType& wcooc,
                        uint nSourceWords, uint nTargetWords,
			const floatSingleWordDictionary& prior_weight,
			bool och_ney_empty_word, bool smoothed_l0_,
			double l0_beta,	double l0_fertpen,
                        const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                        const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
			const Math1D::Vector<double>& log_table,
                        uint fertility_limit = 10000);


  virtual std::string model_name() const = 0;

  double p_zero() const;

  void fix_p0(double p0);

  void write_alignments(const std::string filename) const;

  double AER();

  double AER(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments);

  double f_measure(double alpha = 0.1);

  double f_measure(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments, double alpha = 0.1);

  double DAE_S();

  double DAE_S(const Storage1D<Math1D::Vector<AlignBaseType> >& alignments);

  const NamedStorage1D<Math1D::Vector<double> >& fertility_prob() const;

  const NamedStorage1D<Math1D::Vector<AlignBaseType> >& best_alignments() const;

  void set_fertility_limit(uint new_limit);

  void write_fertilities(std::string filename);

  //improves the passed alignment using hill climbing and
  // returns the probability of the resulting alignment
  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
						       const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
						       Math2D::Matrix<long double>& expansion_prob,
						       Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment)
  = 0;


  void update_alignments_unconstrained();

  void release_memory();

protected:

  void print_uncovered_set(uint state) const;

  uint nUncoveredPositions(uint state) const;

  void compute_uncovered_sets(uint nMaxSkips = 4);

  void cover(uint level);

  void visualize_set_graph(std::string filename);

  void compute_coverage_states();

  //no actual hillclimbing done here, we directly compute a Viterbi alignment and its neighbors
  long double simulate_hmm_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
					const SingleLookupTable& lookup, const HmmWrapper& hmm_wrapper,
					Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob,
					Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment);

  void compute_postdec_alignment(const Math1D::Vector<AlignBaseType>& alignment,
				 double best_prob, const Math2D::Matrix<long double>& expansion_move_prob,
				 const Math2D::Matrix<long double>& swap_move_prob, double threshold,
				 std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment);

  void update_fertility_prob(const Storage1D<Math1D::Vector<double> >& ffert_count, double min_prob = 1e-8);

  inline void update_fertility_counts(const Storage1D<uint>& target,
				      const Math1D::Vector<AlignBaseType>& best_alignment,
				      const Math1D::NamedVector<uint>& fertility,
				      const Math2D::NamedMatrix<long double>& expansion_move_prob,
				      const long double sentence_prob, const long double inv_sentence_prob,
				      Storage1D<Math1D::Vector<double> >& ffert_count);

  inline void update_dict_counts(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
				 const SingleLookupTable& cur_lookup, 
				 const Math1D::Vector<AlignBaseType>& best_alignment,
				 const Math2D::NamedMatrix<long double>& expansion_move_prob,
				 const Math2D::NamedMatrix<long double>& swap_move_prob,
				 const long double sentence_prob, const long double inv_sentence_prob,
				 Storage1D<Math1D::Vector<double> >& fwcount);
    
  void common_prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
					 const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment);



  Math2D::NamedMatrix<ushort> uncovered_set_;
  
  //the first entry denotes the predecessor state, the second the source position covered in the transition
  NamedStorage1D<Math2D::Matrix<uint> > predecessor_sets_;

  //tells for each state how many uncovered positions are in that state
  Math1D::NamedVector<ushort> nUncoveredPositions_; 
  Math1D::NamedVector<ushort> j_before_end_skips_;

  //first_set_[i] marks the first row of <code> uncovered_sets_ </code> where the
  // position i appears
  Math1D::NamedVector<uint> first_set_;
  uint next_set_idx_; // used during the computation of uncovered sets

  //each row is a coverage state, where the first index denotes the number of uncovered set and the
  //second the maximum covered source position
  Math2D::NamedMatrix<uint> coverage_state_;
  Math1D::NamedVector<uint> first_state_;
  NamedStorage1D<Math2D::Matrix<uint> > predecessor_coverage_states_;
  

  const Storage1D<Storage1D<uint> >& source_sentence_;
  const LookupTable& slookup_;
  const Storage1D<Storage1D<uint> >& target_sentence_;

  const CooccuringWordsType& wcooc_;
  SingleWordDictionary& dict_;

  uint nSourceWords_;
  uint nTargetWords_;

  uint maxJ_;
  uint maxI_;

  uint fertility_limit_;

  double p_zero_;
  double p_nonzero_;

  bool fix_p0_;

  uint iter_offs_;

  bool och_ney_empty_word_;
  bool smoothed_l0_;
  double l0_beta_;
  double l0_fertpen_;

  const floatSingleWordDictionary& prior_weight_;

  NamedStorage1D<Math1D::Vector<double> > fertility_prob_;

  NamedStorage1D<Math1D::Vector<AlignBaseType> > best_known_alignment_;

  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > > sure_ref_alignments_;
  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > > possible_ref_alignments_;

  Math1D::Vector<long double> ld_fac_; //precomputation of factorials

  const Math1D::Vector<double>& log_table_;
};


/*************** definition of inline functions ************/

inline void FertilityModelTrainer::update_fertility_counts(const Storage1D<uint>& target,
							   const Math1D::Vector<AlignBaseType>& best_alignment,
							   const Math1D::NamedVector<uint>& fertility,
							   const Math2D::NamedMatrix<long double>& expansion_move_prob,
							   const long double sentence_prob, const long double inv_sentence_prob,
							   Storage1D<Math1D::Vector<double> >& ffert_count) {


  uint curJ = best_alignment.size();
  uint curI = target.size();

  assert(fertility.size() == curI+1);

  for (uint i=1; i <= curI; i++) {

    const uint t_idx = target[i-1];

    Math1D::Vector<double>& cur_fert_count = ffert_count[t_idx];
    
    const uint cur_fert = fertility[i];
	  
    long double addon = sentence_prob;
    for (uint j=0; j < curJ; j++) {
      if (best_alignment[j] == i) {
	for (uint ii=0; ii <= curI; ii++)
	  addon -= expansion_move_prob(j,ii);
      }
      else
	addon -= expansion_move_prob(j,i);
    }
    addon *= inv_sentence_prob;
      
    double daddon = (double) addon;
    if (!(daddon > 0.0)) {
      std::cerr << "STRANGE: fractional weight " << daddon << " for sentence pair with "
		<< curJ << " source words and " << curI << " target words" << std::endl;
      std::cerr << "sentence prob: " << sentence_prob << std::endl;
      std::cerr << "" << std::endl;
    }
      
    cur_fert_count[cur_fert] += addon;

    //NOTE: swap moves do not change the fertilities
    if (cur_fert > 0) {
      long double alt_addon = 0.0;
      for (uint j=0; j < curJ; j++) {
	if (best_alignment[j] == i) {
	  for (uint ii=0; ii <= curI; ii++) {
	    if (ii != i)
	      alt_addon += expansion_move_prob(j,ii);
	  }
	}
      }
      
      cur_fert_count[cur_fert-1] += inv_sentence_prob * alt_addon;
    }
    
    if (cur_fert+1 < fertility_prob_[t_idx].size()) {
      
      long double alt_addon = 0.0;
      for (uint j=0; j < curJ; j++) {
	if (best_alignment[j] != i) {
	  alt_addon += expansion_move_prob(j,i);
	}
      }
	
      cur_fert_count[cur_fert+1] += inv_sentence_prob * alt_addon;
    }
  }

}



inline void FertilityModelTrainer::update_dict_counts(const Storage1D<uint>& cur_source, const Storage1D<uint>& cur_target,
						      const SingleLookupTable& cur_lookup, 
						      const Math1D::Vector<AlignBaseType>& best_alignment,
						      const Math2D::NamedMatrix<long double>& expansion_move_prob,
						      const Math2D::NamedMatrix<long double>& swap_move_prob,
						      const long double sentence_prob, const long double inv_sentence_prob,
						      Storage1D<Math1D::Vector<double> >& fwcount) {

  const uint curJ = cur_source.size();
  const uint curI = cur_target.size();

  for (uint j=0; j < curJ; j++) {

    const uint s_idx = cur_source[j];
    const uint cur_aj = best_alignment[j];
    
    long double addon = sentence_prob;
    for (uint i=0; i <= curI; i++) 
      addon -= expansion_move_prob(j,i);
    for (uint jj=0; jj < curJ; jj++)
      addon -= swap_move_prob(j,jj);
    
    addon *= inv_sentence_prob;
    if (cur_aj != 0) {
      fwcount[cur_target[cur_aj-1]][cur_lookup(j,cur_aj-1)] += addon;
    }
    else {
      fwcount[0][s_idx-1] += addon;
    }
    
    for (uint i=0; i <= curI; i++) {
      
      if (i != cur_aj) {

	const uint t_idx = (i == 0) ? 0 : cur_target[i-1];
	
	long double addon = expansion_move_prob(j,i);
	for (uint jj=0; jj < curJ; jj++) {
	  if (best_alignment[jj] == i)
	    addon += swap_move_prob(j,jj);
	}
	addon *= inv_sentence_prob;
	
	if (i!=0) {
	  fwcount[t_idx][cur_lookup(j,i-1)] += addon;
	}
	else {
	  fwcount[0][s_idx-1] += addon;
	}
      }
    }
  }

}



#endif
