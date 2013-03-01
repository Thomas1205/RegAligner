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

class FertilityModelTrainer {
public:

  FertilityModelTrainer(const Storage1D<Storage1D<uint> >& source_sentence,
                        const LookupTable& slookup,
                        const Storage1D<Storage1D<uint> >& target_sentence,
                        SingleWordDictionary& dict,
                        const CooccuringWordsType& wcooc,
                        uint nSourceWords, uint nTargetWords,
                        const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                        const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                        uint fertility_limit = 10000);

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

protected:

  void print_uncovered_set(uint state) const;

  uint nUncoveredPositions(uint state) const;

  void compute_uncovered_sets(uint nMaxSkips = 4);

  void cover(uint level);

  void visualize_set_graph(std::string filename);

  void compute_coverage_states();

  void compute_postdec_alignment(const Math1D::Vector<AlignBaseType>& alignment,
				 double best_prob, const Math2D::Matrix<long double>& expansion_move_prob,
				 const Math2D::Matrix<long double>& swap_move_prob, double threshold,
				 std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment);


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

  NamedStorage1D<Math1D::Vector<double> > fertility_prob_;

  NamedStorage1D<Math1D::Vector<AlignBaseType> > best_known_alignment_;

  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > > sure_ref_alignments_;
  std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > > possible_ref_alignments_;
};

#endif
