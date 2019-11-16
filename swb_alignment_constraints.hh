/*** Written by Thomas Schoenemann as a private person, since October 2019 ***/

#ifndef SWB_ALIGNMENT_CONSTRAINTS
#define SWB_ALIGNMENT_CONSTRAINTS

#include "mttypes.hh"
#include <set>

enum AlignmentSet {ITGAlignments, IBMSkipAlignments, AllAlignments };

struct AlignmentSetConstraints {

  AlignmentSet align_set_ = AllAlignments;
  uint nMaxSkips_ = 3;
  uint itg_extension_level_ = 0;
  uint itg_max_mid_dev_ = 5;
  uint itg_level3_maxlength_ = 7;
};

//nMaxSkips is relevant only for ext_level 3
bool alignment_satisfies_itg_nonull(const Math1D::Vector<AlignBaseType>& alignment, uint II, uint ext_level = 0, int max_mid_dev = 10000,
                                    uint nMaxSkips = 3, uint level3_maxlength = 8);

bool alignment_satisfies_ibm_nonull(const Math1D::Vector<AlignBaseType>& alignment, uint nMaxSkips);

class IBMConstraintStates {
public:

  void compute_uncovered_sets(uint maxJ, uint nMaxSkips = 3);

  void cover(uint level);

  void compute_coverage_states(uint maxJ);

  uint nUncoveredPositions(uint state) const;

  void print_uncovered_set(uint setnum) const;

  void visualize_set_graph(std::string filename);

  Math2D::NamedMatrix<ushort> uncovered_set_;

  //the first entry denotes the predecessor state, the second the source position covered in the transition
  NamedStorage1D<Math2D::Matrix<uint> > predecessor_sets_;

  //tells for each state how many uncovered positions are in that state
  Math1D::NamedVector<ushort> nUncoveredPositions_;

  //first_set_[i] marks the first row of <code> uncovered_sets_ </code> where the
  // position i appears
  Math1D::NamedVector<uint> first_set_;
  uint next_set_idx_;           // used during the computation of uncovered sets

  //each row is a coverage state, where the first index denotes the number of uncovered set and the
  //second the maximum covered source position
  Math2D::NamedMatrix<uint> coverage_state_;
  Math1D::NamedVector<uint> first_state_;
  NamedStorage1D<Math2D::Matrix<uint> > predecessor_coverage_states_;
  std::set<uint> start_states_;
};

#endif