/*** written by Thomas Schoenemann as a private person without employment, November 2009 ***/

#ifndef ALIGNMENT_ERROR_RATE_HH
#define ALIGNMENT_ERROR_RATE_HH

#include "vector.hh"
#include <map>
#include <set>

//if <code> invert </code> is set, the roles of source and target are swapped
void read_reference_alignment(std::string filename, 
                              std::map<uint, std::set<std::pair<ushort,ushort> > >& sure_alignments,
                              std::map<uint, std::set<std::pair<ushort,ushort> > >& possible_alignments,
                              bool invert = false);

void write_reference_alignment(std::string filename, 
                               std::map<uint, std::set<std::pair<ushort,ushort> > >& sure_alignments,
                               std::map<uint, std::set<std::pair<ushort,ushort> > >& possible_alignments,
                               bool invert = false);


//alignment error rate as in [Och and Ney 2003] for a single sentence pair
// assuming a single-word alignment
double AER(const Storage1D<ushort>& singleword_alignment, 
           const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
           const std::set<std::pair<ushort,ushort> >& possible_ref_alignments);

//alignment error rate as in [Och and Ney 2003] for a single sentence pair
// assuming a general alignment
double AER(const std::set<std::pair<ushort,ushort> >& mt_alignment, 
           const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
           const std::set<std::pair<ushort,ushort> >& possible_ref_alignments);


//f-measure as in [Fraser and Marcu 07] for a single sentence pair
// assuming a single-word alignment
double f_measure(const Storage1D<ushort>& singleword_alignment, 
                 const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                 const std::set<std::pair<ushort,ushort> >& possible_ref_alignments,
                 double alpha = 0.1);

//f-measure as in [Fraser and Marcu 2007] for a single sentence pair
// assuming a general alignment
double f_measure(const std::set<std::pair<ushort,ushort> >& mt_alignment, 
                 const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                 const std::set<std::pair<ushort,ushort> >& possible_ref_alignments,
                 double alpha = 0.1);


// number of definite alignment errors (missing sure, present but impossible links) for a single sentence pair
// assuming a single-word alignment
uint nDefiniteAlignmentErrors(const Storage1D<ushort>& singleword_alignment, 
                              const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                              const std::set<std::pair<ushort,ushort> >& possible_ref_alignments);

// number of definite alignment errors (missing sure, present but impossible links) for a single sentence pair
// assuming a general alignment
uint nDefiniteAlignmentErrors(const std::set<std::pair<ushort,ushort> >& mt_alignment,
                              const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                              const std::set<std::pair<ushort,ushort> >& possible_ref_alignments);


#endif
