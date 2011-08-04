/*** written by Thomas Schoenemann as a private person without employment, October 2009 
 *** and later by Thomas Schoenemann as employee of Lund University, 2010 ***/


#ifndef IBM2_TRAINING_HH
#define IBM2_TRAINING_HH


#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

void train_ibm2(const Storage1D<Storage1D<uint> >& source, 
		const Storage1D<Math2D::Matrix<uint> >& slookup,
		const Storage1D<Storage1D<uint> >& target,
		const CooccuringWordsType& wcooc,
		const CooccuringLengthsType& lcooc,
		uint nSourceWords, uint nTargetWords,
		IBM2AlignmentModel& alignment_model,
		SingleWordDictionary& dict,
		uint nIterations,
		std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
		std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments);


void train_reduced_ibm2(const Storage1D<Storage1D<uint> >& source,
			const Storage1D<Math2D::Matrix<uint> >& slookup,
			const Storage1D<Storage1D<uint> >& target,
			const CooccuringWordsType& wcooc,
			const CooccuringLengthsType& lcooc,
			uint nSourceWords, uint nTargetWords,
			ReducedIBM2AlignmentModel& alignment_model,
			SingleWordDictionary& dict,
			uint nIterations,
			std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
			std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments);


void ibm2_viterbi_training(const Storage1D<Storage1D<uint> >& source, 
			   const Storage1D<Math2D::Matrix<uint> >& slookup,
			   const Storage1D<Storage1D<uint> >& target,
			   const CooccuringWordsType& wcooc,
			   const CooccuringLengthsType& lcooc,
			   uint nSourceWords, uint nTargetWords,
			   ReducedIBM2AlignmentModel& alignment_model,
			   SingleWordDictionary& dict,
			   uint nIterations,
			   std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
			   std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
			   const floatSingleWordDictionary& prior_weight);

#endif
