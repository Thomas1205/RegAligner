/*** written by Thomas Schoenemann as a private person without employment, November 2009 ***/

#ifndef HMM_FORWARD_BACKWARD
#define HMM_FORWARD_BACKWARD

#include "mttypes.hh"
#include "matrix.hh"

void calculate_hmm_forward(const Storage1D<uint>& source_sentence,
			   const Storage1D<uint>& target_sentence,
			   const Math2D::Matrix<uint>& slookup,
			   const SingleWordDictionary& dict,
			   const Math2D::Matrix<double>& align_model,
			   const Math1D::Vector<double>& start_prob,
			   Math2D::Matrix<long double>& forward);


void calculate_hmm_backward(const Storage1D<uint>& source_sentence,
			    const Storage1D<uint>& target_sentence,
			    const Math2D::Matrix<uint>& slookup,
			    const SingleWordDictionary& dict,
			    const Math2D::Matrix<double>& align_model,
			    const Math1D::Vector<double>& start_prob,
			    Math2D::Matrix<long double>& backward,
			    bool include_start_alignment = true);


#endif
