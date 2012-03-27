/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#ifndef ALIGNMENT_COMPUTATION_HH
#define ALIGNMENT_COMPUTATION_HH

#include "mttypes.hh"

void compute_ibm1_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    Storage1D<uint>& viterbi_alignment);

void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    const Math2D::Matrix<double>& align_prob,
                                    Storage1D<uint>& viterbi_alignment);

void compute_fullhmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                       const Math2D::Matrix<uint>& slookup,
                                       const Storage1D<uint>& target_sentence,
                                       const SingleWordDictionary& dict,
                                       const Math2D::Matrix<double>& align_prob,
                                       Storage1D<uint>& viterbi_alignment);

double compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                      const Math2D::Matrix<uint>& slookup,
                                      const Storage1D<uint>& target_sentence,
                                      const SingleWordDictionary& dict,
                                      const Math2D::Matrix<double>& align_prob,
                                      const Math1D::Vector<double>& initial_prob,
                                      Storage1D<uint>& viterbi_alignment, 
				      bool internal_mode = false, bool verbose = false);

void compute_ehmm_optmarginal_alignment(const Storage1D<uint>& source_sentence,
                                        const Math2D::Matrix<uint>& slookup,
                                        const Storage1D<uint>& target_sentence,
                                        const SingleWordDictionary& dict,
                                        const Math2D::Matrix<double>& align_prob,
                                        const Math1D::Vector<double>& initial_prob,
                                        Storage1D<uint>& optmarginal_alignment);


#endif
