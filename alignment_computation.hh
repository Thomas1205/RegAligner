/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#ifndef ALIGNMENT_COMPUTATION_HH
#define ALIGNMENT_COMPUTATION_HH

#include "mttypes.hh"
#include <set>

void compute_ibm1_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    Storage1D<AlignBaseType>& viterbi_alignment);

//posterior decoding for IBM-1
void compute_ibm1_postdec_alignment(const Storage1D<uint>& source_sentence,
				    const Math2D::Matrix<uint>& slookup,
				    const Storage1D<uint>& target_sentence,
				    const SingleWordDictionary& dict,
				    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
				    double threshold = 0.25);


void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    const Math2D::Matrix<double>& align_prob,
                                    Storage1D<AlignBaseType>& viterbi_alignment);

void compute_ibm2_postdec_alignment(const Storage1D<uint>& source_sentence,
                                    const Math2D::Matrix<uint>& slookup,
                                    const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict,
                                    const Math2D::Matrix<double>& align_prob,
				    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
				    double threshold = 0.25);


void compute_fullhmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                       const Math2D::Matrix<uint>& slookup,
                                       const Storage1D<uint>& target_sentence,
                                       const SingleWordDictionary& dict,
                                       const Math2D::Matrix<double>& align_prob,
                                       Storage1D<AlignBaseType>& viterbi_alignment);

long double compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
					   const Math2D::Matrix<uint>& slookup,
					   const Storage1D<uint>& target_sentence,
					   const SingleWordDictionary& dict,
					   const Math2D::Matrix<double>& align_prob,
					   const Math1D::Vector<double>& initial_prob,
					   Storage1D<AlignBaseType>& viterbi_alignment, 
					   bool internal_mode = false, bool verbose = false,
                                           double min_dict_entry = 1e-15);

long double compute_ehmm_viterbi_alignment_with_tricks(const Storage1D<uint>& source_sentence,
						       const Math2D::Matrix<uint>& slookup,
						       const Storage1D<uint>& target_sentence,
						       const SingleWordDictionary& dict,
						       const Math2D::Matrix<double>& align_prob,
						       const Math1D::Vector<double>& initial_prob,
						       Storage1D<AlignBaseType>& viterbi_alignment, 
						       bool internal_mode = false, bool verbose = false,
                                                       double min_dict_entry = 1e-15);


long double compute_sehmm_viterbi_alignment(const Storage1D<uint>& source_sentence,
                                            const Math2D::Matrix<uint>& slookup,
                                            const Storage1D<uint>& target_sentence,
                                            const SingleWordDictionary& dict,
                                            const Math2D::Matrix<double>& align_prob,
                                            const Math1D::Vector<double>& initial_prob,
                                            Storage1D<AlignBaseType>& viterbi_alignment, 
                                            bool internal_mode = false, bool verbose = false,
                                            double min_dict_entry = 1e-15);


void compute_ehmm_optmarginal_alignment(const Storage1D<uint>& source_sentence,
                                        const Math2D::Matrix<uint>& slookup,
                                        const Storage1D<uint>& target_sentence,
                                        const SingleWordDictionary& dict,
                                        const Math2D::Matrix<double>& align_prob,
                                        const Math1D::Vector<double>& initial_prob,
                                        Storage1D<AlignBaseType>& optmarginal_alignment);

void compute_ehmm_postdec_alignment(const Storage1D<uint>& source_sentence,
				    const Math2D::Matrix<uint>& slookup,
				    const Storage1D<uint>& target_sentence,
				    const SingleWordDictionary& dict,
				    const Math2D::Matrix<double>& align_prob,
				    const Math1D::Vector<double>& initial_prob,
				    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
				    double threshold = 0.25);

#endif
