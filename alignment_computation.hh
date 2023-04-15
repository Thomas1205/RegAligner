/*** written by Thomas Schoenemann as a private person, October 2009
**** extensions at the University of Düsseldorf, 2012 and later as a private person ***/

#ifndef ALIGNMENT_COMPUTATION_HH
#define ALIGNMENT_COMPUTATION_HH

#include "mttypes.hh"
#include <set>
#include "hmm_training.hh"

/************ IBM-1 *************/

void compute_ibm1_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, Storage1D<AlignBaseType>& viterbi_alignment);

//posterior decoding for IBM-1
void compute_ibm1_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                    double threshold = 0.25);

void compute_ibm1p0_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                      const SingleWordDictionary& dict, double p0, Storage1D<AlignBaseType>& viterbi_alignment);

//posterior decoding for IBM-1-p0
void compute_ibm1p0_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                      const SingleWordDictionary& dict, double p0,
                                      std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment, double threshold = 0.25);

/************ IBM-2 *************/

void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
                                    Storage1D<AlignBaseType>& viterbi_alignment);

void compute_ibm2_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
                                    const Storage1D<WordClassType>& sclass, Storage1D<AlignBaseType>& viterbi_alignment);

void compute_ibm2_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
                                    std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                    double threshold = 0.25);

void compute_ibm2_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup, const Storage1D<uint>& target_sentence,
                                    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
                                    const Storage1D<WordClassType>& sclass, std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                    double threshold = 0.25);


/************ HMM Viterbi *************/


//NOTE: class-based HMMs do not need a target class dimension, it is implied by i_prev
long double compute_ehmm_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    Storage1D<AlignBaseType>& viterbi_alignment, const HmmOptions& hmm_options, bool internal_mode = false,
    bool verbose = false, double min_dict_entry = 1e-15);

//NOTE: class-based HMMs do not need a target class dimension, it is implied by i_prev
long double compute_ehmmc_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& tclass,
    const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    const HmmOptions& hmm_options, bool internal_mode = false, bool verbose = false,
    double min_dict_entry = 1e-15);

//NOTE: class-based HMMs do not need a target class dimension, it is implied by i_prev
long double compute_ehmmcc_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    const HmmOptions& hmm_options, bool internal_mode = false, bool verbose = false,
    double min_dict_entry = 1e-15);

long double compute_ehmm_viterbi_alignment(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode = false, bool verbose = false, double min_dict_entry = 1e-15);

//NOTE: class-based HMMs do not need a target class dimension, it is implied by i_prev
long double compute_ehmmc_viterbi_alignment(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode = false, bool verbose = false, double min_dict_entry = 1e-15);

long double compute_ehmm_viterbi_alignment_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode = false, bool verbose = false, double min_dict_entry =  1e-15,
    int redpar_limit = 5);

long double compute_sehmm_viterbi_alignment(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode = false, bool verbose = false, double min_dict_entry = 1e-15);

//NOTE: class-based HMMs do not need a target class dimension, it is implied by i_prev
long double compute_sehmmc_viterbi_alignment(const Storage1D<uint>& tclass, const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode = false, bool verbose = false, double min_dict_entry = 1e-15);

long double compute_sehmm_viterbi_alignment_with_tricks(const Math2D::Matrix<double>& dict, const Math2D::Matrix<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    bool internal_mode = false, bool verbose = false, double min_dict_entry = 1e-15,
    int redpar_limit = 5);

//NOTE: class-based HMMs do not need a target class dimension, it is implied by i_prev
long double compute_sehmmcc_viterbi_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, Storage1D<AlignBaseType>& viterbi_alignment,
    const HmmOptions& hmm_options, bool internal_mode = false, bool verbose = false,
    double min_dict_entry = 1e-15);

/************ HMM OptMarg *************/


void compute_ehmm_optmarginal_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                        const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
                                        const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
                                        const HmmOptions& options, Storage1D<AlignBaseType>& optmarginal_alignment);

void compute_ehmmc_optmarginal_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& tclass, const SingleWordDictionary& dict,
    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
    const HmmOptions& options, Storage1D<AlignBaseType>& optmarginal_alignment);

void compute_ehmmcc_optmarginal_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
    const Storage1D<uint>& target_sentence, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
    const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
    const Math1D::Vector<double>& initial_prob, const HmmOptions& options,
    Storage1D<AlignBaseType>& optmarginal_alignment);

/************ HMM Posterior Decoding *************/


void compute_ehmm_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                    const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
                                    const Math2D::Matrix<double>& align_prob, const Math1D::Vector<double>& initial_prob,
                                    const HmmOptions& options, std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                    double threshold = 0.25);

void compute_ehmmc_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                     const Storage1D<uint>& target_sentence, const Storage1D<uint>& tclass,
                                     const SingleWordDictionary& dict, const Math2D::Matrix<double>& align_prob,
                                     const Math1D::Vector<double>& initial_prob, const HmmOptions& options,
                                     std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                     double threshold = 0.25);

void compute_ehmmcc_postdec_alignment(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                      const Storage1D<uint>& target_sentence, const Storage1D<uint>& sclass, const Storage1D<uint>& tclass,
                                      const SingleWordDictionary& dict, const Math3D::Tensor<double>& align_prob,
                                      const Math1D::Vector<double>& initial_prob, const HmmOptions& options,
                                      std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
                                      double threshold = 0.25);

#endif
