/*** written by Thomas Schoenemann as a private person, since October 2009 ***/

#ifndef TRAINING_COMMON_HH
#define TRAINING_COMMON_HH

#include "vector.hh"
#include "mttypes.hh"

#include <map>
#include <set>

const double ibm1_min_dict_entry = 1e-6;
const double ibm2_min_align_param = 1e-7;
const double hmm_min_dict_entry = 1e-6; //with and without classes, also for all bi-word HMMs
const double hmm_min_param_entry = 1e-8; //for init and align params, also for all bi-word HMMs

const double bi1_min_dict_entry = 1e-6;
const double bi2_min_dict_entry = 1e-6;
const double bi2_min_param_entry = 1e-8;
const double bihmm_min_dict_entry = 1e-6;

const double fert_min_param_entry = 1e-8;
const double fert_min_p0 = 1e-12;
const double fert_min_dict_entry = 1e-7;

//for HMM
inline void compute_dictmat(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                            const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
                            Math2D::Matrix<double>& dicttab)
{
  const uint I = target_sentence.size();
  const uint J = source_sentence.size();

  assert(dicttab.xDim() == J);
  assert(dicttab.yDim() == I+1);

  for (uint i=0; i < I; i++) {
    const Math1D::Vector<double>& cur_dict = dict[target_sentence[i]];
    const uint* lrow = slookup.row_ptr(i);
    for (uint j=0; j < J; j++)
      dicttab(j,i) = cur_dict[lrow[j]]; //cur_dict[slookup(j,i)];
  }
  const Math1D::Vector<double>& null_dict = dict[0];
  for (uint j=0; j < J; j++)
    dicttab(j,I) = null_dict[source_sentence[j]-1];
}

//for fertility format
inline void compute_dictmat_fertform(const Storage1D<uint>& source_sentence, const SingleLookupTable& slookup,
                                     const Storage1D<uint>& target_sentence, const SingleWordDictionary& dict,
                                     Math2D::Matrix<double>& dicttab)
{
  const uint I = target_sentence.size();
  const uint J = source_sentence.size();

  assert(dicttab.xDim() == J);
  assert(dicttab.yDim() == I+1);

  for (uint i=0; i < I; i++) {
    const Math1D::Vector<double>& cur_dict = dict[target_sentence[i]];
    const uint* lrow = slookup.row_ptr(i);
    for (uint j=0; j < J; j++)
      dicttab(j,i+1) = cur_dict[lrow[j]]; //cur_dict[slookup(j,i)];
  }
  const Math1D::Vector<double>& null_dict = dict[0];
  for (uint j=0; j < J; j++)
    dicttab(j,0) = null_dict[source_sentence[j]-1];
}

//returns the number of ignored entries
uint set_prior_dict_weights(const std::set<std::pair<uint,uint> >& known_pairs, const CooccuringWordsType& wcooc,
                            floatSingleWordDictionary prior_weight, float init_dict_regularity);

void find_cooccuring_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                           uint nSourceWords, uint nTargetWords, CooccuringWordsType& cooc);

void find_cooccuring_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                           const Storage1D<Math1D::Vector<uint> >& additional_source, const Storage1D<Math1D::Vector<uint> >& additional_target,
                           uint nSourceWords, uint nTargetWords, CooccuringWordsType& cooc);

bool read_cooccuring_words_structure(std::string filename, uint nSourceWords, uint nTargetWords, CooccuringWordsType& cooc);

void find_cooc_monolingual_pairs(const Storage1D<Math1D::Vector<uint> >& sentence, uint voc_size, Storage1D<Storage1D<uint> >& cooc);

void monolingual_pairs_cooc_count(const Storage1D<Math1D::Vector<uint> >& sentence, const Storage1D<Storage1D<uint> >& t_cooc,
                                  Storage1D<Storage1D<uint> >& t_cooc_count);

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    std::map<std::pair<uint,uint>,std::set<uint> >& cooc);

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    uint nSourceWords, uint nTargetWords, Storage1D<Storage1D<std::pair<uint,Storage1D<uint> > > >& cooc);

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    const Storage1D<Storage1D<uint> >& target_cooc, Storage1D<Storage1D<Storage1D<uint> > >& st_cooc);

void count_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    std::map<std::pair<uint,uint>,std::map<uint,uint> >& cooc_count);

void count_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    uint nSourceWords, uint nTargetWords,
    Storage1D<Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > > >& cooc);

void find_cooccuring_lengths(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                             CooccuringLengthsType& cooc);

void generate_wordlookup(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                         const CooccuringWordsType& cooc, uint nSourceWords, LookupTable& slookup, uint max_size = MAX_UINT);

const SingleLookupTable& get_wordlookup(const Storage1D<uint>& source, const Storage1D<uint>& target, const CooccuringWordsType& cooc,
                                        uint nSourceWords, const SingleLookupTable& lookup, SingleLookupTable& aux);

double prob_penalty(double x, double beta);

double prob_pen_prime(double x, double beta);

void update_dict_from_counts(const UnnamedSingleWordDictionary& fdict_count, const floatUnnamedSingleWordDictionary& prior_weight,
                             double dict_weight_sum, bool smoothed_l0, double l0_beta, uint nDictStepIter, UnnamedSingleWordDictionary& dict,
                             double min_prob = 0.0, bool unconstrained_m_step = false);

void dict_m_step(const SingleWordDictionary& fdict_count, const floatSingleWordDictionary& prior_weight,
                 SingleWordDictionary& dict, double alpha, uint nIter = 100, bool smoothed_l0 = false, double l0_beta = 1.0);

void single_dict_m_step(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight,
                        Math1D::Vector<double>& dict, double alpha, uint nIter, bool smoothed_l0, double l0_beta,
                        bool with_slack = true);

void single_dict_m_step_unconstrained(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight,
                                      Math1D::Vector<double>& dict, uint nIter, bool smoothed_l0, double l0_beta, uint L);

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight,
                                 const Math1D::Vector<double>& dict, bool smoothed_l0, double l0_beta);

//for IBM-4/5 (i.e. no alignments to NULL considered)
void par2nonpar_start_prob(const Math1D::Vector<double>& sentence_start_parameters,
                           Storage1D<Math1D::Vector<double> >& sentence_start_prob);

void start_prob_m_step(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                       Math1D::Vector<double>& param, uint nIter = 400);

void start_prob_m_step_unconstrained(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                                     Math1D::Vector<double>& sentence_start_parameters, uint nIter = 400, uint L = 5);


inline void symmetrize_swapmat(Math2D::Matrix<double>& swap_prob, const uint curJ)
{
  for (uint j1 = 0; j1 < curJ; j1++) {

    swap_prob(j1, j1) = 0.0;

    for (uint j2 = j1 + 1; j2 < curJ; j2++) {

      swap_prob(j2, j1) = swap_prob(j1, j2);
    }
  }
}

double dict_reg_term(const SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight, double l0_beta = -1.0);

#endif
