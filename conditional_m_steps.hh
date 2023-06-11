/**************** written by Thomas Schoenemann, Decemeber 2022 ********/

#ifndef CONDITIONAL_M_STEPS_HH
#define CONDITIONAL_M_STEPS_HH

#include "vector.hh"
#include "mttypes.hh"
#include "training_common.hh"

#include <map>
#include <set>

/****************** Dictionary m-steps *******************/

void dict_m_step(const SingleWordDictionary& fdict_count, const floatSingleWordDictionary& prior_weight, size_t nSentences,
                 SingleWordDictionary& dict, double alpha, uint nIter = 100, bool smoothed_l0 = false, double l0_beta = 1.0, double min_prob = 1e-8);

void single_dict_m_step(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight, size_t nSentences,
                        Math1D::Vector<double>& dict, double alpha, uint nIter, bool smoothed_l0, double l0_beta, double min_prob,
                        bool with_slack = true, bool const_prior = false, bool quiet = true);

void single_dict_m_step_unconstrained(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight, size_t nSentences,
                                      Math1D::Vector<double>& dict, uint nIter, bool smoothed_l0, double l0_beta, uint L, double min_prob, bool const_prior = false);

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight, size_t nSentences,
                                 const Math1D::Vector<double>& dict, bool smoothed_l0, double l0_beta);

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, float prior_weight, size_t nSentences,
                                 const Math1D::Vector<double>& dict, bool smoothed_l0, double l0_beta);

/********************** start-prob m-steps ****************/

//for IBM-4/5 (i.e. no alignments to NULL considered)
void par2nonpar_start_prob(const Math1D::Vector<double>& sentence_start_parameters,
                           Storage1D<Math1D::Vector<double> >& sentence_start_prob);

void start_prob_m_step(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                       Math1D::Vector<double>& param, uint nIter = 400, double gd_stepsize = 1.0);

void start_prob_m_step_unconstrained(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                                     Math1D::Vector<double>& sentence_start_parameters, uint nIter = 400, uint L = 5);

void ehmm_init_m_step(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter,
                      ProjectionMode projection_mode = Simplex, double gd_stepsize = 1.0);

void ehmm_init_m_step_projected_lbfgs(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter, int L = 5);

/********************* IBM-2 m-steps **********************/

void reducedibm2_diffpar_m_step(Math3D::Tensor<double>& align_param, const ReducedIBM2ClassAlignmentModel& acount, uint offset, uint c,
                                uint nIter, bool deficient, double gd_stepsize);

void reducedibm2_par_m_step(Math3D::Tensor<double>& align_param, const ReducedIBM2ClassAlignmentModel& acount, uint j, uint c, uint nIter,
                            bool deficient, double gd_stepsize, bool quiet = false);

/********************* HMM m-steps ************************/

double ehmm_m_step_energy(const Math1D::Vector<double>& singleton_count, double grouping_count, const Math2D::Matrix<double>& span_count,
                          const Math1D::Vector<double>& dist_params, uint zero_offset, double grouping_param, int redpar_limit);

void ehmm_m_step(const FullHMMAlignmentModelNoClasses& facount, Math1D::Vector<double>& dist_params, uint zero_offset,
                 uint nIter, double& grouping_param, bool deficient, int redpar_limit, double gd_stepsize, bool quiet = false, ProjectionMode projection_mode = Simplex);

void ehmm_m_step(const Math1D::Vector<double>& singleton_count, const double grouping_count, const Math2D::Matrix<double>& span_count,
                 Math1D::Vector<double>& dist_params, uint zero_offset, double& grouping_param, bool deficient, int redpar_limit,
                 uint nIter, double gd_stepsize, bool quiet = false, ProjectionMode projection_mode = Simplex);

void ehmm_m_step_unconstrained(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params,
                               uint zero_offset, uint nIter, double& grouping_param, bool deficient, int redpar_limit);

void ehmm_m_step_unconstrained_LBFGS(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset, uint nIter,
                                     double& grouping_param, uint L, bool deficient, int redpar_limit);

#endif

