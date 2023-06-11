/**************** written by Thomas Schoenemann, Decemeber 2022 ********/

//m-steps for conditional word alignment models

#include "conditional_m_steps.hh"
#include "projection.hh"
#include "storage_util.hh"
#include "trimatrix.hh"

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight, size_t nSentences,
                                 const Math1D::Vector<double>& dict, bool smoothed_l0, double l0_beta)
{
  double energy = 0.0;

  const uint dict_size = dict.size();
  assert(dict_size == prior_weight.size());
  assert(dict.min() >= 0.0);

  for (uint k = 0; k < dict_size; k++) {

    const double cur_dict_entry = std::max(dict[k], 1e-300);
    energy -= fdict_count[k] * std::log(cur_dict_entry);
  }

  energy /= nSentences;

  //std::cerr << "smoothed_l0: " << smoothed_l0 << ", l0_beta: " << l0_beta << std::endl;
  //std::cerr << "energy without reg: " << energy << std::endl;

  if (!smoothed_l0) {

    const attr_restrict float_A16* pdata = prior_weight.direct_access();
    const attr_restrict double_A16* ddata = dict.direct_access();
    energy += std::inner_product(pdata, pdata+dict_size, ddata, 0.0);

    // for (uint k = 0; k < dict_size; k++) {
    // energy += prior_weight[k] * dict[k];
    // std::cerr << "adding " << (prior_weight[k] * dict[k]) << std::endl;
    // }
  }
  else {
    for (uint k = 0; k < dict_size; k++)
      energy += prior_weight[k] * prob_penalty(dict[k], l0_beta);
  }

  //std::cerr << "final energy: " << energy << ", prior_weight.min(): " << prior_weight.min() << std::endl;

  return energy;
}

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, float prior_weight, size_t nSentences,
                                 const Math1D::Vector<double>& dict, bool smoothed_l0, double l0_beta)
{
  double energy = 0.0;

  const uint dict_size = dict.size();
  assert(dict.min() >= 0.0);

  for (uint k = 0; k < dict_size; k++) {

    const double cur_dict_entry = std::max(dict[k], 1e-300);
    energy -= fdict_count[k] * std::log(cur_dict_entry);
  }

  energy /= nSentences;

  if (!smoothed_l0) {

    energy += prior_weight * dict.sum();
  }
  else {
    double sum = 0.0;
    for (uint k = 0; k < dict_size; k++)
      sum += prob_penalty(dict[k], l0_beta);
    energy += prior_weight * sum;
  }

  return energy;
}

void single_dict_m_step(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight, size_t nSentences, Math1D::Vector<double>& dict,
                        double alpha, uint nIter, bool smoothed_l0, double l0_beta, double min_prob, bool with_slack, bool const_prior, bool quiet)
{
  if (!quiet)
    std::cerr << "single_dict_m_step, with slack: " << with_slack << std::endl;
  assert(fdict_count.min() >= 0.0);
  assert(dict.min() >= 0.0);
  assert(dict.max() <= 1.0);
  assert(min_prob > 0.0);
  assert(min_prob < 0.01);

  const uint dict_size = prior_weight.size();

  //std::cerr << "dict size: " << dict_size << std::endl;
  //std::cerr << "smoothed l0: " << smoothed_l0 << std::endl;

  if (prior_weight.max_abs() == 0.0) {

    const double sum = fdict_count.sum();

    if (sum > 1e-305) {
      for (uint k = 0; k < dict_size; k++) {
        dict[k] = std::max(min_prob, fdict_count[k] / sum);
      }
    }

    return;
  }

  double energy = (const_prior) ? single_dict_m_step_energy(fdict_count, prior_weight[0], nSentences, dict, smoothed_l0, l0_beta)
                  : single_dict_m_step_energy(fdict_count, prior_weight, nSentences, dict, smoothed_l0, l0_beta);

  if (!quiet)
    std::cerr << "initial energy: " << energy << std::endl;

  if (prior_weight.min() >= 0.0)
    assert(energy >= 0.0);

  Math1D::Vector<double> dict_grad(dict_size);
  Math1D::Vector<double> hyp_dict(dict_size);
  Math1D::Vector<double> new_dict(dict_size);

  double slack_entry = std::max(0.0, 1.0 - dict.sum());
  double new_slack_entry = slack_entry;

  if (!with_slack)
    assert(dict.sum() >= 0.99);

  //test if normalized counts give a better starting point
  {
    double fac = dict.sum() / fdict_count.sum();
    for (uint k = 0; k < dict_size; k++)
      hyp_dict[k] = std::max(min_prob, fac * fdict_count[k]);

    //std::cerr << "fac: " << fac << ", hyp dict: " << hyp_dict << std::endl;

    double hyp_energy = (const_prior) ? single_dict_m_step_energy(fdict_count, prior_weight[0], nSentences, hyp_dict, smoothed_l0, l0_beta)
                        : single_dict_m_step_energy(fdict_count, prior_weight, nSentences, hyp_dict, smoothed_l0, l0_beta);

    if (!quiet)
      std::cerr << "energy of normalized counts: " << hyp_energy << std::endl;

    if (hyp_energy < energy) {
      if (!quiet)
        std::cerr << "switching to normalized counts" << std::endl;
      dict = hyp_dict;
      energy = hyp_energy;
    }
  }

  double line_reduction_factor = 0.1;

  for (uint iter = 1; iter <= nIter; iter++) {

    if (!quiet)
      std::cerr << " ###iteration " << iter << ", energy: " << energy << std::endl;

    //set gradient to 0 and recalculate
    for (uint k = 0; k < dict_size; k++) {
      double cur_dict_entry = std::max(min_prob, dict[k]);

      if (!smoothed_l0)
        dict_grad[k] = prior_weight[k] - fdict_count[k] / (cur_dict_entry * nSentences);
      else
        dict_grad[k] = prior_weight[k] * prob_pen_prime(cur_dict_entry, l0_beta) - fdict_count[k] / (cur_dict_entry * nSentences);

      if (isnan(dict_grad[k])) {
        std::cerr << "dict_grad[" << k << "] is nan" << std::endl;
        std::cerr << "prior_weight: " << prior_weight[k] << ", count " << fdict_count[k] << std::endl;
        std::cerr << "smoothed_l0: " << smoothed_l0 << std::endl;
        std::cerr << "cur_dict_entry: " << cur_dict_entry << std::endl;
        if (smoothed_l0)
          std::cerr << "prob_pen_prime: " << prob_pen_prime(cur_dict_entry, l0_beta) << std::endl;
      }

      assert(!isnan(dict_grad[k]));
      assert(!isinf(dict_grad[k]));
    }

    assert(!isnan(dict_grad.sum()));
    assert(!isinf(dict_grad.sum()));

    //go in neg. gradient direction
    double sqr_grad_norm = dict_grad.sqr_norm();
    if (sqr_grad_norm < 1e-5) {
      if (!quiet)
        std::cerr << "CUTOFF because squared gradient norm was " << sqr_grad_norm << std::endl;
      break;
    }

    double real_alpha = alpha / sqrt(sqr_grad_norm);

    //for (uint k = 0; k < dict_size; k++) {
    //  new_dict[k] = dict[k] - real_alpha * dict_grad[k];
    //}
    Math1D::go_in_neg_direction(new_dict, dict, dict_grad, real_alpha);

    new_slack_entry = slack_entry;

    //reproject
    if (with_slack)
      projection_on_simplex_with_slack(new_dict, new_slack_entry, min_prob);
    else {
      projection_on_simplex(new_dict, min_prob);
    }
    assert(new_dict.min() >= 0.0);
    assert(new_dict.max() <= 1.0);

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    bool decreasing = true;

    uint nTries = 0;

    while (decreasing || best_energy > energy) {

      nTries++;

      lambda *= line_reduction_factor;
      const double neg_lambda = 1.0 - lambda;

      //for (uint k = 0; k < dict_size; k++) {
      //  hyp_dict[k] = lambda * new_dict[k] + neg_lambda * dict[k];
      //}
      Math1D::assign_weighted_combination(hyp_dict, lambda, new_dict, neg_lambda, dict);
      assert(hyp_dict.min() >= 0.0);
      assert(hyp_dict.max() <= 1.0);

      const double hyp_energy = (const_prior) ? single_dict_m_step_energy(fdict_count, prior_weight[0], nSentences, hyp_dict, smoothed_l0, l0_beta)
                                : single_dict_m_step_energy(fdict_count, prior_weight, nSentences, hyp_dict, smoothed_l0, l0_beta);
      //std::cerr << "lambda = " << lambda << ", hyp_energy = " << hyp_energy << std::endl;
      if (prior_weight.min() >= 0.0)
        assert(hyp_energy >= 0.0);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        decreasing = true;

        best_lambda = lambda;
      }
      else
        decreasing = false;

      if (best_energy <= 0.95 * energy)
        break;
      if (nTries >= 4 && best_energy <= 0.99 * energy)
        break;

      if (nTries >= 18)
        break;
    }

    if (best_energy > energy) {
      break;
    }

    if (nTries > 12)
      line_reduction_factor *= 0.5;
    else if (nTries > 4)
      line_reduction_factor *= 0.8;

    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint k = 0; k < dict_size; k++)
    //  dict[k] = best_lambda * new_dict[k] + neg_best_lambda * dict[k];
    Math1D::assign_weighted_combination(dict, best_lambda, new_dict, neg_best_lambda, dict);
    assert(dict.min() >= 0.0);
    assert(dict.max() <= 1.0);

    slack_entry = best_lambda * new_slack_entry + neg_best_lambda * slack_entry;
    assert(dict.sum() + slack_entry <= 1.05);

    if (best_energy > energy - 1e-5 || best_lambda < 1e-8) {
      //std::cerr << "CUTOFF after " << iter  << " iterations" << std::endl;
      break;
    }

    energy = best_energy;
  }
}

//use L-BFGS
void single_dict_m_step_unconstrained(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight, size_t nSentences,
                                      Math1D::Vector<double>& dict, uint nIter, bool smoothed_l0, double l0_beta, uint L, double min_prob, bool const_prior)
{
  //NOTE: the energy is NOT scale invariant as it does not account for renormalization
  // => need to modify the gradient

  const uint dict_size = prior_weight.size();

  if (prior_weight.max_abs() == 0.0) {

    const double sum = fdict_count.sum();

    if (sum > 1e-305) {
      for (uint k = 0; k < dict_size; k++) {
        dict[k] = fdict_count[k] / sum;
      }
    }

    return;
  }

  double energy = (const_prior) ? single_dict_m_step_energy(fdict_count, prior_weight[0], nSentences, dict, smoothed_l0, l0_beta)
                  : single_dict_m_step_energy(fdict_count, prior_weight, nSentences, dict, smoothed_l0, l0_beta);

  Math1D::Vector<double> dict_grad(dict_size);
  Math1D::Vector<double> dict_param(dict_size);
  Math1D::Vector<double> param_grad(dict_size);
  Math1D::Vector<double> hyp_dict(dict_size);
  Math1D::Vector<double> hyp_dict_param(dict_size);
  Math1D::Vector<double> search_direction(dict_size);

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(dict_size);
    step[k].resize(dict_size);
  }

  //test if normalized counts give a better starting point
  {
    double fac = dict.sum() / fdict_count.sum();
    for (uint k = 0; k < dict_size; k++)
      hyp_dict[k] = std::max(min_prob, fac * fdict_count[k]);

    //std::cerr << "fac: " << fac << ", hyp dict: " << hyp_dict << std::endl;

    double hyp_energy = (const_prior) ? single_dict_m_step_energy(fdict_count, prior_weight[0], nSentences, hyp_dict, smoothed_l0, l0_beta)
                        : single_dict_m_step_energy(fdict_count, prior_weight, nSentences, hyp_dict, smoothed_l0, l0_beta);

    if (hyp_energy < energy) {
      dict = hyp_dict;
      energy = hyp_energy;
    }
  }

  for (uint k = 0; k < dict_size; k++)
    dict_param[k] = sqrt(dict[k]);

  double line_reduction_factor = 0.5;

  double cur_sqr_sum = 1.0;

  int start_iter = 1;          //changed whenever the curvature condition is violated

  for (int iter = 1; iter <= nIter; iter++) {

    //std::cerr << "L-BFGS iteration " << iter << ", energy: " << energy << std::endl;

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    //the energy is NOT scale invariant as it does not account for renormalization
    // => need to modify the gradient

    dict_grad.set_constant(0.0);
    param_grad.set_constant(0.0);

    double addon = 0.0;

    double sum_subst = 0.0;

    //NOTE: in contrast to the constrained routine (with orthant projection),
    //  here we generally cannot assume that the sum of the squared parameters is 1.
    //   => will need this sum in the gradient calculation

    //std::cerr << "dict sum: " << cur_sqr_sum << std::endl;

    for (uint k = 0; k < dict_size; k++) {

      const double cur_dict_entry = std::max(min_prob, dict[k]);

      //the correct equations demand to divide by cur_sqr_norm everywhere. we do this outside the loop

      //a) regularity term
      double weight = prior_weight[k];
      if (smoothed_l0)
        weight *= prob_pen_prime(cur_dict_entry, l0_beta);

      //dict_grad[k] += weight * (cur_sqr_sum - (cur_dict_entry * cur_sqr_sum) ) / (cur_sqr_sum * cur_sqr_sum);
      //dict_grad[k] += weight * (1.0 - cur_dict_entry) / (cur_sqr_sum);
      dict_grad[k] += weight * (1.0 - cur_dict_entry);  //division by cur_sqr_sum is done outside the loop

      //const double subst = weight * (cur_dict_entry * cur_sqr_sum) / (cur_sqr_sum * cur_sqr_sum);
      //const double subst = weight * (cur_dict_entry) / (cur_sqr_sum);
      const double subst = weight * cur_dict_entry;     //division by cur_sqr_sum is done outside the loop
      sum_subst += subst;

      // for (uint kk=0; kk < dict_size; kk++) {
      //   if (kk != k)
      //     dict_grad[kk] -= subst;
      // }

      //b) entropy term
      //dict_grad[k] -= fdict_count[k] / (cur_dict_entry * cur_sqr_sum);  //numerator
      //addon += fdict_count[k] / cur_sqr_sum; // denominator

      //division by cur_sqr_sum is done outside the loop
      dict_grad[k] -= fdict_count[k] / (cur_dict_entry * nSentences);  //numerator
      addon += fdict_count[k];  // denominator
    }

    for (uint k = 0; k < dict_size; k++) {
      dict_grad[k] += addon;

      double cur_dict_entry = std::max(min_prob, dict[k]);
      double own_weight = prior_weight[k] * prob_pen_prime(cur_dict_entry, l0_beta) * cur_dict_entry;
      double own_subst = sum_subst - own_weight;

      dict_grad[k] -= own_subst;
    }

    dict_grad *= 1.0 / cur_sqr_sum;

    // b) now calculate the gradient for the actual parameters

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = cur_sqr_sum;   //dict_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    double coeff_sum = 0.0;

    for (uint k = 0; k < dict_size; k++) {
      const double wp = dict_param[k];
      const double grad = dict_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      param_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
    }
    for (uint kk = 0; kk < dict_size; kk++)
      param_grad[kk] -= coeff_sum * dict_param[kk];

    // for (uint k=0; k < dict_size; k++) {
    //   param_grad[k] = 2.0 * dict_grad[k] * dict_param[k];
    // }

    double sqr_grad_norm = param_grad.sqr_norm();

    if (sqr_grad_norm < 1e-5) {
      if (true)
        std::cerr << "CUTOFF because squared gradient norm was " << sqr_grad_norm << std::endl;
      break;
    }

    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < dict_size; k++) {

        //cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
        cur_grad_diff[k] += param_grad[k];
        cur_rho += cur_grad_diff[k] * cur_step[k];
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();
      std::cerr << "cur_curv: " << cur_curv << std::endl;

      if (cur_curv < 0.0) {
        //this can happen as our function is not convex and we do not enforce part 2 of the Wolfe conditions
        // (this cannot be done by backtracking line search, see Algorithm 3.5 in [Nocedal & Wright])
        // Our solution is to simply restart L-BFGS now

        start_iter = iter;
      }

      rho[cur_l] = 1.0 / cur_rho;
    }

    search_direction = param_grad;

    if (iter > start_iter) {

      Math1D::Vector<double> alpha(L);

      const int cur_first_iter = std::max<int>(start_iter, iter - L);

      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter - 1; prev_iter >= cur_first_iter; prev_iter--) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double cur_alpha = 0.0;
        for (uint k = 0; k < dict_size; k++) {
          cur_alpha += search_direction[k] * cur_step[k];
        }
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        for (uint k = 0; k < dict_size; k++) {
          search_direction[k] -= cur_alpha * cur_grad_diff[k];
        }
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = 0.0;
        for (uint k = 0; k < dict_size; k++) {
          beta += search_direction[k] * cur_grad_diff[k];
        }
        beta *= rho[prev_l];

        const double gamma = alpha[prev_l] - beta;

        for (uint k = 0; k < dict_size; k++) {
          search_direction[k] += cur_step[k] * gamma;
        }
      }

    }
    else {
      search_direction *= 1.0 / sqrt(search_direction.sqr_norm());
    }

    Math1D::negate(search_direction);

    //NOTE: because our function is nonconvex and we did not enforce the strong Wolfe conditions,
    // the curvature condition is not generally satisfied and we can get a direction of ascent
    // in this case we switch to the negative gradient
    double check_prod = (search_direction % param_grad);

    //std::cerr << "check_prod: " << check_prod << std::endl;

    if (check_prod >= 0.0) {

      INTERNAL_ERROR << " not a search direction" << std::endl;
      exit(1);

      // //std::cerr << "switching to steepest descent for this iter" << std::endl;

      // //TODO: think about whether we should truncate the L-BGFS history in such as case

      // search_direction = param_grad;
      // negate(search_direction);

      // check_prod = -param_grad.sqr_norm();
    }

    double cutoff_offset = 0.05 * check_prod;

    // d) line search

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    bool decreasing = true;

    uint nTries = 0;

    while (decreasing || best_energy > energy) {

      nTries++;

      lambda *= line_reduction_factor;

      for (uint k = 0; k < dict_size; k++) {
        hyp_dict_param[k] = dict_param[k] + lambda * search_direction[k];
      }

      const double sum = hyp_dict.sqr_norm();
      for (uint k = 0; k < dict_size; k++) {
        hyp_dict[k] = std::max(min_prob, hyp_dict_param[k] * hyp_dict_param[k] / sum);
      }

      double hyp_energy = (const_prior) ? single_dict_m_step_energy(fdict_count, prior_weight[0], nSentences, hyp_dict, smoothed_l0, l0_beta)
                          : single_dict_m_step_energy(fdict_count, prior_weight, nSentences, hyp_dict, smoothed_l0, l0_beta);
      //std::cerr << "lambda = " << lambda << ", hyp_energy = " << hyp_energy << std::endl;

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        decreasing = true;

        best_lambda = lambda;
      }
      else
        decreasing = false;

      //TODO: think about checking Wolfe part 2. However, backtracking line search may not be enough then
      if (best_energy <= energy + lambda * cutoff_offset)       //Wolfe part 1
        break;

      // if (best_energy <= 0.95*energy)
      //   break;
      // if (nTries >= 4 && best_energy <= 0.99*energy)
      //   break;

      //if (nTries >= 18)
      if (nTries >= 250)
        break;
    }

    if (nTries > 12)
      line_reduction_factor *= 0.5;
    else if (nTries > 4)
      line_reduction_factor *= 0.8;

    // e) go to the determined point

    if (best_energy < energy) {

      //update the dict and the L-BFGS variables

      uint cur_l = (iter % L);

      Math1D::Vector<double>& cur_step = step[cur_l];
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

      for (uint k = 0; k < dict_size; k++) {
        double step = best_lambda * search_direction[k];
        cur_step[k] = step;
        dict_param[k] += step;

        //prepare for the next iteration
        cur_grad_diff[k] = -param_grad[k];
      }

      double sum = dict_param.sqr_norm();
      for (uint k = 0; k < dict_size; k++) {
        dict[k] = std::max(min_prob, dict_param[k] * dict_param[k] / sum);
      }

      cur_sqr_sum = sum;

      energy = best_energy;
    }
    else {
      std::cerr << "WARNING: failed to get descent, sqr gradient norm: " << param_grad.sqr_norm() << std::endl;
      exit(1);
      break;
    }
  }
}

/********************** IBM-2 *************************/

double reducedibm2_par_m_step_energy(const Math1D::Vector<double>& align_param, const Math1D::Vector<double>& singleton_count,
                                     const Math1D::Vector<double>& span_count)
{
  const uint xDim = align_param.size();

  double energy = 0.0;

  for (uint i = 0; i < xDim; i++)
    energy -= singleton_count[i] * std::log(align_param[i]);

  double param_sum = 0.0;
  for (uint i = 0; i < xDim; i++) {
    param_sum += align_param[i];
    energy += span_count[i] * std::log(param_sum);
  }

  return energy;
}

void reducedibm2_par_m_step(Math3D::Tensor<double>& align_param, const ReducedIBM2ClassAlignmentModel& acount, uint j, uint c, uint nIter,
                            bool deficient, double gd_stepsize, bool quiet)
{
  //std::cerr << "reducedibm2_par_m_step" << std::endl;

  const uint xDim = align_param.xDim();

  Math1D::Vector<double> cur_param(xDim);
  for (uint k = 0; k < xDim; k++)
    cur_param[k] = std::max(ibm2_min_align_param, align_param(k, j, c));

  Math1D::Vector<double> hyp_param(xDim);
  Math1D::Vector<double> new_param(xDim);
  Math1D::Vector<double> grad(xDim);

  Math1D::Vector<double> singleton_count(xDim, 0.0);
  Math1D::Vector<double> span_count(xDim, 0.0);

  for (uint k = 0; k < acount.size(); k++) {

    const Math3D::Tensor<double>& cur_acount = acount[k];
    if (cur_acount.yDim() > j) {

      uint curI = cur_acount.xDim()-1;
      for (uint i=1; i <= curI; i++) {
        singleton_count[i - 1] += cur_acount(i, j, c);
        span_count[curI - 1] += cur_acount(i, j, c);
      }
    }
  }

  double energy = reducedibm2_par_m_step_energy(cur_param, singleton_count, span_count);

  {
    //test start point
    double sum = singleton_count.sum();
    for (uint i = 0; i < xDim; i++)
      hyp_param[i] = std::max(ibm2_min_align_param, singleton_count[i] / sum);

    double hyp_energy = reducedibm2_par_m_step_energy(hyp_param, singleton_count, span_count);

    if (deficient || hyp_energy < energy) {
      if (!quiet)
        std::cerr << "switching to normalized counts: " << hyp_energy << " instead of " << energy << std::endl;

      energy = hyp_energy;
      cur_param = hyp_param;
    }
  }

  if (deficient) {

    align_param.set_x(j, c, cur_param);
    return;
  }

  double line_reduction_factor = 0.5;
  const double alpha = gd_stepsize;

  for (uint iter = 1; iter <= nIter; iter++) {
    //if ((iter % 15) == 0)
    //  std::cerr << "iter " << iter << ", energy: " << energy << std::endl;

    /***** compute gradient *****/
    for (uint i = 0; i < xDim; i++)
      grad[i] = -singleton_count[i] / cur_param[i];

    double param_sum = 0.0;
    for (uint i = 0; i < xDim; i++) {
      param_sum += cur_param[i];
      double cur_grad = span_count[i] / param_sum;
      for (uint k = 0; k <= i; k++)
        grad[k] += cur_grad;
    }

    /**** go in negative gradient direction and reproject ****/

	const double sqr_grad_norm = grad.sqr_norm();
	
	if (sqr_grad_norm < 1e-5)
	  break;
	
	const double real_alpha = alpha / sqrt(sqr_grad_norm);

    for (uint i = 0; i < xDim; i++)
      new_param[i] = cur_param[i] - real_alpha * grad[i];

    projection_on_simplex(new_param.direct_access(), xDim, ibm2_min_align_param);

    /**** find a suitable stepsize ****/

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint i = 0; i < xDim; i++)
        hyp_param[i] = lambda * new_param[i] + neg_lambda * cur_param[i];

      double new_energy = reducedibm2_par_m_step_energy(hyp_param, singleton_count, span_count);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    if (nTrials > 25 || fabs(energy - best_energy) < 1e-4) {
      if (!quiet)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint i = 0; i < xDim; i++)
      cur_param[i] = best_lambda * new_param[i] + neg_best_lambda * cur_param[i];
  }

  align_param.set_x(j, c, cur_param);
}

double reducedibm2_diffpar_m_step_energy(const Math3D::Tensor<double>& align_param, const Math1D::Vector<double>& singleton_count,
    const Math2D::TriMatrix<double>& span_count, uint c)
{
  const uint xDim = align_param.xDim();

  double energy = 0.0;

  for (uint x = 0; x < xDim; x++)
    energy -= singleton_count[x] * std::log(align_param(x, 0, c) );

  for (uint x_start = 0; x_start < xDim; x_start++) {

    double param_sum = 0.0;

    for (uint x_end = x_start; x_end < xDim; x_end++) {

      param_sum += align_param(x_end, 0, c);
      const double count = span_count(x_start, x_end);

      if (count != 0.0)
        energy += count * std::log(param_sum);
    }
  }

  return energy;
}

void reducedibm2_diffpar_m_step(Math3D::Tensor<double>& align_param, const ReducedIBM2ClassAlignmentModel& acount, uint offset, uint c,
                                uint nIter, bool deficient, double gd_stepsize)
{
  const uint xDim = align_param.xDim();

  Math1D::Vector<double> singleton_count(xDim, 0.0);
  Math2D::TriMatrix<double> span_count(xDim, 0.0);

  for (uint k = 0; k < acount.size(); k++) {

    const Math3D::Tensor<double>& cur_acount = acount[k];
    for (uint j = 0; j < cur_acount.yDim(); j++) {

      uint curI = cur_acount.xDim()-1;
      for (uint i=1; i <= curI; i++) {
        singleton_count[offset + j - (i - 1)] += cur_acount(i, j, c);
        span_count(offset + j - (curI - 1), offset + j) += cur_acount(i, j, c);
      }
    }
  }

  Math3D::Tensor<double> hyp_param(xDim, 1, align_param.zDim());
  Math1D::Vector<double> new_param(xDim);
  Math1D::Vector<double> grad(xDim);

  double energy = reducedibm2_diffpar_m_step_energy(align_param, singleton_count, span_count, c);

  {
    //test start point
    double sum = singleton_count.sum();
    for (uint x = 0; x < xDim; x++)
      hyp_param(x, 0, c) = singleton_count[x] / sum;

    double hyp_energy = reducedibm2_diffpar_m_step_energy(hyp_param, singleton_count, span_count, c);

    if (deficient || hyp_energy < energy) {

      //std::cerr << "switching to normalized counts" << std::endl;
      align_param = hyp_param;
      energy = hyp_energy;
    }
  }

  if (deficient)
    return;

  double line_reduction_factor = 0.5;
  const double alpha = gd_stepsize;

  for (uint iter = 1; iter <= nIter; iter++) {
    //if ((iter % 15) == 0)
    //  std::cerr << "iter " << iter << ", energy: " << energy << std::endl;

    /***** compute gradient *****/
    for (uint i = 0; i < xDim; i++)
      grad[i] = -singleton_count[i] / align_param(i, 0, c);

    for (uint x_start = 0; x_start < xDim; x_start++) {

      double param_sum = 0.0;

      for (uint x_end = x_start; x_end < xDim; x_end++) {

        param_sum += align_param(x_end, 0, c);
        const double count = span_count(x_start, x_end);

        if (count != 0.0) {
          const double cur_grad = count / param_sum;
          for (uint x = x_start; x <= x_end; x++)
            grad[x] += cur_grad;
        }
      }
    }

    /**** go in negative gradient direction and reproject ****/

	double sqr_grad_norm = grad.sqr_norm();
	if (sqr_grad_norm < 1e-5)
	  break;
	
	double real_alpha = alpha / sqrt(sqr_grad_norm);

    for (uint i = 0; i < xDim; i++)
      new_param[i] = align_param(i, 0, c) - real_alpha * grad[i];

    projection_on_simplex(new_param.direct_access(), xDim, ibm2_min_align_param);

    /**** find a suitable stepsize ****/

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint i = 0; i < xDim; i++)
        hyp_param(i, 0, c) = lambda * new_param[i] + neg_lambda * align_param(i, 0, c);

      double new_energy = reducedibm2_diffpar_m_step_energy(hyp_param, singleton_count, span_count, c);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    if (nTrials > 15 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint i = 0; i < xDim; i++)
      align_param(i, 0, c) = best_lambda * new_param[i] + neg_best_lambda * align_param(i, 0, c);
  }
}

/********************** HMM ***************************/

double ehmm_m_step_energy(const FullHMMAlignmentModel& facount, const Math1D::Vector<double>& dist_params,
                          uint zero_offset, double grouping_param, int redpar_limit)
{
  //std::cerr << "********** ehmm_m_step_energy << std::endl;
  double energy = 0.0;

  //std::cerr << "grouping_param: " << grouping_param << std::endl;

  for (uint I = 1; I <= facount.size(); I++) {

    if (facount[I - 1].size() > 0) {

      for (int i = 0; i < (int)I; i++) {

        double non_zero_sum = 0.0;

        if (grouping_param < 0.0) {

          for (uint ii = 0; ii < I; ii++)
            non_zero_sum += dist_params[zero_offset + ii - i];

          for (int ii = 0; ii < (int)I; ii++) {

            const double cur_count = facount[I - 1] (ii, i);

            //NOTE: division by non_zero_sum gives a constant due to log laws
            energy -= cur_count * std::log(dist_params[zero_offset + ii - i] / non_zero_sum);
          }
        }
        else {

          double grouping_norm = std::max(0, i - redpar_limit);
          grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

          //double check = 0.0;

          for (int ii = 0; ii < (int)I; ii++) {
            if (abs(ii - i) <= redpar_limit)
              non_zero_sum += dist_params[zero_offset + ii - i];
            else {
              non_zero_sum += grouping_param / grouping_norm;
              //check ++;
            }
          }

          //assert(check == grouping_norm);

          for (int ii = 0; ii < (int)I; ii++) {

            double cur_count = facount[I - 1] (ii, i);
            double cur_param;

            if (abs(ii - i) > redpar_limit)
              cur_param = grouping_param / grouping_norm;
            else
              cur_param = dist_params[zero_offset + ii - i];

            //NOTE: division by non_zero_sum gives a constant due to log laws
            //   same for division by grouping_norm
            energy -= cur_count * std::log(cur_param / non_zero_sum);
          }
        }
      }
    }
    //std::cerr << "intermediate energy: " << energy << std::endl;
  }

  return energy;
}

//new compact variant
double ehmm_m_step_energy(const Math1D::Vector<double>& singleton_count, double grouping_count,
                          const Math2D::Matrix<double>& span_count, const Math1D::Vector<double>& dist_params,
                          uint zero_offset, double grouping_param, int redpar_limit)
{
  //NOTE: we could exploit here that span_count will only be nonzero if zero_offset lies in the span

  //std::cerr << "***** ehmm_m_step_energy " << std::endl;
  //std::cerr << "span_count has dim " << span_count.xDim() << "x" << span_count.yDim() << std::endl;
  //std::cerr << "zero_offset: " << zero_offset << std::endl;

  double energy = 0.0;

  if (grouping_param < 0.0) {

    //std::cerr << "singleton terms " << std::endl;
    //std::cerr << "singleton dim: " << singleton_count.size() << std::endl;
    //std::cerr << "dist_parm dim: " << dist_params.size() << std::endl;
    assert(singleton_count.size() == dist_params.size());
    //singleton terms
    for (uint d = 0; d < singleton_count.size(); d++)
      energy -= singleton_count[d] * std::log(std::max(hmm_min_param_entry, dist_params[d]));

    //normalization terms
    //std::cerr << "normalization terms" << std::endl;
    Math1D::Vector<double> init_sum(zero_offset + 1);
    init_sum[zero_offset] = 0.0;
    init_sum[zero_offset - 1] = dist_params[zero_offset - 1];
    for (int s = zero_offset - 2; s >= 0; s--)
      init_sum[s] = init_sum[s + 1] + dist_params[s];

    for (uint span_start = 0; span_start <= zero_offset; span_start++) {
      //std::cerr << "span_start: " << span_start << std::endl;

      double param_sum = init_sum[span_start];  //dist_params.range_sum(span_start,zero_offset);
      for (uint span_end = zero_offset; span_end < singleton_count.size(); span_end++) {

        param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

        const double cur_count = span_count(span_start, span_end - zero_offset);
        if (cur_count != 0.0)
          energy += cur_count * std::log(param_sum);
      }
    }

  }
  else {

    const uint first_diff = zero_offset - redpar_limit;
    const uint last_diff = zero_offset + redpar_limit;

    for (uint d = first_diff; d <= last_diff; d++)
      energy -= singleton_count[d] * std::log(std::max(hmm_min_param_entry, dist_params[d]));

    //NOTE: because we do not divide grouping_param by the various numbers of affected positions
    //   this energy will differ from the inefficient version by a constant
    energy -= grouping_count * std::log(std::max(hmm_min_param_entry, grouping_param));

    //normalization terms
    for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

      for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

        //there should be plenty of room for speed-ups here

        const double cur_count = span_count(span_start, span_end - zero_offset);
        if (cur_count != 0.0) {

          double param_sum = 0.0;
          if (span_start < first_diff || span_end > last_diff)
            param_sum = grouping_param;

          for (uint d = std::max(first_diff, span_start);
               d <= std::min(span_end, last_diff); d++)
            param_sum += std::max(hmm_min_param_entry, dist_params[d]);

          energy += cur_count * std::log(param_sum);
        }
      }
    }
  }

  return energy;
}

void noncompact_ehmm_m_step(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset, uint nIter,
                            double& grouping_param, bool deficient, int redpar_limit, double gd_stepsize)
{
  std::cerr.precision(8);

  //std::cerr << "init params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;

  bool norm_constraint = true;

  if (grouping_param < 0.0) {
    projection_on_simplex(dist_params, hmm_min_param_entry);
  }
  else {
    projection_on_simplex_with_slack(dist_params.direct_access() + zero_offset - redpar_limit, grouping_param, 2 * redpar_limit + 1, hmm_min_param_entry);
    grouping_param = std::max(hmm_min_param_entry, grouping_param);
  }

  //std::cerr << "init params after projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param after projection: " << grouping_param << std::endl;

  Math1D::Vector<double> dist_grad = dist_params;
  Math1D::Vector<double> new_dist_params = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  double grouping_grad = 0.0;
  double new_grouping_param = grouping_param;
  double hyp_grouping_param = grouping_param;

  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(facount, dist_params, zero_offset, grouping_param, redpar_limit);

  double line_reduction_factor = 0.1;

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  //test if normalized counts give a better starting point
  {
    Math1D::Vector<double> dist_count(dist_params.size(), 0.0);
    double dist_grouping_count = (grouping_param < 0.0) ? -1.0 : 0.0;

    for (uint I = 1; I <= facount.size(); I++) {

      if (facount[I - 1].xDim() != 0) {

        for (int i = 0; i < (int)I; i++) {

          for (int ii = 0; ii < (int)I; ii++) {
            if (grouping_param < 0.0 || abs(ii - i) <= redpar_limit)
              dist_count[zero_offset + ii - i] += facount[I - 1](ii, i);
            else {
              //don't divide by grouping norm, the deficient problem doesn't need it:
              //  due to log laws we get additive constants
              dist_grouping_count += facount[I - 1](ii, i);
            }
          }
        }
      }
    }

    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += dist_count[zero_offset + k];
      norm += dist_grouping_count;

      if (norm > 1e-305) {

        for (uint k = 0; k < dist_count.size(); k++)
          dist_count[k] = std::max(hmm_min_param_entry, dist_count[k] / norm);

        dist_grouping_count = std::max(hmm_min_param_entry, dist_grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(facount, dist_count, zero_offset, dist_grouping_count, redpar_limit);

        //std::cerr << "hyp energy: " << hyp_energy << std::endl;

        if (hyp_energy < energy) {

          dist_params = dist_count;
          grouping_param = dist_grouping_count;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = dist_count.sum();

      if (sum > 1e-305) {
        for (uint k = 0; k < dist_count.size(); k++)
          dist_count[k] = std::max(hmm_min_param_entry, dist_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(facount, dist_count, zero_offset, grouping_param, redpar_limit);

        if (!deficient)
          std::cerr << "hyp energy: " << hyp_energy << std::endl;

        if (hyp_energy < energy) {
          dist_params = dist_count;
          energy = hyp_energy;
        }
      }

    }
  }

  if (deficient)
    return;

  std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  //double alpha  = 0.0001;
  //double alpha  = 0.001;
  double alpha = gd_stepsize;

  for (uint iter = 1; iter <= nIter; iter++) {

    if ((iter % 5) == 0)
      std::cerr << "m-step gd-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;

    //calculate gradient
    for (uint I = 1; I <= facount.size(); I++) {

      if (facount[I - 1].size() > 0) {

        for (int i = 0; i < (int)I; i++) {

          double grouping_norm = std::max(0, i - redpar_limit);
          grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));

          double non_zero_sum = 0.0;
          for (int ii = 0; ii < (int)I; ii++) {
            if (grouping_param < 0.0 || abs(i - ii) <= redpar_limit)
              non_zero_sum += dist_params[zero_offset + ii - i];
            else
              non_zero_sum += grouping_param / grouping_norm;
          }
          //if (grouping_param >= 0.0 && grouping_norm > 0.0)
          //  non_zero_sum += grouping_param;

          double count_sum = 0.0;
          for (int ii = 0; ii < (int)I; ii++) {
            count_sum += facount[I - 1] (ii, i);
          }

          for (int ii = 0; ii < (int)I; ii++) {
            double cur_param = dist_params[zero_offset + ii - i];

            double cur_count = facount[I - 1] (ii, i);

            if (grouping_param < 0.0) {
              dist_grad[zero_offset + ii - i] -= cur_count / cur_param;
            }
            else {

              if (abs(ii - i) > redpar_limit) {
                //NOTE: -std::log( param / norm) = -std::log(param) + std::log(norm)
                // => grouping_norm does NOT enter here
                grouping_grad -= cur_count / grouping_param;
              }
              else {
                dist_grad[zero_offset + ii - i] -= cur_count / cur_param;
              }
            }
          }

          for (int ii = 0; ii < (int)I; ii++) {
            if (grouping_param < 0.0 || abs(ii - i) <= redpar_limit)
              dist_grad[zero_offset + ii - i] += count_sum / non_zero_sum;
            // else
            //   m_grouping_grad += count_sum / (non_zero_sum * grouping_norm);
          }

          if (grouping_param >= 0.0 && grouping_norm > 0.0)
            grouping_grad += count_sum / non_zero_sum;
        }
      }
    }

    //go in gradient direction

    double real_alpha = alpha;

    //TRIAL
    double sqr_grad_norm  = dist_grad.sqr_norm();
    sqr_grad_norm += grouping_grad * grouping_grad;
    if (sqr_grad_norm < 1e-5) {
      if (true)
        std::cerr << "CUTOFF because squared gradient norm was " << sqr_grad_norm << std::endl;
      break;
    }


    real_alpha /= sqrt(sqr_grad_norm);
    //END_TRIAL

    //for (uint k = start_idx; k <= end_idx; k++)
    //  new_dist_params.direct_access(k) = dist_params.direct_access(k) - real_alpha * dist_grad.direct_access(k);
    Math1D::go_in_neg_direction(new_dist_params, dist_params, dist_grad, real_alpha);

    new_grouping_param = grouping_param - real_alpha * grouping_grad;

    //std::cerr << "new params before projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param before projection: " << new_grouping_param << std::endl;

    for (uint k = start_idx; k <= end_idx; k++) {
      if (new_dist_params[k] >= 1e75)
        new_dist_params[k] = 9e74;
      else if (new_dist_params[k] <= -1e75)
        new_dist_params[k] = -9e74;
    }
    if (new_grouping_param >= 1e75)
      new_grouping_param = 9e74;
    else if (new_grouping_param <= -1e75)
      new_grouping_param = -9e74;

    if (norm_constraint) {
      // reproject
      if (grouping_param < 0.0) {
        projection_on_simplex(new_dist_params, hmm_min_param_entry);
      }
      else {
        projection_on_simplex_with_slack(new_dist_params.direct_access() + start_idx, new_grouping_param, 2 * redpar_limit + 1, hmm_min_param_entry);
        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
      }
    }
    else {
      //projection on the positive orthant, followed by renormalization
      //(justified by scale invariance with positive scaling factors)
      // may be faster than the simplex projection

      uint nNeg = 0; //DEBUG

      double sum = 0.0;
      for (uint k = start_idx; k <= end_idx; k++) {

        //DEBUG
        if (new_dist_params[k] < 0.0)
          nNeg++;
        //END_DEBUG

        new_dist_params[k] = std::max(hmm_min_param_entry, new_dist_params[k]);
        sum += new_dist_params[k];
      }
      if (grouping_param >= 0.0) {

        //DEBUG
        if (new_grouping_param < 0.0)
          nNeg++;
        //END_DEBUG

        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
        sum += new_grouping_param;
      }
      //DEBUG
      //std::cerr << "sum: " << sum << ", " << nNeg << " were negative" << std::endl;
      //END_DEBUG

      //projection done, now renormalize to keep the probability constraint
      double inv_sum = 1.0 / sum;
      for (uint k = start_idx; k <= end_idx; k++) {
        new_dist_params[k] = std::max(hmm_min_param_entry,inv_sum * new_dist_params[k]);
      }
      new_grouping_param = std::max(hmm_min_param_entry,inv_sum * new_grouping_param);
    }

    //std::cerr << "new params after projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param after projection: " << new_grouping_param << std::endl;

    //find step-size

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint k = start_idx; k <= end_idx; k++)
        hyp_dist_params.direct_access(k) = std::max(hmm_min_param_entry, lambda * new_dist_params.direct_access(k) + neg_lambda * dist_params.direct_access(k));
      //Math1D::assign_weighted_combination(hyp_dist_params, lambda, new_dist_params, neg_lambda, dist_params);

      if (grouping_param >= 0.0)
        hyp_grouping_param = std::max(hmm_min_param_entry, lambda * new_grouping_param + neg_lambda * grouping_param);

      const double new_energy = ehmm_m_step_energy(facount, hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "hyp energy: " << new_energy << std::endl;
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    // if (nIter > 4)
    //   alpha *= 1.5;

    //DEBUG
    // if (best_lambda == 1.0)
    //   std::cerr << "!!!TAKING FULL STEP!!!" << std::endl;
    //END_DEBUG

    if (nTrials > 25 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k = start_idx; k <= end_idx; k++)
      dist_params.direct_access(k) = std::max(hmm_min_param_entry, neg_best_lambda * dist_params.direct_access(k) +
                                              best_lambda * new_dist_params.direct_access(k));
    //Math1D::assign_weighted_combination(dist_params, best_lambda, new_dist_params, neg_best_lambda, dist_params);

    if (grouping_param >= 0.0)
      grouping_param = std::max(hmm_min_param_entry, best_lambda * new_grouping_param + neg_best_lambda * grouping_param);

    //std::cerr << "updated params: " << dist_params << std::endl;
    //std::cerr << "updated grouping param: " << grouping_param << std::endl;
  }
}

void ehmm_m_step(const FullHMMAlignmentModelNoClasses& facount, Math1D::Vector<double>& dist_params, uint zero_offset,
                 uint nIter, double& grouping_param, bool deficient, int redpar_limit, double gd_stepsize, bool quiet, ProjectionMode projection_mode)
{
  bool norm_constraint = true;

  // 1. collect compact counts from facount

  Math1D::Vector<double> singleton_count(dist_params.size(), 0.0);
  double grouping_count = 0.0;
  Math2D::Matrix<double> span_count(zero_offset + 1, dist_params.size() - zero_offset, 0.0);

  //const uint maxI = dist_params.size() / 2 + 1;

  for (int I = 1; I <= int (facount.size()); I++) {

    //assert(I <= maxI);

    if (facount[I - 1].xDim() != 0) {

      for (int i = 0; i < I; i++) {

        double count_sum = 0.0;
        for (int i_next = 0; i_next < I; i_next++) {

          const double cur_count = facount[I - 1](i_next, i);
          if (grouping_param < 0.0 || abs(i_next - i) <= redpar_limit) {
            singleton_count[zero_offset + i_next - i] += cur_count;
            count_sum += cur_count;
          }
          else {
            double grouping_norm = std::max(0, i - redpar_limit);
            grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));
            grouping_count += cur_count / grouping_norm;
            count_sum += cur_count / grouping_norm;
          }
        }
        span_count(zero_offset - i, I - 1 - i) += count_sum;
      }
    }
  }

  ehmm_m_step(singleton_count, grouping_count, span_count, dist_params, zero_offset, grouping_param, deficient, redpar_limit,
              nIter, gd_stepsize, quiet, projection_mode);
}

void ehmm_m_step(const Math1D::Vector<double>& singleton_count, const double grouping_count, const Math2D::Matrix<double>& span_count,
                 Math1D::Vector<double>& dist_params, uint zero_offset, double& grouping_param, bool deficient, int redpar_limit,
                 uint nIter, double gd_stepsize, bool quiet, ProjectionMode projection_mode)
{

//DEBUG
//quiet = false;
//END_DEBUG

  //std::cerr << "****** ehmm_m_step, quiet: " << quiet << std::endl;
  assert(singleton_count.size() == dist_params.size());

  bool norm_constraint = true;
  // preparations and testing of starting points

  std::cerr.precision(8);

  //std::cerr << "dist params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;

  if (grouping_param >= 0.0) {
    for (uint k = 0; k < dist_params.size(); k++) {
      if (k < start_idx || k > end_idx)
        assert(singleton_count[k] == 0.0);
    }
  }

  if (projection_mode == Simplex) {
    if (grouping_param < 0.0) {
      projection_on_simplex(dist_params, hmm_min_param_entry);
    }
    else {
      projection_on_simplex_with_slack(dist_params.direct_access() + zero_offset - redpar_limit, grouping_param, 2 * redpar_limit + 1, hmm_min_param_entry);
      grouping_param = std::max(hmm_min_param_entry, grouping_param);
    }
  }
  else {

    //projection on the positive orthant, followed by renormalization
    //(justified by scale invariance with positive scaling factors)
    // may be faster than the simplex projection

    double sum = 0.0;
    for (uint k = start_idx; k <= end_idx; k++) {

      dist_params[k] = std::max(hmm_min_param_entry, dist_params[k]);
      sum += dist_params[k];
    }
    if (grouping_param >= 0.0) {

      grouping_param = std::max(hmm_min_param_entry, grouping_param);
      sum += grouping_param;
    }

    double inv_sum = 1.0 / sum;
    for (uint k = start_idx; k <= end_idx; k++) {
      dist_params[k] = std::max(hmm_min_param_entry, inv_sum * dist_params[k]);
    }
    grouping_param = std::max(hmm_min_param_entry, inv_sum * grouping_param);
  }

  //std::cerr << "dist params after projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param after projection: " << grouping_param << std::endl;

  Math1D::Vector<double> dist_grad(dist_params.size(), 0.0);
  Math1D::Vector<double> new_dist_params(dist_params.size(), 0.0);
  Math1D::Vector<double> hyp_dist_params(dist_params.size(), 0.0);

  double grouping_grad = 0.0;
  double new_grouping_param = grouping_param;
  double hyp_grouping_param = grouping_param;

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  //std::cerr << "computing energy" << std::endl;
  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, dist_params,
                  zero_offset, grouping_param, redpar_limit);

  //std::cerr << "computed energy: " << energy << std::endl;

  if (!deficient && energy == 0.0)
    return;

  //test if normalized counts give a better starting point
  {
    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += singleton_count[zero_offset + k];
      norm += grouping_count;

      if (norm < 1e-100)
        return;
      else {

        for (int k = zero_offset-redpar_limit; k <= zero_offset+redpar_limit; k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / norm);

        double hyp_grouping_param = std::max(hmm_min_param_entry, grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

        if (!deficient && !quiet) {
          //if (true) {
          //std::cerr << "norm: " << norm << std::endl;
          //std::cerr << "singleton_count: " << singleton_count << std::endl;
          //std::cerr << "start energy: " << energy << std::endl;
          //std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {
          if (!quiet)
            std::cerr << "switching to normalized counts" << std::endl;
          dist_params = hyp_dist_params;
          grouping_param = hyp_grouping_param;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = singleton_count.sum();
      //std::cerr << "sum: " << sum << std::endl;

      if (sum < 1e-100)
        return;
      else {

        assert(hyp_dist_params.size() == singleton_count.size());
        for (uint k = 0; k < hyp_dist_params.size(); k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, grouping_param, redpar_limit);

        if (!deficient && !quiet) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {
          if (!quiet)
            std::cerr << "switching to normalized counts" << std::endl;
          dist_params = hyp_dist_params;
          energy = hyp_energy;
        }
      }
    }
  }

  if (deficient)
    return;
  if (energy == 0.0)
    return;

  // 3. main loop
  if (!quiet)
    std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  //double alpha  = 0.0001;
  //double alpha  = 0.001;
  double alpha = gd_stepsize;

  double line_reduction_factor = 0.1;

  for (uint iter = 1; iter <= nIter; iter++) {

    if ((iter % 5) == 0 && !quiet)
      std::cerr << "m-step gd-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;

    // a) calculate gradient
    if (grouping_param < 0.0) {
      //fully parametric

      //singleton terms
      for (uint d = 0; d < singleton_count.size(); d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //normalization terms
      for (uint span_start = 0; span_start <= zero_offset; span_start++) {

        Math1D::Vector<double> addon(dist_params.size(), 0.0);

        double param_sum = dist_params.range_sum(span_start, zero_offset);
        for (uint span_end = zero_offset; span_end < dist_params.size(); span_end++) {

          param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            addon[span_end] = cur_count / param_sum;
            // double addon = cur_count / param_sum;

            // for (uint d=span_start; d <= span_end; d++)
            //   dist_grad[d] += addon;
          }
        }

        double sum_addon = 0.0;
        for (int d = dist_params.size() - 1; d >= int (zero_offset); d--) {
          sum_addon += addon[d];
          dist_grad[d] += sum_addon;
        }
        for (int d = zero_offset - 1; d >= int (span_start); d--)
          dist_grad[d] += sum_addon;
      }
    }
    else {
      //reduced parametric

      uint first_diff = zero_offset - redpar_limit;
      uint last_diff = zero_offset + redpar_limit;

      for (uint d = first_diff; d <= last_diff; d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //NOTE: because we do not divide grouping_param by the various numbers of affected positions
      //   this energy will differ from the inefficient version by a constant
      grouping_grad -= grouping_count / std::max(hmm_min_param_entry, grouping_param);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          //there should be plenty of room for speed-ups here

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            double param_sum = 0.0;
            if (span_start < first_diff || span_end > last_diff)
              param_sum = grouping_param;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              param_sum += std::max(hmm_min_param_entry, dist_params[d]);

            double addon = cur_count / param_sum;
            if (span_start < first_diff || span_end > last_diff)
              grouping_grad += addon;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              dist_grad[d] += addon;
          }
        }
      }
    }

    // b) go in gradient direction

    double real_alpha = alpha;

    //TRIAL
    double sqr_grad_norm  = dist_grad.sqr_norm();
    sqr_grad_norm += grouping_grad * grouping_grad;
    if (sqr_grad_norm < 1e-5) {
      if (!quiet)
        std::cerr << "CUTOFF after " << iter << " iterations because of squared gradient norm near 0: " << sqr_grad_norm << std::endl;
      return;
    }

    real_alpha /= sqrt(sqr_grad_norm);
    //END_TRIAL

    //std::cerr << "stepsize: " << real_alpha << std::endl;
    //std::cerr << "dist grad: " << dist_grad << std::endl;

    //for (uint k = start_idx; k <= end_idx; k++)
    //  new_dist_params.direct_access(k) = dist_params.direct_access(k) - real_alpha * dist_grad.direct_access(k);
    Math1D::go_in_neg_direction(new_dist_params, dist_params, dist_grad, real_alpha);
#if 0
    for (uint k = start_idx; k <= end_idx; k++) {
      double check = dist_params.direct_access(k) - real_alpha * dist_grad.direct_access(k);
      std::cerr << "k: " << k << ", should be " << check << ", is " << new_dist_params.direct_access(k) << std::endl;
      std::cerr << "calc " << dist_params.direct_access(k) << " - " << real_alpha << " * " << dist_grad.direct_access(k) << std::endl;
      assert((check - new_dist_params.direct_access(k)) < 1e-5);
    }
#endif

    new_grouping_param = grouping_param - real_alpha * grouping_grad;

    //std::cerr << "new params before projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param before projection: " << new_grouping_param << std::endl;

    // c) projection

    for (uint k = start_idx; k <= end_idx; k++) {
      if (new_dist_params[k] >= 1e75)
        new_dist_params[k] = 9e74;
      else if (new_dist_params[k] <= -1e75)
        new_dist_params[k] = -9e74;
    }
    if (new_grouping_param >= 1e75)
      new_grouping_param = 9e74;
    else if (new_grouping_param <= -1e75)
      new_grouping_param = -9e74;

    if (norm_constraint) {

      if (grouping_param < 0.0) {
        projection_on_simplex(new_dist_params, hmm_min_param_entry);
      }
      else {
        projection_on_simplex_with_slack(new_dist_params.direct_access() + start_idx, new_grouping_param,
                                         2 * redpar_limit + 1, hmm_min_param_entry);
        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
      }
    }
    else {

      //projection on the positive orthant, followed by renormalization
      //(justified by scale invariance with positive scaling factors)
      // may be faster than the simplex projection

      uint nNeg = 0; //DEBUG

      double sum = 0.0;
      for (uint k = start_idx; k <= end_idx; k++) {

        //DEBUG
        if (new_dist_params[k] < 0.0)
          nNeg++;
        //END_DEBUG

        new_dist_params[k] = std::max(hmm_min_param_entry, new_dist_params[k]);
        sum += new_dist_params[k];
      }
      if (grouping_param >= 0.0) {

        //DEBUG
        if (new_grouping_param < 0.0)
          nNeg++;
        //END_DEBUG

        new_grouping_param = std::max(hmm_min_param_entry, new_grouping_param);
        sum += new_grouping_param;
      }
      //DEBUG
      //std::cerr << "sum: " << sum << ", " << nNeg << " were negative" << std::endl;
      //END_DEBUG

      //projection done, now renormalize to keep the probability constraint
      double inv_sum = 1.0 / sum;
      for (uint k = start_idx; k <= end_idx; k++) {
        new_dist_params[k] = std::max(hmm_min_param_entry,inv_sum * new_dist_params[k]);
      }
      new_grouping_param = std::max(hmm_min_param_entry,inv_sum * new_grouping_param);
    }

    //std::cerr << "new params after projection: " << new_dist_params << std::endl;
    //std::cerr << "new grouping_param after projection: " << new_grouping_param << std::endl;

    // d) find step-size

    new_grouping_param = std::max(new_grouping_param, hmm_min_param_entry);

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      //for (uint k = start_idx; k <= end_idx; k++)
      //  hyp_dist_params.direct_access(k) = std::max(hmm_min_param_entry,lambda * new_dist_params.direct_access(k) +
      //                                              neg_lambda * dist_params.direct_access(k));
      Math1D::assign_weighted_combination(hyp_dist_params, lambda, new_dist_params, neg_lambda, dist_params);

      if (grouping_param >= 0.0)
        hyp_grouping_param = std::max(hmm_min_param_entry, lambda * new_grouping_param + neg_lambda * grouping_param);

      double new_energy = ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                                             hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "hyp energy: " << new_energy << std::endl;
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    // if (nIter > 4)
    //   alpha *= 1.5;

    //DEBUG
    // if (best_lambda == 1.0)
    //   std::cerr << "!!!TAKING FULL STEP!!!" << std::endl;
    //END_DEBUG

    if (nTrials > 25 || fabs(energy - best_energy) < 1e-4) {
      if (!quiet)
        std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint k = start_idx; k <= end_idx; k++)
    //  dist_params.direct_access(k) = std::max(hmm_min_param_entry, neg_best_lambda * dist_params.direct_access(k) +
    //                                          best_lambda * new_dist_params.direct_access(k));
    Math1D::assign_weighted_combination(dist_params, best_lambda, new_dist_params, neg_best_lambda, dist_params);

    if (grouping_param >= 0.0)
      grouping_param = std::max(hmm_min_param_entry, best_lambda * new_grouping_param + neg_best_lambda * grouping_param);

#ifndef NDEBUG
    double sum = dist_params.sum();
    if (grouping_param >= 0.0)
      sum += grouping_param;

    if (! (sum >= 0.99 && sum <= 1.01)) {
      std::cerr << "sum: " << sum << std::endl;
      std::cerr << "dist_params: " << dist_params << std::endl;
      std::cerr << "grouping param: " << grouping_param << std::endl;
    }
    assert(sum >= 0.99 && sum <= 1.01);
#endif

    //std::cerr << "updated params: " << dist_params << std::endl;
    //std::cerr << "updated grouping param: " << grouping_param << std::endl;
  }
}

//@returns the denominator of the renormalization expression
inline double unconstrained2constrained_m_step_point(const Math1D::Vector<double>& param, uint start_idx, uint end_idx, bool redpar,
    Math1D::Vector<double>& dist_prob, double& grouping_prob, int redpar_limit)
{
  const uint nParams = param.size();

  double sum = 0.0;

  if (!redpar) {

    for (uint k = 0; k < nParams; k++) {
      double x = param[k];
      dist_prob[k] = x * x;
      sum += x * x;
    }

    assert(sum > 1e-305);
    double inv_sum = 1.0 / sum;
    for (uint k = 0; k < nParams; k++) {
      dist_prob[k] = std::max(hmm_min_param_entry, dist_prob[k] * inv_sum);
    }
  }
  else {

    for (uint k = start_idx; k <= end_idx; k++) {
      double x = param[k - start_idx];
      dist_prob[k] = x * x;
      sum += x * x;
    }
    double x = param[2 * redpar_limit + 1];
    grouping_prob = x * x;
    sum += x * x;

    //std::cerr << "sum: " << sum << std::endl;

    assert(sum > 1e-305);
    double inv_sum = 1.0 / sum;
    for (uint k = start_idx; k <= end_idx; k++) {
      dist_prob[k] = std::max(hmm_min_param_entry, dist_prob[k] * inv_sum);
    }
    grouping_prob = std::max(hmm_min_param_entry, grouping_prob * inv_sum);
  }

  return sum;
}

void ehmm_m_step_unconstrained(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params,
                               uint zero_offset, uint nIter, double& grouping_param, bool deficient, int redpar_limit)
{
  //in this formulation we use parameters p=x^2 to get an unconstrained formulation
  // here we use nonlinear conjugate gradients  and as a special case gradient descent

  std::cerr.precision(8);

  //std::cerr << "init params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;

  const uint nParams = (grouping_param < 0.0) ? dist_params.size() : 2 * redpar_limit + 2;

  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);
  Math1D::Vector<double> prev_work_grad(nParams, 0.0);
  Math1D::Vector<double> hyp_work_param(nParams);

  Math1D::Vector<double> dist_grad = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  double grouping_grad = 0.0;
  double hyp_grouping_param = grouping_param;

  // collect compact counts from facount

  Math1D::Vector<double> singleton_count(dist_params.size(), 0.0);
  double grouping_count = 0.0;
  Math2D::Matrix<double> span_count(zero_offset + 1, dist_params.size() - zero_offset, 0.0);

  for (int I = 1; I <= int (facount.size()); I++) {

    if (facount[I - 1].xDim() != 0) {

      for (int i = 0; i < I; i++) {

        double count_sum = 0.0;
        for (int i_next = 0; i_next < I; i_next++) {

          const double cur_count = facount[I - 1](i_next, i);
          if (grouping_param < 0.0 || abs(i_next - i) <= redpar_limit) {
            singleton_count[zero_offset + i_next - i] += cur_count;
            count_sum += cur_count;
          }
          else {
            double grouping_norm = std::max(0, i - redpar_limit);
            grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));
            grouping_count += cur_count / grouping_norm;
            count_sum += cur_count / grouping_norm;
          }
        }
        span_count(zero_offset - i, I - 1 - i) += count_sum;
      }
    }
  }

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, dist_params,
                  zero_offset, grouping_param, redpar_limit);

  //test if normalized counts give a better starting point
  {
    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += singleton_count[zero_offset + k];
      norm += grouping_count;

      if (norm > 1e-305) {

        for (int k = zero_offset-redpar_limit; k <= zero_offset+redpar_limit; k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / norm);

        double hyp_grouping_param = std::max(hmm_min_param_entry, grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {

          dist_params = hyp_dist_params;
          grouping_param = hyp_grouping_param;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = singleton_count.sum();

      if (sum > 1e-305) {
        for (uint k = 0; k < hyp_dist_params.size(); k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                            hyp_dist_params, zero_offset, grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {
          dist_params = hyp_dist_params;
          energy = hyp_energy;
        }
      }
    }
  }

  if (deficient)
    return;

  //std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  double alpha = 1.0;           //modified below, dependent on the gradient norm

  //extract working params from the current probabilities (the probabilities are the squared working params)
  if (grouping_param < 0.0) {
    for (uint k = 0; k < nParams; k++)
      work_param[k] = sqrt(dist_params[k]);
  }
  else {
    for (uint k = start_idx; k <= end_idx; k++) {

      work_param[k - start_idx] = sqrt(dist_params[k]);
      work_param[2 * redpar_limit + 1] = sqrt(grouping_param);
    }
  }

  double prev_grad_norm = 0.0;

  double line_reduction_factor = 0.5;

  bool restart = false;

  for (uint iter = 1; iter <= nIter; iter++) {

    //if ((iter % 5) == 0)
    std::cerr << "unconstrained m-step gd/nlcg-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;
    work_grad.set_constant(0.0);

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    if (grouping_param < 0.0) {
      //fully parametric

      //singleton terms
      for (uint d = 0; d < singleton_count.size(); d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        Math1D::Vector<double> addon(dist_params.size(), 0.0);

        Math1D::Vector<double> init_sum(zero_offset + 1);
        init_sum[zero_offset] = 0.0;
        init_sum[zero_offset - 1] = dist_params[zero_offset - 1];
        for (int s = zero_offset - 2; s >= 0; s--)
          init_sum[s] = init_sum[s + 1] + dist_params[s];

        double param_sum = init_sum[span_start];        //dist_params.range_sum(span_start,zero_offset);
        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            addon[span_end] = cur_count / param_sum;
            // double addon = cur_count / param_sum;

            // for (uint d=span_start; d <= span_end; d++)
            //   dist_grad[d] += addon;
          }
        }

        double sum_addon = 0.0;
        for (int d = zero_offset + span_count.yDim() - 1; d >= int (zero_offset); d--) {
          sum_addon += addon[d];
          dist_grad[d] += sum_addon;
        }
        for (int d = zero_offset - 1; d >= int (span_start); d--)
          dist_grad[d] += sum_addon;
      }
    }
    else {
      //reduced parametric

      uint first_diff = zero_offset - redpar_limit;
      uint last_diff = zero_offset + redpar_limit;

      for (uint d = first_diff; d <= last_diff; d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //NOTE: because we do not divide grouping_param by the various numbers of affected positions
      //   this energy will differ from the inefficient version by a constant
      grouping_grad -= grouping_count / std::max(hmm_min_param_entry, grouping_param);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          //there should be plenty of room for speed-ups here

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            double param_sum = 0.0;
            if (span_start < first_diff || span_end > last_diff)
              param_sum = grouping_param;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              param_sum += std::max(hmm_min_param_entry, dist_params[d]);

            double addon = cur_count / param_sum;
            if (span_start < first_diff || span_end > last_diff)
              grouping_grad += addon;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              dist_grad[d] += addon;
          }
        }
      }
    }

    // b) now calculate the gradient for the actual parameters, store in work_grad

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    if (grouping_param < 0.0) {

      double coeff_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        const double wp = work_param[k];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;

        work_grad[k] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }
    else {

      double coeff_sum = 0.0;

      for (uint k = start_idx; k <= end_idx; k++) {
        const double wp = work_param[k - start_idx];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;
        work_grad[k - start_idx] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      const double wp = work_param[2 * redpar_limit + 1];
      const double grad = grouping_grad;
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[2 * redpar_limit + 1] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //        work_grad[kk] -= coeff * work_param[kk];

      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }

    double new_grad_norm = work_grad.sqr_norm();

    //in NLCG mode: modify the search direction
    if (iter == 1 || restart) {
      //if (true) {
      std::cerr << "RESTART" << std::endl;
      search_direction = work_grad;
      Math1D::negate(search_direction);
    }
    else {
      double beta_fr = new_grad_norm / prev_grad_norm;  //Fletcher-Reeves variant

      double numerator = 0.0;
      double beta_hs_denom = 0.0;
      for (uint k = 0; k < nParams; k++) {

        const double diff = work_grad[k] - prev_work_grad[k];
        numerator += diff * work_grad[k];
        beta_hs_denom += diff * search_direction[k];
      }

      double beta_hs = numerator / beta_hs_denom;
      //beta_hs = std::max(0.0,beta_hs);
      if (beta_hs < -beta_fr)
        beta_hs = -beta_fr;
      if (beta_hs > beta_fr)
        beta_hs = beta_fr;

      double beta = beta_hs;

      for (uint k = 0; k < nParams; k++) {
        search_direction[k] = -work_grad[k] + beta * search_direction[k];
      }
    }

    alpha = 1.0 / sqrt(search_direction.sqr_norm());

    // c) line search

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double best_alpha = alpha;

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      if (nIter > 1)
        alpha *= line_reduction_factor;

      for (uint k = 0; k < nParams; k++)
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];

      //calculate corresponding probability distribution (square the params and renormalize)
      unconstrained2constrained_m_step_point(hyp_work_param, start_idx, end_idx, grouping_param >= 0,
                                             hyp_dist_params, hyp_grouping_param, redpar_limit);

      double new_energy = ehmm_m_step_energy(singleton_count, grouping_count, span_count,
                                             hyp_dist_params, zero_offset, hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "alpha: " << alpha << ", hyp energy: " << new_energy << std::endl;
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    //d) go to the determined point

    if (best_energy >= energy - 1e-4) {
      if (!restart)
        restart = true;
      else {
        std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy) << std::endl;
        std::cerr << "last squared gradient norm: " << new_grad_norm << std::endl;
        break;
      }
    }
    else
      restart = false;

    if (best_energy < energy) {
      energy = best_energy;

      for (uint k = 0; k < nParams; k++)
        work_param[k] += best_alpha * search_direction[k];

      //calculate corresponding probability distribution (square the params and renormalize)

      unconstrained2constrained_m_step_point(work_param, start_idx, end_idx, grouping_param >= 0, dist_params,
                                             grouping_param, redpar_limit);

      double sum = dist_params.sum();
      if (grouping_param >= 0.0)
        sum += grouping_param;

      assert(sum >= 0.99 && sum <= 1.01);
    }

    prev_work_grad = work_grad; //could be more efficient
    prev_grad_norm = new_grad_norm;
  }

}

void ehmm_m_step_unconstrained_LBFGS(const FullHMMAlignmentModel& facount, Math1D::Vector<double>& dist_params, uint zero_offset, uint nIter,
                                     double& grouping_param, uint L, bool deficient, int redpar_limit)
{
  //in this formulation we use parameters p=x^2 to get an unconstrained formulation

  std::cerr.precision(8);

  //std::cerr << "init params before projection: " << dist_params << std::endl;
  //std::cerr << "grouping_param before projection: " << grouping_param << std::endl;

  const uint start_idx = (grouping_param < 0.0) ? 0 : zero_offset - redpar_limit;
  const uint end_idx = (grouping_param < 0.0) ? dist_params.size() - 1 : zero_offset + redpar_limit;

  const uint nParams = (grouping_param < 0.0) ? dist_params.size() : 2 * redpar_limit + 2;

  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);
  Math1D::Vector<double> hyp_work_param(nParams);

  Math1D::Vector<double> dist_grad = dist_params;
  Math1D::Vector<double> hyp_dist_params = dist_params;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  double grouping_grad = 0.0;
  double hyp_grouping_param = grouping_param;

  // collect compact counts from facount

  Math1D::Vector<double> singleton_count(dist_params.size(), 0.0);
  double grouping_count = 0.0;
  Math2D::Matrix<double> span_count(zero_offset + 1, dist_params.size() - zero_offset, 0.0);

  for (int I = 1; I <= int (facount.size()); I++) {

    if (facount[I - 1].xDim() != 0) {

      for (int i = 0; i < I; i++) {

        double count_sum = 0.0;
        for (int i_next = 0; i_next < I; i_next++) {

          double cur_count = facount[I - 1] (i_next, i);
          if (grouping_param < 0.0 || abs(i_next - i) <= redpar_limit) {
            singleton_count[zero_offset + i_next - i] += cur_count;
            count_sum += cur_count;
          }
          else {
            double grouping_norm = std::max(0, i - redpar_limit);
            grouping_norm += std::max(0, int (I) - 1 - (i + redpar_limit));
            grouping_count += cur_count / grouping_norm;
            count_sum += cur_count / grouping_norm;
          }
        }
        span_count(zero_offset - i, I - 1 - i) += count_sum;
      }
    }
  }

  //NOTE: the deficient closed-form solution does not necessarily have a lower energy INCLUDING the normalization term

  double energy = (deficient) ? 0.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, dist_params,
                  zero_offset, grouping_param, redpar_limit);

  //test if normalized counts give a better starting point
  {
    if (grouping_param >= 0.0) {
      //reduced parametric

      double norm = 0.0;
      for (int k = -redpar_limit; k <= redpar_limit; k++)
        norm += singleton_count[zero_offset + k];
      norm += grouping_count;

      if (norm > 1e-305) {

        for (int k = zero_offset-redpar_limit; k <= zero_offset+redpar_limit; k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / norm);

        double hyp_grouping_param = std::max(hmm_min_param_entry, grouping_count / norm);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, hyp_dist_params, zero_offset,
                            hyp_grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {

          dist_params = hyp_dist_params;
          grouping_param = hyp_grouping_param;
          energy = hyp_energy;
        }
      }
    }
    else {
      //fully parametric

      double sum = singleton_count.sum();

      if (sum > 1e-305) {
        for (uint k = 0; k < hyp_dist_params.size(); k++)
          hyp_dist_params[k] = std::max(hmm_min_param_entry, singleton_count[k] / sum);

        double hyp_energy = (deficient) ? -1.0 : ehmm_m_step_energy(singleton_count, grouping_count, span_count, hyp_dist_params, zero_offset,
                            grouping_param, redpar_limit);

        if (!deficient) {
          std::cerr << "start energy: " << energy << std::endl;
          std::cerr << "hyp energy: " << hyp_energy << std::endl;
        }

        if (hyp_energy < energy) {
          dist_params = hyp_dist_params;
          energy = hyp_energy;
        }
      }
    }
  }

  if (deficient)
    return;

  //std::cerr << "start m-energy: " << energy << std::endl;

  assert(grouping_param < 0.0 || grouping_param >= hmm_min_param_entry);

  //extract working params from the current probabilities (the probabilities are the squared working params)
  if (grouping_param < 0.0) {
    for (uint k = 0; k < nParams; k++)
      work_param[k] = sqrt(dist_params[k]);
  }
  else {
    for (uint k = start_idx; k <= end_idx; k++) {

      work_param[k - start_idx] = sqrt(dist_params[k]);
      work_param[2 * redpar_limit + 1] = sqrt(grouping_param);
    }
  }

  double scale = 1.0;

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint iter = 1; iter <= nIter; iter++) {

    if ((iter % 5) == 0)
      std::cerr << "unconstrained m-step L-BFGS(" << L << ")-iter #" << iter << ", cur energy: " << energy << std::endl;

    dist_grad.set_constant(0.0);
    grouping_grad = 0.0;

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    if (grouping_param < 0.0) {
      //fully parametric

      //singleton terms
      for (uint d = 0; d < singleton_count.size(); d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        Math1D::Vector<double> addon(dist_params.size(), 0.0);

        Math1D::Vector<double> init_sum(zero_offset + 1);
        init_sum[zero_offset] = 0.0;
        init_sum[zero_offset - 1] = dist_params[zero_offset - 1];
        for (int s = zero_offset - 2; s >= 0; s--)
          init_sum[s] = init_sum[s + 1] + dist_params[s];

        double param_sum = init_sum[span_start];        //dist_params.range_sum(span_start,zero_offset);
        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          param_sum += std::max(hmm_min_param_entry, dist_params[span_end]);

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            addon[span_end] = cur_count / param_sum;
            // double addon = cur_count / param_sum;

            // for (uint d=span_start; d <= span_end; d++)
            //   dist_grad[d] += addon;
          }
        }

        double sum_addon = 0.0;
        for (int d = zero_offset + span_count.yDim() - 1; d >= int (zero_offset); d--) {
          sum_addon += addon[d];
          dist_grad[d] += sum_addon;
        }
        for (int d = zero_offset - 1; d >= int (span_start); d--)
          dist_grad[d] += sum_addon;
      }
    }
    else {
      //reduced parametric

      uint first_diff = zero_offset - redpar_limit;
      uint last_diff = zero_offset + redpar_limit;

      for (uint d = first_diff; d <= last_diff; d++)
        dist_grad[d] -= singleton_count[d] / std::max(hmm_min_param_entry, dist_params[d]);

      //NOTE: because we do not divide grouping_param by the various numbers of affected positions
      //   this energy will differ from the inefficient version by a constant
      grouping_grad -= grouping_count / std::max(hmm_min_param_entry, grouping_param);

      //normalization terms
      for (uint span_start = 0; span_start < span_count.xDim(); span_start++) {

        for (uint span_end = zero_offset; span_end < zero_offset + span_count.yDim(); span_end++) {

          //there should be plenty of room for speed-ups here

          const double cur_count = span_count(span_start, span_end - zero_offset);
          if (cur_count != 0.0) {

            double param_sum = 0.0;
            if (span_start < first_diff || span_end > last_diff)
              param_sum = grouping_param;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              param_sum += std::max(hmm_min_param_entry, dist_params[d]);

            double addon = cur_count / param_sum;
            if (span_start < first_diff || span_end > last_diff)
              grouping_grad += addon;

            for (uint d = std::max(first_diff, span_start); d <= std::min(span_end, last_diff); d++)
              dist_grad[d] += addon;
          }
        }
      }
    }

    // b) now calculate the gradient for the actual parameters, store in work_grad

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = scale; //work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    std::cerr << "scale: " << denom << std::endl;

    if (grouping_param < 0.0) {

      double coeff_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        const double wp = work_param[k];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;

        work_grad[k] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }
    else {

      double coeff_sum = 0.0;

      for (uint k = start_idx; k <= end_idx; k++) {
        const double wp = work_param[k - start_idx];
        const double grad = dist_grad[k];
        const double param_sqr = wp * wp;
        const double coeff = 2.0 * grad * param_sqr / denom_sqr;
        work_grad[k - start_idx] += 2.0 * grad * wp / denom;

        coeff_sum += coeff;
        // for (uint kk=0; kk < nParams; kk++)
        //   work_grad[kk] -= coeff * work_param[kk];
      }
      const double wp = work_param[2 * redpar_limit + 1];
      const double grad = grouping_grad;
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[2 * redpar_limit + 1] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //        work_grad[kk] -= coeff * work_param[kk];

      for (uint kk = 0; kk < nParams; kk++)
        work_grad[kk] -= coeff_sum * work_param[kk];
    }

    double new_grad_norm = work_grad.sqr_norm();

    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < nParams; k++) {

        //cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
        cur_grad_diff[k] += work_grad[k];
        cur_rho += cur_grad_diff[k] * cur_step[k];
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      if (cur_curv <= 0) {
        //this can happen as our function is not convex and we do not enforce part 2 of the Wolfe conditions
        // (this cannot be done by backtracking line search, see Algorithm 3.5 in [Nocedal & Wright])
        // Our solution is to simply restart L-BFGS now
        start_iter = iter;
      }

      rho[cur_l] = 1.0 / cur_rho;
    }

    search_direction = work_grad;

    if (iter > start_iter) {

      Math1D::Vector<double> alpha(L);

      const int cur_first_iter = std::max<int>(start_iter, iter - L);

      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter - 1; prev_iter >= cur_first_iter; prev_iter--) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double cur_alpha = search_direction % cur_step;
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        search_direction.add_vector_multiple(cur_grad_diff, -cur_alpha);
        // for (uint k=0; k < nParams; k++)
        //   search_direction[k] -= cur_alpha * cur_grad_diff[k];
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = search_direction % cur_grad_diff;
        beta *= rho[prev_l];

        const double gamma = alpha[prev_l] - beta;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] += cur_step[k] * gamma;
        }
      }

    }
    else {
      search_direction *= 1.0 / sqrt(search_direction.sqr_norm());
    }

    Math1D::negate(search_direction);

    // d) line search

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double alpha = 1.0;
    double best_alpha = alpha;

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      if (nTrials > 1)
        alpha *= line_reduction_factor;

      for (uint k = 0; k < nParams; k++)
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];

      //calculate corresponding probability distribution (square the params and renormalize)

      unconstrained2constrained_m_step_point(hyp_work_param, start_idx, end_idx, (grouping_param >= 0),
                                             hyp_dist_params, hyp_grouping_param, redpar_limit);

      double new_energy = ehmm_m_step_energy(singleton_count, grouping_count, span_count, hyp_dist_params, zero_offset,
                                             hyp_grouping_param, redpar_limit);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }

      //std::cerr << "lambda: " << lambda << ", hyp energy: " << new_energy << std::endl;
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point

    if (best_energy >= energy - 1e-4) {
      std::
      cerr << "CUTOFF after " << iter << " iterations, last gain: " <<
           (energy - best_energy) << std::endl;
      std::cerr << "last squared gradient norm: " << new_grad_norm << std::endl;
      break;
    }

    energy = best_energy;

    uint cur_l = (iter % L);

    Math1D::Vector<double>& cur_step = step[cur_l];
    Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

    for (uint k = 0; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    //calculate corresponding probability distribution (square the params and renormalize)
    scale = unconstrained2constrained_m_step_point(work_param, start_idx, end_idx, grouping_param >= 0, dist_params, grouping_param, redpar_limit);

    double sum = dist_params.sum();
    if (grouping_param >= 0.0)
      sum += grouping_param;

    assert(sum >= 0.99 && sum <= 1.01);
  }
}

double ehmm_init_m_step_energy(const InitialAlignmentProbability& init_acount, const Math1D::Vector<double>& init_params)
{
  double energy = 0.0;

  for (uint I = 0; I < init_acount.size(); I++) {

    if (init_acount[I].size() > 0) {

      double non_zero_sum = 0.0;
      for (uint i = 0; i <= I; i++)
        non_zero_sum += init_params[i];

      for (uint i = 0; i <= I; i++) {

        energy -= init_acount[I][i] * std::log(init_params[i] / non_zero_sum);
      }
    }
  }

  return energy;
}

void ehmm_init_m_step(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter,
                      ProjectionMode projection_mode, double gd_stepsize)
{
  projection_on_simplex(init_params, hmm_min_param_entry);

  Math1D::Vector<double> m_init_grad = init_params;
  Math1D::Vector<double> new_init_params = init_params;
  Math1D::Vector<double> hyp_init_params = init_params;

  double energy = ehmm_init_m_step_energy(init_acount, init_params);

  {
    //try to find a better starting point
    Math1D::Vector<double> init_count(init_params.size(), 0.0);

    for (uint I = 1; I <= init_acount.size(); I++) {

      if (init_acount[I - 1].size() != 0) {
        for (uint i = 0; i < I; i++) {
          init_count[i] += init_acount[I - 1][i];
        }
      }
    }

    const double sum = init_count.sum();

    for (uint k=0; k < init_count.size(); k++)
      init_count[k] = std::max(hmm_min_param_entry, init_count[k] / sum);

    double hyp_energy = ehmm_init_m_step_energy(init_acount, init_count);

    if (hyp_energy < energy) {
      init_params = init_count;
      energy = hyp_energy;
    }
  }

  std::cerr << "start m-energy: " << energy << std::endl;

  for (uint iter = 1; iter <= nIter; iter++) {

    m_init_grad.set_constant(0.0);

    //calculate gradient
    for (uint I = 0; I < init_acount.size(); I++) {

      if (init_acount[I].size() > 0) {

        double non_zero_sum = 0.0;
        for (uint i = 0; i <= I; i++)
          non_zero_sum += init_params[i];

        double count_sum = 0.0;
        for (uint i = 0; i <= I; i++) {
          count_sum += init_acount[I][i];

          double cur_param = init_params[i];

          m_init_grad[i] -= init_acount[I][i] / cur_param;
        }

        for (uint i = 0; i <= I; i++) {
          m_init_grad[i] += count_sum / non_zero_sum;
        }
      }
    }

    //std::cerr << "gradient calculated" << std::endl;

    //TRIAL
    //double alpha = 1.0 / sqrt(m_init_grad.sqr_norm());
    //END_TRIAL

    double sqr_grad_norm = m_init_grad.sqr_norm();
    if (sqr_grad_norm < 1e-5) {
      if (true)
        std::cerr << "CUTOFF because squared gradient norm was " << sqr_grad_norm << std::endl;
      break;
    }

    double alpha = gd_stepsize / sqrt(sqr_grad_norm);

    //for (uint k = 0; k < init_params.size(); k++) {
    //  new_init_params.direct_access(k) = init_params.direct_access(k) - alpha * m_init_grad.direct_access(k);
    //  assert(!isnan(new_init_params[k]));
    //}
    Math1D::go_in_neg_direction(new_init_params, init_params, m_init_grad, alpha);

    //std::cerr << "projecting " << new_init_params << std::endl;

    // reproject
    for (uint k = 0; k < init_params.size(); k++) {
      if (new_init_params[k] >= 1e75)
        new_init_params[k] = 9e74;
      if (new_init_params[k] <= -1e75)
        new_init_params[k] = -9e74;
    }

    if (projection_mode == Simplex)
      projection_on_simplex(new_init_params, hmm_min_param_entry);
    else {

      double sum = 0;
      for (uint k = 0; k < init_params.size(); k++) {
        new_init_params[k] = std::max(hmm_min_param_entry, new_init_params[k]);
        sum += new_init_params[k];
      }

      //projection on orthant done, now renormalize to keep the probability constraint
      new_init_params *= 1.0 / sum;
    }

    //std::cerr << "projection done" << std::endl;

    //find step-size
    double best_energy = 1e300;

    double lambda = 1.0;
    double line_reduction_factor = 0.5;
    double best_lambda = lambda;

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      //for (uint k = 0; k < init_params.size(); k++)
      //  hyp_init_params.direct_access(k) = lambda * new_init_params.direct_access(k) + neg_lambda * init_params.direct_access(k);
      Math1D::assign_weighted_combination(hyp_init_params, lambda, new_init_params, neg_lambda, init_params);

      double hyp_energy = ehmm_init_m_step_energy(init_acount, hyp_init_params);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint k = 0; k < init_params.size(); k++)
    //  init_params.direct_access(k) = neg_best_lambda * init_params.direct_access(k) + best_lambda * new_init_params.direct_access(k);
    Math1D::assign_weighted_combination(init_params, best_lambda, new_init_params, neg_best_lambda, init_params);

    if (nTrials > 25 || best_lambda < 1e-12 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF" << std::endl;
      break;
    }

    energy = best_energy;

    if ((iter % 5) == 0)
      std::cerr << "init m-step gd-iter #" << iter << ", energy: " << energy << std::endl;
  }
}

double start_prob_m_step_energy(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                                const Math1D::Vector<double>& param)
{
  double energy = 0.0;

  const uint nParams = param.size();

  for (uint j = 0; j < nParams; j++)
    energy -= singleton_count[j] * std::log(std::max(1e-15, param[j]));

  double param_sum = 0.0;
  for (uint j = 0; j < nParams; j++) {

    param_sum += std::max(1e-15, param[j]);
    energy += norm_count[j] * std::log(param_sum);
  }

  return energy;
}

void start_prob_m_step(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                       Math1D::Vector<double>& sentence_start_parameters, uint nIter, double gd_stepsize)
{
  const uint nParams = sentence_start_parameters.size();

  Math1D::Vector<double> param_grad(nParams);
  Math1D::Vector<double> new_param(nParams);
  Math1D::Vector<double> hyp_param(nParams);

  for (uint j = 0; j < nParams; j++)
    sentence_start_parameters[j] =  std::max(1e-10, sentence_start_parameters[j]);

  double energy = start_prob_m_step_energy(singleton_count, norm_count, sentence_start_parameters);

  //check if normalizing the singleton count gives a better starting point
  {
    const double sum = singleton_count.sum();

    if (sum > 1e-305) {

      for (uint j = 0; j < nParams; j++)
        hyp_param[j] = std::max(1e-10, singleton_count[j] / sum);

      const double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < energy) {

        sentence_start_parameters = hyp_param;
        energy = hyp_energy;
      }
    }
  }

  std::cerr << "start energy: " << energy << std::endl;

  double alpha = gd_stepsize;

  for (uint iter = 1; iter <= nIter; iter++) {

    param_grad.set_constant(0.0);

    //calculate gradient

    for (uint j = 0; j < nParams; j++)
      param_grad[j] -= singleton_count[j] / std::max(1e-10, sentence_start_parameters[j]);

    Math1D::Vector<double> addon(nParams);

    double param_sum = 0.0;
    for (uint j = 0; j < nParams; j++) {

      param_sum += std::max(1e-15, sentence_start_parameters[j]);

      addon[j] = norm_count[j] / param_sum;
      // const double addon = norm_count[j] / param_sum;

      // for (uint jj=0; jj <= j; jj++)
      //        param_grad[jj] += addon;
    }

    double addon_sum = 0.0;
    for (int j = nParams - 1; j >= 0; j--) {

      addon_sum += addon[j];
      param_grad[j] += addon_sum;
    }

    //go in neg. gradient direction
    //for (uint k = 0; k < sentence_start_parameters.size(); k++)
    //  new_param[k] = sentence_start_parameters[k] - alpha * param_grad[k];

    //new_param = sentence_start_parameters;
    //new_param.add_vector_multiple(param_grad, -alpha);

    double sqr_grad_norm = param_grad.sqr_norm();
    if (sqr_grad_norm < 1e-5) {
      std::cerr << "CUTOFF because squared gradient norm was " << sqr_grad_norm << std::endl;
      break;
    }

    double real_alpha = alpha / sqrt(sqr_grad_norm);
    Math1D::go_in_neg_direction(new_param, sentence_start_parameters, param_grad, alpha);

    //reproject
    projection_on_simplex(new_param, 1e-10);

    //find step-size
    double best_energy = 1e300;
    bool decreasing = true;

    double lambda = 1.0;
    double best_lambda = 1.0;

    uint nIter = 0;

    while (best_energy > energy || decreasing) {

      nIter++;

      lambda *= 0.5;
      double neg_lambda = 1.0 - lambda;

      //for (uint k = 0; k < new_param.size(); k++)
      //  hyp_param[k] = neg_lambda * sentence_start_parameters[k] + lambda * new_param[k];
      Math1D::assign_weighted_combination(hyp_param, neg_lambda, sentence_start_parameters, lambda, new_param);

      const double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < best_energy) {

        decreasing = true;
        best_lambda = lambda;
        best_energy = hyp_energy;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }

    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint k = 0; k < new_param.size(); k++)
    //  sentence_start_parameters[k] = neg_best_lambda * sentence_start_parameters[k] + best_lambda * new_param[k];
    Math1D::assign_weighted_combination(sentence_start_parameters, neg_best_lambda, sentence_start_parameters, best_lambda, new_param);

    energy = best_energy;

    if ((iter % 5) == 0)
      std::cerr << "#init m-step iter " << iter << ", energy: " << energy << std::endl;
    energy = best_energy;
  }
}

//use L-BFGS
void start_prob_m_step_unconstrained(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                                     Math1D::Vector<double>& sentence_start_parameters, uint nIter, uint L)
{
  //in this formulation we use parameters p=x^2 to get an unconstrained formulation
  // here we use L-BFGS

  const uint nParams = sentence_start_parameters.size();

  Math1D::Vector<double> param_grad(nParams);
  Math1D::Vector<double> hyp_param(nParams);
  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);

  for (uint j = 0; j < nParams; j++)
    sentence_start_parameters[j] = std::max(1e-10, sentence_start_parameters[j]);

  double energy = start_prob_m_step_energy(singleton_count, norm_count, sentence_start_parameters);

  //check if normalizing the singleton count gives a better starting point
  {
    const double sum = singleton_count.sum();

    if (sum > 1e-305) {

      for (uint j = 0; j < nParams; j++)
        hyp_param[j] = std::max(1e-10, singleton_count[j] / sum);

      const double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < energy) {

        sentence_start_parameters = hyp_param;
        energy = hyp_energy;
      }
    }
  }

  std::cerr << "start energy: " << energy << std::endl;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  //extract working params from the current probabilities (the probabilities are the squared working params)
  for (uint k = 0; k < nParams; k++)
    work_param[k] = sqrt(sentence_start_parameters[k]);

  double scale = 1.0;

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint iter = 1; iter <= nIter; iter++) {

    param_grad.set_constant(0.0);
    work_grad.set_constant(0.0);

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    for (uint j = 0; j < nParams; j++)
      param_grad[j] -= singleton_count[j] / std::max(1e-10, sentence_start_parameters[j]);

    Math1D::Vector<double> addon(nParams);

    double param_sum = 0.0;
    for (uint j = 0; j < nParams; j++) {

      param_sum += std::max(1e-10, sentence_start_parameters[j]);

      addon[j] = norm_count[j] / param_sum;
      // const double addon = norm_count[j] / param_sum;

      // for (uint jj=0; jj <= j; jj++)
      //        param_grad[jj] += addon;
    }

    double addon_sum = 0.0;
    for (int j = nParams - 1; j >= 0; j--) {

      addon_sum += addon[j];
      param_grad[j] += addon_sum;
    }

    // b) now calculate the gradient for the actual parameters

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = scale; //work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    double coeff_sum = 0.0;

    for (uint k = 0; k < nParams; k++) {
      const double wp = work_param[k];
      const double grad = param_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 0; kk < nParams; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

    double new_grad_norm = work_grad.sqr_norm();

    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < nParams; k++) {

        //cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
        cur_grad_diff[k] += work_grad[k];
        cur_rho += cur_grad_diff[k] * cur_step[k];
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      if (cur_curv <= 0) {
        //this can happen as our function is not convex and we do not enforce part 2 of the Wolfe conditions
        // (this cannot be done by backtracking line search, see Algorithm 3.5 in [Nocedal & Wright])
        // Our solution is to simply restart L-BFGS now
        start_iter = iter;
      }

      rho[cur_l] = 1.0 / cur_rho;
    }

    search_direction = work_grad;

    if (iter > start_iter) {

      Math1D::Vector<double> alpha(L);

      const int cur_first_iter = std::max<int>(start_iter, iter - L);

      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter - 1; prev_iter >= cur_first_iter; prev_iter--) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double cur_alpha = 0.0;
        for (uint k = 0; k < nParams; k++) {
          cur_alpha += search_direction[k] * cur_step[k];
        }
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] -= cur_alpha * cur_grad_diff[k];
        }
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = 0.0;
        for (uint k = 0; k < nParams; k++) {
          beta += search_direction[k] * cur_grad_diff[k];
        }
        beta *= rho[prev_l];

        const double gamma = alpha[prev_l] - beta;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] += cur_step[k] * gamma;
        }
      }

    }
    else {
      search_direction *= 1.0 / sqrt(search_direction.sqr_norm());
    }

    Math1D::negate(search_direction);

    // d) line search

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double alpha = 1.0;
    double best_alpha = alpha;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      if (nIter > 1)
        alpha *= line_reduction_factor;

      double sqr_sum = 0.0;
      for (uint k = 0; k < nParams; k++) {
        hyp_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_param[k] * hyp_param[k];
      }
      //convert to the corresponding prob. distribution
      for (uint k = 0; k < nParams; k++) {
        hyp_param[k] = std::max(1e-10, hyp_param[k] * hyp_param[k] / sqr_sum);
      }

      const double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point

    if (best_energy >= energy - 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy)
                << ", final energy: " << energy << std::endl;
      std::cerr << "last squared gradient norm: " << new_grad_norm << std::endl;
      break;
    }

    energy = best_energy;

    uint cur_l = (iter % L);

    Math1D::Vector<double>& cur_step = step[cur_l];
    Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

    scale = 0.0;
    for (uint k = 0; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      scale += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    //calculate corresponding probability distribution (square the params and renormalize)
    for (uint k = 0; k < nParams; k++)
      sentence_start_parameters[k] =
        std::max(1e-10, work_param[k] * work_param[k] / scale);

    if ((iter % 5) == 0) {
      std::cerr << "#init m-step L-BFGS iter " << iter << ", energy: " << energy << std::endl;
      std::cerr << "sqr sum: " << scale << std::endl;
    }
  }
}

/************* from here on quadratic quasi newton methods **********************/

// void compute_psd_mat(const Storage1D<Math1D::Vector<double> >& grad_diff, const Storage1D<Math1D::Vector<double> >& step,
// double sigma, Math2D::Matrix<double>& psd_mat, const uint nParams, int start_iter, int iter)
// {
// //implements the forumla from [Byrd et al., Technical report page 19]

// const int L = grad_diff.size();
// start_iter = std::max(iter - L, start_iter);
// psd_mat.resize_dirty(nParams,nParams);
// psd_mat.set_constant(0.0);
// for (uint k = 0; k < nParams; k++)
// psd_mat(k,k) = sigma;

// Storage1D<Math1D::Vector<double> > vec_a(L);
// Storage1D<Math1D::Vector<double> > vec_b(L);
// for (int k = 0; k < L; k++) {
// vec_a[k].resize(nParams);
// vec_b[k].resize(nParams);
// }

// Math1D::Vector<double> temp(nParams);

// for (int k = start_iter; k < iter; k++) {
// int kk = k % L;
// const Math1D::Vector<double>& cur_grad_diff = grad_diff[kk]; //y in the paper
// const Math1D::Vector<double>& cur_step = step[kk]; // s in the paper

// //std::cerr << "k: " << k << ", cur_grad_diff: " << cur_grad_diff << std::endl;
// //std::cerr << "cur_step: " << cur_step << std::endl;

// //a) compute b_k
// Math1D::Vector<double>& b = vec_b[kk];
// b = cur_grad_diff;
// b *= 1.0 / sqrt(cur_grad_diff % cur_step);
// psd_mat += outer_product(b,b);
// assert(!isnan(b.sum()));

// Math1D::Vector<double>& a = vec_a[kk];
// a = cur_step;
// a *= sigma;

// for (int i = start_iter; i < k; i++) {
// temp = vec_b[i % L];
// temp *= vec_b[i % L] % cur_step;
// a += temp;

// temp = vec_a[i % L];
// temp *= vec_a[i % L] % cur_step;
// a -= temp;
// }

// a *= 1.0 / sqrt(cur_step % a);
// assert(!isnan(a.sum()));

// psd_mat -= outer_product(a,a);
// assert(!isnan(psd_mat.sum()));
// }
// }

void mul_with_psd_mat(const Storage1D<Math1D::Vector<double> >& grad_diff, const Storage1D<Math1D::Vector<double> >& step,
                      double sigma, const int nParams, int start_iter, int iter,
                      const Math1D::Vector<double>& to_mul, Math1D::Vector<double>& result)
{
  //implements the formula from [Byrd et al., Technical report page 19]

  const int L = grad_diff.size();
  start_iter = std::max(iter - L, start_iter);
  assert(step.size() == L);

  result = to_mul;
  result *= sigma;

  Storage1D<Math1D::Vector<double> > vec_a(L);
  Storage1D<Math1D::Vector<double> > vec_b(L);
  for (int k = 0; k < L; k++) {
    vec_a[k].resize(nParams);
    vec_b[k].resize(nParams);
  }

  Math1D::Vector<double> temp(nParams);

#if 1
  for (int k = start_iter; k < iter; k++) {
    int kk = k % L;
    const Math1D::Vector<double>& cur_grad_diff = grad_diff[kk]; //y in the paper
    const Math1D::Vector<double>& cur_step = step[kk]; // s in the paper

    //std::cerr << "k: " << k << ", cur_grad_diff: " << cur_grad_diff << std::endl;
    //std::cerr << "cur_step: " << cur_step << std::endl;

    //a) compute b_k
    Math1D::Vector<double>& b = vec_b[kk];
    b = cur_grad_diff;
    b *= 1.0 / sqrt(cur_grad_diff % cur_step);
    assert(!isnan(b.sum()));

    //psd_mat += outer_product(b,b);
    //result += outer_product(b,b) * to_mul;
    Math1D::Vector<double> temp = b;
    temp *= b % to_mul;
    result += temp;

    Math1D::Vector<double>& a = vec_a[kk];
    a = cur_step;
    a *= sigma;

    for (int i = start_iter; i < k; i++) {
      temp = vec_b[i % L];
      temp *= vec_b[i % L] % cur_step;
      a += temp;

      temp = vec_a[i % L];
      temp *= vec_a[i % L] % cur_step;
      a -= temp;
    }

    a *= 1.0 / sqrt(cur_step % a);
    assert(!isnan(a.sum()));

    //psd_mat -= outer_product(a,a);
    //assert(!isnan(psd_mat.sum()));
    //result -= outer_product(a,a) * to_mul;
    temp = a;
    temp *= a % to_mul;
    result -= temp;
  }
#endif
}

double quadratic_plbfgs_energy(double energy_offs, double sigma, int iter, int start_iter,
                               const Math1D::Vector<double>& cur_params, Math1D::Vector<double>& new_params,
                               const Math1D::Vector<double>& super_gradient,
                               const Storage1D<Math1D::Vector<double> >& grad_diff,
                               const Storage1D<Math1D::Vector<double> >& step)
{
  const int nParams = cur_params.size();
  const int L = grad_diff.size();
  assert(step.size() == L);
  double energy = energy_offs;
  Math1D::Vector<double> param_diff = new_params - cur_params;
  energy += super_gradient % param_diff;
  start_iter = std::max(iter - L, start_iter);

  //Math2D::Matrix<double> psd_mat(nParams, nParams, 0.0);
  //compute_psd_mat(grad_diff, step, sigma, psd_mat, nParams, start_iter, iter);

  Math1D::Vector<double> result(nParams);

  mul_with_psd_mat(grad_diff, step, sigma, nParams, start_iter, iter, param_diff, result);
  //energy += 0.5 * (param_diff % (psd_mat * param_diff));
  energy += 0.5 * (param_diff % result);

  return energy;
}

void solve_quadratic_plbfgs_problem(const Math1D::Vector<double>& cur_params, Math1D::Vector<double>& new_params,
                                    double energy_offs, double sigma, int iter, int start_iter,
                                    const Math1D::Vector<double>& super_gradient,
                                    const Storage1D<Math1D::Vector<double> >& grad_diff,
                                    const Storage1D<Math1D::Vector<double> >& step)
{
  const int L = grad_diff.size();
  assert(step.size() == L);
  const uint nParams = cur_params.size();
  start_iter = std::max(iter - L, start_iter);

  std::cerr << "**** solve_quadratic_plbfgs_problem " << std::endl;

  Math1D::Vector<double> temp_params(nParams);
  Math1D::Vector<double> hyp_params(nParams);

  //Math2D::Matrix<double> psd_mat(nParams, nParams, 0.0);
  //compute_psd_mat(grad_diff, step, sigma, psd_mat, nParams, start_iter, iter);

  new_params = cur_params;
  double energy = energy_offs;
  std::cerr << "start energy: " << energy << std::endl;
  double check = quadratic_plbfgs_energy(energy_offs, sigma, iter, start_iter, cur_params, new_params,
                                         super_gradient, grad_diff, step);
  std::cerr << "check: " << check << std::endl;
  assert(energy == check);

  for (int sub_iter = 1; sub_iter <= 10; sub_iter++) {

    std::cerr << "-----sub-iter " << sub_iter << ", energy " << energy << std::endl;

    Math1D::Vector<double> cur_diff = new_params - cur_params;
    Math1D::Vector<double> gradient = super_gradient;
    Math1D::Vector<double> temp;
    mul_with_psd_mat(grad_diff, step, sigma, nParams, start_iter, iter, cur_diff, temp);
    gradient += temp;
    //Math1D::Vector<double> gradient = super_gradient + psd_mat * cur_diff;

    double alpha = 1.0;
    for (uint k = 0; k < nParams; k++)
      temp_params[k] = new_params[k] - alpha * gradient[k];
    projection_on_simplex(temp_params, hmm_min_param_entry);

    hyp_params = temp_params - new_params;
    if (hyp_params.sqr_norm() < 1e-5) {
      std::cerr << "CUTOFF for inner problem because the points are nearly the same" << std::endl;
      break;
    }

    double best_energy = energy;
    double best_alpha = 1.0;
    alpha = 1.0;

    while (alpha > 1e-12) {

      alpha *= 0.75;

      const double neg_alpha = 1.0 - alpha;
      for (uint k = 0; k < nParams; k++)
        hyp_params[k] = alpha * temp_params[k] + neg_alpha * new_params[k];

      double hyp_energy = quadratic_plbfgs_energy(energy_offs, sigma, iter, start_iter, cur_params, hyp_params,
                          super_gradient, grad_diff, step);


      //std::cerr << "-----alpha: " << alpha << ", hyp energy: " << hyp_energy << std::endl;

      if (hyp_energy < best_energy) {
        best_alpha = alpha;
        best_energy = hyp_energy;
      }
      else if (best_energy < energy)
        break;
    }

    if (alpha <= 1e-12) {
      std::cerr << "----CUTOFF of inner problem because stepsize was <= 1e-12" << std::endl;
      break;
    }

    const double neg_best_alpha = 1.0 - best_alpha;
    for (uint k = 0; k < nParams; k++)
      new_params[k] = best_alpha * temp_params[k] + neg_best_alpha * new_params[k];
    energy = best_energy;
  }
}

void ehmm_init_m_step_projected_lbfgs(const InitialAlignmentProbability& init_acount, Math1D::Vector<double>& init_params, uint nIter,
                                      int L)
{
  std::cerr << "********** ehmm_init_m_step_projected_lbfgs " << std::endl;
  projection_on_simplex(init_params, hmm_min_param_entry);

  assert(L >= 1);
  const int nParams = init_params.size();

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);

  for (int l=0; l < L; l++) {

    grad_diff[l].resize(nParams);
    step[l].resize(nParams);
  }

  Math1D::Vector<double> m_init_grad = init_params;
  Math1D::Vector<double> new_init_params = init_params;
  Math1D::Vector<double> hyp_init_params = init_params;

  double energy = ehmm_init_m_step_energy(init_acount, init_params);

  double line_reduction_factor = 0.75;

  {
    //try to find a better starting point
    Math1D::Vector<double> init_count(init_params.size(), 0.0);

    for (uint I = 1; I <= init_acount.size(); I++) {

      if (init_acount[I - 1].size() != 0) {
        for (uint i = 0; i < I; i++) {
          init_count[i] += init_acount[I - 1][i];
        }
      }
    }

    const double sum = init_count.sum();

    for (uint k=0; k < init_count.size(); k++)
      init_count[k] = std::max(hmm_min_param_entry, init_count[k] / sum);

    double hyp_energy = ehmm_init_m_step_energy(init_acount, init_count);

    if (hyp_energy < energy) {
      init_params = init_count;
      energy = hyp_energy;
    }
  }

  std::cerr << "start m-energy: " << energy << std::endl;

  uint start_iter = 1;
  for (uint iter = 1; iter <= nIter; iter++) {

    m_init_grad.set_constant(0.0);

    //calculate gradient
    for (uint I = 0; I < init_acount.size(); I++) {

      if (init_acount[I].size() > 0) {

        double non_zero_sum = 0.0;
        for (uint i = 0; i <= I; i++)
          non_zero_sum += init_params[i];

        double count_sum = 0.0;
        for (uint i = 0; i <= I; i++) {
          count_sum += init_acount[I][i];

          double cur_param = init_params[i];

          m_init_grad[i] -= init_acount[I][i] / cur_param;
        }

        for (uint i = 0; i <= I; i++) {
          m_init_grad[i] += count_sum / non_zero_sum;
        }
      }
    }

    double sqr_norm = m_init_grad.sqr_norm();
    if (sqr_norm < 1e-5) {
      std::cerr << "cutoff after " << iter << " iterations because gradient was near zero: " << sqr_norm << std::endl;
      break;
    }

    double cur_curv = 0.0;
    if (iter > 1) {
      //update grad_diff and cur_rho
      uint cur_l = (iter-1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k=0; k < nParams; k++) {

        //cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
        cur_grad_diff[k] += m_init_grad[k];
        cur_rho += cur_grad_diff[k] * cur_step[k];
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();
      std::cerr << "cur_curv: " << cur_curv << std::endl;

      if (cur_curv <= 0.0) {
        //this can happen if the function is not (strictly) convex
        start_iter = iter;
      }
    }

    std::cerr << "iter: " << iter << ", start_iter: " << start_iter << std::endl;
    if (iter == start_iter) {

      //go in search direction and project
      Math1D::go_in_neg_direction(new_init_params, init_params, m_init_grad, 1.0 / sqrt(sqr_norm));
      projection_on_simplex(new_init_params, hmm_min_param_entry);
    }
    else {
      //solve the quadratic problem
      double sigma = 1.0 / sqrt(sqr_norm);
      solve_quadratic_plbfgs_problem(init_params, new_init_params, energy, sigma, iter, start_iter, m_init_grad,
                                     grad_diff, step);
    }

    hyp_init_params = new_init_params - init_params;
    if (hyp_init_params.sqr_norm() <= 1e-15) {
      std::cerr << "CUTOFF after " << iter << " iterations because projected point was only a distance of "
                << hyp_init_params.sqr_norm() << " away" << std::endl;
      break;
    }


    //find step-size
    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nIter = 0;

    bool decreasing = true;

    std::cerr << "energy to beat: " << energy << std::endl;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy < energy) {
        break;
      }

      //std::cerr << "multiplying lambda with " << line_reduction_factor << std::endl;
      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      //for (uint k = 0; k < init_params.size(); k++)
      //  hyp_init_params.direct_access(k) = lambda * new_init_params.direct_access(k) + neg_lambda * init_params.direct_access(k);
      Math1D::assign_weighted_combination(hyp_init_params, lambda, new_init_params, neg_lambda, init_params);

      double hyp_energy = ehmm_init_m_step_energy(init_acount, hyp_init_params);

      std::cerr << "lambda: " << lambda << ", energy: " << hyp_energy << std::endl;

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    //std::cerr << "----nIter: " << nIter << std::endl;
    if (nIter > 5) {
      line_reduction_factor *= 0.9;
      //std::cerr << "new line_reduction_factor: " << line_reduction_factor << std::endl;
    }

    if (nIter > 25 || best_lambda < 1e-12 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF because of too many trials" << std::endl;
      break;
    }

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    //c) update the variables and the step vectors
    if (best_energy <= energy) {

      uint cur_l = (iter % L);

      Math1D::Vector<double>& cur_step = step[cur_l];
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

      for (uint k=0; k < nParams; k++) {

        double new_val = best_lambda * new_init_params[k] + neg_best_lambda * init_params[k];
        double step = new_val - init_params[k];
        cur_step[k] = step;
        init_params[k] = new_val;

        //prepare for the next iteration
        cur_grad_diff[k] = -m_init_grad[k];
      }

      energy = best_energy;
    }
    else {
      INTERNAL_ERROR << " failed to get descent" << std::endl;
      exit(1);
    }

    if ((iter % 1) == 0)
      std::cerr << "init m-step plbfgs-iter #" << iter << ", energy: " << energy << std::endl;
  }
}