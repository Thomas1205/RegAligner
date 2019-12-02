/*** written by Thomas Schoenemann as a private person without employment, October 2009
 *** and later by Thomas Schoenemann as employee of Lund University, 2010 ***/

#include "ibm2_training.hh"

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "matrix.hh"
#include "trimatrix.hh"

#include "training_common.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"
#include "projection.hh"

IBM2Options::IBM2Options(uint nSourceWords, uint nTargetWords, std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType > > >& sure_ref_alignments,
                         std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments):
  nIterations_(5), smoothed_l0_(false), l0_beta_(1.0), print_energy_(true),
  nSourceWords_(nSourceWords), nTargetWords_(nTargetWords), dict_m_step_iter_(45),
  sure_ref_alignments_(sure_ref_alignments), possible_ref_alignments_(possible_ref_alignments)
{
}


double ibm2_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                       const IBM2AlignmentModel& align_model, const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords)
{
  //std::cerr << "calculating IBM 2 perplexity" << std::endl;

  double sum = 0.0;

  const size_t nSentences = target.size();
  assert(slookup.size() == nSentences);

  SingleLookupTable aux_lookup;

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    uint k = 0;
    for (; k < align_model[curI].size(); k++) {
      if (align_model[curI][k].yDim() == curJ)
        break;
    }

    assert(k < align_model[curI].size());
    const Math2D::Matrix<double>& cur_align_model = align_model[curI][k];

    for (uint j = 0; j < curJ; j++) {

      const uint s_idx = cur_source[j];
      double cur_sum = cur_align_model(0, j) * dict[0][s_idx - 1];

      for (uint i = 0; i < curI; i++) {
        const uint t_idx = cur_target[i];
        cur_sum += cur_align_model(i + 1, j) * dict[t_idx][cur_lookup(j, i)];
      }
      sum -= std::log(cur_sum);
    }
  }

  return sum / nSentences;
}

double ibm2_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                   const Storage1D<Math1D::Vector<uint> >& target, const IBM2AlignmentModel& align_model,
                   const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords,
                   const floatSingleWordDictionary& prior_weight, double l0_beta, bool smoothed_l0, double dict_weight_sum)
{
  double energy = 0.0;

  if (dict_weight_sum != 0.0) {
    for (uint i = 0; i < dict.size(); i++) {

      const uint size = dict[i].size();

      for (uint k = 0; k < size; k++) {
        if (smoothed_l0)
          energy += prior_weight[i][k] * prob_penalty(dict[i][k], l0_beta);
        else
          energy += prior_weight[i][k] * dict[i][k];
      }
    }

    energy /= target.size();      //since the perplexity is also divided by that amount
  }

  energy += ibm2_perplexity(source, slookup, target, align_model, dict, wcooc, nSourceWords);

  return energy;
}


void train_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, uint nSourceWords, uint nTargetWords,
                IBM2AlignmentModel& alignment_model, SingleWordDictionary& dict, uint nIterations,
                std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                const floatSingleWordDictionary& prior_weight, double l0_beta, bool smoothed_l0, uint dict_m_step_iter)
{
  std::cerr << "starting IBM 2 training" << std::endl;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < nTargetWords; i++) {
    dict_weight_sum += fabs(prior_weight[i].sum());
  }

  assert(wcooc.size() == nTargetWords);
  //NOTE: the dicitionary is assumed to be initialized

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  //initialize alignment model
  alignment_model.resize_dirty(lcooc.size());
  for (uint I = 0; I < lcooc.size(); I++) {

    alignment_model[I].resize_dirty(lcooc[I].size());

    for (uint k = 0; k < lcooc[I].size(); k++) {
      uint J = lcooc[I][k];

      alignment_model[I][k].resize_dirty(I + 1, J);
      alignment_model[I][k].set_constant(1.0 / (I + 1));
    }
  }

  SingleLookupTable aux_lookup;

  SingleWordDictionary fwcount(nTargetWords,MAKENAME(fwcount));
  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict[i].size());
  }

  IBM2AlignmentModel facount(alignment_model.size(), MAKENAME(facount));
  for (uint I = 0; I < lcooc.size(); I++) {
    uint cur_length = lcooc[I].size();
    facount[I].resize_dirty(cur_length);

    for (uint k = 0; k < lcooc[I].size(); k++) {
      uint J = lcooc[I][k];

      facount[I][k].resize_dirty(I + 1, J);
    }
  }

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting IBM 2 iteration #" << iter << std::endl;

    //set counts to 0
    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
    }
    for (uint I = 0; I < lcooc.size(); I++) {
      uint cur_length = lcooc[I].size();

      for (uint k = 0; k < cur_length; k++) {
        facount[I][k].set_constant(0.0);
      }
    }

    for (size_t s = 0; s < nSentences; s++) {

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      uint k = 0;
      for (; k < alignment_model[curI].size(); k++) {
        if (alignment_model[curI][k].yDim() == curJ)
          break;
      }

      assert(k < alignment_model[curI].size());
      const Math2D::Matrix<double>& cur_align_model = alignment_model[curI][k];
      Math2D::Matrix<double>& cur_facount = facount[curI][k];

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        double coeff = dict[0][s_idx - 1] * cur_align_model(0, j);

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          coeff += dict[t_idx][cur_lookup(j, i)] * cur_align_model(i + 1, j);
        }

        coeff = 1.0 / coeff;
        assert(!isnan(coeff));

        double addon;
        addon = coeff * dict[0][s_idx - 1] * cur_align_model(0, j);

        fwcount[0][s_idx - 1] += addon;
        cur_facount(j, 0) += addon;

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          const uint l = cur_lookup(j, i);

          addon = coeff * dict[t_idx][l] * cur_align_model(i + 1, j);

          //update dict
          fwcount[t_idx][l] += addon;

          //update alignment
          cur_facount(j, i + 1) += addon;
        }
      }
    }

    update_dict_from_counts(fwcount, prior_weight, dict_weight_sum, smoothed_l0, l0_beta, dict_m_step_iter, dict,
                            ibm1_min_dict_entry, MSSolvePGD);

    //compute new alignment model from normalized fractional counts
    for (uint I = 0; I < alignment_model.size(); I++) {

      for (uint k = 0; k < alignment_model[I].size(); k++) {
        uint J = alignment_model[I][k].yDim();
        assert(alignment_model[I][k].xDim() == (I + 1));

        for (uint j = 0; j < J; j++) {

          double sum = facount[I][k].row_sum(j);

          if (sum > 1e-307) {
            sum = 1.0 / sum;
            assert(!isnan(sum));

            for (uint i = 0; i <= I; i++)
              alignment_model[I][k](i, j) = std::max(ibm2_min_align_param, sum * facount[I][k](i, j));
          }
          else {
            std::cerr << "WARNING : did not update alignment prob because sum is " << sum << std::endl;
          }
        }
      }
    }

    std::cerr << "IBM 2 perplexity after iteration #" << iter << ": "
              << ibm2_energy(source, slookup, target, alignment_model, dict, wcooc, nSourceWords, prior_weight, l0_beta, smoothed_l0, dict_weight_sum)
              << std::endl;

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = possible_ref_alignments.begin();
           it != possible_ref_alignments.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;

        const uint curJ = source[s].size();
        const uint curI = target[s].size();

        uint k = 0;
        for (; k < alignment_model[curI].size(); k++) {
          if (alignment_model[curI][k].yDim() == curJ)
            break;
        }

        assert(k < alignment_model[curI].size());
        const Math2D::Matrix<double>& cur_align_model = alignment_model[curI][k];

        const SingleLookupTable& cur_lookup = get_wordlookup(source[s], target[s], wcooc, nSourceWords, slookup[s], aux_lookup);

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm2_viterbi_alignment(source[s], cur_lookup, target[s], dict, cur_align_model, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, sure_ref_alignments[s + 1], possible_ref_alignments[s + 1]);
        sum_fmeasure += f_measure(viterbi_alignment, sure_ref_alignments[s + 1], possible_ref_alignments[s + 1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, sure_ref_alignments[s + 1], possible_ref_alignments[s + 1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM2 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM2 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM2 Viterbi-DAE/S after iteration #" << iter << ": " << sum_fmeasure << std::endl;
    }
  }
}

/******************************************************** reduced IBM-2 ******************************************************************/

void par2nonpar_reduced_ibm2alignment_model(const Math2D::Matrix<double>& align_param, const Math1D::Vector<double>& source_fert,
    ReducedIBM2AlignmentModel& alignment_model, IBM23ParametricMode par_mode, uint offset, bool deficient)
{
  for (uint k = 0; k < alignment_model.size(); k++) {

    for (uint j = 0; j < alignment_model[k].yDim(); j++) {

      double sum = 0.0;

      if (par_mode == IBM23ParByPosition) {

        if (deficient)
          sum = 1.0;
        else {
          for (uint i = 1; i < alignment_model[k].xDim(); i++)
            sum += align_param(i - 1, j);
        }

        if (sum > 1e-305) {
          const double inv_sum = 1.0 / sum;
          alignment_model[k](0, j) = source_fert[0];
          for (uint i = 1; i < alignment_model[k].xDim(); i++) {
            alignment_model[k](i, j) = source_fert[1] * align_param(i - 1, j) * inv_sum;
          }
        }
      }
      else {

        if (deficient)
          sum = 1.0;
        else {
          for (uint i = 1; i < alignment_model[k].xDim(); i++)
            sum += align_param(offset + j - (i - 1), 0);
        }

        if (sum > 1e-305) {
          const double inv_sum = 1.0 / sum;
          alignment_model[k](0, j) = source_fert[0];
          for (uint i = 1; i < alignment_model[k].xDim(); i++) {
            alignment_model[k](i, j) = source_fert[1] * align_param(offset + j - (i - 1), 0) * inv_sum;
          }
        }
      }
    }
  }
}

void nonpar2par_reduced_ibm2alignment_model(Math2D::Matrix<double>& align_param, const ReducedIBM2AlignmentModel& align_model)
{
  align_param.set_constant(0.0);

  for (uint I = 0; I < align_model.size(); I++) {

    for (uint y = 0; y < align_model[I].yDim(); y++)
      for (uint i = 1; i < align_model[I].xDim(); i++)
        align_param(i-1, y) += align_model[I](i, y);
  }

  for (uint y = 0; y < align_param.yDim(); y++) {

    double sum = align_param.row_sum(y);
    assert(sum > 1e-305);

    for (uint x = 0; x < align_param.xDim(); x++)
      align_param(x, y) /= sum;
  }
}

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

void reducedibm2_par_m_step(Math2D::Matrix<double>& align_param, const ReducedIBM2AlignmentModel& acount, uint j, uint nIter, bool deficient)
{
  //std::cerr << "reducedibm2_par_m_step" << std::endl;

  const uint xDim = align_param.xDim();

  Math1D::Vector<double> cur_param(xDim);
  for (uint k = 0; k < xDim; k++)
    cur_param[k] = std::max(ibm2_min_align_param, align_param(k, j));

  Math1D::Vector<double> hyp_param(xDim);
  Math1D::Vector<double> new_param(xDim);
  Math1D::Vector<double> grad(xDim);

  Math1D::Vector<double> singleton_count(xDim, 0.0);
  Math1D::Vector<double> span_count(xDim, 0.0);

  for (uint k = 0; k < acount.size(); k++) {

    const Math2D::Matrix<double>& cur_acount = acount[k];
    if (cur_acount.yDim() > j) {

      uint curI = cur_acount.xDim()-1;
      for (uint i=1; i <= curI; i++) {
        singleton_count[i - 1] += cur_acount(i,j);
        span_count[curI - 1] += cur_acount(i,j);
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
      std::cerr << "switching to normalized counts: " << hyp_energy << " instead of " << energy << std::endl;

      energy = hyp_energy;
      cur_param = hyp_param;
    }
  }

  if (deficient) {

    for (uint k = 0; k < xDim; k++)
      align_param(k, j) = cur_param[k];

    return;
  }

  double line_reduction_factor = 0.5;
  const double alpha = 100.0;

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

    for (uint i = 0; i < xDim; i++)
      new_param[i] = cur_param[i] - alpha * grad[i];

    projection_on_simplex(new_param.direct_access(), xDim, ibm2_min_align_param);

    /**** find a suitable stepsize ****/

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
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

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    if (nIter > 15 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint i = 0; i < xDim; i++)
      cur_param[i] = best_lambda * new_param[i] + neg_best_lambda * cur_param[i];
  }

  for (uint k = 0; k < xDim; k++)
    align_param(k, j) = cur_param[k];
}

double reducedibm2_diffpar_m_step_energy(const Math2D::Matrix<double>& align_param, const Math1D::Vector<double>& singleton_count,
    const Math2D::TriMatrix<double>& span_count)
{
  const uint xDim = align_param.xDim();

  double energy = 0.0;

  for (uint x = 0; x < xDim; x++)
    energy -= singleton_count[x] * std::log(align_param(x, 0) );

  for (uint x_start = 0; x_start < xDim; x_start++) {

    double param_sum = 0.0;

    for (uint x_end = x_start; x_end < xDim; x_end++) {

      param_sum += align_param(x_end, 0);
      const double count = span_count(x_start, x_end);

      if (count != 0.0)
        energy += count * std::log(param_sum);
    }
  }

  return energy;
}

void reducedibm2_diffpar_m_step(Math2D::Matrix<double>& align_param, const ReducedIBM2AlignmentModel& acount, uint offset,
                                uint nIter, bool deficient)
{
  const uint xDim = align_param.xDim();

  Math1D::Vector<double> singleton_count(xDim, 0.0);
  Math2D::TriMatrix<double> span_count(xDim, 0.0);

  for (uint k = 0; k < acount.size(); k++) {

    const Math2D::Matrix<double>& cur_acount = acount[k];
    for (uint j = 0; j < cur_acount.yDim(); j++) {

      uint curI = cur_acount.xDim()-1;
      for (uint i=1; i <= curI; i++) {
        singleton_count[offset + j - (i - 1)] += cur_acount(i, j);
        span_count(offset + j - (curI - 1), offset + j) += cur_acount(i, j);
      }
    }
  }

  Math2D::Matrix<double> hyp_param(xDim, 1);
  Math1D::Vector<double> new_param(xDim);
  Math1D::Vector<double> grad(xDim);

  double energy = reducedibm2_diffpar_m_step_energy(align_param, singleton_count, span_count);

  {
    //test start point
    double sum = singleton_count.sum();
    for (uint x = 0; x < xDim; x++)
      hyp_param(x, 0) = singleton_count[x] / sum;

    double hyp_energy = reducedibm2_diffpar_m_step_energy(hyp_param, singleton_count, span_count);

    if (deficient || hyp_energy < energy) {

      std::cerr << "switching to normalized counts" << std::endl;
      align_param = hyp_param;
      energy = hyp_energy;
    }
  }

  if (deficient)
    return;

  double line_reduction_factor = 0.5;
  const double alpha = 100.0;

  for (uint iter = 1; iter <= nIter; iter++) {
    if ((iter % 15) == 0)
      std::cerr << "iter " << iter << ", energy: " << energy << std::endl;

    /***** compute gradient *****/
    for (uint i = 0; i < xDim; i++)
      grad[i] = -singleton_count[i] / align_param(i, 0);

    for (uint x_start = 0; x_start < xDim; x_start++) {

      double param_sum = 0.0;

      for (uint x_end = x_start; x_end < xDim; x_end++) {

        param_sum += align_param(x_end, 0);
        const double count = span_count(x_start, x_end);

        if (count != 0.0) {
          const double cur_grad = count / param_sum;
          for (uint x = x_start; x <= x_end; x++)
            grad[x] += cur_grad;
        }
      }
    }

    /**** go in negative gradient direction and reproject ****/

    for (uint i = 0; i < xDim; i++)
      new_param[i] = align_param(i, 0) - alpha * grad[i];

    projection_on_simplex(new_param.direct_access(), xDim, ibm2_min_align_param);

    /**** find a suitable stepsize ****/

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      lambda *= line_reduction_factor;

      double neg_lambda = 1.0 - lambda;

      for (uint i = 0; i < xDim; i++)
        hyp_param(i, 0) = lambda * new_param[i] + neg_lambda * align_param(i, 0);

      double new_energy = reducedibm2_diffpar_m_step_energy(hyp_param, singleton_count, span_count);

      if (new_energy < best_energy) {
        best_energy = new_energy;
        best_lambda = lambda;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    if (nIter > 15 || fabs(energy - best_energy) < 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    energy = best_energy;

    //set new values
    double neg_best_lambda = 1.0 - best_lambda;

    for (uint i = 0; i < xDim; i++)
      align_param(i, 0) = best_lambda * new_param[i] + neg_best_lambda * align_param(i, 0);
  }
}

double reduced_ibm2_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                               const Storage1D<Math1D::Vector<uint> >& target, const ReducedIBM2AlignmentModel& align_model,
                               const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords)
{
  //std::cerr << "calculating ReducedIBM 2 perplexity" << std::endl;

  double sum = 0.0;

  const size_t nSentences = target.size();
  assert(slookup.size() == nSentences);

  SingleLookupTable aux_lookup;

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const Math2D::Matrix<double>& cur_align_model = align_model[curI];

    for (uint j = 0; j < curJ; j++) {

      const uint s_idx = cur_source[j];
      double cur_sum = cur_align_model(0, j) * dict[0][s_idx - 1];

      for (uint i = 0; i < curI; i++) {
        const uint t_idx = cur_target[i];
        cur_sum += cur_align_model(i + 1, j) * dict[t_idx][cur_lookup(j, i)];
      }
      sum -= std::log(cur_sum);
    }
  }

  return sum / nSentences;
}

double reduced_ibm2_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                           const Storage1D<Math1D::Vector<uint> >& target, const ReducedIBM2AlignmentModel& align_model,
                           const SingleWordDictionary& dict, const CooccuringWordsType& wcooc, uint nSourceWords,
                           const floatSingleWordDictionary& prior_weight, double l0_beta, bool smoothed_l0, double dict_weight_sum)
{
  double energy = 0.0;

  if (dict_weight_sum != 0.0) {
    for (uint i = 0; i < dict.size(); i++) {

      const uint size = dict[i].size();

      for (uint k = 0; k < size; k++) {
        if (smoothed_l0)
          energy += prior_weight[i][k] * prob_penalty(dict[i][k], l0_beta);
        else
          energy += prior_weight[i][k] * dict[i][k];
      }
    }

    energy /= target.size();      //since the perplexity is also divided by that amount
  }

  energy += reduced_ibm2_perplexity(source, slookup, target, align_model, dict, wcooc, nSourceWords);

  return energy;
}

void train_reduced_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2AlignmentModel& alignment_model,
                        Math2D::Matrix<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                        const IBM2Options& options, const floatSingleWordDictionary& prior_weight)
{
  const uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  const IBM23ParametricMode par_mode = options.ibm2_mode_;

  if (source_fert.size() != 2)
    source_fert.resize(2);
  source_fert[0] = options.p0_;
  source_fert[1] = 1.0 - options.p0_;

  std::cerr << "starting reduced IBM 2 training" << std::endl;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += fabs(prior_weight[i].sum());
  }

  assert(wcooc.size() == options.nTargetWords_);
  //NOTE: the dicitionary is assumed to be initialized

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  //initialize alignment model
  alignment_model.resize_dirty(lcooc.size());
  for (uint I = 0; I < lcooc.size(); I++) {

    if (lcooc[I].size() > 0) {

      uint maxJ = lcooc[I].max();

      if (maxJ > 0) {
        alignment_model[I].resize_dirty(I + 1, maxJ);
        alignment_model[I].set_constant(1.0 / (I + 1));
      }
    }
  }

  uint maxJ = 0;
  uint maxI = 0;
  for (size_t s = 0; s < nSentences; s++) {
    maxJ = std::max<uint>(maxJ, source[s].size());
    maxI = std::max<uint>(maxI, target[s].size());
  }

  if (par_mode != IBM23ParByDifference)
    align_param.resize(maxI, maxJ, 1.0);
  else
    align_param.resize(maxJ + maxI - 1, 1, 1.0);

  SingleLookupTable aux_lookup;

  //TODO: estimate first alignment model from IBM1 dictionary

  SingleWordDictionary fwcount(options.nTargetWords_, MAKENAME(fwcount));
  for (uint i = 0; i < options.nTargetWords_; i++) {
    fwcount[i].resize(dict[i].size());
  }

  ReducedIBM2AlignmentModel facount(alignment_model.size(), MAKENAME(facount));
  for (uint I = 0; I < lcooc.size(); I++) {
    facount[I].resize_dirty(alignment_model[I].xDim(),alignment_model[I].yDim());
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting reduced IBM 2 iteration #" << iter << std::endl;

    //set counts to 0
    for (uint i = 0; i < options.nTargetWords_; i++) {
      fwcount[i].set_constant(0.0);
    }
    for (uint I = 0; I < lcooc.size(); I++) {
      facount[I].set_constant(0.0);
    }

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      const Math2D::Matrix<double>& cur_align_model = alignment_model[curI];
      Math2D::Matrix<double>& cur_facount = facount[curI];

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

      assert(cur_align_model.yDim() >= curJ);
      assert(cur_facount.yDim() >= curJ);

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        double coeff = dict[0][s_idx - 1] * cur_align_model(0, j);

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          coeff += dict[t_idx][cur_lookup(j, i)] * cur_align_model(i + 1, j);
        }

        coeff = 1.0 / coeff;
        assert(!isnan(coeff));

        double addon;
        addon = coeff * dict[0][s_idx - 1] * cur_align_model(0, j);

        fwcount[0][s_idx - 1] += addon;
        cur_facount(0, j) += addon;

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          const uint l = cur_lookup(j, i);

          addon = coeff * dict[t_idx][l] * cur_align_model(i + 1, j);

          //update dict
          fwcount[t_idx][l] += addon;

          //update alignment
          cur_facount(i + 1, j) += addon;
        }
      }
    }

    update_dict_from_counts(fwcount, prior_weight, dict_weight_sum, smoothed_l0, l0_beta, options.dict_m_step_iter_, dict,
                            ibm1_min_dict_entry, options.unconstrained_m_step_);

    //compute new alignment model from normalized fractional counts
    if (par_mode == IBM23Nonpar) {

      for (uint I = 0; I < alignment_model.size(); I++) {

        uint J = alignment_model[I].yDim();
        if (J > 0) {
          assert(alignment_model[I].xDim() == (I + 1));

          for (uint j = 0; j < J; j++) {

            double sum = facount[I].row_sum(j);

            if (sum > 1e-305) {
              sum = 1.0 / sum;

              for (uint i = 0; i <= I; i++)
                alignment_model[I](i, j) = std::max(ibm2_min_align_param, sum * facount[I](i, j));
            }
          }
        }
      }
    }
    else {

      if (par_mode == IBM23ParByPosition) {

        for (uint j = 0; j < align_param.yDim(); j++) {
          reducedibm2_par_m_step(align_param, facount, j, options.align_m_step_iter_, options.deficient_);
        }
      }
      else {
        reducedibm2_diffpar_m_step(align_param, facount, maxI - 1, options.align_m_step_iter_, options.deficient_);
      }

      par2nonpar_reduced_ibm2alignment_model(align_param, source_fert, alignment_model, par_mode, maxI - 1);
    }

    if (options.print_energy_) {
      std::cerr << "reduced IBM 2 energy after iteration #" << iter << ": "
                << reduced_ibm2_energy(source, slookup, target, alignment_model, dict, wcooc, options.nSourceWords_, prior_weight,
                                       l0_beta, smoothed_l0, dict_weight_sum)
                << std::endl;
    }

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];

        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

        nContributors++;

        const uint curI = cur_target.size();
        const Math2D::Matrix<double>& cur_align_model = alignment_model[curI];

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm2_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
        sum_fmeasure += f_measure(viterbi_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm2_postdec_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
        sum_postdec_fmeasure += f_measure(postdec_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### ReducedIBM2 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### ReducedIBM2 Postdec-AER after iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### ReducedIBM2 Postdec-fmeasure after iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### ReducedIBM2 Postdec-DAE/S after iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  }

  if (par_mode == IBM23Nonpar)
    nonpar2par_reduced_ibm2alignment_model(align_param, alignment_model);
}

void reduced_ibm2_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                                   const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2AlignmentModel& alignment_model,
                                   Math2D::Matrix<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                                   const IBM2Options& options, const floatSingleWordDictionary& prior_weight, const Math1D::Vector<double>& xlogx_table)
{
  const uint nIter = options.nIterations_;
  const IBM23ParametricMode par_mode = options.ibm2_mode_;

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  if (source_fert.size() != 2)
    source_fert.resize(2);
  source_fert[0] = options.p0_;
  source_fert[1] = 1.0 - options.p0_;

  //initialize alignment model
  alignment_model.resize_dirty(lcooc.size());
  ReducedIBM2AlignmentModel acount(lcooc.size(), MAKENAME(acount));

  for (uint I = 0; I < lcooc.size(); I++) {

    if (lcooc[I].size() > 0) {

      uint maxJ = lcooc[I].max();

      if (maxJ > 0) {
        alignment_model[I].resize_dirty(I + 1, maxJ);
        alignment_model[I].set_constant(1.0 / (I + 1));
        acount[I].resize_dirty(I + 1, maxJ);
      }
    }
  }

  uint maxJ = 0;
  uint maxI = 0;
  for (size_t s = 0; s < nSentences; s++) {
    maxJ = std::max<uint>(maxJ, source[s].size());
    maxI = std::max<uint>(maxI, target[s].size());
  }
  if (par_mode != IBM23ParByDifference)
    align_param.resize(maxI,maxJ, 1.0);
  else
    align_param.resize(maxJ + maxI - 1, 1, 1.0);

  SingleLookupTable aux_lookup;

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    viterbi_alignment[s].resize(cur_source.size());
  }

  //fractional counts used for EM-iterations
  NamedStorage1D<Math1D::Vector<double> > dcount(options.nTargetWords_, MAKENAME(dcount));

  for (uint i = 0; i < options.nTargetWords_; i++) {
    dcount[i].resize(dict[i].size());
    dcount[i].set_constant(0);
  }

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting ReducedIBM2 Viterbi iteration " << iter << std::endl;

    for (uint i = 0; i < options.nTargetWords_; i++)
      dcount[i].set_constant(0);

    for (uint I = 0; I < acount.size(); I++)
      acount[I].set_constant(0.0);

    double sum = 0.0;

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

      const Math2D::Matrix<double>& cur_align_model = alignment_model[curI];
      Math2D::Matrix<double>& cur_acount = acount[curI];

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = source[s][j];

        uint arg_max = MAX_UINT;

        double max = dict[0][s_idx - 1] * cur_align_model(0, j);
        arg_max = 0;

        for (uint i = 0; i < curI; i++) {

          double hyp = dict[cur_target[i]][cur_lookup(j, i)] * cur_align_model(i + 1, j);

          //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;

          if (hyp > max) {
            max = hyp;
            arg_max = i + 1;
          }
        }

        //std::cerr << "arg_min: " << arg_min << std::endl;

        sum -= std::log(max);

        viterbi_alignment[s][j] = arg_max;
        cur_acount(arg_max, j)++;

        if (arg_max == 0)
          dcount[0][s_idx - 1]++;
        else
          dcount[cur_target[arg_max - 1]][cur_lookup(j, arg_max - 1)]++;
      }
    }

    for (uint i=0; i < options.nTargetWords_; i++) {
      for (uint k=0; k < dcount[i].size(); k++) {
        if (dcount[i][k] > 0)
          sum += prior_weight[i][k];
      }
    }

    sum /= nSentences;
    std::cerr << "energy before ICM: " << sum << std::endl;

    if (par_mode != IBM23Nonpar) {

      if (par_mode == IBM23ParByPosition) {

        for (uint j = 0; j < align_param.yDim(); j++)
          reducedibm2_par_m_step(align_param, acount, j, options.align_m_step_iter_, options.deficient_);
      }
      else {
        reducedibm2_diffpar_m_step(align_param, acount, maxI - 1, options.align_m_step_iter_, options.deficient_);
      }
    }

    /*** ICM phase ***/

    if (true) {
      uint nSwitches = 0;

      Math1D::Vector<double> dict_sum(dcount.size());
      for (uint k = 0; k < dcount.size(); k++)
        dict_sum[k] = dcount[k].sum();

      for (size_t s = 0; s < nSentences; s++) {

        //std::cerr << "s: " << s << std::endl;

        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];

        const uint curJ = source[s].size();
        const uint curI = target[s].size();

        const SingleLookupTable& cur_lookup = slookup[s];
        Math2D::Matrix<double>& cur_acount = acount[curI];
        const Math2D::Matrix<double>& cur_align_model = alignment_model[curI];

        for (uint j = 0; j < curJ; j++) {

          //std::cerr << "j: " << j << std::endl;

          double best_change = 0.0;
          const ushort cur_aj = viterbi_alignment[s][j];
          ushort best_i = cur_aj;

          uint cur_target_word = (cur_aj == 0) ? 0 : cur_target[cur_aj - 1];

          Math1D::Vector<double>& cur_dictcount = dcount[cur_target_word];
          double cur_dictsum = dict_sum[cur_target_word];
          uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);

          for (uint i = 0; i <= curI; i++) {

            //std::cerr << "i: " << i << std::endl;

            if (i != cur_aj) {

              uint new_target_word = (i == 0) ? 0 : cur_target[i - 1];

              double hyp_dictsum = dict_sum[new_target_word];
              uint hyp_idx = (i == 0) ? cur_source[j] - 1 : cur_lookup(j, i - 1);

              double change = 0.0;

              assert(cur_acount(cur_aj, j) > 0);

              if (par_mode == IBM23Nonpar) {
                if (cur_acount(i, j) > 0) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_acount(i, j)];
                  change += -xlogx_table[cur_acount(i, j) + 1];
                }

                if (cur_acount(cur_aj, j) > 1) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_acount(cur_aj, j)];
                  change += -xlogx_table[cur_acount(cur_aj, j) - 1];
                }
              }
              else {

                //NOTE: in deficient mode we could to update calculations

                change -= -std::log(cur_align_model(cur_aj, j));
                change += -std::log(cur_align_model(i, j));
              }

              assert(!isnan(change));

              if (cur_target_word != new_target_word) {

                if (cur_dictsum > 1.0) {
                  //exploit log(1) = 0
                  change -= xlogx_table[cur_dictsum];
                  change += xlogx_table[cur_dictsum - 1];
                }

                //prior_weight is always relevant
                if (cur_dictcount[cur_idx] > 1) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_dictcount[cur_idx]];
                  change += -xlogx_table[cur_dictcount[cur_idx] - 1];
                }
                else
                  change -= prior_weight[cur_target_word][cur_idx];

                Math1D::Vector<double>& hyp_dictcount = dcount[new_target_word];

                if (hyp_dictsum > 0.0) {
                  //exploit log(1) = 0
                  change -= xlogx_table[hyp_dictsum];
                  change += xlogx_table[hyp_dictsum + 1];
                }

                //prior_weight is always relevant
                if (hyp_dictcount[hyp_idx] > 0) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[hyp_dictcount[hyp_idx]];
                  change += -xlogx_table[hyp_dictcount[hyp_idx] + 1];
                }
                else
                  change += prior_weight[new_target_word][hyp_idx];
              }

              assert(!isnan(change));

              if (best_change > change) {
                best_change = change;
                best_i = i;
              }
            }
          }

          if (best_change < -1e-2) {

            nSwitches++;

            viterbi_alignment[s][j] = best_i;

            uint new_target_word = (best_i == 0) ? 0 : cur_target[best_i - 1];

            if (cur_target_word != new_target_word) {

              uint hyp_idx = (best_i == 0) ? cur_source[j] - 1 : cur_lookup(j, best_i - 1);
              Math1D::Vector<double>& hyp_dictcount = dcount[new_target_word];

              cur_dictcount[cur_idx] -= 1.0;
              hyp_dictcount[hyp_idx] += 1.0;
              dict_sum[cur_target_word] -= 1.0;
              dict_sum[new_target_word] += 1.0;
            }

            cur_acount(cur_aj, j)--;
            cur_acount(best_i, j)++;
          }
        }
      }

      std::cerr << nSwitches << " switches in ICM" << std::endl;
    }

#ifndef NDEBUG
    Math1D::Vector<uint> count_count(6, 0);

    for (uint i = 0; i < options.nTargetWords_; i++) {
      for (uint k = 0; k < dcount[i].size(); k++) {
        if (dcount[i][k] < count_count.size())
          count_count[dcount[i][k]]++;
      }
    }

    std::cerr << "count count (lower end): " << count_count << std::endl;
#endif

    /*** recompute the dictionary ***/
    for (uint i = 0; i < options.nTargetWords_; i++) {

      //std::cerr << "i: " << i << std::endl;

      const double sum = dcount[i].sum();

      if (sum > 1e-307) {

        const double inv_sum = 1.0 / sum;
        assert(!isnan(inv_sum));

        for (uint k = 0; k < dcount[i].size(); k++)
          dict[i][k] = std::max(ibm1_min_dict_entry, dcount[i][k] * inv_sum);
      }
      else {
        //std::cerr << "WARNING : did not update dictionary entries because sum is " << sum << std::endl;
      }
    }

    //compute new alignment model from normalized fractional counts
    if (par_mode == IBM23Nonpar) {

      for (uint I = 0; I < alignment_model.size(); I++) {

        uint J = alignment_model[I].yDim();
        if (J > 0) {
          assert(alignment_model[I].xDim() == (I + 1));

          for (uint j = 0; j < J; j++) {

            double sum = acount[I].row_sum(j);

            if (sum > 1e-305) {
              const double inv_sum = 1.0 / sum;

              for (uint i = 0; i <= I; i++) {
                alignment_model[I](i, j) = std::max(ibm2_min_align_param, inv_sum * acount[I](i, j));
              }
            }
          }
        }
      }
    }
    else {

      if (par_mode == IBM23ParByPosition) {

        for (uint j = 0; j < align_param.yDim(); j++) {
          reducedibm2_par_m_step(align_param, acount, j, options.align_m_step_iter_, options.deficient_);
        }
      }
      else {
        reducedibm2_diffpar_m_step(align_param, acount, maxI - 1, options.align_m_step_iter_, options.deficient_);
      }

      par2nonpar_reduced_ibm2alignment_model(align_param, source_fert, alignment_model, par_mode, maxI - 1);
    }

    double energy = 0.0;

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

      const Math2D::Matrix<double>& cur_align_model = alignment_model[curI];

      for (uint j = 0; j < curJ; j++) {
        const uint aj = viterbi_alignment[s][j];
        energy -= std::log(cur_align_model(aj,j));
        if (aj == 0)
          energy -= std::log(dict[0][cur_source[j]-1]);
        else
          energy -= std::log(dict[cur_target[aj-1]][cur_lookup(j, aj-1)]);
      }
    }

    for (uint i=0; i < options.nTargetWords_; i++) {
      for (uint k=0; k < dcount[i].size(); k++) {
        if (dcount[i][k] > 0)
          energy += prior_weight[i][k];
      }
    }

    energy /= source.size();

    //std::cerr << "number of total alignments: " << sum_sum << std::endl;
    std::cerr << "energy: " << energy << std::endl;

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment[s], options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
        sum_fmeasure += f_measure(viterbi_alignment[s], options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment[s], options.sure_ref_alignments_[s + 1], options.possible_ref_alignments_[s + 1]);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### ReducedIBM2 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;
    }
  }

  if (par_mode == IBM23Nonpar)
    nonpar2par_reduced_ibm2alignment_model(align_param, alignment_model);
}
