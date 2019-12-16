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

IBM2Options::IBM2Options(uint nSourceWords, uint nTargetWords, RefAlignmentStructure& sure_ref_alignments,
                         RefAlignmentStructure& possible_ref_alignments):
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

    update_dict_from_counts(fwcount, prior_weight, nSentences, dict_weight_sum, smoothed_l0, l0_beta, dict_m_step_iter, dict,
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

    std::cerr << "IBM 2 energy after iteration #" << iter << ": "
              << ibm2_energy(source, slookup, target, alignment_model, dict, wcooc, nSourceWords, prior_weight, l0_beta, smoothed_l0, dict_weight_sum)
              << std::endl;

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (RefAlignmentStructure::const_iterator it = possible_ref_alignments.begin();
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

void par2nonpar_reduced_ibm2alignment_model(const Math3D::Tensor<double>& align_param, const Math1D::Vector<double>& source_fert,
    ReducedIBM2ClassAlignmentModel& alignment_model, IBM23ParametricMode par_mode, uint offset, bool deficient)
{
  for (uint k = 0; k < alignment_model.size(); k++) {

    for (uint c = 0; c < alignment_model[k].zDim(); c++) {

      for (uint j = 0; j < alignment_model[k].yDim(); j++) {

        double sum = 0.0;

        if (par_mode == IBM23ParByPosition) {

          if (deficient)
            sum = 1.0;
          else {
            for (uint i = 1; i < alignment_model[k].xDim(); i++)
              sum += align_param(i - 1, j, c);
          }

          if (sum > 1e-305) {
            const double inv_sum = 1.0 / sum;
            alignment_model[k](0, j, c) = source_fert[0];
            for (uint i = 1; i < alignment_model[k].xDim(); i++) {
              alignment_model[k](i, j, c) = source_fert[1] * align_param(i - 1, j, c) * inv_sum;
            }
          }
        }
        else {

          if (deficient)
            sum = 1.0;
          else {
            for (uint i = 1; i < alignment_model[k].xDim(); i++)
              sum += align_param(offset + j - (i - 1), 0, c);
          }

          if (sum > 1e-305) {
            const double inv_sum = 1.0 / sum;
            alignment_model[k](0, j, c) = source_fert[0];
            for (uint i = 1; i < alignment_model[k].xDim(); i++) {
              alignment_model[k](i, j, c) = source_fert[1] * align_param(offset + j - (i - 1), 0, c) * inv_sum;
            }
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

void nonpar2par_reduced_ibm2alignment_model(Math3D::Tensor<double>& align_param, const ReducedIBM2ClassAlignmentModel& align_model)
{
  align_param.set_constant(0.0);

  for (uint I = 0; I < align_model.size(); I++) {

    for (uint c = 0; c < align_model[I].zDim(); c++)
      for (uint y = 0; y < align_model[I].yDim(); y++)
        for (uint i = 1; i < align_model[I].xDim(); i++)
          align_param(i-1, y, c) += align_model[I](i, y, c);
  }

  for (uint c = 0; c < align_param.zDim(); c++) {
    for (uint y = 0; y < align_param.yDim(); y++) {

      double sum = align_param.sum_x(y, c);
      assert(sum > 1e-305);

      for (uint x = 0; x < align_param.xDim(); x++)
        align_param(x, y, c) /= sum;
    }
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

void reducedibm2_par_m_step(Math3D::Tensor<double>& align_param, const ReducedIBM2ClassAlignmentModel& acount, uint j, uint c, uint nIter,
                            bool deficient, bool quiet = false)
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
                                uint nIter, bool deficient)
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

    for (uint i = 0; i < xDim; i++)
      new_param[i] = align_param(i, 0, c) - alpha * grad[i];

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
      align_param(i, 0, c) = best_lambda * new_param[i] + neg_best_lambda * align_param(i, 0, c);
  }
}

double reduced_ibm2_perplexity(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                               const Storage1D<Math1D::Vector<uint> >& target, const ReducedIBM2ClassAlignmentModel& align_model,
                               const SingleWordDictionary& dict, const Math1D::Vector<WordClassType>& sclass,
                               const CooccuringWordsType& wcooc, uint nSourceWords)
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

    const Math3D::Tensor<double>& cur_align_model = align_model[curI];

    for (uint j = 0; j < curJ; j++) {

      //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
      //  We could cheat if we only want training/word alignment. But we just take the previous word
      const uint c = (j == 0) ? 0 : sclass[cur_source[j - 1]];

      const uint s_idx = cur_source[j];
      double cur_sum = cur_align_model(0, j, c) * dict[0][s_idx - 1];

      for (uint i = 0; i < curI; i++) {
        const uint t_idx = cur_target[i];
        cur_sum += cur_align_model(i + 1, j, c) * dict[t_idx][cur_lookup(j, i)];
      }
      sum -= std::log(cur_sum);
    }
  }

  return sum / nSentences;
}

double reduced_ibm2_energy(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                           const Storage1D<Math1D::Vector<uint> >& target, const ReducedIBM2ClassAlignmentModel& align_model,
                           const SingleWordDictionary& dict, const Math1D::Vector<WordClassType>& sclass, const CooccuringWordsType& wcooc, uint nSourceWords,
                           const floatSingleWordDictionary& prior_weight, double l0_beta, bool smoothed_l0, double dict_weight_sum)
{
  double energy = 0.0;

  if (dict_weight_sum != 0.0) {
    for (uint i = 0; i < dict.size(); i++) {

      const Math1D::Vector<double>& cur_dict = dict[i];
      const Math1D::Vector<float>& cur_prior = prior_weight[i];

      const uint size = cur_dict.size();

      if (smoothed_l0) {
        for (uint k = 0; k < size; k++)
          energy += cur_prior[k] * prob_penalty(cur_dict[k], l0_beta);
      }
      else {
        for (uint k = 0; k < size; k++)
          energy += cur_prior[k] * cur_dict[k];
      }
    }
  }

  energy += reduced_ibm2_perplexity(source, slookup, target, align_model, dict, sclass, wcooc, nSourceWords);

  return energy;
}

void train_reduced_ibm2(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                        const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2ClassAlignmentModel& alignment_model,
                        Math3D::Tensor<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                        const Math1D::Vector<WordClassType>& sclass, const IBM2Options& options, const floatSingleWordDictionary& prior_weight)
{
  const uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  const uint nClasses = sclass.max() + 1;

  const IBM23ParametricMode par_mode = options.ibm2_mode_;

  if (source_fert.size() != 2)
    source_fert.resize(2);
  source_fert[0] = (options.p0_ >= 0.0 && options.p0_ < 1.0) ? options.p0_ : 0.02;
  source_fert[1] = 1.0 - source_fert[0];

  std::cerr << "starting reduced IBM 2 training" << std::endl;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
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
        alignment_model[I].resize_dirty(I + 1, maxJ, nClasses);
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
    align_param.resize(maxI, maxJ, nClasses, 1.0);
  else
    align_param.resize(maxJ + maxI - 1, 1, nClasses, 1.0);

  SingleLookupTable aux_lookup;

  //TODO: estimate first alignment model from IBM1 dictionary

  SingleWordDictionary fwcount(options.nTargetWords_, MAKENAME(fwcount));
  for (uint i = 0; i < options.nTargetWords_; i++) {
    fwcount[i].resize(dict[i].size());
  }

  ReducedIBM2ClassAlignmentModel facount(alignment_model.size(), MAKENAME(facount));
  for (uint I = 0; I < lcooc.size(); I++) {
    facount[I].resize_dirty(alignment_model[I].dims());
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

      const Math3D::Tensor<double>& cur_align_model = alignment_model[curI];
      Math3D::Tensor<double>& cur_facount = facount[curI];

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

      assert(cur_align_model.yDim() >= curJ);
      assert(cur_facount.yDim() >= curJ);

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
        //  We could cheat if we only want training/word alignment. But we just take the previous word
        const uint c = (j == 0) ? 0 : sclass[cur_source[j - 1]];

        double coeff = dict[0][s_idx - 1] * cur_align_model(0, j, c);

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          coeff += dict[t_idx][cur_lookup(j, i)] * cur_align_model(i + 1, j, c);
        }

        coeff = 1.0 / coeff;
        assert(!isnan(coeff));

        double addon;
        addon = coeff * dict[0][s_idx - 1] * cur_align_model(0, j, c);

        fwcount[0][s_idx - 1] += addon;
        cur_facount(0, j, c) += addon;

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          const uint l = cur_lookup(j, i);

          addon = coeff * dict[t_idx][l] * cur_align_model(i + 1, j, c);

          //update dict
          fwcount[t_idx][l] += addon;

          //update alignment
          cur_facount(i + 1, j, c) += addon;
        }
      }
    }

    update_dict_from_counts(fwcount, prior_weight, nSentences, dict_weight_sum, smoothed_l0, l0_beta, options.dict_m_step_iter_, dict,
                            ibm1_min_dict_entry, options.unconstrained_m_step_);

    //compute new alignment model from normalized fractional counts
    if (par_mode == IBM23Nonpar) {

      for (uint I = 0; I < alignment_model.size(); I++) {

        uint J = alignment_model[I].yDim();
        if (J > 0) {
          assert(alignment_model[I].xDim() == (I + 1));

          for (uint c = 0; c < nClasses; c++) {

            for (uint j = 0; j < J; j++) {

              double sum = facount[I].sum_x(j, c);

              if (sum > 1e-305) {
                sum = 1.0 / sum;

                for (uint i = 0; i <= I; i++)
                  alignment_model[I](i, j, c) = std::max(ibm2_min_align_param, sum * facount[I](i, j, c));
              }
            }
          }
        }
      }
    }
    else {

      if (par_mode == IBM23ParByPosition) {

        for (uint c = 0; c < nClasses; c++) {
          for (uint j = 0; j < align_param.yDim(); j++) {
            reducedibm2_par_m_step(align_param, facount, j, c, options.align_m_step_iter_, options.deficient_, (nClasses > 1));
          }
        }
      }
      else {
        for (uint c = 0; c < nClasses; c++)
          reducedibm2_diffpar_m_step(align_param, facount, maxI - 1, c, options.align_m_step_iter_, options.deficient_);
      }

      par2nonpar_reduced_ibm2alignment_model(align_param, source_fert, alignment_model, par_mode, maxI - 1);
    }

    if (options.print_energy_) {
      std::cerr << "ReducedIBM-2 energy after iteration #" << iter << ": "
                << reduced_ibm2_energy(source, slookup, target, alignment_model, dict, sclass, wcooc, options.nSourceWords_,
                                       prior_weight, l0_beta, smoothed_l0, dict_weight_sum)
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

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];

        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        const uint curI = cur_target.size();
        const Math3D::Tensor<double>& cur_align_model = alignment_model[curI];

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm2_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, sclass, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm2_postdec_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, sclass, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
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

void train_reduced_ibm2_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                                       const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2ClassAlignmentModel& alignment_model,
                                       Math3D::Tensor<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                                       const Math1D::Vector<WordClassType>& sclass, const IBM2Options& options, const floatSingleWordDictionary& prior_weight)
{
  const uint nIter = options.nIterations_;
  bool smoothed_l0 = options.smoothed_l0_;
  double l0_beta = options.l0_beta_;

  const uint nClasses = sclass.max() + 1;

  const IBM23ParametricMode par_mode = options.ibm2_mode_;

  if (source_fert.size() != 2)
    source_fert.resize(2);
  source_fert[0] = (options.p0_ >= 0.0 && options.p0_ < 1.0) ? options.p0_ : 0.02;
  source_fert[1] = 1.0 - source_fert[0];

  std::cerr << "starting reduced IBM 2 training" << std::endl;

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

  double alpha = 100.0;
  double line_reduction_factor = 0.5;

  uint nSuccessiveReductions = 0;

  Math1D::Vector<double> slack_vector(options.nTargetWords_, 0.0);
  Math1D::Vector<double> new_slack_vector(options.nTargetWords_, 0.0);

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
        alignment_model[I].resize_dirty(I + 1, maxJ, nClasses);
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
    align_param.resize(maxI, maxJ, nClasses, 1.0);
  else
    align_param.resize(maxJ + maxI - 1, 1, nClasses, 1.0);

  const uint offset = maxI - 1;

  SingleLookupTable aux_lookup;

  //TODO: estimate first alignment model from IBM1 dictionary

  SingleWordDictionary wgrad(options.nTargetWords_, MAKENAME(wgrad));
  SingleWordDictionary new_dict(options.nTargetWords_, MAKENAME(new_dict));
  SingleWordDictionary hyp_dict(options.nTargetWords_, MAKENAME(new_dict));
  for (uint i = 0; i < options.nTargetWords_; i++) {
    wgrad[i].resize(dict[i].size());
    new_dict[i].resize(dict[i].size());
    hyp_dict[i].resize(dict[i].size());
  }

  ReducedIBM2ClassAlignmentModel agrad(alignment_model.size(), MAKENAME(agrad));
  ReducedIBM2ClassAlignmentModel new_alignment_model(alignment_model.size(), MAKENAME(new_alignment_model));
  ReducedIBM2ClassAlignmentModel hyp_alignment_model(alignment_model.size(), MAKENAME(hyp_alignment_model));
  for (uint I = 0; I < lcooc.size(); I++) {
    agrad[I].resize_dirty(alignment_model[I].dims());
    new_alignment_model[I].resize_dirty(alignment_model[I].dims());
    hyp_alignment_model[I].resize_dirty(alignment_model[I].dims());
  }

  Math3D::Tensor<double> align_param_grad(align_param.dims());
  Math3D::Tensor<double> hyp_align_param(align_param.dims());
  Math3D::Tensor<double> new_align_param(align_param.dims());

  double energy = reduced_ibm2_energy(source, slookup, target, alignment_model, dict, sclass, wcooc, options.nSourceWords_,
                                      prior_weight, l0_beta, smoothed_l0, dict_weight_sum);

  std::cerr << "start_energy: " << energy << std::endl;

  assert(!isnan(energy));

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "starting reduced IBM 2 gd-iteration #" << iter << std::endl;

    for (uint i = 0; i < options.nTargetWords_; i++) {
      wgrad[i].set_constant(0.0);
    }
    for (uint I = 0; I < lcooc.size(); I++) {
      agrad[I].set_constant(0.0);
    }

    if (par_mode != IBM23Nonpar) {
      align_param_grad.set_constant(0.0);
    }

    /****** compute gradient *****/

    for (size_t s = 0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      const Math3D::Tensor<double>& cur_align_model = alignment_model[curI];
      Math3D::Tensor<double>& cur_agrad = agrad[curI];

      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
        //  We could cheat if we only want training/word alignment. But we just take the previous word
        const uint c = (j == 0) ? 0 : sclass[cur_source[j - 1]];

        double coeff = dict[0][s_idx - 1] * cur_align_model(0, j, c);

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];
          coeff += dict[t_idx][cur_lookup(j, i)] * cur_align_model(i + 1, j, c);
        }

        coeff = 1.0 / coeff;
        assert(!isnan(coeff));

        wgrad[0][s_idx - 1] -= coeff * cur_align_model(0, j, c);
        cur_agrad(0, j, c) -= coeff * dict[0][s_idx - 1];

        for (uint i = 0; i < curI; i++) {
          const uint t_idx = cur_target[i];

          wgrad[t_idx][cur_lookup(j, i)] -= coeff * cur_align_model(i + 1, j, c);
          cur_agrad(i + 1, j, c) -= coeff * dict[t_idx][cur_lookup(j, i)];
        }
      }
    } //loop over sentences finished


    for (uint i = 0; i < options.nTargetWords_; i++) {
      wgrad[i] *= 1.0 / nSentences;
    }
    for (uint I = 0; I < lcooc.size(); I++) {
      agrad[I] *= 1.0 / nSentences;
    }

    if (dict_weight_sum != 0.0) {
      for (uint i = 0; i < options.nTargetWords_; i++) {

        const Math1D::Vector<double>& cur_dict = dict[i];
        const Math1D::Vector<float>& cur_prior = prior_weight[i];
        Math1D::Vector<double>& cur_dict_grad = wgrad[i];
        const uint size = cur_dict.size();

        for (uint k = 0; k < size; k++) {
          if (smoothed_l0)
            cur_dict_grad[k] += cur_prior[k] * prob_pen_prime(cur_dict[k], l0_beta);
          else
            cur_dict_grad[k] += cur_prior[k];
        }
      }
    }

    if (par_mode != IBM23Nonpar) {
      for (uint I = 0; I < lcooc.size(); I++) {
        
        if (par_mode == IBM23ParByPosition) {

          for (uint c = 0; c < agrad[I].zDim(); c++) {
            for (uint j = 0; j < agrad[I].yDim(); j++) {

              if (options.deficient_) {
                for (uint i = 1; i < agrad[I].xDim(); i++)
                  align_param_grad(i - 1, j, c) += agrad[I](i, j, c);
              }
              else {
                
                // for detals on the quotient rule see IBM3Trainer::compute_dist_param_gradient

                double param_sum = 0.0;
                double product_sum = 0.0;
                
                for (uint i = 1; i < agrad[I].xDim(); i++) {
                  param_sum += align_param(i - 1, j, c);
                  product_sum += agrad[I](i, j, c) * align_param(i - 1, j, c);
                }                

                param_sum = std::max(ibm2_min_align_param, param_sum);
                const double combined = -product_sum / (param_sum * param_sum);

                for (uint i = 1; i < agrad[I].xDim(); i++) {
                  align_param_grad(i - 1, j, c) += combined  //combined term
                                                   + (agrad[I](i, j, c) / param_sum);   //term for j
                }                               
              }
            }
          }
        }
        else {
          for (uint c = 0; c < agrad[I].zDim(); c++) {

            if (options.deficient_) {
              for (uint j = 0; j < agrad[I].yDim(); j++)
                for (uint i = 1; i < agrad[I].xDim(); i++)
                  align_param_grad(offset + j - (i - 1), 0, c) += agrad[I](i, j, c);
            }
            else {
              for (uint j = 0; j < agrad[I].yDim(); j++) {

                // for detals on the quotient rule see IBM3Trainer::compute_dist_param_gradient

                double param_sum = 0.0;
                double product_sum = 0.0;

                for (uint i = 1; i < agrad[I].xDim(); i++) {
                  param_sum += align_param(offset + j - (i - 1), 0, c);
                  product_sum += agrad[I](i, j, c) * align_param(offset + j - (i - 1), 0, c);
                }                
               
                param_sum = std::max(ibm2_min_align_param, param_sum);
                const double combined = -product_sum / (param_sum * param_sum);

                for (uint i = 1; i < agrad[I].xDim(); i++) {
                  align_param_grad(offset + j - (i - 1), 0, c) += combined  //combined term
                                                                  + (agrad[I](i, j, c) / param_sum);   //term for j
                }                                               
              }
            }
          }
        }
      }
      
      align_param_grad *= source_fert[1];
    }

    /**** move in gradient direction ****/
    double real_alpha = alpha;

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //for (uint k = 0; k < dict[i].size(); k++)
      //  new_dict[i][k] = dict[i][k] - real_alpha * dict_grad[i][k];
      Math1D::go_in_neg_direction(new_dict[i], dict[i], wgrad[i], real_alpha);
    }

    if (dict_weight_sum != 0.0)
      new_slack_vector = slack_vector;

    if (par_mode == IBM23Nonpar) {
      for (uint I = 0; I < lcooc.size(); I++) {

        if (agrad[I].size() > 0)
          Math3D::go_in_neg_direction(new_alignment_model[I], alignment_model[I], agrad[I], real_alpha);
      }
    }
    else {
      Math3D::go_in_neg_direction(new_align_param, align_param, align_param_grad, real_alpha);
    }

    /**** reproject on the simplices [Michelot 1986] ****/

    for (uint i = 0; i < options.nTargetWords_; i++) {

      const uint nCurWords = new_dict[i].size();

      if (dict_weight_sum != 0.0)
        projection_on_simplex_with_slack(new_dict[i].direct_access(), slack_vector[i], nCurWords, ibm1_min_dict_entry);
      else
        projection_on_simplex(new_dict[i].direct_access(), nCurWords, ibm1_min_dict_entry);
    }

    if (par_mode == IBM23Nonpar) {
      for (uint I = 0; I < lcooc.size(); I++) {

        Math1D::Vector<double> temp(new_alignment_model[I].xDim());

        for (uint c = 0; c < new_alignment_model[I].zDim(); c ++) {
          for (uint y = 0; y < new_alignment_model[I].yDim(); y ++) {

            new_alignment_model[I].get_x(y, c, temp);
            projection_on_simplex(temp.direct_access(), temp.size(), ibm2_min_align_param);
            new_alignment_model[I].set_x(y, c, temp);
            
            assert(!isnan(temp.sum()));
          }
        }
      }
    }
    else {
      
      Math1D::Vector<double> temp(new_align_param.xDim());
      
      for (uint c = 0; c < new_align_param.zDim(); c++) {
        for (uint y = 0; y < new_align_param.yDim(); y++) {
        
          new_align_param.get_x(y, c, temp);
          projection_on_simplex(temp.direct_access(), temp.size(), ibm2_min_align_param);
          new_align_param.set_x(y, c, temp);
        }
      }
    }

    /**** find a suitable step size ****/

    double lambda = 1.0;
    double best_lambda = 1.0;

    double hyp_energy = 1e300;

    uint nInnerIter = 0;

    bool decreasing = true;

    while (hyp_energy > energy || decreasing) {

      nInnerIter++;

      if (hyp_energy <= 0.95 * energy)
        break;

      if (hyp_energy < 0.99 * energy && nInnerIter > 3)
        break;

      lambda *= line_reduction_factor;

      const double neg_lambda = 1.0 - lambda;

      for (uint i = 0; i < options.nTargetWords_; i++) {

        //for (uint k = 0; k < dict[i].size(); k++)
        //  hyp_dict[i][k] = neg_lambda * dict[i][k] + lambda * new_dict[i][k];

        assert(dict[i].size() == hyp_dict[i].size());
        Math1D::assign_weighted_combination(hyp_dict[i], neg_lambda, dict[i], lambda, new_dict[i]);
      }

      if (par_mode == IBM23Nonpar) {

        for (uint I = 0; I < lcooc.size(); I++) {

          if (hyp_alignment_model[I].size() > 0)
            Math3D::assign_weighted_combination(hyp_alignment_model[I], neg_lambda, alignment_model[I], lambda, new_alignment_model[I]);
        }
      }
      else {

        Math3D::assign_weighted_combination(hyp_align_param, neg_lambda, align_param, lambda, new_align_param);

        par2nonpar_reduced_ibm2alignment_model(hyp_align_param, source_fert, hyp_alignment_model, par_mode, maxI - 1);
      }

      double new_energy = reduced_ibm2_energy(source, slookup, target, hyp_alignment_model, hyp_dict, sclass, wcooc, options.nSourceWords_,
                                              prior_weight, l0_beta, smoothed_l0, dict_weight_sum);

      assert(!isnan(new_energy));

      std::cerr << "new hyp: " << new_energy << ", previous: " << hyp_energy << std::endl;

      if (new_energy < hyp_energy) {
        hyp_energy = new_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;
    }

    if (nInnerIter > 4) {
      nSuccessiveReductions++;
    }
    else {
      nSuccessiveReductions = 0;
    }

    if (nSuccessiveReductions > 15) {
      line_reduction_factor *= 0.9;
      nSuccessiveReductions = 0;
    }
    //     std::cerr << "alpha: " << alpha << std::endl;

    const double neg_best_lambda = 1.0 - best_lambda;

    for (uint i = 0; i < options.nTargetWords_; i++) {

      //for (uint k = 0; k < dict[i].size(); k++)
      //  dict[i][k] = neg_best_lambda * dict[i][k] + best_lambda * new_dict[i][k];

      Math1D::assign_weighted_combination(dict[i], neg_best_lambda, dict[i], best_lambda, new_dict[i]);
    }
    if (dict_weight_sum > 0.0)
      Math1D::assign_weighted_combination(slack_vector, neg_best_lambda, slack_vector, best_lambda, new_slack_vector);

    if (par_mode == IBM23Nonpar) {
      for (uint I = 0; I < lcooc.size(); I++) {

        if (agrad[I].size() > 0)
          Math3D::assign_weighted_combination(alignment_model[I], neg_best_lambda, alignment_model[I], best_lambda, new_alignment_model[I]);
      }
    }
    else {

      Math3D::assign_weighted_combination(align_param, neg_best_lambda, align_param, lambda, new_align_param);
      par2nonpar_reduced_ibm2alignment_model(align_param, source_fert, alignment_model, par_mode, maxI - 1);
    }

    energy = hyp_energy;

    if (options.print_energy_)
      std::cerr << "ReducedIBM-2 energy after gd-iteration #" << iter << ": " << energy << std::endl;

    /************* compute alignment error rate ****************/
    if (!options.possible_ref_alignments_.empty()) {

      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      double sum_postdec_aer = 0.0;
      double sum_postdec_fmeasure = 0.0;
      double sum_postdec_daes = 0.0;

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        const Storage1D<uint>& cur_source = source[s];
        const Storage1D<uint>& cur_target = target[s];

        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, options.nSourceWords_, slookup[s], aux_lookup);

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        const uint curI = cur_target.size();
        const Math3D::Tensor<double>& cur_align_model = alignment_model[curI];

        //compute viterbi alignment
        Storage1D<AlignBaseType> viterbi_alignment;
        compute_ibm2_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, sclass, viterbi_alignment);

        //add alignment error rate
        sum_aer += AER(viterbi_alignment, cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment, cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment, cur_sure, cur_possible);

        std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
        compute_ibm2_postdec_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, sclass, postdec_alignment);

        sum_postdec_aer += AER(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_fmeasure += f_measure(postdec_alignment, cur_sure, cur_possible);
        sum_postdec_daes += nDefiniteAlignmentErrors(postdec_alignment, cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### ReducedIBM2 Viterbi-AER after gd-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-fmeasure after gd-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-DAE/S after gd-iteration #" << iter << ": " << nErrors << std::endl;

      sum_postdec_aer *= 100.0 / nContributors;
      sum_postdec_fmeasure /= nContributors;
      sum_postdec_daes /= nContributors;

      std::cerr << "#### ReducedIBM2 Postdec-AER after gd-iteration #" << iter << ": " << sum_postdec_aer << " %" << std::endl;
      std::cerr << "#### ReducedIBM2 Postdec-fmeasure after gd-iteration #" << iter << ": " << sum_postdec_fmeasure << std::endl;
      std::cerr << "#### ReducedIBM2 Postdec-DAE/S after gd-iteration #" << iter << ": " << sum_postdec_daes << std::endl;
    }
  }

  if (par_mode == IBM23Nonpar)
    nonpar2par_reduced_ibm2alignment_model(align_param, alignment_model);
}

void reduced_ibm2_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                                   const CooccuringWordsType& wcooc, const CooccuringLengthsType& lcooc, ReducedIBM2ClassAlignmentModel& alignment_model,
                                   Math3D::Tensor<double>& align_param, Math1D::Vector<double>& source_fert, SingleWordDictionary& dict,
                                   const Math1D::Vector<WordClassType>& sclass, const IBM2Options& options, const floatSingleWordDictionary& prior_weight,
                                   const Math1D::Vector<double>& xlogx_table)
{
  const uint nIter = options.nIterations_;
  const IBM23ParametricMode par_mode = options.ibm2_mode_;

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  const uint nClasses = sclass.max() + 1;

  if (source_fert.size() != 2)
    source_fert.resize(2);
  source_fert[0] = (options.p0_ >= 0.0 && options.p0_ < 1.0) ? options.p0_ : 0.02;
  source_fert[1] = 1.0 - source_fert[0];

  //initialize alignment model
  alignment_model.resize_dirty(lcooc.size());
  ReducedIBM2ClassAlignmentModel acount(lcooc.size(), MAKENAME(acount));

  for (uint I = 0; I < lcooc.size(); I++) {

    if (lcooc[I].size() > 0) {

      uint maxJ = lcooc[I].max();

      if (maxJ > 0) {
        alignment_model[I].resize_dirty(I + 1, maxJ, nClasses);
        alignment_model[I].set_constant(1.0 / (I + 1));
        acount[I].resize_dirty(I + 1, maxJ, nClasses);
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
    align_param.resize(maxI, maxJ, nClasses, 1.0);
  else
    align_param.resize(maxJ + maxI - 1, 1, nClasses, 1.0);

  SingleLookupTable aux_lookup;

  Storage1D<Math1D::Vector<AlignBaseType> > viterbi_alignment(source.size());

  double dict_weight_sum = 0.0;
  for (uint i = 0; i < options.nTargetWords_; i++) {
    dict_weight_sum += prior_weight[i].max_abs();
  }

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

      const Math3D::Tensor<double>& cur_align_model = alignment_model[curI];
      Math3D::Tensor<double>& cur_acount = acount[curI];

      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];

        //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
        //  We could cheat if we only want training/word alignment. But we just take the previous word
        const uint c = (j == 0) ? 0 : sclass[cur_source[j - 1]];

        uint arg_max = MAX_UINT;

        double max = dict[0][s_idx - 1] * cur_align_model(0, j, c);
        arg_max = 0;

        for (uint i = 0; i < curI; i++) {

          double hyp = dict[cur_target[i]][cur_lookup(j, i)] * cur_align_model(i + 1, j, c);

          //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;

          if (hyp > max) {
            max = hyp;
            arg_max = i + 1;
          }
        }

        //std::cerr << "arg_min: " << arg_min << std::endl;

        sum -= std::log(max);

        viterbi_alignment[s][j] = arg_max;
        cur_acount(arg_max, j, c)++;

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

        for (uint c = 0; c < nClasses; c++) {
          for (uint j = 0; j < align_param.yDim(); j++)
            reducedibm2_par_m_step(align_param, acount, j, c, options.align_m_step_iter_, options.deficient_, (nClasses > 1));
        }
      }
      else {
        for (uint c = 0; c < nClasses; c++)
          reducedibm2_diffpar_m_step(align_param, acount, maxI - 1, c, options.align_m_step_iter_, options.deficient_);
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
        Math3D::Tensor<double>& cur_acount = acount[curI];
        const Math3D::Tensor<double>& cur_align_model = alignment_model[curI];

        for (uint j = 0; j < curJ; j++) {

          //std::cerr << "j: " << j << std::endl;

          //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
          //  We could cheat if we only want training/word alignment. But we just take the previous word
          const uint c = (j == 0) ? 0 : sclass[cur_source[j - 1]];

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

              assert(cur_acount(cur_aj, j, c) > 0);

              if (par_mode == IBM23Nonpar) {
                if (cur_acount(i, j, c) > 0) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_acount(i, j, c)];
                  change += -xlogx_table[cur_acount(i, j, c) + 1];
                }

                if (cur_acount(cur_aj, j, c) > 1) {
                  //exploit log(1) = 0
                  change -= -xlogx_table[cur_acount(cur_aj, j, c)];
                  change += -xlogx_table[cur_acount(cur_aj, j, c) - 1];
                }
              }
              else {

                //NOTE: in deficient mode we could to update calculations

                change -= -std::log(cur_align_model(cur_aj, j, c));
                change += -std::log(cur_align_model(i, j, c));
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

            cur_acount(cur_aj, j, c)--;
            cur_acount(best_i, j, c)++;
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

          for (uint c = 0; c < nClasses; c++) {
            for (uint j = 0; j < J; j++) {

              double sum = acount[I].sum_x(j, c);

              if (sum > 1e-305) {
                const double inv_sum = 1.0 / sum;

                for (uint i = 0; i <= I; i++) {
                  alignment_model[I](i, j, c) = std::max(ibm2_min_align_param, inv_sum * acount[I](i, j, c));
                }
              }
            }
          }
        }
      }
    }
    else {

      if (par_mode == IBM23ParByPosition) {

        for (uint c = 0; c < nClasses; c++) {
          for (uint j = 0; j < align_param.yDim(); j++) {
            reducedibm2_par_m_step(align_param, acount, j, c, options.align_m_step_iter_, options.deficient_, (nClasses > 1));
          }
        }
      }
      else {
        for (uint c = 0; c < nClasses; c++)
          reducedibm2_diffpar_m_step(align_param, acount, maxI - 1, c, options.align_m_step_iter_, options.deficient_);
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

      const Math3D::Tensor<double>& cur_align_model = alignment_model[curI];

      for (uint j = 0; j < curJ; j++) {

        //NOTE: a generative model does not allow to condition on sclass[source_sentence[j]]
        //  We could cheat if we only want training/word alignment. But we just take the previous word
        const uint c = (j == 0) ? 0 : sclass[cur_source[j - 1]];

        const uint aj = viterbi_alignment[s][j];
        energy -= std::log(cur_align_model(aj, j, c));
        if (aj == 0)
          energy -= std::log(dict[0][cur_source[j] - 1]);
        else
          energy -= std::log(dict[cur_target[aj - 1]][cur_lookup(j, aj - 1)]);
      }
    }

    if (dict_weight_sum > 0.0) {
      for (uint i=0; i < options.nTargetWords_; i++) {
        for (uint k=0; k < dcount[i].size(); k++) {
          if (dcount[i][k] > 0)
            energy += prior_weight[i][k];
        }
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

      for (RefAlignmentStructure::const_iterator it = options.possible_ref_alignments_.begin();
           it != options.possible_ref_alignments_.end(); it++) {

        uint s = it->first - 1;

        if (s >= nSentences)
          break;

        nContributors++;

        const AlignmentStructure& cur_sure = options.sure_ref_alignments_[s + 1];
        const AlignmentStructure& cur_possible = it->second;

        //add alignment error rate
        sum_aer += AER(viterbi_alignment[s], cur_sure, cur_possible);
        sum_fmeasure += f_measure(viterbi_alignment[s], cur_sure, cur_possible);
        nErrors += nDefiniteAlignmentErrors(viterbi_alignment[s], cur_sure, cur_possible);
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### ReducedIBM-2 Viterbi-AER after Viterbi-iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### ReducedIBM-2 Viterbi-fmeasure after Viterbi-iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### ReducedIBM-2 Viterbi-DAE/S after Viterbi-iteration #" << iter << ": " << nErrors << std::endl;
    }
  }

  if (par_mode == IBM23Nonpar)
    nonpar2par_reduced_ibm2alignment_model(align_param, alignment_model);
}