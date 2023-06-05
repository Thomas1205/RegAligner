/**** written by Thomas Schoenemann as a private person, since February 2013 ****/

#include "ibm5_training.hh"

#include "timing.hh"
#include "projection.hh"
#include "training_common.hh"   // for get_wordlookup(), dictionary and start-prob m-step
#include "stl_util.hh"
#include "storage_util.hh"

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"

IBM5Trainer::IBM5Trainer(const Storage1D<Math1D::Vector<uint> >& source_sentence, const LookupTable& slookup,
                         const Storage1D<Math1D::Vector<uint> >& target_sentence,
                         const RefAlignmentStructure& sure_ref_alignments, const RefAlignmentStructure& possible_ref_alignments,
                         SingleWordDictionary& dict, const CooccuringWordsType& wcooc, const Math1D::Vector<uint>& tfert_class, uint nSourceWords,
                         uint nTargetWords, const floatSingleWordDictionary& prior_weight,
                         const Storage1D<WordClassType>& source_class, const Storage1D<WordClassType>& target_class,
                         const Math1D::Vector<double>& log_table, const Math1D::Vector<double>& xlogx_table,
                         const FertModelOptions& options)
  : FertilityModelTrainer(source_sentence, slookup, target_sentence, dict, wcooc, tfert_class, nSourceWords, nTargetWords, prior_weight,
                          sure_ref_alignments, possible_ref_alignments, log_table, xlogx_table, options, true),
    inter_distortion_prob_(maxJ_ + 1), intra_distortion_prob_(maxJ_ + 1),
    sentence_start_prob_(maxJ_ + 1), distortion_type_(options.ibm5_distortion_type_),
	//nonpar_distortion_(options.ibm5_nonpar_distortion_),
    use_sentence_start_prob_(!options.uniform_sentence_start_prob_), uniform_intra_prob_(options.uniform_intra_prob_),
    cept_start_mode_(options.cept_start_mode_), intra_dist_mode_(options.intra_dist_mode_),
    source_class_(source_class), target_class_(target_class), deficient_(options.deficient_),
	inter_dist_grouping_(options.ibm5_dist_grouping_), dist_grouping_limit_(options.dist_grouping_limit_),
    dist_m_step_iter_(options.dist_m_step_iter_), start_m_step_iter_(options.start_m_step_iter_)
{
  uint max_source_class = 0;
  uint min_source_class = MAX_UINT;
  for (uint j = 1; j < source_class_.size(); j++) {
    max_source_class = std::max<uint>(max_source_class, source_class_[j]);
    min_source_class = std::min<uint>(min_source_class, source_class_[j]);
  }
  if (min_source_class > 0) {
    for (uint j = 1; j < source_class_.size(); j++)
      source_class_[j] -= min_source_class;
    max_source_class -= min_source_class;
  }

  nSourceClasses_ = max_source_class + 1;

  uint max_target_class = 0;
  uint min_target_class = MAX_UINT;
  for (uint i = 1; i < target_class_.size(); i++) {
    max_target_class = std::max<uint>(max_target_class, target_class_[i]);
    min_target_class = std::min<uint>(min_target_class, target_class_[i]);
  }
  if (min_target_class > 0) {
    for (uint i = 1; i < target_class_.size(); i++)
      target_class_[i] -= min_target_class;
    max_target_class -= min_target_class;
  }

  nTargetClasses_ = max_target_class + 1;
  
  std::set<uint> seenJs;
  for (size_t s = 0; s < source_sentence.size(); s++)
    seenJs.insert(source_sentence[s].size());

  const uint nClasses = (intra_dist_mode_ == IBM4IntraDistModeSource) ? nSourceClasses_ : nTargetClasses_;
  
  const uint nDisplacements = 2 * maxJ_-1;
  if (nDisplacements <= (2 * dist_grouping_limit_ +1) ) {
	inter_dist_grouping_ = DistGroupModeOff;
	dist_grouping_limit_ = 1;
  }
  displacement_offset_ = maxJ_-1;

  intra_distortion_param_.resize(maxJ_, nClasses, 1.0 / maxJ_);
  sentence_start_parameters_.resize(maxJ_, 1.0 / maxJ_);

  for (std::set<uint>::const_iterator it = seenJs.begin(); it != seenJs.end(); it++) {
    const uint J = *it;
    sentence_start_prob_[J].resize(J, 1.0 / J);
  }

  for (uint J=1; J <= maxJ_; J++) {
    //need this for all J as the number of open positions counts
    intra_distortion_prob_[J].resize(J, nClasses, 1.0 / J);
    inter_distortion_prob_[J].resize(J, maxJ_, nSourceClasses_, 1.0 / J);
  }

  inter_distortion_param_.resize(nDisplacements, nSourceClasses_, 1.0 / nDisplacements);
  inter_grouping_prob_.resize(nSourceClasses_,0.0);
  inter_dist_pospar_param_.resize(maxJ_+1, maxJ_+1, nSourceClasses_);
  inter_pospar_grouping_prob_.resize(maxJ_+1,nSourceClasses_,0.0);
  
  if (distortion_type_ != IBM23Nonpar) {
	if (inter_dist_grouping_ != DistGroupModeOff) {
	  inter_distortion_param_.set_constant(0.8 / (2*dist_grouping_limit_+1));
      inter_dist_pospar_param_.set_constant(0.8 / (dist_grouping_limit_+1));
	  inter_grouping_prob_.set_constant(0.2);
	  inter_pospar_grouping_prob_.set_constant(0.2);
	}
	else {
	  inter_distortion_param_.set_constant(1.0 / nDisplacements);
      inter_dist_pospar_param_.set_constant(1.0 / (maxJ_+1));
	  inter_grouping_prob_.set_constant(0.0);
	  inter_pospar_grouping_prob_.set_constant(0.0);		
	}
	par2nonpar_inter_distortion();
    par2nonpar_intra_distortion();
  }
}

/*virtual*/ std::string IBM5Trainer::model_name() const
{
  return "IBM-5";
}

void IBM5Trainer::par2nonpar_inter_distortion()
{
  if (distortion_type_ == IBM23Nonpar)
	return;

  for (uint J = 1; J < inter_distortion_prob_.size(); J++) {

    for (uint s = 0; s < inter_distortion_prob_[J].zDim(); s++) {

      for (int prev_pos = 0; prev_pos < inter_distortion_prob_[J].yDim(); prev_pos++) {

		if (distortion_type_ == IBM23ParByPosition) {

		  const uint nDistGroupingTerms = maxJ_+1 - (dist_grouping_limit_+1);
			
		  double sum = 0.0;
		  uint nLong = 0;
		  for (uint j = 0; j < J; j++)
		  {
			if (inter_dist_grouping_ == DistGroupModeOff || j <= dist_grouping_limit_)
			  sum += inter_dist_pospar_param_(j,prev_pos,s);
		    else if (inter_dist_grouping_ == DistGroupModeDivide)
			  sum += inter_pospar_grouping_prob_(prev_pos,s) / nDistGroupingTerms;
		    else {
			  assert(inter_dist_grouping_ == DistGroupModeExpand);
			  nLong++;
			}
		  }
		  if (nLong > 0) 
		  {
			assert(inter_dist_grouping_ == DistGroupModeExpand);
			assert(dist_grouping_limit_ < J);
			sum += inter_pospar_grouping_prob_(prev_pos,s);
			assert(sum >= 0.99 && sum <= 1.01); //for pospar always the entire sum of non-grouped probs enters
		  }
		  
		  if (sum > 1e-300) {
			
			double inv_sum = (deficient_) ? 1.0 : 1.0 / sum;
			for (uint j = 0; j < J; j++)
		    {
			  if (inter_dist_grouping_ == DistGroupModeOff || j <= dist_grouping_limit_)
				inter_distortion_prob_[J](j, prev_pos, s) = inter_dist_pospar_param_(j,prev_pos,s) * inv_sum;
			  else if (inter_dist_grouping_ == DistGroupModeDivide)
				inter_distortion_prob_[J](j, prev_pos, s) = inter_grouping_prob_[s] * inv_sum / nDistGroupingTerms;
			  else 
				inter_distortion_prob_[J](j, prev_pos, s) = inter_grouping_prob_[s] * inv_sum / nLong; 
			}
		  }
		}
		else if (distortion_type_ == IBM23ParByDifference) {
			
		  const uint nDistGroupingTerms = inter_distortion_param_.xDim() - (2*dist_grouping_limit_+1);
		  double sum = 0.0;
		  uint nLong = 0;
		  for (int j = 0; j < J; j++) {
			if (inter_dist_grouping_ == DistGroupModeOff || abs(j - prev_pos) <= dist_grouping_limit_)
              sum += inter_distortion_param_(displacement_offset_ + j - prev_pos, s);
		    else if (inter_dist_grouping_ == DistGroupModeDivide) {
			  assert(abs(j-prev_pos) > dist_grouping_limit_);
			  sum += inter_grouping_prob_[s] / nDistGroupingTerms;
			}
		    else {
			  assert(inter_dist_grouping_ == DistGroupModeExpand);
			  assert(j > dist_grouping_limit_);
			  nLong++;
			}
		  }
		  if (nLong > 0) {
			assert(inter_dist_grouping_ == DistGroupModeExpand);
			sum += inter_grouping_prob_[s];
		  }
		 
		  if (sum > 1e-300) {
			double inv_sum = (deficient_) ? 1.0 : 1.0 / sum;
			
			for (int j = 0; j < J; j++) {
			  if (inter_dist_grouping_ == DistGroupModeOff || abs(j-prev_pos) <= dist_grouping_limit_)
                inter_distortion_prob_[J](j, prev_pos, s) = inv_sum * inter_distortion_param_(displacement_offset_ + j - prev_pos, s);
		      else if (inter_dist_grouping_ == DistGroupModeDivide) {
				assert(abs(j-prev_pos) > dist_grouping_limit_);
			    inter_distortion_prob_[J](j, prev_pos, s) = inv_sum * inter_grouping_prob_[s] / nDistGroupingTerms;
			  }
			  else //DistGroupModeExpand
				inter_distortion_prob_[J](j, prev_pos, s) = inv_sum * inter_grouping_prob_[s] / nLong; 
		    }
		  }
		}
	  }
    }
  }
}

void IBM5Trainer::par2nonpar_intra_distortion()
{
  for (uint J = 1; J < intra_distortion_prob_.size(); J++) {

    for (uint s = 0; s < intra_distortion_prob_[J].yDim(); s++) {

      double sum = 0.0;
      if (deficient_)
        sum = 1.0;
      else {
        for (uint j = 0; j < intra_distortion_prob_[J].xDim(); j++)
          sum += intra_distortion_param_(j, s);
      }

      assert(sum > 1e-305);

      for (uint j = 0; j < intra_distortion_prob_[J].xDim(); j++)
        intra_distortion_prob_[J] (j, s) = std::max(fert_min_param_entry, intra_distortion_param_(j, s) / sum);
    }
  }
}

/*virtual*/ long double IBM5Trainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment) const
{
  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));
  Storage1D<std::vector<ushort> > aligned_source_words(curI + 1);
  for (uint i = 0; i <= curI; i++)
    aligned_source_words[i].reserve(10);

  for (uint j = 0; j < curJ; j++) {
    const uint aj = alignment[j];
    aligned_source_words[aj].push_back(j);
    fertility[aj]++;

    if (aj == 0) {
      prob *= dict_[0][source[j] - 1];
      //DEBUG
      if (isnan(prob))
        std::cerr << "prob nan after empty word dict prob" << std::endl;
      //END_DEBUG
    }
    else {
      const uint ti = target[aj - 1];
      prob *= dict_[ti][lookup(j, aj - 1)];
    }
  }

  const uint zero_fert = fertility[0];
  if (curJ < 2 * zero_fert)
    return 0.0;

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = target[i - 1];
    prob *= fertility_prob_[t_idx][fertility[i]];
  }

  //DEBUG
  if (isnan(prob))
    std::cerr << "prob nan after fertility probs" << std::endl;
  //END_DEBUG

  prob *= distortion_prob(source, target, aligned_source_words);

  //handle empty word -- dictionary probs were handled above
  assert(zero_fert <= 2 * curJ);

  prob *= choose_factor_[curJ][zero_fert];
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  prob *= p_zero_pow_[zero_fert];
  prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  assert(!isnan(prob));

  return prob;
}

//NOTE: the vectors need to be sorted
long double IBM5Trainer::distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const Storage1D<std::vector<ushort> >& aligned_source_words) const
{
  const uint J = source.size();

  long double prob = 1.0;

  uint prev_center = MAX_UINT;

  Storage1D<bool> fixed(J, false);

  uint nOpen = J;

  for (uint i = 1; i < aligned_source_words.size(); i++) {

    const std::vector<ushort>& cur_aligned_source_words = aligned_source_words[i];

    if (cur_aligned_source_words.size() > 0) {

      const uint first_j = cur_aligned_source_words[0];

      //a) head of the cept
      if (prev_center == MAX_UINT) {

        assert(cept_start_mode_ == IBM4UNIFORM || nOpen == J);

        if (cept_start_mode_ == IBM4UNIFORM)
          prob *= 1.0 / nOpen;
        else
          prob *= sentence_start_prob_[J][first_j];
      }
      else {

        const uint sclass = source_class_[source[first_j]];

        const uint nAvailable = nOpen - (cur_aligned_source_words.size() - 1);

        const Math3D::Tensor<double>& cur_inter_distortion = inter_distortion_prob_[nAvailable];

        uint pos_first_j = MAX_UINT;
        uint pos_prev_center = MAX_UINT;

        uint nCurOpen = 0;
        for (uint j = 0; j <= std::max(first_j, prev_center); j++) {

          if (j == first_j)
            pos_first_j = nCurOpen;

          if (!fixed[j])
            nCurOpen++;

          if (j == prev_center)
            pos_prev_center = nCurOpen;
        }

        //DEBUG
        if (pos_prev_center >= cur_inter_distortion.yDim()) {

          std::cerr << "J= " << J << ", prev_center=" << prev_center << ", pos_prev_center = " << pos_prev_center << std::endl;
          std::cerr << "fixed: " << fixed << std::endl;
        }
        //END_DEBUG

        assert(pos_prev_center < cur_inter_distortion.yDim());

        prob *= cur_inter_distortion(pos_first_j, pos_prev_center, sclass);
      }

      fixed[first_j] = true;
      nOpen--;

      //b) body of the cept
      for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

        const uint cur_j = cur_aligned_source_words[k];

        const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? source_class_[source[cur_j]] : target_class_[target[i - 1]];

        uint pos = MAX_UINT;

        uint nAvailable = 0;
        for (uint j = cur_aligned_source_words[k - 1] + 1; j < J; j++) {

          if (j == cur_j)
            pos = nAvailable;

          if (!fixed[j])
            nAvailable++;
        }

        nAvailable -= cur_aligned_source_words.size() - 1 - k;

        prob *= intra_distortion_prob_[nAvailable](pos, cur_class);

        fixed[cur_j] = true;
        nOpen--;
      }

      //c) calculate the center of the cept
      switch (cept_start_mode_) {
      case IBM4CENTER: {

        //compute the center of this cept and store the result in prev_cept_center
        double sum = vec_sum(cur_aligned_source_words);
        prev_center = (int)round(sum / cur_aligned_source_words.size());
        break;
      }
      case IBM4FIRST:
        prev_center = first_j;
        break;
      case IBM4LAST:
        prev_center = cur_aligned_source_words.back();
        break;
      case IBM4UNIFORM:
        break;
      default:
        assert(false);
      }
    }
  }

  assert(!isnan(prob));

  return prob;
}

void IBM5Trainer::init_from_prevmodel(FertilityModelTrainerBase* prev_model, const HmmWrapperBase* passed_wrapper, bool clear_prev,
                                      bool count_collection, bool viterbi)
{
  std::cerr << "******** initializing IBM-5 from " << prev_model->model_name() << " *******" << std::endl;

  if (count_collection) {

    best_known_alignment_ = prev_model->best_alignments();

    if (!viterbi)
      train_em(1, prev_model, passed_wrapper);
    else
      train_viterbi(1, prev_model, passed_wrapper);
  }
  else {

    best_known_alignment_ = prev_model->update_alignments_unconstrained(true, passed_wrapper);

    FertilityModelTrainer* fert_model = dynamic_cast<FertilityModelTrainer*>(prev_model);
    IBM4Trainer* ibm4 = dynamic_cast<IBM4Trainer*>(fert_model);

    if (use_sentence_start_prob_ && ibm4 != 0) {
      sentence_start_parameters_ = ibm4->sentence_start_parameters();
      par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
    }

    if (fert_model == 0) {
      init_fertilities(0);      //alignments were already updated and set
    }
    else {

      for (uint k = 1; k < fertility_prob_.size(); k++) {
        fertility_prob_[k] = fert_model->fertility_prob()[k];

        //EXPERIMENTAL
        for (uint l = 0; l < fertility_prob_[k].size(); l++) {
          if (l <= fertility_limit_[k])
            fertility_prob_[k][l] = 0.95 * std::max(fert_min_param_entry, fertility_prob_[k][l])
                                    + 0.05 / std::min<uint>(fertility_prob_[k].size(), fertility_limit_[k] + 1);
          else
            fertility_prob_[k][l] = 0.95 * fertility_prob_[k][l];
        }
        //END_EXPERIMENTAL
      }
    }

    if (!fix_p0_ && fert_model != 0) {
      p_zero_ = fert_model->p_zero();
      p_nonzero_ = 1.0 - p_zero_;
    }

    //init distortion models from best known alignments
    Storage1D<Math3D::Tensor<double> > inter_distortion_count = inter_distortion_prob_;
    Math2D::Matrix<double> single_diff_count = inter_distortion_param_;
    Math3D::Tensor<double> inter_dist_pospar_count(maxJ_+1,maxJ_+1,nSourceClasses_,0.0);

    Storage1D<Math2D::Matrix<double> > intra_distortion_count = intra_distortion_prob_;
    Math2D::Matrix<double> intra_distparam_count = intra_distortion_param_;

    /*** clear counts ***/
    for (uint J = 1; J < inter_distortion_count.size(); J++)
      inter_distortion_count[J].set_constant(0.0);
    single_diff_count.set_constant(0.0);

    if (!uniform_intra_prob_) {
      for (uint J = 1; J < intra_distortion_count.size(); J++)
        intra_distortion_count[J].set_constant(0.0);
      intra_distparam_count.set_constant(0.0);
    }

    //TODO: also estimate sentence_start_prob if ibm4==0

    for (size_t s = 0; s < source_sentence_.size(); s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];

      const Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      NamedStorage1D<std::vector<ushort> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));

      for (uint j = 0; j < curJ; j++) {
        const uint aj = cur_alignment[j];
        aligned_source_words[aj].push_back(j);
      }

      {
        uint prev_center = MAX_UINT;

        Storage1D<bool> fixed(curJ, false);

        uint nOpen = curJ;

        for (uint i = 1; i <= curI; i++) {

          if (aligned_source_words[i].size() > 0) {

            const std::vector<ushort>& cur_aligned_source_words = aligned_source_words[i];

            //a) head of the cept
            const uint first_j = cur_aligned_source_words[0];

            if (prev_center != MAX_UINT) {      // currently not estimating a start probability

              const uint sclass = source_class_[cur_source[first_j]];

              const uint nAvailable = nOpen - (cur_aligned_source_words.size() - 1);

              Math3D::Tensor<double>& cur_inter_distortion_count = inter_distortion_count[nAvailable];

              uint pos_first_j = MAX_UINT;
              uint pos_prev_center = MAX_UINT;

              uint nCurOpen = 0;
              for (uint j = 0; j <= std::max(first_j, prev_center); j++) {

                if (j == first_j)
                  pos_first_j = nCurOpen;

                if (!fixed[j])
                  nCurOpen++;

                if (j == prev_center)
                  pos_prev_center = nCurOpen;
              }

              cur_inter_distortion_count(pos_first_j, pos_prev_center, sclass) += 1.0;
			  if (distortion_type_ == IBM23ParByDifference)
                single_diff_count(displacement_offset_ + pos_first_j - pos_prev_center, sclass) += 1.0;
              else
				inter_dist_pospar_count(pos_first_j, pos_prev_center, sclass) += 1.0;
			}

            fixed[first_j] = true;
            nOpen--;

            //b) body of the cept
            if (!uniform_intra_prob_) {
              for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

                const uint cur_j = cur_aligned_source_words[k];

                const uint sclass = source_class_[cur_source[cur_j]];
				const uint tclass = target_class_[cur_target[i-1]];
				const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass : tclass;

                uint pos = MAX_UINT;

                uint nAvailable = 0;
                for (uint j = cur_aligned_source_words[k - 1] + 1; j < curJ; j++) {

                  if (j == cur_j)
                    pos = nAvailable;

                  if (!fixed[j])
                    nAvailable++;
                }

                nAvailable -= cur_aligned_source_words.size() - 1 - k;

                intra_distortion_count[nAvailable](pos, cur_class) += 1.0;
                intra_distparam_count(pos, cur_class) += 1.0;

                fixed[cur_j] = true;
                nOpen--;
              }
            }

            //c) calculate the center of the cept
            switch (cept_start_mode_) {
            case IBM4CENTER: {

              //compute the center of this cept and store the result in prev_cept_center
              double sum = vec_sum(cur_aligned_source_words);
              prev_center = (int)round(sum / cur_aligned_source_words.size());
              break;
            }
            case IBM4FIRST:
              prev_center = first_j;
              break;
            case IBM4LAST:
              prev_center = cur_aligned_source_words.back();
              break;
            case IBM4UNIFORM:
              break;
            default:
              assert(false);
            }
          }
        }
      }
    }

    //std::cerr << "A" << std::endl;
	
    //update inter distortion probabilities
    if (distortion_type_ == IBM23Nonpar) {

	  const uint nDistGroupingTerms = maxJ_+1 - (dist_grouping_limit_+1);

      for (uint J = 1; J < inter_distortion_count.size(); J++) {

		//std::cerr << "J: " << J << std::endl;
        for (uint y = 0; y < inter_distortion_count[J].yDim(); y++) {
          for (uint z = 0; z < inter_distortion_count[J].zDim(); z++) {

            double sum = 0.0;
            for (uint j = 0; j < inter_distortion_count[J].xDim(); j++)
              sum += inter_distortion_count[J](j, y, z);

            if (sum > 1e-305) {
				
			  if (inter_dist_grouping_ == DistGroupModeOff) {
                for (uint j = 0; j < inter_distortion_count[J].xDim(); j++)
                  inter_distortion_prob_[J](j, y, z) =
                    0.95 * std::max(fert_min_param_entry, inter_distortion_count[J](j, y, z) / sum)
                    + 0.05 * inter_distortion_prob_[J](j, y, z); 
			  }
			  else {
				  
				double longparam_sum = 0.0;
				for (uint j = dist_grouping_limit_+1; j < inter_distortion_count[J].xDim();  j++)
				  longparam_sum += inter_distortion_count[J](j, y, z);
			    if (inter_dist_grouping_ == DistGroupModeDivide) {
  			      sum -= longparam_sum;
				  longparam_sum /= nDistGroupingTerms;
				  sum += longparam_sum;
				}
			  
			    for (uint j = 0; j < inter_distortion_count[J].xDim(); j++)
				  inter_distortion_prob_[J](j, y, z) = std::max(fert_min_param_entry, inter_distortion_count[J](j, y, z) / sum);
			    const uint nGroup = inter_distortion_count[J].xDim() - (dist_grouping_limit_+1);
			    for (uint j = dist_grouping_limit_+1; j < inter_distortion_count[J].xDim();  j++)
				  inter_distortion_prob_[J](j, y, z) = longparam_sum / (sum * nGroup);
			  }
			}
          }
        }
      }
    }
    else if (distortion_type_ == IBM23ParByDifference) {

	  const uint nDistGroupingTerms = inter_distortion_param_.xDim() - (2*dist_grouping_limit_-1);

      for (uint s = 0; s < inter_distortion_param_.yDim(); s++) {

		if (inter_dist_grouping_ != DistGroupModeOff) {
			
		  double sum = 0.0;
		  double grouping_param = 0.0;
		  for (int d = 0; d < inter_distortion_param_.xDim(); d++) {
			if (abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
			  sum += single_diff_count(d, s);
		    else 
			  grouping_param += single_diff_count(d, s);
		  }
		  sum += grouping_param;
		  
		  if (sum > 1e-50) {
			double inv_sum = 1.0 / sum;
			for (int d = 0; d < inter_distortion_param_.xDim(); d++) {
			  if (abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
				inter_distortion_param_(d, s) =
					0.95 * std::max(fert_min_param_entry, single_diff_count(d, s)* inv_sum) + 0.05 * inter_distortion_param_(d, s);  
			  else {
				inter_distortion_param_(d, s) = std::max(fert_min_param_entry, inv_sum * grouping_param) / nDistGroupingTerms;
			  }
			}
			inter_grouping_prob_[s] = std::max(fert_min_param_entry, inv_sum * grouping_param);
		  }
		}
		else { // dist-grouping off
          double sum = 0.0;
          for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
            sum += single_diff_count(j, s);

          assert(sum > 1e-305);
		  assert(inter_distortion_param_.row_sum(s) >= 0.99 && inter_distortion_param_.row_sum(s) <= 1.01);

          for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
            inter_distortion_param_(j, s) =
              0.95 * std::max(fert_min_param_entry, single_diff_count(j, s) / sum) + 0.05 * inter_distortion_param_(j, s);
		  assert(inter_distortion_param_.row_sum(s) >= 0.99 && inter_distortion_param_.row_sum(s) <= 1.01);
		}
	  }

      par2nonpar_inter_distortion();
    }
	else { //parbypos
		
	  const uint nDistGroupingTerms = maxJ_+1 - (dist_grouping_limit_+1);

      for (uint s = 0; s < inter_distortion_param_.yDim(); s++) {
		  
		for (uint prev_pos = 0; prev_pos <= maxJ_; prev_pos++) {

 		  if (inter_dist_grouping_ != DistGroupModeOff) {

		    double sum = 0.0;
		    double grouping_count = 0.0;
		    for (uint j = 0; j < inter_dist_pospar_param_.xDim(); j++) {
			  if (j <= dist_grouping_limit_)
			    sum += inter_dist_pospar_count(j, prev_pos, s);
		      else
			    grouping_count += inter_dist_pospar_count(j, prev_pos, s);
		    }
		    sum += grouping_count;
		  
		    if (sum > 1e-300) {
		      double inv_sum = 1.0 / sum;
		      inter_pospar_grouping_prob_(prev_pos,s) = std::max(fert_min_param_entry, inv_sum * grouping_count);
		      for (uint j = 0; j < inter_dist_pospar_param_.xDim(); j++) {
		        if (j <= dist_grouping_limit_)
				  inter_dist_pospar_param_(j, prev_pos, s) = std::max(fert_min_param_entry, inv_sum * inter_dist_pospar_count(j, prev_pos, s));
			    else
				  inter_dist_pospar_param_(j, prev_pos, s) = std::max(fert_min_param_entry,inv_sum * grouping_count) / nDistGroupingTerms;
			  }
			  inter_pospar_grouping_prob_(prev_pos, s) = std::max(fert_min_param_entry, inv_sum * grouping_count);
		    }
		  }
	 	  else {
		    //dist-grouping off
		    double sum = 0.0;
		    for (uint d = 0; d < inter_dist_pospar_count.xDim(); d++)
			  sum += inter_dist_pospar_count(d, prev_pos, s);
		    if (sum > 1e-300) {
			  double inv_sum = 1.0 / sum;
			  for (uint d = 0; d < inter_dist_pospar_count.xDim(); d++)
			   inter_dist_pospar_param_(d, prev_pos, s) = 0.95 * std::max(inv_sum * inter_dist_pospar_count(d, prev_pos, s), fert_min_param_entry)
				  + 0.05 / inter_dist_pospar_param_.xDim();
			}
		  }
		}
	  }
	  
	  par2nonpar_inter_distortion();
	}

    //std::cerr << "B" << std::endl;

    //update intra distortion probabilities
    if (distortion_type_ == IBM23Nonpar) {

      for (uint J = 1; J < intra_distortion_prob_.size(); J++) {

        for (uint s = 0; s < intra_distortion_prob_[J].yDim(); s++) {

          double sum = 0.0;
          for (uint j = 0; j < J; j++)
            sum += intra_distortion_count[J] (j, s);

          if (sum > 1e-305) {
            for (uint j = 0; j < J; j++)
              intra_distortion_prob_[J](j, s) =
                0.95 * std::max(fert_min_param_entry, intra_distortion_count[J] (j, s) / sum) + 0.05 * intra_distortion_prob_[J] (j, s);
          }
        }
      }
    }
    else {

      for (uint s = 0; s < intra_distparam_count.yDim(); s++) {

        double sum = 0.0;
        for (uint j = 0; j < intra_distparam_count.xDim(); j++)
          sum += intra_distparam_count(j, s);

        assert(sum > 1e-305);

        for (uint j = 0; j < intra_distparam_count.xDim(); j++)
          intra_distortion_param_(j, s) =
            0.95 * std::max(fert_min_param_entry, intra_distparam_count(j, s) / sum)  + 0.05 * intra_distortion_param_(j, s);
      }

      par2nonpar_intra_distortion();
    }
  }

  if (clear_prev)
    prev_model->release_memory();
}

long double IBM5Trainer::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, const SingleLookupTable& lookup,
    uint& nIter, Math1D::Vector<uint>& fertility, Math2D::Matrix<long double>& expansion_prob,
    Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment) const
{
  /**** calculate probability of the passed alignment *****/

  double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size();

  Storage1D<std::vector<ushort> > aligned_source_words(curI + 1);
  for (uint i = 0; i <= curI; i++)
    aligned_source_words[i].reserve(10);

  fertility.set_constant(0);

  for (uint j = 0; j < curJ; j++) {

    const uint aj = alignment[j];
    aligned_source_words[aj].push_back(j);
    fertility[aj]++;
  }

  Math2D::Matrix<double> dict(curJ,curI+1);
  compute_dictmat_fertform(source, lookup, target, dict_, dict);

  long double base_distortion_prob = distortion_prob(source, target, aligned_source_words);
  long double base_prob = base_distortion_prob;

  for (uint i = 1; i <= curI; i++) {
    uint t_idx = target[i - 1];
    //NOTE: no factorial here (nondeficient!)
    base_prob *= fertility_prob_[t_idx][fertility[i]];
  }
  for (uint j = 0; j < curJ; j++) {

    uint aj = alignment[j];
    base_prob *= dict(j,aj);
  }

  uint zero_fert = fertility[0];
  update_nullpow(zero_fert, curJ - 2 * zero_fert);
  base_prob *= choose_factor_[curJ][zero_fert];
  base_prob *= p_zero_pow_[zero_fert];
  base_prob *= p_nonzero_pow_[curJ - 2 * zero_fert];

  //DEBUG
#ifndef NDEBUG
  assert(!isnan(base_prob));
  long double check_prob = alignment_prob(source, target, lookup, alignment);
  long double check_ratio = base_prob / check_prob;
  assert(check_ratio >= 0.99 && check_ratio <= 1.01);
#endif
  //END_DEBUG

  uint count_iter = 0;

  Storage1D<std::vector<ushort> > hyp_aligned_source_words = aligned_source_words;

  swap_prob.resize(curJ, curJ);
  expansion_prob.resize(curJ, curI + 1);
  //swap_prob.set_constant(0.0);
  //expansion_prob.set_constant(0.0);

  long double empty_word_increase_const = 0.0;
  long double empty_word_decrease_const = 0.0;

  while (true) {

    count_iter++;
    nIter++;

    //std::cerr << "****************** starting new nondef hc iteration, current best prob: " << base_prob << std::endl;

    if (empty_word_increase_const == 0.0 && p_zero_ > 0.0) {
      if (curJ >= 2 * (zero_fert + 1)) {

        empty_word_increase_const = (curJ - 2 * zero_fert) * (curJ - 2 * zero_fert - 1) * p_zero_
                                    / ((curJ - zero_fert) * (zero_fert + 1) * p_nonzero_ * p_nonzero_);

#ifndef NDEBUG
        long double old_const =
          ldchoose(curJ - zero_fert - 1, zero_fert + 1) * p_zero_ / (ldchoose(curJ - zero_fert, zero_fert) * p_nonzero_ * p_nonzero_);

        long double ratio = empty_word_increase_const / old_const;

        assert(ratio >= 0.975 && ratio <= 1.05);
#endif
      }
    }

    if (empty_word_decrease_const == 0.0 && zero_fert > 0) {

      empty_word_decrease_const = (curJ - zero_fert + 1) * zero_fert * p_nonzero_ * p_nonzero_
                                  / ((curJ - 2 * zero_fert + 1) * (curJ - 2 * zero_fert + 2) * p_zero_);

#ifndef NDEBUG
      long double old_const = ldchoose(curJ - zero_fert + 1, zero_fert - 1) * p_nonzero_ * p_nonzero_
                              / (ldchoose(curJ - zero_fert, zero_fert) * p_zero_);

      long double ratio = empty_word_decrease_const / old_const;

      assert(ratio >= 0.975 && ratio <= 1.05);
#endif
    }

    long double best_prob = base_prob;
    bool best_change_is_move = false;
    uint best_move_j = MAX_UINT;
    uint best_move_aj = MAX_UINT;
    uint best_swap_j1 = MAX_UINT;
    uint best_swap_j2 = MAX_UINT;

    /**** scan neighboring alignments and keep track of the best one that is better
     ****  than the current alignment  ****/

    /**** expansion moves ****/

    for (uint j = 0; j < curJ; j++) {

      //std::cerr << "j: " << j << std::endl;

      const uint s_idx = source[j];
      const uint aj = alignment[j];
      const uint tidx = (aj > 0) ? target[aj - 1] : 0;
      const Math1D::Vector<double>& fert_prob_aj = fertility_prob_[tidx];

      const uint aj_fert = fertility[aj];
      expansion_prob(j, aj) = 0.0;

      vec_erase(hyp_aligned_source_words[aj], (ushort) j);

      const double old_dict_prob = dict(j,aj);

      long double incremental_prob_common = base_prob / (base_distortion_prob * old_dict_prob);
      if (aj > 0)
        incremental_prob_common *= fert_prob_aj[aj_fert - 1] / fert_prob_aj[aj_fert];
      else
        incremental_prob_common *= empty_word_decrease_const;

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {

        if (aj == cand_aj) {
          expansion_prob(j, cand_aj) = 0.0;
          continue;
        }

        const uint t_idx = (cand_aj > 0) ? target[cand_aj - 1] : 0;
        const Math1D::Vector<double>& fert_prob_t_idx = fertility_prob_[t_idx];
        const uint cand_aj_fert = fertility[cand_aj];

        if (cand_aj > 0 && ((cand_aj_fert + 1) > fertility_limit_[t_idx])) {
          expansion_prob(j, cand_aj) = 0.0;
          continue;
        }
        if (cand_aj == 0 && 2 * fertility[0] + 2 > curJ) {      //better to check this before computing distortion probs
          expansion_prob(j, cand_aj) = 0.0;
          continue;
        }

        const double new_dict_prob = dict(j,cand_aj);

        if (new_dict_prob < 1e-8)
          expansion_prob(j, cand_aj) = 0.0;
        else {
          //hyp_aligned_source_words[cand_aj].push_back(j);
          //vec_sort(hyp_aligned_source_words[cand_aj]);
          sorted_vec_insert<ushort>(hyp_aligned_source_words[cand_aj], j);

          long double incremental_prob = incremental_prob_common;
          incremental_prob *= distortion_prob(source, target, hyp_aligned_source_words) * new_dict_prob;

          if (cand_aj > 0) {
            incremental_prob *= fert_prob_t_idx[cand_aj_fert + 1] / fert_prob_t_idx[cand_aj_fert];
          }
          else {
            if (curJ < 2 * zero_fert + 2) {
              incremental_prob = 0.0;
            }
            else {
              //compute null-fert-model (zero-fert goes up by 1)
              incremental_prob *= empty_word_increase_const;
            }
          }

#ifndef NDEBUG
          Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
          hyp_alignment[j] = cand_aj;
          long double cand_prob = alignment_prob(source, target, lookup, hyp_alignment);

          if (cand_prob > 1e-250) {

            double ratio = incremental_prob / cand_prob;

            if (!(ratio >= 0.99 && ratio <= 1.01)) {
              std::cerr << "j: " << j << ", aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
              std::cerr << "incremental: " << incremental_prob << ", standalone: " << cand_prob << std::endl;
            }
            assert(ratio >= 0.99 && ratio <= 1.01);
          }
#endif

          expansion_prob(j, cand_aj) = incremental_prob;

          if (incremental_prob > best_prob) {
            best_change_is_move = true;
            best_prob = incremental_prob;
            best_move_j = j;
            best_move_aj = cand_aj;
          }
          //restore for the next iteration
          hyp_aligned_source_words[cand_aj] = aligned_source_words[cand_aj];
        }
      }

      hyp_aligned_source_words[aj] = aligned_source_words[aj];
    }

    /**** swap moves ****/
    for (uint j1 = 0; j1 < curJ; j1++) {

      //std::cerr << "j1: " << j1 << std::endl;

      const uint aj1 = alignment[j1];

      const long double base_prob_common = base_prob / (base_distortion_prob * dict(j1,aj1));

      for (uint j2 = j1 + 1; j2 < curJ; j2++) {

        //std::cerr << "j2: " << j2 << std::endl;

        const uint aj2 = alignment[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1, j2) = 0.0;
        }
        else {

          //vec_replace(hyp_aligned_source_words[aj2], (ushort) j2, (ushort) j1);
          //vec_replace(hyp_aligned_source_words[aj1], (ushort) j1, (ushort) j2);
          //vec_sort(hyp_aligned_source_words[aj1]);
          //vec_sort(hyp_aligned_source_words[aj2]);

          vec_replace_maintainsort<ushort>(hyp_aligned_source_words[aj2], j2, j1);
          vec_replace_maintainsort<ushort>(hyp_aligned_source_words[aj1], j1, j2);

          long double incremental_prob = base_prob_common * distortion_prob(source, target, hyp_aligned_source_words);
          incremental_prob *= dict(j2,aj1);
          incremental_prob *= dict(j1,aj2) / dict(j2,aj2);

#ifndef NDEBUG
          Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
          std::swap(hyp_alignment[j1], hyp_alignment[j2]);
          long double cand_prob = alignment_prob(source, target, lookup, hyp_alignment);

          if (cand_prob > 1e-250) {

            double ratio = cand_prob / incremental_prob;
            assert(ratio > 0.99 && ratio < 1.01);
          }
#endif

          swap_prob(j1, j2) = incremental_prob;

          if (incremental_prob > best_prob) {
            best_change_is_move = false;
            best_prob = incremental_prob;
            best_swap_j1 = j1;
            best_swap_j2 = j2;
          }
          //restore for the next iteration
          hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
          hyp_aligned_source_words[aj2] = aligned_source_words[aj2];
        }
      }
    }

    /**** update to best alignment ****/

    if (best_prob < improvement_factor * base_prob || count_iter > nMaxHCIter_) {
      if (count_iter > nMaxHCIter_)
        std::cerr << "HC Iteration limit reached" << std::endl;
      break;
    }
    //update alignment
    if (best_change_is_move) {
      const uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      //std::cerr << "moving source pos" << best_move_j << " from " << cur_aj << " to " << best_move_aj << std::endl;

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;

      if (cur_aj * best_move_aj == 0) {
        //signal recomputation
        zero_fert = fertility[0];
        empty_word_increase_const = 0.0;
        empty_word_decrease_const = 0.0;
      }

      vec_erase<ushort>(aligned_source_words[cur_aj], best_move_j);
      sorted_vec_insert<ushort>(aligned_source_words[best_move_aj], best_move_j);

      hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
      hyp_aligned_source_words[best_move_aj] = aligned_source_words[best_move_aj];
    }
    else {
      //std::cerr << "swapping: j1=" << best_swap_j1 << std::endl;
      //std::cerr << "swapping: j2=" << best_swap_j2 << std::endl;

      uint cur_aj1 = alignment[best_swap_j1];
      uint cur_aj2 = alignment[best_swap_j2];

      //std::cerr << "old aj1: " << cur_aj1 << std::endl;
      //std::cerr << "old aj2: " << cur_aj2 << std::endl;

      assert(cur_aj1 != cur_aj2);

      alignment[best_swap_j1] = cur_aj2;
      alignment[best_swap_j2] = cur_aj1;

      vec_replace_maintainsort<ushort>(aligned_source_words[cur_aj2], best_swap_j2, best_swap_j1);
      vec_replace_maintainsort<ushort>(aligned_source_words[cur_aj1], best_swap_j1, best_swap_j2);

      hyp_aligned_source_words[cur_aj1] = aligned_source_words[cur_aj1];
      hyp_aligned_source_words[cur_aj2] = aligned_source_words[cur_aj2];
    }

    base_prob = best_prob;
    base_distortion_prob = distortion_prob(source, target, aligned_source_words);
  }

  //symmetrize swap_prob
  symmetrize_swapmat(swap_prob, curJ);

  return base_prob;
}

double IBM5Trainer::inter_distortion_diffpar_m_step_energy(const Math2D::Matrix<double>& single_diff_count, 
														   const Math3D::Tensor<double>& diff_span_count,
														   uint sclass, const Math2D::Matrix<double>& param,
														   double grouping_prob) const
{
  double energy = 0.0;

  for (int d = 0; d < param.xDim(); d++) {
    if (inter_dist_grouping_ == DistGroupModeOff || abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
	  energy -= single_diff_count(d, sclass) * std::log(std::max(fert_min_param_entry, param(d, sclass)));
    else
	  energy -= single_diff_count(d, sclass) * std::log(grouping_prob);
  }
  
  const uint nDistGroupingTerms = param.xDim() - (2*dist_grouping_limit_+1);

  //for (uint d_start = 0; d_start < diff_span_count.xDim(); d_start++) {
  for (uint d_start = 0; d_start < std::min<uint>(displacement_offset_+1,diff_span_count.xDim()); d_start++) {

    const uint yDim = diff_span_count.yDim();

    //double param_sum = 0.0;
    for (uint d_end = d_start; d_end < yDim; d_end++) {
    
	  
      //param_sum += std::max(fert_min_param_entry, param(d_end, sclass));
      const double count = diff_span_count(d_start, d_end, sclass);
      if (count != 0.0) {

	    double param_sum = 0.0;
	    for (int d = d_start; d <= d_end; d++) {
		  if (inter_dist_grouping_ == DistGroupModeOff || abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
			param_sum += param(d,sclass);
		  else if (inter_dist_grouping_ == DistGroupModeDivide && abs(d - (int) displacement_offset_) > dist_grouping_limit_)
			param_sum += grouping_prob / nDistGroupingTerms;
	    } 
		if (inter_dist_grouping_ == DistGroupModeExpand && 
			(d_start < displacement_offset_-dist_grouping_limit_ || d_end > displacement_offset_ + dist_grouping_limit_))
		  param_sum += grouping_prob;
        
		energy += count * std::log(param_sum);
      }
    }
  }

  return energy;
}

double IBM5Trainer::inter_distortion_pospar_m_step_energy(const Math3D::Tensor<double>& single_pos_count, const Math3D::Tensor<double>& pos_span_count,
														  uint sclass, uint prev_center, const Math3D::Tensor<double>& param, double grouping_prob) const
{
  const uint nDistGroupingTerms = maxJ_+1 - (dist_grouping_limit_+1);
  double energy = 0.0;
  
  for (int d = 0; d < param.xDim(); d++) {
    if (inter_dist_grouping_ == DistGroupModeOff || abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
	  energy -= single_pos_count(d, prev_center, sclass) * std::log(std::max(fert_min_param_entry, param(d, prev_center, sclass)));
    else
	  energy -= single_pos_count(d, prev_center, sclass) * std::log(grouping_prob);
  }

  double param_sum = 0.0;
  uint nLong = 0;
  for (uint d = 0; d < pos_span_count.xDim(); d++) {
	
	if (inter_dist_grouping_ == DistGroupModeOff || d <= dist_grouping_limit_)
	  param_sum += param(d, prev_center, sclass);
    else if (inter_dist_grouping_ == DistGroupModeDivide)
	  param_sum += grouping_prob / nDistGroupingTerms;
    else {
	  assert(inter_dist_grouping_ == DistGroupModeExpand);
	  nLong++;
	}
  
    double full_sum = param_sum;
    if (nLong > 0)
      full_sum += grouping_prob;	

	energy += pos_span_count(d, prev_center, sclass) * std::log(full_sum); 
  }	  
  
  return energy;
}


void IBM5Trainer::inter_distortion_diffpar_m_step(const Math2D::Matrix<double>& single_diff_count, const Math3D::Tensor<double>& diff_span_count, 
												  uint sclass)
{
  assert(distortion_type_ == IBM23ParByDifference);
  const uint nParams = inter_distortion_param_.xDim();
  const int nDistGroupingTerms = nParams - (2*dist_grouping_limit_+1);

  for (uint k = 0; k < nParams; k++)
    inter_distortion_param_(k, sclass) = std::max(fert_min_param_entry, inter_distortion_param_(k, sclass));

  Math1D::Vector<double> gradient(inter_distortion_param_.xDim());
  Math1D::Vector<double> new_param(inter_distortion_param_.xDim());
  Math2D::Matrix<double> hyp_param = inter_distortion_param_;

  double hyp_grouping_prob = 0.0;
  double new_grouping_prob = 0.0;

  double energy = (deficient_) ? 0.0 : inter_distortion_diffpar_m_step_energy(single_diff_count, diff_span_count, sclass, 
																			  inter_distortion_param_, inter_grouping_prob_[sclass]);

  //check if normalized expectations give a better starting point
  if (inter_dist_grouping_ != DistGroupModeOff) {
	hyp_grouping_prob = 0.0;
	double sum = 0.0;
	
	for (int d = 0; d < nParams; d++)
	{
	  if (abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
		sum += single_diff_count(d, sclass);
	  else 
	    hyp_grouping_prob += single_diff_count(d, sclass);
	}
	hyp_grouping_prob = std::max(fert_min_param_entry, hyp_grouping_prob);
	sum += hyp_grouping_prob;

	if (sum > 1e-50) {
      for (int d = 0; d < nParams; d++) {
	    if (abs(d - (int) displacement_offset_) <= dist_grouping_limit_)		  
          hyp_param(d, sclass) = std::max(fert_min_param_entry, single_diff_count(d, sclass) / sum);
	    else
		  hyp_param(d, sclass) = std::max(fert_min_param_entry, hyp_grouping_prob / (sum * nDistGroupingTerms));
	  }
	  hyp_grouping_prob = std::max(fert_min_param_entry, hyp_grouping_prob / sum);
	 
      double hyp_energy = (deficient_) ? -1.0 
		: inter_distortion_diffpar_m_step_energy(single_diff_count, diff_span_count, sclass, hyp_param, hyp_grouping_prob);

      //NOTE: the closed-form deficient solution does not necessarily reduce the energy INCLUDING the normalization term
      if (hyp_energy < energy) {

        for (uint d = 0; d < nParams; d++)
          inter_distortion_param_(d, sclass) = hyp_param(d, sclass);
	    inter_grouping_prob_[sclass] = hyp_grouping_prob;

        energy = hyp_energy;
      }
	}
  }
  else {
    
    const double sum = single_diff_count.row_sum(sclass);

	if (sum > 1e-50) {
      for (uint d = 0; d < nParams; d++)
        hyp_param(d, sclass) = std::max(fert_min_param_entry, single_diff_count(d, sclass) / sum);

      double hyp_energy = (deficient_) ? -1.0 : inter_distortion_diffpar_m_step_energy(single_diff_count, diff_span_count, sclass, 
																					   hyp_param, hyp_grouping_prob);

      //NOTE: the closed-form deficient solution does not necessarily reduce the energy INCLUDING the normalization term
      if (hyp_energy < energy) {

        for (uint d = 0; d < nParams; d++)
          inter_distortion_param_(d, sclass) = hyp_param(d, sclass);

        energy = hyp_energy;
	  }
	}
  }

  if (deficient_)
    return;

  if (nSourceClasses_ <= 4)
    std::cerr << "inter m-step start energy: " << energy << std::endl;

  double alpha = gd_stepsize_;
  double line_reduction_factor = 0.1;

  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {

    if ((iter % 5) == 0) {

      if (nSourceClasses_ <= 4)
        std::cerr << "inter m-step iteration #" << iter << ", energy: " << energy << std::endl;
    }

    gradient.set_constant(0.0);
	double grouping_param_grad = 0.0;

    /*** 1. calculate gradient ***/

	if (inter_dist_grouping_ != DistGroupModeOff) {
		
	  //a) singleton terms
      for (int d = 0; d < nParams; d++) {
		if (abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
          gradient[d] -= single_diff_count(d, sclass) / std::max(fert_min_param_entry, inter_distortion_param_(d, sclass));
		else
		  grouping_param_grad -= single_diff_count(d, sclass) / inter_grouping_prob_[sclass];
	  }

	  //b) normalization terms
      //for (uint d_start = 0; d_start < diff_span_count.xDim(); d_start++) {
      for (uint d_start = 0; d_start < std::min<uint>(displacement_offset_+1,diff_span_count.xDim()); d_start++) {
 
        //double param_sum = 0.0;
        for (int d_end = d_start; d_end < diff_span_count.yDim(); d_end++) {

		  const double count = diff_span_count(d_start,d_end,sclass);
		  if (count == 0.0)
			continue;
		  
		  double param_sum = 0.0;
		  uint nLong = 0;
		  for (int d = d_start; d <= d_end; d++) {
			if (distortion_type_ == DistGroupModeOff || abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
			  param_sum += std::max(fert_min_param_entry, inter_distortion_param_(d, sclass));
		    else if (distortion_type_ == DistGroupModeDivide)
			  param_sum += inter_grouping_prob_[sclass] / nDistGroupingTerms;
		    else if (distortion_type_ == DistGroupModeExpand)
			  nLong++;
		  }
		  if (nLong > 0) {
			assert(distortion_type_ == DistGroupModeExpand);
			param_sum += inter_grouping_prob_[sclass];
		  }
		  
		  for (int d = d_start; d <= d_end; d++) {
			if (abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
			  gradient[d] += count / param_sum;
		    else if (distortion_type_ == DistGroupModeDivide)
			  grouping_param_grad += count / (nDistGroupingTerms * param_sum);
		  }
		  if (nLong > 0) {
			assert(distortion_type_ == DistGroupModeExpand);
			grouping_param_grad += count / param_sum;
		  }

		  // if (abs(d_end - (int) displacement_offset_) <= dist_grouping_limit_)
            // param_sum += std::max(fert_min_param_entry, inter_distortion_param_(d_end, sclass));
		  // else
			// param_sum += grouping_param / nDistGroupingTerms;
		
		  // double addon = diff_span_count(d_start,d_end,sclass) / param_sum;
          // if (addon == 0.0)
			// continue;
		  // for (int d=d_start; d <= d_end; d++) {
            // if (abs(d - (int) displacement_offset_) <= dist_grouping_limit_)
			  // gradient[d] += addon;
			// else 
			  // grouping_param_grad += addon / nDistGroupingTerms;
		  // }
		}
	  }		
	}
	else { //dist grouping off
	
	  // a) singleton terms
      for (uint d = 0; d < nParams; d++)
        gradient[d] -= single_diff_count(d, sclass) / std::max(fert_min_param_entry, inter_distortion_param_(d, sclass));

	  // b) normalization terms
      Math1D::Vector<double> addon(diff_span_count.yDim());

      //for (uint d_start = 0; d_start < diff_span_count.xDim(); d_start++) {
      for (uint d_start = 0; d_start < std::min<uint>(displacement_offset_+1,diff_span_count.xDim()); d_start++) {
		  
        double param_sum = 0.0;
        for (uint d_end = d_start; d_end < diff_span_count.yDim(); d_end++) {
        
          param_sum += std::max(fert_min_param_entry, inter_distortion_param_(d_end, sclass));

          addon[d_end] = diff_span_count(d_start, d_end, sclass) / param_sum;
          // double addon = diff_span_count(d_start,d_end,sclass) / param_sum;

          // for (uint d=d_start; d <= d_end; d++)
          //   gradient[d] += addon;
        }

        double addon_sum = 0.0;
        for (int d = diff_span_count.yDim() - 1; d >= int (d_start); d--) {

          addon_sum += addon[d];
          gradient[d] += addon_sum;
		}
	  }
    }

    /*** 2. go in neg. gradient direction ***/
    double sqr_grad_norm = gradient.sqr_norm() + grouping_param_grad*grouping_param_grad;
    if (sqr_grad_norm < 1e-5) {
      std::cerr << "CUTOFF after " << iter << "iterations because squared gradient norm was " << sqr_grad_norm << std::endl;
      break;
    }

    double real_alpha = alpha / sqrt(sqr_grad_norm);
	
	Math1D::Vector<double> temp(2*dist_grouping_limit_+1);
	double new_grouping_param = inter_grouping_prob_[sclass];
	
	if (inter_dist_grouping_ != DistGroupModeOff) {
	
	  new_grouping_param -= real_alpha * grouping_param_grad;
	  for (int d = -dist_grouping_limit_; d <= dist_grouping_limit_; d++)
		temp[d+dist_grouping_limit_] = inter_distortion_param_(d+displacement_offset_,sclass) 
			- real_alpha * gradient[d+displacement_offset_];
	}
	else {
	  for (uint j = 0; j < nParams; j++)
        new_param[j] = inter_distortion_param_(j, sclass) - real_alpha * gradient[j];
	}

    /*** 3. reproject ***/
	if (inter_dist_grouping_ != DistGroupModeOff) {
	  projection_on_simplex_with_slack(temp.direct_access(), new_grouping_param, temp.size(), fert_min_param_entry);
	  new_param.set_constant(std::max(fert_min_param_entry, new_grouping_param / nDistGroupingTerms));
	  for (int d = -dist_grouping_limit_; d <= dist_grouping_limit_; d++)
	    new_param[d+displacement_offset_] = temp[d+dist_grouping_limit_]; 
	}
	else {
      projection_on_simplex(new_param, fert_min_param_entry);
	}

    /*** 4. find appropriate step size ***/
    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nTrials = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nTrials++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      for (uint j = 0; j < nParams; j++)
        hyp_param(j, sclass) = lambda * new_param[j] + neg_lambda * inter_distortion_param_(j,sclass);
	  hyp_grouping_prob = lambda * new_grouping_param + neg_lambda * inter_grouping_prob_[sclass];

      double hyp_energy = inter_distortion_diffpar_m_step_energy(single_diff_count, diff_span_count, sclass, hyp_param, hyp_grouping_prob);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nTrials > 5 && best_energy < 0.975 * energy)
        break;

      if (nTrials > 25 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;

    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nTrials > 6)
      line_reduction_factor *= 0.9;

    //EXPERIMENTAL
    // if (nIter > 4)
    //   alpha *= 1.5;
    //END_EXPERIMENTAL

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint j = 0; j < nParams; j++)
      inter_distortion_param_(j, sclass) = best_lambda * new_param[j] + neg_best_lambda * inter_distortion_param_(j, sclass);
    inter_grouping_prob_[sclass] = best_lambda * new_grouping_param + neg_best_lambda * inter_grouping_prob_[sclass];


    energy = best_energy;
  }
}

void IBM5Trainer::inter_distortion_pospar_m_step(const Math3D::Tensor<double>& single_pos_count, const Math3D::Tensor<double>& pos_span_count,
												 uint sclass, uint prev_pos)
{
  const uint nPosParams = (inter_dist_grouping_ == DistGroupModeOff) ? maxJ_+1 : dist_grouping_limit_+1;	
  const uint nDistGroupingTerms = maxJ_+1 - nPosParams;
	
  inter_dist_pospar_param_.ensure_min(fert_min_param_entry);
  inter_pospar_grouping_prob_.ensure_min(fert_min_param_entry);  
  Math3D::Tensor<double> hyp_dist_param = inter_dist_pospar_param_;
  double hyp_grouping_prob = 0.0;
  double new_grouping_prob = (inter_dist_grouping_ == DistGroupModeOff) ? 0.0 : inter_pospar_grouping_prob_(prev_pos,sclass);
  double grouping_grad = 0.0;
  
  Math1D::Vector<double> new_dist_param(nPosParams);
  Math1D::Vector<double> dist_param_grad(nPosParams); 
  
  double energy = inter_distortion_pospar_m_step_energy(single_pos_count, pos_span_count, sclass, prev_pos, 
														inter_dist_pospar_param_, inter_pospar_grouping_prob_(prev_pos,sclass));
  
  { //test if normalized counts give a better starting point

	for (uint d = 0; d < single_pos_count.xDim(); d++)
	  hyp_dist_param(d, prev_pos, sclass) = 0.0;
    hyp_grouping_prob = 0.0;
	
	double sum = 0.0;
	for (uint d = 0; d < single_pos_count.xDim(); d++)
	{
	  const double count = single_pos_count(d, prev_pos, sclass);
	  sum += count;
	  if (inter_dist_grouping_ == DistGroupModeOff || d <= dist_grouping_limit_)
		hyp_dist_param(d, prev_pos, sclass) += count;
	  else 
		hyp_grouping_prob += count;
	}
	
	if (sum > 1e-300) {
	  double inv_sum = 1.0 / sum;
	  
	  for (uint d = 0; d < nPosParams; d++)
		hyp_dist_param(d, prev_pos, sclass) = std::max(fert_min_param_entry, inv_sum * hyp_dist_param(d, prev_pos, sclass));
	  if (inter_dist_grouping_ != DistGroupModeOff)
	    hyp_grouping_prob = std::max(fert_min_param_entry, inv_sum * hyp_grouping_prob);
	
	  double hyp_energy = (deficient_) ? -1.0 : inter_distortion_pospar_m_step_energy(single_pos_count, pos_span_count, sclass, prev_pos, 
																					  hyp_dist_param, hyp_grouping_prob);
	
	  if (hyp_energy < energy) {
		if (nSourceClasses_ <= 4)  
		  std::cerr << "switching to normalized counts" << std::endl;
		energy = hyp_energy;
	    for (uint d = 0; d < nPosParams; d++)
		  inter_dist_pospar_param_(d, prev_pos, sclass) = hyp_dist_param(d, prev_pos, sclass);
	    inter_pospar_grouping_prob_(prev_pos, sclass) = hyp_grouping_prob;
	  }
	}
  }
  
  if (deficient_)
	return;

  double line_reduction_factor = 0.5;
  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {

    if ((iter % 5) == 0) {

      if (nSourceClasses_ <= 4)
        std::cerr << "inter m-step iteration #" << iter << ", energy: " << energy << std::endl;
    }

	dist_param_grad.set_constant(0.0);
	grouping_grad = 0.0;
	
	//1. compute the gradient
	//a) singleton terms
    for (int d = 0; d <= maxJ_; d++) {
      if (inter_dist_grouping_ == DistGroupModeOff || d <= dist_grouping_limit_)
	    dist_param_grad[d] -= single_pos_count(d, prev_pos, sclass) / inter_dist_pospar_param_(d, prev_pos, sclass);
      else
	    grouping_grad -= single_pos_count(d, prev_pos, sclass) / inter_pospar_grouping_prob_(prev_pos,sclass);
    }

	//b) normalization terms
    double param_sum = 0.0;
    uint nLong = 0;
    for (uint d = 0; d < pos_span_count.xDim(); d++) {
	
	  if (inter_dist_grouping_ == DistGroupModeOff || d <= dist_grouping_limit_)
	    param_sum += inter_dist_pospar_param_(d, prev_pos, sclass);
      else if (inter_dist_grouping_ == DistGroupModeDivide)
	    param_sum += inter_pospar_grouping_prob_(prev_pos,sclass) / nDistGroupingTerms;
      else {
	    assert(inter_dist_grouping_ == DistGroupModeExpand);
	    nLong++;
	  }
  
      double full_sum = param_sum;
      if (nLong > 0)
        full_sum += inter_pospar_grouping_prob_(prev_pos,sclass);	
	  const double count = pos_span_count(d, prev_pos, sclass);
	  if (count == 0.0)
		continue;

	  for (uint dd = 0; dd <= d; dd++)
	  {
		if (inter_dist_grouping_ == DistGroupModeOff || dd <= dist_grouping_limit_)
	      dist_param_grad[dd] += count / full_sum;
	    else if (inter_dist_grouping_ == DistGroupModeDivide)
		  grouping_grad += count / (full_sum * nDistGroupingTerms);
	  }		  
		
	  if (nLong > 0)
	    grouping_grad += count / full_sum;  
		
	  //energy += pos_span_count(d, prev_center, sclass) * std::log(full_sum); 
    }	  
	
	double sqr_grad_norm = dist_param_grad.sqr_norm() + grouping_grad * grouping_grad;
	if (sqr_grad_norm < 1e-5)
	{
	  if (nSourceClasses_ <= 4)	
	    std::cerr << "CUTOFF because gradient was nearly zero: " << sqr_grad_norm << std::endl;
	  break;
	}
	
	double real_alpha = 10.0 / sqrt(sqr_grad_norm);

    //2. go in neg gradient direction and reproject
	for (uint d = 0; d < nPosParams; d++) {
	  new_dist_param[d] = inter_dist_pospar_param_(d, prev_pos, sclass) - real_alpha * dist_param_grad[d];
	  if (isnan(new_dist_param[d])) {
		std::cerr << "params: " << new_dist_param << std::endl;
		std::cerr << "gradient: " << dist_param_grad << std::endl;
		std::cerr << "real_alpha: " << real_alpha << std::endl;
	  }
	  assert(!isnan(new_dist_param[d]));
	}
    if (inter_dist_grouping_ != DistGroupModeOff) {
      new_grouping_prob = inter_pospar_grouping_prob_(prev_pos, sclass) - real_alpha * grouping_grad;
	  assert(!isnan(new_grouping_prob));
      projection_on_simplex_with_slack(new_dist_param.direct_access(), new_grouping_prob, nPosParams, fert_min_param_entry);
	}
	else {
	  projection_on_simplex(new_dist_param, fert_min_param_entry);
	}
	
	//3. find a suitable step size
    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nTrials = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nTrials++;

      lambda *= line_reduction_factor;
      const double neg_lambda = 1.0 - lambda;

	  for (uint d = 0; d < nPosParams; d++)
	    hyp_dist_param(d, prev_pos, sclass) = lambda * new_dist_param[d] + neg_lambda * inter_dist_pospar_param_(d, prev_pos, sclass);
	  hyp_grouping_prob = lambda * new_grouping_prob + neg_lambda * inter_pospar_grouping_prob_(prev_pos,sclass);

      double hyp_energy = inter_distortion_pospar_m_step_energy(single_pos_count, pos_span_count, sclass, prev_pos, 
															    hyp_dist_param, hyp_grouping_prob);
																
	  assert(!isnan(hyp_energy));
	  
      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nTrials > 5 && best_energy < 0.975 * energy)
        break;

      if (nTrials > 25 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;
	
    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nTrials > 6)
      line_reduction_factor *= 0.9;

    //EXPERIMENTAL
    // if (nIter > 4)
    //   alpha *= 1.5;
    //END_EXPERIMENTAL

    const double neg_best_lambda = 1.0 - best_lambda;
	for (uint d = 0; d < nPosParams; d++)
	  inter_dist_pospar_param_(d, prev_pos, sclass) = best_lambda * new_dist_param[d] + neg_best_lambda * inter_dist_pospar_param_(d, prev_pos, sclass);
	inter_pospar_grouping_prob_(prev_pos,sclass) = best_lambda * new_grouping_prob + neg_best_lambda * inter_pospar_grouping_prob_(prev_pos,sclass);

    energy = best_energy;	
  }
}

void IBM5Trainer::inter_distortion_diffpar_m_step_unconstrained(const Math2D::Matrix<double>& single_diff_count, const Math3D::Tensor<double>& diff_span_count,
    uint sclass, uint L)
{
  const uint nParams = inter_distortion_param_.xDim();

  for (uint k = 0; k < nParams; k++)
    inter_distortion_param_(k, sclass) = std::max(fert_min_param_entry, inter_distortion_param_(k, sclass));

  Math1D::Vector<double> gradient(nParams);
  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> hyp_work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> search_direction(nParams);
  Math2D::Matrix<double> hyp_param = inter_distortion_param_;

  double energy = (deficient_) ? 0.0 : inter_distortion_diffpar_m_step_energy(single_diff_count, diff_span_count, sclass, inter_distortion_param_, 0.0);

  {
    //check if normalized expectations give a better starting point

    const double sum = single_diff_count.row_sum(sclass);

    for (uint d = 0; d < nParams; d++)
      hyp_param(d, sclass) = std::max(fert_min_param_entry, single_diff_count(d, sclass) / sum);

    //NOTE: the closed-form deficient solution does not necessarily reduce the energy INCLUDING the normalization term
    double hyp_energy = (deficient_) ? -1.0 : inter_distortion_diffpar_m_step_energy(single_diff_count, diff_span_count, sclass, 
																					 hyp_param, 0.0);

    if (hyp_energy < energy) {

      for (uint d = 0; d < nParams; d++)
        inter_distortion_param_(d, sclass) = hyp_param(d, sclass);

      energy = hyp_energy;
    }
  }

  if (deficient_)
    return;

  if (nSourceClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint k = 0; k < nParams; k++)
    work_param[k] = sqrt(inter_distortion_param_(k, sclass));

  double scale = 1.0;

  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {

    if ((iter % 5) == 0) {

      if (nSourceClasses_ <= 4)
        std::cerr << "L-BFGS inter m-step iteration #" << iter << ", energy: " << energy << std::endl;
    }

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    gradient.set_constant(0.0);
    work_grad.set_constant(0.0);

    for (uint j = 0; j < nParams; j++)
      gradient[j] -= single_diff_count(j, sclass) / std::max(fert_min_param_entry, inter_distortion_param_(j,sclass));

    Math1D::Vector<double> addon(diff_span_count.yDim());

    for (uint d_start = 0; d_start < diff_span_count.xDim(); d_start++) {

      double param_sum = 0.0;
      for (uint d_end = d_start; d_end < diff_span_count.yDim(); d_end++) {

        param_sum += std::max(fert_min_param_entry, inter_distortion_param_(d_end, sclass));

        addon[d_end] = diff_span_count(d_start, d_end, sclass) / param_sum;
        // double addon = diff_span_count(d_start,d_end,sclass) / param_sum;

        // for (uint d=d_start; d <= d_end; d++)
        //   gradient[d] += addon;
      }

      double addon_sum = 0.0;
      for (int d = diff_span_count.yDim() - 1; d >= int (d_start); d--) {

        addon_sum += addon[d];
        gradient[d] += addon_sum;
      }
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
      const double grad = gradient[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 0; kk < nParams; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

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

        search_direction.add_vector_multiple(cur_step, gamma);
        // for (uint k=0; k < nParams; k++)
        //   search_direction[k] += cur_step[k] * gamma;
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

      double sqr_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_work_param[k] * hyp_work_param[k];
      }

      for (uint k = 0; k < nParams; k++)
        hyp_param(k, sclass) = std::max(fert_min_param_entry, hyp_work_param[k] * hyp_work_param[k] / sqr_sum);

      double hyp_energy = inter_distortion_diffpar_m_step_energy(single_diff_count, diff_span_count, sclass, hyp_param, 0.0);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point

    if (best_energy >= energy - 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy)
                << ", final energy: " << energy << std::endl;
      std::cerr << "last squared gradient norm: " << work_grad.sqr_norm() << std::endl;
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

    for (uint k = 0; k < nParams; k++)
      inter_distortion_param_(k, sclass) = std::max(fert_min_param_entry, work_param[k] * work_param[k] / scale);
  }
}

double IBM5Trainer::intra_distortion_m_step_energy(const Math2D::Matrix<double>& single_diff_count, const Math2D::Matrix<double>& diff_span_count,
    uint sclass, const Math2D::Matrix<double>& param) const
{
  const uint xDim = param.xDim();

  double energy = 0.0;

  for (uint j = 0; j < xDim; j++)
    energy -= single_diff_count(j, sclass) * std::log(std::max(fert_min_param_entry, param(j, sclass)));

  double param_sum = 0.0;
  for (uint J = 0; J < xDim; J++) {

    param_sum += std::max(fert_min_param_entry, param(J, sclass));
    energy += diff_span_count(J, sclass) * std::log(param_sum);
  }

  return energy;
}

void IBM5Trainer::intra_distortion_m_step(const Math2D::Matrix<double>& single_diff_count, const Math2D::Matrix<double>& diff_span_count, uint sclass)
{
  const uint nParams = intra_distortion_param_.xDim();

  Math1D::Vector<double> gradient(nParams);
  Math1D::Vector<double> new_param(nParams);
  Math2D::Matrix<double> hyp_param(intra_distortion_param_.xDim(), intra_distortion_param_.yDim());

  for (uint j = 0; j < nParams; j++)
    intra_distortion_param_(j, sclass) = std::max(fert_min_param_entry, intra_distortion_param_(j, sclass));

  double energy = (deficient_) ? 0.0 : intra_distortion_m_step_energy(single_diff_count, diff_span_count, sclass, intra_distortion_param_);

  {
    //check if normalized expectations give a better starting point

    const double sum = single_diff_count.row_sum(sclass);

    for (uint j = 0; j < nParams; j++)
      hyp_param(j, sclass) = std::max(fert_min_param_entry, single_diff_count(j, sclass) / sum);

    //NOTE: the closed-form deficient solution does not necessarily reduce the energy INCLUDING the normalization term
    double hyp_energy = (deficient_) ? -1.0 : intra_distortion_m_step_energy(single_diff_count, diff_span_count, sclass, hyp_param);

    if (hyp_energy < energy) {

      for (uint j = 0; j < gradient.size(); j++)
        intra_distortion_param_(j, sclass) = hyp_param(j, sclass);

      energy = hyp_energy;
    }
  }

  if (deficient_)
    return;

  if (nSourceClasses_ <= 4)
    std::cerr << "intra m-step start energy: " << energy << std::endl;

  double alpha = 0.1;
  double line_reduction_factor = 0.35;

  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {

    gradient.set_constant(0.0);

    if ((iter % 5) == 0) {

      if (nSourceClasses_ <= 4)
        std::cerr << "intra m-step iteration #" << iter << ", energy: " << energy << std::endl;
    }

    /*** 1. calculate gradient ***/
    for (uint j = 0; j < nParams; j++)
      gradient[j] -= single_diff_count(j, sclass) / std::max(fert_min_param_entry, intra_distortion_param_(j, sclass));

    double param_sum = 0.0;
    for (uint J = 0; J < nParams; J++) {

      param_sum += std::max(fert_min_param_entry, intra_distortion_param_(J, sclass));
      double addon = diff_span_count(J, sclass) / param_sum;

      for (uint j = 0; j <= J; j++)
        gradient[j] += addon;
    }

    /*** 2. go in neg. gradient direction ***/
    double sqr_grad_norm = gradient.sqr_norm();
    if (sqr_grad_norm < 1e-5) {
      std::cerr << "CUTOFF after " << iter << " iterations because squared gradient norm was " << sqr_grad_norm << std::endl;
      break;
    }

    double real_alpha = alpha / sqrt(sqr_grad_norm);
    for (uint j = 0; j < gradient.size(); j++)
      new_param[j] = intra_distortion_param_(j, sclass) - real_alpha * gradient[j];

    /*** 3. reproject ***/
    projection_on_simplex(new_param, fert_min_param_entry);

    /*** 4. find appropriate step size ***/
    double best_lambda = 1.0;
    double lambda = 1.0;

    double best_energy = 1e300;

    uint nTrials = 0;

    bool decreasing = false;

    while (decreasing || best_energy > energy) {

      nTrials++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
        hyp_param(j, sclass) = lambda * new_param[j] + neg_lambda * intra_distortion_param_(j, sclass);

      double hyp_energy = intra_distortion_m_step_energy(single_diff_count, diff_span_count, sclass, hyp_param);

      if (hyp_energy < best_energy) {

        best_energy = hyp_energy;
        best_lambda = lambda;
        decreasing = true;
      }
      else
        decreasing = false;

      if (nTrials > 5 && best_energy < 0.975 * energy)
        break;

      if (nTrials > 25 && lambda < 1e-12)
        break;
    }
    //std::cerr << "best lambda: " << best_lambda << std::endl;

    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    if (nTrials > 6)
      line_reduction_factor *= 0.9;

    //EXPERIMENTAL
    // if (nIter > 4)
    //   alpha *= 1.5;
    //END_EXPERIMENTAL

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
      intra_distortion_param_(j, sclass) = best_lambda * new_param[j] + neg_best_lambda * intra_distortion_param_(j, sclass);

    energy = best_energy;
  }
}

void IBM5Trainer::intra_distortion_m_step_unconstrained(const Math2D::Matrix<double>& single_diff_count, const Math2D::Matrix<double>& diff_span_count,                                                        uint sclass, uint L)
{
  const uint nParams = intra_distortion_param_.xDim();

  Math1D::Vector<double> gradient(nParams);
  Math1D::Vector<double> work_param(nParams);
  Math1D::Vector<double> work_grad(nParams);
  Math1D::Vector<double> hyp_work_param(nParams);
  Math1D::Vector<double> search_direction(nParams);
  Math2D::Matrix<double> hyp_param(nParams, intra_distortion_param_.yDim());

  for (uint j = 0; j < nParams; j++)
    intra_distortion_param_(j, sclass) = std::max(fert_min_param_entry, intra_distortion_param_(j, sclass));

  double energy = (deficient_) ? 0.0 : intra_distortion_m_step_energy(single_diff_count, diff_span_count, sclass, intra_distortion_param_);

  {
    //check if normalized expectations give a better starting point

    const double sum = single_diff_count.row_sum(sclass);

    for (uint j = 0; j < nParams; j++)
      hyp_param(j, sclass) = std::max(fert_min_param_entry, single_diff_count(j, sclass) / sum);

    //NOTE: the closed-form deficient solution does not necessarily reduce the energy INCLUDING the normalization term
    double hyp_energy = (deficient_) ? -1.0 : intra_distortion_m_step_energy(single_diff_count, diff_span_count, sclass, hyp_param);

    if (hyp_energy < energy) {

      for (uint j = 0; j < gradient.size(); j++)
        intra_distortion_param_(j, sclass) = hyp_param(j, sclass);

      energy = hyp_energy;
    }
  }

  if (deficient_)
    return;

  if (nSourceClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint k = 0; k < nParams; k++)
    work_param[k] = sqrt(intra_distortion_param_(k, sclass));

  double scale = 1.0;

  for (uint iter = 1; iter <= dist_m_step_iter_; iter++) {

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    gradient.set_constant(0.0);

    if ((iter % 5) == 0) {

      if (nSourceClasses_ <= 4)
        std::cerr << "L-BFGS intra m-step iteration #" << iter << ", energy: " << energy << std::endl;
    }

    for (uint j = 0; j < nParams; j++)
      gradient[j] -= single_diff_count(j, sclass) / std::max(fert_min_param_entry, intra_distortion_param_(j,sclass));

    double param_sum = 0.0;
    for (uint J = 0; J < nParams; J++) {

      param_sum += intra_distortion_param_(J, sclass);
      double addon = diff_span_count(J, sclass) / param_sum;

      for (uint j = 0; j <= J; j++)
        gradient[j] += addon;
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
      const double grad = gradient[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 0; kk < nParams; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

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

    uint nTrials = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nTrials++;
      if (nTrials > 15 && best_energy < energy) {
        break;
      }

      if (nTrials > 1)
        alpha *= line_reduction_factor;

      double sqr_sum = 0.0;

      for (uint k = 0; k < nParams; k++) {
        hyp_work_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_work_param[k] * hyp_work_param[k];
      }

      for (uint k = 0; k < nParams; k++)
        hyp_param(k, sclass) = std::max(fert_min_param_entry, hyp_work_param[k] * hyp_work_param[k] / sqr_sum);

      double hyp_energy = intra_distortion_m_step_energy(single_diff_count, diff_span_count, sclass, hyp_param);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nTrials > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point

    if (best_energy >= energy - 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy)
                << ", final energy: " << energy << std::endl;
      std::cerr << "last squared gradient norm: " << work_grad.sqr_norm() << std::endl;
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

    for (uint k = 0; k < nParams; k++)
      intra_distortion_param_(k, sclass) = std::max(fert_min_param_entry, work_param[k] * work_param[k] / scale);
  }
}

void IBM5Trainer::train_em(uint nIter, FertilityModelTrainerBase* fert_trainer, const HmmWrapperBase* passed_wrapper)
{
  const size_t nSentences = source_sentence_.size();

  std::cerr << "starting IBM-5 training without constraints";
  if (fert_trainer != 0)
    std::cerr << " (init from " << fert_trainer->model_name() << ") ";
  std::cerr << std::endl;

  double max_perplexity = 0.0;
  double approx_sum_perplexity = 0.0;

  SingleLookupTable aux_lookup;

  double dict_weight_sum = (prior_weight_active_) ? 1.0 : 0.0; //only used as a flag

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords, MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  double fzero_count;
  double fnonzero_count;

  Storage1D<Math3D::Tensor<double> > inter_distortion_count = inter_distortion_prob_;

  //new variant
  Math2D::Matrix<double> single_diff_count(2*maxJ_,nSourceClasses_);
  Math3D::Tensor<double> single_pos_count(maxJ_+1,maxJ_+1,nSourceClasses_,0.0);

  //NOTE: unlike for the IBM-4, here the second index can also be one smaller than the displacement offset
  //  but of course the end of a span is never smaller than its start
  Math3D::Tensor<double> inter_span_count(displacement_offset_ + 1, inter_distortion_param_.xDim(), inter_distortion_param_.yDim());
  Math3D::Tensor<double> inter_pos_span_count(maxJ_+1,maxJ_+1,inter_distortion_param_.yDim());

  Storage1D<Math2D::Matrix<double> > intra_distortion_count = intra_distortion_prob_;

  //new variant
  Math2D::Matrix<double> intra_distparam_count = intra_distortion_param_;
  Math2D::Matrix<double> intra_span_count = intra_distortion_param_;

  Storage1D<Math1D::Vector<double> > sentence_start_count = sentence_start_prob_;

  //new variant
  Math1D::Vector<double> fsentence_start_count(maxJ_);
  Math1D::Vector<double> fstart_span_count(maxJ_);

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    std::cerr << "******* IBM-5 EM-iteration " << iter << std::endl;

    if (passed_wrapper != 0
        && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    uint sum_iter = 0;

    /*** clear counts ***/
    for (uint J = 1; J < inter_distortion_count.size(); J++)
      inter_distortion_count[J].set_constant(0.0);
    single_diff_count.set_constant(0.0);
    inter_span_count.set_constant(0.0);
	inter_pos_span_count.set_constant(0.0);
	single_pos_count.set_constant(0.0);

    for (uint J = 1; J < intra_distortion_count.size(); J++) {
      intra_distortion_count[J].set_constant(0.0);
      sentence_start_count[J].set_constant(0.0);
    }
    intra_distparam_count.set_constant(0.0);
    intra_span_count.set_constant(0.0);

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    fsentence_start_count.set_constant(0.0);
    fstart_span_count.set_constant(0.0);

    max_perplexity = 0.0;
    approx_sum_perplexity = 0.0;

    double hillclimbtime = 0.0;
    double countcollecttime = 0.0;

    for (size_t s = 0; s < nSentences; s++) {

      if ((s % 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      //std::cerr << "J=" << curJ << ", I=" << curI << std::endl;

      Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      std::clock_t tHillclimbStart, tHillclimbEnd;
      tHillclimbStart = std::clock();

      long double best_prob = 0.0;

      //std::cerr << "calling hillclimbing" << std::endl;

      if (fert_trainer != 0 && iter == 1) {
        best_prob = fert_trainer->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter,
																   fertility, expansion_move_prob, swap_move_prob, cur_alignment);
      }
      else {
        best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
													 expansion_move_prob, swap_move_prob, cur_alignment);
      }
      max_perplexity -= std::log(best_prob);

      //std::cerr << "back from hillclimbing" << std::endl;
      //std::cerr << "alignment: " << best_known_alignment_[s] << std::endl;

      tHillclimbEnd = std::clock();

      hillclimbtime += diff_seconds(tHillclimbEnd, tHillclimbStart);

      const long double expansion_prob = expansion_move_prob.sum();
      const long double swap_prob = swap_mass(swap_move_prob);  //0.5 * swap_move_prob.sum();

      const long double sentence_prob = best_prob + expansion_prob + swap_prob;

      assert(!isnan(sentence_prob));

      approx_sum_perplexity -= std::log(sentence_prob);

      const long double inv_sentence_prob = 1.0 / sentence_prob;

      /****** collect counts ******/

      /**** update empty word counts *****/

      update_zero_counts(best_known_alignment_[s], fertility, expansion_move_prob, swap_prob, best_prob,
                         sentence_prob, inv_sentence_prob, fzero_count, fnonzero_count);

      /**** update fertility counts *****/
      update_fertility_counts(cur_target, best_known_alignment_[s], fertility,
                              expansion_move_prob, sentence_prob, inv_sentence_prob, ffert_count);

      /**** update dictionary counts *****/
      update_dict_counts(cur_source, cur_target, cur_lookup, best_known_alignment_[s], expansion_move_prob,
                         swap_move_prob, sentence_prob, inv_sentence_prob, fwcount);

      std::clock_t tCountCollectStart, tCountCollectEnd;
      tCountCollectStart = std::clock();

      /**** update distortion counts *****/

      fstart_span_count[curJ - 1] += 1.0;

      //std::cerr << "updating distortion counts" << std::endl;

      //1. Viterbi alignment
      NamedStorage1D<std::vector<ushort> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));
      for (uint j = 0; j < curJ; j++)
        aligned_source_words[cur_alignment[j]].push_back(j);

      {
        const double increment = best_prob * inv_sentence_prob;

        uint prev_center = MAX_UINT;

        Storage1D<bool> fixed(curJ, false);

        uint nOpen = curJ;

        for (uint i = 1; i <= curI; i++) {

          if (fertility[i] > 0) {

            const std::vector<ushort>& cur_aligned_source_words = aligned_source_words[i];
            assert(cur_aligned_source_words.size() == fertility[i]);

            //a) head of the cept
            const uint first_j = cur_aligned_source_words[0];

            if (prev_center != MAX_UINT) {      // currently not estimating a start probability

              const uint sclass = source_class_[cur_source[first_j]];

              const uint nAvailable = nOpen - (cur_aligned_source_words.size() - 1);

              Math3D::Tensor<double>& cur_inter_distortion_count = inter_distortion_count[nAvailable];

              uint pos_first_j = MAX_UINT;
              uint pos_prev_center = MAX_UINT;

              uint nCurOpen = 0;
              for (uint j = 0; j <= std::max(first_j, prev_center); j++) {

                if (j == first_j)
                  pos_first_j = nCurOpen;

                if (!fixed[j])
                  nCurOpen++;

                if (j == prev_center)
                  pos_prev_center = nCurOpen;
              }

              //DEBUG
              if (pos_prev_center >= cur_inter_distortion_count.yDim()) {

                std::cerr << "J= " << curJ << ", prev_center=" << prev_center << ", pos_prev_center = " << pos_prev_center << std::endl;
                std::cerr << "fixed: " << fixed << std::endl;
              }
              //END_DEBUG

              assert(pos_prev_center < cur_inter_distortion_count.yDim());

              cur_inter_distortion_count(pos_first_j, pos_prev_center, sclass) += increment;
              single_diff_count(displacement_offset_ + pos_first_j - pos_prev_center, sclass) += increment;
              inter_span_count(displacement_offset_ - pos_prev_center, displacement_offset_ + nAvailable - 1 - pos_prev_center, sclass) += increment;
              
			  single_pos_count(pos_first_j, pos_prev_center, sclass) += increment;
			  inter_pos_span_count(nAvailable,pos_prev_center,sclass) += increment;
			}
            else if (use_sentence_start_prob_) {
              sentence_start_count[curJ][first_j] += increment;

              fsentence_start_count[first_j] += increment;
            }

            fixed[first_j] = true;
            nOpen--;

            //b) body of the cept
            for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

              const uint cur_j = cur_aligned_source_words[k];

              const uint sclass = source_class_[cur_source[cur_j]];
			  const uint tclass = target_class_[cur_target[i-1]];
			  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass : tclass;

              uint pos = MAX_UINT;

              uint nAvailable = 0;
              for (uint j = cur_aligned_source_words[k - 1] + 1; j < curJ; j++) {

                if (j == cur_j)
                  pos = nAvailable;

                if (!fixed[j])
                  nAvailable++;
              }

              nAvailable -= cur_aligned_source_words.size() - 1 - k;

              intra_distortion_count[nAvailable] (pos, cur_class) += increment;
              intra_distparam_count(pos, cur_class) += increment;
              intra_span_count(nAvailable - 1, cur_class) += increment;

              fixed[cur_j] = true;
              nOpen--;
            }

            //c) calculate the center of the cept
            switch (cept_start_mode_) {
            case IBM4CENTER: {

              //compute the center of this cept and store the result in prev_cept_center
              double sum = vec_sum(cur_aligned_source_words);
              prev_center = (int)round(sum / cur_aligned_source_words.size());
              break;
            }
            case IBM4FIRST:
              prev_center = first_j;
              break;
            case IBM4LAST:
              prev_center = cur_aligned_source_words.back();
              break;
            case IBM4UNIFORM:
              break;
            default:
              assert(false);
            }
          }
        }
      }

      NamedStorage1D<std::vector<ushort> > hyp_aligned_source_words(MAKENAME(hyp_aligned_source_words));
      hyp_aligned_source_words = aligned_source_words;

      //2. expansion moves
      for (uint j = 0; j < curJ; j++) {

        //std::cerr << "j: " << j << std::endl;

        uint cur_aj = best_known_alignment_[s][j];

        vec_erase(hyp_aligned_source_words[cur_aj], (ushort) j);

        for (uint aj = 0; aj <= curI; aj++) {

          if (expansion_move_prob(j, aj) > best_prob * 1e-11) {

            const double increment = expansion_move_prob(j, aj) * inv_sentence_prob;

            //hyp_aligned_source_words[aj].push_back(j);
            //vec_sort(hyp_aligned_source_words[aj]);
            sorted_vec_insert<ushort>(hyp_aligned_source_words[aj], j);

            uint prev_center = MAX_UINT;

            Storage1D<bool> fixed(curJ, false);

            uint nOpen = curJ;

            for (uint i = 1; i <= curI; i++) {

              const std::vector<ushort>& cur_aligned_source_words = hyp_aligned_source_words[i];

              if (cur_aligned_source_words.size() > 0) {

                //a) head of the cept
                const uint first_j = cur_aligned_source_words[0];

                if (prev_center != MAX_UINT) {  // currently not estimating a start probability

                  const uint sclass = source_class_[cur_source[first_j]];

                  const uint nAvailable = nOpen - (cur_aligned_source_words.size() - 1);

                  Math3D::Tensor<double>& cur_inter_distortion_count = inter_distortion_count[nAvailable];

                  uint pos_first_j = MAX_UINT;
                  uint pos_prev_center = MAX_UINT;

                  uint nCurOpen = 0;
                  for (uint j = 0; j <= std::max(first_j, prev_center); j++) {

                    if (j == first_j)
                      pos_first_j = nCurOpen;

                    if (!fixed[j])
                      nCurOpen++;

                    if (j == prev_center)
                      pos_prev_center = nCurOpen;
                  }

                  //DEBUG
                  // if (pos_prev_center >= cur_inter_distortion_count.yDim()) {

                  //   std::cerr << "J= " << curJ << ", prev_center=" << prev_center << ", pos_prev_center = " << pos_prev_center << std::endl;
                  //   std::cerr << "fixed: " << fixed << std::endl;
                  // }
                  //END_DEBUG

                  assert(pos_prev_center < cur_inter_distortion_count.yDim());

                  cur_inter_distortion_count(pos_first_j, pos_prev_center, sclass) += increment;
                  single_diff_count(displacement_offset_ + pos_first_j - pos_prev_center, sclass) += increment;
                  inter_span_count(displacement_offset_ - pos_prev_center, displacement_offset_ + nAvailable - 1 - pos_prev_center, sclass) += increment;
                  
				  single_pos_count(pos_first_j, pos_prev_center, sclass) += increment;
				  inter_pos_span_count(nAvailable,pos_prev_center,sclass) += increment;
				}
                else if (use_sentence_start_prob_) {
                  sentence_start_count[curJ][first_j] += increment;

                  fsentence_start_count[first_j] += increment;
                }

                fixed[first_j] = true;
                nOpen--;

                //b) body of the cept
                for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

                  const uint cur_j = cur_aligned_source_words[k];

                  const uint sclass = source_class_[cur_source[cur_j]];
	  			  const uint tclass = target_class_[cur_target[i-1]];
				  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass : tclass;

                  uint pos = MAX_UINT;

                  uint nAvailable = 0;
                  for (uint j = cur_aligned_source_words[k - 1] + 1; j < curJ; j++) {

                    if (j == cur_j)
                      pos = nAvailable;

                    if (!fixed[j])
                      nAvailable++;
                  }

                  nAvailable -= cur_aligned_source_words.size() - 1 - k;

                  intra_distortion_count[nAvailable] (pos, cur_class) += increment;
                  intra_distparam_count(pos, cur_class) += increment;
                  intra_span_count(nAvailable - 1, cur_class) += increment;

                  fixed[cur_j] = true;
                  nOpen--;
                }

                //c) calculate the center of the cept
                switch (cept_start_mode_) {
                case IBM4CENTER: {

                  //compute the center of this cept and store the result in prev_cept_center
                  double sum = vec_sum(cur_aligned_source_words);
                  prev_center = (int)round(sum / cur_aligned_source_words.size());
                  break;
                }
                case IBM4FIRST:
                  prev_center = first_j;
                  break;
                case IBM4LAST:
                  prev_center = cur_aligned_source_words.back();
                  break;
                case IBM4UNIFORM:
                  break;
                default:
                  assert(false);
                }

              }
            }

            //restore hyp_aligned_source_words
            hyp_aligned_source_words[aj] = aligned_source_words[aj];
          }
        }

        //restore hyp_aligned_source_words
        hyp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
      }

      //3. swap moves
      for (uint j1 = 0; j1 < curJ - 1; j1++) {

        //std::cerr << "j1: " << j1 << std::endl;

        const uint aj1 = best_known_alignment_[s][j1];

        for (uint j2 = j1 + 1; j2 < curJ; j2++) {

          if (swap_move_prob(j1, j2) > best_prob * 1e-11) {

            const uint aj2 = best_known_alignment_[s][j2];

            vec_replace_maintainsort<ushort>(hyp_aligned_source_words[aj2], j2, j1);
            vec_replace_maintainsort<ushort>(hyp_aligned_source_words[aj1], j1, j2);

            const double increment = swap_move_prob(j1, j2) * inv_sentence_prob;

            uint prev_center = MAX_UINT;

            Storage1D<bool> fixed(curJ, false);

            uint nOpen = curJ;

            for (uint i = 1; i <= curI; i++) {

              const std::vector < ushort >& cur_aligned_source_words =
                hyp_aligned_source_words[i];

              if (cur_aligned_source_words.size() > 0) {

                //a) head of the cept
                const uint first_j = cur_aligned_source_words[0];

                if (prev_center != MAX_UINT) {  // currently not estimating a start probability

                  const uint sclass = source_class_[cur_source[first_j]];

                  const uint nAvailable = nOpen - (cur_aligned_source_words.size() - 1);

                  Math3D::Tensor<double>& cur_inter_distortion_count = inter_distortion_count[nAvailable];

                  uint pos_first_j = MAX_UINT;
                  uint pos_prev_center = MAX_UINT;

                  uint nCurOpen = 0;
                  for (uint j = 0; j <= std::max(first_j, prev_center); j++) {

                    if (j == first_j)
                      pos_first_j = nCurOpen;

                    if (!fixed[j])
                      nCurOpen++;

                    if (j == prev_center)
                      pos_prev_center = nCurOpen;
                  }

                  //DEBUG
                  if (pos_prev_center >= cur_inter_distortion_count.yDim()) {

                    std::cerr << "J= " << curJ << ", prev_center=" << prev_center
                              << ", pos_prev_center = " << pos_prev_center << std::endl;
                    std::cerr << "fixed: " << fixed << std::endl;
                  }
                  //END_DEBUG

                  assert(pos_prev_center < cur_inter_distortion_count.yDim());

                  cur_inter_distortion_count(pos_first_j, pos_prev_center, sclass) += increment;
                  single_diff_count(displacement_offset_ + pos_first_j - pos_prev_center, sclass) += increment;
                  inter_span_count(displacement_offset_ - pos_prev_center, displacement_offset_ + nAvailable - 1 - pos_prev_center, sclass) += increment;
                  
				  single_pos_count(pos_first_j, pos_prev_center, sclass) += increment;
				  inter_pos_span_count(nAvailable,pos_prev_center,sclass) += increment;
				}
                else if (use_sentence_start_prob_) {
                  sentence_start_count[curJ][first_j] += increment;

                  fsentence_start_count[first_j] += increment;
                }

                fixed[first_j] = true;
                nOpen--;

                //b) body of the cept
                for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

                  const uint cur_j = cur_aligned_source_words[k];

                  const uint sclass = source_class_[cur_source[cur_j]];
	  			  const uint tclass = target_class_[cur_target[i-1]];
				  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass : tclass;

                  uint pos = MAX_UINT;

                  uint nAvailable = 0;
                  for (uint j = cur_aligned_source_words[k - 1] + 1; j < curJ; j++) {

                    if (j == cur_j)
                      pos = nAvailable;

                    if (!fixed[j])
                      nAvailable++;
                  }

                  nAvailable -= cur_aligned_source_words.size() - 1 - k;

                  intra_distortion_count[nAvailable] (pos, cur_class) += increment;
                  intra_distparam_count(pos, cur_class) += increment;
                  intra_span_count(nAvailable - 1, cur_class) += increment;

                  fixed[cur_j] = true;
                  nOpen--;
                }

                //c) calculate the center of the cept
                switch (cept_start_mode_) {
                case IBM4CENTER: {

                  //compute the center of this cept and store the result in prev_cept_center
                  double sum = vec_sum(cur_aligned_source_words);
                  prev_center = (int)round(sum / cur_aligned_source_words.size());
                  break;
                }
                case IBM4FIRST:
                  prev_center = first_j;
                  break;
                case IBM4LAST:
                  prev_center = cur_aligned_source_words.back();
                  break;
                case IBM4UNIFORM:
                  break;
                default:
                  assert(false);
                }
              }
            }

            //restore hyp_aligned_source_words
            hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
            hyp_aligned_source_words[aj2] = aligned_source_words[aj2];
          }
        }
      }
    } //loop over sentences finished

    double reg_term = regularity_term();        // we need the reg-term before the parameter update!

    /***** update probability models from counts *******/

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      std::cerr << "zero counts: " << fzero_count << ", " << fnonzero_count << std::endl;
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max(fert_min_p0, fzero_count / fsum);
      p_nonzero_ = std::max(fert_min_p0, fnonzero_count / fsum);
    }

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    assert(!isnan(p_zero_));
    assert(!isnan(p_nonzero_));

    //DEBUG
    uint nZeroAlignments = 0;
    uint nAlignments = 0;
    for (size_t s = 0; s < source_sentence_.size(); s++) {

      nAlignments += source_sentence_[s].size();

      for (uint j = 0; j < source_sentence_[s].size(); j++) {
        if (best_known_alignment_[s][j] == 0)
          nZeroAlignments++;
      }
    }
    std::cerr << "percentage of zero-aligned words: "
              << (((double)nZeroAlignments) / ((double)nAlignments)) << std::endl;
    //END_DEBUG

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, nSentences, dict_weight_sum, smoothed_l0_, l0_beta_, dict_m_step_iter_, dict_, fert_min_dict_entry,
                            msolve_mode_ != MSSolvePGD, gd_stepsize_);

    //update fertility probabilities
    update_fertility_prob(ffert_count, fert_min_param_entry);

    //update inter distortion probabilities
    if (distortion_type_ == IBM23Nonpar) {
		
	  const uint nDistGroupingTerms = maxJ_+1 - (dist_grouping_limit_+1);

	  //NOTE: inter_dist_grouping_ is ignored for nonpar distortion
      for (uint J = 1; J < inter_distortion_count.size(); J++) {

        for (uint y = 0; y < inter_distortion_count[J].yDim(); y++) {
          for (uint z = 0; z < inter_distortion_count[J].zDim(); z++) {

            double sum = 0.0;
			
            for (uint j = 0; j < inter_distortion_count[J].xDim(); j++)
              sum += inter_distortion_count[J](j, y, z);
 
            if (sum > 1e-305) {

			  if (inter_dist_grouping_ == DistGroupModeOff)
			  {
                for (uint j = 0; j < inter_distortion_count[J].xDim(); j++)
                  inter_distortion_prob_[J](j, y, z) = std::max(fert_min_param_entry, inter_distortion_count[J](j, y, z) / sum);
			  }
			  else {
				double longparam_sum = 0.0;
			    for (uint j = dist_grouping_limit_+1; j < inter_distortion_count[J].xDim(); j++)
				  longparam_sum += inter_distortion_count[J](j, y, z);
			    if (inter_dist_grouping_ == DistGroupModeDivide) {
  			      sum -= longparam_sum;
				  longparam_sum /= nDistGroupingTerms;
				  sum += longparam_sum;
				}
				
			    for (uint j = 0; j <= std::min<uint>(dist_grouping_limit_,J-1); j++)
				  inter_distortion_prob_[J](j, y, z) = std::max(fert_min_param_entry, inter_distortion_count[J](j, y, z) / sum);
			    for (uint j = dist_grouping_limit_+1; j < inter_distortion_count[J].xDim(); j++)
				  inter_distortion_prob_[J](j, y, z) = longparam_sum / (sum * (inter_distortion_count[J].xDim() - (dist_grouping_limit_+1)));
			  }
			}
          }
        }
      }
    }
    else {

      for (uint s = 0; s < inter_distortion_param_.yDim(); s++) {

		if (distortion_type_ == IBM23ParByDifference) {
          if (msolve_mode_ == MSSolvePGD)
            inter_distortion_diffpar_m_step(single_diff_count, inter_span_count, s);
          else
            inter_distortion_diffpar_m_step_unconstrained(single_diff_count,inter_span_count,s);
		}
		else {
		  for (uint prev_pos = 0; prev_pos < inter_dist_pospar_param_.yDim(); prev_pos++)
		    inter_distortion_pospar_m_step(single_pos_count, inter_pos_span_count, s, prev_pos);
		}
	  }

      par2nonpar_inter_distortion();
    }

    //update intra distortion probabilities
    if (!uniform_intra_prob_) {
      if (distortion_type_ == IBM23Nonpar) {

        for (uint J = 1; J < intra_distortion_prob_.size(); J++) {

          for (uint s = 0; s < intra_distortion_prob_[J].yDim(); s++) {

            double sum = intra_distortion_count[J].row_sum(s);

            if (sum > 1e-305) {
              for (uint j = 0; j < J; j++)
                intra_distortion_prob_[J](j, s) = std::max(fert_min_param_entry, intra_distortion_count[J] (j, s) / sum);
            }
          }
        }
      }
      else {

        Math2D::Matrix<double> hyp_param = intra_distortion_param_;

        // call m-steps
        for (uint s = 0; s < intra_distortion_param_.yDim(); s++) {

          if (msolve_mode_ == MSSolvePGD)
            intra_distortion_m_step(intra_distparam_count, intra_span_count, s);
          else
            intra_distortion_m_step_unconstrained(intra_distparam_count,intra_span_count,s);
        }

        par2nonpar_intra_distortion();
      }
    }

    if (use_sentence_start_prob_) {

      if (msolve_mode_ == MSSolvePGD)
        start_prob_m_step(fsentence_start_count, fstart_span_count, sentence_start_parameters_, start_m_step_iter_, gd_stepsize_);
      else
        start_prob_m_step_unconstrained(fsentence_start_count, fstart_span_count, sentence_start_parameters_, start_m_step_iter_);

      par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
    }

    max_perplexity /= source_sentence_.size();
    approx_sum_perplexity /= source_sentence_.size();

    max_perplexity += reg_term;
    approx_sum_perplexity += reg_term;

    std::string transfer = (fert_trainer != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "IBM-5 max-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << max_perplexity << std::endl;
    std::cerr << "IBM-5 approx-sum-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << approx_sum_perplexity << std::endl;
    std::cerr << (((double)sum_iter) / source_sentence_.size())
              << " average hillclimbing iterations per sentence pair" << std::endl;

    printEval(iter, transfer, "EM");
  }

  if (distortion_type_ == IBM23Nonpar) {

    //we still update <code> inter_distortion_param_ </code>  and <code> intra_distortion_param_ </code>
    //so that we can use them when computing external alignments

    //a) inter
    inter_distortion_param_.set_constant(0.0);

    for (uint s = 0; s < inter_distortion_param_.yDim(); s++) {

      for (uint J = 0; J < inter_distortion_count.size(); J++) {

        for (uint prev_pos = 0; prev_pos < inter_distortion_count[J].yDim(); prev_pos++) {

          for (uint j = 0; j < J; j++) {

            double cur_weight = inter_distortion_count[J] (j, prev_pos, s);
            inter_distortion_param_(displacement_offset_ + j - prev_pos, s) += cur_weight;
          }
        }
      }

      double sum = 0.0;
      for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
        sum += inter_distortion_param_(j, s);

      if (sum > 1e-305) {

        for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
          inter_distortion_param_(j, s) = std::max(1e-8, inter_distortion_param_(j, s) / sum);
      }
      else {
        for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
          inter_distortion_param_(j, s) = 1.0 / inter_distortion_param_.xDim();
      }
    }

    //b) intra
    intra_distortion_param_.set_constant(0.0);

    for (uint s = 0; s < intra_distortion_param_.yDim(); s++) {

      for (uint J = 0; J < intra_distortion_count.size(); J++) {

        for (uint j = 0; j < J; j++)
          intra_distortion_param_(j, s) += intra_distortion_count[J] (j, s);
      }

      double sum = 0.0;
      for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
        sum += intra_distortion_param_(j, s);

      if (sum > 1e-305) {

        for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
          intra_distortion_param_(j, s) = std::max(1e-8, intra_distortion_param_(j, s) / sum);
      }
      else {

        for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
          intra_distortion_param_(j, s) = 1.0 / intra_distortion_param_.xDim();
      }
    }
  }

  iter_offs_ = iter - 1;
}

void IBM5Trainer::train_viterbi(uint nIter, FertilityModelTrainerBase* fert_trainer, const HmmWrapperBase* passed_wrapper)
{
  const size_t nSentences = source_sentence_.size();

  std::cerr << "starting IBM-5 training without constraints";
  if (fert_trainer != 0)
    std::cerr << " (init from " << fert_trainer->model_name() << ") ";
  std::cerr << std::endl;

  double max_perplexity = 0.0;

  SingleLookupTable aux_lookup;

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords, MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords, MAKENAME(ffert_count));
  NamedStorage1D<Math1D::Vector<double> > ffertclass_count(MAKENAME(ffert_count));

  for (uint i = 0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  if (fertprob_sharing_) {
    ffertclass_count.resize(nTFertClasses_);
    for (uint i = 1; i < nTargetWords; i++) {
      uint c = tfert_class_[i];
      if (fertility_prob_[i].size() > ffertclass_count[c].size())
        ffertclass_count[c].resize_dirty(fertility_prob_[i].size());
    }
  }

  double fzero_count;
  double fnonzero_count;

  Storage1D<Math3D::Tensor<double> > inter_distortion_count = inter_distortion_prob_;
  Math3D::Tensor<double> single_pos_count(maxJ_+1,maxJ_+1, nSourceClasses_,0.0);

  //new variant
  Math2D::Matrix<double> single_diff_count(inter_distortion_param_.dims(),0.0);
  Math3D::Tensor<double> inter_span_count(displacement_offset_ + 1, inter_distortion_param_.xDim(), inter_distortion_param_.yDim());
  Math3D::Tensor<double> inter_pos_span_count(maxJ_+1,maxJ_+1,single_diff_count.yDim());

  Storage1D<Math2D::Matrix<double> > intra_distortion_count = intra_distortion_prob_;

  //new variant
  Math2D::Matrix<double> intra_distparam_count = intra_distortion_param_;
  Math2D::Matrix<double> intra_span_count = intra_distortion_param_;

  Storage1D<Math1D::Vector<double> > sentence_start_count = sentence_start_prob_;

  //new variant
  Math1D::Vector<double> fsentence_start_count(maxJ_);
  Math1D::Vector<double> fstart_span_count(maxJ_);

  uint iter;
  for (iter = 1 + iter_offs_; iter <= nIter + iter_offs_; iter++) {

    std::cerr << "******* IBM-5 EM-iteration " << iter << std::endl;

    if (passed_wrapper != 0
        && (hillclimb_mode_ == HillclimbingRestart || (hillclimb_mode_ == HillclimbingReinit && (iter-iter_offs_) == 1)  ) )
      set_hmm_alignments(*passed_wrapper);

    uint sum_iter = 0;

    /*** clear counts ***/
    for (uint J = 1; J < inter_distortion_count.size(); J++)
      inter_distortion_count[J].set_constant(0.0);
    single_diff_count.set_constant(0.0);
    inter_span_count.set_constant(0.0);
	inter_pos_span_count.set_constant(0.0);

    for (uint J = 1; J < intra_distortion_count.size(); J++) {
      intra_distortion_count[J].set_constant(0.0);
      sentence_start_count[J].set_constant(0.0);
    }
    intra_distparam_count.set_constant(0.0);
    intra_span_count.set_constant(0.0);

    fsentence_start_count.set_constant(0.0);
    fstart_span_count.set_constant(0.0);

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    for (uint i = 0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    if (fertprob_sharing_) {
      for (uint c = 0; c < ffertclass_count.size(); c++)
        ffertclass_count[c].set_constant(0.0);
    }

    max_perplexity = 0.0;

    double hillclimbtime = 0.0;
    double countcollecttime = 0.0;

    for (size_t s = 0; s < nSentences; s++) {

      if ((s % 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();

      Math1D::Vector<AlignBaseType>& cur_alignment = best_known_alignment_[s];

      //std::cerr << "J=" << curJ << ", I=" << curI << std::endl;

      Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ, curJ, MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ, curI + 1, MAKENAME(expansion_move_prob));

      std::clock_t tHillclimbStart, tHillclimbEnd;
      tHillclimbStart = std::clock();

      long double best_prob = 0.0;

      //std::cerr << "calling hillclimbing" << std::endl;

      //if (hillclimb_mode_ == HillclimbingRestart)
      //  best_known_alignment_[s] = initial_alignment[s];

      if (fert_trainer != 0 && iter == 1) {
        //std::cerr << "calling IBM-4 hillclimbing" << std::endl;
        best_prob = fert_trainer->update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter,
                    fertility, expansion_move_prob, swap_move_prob, cur_alignment);
      }
      else {
        best_prob = update_alignment_by_hillclimbing(cur_source, cur_target, cur_lookup, sum_iter, fertility,
                    expansion_move_prob, swap_move_prob, cur_alignment);
      }
      max_perplexity -= logl(best_prob);

      //std::cerr << "back from hillclimbing" << std::endl;
      //std::cerr << "alignment: " << best_known_alignment_[s] << std::endl;

      tHillclimbEnd = std::clock();

      hillclimbtime += diff_seconds(tHillclimbEnd, tHillclimbStart);

      /**** update empty word counts *****/

      fzero_count += fertility[0];
      fnonzero_count += curJ - 2 * fertility[0];

      /**** update fertility counts *****/
      for (uint i = 1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i - 1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }

      /**** update dictionary counts *****/
      for (uint j = 0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = cur_alignment[j];

        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj - 1]][cur_lookup(j, cur_aj - 1)] += 1.0;
        }
        else {
          fwcount[0][s_idx - 1] += 1.0;
        }
      }

      /**** update distortion counts *****/

      NamedStorage1D<std::vector<ushort> > aligned_source_words(curI + 1, MAKENAME(aligned_source_words));
      for (uint j = 0; j < curJ; j++)
        aligned_source_words[cur_alignment[j]].push_back(j);

      {
        uint prev_center = MAX_UINT;

        Storage1D<bool> fixed(curJ, false);

        uint nOpen = curJ;

        for (uint i = 1; i <= curI; i++) {

          if (fertility[i] > 0) {

            const std::vector<ushort>& cur_aligned_source_words = aligned_source_words[i];
            assert(cur_aligned_source_words.size() == fertility[i]);

            //a) head of the cept
            const uint first_j = cur_aligned_source_words[0];

            if (prev_center != MAX_UINT) {      // currently not estimating a start probability

              const uint sclass = source_class_[cur_source[first_j]];
              const uint nAvailable = nOpen - (cur_aligned_source_words.size() - 1);

              Math3D::Tensor<double>& cur_inter_distortion_count = inter_distortion_count[nAvailable];

              uint pos_first_j = MAX_UINT;
              uint pos_prev_center = MAX_UINT;

              uint nCurOpen = 0;
              for (uint j = 0; j <= std::max(first_j, prev_center); j++) {

                if (j == first_j)
                  pos_first_j = nCurOpen;

                if (!fixed[j])
                  nCurOpen++;

                if (j == prev_center)
                  pos_prev_center = nCurOpen;
              }

              cur_inter_distortion_count(pos_first_j, pos_prev_center, sclass) += 1.0;
              single_diff_count(displacement_offset_ + pos_first_j - pos_prev_center, sclass) += 1.0;
              inter_span_count(displacement_offset_ - pos_prev_center, displacement_offset_ + nAvailable - 1 - pos_prev_center, sclass) += 1.0;
              
			  single_pos_count(pos_first_j, pos_prev_center, sclass) += 1.0;
			  inter_pos_span_count(nAvailable, pos_prev_center, sclass) += 1.0;
			}
            else if (use_sentence_start_prob_) {
              sentence_start_count[curJ][first_j] += 1.0;
              fsentence_start_count[first_j] += 1.0;
              fstart_span_count[curJ - 1] += 1.0;
            }

            fixed[first_j] = true;
            nOpen--;

            //b) body of the cept
            for (uint k = 1; k < cur_aligned_source_words.size(); k++) {

              const uint cur_j = cur_aligned_source_words[k];

              const uint sclass = source_class_[cur_source[cur_j]];
  			  const uint tclass = target_class_[cur_target[i-1]];
			  const uint cur_class = (intra_dist_mode_ == IBM4IntraDistModeSource) ? sclass : tclass;

              uint pos = MAX_UINT;

              uint nAvailable = 0;
              for (uint j = cur_aligned_source_words[k - 1] + 1; j < curJ; j++) {

                if (j == cur_j)
                  pos = nAvailable;

                if (!fixed[j])
                  nAvailable++;
              }

              nAvailable -= cur_aligned_source_words.size() - 1 - k;

              intra_distortion_count[nAvailable](pos, cur_class) += 1.0;
              intra_distparam_count(pos, cur_class) += 1.0;
              intra_span_count(nAvailable - 1, cur_class) += 1.0;

              fixed[cur_j] = true;
              nOpen--;
            }

            //c) calculate the center of the cept
            switch (cept_start_mode_) {
            case IBM4CENTER: {

              //compute the center of this cept and store the result in prev_cept_center
              double sum = vec_sum(cur_aligned_source_words);
              prev_center = (int)round(sum / cur_aligned_source_words.size());
              break;
            }
            case IBM4FIRST:
              prev_center = first_j;
              break;
            case IBM4LAST:
              prev_center = cur_aligned_source_words.back();
              break;
            case IBM4UNIFORM:
              break;
            default:
              assert(false);
            }

          }
        }
      }
    }  //loop over sentences finished

    max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
    max_perplexity /= source_sentence_.size();

    std::cerr << (((double)sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" << std::endl;

    std::string transfer = (fert_trainer != 0 && iter == 1) ? " (transfer) " : "";

    std::cerr << "IBM-5 max-perplex-energy in between iterations #" << (iter - 1)
              << " and " << iter << transfer << ": " << max_perplexity << std::endl;

    /***** update probability models from counts *******/

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      std::cerr << "zero counts: " << fzero_count << ", " << fnonzero_count << std::endl;
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = std::max(fert_min_p0,fzero_count / fsum);
      p_nonzero_ = std::max(fert_min_p0,fnonzero_count / fsum);
    }

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    assert(!isnan(p_zero_));
    assert(!isnan(p_nonzero_));

    //DEBUG
    uint nZeroAlignments = 0;
    uint nAlignments = 0;
    for (size_t s = 0; s < source_sentence_.size(); s++) {

      nAlignments += source_sentence_[s].size();

      for (uint j = 0; j < source_sentence_[s].size(); j++) {
        if (best_known_alignment_[s][j] == 0)
          nZeroAlignments++;
      }
    }
    std::cerr << "percentage of zero-aligned words: "
              << (((double)nZeroAlignments) / ((double)nAlignments)) << std::endl;
    //END_DEBUG

    //update dictionary
    update_dict_from_counts(fwcount, prior_weight_, nSentences, 0.0, false, 0.0, 0, dict_, fert_min_dict_entry,
                            msolve_mode_ != MSSolvePGD, gd_stepsize_);

    //update fertility probabilities
    update_fertility_prob(ffert_count, fert_min_param_entry, false);    //needed at least with fert-prob-sharing

    //update inter distortion probabilities
    if (distortion_type_ == IBM23Nonpar) {

	  const uint nDistGroupingTerms = maxJ_+1 - (dist_grouping_limit_+1);

      for (uint J = 1; J < inter_distortion_count.size(); J++) {

        for (uint y = 0; y < inter_distortion_count[J].yDim(); y++) {
          for (uint z = 0; z < inter_distortion_count[J].zDim(); z++) {

            double sum = 0.0;
            for (uint j = 0; j < inter_distortion_count[J].xDim(); j++)
              sum += inter_distortion_count[J](j, y, z);

            if (sum > 1e-305) {

			  if (inter_dist_grouping_ != DistGroupModeExpand) {
                for (uint j = 0; j < inter_distortion_count[J].xDim(); j++)
                  inter_distortion_prob_[J](j, y, z) = std::max(fert_min_param_entry, inter_distortion_count[J](j, y, z) / sum);
			  }
  			  else {
				double longparam_sum = 0.0;
			    for (uint j = dist_grouping_limit_+1; j < inter_distortion_count[J].xDim(); j++)
				  longparam_sum += inter_distortion_count[J](j, y, z);
			    if (inter_dist_grouping_ == DistGroupModeDivide) {
  			      sum -= longparam_sum;
				  longparam_sum /= nDistGroupingTerms;
				  sum += longparam_sum;
				}
			  			  
			    for (uint j = 0; j <= std::min<uint>(dist_grouping_limit_,J-1); j++)
				  inter_distortion_prob_[J](j, y, z) = std::max(fert_min_param_entry, inter_distortion_count[J](j, y, z) / sum);
			    for (uint j = dist_grouping_limit_+1; j < inter_distortion_count[J].xDim(); j++)
				  inter_distortion_prob_[J](j, y, z) = longparam_sum / (sum * (inter_distortion_count[J].xDim() - (dist_grouping_limit_+1)));
			  }
			}
          }
        }
      }
    }
    else {

      for (uint s = 0; s < inter_distortion_param_.yDim(); s++) {

		if (distortion_type_ == IBM23ParByDifference)
          inter_distortion_diffpar_m_step(single_diff_count, inter_span_count, s);
        else  
		{
		  for (uint prev_pos = 0; prev_pos <= maxJ_; prev_pos++)
  		    inter_distortion_pospar_m_step(single_pos_count, inter_pos_span_count, s, prev_pos);
		}
	  }

      par2nonpar_inter_distortion();
    }

    //update intra distortion probabilities
    if (!uniform_intra_prob_) {
      if (distortion_type_ == IBM23Nonpar) {

        for (uint J = 1; J < intra_distortion_prob_.size(); J++) {

          for (uint s = 0; s < intra_distortion_prob_[J].yDim(); s++) {

            double sum = intra_distortion_count[J].row_sum(s);

            if (sum > 1e-305) {
              for (uint j = 0; j < J; j++)
                intra_distortion_prob_[J] (j, s) = std::max(fert_min_param_entry, intra_distortion_count[J] (j, s) / sum);
            }
          }
        }
      }
      else {

        Math2D::Matrix<double> hyp_param = intra_distortion_param_;

        // call m-steps
        for (uint s = 0; s < intra_distortion_param_.yDim(); s++) {

          intra_distortion_m_step(intra_distparam_count, intra_span_count, s);
        }

        par2nonpar_intra_distortion();
      }
    }

    if (use_sentence_start_prob_) {
      if (msolve_mode_ == MSSolvePGD)
        start_prob_m_step(fsentence_start_count, fstart_span_count, sentence_start_parameters_, start_m_step_iter_, gd_stepsize_);
      else
        start_prob_m_step_unconstrained(fsentence_start_count, fstart_span_count, sentence_start_parameters_, start_m_step_iter_);
      par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
    }

    if (fert_trainer == 0) { // no point doing ICM in a transfer iteration

      std::cerr << "starting ICM" << std::endl;

      const double log_pzero = std::log(p_zero_);
      const double log_pnonzero = std::log(p_nonzero_);

      Math1D::NamedVector < uint > dict_sum(fwcount.size(), MAKENAME(dict_sum));
      for (uint k = 0; k < fwcount.size(); k++)
        dict_sum[k] = fwcount[k].sum();

      if (fertprob_sharing_) {

        for (uint i = 1; i < nTargetWords; i++) {
          uint c = tfert_class_[i];
          for (uint k = 0; k < ffert_count[i].size(); k++)
            ffertclass_count[c][k] += ffert_count[i][k];
        }
      }

      uint nSwitches = 0;

      for (size_t s = 0; s < nSentences; s++) {

        if ((s % 10000) == 0)
          std::cerr << "sentence pair #" << s << std::endl;

        const Storage1D<uint>& cur_source = source_sentence_[s];
        const Storage1D<uint>& cur_target = target_sentence_[s];
        const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc_, nSourceWords_, slookup_[s], aux_lookup);

        Math1D::Vector < AlignBaseType >& cur_best_known_alignment = best_known_alignment_[s];

        const uint curI = cur_target.size();
        const uint curJ = cur_source.size();

        Math1D::NamedVector<uint> fertility(curI + 1, 0, MAKENAME(fertility));

        for (uint j = 0; j < curJ; j++)
          fertility[cur_best_known_alignment[j]]++;

        NamedStorage1D<std::vector<ushort> > hyp_aligned_source_words(curI + 1, MAKENAME(hyp_aligned_source_words));

        for (uint j = 0; j < curJ; j++) {

          uint aj = cur_best_known_alignment[j];
          hyp_aligned_source_words[aj].push_back(j);
        }

        double cur_neglog_distort_prob = -logl(distortion_prob(cur_source, cur_target, hyp_aligned_source_words));

        for (uint j = 0; j < curJ; j++) {

          const uint cur_aj = best_known_alignment_[s][j];
          const uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj - 1];
          const uint cur_idx = (cur_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, cur_aj - 1);
          std::vector<ushort>& cur_hyp_aligned_source_words = hyp_aligned_source_words[cur_aj];
          Math1D::Vector<double>& cur_fert_count = ffert_count[cur_word];
          Math1D::Vector<double>& cur_dictcount = fwcount[cur_word];

          double best_change = 0.0;
          uint new_aj = cur_aj;

          for (uint i = 0; i <= curI; i++) {

            /**** dict ***/
            //std::cerr << "i: " << i << ", cur_aj: " << cur_aj << std::endl;

            bool allowed = (cur_aj != i && (i != 0 || 2 * fertility[0] + 2 <= curJ));

            if (i != 0 && (fertility[i] + 1) > fertility_limit_[cur_word])
              allowed = false;

            if (allowed) {

              const uint new_target_word = (i == 0) ? 0 : cur_target[i - 1];
              const Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

              vec_erase<ushort>(cur_hyp_aligned_source_words, j);
              sorted_vec_insert<ushort>(hyp_aligned_source_words[i], j);

              //std::cerr << "cur_word: " << cur_word << std::endl;
              //std::cerr << "new_word: " << new_target_word << std::endl;

              double change = 0.0;

              const Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
              const uint hyp_idx = (i == 0) ? cur_source[j] - 1 : cur_lookup(j, i - 1);

              change += common_icm_change(fertility, log_pzero, log_pnonzero, dict_sum, cur_dictcount, hyp_dictcount,
                                          prior_weight_[cur_word], prior_weight_[new_target_word], cur_fert_count, hyp_fert_count, ffertclass_count,
                                          cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, i, curJ);

              //std::cerr << "dist" << std::endl;

              /***** distortion ****/
              change -= cur_neglog_distort_prob;
              change -= logl(distortion_prob(cur_source, cur_target, hyp_aligned_source_words));

              if (change < best_change) {
                best_change = change;
                new_aj = i;
              }
              //rollback
              vec_erase<ushort>(hyp_aligned_source_words[i], j);
              sorted_vec_insert<ushort>(cur_hyp_aligned_source_words, j);
            }
          }

          if (best_change < -0.01) {

            //std::cerr << "changing!!" << std::endl;

            cur_best_known_alignment[j] = new_aj;
            nSwitches++;

            const uint new_target_word = (new_aj == 0) ? 0 : cur_target[new_aj - 1];
            Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word];
            const uint hyp_idx = (new_aj == 0) ? cur_source[j] - 1 : cur_lookup(j, new_aj - 1);
            Math1D::Vector<double>& hyp_fert_count = ffert_count[new_target_word];

            common_icm_count_change(dict_sum, cur_dictcount, hyp_dictcount, cur_fert_count, hyp_fert_count,
                                    ffertclass_count, cur_word, new_target_word, cur_idx, hyp_idx, cur_aj, new_aj, fertility);

            if (cur_aj == 0) {
              fnonzero_count += 2.0;
              fzero_count--;
            }

            if (new_aj == 0) {
              fnonzero_count -= 2.0;
              fzero_count++;
            }

            vec_erase<ushort>(cur_hyp_aligned_source_words, j);
            sorted_vec_insert<ushort>(hyp_aligned_source_words[new_aj], j);

            cur_neglog_distort_prob = -logl(distortion_prob(cur_source, cur_target, hyp_aligned_source_words));
          }
        }
      }  //ICM-loop over sentences finished

      std::cerr << nSwitches << " changes in ICM stage" << std::endl;

      //update p_zero_ and p_nonzero_
      if (!fix_p0_) {
        std::cerr << "zero counts: " << fzero_count << ", " << fnonzero_count << std::endl;
        double fsum = fzero_count + fnonzero_count;
        p_zero_ = std::max(fert_min_p0,fzero_count / fsum);
        p_nonzero_ = std::max(fert_min_p0,fnonzero_count / fsum);
      }

      //update fertility probabilities
      update_fertility_prob(ffert_count, fert_min_param_entry, false);

      //update dictionary
      update_dict_from_counts(fwcount, prior_weight_, nSentences, 0.0, false, 0.0, 0, dict_, fert_min_dict_entry);

      //TODO: think about whether to update distortions parameters here as well (would need to update the counts)

      max_perplexity = 0.0;
      for (size_t s = 0; s < source_sentence_.size(); s++) {
        max_perplexity -= logl(FertilityModelTrainer::alignment_prob(s, best_known_alignment_[s]));
      }

      max_perplexity += exact_l0_reg_term(fwcount, ffert_count);
      max_perplexity /= source_sentence_.size();

      std::cerr << "IBM-5 max-perplex-energy after iteration #" << iter << transfer << ": " << max_perplexity << std::endl;
    }

    if (possible_ref_alignments_.size() > 0) {

      std::cerr << "#### IBM-5-AER after iteration #" << iter << transfer << ": "
                << FertilityModelTrainerBase::AER() << std::endl;
      std::cerr << "#### IBM-5-fmeasure after iteration #" << iter << transfer
                << ": " << FertilityModelTrainerBase::f_measure() << std::endl;
      std::cerr << "#### IBM-5-DAE/S after iteration #" << iter << transfer
                << ": " << FertilityModelTrainerBase::DAE_S() << std::endl;
    }
  }

  if (distortion_type_ == IBM23Nonpar) {

    //we still update <code> inter_distortion_param_ </code>  and <code> intra_distortion_param_ </code>
    //so that we can use them when computing external alignments

    //a) inter
    inter_distortion_param_.set_constant(0.0);

    for (uint s = 0; s < inter_distortion_param_.yDim(); s++) {

      for (uint J = 0; J < inter_distortion_count.size(); J++) {

        for (uint prev_pos = 0; prev_pos < inter_distortion_count[J].yDim();
             prev_pos++) {

          for (uint j = 0; j < J; j++) {

            double cur_weight = inter_distortion_count[J] (j, prev_pos, s);
            inter_distortion_param_(displacement_offset_ + j - prev_pos, s) += cur_weight;
          }
        }
      }

      double sum = 0.0;
      for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
        sum += inter_distortion_param_(j, s);

      if (sum > 1e-305) {

        for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
          inter_distortion_param_(j, s) = std::max(1e-8, inter_distortion_param_(j, s) / sum);
      }
      else {
        for (uint j = 0; j < inter_distortion_param_.xDim(); j++)
          inter_distortion_param_(j, s) = 1.0 / inter_distortion_param_.xDim();
      }
    }

    //b) intra
    intra_distortion_param_.set_constant(0.0);

    for (uint s = 0; s < intra_distortion_param_.yDim(); s++) {

      for (uint J = 0; J < intra_distortion_count.size(); J++) {

        for (uint j = 0; j < J; j++)
          intra_distortion_param_(j, s) += intra_distortion_count[J] (j, s);
      }

      double sum = 0.0;
      for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
        sum += intra_distortion_param_(j, s);

      if (sum > 1e-305) {

        for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
          intra_distortion_param_(j, s) =
            std::max(1e-8, intra_distortion_param_(j, s) / sum);
      }
      else {

        for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
          intra_distortion_param_(j, s) = 1.0 / intra_distortion_param_.xDim();
      }
    }
  }

  iter_offs_ = iter - 1;
}

/* virtual */
void IBM5Trainer::prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
    const SingleLookupTable& lookup, Math1D::Vector<AlignBaseType>& alignment)
{
  const uint J = source.size();
  const uint I = target.size();

  common_prepare_external_alignment(source, target, lookup, alignment);

  //DEBUG
  Math1D::Vector<uint> fertility(I + 1, 0);

  for (uint j=0; j < J; j++)
    fertility[alignment[j]]++;

  for (uint i = 1; i <= I; i++) {

    assert(fertility[i] <= fertility_limit_[target[i - 1]]);
    const double prob = fertility_prob_[target[i - 1]][fertility[i]];
    if (! (prob > 0.0 && prob <= 1.0 ) ) {
      std::cerr << "fertility prob distribution: " << fertility_prob_[target[i - 1]] << std::endl;
    }

    assert(prob > 0.0 && prob <= 1.0 );
    assert(fertility[i] <= fertility_limit_[target[i-1]]);
  }
  //END_DEBUG

  /*** check if distortion tables are large enough ***/

  if (J > maxJ_) {

	TODO("intergrate pospar and grouping");

    Math2D::Matrix<double> new_inter_param(2 * J - 1, nSourceClasses_, 1e-8);

    for (uint s = 0; s < inter_distortion_param_.yDim(); s++) {

      for (int j = -int (maxJ_-1); j <= int (maxJ_-1); j++)
        new_inter_param(j + J, s) = inter_distortion_param_(j + displacement_offset_, s);
    }

	if (distortion_type_ == IBM23ParByDifference)
      displacement_offset_ = J-1;

    inter_distortion_param_ = new_inter_param;
    inter_distortion_prob_.resize(J + 1);
	inter_grouping_prob_.resize(J+1,1e-8);

    for (uint JJ = 1; JJ < inter_distortion_prob_.size(); JJ++) {

      if (inter_distortion_prob_[JJ].yDim() < J)
        inter_distortion_prob_[JJ].resize(inter_distortion_prob_[JJ].xDim(), J, nSourceClasses_, 1e-8);
    }

    for (uint JJ = 1; JJ <= J; JJ++) {

      uint prev_yDim = inter_distortion_prob_[JJ].yDim();

      inter_distortion_prob_[JJ].resize(JJ, J, nSourceClasses_, 1e-8);

      if (distortion_type_ == IBM23Nonpar) { //otherwise the update is done by par2nonpar_.. below

        for (uint y = prev_yDim; y < inter_distortion_prob_[JJ].yDim(); y++) {
          for (uint z = 0; z < inter_distortion_prob_[JJ].zDim(); z++) {

            double sum = 0.0;
            for (int j = 0; j < int (inter_distortion_prob_[JJ].xDim()); j++)
              sum += inter_distortion_param_(j - y + displacement_offset_, z);

            assert(sum > 1e-305);

            if (sum > 1e-305) {

              for (uint j = 0; j < inter_distortion_prob_[JJ].xDim(); j++)
                inter_distortion_prob_[JJ] (j, y, z) =
                  std::max(1e-8, inter_distortion_param_(j - y + displacement_offset_, z) / sum);
            }
          }
        }
      }
    }

    if (distortion_type_ != IBM23Nonpar)
      par2nonpar_inter_distortion();

    Math2D::Matrix<double> new_intra_param(J, nSourceClasses_, 1e-8);
    for (uint s = 0; s < intra_distortion_param_.yDim(); s++)
      for (uint j = 0; j < intra_distortion_param_.xDim(); j++)
        new_intra_param(j, s) = intra_distortion_param_(j, s);

    intra_distortion_param_ = new_intra_param;
    intra_distortion_prob_.resize(J + 1);

    for (uint JJ = 1; JJ <= J; JJ++) {

      intra_distortion_prob_[JJ].resize(JJ, nSourceClasses_, 1e-8);

      if (distortion_type_ == IBM23Nonpar && JJ > maxJ_) {   //for parametric distortion the update is done by par2nonpar_... below

        for (uint s = 0; s < intra_distortion_param_.yDim(); s++) {

          double sum = 0.0;
          for (uint j = 0; j < JJ; j++)
            sum += intra_distortion_param_(j, s);

          for (uint j = 0; j < JJ; j++)
            intra_distortion_prob_[JJ](j, s) = intra_distortion_param_(j, s) / sum;
        }
      }
    }

    if (distortion_type_ == IBM23Nonpar)
      par2nonpar_intra_distortion();

    maxJ_ = J;
  }

  if (use_sentence_start_prob_) {

    if (sentence_start_prob_.size() <= J) {
      sentence_start_prob_.resize(J + 1);
    }

    if (sentence_start_prob_[J].size() < J) {
      sentence_start_prob_[J].resize(J, 1.0 / J);
    }

    par2nonpar_start_prob(sentence_start_parameters_, sentence_start_prob_);
  }

}
