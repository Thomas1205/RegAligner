/*** ported here from singleword_fertility_training ****/
/** author: Thomas Schoenemann. This file was generated while Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012 ***/

#include "ibm4_training.hh"

#include "combinatoric.hh"
#include "timing.hh"
#include "projection.hh"
#include "ibm1_training.hh" //for the dictionary m-step

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

#include <fstream>
#include <set>
#include "stl_out.hh"


IBM4Trainer::IBM4Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
                         const Storage1D<Math2D::Matrix<uint> >& slookup,
                         const Storage1D<Storage1D<uint> >& target_sentence,
                         const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
                         const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
                         SingleWordDictionary& dict,
                         const CooccuringWordsType& wcooc,
                         uint nSourceWords, uint nTargetWords,
                         const floatSingleWordDictionary& prior_weight,
                         const Storage1D<WordClassType>& source_class,
                         const Storage1D<WordClassType>& target_class,
                         bool och_ney_empty_word,
                         bool use_sentence_start_prob,
                         bool no_factorial, 
                         bool reduce_deficiency,
                         IBM4CeptStartMode cept_start_mode, bool smoothed_l0, double l0_beta, double l0_fertpen)
  : FertilityModelTrainer(source_sentence,slookup,target_sentence,dict,wcooc,
                          nSourceWords,nTargetWords,sure_ref_alignments,possible_ref_alignments),
    cept_start_prob_(MAKENAME(cept_start_prob_)),
    within_cept_prob_(MAKENAME(within_cept_prob_)), 
    sentence_start_parameters_(MAKENAME(sentence_start_parameters_)),
    source_class_(source_class), target_class_(target_class),
    och_ney_empty_word_(och_ney_empty_word), cept_start_mode_(cept_start_mode),
    use_sentence_start_prob_(use_sentence_start_prob), no_factorial_(no_factorial), reduce_deficiency_(reduce_deficiency),
    prior_weight_(prior_weight), smoothed_l0_(smoothed_l0), l0_beta_(l0_beta), l0_fertpen_(l0_fertpen), fix_p0_(false)
{

  const uint nDisplacements = 2*maxJ_-1;
  displacement_offset_ = maxJ_-1;

  uint max_source_class = 0;
  for (uint j=0; j < source_class_.size(); j++) {
    max_source_class = std::max<uint>(max_source_class,source_class_[j]);
  }

  uint max_target_class = 0;
  for (uint i=0; i < target_class_.size(); i++) {
    max_target_class = std::max<uint>(max_target_class,target_class_[i]);
  }

  nSourceClasses_ = max_source_class+1;
  nTargetClasses_ = max_target_class+1;

  cept_start_prob_.resize(nSourceClasses_,nTargetClasses_,2*maxJ_-1);
  within_cept_prob_.resize(nTargetClasses_,maxJ_);

  cept_start_prob_.set_constant(1.0 / nDisplacements);


  within_cept_prob_.set_constant(1.0 / (maxJ_-1));
  for (uint x=0; x < within_cept_prob_.xDim(); x++)
    within_cept_prob_(x,0) = 0.0;

  if (use_sentence_start_prob_) {
    sentence_start_parameters_.resize(maxJ_, 1.0 / maxJ_);
    sentence_start_prob_.resize(maxJ_+1);
  }

  std::set<uint> seenJs;
  for (uint s=0; s < source_sentence.size(); s++)
    seenJs.insert(source_sentence[s].size());

  inter_distortion_prob_.resize(maxJ_+1);
  intra_distortion_prob_.resize(maxJ_+1);


  for (uint J=1; J <= maxJ_; J++) {
    if (seenJs.find(J) != seenJs.end()) {

      sentence_start_prob_[J].resize(J,0.0);

      for (uint j=0; j < J; j++)
        sentence_start_prob_[J][j] = sentence_start_parameters_[j];

      inter_distortion_prob_[J].resize(1,1);
    }
  }

  for (uint s=0; s < source_sentence_.size(); s++) {

    const uint curJ = source_sentence_[s].size();
    const uint curI = target_sentence_[s].size();

    uint max_t = 0;
    
    for (uint i=0; i < curI; i++) {
      const uint tclass = target_class_[target_sentence_[s][i]];
      max_t = std::max(max_t,tclass);
    }

    if (intra_distortion_prob_[curJ].xDim() < max_t+1)
      intra_distortion_prob_[curJ].resize_dirty(max_t+1,curJ,curJ);

    for (uint j=0; j < curJ; j++) {
      const uint sclass = source_class_[source_sentence_[s][j]];

      if (inter_distortion_prob_[curJ].xDim() <= sclass || inter_distortion_prob_[curJ].yDim() <= max_t)
        inter_distortion_prob_[curJ].resize(std::max<uint>(inter_distortion_prob_[curJ].xDim(),sclass+1),
                                            std::max<uint>(inter_distortion_prob_[curJ].yDim(),max_t+1));

      for (uint i=0; i < curI; i++) {
        const uint tclass = target_class_[target_sentence_[s][i]];

        if (curJ <= 10 || nSourceClasses_ == 1 || nTargetClasses_ == 1) {
          
          if (inter_distortion_prob_[curJ](sclass,tclass).size() == 0) {
            inter_distortion_prob_[curJ](sclass,tclass).resize(curJ,curJ);
            
            for (int j1=0; j1 < (int) curJ; j1++) {
              for (int j2=0; j2 < (int) curJ; j2++) {
                inter_distortion_prob_[curJ](sclass,tclass)(j2,j1) = cept_start_prob_(sclass,tclass,j2-j1+displacement_offset_);
              }
            }
          }
        }
      }
    }
  }


  for (uint J=1; J <= maxJ_; J++) {
    if (seenJs.find(J) != seenJs.end()) {

      for (int j1=0; j1 < (int) J; j1++) {
        for (int j2=j1+1; j2 < (int) J; j2++) {

          for (uint y = 0; y < intra_distortion_prob_[J].xDim(); y++)
            intra_distortion_prob_[J](y,j2,j1) = within_cept_prob_(y,j2-j1);
        }
      }
    
    }
  }
}

void IBM4Trainer::fix_p0(double p0) {
  p_zero_ = p0;
  p_nonzero_ = 1.0 - p0;
  fix_p0_ = true;
}


void IBM4Trainer::par2nonpar_inter_distortion() {

  for (int J=1; J <= (int) maxJ_; J++) {

    if (inter_distortion_prob_[J].size() > 0) {

      for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++) {
        for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++) {

          if (inter_distortion_prob_[J](x,y).size() > 0) {

            assert(inter_distortion_prob_[J](x,y).xDim() == uint(J) && inter_distortion_prob_[J](x,y).yDim() == uint(J));

            if (reduce_deficiency_) {

              for (int j1=0; j1 < J; j1++) {
              
                double sum = 0.0;
                
                for (int j2=0; j2 < J; j2++) {
                  sum += cept_start_prob_(x,y,j2-j1+displacement_offset_);
                  assert(!isnan(sum));
                }
                
                if (sum > 1e-305) {
                  for (int j2=0; j2 < J; j2++) {
                    inter_distortion_prob_[J](x,y)(j2,j1) = 
                      std::max(1e-8,cept_start_prob_(x,y,j2-j1+displacement_offset_) / sum);
                  }
                }
                else if (j1 > 0) {
                  //std::cerr << "WARNING: sum too small for inter prob " << j1 << ", not updating." << std::endl;
                }
              }
            }
            else {
              for (int j1=0; j1 < J; j1++) 
                for (int j2=0; j2 < J; j2++) 
                  inter_distortion_prob_[J](x,y)(j2,j1) = cept_start_prob_(x,y,j2-j1+displacement_offset_);
            }
          }
        }
      }
    }
  }
}

void IBM4Trainer::par2nonpar_inter_distortion(int J, uint sclass, uint tclass) {

  if (inter_distortion_prob_[J].xDim() <= sclass || inter_distortion_prob_[J].xDim() <= tclass)
    inter_distortion_prob_[J].resize(std::max<uint>(inter_distortion_prob_[J].xDim(),sclass+1),
                                     std::max<uint>(inter_distortion_prob_[J].yDim(),tclass+1));
  if (inter_distortion_prob_[J](sclass,tclass).size() == 0)
    inter_distortion_prob_[J](sclass,tclass).resize(J,J,1.0 / J);

  if (reduce_deficiency_) {
  
    for (int j1=0; j1 < J; j1++) {
      
      double sum = 0.0;
      
      for (int j2=0; j2 < J; j2++) {
        sum += cept_start_prob_(sclass,tclass,j2-j1+displacement_offset_);
        assert(!isnan(sum));
      }
      
      if (sum > 1e-305) {
        for (int j2=0; j2 < J; j2++) {
          inter_distortion_prob_[J](sclass,tclass)(j2,j1) = 
            std::max(1e-8,cept_start_prob_(sclass,tclass,j2-j1+displacement_offset_) / sum);
        }
      }
      else if (j1 > 0) {
        //std::cerr << "WARNING: sum too small for inter prob " << j1 << ", not updating." << std::endl;
      }
    }
  }
  else {
    for (int j1=0; j1 < J; j1++) 
      for (int j2=0; j2 < J; j2++) 
        inter_distortion_prob_[J](sclass,tclass)(j2,j1) = cept_start_prob_(sclass,tclass,j2-j1+displacement_offset_);
  }
}


void IBM4Trainer::par2nonpar_intra_distortion() {

  for (int J=1; J <= (int) maxJ_; J++) {

    if (intra_distortion_prob_[J].size() > 0) {

      for (uint x=0; x < intra_distortion_prob_[J].xDim(); x++) {

        if (reduce_deficiency_) {

          for (int j1=0; j1 < J-1; j1++) {
            
            double sum = 0.0;
            
            for (int j2=j1+1; j2 < J; j2++) {
              sum += within_cept_prob_(x,j2-j1);
            }
            
            if (sum > 1e-305) {
              for (int j2=j1+1; j2 < J; j2++) {
                intra_distortion_prob_[J](x,j2,j1) = std::max(1e-8,within_cept_prob_(x,j2-j1) / sum);
              }
            }
            else {
              std::cerr << "WARNING: sum too small for intra prob " << j1 << ", J=" << J << ", not updating." << std::endl;
            }
          }
	}
        else {
          for (int j1=0; j1 < J-1; j1++)
            for (int j2=j1+1; j2 < J; j2++)
              intra_distortion_prob_[J](x,j2,j1) = within_cept_prob_(x,j2-j1);
        }
      }
    }
  }
}

void IBM4Trainer::par2nonpar_start_prob() {

  for (uint J=1; J <= maxJ_; J++) {
    if (sentence_start_prob_[J].size() > 0) {

      double sum = 0.0;

      for (uint j=0; j < J; j++)
        sum += sentence_start_parameters_[j];

      if (sum > 1e-305) {
        double inv_sum = 1.0 / sum;
        for (uint j=0; j < J; j++)
          sentence_start_prob_[J][j] = std::max(1e-8,inv_sum * sentence_start_parameters_[j]);
      }
      else {
        std::cerr << "WARNING: sum too small for start prob " << J << ", not updating." << std::endl;
      }
    }
  }
}

double IBM4Trainer::inter_distortion_m_step_energy(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                                                   const std::map<DistortCount,double>& sparse_inter_distort_count,
                                                   const Math3D::Tensor<double>& inter_param, uint class1, uint class2) {

  double energy = 0.0;

  for (int J=1; J <= (int) maxJ_; J++) {

    if (inter_distort_count[J].xDim() > class1 && inter_distort_count[J].yDim() > class2) {

      const Math2D::Matrix<double>& cur_count = inter_distort_count[J](class1,class2);

      if (cur_count.size() == 0)
        continue;

      for (int j1=0; j1 < J; j1++) {

        double sum = 0.0;

        for (int j2=0; j2 < J; j2++) {
          sum += std::max(1e-15,inter_param(class1,class2,j2-j1 + displacement_offset_));
        }


        for (int j2=0; j2 < J; j2++) {

          const double count = cur_count(j2,j1);
          if (count == 0.0)
            continue;

          const double cur_param = std::max(1e-15, inter_param(class1,class2,j2-j1 + displacement_offset_));

          energy -= count * std::log( cur_param / sum);
          if (isnan(energy)) {

            std::cerr << "j1: " << j1 << ", j2: " << j2 << std::endl;
            std::cerr << "added " << cur_count(j2,j1) << "* log(" << cur_param 
                      << "/" << sum << ")" << std::endl;
          }
          assert(!isnan(energy));
        }
      }
    }
  }

  for (std::map<DistortCount,double>::const_iterator it = sparse_inter_distort_count.begin(); it != sparse_inter_distort_count.end(); it++) {

    const DistortCount& dist_count = it->first;
    const double weight = it->second;

    uchar J = dist_count.J_;
    int j1 = dist_count.j_prev_;

    double sum = 0.0;

    for (int j2=0; j2 < J; j2++) {
      sum += std::max(1e-15,inter_param(class1,class2,j2-j1 + displacement_offset_));
    }
    
    int j2 = dist_count.j_;

    const double cur_param = std::max(1e-15, inter_param(class1,class2,j2-j1 + displacement_offset_));

    energy -= weight * std::log( cur_param / sum);
    if (isnan(energy)) {
      
      std::cerr << "j1: " << j1 << ", j2: " << j2 << std::endl;
      std::cerr << "added " << weight << "* log(" << cur_param 
                << "/" << sum << ")" << std::endl;
    }
    assert(!isnan(energy));
  }

  return energy;
}

double IBM4Trainer::inter_distortion_m_step_energy(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                                                   const std::vector<std::pair<DistortCount,double> >& sparse_inter_distort_count,
                                                   const Math3D::Tensor<double>& inter_param, uint class1, uint class2) {


  double energy = 0.0;

  for (int J=1; J <= (int) maxJ_; J++) {

    if (inter_distort_count[J].xDim() > class1 && inter_distort_count[J].yDim() > class2) {

      const Math2D::Matrix<double>& cur_count = inter_distort_count[J](class1,class2);

      if (cur_count.size() == 0)
        continue;

      for (int j1=0; j1 < J; j1++) {

        double sum = 0.0;

        for (int j2=0; j2 < J; j2++) {
          sum += std::max(1e-15,inter_param(class1,class2,j2-j1 + displacement_offset_));
        }

        for (int j2=0; j2 < J; j2++) {

          const double count = cur_count(j2,j1);
          if (count == 0.0)
            continue;

          const double cur_param = std::max(1e-15, inter_param(class1,class2,j2-j1 + displacement_offset_));

          energy -= count * std::log( cur_param / sum);
          if (isnan(energy)) {

            std::cerr << "j1: " << j1 << ", j2: " << j2 << std::endl;
            std::cerr << "added " << cur_count(j2,j1) << "* log(" << cur_param 
                      << "/" << sum << ")" << std::endl;
          }
          assert(!isnan(energy));
        }
      }
    }
  }

  for (std::vector<std::pair<DistortCount,double> >::const_iterator it = sparse_inter_distort_count.begin(); 
       it != sparse_inter_distort_count.end(); it++) {

    const DistortCount& dist_count = it->first;
    const double weight = it->second;

    uchar J = dist_count.J_;
    int j1 = dist_count.j_prev_;

    double sum = 0.0;

    for (int j2=0; j2 < J; j2++) {
      sum += std::max(1e-15,inter_param(class1,class2,j2-j1 + displacement_offset_));
    }
    
    int j2 = dist_count.j_;

    const double cur_param = std::max(1e-15, inter_param(class1,class2,j2-j1 + displacement_offset_));

    energy -= weight * std::log( cur_param / sum);
    if (isnan(energy)) {
      
      std::cerr << "j1: " << j1 << ", j2: " << j2 << std::endl;
      std::cerr << "added " << weight << "* log(" << cur_param 
                << "/" << sum << ")" << std::endl;
    }
    assert(!isnan(energy));
  }

  return energy;
}

double IBM4Trainer::intra_distortion_m_step_energy(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                                                   const Math2D::Matrix<double>& intra_param, uint word_class) {


  double energy = 0.0;

  for (int J=1; J <= (int) maxJ_; J++) {

    if (intra_distort_count[J].xDim() > word_class) {

      const Math3D::Tensor<double>& cur_count = intra_distort_count[J];

      for (int j1=0; j1 < J; j1++) {

        double sum = 0.0;

        for (int j2=j1+1; j2 < J; j2++) {
          sum += std::max(1e-15,intra_param(word_class,j2-j1));
        }

        for (int j2=j1+2; j2 < J; j2++) {
          double cur_param = std::max(1e-15, intra_param(word_class,j2-j1));

          energy -= cur_count(word_class,j2,j1) * std::log( cur_param / sum);
        }
      }
    }
  }
  
  return energy;
}

void IBM4Trainer::inter_distortion_m_step(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                                          const std::map<DistortCount,double>& sparse_inter_distort_count,
                                          uint class1, uint class2) {

  //iterating over a vector is a lot faster than iterating over a map -> copy
  std::vector<std::pair<DistortCount,double> > vec_sparse_inter_distort_count;
  vec_sparse_inter_distort_count.reserve(sparse_inter_distort_count.size());
  for (std::map<DistortCount,double>::const_iterator it = sparse_inter_distort_count.begin();
       it != sparse_inter_distort_count.end(); it++)
    vec_sparse_inter_distort_count.push_back(*it);

  Math3D::Tensor<double> new_ceptstart_prob = cept_start_prob_;
  Math3D::Tensor<double> hyp_ceptstart_prob = cept_start_prob_;
  Math1D::Vector<double> ceptstart_grad(cept_start_prob_.zDim());

  double alpha = 0.01;

  double energy = inter_distortion_m_step_energy(inter_distort_count,vec_sparse_inter_distort_count,cept_start_prob_,class1,class2);

  if (nSourceClasses_*nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  uint maxIter = (nSourceClasses_*nTargetClasses_ <= 4) ? 200 : 15;
  
  for (uint iter = 1; iter <= maxIter; iter++) {
    
    ceptstart_grad.set_constant(0.0);

    //compute gradient
    for (int J=1; J <= (int) maxJ_; J++) {

      if (inter_distort_count[J].xDim() > class1 && inter_distort_count[J].yDim() > class2) {

        const Math2D::Matrix<double>& cur_count = inter_distort_count[J](class1,class2);

        if (cur_count.size() == 0)
          continue;

        for (int j1=0; j1 < J; j1++) {

          double sum = 0.0;

          for (int j2=0; j2 < J; j2++) {
            sum += std::max(1e-15,cept_start_prob_(class1,class2,j2-j1 + displacement_offset_));
            assert(!isnan(cept_start_prob_(class1,class2,j2-j1 + displacement_offset_)));
          }

	  if (sum < 1e-100)
	    continue;  //this can happen for j1=0 (and J=1)

          double count_sum = 0.0;
          for (int j2=0; j2 < J; j2++) {

            const double count = cur_count(j2,j1);

            if (count == 0.0)
              continue;

            count_sum += count;

            const double cur_param = std::max(1e-15,cept_start_prob_(class1,class2,j2-j1 + displacement_offset_));
            ceptstart_grad[j2-j1 + displacement_offset_] -= 
              count / cur_param;
            assert(!isnan(ceptstart_grad[j2-j1 + displacement_offset_]));
          }

          for (int j2=0; j2 < J; j2++) {
            ceptstart_grad[j2-j1 + displacement_offset_] += count_sum / sum;
            assert(!isnan(ceptstart_grad[j2-j1 + displacement_offset_]));
          }
        }
      }
    }


    for (std::vector<std::pair<DistortCount,double> >::const_iterator it = vec_sparse_inter_distort_count.begin(); 
         it != vec_sparse_inter_distort_count.end(); it++) {

      const DistortCount& dist_count = it->first;
      const double weight = it->second;
      
      uchar J = dist_count.J_;
      int j1 = dist_count.j_prev_;
      
      double sum = 0.0;
      
      for (int j2=0; j2 < J; j2++) {
        sum += std::max(1e-15,cept_start_prob_(class1,class2,j2-j1 + displacement_offset_));
      }
      
      int j2 = dist_count.j_;
      
      const double cur_param = std::max(1e-15, cept_start_prob_(class1,class2,j2-j1 + displacement_offset_));

      ceptstart_grad[j2-j1 + displacement_offset_] -= weight / cur_param;

      assert(!isnan(ceptstart_grad[j2-j1 + displacement_offset_]));

      for (int jj=0; jj < J; jj++) {
        ceptstart_grad[jj-j1 + displacement_offset_] += weight / sum;
      }
    }

    //go in neg. gradient direction
    for (uint k=0; k < cept_start_prob_.zDim(); k++) {

      new_ceptstart_prob(class1,class2,k) = cept_start_prob_(class1,class2,k) - alpha * ceptstart_grad[k];
    }

    //reproject
    Math1D::Vector<double> temp(cept_start_prob_.zDim());
    for (uint k=0; k < temp.size(); k++)
      temp[k] = new_ceptstart_prob(class1,class2,k);
    
    projection_on_simplex(temp.direct_access(),cept_start_prob_.zDim());

    for (uint k=0; k < temp.size(); k++)
      new_ceptstart_prob(class1,class2,k) = temp[k];
    

    double best_energy = 1e300;
    bool decreasing = true;

    double lambda = 1.0;
    double best_lambda = 1.0;

    while (best_energy > energy || decreasing) {

      lambda *= 0.5;
      double neg_lambda = 1.0 - lambda;

      for (uint k=0; k < cept_start_prob_.zDim(); k++) 
        hyp_ceptstart_prob(class1,class2,k) = neg_lambda * cept_start_prob_(class1,class2,k) 
          + lambda * new_ceptstart_prob(class1,class2,k);

      double hyp_energy = inter_distortion_m_step_energy(inter_distort_count,vec_sparse_inter_distort_count,hyp_ceptstart_prob,class1,class2);

      if (hyp_energy < best_energy) {

        decreasing = true;
        best_lambda = lambda;
        best_energy = hyp_energy;
      }
      else
        decreasing = false;
    }

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k=0; k < cept_start_prob_.zDim(); k++) 
      cept_start_prob_(class1,class2,k) = neg_best_lambda * cept_start_prob_(class1,class2,k) 
        + best_lambda * new_ceptstart_prob(class1,class2,k);

    energy = best_energy;

    if (  (nSourceClasses_*nTargetClasses_ <= 4) && (iter % 5) == 0)
      std::cerr << "iteration " << iter << ", inter energy: " << energy << std::endl;

    if (best_lambda < 1e-5)
      break;
  }
}

void IBM4Trainer::intra_distortion_m_step(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                                          uint word_class) {


  Math2D::Matrix<double> new_within_cept_prob = within_cept_prob_;
  Math2D::Matrix<double> hyp_within_cept_prob = within_cept_prob_;
  Math1D::Vector<double> within_cept_grad(within_cept_prob_.yDim());

  double alpha = 0.01;

  double energy = intra_distortion_m_step_energy(intra_distort_count,within_cept_prob_,word_class);

  if (nTargetClasses_ <= 4)
    std::cerr << "start energy: " << energy << std::endl;

  const uint maxIter = (nTargetClasses_ <= 4) ? 100 : 15;

  for (uint iter = 1; iter <= maxIter; iter++) {
    
    within_cept_grad.set_constant(0.0);

    //calculate gradient

    for (int J=1; J <= (int) maxJ_; J++) {

      const Math3D::Tensor<double>& cur_distort_count = intra_distort_count[J];

      if (cur_distort_count.xDim() > word_class) {

        for (int j1=0; j1 < J; j1++) {

          double sum = 0.0;

          for (int j2=j1+1; j2 < J; j2++) {
            sum += std::max(1e-15,within_cept_prob_(word_class,j2-j1));
          }

          if (sum < 1e-100)
            continue;  //this can happen for j1=0 (and J=1)

          double count_sum = 0.0;
          for (int j2=j1+1; j2 < J; j2++) {

            double cur_count = cur_distort_count(word_class,j2,j1);

            count_sum += cur_count;

            double cur_param = std::max(1e-15,within_cept_prob_(word_class,j2-j1));
            within_cept_grad[j2-j1] -= cur_count / cur_param;
          }

          for (int j2=j1+1; j2 < J; j2++) {
            within_cept_grad[j2-j1] += count_sum / sum;
          }
        }
      }
    }

    //go in neg. gradient direction
    for (uint k=0; k < within_cept_prob_.yDim(); k++) {

      new_within_cept_prob(word_class,k) = within_cept_prob_(word_class,k) - alpha * within_cept_grad[k];
    }

    //reproject
    Math1D::Vector<double> temp(within_cept_prob_.yDim());
    for (uint k=0; k < temp.size(); k++)
      temp[k] = new_within_cept_prob(word_class,k);
    
    projection_on_simplex(temp.direct_access(),temp.size());

    for (uint k=0; k < temp.size(); k++)
      new_within_cept_prob(word_class,k) = temp[k];
    

    double best_energy = 1e300;
    bool decreasing = true;

    double lambda = 1.0;
    double best_lambda = 1.0;

    while (best_energy > energy || decreasing) {

      lambda *= 0.5;
      double neg_lambda = 1.0 - lambda;

      for (uint k=0; k < within_cept_prob_.yDim(); k++)
        hyp_within_cept_prob(word_class,k) = neg_lambda * within_cept_prob_(word_class,k) 
          + lambda * new_within_cept_prob(word_class,k);

      double hyp_energy = intra_distortion_m_step_energy(intra_distort_count,hyp_within_cept_prob,word_class);

      if (hyp_energy < best_energy) {

        decreasing = true;
        best_lambda = lambda;
        best_energy = hyp_energy;
      }
      else
        decreasing = false;
    }

    double neg_best_lambda = 1.0 - best_lambda;

    for (uint k=0; k < within_cept_prob_.yDim(); k++) 
      within_cept_prob_(word_class,k) = neg_best_lambda * within_cept_prob_(word_class,k) 
        + best_lambda * new_within_cept_prob(word_class,k);

    energy = best_energy;

    if ((nTargetClasses_ <= 4) && (iter % 5) == 0)
      std::cerr << "iteration " << iter << ", intra energy: " << energy << std::endl;

    if (best_lambda < 1e-5)
      break;
  }
}

double IBM4Trainer::start_prob_m_step_energy(const Storage1D<Math1D::Vector<double> >& start_count,
                                             Math1D::Vector<double>& param) {

  double energy = 0.0;

  for (uint J=1; J <= maxJ_; J++) {

    if (sentence_start_prob_[J].size() > 0) {

      double sum = 0.0;
      double count_sum = 0.0;

      for (uint j=0; j < J; j++) {
        count_sum += start_count[J][j];
        sum += param[j];
      }

      for (uint j=0; j < J; j++) {
        energy -= start_count[J][j] * std::log(std::max(1e-15,param[j]));
      }

      energy += count_sum * std::log(sum);
    }
  }

  return energy;
}

void IBM4Trainer::start_prob_m_step(const Storage1D<Math1D::Vector<double> >& start_count) {

  Math1D::Vector<double> param_grad = sentence_start_parameters_;
  Math1D::Vector<double> new_param = sentence_start_parameters_;
  Math1D::Vector<double> hyp_param = sentence_start_parameters_;

  double energy = start_prob_m_step_energy(start_count,sentence_start_parameters_);

  std::cerr << "start energy: " << energy << std::endl;

  double alpha = 0.01;

  for (uint iter=1; iter <= 50; iter++) {

    param_grad.set_constant(0.0);

    //calculate gradient
    for (uint J=1; J <= maxJ_; J++) {

      if (sentence_start_prob_[J].size() > 0) {

        double sum = 0.0;
        double count_sum = 0.0;
	
        for (uint j=0; j < J; j++) {
          count_sum += start_count[J][j];
          sum += std::max(1e-15,sentence_start_parameters_[j]);
        }
	
        for (uint j=0; j < J; j++) {

          param_grad[j] -= start_count[J][j] / std::max(1e-15,sentence_start_parameters_[j]);
          param_grad[j] += count_sum / sum;
        }
      }
    }

    //go in neg. gradient direction
    for (uint k=0; k < param_grad.size(); k++)
      new_param[k] = sentence_start_parameters_[k] - alpha * param_grad[k];

    //reproject
    projection_on_simplex(new_param.direct_access(), new_param.size());

    //find step-size
    double best_energy = 1e300;
    bool decreasing = true;

    double lambda = 1.0;
    double best_lambda = 1.0;

    while (best_energy > energy || decreasing) {

      lambda *= 0.5;
      double neg_lambda = 1.0 - lambda;

      for (uint k=0; k < new_param.size(); k++)
        hyp_param[k] = neg_lambda * sentence_start_parameters_[k]
          + lambda * new_param[k];

      double hyp_energy = start_prob_m_step_energy(start_count,hyp_param);

      if (hyp_energy < best_energy) {

        decreasing = true;
        best_lambda = lambda;
        best_energy = hyp_energy;
      }
      else
        decreasing = false;
    }

    double neg_best_lambda = 1.0 - best_lambda;
    
    for (uint k=0; k < new_param.size(); k++)
      sentence_start_parameters_[k] = neg_best_lambda * sentence_start_parameters_[k]
        + best_lambda * new_param[k];
    
    energy = best_energy;

    if ((iter % 5) == 0)
      std::cerr << "energy: " << energy << std::endl;
  }
}


void IBM4Trainer::init_from_ibm3(IBM3Trainer& ibm3trainer, bool clear_ibm3, 
				 bool collect_counts, bool viterbi) {

  std::cerr << "******** initializing IBM-4 from IBM-3 *******" << std::endl;

  fertility_prob_.resize(ibm3trainer.fertility_prob().size());
  for (uint k=0; k < fertility_prob_.size(); k++) {
    fertility_prob_[k] = ibm3trainer.fertility_prob()[k];

    //EXPERIMENTAL
    for (uint l=0; l < fertility_prob_[k].size(); l++) {
      if (l <= fertility_limit_)
	fertility_prob_[k][l] = 0.95 * fertility_prob_[k][l] 
	  + 0.05 / std::min<uint>(fertility_prob_[k].size(),fertility_limit_);
      else
	fertility_prob_[k][l] = 0.95 * fertility_prob_[k][l];
    }
    //END_EXPERIMENTAL
  }


  for (size_t s=0; s < source_sentence_.size(); s++) 
    best_known_alignment_[s] = ibm3trainer.best_alignments()[s];

  if (!fix_p0_) {
    p_zero_ = ibm3trainer.p_zero();
    p_nonzero_ = 1.0 - p_zero_;
  }  

  if (collect_counts) {

    cept_start_prob_.set_constant(0.0);
    within_cept_prob_.set_constant(0.0);
    sentence_start_parameters_.set_constant(0.0);

    if (viterbi) {
      train_viterbi(1,&ibm3trainer);
    }
    else {
      train_unconstrained(1,&ibm3trainer);
    }

    if (clear_ibm3)
      ibm3trainer.release_memory();
  }
  else {

    if (clear_ibm3)
      ibm3trainer.release_memory();
    
    //init distortion models from best known alignments
    cept_start_prob_.set_constant(0.0);
    within_cept_prob_.set_constant(0.0);
    sentence_start_parameters_.set_constant(0.0);
 
    for (size_t s=0; s < source_sentence_.size(); s++) {

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];

      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      
      NamedStorage1D<std::set<int> > aligned_source_words(curI+1,MAKENAME(aligned_source_words));
    
      for (uint j=0; j < curJ; j++) {
	const uint aj = best_known_alignment_[s][j];
	aligned_source_words[aj].insert(j);
      }
      
      int prev_center = -100;
    
      for (uint i=1; i <= curI; i++) {

        const uint tclass = target_class_[cur_target[i-1]];
      
	if (!aligned_source_words[i].empty()) {
	  
	  double sum_j = 0;
	  uint nAlignedWords = 0;
	  
	  std::set<int>::iterator ait = aligned_source_words[i].begin();
	  const int first_j = *ait;
	  sum_j += first_j;
	  nAlignedWords++;
	  
	  //collect counts for the head model
	  if (prev_center >= 0) {
            const uint sclass = source_class_[cur_source[prev_center]];
	    int diff =  first_j - prev_center;
	    diff += displacement_offset_;
	    cept_start_prob_(sclass,tclass,diff) += 1.0;
	  }
	  else if (use_sentence_start_prob_)
	    sentence_start_parameters_[first_j] += 1.0;
	  
	  //collect counts for the within-cept model
	  int prev_j = first_j;
	  for (++ait; ait != aligned_source_words[i].end(); ait++) {
	    
	    const int cur_j = *ait;
	    sum_j += cur_j;
	    nAlignedWords++;
	    
	    int diff = cur_j - prev_j;
	    within_cept_prob_(tclass,diff) += 1.0;
	    
	    prev_j = cur_j;
	  }
	  
	  //update prev_center
	  switch (cept_start_mode_) {
	  case IBM4CENTER:
	    prev_center = (int) round(sum_j / nAlignedWords);
	    break;
	  case IBM4FIRST:
	    prev_center = first_j;
	    break;
	  case IBM4LAST:
	    prev_center = prev_j;
	    break;
	  case IBM4UNIFORM:
	    prev_center = (int) round(sum_j / nAlignedWords);
	    break;	  
	  }
	}
      }
    }

    //now that all counts are collected, initialize the distributions
  
    //a) cept start
    for (uint x=0; x < cept_start_prob_.xDim(); x++) {
      for (uint y=0; y < cept_start_prob_.yDim(); y++) {
	
	double sum = 0.0;
	for (uint d=0; d < cept_start_prob_.zDim(); d++)
	  sum += cept_start_prob_(x,y,d);
	
        if (sum > 1e-300) {
          const double count_factor = 0.9 / sum;
          const double uniform_share = 0.1 / cept_start_prob_.zDim();
	
          for (uint d=0; d < cept_start_prob_.zDim(); d++)
            cept_start_prob_(x,y,d) = count_factor * cept_start_prob_(x,y,d) + uniform_share;
        }
        else {
          //this combination did not occur in the viterbi alignments
          //but it may still be possible in the data
          for (uint d=0; d < cept_start_prob_.zDim(); d++)
            cept_start_prob_(x,y,d) = 1.0 / cept_start_prob_.zDim();
        }
      }
    }

    par2nonpar_inter_distortion();

    //b) within-cept
    for (uint x=0; x < within_cept_prob_.xDim(); x++) {
    
      double sum = 0.0;
      for (uint d=0; d < within_cept_prob_.yDim(); d++)
	sum += within_cept_prob_(x,d);
     
      if (sum > 1e-300) {
        const double count_factor = 0.9 / sum;
        const double uniform_share = 0.1 / ( within_cept_prob_.yDim()-1 );
        
        for (uint d=0; d < within_cept_prob_.yDim(); d++) {
          if (d == 0) {
            //zero-displacements are impossible within cepts
            within_cept_prob_(x,d) = 0.0;
          }
          else 
            within_cept_prob_(x,d) = count_factor * within_cept_prob_(x,d) + uniform_share;
        }
      }
      else {
        for (uint d=0; d < within_cept_prob_.yDim(); d++) {
          if (d == 0) {
            //zero-displacements are impossible within cepts
            within_cept_prob_(x,d) = 0.0;
          }
          else 
            within_cept_prob_(x,d) = 1.0 / (within_cept_prob_.yDim()-1);
        }
      }
    }

    par2nonpar_intra_distortion();

    //c) sentence start prob
    if (use_sentence_start_prob_) {
      sentence_start_parameters_ *= 1.0 / sentence_start_parameters_.sum();

      //TEMP
      for (uint J=1; J <= maxJ_; J++) {
	if (sentence_start_prob_[J].size() > 0) {
	  for (uint j=0; j < J; j++)
	    sentence_start_prob_[J][j] = sentence_start_parameters_[j];
	}
      }
      //END_TEMP

      par2nonpar_start_prob();
    }
  }

  //DEBUG
#ifndef NDEBUG
  std::cerr << "checking" << std::endl;

  for (size_t s=0; s < source_sentence_.size(); s++) {

    long double align_prob = alignment_prob(s,best_known_alignment_[s]);

    if (isinf(align_prob) || isnan(align_prob) || align_prob == 0.0) {

      std::cerr << "ERROR: initial align-prob for sentence " << s << " has prob " << align_prob << std::endl;
      exit(1);
    }
  }
#endif
  //END_DEBUG
}

void IBM4Trainer::update_alignments_unconstrained() {

  Math2D::NamedMatrix<long double> expansion_prob(MAKENAME(expansion_prob));
  Math2D::NamedMatrix<long double> swap_prob(MAKENAME(swap_prob));

  for (size_t s=0; s < source_sentence_.size(); s++) {


    if (nSourceClasses_*nTargetClasses_ >= 10 && (s%25) == 0) {
      for (uint J=11; J < inter_distortion_prob_.size(); J++) {
        
        for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
          for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
            inter_distortion_prob_[J](x,y).resize(0,0);
      }
    }

    const uint curI = target_sentence_[s].size();
    Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));
    
    uint nIter=0;
    update_alignment_by_hillclimbing(source_sentence_[s],target_sentence_[s],slookup_[s],
				     nIter,fertility,expansion_prob,swap_prob,best_known_alignment_[s]);
  }
}


long double IBM4Trainer::alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) {

  return alignment_prob(source_sentence_[s],target_sentence_[s],slookup_[s],alignment);
}

long double IBM4Trainer::alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                        const Math2D::Matrix<uint>& lookup,const Math1D::Vector<AlignBaseType>& alignment) {

  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));
  NamedStorage1D<std::set<int> > aligned_source_words(curI+1,MAKENAME(aligned_source_words));

  const Storage2D<Math2D::Matrix<float> >& cur_inter_distortion_prob =  inter_distortion_prob_[curJ];
  const Math3D::Tensor<float>& cur_intra_distortion_prob =  intra_distortion_prob_[curJ];

  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];
  
  for (uint j=0; j < curJ; j++) {
    const uint aj = alignment[j];
    aligned_source_words[aj].insert(j);
    fertility[aj]++;
    
    if (aj == 0) {
      prob *= dict_[0][source[j]-1];
      //DEBUG
      if (isnan(prob))
        std::cerr << "prob nan after empty word dict prob" << std::endl;
      //END_DEBUG
    }
  }
  
  if (curJ < 2*fertility[0])
    return 0.0;

  for (uint i=1; i <= curI; i++) {
    uint t_idx = target[i-1];
    prob *= fertility_prob_[t_idx][fertility[i]];
    if (!no_factorial_)
      prob *= ldfac(fertility[i]);
  }

  //DEBUG
  if (isnan(prob))
    std::cerr << "prob nan after fertility probs" << std::endl;
  //END_DEBUG


  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;

  for (uint i=1; i <= curI; i++) {

    //NOTE: a dependence on word classes is currently not implemented
    
    if (fertility[i] > 0) {
      const uint ti = target[i-1];
      const uint tclass = target_class_[ti];

      const int first_j = *aligned_source_words[i].begin();

      //handle the head of the cept
      if (prev_cept_center != -1) {

        const int first_j = *aligned_source_words[i].begin();
        prob *= dict_[ti][lookup(first_j,i-1)];
        //DEBUG
        if (isnan(prob))
          std::cerr << "prob nan after dict-prob, pc != -1, i=" << i << std::endl;
        //END_DEBUG


        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[prev_cept_center]];

          if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
            par2nonpar_inter_distortion(curJ,sclass,tclass);

          prob *= cur_inter_distortion_prob(sclass,tclass)(first_j,prev_cept_center); 

          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after inter-distort prob, i=" << i << std::endl;
          //END_DEBUG
        }
        else
          prob /= curJ;
      }
      else {
	prob *= dict_[ti][lookup(first_j,i-1)];

        if (use_sentence_start_prob_) {
          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after dict-prob, pc == -1, i=" << i << std::endl;
          //END_DEBUG

          prob *= cur_sentence_start_prob[first_j];

          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after sent start prob, pc == -1, i=" << i << std::endl;
          //END_DEBUG
        }
        else {

          //DEBUG
          if (isnan(prob))
            std::cerr << "prob nan after dict-prob, pc == -1, i=" << i << std::endl;
          //END_DEBUG

          prob *= 1.0 / curJ;
        }
      }

      //handle the body of the cept
      int prev_j = first_j;
      std::set<int>::iterator ait = aligned_source_words[i].begin();
      for (++ait; ait != aligned_source_words[i].end(); ait++) {

        const int cur_j = *ait;
        prob *= dict_[ti][lookup(cur_j,i-1)] * cur_intra_distortion_prob(tclass,cur_j,prev_j);
	
        //DEBUG
        if (isnan(prob))
          std::cerr << "prob nan after combined body-prob, i=" << i << std::endl;
        //END_DEBUG

        prev_j = cur_j;
      }

      //compute the center of this cept and store the result in prev_cept_center
      double sum = 0.0;
      for (std::set<int>::iterator ait = aligned_source_words[i].begin(); ait != aligned_source_words[i].end(); ait++) {
        sum += *ait;
      }

      switch (cept_start_mode_) {
      case IBM4CENTER :
        prev_cept_center = (int) round(sum / fertility[i]);
        break;
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST:
        prev_cept_center = prev_j;
        break;
      case IBM4UNIFORM:
        prev_cept_center = (int) round(sum / fertility[i]);
        break;
      default:
        assert(false);
      }

      assert(prev_cept_center >= 0);
    }
  }

  //handle empty word
  assert(fertility[0] <= 2*curJ);

  //dictionary probs were handled above
  
  prob *= ldchoose(curJ-fertility[0],fertility[0]);
  for (uint k=1; k <= fertility[0]; k++)
    prob *= p_zero_;
  for (uint k=1; k <= curJ-2*fertility[0]; k++)
    prob *= p_nonzero_;

  if (och_ney_empty_word_) {

    for (uint k=1; k<= fertility[0]; k++)
      prob *= ((long double) k) / curJ;
  }

  return prob;
}


long double IBM4Trainer::distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
					 const Math1D::Vector<AlignBaseType>& alignment) {

  const uint curI = target.size();
  const uint curJ = source.size();

  assert(alignment.size() == curJ);

  
  NamedStorage1D<std::vector<AlignBaseType> > aligned_source_words(curI+1,MAKENAME(aligned_source_words));

  for (uint j=0; j < curJ; j++) {
    const uint aj = alignment[j];
    aligned_source_words[aj].push_back(j);
  }

  return distortion_prob(source,target,aligned_source_words);
}


//NOTE: the vectors need to be sorted
long double IBM4Trainer::distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
					 const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) {

  long double prob = 1.0;

  const uint curI = target.size();
  const uint curJ = source.size();

  const Storage2D<Math2D::Matrix<float> >& cur_inter_distortion_prob =  inter_distortion_prob_[curJ];
  const Math3D::Tensor<float>& cur_intra_distortion_prob =  intra_distortion_prob_[curJ];

  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  
  if (curJ < 2*aligned_source_words[0].size())
    return 0.0;

  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;

  for (uint i=1; i <= curI; i++) {

    if (aligned_source_words[i].size() > 0) {

      const uint ti = target[i-1];
      const uint tclass = target_class_[ti];

      const int first_j = aligned_source_words[i][0];

      //handle the head of the cept
      if (prev_cept_center != -1) {

        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[prev_cept_center]];

          if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
            par2nonpar_inter_distortion(curJ,sclass,tclass);

          prob *= cur_inter_distortion_prob(sclass,tclass)(first_j,prev_cept_center); 
        }
        else
          prob /= curJ;
      }
      else {
        if (use_sentence_start_prob_) {
          prob *= cur_sentence_start_prob[first_j];
        }
        else {
          prob *= 1.0 / curJ;
        }
      }

      //handle the body of the cept
      int prev_j = first_j;
      for (uint k=1; k < aligned_source_words[i].size(); k++) {

        const int cur_j = aligned_source_words[i][k];
        prob *= cur_intra_distortion_prob(tclass,cur_j,prev_j);
	
        prev_j = cur_j;
      }

      switch (cept_start_mode_) {
      case IBM4CENTER : {

	//compute the center of this cept and store the result in prev_cept_center
	double sum = 0.0;
	for (uint k=0; k < aligned_source_words[i].size(); k++) {
	  sum += aligned_source_words[i][k];
	}

        prev_cept_center = (int) round(sum / aligned_source_words[i].size());
        break;
      }
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST:
        prev_cept_center = prev_j;
        break;
      case IBM4UNIFORM:
        prev_cept_center = first_j;
        break;
      default:
        assert(false);
      }

      assert(prev_cept_center >= 0);
    }
  }


  return prob;
}

void IBM4Trainer::print_alignment_prob_factors(const Storage1D<uint>& source, const Storage1D<uint>& target, 
					       const Math2D::Matrix<uint>& lookup, const Math1D::Vector<AlignBaseType>& alignment) {


  long double prob = 1.0;

  const Storage1D<uint>& cur_source = source;
  const Storage1D<uint>& cur_target = target;
  const Math2D::Matrix<uint>& cur_lookup = lookup;

  const uint curI = cur_target.size();
  const uint curJ = cur_source.size();

  assert(alignment.size() == curJ);

  Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));
  NamedStorage1D<std::set<int> > aligned_source_words(curI+1,MAKENAME(aligned_source_words));

  const Storage2D<Math2D::Matrix<float> >& cur_inter_distortion_prob =  inter_distortion_prob_[curJ];
  const Math3D::Tensor<float>& cur_intra_distortion_prob =  intra_distortion_prob_[curJ];

  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];
  
  for (uint j=0; j < curJ; j++) {
    const uint aj = alignment[j];
    aligned_source_words[aj].insert(j);
    fertility[aj]++;
    
    if (aj == 0) {
      prob *= dict_[0][cur_source[j]-1];

      std::cerr << "mult by dict-prob for empty word, factor: " << dict_[0][cur_source[j]-1] 
		<< ", result: " << prob << std::endl;
    }
  }
  
  if (curJ < 2*fertility[0]) {

    std::cerr << "ERROR: too many zero-aligned words, returning 0.0" << std::endl;
    return;
  }

  for (uint i=1; i <= curI; i++) {
    uint t_idx = cur_target[i-1];
    prob *= fertility_prob_[t_idx][fertility[i]];

    std::cerr << "mult by fert-prob " << fertility_prob_[t_idx][fertility[i]] 
	      << ", result: " << prob << std::endl;

    if (!no_factorial_) {
      prob *= ldfac(fertility[i]);

      std::cerr << "mult by factorial " << ldfac(fertility[i]) 
		<< ", result: " << prob << std::endl;
    }
  }


  //handle cepts with one or more aligned source words
  int prev_cept_center = -1;

  for (uint i=1; i <= curI; i++) {

    //NOTE: a dependence on word classes is currently not implemented
    
    if (fertility[i] > 0) {
      const uint ti = cur_target[i-1];
      const uint tclass = target_class_[ti];

      const int first_j = *aligned_source_words[i].begin();

      //handle the head of the cept
      if (prev_cept_center != -1) {

        const int first_j = *aligned_source_words[i].begin();
        prob *= dict_[ti][cur_lookup(first_j,i-1)];

	std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(first_j,i-1)] 
		  << ", result: " << prob << std::endl;


        if (cept_start_mode_ != IBM4UNIFORM) {

          const uint sclass = source_class_[source[prev_cept_center]];

          if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
            par2nonpar_inter_distortion(curJ,sclass,tclass);

          prob *= cur_inter_distortion_prob(sclass,tclass)(first_j,prev_cept_center); 

	  std::cerr << "mult by distortion-prob " << cur_inter_distortion_prob(sclass,tclass)(first_j,prev_cept_center)
		    << ", result: " << prob << std::endl;
        }
        else {
          prob /= curJ;

	  std::cerr << "div by " << curJ << ", result: " << prob << std::endl;
	}
      }
      else {
        if (use_sentence_start_prob_) {
          prob *= dict_[ti][cur_lookup(first_j,i-1)];

	  std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(first_j,i-1)]
		    << ", result: " << prob << std::endl;

          prob *= cur_sentence_start_prob[first_j];

	  std::cerr << "mult by start prob " << cur_sentence_start_prob[first_j] 
		    << ", result: " << prob << std::endl;
        }
        else {
          prob *= dict_[ti][cur_lookup(first_j,i-1)];

	  std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(first_j,i-1)]
		    << ", result: " << prob << std::endl;

          prob *= 1.0 / curJ;

	  std::cerr << "div by " << curJ << ", result: " << prob << std::endl;
        }
      }

      //handle the body of the cept
      int prev_j = first_j;
      std::set<int>::iterator ait = aligned_source_words[i].begin();
      for (++ait; ait != aligned_source_words[i].end(); ait++) {

        const int cur_j = *ait;
        prob *= dict_[ti][cur_lookup(cur_j,i-1)] * cur_intra_distortion_prob(tclass,cur_j,prev_j);
	
	std::cerr << "mult by dict-prob " << dict_[ti][cur_lookup(cur_j,i-1)] << " and distortion-prob "
		  << cur_intra_distortion_prob(tclass,cur_j,prev_j) << ", result: " << prob 
		  << ", target index: " << ti << ", source index: " << cur_lookup(cur_j,i-1) << std::endl;

        prev_j = cur_j;
      }

      //compute the center of this cept and store the result in prev_cept_center
      double sum = 0.0;
      for (std::set<int>::iterator ait = aligned_source_words[i].begin(); ait != aligned_source_words[i].end(); ait++) {
        sum += *ait;
      }

      switch (cept_start_mode_) {
      case IBM4CENTER :
        prev_cept_center = (int) round(sum / fertility[i]);
        break;
      case IBM4FIRST:
        prev_cept_center = first_j;
        break;
      case IBM4LAST:
        prev_cept_center = prev_j;
        break;
      case IBM4UNIFORM:
        prev_cept_center = (int) round(sum / fertility[i]);
        break;
      default:
        assert(false);
      }

      assert(prev_cept_center >= 0);
    }
  }

  //handle empty word
  assert(fertility[0] <= 2*curJ);

  //dictionary probs were handled above
  
  prob *= ldchoose(curJ-fertility[0],fertility[0]);

  std::cerr << "mult by ldchoose " << ldchoose(curJ-fertility[0],fertility[0]) << ", result: " << prob << std::endl;

  for (uint k=1; k <= fertility[0]; k++) {
    prob *= p_zero_;

    std::cerr << "mult by p0 " << p_zero_ << ", result: " << prob << std::endl;
  }
  for (uint k=1; k <= curJ-2*fertility[0]; k++) {
    prob *= p_nonzero_;

    std::cerr << "mult by p1 " << p_nonzero_ << ", result: " << prob << std::endl;
  }

  if (och_ney_empty_word_) {

    for (uint k=1; k<= fertility[0]; k++) {
      prob *= ((long double) k) / curJ;

      std::cerr << "mult by k/curJ = " << (((long double) k) / curJ) << ", result: " << prob << std::endl;
    }
  }
}

long double IBM4Trainer::update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                                          const Math2D::Matrix<uint>& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
                                                          Math2D::Matrix<long double>& expansion_prob,
                                                          Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment) {

  const double improvement_factor = 1.001;

  const uint curI = target.size();
  const uint curJ = source.size(); 


  fertility.resize(curI+1);

  long double base_prob = alignment_prob(source,target,lookup,alignment);

  swap_prob.resize(curJ,curJ);
  expansion_prob.resize(curJ,curI+1);
  swap_prob.set_constant(0.0);
  expansion_prob.set_constant(0.0);


  uint count_iter = 0;

  const Storage2D<Math2D::Matrix<float> >& cur_inter_distortion_prob =  inter_distortion_prob_[curJ];
  const Math3D::Tensor<float>& cur_intra_distortion_prob =  intra_distortion_prob_[curJ];
  const Math1D::Vector<double>& cur_sentence_start_prob = sentence_start_prob_[curJ];

  //source words are listed in ascending order
  NamedStorage1D< std::vector<AlignBaseType> > aligned_source_words(curI+1,MAKENAME(aligned_source_words));
  
  fertility.set_constant(0);
  for (uint j=0; j < curJ; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    aligned_source_words[aj].push_back(j);
  }

  long double base_distortion_prob = distortion_prob(source,target,aligned_source_words);


  while (true) {    

    Math1D::NamedVector<uint> prev_cept(curI+1,MAX_UINT,MAKENAME(prev_cept));
    Math1D::NamedVector<uint> next_cept(curI+1,MAX_UINT,MAKENAME(next_cept));
    Math1D::NamedVector<uint> cept_center(curI+1,MAX_UINT,MAKENAME(cept_center));

    uint prev_i = MAX_UINT;
    for (uint i=1; i <= curI; i++) {

      if (fertility[i] > 0) {
	
        prev_cept[i] = prev_i;
        if (prev_i != MAX_UINT)
          next_cept[prev_i] = i;

        switch (cept_start_mode_) {
        case IBM4CENTER: {
          double sum_j = 0.0;
          for (uint k=0; k < aligned_source_words[i].size(); k++)
            sum_j += aligned_source_words[i][k];
          cept_center[i] = (int) round(sum_j / aligned_source_words[i].size());
          break;
        }
        case IBM4FIRST:
          cept_center[i] = aligned_source_words[i][0];
          break;
        case IBM4LAST:
          cept_center[i] = aligned_source_words[i][aligned_source_words[i].size()-1];
          break;
        case IBM4UNIFORM:
          break;
        default:
          assert(false);
        }

        prev_i = i;
      }
    }

    count_iter++;
    nIter++;

    if (count_iter > 50)
      break;

    //std::cerr << "****************** starting new hillclimb iteration, current best prob: " << base_prob << std::endl;

    bool improvement = false;

    long double best_prob = base_prob;
    bool best_change_is_move = false;
    uint best_move_j = MAX_UINT;
    uint best_move_aj = MAX_UINT;
    uint best_swap_j1 = MAX_UINT;
    uint best_swap_j2 = MAX_UINT;

    /**** scan neighboring alignments and keep track of the best one that is better 
     ****  than the current alignment  ****/

    //std::clock_t tStartExp,tEndExp;
    //tStartExp = std::clock();

    //a) expansion moves

    NamedStorage1D< std::vector<AlignBaseType> > hyp_aligned_source_words(MAKENAME(hyp_aligned_source_words));
    hyp_aligned_source_words = aligned_source_words;

    for (uint j=0; j < curJ; j++) {

      const uint aj = alignment[j];
      assert(fertility[aj] > 0);
      expansion_prob(j,aj) = 0.0;

      const uint s_idx = source[j];

      const double old_dict_prob = (aj == 0) ? dict_[0][s_idx-1] : dict_[target[aj-1]][lookup(j,aj-1)];

      //for now, to be sure:
      hyp_aligned_source_words[aj] = aligned_source_words[aj];

      const uint prev_i = prev_cept[aj];
      const uint next_i = next_cept[aj];

      for (uint cand_aj = 0; cand_aj <= curI; cand_aj++) {
      
        if (cand_aj != aj) {

          long double hyp_prob = 0.0;

          bool incremental_calculation = false;

          const double new_dict_prob = (cand_aj == 0) ? dict_[0][s_idx-1] : dict_[target[cand_aj-1]][lookup(j,cand_aj-1)];

	  //EXPERIMENTAL (prune constellations with very unlikely translation probs.)
          if (new_dict_prob < 1e-10) {
            expansion_prob(j,cand_aj) = 0.0;
            continue;
          }
	  if (cand_aj != 0 && (fertility[cand_aj]+1) > fertility_limit_) {

            expansion_prob(j,cand_aj) = 0.0;
            continue;
	  }
	  else if (curJ < 2*fertility[0]+2) {

            expansion_prob(j,cand_aj) = 0.0;
            continue;
	  }
	  //END_EXPERIMENTAL

          long double incoming_prob = new_dict_prob; 
          long double leaving_prob = old_dict_prob; 

          if (aj != 0 && cand_aj != 0) {

            if (next_i != MAX_UINT &&
                (((prev_i != MAX_UINT && cand_aj < prev_i) || (prev_i == MAX_UINT && cand_aj > next_i)) 
                 || ((next_i != MAX_UINT && cand_aj > next_i) 
                     || (next_i == MAX_UINT && prev_i != MAX_UINT && cand_aj < prev_i)   )  ) ) {

              incremental_calculation = true;

              const uint prev_ti = target[aj-1];
              const uint new_ti = target[cand_aj-1];

              incoming_prob *= fertility_prob_[new_ti][fertility[cand_aj]+1];
              incoming_prob *= fertility_prob_[prev_ti][fertility[aj]-1];

              if (!no_factorial_) {
                incoming_prob *= ldfac(fertility[cand_aj]+1);
                incoming_prob *= ldfac(fertility[aj]-1);
              }

              leaving_prob *= fertility_prob_[new_ti][fertility[cand_aj]];
              leaving_prob *= fertility_prob_[prev_ti][fertility[aj]];

              assert(leaving_prob > 0.0);

              if (!no_factorial_) {
                leaving_prob *= ldfac(fertility[cand_aj]);
                leaving_prob *= ldfac(fertility[aj]);
              }

              const uint prev_aj_fert = fertility[aj];

              /***************************** 1. changes regarding aj ******************************/
              if (prev_aj_fert > 1) {
                //the cept aj remains

                uint jnum;
                for (jnum = 0; jnum < prev_aj_fert; jnum++) {
                  if (aligned_source_words[aj][jnum] == j)
                    break;
                }

                assert (jnum < aligned_source_words[aj].size());

                //calculate new center of aj
                uint new_aj_center = MAX_UINT;
                switch (cept_start_mode_) {
                case IBM4CENTER : {
                  double sum_j = 0.0;
                  for (uint k=0; k < prev_aj_fert; k++) {
                    if (k != jnum)
                      sum_j += aligned_source_words[aj][k];
                  }
                  new_aj_center = (int) round(sum_j / (aligned_source_words[aj].size()-1));
                  break;
                }
                case IBM4FIRST : {
                  if (jnum == 0)
                    new_aj_center = aligned_source_words[aj][1];
                  else {
                    new_aj_center = aligned_source_words[aj][0];
                    assert(new_aj_center == cept_center[aj]);
                  }
                  break;
                }
                case IBM4LAST : {
                  if (jnum+1 == prev_aj_fert)
                    new_aj_center = aligned_source_words[aj][prev_aj_fert-2];
                  else {
                    new_aj_center = aligned_source_words[aj][prev_aj_fert-1];
                    assert(new_aj_center == cept_center[aj]);
                  }
                  break;
                }
                case IBM4UNIFORM :
                  break;
                default:
                  assert(false);
                }
		
                //re-calculate the transition aj -> next_i
                if (next_i != MAX_UINT && new_aj_center != cept_center[aj]) { 
                  const uint old_sclass = source_class_[source[cept_center[aj]]];
                  const uint tclass = target_class_[target[next_i-1]];

                  if (cur_inter_distortion_prob(old_sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,old_sclass,tclass);

                  leaving_prob *= cur_inter_distortion_prob(old_sclass,tclass)(aligned_source_words[next_i][0],cept_center[aj]);

                  assert(leaving_prob > 0.0);

                  const uint new_sclass = source_class_[source[new_aj_center]];

                  if (cur_inter_distortion_prob(new_sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,new_sclass,tclass);

                  incoming_prob *= cur_inter_distortion_prob(new_sclass,tclass)(aligned_source_words[next_i][0],new_aj_center);
                }

                if (jnum == 0) {
                  //the transition prev_i -> aj is affected

                  const uint tclass = target_class_[prev_ti];
		  
                  if (prev_i != MAX_UINT) {
                    const uint sclass = source_class_[source[cept_center[prev_i]]];

                    if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,sclass,tclass);

                    leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(j,cept_center[prev_i]);
                    incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(aligned_source_words[aj][1],cept_center[prev_i]);
                  }
                  else if (use_sentence_start_prob_) {
                    leaving_prob *= cur_sentence_start_prob[j];
                    incoming_prob *= cur_sentence_start_prob[aligned_source_words[aj][1]];
                  }

                  assert(leaving_prob > 0.0);

                  leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][1],j);

                  assert(leaving_prob > 0.0);
                }
                else {
                  const uint tclass = target_class_[prev_ti];
		  
                  leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][jnum],aligned_source_words[aj][jnum-1]);

                  assert(leaving_prob > 0.0);

                  if (jnum+1 < prev_aj_fert) {
                    leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][jnum+1], aligned_source_words[aj][jnum]);

                    assert(leaving_prob > 0.0);

                    incoming_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][jnum+1],aligned_source_words[aj][jnum-1]);
                  }
                }

              }
              else {
                //the cept aj vanishes

                //erase the transitions prev_i -> aj    and    aj -> next_i
                if (prev_i != MAX_UINT) {
                  const uint sclass = source_class_[source[cept_center[prev_i]]];
                  const uint tclass = target_class_[prev_ti];

                  if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,sclass,tclass);

                  leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(j,cept_center[prev_i]);

                  assert(leaving_prob > 0.0);
                }		
                else if (use_sentence_start_prob_) {
                  leaving_prob *= cur_sentence_start_prob[j];

                  assert(leaving_prob > 0.0);
                }

                if (next_i != MAX_UINT) {
                  const uint sclass = source_class_[source[j]];
                  const uint tclass = target_class_[target[next_i-1]];

                  if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,sclass,tclass);

                  leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(aligned_source_words[next_i][0],j);

                  assert(leaving_prob > 0.0);
                }
                
                //introduce the transition prev_i -> next_i
                if (prev_i != MAX_UINT) {
                  if (next_i != MAX_UINT) {
                    const uint sclass = source_class_[source[cept_center[prev_i]]];
                    const uint tclass = target_class_[target[next_i-1]];

                    if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,sclass,tclass);

                    incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(aligned_source_words[next_i][0],cept_center[prev_i]);
                  }
                }
                else if (use_sentence_start_prob_)
                  incoming_prob *= cur_sentence_start_prob[aligned_source_words[next_i][0]];
              }
	      
              /********************** 2. changes regarding cand_aj **********************/
              uint cand_prev_i = MAX_UINT;
              for (uint k=cand_aj-1; k > 0; k--) {
                if (fertility[k] > 0) {
                  cand_prev_i = k;
                  break;
                }
              }
              uint cand_next_i = MAX_UINT;
              for (uint k=cand_aj+1; k <= curI; k++) {
                if (fertility[k] > 0) {
                  cand_next_i = k;
                  break;
                }
              }

              if (fertility[cand_aj] > 0) {
                //the cept cand_aj was already there

                uint insert_pos = 0;
                for (; insert_pos < fertility[cand_aj] 
                       && aligned_source_words[cand_aj][insert_pos] < j; insert_pos++) {
                  //empty body
                }

                const uint tclass = target_class_[new_ti];

                if (insert_pos == 0) {

                  if (cand_prev_i == MAX_UINT) {

                    if (use_sentence_start_prob_) {
                      leaving_prob *= cur_sentence_start_prob[aligned_source_words[cand_aj][0]];
                      incoming_prob *= cur_sentence_start_prob[j];

                      assert(leaving_prob > 0.0);

                      incoming_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[cand_aj][0],j);
                    }
                  }
                  else {
                    const uint sclass = source_class_[source[cept_center[cand_prev_i]]];

                    if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,sclass,tclass);

                    leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(aligned_source_words[cand_aj][0],cept_center[cand_prev_i]);

                    assert(leaving_prob > 0.0);

                    incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(j,cept_center[cand_prev_i]);
                    incoming_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[cand_aj][0],j);
                  }
                }
                else if (insert_pos < fertility[cand_aj]) {
		  
                  leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[cand_aj][insert_pos],
                                                            aligned_source_words[cand_aj][insert_pos-1]);

                  assert(leaving_prob > 0.0);

                  incoming_prob *= cur_intra_distortion_prob(tclass,j,aligned_source_words[cand_aj][insert_pos-1]);
                  incoming_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[cand_aj][insert_pos],j);
                }
                else {
                  //insert at the end
                  assert(insert_pos == fertility[cand_aj]);

                  incoming_prob *= cur_intra_distortion_prob(tclass,j,aligned_source_words[cand_aj][insert_pos-1]);
                }

                if (cand_next_i != MAX_UINT) {
                  //calculate new center of cand_aj

                  uint new_cand_aj_center = MAX_UINT;
                  switch (cept_start_mode_) {
                  case IBM4CENTER : {
                    double sum_j = j;
                    for (uint k=0; k < fertility[cand_aj]; k++)
                      sum_j += aligned_source_words[cand_aj][k];

                    new_cand_aj_center = (int) round(sum_j / (fertility[cand_aj]+1) );
                    break;
                  }
                  case IBM4FIRST : {
                    if (insert_pos == 0)
                      new_cand_aj_center = j;
                    else 
                      new_cand_aj_center = cept_center[cand_aj];
                    break;
                  }
                  case IBM4LAST : {
                    if (insert_pos > fertility[cand_aj])
                      new_cand_aj_center = j;
                    else
                      new_cand_aj_center = cept_center[cand_aj];
                    break;
                  }
                  case IBM4UNIFORM:
                    break;
                  default:
                    assert(false);
                  } //end of switch-statement
		  
                  if (new_cand_aj_center != cept_center[cand_aj] && cept_center[cand_aj] != new_cand_aj_center) {
                    const uint old_sclass = source_class_[source[cept_center[cand_aj]]];
                    const uint tclass = target_class_[target[cand_next_i-1]];

                    if (cur_inter_distortion_prob(old_sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,old_sclass,tclass);

                    leaving_prob *= cur_inter_distortion_prob(old_sclass,tclass)(aligned_source_words[cand_next_i][0],cept_center[cand_aj]);
		    
                    assert(leaving_prob > 0.0);

                    const uint new_sclass = source_class_[source[new_cand_aj_center]];

                    if (cur_inter_distortion_prob(new_sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,new_sclass,tclass);

                    incoming_prob *= cur_inter_distortion_prob(new_sclass,tclass)(aligned_source_words[cand_next_i][0],new_cand_aj_center);
                  }
                }
              }
              else {
                //the cept cand_aj is newly created

                //erase the transition cand_prev_i -> cand_next_i (if existent)
                if (cand_prev_i != MAX_UINT && cand_next_i != MAX_UINT) {

                  const uint sclass = source_class_[source[cept_center[cand_prev_i]]];
                  const uint tclass = target_class_[target[cand_next_i-1]];

                  if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,sclass,tclass);

                  leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(aligned_source_words[cand_next_i][0],cept_center[cand_prev_i]);

                  assert(leaving_prob > 0.0);
                }
                else if (cand_prev_i == MAX_UINT) {
		  
                  assert(cand_next_i != MAX_UINT);
                  if (use_sentence_start_prob_)
                    leaving_prob *= cur_sentence_start_prob[aligned_source_words[cand_next_i][0]];

                  assert(leaving_prob > 0.0);
                }
                else {
                  //nothing to do here
                }

                //introduce the transitions cand_prev_i -> cand_aj    and   cand_aj -> cand_next_i
                if (cand_prev_i != MAX_UINT) {
                  const uint sclass = source_class_[source[cept_center[cand_prev_i]]];
                  const uint tclass = target_class_[new_ti];

                  if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,sclass,tclass);

                  incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(j,cept_center[cand_prev_i]);
                }
                else
                  incoming_prob *= cur_sentence_start_prob[j];

                if (cand_next_i != MAX_UINT) {
                  const uint sclass = source_class_[source[j]];
                  const uint tclass = target_class_[target[cand_next_i-1]];

                  if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,sclass,tclass);

                  incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(aligned_source_words[cand_next_i][0],j);
                }
              }

              assert(leaving_prob > 0.0);

              hyp_prob = base_prob * incoming_prob / leaving_prob;

	      if (isnan(hyp_prob)) {

    		std::cerr << "######in exp. move for j=" << j << " cur aj: " << aj << ", candidate: " << cand_aj << std::endl;

		std::cerr << "hyp_prob: " << hyp_prob << std::endl;
		std::cerr << "base: " << base_prob << ", incoming: " << incoming_prob << ", leaving: " << leaving_prob << std::endl;
		std::cerr << "hc iter: " << count_iter << std::endl;
		exit(1);
	      }
	      
#ifndef NDEBUG
              //DEBUG
              Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
              hyp_alignment[j] = cand_aj;
              long double check_prob = alignment_prob(source,target,lookup,hyp_alignment);

              if (check_prob > 0.0) {
		
                long double check_ratio = hyp_prob / check_prob;
		
                if (! (check_ratio > 0.99 && check_ratio < 1.01)) {

                  std::cerr << "****************************************************************" << std::endl;
                  std::cerr << "expansion: moving j=" << j << " to cand_aj=" << cand_aj << " from aj=" << aj
                            << std::endl;
                  std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
                  std::cerr << "base alignment: " << alignment << std::endl;
                  std::cerr << "actual prob: " << check_prob << std::endl;
                  std::cerr << "incremental hyp_prob: " << hyp_prob << std::endl;		  
                  std::cerr << "(base_prob: " << base_prob << ")" << std::endl;
                  std::cerr << "################################################################" << std::endl;
                }
		
                if (check_prob > 1e-15 * base_prob)
                  assert(check_ratio > 0.99 && check_ratio < 1.01);
              }
              //END_DEBUG
#endif
	      
            }
          }
          else if (cand_aj == 0) {
            //NOTE: this time we also handle the cases where next_i == MAX_UINT or where prev_i == MAX_UINT

            incremental_calculation = true;

            assert(aj != 0);

            const uint prev_ti = target[aj-1];
            const uint prev_aj_fert = fertility[aj];
            const uint prev_zero_fert = fertility[0];
            const uint new_zero_fert = prev_zero_fert+1;

            if (curJ < 2*new_zero_fert) {
              hyp_prob = 0.0;
            }
            else {

              incoming_prob *= fertility_prob_[prev_ti][fertility[aj]-1];
              if (!no_factorial_)
                incoming_prob *= ldfac(fertility[aj]-1);

              incoming_prob *= ldchoose(curJ-new_zero_fert,new_zero_fert);

              for (uint k=1; k <= new_zero_fert; k++)
                incoming_prob *= p_zero_;
              for (uint k=1; k <= curJ-2*new_zero_fert; k++)
                incoming_prob *= p_nonzero_;

              if (och_ney_empty_word_) {
                for (uint k=1; k <= new_zero_fert; k++)
                  incoming_prob *= ((long double) k) / curJ;
              }

              leaving_prob *= fertility_prob_[prev_ti][fertility[aj]];
              if (!no_factorial_)
                leaving_prob *= ldfac(fertility[aj]);

              leaving_prob *= ldchoose(curJ-prev_zero_fert,prev_zero_fert);

              for (uint k=1; k <= prev_zero_fert; k++)
                leaving_prob *= p_zero_;
              for (uint k=1; k <= curJ-2*prev_zero_fert; k++)
                leaving_prob *= p_nonzero_;

              if (och_ney_empty_word_) {
                for (uint k=1; k <= prev_zero_fert; k++)
                  leaving_prob *= ((long double) k) / curJ;
              }

              if (prev_aj_fert > 1 ) {
                //the cept aj remains

                uint jnum;
                for (jnum = 0; jnum < prev_aj_fert; jnum++) {
                  if (aligned_source_words[aj][jnum] == j)
                    break;
                }

                assert (jnum < aligned_source_words[aj].size());

                if (next_i != MAX_UINT) {
                  //calculate new center of aj
                  uint new_aj_center = MAX_UINT;
                  switch (cept_start_mode_) {
                  case IBM4CENTER : {
                    double sum_j = 0.0;
                    for (uint k=0; k < prev_aj_fert; k++) {
                      if (k != jnum)
                        sum_j += aligned_source_words[aj][k];
                    }
                    new_aj_center = (uint) round(sum_j / (aligned_source_words[aj].size()-1));
                    break;
                  }
                  case IBM4FIRST : {
                    if (jnum == 0)
                      new_aj_center = aligned_source_words[aj][1];
                    else {
                      new_aj_center = aligned_source_words[aj][0];
                      assert(new_aj_center == cept_center[aj]);
                    }
                    break;
                  }
                  case IBM4LAST : {
                    if (jnum+1 == prev_aj_fert)
                      new_aj_center = aligned_source_words[aj][prev_aj_fert-2];
                    else {
                      new_aj_center = aligned_source_words[aj][prev_aj_fert-1];
                      assert(new_aj_center == cept_center[aj]);
                    }
                    break;
                  }
                  case IBM4UNIFORM :
                    break;
                  default:
                    assert(false);
                  }
		  
                  //re-calculate the transition aj -> next_i
                  if (cept_center[aj] != new_aj_center) {
                    const uint old_sclass = source_class_[source[cept_center[aj]]];
                    const uint tclass = target_class_[target[next_i-1]];
                    const uint new_sclass = source_class_[source[new_aj_center]];
                    
                    if (cur_inter_distortion_prob(old_sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,old_sclass,tclass);

                    leaving_prob *= cur_inter_distortion_prob(old_sclass,tclass)(aligned_source_words[next_i][0],cept_center[aj]);
                    
                    if (cur_inter_distortion_prob(new_sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,new_sclass,tclass);

                    incoming_prob *= cur_inter_distortion_prob(new_sclass,tclass)(aligned_source_words[next_i][0],new_aj_center);
                  }
                }

                if (jnum == 0) {
                  //the transition prev_i -> aj is affected

                  const uint tclass = target_class_[prev_ti];

                  if (prev_i != MAX_UINT) {
                    const uint sclass = source_class_[source[cept_center[prev_i]]];

                    if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,sclass,tclass);

                    leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(j,cept_center[prev_i]);
                  }
                  else if (use_sentence_start_prob_) {
                    leaving_prob *= sentence_start_prob_[curJ][j];
                  }
                  leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][1],j);

                  if (prev_i != MAX_UINT) {
                    const uint sclass = source_class_[source[cept_center[prev_i]]];

                    if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,sclass,tclass);

                    incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(aligned_source_words[aj][1],cept_center[prev_i]);
                  }
                  else if (use_sentence_start_prob_)
                    incoming_prob *= sentence_start_prob_[curJ][aligned_source_words[aj][1]];
                }
                else {
                  const uint tclass = target_class_[prev_ti];
		  
                  leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][jnum],aligned_source_words[aj][jnum-1]);

                  if (jnum+1 < prev_aj_fert) {
                    leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][jnum+1],aligned_source_words[aj][jnum]);
		    
                    incoming_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[aj][jnum+1],aligned_source_words[aj][jnum-1]);
                  }
                }
              }
              else {
                //the cept aj vanishes

                //erase the transitions prev_i -> aj    and    aj -> next_i
                if (prev_i != MAX_UINT) {
                  const uint tclass = target_class_[prev_ti];
                  const uint sclass = source_class_[source[cept_center[prev_i]]];

                  if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,sclass,tclass);

                  leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(j,cept_center[prev_i]);
                }
                else if (use_sentence_start_prob_) {
                  leaving_prob *= sentence_start_prob_[curJ][j];
                }

                if (next_i != MAX_UINT) {
                  const uint old_sclass = source_class_[source[j]];
                  const uint tclass = target_class_[target[next_i-1]];

                  if (cur_inter_distortion_prob(old_sclass,tclass).size() == 0)
                    par2nonpar_inter_distortion(curJ,old_sclass,tclass);

                  leaving_prob *= cur_inter_distortion_prob(old_sclass,tclass)(aligned_source_words[next_i][0],j);
                  
                  //introduce the transition prev_i -> next_i
                  if (prev_i != MAX_UINT) {
                    const uint new_sclass = source_class_[source[cept_center[prev_i]]];

                    if (cur_inter_distortion_prob(new_sclass,tclass).size() == 0)
                      par2nonpar_inter_distortion(curJ,new_sclass,tclass);

                    incoming_prob *= cur_inter_distortion_prob(new_sclass,tclass)(aligned_source_words[next_i][0],cept_center[prev_i]);
                  }
                  else if (use_sentence_start_prob_) {
                    incoming_prob *= sentence_start_prob_[curJ][aligned_source_words[next_i][0]];
                  } 
                }
              }
	    
              assert(leaving_prob > 0.0);
              
              hyp_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
              //DEBUG
              Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
              hyp_alignment[j] = cand_aj;
              long double check_prob = alignment_prob(source,target,lookup,hyp_alignment);
	      
              if (check_prob != 0.0) {
                
                long double check_ratio = hyp_prob / check_prob;
		
                if (! (check_ratio > 0.99 && check_ratio < 1.01) ) {
                  //if (true) {
                  
                  std::cerr << "incremental prob: " << hyp_prob << std::endl;
                  std::cerr << "actual prob: " << check_prob << std::endl;

                  std::cerr << ", J=" << curJ << ", I=" << curI << std::endl;
                  std::cerr << "base alignment: " << alignment << std::endl;
                  std::cerr << "moving source word " << j << " from " << alignment[j] << " to 0"
                            << std::endl;
                }
		
                if (check_prob > 1e-12 * best_prob)
                  assert (check_ratio > 0.99 && check_ratio < 1.01);
              }
              //END_DEBUG
#endif
            }	      
          }

          if (!incremental_calculation) {

	    std::vector<AlignBaseType>::iterator it = std::find(hyp_aligned_source_words[aj].begin(),
                                                                hyp_aligned_source_words[aj].end(),j);
	    hyp_aligned_source_words[aj].erase(it);

	    hyp_aligned_source_words[cand_aj].push_back(j);
	    std::sort(hyp_aligned_source_words[cand_aj].begin(),hyp_aligned_source_words[cand_aj].end());

	    hyp_prob = base_prob * distortion_prob(source,target,hyp_aligned_source_words)
	      / base_distortion_prob;

            assert(cand_aj != 0); //since we handle that case above

	    const uint prev_ti = (aj != 0) ? target[aj-1] : 0;
	    const uint new_ti = (cand_aj != 0) ? target[cand_aj-1] : 0;

            incoming_prob *= fertility_prob_[new_ti][fertility[cand_aj]+1];
	    if (aj != 0)
	      incoming_prob *= fertility_prob_[prev_ti][fertility[aj]-1];
	    
	    if (!no_factorial_) {
              incoming_prob *= ldfac(fertility[cand_aj]+1);
	      if (aj != 0)
		incoming_prob *= ldfac(fertility[aj]-1);
	    }

            leaving_prob *= fertility_prob_[new_ti][fertility[cand_aj]];
	    if (aj != 0)
	      leaving_prob *= fertility_prob_[prev_ti][fertility[aj]];
	    
	    assert(leaving_prob > 0.0);
	    
	    if (!no_factorial_) {
              leaving_prob *= ldfac(fertility[cand_aj]);
	      if (aj != 0)
		leaving_prob *= ldfac(fertility[aj]);
	    }


	    uint prev_zero_fert = fertility[0];
	    uint new_zero_fert = prev_zero_fert;
	    
	    if (aj == 0) {
	      new_zero_fert--;
	    }
	    else if (cand_aj == 0) {
	      new_zero_fert++;
	    }

	    if (prev_zero_fert != new_zero_fert) {

	      leaving_prob *= ldchoose(curJ-prev_zero_fert,prev_zero_fert);
	      for (uint k=1; k <= prev_zero_fert; k++)
		leaving_prob *= p_zero_;
	      for (uint k=1; k <= curJ-2*prev_zero_fert; k++)
		leaving_prob *= p_nonzero_;
	      
	      if (och_ney_empty_word_) {
		
		for (uint k=1; k<= prev_zero_fert; k++)
		  leaving_prob *= ((long double) k) / curJ;
	      }


	      incoming_prob *= ldchoose(curJ-new_zero_fert,new_zero_fert);
	      for (uint k=1; k <= new_zero_fert; k++)
		incoming_prob *= p_zero_;
	      for (uint k=1; k <= curJ-2*new_zero_fert; k++)
		incoming_prob *= p_nonzero_;
	      
	      if (och_ney_empty_word_) {
		
		for (uint k=1; k<= new_zero_fert; k++)
		  incoming_prob *= ((long double) k) / curJ;
	      }
	    }

	    hyp_prob *= incoming_prob / leaving_prob; 

	    //restore for next loop execution
	    hyp_aligned_source_words[aj] = aligned_source_words[aj];
	    hyp_aligned_source_words[cand_aj] = aligned_source_words[cand_aj];


	    //DEBUG
	    if (isnan(hyp_prob)) {
	      std::cerr << "incoming: " << incoming_prob << std::endl;
	      std::cerr << "leaving: " << leaving_prob << std::endl;
	    }
	    //END_DEBUG

#ifndef NDEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j] = cand_aj;

	    long double check = alignment_prob(source,target,lookup,hyp_alignment);

	    if (check > 1e-250) {

	      if (! (check / hyp_prob <= 1.005 && check / hyp_prob >= 0.995)) {
		std::cerr << "j: " << j << ", aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
		std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
                std::cerr << "base alignment: " << alignment << std::endl;

                std::cerr << "no_factorial_: " << no_factorial_ << std::endl;
                std::cerr << "prev_zero_fert: " << prev_zero_fert << ", new_zero_fert: " << new_zero_fert << std::endl;
	      }

	      assert(check / hyp_prob <= 1.005);
	      assert(check / hyp_prob >= 0.995);
	    }
	    else if (check > 0.0) {

	      if (! (check / hyp_prob <= 1.5 && check / hyp_prob >= 0.666)) {
		std::cerr << "aj: " << aj << ", cand_aj: " << cand_aj << std::endl;
		std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
	      }

	      assert(check / hyp_prob <= 1.5);
	      assert(check / hyp_prob >= 0.666);

	    }
	    else
	      assert(hyp_prob == 0.0);
#endif
          }
	    
          expansion_prob(j,cand_aj) = hyp_prob;

          if (isnan(expansion_prob(j,cand_aj))) {

            std::cerr << "nan in exp. move for j=" << j << ", " << aj << " -> " << cand_aj << std::endl;
            std::cerr << "current alignment: " << aj << std::endl;
            std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
            std::cerr << "incremental calculation: " << incremental_calculation << std::endl;

            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j] = cand_aj;
	    std::cerr << "prob. of start alignment: " 
		      << alignment_prob(source,target,lookup,alignment) << std::endl;

            std::cerr << "check prob: " << alignment_prob(source,target,lookup,hyp_alignment) << std::endl;	    
	    std::cerr << "base distortion prob: " << base_distortion_prob << std::endl;
	    std::cerr << "check-hyp distortion prob: " << distortion_prob(source,target,hyp_alignment) << std::endl;

	    print_alignment_prob_factors(source,target,lookup,alignment);
          }

          assert(!isnan(expansion_prob(j,cand_aj)));
          assert(!isinf(expansion_prob(j,cand_aj)));
	  
          if (hyp_prob > improvement_factor*best_prob) {
	    
            best_prob = hyp_prob;
            improvement = true;
            best_change_is_move = true;
            best_move_j = j;
            best_move_aj = cand_aj;
          }
        }    
      }
    }

    //tEndExp = std::clock();
    //std::clock_t tStartSwap,tEndSwap;
    //tStartSwap = std::clock();

    //for now, to be sure:
    hyp_aligned_source_words = aligned_source_words;

    //b) swap moves
    for (uint j1=0; j1 < curJ; j1++) {

      swap_prob(j1,j1) = 0.0;
      
      const uint aj1 = alignment[j1];

      for (uint j2 = j1+1; j2 < curJ; j2++) {

        const uint aj2 = alignment[j2];

        if (aj1 == aj2) {
          //we do not want to count the same alignment twice
          swap_prob(j1,j2) = 0.0;
        }
        else {

	  //EXPERIMENTAL (prune constellations with very unlikely translation probs.)
	  if (aj1 != 0) {
            const uint taj1 = target[aj1-1];
	    if (dict_[taj1][lookup(j2,aj1-1)] < 1e-10)
	      continue;
	  }
	  else {
	    if (dict_[0][source[j2]-1] < 1e-10)
	      continue;
	  }
	  if (aj2 != 0) {
            const uint taj2 = target[aj2-1];
	    if (dict_[taj2][lookup(j1,aj2-1)] < 1e-10)
	      continue;
	  }
	  else {
	    if (dict_[0][source[j1]-1] < 1e-10)
	      continue;
	  }
	  //END_EXPERIMENTAL


          long double hyp_prob = 0.0;

          if (aj1 != 0 && aj2 != 0 && 
              cept_start_mode_ != IBM4UNIFORM &&
              aligned_source_words[aj1].size() == 1 && aligned_source_words[aj2].size() == 1) {
            //both affected cepts are one-word cepts

            const uint taj1 = target[aj1-1];
            const uint taj2 = target[aj2-1];

            long double leaving_prob = 1.0;
            long double incoming_prob = 1.0;

            leaving_prob *= dict_[taj1][lookup(j1,aj1-1)];
            leaving_prob *= dict_[taj2][lookup(j2,aj2-1)];
            incoming_prob *= dict_[taj2][lookup(j1,aj2-1)];
            incoming_prob *= dict_[taj1][lookup(j2,aj1-1)];

            uint temp_aj1 = aj1;
            uint temp_aj2 = aj2;
            uint temp_j1 = j1;
            uint temp_j2 = j2;
            if (aj1 > aj2) {
              temp_aj1 = aj2;
              temp_aj2 = aj1;
              temp_j1 = j2;
              temp_j2 = j1;
            }

            // 1. entering cept temp_aj1
            if (prev_cept[temp_aj1] != MAX_UINT) {
              const uint sclass = source_class_[source[cept_center[prev_cept[temp_aj1]]]];
              const uint tclass = target_class_[target[temp_aj1-1]];

              if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass,tclass);

              leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(temp_j1,cept_center[prev_cept[temp_aj1]]);
              incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(temp_j2,cept_center[prev_cept[temp_aj1]]);
            }
            else if (use_sentence_start_prob_) {
              leaving_prob *= sentence_start_prob_[curJ][temp_j1];
              incoming_prob *= sentence_start_prob_[curJ][temp_j2];
            }

            // 2. leaving cept temp_aj1 and entering cept temp_aj2
            if (prev_cept[temp_aj2] != temp_aj1) {

              //a) leaving cept aj1
              const uint next_i = next_cept[temp_aj1];
              if (next_i != MAX_UINT) {
		
                const uint sclass1 = source_class_[source[temp_j1]];
                const uint sclass2 = source_class_[source[temp_j2]];
                const uint tclass = target_class_[target[next_i-1]];

                if (cur_inter_distortion_prob(sclass1,tclass).size() == 0)
                  par2nonpar_inter_distortion(curJ,sclass1,tclass);

                leaving_prob *= cur_inter_distortion_prob(sclass1,tclass)(aligned_source_words[next_i][0],temp_j1);

                if (cur_inter_distortion_prob(sclass2,tclass).size() == 0)
                  par2nonpar_inter_distortion(curJ,sclass2,tclass);

                incoming_prob *= cur_inter_distortion_prob(sclass2,tclass)(aligned_source_words[next_i][0],temp_j2);
              }
	      
              //b) entering cept temp_aj2
              const uint sclass = source_class_[source[cept_center[prev_cept[temp_aj2]]]];
              const uint tclass = target_class_[target[temp_aj2-1]];

              if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass,tclass);

              leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(temp_j2,cept_center[prev_cept[temp_aj2]]);
              incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(temp_j1,cept_center[prev_cept[temp_aj2]]);
            }
            else {
              //leaving cept temp_aj1 is simultaneously entering cept temp_aj2
              //NOTE: the aligned target word is here temp_aj2-1 in both the incoming and the leaving term

              const uint sclass1 = source_class_[source[temp_j1]];
              const uint tclass = target_class_[target[temp_aj2-1]];
              const uint sclass2 = source_class_[source[temp_j2]];

              if (cur_inter_distortion_prob(sclass1,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass1,tclass);

              leaving_prob *= cur_inter_distortion_prob(sclass1,tclass)(temp_j2,temp_j1);

              if (cur_inter_distortion_prob(sclass2,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass2,tclass);

              incoming_prob *= cur_inter_distortion_prob(sclass2,tclass)(temp_j1,temp_j2);
            }
	    
            // 3. leaving cept temp_aj2
            if (next_cept[temp_aj2] != MAX_UINT) {
              const uint sclass1 = source_class_[source[temp_j2]];
              const uint sclass2 = source_class_[source[temp_j1]];
	      const uint tclass = target_class_[target[next_cept[temp_aj2]-1]];

              if (cur_inter_distortion_prob(sclass1,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass1,tclass);

              leaving_prob *= cur_inter_distortion_prob(sclass1,tclass)(aligned_source_words[next_cept[temp_aj2]][0],temp_j2);

              if (cur_inter_distortion_prob(sclass2,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass2,tclass);

              incoming_prob *= cur_inter_distortion_prob(sclass2,tclass)(aligned_source_words[next_cept[temp_aj2]][0],temp_j1);
            }
	  
            hyp_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
            //DEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j1] = aj2;
            hyp_alignment[j2] = aj1;
            long double check_prob = alignment_prob(source,target,lookup,hyp_alignment);
            
            if (check_prob > 0.0) {
              
              long double check_ratio = check_prob / hyp_prob;
              
              if (! (check_ratio > 0.99 && check_ratio < 1.01)) {
                
                std::cerr << "******* swapping " << j1 << "->" << aj1 << " and " << j2 << "->" << aj2 << std::endl;
                std::cerr << " curJ: " << curJ << ", curI: " << curI << std::endl;
                std::cerr << "base alignment: " << alignment << std::endl;
                std::cerr << "actual prob: " << check_prob << std::endl;
                std::cerr << "incremental_hyp_prob: " << hyp_prob << std::endl;
                
              }
              
              assert(check_ratio > 0.99 && check_ratio < 1.01);
            }
            //END_DEBUG
#endif
          }
          else if (aj1 != 0 && aj2 != 0 && 
                   prev_cept[aj1] != aj2 && prev_cept[aj2] != aj1) {

            const uint taj1 = target[aj1-1];
            const uint taj2 = target[aj2-1];

            long double leaving_prob = 1.0;
            long double incoming_prob = 1.0;

            leaving_prob *= dict_[taj1][lookup(j1,aj1-1)];
            leaving_prob *= dict_[taj2][lookup(j2,aj2-1)];
            incoming_prob *= dict_[taj2][lookup(j1,aj2-1)];
            incoming_prob *= dict_[taj1][lookup(j2,aj1-1)];

            uint temp_aj1 = aj1;
            uint temp_aj2 = aj2;
            uint temp_j1 = j1;
            uint temp_j2 = j2;
            if (aj1 > aj2) {
              temp_aj1 = aj2;
              temp_aj2 = aj1;
              temp_j1 = j2;
              temp_j2 = j1;
            }

            uint old_j1_num = MAX_UINT;
            for (uint k=0; k < fertility[temp_aj1]; k++) {
              if (aligned_source_words[temp_aj1][k] == temp_j1) {
                old_j1_num = k;
                break;
              }
            }
            assert(old_j1_num != MAX_UINT);

            uint old_j2_num = MAX_UINT;
            for (uint k=0; k < fertility[temp_aj2]; k++) {
              if (aligned_source_words[temp_aj2][k] == temp_j2) {
                old_j2_num = k;
                break;
              }
            }
            assert(old_j2_num != MAX_UINT);

            std::vector<AlignBaseType> new_temp_aj1_aligned_source_words = aligned_source_words[temp_aj1];
            new_temp_aj1_aligned_source_words[old_j1_num] = temp_j2;
            std::sort(new_temp_aj1_aligned_source_words.begin(),new_temp_aj1_aligned_source_words.end());

            uint new_temp_aj1_center = MAX_UINT;
            switch (cept_start_mode_) {
            case IBM4CENTER : {
              double sum_j = 0.0;
              for (uint k=0; k < fertility[temp_aj1]; k++) {
                sum_j += new_temp_aj1_aligned_source_words[k];
              }
              new_temp_aj1_center = (uint) round(sum_j / fertility[temp_aj1]);
              break;
            }
            case IBM4FIRST : {
              new_temp_aj1_center = new_temp_aj1_aligned_source_words[0];
              break;
            }
            case IBM4LAST : {
              new_temp_aj1_center = new_temp_aj1_aligned_source_words[fertility[temp_aj1]-1];	      
              break;
            }
            case IBM4UNIFORM : {
              break;
            }
            default: assert(false);
            }
	    
            const int old_head1 = aligned_source_words[temp_aj1][0];
            const int new_head1 = new_temp_aj1_aligned_source_words[0];

            if (old_head1 != new_head1) {
              if (prev_cept[temp_aj1] != MAX_UINT) {
                const uint sclass = source_class_[source[cept_center[prev_cept[temp_aj1]]]];
                const uint tclass = target_class_[target[temp_aj1-1]];

                if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                  par2nonpar_inter_distortion(curJ,sclass,tclass);

                leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(old_head1,cept_center[prev_cept[temp_aj1]]);
                incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(new_head1,cept_center[prev_cept[temp_aj1]]);
              }
              else if (use_sentence_start_prob_) {
                leaving_prob *= sentence_start_prob_[curJ][old_head1];
                incoming_prob *= sentence_start_prob_[curJ][new_head1];
              }
            }

            for (uint k=1; k < fertility[temp_aj1]; k++) {
              const uint tclass = target_class_[target[temp_aj1-1]];
              leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[temp_aj1][k],aligned_source_words[temp_aj1][k-1]);
              incoming_prob *= cur_intra_distortion_prob(tclass,new_temp_aj1_aligned_source_words[k], 
                                                         new_temp_aj1_aligned_source_words[k-1]);
            }

            //transition to next cept
            if (next_cept[temp_aj1] != MAX_UINT) {
              const int next_head = aligned_source_words[next_cept[temp_aj1]][0];
	      
              const uint sclass1 = source_class_[source[cept_center[temp_aj1]]];
              const uint sclass2 = source_class_[source[new_temp_aj1_center]];
              const uint tclass = target_class_[target[next_cept[temp_aj1]-1]];

              if (cur_inter_distortion_prob(sclass1,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass1,tclass);

              leaving_prob *=  cur_inter_distortion_prob(sclass1,tclass)(next_head,cept_center[temp_aj1]);
              
              if (cur_inter_distortion_prob(sclass2,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass2,tclass);

              incoming_prob *= cur_inter_distortion_prob(sclass2,tclass)(next_head,new_temp_aj1_center);
            }

            std::vector<AlignBaseType> new_temp_aj2_aligned_source_words = aligned_source_words[temp_aj2];
            new_temp_aj2_aligned_source_words[old_j2_num] = temp_j1;
            std::sort(new_temp_aj2_aligned_source_words.begin(),new_temp_aj2_aligned_source_words.end());

            uint new_temp_aj2_center = MAX_UINT;
            switch (cept_start_mode_) {
            case IBM4CENTER : {
              double sum_j = 0.0;
              for (uint k=0; k < fertility[temp_aj2]; k++) {
                sum_j += new_temp_aj2_aligned_source_words[k];
              }
              new_temp_aj2_center = (uint) round(sum_j / fertility[temp_aj2]);
              break;
            }
            case IBM4FIRST : {
              new_temp_aj2_center = new_temp_aj2_aligned_source_words[0];
              break;
            }
            case IBM4LAST : {
              new_temp_aj2_center = new_temp_aj2_aligned_source_words[fertility[temp_aj2]-1];
              break;
            }
            case IBM4UNIFORM : {
              break;
            }
            default: assert(false);
            }

            const int old_head2 = aligned_source_words[temp_aj2][0];
            const int new_head2 = new_temp_aj2_aligned_source_words[0];

            if (old_head2 != new_head2) {
              if (prev_cept[temp_aj2] != MAX_UINT) {
                const uint sclass = source_class_[source[cept_center[prev_cept[temp_aj2]]]];
                const uint tclass = target_class_[target[temp_aj2-1]];

                if (cur_inter_distortion_prob(sclass,tclass).size() == 0)
                  par2nonpar_inter_distortion(curJ,sclass,tclass);

                leaving_prob *= cur_inter_distortion_prob(sclass,tclass)(old_head2,cept_center[prev_cept[temp_aj2]]);
                incoming_prob *= cur_inter_distortion_prob(sclass,tclass)(new_head2,cept_center[prev_cept[temp_aj2]]);
              }
              else if (use_sentence_start_prob_) {
                leaving_prob *= sentence_start_prob_[curJ][old_head2];
                incoming_prob *= sentence_start_prob_[curJ][new_head2];
              }
            }
	    
            for (uint k=1; k < fertility[temp_aj2]; k++) {
              const uint tclass = target_class_[target[temp_aj2-1]];
              leaving_prob *= cur_intra_distortion_prob(tclass,aligned_source_words[temp_aj2][k],aligned_source_words[temp_aj2][k-1]);
              incoming_prob *= cur_intra_distortion_prob(tclass,new_temp_aj2_aligned_source_words[k],
                                                         new_temp_aj2_aligned_source_words[k-1]);
            }

            //transition to next cept
            if (next_cept[temp_aj2] != MAX_UINT && cept_center[temp_aj2] != new_temp_aj2_center) {
              const int next_head = aligned_source_words[next_cept[temp_aj2]][0];
	      
              const uint sclass1 = source_class_[source[cept_center[temp_aj2]]];
              const uint sclass2 = source_class_[source[new_temp_aj2_center]];
              const uint tclass = target_class_[target[next_cept[temp_aj2]-1]];

              if (cur_inter_distortion_prob(sclass1,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass1,tclass);

              leaving_prob *= cur_inter_distortion_prob(sclass1,tclass)(next_head,cept_center[temp_aj2]);

              if (cur_inter_distortion_prob(sclass2,tclass).size() == 0)
                par2nonpar_inter_distortion(curJ,sclass2,tclass);

              incoming_prob *= cur_inter_distortion_prob(sclass2,tclass)(next_head,new_temp_aj2_center);
            }

            hyp_prob = base_prob * incoming_prob / leaving_prob;

#ifndef NDEBUG
            //DEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j1] = aj2;
            hyp_alignment[j2] = aj1;
            long double check_prob = alignment_prob(source,target,lookup,hyp_alignment);
                    
            if (check_prob > 0.0) {
              
              long double check_ratio = check_prob / hyp_prob;
              
              if (! (check_ratio > 0.99 && check_ratio < 1.01)) {
                
                std::cerr << "******* swapping " << j1 << "->" << aj1 << " and " << j2 << "->" << aj2 << std::endl;
                std::cerr << "curJ: " << curJ << ", curI: " << curI << std::endl;
                std::cerr << "base alignment: " << alignment << std::endl;
                std::cerr << "actual prob: " << check_prob << std::endl;
                std::cerr << "incremental_hyp_prob: " << hyp_prob << std::endl;
                std::cerr << "(base prob: " << base_prob << ")" << std::endl;
              }
              
              if (check_prob > 1e-12*base_prob) 
                assert(check_ratio > 0.99 && check_ratio < 1.01);
            }
            //END_DEBUG
#endif

          }
          else {

	    std::vector<AlignBaseType>::iterator it = std::find(hyp_aligned_source_words[aj1].begin(),
                                                                hyp_aligned_source_words[aj1].end(),j1);
	    hyp_aligned_source_words[aj1].erase(it);
	    it = std::find(hyp_aligned_source_words[aj2].begin(),hyp_aligned_source_words[aj2].end(),j2);
	    hyp_aligned_source_words[aj2].erase(it);
	    hyp_aligned_source_words[aj1].push_back(j2);
	    hyp_aligned_source_words[aj2].push_back(j1);

	    std::sort(hyp_aligned_source_words[aj1].begin(), hyp_aligned_source_words[aj1].end());
	    std::sort(hyp_aligned_source_words[aj2].begin(), hyp_aligned_source_words[aj2].end());

	    hyp_prob = base_prob * distortion_prob(source,target,hyp_aligned_source_words)
	      / base_distortion_prob;


	    const uint ti1 = (aj1 != 0) ? target[aj1-1] : 0;
	    const uint ti2 = (aj2 != 0) ? target[aj2-1] : 0;

	    long double incoming_prob; 
	    long double leaving_prob;

	    if (aj1 != 0) {
	      leaving_prob = dict_[ti1][lookup(j1,aj1-1)];
	      incoming_prob = dict_[ti1][lookup(j2,aj1-1)];
	    }
	    else {
	      leaving_prob = dict_[0][source[j1]-1];
	      incoming_prob = dict_[0][source[j2]-1];
	    }
	    
	    if (aj2 != 0) {
	      leaving_prob *= dict_[ti2][lookup(j2,aj2-1)];
	      incoming_prob *= dict_[ti2][lookup(j1,aj2-1)];
	    }
	    else {
	      leaving_prob *= dict_[0][source[j2]-1];
	      incoming_prob *= dict_[0][source[j1]-1];
	    }
	    
	    assert(leaving_prob > 0.0);
	    
	    hyp_prob *= incoming_prob / leaving_prob; 

	    //restore for next loop execution:
	    hyp_aligned_source_words[aj1] = aligned_source_words[aj1];
	    hyp_aligned_source_words[aj2] = aligned_source_words[aj2];

#ifndef NDEBUG
            Math1D::Vector<AlignBaseType> hyp_alignment = alignment;
            hyp_alignment[j1] = aj2;
            hyp_alignment[j2] = aj1;

	    long double check = alignment_prob(source,target,lookup,hyp_alignment);

	    if (check > 1e-250) {

	      if (! (check / hyp_prob <= 1.005 && check / hyp_prob >= 0.995)) {
		std::cerr << "aj1: " << aj1 << ", aj2: " << aj2 << std::endl;
		std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
	      }

	      assert(check / hyp_prob <= 1.005);
	      assert(check / hyp_prob >= 0.995);
	    }
	    else if (check > 0.0) {

	      if (! (check / hyp_prob <= 1.5 && check / hyp_prob >= 0.666)) {
		std::cerr << "aj1: " << aj1 << ", aj2: " << aj2 << std::endl;
		std::cerr << "calculated: " << hyp_prob << ", should be: " << check << std::endl;
	      }

	      assert(check / hyp_prob <= 1.5);
	      assert(check / hyp_prob >= 0.666);

	    }
	    else
	      assert(hyp_prob == 0.0);
#endif
          } //end of case 3

          assert(!isnan(hyp_prob));

          swap_prob(j1,j2) = hyp_prob;

          if (hyp_prob > improvement_factor*best_prob) {
	    
            improvement = true;
            best_change_is_move = false;
            best_prob = hyp_prob;
            best_swap_j1 = j1;
            best_swap_j2 = j2;
          }
        }

        assert(!isnan(swap_prob(j1,j2)));
        assert(!isinf(swap_prob(j1,j2)));

        swap_prob(j2,j1) = swap_prob(j1,j2);

      }
    }

    //tEndSwap = std::clock();

    //update alignment if a better one was found
    if (!improvement)
      break;

    //update alignment
    if (best_change_is_move) {
      uint cur_aj = alignment[best_move_j];
      assert(cur_aj != best_move_aj);

      alignment[best_move_j] = best_move_aj;
      fertility[cur_aj]--;
      fertility[best_move_aj]++;

      aligned_source_words[best_move_aj].push_back(best_move_j);
      std::sort(aligned_source_words[best_move_aj].begin(), aligned_source_words[best_move_aj].end());

      std::vector<AlignBaseType>::iterator it = std::find(aligned_source_words[cur_aj].begin(),
                                                          aligned_source_words[cur_aj].end(),best_move_j);
      assert(it != aligned_source_words[cur_aj].end());

      aligned_source_words[cur_aj].erase(it);
    }
    else {

      uint cur_aj1 = alignment[best_swap_j1];
      uint cur_aj2 = alignment[best_swap_j2];

      assert(cur_aj1 != cur_aj2);
      
      alignment[best_swap_j1] = cur_aj2;
      alignment[best_swap_j2] = cur_aj1;

      //NOTE: the fertilities are not affected here
      for (uint k=0; k < aligned_source_words[cur_aj2].size(); k++) {
	if (aligned_source_words[cur_aj2][k] == best_swap_j2) {
	  aligned_source_words[cur_aj2][k] = best_swap_j1;
          break;
        }
      }
      for (uint k=0; k < aligned_source_words[cur_aj1].size(); k++) {
	if (aligned_source_words[cur_aj1][k] == best_swap_j1) {
	  aligned_source_words[cur_aj1][k] = best_swap_j2;
          break;
        }
      }

      std::sort(aligned_source_words[cur_aj1].begin(), aligned_source_words[cur_aj1].end());
      std::sort(aligned_source_words[cur_aj2].begin(), aligned_source_words[cur_aj2].end());
    }

    base_prob = best_prob;    

#ifndef NDEBUG
    double check_ratio = alignment_prob(source,target,lookup,alignment) / base_prob;

    if (base_prob > 1e-250) {

      if ( !(check_ratio >= 0.99 && check_ratio <= 1.01)) {
        std::cerr << "check: " << alignment_prob(source,target,lookup,alignment) << std::endl;;
      }

      assert(check_ratio >= 0.99 && check_ratio <= 1.01);
    }
#endif

    base_distortion_prob = distortion_prob(source,target,aligned_source_words);
  }

  return base_prob;
}


long double IBM4Trainer::compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                                    const Math2D::Matrix<uint>& lookup,
                                                    Math1D::Vector<AlignBaseType>& alignment) {

  const uint J = source.size();
  const uint I = target.size();

  if (alignment.size() != J)
    alignment.resize(J,1);

  Math1D::Vector<uint> fertility(I+1,0);

  for (uint j=0; j < J; j++) {
    const uint aj = alignment[j];

    if (aj > 0) {
      if (dict_[target[aj-1]][lookup(j,aj-1)] < 1e-15)
	dict_[target[aj-1]][lookup(j,aj-1)] = 1e-15;
    }
    else {
      if (dict_[0][source[j]-1] < 1e-15)
	dict_[0][source[j]-1] = 1e-15;
    }

    fertility[aj]++;
  }

  if (fertility[0] > 0 && p_zero_ < 1e-12)
    p_zero_ = 1e-12;
  
  if (2*fertility[0] > J) {
    
    for (uint j=0; j < J; j++) {
      
      if (alignment[j] == 0) {
	
        alignment[j] = 1;
        fertility[0]--;
        fertility[1]++;	

	if (dict_[target[0]][lookup(j,0)] < 1e-12)
	  dict_[target[0]][lookup(j,0)] = 1e-12;
      }
    }
  }


  /*** check if respective distortion table is present. If not, create one from the parameters ***/

  int oldJ = (cept_start_prob_.zDim() + 1) / 2;

  bool update = false;

  if (oldJ < int(J)) {
    update = true;

    //inter params
    IBM4CeptStartModel new_param(cept_start_prob_.xDim(),cept_start_prob_.yDim(),2*J-1,0.0,MAKENAME(new_param));
    uint new_zero_offset = J-1;
    for (int j = -int(maxJ_)+1; j <= int(maxJ_)-1; j++) {

      for (uint w1=0; w1 < cept_start_prob_.xDim(); w1++) 
        for (uint w2=0; w2 < cept_start_prob_.yDim(); w2++) 
          new_param(w1,w2,new_zero_offset + j) = cept_start_prob_(w1,w2,displacement_offset_ + j);

    }
    cept_start_prob_ = new_param;

    //intra params

    IBM4WithinCeptModel new_wi_model(within_cept_prob_.xDim(),J,0.0,MAKENAME(new_wi_model)); 

    for (uint c=0; c < new_wi_model.xDim(); c++) {

      for (uint k=0; k < within_cept_prob_.yDim(); k++)
	new_wi_model(c,k) = within_cept_prob_(c,k);
    }

    within_cept_prob_ = new_wi_model;

    displacement_offset_ = new_zero_offset;

    maxJ_ = J;

    sentence_start_parameters_.resize(J,0.0);
  }

  if (inter_distortion_prob_.size() <= J) {
    update = true;
    inter_distortion_prob_.resize(J+1);
  }
  if (intra_distortion_prob_.size() <= J) {
    update = true;
    intra_distortion_prob_.resize(J+1);
  }
 
  uint max_s=0;
  uint max_t=0;
  for (uint s=0; s < source.size(); s++) {
    max_s = std::max<uint>(max_s,source_class_[source[s]]);
  }
  for (uint t=0; t < target.size(); t++) {
    max_t = std::max<uint>(max_t,target_class_[target[t]]);
  }
  inter_distortion_prob_[J].resize(std::max<uint>(inter_distortion_prob_[J].xDim(),max_s+1),
                                   std::max<uint>(inter_distortion_prob_[J].yDim(),max_t+1));

  if (intra_distortion_prob_[J].xDim() <= max_t) {
    update = true;
    intra_distortion_prob_[J].resize(max_t+1,J,J);
  }

  if (use_sentence_start_prob_) {

    if (sentence_start_prob_.size() <= J) {
      update = true;
      sentence_start_prob_.resize(J+1);
    }

    if (sentence_start_prob_[J].size() < J) {
      update = true;
      sentence_start_prob_[J].resize(J);
    } 
  }  

  if (update) {
    par2nonpar_inter_distortion();

    par2nonpar_intra_distortion();
    if (use_sentence_start_prob_) 
      par2nonpar_start_prob();
  }


  /*** check if fertility tables are large enough ***/
  for (uint i=0; i < I; i++) {

    if (fertility_prob_[target[i]].size() < J+1)
      fertility_prob_[target[i]].resize(J+1,1e-15);

    if (fertility_prob_[target[i]][fertility[i+1]] < 1e-15)
      fertility_prob_[target[i]][fertility[i+1]] = 1e-15;

    if (fertility_prob_[target[i]].sum() < 0.5)
      fertility_prob_[target[i]].set_constant(1.0 / fertility_prob_[target[i]].size());

    if (fertility_prob_[target[i]][fertility[i+1]] < 1e-8)
      fertility_prob_[target[i]][fertility[i+1]] = 1e-8;
  }

  /*** check if a source word does not have a translation (with non-zero prob.) ***/
  for (uint j=0; j < J; j++) {
    uint src_idx = source[j];

    double sum = dict_[0][src_idx-1];
    for (uint i=0; i < I; i++)
      sum += dict_[target[i]][lookup(j,i)];

    if (sum < 1e-100) {
      for (uint i=0; i < I; i++)
        dict_[target[i]][lookup(j,i)] = 1e-15;
    }

    uint aj = alignment[j];
    if (aj == 0) {
      if (dict_[0][src_idx-1] < 1e-20)
        dict_[0][src_idx-1] = 1e-20;
    }
    else {
      if (dict_[target[aj-1]][lookup(j,aj-1)] < 1e-20)
        dict_[target[aj-1]][lookup(j,aj-1)] = 1e-20;
    }
  }

  //create matrices
  Math2D::Matrix<long double> expansion_prob(J,I+1);
  Math2D::Matrix<long double> swap_prob(J,J);
  
  uint nIter;
  
  return update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility,
                                          expansion_prob, swap_prob, alignment);
}

// <code> start_alignment </code> is used as initialization for hillclimbing and later modified
// the extracted alignment is written to <code> postdec_alignment </code>
void IBM4Trainer::compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
						     const Math2D::Matrix<uint>& lookup,
						     Math1D::Vector<AlignBaseType>& alignment,
						     std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
						     double threshold) {

  postdec_alignment.clear();

  const uint J = source.size();
  const uint I = target.size();

  if (alignment.size() != J)
    alignment.resize(J,1);

  Math1D::Vector<uint> fertility(I+1,0);

  for (uint j=0; j < J; j++) {
    const uint aj = alignment[j];

    if (aj > 0) {
      if (dict_[target[aj-1]][lookup(j,aj-1)] < 1e-15)
	dict_[target[aj-1]][lookup(j,aj-1)] = 1e-15;
    }
    else {
      if (dict_[0][source[j]-1] < 1e-15)
	dict_[0][source[j]-1] = 1e-15;
    }

    fertility[aj]++;
  }

  if (fertility[0] > 0 && p_zero_ < 1e-12)
    p_zero_ = 1e-12;
  
  if (2*fertility[0] > J) {
    
    for (uint j=0; j < J; j++) {
      
      if (alignment[j] == 0) {
	
        alignment[j] = 1;
        fertility[0]--;
        fertility[1]++;	

	if (dict_[target[0]][lookup(j,0)] < 1e-12)
	  dict_[target[0]][lookup(j,0)] = 1e-12;
      }
    }
  }


  /*** check if respective distortion table is present. If not, create one from the parameters ***/

  int oldJ = (cept_start_prob_.zDim() + 1) / 2;

  bool update = false;

  if (oldJ < int(J)) {
    update = true;

    //inter params
    IBM4CeptStartModel new_param(cept_start_prob_.xDim(),cept_start_prob_.yDim(),2*J-1,0.0,MAKENAME(new_param));
    uint new_zero_offset = J-1;
    for (int j = -int(maxJ_)+1; j <= int(maxJ_)-1; j++) {

      for (uint w1=0; w1 < cept_start_prob_.xDim(); w1++) 
        for (uint w2=0; w2 < cept_start_prob_.yDim(); w2++) 
          new_param(w1,w2,new_zero_offset + j) = cept_start_prob_(w1,w2,displacement_offset_ + j);

    }
    cept_start_prob_ = new_param;

    //intra params

    //const uint nNewDisplacements = 2*J-1;

    IBM4WithinCeptModel new_wi_model(within_cept_prob_.xDim(),J,0.0,MAKENAME(new_wi_model)); 

    for (uint c=0; c < new_wi_model.xDim(); c++) {

      for (uint k=0; k < within_cept_prob_.yDim(); k++)
	new_wi_model(c,k) = within_cept_prob_(c,k);
    }

    within_cept_prob_ = new_wi_model;

    displacement_offset_ = new_zero_offset;

    maxJ_ = J;

    sentence_start_parameters_.resize(J,0.0);
  }

  if (inter_distortion_prob_.size() <= J) {
    update = true;
    inter_distortion_prob_.resize(J+1);
  }
  if (intra_distortion_prob_.size() <= J) {
    update = true;
    intra_distortion_prob_.resize(J+1);
  }
    
  if (inter_distortion_prob_[J].yDim() <= J) {
    update = true;
    inter_distortion_prob_[J].resize(1,1);
    for (uint c1 = 0; c1 < inter_distortion_prob_[J].xDim(); c1++)
      for (uint c2 = 0; c2 < inter_distortion_prob_[J].yDim(); c2++)
	inter_distortion_prob_[J](c1,c2).resize(J,J);
  }

  if (intra_distortion_prob_[J].yDim() <= J) {
    update = true;
    intra_distortion_prob_[J].resize(within_cept_prob_.xDim(),J,J);
  }

  if (use_sentence_start_prob_) {

    if (sentence_start_prob_.size() <= J) {
      update = true;
      sentence_start_prob_.resize(J+1);
    }

    if (sentence_start_prob_[J].size() < J) {
      update = true;
      sentence_start_prob_[J].resize(J);
    } 
  }  

  if (update) {

    par2nonpar_inter_distortion();

    par2nonpar_intra_distortion();
    if (use_sentence_start_prob_) 
      par2nonpar_start_prob();
  }

  /*** check if fertility tables are large enough ***/
  for (uint i=0; i < I; i++) {

    if (fertility_prob_[target[i]].size() < J+1)
      fertility_prob_[target[i]].resize(J+1,1e-15);

    if (fertility_prob_[target[i]][fertility[i+1]] < 1e-15)
      fertility_prob_[target[i]][fertility[i+1]] = 1e-15;

    if (fertility_prob_[target[i]].sum() < 0.5)
      fertility_prob_[target[i]].set_constant(1.0 / fertility_prob_[target[i]].size());

    if (fertility_prob_[target[i]][fertility[i+1]] < 1e-8)
      fertility_prob_[target[i]][fertility[i+1]] = 1e-8;
  }

  /*** check if a source word does not have a translation (with non-zero prob.) ***/
  for (uint j=0; j < J; j++) {
    uint src_idx = source[j];

    double sum = dict_[0][src_idx-1];
    for (uint i=0; i < I; i++)
      sum += dict_[target[i]][lookup(j,i)];

    if (sum < 1e-100) {
      for (uint i=0; i < I; i++)
        dict_[target[i]][lookup(j,i)] = 1e-15;
    }

    uint aj = alignment[j];
    if (aj == 0) {
      if (dict_[0][src_idx-1] < 1e-20)
        dict_[0][src_idx-1] = 1e-20;
    }
    else {
      if (dict_[target[aj-1]][lookup(j,aj-1)] < 1e-20)
        dict_[target[aj-1]][lookup(j,aj-1)] = 1e-20;
    }
  }

  //create matrices
  Math2D::Matrix<long double> expansion_move_prob(J,I+1);
  Math2D::Matrix<long double> swap_move_prob(J,J);
  
  uint nIter;
  long double best_prob;

  best_prob = update_alignment_by_hillclimbing(source, target, lookup, nIter, fertility,
					       expansion_move_prob, swap_move_prob, alignment);


  const long double expansion_prob = expansion_move_prob.sum();
  const long double swap_prob =  0.5 * swap_move_prob.sum();
  
  const long double sentence_prob = best_prob + expansion_prob +  swap_prob;
  
  /**** calculate sums ***/
  Math2D::Matrix<long double> marg(J,I+1,0.0);

  for (uint j=0; j < J; j++) {
    marg(j, alignment[j]) += best_prob;
    for (uint i=0; i <= I; i++) {
      marg(j,i) += expansion_move_prob(j,i);
      for (uint jj=0; jj < J; jj++) {
	if (jj != j) {
	  marg(jj,alignment[jj]) += expansion_move_prob(j,i);
	}
      }
    }
    for (uint jj=j+1; jj < J; jj++) {
      marg(j,alignment[jj]) += swap_move_prob(j,jj);
      marg(jj,alignment[j]) += swap_move_prob(j,jj);

      for (uint jjj=0; jjj < J; jjj++)
	if (jjj != j && jjj != jj)
	  marg(jjj,alignment[jjj]) += swap_move_prob(j,jj);
    }
  }

  /*** compute marginals and threshold ***/
  for (uint j=0; j < J; j++) {

    //DEBUG
    long double check = 0.0;
    for (uint i=0; i <= I; i++)
      check += marg(j,i);
    long double ratio = sentence_prob/check;
    assert( ratio >= 0.99);
    assert( ratio <= 1.01);
    //END_DEBUG

    for (uint i=1; i <= I; i++) {

      long double cur_marg = marg(j,i) / sentence_prob;

      if (cur_marg >= threshold) {
	postdec_alignment.insert(std::make_pair(j+1,i));
      }
    }
  }
}

DistortCount::DistortCount(uchar J, uchar j, uchar j_prev)
  : J_(J), j_(j), j_prev_(j_prev) {}

bool operator<(const DistortCount& d1, const DistortCount& d2) {
  if (d1.J_ != d2.J_)
    return (d1.J_ < d2.J_);
  if (d1.j_ != d2.j_)
    return (d1.j_ < d2.j_);
  return (d1.j_prev_ < d2.j_prev_);
}

void IBM4Trainer::train_unconstrained(uint nIter, IBM3Trainer* ibm3) {

  std::cerr << "starting IBM-4 training without constraints";
  if (ibm3 != 0)
    std::cerr << " (init from IBM-3) ";
  std::cerr << std::endl;


  double max_perplexity = 0.0;
  double approx_sum_perplexity = 0.0;

  IBM4CeptStartModel fceptstart_count(cept_start_prob_.xDim(),cept_start_prob_.yDim(),2*maxJ_-1,MAKENAME(fceptstart_count));
  IBM4WithinCeptModel fwithincept_count(within_cept_prob_.xDim(),within_cept_prob_.yDim(),MAKENAME(fwithincept_count));
  Math1D::NamedVector<double> fsentence_start_count(maxJ_,MAKENAME(fsentence_start_count));

  Storage1D<Storage2D<Math2D::Matrix<double> > > inter_distort_count(maxJ_+1);
  Storage1D<Math3D::Tensor<double> > intra_distort_count(maxJ_+1);
  
  Storage1D<Math1D::Vector<double> > sentence_start_count(maxJ_+1);

  for (uint J=1; J <= maxJ_; J++) {

    if (inter_distortion_prob_[J].size() > 0) {
      inter_distort_count[J].resize(inter_distortion_prob_[J].xDim(),inter_distortion_prob_[J].yDim());

      if (J <= 10 || nSourceClasses_*nTargetClasses_ <= 10) {
	for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
	  for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
	    inter_distort_count[J](x,y).resize(inter_distortion_prob_[J](x,y).xDim(),inter_distortion_prob_[J](x,y).yDim(),0.0);
      }
      
      intra_distort_count[J].resize(intra_distortion_prob_[J].xDim(),intra_distortion_prob_[J].yDim(),
                                    intra_distortion_prob_[J].zDim(),0.0);
    }
    
    sentence_start_count[J].resize(J,0);
  }

  double dict_weight_sum = 0.0;
  for (uint i=0; i < nTargetWords_; i++) {
    dict_weight_sum += fabs(prior_weight_[i].sum());
  }

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords,MAKENAME(ffert_count));

  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  long double fzero_count;
  long double fnonzero_count;

  double hillclimbtime = 0.0;
  double countcollecttime = 0.0;

  for (uint iter=1; iter <= nIter; iter++) {

    Storage2D<std::map<DistortCount,double> > sparse_inter_distort_count(nSourceClasses_,nTargetClasses_);

    std::cerr << "******* IBM-4 EM-iteration " << iter << std::endl;

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    fceptstart_count.set_constant(0.0);
    fwithincept_count.set_constant(0.0);
    fsentence_start_count.set_constant(0.0);

    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    for (uint J=1; J <= maxJ_; J++) {
      if (inter_distort_count[J].size() > 0) {
        for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
          for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
            inter_distort_count[J](x,y).set_constant(0.0);
      }
      intra_distort_count[J].set_constant(0.0);
      sentence_start_count[J].set_constant(0.0);
    }

    max_perplexity = 0.0;
    approx_sum_perplexity = 0.0;

    for (size_t s=0; s < source_sentence_.size(); s++) {

      if ((s% 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;


      if (nSourceClasses_*nTargetClasses_ >= 10 && (s%25) == 0) {
        for (uint J=11; J < inter_distortion_prob_.size(); J++) {

          for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
            for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
              inter_distortion_prob_[J](x,y).resize(0,0);
        }
      }
      
      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const Math2D::Matrix<uint>& cur_lookup = slookup_[s];
      
      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      
      Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ,curJ,MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ,curI+1,MAKENAME(expansion_move_prob));

      std::clock_t tHillclimbStart, tHillclimbEnd;
      tHillclimbStart = std::clock();

      long double best_prob = 0.0;

      if (ibm3 != 0 && iter == 1) {
	best_prob = ibm3->update_alignment_by_hillclimbing(cur_source,cur_target,cur_lookup,sum_iter,fertility,
							   expansion_move_prob,swap_move_prob,best_known_alignment_[s]);	
      }
      else {
	best_prob = update_alignment_by_hillclimbing(cur_source,cur_target,cur_lookup,sum_iter,fertility,
						     expansion_move_prob,swap_move_prob,best_known_alignment_[s]);
      }
      max_perplexity -= std::log(best_prob);

      tHillclimbEnd = std::clock();

      hillclimbtime += diff_seconds(tHillclimbEnd,tHillclimbStart);

      const long double expansion_prob = expansion_move_prob.sum();
      const long double swap_prob =  0.5 * swap_move_prob.sum();

      const long double sentence_prob = best_prob + expansion_prob +  swap_prob;

      approx_sum_perplexity -= std::log(sentence_prob);
      
      const long double inv_sentence_prob = 1.0 / sentence_prob;

      /**** update empty word counts *****/
	
      double cur_zero_weight = best_prob;
      for (uint j=0; j < curJ; j++) {
        if (best_known_alignment_[s][j] == 0) {
	  
          for (uint jj=j+1; jj < curJ; jj++) {
            if (best_known_alignment_[s][jj] != 0)
              cur_zero_weight += swap_move_prob(j,jj);
          }
        }
      }
      cur_zero_weight *= inv_sentence_prob;

      assert(!isnan(cur_zero_weight));
      assert(!isinf(cur_zero_weight));
      
      fzero_count += cur_zero_weight * (fertility[0]);
      fnonzero_count += cur_zero_weight * (curJ - 2*fertility[0]);

      if (curJ >= 2*(fertility[0]+1)) {
        long double inc_zero_weight = 0.0;
        for (uint j=0; j < curJ; j++)
          inc_zero_weight += expansion_move_prob(j,0);
	
        inc_zero_weight *= inv_sentence_prob;
        fzero_count += inc_zero_weight * (fertility[0]+1);
        fnonzero_count += inc_zero_weight * (curJ -2*(fertility[0]+1));

        assert(!isnan(inc_zero_weight));
        assert(!isinf(inc_zero_weight));
      }

      if (fertility[0] > 1) {
        long double dec_zero_weight = 0.0;
        for (uint j=0; j < curJ; j++) {
          if (best_known_alignment_[s][j] == 0) {
            for (uint i=1; i <= curI; i++)
              dec_zero_weight += expansion_move_prob(j,i);
          }
        }
      
        dec_zero_weight *= inv_sentence_prob;

        fzero_count += dec_zero_weight * (fertility[0]-1);
        fnonzero_count += dec_zero_weight * (curJ -2*(fertility[0]-1));

        assert(!isnan(dec_zero_weight));
        assert(!isinf(dec_zero_weight));
      }

      /**** update fertility counts *****/
      for (uint i=1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i-1];

        long double addon = sentence_prob;
        for (uint j=0; j < curJ; j++) {
          if (best_known_alignment_[s][j] == i) {
            for (uint ii=0; ii <= curI; ii++)
              addon -= expansion_move_prob(j,ii);
          }
          else
            addon -= expansion_move_prob(j,i);
        }
        addon *= inv_sentence_prob;

        double daddon = (double) addon;
        if (!(daddon > 0.0)) {
          std::cerr << "STRANGE: fractional weight " << daddon << " for sentence pair #" << s << " with "
                    << curJ << " source words and " << curI << " target words" << std::endl;
          std::cerr << "best alignment prob: " << best_prob << std::endl;
          std::cerr << "sentence prob: " << sentence_prob << std::endl;
          std::cerr << "" << std::endl;

          //DEBUG
          exit(1);
          //END_DEBUG
        }

        ffert_count[t_idx][cur_fert] += addon;

        //NOTE: swap moves do not change the fertilities
        if (cur_fert > 0) {
          long double alt_addon = 0.0;
          for (uint j=0; j < curJ; j++) {
            if (best_known_alignment_[s][j] == i) {
              for (uint ii=0; ii <= curI; ii++) {
                if (ii != i)
                  alt_addon += expansion_move_prob(j,ii);
              }
            }
          }

          ffert_count[t_idx][cur_fert-1] += inv_sentence_prob * alt_addon;
        }

        if (cur_fert+1 < fertility_prob_[t_idx].size()) {

          long double alt_addon = 0.0;
          for (uint j=0; j < curJ; j++) {
            if (best_known_alignment_[s][j] != i) {
              alt_addon += expansion_move_prob(j,i);
            }
          }

          ffert_count[t_idx][cur_fert+1] += inv_sentence_prob * alt_addon;
        }
      }

      /**** update dictionary counts *****/
      for (uint j=0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = best_known_alignment_[s][j];

        long double addon = sentence_prob;
        for (uint i=0; i <= curI; i++) 
          addon -= expansion_move_prob(j,i);
        for (uint jj=0; jj < curJ; jj++)
          addon -= swap_move_prob(j,jj);

        addon *= inv_sentence_prob;
        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj-1]][cur_lookup(j,cur_aj-1)] += addon;
        }
        else {
          fwcount[0][s_idx-1] += addon;
        }

        for (uint i=0; i <= curI; i++) {

          if (i != cur_aj) {

            long double addon = expansion_move_prob(j,i);
            for (uint jj=0; jj < curJ; jj++) {
              if (best_known_alignment_[s][jj] == i)
                addon += swap_move_prob(j,jj);
            }
            addon *= inv_sentence_prob;

            if (i!=0) {
              fwcount[cur_target[i-1]][cur_lookup(j,i-1)] += addon;
            }
            else {
              fwcount[0][s_idx-1] += addon;
            }
          }
        }
      }

      std::clock_t tCountCollectStart, tCountCollectEnd;
      tCountCollectStart = std::clock();

      /**** update distortion counts *****/
      NamedStorage1D<std::set<int> > aligned_source_words(curI+1,MAKENAME(aligned_source_words));
      Math1D::NamedVector<int> cept_center(curI+1,-100,MAKENAME(cept_center));

      //denotes the largest preceding target position that produces source words
      Math1D::NamedVector<int> prev_cept(curI+1,-100,MAKENAME(prev_cept));
      Math1D::NamedVector<int> first_aligned_source_word(curI+1,-100,
                                                         MAKENAME(first_aligned_source_word));
      Math1D::NamedVector<int> second_aligned_source_word(curI+1,-100,
                                                          MAKENAME(second_aligned_source_word));

      for (uint j=0; j < curJ; j++) {
        const uint cur_aj = best_known_alignment_[s][j];
        aligned_source_words[cur_aj].insert(j);	
      }

      int cur_prev_cept = -100;
      for (uint i=0; i <= curI; i++) {

        assert(aligned_source_words[i].size() == fertility[i]);

        if (fertility[i] > 0) {
	  
          std::set<int>::iterator ait = aligned_source_words[i].begin();
          first_aligned_source_word[i] = *ait;

          int prev_j = *ait;
          if (fertility[i] > 1) {
            ait++;
            second_aligned_source_word[i] = *ait;
            prev_j = *ait;
          } 	    

          switch (cept_start_mode_) {
          case IBM4CENTER: {
	    double sum = 0.0;
	    for (std::set<int>::iterator ait = aligned_source_words[i].begin(); ait != aligned_source_words[i].end(); ait++) {
	      sum += *ait;
	    }
            cept_center[i] = (int) round(sum / fertility[i]);
            break;
	  }
          case IBM4FIRST:
            cept_center[i] = first_aligned_source_word[i];
            break;
          case IBM4LAST:
            cept_center[i] = prev_j;
            break;
          case IBM4UNIFORM:
	    cept_center[i] = first_aligned_source_word[i];
            break;
          default:
            assert(false);
          }

          prev_cept[i] = cur_prev_cept;
          cur_prev_cept = i;
        }
      }

      // 1. handle viterbi alignment
      for (uint i=1; i <= curI; i++) {

        const uint tclass = target_class_[ cur_target[i-1] ];

        const long double cur_prob = inv_sentence_prob * best_prob;
	
        if (fertility[i] > 0) {
          
          const int cur_prev_cept = prev_cept[i];
	  
          //a) update head prob
          if (cur_prev_cept >= 0) {

            const uint sclass = source_class_[ cur_source[cept_center[cur_prev_cept]]  ];

            int diff = first_aligned_source_word[i] - cept_center[cur_prev_cept];
            diff += displacement_offset_;

            fceptstart_count(sclass,tclass,diff) += cur_prob;

	    if (inter_distort_count[curJ](sclass,tclass).size() == 0)
	      sparse_inter_distort_count(sclass,tclass)[DistortCount(curJ,first_aligned_source_word[i],cept_center[cur_prev_cept])] += cur_prob;
	    else
	      inter_distort_count[curJ](sclass,tclass)(first_aligned_source_word[i],cept_center[cur_prev_cept]) += cur_prob;
          }
          else if (use_sentence_start_prob_) {
            fsentence_start_count[first_aligned_source_word[i]] += cur_prob;
            sentence_start_count[curJ][first_aligned_source_word[i]] += cur_prob;
          }

          //b) update within-cept prob
          int prev_aligned_j = first_aligned_source_word[i];
          std::set<int>::iterator ait = aligned_source_words[i].begin();
          ait++;
          for (;ait != aligned_source_words[i].end(); ait++) {

            const int cur_j = *ait;
            int diff = cur_j - prev_aligned_j;
            fwithincept_count(tclass,diff) += cur_prob;
            intra_distort_count[curJ](tclass,cur_j,prev_aligned_j) += cur_prob;

            prev_aligned_j = cur_j;
          }
        }
      }

      // 2. handle expansion moves
      NamedStorage1D<std::set<int> > exp_aligned_source_words(MAKENAME(exp_aligned_source_words));
      exp_aligned_source_words = aligned_source_words;

      for (uint exp_j=0; exp_j < curJ; exp_j++) {

        const uint cur_aj = best_known_alignment_[s][exp_j];

        for (uint exp_i=0; exp_i <= curI; exp_i++) {

          long double cur_prob = expansion_move_prob(exp_j,exp_i);

          if (cur_prob > best_prob * 1e-11) {

            cur_prob *= inv_sentence_prob;
	    
	    //modify
            exp_aligned_source_words[cur_aj].erase(exp_j);
            exp_aligned_source_words[exp_i].insert(exp_j);
	    
            int prev_center = -100;
	    
            for (uint i=1; i <= curI; i++) {
	    
              if (!exp_aligned_source_words[i].empty()) {

                const uint tclass = target_class_[cur_target[i-1]];
                
                double sum_j = 0;
                uint nAlignedWords = 0;

                std::set<int>::iterator ait = exp_aligned_source_words[i].begin();
                const int first_j = *ait;
                sum_j += *ait;
                nAlignedWords++;
		
                //collect counts for the head model
                if (prev_center >= 0) {
                  const uint sclass = source_class_[ cur_source[prev_center] ];

                  int diff =  first_j - prev_center;
                  diff += displacement_offset_;
                  fceptstart_count(sclass,tclass,diff) += cur_prob;

		  if (inter_distort_count[curJ](sclass,tclass).size() == 0)
		    sparse_inter_distort_count(sclass,tclass)[DistortCount(curJ,first_j,prev_center)] += cur_prob;
		  else
		    inter_distort_count[curJ](sclass,tclass)(first_j,prev_center) += cur_prob;
                }
                else if (use_sentence_start_prob_) {
                  fsentence_start_count[first_j] += cur_prob;
                  sentence_start_count[curJ][first_j] += cur_prob;
                }

                //collect counts for the within-cept model
                int prev_j = first_j;
                for (ait++; ait != exp_aligned_source_words[i].end(); ait++) {

                  const int cur_j = *ait;
                  sum_j += cur_j;
                  nAlignedWords++;

                  int diff = cur_j - prev_j;
                  fwithincept_count(tclass,diff) += cur_prob;
                  intra_distort_count[curJ](tclass,cur_j,prev_j) += cur_prob;

                  prev_j = cur_j;
                }

                //update prev_center
                switch (cept_start_mode_) {
                case IBM4CENTER:
                  prev_center = (int) round(sum_j / nAlignedWords);
                  break;
                case IBM4FIRST:
                  prev_center = first_j;
                  break;
                case IBM4LAST:
                  prev_center = prev_j;
                  break;
                case IBM4UNIFORM:
                  prev_center = (int) round(sum_j / nAlignedWords);
                  break;
                default:
                  assert(false);
                }
              }
	    }

	    //restore
            //exp_aligned_source_words[cur_aj].insert(exp_j);
            //exp_aligned_source_words[exp_i].erase(exp_j);
            exp_aligned_source_words[cur_aj] = aligned_source_words[cur_aj];
            exp_aligned_source_words[exp_i] = aligned_source_words[exp_i];

          }
        }
      }
      
      //3. handle swap moves
      NamedStorage1D<std::set<int> > swap_aligned_source_words(MAKENAME(swap_aligned_source_words));
      swap_aligned_source_words = aligned_source_words;

      for (uint swap_j1 = 0; swap_j1 < curJ; swap_j1++) {

        const uint aj1 = best_known_alignment_[s][swap_j1];

        for (uint swap_j2 = 0; swap_j2 < curJ; swap_j2++) {
	  
          long double cur_prob = swap_move_prob(swap_j1, swap_j2);

          if (cur_prob > best_prob * 1e-11) {

            cur_prob *= inv_sentence_prob;

            const uint aj2 = best_known_alignment_[s][swap_j2];

	    //modify
            swap_aligned_source_words[aj1].erase(swap_j1);
            swap_aligned_source_words[aj1].insert(swap_j2);
            swap_aligned_source_words[aj2].erase(swap_j2);
            swap_aligned_source_words[aj2].insert(swap_j1);

            int prev_center = -100;
	    
            for (uint i=1; i <= curI; i++) {
	    
              if (!swap_aligned_source_words[i].empty()) {

                const uint tclass = target_class_[cur_target[i-1]];

                double sum_j = 0;
                uint nAlignedWords = 0;

                std::set<int>::iterator ait = swap_aligned_source_words[i].begin();
                const int first_j = *ait;
                sum_j += *ait;
                nAlignedWords++;
		
                //collect counts for the head model
                if (prev_center >= 0) {

                  const uint sclass = source_class_[ cur_source[prev_center] ];

                  int diff =  first_j - prev_center;
                  diff += displacement_offset_;
                  fceptstart_count(sclass,tclass,diff) += cur_prob;

		  if (inter_distort_count[curJ](sclass,tclass).size() == 0)
		    sparse_inter_distort_count(sclass,tclass)[DistortCount(curJ,first_j,prev_center)] += cur_prob;
		  else
		    inter_distort_count[curJ](sclass,tclass)(first_j,prev_center) += cur_prob;
                }
                else if (use_sentence_start_prob_) {
                  fsentence_start_count[first_j] += cur_prob;
                  sentence_start_count[curJ][first_j] += cur_prob;
                }
		
                //collect counts for the within-cept model
                int prev_j = first_j;
                for (ait++; ait != swap_aligned_source_words[i].end(); ait++) {

                  const int cur_j = *ait;
                  sum_j += cur_j;
                  nAlignedWords++;

                  int diff = cur_j - prev_j;
                  fwithincept_count(tclass,diff) += cur_prob;
                  intra_distort_count[curJ](tclass,cur_j,prev_j) += cur_prob;

                  prev_j = cur_j;
                }

                //update prev_center
                switch (cept_start_mode_) {
                case IBM4CENTER:
                  prev_center = (int) round(sum_j / nAlignedWords);
                  break;
                case IBM4FIRST:
                  prev_center = first_j;
                  break;
                case IBM4LAST:
                  prev_center = prev_j;
                  break;
                case IBM4UNIFORM:
                  prev_center = (int) round(sum_j / nAlignedWords);
                  break;
                default:
                  assert(false);
                }
              }
	    }

	    //restore
            swap_aligned_source_words[aj1] = aligned_source_words[aj1];
	    swap_aligned_source_words[aj2] = aligned_source_words[aj2];
          }
        }
      }

      tCountCollectEnd = std::clock();
      countcollecttime += diff_seconds(tCountCollectEnd,tCountCollectStart);

    } //loop over sentences finished

    if (nSourceClasses_*nTargetClasses_ >= 10) {
      for (uint J=11; J < inter_distortion_prob_.size(); J++) {
        
        for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
          for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
            inter_distortion_prob_[J](x,y).resize(0,0);
      }
    }

    /***** update probability models from counts *******/

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = fzero_count / fsum;
      p_nonzero_ = fnonzero_count / fsum;
    }

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    assert(!isnan(p_zero_));
    assert(!isnan(p_nonzero_));

    //DEBUG
    uint nZeroAlignments = 0;
    uint nAlignments = 0;
    for (size_t s=0; s < source_sentence_.size(); s++) {

      nAlignments += source_sentence_[s].size();

      for (uint j=0; j < source_sentence_[s].size(); j++) {
        if (best_known_alignment_[s][j] == 0)
          nZeroAlignments++;
      }
    }
    std::cerr << "percentage of zero-aligned words: " 
              << (((double) nZeroAlignments) / ((double) nAlignments)) << std::endl;
    //END_DEBUG

    //update dictionary
    if (dict_weight_sum == 0.0) {
      for (uint i=0; i < nTargetWords; i++) {

        const double sum = fwcount[i].sum();
	
        if (sum > 1e-305) {
          double inv_sum = 1.0 / sum;
	  
          if (isnan(inv_sum)) {
            std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
            std::cerr << "sum = " << fwcount[i].sum() << std::endl;
            std::cerr << "number of cooccuring source words: " << fwcount[i].size() << std::endl;
          }
	  
          assert(!isnan(inv_sum));
	
          for (uint k=0; k < fwcount[i].size(); k++) {
            dict_[i][k] = fwcount[i][k] * inv_sum;
          }
        }
        else {
          //std::cerr << "WARNING: did not update dictionary entries because the sum was " << sum << std::endl;
        }
      }
    }
    else {


      for (uint i=0; i < nTargetWords; i++) {
	
        const double sum = fwcount[i].sum();
        const double prev_sum = dict_[i].sum();

        if (sum > 1e-305) {
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint k=0; k < fwcount[i].size(); k++) {
            dict_[i][k] = fwcount[i][k] * prev_sum * inv_sum;
          }
        }
      }

      double alpha = 100.0;
      if (iter > 2)
        alpha = 1.0;
      if (iter > 5)
        alpha = 0.1;

      dict_m_step(fwcount, prior_weight_, dict_, alpha, 45, smoothed_l0_, l0_beta_);
    }

    //update fertility probabilities
    for (uint i=1; i < nTargetWords; i++) {

      const double sum = ffert_count[i].sum();

      if (sum > 1e-305) {

        if (fertility_prob_[i].size() > 0) {
          assert(sum > 0.0);     
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint f=0; f < fertility_prob_[i].size(); f++)
            fertility_prob_[i][f] = inv_sum * ffert_count[i][f];
        }
        else {
          //std::cerr << "WARNING: target word #" << i << " does not occur" << std::endl;
        }
      }
      else {
        //std::cerr << "WARNING: did not update fertility count because sum was " << sum << std::endl;
      }
    }

    //update distortion probabilities

    //a) cept-start

    for (uint x=0; x < cept_start_prob_.xDim(); x++) {
      std::cerr << "calling inter-m-step(" << x << ",*)" << std::endl;
      
      for (uint y=0; y < cept_start_prob_.yDim(); y++) {
	
	double sum = 0.0;
	for (uint d=0; d < cept_start_prob_.zDim(); d++) 
	  sum += fceptstart_count(x,y,d);
	
	if (sum > 1e-305) {
	  
	  const double inv_sum = 1.0 / sum;
	  
	  if (!reduce_deficiency_) {
	    for (uint d=0; d < cept_start_prob_.zDim(); d++) 
	      cept_start_prob_(x,y,d) = std::max(1e-8,inv_sum * fceptstart_count(x,y,d));
	  }
	  else {
	    
	    IBM4CeptStartModel hyp_cept_start_prob = cept_start_prob_;
            
	    for (uint d=0; d < cept_start_prob_.zDim(); d++) 
	      hyp_cept_start_prob(x,y,d) = inv_sum * fceptstart_count(x,y,d);
	    
	    double cur_energy = inter_distortion_m_step_energy(inter_distort_count,sparse_inter_distort_count(x,y),
							       cept_start_prob_,x,y);
	    double hyp_energy = inter_distortion_m_step_energy(inter_distort_count,sparse_inter_distort_count(x,y),
							       hyp_cept_start_prob,x,y);
	    
	    if (hyp_energy < cur_energy)
	      cept_start_prob_ = hyp_cept_start_prob;
	  }
	}
	
	if (reduce_deficiency_) 
	  inter_distortion_m_step(inter_distort_count,sparse_inter_distort_count(x,y),x,y);
      }
    }

    par2nonpar_inter_distortion();
    
    //b) within-cept
    for (uint x=0; x < within_cept_prob_.xDim(); x++) {
      
      IBM4WithinCeptModel hyp_withincept_prob = within_cept_prob_;
      double sum = 0.0;
      for (uint d=0; d < within_cept_prob_.yDim(); d++)
	sum += fwithincept_count(x,d);
      
      if (sum > 1e-305) {
	
	const double inv_sum = 1.0 / sum;
        
	if (!reduce_deficiency_) {
	  for (uint d=0; d < within_cept_prob_.yDim(); d++)
	    within_cept_prob_(x,d) = std::max(inv_sum * fwithincept_count(x,d),1e-8);
	}
	else {
	  for (uint d=0; d < within_cept_prob_.yDim(); d++)
	    hyp_withincept_prob(x,d) = inv_sum * fwithincept_count(x,d);
	  
	  double cur_energy = intra_distortion_m_step_energy(intra_distort_count,within_cept_prob_,x);
	  double hyp_energy = intra_distortion_m_step_energy(intra_distort_count,hyp_withincept_prob,x);
          
	  if (hyp_energy < cur_energy)
	    within_cept_prob_ = hyp_withincept_prob; 
	}
      }
        
      std::cerr << "intra-m-step(" << x << ")" << std::endl;
      if (reduce_deficiency_)
	intra_distortion_m_step(intra_distort_count,x);
    }
    
    par2nonpar_intra_distortion();

    //c) sentence start prob
    if (use_sentence_start_prob_) {

      if (iter == 1) {
        fsentence_start_count *= 1.0 / fsentence_start_count.sum();
        sentence_start_parameters_ = fsentence_start_count;
      }

      start_prob_m_step(sentence_start_count);
      par2nonpar_start_prob();
    }

    double reg_term = 0.0;
    for (uint i=0; i < dict_.size(); i++)
      for (uint k=0; k < dict_[i].size(); k++) {
	if (smoothed_l0_)
	  reg_term += prior_weight_[i][k] * prob_penalty(dict_[i][k],l0_beta_);
	else
	  reg_term += prior_weight_[i][k] * dict_[i][k];
      }
    
    max_perplexity += reg_term;
    approx_sum_perplexity += reg_term;


    max_perplexity /= source_sentence_.size();
    approx_sum_perplexity /= source_sentence_.size();

    std::string transfer = (ibm3 != 0 && iter == 1) ? " (transfer) " : ""; 

    std::cerr << "IBM-4 max-perplex-energy in between iterations #" << (iter-1) << " and " << iter << transfer << ": "
              << max_perplexity << std::endl;
    std::cerr << "IBM-4 approx-sum-perplex-energy in between iterations #" << (iter-1) << " and " << iter << transfer << ": "
              << approx_sum_perplexity << std::endl;
    
    if (possible_ref_alignments_.size() > 0) {
      
      std::cerr << "#### IBM-4-AER in between iterations #" << (iter-1) << " and " << iter << ": " << AER() << std::endl;
      std::cerr << "#### IBM-4-fmeasure in between iterations #" << (iter-1) << " and " << iter << ": " << f_measure() << std::endl;
      std::cerr << "#### IBM-4-DAE/S in between iterations #" << (iter-1) << " and " << iter << ": " 
                << DAE_S() << std::endl;
    }

    std::cerr << (((double) sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" 
              << std::endl;     
  }

  std::cerr << "spent " << hillclimbtime << " seconds on IBM-4-hillclimbing" << std::endl;
  std::cerr << "spent " << countcollecttime << " seconds on IBM-4-distortion count collection" << std::endl;

}

void IBM4Trainer::train_viterbi(uint nIter, IBM3Trainer* ibm3) {


  std::cerr << "starting IBM-4 Viterbi-training without constraints" << std::endl;

  double max_perplexity = 0.0;

  IBM4CeptStartModel fceptstart_count(cept_start_prob_.xDim(),cept_start_prob_.yDim(),2*maxJ_-1,MAKENAME(fceptstart_count));
  IBM4WithinCeptModel fwithincept_count(within_cept_prob_.xDim(),within_cept_prob_.yDim(),MAKENAME(fwithincept_count));
  Math1D::NamedVector<double> fsentence_start_count(maxJ_,MAKENAME(fsentence_start_count));

  Storage1D<Storage2D<Math2D::Matrix<double> > > inter_distort_count(maxJ_+1);
  Storage1D<Math3D::Tensor<double> > intra_distort_count(maxJ_+1);

  Storage1D<Math1D::Vector<double> > sentence_start_count(maxJ_+1);

  for (uint J=1; J <= maxJ_; J++) {

    if (inter_distortion_prob_[J].size() > 0) {
      inter_distort_count[J].resize(inter_distortion_prob_[J].xDim(),inter_distortion_prob_[J].yDim());

      if (J <= 25) {
        for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
          for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
            inter_distort_count[J](x,y).resize(inter_distortion_prob_[J](x,y).xDim(),inter_distortion_prob_[J](x,y).yDim(),0.0);
      }
      intra_distort_count[J].resize(intra_distortion_prob_[J].xDim(),intra_distortion_prob_[J].yDim(),
                                    intra_distortion_prob_[J].zDim(),0.0);
    }
    
    sentence_start_count[J].resize(J,0);
  }

  const uint nTargetWords = dict_.size();

  NamedStorage1D<Math1D::Vector<double> > fwcount(nTargetWords,MAKENAME(fwcount));
  NamedStorage1D<Math1D::Vector<double> > ffert_count(nTargetWords,MAKENAME(ffert_count));

  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(dict_[i].size());
    ffert_count[i].resize_dirty(fertility_prob_[i].size());
  }

  long double fzero_count;
  long double fnonzero_count;

  for (uint iter=1; iter <= nIter; iter++) {

    std::cerr << "******* IBM-4 Viterbi-iteration #" << iter << std::endl;

    uint sum_iter = 0;

    fzero_count = 0.0;
    fnonzero_count = 0.0;

    fceptstart_count.set_constant(0.0);
    fwithincept_count.set_constant(0.0);
    fsentence_start_count.set_constant(0.0);

    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
      ffert_count[i].set_constant(0.0);
    }

    for (uint J=1; J <= maxJ_; J++) {
      if (inter_distort_count[J].size() > 0) {
        for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
          for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
            inter_distort_count[J](x,y).set_constant(0.0);
      }
      intra_distort_count[J].set_constant(0.0);
      sentence_start_count[J].set_constant(0.0);
    }

    Storage2D<std::map<DistortCount,double> > sparse_inter_distort_count(nSourceClasses_,nTargetClasses_);

    max_perplexity = 0.0;

    for (size_t s=0; s < source_sentence_.size(); s++) {

      //DEBUG
      uint prev_sum_iter = sum_iter;
      //END_DEBUG

      if ((s% 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      if (nSourceClasses_*nTargetClasses_ >= 10 && (s%25) == 0) {
        for (uint J=11; J < inter_distortion_prob_.size(); J++) {

          for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
            for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
              inter_distortion_prob_[J](x,y).resize(0,0);
        }
      }
      
      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const Math2D::Matrix<uint>& cur_lookup = slookup_[s];
      
      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      
      Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));

      Math2D::NamedMatrix<long double> swap_move_prob(curJ,curJ,MAKENAME(swap_move_prob));
      Math2D::NamedMatrix<long double> expansion_move_prob(curJ,curI+1,MAKENAME(expansion_move_prob));

      //std::clock_t tHillclimbStart, tHillclimbEnd;
      //tHillclimbStart = std::clock();


      long double best_prob = 0.0;

      if (ibm3 != 0 && iter == 1) {
	best_prob = ibm3->update_alignment_by_hillclimbing(cur_source,cur_target,cur_lookup,sum_iter,fertility,
							   expansion_move_prob,swap_move_prob,best_known_alignment_[s]);	

	//DEBUG
	long double align_prob = alignment_prob(s,best_known_alignment_[s]);
      
	if (isinf(align_prob) || isnan(align_prob) || align_prob == 0.0) {
	  
	  std::cerr << "ERROR: after hillclimbing: align-prob for sentence " << s << " has prob " << align_prob << std::endl;
	  
	  print_alignment_prob_factors(source_sentence_[s], target_sentence_[s], slookup_[s], best_known_alignment_[s]);
	  
	  exit(1);
	}
	//END_DEBUG

      }
      else {
	best_prob = update_alignment_by_hillclimbing(cur_source,cur_target,cur_lookup,sum_iter,fertility,
						     expansion_move_prob,swap_move_prob,best_known_alignment_[s]);
      }

      max_perplexity -= std::log(best_prob);

      //DEBUG
      if (isinf(max_perplexity)) {

	std::cerr << "ERROR: inf after sentence  " << s << ", last alignment prob: " << best_prob << std::endl;
	std::cerr << "J = " << curJ << ", I = " << curI << std::endl;
	std::cerr << "preceding number of hillclimbing iterations: " << (sum_iter - prev_sum_iter) << std::endl;

	exit(1);
      }
      //END_DEBUG

      //tHillclimbEnd = std::clock();

      /**** update empty word counts *****/

      fzero_count += fertility[0];
      fnonzero_count += curJ - 2*fertility[0];

      /**** update fertility counts *****/
      for (uint i=1; i <= curI; i++) {

        const uint cur_fert = fertility[i];
        const uint t_idx = cur_target[i-1];

        ffert_count[t_idx][cur_fert] += 1.0;
      }

      /**** update dictionary counts *****/
      for (uint j=0; j < curJ; j++) {

        const uint s_idx = cur_source[j];
        const uint cur_aj = best_known_alignment_[s][j];

        if (cur_aj != 0) {
          fwcount[cur_target[cur_aj-1]][cur_lookup(j,cur_aj-1)] += 1.0;
        }
        else {
          fwcount[0][s_idx-1] += 1.0;
        }
      }

      /**** update distortion counts *****/
      NamedStorage1D<std::set<int> > aligned_source_words(curI+1,MAKENAME(aligned_source_words));
      Math1D::NamedVector<int> cept_center(curI+1,-100,MAKENAME(cept_center));

      //denotes the largest preceding target position that produces source words
      Math1D::NamedVector<int> prev_cept(curI+1,-100,MAKENAME(prev_cept));
      Math1D::NamedVector<int> first_aligned_source_word(curI+1,-100,
                                                         MAKENAME(first_aligned_source_word));
      Math1D::NamedVector<int> second_aligned_source_word(curI+1,-100,
                                                          MAKENAME(second_aligned_source_word));

      for (uint j=0; j < curJ; j++) {
        const uint cur_aj = best_known_alignment_[s][j];
        aligned_source_words[cur_aj].insert(j);	
      }

      int cur_prev_cept = -100;
      for (uint i=0; i <= curI; i++) {

        assert(aligned_source_words[i].size() == fertility[i]);

        if (fertility[i] > 0) {
	  
          std::set<int>::iterator ait = aligned_source_words[i].begin();
          first_aligned_source_word[i] = *ait;

          uint prev_j = *ait;
          if (fertility[i] > 1) {
            ait++;
            second_aligned_source_word[i] = *ait;
            prev_j = *ait;
          } 	    

          double sum = 0.0;
          for (std::set<int>::iterator ait = aligned_source_words[i].begin(); ait != aligned_source_words[i].end(); ait++) {
            sum += *ait;
          }

          switch (cept_start_mode_) {
          case IBM4CENTER:
            cept_center[i] = (uint) round(sum / fertility[i]);
            break;
          case IBM4FIRST:
            cept_center[i] = first_aligned_source_word[i];
            break;
          case IBM4LAST:
            cept_center[i] = prev_j;
            break;
          case IBM4UNIFORM:
            cept_center[i] = (uint) round(sum / fertility[i]);
            break;
          default:
            assert(false);
          }

          prev_cept[i] = cur_prev_cept;
          cur_prev_cept = i;
        }
      }

      // handle viterbi alignment
      for (uint i=1; i <= curI; i++) {

        const uint tclass = target_class_[ cur_target[i-1] ];

        if (fertility[i] > 0) {

          const int cur_prev_cept = prev_cept[i];
	  
          //a) update head prob
          if (cur_prev_cept >= 0) {

            const uint sclass = source_class_[ cur_source[cept_center[cur_prev_cept]]  ];

            int diff = first_aligned_source_word[i] - cept_center[cur_prev_cept];
            diff += displacement_offset_;

            fceptstart_count(sclass,tclass,diff) += 1.0;

	    if (inter_distort_count[curJ](sclass,tclass).size() == 0)
	      sparse_inter_distort_count(sclass,tclass)[DistortCount(curJ,first_aligned_source_word[i],cept_center[cur_prev_cept])] += 1.0;
	    else
	      inter_distort_count[curJ](sclass,tclass)(first_aligned_source_word[i],cept_center[cur_prev_cept]) += 1.0;
          }
          else if (use_sentence_start_prob_) {
            fsentence_start_count[first_aligned_source_word[i]] += 1.0;
            sentence_start_count[curJ][first_aligned_source_word[i]] += 1.0;
          }

          //b) update within-cept prob
          int prev_aligned_j = first_aligned_source_word[i];
          std::set<int>::iterator ait = aligned_source_words[i].begin();
          ait++;
          for (;ait != aligned_source_words[i].end(); ait++) {

            const int cur_j = *ait;
            int diff = cur_j - prev_aligned_j;
            fwithincept_count(tclass,diff) += 1.0;
            intra_distort_count[curJ](tclass,cur_j,prev_aligned_j) += 1.0;

            prev_aligned_j = cur_j;
          }
        }
      }
    } // loop over sentences finished

    /***** update probability models from counts *******/

    //update p_zero_ and p_nonzero_
    if (!fix_p0_) {
      double fsum = fzero_count + fnonzero_count;
      p_zero_ = fzero_count / fsum;
      p_nonzero_ = fnonzero_count / fsum;
    }

    std::cerr << "new p_zero: " << p_zero_ << std::endl;

    //DEBUG
    uint nZeroAlignments = 0;
    uint nAlignments = 0;
    for (size_t s=0; s < source_sentence_.size(); s++) {

      nAlignments += source_sentence_[s].size();

      for (uint j=0; j < source_sentence_[s].size(); j++) {
        if (best_known_alignment_[s][j] == 0)
          nZeroAlignments++;
      }
    }
    std::cerr << "percentage of zero-aligned words: " 
              << (((double) nZeroAlignments) / ((double) nAlignments)) << std::endl;
    //END_DEBUG

    //update dictionary
    for (uint i=0; i < nTargetWords; i++) {

      const double sum = fwcount[i].sum();
      
      if (sum > 1e-305) {
        double inv_sum = 1.0 / sum;
	
        if (isnan(inv_sum)) {
          std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
          std::cerr << "sum = " << fwcount[i].sum() << std::endl;
          std::cerr << "number of cooccuring source words: " << fwcount[i].size() << std::endl;
        }
	
        assert(!isnan(inv_sum));
	
        for (uint k=0; k < fwcount[i].size(); k++) {
          dict_[i][k] = fwcount[i][k] * inv_sum;

	  if (dict_[i][k] > 1e-8)
	    max_perplexity += prior_weight_[i][k];
        }
      }
      else {
        //std::cerr << "WARNING: did not update dictionary entries because the sum was " << sum << std::endl;
      }
    }

    //update fertility probabilities
    for (uint i=1; i < nTargetWords; i++) {

      const double sum = ffert_count[i].sum();

      if (sum > 1e-305) {

        if (fertility_prob_[i].size() > 0) {
          assert(sum > 0.0);     
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint f=0; f < fertility_prob_[i].size(); f++)
            fertility_prob_[i][f] = inv_sum * ffert_count[i][f];
        }
        else {
          //std::cerr << "WARNING: target word #" << i << " does not occur" << std::endl;
        }
      }
      else {
        //std::cerr << "WARNING: did not update fertility count because sum was " << sum << std::endl;
      }
    }

    //update distortion probabilities

    //a) cept-start
    for (uint x=0; x < cept_start_prob_.xDim(); x++) {
      for (uint y=0; y < cept_start_prob_.yDim(); y++) {

        double sum = 0.0;
        for (uint d=0; d < cept_start_prob_.zDim(); d++) 
          sum += fceptstart_count(x,y,d);
          
        if (sum > 1e-305) {
          
          const double inv_sum = 1.0 / sum;

          if (!reduce_deficiency_) {
            for (uint d=0; d < cept_start_prob_.zDim(); d++) 
              cept_start_prob_(x,y,d) = std::max(1e-8,inv_sum * fceptstart_count(x,y,d));
          }
          else {
            IBM4CeptStartModel hyp_cept_start_prob = cept_start_prob_;
            
            for (uint d=0; d < cept_start_prob_.zDim(); d++) 
              hyp_cept_start_prob(x,y,d) = inv_sum * fceptstart_count(x,y,d);
            
            double cur_energy = inter_distortion_m_step_energy(inter_distort_count,sparse_inter_distort_count(x,y),
                                                               cept_start_prob_,x,y);
            double hyp_energy = inter_distortion_m_step_energy(inter_distort_count,sparse_inter_distort_count(x,y),
                                                               hyp_cept_start_prob,x,y);
            
            if (hyp_energy < cur_energy)
              cept_start_prob_ = hyp_cept_start_prob;
          }
        }

        if (reduce_deficiency_) 
          inter_distortion_m_step(inter_distort_count,sparse_inter_distort_count(x,y),x,y);  
      }
    }

    par2nonpar_inter_distortion();
    
    //b) within-cept

    for (uint x=0; x < within_cept_prob_.xDim(); x++) {
      
      IBM4WithinCeptModel hyp_withincept_prob = within_cept_prob_;
      double sum = 0.0;
      for (uint d=0; d < within_cept_prob_.yDim(); d++)
        sum += fwithincept_count(x,d);
      
      if (sum > 1e-305) {
        
        const double inv_sum = 1.0 / sum;
        
        if (!reduce_deficiency_) {
          for (uint d=0; d < within_cept_prob_.yDim(); d++)
            within_cept_prob_(x,d) = std::max(inv_sum * fwithincept_count(x,d),1e-8);
        }
        else {
          for (uint d=0; d < within_cept_prob_.yDim(); d++)
            hyp_withincept_prob(x,d) = inv_sum * fwithincept_count(x,d);
          
          double cur_energy = intra_distortion_m_step_energy(intra_distort_count,within_cept_prob_,x);
          double hyp_energy = intra_distortion_m_step_energy(intra_distort_count,hyp_withincept_prob,x);
          
          if (hyp_energy < cur_energy)
            within_cept_prob_ = hyp_withincept_prob; 
        }
      }
        
      std::cerr << "intra-m-step(" << x << ")" << std::endl;
      if (reduce_deficiency_)
        intra_distortion_m_step(intra_distort_count,x);
    }

    par2nonpar_intra_distortion();


    //c) sentence start prob
    if (use_sentence_start_prob_) {

      if (iter == 1) {
        fsentence_start_count *= 1.0 / fsentence_start_count.sum();
        sentence_start_parameters_ = fsentence_start_count;
      }

      start_prob_m_step(sentence_start_count);
      par2nonpar_start_prob();
    }

    //DEBUG
    for (size_t s=0; s < source_sentence_.size(); s++) {
      
      long double align_prob = alignment_prob(s,best_known_alignment_[s]);
      
      if (isinf(align_prob) || isnan(align_prob) || align_prob == 0.0) {
	
	std::cerr << "ERROR: after parameter update: align-prob for sentence " << s << " has prob " << align_prob << std::endl;

	print_alignment_prob_factors(source_sentence_[s], target_sentence_[s], slookup_[s], best_known_alignment_[s]);

	exit(1);
      }
    }
    //END_DEBUG

    max_perplexity /= source_sentence_.size();

    //ICM STAGE
    std::cerr << "starting ICM" << std::endl;

    Math1D::NamedVector<uint> dict_sum(fwcount.size(),MAKENAME(dict_sum));
    for (uint k=0; k < fwcount.size(); k++)
      dict_sum[k] = fwcount[k].sum();

    uint nSwitches = 0;

    for (size_t s=0; s < source_sentence_.size(); s++) {

      if ((s% 10000) == 0)
        std::cerr << "sentence pair #" << s << std::endl;

      if (nSourceClasses_*nTargetClasses_ >= 10 && (s%25) == 0) {
        for (uint J=11; J < inter_distortion_prob_.size(); J++) {

          for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
            for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
              inter_distortion_prob_[J](x,y).resize(0,0);
        }
      }

      const Storage1D<uint>& cur_source = source_sentence_[s];
      const Storage1D<uint>& cur_target = target_sentence_[s];
      const Math2D::Matrix<uint>& cur_lookup = slookup_[s];
      
      const uint curI = cur_target.size();
      const uint curJ = cur_source.size();
      
      Math1D::NamedVector<uint> fertility(curI+1,0,MAKENAME(fertility));

      for (uint j=0; j < curJ; j++)
	fertility[best_known_alignment_[s][j]]++;

      double cur_distort_prob = distortion_prob(cur_source,cur_target,best_known_alignment_[s]);

      NamedStorage1D<std::vector<AlignBaseType> > hyp_aligned_source_words(curI+1,MAKENAME(hyp_aligned_source_words));

      for (uint j=0; j < curJ; j++) {

      	uint aj = best_known_alignment_[s][j];
      	hyp_aligned_source_words[aj].push_back(j);
      }


      for (uint j=0; j < curJ; j++) {

        for (uint i = 0; i <= curI; i++) {

          uint cur_aj = best_known_alignment_[s][j];
          uint cur_word = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];

          /**** dict ***/

          bool allowed = (cur_aj != i && (i != 0 || 2*fertility[0]+2 <= curJ));

	  if (i != 0 && (fertility[i]+1) > fertility_limit_)
	    allowed = false;

          if (allowed) {

            hyp_aligned_source_words[cur_aj].erase(std::find(hyp_aligned_source_words[cur_aj].begin(),
                                                             hyp_aligned_source_words[cur_aj].end(),j));
            hyp_aligned_source_words[i].push_back(j);
            std::sort(hyp_aligned_source_words[i].begin(),hyp_aligned_source_words[i].end());

            uint new_target_word = (i == 0) ? 0 : cur_target[i-1];

            double change = 0.0;

            Math1D::Vector<double>& cur_dictcount = fwcount[cur_word]; 
            Math1D::Vector<double>& hyp_dictcount = fwcount[new_target_word]; 

	    uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);
	    
	    double cur_dictsum = dict_sum[cur_word];
	    
	    uint hyp_idx = (i == 0) ? cur_source[j]-1 : cur_lookup(j,i-1);
	    

	    if (cur_word != new_target_word) {

	      if (dict_sum[new_target_word] > 0)
		change -= double(dict_sum[new_target_word]) * std::log( dict_sum[new_target_word] );
	      change += double(dict_sum[new_target_word]+1.0) * std::log( dict_sum[new_target_word]+1.0 );
	      
	      if (fwcount[new_target_word][hyp_idx] > 0)
		change -= double(fwcount[new_target_word][hyp_idx]) * 
		  (-std::log(fwcount[new_target_word][hyp_idx]));
	      else
		change += prior_weight_[new_target_word][hyp_idx]; 
	      
	      change += double(fwcount[new_target_word][hyp_idx]+1) * 
		(-std::log(fwcount[new_target_word][hyp_idx]+1.0));
	      
	      change -= double(cur_dictsum) * std::log(cur_dictsum);
	      if (cur_dictsum > 1)
		change += double(cur_dictsum-1) * std::log(cur_dictsum-1.0);
	      
	      change -= - double(cur_dictcount[cur_idx]) * std::log(cur_dictcount[cur_idx]);
	      
	      if (cur_dictcount[cur_idx] > 1) {
		change += double(cur_dictcount[cur_idx]-1) * (-std::log(cur_dictcount[cur_idx]-1));
	      }
	      else
		change -= prior_weight_[cur_word][cur_idx];
	    
	      /***** fertilities (only affected if the old and new target word differ) ****/
	      
	      //note: currently not updating f_zero / f_nonzero
	      if (cur_aj == 0) {
		
		uint zero_fert = fertility[0];
		
		change -= - std::log(ldchoose(curJ-zero_fert,zero_fert));
		change -= -std::log(p_zero_);
		
		if (och_ney_empty_word_) {
		  change -= -std::log(((long double) zero_fert) / curJ);
		}
		
		uint new_zero_fert = zero_fert-1;
		change += - std::log(ldchoose(curJ-new_zero_fert,new_zero_fert));
		change += 2.0*(-std::log(p_nonzero_));
	      }
	      else {

		if (!no_factorial_)
		  change -= - std::log(fertility[cur_aj]);
		

		double c = ffert_count[cur_word][fertility[cur_aj]];
		change -= -c * std::log(c);
		if (c > 1)
		  change += -(c-1) * std::log(c-1);
		
		double c2 = ffert_count[cur_word][fertility[cur_aj]-1];
		
		if (c2 > 0)
		  change -= -c2 * std::log(c2);
		change += -(c2+1) * std::log(c2+1);
	      }
	    }

	    if (i == 0) {

	      uint zero_fert = fertility[0];

	      change -= -std::log(ldchoose(curJ-zero_fert,zero_fert));
	      change -= 2.0*(-std::log(p_nonzero_));
	      
	      uint new_zero_fert = zero_fert+1;
	      change += - std::log(ldchoose(curJ-new_zero_fert,new_zero_fert));
	      change += -std::log(p_zero_);
	      
	      if (och_ney_empty_word_) {
		change += -std::log(((long double) new_zero_fert) / curJ);
	      }
	    }
	    else {

	      if (!no_factorial_)
		change -= - std::log(fertility[i]+1);
		
	      double c = ffert_count[new_target_word][fertility[i]];
	      change -= -c * std::log(c);
	      if (c > 1)
		change += -(c-1) * std::log(c-1);
	      else
		change -= l0_fertpen_;
	      
	      double c2 = ffert_count[new_target_word][fertility[i]+1];
	      if (c2 > 0)
		change -= -c2 * std::log(c2);
	      else
		change += l0_fertpen_;
	      change += -(c2+1) * std::log(c2+1);
	    }
	    
	    /***** distortion ****/
	    change -= - std::log(cur_distort_prob);

	    const long double hyp_distort_prob = distortion_prob(cur_source,cur_target,hyp_aligned_source_words);
	    change += - std::log(hyp_distort_prob);
	    
	    if (change < -0.01) {

	      best_known_alignment_[s][j] = i;

	      nSwitches++;
	      
	      uint cur_idx = (cur_aj == 0) ? cur_source[j]-1 : cur_lookup(j,cur_aj-1);
	      
	      uint hyp_idx = (i == 0) ? cur_source[j]-1 : cur_lookup(j,i-1);
	      
	      //dict
	      cur_dictcount[cur_idx] -= 1;
	      hyp_dictcount[hyp_idx] += 1;
	      dict_sum[cur_word] -= 1;
	      dict_sum[new_target_word] += 1;

	      //fert
	      if (cur_word != 0) {
		uint prev_fert = fertility[cur_aj];
		assert(prev_fert != 0);
		ffert_count[cur_word][prev_fert] -= 1;
		ffert_count[cur_word][prev_fert-1] += 1;
	      }
	      if (new_target_word != 0) {
		uint prev_fert = fertility[i];
		ffert_count[new_target_word][prev_fert] -= 1;
		ffert_count[new_target_word][prev_fert+1] += 1;
	      }

	      fertility[cur_aj]--;
	      fertility[i]++;
	      
	      cur_distort_prob = hyp_distort_prob; //distortion_prob(cur_source,cur_target,best_known_alignment_[s]);
	    }
	    else {

              hyp_aligned_source_words[i].erase(std::find(hyp_aligned_source_words[i].begin(),
                                                          hyp_aligned_source_words[i].end(),j));
              hyp_aligned_source_words[cur_aj].push_back(j);
              std::sort(hyp_aligned_source_words[cur_aj].begin(),hyp_aligned_source_words[cur_aj].end());
	    }
          }
	}	

      }
    } //loop over sentences finished
    std::cerr << nSwitches << " changes in ICM stage" << std::endl;

    if (nSourceClasses_*nTargetClasses_ >= 10) {
      for (uint J=11; J < inter_distortion_prob_.size(); J++) {
        
        for (uint y=0; y < inter_distortion_prob_[J].yDim(); y++)
          for (uint x=0; x < inter_distortion_prob_[J].xDim(); x++)
            inter_distortion_prob_[J](x,y).resize(0,0);
      }
    }


    //update dictionary
    for (uint i=0; i < nTargetWords; i++) {

      const double sum = fwcount[i].sum();
      
      if (sum > 1e-305) {
        double inv_sum = 1.0 / sum;
	
        if (isnan(inv_sum)) {
          std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
          std::cerr << "sum = " << fwcount[i].sum() << std::endl;
          std::cerr << "number of cooccuring source words: " << fwcount[i].size() << std::endl;
        }
	
        assert(!isnan(inv_sum));
	
        for (uint k=0; k < fwcount[i].size(); k++) {
          dict_[i][k] = fwcount[i][k] * inv_sum;
        }
      }
      else {
        //std::cerr << "WARNING: did not update dictionary entries because the sum was " << sum << std::endl;
      }
    }

    //update fertility probabilities
    for (uint i=1; i < nTargetWords; i++) {

      const double sum = ffert_count[i].sum();

      if (sum > 1e-305) {

        if (fertility_prob_[i].size() > 0) {
          assert(sum > 0.0);     
          const double inv_sum = 1.0 / sum;
          assert(!isnan(inv_sum));
	  
          for (uint f=0; f < fertility_prob_[i].size(); f++)
            fertility_prob_[i][f] = inv_sum * ffert_count[i][f];
        }
        else {
          //std::cerr << "WARNING: target word #" << i << " does not occur" << std::endl;
        }
      }
      else {
        //std::cerr << "WARNING: did not update fertility count because sum was " << sum << std::endl;
      }
    }

    //TODO: think about updating distortion here as well (will have to recollect counts from the best known alignments)

    std::string transfer = (ibm3 != 0 && iter == 1) ? " (transfer) " : ""; 

    std::cerr << "IBM-4 max-perplex-energy in between iterations #" << (iter-1) << " and " << iter << transfer << ": "
              << max_perplexity << std::endl;
    if (possible_ref_alignments_.size() > 0) {
      
      std::cerr << "#### IBM-4-AER in between iterations #" << (iter-1) << " and " << iter << transfer << ": " << AER() << std::endl;
      std::cerr << "#### IBM-4-fmeasure in between iterations #" << (iter-1) << " and " << iter << transfer << ": " << f_measure() << std::endl;
      std::cerr << "#### IBM-4-DAE/S in between iterations #" << (iter-1) << " and " << iter << transfer << ": " 
                << DAE_S() << std::endl;
    }

    std::cerr << (((double) sum_iter) / source_sentence_.size()) << " average hillclimbing iterations per sentence pair" 
              << std::endl;     

  }
}


void IBM4Trainer::write_postdec_alignments(const std::string filename, double thresh) {

  std::ostream* out;

#ifdef HAS_GZSTREAM
  if (string_ends_with(filename,".gz")) {
    out = new ogzstream(filename.c_str());
  }
  else {
    out = new std::ofstream(filename.c_str());
  }
#else
  out = new std::ofstream(filename.c_str());
#endif


  for (uint s=0; s < source_sentence_.size(); s++) {
    
    Math1D::Vector<AlignBaseType> viterbi_alignment = best_known_alignment_[s];
    std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;
  
    compute_external_postdec_alignment(source_sentence_[s], target_sentence_[s], slookup_[s],
				       viterbi_alignment, postdec_alignment, thresh);

    for(std::set<std::pair<AlignBaseType,AlignBaseType> >::iterator it = postdec_alignment.begin(); 
	it != postdec_alignment.end(); it++) {
      
      (*out) << (it->second-1) << " " << (it->first-1) << " ";
    }
    (*out) << std::endl;

  }
}
