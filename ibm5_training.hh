/**** written by Thomas Schoenemann as a private person without employment, February 2013 ****/

#ifndef IBM5_TRAINING_HH
#define IBM5_TRAINING_HH

#include "ibm4_training.hh"

class IBM5Trainer : public FertilityModelTrainer {
public: 

  IBM5Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
              const LookupTable& slookup,
              const Storage1D<Storage1D<uint> >& target_sentence,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
              SingleWordDictionary& dict,
              const CooccuringWordsType& wcooc,
              uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
              const Storage1D<WordClassType>& source_class,
	      const Math1D::Vector<double>& log_table,
	      bool nonpar_distortion = false,
              IBM4CeptStartMode cept_start_mode = IBM4CENTER,
              bool smoothed_l0 = false, double l0_beta = 1.0, double l0_fertpen = 0.0);

  virtual std::string model_name() const;

  void init_from_ibm4(IBM4Trainer& ibm4, bool count_collection = false, bool viterbi = false);

  void train_unconstrained(uint nIter, FertilityModelTrainer* fert_trainer = 0, HmmWrapper* wrapper = 0);

  void train_viterbi(uint nIter, FertilityModelTrainer* fert_trainer = 0, HmmWrapper* wrapper = 0);


  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
						       const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
						       Math2D::Matrix<long double>& expansion_prob,
						       Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment);

protected:

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                             const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment);

  //NOTE: the vectors need to be sorted
  long double distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
			      const Storage1D<std::vector<AlignBaseType> >& aligned_source_words);

  void par2nonpar_inter_distortion();

  double inter_distortion_m_step_energy(uint sclass, const Storage1D<Math3D::Tensor<double> >& count, const Math2D::Matrix<double>& param);

  void inter_distortion_m_step(uint sclass, const Storage1D<Math3D::Tensor<double> >& count);


  void par2nonpar_intra_distortion();

  double intra_distortion_m_step_energy(uint sclass, const Storage1D<Math2D::Matrix<double> >& count, const Math2D::Matrix<double>& param);

  void intra_distortion_m_step(uint sclass, const Storage1D<Math2D::Matrix<double> >& count);

  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
					  const SingleLookupTable& lookup,
					  Math1D::Vector<AlignBaseType>& alignment);


  //dependence on the previous center (yDim) and source word classes (zDim)
  Storage1D<Math3D::Tensor<double> > inter_distortion_prob_; 
  Math2D::Matrix<double> inter_distortion_param_;   

  uint displacement_offset_;

  //dependence on source word classes (yDim)
  Storage1D<Math2D::Matrix<double> > intra_distortion_prob_; 
  Math2D::Matrix<double> intra_distortion_param_; 

  bool nonpar_distortion_;

  IBM4CeptStartMode cept_start_mode_;

  Storage1D<WordClassType> source_class_;

  uint nSourceClasses_;
};

#endif
