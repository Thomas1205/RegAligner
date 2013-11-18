/** author: Thomas Schoenemann. This file was generated from singleword_fertility_training while 
    Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012. It was subsequently 
    modified and extended, both at the University of Düsseldorf and in his free time. ***/

#ifndef IBM4_TRAINING_HH
#define IBM4_TRAINING_HH

#include "singleword_fertility_training.hh"

struct DistortCount {

  DistortCount(uchar J, uchar j, uchar j_prev);

  uchar J_;
  uchar j_;
  uchar j_prev_;
};

struct IBM4CacheStruct {

  IBM4CacheStruct(uchar j, WordClassType sc, WordClassType tc);

  uchar j_;
  WordClassType sclass_;
  WordClassType tclass_;
};

bool operator<(const IBM4CacheStruct& c1, const IBM4CacheStruct& c2);

class IBM4Trainer : public FertilityModelTrainer {
public: 

  IBM4Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
              const LookupTable& slookup,
              const Storage1D<Storage1D<uint> >& target_sentence,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
              SingleWordDictionary& dict,
              const CooccuringWordsType& wcooc,
              uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
              const Storage1D<WordClassType>& source_class,
              const Storage1D<WordClassType>& target_class,  
	      const Math1D::Vector<double>& log_table,
              bool och_ney_empty_word = false,
              bool use_sentence_start_prob = false,
              bool no_factorial = true, 
              bool reduce_deficiency = true,
              bool nondeficient = false,  
              IBM4CeptStartMode cept_start_mode = IBM4CENTER,
	      IBM4InterDistMode inter_dist_mode = IBM4InterDistModePrevious, 
	      IBM4IntraDistMode intra_dist_mode = IBM4IntraDistModeSource,
              bool smoothed_l0 = false, double l0_beta = 1.0, double l0_fertpen = 0.0);


  virtual std::string model_name() const;

  void init_from_ibm3(FertilityModelTrainer& fert_trainer, bool clear_ibm3 = true, 
		      bool count_collection = false, bool viterbi = false);

  //training without constraints on uncovered positions.
  //This is based on the EM-algorithm where the E-step uses heuristics
  void train_unconstrained(uint nIter, FertilityModelTrainer* fert_trainer = 0, HmmWrapper* wrapper = 0);

  //unconstrained Viterbi training
  void train_viterbi(uint nIter, FertilityModelTrainer* fert_trainer = 0, HmmWrapper* wrapper = 0);

  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
						       const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
						       Math2D::Matrix<long double>& expansion_prob,
						       Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment);


  const Math1D::Vector<double>& sentence_start_parameters() const;

protected:

  double inter_distortion_prob(int j, int j_prev, uint sclass, uint tclass, uint J) const;

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                             const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment);

  long double nondeficient_alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                          const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment);

  long double distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
			      const Math1D::Vector<AlignBaseType>& alignment);

  //NOTE: the vectors need to be sorted
  long double distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
			      const Storage1D<std::vector<AlignBaseType> >& aligned_source_words);

  //NOTE: the vectors need to be sorted
  long double nondeficient_distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                           const Storage1D<std::vector<AlignBaseType> >& aligned_source_words);

  void print_alignment_prob_factors(const Storage1D<uint>& source, const Storage1D<uint>& target, 
				    const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment);

  long double alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment);

  long double nondeficient_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                        const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
                                        Math2D::Matrix<long double>& expansion_prob,
                                        Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment);


  void par2nonpar_inter_distortion();

  void par2nonpar_inter_distortion(int J, uint sclass, uint tclass);

  void par2nonpar_intra_distortion();

  double inter_distortion_m_step_energy(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                                        const std::map<DistortCount,double>& sparse_inter_distort_count,
                                        const Math3D::Tensor<double>& inter_param, uint class1, uint class2);

  double inter_distortion_m_step_energy(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                                        const std::vector<std::pair<DistortCount,double> >& sparse_inter_distort_count,
                                        const Math3D::Tensor<double>& inter_param, uint class1, uint class2);


  double intra_distortion_m_step_energy(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                                        const Math2D::Matrix<double>& intra_param, uint word_class);

  void inter_distortion_m_step(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                               const std::map<DistortCount,double>& sparse_inter_distort_count,
                               uint class1, uint class2);

  void intra_distortion_m_step(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                               uint word_class);


  //compact form
  double nondeficient_inter_m_step_energy(const IBM4CeptStartModel& singleton_count,
					  const std::vector<Math1D::Vector<uchar,uchar> >& open_diff,
					  const std::vector<double>& weight,
                                          const IBM4CeptStartModel& param, uint sclass, uint tclass);

  //compact form with interpolation
  double nondeficient_inter_m_step_energy(const Math1D::Vector<double>& singleton_count,
					  const std::vector<double>& norm_weight,
					  const Math1D::Vector<double>& param1, const Math1D::Vector<double>& param2, 
					  const Math1D::Vector<double>& sum1, const Math1D::Vector<double>& sum2, 
					  double lambda);

  //compact form
  void nondeficient_inter_m_step_with_interpolation(const IBM4CeptStartModel& singleton_count,
						    const std::vector<Math1D::Vector<uchar,uchar> >& open_diff,
						    const std::vector<double>& weight,
						    uint sclass, uint tclass, double start_energy);

  //compact form
  double nondeficient_intra_m_step_energy(const IBM4WithinCeptModel& singleton_count,
					  const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count,
                                          const IBM4WithinCeptModel& param, uint sclass);

  //compact form
  void nondeficient_intra_m_step(const IBM4WithinCeptModel& singleton_count,
				 const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count, uint sclass);


  virtual void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
					  const SingleLookupTable& lookup,
					  Math1D::Vector<AlignBaseType>& alignment);


  //indexed by (target word class idx, source word class idx, displacement)
  IBM4CeptStartModel cept_start_prob_; //note: displacements of 0 are possible here (the center of a cept need not be an aligned word)
  //indexed by (source word class, displacement)
  IBM4WithinCeptModel within_cept_prob_; //note: displacements of 0 are impossible

  Math1D::NamedVector<double> sentence_start_parameters_;
  Storage1D<Math1D::Vector<double> > sentence_start_prob_;

  Storage1D<Storage2D<Math2D::Matrix<float,ushort> > > inter_distortion_prob_;
  Storage1D<Math3D::Tensor<float> > intra_distortion_prob_;

  mutable Storage1D<std::map<ushort, std::map<IBM4CacheStruct,float> > > inter_distortion_cache_;

  Storage1D<WordClassType> source_class_;
  Storage1D<WordClassType> target_class_;  

  int displacement_offset_;

  IBM4CeptStartMode cept_start_mode_;
  IBM4InterDistMode inter_dist_mode_;
  IBM4IntraDistMode intra_dist_mode_;

  bool use_sentence_start_prob_;
  bool reduce_deficiency_;

  uint nSourceClasses_;
  uint nTargetClasses_;

  bool nondeficient_;

  //if there are many word classes, inter distortion tables are only kept for J<=storage_limit_ and some very frequent ones
  const ushort storage_limit_; 
};

#endif
