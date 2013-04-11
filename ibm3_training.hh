/** author: Thomas Schoenemann. This file was generated from singleword_fertility_training while 
    Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012. It was subsequently 
    modified and extended, both at the University of Düsseldorf and in his free time. ***/

#ifndef IBM3_TRAINING_HH
#define IBM3_TRAINING_HH


#include "singleword_fertility_training.hh"


class IBM3Trainer : public FertilityModelTrainer {
public:
  
  IBM3Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
              const LookupTable& slookup,
              const Storage1D<Storage1D<uint> >& target_sentence,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >& possible_ref_alignments,
              SingleWordDictionary& dict,
              const CooccuringWordsType& wcooc,
              uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
	      const Math1D::Vector<double>& log_table,
              bool parametric_distortion = false,
              bool och_ney_empty_word = true, 
              bool nondeficient = false, 
              bool viterbi_ilp = false, double l0_fertpen = 0.0,
              bool smoothed_l0 = false, double l0_beta = 1.0);


  virtual std::string model_name() const;

  void init_from_hmm(HmmWrapper& wrapper,
		     bool count_collection = false,
                     bool viterbi = false);

  //training without constraints on uncovered positions.
  //This is based on the EM-algorithm, where the E-step uses heuristics
  void train_unconstrained(uint nIter, HmmWrapper* wrapper = 0);

  //unconstrained Viterbi training
  void train_viterbi(uint nIter, HmmWrapper* wrapper = 0, bool use_ilp = false);

  //Viterbi-training with IBM reordering constraints. 
  void train_with_ibm_constraints(uint nIter, uint maxFertility, uint nMaxSkips = 4, bool verbose = false);

  //Viterbi-training with ITG reordering constraints. 
  void train_with_itg_constraints(uint nIter, bool extended_reordering = false, bool verbose = false);

  virtual long double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
						 const SingleLookupTable& lookup,
						 Math1D::Vector<AlignBaseType>& alignment);

  // <code> start_alignment </code> is used as initialization for hillclimbing and later modified
  // the extracted alignment is written to <code> postdec_alignment </code>
  virtual void compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
						  const SingleLookupTable& lookup,
						  Math1D::Vector<AlignBaseType>& start_alignment,
						  std::set<std::pair<AlignBaseType,AlignBaseType> >& postdec_alignment,
						  double threshold = 0.25);

  virtual long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
						       const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
						       Math2D::Matrix<long double>& expansion_prob,
						       Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment);

  
protected:


  void par2nonpar_distortion(ReducedIBM3DistortionModel& prob);

  double par_distortion_m_step_energy(const ReducedIBM3DistortionModel& fdistort_count,
                                      const Math2D::Matrix<double>& param, uint i);

  double par_distortion_m_step_energy(const ReducedIBM3DistortionModel& fdistort_count,
                                      const Math1D::Vector<double>& param, uint i);

  void par_distortion_m_step(const ReducedIBM3DistortionModel& fdistort_count, uint i, double start_energy);


  double nondeficient_m_step_energy(const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count,
                                    const Math2D::Matrix<double>& param, uint i);

  double nondeficient_m_step_energy(const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, 
                                    const std::vector<double>& count,
                                    const Math2D::Matrix<double>& param, uint i);

  double nondeficient_m_step_energy(const std::vector<double>& count, const Storage1D<uchar>& filled_pos,
                                    const Math2D::Matrix<double>& param1, const Math2D::Matrix<double>& param2,
                                    const Math1D::Vector<double>& sum1, const Math1D::Vector<double>& sum2, 
                                    uint i, double lambda);
  
  double nondeficient_m_step_energy(const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count,
                                    const Storage1D<Math2D::Matrix<double> >& param, uint i, uint J);

  double nondeficient_m_step(const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count, uint i, double start_energy);

  double nondeficient_m_step_with_interpolation(const std::vector<Math1D::Vector<uchar,uchar> >& open_pos, 
                                                const std::vector<double>& count,
                                                uint i, double start_energy);

  double nondeficient_m_step(const std::vector<std::pair<Math1D::Vector<uchar,uchar>,double> >& count, uint i, uint J);

  long double alignment_prob(uint s, const Math1D::Vector<AlignBaseType>& alignment) const;

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                             const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment) const;

  long double nondeficient_alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                          const SingleLookupTable& lookup, const Math1D::Vector<AlignBaseType>& alignment) const;

  //this EXCLUDES the empty word
  long double nondeficient_distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                           const Storage1D<std::vector<AlignBaseType> >& aligned_source_words) const;

  long double nondeficient_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                        const SingleLookupTable& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
                                        Math2D::Matrix<long double>& expansion_prob,
                                        Math2D::Matrix<long double>& swap_prob, Math1D::Vector<AlignBaseType>& alignment);


  long double compute_itg_viterbi_alignment_noemptyword(uint s, bool extended_reordering = false);

  //@param time_limit: maximum amount of seconds spent in the ILP-solver.
  //          values <= 0 indicate that no time limit is set
  long double compute_viterbi_alignment_ilp(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                            const SingleLookupTable& lookup, uint max_fertility,
                                            Math1D::Vector<AlignBaseType>& alignment, double time_limit = -1.0);

  void itg_traceback(uint s, const NamedStorage1D<Math3D::NamedTensor<uint> >& trace, uint J, uint j, uint i, uint ii);

  long double compute_ibmconstrained_viterbi_alignment_noemptyword(uint s, uint maxFertility, uint nMaxSkips);


  void prepare_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
				  const SingleLookupTable& lookup,
				  Math1D::Vector<AlignBaseType>& alignment);

  ReducedIBM3DistortionModel distortion_prob_;
  Math2D::Matrix<double> distortion_param_;

  bool parametric_distortion_;
  bool viterbi_ilp_;


  bool nondeficient_;
  mutable Storage1D<std::map<Storage1D<bool>,double> > denom_cache_;
};

void add_nondef_count(const Storage1D<std::vector<uchar> >& aligned_source_words, uint i, uint J,
		      std::map<Math1D::Vector<uchar,uchar>,double>& count_map, double count);


#endif
