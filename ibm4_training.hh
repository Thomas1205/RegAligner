/*** ported here from singleword_fertility_training ****/
/** author: Thomas Schoenemann. This file was generated while Thomas Schoenemann was with the University of Düsseldorf, Germany, 2012 ***/

#ifndef IBM4_TRAINING_HH
#define IBM4_TRAINING_HH

#include "ibm3_training.hh"

enum IBM4CeptStartMode { IBM4CENTER, IBM4FIRST, IBM4LAST, IBM4UNIFORM };

class IBM4Trainer : public FertilityModelTrainer {
public: 

  IBM4Trainer(const Storage1D<Storage1D<uint> >& source_sentence,
              const Storage1D<Math2D::Matrix<uint> >& slookup,
              const Storage1D<Storage1D<uint> >& target_sentence,
              const std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_ref_alignments,
              const std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_ref_alignments,
              SingleWordDictionary& dict,
              const CooccuringWordsType& wcooc,
              uint nSourceWords, uint nTargetWords,
              const floatSingleWordDictionary& prior_weight,
              const Storage1D<WordClassType>& source_class,
              const Storage1D<WordClassType>& target_class,  
              bool och_ney_empty_word = false,
              bool use_sentence_start_prob = false,
              bool no_factorial = true,
              IBM4CeptStartMode cept_start_mode = IBM4CENTER,
              bool smoothed_l0 = false, double l0_beta = 1.0, double l0_fertpen = 0.0);


  void init_from_ibm3(IBM3Trainer& ibm3trainer, bool clear_ibm3 = true, 
		      bool count_collection = false, bool viterbi = false);

  //training without constraints on maximal fertility or uncovered positions.
  //This is based on the EM-algorithm where the E-step uses heuristics
  void train_unconstrained(uint nIter, IBM3Trainer* ibm3 = 0);

  //unconstrained Viterbi training
  void train_viterbi(uint nIter, IBM3Trainer* ibm3 = 0);

  void update_alignments_unconstrained();

  void fix_p0(double p0);

  double compute_external_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
                                    const Math2D::Matrix<uint>& lookup,
                                    Math1D::Vector<ushort>& alignment);

  // <code> start_alignment </code> is used as initialization for hillclimbing and later modified
  // the extracted alignment is written to <code> postdec_alignment </code>
  void compute_external_postdec_alignment(const Storage1D<uint>& source, const Storage1D<uint>& target,
					  const Math2D::Matrix<uint>& lookup,
					  Math1D::Vector<ushort>& start_alignment,
					  std::set<std::pair<ushort,ushort> >& postdec_alignment,
					  double threshold = 0.25);


  void write_postdec_alignments(const std::string filename, double thresh);

protected:

  long double alignment_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                             const Math2D::Matrix<uint>& lookup, const Math1D::Vector<ushort>& alignment);

  long double distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
			      const Math1D::Vector<ushort>& alignment);

  //NOTE: the vectors need to be sorted
  long double distortion_prob(const Storage1D<uint>& source, const Storage1D<uint>& target, 
			      const Storage1D<std::vector<ushort> >& aligned_source_words);

  void print_alignment_prob_factors(const Storage1D<uint>& source, const Storage1D<uint>& target, 
				    const Math2D::Matrix<uint>& lookup, const Math1D::Vector<ushort>& alignment);

  long double alignment_prob(uint s, const Math1D::Vector<ushort>& alignment);

  long double update_alignment_by_hillclimbing(const Storage1D<uint>& source, const Storage1D<uint>& target, 
                                               const Math2D::Matrix<uint>& lookup, uint& nIter, Math1D::Vector<uint>& fertility,
                                               Math2D::Matrix<long double>& expansion_prob,
                                               Math2D::Matrix<long double>& swap_prob, Math1D::Vector<ushort>& alignment);

  void par2nonpar_inter_distortion();

  void par2nonpar_inter_distortion(int J, uint sclass, uint tclass);

  void par2nonpar_intra_distortion();

  void par2nonpar_start_prob();

  double inter_distortion_m_step_energy(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                                        const Math3D::Tensor<double>& inter_param, uint class1, uint class2);

  double intra_distortion_m_step_energy(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                                        const Math2D::Matrix<double>& intra_param, uint word_class);

  void inter_distortion_m_step(const Storage1D<Storage2D<Math2D::Matrix<double> > >& inter_distort_count,
                               uint class1, uint class2);

  void intra_distortion_m_step(const Storage1D<Math3D::Tensor<double> >& intra_distort_count,
                               uint word_class);

  double start_prob_m_step_energy(const Storage1D<Math1D::Vector<double> >& start_count, Math1D::Vector<double>& param);

  void start_prob_m_step(const Storage1D<Math1D::Vector<double> >& start_count);


  //indexed by (target word class idx, source word class idx, displacement)
  IBM4CeptStartModel cept_start_prob_; //note: displacements of 0 are possible here (the center of a cept need not be an aligned word)
  //indexed by (source word class, displacement)
  IBM4WithinCeptModel within_cept_prob_; //note: displacements of 0 are impossible

  Math1D::NamedVector<double> sentence_start_parameters_;
  Storage1D<Math1D::Vector<double> > sentence_start_prob_;

  Storage1D<Storage2D<Math2D::Matrix<float> > > inter_distortion_prob_;
  Storage1D<Math3D::Tensor<float> > intra_distortion_prob_;

  Storage1D<WordClassType> source_class_;
  Storage1D<WordClassType> target_class_;  

  uint displacement_offset_;

  double p_zero_;
  double p_nonzero_;

  bool och_ney_empty_word_;
  IBM4CeptStartMode cept_start_mode_;
  bool use_sentence_start_prob_;
  bool no_factorial_;
  const floatSingleWordDictionary& prior_weight_;
  bool smoothed_l0_;
  double l0_beta_;
  double l0_fertpen_;

  bool fix_p0_;

  uint nSourceClasses_;
  uint nTargetClasses_;

};

#endif
