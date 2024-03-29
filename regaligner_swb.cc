/*** written by Thomas Schoenemann
 *** initially as a private person without employment, October 2009
 *** later continued as an employee of Lund University, Sweden, 2010 - March 2011
 *** and later as a private person taking a time off, ***
 *** in small parts at the University of Pisa, Italy, October 2011
 *** and at the University of Düsseldorf, Germany, 2012 ***/

#include "application.hh"
#include "corpusio.hh"
#include "ibm1_training.hh"
#include "ibm1p0_training.hh"
#include "ibm2_training.hh"
#include "hmm_training.hh"
#include "hmm_fert_interface.hh"
#include "ibm3_training.hh"
#include "ibm4_training.hh"
#include "ibm5_training.hh"
#include "fertility_hmm.hh"
#include "fertility_hmmcc.hh"
#include "timing.hh"
#include "training_common.hh"
#include "alignment_computation.hh"
#include "alignment_error_rate.hh"      //for read_reference_alignment
#include "stringprocessing.hh"

#include <fstream>

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

int main(int argc, char** argv)
{

  if (argc == 1 || strings_equal(argv[1], "-h")) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << "**************** Inputs *******************" << std::endl
              << " -s <file> : source file (coded as indices)" << std::endl
              << " -t <file> : target file (coded as indices)" << std::endl
              << " [-ds <file>] : additional source file (word indices) " << std::endl
              << " [-dt <file>] : additional target file (word indices) " << std::endl
              << " [-sclasses <file>] : source word classes (for IBM-4)" << std::endl
              << " [-tclasses <file>] : target word classes (for HMM and IBM-4)" << std::endl
              << " [-tfert-classes <file>] : target fertility classe (IBM-3/4/5)" << std::endl
              << " [-refa <file>] : file containing gold alignments (sure and possible)" << std::endl
              << " [-invert-biling-data] : switch source and target for prior dict and gold alignments" << std::endl
              << "**************** Outputs *******************" << std::endl
              << " [-o <file>] : the determined dictionary is written to this file" << std::endl
              << " -oa <file> : the determined alignment is written to this file" << std::endl
              << "**************** Main Options **********************" << std::endl
              << " [-method ( em | gd | viterbi )] : use EM, gradient descent or Viterbi training (default EM) " << std::endl
              << " [-dict-regularity <double>] : regularity weight for L0 or L1 regularization" << std::endl
              << " [-sparse-reg] : activate L0/1-regularity only for rarely occuring target words" << std::endl
              << " [-fertpen <double>]: regularity weight for fertilities in IBM-3/4/5" << std::endl
              << " [-prior-dict <file>] : file for index pairs that occur in a dictionary" << std::endl
              << " [-l0-beta <double>] : smoothing parameter for the L0-norm in EM-training" << std::endl
              << " [-ibm1-iter <uint>] : iterations for the IBM-1 model (default 5)" << std::endl
              << " [-ibm2-iter <uint>] : iterations for the IBM-2 model (default 0)" << std::endl
              << " [-hmm-iter <uint>]  : iterations for the HMM model (default 5)" << std:: endl
              << " [-ibm3-iter <uint>] : iterations for the IBM-3 model (default 5)" << std::endl
              << " [-ibm4-iter <uint>] : iterations for the IBM-4 model (default 5)" << std::endl
              << " [-ibm5-iter <uint>] : iterations for the IBM-5 model (default 0)" << std::endl
			  << " [-fert-hmm-iter <uint>] : iterations for the HMM model (default 0)" << std::endl
              << " [-dict-iter <uint>] : iterations for the dictionary m-step (with regularity terms, default 45)" << std::endl
              << " [-nondef-iter <uint>] : iterations for nondeficient m-steps of IBM-3/4" << std::endl
              << " [-start-param-iter <uint>] : iterations for start m-steps (HMM and IBM-4/5)" << std::endl
              << " [-main-param-iter <uint>] : iterations for most alignment/distortion m-steps (HMM and IBM-4/5)" << std::endl
              << " [-rare-threshold <uint>] : maximum number of occurences of a source or target word to classify as sparse, default: 3" << std::endl
              << " [-postdec-thresh <double>] : threshold for posterior decoding" << std::endl
              << " [-dont-print-energy] : do not print the energy (speeds up EM for IBM-1, IBM-2 and HMM)" << std::endl
              << " [-max-lookup <uint>] : only store lookup tables up to this size to save memory. Default: 65535" << std::endl
              << "************ Options for the IBM-1 **************************"  << std::endl
              << " [-ibm1-p0 <double>] : empty word prob for a variant of IBM-1 (default -1 = off)" << std::endl
              << " [-ibm1-method (main | em | gd | viterbi)]: special method for IBM-1, default: main=as the others" << std::endl
              << " [-dict-init (uni | cooc): default uniform start prob" << std::endl
              << "************ Options for IBM-2 only **************************"  << std::endl
              << " [-ibm2-p0 <double>] : empty word prob for the IBM-2 (default 0.02)" << std::endl
              << " [-ibm2-alignment] (pos | diff | nonpar) : parametric alignment model for IBM-2, default pos" << std::endl
              << "************ Options for HMM only *****************" << std::endl
              << " [-hmm-p0 <double>] : fix HMM p0" << std::endl
              << " [-hmm-alignment (fullpar | redpar | nonpar | nonpar2)] : default redpar as in [Vogel,Ney,Tillmann]" << std::endl
              << " [-hmm-init-type (par | nonpar | fix | fix2)] : default par" << std::endl
              << " [-hmm-start-empty-word] : HMM with extra empty word " << std::endl
              << " [-hmm-redpar-limit <uint>] : limit for redpar, default 5 as in [Vogel, Ney, Tillmann]" << std::endl
              << " [-hmm-gd-stepsize <double>] : stepsize for gradient descent" << std::endl
              << "************ Options affecting a mixed set of models ****************"
              << " [-transfer-mode (no | viterbi | posterior)] : how to init HMM and IBM-2 from previous, default: no" << std::endl
              << " [-deficient-h25] : introduce deficiency for IBM-2, HMM and IBM-5 by not dividing by the param sum" << std::endl
              << " [-no-h23-classes] : don't use word classes for HMM" << std::endl
              << " [-gd-stepsize <double>]: stepsize for m-steps" << std::endl
              << "************ Options affecting several (or all) fertility based models ***************" << std::endl
              << " [-p0 <double>] : fix probability for empty alignments for IBM-3/4/5" << std::endl
              << " [-count-collection] : collect counts from the previous model when initializing IBM-3/4/5 as in [Brown et al.]" << std::endl
              << " [-hillclimb-mode (reuse | restart | reinit)] : default reuse" << std::endl
              << " [-org-empty-word] : for IBM-3/4 use empty word as originally published" << std::endl
              << " [-nondeficient] : train nondeficient variants of IBM-3/4" << std::endl
              << " [-fert-limit <uint>] : fertility limit for IBM-3/4/5, default: 9" << std::endl
              << " [-rare-fert-limit <uint>] : fertility limit for rare words and IBM-3/4, default: 9" << std::endl
              << " [-ibm45-mode (first | center | last | uniform)] : (default first), center for IBM-4/5 as in [Brown et al.]." << std::endl
              << "             Defines the dependence of IBM4/5 on the previous cept" << std::endl
              << " [-ibm45-intra-dist-mode (source | target)] : with default source: intra word class dependency for IBM-4/5 as in [Brown et al.]" << std::endl
              << " [-ibm45-uniform-start-prob] : use uniform start prob for IBM-4/5 as in GIZA++" << std::endl
              << "************ Options for IBM-3 only *****************" << std::endl
              << " [-ibm3-distortion] (pos | diff | nonpar)] : parametric distortion model for IBM-3, default pos" << std::endl
              << " [-ibm3-extra-deficient] : don't renormalize parametric distortion for IBM-3" << std::endl
              << " [-constraint-mode (unconstrained | itg | ibm)] : mode for IBM-3" << std::endl
              << " [-itg-max-mid-dev <uint>] : default 8" << std::endl
              << " [-itg-ext-level <uint>] : level of extension beyond ITG, default 0 (no extensions)" << std::endl
              << " [-ibm-max-skip <uint>] : skips allowed in the IBM constraints, default 3" << std::endl
              << " [-ilp-mode (off | compute | center)] : compute Viterbi alignments via ILP" << std::endl
              << " [-utmost-ilp-precision] : go for highly exact ILP (slower)" << std::endl
              << "************ Options for IBM-4 only *****************" << std::endl
              << " [-ibm4-inter-dist-mode (previous | current)] : with default previous:  word class dependency as in [Brown et al.]" << std::endl
              << " [-ibm4-reduce-deficiency] : renormalize probabilities for IBM-4 to stay inside the sentence" << std::endl
              << " [-ibm4-deficient-null (intra | uniform)] : default intra, uniform as in [Och & Ney]" << std::endl
              << "************ Options for IBM-5 only *****************" << std::endl
              << " [-ibm5-distortion (nonpar | diff | pos) ] : nonparametric distortion for IBM-5" << std::endl
              << std::endl << std::endl;

    std::cerr << "this program estimates p(s|t). Parameters in []-brackets are optional." << std::endl;;

    exit(0);
  }

  const int nParams = 70;
  ParamDescr params[nParams] = {
    {"-s", mandInFilename, 0, ""}, {"-t", mandInFilename, 0, ""}, {"-ds", optInFilename, 0, ""}, {"-dt", optInFilename, 0, ""},
    {"-o", optOutFilename, 0, ""}, {"-oa", mandOutFilename, 0, ""},  {"-refa", optInFilename, 0, ""}, {"-invert-biling-data", flag, 0, ""},
    {"-dict-regularity", optWithValue, 1, "0.0"}, {"-hillclimb-mode", optWithValue, 1, "reuse"}, {"-sparse-reg", flag, 0, ""},
    {"-prior-dict", optInFilename, 0, ""}, {"-hmm-iter", optWithValue, 1, "5"}, {"-method", optWithValue, 1, "em"}, {"-ibm1-iter", optWithValue, 1, "5"},
    {"-ibm2-iter", optWithValue, 1, "0"}, {"-ibm3-iter", optWithValue, 1, "5"}, {"-ibm4-iter", optWithValue, 1, "5"}, {"-ibm5-iter", optWithValue, 1, "0"},
    {"-fertpen", optWithValue, 1, "0.0"}, {"-constraint-mode", optWithValue, 1, "unconstrained"},  {"-l0-beta", optWithValue, 1, "-1.0"},
    {"-ibm45-mode", optWithValue, 1, "first"}, {"-fert-limit", optWithValue, 1, "9"}, {"-postdec-thresh", optWithValue, 1, "-1.0"},
    {"-hmm-alignment", optWithValue, 1, "redpar"}, {"-p0", optWithValue, 1, "-1.0"}, {"-org-empty-word", flag, 0, ""}, {"-ibm3-distortion", optWithValue, 1, "pos"},
    {"-hmm-init-type", optWithValue, 1, "par"}, {"-dont-print-energy", flag, 0, ""}, {"-transfer-mode", optWithValue, 1, "no"},
    {"-dict-struct", optWithValue, 0, ""}, {"-ibm4-reduce-deficiency", flag, 0, ""}, {"-count-collection", flag, 0, ""},
    {"-sclasses", optInFilename, 0, ""}, {"-tclasses", optInFilename, 0, ""}, {"-tfert-classes", optInFilename, 0, ""}, {"-max-lookup", optWithValue, 1, "65535"},
    {"-ibm4-inter-dist-mode", optWithValue, 1, "previous"}, {"-ibm45-intra-dist-mode", optWithValue, 1, "source"}, {"-nondeficient", flag, 0, ""},
    {"-ilp-mode", optWithValue, 1, "off"}, {"-utmost-ilp-precision", flag, 0, ""}, {"-hmm-start-empty-word", flag, 0, ""}, {"-ibm3-extra-deficient", flag, 0, ""},
    {"-deficient-h25", flag, 0, ""},{"-ibm4-deficient-null", optWithValue, 1, "intra"}, {"-rare-fert-limit", optWithValue, 1, "3"},
    {"-ibm2-alignment", optWithValue, 1, "pos"}, {"-no-h23-classes", flag, 0, ""}, {"-itg-max-mid-dev",optWithValue,1,"8"},{"-itg-ext-level",optWithValue,1,"0"},
    {"-ibm-max-skip", optWithValue,1,"3"},{"-dict-iter",optWithValue,1,"200"}, {"-nondef-iter",optWithValue,1,"250"}, 
	{"-ibm5-distortion", optWithValue, 1, "pos"}, {"-ibm45-uniform-start-prob", flag, 0, ""}, {"-start-param-iter", optWithValue, 1, "250"}, 
	{"-main-param-iter", optWithValue, 1, "500"}, {"-ibm2-p0", optWithValue, 1, "-1.0"}, {"-ibm1-method", optWithValue, 1, "main"}, {"-rare-threshold", optWithValue, 1, "3"},
    {"-hmm-p0",optWithValue,0,""}, {"-ibm1-p0",optWithValue,1,"-1.0"}, {"-hmm-redpar-limit", optWithValue, 1, "5"},
    {"-dict-init", optWithValue, 1, "uni"}, {"-hmm-gd-stepsize", optWithValue, 1, "100000.0" }, {"-gd-stepsize",optWithValue,1,"100000.0"},
	{"-fert-hmm-iter", optWithValue, 1, "0"}
  };

  Application app(argc, argv, params, nParams);

  NamedStorage1D<Math1D::Vector<uint> > source_sentence(MAKENAME(source_sentence));
  NamedStorage1D<Math1D::Vector<uint> > target_sentence(MAKENAME(target_sentence));

  NamedStorage1D<Math1D::Vector<uint> > dev_source_sentence(MAKENAME(dev_source_sentence));
  NamedStorage1D<Math1D::Vector<uint> > dev_target_sentence(MAKENAME(dev_target_sentence));

  RefAlignmentStructure sure_ref_alignments;
  RefAlignmentStructure possible_ref_alignments;

  uint ibm1_iter = convert<uint>(app.getParam("-ibm1-iter"));
  uint ibm2_iter = convert<uint>(app.getParam("-ibm2-iter"));
  uint hmm_iter = convert<uint>(app.getParam("-hmm-iter"));

  uint ibm3_iter = convert<uint>(app.getParam("-ibm3-iter"));
  uint ibm4_iter = convert<uint>(app.getParam("-ibm4-iter"));
  uint ibm5_iter = convert<uint>(app.getParam("-ibm5-iter"));

  uint fert_hmm_iter = convert<uint>(app.getParam("-fert-hmm-iter"));

  bool collect_counts = app.is_set("-count-collection");

  HillclimbingMode hillclimb_mode = HillclimbingReuse;

  std::string hillclimb_string = downcase(app.getParam("-hillclimb-mode"));
  if (hillclimb_string == "restart")
    hillclimb_mode = HillclimbingRestart;
  else if (hillclimb_string == "reinit")
    hillclimb_mode = HillclimbingReinit;
  else if (hillclimb_string != "reuse") {
    USER_ERROR << "unknown hillclimbing mode \"" << hillclimb_string << "\". Exiting." << std::endl;
    exit(1);
  }

  std::string method = downcase(app.getParam("-method"));
  if (method != "em" && method != "gd" && method != "viterbi" && method != "l-bfgs") {
    USER_ERROR << "unknown method \"" << method << "\"" << std::endl;
    exit(1);
  }

  std::string ibm1_method = downcase(app.getParam("-ibm1-method"));
  if (ibm1_method != "main" && ibm1_method != "em" && ibm1_method != "gd" && ibm1_method != "viterbi" && ibm1_method != "l-bfgs") {
    USER_ERROR << "unknown ibm1-method \"" << ibm1_method << "\"" << std::endl;
    exit(1);
  }
  if (ibm1_method == "main")
    ibm1_method = method;

  double l0_fertpen = convert<double>(app.getParam("-fertpen"));
  if (method == "viterbi")
    l0_fertpen *= source_sentence.size();

  double l0_beta = convert<double>(app.getParam("-l0-beta"));
  bool em_l0 = (l0_beta > 0.0 && method != "viterbi");

  uint fert_limit = convert<uint>(app.getParam("-fert-limit"));
  uint rare_fert_limit = convert<uint> (app.getParam("-rare-fert-limit"));
  uint nMaxRareOccurences = convert<uint>(app.getParam("-rare-threshold"));

  const uint max_lookup = convert<uint>(app.getParam("-max-lookup"));
  double postdec_thresh = convert<double>(app.getParam("-postdec-thresh"));
  double fert_p0 = convert<double>(app.getParam("-p0"));
  double gd_stepsize = convert<double>(app.getParam("-gd-stepsize"));
  const uint dict_m_step_iter = convert<uint>(app.getParam("-dict-iter"));

  MStepSolveMode msolve_mode = MSSolvePGD;

  std::clock_t tStartRead, tEndRead;
  tStartRead = std::clock();

  if (app.getParam("-s") == app.getParam("-t")) {
    std::cerr << "WARNING: files for source and target sentences are identical!" << std::endl;
  }

  read_monolingual_corpus(app.getParam("-s"), source_sentence);
  read_monolingual_corpus(app.getParam("-t"), target_sentence);

  if (source_sentence.size() != target_sentence.size()) {
    std::cerr << "ERROR: source and target must have the same number of sentences! Exiting..." << std::endl;
    exit(1);
  }

  if (app.is_set("-refa")) {
    read_reference_alignment(app.getParam("-refa"), sure_ref_alignments, possible_ref_alignments,
                             app.is_set("-invert-biling-data"));

    if (!reference_consistent(possible_ref_alignments, source_sentence, target_sentence)) {
      std::cerr << "ERROR: inconsistent reference alignments given. Exiting..." << std::endl;
      exit(1);
    }
  }

  if (app.is_set("-ds") != app.is_set("-dt")) {
    std::cerr << "WARNING: you need to specify both -ds and -dt . Ignoring additional sentences" << std::endl;
  }

  bool dev_present = app.is_set("-ds") && app.is_set("-dt");

  if (dev_present) {

    if (app.getParam("-ds") == app.getParam("-dt")) {
      std::cerr << "WARNING: dev-files for source and target sentences are identical!" << std::endl;
    }

    if (app.getParam("-s") == app.getParam("-ds")) {
      std::cerr << "WARNING: same file for source part of main corpus and development corpus" << std::endl;
    }
    if (app.getParam("-t") == app.getParam("-dt")) {
      std::cerr << "WARNING: same file for target part of main corpus and development corpus" << std::endl;
    }

    read_monolingual_corpus(app.getParam("-ds"), dev_source_sentence);
    read_monolingual_corpus(app.getParam("-dt"), dev_target_sentence);
  }

  tEndRead = std::clock();
  std::cerr << "reading the corpus took " << diff_seconds(tEndRead, tStartRead) << " seconds." << std::endl;

  assert(source_sentence.size() == target_sentence.size());

  const size_t nSentences = source_sentence.size();

  uint maxI = 0;
  uint maxJ = 0;
  std::set<uint> allSeenIs;

  for (size_t s = 0; s < nSentences; s++) {

    uint curJ = source_sentence[s].size();
    uint curI = target_sentence[s].size();
    allSeenIs.insert(curI);

    maxI = std::max<uint>(maxI, curI);
    maxJ = std::max<uint>(maxJ, curJ);

    if (9 * curJ < curI || 9 * curI < curJ) {
      std::cerr << "WARNING: GIZA++ would ignore sentence pair #" << (s + 1) << ": J=" << curJ << ", I=" << curI << std::endl;
    }

    if (curJ == 0 || curI == 0) {
      USER_ERROR << "empty sentences are not allowed. Clean up your data!. Exiting." << std::endl;
      exit(1);
    }
  }

  if (maxJ > 254 || maxI > 254) {
    USER_ERROR << " maximum sentence length is 254. Clean up your data!. Exiting." << std::endl;
    exit(1);
  }

  uint maxAllI = maxI;
  for (size_t s = 0; s < dev_source_sentence.size(); s++) {
    const uint curI = dev_target_sentence[s].size();
    maxAllI = std::max<uint>(maxAllI, curI);
    allSeenIs.insert(curI);
  }

  uint nSourceWords = 0;
  uint nTargetWords = 0;

  for (size_t s = 0; s < nSentences; s++) {

    nSourceWords = std::max(nSourceWords, source_sentence[s].max() + 1);
    nTargetWords = std::max(nTargetWords, target_sentence[s].max() + 1);

    if (source_sentence[s].min() * target_sentence[s].min() == 0) {     //short for ||
      USER_ERROR << " index 0 is reserved for the empty word. Exiting.." << std::endl;
      exit(1);
    }
  }

  for (size_t s = 0; s < dev_source_sentence.size(); s++) {

    nSourceWords = std::max(nSourceWords, dev_source_sentence[s].max() + 1);
    nTargetWords = std::max(nTargetWords, dev_target_sentence[s].max() + 1);

    if (dev_source_sentence[s].min() * dev_target_sentence[s].min() == 0) {     //short for ||
      USER_ERROR << " index 0 is reserved for the empty word. Exiting.." << std::endl;
      exit(1);
    }
  
    if (dev_source_sentence[s].size() > maxJ || dev_target_sentence[s].size() > maxI) {
	  USER_ERROR << " dev sentences may not be longer than the longest training sentence. Exiting.." << std::endl;
	  exit(1);
	}
  }

  Math1D::Vector<WordClassType> source_class(nSourceWords, 0);
  Math1D::Vector<WordClassType> target_class(nTargetWords, 0);

  if (app.is_set("-sclasses"))
    read_word_classes(app.getParam("-sclasses"), source_class);
  if (app.is_set("-tclasses"))
    read_word_classes(app.getParam("-tclasses"), target_class);

  const uint nSourceClasses = source_class.max() + 1;
  const uint nTargetClasses = target_class.max() + 1;

  CooccuringWordsType wcooc(MAKENAME(wcooc));
  CooccuringLengthsType lcooc(MAKENAME(lcooc));
  SingleWordDictionary dict(MAKENAME(dict));

  Math1D::Vector<double> source_fert(2,0.5);

  ReducedIBM2ClassAlignmentModel reduced_ibm2align_model(MAKENAME(reduced_ibm2align_model));
  Math3D::NamedTensor<double> reduced_ibm2align_param(MAKENAME(reduced_ibm2align_param));
  Math1D::Vector<WordClassType> ibm2_sclass;

  FullHMMAlignmentModel hmmalign_model(MAKENAME(hmmalign_model));
  FullHMMAlignmentModelSingleClass hmmcalign_model(MAKENAME(hmmcalign_model));
  InitialAlignmentProbability initial_prob(MAKENAME(initial_prob));


  Math1D::Vector<double> hmm_init_params;
  Math1D::Vector<double> hmm_dist_params;
  double hmm_dist_grouping_param = -1.0;

  Math2D::Matrix<double> hmmc_dist_params;
  Math1D::Vector<double> hmmc_dist_grouping_param;

  Storage2D<Math1D::Vector<double> > hmmcc_dist_params(nSourceClasses, nTargetClasses);
  hmmcc_dist_params(0,0).resize(1,0.0);
  Math2D::Matrix<double> hmmcc_dist_grouping_param(nSourceClasses, nTargetClasses);

  Math1D::Vector<double> dev_hmm_init_params(maxAllI, 0.0);
  Math1D::Vector<double> dev_hmm_dist_params(std::max<uint>(2 * maxAllI - 1, 0), 0.0);
  FullHMMAlignmentModel dev_hmmalign_model(MAKENAME(dev_hmmalign_model));
  InitialAlignmentProbability dev_initial_prob(MAKENAME(dev_initial_prob));

  Math2D::Matrix<double> dev_hmmc_dist_params(std::max<uint>(2 * maxAllI - 1, 0), hmmc_dist_params.yDim(), 0.0);
  FullHMMAlignmentModelSingleClass dev_hmmcalign_model(MAKENAME(dev_hmmcalign_model));

  uint dev_zero_offset = maxAllI; //max_devI - 1;

  for (size_t s = 0; s < nSentences; s++) {

    const Math1D::Vector<uint>& source = source_sentence[s];
    const Math1D::Vector<uint>& target = target_sentence[s];

    for (uint j = 0; j < source.size(); j++) {
      for (uint i = 0; i < target.size(); i++) {
        hmmcc_dist_params(source_class[source[j]], target_class[target[i]]).resize(1,0.0);
      }
    }
  }

  for (size_t s = 0; s < dev_source_sentence.size(); s++) {
    const Math1D::Vector<uint>& source = dev_source_sentence[s];
    const Math1D::Vector<uint>& target = dev_target_sentence[s];

    for (uint j = 0; j < source.size(); j++) {
      for (uint i = 0; i < target.size(); i++) {
        hmmcc_dist_params(source_class[source[j]], target_class[target[i]]).resize(1,0.0);
      }
    }
  }

  uint nClassCooc = 0;
  for (uint sc = 0; sc < hmmcc_dist_params.xDim(); sc++) {
    for (uint tc = 0; tc < hmmcc_dist_params.yDim(); tc++) {
      if (hmmcc_dist_params(sc, tc).size() > 0)
        nClassCooc++;
    }
  }

  double dict_regularity = convert<double>(app.getParam("-dict-regularity"));

  if (method == "viterbi")
    dict_regularity *= source_sentence.size();

  std::string hmm_string = downcase(app.getParam("-hmm-alignment"));
  if (hmm_string != "redpar" && hmm_string != "fullpar"
      && hmm_string != "nonpar" && hmm_string != "nonpar2") {
    std::cerr << "ERROR: \"" << hmm_string << "\" is not a valid hmm type. Exiting." << std::endl;
    exit(1);
  }

  HmmAlignProbType hmm_align_mode = HmmAlignProbReducedpar;
  if (hmm_string == "fullpar")
    hmm_align_mode = HmmAlignProbFullpar;
  else if (hmm_string == "nonpar")
    hmm_align_mode = HmmAlignProbNonpar;
  else if (hmm_string == "nonpar2")
    hmm_align_mode = HmmAlignProbNonpar2;

  HmmInitProbType hmm_init_mode = HmmInitPar;
  if (method == "viterbi")
    hmm_init_mode = HmmInitFix2;

  std::string hmm_init_string = downcase(app.getParam("-hmm-init-type"));
  if (hmm_init_string != "par" && hmm_init_string != "nonpar" && hmm_init_string != "fix"
      && hmm_init_string != "fix2") {
    std::cerr << "ERROR: \"" << hmm_init_string << "\" is not a valid hmm init type. Exiting..." << std::endl;
    exit(1);;
  }
  if (hmm_init_string == "par")
    hmm_init_mode = HmmInitPar;
  else if (hmm_init_string == "nonpar")
    hmm_init_mode = HmmInitNonpar;
  else if (hmm_init_string == "fix")
    hmm_init_mode = HmmInitFix;
  else if (hmm_init_string == "fix2")
    hmm_init_mode = HmmInitFix2;

  std::cerr << "finding cooccuring words" << std::endl;
  bool read_in = false;
  if (app.is_set("-dict-struct")) {

    read_in = read_cooccuring_words_structure(app.getParam("-dict-struct"), nSourceWords, nTargetWords, wcooc);
  }
  if (!read_in) {
    find_cooccuring_words(source_sentence, target_sentence, dev_source_sentence,
                          dev_target_sentence, nSourceWords, nTargetWords, wcooc);
  }

  std::cerr << "generating lookup table" << std::endl;
  LookupTable slookup;
  generate_wordlookup(source_sentence, target_sentence, wcooc, nSourceWords, slookup, max_lookup);

  floatSingleWordDictionary prior_weight(nTargetWords, MAKENAME(prior_weight));
  Math1D::Vector<float> prior_t0_weight(nTargetWords,dict_regularity); //not used but needs to be passed

  std::set<PriorPair> known_pairs;
  if (app.is_set("-prior-dict"))
    read_prior_dict(app.getParam("-prior-dict"), known_pairs, app.is_set("-invert-biling-data"));

  // std::set<std::pair<uint,uint> > known_pairs;
  // if (app.is_set("-prior-dict"))
  // read_prior_dict(app.getParam("-prior-dict"), known_pairs, app.is_set("-invert-biling-data"));

  for (uint i = 0; i < nTargetWords; i++) {
    uint size = (i == 0) ? nSourceWords - 1 : wcooc[i].size();
    prior_weight[i].resize(size, 0.0);
  }

  if (known_pairs.size() > 0) {

    if (dict_regularity == 0) {
      std::cerr << "WARNING: prior dict given, but regularity weight is 0" << std::endl;
    }

    //uint nIgnored = set_prior_dict_weights(known_pairs, wcooc, prior_weight, dict_regularity);
    uint nIgnored = set_prior_dict_weights(known_pairs, wcooc, prior_weight, prior_t0_weight, dict_regularity);

    // for (uint i = 0; i < nTargetWords; i++)
    // prior_weight[i].set_constant(dict_regularity);

    // uint nIgnored = 0;

    // std::cerr << "processing read list" << std::endl;

    // for (std::set<std::pair<uint,uint> >::iterator it = known_pairs.begin(); it != known_pairs.end(); it++) {

    // uint tword = it->first;
    // uint sword = it->second;

    // if (tword >= wcooc.size()) {
    // std::cerr << "tword out of range: " << tword << std::endl;
    // }

    // if (tword == 0) {
    // prior_weight[0][sword-1] = 0.0;
    // }
    // else {
    // uint pos = std::lower_bound(wcooc[tword].direct_access(), wcooc[tword].direct_access() + wcooc[tword].size(), sword) - wcooc[tword].direct_access();

    // if (pos < wcooc[tword].size() && wcooc[tword][pos] == sword) {
    // prior_weight[tword][pos] = 0.0;
    // }
    // else {
    // nIgnored++;
    // //std::cerr << "WARNING: ignoring entry of prior dictionary" << std::endl;
    // }
    // }
    // }

    std::cerr << "ignored " << nIgnored << " entries of prior dictionary" << std::endl;
  }
  else {

    Math1D::Vector<double> distribution_weight(nTargetWords, dict_regularity);

    if (app.is_set("-sparse-reg")) {

      distribution_weight.set_constant(0.0);

      for (size_t s = 0; s < target_sentence.size(); s++) {

        for (uint i = 0; i < target_sentence[s].size(); i++) {
          distribution_weight[target_sentence[s][i]] += 1.0;
        }
      }

      const uint cutoff = nMaxRareOccurences;

      uint nSparse = 0;
      for (uint i = 0; i < nTargetWords; i++) {
        if (distribution_weight[i] >= cutoff + 1)
          distribution_weight[i] = 0.0;
        else {
          nSparse++;
          //std::cerr << "sparse word: " << distribution_weight[i] << std::endl;
          distribution_weight[i] = (cutoff + 1) - distribution_weight[i];
        }
      }
      distribution_weight[0] = 0.0;
      distribution_weight *= dict_regularity;
      std::cerr << "reg_sum: " << distribution_weight.sum() << std::endl;
      std::cerr << nSparse << " sparse words" << std::endl;
    }

    for (uint i = 0; i < nTargetWords; i++)
      prior_weight[i].set_constant(distribution_weight[i]);
  }

  AlignmentSetConstraints align_constraints;
  std::string constraint_mode = downcase(app.getParam("-constraint-mode"));
  if (constraint_mode == "itg")
    align_constraints.align_set_ = ITGAlignments;
  else if (constraint_mode == "ibm")
    align_constraints.align_set_ = IBMSkipAlignments;
  else if (constraint_mode != "unconstrained") {

    USER_ERROR << "unknown constraint mode: \"" << constraint_mode << "\". Exiting" << std::endl;
    exit(1);
  }
  align_constraints.itg_max_mid_dev_ = convert<uint>(app.getParam("-itg-max-mid-dev"));
  align_constraints.itg_extension_level_ = convert<uint>(app.getParam("-itg-ext-level"));
  align_constraints.nMaxSkips_ = convert<uint>(app.getParam("-ibm-max-skip"));

  std::string ibm2_alignment_string = downcase(app.getParam("-ibm2-alignment"));
  IBM23ParametricMode ibm2_align_mode = IBM23ParByPosition;
  if (ibm2_alignment_string == "diff" || ibm2_alignment_string == "diffpar")
    ibm2_align_mode = IBM23ParByDifference;
  else if (ibm2_alignment_string == "nonpar")
    ibm2_align_mode = IBM23Nonpar;
  else if (ibm2_alignment_string != "pos" && ibm2_alignment_string != "pospar") {
    USER_ERROR << " unknown ibm2-alignment: \"" << ibm2_alignment_string << "\". Exiting... " << std::endl;
    exit(1);
  }

  std::string ibm3_distortion_string = downcase(app.getParam("-ibm3-distortion"));
  IBM23ParametricMode ibm3_dist_mode = IBM23ParByPosition;
  if (ibm3_distortion_string == "diff" || ibm3_distortion_string == "diffpar")
    ibm3_dist_mode = IBM23ParByDifference;
  else if (ibm3_distortion_string == "nonpar")
    ibm3_dist_mode = IBM23Nonpar;
  else if (ibm3_distortion_string != "pos" && ibm3_distortion_string != "pospar") {
    USER_ERROR << " unknown ibm3-distortion: \"" << ibm3_distortion_string << "\" " << std::endl;
    exit(1);
  }

  std::string ilp_string = downcase(app.getParam("-ilp-mode"));
  IlpMode ilp_mode = IlpOff;
  if (ilp_string == "compute" || ilp_string == "computeonly" || ilp_string == "compute-only")
    ilp_mode = IlpComputeOnly;
  else if (ilp_string == "center")
    ilp_mode = IlpCenter;
  else if (ilp_string != "off") {
    USER_ERROR << " unknown ilp-mode \"" << ilp_string << "\" " << std::endl;
    exit(1);
  }

  IBM4CeptStartMode ibm4_cept_mode = IBM4FIRST;
  std::string ibm4_mode = downcase(app.getParam("-ibm45-mode"));
  if (ibm4_mode == "first")
    ibm4_cept_mode = IBM4FIRST;
  else if (ibm4_mode == "center")
    ibm4_cept_mode = IBM4CENTER;
  else if (ibm4_mode == "last")
    ibm4_cept_mode = IBM4LAST;
  else {
    USER_ERROR << "unknown ibm4 mode: \"" << ibm4_mode << "\"" << std::endl;
    exit(1);
  }

  std::string ibm4_inter_dist_string = downcase(app.getParam("-ibm4-inter-dist-mode"));
  std::string ibm4_intra_dist_string = downcase(app.getParam("-ibm45-intra-dist-mode"));

  IBM4InterDistMode ibm4_inter_dist_mode = IBM4InterDistModePrevious;
  IBM4IntraDistMode ibm4_intra_dist_mode = IBM4IntraDistModeSource;

  if (ibm4_inter_dist_string == "current")
    ibm4_inter_dist_mode = IBM4InterDistModeCurrent;
  else if (ibm4_inter_dist_string != "previous") {
    USER_ERROR << " unknown ibm4-inter-dist-mode: \"" << ibm4_inter_dist_string << "\" " << std::endl;
    exit(1);
  }

  if (ibm4_intra_dist_string == "target")
    ibm4_intra_dist_mode = IBM4IntraDistModeTarget;
  else if (ibm4_intra_dist_string != "source") {
    USER_ERROR << " unknown ibm4-intra-dist-mode: \"" << ibm4_intra_dist_string << "\" " << std::endl;
    exit(1);
  }

  FertNullModel empty_word_model = FertNullIntra;
  if (app.is_set("-org-empty-word"))
    empty_word_model = FertNullNondeficient;
  else {

    std::string ibm4nullstring = app.getParam("-ibm4-deficient-null");
    if (ibm4nullstring == "uniform" || ibm4nullstring == "plain")
      empty_word_model = FertNullOchNey;
    else if (ibm4nullstring != "intra") {
      USER_ERROR << "unknown ibm4-deficient-null mode: \"" << ibm4nullstring << "\". Exiting." << std::endl;
      exit(1);
    }
  }

  const uint main_m_step_iter = convert<uint>(app.getParam("-main-param-iter"));
  const uint start_m_step_iter = convert<uint>(app.getParam("-start-param-iter"));

  const double ibm1_p0 = convert<double>(app.getParam("-ibm1-p0"));

  IBM1Options ibm1_options(nSourceWords, nTargetWords, sure_ref_alignments, possible_ref_alignments);
  ibm1_options.nIterations_ = ibm1_iter;
  ibm1_options.smoothed_l0_ = em_l0;
  ibm1_options.l0_beta_ = l0_beta;
  ibm1_options.print_energy_ = !app.is_set("-dont-print-energy");
  ibm1_options.unconstrained_m_step_ = (msolve_mode != MSSolvePGD);
  ibm1_options.dict_m_step_iter_ = dict_m_step_iter;
  ibm1_options.p0_ = ibm1_p0;
  ibm1_options.gd_stepsize_ = gd_stepsize;

  std::string sDictInit = downcase(app.getParam("-dict-init"));
  if (sDictInit == "uni" || sDictInit == "uniform")
    ibm1_options.uniform_dict_init_ = true;
  else if (sDictInit == "cooc")
    ibm1_options.uniform_dict_init_ = false;
  else {
    USER_ERROR << " unknown dict-init: \"" << sDictInit << "\" " << std::endl;
    exit(1);
  }

  IBM2Options ibm2_options(nSourceWords, nTargetWords, sure_ref_alignments, possible_ref_alignments);
  ibm2_options.nIterations_ = ibm1_iter;
  ibm2_options.smoothed_l0_ = em_l0;
  ibm2_options.l0_beta_ = l0_beta;
  ibm2_options.print_energy_ = !app.is_set("-dont-print-energy");
  ibm2_options.unconstrained_m_step_ = (msolve_mode != MSSolvePGD);
  ibm2_options.ibm2_mode_ = ibm2_align_mode;
  ibm2_options.dict_m_step_iter_ = dict_m_step_iter;
  ibm2_options.deficient_ = app.is_set("-deficient-h25");
  ibm2_options.align_m_step_iter_ = main_m_step_iter;
  ibm2_options.p0_ = convert<double>(app.getParam("-ibm2-p0"));
  ibm2_options.ibm1_p0_ = ibm1_p0;
  ibm2_options.gd_stepsize_ = gd_stepsize;

  HmmOptions hmm_options(nSourceWords, nTargetWords, reduced_ibm2align_model, ibm2_sclass, sure_ref_alignments, possible_ref_alignments);
  hmm_options.nIterations_ = hmm_iter;
  hmm_options.align_type_ = hmm_align_mode;
  hmm_options.init_type_ = hmm_init_mode;
  hmm_options.smoothed_l0_ = em_l0;
  hmm_options.l0_beta_ = l0_beta;
  hmm_options.start_empty_word_ = app.is_set("-hmm-start-empty-word");
  hmm_options.deficient_ = app.is_set("-deficient-h25");
  hmm_options.print_energy_ = !app.is_set("-dont-print-energy");
  hmm_options.msolve_mode_ = msolve_mode;
  hmm_options.dict_m_step_iter_ = dict_m_step_iter;
  hmm_options.init_m_step_iter_ = start_m_step_iter;
  hmm_options.align_m_step_iter_ = main_m_step_iter;
  hmm_options.redpar_limit_ = std::min(maxI-3,convert<uint>(app.getParam("-hmm-redpar-limit")));
  hmm_options.ibm1_p0_ = ibm1_p0;
  hmm_options.gd_stepsize_ = convert<double>(app.getParam("-hmm-gd-stepsize"));

  if (app.is_set("-hmm-p0")) {
    double p0 = convert<double>(app.getParam("-hmm-p0"));
    if (p0 >= 0.0 && p0 < 1.0) {
      source_fert[0] = p0;
      source_fert[1] = 1.0 - p0;
      hmm_options.fix_p0_ = true;
    }
  }

  std::string ibm1_transfer_mode = downcase(app.getParam("-transfer-mode"));
  if (ibm1_transfer_mode != "no" && ibm1_transfer_mode != "viterbi" && ibm1_transfer_mode != "posterior") {
    std::cerr << "ERROR: unknown mode \"" << ibm1_transfer_mode
              << "\" for transfer from IBM-1 to HMM. Exiting." << std::endl;
    exit(1);
  }

  hmm_options.transfer_mode_ = TransferNo;
  if (ibm1_transfer_mode == "posterior") {
    ibm2_options.transfer_mode_ = TransferPosterior;
    hmm_options.transfer_mode_ = TransferPosterior;
  }
  else if (ibm1_transfer_mode == "viterbi") {
    ibm2_options.transfer_mode_ = TransferViterbi;
    hmm_options.transfer_mode_ = TransferViterbi;
  }

  FertModelOptions fert_options;
  fert_options.l0_fertpen_ = l0_fertpen;
  fert_options.cept_start_mode_ = ibm4_cept_mode;
  fert_options.inter_dist_mode_ = ibm4_inter_dist_mode;
  fert_options.intra_dist_mode_ = ibm4_intra_dist_mode;
  fert_options.viterbi_ilp_mode_ = ilp_mode;
  fert_options.utmost_ilp_precision_ = app.is_set("-utmost-ilp-precision");
  fert_options.msolve_mode_ = msolve_mode;
  fert_options.l0_beta_ = l0_beta;
  fert_options.p0_ = fert_p0;
  fert_options.par_mode_ = ibm3_dist_mode;
  fert_options.deficient_ = app.is_set("-deficient-h25");
  fert_options.nondeficient_ = app.is_set("-nondeficient");
  fert_options.reduce_deficiency_ = app.is_set("-ibm4-reduce-deficiency");
  
  //fert_options.ibm5_nonpar_distortion_ = app.is_set("-ibm5-nonpar-distortion");
  std::string sIBM5Distortion = downcase(app.getParam("-ibm5-distortion"));
  if (sIBM5Distortion == "nonpar")
	fert_options.ibm5_distortion_type_ = IBM23Nonpar;
  else if (sIBM5Distortion == "diffpar" || sIBM5Distortion == "diff")
	fert_options.ibm5_distortion_type_ = IBM23ParByDifference;
  else if (sIBM5Distortion == "pospar" || sIBM5Distortion == "pos")
	fert_options.ibm5_distortion_type_ = IBM23ParByPosition;
  else {
	std::cerr << "ERROR: unknonw ibm5-distortion: \"" << sIBM5Distortion << "\". Exiting." << std::endl;
	exit(1);
  }
  
  fert_options.ibm3_extra_deficient_ = app.is_set("-ibm3-extra-deficient");
  fert_options.empty_word_model_ = empty_word_model;
  fert_options.dict_m_step_iter_ = dict_m_step_iter;
  fert_options.nondef_dist34_m_step_iter_ = convert<uint>(app.getParam("-nondef-iter"));
  fert_options.uniform_sentence_start_prob_ = app.is_set("-ibm45-uniform-start-prob");
  fert_options.dist_m_step_iter_ = main_m_step_iter;
  fert_options.start_m_step_iter_ = start_m_step_iter;
  fert_options.gd_stepsize_ = gd_stepsize;

  Math1D::NamedVector<double> log_table(MAKENAME(log_table));
  Math1D::NamedVector<double> xlogx_table(MAKENAME(xlogx_table));
  if (method == "viterbi" || constraint_mode != "unconstrained") {

    size_t sumJ = 0;
    for (size_t s = 0; s < source_sentence.size(); s++)
      sumJ += source_sentence[s].size();

    log_table.resize(sumJ + 1);
    log_table[0] = 0.0;
    for (uint k = 1; k <= sumJ; k++)
      log_table[k] = std::log(k);

    xlogx_table.resize(sumJ + 1);
    xlogx_table[0] = 0.0;
    xlogx_table[1] = 0.0;       //log(1) = 0
    for (uint k = 2; k <= sumJ; k++)
      xlogx_table[k] = k * log_table[k];
  }

  /**** now start training *****/

  std::clock_t tStartTrain = std::clock();

  std::cerr << "starting IBM 1 training" << std::endl;

  /*** IBM-1 ***/

  if (ibm1_method == "em") {
    if (ibm1_p0 >= 0.0 && ibm1_p0 < 1.0)
      train_ibm1p0(source_sentence, slookup, target_sentence, wcooc, dict, prior_weight, ibm1_options);
    else
      train_ibm1(source_sentence, slookup, target_sentence, wcooc, dict, prior_weight, ibm1_options);
  }
  else if (ibm1_method == "gd") {
    if (ibm1_p0 >= 0.0 && ibm1_p0 < 1.0)
      train_ibm1p0_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, dict, prior_weight, ibm1_options);
    else
      train_ibm1_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, dict, prior_weight, ibm1_options);
  }
  else if (ibm1_method == "l-bfgs") {

    train_ibm1_lbfgs_stepcontrol(source_sentence, slookup, target_sentence, wcooc, dict, prior_weight, ibm1_options);
  }
  else {
    if (ibm1_p0 >= 0.0 && ibm1_p0 < 1.0)
      ibm1p0_viterbi_training(source_sentence, slookup, target_sentence, wcooc, dict, prior_weight, ibm1_options, xlogx_table);
    else
      ibm1_viterbi_training(source_sentence, slookup, target_sentence, wcooc, dict, prior_weight, ibm1_options, xlogx_table);
  }

  /*** IBM-2 ***/

  if (ibm2_iter > 0) {

    ibm2_sclass.resize(source_class.size(),0);
    if (!app.is_set("-no-h23-classes"))
      ibm2_sclass = source_class;

    find_cooccuring_lengths(source_sentence, target_sentence, lcooc);

    if (method == "em") {

      train_reduced_ibm2(source_sentence, slookup, target_sentence, wcooc, lcooc, reduced_ibm2align_model, reduced_ibm2align_param, source_fert,
                         dict, ibm2_sclass, ibm2_options, prior_weight);
    }
    else if (method == "gd") {

      train_reduced_ibm2_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, lcooc, reduced_ibm2align_model, reduced_ibm2align_param, source_fert,
                                        dict, ibm2_sclass, ibm2_options, prior_weight);
    }
    else {

      reduced_ibm2_viterbi_training(source_sentence, slookup, target_sentence, wcooc, lcooc, reduced_ibm2align_model, reduced_ibm2align_param,
                                    source_fert, dict, ibm2_sclass, ibm2_options, prior_weight, xlogx_table);
    }
  }

  /*** HMM ***/

  HmmWrapperNoClasses hmm_wrapper(hmmalign_model, dev_hmmalign_model, initial_prob, hmm_dist_params, hmm_dist_grouping_param, hmm_options);
  HmmWrapperWithTargetClasses hmmc_wrapper(hmmcalign_model, dev_hmmcalign_model, initial_prob, hmmc_dist_params, hmmc_dist_grouping_param, target_class, hmm_options);
  HmmWrapperDoubleClasses hmmcc_wrapper(hmmcc_dist_params, hmmcc_dist_grouping_param, source_fert, initial_prob, source_class, target_class,
                                        hmm_options, maxAllI - 1);

  HmmWrapperBase* used_hmm = 0;

  if (method == "em") {

    if (app.is_set("-no-h23-classes") || ((nSourceClasses == 1) && (nTargetClasses == 1))) {
      train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, hmmalign_model, hmm_dist_params,
                         hmm_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
      used_hmm = &hmm_wrapper;
    }
    else if (nSourceClasses == 1) {
      train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, target_class, hmmcalign_model, hmmc_dist_params,
                         hmmc_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
      used_hmm = &hmmc_wrapper;
    }
    else {
      train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, source_class, target_class, hmmcc_dist_params,
                         hmmcc_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
      used_hmm = &hmmcc_wrapper;
    }
  }
  else if (method == "gd" || method == "l-bfgs") {

    if (app.is_set("-no-h23-classes") || target_class.max() == 0) {
      train_extended_hmm_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, hmmalign_model, hmm_dist_params,
                                        hmm_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
      used_hmm = &hmm_wrapper;
    }
    else if (nSourceClasses == 1) {
      train_extended_hmm_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, target_class, hmmcalign_model, hmmc_dist_params,
                                        hmmc_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
      used_hmm = &hmmc_wrapper;
    }
	else {
	  TODO("gd for HMM-DoubleClass");
	}
  }
  else {

    if (app.is_set("-no-h23-classes") || ((nSourceClasses == 1) && (nTargetClasses == 1))) {
      viterbi_train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, hmmalign_model, hmm_dist_params,
                                 hmm_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, xlogx_table, maxAllI);

      used_hmm = &hmm_wrapper;
    }
    else if (nSourceClasses == 1) {
      viterbi_train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, target_class, hmmcalign_model, hmmc_dist_params,
                                 hmmc_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, xlogx_table, maxAllI);
      used_hmm = &hmmc_wrapper;
    }
    else {

	  if (hmm_options.align_type_ == HmmAlignProbNonpar || hmm_options.align_type_ == HmmAlignProbNonpar2) {
		std::cerr << "WARNING: nonpar selected, but not supported. Will be treated like redpar" << std::endl;
	    hmm_options.align_type_ == HmmAlignProbReducedpar;
	  }
	  
      viterbi_train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, source_class, target_class, hmmcc_dist_params,
                                 hmmcc_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, xlogx_table,
                                 maxAllI);

      used_hmm = &hmmcc_wrapper;
    }
  }

  //std::cerr << "AA" << std::endl;
  used_hmm->fill_dist_params(hmm_dist_params, hmm_dist_grouping_param);
  //std::cerr << "BB" << std::endl;
  used_hmm->fill_dist_params(nTargetClasses, hmmc_dist_params, hmmc_dist_grouping_param);
  //std::cerr << "CC" << std::endl;
  used_hmm->fill_dist_params(nSourceClasses, nTargetClasses, hmmcc_dist_params, hmmcc_dist_grouping_param);

  if (hmmcalign_model.size() == 0) {

    const uint nClasses = target_class.max() + 1;

    hmmcalign_model.resize(maxAllI);
    for (std::set<uint>::const_iterator it = allSeenIs.begin(); it != allSeenIs.end(); it++) {
      uint I = *it;
      hmmcalign_model[I-1].resize(I+1, I, nClasses);
    }

    par2nonpar_hmm_alignment_model(hmmc_dist_params, maxAllI-1, hmmc_dist_grouping_param, source_fert, hmm_options.align_type_,
                                   hmm_options.deficient_, hmm_options.redpar_limit_, hmmcalign_model);
  }
  if (hmmalign_model.size() == 0) {

    hmmalign_model.resize(maxAllI);
    for (std::set<uint>::const_iterator it = allSeenIs.begin(); it != allSeenIs.end(); it++) {
      uint I = *it;
      hmmalign_model[I-1].resize(I + 1, I);
    }

    par2nonpar_hmm_alignment_model(hmm_dist_params, maxAllI - 1, hmm_dist_grouping_param, source_fert, hmm_options.align_type_,
                                   hmm_options.deficient_, hmmalign_model, hmm_options.redpar_limit_);
  }

  std::cerr << "creating wrapper" << std::endl;



  // if (method == "em") {

  // if (app.is_set("-no-h23-classes") || target_class.max() == 0) {
  // train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, hmmalign_model, hmm_dist_params,
  // hmm_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
  // }
  // else {
  // train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, target_class, hmmcalign_model, hmmc_dist_params,
  // hmmc_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
  // }
  // }
  // else if (method == "gd" || method == "l-bfgs") {

  // if (app.is_set("-no-h23-classes") || target_class.max() == 0) {
  // train_extended_hmm_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, hmmalign_model, hmm_dist_params,
  // hmm_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict,
  // prior_weight, hmm_options, maxAllI);
  // }
  // else {
  // train_extended_hmm_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, target_class, hmmcalign_model, hmmc_dist_params,
  // hmmc_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, maxAllI);
  // }
  // }
  // else {

  // if (app.is_set("-no-h23-classes") || target_class.max() == 0) {
  // viterbi_train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, hmmalign_model, hmm_dist_params,
  // hmm_dist_grouping_param, source_fert, initial_prob, hmm_init_params, dict, prior_weight, hmm_options, xlogx_table, maxAllI);
  // }
  // else {
  // viterbi_train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, target_class, hmmcalign_model,
  // hmmc_dist_params, hmmc_dist_grouping_param, source_fert, initial_prob, hmm_init_params,
  // dict, prior_weight, hmm_options, xlogx_table, maxAllI);
  // }
  // }

  // if (hmmcalign_model.size() == 0) {

  // const uint nClasses = target_class.max() + 1;

  // hmmc_dist_params.resize(hmm_dist_params.size(), nClasses);
  // for (uint c = 0; c < nClasses; c++)
  // for (uint k = 0; k < hmm_dist_params.size(); k++)
  // hmmc_dist_params(k, c) = hmm_dist_params[k];
  // hmmc_dist_grouping_param.resize(nClasses);
  // for (uint c = 0; c < nClasses; c++)
  // hmmc_dist_grouping_param[c] = hmm_dist_grouping_param;

  // hmmcalign_model.resize(hmmalign_model.size());
  // for (uint I = 0; I < hmmalign_model.size(); I++) {

  // hmmcalign_model[I].resize(hmmalign_model[I].xDim(), hmmalign_model[I].yDim(), nClasses);
  // for (uint c = 0; c < nClasses; c++)
  // for (uint x = 0; x < hmmalign_model[I].xDim(); x++)
  // for (uint y = 0; y < hmmalign_model[I].yDim(); y++)
  // hmmcalign_model[I](x, y, c) = hmmalign_model[I](x, y);
  // }
  // }
  // if (hmmalign_model.size() == 0) {

  // hmm_dist_params.resize(hmmc_dist_params.xDim());
  // for (uint k = 0; k < hmm_dist_params.size(); k++) {
  // double sum = 0.0;
  // for (uint c = 0; c < hmmc_dist_params.yDim(); c++)
  // sum += hmmc_dist_params(k, c);
  // hmm_dist_params[k] = sum / hmmc_dist_params.yDim();
  // }
  // hmm_dist_grouping_param = hmmc_dist_grouping_param.sum() / hmmc_dist_grouping_param.size();
  // hmmalign_model.resize(hmmcalign_model.size());
  // for (uint I = 0; I < hmmalign_model.size(); I++) {
  // hmmalign_model[I].resize(hmmcalign_model[I].xDim(), hmmcalign_model[I].yDim());
  // for (uint x = 0; x < hmmalign_model[I].xDim(); x++) {
  // for (uint y = 0; y < hmmalign_model[I].yDim(); y++) {
  // double sum = 0.0;
  // for (uint c = 0; c < hmmcalign_model[I].zDim(); c++)
  // sum += hmmcalign_model[I] (x, y, c);
  // hmmalign_model[I] (x, y) = sum / hmmcalign_model[I].zDim();
  // }
  // }
  // }
  // }

  HmmFertInterfaceTargetClasses hmmc_interface(hmmc_wrapper, source_sentence, slookup, target_sentence, sure_ref_alignments,
      possible_ref_alignments, dict, wcooc, nSourceWords, nTargetWords, fert_limit);

  hmmc_interface.set_fertility_limit(fert_limit);
  if (rare_fert_limit < fert_limit)
    hmmc_interface.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  HmmFertInterfaceDoubleClasses hmmcc_interface(hmmcc_wrapper, source_sentence, slookup, target_sentence, sure_ref_alignments,
      possible_ref_alignments, dict, wcooc, nSourceWords, nTargetWords, fert_limit);

  hmmcc_interface.set_fertility_limit(fert_limit);
  if (rare_fert_limit < fert_limit)
    hmmcc_interface.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  const HmmWrapperBase* passed_wrapper = (hillclimb_mode == HillclimbingReuse) ? 0 : used_hmm;

  FertilityModelTrainerBase* last_model = 0;
  if (used_hmm == &hmmc_wrapper)
    last_model = &hmmc_interface;
  else
    last_model = &hmmcc_interface;

  // HmmWrapperNoClasses hmm_wrapper(hmmalign_model, dev_hmmalign_model, initial_prob, hmm_dist_params, hmm_dist_grouping_param, hmm_options);
  // HmmWrapperWithTargetClasses hmmc_wrapper(hmmcalign_model, dev_hmmcalign_model, initial_prob, hmmc_dist_params, hmmc_dist_grouping_param, target_class, hmm_options);

  // HmmFertInterfaceTargetClasses hmm_interface(hmmc_wrapper, source_sentence, slookup, target_sentence, sure_ref_alignments,
  // possible_ref_alignments, dict, wcooc, nSourceWords, nTargetWords, fert_limit);

  // hmm_interface.set_fertility_limit(fert_limit);
  // if (rare_fert_limit < fert_limit)
  // hmm_interface.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  //const HmmWrapperBase* passed_wrapper = (hillclimb_mode == HillclimbingReuse) ? 0 : &hmmc_wrapper;

  if (fert_p0 == 0.0 && ibm3_iter+ibm4_iter+ibm5_iter > 0) {
    //fert interface only uses the classed models

    source_fert[0] = 0.0;
    source_fert[1] = 1.0;

    //it suffices to set to NULL entries to 0 as the probability scores do not matter when called by fertility models

    for (uint I = 1; I <= initial_prob.size(); I++) {

      if (initial_prob[I-1].size() > 0)
        initial_prob[I-1].range_set_constant(0.0, I, initial_prob[I-1].size()-I);
    }

    for (uint I = 1; I <= hmmcalign_model.size(); I++) {

      if (hmmcalign_model[I-1].size() > 0) {
        for (uint c=0; c < hmmcalign_model[I-1].zDim(); c++) {
          for (uint y=0; y < hmmcalign_model[I-1].yDim(); y++) {
            for (uint x=I; x < hmmcalign_model[I-1].xDim(); x++)
              hmmcalign_model[I-1](x,y,c) = 0.0;
          }
        }
      }
    }
  }


  Math1D::Vector<uint> tfert_class(nTargetWords);
  if (app.is_set("-tfert-classes")) {
    read_word_classes(app.getParam("-tfert-classes"), tfert_class);
	if (tfert_class.size() != nTargetWords) {
	  USER_ERROR << "wrong tfert-classes, size mismatch. Exiting..." << std::endl;
	  exit(1);
	}
  }
  else {
    for (uint i = 0; i < nTargetWords; i++) {
      tfert_class[i] = i;
    }
  }

  //check the word classes
  // for (uint i=0; i < tfert_class.size(); i++) {
    // if (tfert_class[i] != i && tfert_class[i] < nTargetClasses) {
      // std::cerr << "error with tfert-classes: word " << i << " points to smaller than nWords and not to self. Exiting" << std::endl;
      // exit(1);
    // }
  // }

  //FertilityModelTrainerBase* last_model = &hmm_interface;

  /*** IBM-3 ***/

  std::cerr << "handling IBM-3" << std::endl;

  Math1D::Vector<WordClassType> i3_tclass(nTargetWords,0);
  if (!app.is_set("-no-h23-classes"))
    i3_tclass = target_class;

  IBM3Trainer ibm3_trainer(source_sentence, slookup, target_sentence, i3_tclass, sure_ref_alignments, possible_ref_alignments,
                           dict, wcooc, tfert_class, nSourceWords, nTargetWords, prior_weight, log_table, xlogx_table, fert_options);

  ibm3_trainer.set_fertility_limit(fert_limit);
  if (rare_fert_limit < fert_limit)
    ibm3_trainer.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  if (ibm3_iter > 0) {

    ibm3_trainer.init_from_prevmodel(last_model, passed_wrapper, true, collect_counts, method == "viterbi");

    if (collect_counts)
      ibm3_iter--;

    if (ibm3_iter > 0) {
      std::string constraint_mode = downcase(app.getParam("-constraint-mode"));
      if (constraint_mode == "itg") {

        if (method == "em" || method == "gd" || dict_regularity != 0.0 || l0_fertpen != 0.0)
          std::cerr << "WARNING: ITG-constrained IBM3-training is Viterbi without regularity terms!" << std::endl;

        ibm3_trainer.train_viterbi(ibm3_iter, align_constraints, 0, passed_wrapper);
      }
      else if (constraint_mode == "ibm") {

        if (method == "em" || method == "gd" || dict_regularity != 0.0 || l0_fertpen != 0.0)
          std::cerr << "WARNING: IBM-constrained IBM3-training is Viterbi without regularity terms!" << std::endl;

        ibm3_trainer.train_viterbi(ibm3_iter, align_constraints, 0, passed_wrapper);
      }
      else if (constraint_mode == "unconstrained") {

        if (method == "em" || method == "gd") {
          ibm3_trainer.train_em(ibm3_iter, 0, passed_wrapper);
        }
        else
          ibm3_trainer.train_viterbi(ibm3_iter, align_constraints, 0, passed_wrapper);
      }
      else {
        USER_ERROR << "unknown constraint mode: \"" << constraint_mode << "\". Exiting" << std::endl;
        exit(1);
      }
    }

    last_model = &ibm3_trainer;
  }

  /************ Fertility-HMM *************/
  std::cerr << "handling Fertility-HMM" << std::endl;

  FertilityHMMTrainer fert_hmm(source_sentence, slookup, target_sentence, target_class, sure_ref_alignments, possible_ref_alignments, dict, wcooc, tfert_class,
                               nSourceWords, nTargetWords, prior_weight, log_table, xlogx_table, hmm_options, fert_options);

  fert_hmm.set_fertility_limit(fert_limit);
  if (rare_fert_limit < fert_limit)
    fert_hmm.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  FertilityHMMTrainerDoubleClass fert_hmmcc(source_sentence, slookup, target_sentence, source_class, target_class,
											sure_ref_alignments, possible_ref_alignments, dict, wcooc, tfert_class,
											source_fert, maxAllI - 1, nSourceWords, nTargetWords, prior_weight, log_table, xlogx_table,
											hmm_options, fert_options, hmmcc_dist_params, hmmcc_dist_grouping_param, false);

  fert_hmmcc.set_fertility_limit(fert_limit);
  if (rare_fert_limit < fert_limit)
    fert_hmmcc.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  if (fert_hmm_iter > 0) {
	
    if (nSourceClasses == 1) {
      fert_hmm.init_from_prevmodel(last_model, passed_wrapper, hmmc_dist_params, hmmc_dist_grouping_param, true, collect_counts, method == "viterbi");

      if (collect_counts)
        fert_hmm_iter--;

      if (fert_hmm_iter > 0) {
        if (method != "viterbi")
          fert_hmm.train_em(fert_hmm_iter, 0, passed_wrapper);
        else
          fert_hmm.train_viterbi(fert_hmm_iter, 0, passed_wrapper);
      }

      last_model = &fert_hmm;
    }
    else {
      fert_hmmcc.init_from_prevmodel(last_model, passed_wrapper, hmmcc_dist_params, hmmcc_dist_grouping_param,  true, collect_counts, method == "viterbi");

      if (collect_counts)
        fert_hmm_iter--;

      if (fert_hmm_iter > 0) {
        if (method != "viterbi")
          fert_hmmcc.train_em(fert_hmm_iter, 0, passed_wrapper);
        else
          fert_hmmcc.train_viterbi(fert_hmm_iter, 0, passed_wrapper);
      }

      last_model = &fert_hmmcc;
    }
  }

  /*** IBM-4 ***/

  std::cerr << "handling IBM-4" << std::endl;

  IBM4Trainer ibm4_trainer(source_sentence, slookup, target_sentence, sure_ref_alignments, possible_ref_alignments, dict, wcooc, tfert_class,
                           nSourceWords, nTargetWords, prior_weight, source_class, target_class, log_table, xlogx_table, fert_options);

  ibm4_trainer.set_fertility_limit(fert_limit);
  if (rare_fert_limit < fert_limit)
    ibm4_trainer.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  if (ibm4_iter > 0) {

    ibm4_trainer.init_from_prevmodel(last_model, passed_wrapper, true, collect_counts, method == "viterbi");

    if (collect_counts)
      ibm4_iter--;

    if (ibm4_iter > 0) {
      if (method == "viterbi")
        ibm4_trainer.train_viterbi(ibm4_iter, 0, passed_wrapper);
      else
        ibm4_trainer.train_em(ibm4_iter, 0, passed_wrapper);
    }

    last_model = &ibm4_trainer;
  }

  /**** IBM-5 *****/

  std::cerr << "handling IBM-5" << std::endl;

  IBM5Trainer ibm5_trainer(source_sentence, slookup, target_sentence, sure_ref_alignments, possible_ref_alignments, dict, wcooc, tfert_class,
                           nSourceWords, nTargetWords, prior_weight, source_class, target_class, log_table, xlogx_table, fert_options);

  ibm5_trainer.set_fertility_limit(fert_limit);
  if (rare_fert_limit < fert_limit)
    ibm5_trainer.set_rare_fertility_limit(rare_fert_limit, nMaxRareOccurences);

  if (ibm5_iter > 0) {

    ibm5_trainer.init_from_prevmodel(last_model, passed_wrapper, true, collect_counts, method == "viterbi");

    if (collect_counts)
      ibm5_iter--;

    if (ibm5_iter > 0) {
      if (method == "viterbi")
        ibm5_trainer.train_viterbi(ibm5_iter, 0, passed_wrapper);
      else
        ibm5_trainer.train_em(ibm5_iter, 0, passed_wrapper);
    }

    last_model = &ibm5_trainer;
  }

  std::clock_t tEndTrain = std::clock();

  std::cerr << "core training took " << (diff_seconds(tEndTrain, tStartTrain) / 60.0) << " minutes." << std::endl;

  LookupTable dev_slookup;
  if (dev_present) {
    generate_wordlookup(dev_source_sentence, dev_target_sentence, wcooc, nSourceWords, dev_slookup);

    if (app.is_set("-prior-dict")) {

      Math1D::Vector<uint> target_count(nTargetWords, 0);
      for (size_t s = 0; s < target_sentence.size(); s++)
        for (uint i=0; i < target_sentence[s].size(); i++)
          target_count[target_sentence[s][i]]++;

      for (uint i=1; i < nTargetWords; i++) {

        if (target_count[i] == 0 && prior_weight[i].max_abs() != 0.0) {

          uint count = 0;
          for (uint k = 0; k < prior_weight[i].size(); k++) {
            if (prior_weight[i][k] == 0.0)
              count++;
          }
          if (count > 0) {
            for (uint k = 0; k < prior_weight[i].size(); k++) {
              if (prior_weight[i][k] == 0.0)
                dict[i][k] = 1.0 / count;
              else
                dict[i][k] = 1e-8;
            }
          }
        }
      }
    }
  }

  /*** write alignments ***/

  int max_devJ = 0;
  int max_devI = 0;

  std::set<uint> dev_seenIs;

  std::string dev_file = app.getParam("-oa") + ".dev";
  if (string_ends_with(app.getParam("-oa"), ".gz"))
    dev_file += ".gz";

  if (dev_present) {
    for (size_t s = 0; s < dev_source_sentence.size(); s++) {

      const int curI = dev_target_sentence[s].size();
      const int curJ = dev_source_sentence[s].size();

      dev_seenIs.insert(curI);

      max_devJ = std::max(max_devJ, curJ);
      max_devI = std::max(max_devI, curI);
    }
  }


  if (dev_present) {

    if (ibm2_iter > 0) {

      uint nClasses = reduced_ibm2align_param.zDim();
      uint offset = std::max<uint>(maxI,max_devI) - 1;

      if (reduced_ibm2align_model.size() < max_devI+1)
        reduced_ibm2align_model.resize(max_devI+1);

      if (max_devJ > maxJ || max_devI > maxI) {
        if (ibm2_align_mode != IBM23ParByDifference) {
          reduced_ibm2align_param.resize(std::max<uint>(maxI,max_devI), std::max<uint>(maxJ,max_devJ), nClasses, ibm2_min_align_param);
        }
        else {

          Math3D::Tensor<double> temp_param(std::max<uint>(maxJ, max_devJ) + std::max<uint>(maxI, max_devI) - 1, 1, nClasses, ibm2_min_align_param);

          for (uint c = 0; c < nClasses; c++)
            for (uint j = 0; j < maxJ; j++)
              for (uint i = 0; i < maxI; i++)
                temp_param(offset + j - i, 0, c) = reduced_ibm2align_param(maxI - 1 + j - i, 0, c);

          reduced_ibm2align_param = temp_param;
        }
      }

      if (ibm2_align_mode != IBM23Nonpar) {

        for (size_t s = 0; s < dev_source_sentence.size(); s++) {

          const int curI = dev_target_sentence[s].size();
          const int curJ = dev_source_sentence[s].size();

          if (reduced_ibm2align_model[curI].yDim() < curJ)
            reduced_ibm2align_model[curI].resize(curI + 1, curJ, nClasses);
        }

        par2nonpar_reduced_ibm2alignment_model(reduced_ibm2align_param, source_fert, reduced_ibm2align_model, ibm2_align_mode, offset);
      }
      else {

        ReducedIBM2ClassAlignmentModel temp_reduced_ibm2align_model(max_devI+1, MAKENAME(temp_reduced_ibm2align_model));

        for (size_t s = 0; s < dev_source_sentence.size(); s++) {

          const int curI = dev_target_sentence[s].size();
          const int curJ = dev_source_sentence[s].size();

          if (temp_reduced_ibm2align_model[curI].yDim() < curJ)
            temp_reduced_ibm2align_model[curI].resize(curI + 1, curJ, nClasses);
        }

        par2nonpar_reduced_ibm2alignment_model(reduced_ibm2align_param, source_fert, temp_reduced_ibm2align_model, IBM23ParByPosition, offset);

        for (uint I = 0; I < temp_reduced_ibm2align_model.size(); I++) {

          size_t prev_xDim = reduced_ibm2align_model[I].xDim();
          size_t prev_yDim = reduced_ibm2align_model[I].yDim();

          reduced_ibm2align_model[I].resize(std::max(prev_xDim,temp_reduced_ibm2align_model[I].xDim()),
                                            std::max(prev_yDim,temp_reduced_ibm2align_model[I].yDim()), nClasses);

          for (uint c = 0; c < nClasses; c++)
            for (uint x = prev_xDim; x < temp_reduced_ibm2align_model[I].xDim(); x++)
              for (uint y = prev_yDim; y < temp_reduced_ibm2align_model[I].yDim(); y++)
                reduced_ibm2align_model[I](x, y, c) = temp_reduced_ibm2align_model[I](x, y, c);
        }
      }
    }

    uint train_zero_offset = maxI - 1;

    //std::cerr << "AA" << std::endl;

    //handle case where init and/or distance parameters were not estimated above for <emph>train</emph>
    if (hmm_init_mode == HmmInitNonpar || hmm_init_params.sum() < 1e-5) {

      hmm_init_params.resize(maxI);
      hmm_init_params.set_constant(0.0);

      for (uint I = 1; I <= maxI; I++) {

        for (uint l = 0; l < std::min < uint > (I, initial_prob[I - 1].size()); l++) {
          if (l < hmm_init_params.size())
            hmm_init_params[l] += initial_prob[I - 1][l];
        }
      }

      double sum = hmm_init_params.sum();
      assert(sum > 0.0);
      hmm_init_params *= 1.0 / sum;
    }
    //std::cerr << "BB" << std::endl;

    if (hmm_align_mode == HmmAlignProbNonpar || hmm_align_mode == HmmAlignProbNonpar2 || hmm_dist_params.sum() < 1e-5) {

      hmm_dist_grouping_param = -1.0;

      hmm_dist_params.resize(2 * maxI - 1);
      hmm_dist_params.set_constant(0.0);

      for (uint I = 1; I <= maxI; I++) {

        for (uint i1 = 0; i1 < hmmalign_model[I - 1].yDim(); i1++)
          for (uint i2 = 0; i2 < I; i2++)
            hmm_dist_params[train_zero_offset + i2 - i1] += hmmalign_model[I - 1] (i1, i2);
      }

      hmm_dist_params *= 1.0 / hmm_dist_params.sum();
    }

    if ((hmm_init_mode == HmmInitNonpar && hmm_align_mode == HmmAlignProbNonpar) || source_fert.sum() < 0.95) {

      source_fert.resize(2);
      source_fert[0] = 0.02;
      source_fert[1] = 0.98;
    }
    //std::cerr << "CC" << std::endl;

    for (uint i = 0; i < std::min<uint>(max_devI, hmm_init_params.size()); i++) {
      dev_hmm_init_params[i] = hmm_init_params[i];

      dev_hmm_dist_params[dev_zero_offset - i] = hmm_dist_params[train_zero_offset - i];
      dev_hmm_dist_params[dev_zero_offset + i] = hmm_dist_params[train_zero_offset + i];
    }

    //std::cerr << "DD" << std::endl;

    dev_hmmalign_model.resize(max_devI + 1);
    dev_initial_prob.resize(max_devI + 1);

    for (std::set<uint>::const_iterator it = dev_seenIs.begin(); it != dev_seenIs.end(); it++) {

      uint I = *it;

      dev_hmmalign_model[I - 1].resize(I + 1, I, 0.0);  //because of empty words
      dev_initial_prob[I - 1].resize(2 * I, 0.0);
    }

    //std::cerr << "EE" << std::endl;

    if (hmm_init_mode != HmmInitFix && hmm_init_mode != HmmInitFix2) {
      par2nonpar_hmm_init_model(dev_hmm_init_params, source_fert, HmmInitPar, dev_initial_prob);
      if (hmm_init_mode == HmmInitNonpar) {
        for (uint I = 0; I < std::min(dev_initial_prob.size(), initial_prob.size()); I++) {
          if (dev_initial_prob[I].size() > 0 && initial_prob[I].size() > 0)
            dev_initial_prob[I] = initial_prob[I];
        }
      }
    }
    else {
      par2nonpar_hmm_init_model(dev_hmm_init_params, source_fert, hmm_init_mode, dev_initial_prob);
    }

    //std::cerr << "FF" << std::endl;

    HmmAlignProbType mode = hmm_align_mode;
    if (mode == HmmAlignProbNonpar || hmm_align_mode == HmmAlignProbNonpar2)
      mode = HmmAlignProbFullpar;

    par2nonpar_hmm_alignment_model(dev_hmm_dist_params, dev_zero_offset, hmm_dist_grouping_param, source_fert, mode,
                                   hmm_options.deficient_, dev_hmmalign_model, hmm_options.redpar_limit_);

    if (hmm_align_mode == HmmAlignProbNonpar || hmm_align_mode == HmmAlignProbNonpar2) {

      for (uint I = 0; I < std::min(dev_hmmalign_model.size(), hmmalign_model.size()); I++) {
        if (dev_hmmalign_model.size() > 0 && hmmalign_model.size() > 0)
          dev_hmmalign_model[I] = hmmalign_model[I];
      }
    }

    for (uint e = 0; e < dict.size(); e++) {
      if (dict[e].sum() == 0.0)
        dict[e].set_constant(1e-5);
    }
  }

  if (last_model != &hmmc_interface && last_model != &hmmcc_interface) {

    std::cerr << "updating final alignments" << std::endl;
    last_model->update_alignments_unconstrained();

    if (postdec_thresh <= 0.0)
      last_model->write_alignments(app.getParam("-oa"));
    else
      last_model->write_postdec_alignments(app.getParam("-oa"), postdec_thresh);

    if (dev_present) {

      std::cerr << "dev sentences present" << std::endl;

      Math1D::Vector<AlignBaseType> viterbi_alignment;
      std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;

      std::ostream* dev_alignment_stream;

#ifdef HAS_GZSTREAM
      if (string_ends_with(app.getParam("-oa"), ".gz")) {
        dev_alignment_stream = new ogzstream(dev_file.c_str());
      }
      else
        dev_alignment_stream = new std::ofstream(dev_file.c_str());

#else
      dev_alignment_stream = new std::ofstream(dev_file.c_str());
#endif

      for (size_t s = 0; s < dev_source_sentence.size(); s++) {

        //std::cerr << "s: " << s << std::endl;

        const uint curI = dev_target_sentence[s].size();

        //initialize by HMM
        const uint curJ = dev_source_sentence[s].size();

        SingleLookupTable aux_lookup;
        const SingleLookupTable& cur_lookup = get_wordlookup(dev_source_sentence[s], dev_target_sentence[s], wcooc, nSourceWords, aux_lookup, aux_lookup);

        used_hmm->compute_ehmm_viterbi_alignment(dev_source_sentence[s], cur_lookup, dev_target_sentence[s], dict, viterbi_alignment, false, false);


        // compute_ehmm_viterbi_alignment(dev_source_sentence[s], dev_slookup[s], dev_target_sentence[s], dict,
        // dev_hmmalign_model[curI - 1], dev_initial_prob[curI - 1], viterbi_alignment, hmm_options, false);

        if (postdec_thresh <= 0.0) {

          //std::cerr << "standard alignment computation" << std::endl;

          last_model->compute_external_alignment(dev_source_sentence[s], dev_target_sentence[s], dev_slookup[s], viterbi_alignment, &align_constraints);

          for (uint j = 0; j < viterbi_alignment.size(); j++) {
            if (viterbi_alignment[j] > 0)
              (*dev_alignment_stream) << (viterbi_alignment[j] - 1) << " " << j << " ";
          }

          (*dev_alignment_stream) << std::endl;
        }
        else {

          last_model->compute_external_postdec_alignment(dev_source_sentence[s], dev_target_sentence[s], dev_slookup[s],
              viterbi_alignment, postdec_alignment, postdec_thresh);

          for (std::set<std::pair<AlignBaseType,AlignBaseType> >::const_iterator it = postdec_alignment.begin();
               it != postdec_alignment.end(); it++) {

            (*dev_alignment_stream) << (it->second - 1) << " " << (it->first - 1) << " ";
          }
          (*dev_alignment_stream) << std::endl;
        }
      }
      delete dev_alignment_stream;
    }
  }
  else { //no fertility models trained

    std::ostream* alignment_stream;

#ifdef HAS_GZSTREAM
    if (string_ends_with(app.getParam("-oa"), ".gz")) {
      alignment_stream = new ogzstream(app.getParam("-oa").c_str());
    }
    else {
      alignment_stream = new std::ofstream(app.getParam("-oa").c_str());
    }
#else
    alignment_stream = new std::ofstream(app.getParam("-oa").c_str());
#endif

    Storage1D<AlignBaseType> viterbi_alignment;
    std::set<std::pair<AlignBaseType,AlignBaseType> > postdec_alignment;

    for (size_t s = 0; s < nSentences; s++) {

      const Math1D::Vector<uint>& cur_source = source_sentence[s];
      const Math1D::Vector<uint>& cur_target = target_sentence[s];

      const uint curI = cur_target.size();

      SingleLookupTable aux_lookup;
      const SingleLookupTable& cur_lookup = get_wordlookup(cur_source, cur_target, wcooc, nSourceWords, slookup[s], aux_lookup);

      if (hmm_iter > 0) {

        if (postdec_thresh <= 0.0) {

          used_hmm->compute_ehmm_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, viterbi_alignment, false, false);

          //compute_ehmm_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, hmmalign_model[curI - 1],
          //                               initial_prob[curI - 1], viterbi_alignment, hmm_options, false);
        }
        else {

          used_hmm->compute_ehmm_postdec_alignment(cur_source, cur_lookup, cur_target, dict, postdec_alignment);

          //compute_ehmm_postdec_alignment(cur_source, cur_lookup, cur_target, dict, hmmalign_model[curI - 1],
          //                               initial_prob[curI - 1], hmm_options, postdec_alignment, postdec_thresh);
        }
      }
      else if (ibm2_iter > 0) {

        const Math3D::Tensor<double>& cur_align_model = reduced_ibm2align_model[curI];

        if (postdec_thresh <= 0.0)
          compute_ibm2_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, ibm2_sclass, viterbi_alignment);
        else
          compute_ibm2_postdec_alignment(cur_source, cur_lookup, cur_target, dict, cur_align_model, ibm2_sclass, postdec_alignment, postdec_thresh);
      }
      else {

        if (postdec_thresh <= 0.0) {
          if (ibm1_p0 < 0.0)
            compute_ibm1_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, viterbi_alignment);
          else
            compute_ibm1p0_viterbi_alignment(cur_source, cur_lookup, cur_target, dict, ibm1_p0, viterbi_alignment);
        }
        else {
          if (ibm1_p0 < 0.0)
            compute_ibm1_postdec_alignment(cur_source, cur_lookup, cur_target, dict, postdec_alignment, postdec_thresh);
          else
            compute_ibm1p0_postdec_alignment(cur_source, cur_lookup, cur_target, dict, ibm1_p0, postdec_alignment, postdec_thresh);
        }
      }

      if (postdec_thresh <= 0.0) {
        for (uint j = 0; j < viterbi_alignment.size(); j++) {
          if (viterbi_alignment[j] > 0)
            (*alignment_stream) << (viterbi_alignment[j] - 1) << " " << j << " ";
        }
      }
      else {

        for (std::set<std::pair<AlignBaseType,AlignBaseType> >::const_iterator it = postdec_alignment.begin();
             it != postdec_alignment.end(); it++) {

          (*alignment_stream) << (it->second - 1) << " " << (it->first - 1) << " ";
        }
      }

      (*alignment_stream) << std::endl;
    }

    delete alignment_stream;

    if (dev_present) {

      std::cerr << "dev sentences present" << std::endl;

      std::ostream* dev_alignment_stream;

#ifdef HAS_GZSTREAM
      if (string_ends_with(app.getParam("-oa"), ".gz")) {
        dev_alignment_stream = new ogzstream(dev_file.c_str());
      }
      else {
        dev_alignment_stream = new std::ofstream(dev_file.c_str());
      }
#else
      dev_alignment_stream = new std::ofstream(dev_file.c_str());
#endif

      for (size_t s = 0; s < dev_source_sentence.size(); s++) {

        const Math1D::Vector<uint>& cur_dev_source = dev_source_sentence[s];
        const Math1D::Vector<uint>& cur_dev_target = dev_target_sentence[s];

        const SingleLookupTable& cur_lookup = dev_slookup[s];

        const uint curI = cur_dev_target.size();

        if (hmm_iter > 0) {

          if (postdec_thresh <= 0.0) {

            used_hmm->compute_ehmm_viterbi_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, viterbi_alignment, false, false);

            //compute_ehmm_viterbi_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, dev_hmmalign_model[curI - 1], dev_initial_prob[curI - 1],
            //                               viterbi_alignment, hmm_options, false);
          }
          else {

            used_hmm->compute_ehmm_postdec_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, postdec_alignment);

            //compute_ehmm_postdec_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, dev_hmmalign_model[curI - 1], dev_initial_prob[curI - 1],
            //                               hmm_options, postdec_alignment, postdec_thresh);
          }
        }
        else if (ibm2_iter > 0) {

          const Math3D::Tensor<double>& cur_align_model = reduced_ibm2align_model[curI];

          if (postdec_thresh <= 0.0)
            compute_ibm2_viterbi_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, cur_align_model, ibm2_sclass, viterbi_alignment);
          else
            compute_ibm2_postdec_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, cur_align_model, ibm2_sclass, postdec_alignment, postdec_thresh);
        }
        else {

          if (postdec_thresh <= 0.0) {
            if (ibm1_p0 < 0.0)
              compute_ibm1_viterbi_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, viterbi_alignment);
            else
              compute_ibm1p0_viterbi_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, ibm1_p0, viterbi_alignment);
          }
          else {
            if (ibm1_p0 < 0.0)
              compute_ibm1_postdec_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, postdec_alignment, postdec_thresh);
            else
              compute_ibm1p0_postdec_alignment(cur_dev_source, cur_lookup, cur_dev_target, dict, ibm1_p0, postdec_alignment, postdec_thresh);
          }
        }

        if (postdec_thresh <= 0.0) {
          for (uint j = 0; j < viterbi_alignment.size(); j++) {
            if (viterbi_alignment[j] > 0)
              (*dev_alignment_stream) << (viterbi_alignment[j] - 1) << " " << j << " ";
          }
        }
        else {

          for (std::set<std::pair<AlignBaseType,AlignBaseType> >::const_iterator it = postdec_alignment.begin();
               it != postdec_alignment.end(); it++) {

            (*dev_alignment_stream) << (it->second - 1) << " " << (it->first - 1) << " ";
          }
        }

        (*dev_alignment_stream) << std::endl;
      }

      delete dev_alignment_stream;
    }
  }

  /*** write dictionary ***/

  if (app.is_set("-o")) {
    std::ofstream out(app.getParam("-o").c_str());
    for (uint j = 0; j < nTargetWords; j++) {
      for (uint k = 0; k < dict[j].size(); k++) {
        uint word = (j > 0) ? wcooc[j][k] : k + 1;
        if (dict[j][k] > 1e-7)
          out << j << " " << word << " " << dict[j][k] << std::endl;
      }
    }
    out.close();
  }

}
