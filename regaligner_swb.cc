/*** written by Thomas Schoenemann 
 *** initially as a private person without employment, October 2009 
 *** later continued as an employee of Lund University, Sweden, 2010 - March 2011
 *** and later as a private person taking a time off ***/

#include "makros.hh"
#include "application.hh"
#include "corpusio.hh"
#include "ibm1_training.hh"
#include "ibm2_training.hh"
#include "hmm_training.hh"
#include "timing.hh"
#include "training_common.hh"
#include "alignment_computation.hh"
#include "singleword_fertility_training.hh"
#include "alignment_error_rate.hh"

#include <fstream>

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

int main(int argc, char** argv) {

  if (argc == 1 || strings_equal(argv[1],"-h")) {
    
    std::cerr << "USAGE: " << argv[0] << std::endl
              << " -s <file> : source file (coded as indices)" << std::endl
              << " -t <file> : target file (coded as indices)" << std::endl
              << " [-ds <file>] : additional source file (word indices) " << std::endl
              << " [-dt <file>] : additional target file (word indices) " << std::endl
              << " [-refa <file>] : file containing gold alignments (sure and possible)" << std::endl
              << " [-invert-biling-data] : switch source and target for prior dict and gold alignments" << std::endl
              << " [-method ( em | gd | viterbi )] : use EM, gradient descent or Viterbi training (default EM) " << std::endl
              << " [-dict-regularity <double>] : regularity weight for L0 or L1 regularization" << std::endl
              << " [-sparse-reg] : activate L1-regularity only for rarely occuring target words" << std::endl
              << " [-fertpen <double>]: regularity weight for fertilities in IBM3&4" << std::endl
              << " [-prior-dict <file>] : file for index pairs that occur in a dictionary" << std::endl
              << " [-l0-beta <double>] : smoothing parameter for the L0-norm in EM-training" << std::endl
              << " [-hmm-iter <uint> ]: iterations for the HMM model (default 20)" << std::endl
              << " [-ibm1-iter <uint> ]: iterations for the IBM-1 model (default 10)" << std::endl
              << " [-ibm2-iter <uint> ]: iterations for the IBM-2 model (default 0)" << std::endl
              << " [-ibm3-iter <uint> ]: iterations for the IBM-3 model (default 0)" << std::endl
              << " [-ibm4-iter <uint> ]: iterations for the IBM-4 model (default 0)" << std::endl
	      << " [-ibm4-mode (first | center | last) ] : (default first)" << std::endl
              << " [-constraint-mode (unconstrained | itg | ibm) " << std::endl
              << " [-o <file>] : the determined dictionary is written to this file" << std::endl
              << " -oa <file> : the determined alignment is written to this file" << std::endl
              << std::endl;

    std::cerr << "this program estimates p(s|t)" << std::endl;;

    exit(0);
  }

  const int nParams = 21;
  ParamDescr  params[nParams] = {{"-s",mandInFilename,0,""},{"-t",mandInFilename,0,""},
                                 {"-ds",optInFilename,0,""},{"-dt",optInFilename,0,""},
                                 {"-o",optOutFilename,0,""},{"-oa",mandOutFilename,0,""},
                                 {"-refa",optInFilename,0,""},{"-invert-biling-data",flag,0,""},
                                 {"-dict-regularity",optWithValue,1,"0.0"},
                                 {"-sparse-reg",flag,0,""},{"-prior-dict",optInFilename,0,""},
                                 {"-hmm-iter",optWithValue,1,"5"},{"-method",optWithValue,1,"em"},
                                 {"-ibm1-iter",optWithValue,1,"5"},{"-ibm2-iter",optWithValue,1,"0"},
                                 {"-ibm3-iter",optWithValue,0,""},{"-ibm4-iter",optWithValue,0,""},
                                 {"-fertpen",optWithValue,1,"0.0"},{"-constraint-mode",optWithValue,1,"unconstrained"},
				 {"-l0-beta",optWithValue,1,"-1.0"},{"-ibm4-mode",optWithValue,1,"first"}};

  Application app(argc,argv,params,nParams);

  NamedStorage1D<Storage1D<uint> > source_sentence(MAKENAME(source_sentence));
  NamedStorage1D<Storage1D<uint> > target_sentence(MAKENAME(target_sentence));

  NamedStorage1D<Storage1D<uint> > dev_source_sentence(MAKENAME(dev_source_sentence));
  NamedStorage1D<Storage1D<uint> > dev_target_sentence(MAKENAME(dev_target_sentence));  

  std::map<uint,std::set<std::pair<ushort,ushort> > > sure_ref_alignments;
  std::map<uint,std::set<std::pair<ushort,ushort> > > possible_ref_alignments;

  if (app.is_set("-refa")) {
    read_reference_alignment(app.getParam("-refa"), sure_ref_alignments, possible_ref_alignments,
                             app.is_set("-invert-biling-data"));
  }

  uint ibm1_iter = convert<uint>(app.getParam("-ibm1-iter")); 
  uint ibm2_iter = convert<uint>(app.getParam("-ibm2-iter"));
  uint hmm_iter = convert<uint>(app.getParam("-hmm-iter"));

  uint ibm3_iter = 0; 
  uint ibm4_iter = 0;

  if (app.is_set("-ibm3-iter"))
    ibm3_iter = convert<uint>(app.getParam("-ibm3-iter"));
  if (app.is_set("-ibm4-iter"))
    ibm4_iter = convert<uint>(app.getParam("-ibm4-iter"));

  std::string method = app.getParam("-method");

  if (method != "em" && method != "gd" && method != "viterbi") {
    USER_ERROR << "unknown method \"" << method << "\"" << std::endl;
    exit(1);
  }

  double l0_fertpen = convert<double>(app.getParam("-fertpen"));

  double l0_beta = convert<double>(app.getParam("-l0-beta"));
  bool em_l0 = (l0_beta > 0);

  std::clock_t tStartRead, tEndRead;
  tStartRead = std::clock();

  if (app.getParam("-s") == app.getParam("-t")) {

    std::cerr << "WARNING: files for source and target sentences are identical!" << std::endl;
  }

  read_monolingual_corpus(app.getParam("-s"), source_sentence);
  read_monolingual_corpus(app.getParam("-t"), target_sentence);

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
  std::cerr << "reading the corpus took " << diff_seconds(tEndRead,tStartRead) << " seconds." << std::endl;

  assert(source_sentence.size() == target_sentence.size());

  uint nSentences = source_sentence.size();

  uint maxI = 0;

  for (size_t s=0; s < source_sentence.size(); s++) {

    uint curJ = source_sentence[s].size();
    uint curI = target_sentence[s].size();

    maxI = std::max<uint>(maxI,curI);

    if (9*curJ < curI || 9*curI < curJ) {

      std::cerr << "WARNING: GIZA++ would ignore sentence pair #" << (s+1) << ": J=" << curJ << ", I=" << curI << std::endl;
    }
  }

  uint nSourceWords = 0;
  uint nTargetWords = 0;

  for (size_t s=0; s < source_sentence.size(); s++) {

    for (uint k=0; k < source_sentence[s].size(); k++)
      nSourceWords = std::max(nSourceWords,source_sentence[s][k]+1);

    for (uint k=0; k < target_sentence[s].size(); k++)
      nTargetWords = std::max(nTargetWords,target_sentence[s][k]+1);
  }

  CooccuringWordsType wcooc(MAKENAME(wcooc));
  CooccuringLengthsType lcooc(MAKENAME(lcooc));
  SingleWordDictionary dict(MAKENAME(dict));
  ReducedIBM2AlignmentModel reduced_ibm2align_model(MAKENAME(reduced_ibm2align_model));
  FullHMMAlignmentModel hmmalign_model(MAKENAME(hmmalign_model));
  InitialAlignmentProbability initial_prob(MAKENAME(initial_prob));

  Math1D::Vector<double> source_fert;

  Math1D::Vector<double> hmm_init_params;
  Math1D::Vector<double> hmm_dist_params;
  double hmm_dist_grouping_param = -1.0;

  HmmAlignProbType train_dist_mode = HmmAlignProbReducedpar;
  if (method == "viterbi")
    train_dist_mode = HmmAlignProbFullpar;

  std::cerr << "finding cooccuring words" << std::endl;
  find_cooccuring_words(source_sentence, target_sentence, dev_source_sentence, dev_target_sentence, 
                        nSourceWords, nTargetWords, wcooc);
  
  std::cerr << "generating lookup table" << std::endl;
  Storage1D<Math2D::Matrix<uint> > slookup;
  generate_wordlookup(source_sentence, target_sentence, wcooc, slookup);

  double dict_regularity = convert<double>(app.getParam("-dict-regularity"));
    
  floatSingleWordDictionary prior_weight(nTargetWords, MAKENAME(prior_weight));
      
  if (app.is_set("-ibm3-iter"))
    ibm3_iter = convert<uint>(app.getParam("-ibm3-iter"));
  if (app.is_set("-ibm4-iter"))
    ibm4_iter = convert<uint>(app.getParam("-ibm4-iter"));

  Math1D::Vector<double> distribution_weight;

  std::set<std::pair<uint, uint> > known_pairs;
  if (app.is_set("-prior-dict"))
    read_prior_dict(app.getParam("-prior-dict"), known_pairs, app.is_set("-invert-biling-data"));
  
  for (uint i=0; i < nTargetWords; i++)
    prior_weight[i].resize(wcooc[i].size(),0.0);
  
  if (known_pairs.size() > 0) {
    
    for (uint i=0; i < nTargetWords; i++)
      prior_weight[i].set_constant(dict_regularity);
    
    uint nIgnored = 0;
    
    std::cerr << "processing read list" << std::endl;
    
    for (std::set<std::pair<uint, uint> >::iterator it = known_pairs.begin(); it != known_pairs.end() ; it++) {
      
      uint tword = it->first;
      uint sword = it->second;
      
      if (tword >= wcooc.size()) {
        std::cerr << "tword out of range: " << tword << std::endl;
      }

      uint pos = std::lower_bound(wcooc[tword].direct_access(), wcooc[tword].direct_access() + wcooc[tword].size(), sword) - 
        wcooc[tword].direct_access();

      if (pos < wcooc[tword].size()) {
        prior_weight[tword][pos] = 0.0;
      }
      else {
        nIgnored++;
        //std::cerr << "WARNING: ignoring entry of prior dictionary" << std::endl;
      }
    }
    
    std::cerr << "ignored " << nIgnored << " entries of prior dictionary" << std::endl;
  }
  else {
    
    distribution_weight.resize(nTargetWords,0.0);
    
    if (!app.is_set("-sparse-reg")) {
      distribution_weight.set_constant(dict_regularity);
    }
    else {
      for (size_t s=0; s < target_sentence.size(); s++) {
	
        for (uint i=0; i < target_sentence[s].size(); i++) {
          distribution_weight[target_sentence[s][i]] += 1.0;
        }
      }
      
      uint cutoff = 6;
      
      uint nSparse = 0;
      for (uint i=0; i < nTargetWords; i++) {
        if (distribution_weight[i] >= cutoff+1) 
          distribution_weight[i] = 0.0;
        else {
          nSparse++;
          //std::cerr << "sparse word: " << distribution_weight[i] << std::endl;
          distribution_weight[i] = (cutoff+1) - distribution_weight[i];
        }
      }
      distribution_weight[0] = 0.0;
      distribution_weight *= dict_regularity;
      std::cerr << "reg_sum: " << distribution_weight.sum() << std::endl;
      std::cerr << nSparse << " sparse words" << std::endl;
    }
    
    for (uint i=0; i < nTargetWords; i++)
      prior_weight[i].set_constant(distribution_weight[i]);
  }


  /**** now start training *****/

  std::cerr << "starting IBM 1 training" << std::endl;

  /*** IBM-1 ***/

  if (method == "em") {

    train_ibm1(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords, dict, ibm1_iter,
               sure_ref_alignments, possible_ref_alignments, prior_weight, em_l0, l0_beta);

  }
  else if (method == "gd") {

    train_ibm1_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords, dict, ibm1_iter,
                              sure_ref_alignments, possible_ref_alignments, prior_weight, em_l0, l0_beta); 
  }
  else {

    ibm1_viterbi_training(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords, dict, 
                          ibm1_iter, sure_ref_alignments, possible_ref_alignments, prior_weight);

  }

  /*** IBM-2 ***/

  if (ibm2_iter > 0) {
    
    find_cooccuring_lengths(source_sentence, target_sentence, lcooc);

    if (method == "em") {

      train_reduced_ibm2(source_sentence,  slookup, target_sentence, wcooc, lcooc,
                         nSourceWords, nTargetWords, reduced_ibm2align_model, dict, ibm2_iter,
                         sure_ref_alignments, possible_ref_alignments);
    }
    else if (method == "gd") {

      std::cerr << "WARNING: IBM-2 is not available with gradient descent" << std::endl;
      train_reduced_ibm2(source_sentence,  slookup, target_sentence, wcooc, lcooc,
                         nSourceWords, nTargetWords, reduced_ibm2align_model, dict, ibm2_iter,
                         sure_ref_alignments, possible_ref_alignments);
    }
    else {

      ibm2_viterbi_training(source_sentence, slookup, target_sentence, wcooc, lcooc, nSourceWords, nTargetWords, 
                            reduced_ibm2align_model, dict, ibm2_iter, sure_ref_alignments, possible_ref_alignments, 
                            prior_weight);
    }
  }

  /*** HMM ***/

  if (method == "em") {

    train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords,
                       hmmalign_model, hmm_dist_params, hmm_dist_grouping_param, source_fert,
                       initial_prob, hmm_init_params, dict, hmm_iter, HmmInitPar, train_dist_mode,
                       sure_ref_alignments, possible_ref_alignments, prior_weight, em_l0, l0_beta);
  }
  else if (method == "gd") {

    train_extended_hmm_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords,
                                      hmmalign_model, hmm_dist_params, hmm_dist_grouping_param, source_fert,
                                      initial_prob, hmm_init_params, dict, hmm_iter, HmmInitPar, train_dist_mode,
                                      sure_ref_alignments, possible_ref_alignments, prior_weight, em_l0, l0_beta); 
  }
  else {

    viterbi_train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords,
                               hmmalign_model, hmm_dist_params, hmm_dist_grouping_param, source_fert,
                               initial_prob, dict, hmm_iter, HmmInitFix, train_dist_mode, false,  
                               sure_ref_alignments, possible_ref_alignments, prior_weight);
  }
  
  /*** IBM-3 ***/

  std::cerr << "handling IBM-3" << std::endl;

  IBM3Trainer ibm3_trainer(source_sentence, slookup, target_sentence, 
                           sure_ref_alignments, possible_ref_alignments,
                           dict, wcooc, nSourceWords, nTargetWords, prior_weight, 
                           true, true, false, l0_fertpen, em_l0, l0_beta);
  
  if (ibm3_iter+ibm4_iter > 0)
    ibm3_trainer.init_from_hmm(hmmalign_model,initial_prob,train_dist_mode);

  if (ibm3_iter > 0) {

    if (method == "em" || method == "gd") {
      
      std::string constraint_mode = app.getParam("-constraint-mode");

      if (constraint_mode == "unconstrained") {
        ibm3_trainer.train_unconstrained(ibm3_iter);
      }
      else if (constraint_mode == "itg") 
        ibm3_trainer.train_with_itg_constraints(ibm3_iter,true);
      else if (constraint_mode == "ibm") 
        ibm3_trainer.train_with_ibm_constraints(ibm3_iter,5,3);
      else {
        USER_ERROR << "unknown constraint mode: \"" << constraint_mode << "\". Exiting" << std::endl;
        exit(1);
      }
    }
    else
      ibm3_trainer.train_viterbi(ibm3_iter,false);
  
    //ibm3_trainer.update_alignments_unconstrained();
  }

  /*** IBM-4 ***/

  std::cerr << "handling IBM-4" << std::endl;

  IBM4CeptStartMode ibm4_cept_mode = IBM4FIRST;
  std::string ibm4_mode = app.getParam("-ibm4-mode");
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

  
  IBM4Trainer ibm4_trainer(source_sentence, slookup, target_sentence, 
                           sure_ref_alignments, possible_ref_alignments,
                           dict, wcooc, nSourceWords, nTargetWords, prior_weight, true, true, true,
                           ibm4_cept_mode, em_l0, l0_beta, l0_fertpen);

  if (ibm4_iter > 0) {
    bool collect_counts = false;
    
    ibm4_trainer.init_from_ibm3(ibm3_trainer,true,collect_counts,method == "viterbi");
    
    if (collect_counts)
      ibm4_iter--;
    
    if (method == "viterbi")
      ibm4_trainer.train_viterbi(ibm4_iter);
    else
      ibm4_trainer.train_unconstrained(ibm4_iter);

    //ibm4_trainer.update_alignments_unconstrained();
  }

  Storage1D<Math2D::Matrix<uint> > dev_slookup;
  if (dev_present) {
    generate_wordlookup(dev_source_sentence, dev_target_sentence, wcooc, dev_slookup);
  }

  /*** write alignments ***/

  int max_devJ = 0;
  int max_devI = 0;

  std::set<uint> dev_seenIs;

  std::string dev_file = app.getParam("-oa") + ".dev";
  if (string_ends_with(app.getParam("-oa"),".gz"))
    dev_file += ".gz";

  if (dev_present) {
    for (size_t s = 0; s < dev_source_sentence.size(); s++) {

      const int curI = dev_target_sentence[s].size();
      const int curJ = dev_source_sentence[s].size();

      dev_seenIs.insert(curI);

      max_devJ = std::max(max_devJ,curJ);
      max_devI = std::max(max_devI,curI);	

      //dev_lengthJ[curJ] = std::max(curI,dev_lengthJ[curJ]);
      //dev_lengthI[curI] = std::max(curJ,dev_lengthI[curI]);
    }
  }

  Math1D::Vector<double> dev_hmm_init_params(max_devI,0.0);
  Math1D::Vector<double> dev_hmm_dist_params(std::max(2*max_devI-1,0),0.0);
  FullHMMAlignmentModel dev_hmmalign_model(MAKENAME(dev_hmmalign_model));
  InitialAlignmentProbability dev_initial_prob(MAKENAME(dev_initial_prob));

  uint dev_zero_offset = max_devI - 1;

  if (dev_present) {

    HmmAlignProbType dev_dist_mode = HmmAlignProbReducedpar;

    if (hmm_dist_grouping_param < 0.0) {
      dev_dist_mode = HmmAlignProbFullpar;
    }

    //std::cerr << "AA" << std::endl;

    //handle case where init and/or distance parameters were not estimated above for _train_
    if (hmm_init_params.size() == 0) {

      hmm_init_params.resize(maxI,0.0);

      for (uint k=0; k < initial_prob.size(); k++) {

	for (uint l=0; l < initial_prob[k].size(); l++) {
	  if (l < hmm_init_params.size()) 
	    hmm_init_params[l] += initial_prob[k][l];
	}
      }

      double sum = hmm_init_params.sum();
      assert(sum > 0.0);
      hmm_init_params *= 1.0 / sum;
    }

    //std::cerr << "BB" << std::endl;

    if (hmm_dist_params.size() == 0) {

      dev_dist_mode = HmmAlignProbFullpar;
      hmm_dist_grouping_param = -1.0;

      hmm_dist_params.resize(2*maxI-1,0.0);

      source_fert.resize(2);
      source_fert.set_constant(0.0);
    }

    //std::cerr << "CC" << std::endl;
      
    uint train_zero_offset = maxI - 1;

    for (uint i=0; i < std::min<uint>(max_devI,hmm_init_params.size()); i++) {
      dev_hmm_init_params[i] = hmm_init_params[i];	
	
      dev_hmm_dist_params[dev_zero_offset - i] = hmm_dist_params[train_zero_offset - i];
      dev_hmm_dist_params[dev_zero_offset + i] = hmm_dist_params[train_zero_offset + i];
    }

    //std::cerr << "DD" << std::endl;

    dev_hmmalign_model.resize(max_devI+1);
    dev_initial_prob.resize(max_devI+1);

    for (std::set<uint>::iterator it = dev_seenIs.begin(); it != dev_seenIs.end(); it++) {

      uint I = *it;

      dev_hmmalign_model[I-1].resize(I+1,I,0.0); //because of empty words
      dev_initial_prob[I-1].resize(2*I,0.0);
    }

    //std::cerr << "EE" << std::endl;
      
    par2nonpar_hmm_init_model(dev_hmm_init_params, source_fert, HmmInitPar, dev_initial_prob);
      
    //std::cerr << "FF" << std::endl;

    par2nonpar_hmm_alignment_model(dev_hmm_dist_params, dev_zero_offset, hmm_dist_grouping_param, source_fert,
				   dev_dist_mode, dev_hmmalign_model);

    for (uint e=0; e < dict.size(); e++) {
      if (dict[e].sum() == 0.0)
	dict[e].set_constant(1e-5);
    }
  }



  if (ibm4_iter > 0) {
    ibm4_trainer.write_alignments(app.getParam("-oa"));

    if (dev_present) {
      
      std::cerr << "dev sentences present" << std::endl;
      
      Math1D::Vector<ushort> viterbi_alignment;
      
      std::ostream* dev_alignment_stream;
      
#ifdef HAS_GZSTREAM
      if (string_ends_with(app.getParam("-oa"),".gz")) {
	dev_alignment_stream = new ogzstream(dev_file.c_str());
      }
      else {
#else
      if (true) {
#endif
        dev_alignment_stream = new std::ofstream(dev_file.c_str());
      }

      for (size_t s = 0; s < dev_source_sentence.size(); s++) {
	
	//std::cerr << "s: " << s << std::endl;
	
	const uint curI = dev_target_sentence[s].size();
	
	//initialize by HMM
	compute_ehmm_viterbi_alignment(dev_source_sentence[s],dev_slookup[s], dev_target_sentence[s], 
				       dict, dev_hmmalign_model[curI-1], dev_initial_prob[curI-1],
				       viterbi_alignment, false);
		
	ibm4_trainer.compute_external_alignment(dev_source_sentence[s],dev_target_sentence[s],dev_slookup[s],
						viterbi_alignment);
	
	for (uint j=0; j < viterbi_alignment.size(); j++) { 
	  if (viterbi_alignment[j] > 0)
	    (*dev_alignment_stream) << (viterbi_alignment[j]-1) << " " << j << " ";
	}
	
	(*dev_alignment_stream) << std::endl;
      }
      delete dev_alignment_stream;
    }
  }
  else if (ibm3_iter > 0) {
    ibm3_trainer.write_alignments(app.getParam("-oa"));

    if (dev_present) {

      Math1D::Vector<ushort> viterbi_alignment;
      
      std::ostream* dev_alignment_stream;
      
#ifdef HAS_GZSTREAM
      if (string_ends_with(app.getParam("-oa"),".gz")) {
	dev_alignment_stream = new ogzstream(dev_file.c_str());
      }
      else {
#else
      if (true) {
#endif
        dev_alignment_stream = new std::ofstream(dev_file.c_str());
      }

      for (size_t s = 0; s < dev_source_sentence.size(); s++) {
	  
	const uint curI = dev_target_sentence[s].size();
	
	//initialize by HMM
	compute_ehmm_viterbi_alignment(dev_source_sentence[s],dev_slookup[s], dev_target_sentence[s], 
				       dict, dev_hmmalign_model[curI-1], dev_initial_prob[curI-1],
				       viterbi_alignment,false);
	
	ibm3_trainer.compute_external_alignment(dev_source_sentence[s],dev_target_sentence[s],dev_slookup[s],
						viterbi_alignment);
	
	for (uint j=0; j < viterbi_alignment.size(); j++) { 
	  if (viterbi_alignment[j] > 0)
	    (*dev_alignment_stream) << (viterbi_alignment[j]-1) << " " << j << " ";
	}
	
	(*dev_alignment_stream) << std::endl;
      }
      delete dev_alignment_stream;
    }  
  }
  else {

    std::ostream* alignment_stream;

#ifdef HAS_GZSTREAM
    if (string_ends_with(app.getParam("-oa"),".gz")) {
      alignment_stream = new ogzstream(app.getParam("-oa").c_str());
    }
    else {
#else
    if (true) {
#endif
      alignment_stream = new std::ofstream(app.getParam("-oa").c_str());
    }

    Storage1D<ushort> viterbi_alignment;

    for (size_t s = 0; s < nSentences; s++) {

      const uint curI = target_sentence[s].size();

      if (hmm_iter > 0) {
	
        compute_ehmm_viterbi_alignment(source_sentence[s],slookup[s], target_sentence[s], 
                                       dict, hmmalign_model[curI-1], initial_prob[curI-1], viterbi_alignment);
      }
      else if (ibm2_iter > 0) {
	
        const Math2D::Matrix<double>& cur_align_model = reduced_ibm2align_model[curI];

        compute_ibm2_viterbi_alignment(source_sentence[s], slookup[s], target_sentence[s], dict, 
                                       cur_align_model, viterbi_alignment);
      }
      else {

        compute_ibm1_viterbi_alignment(source_sentence[s], slookup[s], target_sentence[s], dict, viterbi_alignment);
      }

      for (uint j=0; j < viterbi_alignment.size(); j++) { 
	if (viterbi_alignment[j] > 0)
	    (*alignment_stream) << (viterbi_alignment[j]-1) << " " << j << " ";
      }

      (*alignment_stream) << std::endl;
    }

    delete alignment_stream;
    
    if (dev_present) {
      
      std::ostream* dev_alignment_stream;
      
#ifdef HAS_GZSTREAM
      if (string_ends_with(app.getParam("-oa"),".gz")) {
	dev_alignment_stream = new ogzstream(dev_file.c_str());
      }
      else {
#else
      if (true) {
#endif
        dev_alignment_stream = new std::ofstream(dev_file.c_str());
      }
	
	
      for (size_t s = 0; s < dev_source_sentence.size(); s++) {
          
	const uint curI = dev_target_sentence[s].size();
        
	if (hmm_iter > 0) {
	  
	  compute_ehmm_viterbi_alignment(dev_source_sentence[s],dev_slookup[s], dev_target_sentence[s], 
					 dict, dev_hmmalign_model[curI-1], dev_initial_prob[curI-1],
					 viterbi_alignment, false);
	}
	else if (ibm2_iter > 0) {
	    
	  if (s == 0) 
	    std::cerr << "Warning: derivation of alignments for IBM-2 is currently only supported on the training set" 
		      << std::endl;
	}
	else {
	  
	  compute_ibm1_viterbi_alignment(dev_source_sentence[s], dev_slookup[s], 
					 dev_target_sentence[s], dict, viterbi_alignment);
	}

	for (uint j=0; j < viterbi_alignment.size(); j++) { 
	  if (viterbi_alignment[j] > 0)
	    (*dev_alignment_stream) << (viterbi_alignment[j]-1) << " " << j << " ";
	}
	
	(*dev_alignment_stream) << std::endl;
      }
      
      delete dev_alignment_stream;
    }    
  }

  /*** write dictionary ***/
  
  if (app.is_set("-o")) {
    std::ofstream out(app.getParam("-o").c_str());
    for (uint j=0; j < nTargetWords; j++) {
      for (uint k=0; k < dict[j].size(); k++) {
        if (dict[j][k] > 1e-7)
          out << j << " " << wcooc[j][k] << " " << dict[j][k] << std::endl;
      }
    }
    out.close();
  }

}
