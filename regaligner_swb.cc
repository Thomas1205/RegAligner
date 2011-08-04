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

int main(int argc, char** argv) {

  if (argc == 1 || strings_equal(argv[1],"-h")) {
    
    std::cerr << "USAGE: " << argv[0] << std::endl
	      << " -s <file> : source file (coded as indices)" << std::endl
	      << " -t <file> : target file (coded as indices)" << std::endl
	      << " [-refa <file>] : file containing gold alignments (sure and possible)" << std::endl
	      << " [-invert-biling-data] : switch source and target for prior dict and gold alignments" << std::endl
	      << " -method ( em | gd | viterbi ) : use EM, gradient descent or Viterbi training (default EM) " << std::endl
	      << " [-dict-regularity <double>] : regularity weight for L0 or L1 regularization" << std::endl
	      << " [-sparse-reg] : activate L1-regularity only for rarely occuring target words" << std::endl
	      << " [-fertpen <double>]: regularity weight for fertilities in IBM3&4" << std::endl
	      << " [-prior-dict <file>] : file for index pairs that occur in a dictionary" << std::endl
	      << " [-hmm-iter <uint> ]: iterations for the HMM model (default 20)" << std::endl
	      << " [-ibm1-iter <uint> ]: iterations for the IBM-1 model (default 10)" << std::endl
	      << " [-ibm2-iter <uint> ]: iterations for the IBM-2 model (default 0)" << std::endl
	      << " [-ibm3-iter <uint> ]: iterations for the IBM-3 model (default 0)" << std::endl
	      << " [-ibm4-iter <uint> ]: iterations for the IBM-4 model (default 0)" << std::endl
	      << " -o <file>  : the determined dictionary is written to this file" << std::endl
	      << " -oa <file> : the determined alignment is written to this file" << std::endl
	      << std::endl;

    std::cerr << "this program estimates p(s|t)" << std::endl;;

    exit(0);
  }

  const int nParams = 14;
  ParamDescr  params[nParams] = {{"-s",mandInFilename,0,""},{"-t",mandInFilename,0,""},
				 {"-o",optOutFilename,0,""},{"-oa",mandOutFilename,0,""},
				 {"-refa",optInFilename,0,""},{"-invert-biling-data",flag,0,""},
				 {"-dict-regularity",optWithValue,1,"0.0"},
				 {"-sparse-reg",flag,0,""},{"-prior-dict",optInFilename,0,""},
				 {"-hmm-iter",optWithValue,1,"20"},{"-method",optWithValue,1,"em"}
				 {"-ibm1-iter",optWithValue,1,"10"},{"-ibm2-iter",optWithValue,1,"0"},
				 {"-ibm3-iter",optWithValue,0,""},{"-ibm4-iter",optWithValue,0,""},
				 {"-fertpen",optWithValue,1,"0.0"}};

  Application app(argc,argv,params,nParams);

  NamedStorage1D<Storage1D<uint> > source_sentence(MAKENAME(source_sentence));
  NamedStorage1D<Storage1D<uint> > target_sentence(MAKENAME(target_sentence));
  
  std::map<uint,std::set<std::pair<uint,uint> > > sure_ref_alignments;
  std::map<uint,std::set<std::pair<uint,uint> > > possible_ref_alignments;

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

  timeval tStartRead, tEndRead;
  gettimeofday(&tStartRead,0);

  read_monolingual_corpus(app.getParam("-s"), source_sentence);
  read_monolingual_corpus(app.getParam("-t"), target_sentence);

  gettimeofday(&tEndRead,0);
  std::cerr << "reading the corpus took " << diff_seconds(tEndRead,tStartRead) << " seconds." << std::endl;

  assert(source_sentence.size() == target_sentence.size());

  uint nSentences = source_sentence.size();

  for (uint s=0; s < source_sentence.size(); s++) {

    uint curJ = source_sentence[s].size();
    uint curI = target_sentence[s].size();

    if (9*curJ < curI || 9*curI < curJ) {

      std::cerr << "WARNING: GIZA++ would ignore sentence pair #" << (s+1) << ": J=" << curJ << ", I=" << curI << std::endl;
    }
  }

  uint nSourceWords = 0;
  uint nTargetWords = 0;

  for (uint s=0; s < source_sentence.size(); s++) {

    for (uint k=0; k < source_sentence[s].size(); k++)
      nSourceWords = std::max(nSourceWords,source_sentence[s][k]+1);

    for (uint k=0; k < target_sentence[s].size(); k++)
      nTargetWords = std::max(nTargetWords,target_sentence[s][k]+1);
  }

  CooccuringWordsType wcooc(MAKENAME(wcooc));
  CooccuringLengthsType lcooc(MAKENAME(lcooc));
  SingleWordDictionary dict(MAKENAME(dict));
  IBM2AlignmentModel ibm2align_model(MAKENAME(ibm2align_model));
  ReducedIBM2AlignmentModel reduced_ibm2align_model(MAKENAME(reduced_ibm2align_model));
  FullHMMAlignmentModel hmmalign_model(MAKENAME(hmmalign_model));
  InitialAlignmentProbability initial_prob(MAKENAME(initial_prob));

  timeval startProcess, endProcess;
  gettimeofday(&startProcess,0);

  std::cerr << "finding cooccuring words" << std::endl;
  find_cooccuring_words(source_sentence, target_sentence, nSourceWords, nTargetWords, wcooc);
  
  std::cerr << "generating lookup table" << std::endl;
  Storage1D<Math2D::Matrix<uint> > slookup;
  generate_wordlookup(source_sentence, target_sentence, wcooc, slookup);

  double dict_regularity = convert<double>(app.getParam("-dict-regularity"));
    
  floatSingleWordDictionary prior_weight(nTargetWords, MAKENAME(prior_weight));
      
  if (app.is_set("-ibm3-iter"))
    ibm3_em_iter = convert<uint>(app.getParam("-ibm3-iter"));
  if (app.is_set("-ibm4-iter"))
    ibm4_em_iter = convert<uint>(app.getParam("-ibm4-iter"));

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
      for (uint s=0; s < target_sentence.size(); s++) {
	
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
	       sure_ref_alignments, possible_ref_alignments, prior_weight);

  }
  else if (method == "gd") {

    train_ibm1_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords, dict, ibm1_ter,
			      sure_ref_alignments, possible_ref_alignments, prior_weight); 
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
			 nSourceWords, nTargetWords, reduced_ibm2align_model, dict, ibm2_em_iter,
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
		       hmmalign_model, initial_prob, dict, hmm_iter, HmmInitPar, HmmAlignProbReducedpar,
		       sure_ref_alignments, possible_ref_alignments, prior_weight);
  }
  else if (method == "gd") {

    train_extended_hmm_gd_stepcontrol(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords,
				      hmmalign_model, initial_prob, dict, 50, HmmInitPar, HmmAlignProbReducedpar,
				      sure_ref_alignments, possible_ref_alignments, prior_weight); 
  }
  else {

    viterbi_train_extended_hmm(source_sentence, slookup, target_sentence, wcooc, nSourceWords, nTargetWords,
			       hmmalign_model, initial_prob, dict, hmm_iter, HmmInitFix, 
			       HmmAlignProbFullpar, false,  
			       sure_ref_alignments, possible_ref_alignments, prior_weight);
  }
  
  /*** IBM-3 ***/

  std::cerr << "handling IBM-3" << std::endl;

  IBM3Trainer ibm3_trainer(source_sentence, slookup, target_sentence, 
			   sure_ref_alignments, possible_ref_alignments,
			   dict, wcooc, nSourceWords, nTargetWords, prior_weight, 
			   true, true, false, l0_fertpen);
  
  ibm3_trainer.init_from_hmm(hmmalign_model,initial_prob);
  if (method == "em" || method == "gd") {
    ibm3_trainer.train_unconstrained(ibm3_iter);
    //ibm3_trainer.train_with_itg_constraints(1,true);
    //ibm3_trainer.train_with_ibm_constraints(1,5,4);
  }
  else
    ibm3_trainer.train_viterbi(ibm3_iter,false);
  
  if (ibm3_iter > 0) {
    ibm3_trainer.update_alignments_unconstrained();
  }

  /*** IBM-4 ***/

  std::cerr << "handling IBM-4" << std::endl;
  
  IBM4Trainer ibm4_trainer(source_sentence, slookup, target_sentence, 
			   sure_ref_alignments, possible_ref_alignments,
			   dict, wcooc, nSourceWords, nTargetWords, prior_weight, true, true, true,
			   IBM4FIRST);
  ibm4_trainer.init_from_ibm3(ibm3_trainer);
  //ibm4_trainer.update_alignments_unconstrained();
  
  if (!app.is_set("-viterbi"))
    ibm4_trainer.train_viterbi(ibm4_iter);
  else
    ibm4_trainer.train_unconstrained(ibm4_iter);

  /*** write alignments ***/
  if (ibm4_iter > 0) {
    ibm4_trainer.write_alignments(app.getParam("-oa"));
  }
  else if (ibm3_iter > 0) {
    ibm3_trainer.write_alignments(app.getParam("-oa"));
  }
  else {

    std::ofstream alignment_stream(app.getParam("-oa").c_str());

    Storage1D<uint> viterbi_alignment;

    for (uint s = 0; s < nSentences; s++) {

      if (hmm_iter > 0) {
	
	compute_ehmm_viterbi_alignment(source_sentence[s],slookup[s], target_sentence[s], 
				       dict, hmmalign_model[curI-1], initial_prob[curI-1], viterbi_alignment);
      }
      else if (ibm2_iter > 0) {
	
	const uint curI = target_sentence[s].size();
	const Math2D::Matrix<double>& cur_align_model = ibm2align_model[curI];

	compute_ibm2_viterbi_alignment(source_sentence[s], slookup[s], target_sentence[s], dict, 
				       cur_align_model, viterbi_alignment);
      }
      else {

	compute_ibm1_viterbi_alignment(source_sentence[s], slookup[s], target_sentence[s], dict, viterbi_alignment);
      }


      for (uint j=0; j < viterbi_alignment.size(); j++) 
	alignment_stream << viterbi_alignment[j] << " ";
      alignment_stream << std::endl;
    }

    alignment_stream.close();
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
