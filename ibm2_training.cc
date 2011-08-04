/*** written by Thomas Schoenemann as a private person without employment, October 2009 
 *** and later by Thomas Schoenemann as employee of Lund University, 2010 ***/


#include "ibm2_training.hh"

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "matrix.hh"

#include "training_common.hh"
#include "alignment_error_rate.hh"
#include "alignment_computation.hh"
#include "projection.hh"

double ibm2_perplexity( const Storage1D<Storage1D<uint> >& source,
			const Storage1D<Math2D::Matrix<uint> >& slookup,
			const Storage1D< Storage1D<uint> >& target,
			const IBM2AlignmentModel& align_model,
			const SingleWordDictionary& dict) {

  std::cerr << "calculating IBM 2 perplexity" << std::endl;

  double sum = 0.0;

  const uint nSentences = target.size();
  assert(slookup.size() == nSentences);

  for (uint s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const Math2D::Matrix<uint>& cur_lookup = slookup[s]; 
    
    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    uint k=0;
    for (; k < align_model[curI].size(); k++) {
      if (align_model[curI][k].xDim() == curJ)
	break;
    }

    assert(k < align_model[curI].size());
    const Math2D::Matrix<double>& cur_align_model = align_model[curI][k];

    for (uint j=0; j < curJ; j++) {

      uint s_idx = cur_source[j];
      double cur_sum = cur_align_model(j,0)*dict[0][s_idx-1];

      for (uint i=0; i < curI; i++) {
	uint t_idx = cur_target[i];
	cur_sum += cur_align_model(j,i+1)*dict[t_idx][cur_lookup(j,i)];
      }
      sum -= std::log(cur_sum);
    }
  }

  return sum / nSentences;
}

void train_ibm2(const Storage1D<Storage1D<uint> >& source, 
		const Storage1D<Math2D::Matrix<uint> >& slookup,
		const Storage1D<Storage1D<uint> >& target,
		const CooccuringWordsType& wcooc,
		const CooccuringLengthsType& lcooc,
		uint nSourceWords, uint nTargetWords,
		IBM2AlignmentModel& alignment_model,
		SingleWordDictionary& dict,
		uint nIterations,
		std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
		std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments) {

  std::cerr << "starting IBM 2 training" << std::endl;

  assert(wcooc.size() == nTargetWords);
  //NOTE: the dicitionary is assumed to be initialized

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  //initialize alignment model
  alignment_model.resize_dirty(lcooc.size());
  for (uint I=0; I < lcooc.size(); I++) {

    alignment_model[I].resize_dirty(lcooc[I].size());

    for (uint k=0; k < lcooc[I].size(); k++) {
      uint J = lcooc[I][k];

      alignment_model[I][k].resize_dirty(J,I+1);
      alignment_model[I][k].set_constant(1.0 / (I+1));
    }
  }
  
  Storage1D<Math1D::Vector<double> > fwcount(nTargetWords);
  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(wcooc[i].size());
  }
  
  IBM2AlignmentModel facount(alignment_model.size(),MAKENAME(facount));
  for (uint I=0; I < lcooc.size(); I++) {
    uint cur_length = lcooc[I].size();
    facount[I].resize_dirty(cur_length);
    
    for (uint k=0; k < lcooc[I].size(); k++) {
      uint J = lcooc[I][k];

      facount[I][k].resize_dirty(J,I+1);
    }
  }
    
  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting IBM 2 iteration #" << iter << std::endl;

    //set counts to 0
    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
    }
    for (uint I=0; I < lcooc.size(); I++) {
      uint cur_length = lcooc[I].size();
      
      for (uint k=0; k < cur_length; k++) {
	facount[I][k].set_constant(0.0);
      }
    }
    
    for (uint s=0; s < nSentences; s++) {
      
      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      uint k=0;
      for(; k < alignment_model[curI].size(); k++) {
	if (alignment_model[curI][k].xDim() == curJ)
	  break;
      }

      assert(k < alignment_model[curI].size());
      const Math2D::Matrix<double>& cur_align_model = alignment_model[curI][k];
      const Math2D::Matrix<double>& cur_facount= facount[curI][k];

      for (uint j=0; j < curJ; j++) {

	const uint s_idx = cur_source[j];

	double coeff = dict[0][s_idx-1]*cur_align_model(j,0);

	for (uint i=0; i < curI; i++) {
	  const uint t_idx = cur_target[i];
	  coeff += dict[t_idx][slookup[s](j,i)] * cur_align_model(j,i+1);
	}

	coeff = 1.0 / coeff;
	assert(!isnan(coeff));

	double addon;
	addon = coeff*dict[0][s_idx-1]*cur_align_model(j,0);

	fwcount[0][s_idx-1] += addon;
	cur_facount(j,0) += addon;

	for (uint i=0; i < curI; i++) {
	  const uint t_idx = cur_target[i];
	  const uint l = slookup[s](j,i);

	  addon = coeff*dict[t_idx][l]*cur_align_model(j,i+1);
	  
	  //update dict
	  fwcount[t_idx][l] += addon;
	  
	  //update alignment
	  cur_facount(j,i+1) += addon;
	}
      }
    }

    //compute new dict from normalized fractional counts
    for (uint i=0; i < nTargetWords; i++) {

      const double sum = fwcount[i].sum();
      if (sum > 1e-307) {
	const double inv_sum = 1.0 / sum;
      
	if (isnan(inv_sum)) {
	  std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
	  std::cerr << "sum = " << fwcount[i].sum() << std::endl;
	  std::cerr << "number of cooccuring source words: " << fwcount[i].size() << std::endl;
	}
	
	assert(!isnan(inv_sum));
	
	for (uint k=0; k < fwcount[i].size(); k++) {
	  dict[i][k] = fwcount[i][k] * inv_sum;
	}
      }
      else {
	std::cerr << "WARNING : did not update dictionary entries because sum is " << sum << std::endl;
      }
    }

    //compute new alignment model from normalized fractional counts
    for (uint I=0; I < alignment_model.size(); I++) {

      for (uint k=0; k < alignment_model[I].size(); k++) {
	uint J = alignment_model[I][k].xDim();
	assert(alignment_model[I][k].yDim() == (I+1));

	for (uint j=0; j < J; j++) {

	  double sum = 0.0;
	  for (uint i=0; i <= I; i++)
	    sum += facount[I][k](j,i);

	  if ( sum > 1e-307) {
	    sum = 1.0 / sum;
	    assert(!isnan(sum));
	    
	    for (uint i=0; i <= I; i++)
	      alignment_model[I][k](j,i) = sum * facount[I][k](j,i);
	  }
	  else {
	    std::cerr << "WARNING : did not update alignment prob because sum is " << sum << std::endl;
	  }
	}
      }
    }
    
    std::cerr << "IBM 2 perplexity after iteration #" << iter << ": "
	      << ibm2_perplexity(source, slookup, target, alignment_model, dict)
	      << std::endl;


    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (uint s=0; s < nSentences; s++) {

	if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

	  nContributors++;

	  const uint curJ = source[s].size();
	  const uint curI = target[s].size();
	  
	  uint k=0;
	  for(; k < alignment_model[curI].size(); k++) {
	    if (alignment_model[curI][k].xDim() == curJ)
	      break;
	  }
	  
	  assert(k < alignment_model[curI].size());
	  const Math2D::Matrix<double>& cur_align_model = alignment_model[curI][k];

	  //compute viterbi alignment
	  Storage1D<uint> viterbi_alignment;
	  compute_ibm2_viterbi_alignment(source[s], slookup[s], target[s], dict, cur_align_model,
					 viterbi_alignment);
  
	  //add alignment error rate
	  sum_aer += AER(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	  sum_fmeasure += f_measure(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	  nErrors += nDefiniteAlignmentErrors(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	}
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


double reduced_ibm2_perplexity( const Storage1D<Storage1D<uint> >& source,
				const Storage1D<Math2D::Matrix<uint> >& slookup,
				const Storage1D< Storage1D<uint> >& target,
				const ReducedIBM2AlignmentModel& align_model,
				const SingleWordDictionary& dict) {

  std::cerr << "calculating reduced IBM 2 perplexity" << std::endl;

  double sum = 0.0;

  const uint nSentences = target.size();
  assert(slookup.size() == nSentences);

  for (uint s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];
    const Math2D::Matrix<uint>& cur_lookup = slookup[s]; 
    
    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    const Math2D::Matrix<double>& cur_align_model = align_model[curI];

    for (uint j=0; j < curJ; j++) {

      uint s_idx = cur_source[j];
      double cur_sum = cur_align_model(j,0)*dict[0][s_idx-1];

      for (uint i=0; i < curI; i++) {
	uint t_idx = cur_target[i];
	cur_sum += cur_align_model(j,i+1)*dict[t_idx][cur_lookup(j,i)];
      }
      sum -= std::log(cur_sum);
    }
  }

  return sum / nSentences;
}


void train_reduced_ibm2(const Storage1D<Storage1D<uint> >& source,
			const Storage1D<Math2D::Matrix<uint> >& slookup,
			const Storage1D<Storage1D<uint> >& target,
			const CooccuringWordsType& wcooc,
			const CooccuringLengthsType& lcooc,
			uint nSourceWords, uint nTargetWords,
			ReducedIBM2AlignmentModel& alignment_model,
			SingleWordDictionary& dict,
			uint nIterations,
			std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
			std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments) {

  std::cerr << "starting reduced IBM 2 training" << std::endl;

  assert(wcooc.size() == nTargetWords);
  //NOTE: the dicitionary is assumed to be initialized

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  //initialize alignment model
  alignment_model.resize_dirty(lcooc.size());
  for (uint I=0; I < lcooc.size(); I++) {

    uint maxJ = 0;
    for (uint k=0; k < lcooc[I].size(); k++) {
      uint curJ = lcooc[I][k];
      if (curJ > maxJ)
	maxJ = curJ;
    }

    if (maxJ > 0) {
      alignment_model[I].resize_dirty(maxJ,I+1);
      alignment_model[I].set_constant(1.0/(I+1));
    }
  }

  //TODO: estimate first alignment model from IBM1 dictionary
  
  Storage1D<Math1D::Vector<double> > fwcount(nTargetWords);
  for (uint i=0; i < nTargetWords; i++) {
    fwcount[i].resize(wcooc[i].size());
  }


  ReducedIBM2AlignmentModel facount(alignment_model.size(),MAKENAME(facount));
  for (uint I=0; I < lcooc.size(); I++) {
    uint maxJ = alignment_model[I].xDim();
    if (maxJ > 0) 
      facount[I].resize_dirty(maxJ,I+1);
  }
    
  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "starting reduced IBM 2 iteration #" << iter << std::endl;

    //set counts to 0
    for (uint i=0; i < nTargetWords; i++) {
      fwcount[i].set_constant(0.0);
    }
    for (uint I=0; I < lcooc.size(); I++) {
      if (facount[I].xDim() > 0)
	facount[I].set_constant(0.0);
    }
    
    for (uint s=0; s < nSentences; s++) {
      
      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint curJ = cur_source.size();
      const uint curI = cur_target.size();

      const Math2D::Matrix<double>& cur_align_model = alignment_model[curI];
      const Math2D::Matrix<double>& cur_facount= facount[curI];

      assert(cur_align_model.xDim() >= curJ);
      assert(cur_facount.xDim() >= curJ);


      for (uint j=0; j < curJ; j++) {

	const uint s_idx = cur_source[j];

	double coeff = dict[0][s_idx-1]*cur_align_model(j,0);

	for (uint i=0; i < curI; i++) {
	  const uint t_idx = cur_target[i];
	  coeff += dict[t_idx][slookup[s](j,i)] * cur_align_model(j,i+1);
	}

	coeff = 1.0 / coeff;
	assert(!isnan(coeff));

	double addon;
	addon = coeff*dict[0][s_idx-1]*cur_align_model(j,0);

	fwcount[0][s_idx-1] += addon;
	cur_facount(j,0) += addon;

	for (uint i=0; i < curI; i++) {
	  const uint t_idx = cur_target[i];
	  const uint l = slookup[s](j,i);

	  addon = coeff*dict[t_idx][l]*cur_align_model(j,i+1);
	  
	  //update dict
	  fwcount[t_idx][l] += addon;
	  
	  //update alignment
	  cur_facount(j,i+1) += addon;
	}
      }
    }
    
    //compute new dict from normalized fractional counts
    for (uint i=0; i < nTargetWords; i++) {
      double inv_sum = 1.0 / fwcount[i].sum();
      
      if (isnan(inv_sum)) {
	std::cerr << "invsum " << inv_sum << " for target word #" << i << std::endl;
	std::cerr << "sum = " << fwcount[i].sum() << std::endl;
	std::cerr << "number of cooccuring source words: " << fwcount[i].size() << std::endl;
      }
      
      assert(!isnan(inv_sum));

      for (uint k=0; k < fwcount[i].size(); k++) {
	dict[i][k] = fwcount[i][k] * inv_sum;
      }
    }

    //compute new alignment model from normalized fractional counts
    for (uint I=0; I < alignment_model.size(); I++) {

      uint J = alignment_model[I].xDim();
      if (J > 0) {
	assert(alignment_model[I].yDim() == (I+1));
	
	for (uint j=0; j < J; j++) {
	  
	  double sum = 0.0;
	  for (uint i=0; i <= I; i++)
	    sum += facount[I](j,i);
	  
	  sum = 1.0 / sum;
	  assert(!isnan(sum));
	  
	  for (uint i=0; i <= I; i++)
	    alignment_model[I](j,i) = sum * facount[I](j,i);
	}
      }
    }

    std::cerr << "reduced IBM 2 perplexity after iteration #" << iter << ": "
	      << reduced_ibm2_perplexity(source, slookup, target, alignment_model, dict)
	      << std::endl;    

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;

      for (uint s=0; s < nSentences; s++) {

	if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

	  nContributors++;

	  const uint curI = target[s].size();
	  const Math2D::Matrix<double>& cur_align_model = alignment_model[curI];

	  //compute viterbi alignment
	  Storage1D<uint> viterbi_alignment;
	  compute_ibm2_viterbi_alignment(source[s], slookup[s], target[s], dict, cur_align_model,
					 viterbi_alignment);
  
	  //add alignment error rate
	  sum_aer += AER(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	  sum_fmeasure += f_measure(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	  nErrors += nDefiniteAlignmentErrors(viterbi_alignment,sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	}
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### ReducedIBM2 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### ReducedIBM2 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;
    }

  }
}


void ibm2_viterbi_training(const Storage1D<Storage1D<uint> >& source, 
			   const Storage1D<Math2D::Matrix<uint> >& slookup,
			   const Storage1D<Storage1D<uint> >& target,
			   const CooccuringWordsType& wcooc,
			   const CooccuringLengthsType& lcooc,
			   uint nSourceWords, uint nTargetWords,
			   ReducedIBM2AlignmentModel& alignment_model,
			   SingleWordDictionary& dict,
			   uint nIterations,
			   std::map<uint,std::set<std::pair<uint,uint> > >& sure_ref_alignments,
			   std::map<uint,std::set<std::pair<uint,uint> > >& possible_ref_alignments,
			   const floatSingleWordDictionary& prior_weight) {

  //initialize alignment model
  alignment_model.resize_dirty(lcooc.size());
  ReducedIBM2AlignmentModel acount(lcooc.size(),MAKENAME(acount));

  for (uint I=0; I < lcooc.size(); I++) {

    uint maxJ = 0;
    for (uint k=0; k < lcooc[I].size(); k++) {
      uint curJ = lcooc[I][k];
      if (curJ > maxJ)
	maxJ = curJ;
    }

    if (maxJ > 0) {
      alignment_model[I].resize_dirty(maxJ,I+1);
      alignment_model[I].set_constant(1.0/(I+1));
      acount[I].resize_dirty(maxJ,I+1);
    }
  }

  const uint nSentences = source.size();
  assert(nSentences == target.size());
  
  Storage1D<Math1D::Vector<uint> > viterbi_alignment(source.size());

  for (uint s=0; s < nSentences; s++) {
    
    const Storage1D<uint>& cur_source = source[s];
    
    viterbi_alignment[s].resize(cur_source.size());
  }
  
  //fractional counts used for EM-iterations
  NamedStorage1D<Math1D::Vector<double> > dcount(nTargetWords,MAKENAME(dcount));

  for (uint i=0; i < nTargetWords; i++) {
    dcount[i].resize(wcooc[i].size());
    dcount[i].set_constant(0);
  }

  Math1D::NamedVector<uint> prev_wsum(nTargetWords,0,MAKENAME(prev_wsum));  

  for (uint iter = 1; iter <= nIterations; iter++) {

    std::cerr << "###iter " << iter << std::endl;

    for (uint i=0; i < nTargetWords; i++) {      
      dcount[i].set_constant(0);
    }

    for (uint I=0; I < acount.size(); I++) 
      acount[I].set_constant(0.0);

    double sum = 0.0;

    for (uint s=0; s < nSentences; s++) {

      //std::cerr << "s: " << s << std::endl;

      const Storage1D<uint>& cur_source = source[s];
      const Storage1D<uint>& cur_target = target[s];

      const uint nCurSourceWords = cur_source.size();
      const uint nCurTargetWords = cur_target.size();
      const Math2D::Matrix<uint>& cur_lookup = slookup[s];

      const Math2D::Matrix<double>& cur_align_model = alignment_model[nCurTargetWords];
      
      for (uint j=0; j < nCurSourceWords; j++) {
	
	const uint s_idx = source[s][j];

	double min = 1e50;
	uint arg_min = MAX_UINT;

	if (iter == 1) {

	  min = -std::log(dict[0][s_idx-1])* cur_align_model(j,0);
	  arg_min = 0;

	  for (uint i=0; i < nCurTargetWords; i++) {

	    double hyp = -std::log(dict[cur_target[i]][cur_lookup(j,i)]*cur_align_model(j,i+1) );

	    //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;
	    
	    if (hyp < min) {
	      min = hyp;
	      arg_min = i+1;
	    }
	  }
	}
	else {
	  
	  if (dict[0][s_idx-1] == 0.0 || cur_align_model(j,0) == 0.0) {
	
	    min = 1e20;
	  }
	  else {
	    
	    min = -std::log(dict[0][s_idx-1] * cur_align_model(j,0) );
	  }
	  arg_min = 0;
	  
	  for (uint i=0; i < nCurTargetWords; i++) {
	    
	    double hyp;
	    
	    if (dict[cur_target[i]][cur_lookup(j,i)] == 0.0 || cur_align_model(j,i+1) == 0) {
	      
	      hyp = 1e20;
	    }
	    else {
	      
	      hyp = -std::log( dict[cur_target[i]][cur_lookup(j,i)] * cur_align_model(j,i+1));
	    }
	    
	    //std::cerr << "hyp: " << hyp << ", min: " << min << std::endl;

	    if (hyp < min) {
	      min = hyp;
	      arg_min = i+1;
	    }
	    
	  }
	}

	//std::cerr << "arg_min: " << arg_min << std::endl;

	sum += min;
	
	viterbi_alignment[s][j] = arg_min;

	if (arg_min == 0) {
	  dcount[0][s_idx-1]++;
	  acount[nCurTargetWords](j,0)++;
	}
	else {
	  dcount[cur_target[arg_min-1]][cur_lookup(j,arg_min-1)]++;
	  acount[nCurTargetWords](j,arg_min)++;
	}
      }
    }
   
    //std::cerr << "sum: " << sum << std::endl;

    /*** ICM phase ***/

    if (true) {
      uint nSwitches = 0;

      Math1D::Vector<double> dict_sum(dcount.size());
      for (uint k=0; k < dcount.size(); k++)
	dict_sum[k] = dcount[k].sum();


      Math1D::Vector<double> w_cur_contrib(nTargetWords,0.0);
      Math1D::Vector<double> w_inc_contrib(nTargetWords,0.0);
      Math1D::Vector<double> w_dec_contrib(nTargetWords,0.0);

      for (uint w=0; w < w_cur_contrib.size(); w++) {

	for (uint k=0; k < dcount[w].size(); k++) {
	  
	  if (dcount[w][k] > 0) {
	    w_cur_contrib[w] += dcount[w][k] * (-std::log(dcount[w][k] / dict_sum[w] ));
	    w_inc_contrib[w] += dcount[w][k] * (-std::log(dcount[w][k] / (dict_sum[w]+1.0) ));
	    if (dict_sum[w] > 1.0)
	      w_dec_contrib[w] += dcount[w][k] * (-std::log(dcount[w][k] / (dict_sum[w]-1.0) ));
	  }
	}
      }
      
      for (uint s=0; s < nSentences; s++) {

	const Storage1D<uint>& cur_source = source[s];
	const Storage1D<uint>& cur_target = target[s];
	
	const uint curJ = source[s].size();
	const uint curI = target[s].size();
	
	const Math2D::Matrix<uint>& cur_lookup = slookup[s];
	const Math2D::Matrix<double>& cur_acount = acount[curI];
	
	for (uint j=0; j < curJ; j++) {
	  
	  for (uint i=0; i <= curI; i++) {
	    
	    //note: cur_aj can change during the loop over i
	    uint cur_aj = viterbi_alignment[s][j];
	    
	    if (i != cur_aj) {
	      
	      uint cur_dict_num = (cur_aj == 0) ? 0 : target[s][cur_aj-1];
	      
	      Math1D::Vector<double>& cur_dictcount = dcount[cur_dict_num]; 
	      double cur_dictsum = dict_sum[cur_dict_num]; 
	      
	      uint cur_idx = (cur_aj == 0) ? source[s][j]-1 : cur_lookup(j,cur_aj-1);
	      uint hyp_dict_num = (i == 0) ? 0 : target[s][i-1];
	      
	      Math1D::Vector<double>& hyp_dictcount = dcount[hyp_dict_num];
	      double hyp_dictsum = dict_sum[hyp_dict_num]; 
	    
	      uint hyp_idx = (i == 0) ? source[s][j]-1 : cur_lookup(j,i-1);;
	      
	      double change = 0.0;

	      uint cur_target_word = (cur_aj == 0) ? 0 : cur_target[cur_aj-1];
	      uint new_target_word = (i == 0) ? 0 : cur_target[i-1];

	      assert(cur_acount(j,cur_aj) > 0);
	      
	      if (cur_acount(j,i) != 0)
		change -= -cur_acount(j,i) * std::log(cur_acount(j,i));
	      change -= -cur_acount(j,cur_aj) * std::log(cur_acount(j,cur_aj));
	      
	      change += -(cur_acount(j,i)+1) * std::log(cur_acount(j,i)+1);
	      if (cur_acount(j,cur_aj) > 1)
		change += -(cur_acount(j,cur_aj)-1) * std::log(cur_acount(j,cur_aj)-1);

	      assert(!isnan(change));

	      if (cur_target_word != new_target_word) {
		change -= w_cur_contrib[cur_target_word];
		
		if (cur_dictsum > 1.0) {
		  change += w_dec_contrib[cur_target_word];
		  change -= (cur_dictcount[cur_idx]) * (-std::log((cur_dictcount[cur_idx]) / (cur_dictsum-1.0)));
		  
		  if (cur_dictcount[cur_idx] > 1) {
		    change += (cur_dictcount[cur_idx]-1) * (-std::log((cur_dictcount[cur_idx]-1) / (cur_dictsum-1.0)));
		  }
		  else
		    change -= prior_weight[cur_dict_num][cur_idx];
		}
		
		change -= w_cur_contrib[new_target_word]; 
		change += w_inc_contrib[new_target_word]; 
		
		if (dcount[new_target_word][hyp_idx] > 0)
		  change -= dcount[new_target_word][hyp_idx] * 
		    (-std::log(dcount[new_target_word][hyp_idx] / (dict_sum[new_target_word]+1.0)));
		else
		  change += prior_weight[cur_dict_num][cur_idx]; 
		change += (dcount[new_target_word][hyp_idx]+1) * 
		  (-std::log((dcount[new_target_word][hyp_idx]+1.0) / (dict_sum[new_target_word]+1.0)));
	      }

	      assert(!isnan(change));
	      
	      if (change < -1e-2) {
		
		nSwitches++;

		uint prev_word = cur_target_word; 
		uint new_word = new_target_word;
		
		viterbi_alignment[s][j] = i;

		if (cur_target_word != new_target_word) {

		  cur_dictcount[cur_idx] -= 1.0;
		  hyp_dictcount[hyp_idx] += 1.0;
		  dict_sum[cur_dict_num] -= 1.0;
		  dict_sum[hyp_dict_num] += 1.0;
		  
		  //recompute the stored values for the two affected words
		  w_cur_contrib[new_word] = 0.0;
		  w_inc_contrib[new_word] = 0.0; 
		  w_dec_contrib[new_word] = 0.0; 
		  for (uint k=0; k < dcount[new_word].size(); k++) {
		    
		    if (dcount[new_word][k] > 0) {
		      w_cur_contrib[new_word] += dcount[new_word][k] * 
			(-std::log(dcount[new_word][k] / dict_sum[new_word] ));
		      w_inc_contrib[new_word] += dcount[new_word][k] * 
			(-std::log(dcount[new_word][k] / (dict_sum[new_word]+1.0) ));
		      if (dict_sum[new_word] > 1.0)
			w_dec_contrib[new_word] += dcount[new_word][k] * 
			  (-std::log(dcount[new_word][k] / (dict_sum[new_word]-1.0) ));
		    }
		  }
		  w_cur_contrib[prev_word] = 0.0;
		  w_inc_contrib[prev_word] = 0.0; 
		  w_dec_contrib[prev_word] = 0.0; 
		  for (uint k=0; k < dcount[prev_word].size(); k++) {
		    
		    if (dcount[prev_word][k] > 0) {
		      w_cur_contrib[prev_word] += dcount[prev_word][k] * 
			(-std::log(dcount[prev_word][k] / dict_sum[prev_word] ));
		      w_inc_contrib[prev_word] += dcount[prev_word][k] * 
			(-std::log(dcount[prev_word][k] / (dict_sum[prev_word]+1.0) ));
		      if (dict_sum[prev_word] > 1.0)
			w_dec_contrib[prev_word] += dcount[prev_word][k] * 
			  (-std::log(dcount[prev_word][k] / (dict_sum[prev_word]-1.0) ));
		      
		    }
		  }
		}

		acount[curI](j,cur_aj)--;
		acount[curI](j,i)++;		
	      }
	      
	    }
	  }
	}
      }
      
      std::cerr << nSwitches << " switches in ICM" << std::endl;
    }

    Math1D::Vector<uint> count_count(6,0);

    for (uint i=0; i < nTargetWords; i++) {      
      for (uint k=0; k < dcount[i].size(); k++) {
	if (dcount[i][k] < count_count.size())
	  count_count[dcount[i][k]]++;
      }
    }

    std::cerr << "count count (lower end): " << count_count << std::endl;

    /*** recompute the dictionary ***/
    double energy = 0.0;

    double sum_sum = 0.0;

    for (uint i=0; i < nTargetWords; i++) {

      //std::cerr << "i: " << i << std::endl;

      const double sum = dcount[i].sum();
      prev_wsum[i] = (uint) sum;

      sum_sum += sum;

      if (sum > 1e-307) {

	energy += sum * std::log(sum);

	const double inv_sum = 1.0 / sum;
	assert(!isnan(inv_sum));
	
	for (uint k=0; k < dcount[i].size(); k++) {
	  dict[i][k] = dcount[i][k] * inv_sum;

	  if (dcount[i][k] > 0) {
	    energy -= dcount[i][k] * std::log(dcount[i][k]);
	    energy += prior_weight[i][k]; //dict_penalty; 
	  }
	}
      }
      else {
	//std::cerr << "WARNING : did not update dictionary entries because sum is " << sum << std::endl;
      }
    }


    //compute new alignment model from normalized fractional counts
    for (uint I=0; I < alignment_model.size(); I++) {

      uint J = alignment_model[I].xDim();
      if (J > 0) {
	assert(alignment_model[I].yDim() == (I+1));
	
	for (uint j=0; j < J; j++) {
	  
	  double sum = 0.0;
	  for (uint i=0; i <= I; i++)
	    sum += acount[I](j,i);

	  if (sum > 1e-305) {
	    double inv_sum = 1.0 / sum;
	    assert(!isnan(sum));
	    
	    for (uint i=0; i <= I; i++) {
	      alignment_model[I](j,i) = inv_sum * acount[I](j,i);
	      
	      if (acount[I](j,i) > 0.0)
		energy -= acount[I](j,i) * std::log( acount[I](j,i) / sum );
	    }
	  }
	}
      }
    }
    
    //std::cerr << "number of total alignments: " << sum_sum << std::endl;
    std::cerr << "energy: " << energy << std::endl;

    /************* compute alignment error rate ****************/
    if (!possible_ref_alignments.empty()) {
      
      double sum_aer = 0.0;
      double sum_fmeasure = 0.0;
      double nErrors = 0.0;
      uint nContributors = 0;


      for (uint s=0; s < nSentences; s++) {

	if (possible_ref_alignments.find(s+1) != possible_ref_alignments.end()) {

	  nContributors++;

	  //add alignment error rate
	  sum_aer += AER(viterbi_alignment[s],sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	  sum_fmeasure += f_measure(viterbi_alignment[s],sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	  nErrors += nDefiniteAlignmentErrors(viterbi_alignment[s],sure_ref_alignments[s+1],possible_ref_alignments[s+1]);
	}
      }

      sum_aer *= 100.0 / nContributors;
      sum_fmeasure /= nContributors;
      nErrors /= nContributors;

      std::cerr << "#### IBM2 Viterbi-AER after iteration #" << iter << ": " << sum_aer << " %" << std::endl;
      std::cerr << "#### IBM2 Viterbi-fmeasure after iteration #" << iter << ": " << sum_fmeasure << std::endl;
      std::cerr << "#### IBM2 Viterbi-DAE/S after iteration #" << iter << ": " << nErrors << std::endl;

    }
  }
}

