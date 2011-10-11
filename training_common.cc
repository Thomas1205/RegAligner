/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "training_common.hh"

#include <vector>
#include <set>
#include <map>
#include <algorithm>

void find_cooccuring_words(const Storage1D<Storage1D<uint> >& source, 
			   const Storage1D<Storage1D<uint> >& target,
			   uint nSourceWords, uint nTargetWords,
			   CooccuringWordsType& cooc) {
  
  Storage1D<Storage1D<uint> > additional_source;
  Storage1D<Storage1D<uint> > additional_target;

  find_cooccuring_words(source,target,additional_source,additional_target,nSourceWords,nTargetWords,cooc);
}


void find_cooccuring_words(const Storage1D<Storage1D<uint> >& source, 
			   const Storage1D<Storage1D<uint> >& target,
			   const Storage1D<Storage1D<uint> >& additional_source, 
			   const Storage1D<Storage1D<uint> >& additional_target,
			   uint nSourceWords, uint nTargetWords,
			   CooccuringWordsType& cooc) {

  NamedStorage1D<std::set<uint> > coocset(nTargetWords,MAKENAME(coocset));
  for (uint j=0; j < nSourceWords-1; j++)
    coocset[0].insert(j);

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  cooc.resize_dirty(nTargetWords);

  for (uint s=0; s < nSentences; s++) {

    if ((s%10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;
    
    const uint nCurSourceWords = source[s].size();
    const uint nCurTargetWords = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i=0; i < nCurTargetWords; i++) {
      uint t_idx = cur_target[i]; 
      assert(t_idx < nTargetWords);

      std::set<uint>& cur_set = coocset[t_idx];

      for (uint j=0; j < nCurSourceWords; j++) {
	uint s_idx = cur_source[j]; 
	assert(s_idx < nSourceWords);
	
	cur_set.insert(s_idx);
      }
    }
  }

  const uint nAddSentences = additional_source.size();
  assert(nAddSentences == additional_target.size());

  for (uint s=0; s < nAddSentences; s++) {

    if ((s%10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;
    
    const uint nCurSourceWords = additional_source[s].size();
    const uint nCurTargetWords = additional_target[s].size();

    const Storage1D<uint>& cur_source = additional_source[s];
    const Storage1D<uint>& cur_target = additional_target[s];

    for (uint i=0; i < nCurTargetWords; i++) {
      uint t_idx = cur_target[i]; 
      assert(t_idx < nTargetWords);

      std::set<uint>& cur_set = coocset[t_idx];

      for (uint j=0; j < nCurSourceWords; j++) {
	uint s_idx = cur_source[j]; 
	assert(s_idx < nSourceWords);
	
	cur_set.insert(s_idx);
      }
    }
  }

  for (uint i=0; i < nTargetWords; i++) {
    
    const uint size = coocset[i].size();
    cooc[i].resize_dirty(size);
    uint j=0;
    for (std::set<uint>::iterator it = coocset[i].begin(); it != coocset[i].end(); it++, j++)
      cooc[i][j] = *it;
    
    coocset[i].clear();
  }

  double sum_cooc = 0.0;
  for (uint i=0; i < nTargetWords; i++) {
    sum_cooc += cooc[i].size();
  }
  
  std::cerr << "average number of cooccuring words: " << (sum_cooc / nTargetWords) << std::endl;
}


void find_cooc_monolingual_pairs(const Storage1D<Storage1D<uint> >& sentence,
				 uint voc_size, Storage1D<Storage1D<uint> >& cooc) {


  const uint nSentences = sentence.size();
  cooc.resize(voc_size);
  for (uint k=0; k < voc_size; k++)
    cooc[k].resize(0);

  cooc[0].resize(voc_size);
  for (uint k=0; k < voc_size; k++)
    cooc[0][k] = k;

  for (uint s=0; s < nSentences; s++) {

    //std::cerr << "s: " << s << std::endl;

    const Storage1D<uint>& cur_sentence = sentence[s];

    const uint curI = cur_sentence.size();

    for (uint i1=0; i1 < curI-1; i1++) {
      
      uint w1 =  cur_sentence[i1];

      for (uint i2 = i1+1; i2 < curI; i2++) {

	uint w2 =  cur_sentence[i2];

	uint l=0;
	for ( ; l < cooc[w1].size() && cooc[w1][l] != w2; l++) {
	  ;
	}

	if (l >= cooc[w1].size()) {
	  cooc[w1].resize(cooc[w1].size()+1);
	  cooc[w1][l] = w2;
	}
      }
    }
  }

  //finally sort
  for (uint k=1; k < voc_size; k++)
    std::sort(cooc[k].direct_access(), cooc[k].direct_access() + cooc[k].size());
}

void monolingual_pairs_cooc_count(const Storage1D<Storage1D<uint> >& sentence,
				  const Storage1D<Storage1D<uint> >&t_cooc, Storage1D<Storage1D<uint> >&t_cooc_count) {

  t_cooc_count.resize(t_cooc.size());
  for (uint k=0; k < t_cooc.size(); k++)
    t_cooc_count[k].resize(t_cooc[k].size(),0);

  uint nSentences = sentence.size();

  for (uint s=0; s < nSentences; s++) {

    //std::cerr << "s: " << s << std::endl;

    const Storage1D<uint>& cur_sentence = sentence[s];

    const uint curI = cur_sentence.size();

    for (uint i1=0; i1 < curI-1; i1++) {
      
      uint w1 =  cur_sentence[i1];

      for (uint i2 = i1+1; i2 < curI; i2++) {

	uint w2 =  cur_sentence[i2];

	uint* ptr = std::lower_bound(t_cooc[w1].direct_access(),t_cooc[w1].direct_access()+t_cooc[w1].size(), w2);

	uint k = ptr - t_cooc[w1].direct_access();
	t_cooc_count[w1][k] += 1;
      }
    }
  }
}



void find_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
					     const Storage1D<Storage1D<uint> >& target,
					     const Storage1D<Storage1D<uint> >& target_cooc,
					     Storage1D<Storage1D<Storage1D<uint> > >& st_cooc) {

  st_cooc.resize(target_cooc.size());
  for (uint i=0; i < target_cooc.size(); i++)
    st_cooc[i].resize(target_cooc[i].size());

  const uint nSentences = source.size();

  for (uint s=0; s < nSentences; s++) {

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1=0; i1 < curI; i1++) {

      uint start_i2 = (i1 == 0) ? 0 : i1+1;

      uint t1 = (i1 == 0) ? 0 : cur_target[i1-1];

      const Storage1D<uint>& cur_tcooc = target_cooc[t1];

      for (uint i2 = start_i2; i2 <= curI; i2++) {

	uint t2 = (i2 == 0) ? 0 : cur_target[i2-1];

	const uint* ptr = std::lower_bound(cur_tcooc.direct_access(), cur_tcooc.direct_access() + cur_tcooc.size(), t2);

	assert(*ptr == t2);

	assert(ptr >= cur_tcooc.direct_access());
	assert(ptr < cur_tcooc.direct_access() + cur_tcooc.size());

	uint diff = ptr - cur_tcooc.direct_access();

	Storage1D<uint>& cur_scooc = st_cooc[t1][diff];

	for (uint j=0; j < curJ; j++) {

	  uint s_idx = cur_source[j];

	  uint l=0;
	  for ( ; l < cur_scooc.size() && cur_scooc[l] != s_idx; l++) {
	    ;
	  }

	  if (l >= cur_scooc.size()) {

	    cur_scooc.resize(cur_scooc.size()+1);
	    cur_scooc[l] = s_idx;
	  }
	}
      }
    }
  }  

  //finally sort
  for (uint i=0; i < st_cooc.size(); i++)
    for (uint k=0; k < st_cooc[i].size(); k++)
      std::sort(st_cooc[i][k].direct_access(), st_cooc[i][k].direct_access() + st_cooc[i][k].size());
}


void find_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
					     const Storage1D<Storage1D<uint> >& target,
					     std::map<std::pair<uint,uint>, std::set<uint> >& cooc) {


  const uint nSentences = source.size();
  assert(nSentences == target.size());

  for (uint s=0; s < nSentences; s++) {

    if ((s%10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;
    
    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1-1];

      uint start_i2 = (i1 == 0) ? 0 : i1+1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

	const uint t2 = (i2 == 0) ? 0 : cur_target[i2-1];
	
	std::pair<uint,uint> tpair = std::make_pair(t1,t2);
	
	for (uint j=0; j < curJ; j++)
	  cooc[tpair].insert(cur_source[j]);
      }
    }
  }

}


void find_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
					     const Storage1D<Storage1D<uint> >& target,
					     uint nSourceWords, uint nTargetWords,
					     Storage1D<Storage1D<std::pair<uint,Storage1D<uint> > > >& cooc) {

  cooc.resize(nTargetWords);

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  for (uint s=0; s < nSentences; s++) {

    if ((s%10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;
    
    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1-1];

      uint start_i2 = (i1 == 0) ? 0 : i1+1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

	const uint t2 = (i2 == 0) ? 0 : cur_target[i2-1];
	
	uint t2_cooc_idx = MAX_UINT;
	for (uint k=0; k < cooc[t1].size(); k++) {
	  if (cooc[t1][k].first == t2) {
	    t2_cooc_idx = k;
	    break;
	  }
	}

	if (t2_cooc_idx == MAX_UINT) {
	  t2_cooc_idx = cooc[t1].size();
	  cooc[t1].resize(cooc[t1].size()+1);
	  cooc[t1][t2_cooc_idx].first = t2;
	}

 	for (uint j=0; j < curJ; j++) {

	  uint j_cooc_idx = MAX_UINT;
	  for (uint k=0; k < cooc[t1][t2_cooc_idx].second.size(); k++) {

	    if (cooc[t1][t2_cooc_idx].second[k] == cur_source[j]) {
	      j_cooc_idx = k;
	      break;
	    }
	  }

	  if (j_cooc_idx == MAX_UINT) {
	    j_cooc_idx = cooc[t1][t2_cooc_idx].second.size();
	    cooc[t1][t2_cooc_idx].second.resize(cooc[t1][t2_cooc_idx].second.size()+1);
	    cooc[t1][t2_cooc_idx].second[j_cooc_idx] = cur_source[j];
	  }
	}
      }
    }
  }
}


void count_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
					      const Storage1D<Storage1D<uint> >& target,
					      std::map<std::pair<uint,uint>, std::map<uint,uint> >& cooc_count) {


  const uint nSentences = source.size();
  assert(nSentences == target.size());

  for (uint s=0; s < nSentences; s++) {

    if ((s%10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;
    
    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1-1];

      uint start_i2 = (i1 == 0) ? 0 : i1+1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

	const uint t2 = (i2 == 0) ? 0 : cur_target[i2-1];
	
	std::pair<uint,uint> tpair = std::make_pair(t1,t2);
	
	for (uint j=0; j < curJ; j++)
	  cooc_count[tpair][cur_source[j]]++;
      }
    }
  }

}


bool operator<(const std::pair<uint,Storage1D<std::pair<uint,uint> > >& x1, 
	       const std::pair<uint,Storage1D<std::pair<uint,uint> > >& x2) {

//   if (x1.first == x2.first)
//     std::cerr << "identical: " << x1.first << std::endl;

  assert(x1.first != x2.first);

  return (x1.first < x2.first);
}

bool operator<(const std::pair<uint,uint>& x1, const std::pair<uint,uint>& x2) {

  if (x1.first == x2.first)
    return (x1.second < x2.second);
  else
    return (x1.first < x2.first);
}


void count_cooc_target_pairs_and_source_words(const Storage1D<Storage1D<uint> >& source, 
					      const Storage1D<Storage1D<uint> >& target,
					      uint nSourceWords, uint nTargetWords,
					      Storage1D<Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > > >& cooc) {


  cooc.resize(nTargetWords);
  cooc[0].resize(nTargetWords);
  for (uint k=0; k < nTargetWords; k++) {
    cooc[0][k].first = k;
  }

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  /**** stage 1: find coocuring target words ****/  

  for (uint s=0; s < nSentences; s++) {

    if ((s%100) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curI = target[s].size();

    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 1; i1 <= curI; i1++) {

      const uint t1 = cur_target[i1-1];

      Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > >& cur_cooc = cooc[t1];

      assert(t1 != 0);

      for (uint i2 = i1+1; i2 <= curI; i2++) {

	const uint t2 = cur_target[i2-1];
	
	uint t2_cooc_idx = MAX_UINT;
	for (uint k=0; k < cur_cooc.size(); k++) {
	  if (cur_cooc[k].first == t2) {
	    t2_cooc_idx = k;
	    break;
	  }
	}

	if (t2_cooc_idx == MAX_UINT) {
	  t2_cooc_idx = cur_cooc.size();
	  cur_cooc.resize(cooc[t1].size()+1);
	  cur_cooc[t2_cooc_idx].first = t2;
	}
      }
    }
  }
  
  /**** stage 2: sort the vectors ****/
  for (uint i=0; i < cooc.size(); i++) {
    //std::cerr << "sorting #" << i << std::endl;
    std::sort(cooc[i].direct_access(), cooc[i].direct_access() + cooc[i].size());
  }

  /**** stage 3: find coocuring source words and target pairs ****/
  for (uint s=0; s < nSentences; s++) {

    if ((s%100) == 0)
      std::cerr << "sentence pair number " << s << std::endl;
    
    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1-1];

      Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > >& cur_cooc = cooc[t1];

      uint start_i2 = (i1 == 0) ? 0 : i1+1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

	const uint t2 = (i2 == 0) ? 0 : cur_target[i2-1];
	
	uint t2_cooc_idx = MAX_UINT;
	if (i1 == 0)
	  t2_cooc_idx = t2;
	else {
	  for (uint k=0; k < cur_cooc.size(); k++) {
	    if (cur_cooc[k].first == t2) {
	      t2_cooc_idx = k;
	      break;
	    }
	  }
	}

	if (t2_cooc_idx == MAX_UINT) {
	  assert(false);
	  t2_cooc_idx = cur_cooc.size();
	  cur_cooc.resize(cur_cooc.size()+1);
	  cur_cooc[t2_cooc_idx].first = t2;
	}

	//std::pair<uint,Storage1D<std::pair<uint,uint> > >& cur_pair = cur_cooc[t2_cooc_idx];

	Storage1D<std::pair<uint,uint> >& cur_vec = cur_cooc[t2_cooc_idx].second;

 	for (uint j=0; j < curJ; j++) {

	  const uint s_idx = cur_source[j];

	  uint j_cooc_idx = MAX_UINT;
	  for (uint k=0; k < cur_vec.size(); k++) {

	    if (cur_vec[k].first == s_idx) {
	      j_cooc_idx = k;
	      break;
	    }
	  }

	  if (j_cooc_idx == MAX_UINT) {
	    j_cooc_idx = cur_vec.size();
	    cur_vec.resize(cur_vec.size()+1);
	    cur_vec[j_cooc_idx] = std::make_pair(s_idx,0);
	  }
	  cur_vec[j_cooc_idx].second++;
	}
      }
    }
  }
  
  /*** stage 4: sort the indivialual source vectors ****/
  for (uint i=0; i < cooc.size(); i++) {
    for (uint k=0; k < cooc[i].size(); k++)
      std::sort(cooc[i][k].second.direct_access(), cooc[i][k].second.direct_access() + cooc[i][k].second.size());
  }

}


void find_cooccuring_lengths(const Storage1D<Storage1D<uint> >& source, 
			     const Storage1D<Storage1D<uint> >& target,
			     CooccuringLengthsType& cooc) {

  std::map<uint, std::set<uint> > coocvec;

  const uint nSentences = target.size();
  uint max_tlength = 0;

  for (uint s=0; s < nSentences; s++) {
    uint cur_tlength = target[s].size();
    if (cur_tlength > max_tlength)
      max_tlength = cur_tlength;

    coocvec[cur_tlength].insert(source[s].size());
  }

  cooc.resize_dirty(max_tlength+1);
  for (std::map<uint, std::set<uint> >::iterator it = coocvec.begin(); it != coocvec.end(); it++) {

    const uint tlength = it->first;
    cooc[tlength].resize_dirty(it->second.size());
    uint k=0;
    for (std::set<uint>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++, k++) {
      cooc[tlength][k] = *it2;
    }
  }
}

void generate_wordlookup(const Storage1D<Storage1D<uint> >& source, 
			 const Storage1D<Storage1D<uint> >& target,
			 const CooccuringWordsType& cooc,
			 Storage1D<Math2D::Matrix<uint> >& slookup) {

  const uint nSentences = source.size();
  slookup.resize_dirty(nSentences);

  for (uint s=0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint nCurSourceWords = cur_source.size();
    const uint nCurTargetWords = cur_target.size();

    slookup[s].resize_dirty(nCurSourceWords,nCurTargetWords);
    for (uint i=0; i < nCurTargetWords; i++) {
      uint tidx = cur_target[i];
      for (uint j=0; j < nCurSourceWords; j++) {
	
	uint sidx = cur_source[j];
	uint* ptr = std::lower_bound(cooc[tidx].direct_access(), cooc[tidx].direct_access()+cooc[tidx].size(),sidx);
	assert((*ptr) == sidx);
	
	uint idx = ptr - cooc[tidx].direct_access();
	assert(idx < cooc[tidx].size());
	slookup[s](j,i) = idx;
      }
    }
  }
}

