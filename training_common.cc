/*** written by Thomas Schoenemann as a private person, since October 2009 ***/

#include "training_common.hh"
#include "projection.hh"
#include "storage_util.hh"
#include "storage_stl_interface.hh"

#include <vector>
#include <set>
#include <map>
#include <algorithm>

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

uint set_prior_dict_weights(const std::set<std::pair<uint,uint> >& known_pairs, const CooccuringWordsType& wcooc,
                            floatSingleWordDictionary prior_weight, float init_dict_regularity)
{

  uint nIgnored = 0;

  uint nTargetWords = prior_weight.size();

  for (uint i = 0; i < nTargetWords; i++)
    prior_weight[i].set_constant(init_dict_regularity);

  std::cerr << "processing read list" << std::endl;

  for (std::set<std::pair<uint,uint> >::const_iterator it = known_pairs.begin(); it != known_pairs.end(); it++) {

    uint tword = it->first;
    uint sword = it->second;

    if (tword >= wcooc.size()) {
      std::cerr << "tword out of range: " << tword << std::endl;
    }

    if (tword == 0) {
      prior_weight[0][sword-1] = 0.0;
    }
    else {
      uint pos = std::lower_bound(wcooc[tword].direct_access(), wcooc[tword].direct_access() + wcooc[tword].size(), sword) - wcooc[tword].direct_access();

      if (pos < wcooc[tword].size() && wcooc[tword][pos] == sword) {
        prior_weight[tword][pos] = 0.0;
      }
      else {
        nIgnored++;
        //std::cerr << "WARNING: ignoring entry of prior dictionary" << std::endl;
      }
    }
  }

  return nIgnored;
}

void find_cooccuring_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                           uint nSourceWords, uint nTargetWords, CooccuringWordsType& cooc)
{
  Storage1D<Math1D::Vector<uint> > additional_source;
  Storage1D<Math1D::Vector<uint> > additional_target;

  find_cooccuring_words(source, target, additional_source, additional_target, nSourceWords, nTargetWords, cooc);
}

void find_cooccuring_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                           const Storage1D<Math1D::Vector<uint> >& additional_source, const Storage1D<Math1D::Vector<uint> >& additional_target,
                           uint nSourceWords, uint nTargetWords, CooccuringWordsType& cooc)
{
  NamedStorage1D<std::set<uint> >coocset(nTargetWords, MAKENAME(coocset));
  // for (uint j=0; j < nSourceWords-1; j++)
  //   coocset[0].insert(j);

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  cooc.resize_dirty(nTargetWords);

  for (uint s = 0; s < nSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    // Math1D::Vector<uint> s_prev_occ(curJ,MAX_UINT);
    // std::map<uint,uint> s_first_occ;
    // for (uint j=0; j < curJ; j++) {

    //   uint sidx = cur_source[j];
    //   assert(sidx > 0);

    //   std::map<uint,uint>::iterator it = s_first_occ.find(sidx);

    //   if (it != s_first_occ.end())
    //     s_prev_occ[j] = it->second;
    //   else
    //     s_first_occ[sidx] = j;
    // }

    std::set<uint> source_set;
    for (uint j = 0; j < curJ; j++) {
      source_set.insert(cur_source[j]);
    }

    //iterating over a vector is faster than iterating over a set -> copy
    std::vector<uint> source_words;
    for (std::set<uint>::iterator it = source_set.begin(); it != source_set.end(); it++)
      source_words.push_back(*it);

    for (uint i = 0; i < curI; i++) {
      uint t_idx = cur_target[i];
      assert(t_idx < nTargetWords);

      std::set<uint>& cur_set = coocset[t_idx];

      const uint nWords = source_words.size();
      for (uint k = 0; k < nWords; k++)
        cur_set.insert(source_words[k]);

      // for (uint j=0; j < curJ; j++) {

      //   if (s_prev_occ[j] == MAX_UINT) {
      //     uint s_idx = cur_source[j];
      //     assert(s_idx < nSourceWords);

      //     cur_set.insert(s_idx);
      //   }
      // }
    }
  }

  const uint nAddSentences = additional_source.size();
  assert(nAddSentences == additional_target.size());

  for (uint s = 0; s < nAddSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint nCurSourceWords = additional_source[s].size();
    const uint nCurTargetWords = additional_target[s].size();

    const Storage1D<uint>& cur_source = additional_source[s];
    const Storage1D<uint>& cur_target = additional_target[s];

    for (uint i = 0; i < nCurTargetWords; i++) {
      uint t_idx = cur_target[i];
      assert(t_idx < nTargetWords);

      std::set < uint >& cur_set = coocset[t_idx];

      for (uint j = 0; j < nCurSourceWords; j++) {
        uint s_idx = cur_source[j];
        assert(s_idx < nSourceWords);

        cur_set.insert(s_idx);
      }
    }
  }

  for (uint i = 0; i < nTargetWords; i++) {

    const uint size = coocset[i].size();
    cooc[i].resize_dirty(size);
    assign(cooc[i], coocset[i]);
    coocset[i].clear();
  }

  uint max_cooc = 0;
  double sum_cooc = 0.0;
  for (uint i = 1; i < nTargetWords; i++) {
    sum_cooc += cooc[i].size();
    max_cooc = std::max < uint > (max_cooc, cooc[i].size());
  }

  std::cerr << "average number of cooccuring words: " << (sum_cooc / (nTargetWords - 1)) << ", maximum: " << max_cooc << std::endl;
}

bool read_cooccuring_words_structure(std::string filename, uint nSourceWords, uint nTargetWords, CooccuringWordsType& cooc)
{
  std::istream* in;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(filename))
    in = new igzstream(filename.c_str());
  else
    in = new std::ifstream(filename.c_str());
#else
  in = new std::ifstream(filename.c_str());
#endif

  uint size = nSourceWords * 10;
  char* cline = new char[size];

  std::vector<std::vector<uint> > temp_cooc;
  temp_cooc.push_back(std::vector<uint>());

  while (in->getline(cline, size)) {

    std::vector<std::string> tokens;

    temp_cooc.push_back(std::vector<uint>());

    std::string line = cline;
    tokenize(line, tokens, ' ');

    for (uint k = 0; k < tokens.size(); k++) {
      uint idx = convert<uint> (tokens[k]);
      if (idx >= nSourceWords) {
        std::cerr << "ERROR: index exceeds number of source words" << std::endl;
        delete in;
        return false;
      }
      temp_cooc.back().push_back(idx);
    }
  }

  delete[]cline;
  delete in;

  if (temp_cooc.size() != nTargetWords) {
    std::cerr << "ERROR: dict structure has wrong number of lines: " << temp_cooc.size() << " instead of " << nTargetWords << std::endl;
    return false;
  }

  cooc.resize(nTargetWords);

  //now copy temp_cooc -> vector
  for (uint i = 0; i < nTargetWords; i++) {

    const uint size = temp_cooc[i].size();
    cooc[i].resize_dirty(size);
    uint j = 0;
    for (std::vector<uint>::iterator it = temp_cooc[i].begin(); it != temp_cooc[i].end(); it++, j++)
      cooc[i][j] = *it;

    temp_cooc[i].clear();
  }

  double sum_cooc = 0.0;
  uint max_cooc = 0;
  for (uint i = 0; i < nTargetWords; i++) {
    sum_cooc += cooc[i].size();
    max_cooc = std::max<uint>(max_cooc, cooc[i].size());
  }

  std::cerr << "average number of cooccuring words: " << (sum_cooc / nTargetWords) << ", maximum: " << max_cooc << std::endl;

  return true;
}

void find_cooc_monolingual_pairs(const Storage1D<Math1D::Vector<uint> >& sentence, uint voc_size,
                                 Storage1D<Storage1D<uint> >& cooc)
{
  const uint nSentences = sentence.size();
  cooc.resize(voc_size);
  for (uint k = 0; k < voc_size; k++)
    cooc[k].resize(0);

  cooc[0].resize(voc_size);
  for (uint k = 0; k < voc_size; k++)
    cooc[0][k] = k;

  for (uint s = 0; s < nSentences; s++) {

    //std::cerr << "s: " << s << std::endl;

    const Storage1D<uint>& cur_sentence = sentence[s];

    const uint curI = cur_sentence.size();

    for (uint i1 = 0; i1 < curI - 1; i1++) {

      uint w1 = cur_sentence[i1];

      for (uint i2 = i1 + 1; i2 < curI; i2++) {

        uint w2 = cur_sentence[i2];

        uint l = 0;
        for (; l < cooc[w1].size() && cooc[w1][l] != w2; l++) {
          ;
        }

        if (l >= cooc[w1].size()) {
          cooc[w1].resize(cooc[w1].size() + 1);
          cooc[w1][l] = w2;
        }
      }
    }
  }

  //finally sort
  for (uint k = 1; k < voc_size; k++)
    std::sort(cooc[k].direct_access(), cooc[k].direct_access() + cooc[k].size());
}

void monolingual_pairs_cooc_count(const Storage1D<Math1D::Vector<uint> >& sentence, const Storage1D<Storage1D<uint> >& t_cooc,
                                  Storage1D<Storage1D<uint> >& t_cooc_count)
{
  t_cooc_count.resize(t_cooc.size());
  for (uint k = 0; k < t_cooc.size(); k++)
    t_cooc_count[k].resize(t_cooc[k].size(), 0);

  uint nSentences = sentence.size();

  for (uint s = 0; s < nSentences; s++) {

    //std::cerr << "s: " << s << std::endl;

    const Storage1D<uint>& cur_sentence = sentence[s];

    const uint curI = cur_sentence.size();

    for (uint i1 = 0; i1 < curI - 1; i1++) {

      uint w1 = cur_sentence[i1];

      for (uint i2 = i1 + 1; i2 < curI; i2++) {

        uint w2 = cur_sentence[i2];

        const uint* ptr = std::lower_bound(t_cooc[w1].direct_access(), t_cooc[w1].direct_access() + t_cooc[w1].size(), w2);

        uint k = ptr - t_cooc[w1].direct_access();
        t_cooc_count[w1][k] += 1;
      }
    }
  }
}

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    const Storage1D<Storage1D<uint> >& target_cooc, Storage1D<Storage1D<Storage1D<uint> > >& st_cooc)
{
  st_cooc.resize(target_cooc.size());
  for (uint i = 0; i < target_cooc.size(); i++)
    st_cooc[i].resize(target_cooc[i].size());

  const uint nSentences = source.size();

  for (uint s = 0; s < nSentences; s++) {

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 < curI; i1++) {

      uint start_i2 = (i1 == 0) ? 0 : i1 + 1;

      uint t1 = (i1 == 0) ? 0 : cur_target[i1 - 1];

      const Storage1D<uint>& cur_tcooc = target_cooc[t1];

      for (uint i2 = start_i2; i2 <= curI; i2++) {

        uint t2 = (i2 == 0) ? 0 : cur_target[i2 - 1];

        const uint* ptr = std::lower_bound(cur_tcooc.direct_access(), cur_tcooc.direct_access() + cur_tcooc.size(), t2);

        assert(*ptr == t2);

        assert(ptr >= cur_tcooc.direct_access());
        assert(ptr < cur_tcooc.direct_access() + cur_tcooc.size());

        uint diff = ptr - cur_tcooc.direct_access();

        Storage1D<uint>& cur_scooc = st_cooc[t1][diff];

        for (uint j = 0; j < curJ; j++) {

          uint s_idx = cur_source[j];

          uint l = 0;
          for (; l < cur_scooc.size() && cur_scooc[l] != s_idx; l++) {
            ;
          }

          if (l >= cur_scooc.size()) {

            cur_scooc.resize(cur_scooc.size() + 1);
            cur_scooc[l] = s_idx;
          }
        }
      }
    }
  }

  //finally sort
  for (uint i = 0; i < st_cooc.size(); i++)
    for (uint k = 0; k < st_cooc[i].size(); k++)
      std::sort(st_cooc[i][k].direct_access(), st_cooc[i][k].direct_access() + st_cooc[i][k].size());
}

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    std::map<std::pair<uint,uint>,std::set<uint> >& cooc)
{
  const uint nSentences = source.size();
  assert(nSentences == target.size());

  for (uint s = 0; s < nSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1 - 1];

      uint start_i2 = (i1 == 0) ? 0 : i1 + 1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

        const uint t2 = (i2 == 0) ? 0 : cur_target[i2 - 1];

        std::pair<uint,uint> tpair = std::make_pair(t1, t2);

        for (uint j = 0; j < curJ; j++)
          cooc[tpair].insert(cur_source[j]);
      }
    }
  }

}

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    uint /*nSourceWords */, uint nTargetWords, Storage1D<Storage1D<std::pair<uint,Storage1D<uint> > > >& cooc)
{
  cooc.resize(nTargetWords);

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  for (uint s = 0; s < nSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1 - 1];

      uint start_i2 = (i1 == 0) ? 0 : i1 + 1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

        const uint t2 = (i2 == 0) ? 0 : cur_target[i2 - 1];

        uint t2_cooc_idx = MAX_UINT;
        for (uint k = 0; k < cooc[t1].size(); k++) {
          if (cooc[t1][k].first == t2) {
            t2_cooc_idx = k;
            break;
          }
        }

        if (t2_cooc_idx == MAX_UINT) {
          t2_cooc_idx = cooc[t1].size();
          cooc[t1].resize(cooc[t1].size() + 1);
          cooc[t1][t2_cooc_idx].first = t2;
        }

        for (uint j = 0; j < curJ; j++) {

          uint j_cooc_idx = MAX_UINT;
          for (uint k = 0; k < cooc[t1][t2_cooc_idx].second.size(); k++) {

            if (cooc[t1][t2_cooc_idx].second[k] == cur_source[j]) {
              j_cooc_idx = k;
              break;
            }
          }

          if (j_cooc_idx == MAX_UINT) {
            j_cooc_idx = cooc[t1][t2_cooc_idx].second.size();
            cooc[t1][t2_cooc_idx].second.resize(cooc[t1][t2_cooc_idx].second.size() + 1);
            cooc[t1][t2_cooc_idx].second[j_cooc_idx] = cur_source[j];
          }
        }
      }
    }
  }
}

void count_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    std::map<std::pair<uint,uint>,std::map<uint,uint> >& cooc_count)
{
  const uint nSentences = source.size();
  assert(nSentences == target.size());

  for (uint s = 0; s < nSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1 - 1];

      uint start_i2 = (i1 == 0) ? 0 : i1 + 1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

        const uint t2 = (i2 == 0) ? 0 : cur_target[i2 - 1];

        std::pair<uint,uint> tpair = std::make_pair(t1, t2);

        for (uint j = 0; j < curJ; j++)
          cooc_count[tpair][cur_source[j]]++;
      }
    }
  }
}

bool operator<(const std::pair<uint,Storage1D<std::pair<uint,uint> > >& x1,
               const std::pair<uint,Storage1D<std::pair<uint,uint> > >& x2)
{
  //   if (x1.first == x2.first)
  //     std::cerr << "identical: " << x1.first << std::endl;

  assert(x1.first != x2.first);

  return (x1.first < x2.first);
}

bool operator<(const std::pair<uint,uint>& x1, const std::pair<uint,uint>& x2)
{
  if (x1.first == x2.first)
    return (x1.second < x2.second);
  else
    return (x1.first < x2.first);
}

void count_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    uint /*nSourceWords */, uint nTargetWords,
    Storage1D<Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > > >& cooc)
{
  cooc.resize(nTargetWords);
  cooc[0].resize(nTargetWords);
  for (uint k = 0; k < nTargetWords; k++) {
    cooc[0][k].first = k;
  }

  const uint nSentences = source.size();
  assert(nSentences == target.size());

  /**** stage 1: find coocuring target words ****/

  for (uint s = 0; s < nSentences; s++) {

    if ((s % 100) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curI = target[s].size();

    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 1; i1 <= curI; i1++) {

      const uint t1 = cur_target[i1 - 1];

      Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > >& cur_cooc = cooc[t1];

      assert(t1 != 0);

      for (uint i2 = i1 + 1; i2 <= curI; i2++) {

        const uint t2 = cur_target[i2 - 1];

        uint t2_cooc_idx = MAX_UINT;
        for (uint k = 0; k < cur_cooc.size(); k++) {
          if (cur_cooc[k].first == t2) {
            t2_cooc_idx = k;
            break;
          }
        }

        if (t2_cooc_idx == MAX_UINT) {
          t2_cooc_idx = cur_cooc.size();
          cur_cooc.resize(cooc[t1].size() + 1);
          cur_cooc[t2_cooc_idx].first = t2;
        }
      }
    }
  }

  /**** stage 2: sort the vectors ****/
  for (uint i = 0; i < cooc.size(); i++) {
    //std::cerr << "sorting #" << i << std::endl;
    std::sort(cooc[i].direct_access(), cooc[i].direct_access() + cooc[i].size());
  }

  /**** stage 3: find coocuring source words and target pairs ****/
  for (uint s = 0; s < nSentences; s++) {

    if ((s % 100) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    for (uint i1 = 0; i1 <= curI; i1++) {

      const uint t1 = (i1 == 0) ? 0 : cur_target[i1 - 1];

      Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > >& cur_cooc = cooc[t1];

      uint start_i2 = (i1 == 0) ? 0 : i1 + 1;
      for (uint i2 = start_i2; i2 <= curI; i2++) {

        const uint t2 = (i2 == 0) ? 0 : cur_target[i2 - 1];

        uint t2_cooc_idx = MAX_UINT;
        if (i1 == 0)
          t2_cooc_idx = t2;
        else {
          for (uint k = 0; k < cur_cooc.size(); k++) {
            if (cur_cooc[k].first == t2) {
              t2_cooc_idx = k;
              break;
            }
          }
        }

        if (t2_cooc_idx == MAX_UINT) {
          assert(false);
          t2_cooc_idx = cur_cooc.size();
          cur_cooc.resize(cur_cooc.size() + 1);
          cur_cooc[t2_cooc_idx].first = t2;
        }
        //std::pair<uint,Storage1D<std::pair<uint,uint> > >& cur_pair = cur_cooc[t2_cooc_idx];

        Storage1D<std::pair<uint,uint> >& cur_vec = cur_cooc[t2_cooc_idx].second;

        for (uint j = 0; j < curJ; j++) {

          const uint s_idx = cur_source[j];

          uint j_cooc_idx = MAX_UINT;
          for (uint k = 0; k < cur_vec.size(); k++) {

            if (cur_vec[k].first == s_idx) {
              j_cooc_idx = k;
              break;
            }
          }

          if (j_cooc_idx == MAX_UINT) {
            j_cooc_idx = cur_vec.size();
            cur_vec.resize(cur_vec.size() + 1);
            cur_vec[j_cooc_idx] = std::make_pair(s_idx, 0);
          }
          cur_vec[j_cooc_idx].second++;
        }
      }
    }
  }

  /*** stage 4: sort the indivialual source vectors ****/
  for (uint i = 0; i < cooc.size(); i++) {
    for (uint k = 0; k < cooc[i].size(); k++)
      std::sort(cooc[i][k].second.direct_access(), cooc[i][k].second.direct_access() + cooc[i][k].second.size());
  }
}

void find_cooccuring_lengths(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                             CooccuringLengthsType& cooc)
{
  std::map<uint,std::set<uint> > coocvec;

  const uint nSentences = target.size();
  uint max_tlength = 0;

  for (uint s = 0; s < nSentences; s++) {
    uint cur_tlength = target[s].size();
    if (cur_tlength > max_tlength)
      max_tlength = cur_tlength;

    coocvec[cur_tlength].insert(source[s].size());
  }

  cooc.resize_dirty(max_tlength + 1);
  for (std::map<uint,std::set<uint> >::iterator it = coocvec.begin(); it != coocvec.end(); it++) {

    const uint tlength = it->first;
    cooc[tlength].resize_dirty(it->second.size());
    uint k = 0;
    for (std::set<uint>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++, k++) {
      cooc[tlength][k] = *it2;
    }
  }
}

void generate_wordlookup(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                         const CooccuringWordsType& cooc, uint nSourceWords, LookupTable& slookup, uint max_size)
{
  const uint nSentences = source.size();
  slookup.resize_dirty(nSentences);

  for (uint s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint J = cur_source.size();
    const uint I = cur_target.size();

    SingleLookupTable& cur_lookup = slookup[s];

    if (J * I <= max_size) {

      cur_lookup.resize_dirty(J, I);
    }
    //NOTE: even if we don't store the lookup table, we still check for consistency at this point

    Math1D::Vector<uint> s_prev_occ(J, MAX_UINT);
    std::map<uint,uint> s_first_occ;
    for (uint j = 0; j < J; j++) {

      const uint sidx = cur_source[j];
      assert(sidx > 0);

      std::map<uint,uint>::iterator it = s_first_occ.find(sidx);

      if (it != s_first_occ.end())
        s_prev_occ[j] = it->second;
      else
        s_first_occ[sidx] = j;
    }

    std::map<uint,uint> first_occ;

    for (uint i = 0; i < I; i++) {
      uint tidx = cur_target[i];

      std::map<uint,uint>::iterator it = first_occ.find(tidx);
      if (it != first_occ.end()) {
        uint prev_i = it->second;

        if (cur_lookup.size() > 0) {
          for (uint j = 0; j < J; j++)
            cur_lookup(j, i) = cur_lookup(j, prev_i);
        }
      }
      else {

        first_occ[tidx] = i;

        const Math1D::Vector<uint>& cur_cooc = cooc[tidx];
        const size_t cur_size = cur_cooc.size();

        if (cur_size == nSourceWords - 1) {

          //std::cerr << "happens" << std::endl;

          if (cur_lookup.size() > 0) {

            for (uint j = 0; j < J; j++) {

              const uint sidx = cur_source[j] - 1;
              cur_lookup(j, i) = sidx;
            }
          }
        }
        else {

          const uint* start = cur_cooc.direct_access();
          const uint* end = cur_cooc.direct_access() + cur_size;

          for (uint j = 0; j < J; j++) {

            const uint sidx = cur_source[j];

            const uint prev_occ = s_prev_occ[j];

            if (prev_occ != MAX_UINT) {
              if (cur_lookup.size() > 0)
                cur_lookup(j, i) = cur_lookup(prev_occ, i);
            }
            else {

              const uint* ptr = std::lower_bound(start, end, sidx);

              if (ptr == end || (*ptr) != sidx) {
                INTERNAL_ERROR << " word not found. Exiting." << std::endl;
                exit(1);
              }

              if (cur_lookup.size() > 0) {
                const uint idx = ptr - start;
                assert(idx < cooc[tidx].size());
                cur_lookup(j, i) = idx;
              }
            }
          }
        }
      }
    }
  }
}

const SingleLookupTable& get_wordlookup(const Storage1D<uint>& source, const Storage1D<uint>& target, const CooccuringWordsType& cooc,
                                        uint nSourceWords, const SingleLookupTable& lookup, SingleLookupTable& aux)
{
  const uint J = source.size();
  const uint I = target.size();

  if (lookup.xDim() == J && lookup.yDim() == I)
    return lookup;

  aux.resize_dirty(J, I);

  std::map<uint,uint> first_occ;

  Math1D::Vector<uint> s_prev_occ(J, MAX_UINT);
  std::map<uint,uint> s_first_occ;
  for (uint j = 0; j < J; j++) {

    uint sidx = source[j];
    assert(sidx > 0);

    std::map<uint,uint>::iterator it = s_first_occ.find(sidx);

    if (it != s_first_occ.end())
      s_prev_occ[j] = it->second;
    else
      s_first_occ[sidx] = j;
  }

  for (uint i = 0; i < I; i++) {
    uint tidx = target[i];

    std::map<uint,uint>::iterator it = first_occ.find(tidx);
    if (it != first_occ.end()) {
      uint prev_i = it->second;

      for (uint j = 0; j < J; j++)
        aux(j, i) = aux(j, prev_i);
    }
    else {

      first_occ[tidx] = i;

      const Math1D::Vector<uint>& cur_cooc = cooc[tidx];
      const size_t cur_size = cur_cooc.size();

      double ratio = double (cur_size) / double (nSourceWords);

      const uint* start = cur_cooc.direct_access();
      const uint* end = cur_cooc.direct_access() + cur_size;

      if (cur_size == nSourceWords - 1) {

        for (uint j = 0; j < J; j++)
          aux(j, i) = source[j] - 1;
      }
      else {

        //const uint* last = start + cur_size/2;

        for (uint j = 0; j < J; j++) {

          const uint prev_occ = s_prev_occ[j];

          if (prev_occ != MAX_UINT) {
            aux(j, i) = aux(prev_occ, i);
          }
          else {

            const uint sidx = source[j];

            const uint* ptr;

#if 1
            //const uint guess_idx = sidx-1;
            const uint guess_idx = floor((sidx - 1) * ratio);

            if (guess_idx < cur_size) {
              //if (false) {

              const uint* guess = start + guess_idx;

              const uint p = *(guess);
              if (p == sidx) {
                aux(j, i) = guess_idx;
                continue;
              }
              else if (p < sidx)
                ptr = std::lower_bound(guess, end, sidx);
              else
                ptr = std::lower_bound(start, guess, sidx);
            }
            else
              ptr = std::lower_bound(start, end, sidx);
#else
            //experimental result: slower
            uint p = *last;
            if (p == sidx)
              ptr = last;
            else if (p < sidx)
              ptr = std::lower_bound(last, end, sidx);
            else
              ptr = std::lower_bound(start, last, sidx);

            last = ptr;
#endif

#ifdef SAFE_MODE
            if (ptr == end || (*ptr) != sidx) {
              INTERNAL_ERROR << " word not found. Exiting." << std::endl;
              exit(1);
            }
#endif

            const uint idx = ptr - start;
            assert(idx < cur_size);
            aux(j, i) = idx;
          }
        }
      }
    }
  }

  return aux;
}

double prob_penalty(double x, double beta)
{
  return 1.0 - std::exp(-x / beta);
}

double prob_pen_prime(double x, double beta)
{
  return -prob_penalty(x, beta) / beta;
}

void update_dict_from_counts(const UnnamedSingleWordDictionary& fdict_count, const floatUnnamedSingleWordDictionary& prior_weight,
                             double dict_weight_sum, bool smoothed_l0, double l0_beta, uint nDictStepIter, UnnamedSingleWordDictionary& dict,
                             double min_prob, bool unconstrained_m_step)
{
  if (dict_weight_sum > 0.0) {

    double alpha = 1.0;

    for (uint i = 0; i < fdict_count.size(); i++) {

      //std::cerr << "i: " << i << ", " << dict[i].size() << " entries" << std::endl;

      const Math1D::Vector<double>& cur_count = fdict_count[i];
      const Math1D::Vector<float>& cur_prior = prior_weight[i];
      Math1D::Vector<double>& cur_dict = dict[i];

      double cur_energy = single_dict_m_step_energy(cur_count, cur_prior, cur_dict, smoothed_l0, l0_beta);

      Math1D::Vector<double> hyp_dict = cur_count;

      const double sum = cur_count.sum();
      const double prev_sum = cur_dict.sum();

      if (sum > 1e-305) {
        const double inv_sum = 1.0 / sum;
        assert(!isnan(inv_sum));

        for (uint k = 0; k < cur_count.size(); k++) {
          hyp_dict[k] *= prev_sum * inv_sum;
        }

        double hyp_energy = single_dict_m_step_energy(cur_count, cur_prior, hyp_dict, smoothed_l0, l0_beta);

        //NOTE: if the entries in prior_weight[i] are all the same, hyp_energy should always be small or equal to cur_energy
        if (hyp_energy < cur_energy)
          cur_dict = hyp_dict;
      }
      if (!unconstrained_m_step)
        single_dict_m_step(cur_count, cur_prior, cur_dict, alpha, nDictStepIter, smoothed_l0, l0_beta, false);
      else
        single_dict_m_step_unconstrained(cur_count, cur_prior, cur_dict, nDictStepIter, smoothed_l0, l0_beta, 5);

      for (uint k = 0; k < cur_count.size(); k++)
        cur_dict[k] = std::max(min_prob, cur_dict[k]);

      //DEBUG
      // if (i >= 14)
      //        exit(1);
      //END_DEBUG
    }
  }
  else {

    for (uint i = 0; i < fdict_count.size(); i++) {

      const Math1D::Vector<double>& cur_count = fdict_count[i];
      Math1D::Vector<double>& cur_dict = dict[i];

      const double sum = cur_count.sum();
      if (sum > 1e-307) {
        const double inv_sum = 1.0 / sum;
        assert(!isnan(inv_sum));

        for (uint k = 0; k < cur_count.size(); k++)
          cur_dict[k] = std::max(min_prob, cur_count[k] * inv_sum);

        //DEBUG
        // double dict_sum = dict[i].sum();
        // if (dict_sum > 1.001)
        //   std::cerr << "WARNING: dict sum=" << dict_sum << std::endl;
        //END_DEBUG
      }
      else {
        // std::cerr << "WARNING : did not update dictionary entries for target word #" << i
        //           << " because sum is " << sum << "( dict-size = " << dict[i].size() << " )" << std::endl;
      }
    }
  }
}

double single_dict_m_step_energy(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight,
                                 const Math1D::Vector<double>& dict, bool smoothed_l0, double l0_beta)
{
  double energy = 0.0;

  const uint dict_size = dict.size();

  if (!smoothed_l0) {
    for (uint k = 0; k < dict_size; k++)
      energy += prior_weight[k] * dict[k];
  }
  else {
    for (uint k = 0; k < dict_size; k++)
      energy += prior_weight[k] * prob_penalty(dict[k], l0_beta);
  }

  for (uint k = 0; k < dict_size; k++) {

    const double cur_dict_entry = std::max(dict[k], 1e-300);

    energy -= fdict_count[k] * std::log(cur_dict_entry);
  }

  return energy;
}

void single_dict_m_step(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight,
                        Math1D::Vector<double>& dict, double alpha, uint nIter, bool smoothed_l0, double l0_beta, bool with_slack)
{
  //std::cerr << "hi, slack: " << with_slack << std::endl;

  const uint dict_size = prior_weight.size();

  //std::cerr << "dict size: " << dict_size << std::endl;
  //std::cerr << "smoothed l0: " << smoothed_l0 << std::endl;

  if (prior_weight.max_abs() == 0.0) {

    const double sum = fdict_count.sum();

    if (sum > 1e-305) {
      for (uint k = 0; k < dict_size; k++) {
        dict[k] = fdict_count[k] / sum;
      }
    }

    return;
  }

  double energy = single_dict_m_step_energy(fdict_count, prior_weight, dict, smoothed_l0, l0_beta);

  Math1D::Vector<double> dict_grad = dict;
  Math1D::Vector<double> hyp_dict = dict;
  Math1D::Vector<double> new_dict = dict;

  double slack_entry = 1.0 - dict.sum();
  double new_slack_entry = slack_entry;

  if (!with_slack)
    assert(dict.sum() >= 0.99);

  double line_reduction_factor = 0.5;

  //NOTE: the norm constraint can only be turned off when there is no slack
  // in orthant mode we would get a problem because the slack variable has no gradient
  bool norm_constraint = true; //DON'T PUBLISH

  for (uint iter = 1; iter <= nIter; iter++) {

    //std::cerr << " ###iteration " << iter << ", energy: " << energy << std::endl;

    //set gradient to 0 and recalculate
    if (with_slack || norm_constraint) {
      for (uint k = 0; k < dict_size; k++) {
        double cur_dict_entry = std::max(1e-15, dict[k]);

        if (!smoothed_l0)
          dict_grad[k] = prior_weight[k] - fdict_count[k] / cur_dict_entry;
        else
          dict_grad[k] = prior_weight[k] * prob_pen_prime(cur_dict_entry, l0_beta) - fdict_count[k] / cur_dict_entry;
      }
    }
    else {   //DON'T PUBLISH -- START

      //the energy is NOT scale invariant as it does not account for renormalization
      // => need to modify the gradient

      dict_grad.set_constant(0.0);

      double addon = 0.0;

      double sum_subst = 0.0;

      for (uint k = 0; k < dict_size; k++) {

        const double cur_dict_entry = std::max(1e-15, dict[k]);

        //a) regularity term
        if (!smoothed_l0) {
          dict_grad[k] += prior_weight[k] * (1.0 - cur_dict_entry);

          const double subst = prior_weight[k] * cur_dict_entry;
          sum_subst += subst;

          // for (uint kk=0; kk < dict_size; kk++) { //NOTE: can be simplified if prior_weight has equal entries everywhere
          //   if (kk != k)
          //     dict_grad[kk] -= subst;
          // }
        }
        else {
          double weight = prior_weight[k] * prob_pen_prime(cur_dict_entry, l0_beta);

          dict_grad[k] += weight * (1.0 - cur_dict_entry);

          const double subst = weight * cur_dict_entry;
          sum_subst += subst;

          // for (uint kk=0; kk < dict_size; kk++) {
          //   if (kk != k)
          //     dict_grad[kk] -= subst;
          // }
        }

        //b) entropy term
        dict_grad[k] -= fdict_count[k] / cur_dict_entry;        //numerator
        addon += fdict_count[k];        // denominator
      }

      for (uint k = 0; k < dict_size; k++) {
        dict_grad[k] += addon;

        double cur_dict_entry = std::max(1e-15, dict[k]);
        double own_weight = prior_weight[k] * prob_pen_prime(cur_dict_entry, l0_beta) * cur_dict_entry;
        double own_subst = sum_subst - own_weight;

        dict_grad[k] -= own_subst;
      }
    }   //DON'T PUBLISH -- END

    //go in neg. gradient direction
    //for (uint k = 0; k < dict_size; k++) {
    //  new_dict[k] = dict[k] - alpha * dict_grad[k];
    //}

    //new_dict = dict;
    //new_dict.add_vector_multiple(dict_grad, -alpha);

    Makros::go_in_neg_direction(new_dict.direct_access(), new_dict.size(), dict.direct_access(), dict_grad.direct_access(), alpha);

    new_slack_entry = slack_entry;

    //reproject
    if (with_slack)
      projection_on_simplex_with_slack(new_dict.direct_access(), new_slack_entry, new_dict.size(), 1e-15);
    else {
      if (norm_constraint)
        projection_on_simplex(new_dict.direct_access(), new_dict.size(), 1e-15);
      else {

        //std::cerr << "orthant projection" << std::endl;

        //orthant projection followed by renormalization
        double sum = 0.0;

        for (uint k = 0; k < dict_size; k++) {
          new_dict[k] = std::max(1e-15, new_dict[k]);
          sum += new_dict[k];
        }

        //std::cerr << "sum: " << sum << std::endl;

        if (sum < 0.01) {
          std::cerr << "WARNING: sum = " << sum << " in iteration " << iter << std::endl;
        }

        double inv_sum = 1.0 / sum;
        for (uint k = 0; k < dict_size; k++) {
          new_dict[k] = std::max(1e-15, new_dict[k] * inv_sum);
        }
      }
    }

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    bool decreasing = true;

    uint nTries = 0;

    while (decreasing || best_energy > energy) {

      nTries++;

      lambda *= line_reduction_factor;
      double neg_lambda = 1.0 - lambda;

      //for (uint k = 0; k < dict_size; k++) {
      //  hyp_dict[k] = lambda * new_dict[k] + neg_lambda * dict[k];
      //}
      Makros::assign_weighted_combination(hyp_dict.direct_access(), dict_size, lambda, new_dict.direct_access(), neg_lambda, dict.direct_access());

      double hyp_energy = single_dict_m_step_energy(fdict_count, prior_weight, hyp_dict, smoothed_l0, l0_beta);
      //std::cerr << "lambda = " << lambda << ", hyp_energy = " << hyp_energy << std::endl;

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        decreasing = true;

        best_lambda = lambda;
      }
      else
        decreasing = false;

      if (best_energy <= 0.95 * energy)
        break;
      if (nTries >= 4 && best_energy <= 0.99 * energy)
        break;

      if (nTries >= 18)
        break;
    }

    if (best_energy > energy) {
      break;
    }

    if (nTries > 12)
      line_reduction_factor *= 0.5;
    else if (nTries > 4)
      line_reduction_factor *= 0.8;

    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint k = 0; k < dict_size; k++) {
    //  dict[k] = best_lambda * new_dict[k] + neg_best_lambda * dict[k];
    //}
    Makros::assign_weighted_combination(dict.direct_access(), dict_size, best_lambda, new_dict.direct_access(), neg_best_lambda, dict.direct_access());

    slack_entry = best_lambda * new_slack_entry + neg_best_lambda * slack_entry;

    if (best_energy > energy - 1e-5 || best_lambda < 1e-8) {
      //std::cerr << "CUTOFF after " << iter  << " iterations" << std::endl;
      break;
    }

    energy = best_energy;
  }
}

//use L-BFGS
void single_dict_m_step_unconstrained(const Math1D::Vector<double>& fdict_count, const Math1D::Vector<float>& prior_weight,
                                      Math1D::Vector<double>& dict, uint nIter, bool smoothed_l0, double l0_beta, uint L)
{
  //NOTE: the energy is NOT scale invariant as it does not account for renormalization
  // => need to modify the gradient

  const uint dict_size = prior_weight.size();

  if (prior_weight.max_abs() == 0.0) {

    const double sum = fdict_count.sum();

    if (sum > 1e-305) {
      for (uint k = 0; k < dict_size; k++) {
        dict[k] = fdict_count[k] / sum;
      }
    }

    return;
  }

  double energy = single_dict_m_step_energy(fdict_count, prior_weight, dict, smoothed_l0, l0_beta);

  Math1D::Vector<double> dict_grad = dict;
  Math1D::Vector<double> dict_param(dict_size);
  Math1D::Vector<double> param_grad = dict;
  Math1D::Vector<double> hyp_dict(dict_size);
  Math1D::Vector<double> hyp_dict_param(dict_size);
  Math1D::Vector<double> search_direction(dict_size);

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(dict_size);
    step[k].resize(dict_size);
  }

  for (uint k = 0; k < dict_size; k++)
    dict_param[k] = sqrt(dict[k]);

  double line_reduction_factor = 0.5;

  double cur_sqr_sum = 1.0;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint iter = 1; iter <= nIter; iter++) {

    //std::cerr << "L-BFGS iteration " << iter << ", energy: " << energy << std::endl;

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    //the energy is NOT scale invariant as it does not account for renormalization
    // => need to modify the gradient

    dict_grad.set_constant(0.0);
    param_grad.set_constant(0.0);

    double addon = 0.0;

    double sum_subst = 0.0;

    //NOTE: in contrast to the constrained routine (with orthant projection),
    //  here we generally cannot assume that the sum of the squared parameters is 1.
    //   => will need this sum in the gradient calculation

    //std::cerr << "dict sum: " << cur_sqr_sum << std::endl;

    for (uint k = 0; k < dict_size; k++) {

      //const double cur_dict_entry = std::max(1e-15, dict[k]);
      const double cur_dict_entry = std::max(1e-300, dict[k]);

      //the correct equations demand to divide by cur_sqr_norm everywhere. we do this outside the loop

      //a) regularity term
      double weight = prior_weight[k];
      if (smoothed_l0)
        weight *= prob_pen_prime(cur_dict_entry, l0_beta);

      //dict_grad[k] += weight * (cur_sqr_sum - (cur_dict_entry * cur_sqr_sum) ) / (cur_sqr_sum * cur_sqr_sum);
      //dict_grad[k] += weight * (1.0 - cur_dict_entry) / (cur_sqr_sum);
      dict_grad[k] += weight * (1.0 - cur_dict_entry);  //division by cur_sqr_sum is done outside the loop

      //const double subst = weight * (cur_dict_entry * cur_sqr_sum) / (cur_sqr_sum * cur_sqr_sum);
      //const double subst = weight * (cur_dict_entry) / (cur_sqr_sum);
      const double subst = weight * cur_dict_entry;     //division by cur_sqr_sum is done outside the loop
      sum_subst += subst;

      // for (uint kk=0; kk < dict_size; kk++) {
      //   if (kk != k)
      //     dict_grad[kk] -= subst;
      // }

      //b) entropy term
      //dict_grad[k] -= fdict_count[k] / (cur_dict_entry * cur_sqr_sum);  //numerator
      //addon += fdict_count[k] / cur_sqr_sum; // denominator

      //division by cur_sqr_sum is done outside the loop
      dict_grad[k] -= fdict_count[k] / cur_dict_entry;  //numerator
      addon += fdict_count[k];  // denominator
    }

    for (uint k = 0; k < dict_size; k++) {
      dict_grad[k] += addon;

      double cur_dict_entry = std::max(1e-15, dict[k]);
      double own_weight = prior_weight[k] * prob_pen_prime(cur_dict_entry, l0_beta) * cur_dict_entry;
      double own_subst = sum_subst - own_weight;

      dict_grad[k] -= own_subst;
    }

    dict_grad *= 1.0 / cur_sqr_sum;

    // b) now calculate the gradient for the actual parameters

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = cur_sqr_sum;   //dict_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    double coeff_sum = 0.0;

    for (uint k = 0; k < dict_size; k++) {
      const double wp = dict_param[k];
      const double grad = dict_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      param_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
    }
    for (uint kk = 0; kk < dict_size; kk++)
      param_grad[kk] -= coeff_sum * dict_param[kk];

    // for (uint k=0; k < dict_size; k++) {
    //   param_grad[k] = 2.0 * dict_grad[k] * dict_param[k];
    // }

    double sqr_grad_norm = param_grad.sqr_norm();

    if (sqr_grad_norm < 0.01) {
      break;
    }
    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < dict_size; k++) {

        //cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
        cur_grad_diff[k] += param_grad[k];
        cur_rho += cur_grad_diff[k] * cur_step[k];
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      if (cur_curv < 0.0) {
        //this can happen as our function is not convex and we do not enforce part 2 of the Wolfe conditions
        // (this cannot be done by backtracking line search, see Algorithm 3.5 in [Nocedal & Wright])
        // Our solution is to simply restart L-BFGS now

        start_iter = iter;
      }

      rho[cur_l] = 1.0 / cur_rho;
    }

    search_direction = param_grad;

    if (iter > start_iter) {

      Math1D::Vector<double> alpha(L);

      const int cur_first_iter = std::max < int >(start_iter, iter - L);

      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter - 1; prev_iter >= cur_first_iter; prev_iter--) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double cur_alpha = 0.0;
        for (uint k = 0; k < dict_size; k++) {
          cur_alpha += search_direction[k] * cur_step[k];
        }
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        for (uint k = 0; k < dict_size; k++) {
          search_direction[k] -= cur_alpha * cur_grad_diff[k];
        }
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = 0.0;
        for (uint k = 0; k < dict_size; k++) {
          beta += search_direction[k] * cur_grad_diff[k];
        }
        beta *= rho[prev_l];

        const double gamma = alpha[prev_l] - beta;

        for (uint k = 0; k < dict_size; k++) {
          search_direction[k] += cur_step[k] * gamma;
        }
      }

    }
    else {
      search_direction *= 1.0 / sqrt(search_direction.sqr_norm());
    }

    negate(search_direction);

    //NOTE: because our function is nonconvex and we did not enforce the strong Wolfe conditions,
    // the curvature condition is not generally satisfied and we can get a direction of ascent
    // in this case we switch to the negative gradient
    double check_prod = (search_direction % param_grad);

    //std::cerr << "check_prod: " << check_prod << std::endl;

    if (check_prod >= 0.0) {

      INTERNAL_ERROR << " not a search direction" << std::endl;
      exit(1);

      // //std::cerr << "switching to steepest descent for this iter" << std::endl;

      // //TODO: think about whether we should truncate the L-BGFS history in such as case

      // search_direction = param_grad;
      // negate(search_direction);

      // check_prod = -param_grad.sqr_norm();
    }

    double cutoff_offset = 0.05 * check_prod;

    // d) line search

    double best_energy = 1e300;

    double lambda = 1.0;
    double best_lambda = lambda;

    bool decreasing = true;

    uint nTries = 0;

    while (decreasing || best_energy > energy) {

      nTries++;

      lambda *= line_reduction_factor;

      for (uint k = 0; k < dict_size; k++) {
        hyp_dict_param[k] = dict_param[k] + lambda * search_direction[k];
      }

      double sum = hyp_dict.sqr_norm();
      for (uint k = 0; k < dict_size; k++) {
        hyp_dict[k] = std::max(1e-15, hyp_dict_param[k] * hyp_dict_param[k] / sum);
      }

      double hyp_energy = single_dict_m_step_energy(fdict_count, prior_weight, hyp_dict, smoothed_l0, l0_beta);
      //std::cerr << "lambda = " << lambda << ", hyp_energy = " << hyp_energy << std::endl;

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        decreasing = true;

        best_lambda = lambda;
      }
      else
        decreasing = false;

      //TODO: think about checking Wolfe part 2. However, backtracking line search may not be enough then
      if (best_energy <= energy + lambda * cutoff_offset)       //Wolfe part 1
        break;

      // if (best_energy <= 0.95*energy)
      //   break;
      // if (nTries >= 4 && best_energy <= 0.99*energy)
      //   break;

      //if (nTries >= 18)
      if (nTries >= 250)
        break;
    }

    if (nTries > 12)
      line_reduction_factor *= 0.5;
    else if (nTries > 4)
      line_reduction_factor *= 0.8;

    // e) go to the determined point

    if (best_energy < energy) {

      //update the dict and the L-BFGS variables

      uint cur_l = (iter % L);

      Math1D::Vector<double>& cur_step = step[cur_l];
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

      for (uint k = 0; k < dict_size; k++) {
        double step = best_lambda * search_direction[k];
        cur_step[k] = step;
        dict_param[k] += step;

        //prepare for the next iteration
        cur_grad_diff[k] = -param_grad[k];
      }

      double sum = dict_param.sqr_norm();
      for (uint k = 0; k < dict_size; k++) {
        dict[k] = std::max(1e-15, dict_param[k] * dict_param[k] / sum);
      }

      cur_sqr_sum = sum;

      energy = best_energy;
    }
    else {
      std::
      cerr << "WARNING: failed to get descent, sqr gradient norm: " << param_grad.sqr_norm() << std::endl;
      exit(1);
      break;
    }
  }
}

//NOTE: the function to be minimized can be decomposed over the target words
void dict_m_step(const SingleWordDictionary& fdict_count, const floatSingleWordDictionary& prior_weight,
                 SingleWordDictionary& dict, double alpha, uint nIter, bool smoothed_l0, double l0_beta)
{
  for (uint k = 0; k < dict.size(); k++)
    single_dict_m_step(fdict_count[k], prior_weight[k], dict[k], alpha, nIter, smoothed_l0, l0_beta);
}

void par2nonpar_start_prob(const Math1D::Vector<double>& sentence_start_parameters, Storage1D<Math1D::Vector<double> >& sentence_start_prob)
{
  for (uint J = 1; J < sentence_start_prob.size(); J++) {
    if (sentence_start_prob[J].size() > 0) {

      double sum = 0.0;

      for (uint j = 0; j < J; j++)
        sum += sentence_start_parameters[j];

      if (sum > 1e-305) {
        const double inv_sum = 1.0 / sum;
        for (uint j = 0; j < J; j++)
          sentence_start_prob[J][j] =
            std::max(1e-8, inv_sum * sentence_start_parameters[j]);
      }
      else {
        std::cerr << "WARNING: sum too small for start prob " << J << ", not updating." << std::endl;
      }
    }
  }

}

double start_prob_m_step_energy(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                                const Math1D::Vector<double>& param)
{
  double energy = 0.0;

  const uint nParams = param.size();

  for (uint j = 0; j < nParams; j++)
    energy -= singleton_count[j] * std::log(std::max(1e-15, param[j]));

  double param_sum = 0.0;
  for (uint j = 0; j < nParams; j++) {

    param_sum += std::max(1e-15, param[j]);
    energy += norm_count[j] * std::log(param_sum);
  }

  return energy;
}

void start_prob_m_step(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                       Math1D::Vector<double>& sentence_start_parameters, uint nIter)
{
  Math1D::Vector<double> param_grad = sentence_start_parameters;
  Math1D::Vector<double> new_param = sentence_start_parameters;
  Math1D::Vector<double> hyp_param = sentence_start_parameters;

  const uint nParams = sentence_start_parameters.size();

  for (uint j = 0; j < nParams; j++)
    sentence_start_parameters[j] =  std::max(1e-10, sentence_start_parameters[j]);

  double energy = start_prob_m_step_energy(singleton_count, norm_count, sentence_start_parameters);

  //check if normalizing the singleton count gives a better starting point
  {
    double sum = singleton_count.sum();

    if (sum > 1e-305) {

      for (uint j = 0; j < nParams; j++)
        hyp_param[j] = std::max(1e-10, singleton_count[j] / sum);

      double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < energy) {

        sentence_start_parameters = hyp_param;
        energy = hyp_energy;
      }
    }
  }

  std::cerr << "start energy: " << energy << std::endl;

  double alpha = 0.01;

  for (uint iter = 1; iter <= nIter; iter++) {

    param_grad.set_constant(0.0);

    //calculate gradient

    for (uint j = 0; j < nParams; j++)
      param_grad[j] -= singleton_count[j] / std::max(1e-10, sentence_start_parameters[j]);

    Math1D::Vector<double> addon(nParams);

    double param_sum = 0.0;
    for (uint j = 0; j < nParams; j++) {

      param_sum += std::max(1e-15, sentence_start_parameters[j]);

      addon[j] = norm_count[j] / param_sum;
      // const double addon = norm_count[j] / param_sum;

      // for (uint jj=0; jj <= j; jj++)
      //        param_grad[jj] += addon;
    }

    double addon_sum = 0.0;
    for (int j = nParams - 1; j >= 0; j--) {

      addon_sum += addon[j];
      param_grad[j] += addon_sum;
    }

    //go in neg. gradient direction
    //for (uint k = 0; k < sentence_start_parameters.size(); k++)
    //  new_param[k] = sentence_start_parameters[k] - alpha * param_grad[k];
  
    //new_param = sentence_start_parameters;
    //new_param.add_vector_multiple(param_grad, -alpha);

    Makros::go_in_neg_direction(new_param.direct_access(), new_param.size(), sentence_start_parameters.direct_access(), 
                                param_grad.direct_access(), alpha);

    //reproject
    projection_on_simplex(new_param.direct_access(), new_param.size(), 1e-10);

    //find step-size
    double best_energy = 1e300;
    bool decreasing = true;

    double lambda = 1.0;
    double best_lambda = 1.0;

    uint nIter = 0;

    while (best_energy > energy || decreasing) {

      nIter++;

      lambda *= 0.5;
      double neg_lambda = 1.0 - lambda;

      //for (uint k = 0; k < new_param.size(); k++)
      //  hyp_param[k] = neg_lambda * sentence_start_parameters[k] + lambda * new_param[k];
      Makros::assign_weighted_combination(hyp_param.direct_access(), new_param.size(), neg_lambda, sentence_start_parameters.direct_access(),
                                          lambda, new_param.direct_access());

      double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < best_energy) {

        decreasing = true;
        best_lambda = lambda;
        best_energy = hyp_energy;
      }
      else
        decreasing = false;

      if (nIter > 5 && best_energy < 0.975 * energy)
        break;

      if (nIter > 15 && lambda < 1e-12)
        break;
    }

    if (best_energy >= energy) {
      std::cerr << "CUTOFF after " << iter << " iterations" << std::endl;
      break;
    }

    double neg_best_lambda = 1.0 - best_lambda;

    //for (uint k = 0; k < new_param.size(); k++)
    //  sentence_start_parameters[k] = neg_best_lambda * sentence_start_parameters[k] + best_lambda * new_param[k];
    Makros::assign_weighted_combination(sentence_start_parameters.direct_access(), new_param.size(), neg_best_lambda,
                                        sentence_start_parameters.direct_access(), best_lambda, new_param.direct_access());

    energy = best_energy;

    if ((iter % 5) == 0)
      std::cerr << "#init m-step iter " << iter << ", energy: " << energy << std::endl;
  }
}

//use L-BFGS
void start_prob_m_step_unconstrained(const Math1D::Vector<double>& singleton_count, const Math1D::Vector<double>& norm_count,
                                     Math1D::Vector<double>& sentence_start_parameters, uint nIter, uint L)
{
  //in this formulation we use parameters p=x^2 to get an unconstrained formulation
  // here we use L-BFGS

  const uint nParams = sentence_start_parameters.size();

  Math1D::Vector<double> param_grad = sentence_start_parameters;
  Math1D::Vector<double> hyp_param = sentence_start_parameters;
  Math1D::Vector<double> work_param = sentence_start_parameters;
  Math1D::Vector<double> work_grad = sentence_start_parameters;
  Math1D::Vector<double> search_direction = sentence_start_parameters;

  for (uint j = 0; j < nParams; j++)
    sentence_start_parameters[j] = std::max(1e-10, sentence_start_parameters[j]);

  double energy = start_prob_m_step_energy(singleton_count, norm_count, sentence_start_parameters);

  //check if normalizing the singleton count gives a better starting point
  {
    double sum = singleton_count.sum();

    if (sum > 1e-305) {

      for (uint j = 0; j < nParams; j++)
        hyp_param[j] = std::max(1e-10, singleton_count[j] / sum);

      double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < energy) {

        sentence_start_parameters = hyp_param;
        energy = hyp_energy;
      }
    }
  }

  std::cerr << "start energy: " << energy << std::endl;

  Storage1D<Math1D::Vector<double> > grad_diff(L);
  Storage1D<Math1D::Vector<double> > step(L);
  Math1D::Vector<double> rho(L);

  for (uint k = 0; k < L; k++) {
    grad_diff[k].resize(nParams);
    step[k].resize(nParams);
  }

  //extract working params from the current probabilities (the probabilities are the squared working params)
  for (uint k = 0; k < nParams; k++)
    work_param[k] = sqrt(sentence_start_parameters[k]);

  double scale = 1.0;

  double line_reduction_factor = 0.75;

  uint start_iter = 1;          //changed whenever the curvature condition is violated

  for (uint iter = 1; iter <= nIter; iter++) {

    param_grad.set_constant(0.0);
    work_grad.set_constant(0.0);

    // a) calculate gradient w.r.t. the probabilities, not the parameters

    for (uint j = 0; j < nParams; j++)
      param_grad[j] -= singleton_count[j] / std::max(1e-10, sentence_start_parameters[j]);

    Math1D::Vector<double> addon(nParams);

    double param_sum = 0.0;
    for (uint j = 0; j < nParams; j++) {

      param_sum += std::max(1e-10, sentence_start_parameters[j]);

      addon[j] = norm_count[j] / param_sum;
      // const double addon = norm_count[j] / param_sum;

      // for (uint jj=0; jj <= j; jj++)
      //        param_grad[jj] += addon;
    }

    double addon_sum = 0.0;
    for (int j = nParams - 1; j >= 0; j--) {

      addon_sum += addon[j];
      param_grad[j] += addon_sum;
    }

    // b) now calculate the gradient for the actual parameters

    // each dist_grad[k] has to be diffentiated for each work_param[k']
    // we have to differentiate work_param[k]² / (\sum_k' work_param[k']²)
    // u(x) = work_param[k]², v(x) = (\sum_k' work_param[k']²)
    // quotient rule gives the total derivative  dist_grad[k] * (u'(x)*v(x) - v'(x)u(x)) / v(x)²
    // for k'!=k : dist_grad[k] * ( -2*work_param[k'] * work_param[k]²) / denom²
    // for k: dist_grad[k] * (2*work_param[k]*denom - 2*work_param[k]³) / denom²

    const double denom = scale; //work_param.sqr_norm();
    const double denom_sqr = denom * denom;

    //std::cerr << "scale: " << denom << std::endl;

    double coeff_sum = 0.0;

    for (uint k = 0; k < nParams; k++) {
      const double wp = work_param[k];
      const double grad = param_grad[k];
      const double param_sqr = wp * wp;
      const double coeff = 2.0 * grad * param_sqr / denom_sqr;

      work_grad[k] += 2.0 * grad * wp / denom;

      coeff_sum += coeff;
      // for (uint kk=0; kk < nParams; kk++)
      //   work_grad[kk] -= coeff * work_param[kk];
    }
    for (uint kk = 0; kk < nParams; kk++)
      work_grad[kk] -= coeff_sum * work_param[kk];

    double new_grad_norm = work_grad.sqr_norm();

    // c) determine the search direction

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter - 1) % L;
      Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];
      const Math1D::Vector<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k = 0; k < nParams; k++) {

        //cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
        cur_grad_diff[k] += work_grad[k];
        cur_rho += cur_grad_diff[k] * cur_step[k];
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      if (cur_curv <= 0) {
        //this can happen as our function is not convex and we do not enforce part 2 of the Wolfe conditions
        // (this cannot be done by backtracking line search, see Algorithm 3.5 in [Nocedal & Wright])
        // Our solution is to simply restart L-BFGS now
        start_iter = iter;
      }

      rho[cur_l] = 1.0 / cur_rho;
    }

    search_direction = work_grad;

    if (iter > start_iter) {

      Math1D::Vector<double> alpha(L);

      const int cur_first_iter = std::max<int>(start_iter, iter - L);

      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter - 1; prev_iter >= cur_first_iter; prev_iter--) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double cur_alpha = 0.0;
        for (uint k = 0; k < nParams; k++) {
          cur_alpha += search_direction[k] * cur_step[k];
        }
        cur_alpha *= rho[prev_l];
        alpha[prev_l] = cur_alpha;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] -= cur_alpha * cur_grad_diff[k];
        }
      }

      //we use a scaled identity as base matrix (q=r=search_direction)
      search_direction *= cur_curv;

      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < int (iter); prev_iter++) {

        uint prev_l = prev_iter % L;

        const Math1D::Vector<double>& cur_step = step[prev_l];
        const Math1D::Vector<double>& cur_grad_diff = grad_diff[prev_l];

        double beta = 0.0;
        for (uint k = 0; k < nParams; k++) {
          beta += search_direction[k] * cur_grad_diff[k];
        }
        beta *= rho[prev_l];

        const double gamma = alpha[prev_l] - beta;

        for (uint k = 0; k < nParams; k++) {
          search_direction[k] += cur_step[k] * gamma;
        }
      }

    }
    else {
      search_direction *= 1.0 / sqrt(search_direction.sqr_norm());
    }

    negate(search_direction);

    // d) line search

    double best_energy = 1e300;

    //std::cerr << "fullstep energy: " << hyp_energy << std::endl;

    double alpha = 1.0;
    double best_alpha = alpha;

    uint nIter = 0;

    bool decreasing = true;

    while (decreasing || best_energy > energy) {

      nIter++;
      if (nIter > 15 && best_energy > energy) {
        break;
      }

      if (nIter > 1)
        alpha *= line_reduction_factor;

      double sqr_sum = 0.0;
      for (uint k = 0; k < nParams; k++) {
        hyp_param[k] = work_param[k] + alpha * search_direction[k];
        sqr_sum += hyp_param[k] * hyp_param[k];
      }
      //convert to the corresponding prob. distribution
      for (uint k = 0; k < nParams; k++) {
        hyp_param[k] = std::max(1e-10, hyp_param[k] * hyp_param[k] / sqr_sum);
      }

      double hyp_energy = start_prob_m_step_energy(singleton_count, norm_count, hyp_param);

      if (hyp_energy < best_energy) {
        best_energy = hyp_energy;
        best_alpha = alpha;

        decreasing = true;
      }
      else {
        decreasing = false;
      }
    }

    if (nIter > 5)
      line_reduction_factor *= 0.9;

    //e) go to the determined point

    if (best_energy >= energy - 1e-4) {
      std::cerr << "CUTOFF after " << iter << " iterations, last gain: " << (energy - best_energy)
                << ", final energy: " << energy << std::endl;
      std::cerr << "last squared gradient norm: " << new_grad_norm << std::endl;
      break;
    }

    energy = best_energy;

    uint cur_l = (iter % L);

    Math1D::Vector<double>& cur_step = step[cur_l];
    Math1D::Vector<double>& cur_grad_diff = grad_diff[cur_l];

    scale = 0.0;
    for (uint k = 0; k < nParams; k++) {
      double step = best_alpha * search_direction[k];
      cur_step[k] = step;
      work_param[k] += step;
      scale += work_param[k] * work_param[k];

      //prepare for the next iteration
      cur_grad_diff[k] = -work_grad[k];
    }

    //calculate corresponding probability distribution (square the params and renormalize)
    for (uint k = 0; k < nParams; k++)
      sentence_start_parameters[k] =
        std::max(1e-10, work_param[k] * work_param[k] / scale);

    if ((iter % 5) == 0) {
      std::cerr << "#init m-step L-BFGS iter " << iter << ", energy: " << energy << std::endl;
      std::cerr << "sqr sum: " << scale << std::endl;
    }
  }
}

