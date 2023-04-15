/*** written by Thomas Schoenemann as a private person, since October 2009 ***/

#include "training_common.hh"
#include "projection.hh"
#include "storage_util.hh"
#include "storage_stl_interface.hh"
#include "sorted_set.hh"
#include "sorted_map.hh"
#include "conditional_m_steps.hh"

#include <vector>
#include <set>
#include <unordered_set>
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

//returns the number of ignored entries
uint set_prior_dict_weights(const std::set<PriorPair>& known_pairs, const CooccuringWordsType& wcooc,
                            floatSingleWordDictionary prior_weight, Math1D::Vector<float>& prior_t0_weight,
                            float dict_regularity)
{
  uint nIgnored = 0;

  uint nTargetWords = prior_weight.size();

  for (uint i = 0; i < nTargetWords; i++)
    prior_weight[i].set_constant(dict_regularity);
  prior_t0_weight.set_constant(dict_regularity);

  std::cerr << "processing read list" << std::endl;

  for (std::set<PriorPair>::const_iterator it = known_pairs.begin(); it != known_pairs.end(); it++) {

    uint tword = it->t_idx_;
    uint sword = it->s_idx_;

    if (tword >= wcooc.size()) {
      std::cerr << "tword out of range: " << tword << std::endl;
    }

    if (sword == 0) {
      if (prior_t0_weight.size() > 0)
        prior_t0_weight[sword-1] = dict_regularity * it->multiplicator_;
    }
    else if (tword == 0) {
      prior_weight[0][sword-1] = dict_regularity * it->multiplicator_;
    }
    else {
      //uint pos = std::lower_bound(wcooc[tword].direct_access(), wcooc[tword].direct_access() + wcooc[tword].size(), sword) - wcooc[tword].direct_access();
      uint pos = Routines::binsearch(wcooc[tword].direct_access(), sword, wcooc[tword].size());

      if (pos < wcooc[tword].size() && wcooc[tword][pos] == sword) {
        prior_weight[tword][pos] = dict_regularity * it->multiplicator_;
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

struct WordHash {

  size_t operator()(uint key)
  {
    return (key & 255);
  }
};

void find_cooccuring_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
                           const Storage1D<Math1D::Vector<uint> >& additional_source, const Storage1D<Math1D::Vector<uint> >& additional_target,
                           uint nSourceWords, uint nTargetWords, CooccuringWordsType& cooc)
{
  const size_t nSentences = source.size();
  assert(nSentences == target.size());
  cooc.resize_dirty(nTargetWords);

#if 1
  //NOTE: tree sets are terribly inefficient

  NamedStorage1D<SortedSet<uint> > coocset(nTargetWords, MAKENAME(coocset));

  for (size_t s = 0; s < nSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    SortedSet<uint> source_set;
    for (uint j = 0; j < curJ; j++)
      source_set.insert(cur_source[j]);
    const std::vector<uint>& unsorted = source_set.sorted_data();
    const uint len = unsorted.size();

    for (uint i = 0; i < curI; i++) {
      const uint t_idx = cur_target[i];
      assert(t_idx < nTargetWords);
      SortedSet<uint>& cur_set = coocset[t_idx];

      for (uint k = 0; k < len; k++)
        cur_set.insert(unsorted[k]);
    }
  }

  const size_t nAddSentences = additional_source.size();
  assert(nAddSentences == additional_target.size());

  for (size_t s = 0; s < nAddSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint nCurSourceWords = additional_source[s].size();
    const uint nCurTargetWords = additional_target[s].size();

    const Storage1D<uint>& cur_source = additional_source[s];
    const Storage1D<uint>& cur_target = additional_target[s];

    for (uint i = 0; i < nCurTargetWords; i++) {
      const uint t_idx = cur_target[i];
      assert(t_idx < nTargetWords);

      SortedSet<uint>& cur_set = coocset[t_idx];

      for (uint j = 0; j < nCurSourceWords; j++) {
        const uint s_idx = cur_source[j];
        assert(s_idx < nSourceWords);

        cur_set.insert(s_idx);
      }
    }
  }

  for (uint i = 0; i < nTargetWords; i++) {
    assign(cooc[i],coocset[i].sorted_data());
    coocset[i].clear();
  }

#else
  NamedStorage1D<std::set<uint> > coocset(nTargetWords, MAKENAME(coocset));

  for (size_t s = 0; s < nSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint curJ = source[s].size();
    const uint curI = target[s].size();

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    std::set<uint> source_set;
    for (uint j = 0; j < curJ; j++)
      source_set.insert(cur_source[j]);

    //iterating over a vector is faster than iterating over a set -> copy
    std::vector<uint> source_words;
    assign(source_words, source_set);
    //source_words.reserve(source_set.size());
    //for (std::set<uint>::const_iterator it = source_set.begin(); it != source_set.end(); it++)
    //  source_words.push_back(*it);

    const uint nWords = source_words.size();
    for (uint i = 0; i < curI; i++) {
      const uint t_idx = cur_target[i];
      assert(t_idx < nTargetWords);

      std::set<uint>& cur_set = coocset[t_idx];
      for (uint k = 0; k < nWords; k++)
        cur_set.insert(source_words[k]);
    }
  }

  const size_t nAddSentences = additional_source.size();
  assert(nAddSentences == additional_target.size());

  for (size_t s = 0; s < nAddSentences; s++) {

    if ((s % 10000) == 0)
      std::cerr << "sentence pair number " << s << std::endl;

    const uint nCurSourceWords = additional_source[s].size();
    const uint nCurTargetWords = additional_target[s].size();

    const Storage1D<uint>& cur_source = additional_source[s];
    const Storage1D<uint>& cur_target = additional_target[s];

    for (uint i = 0; i < nCurTargetWords; i++) {
      const uint t_idx = cur_target[i];
      assert(t_idx < nTargetWords);

      std::set<uint>& cur_set = coocset[t_idx];

      for (uint j = 0; j < nCurSourceWords; j++) {
        const uint s_idx = cur_source[j];
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
#endif

  uint max_cooc = 0;
  double sum_cooc = 0.0;
  for (uint i = 1; i < nTargetWords; i++) {
    sum_cooc += cooc[i].size();
    max_cooc = std::max<uint>(max_cooc, cooc[i].size());
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

  delete[] cline;
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
                                 Storage1D<Storage1D<uint> >& cooc, uint minOcc)
{
  const size_t nSentences = sentence.size();
  cooc.resize(voc_size);
  for (uint k = 0; k < voc_size; k++)
    cooc[k].resize(0);

  cooc[0].resize(voc_size);
  for (uint k = 0; k < voc_size; k++)
    cooc[0][k] = k;

  Storage1D<SortedMap<uint,uint> > cc(voc_size);

  for (size_t s = 0; s < nSentences; s++) {

    //std::cerr << "s: " << s << std::endl;

    const Storage1D<uint>& cur_sentence = sentence[s];

    const uint curI = cur_sentence.size();

    for (uint i1 = 0; i1 < curI - 1; i1++) {

      const uint w1 = cur_sentence[i1];

      for (uint i2 = i1 + 1; i2 < curI; i2++) {

        const uint w2 = cur_sentence[i2];
        cc[w1][w2]++;
      }
    }
  }

  for (uint k = 1; k < voc_size; k++) {
    uint j = 0;
    const std::vector<uint>& key = cc[k].key();
    const std::vector<uint>& value = cc[k].value();
    for (uint i = 0; i < key.size(); i++) {
      if (value[i] >= minOcc)
        j++;
    }
    cooc[k].resize(j);
    j = 0;
    for (uint i = 0; i < key.size(); i++) {
      if (value[i] >= minOcc) {
        cooc[k][j] = key[i];
        j++;
      }
    }
  }
}

void monolingual_pairs_cooc_count(const Storage1D<Math1D::Vector<uint> >& sentence, const Storage1D<Storage1D<uint> >& t_cooc,
                                  Storage1D<Storage1D<uint> >& t_cooc_count)
{
  t_cooc_count.resize(t_cooc.size());
  for (uint k = 0; k < t_cooc.size(); k++)
    t_cooc_count[k].resize(t_cooc[k].size(), 0);

  size_t nSentences = sentence.size();

  for (size_t s = 0; s < nSentences; s++) {

    //std::cerr << "s: " << s << std::endl;

    const Storage1D<uint>& cur_sentence = sentence[s];

    const uint curI = cur_sentence.size();

    for (uint i1 = 0; i1 < curI - 1; i1++) {

      uint w1 = cur_sentence[i1];

      for (uint i2 = i1 + 1; i2 < curI; i2++) {

        uint w2 = cur_sentence[i2];
        const uint k = Routines::binsearch(t_cooc[w1].direct_access(), w2, t_cooc[w1].size());

        if (k < t_cooc[w1].size())
          t_cooc_count[w1][k] += 1;
      }
    }
  }
}

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    const Storage1D<Storage1D<uint> >& target_cooc, Storage1D<Storage1D<Storage1D<uint> > >& st_cooc, uint minOcc)
{
  st_cooc.resize(target_cooc.size());
  for (uint i = 0; i < target_cooc.size(); i++)
    st_cooc[i].resize(target_cooc[i].size());

  const size_t nSentences = source.size();

  Storage1D<Storage1D<SortedMap<uint,uint> > > st_cc(target_cooc.size());
  for (uint i = 0; i < target_cooc.size(); i++)
    st_cc[i].resize(target_cooc[i].size());

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint curJ = cur_source.size();
    const uint curI = cur_target.size();

    for (uint i1 = 0; i1 < curI; i1++) {

      uint start_i2 = (i1 == 0) ? 0 : i1 + 1;

      uint t1 = (i1 == 0) ? 0 : cur_target[i1 - 1];

      const Storage1D<uint>& cur_tcooc = target_cooc[t1];

      for (uint i2 = start_i2; i2 <= curI; i2++) {

        uint t2 = (i2 == 0) ? 0 : cur_target[i2 - 1];
        uint k = Routines::binsearch(cur_tcooc.direct_access(), t2, cur_tcooc.size());

        if (k >= cur_tcooc.size())
          continue;

        Storage1D<uint>& cur_scooc = st_cooc[t1][k];
        SortedMap<uint,uint>& cur_scc = st_cc[t1][k];

        for (uint j = 0; j < curJ; j++) {

          const uint s_idx = cur_source[j];
          cur_scc[s_idx]++;
        }
      }
    }
  }

  for (uint i = 0; i < st_cc.size(); i++) {
    for (uint k = 0; k < st_cc[i].size(); k++) {
      uint nKeep = 0;
      const std::vector<uint>& key = st_cc[i][k].key();
      const std::vector<uint>& value = st_cc[i][k].value();
      for (uint l=0; l < value.size(); l++) {
        if (l == 0 || i == 0 || value[l] >= minOcc)
          nKeep++;
      }
      uint j = 0;
      st_cooc[i][k].resize_dirty(nKeep);
      for (uint l=0; l < value.size(); l++) {
        if (l == 0 || i == 0 || value[l] >= minOcc) {
          st_cooc[i][k][j] = key[l];
          j++;
        }
      }
    }
  }
}

void find_cooc_target_pairs_and_source_words(const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target,
    std::map<std::pair<uint,uint>,std::set<uint> >& cooc)
{
  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  for (size_t s = 0; s < nSentences; s++) {

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

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  for (size_t s = 0; s < nSentences; s++) {

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
  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  for (size_t s = 0; s < nSentences; s++) {

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
    uint /*nSourceWords */, uint nTargetWords, Storage1D<Storage1D<std::pair<uint,Storage1D<std::pair<uint,uint> > > > >& cooc)
{
  cooc.resize(nTargetWords);
  cooc[0].resize(nTargetWords);
  for (uint k = 0; k < nTargetWords; k++) {
    cooc[0][k].first = k;
  }

  const size_t nSentences = source.size();
  assert(nSentences == target.size());

  /**** stage 1: find coocuring target words ****/

  for (size_t s = 0; s < nSentences; s++) {

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
  for (size_t s = 0; s < nSentences; s++) {

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

  const size_t nSentences = target.size();
  uint max_tlength = 0;

  for (size_t s = 0; s < nSentences; s++) {
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
  const size_t nSentences = source.size();
  slookup.resize_dirty(nSentences);

  for (size_t s = 0; s < nSentences; s++) {

    const Storage1D<uint>& cur_source = source[s];
    const Storage1D<uint>& cur_target = target[s];

    const uint J = cur_source.size();
    const uint I = cur_target.size();

    SingleLookupTable& cur_lookup = slookup[s];

    if (J * I <= max_size)
      cur_lookup.resize_dirty(J, I);

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

          for (uint j = 0; j < J; j++) {

            const uint sidx = cur_source[j];
            const uint prev_occ = s_prev_occ[j];

            if (prev_occ != MAX_UINT) {
              if (cur_lookup.size() > 0)
                cur_lookup(j, i) = cur_lookup(prev_occ, i);
            }
            else {

              const uint idx = Routines::binsearch(start, sidx, cur_size);
              if (idx >= cur_size) {
                INTERNAL_ERROR << " word not found. Exiting." << std::endl;
                exit(1);
              }

              if (cur_lookup.size() > 0) {
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
      //const uint* end = cur_cooc.direct_access() + cur_size;

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

            uint idx = 0;
            //const uint guess_idx = sidx-1;
            const uint guess_idx = floor((sidx - 1) * ratio);

            if (guess_idx < cur_size) {

              const uint* guess = start + guess_idx;
              const uint p = *(guess);
              if (p == sidx)
                idx = guess_idx;
              else if (p < sidx)
                idx = Routines::binsearch(guess+1, sidx, cur_size-guess_idx-1); //std::lower_bound(guess, end, sidx);
              else
                idx = Routines::binsearch(start, sidx, guess_idx); //std::lower_bound(start, guess, sidx);
            }
            else
              idx = Routines::binsearch(start, sidx, cur_size); //std::lower_bound(start, end, sidx);

#ifdef SAFE_MODE
            if (idx >= cur_size) {
              INTERNAL_ERROR << " word not found. Exiting." << std::endl;
              exit(1);
            }
#endif
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
  assert(beta > 0.0);
  return 1.0 - std::exp(-x / beta);
}

double prob_pen_prime(double x, double beta)
{
  assert(beta > 0.0);
  return -prob_penalty(x, beta) / beta;
}

void update_dict_from_counts(const UnnamedSingleWordDictionary& fdict_count, const floatUnnamedSingleWordDictionary& prior_weight,
                             size_t nSentences, double dict_weight_sum, bool smoothed_l0, double l0_beta, uint nDictStepIter, UnnamedSingleWordDictionary& dict,
                             double min_prob, bool unconstrained_m_step, double alpha)
{
  if (dict_weight_sum > 0.0) {

    for (uint i = 0; i < fdict_count.size(); i++) {

      //std::cerr << "i: " << i << ", " << dict[i].size() << " entries" << std::endl;

      const Math1D::Vector<double>& cur_count = fdict_count[i];

      const double sum = cur_count.sum();
      if (sum == 0.0)
        continue;

      const Math1D::Vector<float>& cur_prior = prior_weight[i];
      Math1D::Vector<double>& cur_dict = dict[i];

      bool prior_const = cur_prior.is_constant();


      Math1D::Vector<double> hyp_dict = cur_count;

      const double prev_sum = cur_dict.sum();

      const double inv_sum = 1.0 / sum;
      assert(!isnan(inv_sum));

      for (uint k = 0; k < cur_count.size(); k++) {
        hyp_dict[k] *= prev_sum * inv_sum;
      }

      if (prior_const && cur_prior[0] == 0.0) {
        cur_dict = hyp_dict;
      }
      else {

        const double cur_energy = single_dict_m_step_energy(cur_count, cur_prior, nSentences, cur_dict, smoothed_l0, l0_beta);
        const double hyp_energy = single_dict_m_step_energy(cur_count, cur_prior, nSentences, hyp_dict, smoothed_l0, l0_beta);

        //NOTE: if the entries in prior_weight[i] are all the same, hyp_energy should always be small or equal to cur_energy
        if (hyp_energy < cur_energy) {
          cur_dict = hyp_dict;
          //std::cerr << "switching to energy " << hyp_energy << std::endl;
        }

        if (!unconstrained_m_step) {
          if (i != 0)
            //if (true)
            single_dict_m_step(cur_count, cur_prior, nSentences, cur_dict, alpha, nDictStepIter, smoothed_l0, l0_beta, min_prob, true, prior_const);
          else {
            single_dict_m_step(cur_count, cur_prior, nSentences, cur_dict, alpha, nDictStepIter, smoothed_l0, l0_beta, min_prob, true, prior_const,
                               false);

            // ConstrainedSmoothMinizerOptions options;
            // options.min_param_entry_ = 1e-300;
            // options.initial_line_reduction_factor_ = 0.5;

            // double slack = 1.0 - dict[i].sum();
            // DictMStepMinimizer minimizer(options, dict[i].size()+1, cur_count, cur_prior, smoothed_l0, l0_beta, nSentences, slack);
            // Math1D::Vector<double> params(dict[i].size()+1);
            // params[dict[i].size()] = slack;
            // for (uint k = 0; k < dict[i].size(); k++)
            // params[k] = dict[i][k];
            // minimizer.optimize_projected_lbfgs(params, 5);
            // for (uint k = 0; k < dict[i].size(); k++)
            // dict[i][k] = params[k];
          }
        }
        else
          single_dict_m_step_unconstrained(cur_count, cur_prior, nSentences, cur_dict, nDictStepIter, smoothed_l0, l0_beta, 5, min_prob, prior_const);

        for (uint k = 0; k < cur_count.size(); k++)
          cur_dict[k] = std::max(min_prob, cur_dict[k]);
      }
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

//NOTE: the function to be minimized can be decomposed over the target words
void dict_m_step(const SingleWordDictionary& fdict_count, const floatSingleWordDictionary& prior_weight, size_t nSentences,
                 SingleWordDictionary& dict, double alpha, uint nIter, bool smoothed_l0, double l0_beta, double min_prob)
{
  for (uint k = 0; k < dict.size(); k++)
    single_dict_m_step(fdict_count[k], prior_weight[k], nSentences, dict[k], alpha, nIter, smoothed_l0, l0_beta, min_prob);
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

double dict_reg_term(const SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight, double l0_beta)
{

  bool smoothed_l0 = (l0_beta > 0.0);

  double reg_term = 0.0;
  for (uint i = 0; i < dict.size(); i++) {

    const Math1D::Vector<double>& cur_dict = dict[i];
    const Math1D::Vector<float>& cur_prior = prior_weight[i];

    assert(cur_dict.size() == cur_prior.size());

    if (smoothed_l0) {
      for (uint k = 0; k < cur_dict.size(); k++)
        reg_term += cur_prior[k] * prob_penalty(cur_dict[k], l0_beta);
    }
    else {

      const ALIGNED16 float* pdata = cur_prior.direct_access();
      const ALIGNED16 double* ddata = cur_dict.direct_access();

      reg_term += std::inner_product(pdata,pdata+cur_prior.size(),ddata, 0.0);

      //for (uint k = 0; k < cur_dict.size(); k++)
      //  reg_term += cur_prior[k] * cur_dict[k];
    }
  }

  return reg_term;
}

