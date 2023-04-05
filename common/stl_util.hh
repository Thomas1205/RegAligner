/*** written by Thomas Schoenemann as a private person without employment, March 2013 ****/

#ifndef STL_UTIL_HH
#define STL_UTIL_HH

#include "makros.hh"
#include "sorting.hh"
#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>

template<typename T>
T vec_sum(const std::vector<T>& vec) noexcept;

template<typename T>
T set_sum(const std::set<T>& s) noexcept;

template<typename T>
T vec_min(const std::vector<T>& vec) noexcept;

template<typename T>
T vec_max(const std::vector<T>& vec) noexcept;

template<typename T>
inline typename std::vector<T>::iterator vec_find(std::vector<T>& vec, T element) noexcept;

template<typename T>
inline typename std::vector<T>::const_iterator vec_find(const std::vector<T>& vec, T element) noexcept;

template<typename T>
inline bool contains(const std::set<T>& s, T element) noexcept;

template<typename T>
inline bool contains(const std::vector<T>& v, T element) noexcept;

template<typename T>
inline void vec_sort(std::vector<T>& vec) noexcept;

template<typename T>
inline void vec_bubble_sort(std::vector<T>& vec) noexcept;

template<typename T>
inline void vec_erase(std::vector<T>& vec, T toErase) noexcept;

template<typename T>
inline void vec_replace(std::vector<T>& vec, T toErase, T toInsert) noexcept;

template<typename T>
inline void vec_replace_maintainsort(std::vector<T>& vec, const T toErase, const T toInsert) noexcept;

template<typename T>
inline void vec_replacepos_maintainsort(std::vector<T>& vec, const size_t replace_pos, const T toInsert) noexcept;

template<typename T>
inline void sorted_vec_insert(std::vector<T>& vec, const T toInsert) noexcept;

//binary search, returns MAX_UINT if key is not found, otherwise the position in the vector
template<typename T, typename Less = std::less<T>, typename Equal = std::equal_to<T>, typename Vec = std::vector<T> >
size_t binsearch(const Vec& vec, const T key) noexcept;

template<typename T, typename Less = std::less<T>, typename Equal = std::equal_to<T>, typename Vec = std::vector<T> >
size_t binsearch_insertpos(const Vec& vec, const T key) noexcept;

//binary search in a vector with (key,value) pairs, sorted by key-values w.r.t. the operator <
//returns MAX_UINT if key is not found, otherwise the position in the vector
template<typename TK, typename TV>
size_t binsearch_keyvalue(const std::vector<std::pair<TK,TV> >& vec, const TK key) noexcept;

template<typename T1, typename T2>
class ComparePairByFirst {
public:

  bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2) noexcept;
};

template<typename T1, typename T2>
class ComparePairBySecond {
public:

  bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2) noexcept;
};

namespace Makros {

  template<typename T>
  class Typename<std::vector<T> > {
  public:

    std::string name() const
    {
      return "std::vector<" + Makros::Typename<T>() + "> ";
    }
  };

  template<typename T>
  class Typename<std::set<T> > {
  public:

    std::string name() const
    {
      return "std::set<" + Makros::Typename<T>() + "> ";
    }
  };

  template<typename T1, typename T2>
  class Typename<std::map<T1,T2> > {
  public:

    std::string name() const
    {
      return "std::map<" + Makros::Typename<T1>() + "," + Makros::Typename<T1>() + "> ";
    }
  };

  template<typename T1, typename T2>
  class Typename<std::pair<T1,T2> > {
  public:

    std::string name() const
    {
      return "std::pair<" + Makros::Typename<T1>() + "," + Makros::Typename<T1>() + "> ";
    }
  };

}

/*********** implementation *********/

template<typename T>
T vec_sum(const std::vector<T>& vec) noexcept
{
  return std::accumulate(vec.begin(),vec.end(),T());

  // T sum = T();

  // for (typename std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); it++)
  //   sum += *it;

  // return sum;
}

template<typename T>
T set_sum(const std::set<T>& s) noexcept
{
  return std::accumulate(s.begin(),s.end(),T());

  // T sum = T();

  // for (typename std::set<T>::const_iterator it = s.begin(); it != s.end(); it++)
  //   sum += *it;

  // return sum;
}

template<typename T>
T vec_min(const std::vector<T>& vec) noexcept
{
  assert(vec.size() > 0);

  return *std::min_element(vec.begin(),vec.end());
}

template<typename T>
T vec_max(const std::vector<T>& vec) noexcept
{
  assert(vec.size() > 0);

  return *std::max_element(vec.begin(),vec.end());
}

template<typename T>
inline typename std::vector<T>::const_iterator vec_find(const std::vector<T>& vec, T element) noexcept
{
  return std::find(vec.begin(),vec.end(),element);
}

template<typename T>
inline typename std::vector<T>::iterator vec_find(std::vector<T>& vec, T element) noexcept
{
  return std::find(vec.begin(),vec.end(),element);
}

template<typename T>
inline bool contains(const std::set<T>& s, T element) noexcept
{
  return s.find(element) != s.end();
}

template<typename T>
inline bool contains(const std::vector<T>& v, T element) noexcept
{
  return std::find(v.begin(),v.end(),element) != v.end();
}

template<typename T>
inline void vec_sort(std::vector<T>& vec) noexcept
{
  std::sort(vec.begin(),vec.end());
}

template<typename T>
inline void vec_bubble_sort(std::vector<T>& vec) noexcept
{
  bubble_sort(vec.data(), vec.size());
}

template<typename T>
inline void vec_erase(std::vector<T>& vec, T toErase) noexcept
{
#ifdef SAFE_MODE
  assert(vec_find(vec,toErase) != vec.end());
#endif
  vec.erase(vec_find(vec,toErase));
}

template<typename T>
inline void vec_replace(std::vector<T>& vec, T toErase, T toInsert) noexcept
{
#ifdef SAFE_MODE
  assert(vec_find(vec,toErase) != vec.end());
#endif
  *(vec_find(vec,toErase)) = toInsert;
}

template<typename T>
inline void vec_replace_maintainsort(std::vector<T>& vec, const T toErase, const T toInsert) noexcept
{
  const size_t size = vec.size();
  size_t i = 0;
  for (; i < size; i++) {
    if (vec[i] == toErase) {

      if (i > 0 && toInsert < vec[i-1]) {
        size_t npos = i-1;
        while (npos > 0 && toInsert < vec[npos-1])
          npos--;

        for (size_t k = i; k > npos; k--)
          vec[k] = vec[k-1];
        vec[npos] = toInsert;
      }
      else if (i+1 < size && vec[i+1] < toInsert) {
        size_t npos = i+1;
        while (npos+1 < size && vec[npos+1] < toInsert)
          npos++;

        for (size_t k = i; k < npos; k++)
          vec[k] = vec[k+1];
        vec[npos] = toInsert;
      }
      else {
        vec[i] = toInsert;
      }

      break;
    }
  }

  assert(i < size);
  //assert(is_sorted(vec.data(), size));
}

template<typename T>
inline void vec_replacepos_maintainsort(std::vector<T>& vec, const size_t replace_pos, const T toInsert) noexcept
{
  const size_t size = vec.size();
  assert(replace_pos < size);

  if (replace_pos > 0 && toInsert < vec[replace_pos-1]) {
    size_t npos = replace_pos-1;
    while (npos > 0 && toInsert < vec[npos-1])
      npos--;

    for (size_t k = replace_pos; k > npos; k--)
      vec[k] = vec[k-1];
    vec[npos] = toInsert;
  }
  else if (replace_pos+1 < size && vec[replace_pos+1] < toInsert) {
    size_t npos = replace_pos+1;
    while (npos+1 < size && vec[npos+1] < toInsert)
      npos++;

    for (size_t k = replace_pos; k < npos; k++)
      vec[k] = vec[k+1];
    vec[npos] = toInsert;
  }
  else {
    vec[replace_pos] = toInsert;
  }
}

template<typename T>
inline void sorted_vec_insert(std::vector<T>& vec, const T toInsert) noexcept
{
  //standard find may be faster for short vectors
  uint inspos = binsearch_insertpos(vec, toInsert);
  vec.insert(vec.begin() + inspos, toInsert);
}

template<typename T1, typename T2>
bool ComparePairByFirst<T1,T2>::operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2) noexcept
{
  if (p1.first == p2.first)
    return (p1.second < p2.second);

  return p1.first < p2.first;
}

template<typename T1, typename T2>
bool ComparePairBySecond<T1,T2>::operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2) noexcept
{
  if (p1.second == p2.second)
    return (p1.first < p2.first);

  return p1.second < p2.second;
}

//binary search, returns MAX_UINT if key is not found, otherwise the position in the vector
template<typename T, typename Less = std::less<T>, typename Equal = std::equal_to<T>, typename Vec = std::vector<T> >
size_t binsearch(const Vec& vec, const T key) noexcept
{
  const size_t size = vec.size();
  static const Less less;
  static const Equal equal;
  if (size == 0 || less(key,vec[0]) || less(vec[size-1],key))
    return MAX_UINT;

  size_t lower = 0;
  size_t upper = size-1;
  if (equal(vec[lower],key))
    return lower;
  if (equal(vec[upper],key))
    return upper;

  while (lower+1 < upper) {
    assert(less(vec[lower],key));
    assert(less(key,vec[upper]));

    const size_t middle = (lower+upper) >> 1;  // (lower+upper)/2;
    assert(less(lower,middle) && less(middle,upper));
    if (equal(vec[middle],key))
      return middle;
    else if (less(vec[middle],key))
      lower = middle;
    else
      upper = middle;
  }

  return MAX_UINT;
}

template<typename T, typename Less = std::less<T>, typename Equal = std::equal_to<T>, typename Vec = std::vector<T> >
size_t binsearch_insertpos(const Vec& vec, const T key) noexcept
{
  const size_t size = vec.size();
  static const Less less;
  static const Equal equal; 
  if (size == 0 || !less(vec[0],key))
    return 0;

  if (less(vec[size-1],key))
    return size;

  size_t lower = 0;
  size_t upper = size-1;
  if (equal(vec[upper],key))
    return upper;

  while (lower+1 < upper) {
    assert(less(vec[lower],key));
    assert(less(key,vec[upper]));

    const size_t middle = (lower+upper) >> 1;  // (lower+upper)/2;
    assert(middle > lower && middle < upper);
    if (equal(vec[middle],key))
      return middle;
    else if (less(vec[middle],key))
      lower = middle;
    else
      upper = middle;
  }

  assert(lower+1 == upper);
  return upper;
}

template<typename TK, typename TV>
size_t binsearch_keyvalue(const std::vector<std::pair<TK,TV> >& vec, const TK key) noexcept
{
  const size_t size = vec.size();
  if (size == 0 || key < vec[0].first || key > vec[size-1].first)
    return MAX_UINT;

  size_t lower = 0;
  size_t upper = size-1;
  if (vec[lower].first == key)
    return lower;
  if (vec[upper].first == key)
    return upper;

  while (lower+1 < upper) {
    assert(vec[lower].first < key);
    assert(vec[upper].first > key);

    const size_t middle = (lower+upper) >> 1;  // (lower+upper)/2;
    assert(middle > lower && middle < upper);
    if (vec[middle].first == key)
      return middle;
    else if (vec[middle].first < key)
      lower = middle;
    else
      upper = middle;
  }

  return MAX_UINT;
}

#endif
