/*** written by Thomas Schoenemann as a private person without employment, March 2013 ****/

#ifndef STL_UTIL_HH
#define STL_UTIL_HH

#include "makros.hh"
#include <vector>
#include <set>
#include <map>
#include <numeric>

template <typename T>
T vec_sum(const std::vector<T>& vec);

template <typename T>
T set_sum(const std::set<T>& s);

template <typename T>
T vec_min(const std::vector<T>& vec);

template <typename T>
T vec_max(const std::vector<T>& vec);

template <typename T>
inline typename std::vector<T>::iterator vec_find(std::vector<T>& vec, T element);

template <typename T>
inline typename std::vector<T>::const_iterator vec_find(const std::vector<T>& vec, T element);

template <typename T>
inline bool contains(const std::set<T>& s, T element);

template <typename T>
inline bool contains(const std::vector<T>& v, T element);

template <typename T>
inline void vec_sort(std::vector<T>& vec);

template <typename T>
inline void vec_erase(std::vector<T>& vec, T toErase);

template <typename T>
inline void vec_replace(std::vector<T>& vec, T toErase, T toInsert);

template<typename T1, typename T2>
class ComparePairByFirst {
public:

  bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2);
};

template<typename T1, typename T2>
class ComparePairBySecond {
public:

  bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2);
};


//binary search in a vector with (key,value) pairs, sorted by key-values w.r.t. the operator <
//returns MAX_UINT if key is not found, otherwise the position in the vector
template<typename TK, typename TV>
size_t binsearch_keyvalue(const std::vector<std::pair<TK,TV> >& vec, TK key);


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

template <typename T>
T vec_sum(const std::vector<T>& vec)
{

  return std::accumulate(vec.begin(),vec.end(),T());

  // T sum = T();

  // for (typename std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); it++)
  //   sum += *it;

  // return sum;
}

template <typename T>
T set_sum(const std::set<T>& s)
{

  return std::accumulate(s.begin(),s.end(),T());

  // T sum = T();

  // for (typename std::set<T>::const_iterator it = s.begin(); it != s.end(); it++)
  //   sum += *it;

  // return sum;
}

template <typename T>
T vec_min(const std::vector<T>& vec)
{

  assert(vec.size() > 0);

  return *std::min_element(vec.begin(),vec.end());
}

template <typename T>
T vec_max(const std::vector<T>& vec)
{

  assert(vec.size() > 0);

  return *std::max_element(vec.begin(),vec.end());
}


template <typename T>
inline typename std::vector<T>::const_iterator vec_find(const std::vector<T>& vec, T element)
{

  return std::find(vec.begin(),vec.end(),element);
}

template <typename T>
inline typename std::vector<T>::iterator vec_find(std::vector<T>& vec, T element)
{

  return std::find(vec.begin(),vec.end(),element);
}

template <typename T>
inline bool contains(const std::set<T>& s, T element)
{

  return s.find(element) != s.end();
}

template <typename T>
inline bool contains(const std::vector<T>& v, T element)
{

  return std::find(v.begin(),v.end(),element) != v.end();
}

template <typename T>
inline void vec_sort(std::vector<T>& vec)
{

  std::sort(vec.begin(),vec.end());
}

template <typename T>
inline void vec_erase(std::vector<T>& vec, T toErase)
{
#ifdef SAFE_MODE
  assert(vec_find(vec,toErase) != vec.end());
#endif
  vec.erase(vec_find(vec,toErase));
}


template <typename T>
inline void vec_replace(std::vector<T>& vec, T toErase, T toInsert)
{

#ifdef SAFE_MODE
  assert(vec_find(vec,toErase) != vec.end());
#endif
  *(vec_find(vec,toErase)) = toInsert;
}

template<typename T1, typename T2>
bool ComparePairByFirst<T1,T2>::operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2)
{

  if (p1.first == p2.first)
    return (p1.second < p2.second);

  return p1.first < p2.first;
}

template<typename T1, typename T2>
bool ComparePairBySecond<T1,T2>::operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2)
{

  if (p1.second == p2.second)
    return (p1.first < p2.first);

  return p1.second < p2.second;
}


template<typename TK, typename TV>
size_t binsearch_keyvalue(const std::vector<std::pair<TK,TV> >& vec, TK key)
{

  size_t size = vec.size();
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

    size_t middle = (lower+upper)/2;
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
