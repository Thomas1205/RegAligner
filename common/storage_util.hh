/**** written by Thomas Schoenemann as a private person without employment, March 2013 ****/

#ifndef STORAGE_UTIL_HH
#define STORAGE_UTIL_HH

#include "vector.hh"
#include "flexible_storage1D.hh"
#include "matrix.hh"
#include "tensor.hh"
#include "sorting.hh"
#include "routines.hh"

//can use std::iota from <numeric> since C++-11
template<typename T, typename ST>
inline void set_idfunc(Storage1D<T, ST>& vec)
{
  const ST size = vec.size();
  ST i;
  for (i = 0; i < size; i++)
    vec[i] = (T) i;
}

//can use std::iota from <numeric> since C++-11
template<typename T, typename ST>
inline void set_idfunc(FlexibleStorage1D<T, ST>& vec)
{
  const ST size = vec.size();
  ST i;
  for (i = 0; i < size; i++)
    vec[i] = (T) i;
}

template<typename T, typename ST>
inline bool approx_equal(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2, T tolerance) leaf_const;

template<typename T, typename ST>
inline bool approx_equal(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2, T tolerance)
{
  if (v1.size() != v2.size())
    return false;
  for (ST i = 0; i < v1.size(); i++) {
    if (Makros::abs<T>(v1[i] - v2[i]) > tolerance)
      return false;
  }

  return true;
}

template<typename T, typename ST>
inline bool approx_equal(const FlexibleStorage1D<T,ST>& v1, const FlexibleStorage1D<T,ST>& v2, T tolerance) leaf_const;

template<typename T, typename ST>
inline bool approx_equal(const FlexibleStorage1D<T,ST>& v1, const FlexibleStorage1D<T,ST>& v2, T tolerance)
{
  if (v1.size() != v2.size())
    return false;
  for (ST i = 0; i < v1.size(); i++) {
    if (Makros::abs<T>(v1[i] - v2[i]) > tolerance)
      return false;
  }

  return true;
}

namespace Math1D {

  template<typename T>
  inline void negate(Math1D::Vector<T>& vec)
  {
    const size_t size = vec.size();
    for (size_t k=0; k < size; k++)
      vec[k] = -vec[k];
  }
}

namespace Math2D {

  template<typename T>
  inline void negate(Math2D::Matrix<T>& mat)
  {
    const size_t size = mat.size();
    for (size_t k=0; k < size; k++)
      mat.direct_access(k) = -mat.direct_access(k);
  }
}

namespace Math3D {

  template<typename T>
  inline void negate(Math3D::Tensor<T>& ten)
  {
    const size_t size = ten.size();
    for (size_t k=0; k < size; k++)
      ten.direct_access(k) = -ten.direct_access(k);
  }
}

template<typename T>
inline void sort_storage1D(Storage1D<T>& stor)
{
  std::sort(stor.direct_access(),stor.direct_access()+stor.size());
}

template<typename T>
inline void bubble_sort_storage1D(Storage1D<T>& stor)
{
  bubble_sort(stor.direct_access(),stor.size());
}

template<typename T, typename ST>
inline ST find_in_storage1D(const Storage1D<T,ST>& stor, T element)
{
  return std::find(stor.direct_access(),stor.direct_access()+stor.size(),element) - stor.direct_access();
}

template<typename T, typename ST>
inline ST find_in_flexstorage1D(const FlexibleStorage1D<T,ST>& stor, T element)
{
  return std::find(stor.direct_access(),stor.direct_access()+stor.size(),element) - stor.direct_access();
}

template<typename T, typename ST>
inline bool contains(const Storage1D<T,ST>& stor, T element)
{
  T* end = stor.direct_access()+stor.size();

  return (std::find(stor.direct_access(),end,element) != end);
}

template <typename T, typename ST>
inline void vec_replace_maintainsort(FlexibleStorage1D<T,ST>& vec, const T toErase, const T toInsert)
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

template <typename T, typename ST>
inline void large_vec_replace_maintainsort(FlexibleStorage1D<T,ST>& vec, const T toErase, const T toInsert)
{
  const size_t size = vec.size();
  size_t i = Routines::binsearch(vec.direct_access(), toErase, vec.size());
  assert(i < size);

  if (i > 0 && toInsert < vec[i-1]) {
    size_t npos = i-1;
    while (npos > 0 && toInsert < vec[npos-1])
      npos--;

    Routines::upshift_array(vec.direct_access(), i, 1, npos);
    //for (size_t k = i; k > npos; k--)
    //  vec[k] = vec[k-1];
    vec[npos] = toInsert;
  }
  else if (i+1 < size && vec[i+1] < toInsert) {
    size_t npos = i+1;
    while (npos+1 < size && vec[npos+1] < toInsert)
      npos++;

    Routines::downshift_array(vec.direct_access(), i, npos, 1);
    //for (size_t k = i; k < npos; k++)
    // vec[k] = vec[k+1];
    vec[npos] = toInsert;
  }
  else {
    vec[i] = toInsert;
  }
  //assert(is_sorted(vec.data(), size));
}

#endif
