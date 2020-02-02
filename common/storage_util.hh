/**** written by Thomas Schoenemann as a private person without employment, March 2013 ****/

#ifndef STORAGE_UTIL_HH
#define STORAGE_UTIL_HH


#include "storage1D.hh"
#include "vector.hh"
#include "matrix.hh"
#include "tensor.hh"
#include "sorting.hh"

template<typename T>
inline void negate(Math1D::Vector<T>& vec)
{
  const size_t size = vec.size();
  for (size_t k=0; k < size; k++)
    vec[k] = -vec[k];
}

template<typename T>
inline void negate(Math2D::Matrix<T>& mat)
{
  const size_t size = mat.size();
  for (size_t k=0; k < size; k++)
    mat.direct_access(k) = -mat.direct_access(k);
}

template<typename T>
inline void negate(Math3D::Tensor<T>& ten)
{
  const size_t size = ten.size();
  for (size_t k=0; k < size; k++)
    ten.direct_access(k) = -ten.direct_access(k);
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
#endif
