/**** written by Thomas Schoenemann as a private person without employment, April 2013 ****/

#ifndef STORAGE_STL_INTERFACE
#define STORAGE_STL_INTERFACE

#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include "storage1D.hh"
#include "flexible_storage1D.hh"
#include "storage2D.hh"
#include "storage3D.hh"

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const std::vector<T2>& source) noexcept;

template<typename T1, typename T2, typename ST>
void assign(FlexibleStorage1D<T1,ST>& target, const std::vector<T2>& source) noexcept;

template<typename T, typename ST, typename K>
void assign(Storage1D<std::pair<K,T>,ST>& target, const std::map<K,T>& source) noexcept;

template<typename T, typename ST, typename K, typename H>
void assign(Storage1D<std::pair<K,T>,ST>& target, const std::unordered_map<K,T,H>& source) noexcept;

template<typename T, typename ST, typename K>
void assign(std::vector<std::pair<K,T>,ST>& target, const std::map<K,T>& source) noexcept;

template<typename T, typename ST, typename K, typename H>
void assign(std::vector<std::pair<K,T>,ST>& target, const std::unordered_map<K,T,H>& source) noexcept;

template<typename T, typename ST>
void assign(Storage1D<T,ST>& target, const std::set<T>& source) noexcept;

template<typename T, typename ST, typename K>
void assign(Storage1D<K,ST>& target1, Storage1D<T,ST>& target2, const std::map<K,T>& source) noexcept;

template<typename T, typename ST, typename K, typename H>
void assign(Storage1D<K,ST>& target1, Storage1D<T,ST>& target2, const std::unordered_map<K,T,H>& source) noexcept;

template<typename T, typename K>
void assign(std::vector<K>& target1, std::vector<T>& target2, const std::map<K,T>& source) noexcept;

template<typename T, typename K, typename H>
void assign(std::vector<K>& target1, std::vector<T>& target2, const std::unordered_map<K,T,H>& source) noexcept;

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const Storage1D<T2,ST>& source) noexcept;

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const FlexibleStorage1D<T2,ST>& source) noexcept;

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const Storage1D<T2,ST>& source) noexcept;

template<typename T1, typename T2, typename ST>
void assign(Storage2D<T1,ST>& target, const Storage2D<T2,ST>& source) noexcept;

template<typename T1, typename T2, typename ST>
void assign(Storage3D<T1,ST>& target, const Storage3D<T2,ST>& source) noexcept;


/************** implementation *************/

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const std::vector<T2>& source) noexcept
{
  //std::copy() is sometimes faster, but not consistently for different vector sizes (with g++)

  target.resize_dirty(source.size());
  for (uint k=0; k < source.size(); k++)
    target[k] = source[k];
}

template<typename T1, typename T2, typename ST>
void assign(FlexibleStorage1D<T1,ST>& target, const std::vector<T2>& source) noexcept
{
  target.clear();
  if (target.reserved_size() < source.size())
    target.reserve(source.size());

  for (size_t i=0; i < source.size(); i++)
    target.append(source[i]);
}

template<typename T1, typename T2, typename ST>
void assign_copy(Storage1D<T1,ST>& target, const std::vector<T2>& source) noexcept
{
  target.resize_dirty(source.size());
  std::copy(source.cbegin(),source.cend(),target.direct_access());
}

template<typename T, typename ST, typename K>
void assign(Storage1D<std::pair<K,T>,ST>& target, const std::map<K,T>& source) noexcept
{
  //TODO: think about std::copy()
  target.resize_dirty(source.size());
  uint k=0;
  for (typename std::map<K,T>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target[k] = *it;
    k++;
  }
}

template<typename T, typename ST, typename K, typename H>
void assign(Storage1D<std::pair<K,T>,ST>& target, const std::unordered_map<K,T,H>& source) noexcept
{
  //TODO: think about std::copy()
  target.resize_dirty(source.size());
  uint k=0;
  for (typename std::unordered_map<K,T,H>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target[k] = *it;
    k++;
  }
}

template<typename T, typename ST, typename K>
void assign(std::vector<std::pair<K,T>,ST>& target, const std::map<K,T>& source) noexcept
{
  target.clear();
  target.reserve(source.size());

  for (typename std::map<K,T>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target.push_back(*it);
  }
}

template<typename T, typename ST, typename K, typename H>
void assign(std::vector<std::pair<K,T>,ST>& target, const std::unordered_map<K,T,H>& source) noexcept
{
  target.clear();
  target.reserve(source.size());

  for (typename std::unordered_map<K,T,H>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target.push_back(*it);
  }
}

template<typename T, typename ST>
void assign(Storage1D<T,ST>& target, const std::set<T>& source) noexcept
{
  target.resize_dirty(source.size());
  uint k=0;
  for (typename std::set<T>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target[k] = *it;
    k++;
  }
}

template<typename T, typename ST, typename K>
void assign(Storage1D<K,ST>& target1, Storage1D<T,ST>& target2, const std::map<K,T>& source) noexcept
{
  target1.resize_dirty(source.size());
  target2.resize_dirty(source.size());

  uint k=0;
  for (typename std::map<K,T>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target1[k] = it->first;
    target2[k] = it->second;
    k++;
  }
}

template<typename T, typename ST, typename K, typename H>
void assign(Storage1D<K,ST>& target1, Storage1D<T,ST>& target2, const std::unordered_map<K,T,H>& source) noexcept
{
  target1.resize_dirty(source.size());
  target2.resize_dirty(source.size());

  uint k=0;
  for (typename std::unordered_map<K,T,H>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target1[k] = it->first;
    target2[k] = it->second;
    k++;
  }
}

template<typename T, typename K>
void assign(std::vector<K>& target1, std::vector<T>& target2, const std::map<K,T>& source) noexcept
{
  target1.clear();
  target1.reserve(source.size());
  target2.clear();
  target2.reserve(source.size());

  for (typename std::map<K,T>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target1.push_back(it->first);
    target2.push_back(it->second);
  }
}

template<typename T, typename K, typename H>
void assign(std::vector<K>& target1, std::vector<T>& target2, const std::unordered_map<K,T,H>& source) noexcept
{
  target1.clear();
  target1.reserve(source.size());
  target2.clear();
  target2.reserve(source.size());

  for (typename std::unordered_map<K,T,H>::const_iterator it = source.cbegin(); it != source.cend(); ++it) {
    target1.push_back(it->first);
    target2.push_back(it->second);
  }
}

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const Storage1D<T2,ST>& source) noexcept
{
  //TODO: think about std::copy()

  target.clear();
  target.reserve(source.size());

  for (uint k=0; k < source.size(); k++)
    target.push_back(source[k]);
}

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const FlexibleStorage1D<T2,ST>& source) noexcept
{
  target.clear();
  target.reserve(source.size());

  for (uint k=0; k < source.size(); k++)
    target.push_back(source[k]);
}

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const Storage1D<T2,ST>& source) noexcept
{
  target.resize_dirty(source.size());

  //at least for g++ std::copy is faster
  std::copy(source.direct_access(),source.direct_access()+source.size(),target.direct_access());

  //for (uint k=0; k < source.size(); k++)
  //  target[k] = (T1) source[k];
}

template<typename T1, typename T2, typename ST>
void assign(Storage2D<T1,ST>& target, const Storage2D<T2,ST>& source) noexcept
{
  target.resize_dirty(source.xDim(),source.yDim());

  //at least for g++ std::copy is faster
  std::copy(source.direct_access(),source.direct_access()+source.size(),target.direct_access());

  //for (uint k=0; k < source.size(); k++)
  //  target.direct_access(k) = (T1) source.direct_access(k);
}

template<typename T1, typename T2, typename ST>
void assign(Storage3D<T1,ST>& target, const Storage3D<T2,ST>& source) noexcept
{
  target.resize_dirty(source.xDim(),source.yDim(),source.zDim());

  //at least for g++ std::copy is faster
  std::copy(source.direct_access(),source.direct_access()+source.size(),target.direct_access());

  //for (uint k=0; k < source.size(); k++)
  //  target.direct_access(k) = (T1) source.direct_access(k);
}

#endif
