/**** written by Thomas Schoenemann as a private person without employment, April 2013 ****/

#ifndef STORAGE_STL_INTERFACE
#define STORAGE_STL_INTERFACE

#include <vector>
#include <map>
#include <set>
#include "storage1D.hh"
#include "storage2D.hh"
#include "storage3D.hh"

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const std::vector<T2>& source);

template<typename T1, typename T2, typename ST>
void assign(FlexibleStorage1D<T1,ST>& target, const std::vector<T2>& source);

template<typename T, typename ST, typename X>
void assign(Storage1D<std::pair<X,T>,ST>& target, const std::map<X,T>& source);

template<typename T, typename ST>
void assign(Storage1D<T,ST>& target, const std::set<T>& source);

template<typename T, typename ST, typename X>
void assign(Storage1D<X,ST>& target1, Storage1D<T,ST>& target2, const std::map<X,T>& source);

template<typename T, typename X>
void assign(std::vector<X>& target1, std::vector<T>& target2, const std::map<X,T>& source);

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const Storage1D<T2,ST>& source);

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const FlexibleStorage1D<T2,ST>& source);

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const Storage1D<T2,ST>& source);

template<typename T1, typename T2, typename ST>
void assign(Storage2D<T1,ST>& target, const Storage2D<T2,ST>& source);

template<typename T1, typename T2, typename ST>
void assign(Storage3D<T1,ST>& target, const Storage3D<T2,ST>& source);


/************** implementation *************/

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const std::vector<T2>& source)
{
  //std::copy() is sometimes faster, but not consistently for different vector sizes (with g++)

  target.resize_dirty(source.size());
  for (uint k=0; k < source.size(); k++)
    target[k] = source[k];
}

template<typename T1, typename T2, typename ST>
void assign(FlexibleStorage1D<T1,ST>& target, const std::vector<T2>& source) 
{
  target.clear();
  if (target.reserved_size() < source.size())
    target.reserve(source.size());
  
  for (size_t i=0; i < source.size(); i++)
    target.append(source[i]);
}

template<typename T1, typename T2, typename ST>
void assign_copy(Storage1D<T1,ST>& target, const std::vector<T2>& source)
{
  target.resize_dirty(source.size());
  std::copy(source.begin(),source.end(),target.direct_access());
}


template<typename T, typename ST, typename X>
void assign(Storage1D<std::pair<X,T>,ST>& target, const std::map<X,T>& source)
{
  //TODO: think about std::copy()
  target.resize_dirty(source.size());
  uint k=0;
  for (typename std::map<X,T>::const_iterator it = source.begin(); it != source.end(); it++) {
    target[k] = *it;
    k++;
  }
}

template<typename T, typename ST>
void assign(Storage1D<T,ST>& target, const std::set<T>& source)
{
  target.resize_dirty(source.size());
  uint k=0;
  for (typename std::set<T>::const_iterator it = source.begin(); it != source.end(); it++) {
    target[k] = *it;
    k++;
  }
}

template<typename T, typename ST, typename X>
void assign(Storage1D<X,ST>& target1, Storage1D<T,ST>& target2, const std::map<X,T>& source)
{
  target1.resize_dirty(source.size());
  target2.resize_dirty(source.size());

  uint k=0;
  for (typename std::map<X,T>::const_iterator it = source.begin(); it != source.end(); it++) {
    target1[k] = it->first;
    target2[k] = it->second;
    k++;
  }
}

template<typename T, typename X>
void assign(std::vector<X>& target1, std::vector<T>& target2, const std::map<X,T>& source)
{
  target1.clear();
  target1.reserve(source.size());
  target2.clear();
  target2.reserve(source.size());

  for (typename std::map<X,T>::const_iterator it = source.begin(); it != source.end(); it++) {
    target1.push_back(it->first);
    target2.push_back(it->second);
  }
}

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const Storage1D<T2,ST>& source)
{
  //TODO: think about std::copy()

  target.clear();
  target.reserve(source.size());

  for (uint k=0; k < source.size(); k++)
    target.push_back(source[k]);
}

template<typename T1, typename T2, typename ST>
void assign(std::vector<T1>& target, const FlexibleStorage1D<T2,ST>& source)
{
  target.clear();
  target.reserve(source.size());

  for (uint k=0; k < source.size(); k++)
    target.push_back(source[k]);
}

template<typename T1, typename T2, typename ST>
void assign(Storage1D<T1,ST>& target, const Storage1D<T2,ST>& source)
{
  target.resize_dirty(source.size());

  //at least for g++ std::copy is faster
  std::copy(source.direct_access(),source.direct_access()+source.size(),target.direct_access());

  //for (uint k=0; k < source.size(); k++)
  //  target[k] = (T1) source[k];
}

template<typename T1, typename T2, typename ST>
void assign(Storage2D<T1,ST>& target, const Storage2D<T2,ST>& source)
{
  target.resize_dirty(source.xDim(),source.yDim());

  //at least for g++ std::copy is faster
  std::copy(source.direct_access(),source.direct_access()+source.size(),target.direct_access());

  //for (uint k=0; k < source.size(); k++)
  //  target.direct_access(k) = (T1) source.direct_access(k);
}

template<typename T1, typename T2, typename ST>
void assign(Storage3D<T1,ST>& target, const Storage3D<T2,ST>& source)
{
  target.resize_dirty(source.xDim(),source.yDim(),source.zDim());

  //at least for g++ std::copy is faster
  std::copy(source.direct_access(),source.direct_access()+source.size(),target.direct_access());

  //for (uint k=0; k < source.size(); k++)
  //  target.direct_access(k) = (T1) source.direct_access(k);
}

#endif
