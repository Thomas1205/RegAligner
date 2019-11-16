/*** written by Thomas Schoenemann as a private person without employment, December 2012 ****/


#ifndef NESTED_STORAGE_1D
#define NESTED_STORAGE_1D

#include "storage1D.hh"
#include <vector>

template<typename T, typename ST>
class NestedStorage1D {
public:

  NestedStorage1D();

  NestedStorage1D(const Storage1D<Storage1D<T> >& vec_of_vecs);

  NestedStorage1D(const std::vector<std::vector<T> >& vec_of_vecs);

  ~NestedStorage1D();

  const T* operator[](ST k) const;

  T* operator[](ST k);

  T& operator()(uint k, uint l);

  void operator=(const Storage1D<Storage1D<T> >& vec_of_vecs);

  void operator=(const std::vector<std::vector<T> >& vec_of_vecs);

  //careful: this is only useful for READ access, changes made to the returned object will not affect this class
  Storage1D<T> get(uint k) const;

  const T& operator()(uint k, uint l) const;

  ST size() const;

  ST dim(uint k) const;

protected:

  T* data_;

  Storage1D<ST> start_;
};

/**************** implemtentation ************/

template<typename T, typename ST> NestedStorage1D<T,ST>::NestedStorage1D() : start_(1,0)
{
  data_ = new T[0];
}

template<typename T, typename ST> NestedStorage1D<T,ST>::NestedStorage1D(const Storage1D<Storage1D<T> >& vec_of_vecs) : start_(vec_of_vecs.size()+1)
{

  ST size = 0;
  for (uint k=0; k < vec_of_vecs.size(); k++) {
    start_[k] = size;
    size += vec_of_vecs[k].size();
  }
  start_[vec_of_vecs.size()] = size;


  data_ = new T[size];

  for (uint k=0; k < vec_of_vecs.size(); k++) {
    for (uint l=0; l < vec_of_vecs[k].size(); l++)
      data_[start_[k]+l] = vec_of_vecs[k][l];
  }
}


template<typename T, typename ST> NestedStorage1D<T,ST>::NestedStorage1D(const std::vector<std::vector<T> >& vec_of_vecs) : start_(vec_of_vecs.size()+1)
{

  ST size = 0;
  for (uint k=0; k < vec_of_vecs.size(); k++) {
    start_[k] = size;
    size += vec_of_vecs[k].size();
  }
  start_[vec_of_vecs.size()] = size;

  data_ = new T[size];

  for (uint k=0; k < vec_of_vecs.size(); k++) {
    for (uint l=0; l < vec_of_vecs[k].size(); l++)
      data_[start_[k]+l] = vec_of_vecs[k][l];
  }
}

template<typename T, typename ST> NestedStorage1D<T,ST>::~NestedStorage1D()
{
  delete[] data_;
}


template<typename T, typename ST>
void NestedStorage1D<T,ST>::operator=(const Storage1D<Storage1D<T> >& vec_of_vecs)
{

  delete[] data_;
  start_.resize(vec_of_vecs.size());

  ST size = 0;
  for (uint k=0; k < vec_of_vecs.size(); k++) {
    start_[k] = size;
    size += vec_of_vecs[k].size();
  }
  start_[vec_of_vecs.size()] = size;

  data_ = new T[size];

  for (uint k=0; k < vec_of_vecs.size(); k++) {
    for (uint l=0; l < vec_of_vecs[k].size(); l++)
      data_[start_[k]+l] = vec_of_vecs[k][l];
  }
}

template<typename T, typename ST>
void NestedStorage1D<T,ST>::operator=(const std::vector<std::vector<T> >& vec_of_vecs)
{

  delete[] data_;
  start_.resize(vec_of_vecs.size());

  ST size = 0;
  for (uint k=0; k < vec_of_vecs.size(); k++) {
    start_[k] = size;
    size += vec_of_vecs[k].size();
  }
  start_[vec_of_vecs.size()] = size;

  data_ = new T[size];

  for (uint k=0; k < vec_of_vecs.size(); k++) {
    for (uint l=0; l < vec_of_vecs[k].size(); l++)
      data_[start_[k]+l] = vec_of_vecs[k][l];
  }
}

template<typename T, typename ST>
Storage1D<T> NestedStorage1D<T,ST>::get(uint k) const
{
#ifdef SAFE_MODE
  if (k >= start_.size()-1) {
    INTERNAL_ERROR << "    invalid access on element " << k
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }
#endif

  Storage1D<T> result(dim(k));
  for (uint l=start_[k]; l < start_[k+1]; l++)
    result[l-start_[k] = data_[l]];
}

template<typename T, typename ST>
const T* NestedStorage1D<T,ST>::operator[](ST k) const
{

#ifdef SAFE_MODE
  if (k >= start_.size()-1) {
    INTERNAL_ERROR << "    invalid access on element " << k
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }
#endif

  return data_+start_[k];
}


template<typename T, typename ST>
T* NestedStorage1D<T,ST>::operator[](ST k)
{

#ifdef SAFE_MODE
  if (k >= start_.size()-1) {
    INTERNAL_ERROR << "    invalid access on element " << k
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }
#endif
}


template<typename T, typename ST>
T& NestedStorage1D<T,ST>::operator()(uint k, uint l)
{

#ifdef SAFE_MODE
  if (k >= start_.size()-1) {
    INTERNAL_ERROR << "    invalid access on element " << k
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }

  if (start_[k]+l >= start_[k+1]) {
    INTERNAL_ERROR << "    invalid access on element " << k << "," << l
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }
#endif

  return data_[start_[k]+l];
}

template<typename T, typename ST>
const T& NestedStorage1D<T,ST>::operator()(uint k, uint l) const
{

#ifdef SAFE_MODE
  if (k >= start_.size()-1) {
    INTERNAL_ERROR << "    invalid access on element " << k
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }

  if (start_[k]+l >= start_[k+1]) {
    INTERNAL_ERROR << "    invalid access on element " << k << "," << l
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }
#endif

  return data_[start_[k]+l];
}


template<typename T, typename ST>
ST NestedStorage1D<T,ST>::size() const
{

  return start_.size()-1;
}

template<typename T, typename ST>
ST NestedStorage1D<T,ST>::dim(uint k) const
{
#ifdef SAFE_MODE
  if (k >= start_.size()-1) {
    INTERNAL_ERROR << "    invalid access on element " << k
                   << " for NestedStorage1D of type " << typeid(T).name()
                   << " with " << (start_.size_()-1) << " elements. exiting." << std::endl;
    exit(1);
  }
#endif

  return start_[k+1] - start_[k];
}

#endif
