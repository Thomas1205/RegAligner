/*** written by Thomas Schoenemann as an employee of Lund University, August 2010 ***/

#include "storage2D.hh"
#include <cstring>

#if 0 //obsolete

template<>
Storage2D<int>::Storage2D(const Storage2D<int>& toCopy) 
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  size_ = toCopy.size();

  assert(size_ == xDim_*yDim_);

  if (size_ == 0)
    data_ = 0;
  else {
    data_ = new int[size_];  
    
    memcpy(data_, toCopy.direct_access(), size_ * sizeof(int));
  }
}

template<>
Storage2D<uint>::Storage2D(const Storage2D<uint>& toCopy) 
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  size_ = toCopy.size();

  assert(size_ == xDim_*yDim_);

  if (size_ == 0)
    data_ = 0;
  else {
    data_ = new uint[size_];  
    
    memcpy(data_, toCopy.direct_access(), size_ * sizeof(uint));
  }  
}

template<>
Storage2D<float>::Storage2D(const Storage2D<float>& toCopy) 
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  size_ = toCopy.size();

  assert(size_ == xDim_*yDim_);

  if (size_ == 0)
    data_ = 0;
  else {
    data_ = new float[size_];  
    
    memcpy(data_, toCopy.direct_access(), size_ * sizeof(float));
  }
}

template<>
Storage2D<double>::Storage2D(const Storage2D<double>& toCopy) 
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  size_ = toCopy.size();

  assert(size_ == xDim_*yDim_);

  if (size_ == 0)
    data_ = 0;
  else {
    data_ = new double[size_];  
    
    memcpy(data_, toCopy.direct_access(), size_ * sizeof(double));
  }
}

template<>
Storage2D<long double>::Storage2D(const Storage2D<long double>& toCopy) 
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  size_ = toCopy.size();

  assert(size_ == xDim_*yDim_);

  if (size_ == 0)
    data_ = 0;
  else {
    data_ = new long double[size_];  
    
    memcpy(data_, toCopy.direct_access(), size_ * sizeof(long double));
  }
}

template<>
void Storage2D<int>::operator=(const Storage2D<int>& toCopy)
{
  if (size_ != toCopy.size()) {
    if (data_ != 0)
      delete[] data_;

    size_ = toCopy.size();
    data_ = new int[size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  assert(size_ == xDim_*yDim_);

  memcpy(data_,toCopy.direct_access(),size_*sizeof(int));
}

template<>
void Storage2D<uint>::operator=(const Storage2D<uint>& toCopy)
{
  if (size_ != toCopy.size()) {
    if (data_ != 0)
      delete[] data_;

    size_ = toCopy.size();
    data_ = new uint[size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  assert(size_ == xDim_*yDim_);

  memcpy(data_,toCopy.direct_access(),size_*sizeof(uint));
}

template<>
void Storage2D<float>::operator=(const Storage2D<float>& toCopy)
{
  if (size_ != toCopy.size()) {
    if (data_ != 0)
      delete[] data_;

    size_ = toCopy.size();
    data_ = new float[size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  assert(size_ == xDim_*yDim_);

  memcpy(data_,toCopy.direct_access(),size_*sizeof(float));
}

template<>
void Storage2D<double>::operator=(const Storage2D<double>& toCopy)
{
  if (size_ != toCopy.size()) {
    if (data_ != 0)
      delete[] data_;

    size_ = toCopy.size();
    data_ = new double[size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  assert(size_ == xDim_*yDim_);

  memcpy(data_,toCopy.direct_access(),size_*sizeof(double));
}

template<>
void Storage2D<long double>::operator=(const Storage2D<long double>& toCopy)
{
  if (size_ != toCopy.size()) {
    if (data_ != 0)
      delete[] data_;

    size_ = toCopy.size();
    data_ = new long double[size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  assert(size_ == xDim_*yDim_);

  memcpy(data_,toCopy.direct_access(),size_*sizeof(long double));
}

template<>
void Storage2D<int>::set_row(size_t y, const Storage1D<int>& row_vec) 
{ 
  assert(y < yDim_);
  assert(row_vec.size() == xDim_);

  int* data = StorageBase<T,ST>::data_ + y * xDim_;
  memcpy(data, row_vec.direct_access(), xDim_*sizeof(int));
}

template<>
void Storage2D<uint>::set_row(size_t y, const Storage1D<uint>& row_vec) 
{ 
  assert(y < yDim_);
  assert(row_vec.size() == xDim_);

  uint* data = StorageBase<T,ST>::data_ + y * xDim_;
  memcpy(data, row_vec.direct_access(), xDim_*sizeof(uint));
}

template<>
void Storage2D<float>::set_row(size_t y, const Storage1D<float>& row_vec) 
{
  assert(y < yDim_); 
  assert(row_vec.size() == xDim_);

  float* data = StorageBase<T,ST>::data_ + y * xDim_;
  memcpy(data, row_vec.direct_access(), xDim_*sizeof(float));
}

template<>
void Storage2D<double>::set_row(size_t y, const Storage1D<double>& row_vec) 
{
  assert(y < yDim_);  
  assert(row_vec.size() == xDim_);

  double* data = StorageBase<T,ST>::data_ + y * xDim_;
  memcpy(data, row_vec.direct_access(), xDim_*sizeof(double));
}

template<>
void Storage2D<long double>::set_row(size_t y, const Storage1D<long double>& row_vec) 
{
  assert(y < yDim_);  
  assert(row_vec.size() == xDim_);

  long double* data = StorageBase<T,ST>::data_ + y * xDim_;
  memcpy(data, row_vec.direct_access(), xDim_*sizeof(long double));
}

#endif