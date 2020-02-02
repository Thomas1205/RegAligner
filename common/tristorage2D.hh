/*-*-c++-*-*/
/*** written by Thomas Schoenemann as a private person without employment, March 2015 ***/
/*** if you desire the checked version, make sure your compiler defines the option SAFE_MODE on the command line ***/

#ifndef TRISTORAGE2D_HH
#define TRISTORAGE2D_HH

#include "makros.hh"
#include "storage_base.hh"

//two-dimensional container class for objects of any type T, where only have the pattern can be filled/defined
//you can use this either for a triangular pattern or for a symmetric one
//(as long as we use only the storage it does not matter much, except when printing. for matrix usage it does matter, e.g. in the matrix-vector product)
//(neither mathematical nor streaming operations need to be defined on T)
template<typename T, typename ST = size_t>
class TriStorage2D : public StorageBase<T,ST> {
public:

  typedef StorageBase<T,ST> Base;

  //default constructor
  TriStorage2D();

  TriStorage2D(ST dim);

  TriStorage2D(ST dim, T default_value);

  //copy constructor
  TriStorage2D(const TriStorage2D<T,ST>& toCopy);

  ~TriStorage2D();

  virtual const std::string& name() const;

  //existing elements are preserved, new ones are uninitialized
  void resize(ST newDim);

  //existing elements are preserved, new ones are filled with the second argument
  void resize(ST newDim, const T fill_value);

  //all elements are uninitialized after this operation
  void resize_dirty(ST newDim);

  //access on an element (handling is symmetric, i.e. accessing (x,y) is equivalent to accessing (y,x) )
  inline const T& operator()(ST x, ST y) const;

  inline T& operator()(ST x, ST y);

  void operator=(const TriStorage2D<T,ST>& toCopy);

  inline T* row_ptr(ST y);
  
  inline const T* row_ptr(ST y) const;

  inline ST dim() const;

protected:

  ST dim_;
  static const std::string tristor2D_name_;
};


template<typename T, typename ST=size_t>
class NamedTriStorage2D : public TriStorage2D<T,ST> {
public:

  NamedTriStorage2D();

  NamedTriStorage2D(std::string name);

  NamedTriStorage2D(ST dim, std::string name);

  NamedTriStorage2D(ST dim, T default_value, std::string name);

  virtual const std::string& name() const;

  inline void operator=(const TriStorage2D<T,ST>& toCopy);

  //NOTE: the name is NOT copied
  inline void operator=(const NamedTriStorage2D<T,ST>& toCopy);

protected:
  std::string name_;
};


template<typename T, typename ST>
bool operator==(const TriStorage2D<T,ST>& v1, const TriStorage2D<T,ST>& v2);

template<typename T, typename ST>
bool operator!=(const TriStorage2D<T,ST>& v1, const TriStorage2D<T,ST>& v2);

namespace Makros {

  template<typename T, typename ST>
  class Typename<TriStorage2D<T,ST> > {
  public:

    std::string name() const
    {
      return "TriStorage2D<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<TriStorage2D<T> > {
  public:

    std::string name() const
    {
      return "TriStorage2D<" + Typename<T>() + "> ";
    }
  };

  template<typename T, typename ST>
  class Typename<NamedTriStorage2D<T,ST> > {
  public:

    std::string name() const
    {
      return "NamedTriStorage2D<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<NamedTriStorage2D<T> > {
  public:

    std::string name() const
    {
      return "NamedTriStorage2D<" + Typename<T>() + "> ";
    }
  };

}


/**************************** implementation **************************************/

template<typename T, typename ST>
/*static*/ const std::string TriStorage2D<T,ST>::tristor2D_name_ = "unnamed TriStorage2D";

//constructors
template<typename T, typename ST> 
TriStorage2D<T,ST>::TriStorage2D() : StorageBase<T,ST>(), dim_(0) {}

template<typename T, typename ST> 
TriStorage2D<T,ST>::TriStorage2D(ST dim) : StorageBase<T,ST>(dim*(dim+1) / 2), dim_(dim)
{
}

template<typename T, typename ST> 
TriStorage2D<T,ST>::TriStorage2D(ST dim, T default_value) : StorageBase<T,ST>(dim*(dim+1) / 2, default_value), dim_(dim)
{
}

//copy constructor
template<typename T, typename ST> 
TriStorage2D<T,ST>::TriStorage2D(const TriStorage2D<T,ST>& toCopy)
{
  dim_ = toCopy.dim();
  StorageBase<T,ST>::size_ = toCopy.size();

  if (StorageBase<T,ST>::size_ == 0)
    StorageBase<T,ST>::data_ = 0;
  else {
    StorageBase<T,ST>::data_ = new T[StorageBase<T,ST>::size_];

    Makros::unified_assign(StorageBase<T,ST>::data_, toCopy.direct_access(), StorageBase<T,ST>::size_);

    //for (ST i = 0; i < size_; i++)
    //  data_[i] = toCopy.direct_access(i);
  }
}

//destructor
template <typename T, typename ST> 
TriStorage2D<T,ST>::~TriStorage2D()
{
}

template<typename T, typename ST>
void TriStorage2D<T,ST>::operator=(const TriStorage2D<T,ST>& toCopy)
{
  dim_ = toCopy.dim();
  StorageBase<T,ST>::size_ = toCopy.size();

  //room for improvement here: could check if the dimensions already match, then we can reuse data_

  if (StorageBase<T,ST>::data_ != 0)
    delete[] StorageBase<T,ST>::data_;

  if (StorageBase<T,ST>::size_ == 0)
    StorageBase<T,ST>::data_ = 0;
  else {
    StorageBase<T,ST>::data_ = new T[StorageBase<T,ST>::size_];

    Makros::unified_assign(StorageBase<T,ST>::data_, toCopy.direct_access(), StorageBase<T,ST>::size_);

    //for (ST i = 0; i < size_; i++)
    //  data_[i] = toCopy.direct_access(i);
  }
}

template <typename T, typename ST>
/*virtual*/ const std::string& TriStorage2D<T,ST>::name() const
{
  return tristor2D_name_;
}

template<typename T, typename ST>
inline T* TriStorage2D<T,ST>::row_ptr(ST y) 
{
  assert(y < dim_);
  return StorageBase<T,ST>::data_ + (y*(y+1))/2;  
}
  
template<typename T, typename ST>
inline const T* TriStorage2D<T,ST>::row_ptr(ST y) const 
{
  assert(y < dim_);
  return StorageBase<T,ST>::data_ + (y*(y+1))/2;
}

template<typename T, typename ST>
inline ST TriStorage2D<T,ST>::dim() const
{
  return dim_;
}

template <typename T, typename ST>
inline const T& TriStorage2D<T,ST>::operator()(ST x, ST y) const
{
#ifdef SAFE_MODE
  if (x >= dim_ || y >= dim_) {
    INTERNAL_ERROR << "    const access on element(" << x << "," << y
                   << ") exceeds storage dimensions of (" << dim_ << "," << dim_ << ")" << std::endl;
    std::cerr << "      in TriStorage2D \"" << this->name() << "\" of type "
              << Makros::Typename<T>()
              << ". Exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  if (x > y)
    std::swap(x,y);

  return StorageBase<T,ST>::data_[(y*(y+1))/2+x];
}

template <typename T, typename ST>
inline T& TriStorage2D<T,ST>::operator()(ST x, ST y)
{
#ifdef SAFE_MODE
  if (x >= dim_ || y >= dim_) {
    INTERNAL_ERROR << "    access on element(" << x << "," << y
                   << ") exceeds storage dimensions of (" << dim_ << "," << dim_ << ")" << std::endl;
    std::cerr << "      in TriStorage2D \"" << this->name() << "\" of type "
              << Makros::Typename<T>()
              << ". Exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif

  //std::cerr << " access(" << x << "," << y << ")"; // << std::endl;

  if (x > y)
    std::swap(x,y);

  return StorageBase<T,ST>::data_[(y*(y+1))/2+x];
}

//existing elements are preserved, new ones are uninitialized
template<typename T, typename ST>
void TriStorage2D<T,ST>::resize(ST newDim)
{
  if (newDim != dim_) {

    ST new_size = newDim*(newDim+1) / 2;
    T* new_data = new T[new_size];

    Makros::unified_assign(new_data, StorageBase<T,ST>::data_, std::min(StorageBase<T,ST>::size_,new_size));

    //copy existing values
    //for (ST i=0; i < std::min(size_,new_size); i++)
    //  new_data[i] = data_[i];

    if (StorageBase<T,ST>::data_ != 0)
      delete[] StorageBase<T,ST>::data_;

    StorageBase<T,ST>::data_ = new_data;
    StorageBase<T,ST>::size_ = new_size;
    dim_ = newDim;
  }
}

//existing elements are preserved, new ones are filled with the second argument
template<typename T, typename ST>
void TriStorage2D<T,ST>::resize(ST newDim, const T fill_value)
{
  if (newDim != dim_) {

    ST new_size = newDim*(newDim+1) / 2;
    T* new_data = new T[new_size];

    Makros::unified_assign(new_data, StorageBase<T,ST>::data_, std::min(StorageBase<T,ST>::size_,new_size));

    //copy existing values
    //for (ST i=0; i < std::min(size_,new_size); i++)
    //  new_data[i] = data_[i];

    //fill new values
    if (new_size > StorageBase<T,ST>::size_)
      std::fill_n(new_data+StorageBase<T,ST>::size_,new_size-StorageBase<T,ST>::size_,fill_value);
    
    // for (ST i=std::min(size_,new_size); i < new_size; i++)
    //   new_data[i] = fill_value;

    if (StorageBase<T,ST>::data_ != 0)
      delete[] StorageBase<T,ST>::data_;

    StorageBase<T,ST>::data_ = new_data;
    StorageBase<T,ST>::size_ = new_size;
    dim_ = newDim;
  }
}

template<typename T, typename ST>
void TriStorage2D<T,ST>::resize_dirty(ST newDim)
{
  if (newDim != dim_) {
    if (StorageBase<T,ST>::data_ != 0) {
      delete[] StorageBase<T,ST>::data_;
    }

    dim_ = newDim;
    StorageBase<T,ST>::size_ = dim_*(dim_+1) / 2;
    StorageBase<T,ST>::data_ = new T[StorageBase<T,ST>::size_];
  }
}


template<typename T, typename ST>
bool operator==(const TriStorage2D<T,ST>& v1, const TriStorage2D<T,ST>& v2)
{
  if (v1.size() != v2.size())
    return false;

  uint i;
  for (i=0; i < v1.size(); i++) {
    if (v1.direct_access(i) != v2.direct_access(i))
      return false;
  }

  return true;
}

template<typename T, typename ST>
bool operator!=(const TriStorage2D<T,ST>& v1, const TriStorage2D<T,ST>& v2)
{
  if (v1.size() != v2.size())
    return true;

  uint i;
  for (i=0; i < v1.size(); i++) {
    if (v1.direct_access(i) != v2.direct_access(i))
      return true;
  }

  return false;
}

/***** implementation of NamedTriStorage2D ********/

template<typename T, typename ST> 
NamedTriStorage2D<T,ST>::NamedTriStorage2D() {}

template<typename T, typename ST> 
NamedTriStorage2D<T,ST>::NamedTriStorage2D(std::string name) : name_(name) {}

template<typename T, typename ST> 
NamedTriStorage2D<T,ST>::NamedTriStorage2D(ST dim, std::string name) : TriStorage2D<T,ST>(dim), name_(name) {}

template<typename T, typename ST> 
NamedTriStorage2D<T,ST>::NamedTriStorage2D(ST dim, T default_value, std::string name) : TriStorage2D<T,ST>(dim,default_value), name_(name) {}

template<typename T, typename ST>
/*virtual*/ const std::string& NamedTriStorage2D<T,ST>::name() const
{
  return name_;
}

template<typename T, typename ST>
inline void NamedTriStorage2D<T,ST>::operator=(const TriStorage2D<T,ST>& toCopy)
{
  TriStorage2D<T,ST>::operator=(toCopy);
}

//NOTE: the name is NOT copied
template<typename T, typename ST>
inline void NamedTriStorage2D<T,ST>::operator=(const NamedTriStorage2D<T,ST>& toCopy)
{
  TriStorage2D<T,ST>::operator=(toCopy);
}

#endif
