/*-*-c++-*-*/
/*** first version written by Thomas Schoenemann as a private person without employment, September 2009 ***/
/*** much refined by Thomas Schoenemann  at Lund University, Sweden, the University of Pisa, Italy, ***
 *** and the University of Düsseldorf, Germany 2010 - 2012 **/
/*** if you desire the checked version, make sure your compiler defines the option SAFE_MODE on the command line ***/

#ifndef STORAGE2D_HH
#define STORAGE2D_HH

#include "storage1D.hh"

//two-dimensional container class for objects of any type T
//(i.e. neither mathematical nor streaming operations need to be defined on T)
template<typename T, typename ST = size_t>
class Storage2D : public StorageBase<T,ST> {
public:

  //default constructor
  Storage2D();

  Storage2D(ST xDim, ST yDim);

  Storage2D(ST xDim, ST yDim, T default_value);

  Storage2D(const std::pair<ST,ST> dims);

  Storage2D(const std::pair<ST,ST> dims, T default_value);

  //copy constructor
  Storage2D(const Storage2D<T,ST>& toCopy);

  ~Storage2D();

  virtual const std::string& name() const;

  //saves all existing entries, new positions contain undefined data
  void resize(ST newxDim, ST newyDim);

  //saves all existing entries, new positions are filled with <code> fill_value </code>
  void resize(ST newxDim, ST newyDim, const T fill_value);

  //all elements are uninitialized after this operation
  void resize_dirty(ST newxDim, ST newyDim);

  //access on an element
  inline const T& operator()(ST x, ST y) const;

  inline T& operator()(ST x, ST y);

  void operator=(const Storage2D<T,ST>& toCopy);

#ifdef SAFE_MODE
  //for some reason g++ allows to assign an object of type T, but this does NOT produce the effect one would expect
  // => define this operator in safe mode, only to check that such an assignment is not made
  void operator=(const T& invalid_object);
#endif
  
  inline T* row_ptr(ST y);
  
  inline T* row_ptr(ST y) const;  

  inline T value(ST i) const;

  void set_row(ST y, const Storage1D<T,ST>& row_vec);

  inline ST xDim() const;

  inline ST yDim() const;

  inline std::pair<ST,ST> dims() const;

protected:

  ST xDim_;
  ST yDim_;
  static const std::string stor2D_name_;
};

template<typename T, typename ST=size_t>
class NamedStorage2D : public Storage2D<T,ST> {
public:

  NamedStorage2D();

  NamedStorage2D(std::string name);

  NamedStorage2D(ST xDim, ST yDim, std::string name);

  NamedStorage2D(ST xDim, ST yDim, T default_value, std::string name);

  NamedStorage2D(const std::pair<ST,ST> dims, std::string name);

  NamedStorage2D(const std::pair<ST,ST> dims, T default_value, std::string name);

  virtual const std::string& name() const;

  inline void operator=(const Storage2D<T,ST>& toCopy);

  //NOTE: the name is NOT copied
  inline void operator=(const NamedStorage2D<T,ST>& toCopy);

protected:
  std::string name_;
};


template<typename T, typename ST>
bool operator==(const Storage2D<T,ST>& v1, const Storage2D<T,ST>& v2);

template<typename T, typename ST>
bool operator!=(const Storage2D<T,ST>& v1, const Storage2D<T,ST>& v2);


namespace Makros {

  template<typename T, typename ST>
  class Typename<Storage2D<T,ST> > {
  public:

    std::string name() const
    {
      return "Storage2D<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Storage2D<T> > {
  public:

    std::string name() const
    {
      return "Storage2D<" + Typename<T>() + "> ";
    }
  };


  template<typename T, typename ST>
  class Typename<NamedStorage2D<T,ST> > {
  public:

    std::string name() const
    {
      return "NamedStorage2D<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<NamedStorage2D<T> > {
  public:

    std::string name() const
    {
      return "NamedStorage2D<" + Typename<T>() + "> ";
    }
  };

}

/**************************** implementation **************************************/

template<typename T, typename ST>
/*static*/ const std::string Storage2D<T,ST>::stor2D_name_ = "unnamed 2Dstorage";

//constructors
template<typename T, typename ST> 
Storage2D<T,ST>::Storage2D() : StorageBase<T,ST>(), xDim_(0), yDim_(0) {}

template<typename T, typename ST> 
Storage2D<T,ST>::Storage2D(ST xDim, ST yDim) : StorageBase<T,ST>(xDim*yDim), xDim_(xDim), yDim_(yDim)
{
}

template<typename T, typename ST> 
Storage2D<T,ST>::Storage2D(ST xDim, ST yDim, const T default_value) 
  : StorageBase<T,ST>(xDim*yDim, default_value), xDim_(xDim), yDim_(yDim)
{
}

template<typename T, typename ST> 
Storage2D<T,ST>::Storage2D(const std::pair<ST,ST> dims) 
  : StorageBase<T,ST>(dims.first*dims.second), xDim_(dims.first), yDim_(dims.second)
{
}

template<typename T, typename ST> 
Storage2D<T,ST>::Storage2D(const std::pair<ST,ST> dims, T default_value) 
  : StorageBase<T,ST>(dims.first*dims.second, default_value), xDim_(dims.first), yDim_(dims.second)
{
}

//copy constructor
template<typename T, typename ST> 
Storage2D<T,ST>::Storage2D(const Storage2D<T,ST>& toCopy) : StorageBase<T,ST>(toCopy.xDim()*toCopy.yDim())
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();

  const ST size = StorageBase<T,ST>::size_;
  assert(size == xDim_*yDim_);

  if (size > 0) 
    Makros::unified_assign(StorageBase<T,ST>::data_, toCopy.direct_access(), size);
}

// template<>
// Storage2D<int>::Storage2D(const Storage2D<int>& toCopy);

// template<>
// Storage2D<uint>::Storage2D(const Storage2D<uint>& toCopy);

// template<>
// Storage2D<float>::Storage2D(const Storage2D<float>& toCopy);

// template<>
// Storage2D<double>::Storage2D(const Storage2D<double>& toCopy);

// template<>
// Storage2D<long double>::Storage2D(const Storage2D<long double>& toCopy);


//destructor
template <typename T, typename ST> 
Storage2D<T,ST>::~Storage2D()
{
}

template<typename T, typename ST>
const std::string& Storage2D<T,ST>::name() const
{
  return Storage2D<T,ST>::stor2D_name_;
}

template<typename T, typename ST>
inline T* Storage2D<T,ST>::row_ptr(ST y)
{
  assert(y < yDim_);
  return StorageBase<T,ST>::data_ + y * xDim_;
} 
  
template<typename T, typename ST>  
inline T* Storage2D<T,ST>::row_ptr(ST y) const
{
  assert(y < yDim_);
  return StorageBase<T,ST>::data_ + y * xDim_;  
}

template<typename T, typename ST>  
void Storage2D<T,ST>::set_row(ST y, const Storage1D<T,ST>& row_vec) 
{ 
  assert(y < yDim_);
  assert(row_vec.size() == xDim_);

  T* data = StorageBase<T,ST>::data_ + y * xDim_;
  Makros::unified_assign(data, row_vec.direct_access(), xDim_);
}

// template<>
// void Storage2D<int>::set_row(size_t y, const Storage1D<int>& row_vec);

// template<>
// void Storage2D<uint>::set_row(size_t y, const Storage1D<uint>& row_vec);

// template<>
// void Storage2D<float>::set_row(size_t y, const Storage1D<float>& row_vec);

// template<>
// void Storage2D<double>::set_row(size_t y, const Storage1D<double>& row_vec);

// template<>
// void Storage2D<long double>::set_row(size_t y, const Storage1D<long double>& row_vec);

template<typename T, typename ST>
inline T Storage2D<T,ST>::value(ST i) const
{
  return StorageBase<T,ST>::data_[i];
}

template<typename T, typename ST>
inline ST Storage2D<T,ST>::xDim() const
{
  return xDim_;
}

template<typename T, typename ST>
inline ST Storage2D<T,ST>::yDim() const
{
  return yDim_;
}

template<typename T, typename ST>
inline std::pair<ST,ST> Storage2D<T,ST>::dims() const
{
  return std::make_pair(xDim_,yDim_);
}

template <typename T, typename ST>
OPTINLINE const T& Storage2D<T,ST>::operator()(ST x, ST y) const
{
#ifdef SAFE_MODE
  if (x >= xDim_ || y >= yDim_) {
    INTERNAL_ERROR << "    const access on element(" << x << "," << y
                   << ") exceeds storage dimensions of (" << xDim_ << "," << yDim_ << ")" << std::endl;
    std::cerr << "      in 2Dstorage \"" << this->name() << "\" of type "
              << Makros::Typename<T>()
              //<< Makros::get_typename(typeid(T).name())
              << ". Exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  return StorageBase<T,ST>::data_[y*xDim_+x];
}


template <typename T, typename ST>
OPTINLINE T& Storage2D<T,ST>::operator()(ST x, ST y)
{
#ifdef SAFE_MODE
  if (x >= xDim_ || y >= yDim_) {
    INTERNAL_ERROR << "    access on element(" << x << "," << y
                   << ") exceeds storage dimensions of (" << xDim_ << "," << yDim_ << ")" << std::endl;
    std::cerr << "   in 2Dstorage \"" << this->name() << "\" of type "
              << Makros::Typename<T>()
              //<< typeid(T).name()
              //<< Makros::get_typename(typeid(T).name())
              << ". exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  return StorageBase<T,ST>::data_[y*xDim_+x];
}

template <typename T, typename ST>
void Storage2D<T,ST>::operator=(const Storage2D<T,ST>& toCopy)
{
  if (StorageBase<T,ST>::size_ != toCopy.size()) {
    if (StorageBase<T,ST>::data_ != 0)
      delete[] StorageBase<T,ST>::data_;

    StorageBase<T,ST>::size_ = toCopy.size();
    StorageBase<T,ST>::data_ = new T[StorageBase<T,ST>::size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  
  const ST size = StorageBase<T,ST>::size_;
  
  assert(size == xDim_*yDim_);
  Makros::unified_assign(StorageBase<T,ST>::data_, toCopy.direct_access(), size);

  // for (ST i = 0; i < size; i++)
    // data_[i] = toCopy.value(i);

  //this is faster for basic types but it fails for complex types where e.g. arrays have to be copied
  //memcpy(data_,toCopy.direct_access(),size_*sizeof(T));
}


#ifdef SAFE_MODE
//for some reason g++ allows to assign an object of type T, but this does NOT produce the effect one would expect
// => define this operator in safe mode, only to check that such an assignment is not made
template<typename T,typename ST>
void Storage2D<T,ST>::operator=(const T& invalid_object)
{
  INTERNAL_ERROR << "assignment of an atomic entity to Storage2D \"" << this->name() << "\" of type "
                 << Makros::Typename<T>()
                 << " with " << StorageBase<T,ST>::size_ << " elements. exiting." << std::endl;
}
#endif


// template<>
// void Storage2D<int>::operator=(const Storage2D<int>& toCopy);

// template<>
// void Storage2D<uint>::operator=(const Storage2D<uint>& toCopy);

// template<>
// void Storage2D<float>::operator=(const Storage2D<float>& toCopy);

// template<>
// void Storage2D<double>::operator=(const Storage2D<double>& toCopy);

// template<>
// void Storage2D<long double>::operator=(const Storage2D<long double>& toCopy);


template <typename T, typename ST>
void Storage2D<T,ST>::resize(ST newxDim, ST newyDim)
{
  if (StorageBase<T,ST>::data_ == 0) {
    StorageBase<T,ST>::data_ = new T[newxDim*newyDim];
  }
  else if (newxDim != xDim_ || newyDim != yDim_) {

    T* new_data = new T[newxDim*newyDim];

    /* copy data */
    for (ST y=0; y < std::min(yDim_,newyDim); y++)
      for (ST x=0; x < std::min(xDim_,newxDim); x++)
        new_data[y*newxDim+x] = StorageBase<T,ST>::data_[y*xDim_+x];

    delete[] StorageBase<T,ST>::data_;
    StorageBase<T,ST>::data_ = new_data;
  }

  xDim_ = newxDim;
  yDim_ = newyDim;
  StorageBase<T,ST>::size_ = xDim_*yDim_;
}

template <typename T, typename ST>
void Storage2D<T,ST>::resize(ST newxDim, ST newyDim, const T fill_value)
{
  const uint newsize = newxDim*newyDim;

  if (StorageBase<T,ST>::data_ == 0) {
    StorageBase<T,ST>::data_ = new T[newsize];
    std::fill_n(StorageBase<T,ST>::data_,newsize,fill_value);
  }
  else if (newxDim != xDim_ || newyDim != yDim_) {

    T* new_data = new T[newsize];
    std::fill_n(new_data,newsize,fill_value);

    //for (ST i=0; i < newsize; i++)
    //  new_data[i] = fill_value;

    /* copy data */
    for (ST y=0; y < std::min(yDim_,newyDim); y++)
      for (ST x=0; x < std::min(xDim_,newxDim); x++)
        new_data[y*newxDim+x] = StorageBase<T,ST>::data_[y*xDim_+x];

    delete[] StorageBase<T,ST>::data_;
    StorageBase<T,ST>::data_ = new_data;
  }

  xDim_ = newxDim;
  yDim_ = newyDim;
  StorageBase<T,ST>::size_ = newsize;
}

template<typename T, typename ST>
void Storage2D<T,ST>::resize_dirty(ST newxDim, ST newyDim)
{
  if (newxDim != xDim_ || newyDim != yDim_) {
    if (StorageBase<T,ST>::data_ != 0) {
      delete[] StorageBase<T,ST>::data_;
    }

    xDim_ = newxDim;
    yDim_ = newyDim;
    StorageBase<T,ST>::size_ = xDim_*yDim_;

    StorageBase<T,ST>::data_ = new T[StorageBase<T,ST>::size_];
  }
}

/***** implementation of NamedStorage2D ********/

template<typename T, typename ST> 
NamedStorage2D<T,ST>::NamedStorage2D() : Storage2D<T,ST>(), name_("yyy") {}

template<typename T, typename ST> 
NamedStorage2D<T,ST>::NamedStorage2D(std::string name) : Storage2D<T,ST>(), name_(name) {}

template<typename T, typename ST> 
NamedStorage2D<T,ST>::NamedStorage2D(ST xDim, ST yDim, std::string name) : Storage2D<T,ST>(xDim,yDim), name_(name) {}

template<typename T, typename ST> 
NamedStorage2D<T,ST>::NamedStorage2D(ST xDim, ST yDim, T default_value, std::string name)
  : Storage2D<T,ST>(xDim,yDim,default_value), name_(name) {}

template<typename T, typename ST> 
NamedStorage2D<T,ST>::NamedStorage2D(const std::pair<ST,ST> dims, std::string name) 
  : Storage2D<T,ST>(dims), name_(name) {}

template<typename T, typename ST> 
NamedStorage2D<T,ST>::NamedStorage2D(const std::pair<ST,ST> dims, T default_value, std::string name)
  : Storage2D<T,ST>(dims,default_value), name_(name) {}

template<typename T, typename ST>
/*virtual*/ const std::string& NamedStorage2D<T,ST>::name() const
{
  return name_;
}

template<typename T, typename ST>
inline void NamedStorage2D<T,ST>::operator=(const Storage2D<T,ST>& toCopy)
{
  Storage2D<T,ST>::operator=(toCopy);
}

//NOTE: the name is NOT copied
template<typename T, typename ST>
inline void NamedStorage2D<T,ST>::operator=(const NamedStorage2D<T,ST>& toCopy)
{
  Storage2D<T,ST>::operator=(static_cast<Storage2D<T,ST> >(toCopy));
}


template<typename T, typename ST>
bool operator==(const Storage2D<T,ST>& v1, const Storage2D<T,ST>& v2)
{
  if (v1.xDim() != v2.xDim() || v1.yDim() != v2.yDim())
    return false;

  for (ST i=0; i < v1.size(); i++) {
    if (v1.direct_access(i) != v2.direct_access(i))
      return false;
  }

  return true;
}

template<typename T, typename ST>
bool operator!=(const Storage2D<T,ST>& v1, const Storage2D<T,ST>& v2)
{
  return !operator==(v1,v2);
}

#endif
