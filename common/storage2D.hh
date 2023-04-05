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

  using Base = StorageBase<T,ST>;

  //default constructor
  explicit Storage2D();

  explicit Storage2D(ST xDim, ST yDim);

  explicit Storage2D(ST xDim, ST yDim, T default_value);

  explicit Storage2D(const std::pair<ST,ST> dims);

  explicit Storage2D(const std::pair<ST,ST> dims, T default_value);

  //copy constructor
  Storage2D(const Storage2D<T,ST>& toCopy);

  //move constructor
  Storage2D(Storage2D<T,ST>&& toTake);

  ~Storage2D() = default;

  virtual const std::string& name() const;

  //saves all existing entries, new positions contain undefined data
  void resize(ST newxDim, ST newyDim) noexcept;

  inline void resize(const std::pair<ST,ST> dims) noexcept
  {
    resize(dims.first, dims.second);
  }

  //saves all existing entries, new positions are filled with <code> fill_value </code>
  void resize(ST newxDim, ST newyDim, const T fill_value) noexcept;

  inline void resize(const std::pair<ST,ST> dims, const T fill_value) noexcept
  {
    resize(dims.first, dims.second, fill_value);
  }

  //all elements are uninitialized after this operation
  void resize_dirty(ST newxDim, ST newyDim) noexcept;

  inline void resize_dirty(const std::pair<ST,ST> dims) noexcept
  {
    resize_dirty(dims.first, dims.second);
  }

  //access on an element
  inline const T& operator()(ST x, ST y) const noexcept;

  inline T& operator()(ST x, ST y) noexcept;

  Storage2D<T,ST>& operator=(const Storage2D<T,ST>& toCopy) noexcept;

  Storage2D<T,ST>& operator=(Storage2D<T,ST>&& toTake) noexcept;

  inline T* row_ptr(ST y) noexcept;

  inline T* row_ptr(ST y) const noexcept;

  inline T value(ST i) const noexcept;

  void set_row(ST y, const Storage1D<T,ST>& row_vec) noexcept;

  void get_row(ST y, Storage1D<T,ST>& row_vec) const noexcept;

  inline ST xDim() const noexcept;

  inline ST yDim() const noexcept;

  inline std::pair<ST,ST> dims() const noexcept;

  void swap(Storage2D<T,ST>& toSwap) noexcept;

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

  ~NamedStorage2D() = default;

  virtual const std::string& name() const;

  inline void operator=(const Storage2D<T,ST>& toCopy);

  inline void operator=(Storage2D<T,ST>&& toTake);

  //NOTE: the name is NOT copied
  inline void operator=(const NamedStorage2D<T,ST>& toCopy);

  //NOTE: the name is NOT taken
  inline void operator=(NamedStorage2D<T,ST>&& toTake);

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
template<typename T, typename ST> Storage2D<T,ST>::Storage2D() : StorageBase<T,ST>(), xDim_(0), yDim_(0) {}

template<typename T, typename ST> Storage2D<T,ST>::Storage2D(ST xDim, ST yDim) : StorageBase<T,ST>(xDim*yDim), xDim_(xDim), yDim_(yDim)
{
}

template<typename T, typename ST> Storage2D<T,ST>::Storage2D(ST xDim, ST yDim, const T default_value)
  : StorageBase<T,ST>(xDim*yDim, default_value), xDim_(xDim), yDim_(yDim)
{
}

template<typename T, typename ST> Storage2D<T,ST>::Storage2D(const std::pair<ST,ST> dims)
  : StorageBase<T,ST>(dims.first*dims.second), xDim_(dims.first), yDim_(dims.second)
{
}

template<typename T, typename ST> Storage2D<T,ST>::Storage2D(const std::pair<ST,ST> dims, T default_value)
  : StorageBase<T,ST>(dims.first*dims.second, default_value), xDim_(dims.first), yDim_(dims.second)
{
}

//copy constructor
template<typename T, typename ST> Storage2D<T,ST>::Storage2D(const Storage2D<T,ST>& toCopy) : StorageBase<T,ST>(toCopy.xDim()*toCopy.yDim())
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();

  const ST size = Base::size_;
  assert(size == xDim_*yDim_);

  if (size > 0)
    Makros::unified_assign(Base::data_, toCopy.direct_access(), size);
}

//move constructor
template<typename T, typename ST> Storage2D<T,ST>::Storage2D(Storage2D<T,ST>&& toTake) : StorageBase<T,ST>(toTake)
{
  xDim_ = toTake.xDim_;
  yDim_ = toTake.yDim_;
}

template<typename T, typename ST>
const std::string& Storage2D<T,ST>::name() const
{
  return Storage2D<T,ST>::stor2D_name_;
}

template<typename T, typename ST>
inline T* Storage2D<T,ST>::row_ptr(ST y) noexcept
{
  assert(y <= yDim_); //allow use as endpointer for last row
  return Base::data_ + y * xDim_;
}

template<typename T, typename ST>
inline T* Storage2D<T,ST>::row_ptr(ST y) const noexcept
{
  assert(y <= yDim_); //allow use as endpointer for last row
  return Base::data_ + y * xDim_;
}

template<typename T, typename ST>
void Storage2D<T,ST>::set_row(ST y, const Storage1D<T,ST>& row_vec) noexcept
{
  assert(y < yDim_);
  assert(row_vec.size() == xDim_);

  T* data = Base::data_ + y * xDim_;
  Makros::unified_assign(data, row_vec.direct_access(), xDim_);
}

template<typename T, typename ST>
void Storage2D<T,ST>::get_row(ST y, Storage1D<T,ST>& row_vec) const noexcept
{
  assert(y < yDim_);
  assert(row_vec.size() == xDim_);

  const T* data = Base::data_ + y * xDim_;
  Makros::unified_assign(row_vec.direct_access(), data, xDim_);
}

template<typename T, typename ST>
inline T Storage2D<T,ST>::value(ST i) const noexcept
{
  return Base::data_[i];
}

template<typename T, typename ST>
inline ST Storage2D<T,ST>::xDim() const noexcept
{
  return xDim_;
}

template<typename T, typename ST>
inline ST Storage2D<T,ST>::yDim() const noexcept
{
  return yDim_;
}

template<typename T, typename ST>
inline std::pair<ST,ST> Storage2D<T,ST>::dims() const noexcept
{
  return std::make_pair(xDim_,yDim_);
}

template<typename T, typename ST>
OPTINLINE const T& Storage2D<T,ST>::operator()(ST x, ST y) const noexcept
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
  return Base::data_[y*xDim_+x];
}


template<typename T, typename ST>
OPTINLINE T& Storage2D<T,ST>::operator()(ST x, ST y) noexcept
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
  return Base::data_[y*xDim_+x];
}

template<typename T, typename ST>
Storage2D<T,ST>&  Storage2D<T,ST>::operator=(const Storage2D<T,ST>& toCopy) noexcept
{
  if (Base::size_ != toCopy.size()) {
    if (Base::data_ != 0)
      delete[] Base::data_;

    Base::size_ = toCopy.size();
    Base::data_ = new T[Base::size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();

  const ST size = Base::size_;

  assert(size == xDim_*yDim_);
  Makros::unified_assign(Base::data_, toCopy.direct_access(), size);
  // for (ST i = 0; i < size; i++)
  // data_[i] = toCopy.value(i);

  return *this;
}

template<typename T, typename ST>
Storage2D<T,ST>& Storage2D<T,ST>::operator=(Storage2D<T,ST>&& toTake) noexcept
{
  delete[] Base::data_;
  Base::data_ = toTake.data_;
  toTake.data_ = 0;

  xDim_ = toTake.xDim();
  yDim_ = toTake.yDim();
  Base::size_ = toTake.size();
  return *this;
}

template<typename T, typename ST>
void Storage2D<T,ST>::resize(ST newxDim, ST newyDim) noexcept
{
  if (Base::data_ == 0) {
    Base::data_ = new T[newxDim*newyDim];
  }
  else if (newxDim != xDim_ || newyDim != yDim_) {

    T* new_data = new T[newxDim*newyDim];

    /* copy data */
    for (ST y=0; y < std::min(yDim_,newyDim); y++)
      for (ST x=0; x < std::min(xDim_,newxDim); x++)
        new_data[y*newxDim+x] = std::move(Base::data_[y*xDim_+x]);

    delete[] Base::data_;
    Base::data_ = new_data;
  }

  xDim_ = newxDim;
  yDim_ = newyDim;
  Base::size_ = xDim_*yDim_;
}

template<typename T, typename ST>
void Storage2D<T,ST>::resize(ST newxDim, ST newyDim, const T fill_value) noexcept
{
  const uint newsize = newxDim*newyDim;

  if (Base::data_ == 0) {
    Base::data_ = new T[newsize];
    std::fill_n(Base::data_,newsize,fill_value);
  }
  else if (newxDim != xDim_ || newyDim != yDim_) {

    T* new_data = new T[newsize];
    std::fill_n(new_data,newsize,fill_value);

    //for (ST i=0; i < newsize; i++)
    //  new_data[i] = fill_value;

    /* copy data */
    for (ST y=0; y < std::min(yDim_,newyDim); y++)
      for (ST x=0; x < std::min(xDim_,newxDim); x++)
        new_data[y*newxDim+x] = std::move(Base::data_[y*xDim_+x]);

    delete[] Base::data_;
    Base::data_ = new_data;
  }

  xDim_ = newxDim;
  yDim_ = newyDim;
  Base::size_ = newsize;
}

template<typename T, typename ST>
void Storage2D<T,ST>::resize_dirty(ST newxDim, ST newyDim) noexcept
{
  if (newxDim != xDim_ || newyDim != yDim_) {
    if (Base::data_ != 0) {
      delete[] Base::data_;
    }

    xDim_ = newxDim;
    yDim_ = newyDim;
    Base::size_ = xDim_*yDim_;

    Base::data_ = new T[Base::size_];
  }
}

template<typename T, typename ST>
void Storage2D<T,ST>::swap(Storage2D<T,ST>& toSwap) noexcept
{
  std::swap(Base::data_, toSwap.data_);
  std::swap(Base::size_, toSwap.size_);
  std::swap(xDim_, toSwap.xDim_);
  std::swap(yDim_, toSwap.yDim_);
}

/***** implementation of NamedStorage2D ********/

template<typename T, typename ST> NamedStorage2D<T,ST>::NamedStorage2D() : Storage2D<T,ST>(), name_("yyy") {}

template<typename T, typename ST> NamedStorage2D<T,ST>::NamedStorage2D(std::string name) : Storage2D<T,ST>(), name_(name) {}

template<typename T, typename ST> NamedStorage2D<T,ST>::NamedStorage2D(ST xDim, ST yDim, std::string name) : Storage2D<T,ST>(xDim,yDim), name_(name) {}

template<typename T, typename ST> NamedStorage2D<T,ST>::NamedStorage2D(ST xDim, ST yDim, T default_value, std::string name)
  : Storage2D<T,ST>(xDim,yDim,default_value), name_(name) {}

template<typename T, typename ST> NamedStorage2D<T,ST>::NamedStorage2D(const std::pair<ST,ST> dims, std::string name)
  : Storage2D<T,ST>(dims), name_(name) {}

template<typename T, typename ST> NamedStorage2D<T,ST>::NamedStorage2D(const std::pair<ST,ST> dims, T default_value, std::string name)
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

template<typename T, typename ST>
inline void NamedStorage2D<T,ST>::operator=(Storage2D<T,ST>&& toTake)
{
  Storage2D<T,ST>::operator=(toTake);
}

//NOTE: the name is NOT copied
template<typename T, typename ST>
inline void NamedStorage2D<T,ST>::operator=(const NamedStorage2D<T,ST>& toCopy)
{
  Storage2D<T,ST>::operator=(static_cast<const Storage2D<T,ST>&>(toCopy));
}

//NOTE: the name is NOT taken
template<typename T, typename ST>
inline void NamedStorage2D<T,ST>::operator=(NamedStorage2D<T,ST>&& toTake)
{
  Storage2D<T,ST>::operator=(static_cast<Storage2D<T,ST>&&>(toTake));
}

template<typename T, typename ST>
bool operator==(const Storage2D<T,ST>& v1, const Storage2D<T,ST>& v2)
{
  if (v1.xDim() != v2.xDim() || v1.yDim() != v2.yDim())
    return false;

  if (std::is_trivially_copyable<T>::value) {
    return Routines::equals(v1.direct_access(), v2.direct_access(), v1.size());
  }
  else {

    for (ST i=0; i < v1.size(); i++) {
      if (v1.direct_access(i) != v2.direct_access(i))
        return false;
    }
  }

  return true;
}

template<typename T, typename ST>
bool operator!=(const Storage2D<T,ST>& v1, const Storage2D<T,ST>& v2)
{
  return !operator==(v1,v2);
}

#endif
