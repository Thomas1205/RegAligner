/*-*-c++-*-*/
/*** first version written by Thomas Schoenemann as a private person without employment, September 2009 ***/
/*** much refined by Thomas Schoenemann  at Lund University, Sweden, the University of Pisa, Italy, ***
 *** and the University of Düsseldorf, Germany 2010 - 2012 **/
/*** if you desire the checked version, make sure your compiler defines the option SAFE_MODE on the command line ***/

#ifndef STORAGE_3D_HH
#define STORAGE_3D_HH

#include "storage1D.hh"

template<typename ST>
struct Dim3D {

  Dim3D(ST xDim, ST yDim, ST zDim) : xDim_(xDim), yDim_(yDim), zDim_(zDim) {}

  ST xDim_;
  ST yDim_;
  ST zDim_;
};

template<typename ST>
bool operator==(const Dim3D<ST>& d1, const Dim3D<ST>& d2) noexcept
{

  return (d1.xDim_ == d2.xDim_ && d1.yDim_ == d2.yDim_ && d1.zDim_ == d2.zDim_);
}

template<typename ST>
bool operator!=(const Dim3D<ST>& d1, const Dim3D<ST>& d2) noexcept
{

  return (d1.xDim_ != d2.xDim_ || d1.yDim_ != d2.yDim_ || d1.zDim_ != d2.zDim_);
}

template<typename T, typename ST=size_t>
class Storage3D : public StorageBase<T,ST> {
public:

  using Base = StorageBase<T,ST>;

  explicit Storage3D();

  //copy constructor
  Storage3D(const Storage3D<T,ST>& toCopy);

  //move constructor
  Storage3D(Storage3D<T,ST>&& toTake);

  explicit Storage3D(ST xDim, ST yDim, ST zDim);

  explicit Storage3D(const Dim3D<ST> dims);

  explicit Storage3D(ST xDim, ST yDim, ST zDim, const T default_value);

  explicit Storage3D(const Dim3D<ST> dims, const T default_value);

  ~Storage3D() = default;

  inline const T& operator()(ST x, ST y, ST z) const noexcept;

  inline T& operator()(ST x, ST y, ST z) noexcept;

  virtual const std::string& name() const;

  inline ST xDim() const noexcept;

  inline ST yDim() const noexcept;

  inline ST zDim() const noexcept;

  Dim3D<ST> dims() const noexcept;

  void set_x(ST y, ST z, const Storage1D<T,ST>& vec) noexcept;

  void get_x(ST y, ST z, Storage1D<T,ST>& vec) const noexcept;

  void set_y(ST x, ST z, const Storage1D<T,ST>& vec) noexcept;

  void get_y(ST x, ST z, Storage1D<T,ST>& vec) const noexcept;

  void set_z(ST x, ST y, const Storage1D<T,ST>& vec) noexcept;

  void get_z(ST x, ST y, Storage1D<T,ST>& vec) const noexcept;

  Storage3D<T,ST>& operator=(const Storage3D<T,ST>& toCopy) noexcept;

  Storage3D<T,ST>& operator=(Storage3D<T,ST>&& toTake) noexcept;

  //existing positions are copied, new ones are uninitialized
  void resize(ST newxDim, ST newyDim, ST newzDim) noexcept;

  inline void resize(const Dim3D<ST> dims) noexcept
  {
    resize(dims.xDim_, dims.yDim_, dims.zDim_);
  }

  //existing positions are copied, new ones are uninitialized
  void resize(ST newxDim, ST newyDim, ST newzDim, const T default_value) noexcept;

  inline void resize(const Dim3D<ST> dims, T default_value) noexcept
  {
    resize(dims.xDim_, dims.yDim_, dims.zDim_, default_value);
  }

  //all elements are uninitialized after this operation
  void resize_dirty(ST newxDim, ST newyDim, ST newzDim) noexcept;

  inline void resize_dirty(const Dim3D<ST> dims) noexcept
  {
    resize_dirty(dims.xDim_, dims.yDim_, dims.zDim_);
  }

  void swap(Storage3D<T,ST>& toSwap) noexcept;

protected:
  ST xDim_;
  ST yDim_;
  ST zDim_;

  static const std::string stor3D_name_;
};


template<typename T, typename ST=size_t>
class NamedStorage3D : public Storage3D<T,ST> {
public:

  NamedStorage3D();

  NamedStorage3D(std::string name);

  NamedStorage3D(ST xDim, ST yDim, ST zDim, std::string name);

  NamedStorage3D(ST xDim, ST yDim, ST zDim, T default_value, std::string name);

  ~NamedStorage3D() = default;

  virtual const std::string& name() const;

  inline void operator=(const Storage3D<T,ST>& toCopy);

  inline void operator=(Storage3D<T,ST>&& toTake);

  //NOTE: the name is NOT copied
  inline void operator=(const NamedStorage3D<T,ST>& toCopy);

  //NOTE: the name is NOT taken
  inline void operator=(NamedStorage3D<T,ST>&& toTake);

protected:
  std::string name_;
};

template<typename T, typename ST>
bool operator==(const Storage3D<T,ST>& v1, const Storage3D<T,ST>& v2);

template<typename T, typename ST>
bool operator!=(const Storage3D<T,ST>& v1, const Storage3D<T,ST>& v2);


namespace Makros {

  template<typename T, typename ST>
  class Typename<Storage3D<T,ST> > {
  public:

    std::string name() const
    {
      return "Storage3D<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Storage3D<T> > {
  public:

    std::string name() const
    {
      return "Storage3D<" + Makros::Typename<T>() + "> ";
    }
  };


  template<typename T, typename ST>
  class Typename<NamedStorage3D<T,ST> > {
  public:

    std::string name() const
    {
      return "NamedStorage3D<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<NamedStorage3D<T> > {
  public:

    std::string name() const
    {
      return "NamedStorage3D<" + Makros::Typename<T>() + "> ";
    }
  };

}


/******************************************** implementation **************************************************/
template<typename T, typename ST>
/*static*/ const std::string Storage3D<T,ST>::stor3D_name_ = "unnamed Storage3D";

template<typename T, typename ST> Storage3D<T,ST>::Storage3D() : StorageBase<T,ST>(), xDim_(0), yDim_(0), zDim_(0) {}

template<typename T, typename ST> Storage3D<T,ST>::Storage3D(const Storage3D<T,ST>& toCopy) : StorageBase<T,ST>(toCopy.xDim()*toCopy.yDim()*toCopy.zDim())
{
  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  zDim_ = toCopy.zDim();

  Makros::unified_assign(Base::data_, toCopy.direct_access(), Base::size_);

  //for (ST i=0; i < size_; i++)
  //  data_[i] = toCopy.direct_access(i);
}

//move constructor
template<typename T, typename ST> Storage3D<T,ST>::Storage3D(Storage3D<T,ST>&& toTake) : StorageBase<T,ST>(toTake)
{
  xDim_ = toTake.xDim();
  yDim_ = toTake.yDim();
  zDim_ = toTake.zDim();
}

template<typename T, typename ST> Storage3D<T,ST>::Storage3D(ST xDim, ST yDim, ST zDim) : StorageBase<T,ST>(xDim*yDim*zDim), xDim_(xDim), yDim_(yDim), zDim_(zDim)
{
}

template<typename T, typename ST> Storage3D<T,ST>::Storage3D(const Dim3D<ST> dims)
  : StorageBase<T,ST>(dims.xDim_*dims.yDim_*dims.zDim_), xDim_(dims.xDim_), yDim_(dims.yDim_), zDim_(dims.zDim_) {}

template<typename T, typename ST> Storage3D<T,ST>::Storage3D(ST xDim, ST yDim, ST zDim, const T default_value)
  : StorageBase<T,ST>(xDim*yDim*zDim,default_value), xDim_(xDim), yDim_(yDim), zDim_(zDim)
{
}

template<typename T, typename ST> Storage3D<T,ST>::Storage3D(const Dim3D<ST> dims, const T default_value)
  : StorageBase<T,ST>(dims.xDim_*dims.yDim_*dims.zDim_,default_value), xDim_(dims.xDim_), yDim_(dims.yDim_), zDim_(dims.zDim_) {}

template<typename T, typename ST>
void Storage3D<T,ST>::set_x(ST y, ST z, const Storage1D<T,ST>& vec) noexcept
{
  assert(y < yDim_);
  assert(z < zDim_);
  assert(vec.size() == xDim_);

  for (ST x = 0; x < xDim_; x++)
    (*this)(x, y, z) = vec.direct_access(x);
}

template<typename T, typename ST>
void Storage3D<T,ST>::get_x(ST y, ST z, Storage1D<T,ST>& vec) const noexcept
{
  assert(y < yDim_);
  assert(z < zDim_);
  assert(vec.size() == xDim_);

  for (ST x = 0; x < xDim_; x++)
    vec.direct_access(x) = (*this)(x, y, z);
}

template<typename T, typename ST>
void Storage3D<T,ST>::set_y(ST x, ST z, const Storage1D<T,ST>& vec) noexcept
{
  assert(x < xDim_);
  assert(z < zDim_);
  assert(vec.size() == yDim_);

  for (ST y = 0; y < yDim_; y++)
    (*this)(x, y, z) = vec.direct_access(y);
}

template<typename T, typename ST>
void Storage3D<T,ST>::get_y(ST x, ST z, Storage1D<T,ST>& vec) const noexcept
{
  assert(x < xDim_);
  assert(z < zDim_);
  assert(vec.size() == yDim_);

  for (ST y = 0; y < yDim_; y++)
    vec.direct_access(y) = (*this)(x, y, z);
}

template<typename T, typename ST>
void Storage3D<T,ST>::set_z(ST x, ST y, const Storage1D<T,ST>& vec) noexcept
{
  assert(x < xDim_);
  assert(y < yDim_);
  assert(vec.size() == zDim_);

  T* data =  Base::data_ + (y*xDim_+x) * zDim_;

  Makros::unified_assign(data, vec.direct_access(), zDim_);

  //for (ST z = 0; z < zDim_; z++)
  //  data[z] = vec.direct_access(z);
}

template<typename T, typename ST>
void Storage3D<T,ST>::get_z(ST x, ST y, Storage1D<T,ST>& vec) const noexcept
{
  assert(x < xDim_);
  assert(y < yDim_);
  assert(vec.size() == zDim_);

  const T* data =  Base::data_ + (y*xDim_+x) * zDim_;

  Makros::unified_assign(vec.direct_access(), data, zDim_);
}

template<typename T, typename ST>
inline const T& Storage3D<T,ST>::operator()(ST x, ST y, ST z) const noexcept
{
#ifdef SAFE_MODE
  if (x >= xDim_ || y >= yDim_ || z >= zDim_) {

    INTERNAL_ERROR << "     invalid const access on element (" << x << "," << y << "," << z << ") of 3D-storage \""
                   << this->name() << "\" of type "
                   << Makros::Typename<T>()
                   //<< Makros::get_typename(typeid(T).name())
                   << ":" << std::endl;
    std::cerr << "     dimensions " << xDim_ << "x" << yDim_ << "x" << zDim_ << " exceeded. Exiting..." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  return Base::data_[(y*xDim_+x)*zDim_+z];
}

template<typename T, typename ST>
inline T& Storage3D<T,ST>::operator()(ST x, ST y, ST z) noexcept
{
#ifdef SAFE_MODE
  if (x >= xDim_ || y >= yDim_ || z >= zDim_) {

    INTERNAL_ERROR << "     invalid access on element (" << x << "," << y << "," << z << ") of 3D-storage \""
                   << this->name() << "\" of type "
                   //<< Makros::Typename<T>()
                   << typeid(T).name()
                   //<< Makros::get_typename(typeid(T).name())
                   << ":" << std::endl;
    std::cerr << "     dimensions " << xDim_ << "x" << yDim_ << "x" << zDim_ << " exceeded. Exiting..." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  return Base::data_[(y*xDim_+x)*zDim_+z];
}

template<typename T, typename ST>
/*virtual*/ const std::string& Storage3D<T,ST>::name() const
{
  return stor3D_name_;
}

template<typename T, typename ST>
inline ST Storage3D<T,ST>::xDim() const noexcept
{
  return xDim_;
}

template<typename T, typename ST>
inline ST Storage3D<T,ST>::yDim() const noexcept
{
  return yDim_;
}

template<typename T, typename ST>
inline ST Storage3D<T,ST>::zDim() const noexcept
{
  return zDim_;
}

template<typename T, typename ST>
Dim3D<ST> Storage3D<T,ST>::dims() const noexcept
{
  return Dim3D<ST>(xDim_,yDim_,zDim_);
}

template<typename T, typename ST>
Storage3D<T,ST>& Storage3D<T,ST>::operator=(const Storage3D<T,ST>& toCopy) noexcept
{
  if (Base::size_ != toCopy.size()) {
    if (Base::data_ != 0) {
      delete[] Base::data_;
    }

    Base::size_ = toCopy.size();
    Base::data_ = new T[Base::size_];
  }

  xDim_ = toCopy.xDim();
  yDim_ = toCopy.yDim();
  zDim_ = toCopy.zDim();

  const size_t size = Base::size_;
  assert(size == xDim_*yDim_*zDim_);

  Makros::unified_assign(Base::data_, toCopy.direct_access(), size);

  // for (ST i=0; i < size_; i++)
  // data_[i] = toCopy.direct_access(i);
  return *this;
}

template<typename T, typename ST>
Storage3D<T,ST>& Storage3D<T,ST>::operator=(Storage3D<T,ST>&& toTake) noexcept
{
  delete[] Base::data_;
  Base::data_ = toTake.data_;
  toTake.data_ = 0;

  xDim_ = toTake.xDim();
  yDim_ = toTake.yDim();
  zDim_ = toTake.zDim();

  Base::size_ = toTake.size();
  return *this;
}

//existing positions are copied, new ones are uninitialized
template<typename T, typename ST>
void Storage3D<T,ST>::resize(ST newxDim, ST newyDim, ST newzDim) noexcept
{
  const ST new_size = newxDim*newyDim*newzDim;

  if (newxDim != xDim_ || newyDim != yDim_ || newzDim != zDim_) {
    T* new_data = new T[new_size];

    if (Base::data_ != 0) {

      //copy existing elements
      for (ST x=0; x < std::min(xDim_,newxDim); x++) {
        for (ST y=0; y < std::min(yDim_,newyDim); y++) {
          for (ST z=0; z < std::min(zDim_,newzDim); z++) {
            new_data[(y*newxDim+x)*newzDim+z] = std::move(Base::data_[(y*xDim_+x)*zDim_+z]);
          }
        }
      }

      delete[] Base::data_;
    }
    Base::data_ = new_data;
    Base::size_ = new_size;
    xDim_ = newxDim;
    yDim_ = newyDim;
    zDim_ = newzDim;
  }
}

//existing positions are copied, new ones are uninitialized
template<typename T, typename ST>
void Storage3D<T,ST>::resize(ST newxDim, ST newyDim, ST newzDim, const T default_value) noexcept
{
  const ST new_size = newxDim*newyDim*newzDim;

  if (newxDim != xDim_ || newyDim != yDim_ || newzDim != zDim_) {

    T* new_data = new T[new_size];
    std::fill_n(new_data,new_size,default_value);
    //for (ST i=0; i < new_size; i++)
    //  new_data[i] = default_value;

    if (Base::data_ != 0) {

      //copy existing elements
      for (ST x=0; x < std::min(xDim_,newxDim); x++) {
        for (ST y=0; y < std::min(yDim_,newyDim); y++) {
          for (ST z=0; z < std::min(zDim_,newzDim); z++) {
            new_data[(y*newxDim+x)*newzDim+z] = std::move(Base::data_[(y*xDim_+x)*zDim_+z]);
          }
        }
      }

      delete[] Base::data_;
    }
    Base::data_ = new_data;
    Base::size_ = new_size;
    xDim_ = newxDim;
    yDim_ = newyDim;
    zDim_ = newzDim;
  }
}

//all elements are uninitialized after this operation
template<typename T, typename ST>
void Storage3D<T,ST>::resize_dirty(ST newxDim, ST newyDim, ST newzDim) noexcept
{
  if (newxDim != xDim_ || newyDim != yDim_ || newzDim != zDim_) {

    if (Base::data_ != 0)
      delete[] Base::data_;

    xDim_ = newxDim;
    yDim_ = newyDim;
    zDim_ = newzDim;
    Base::size_ = xDim_*yDim_*zDim_;

    Base::data_ = new T[Base::size_];
  }
}

template<typename T, typename ST>
void Storage3D<T,ST>::swap(Storage3D<T,ST>& toSwap) noexcept
{
  std::swap(Base::data_, toSwap.data_);
  std::swap(Base::size_, toSwap.size_);
  std::swap(xDim_, toSwap.xDim_);
  std::swap(yDim_, toSwap.yDim_);
  std::swap(zDim_, toSwap.zDim_);
}

/***********************/

template<typename T, typename ST> NamedStorage3D<T,ST>::NamedStorage3D() : Storage3D<T,ST>(), name_("yyy") {}

template<typename T, typename ST> NamedStorage3D<T,ST>::NamedStorage3D(std::string name) : Storage3D<T,ST>(), name_(name) {}

template<typename T, typename ST> NamedStorage3D<T,ST>::NamedStorage3D(ST xDim, ST yDim, ST zDim, std::string name) :
  Storage3D<T,ST>(xDim,yDim,zDim), name_(name) {}

template<typename T, typename ST> NamedStorage3D<T,ST>::NamedStorage3D(ST xDim, ST yDim, ST zDim, T default_value, std::string name)
  : Storage3D<T,ST>(xDim,yDim,zDim,default_value), name_(name) {}

template<typename T, typename ST>
/*virtual*/ const std::string& NamedStorage3D<T,ST>::name() const
{
  return name_;
}

template<typename T, typename ST>
inline void NamedStorage3D<T,ST>::operator=(const Storage3D<T,ST>& toCopy)
{
  Storage3D<T,ST>::operator=(toCopy);
}

template<typename T, typename ST>
inline void NamedStorage3D<T,ST>::operator=(Storage3D<T,ST>&& toTake)
{
  Storage3D<T,ST>::operator=(toTake);
}

//NOTE: the name is NOT copied
template<typename T, typename ST>
inline void NamedStorage3D<T,ST>::operator=(const NamedStorage3D<T,ST>& toCopy)
{
  Storage3D<T,ST>::operator=(static_cast<const Storage3D<T,ST>& >(toCopy));
}

//NOTE: the name is NOT copied
template<typename T, typename ST>
inline void NamedStorage3D<T,ST>::operator=(NamedStorage3D<T,ST>&& toTake)
{
  Storage3D<T,ST>::operator=(static_cast<Storage3D<T,ST>&&>(toTake));
}

template<typename T, typename ST>
bool operator==(const Storage3D<T,ST>& v1, const Storage3D<T,ST>& v2)
{
  if (v1.xDim() != v2.xDim() || v1.yDim() != v2.yDim() || v1.zDim() != v2.zDim())
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
bool operator!=(const Storage3D<T,ST>& v1, const Storage3D<T,ST>& v2)
{
  return !operator==(v1,v2);
}

#endif
