/*-*-c++-*-*/
/*** first version written by Thomas Schoenemann as a private person without employment, September 2009 ***/
/*** much refined by Thomas Schoenemann  at Lund University, Sweden, the University of Pisa, Italy, ***
 *** and the University of Düsseldorf, Germany 2010 - 2012 **/
/*** if you desire the checked version, make sure your compiler defines the option SAFE_MODE on the command line ***/


#ifndef STORAGE1D_HH
#define STORAGE1D_HH

#include "makros.hh"
#include "routines.hh"
#include "storage_base.hh"
#include <cstring>
#include <type_traits>

template<typename T, typename ST=size_t>
class Storage1D : public StorageBase<T,ST> {
public:

  using Base = StorageBase<T,ST>;

  using PassType = typename std::conditional<std::is_fundamental<T>::value || std::is_pointer<T>::value, const T, const T&>::type;

  explicit Storage1D() noexcept;

  explicit Storage1D(ST size); //new throws exceptions

  explicit Storage1D(ST size, PassType default_value); //new throws exceptions

  Storage1D(const std::initializer_list<T>& init); //new throws exceptions

  //copy constructor
  Storage1D(const Storage1D<T,ST>& toCopy); //new throws exceptions

  //move constructor
  Storage1D(Storage1D<T,ST>&& toTake) noexcept;

  ~Storage1D() = default;

  virtual const std::string& name() const;

  inline const T& operator[](ST i) const noexcept;

  inline T& operator[](ST i) noexcept;

  Storage1D<T,ST>& operator=(const Storage1D<T,ST>& toCopy); //new throws exceptions
 
  Storage1D<T,ST>& operator=(Storage1D<T,ST>&& toTake) noexcept; 

  Storage1D<T,ST>& operator=(const std::initializer_list<T>& init); //new throws exceptions

  T back() const noexcept;

  T& back() noexcept;

  //maintains the values of existing positions, new ones are undefined
  void resize(ST new_size); //new throws exceptions

  //maintains the values of existing positions, new ones are undefined. Swapping may be faster if e.g. T is std::vector or Storage1D
  //this should be superfluous now that we use unified_move_assign
  template<class swap_op = SwapOp<T>>
  void resize_swap(ST newsize, swap_op op); //new throws exceptions

  //maintains the values of exisitng positions, new ones are filled with <code> fill_value </code>
  void resize(ST new_size, PassType fill_value); //new throws exceptions

  //all elements are undefined after this operation
  void resize_dirty(ST new_size); //new throws exceptions

  inline void range_set_constant(PassType constant, ST start, ST length) noexcept;

  void swap(Storage1D<T,ST>& toSwap) noexcept;
  
protected:

  static const std::string stor1D_name_;
};

template<typename T, typename ST=size_t>
class NamedStorage1D : public Storage1D<T,ST> {
public:

  NamedStorage1D();

  NamedStorage1D(std::string name);

  NamedStorage1D(ST size, std::string name);

  NamedStorage1D(ST size, T default_value, std::string name);

  NamedStorage1D(const NamedStorage1D<T,ST>& toCopy);

  NamedStorage1D(NamedStorage1D<T,ST>&& toTake);

  ~NamedStorage1D() = default;

  virtual const std::string& name() const;

  inline void operator=(const Storage1D<T,ST>& toCopy);

  inline void operator=(Storage1D<T,ST>&& toTake);

  //NOTE: the name is NOT copied
  inline void operator=(const NamedStorage1D<T,ST>& toCopy);

  //NOTE: the name is NOT taken
  inline void operator=(NamedStorage1D<T,ST>&& toTake);

protected:
  std::string name_;
};

template<typename T, typename ST>
std::ostream& operator<<(std::ostream& s, const Storage1D<T,ST>& v);

template<typename T, typename ST>
inline bool operator==(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

template<typename T, typename ST>
bool operator!=(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

//NOTE: operators implement lexicographical order. If you want speed, you should compare sizes first!

template<typename T,typename ST>
bool operator<(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

template<typename T,typename ST>
bool operator<=(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

template<typename T,typename ST>
bool operator>(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

template<typename T,typename ST>
bool operator>=(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

namespace Makros {


  template<typename T, typename ST>
  class Typename<Storage1D<T,ST> > {
  public:

    std::string name() const
    {
      return "Storage1D<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Storage1D<T> > {
  public:

    std::string name() const
    {
      return "Storage1D<" + Makros::Typename<T>() + "> ";
    }
  };

  template<typename T, typename ST>
  class Typename<NamedStorage1D<T,ST> > {
  public:

    std::string name() const
    {
      return "NamedStorage1D<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<NamedStorage1D<T> > {
  public:

    std::string name() const
    {
      return "NamedStorage1D<" + Makros::Typename<T>() + "> ";
    }
  };

}

/********************************************** implementation ************************************/

/******* implementation of Storage1D *********/

template<typename T,typename ST>
/*static*/ const std::string Storage1D<T,ST>::stor1D_name_ = "unnamed 1Dstorage";

template<typename T,typename ST> Storage1D<T,ST>::Storage1D() noexcept : StorageBase<T,ST>() {}

template<typename T,typename ST> Storage1D<T,ST>::Storage1D(ST size) : StorageBase<T,ST>(size)
{
}

template<typename T,typename ST> Storage1D<T,ST>::Storage1D(ST size, Storage1D<T,ST>::PassType default_value) : StorageBase<T,ST>(size, default_value)
{
}

template<typename T,typename ST> Storage1D<T,ST>::Storage1D(const std::initializer_list<T>& init) : StorageBase<T,ST>(init)
{
}

//copy constructor
template<typename T,typename ST> Storage1D<T,ST>::Storage1D(const Storage1D<T,ST>& toCopy) : StorageBase<T,ST>(toCopy)
{
}

//move constructor
template<typename T,typename ST> Storage1D<T,ST>::Storage1D(Storage1D<T,ST>&& toTake) noexcept : StorageBase<T,ST>(toTake)
{
}

template<typename T,typename ST>
Storage1D<T,ST>& Storage1D<T,ST>::operator=(Storage1D<T,ST>&& toTake) noexcept
{
  StorageBase<T,ST>::operator=(toTake);
  return *this;
}

template<typename T,typename ST>
Storage1D<T,ST>& Storage1D<T,ST>::operator=(const Storage1D<T,ST>& toCopy)
{
  StorageBase<T,ST>::operator=(toCopy);
  return *this;
}

template<typename T,typename ST>
Storage1D<T,ST>& Storage1D<T,ST>::operator=(const std::initializer_list<T>& init)
{
  const ST size = init.size();
  if (size != Base::size_) {
    delete[] Base::data_;
    Base::data_ = new T[size];
    Base::size_ = size;
  }
  std::copy(init.begin(),init.end(),Base::data_);
  return *this;
}

template<typename T,typename ST>
inline void Storage1D<T,ST>::range_set_constant(Storage1D<T,ST>::PassType constant, ST start, ST length) noexcept
{
  assert(start+length <= Base::size_);

  std::fill_n(Base::data_+start,length,constant); //experimental result: fill_n is usually faster
}

template<typename T,typename ST>
/*virtual*/ const std::string& Storage1D<T,ST>::name() const
{
  return Storage1D::stor1D_name_;
}

template<typename T,typename ST>
inline const T& Storage1D<T,ST>::operator[](ST i) const noexcept
{
#ifdef SAFE_MODE
  if (i >= Base::size_) {

    INTERNAL_ERROR << "    invalid const access on element " << i
                   << " for Storage1D " <<  "\"" << this->name() << "\" of type "
                   << Makros::Typename<T>()
                   //<< typeid(T).name()
                   << " with " << Base::size_ << " elements. exiting." << std::endl;

    print_trace();
    exit(1);
  }
#endif
  return Base::data_[i];
}

template<typename T,typename ST>
inline T& Storage1D<T,ST>::operator[](ST i) noexcept
{
#ifdef SAFE_MODE
  if (i >= Base::size_) {

    INTERNAL_ERROR << "    invalid access on element " << i
                   << " for Storage1D \"" << this->name() << "\" of type "
                   << Makros::Typename<T>()
                   << " with " << Base::size_ << " elements. exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  return Base::data_[i];
}

template<typename T,typename ST>
T Storage1D<T,ST>::back() const noexcept
{
  assert(Base::size_ > 0);
  return Base::data_[Base::size_-1];
}

template<typename T,typename ST>
T& Storage1D<T,ST>::back() noexcept
{
  assert(Base::size_ > 0);
  return Base::data_[Base::size_-1];
}

//maintains the values of existing positions, new ones are undefined
template<typename T,typename ST>
void Storage1D<T,ST>::resize(ST new_size)
{
  if (Base::data_ == 0) {
    Base::data_ = new T[new_size];
  }
  else if (Base::size_ != new_size) {

    T* new_data = new T[new_size];

    const ST size = std::min(Base::size_,new_size);

    Makros::unified_move_assign(new_data, Base::data_, size);

    delete[] Base::data_;
    Base::data_ = new_data;
  }

  Base::size_ = new_size;
}

//maintains the values of existing positions, new ones are undefined
template<typename T, typename ST>
template<class swap_op>
void Storage1D<T,ST>::resize_swap(ST new_size, swap_op op)
{
  if (Base::data_ == 0) {
    Base::data_ = new T[new_size];
  }
  else if (Base::size_ != new_size) {

    T* new_data = new T[new_size];

    const ST size = std::min(Base::size_,new_size);

    for (ST i=0; i < size; i++)
      op(new_data[i],Base::data_[i]);

    delete[] Base::data_;
    Base::data_ = new_data;
  }

  Base::size_ = new_size;
}

//maintains the values of existing positions, new ones are filled with <code> fill_value </code>
template<typename T,typename ST>
void Storage1D<T,ST>::resize(ST new_size, Storage1D<T,ST>::PassType fill_value)
{
  if (Base::data_ == 0) {

    //DEBUG
    // T* ptr = new T[new_size];
    // if (((size_t)((void*)ptr)) % 16 != 0) {
    //   WARNING << " pointer does not satisfy alignment boundary of 16!!: " << ptr << std::endl;
    // }
    // data_ = ptr;
    //END_DEBUG
    Base::data_ = new T[new_size];

    std::fill(Base::data_, Base::data_+new_size, fill_value); //fill and fill_n are of equal speed
  }
  else if (Base::size_ != new_size) {
    T* new_data = new T[new_size];

    if (new_size > Base::size_)
      std::fill_n(new_data+Base::size_,new_size-Base::size_,fill_value);

    const ST size = std::min(Base::size_,new_size);

    Makros::unified_move_assign(new_data, Base::data_, size);

    // for (size_t i=0; i < size; i++)
    // new_data[i] = data_[i];

    delete[] Base::data_;
    Base::data_ = new_data;
  }

  Base::size_ = new_size;
}

//all elements are undefined after this operation
template<typename T,typename ST>
void Storage1D<T,ST>::resize_dirty(ST new_size) 
{
  if (Base::size_ != new_size) {
    if (Base::data_ != 0)
      delete[] Base::data_;

    //DEBUG
    // T* ptr = new T[new_size];
    // if (((size_t)((void*)ptr)) % 16 != 0) {
    //   WARNING << " pointer does not satisfy alignment boundary of 16!!: " << ptr << std::endl;
    // }
    // data_ = ptr;
    //END_DEBUG
    Base::data_ = new T[new_size];
  }
  Base::size_ = new_size;
}

template<typename T,typename ST>
void Storage1D<T,ST>::swap(Storage1D<T,ST>& toSwap) noexcept
{
  std::swap(Base::data_, toSwap.data_);
  std::swap(Base::size_, toSwap.size_);
}

/******** implementation of NamedStorage1D ***************/

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D() : Storage1D<T,ST>(), name_("yyy") {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(std::string name) : Storage1D<T,ST>(), name_(name) {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(ST size, std::string name) : Storage1D<T,ST>(size), name_(name) {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(ST size, T default_value, std::string name) : Storage1D<T,ST>(size,default_value), name_(name) {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(const NamedStorage1D<T,ST>& toCopy) : Storage1D<T,ST>(toCopy), name_("yyy") {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(NamedStorage1D<T,ST>&& toTake) : Storage1D<T,ST>(toTake), name_("yyy") {}

template<typename T,typename ST>
/*virtual*/ const std::string& NamedStorage1D<T,ST>::name() const
{
  return name_;
}

template<typename T,typename ST>
inline void NamedStorage1D<T,ST>::operator=(const Storage1D<T,ST>& toCopy)
{
  Storage1D<T,ST>::operator=(toCopy);
}

template<typename T,typename ST>
inline void NamedStorage1D<T,ST>::operator=(Storage1D<T,ST>&& toTake)
{
  Storage1D<T,ST>::operator=(toTake);
}

//NOTE: the name is NOT copied
template<typename T,typename ST>
inline void NamedStorage1D<T,ST>::operator=(const NamedStorage1D<T,ST>& toCopy)
{
  Storage1D<T,ST>::operator=(static_cast<const Storage1D<T,ST>&>(toCopy));
}

//NOTE: the name is NOT taken
template<typename T,typename ST>
inline void NamedStorage1D<T,ST>::operator=(NamedStorage1D<T,ST>&& toTake)
{
  Storage1D<T,ST>::operator=(static_cast<Storage1D<T,ST>&&>(toTake));
}

template<typename T,typename ST>
std::ostream& operator<<(std::ostream& s, const Storage1D<T,ST>& v)
{
  s << "[ ";
  for (int i=0; i < ((int) v.size()) - 1; i++)
    s << v[i] << ",";
  if (v.size() > 0)
    s << v[v.size()-1];
  s << " ]";

  return s;
}

template<typename T,typename ST>
inline bool operator==(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2)
{
  if (v1.size() != v2.size())
    return false;

  if (std::is_trivially_copyable<T>::value) {
    return Routines::equals(v1.direct_access(), v2.direct_access(), v1.size());
  }
  else {
    for (ST k=0; k < v1.size(); k++) {
      if (v1[k] != v2[k])
        return false;
    }
    return true;
  }
}

template<typename T,typename ST>
bool operator!=(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2)
{
  return !operator==(v1,v2);
}

template<typename T,typename ST>
bool operator<(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2)
{
  for (ST k=0; k < std::min(v1.size(),v2.size()); k++) {
    if (v1[k] != v2[k])
      return (v1[k] < v2[k]);
  }

  return (v1.size() < v2.size());
}

template<typename T,typename ST>
bool operator<=(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2)
{
  for (ST k=0; k < std::min(v1.size(),v2.size()); k++) {
    if (v1[k] != v2[k])
      return (v1[k] < v2[k]);
  }

  return (v1.size() <= v2.size());
}

template<typename T,typename ST>
bool operator>(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2)
{
  return !operator<=(v1,v2);
}

template<typename T,typename ST>
bool operator>=(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2)
{
  return !operator<(v1,v2);
}

#endif
