/*-*-c++-*-*/
/*** first version written by Thomas Schoenemann as a private person without employment, September 2009 ***/
/*** much refined by Thomas Schoenemann  at Lund University, Sweden, the University of Pisa, Italy, ***
 *** and the University of Düsseldorf, Germany 2010 - 2012 **/
/*** if you desire the checked version, make sure your compiler defines the option SAFE_MODE on the command line ***/


#ifndef STORAGE1D_HH
#define STORAGE1D_HH

#include "makros.hh"
#include "storage_base.hh"
#include <cstring>

template<typename T>
class SwapOp {
public:
  
  void operator()(T& val1, T& val2) const {
    std::swap(val1, val2);
  }
};

template<typename T>
class SpecialSwapOp {
public:
  
  void operator()(T& val1, T& val2) const
  {
    val1.swap(val2);
  }
};

template<typename T, typename ST=size_t>
class Storage1D : public StorageBase<T,ST> {
public:

  typedef StorageBase<T,ST> Base;

  Storage1D();

  Storage1D(ST size);

  Storage1D(ST size, const T default_value);

  //copy constructor
  Storage1D(const Storage1D<T,ST>& toCopy);

  ~Storage1D();

  virtual const std::string& name() const;

  inline const T& operator[](ST i) const;

  inline T& operator[](ST i);

  void operator=(const Storage1D<T,ST>& toCopy);

#ifdef SAFE_MODE
  //for some reason g++ allows to assign an object of type T, but this does NOT produce the effect one would expect
  // => define this operator in safe mode, only to check that such an assignment is not made
  void operator=(const T& invalid_object);
#endif

  T back() const;
  
  T& back();

  //maintains the values of existing positions, new ones are undefined
  void resize(ST new_size);
  
  //maintains the values of existing positions, new ones are undefined. Swapping may be faster if e.g. T is std::vector or Storage1D
  template<class swap_op = SwapOp<T>>
  void resize_swap(ST newsize, swap_op op);

  //maintains the values of exisitng positions, new ones are filled with <code> fill_value </code>
  void resize(ST new_size, const T fill_value);

  //all elements are undefined after this operation
  void resize_dirty(ST new_size);

  inline void range_set_constant(const T constant, ST start, ST length);
  
  void swap(Storage1D<T,ST>& toSwap);
  
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

  virtual const std::string& name() const;

  inline void operator=(const Storage1D<T,ST>& toCopy);

  //NOTE: the name is NOT copied
  inline void operator=(const NamedStorage1D<T,ST>& toCopy);

protected:
  std::string name_;
};

template<typename T, typename ST>
std::ostream& operator<<(std::ostream& s, const Storage1D<T,ST>& v);

template<typename T, typename ST>
bool operator==(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

template<typename T, typename ST>
bool operator!=(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2);

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

/***********************/

//this class is meant to replace std::vector with its push_back() functionality.
// It has slightly less functionality, though. E.g. erase() is not available.
template<typename T, typename ST=size_t>
class FlexibleStorage1D {
public:

  FlexibleStorage1D();

  FlexibleStorage1D(ST reserved_size);

  //copy constructor
  FlexibleStorage1D(const FlexibleStorage1D<T,ST>& toCopy);

  ~FlexibleStorage1D();

  virtual const std::string& name() const;

  inline T& operator[](ST i) const;

  void resize(ST size, bool exact_fit = false);
  
  void shrink(ST size);
  
  void shrink_by(ST reduction);
  
  void reserve(ST size);

  //will not free memory
  void clear();

  void set_constant(T val);

  void operator=(const FlexibleStorage1D<T,ST>& toCopy);

  inline T back() const;

  inline T& back();

  ST append(T val);  
  
  //shortcut when you are sure the allocated memory suffices
  inline void append_trusting(T val);

  inline void push_back(T val);

  void append(Storage1D<T,ST>& toAppend);

  void append(FlexibleStorage1D<T,ST>& toAppend);

  inline ST size() const;

  inline ST reserved_size() const;

  T* direct_access();

  const T* direct_access() const;

  void swap(FlexibleStorage1D<T,ST>& toSwap);

protected:

  T* data_;
  ST size_;
  ST reserved_size_;
  static const std::string flex_stor1D_name_;
};

template<typename T, typename ST>
std::ostream& operator<<(std::ostream& s, const FlexibleStorage1D<T,ST>& v);

template<typename T, typename ST>
bool operator==(const FlexibleStorage1D<T,ST>& v1, const FlexibleStorage1D<T,ST>& v2);

template<typename T, typename ST>
bool operator!=(const FlexibleStorage1D<T,ST>& v1, const FlexibleStorage1D<T,ST>& v2);

template<typename T, typename ST=size_t>
class NamedFlexibleStorage1D : public FlexibleStorage1D<T,ST> {
public:

  NamedFlexibleStorage1D();

  NamedFlexibleStorage1D(const std::string& name);

  NamedFlexibleStorage1D(ST reserved_size, const std::string& name);

  //copy constructors
  NamedFlexibleStorage1D(const NamedFlexibleStorage1D<T,ST>& toCopy);

  NamedFlexibleStorage1D(const FlexibleStorage1D<T,ST>& toCopy);

  virtual const std::string& name() const;

  //operators
  void operator=(const NamedFlexibleStorage1D<T,ST>& toCopy);

  void operator=(const FlexibleStorage1D<T,ST>& toCopy);

protected:
  std::string name_;
};

/********************************************** implementation ************************************/

/******* implementation of Storage1D *********/

template<typename T,typename ST>
/*static*/ const std::string Storage1D<T,ST>::stor1D_name_ = "unnamed 1Dstorage";

template<typename T,typename ST> 
Storage1D<T,ST>::Storage1D() : StorageBase<T,ST>() {}

template<typename T,typename ST> 
Storage1D<T,ST>::Storage1D(ST size) : StorageBase<T,ST>(size)
{
}

template<typename T,typename ST> 
Storage1D<T,ST>::Storage1D(ST size, const T default_value) : StorageBase<T,ST>(size, default_value)
{
}

//copy constructor
template<typename T,typename ST> 
Storage1D<T,ST>::Storage1D(const Storage1D<T,ST>& toCopy) : StorageBase<T,ST>(toCopy.size())
{
  Makros::unified_assign(Base::data_, toCopy.direct_access(), Base::size_);
}

template<typename T,typename ST>
inline void Storage1D<T,ST>::range_set_constant(const T constant, ST start, ST length)
{
  assert(start+length <= Base::size_);

  std::fill_n(Base::data_+start,length,constant); //experimental result: fill_n is usually faster
}

template<typename T,typename ST> Storage1D<T,ST>::~Storage1D()
{
}

template<typename T,typename ST>
/*virtual*/ const std::string& Storage1D<T,ST>::name() const
{
  return Storage1D::stor1D_name_;
}

template<typename T,typename ST>
inline const T& Storage1D<T,ST>::operator[](ST i) const
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
inline T& Storage1D<T,ST>::operator[](ST i)
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
T Storage1D<T,ST>::back() const
{
  assert(Base::size_ > 0);
  return Base::data_[Base::size_-1];
}

template<typename T,typename ST>
T& Storage1D<T,ST>::back()
{
  assert(Base::size_ > 0);
  return Base::data_[Base::size_-1];  
}

template<typename T,typename ST>
void Storage1D<T,ST>::operator=(const Storage1D<T,ST>& toCopy)
{
  if (Base::size_ != toCopy.size()) {

    if (Base::data_ != 0)
      delete[] Base::data_;

    Base::size_ = toCopy.size();
    Base::data_ = new T[Base::size_];
  }

  Makros::unified_assign(Base::data_, toCopy.direct_access(), Base::size_);

  // const ST size = size_;
  // for (ST i=0; i < size; i++) {
    // data_[i] = toCopy.direct_access(i);
  // }

  //this is faster for basic types but it fails for complex types where e.g. arrays have to be copied
  //memcpy(data_,toCopy.direct_access(),size_*sizeof(T));
}

#ifdef SAFE_MODE
//for some reason g++ allows to assign an object of type T, but this does NOT produce the effect one would expect
// => define this operator in safe mode, only to check that such an assignment is not made
template<typename T,typename ST>
void Storage1D<T,ST>::operator=(const T& invalid_object)
{
  INTERNAL_ERROR << "assignment of an atomic entity to Storage1D \"" << this->name() << "\" of type "
                 << Makros::Typename<T>()
                 << " with " << Base::size_ << " elements. exiting." << std::endl;
}
#endif

//maintains the values of existing positions, new ones are undefined
template<typename T,typename ST>
void Storage1D<T,ST>::resize(ST new_size)
{
  if (Base::data_ == 0) {
    //DEBUG
    // T* ptr = new T[new_size];
    // if (((size_t)((void*)ptr)) % 16 != 0) {
    //   WARNING << " pointer does not satisfy alignment boundary of 16!!: " << ptr << std::endl;
    //   std::cerr << "type "  << Makros::Typename<T>() << std::endl;
    // }
    // data_ = ptr;
    //END_DEBUG
    Base::data_ = new T[new_size];
  }
  else if (Base::size_ != new_size) {

    //DEBUG
    // T* ptr = new T[new_size];
    // if (((size_t)((void*)ptr)) % 16 != 0) {
    //   WARNING << " pointer does not satisfy alignment boundary of 16!!: " << ptr << std::endl;
    // }
    // T_A16* new_data = ptr;
    //END_DEBUG
    T* new_data = new T[new_size];

    const ST size = std::min(Base::size_,new_size);

    Makros::unified_assign(new_data, Base::data_, size);

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
void Storage1D<T,ST>::resize(ST new_size, const T fill_value)
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
    
    Makros::unified_assign(new_data, Base::data_, size);
   
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
void Storage1D<T,ST>::swap(Storage1D<T,ST>& toSwap) 
{
  std::swap(Base::data_, toSwap.data_);
  std::swap(Base::size_, toSwap.size_);
}

/******** implementation of NamedStorage1D ***************/

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D() : Storage1D<T,ST>(), name_("yyy") {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(std::string name) : Storage1D<T,ST>(), name_(name) {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(ST size, std::string name) : Storage1D<T,ST>(size), name_(name) {}

template<typename T,typename ST> NamedStorage1D<T,ST>::NamedStorage1D(ST size, T default_value, std::string name) :
  Storage1D<T,ST>(size,default_value), name_(name) {}

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

//NOTE: the name is NOT copied
template<typename T,typename ST>
inline void NamedStorage1D<T,ST>::operator=(const NamedStorage1D<T,ST>& toCopy)
{
  Storage1D<T,ST>::operator=(static_cast<const Storage1D<T,ST>&>(toCopy));
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
bool operator==(const Storage1D<T,ST>& v1, const Storage1D<T,ST>& v2)
{
  if (v1.size() != v2.size())
    return false;

  for (ST k=0; k < v1.size(); k++) {
    if (v1[k] != v2[k])
      return false;
  }
  return true;
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

/******* implementation of FlexibleStorage1D *********/

template<typename T, typename ST>
/*static*/ const std::string FlexibleStorage1D<T,ST>::flex_stor1D_name_ = "unnamed flexible 1Dstorage";

template<typename T, typename ST> 
FlexibleStorage1D<T,ST>::FlexibleStorage1D() : size_(0)
{
  reserved_size_ = 4;
  data_ = new T[reserved_size_];
}

template<typename T, typename ST> 
FlexibleStorage1D<T,ST>::FlexibleStorage1D(ST reserved_size)  : size_(0), reserved_size_(reserved_size)
{
  data_ = new T[reserved_size_];
}

//copy constructor
template<typename T, typename ST> 
FlexibleStorage1D<T,ST>::FlexibleStorage1D(const FlexibleStorage1D<T,ST>& toCopy)
{
  size_ = toCopy.size();
  reserved_size_ = toCopy.reserved_size();

  data_ = new T[reserved_size_];
  
  Makros::unified_assign(data_, toCopy.direct_access(), size_);
  
  //for (uint k=0; k < toCopy.size(); k++)
  //  data_[k] = toCopy[k];
}

template<typename T, typename ST> 
void FlexibleStorage1D<T,ST>::swap(FlexibleStorage1D<T,ST>& toSwap)
{
  std::swap(data_,toSwap.data_);
  std::swap(size_,toSwap.size_);
  std::swap(reserved_size_,toSwap.reserved_size_);
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::operator=(const FlexibleStorage1D<T,ST>& toCopy)
{
  uint new_res = toCopy.reserved_size();
  if (new_res != reserved_size_) {
    reserved_size_ = new_res;

    if (data_ != 0)
      delete[] data_;
    data_ = new T[reserved_size_];
  }

  size_ = toCopy.size();

  Makros::unified_assign(data_, toCopy.direct_access(), size_);

  //for (uint k=0; k < size_; k++)
  //  data_[k] = toCopy[k];
}

template<typename T, typename ST>
/*virtual*/ const std::string& FlexibleStorage1D<T,ST>::name() const
{
  return flex_stor1D_name_;
}

template<typename T, typename ST> 
FlexibleStorage1D<T,ST>::~FlexibleStorage1D()
{
  delete[] data_;
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::set_constant(T val)
{
  for (ST k=0; k < size_; k++)
    data_[k] = val;
}

template<typename T, typename ST>
inline ST FlexibleStorage1D<T,ST>::size() const
{
  return size_;
}

template<typename T, typename ST>
inline ST FlexibleStorage1D<T,ST>::reserved_size() const
{
  return reserved_size_;
}

template<typename T, typename ST>
inline T FlexibleStorage1D<T,ST>::back() const
{
  assert(size_ > 0);
  assert(data_ != 0);
  return data_[size_-1];
}

template<typename T, typename ST>
inline T& FlexibleStorage1D<T,ST>::back()
{
  assert(size_ > 0);
  assert(data_ != 0);
  return data_[size_-1];
}

template<typename T, typename ST>
ST FlexibleStorage1D<T,ST>::append(T val)
{
  if (size_ == reserved_size_) {

    reserved_size_ = size_t(1.2 * reserved_size_) + 4;

    T* new_data = new T[reserved_size_];
    
    Makros::unified_assign(new_data, data_, size_);
    
    //for (uint k=0; k < size_; k++)
    //  new_data[k] = data_[k];

    delete[] data_;
    data_ = new_data;
  }

  const ST k = size_;
  data_[k] = val;

  size_++;

  return k;
}

//shortcut when you are sure the allocated memory suffices
template<typename T, typename ST>
inline void FlexibleStorage1D<T,ST>::append_trusting(T val)
{
  assert(size_ < reserved_size_);
  data_[size_] = val;
  size_++;
}

template<typename T, typename ST>
inline void FlexibleStorage1D<T,ST>::push_back(T val) {
  append(val); 
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::append(Storage1D<T,ST>& toAppend)
{
  if (reserved_size_ < size_ + toAppend.size()) {

    reserved_size_ = size_ + toAppend.size() + 2;

    T* new_data = new T[reserved_size_];
    
    Makros::unified_assign(new_data, data_, size_);
    
    //for (uint k=0; k < size_; k++)
    //  new_data[k] = data_[k];

    delete[] data_;
    data_ = new_data;
  }

  for (ST k=0; k < toAppend.size(); k++) {
    data_[size_] = toAppend[k];
    size_++;
  }
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::append(FlexibleStorage1D<T,ST>& toAppend)
{
  if (reserved_size_ < size_ + toAppend.size()) {

    reserved_size_ = size_ + toAppend.size() + 2;

    T* new_data = new T[reserved_size_];
    
    Makros::unified_assign(new_data, data_, size_);
    
    //for (uint k=0; k < size_; k++)
    //  new_data[k] = data_[k];

    delete[] data_;
    data_ = new_data;
  }

  for (ST k=0; k < toAppend.size(); k++) {
    data_[size_] = toAppend[k];
    size_++;
  }
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::resize(ST size, bool exact_fit)
{
  if (size > reserved_size_ || size < (reserved_size_ / 3) ) {

    reserved_size_ = size;
    T* new_data = new T[reserved_size_];
    
    Makros::unified_assign(new_data, data_, std::min(size_,size));
    
    //for (uint k=0; k < std::min(size_,size); k++)
    //  new_data[k] = data_[k];

    delete[] data_;
    data_ = new_data;
  }

  size_ = size;

  if (exact_fit && size_ != reserved_size_) {

    reserved_size_ = size_;
    T* new_data = new T[reserved_size_];
   
    Makros::unified_assign(new_data, data_, size_);
   
    //for (uint k=0; k < size_; k++)
    //  new_data[k] = data_[k];

    delete[] data_;
    data_ = new_data;
  }
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::shrink(ST size)
{
  assert(size <= size_);
  size_ = size;
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::shrink_by(ST reduction)
{
  assert(reduction <= size_);
  size_ -= reduction;
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::reserve(ST size) 
{
  if (size > size_ && size != reserved_size_) {
    
    reserved_size_ = size;
    T* new_data = new T[reserved_size_];
    
    Makros::unified_assign(new_data, data_, size_);
    
    delete[] data_;
    data_ = new_data;    
  }
}

template<typename T, typename ST>
void FlexibleStorage1D<T,ST>::clear()
{
  size_ = 0;
}

template<typename T, typename ST>
inline T& FlexibleStorage1D<T,ST>::operator[](ST i) const
{

#ifdef SAFE_MODE
  if (i >= size_) {
    INTERNAL_ERROR << "    invalid access on element " << i
                   << " for FlexibleStorage1D " <<  "\"" << this->name() << "\" of type "
                   //<< Makros::Typename<T>()
                   << typeid(T).name()
                   << " with " << size_ << " (valid) elements. exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  return data_[i];
}

template<typename T, typename ST>
T* FlexibleStorage1D<T,ST>::direct_access()
{
  return data_;
}

template<typename T, typename ST>
const T* FlexibleStorage1D<T,ST>::direct_access() const
{
  return data_;
}

template<typename T, typename ST>
std::ostream& operator<<(std::ostream& s, const FlexibleStorage1D<T,ST>& v)
{
  s << "[ ";
  for (int i=0; i < ((int) v.size()) - 1; i++)
    s << v[i] << ",";
  if (v.size() > 0)
    s << v[v.size()-1];
  s << " ]";

  return s;
}

template<typename T, typename ST>
bool operator==(const FlexibleStorage1D<T,ST>& v1, const FlexibleStorage1D<T,ST>& v2)
{
  if (v1.size() != v2.size())
    return false;

  for (ST k=0; k < v1.size(); k++) {
    if (v1[k] != v2[k])
      return false;
  }
  return true;
}

template<typename T, typename ST>
bool operator!=(const FlexibleStorage1D<T,ST>& v1, const FlexibleStorage1D<T,ST>& v2)
{
  if (v1.size() != v2.size())
    return true;

  for (ST k=0; k < v1.size(); k++) {
    if (v1[k] != v2[k])
      return true;
  }
  return false;
}

/***********************************/

template<typename T, typename ST> NamedFlexibleStorage1D<T,ST>::NamedFlexibleStorage1D() : name_("unfs1d") {}

template<typename T, typename ST> NamedFlexibleStorage1D<T,ST>::NamedFlexibleStorage1D(const std::string& name) : name_(name)
{
}

template<typename T, typename ST> NamedFlexibleStorage1D<T,ST>::NamedFlexibleStorage1D(ST reserved_size, const std::string& name) :
  FlexibleStorage1D<T,ST>(reserved_size), name_(name) {}

//Note: the name is NOT copied
template<typename T, typename ST> NamedFlexibleStorage1D<T,ST>::NamedFlexibleStorage1D(const NamedFlexibleStorage1D<T,ST>& toCopy) :
  FlexibleStorage1D<T,ST>(toCopy), name_("unfs1d")
{
}

template<typename T, typename ST> NamedFlexibleStorage1D<T,ST>::NamedFlexibleStorage1D(const FlexibleStorage1D<T,ST>& toCopy) :
  FlexibleStorage1D<T,ST>(toCopy), name_("unfs1d")
{
}

template<typename T, typename ST>
/*virtual*/ const std::string& NamedFlexibleStorage1D<T,ST>::name() const
{
  return name_;
}

template<typename T, typename ST>
void NamedFlexibleStorage1D<T,ST>::operator=(const NamedFlexibleStorage1D<T,ST>& toCopy)
{
  FlexibleStorage1D<T,ST>::operator=(toCopy);
}

template<typename T, typename ST>
void NamedFlexibleStorage1D<T,ST>::operator=(const FlexibleStorage1D<T,ST>& toCopy)
{
  FlexibleStorage1D<T,ST>::operator=(toCopy);
}

#endif
