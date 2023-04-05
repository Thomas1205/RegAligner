/*** written by Thomas Schoenemann as a private person without employment, September 2009 ***/

#ifndef VECTOR_HH
#define VECTOR_HH

#include "makros.hh"
#include "storage1D.hh"
#include "sorting.hh"
#include "routines.hh"
#include <numeric>

namespace Math1D {

  /**********************************************/
  /*************** unnamed vector ***************/
  /**********************************************/

  //instantiate this only with data types where new is guaranteed to return a 16-byte aligned pointer
  template<typename T,typename ST = size_t>
  class Vector : public ::Storage1D<T,ST> {
  public:

    using Base = Storage1D<T,ST>;

    //according to https://gcc.gnu.org/onlinedocs/gcc-7.2.0/gcc/Common-Type-Attributes.html#Common-Type-Attributes , alignment has to be expressed like this:
    typedef T T_A16 ALIGNED16;

    explicit Vector();

    explicit Vector(ST size);

    explicit Vector(ST size, const T default_value);

    Vector(const std::initializer_list<T>& init);

    //copy constructor
    Vector(const Vector<T,ST>& toCopy);

    //move constructor
    Vector(Vector<T,ST>&& toTake);

    ~Vector();

    Vector<T,ST>& operator=(const Vector<T,ST>& toCopy);

    Vector<T,ST>& operator=(Vector<T,ST>&& toTake);

    Vector<T,ST>& operator=(const std::initializer_list<T>& init);

    //redefining the operator[] methods because for basic types we can indicate 16-byte alignment
    inline const T& operator[](ST i) const noexcept;

    inline T& operator[](ST i) noexcept;

    //redefining the direct_access methods because for basic types we can indicate 16-byte alignment
    inline T* direct_access() noexcept FLAGALIGNED16;

    inline const T* direct_access() const noexcept FLAGALIGNED16;

    inline T& direct_access(ST i) noexcept;

    inline T direct_access(ST i) const noexcept;

    //redefining the set_constant method because for basic types we can indicate 16-byte alignment
    inline void set_constant(T constant) noexcept;

    inline T sum() const noexcept;

    inline T sum_abs() const noexcept;

    inline T range_sum(ST start, ST end) const noexcept;

    /*** maximal element ***/
    T max() const noexcept;

    /*** minimal element ***/
    T min() const noexcept;

    /*** maximal absolute element = l-infinity norm ***/
    T max_abs() const noexcept;

    inline void ensure_min(T lower_limit) noexcept;

    /*** L2-norm of the vector ***/
    inline double norm() const noexcept;

    inline T norm_T() const noexcept;

    inline double sqr_norm() const noexcept;

    /*** L1-norm of the vector ***/
    inline double norm_l1() const noexcept;

    inline void add_constant(const T addon) noexcept;

    //note: with g++-4.8.5 it is a lot faster to use set_constant(0.0)
    void set_zeros() noexcept;

    inline void add_vector_multiple(const Math1D::Vector<T,ST>& vec, const T alpha) noexcept;

    bool is_sorted() const noexcept;

    void operator+=(const Vector<T,ST>& v) noexcept;

    void operator-=(const Vector<T,ST>& v) noexcept;

    void operator*=(const T constant) noexcept;

    void elem_mul(const Vector<T,ST>& v) noexcept;

    void elem_div(const Vector<T,ST>& v) noexcept;

    virtual const std::string& name() const;

  protected:
    static const std::string vector_name_;
  };

  /**********************************************/
  /***************** named vector ****************/
  /**********************************************/
  template<typename T,typename ST = size_t>
  class NamedVector : public Vector<T,ST> {
  public:

    typedef Vector<T,ST> Base;

    NamedVector();

    NamedVector(std::string name);

    NamedVector(ST size, std::string name);

    NamedVector(ST size, T default_value, std::string name);

    ~NamedVector();

    void set_name(std::string new_name);

    virtual const std::string& name() const;

    inline void operator=(const Vector<T,ST>& v);

    //NOTE: the name is NOT copied
    inline void operator=(const NamedVector<T,ST>& v);


  protected:
    std::string name_;
  };

  //NOTE: dest can be the same as src1 or src2
  inline void go_in_neg_direction(Math1D::Vector<double>& dest, const Math1D::Vector<double>& src1, const Math1D::Vector<double>& src2, double alpha) noexcept
  {
    assert(dest.size() == src1.size());
    assert(dest.size() == src2.size());
    Routines::go_in_neg_direction(dest.direct_access(), dest.size(), src1.direct_access(), src2.direct_access(), alpha);
  }

  //NOTE: dest can be the same as src1 or src2
  inline void assign_weighted_combination(Math1D::Vector<double>& dest, double w1, const Math1D::Vector<double>& src1,
                                          double w2, const Math1D::Vector<double>& src2) noexcept
  {
    assert(dest.size() == src1.size());
    assert(dest.size() == src2.size());
    Routines::assign_weighted_combination(dest.direct_access(), dest.size(), w1, src1.direct_access(), w2, src2.direct_access());
  }

  /***********************************************/
  /*************** operators *********************/
  /***********************************************/
  template<typename T,typename ST>
  Vector<T,ST> operator+(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept;

  template<typename T,typename ST>
  Vector<T,ST> operator-(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept;

  //scalar product of two vectors
  template<typename T,typename ST>
  inline T operator%(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept;

  template<typename T,typename ST>
  Vector<T,ST> cross(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept;

  //streaming
  template<typename T,typename ST>
  std::ostream& operator<<(std::ostream& s, const Vector<T,ST>& v);
}

namespace Makros {

  template<typename T, typename ST>
  class Typename<Math1D::Vector<T,ST> > {
  public:

    std::string name() const
    {
      return "Math1D::Vector<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math1D::Vector<T> > {
  public:

    std::string name() const
    {
      return "Math1D::Vector<" + Makros::Typename<T>() + "> ";
    }
  };

  template<typename T, typename ST>
  class Typename<Math1D::NamedVector<T,ST> > {
  public:

    std::string name() const
    {
      return "Math1D::NamedVector<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };


  template<typename T>
  class Typename<Math1D::NamedVector<T> > {
  public:

    std::string name() const
    {
      return "Math1D::NamedVector<" + Makros::Typename<T>() + "> ";
    }
  };
}

/******************************************** implementation *****************************************************/

namespace Math1D {

  template<typename T,typename ST>
  /*static*/ const std::string Vector<T,ST>::vector_name_ = "unnamed vector";

  template<typename T,typename ST> Vector<T,ST>::Vector() : Storage1D<T,ST>() {}

  template<typename T,typename ST> Vector<T,ST>::Vector(ST size) : Storage1D<T,ST>(size) {}

  template<typename T,typename ST> Vector<T,ST>::Vector(ST size, const T default_value) : Storage1D<T,ST>(size)
  {
    set_constant(default_value);
  }

  template<typename T,typename ST> Vector<T,ST>::Vector(const std::initializer_list<T>& init) : Storage1D<T,ST>(init)
  {
  }

  //copy constructor
  template<typename T,typename ST> Vector<T,ST>::Vector(const Vector<T,ST>& toCopy) : Storage1D<T,ST>(toCopy) {}

  //move constructor
  template<typename T,typename ST> Vector<T,ST>::Vector(Vector<T,ST>&& toTake) : Storage1D<T,ST>(toTake) {}

  template<typename T,typename ST>
  Vector<T,ST>& Vector<T,ST>::operator=(Vector<T,ST>&& toTake)
  {
    StorageBase<T,ST>::operator=(toTake);
    return *this;
  }

  template<typename T,typename ST>
  Vector<T,ST>& Vector<T,ST>::operator=(const Vector<T,ST>& toCopy)
  {
    StorageBase<T,ST>::operator=(toCopy);
    return *this;
  }

  template<typename T,typename ST>
  Vector<T,ST>& Vector<T,ST>::operator=(const std::initializer_list<T>& init)
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

  template<typename T,typename ST> Vector<T,ST>::~Vector() {}

  template<typename T,typename ST>
  inline T Vector<T,ST>::sum() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);

    //at least with g++ accumulate is faster
    return std::accumulate(data,data+size,(T)0);

    // T result = (T) 0;
    // for (ST i=0; i < size; i++) {
    //   result += data[i];
    // }

    // return result;
  }

  template<typename T,typename ST>
  inline T Vector<T,ST>::sum_abs() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);

    T result = (T) 0;
    for (ST i=0; i < size; i++)
      result += Makros::abs<T>(data[i]);

    return result;
  }

  template<typename T,typename ST>
  inline T Vector<T,ST>::range_sum(ST start, ST end) const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);
    assert(start <= end);
    assert(start < size);
    assert(end <= size);

    return std::accumulate(data+start,data+end,(T)0);

    // T result = 0.0;
    // for (ST i=start; i < end; i++) {
    //   result += data[i];
    // }

    // return result;
  }

  template<typename T,typename ST>
  void Vector<T,ST>::set_zeros() noexcept
  {
    memset(Base::data_,0,Base::size_*sizeof(T));
  }

  template<typename T,typename ST>
  inline T* Vector<T,ST>::direct_access() noexcept //not allowed to repeat FLAGALIGNED16 in definition
  {
    T_A16* data = Base::data_;
    assertAligned16(data);
    return data;
  }

  template<typename T,typename ST>
  inline const T* Vector<T,ST>::direct_access() const noexcept //not allowed to repeat FLAGALIGNED16 in definition
  {
    const T_A16* data = Base::data_;
    assertAligned16(data);
    return data;
  }

  template<typename T,typename ST>
  inline T& Vector<T,ST>::direct_access(ST i) noexcept
  {
    T_A16* data = Base::data_;
    assertAligned16(data);

    return data[i];
  }

  template<typename T,typename ST>
  inline T Vector<T,ST>::direct_access(ST i) const noexcept
  {
    const T_A16* data = Base::data_;
    assertAligned16(data);

    return data[i];
  }

  //redefining the set_constant method because for basic types we can indicate 16-byte alignment
  template<typename T,typename ST>
  inline void Vector<T,ST>::set_constant(const T constant) noexcept
  {
    T_A16* data = Base::data_;
    const ST size = Base::size_;
    assertAligned16(data);

    std::fill_n(data,size,constant); //experimental result: fill_n is usually faster
  }

  /*** maximal element ***/
  template<typename T,typename ST>
  inline T Vector<T,ST>::max() const noexcept
  {
    const ST size = Base::size_;

    if (size > 0) {
      // g++ 4.8.5 uses avx instructions for this
      const T_A16* data = Base::data_;
      assertAligned16(data);

      return *std::max_element(data, data + size);
    }
    else
      return std::numeric_limits<T>::min();
  }

  template<>
  float Vector<float>::max() const noexcept;

  template<typename T,typename ST>
  T Vector<T,ST>::min() const noexcept
  {
    const ST size = Base::size_;

    if (size > 0) {
      // g++ 4.8.5 uses avx instructions for this
      const T_A16* data = Base::data_;
      assertAligned16(data);

      return *std::min_element(data, data + size);
    }
    else
      return std::numeric_limits<T>::max();
  }

  template<>
  float Vector<float>::min() const noexcept;

  /*** maximal absolute element = l-infinity norm ***/
  template<typename T,typename ST>
  T Vector<T,ST>::max_abs() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    T maxel = (T) 0;
    for (ST i=0; i < size; i++) {
      const T candidate = Makros::abs<T>(data[i]);
      maxel = std::max(maxel,candidate);
    }

    return maxel;
  }

  template<typename T,typename ST>
  inline void Vector<T,ST>::ensure_min(T lower_limit) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;

    for (ST i=0; i < size; i++)
      data[i] = std::max(lower_limit,data[i]);
  }

  /*** L2-norm of the vector ***/
  template<typename T,typename ST>
  inline double Vector<T,ST>::norm() const noexcept
  {
    const ST size = Base::size_;

    const T_A16* data = Base::data_;
    assertAligned16(data);

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      const double cur = (double) data[i];
      result += cur*cur;
    }

    return sqrt(result);
  }

  /*** L2-norm of the vector ***/
  template<typename T,typename ST>
  inline T Vector<T,ST>::norm_T() const noexcept
  {
    const ST size = Base::size_;

    const T_A16* data = Base::data_;
    assertAligned16(data);

    T result = (T) 0;
    for (ST i=0; i < size; i++) {
      const T cur = data[i];
      result += cur*cur;
    }

    return Makros::sqrt<T>(result);
  }

  template<typename T,typename ST>
  inline double Vector<T,ST>::sqr_norm() const noexcept
  {
    const ST size = Base::size_;

    const T_A16* data = Base::data_;
    assertAligned16(data);

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      const double cur = (double) data[i];
      result += cur*cur;
    }

    return result;
  }

  /*** L1-norm of the vector ***/
  template<typename T,typename ST>
  inline double Vector<T,ST>::norm_l1() const noexcept
  {
    const ST size = Base::size_;

    const T_A16* data = Base::data_;
    assertAligned16(data);

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      result += Makros::abs<T>(data[i]);
    }

    return result;
  }

  template<typename T,typename ST>
  inline void Vector<T,ST>::add_constant(const T addon) noexcept
  {
    T_A16* data = Base::data_;
    assertAligned16(data);

    const ST size = Base::size_;
    for (ST i=0; i < size; i++)
      data[i] += addon;
  }

  template<typename T,typename ST>
  bool Vector<T,ST>::is_sorted() const noexcept
  {
    return ::is_sorted(Base::data_, Base::size_);
  }

  template<typename T,typename ST>
  inline void Vector<T,ST>::add_vector_multiple(const Math1D::Vector<T,ST>& v, const T alpha) noexcept
  {
    const ST size = Base::size_;

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (v.size() != size) {
      INTERNAL_ERROR << "   cannot add multiple of vector \"" << v.name() << "\" to vector \""
                     << this->name() << "\":" << std::endl
                     << "   sizes " << v.size() << " and " << this->size() << " mismatch. Exiting..."
                     << std::endl;
      exit(0);
    }
#endif

    ST i;

    // for (i=0; i < size; i++)
    //   Base::data_[i] += alpha * v.direct_access(i);

    T_A16* attr_restrict dptr = Base::data_;
    const T_A16* attr_restrict vptr = v.direct_access();

    assertAligned16(dptr);
    assertAligned16(vptr);

    for (i=0; i < size; i++)
      dptr[i] += alpha * vptr[i];
  }

  template<>
  inline void Vector<double>::add_vector_multiple(const Math1D::Vector<double>& v, const double alpha) noexcept
  {
    const size_t size = Base::size_;

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (v.size() != size) {
      INTERNAL_ERROR << "   cannot add multiple of vector \"" << v.name() << "\" to vector \""
                     << this->name() << "\":" << std::endl
                     << "   sizes " << v.size() << " and " << this->size() << " mismatch. Exiting..."
                     << std::endl;
      exit(0);
    }
#endif

    Routines::array_add_multiple(Base::data_, size, alpha, v.direct_access());
  }

  template<typename T,typename ST>
  void Vector<T,ST>::operator+=(const Vector<T,ST>& v) noexcept
  {
    const ST size = Base::size_;

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (v.size() != size) {
      INTERNAL_ERROR << "   cannot add vector \"" << v.name() << "\" to vector \""
                     << this->name() << "\":" << std::endl
                     << "   sizes " << v.size() << " and " << this->size() << " mismatch. Exiting..."
                     << std::endl;
      exit(0);
    }
#endif

    ST i;

    // for (i=0; i < size; i++)
    //   Base::data_[i] += v.direct_access(i);

    T_A16* attr_restrict dptr = Base::data_;
    const T_A16* attr_restrict vptr = v.direct_access();

    assertAligned16(dptr);
    assertAligned16(vptr);

    for (i=0; i < size; i++)
      dptr[i] += vptr[i];
  }

  template<typename T,typename ST>
  void Vector<T,ST>::operator-=(const Vector<T,ST>& v) noexcept
  {
    const ST size = Base::size_;

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (v.size() != size) {
      INTERNAL_ERROR << "   cannot subtract vector \"" << v.name() << "\" from vector \""
                     << this->name() << "\":" << std::endl
                     << "   sizes " << v.size() << " and " << this->size() << " mismatch. Exiting..."
                     << std::endl;
      exit(0);
    }
#endif

    ST i;
    // for (i=0; i < size; i++)
    //   Base::data_[i] -= v.direct_access(i);

    T_A16* attr_restrict dptr = Base::data_;
    const T_A16* attr_restrict vptr = v.direct_access();

    for (i=0; i < size; i++)
      dptr[i] -= vptr[i];
  }

  template<typename T,typename ST>
  void Vector<T,ST>::operator*=(const T constant) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;

    assertAligned16(data);

    ST i;
    for (i=0; i < size; i++)
      data[i] *= constant;
  }

  template<>
  void Vector<float>::operator*=(const float scalar) noexcept;

  template<>
  void Vector<double>::operator*=(const double scalar) noexcept;

  template<typename T,typename ST>
  /*virtual*/ const std::string& Vector<T,ST>::name() const
  {
    return Vector<T,ST>::vector_name_;
  }

  template<typename T,typename ST>
  void Vector<T,ST>::elem_mul(const Vector<T,ST>& v) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;
    const T_A16* vdata = v.direct_access();

    assert(size == v.size());
    for (ST i = 0; i < size; i++)
      data[i] *= vdata[i];
  }

  template<typename T,typename ST>
  void Vector<T,ST>::elem_div(const Vector<T,ST>& v) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;
    const T_A16* vdata = v.direct_access();

    assert(size == v.size());
    for (ST i = 0; i < size; i++)
      data[i] /= vdata[i];
  }

  /************** implementation of NamedVector **********/

  template<typename T,typename ST> NamedVector<T,ST>::NamedVector() : Vector<T,ST>(), name_("yyy") {}

  template<typename T,typename ST> NamedVector<T,ST>::NamedVector(std::string name) : Vector<T,ST>(), name_(name) {}

  template<typename T,typename ST> NamedVector<T,ST>::NamedVector(ST size, std::string name) : Vector<T,ST>(size), name_(name) {}

  template<typename T,typename ST> NamedVector<T,ST>::NamedVector(ST size, T default_value, std::string name) :
    Vector<T,ST>(size,default_value), name_(name) {}

  template<typename T,typename ST> NamedVector<T,ST>::~NamedVector() {}

  template<typename T,typename ST>
  void NamedVector<T,ST>::set_name(std::string new_name)
  {
    name_ = new_name;
  }

  template<typename T,typename ST>
  /*virtual*/ const std::string& NamedVector<T,ST>::name() const
  {
    return name_;
  }

  template<typename T,typename ST>
  inline void NamedVector<T,ST>::operator=(const Vector<T,ST>& v)
  {
    Base::operator=(v);
  }

  //NOTE: the name is NOT copied
  template<typename T,typename ST>
  inline void NamedVector<T,ST>::operator=(const NamedVector<T,ST>& v)
  {
    Base::operator=(v);
  }

  /************** implementation of stand-alone routines **********************/

  template<typename T,typename ST>
  Vector<T,ST> operator+(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept
  {
    typedef T ALIGNED16 T_A16;

    const ST size = v1.size();

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (size != v2.size()) {
      INTERNAL_ERROR << "     cannot add vectors \"" << v1.name() << "\" and \""
                     << v2.name() << "\":" << std::endl
                     << "    sizes " << v1.size() << " and " << v2.size() << " mismatch. Exiting..."
                     << std::endl;
      exit(1);
    }
#endif

    Vector<T,ST> result(size);
    ST i;
    // for (i = 0; i < size; i++)
    //   result.direct_access(i) = v1.direct_access(i) + v2.direct_access(i);

    T_A16* attr_restrict resptr = result.direct_access();
    const T_A16* attr_restrict v1ptr = v1.direct_access();
    const T_A16* attr_restrict v2ptr = v2.direct_access();

    assertAligned16(resptr);
    assertAligned16(v1ptr);
    assertAligned16(v2ptr);

    for (i = 0; i < size; i++)
      resptr[i] = v1ptr[i] + v2ptr[i];

    return result;
  }

  template<typename T,typename ST>
  Vector<T,ST> operator-(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept
  {
    typedef T ALIGNED16 T_A16;

    const ST size = v1.size();

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (size != v2.size()) {
      INTERNAL_ERROR << "     cannot subtract vector \"" << v2.name() << "\" from \""
                     << v1.name() << "\":" << std::endl
                     << "    sizes " << v2.size() << " and " << v1.size() << " mismatch. Exiting..."
                     << std::endl;
      exit(1);
    }
#endif

    Vector<T,ST> result(size);
    ST i;

    // for (i = 0; i < size; i++)
    //   result.direct_access(i) = v1.direct_access(i) - v2.direct_access(i);

    T_A16* attr_restrict resptr = result.direct_access();
    const T_A16* attr_restrict v1ptr = v1.direct_access();
    const T_A16* attr_restrict v2ptr = v2.direct_access();

    assertAligned16(resptr);
    assertAligned16(v1ptr);
    assertAligned16(v2ptr);

    for (i = 0; i < size; i++)
      resptr[i] = v1ptr[i] - v2ptr[i];

    return result;
  }

  template<typename T,typename ST>
  inline T operator%(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept
  {
    typedef T ALIGNED16 T_A16;

    const ST size = v1.size();

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (size != v2.size()) {
      INTERNAL_ERROR << "     cannot compute scalar product of vectors \""
                     << v1.name() << "\" and \"" << v2.name() << "\":" << std::endl
                     << "      sizes " << v1.size() << " and " << v2.size() << " mismatch. exiting."
                     << std::endl;
      exit(1);
    }
#endif

    const attr_restrict T_A16* data1 = v1.direct_access();
    const attr_restrict T_A16* data2 = v2.direct_access();

    assertAligned16(data1);
    assertAligned16(data2);

    //g++ uses packed fused multiply-add, but ignores the alignment information
    return Routines::dotprod(data1,data2,size);
    //return std::inner_product(data1,data1+size,data2, (T) 0);
  }

  template<typename T,typename ST>
  std::ostream& operator<<(std::ostream& s, const Vector<T,ST>& v)
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
  Vector<T,ST> cross(const Vector<T,ST>& v1, const Vector<T,ST>& v2) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (v1.size() != 3 || v2.size() != 3) {
      INTERNAL_ERROR << "      the cross product is only defined for vectors of size 3." << std::endl
                     << "                  here applied for vectors of size " << v1.size() << " and "
                     << v2.size() << ". exiting." << std::endl;
      exit(1);
    }
#endif

    Vector<T,ST> result(3);
    result[0] = v1[1]*v2[2] - v1[2]*v2[1];
    result[1] = v1[2]*v2[0] - v1[0]*v2[2];
    result[2] = v1[0]*v2[1] - v1[1]*v2[0];

    return result;
  }


}//end of namespace Math1D


template<typename T,typename ST>
inline const T& Math1D::Vector<T,ST>::operator[](ST i) const noexcept
{
#ifdef SAFE_MODE
  if (i >= Base::size_) {

    INTERNAL_ERROR << "    invalid const access on element " << i
                   << " for Math1D::Vector " <<  "\"" << this->name() << "\" of type "
                   << Makros::Typename<T>().name()
                   //<< typeid(T).name()
                   << " with " << Base::size_ << " elements. exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  const T_A16* data = Base::data_;
  return data[i];
}

template<typename T,typename ST>
inline T& Math1D::Vector<T,ST>::operator[](ST i) noexcept
{
#ifdef SAFE_MODE
  if (i >= Base::size_) {

    INTERNAL_ERROR << "    invalid access on element " << i
                   << " for Math1D::Vector " <<  "\"" << this->name() << "\" of type "
                   << Makros::Typename<T>().name()
                   //<< typeid(T).name()
                   << " with " << Base::size_ << " elements. exiting." << std::endl;
    print_trace();
    exit(1);
  }
#endif
  T_A16* data = Base::data_;
  return data[i];
}


#endif
