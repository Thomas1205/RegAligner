/**** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#ifndef TENSOR_HH
#define TENSOR_HH

#include <fstream>
#include <numeric>

#include "storage3D.hh"
#include "fileio.hh"
#include "routines.hh"

namespace Math3D {

  /*** Tensor class, i.e. mathematical operations need to be defined on T ****/
  template<typename T, typename ST=size_t>
  class Tensor : public Storage3D<T,ST> {
  public:

    using Base = Storage3D<T,ST>;

    //according to https://gcc.gnu.org/onlinedocs/gcc-7.2.0/gcc/Common-Type-Attributes.html#Common-Type-Attributes , alignment has to be expressed like this:
    typedef T T_A16 ALIGNED16;

    explicit Tensor();

    explicit Tensor(ST xDim, ST yDim, ST zDim);

    explicit Tensor(const Dim3D<ST> dims);

    explicit Tensor(ST xDim, ST yDim, ST zDim, const T default_value);

    explicit Tensor(const Dim3D<ST> dims, T default_value);

    //copy constructor
    Tensor(const Tensor<T,ST>& toCopy) = default;

    //move constructor
    Tensor(Tensor<T,ST>&& toTake) = default;

    ~Tensor();

    virtual const std::string& name() const;

    //note: with g++-4.8.5 it is a lot faster to used set_constant(0.0)
    void set_zeros() noexcept;

    inline void add_const(T addon) noexcept;

    inline void add_tensor_multiple(const Tensor<T,ST>& toAdd, const T alpha) noexcept;

    inline double norm() const noexcept;

    inline double sqr_norm() const noexcept;

    inline double norm(ST x, ST y) const noexcept;

    inline double sqr_norm(ST x, ST y) const noexcept;

    /**** summing ****/

    inline T sum(ST x, ST y) const noexcept;

    inline T sum_x(ST y, ST z) const noexcept;

    inline T sum_y(ST x, ST z) const noexcept;

    inline T sum_z(ST x, ST y) const noexcept;

    inline T sum() const noexcept;

    /**** min/max ****/

    T max() const noexcept;

    T min() const noexcept;

    inline void ensure_min(T lower_limit) noexcept;

    inline T min(ST x, ST y) const noexcept;

    inline T min_x(ST y, ST z) const noexcept;

    inline T min_y(ST x, ST z) const noexcept;

    inline T min_z(ST x, ST y) const noexcept;

    inline T min(ST z) const noexcept;

    inline T max_x(ST y, ST z) const noexcept;

    inline T max_y(ST x, ST z) const noexcept;

    inline T max_z(ST x, ST y) const noexcept;

    inline T max(ST z) const noexcept;

    inline T max_abs() const noexcept;

    inline T max_abs(ST z) const noexcept;

    double max_vector_norm() const noexcept;

    void operator=(const Tensor<T,ST>& toCopy) noexcept;

    Tensor<T,ST>& operator=(Tensor<T,ST>&& toTake) = default;

    void operator+=(const Tensor<T,ST>& toAdd) noexcept;

    void operator-=(const Tensor<T,ST>& toSub) noexcept;

    void operator*=(const T scalar) noexcept;

    void elem_mul(const Tensor<T,ST>& v) noexcept;

    void elem_div(const Tensor<T,ST>& v) noexcept;

    //returns if the operation was successful
    bool savePPM(std::string filename, size_t max_intensity, bool fit_to_range = true) const;

  protected:
    static const std::string tensor_name_;
  };

  /**** named Tensor class ****/
  template <typename T, typename ST=size_t>
  class NamedTensor : public Tensor<T,ST> {
  public:

    NamedTensor();

    NamedTensor(std::string name);

    NamedTensor(ST xDim, ST yDim, ST zDim, std::string name);

    NamedTensor(ST xDim, ST yDim, ST zDim, T default_value, std::string name);

    ~NamedTensor();

    virtual const std::string& name() const;

    void set_name(std::string name);

    inline void operator=(const Tensor<T,ST>& toCopy);

    //NOTE: the name is NOT copied
    inline void operator=(const NamedTensor<T,ST>& toCopy);

  protected:
    std::string name_;
  };

  //NOTE: dest can be the same as src1 or src2
  inline void go_in_neg_direction(Math3D::Tensor<double>& dest, const Math3D::Tensor<double>& src1, const Math3D::Tensor<double>& src2, double alpha) noexcept
  {
    assert(dest.dims() == src1.dims());
    assert(dest.dims() == src2.dims());
    Routines::go_in_neg_direction(dest.direct_access(), dest.size(), src1.direct_access(), src2.direct_access(), alpha);
  }

  //NOTE: dest can be the same as src1 or src2
  inline void assign_weighted_combination(Math3D::Tensor<double>& dest, double w1, const Math3D::Tensor<double>& src1,
                                          double w2, const Math3D::Tensor<double>& src2) noexcept
  {
    assert(dest.dims() == src1.dims());
    assert(dest.dims() == src2.dims());
    Routines::assign_weighted_combination(dest.direct_access(), dest.size(), w1, src1.direct_access(), w2, src2.direct_access());
  }

  /**** stand-alone operators ****/
  template<typename T, typename ST>
  Tensor<T,ST> operator+(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2) noexcept;

  template<typename T, typename ST>
  Tensor<T,ST> operator-(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2) noexcept;
}

namespace Makros {

  template<typename T, typename ST>
  class Typename<Math3D::Tensor<T,ST> > {
  public:

    std::string name() const
    {
      return "Math3D::Tensor<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math3D::Tensor<T> > {
  public:

    std::string name() const
    {
      return "Math3D::Tensor<" + Makros::Typename<T>() + "> ";
    }
  };

  template<typename T, typename ST>
  class Typename<Math3D::NamedTensor<T,ST> > {
  public:

    std::string name() const
    {
      return "Math3D::NamedTensor<" + Makros::Typename<T>() + "," + Makros::Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math3D::NamedTensor<T> > {
  public:

    std::string name() const
    {
      return "Math3D::NamedTensor<" + Makros::Typename<T>() + "> ";
    }
  };
}


namespace Math3D {

  /********************************** implementation **********************************/

  /*** implementation of (unnamed) Tensor ***/
  template<typename T, typename ST>
  /*static*/ const std::string Tensor<T,ST>::tensor_name_ = "unnamed tensor";

  template<typename T, typename ST> Tensor<T,ST>::Tensor() : Storage3D<T,ST>() {}

  template<typename T, typename ST> Tensor<T,ST>::Tensor(ST xDim, ST yDim, ST zDim) : Storage3D<T,ST>(xDim,yDim,zDim) {}

  template<typename T, typename ST> Tensor<T,ST>::Tensor(const Dim3D<ST> dims) : Storage3D<T,ST>(dims) {}

  template<typename T, typename ST> Tensor<T,ST>::
  Tensor(ST xDim, ST yDim, ST zDim, const T default_value) :
    Storage3D<T,ST>(xDim,yDim,zDim,default_value) {}

  template<typename T, typename ST> Tensor<T,ST>::Tensor(const Dim3D<ST> dims, T default_value) : Storage3D<T,ST>(dims, default_value) {}

  template<typename T, typename ST> Tensor<T,ST>::~Tensor() {}

  template<typename T, typename ST>
  /*virtual*/ const std::string& Tensor<T,ST>::name() const
  {
    return tensor_name_;
  }

  template<typename T,typename ST>
  void Tensor<T,ST>::set_zeros() noexcept
  {
    memset(Base::data_,0,Base::size()*sizeof(T));
  }

  template<typename T, typename ST>
  inline void Tensor<T,ST>::add_const(const T addon) noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);

    for (ST i=0; i < size; i++)
      data[i] += addon;
  }

  template<typename T, typename ST>
  inline void Tensor<T,ST>::add_tensor_multiple(const Tensor<T,ST>& toAdd, const T alpha) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dims() != Base::dims()) {
      INTERNAL_ERROR << "    cannot add multiple of tensor \"" << toAdd.name() << "\" to tensor \""
                     << this->name() << "\":" << std::endl
                     << "    sizes " << toAdd.xDim() << "x" << toAdd.yDim() << "x" << toAdd.zDim()
                     << " and " << Base::xDim_ << "x" << Base::yDim_ << "x"
                     << Base::zDim_ << " are incompatible. Exiting..." << std::endl;
      exit(1);
    }
#endif

    const ST size = Base::size_;

    ST i;
    for (i=0; i < size; i++)
      Base::data_[i] += alpha * toAdd.direct_access(i);
  }

  template<>
  inline void Tensor<double>::add_tensor_multiple(const Tensor<double>& toAdd, const double alpha) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dims() != Base::dims()) {
      INTERNAL_ERROR << "    cannot add multiple of tensor \"" << toAdd.name() << "\" to tensor \""
                     << this->name() << "\":" << std::endl
                     << "    sizes " << toAdd.xDim() << "x" << toAdd.yDim() << "x" << toAdd.zDim()
                     << " and " << Base::xDim_ << "x" << Base::yDim_ << "x"
                     << Base::zDim_ << " are incompatible. Exiting..." << std::endl;
      exit(1);
    }
#endif

    Routines::array_add_multiple(Base::data_, Base::size_, alpha, toAdd.direct_access());
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::operator=(const Tensor<T,ST>& toCopy) noexcept
  {
    Base::operator=(toCopy);
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::operator+=(const Tensor<T,ST>& toAdd) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dims() != Base::dims()) {
      INTERNAL_ERROR << "    illegal addition of tensor \"" << toAdd.name() << "\" to tensor \""
                     << this->name() << "\":" << std::endl
                     << "    sizes " << toAdd.xDim() << "x" << toAdd.yDim() << "x" << toAdd.zDim()
                     << " and " << Base::xDim_ << "x" << Base::yDim_ << "x"
                     << Base::zDim_ << " are incompatible. Exiting..."
                     << std::endl;
      exit(1);
    }
#endif

    const ST size = Base::size_;
    T_A16* attr_restrict data = Base::data_;
    const T_A16* attr_restrict data2 = toAdd.direct_access();

    ST i;
    for (i=0; i < size; i++)
      data[i] += data2[i];
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::operator-=(const Tensor<T,ST>& toSub) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toSub.dims() != Base::dims()) {
      INTERNAL_ERROR << "    illegal subtraction of tensor \"" << toSub.name() << "\" from tensor \""
                     << this->name() << "\":" << std::endl
                     << "    sizes " << toSub.xDim() << "x" << toSub.yDim() << "x" << toSub.zDim()
                     << " and " << Base::xDim_ << "x" << Base::yDim_ << "x"
                     << Base::zDim_ << " are incompatible. Exiting..."
                     << std::endl;
      exit(1);
    }
#endif

    const ST size = Base::size_;
    T_A16* attr_restrict data = Base::data_;
    const T_A16* attr_restrict data2 = toSub.direct_access();

    ST i;
    for (i=0; i < size; i++)
      data[i] -= data2[i]; //toSub.direct_access(i);
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::operator*=(const T scalar) noexcept
  {

    const ST size = Base::size_;
    T_A16* data = Base::data_;

    ST i;
    for (i=0; i < size; i++)
      data[i] *= scalar;
  }

  template<>
  void Tensor<float>::operator*=(const float scalar) noexcept;

  template<>
  void Tensor<double>::operator*=(const double scalar) noexcept;

  template<typename T, typename ST>
  inline double Tensor<T,ST>::norm() const noexcept
  {
    const T_A16* data = Base::data_;

    double result = 0.0;
    for (ST i=0; i < Base::size_; i++) {
      const T temp = data[i];
      result += temp*temp;
    }

    return sqrt(result);
  }

  template<typename T, typename ST>
  inline double Tensor<T,ST>::sqr_norm() const noexcept
  {
    const T_A16* data = Base::data_;

    double result = 0.0;
    for (ST i=0; i < Base::size_; i++) {
      const T temp = data[i];
      result += temp*temp;
    }
    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum(ST x, ST y) const noexcept
  {
    T result = (T) 0;
    ST offs = (y*Base::xDim_+x)*Base::zDim_;

    const T_A16* bdata = Base::data_;
    const T* data = bdata + offs;

    for (ST z=0; z < Base::zDim_; z++)
      result += data[offs+z];

    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum_x(ST y, ST z) const noexcept
  {
    T result = (T) 0;

    for (uint x=0; x < Base::xDim_; x++)
      result += (*this)(x,y,z);

    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum_y(ST x, ST z) const noexcept
  {
    T result = (T) 0;

    for (uint y=0; y < Base::yDim; y++)
      result += (*this)(x,y,z);

    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum_z(ST x, ST y) const noexcept
  {
    T result = (T) 0;
    ST offs = (y*Base::xDim_+x)*Base::zDim_;

    const T_A16* bdata = Base::data_;
    const T* data = bdata + offs;

    for (ST z=0; z < Base::zDim_; z++)
      result += data[z];

    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::min(ST x, ST y) const noexcept
  {
    const T_A16* data = Base::data_;

    return *std::min_element(data+(y*Base::xDim_+x)*Base::zDim_,
                             data+(y*Base::xDim_+x+1)*Base::zDim_);
  }


  template<typename T, typename ST>
  inline T Tensor<T,ST>::min_x(ST y, ST z) const noexcept
  {
    T min_el = std::numeric_limits<T>::max();
    for (ST x = 0; x < Base::xDim_; x++)
      min_el = std::min(min_el,(*this)(x, y, z));

    return min_el;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::min_y(ST x, ST z) const noexcept
  {
    T min_el = std::numeric_limits<T>::min();
    for (ST y = 0; y < Base::yDim_; y++)
      min_el = std::max(min_el,(*this)(x, y, z));

    return min_el;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::min_z(ST x, ST y) const noexcept
  {
    return min(x,y);
  }

  template<typename T, typename ST>
  inline void Tensor<T,ST>::ensure_min(T lower_limit) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;

    for (ST i=0; i < size; i++)
      data[i] = std::max(lower_limit,data[i]);
  }

  template<typename T, typename ST>
  double Tensor<T,ST>::norm(ST x, ST y) const noexcept
  {
    const T_A16* data = Base::data_;

    double result = 0.0;
    ST offs = (y*Base::xDim_+x)*Base::zDim_;

    for (ST z=0; z < Base::zDim_; z++) {
      const T temp = data[offs+z];
      result += temp*temp;
    }

    return sqrt(result);
  }

  template<typename T, typename ST>
  inline double Tensor<T,ST>::sqr_norm(ST x, ST y) const noexcept
  {
    const T_A16* data = Base::data_;

    double result = 0.0;
    ST offs = (y*Base::xDim_+x)*Base::zDim_;

    for (ST z=0; z < Base::zDim_; z++) {
      const T temp = data[offs+z];
      result += temp*temp;
    }

    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);

    return std::accumulate(data,data+size,(T)0);

    // T result = 0.0;
    // for (ST i=0; i < Base::size(); i++) {
    //   result += Base::data_[i];
    // }

    // return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::max_x(ST y, ST z) const noexcept
  {
    T max_el = std::numeric_limits<T>::min();
    for (ST x = 0; x < Base::xDim_; x++)
      max_el = std::max(max_el, (*this)(x, y, z));

    return max_el;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::max_y(ST x, ST z) const noexcept
  {
    T max_el = std::numeric_limits<T>::min();
    for (ST y = 0; y < Base::yDim_; y++)
      max_el = std::max(max_el, (*this)(x, y, z));

    return max_el;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::max_z(ST x, ST y) const noexcept
  {
    return max(x,y);
  }

  template<typename T, typename ST>
  T Tensor<T,ST>::max() const noexcept
  {
    //     T max_el = std::numeric_limits<T>::min();
    //     for (ST i=0; i < Base::size_; i++)
    //       max_el = std::max(Base::data_[i],max_el);
    //     return max_el;

    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);

    return *std::max_element(data,data+size);
  }

  template<>
  float Tensor<float>::max() const noexcept;

  template<typename T, typename ST>
  T Tensor<T,ST>::min() const noexcept
  {
    //     T min_el = std::numeric_limits<T>::max();
    //     for (ST i=0; i < Base::size_; i++)
    //       min_el = std::min(Base::data_[i],min_el);
    //     return min_el;

    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);

    return *std::min_element(data,data+size);
  }

  template<>
  float Tensor<float>::min() const noexcept;

  template<typename T, typename ST>
  inline T Tensor<T,ST>::max_abs() const noexcept
  {
    const T_A16* data = Base::data_;

    T max_el = std::numeric_limits<T>::min();
    for (ST i=0; i < Base::size_; i++) {
      const T cur = Makros::abs<T>(data[i]);
      max_el = std::max(cur,max_el);
    }

    return max_el;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::max(ST z) const noexcept
  {
    const T_A16* data = Base::data_;

    T max_el = std::numeric_limits<T>::min();
    for (ST i=z; i < Base::size_; i+=z)
      max_el = std::max(data[i],max_el);

    return max_el;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::min(ST z) const noexcept
  {
    const T_A16* data = Base::data_;

    T min_el = std::numeric_limits<T>::max();
    for (ST i=z; i < Base::size_; i+=z)
      min_el = std::min(data[i],min_el);

    return min_el;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::max_abs(ST z) const noexcept
  {
    const T_A16* data = Base::data_;

    T max_el = std::numeric_limits<T>::min();
    for (ST i=z; i < Base::size_; i+=z) {
      T cur = Makros::abs<T>(data[i]);
      max_el = std::max(cur,max_el);
    }

    return max_el;
  }

  template<typename T, typename ST>
  inline double Tensor<T,ST>::max_vector_norm() const noexcept
  {
    const T_A16* data = Base::data_;

    double max_norm = 0.0;

    for (ST y=0; y < Base::yDim_; y++) {
      for (ST x=0; x < Base::xDim_; x++) {
        double cur_norm = 0.0;
        ST base = (y*Base::xDim_+x)*Base::zDim_;
        for (ST z=0; z < Base::zDim_; z++) {
          T cur_datum = data[base+z];
          cur_norm += cur_datum*cur_datum;
        }
        cur_norm = sqrt(cur_norm);
        max_norm = std::max(max_norm,cur_norm);
      }
    }
    return max_norm;
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::elem_mul(const Tensor<T,ST>& v) noexcept
  {
    assert(Base::dims() == v.dims());
    for (ST i = 0; i < Base::size_; i++)
      Base::data_[i] *= v.direct_access(i);
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::elem_div(const Tensor<T,ST>& v) noexcept
  {
    assert(Base::dims() == v.dims());
    for (ST i = 0; i < Base::size_; i++)
      Base::data_[i] /= v.direct_access(i);
  }

  template<typename T, typename ST>
  bool Tensor<T,ST>::savePPM(std::string filename, size_t max_intensity, bool fit_to_range) const
  {
    ST zDim = this->zDim();

    if (zDim != 1 && zDim != 3) {
      std::cerr << "WARNING: cannot save a tensor with " << zDim << " channels as either pgm or ppm. Operation aborted."
                << std::endl;
      return false;
    }

    std::ofstream of(filename.c_str());

    if (!of.is_open()) {
      IO_ERROR << " while saving PGM: could not write file \"" << filename
               << "\". Please check if the path is correct." << std::endl;
      return false;
    }

    if (zDim == 1)
      of << "P5\n";
    else
      of << "P6\n";
    of << Base::xDim_ << " " << Base::yDim_ << "\n" << max_intensity;

    //Reopen in binary mode to avoid silent conversion from '\n' to "\r\n" under Windows
    of.close();
    of.open(filename.c_str(), std::ios::binary | std::ios::app);
    of << '\n';

    for (ST i=0; i < Base::size_; i++) {

      if (max_intensity < 256) {
        T cur_datum = Base::data_[i];
        if (fit_to_range) {
          cur_datum = std::max(cur_datum,(T) 0);
          cur_datum = std::min(cur_datum,(T) max_intensity);
        }
        uchar c = uchar(cur_datum);
        of << c;
      }
      else {
        TODO("handle sizes > 255 when saving PPMs (or PGMs)");
      }
    }

    return true;
  }


  /*** implementation of NamedTensor ***/

  template<typename T, typename ST> NamedTensor<T,ST>::NamedTensor() : Tensor<T,ST>(), name_("yyy") {}

  template<typename T, typename ST> NamedTensor<T,ST>::NamedTensor(std::string name) : Tensor<T,ST>(), name_(name) {}

  template<typename T, typename ST> NamedTensor<T,ST>::NamedTensor(ST xDim, ST yDim, ST zDim, std::string name) :
    Tensor<T,ST>(xDim,yDim,zDim), name_(name) {}

  template<typename T, typename ST> NamedTensor<T,ST>::NamedTensor(ST xDim, ST yDim, ST zDim, T default_value, std::string name) :
    Tensor<T,ST>(xDim,yDim,zDim,default_value), name_(name) {}

  template<typename T, typename ST> NamedTensor<T,ST>::~NamedTensor() {}

  template<typename T, typename ST>
  /*virtual*/ const std::string& NamedTensor<T,ST>::name() const
  {
    return name_;
  }

  template<typename T, typename ST>
  void NamedTensor<T,ST>::set_name(std::string name)
  {
    name_ = name;
  }

  template<typename T, typename ST>
  inline void NamedTensor<T,ST>::operator=(const Tensor<T,ST>& toCopy)
  {
    Tensor<T,ST>::operator=(toCopy);
  }

  //NOTE: the name is NOT copied
  template<typename T, typename ST>
  inline void NamedTensor<T,ST>::operator=(const NamedTensor<T,ST>& toCopy)
  {
    Tensor<T,ST>::operator=(toCopy);
  }

  /*** implementation of stand-alone operators and routines ****/
  template<typename T, typename ST>
  Tensor<T,ST> operator+(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (v1.xDim() != v2.xDim() || v1.yDim() != v2.yDim() || v1.zDim() != v2.zDim() ) {

      INTERNAL_ERROR << "    cannot add vectors \"" << v1.name() << "\" and \"" << v2.name()
                     << "\":" << std::endl
                     << "    sizes " <<  v1.xDim() << "x" << v1.yDim() << "x" << v1.zDim() << " and "
                     << v2.xDim() << "x" << v2.yDim() << "x" << v2.zDim()  << " are incompatible. Exiting..." << std::endl;
      exit(1);
    }
#endif

    Tensor<T,ST> result(v1.xDim(), v1.yDim(), v1.zDim());
    const ST size = v1.size();

    ST i;
    for (i=0; i < size; i++)
      result.direct_access(i) = v1.direct_access(i)+v2.direct_access(i);

    return result;
  }

  template<typename T, typename ST>
  Tensor<T,ST> operator-(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (v1.xDim() != v2.xDim() || v1.yDim() != v2.yDim() || v1.zDim() != v2.zDim() ) {

      INTERNAL_ERROR << "    cannot subtract vector \"" << v2.name() << "\" from \"" << v1.name()
                     << "\":" << std::endl
                     << "    sizes " <<  v2.xDim() << "x" << v2.yDim() << "x" << v2.zDim() << " and "
                     << v1.xDim() << "x" << v1.yDim() << "x" << v1.zDim()  << " are incompatible. Exiting..." << std::endl;
      exit(1);
    }
#endif

    Tensor<T,ST> result(v1.xDim(), v1.yDim(), v1.zDim());
    const ST size = v1.size();

    ST i;
    for (i=0; i < size; i++)
      result.direct_access(i) = v1.direct_access(i)-v2.direct_access(i);

    return result;
  }


} //end of namespace Math3D


#endif
