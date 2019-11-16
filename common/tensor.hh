/**** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#ifndef TENSOR_HH
#define TENSOR_HH

#include <fstream>
#include <numeric>

#include "storage3D.hh"
#include "fileio.hh"

namespace Math3D {

  /*** Tensor class, i.e. mathematical operations need to be defined on T ****/
  template<typename T, typename ST=size_t>
  class Tensor : public Storage3D<T,ST> {
  public:

    typedef T ALIGNED16 T_A16;

    Tensor();

    Tensor(ST xDim, ST yDim, ST zDim);

    Tensor(ST xDim, ST yDim, ST zDim, const T default_value);

    ~Tensor();

    virtual const std::string& name() const;

    //note: with g++-4.8.5 it is a lot faster to used set_constant(0.0)
    void set_zeros();

    inline void add_const(T addon);

    void add_tensor_multiple(const Tensor<T,ST>& toAdd, const T alpha);

    void operator+=(const Tensor<T,ST>& toAdd);

    void operator-=(const Tensor<T,ST>& toSub);

    void operator*=(const T scalar);

    inline double norm() const;

    inline double sqr_norm() const;

    inline double norm(ST x, ST y) const;

    inline double sqr_norm(ST x, ST y) const;

    inline T sum(ST x, ST y) const;

    inline T sum_x(ST y, ST z) const;

    inline T sum_y(ST x, ST z) const;
    
    inline T sum_z(ST x, ST y) const;

    inline T min(ST x, ST y) const;

    inline T sum() const;

    T max() const;

    T min() const;

    T max_abs() const;

    T max(ST z) const;

    T min(ST z) const;

    T max_abs(ST z) const;

    double max_vector_norm() const;

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


  /**** stand-alone operators ****/
  template<typename T, typename ST>
  Tensor<T,ST> operator+(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2);

  template<typename T, typename ST>
  Tensor<T,ST> operator-(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2);
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

  template<typename T, typename ST> Tensor<T,ST>::Tensor(ST xDim, ST yDim, ST zDim, const T default_value) :
    Storage3D<T,ST>(xDim,yDim,zDim,default_value) {}

  template<typename T, typename ST> Tensor<T,ST>::~Tensor() {}

  template<typename T, typename ST>
  /*virtual*/ const std::string& Tensor<T,ST>::name() const
  {
    return tensor_name_;
  }

  template<typename T,typename ST>
  void Tensor<T,ST>::set_zeros()
  {
    memset(Storage3D<T,ST>::data_,0,Storage3D<T,ST>::size()*sizeof(T));
  }


  template<typename T, typename ST>
  inline void Tensor<T,ST>::add_const(const T addon)
  {
    const ST size = Storage3D<T,ST>::size_;
    const T_A16* data = Storage3D<T,ST>::data_;

    assertAligned16(data);

    for (ST i=0; i < size; i++)
      data[i] += addon;
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::add_tensor_multiple(const Tensor<T,ST>& toAdd, const T alpha)
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.xDim() != Storage3D<T,ST>::xDim_
        || toAdd.yDim() != Storage3D<T,ST>::yDim_
        || toAdd.zDim() != Storage3D<T,ST>::zDim_) {
      INTERNAL_ERROR << "    cannot add multiple of tensor \"" << toAdd.name() << "\" to tensor \""
                     << this->name() << "\":" << std::endl
                     << "    sizes " << toAdd.xDim() << "x" << toAdd.yDim() << "x" << toAdd.zDim()
                     << " and " << Storage3D<T,ST>::xDim_ << "x" << Storage3D<T,ST>::yDim_ << "x"
                     << Storage3D<T,ST>::zDim_ << " are incompatible. Exiting..."
                     << std::endl;
      exit(1);
    }
#endif

    const ST size = Storage3D<T,ST>::size_;

    ST i;
    for (i=0; i < size; i++)
      Storage3D<T,ST>::data_[i] += alpha * toAdd.direct_access(i);
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::operator+=(const Tensor<T,ST>& toAdd)
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.xDim() != Storage3D<T,ST>::xDim_
        || toAdd.yDim() != Storage3D<T,ST>::yDim_
        || toAdd.zDim() != Storage3D<T,ST>::zDim_) {
      INTERNAL_ERROR << "    illegal addition of tensor \"" << toAdd.name() << "\" to tensor \""
                     << this->name() << "\":" << std::endl
                     << "    sizes " << toAdd.xDim() << "x" << toAdd.yDim() << "x" << toAdd.zDim()
                     << " and " << Storage3D<T,ST>::xDim_ << "x" << Storage3D<T,ST>::yDim_ << "x"
                     << Storage3D<T,ST>::zDim_ << " are incompatible. Exiting..."
                     << std::endl;
      exit(1);
    }
#endif

    const ST size = Storage3D<T,ST>::size_;
    T_A16* attr_restrict data = Storage3D<T,ST>::data_;
    const T_A16* attr_restrict data2 = toAdd.direct_access();

    ST i;
    for (i=0; i < size; i++)
      data[i] += data2[i];
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::operator-=(const Tensor<T,ST>& toSub)
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toSub.xDim() != Storage3D<T,ST>::xDim_
        || toSub.yDim() != Storage3D<T,ST>::yDim_
        || toSub.zDim() != Storage3D<T,ST>::zDim_) {
      INTERNAL_ERROR << "    illegal subtraction of tensor \"" << toSub.name() << "\" from tensor \""
                     << this->name() << "\":" << std::endl
                     << "    sizes " << toSub.xDim() << "x" << toSub.yDim() << "x" << toSub.zDim()
                     << " and " << Storage3D<T,ST>::xDim_ << "x" << Storage3D<T,ST>::yDim_ << "x"
                     << Storage3D<T,ST>::zDim_ << " are incompatible. Exiting..."
                     << std::endl;
      exit(1);
    }
#endif

    const ST size = Storage3D<T,ST>::size_;
    T_A16* attr_restrict data = Storage3D<T,ST>::data_;
    const T_A16* attr_restrict data2 = toSub.direct_access();

    ST i;
    for (i=0; i < size; i++)
      data[i] -= data2[i]; //toSub.direct_access(i);
  }

  template<typename T, typename ST>
  void Tensor<T,ST>::operator*=(const T scalar)
  {

    const ST size = Storage3D<T,ST>::size_;
    T_A16* data = Storage3D<T,ST>::data_;

    ST i;
    for (i=0; i < size; i++)
      data[i] *= scalar;
  }

  template<>
  void Tensor<float>::operator*=(const float scalar);

  template<>
  void Tensor<double>::operator*=(const double scalar);

  template<typename T, typename ST>
  inline double Tensor<T,ST>::norm() const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    double result = 0.0;
    for (ST i=0; i < Storage3D<T,ST>::size_; i++) {
      const T temp = data[i];
      result += temp*temp;
    }
    return sqrt(result);
  }

  template<typename T, typename ST>
  inline double Tensor<T,ST>::sqr_norm() const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    double result = 0.0;
    for (ST i=0; i < Storage3D<T,ST>::size_; i++) {
      const T temp = data[i];
      result += temp*temp;
    }
    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum(ST x, ST y) const
  {
    T result = (T) 0;
    ST offs = (y*Storage3D<T,ST>::xDim_+x)*Storage3D<T,ST>::zDim_;

    const T_A16* bdata = Storage3D<T,ST>::data_;
    const T* data = bdata + offs;

    for (ST z=0; z < Storage3D<T,ST>::zDim_; z++) 
      result += data[offs+z];
    
    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum_x(ST y, ST z) const 
  {
    T result = (T) 0;

    for (uint x=0; x < Storage3D<T,ST>::xDim_; x++)
      result += (*this)(x,y,z);

    return result;
  }
  
  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum_y(ST x, ST z) const
  {
    T result = (T) 0;

    for (uint y=0; y < Storage3D<T,ST>::yDim; y++)
      result += (*this)(x,y,z);

    return result;    
  }
    
  template<typename T, typename ST>  
  inline T Tensor<T,ST>::sum_z(ST x, ST y) const
  {
    T result = (T) 0;
    ST offs = (y*Storage3D<T,ST>::xDim_+x)*Storage3D<T,ST>::zDim_;

    const T_A16* bdata = Storage3D<T,ST>::data_;
    const T* data = bdata + offs;

    for (ST z=0; z < Storage3D<T,ST>::zDim_; z++)
      result += data[z];

    return result;    
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::min(ST x, ST y) const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    return *std::min_element(data+(y*Storage3D<T,ST>::xDim_+x)*Storage3D<T,ST>::zDim_,
                             data+(y*Storage3D<T,ST>::xDim_+x+1)*Storage3D<T,ST>::zDim_);
  }


  template<typename T, typename ST>
  double Tensor<T,ST>::norm(ST x, ST y) const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    double result = 0.0;
    ST offs = (y*Storage3D<T,ST>::xDim_+x)*Storage3D<T,ST>::zDim_;

    for (ST z=0; z < Storage3D<T,ST>::zDim_; z++) {
      const T temp = data[offs+z];
      result += temp*temp;
    }

    return sqrt(result);
  }

  template<typename T, typename ST>
  inline double Tensor<T,ST>::sqr_norm(ST x, ST y) const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    double result = 0.0;
    ST offs = (y*Storage3D<T,ST>::xDim_+x)*Storage3D<T,ST>::zDim_;

    for (ST z=0; z < Storage3D<T,ST>::zDim_; z++) {
      const T temp = data[offs+z];
      result += temp*temp;
    }

    return result;
  }

  template<typename T, typename ST>
  inline T Tensor<T,ST>::sum() const
  {
    const ST size = Storage3D<T,ST>::size_;
    const T_A16* data = Storage3D<T,ST>::data_;

    assertAligned16(data);

    return std::accumulate(data,data+size,(T)0);

    // T result = 0.0;
    // for (ST i=0; i < Storage3D<T,ST>::size(); i++) {
    //   result += Storage3D<T,ST>::data_[i];
    // }

    // return result;
  }

  template<typename T, typename ST>
  T Tensor<T,ST>::max() const
  {

    //     T max_el = std::numeric_limits<T,ST>::min();
    //     for (ST i=0; i < Storage3D<T,ST>::size_; i++)
    //       max_el = std::max(Storage3D<T,ST>::data_[i],max_el);
    //     return max_el;

    const ST size = Storage3D<T,ST>::size_;
    const T_A16* data = Storage3D<T,ST>::data_;

    assertAligned16(data);

    return *std::max_element(data,data+size);
  }

  template<>
  float Tensor<float>::max() const;

  template<typename T, typename ST>
  T Tensor<T,ST>::min() const
  {

    //     T min_el = std::numeric_limits<T,ST>::max();
    //     for (ST i=0; i < Storage3D<T,ST>::size_; i++)
    //       min_el = std::min(Storage3D<T,ST>::data_[i],min_el);
    //     return min_el;

    const ST size = Storage3D<T,ST>::size_;
    const T_A16* data = Storage3D<T,ST>::data_;

    assertAligned16(data);

    return *std::min_element(data,data+size);
  }

  template<>
  float Tensor<float>::min() const;

  template<typename T, typename ST>
  T Tensor<T,ST>::max_abs() const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    T max_el = std::numeric_limits<T>::min();
    for (ST i=0; i < Storage3D<T,ST>::size_; i++) {
      const T cur = Makros::abs<T>(data[i]);
      max_el = std::max(cur,max_el);
    }

    return max_el;
  }

  template<typename T, typename ST>
  T Tensor<T,ST>::max(ST z) const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    T max_el = std::numeric_limits<T>::min();
    for (ST i=z; i < Storage3D<T,ST>::size_; i+=z)
      max_el = std::max(data[i],max_el);

    return max_el;
  }

  template<typename T, typename ST>
  T Tensor<T,ST>::min(ST z) const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    T min_el = std::numeric_limits<T>::max();
    for (ST i=z; i < Storage3D<T,ST>::size_; i+=z)
      min_el = std::min(data[i],min_el);

    return min_el;
  }

  template<typename T, typename ST>
  T Tensor<T,ST>::max_abs(ST z) const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    T max_el = std::numeric_limits<T>::min();
    for (ST i=z; i < Storage3D<T,ST>::size_; i+=z) {
      T cur = Makros::abs<T>(data[i]);
      max_el = std::max(cur,max_el);
    }

    return max_el;
  }

  template<typename T, typename ST>
  inline double Tensor<T,ST>::max_vector_norm() const
  {
    const T_A16* data = Storage3D<T,ST>::data_;

    double max_norm = 0.0;

    for (ST y=0; y < Storage3D<T,ST>::yDim_; y++) {
      for (ST x=0; x < Storage3D<T,ST>::xDim_; x++) {
        double cur_norm = 0.0;
        ST base = (y*Storage3D<T,ST>::xDim_+x)*Storage3D<T,ST>::zDim_;
        for (ST z=0; z < Storage3D<T,ST>::zDim_; z++) {
          T cur_datum = data[base+z];
          cur_norm += cur_datum*cur_datum;
        }
        cur_norm = sqrt(cur_norm);
        max_norm = std::max(max_norm,cur_norm);
      }
    }
    return max_norm;
  }

  // template<typename T, typename ST>
  // void Tensor<T,ST>::set_constant(const T constant) {

  //   const ST size = Storage3D<T,ST>::size_;

  //   ST i;
  //   for (i=0; i < size; i++)
  //     Storage3D<T,ST>::data_[i] = constant;
  // }

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
    of << Storage3D<T,ST>::xDim_ << " " << Storage3D<T,ST>::yDim_ << "\n" << max_intensity;

    //Reopen in binary mode to avoid silent conversion from '\n' to "\r\n" under Windows
    of.close();
    of.open(filename.c_str(), std::ios::binary | std::ios::app);
    of << '\n';

    for (ST i=0; i < Storage3D<T,ST>::size_; i++) {

      if (max_intensity < 256) {
        T cur_datum = Storage3D<T,ST>::data_[i];
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
  Tensor<T,ST> operator+(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2)
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
  Tensor<T,ST> operator-(const Tensor<T,ST>& v1, const Tensor<T,ST>& v2)
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
