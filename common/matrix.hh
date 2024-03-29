/**** written by Thomas Schoenemann as a private person without employment, September 2009 ***/

#ifndef MATRIX_HH
#define MATRIX_HH

#include <fstream>

#include "storage2D.hh"
#include "vector.hh"
#include "routines.hh"

namespace Math2D {

  /**************** unnamed Matrix ***********/

  //matrix class, i.e. mathematical and streaming operations need to be defined on T
  template<typename T, typename ST = size_t>
  class Matrix : public ::Storage2D<T,ST> {
  public:

    using Base = Storage2D<T,ST>;

    //according to https://gcc.gnu.org/onlinedocs/gcc-7.2.0/gcc/Common-Type-Attributes.html#Common-Type-Attributes , alignment has to be expressed like this:
    typedef T T_A16 ALIGNED16;

    /*---- constructors -----*/
    explicit Matrix() noexcept;

    explicit Matrix(ST xDim, ST yDim);

    explicit Matrix(ST xDim, ST yDim, const T default_value);

    explicit Matrix(const std::pair<ST,ST> dims);

    explicit Matrix(const std::pair<ST,ST> dims, T default_value);

    //copy constructor
    Matrix(const Matrix<T,ST>& toCopy) = default;

    //move constructor
    Matrix(Matrix<T,ST>&& toTake) noexcept = default;

    /*---- destructor ----*/
    ~Matrix();

    virtual const std::string& name() const;

    //note: with g++-4.8.5 it is a lot faster to used set_constant(0.0)
    void set_zeros() noexcept;

    inline T sum() const noexcept;

    inline T row_sum(ST y) const noexcept;

    /*** maximal element ***/
    T max() const noexcept;

    /*** minimal element ***/
    T min() const noexcept;

    inline T row_min(ST y) const noexcept;

    inline T row_max(ST y) const noexcept;

    /*** maximal absolute element = l-infinity norm ***/
    T max_abs() const noexcept;

    inline void ensure_min(T lower_limit) noexcept;

    /*** L2-norm of the matrix ***/
    inline double norm() const noexcept;

    /*** squared L2-norm ***/
    inline double sqr_norm() const noexcept;

    /*** L1-norm of the matrix ***/
    inline double norm_l1() const noexcept;

    inline void add_constant(const T addon) noexcept;

    inline void add_matrix_multiple(const Matrix<T,ST>& toAdd, const T alpha) noexcept;

    inline void operator=(const Matrix<T,ST>& toCopy) noexcept;

    Matrix<T,ST>& operator=(Matrix<T,ST>& toTake) = default;

    //---- mathematical operators ----

    //addition of another matrix of equal dimensions
    void operator+=(const Matrix<T,ST>& toAdd) noexcept;

    void operator-=(const Matrix<T,ST>& toSub) noexcept;

    //multiplication with a scalar
    void operator*=(const T scalar) noexcept;

    void elem_mul(const Matrix<T,ST>& v) noexcept;

    void elem_div(const Matrix<T,ST>& v) noexcept;

    //returns if the operation was successful
    bool savePGM(const std::string& filename, size_t max_intensity, bool fit_to_range = true) const;

  protected:
    static const std::string matrix_name_;
  };

  /******************** Named Matrix ************************/

  template <typename T, typename ST=size_t>
  class NamedMatrix : public Matrix<T,ST> {
  public:
    NamedMatrix();

    NamedMatrix(std::string name);

    NamedMatrix(ST xDim, ST yDim, std::string name);

    NamedMatrix(ST xDim, ST yDim, T default_value, std::string name);

    NamedMatrix(const std::pair<ST,ST> dims, std::string name);

    NamedMatrix(const std::pair<ST,ST> dims, T default_value, std::string name);

    ~NamedMatrix();

    inline void operator=(const Matrix<T,ST>& toCopy);

    //NOTE: does NOT copy the name
    inline void operator=(const NamedMatrix<T,ST>& toCopy);

    virtual const std::string& name() const;

    void set_name(std::string new_name);

  protected:
    std::string name_;
  };

  //NOTE: dest can be the same as src1 or src2
  inline void go_in_neg_direction(Math2D::Matrix<double>& dest, const Math2D::Matrix<double>& src1, const Math2D::Matrix<double>& src2, double alpha) noexcept
  {
    assert(dest.dims() == src1.dims());
    assert(dest.dims() == src2.dims());
    Routines::go_in_neg_direction(dest.direct_access(), dest.size(), src1.direct_access(), src2.direct_access(), alpha);
  }

  //NOTE: dest can be the same as src1 or src2
  inline void assign_weighted_combination(Math2D::Matrix<double>& dest, double w1, const Math2D::Matrix<double>& src1,
                                          double w2, const Math2D::Matrix<double>& src2) noexcept
  {
    assert(dest.dims() == src1.dims());
    assert(dest.dims() == src2.dims());
    Routines::assign_weighted_combination(dest.direct_access(), dest.size(), w1, src1.direct_access(), w2, src2.direct_access());
  }

  /***************** stand-alone operators and routines ********************/

  template<typename T, typename ST>
  Matrix<T,ST> operator+(const Matrix<T,ST>& m1, const Matrix<T,ST>& m2) noexcept;

  template<typename T, typename ST>
  Matrix<T,ST> operator*(const Matrix<T,ST>& m1, const Matrix<T,ST>& m2) noexcept;

  //streaming
  template <typename T, typename ST>
  std::ostream& operator<<(std::ostream& s, const Matrix<T,ST>& m);

  template<typename T, typename ST>
  Matrix<T,ST> transpose(const Matrix<T,ST>& m) noexcept;

  template<typename T, typename ST>
  Math1D::Vector<T,ST> operator*(const Matrix<T,ST>& m, const Math1D::Vector<T,ST>& v) noexcept;
}


namespace Makros {

  template<typename T, typename ST>
  class Typename<Math2D::Matrix<T,ST> > {
  public:

    std::string name() const
    {
      return "Math2D::Matrix<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math2D::Matrix<T> > {
  public:

    std::string name() const
    {
      return "Math2D::Matrix<" + Typename<T>() + "> ";
    }
  };

  template<typename T, typename ST>
  class Typename<Math2D::NamedMatrix<T,ST> > {
  public:

    std::string name() const
    {
      return "Math2D::NamedMatrix<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };


  template<typename T>
  class Typename<Math2D::NamedMatrix<T> > {
  public:

    std::string name() const
    {
      return "Math2D::NamedMatrix<" + Typename<T>() + "> ";
    }
  };

}


/******************************** implementation ********************************/

namespace Math2D {

  /****** implementation of (unnamed) Matrix ******/

  template<typename T, typename ST>
  /*static*/ const std::string Matrix<T,ST>::matrix_name_ = "unnamed matrix";

  template<typename T, typename ST> Matrix<T,ST>::Matrix() noexcept : Storage2D<T,ST>() {}

  template<typename T, typename ST> Matrix<T,ST>::Matrix(ST xDim, ST yDim) : Storage2D<T,ST>(xDim, yDim)  {}

  template<typename T, typename ST> Matrix<T,ST>::Matrix(ST xDim, ST yDim, const T default_value) : Storage2D<T,ST>(xDim, yDim, default_value) {}

  template<typename T, typename ST> Matrix<T,ST>::Matrix(const std::pair<ST,ST> dims) : Storage2D<T,ST>(dims)  {}

  template<typename T, typename ST> Matrix<T,ST>::Matrix(const std::pair<ST,ST> dims, T default_value) : Storage2D<T,ST>(dims, default_value) {}

  template<typename T,typename ST>
  void Matrix<T,ST>::set_zeros() noexcept
  {
    memset(Base::data_,0,Base::size_*sizeof(T));
  }

  template<typename T, typename ST> Matrix<T,ST>::~Matrix() {}

  template<typename T, typename ST>
  /*virtual*/ const std::string& Matrix<T,ST>::name() const
  {
    return Matrix<T,ST>::matrix_name_;
  }

  template<typename T, typename ST>
  inline T Matrix<T,ST>::sum() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    assertAligned16(data);

    return std::accumulate(data,data+size,(T)0);

    // T result = (T) 0;
    // for (ST i=0; i < size; i++)
    //   result += Base::data_[i];

    // return result;
  }

  template<typename T, typename ST>
  inline T Matrix<T,ST>::row_sum(ST y) const noexcept
  {
    const ST yDim = Base::yDim_;
    const ST xDim = Base::xDim_;
    assert(y < yDim);
    const T_A16* data = Base::data_;
    return std::accumulate(data+y*xDim,data+(y+1)*xDim,(T)0);
  }

  /*** maximal element ***/
  template<typename T, typename ST>
  T Matrix<T,ST>::max() const noexcept
  {

    //     T maxel = std::numeric_limits<T,ST>::min();
    //     for (ST i=0; i < Base::size_; i++) {
    //       if (Base::data_[i] > maxel)
    // 	       maxel = Base::data_[i];
    //     }
    //     return maxel;

    const T_A16* data = Base::data_;
    const ST size = Base::size_;

    assertAligned16(data);

    return *std::max_element(data,data+size);
  }

  template<>
  float Matrix<float>::max() const noexcept;

  /*** minimal element ***/
  template<typename T, typename ST>
  T Matrix<T,ST>::min() const noexcept
  {
    //     T minel = std::numeric_limits<T,ST>::max();
    //     for (ST i=0; i < Base::size_; i++) {
    //       if (Base::data_[i] < minel)
    //         minel = Base::data_[i];
    //     }
    //     return minel;

    const T_A16* data = Base::data_;
    const ST size = Base::size_;

    assertAligned16(data);

    return *std::min_element(data,data+size);
  }

  template<>
  float Matrix<float>::min() const noexcept;

  template<typename T, typename ST>
  inline T Matrix<T,ST>::row_min(ST y) const noexcept
  {
    const T* data = row_ptr(y);
    return *std::min_element(data,data+Base::size_);
  }

  template<typename T, typename ST>
  inline T Matrix<T,ST>::row_max(ST y) const noexcept
  {
    const T* data = row_ptr(y);
    return *std::max_element(data,data+Base::size_);
  }

  /*** maximal absolute element = l-infinity norm ***/
  template<typename T, typename ST>
  T Matrix<T,ST>::max_abs() const noexcept
  {
    const T_A16* data = Base::data_;

    T maxel = (T) 0;
    for (ST i=0; i < Base::size_; i++) {
      const T candidate = Makros::abs<T>(data[i]);
      maxel = std::max(maxel,candidate);
    }

    return maxel;
  }

  template<typename T, typename ST>
  inline void Matrix<T,ST>::ensure_min(T lower_limit) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;

    for (ST i=0; i < size; i++)
      data[i] = std::max(lower_limit,data[i]);
  }

  /*** L2-norm of the matrix ***/
  template<typename T, typename ST>
  inline double Matrix<T,ST>::norm() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      const double cur = (double) data[i];
      result += cur*cur;
    }

    return sqrt(result);
  }

  template<typename T, typename ST>
  inline double Matrix<T,ST>::sqr_norm() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      const double cur = (double) data[i];
      result += cur*cur;
    }

    return result;
  }

  /*** L1-norm of the matrix ***/
  template<typename T, typename ST>
  inline double Matrix<T,ST>::norm_l1() const noexcept
  {
    const ST size = Base::size_;
    const T_A16* data = Base::data_;

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      result += Makros::abs<T>(data[i]);
    }

    return result;
  }

  template<typename T, typename ST>
  inline void Matrix<T,ST>::add_constant(const T addon) noexcept
  {
    T_A16* data = Base::data_;
    const ST size = Base::size_;

    assertAligned16(data);

    for (ST i=0; i < size; i++)
      data[i] += addon;
  }

  template<typename T, typename ST>
  inline void Matrix<T,ST>::add_matrix_multiple(const Matrix<T,ST>& toAdd, const T alpha) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dims() != Base::dims()) {
      INTERNAL_ERROR << "    dimension mismatch ("
                     << Base::xDim_ << "," << Base::yDim_ << ") vs. ("
                     << toAdd.xDim() << "," << toAdd.yDim() << ")." << std::endl;
      std::cerr << "     When multiple of adding matrix \"" << toAdd.name() << "\" to  matrix \""
                << this->name() << "\". Exiting" << std::endl;
      exit(1);
    }
#endif

    //assert( Base::size_ == Base::xDim_*Base::yDim_ );

    T_A16* attr_restrict data = Base::data_;
    const T_A16* attr_restrict data2 = toAdd.direct_access();
    const ST size = Base::size_;

    assertAligned16(data);
    assertAligned16(data2);

    for (ST i=0; i < size; i++)
      data[i] += alpha * data2[i]; //toAdd.direct_access(i);
  }

  template<>
  inline void Matrix<double>::add_matrix_multiple(const Matrix<double>& toAdd, const double alpha) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dims() != Base::dims()) {
      INTERNAL_ERROR << "    dimension mismatch ("
                     << Base::xDim_ << "," << Base::yDim_ << ") vs. ("
                     << toAdd.xDim() << "," << toAdd.yDim() << ")." << std::endl;
      std::cerr << "     When multiple of adding matrix \"" << toAdd.name() << "\" to  matrix \""
                << this->name() << "\". Exiting" << std::endl;
      exit(1);
    }
#endif

    Routines::array_add_multiple(Base::data_, Base::size_, alpha, toAdd.direct_access());
  }

  template<typename T, typename ST>
  inline void Matrix<T,ST>::operator=(const Matrix<T,ST>& toCopy) noexcept
  {
    Base::operator=(toCopy);
  }

  //addition of another matrix of equal dimensions
  template<typename T, typename ST>
  void Matrix<T,ST>::operator+=(const Matrix<T,ST>& toAdd) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dims() != Base::dims()) {
      INTERNAL_ERROR << "    dimension mismatch in matrix addition(+=): ("
                     << Base::xDim_ << "," << Base::yDim_ << ") vs. ("
                     << toAdd.xDim() << "," << toAdd.yDim() << ")." << std::endl;
      std::cerr << "     When adding matrix \"" << toAdd.name() << "\" to  matrix \""
                << this->name() << "\". Exiting" << std::endl;
      exit(1);
    }
#endif

    T_A16* attr_restrict data = Base::data_;
    const T_A16* attr_restrict data2 = toAdd.direct_access();
    const ST size = Base::size_;

    assertAligned16(data);
    assertAligned16(data2);

    //assert( Base::size_ == Base::xDim_*Base::yDim_ );
    for (ST i=0; i < size; i++)
      data[i] += data2[i]; //toAdd.direct_access(i);
  }

  template<typename T, typename ST>
  void Matrix<T,ST>::operator-=(const Matrix<T,ST>& toSub) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toSub.dims() != Base::dims()) {
      INTERNAL_ERROR << "    dimension mismatch in matrix subtraction(-=): ("
                     << Base::xDim_ << "," << Base::yDim_ << ") vs. ("
                     << toSub.xDim() << "," << toSub.yDim() << ")." << std::endl;
      std::cerr << "     When subtracting matrix \"" << toSub.name() << "\" from  matrix \""
                << this->name() << "\". Exiting" << std::endl;
      exit(1);
    }
#endif

    T_A16* attr_restrict data = Base::data_;
    const T_A16* attr_restrict data2 = toSub.direct_access();
    const ST size = Base::size_;

    assertAligned16(data);
    assertAligned16(data2);

    //assert(Base::size_ == Base::xDim_*Base::yDim_);
    for (ST i=0; i < size; i++)
      data[i] -= data2[i]; //toSub.direct_access(i);
  }

  //multiplication with a scalar
  template<typename T, typename ST>
  void Matrix<T,ST>::operator*=(const T scalar) noexcept
  {
    T_A16* data = Base::data_;
    const ST size = Base::size_;

    assertAligned16(data);

    //assert(Base::size_ == Base::xDim_*Base::yDim_);
    ST i;
    for (i=0; i < size; i++)
      data[i] *= scalar;
  }

  template<>
  void Matrix<float>::operator*=(const float scalar) noexcept;

  template<>
  void Matrix<double>::operator*=(const double scalar) noexcept;

  template<typename T, typename ST>
  void Matrix<T,ST>::elem_mul(const Matrix<T,ST>& v) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;
    const T_A16* vdata = v.direct_access();

    assert(Base::xDim_ == v.xDim() && Base::yDim_ == v.yDim());
    for (ST i = 0; i < size; i++)
      data[i] *= vdata[i];
  }

  template<typename T, typename ST>
  void Matrix<T,ST>::elem_div(const Matrix<T,ST>& v) noexcept
  {
    const ST size = Base::size_;
    T_A16* data = Base::data_;
    const T_A16* vdata = v.direct_access();

    assert(Base::xDim_ == v.xDim() && Base::yDim_ == v.yDim());
    for (ST i = 0; i < size; i++)
      data[i] /= vdata[i];
  }

  //@returns if the operation was successful
  template<typename T, typename ST>
  bool Matrix<T,ST>::savePGM(const std::string& filename, size_t max_intensity, bool fit_to_range) const
  {
    std::ofstream of(filename.c_str());

    if (!of.is_open()) {
      IO_ERROR << " while saving PGM: could not write file \"" << filename
               << "\". Please check if the path is correct." << std::endl;
      return false;
    }

    of << "P5\n" << Base::xDim_ << " " << Base::yDim_ << "\n" << max_intensity;

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
        uchar c = cur_datum;
        of << c;
      }
      else {
        TODO("handle max_intensity > 255 when saving PGMs");
      }
    }

    return true;
  }


  /***************** implementation of Named Matrix ***********************/

  template<typename T, typename ST> NamedMatrix<T,ST>::NamedMatrix() : Matrix<T,ST>(), name_("zzz") {}

  template<typename T, typename ST> NamedMatrix<T,ST>::NamedMatrix(std::string name) : Matrix<T,ST>(), name_(name) {}

  template<typename T, typename ST> NamedMatrix<T,ST>::NamedMatrix(ST xDim, ST yDim, std::string name)
    : Matrix<T,ST>(xDim, yDim), name_(name) {}

  template<typename T, typename ST> NamedMatrix<T,ST>::NamedMatrix(ST xDim, ST yDim, T default_value, std::string name)
    : Matrix<T,ST>(xDim,yDim,default_value), name_(name) {}

  template<typename T, typename ST> NamedMatrix<T,ST>::NamedMatrix(const std::pair<ST,ST> dims, std::string name)
    : Matrix<T,ST>(dims), name_(name) {}

  template<typename T, typename ST> NamedMatrix<T,ST>::NamedMatrix(const std::pair<ST,ST> dims, T default_value, std::string name)
    : Matrix<T,ST>(dims,default_value), name_(name) {}

  template<typename T, typename ST> NamedMatrix<T,ST>::~NamedMatrix() {}

  template<typename T, typename ST>
  inline void NamedMatrix<T,ST>::operator=(const Matrix<T,ST>& toCopy)
  {
    Matrix<T,ST>::operator=(toCopy);
  }

  //NOTE: does NOT copy the name
  template<typename T, typename ST>
  inline void NamedMatrix<T,ST>::operator=(const NamedMatrix<T,ST>& toCopy)
  {
    Matrix<T,ST>::operator=(toCopy);
  }

  template<typename T, typename ST>
  /*virtual*/ const std::string& NamedMatrix<T,ST>::name() const
  {
    return name_;
  }

  template<typename T, typename ST>
  void NamedMatrix<T,ST>::set_name(std::string new_name)
  {
    name_ = new_name;
  }

  /***************** implementation of stand-alone operators **************/

  //implementation of stand-alone operators
  template<typename T, typename ST>
  Matrix<T,ST> operator+(const Matrix<T,ST>& m1, const Matrix<T,ST>& m2) noexcept
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (m1.dims() != m2.dims()) {

      INTERNAL_ERROR << "     dimension mismatch in matrix addition(+): ("
                     << m1.xDim() << "," << m1.yDim() << ") vs. ("
                     << m2.xDim() << "," << m2.yDim() << ")." << std::endl;
      std::cerr << "     When adding matrices \"" << m1.name() << "\" and\""
                << m2.name() << "\". Exiting..." << std::endl;

      exit(1);
    }
#endif

    Matrix<T,ST> result(m1.xDim(),m1.yDim());
    ST i;
    const ST size = m1.size();
    for (i=0; i < size; i++)
      result.direct_access(i) = m1.value(i) + m2.value(i);

    return result;
  }

  template<typename T, typename ST>
  Matrix<T,ST> operator*(const Matrix<T,ST>& m1, const Matrix<T,ST>& m2) noexcept
  {
    //there is room for optimization here
    // but if you want efficient code you should never call a routine that returns a matrix - except if most of your run-time lies elsewhere

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (m1.xDim() != m2.yDim()) {
      INTERNAL_ERROR << "     dimension mismatch in matrix multiplication(*): ("
                     << m1.xDim() << "," << m1.yDim() << ") vs. ("
                     << m2.xDim() << "," << m2.yDim() << ")." << std::endl;
      std::cerr << "     When multiplying matrices \"" << m1.name() << "\" and \""
                << m2.name() << "\". Exiting..." << std::endl;
      exit(1);
    }
#endif

    const ST xDim = m2.xDim();
    const ST yDim = m1.yDim();
    const ST zDim = m1.xDim();

    Matrix<T,ST> result(xDim,yDim);
    ST y,x,z;
    T sum;

    for (y=0; y < yDim; y++) {
      for (x=0; x < xDim; x++) {

        sum = (T) 0;
        for (z=0; z < zDim; z++) {
          //sum += m1(z,y) * m2(x,z);
          sum += m1.direct_access(y*zDim+z) * m2.direct_access(z*xDim+x);
        }

        //result(x,y) = sum;
        result.direct_access(y*xDim+x) = sum;
      }
    }

    return result;
  }

  // streaming
  template <typename T, typename ST>
  std::ostream& operator<<(std::ostream& s, const Matrix<T,ST>& m)
  {
    const ST xDim = m.xDim();
    const ST yDim = m.yDim();

    if (xDim == 0 || yDim == 0)
      s << "()";
    else if (yDim == 1) {
      s << "(";
      for (ST x=0; x < xDim; x++)
        s << " " << m.direct_access(x);
      s << " )";
    }
    else {
      s << "( " << std::endl;
      for (ST y=0; y < yDim; y++) {
        for (ST x=0; x < xDim; x++) {
          s << " " << m.direct_access(y*xDim+x);
        }
        s << std::endl;
      }
      s << ")";
    }

    return s;
  }


  template<typename T, typename ST>
  Matrix<T,ST> transpose(const Matrix<T,ST>& m) noexcept
  {
    const ST xDim = m.xDim();
    const ST yDim = m.yDim();

    Matrix<T,ST> result(yDim,xDim);
    ST x,y;
    for (x=0; x < xDim; x++) {
      for (y=0; y < yDim; y++) {
        result.direct_access(x*yDim+y) = m.direct_access(y*xDim+x);
        //result(y,x) = m(x,y);
      }
    }

    return result;
  }


  template<typename T, typename ST>
  Math1D::Vector<T,ST> operator*(const Matrix<T,ST>& m, const Math1D::Vector<T,ST>& v) noexcept
  {
    //there is room for optimization here
    // but if you want efficient code you should never call a routine that returns a vector - except if most of your run-time lies elsewhere

    const ST xDim = m.xDim();
    const ST yDim = m.yDim();

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (xDim != v.size()) {
      INTERNAL_ERROR << "     cannot multiply matrix \"" << m.name() << "\" with vector \""
                     << v.name() << "\":" << std::endl
                     << "     dimensions " << xDim << "x" << yDim << " and " << v.size()
                     << " mismatch. Exiting..." << std::endl;
      exit(1);
    }
#endif

    Math1D::Vector<T,ST> result(yDim);
    ST y,x;
    T sum;
    for (y=0; y < yDim; y++) {
      sum = (T) 0;
      for (x=0; x < xDim; x++)
        sum += m(x,y) * v[x];
      result[y] = sum;
    }

    return result;
  }

} //end of namespace Math2D

#endif
