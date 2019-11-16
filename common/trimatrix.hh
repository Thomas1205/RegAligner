/*-*-c++-*-*/
/*** written by Thomas Schoenemann as a private person without employment, March 2015 ***/
/*** if you desire the checked version, make sure your compiler defines the option SAFE_MODE on the command line ***/

#include <cmath>
#include <cassert>

#include "tristorage2D.hh"
#include "vector.hh"
#include "matrix.hh" //since we want to implement assigning a triangular matrix to a full one

#ifndef TRIMATRIX2D_HH
#define TRIMATRIX2D_HH


namespace Math2D {

  //matrix class with a triangular access or storage pattern
  //i.e. mathematical and streaming operations need to be defined on T
  // this is a base class for two common types of such matrices: upper/lower triangular matrices and symmetrical ones
  //   the two require different handling when printing them or summing the elements,
  //     but also when multiplying them with a vector or triangular matrix, transposing them  and calculating their determinant
  template<typename T, typename ST = size_t>
  class TriMatrix : public ::TriStorage2D<T,ST> {
  public:

    typedef T ALIGNED16 T_A16;

    /*---- constructors -----*/
    TriMatrix();

    TriMatrix(ST dim);

    TriMatrix(ST dim, const T default_value);

    /*---- destructor ----*/
    ~TriMatrix();

    virtual const std::string& name() const;


    /*** maximal element ***/
    T max() const;

    /*** minimal element ***/
    T min() const;

    /*** maximal absolute element = l-infinity norm ***/
    T max_abs() const;

    void add_constant(const T addon);

    //---- mathematical operators ----

    //multiplication with a scalar
    void operator*=(const T scalar);

  protected:
    //the following routines are protected to avoid adding (etc.) e.g. a symmetric matrix to a lower triangular one
    // (the implemented effect is not the expected one, in fact the correct result would be a full matrix without symmetry)
    void add_matrix_multiple(const TriMatrix<T,ST>& toAdd, T alpha);

    //addition of another matrix of equal dimensions
    void operator+=(const TriMatrix<T,ST>& toAdd);

    void operator-=(const TriMatrix<T,ST>& toSub);

    static const std::string trimatrix_name_;
  };


  template <typename T, typename ST=size_t>
  class NamedTriMatrix : public TriMatrix<T,ST> {
  public:
    NamedTriMatrix();

    NamedTriMatrix(std::string name);

    NamedTriMatrix(ST dim, std::string name);

    NamedTriMatrix(ST dim, T default_value, std::string name);

    inline void operator=(const TriMatrix<T,ST>& toCopy);

    //NOTE: does NOT copy the name
    inline void operator=(const NamedTriMatrix<T,ST>& toCopy);

    virtual const std::string& name() const;

    void set_name(std::string new_name);

  protected:
    std::string name_;
  };



  //(lower or upper) triangular matrix, based on TriMatrix
  // NOTE: operator()(x,y) gives the same as operator()(y,x). If you want asymmetric behavior use TriangularMatrixAsymAccess below
  template<typename T, typename ST = size_t>
  class TriangularMatrix : public TriMatrix<T,ST> {
  public:

    typedef T ALIGNED16 T_A16;

    /*---- constructors -----*/
    TriangularMatrix(bool lower_triangular=true);

    TriangularMatrix(ST dim, bool lower_triangular);

    TriangularMatrix(ST dim, const T default_value, bool lower_triangular);

    /*---- destructor ----*/
    ~TriangularMatrix();

    virtual const std::string& name() const;

    bool is_lower_triangular() const;

    //transform a lower into an upper triangular matrix (very cheap operation: only requires toggling a flag)
    void transpose();

    T sum() const;

    T determinant() const;

    /*** L2-norm of the matrix ***/
    double norm() const;

    /*** squared L2-norm ***/
    double sqr_norm() const;

    /*** L1-norm of the matrix ***/
    double norm_l1() const;


    // --- routines to access the protected members of TriMatrix

    void add_matrix_multiple(const TriangularMatrix<T,ST>& toAdd, const T alpha);

    //addition of another matrix of equal dimensions
    void operator+=(const TriangularMatrix<T,ST>& toAdd);

    void operator-=(const TriangularMatrix<T,ST>& toSub);


  protected:

    bool is_lower_triangular_; //tells if lower or upper triangular matrix

    static const std::string triangular_matrix_name_;
  };


  //streaming
  template <typename T, typename ST>
  std::ostream& operator<<(std::ostream& s, const TriangularMatrix<T,ST>& m);

  template<typename T, typename ST>
  TriangularMatrix<T,ST> operator+(const TriangularMatrix<T,ST>& m1, const TriangularMatrix<T,ST>& m2);

  template<typename T, typename ST>
  Math1D::Vector<T,ST> operator*(const TriangularMatrix<T,ST>& m, const Math1D::Vector<T,ST>& vec);

  template<typename T, typename ST>
  void assign(Matrix<T,ST>& dest, const TriangularMatrix<T,ST>& src);


  /**************** ***********/
  //like TriangularMatrix, but operator()(x,y) is asymmetric: only the correct half returns actual elements
  template<typename T, typename ST = size_t>
  class TriangularMatrixAsymAccess : public TriangularMatrix<T,ST> {
  public:

    typedef T ALIGNED16 T_A16;

    /*---- constructors -----*/
    TriangularMatrixAsymAccess(bool lower_triangular=true);

    TriangularMatrixAsymAccess(ST dim, bool lower_triangular);

    TriangularMatrixAsymAccess(ST dim, const T default_value, bool lower_triangular);

    /*---- destructor ----*/
    ~TriangularMatrixAsymAccess();

    virtual const std::string& name() const;

    //access on an element (handling is asymmetric)
    inline const T& operator()(ST x, ST y) const;

    inline T& operator()(ST x, ST y);

  protected:
    static const std::string triasym_matrix_name_;
    static const T zero_element_;
  };


  /**************** ***********/
  //symmetric matrix, based on TriMatrix
  template<typename T, typename ST = size_t>
  class SymmetricMatrix : public TriMatrix<T,ST> {
  public:

    typedef T ALIGNED16 T_A16;

    /*---- constructors -----*/
    SymmetricMatrix();

    SymmetricMatrix(ST dim);

    SymmetricMatrix(ST dim, const T default_value);

    /*---- destructor ----*/
    ~SymmetricMatrix();

    virtual const std::string& name() const;

    T sum() const;

    //TODO/future ideas:
    // - (permuted?) cholesky factorization
    // - inversion/solution of linear systems via cholesky factorization
    // - determinant via cholesky factorization


    // --- routines to access the protected members of TriMatrix

    void add_matrix_multiple(const SymmetricMatrix<T,ST>& toAdd, const T alpha);

    //addition of another matrix of equal dimensions
    void operator+=(const SymmetricMatrix<T,ST>& toAdd);

    void operator-=(const SymmetricMatrix<T,ST>& toSub);


  protected:
    static const std::string symmetric_matrix_name_;
  };


  //streaming
  template <typename T, typename ST>
  std::ostream& operator<<(std::ostream& s, const SymmetricMatrix<T,ST>& m);

  template<typename T, typename ST>
  SymmetricMatrix<T,ST> operator+(const SymmetricMatrix<T,ST>& m1, const SymmetricMatrix<T,ST>& m2);

  template<typename T, typename ST>
  Math1D::Vector<T,ST> operator*(const SymmetricMatrix<T,ST>& m, const Math1D::Vector<T,ST>& vec);

  template<typename T, typename ST>
  void assign(Matrix<T,ST>& dest, const SymmetricMatrix<T,ST>& src);

}

namespace Makros {

  template<typename T, typename ST>
  class Typename<Math2D::TriMatrix<T,ST> > {
  public:

    std::string name() const
    {

      return "Math2D::TriMatrix<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math2D::TriMatrix<T> > {
  public:

    std::string name() const
    {

      return "Math2D::TriMatrix<" + Typename<T>() + "> ";
    }
  };

  template<typename T, typename ST>
  class Typename<Math2D::NamedTriMatrix<T,ST> > {
  public:

    std::string name() const
    {

      return "Math2D::NamedTriMatrix<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math2D::NamedTriMatrix<T> > {
  public:

    std::string name() const
    {

      return "Math2D::NamedTriMatrix<" + Typename<T>() + "> ";
    }
  };


  template<typename T, typename ST>
  class Typename<Math2D::TriangularMatrix<T,ST> > {
  public:

    std::string name() const
    {

      return "Math2D::TriangularMatrix<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math2D::TriangularMatrix<T> > {
  public:

    std::string name() const
    {

      return "Math2D::TriangularMatrix<" + Typename<T>() + "> ";
    }
  };


  template<typename T, typename ST>
  class Typename<Math2D::TriangularMatrixAsymAccess<T,ST> > {
  public:

    std::string name() const
    {

      return "Math2D::TriangularMatrixAsymAccess<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math2D::TriangularMatrixAsymAccess<T> > {
  public:

    std::string name() const
    {

      return "Math2D::TriangularMatrixAsymAccess<" + Typename<T>() + "> ";
    }
  };


  template<typename T, typename ST>
  class Typename<Math2D::SymmetricMatrix<T,ST> > {
  public:

    std::string name() const
    {

      return "Math2D::SymmetricMatrix<" + Typename<T>() + "," + Typename<ST>() + "> ";
    }
  };

  template<typename T>
  class Typename<Math2D::SymmetricMatrix<T> > {
  public:

    std::string name() const
    {

      return "Math2D::SymmetricMatrix<" + Typename<T>() + "> ";
    }
  };

}


/******************************** implementation ********************************/

namespace Math2D {

  /****** implementation of (unnamed) TriMatrix ******/

  template<typename T, typename ST>
  /*static*/ const std::string TriMatrix<T,ST>::trimatrix_name_ = "unnamed trimatrix";

  template<typename T, typename ST> TriMatrix<T,ST>::TriMatrix() : TriStorage2D<T,ST>() {}

  template<typename T, typename ST> TriMatrix<T,ST>::TriMatrix(ST dim) : TriStorage2D<T,ST>(dim)  {}

  template<typename T, typename ST> TriMatrix<T,ST>::TriMatrix(ST dim, const T default_value) : TriStorage2D<T,ST>(dim, default_value) {}

  template<typename T, typename ST> TriMatrix<T,ST>::~TriMatrix() {}

  template<typename T, typename ST>
  /*virtual*/ const std::string& TriMatrix<T,ST>::name() const
  {
    return TriMatrix<T,ST>::trimatrix_name_;
  }

  /*** maximal element ***/
  template<typename T, typename ST>
  T TriMatrix<T,ST>::max() const
  {

    const T_A16* data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    assertAligned16(data);

    return *std::max_element(data,data+size);
  }

  /*** minimal element ***/
  template<typename T, typename ST>
  T TriMatrix<T,ST>::min() const
  {

    const T_A16* data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    assertAligned16(data);

    return *std::min_element(data,data+size);
  }

  /*** maximal absolute element = l-infinity norm ***/
  template<typename T, typename ST>
  T TriMatrix<T,ST>::max_abs() const
  {

    const T_A16* data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    assertAligned16(data);

    T maxel = (T) 0;
    for (ST i=0; i < size; i++) {
      const T candidate = Makros::abs<T>(data[i]);
      // if (candidate < ((T) 0))
      //   candidate = -candidate;
      maxel = std::max(maxel,candidate);
    }

    return maxel;
  }

  template<typename T, typename ST>
  void TriMatrix<T,ST>::add_constant(const T addon)
  {

    const T_A16* data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    assertAligned16(data);

    for (ST i=0; i < size; i++)
      data[i] += addon;
  }

  template<typename T, typename ST>
  void TriMatrix<T,ST>::add_matrix_multiple(const TriMatrix<T,ST>& toAdd, T alpha)
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dim() != TriStorage2D<T,ST>::dim_) {
      INTERNAL_ERROR << "    dimension mismatch ("
                     << TriStorage2D<T,ST>::dim_ << "," << TriStorage2D<T,ST>::fim_ << ") vs. ("
                     << toAdd.dim() << "," << toAdd.dim() << ")." << std::endl;
      std::cerr << "     When multiple of adding trimatrix \"" << toAdd.name() << "\" to  trimatrix \""
                << this->name() << "\". Exiting" << std::endl;
      exit(1);
    }
#endif

    T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;
    const T_A16* attr_restrict data2 = toAdd.direct_access();

    const ST size = TriStorage2D<T,ST>::size_;

    assert(toAdd.size() == size);

    for (ST i=0; i < size; i++)
      data[i] += alpha * data2[i]; //toAdd.direct_access(i);
  }


  //addition of another matrix of equal dimensions
  template<typename T, typename ST>
  void TriMatrix<T,ST>::operator+=(const TriMatrix<T,ST>& toAdd)
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toAdd.dim() != TriStorage2D<T,ST>::dim_) {
      INTERNAL_ERROR << "    dimension mismatch in matrix addition(+=): ("
                     << TriStorage2D<T,ST>::dim_ << "," << TriStorage2D<T,ST>::dim_ << ") vs. ("
                     << toAdd.dim() << "," << toAdd.dim() << ")." << std::endl;
      std::cerr << "     When adding trimatrix \"" << toAdd.name() << "\" to  trimatrix \""
                << this->name() << "\". Exiting" << std::endl;
      exit(1);
    }
#endif

    T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;
    const T_A16* attr_restrict data2 = toAdd.direct_access();

    const ST size = TriStorage2D<T,ST>::size_;

    for (ST i=0; i < size; i++)
      data[i] += data2[i]; //toAdd.direct_access(i);
  }

  template<typename T, typename ST>
  void TriMatrix<T,ST>::operator-=(const TriMatrix<T,ST>& toSub)
  {

#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (toSub.dim() != TriStorage2D<T,ST>::dim_) {
      INTERNAL_ERROR << "    dimension mismatch in matrix subtraction(-=): ("
                     << TriStorage2D<T,ST>::dim_ << "," << TriStorage2D<T,ST>::dim_ << ") vs. ("
                     << toSub.dim() << "," << toSub.dim() << ")." << std::endl;
      std::cerr << "     When subtracting trimatrix \"" << toSub.name() << "\" from  trimatrix \""
                << this->name() << "\". Exiting" << std::endl;
      exit(1);
    }
#endif

    T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;
    const T_A16* attr_restrict data2 = toSub.direct_access();

    const ST size = TriStorage2D<T,ST>::size_;

    for (ST i=0; i < size; i++)
      data[i] -= data2[i]; //toSub.direct_access(i);
  }

  //multiplication with a scalar
  template<typename T, typename ST>
  void TriMatrix<T,ST>::operator*=(const T scalar)
  {

    T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;

    const ST size = TriStorage2D<T,ST>::size_;

    ST i;
    for (i=0; i < TriStorage2D<T,ST>::size_; i++)
      data[i] *= scalar;
  }


  /****** implementation of NamedTriMatrix and related stand-alone operators ******/


  template<typename T, typename ST> NamedTriMatrix<T,ST>::NamedTriMatrix() {}

  template<typename T, typename ST> NamedTriMatrix<T,ST>::NamedTriMatrix(std::string name) : name_(name) {}

  template<typename T, typename ST> NamedTriMatrix<T,ST>::NamedTriMatrix(ST dim, std::string name) : TriMatrix<T,ST>(dim), name_(name) {}

  template<typename T, typename ST> NamedTriMatrix<T,ST>::NamedTriMatrix(ST dim, T default_value, std::string name) : TriMatrix<T,ST>(dim,default_value), name_(name) {}

  template<typename T, typename ST>
  inline void NamedTriMatrix<T,ST>::operator=(const TriMatrix<T,ST>& toCopy)
  {
    TriMatrix<T,ST>::operator=(toCopy);
  }

  //NOTE: does NOT copy the name
  template<typename T, typename ST>
  inline void NamedTriMatrix<T,ST>::operator=(const NamedTriMatrix<T,ST>& toCopy)
  {
    TriMatrix<T,ST>::operator=(toCopy);
  }

  template<typename T, typename ST>
  /*virtual*/ const std::string& NamedTriMatrix<T,ST>::name() const
  {
    return name_;
  }

  template<typename T, typename ST>
  void NamedTriMatrix<T,ST>::set_name(std::string new_name)
  {
    name_ = new_name;
  }




  /****** implementation of (unnamed) TriangularMatrix and related stand-alone operators ******/

  template<typename T, typename ST>
  /*static*/ const std::string TriangularMatrix<T,ST>::triangular_matrix_name_ = "unnamed triangular matrix";

  template<typename T, typename ST> TriangularMatrix<T,ST>::TriangularMatrix(bool lower_triangular) : TriMatrix<T,ST>(), is_lower_triangular_(lower_triangular) {}

  template<typename T, typename ST> TriangularMatrix<T,ST>::TriangularMatrix(ST dim, bool lower_triangular) : TriMatrix<T,ST>(dim), is_lower_triangular_(lower_triangular)  {}

  template<typename T, typename ST> TriangularMatrix<T,ST>::TriangularMatrix(ST dim, const T default_value, bool lower_triangular) :
    TriMatrix<T,ST>(dim, default_value), is_lower_triangular_(lower_triangular)  {}

  template<typename T, typename ST> TriangularMatrix<T,ST>::~TriangularMatrix() {}

  template<typename T, typename ST>
  bool TriangularMatrix<T,ST>::is_lower_triangular() const
  {
    return is_lower_triangular_;
  }

  //transform a lower into an upper triangular matrix (very cheap operation: only requires toggling a flag)
  template<typename T, typename ST>
  void TriangularMatrix<T,ST>::transpose()
  {
    is_lower_triangular_ = !is_lower_triangular_;
  }


  template<typename T, typename ST>
  /*virtual*/ const std::string& TriangularMatrix<T,ST>::name() const
  {
    return TriangularMatrix<T,ST>::triangular_matrix_name_;
  }

  template<typename T, typename ST>
  T TriangularMatrix<T,ST>::sum() const
  {

    const T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    T result = (T) 0;
    for (ST i=0; i < size; i++)
      result += data[i];

    return result;
  }

  /*** L2-norm of the matrix ***/
  template<typename T, typename ST>
  double TriangularMatrix<T,ST>::norm() const
  {

    const T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      const double cur = (double) data[i];
      result += cur*cur;
    }

    return sqrt(result);
  }

  template<typename T, typename ST>
  double TriangularMatrix<T,ST>::sqr_norm() const
  {

    const T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      const double cur = (double) data[i];
      result += cur*cur;
    }

    return result;
  }

  /*** L1-norm of the matrix ***/
  template<typename T, typename ST>
  double TriangularMatrix<T,ST>::norm_l1() const
  {

    const T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;
    const ST size = TriStorage2D<T,ST>::size_;

    double result = 0.0;
    for (ST i=0; i < size; i++) {
      result += std::abs(data[i]);
    }

    return result;
  }


  template<typename T, typename ST>
  T TriangularMatrix<T,ST>::determinant() const
  {

    const ST dim = TriStorage2D<T,ST>::dim_;

    T result = (T) 0;
    for (ST i=0; i < dim; i++)
      result *= this->operator()(i,i);

    return result;
  }

  template<typename T, typename ST>
  void TriangularMatrix<T,ST>::add_matrix_multiple(const TriangularMatrix<T,ST>& toAdd, const T alpha)
  {
    TriMatrix<T,ST>::add_matrix_multiple(toAdd,alpha);
  }

  //addition of another matrix of equal dimensions
  template<typename T, typename ST>
  void TriangularMatrix<T,ST>::operator+=(const TriangularMatrix<T,ST>& toAdd)
  {
    TriMatrix<T,ST>::operator+=(toAdd);
  }

  template<typename T, typename ST>
  void TriangularMatrix<T,ST>::operator-=(const TriangularMatrix<T,ST>& toSub)
  {
    TriMatrix<T,ST>::operator-=(toSub);
  }



  //streaming
  template <typename T, typename ST>
  std::ostream& operator<<(std::ostream& s, const TriangularMatrix<T,ST>& m)
  {

    const ST dim = m.dim();

    if (dim == 0)
      s << "()";
    else if (dim == 1) {
      s << "( " << m.direct_access(0) << " )";
    }
    else {
      s << "( " << std::endl;

      if (m.is_lower_triangular()) {
        for (ST y=0; y < dim; y++) {
          for (ST x=0; x <= y; x++)
            s << " " << m(x,y);
          for (ST x=y+1; x < dim; x++)
            s << " " << (T (0));
          s << std::endl;
        }
      }
      else {
        for (ST y=0; y < dim; y++) {
          for (ST x=0; x < y; x++)
            s << " " << (T (0));
          for (ST x=y; x < dim; x++)
            s << " " << m(x,y);
          s << std::endl;
        }
      }

      s << ")";
    }

    return s;
  }

  template<typename T, typename ST>
  TriangularMatrix<T,ST> operator+(const TriangularMatrix<T,ST>& m1, const TriangularMatrix<T,ST>& m2)
  {


#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (m1.dim() != m2.dim()) {

      INTERNAL_ERROR << "     dimension mismatch in triangular matrix addition(+): ("
                     << m1.dim() << "," << m1.dim() << ") vs. ("
                     << m2.dim() << "," << m2.dim() << ")." << std::endl;
      std::cerr << "     When adding triangular matrices \"" << m1.name() << "\" and\""
                << m2.name() << "\". Exiting..." << std::endl;

      exit(1);
    }
#endif

    typedef T ALIGNED16 T_A16;

    const T_A16* attr_restrict data = m1.direct_access();
    const T_A16* attr_restrict data2 = m2.direct_access();
    const ST size = m1.size();

    TriangularMatrix<T,ST> result(m1.dim());
    T_A16* attr_restrict data3 = result.direct_access();

    ST i;
    for (i=0; i <size; i++)
      data3[i] = data[i] + data2[i];

    return result;
  }


  //CAUTION: this currently reflects the product for a LOWER triangular matrix
  template<typename T, typename ST>
  Math1D::Vector<T,ST> operator*(const TriangularMatrix<T,ST>& m, const Math1D::Vector<T,ST>& vec)
  {

    const ST dim = m.dim();

    Math1D::Vector<T,ST> result(dim);

    if (m.is_lower_triangular()) {
      for (ST y=0; y < dim; y++) {

        T sum = T(0);
        for (ST x=0; x <= y; x++)
          sum += m(x,y) * vec[x];

        result[y] = sum;
      }
    }
    else {
      for (ST y=0; y < dim; y++) {

        T sum = T(0);
        for (ST x=y; x < dim; x++)
          sum += m(x,y) * vec[x];

        result[y] = sum;
      }
    }

    return result;
  }


  template<typename T, typename ST>
  void assign(Matrix<T,ST>& dest, const TriangularMatrix<T,ST>& src)
  {

    const ST dim = src.dim();

    dest.resize_dirty(dim,dim);

    if (src.is_lower_triangular()) {
      for (ST y=0; y < dim; y++) {
        for (ST x=0; x <= y; x++)
          dest(x,y) = src(x,y);
        for (uint x=y+1; x < dim; x++)
          dest(x,y) = 0.0;
      }
    }
    else {
      for (ST y=0; y < dim; y++) {
        for (ST x=0; x < y; x++)
          dest(x,y) = 0.0;
        for (uint x=y; x < dim; x++)
          dest(x,y) = src(x,y);
      }
    }
  }

  /****** implementation of (unnamed) TriangularMatrixAsymAccess ******/

  template<typename T, typename ST>
  /*static*/ const std::string TriangularMatrixAsymAccess<T,ST>::triasym_matrix_name_ = "unnamed triasymaccess matrix";

  template<typename T, typename ST>
  /*static*/ const T TriangularMatrixAsymAccess<T,ST>::zero_element_ = T (0);

  template<typename T, typename ST> TriangularMatrixAsymAccess<T,ST>::TriangularMatrixAsymAccess(bool lower_triangular) :
    TriangularMatrix<T,ST>(lower_triangular) {}

  template<typename T, typename ST> TriangularMatrixAsymAccess<T,ST>::TriangularMatrixAsymAccess(ST dim, bool lower_triangular) :
    TriangularMatrix<T,ST>(dim,lower_triangular)  {}

  template<typename T, typename ST> TriangularMatrixAsymAccess<T,ST>::TriangularMatrixAsymAccess(ST dim, const T default_value, bool lower_triangular) :
    TriangularMatrix<T,ST>(dim, default_value, lower_triangular)  {}

  template<typename T, typename ST> TriangularMatrixAsymAccess<T,ST>::~TriangularMatrixAsymAccess() {}


  template<typename T, typename ST>
  /*virtual*/ const std::string& TriangularMatrixAsymAccess<T,ST>::name() const
  {
    return TriangularMatrixAsymAccess<T,ST>::triasym_matrix_name_;
  }

  //access on an element (handling is asymmetric, if the desired element is outside the triangle we return the zero element)
  template<typename T, typename ST>
  inline const T& TriangularMatrixAsymAccess<T,ST>::operator()(ST x, ST y) const
  {

#ifdef SAFE_MODE
    if (x >= TriStorage2D<T,ST>::dim_ || y >= TriStorage2D<T,ST>::dim_) {
      INTERNAL_ERROR << "    access on element(" << x << "," << y
                     << ") exceeds storage dimensions of (" << TriStorage2D<T,ST>::dim_ << "," << TriStorage2D<T,ST>::dim_ << ")" << std::endl;
      std::cerr << "      in TriangularMatrixAsymAccess \"" << this->name() << "\" of type "
                << Makros::Typename<T>()
                << ". Exiting." << std::endl;
      exit(1);
    }
#endif
    if (TriangularMatrix<T,ST>::is_lower_triangular_) {
      if (x > y)
        return zero_element_;
    }
    else {
      if (x < y)
        return zero_element_;
      else
        std::swap(x,y);
    }

    return TriStorage2D<T,ST>::data_[(y*(y+1))/2+x];
  }

  template<typename T, typename ST>
  inline T& TriangularMatrixAsymAccess<T,ST>::operator()(ST x, ST y)
  {

    //essentially the same access pattern as for symmetric access, only in safe mode failure is returned

#ifdef SAFE_MODE
    if (x >= TriStorage2D<T,ST>::dim_ || y >= TriStorage2D<T,ST>::dim_
        || (TriangularMatrix<T,ST>::is_lower_triangular_ && x > y) || (!TriangularMatrix<T,ST>::is_lower_triangular_ && x < y) ) {
      INTERNAL_ERROR << "    access on element(" << x << "," << y
                     << ") exceeds storage dimensions of (" << TriStorage2D<T,ST>::dim_
                     << "," << TriStorage2D<T,ST>::dim_ << ")" << std::endl;
      std::cerr << "      in "  <<   (TriangularMatrix<T,ST>::is_lower_triangular_ ? "lower" : "upper")
                <<  " TriangularMatrixAsymAccess of type \"" << this->name() << "\" of type "
                << Makros::Typename<T>()
                << ". Exiting." << std::endl;
      exit(1);
    }
#endif

    if (x > y)
      std::swap(x,y);

    return TriStorage2D<T,ST>::data_[(y*(y+1))/2+x];
  }


  /****** implementation of (unnamed) SymmetricMatrix and related stand-alone operators ******/

  template<typename T, typename ST>
  /*static*/ const std::string SymmetricMatrix<T,ST>::symmetric_matrix_name_ = "unnamed symmetric matrix";

  template<typename T, typename ST> SymmetricMatrix<T,ST>::SymmetricMatrix() : TriMatrix<T,ST>() {}

  template<typename T, typename ST> SymmetricMatrix<T,ST>::SymmetricMatrix(ST dim) : TriMatrix<T,ST>(dim)  {}

  template<typename T, typename ST> SymmetricMatrix<T,ST>::SymmetricMatrix(ST dim, T default_value) : TriMatrix<T,ST>(dim, default_value) {}

  template<typename T, typename ST> SymmetricMatrix<T,ST>::~SymmetricMatrix() {}

  template<typename T, typename ST>
  /*virtual*/ const std::string& SymmetricMatrix<T,ST>::name() const
  {
    return SymmetricMatrix<T,ST>::symmetric_matrix_name_;
  }

  template<typename T, typename ST>
  T SymmetricMatrix<T,ST>::sum() const
  {

    const ST dim = TriStorage2D<T,ST>::dim_;

    const T_A16* attr_restrict data = TriStorage2D<T,ST>::data_;

    T result = (T) 0;
    ST i=0;
    for (ST y=0; y < dim; y++) {
      for (ST x=0; x < y; x++,i++) { //note that i is increased, too!!
        result += 2.0 * data[i];
      }
      result += data[i];
      i++;
    }

    return result;
  }


  template<typename T, typename ST>
  void SymmetricMatrix<T,ST>::add_matrix_multiple(const SymmetricMatrix<T,ST>& toAdd, T alpha)
  {
    TriMatrix<T,ST>::add_matrix_multiple(toAdd,alpha);
  }

  //addition of another matrix of equal dimensions
  template<typename T, typename ST>
  void SymmetricMatrix<T,ST>::operator+=(const SymmetricMatrix<T,ST>& toAdd)
  {
    TriMatrix<T,ST>::operator+=(toAdd);
  }

  template<typename T, typename ST>
  void SymmetricMatrix<T,ST>::operator-=(const SymmetricMatrix<T,ST>& toSub)
  {
    TriMatrix<T,ST>::operator-=(toSub);
  }



  //streaming
  template <typename T, typename ST>
  std::ostream& operator<<(std::ostream& s, const SymmetricMatrix<T,ST>& m)
  {

    const ST dim = m.dim();

    if (dim == 0)
      s << "()";
    else if (dim == 1) {
      s << "( " << m.direct_access(0) << " )";
    }
    else {
      s << "( " << std::endl;
      for (ST y=0; y < dim; y++) {
        for (ST x=0; x <= y; x++) {
          s << " " << m(x,y);
        }
        for (ST x=y+1; x < dim; x++) {
          s << " " << m(y,x);
        }
        s << std::endl;
      }
      s << ")";
    }

    return s;
  }


  template<typename T, typename ST>
  SymmetricMatrix<T,ST> operator+(const SymmetricMatrix<T,ST>& m1, const SymmetricMatrix<T,ST>& m2)
  {


#ifndef DONT_CHECK_VECTOR_ARITHMETIC
    if (m1.dim() != m2.dim()) {

      INTERNAL_ERROR << "     dimension mismatch in triangular matrix addition(+): ("
                     << m1.dim() << "," << m1.dim() << ") vs. ("
                     << m2.dim() << "," << m2.dim() << ")." << std::endl;
      std::cerr << "     When adding triangular matrices \"" << m1.name() << "\" and\""
                << m2.name() << "\". Exiting..." << std::endl;

      exit(1);
    }
#endif

    typedef T ALIGNED16 T_A16;

    const T_A16* attr_restrict data = m1.direct_access();
    const T_A16* attr_restrict data2 = m2.direct_access();
    const ST size = m1.size();

    SymmetricMatrix<T,ST> result(m1.dim());
    T_A16* attr_restrict data3 = result.direct_access();

    ST i;
    for (i=0; i < size; i++)
      data3[i] = data[i] + data2[i];

    return result;
  }


  template<typename T, typename ST>
  Math1D::Vector<T,ST> operator*(const SymmetricMatrix<T,ST>& m, const Math1D::Vector<T,ST>& vec)
  {

    const ST dim = m.dim();

    Math1D::Vector<T,ST> result(dim);

    for (ST y=0; y < dim; y++) {

      T sum = T(0);
      for (ST x=0; x < dim; x++) {
        sum += m(x,y) * vec[x];
      }

      result[y] = sum;
    }

    return result;
  }


  template<typename T, typename ST>
  void assign(Matrix<T,ST>& dest, const SymmetricMatrix<T,ST>& src)
  {

    const ST dim = src.dim();

    dest.resize_dirty(dim,dim);
    for (ST y=0; y < dim; y++) {
      for (ST x=0; x < dim; x++)
        dest(x,y) = src(x,y);
    }
  }

}


#endif
