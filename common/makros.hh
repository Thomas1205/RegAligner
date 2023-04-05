/*** written by Thomas Schoenemann as a private person without employment, September 2009 ***/

#ifndef MAKROS_HH
#define MAKROS_HH

#include <cassert>
#include <iostream>
#include <limits> //includes numeric_limits
#include <sstream>
#include <string>
#include <iomanip>
#include <cstdlib> //includes the exit-function
#include <typeinfo>
#include <cmath> //provides abs, sqrt, pow, log and exp functions
#include <algorithm> //necessary?
#include <numeric> //necessary? (provides accumulate and inner_product)

#include <string.h> //memcpy
#include <type_traits>

#ifdef WIN32
namespace {
  inline bool isnan(double x)
  {
    return (x != x);
  }
}
#define M_PI 3.1415926535897931
#else
using std::isnan;
using std::isinf;
#endif


#ifdef GNU_COMPILER

#define assertAligned16(p) assert( ((size_t)p) % 16 == 0);

//apparently, g++ no longer makes use of restrict attributes
#define attr_restrict __restrict
#define ref_attr_restrict __restrict
//#define attr_restrict [[restrict]] //g++ ignores this
//#define ref_attr_restrict [[restrict]] //g++ ignores this
//functions getting a pointer and changing its data are not leaf_const (https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#Common-Function-Attributes)
#define leaf_const __attribute__((leaf)) __attribute__((const))
//#define leaf_const [[leaf]] [[const]]

//pointers returned by new are guaranteed to have an address that is divisible by 16 if the type is a basic one
//it is convenient to give the compiler this hint so that he need not handle unaligned cases
#define ALIGNED16 __attribute__ ((aligned(16)))
//#define ALIGNED16 [[align(16)]] //g++ ignores this attribute
#define FLAGALIGNED16 __attribute__((assume_aligned(16)))

#include <execinfo.h>

inline void print_trace (void)
{
  void* array[15];
  size_t size;
  char** strings;
  size_t i;

  size = backtrace (array, 15);
  strings = backtrace_symbols (array, size);

  std::cerr << "Obtained " << size << " stack frames" << std::endl;

  for (i = 0; i < size; i++)
    std::cerr << strings[i] << std::endl;

  free (strings);
}

#else

//for non-g++ compilers, we do not express alignments, so we also don't check it!
#define assertAligned16(p)
#define attr_restrict
#define ref_attr_restrict
#define ALIGNED16
#define FLAGALIGNED16
#define leaf_const
inline void print_trace (void) {}
#endif

//because c++ is missing this keyword: (C++-11 has final and override)
#define abstract

/******************** Data Macros *****************************/
//NOTE: since C++-11, using is preferred over typedef
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;
using Int64 = long long int;
using UInt64 = unsigned long long int;
//according to https://gcc.gnu.org/onlinedocs/gcc-7.2.0/gcc/Common-Type-Attributes.html#Common-Type-Attributes , alignment has to be expressed like this:
typedef double double_A16 ALIGNED16;
typedef float float_A16 ALIGNED16;
typedef char char_A16 ALIGNED16;

template<typename T>
using T_A16 = T ALIGNED16;

#define MIN_DOUBLE -1.0*std::numeric_limits<double>::max()
#define MAX_DOUBLE std::numeric_limits<double>::max()
#define MIN_LONGDOUBLE -1.0*std::numeric_limits<long double>::max()
#define MAX_LONGDOUBLE std::numeric_limits<long double>::max()
#define HIGH_DOUBLE (0.1*MAX_DOUBLE)
#define EPS_DOUBLE std::numeric_limits<double>::epsilon()
#define EPS_LONGDOUBLE std::numeric_limits<long double>::epsilon()
#define MIN_FLOAT  -1.0f*std::numeric_limits<float>::max()
#define MAX_FLOAT  std::numeric_limits<float>::max()
#define HIGH_FLOAT (0.1f*MAX_FLOAT)
#define EPS_FLOAT  std::numeric_limits<float>::epsilon()
#define MAX_INT std::numeric_limits<int>::max()
#define MAX_UINT std::numeric_limits<uint>::max()
#define MAX_SIZE_T std::numeric_limits<size_t>::max()
#define MIN_LONG std::numeric_limits<long long>::min()
#define MAX_LONG std::numeric_limits<long long>::max()
#define MAX_ULONG std::numeric_limits<unsigned long long>::max()
#define MAX_USHORT std::numeric_limits<ushort>::max()

#ifndef NAN
#define NAN sqrt(-1.0)
#endif

enum NormType {L1,L2,L0_5};

enum DifferenceType {SquaredDiffs,AbsDiffs};

enum RegularityType {SquaredDiffReg,AbsDiffReg,TVReg};


/**** helpful routines ****/

namespace Makros {

  //making log, exp, pow and abs a template is convenient when you want to call the proper function inside your own template

  template<typename T>
  inline T log(T arg) noexcept
  {
    return T(::log(double(arg)));
  }

  //specializations:
  template<>
  inline float log(float arg) noexcept
  {
    return logf(arg);
  }

  template<>
  inline double log(double arg) noexcept
  {
    return ::log(arg);
  }

  template<>
  inline long double log(long double arg) noexcept
  {
    return logl(arg);
  }

  template<typename T>
  inline T sqrt(T arg) noexcept
  {
    return T(::sqrt(double(arg)));
  }

  //specializations:
  template<>
  inline float sqrt(float arg) noexcept
  {
    return sqrtf(arg);
  }

  template<>
  inline double sqrt(double arg) noexcept
  {
    return ::sqrt(arg);
  }

  template<>
  inline long double sqrt(long double arg) noexcept
  {
    return sqrtl(arg);
  }

  template<typename T>
  inline T exp(T arg) noexcept
  {
    return T(::exp(double(arg)));
  }

  //specializations:
  template<>
  inline float exp(float arg) noexcept
  {
    return expf(arg);
  }

  template<>
  inline double exp(double arg) noexcept
  {
    return ::exp(arg);
  }

  template<>
  inline long double exp(long double arg) noexcept
  {
    return expl(arg);
  }

  template<typename T>
  inline T pow(T base, T exponent) noexcept
  {
    return T(::pow(double(base),double(exponent)));
  }

  //specializations:
  template<>
  inline float pow(float base, float exponent) noexcept
  {
    return powf(base,exponent);
  }

  template<>
  inline double pow(double base, double exponent) noexcept
  {
    return ::pow(base,exponent);
  }

  template<>
  inline long double pow(long double base, long double exponent) noexcept
  {
    return powl(base,exponent);
  }

  template<typename T>
  inline T abs(T arg) noexcept
  {
    if (std::is_unsigned<T>::value) //will be if constexpr when going to C++-17
      return arg;
    if (std::is_floating_point<T>::value)
      return fabs(arg);
    return std::abs(arg);
  }

  template<>
  inline uchar abs(uchar arg) noexcept
  {
    return arg;
  }

  template<>
  inline ushort abs(ushort arg) noexcept
  {
    return arg;
  }

  template<>
  inline uint abs(uint arg) noexcept
  {
    return arg;
  }

  template<>
  inline UInt64 abs(UInt64 arg) noexcept
  {
    return arg;
  }

  template<>
  inline Int64 abs(Int64 arg) noexcept
  {
    return llabs(arg);
  }

  template<>
  inline float abs(float arg) noexcept
  {
    return fabsf(arg);
  }

  template<>
  inline double abs(double arg) noexcept
  {
    return fabs(arg);
  }

  template<>
  inline long double abs(long double arg) noexcept
  {
    return fabsl(arg);
  }

  inline void copy_byte_array(char_A16* attr_restrict dest, const char_A16* attr_restrict source, const size_t nBytes) noexcept
  {
#if !defined(USE_SSE) || USE_SSE < 4
    memcpy(dest, source, nBytes);
#else
    //experimentally, memcpy is significantly faster
    size_t i = 0;
#if USE_SSE >= 5
    for (; i + 31 < nBytes; i += 32) {

      asm __volatile__ ("vmovdqu %[d], %%ymm9 \n\t"
                        "vmovdqu %%ymm9, %[s] \n\t"
                        : [d] "=m" (dest[i]) : [s] "m" (source[i]) : "ymm9", "memory");
    }
#endif
    for (; i + 15 < nBytes; i += 16) {

      asm __volatile__ ("movdqa %[d], %%xmm9 \n\t"
                        "movdqa %%xmm9, %[s] \n\t"
                        : [d] "=m" (dest[i]) : [s] "m" (source[i]) : "xmm9", "memory");
    }

    for (; i < nBytes; i++)
      dest[i] = source[i];
#endif
  }

  template<typename T>
  inline void unified_assign(T* attr_restrict dest, const T* attr_restrict source, size_t size) noexcept
  {
    if (std::is_trivially_copyable<T>::value) //will be if constexpr when going to C++-17
      memcpy(dest, source, size * sizeof(T));
    else {
      for (size_t i=0; i < size; i++)
        dest[i] = source[i];
    }
  }

  //some specializations for compilers who do not optimize entirely
  template<>
  inline void unified_assign(char* attr_restrict dest, const char* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(char));
  }

  template<>
  inline void unified_assign(uchar* attr_restrict dest, const uchar* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(uchar));
  }

  template<>
  inline void unified_assign(short* attr_restrict dest, const short* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(short));
  }

  template<>
  inline void unified_assign(ushort* attr_restrict dest, const ushort* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(ushort));
  }

  template<>
  inline void unified_assign(int* attr_restrict dest, const int* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(int));
  }

  template<>
  inline void unified_assign(uint* attr_restrict dest, const uint* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(uint));
  }

  template<>
  inline void unified_assign(float* attr_restrict dest, const float* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(float));
  }

  template<>
  inline void unified_assign(double* attr_restrict dest, const double* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(double));
  }

  template<>
  inline void unified_assign(long double* attr_restrict dest, const long double* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(long double));
  }

  template<typename T>
  inline void unified_move_assign(T* attr_restrict dest, const T* attr_restrict source, size_t size) noexcept
  {
    if (std::is_trivially_copyable<T>::value) //will be if constexpr when going to C++-17
      memcpy(dest, source, size * sizeof(T));
    else {
      for (size_t i=0; i < size; i++)
        dest[i] = std::move(source[i]);
    }
  }

  //some specializations for compilers who do not optimize entirely
  template<>
  inline void unified_move_assign(int* attr_restrict dest, const int* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(int));
  }

  template<>
  inline void unified_move_assign(uint* attr_restrict dest, const uint* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(uint));
  }

  template<>
  inline void unified_move_assign(float* attr_restrict dest, const float* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(float));
  }

  template<>
  inline void unified_move_assign(double* attr_restrict dest, const double* attr_restrict source, size_t size) noexcept
  {
    memcpy(dest, source, size * sizeof(double));
  }


  inline size_t highest_bit(size_t val) noexcept
  {
    assert(val > 0);
    size_t ret = 0;
#ifndef USE_ASM
    val >>= 1;
    while (val > 0) {
      ret++;
      val >>= 1;
    }
#else
    __asm__ volatile ("bsr %%rcx, %%rbx \n\t" //bit scan reverse
                      : [ret] "+b"(ret) : [val] "c" (val) : "cc" );
#endif

    return ret;
  }
}

template<typename T>
std::string toString(T obj, uint width=1)
{
  std::ostringstream s;

  s << std::setw(width) << std::setfill('0') << obj;
  return s.str();
}

namespace Makros {

  void register_typename(const std::string& id, const std::string& fullname);

  std::string get_typename(const std::string& id);

  template<typename T>
  class Typename {
  public:

    std::string name() const;
  };

  template<typename T>
  std::string Typename<T>::name() const
  {
    return get_typename(typeid(T).name());
  }

  //specializations:

  template<typename T>
  class Typename<const T> {
  public:

    std::string name() const
    {
      return "const " +Typename<T>().name();
    }
  };


  template<typename T>
  class Typename<T*> {
  public:

    std::string name() const
    {
      return Typename<T>().name() + "*";
    }
  };
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Makros::Typename<T>& t)
{
  out << t.name();
  return out;
}

template<typename T>
std::string operator+(std::string s, const Makros::Typename<T>& t)
{
  return s + t.name();
}

/***********************/

template<typename T>
inline T convert(const std::string s)
{
  std::istringstream is(s);
  T result;

  is >> result;
  if (is.bad() || is.fail()) {
    std::cerr << "ERROR: conversion of \"" << s << "\" to " << Makros::Typename<T>().name()
              << " failed. Exiting." << std::endl;
    exit(1);
  }
  if (!is.eof()) {

    //check if the string contains additional characters that are not whitespace
    char c;
    while (is >> c) {
      if (c != ' ' && c != '\n' && c != 13 && c != 10) {
        std::cerr << "WARNING AFTER CONVERSION: string contains additional characters" << std::endl;
        break;
      }
    }
  }

  return result;
}

template<>
inline uint convert<uint>(const std::string s)
{
  uint result = 0;
  char c;
  uint i=0;
  for (; i < s.size(); i++) {
    c = s[i];

    if (c < '0' || c > '9') {
      std::cerr << "ERROR: conversion of \"" << s << "\" to uint failed. Exiting." << std::endl;
      exit(1);
    }
    result = 10*result + (c - '0');
  }

  return result;
}

//C++20 has bit_cast in <bit>
template<typename T1, typename T2>
T2 reinterpret(const T1 arg)  noexcept
{
  assert(sizeof(T1) == sizeof(T2));
  return *reinterpret_cast<T2*>(&arg);
}

template<typename T1, typename T2>
void operator+=(std::pair<T1,T2>& x, const std::pair<T1,T2>& y) noexcept
{
  x.first += y.first;
  x.second += y.second;
}


/********************* Code Macros ****************************/
#define TODO(s) { std::cerr << "TODO ERROR[" << __FILE__ << ":" << __LINE__ << "]: feature \"" << (s) << "\" is currently not implemented. Exiting..." << std::endl; exit(1); }
#define EXIT(s) { std::cerr << __FILE__ << ":" << __LINE__ << ": " <<  s << std::endl; exit(1); }
#define MAKENAME(s) std::string(#s) + std::string("[") + std::string(__FILE__) + std::string(":") + toString(__LINE__) + std::string("]")

#ifdef SAFE_MODE
#define OPTINLINE
#else
#define OPTINLINE inline
#endif

#define INTERNAL_ERROR std::cerr << "INTERNAL ERROR[" << __FILE__ << ":" << __LINE__ << "]:" << std::endl
#define USER_ERROR std::cerr << "ERROR: "
#define IO_ERROR std::cerr << "I/O ERROR[" << __FILE__ << ":" << __LINE__ << "]:" << std::endl
#define WARNING std::cerr << "WARNING[" << __FILE__ << ":" << __LINE__ << "]:" << std::endl

template<typename T>
inline T sign(T arg) noexcept
{
  if (arg < ((T) 0.0) )
    return ((T) -1.0);
  else if (arg == ((T) 0.0))
    return ((T) 0.0);
  else
    return ((T) 1.0);
}

template<typename T>
inline T robust_sign(T arg, T tolerance) noexcept
{
  if (arg < ((T) -tolerance) )
    return ((T) -1.0);
  else if (arg > ((T) tolerance))
    return ((T) 1.0);
  else
    return ((T) 0.0);
}

//load a cache line into the L0 processor cache
template<typename T>
inline void prefetcht0(const T* ptr) noexcept
{
#if USE_SSE >= 1
  //prefetch is part of SSE1
  asm __volatile__ ("prefetcht0 %[ptr]" : : [ptr] "m" (ptr[0]));
#endif
}

//load a cache line into the L1 processor cache
template<typename T>
inline void prefetcht1(const T* ptr) noexcept
{
#if USE_SSE >= 1
  //prefetch is part of SSE1
  asm ("prefetcht1 %[ptr]" : : [ptr] "m" (ptr[0]));
#endif
}

//load a cache line into the L2 processor cache
template<typename T>
inline void prefetcht2(const T* ptr) noexcept
{
#if USE_SSE >= 1
  //prefetch is part of SSE1
  asm ("prefetcht2 %[ptr]" : : [ptr] "m" (ptr[0]));
#endif
}

//load a cache line into the nontemporal cache, when you think you won't need the data again
template<typename T>
inline void prefetchnta(const T* ptr) noexcept
{
#if USE_SSE >= 1
  //prefetch is part of SSE1
  asm ("prefetchnta %[ptr]" : : [ptr] "m" (ptr[0]));
#endif
}

//NOTE: x86-64 also has prefetchw and prefetchwt1 (load and signal intent to write). And clflush writes a cache line back to mem, freeing the space in the cache

namespace Makros {

  template<typename T1, typename T2>
  class first_lower {
  public:
    inline bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2) const noexcept
    {
      return (p1.first < p2.first);
    }
  };

  template<typename T1, typename T2>
  class first_higher {
  public:
    inline bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2) const noexcept
    {
      return (p1.first > p2.first);
    }
  };

} //end of namespace Makros

template<typename T>
class SwapOp {
public:

  inline void operator()(T& val1, T& val2) const  noexcept
  {
    std::swap(val1, val2);
  }
};

template<typename T>
class SpecialSwapOp {
public:

  inline void operator()(T& val1, T& val2) const noexcept
  {
    val1.swap(val2);
  }
};

#endif
