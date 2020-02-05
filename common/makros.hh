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
#include <cmath>
#include <algorithm>

#include <string.h> //memcpy

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

#define attr_restrict __restrict
#define ref_attr_restrict __restrict
//pointers returned by new are guaranteed to have an address that is divisible by 16 if the type is a basic one
//it is convenient to give the compiler this hint so that he need not handle unaligned cases
#define ALIGNED16 __attribute__ ((aligned(16)))
#define assertAligned16(p) assert( ((size_t)p) % 16 == 0);

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
#define attr_restrict
#define ref_attr_restrict
#define ALIGNED16
#define assertAligned16(p)
inline void print_trace (void) {}
#endif

//because c++ is missing those keywords:
#define abstract
#define overide
#define overrides

/******************** Data Macros *****************************/
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef long long int Int64;
typedef unsigned long long int UInt64;
typedef double ALIGNED16 double_A16;
typedef float ALIGNED16 float_A16;

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
  inline T log(T arg)
  {
    return T(::log(double(arg)));
  }

  //specializations:
  template<>
  inline float log(float arg)
  {
    return logf(arg);
  }

  template<>
  inline double log(double arg)
  {
    return ::log(arg);
  }

  template<>
  inline long double log(long double arg)
  {
    return logl(arg);
  }

  template<typename T>
  inline T sqrt(T arg) {
    return T(::sqrt(double(arg)));
  }
  
  //specializations:
  template<>
  inline float sqrt(float arg) {
    return sqrtf(arg);
  }

  template<>
  inline double sqrt(double arg) {
    return ::sqrt(arg);
  }

  template<>
  inline long double sqrt(long double arg) {
    return sqrtl(arg);
  }

  template<typename T>
  inline T exp(T arg)
  {
    return T(::exp(double(arg)));
  }

  //specializations:
  template<>
  inline float exp(float arg)
  {
    return expf(arg);
  }

  template<>
  inline double exp(double arg)
  {
    return ::exp(arg);
  }

  template<>
  inline long double exp(long double arg)
  {
    return expl(arg);
  }

  template<typename T>
  inline T pow(T base, T exponent)
  {
    return T(::pow(double(base),double(exponent)));
  }

  //specializations:
  template<>
  inline float pow(float base, float exponent)
  {
    return powf(base,exponent);
  }

  template<>
  inline double pow(double base, double exponent)
  {
    return ::pow(base,exponent);
  }

  template<>
  inline long double pow(long double base, long double exponent)
  {
    return powl(base,exponent);
  }

  template<typename T>
  inline T abs(T arg)
  {
    return std::abs(arg);
  }

  template<>
  inline uchar abs(uchar arg)
  {
    return arg;
  }

  template<>
  inline ushort abs(ushort arg)
  {
    return arg;
  }

  template<>
  inline uint abs(uint arg)
  {
    return arg;
  }

  template<>
  inline UInt64 abs(UInt64 arg)
  {
    return arg;
  }

  template<>
  inline Int64 abs(Int64 arg) 
  {
    return llabs(arg);
  }

  template<>
  inline float abs(float arg)
  {
    return fabsf(arg);
  }

  template<>
  inline double abs(double arg)
  {
    return fabs(arg);
  }

  template<>
  inline long double abs(long double arg)
  {
    return fabsl(arg);
  }

  template<typename T>
  inline void unified_assign(T* attr_restrict dest, const T* attr_restrict source, size_t size)
  {
    for (size_t i=0; i < size; i++)
      dest[i] = source[i];
  }
  
  template<>
  inline void unified_assign(char* attr_restrict dest, const char* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(char));
  }  
  
  template<>
  inline void unified_assign(uchar* attr_restrict dest, const uchar* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(uchar));
  }  
  
  template<>
  inline void unified_assign(short* attr_restrict dest, const short* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(short));
  }  

  template<>
  inline void unified_assign(ushort* attr_restrict dest, const ushort* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(ushort));
  }  
  
  template<>
  inline void unified_assign(int* attr_restrict dest, const int* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(int));
  }  

  template<>
  inline void unified_assign(uint* attr_restrict dest, const uint* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(uint));
  }  

  template<>
  inline void unified_assign(float* attr_restrict dest, const float* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(float));
  }  
  
  template<>
  inline void unified_assign(double* attr_restrict dest, const double* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(double));
  }  

  template<>
  inline void unified_assign(long double* attr_restrict dest, const long double* attr_restrict source, size_t size) 
  {
    memcpy(dest, source, size * sizeof(long double));
  }    

  inline size_t highest_bit(size_t val) 
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

template <typename T>
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

template<typename T1, typename T2>
void operator+=(std::pair<T1,T2>& x, const std::pair<T1,T2>& y)
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
inline T sign(T arg)
{
  if (arg < ((T) 0.0) )
    return ((T) -1.0);
  else if (arg == ((T) 0.0))
    return ((T) 0.0);
  else
    return ((T) 1.0);
}

template<typename T>
inline T robust_sign(T arg, T tolerance)
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
inline void prefetcht0(const T* ptr)
{
#if USE_SSE >= 1
  //prefetch is part of SSE1
  asm __volatile__ ("prefetcht0 %[ptr]" : : [ptr] "m" (ptr[0]));
#endif
}

//load a cache line into the L1 processor cache
template<typename T>
inline void prefetcht1(const T* ptr)
{
#if USE_SSE >= 1
  //prefetch is part of SSE1
  asm ("prefetcht1 %[ptr]" : : [ptr] "m" (ptr[0]));
#endif
}

//load a cache line into the L2 processor cache
template<typename T>
inline void prefetcht2(const T* ptr)
{
#if USE_SSE >= 1
  //prefetch is part of SSE1
  asm ("prefetcht2 %[ptr]" : : [ptr] "m" (ptr[0]));
#endif
}

namespace Makros {

  inline float max(const float_A16* data, size_t nData)
  {
    float max_val=MIN_FLOAT;
    float cur_datum;
    size_t i;

//#if !defined(USE_SSE) || USE_SSE < 2
#if 1 // g++ 4.8.5 uses avx instructions automatically
    for (i=0; i < nData; i++) {
      cur_datum = data[i];
      max_val = std::max(max_val,cur_datum);
    }
#else
    //movaps is part of SSE2

    float tmp[4] = {MIN_FLOAT,MIN_FLOAT,MIN_FLOAT,MIN_FLOAT};
    const float* fptr;

    asm __volatile__ ("movaps %[tmp], %%xmm6" : : [tmp] "m" (tmp[0]) : "xmm6");
    for (i=0; (i+4) <= nData; i += 4) {
      fptr = data+i;
      asm __volatile__ ("movaps %[fptr], %%xmm7\n\t"
                        "maxps %%xmm7, %%xmm6" : : [fptr] "m" (fptr[0]) : "xmm6", "xmm7");

    }
    asm __volatile__ ("movups %%xmm6, %[tmp]" : [tmp] "=m" (tmp[0]) : : "memory");
    for (i=0; i < 4; i++)
      max_val = std::max(max_val,tmp[i]);

    for (i= nData - (nData % 4); i < nData; i++) {
      cur_datum = data[i];
      if (cur_datum > max_val)
        max_val = cur_datum;
    }
#endif

    return max_val;
  }

  inline float min(const float_A16* data, size_t nData)
  {
    float min_val=MAX_FLOAT;
    float cur_datum;
    size_t i;

    //#if !defined(USE_SSE) || USE_SSE < 2
#if 1 // g++ 4.8.5 uses avx instructions automatically
    for (i=0; i < nData; i++) {
      cur_datum = data[i];
      min_val = std::min(min_val,cur_datum);
    }
#else
    //movaps is part of SSE2

    float tmp[4] = {MAX_FLOAT,MAX_FLOAT,MAX_FLOAT,MAX_FLOAT};
    const float* fptr;

    asm __volatile__ ("movaps %[tmp], %%xmm6" : : [tmp] "m" (tmp[0]) : "xmm6");
    for (i=0; (i+4) <= nData; i += 4) {
      fptr = data+i;
      asm __volatile__ ("movaps %[fptr], %%xmm7 \n\t"
                        "minps %%xmm7, %%xmm6 \n\t" : : [fptr] "m" (fptr[0]) : "xmm6", "xmm7");
    }
    asm __volatile__ ("movups %%xmm6, %[tmp]" : [tmp] "=m" (tmp[0]) :  : "memory");
    for (i=0; i < 4; i++)
      min_val = std::min(min_val,tmp[i]);

    for (i= nData - (nData % 4); i < nData; i++) {
      cur_datum = data[i];
      if (cur_datum < min_val)
        min_val = cur_datum;
    }
#endif

    return min_val;
  }


  inline void find_max_and_argmax(const float_A16* data, const size_t nData, float& max_val, size_t& arg_max)
  {
    max_val = MIN_FLOAT;
    arg_max = MAX_UINT;

    assertAligned16(data);

#if !defined(USE_SSE) || USE_SSE < 4

    if (nData > 0) {
      const float* ptr = std::max_element(data,data+nData);
      max_val = *ptr;
      arg_max = ptr - data;
    }

    // for (i=0; i < nData; i++) {
    //   float cur_val = data[i];

    //   if (cur_val > max_val) {
    //     max_val = cur_val;
    //     arg_max = i;
    //   }
    // }
#elif USE_SSE >= 5

    //use AVX - align16 is no good, need align32 for aligned moves
    
    float val = MIN_FLOAT;
    const uint one = 1;
    float inc = *reinterpret_cast<const float*>(&one);
    size_t i = 0;
    const float* fptr;
    
    asm __volatile__ ("vbroadcastss %[tmp], %%ymm6 \n\t" //ymm6 is max register
                      "vxorps %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmax) to zero
                      "vbroadcastss %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                      "vxorps %%ymm3, %%ymm3, %%ymm3 \n\t" //sets ymm3 (= current set index) to zero
                      : : [tmp] "m" (val), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");
    
    for (i=0; (i+8) <= nData; i += 8) {
      fptr = data+i;
  
      assertAligned16(fptr);

      asm __volatile__ ("vmovups %[fptr], %%ymm7 \n\t"
                        "vcmpnleps %%ymm6, %%ymm7, %%ymm0 \n\t" 
                        "vblendvps %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                        "vblendvps %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                        "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                        : : [fptr] "m" (fptr[0]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
    }
    
    float tmp[8];
    uint itemp[8];
    
    asm __volatile__ ("vmovups %%ymm6, %[tmp] \n\t"
                      "vmovups %%ymm5, %[itemp]"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    float cur_val;
    for (i=0; i < 8; i++) {
      cur_val = tmp[i];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = 8*itemp[i] + i;
      }
    }

    for (i= nData - (nData % 8); i < nData; i++) {
      cur_val = data[i];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = i;
      }
    }

#else
    //blendvps is part of SSE4

    size_t i;
    float cur_val;

    assert(nData <= 17179869183);

    float tmp[4] = {MIN_FLOAT,MIN_FLOAT,MIN_FLOAT,MIN_FLOAT};
    const float* fptr;

    wchar_t itemp[4] = {1,1,1,1}; //increment array

    asm __volatile__ ("movups %[tmp], %%xmm6 \n\t" //xmm6 is max register
                      "xorps %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmax) to zero
                      "movups %[itemp], %%xmm4 \n\t" //xmm4 is increment register
                      "xorps %%xmm3, %%xmm3 \n\t" //sets xmm3 (= current set index) to zero
                      : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) : "xmm3", "xmm4", "xmm5", "xmm6");

    for (i=0; (i+4) <= nData; i += 4) {
      fptr = data+i;

      asm __volatile__ ("movaps %[fptr], %%xmm7 \n\t"
                        "movaps %%xmm7, %%xmm0 \n\t"
                        "cmpnleps %%xmm6, %%xmm0 \n\t"
                        "blendvps %%xmm7, %%xmm6 \n\t"
                        "blendvps %%xmm3, %%xmm5 \n\t"
                        "paddd %%xmm4, %%xmm3 \n\t"
                        : : [fptr] "m" (fptr[0]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
    }

    asm __volatile__ ("movups %%xmm6, %[tmp] \n\t"
                      "movups %%xmm5, %[itemp]"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    for (i=0; i < 4; i++) {
      cur_val = tmp[i];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = 4*itemp[i] + i;
      }
    }

    for (i= nData - (nData % 4); i < nData; i++) {
      cur_val = data[i];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = i;
      }
    }
#endif
  }

  inline void find_max_and_argmax(const double_A16* data, const size_t nData, double& max_val, size_t& arg_max)
  {
    max_val = MIN_DOUBLE;
    arg_max = MAX_UINT;

    assertAligned16(data);

#if !defined(USE_SSE) || USE_SSE < 4
//#if 1
    if (nData > 0) {
      const double* ptr = std::max_element(data,data+nData);
      max_val = *ptr;
      arg_max = ptr - data;
    }

    // for (i=0; i < nData; i++) {
    //   double cur_val = data[i];

    //   if (cur_val > max_val) {
    //     max_val = cur_val;
    //     arg_max = i;
    //   }
    // }
#elif USE_SSE >= 5

    //use AVX  - align16 is no good, need align32 for aligned moves
    
    assert(sizeof(size_t) == 8);
    
    double val = MIN_DOUBLE;
    const size_t one = 1;
    double inc = *reinterpret_cast<const double*>(&one);
    size_t i = 0;
    const double* dptr;
    
    asm __volatile__ ("vbroadcastsd %[tmp], %%ymm6 \n\t" //ymm6 is max register
                      "vxorpd %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmax) to zero
                      "vbroadcastsd %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                      "vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t" //sets ymm3 (= current set index) to zero
                      : : [tmp] "m" (val), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");    
    
    for (i=0; (i+4) <= nData; i += 4) {
      dptr = data+i;

      asm __volatile__ ("vmovupd %[dptr], %%ymm7 \n\t"
                        "vcmpnlepd %%ymm6, %%ymm7, %%ymm0 \n\t"
                        "vblendvpd %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                        "vblendvpd %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                        "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                        : : [dptr] "m" (dptr[0]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
    }   
    
    
    double tmp[4];
    size_t itemp[4];
    
    asm __volatile__ ("vmovups %%ymm6, %[tmp] \n\t"
                      "vmovups %%ymm5, %[itemp]"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    double cur_val;
                      
    for (i=0; i < 4; i++) {
      cur_val = tmp[i];
      //std::cerr << "cur val: " << cur_val << std::endl;
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = 4*itemp[i] + i;
      }
    }

    //std::cerr << "minval: " << min_val << std::endl;
    
    for (i = nData - (nData % 4); i < nData; i++) {      
      cur_val = data[i];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = i;
      }
    }

    
#else

    size_t i;
    double cur_val;

    assert(nData < 8589934592);

    volatile double tmp[2] = {MIN_DOUBLE,MIN_DOUBLE};
    const double* dptr;

    volatile wchar_t itemp[4] = {0,1,0,1};


    asm __volatile__ ("movupd %[tmp], %%xmm6 \n\t"
                      "xorpd %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmax) to zero
                      "movupd %[itemp], %%xmm4 \n\t"
                      "xorpd %%xmm3, %%xmm3 \n\t" //sets xmm3 (= current set index)
                      : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) :  "xmm3", "xmm4", "xmm5", "xmm6");

    for (i=0; (i+2) <= nData; i += 2) {
      dptr = data+i;


      asm __volatile__ ("movapd %[dptr], %%xmm7 \n\t"
                        "movapd %%xmm7, %%xmm0 \n\t"
                        "cmpnlepd %%xmm6, %%xmm0 \n\t"
                        "blendvpd %%xmm7, %%xmm6 \n\t"
                        "blendvpd %%xmm3, %%xmm5 \n\t"
                        "paddd %%xmm4, %%xmm3 \n\t"
                        : : [dptr] "m" (dptr[0]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
    }

    asm __volatile__ ("movupd %%xmm6, %[tmp] \n\t"
                      "movupd %%xmm5, %[itemp] \n\t"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    assert(itemp[0] == 0);
    assert(itemp[2] == 0);

    for (i=0; i < 2; i++) {
      cur_val = tmp[i];
      //std::cerr << "cur val: " << cur_val << std::endl;
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = 2*itemp[2*i+1] + i;
      }
    }

    //std::cerr << "minval: " << min_val << std::endl;

    if ((nData % 2) == 1) {
      cur_val = data[nData-1];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = nData-1;
      }
    }
#endif
  }

  inline void find_min_and_argmin(const float_A16* data, const size_t nData, float& min_val, size_t& arg_min)
  {
    min_val = MAX_FLOAT;
    arg_min = MAX_UINT;

    assertAligned16(data);

#if !defined(USE_SSE) || USE_SSE < 4

    if (nData > 0) {
      const float* ptr = std::min_element(data,data+nData);
      min_val = *ptr;
      arg_min = ptr - data;
    }

    // for (i=0; i < nData; i++) {
    //   float cur_val = data[i];

    //   if (cur_val < min_val) {
    //     min_val = cur_val;
    //     arg_min = i;
    //   }
    // }
#elif USE_SSE >= 5

    //use AVX  - align16 is no good, need align32 for aligned moves
    
    float val = MAX_FLOAT;
    const uint one = 1;
    float inc = *reinterpret_cast<const float*>(&one);
    size_t i = 0;
    const float* fptr;
    
    asm __volatile__ ("vbroadcastss %[tmp], %%ymm6 \n\t" //ymm6 is min register
                      "vxorps %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmin) to zero
                      "vbroadcastss %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                      "vxorps %%ymm3, %%ymm3, %%ymm3 \n\t" //sets ymm3 (= current set index) to zero
                      : : [tmp] "m" (val), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");
    
    for (i=0; (i+8) <= nData; i += 8) {
      fptr = data+i;

      asm __volatile__ ("vmovups %[fptr], %%ymm7 \n\t"
                        "vcmpltps %%ymm6, %%ymm7, %%ymm0 \n\t" //destination is last
                        "vblendvps %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                        "vblendvps %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                        "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                        : : [fptr] "m" (fptr[0]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
    }

    float tmp[8];
    uint itemp[8];
    
    asm __volatile__ ("vmovups %%ymm6, %[tmp] \n\t"
                      "vmovups %%ymm5, %[itemp]"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    float cur_val;

    for (i=0; i < 8; i++) {
      cur_val = tmp[i];
      //std::cerr << "cur val: " << cur_val << std::endl;
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = 8*itemp[i] + i;
      }
    }

    //std::cerr << "minval: " << min_val << std::endl;

    for (i= nData - (nData % 8); i < nData; i++) {
      cur_val = data[i];
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = i;
      }
    }


#else
    //blendvps is part of SSE4

    size_t i;
    float cur_val;

    assert(nData <= 17179869183);

    volatile float tmp[4] = {MAX_FLOAT,MAX_FLOAT,MAX_FLOAT,MAX_FLOAT};
    const float* fptr;

    volatile wchar_t itemp[4] = {1,1,1,1};

    asm __volatile__ ("movups %[tmp], %%xmm6 \n\t"
                      "xorps %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmin) to zero
                      "movups %[itemp], %%xmm4 \n\t"
                      "xorps %%xmm3, %%xmm3 \n\t" //contains candidate argmin
                      : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) :  "xmm3", "xmm4", "xmm5", "xmm6");

    for (i=0; (i+4) <= nData; i += 4) {
      fptr = data+i;

      asm __volatile__ ("movaps %[fptr], %%xmm7 \n\t"
                        "movaps %%xmm7, %%xmm0 \n\t"
                        "cmpltps %%xmm6, %%xmm0 \n\t"
                        "blendvps %%xmm7, %%xmm6 \n\t"
                        "blendvps %%xmm3, %%xmm5 \n\t"
                        "paddd %%xmm4, %%xmm3 \n\t"
                        : : [fptr] "m" (fptr[0]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
    }

    asm __volatile__ ("movups %%xmm6, %[tmp] \n\t"
                      "movups %%xmm5, %[itemp] \n\t"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    //std::cerr << "intermediate minval: " << min_val << std::endl;

    for (i=0; i < 4; i++) {
      cur_val = tmp[i];
      //std::cerr << "cur val: " << cur_val << std::endl;
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = 4*itemp[i] + i;
      }
    }

    //std::cerr << "minval: " << min_val << std::endl;

    for (i= nData - (nData % 4); i < nData; i++) {
      cur_val = data[i];
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = i;
      }
    }
#endif
  }

  inline void find_min_and_argmin(const double_A16* data, const size_t nData, double& min_val, size_t& arg_min)
  {

    min_val = MAX_DOUBLE;
    arg_min = MAX_UINT;

    assertAligned16(data);

#if !defined(USE_SSE) || USE_SSE < 4

    if (nData > 0) {
      const double* ptr = std::min_element(data,data+nData);
      min_val = *ptr;
      arg_min = ptr - data;
    }

    // for (i=0; i < nData; i++) {
    //   double cur_val = data[i];

    //   if (cur_val < min_val) {
    //     min_val = cur_val;
    //     arg_min = i;
    //   }
    // }
#elif USE_SSE >= 5

    //use AVX  - align16 is no good, need align32 for aligned moves
    
    assert(sizeof(size_t) == 8);
    
    double val = MAX_DOUBLE;
    const size_t one = 1;
    double inc = *reinterpret_cast<const double*>(&one);
    size_t i = 0;
    const double* dptr;
    
    asm __volatile__ ("vbroadcastsd %[tmp], %%ymm6 \n\t" //ymm6 is min register
                      "vxorpd %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmin) to zero
                      "vbroadcastsd %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                      "vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t" //sets ymm3 (= current set index) to zero
                      : : [tmp] "m" (val), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");    
    
    for (i=0; (i+4) <= nData; i += 4) {
      dptr = data+i;
      
      asm __volatile__ ("vmovupd %[dptr], %%ymm7 \n\t"
                        "vcmpltpd %%ymm6, %%ymm7, %%ymm0 \n\t" //destination is last
                        "vblendvpd %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                        "vblendvpd %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                        "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                        : : [dptr] "m" (dptr[0]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");      
    }

    double tmp[4];
    size_t itemp[4];
    
    asm __volatile__ ("vmovupd %%ymm6, %[tmp] \n\t"
                      "vmovupd %%ymm5, %[itemp]"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    double cur_val;

    for (i=0; i < 4; i++) {
      cur_val = tmp[i];
      //std::cerr << "cur val: " << cur_val << std::endl;
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = 4*itemp[i] + i;
      }
    }

    //std::cerr << "minval: " << min_val << std::endl;

    for (i = nData - (nData % 4); i < nData; i++) {
      cur_val = data[i];
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = i;
      }
    }
#else

    size_t i;
    double cur_val;

    assert(nData < 8589934592);

    volatile double tmp[2] = {MAX_DOUBLE,MAX_DOUBLE};
    const double* dptr;

    volatile wchar_t itemp[4] = {0,1,0,1};


    asm __volatile__ ("movupd %[tmp], %%xmm6 \n\t"
                      "xorpd %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmin) to zero
                      "movupd %[itemp], %%xmm4 \n\t"
                      "xorpd %%xmm3, %%xmm3 \n\t" //contains candidate argmin
                      : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) :  "xmm3", "xmm4", "xmm5", "xmm6");

    for (i=0; (i+2) <= nData; i += 2) {
      dptr = data+i;

      asm __volatile__ ("movapd %[dptr], %%xmm7 \n\t"
                        "movapd %%xmm7, %%xmm0 \n\t"
                        "cmpltpd %%xmm6, %%xmm0 \n\t"
                        "blendvpd %%xmm7, %%xmm6 \n\t"
                        "blendvpd %%xmm3, %%xmm5 \n\t"
                        "paddd %%xmm4, %%xmm3 \n\t"
                        : : [dptr] "m" (dptr[0]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
    }

    asm __volatile__ ("movupd %%xmm6, %[tmp] \n\t"
                      "movupd %%xmm5, %[itemp] \n\t"
                      : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

    assert(itemp[0] == 0);
    assert(itemp[2] == 0);

    for (i=0; i < 2; i++) {
      cur_val = tmp[i];
      //std::cerr << "cur val: " << cur_val << std::endl;
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = 2*itemp[2*i+1] + i;
      }
    }

    //std::cerr << "minval: " << min_val << std::endl;

    if ((nData % 2) == 1) {
      cur_val = data[nData-1];
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = nData-1;
      }
    }

#endif
  }


  inline void mul_array(float_A16* data, const size_t nData, const float constant)
  {
    assertAligned16(data);
    
    size_t i;
#if !defined(USE_SSE) || USE_SSE < 2
    for (i=0; i < nData; i++) { //g++ uses packed avx mul, but after checking alignment
      data[i] *= constant;
    }
#elif USE_SSE >= 5

    // AVX  - align16 is no good, need align32 for aligned moves

    asm __volatile__ ("vbroadcastss %[tmp], %%ymm7 \n\t"
                      : : [tmp] "m" (constant) : "ymm7");

    float* fptr;

    for (i=0; i+8 <= nData; i+=8) {
      fptr = data + i;
      asm volatile ("vmovups %[fptr], %%ymm6 \n\t"
                    "vmulps %%ymm7, %%ymm6, %%ymm6 \n\t"
                    "vmovups %%ymm6, %[fptr] \n\t"
                    : [fptr] "+m" (fptr[0]) : : "ymm6", "memory");
    }

    for (i= nData - (nData % 8); i < nData; i++) {
      data[i] *= constant;
    }   
#else
    float temp[4];
    float* fptr;
    for (i=0; i < 4; i++)
      temp[i] = constant;
    asm volatile ("movaps %[temp], %%xmm7" : : [temp] "m" (temp[0]) : "xmm7" );

    for (i=0; i+4 <= nData; i+=4) {
      fptr = data + i;
      asm volatile ("movaps %[fptr], %%xmm6 \n\t"
                    "mulps %%xmm7, %%xmm6 \n\t"
                    "movaps %%xmm6, %[fptr] \n\t"
                    : [fptr] "+m" (fptr[0]) : : "xmm6", "memory");
    }

    for (i= nData - (nData % 4); i < nData; i++) {
      data[i] *= constant;
    }
#endif
  }

  inline void mul_array(double_A16* data, const size_t nData, const double constant)
  {
    size_t i;
#if !defined(USE_SSE) || USE_SSE < 2
    for (i=0; i < nData; i++) {
      data[i] *= constant;
    }
#elif USE_SSE >= 5

    // AVX  - align16 is no good, need align32 for aligned moves

    asm __volatile__ ("vbroadcastsd %[tmp], %%ymm7 \n\t"
                      : : [tmp] "m" (constant) : "ymm7");

    double* dptr;

    for (i=0; i+4 <= nData; i+=4) {
      dptr = data + i;

      asm volatile ("vmovupd %[dptr], %%ymm6 \n\t"
                    "vmulpd %%ymm7, %%ymm6, %%ymm6 \n\t"
                    "vmovupd %%ymm6, %[dptr] \n\t"
                    : [dptr] "+m" (dptr[0]) : : "ymm6", "memory");
    }

    for (i= nData - (nData % 4); i < nData; i++) 
      data[i] *= constant;

#else
    double temp[2];
    double* dptr;
    for (i=0; i < 2; i++)
      temp[i] = constant;
    asm volatile ("movupd %[temp], %%xmm7" : : [temp] "m" (temp[0]) : "xmm7" );

    for (i=0; i+2 <= nData; i+=2) {
      dptr = data + i;

      asm volatile ("movapd %[dptr], %%xmm6 \n\t"
                    "mulpd %%xmm7, %%xmm6 \n\t"
                    "movupd %%xmm6, %[dptr] \n\t"
                    : [dptr] "+m" (dptr[0]) : : "xmm6", "memory");
    }

    for (i= nData - (nData % 2); i < nData; i++) 
      data[i] *= constant;
#endif
  }

  //performs data[i] -= factor*data2[i] for each i
  //this is a frequent operation in the conjugate gradient algorithm
  inline void array_subtract_multiple(double_A16* attr_restrict data, const size_t nData, double factor,
                                      const double_A16* attr_restrict data2)
  {
    assertAligned16(data);
    
    size_t i;
#if !defined(USE_SSE) || USE_SSE < 2
    for (i=0; i < nData; i++)
      data[i] -= factor*data2[i];
#elif USE_SSE >= 5

    // AVX  - align16 is no good, need align32 for aligned moves

    asm __volatile__ ("vbroadcastsd %[tmp], %%ymm7 \n\t"
                      : : [tmp] "m" (factor) : "ymm7");
                      
    double* dptr;
    const double* cdptr;

    for (i=0; i+4 <= nData; i+=4) {
      cdptr = data2+i;
      dptr = data+i;

      //TODO: check for usage of VFNADD

      asm volatile ("vmovupd %[cdptr], %%ymm6 \n\t"
                    "vmulpd %%ymm7, %%ymm6, %%ymm6 \n\t" //destination goes last
                    "vmovupd %[dptr], %%ymm5 \n\t"
                    "vsubpd %%ymm6, %%ymm5, %%ymm5 \n\t" //destination goes last
                    "vmovupd %%ymm5, %[dptr] \n\t"
                    : [dptr] "+m" (dptr[0]) : [cdptr] "m" (cdptr[0]) : "ymm5", "ymm6", "memory");
    }

    for (i= nData - (nData % 4); i < nData; i++)
      data[i] -= factor*data2[i];                      
#else
    double temp[2];
    double* dptr;
    const double* cdptr;
    for (i=0; i < 2; i++)
      temp[i] = factor;
    asm volatile ("movupd %[temp], %%xmm7" : : [temp] "m" (temp[0]) : "xmm7" );
    for (i=0; i+2 <= nData; i+=2) {
      cdptr = data2+i;
      dptr = data+i;

      asm volatile ("movapd %[cdptr], %%xmm6 \n\t"
                    "mulpd %%xmm7, %%xmm6 \n\t"
                    "movapd %[dptr], %%xmm5 \n\t"
                    "subpd %%xmm6, %%xmm5 \n\t"
                    "movapd %%xmm5, %[dptr] \n\t"
                    : [dptr] "+m" (dptr[0]) : [cdptr] "m" (cdptr[0]) : "xmm5", "xmm6", "memory");
    }

    for (i= nData - (nData % 2); i < nData; i++)
      data[i] -= factor*data2[i];
#endif
  }
  
  inline void array_add_multiple(double_A16* attr_restrict data, const size_t nData, double factor,
                                 const double_A16* attr_restrict data2) 
  {                                      
    array_subtract_multiple(data, nData, -factor, data2);
  }
  
  //NOTE: despite attr_restrict, you can safely pass the same for dest and src1 or src2
  inline void go_in_neg_direction(double_A16* attr_restrict dest, const size_t nData, const double_A16* attr_restrict src1,
                                  const double_A16* attr_restrict src2, double alpha)
  {
    assertAligned16(dest);
    
#if !defined(USE_SSE) || USE_SSE < 5
    for (size_t i=0; i < nData; i++)
      dest[i] = src1[i] - alpha * src2[i];
#else

    // AVX  - align16 is no good, need align32 for aligned moves
  
    asm __volatile__ ("vbroadcastsd %[w1], %%ymm0 \n\t" //ymm0 = w1
                      : : [w1] "m" (alpha) : "ymm0");
    
    double* dest_ptr;
    const double* s1_ptr;
    const double* s2_ptr;
    size_t i;
    for (i=0; i+4 <= nData; i+= 4) {
      dest_ptr = dest + i;
      s1_ptr = src1 + i;
      s2_ptr = src2 + i;

      asm volatile ("vmovupd %[s2_ptr], %%ymm3 \n\t"
#if 0      
                    "vmulpd %%ymm0, %%ymm3, %%ymm3 \n\t" //destination goes last
                    "vmovupd %[s1_ptr], %%ymm2 \n\t"
                    "vsubpd %%ymm3, %%ymm2, %%ymm2 \n\t" //destination goes last
#else
                    "vmovupd %[s1_ptr], %%ymm2 \n\t"
                    "vfnmadd231pd %%ymm0, %%ymm3, %%ymm2 \n\t" //destination goes last
#endif  
                    "vmovupd %%ymm2, %[dest]"
                    : [dest] "+m" (dest_ptr[0]) : [s1_ptr] "m" (s1_ptr[0]), [s2_ptr] "m" (s2_ptr[0]) : "ymm2", "ymm3", "memory");      
    }    
 
    for (i= nData - (nData % 4); i < nData; i++) 
      dest[i] = src1[i] - alpha * src2[i]; 
#endif
  }                                    

  //NOTE: despite attr_restrict, you can safely pass the same for dest and src1 or src2
  inline void assign_weighted_combination(double_A16* attr_restrict dest, const size_t nData, double w1, const double_A16* attr_restrict src1,
                                          double w2, const double_A16* attr_restrict src2) 
  {
#if !defined(USE_SSE) || USE_SSE < 5
    for (size_t i=0; i < nData; i++)
      dest[i] = w1 * src1[i] + w2 * src2[i];
#else
    //use AVX
  
    asm __volatile__ ("vbroadcastsd %[w1], %%ymm0 \n\t" //ymm0 = w1
                      "vbroadcastsd %[w2], %%ymm1 \n\t" //ymm1 = w2
                      : : [w1] "m" (w1), [w2] "m" (w2) : "ymm0", "ymm1");

    double* dest_ptr;
    const double* s1_ptr;
    const double* s2_ptr;
    size_t i;
    for (i=0; i+4 <= nData; i+= 4) {
      dest_ptr = dest + i;
      s1_ptr = src1 + i;
      s2_ptr = src2 + i;
            
      asm volatile ("vmovupd %[s1_ptr], %%ymm2 \n\t"
                    "vmulpd %%ymm0, %%ymm2, %%ymm2 \n\t" //destination goes last
                    "vmovupd %[s2_ptr], %%ymm3 \n\t"
#if 0
                    "vmulpd %%ymm1, %%ymm3, %%ymm3 \n\t" //destination goes last
                    "vaddpd %%ymm3, %%ymm2, %%ymm2 \n\t" //destination goes last
#else
                    "vfmadd231pd %%ymm3, %%ymm1, %%ymm2 \n\t" //destination goes last
#endif  
                    "vmovupd %%ymm2, %[dest]"
                    : [dest] "+m" (dest_ptr[0]) : [s1_ptr] "m" (s1_ptr[0]), [s2_ptr] "m" (s2_ptr[0]) : "ymm2", "ymm3", "memory");
    }
    
    for (i= nData - (nData % 4); i < nData; i++) 
      dest[i] = w1 * src1[i] + w2 * src2[i];
#endif
  }  

  //binary search, returns MAX_UINT if key is not found, otherwise the position in the vector
  template<typename T>
  size_t binsearch(const T* data, T key, const size_t nData)
  {
    if (nData == 0 || key < data[0] || key > data[nData-1])
      return MAX_UINT;

    size_t lower = 0;
    size_t upper = nData-1;
    if (data[lower] == key)
      return lower;
    if (data[upper] == key)
      return upper;

    while (lower+1 < upper) {
      assert(data[lower] < key);
      assert(data[upper] > key);

      size_t middle = (lower+upper)/2;
      assert(middle > lower && middle < upper);
      if (data[middle] == key)
        return middle;
      else if (data[middle] < key)
        lower = middle;
      else
        upper = middle;
    }

    return MAX_UINT;
  }

  template<typename T>
  size_t binsearch_insertpos(const T* data, T key, const size_t nData)
  {
    if (nData == 0 || key <= data[0])
      return 0;

    if (key > data[nData-1])
      return nData;
  
    size_t lower = 0;
    size_t upper = nData-1;
    if (data[upper] == key)
      return upper;

    while (lower+1 < upper) {
      assert(data[lower] < key);
      assert(data[upper] > key);

      size_t middle = (lower+upper) >> 1;  // (lower+upper)/2;
      assert(middle > lower && middle < upper);
      if (data[middle] == key)
        return middle;
      else if (data[middle] < key)
        lower = middle;
      else
        upper = middle;
    }

    assert(lower+1 == upper);
    return upper;  
  }

  template<typename T1, typename T2>
  class first_lower {
  public:
    bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2)
    {
      return (p1.first < p2.first);
    }
  };

  template<typename T1, typename T2>
  class first_higher {
  public:
    bool operator()(const std::pair<T1,T2>& p1, const std::pair<T1,T2>& p2)
    {
      return (p1.first > p2.first);
    }
  };


} //end of namespace Makros



#endif
