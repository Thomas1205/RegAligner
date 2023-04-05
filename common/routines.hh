/****** written by Thomas Schoenemann as a private person. moved here in February 2020 from makros.hh ****/
/****** Some basic routines, plus binary search algorithms ******/

#ifndef ROUTINES_HH
#define ROUTINES_HH

#include "makros.hh"
#include <algorithm> //for reverse, max, min, find
#include <cassert>

//Explanations for USE_SSE:
// - values 1-4 are SSE1-4 (note that x86_64 always has at least 2), 4 includes 4.1 and 4.2
// - 5 is AVX with 256 bit
// - 6 is FMA
// - 7 is AVX2. You need it for 256 bit packed integer math

//NOTE: [Kusswurm, Modern X86 assembly language programming] recommends to use 128-bit AVX instructions instead of SSE, even though they often list one or two more arguments
//  My own experiments have shown that that does not require more space in the executable. But rigorously moving to AVX is TODO

//NOTE: imul and umul produce the exact same low integer. Only the high one differs

//NOTE: [Intel® 64 and IA-32 Architectures Software Developer’s Manual Vol 2.] clarifies that use of z/y/xmm8-15 and r8-15 makes the instructions and hence the executable longer

//  In contrast to the Intel manuals, AT&T syntax is inverted. That means that the destination is the last argument.
//   As for addressing: (https://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html)
//    Intel mov     eax,[ecx]            AT&T  movl    (%ecx),%eax
//    Intel mov     eax,[ebx+3]          AT&T  movl    3(%ebx),%eax
//    Intel add     eax,[ebx+ecx*2h]     AT&T  addl    (%ebx,%ecx,0x2),%eax


//NOTES on instruction lengths for equivalent instructions (some VEXes are 3 bytes, others 2 bytes):
// vmodupd (VEX.128.66.0F.WIG 10) and vmovdqu (VEX.128.F3.0F.WIG 6F) seem longer than vmovups (VEX.128.0F.WIG 10). But experiments show they have equal length

//NOTE: in C++20, register will be removed -> we remove it already

#ifdef USE_ASM
static_assert(sizeof(uint) == 4, "wrong size");
static_assert(sizeof(int) == 4, "wrong size");
static_assert(sizeof(size_t) == 8, "wrong size");
static_assert(sizeof(Int64) == 8, "wrong size");
static_assert(sizeof(UInt64) == 8, "wrong size");
static_assert(sizeof(float) == 4, "wrong size");
static_assert(sizeof(double) == 8, "wrong size");
#endif

namespace Routines {

  /***************************** declarations ******************************/

  /***************** reverse *******************/

  template<typename T, typename Swap = SwapOp<T> >
  inline void reverse(T* data, const size_t nData) noexcept;

  /***************** downshift *****************/

  template<typename T>
  inline void downshift_array(T* data, const uint pos, const uint shift, const uint nData) noexcept;

  /***************** upshift *****************/

  template<typename T>
  inline void upshift_array(T* data, const int pos, const int last, const int shift) noexcept;

  /***************** binary search in sorted data *************/

  template<typename T, typename Less = std::less<T>, typename Equal = std::equal_to<T>, typename ST = size_t>
  inline ST binsearch(const T* data, const T key, const ST nData) noexcept;

  template<typename T, typename Less = std::less<T>, typename Equal = std::equal_to<T>, typename ST = size_t>
  inline ST binsearch_insertpos(const T* data, const T key, const ST nData) noexcept;

  template<typename T, typename ST, typename Less = std::less<T>, typename Equal = std::equal_to<T>  >
  inline ST index_binsearch_insertpos(const T* data, const T key, const ST* index, const ST nData) noexcept;


  /***************** find unique *****************/

  //if T has size 1,2,4 or 8 this is always a bit-based comparison
  template<typename T, typename Equal = std::equal_to<T> >
  inline uint find_unique(const T* data, const T key, const uint nData) noexcept;

  inline uint find_unique_uint(const uint* data, const uint key, const uint nData) noexcept;

  inline uint find_unique_int(const int* data, const int key, const uint nData) noexcept;

  //non-standard treatment of NAN and INF, just bit-comparisons
  inline uint find_unique_float(const float* data, const float key, const uint nData) noexcept;

  /***************** find first *****************/

  //if T has size 1,2,4 or 8 this is always a bit-based comparison
  template<typename T, typename Equal = std::equal_to<T> >
  inline uint find_first(const T* data, const T key, const uint nData) noexcept;

  inline uint find_first_uint(const uint* data, const uint key, const uint nData) noexcept;

  inline uint find_first_int(const int* data, const int key, const uint nData) noexcept;

  /***************** contains *****************/

  inline bool contains_nan(const double_A16* data, const size_t nData) noexcept;

  inline bool contains_nan(const float_A16* data, const size_t nData) noexcept;

  //if T has size 1,2,4 or 8 this is always a bit-based comparison
  template<typename T>
  inline bool contains(const T* data, const T key, const size_t nData) noexcept;

  inline bool contains_uchar(const uchar* data, const uchar key, const size_t nData) noexcept;

  inline bool contains_ushort(const ushort* data, const ushort key, const size_t nData) noexcept;

  inline bool contains_uint(const uint* data, const uint key, const size_t nData) noexcept;

  inline bool contains_uint64(const UInt64* data, const UInt64 key, const size_t nData) noexcept;

  /***************** equals ****************/

  //if T has size 1,2,4 or 8 this is always a bit-based comparison
  template<typename T>
  inline bool equals(const T* data1, const T* data2, const size_t nData) noexcept;

  inline bool equals_uchar(const uchar* data1, const uchar* data2, const size_t nData) noexcept;

  inline bool equals_ushort(const ushort* data1, const ushort* data2, const size_t nData) noexcept;

  inline bool equals_uint(const uint* data1, const uint* data2, const size_t nData) noexcept;

  inline bool equals_uint64(const UInt64* data1, const UInt64* data2, const size_t nData) noexcept;

  /***************** min, max, min+arg_min, max+arg_max *******/

  inline void find_max_and_argmax(const double_A16* data, const size_t nData, double& max_val, size_t& arg_max) noexcept;

  inline void find_min_and_argmin(const double_A16* data, const size_t nData, double& min_val, size_t& arg_max) noexcept;

  inline void find_max_and_argmax(const float_A16* data, const size_t nData, float& max_val, size_t& arg_max) noexcept;

  inline void find_min_and_argmin(const float_A16* data, const size_t nData, float& min_val, size_t& arg_max) noexcept;

  /***************** array additions with multiplications *************/

  //performs data[i] -= factor*data2[i] for each i
  //this is a frequent operation in the conjugate gradient algorithm
  inline void array_subtract_multiple(double_A16* attr_restrict data, const size_t nData, double factor,
                                      const double_A16* attr_restrict data2) noexcept;

  inline void array_add_multiple(double_A16* attr_restrict data, const size_t nData, double factor,
                                 const double_A16* attr_restrict data2) noexcept;

  //NOTE: despite attr_restrict, you can safely pass the same for dest and src1 or src2
  inline void go_in_neg_direction(double_A16* attr_restrict dest, const size_t nData, const double_A16* attr_restrict src1,
                                  const double_A16* attr_restrict src2, double alpha) noexcept;

  //NOTE: despite attr_restrict, you can safely pass the same for dest and src1 or src2
  inline void assign_weighted_combination(double_A16* attr_restrict dest, const size_t nData, double w1, const double_A16* attr_restrict src1,
                                          double w2, const double_A16* attr_restrict src2) noexcept;


  /***************************** implementation ******************************/

  /***************** reverse *******************/

  //no plans for specialized 16 routines at present

  inline void nontrivial_reverse_byte_array(uchar* data, const size_t nData) noexcept
  {
    assert(nData >= 2);
#if !defined(USE_SSE) || USE_SSE < 5
    std::reverse(data, data + nData);
#else
    if (nData < 4)
      std::swap(data[0], data[nData-1]);
    else if (nData == 4) {

      uint temp;
      asm __volatile__ ("movbel %[d], %[reg] \n\t"
                        "movl %[reg], %[d] \n\t"
                        : [d] "+m" (data[0]), [reg] "=r" (temp) : : "memory");
    }
    else if (nData < 8) {

      ushort temp1;
      ushort temp2;

      size_t low = 0;
      size_t high = nData - 2;
      for (; low + 2 <= high; low += 2, high -= 2) {
        asm __volatile__ ("movbew %[d1], %[reg1] \n\t"
                          "movbew %[d2], %[reg2] \n\t"
                          "movw %[reg1], %[d2] \n\t"
                          "movw %[reg2], %[d1] \n\t"
                          : [d1] "+m" (data[low]), [reg1] "=r" (temp1), [d2] "+m" (data[high]), [reg2] "=r" (temp2) : : "memory");
      }

      if (low+1 == high)
        std::swap(data[low],data[low+2]);
      if (low == high) {

        asm __volatile__ ("movbew %[d], %[reg] \n\t"
                          "movw %[reg], %[d] \n\t"
                          : [d] "+m" (data[low]), [reg] "=r" (temp1) : : "memory");
      }
    }
    else {

      uint temp1;
      uint temp2;

      size_t low = 0;
      size_t high = nData - 4;
      for (; low + 4 <= high; low += 4, high -= 4) {
        asm __volatile__ ("movbel %[d1], %[reg1] \n\t"
                          "movbel %[d2], %[reg2] \n\t"
                          "movl %[reg1], %[d2] \n\t"
                          "movl %[reg2], %[d1] \n\t"
                          : [d1] "+m" (data[low]), [reg1] "=r" (temp1), [d2] "+m" (data[high]), [reg2] "=r" (temp2) : : "memory");
      }

      ushort tu1;
      ushort tu2;
      high += 2;
      for (; low + 2 <= high; low += 2, high -= 2) {
        asm __volatile__ ("movbew %[d1], %[reg1] \n\t"
                          "movbew %[d2], %[reg2] \n\t"
                          "movw %[reg1], %[d2] \n\t"
                          "movw %[reg2], %[d1] \n\t"
                          : [d1] "+m" (data[low]), [reg1] "=r" (tu1), [d2] "+m" (data[high]), [reg2] "=r" (tu2) : : "memory");
      }

      if (low+1 == high)
        std::swap(data[low],data[low+2]);
      if (low == high) {
        asm __volatile__ ("movbew %[d], %[reg] \n\t"
                          "movw %[reg], %[d] \n\t"
                          : [d] "+m" (data[low]), [reg] "=r" (tu1) : : "memory");
      }
    }
#endif
  }

  inline void reverse_byte_array(uchar* data, const size_t nData) noexcept
  {
#if !defined(USE_SSE) || USE_SSE < 5
    std::reverse(data, data + nData);
#else
    if (nData < 2)
      return;
    nontrivial_reverse_byte_array(data, nData);
#endif
  }

  inline void nontrivial_reverse_uint_array(uint* data, const size_t nData) noexcept
  {
    assert(nData >= 2);
#if !defined(USE_SSE) || USE_SSE < 5
    std::reverse(data, data + nData);
#else
    if (nData < 4)
      std::swap(data[0], data[nData-1]);
    else if (nData == 4) {
      asm __volatile__ ("vmovdqu %[d], %%xmm0 \n\t"
                        "vpshufd $27, %%xmm0, %%xmm0 \n\t"
                        "vmovdqu %%xmm0, %[d] \n\t"
                        : [d] "+m" (data[0]) : : "xmm0", "memory");
    }
    else if (nData < 8)
      std::reverse(data, data + nData);
    else {

      size_t low = 0;
      size_t high = nData - 4;
      for (; low + 4 <= high; low += 4, high -= 4) {

        // lowest gets 3, second lowest 2, third lowest 1, highest 0 => immediate byte = 3 + 2*4 + 1*16 = 27
        asm __volatile__ ("vmovdqu %[l], %%xmm0 \n\t"
                          "vmovdqu %[h], %%xmm1 \n\t"
                          "vpshufd $27, %%xmm0, %%xmm0 \n\t"
                          "vpshufd $27, %%xmm1, %%xmm1 \n\t"
                          "vmovdqu %%xmm1, %[l] \n\t"
                          "vmovdqu %%xmm0, %[h] \n\t"
                          : [l] "+m" (data[low]), [h] "+m" (data[high]) : : "xmm0", "xmm1", "memory");
      }

      high += 3;
      //std::cerr << "low: " << low << ", high: " << high << std::endl;
      for (; low < high; low++, high--) {
        if (low + 3 == high) {
          asm __volatile__ ("vmovdqu %[d], %%xmm0 \n\t"
                            "vpshufd $27, %%xmm0, %%xmm0 \n\t"
                            "vmovdqu %%xmm0, %[d] \n\t"
                            : [d] "+m" (data[low]) : : "xmm0", "memory");
          break;
        }
        std::swap(data[low],data[high]);
      }
    }
#endif
  }

  inline void reverse_uint_array(uint* data, const size_t nData) noexcept
  {
#if !defined(USE_SSE) || USE_SSE < 5
    std::reverse(data, data + nData);
#else
    if (nData < 2)
      return;
    nontrivial_reverse_uint_array(data, nData);
#endif
  }

  inline void reverse_4doubles(double* data) noexcept
  {

    asm __volatile__ ("vmovupd %[d], %%ymm0 \n\t"
                      "vperm2f128 $1, %%ymm0, %%ymm0, %%ymm0 \n\t"
                      "vpermilpd $5, %%ymm0, %%ymm0 \n\t"
                      "vmovupd %%ymm0, %[d] \n\t"
                      : [d] "+m" (data[0]) : : "ymm0", "memory");
  }

  inline void nontrivial_reverse_double_array(double* data, const size_t nData) noexcept
  {
    assert(nData >= 2);
#if !defined(USE_SSE) || USE_SSE < 5
    std::reverse(data, data + nData);
#else
    if (nData < 4)
      std::swap(data[0], data[nData-1]);
    else if (nData == 4)
      reverse_4doubles(data);
    else if (nData < 8)
      std::reverse(data, data + nData);
    else {

      size_t low = 0;
      size_t high = nData - 4;
      for (; low + 4 <= high; low += 4, high -= 4) {

        asm __volatile__ ("vmovupd %[l], %%ymm0 \n\t"
                          "vmovupd %[h], %%ymm1 \n\t"
                          "vperm2f128 $1, %%ymm0, %%ymm0, %%ymm0 \n\t"
                          "vperm2f128 $1, %%ymm1, %%ymm1, %%ymm1 \n\t"
                          "vpermilpd $5, %%ymm0, %%ymm0 \n\t"
                          "vpermilpd $5, %%ymm1, %%ymm1 \n\t"
                          "vmovupd %%ymm1, %[l] \n\t"
                          "vmovupd %%ymm0, %[h] \n\t"
                          : [l] "+m" (data[low]), [h] "+m" (data[high]) : : "ymm0", "ymm1", "memory");
      }

      high += 3;
      for (; low < high; low++, high--) {
        if (low + 3 == high) {
          reverse_4doubles(data + low);
          break;
        }
        std::swap(data[low],data[high]);
      }
    }
#endif
  }

  inline void reverse_double_array(double* data, const size_t nData) noexcept
  {
#if !defined(USE_SSE) || USE_SSE < 5
    std::reverse(data, data + nData);
#else
    if (nData < 2)
      return;
    nontrivial_reverse_double_array(data, nData);
#endif
  }

  template<typename T, typename Swap>
  inline void nontrivial_reverse(T* data, const size_t nData) noexcept
  {
    assert(nData >= 2);
    if (sizeof(T) == 1)
      nontrivial_reverse_byte_array((uchar*) data, nData);
    else if (sizeof(T) == 4)
      nontrivial_reverse_uint_array((uint*) data, nData);
    else if (sizeof(T) == 8)
      nontrivial_reverse_double_array((double*) data, nData);
    else {
      const static Swap swapobj;

      size_t start = 0;
      size_t end = nData-1;
      while (start < end) {
        swapobj(data[start],data[end]);
        start++;
        end--;
      }
    }
  }

  template<>
  inline void nontrivial_reverse<uint,SwapOp<uint> >(uint* data, const size_t nData) noexcept
  {
    nontrivial_reverse_uint_array(data, nData);
  }

  template<>
  inline void nontrivial_reverse<int,SwapOp<int> >(int* data, const size_t nData) noexcept
  {
    nontrivial_reverse_uint_array((uint*) data, nData);
  }

  template<>
  inline void nontrivial_reverse<float,SwapOp<float> >(float* data, const size_t nData) noexcept
  {
    nontrivial_reverse_uint_array((uint*) data, nData);
  }

  template<>
  inline void nontrivial_reverse<double,SwapOp<double> >(double* data, const size_t nData) noexcept
  {
    nontrivial_reverse_double_array(data, nData);
  }

  template<>
  inline void nontrivial_reverse<Int64,SwapOp<Int64> >(Int64* data, const size_t nData) noexcept
  {
    nontrivial_reverse_double_array((double*) data, nData);
  }

  template<>
  inline void nontrivial_reverse<UInt64,SwapOp<UInt64> >(UInt64* data, const size_t nData) noexcept
  {
    nontrivial_reverse_double_array((double*) data, nData);
  }

  template<typename T, typename Swap = SwapOp<T> >
  inline void reverse(T* data, const size_t nData) noexcept
  {
    if (nData >= 2)
      nontrivial_reverse<T,Swap>(data, nData);
  }

  /***************** downshift *****************/

  inline void downshift_uint_array(uint* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    assert(shift <= nData);
    uint i = pos;
    const uint end = nData-shift;
#if !defined(USE_SSE) || USE_SSE < 2
    //for (; i < end; i++)
    //  data[i] = data[i+shift];
    memmove(data+pos,data+pos+shift,(end-pos+shift-1)*sizeof(uint));
#else

    //roughly the same performance as memmove

#if USE_SSE >= 5
    for (; i + 8 <= end; i += 8) {
      uint* out_ptr = data+i;
      const uint* in_ptr = out_ptr + shift;

      asm __volatile__ ("vmovdqu %[inp], %%ymm7 \n\t"
                        "vmovdqu %%ymm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "ymm7", "memory");
    }
#endif

    //movdqu is SSE2
    for (; i + 4 <= end; i += 4) {
      uint* out_ptr = data+i;
      const uint* in_ptr = out_ptr + shift;

      asm __volatile__ ("movdqu %[inp], %%xmm7 \n\t"
                        "movdqu %%xmm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "xmm7", "memory");
    }
    for (; i < end; i++) {
      data[i] = data[i+shift];
    }
#endif
  }

  inline void downshift_int_array(int* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    static_assert(sizeof(int) == sizeof(uint), "wrong size");
    downshift_uint_array((uint*) data, pos, shift, nData);
  }

  inline void downshift_float_array(float* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    static_assert(sizeof(int) == sizeof(uint), "wrong size");
    downshift_uint_array((uint*) data, pos, shift, nData);
  }

  inline void downshift_double_array(double* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    uint i = pos;
    const uint end = nData-shift;
#if !defined(USE_SSE) || USE_SSE < 2
    //for (; i < end; i++)
    //  data[i] = data[i+shift];
    memmove(data+pos,data+pos+shift,(end-pos+shift-1)*sizeof(double));
#else

    //roughly the same performance as memmove

#if USE_SSE >= 5
    for (; i + 4 <= end; i += 4) {

      double* out_ptr = data+i;
      const double* in_ptr = out_ptr + shift;

      asm __volatile__ ("vmovupd %[inp], %%ymm7 \n\t"
                        "vmovupd %%ymm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "ymm7", "memory");
    }
#endif

    //movupd is SSE2
    for (; i + 2 <= end; i += 2) {

      double* out_ptr = data+i;
      const double* in_ptr = out_ptr + shift;

      asm __volatile__ ("movupd %[inp], %%xmm7 \n\t"
                        "movupd %%xmm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "xmm7", "memory");
    }
    for (; i < end; i++)
      data[i] = data[i+shift];
#endif
  }

  template<typename T>
  inline void downshift_array(T* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    //c++-20 offers shift_left and shift_right in <algorithm>
    const uint end = nData-shift;
    if (std::is_trivially_copyable<T>::value) {
      memmove((void*) (data+pos), (void*) (data+pos+shift),(end-pos+shift-1)*sizeof(T));
    }
    else {
      uint i = pos;
      for (; i < end; i++)
        data[i] = std::move(data[i+shift]);
    }
  }

  template<>
  inline void downshift_array(char* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    //test of the specialized routine downshift_double_array is TODO
    const uint end = nData-shift;
    memmove((void*) (data+pos), (void*) (data+pos+shift),(end-pos+shift-1)*sizeof(char));
  }

  template<>
  inline void downshift_array(uchar* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    //test of the specialized routine downshift_double_array is TODO
    const uint end = nData-shift;
    memmove((void*) (data+pos),data+pos+shift,(end-pos+shift-1)*sizeof(uchar));
  }

  template<>
  inline void downshift_array(short* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    //test of the specialized routine downshift_ushort_array is TODO
    const uint end = nData-shift;
    memmove((void*) (data+pos), (void*) (data+pos+shift),(end-pos+shift-1)*sizeof(short));
  }

  template<>
  inline void downshift_array(ushort* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    //test of the specialized routine downshift_double_array is TODO
    const uint end = nData-shift;
    memmove((void*) (data+pos), (void*) (data+pos+shift),(end-pos+shift-1)*sizeof(ushort));
  }

  template<>
  inline void downshift_array(uint* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    downshift_uint_array(data, pos, shift, nData);
  }

  template<>
  inline void downshift_array(int* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    downshift_int_array(data, pos, shift, nData);
  }

  template<>
  inline void downshift_array(float* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    downshift_float_array(data, pos, shift, nData);
  }

  template<>
  inline void downshift_array(double* data, const uint pos, const uint shift, const uint nData) noexcept
  {
    //test of the specialized routine downshift_double_array is TODO
    const uint end = nData-shift;
    memmove((void*) (data+pos), (void*) (data+pos+shift),(end-pos+shift-1)*sizeof(double));
  }

  /***************** upshift *****************/

  inline void upshift_uint_array(uint* data, const int pos, const int last, const int shift) noexcept
  {
    assert(shift > 0);
    int k = last;
#if !defined(USE_SSE) || USE_SSE < 2
    //for (; k >= pos+shift; k--)
    //  data[k] = data[k-shift];
    memmove((void*) (data+pos+shift), (void*) (data+pos),(last-pos-shift+1)*sizeof(uint));
#else

    //roughly the same performance as memmove

#if USE_SSE >= 5
    for (; k-7 >= pos+shift; k -= 8) {
      uint* out_ptr = data + k - 7;
      const uint* in_ptr = out_ptr - shift;

      asm __volatile__ ("vmovdqu %[inp], %%ymm7 \n\t"
                        "vmovdqu %%ymm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "ymm7", "memory");
    }
#endif

    //movdqu is SSE2
    for (; k-3 >= pos+shift; k -= 4) {
      uint* out_ptr = data + k - 3;
      const uint* in_ptr = out_ptr - shift;

      asm __volatile__ ("movdqu %[inp], %%xmm7 \n\t"
                        "movdqu %%xmm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "xmm7", "memory");
    }

    for (; k >= pos+shift; k--)
      data[k] = data[k-shift];
#endif
  }

  inline void upshift_int_array(int* data, const int pos, const int last, const int shift) noexcept
  {
    static_assert(sizeof(int) == sizeof(uint), "wrong size");
    upshift_uint_array((uint*) data, pos, last, shift);
  }

  inline void upshift_float_array(float* data, const int pos, const int last, const int shift) noexcept
  {
    static_assert(sizeof(float) == sizeof(uint), "wrong size");
    upshift_uint_array((uint*) data, pos, last, shift);
  }

  inline void upshift_double_array(double* data, const int pos, const int last, const int shift) noexcept
  {
    assert(shift > 0);
    int k = last;
#if !defined(USE_SSE) || USE_SSE < 2
    //for (; k >= pos+shift; k--)
    //  data[k] = data[k-shift];
    memmove((void*) (data+pos+shift), (void*) (data+pos),(last-pos-shift+1)*sizeof(double));
#else

    //roughly the same speed as memove

#if USE_SSE >= 5
    for (; k-4-shift > pos; k -= 4) {
      double* out_ptr = data + k - 3;
      const double* in_ptr = out_ptr - shift;

      asm __volatile__ ("vmovupd %[inp], %%ymm7 \n\t"
                        "vmovupd %%ymm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "ymm7", "memory");
    }
#endif

    //movupd is SSE2
    for (; k-2 > pos+shift; k -= 2) {
      double* out_ptr = data + k - 1;
      const double* in_ptr = out_ptr - shift;

      asm __volatile__ ("movupd %[inp], %%xmm7 \n\t"
                        "movupd %%xmm7, %[outp] \n\t"
                        : [outp] "=m" (out_ptr[0]) : [inp] "m" (in_ptr[0]) : "xmm7", "memory");
    }


    for (; k >= pos+shift; k--)
      data[k] = data[k-shift];
#endif
  }

  template<typename T>
  inline void standard_upshift_array(T* data, const int pos, const int last, const int shift) noexcept
  {
	//std::cerr << "standard_upshift_array for " << data[0] << std::endl;
    for (int k = last; k >= pos+shift; k--) {
	  //std::cerr << "assign " << k << " with " << (k-shift) << std::endl;
      data[k] = std::move(data[k-shift]);
	  //std::cerr << "done" << std::endl;
	}
  }

  template<typename T>
  inline void upshift_array(T* data, const int pos, const int last, const int shift) noexcept
  {
    assert(shift > 0);
    //c++-20 offers shift_left and shift_right in <algorithm>
    if (std::is_trivially_copyable<T>::value) {
      memmove((void*) (data+pos+shift), (void*) (data+pos),(last-pos-shift+1)*sizeof(T));
    }
    else {
      standard_upshift_array(data,pos,last,shift);
    }
  }

  template<>
  inline void upshift_array(uint* data, const int pos, const int last, const int shift) noexcept
  {
    upshift_uint_array(data, pos, last, shift);
  }

  template<>
  inline void upshift_array(int* data, const int pos, const int last, const int shift) noexcept
  {
    upshift_int_array(data, pos, last, shift);
  }

  template<>
  inline void upshift_array(float* data, const int pos, const int last, const int shift) noexcept
  {
    upshift_float_array(data, pos, last, shift);
  }

  template<>
  inline void upshift_array(double* data, const int pos, const int last, const int shift) noexcept
  {
    upshift_double_array(data, pos, last, shift);
  }

  /***************** find unique *****************/

  //data should contain key at most once
  inline uint find_unique_uint(const uint* data, const uint key, const uint nData) noexcept
  {
    uint i = 0;
#if !defined(USE_SSE) || USE_SSE < 5
    for (; i < nData; i++) {
      if (data[i] == key)
        return i;
    }
#else

    //NOTE: if data is not unique, i.e. contains key more than once, this may return incorrect results
    //  if occurences are close together, it may return the sum of their positions

    //NOTE: we can go for ymm registers, if we use VEXTRACTF128 to send down the upper half

    //std::cerr << "find_unique_uint(key: " << key << ")" << std::endl;

    static const uint ind[8] = {0, 1, 2, 3, 4, 5, 6, 7};

#if USE_SSE >= 7
    //need AVX2 for 256 bit packed integers. AVX2 has vpbroadcastd

    if (nData >= 48) {

      const uint uinc = 8;

      asm __volatile__ ("vxorps %%ymm2, %%ymm2, %%ymm2 \n\t" // set ymm2 (the array of found positions) to zero
                        "vmovdqu %[ind], %%ymm3  \n\t" //ymm3 contains the indices
                        "vpbroadcastd %[uinc], %%ymm4 \n\t"
                        "vpbroadcastd %[ukey], %%ymm5 \n\t"
                        : : [ind] "m" (ind[0]), [uinc] "m" (uinc), [ukey] "m" (key)
                        : "ymm2", "ymm3", "ymm4", "ymm5");

      uint res = MAX_UINT;
      for (; i+8 <= nData; i+=8) {

        //NOTE: jumps outside of asm blocks are allowed only in asm goto, but that cannot have outputs (and local variables do not seem to get assembler names)
        // => probably best to write the loop in assembler, too

        //NOTE: an alternative to vptest; jz; would be vmovmskps xmm, cx; jcxz;

        // Assembler wishlist: horizontal min and max -> would make the uniqueness assumption superflous (not even present in AVX-512)

        asm __volatile__ ("vmovdqu %[d], %%ymm0  \n\t"
                          "vpcmpeqd %%ymm5, %%ymm0, %%ymm0 \n\t" //xmm0 is overwritten with mask (all 1s on equal)
                          "vpblendvb %%ymm0, %%ymm3, %%ymm2, %%ymm2  \n\t" //if xmm0 flags 1, the index is written
                          "vptest %%ymm0, %%ymm0 \n\t" //sets the zero flag iff xmm0 is all 0
                          "jz 1f \n\t" //jump if no equals
                          "vextracti128 $1, %%ymm2, %%xmm0 \n\t"
                          "vphaddd %%ymm0, %%ymm2, %%ymm2 \n\t"
                          "vphaddd %%ymm0, %%ymm2, %%ymm2 \n\t" //(ymm0 is irrelevant)
                          "vphaddd %%ymm0, %%ymm2, %%ymm2 \n\t" //(ymm0 is irrelevant)
                          "vpextrd $0, %%xmm2, %0 \n\t" //here xmm!
                          "1: vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t"
                          : "+g" (res) : [d] "m" (data[i]) : "ymm0", "ymm2", "ymm3");

        if (res != MAX_UINT)
          return res;
      }
    }
#endif

    if (nData - i >= 12) {

      const float fkey = reinterpret<const uint, const float>(key);
      const float finc = reinterpret<const uint, const float>(4);

      if (i == 0) {
        asm __volatile__ ("vxorps %%xmm2, %%xmm2, %%xmm2 \n\t" // set xmm2 (the array of found positions) to zero
                          "vmovdqu %[ind], %%xmm3  \n\t" //xmm3 contains the indices
                          "vbroadcastss %[finc], %%xmm4 \n\t"
                          "vbroadcastss %[fkey], %%xmm5 \n\t"
                          : : [ind] "m" (ind[0]), [finc] "m" (finc), [fkey] "m" (fkey)
                          : "xmm2", "xmm3", "xmm4", "xmm5");
      }
      else {
        asm __volatile__ ("vbroadcastss %[finc], %%xmm4 \n\t"
                          : : [finc] "m" (finc) : "xmm4");
      }

      uint res = MAX_UINT;
      for (; i+4 <= nData; i+=4) {

        //NOTE: jumps outside of asm blocks are allowed only in asm goto, but that cannot have outputs (and local variables do not seem to get assembler names)
        // => probably best to write the loop in assembler, too

        //NOTE: an alternative to vptest; jz would be movmskps cx, xmm; jcxz;

        // Assembler wishlist: horizontal min and max -> would make the uniqueness assumption superflous (not even present in AVX-512)

        asm __volatile__ ("vmovdqu %1, %%xmm0  \n\t"
                          "vpcmpeqd %%xmm5, %%xmm0, %%xmm0 \n\t" //xmm0 is overwritten with mask (all 1s on equal)
                          "vpblendvb %%xmm0, %%xmm3, %%xmm2, %%xmm2  \n\t" //if xmm0 flags 1, the index is written
                          "vptest %%xmm0, %%xmm0 \n\t" //sets the zero flag iff xmm0 is all 0
                          "jz 1f \n\t" //jump if no equals
                          "vphaddd %%xmm0, %%xmm2, %%xmm2 \n\t" //(xmm0 is irrelevant)
                          "vphaddd %%xmm0, %%xmm2, %%xmm2 \n\t" //(xmm0 is irrelevant)
                          "vpextrd $0, %%xmm2, %0 \n\t"
                          "1: vpaddd %%xmm4, %%xmm3, %%xmm3 \n\t"
                          : "+g" (res) : "m" (data[i]) : "xmm0", "xmm2", "xmm3");

        if (res != MAX_UINT)
          return res;
      }
    }

    for (; i < nData; i++) {
      if (data[i] == key)
        return i;
    }
#endif
    return MAX_UINT;
  }

  //data should contain key at most once
  inline uint find_unique_int(const int* data, const int key, const uint nData) noexcept
  {
    return find_unique_uint((const uint*) data, (const uint) key, nData);
  }

  inline uint find_unique_float(const float* data, const float key, const uint nData) noexcept
  {
    //NOTE: raw byte equality compare gives non-standard treatment of nan and +/- inf
    return find_unique_uint((const uint*) data, reinterpret<const float, const uint>(key), nData);
  }

  template<typename T, typename Equal>
  inline uint find_unique(const T* data, const T key, const uint nData) noexcept
  {
	//NOTE: we get a problem with compound types that consume more data than they fill. The extra bytes cannot be compared
    if (sizeof(T) == 4 && !std::is_compound<T>::value) {
      //NOTE: this will do bit-based equality comparisons even for floating point types (i.e. non-standard treatment of inf and nan)!
      return find_unique_uint((const uint*) data, reinterpret<const T, const uint>(key), nData);
    }
    else {
      const static Equal equal;
      for (uint i = 0; i < nData; i++) {
        if (equal(data[i],key))
          return i;
      }
      return MAX_UINT;
      //return std::find<T,Equal>(data, data + nData, key) - data; //std::find cannot take Equal!
    }
  }

  /***************** find first *****************/

  inline uint find_first_uint(const uint* data, const uint key, const uint nData) noexcept
  {
    uint i = 0;
#if !defined(USE_SSE) || USE_SSE < 5
    for (; i < nData; i++) {
      if (data[i] == key)
        return i;
    }
#else

    //NOTE: if data is not unique, i.e. contains key more than once, this may return incorrect results
    //  if occurences are close together, it may return the sum of their positions

    //NOTE: we can go for ymm registers, if we use VEXTRACTF128 to send down the upper half

    std::cerr << "find_first_uint(key: " << key << ")" << std::endl;

    if (nData >= 12) {
      static const uint ind[4] = {0, 1, 2, 3};

      const float finc = reinterpret<const uint, const float>(4); //*reinterpret_cast<const float*>(&iinc);
      const float fkey = reinterpret<const uint, const float>(key); //*reinterpret_cast<const float*>(&key);
      const float fmax = reinterpret<const uint, const float>(MAX_UINT);

      asm __volatile__ ("vbroadcastss %[fmax], %%xmm2 \n\t" // set xmm2 (the array of found positions) to MAX_UINT
                        "vmovdqu %[ind], %%xmm3  \n\t" //xmm3 contains the indices
                        "vbroadcastss %[finc], %%xmm4 \n\t"
                        "vbroadcastss %[fkey], %%xmm5 \n\t"
                        : : [ind] "m" (ind[0]), [finc] "m" (finc), [fkey] "m" (fkey), [fmax] "m" (fmax)
                        : "xmm2", "xmm3", "xmm4", "xmm5");

      uint res = MAX_UINT;
      for (; i+4 <= nData; i+=4) {
        //NOTE: jumps outside of asm blocks are allowed only in asm goto, but that cannot have outputs (and local variables do not seem to get assembler names)
        // => probably best to write the loop in assembler, too

        // Assembler wishlist: horizontal min and max -> would save the lengthy manual code

        asm __volatile__ ("vmovdqu %[dat], %%xmm0  \n\t"
                          "vpcmpeqd %%xmm5, %%xmm0, %%xmm0 \n\t" //xmm0 is overwritten with mask (all 1s on equal)
                          "pblendvb %%xmm3, %%xmm2  \n\t" //if xmm0 flags 1, the index is written
                          "vptest %%xmm0, %%xmm0 \n\t" //sets the zero flag iff xmm0 is all 0
                          "jz 1f \n\t" //jump if no equals
                          //manual phmin
                          "vmovhlps %%xmm2, %%xmm1, %%xmm1 \n\t" //move high two ints of xmm2 to low in xmm1
                          //"vpsrldq $8, %%xmm2, %%xmm1 \n\t" //BYTE suffle for the entire reg: move high two ints of xmm2 to low in xmm1: dq is byte shift
                          "pminud %%xmm1, %%xmm2   \n\t"
                          "vpsrldq $4, %%xmm2, %%xmm1 \n\t"
                          "pminud %%xmm1, %%xmm2   \n\t"
                          "vpextrd $0, %%xmm2, %0 \n\t"
                          "1: paddd %%xmm4, %%xmm3 \n\t"
                          : "+g" (res) : [dat] "m" (data[i]) : "xmm0", "xmm1", "xmm2", "xmm3");

        if (res != MAX_UINT)
          return res;
      }
    }

    for (; i < nData; i++) {
      if (data[i] == key)
        return i;
    }
#endif
    return MAX_UINT;
  }

  inline uint find_first_int(const int* data, const int key, const uint nData) noexcept
  {
    return find_first_uint((uint*) data, key, nData);
  }

  template<typename T, typename Equal>
  inline uint find_first(const T* data, const T key, const uint nData) noexcept
  {
	//NOTE: we get a problem with compound types that consume extra bytes. The extra bytes cannot be compared
    if (sizeof(T) == 4 && !std::is_compound<T>::value) {
      //NOTE: this will do bit-based equality comparisons even for floating point types (i.e. non-standard treatment of inf and nan)!
      return find_first_uint((const uint*) data, reinterpret<const T, const uint>(key), nData);
    }
    else {
      const static Equal equal;
      for (uint i = 0; i < nData; i++) {
        if (equal(data[i],key))
          return i;
      }
      return MAX_UINT;
      //return std::find<T,Equal>(data, data + nData, key) - data; //std::find cannot take Equal
    }
  }

  /******** contains *******/

  inline bool contains_nan(const float_A16* data, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (std::isnan(data[i]))
        return true;
    }
#else

    uchar found = 0;

    for (; (i+8) <= nData; i += 8) {

      asm __volatile__ ("vmovups %[fptr], %%ymm7\n\t"
                        "vcmpunordps %%ymm7, %%ymm7, %%ymm7 \n\t"
                        "vptest %%ymm7, %%ymm7 \n\t" //sets the zero flag iff xmm7 is all 0
                        "setnz %[f] \n\t"
                        //"jz 1f \n\t" //jump if no equals
                        //"movl $1, %[f] \n\t"
                        //"1: \n\t"
                        : [f] "+m" (found) : [fptr] "m" (data[i]) : "ymm7");

      if (found != 0)
        return true;
    }

    for (; (i+4) <= nData; i += 4) {

      asm __volatile__ ("vmovaps %[fptr], %%xmm7\n\t"
                        "vcmpunordps %%xmm7, %%xmm7, %%xmm7 \n\t"
                        "vptest %%xmm7, %%xmm7 \n\t" //sets the zero flag iff xmm7 is all 0
                        "setnz %[f] \n\t"
                        //"jz 1f \n\t" //jump if no equals
                        //"movl $1, %[f] \n\t"
                        //"1: \n\t"
                        : [f] "+r" (found) : [fptr] "m" (data[i]) : "xmm7");

      if (found != 0)
        return true;
    }

    for (; i < nData; i++) {
      if (std::isnan(data[i]))
        return true;
    }
#endif

    return false;
  }

  inline bool contains_nan(const double_A16* data, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (std::isnan(data[i]))
        return true;
    }
#else

    uchar found = 0;

    for (; (i+4) <= nData; i += 4) {

      asm __volatile__ ("vmovupd %[fptr], %%ymm7\n\t"
                        "vcmpunordpd %%ymm7, %%ymm7, %%ymm7 \n\t"
                        "vptest %%ymm7, %%ymm7 \n\t" //sets the zero flag iff xmm7 is all 0
                        "setnz %[f] \n\t"
                        //"jz 1f \n\t" //jump if no equals
                        //"movl $1, %[f] \n\t"
                        //"1: \n\t"
                        : [f] "+r" (found) : [fptr] "m" (data[i]) : "ymm7");

      if (found != 0)
        return true;
    }

    for (; i < nData; i++) {
      if (std::isnan(data[i]))
        return true;
    }
#endif

    return false;
  }

  template<typename T>
  inline bool contains(const T* data, const T key, const size_t nData) noexcept
  {
	//NOTE: we get a problem here with compound types. a struct with a uint and a ushort is assigned size 8, but the extra two bytes are not
	//       comparable.
	//std::cerr << "************** Routines::contains" << std::endl;
	//std::cerr << "sizeof(T): " << sizeof(T) << std::endl;
    if (sizeof(T) == 1) {
      return contains_uchar((const uchar*) data, reinterpret<const T, const uchar>(key), nData);
    }
    else if (sizeof(T) == 2 && !std::is_compound<T>::value) {
      return contains_ushort((const ushort*) data, reinterpret<const T, const ushort>(key), nData);
    }
    else if (sizeof(T) == 4 && !std::is_compound<T>::value) {
      //NOTE: this will do bit-based equality comparisons even for floating point types (i.e. non-standard treatment of inf and nan)!
      return contains_uint((const uint*) data, reinterpret<const T, const uint>(key), nData);
    }
    else if (sizeof(T) == 8 && !std::is_compound<T>::value) {
      //NOTE: this will do bit-based equality comparisons even for floating point types (i.e. non-standard treatment of inf and nan)!
      return contains_uint64((const UInt64*) data, reinterpret<const T, const UInt64>(key), nData);
    }
    else {
	  //std::cerr << "-----calling std::find" << std::endl;
      return (std::find(data, data + nData, key) != data + nData);
	}
  }

  inline bool contains_uchar(const uchar* data, const uchar key, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#else

    uchar found = 0;

    if (nData >= 16) {

      const uint u16key = key;
      const uint ukey = u16key + (u16key << 8);
      const float fkey = reinterpret<const uint, const float>(ukey + (ukey << 16) );

      asm __volatile__ ("vbroadcastss %[fkey], %%ymm6 \n\t"
                        : : [fkey] "m" (fkey) : "ymm6");

#if USE_SSE >= 6
      for (; i + 32 <= nData; i += 32) {

        asm __volatile__ ("vmovdqu %[dat], %%ymm7  \n\t"
                          "vpcmpeqb %%ymm6, %%ymm7, %%ymm7 \n\t" //ymm7 is overwritten with mask (all 1s on equal)
                          "vptest %%ymm7, %%ymm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "ymm7");

        if (found != 0)
          return true;
      }
#endif

      for (; i + 16 <= nData; i += 16) {

        asm __volatile__ ("vmovdqu %[dat], %%xmm7  \n\t"
                          "vpcmpeqb %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                          "vptest %%xmm7, %%xmm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "xmm7");

        if (found != 0)
          return true;
      }
    }

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#endif

    return false;
  }

  inline bool contains_ushort(const ushort* data, const ushort key, const size_t nData) noexcept
  {

    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#else

    uchar found = 0;

    if (nData >= 8) {

      const uint ukey = key;
      const float fkey = reinterpret<const uint, const float>(ukey + (ukey << 16) );

      asm __volatile__ ("vbroadcastss %[fkey], %%ymm6 \n\t"
                        : : [fkey] "m" (fkey) : "ymm6");

#if USE_SSE >= 6
      for (; i + 16 <= nData; i += 16) {

        asm __volatile__ ("vmovdqu %[dat], %%ymm7  \n\t"
                          "vpcmpeqw %%ymm6, %%ymm7, %%ymm7 \n\t" //ymm7 is overwritten with mask (all 1s on equal)
                          "vptest %%ymm7, %%ymm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "ymm7");

        if (found != 0)
          return true;
      }
#endif

      for (; i + 8 <= nData; i += 8) {

        asm __volatile__ ("vmovdqu %[dat], %%xmm7  \n\t"
                          "vpcmpeqw %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                          "vptest %%xmm7, %%xmm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "xmm7");

        if (found != 0)
          return true;
      }
    }

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#endif

    return false;
  }

  inline bool contains_uint(const uint* data, const uint key, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#else

    uchar found = 0;

    if (nData >= 8) {

      const float fkey = reinterpret<const uint, const float>(key); //*reinterpret_cast<const float*>(&key);
      asm __volatile__ ("vbroadcastss %[fkey], %%ymm6 \n\t"
                        : : [fkey] "m" (fkey) : "ymm6");

#if USE_SSE >= 6
      for (; i + 8 <= nData; i += 8) {

        asm __volatile__ ("vmovdqu %[dat], %%ymm7  \n\t"
                          "vpcmpeqd %%ymm6, %%ymm7, %%ymm7 \n\t" //ymm7 is overwritten with mask (all 1s on equal)
                          "vptest %%ymm7, %%ymm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "ymm7");

        if (found != 0)
          return true;
      }
#endif

      for (; i + 4 <= nData; i += 4) {

        asm __volatile__ ("vmovdqu %[dat], %%xmm7  \n\t"
                          "vpcmpeqd %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                          "vptest %%xmm7, %%xmm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "xmm7");

        if (found != 0)
          return true;
      }
    }

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#endif

    return false;
  }

  inline bool contains_uint64(const UInt64* data, const UInt64 key, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#else

    uchar found = 0;

    if (nData >= 4) {

      const double dkey = reinterpret<const UInt64, const double>(key); //*reinterpret_cast<const float*>(&key);

      asm __volatile__ ("vbroadcastsd %[dkey], %%ymm6 \n\t"
                        : : [dkey] "m" (dkey) : "ymm6");

#if USE_SSE >= 6
      for (; i + 4 <= nData; i += 4) {

        asm __volatile__ ("vmovdqu %[dat], %%ymm7  \n\t"
                          "vpcmpeqq %%ymm6, %%ymm7, %%ymm7 \n\t" //ymm7 is overwritten with mask (all 1s on equal)
                          "vptest %%ymm7, %%ymm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "ymm7");

        if (found != 0)
          return true;

      }
#endif

      for (; i + 2 <= nData; i += 2) {

        asm __volatile__ ("vmovdqu %[dat], %%xmm7  \n\t"
                          "vpcmpeqq %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                          "vptest %%xmm7, %%xmm7 \n\t" //sets the zero flag iff xmm7 is all 0
                          "setnz %[f] \n\t"
                          : [f] "+r" (found) : [dat] "m" (data[i]) : "xmm7");

        if (found != 0)
          return true;
      }
    }

    for (; i < nData; i++) {
      if (data[i] == key)
        return true;
    }
#endif

    return false;
  }

  /***************** equals ****************/

  //if T has size 1,2,4 or 8 this is always a bit-based comparison
  template<typename T>
  inline bool equals(const T* data1, const T* data2, const size_t nData) noexcept
  {
	//NOTE: we get a problem with compound types that use more bytes than they fill: The extra bytes cannot be compared
    if (sizeof(T) == 1) {
      return equals_uchar((uchar*) data1, (uchar*) data2, nData);
    }
    else if (sizeof(T) == 2 && !std::is_compound<T>::value) {
      return equals_ushort((ushort*) data1, (ushort*) data2, nData);
    }
    else if (sizeof(T) == 4 && !std::is_compound<T>::value) {
      //NOTE: this will do bit-based equality comparisons even for floating point types (i.e. non-standard treatment of inf and nan)!
      return equals_uint((uint*) data1, (uint*) data2, nData);
    }
    else if (sizeof(T) == 8 && !std::is_compound<T>::value) {
      //NOTE: this will do bit-based equality comparisons even for floating point types (i.e. non-standard treatment of inf and nan)!
      return equals_uint64((UInt64*) data1, (UInt64*) data2, nData);
    }
    else {
      for (size_t i = 0; i < nData; i++) {
        if (data1[i] != data2[i])
          return false;
      }
      return true;
    }
  }

  inline bool equals_uchar(const uchar* data1, const uchar* data2, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i + 8 <= nData; i += 8) {

      const UInt64* t1 = (UInt64*) (data1 + i);
      const UInt64* t2 = (UInt64*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i + 4 <= nData; i += 4) {

      const uint* t1 = (uint*) (data1 + i);
      const uint* t2 = (uint*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#else

    uchar found = 0;

    const float fkey = reinterpret<const uint, const float>(0xFFFFFFFF);
    asm __volatile__ ("vbroadcastss %[fkey], %%ymm5 \n\t"
                      : : [fkey] "m" (fkey) : "ymm5");

#if USE_SSE >= 6
    for (; i + 32 <= nData; i += 32) {

      asm __volatile__ ("vmovdqu %[dat1], %%ymm6  \n\t"
                        "vmovdqu %[dat2], %%ymm7  \n\t"
                        "vpcmpeqb %%ymm6, %%ymm7, %%ymm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%ymm5, %%ymm7, %%ymm7 \n\t" //negate via xor with all 1s
                        "vptest %%ymm7, %%ymm7 \n\t" ///sets the zero flag iff (the now negated) ymm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "ymm6", "ymm7");

      if (found != 0)
        return false;
    }
#endif

    for (; i + 16 <= nData; i += 16) {

      asm __volatile__ ("vmovdqu %[dat1], %%xmm6  \n\t"
                        "vmovdqu %[dat2], %%xmm7  \n\t"
                        "vpcmpeqb %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%xmm5, %%xmm7, %%xmm7 \n\t" //negate via xor with all 1s
                        "vptest %%xmm7, %%xmm7 \n\t" ///sets the zero flag iff (the now negated) xmm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "xmm6", "xmm7");

      if (found != 0)
        return false;
    }

    for (; i + 8 <= nData; i += 8) {

      const UInt64* t1 = (UInt64*) (data1 + i);
      const UInt64* t2 = (UInt64*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i + 4 <= nData; i += 4) {

      const uint* t1 = (uint*) (data1 + i);
      const uint* t2 = (uint*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#endif

    return true;
  }

  inline bool equals_ushort(const ushort* data1, const ushort* data2, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i + 4 <= nData; i += 4) {

      const UInt64* t1 = (UInt64*) (data1 + i);
      const UInt64* t2 = (UInt64*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i + 2 <= nData; i += 2) {

      const uint* t1 = (uint*) (data1 + i);
      const uint* t2 = (uint*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#else

    uchar found = 0;

    const float fkey = reinterpret<const uint, const float>(0xFFFFFFFF);
    asm __volatile__ ("vbroadcastss %[fkey], %%ymm5 \n\t"
                      : : [fkey] "m" (fkey) : "ymm5");

#if USE_SSE >= 6
    for (; i + 16 <= nData; i += 16) {

      asm __volatile__ ("vmovdqu %[dat1], %%ymm6  \n\t"
                        "vmovdqu %[dat2], %%ymm7  \n\t"
                        "vpcmpeqw %%ymm6, %%ymm7, %%ymm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%ymm5, %%ymm7, %%ymm7 \n\t" //negate via xor with all 1s
                        "vptest %%ymm7, %%ymm7 \n\t" ///sets the zero flag iff (the now negated) ymm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "ymm6", "ymm7");

      if (found != 0)
        return false;
    }
#endif

    for (; i + 8 <= nData; i += 8) {

      asm __volatile__ ("vmovdqu %[dat1], %%xmm6  \n\t"
                        "vmovdqu %[dat2], %%xmm7  \n\t"
                        "vpcmpeqw %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%xmm5, %%xmm7, %%xmm7 \n\t" //negate via xor with all 1s
                        "vptest %%xmm7, %%xmm7 \n\t" ///sets the zero flag iff (the now negated) xmm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "xmm6", "xmm7");

      if (found != 0)
        return false;
    }

    for (; i + 4 <= nData; i += 4) {

      const UInt64* t1 = (UInt64*) (data1 + i);
      const UInt64* t2 = (UInt64*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i + 2 <= nData; i += 2) {

      const uint* t1 = (uint*) (data1 + i);
      const uint* t2 = (uint*) (data2 + i);

      if ((*t1) != (*t2))
        return false;
    }

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#endif

    return true;
  }

  inline bool equals_uint(const uint* data1, const uint* data2, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#else

    uchar found = 0;

    const float fkey = reinterpret<const uint, const float>(0xFFFFFFFF);
    asm __volatile__ ("vbroadcastss %[fkey], %%ymm5 \n\t"
                      : : [fkey] "m" (fkey) : "ymm5");

#if USE_SSE >= 6
    for (; i + 8 <= nData; i += 8) {

      asm __volatile__ ("vmovdqu %[dat1], %%ymm6  \n\t"
                        "vmovdqu %[dat2], %%ymm7  \n\t"
                        "vpcmpeqd %%ymm6, %%ymm7, %%ymm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%ymm5, %%ymm7, %%ymm7 \n\t" //negate via xor with all 1s
                        "vptest %%ymm7, %%ymm7 \n\t" ///sets the zero flag iff (the now negated) ymm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "ymm6", "ymm7");

      if (found != 0)
        return false;
    }
#endif

    for (; i + 4 <= nData; i += 4) {

      asm __volatile__ ("vmovdqu %[dat1], %%xmm6  \n\t"
                        "vmovdqu %[dat2], %%xmm7  \n\t"
                        "vpcmpeqd %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%xmm5, %%xmm7, %%xmm7 \n\t" //negate via xor with all 1s
                        "vptest %%xmm7, %%xmm7 \n\t" ///sets the zero flag iff (the now negated) xmm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "xmm6", "xmm7");

      if (found != 0)
        return false;
    }

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#endif

    return true;
  }

  inline bool equals_uint64(const UInt64* data1, const UInt64* data2, const size_t nData) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#else

    uchar found = 0;

    const float fkey = reinterpret<const uint, const float>(0xFFFFFFFF);
    asm __volatile__ ("vbroadcastss %[fkey], %%ymm5 \n\t"
                      : : [fkey] "m" (fkey) : "ymm5");

#if USE_SSE >= 6
    for (; i + 4 <= nData; i += 4) {

      asm __volatile__ ("vmovdqu %[dat1], %%ymm6  \n\t"
                        "vmovdqu %[dat2], %%ymm7  \n\t"
                        "vpcmpeqq %%ymm6, %%ymm7, %%ymm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%ymm5, %%ymm7, %%ymm7 \n\t" //negate via xor with all 1s
                        "vptest %%ymm7, %%ymm7 \n\t" ///sets the zero flag iff (the now negated) ymm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "ymm6", "ymm7");

      if (found != 0)
        return false;
    }
#endif

    for (; i + 2 <= nData; i += 2) {

      asm __volatile__ ("vmovdqu %[dat1], %%xmm6  \n\t"
                        "vmovdqu %[dat2], %%xmm7  \n\t"
                        "vpcmpeqq %%xmm6, %%xmm7, %%xmm7 \n\t" //xmm7 is overwritten with mask (all 1s on equal)
                        //need to test if xmm7 contains 0s now
                        "vxorps %%xmm5, %%xmm7, %%xmm7 \n\t" //negate via xor with all 1s
                        "vptest %%xmm7, %%xmm7 \n\t" ///sets the zero flag iff (the now negated) xmm7 is all 0
                        "setnz %[f] \n\t"
                        : [f] "+r" (found) : [dat1] "m" (data1[i]), [dat2] "m" (data2[i]) : "xmm6", "xmm7");

      if (found != 0)
        return false;
    }

    for (; i < nData; i++) {
      if (data1[i] != data2[i])
        return false;
    }
#endif

    return true;
  }

  /******** min, max, min+arg_min, max+arg_max *******/

  inline float max(const float_A16* data, const size_t nData) noexcept
  {
    float max_val=MIN_FLOAT;
    float cur_datum;
    size_t i = 0;

//#if !defined(USE_SSE) || USE_SSE < 2
#if 1 // g++ 4.8.5 uses avx instructions automatically
    for (; i < nData; i++) {
      cur_datum = data[i];
      max_val = std::max(max_val,cur_datum);
    }
#else
    //movaps is part of SSE2

    float tmp[4] = {MIN_FLOAT,MIN_FLOAT,MIN_FLOAT,MIN_FLOAT}; //reused as output, static not useful

    //maxps can take an unaligned mem arg!
    asm __volatile__ ("movaps %[tmp], %%xmm6"
                      : : [tmp] "m" (tmp[0]) : "xmm6");
    for (; (i+4) <= nData; i += 4) {
      asm __volatile__ ("movaps %[fptr], %%xmm7\n\t"
                        "maxps %%xmm7, %%xmm6"
                        : : [fptr] "m" (data[i]) : "xmm6", "xmm7");

    }
    asm __volatile__ ("movups %%xmm6, %[tmp]"
                      : [tmp] "=m" (tmp[0]) : : "memory");
    for (k=0; k < 4; k++)
      max_val = std::max(max_val,tmp[k]);

    for (; i < nData; i++) {
      cur_datum = data[i];
      if (cur_datum > max_val)
        max_val = cur_datum;
    }
#endif

    return max_val;
  }

  inline float min(const float_A16* data, const size_t nData) noexcept
  {
    float min_val=MAX_FLOAT;
    float cur_datum;
    size_t i = 0;

    //#if !defined(USE_SSE) || USE_SSE < 2
#if 1 // g++ 4.8.5 uses avx instructions automatically
    for (; i < nData; i++) {
      cur_datum = data[i];
      min_val = std::min(min_val,cur_datum);
    }
#else
    //movaps is part of SSE2

    float tmp[4] = {MAX_FLOAT,MAX_FLOAT,MAX_FLOAT,MAX_FLOAT}; //reused as output, static not useful

    //minps can take an unaligned mem arg!
    asm __volatile__ ("movaps %[tmp], %%xmm6"
                      : : [tmp] "m" (tmp[0]) : "xmm6");
    for (; (i+4) <= nData; i += 4) {
      asm __volatile__ ("movaps %[fptr], %%xmm7 \n\t"
                        "minps %%xmm7, %%xmm6 \n\t"
                        : : [fptr] "m" (data[i]) : "xmm6", "xmm7");
    }
    asm __volatile__ ("movups %%xmm6, %[tmp]"
                      : [tmp] "=m" (tmp[0]) :  : "memory");
    for (k=0; k < 4; k++)
      min_val = std::min(min_val,tmp[k]);

    for (; i < nData; i++) {
      cur_datum = data[i];
      if (cur_datum < min_val)
        min_val = cur_datum;
    }
#endif

    return min_val;
  }

  inline void find_max_and_argmax(const float_A16* data, const size_t nData, float& max_val, size_t& arg_max) noexcept
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
#else

    size_t i = 0;
    float cur_val;

#if USE_SSE >= 5

    //use AVX - align16 is no good, need align32 for aligned moves

    if (nData >= 12) {

      //const uint one = 1;
      const float inc = reinterpret<const uint, const float>(1); //*reinterpret_cast<const float*>(&one);

      //TODO: check out VPBROADCASTD for inc (but it's AVX2)

      asm __volatile__ ("vmovups %[d], %%ymm6 \n\t" //ymm6 is max register
                        "vxorps %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmax) to zero
                        "vbroadcastss %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                        "vmovups %%ymm4, %%ymm3 \n\t" //sets ymm3 (= current set index) to ones (next batch)
                        : : [d] "m" (data[i]), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");

      for (i += 8; (i+8) <= nData; i += 8) {
        asm __volatile__ ("vmovups %[fptr], %%ymm7 \n\t" //load needed, used multiple times
                          "vcmpnleps %%ymm6, %%ymm7, %%ymm0 \n\t"
                          "vblendvps %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                          "vblendvps %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                          "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                          : : [fptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }

#if 1
      //if at least four remain, go for xmm. But AVX will zero the upper 128 Bit => either use SSE or blend needs to be for ymm
      for (; (i+4) <= nData; i += 4) {
        asm __volatile__ ("vmovups %[fptr], %%xmm7 \n\t" //load needed, used multiple times
                          "vcmpnleps %%xmm6, %%xmm7, %%xmm0 \n\t"
                          "vblendvps %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //needs to be ymm!
                          "vblendvps %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //needs ti be ymm!
                          "vpaddd %%xmm4, %%xmm3, %%xmm3 \n\t" //could be ymm
                          : : [fptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }
#endif

      float tmp[8];
      uint itemp[8];

      asm __volatile__ ("vmovups %%ymm6, %[tmp] \n\t"
                        "vmovups %%ymm5, %[itemp]"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      for (uint k=0; k < 8; k++) {
        cur_val = tmp[k];
        if (cur_val > max_val) {
          max_val = cur_val;
          arg_max = (itemp[k] << 3) + k; //8*itemp[k] + k;
        }
      }
    }

#else
    //blendvps is part of SSE4

    if (nData >= 8) {
      assert(nData <= 17179869183);

      float tmp[4] = {MIN_FLOAT,MIN_FLOAT,MIN_FLOAT,MIN_FLOAT}; //reused as output, static not useful
      uint itemp[4] = {1,1,1,1}; //increment array, reused as output, static not useful
      static_assert(sizeof(uint) == 4, "wrong size");

      asm __volatile__ ("movups %[tmp], %%xmm6 \n\t" //xmm6 is max register
                        "xorps %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmax) to zero
                        "movups %[itemp], %%xmm4 \n\t" //xmm4 is increment register
                        "xorps %%xmm3, %%xmm3 \n\t" //sets xmm3 (= current set index) to zero
                        : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) : "xmm3", "xmm4", "xmm5", "xmm6");

      for (; (i+4) <= nData; i += 4) {
        asm __volatile__ ("movaps %[fptr], %%xmm7 \n\t" //load needed, used multiple times
                          "movaps %%xmm7, %%xmm0 \n\t"
                          "cmpnleps %%xmm6, %%xmm0 \n\t"
                          "blendvps %%xmm7, %%xmm6 \n\t"
                          "blendvps %%xmm3, %%xmm5 \n\t"
                          "paddd %%xmm4, %%xmm3 \n\t"
                          : : [fptr] "m" (data[i]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
      }

      asm __volatile__ ("movups %%xmm6, %[tmp] \n\t"
                        "movups %%xmm5, %[itemp]"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      for (uint k=0; k < 4; k++) {
        cur_val = tmp[k];
        if (cur_val > max_val) {
          max_val = cur_val;
          arg_max = 4*itemp[k] + k;
        }
      }
      assert(i == nData - (nData % 4));
    }
#endif

    for (; i < nData; i++) {
      cur_val = data[i];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = i;
      }
    }
#endif
  }

  inline void find_max_and_argmax(const double_A16* data, const size_t nData, double& max_val, size_t& arg_max) noexcept
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
#else

    size_t i = 0;
    double cur_val;

    static_assert(sizeof(size_t) == 8, "wrong size");

#if USE_SSE >= 5

    //use AVX  - align16 is no good, need align32 for aligned moves

    if (nData >= 6) {

      //TODO: make inc 4, safes final shuffle
      //const size_t one = 1;
      const double inc = reinterpret<const size_t, const double>(1); //*reinterpret_cast<const double*>(&one);

      //write first line directly -> save broadcast for ymm6, ymm3 starts like ymm4
      asm __volatile__ ("vmovupd %[dptr], %%ymm6 \n\t" //ymm6 is max register
                        "vxorpd %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmax) to zero
                        "vbroadcastsd %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                        "vmovupd %%ymm4, %%ymm3 \n\t" // current set indices start at 1 (next batch)
                        : : [dptr] "m" (data[i]), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");

      for (i += 4; (i+4) <= nData; i += 4) {
        asm __volatile__ ("vmovupd %[dptr], %%ymm7 \n\t" //load needed, used multiple times
                          "vcmpnlepd %%ymm6, %%ymm7, %%ymm0 \n\t"
                          "vblendvpd %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                          "vblendvpd %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                          "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                          : : [dptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }

#if 1
      //if two or three remain, go for xmm. But AVX will zero the upper 128 Bit => either use SSE or blend needs to be for ymm
      for (; (i+2) <= nData; i += 2) {
        asm __volatile__ ("vmovupd %[dptr], %%xmm7 \n\t" //load needed, used multiple times
                          "vcmpnlepd %%xmm6, %%xmm7, %%xmm0 \n\t" //high 128 bit are zeroed (no flags)
                          "vblendvpd %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //needs to be ymm!
                          "vblendvpd %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //needs to be ymm!
                          "vpaddd %%xmm4, %%xmm3, %%xmm3 \n\t" //could also be ymm
                          : : [dptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }
#endif

      double tmp[4];
      size_t itemp[4];

      asm __volatile__ ("vmovups %%ymm6, %[tmp] \n\t"
                        "vpsllq $2, %%ymm5, %%ymm5 \n\t" //for 64 bit we can do the shifting in AVX without reducing the maximal nData
                        "vmovups %%ymm5, %[itemp] \n\t"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      for (uint k=0; k < 4; k++) {
        cur_val = tmp[k];
        //std::cerr << "cur val: " << cur_val << std::endl;
        if (cur_val > max_val) {
          max_val = cur_val;
          arg_max = itemp[k] + k;
        }
      }
      //std::cerr << "minval: " << min_val << std::endl;
    }

#else

    if (nData >= 4) {
      assert(nData < 8589934592);

      //note: broadcast is better implemented by movddup (SSE3). But this code is no longer improved
      double tmp[2] = {MIN_DOUBLE,MIN_DOUBLE}; //reused as output, static not useful
      uint itemp[4] = {0,1,0,1}; //reused as output, static not useful
      static_assert(sizeof(uint) == 4, "wrong size");

      asm __volatile__ ("movupd %[tmp], %%xmm6 \n\t"
                        "xorpd %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmax) to zero
                        "movupd %[itemp], %%xmm4 \n\t"
                        "xorpd %%xmm3, %%xmm3 \n\t" //sets xmm3 (= current set index)
                        : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) :  "xmm3", "xmm4", "xmm5", "xmm6");

      for (; (i+2) <= nData; i += 2) {
        asm __volatile__ ("movapd %[dptr], %%xmm7 \n\t" //load needed, used multiple times
                          "movapd %%xmm7, %%xmm0 \n\t"
                          "cmpnlepd %%xmm6, %%xmm0 \n\t"
                          "blendvpd %%xmm7, %%xmm6 \n\t"
                          "blendvpd %%xmm3, %%xmm5 \n\t"
                          "paddd %%xmm4, %%xmm3 \n\t"
                          : : [dptr] "m" (data[i]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
      }

      asm __volatile__ ("movupd %%xmm6, %[tmp] \n\t"
                        "movupd %%xmm5, %[itemp] \n\t"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      assert(itemp[0] == 0);
      assert(itemp[2] == 0);

      for (uint k=0; k < 2; k++) {
        cur_val = tmp[k];
        //std::cerr << "cur val: " << cur_val << std::endl;
        if (cur_val > max_val) {
          max_val = cur_val;
          arg_max = 2*itemp[2*k+1] + k;
        }
      }

      //std::cerr << "minval: " << min_val << std::endl;
      assert(i == nData - (nData % 2));
    }

#endif

    for (; i < nData; i++) {
      cur_val = data[i];
      if (cur_val > max_val) {
        max_val = cur_val;
        arg_max = i;
      }
    }
#endif
  }

  inline void find_min_and_argmin(const float_A16* data, const size_t nData, float& min_val, size_t& arg_min) noexcept
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
#else

    size_t i = 0;
    float cur_val;

#if USE_SSE >= 5

    //use AVX  - align16 is no good, need align32 for aligned moves

    if (nData >= 12) {

      //const uint one = 1;
      const float inc = reinterpret<const uint, const float>(1); //*reinterpret_cast<const float*>(&one);

      asm __volatile__ ("vmovups %[d], %%ymm6 \n\t" //ymm6 is max register
                        "vxorps %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmax) to zero
                        "vbroadcastss %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                        "vmovups %%ymm4, %%ymm3 \n\t" //sets ymm3 (= current set index) to ones (next batch)
                        : : [d] "m" (data[i]), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");


      for (i += 8; (i+8) <= nData; i += 8) {
        asm __volatile__ ("vmovups %[fptr], %%ymm7 \n\t" //load needed, used multiple times
                          "vcmpltps %%ymm6, %%ymm7, %%ymm0 \n\t" //destination is last
                          "vblendvps %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                          "vblendvps %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                          "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                          : : [fptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }

#if 1
      //if at least four remain, go for xmm. But AVX will zero the upper 128 Bit => either use SSE or blend needs to be for ymm
      for (; (i+4) <= nData; i += 4) {
        asm __volatile__ ("vmovups %[fptr], %%xmm7 \n\t" //load needed, used multiple times
                          "vcmpltps %%xmm6, %%xmm7, %%xmm0 \n\t"
                          "vblendvps %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //needs to be ymm!
                          "vblendvps %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //needs ti be ymm!
                          "vpaddd %%xmm4, %%xmm3, %%xmm3 \n\t" //could be ymm
                          : : [fptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }
#endif

      float tmp[8];
      uint itemp[8];

      asm __volatile__ ("vmovups %%ymm6, %[tmp] \n\t"
                        "vmovups %%ymm5, %[itemp]"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      for (uint k=0; k < 8; k++) {
        cur_val = tmp[k];
        //std::cerr << "cur val: " << cur_val << std::endl;
        if (cur_val < min_val) {
          min_val = cur_val;
          arg_min = (itemp[k] << 3) + k; //8*itemp[k] + k;
        }
      }
    }

#else
    //blendvps is part of SSE4

    if (nData >= 8) {

      assert(nData <= 17179869183);

      float tmp[4] = {MAX_FLOAT,MAX_FLOAT,MAX_FLOAT,MAX_FLOAT}; //reused as output, static not useful
      uint itemp[4] = {1,1,1,1}; //reused as output, static not useful
      static_assert(sizeof(uint) == 4, "wrong size");

      asm __volatile__ ("movups %[tmp], %%xmm6 \n\t"
                        "xorps %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmin) to zero
                        "movups %[itemp], %%xmm4 \n\t"
                        "xorps %%xmm3, %%xmm3 \n\t" //contains candidate argmin
                        : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) :  "xmm3", "xmm4", "xmm5", "xmm6");

      for (; (i+4) <= nData; i += 4) {
        asm __volatile__ ("movaps %[fptr], %%xmm7 \n\t" //load needed, used multiple times
                          "movaps %%xmm7, %%xmm0 \n\t"
                          "cmpltps %%xmm6, %%xmm0 \n\t"
                          "blendvps %%xmm7, %%xmm6 \n\t"
                          "blendvps %%xmm3, %%xmm5 \n\t"
                          "paddd %%xmm4, %%xmm3 \n\t"
                          : : [fptr] "m" (data[i]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
      }

      asm __volatile__ ("movups %%xmm6, %[tmp] \n\t"
                        "movups %%xmm5, %[itemp] \n\t"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      //std::cerr << "intermediate minval: " << min_val << std::endl;

      for (uint k=0; k < 4; k++) {
        cur_val = tmp[k];
        //std::cerr << "cur val: " << cur_val << std::endl;
        if (cur_val < min_val) {
          min_val = cur_val;
          arg_min = 4*itemp[k] + k;
        }
      }
      assert(i == nData - (nData % 4));
    }

#endif

    //std::cerr << "minval: " << min_val << std::endl;

    for (; i < nData; i++) {
      cur_val = data[i];
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = i;
      }
    }
#endif
  }

  inline void find_min_and_argmin(const double_A16* data, const size_t nData, double& min_val, size_t& arg_min) noexcept
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
#else

    size_t i = 0;
    double cur_val;

    static_assert(sizeof(size_t) == 8, "wrong size");

#if USE_SSE >= 5

    //use AVX  - align16 is no good, need align32 for aligned moves

    if (nData >= 8) {

      //TODO: make inc 4, safes final shuffle
      //const size_t one = 1;
      const double inc = reinterpret<const size_t, const double>(1); //*reinterpret_cast<const double*>(&one);

      //write first line directly -> save broadcast for ymm6, ymm3 starts like ymm4
      asm __volatile__ ("vmovupd %[dptr], %%ymm6 \n\t" //ymm6 is max register
                        "vxorpd %%ymm5, %%ymm5, %%ymm5 \n\t" //sets ymm5 (= argmax) to zero
                        "vbroadcastsd %[itemp], %%ymm4 \n\t" //ymm4 is increment register
                        "vmovupd %%ymm4, %%ymm3 \n\t" // current set indices start at 1 (next batch)
                        : : [dptr] "m" (data[i]), [itemp] "m" (inc) : "ymm3", "ymm4", "ymm5", "ymm6");

      for (i += 4; (i+4) <= nData; i += 4) {
        asm __volatile__ ("vmovupd %[dptr], %%ymm7 \n\t" //load needed, used multiple times
                          "vcmpltpd %%ymm6, %%ymm7, %%ymm0 \n\t" //destination is last
                          "vblendvpd %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //destination is last
                          "vblendvpd %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //destination is last
                          "vpaddd %%ymm4, %%ymm3, %%ymm3 \n\t" //destination is last
                          : : [dptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }

#if 1
      //if two or three remain, go for xmm. But AVX will zero the upper 128 Bit => either use SSE or blend needs to be for ymm
      for (; (i+2) <= nData; i += 2) {
        asm __volatile__ ("vmovupd %[dptr], %%xmm7 \n\t" //load needed, used multiple times
                          "vcmpltpd %%xmm6, %%xmm7, %%xmm0 \n\t" //high 128 bit are zeroed (no flags)
                          "vblendvpd %%ymm0, %%ymm7, %%ymm6, %%ymm6 \n\t" //needs to be ymm!
                          "vblendvpd %%ymm0, %%ymm3, %%ymm5, %%ymm5 \n\t" //needs to be ymm!
                          "vpaddd %%xmm4, %%xmm3, %%xmm3 \n\t" //could also be ymm
                          : : [dptr] "m" (data[i]) : "ymm0", "ymm3", "ymm5", "ymm6", "ymm7");
      }
#endif

      double tmp[4];
      size_t itemp[4];

      asm __volatile__ ("vmovupd %%ymm6, %[tmp] \n\t"
                        "vpsllq $2, %%ymm5, %%ymm5 \n\t" //for 64 bit we can do the shifting in AVX without reducing the maximal nData
                        "vmovupd %%ymm5, %[itemp]"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      for (uint k=0; k < 4; k++) {
        cur_val = tmp[k];
        //std::cerr << "cur val: " << cur_val << std::endl;
        if (cur_val < min_val) {
          min_val = cur_val;
          arg_min = itemp[k] + k;
        }
      }
    }

#else

    if (nData >= 4) {

      assert(nData < 8589934592);

      double tmp[2] = {MAX_DOUBLE,MAX_DOUBLE}; //reused as output, static not useful
      uint itemp[4] = {0,1,0,1}; //reused as output, static not useful
      static_assert(sizeof(uint) == 4, "wrong size");

      //note: broadcast is better implemented by movddup (SSE3). But this code is no longer improved
      asm __volatile__ ("movupd %[tmp], %%xmm6 \n\t"
                        "xorpd %%xmm5, %%xmm5 \n\t" //sets xmm5 (= argmin) to zero
                        "movupd %[itemp], %%xmm4 \n\t"
                        "xorpd %%xmm3, %%xmm3 \n\t" //contains candidate argmin
                        : : [tmp] "m" (tmp[0]), [itemp] "m" (itemp[0]) :  "xmm3", "xmm4", "xmm5", "xmm6");

      for (; (i+2) <= nData; i += 2) {
        asm __volatile__ ("movapd %[dptr], %%xmm7 \n\t" //load needed, used multiple times
                          "movapd %%xmm7, %%xmm0 \n\t"
                          "cmpltpd %%xmm6, %%xmm0 \n\t"
                          "blendvpd %%xmm7, %%xmm6 \n\t"
                          "blendvpd %%xmm3, %%xmm5 \n\t"
                          "paddd %%xmm4, %%xmm3 \n\t"
                          : : [dptr] "m" (data[i]) : "xmm0", "xmm3", "xmm5", "xmm6", "xmm7");
      }

      asm __volatile__ ("movupd %%xmm6, %[tmp] \n\t"
                        "movupd %%xmm5, %[itemp] \n\t"
                        : [tmp] "=m" (tmp[0]), [itemp] "=m" (itemp[0]) : : "memory");

      assert(itemp[0] == 0);
      assert(itemp[2] == 0);

      for (uint k=0; k < 2; k++) {
        cur_val = tmp[k];
        //std::cerr << "cur val: " << cur_val << std::endl;
        if (cur_val < min_val) {
          min_val = cur_val;
          arg_min = 2*itemp[2*k+1] + k;
        }
      }
      assert(i == nData - (nData%2));
    }
#endif

    //std::cerr << "minval: " << min_val << std::endl;

    for (; i < nData; i++) {
      cur_val = data[i];
      if (cur_val < min_val) {
        min_val = cur_val;
        arg_min = i;
      }
    }
#endif
  }

  /******************** array mul *********************/

  inline void mul_array(float_A16* data, const size_t nData, const float constant) noexcept
  {
    assertAligned16(data);

    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 2
    for (; i < nData; i++) { //g++ uses packed avx mul, but after checking alignment
      data[i] *= constant;
    }
#elif USE_SSE >= 5

    // AVX  - align16 is no good, need align32 for aligned moves

    asm __volatile__ ("vbroadcastss %[tmp], %%ymm7 \n\t"
                      : : [tmp] "m" (constant) : "ymm7");

    //vmulps can take an unaligned mem arg!
    for (; i+8 <= nData; i+=8) {
      asm volatile (//"vmovups %[fptr], %%ymm6 \n\t"
        //"vmulps %%ymm6, %%ymm7, %%ymm6 \n\t"
        "vmulps %[fptr], %%ymm7, %%ymm6 \n\t"
        "vmovups %%ymm6, %[fptr] \n\t"
        : [fptr] "+m" (data[i]) : : "ymm6", "memory");
    }

    for (; i+4 <= nData; i+=4) {
      asm volatile (//"vmovups %[fptr], %%xmm6 \n\t"
        //"vmulps %%xmm6, %%xmm7, %%xmm6 \n\t"
        "vmulps %[fptr], %%xmm7, %%xmm6 \n\t"
        "vmovups %%xmm6, %[fptr] \n\t"
        : [fptr] "+m" (data[i]) : : "ymm6", "memory");
    }

    for (; i < nData; i++)
      data[i] *= constant;
#else
    float temp[4];
    for (; i < 4; i++)
      temp[i] = constant;
    asm volatile ("movaps %[temp], %%xmm7" : : [temp] "m" (temp[0]) : "xmm7" );

    i = 0;
    for (; i+4 <= nData; i+=4) {
      asm volatile ("movaps %[fptr], %%xmm6 \n\t"
                    "mulps %%xmm7, %%xmm6 \n\t"
                    "movaps %%xmm6, %[fptr] \n\t"
                    : [fptr] "+m" (data[i]) : : "xmm6", "memory");
    }

    for (; i < nData; i++)
      data[i] *= constant;
#endif
  }

  inline void mul_array(double_A16* data, const size_t nData, const double constant) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 2
    for (; i < nData; i++) {
      data[i] *= constant;
    }
#else
#if USE_SSE >= 5

    // AVX  - align16 is no good, need align32 for aligned moves

    asm __volatile__ ("vbroadcastsd %[tmp], %%ymm7 \n\t"
                      : : [tmp] "m" (constant) : "ymm7");

    //vmulpd can take an unaligned mem arg!
    for (; i+4 <= nData; i+=4) {
      asm volatile (//"vmovupd %[dptr], %%ymm6 \n\t"
        //"vmulpd %%ymm6, %%ymm7, %%ymm6 \n\t"
        "vmulpd %[dptr], %%ymm7, %%ymm6 \n\t"
        "vmovupd %%ymm6, %[dptr] \n\t"
        : [dptr] "+m" (data[i]) : : "ymm6", "memory");
    }

    for (; i+2 <= nData; i+=2) {
      asm volatile (//"vmovupd %[dptr], %%xmm6 \n\t"
        //"vmulpd %%xmm6, %%xmm7, %%xmm6 \n\t"
        "vmulpd %[dptr], %%xmm7, %%xmm6 \n\t"
        "vmovupd %%xmm6, %[dptr] \n\t"
        : [dptr] "+m" (data[i]) : : "ymm6", "memory");
    }

#else
    double temp[2] = {constant, constant};

    //note: broadcast is better implemented by movddup (SSE3). But this code is no longer improved
    asm volatile ("movupd %[temp], %%xmm7" : : [temp] "m" (temp[0]) : "xmm7" );

    for (; i+2 <= nData; i+=2) {
      asm volatile ("movapd %[dptr], %%xmm6 \n\t"
                    "mulpd %%xmm7, %%xmm6 \n\t"
                    "movupd %%xmm6, %[dptr] \n\t"
                    : [dptr] "+m" (data[i]) : : "xmm6", "memory");
    }
#endif

    for (; i < nData; i++)
      data[i] *= constant;
#endif
  }

  /******** array additions with multiplications *************/

  //performs data[i] -= factor*data2[i] for each i
  //this is a frequent operation in the conjugate gradient algorithm
  inline void array_subtract_multiple(double_A16* attr_restrict data, const size_t nData, double factor,
                                      const double_A16* attr_restrict data2) noexcept
  {
    assertAligned16(data);

    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 2
    for (; i < nData; i++)
      data[i] -= factor * data2[i];
#else

#if USE_SSE >= 5
    // AVX  - align16 is no good, need align32 for aligned moves

    asm __volatile__ ("vbroadcastsd %[tmp], %%ymm7 \n\t"
                      : : [tmp] "m" (factor) : "ymm7");

    for (; i+4 <= nData; i+=4) {
      //vmulpd can take an unaligned mem arg, vfnadd too!
      asm volatile ("vmovupd %[dptr], %%ymm5 \n\t"
#if USE_SSE >= 6
                    "vmulpd %[cdptr], %%ymm7, %%ymm6 \n\t" //destination goes last
                    "vsubpd %%ymm6, %%ymm5, %%ymm5 \n\t" //destination goes last (cannot use memarg here, first isn't dptr!)
#else
                    "vfnmadd231pd %[cdptr], %%ymm7, %%ymm5 \n\t"
#endif
                    "vmovupd %%ymm5, %[dptr] \n\t"
                    : [dptr] "+m" (data[i]) : [cdptr] "m" (data2[i]) : "ymm5", "ymm6", "memory");
    }

    for (; i+2 <= nData; i+=2) {
      //vmulpd can take an unaligned mem arg, vfnadd too!
      asm volatile ("vmovupd %[dptr], %%xmm5 \n\t"
#if USE_SSE >= 6
                    "vmulpd %[cdptr], %%xmm7, %%xmm6 \n\t" //destination goes last
                    "vsubpd %%xmm6, %%xmm5, %%xmm5 \n\t" //destination goes last (cannot use memarg here, first isn't dptr!)
#else
                    "vfnmadd231pd %[cdptr], %%xmm7, %%xmm5 \n\t"
#endif
                    "vmovupd %%xmm5, %[dptr] \n\t"
                    : [dptr] "+m" (data[i]) : [cdptr] "m" (data2[i]) : "ymm5", "ymm6", "memory");
    }

#else
    double temp[2] = {factor, factor};

    //note: broadcast is better implemented by movddup (SSE3). But this code is no longer improved
    asm volatile ("movupd %[temp], %%xmm7" : : [temp] "m" (temp[0]) : "xmm7" );
    for (; i+2 <= nData; i+=2) {
      asm volatile ("movapd %[cdptr], %%xmm6 \n\t"
                    "mulpd %%xmm7, %%xmm6 \n\t"
                    "movapd %[dptr], %%xmm5 \n\t"
                    "subpd %%xmm6, %%xmm5 \n\t"
                    "movapd %%xmm5, %[dptr] \n\t"
                    : [dptr] "+m" (data[i]) : [cdptr] "m" (data2[i]) : "xmm5", "xmm6", "memory");
    }
#endif

    for (; i < nData; i++)
      data[i] -= factor * data2[i];
#endif
  }

  inline void array_add_multiple(double_A16* attr_restrict data, const size_t nData, double factor,
                                 const double_A16* attr_restrict data2) noexcept
  {
    array_subtract_multiple(data, nData, -factor, data2);
  }

  //NOTE: despite attr_restrict, you can safely pass the same for dest and src1 or src2
  inline void go_in_neg_direction(double_A16* attr_restrict dest, const size_t nData, const double_A16* attr_restrict src1,
                                  const double_A16* attr_restrict src2, double alpha) noexcept
  {
    assertAligned16(dest);
    //std::cerr << "go_in_neg_direction " << USE_SSE << std::endl;

    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5
    for (; i < nData; i++)
      dest[i] = src1[i] - alpha * src2[i];
#else

    // AVX  - align16 is no good, need align32 for aligned moves
    //NOTE: for large arrays, using vmovntpd instead of vmovupd might be better, but it would need 32-byte alignment!

    asm __volatile__ ("vbroadcastsd %[w1], %%ymm0 \n\t" //ymm0 = w1
                      : : [w1] "m" (alpha) : "ymm0");


    //vmulpd can take an unaligned mem arg, vfnmadd too!
    //vsubpd can take an unaligned mem arg
    for (; i+4 <= nData; i+= 4) {
      asm volatile ("vmovupd %[s2_ptr], %%ymm3 \n\t"
#if USE_SSE < 6
                    //"vmulpd %%ymm0, %%ymm3, %%ymm3 \n\t" //destination goes last
                    "vmulpd %[s2_ptr], %%ymm0, %%ymm3 \n\t" //destination goes last
                    "vmovupd %[s1_ptr], %%ymm2 \n\t"
                    "vsubpd %%ymm3, %%ymm2, %%ymm2 \n\t" //destination goes last
                    //"vsubpd %[s1_ptr], %%ymm3, %%ymm2 \n\t" //destination goes last
#else
                    "vmovupd %[s1_ptr], %%ymm2 \n\t"
                    "vfnmadd231pd %%ymm0, %%ymm3, %%ymm2 \n\t" //destination goes last
                    //"vfnmadd132pd %[s2_ptr], %%ymm0, %%ymm2 \n\t" //destination goes last
                    //"vfmsub231pd %[s2_ptr], %%ymm0, %%ymm2 \n\t" //destination goes last
#endif
                    "vmovupd %%ymm2, %[dest] \n\t"
                    : [dest] "+m" (dest[i]) : [s1_ptr] "m" (src1[i]), [s2_ptr] "m" (src2[i]) :  "ymm0", "ymm2", "ymm3", "memory");
    }

    for (; i+2 < nData; i+= 2) {
      asm volatile (
#if USE_SSE < 6
        "vmulpd %[s2_ptr], %%xmm0, %%xmm3 \n\t" //destination goes last
        "vsubpd %[s1_ptr], %%xmm3, %%xmm2 \n\t" //destination goes last
#else
        "vmovupd %[s1_ptr], %%xmm2 \n\t"
        "vfnmadd231pd %[s2_ptr], %%xmm0, %%xmm2 \n\t" //destination goes last
#endif
        "vmovupd %%xmm2, %[dest] \n\t"
        : [dest] "+m" (dest[i]) : [s1_ptr] "m" (src1[i]), [s2_ptr] "m" (src2[i]) : "ymm0", "ymm2", "ymm3", "memory");
    }

    for (; i < nData; i++)
      dest[i] = src1[i] - alpha * src2[i];
#endif
  }

  //NOTE: despite attr_restrict, you can safely pass the same for dest and src1 or src2
  inline void assign_weighted_combination(double_A16* attr_restrict dest, const size_t nData, double w1, const double_A16* attr_restrict src1,
                                          double w2, const double_A16* attr_restrict src2) noexcept
  {
    size_t i = 0;
#if !defined(USE_SSE) || USE_SSE < 5
    for (; i < nData; i++)
      dest[i] = w1 * src1[i] + w2 * src2[i];
#else
    //use AVX

    asm __volatile__ ("vbroadcastsd %[w1], %%ymm0 \n\t" //ymm0 = w1
                      "vbroadcastsd %[w2], %%ymm1 \n\t" //ymm1 = w2
                      : : [w1] "m" (w1), [w2] "m" (w2) : "ymm0", "ymm1");

    //vmulpd can take an unaligned mem arg, vfnmadd too!
    //vaddpd can take an unaligned mem arg!
    for (; i+4 <= nData; i+= 4) {
      asm volatile ("vmulpd %[s1_ptr], %%ymm0, %%ymm2 \n\t" //destination goes last
                    //"vmovupd %[s2_ptr], %%ymm3 \n\t"
#if USE_SSE < 6
                    //"vmulpd %%ymm1, %%ymm3, %%ymm3 \n\t" //destination goes last
                    "vmulpd %[s2_ptr], %%ymm1, %%ymm3 \n\t" //destination goes last
                    "vaddpd %%ymm3, %%ymm2, %%ymm2 \n\t" //destination goes last
#else
                    //"vfmadd231pd %%ymm3, %%ymm1, %%ymm2 \n\t" //destination goes last
                    "vfmadd231pd %[s2_ptr], %%ymm1, %%ymm2 \n\t" //destination goes last
#endif
                    "vmovupd %%ymm2, %[dest] \n\t"
                    : [dest] "+m" (dest[i]) : [s1_ptr] "m" (src1[i]), [s2_ptr] "m" (src2[i]) : "ymm2", "ymm3", "memory");
    }

    for (; i+2 < nData; i+= 2) {
      asm volatile ("vmulpd %[s1_ptr], %%xmm0, %%xmm2 \n\t"
#if USE_SSE < 6
                    "vmulpd %[s2_ptr], %%xmm1, %%xmm3 \n\t" //destination goes last
                    "vaddpd %%xmm3, %%xmm2, %%xmm2 \n\t" //destination goes last
#else
                    "vfmadd231pd %[s2_ptr], %%xmm1, %%xmm2 \n\t" //destination goes last
#endif
                    "vmovupd %%xmm2, %[dest] \n\t"
                    : [dest] "+m" (dest[i]) : [s1_ptr] "m" (src1[i]), [s2_ptr] "m" (src2[i]) : "ymm2", "ymm3", "memory");
    }

    for (; i < nData; i++)
      dest[i] = w1 * src1[i] + w2 * src2[i];
#endif
  }

  /***************** dot product  *****************/

  template<typename T>
  inline T dotprod(const T* data1, const T* data2, const size_t size) noexcept
  {
    return std::inner_product(data1, data1+size, data2, (T) 0);
  }

  template<>
  inline double dotprod(const double* data1, const double* data2, const size_t size) noexcept
  {
#if !defined(USE_SSE) || USE_SSE < 5
    return std::inner_product((double_A16*) data1, (double_A16*) data1+size, data2, 0.0);
#else
    //checked: g++ does not use dppd. It uses 256 bit instead, but the running times are the same

    //NOTE: unlike vpps, vppd is not available for 256 bit, not even in AVX-512

    asm __volatile__ ("vxorpd %%xmm4, %%xmm4, %%xmm4 \n\t" : : : "xmm4");

    double result = 0.0;

    size_t i = 0;
    for (; i + 2 <= size; i += 2) {
      //std::cerr << "i: " << i << std::endl;

      //vdppd can take a mem arg!
      asm __volatile__ ("vmovupd %[d2], %%xmm7 \n\t"
                        "vdppd $49, %[d1], %%xmm7, %%xmm5 \n\t" //include all, write in first (hence second is set to 0)
                        "vaddsd %%xmm5, %%xmm4, %%xmm4 \n\t"
                        : : [d1] "m" (data1[i]), [d2] "m" (data2[i]) : "xmm7", "xmm4", "xmm5");

      //std::cerr << "state after add: " << temp[0] << "," << temp[1] << std::endl;
    }

    asm __volatile__ ("vmovlpd %%xmm4, %0 \n\t" : "+m" (result) : : );

    for (; i < size; i++)
      result += data1[i] * data2[i];

    return result;
#endif
  }

  /******************** binary search *********************/

  //binary search, returns MAX_UINT if key is not found, otherwise the position in the vector
  template<typename T, typename Less, typename Equal, typename ST>
  inline ST binsearch(const T* data, const T key, const ST nData) noexcept
  {
    const static Less less;
    const static Equal equal;

#ifndef NDEBUG
    for (ST i = 1; i < nData; i++)
      assert(!less(data[i],data[i-1]));
#endif

    if (nData == 0 || less(key,data[0]) || less(data[nData-1],key))
      return MAX_UINT;

    ST lower = 0;
    ST upper = nData-1;
    if (equal(data[lower],key))
      return lower;
    if (equal(data[upper],key))
      return upper;

    while (lower+1 < upper) {
      assert(less(data[lower],key));
      assert(less(key,data[upper]));

      const size_t middle = (lower+upper) >> 1;  // (lower+upper)/2;
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


  template<typename T, typename Less, typename Equal, typename ST>
  inline ST binsearch_insertpos(const T* data, const T key, const ST nData) noexcept
  {
    const static Less less;
    const static Equal equal;

#ifndef NDEBUG
    for (ST i = 1; i < nData; i++)
      assert(!less(data[i],data[i-1]));
#endif

    if (nData == 0 || less(key,data[0]) || equal(key,data[0]))
      return 0;

    if (less(data[nData-1], key))
      return nData;

    ST lower = 0;
    ST upper = nData-1;
    if (equal(data[upper],key))
      return upper;

    while (lower+1 < upper) {
      assert(less(data[lower],key));
      assert(less(key,data[upper]));

      const size_t middle = (lower+upper) >> 1;  // (lower+upper)/2;
      assert(middle > lower && middle < upper);
      const T md = data[middle];
      if (equal(md,key))
        return middle;
      else if (less(md,key))
        lower = middle;
      else
        upper = middle;
    }

    assert(lower+1 == upper);
    return upper;
  }

  template<typename T, typename ST, typename Less, typename Equal>
  inline ST index_binsearch_insertpos(const T* data, const T key, const ST* index, const ST nData) noexcept
  {
    const static Less less;
    const static Equal equal;

    if (nData == 0 || less(key,data[index[0]]) || equal(key,data[index[0]]))
      return 0;

    if (less(data[index[nData-1]], key))
      return nData;

    ST lower = 0;
    ST upper = nData-1;
    if (equal(data[index[upper]],key))
      return upper;

    while (lower+1 < upper) {
      assert(less(data[index[lower]],key));
      assert(less(key,data[index[upper]]));

      const size_t middle = (lower+upper) >> 1;  // (lower+upper)/2;
      assert(middle > lower && middle < upper);
      const T md = data[index[middle]];
      if (equal(md,key))
        return middle;
      else if (less(md,key))
        lower = middle;
      else
        upper = middle;
    }

    assert(lower+1 == upper);
    return upper;
  }

}

#endif