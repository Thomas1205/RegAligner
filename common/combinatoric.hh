/*** first version written by Thomas Schoenemann as a private person without employment, November 2009 ***/
/*** refined at the University of Düsseldorf, Germany, 2012 ***/

#ifndef COMBINATORIC_HH
#define COMBINATORIC_HH

#include "makros.hh"
#include "vector.hh"

uint fac(uint n);

long double ldfac(uint n);

uint choose(uint n, uint k);

long double ldchoose(uint n, uint k);

long double ldchoose(uint n, uint k, const Math1D::Vector<long double>& ld_fac);

//NOTE: C++-17 provides std::gcd in <numeric>

// greatest common divisor via the Euclidean algorithm
uint gcd(uint n1, uint n2);

// greatest common divisor via the Euclidean algorithm
inline UInt64 gcd64(UInt64 n1, UInt64 n2);

//returns if n is a prime number (0,1,2,3 all count as prime)
bool is_prime(const uint n);

//returns 1 for primes
uint lowest_divisor(const uint n);

/**** implementation ****/

inline UInt64 gcd64(UInt64 n1, UInt64 n2)
{
  if (n1 < n2)
    std::swap(n1,n2);

  while (n2 != 0) {
    const UInt64 t = n2;
    n2 = n1 % n2;
    n1 = t;
  }

  return n1;
}

inline UInt64 gcd_mixed_128_64(UInt64 n1_high, UInt64 n1_low, UInt64 n2)
{
  assert(n1_high > 0);

#ifdef USE_ASM
  if (n2 <= 1)
    return n2;

  UInt64 t = n2;
  asm volatile ("movq %[n1_high], %%rdx \n\t"
                "movq %[n1_low], %%rax \n\t"
                "divq %[n2] \n\t"
                "movq %%rdx, %[n2]"
                : [n2] "+g" (n2) : [n1_high] "g" (n1_high), [n1_low] "g" (n1_low) : "rdx", "rax");
  n1_low = t;

  while(n2 != 0) {
    t = n2;
    n2 = n1_low % n2;
    n1_low = t;
  }

  return n1_low;
#endif
}

#endif
