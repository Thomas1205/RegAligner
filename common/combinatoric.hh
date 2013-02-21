/*** first version written by Thomas Schoenemann as a private person without employment, November 2009 ***/
/*** refined at the University of Düsseldorf, Germany, 2012 ***/

#ifndef COMBINATORIC_HH
#define COMBINATORIC_HH

#include "makros.hh"

uint fac(uint n);

long double ldfac(uint n);

uint choose(uint n, uint k);

long double ldchoose(uint n, uint k);

// greatest common divisor via the Euclidean algorithm
inline long gcd64(unsigned long n1, unsigned long n2);

// greatest common divisor via the Euclidean algorithm
uint gcd(uint n1, uint n2);


/**** implementation ****/

inline long gcd64(unsigned long n1, unsigned long n2) {

  if (n1 < n2)
    std::swap(n1,n2);

  while (n2 != 0) {
    unsigned long t = n2;
    n2 = n1 % n2;
    n1 = t;
  }

  return n1;
}


#endif
