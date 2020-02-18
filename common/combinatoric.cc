/*** first version written by Thomas Schoenemann as a private person without employment, November 2009 ***/
/*** refined at the University of Düsseldorf, Germany, 2012 ***/

#include "combinatoric.hh"

uint fac(uint n)
{
  uint r = 1;
  for (uint k=2; k <= n; k++)
    r *= k;

  return r;
}

long double ldfac(uint n)
{
  long double r = 1.0;

  for (uint k=2; k <= n; k++)
    r *= k;

  return r;
}

uint choose(uint n, uint k)
{
  assert(k <= n);
  assert(n >= 1);

  if (k == n || k == 0)
    return 1;

  uint r = n;
  for (uint i=n-1; i > (n-k); i--)
    r *= i;

  for (uint i=2; i <= k; i++)
    r /= i;

  return r;
}

long double ldchoose(uint n, uint k)
{
  assert(k <= n);
  assert(n >= 1);

  if (k == n || k == 0)
    return 1;

  long double r = n;
  for (uint i=n-1; i > (n-k); i--)
    r *= i;

  for (uint i=2; i <= k; i++)
    r /= i;

  return r;
}

long double ldchoose(uint n, uint k, const Math1D::Vector<long double>& ld_fac)
{
  assert(k <= n);
  assert(n >= 1);
  assert(k < ld_fac.size());

  if (k == n || k == 0)
    return 1;

  long double r = n;
  for (uint i=n-1; i > (n-k); i--)
    r *= i;

  r /= ld_fac[k];

  return r;
}

// greatest common divisor via the Euclidean algorithm
uint gcd(uint n1, uint n2)
{
  if (n1 < n2)
    std::swap(n1,n2);

  while (n2 != 0) {
    const uint t = n2;
    n2 = n1 % n2;
    n1 = t;
  }

  return n1;
}

//returns if n is a prime number (0,1,2,3 all count as prime)
bool is_prime(const uint n)
{
  if (n <= 3)
    return true;
  if ((n & 1) == 0)
    return false; //even an >= 4
  
  const uint limit = sqrt(n + 0.1); //a non-prime must have a divisor <= its square root
  for (uint i = 5; i <= limit; i += 2) { //even numbers are no use
    if ( (n % i) == 0)
      return false;
  }
  
  return true;
}

//returns 1 for primes
uint lowest_divisor(const uint n) {
  
  if (n <= 3)
    return 1;
  if ((n & 1) == 0) {    
    //room for improvements here
    return 2 * lowest_divisor(n >> 1); //even an >= 4
  }
  
  const uint limit = sqrt(n + 0.1); //a non-prime must have a divisor <= its square root
  for (uint i = 5; i <= limit; i += 2) { //even numbers are no use
    if ( (n % i) == 0)
      return i;
  }
  
  return 1;  
}

