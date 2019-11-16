/*** written by Thomas Schoenemann as a private person without employment, September 2009 ***/

#include "vector.hh"

namespace Math1D {

  template<>
  float Vector<float>::max() const
  {
    return Makros::max(Storage1D<float>::data_, Storage1D<float>::size_);
  }

  template<>
  float Vector<float>::min() const
  {
    return Makros::min(Storage1D<float>::data_, Storage1D<float>::size_);
  }

  template<>
  void Vector<float>::operator*=(const float scalar)
  {
    Makros::mul_array(Storage1D<float>::data_, Storage1D<float>::size_, scalar);
  }

  template<>
  void Vector<double>::operator*=(const double scalar)
  {
    Makros::mul_array(Storage1D<double>::data_, Storage1D<double>::size_, scalar);
  }


}
