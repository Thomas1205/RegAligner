/*** written by Thomas Schoenemann as a private person without employment, September 2009 ***/

#include "matrix.hh"
#include "routines.hh"

namespace Math2D {

  template<>
  float Matrix<float>::max() const
  {
    return Routines::max(Storage2D<float>::data_, Storage2D<float>::size_);
  }

  template<>
  float Matrix<float>::min() const
  {
    return Routines::min(Storage2D<float>::data_, Storage2D<float>::size_);
  }

  template<>
  void Matrix<float>::operator*=(const float scalar)
  {
    Routines::mul_array(Storage2D<float>::data_, Storage2D<float>::size_, scalar);
  }

  template<>
  void Matrix<double>::operator*=(const double scalar)
  {
    Routines::mul_array(Storage2D<double>::data_, Storage2D<double>::size_, scalar);
  }
}
