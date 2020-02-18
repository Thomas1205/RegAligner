/**** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#include "tensor.hh"
#include "routines.hh"

namespace Math3D {

  template<>
  float Tensor<float>::max() const
  {
    return Routines::max(Storage3D<float>::data_, Storage3D<float>::size_);
  }

  template<>
  float Tensor<float>::min() const
  {
    return Routines::min(Storage3D<float>::data_, Storage3D<float>::size_);
  }

  template<>
  void Tensor<float>::operator*=(const float scalar)
  {
    Routines::mul_array(Storage3D<float>::data_, Storage3D<float>::size_, scalar);
  }

  template<>
  void Tensor<double>::operator*=(const double scalar)
  {
    Routines::mul_array(Storage3D<double>::data_, Storage3D<double>::size_, scalar);
  }

}

