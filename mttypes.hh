/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** refined at the University of Düsseldorf, Germany 2012 ***/

#ifndef MTTYPES_HH
#define MTTYPES_HH

#include "vector.hh"
#include "matrix.hh"
#include "tensor.hh"

#include <set>
#include <map>

using WordClassType = ushort;
using AlignBaseType = ushort;

using SingleWordDictionary = NamedStorage1D<Math1D::Vector<double> >;
using UnnamedSingleWordDictionary = Storage1D<Math1D::Vector<double> >;

using floatSingleWordDictionary = NamedStorage1D<Math1D::Vector<float> >;
using floatUnnamedSingleWordDictionary = Storage1D<Math1D::Vector<float> >;

using CooccuringWordsType = NamedStorage1D<Math1D::Vector<uint> >;

using RefAlignmentStructure = std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > >;
using AlignmentStructure = std::set<std::pair<AlignBaseType,AlignBaseType> >;

//access: [target length][source length](source pos, target pos)
using IBM2AlignmentModel = NamedStorage1D<Storage1D<Math2D::Matrix<double> > >;
//this gets rid of the dependence on the length of the source sentence
using ReducedIBM2AlignmentModel = NamedStorage1D<Math2D::Matrix<double> >;
using ReducedIBM2ClassAlignmentModel = NamedStorage1D<Math3D::Tensor<double> >;

using Bi2AlignmentModel = NamedStorage1D<Math3D::Tensor<double> >;

using CooccuringLengthsType = NamedStorage1D<Math1D::Vector<uint> >;

using FullHMMAlignmentModel = NamedStorage1D<Math2D::Matrix<double> >;
using FullHMMAlignmentModelNoClasses = NamedStorage1D<Math2D::Matrix<double> >;
using FullHMMAlignmentModelSingleClass = NamedStorage1D<Math3D::Tensor<double> >;
using InitialAlignmentProbability = NamedStorage1D<Math1D::Vector<double> >;

using ReducedIBM3DistortionModel = NamedStorage1D<Math2D::Matrix<double> >;
using ReducedIBM3ClassDistortionModel = NamedStorage1D<Math3D::Tensor<double> >;

//indexed by (source word class idx, target word class idx, displacement)
using IBM4CeptStartModel = Math3D::NamedTensor<double>;
//indexed by (source word class, displacement)
using IBM4WithinCeptModel = Math2D::NamedMatrix<double>;

using SingleLookupTable = Math2D::Matrix<uint,ushort>;
using LookupTable = Storage1D<SingleLookupTable>;

enum TransferMode {TransferNo, TransferViterbi, TransferPosterior, TransferInvalid};

enum HmmInitProbType {HmmInitFix, HmmInitNonpar, HmmInitPar, HmmInitFix2, HmmInitInvalid};

enum HmmAlignProbType {HmmAlignProbNonpar, HmmAlignProbFullpar, HmmAlignProbReducedpar, HmmAlignProbNonpar2, HmmAlignProbInvalid};

enum IBM23ParametricMode {IBM23ParByPosition, IBM23ParByDifference, IBM23Nonpar};

enum MStepSolveMode {MSSolvePGD, MSSolveLBFGS, MSSolveGD, MSSolvePLBFGS};

enum ProjectionMode {Simplex, PosOrthant};

enum IlpMode {IlpOff, IlpComputeOnly, IlpCenter};

enum FertNullModel {FertNullNondeficient, FertNullOchNey, FertNullIntra};

enum IBM4CeptStartMode { IBM4CENTER, IBM4FIRST, IBM4LAST, IBM4UNIFORM};

//what target word to condition on. Previous is as proposed by Brown et al.
enum IBM4InterDistMode {IBM4InterDistModePrevious, IBM4InterDistModeCurrent}; 

//what word to condition on for the intra probability. Source is as proposed by Brown et al.
enum IBM4IntraDistMode {IBM4IntraDistModeSource, IBM4IntraDistModeTarget};

#endif
