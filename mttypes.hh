/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** refined at the University of Düsseldorf, Germany 2012 ***/

#ifndef MTTYPES_HH
#define MTTYPES_HH

#include "vector.hh"
#include "matrix.hh"
#include "tensor.hh"

#include <set>
#include <map>

typedef ushort WordClassType;
typedef ushort AlignBaseType;

typedef NamedStorage1D<Math1D::Vector<double> > SingleWordDictionary;
typedef Storage1D<Math1D::Vector<double> > UnnamedSingleWordDictionary;

typedef NamedStorage1D<Math1D::Vector<float> > floatSingleWordDictionary;
typedef Storage1D<Math1D::Vector<float> > floatUnnamedSingleWordDictionary;

typedef NamedStorage1D<Math1D::Vector<uint> > CooccuringWordsType;

typedef std::map<uint,std::set<std::pair<AlignBaseType,AlignBaseType> > > RefAlignmentStructure;
typedef std::set<std::pair<AlignBaseType,AlignBaseType> > AlignmentStructure;

//access: [target length][source length](source pos, target pos)
typedef NamedStorage1D<Storage1D<Math2D::Matrix<double> > > IBM2AlignmentModel;
//this gets rid of the dependence on the length of the source sentence
typedef NamedStorage1D<Math2D::Matrix<double> > ReducedIBM2AlignmentModel;
typedef NamedStorage1D<Math3D::Tensor<double> > ReducedIBM2ClassAlignmentModel;

typedef NamedStorage1D<Math3D::Tensor<double> > Bi2AlignmentModel;

typedef NamedStorage1D<Math1D::Vector<uint> > CooccuringLengthsType;

typedef NamedStorage1D<Math2D::Matrix<double> > FullHMMAlignmentModel;
typedef NamedStorage1D<Math2D::Matrix<double> > FullHMMAlignmentModelNoClasses;
typedef NamedStorage1D<Math3D::Tensor<double> > FullHMMAlignmentModelSingleClass;
typedef NamedStorage1D<Math1D::Vector<double> > InitialAlignmentProbability;

typedef NamedStorage1D<Math2D::Matrix<double> > ReducedIBM3DistortionModel;
typedef NamedStorage1D<Math3D::Tensor<double> > ReducedIBM3ClassDistortionModel;

//indexed by (source word class idx, target word class idx, displacement)
typedef Math3D::NamedTensor<double> IBM4CeptStartModel;
//indexed by (source word class, displacement)
typedef Math2D::NamedMatrix<double> IBM4WithinCeptModel;

typedef Math2D::Matrix<uint,ushort> SingleLookupTable;
typedef Storage1D<SingleLookupTable> LookupTable;

enum TransferMode { TransferNo, TransferViterbi, TransferPosterior, TransferInvalid };

enum HmmInitProbType {HmmInitFix, HmmInitNonpar, HmmInitPar, HmmInitFix2, HmmInitInvalid};

enum HmmAlignProbType {HmmAlignProbNonpar, HmmAlignProbFullpar, HmmAlignProbReducedpar, HmmAlignProbNonpar2, HmmAlignProbInvalid};

enum IBM23ParametricMode {IBM23ParByPosition, IBM23ParByDifference, IBM23Nonpar};

enum MStepSolveMode {MSSolvePGD, MSSolveLBFGS, MSSolveGD};

enum ProjectionMode {Simplex, PosOrthant};

enum IlpMode {IlpOff, IlpComputeOnly, IlpCenter};

enum FertNullModel {FertNullNondeficient, FertNullOchNey, FertNullIntra};

enum IBM4CeptStartMode { IBM4CENTER, IBM4FIRST, IBM4LAST, IBM4UNIFORM };

//what target word to condition on. Previous is as proposed by Brown et al.
enum IBM4InterDistMode {IBM4InterDistModePrevious, IBM4InterDistModeCurrent}; 

//what word to condition on for the intra probability. Source is as proposed by Brown et al.
enum IBM4IntraDistMode {IBM4IntraDistModeSource, IBM4IntraDistModeTarget};

#endif
