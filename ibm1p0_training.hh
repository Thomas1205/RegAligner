/**** Written by Thomas Schoenemann as a private person, November 2019 ****/

#ifndef IBM1P0_TRAINING_HH
#define IBM1P0_TRAINING_HH

#include "ibm1_training.hh"

void train_ibm1p0(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup, const Storage1D<Math1D::Vector<uint> >& target,
                  const CooccuringWordsType& cooc, SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                  const IBM1Options& options);

void train_ibm1p0_gd_stepcontrol(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                                 const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& cooc,
                                 SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                                 const IBM1Options& options);

void ibm1p0_viterbi_training(const Storage1D<Math1D::Vector<uint> >& source, const LookupTable& slookup,
                             const Storage1D<Math1D::Vector<uint> >& target, const CooccuringWordsType& cooc,
                             SingleWordDictionary& dict, const floatSingleWordDictionary& prior_weight,
                             const IBM1Options& options, const Math1D::Vector<double>& xlogx_table);

#endif
