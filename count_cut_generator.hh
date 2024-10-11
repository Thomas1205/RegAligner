/******** written by Thomas Schoenemann as an employee of Lund University, Sweden, 2011 ********/

#ifndef COUNT_CUT_GENERATOR
#define COUNT_CUT_GENERATOR

#include "vector.hh"
#include "matrix.hh"
#include "sparse_matrix_description.hh"
#include "CglGomory.hpp"


class CountCutGenerator : public CglCutGenerator {
public:

  CountCutGenerator(const SparseMatrixDescription<double>& lp_descr, const Math1D::Vector<uint>& row_start,
                    const Math2D::Matrix<uint>& active_rows, double non_count_sign = -1.0);

  virtual void generateCuts( const OsiSolverInterface& si, OsiCuts& cs,
                             const CglTreeInfo info = CglTreeInfo());

  virtual CglCutGenerator* clone() const;

protected:

  const SparseMatrixDescription<double>& lp_descr_;
  const Math1D::Vector<uint>& row_start_;

  const Math2D::Matrix<uint>& active_rows_;  //first entry: row num. ,
  //second entry: idx of first count var,
  //third entry: count represented by first var
  //fourth entry: idx of last valid var

  double non_count_sign_;
};

class CountColCutGenerator : public CglCutGenerator {
public:

  CountColCutGenerator(const SparseMatrixDescription<double>& lp_descr, const Math1D::Vector<uint>& row_start,
                       const Math2D::Matrix<uint>& active_rows, double non_count_sign = -1.0);

  virtual void generateCuts( const OsiSolverInterface& si, OsiCuts& cs,
                             const CglTreeInfo info = CglTreeInfo());

  virtual CglCutGenerator* clone() const;


protected:

  const SparseMatrixDescription<double>& lp_descr_;
  const Math1D::Vector<uint>& row_start_;

  const Math2D::Matrix<uint>& active_rows_;  //first entry: row num. ,
  //second entry: idx of first count var,
  //third entry: count represented by first var
  //fourth entry: idx of last valid var

  double non_count_sign_;
};

#endif
