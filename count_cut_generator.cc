/******** written by Thomas Schoenemann as an employee of Lund University, Sweden, 2011 ********/

#include "count_cut_generator.hh"


CountCutGenerator::CountCutGenerator(const SparseMatrixDescription<double>& lp_descr, 
				     const Math1D::Vector<uint>& row_start,
				     const Math2D::Matrix<uint>& active_rows, 
				     double non_count_sign) : 
  lp_descr_(lp_descr), row_start_(row_start), active_rows_(active_rows), non_count_sign_(non_count_sign) {
}


/*virtual*/ CglCutGenerator * CountCutGenerator::clone() const {

  return new CountCutGenerator(lp_descr_,row_start_,active_rows_,non_count_sign_);
}

/*virtual*/ void CountCutGenerator::generateCuts( const OsiSolverInterface & si, OsiCuts & cs,
						  const CglTreeInfo info) {


  //4 - set global cut flag if at root node
  //8 - set global cut flag if at root node and first pass
  //16 - set global cut flag and make cuts globally valid

  if ( (info.options & (4+8+16) ) != 0) //these cuts are not globally valid when not derived at the root node
    return;

  //std::cerr << "**** CountCutGenerator::generateCuts()" << std::endl;

  const double* cur_lp_solution = si.getColSolution();
  const uint nSolverVars = si.getNumCols();

  for (uint r = 0; r < active_rows_.xDim(); r++) {

    //std::cerr << "r: " << r << std::endl;

    uint row = active_rows_(r,0);
    uint first_var = active_rows_(r,1);
    assert(first_var < nSolverVars);
    uint last_var = std::min(active_rows_(r,2),nSolverVars-1);
    uint first_count = 0;
    if (active_rows_.yDim() >= 4)
      first_count = active_rows_(r,3);

    assert(last_var >= first_var);
    uint nCountVars = last_var - first_var + 1;

    //a) make a list of the relevant vars on the other side of the constraint
    uint row_size = row_start_[row+1] - row_start_[row];

    //std::cerr << "A" << std::endl;

    std::vector<std::pair<double,uint> > vec;
    vec.reserve(row_size);

    for (uint v=row_start_[row]; v < row_start_[row+1]; v++) {

      const double val = lp_descr_.value()[v];
      const uint idx = lp_descr_.col_indices()[v];
      if (idx >= first_var && idx <= last_var)
	assert(fabs(val) == (idx-first_var+first_count));

      if (val * non_count_sign_ > 0.0) {

	assert( !(idx >= first_var && idx <= last_var) );
	assert( fabs(val) == 1.0);

	vec.push_back(std::make_pair(cur_lp_solution[idx],idx));
      }
    }

    if (vec.size() <= 1)
      continue;

    // if (nCountVars != (nUsed+1)) {
    //   std::cerr << "WARNING: mismatch of counts: " << nCountVars << " counts, " << nUsed << " entries on the other side" << std::endl;
    //   std::cerr << "row_size: " << row_size << std::endl;
    // }
    
    //std::cerr << "B" << std::endl;

    //b) sort the values
    std::sort(vec.begin(),vec.end(),Makros::first_higher<double,uint>());

    // for (uint k1 = 0; k1 < nUsed-1; k1++) {
    //   for (uint k2 = 0; k2 < nUsed-1-k1; k2++) {
	
    // 	if (var_values[k2] < var_values[k2+1]) {
	  
    // 	  std::swap(var_values[k2],var_values[k2+1]);
    // 	  std::swap(var_indices[k2],var_indices[k2+1]);
    // 	}
    //   }
    // }

    //std::cerr << "C" << std::endl;

    //c) scan for violated forward cuts and add the inequalities
    double fwd_running_sum = 0.0;
    double fwd_count_var_sum = 0.0;

    for (uint c=0; c < first_count; c++)
      fwd_running_sum += vec[c].first;// var_values[c];

    for (uint c=first_count; c < first_count+nCountVars; c++) {

      //if (c >= nUsed)
      if (c >= vec.size())
	break;

      fwd_running_sum += vec[c].first; //var_values[c];
      fwd_count_var_sum += cur_lp_solution[c];

      if (fwd_running_sum + fwd_count_var_sum >= (first_count + c + 1.01)) {

	const uint nVarsInCut = c+1 + (c-first_count)+1;

	//std::cerr << "nVarsInCut: "<< nVarsInCut << std::endl;
	
	int* col_idx = new int[nVarsInCut];
	double* coeff = new double[nVarsInCut];
	
	std::fill_n(coeff,nVarsInCut,1.0); 
	
	for (uint nn=0; nn <= c; nn++) {
	  col_idx[nn] = vec[nn].second; //var_indices[nn];
	}
	for (uint nn=first_count; nn <= c; nn++)
	  col_idx[(c+1) + nn-first_count] = first_var + nn-first_count;
		
	OsiRowCut newCut;
	newCut.setRow(nVarsInCut,col_idx,coeff,false);
	newCut.setLb(0.0);
	newCut.setUb(c+1);   
	
	cs.insert(newCut);  
		
	delete[] col_idx;
	delete[] coeff;
      }
    }

    //d) scan for violated backward cuts and add the inequalities
    if (first_count == 0 && nCountVars == vec.size()+1) {  //TODO: think about making this more general
      double bwd_running_sum = 0.0;
      double bwd_count_var_sum = 0.0;

      //std::cerr << "checking backward cuts" << std::endl;
      
      for (uint c=0; c < nCountVars; c++) {
	
	//if (c >= nUsed)
	if (c >= vec.size())
	  break;
	
	bwd_running_sum += vec[vec.size() - 1 - c].first; //var_values[nUsed - 1 - c];
	bwd_count_var_sum += cur_lp_solution[last_var-c];

	if (bwd_running_sum + bwd_count_var_sum >= (c + 1.01)) {
	  
	  //std::cerr << "backward cut with size " << (2*c+2) << std::endl;

	  //add the cut
	  int* col_idx = new int[2*c+2];
	  double* coeff = new double[2*c+2];
	  
	  for (uint k=0; k <= c; k++) {
	    col_idx[k] = last_var-k;
	    coeff[k] = 1.0;
	  }

	  for (uint k=0; k <= c; k++) {
	    col_idx[c+1 + k] = vec[vec.size() - 1 - k].second; //var_indices[nUsed - 1 - k];
	    coeff[c+1+k] = -1.0;
	  }
		
	  OsiRowCut newCut;
	  newCut.setRow(2*c+2,col_idx,coeff,false);
	  newCut.setLb(-1e20);
	  newCut.setUb(0.0);   
	  
	  delete[] col_idx;
	  delete[] coeff;
	  
	  cs.insert(newCut); 
	}
      }
    }
  }

  //std::cerr << "leaving" << std::endl;
}


/*********************************************************************************/

CountColCutGenerator::CountColCutGenerator(const SparseMatrixDescription<double>& lp_descr, 
					   const Math1D::Vector<uint>& row_start,
					   const Math2D::Matrix<uint>& active_rows, 
					   double non_count_sign) : 
  lp_descr_(lp_descr), row_start_(row_start), active_rows_(active_rows), non_count_sign_(non_count_sign) {
}

/*virtual*/ CglCutGenerator * CountColCutGenerator::clone() const {
  
  return new CountColCutGenerator(lp_descr_,row_start_,active_rows_,non_count_sign_);
}

/*virtual*/ void CountColCutGenerator::generateCuts( const OsiSolverInterface & si, OsiCuts & cs,
						     const CglTreeInfo info) {

  if (info.pass > 0)
    return; //no use applying this several times per node

  //std::cerr << "**** CountCutGenerator::generateCuts()" << std::endl;

  //const double* cur_lp_solution = si.getColSolution();

  std::vector<int> var_idx;
  //std::vector<double> var_ub;

  const uint nSolverVars = si.getNumCols();
  const double* colLower = si.getColLower();
  const double* colUpper = si.getColUpper();

  for (uint r = 0; r < active_rows_.xDim(); r++) {

    //std::cerr << "r: " << r << std::endl;

    uint row = active_rows_(r,0);
    uint first_var = active_rows_(r,1);
    assert(first_var < nSolverVars);
    uint last_var = std::min(active_rows_(r,2),nSolverVars-1);
    uint first_count = 0;
    if (active_rows_.yDim() >= 4)
      first_count = active_rows_(r,3);

    assert(last_var >= first_var);
    uint nCountVars = last_var - first_var + 1;    

    uint nFree = 0;
    uint nOne = 0;
    //uint nZero = 0;
 
    for (uint v=row_start_[row]; v < row_start_[row+1]; v++) {

      const double val = lp_descr_.value()[v];
      const uint idx = lp_descr_.col_indices()[v];

      if (idx >= first_var && idx <= last_var)
	assert(fabs(val) == (idx-first_var+first_count));

      if (val * non_count_sign_ > 0.0) {

	assert( !(idx >= first_var && idx <= last_var) );
	assert( fabs(val) == 1.0);
	
	if (colLower[idx] >= 0.99)
	  nOne++;
	else if (colUpper[idx] <= 0.01)
	  ; //nZero++;
	else
	  nFree++;
      }
    }

    //count vars for counts < nOne can be fixed to 0
    for (uint k=first_count; k < nOne; k++) {

      uint idx = first_var+k-first_count;
      if (idx >= nSolverVars)
	continue;

      if (colUpper[idx] >= 0.99) {
	var_idx.push_back(idx);
	//var_ub.push_back(0.0);
      }
    }

    //count vars for counts > nOne+nFree can be fixed to 0
    for (uint k=nOne+nFree+1; k < nCountVars+first_count; k++) {

      uint idx = first_var+k-first_count;
      if (idx >= nSolverVars)
	continue;

      assert(idx <= last_var);

      if (colUpper[idx] >= 0.99) {
	var_idx.push_back(idx);
	//var_ub.push_back(0.0);
      }      
    }
  }  

  if (var_idx.size() > 0) {

    std::vector<double> var_ub(var_idx.size(),0.0);

    OsiColCut newCut;
    newCut.setUbs(var_idx.size(),var_idx.data(),var_ub.data());

    cs.insert(newCut); 
  }
}
