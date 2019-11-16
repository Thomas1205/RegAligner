/**** written by Thomas Schoenemann as a private person without employment, May 2013 ****/

#ifndef SORTING_HH
#define SORTING_HH


template<typename T, typename ST>
void index_bubble_sort(T* data, ST* indices, ST nData);


template<typename T, typename ST>
void index_merge_sort(T* data, ST* indices, ST nData);

template<typename T, typename ST>
void index_quick_sort(T* data, ST* indices, ST nData);


/************ implementation  *************/


template<typename T, typename ST>
void index_bubble_sort(T* data, ST* indices, ST nData)
{

  for (uint i=0; i < nData; i++)
    indices[i] = i;

  for (uint k=0; k < nData-1; k++) {
    std::cerr << "k: " << k << "/" << nData << std::endl;

    for (uint l=0; l < nData-1-k; l++) {
      if (data[indices[l]] > data[indices[l+1]])
        std::swap(indices[l],indices[l+1]);
    }
  }
}

template<typename T, typename ST>
void aux_index_quick_sort(T* data, ST* indices, ST nData);

template<typename T, typename ST>
void aux_index_merge_sort(T* data, ST* indices, ST nData)
{

  //std::cerr << "nData: "<< nData << std::endl;

  // if (nData <= 12 /*8*/) {

  //   //bubble sort
  //   for (uint k=0; k < nData-1; k++) {

  //     //std::cerr << "k: " << k << std::endl;

  //     for (uint l=0; l < nData-1-k; l++) {
  // 	//std::cerr << "l: " << l << std::endl;

  // 	//std::cerr << "index1: " << indices[l] << std::endl;
  // 	//std::cerr << "index2: " << indices[l+1] << std::endl;

  // 	if (data[indices[l]] > data[indices[l+1]])
  // 	  std::swap(indices[l],indices[l+1]);
  //     }
  //   }
  // }
  if (nData <= 256) {

    aux_index_quick_sort(data,indices,nData);
  }
  else {

    const ST half = nData / 2;
    const ST nData2 = nData-half;

    ST* aux_indices = new ST[nData];

    memcpy(aux_indices,indices,half*sizeof(ST));

    aux_index_merge_sort(data,aux_indices,half);
    //aux_index_quick_sort(data,aux_indices,half);

    memcpy(aux_indices+half,indices+half,nData2*sizeof(ST));

    aux_index_merge_sort(data,aux_indices+half,nData2);
    //aux_index_quick_sort(data,aux_indices+half,nData2);

    const ST* index1 = aux_indices;
    const ST* index2 = aux_indices+half;

    ST k1=0;
    ST k2=0;
    ST k=0;

    //while(k1 < half && k2 < nData2) {
    while (true) {

      const ST idx1 = index1[k1];
      const ST idx2 = index2[k2];
      if (data[idx1] <= data[idx2]) {
        indices[k] = idx1;
        k1++;
        if (k1 >= half)
          break;
      }
      else {
        indices[k] = idx2;
        k2++;
        if (k2 >= nData2)
          break;
      }

      k++;
    }


    memcpy(indices+k,index1+k1,(half-k1)*sizeof(ST));
    memcpy(indices+k,index2+k2,(nData2-k2)*sizeof(ST));

    // while (k1 < half) {
    //   indices[k] = index1[k1];
    //   k1++;
    //   k++;
    // }
    // while (k2 < nData2) {
    //   indices[k] = index2[k2];
    //   k2++;
    //   k++;
    // }

    delete[] aux_indices;
  }
}


template<typename T, typename ST>
void aux_index_merge_sort_4split(T* data, ST* indices, ST nData)
{

  //std::cerr << "nData: "<< nData << std::endl;

  if (nData <= 8 /*8*/) {

    //bubble sort
    for (uint k=0; k < nData-1; k++) {

      //std::cerr << "k: " << k << std::endl;

      for (uint l=0; l < nData-1-k; l++) {
        //std::cerr << "l: " << l << std::endl;

        //std::cerr << "index1: " << indices[l] << std::endl;
        //std::cerr << "index2: " << indices[l+1] << std::endl;

        if (data[indices[l]] > data[indices[l+1]])
          std::swap(indices[l],indices[l+1]);
      }
    }
  }
  else if (nData <= 32) {
    aux_index_merge_sort(data,indices,nData);
  }
  else {

    uint nData1to3 = nData/4;
    uint nData4 = nData-3*nData1to3;

    Storage1D<ST> index[4];

    index[0].resize(nData1to3);
    memcpy(index[0].direct_access(),indices,nData1to3*sizeof(ST));

    aux_index_merge_sort(data,index[0].direct_access(),nData1to3);

    index[1].resize(nData1to3);
    memcpy(index[1].direct_access(),indices+nData1to3,nData1to3*sizeof(ST));

    aux_index_merge_sort(data,index[1].direct_access(),nData1to3);

    index[2].resize(nData1to3);
    memcpy(index[2].direct_access(),indices+2*nData1to3,nData1to3*sizeof(ST));

    aux_index_merge_sort(data,index[2].direct_access(),nData1to3);

    index[3].resize(nData4);
    memcpy(index[3].direct_access(),indices+3*nData1to3,nData4*sizeof(ST));

    aux_index_merge_sort(data,index[3].direct_access(),nData4);


    ST k_idx[4] = {0,0,0,0};
    ST k_limit[4] = {nData1to3,nData1to3,nData1to3,nData4};

    uint k=0;
    while (k < nData) {

      uint arg_best = MAX_UINT;
      uint idx;
      T best;
      for (uint kk=0; kk < 4; kk++) {

        const uint cur_idx = k_idx[kk];

        if (cur_idx < k_limit[kk]) {

          const uint real_idx = index[kk][cur_idx];

          const T cur_val = data[real_idx];

          if (arg_best == MAX_UINT || cur_val < best) {
            best = cur_val;
            arg_best=kk;
            idx = real_idx;
          }
        }
      }

      indices[k] = idx;
      k_idx[arg_best]++;
      k++;
    }

  }
}

template<typename T, typename ST>
void index_merge_sort(T* data, ST* indices, ST nData)
{

  //std::cerr << "sorting " << nData << " entries" << std::endl;

  for (uint i=0; i < nData; i++)
    indices[i] = i;

  aux_index_merge_sort(data,indices,nData);
}


template<typename T, typename ST>
void aux_index_quick_sort(T* data, ST* indices, ST nData)
{

  //std::cerr << "nData: " << nData << std::endl;

  if (nData <= 8 /*8*/) {

    //bubble sort
    for (uint k=0; k < nData-1; k++) {

      //std::cerr << "k: " << k << std::endl;

      for (uint l=0; l < nData-1-k; l++) {
        //std::cerr << "l: " << l << std::endl;

        //std::cerr << "index1: " << indices[l] << std::endl;
        //std::cerr << "index2: " << indices[l+1] << std::endl;

        if (data[indices[l]] > data[indices[l+1]])
          std::swap(indices[l],indices[l+1]);
      }
    }
  }
  else {

    const T pivot = data[indices[nData-1]];

    int i=0;
    int j=nData-2; //not the perfect type (in view of ST)

    // if (nData <= 75) {
    //   std::cerr << "sequence: ";
    //   for (uint k=0; k < nData; k++)
    // 	std::cerr << data[indices[k]] << " ";
    //   std::cerr << std::endl;
    // }

    while (true) {

      while(i < int(nData-1) && data[indices[i]] <= pivot)
        i++;

      while(j >= 0 && data[indices[j]] >= pivot)
        j--;

      if (i >= j)
        break;

      std::swap(indices[i],indices[j]);
      i++;
      j--;
    }

    //if (pivot < data[indices[i]])
    if (i != nData-1)
      std::swap(indices[i],indices[nData-1]);


    // if (nData <= 75) {
    //   std::cerr << "sequence after shifting: ";
    //   for (uint k=0; k < nData; k++)
    // 	std::cerr << data[indices[k]] << " ";
    //   std::cerr << std::endl;
    // }


    //std::cerr << "i: " << i << ", j: " << j << std::endl;

    //recursive calls
    const int n1 = i-1;
    const int n2 = nData-i-1;

    if (n1 > 1) { //no need to sort a single entry
      aux_index_quick_sort(data,indices,ST(n1));
    }
    if (n2 > 1) {
      aux_index_quick_sort(data,indices+i+1,ST(n2));
    }
  }

}

template<typename T, typename ST>
void index_quick_sort(T* data, ST* indices, ST nData)
{

  //std::cerr << "sorting " << nData << " entries" << std::endl;

  for (uint i=0; i < nData; i++)
    indices[i] = i;

  aux_index_quick_sort(data,indices,nData);
}


#endif
