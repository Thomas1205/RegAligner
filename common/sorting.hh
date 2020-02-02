/**** written by Thomas Schoenemann as a private person without employment, May 2013 ****/

#ifndef SORTING_HH
#define SORTING_HH

//NOTE: all routines are so far fixed to operator<

#ifndef MERGE_THRESH
#define MERGE_THRESH 256
#endif

#include "storage1D.hh"

template<typename T>
bool is_sorted(const T* data, const size_t nData);

template<typename T>
bool is_reverse_sorted(const T* data, const size_t nData);

template<typename T>
bool is_unique_sorted(const T* data, const size_t nData);

template<typename T>
bool is_unique_reverse_sorted(const T* data, const size_t nData);

/***** plain sorting ****/

template<typename T>
inline void bubble_sort(T* data, const size_t nData);

template<typename T>
inline void batch_bubble_sort(T* data, const size_t nData);

template<typename T>
inline void bubble_sort_prepro(T* data, const size_t nData);

template<typename T>
inline void shift_bubble_sort(T* data, const size_t nData);

template<typename T>
void merge_sort(T* data, const size_t nData);

/***** key-value sort *****/

template<typename T1, typename T2>
inline void bubble_sort_key_value(T1* key, T2* value, const size_t nData);

template<typename T1, typename T2>
inline void batch_bubble_sort_key_value(T1* key, T2* value, const size_t nData);

template<typename T1, typename T2>
inline void bubble_sort_key_value_prepro(T1* key, T2* value, const size_t nData);

template<typename T1, typename T2>
inline void shift_bubble_sort_key_value(T1* key, T2* value, const size_t nData);

template<typename T1, typename T2>
inline void merge_sort_key_value(T1* key, T2* value, const size_t nData);

/***** index sorting *****/

template<typename T, typename ST>
inline void index_bubble_sort(const T* data, ST* indices, ST nData);

template<typename T, typename ST>
inline void index_batch_bubble_sort(const T* data, ST* indices, ST nData);

template<typename T, typename ST>
void index_merge_sort(const T* data, ST* indices, const ST nData);

template<typename T, typename ST>
void index_quick_sort(const T* data, ST* indices, const ST nData);


/************ implementation  *************/

template<typename T>
bool is_sorted(const T* data, const size_t nData) 
{
  for (size_t i=1; i < nData; i++) {
    if (data[i] < data[i-1])
      return false;
  }

  return true;
}

template<typename T>
bool is_reverse_sorted(const T* data, const size_t nData)
{
  for (size_t i=1; i < nData; i++) {
    if (data[i-1] < data[i])
      return false;
  }

  return true;
}

template<typename T>
bool is_unique_sorted(const T* data, const size_t nData) 
{
  for (size_t i=1; i < nData; i++) {
    if (data[i] <= data[i-1])
      return false;
  }

  return true;
}

template<typename T>
bool is_unique_reverse_sorted(const T* data, const size_t nData)
{
  for (size_t i=1; i < nData; i++) {
    if (data[i-1] <= data[i])
      return false;
  }

  return true;  
}

/**** plain ***/

template<typename T>
inline void bubble_sort(T* data, const size_t nData) 
{
  size_t last_flip = nData;
  
  while (last_flip > 1) {
    
    size_t flip = 0;
    
    for (size_t l=1; l < last_flip; l++) {

      if (data[l] < data[l-1]) {
        std::swap(data[l],data[l-1]);
        flip = l;
      }      
    }  

    last_flip = flip;
  }
}

template<typename T>
inline void batch_bubble_sort(T* data, const size_t nData) 
{
  size_t last_flip = nData;
  
  while (last_flip > 1) {
    
    size_t flip = 0;
    
    for (size_t l=1; l < last_flip; l++) {

      if (data[l] < data[l-1]) {

        size_t end = l+1;
        for (; end < last_flip; end++) {
                                        
          if (!(data[end] < data[end-1])) 
            break;
        }

        std::reverse(data+l-1, data+end);        
        
        flip = l;
        l = end-2; //the last swapped index needs to be compared again (l is incremented by the loop)
      }      
    }  

    last_flip = flip;
  }
}


template<typename T>
inline void bubble_sort_prepro(T* data, const size_t nData)
{
  const size_t thresh = 5;
  
  if (nData > thresh) {
    //try to ameliorate some bad cases
    for (uint i = nData-1; i >= thresh-1; i--) {
      if ( data[i] < data[0] )
        std::swap(data[i],data[0]);
    }
  }  
}

template<typename T>
inline void shift_bubble_sort(T* data, const size_t nData) 
{     
  for (size_t l=1; l < nData; l++) 
  {
#if 1
    const T curdat = data[l];
    size_t inspos = l;
    while (inspos > 0 && curdat < data[inspos-1]) 
      inspos--;
    
    if (inspos != l) {      
      for (size_t ll = l; ll > inspos; ll--)
        data[ll] = data[ll-1];
      data[inspos] = curdat;
    }
#else
    if (data[l] < data[l-1]) {
        
      std::swap(data[l],data[l-1]);
      for (size_t ll = l-1; ll > 0; ll--) {
        if (data[ll] < data[ll-1]) 
          std::swap(data[ll],data[ll-1]);
        else
          break;
      }
    }
#endif
  }  
}

template<typename T>
void merge_sort(T* data, const size_t nData) 
{
  if (nData <= MERGE_THRESH)
    bubble_sort(data, nData);
  else {

    const size_t half = nData / 2;
    const size_t nData2 = nData-half;

    T* aux_data = new T[nData];

    Makros::unified_assign(aux_data, data, half);
    //memcpy(aux_data,data,half*sizeof(T));

    merge_sort(aux_data,half);

    Makros::unified_assign(aux_data+half, data+half, nData2);
    //memcpy(aux_data+half,data+half,nData2*sizeof(T));

    merge_sort(data+half,nData2);

    //now merge the lists
    
    const T* data1 = aux_data;
    const T* data2 = aux_data+half;

    size_t k1=0;
    size_t k2=0;
    size_t k=0;

    //while(k1 < half && k2 < nData2) {
    while (true) {

      const T d1 = data1[k1];
      const T d2 = data2[k2];
      if (d1 <= d2) {
        data[k] = d1;
        k1++;
        if (k1 >= half)
          break;
      }
      else {
        data[k] = d2;
        k2++;
        if (k2 >= nData2)
          break;
      }

      k++;
    }

    //copy the rest
    Makros::unified_assign(data+k, data1+k1, half-k1);
    //memcpy(data+k,data1+k1,(half-k1)*sizeof(ST));
    Makros::unified_assign(data+k, data2+k1, nData2-k2);
    //memcpy(data+k,data2+k2,(nData2-k2)*sizeof(ST));

    delete[] aux_data;    
  }
}

/**** key-value ****/

template<typename T1, typename T2>
inline void bubble_sort_key_value(T1* key, T2* value, const size_t nData) 
{
  size_t last_flip = nData;

  while (last_flip > 1) {

    size_t flip = 0;

    for (size_t j=1; j < last_flip; j++) {

      if (key[j] < key[j-1]) {

        std::swap(key[j],key[j-1]);
        std::swap(value[j],value[j-1]);
        flip = j;
      }
    }

    last_flip = flip;    
  }
}

template<typename T1, typename T2>
inline void batch_bubble_sort_key_value(T1* key, T2* value, const size_t nData) {

  size_t last_flip = nData;
  
  while (last_flip > 1) {

    //std::cerr << "*** keys: ";
    //for (uint k = 0; k < nData; k++)
    //  std::cerr << key[k] << " ";
    //std::cerr << std::endl;
    //std::cerr << "last flip: " << last_flip << std::endl;
    
    size_t flip = 0;
    
    for (size_t l=1; l < last_flip; l++) {

      //std::cerr << "l: " << l << std::endl;

      if (key[l] < key[l-1]) {

        size_t end = l+1;
        for (; end < last_flip; end++) {
          if (!(key[end] < key[end-1])) 
            break;
        }

        //std::cerr << "reversing from " << (l-1) << " to excl. " << end << std::endl;

        std::reverse(key+l-1, key+end);
        std::reverse(value+l-1, value+end);
        flip = l;
        l = end-2; //the last swapped index needs to be compared again (l is incremented by the loop)
      }            
    }  

    //std::cerr << "*** keys after: ";
    //for (uint k = 0; k < nData; k++)
    //  std::cerr << key[k] << " ";
    //std::cerr << std::endl;

    last_flip = flip;
  }
}

template<typename T1, typename T2>
inline void bubble_sort_key_value_prepro(T1* key, T2* value, const size_t nData)
{
  const size_t thresh = 5;
  
  if (nData > thresh) {
    //try to ameliorate some bad cases
    for (uint i = nData-1; i >= thresh-1; i--) {
      if (key[i] < key[0]) {
        std::swap(key[i],key[0]);
        std::swap(value[i],value[0]);
      }
    }
  }  
}

template<typename T1, typename T2>
inline void shift_bubble_sort_key_value(T1* key, T2* value, const size_t nData)
{
  for (size_t j=1; j < nData; j++) 
  {
#if 1
    const T1 curkey = key[j];
    size_t inspos = j;
    while (inspos > 0 && curkey < key[inspos-1]) 
      inspos--;
    
    if (inspos != j) {      
      const T2 curval = value[j];
      for (size_t jj = j; jj > inspos; jj--) {
        key[jj] = key[jj-1];
        value[jj] = value[jj-1];
      }
      key[inspos] = curkey;
      value[inspos] = curval;
    }
#else
    if (key[j] < key[j-1]) {

      std::swap(key[j],key[j-1]);
      std::swap(value[j],value[j-1]);
        
      for (size_t jj = j-1; jj > 0; jj--) {
        if (key[jj] < key[jj-1]) {
          std::swap(key[jj],key[jj-1]);
          std::swap(value[jj],value[jj-1]);
        }
        else
          break;
      }
    }
#endif    
  }
}

template<typename T1, typename T2>
inline void merge_sort_key_value(T1* key, T2* value, const size_t nData) 
{
  if (nData <= MERGE_THRESH)
    bubble_sort(key, value, nData);
  else {

    const size_t half = nData / 2;
    const size_t nData2 = nData-half;

    T1* aux_key = new T1[nData];
    T2* aux_value = new T2[nData];

    Makros::unified_assign(aux_key, key, half);
    Makros::unified_assign(aux_value, value, half);

    merge_sort(aux_key,aux_value,half);

    Makros::unified_assign(aux_key+half, key+half, nData2);
    Makros::unified_assign(aux_value+half, value+half, nData2);

    merge_sort(aux_key+half,aux_value,half,nData2);

    //now merge the lists
    
    const T1* key1 = aux_key;
    const T1* key2 = aux_key+half;
    const T2* value1 = aux_value;
    const T2* value2 = aux_value+half;    

    size_t k1=0;
    size_t k2=0;
    size_t k=0;

    //while(k1 < half && k2 < nData2) {
    while (true) {

      const T1 d1 = key1[k1];
      const T1 d2 = key2[k2];
      if (d1 <= d2) {
        key[k] = d1;
        value[k] = value1[k1];
        k1++;
        if (k1 >= half)
          break;
      }
      else {
        key[k] = d2;
        value[k] = value2[k2];
        k2++;
        if (k2 >= nData2)
          break;
      }

      k++;
    }

    //copy the rest
    Makros::unified_assign(key+k, key1+k1, half-k1);
    Makros::unified_assign(value+k, value1+k1, half-k1);
    Makros::unified_assign(key+k, key2+k1, nData2-k2);
    Makros::unified_assign(value+k, value2+k1, nData2-k2);

    delete[] aux_key;
    delete[] aux_value;
  }
}

/**** index ****/

template<typename T, typename ST>
inline void index_bubble_sort(const T* data, ST* indices, const ST nData)
{
  for (uint i=0; i < nData; i++)
    indices[i] = i;

  uint last_flip = nData;
  
  while (last_flip > 1) {
    
    uint flip = 0;
    
    for (uint l=1; l < last_flip; l++) {

      if (data[indices[l]] > data[indices[l-1]]) {
        std::swap(indices[l],indices[l-1]);
        flip = l;
      }      
    }    
  }

  // for (uint k=0; k < nData-1; k++) {
    // std::cerr << "k: " << k << "/" << nData << std::endl;

    // for (uint l=0; l < nData-1-k; l++) {
      // if (data[indices[l]] > data[indices[l+1]])
        // std::swap(indices[l],indices[l+1]);
    // }
  // }
}

template<typename T, typename ST>
inline void index_batch_bubble_sort(const T* data, ST* indices, ST nData)
{
  size_t last_flip = nData;
  
  while (last_flip > 1) {
    
    size_t flip = 0;
    
    for (size_t l=1; l < last_flip; l++) {

      if (data[indices[l]] < data[indices[l-1]]) {

        size_t end = l+1;
        for (; end < last_flip; end++) {
                                        
          if (!(data[indices[end]] < data[indices[end-1]])) 
            break;
        }

        std::reverse(indices+l-1, indices+end);        
        
        flip = l;
        l = end-2; //the last swapped index needs to be compared again (l is incremented by the loop)
      }      
    }  

    last_flip = flip;
  }
}

template<typename T, typename ST>
void aux_index_quick_sort(const T* data, ST* indices, const ST nData);

template<typename T, typename ST>
void aux_index_merge_sort(const T* data, ST* indices, const ST nData)
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
  if (nData <= MERGE_THRESH) {

    aux_index_quick_sort(data,indices,nData);
  }
  else {

    const ST half = nData / 2;
    const ST nData2 = nData-half;

    ST* aux_indices = new ST[nData];

    Makros::unified_assign(aux_indices, indices, half);
    //memcpy(aux_indices,indices,half*sizeof(ST));

    aux_index_merge_sort(data,aux_indices,half);
    //aux_index_quick_sort(data,aux_indices,half);

    Makros::unified_assign(aux_indices+half, indices+half, nData2);
    //memcpy(aux_indices+half,indices+half,nData2*sizeof(ST));

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

    //copy the rest
    Makros::unified_assign(indices+k, index1+k1, half-k1);
    //memcpy(indices+k,index1+k1,(half-k1)*sizeof(ST));
    Makros::unified_assign(indices+k,index2+k2,nData2-k2);
    //memcpy(indices+k,index2+k2,(nData2-k2)*sizeof(ST));

    delete[] aux_indices;
  }
}

//this is still fixed to memcpy
template<typename T, typename ST>
void aux_index_merge_sort_4split(const T* data, ST* indices, const ST nData)
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
void index_merge_sort(const T* data, ST* indices, const ST nData)
{
  //std::cerr << "sorting " << nData << " entries" << std::endl;

  for (uint i=0; i < nData; i++)
    indices[i] = i;

  aux_index_merge_sort(data,indices,nData);
}


template<typename T, typename ST>
void aux_index_quick_sort(const T* data, ST* indices, const ST nData)
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
void index_quick_sort(const T* data, ST* indices, const ST nData)
{
  //std::cerr << "sorting " << nData << " entries" << std::endl;

  for (uint i=0; i < nData; i++)
    indices[i] = i;

  aux_index_quick_sort(data,indices,nData);
}


#endif
