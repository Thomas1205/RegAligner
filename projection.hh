/**** written by Thomas Schoenemann as an employee of Lund University, July 2010 ****/

#ifndef PROJECTION_HH
#define PROJECTION_HH

#include "storage1D.hh"

template <typename T>
inline void projection_on_simplex(T* data, const uint nData) {

  /**** projection on a simplex [Michelot 1986]****/

  assert(nData > 0);
  
  uint nNonZeros = nData;

  //TEST
  uint iter = 0;
  int start = 0;
  int end = nData-1;
  //END_TEST

  while (true) {

    //TEST
    iter++;
      
    if ((iter % 5) == 0) {

      while (data[start] == 0.0)
        start++;
      while(data[end] == 0.0)
        end--;

      assert(start <= end);
      assert(end >= 0);
      assert(start < int(nData));
    }
    //END_TEST

    //a) project onto the plane
    T mean_dev = - 1.0;

    //for (uint k=0; k < nData; k++)
    for (int k=start; k <= end; k++) {
      mean_dev += data[k];
      assert(fabs(data[k]) < 1e75);
    }
    
    mean_dev /= nNonZeros;
    assert(!isnan(mean_dev));
      
    //b) subtract mean
    bool all_pos = true;

    const bool first_iter = (nNonZeros == nData);

    //for (uint k=0; k < nData; k++) {
    for (int k=start; k <= end; k++) {


      T temp = data[k];

      if (first_iter || temp != 0.0) {
        temp -= mean_dev;

        if (temp < 1e-12) {
          all_pos = false;
          temp = 0.0;
          nNonZeros--;
        }
        data[k] = temp;
      }
    }

    if (all_pos)
      break;
  }
}

template<typename T>
inline void projection_on_simplex_with_slack(T* data, T& slack, uint nData) {

  uint nNonZeros = nData + 1;

  while (nNonZeros > 0) {
      
    //a) project onto the plane
    T mean_dev = - 1.0 + slack;
    for (uint k=0; k < nData; k++) {
      mean_dev += data[k];
      assert(fabs(data[k]) < 1e75);
    }
    
    mean_dev /= nNonZeros;
      
    //b) subtract mean
    bool all_pos = true;
    
    if (nNonZeros == (nData+1) || slack != 0.0) {
      slack -= mean_dev;

      if (slack < 0.0)
        all_pos = false;
    }

    for (uint k=0; k < nData; k++) {
      
      if (nNonZeros == (nData+1) || data[k] != 0.0) {
        data[k] -= mean_dev;
	
        if (data[k] < 0.0)
          all_pos = false;
      }
    }
    
    if (all_pos)
      break;
    
    //c) fix negatives to 0
    nNonZeros = nData+1;
    if (slack < 1e-8) {
      slack = 0.0;
      nNonZeros--;
    }

    for (uint k=0; k < nData; k++) {
	
      if (data[k] < 1e-8) {
        data[k] = 0.0;
        nNonZeros--;
      }
    }
  }
}


template <typename T>
inline void fast_projection_on_simplex(T* data, const uint nData) {

  //std::cerr << "fast proj." << std::endl;

  //as in [Duchi et al. ICML 2008] and [Shalev-Shwartz and Singer JMLR '06]
  assert(nData >= 1);

  Storage1D<std::pair<T,uint> > sorted(nData);
  for (uint k=0; k < nData; k++)
    sorted[k] = std::make_pair(data[k],k);

  std::sort(sorted.direct_access(),sorted.direct_access()+nData); //highest values will be at the back now

  T sum = sorted[nData-1].first - 1.0; //we already subtract 1.0 here

  uint nPos = 1;
  int k_break = nData-2;
  while (k_break >= 0) {

    double cur = sorted[k_break].first;
    double hyp_sum = sum+cur;

    if ((cur - hyp_sum / nPos) <= 1e-6)
      break;
    
    sum = hyp_sum;
    k_break--;
    nPos++;
  }

  double sub = sum / nPos;

  for (int k=0; k <= k_break; k++)
    data[sorted[k].second] = 0.0;
  for (int k=k_break+1; k < int(nData); k++) {
    //data[sorted[k].second] = sorted[k].first - sub;
    data[sorted[k].second] = std::max(sorted[k].first - sub,0.0);
    assert(data[sorted[k].second] >= 0.0);
  }
  //data[sorted[k].second] -= sub;

}

template<typename T>
inline void fast_projection_on_simplex_with_slack(T* data, T& slack, uint nData) {

  //std::cerr << "fast proj." << std::endl;

  //as in [Duchi et al. ICML 2008] and [Shalev-Shwartz and Singer JMLR '06]

  Storage1D<std::pair<T,uint> > sorted(nData+1);
  for (uint k=0; k < nData; k++)
    sorted[k] = std::make_pair(data[k],k);
  sorted[nData] = std::make_pair(slack,nData);


  std::sort(sorted.direct_access(),sorted.direct_access()+nData+1); //highest values will be at the back now

  T sum = sorted[nData].first - 1.0; //we already subtract 1.0 here

  uint nPos = 1;
  int k_break = nData-1;
  while (k_break >= 0) {

    double cur = sorted[k_break].first;
    double hyp_sum = sum+cur;

    if ((cur - hyp_sum / nPos) <= 1e-6)
      break;
    
    sum = hyp_sum;
    k_break--;
    nPos++;
  }

  double sub = sum / nPos;

  for (int k=0; k <= k_break; k++) {
    const uint idx = sorted[k].second;
    if (idx < nData) 
      data[idx] = 0.0;
    else
      slack = 0.0;
  }
  for (int k=k_break+1; k < int(nData+1); k++) {
    const uint idx = sorted[k].second;
    if (idx < nData) {
      //data[idx] = sorted[k].first - sub;
      data[idx] = std::max(0.0, sorted[k].first - sub);
      //data[idx] -= sub;
      assert(data[idx] >= 0.0);
    }
    else {
      //slack -= sub;
      slack = std::max(0.0, slack - sub);

      assert(slack >= 0.0);
    }
  }

}

#endif
