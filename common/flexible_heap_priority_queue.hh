/************** written by Thomas Schoenemann, November 2023 **************/
/************** heap priority queues as needec by selection/tree-sort ***********/

#ifndef FLEXIBLE_HEAP_PRIORITY_QUEUE
#define FLEXIBLE_HEAP_PRIORITY_QUEUE

#include "storage1D.hh"

template<typename T, typename Less = std::less<T> >
class FlexibleHeapPriorityQueue {
public:

  FlexibleHeapPriorityQueue(size_t reserved_size);
  
  void insert(const T key);
  
  void move_insert(T&& key);
  
  void insert_first(const T key) {
	assert(size_ == 1);
	heap_[1] = key;
    size_++;
  }
  
  void insert_second(const T key) {
	
	const static Less less;
	assert(size_ == 2);
	if (less(key,heap_[1])) {
	  heap_[2] = heap_[1];
	  heap_[1] = key;
	}
	else
	  heap_[2] = key;
	size_++;
  }
  
  void insert_first_pair(T&& first, T&& second) {
	const static Less less;
	assert(size_ == 1); //empty
	if (less(second,first)) {
	  heap_[1] = second;
	  heap_[2] = first;
	}
	else {
	  heap_[1] = first;
	  heap_[2] = second;
	}
	
	size_ = 3;
  }
  
  T extract_min();

  T move_extract_min();
  
  T extract_sole_min() {
	assert(size_ == 2);
	size_ = 1;
	return heap_[1];
  }
  
  T extract_pair_min() {
	assert(size_ == 3);
	const T retval = heap_[1];
	heap_[1] = heap_[2];
	size_--;
	return retval;
  }
  
  void extract_last_pair(T& first, T& second) {
	assert(size_ == 3);
	second = std::move(heap_[2]);
    first = std::move(heap_[1]);
	size_ = 1;	
  }
  
  size_t size() const {
	return size_;
  }

protected:

  void heap_ascend(size_t heap_idx);

  void heap_descend(size_t heap_idx);

  Storage1D<T> heap_;	
  size_t size_;
};

/*********************************/

template<typename TK, typename TV, typename Less = std::less<TK> >
class FlexibleHeapPriorityQueueKeyValue {
public:

  FlexibleHeapPriorityQueueKeyValue(size_t reserved_size);

  void insert(TK&& key, TV&& value);

  void extract_min(TK& key, TV& value);

protected:
  Storage1D<TK> key_;
  Storage1D<TV> value_;

  size_t size_;
};

/******************************/

template<typename T, typename ST = size_t, typename Less = std::less<T> > 
class FlexibleHeapPriorityQueueIndexed {
public:

  FlexibleHeapPriorityQueueIndexed(ST size, const T* data) 
    : heap_(size+1), data_(data), size_(1) {}
	
  void insert(const ST idx);

  void extract_min(ST& idx);	

protected:
  Storage1D<ST> heap_;
  const T* data_;  
  
  ST size_;
};


/**************** implementation **********************/

template<typename T, typename Less>
FlexibleHeapPriorityQueue<T, Less>::FlexibleHeapPriorityQueue(size_t reserved_size) 
: heap_(reserved_size+1), size_(1)  
{
}

template<typename T, typename Less>
void FlexibleHeapPriorityQueue<T, Less>::insert(const T key) 
{
  assert(size_ < heap_.size());
#if 0
  heap_[size_] = key;
  size_++;
  
  if (size_ > 2)
    heap_descend(size_-1);
#else
  size_t i = size_;
  size_++;

  const static Less less;

  while (i > 1) {

    const size_t parent = i >> 1; //i / 2;
    if (less(key,heap_[parent])) {
      heap_[i] = std::move(heap_[parent]);
	  i = parent;
    }
    else
      break;
  }	
  
  heap_[i] = key;
#endif
}

template<typename T, typename Less>
void FlexibleHeapPriorityQueue<T, Less>::move_insert(T&& key) 
{
  assert(size_ < heap_.size());
  const static Less less;
  
  size_t i = size_;
  size_++;

  const T curkey = key;

  while (i > 1) {

    const size_t parent = i >> 1; //i / 2;
    if (less(curkey,heap_[parent])) {
      heap_[i] = std::move(heap_[parent]);
	  i = parent;
    }
    else
      break;
  }	
  
  heap_[i] = curkey;  
}

template<typename T, typename Less>
T FlexibleHeapPriorityQueue<T, Less>::extract_min()
{
  //std::cerr << "extract_min, size: " << size_ << std::endl;
	
  assert(size_ > 1);
  assert(size_ <= heap_.size());
  const T retval = std::move(heap_[1]);
  size_--;	
  
#if 0
  if (size_ > 1) {
    heap_[1] = heap_[size_];
	if (size_ > 2) {
	  //std::cerr << "calling heap_ascend" << std::endl;
	  heap_ascend(1);
	}
  }
#else
  size_t i = 1;
  const T value = std::move(heap_[size_]);
  const static Less less;
  while (true) {

    const size_t j1 = i << 1; //2*i;

    if (j1 >= size_)
      break;

    size_t better_j = j1;

    if (j1 == size_-1) {
      //only one child exists - nothing to do
    }
    else {
      //both children exist

      const size_t j2 = j1 | 1; //j1+1;

      if (less(heap_[j2],heap_[j1]))
        better_j = j2;
	}
	
    if (less(heap_[better_j], value)) {
      heap_[i] = std::move(heap_[better_j]); 
	  i = better_j;
    }
    else
      break;
  } 
  
  heap_[i] = value;
#endif
	
  return retval;
}

template<typename T, typename Less>
T FlexibleHeapPriorityQueue<T, Less>::move_extract_min()
{
  //std::cerr << "extract_min, size: " << size_ << std::endl;
	
  assert(size_ > 1);
  assert(size_ <= heap_.size());
  const T retval = std::move(heap_[1]);
  size_--;	
  
  size_t i = 1;
  T value = std::move(heap_[size_]);
  const T& cvalue = value;
  
  const static Less less;
  while (true) {

    const size_t j1 = i << 1; //2*i;

    if (j1 >= size_)
      break;

    size_t better_j = j1;

    if (j1 == size_-1) {
      //only one child exists - nothing to do
    }
    else {
      //both children exist

      const size_t j2 = j1 | 1; //j1+1;

      if (less(heap_[j2],heap_[j1]))
        better_j = j2;
	}
	
    if (less(heap_[better_j], cvalue)) {
      heap_[i] = std::move(heap_[better_j]); 
	  i = better_j;
    }
    else
      break;
  } 
  
  heap_[i] = std::move(value);
	
  return retval;
}


template<typename T, typename Less>
void FlexibleHeapPriorityQueue<T, Less>::heap_ascend(size_t idx) 
{
  const static Less less;

  size_t i = idx;
  assert(i < size_);

  const T value = heap_[i];

  while (true) {

    const size_t j1 = i << 1; //2*i;

    if (j1 >= size_)
      break;

    size_t better_j = j1;

    if (j1 == size_-1) {
      //only one child exists - nothing to do
    }
    else {
      //both children exist

      const size_t j2 = j1 | 1; // j1+1;

      if (less(heap_[j2],heap_[j1]))
        better_j = j2;
	}
	
    if (less(heap_[better_j], value)) {
      heap_[i] = std::move(heap_[better_j]); 
	  i = better_j;
    }
    else
      break;
  } 
  
  heap_[i] = value;
}

template<typename T, typename Less>
void FlexibleHeapPriorityQueue<T, Less>::heap_descend(size_t idx) 
{
  const static Less less;

  size_t i = idx;
  assert(i < size_);
  
  const T value = heap_[i];
	
  while (i > 1) {

    const size_t parent = i >> 1; //i / 2;
    if (less(value,heap_[parent])) {
      heap_[i] = std::move(heap_[parent]);
	  i = parent;
    }
    else
      break;
  }	
  
  heap_[i] = value;
}


/**************************************************************/

template<typename TK, typename TV, typename Less>
FlexibleHeapPriorityQueueKeyValue<TK,TV,Less>::FlexibleHeapPriorityQueueKeyValue(size_t reserved_size)
: key_(reserved_size+1), value_(reserved_size+1), size_(1)  
{
}

template<typename TK, typename TV, typename Less>
void FlexibleHeapPriorityQueueKeyValue<TK,TV,Less>::insert(TK&& key, TV&& value)
{
  size_t i = size_;
  size_++;

  const static Less less;

  while (i > 1) {

    const size_t parent = i >> 1; //i / 2;
    if (less(key,key_[parent])) {
      key_[i] = std::move(key_[parent]);
	  value_[i] = std::move(value_[parent]);
	  i = parent;
    }
    else
      break;
  }	
  
  key_[i] = key;
  value_[i] = value;
}

template<typename TK, typename TV, typename Less>
void FlexibleHeapPriorityQueueKeyValue<TK,TV,Less>::extract_min(TK& key, TV& value)
{
  assert(size_ > 1);
  assert(size_ <= key_.size());
  key = std::move(key_[1]);
  value = std::move(value_[1]);
  size_--;	
  
  size_t i = 1;
  TK val = std::move(key_[size_]);
  TV val2 = std::move(value_[size_]);
  const TK& cval = val;
  
  const static Less less;
  while (true) {

    const size_t j1 = i << 1; //2*i;

    if (j1 >= size_)
      break;

    size_t better_j = j1;

    if (j1 == size_-1) {
      //only one child exists - nothing to do
    }
    else {
      //both children exist

      const size_t j2 = j1 | 1; //j1+1;

      if (less(key_[j2],key_[j1]))
        better_j = j2;
	}
	
    if (less(key_[better_j], cval)) {
      key_[i] = std::move(key_[better_j]);
	  value_[i] = std::move(value_[better_j]);
	  i = better_j;
    }
    else
      break;
  } 
  
  key_[i] = std::move(val);
  value_[i] = std::move(val2);
}

/***********************************/

template<typename T, typename ST, typename Less> 	
void FlexibleHeapPriorityQueueIndexed<T,ST,Less>::insert(const ST idx)
{
  const static Less less;
  
  size_t i = size_;
  size_++;
  
  const T di = data_[idx];

  while (i > 1) {

    const size_t parent = i >> 1; //i / 2;
    if (less(di,data_[heap_[parent]])) {
      heap_[i] = heap_[parent];
	  i = parent;
    }
    else
      break;
  }	
  
  heap_[i] = idx;
}

template<typename T, typename ST, typename Less> 	
void FlexibleHeapPriorityQueueIndexed<T,ST,Less>::extract_min(ST& idx)
{
  const static Less less;
  idx = heap_[1];

  assert(size_ > 1);
  assert(size_ <= heap_.size());
  size_--;	

  size_t i = 1;
  const T value = data_[heap_[size_]];
  const ST cur_idx = heap_[size_];
  while (true) {

    const size_t j1 = i << 1; //2*i;

    if (j1 >= size_)
      break;

    size_t better_j = j1;

    if (j1 == size_-1) {
      //only one child exists - nothing to do
    }
    else {
      //both children exist

      const size_t j2 = j1 | 1; //j1+1;

      if (less(data_[heap_[j2]],data_[heap_[j1]]))
        better_j = j2;
	}
	
    if (less(data_[heap_[better_j]], value)) {
      heap_[i] = heap_[better_j]; 
	  i = better_j;
    }
    else
      break;
  } 
 	
  heap_[i] = cur_idx;
}

#endif