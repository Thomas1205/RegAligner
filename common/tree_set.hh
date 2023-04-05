/************* written by Thomas Schoenemann as a private person, December 2019 ********/

#ifndef TREESET_HH
#define TREESET_HH

#include <vector>
#include <iostream>
#include "makros.hh"
#include "stl_out.hh" //DEBUG
#include "storage1D.hh"

template<typename T>
class TreeSet {
public:

  using PassType = typename std::conditional<std::is_fundamental<T>::value || std::is_pointer<T>::value, const T, const T&>::type;

  TreeSet()
  {
    data_.push_back(T()); //so far we do not use the first element
  }

  TreeSet(const TreeSet<T>& s)
  {
    data_ = s.data_;
  }

  TreeSet(TreeSet<T>&& s)
  {
    data_ = s.data_;
  }

  TreeSet(const std::initializer_list<T>& init)
  {
    data_.reserve(init.size()+1);
    data_.push_back(T()); //so far we do not use the first element
    for (typename std::initializer_list<T>::const_iterator it = init.begin(); it != init.end(); it++)
      insert(*it);
  }

  void swap(TreeSet<T>& other);

  size_t size() const;

  size_t capacity() const;

  bool contains(PassType val) const;

  size_t element_num(PassType val) const;

  T min() const;

  T max() const;

  //returns true if val is new
  bool insert(PassType val);

  //returns true if val is new
  bool move_insert(T&& val);

  void insert_new(PassType val);

  void move_insert_new(T&& val);

  //for compatibility with the other sets (use in templates etc.)
  inline void insert_largest(PassType val)
  {
    insert_new(val);
  }

  //for compatibility with the other sets (use in templates etc.)
  inline void move_insert_largest(T&& val)
  {
    move_insert_new(val);
  }

  //returns true if val was in the tree
  bool erase(PassType val);

  //returns true if out was in the tree
  bool replace(PassType out, PassType in);

  void clear();

  void reserve(size_t size);

  //NOTE: element 0 is just a filler
  const std::vector<T>& unsorted_data() const;

  std::vector<T> sorted_data() const;

  void get_sorted_data(Storage1D<T>& target) const;

  std::vector<T> get_sorted_head(const size_t nElements) const;

  //if val is found: all data >= val, otherwise empty vector
  std::vector<T> get_sorted_data_from_val(PassType val) const;

  //future work may include:
  // - reverse sorted retrieval
  // - get nth element

protected:

  //correction for the case that the leaf i is the only wrong node in the tree (if there is any)
  void correct_leaf(size_t i);

  std::vector<T> data_;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const TreeSet<T>& set);

template<typename T>
bool operator==(const TreeSet<T>& set1, const TreeSet<T>& set2);

/************** implementation ***********/

template<typename T>
size_t TreeSet<T>::size() const
{
  return data_.size() - 1;
}

template<typename T>
size_t TreeSet<T>::capacity() const
{
  return data_.capacity() - 1;
}

template<typename T>
void TreeSet<T>::swap(TreeSet<T>& other)
{
  data_.swap(other.data_);
}

template<typename T>
void TreeSet<T>::clear()
{
  data_.resize(1);
}

template<typename T>
void TreeSet<T>::reserve(size_t size)
{
  data_.reserve(size+1);
}

template<typename T>
bool TreeSet<T>::contains(PassType val) const
{
  const size_t size = data_.size();
  size_t i = 1;
  while (true) {

    if (i >= size)
      return false;
    if (val == data_[i])
      return true;
    if (val < data_[i])
      i <<= 1; //i *= 2;
    else {
      i = (i << 1) | 1; //2*i+1;
    }
  }
}

template<typename T>
T TreeSet<T>::min() const
{
  const size_t size = data_.size();
  if (size == 1)
    return data_[0];

  uint i = 1;
  while (true) {
    //if (2*i >= size)
    if (i << 1 >= size)
      break;
    i <<= 1; //i *= 2;
  }

  return data_[i];
}

template<typename T>
T TreeSet<T>::max() const
{
  const size_t size = data_.size();
  if (size == 1)
    return data_[0];

  uint i = 1;
  while (true) {
    //if (2*i+1 >= size)
    if (((i << 1) | 1) >= size)
      break;
    i = (i << 1) | 1; //2*i+1;
  }

  return data_[i];
}

template<typename T>
size_t TreeSet<T>::element_num(PassType val) const
{
  //std::cerr << "element_num(" << val << ")" << std::endl;
  //std::cerr << "data: " << data_ << std::endl;

  const size_t size = data_.size();
  if (size == 1)
    return MAX_UINT;

  const size_t highestFullLevel =  Makros::highest_bit(size)-1; //note that n full levels have 2^n-1 elements

  size_t i = 1;
  size_t num = 0;
  size_t level = 0;
  while (true) {

    if (i >= size)
      return MAX_UINT;
    if (val < data_[i]) {
      //if (2*i >= size)
      if ((i << 1) >= size)
        return MAX_UINT;

      i <<= 1; //i *= 2; //go to left
      level++;
    }
    else {

      //entire left tree is first in order
      //calc number of elements in the left tree
#if 0
      uint ii = 2*i; //this is the root of the left tree
      uint base = 1;
      while (true) {
        if (ii >= size)
          break;
        if (ii+base >= size) {
          num += size-ii;
          break;
        }
        num += base;
        ii *= 2;
        base *= 2;
      }
#else
      //left tree has root 2*i at level level+1
      //if (2*i < size) {
      if ((i << 1) < size) {
        //left tree exists
        //the levels level+1 to highestFullLevel are full
        //if (4*i < size) {
        if ((i << 2) < size) {
          const size_t exponent = (highestFullLevel - level); //number of fully filled levels
          num += (1 << exponent) - 1; //number of elements in the fully filled levels
          const size_t base = i * (1 << (exponent+1));
          if (base < size) {
            //const size_t next_base = (2*i+1) * (1 << exponent);
            const size_t next_base = ((i << 1) + 1) * (1 << exponent);
            num += std::min(size,next_base)-base;
          }
        }
        else
          num += 1; //only one leaf to the left
      }
#endif

      if (val == data_[i])
        return num;

      //if (2*i+1 >= size)
      if (((i << 1) | 1) >= size)
        return MAX_UINT;

      //right exists
      num++; //for the current node

      i = (i << 1) | 1; // 2*i+1; // go to right
      level++;
    }
  }

  //should never be reached
  return num;
}

template<typename T>
void TreeSet<T>::correct_leaf(size_t i)
{
  //std::cerr << "correct_leaf(" << i << "), size: " << data_.size() << std::endl;
  //std::cerr << "data: " << data_ << std::endl;

  const size_t size = data_.size();

  assert(i > 1); //not the trivial case with only one node in the tree
#ifdef SAFE_MODE
  if ( !(i < data_.size() && 2*i >= data_.size()) )
    print_trace();
#endif
  assert(i < data_.size() && 2*i >= data_.size()); //really a leaf
  const T val = data_[i];

  //find highest violated node
  bool from_right = false;
  bool highest_from_right = false;
  size_t highest = i;
  size_t j = i;
  while (j > 1) {

    from_right = ((j & 1) != 0); //((j % 2) == 1); // != 0 is reflected in the zero flag -> better than == 1
    j >>= 1; // j /= 2;
    if ((from_right && val < data_[j]) || (!from_right && data_[j] < val)) {
      highest = j;
      highest_from_right = from_right;
    }
  }

  //std::cerr << "highest: " << highest << std::endl;

  if (highest != i) {
    //correction necessary
    std::swap(data_[i], data_[highest]);
    correct_leaf(i); // this will not reach highest again (if i was the only wrong node in the tree)

    if (highest_from_right) {
      //need to check the left side of the tree: is val bigger than the largest in the left?
      // we know that highest has at least a left child

      size_t k = highest << 1; //2*highest; //left once
      //std::cerr << "start k: " << k << ", size: " << size << ", highest: " << highest << std::endl;
      //while (2*k < size) {
      while ((k << 1) < size) {
        //k = std::min(size-1,2*k+1); //right always, except when only the left leaf exists
        k = std::min(size-1,(k << 1) | 1); //right always, except when only the left leaf exists
      }

      assert(k != highest);
      assert(k < data_.size() && 2*k >= data_.size()); //k must be leaf

      //if (k % 2 == 0 && k != 2*highest) {
      if ((k & 1) == 0 && k != (highest << 1)) {
        //ended in left where right should be, need additional swap
        //if (data_[highest] < data_[k/2]) {
        if (data_[highest] < data_[k >> 1]) {
          //std::swap(data_[k],data_[k/2]);
          std::swap(data_[k],data_[k >> 1]);
          std::swap(data_[highest], data_[k]);
          correct_leaf(k);
        }
      }
      else if (data_[highest] < data_[k]) {
        std::swap(data_[highest], data_[k]);
        correct_leaf(k);
      }
    }
    else {
      //need to check the right side of the tree: is val smaller than the smallest in the right?

      //if (2*highest+1 < size) { // right children exist
      if (((highest << 1) | 1) < size) { // right children exist
        size_t k = (highest << 1) | 1; // 2*highest+1; //right once
        //while (2*k < size) {
        while ((k << 1) < size) {
          k <<= 1; //k *= 2; //left always
        }
        assert(k != highest);
        assert(k < data_.size() && 2*k >= data_.size()); //k must be leaf

        if (data_[k] < data_[highest]) {
          std::swap(data_[highest], data_[k]);
          correct_leaf(k);
        }
      }
    }
  }
}

//returns true if val is new
template<typename T>
bool TreeSet<T>::insert(PassType val)
{
  if (contains(val))
    return false;

  data_.push_back(val);
  if (data_.size() > 2)
    correct_leaf(data_.size()-1);

  return true;
}

//returns true if val is new
template<typename T>
bool TreeSet<T>::move_insert(T&& val)
{
  if (contains(val))
    return false;

  data_.push_back(val);
  if (data_.size() > 2)
    correct_leaf(data_.size()-1);

  return true;
}

template<typename T>
void TreeSet<T>::insert_new(PassType val)
{
  assert(!contains(val));

  data_.push_back(val);
  if (data_.size() > 2)
    correct_leaf(data_.size()-1);
}

template<typename T>
void TreeSet<T>::move_insert_new(T&& val)
{
  assert(!contains(val));

  data_.push_back(val);
  if (data_.size() > 2)
    correct_leaf(data_.size()-1);
}

//returns true if val was in the tree
template<typename T>
bool TreeSet<T>::erase(PassType val)
{
  //std::cerr << "erase (" << val << ")" << std::endl;
  //std::cerr << "data: " << data_ << std::endl;

  //first a find
  const size_t size = data_.size();
  size_t i = 1;
  while (true) {

    if (i >= size)
      return false;
    if (val == data_[i])
      break;
    if (val < data_[i])
      i <<= 1; //i *= 2;
    else {
      i = (i << 1) | 1; //2*i+1;
    }
  }

  //std::cerr << "found at " << i << ", size: " << size << std::endl;

  //the node can be in the interior of the tree!
  if (i == size-1) {
    data_.resize(size-1);
  }
  //else if (2*i >= size) { //leaf
  else if ((i << 1) >= size) { //leaf
    std::swap(data_[i], data_[size-1]);
    data_.resize(size-1);
    correct_leaf(i);
  }
  else {
    //node is interior
    //if (2*i == size-1) {
    if ((i << 1) == size-1) {
      // node has only left child -> replace
      //std::swap(data_[i],data_[2*i]);
      std::swap(data_[i],data_[i << 1]);
      data_.resize(size-1);
    }
    else {
      size_t j = i << 1; //2*i;
      if (val < data_[size-1]) {
        //std::cerr << "find right" << std::endl;
        //find smallest in right tree
        j = (i << 1) | 1; //2*i+1; //right once
        //while (2*j < size) {
        while ((j << 1) < size) {
          j <<= 1; //j *= 2; //left constantly
        }
      }
      else {
        //std::cerr << "find left" << std::endl;
        //find largest in left tree
        j = i << 1; //2*i; //left once
        //while (2*j < size) {
        while ((j << 1) < size) {
          //j = std::min(size-1,2*j+1); //right always, except when only the left leaf exists
          j = std::min(size-1,(j << 1) | 1); //right always, except when only the left leaf exists
        }
        //if ((j % 2) == 0 && j != 2*i) //ended in left where right should be, need additional swap
        if ((j & 1) == 0 && j != (i << 1)) { //ended in left where right should be, need additional swap
          //std::swap(data_[j],data_[j/2]);
          std::swap(data_[j],data_[j >> 1]);
        }
      }
      //std::cerr << "found leaf " << j << std::endl;

      std::swap(data_[i], data_[j]); //now val is at leaf j
      std::swap(data_[j], data_[size-1]);
      data_.resize(size-1);
      if (j < size-1)
        correct_leaf(j);
    }
  }

  //assert(std::find(data_.begin()+1, data_.end(), val) == data_.end() );
  return true;
}

template<typename T>
bool TreeSet<T>::replace(PassType out, PassType in)
{
  assert(!contains(in));
  assert(out != in);

  //first a find
  const size_t size = data_.size();
  if (size == 2 && data_[1] == out) {
    data_[1] = in;
    return true;
  }

  size_t i = 1;
  while (true) {

    if (i >= size)
      break;
    if (out == data_[i])
      break;
    if (out < data_[i])
      i <<= 1; //i *= 2;
    else {
      //i = 2*i+1;
      i = (i << 1) + 1;
    }
  }

  if (i >= size) {
    insert(in);
    return false;
  }
  else {
    //if (2*i >= size) {
    if ((i << 1) >= size) {
      //hit a leaf, easy
      assert(i < data_.size() && 2*i >= data_.size());
      data_[i] = in;
      correct_leaf(i);
    }
    else {
      size_t j = i;
      //if (in > out && 2*i+1 < size) {
      if (in > out && ((i << 1) | 1) < size) {
        //take lowest from right
        j = (i << 1) | 1; // 2*i+1; //right once
        //while (2*j < size) {
        while ((j << 1) < size) {
          j <<= 1; //j *= 2; //left always
        }
      }
      else {
        //take highest from left
        j <<= 1; //j = 2*i; //left once
        //while (2*j < size) {
        while ((j << 1) < size) {
          //j = std::min(size-1,2*j+1); //right always, except when only the left leaf exists
          j = std::min(size-1,(j << 1) | 1); //right always, except when only the left leaf exists
        }
        //if ((j % 2) == 0 && j != 2*i) //ended in left where right should be, need additional swap
        if ((j & 1) == 0 && j != (i << 1)) { //ended in left where right should be, need additional swap
          //std::swap(data_[j],data_[j/2]);
          std::swap(data_[j],data_[j >> 1]);
        }
      }
      assert(j < data_.size() && 2*j >= data_.size());
      data_[i] = data_[j];
      data_[j] = in;
      correct_leaf(j);
    }
    return true;
  }
}

template<typename T>
const std::vector<T>& TreeSet<T>::unsorted_data() const
{
  return data_;
}

template<typename T>
std::vector<T> TreeSet<T>::sorted_data() const
{
  const size_t size = data_.size();
  std::vector<T> result;
  result.reserve(size-1);

  size_t i = 1;
  bool descending = true;
  bool from_right = true;
  while (i > 0) {

    //if (2*i >= size) {
    if ((i << 1) >= size) {
      //leaf
      result.push_back(data_[i]);
      descending = false;
      from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
      i >>= 1; //i /= 2;
    }
    else if (descending) {
      i <<= 1; //i *= 2; //go to left child
    }
    else {
      if (from_right) {
        from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
        i >>= 1; //i /= 2;
      }
      else {
        result.push_back(data_[i]);
        //if (i*2+1 < size) {
        if (((i << 1) | 1) < size) {
          //go to right child
          descending = true;
          i = (i << 1) | 1; // i*2+1;
        }
        else {
          //there is no right child -> up
          from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
          i >>= 1; //i /= 2;
        }
      }
    }
  }

  return result;
}

template <typename T>
void TreeSet<T>::get_sorted_data(Storage1D<T>& result) const
{
  const size_t size = data_.size();
  result.resize_dirty(size-1);

  size_t k = 0;
  size_t i = 1;
  bool descending = true;
  bool from_right = true;
  while (i > 0) {

    //if (2*i >= size) {
    if ((i << 1) >= size) {
      //leaf
      result[k] = data_[i];
      k++;
      descending = false;
      from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
      i >>= 1; //i /= 2;
    }
    else if (descending) {
      i <<= 1; //i *= 2; //go to left child
    }
    else {
      if (from_right) {
        from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
        i >>= 1; //i /= 2;
      }
      else {
        result[k] = data_[i];
        k++;
        //if (i*2+1 < size) {
        if (((i << 1) | 1) < size) {
          //go to right child
          descending = true;
          i = (i << 1) | 1; // i*2+1;
        }
        else {
          //there is no right child -> up
          //from_right = ((i%2) == 1);
          from_right = ((i & 1) != 0); // != 0 is reflected in the zero flag -> better than == 1
          i >>= 1; //i /= 2;
        }
      }
    }
  }
}

template <typename T>
std::vector<T> TreeSet<T>::get_sorted_head(size_t nElements) const
{
  const size_t size = data_.size();
  std::vector<T> result;
  result.reserve(size-1);

  size_t i = 1;
  bool descending = true;
  bool from_right = true;
  while (i > 0 && result.size() < nElements) {

    //if (2*i >= size) {
    if ((i << 1) >= size) {
      //leaf
      result.push_back(data_[i]);
      descending = false;
      from_right = ((i & 1) == 1); //((i%2) == 1);
      i >>= 1; //i /= 2;
    }
    else if (descending) {
      i <<= 1; //i *= 2; //go to left child
    }
    else {
      if (from_right) {
        from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
        i >>= 1; //i /= 2;
      }
      else {
        result.push_back(data_[i]);
        //if (i*2+1 < size) {
        if (((i << 1) | 1) < size) {
          //go to right child
          descending = true;
          i = (i << 1) | 1; //i*2+1;
        }
        else {
          //there is no right child -> up
          from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
          i >>= 1; //i /= 2;
        }
      }
    }
  }

  return result;
}

template <typename T>
std::vector<T> TreeSet<T>::get_sorted_data_from_val(PassType val) const
{
  const size_t size = data_.size();
  std::vector<T> result;

  if (size == 1)
    return result;

  size_t i = 1;
  while (i < size) {

    if (val == data_[i])
      break;
    if (val < data_[i])
      i <<= 1; //i *= 2;
    else
      i = (i << 1) | 1; //2*i+1;
  }

  if (i >= size)
    i >>= 1; //i /= 2; //back to the leaf

  result.reserve(size-1);
  bool descending = false; //start with false because we do not want to visit the left tree
  bool from_right = false; //just back from left (quasi)

  if (data_[i] < val) {
    //the leaf is not in the sequence. for simplicity, compare everywhere (by design already the next data should be > val)

    while (i > 0) {

      //if (2*i >= size) {
      if ((i << 1) >= size) {
        //leaf
        descending = false;
        from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
        i >>= 1; //i /= 2;
        break;
      }
      else if (descending) {
        i <<= 1; //i *= 2; //go to left child
      }
      else {
        if (from_right) {
          from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
          i >>= 1; //i /= 2;
        }
        else {
          //if (i*2+1 < size) {
          if (((i << 1) | 1) < size) {
            //go to right child
            descending = true;
            i = (i << 1) | 1; //i*2+1;
          }
          else {
            //there is no right child -> up
            from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
            i >>= 1; //i /= 2;
          }
          break;
        }
      }
    }
  }

  while (i > 0) {

    //if (2*i >= size) {
    if ((i << 1) >= size) {
      //leaf
      result.push_back(data_[i]);
      descending = false;
      from_right = ((i & 1) != 0); //((i%2) == 1);  // != 0 is reflected in the zero flag -> better than == 1
      i /= 2;
    }
    else if (descending) {
      i <<= 1; //i *= 2; //go to left child
    }
    else {
      if (from_right) {
        from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
        i >>= 1; //i /= 2;
      }
      else {
        result.push_back(data_[i]);
        //if (i*2+1 < size) {
        if (((i << 1) | 1) < size) {
          //go to right child
          descending = true;
          i = (i << 1) | 1; //i*2+1;
        }
        else {
          //there is no right child -> up
          from_right = ((i & 1) != 0); //((i%2) == 1); // != 0 is reflected in the zero flag -> better than == 1
          i >>= 1; //i /= 2;
        }
      }
    }
  }

  return result;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const TreeSet<T>& treeset)
{
  const std::vector<T>& data = treeset.unsorted_data();
  const size_t size = data.size();

  os << "{ ";
  size_t i = 1;
  bool descending = true;
  bool from_right = true;
  bool first = true;
  while (i > 0) {

    if (2*i >= size) {
      //leaf
      if (!first)
        os << ", ";
      os << data[i];
      first = false;
      descending = false;
      from_right = ((i & 1) == 1); //((i%2) == 1);
      i >>= 1; //i /= 2;
    }
    else if (descending) {
      i <<= 1; //i *= 2; //go to left child
    }
    else {
      if (from_right) {
        from_right = ((i & 1) == 1); //((i%2) == 1);
        i >>= 1; //i /= 2;
      }
      else {
        if (!first)
          os << ", ";
        os << data[i];
        if (i*2+1 < size) {
          //go to right child
          descending = true;
          first = false;
          i = i*2+1;
        }
        else {
          //there is no right child -> up
          from_right = ((i & 1) == 1); //((i%2) == 1);
          i >>= 1; //i /= 2;
        }
      }
    }
  }
  os << " }";

  return os;
}

template<typename T>
bool operator==(const TreeSet<T>& set1, const TreeSet<T>& set2)
{
  return (set1.unsorted_data() == set2.unsorted_data());
}

#endif