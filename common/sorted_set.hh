/************* written by Thomas Schoenemann as a private person, February 2020 ********/

#ifndef SORTEDSET_HH
#define SORTEDSET_HH

#include "stl_util.hh"
#include "routines.hh"

template<typename T>
class SortedSet {
public:

  SortedSet() {}

  SortedSet(const SortedSet<T>& toCopy);

  void swap(SortedSet<T>& other)
  {
    data_.swap(other.data_);
  }

  size_t size() const
  {
    return data_.size();
  }

  size_t capacity() const
  {
    return data_.capacity();
  }

  void reserve(size_t size)
  {
    data_.reserve(size);
  }

  void clear()
  {
    data_.clear();
  }

  //for compatibility with the other sets, e.g. use in templates (data are always sorted)
  const std::vector<T>& unsorted_data() const
  {
    return data_;
  }

  const std::vector<T>& sorted_data() const
  {
    return data_;
  }

  bool contains(const T val) const;

  //returns true if val is new
  bool insert(const T val);

  void insert_new(const T val);
  
  void insert_largest(const T val);

  //returns true if val was in the tree
  bool erase(const T val);

  //returns true if out was in the tree
  bool replace(const T out, const T in);

protected:

  std::vector<T> data_;
};

/********************** implementation ************************/

template<typename T> 
SortedSet<T>::SortedSet(const SortedSet<T>& toCopy)
  : data_(toCopy.data_) {}

template<typename T>
bool SortedSet<T>::contains(const T val) const
{
  return (binsearch(data_, val) != MAX_UINT);
}

//returns true if val is new
template<typename T>
bool SortedSet<T>::insert(const T val)
{
  //std::cerr << "insert" << std::endl;
  const size_t size = data_.size();
  const size_t inspos = binsearch_insertpos(data_, val);
  if (inspos >= size) {
    data_.push_back(val);
    return true;
  }

  if (data_[inspos] == val)
    return false;

  data_.push_back(T());

  Routines::upshift_array(data_.data(), inspos, size, 1);
  //for (uint k = size; k > inspos; k--)
  //  data_[k] = data_[k-1];

  data_[inspos] = val;
  return true;
}

//returns true if val is new
template<typename T>
void SortedSet<T>::insert_new(const T val)
{
  //std::cerr << "insert" << std::endl;
  const size_t size = data_.size();
  const size_t inspos = binsearch_insertpos(data_, val);
  assert(inspos >= size || data_[inspos] != val);
  if (inspos >= size) {
    data_.push_back(val);
  }

  data_.push_back(T());

  Routines::upshift_array(data_.data(), inspos, size, 1);
  //for (uint k = size; k > inspos; k--)
  //  data_[k] = data_[k-1];

  data_[inspos] = val;
}

template<typename T>
void SortedSet<T>::insert_largest(const T val)
{
  assert(data_.size() == 0 || data_.back() < val);
  data_.push_back(val);
}

//returns true if val was in the tree
template<typename T>
bool SortedSet<T>::erase(const T val)
{
  //std::cerr << "erase " << val << " from " << data_ << std::endl;
  const size_t pos = binsearch(data_, val);
  if (pos == MAX_UINT)
    return false;

  const size_t size = data_.size();
  Routines::downshift_array(data_.data(), pos, 1, size);
  //for (uint k = pos; k < size-1; k++)
  //  data_[k] = data_[k+1];

  data_.resize(size-1);
  return true;
}

//returns true if out was in the tree
template<typename T>
bool SortedSet<T>::replace(const T out, const T in)
{
  assert(!contains(in));

#if 0
  bool b = erase(out);
  insert(in);
  return b;
#else
  const size_t size = data_.size();
  const size_t pos = binsearch(data_, out);
  if (pos < size) {

    if (pos > 0 && in < data_[pos-1]) {
      size_t npos = pos-1;
      while (npos > 0 && in < data_[npos-1])
        npos--;

      for (size_t k = pos; k > npos; k--)
        data_[k] = data_[k-1];
      data_[npos] = in;
    }
    else if (pos+1 < size && data_[pos+1] < in) {
      size_t npos = pos+1;
      while (npos+1 < size && data_[npos+1] < in)
        npos++;

      for (size_t k = pos; k < npos; k++)
        data_[k] = data_[k+1];
      data_[npos] = in;
    }
    else
      data_[pos] = in;

    return true;
  }
  else {

    insert(in);
    return false;
  }
#endif
}

#endif