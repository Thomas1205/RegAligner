/************* written by Thomas Schoenemann as a private person, February 2020 ********/

#ifndef SORTEDSET_HH
#define SORTEDSET_HH

#include "stl_util.hh"
#include "routines.hh"
#include "unsorted_set.hh"

template<typename T, typename Less = std::less<T>, typename Equal = std::equal_to<T> >
class SortedSet : public SetBase<T> {
public:

  using Base = SetBase<T>;
  using PassType = typename std::conditional<std::is_fundamental<T>::value || std::is_pointer<T>::value, const T, const T&>::type;

  SortedSet() {}

  SortedSet(const SortedSet<T,Less,Equal>& toCopy) : SetBase<T>(toCopy) {}

  SortedSet(SortedSet<T,Less,Equal>&& toTake) : SetBase<T>(toTake) {}

  SortedSet(const std::initializer_list<T>& init);

  void operator=(const SortedSet<T,Less,Equal>& toCopy);

  void operator=(SortedSet<T,Less,Equal>&& toTake);

  //for compatibility with the other sets, e.g. use in templates (data are always sorted)
  const std::vector<T>& unsorted_data() const noexcept
  {
    return Base::data_;
  }

  const std::vector<T>& sorted_data() const noexcept
  {
    return Base::data_;
  }

  bool contains(const PassType val) const noexcept;

  //returns true if val is new
  bool insert(const PassType val) noexcept;

  //returns true if val is new
  bool move_insert(T&& val) noexcept;

  void insert_new(const PassType val) noexcept;

  void move_insert_new(T&& val) noexcept;

  void insert_largest(const PassType val) noexcept;

  void move_insert_largest(T&& val) noexcept;

  //returns true if val was in the tree
  bool erase(const PassType val) noexcept;

  //returns true if out was in the tree
  bool replace(const PassType out, const PassType in) noexcept;

  //returns true if out was in the tree
  bool move_replace(const PassType out, T&& in) noexcept;
};

template<typename T, typename Less, typename Equal>
std::ostream& operator<<(std::ostream& os, const SortedSet<T,Less,Equal>& set);

template<typename T, typename Less, typename Equal>
bool operator==(const SortedSet<T,Less,Equal>& set1, const SortedSet<T,Less,Equal>& set2) noexcept;

/********************** implementation ************************/

template<typename T, typename Less, typename Equal> SortedSet<T,Less,Equal>::SortedSet(const std::initializer_list<T>& init)
{
  Base::data_.reserve(init.size());
  for (typename std::initializer_list<T>::const_iterator it = init.begin(); it != init.end(); it++)
    insert(*it);
}

template<typename T, typename Less, typename Equal>
void SortedSet<T,Less,Equal>::operator=(const SortedSet<T,Less,Equal>& toCopy)
{
  Base::data_ = toCopy.data_;
}

template<typename T, typename Less, typename Equal>
void SortedSet<T,Less,Equal>::operator=(SortedSet<T,Less,Equal>&& toTake)
{
  Base::data_.swap(toTake.data_);
}

template<typename T, typename Less, typename Equal>
bool SortedSet<T,Less,Equal>::contains(const PassType val) const noexcept
{
  return (Routines::binsearch<T,Less,Equal>(Base::data_.data(), val, Base::data_.size()) < Base::data_.size());
}

//returns true if val is new
template<typename T, typename Less, typename Equal>
bool SortedSet<T,Less,Equal>::insert(const PassType val) noexcept
{
  //std::cerr << "insert" << std::endl;
  const size_t size = Base::data_.size();
  const size_t inspos = Routines::binsearch_insertpos<T,Less,Equal,size_t>(Base::data_.data(), val, Base::data_.size());
  if (inspos >= size) {
    Base::data_.push_back(val);
    return true;
  }

  const static Equal equal;
  if (equal(Base::data_[inspos],val))
    return false;

  Base::data_.push_back(T());

  Routines::upshift_array(Base::data_.data(), inspos, size, 1);
  //for (uint k = size; k > inspos; k--)
  //  data_[k] = data_[k-1];

  Base::data_[inspos] = val;
  return true;
}

//returns true if val is new
template<typename T, typename Less, typename Equal>
bool SortedSet<T,Less,Equal>::move_insert(T&& val) noexcept
{
  //std::cerr << "insert" << std::endl;
  const size_t size = Base::data_.size();
  const size_t inspos = binsearch_insertpos<T,Less,Equal>(Base::data_, val);
  if (inspos >= size) {
    Base::data_.push_back(val);
    return true;
  }

  const static Equal equal;
  if (equal(Base::data_[inspos],val))
    return false;

  Base::data_.push_back(T());

  Routines::upshift_array(Base::data_.data(), inspos, size, 1);
  //for (uint k = size; k > inspos; k--)
  //  Base::data_[k] = Base::data_[k-1];

  Base::data_[inspos] = val;
  return true;
}

//returns true if val is new
template<typename T, typename Less, typename Equal>
void SortedSet<T,Less,Equal>::insert_new(const PassType val) noexcept
{
  //std::cerr << "insert" << std::endl;
  const size_t size = Base::data_.size();
  const size_t inspos = binsearch_insertpos<T,Less>(Base::data_, val);
  const static Equal equal;
  assert(inspos >= size || !equal(Base::data_[inspos],val));
  if (inspos >= size) {
    Base::data_.push_back(val);
  }

  Base::data_.push_back(T());

  Routines::upshift_array(Base::data_.data(), inspos, size, 1);
  //for (uint k = size; k > inspos; k--)
  //  Base::data_[k] = Base::data_[k-1];

  Base::data_[inspos] = val;
}

//returns true if val is new
template<typename T, typename Less, typename Equal>
void SortedSet<T,Less,Equal>::move_insert_new(T&& val) noexcept
{
  //std::cerr << "insert" << std::endl;
  const size_t size = Base::data_.size();
  const size_t inspos = binsearch_insertpos<T,Less>(Base::data_, val);
  const static Equal equal;
  assert(inspos >= size || !equal(Base::data_[inspos],val));
  if (inspos >= size) {
    Base::data_.push_back(val);
  }

  Base::data_.push_back(T());

  Routines::upshift_array(Base::data_.data(), inspos, size, 1);
  //for (uint k = size; k > inspos; k--)
  //  Base::data_[k] = Base::data_[k-1];

  Base::data_[inspos] = val;
}

template<typename T, typename Less, typename Equal>
void SortedSet<T,Less,Equal>::insert_largest(const PassType val) noexcept
{
  const static Less less;
  assert(Base::data_.size() == 0 || less(Base::data_.back(),val));
  Base::data_.push_back(val);
}

template<typename T, typename Less, typename Equal>
void SortedSet<T,Less,Equal>::move_insert_largest(T&& val) noexcept
{
  const static Less less;

  assert(Base::data_.size() == 0 || less(Base::data_.back(),val));
  Base::data_.push_back(val);
}

//returns true if val was in the tree
template<typename T, typename Less, typename Equal>
bool SortedSet<T,Less,Equal>::erase(const PassType val) noexcept
{
  //std::cerr << "erase " << val << " from " << data_ << std::endl;
  const size_t size = Base::data_.size();
  const size_t pos = binsearch<T,Less,Equal>(Base::data_, val);
  if (pos >= size)
    return false;

  Routines::downshift_array(Base::data_.data(), pos, 1, size);
  //for (uint k = pos; k < size-1; k++)
  //  data_[k] = data_[k+1];

  Base::data_.resize(size-1);
  return true;
}

//returns true if out was in the tree
template<typename T, typename Less, typename Equal>
bool SortedSet<T,Less,Equal>::replace(const PassType out, const PassType in) noexcept
{
  assert(!contains(in));

#if 0
  bool b = erase(out);
  insert(in);
  return b;
#else
  const size_t size = Base::data_.size();
  const size_t pos = binsearch<T,Less,Equal>(Base::data_, out);
  const static Less less;
  if (pos < size) {

    if (pos > 0 && in < Base::data_[pos-1]) {
      size_t npos = pos-1;
      while (npos > 0 && in < Base::data_[npos-1])
        npos--;

      //TODO: call shift function - untested
      //Routines::upshift_array(Base::data_.data(), npos, pos, 1);
      for (size_t k = pos; k > npos; k--)
        Base::data_[k] = Base::data_[k-1];
      Base::data_[npos] = in;
    }
    else if (pos+1 < size && less(Base::data_[pos+1],in)) {
      size_t npos = pos+1;
      while (npos+1 < size && less(Base::data_[npos+1],in))
        npos++;

      //TODO: call shift function -- untested
      //Routines::downshift_array(Base::data_.data(), pos, 1, npos+1);
      for (size_t k = pos; k < npos; k++)
        Base::data_[k] = Base::data_[k+1];
      Base::data_[npos] = in;
    }
    else
      Base::data_[pos] = in;

    return true;
  }
  else {

    insert(in);
    return false;
  }
#endif
}

//returns true if out was in the tree
template<typename T, typename Less, typename Equal>
bool SortedSet<T,Less,Equal>::move_replace(const PassType out, T&& in) noexcept
{
  assert(!contains(in));

#if 0
  bool b = erase(out);
  insert(in);
  return b;
#else
  const size_t size = Base::data_.size();
  const size_t pos = binsearch<T,Less,Equal>(Base::data_, out);
  const static Less less;
  if (pos < size) {

    if (pos > 0 && in < Base::data_[pos-1]) {
      size_t npos = pos-1;
      while (npos > 0 && less(in,Base::data_[npos-1]))
        npos--;

      //TODO: call shift function
      for (size_t k = pos; k > npos; k--)
        Base::data_[k] = Base::data_[k-1];
      Base::data_[npos] = in;
    }
    else if (pos+1 < size && less(Base::data_[pos+1],in)) {
      size_t npos = pos+1;
      while (npos+1 < size && less(Base::data_[npos+1],in))
        npos++;

      //TODO: call shift function
      for (size_t k = pos; k < npos; k++)
        Base::data_[k] = Base::data_[k+1];
      Base::data_[npos] = in;
    }
    else
      Base::data_[pos] = in;

    return true;
  }
  else {

    insert(in);
    return false;
  }
#endif
}

template<typename T, typename Less, typename Equal>
std::ostream& operator<<(std::ostream& os, const SortedSet<T,Less,Equal>& set)
{
  const std::vector<T>& data = set.sorted_data();
  const size_t size = data.size();

  os << "{ ";
  for (size_t i = 0; i < size; i++) {
    if (i != 0)
      os << ", ";
    os << data[i];
  }

  os << " }";
  return os;
}

template<typename T, typename Less, typename Equal>
bool operator==(const SortedSet<T,Less,Equal>& set1, const SortedSet<T,Less,Equal>& set2) noexcept
{
  return (set1.sorted_data() == set2.sorted_data());
}

#endif