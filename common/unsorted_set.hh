/************** written by Thomas Schoenemann as a private person, February 2020 ***********/

#ifndef UNSORTED_SET_HH
#define UNSORTED_SET_HH

#include "sorting.hh"
#include "routines.hh"

#ifndef USET_SORT_ALG
#define USET_SORT_ALG bubble_sort
#endif

//find in a sequence without duplicates
template<typename T, typename Equal = std::equal_to<T> >
inline typename std::vector<T>::const_iterator set_find(const std::vector<T>& vec, const T val) noexcept;

template<>
inline std::vector<uint>::const_iterator set_find(const std::vector<uint>& vec, const uint val) noexcept;

template<typename T, typename Equal = std::equal_to<T> >
inline bool set_contains(const std::vector<T>& vec, const T val) noexcept;

template<>
inline bool set_contains(const std::vector<uint>& vec, const uint val) noexcept;

template<typename T>
class SetBase {
public:

  SetBase() {}

  SetBase(const SetBase<T>& toCopy)
  {
    data_ = toCopy.data_;
  }

  SetBase(SetBase<T>&& toTake)
  {
    data_ = std::move(toTake.data_);
  }

  size_t size() const noexcept
  {
    return data_.size();
  }

  size_t capacity() const noexcept
  {
    return data_.capacity();
  }

  void reserve(size_t size) noexcept
  {
    data_.reserve(size);
  }

  void clear() noexcept
  {
    data_.clear();
  }

protected:

  std::vector<T> data_;
};

template<typename T, typename Equal = std::equal_to<T> >
class UnsortedSet : public SetBase<T> {
public:

  using Base = SetBase<T>;
  using PassType = typename std::conditional<std::is_fundamental<T>::value || std::is_pointer<T>::value, const T, const T&>::type;

  UnsortedSet() {}

  UnsortedSet(const UnsortedSet<T,Equal>& toCopy) : SetBase<T>(toCopy) {}

  UnsortedSet(UnsortedSet<T,Equal>&& toTake) : SetBase<T>(toTake) {}

  UnsortedSet(const std::initializer_list<T>& init);

  void operator=(const UnsortedSet<T,Equal>& toCopy);

  void operator=(UnsortedSet<T,Equal>&& toTake);

  void swap(UnsortedSet<T,Equal>& other) noexcept
  {
    Base::data_.swap(other.data_);
  }

  const std::vector<T>& unsorted_data() const noexcept
  {
    return Base::data_;
  }

  //const qualifier doesn't make sense here - not returning a reference
  const std::vector<T> sorted_data() const noexcept
  {
    std::vector<T> result = Base::data_;
    USET_SORT_ALG(result.data(), result.size());
    assert(is_unique_sorted(result.data(), result.size()));

    return result;
  }

  void get_sorted_data(Storage1D<T>& target) const noexcept
  {
    assign(target, Base::data_);
    USET_SORT_ALG(target.direct_access(), target.size());
  }

  bool contains(const PassType val) const noexcept;

  //returns true if val is new
  bool insert(PassType val) noexcept;

  //returns true if val is new
  bool move_insert(T&& val) noexcept;

  void insert_new(const PassType val) noexcept;

  void move_insert_new(T&& val) noexcept;

  //for compatibility with the other sets (use in templates etc.)
  inline void insert_largest(const PassType val) noexcept
  {
    insert_new(val);
  }

  //for compatibility with the other sets (use in templates etc.)
  inline void move_insert_largest(T&& val) noexcept
  {
    move_insert_new(val);
  }

  //returns true if val was in the tree
  bool erase(const PassType val) noexcept;

  //returns true if out was in the tree
  bool replace(const PassType out, PassType in) noexcept;

  //returns true if out was in the tree
  bool move_replace(const PassType out, T&& in) noexcept;
};

template<typename T, typename Equal>
std::ostream& operator<<(std::ostream& os, const UnsortedSet<T,Equal>& set);

/********************/

template<typename T, typename Less = std::less<T>, typename Equal=std::equal_to<T> >
class UnsortedSetExploitSort : public SetBase<T>  {
public:

  using Base = SetBase<T>;
  using PassType = typename std::conditional<std::is_fundamental<T>::value || std::is_pointer<T>::value, const T, const T&>::type;

  UnsortedSetExploitSort() {}

  UnsortedSetExploitSort(const UnsortedSetExploitSort<T,Less,Equal>& toCopy) : SetBase<T>(toCopy), is_sorted_(toCopy.is_sorted_) {}

  UnsortedSetExploitSort(UnsortedSetExploitSort<T,Less,Equal>&& toTake) : SetBase<T>(toTake), is_sorted_(toTake.is_sorted_) {}

  UnsortedSetExploitSort(const std::initializer_list<T>& init);

  void operator=(const UnsortedSet<T,Equal>& toCopy);

  void operator=(UnsortedSet<T,Equal>&& toTake);

  void swap(UnsortedSetExploitSort<T,Less,Equal>& other) noexcept
  {
    Base::data_.swap(other.data_);
    std::swap(is_sorted_, other.is_sorted_);
  }

  void clear() noexcept
  {
    Base::data_.clear();
    is_sorted_ = true;
  }

  const std::vector<T>& unsorted_data() const noexcept
  {
    return Base::data_;
  }

  const std::vector<T>& sorted_data() noexcept
  {
    if (!is_sorted_) {
      USET_SORT_ALG(Base::data_.data(), Base::data_.size());
      //std::cerr << "result: " << result << std::endl;
      assert(is_unique_sorted(Base::data_.data(), Base::data_.size()));
      is_sorted_ = true;
    }

    return Base::data_;
  }

  void get_sorted_data(Storage1D<T>& target) const noexcept
  {
    assign(target, Base::data_);
    if (!is_sorted_)
      USET_SORT_ALG(target.direct_access(), target.size());
  }

  bool contains(const PassType val) const noexcept;

  //returns true if val is new
  bool insert(const PassType val) noexcept;

  //returns true if val is new
  bool move_insert(T&& val) noexcept;

  void insert_new(const PassType val) noexcept;

  void move_insert_new(T&& val) noexcept;

  //for compatibility with the other sets (use in templates etc.)
  inline void insert_largest(const PassType val) noexcept;

  //for compatibility with the other sets (use in templates etc.)
  inline void move_insert_largest(T&& val) noexcept;

  //returns true if val was in the tree
  bool erase(const PassType val) noexcept;

  //returns true if out was in the tree
  bool replace(const PassType out, const PassType in) noexcept;

  //returns true if out was in the tree
  bool move_replace(const PassType out, T&& in) noexcept;

protected:

  bool is_sorted_ = true;
};

template<typename T, typename Less, typename Equal>
std::ostream& operator<<(std::ostream& os, const UnsortedSetExploitSort<T,Less,Equal>& set);

/********************** implementation of helpers ************************/

//find in a sequence without duplicates
template<typename T, typename Equal = std::equal_to<T> >
inline typename std::vector<T>::const_iterator set_find(const std::vector<T>& vec, const T val) noexcept
{
  if (std::is_trivially_copyable<T>::value) {
    const uint pos = Routines::find_unique(vec.data(), val, vec.size());
    if (pos >= vec.size())
      return vec.end();
    else
      return vec.begin() + pos;
  }
  else
    return std::find<T,Equal>(vec.begin(), vec.end(), val);
}

template<>
inline std::vector<uint>::const_iterator set_find(const std::vector<uint>& vec, const uint val) noexcept
{
  const uint pos = Routines::find_unique_uint(vec.data(), val, vec.size());
  if (pos >= vec.size())
    return vec.end();
  else
    return (vec.begin() + pos);
}

template<typename T, typename Equal>
inline bool set_contains(const std::vector<T>& vec, const T val) noexcept
{
  const static Equal equal;
  if (std::is_trivially_copyable<T>::value)
    return Routines::contains(vec.data(), val);
  else
    return (std::find<T,Equal>(vec.begin(), vec.end(), val) != vec.end());
}

template<>
inline bool set_contains(const std::vector<uint>& vec, const uint val) noexcept
{
  return Routines::contains_uint(vec.data(), val, vec.size());
}

/********************** implementation of UnsortedSet ************************/

template<typename T, typename Equal> UnsortedSet<T,Equal>::UnsortedSet(const std::initializer_list<T>& init)
{
  Base::data_.reserve(init.size());
  for (typename std::initializer_list<T>::const_iterator it = init.begin(); it != init.end(); it++)
    insert(*it);
}

template<typename T, typename Equal>
void UnsortedSet<T,Equal>::operator=(const UnsortedSet<T,Equal>& toCopy)
{
  Base::data_ = toCopy.data_;
}

template<typename T, typename Equal>
void UnsortedSet<T,Equal>::operator=(UnsortedSet<T,Equal>&& toTake)
{
  Base::data_.swap(toTake.data_);
}

template<typename T, typename Equal>
bool UnsortedSet<T,Equal>::contains(const PassType val) const noexcept
{
  return set_contains(Base::data_, val);
}

//returns true if val is new
template<typename T, typename Equal>
bool UnsortedSet<T,Equal>::insert(const PassType val) noexcept
{
  if (set_contains<T,Equal>(Base::data_, val))
    return false;

  Base::data_.push_back(val);
  return true;
}

//returns true if val is new
template<typename T, typename Equal>
bool UnsortedSet<T,Equal>::move_insert(T&& val) noexcept
{
  if (set_contains<T,Equal>(Base::data_, val))
    return false;

  Base::data_.push_back(val);
  return true;
}

template<typename T, typename Equal>
void UnsortedSet<T,Equal>::insert_new(const PassType val) noexcept
{
  assert(!contains(val));
  Base::data_.push_back(val);
}

template<typename T, typename Equal>
void UnsortedSet<T,Equal>::move_insert_new(T&& val) noexcept
{
  assert(!contains(val));
  Base::data_.push_back(val);
}

//returns true if val was in the tree
template<typename T, typename Equal>
bool UnsortedSet<T,Equal>::erase(const PassType val) noexcept
{
  const typename std::vector<T>::const_iterator it = set_find<T,Equal>(Base::data_, val);
  if (it == Base::data_.end())
    return false;

  size_t pos = it - Base::data_.begin();
  Base::data_[pos] = Base::data_.back();
  Base::data_.resize(Base::data_.size()-1);
  return true;
}

//returns true if out was in the tree
template<typename T, typename Equal>
bool UnsortedSet<T,Equal>::replace(const PassType out, const PassType in) noexcept
{
  assert(!contains(in));

  typename std::vector<T>::const_iterator it = set_find<T,Equal>(Base::data_, out);
  if (it == Base::data_.end()) {
    Base::data_.push_back(in);
    return false;
  }

  Base::data_[it - Base::data_.begin()] = in;
  return true;
}

//returns true if out was in the tree
template<typename T, typename Equal>
bool UnsortedSet<T,Equal>::move_replace(const PassType out, T&& in) noexcept
{
  assert(!contains(in));

  typename std::vector<T>::const_iterator it = set_find<T,Equal>(Base::data_, out);
  if (it == Base::data_.end()) {
    Base::data_.push_back(in);
    return false;
  }

  Base::data_[it - Base::data_.begin()] = in;
  return true;
}

template<typename T, typename Equal>
std::ostream& operator<<(std::ostream& os, const UnsortedSet<T,Equal>& set)
{
  const std::vector<T> data = set.sorted_data();
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

/********************** implementation of UnsortedSetExploitSort ************************/

template<typename T, typename Less, typename Equal> UnsortedSetExploitSort<T,Less,Equal>::UnsortedSetExploitSort(const std::initializer_list<T>& init)
{
  Base::data_.reserve(init.size());
  for (typename std::initializer_list<T>::const_iterator it = init.begin(); it != init.end(); it++)
    insert(*it);
}
template<typename T, typename Less, typename Equal>
void UnsortedSetExploitSort<T,Less,Equal>::operator=(const UnsortedSet<T,Equal>& toCopy)
{
  Base::data_ = toCopy.data_;
}

template<typename T, typename Less, typename Equal>
void UnsortedSetExploitSort<T,Less,Equal>::operator=(UnsortedSet<T,Equal>&& toTake)
{
  Base::data_.swap(toTake.data_);
}

template<typename T, typename Less, typename Equal>
bool UnsortedSetExploitSort<T,Less,Equal>::contains(const PassType val) const noexcept
{
  if (!is_sorted_) {
    //return (set_find(data_, val) != data_.end());
    return set_contains<T,Equal>(Base::data_, val);
  }
  else
    return (binsearch<T,Less,Equal>(Base::data_, val) != MAX_UINT);
}

//returns true if val is new
template<typename T, typename Less, typename Equal>
bool UnsortedSetExploitSort<T,Less,Equal>::insert(const PassType val) noexcept
{
  const size_t size = Base::data_.size();
  const static Less less;
  bool is_new = false;
  if (!is_sorted_) {
    //is_new = (set_find(data_, val) != data_.end());
    is_new = !set_contains(Base::data_, val);
  }
  else
    is_new = (binsearch<T,Less,Equal>(Base::data_, val) != MAX_UINT);

  if (!is_new) {
    if (is_sorted_ && size > 0 && less(val,Base::data_.back()))
      is_sorted_ = false;
    Base::data_.push_back(val);
  }
  return is_new;
}

//returns true if val is new
template<typename T, typename Less, typename Equal>
bool UnsortedSetExploitSort<T,Less,Equal>::move_insert(T&& val) noexcept
{
  const size_t size = Base::data_.size();
  const static Less less;
  bool is_new = false;
  if (!is_sorted_) {
    is_new = !set_contains(Base::data_, val);
  }
  else
    is_new = (binsearch<T,Less,Equal>(Base::data_, val) != MAX_UINT);

  if (!is_new) {
    if (is_sorted_ && size > 0 && less(val,Base::data_.back()))
      is_sorted_ = false;
    Base::data_.push_back(val);
  }
  return is_new;
}

template<typename T, typename Less, typename Equal>
void UnsortedSetExploitSort<T,Less,Equal>::insert_new(const PassType val) noexcept
{
  const static Less less;
  assert(!contains(val));
  if (is_sorted_) {
    if (Base::data_.size() > 0 && less(val,Base::data_.back()))
      is_sorted_ = false;
  }
  Base::data_.push_back(val);
}

template<typename T, typename Less, typename Equal>
void UnsortedSetExploitSort<T,Less,Equal>::move_insert_new(T&& val) noexcept
{
  const static Less less;
  assert(!contains(val));
  if (is_sorted_) {
    if (Base::data_.size() > 0 && less(val,Base::data_.back()))
      is_sorted_ = false;
  }
  Base::data_.push_back(val);
}

//for compatibility with the other sets (use in templates etc.)
template<typename T, typename Less, typename Equal>
inline void UnsortedSetExploitSort<T,Less,Equal>::insert_largest(const PassType val) noexcept
{
  assert(!contains(val));
  Base::data_.push_back(val);
}

//for compatibility with the other sets (use in templates etc.)
template<typename T, typename Less, typename Equal>
inline void UnsortedSetExploitSort<T,Less,Equal>::move_insert_largest(T&& val) noexcept
{
  assert(!contains(val));
  Base::data_.push_back(val);
}

//returns true if val was in the tree
template<typename T, typename Less, typename Equal>
bool UnsortedSetExploitSort<T,Less,Equal>::erase(const PassType val) noexcept
{
  const size_t size = Base::data_.size();
  size_t pos = 0;
  if (!is_sorted_) {
    const typename std::vector<T>::const_iterator it = set_find(Base::data_, val);
    if (it == Base::data_.end())
      return false;
    pos = it - Base::data_.begin();
  }
  else {
    pos = binsearch<T,Less,Equal>(Base::data_, val, Base::data_);
    if (pos == MAX_UINT)
      return false;
  }

  if (pos != size - 1) {
    Base::data_[pos] = Base::data_.back();
    is_sorted_ = false;
  }
  if (size <= 2)
    is_sorted_ = true;
  Base::data_.resize(size-1);
  return true;
}

//returns true if out was in the tree
template<typename T, typename Less, typename Equal>
bool UnsortedSetExploitSort<T,Less,Equal>::replace(const PassType out, const PassType in) noexcept
{
  assert(!contains(in));
  const static Less less;
  const size_t size = Base::data_.size();
  size_t pos = 0;
  if (!is_sorted_) {
    const typename std::vector<T>::const_iterator it = set_find(Base::data_, out);
    if (it == Base::data_.end())
      pos = MAX_UINT;
    else
      pos = it - Base::data_.begin();
  }
  else {
    pos = binsearch<T,Less,Equal>(Base::data_, out);
  }

  if (pos == MAX_UINT) {
    if (is_sorted_ && size > 0 && less(in,Base::data_.back()))
      is_sorted_ = false;
    Base::data_.push_back(in);
    return false;
  }

  is_sorted_ = false;
  Base::data_[pos] = in;
  return true;
}

//returns true if out was in the tree
template<typename T, typename Less, typename Equal>
bool UnsortedSetExploitSort<T,Less,Equal>::move_replace(const PassType out, T&& in) noexcept
{
  assert(!contains(in));
  const static Less less;
  const size_t size = Base::data_.size();
  size_t pos = 0;
  if (!is_sorted_) {
    const typename std::vector<T>::const_iterator it = set_find(Base::data_, out);
    if (it == Base::data_.end())
      pos = MAX_UINT;
    else
      pos = it - Base::data_.begin();
  }
  else {
    pos = binsearch<T,Less,Equal>(Base::data_, out);
  }

  if (pos == MAX_UINT) {
    if (is_sorted_ && size > 0 && less(in,Base::data_.back()))
      is_sorted_ = false;
    Base::data_.push_back(in);
    return false;
  }

  is_sorted_ = false;
  Base::data_[pos] = in;
  return true;
}

template<typename T, typename Less, typename Equal>
std::ostream& operator<<(std::ostream& os, const UnsortedSetExploitSort<T,Less,Equal>& set)
{
  std::vector<T> data;
  get_sorted_data(data);
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

#endif