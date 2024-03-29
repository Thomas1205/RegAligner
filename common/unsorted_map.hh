/**************** written by Thomas Schoenemann as a private person, June 2020 ********************/

#ifndef UNSORTED_MAP_HH
#define UNSORTED_MAP_HH

#include <vector>
#include "routines.hh"
#include "flexible_storage1D.hh"
#include "hash_map.hh"

//so far operator< is fixed

template<typename Key, typename Value, typename KVec = std::vector<Key>, typename VVec = std::vector<Value> >
class MapBase {
public:

  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;

  MapBase() {};

  MapBase(const MapBase<Key, Value>& toCopy) : key_(toCopy.key_), value_(toCopy.value_) {}

  MapBase(MapBase<Key, Value>&& toTake) : key_(std::move(toTake.key_)), value_(std::move(toTake.value_)) {}

  ~MapBase() {}

  size_t size() const noexcept
  {
    return key_.size();
  }

  void reserve(size_t reservedSize)
  {
    key_.reserve(reservedSize);
    value_.reserve(reservedSize);
  }

  void swap(MapBase<Key, Value>& toSwap) noexcept
  {
    key_.swap_(toSwap.key_);
    value_.swap(toSwap.value_);
  }

  //NOTE: keys may be unsorted
  const KVec& key() const noexcept
  {
    return key_;
  }

  //NOTE: values correspond to keys, which may be unsorted
  const VVec& value() const noexcept
  {
    return value_;
  }
  
  //NOTE: values correspond to keys, which may be unsorted
  VVec& value() noexcept
  {
    return value_;
  }  

  void clear() noexcept
  {
    key_.clear();
    value_.clear();
  }

protected:

  KVec key_;
  VVec value_;
};

template<typename Key, typename Value, typename KVec = std::vector<Key>, typename VVec = std::vector<Value> >
std::ostream& operator<<(std::ostream& s, const MapBase<Key,Value,KVec,VVec>& m)
{
  const KVec& key = m.key();
  const VVec& value = m.value();
  s << "[ ";
  for (size_t i = 0; i < key.size(); i++) {
    std::cerr << key[i] << "->" << value[i];
    if (i+1 < key.size())
      std::cerr << ", ";
  }
  s << " ]";
  return s;
}

//use this if your keys are small and easily comparable
//so far this map does not offer erasing
template<typename Key, typename Value, typename KVec = std::vector<Key>, typename VVec = std::vector<Value>,
         typename Equal = std::equal_to<Key> >
class UnsortedMap : public MapBase<Key,Value,KVec,VVec> {
public:

  using Base = MapBase<Key,Value,KVec,VVec>;
  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;

  UnsortedMap() {};

  UnsortedMap(const UnsortedMap<Key, Value, KVec, VVec, Equal>& toCopy) : MapBase<Key,Value,KVec,VVec>(toCopy) {}

  UnsortedMap(UnsortedMap<Key, Value, KVec, VVec, Equal>&& toTake) : MapBase<Key,Value,KVec,VVec>(toTake) {}

  ~UnsortedMap() {}

  size_t keypos(KeyPassType key) const noexcept;

  bool contains(const KeyPassType key) const noexcept;

  Value& operator[](const KeyPassType key); //vector can throw std::bad_alloc()
  
  Value operator()(const KeyPassType key, const Value backoff_value) const noexcept;

  Value& find_or_insert(const KeyPassType key, const Value& initial_value); //vector can throw std::bad_alloc()

  Value& find_or_insert(Key&& key, const Value& initial_value); //vector can throw std::bad_alloc()

  void operator=(const UnsortedMap<Key, Value, KVec, VVec, Equal>& toCopy); //vector can throw std::bad_alloc()

  void operator=(UnsortedMap<Key, Value, KVec, VVec, Equal>&& toTake); //vector can throw std::bad_alloc()

  template<typename Hash, typename HT, typename ST>
  void assign(const HashMapBase<Key,Value,Hash,HT,ST>& toCopy);

  template<typename Hash, typename HT, typename ST>
  void move_assign(HashMapBase<Key,Value,Hash,HT,ST>&& toTake);
};

template<typename Key, typename Value, typename KVec = std::vector<Key>, typename VVec = std::vector<Value>,
         typename Less = std::less<Key>, typename Equal = std::equal_to<Key> >
class UnsortedMapExploitSort : public MapBase<Key,Value,KVec,VVec> {
public:

  using Base = MapBase<Key,Value,KVec,VVec>;
  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;

  UnsortedMapExploitSort() {};

  UnsortedMapExploitSort(const UnsortedMapExploitSort<Key, Value, KVec, VVec, Less, Equal>& toCopy) : MapBase<Key,Value,KVec,VVec>(toCopy) {}

  UnsortedMapExploitSort(UnsortedMapExploitSort<Key, Value, KVec, VVec, Less, Equal>&& toTake) : MapBase<Key,Value,KVec,VVec>(toTake) {}

  ~UnsortedMapExploitSort() {}

  void swap(UnsortedMapExploitSort<Key, Value, KVec, VVec, Less, Equal>& toSwap) noexcept
  {
    Base::key_.swap_(toSwap.key_);
    Base::value_.swap(toSwap.value_);
    std::swap(is_sorted_,toSwap.is_sorted_);
  }

  size_t keypos(KeyPassType key) const noexcept;

  void clear() noexcept
  {
    Base::key_.clear();
    Base::value_.clear();
    is_sorted_ = true;
  }

  bool contains(const KeyPassType key) const noexcept;

  Value& operator[](const KeyPassType key); //std::vector can throw std::bad_alloc
  
  Value operator()(const KeyPassType key, const Value backoff_value) const noexcept;

  Value& find_or_insert(const KeyPassType key, const Value& initial_value); //vector can throw std::bad_alloc()

  Value& find_or_insert(Key&& key, const Value& initial_value); //vector can throw std::bad_alloc()

  void operator=(const UnsortedMap<Key, Value, KVec, VVec, Equal>& toCopy);

  void operator=(UnsortedMap<Key, Value, KVec, VVec, Equal>&& toTake);

  void operator=(const UnsortedMapExploitSort<Key, Value, KVec, VVec, Less, Equal>& toCopy);

  void operator=(UnsortedMapExploitSort<Key, Value, KVec, VVec, Less, Equal>&& toTake);

  template<typename Hash, typename HT, typename ST>
  void assign(const HashMapBase<Key,Value,Hash,HT,ST>& toCopy);

  template<typename Hash, typename HT, typename ST>
  void move_assign(HashMapBase<Key,Value,Hash,HT,ST>&& toTake);

protected:

  bool is_sorted_ = true;
};

/****************************** implementation *******************************/

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
size_t UnsortedMap<Key,Value,KVec,VVec,Equal>::keypos(KeyPassType key) const noexcept
{
  return Routines::find_unique(Base::key_.data(), key, Base::key_.size());
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
bool UnsortedMap<Key,Value,KVec,VVec,Equal>::contains(const KeyPassType key) const noexcept
{
  return Routines::contains(Base::key_.data(), key, Base::key_.size());
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
Value& UnsortedMap<Key,Value,KVec,VVec,Equal>::operator[](const KeyPassType key)
{
  const size_t size = Base::key_.size();
  const size_t pos = Routines::find_unique<Key,Equal>(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];

  Base::key_.push_back(key);
  Base::value_.push_back(Value());
  return Base::value_.back();
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
Value UnsortedMap<Key,Value,KVec,VVec,Equal>::operator()(const KeyPassType key, const Value backoff_value) const noexcept
{
  const size_t size = Base::key_.size();
  const size_t pos = Routines::find_unique<Key,Equal>(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];

  return backoff_value;	
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
Value& UnsortedMap<Key,Value,KVec,VVec,Equal>::find_or_insert(const KeyPassType key, const Value& initial_value)
{
  const size_t size = Base::key_.size();
  const size_t pos = Routines::find_unique(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];

  Base::key_.push_back(key);
  Base::value_.push_back(initial_value);
  return Base::value_.back();
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
Value& UnsortedMap<Key,Value,KVec,VVec,Equal>::find_or_insert(Key&& key, const Value& initial_value)
{
  const size_t size = Base::key_.size();
  const size_t pos = Routines::find_unique(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];

  Base::key_.push_back(key);
  Base::value_.push_back(initial_value);
  return Base::value_.back();
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
void UnsortedMap<Key,Value,KVec,VVec,Equal>::operator=(const UnsortedMap<Key, Value, KVec, VVec, Equal>& toCopy)
{
  Base::key_ = toCopy.key_;
  Base::value_ = toCopy.value_;
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
void UnsortedMap<Key,Value,KVec,VVec,Equal>::operator=(UnsortedMap<Key, Value, KVec, VVec, Equal>&& toTake)
{
  Base::key_.swap(toTake.key_);
  Base::value_.swap(toTake.value_);
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
template<typename Hash, typename HT, typename ST>
void UnsortedMap<Key,Value,KVec,VVec,Equal>::assign(const HashMapBase<Key,Value,Hash,HT,ST>& toCopy)
{
  const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& key_stack = toCopy.key_stack();
  const FlexibleStorage1D<FlexibleStorage1D<Value,ST>,ST>& value_stack = toCopy.value_stack();

  size_t size = 0;
  for (size_t k=0; k < key_stack.size(); k++)
    size += key_stack[k].size();

  Base::key_.clear();
  Base::key_.reserve(size);
  Base::value_.clear();
  Base::value_.reserve(size);

  for (size_t k=0; k < key_stack.size(); k++) {
    const FlexibleStorage1D<Key,ST>& cur_keys = key_stack[k];
    const FlexibleStorage1D<Value,ST>& cur_values = value_stack[k];
    for (size_t kk = 0; kk < cur_keys.size(); kk++) {
      Base::key_.push_back(cur_keys[kk]);
      Base::value_.push_back(cur_values[kk]);
    }
  }
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Equal>
template<typename Hash, typename HT, typename ST>
void UnsortedMap<Key,Value,KVec,VVec,Equal>::move_assign(HashMapBase<Key,Value,Hash,HT,ST>&& toTake)
{
  const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& key_stack = toTake.key_stack();
  const FlexibleStorage1D<FlexibleStorage1D<Value,ST>,ST>& value_stack = toTake.value_stack();

  size_t size = 0;
  for (size_t k=0; k < key_stack.size(); k++)
    size += key_stack[k].size();

  Base::key_.clear();
  Base::key_.reserve(size);
  Base::value_.clear();
  Base::value_.reserve(size);

  for (size_t k=0; k < key_stack.size(); k++) {
    const FlexibleStorage1D<Key,ST>& cur_keys = key_stack[k];
    const FlexibleStorage1D<Value,ST>& cur_values = value_stack[k];
    for (size_t kk = 0; kk < cur_keys.size(); kk++) {
      Base::key_.push_back(std::move(cur_keys[kk]));
      Base::value_.push_back(std::move(cur_values[kk]));
    }
  }
}

/*********************/

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
size_t UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::keypos(KeyPassType key) const noexcept
{
  if (is_sorted_)
    return Routines::binsearch<Key,Less,Equal>(Base::key_.data(), key, Base::key_.size());
  else
    return Routines::find_unique<Key,Equal>(Base::key_.data(), key, Base::key_.size());
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
bool UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::contains(const KeyPassType key) const noexcept
{
  const size_t size = Base::key_.size();
  //std::cerr << "*********UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::contains" << std::endl;
  //std::cerr << "is sorted: " << is_sorted_ << std::endl;
  if (is_sorted_)
    return (Routines::binsearch(Base::key_.data(), key, size) < size);
  else
    return Routines::contains(Base::key_.data(), key, size);
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value& UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::operator[](const KeyPassType key) 
{
  const size_t size = Base::key_.size();
  const size_t pos = (is_sorted_) ? Routines::binsearch<Key,Less,Equal>(Base::key_.data(), key, size)
                     : Routines::find_unique<Key,Equal>(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];

  const static Less less;
  if (size > 0 && less(key,Base::key_.back()))
    is_sorted_ = false;

  Base::key_.push_back(key);
  Base::value_.push_back(Value());
  return Base::value_.back();
}


template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::operator()(const KeyPassType key, const Value backoff_value) const noexcept
{
  const size_t size = Base::key_.size();
  const size_t pos = (is_sorted_) ? Routines::binsearch<Key,Less,Equal>(Base::key_.data(), key, size)
                     : Routines::find_unique<Key,Equal>(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];
  return backoff_value;	
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value& UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::find_or_insert(const KeyPassType key, const Value& initial_value)
{
  const size_t size = Base::key_.size();
  const size_t pos = (is_sorted_) ? Routines::binsearch<Key,Less,Equal>(Base::key_.data(), key, size)
                     : Routines::find_unique<Key,Equal>(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];

  const static Less less;
  if (size > 0 && less(key,Base::key_.back()))
    is_sorted_ = false;

  Base::key_.push_back(key);
  Base::value_.push_back(initial_value);
  return Base::value_.back();
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value& UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::find_or_insert(Key&& key, const Value& initial_value)
{
  const size_t size = Base::key_.size();
  const size_t pos = (is_sorted_) ? Routines::binsearch<Key,Less,Equal>(Base::key_.data(), key, size)
                     : Routines::find_unique<Key,Equal>(Base::key_.data(), key, size);
  if (pos < size)
    return Base::value_[pos];

  const static Less less;
  if (size > 0 && less(key,Base::key_.back()))
    is_sorted_ = false;

  Base::key_.push_back(key);
  Base::value_.push_back(initial_value);
  return Base::value_.back();
}


template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
void UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::operator=(const UnsortedMap<Key, Value, KVec, VVec, Equal>& toCopy)
{
  Base::key_ = toCopy.key_;
  Base::value_ = toCopy.value_;
  is_sorted_ = false;
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
void UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::operator=(UnsortedMap<Key, Value, KVec, VVec, Equal>&& toTake)
{
  Base::key_.swap(toTake.key_);
  Base::value_.swap(toTake.value_);
  is_sorted_ = false;
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
void UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::operator=(const UnsortedMapExploitSort<Key, Value, KVec, VVec, Less, Equal>& toCopy)
{
  Base::key_ = toCopy.key_;
  Base::value_ = toCopy.value_;
  is_sorted_ = toCopy.is_sorted_;
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
void UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::operator=(UnsortedMapExploitSort<Key, Value, KVec, VVec, Less, Equal>&& toTake)
{
  Base::key_.swap(toTake.key_);
  Base::value_.swap(toTake.value_);
  is_sorted_ = toTake.is_sorted_;
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
template<typename Hash, typename HT, typename ST>
void UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::assign(const HashMapBase<Key,Value,Hash,HT,ST>& toCopy)
{
  const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& key_stack = toCopy.key_stack();
  const FlexibleStorage1D<FlexibleStorage1D<Value,ST>,ST>& value_stack = toCopy.value_stack();

  size_t size = 0;
  for (size_t k=0; k < key_stack.size(); k++)
    size += key_stack[k].size();

  Base::key_.clear();
  Base::key_.reserve(size);
  Base::value_.clear();
  Base::value_.reserve(size);

  for (size_t k=0; k < key_stack.size(); k++) {
    const FlexibleStorage1D<Key,ST>& cur_keys = key_stack[k];
    const FlexibleStorage1D<Value,ST>& cur_values = value_stack[k];
    for (size_t kk = 0; kk < cur_keys.size(); kk++) {
      Base::key_.push_back(cur_keys[kk]);
      Base::value_.push_back(cur_values[kk]);
    }
  }
  is_sorted_ = is_sorted<Key,Less>(Base::key_.data(), size);
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
template<typename Hash, typename HT, typename ST>
void UnsortedMapExploitSort<Key,Value,KVec,VVec,Less,Equal>::move_assign(HashMapBase<Key,Value,Hash,HT,ST>&& toTake)
{
  const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& key_stack = toTake.key_stack();
  const FlexibleStorage1D<FlexibleStorage1D<Value,ST>,ST>& value_stack = toTake.value_stack();

  size_t size = 0;
  for (size_t k=0; k < key_stack.size(); k++)
    size += key_stack[k].size();

  Base::key_.clear();
  Base::key_.reserve(size);
  Base::value_.clear();
  Base::value_.reserve(size);

  for (size_t k=0; k < key_stack.size(); k++) {
    const FlexibleStorage1D<Key,ST>& cur_keys = key_stack[k];
    const FlexibleStorage1D<Value,ST>& cur_values = value_stack[k];
    for (size_t kk = 0; kk < cur_keys.size(); kk++) {
      Base::key_.push_back(std::move(cur_keys[kk]));
      Base::value_.push_back(std::move(cur_values[kk]));
    }
  }
  is_sorted_ = is_sorted<Key,Less>(Base::key_.data(), size);
}

#endif