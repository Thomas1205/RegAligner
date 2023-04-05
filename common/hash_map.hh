/********************** written by Thomas Schoenemann as a private person, June 2020 *************************/

#ifndef HASHMAP_HH
#define HASHMAP_HH

#include "flexible_storage1D.hh"
#include "routines.hh"

template<typename Key, typename Value, typename HashFunc, typename HT = size_t, typename ST = uint>
class HashMapBase {
public:

  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;

  HashMapBase() {}

  HashMapBase(const HashMapBase& toCopy) : hash_value_(toCopy.hash_value_), key_(toCopy.key_), value_(toCopy.value_) {}

  HashMapBase(HashMapBase&& toTake) : hash_value_(std::move(toTake.hash_value_)), key_(std::move(toTake.key_)), value_(std::move(toTake.value_)) {}

  ~HashMapBase() {}

  void swap(HashMapBase& toSwap)
  {
    hash_value_.swap(toSwap.hash_value_);
    key_.swap(toSwap.key_);
    value_.swap(toSwap.value_);
  }

  struct iterator;
  struct const_iterator;

  bool empty() const noexcept;

  ST size() const noexcept;

  ST bucket_count() const noexcept;
  
  void reserve(size_t size) {
	hash_value_.reserve(size);
	key_.reserve(size);
	value_.reserve(size);
  }

  const FlexibleStorage1D<FlexibleStorage1D<Key,ST>, ST>& key_stack() const
  {
    return key_;
  }

  const FlexibleStorage1D<FlexibleStorage1D<Value,ST>, ST>& value_stack() const
  {
    return value_;
  }

  const FlexibleStorage1D<HT,ST>& hash_values() const
  {
    return hash_value_;
  }

  iterator begin() noexcept;

  iterator end() noexcept;

  const_iterator cbegin() const noexcept;

  const_iterator cend() const noexcept;

  void clear()
  {
    hash_value_.clear(true);
    key_.clear(true);
    value_.clear(true);
  }

  struct iterator {

    iterator(HashMapBase<Key,Value,HashFunc,HT,ST>& map) : map_(map) {}

    iterator(HashMapBase<Key,Value,HashFunc,HT,ST>& map, size_t pos, size_t sub_pos) : map_(map), pos_(pos), sub_pos_(sub_pos) {}

    inline iterator& operator++() noexcept
    {

      const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& map_key = map_.key_;

      assert(pos_ < map_key.size());
      if (sub_pos_ + 1 < map_key[pos_].size())
        sub_pos_++;
      else {
        pos_++;
        while (pos_ < map_key.size() && map_key[pos_].size() == 0)
          pos_++;
        sub_pos_ = 0;
      }

      return *this;
    }

    iterator operator++(int) noexcept
    {
      iterator res = *this;
      operator++();
      return res;
    }

    bool operator!=(const HashMapBase<Key,Value,HashFunc,HT,ST>::iterator& i2) const noexcept
    {
      return (pos_ != i2.pos_ || sub_pos_ != i2.sub_pos_);
    }

    //no ideas for this yet
    //std::pair<KeyPassType, Value&> operator*();
    //std::pair<KeyPassType, Value&> operator->();

    const Key& key() const noexcept
    {
      return map_.key_[pos_][sub_pos_];
    }

    Value& value()
    {
      return map_.value_[pos_][sub_pos_];
    }

    size_t pos_ = 0;
    size_t sub_pos_ = 0;
    HashMapBase<Key,Value,HashFunc,HT,ST>& map_;
  };

  struct const_iterator {

    const_iterator(const HashMapBase<Key,Value,HashFunc,HT,ST>& map) : map_(map) {}

    const_iterator(const HashMapBase<Key,Value,HashFunc,HT,ST>& map, size_t pos, size_t sub_pos) : map_(map), pos_(pos), sub_pos_(sub_pos) {}

    bool operator!=(const HashMapBase<Key,Value,HashFunc,HT,ST>::const_iterator& i2) const noexcept
    {
      return (pos_ != i2.pos_ || sub_pos_ != i2.sub_pos_);
    }

    bool operator!=(const HashMapBase<Key,Value,HashFunc,HT,ST>::iterator& i2) const noexcept
    {
      return (pos_ != i2.pos_ || sub_pos_ != i2.sub_pos_);
    }

    inline const_iterator& operator++() noexcept
    {

      const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& map_key = map_.key_;

      assert(pos_ < map_key.size());
      if (sub_pos_ + 1 < map_key[pos_].size())
        sub_pos_++;
      else {
        pos_++;
        while (pos_ < map_key.size() && map_key[pos_].size() == 0)
          pos_++;

        sub_pos_ = 0;
      }

      return *this;
    }

    const_iterator operator++(int) noexcept
    {
      const_iterator res = *this;
      operator++();
      return res;
    }

    const Key& key() const noexcept
    {
      return map_.key_[pos_][sub_pos_];
    }

    const Value& value() const noexcept
    {
      return map_.value_[pos_][sub_pos_];
    }

    size_t pos_ = 0;
    size_t sub_pos_ = 0;
    const HashMapBase<Key,Value,HashFunc,HT,ST>& map_;
  };

  inline Value& inner_op(KeyPassType key, size_t pos, const Value initial_value = Value())
  {

    FlexibleStorage1D<Key,ST>& cur_key_list = key_[pos];
    FlexibleStorage1D<Value,ST>& cur_value_list = value_[pos];
    const size_t size = cur_key_list.size();
    const size_t sub_pos = Routines::find_unique(cur_key_list.direct_access(), key, size);
    if (sub_pos < size)
      return cur_value_list[sub_pos];
    else {
      cur_key_list.push_back(key);
      cur_value_list.append(initial_value);
      return cur_value_list.back();
    }
  }

  inline Value& sorted_inner_op(KeyPassType key, size_t pos, const Value initial_value = Value())
  {
    FlexibleStorage1D<Key,ST>& cur_keys = key_[pos];
    FlexibleStorage1D<Value,ST>& cur_values = value_[pos];
    const uint cur_size = cur_keys.size();
    ST sub_pos = Routines::binsearch_insertpos(cur_keys.direct_access(), key, cur_size);
    if (sub_pos < cur_size && cur_keys[sub_pos] == key)
      return cur_values[sub_pos];

    return inner_insert(pos,sub_pos,key,initial_value);
  }

  inline void outer_insert(size_t pos, HT hash_value) //insert an empty bin
  {
    //std::cerr << "outer_insert, pos: " << pos << ", size: " << key_.size() << ", value to insert: " << hash_value << std::endl;
    //std::cerr << "hash_value_: " << hash_value_ << std::endl;
    const ST size = key_.size();
    if (pos >= size) {
      hash_value_.push_back(hash_value);
      key_.push_back(FlexibleStorage1D<Key,ST>());
      value_.push_back(FlexibleStorage1D<Value,ST>());
      return;
    }

    hash_value_.push_back(0);
    key_.push_back(FlexibleStorage1D<Key,ST>());
    value_.push_back(FlexibleStorage1D<Value,ST>());

    Routines::upshift_array(hash_value_.data(), pos, size, 1);
    Routines::upshift_array(key_.data(), pos, size, 1);
    Routines::upshift_array(value_.data(), pos, size, 1);

    hash_value_[pos] = hash_value;
    key_[pos] = FlexibleStorage1D<Key,ST>();
    value_[pos] = FlexibleStorage1D<Value,ST>();

    //std::cerr << "new hash_value_: " << hash_value_ << std::endl;
  }

  inline Value& inner_insert(ST pos, ST sub_pos, const KeyPassType key, const Value initial_value = Value())
  {
    FlexibleStorage1D<Key,ST>& cur_keys = key_[pos];
    FlexibleStorage1D<Value,ST>& cur_values = value_[pos];
    const ST cur_size = cur_keys.size();
    if (sub_pos >= cur_size) {
      cur_keys.append(key);
      cur_values.append(initial_value);
      return cur_values.back();
    }

    cur_keys.push_back(Key());
    cur_values.push_back(Value());
    Routines::upshift_array(cur_keys.data(), sub_pos, cur_size, 1);
    Routines::upshift_array(cur_values.data(), sub_pos, cur_size, 1);

    cur_keys[sub_pos] = key;
    cur_values[sub_pos] = initial_value;
    return cur_values[sub_pos];
  }

protected:

  FlexibleStorage1D<HT,ST> hash_value_;
  FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST> key_;
  FlexibleStorage1D<FlexibleStorage1D<Value,ST>,ST> value_;

  static HashFunc hash_;
};

template<typename TK, typename TV, typename H, typename HT, typename ST>
std::ostream& operator<<(std::ostream& os, const HashMapBase<TK,TV,H,HT,ST>& m);


//so far this map does not offer erasing
template<typename Key, typename Value, typename HashFunc, typename HT = size_t, typename ST = uint>
class UnsortedHashMap : public HashMapBase<Key, Value, HashFunc, HT, ST> {
public:

  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;
  using Base = HashMapBase<Key, Value, HashFunc, HT, ST>;

  UnsortedHashMap() : HashMapBase<Key,Value,HashFunc,HT,ST>() {}

  UnsortedHashMap(const UnsortedHashMap<Key,Value,HashFunc,HT,ST>& toCopy) : HashMapBase<Key,Value,HashFunc>(toCopy) {}

  UnsortedHashMap(UnsortedHashMap<Key,Value,HashFunc,HT,ST>&& toTake) : HashMapBase<Key,Value,HashFunc>(toTake) {}

  ~UnsortedHashMap() {}

  Value& operator[](KeyPassType key);

  //Value& operator[](Key&& key);

  //bool try_get(KeyPassType key, Value*& ptr) const;

  bool contains(KeyPassType key);

  Value& find_or_insert(const KeyPassType key, const Value& initial_value, ST hint_pos = 0);

  ST find_bin(const HT hash_value) const;

  ST find_or_insert_bin(const HT hash_value);
};

//so far this map does not offer erasing
template<typename Key, typename Value, typename HashFunc, typename HT = size_t, typename ST = uint>
class UnsortedHashMapExploitSort : public HashMapBase<Key, Value, HashFunc, HT, ST> {
public:

  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;
  using Base = HashMapBase<Key, Value, HashFunc, HT, ST>;

  UnsortedHashMapExploitSort() : HashMapBase<Key,Value,HashFunc,HT,ST>() {}

  UnsortedHashMapExploitSort(const UnsortedHashMapExploitSort<Key,Value,HashFunc,HT,ST>& toCopy) : HashMapBase<Key,Value,HashFunc>(toCopy) {}

  UnsortedHashMapExploitSort(UnsortedHashMapExploitSort<Key,Value,HashFunc,HT,ST>&& toTake) : HashMapBase<Key,Value,HashFunc>(toTake) {}

  ~UnsortedHashMapExploitSort() {}

  Value& operator[](KeyPassType key);

  //Value& operator[](Key&& key);

  //bool try_get(KeyPassType key, Value*& ptr) const;

  bool contains(KeyPassType key);

  void swap(UnsortedHashMapExploitSort& toSwap)
  {
    Base::swap(toSwap);
    std::swap(is_sorted_, toSwap.is_sorted_);
  }

  void clear()
  {
    Base::clear();
    is_sorted_ = true;
  }

  Value& find_or_insert(const KeyPassType key, const Value& initial_value);

protected:

  bool is_sorted_ = true;
};

//so far this map does not offer erasing
template<typename Key, typename Value, typename HashFunc, typename HT = size_t, typename ST = uint, bool inner_sort = false>
class SortedHashMap : public HashMapBase<Key, Value, HashFunc, HT, ST> {
public:

  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;
  using Base = HashMapBase<Key, Value, HashFunc, HT, ST>;

  SortedHashMap() : HashMapBase<Key,Value,HashFunc,HT,ST>() {}

  SortedHashMap(const SortedHashMap<Key,Value,HashFunc,HT,ST>& toCopy) : HashMapBase<Key,Value,HashFunc,HT>(toCopy) {}

  SortedHashMap(SortedHashMap<Key,Value,HashFunc,HT,ST>&& toTake) : HashMapBase<Key,Value,HashFunc,HT>(toTake) {}

  ~SortedHashMap() {}

  Value& operator[](KeyPassType key);

  //Value& operator[](Key&& key);

  //bool try_get(KeyPassType key, Value*& ptr) const;

  bool contains(KeyPassType key);

  Value& find_or_insert(const KeyPassType key, const Value& initial_value, ST hint_bin = 0);

  ST find_bin(const HT hash_value) const;

  ST find_or_insert_bin(const HT hash_value);
};

/******************** implementation **************************/

template<typename TK, typename TV, typename H, typename HT, typename ST>
std::ostream& operator<<(std::ostream& os, const HashMapBase<TK,TV,H,HT,ST>& m)
{
  os << "[ ";
#if 0
  for (typename HashMapBase<TK,TV,H,HT,ST>::const_iterator it = m.cbegin(); it != m.cend(); ) {
    os << it.key() << "->" << it.value();
    ++it;
    if (it != m.cend())
      os << ", ";
  }
#else
  const FlexibleStorage1D<FlexibleStorage1D<TK,ST>,ST>& key_stack = m.key_stack();
  const FlexibleStorage1D<FlexibleStorage1D<TV,ST>,ST>& value_stack = m.value_stack();
  assert(key_stack.size() == value_stack.size());

  for (uint bin=0; bin < key_stack.size(); bin++) {
    std::cerr << "<";
    assert(key_stack[bin].size() == value_stack[bin].size());
    for (uint k = 0; k < key_stack[bin].size(); k++) {
      std::cerr << key_stack[bin][k] << "->" << value_stack[bin][k];
    }
    std::cerr << ">";
  }
#endif
  os << " ]";

  return os;
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
bool HashMapBase<Key,Value,HashFunc,HT,ST>::empty() const noexcept
{
  return (key_.size() == 0); //CAUTION: does not consider that bins can be empty
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
ST HashMapBase<Key,Value,HashFunc,HT,ST>::size() const noexcept
{
  ST size = 0;
  for (ST i=0; i < key_.size(); i++)
    size += key_[i].size();

  return size;
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
ST HashMapBase<Key,Value,HashFunc,HT,ST>::bucket_count() const noexcept
{
  return key_.size();
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
typename HashMapBase<Key,Value,HashFunc,HT,ST>::iterator HashMapBase<Key,Value,HashFunc,HT,ST>::begin() noexcept
{
  return iterator(*this);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
typename HashMapBase<Key,Value,HashFunc,HT,ST>::iterator HashMapBase<Key,Value,HashFunc,HT,ST>::end() noexcept
{
  if (key_.size() == 0)
    return iterator(*this);
  else
    return iterator(*this,key_.size(), 0);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
typename HashMapBase<Key,Value,HashFunc,HT,ST>::const_iterator HashMapBase<Key,Value,HashFunc,HT,ST>::cbegin() const noexcept
{
  //std::cerr << "key: " << key_ << std::endl;
  return const_iterator(*this);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
typename HashMapBase<Key,Value,HashFunc,HT,ST>::const_iterator HashMapBase<Key,Value,HashFunc,HT,ST>::cend() const noexcept
{
  if (key_.size() == 0)
    return const_iterator(*this);
  else
    return const_iterator(*this,key_.size(), 0);
}

/*************/

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
Value& UnsortedHashMap<Key,Value,HashFunc,HT,ST>::operator[](KeyPassType key)
{
  const ST size = Base::hash_value_.size();
  const HT cur_hash = Base::hash_(key);
  const ST pos = Routines::find_unique(Base::hash_value_.direct_access(), cur_hash, size);

  if (pos >= size) {
    Base::hash_value_.append(cur_hash);
    Base::key_.increase().append(key);
    return Base::value_.increase().increase();
  }

  return Base::inner_op(key, pos);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
bool UnsortedHashMap<Key,Value,HashFunc,HT,ST>::contains(KeyPassType key)
{
  const ST size = Base::hash_value_.size();
  const HT cur_hash = Base::hash_(key);
  const ST pos = Routines::find_unique(Base::hash_value_.direct_access(), cur_hash, size);

  if (pos >= size)
    return false;

  const FlexibleStorage1D<Key,ST>& cur_key_list = Base::key_[pos];
  const ST sub_pos = Routines::find_unique(cur_key_list.direct_access(), key, cur_key_list.size());
  return (sub_pos < cur_key_list.size());
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
Value& UnsortedHashMap<Key,Value,HashFunc,HT,ST>::find_or_insert(const KeyPassType key, const Value& initial_value, ST hint_bin)
{
  //std::cerr << "**********UnsortedHashMap<Key,Value,HashFunc,HT>::find_or_insert*****************" << std::endl;
  const ST size = Base::hash_value_.size();
  const HT cur_hash = Base::hash_(key);
  //std::cerr << "determining pos " << std::endl;
  const ST pos = (hint_bin < size && Base::hash_value_[hint_bin] == cur_hash) ? hint_bin
                 : Routines::find_unique(Base::hash_value_.direct_access(), cur_hash, size);

  //std::cerr << "pos: " << pos << std::endl;
  if (pos >= size) {
    Base::hash_value_.append(cur_hash);
    Base::key_.increase().append(key);
    Base::value_.increase().append(initial_value);
    return Base::value_.back()[0];
  }

  FlexibleStorage1D<Key,ST>& cur_key_list = Base::key_[pos];
  const ST cur_size = cur_key_list.size();
  //std::cerr << "calling find_unique" << std::endl;
  ST sub_pos = Routines::find_unique(cur_key_list.direct_access(), key, cur_size);
  //std::cerr << "sub_pos: " << sub_pos << std::endl;
  if (sub_pos >= cur_size) {
    cur_key_list.push_back(key);
    Base::value_[pos].push_back(initial_value);
    sub_pos = cur_size;
  }
  return Base::value_[pos][sub_pos];
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
ST UnsortedHashMap<Key,Value,HashFunc,HT,ST>::find_bin(const HT hash_value) const
{
  return Routines::find_unique(Base::hash_value_.direct_access(), hash_value, Base::hash_value_.size());
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
ST UnsortedHashMap<Key,Value,HashFunc,HT,ST>::find_or_insert_bin(const HT hash_value)
{
  const ST size = Base::hash_value_.size();
  ST pos = Routines::find_unique(Base::hash_value_.direct_access(), hash_value, size);
  if (pos >= size) {
    for (uint k = 0; k < size; k++) {
      if (Base::key_[k].size() == 0) {
        Base::hash_value_[k] = hash_value;
        return k;
      }
    }

    Base::hash_value_.push_back(hash_value);
    Base::key_.push_back(FlexibleStorage1D<Key,ST>());
    Base::value_.push_back(FlexibleStorage1D<Value,ST>());
    return size;
  }
  return pos;
}


/*********************/

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
Value& UnsortedHashMapExploitSort<Key,Value,HashFunc,HT,ST>::operator[](KeyPassType key)
{
  const ST size = Base::hash_value_.size();
  const HT cur_hash = Base::hash_(key);
  const ST pos = (is_sorted_) ?
                 Routines::binsearch(Base::hash_value_.direct_access(), cur_hash, size) :
                 Routines::find_unique(Base::hash_value_.direct_access(), cur_hash, size);

  if (pos >= size) {
    if (size > 0 && cur_hash < Base::hash_value_.back())
      is_sorted_ = false;

    Base::hash_value_.append(cur_hash);
    Base::key_.increase().append(key);
    return Base::value_.increase().increase();
  }

  return Base::inner_op(key, pos);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
bool UnsortedHashMapExploitSort<Key,Value,HashFunc,HT,ST>::contains(KeyPassType key)
{
  const ST size = Base::hash_value_.size();
  const HT cur_hash = Base::hash_(key);
  const ST pos = (is_sorted_) ?  Routines::binsearch(Base::hash_value_.direct_access(), cur_hash, size) :
                 Routines::find_unique(Base::hash_value_.direct_access(), cur_hash, size);

  if (pos >= size)
    return false;

  const FlexibleStorage1D<Key,ST>& cur_key_list = Base::key_[pos];
  const ST sub_pos = Routines::find_unique(cur_key_list.direct_access(), key, cur_key_list.size());
  return (sub_pos < cur_key_list.size());
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST>
Value& UnsortedHashMapExploitSort<Key,Value,HashFunc,HT,ST>::find_or_insert(const KeyPassType key, const Value& initial_value)
{
  const ST size = Base::hash_value_.size();
  const HT cur_hash = Base::hash_(key);
  const ST pos = (is_sorted_) ?  Routines::binsearch(Base::hash_value_.direct_access(), cur_hash, size) :
                 Routines::find_unique(Base::hash_value_.direct_access(), cur_hash, size);

  if (pos >= size) {
    Base::hash_value_.append(cur_hash);
    Base::key_.increase().append(key);
    Base::value_.increase().append(initial_value);
    if (is_sorted_ && size > 0 && cur_hash < Base::hash_value_[size-1])
      is_sorted_ = false;
    return Base::value_.back()[0];
  }

  FlexibleStorage1D<Key,ST>& cur_key_list = Base::key_[pos];
  const ST cur_size = cur_key_list.size();
  const ST sub_pos = Routines::find_unique(cur_key_list.direct_access(), key, cur_size);
  if (sub_pos >= cur_size) {
    cur_key_list.push_back(key);
    Base::value_[pos].push_back(initial_value);
    return Base::value_[pos][cur_size];
  }
  return Base::value_[pos][sub_pos];
}

/*********************/

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST, bool inner_sort>
Value& SortedHashMap<Key,Value,HashFunc,HT,ST,inner_sort>::operator[](KeyPassType key)
{
  //std::cerr << "SortedHashMap::operator[]" << std::endl;
  const ST size = Base::hash_value_.size();
  const static HashFunc hash_obj;
  const HT cur_hash = hash_obj(key);
  const ST pos = Routines::binsearch_insertpos(Base::hash_value_.direct_access(), cur_hash, size);

  assert(Base::key_.size() == size);
  assert(Base::value_.size() == size);

  if (pos >= size) {
    Base::hash_value_.append(cur_hash);
    Base::key_.increase().append(key);
    return Base::value_.increase().increase();
  }
  else if (Base::hash_value_[pos] != cur_hash) {

    Base::hash_value_.increase();
    Base::key_.increase();
    Base::value_.increase();

    Routines::upshift_array(Base::hash_value_.data(), pos, size, 1);
    Routines::upshift_array(Base::key_.data(), pos, size, 1);
    Routines::upshift_array(Base::value_.data(), pos, size, 1);

    Base::hash_value_[pos] = cur_hash;
    Base::key_[pos] = FlexibleStorage1D<Key,ST>();
    Base::value_[pos] = FlexibleStorage1D<Value,ST>();

    Base::key_[pos].append(key);
    return Base::value_[pos].increase();
  }

  if (inner_sort)
    return Base::sorted_inner_op(key, pos);
  else
    return Base::inner_op(key, pos);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST, bool inner_sort>
bool SortedHashMap<Key,Value,HashFunc,HT,ST,inner_sort>::contains(KeyPassType key)
{
  //std::cerr << "SortedHashMap::contains()" << std::endl;

  const ST size = Base::hash_value_.size();
  const static HashFunc hash_obj;
  const HT cur_hash = hash_obj(key);
  const ST pos = Routines::binsearch_insertpos<HT,std::less<HT>,std::equal_to<HT>,ST>(Base::hash_value_.direct_access(), cur_hash, size);

  if (pos >= size)
    return false;

  const FlexibleStorage1D<Key,ST>& cur_key_list = Base::key_[pos];
  const size_t cur_size = cur_key_list.size();
  size_t sub_pos = 0;
  if (inner_sort)
    sub_pos = Routines::binsearch<Key,std::less<Key>,std::equal_to<Key>,ST>(cur_key_list.data(),key,cur_size);
  else
    sub_pos = Routines::find_unique(cur_key_list.data(), key, cur_key_list.size());
  return (sub_pos < cur_size);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST, bool inner_sort>
Value& SortedHashMap<Key,Value,HashFunc,HT,ST,inner_sort>::find_or_insert(const KeyPassType key, const Value& initial_value, ST hint_bin)
{
  //std::cerr << "SortedHashMap::find_or_insert(" << key << ")" << std::endl;
  assert(is_unique_sorted(Base::hash_value_.direct_access(),Base::hash_value_.size()));

  const ST size = Base::hash_value_.size();
  const static HashFunc hash_obj;
  const HT cur_hash = hash_obj(key);
  ST pos;
  if (hint_bin < size && Base::hash_value_[hint_bin] == cur_hash)
    pos = hint_bin;
  else {
    //use hint_bin to decrease the search width
    if (hint_bin >= size || hint_bin == 0)
      //if (true)
      pos = Routines::binsearch_insertpos<HT,std::less<HT>,std::equal_to<HT>,ST>(Base::hash_value_.direct_access(), cur_hash, size);
    else if (cur_hash > Base::hash_value_[hint_bin]) {
      pos = Routines::binsearch_insertpos<HT,std::less<HT>,std::equal_to<HT>,ST>(Base::hash_value_.direct_access()+hint_bin+1,
            cur_hash, size-(hint_bin+1))
            + hint_bin + 1;
    }
    else
      pos = Routines::binsearch_insertpos<HT,std::less<HT>,std::equal_to<HT>,ST>(Base::hash_value_.direct_access(), cur_hash, hint_bin);
  }

  if (pos >= size) {
    Base::hash_value_.append(cur_hash);
    Base::key_.increase().append(key);
    Base::value_.increase().append(initial_value);
    assert(is_unique_sorted(Base::hash_value_.direct_access(),Base::hash_value_.size()));
    return Base::value_.back()[0];
  }
  if (Base::hash_value_[pos] != cur_hash) {
    Base::outer_insert(pos,cur_hash);
    assert(is_unique_sorted(Base::hash_value_.direct_access(),Base::hash_value_.size()));
  }

  if (inner_sort)
    return Base::sorted_inner_op(key, pos, initial_value);
  else
    return Base::inner_op(key, pos, initial_value);
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST, bool inner_sort>
ST SortedHashMap<Key,Value,HashFunc,HT,ST,inner_sort>::find_bin(const HT hash_value) const
{
  return Routines::binsearch<HT,std::less<HT>,std::equal_to<HT>,ST>(Base::hash_value_.direct_access(), hash_value, Base::hash_value_.size());
}

template<typename Key, typename Value, typename HashFunc, typename HT, typename ST, bool inner_sort>
ST SortedHashMap<Key,Value,HashFunc,HT,ST,inner_sort>::find_or_insert_bin(const HT hash_value)
{
  //std::cerr << "SortedHashMap::find_or_insert_bin()" << std::endl;
  assert(is_unique_sorted(Base::hash_value_.direct_access(),Base::hash_value_.size()));

  const ST size = Base::hash_value_.size();
  const ST pos = Routines::binsearch_insertpos<HT,std::less<HT>,std::equal_to<HT>,ST>(Base::hash_value_.direct_access(), hash_value, size);

  if (pos < size && Base::hash_value_[pos] == hash_value)
    return pos;

  if (pos < size && Base::key_[pos].size() == 0) {
    Base::hash_value_[pos] = hash_value;
    return pos;
  }

  assert(is_unique_sorted(Base::hash_value_.direct_access(),Base::hash_value_.size()));
  Base::outer_insert(pos, hash_value);
  assert(is_unique_sorted(Base::hash_value_.direct_access(),Base::hash_value_.size()));
  return pos;
}

#endif