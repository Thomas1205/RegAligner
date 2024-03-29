/**************** written by Thomas Schoenemann as a private person, June 2020 ********************/

#ifndef SORTED_MAP_HH
#define SORTED_MAP_HH

#include <vector>
#include "routines.hh"
#include "flexible_storage1D.hh"
#include "unsorted_map.hh"
#include "hash_map.hh"

//so far operator< is fixed

//use this if your keys are small and easily comparable
//so far this map does not offer erasing
template<typename Key, typename Value, typename KVec = std::vector<Key>, typename VVec = std::vector<Value>,
         typename Less = std::less<Key>, typename Equal = std::equal_to<Key> >
class SortedMap : public MapBase<Key,Value,KVec,VVec> {
public:

  using Base = MapBase<Key,Value,KVec,VVec>;
  using KeyPassType = typename std::conditional<std::is_fundamental<Key>::value || std::is_pointer<Key>::value, const Key, const Key&>::type;

  SortedMap() {};

  SortedMap(const SortedMap<Key, Value>& toCopy) : MapBase<Key,Value,KVec,VVec>(toCopy) {}

  SortedMap(SortedMap<Key, Value>&& toTake) : MapBase<Key,Value,KVec,VVec>(toTake) {}

  ~SortedMap() {}

  size_t keypos(const KeyPassType key) const noexcept;

  bool contains(const KeyPassType key) const noexcept;

  Value& find_or_insert(const KeyPassType key, const Value& initial_value); //vector can throw std::bad_alloc()

  Value& find_or_insert(Key&& key, const Value& initial_value); //vector can throw std::bad_alloc()

  Value& operator[](const KeyPassType key); //vector can throw std::bad_alloc()
  
  Value operator()(const KeyPassType key, const Value backoff_value) const noexcept;

  void operator=(const SortedMap<Key, Value>& toCopy);

  void operator=(SortedMap<Key, Value>&& toTake);

  template<typename Hash, typename HT, typename ST>
  void assign(const HashMapBase<Key,Value,Hash,HT,ST>& toCopy);

  template<typename Hash, typename HT, typename ST>
  void move_assign(HashMapBase<Key,Value,Hash,HT,ST>&& toTake);
};

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
size_t SortedMap<Key,Value,KVec,VVec,Less,Equal>::keypos(const KeyPassType key) const noexcept
{
  return Routines::binsearch_insertpos<Key,Less,Equal>(Base::key_.data(), key, Base::key_.size());
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
bool SortedMap<Key,Value,KVec,VVec,Less,Equal>::contains(const KeyPassType key) const noexcept
{
  const size_t size = Base::key_.size();
  return (Routines::binsearch<Key,Less,Equal>(Base::key_.data(), key, size) < size);
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value& SortedMap<Key,Value,KVec,VVec,Less,Equal>::operator[](const KeyPassType key) 
{
  const size_t size = Base::key_.size();
  const size_t inspos = Routines::binsearch_insertpos<Key,Less,Equal>(Base::key_.data(), key, size);

  if (inspos >= size) {
    Base::key_.push_back(key);
    Base::value_.push_back(Value());
    return Base::value_.back();
  }
  else {

    const static Equal equal;
    if (!equal(Base::key_[inspos],key)) {

      Base::key_.push_back(Key());
      Base::value_.push_back(Value());
      Routines::upshift_array(Base::key_.data(), inspos, size, 1);
      Routines::upshift_array(Base::value_.data(), inspos, size, 1);
      Base::key_[inspos] = key;
      Base::value_[inspos] = Value();
    }
    return Base::value_[inspos];
  }
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value SortedMap<Key,Value,KVec,VVec,Less,Equal>::operator()(const KeyPassType key, const Value backoff_value) const noexcept
{
  const size_t size = Base::key_.size();
  const size_t pos = Routines::binsearch<Key,Less,Equal>(Base::key_.data(), key, size);

  if (pos < size)
	return Base::value_[pos];
  return backoff_value;
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value& SortedMap<Key,Value,KVec,VVec,Less,Equal>::find_or_insert(const KeyPassType key, const Value& initial_value)
{
  //std::cerr << "********* find_or_insert with key " << key << std::endl;	
	
  const size_t size = Base::key_.size();
  const size_t inspos = Routines::binsearch_insertpos<Key,Less,Equal>(Base::key_.data(), key, size);

  //std::cerr << "key: " << Base::key_ << ", inspos: " << inspos << std::endl;

  if (inspos >= size) {
    Base::key_.push_back(key);
    Base::value_.push_back(initial_value);
    return Base::value_.back();
  }
  else {

    const static Equal equal;
    if (!equal(Base::key_[inspos],key)) {
	  //std::cerr << "not equal" << std::endl;
      Base::key_.push_back(Key());
      Base::value_.push_back(initial_value);
	  //std::cerr << "after push: " << Base::key_ 
	  //          << std::endl << ", value: " << Base::value_ << std::endl;
      Routines::upshift_array(Base::key_.data(), inspos, size, 1);
      Routines::upshift_array(Base::value_.data(), inspos, size, 1);
	  //std::cerr << "after upshift: " << Base::key_ 
	  //		  << std::endl << ", value: " << Base::value_ << std::endl;
      Base::key_[inspos] = key;
	  //std::cerr << "after key assignment: " << Base::key_ << std::endl;
      Base::value_[inspos] = initial_value;
	  //std::cerr << "after assignment: " << Base::key_ << ", value: " << Base::value_ << std::endl;
    }
    return Base::value_[inspos];
  }
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
Value& SortedMap<Key,Value,KVec,VVec,Less,Equal>::find_or_insert(Key&& key, const Value& initial_value)
{
  //std::cerr << "********* find_or_insert with key " << key << std::endl;	
	
  const size_t size = Base::key_.size();
  const size_t inspos = Routines::binsearch_insertpos<Key,Less,Equal>(Base::key_.data(), key, size);

  //std::cerr << "key: " << Base::key_ << ", inspos: " << inspos << std::endl;

  if (inspos >= size) {
    Base::key_.push_back(key);
    Base::value_.push_back(initial_value);
    return Base::value_.back();
  }
  else {

    const static Equal equal;
    if (!equal(Base::key_[inspos],key)) {
	  //std::cerr << "not equal" << std::endl;
      Base::key_.push_back(Key());
      Base::value_.push_back(initial_value);
	  //std::cerr << "after push: " << Base::key_ 
	  //          << std::endl << ", value: " << Base::value_ << std::endl;
      Routines::upshift_array(Base::key_.data(), inspos, size, 1);
      Routines::upshift_array(Base::value_.data(), inspos, size, 1);
	  //std::cerr << "after upshift: " << Base::key_ 
	  //		  << std::endl << ", value: " << Base::value_ << std::endl;
      Base::key_[inspos] = key;
	  //std::cerr << "after key assignment: " << Base::key_ << std::endl;
      Base::value_[inspos] = initial_value;
	  //std::cerr << "after assignment: " << Base::key_ << ", value: " << Base::value_ << std::endl;
    }
    return Base::value_[inspos];
  }
}


template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
void SortedMap<Key,Value,KVec,VVec,Less,Equal>::operator=(const SortedMap<Key, Value>& toCopy)
{
  Base::key_ = toCopy.key();
  Base::value_ = toCopy.value();
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
void SortedMap<Key,Value,KVec,VVec,Less,Equal>::operator=(SortedMap<Key, Value>&& toTake)
{
  Base::key_.swap(toTake.key_);
  Base::value_.swap(toTake.value_);
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
template<typename Hash, typename HT, typename ST>
void SortedMap<Key,Value,KVec,VVec,Less,Equal>::assign(const HashMapBase<Key,Value,Hash,HT,ST>& toCopy)
{
  const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& key_stack = toCopy.key_stack();
  const FlexibleStorage1D<FlexibleStorage1D<Value,ST>,ST>& value_stack = toCopy.value_stack();

  ST size = 0;
  for (ST k=0; k < key_stack.size(); k++)
    size += key_stack[k].size();

  Base::key_.clear();
  Base::key_.reserve(size);
  Base::value_.clear();
  Base::value_.reserve(size);

  for (ST k=0; k < key_stack.size(); k++) {
    const FlexibleStorage1D<Key,ST>& cur_keys = key_stack[k];
    const FlexibleStorage1D<Value,ST>& cur_values = value_stack[k];
    for (ST kk = 0; kk < cur_keys.size(); kk++) {
      Base::key_.push_back(cur_keys[kk]);
      Base::value_.push_back(cur_values[kk]);
    }
  }

  merge_sort_key_value(Base::key_.data(), Base::value_.data(), size);
}

template<typename Key, typename Value, typename KVec, typename VVec, typename Less, typename Equal>
template<typename Hash, typename HT, typename ST>
void SortedMap<Key,Value,KVec,VVec,Less,Equal>::move_assign(HashMapBase<Key,Value,Hash,HT,ST>&& toTake)
{
  const FlexibleStorage1D<FlexibleStorage1D<Key,ST>,ST>& key_stack = toTake.key_stack();
  const FlexibleStorage1D<FlexibleStorage1D<Value,ST>,ST>& value_stack = toTake.value_stack();

  ST size = 0;
  for (ST k=0; k < key_stack.size(); k++)
    size += key_stack[k].size();

  Base::key_.clear();
  Base::key_.reserve(size);
  Base::value_.clear();
  Base::value_.reserve(size);

  for (ST k=0; k < key_stack.size(); k++) {
    const FlexibleStorage1D<Key,ST>& cur_keys = key_stack[k];
    const FlexibleStorage1D<Value,ST>& cur_values = value_stack[k];
    for (ST kk = 0; kk < cur_keys.size(); kk++) {
      Base::key_.push_back(Key());
      Base::key_.back() = std::move(cur_keys[kk]);
      Base::value_.push_back(Value());
      Base::value_.back() = std::move(cur_values[kk]);
    }
  }

  merge_sort_key_value(Base::key_.data(), Base::value_.data(), size);
}

#endif