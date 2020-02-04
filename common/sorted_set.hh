/************* written by Thomas Schoenemann as a private person, February 2020 ********/

#ifndef SORTEDSET_HH
#define SORTEDSET_HH

#include "stl_util.hh"

template<typename T>
class SortedSet {
public:
	
	SortedSet() {}

	SortedSet(const SortedSet<T>& toCopy);
	
	void swap(TreeSet<T>& other) {
		data_.swap(other.data_);
	}

	size_t size() const {
		return data_.size();
	}
	
	void reserve(size_t size) {
		data_.reserve(size);
	}
	
	void clear() {
		data_.clear();
	}
	
	const std::vector<T>& sorted_data() const {
		return data_;
	}

	bool contains(T val) const;

	//returns true if val is new
	bool insert(T val);

	//returns true if val was in the tree
	bool erase(T val);

	//returns true if out was in the tree
	bool replace(T out, T in);

protected:
		
	std::vector<T> data_;
};

/********************** implementation ************************/

template<typename T>
SortedSet<T>::SortedSet(const SortedSet<T>& toCopy)
	: data_(toCopy.data_) {}

template<typename T>
bool SortedSet<T>::contains(T val) const
{
	return (binsearch(data_, val) != MAX_UINT);
}

//returns true if val is new
template<typename T>
bool SortedSet<T>::insert(T val) 
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
	
	for (uint k = size; k > inspos; k--)
		data_[k] = data_[k-1];
	
	data_[inspos] = val;
	
	return true;
}

//returns true if val was in the tree
template<typename T>
bool SortedSet<T>::erase(T val)
{
	//std::cerr << "erase" << std::endl;
	
	const size_t pos = binsearch(data_, val);
	if (pos == MAX_UINT)
		return false;

	const size_t size = data_.size();
	for (uint k = pos; k < size-1; k++)
		data_[k] = data_[k+1];

	data_.resize(size-1);	
	
	return true;
}

//returns true if out was in the tree
template<typename T>
bool SortedSet<T>::replace(T out, T in)
{
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
			while (npos < size && data_[npos+1] < in)
				npos++;

			for (size_t k = pos; k < npos; k++)
				data_[k] = data_[k-1];
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