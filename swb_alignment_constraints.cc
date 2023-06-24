/*** Written by Thomas Schoenemann as a private person, since October 2019 ***/

#include "swb_alignment_constraints.hh"
#include "combinatoric.hh"
#include "tristorage2D.hh"
#include "stl_util.hh"
#include "stl_out.hh"

#include <fstream>

template<typename T>
bool subalignment_satisfies_ibm_nonull(const Math1D::Vector<T>& alignment, const Storage1D<std::vector<short> >& aligned_source_words,
                                       uint nMaxSkips, uint j1, uint j2)
{
  short mini = 10000;
  short maxi = -1;

  for (uint j = j1; j <= j2; j++) {
    mini = std::min<short>(alignment[j],mini);
    maxi = std::max<short>(alignment[j],maxi);
  }

  for (short i = mini; i <= maxi; i++) {

    const std::vector<short>& cur_aligned = aligned_source_words[i];
    for (uint k=0; k < cur_aligned.size(); k++) {
      uint j = cur_aligned[k];
      if (j < j1 || j > j2)
        return false;
    }
  }

  int max_covered=j1-1;
  std::set<uint> open;
  for (uint i=mini; i <= maxi; i++) {

    const std::vector<short>& cur_aligned = aligned_source_words[i];
    for (uint k=0; k < cur_aligned.size(); k++) {
      int j = cur_aligned[k];
      if (j == max_covered+1)
        max_covered++;
      else if (contains<uint>(open,j))
        open.erase(j);
      else {
        assert(j > max_covered);
        for (uint jj=max_covered+1; jj < j; jj++) {
          assert(!contains<uint>(open,jj));
          open.insert(jj);
        }
        if (open.size() > nMaxSkips)
          return false;
        max_covered = j;
      }
    }
  }

  return true;
}

bool alignment_satisfies_itg_nonull(const Math1D::Vector<AlignBaseType>& alignment, uint I, uint ext_level, int max_mid_dev,
                                    uint nMaxSkips, uint level3_maxlength, uint max_fert)
{
  if (alignment.min() == 0)
    return false; //NULL alignments are not allowed

  //we can discard unaligned target words for this check (if max_mid_dev is high enough)
  const uint J = alignment.size();
  Math1D::Vector<uint> fertility(I+1,0);
  Math1D::Vector<short> inum(I+1,-1);
  Math1D::Vector<short> minj(I+1,10000);
  Math1D::Vector<short> maxj(I+1,-1);

  Storage1D<std::vector<short> > aligned_source_words(I+1);

  for (uint j=0; j < J; j++) {
    const uint aj = alignment[j];
    fertility[aj]++;
    if (ext_level >= 4)
      aligned_source_words[aj].push_back(j);
    minj[aj] = std::min<short>(minj[aj],j);
    maxj[aj] = std::max<short>(maxj[aj],j);
  }

  uint next_num = 0;

  for (uint i=1; i <= I; i++) {
    if (fertility[i] != 0) {
      inum[i] = next_num;
      next_num++;
    }
  }

  TriStorage2D<std::set<std::pair<ushort,ushort> > > parsed(I+1);

  if (ext_level >= 4) {

    for (uint j1=0; j1 < J; j1++) {

      if (minj[alignment[j1]] != j1)
        continue;

      for (uint j2=j1+2; j2 < std::min(j1+level3_maxlength,J); j2++) {

        if (maxj[alignment[j2]] != j2)
          continue;

        if (subalignment_satisfies_ibm_nonull(alignment, aligned_source_words, nMaxSkips, j1, j2)) {

          short mini = 10000;
          short maxi = -1;

          for (uint j = j1; j <= j2; j++) {
            mini = std::min<short>(alignment[j],mini);
            maxi = std::max<short>(alignment[j],maxi);
          }

          if (maxi-mini+1 <= level3_maxlength)
            parsed(mini,maxi).insert(std::make_pair<ushort,ushort>(j1,j2));
        }
      }
    }
  }

  //base case Ilength = 1
  for (int i=1; i <= I; i++) {

    if (fertility[i] > 0) {

      bool closed = true;
      for (uint j=minj[i]+1; j < maxj[i]; j++) {
        if (alignment[j] != i) {
          closed = false;
          break;
        }
      }
      if (!closed && ext_level == 0)
        return false;

      if (closed) {
        //std::cerr << "insert(" << i << "," << i << ")(" << minj[i] << "," << maxj[i] << ")" << std::endl;
        parsed(i,i).insert(std::make_pair<ushort,ushort>(minj[i],maxj[i]));
      }
      else if (ext_level >= 1) {

        // bool passes_ext1 = false;

         //int other = -1;
        // uint count = 0;
        // passes_ext1 = true;
        //for (uint j=minj[i]+1; j < maxj[i]; j++) {
          //const int aj = alignment[j];
          // //std::cerr << "aj: " << aj << std::endl;
          // if (aj != i) {
            // //std::cerr << "values: " << fabs(aj-i) << ", " << fabs(orgi[aj]-orgi[i]) <<  std::endl;
            // if (std::abs(inum[aj]-inum[i]) != 1 || std::abs(aj-i) > 10) {
              // passes_ext1 = false;
              // break;
            // }
            // else {
              // if (other == -1)
                // other = aj;
              // else if (other != aj) {
                // passes_ext1 = false;
                // break;
              // }
              // count++;
            // }
          // }
        // }

        // if (passes_ext1 && fertility[other] != count)
          // passes_ext1 = false;

        // //std::cerr << "passes_ext1: " << passes_ext1 << std::endl;

        // if (passes_ext1) {
          // assert(other != -1);
          // //std::cerr << "insert ext1 (" << std::min(i,other) << "," << std::max(i,other) << ")("
          // //          << minj[i] << "," << maxj[i] << ")" << std::endl;
          // parsed(std::min(i,other),std::max(i,other)).insert(std::make_pair<ushort,ushort>(minj[i],maxj[i]));
        // }
        
		if (ext_level >= 2 && maxj[i]+1 < J && fertility[i] == 2 && maxj[i] == minj[i]+2) {

          int other = alignment[minj[i]+1];
          //std::cerr << "other: "<< other << std::endl;

          //check for i - ii - i - ii
          if (std::abs(inum[other]-inum[i]) == 1 && fertility[other] == 2 && std::abs(other-i) <= 10
              && other == alignment[maxj[i]+1]) {
            //std::cerr << "insert (" << std::min(i,other) << "," << std::max(i,other) << ")("
            //          << std::min(minj[i],minj[other]) << "," << std::max(maxj[i],maxj[other]) << ")" << std::endl;
            parsed(std::min(i,other),std::max(i,other)).insert(std::make_pair<ushort,ushort>(std::min(minj[i],minj[other]),std::max(maxj[i],maxj[other])));
          }
        }
      }
    }
  }

  if (ext_level >= 1) {
	for (uint j1 = 0; j1 < J; j1++) {
	  for (uint j2 = j1+1; j2 < J; j2++) {
		  
		const uint J = j2-j1+1;
		if (!(J <= std::min<int>(10,max_fert+1) && J > 2))
		  continue;
		  
		std::set<int> is;
		for (uint jj = j1; jj <= j2; jj++) 
		  is.insert(alignment[jj]);
		if (is.size() == 2) {
		  std::set<int>::const_iterator it = is.begin();
		  int i1 = *it;
		  it++;
		  int i2 = *it;
		  const uint I = i2-i1+1;
		  if (I <= 10 && minj[i1] >= j1 && minj[i2] >= j1 && maxj[i1] <= j2 && maxj[i2] <= j2) {
			//bool pass = ((alignment[j1] == i1 && alignmet[j2] == i1) //level 1 cannot handle constellations that start or end with different words
		    //	 			|| (alignment[j1] == i2 && alignment[j2] == i2));
			bool pass = (alignment[j1] == alignment[j2]); //level 1 cannot handle constellations that start or end with different words
			for (int ii = i1+1; ii <= i2-1; ii++) {
			  if (fertility[ii] != 0)
			  {
				pass = false;
				break;
			  }
			}				
			if (pass) 
			  parsed(i1,i2).insert(std::make_pair<ushort>(j1,j2));	
		  }
		}
	  }
	}		
  }

  std::set<ushort> dividedis;
  if (ext_level >= 2) {
    for (uint i=1; i <= I; i++) {
      if (fertility[i] > 0 && maxj[i] - minj[i] != fertility[i]-1)
        dividedis.insert(i);
    }

    if (J >= 4) {
      //check for the two 1-1 alignments ITG cannot cover
      for (uint j=0; j < J-3; j++) {
        const uint aj1 = alignment[j];
        const uint aj2 = alignment[j+1];
        const uint aj3 = alignment[j+2];
        const uint aj4 = alignment[j+3];

        if (fertility[aj1] == 1 && fertility[aj2] == 1 && fertility[aj3] == 1 && fertility[aj4] == 1) {

          if (aj2 == aj1+2 && aj3 == aj1-1 && aj4 == aj1+1) {
            parsed(aj1-1,aj1+2).insert(std::make_pair<ushort,ushort>(j,j+3));
          }
          else if (aj2 == aj1-2 && aj3 == aj1+1 && aj4 == aj1-1) {
            parsed(aj1-2,aj1+1).insert(std::make_pair<ushort,ushort>(j,j+3));
          }
        }
      }
    }
  }

  for (ushort Ilength = 2; Ilength <= I; Ilength++) {

    //std::cerr << "Ilength: " << Ilength << std::endl;

    for (ushort i=1; i <= I-Ilength+1; i++) {

      const ushort ii = i+Ilength-1;
      std::set<std::pair<ushort,ushort> >& cur_parsed = parsed(i,ii);

      const uint fi  = fertility[i];
      const uint fii = fertility[ii];

      const std::set<std::pair<ushort,ushort> >& parsed1 = parsed(i+1,ii);
      const std::set<std::pair<ushort,ushort> >& parsed2 = parsed(i,ii-1);

      if (fi == 0) {
        for (std::set<std::pair<ushort,ushort> >::const_iterator it = parsed1.begin(); it != parsed1.end(); it++)
          cur_parsed.insert(*it);
      }
      if (fii == 0) {
        for (std::set<std::pair<ushort,ushort> >::const_iterator it = parsed2.begin(); it != parsed2.end(); it++)
          cur_parsed.insert(*it);
      }

      const int imid_point = i + (Ilength / 2);

      const int i_lower = std::max<int>(i,imid_point-max_mid_dev);
      const int i_upper = std::min<int>(ii,imid_point+max_mid_dev+1);

      for (int isplit = i_lower; isplit < i_upper; isplit++) {

        const std::set<std::pair<ushort,ushort> >& oparsed = parsed(i,isplit);
        const std::set<std::pair<ushort,ushort> >& iparsed = parsed(isplit+1,ii);

        for (std::set<std::pair<ushort,ushort> >::const_iterator it1 = oparsed.begin(); it1 != oparsed.end(); it1++) {
          for (std::set<std::pair<ushort,ushort> >::const_iterator it2 = iparsed.begin(); it2 != iparsed.end(); it2++) {

            //monotone
            if (it2->first == it1->second+1) {

              const int J = it2->second - it1->first + 1;
              const int j_split = it1->second;

              const int jmid_point = it1->first + (J / 2);

              const int j_lower = std::max<int>(it1->first,jmid_point-max_mid_dev);
              const int j_upper = std::min<int>(it2->second,jmid_point+max_mid_dev+1);

              if (j_split >= j_lower && j_split < j_upper) {
                //if (!contains(cur_parsed,std::make_pair(it1->first,it2->second)))
                //  std::cerr << "new insert (" << i << "," << ii << ")(" << it1->first << "," << it2->second << ")" << std::endl;
                cur_parsed.insert(std::make_pair(it1->first,it2->second));
              }
            }

            //reordered
            if (it1->first == it2->second+1) {

              const int J = it1->second - it2->first + 1;
              const int j_split = it2->second;

              const int jmid_point = it2->first + (J / 2);

              const int j_lower = std::max<int>(it2->first,jmid_point-max_mid_dev);
              const int j_upper = std::min<int>(it1->second,jmid_point+max_mid_dev+1);

              if (j_split >= j_lower && j_split < j_upper) {
                //if (!contains(cur_parsed,std::make_pair(it2->first,it1->second)))
                //  std::cerr << "new insert (" << i << "," << ii << ")(" << it2->first << "," << it1->second << ")" << std::endl;
                cur_parsed.insert(std::make_pair(it2->first,it1->second));
              }
            }
          }
        }
      }

      if (ext_level >= 2) {

        const bool i_in = contains(dividedis,i);
        const bool ii_in = contains(dividedis,ii);

        if (fi == 2 && i_in) {

          if (contains(parsed1,std::make_pair<ushort,ushort>(minj[i]+1,maxj[i]-1))) {

            //std::cerr << "ext2 parse case 1 for i=" << i << std::endl;
            //std::cerr << "insert (" << i << "," << ii << ")(" << minj[i] << "," << maxj[i] << ")" << std::endl;

            cur_parsed.insert(std::make_pair<ushort,ushort>(minj[i],maxj[i]));
          }
        }
        if (fii == 2 && ii_in) {

          //std::cerr << "ext2 check for ii=" << ii << ", with i=" << i << std::endl;
          //std::cerr << "looking for " << (minj[ii]+1) << "," << (maxj[ii]-1) << std::endl;
          //std::cerr << "   in " << parsed2 << std::endl;

          if (contains(parsed2,std::make_pair<ushort,ushort>(minj[ii]+1,maxj[ii]-1))) {

            //std::cerr << "ext2 parse case 2 for ii=" << ii << std::endl;
            //std::cerr << "insert (" << i << "," << ii << ")(" << minj[ii] << "," << maxj[ii] << ")" << std::endl;

            cur_parsed.insert(std::make_pair<ushort,ushort>(minj[ii],maxj[ii]));
          }
        }

        if (ext_level >= 3) {

          if (fi == 3 && i_in) {

            //std::cerr << "ext3 cases a+c for i=" << i << ", ii=" << ii << std::endl;

            if ((alignment[minj[i]+1] == i && contains(parsed(i+1,ii),std::make_pair<ushort,ushort>(minj[i]+2,maxj[i]-1)))
                || (alignment[maxj[i]-1] == i && contains(parsed(i+1,ii),std::make_pair<ushort,ushort>(minj[i]+1,maxj[i]-2))) ) {
              //std::cerr << "passed" << std::endl;
              cur_parsed.insert(std::make_pair<ushort,ushort>(minj[i],maxj[i]));
            }
          }
          if (fii == 3 && ii_in) {

            //std::cerr << "ext3 cases b+d for ii=" << ii << ", i=" << i << std::endl;

            if ((alignment[minj[ii]+1] == ii && contains(parsed(i,ii-1),std::make_pair<ushort,ushort>(minj[ii]+2,maxj[ii]-1)))
                || (alignment[maxj[ii]-1] == ii && contains(parsed(i,ii-1),std::make_pair<ushort,ushort>(minj[ii]+1,maxj[ii]-2))) ) {
              //std::cerr << "passed" << std::endl;
              cur_parsed.insert(std::make_pair<ushort,ushort>(minj[ii],maxj[ii]));
            }
          }

          if (fi == 4 && i_in) {

            //std::cerr << "ext3 case e for i=" << i << ", ii=" << ii << std::endl;

            if (alignment[minj[i]+1] == i && alignment[maxj[i]-1] == i && contains(parsed1,std::make_pair<ushort,ushort>(minj[i]+2,maxj[i]-2))) {
              //std::cerr << "passed" << std::endl;
              cur_parsed.insert(std::make_pair<ushort,ushort>(minj[i],maxj[i]));
            }
          }

          if (fii == 4 && ii_in) {

            //std::cerr << "ext3 case e for ii=" << ii << ", i=" << i << std::endl;

            if (alignment[minj[ii]+1] == ii && alignment[maxj[ii]-1] == ii && contains(parsed2,std::make_pair<ushort,ushort>(minj[ii]+2,maxj[ii]-2))) {
              //std::cerr << "passed" << std::endl;
              cur_parsed.insert(std::make_pair<ushort,ushort>(minj[ii],maxj[ii]));
            }
          }
        }
      }
    }
  }

  return contains(parsed(1,I),std::make_pair<ushort,ushort>(0,J-1));
}

bool alignment_satisfies_ibm_nonull(const Math1D::Vector<AlignBaseType>& alignment, uint nMaxSkips)
{
  //std::cerr << "checking alignment " << alignment << std::endl;

  if (alignment.min() == 0)
    return false; //NULL is not allowed!

  const uint J = alignment.size();
  const uint I = alignment.max();

  Storage1D<std::vector<ushort> > aligned_source_words(I+1);
  for (uint j=0; j < J; j++)
    aligned_source_words[alignment[j]].push_back(j);

  int max_covered=-1;
  std::set<uint> open;
  for (uint i=1; i <= I; i++) {

    const std::vector<ushort>& cur_aligned = aligned_source_words[i];
    for (uint k=0; k < cur_aligned.size(); k++) {
      int j = cur_aligned[k];
      if (j == max_covered+1)
        max_covered++;
      else if (contains<uint>(open,j))
        open.erase(j);
      else {
        assert(j > max_covered);
        for (uint jj=max_covered+1; jj < j; jj++) {
          assert(!contains<uint>(open,jj));
          open.insert(jj);
        }
        if (open.size() > nMaxSkips)
          return false;
        max_covered = j;
      }
    }
  }

  return true;
}

#if 0
struct BatchStruct {

  BatchStruct(uint f, uint l, uint s) : first_pos_(f), last_pos_(l), state_(s) {}

  uint first_pos_;
  uint last_pos_;
  uint state_;
};
#endif

void IBMConstraintStates::compute_segments(uint nMaxSkips, uint uncovered_set_num,
    std::vector<std::pair<uint, uint> >& segments) const
{
  // std::cerr << "*********** IBMConstraintStates::compute_segments *************************" << std::endl;
  // std::cerr << "processing set state ";
  // print_uncovered_set(uncovered_set_num);
  // std::cerr << std::endl;

  segments.clear();
  const uint nUncoveredPositions = nUncoveredPositions_[uncovered_set_num];
  if (nUncoveredPositions == 0)
    return;

  if (nMaxSkips == 1) {
    uint pos = uncovered_set_(nMaxSkips-1, uncovered_set_num);
    segments.push_back(std::make_pair(pos,pos));
  }
  else {
    //search for consecutive skips as these can only be inserted at once
    uint last_pos = uncovered_set_(nMaxSkips-1, uncovered_set_num);
    uint cur_pos = last_pos;

    assert((nMaxSkips - nUncoveredPositions) >= 0);
    for (int k = nMaxSkips - 2; k >= (int)(nMaxSkips - nUncoveredPositions); k--) {

      const uint pos = uncovered_set_(k, uncovered_set_num);
      //std::cerr << "k: " << k << ", pos: " << pos << std::endl;
      if (pos == cur_pos - 1)
        cur_pos = pos;
      else {
        segments.push_back(std::make_pair(cur_pos,last_pos));
        last_pos = pos;
        cur_pos = pos;
      }
    }
    //std::cerr << "pushing " << cur_pos << ", " << last_pos << std::endl;
    segments.push_back(std::make_pair(cur_pos,last_pos));

    //std::cerr << "extracted segments: " << segments << std::endl;
  }
}

void IBMConstraintStates::compute_segments(uint* data, uint nMaxSkips, std::vector<std::pair<uint, uint> >& segments) const
{
  //std::cerr << "*********** IBMConstraintStates::compute_segments *************************" << std::endl;
  //std::cerr << "processing set state ";
  //for (uint k = 0; k < nMaxSkips; k++)
  //{
  //std::cerr << data[k] << ", ";
  //}
  //std::cerr << std::endl;

  segments.clear();
  if (data[nMaxSkips-1] == MAX_USHORT)
    return;

  uint nUncoveredPositions = 0;
  for (uint k = 0; k < nMaxSkips; k++) {
    if (data[k] != MAX_USHORT)
      nUncoveredPositions++;
  }

  if (nMaxSkips == 1) {
    uint pos = data[0];
    segments.push_back(std::make_pair(pos,pos));
  }
  else {
    //search for consecutive skips as these can only be inserted at once
    uint last_pos = data[nMaxSkips-1]; //uncovered_set_(nMaxSkips-1, uncovered_set_num);
    uint cur_pos  = last_pos;

    //std::cerr << "starting with cur_pos " << cur_pos << std::endl;

    assert((nMaxSkips - nUncoveredPositions) >= 0);
    for (int k = nMaxSkips - 2; k >= (int)(nMaxSkips - nUncoveredPositions); k--) {

      //std::cerr << "k: " << std::endl;
      const uint pos = data[k]; //uncovered_set_(k, uncovered_set_num);
      if (pos == cur_pos - 1)
        cur_pos = pos;
      else {
        segments.push_back(std::make_pair(cur_pos,last_pos));
        last_pos = pos;
        cur_pos = pos;
      }
    }
    segments.push_back(std::make_pair(cur_pos,last_pos));

    //std::cerr << "extracted segments: " << segments << std::endl;
  }
}

void IBMConstraintStates::compute_uncovered_sets(uint maxJ, uint nMaxSkips)
{
  uint nSets = 1;
  for (uint k=1; k <= nMaxSkips; k++)
    nSets += choose(maxJ,k);
  std::cerr << nSets << " sets of uncovered positions" << std::endl;

  uncovered_set_.resize_dirty(nMaxSkips, nSets);
  uncovered_set_.set_constant(MAX_USHORT);
  first_set_.resize_dirty(maxJ + 2);
  first_set_.set_constant(1);

  next_set_idx_ = 1;            //the first set contains no uncovered positions at all

  if (nMaxSkips > 0) {
    for (uint j = 1; j <= maxJ; j++) {

      first_set_[j] = next_set_idx_;
      uncovered_set_(nMaxSkips - 1, next_set_idx_) = j;

      cover(nMaxSkips - 1);
    }
    first_set_[maxJ + 1] = next_set_idx_;
  }

  assert(nSets == next_set_idx_);

  predecessor_sets_.resize(nSets);

  nUncoveredPositions_.resize_dirty(nSets);
  for (uint state = 0; state < nSets; state++)
    nUncoveredPositions_[state] = nUncoveredPositions(state);

  uint nMaxPredecessors = 0;

  //predecessors for set 0 (no skips)
  {
    std::vector<std::pair<uint,uint> > cur_predecessor_sets;
    for (uint prev_set_num = 1; prev_set_num < next_set_idx_; prev_set_num++) {
      if (nUncoveredPositions_[prev_set_num] == 1) {
        cur_predecessor_sets.push_back(std::make_pair(prev_set_num, uncovered_set_(nMaxSkips-1,prev_set_num)));
      }
    }

    const uint nCurPredecessors = cur_predecessor_sets.size();
    predecessor_sets_[0].resize(2, nCurPredecessors);
    uint k;
    for (k = 0; k < nCurPredecessors; k++) {
      predecessor_sets_[0](0, k) = cur_predecessor_sets[k].first;
      predecessor_sets_[0](1, k) = cur_predecessor_sets[k].second;
    }

    nMaxPredecessors = std::max(nMaxPredecessors, nCurPredecessors);
  }

  for (uint state = 1; state < next_set_idx_; state++) {

    std::vector<std::pair<uint,uint> > cur_predecessor_sets;

    // if (state <= 10) {
    // std::cerr << "########## processing set state " << state << " with uncovered set ";
    // print_uncovered_set(state);
    // std::cerr << std::endl;
    // }

    const uint nUncoveredPositions = nUncoveredPositions_[state];
    const uint highestUncoveredPos = uncovered_set_(nMaxSkips - 1, state);
    uint lowestUncoveredPos = highestUncoveredPos;
    for (int k = nMaxSkips-2; k >= 0; k--) {
      if (uncovered_set_(k,state) < MAX_USHORT) {
        lowestUncoveredPos = uncovered_set_(k,state);
      }
    }

    //if (state <= 10) {
    //std::cerr << "state: " << state << ", highest uncovered pos: " << highestUncoveredPos << std::endl;
    //std::cerr << "lower: " << first_set_[highestUncoveredPos] << std::endl;
    //std::cerr << "higher: " << first_set_[highestUncoveredPos + 1] << std::endl;
    //}

    assert(state >= first_set_[highestUncoveredPos]);
    assert(state < first_set_[highestUncoveredPos + 1]);

    //search for consecutive skips at the end as these can only be inserted at once
    uint nConsecutiveEndSkips = 1;
    for (int k = nMaxSkips - 2; k >= ((int)(nMaxSkips - nUncoveredPositions)); k--) {

      if (uncovered_set_(k, state) == uncovered_set_(k + 1, state) - 1)
        nConsecutiveEndSkips++;
      else
        break;
    }

    //NOTE: a state is always its own predecessor state (by appending a sole target word or covering a j in sequence);
    //  to save memory we omit the entry

    if (nUncoveredPositions == nMaxSkips) {

      //std::cerr << "current set has max number of uncovered positions" << std::endl;

      for (uint k=1; k < nMaxSkips; k++)
        assert(uncovered_set_(k,state) != MAX_USHORT);
      assert(lowestUncoveredPos == uncovered_set_(0,state));

      //predecessor states can only be states with less entries

      assert(nMaxSkips > 0); //otherwise this state cannot exist

      if (nConsecutiveEndSkips == nMaxSkips) {

        if (lowestUncoveredPos > 1) {
          //the current state can only be reached by inserting all skips at once
          cur_predecessor_sets.push_back(std::make_pair(0, highestUncoveredPos + 1));
        }
      }
      else {

        const uint nPrevSkips = nMaxSkips - nConsecutiveEndSkips;

        //the current state can only be reached by inserting the last nConsecutiveEndSkips skips at once
        //  and from the state that contains the remaining skips. Therefore skip_before_end_skips must be the last skip in the predecessor state
        const uint skip_before_end_skips = uncovered_set_(nMaxSkips - nConsecutiveEndSkips - 1, state);

        uint prev_candidate = 0;
        for (prev_candidate = first_set_[skip_before_end_skips]; prev_candidate < first_set_[skip_before_end_skips + 1]; prev_candidate++) {

          if (nUncoveredPositions_[prev_candidate] == nPrevSkips) {

            bool is_predecessor = true;

            for (uint k = 0; k < nPrevSkips; k++) {
              if (uncovered_set_(k + nConsecutiveEndSkips, prev_candidate) != uncovered_set_(k, state)) {
                is_predecessor = false;
                break;
              }
            }

            if (is_predecessor) {
              cur_predecessor_sets.push_back(std::make_pair(prev_candidate, highestUncoveredPos + 1));
              break;
            }
          }
        }

        assert(prev_candidate < first_set_[skip_before_end_skips + 1]);
      }
    }
    else {

      //if (state <= 10)
      //std::cerr << "skip set is not full" << std::endl;

      //the skip set is not full for this uncovered set
      //predecessor entries can be one state with less entries
      //    (where the transition to the current state introduces the final block of consecutive skips)
      // or states with more entries (where the transition to the current state fills a skipped position)

      assert(state > 0);

      const uint nPrevSkips = nUncoveredPositions - nConsecutiveEndSkips;

      //a) find the one predecessor state with less entries
      if (nUncoveredPositions == nConsecutiveEndSkips) {

        if (lowestUncoveredPos > 1) {
          //the current state can only be reached by inserting all skips at once
          cur_predecessor_sets.push_back(std::make_pair(0, highestUncoveredPos + 1));
        }
      }
      else {

        const uint skip_before_end_skips = uncovered_set_(nMaxSkips - nConsecutiveEndSkips - 1, state);

        uint prev_candidate = 0;
        for (prev_candidate = first_set_[skip_before_end_skips]; prev_candidate < first_set_[skip_before_end_skips + 1]; prev_candidate++) {

          if (nUncoveredPositions_[prev_candidate] == nPrevSkips) {

            bool is_predecessor = true;

            for (uint k = nMaxSkips - nUncoveredPositions; k < nMaxSkips - nUncoveredPositions + nPrevSkips; k++) {
              if (uncovered_set_(k + nConsecutiveEndSkips, prev_candidate) != uncovered_set_(k, state)) {
                is_predecessor = false;
                break;
              }
            }

            if (is_predecessor) {
              cur_predecessor_sets.push_back(std::make_pair(prev_candidate, highestUncoveredPos + 1));
              break;
            }
          }
        }
        assert(prev_candidate < first_set_[skip_before_end_skips + 1]);
      }

      //b) find states with exactly one more entry (transitions from these states to the current one close a skipped position)
      std::set<ushort> cur_skips;
      for (uint k=0; k < nMaxSkips; k++) {
        if (uncovered_set_(k,state) != MAX_USHORT)
          cur_skips.insert(uncovered_set_(k,state));
      }

      assert(cur_skips.size() == nUncoveredPositions);

      for (uint prev_candidate = first_set_[nUncoveredPositions+1]; prev_candidate < nSets; prev_candidate++) {

        if (nUncoveredPositions_[prev_candidate] == nUncoveredPositions+1) {

          ushort newpos = MAX_USHORT;
          uint nMismatches = 0;
          for (uint k=0; k < nMaxSkips; k++) {

            const ushort pos = uncovered_set_(k,prev_candidate);
            if (pos == MAX_USHORT)
              continue;

            if (cur_skips.find(pos) == cur_skips.end()) {
              nMismatches++;
              newpos = pos;
              if (nMismatches > 1)
                break;
            }
          }

          if (nMismatches == 1) {
            cur_predecessor_sets.push_back(std::make_pair(prev_candidate, newpos));
          }
        }
      }
    }

    const uint nCurPredecessors = cur_predecessor_sets.size();
    predecessor_sets_[state].resize(2, nCurPredecessors);
    uint k;
    for (k = 0; k < nCurPredecessors; k++) {
      predecessor_sets_[state](0, k) = cur_predecessor_sets[k].first;
      predecessor_sets_[state](1, k) = cur_predecessor_sets[k].second;
    }

    nMaxPredecessors = std::max(nMaxPredecessors, nCurPredecessors);
  }


  std::cerr << "each state has at most " << nMaxPredecessors << " predecessor states" << std::endl;

  uint nTransitions = 0;
  for (uint s = 0; s < nSets; s++)
    nTransitions += predecessor_sets_[s].yDim();

  std::cerr << nTransitions << " transitions" << std::endl;

//visualize_set_graph("stategraph.dot");
}

void IBMConstraintStates::cover(uint level)
{
  //  std::cerr << "*****cover(" << level << ")" << std::endl;

  if (level == 0) {
    next_set_idx_++;
    return;
  }

  const uint ref_set_idx = next_set_idx_;

  next_set_idx_++;              //to account for sets which are not fully filled
  assert(next_set_idx_ <= uncovered_set_.yDim());

  const uint ref_j = uncovered_set_(level, ref_set_idx);
  //std::cerr << "ref_j: " << ref_j << std::endl;
  //std::cerr << "ref_line: ";
  //   for (uint k=0; k < uncovered_set_.xDim(); k++) {

  //     if (uncovered_set_(k,ref_set_idx) == MAX_USHORT)
  //       std::cerr << "-";
  //     else
  //       std::cerr << uncovered_set_(k,ref_set_idx);
  //     std::cerr << ",";
  //   }
  //   std::cerr << std::endl;

  for (uint j = 1; j < ref_j; j++) {

    //std::cerr << "j: " << j << std::endl;

    assert(next_set_idx_ <= uncovered_set_.yDim());

    for (uint k = level; k < uncovered_set_.xDim(); k++)
      uncovered_set_(k, next_set_idx_) = uncovered_set_(k, ref_set_idx);

    uncovered_set_(level - 1, next_set_idx_) = j;

    cover(level - 1);
  }
}

void IBMConstraintStates::compute_coverage_states(uint maxJ)
{
  std::cerr << "************ compute_coverage_states ***********" << std::endl;
  std::cerr << "maxJ: " << maxJ << std::endl;
  const uint nMaxSkips = uncovered_set_.xDim();

  uint nStates = maxJ;     //states for set #0
  for (uint k = 1; k < uncovered_set_.yDim(); k++) {

    const uint highest_uncovered_pos = uncovered_set_(nMaxSkips - 1, k) - 1; //convert 1-based to 0-based
    if (highest_uncovered_pos >= maxJ)
      break;

    nStates += maxJ - 1 - highest_uncovered_pos;
  }

  coverage_state_.resize(2, nStates);

  const uint nUncoveredSets = uncovered_set_.yDim();
  first_state_.resize(maxJ + 1);

  Math2D::NamedMatrix<uint> cov_state_num(uncovered_set_.yDim(), maxJ, MAX_UINT, MAKENAME(cov_state_num));

  uint cur_state = 0;
  for (uint j = 0; j < maxJ; j++) {

    //j is the highest covered source position (j is 0-based)

    first_state_[j] = cur_state;
    coverage_state_(0, cur_state) = 0; //set 0 = no skips
    coverage_state_(1, cur_state) = j;
    cov_state_num(0, j) = cur_state;

    cur_state++;

    //go through sets with skips
    for (uint k = 1; k < first_set_[j+1]; k++) { //for the sets j is 1-based

      const uint highest_uncovered_pos = uncovered_set_(nMaxSkips - 1, k) - 1; //convert from 1-based to 0-based
      assert(highest_uncovered_pos < j);
      if (highest_uncovered_pos < j) {

        coverage_state_(0, cur_state) = k;
        coverage_state_(1, cur_state) = j;
        cov_state_num(k, j) = cur_state;
        cur_state++;
      }
    }
  }
  first_state_[maxJ] = cur_state;

  std::cerr << nStates << " coverage states" << std::endl;

  assert(cur_state == nStates);

  /*** now compute predecessor states ****/
  predecessor_coverage_states_.resize(nStates);

  for (uint state_num = 0; state_num < nStates; state_num++) {

    //if (state_num <= 10)
    //std::cerr << "############ state #" << state_num << std::endl;

    std::vector<std::pair<uint,uint> > cur_predecessor_states;

    const uint highest_covered_source_pos = coverage_state_(1, state_num);
    const uint uncovered_set_idx = coverage_state_(0, state_num);

    // if (state_num <= 10)
    // {
    // std::cerr << "highest covered source pos: " << highest_covered_source_pos << ", uncovered set:";
    // print_uncovered_set(uncovered_set_idx);
    // std::cerr << std::endl;
    // }

    if (uncovered_set_idx == 0) {

      //if (state_num <= 10)
      //std::cerr << "uncovered set is empty" << std::endl;

      //the set of uncovered positions is empty
      if (highest_covered_source_pos > 0) {     //otherwise there are no predecessor states

        //if (state_num <= 10)
        //std::cerr << "highest_covered_source_pos: " << highest_covered_source_pos << std::endl;

        //a) handle transition where the uncovered set is kept
        assert(state_num > 0);
        const uint prev_state = cov_state_num(uncovered_set_idx, highest_covered_source_pos - 1);
        assert(coverage_state_(1, prev_state) == highest_covered_source_pos - 1);
        assert(coverage_state_(0, prev_state) == uncovered_set_idx);

        cur_predecessor_states.push_back(std::make_pair(prev_state, highest_covered_source_pos));

        //b) handle transitions where the uncovered set is changed
        // the predecessor set must have exactly one uncovered position
        const Math2D::Matrix<uint>& cur_predecessor_sets = predecessor_sets_[uncovered_set_idx];
        const uint nPredecessorSets = cur_predecessor_sets.yDim();

        for (uint p = 0; p < nPredecessorSets; p++) {

          const uint covered_source_pos = cur_predecessor_sets(1, p) - 1; //convert from 1-based to 0-based

          //if (state_num <= 10)
          //std::cerr << "**** p: " << p << ", covered source pos: " << covered_source_pos << std::endl;
          if (covered_source_pos < highest_covered_source_pos) {

            const uint predecessor_set = cur_predecessor_sets(0, p);
            assert(nUncoveredPositions_[predecessor_set] == 1);

            const uint prev_idx = cov_state_num(predecessor_set, highest_covered_source_pos);
            //if (state_num <= 10)
            //std::cerr << "prev state: " << prev_idx << std::endl;
            assert(prev_idx != MAX_UINT);
            cur_predecessor_states.push_back(std::make_pair(prev_idx,covered_source_pos));
          }
        }
      }
      else
        assert(state_num == 0);
    }
    else {

      //if (state_num <= 10)
      //std::cerr << "uncovered set is not empty" << std::endl;

      const int highest_uncovered_source_pos = uncovered_set_(nMaxSkips - 1, uncovered_set_idx) - 1; //convert from 1-based to 0-based
      assert(highest_uncovered_source_pos < highest_covered_source_pos);

      //a) handle transition where the uncovered set is kept
      if (highest_covered_source_pos > highest_uncovered_source_pos + 1) {

        //if (state_num <= 10)
        //std::cerr << "transition from self" << std::endl;
        assert(state_num > 0);
        const uint prev_state = cov_state_num(uncovered_set_idx, highest_covered_source_pos - 1);

        assert(prev_state != MAX_UINT);
        assert(coverage_state_(1, prev_state) == highest_covered_source_pos - 1);
        assert(coverage_state_(0, prev_state) == uncovered_set_idx);
        cur_predecessor_states.push_back(std::make_pair(prev_state, highest_covered_source_pos));
      }

      //b) handle transitions where the uncovered set is changed

      //std::cerr << "examining state (";
      //print_uncovered_set(uncovered_set_idx);
      //std::cerr << " ; " << highest_covered_source_pos << " )" << std::endl;

      const Math2D::Matrix<uint>& cur_predecessor_sets = predecessor_sets_[uncovered_set_idx];
      const uint nPredecessorSets = cur_predecessor_sets.yDim();
      for (uint p = 0; p < nPredecessorSets; p++) {

        const int covered_source_pos = cur_predecessor_sets(1, p) - 1; //convert from 1-based to 0-based
        // if (state_num <= 10)
        // std::cerr << "transition " << p << "; covered_source_pos: " << covered_source_pos << ", highest_covered_source_pos: "
        // << highest_covered_source_pos << ", set num " << cur_predecessor_sets(0, p) << std::endl;
        if (covered_source_pos <= highest_covered_source_pos) {
          const uint predecessor_set = cur_predecessor_sets(0, p);

          // if (state_num <= 10) {
          // std::cerr << "predecessor set ";
          // print_uncovered_set(predecessor_set);
          // std::cerr << std::endl;
          // }

          if (nUncoveredPositions_[predecessor_set] > nUncoveredPositions_[uncovered_set_idx]) {

            //if (state_num <= 10) {
            //std::cerr << "transition reduces the number of skips" << std::endl;
            //}

            //transition from previous state to current fills a skipped position (this means the pos following the largest skip is already filled)
            if (uncovered_set_(nMaxSkips-1,predecessor_set) - 1 < highest_covered_source_pos) { //convert from 1-based to 0-based

              //find the index of the predecessor state
              const uint prev_idx = cov_state_num(predecessor_set, highest_covered_source_pos);

              assert(prev_idx != MAX_UINT);
              assert(prev_idx < first_state_[highest_covered_source_pos + 1]);
              assert(coverage_state_(1, prev_idx) == highest_covered_source_pos);

              cur_predecessor_states.push_back(std::make_pair(prev_idx, covered_source_pos));
            }
          }
          else if (covered_source_pos == highest_covered_source_pos) {

            //if (state_num <= 10) {
            //std::cerr << "covered source pos is last" << std::endl;
            //}

            //transition from previous state to current introduces consecutive skips

            assert(nUncoveredPositions_[predecessor_set] < nUncoveredPositions_[uncovered_set_idx]);
            uint nNewSkips = nUncoveredPositions_[uncovered_set_idx] - nUncoveredPositions_[predecessor_set];
            int first_skipped_j = uncovered_set_(nMaxSkips-nNewSkips,uncovered_set_idx) - 1; //convert from 1-based to 0-based
            int prev_highest_covered_source_pos = first_skipped_j-1;

            //if (state_num <= 10) {
            //std::cerr << "prev_highest_covered_source_pos: " << prev_highest_covered_source_pos << std::endl;
            //}

            if (prev_highest_covered_source_pos >= 0) {

              // if (state_num <= 10) {
              // std::cerr << "examining state (" << uncovered_set_idx << "=";
              // print_uncovered_set(uncovered_set_idx);
              // std::cerr << " ; " << highest_covered_source_pos << " )" << std::endl;
              // std::cerr << "set has " << nPredecessorSets << " predecessors, currently at " << p << std::endl;

              // std::cerr << "predecessor set (" << predecessor_set << "=";
              // print_uncovered_set(predecessor_set);
              // std::cerr << ")" << std::endl;

              // std::cerr << "ph: " << prev_highest_covered_source_pos << std::endl;
              // }

              //find the index of the predecessor state
              const uint prev_idx = cov_state_num(predecessor_set, prev_highest_covered_source_pos);
              //std::cerr << "introducing a transition from state #" << prev_idx << std::endl;

              assert(prev_idx != MAX_UINT);
              assert(prev_idx < first_state_[prev_highest_covered_source_pos + 1]);
              assert(coverage_state_(1, prev_idx) == prev_highest_covered_source_pos);

              cur_predecessor_states.push_back(std::make_pair(prev_idx, covered_source_pos));
            }
          }
        }
      }
    }

    /*** copy cur_predecessor_states to predecessor_covered_sets_[state_num] ***/
    predecessor_coverage_states_[state_num].resize(2, cur_predecessor_states.size());
    for (uint k = 0; k < cur_predecessor_states.size(); k++) {
      predecessor_coverage_states_[state_num](0, k) = cur_predecessor_states[k].first;
      predecessor_coverage_states_[state_num](1, k) = cur_predecessor_states[k].second;
    }
  }

  //compute start states
  start_states_.clear();
  for (uint state = 0; state < first_state_[nMaxSkips+1]; state++) {

    //std::cerr << "state: " << state << std::endl;

    const uint set_idx = coverage_state_(0, state);
    const uint max_covered_j = coverage_state_(1, state);
    const uint nCurUncoveredPositions = nUncoveredPositions_[set_idx];

    if (nCurUncoveredPositions == max_covered_j) {

      std::cerr << "allowing start state " << state << " with set ";
      print_uncovered_set(set_idx);
      std::cerr << " and max_covered_j " << max_covered_j << std::endl;

      start_states_.insert(state);
      assert(predecessor_coverage_states_.size() > state);
      assert(predecessor_coverage_states_[state].size() == 0);
    }
  }
}

/************* auxiliary routines ************/

uint IBMConstraintStates::nUncoveredPositions(uint state) const
{

  uint result = uncovered_set_.xDim();

  for (uint k = 0; k < uncovered_set_.xDim(); k++) {
    if (uncovered_set_(k, state) == MAX_USHORT)
      result--;
    else
      break;
  }

  return result;
}

void IBMConstraintStates::print_uncovered_set(uint setnum) const
{
  for (uint k = 0; k < uncovered_set_.xDim(); k++) {

    if (uncovered_set_(k, setnum) == MAX_USHORT)
      std::cerr << "-";
    else
      std::cerr << uncovered_set_(k, setnum);
    if (k+1 < uncovered_set_.xDim())
      std::cerr << ",";
  }
}

void IBMConstraintStates::print_uncovered_set(uint* data, uint nMaxSkips) const
{
  for (uint k = 0; k < uncovered_set_.xDim(); k++) {

    if (data[k] == MAX_USHORT)
      std::cerr << "-";
    else
      std::cerr << data[k];
    if (k+1 < nMaxSkips)
      std::cerr << ",";
  }
}

void IBMConstraintStates::visualize_set_graph(std::string filename)
{

  std::ofstream dotstream(filename.c_str());

  dotstream << "digraph corpus {" << std::endl
            << "node [fontsize=\"6\",height=\".1\",width=\".1\"];" << std::endl;
  dotstream << "ratio=compress" << std::endl;
  dotstream << "page=\"8.5,11\"" << std::endl;

  for (uint state = 0; state < uncovered_set_.yDim(); state++) {

    dotstream << "state" << state << " [shape=record,label=\"";
    for (uint k = 0; k < uncovered_set_.xDim(); k++) {

      if (uncovered_set_(k, state) == MAX_USHORT)
        dotstream << "-";
      else
        dotstream << uncovered_set_(k, state);

      if (k + 1 < uncovered_set_.xDim())
        dotstream << "|";
    }
    dotstream << "\"]" << std::endl;
  }

  for (uint state = 0; state < uncovered_set_.yDim(); state++) {

    for (uint k = 0; k < predecessor_sets_[state].yDim(); k++)
      dotstream << "state" << predecessor_sets_[state] (0, k) << " -> state" << state << std::endl;
  }

  dotstream << "}" << std::endl;
  dotstream.close();
}

