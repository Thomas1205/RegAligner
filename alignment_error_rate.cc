/*** written by Thomas Schoenemann as a private person without employment, November 2009 ***/

#include "makros.hh"
#include "stringprocessing.hh"
#include "alignment_error_rate.hh"
#include <fstream>
#include "fileio.hh"

#ifdef HAS_GZSTREAM
#include "gzstream.h"
#endif

void read_reference_alignment_compact(std::string filename, std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_alignments,
                                      std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_alignments, bool invert)
{

  std::istream* astream;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(filename)) {
    igzstream* new_stream= new igzstream;
    new_stream->open(filename.c_str());
    astream = new_stream;
  }
  else {
    astream = new std::ifstream(filename.c_str());
  }
#else
  astream = new std::ifstream(filename.c_str());
#endif

  char cline[65536];
  std::string line;
  std::vector<std::string> tokens;
  std::vector<std::string> mini_tokens;

  uint nLines = 1;

  while (astream->getline(cline,65536)) {
    //std::cerr << "line: " << line << std::endl;

    line = cline;
    tokenize(line,tokens,' ');

    //std::cerr << tokens.size() << " tokens" << std::endl;

    if (tokens.size() == 0 || tokens[0].size() == 0 || tokens[0][0] != '#') {
      std::cerr << "WARNING: no line number given in line " << nLines << ". line is ignored" << std::endl;
    }
    else {

      uint cur_line_num = convert<uint>(tokens[0].substr(1,tokens[0].size()-1));

      for (uint i=1; i < tokens.size(); i++) {
        tokenize(tokens[i],mini_tokens,'-');
        if (mini_tokens.size() != 2) {
          std::cerr << "WARNING: ignoring invalid entry \"" << tokens[i] << "\" in line " << nLines << std::endl;
        }

        bool sure = (mini_tokens[0][0] != 'P' && mini_tokens[0][0] != 'p');

        ushort source = (sure) ? convert<ushort>(mini_tokens[0]) :
                        convert<ushort>(mini_tokens[0].substr(1,mini_tokens[0].size()));
        ushort target = convert<ushort>(mini_tokens[1]);

        std::pair<ushort,ushort> new_alignment;

        if (!invert)
          new_alignment = std::make_pair(source,target);
        else
          new_alignment = std::make_pair(target,source);

        if (sure)
          sure_alignments[cur_line_num].insert(new_alignment);

        //sure alignments are also possible alignments
        possible_alignments[cur_line_num].insert(new_alignment);
      }
    }

    nLines++;
  }

  delete astream;
}

void read_reference_alignment_standard(std::string filename,
                                       std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_alignments,
                                       std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_alignments,
                                       bool invert)
{


  std::istream* astream;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(filename)) {
    igzstream* new_stream= new igzstream;
    new_stream->open(filename.c_str());
    astream = new_stream;
  }
  else {
    astream = new std::ifstream(filename.c_str());
  }
#else
  astream = new std::ifstream(filename.c_str());
#endif

  char cline[65536];
  std::string line;
  std::vector<std::string> tokens;


  while (astream->getline(cline,65536)) {
    //std::cerr << "line: " << line << std::endl;

    line = cline;
    tokenize(line,tokens,' ');

    //std::cerr << tokens.size() << " tokens" << std::endl;

    if (tokens.size() < 3 || tokens.size() > 4) {
      //skip this line
    }
    else {

      uint anum = convert<uint>(tokens[0]);
      uint first = convert<uint>(tokens[1]);
      uint second = convert<uint>(tokens[2]);
      bool sure = (tokens.size() < 3 || tokens[3] == "S" || tokens[3] == "s");

      std::pair<ushort,ushort> new_alignment;

      if (!invert)
        new_alignment = std::make_pair(first,second);
      else
        new_alignment = std::make_pair(second,first);

      if (sure)
        sure_alignments[anum].insert(new_alignment);

      //sure alignments are also possible alignments
      possible_alignments[anum].insert(new_alignment);
    }

  }

}

void read_reference_alignment(std::string filename,
                              std::map<uint,std::set<std::pair<ushort,ushort> > >& sure_alignments,
                              std::map<uint,std::set<std::pair<ushort,ushort> > >& possible_alignments,
                              bool invert)
{

  std::istream* astream;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(filename)) {
    igzstream* new_stream= new igzstream;
    new_stream->open(filename.c_str());
    astream = new_stream;
  }
  else {
    astream = new std::ifstream(filename.c_str());
  }
#else
  astream = new std::ifstream(filename.c_str());
#endif

  std::string test;
  (*astream) >> test;
#ifdef HAS_GZSTREAM
  if (is_gzip_file(filename))
    ((igzstream*) astream)->close();
  else
    ((std::ifstream*) astream)->close();
#else
  ((std::ifstream*) astream)->close();
#endif

  if (test.size() == 0 || (test[0] != '#' && !is_natural_number(test)) ) {
    USER_ERROR << "failed to read reference alignments. Please check the file. Exiting." << std::endl;
    exit(1);
  }

  bool compact = (test[0] == '#');
  if (compact)
    read_reference_alignment_compact(filename,sure_alignments,possible_alignments,invert);
  else {
    read_reference_alignment_standard(filename,sure_alignments,possible_alignments,invert);
  }
}

bool reference_consistent(const std::map<uint, std::set<std::pair<ushort,ushort> > >& possible_alignments,
                          const Storage1D<Math1D::Vector<uint> >& source, const Storage1D<Math1D::Vector<uint> >& target)
{

  for (std::map<uint, std::set<std::pair<ushort,ushort> > >::const_iterator it1 = possible_alignments.begin();
       it1 != possible_alignments.end(); it1++) {

    uint s = it1->first-1;

    if (s >= source.size())
      break;

    ushort max_s = 0;
    ushort max_t = 0;

    for (std::set<std::pair<ushort,ushort> >::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
      max_s = std::max(max_s,it2->first);
      max_t = std::max(max_t,it2->second);
    }

    if (max_s > source[s].size() || max_t > target[s].size()) {
      IO_ERROR << "reference error for sentence " << it1->first << ": max source: " << max_s << ", J: " << source[s].size()
               << "; max target: " << max_t << ", I: " << target[s].size() << std::endl;
      return false;
    }
  }

  return true;
}

void write_reference_alignment(std::string filename,
                               std::map<uint, std::set<std::pair<ushort,ushort> > >& sure_alignments,
                               std::map<uint, std::set<std::pair<ushort,ushort> > >& possible_alignments,
                               bool invert)
{

  std::ofstream astream(filename.c_str());

  for (std::map<uint, std::set<std::pair<ushort,ushort> > >::const_iterator it = possible_alignments.begin();
       it != possible_alignments.end(); it++) {

    uint sent_num = it->first;

    for (std::set<std::pair<ushort,ushort> >::const_iterator it2 = it->second.begin(); it2 != it->second.end(); it2++) {

      astream << sent_num << " ";
      if (invert)
        astream << it2->second  << " " << it2->first << " ";
      else
        astream << it2->first  << " " << it2->second << " ";

      if (sure_alignments[sent_num].find(*it2) == sure_alignments[sent_num].end())
        astream << "P";
      else
        astream << "S";

      astream << std::endl;
    }
  }
}



double AER(const Storage1D<ushort>& singleword_alignment,
           const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
           const std::set<std::pair<ushort,ushort> >& possible_ref_alignments)
{

  std::set<std::pair<ushort,ushort> > given_alignment;
  for (uint j=0; j < singleword_alignment.size(); j++) {

    if (singleword_alignment[j] > 0) {
      given_alignment.insert(std::make_pair(j+1, singleword_alignment[j]));
    }
  }

  return AER(given_alignment, sure_ref_alignments, possible_ref_alignments);
}

double AER(const std::set<std::pair<ushort,ushort> >& mt_alignment,
           const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
           const std::set<std::pair<ushort,ushort> >& possible_ref_alignments)
{


  uint A = mt_alignment.size();
  uint S = sure_ref_alignments.size();

  uint denom = A+S;
  uint num = 0;

  if (denom == 0) {
    std::cerr << "WARNING: denominator zero for computation of AER" << std::endl;
    return 0.0;
  }

  for (std::set<std::pair<ushort,ushort> >::const_iterator mt_it = mt_alignment.begin();
       mt_it != mt_alignment.end(); mt_it++) {

    if (possible_ref_alignments.find(*mt_it) != possible_ref_alignments.end())
      num++;
    if (sure_ref_alignments.find(*mt_it) != sure_ref_alignments.end())
      num++;
  }

  return 1.0 - ( ((double) num) / ((double) denom) );
}


//f-measure as in [Fraser and Marcu 07] for a single sentence pair
// assuming a single-word alignment
double f_measure(const Storage1D<ushort>& singleword_alignment,
                 const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                 const std::set<std::pair<ushort,ushort> >& possible_ref_alignments,
                 double alpha)
{


  std::set<std::pair<ushort,ushort> > given_alignment;
  for (uint j=0; j < singleword_alignment.size(); j++) {

    if (singleword_alignment[j] > 0) {
      given_alignment.insert(std::make_pair(j+1, singleword_alignment[j]));
    }
  }

  return f_measure(given_alignment, sure_ref_alignments, possible_ref_alignments, alpha);
}

//f-measure as in [Fraser and Marcu 2003] for a single sentence pair
// assuming a general alignment
double f_measure(const std::set<std::pair<ushort,ushort> >& mt_alignment,
                 const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                 const std::set<std::pair<ushort,ushort> >& possible_ref_alignments,
                 double alpha)
{


  uint A = mt_alignment.size();
  uint S = sure_ref_alignments.size();

  double precision = 0.0;
  double recall = 0.0;

  for (std::set<std::pair<ushort,ushort> >::const_iterator mt_it = mt_alignment.begin();
       mt_it != mt_alignment.end(); mt_it++) {

    if (possible_ref_alignments.find(*mt_it) != possible_ref_alignments.end())
      precision++;
    if (sure_ref_alignments.find(*mt_it) != sure_ref_alignments.end())
      recall++;
  }

  precision /= A;
  recall /= S;

  if (precision == 0.0 || recall == 0.0)
    return 0.0;

  double sum = alpha / precision;
  sum += (1-alpha) / recall;

  return 1.0 / sum;
}


double precision(const Storage1D<ushort>& singleword_alignment,
                 const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                 const std::set<std::pair<ushort,ushort> >& possible_ref_alignments)
{

  std::set<std::pair<ushort,ushort> > given_alignment;
  for (uint j=0; j < singleword_alignment.size(); j++) {

    if (singleword_alignment[j] > 0) {
      given_alignment.insert(std::make_pair(j+1, singleword_alignment[j]));
    }
  }

  return precision(given_alignment, sure_ref_alignments, possible_ref_alignments);
}

double precision(const std::set<std::pair<ushort,ushort> >& mt_alignment,
                 const std::set<std::pair<ushort,ushort> >& /*sure_ref_alignments*/,
                 const std::set<std::pair<ushort,ushort> >& possible_ref_alignments)
{


  uint A = mt_alignment.size();

  double precision = 0.0;

  for (std::set<std::pair<ushort,ushort> >::const_iterator mt_it = mt_alignment.begin();
       mt_it != mt_alignment.end(); mt_it++) {

    if (possible_ref_alignments.find(*mt_it) != possible_ref_alignments.end())
      precision++;
  }

  if (A > 0)
    precision /= A;
  else
    precision = 0.0;

  return precision;
}

double recall(const Storage1D<ushort>& singleword_alignment,
              const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
              const std::set<std::pair<ushort,ushort> >& possible_ref_alignments)
{

  std::set<std::pair<ushort,ushort> > given_alignment;
  for (uint j=0; j < singleword_alignment.size(); j++) {

    if (singleword_alignment[j] > 0) {
      given_alignment.insert(std::make_pair(j+1, singleword_alignment[j]));
    }
  }

  return recall(given_alignment, sure_ref_alignments, possible_ref_alignments);
}

double recall(const std::set<std::pair<ushort,ushort> >& mt_alignment,
              const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
              const std::set<std::pair<ushort,ushort> >& /*possible_ref_alignments*/)
{

  uint S = sure_ref_alignments.size();

  double recall = 0.0;

  for (std::set<std::pair<ushort,ushort> >::const_iterator mt_it = mt_alignment.begin();
       mt_it != mt_alignment.end(); mt_it++) {

    if (sure_ref_alignments.find(*mt_it) != sure_ref_alignments.end())
      recall++;
  }

  if (S > 0)
    recall /= S;
  else
    recall = 0.0;

  return recall;
}



// number of definite alignment errors (missing sure, present but impossible links) for a single sentence pair
// assuming a single-word alignment
uint nDefiniteAlignmentErrors(const Storage1D<ushort>& singleword_alignment,
                              const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                              const std::set<std::pair<ushort,ushort> >& possible_ref_alignments)
{

  std::set<std::pair<ushort,ushort> > given_alignment;
  for (uint j=0; j < singleword_alignment.size(); j++) {

    if (singleword_alignment[j] > 0) {
      given_alignment.insert(std::make_pair(j+1, singleword_alignment[j]));
    }
  }

  return nDefiniteAlignmentErrors(given_alignment, sure_ref_alignments, possible_ref_alignments);
}

// number of definite alignment errors (missing sure, present but impossible links) for a single sentence pair
// assuming a general alignment
uint nDefiniteAlignmentErrors(const std::set<std::pair<ushort,ushort> >& mt_alignment,
                              const std::set<std::pair<ushort,ushort> >& sure_ref_alignments,
                              const std::set<std::pair<ushort,ushort> >& possible_ref_alignments)
{

  uint nErrors = 0;

  for (std::set<std::pair<ushort,ushort> >::const_iterator it = sure_ref_alignments.begin();
       it != sure_ref_alignments.end(); it++) {

    if (mt_alignment.find(*it) == mt_alignment.end())
      nErrors++;
  }

  for (std::set<std::pair<ushort,ushort> >::const_iterator it = mt_alignment.begin(); it != mt_alignment.end(); it++) {

    if (possible_ref_alignments.find(*it) == possible_ref_alignments.end())
      nErrors++;
  }

  return nErrors;
}



