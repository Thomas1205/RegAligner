/*** written by Thomas Schoenemann as a private person without employment, September 2009 ***/

#include "makros.hh"
#include <map>

namespace Makros {

  std::map<std::string,std::string> typename_map;

  void register_typename(const std::string& id, const std::string& fullname)
  {
    typename_map[id] = fullname;
  }

  std::string get_typename(const std::string& id)
  {

    std::map<std::string,std::string>::iterator it = typename_map.find(id);
    if (it == typename_map.end())
      return id;
    else
      return it->second;
  }
}
