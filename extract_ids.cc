/*** written by Thomas Schoenemann as a private person without employment, November 2009 ***/

#include "makros.hh"
#include "application.hh"
#include "stringprocessing.hh"
#include <fstream>

int main(int argc, char** argv)
{

  const int nParams = 2;
  ParamDescr  params[nParams] = {{"-i",mandInFilename,0,""},{"-o",mandOutFilename,0,""}};


  Application app(argc,argv,params,nParams);

  std::ifstream in(app.getParam("-i").c_str());
  std::ofstream out(app.getParam("-o").c_str());

  char cline[65536];

  std::string line;

  while(in.getline(cline,65536)) {

    line = cline;

    size_t idpos = line.find("ID=\"");
    if (idpos > line.size())
      idpos = line.find("id=\"");
    if (idpos > line.size())
      idpos = line.find("id = \"");
    if (idpos > line.size())
      idpos = line.find("id =\"");
    if (idpos > line.size())
      idpos = line.find("id= \"");
    if (idpos > line.size())
      idpos = line.find("ID = \"");
    if (idpos > line.size())
      idpos = line.find("ID =\"");
    if (idpos > line.size())
      idpos = line.find("ID= \"");

    if (idpos < line.size()) {
      uint num_start = idpos+3;
      while(num_start < line.size() && line[num_start] != '\"')
        num_start++;
      num_start++;

      std::string num;
      for (uint i=num_start; i < line.size() && line[i] != '\"'; i++)
        num += line[i];
      out << num << std::endl;
    }
  }

  out.close();
}
