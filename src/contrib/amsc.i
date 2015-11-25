%module amsc
%include "std_vector.i"
%include "std_string.i"
%include "std_map.i"
%include "std_set.i"
%include "std_pair.i"
%include "stl.i"

%{
#include "AMSC/AMSC.h"
%}
%include "AMSC/AMSC.h"

%template(AMSCFloat) AMSC<float>;
%template(AMSCDouble) AMSC<double>;

namespace std
{
  %template(vectorFloat) vector<float>;
  %template(vectorDouble) vector<double>;
  %template(vectorString) vector<string>;
  %template(vectorInt) vector<int>;
  %template(setInt) set<int>;
  %template(mapPartition) map< string, vector<int> >;
  %template(mapManifolds) map< int, vector<int> >;
}
