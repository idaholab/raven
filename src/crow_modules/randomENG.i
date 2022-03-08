%module randomENG
%{
#include "randomClass.h"
%}
%include "std_vector.i"
%include "randomClass.h"

namespace std {
   %template(vectd) vector<double>;
   %template(vectd2d) vector< vector<double> >;
   %template(vecti) vector<int>;
   %template(vecti2d) vector< vector<int> >;
};

