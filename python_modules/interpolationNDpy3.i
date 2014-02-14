%module interpolationNDpy3
%{
#include "ND_Interpolation_Functions.h"
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
%}
%include "std_vector.i"
%include "../control_modules/numpy.i"
%include "ND_Interpolation_Functions.h"

namespace std {
   %template(vectd) vector<double>;
   %template(vecti) vector<int>;
};

%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* seq, int n)};
%apply (double IN_ARRAY1[ANY]) {(double seq)};
%apply (double IN_ARRAY2[ANY][ANY]) {(double seq)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* seq, int n, int m)};

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* seq, int n)};
%apply (double INPLACE_ARRAY1[ANY]) {(double seq)};
%apply (double INPLACE_ARRAY2[ANY][ANY]) {(double seq)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* seq, int n, int m)};