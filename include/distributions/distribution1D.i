%module distribution1D
%{
#include "distribution_1D.h"
%}

%include "distribution_1D.h"


 /*
swig -c++ -python -py3 include/distributions/distribution1D.i
g++ -fPIC -c src/distributions/*.C include/distributions/distribution1D_wrap.cxx -Iinclude/distributions/ -I/usr/include/python3.2mu/
g++ -shared *.o -o _distribution1D.so
cp include/distributions/distribution1D.py .
PYTHONPATH=. python3
rm -f *.o *.so distribution1D.py
  */
