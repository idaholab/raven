%module distribution1D
%{
#include "distribution_1D.h"
#include "DistributionContainer.h"
#include "Interpolation_Functions.h"
%}

%include "distribution_1D.h"
%include "DistributionContainer.h"
%include "Interpolation_Functions.h"

 /*
swig -c++ -python -py3 -Iinclude/distributions/ python_modules/distribution1D.i 
g++ -fPIC -c src/distributions/*.C python_modules/distribution1D_wrap.cxx -Iinclude/distributions/ -I/usr/include/python3.2mu/
g++ -shared *.o -o python_modules/_distribution1D.so
PYTHONPATH=python_modules/ python3
import distribution1D
test1 = distribution1D.distribution_1D(1, -3.0, 2.0,  1.0, 1.0)
test1.randGen()

distcont = distribution1D.DistributionContainer.Instance()
distcont.constructDistributionContainer(distribution1D.str_to_string_p("a_dist"),distribution1D.NORMAL_DISTRIBUTION,-1.0,1.0,0.0,1.0)
distcont.randGen(distribution1D.str_to_string_p("a_dist"))

#rm -f *.o *.so distribution1D.py
  */
