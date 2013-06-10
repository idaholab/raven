%module raventools
%{
#include "RavenToolsContainer.h"
%}
%include "RavenToolsContainer.h"


/*
IT IS AN EXAMPLEEEEEEEEEEEE
IT IS COMMENTEDDDDDDDD
swig -c++ -python -py3 -Iinclude/tools/ -I../moose/include/utils/ python_modules/raventools.i 
g++ -fPIC -c src/tools/*.C python_modules/raventools_wrap.cxx -Iinclude/tools/ -I/usr/include/python3.2mu/
g++ -shared *.o -o python_modules/_raventools.so
PYTHONPATH=python_modules/ python3

#rm -f *.o *.so raventools.py

swig -c++ -python -py3 -Iinclude/tools python_modules/raventools.i
g++ -fPIC -c src/tools/*.C python_modules/raventools_wrap.cxx -Iinclude/tools/ -I/usr/include/python3.2mu/ -Iinclude/utilities -Iinclude/base
g++ -shared *.o -o python_modules/_raventools.so

*/


