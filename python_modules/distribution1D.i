%module distribution1D
%{
#include "DistributionContainer.h"
#include "distribution_1D.h"
%}
#%ignore DistributionContainer;
#%ignore NormalDistribution;
#%ignore UniformDistribution;
#%ignore validParams;
#%ignore InputParameters;
#%ignore ~InputParameters;
#%ignore RavenObject;
#include "InputParameters.h"
#include "RavenObject.h"
#include "distribution.h"


#%include "InputParameters.h"
#%include "RavenObject.h"
#%include "distribution.h"
#%include "distribution_1D.h"
#%include "DistributionContainer.h"
#%include "Interpolation_Functions.h"
%nodefault;

class DistributionContainer{
    public:
    static DistributionContainer & Instance();
    void seedRandom(unsigned int seed);
    distribution_type getType (std::string DistAlias);
    double getVariable(std::string paramName,std::string DistAlias);
    void updateVariable(std::string paramName,double newValue,std::string DistAlias);
    double Pdf(std::string DistAlias, double x);     
    double Cdf(std::string DistAlias, double x);     
    double randGen(std::string DistAlias);       
    double randGen(char * DistAlias);       
    double random(); 
    bool checkCdf(double probability, std::vector<double> values);
    bool checkCdf(double probability, double value);
};


 /*
swig -c++ -python -py3 -Iinclude/distributions/ -I../moose/include/utils/ python_modules/distribution1D.i 
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

swig -c++ -python -py3 -Iinclude/tools python_modules/raventools.i
g++ -fPIC -c src/tools/*.C python_modules/raventools_wrap.cxx -Iinclude/tools/ -I/usr/include/python3.2mu/ -Iinclude/utilities
g++ -shared *.o -o python_modules/_raventools.so

  */
