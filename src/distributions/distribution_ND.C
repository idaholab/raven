/*
 * distribution_ND.C
 *
 *  Created on: Feb 6, 2014
 *      Author: alfoa
 *
 */

#include "distribution_ND.h"
//#include "Interpolation_Functions.h"
#include <string>
#include <list>

#define throwError(msg) { std::cerr << "\n\n" << msg << "\n\n"; throw std::runtime_error("Error"); }
//#include "distribution_ND_params.h"


template<>
InputParameters validParams<distributionND>()
{
  InputParameters params = validParams<RavenObject>();
  params.addParam<double>("ProbabilityThreshold", 1.0, "Probability Threshold");
  params.addRequiredParam<std::string>("type","distribution type");
  params.addRequiredParam<std::string>("data_filename","Name of the file containing the data points to be interpolated");
  params.addRequiredParam<PbFunctionType>("function_type","PDF or CDF");
  params.registerBase("distributionND");
  return params;
}

class distributionND;

distributionND::distributionND(const std::string & name, InputParameters parameters):
      RavenObject(name,parameters)
{
   _type          = getParam<std::string>("type");
   _data_filename = getParam<std::string>("data_filename");
   _function_type = getParam<PbFunctionType>("function_type");
   _dis_parameters["ProbabilityThreshold"]  = getParam<double>("ProbabilityThreshold");
   _checkStatus   = false;
}

distributionND::~distributionND(){
}

/*
 * CLASS ND DISTRIBUTION InverseWeight
 */

template<>
InputParameters validParams<MultiDimensionalInverseWeight>(){

   InputParameters params = validParams<distributionND>();
   params.addRequiredParam<double>("p", "Minkowski distance parameter");
   return params;

}

MultiDimensionalInverseWeight::MultiDimensionalInverseWeight(const std::string & name, InputParameters parameters):
    distributionND(name,parameters),
    _interpolator(inverseDistanceWeigthing(_data_filename,getParam<double>("p")))
{
  //_interpolator = dynamic_cast<ND_Interpolation *>(inverseDistanceWeigthing(_data_filename,getParam<double>("p")));
  //_interpolator = inverseDistanceWeigthing(_data_filename,getParam<double>("p"));
}

MultiDimensionalInverseWeight::~MultiDimensionalInverseWeight()
{
}

/*
 * CLASS ND DISTRIBUTION MultiDimensionalScatteredMS
 */

template<>
InputParameters validParams<MultiDimensionalScatteredMS>(){

   InputParameters params = validParams<distributionND>();
   params.addRequiredParam<double>("p", "Minkowski distance parameter");
   params.addRequiredParam<int>("precision", " ");
   return params;

}

MultiDimensionalScatteredMS::MultiDimensionalScatteredMS(const std::string & name, InputParameters parameters):
    distributionND(name,parameters),
    _interpolator(microSphere(_data_filename,getParam<double>("p"),getParam<int>("precision")))
{
  //_interpolator = dynamic_cast<ND_Interpolation *>(microSphere(_data_filename,getParam<double>("p"),getParam<int>("precision")));
  //_interpolator = microSphere(_data_filename,getParam<double>("p"),getParam<int>("precision"));
}

MultiDimensionalScatteredMS::~MultiDimensionalScatteredMS()
{
}

/*
 * CLASS ND DISTRIBUTION MultiDimensionalCartesianSpline
 */

template<>
InputParameters validParams<MultiDimensionalCartesianSpline>(){

   InputParameters params = validParams<distributionND>();
   return params;

}

MultiDimensionalCartesianSpline::MultiDimensionalCartesianSpline(const std::string & name, InputParameters parameters):
    distributionND(name,parameters)
{
  //_interpolator = dynamic_cast<ND_Interpolation *>(NDspline(_data_filename));
  _interpolator = NDspline(_data_filename);
}

MultiDimensionalCartesianSpline::~MultiDimensionalCartesianSpline()
{
}
