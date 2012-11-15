/*
 * distribution.C
 *
 *  Created on: Nov 1, 2012
 *      Author: alfoa
 */
#include "distribution.h"
using namespace std;

template<>
InputParameters validParams<distribution>(){

  InputParameters params = validParams<RavenObject>();

   params.addRequiredParam<double>("xMin", "Minimum coordinate");
   params.addRequiredParam<double>("xMax", "Max coordinate");
   params.addParam<unsigned int>("seed", _defaultSeed ,"RNG seed");
   params.addRequiredParam<distribution_type>("type","distribution type");
   params.addPrivateParam<std::string>("built_by_action", "add_distribution");
   return params;
}


class distribution;

//distribution::distribution(){
//   _type = NORMAL_DISTRIBUTION;
//   _dis_parameters["xMin"] = 0.0;
//   _dis_parameters["xMax"] = 1.0;
//   _seed=_defaultSeed;
//}
//distribution::distribution(double xMin, double xMax, distribution_type type, unsigned int seed){
//   _type = type;
//   _dis_parameters["xMin"] = xMin;
//   _dis_parameters["xMax"] = xMax;
//   _seed = seed;
//}
distribution::distribution(const std::string & name, InputParameters parameters):
      RavenObject(name,parameters)
{
   _type=getParam<distribution_type>("type");
   if(_type != CUSTOM_DISTRIBUTION){
      _dis_parameters["xMin"] = getParam<distribution_type>("xMin");
      _dis_parameters["xMax"] = getParam<distribution_type>("xMax");
   }
   else
   {
      std::vector<double> x_coordinates = getParam<std::vector<double> >("x_coordinates");
     _dis_parameters["xMin"] = x_coordinates[0];
     _dis_parameters["xMax"] = x_coordinates[x_coordinates.size()-1];
     std::vector<double> y_cordinates = getParam<std::vector<double> >("y_coordinates");
     int numberPoints = getParam<int>("numberPoints");
     custom_dist_fit_type fitting_type = getParam<custom_dist_fit_type>("fitting_type");

     _interpolation=Interpolation_Functions(x_coordinates,
                                            y_cordinates,
                                            numberPoints,
                                            fitting_type);
   }
      _seed = getParam<unsigned int>("seed");
}
//distribution::distribution(std::vector<double> x_coordinates, std::vector<double> y_coordinates, int numberPoints, custom_dist_fit_type fitting_type, unsigned int seed){
//   _type = CUSTOM_DISTRIBUTION;
//   _dis_parameters["xMin"] = x_coordinates[0];
//   _dis_parameters["xMax"] = x_coordinates[x_coordinates.size()-1];
//   _seed = seed;
//   _interpolation=Interpolation_Functions(x_coordinates, y_coordinates, numberPoints, fitting_type);
//}

distribution::~distribution(){
}

double
distribution::getVariable(std::string & variableName){

   if(_dis_parameters.find(variableName) != _dis_parameters.end()){
     return _dis_parameters.find(variableName) ->second;
   }
   else{
     mooseError("Parameter " << variableName << " was not found in distribution type " << _type <<".");
     return -1;
   }
}
void
distribution::updateVariable(std::string & variableName, double & newValue){
   if(_dis_parameters.find(variableName) != _dis_parameters.end()){
     // we are sure the variableName is already present in the mapping =>
     // we can update it in the following way
     _dis_parameters[variableName] = newValue;
   }
   else{
     mooseError("Parameter " << variableName << " was not found in distribution type " << _type << ".");
   }
}
distribution_type distribution::getType(){
   return _type;
}

double getDistributionVariable(distribution & dist, std::string & variableName){
  return dist.getVariable(variableName);
}


void DistributionUpdateVariable(distribution & dist, std::string & variableName, double & newValue){
  dist.updateVariable(variableName, newValue);
}

double DistributionPdf(distribution & dist, double & x){
  return dist.Pdf(x);
}

double DistributionCdf(distribution & dist, double & x){
  return dist.Cdf(x);
}

double DistributionRandomNumberGenerator(distribution & dist){
  return dist.RandomNumberGenerator();
}

distribution_type getDistributionType(distribution & dist) {
  return dist.getType();
}
