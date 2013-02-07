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

   params.addParam<double>("xMin", -numeric_limits<double>::max( ),"Lower bound");
   params.addParam<double>("xMax", numeric_limits<double>::max( ),"Upper bound");
   params.addParam<unsigned int>("seed", _defaultSeed ,"RNG seed");
   params.addRequiredParam<std::string>("type","distribution type");
   params.addParam<std::string>("truncation", "Type of truncation"); // Truncation types: 1) pdf_prime(x) = pdf(x)*c   2) [to do] pdf_prime(x) = pdf(x)+c
   params.addPrivateParam<std::string>("built_by_action", "add_distribution");
   params.addParam<unsigned int>("force_distribution", 0 ,"force distribution to be evaluated at: if (0) Don't force distribution, (1) xMin, (2) Mean, (3) xMax");
   return params;
}


class distribution;

distribution::distribution(const std::string & name, InputParameters parameters):
      RavenObject(name,parameters)
{
   _type=getParam<std::string>("type");
   if(_type != "CustomDistribution"){
      _dis_parameters["xMin"] = getParam<double>("xMin");
      _dis_parameters["xMax"] = getParam<double>("xMax");
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
      _force_dist = getParam<unsigned int>("force_distribution");
}

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

std::string &
distribution::getType(){
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

double DistributionRandomNumberGenerator(distribution & dist, double & RNG){
  return dist.RandomNumberGenerator(RNG);
}

double untrDistributionPdf(distribution & dist, double & x){
  return dist.untrPdf(x);
}

double untrDistributionCdf(distribution & dist, double & x){
  return dist.untrCdf(x);
}

double untrDistributionRandomNumberGenerator(distribution & dist, double & RNG){
  return dist.untrRandomNumberGenerator(RNG);
}

std::string getDistributionType(distribution & dist) {
  return dist.getType();
}
