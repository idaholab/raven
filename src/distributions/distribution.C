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

   params.addParam< std::vector<double> >("PBwindow", "Probability window");
   params.addParam< std::vector<double> >("Vwindow" , "Value window");

   params.addParam<double>("ProbabilityThreshold", 1.0, "Probability Threshold");

   params.addParam<unsigned int>("seed", _defaultSeed ,"RNG seed");
   params.addRequiredParam<std::string>("type","distribution type");
   params.addParam<unsigned int>("truncation", 1 , "Type of truncation"); // Truncation types: 1) pdf_prime(x) = pdf(x)*c   2) [to do] pdf_prime(x) = pdf(x)+c
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
     custom_dist_fit_type fitting_type = static_cast<custom_dist_fit_type>((int)getParam<MooseEnum>("fitting_type"));

     _interpolation=Interpolation_Functions(x_coordinates,
                                            y_cordinates,
                                            numberPoints,
                                            fitting_type);
   }
      _seed = getParam<unsigned int>("seed");
      _force_dist = getParam<unsigned int>("force_distribution");
      _dis_parameters["truncation"] = double(getParam<unsigned int>("truncation"));

      _dis_vectorParameters["PBwindow"] = getParam<std::vector<double> >("PBwindow");
      _dis_vectorParameters["Vwindow"] = getParam<std::vector<double> >("Vwindow");

      _dis_parameters["ProbabilityThreshold"] = getParam<double>("ProbabilityThreshold");

      _checkStatus = false;
}

distribution::~distribution(){
}


double
distribution::getVariable(std::string & variableName){
   double res;
   if(_dis_parameters.find(variableName) != _dis_parameters.end()){
	  res = _dis_parameters.find(variableName) ->second;
   }
   else{
     mooseError("Parameter " << variableName << " was not found in distribution type " << _type <<".");
   }
   return res;
}

std::vector<double>
distribution::getVariableVector(std::string  variableName){
	std::vector<double> res;
   if(_dis_vectorParameters.find(variableName) != _dis_vectorParameters.end()){
	 res = _dis_vectorParameters.find(variableName) ->second;
   }
   else{
     mooseError("Parameter " << variableName << " was not found in distribution type " << _type <<".");
   }
   return res;
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

std::vector<std::string>
distribution::getVariableNames(){
  std::vector<std::string> paramtersNames;
  for (std::map<std::string,double>::iterator it = _dis_parameters.begin(); it!= _dis_parameters.end();it++){
    paramtersNames.push_back(it->first);
  }
  return paramtersNames;
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

double windowProcessing(distribution & dist, double & RNG){
	double value;

	if (dist.getVariableVector(std::string("PBwindow")).size()==1) // value Pb window
		value=dist.RandomNumberGenerator(RNG);
	else if(dist.getVariableVector(std::string("PBwindow")).size()==2){	// interval Pb window
		double pbLOW = dist.getVariableVector(std::string("PBwindow"))[0];
		double pbUP  = dist.getVariableVector(std::string("PBwindow"))[1];
		double pb=pbLOW+(pbUP-pbLOW)*RNG;
		value=dist.RandomNumberGenerator(pb);
	}
	else if(dist.getVariableVector(std::string("Vwindow")).size()==1)	// value V window
		value=RNG;
	else if(dist.getVariableVector(std::string("Vwindow")).size()==2){	// interval V window
		double valLOW = dist.getVariableVector(std::string("Vwindow"))[0];
		double valUP  = dist.getVariableVector(std::string("Vwindow"))[1];
		value=valLOW+(valUP-valLOW)*RNG;
	}
	else	// DEFAULT
		value = dist.RandomNumberGenerator(RNG);

	return value;
}

double DistributionRandomNumberGenerator(distribution & dist, double & RNG){
  //double standardRNG = dist.RandomNumberGenerator(RNG);
  double windowedRNG = windowProcessing(dist, RNG);

  return windowedRNG;
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

std::vector<std::string>
getDistributionVariableNames(distribution & dist)
{
  return dist.getVariableNames();
}



