/*
 * distribution.C
 *
 *  Created on: Nov 1, 2012
 *      Author: alfoa
 */
#include "distribution.h"

using namespace std;





double
BasicDistribution::getVariable(std::string & variableName){
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
BasicDistribution::getVariableVector(std::string  variableName){
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
BasicDistribution::updateVariable(std::string & variableName, double & newValue){
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
BasicDistribution::getType(){
   return _type;
}

std::vector<std::string>
BasicDistribution::getVariableNames(){
  std::vector<std::string> paramtersNames;
  for (std::map<std::string,double>::iterator it = _dis_parameters.begin(); it!= _dis_parameters.end();it++){
    paramtersNames.push_back(it->first);
  }
  return paramtersNames;
}

double getDistributionVariable(BasicDistribution & dist, std::string & variableName){
  return dist.getVariable(variableName);
}

void DistributionUpdateVariable(BasicDistribution & dist, std::string & variableName, double & newValue){
  dist.updateVariable(variableName, newValue);
}

double DistributionPdf(BasicDistribution & dist, double & x){
  return dist.Pdf(x);
}

double DistributionCdf(BasicDistribution & dist, double & x){
  return dist.Cdf(x);
}

double windowProcessing(BasicDistribution & dist, double & RNG){
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

double DistributionRandomNumberGenerator(BasicDistribution & dist, double & RNG){
  //double standardRNG = dist.RandomNumberGenerator(RNG);
  double windowedRNG = windowProcessing(dist, RNG);

  return windowedRNG;
}

double untrDistributionPdf(BasicDistribution & dist, double & x){
  return dist.untrPdf(x);
}

double untrDistributionCdf(BasicDistribution & dist, double & x){
  return dist.untrCdf(x);
}

double untrDistributionRandomNumberGenerator(BasicDistribution & dist, double & RNG){
  return dist.untrRandomNumberGenerator(RNG);
}

std::string getDistributionType(BasicDistribution & dist) {
  return dist.getType();
}

std::vector<std::string>
getDistributionVariableNames(BasicDistribution & dist)
{
  return dist.getVariableNames();
}



