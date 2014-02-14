/*
 * distribution.C
 *
 *  Created on: Feb 6, 2014
 *      Author: alfoa
 *
 */

#include "distribution_base_ND.h"
#include <stdexcept>
#include <iostream>
using namespace std;

#define throwError(msg) { std::cerr << "\n\n" << msg << "\n\n"; throw std::runtime_error("Error"); }



BasicDistributionND::BasicDistributionND()
{
}

BasicDistributionND::~BasicDistributionND()
{
}

double
BasicDistributionND::getVariable(std::string & variableName){
   double res;

   if(_dis_parameters.find(variableName) != _dis_parameters.end())
   {
	  res = _dis_parameters.find(variableName) ->second;
   }
   else
   {
     throwError("Parameter " << variableName << " was not found in distribution type " << _type <<".");
   }
   return res;
}

void
BasicDistributionND::updateVariable(std::string & variableName, double & newValue){
   if(_dis_parameters.find(variableName) != _dis_parameters.end())
   {
     _dis_parameters[variableName] = newValue;
   }
   else
   {
     throwError("Parameter " << variableName << " was not found in distribution type " << _type << ".");
   }
}

std::string &
BasicDistributionND::getType(){
   return _type;
}

double
getDistributionVariable(BasicDistributionND & dist, std::string & variableName){
  return dist.getVariable(variableName);
}

void
DistributionUpdateVariable(BasicDistributionND & dist, std::string & variableName, double & newValue){
  dist.updateVariable(variableName, newValue);
}

std::string
getDistributionType(BasicDistributionND & dist) {
  return dist.getType();
}

double DistributionPdf(BasicDistributionND & dist, std::vector<double> & x)
{
  return dist.Pdf(x);
}

double DistributionCdf(BasicDistributionND & dist, std::vector<double> & x)
{
  return dist.Cdf(x);
}

double
DistributionInverseCdf(BasicDistributionND & dist, std::vector<double> & x){
  //double standardRNG = dist.InverseCdf(x);
  return dist.InverseCdf(x);
}


