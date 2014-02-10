/*
 * distribution_base_ND.h
 *
 *  Created on: Feb 6, 2014
 *      Author: alfoa
 *
 */

#ifndef DISTRIBUTION_BASE_ND_H_
#define DISTRIBUTION_BASE_ND_H_

#include <map>
#include <string>
#include "ND_Interpolation_Functions.h"
#include "distribution_min.h"

enum PbFunctionType{PDF,CDF};

class BasicDistributionND
{
public:
  BasicDistributionND();
   virtual ~BasicDistributionND();
   double  getVariable(std::string & variableName);                   	///< getVariable from mapping
   //std::vector<double>  getVariableVector(std::string  variableName);
   void updateVariable(std::string & variableName, double & newValue);

   double  Pdf(std::vector<double> x);
   double  Cdf(std::vector<double> x);
   double  InverseCdf(std::vector<double> x);
   std::string & getType();

protected:
   std::string                   _type;                              ///< Distribution type
   std::string                   _data_filename;
   PbFunctionType                _function_type;
   std::map <std::string,double> _dis_parameters;
   bool                          _checkStatus;
   ND_Interpolation              _interpolator;
};


#endif /* DISTRIBUTION_BASE_ND_H_ */
