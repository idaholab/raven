/*
 * distribution.h
 *
 *  Created on: Nov 1, 2012
 *      Author: alfoa
 */

#ifndef DISTRIBUTION_H_
#define DISTRIBUTION_H_

#include <map>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
//#include "Interpolation_Functions.h"
#include "distribution_min.h"

const int _defaultSeed = 1256955321;

enum truncation {MULTIPLICATIVE=1, SUM=2};

class distribution;

class BasicDistribution 
{
public:
   BasicDistribution();
   virtual ~BasicDistribution();
   double  getVariable(std::string & variableName);                   	///< getVariable from mapping
   std::vector<double>  getVariableVector(std::string  variableName);
   void updateVariable(std::string & variableName, double & newValue); 	///< update variable into the mapping

   virtual double  Pdf(double x) = 0;                           		///< Pdf function at coordinate x
   virtual double  Cdf(double x) = 0;                               	///< Cdf function at coordinate x
   virtual double  RandomNumberGenerator(double RNG) = 0;             ///< RNG

   virtual double untrPdf(double x) = 0;
   virtual double untrCdf(double x) = 0;
   virtual double untrRandomNumberGenerator(double RNG) = 0;

   std::string & getType();                                       		///< Get distribution type
   unsigned int getSeed();                                      		///< Get seed
   std::vector<std::string> getVariableNames();

   //virtual double windowProcessing(distribution & dist, double & RNG);

protected:
   bool hasParameter(std::string);
   std::string _type;                              ///< Distribution type
   std::map <std::string,double> _dis_parameters;  ///< Distribution parameters
   std::map <std::string,std::vector<double> > _dis_vectorParameters;
   //Interpolation_Functions _interpolation;         ///< Interpolation class
   unsigned int _seed;                             ///< seed
   unsigned int _force_dist;                       ///< if 0 => dist works as it is supposed to do
                                                   ///! force distribution to be evaluated at :
                                                   ///! (1) => xMin
                                                   ///! (2) => Mean
                                                   ///! (3) => xMax
   std::vector<double> _PBwindow;
   std::vector<double> _Vwindow;

   bool _checkStatus;
};



#endif /* DISTRIBUTION_H_ */
