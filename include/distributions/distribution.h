/*
 * distribution.h
 *
 *  Created on: Nov 1, 2012
 *      Author: alfoa
 */

#ifndef DISTRIBUTION_H_
#define DISTRIBUTION_H_


#include "Interpolation_Functions.h"
#include "RavenObject.h"
#include "distribution_min.h"
const int _defaultSeed = 1256955321;

enum truncation {MULTIPLICATIVE=1, SUM=2};


template<>
InputParameters validParams<distribution>();

class distribution : public RavenObject
{
public:
   //> constructor for built-in distributions
   distribution(const std::string & name, InputParameters parameters);

   virtual ~distribution();

   double  getVariable(std::string & variableName);                   	///< getVariable from mapping
   void updateVariable(std::string & variableName, double & newValue); 	///< update variable into the mapping

   virtual double  Pdf(double & x) = 0;                           		///< Pdf function at coordinate x
   virtual double  Cdf(double & x) = 0;                               	///< Cdf function at coordinate x
   virtual double  RandomNumberGenerator(double & RNG) = 0;             ///< RNG

   virtual double untrPdf(double & x) = 0;
   virtual double untrCdf(double & x) = 0;
   virtual double untrRandomNumberGenerator(double & RNG) = 0;

   std::string & getType();                                       		///< Get distribution type
   unsigned int & getSeed();                                      		///< Get seed

protected:
   std::string _type;                              ///< Distribution type
   std::map <std::string,double> _dis_parameters;  ///< Distribution parameters
   Interpolation_Functions _interpolation;         ///< Interpolation class
   unsigned int _seed;                             ///< seed
   unsigned int _force_dist;                       ///< if 0 => dist works as it is supposed to do
                                                   ///! force distribution to be evaluated at :
                                                   ///! (1) => xMin
                                                   ///! (2) => Mean
                                                   ///! (3) => xMax
};


#endif /* DISTRIBUTION_H_ */
