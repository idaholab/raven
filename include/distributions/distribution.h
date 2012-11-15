/*
 * distribution.h
 *
 *  Created on: Nov 1, 2012
 *      Author: alfoa
 */

#ifndef DISTRIBUTION_H_
#define DISTRIBUTION_H_

//#include "InputParameters.h"
#include "Interpolation_Functions.h"
#include "RavenObject.h"
#include "distribution_type.h"
#include "distribution_min.h"
const int _defaultSeed = 1256955321;
//double    ErrReturn = -1.0;



template<>
InputParameters validParams<distribution>();

class distribution : public RavenObject
{
public:
   //> constructor for built-in distributions
   distribution(const std::string & name, InputParameters parameters);

//   distribution(double xMin, double xMax, distribution_type type, unsigned int seed);
   //> constructor for custom distributions
//   distribution(std::vector<double> x_coordinates, std::vector<double> y_coordinates, int numberPoints, custom_dist_fit_type fitting_type, unsigned int seed);
   virtual ~distribution();

   double  getVariable(std::string & variableName);                   ///< getVariable from mapping
   void updateVariable(std::string & variableName, double & newValue); ///< update variable into the mapping

   virtual double  Pdf(double & x) = 0;                                   ///< Pdf function at coordinate x
   virtual double  Cdf(double & x) = 0;                                   ///< Cdf function at coordinate x
   virtual double  RandomNumberGenerator() = 0;                           ///< RNG
   distribution_type getType();                                       ///< Get distribution type
   unsigned int & getSeed();                                          ///< Get seed

protected:
   distribution_type _type;                       ///< Distribution type
   std::map <std::string,double> _dis_parameters; ///< Distribution parameters
   Interpolation_Functions _interpolation;        ///< Interpolation class
   unsigned int _seed;                            ///< seed
};


#endif /* DISTRIBUTION_H_ */
