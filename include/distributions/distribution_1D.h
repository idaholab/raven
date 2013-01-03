/*
 * distribution_1D.h
 *
 *  Created on: Mar 22, 2012
 *      Author: MANDD
 *      References:
 *
 *      Tests      : None for the custom
 *
 *      Problems   : Gamma, Beta and Custom distributions have no RNG in place yet
 *      Issues      : None
 *      Complaints   : None
 *      Compliments   : None
 *
 */

#ifndef DISTRIBUTION_1D_H_
#define DISTRIBUTION_1D_H_

#include <string>
#include <vector>
#include "Interpolation_Functions.h"
#include "distribution.h"
#include "distributionFunctions.h"


/*
 * CLASS UNIFORM DISTRIBUTION
 */
class UniformDistribution;

template<>
InputParameters validParams<UniformDistribution>();

class UniformDistribution : public distribution {
public:
   UniformDistribution(const std::string & name, InputParameters parameters);
   virtual ~UniformDistribution();
   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
   // No parameters
};

/*
 * CLASS NORMAL DISTRIBUTION
 */
class NormalDistribution;

template<>
InputParameters validParams<NormalDistribution>();

class NormalDistribution : public distribution {
public:
   NormalDistribution(const std::string & name, InputParameters parameters);
   virtual ~NormalDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
};

/*
 * CLASS LOG NORMAL DISTRIBUTION
 */
class LogNormalDistribution;

template<>
InputParameters validParams<LogNormalDistribution>();

class LogNormalDistribution : public distribution {
public:
   LogNormalDistribution(const std::string & name, InputParameters parameters);
   virtual ~LogNormalDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
};

/*
 * CLASS TRIANGULAR DISTRIBUTION
 */

class TriangularDistribution;

template<>
InputParameters validParams<TriangularDistribution>();

class TriangularDistribution : public distribution {
public:
   TriangularDistribution(const std::string & name, InputParameters parameters);
   virtual ~TriangularDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
};


/*
 * CLASS EXPONENTIAL DISTRIBUTION
 */

class ExponentialDistribution;

template<>
InputParameters validParams<ExponentialDistribution>();

class ExponentialDistribution : public distribution {
public:
	ExponentialDistribution(const std::string & name, InputParameters parameters);
   virtual ~ExponentialDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
};


/*
 * CLASS WEIBULL DISTRIBUTION
 */

class WeibullDistribution;

template<>
InputParameters validParams<WeibullDistribution>();

class WeibullDistribution : public distribution {
public:
   WeibullDistribution(const std::string & name, InputParameters parameters);
   virtual ~WeibullDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
};


/*
 * CLASS CUSTOM DISTRIBUTION
 */

class CustomDistribution;

template<>
InputParameters validParams<CustomDistribution>();

class CustomDistribution : public distribution {
public:
   CustomDistribution(const std::string & name, InputParameters parameters);
   virtual ~CustomDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

protected:
};


#endif
