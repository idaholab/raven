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
//#include "Interpolation_Functions.h"
#include "distribution.h"
#include "distributionFunctions.h"


/*
 * CLASS UNIFORM DISTRIBUTION
 */
class UniformDistribution;


class UniformDistributionBackend;

class BasicUniformDistribution : public virtual BasicDistribution {
public:
  BasicUniformDistribution(double xMin, double xMax);
  virtual ~BasicUniformDistribution();
   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double RNG);        ///< RNG

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrRandomNumberGenerator(double RNG);

protected:
   UniformDistributionBackend * _uniform;
   // No parameters
};


/*
 * CLASS NORMAL DISTRIBUTION
 */
class NormalDistribution;


class NormalDistributionBackend;

class BasicNormalDistribution : public virtual BasicDistribution {
public:
   BasicNormalDistribution(double mu, double sigma);
   virtual ~BasicNormalDistribution();

   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double RNG);        ///< RNG

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrRandomNumberGenerator(double RNG);

protected:
   NormalDistributionBackend * _normal;
};


/*
 * CLASS LOG NORMAL DISTRIBUTION
 */
class LogNormalDistribution;

class LogNormalDistributionBackend;

class BasicLogNormalDistribution : public virtual BasicDistribution {
public:
   BasicLogNormalDistribution(double mu, double sigma);
   virtual ~BasicLogNormalDistribution();

   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double RNG);        ///< RNG

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrRandomNumberGenerator(double RNG);

protected:
   LogNormalDistributionBackend * _logNormal;
};

/*
 * CLASS TRIANGULAR DISTRIBUTION
 */

class TriangularDistribution;

class TriangularDistributionBackend;

class BasicTriangularDistribution : public virtual BasicDistribution {
public:
   BasicTriangularDistribution(double xPeak, double lowerBound, double upperBound);
   virtual ~BasicTriangularDistribution();

   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double RNG);        ///< RNG

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrRandomNumberGenerator(double RNG);

protected:
   TriangularDistributionBackend * _triangular;
};


/*
 * CLASS EXPONENTIAL DISTRIBUTION
 */

class ExponentialDistribution;

class ExponentialDistributionBackend;

class BasicExponentialDistribution : public virtual BasicDistribution {
public:
   BasicExponentialDistribution(double lambda);
   virtual ~BasicExponentialDistribution();

   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double RNG);        ///< RNG

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrRandomNumberGenerator(double RNG);

protected:
   ExponentialDistributionBackend * _exponential;
};

/*
 * CLASS WEIBULL DISTRIBUTION
 */

class WeibullDistribution;

class WeibullDistributionBackend;

class BasicWeibullDistribution : public virtual BasicDistribution {
public:
   BasicWeibullDistribution(double k, double lambda);
   virtual ~BasicWeibullDistribution();

   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double RNG);        ///< RNG

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrRandomNumberGenerator(double RNG);

protected:
   WeibullDistributionBackend * _weibull;
};

/*
 * CLASS CUSTOM DISTRIBUTION
 */

// class CustomDistribution;

// class BasicCustomDistribution : public virtual BasicDistribution {
// public:
//    BasicCustomDistribution(double x_coordinates, double y_coordinates, int fitting_type, double n_points);
//    virtual ~BasicCustomDistribution();

//    double  Pdf(double x);                ///< Pdf function at coordinate x
//    double  Cdf(double x);                ///< Cdf function at coordinate x
//    double  RandomNumberGenerator(double RNG);        ///< RNG

// protected:
// };

#endif
