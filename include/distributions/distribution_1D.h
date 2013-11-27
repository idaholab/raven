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

class UniformDistributionBackend;

class BasicUniformDistribution : public virtual BasicDistribution {
public:
  BasicUniformDistribution(double xMin, double xMax);
  ~BasicUniformDistribution();
   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
   UniformDistributionBackend * _uniform;
   // No parameters
};

class UniformDistribution : public distribution, public BasicUniformDistribution {
 public:
   UniformDistribution(const std::string & name, InputParameters parameters);
   virtual ~UniformDistribution();
   /*double Pdf(double & x) { return BasicUniformDistribution::Pdf(x);}
   double Cdf(double & x) { return BasicUniformDistribution::Cdf(x);}
   double RandomNumberGenerator(double & RNG) 
   { return BasicUniformDistribution::RandomNumberGenerator(RNG);}
   double untrPdf(double & x) { return BasicUniformDistribution::untrPdf(x);}
   double untrCdf(double & x) { return BasicUniformDistribution::untrCdf(x);}
   double untrRandomNumberGenerator(double & RNG) 
   { return BasicUniformDistribution::untrRandomNumberGenerator(RNG);}*/

};

/*
 * CLASS NORMAL DISTRIBUTION
 */
class NormalDistribution;

template<>
InputParameters validParams<NormalDistribution>();

class NormalDistributionBackend;

class BasicNormalDistribution : public virtual BasicDistribution {
public:
   BasicNormalDistribution(double mu, double sigma);
   virtual ~BasicNormalDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
   NormalDistributionBackend * _normal;
};

class NormalDistribution : public distribution, public BasicNormalDistribution {
 public:
  NormalDistribution(const std::string & name, InputParameters parameters);
  virtual ~NormalDistribution();  
};

/*
 * CLASS LOG NORMAL DISTRIBUTION
 */
class LogNormalDistribution;

template<>
InputParameters validParams<LogNormalDistribution>();

class LogNormalDistributionBackend;

class BasicLogNormalDistribution : public virtual BasicDistribution {
public:
   BasicLogNormalDistribution(double mu, double sigma);
   virtual ~BasicLogNormalDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
   LogNormalDistributionBackend * _logNormal;
};

class LogNormalDistribution : public distribution, public BasicLogNormalDistribution {
public:
   LogNormalDistribution(const std::string & name, InputParameters parameters);
   virtual ~LogNormalDistribution();
};

/*
 * CLASS TRIANGULAR DISTRIBUTION
 */

class TriangularDistribution;

template<>
InputParameters validParams<TriangularDistribution>();

class TriangularDistributionBackend;

class BasicTriangularDistribution : public virtual BasicDistribution {
public:
   BasicTriangularDistribution(double xPeak, double lowerBound, double upperBound);
   virtual ~BasicTriangularDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
   TriangularDistributionBackend * _triangular;
};

class TriangularDistribution : public distribution, public BasicTriangularDistribution {
public:
   TriangularDistribution(const std::string & name, InputParameters parameters);
   virtual ~TriangularDistribution();
};


/*
 * CLASS EXPONENTIAL DISTRIBUTION
 */

class ExponentialDistribution;

template<>
InputParameters validParams<ExponentialDistribution>();

class ExponentialDistributionBackend;

class BasicExponentialDistribution : public virtual BasicDistribution {
public:
   BasicExponentialDistribution(double lambda);
   virtual ~BasicExponentialDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
   ExponentialDistributionBackend * _exponential;
};

class ExponentialDistribution : public distribution, public BasicExponentialDistribution {
public:
	ExponentialDistribution(const std::string & name, InputParameters parameters);
   virtual ~ExponentialDistribution();
};

/*
 * CLASS WEIBULL DISTRIBUTION
 */

class WeibullDistribution;

template<>
InputParameters validParams<WeibullDistribution>();

class WeibullDistributionBackend;

class BasicWeibullDistribution : public virtual BasicDistribution {
public:
   BasicWeibullDistribution(double k, double lambda);
   virtual ~BasicWeibullDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

   double  untrPdf(double & x);
   double  untrCdf(double & x);
   double  untrRandomNumberGenerator(double & RNG);

protected:
   WeibullDistributionBackend * _weibull;
};

class WeibullDistribution : public distribution, public BasicWeibullDistribution {
public:
   WeibullDistribution(const std::string & name, InputParameters parameters);
   virtual ~WeibullDistribution();
};

/*
 * CLASS CUSTOM DISTRIBUTION
 */

class CustomDistribution;

template<>
InputParameters validParams<CustomDistribution>();

class BasicCustomDistribution : public virtual BasicDistribution {
public:
   BasicCustomDistribution(double x_coordinates, double y_coordinates, int fitting_type, double n_points);
   virtual ~BasicCustomDistribution();

   double  Pdf(double & x);                ///< Pdf function at coordinate x
   double  Cdf(double & x);                ///< Cdf function at coordinate x
   double  RandomNumberGenerator(double & RNG);        ///< RNG

protected:
};

class CustomDistribution : public distribution, public BasicCustomDistribution {
public:
   CustomDistribution(const std::string & name, InputParameters parameters);
   virtual ~CustomDistribution();
};

#endif
