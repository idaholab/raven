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

class DistributionBackend;

class BasicTruncatedDistribution : public virtual BasicDistribution {
public:
  /*BasicTruncatedDistribution();
  virtual ~BasicTruncatedDistribution();
  virtual double  Pdf(double x) = 0;                           		///< Pdf function at coordinate x
  virtual double  Cdf(double x) = 0;                               	///< Cdf function at coordinate x
  virtual double  InverseCdf(double x) = 0;             ///< x*/

  virtual double untrPdf(double x);
  virtual double untrCdf(double x);
  virtual double untrInverseCdf(double x);
protected: 
  DistributionBackend * _backend;
};

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
   double  InverseCdf(double x);        ///< x

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrInverseCdf(double x);

protected:
   UniformDistributionBackend * _uniform;
   // No parameters
};


/*
 * CLASS NORMAL DISTRIBUTION
 */
class NormalDistribution;


class NormalDistributionBackend;

class BasicNormalDistribution : public  BasicTruncatedDistribution {
public:
   BasicNormalDistribution(double mu, double sigma);
   BasicNormalDistribution(double mu, double sigma, double xMin, double xMax);
   virtual ~BasicNormalDistribution();

   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  InverseCdf(double x);        ///< x

};


/*
 * CLASS LOG NORMAL DISTRIBUTION
 */
class LogNormalDistribution;

class LogNormalDistributionBackend;

class BasicLogNormalDistribution : public BasicTruncatedDistribution {
public:
  BasicLogNormalDistribution(double mu, double sigma);
  virtual ~BasicLogNormalDistribution();
  
  double  Pdf(double x);                ///< Pdf function at coordinate x
  double  Cdf(double x);                ///< Cdf function at coordinate x
  double  InverseCdf(double x);        ///< x

  double untrCdf(double x);
};

/*
 * CLASS LOGISTIC DISTRIBUTION
 */
class LogisticDistribution;


class LogisticDistributionBackend;

class BasicLogisticDistribution : public virtual BasicDistribution {
public:
   BasicLogisticDistribution(double location, double scale);
   virtual ~BasicLogisticDistribution();

   double  Pdf(double x);                ///< Pdf function at coordinate x
   double  Cdf(double x);                ///< Cdf function at coordinate x
   double  InverseCdf(double x);        ///< x

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrInverseCdf(double x);

protected:
   LogisticDistributionBackend * _logistic;
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
   double  InverseCdf(double x);        ///< x

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrInverseCdf(double x);

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
   double  InverseCdf(double x);        ///< x

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrInverseCdf(double x);

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
   double  InverseCdf(double x);        ///< x

   double  untrPdf(double x);
   double  untrCdf(double x);
   double  untrInverseCdf(double x);

protected:
   WeibullDistributionBackend * _weibull;
};

/*
 * CLASS GAMMA DISTRIBUTION
 */

class GammaDistributionBackend;

class BasicGammaDistribution : public virtual BasicDistribution {
public:
  BasicGammaDistribution(double k, double theta, double low);
  virtual ~BasicGammaDistribution();

  double  Pdf(double x);                ///< Pdf function at coordinate x
  double  Cdf(double x);                ///< Cdf function at coordinate x
  double  InverseCdf(double x);        ///< x
  
  double  untrPdf(double x);
  double  untrCdf(double x);
  double  untrInverseCdf(double x);

protected:
  GammaDistributionBackend * _gamma;    
};

/*
 * CLASS BETA DISTRIBUTION
 */

class BetaDistributionBackend;

class BasicBetaDistribution : public virtual BasicDistribution {
public:
  BasicBetaDistribution(double alpha, double beta, double scale);
  virtual ~BasicBetaDistribution();

  double  Pdf(double x);                ///< Pdf function at coordinate x
  double  Cdf(double x);                ///< Cdf function at coordinate x
  double  InverseCdf(double x);        ///< x
  
  double  untrPdf(double x);
  double  untrCdf(double x);
  double  untrInverseCdf(double x);

protected:
  BetaDistributionBackend * _beta;    
};

/*
 * CLASS POISSON DISTRIBUTION
 */

class PoissonDistributionBackend;

class BasicPoissonDistribution : public virtual BasicDistribution {
public:
  BasicPoissonDistribution(double mu);
  virtual ~BasicPoissonDistribution();

  double  Pdf(double x);                ///< Pdf function at coordinate x
  double  Cdf(double x);                ///< Cdf function at coordinate x
  double  InverseCdf(double x);        ///< x
  
  double  untrPdf(double x);
  double  untrCdf(double x);
  double  untrInverseCdf(double x);

protected:
  PoissonDistributionBackend * _poisson;
};

/*
 * CLASS BINOMIAL DISTRIBUTION
 */

class BinomialDistributionBackend;

class BasicBinomialDistribution : public virtual BasicDistribution {
public:
  BasicBinomialDistribution(double n, double p);
  virtual ~BasicBinomialDistribution();

  double  Pdf(double x);                ///< Pdf function at coordinate x
  double  Cdf(double x);                ///< Cdf function at coordinate x
  double  InverseCdf(double x);        ///< x
  
  double  untrPdf(double x);
  double  untrCdf(double x);
  double  untrInverseCdf(double x);

protected:
  BinomialDistributionBackend * _binomial;
};

/*
 * CLASS BERNOULLI DISTRIBUTION
 */

class BernoulliDistributionBackend;

class BasicBernoulliDistribution : public virtual BasicDistribution {
public:
  BasicBernoulliDistribution(double p);
  virtual ~BasicBernoulliDistribution();

  double  Pdf(double x);                ///< Pdf function at coordinate x
  double  Cdf(double x);                ///< Cdf function at coordinate x
  double  InverseCdf(double x);        ///< x
  
  double  untrPdf(double x);
  double  untrCdf(double x);
  double  untrInverseCdf(double x);

protected:
  BernoulliDistributionBackend * _bernoulli;
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
//    double  InverseCdf(double x);        ///< x

// protected:
// };

#endif
