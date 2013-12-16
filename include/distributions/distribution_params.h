/*
 * distribution_params.h
 *
 * Created on Nov 27, 2013
 *  Author cogljj (Joshua J Cogliati)
 */
#ifndef DISTRIBUTION_PARAMS_H_
#define DISTRIBUTION_PARAMS_H_


#include "RavenObject.h"
#include "distribution.h"
#include "distribution_1D.h"

template<>
InputParameters validParams<distribution>();

class distribution : public RavenObject, public virtual BasicDistribution
{
 public:
   //> constructor for built-in distributions
  distribution(const std::string & name, InputParameters parameters);

  virtual ~distribution();

};

template<>
InputParameters validParams<UniformDistribution>();

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

template<>
InputParameters validParams<NormalDistribution>();

class NormalDistribution : public distribution, public BasicNormalDistribution {
 public:
  NormalDistribution(const std::string & name, InputParameters parameters);
  virtual ~NormalDistribution();  
};

template<>
InputParameters validParams<LogNormalDistribution>();

class LogNormalDistribution : public distribution, public BasicLogNormalDistribution {
public:
   LogNormalDistribution(const std::string & name, InputParameters parameters);
   virtual ~LogNormalDistribution();
};

template<>
InputParameters validParams<TriangularDistribution>();

class TriangularDistribution : public distribution, public BasicTriangularDistribution {
public:
   TriangularDistribution(const std::string & name, InputParameters parameters);
   virtual ~TriangularDistribution();
};

template<>
InputParameters validParams<ExponentialDistribution>();

class ExponentialDistribution : public distribution, public BasicExponentialDistribution {
public:
	ExponentialDistribution(const std::string & name, InputParameters parameters);
   virtual ~ExponentialDistribution();
};

template<>
InputParameters validParams<WeibullDistribution>();

class WeibullDistribution : public distribution, public BasicWeibullDistribution {
public:
   WeibullDistribution(const std::string & name, InputParameters parameters);
   virtual ~WeibullDistribution();
};

class GammaDistribution;

template<>
InputParameters validParams<GammaDistribution>();

class GammaDistribution : public distribution, public BasicGammaDistribution {
public:
  GammaDistribution(const std::string & name, InputParameters parameters);
  virtual ~GammaDistribution();
};

class BetaDistribution;

template<>
InputParameters validParams<BetaDistribution>();

class BetaDistribution : public distribution, public BasicBetaDistribution {
public:
  BetaDistribution(const std::string & name, InputParameters parameters);
  virtual ~BetaDistribution();
};

// template<>
// InputParameters validParams<CustomDistribution>();

// class CustomDistribution : public distribution, public BasicCustomDistribution {
// public:
//    CustomDistribution(const std::string & name, InputParameters parameters);
//    virtual ~CustomDistribution();
// };



#endif /* DISTRIBUTION_PARAMS_H_ */
