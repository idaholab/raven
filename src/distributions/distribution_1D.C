/*
 * distribution_1D.cpp
 *
 *  Created on: Mar 22, 2012
 *      Author: MANDD
 *      Modified: alfoa
 *      References:
 *      1- G. Cassella, R.G. Berger, "Statistical Inference", 2nd ed. Pacific Grove, CA: Duxbury Press (2001).
 *
 */

#include "distribution_1D.h"
#include "distributionFunctions.h"
#include <math.h>
#include <cmath>               // needed to use erfc error function
#include <string>
#include "dynamicArray.h"
#include <ctime>
#include <cstdlib>
//#include "Interpolation_Functions.h"
#include <string>
#include <limits>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/triangular.hpp>
#include <boost/math/distributions/exponential.hpp>
#include <boost/math/distributions/weibull.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/beta.hpp>

#define _USE_MATH_DEFINES   // needed in order to use M_PI = 3.14159

#define throwError(msg) { std::cerr << "\n\n" << msg << "\n\n"; throw std::runtime_error("Error"); }

/*
 * CLASS UNIFORM DISTRIBUTION
 */


class UniformDistributionBackend {
public:
  UniformDistributionBackend(double xMin, double xMax) : _backend(xMin,xMax) {
    
  } 
  boost::math::uniform _backend;
};


BasicUniformDistribution::BasicUniformDistribution(double xMin, double xMax)
{
  _dis_parameters["xMin"] = xMin;
  _dis_parameters["xMax"] = xMax;
  _uniform = new UniformDistributionBackend(xMin, xMax);
    
  if (xMin>xMax)
    throwError("ERROR: bounds for uniform distribution are incorrect");  
}

BasicUniformDistribution::~BasicUniformDistribution()
{
  delete _uniform;
}

double
BasicUniformDistribution::Pdf(double x){
  return boost::math::pdf(_uniform->_backend,x);
  /*double value;
   if (x<_dis_parameters.find("xMin") ->second)
      value=0;
   else if (x>_dis_parameters.find("xMax") ->second)
      value=0;
   else
	   value = 1.0/((_dis_parameters.find("xMax") ->second) - (_dis_parameters.find("xMin") ->second));
           return value;*/
}
double
BasicUniformDistribution::Cdf(double x){
  //double value;

  return boost::math::cdf(_uniform->_backend,x); 
   /*double xMax = _dis_parameters.find("xMax") ->second;
   double xMin = _dis_parameters.find("xMin") ->second;

   if (x<xMin)
	   value=0;
   else if (x>xMax)
	   value =1;
   else
	   value = (x-xMin)/(xMax-xMin);

           return value;*/
}
double
BasicUniformDistribution::RandomNumberGenerator(double RNG){
  double value;
    
   if ((RNG<0)&&(RNG>1))
      throwError("ERROR: in the evaluation of RNG for uniform distribution");   

   value = boost::math::quantile(_uniform->_backend,RNG);//(xMin)+RNG*((xMax)-(xMin));
    
   /*
   if(_force_dist == 0){
     xMin = _dis_parameters.find("xMin") ->second;
     xMax = _dis_parameters.find("xMax") ->second;
     value = (xMin)+RNG*((xMax)-(xMin));
 
   }
   else if(_force_dist == 1){
     value = (_dis_parameters.find("xMin") ->second);
   }
   else if(_force_dist == 2){
     value = ((_dis_parameters.find("xMax") ->second) - (_dis_parameters.find("xMin") ->second))/2.0;
   }
   else if(_force_dist == 3){
     value = (_dis_parameters.find("xMax") ->second);
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
     }*/
   return value;
}

double  BasicUniformDistribution::untrPdf(double x){
   double value=Pdf(x);
   return value;
}

double  BasicUniformDistribution::untrCdf(double x){
   double value=Cdf(x);
   return value;
}

double  BasicUniformDistribution::untrRandomNumberGenerator(double RNG){
   double value=RandomNumberGenerator(RNG);
   return value;
}


class NormalDistributionBackend {
public:
  NormalDistributionBackend(double mean, double sd) : _backend(mean, sd) {
    
  }
  boost::math::normal _backend;
};

/*
 * CLASS NORMAL DISTRIBUTION
 */

BasicNormalDistribution::BasicNormalDistribution(double mu, double sigma) {
  _dis_parameters["mu"] = mu; //mean
  _dis_parameters["sigma"] = sigma; //sd
  if(not hasParameter("truncation")) {
    _dis_parameters["truncation"] = 1.0;
  }
  if(not hasParameter("xMin")) {
    _dis_parameters["xMin"] = -std::numeric_limits<double>::max( );
  }
  if(not hasParameter("xMax")) {
    _dis_parameters["xMax"] = std::numeric_limits<double>::max( );
  }
  //std::cout << "mu " << mu << " sigma " << sigma 
  //          << " truncation " << _dis_parameters["truncation"] 
  //          << " xMin " << _dis_parameters["xMin"] 
  //          << " xMax " << _dis_parameters["xMax"] << std::endl;
  _normal = new NormalDistributionBackend(mu, sigma);   
}

BasicNormalDistribution::BasicNormalDistribution(double mu, double sigma, double xMin, double xMax) {
  _dis_parameters["mu"] = mu; //mean
  _dis_parameters["sigma"] = sigma; //sd
  if(not hasParameter("truncation")) {
    _dis_parameters["truncation"] = 1.0;
  }
  _dis_parameters["xMin"] = xMin;
  _dis_parameters["xMax"] = xMax;
  //std::cout << "mu " << mu << " sigma " << sigma 
  //          << " truncation " << _dis_parameters["truncation"] 
  //          << " xMin " << _dis_parameters["xMin"] 
  //          << " xMax " << _dis_parameters["xMax"] << std::endl;
  _normal = new NormalDistributionBackend(mu, sigma);   

}


BasicNormalDistribution::~BasicNormalDistribution(){
  delete _normal;
}

double
BasicNormalDistribution::untrPdf(double x){
  return boost::math::pdf(_normal->_backend, x);
    /*
   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   double value=1/(sqrt(2.0*M_PI*sigma*sigma))*exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
   return value;*/
}

double
BasicNormalDistribution::untrCdf(double x){
  return boost::math::cdf(_normal->_backend, x);
  /*double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   double value=0.5*(1+erf((x-mu)/(sqrt(2*sigma*sigma))));
   return value;*/
}

double
BasicNormalDistribution::untrRandomNumberGenerator(double RNG){
  return boost::math::quantile(_normal->_backend, RNG);
  /*
   double stdNorm;
   double value;

   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

    if (RNG < 0.5)
        stdNorm = -AbramStegunApproximation( sqrt(-2.0*log(RNG)) );
    else
       stdNorm = AbramStegunApproximation( sqrt(-2.0*log(1-RNG)) );

    value = mu + sigma * stdNorm;

    if (RNG == 1){
      value = std::numeric_limits<double>::max();
    }
    return value;*/
}

double
BasicNormalDistribution::Pdf(double x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(untrCdf(xMax) - untrCdf(xMin)) * untrPdf(x);
   else
      value=-1;

   return value;
}

double
BasicNormalDistribution::Cdf(double x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(untrCdf(xMax) - untrCdf(xMin)) * (untrCdf(x) - untrCdf(xMin));
   else
      value=-1;

   return value;
}

double
BasicNormalDistribution::RandomNumberGenerator(double RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
     if (_dis_parameters.find("truncation") ->second == 1){
       double temp=untrCdf(xMin)+RNG*(untrCdf(xMax)-untrCdf(xMin));
       value=untrRandomNumberGenerator(temp);
     }
     else{
       value=-1;
       //throwError("ERROR: force_dist 0 but truncation "<<(_dis_parameters.find("truncation") != _dis_parameters.end())<<","<<_dis_parameters.find("truncation")->second<<" found");
     }
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = _dis_parameters.find("mu") ->second;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag ("<<_force_dist<<"!= 0, 1 , 2, 3)");
   }
   if (RNG == 1){
     value = xMax;
   }
   return value;
}

class LogNormalDistributionBackend {
public:
  LogNormalDistributionBackend(double mean, double sd) : _backend(mean, sd) {
  }
  boost::math::lognormal _backend;
};

/*
 * CLASS LOG NORMAL DISTRIBUTION
 */


BasicLogNormalDistribution::BasicLogNormalDistribution(double mu, double sigma)
{
  _dis_parameters["mu"] = mu;
  _dis_parameters["sigma"] = sigma;
  _logNormal = new LogNormalDistributionBackend(mu, sigma);
    
  if (mu<0)
    throwError("ERROR: incorrect value of mu for lognormaldistribution");  
}

BasicLogNormalDistribution::~BasicLogNormalDistribution()
{
  delete _logNormal;
}

double
BasicLogNormalDistribution::untrPdf(double x){
  return boost::math::pdf(_logNormal->_backend, x);
  /*double value;
   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   if (x<=0)
      value=0;
   else
      value=1/(sqrt(x*x*2.0*M_PI*sigma*sigma))*exp(-(log(x)-mu)*(log(x)-mu)/(2*sigma*sigma));

      return value;*/
}

double
BasicLogNormalDistribution::untrCdf(double x){
  //std::cout << "LogNormalDistribution::untrCdf " << x << std::endl;
  if(x <= 0) {
    return 0.0;
  } else { 
    return boost::math::cdf(_logNormal->_backend, x);
  }
  /*double value;
   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   if (x<=0)
	   value=0;
   else
      value=0.5*(1+erf((log(x)-mu)/(sqrt(2*sigma*sigma))));

      return value;*/
}

double
BasicLogNormalDistribution::untrRandomNumberGenerator(double RNG){
  return boost::math::quantile(_logNormal->_backend, RNG);
  /*  double stdNorm;
   double value;

   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

    if (RNG < 0.5)
        stdNorm = -AbramStegunApproximation( sqrt(-2.0*log(RNG)) );
    else
       stdNorm = AbramStegunApproximation( sqrt(-2.0*log(1-RNG)) );

   value=exp(mu + sigma * stdNorm);

   return value;*/
}

double
BasicLogNormalDistribution::Pdf(double x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(untrCdf(xMax) - untrCdf(xMin)) * untrPdf(x);
   else
      value=-1;

   return value;
}

double
BasicLogNormalDistribution::Cdf(double x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(untrCdf(xMax) - untrCdf(xMin)) * (untrCdf(x)- untrCdf(xMin));
   else
      value=-1;

   return value;
}

double
BasicLogNormalDistribution::RandomNumberGenerator(double RNG){
  double value;
  double xMin = _dis_parameters.find("xMin") ->second;
  double xMax = _dis_parameters.find("xMax") ->second;

   if(_force_dist == 0){
     if (_dis_parameters.find("truncation") ->second == 1){
       double temp=untrCdf(xMin) + RNG * (untrCdf(xMax)-untrCdf(xMin));
       value=untrRandomNumberGenerator(temp);
     }
     else
       value=-1.0;
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = _dis_parameters.find("mu") ->second;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

/*
 * CLASS TRIANGULAR DISTRIBUTION
 */



class TriangularDistributionBackend {
public:
  TriangularDistributionBackend(double lower, double mode, double upper) :
    _backend(lower, mode, upper) { }
  boost::math::triangular _backend;
};


BasicTriangularDistribution::BasicTriangularDistribution(double xPeak, double lowerBound, double upperBound)
{
  _dis_parameters["xPeak"] = xPeak;
  _dis_parameters["lowerBound"] = lowerBound;
  _dis_parameters["upperBound"] = upperBound;
     
    
  if (upperBound < lowerBound)
    throwError("ERROR: bounds for triangular distribution are incorrect");  
  if (upperBound < _dis_parameters.find("xMin") ->second)
    throwError("ERROR: bounds and LB/UB are inconsistent for triangular distribution");
  if (lowerBound > _dis_parameters.find("xMax") ->second)
    throwError("ERROR: bounds and LB/UB are inconsistent for triangular distribution");
  _triangular = new TriangularDistributionBackend(lowerBound, xPeak, upperBound);

}
BasicTriangularDistribution::~BasicTriangularDistribution()
{
  delete _triangular;
}

double
BasicTriangularDistribution::untrPdf(double x){
  return boost::math::pdf(_triangular->_backend,x);
  /*double value;
   double lb = _dis_parameters.find("lowerBound") ->second;
   double ub = _dis_parameters.find("upperBound") ->second;
   double peak = _dis_parameters.find("xPeak") ->second;

   if (x<=lb)
      value=0;
   if ((x>lb)&(x<peak))
      value=2*(x-lb)/(ub-lb)/(peak-lb);
   if ((x>peak)&(x<ub))
      value=2*(ub-x)/(ub-lb)/(ub-peak);
   if (x>=ub)
      value=0;

      return value;*/
}

double  BasicTriangularDistribution::untrCdf(double x){
  return boost::math::cdf(_triangular->_backend,x);
  /*double value;
   double lb = _dis_parameters.find("lowerBound") ->second;
   double ub = _dis_parameters.find("upperBound") ->second;
   double peak = _dis_parameters.find("xPeak") ->second;

   if (x<=lb)
      value=0;
   if ((x>lb)&(x<peak))
      value=(x-lb)*(x-lb)/(ub-lb)/(peak-lb);
   if ((x>peak)&(x<ub))
      value=1-(ub-x)*(ub-x)/(ub-lb)/(ub-peak);
   if (x>=ub)
      value=1;

      return value;*/
}

double
BasicTriangularDistribution::untrRandomNumberGenerator(double RNG){
  return boost::math::quantile(_triangular->_backend,RNG);
  /*double value;
   double lb = _dis_parameters.find("lowerBound") ->second;
   double ub = _dis_parameters.find("upperBound") ->second;
   double peak = _dis_parameters.find("xPeak") ->second;

   double threshold = (peak-lb)/(ub-lb);

   if (RNG<threshold)
      value=lb+sqrt(RNG*(peak-lb)*(ub-lb));
   else
      value=ub-sqrt((1-RNG)*(ub-peak)*(ub-lb));

      return value;*/
}

double
BasicTriangularDistribution::Pdf(double x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
	   if ((x<xMin)||(x>xMax))
		   value=0;
	   else
		   value = 1/(untrCdf(xMax) - untrCdf(xMin)) * untrPdf(x);
   else
      value=-1;

   return value;
}

double
BasicTriangularDistribution::Cdf(double x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value=0;
	  else if (x>xMax)
		  value=1;
	  else{
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * (untrCdf(x)- untrCdf(xMin));
	  }
   else
      value=-1;

   return value;
   }

double
BasicTriangularDistribution::RandomNumberGenerator(double RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
     if (_dis_parameters.find("truncation") ->second == 1){
       double temp=untrCdf(xMin)+RNG*(untrCdf(xMax)-untrCdf(xMin));
       value=untrRandomNumberGenerator(temp);
     }
     else
       value=-1;
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = -1.0;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}



/*
 * CLASS EXPONENTIAL DISTRIBUTION
 */


class ExponentialDistributionBackend {
public:
  ExponentialDistributionBackend(double lambda) : _backend(lambda) {}
  boost::math::exponential _backend;
};


BasicExponentialDistribution::BasicExponentialDistribution(double lambda)
{
  _dis_parameters["lambda"] = lambda;
    
  if (lambda<0)
    throwError("ERROR: incorrect value of lambda for exponential distribution"); 

  _exponential = new ExponentialDistributionBackend(lambda);
}
BasicExponentialDistribution::~BasicExponentialDistribution()
{
  delete _exponential;
}

double
BasicExponentialDistribution::untrPdf(double x){
  return boost::math::pdf(_exponential->_backend, x);
  /*double value;
   double lambda=_dis_parameters.find("lambda") ->second;

   if (x >= 0.0)
      value = lambda*exp(-x*lambda);
   else
	   value=0.0;

           return value;*/
}

double
BasicExponentialDistribution::untrCdf(double x){
  if(x >= 0.0) {
    return boost::math::cdf(_exponential->_backend, x);
  } else {
    return 0.0;
  }
  /*double value;
   double lambda=_dis_parameters.find("lambda") ->second;

   if (x >= 0.0)
      value = 1-exp(-x*lambda);
   else
      value=0.0;

      return value;*/
}

double
BasicExponentialDistribution::untrRandomNumberGenerator(double RNG){
  return boost::math::quantile(_exponential->_backend, RNG);
  /*double lambda=_dis_parameters.find("lambda") ->second;
   double value=-log(1-RNG)/(lambda);
   return value;*/
}

double
BasicExponentialDistribution::Pdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value =0;
	  else if (x>xMax)
		  value =0;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * untrPdf(x);
   else
      value=-1;
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = -1.0;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

double
BasicExponentialDistribution::Cdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value =0;
	  else if (x>xMax)
		  value =1;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * (untrCdf(x)- untrCdf(xMin));
   else
      value=-1;

   return value;
}

double
BasicExponentialDistribution::RandomNumberGenerator(double RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1){
      double temp = untrCdf(xMin)+RNG*(untrCdf(xMax)-untrCdf(xMin));
      value=untrRandomNumberGenerator(temp);
   }
   else
      value=-1;
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = -1.0;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}


/*
 * CLASS WEIBULL DISTRIBUTION
 */


class WeibullDistributionBackend {
public:
  WeibullDistributionBackend(double shape, double scale) : _backend(shape, scale) {
    
  }
  boost::math::weibull _backend;
};


BasicWeibullDistribution::BasicWeibullDistribution(double k, double lambda)
{
  _dis_parameters["k"] = k; //shape
  _dis_parameters["lambda"] = lambda; //scale

  if ((lambda<0) || (k<0))
    throwError("ERROR: incorrect value of k or lambda for weibull distribution");

  _weibull = new WeibullDistributionBackend(k, lambda);
}

BasicWeibullDistribution::~BasicWeibullDistribution()
{
  delete _weibull;
}

double
BasicWeibullDistribution::untrPdf(double x){
  return boost::math::pdf(_weibull->_backend, x);
  /*double lambda = _dis_parameters.find("lambda") ->second;
   double k = _dis_parameters.find("k") ->second;
   double value;

   if (x >= 0)
      value = k/lambda * pow(x/lambda,k-1) * exp(-pow(x/lambda,k));
   else
      value=0;

      return value;*/
}

double
BasicWeibullDistribution::untrCdf(double x){
  if(x >= 0) {
    return boost::math::cdf(_weibull->_backend, x);
  } else {
    return 0.0;
  }
  /*double lambda = _dis_parameters.find("lambda") ->second;
   double k = _dis_parameters.find("k") ->second;
   double value;

   if (x >= 0)
      value = 1.0 - exp(-pow(x/lambda,k));
   else
	   value=0.0;

           return value;*/
}

double
BasicWeibullDistribution::untrRandomNumberGenerator(double RNG){
  return boost::math::quantile(_weibull->_backend, RNG);
  /*double lambda = _dis_parameters.find("lambda") ->second;
   double k = _dis_parameters.find("k") ->second;

   double value = lambda * pow(-log(1.0 - RNG),1/k);
   return value;*/
}

double
BasicWeibullDistribution::Pdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value=0;
	  else if (x>xMax)
		  value=0;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * untrPdf(x);
   else
      value=-1;

   return value;
}

double
BasicWeibullDistribution::Cdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value=0;
	  else if (x>xMax)
		  value=1;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * (untrCdf(x) - untrCdf(xMin));
   else
      value=-1;

   return value;
}

double
BasicWeibullDistribution::RandomNumberGenerator(double RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1){
      double temp = untrCdf(xMin) + RNG * (untrCdf(xMax)-untrCdf(xMin));
      value=untrRandomNumberGenerator(temp);
   }
   else
      value=-1;
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = -1.0;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

/*
 * CLASS GAMMA DISTRIBUTION
 */


class GammaDistributionBackend {
public:
  GammaDistributionBackend(double shape, double scale) : _backend(shape, scale) {
    
  }
  boost::math::gamma_distribution<> _backend;
};


BasicGammaDistribution::BasicGammaDistribution(double k, double theta, double low)
{
  _dis_parameters["k"] = k; //shape
  _dis_parameters["theta"] = theta; //scale
  _dis_parameters["low"] = low; //low value shift. 0.0 would be a regular gamma 
  // distribution

  if ((theta<0) || (k<0))
    throwError("ERROR: incorrect value of k or theta for gamma distribution");

  _gamma = new GammaDistributionBackend(k, theta);
}

BasicGammaDistribution::~BasicGammaDistribution()
{
  delete _gamma;
}

double
BasicGammaDistribution::untrPdf(double x){
  return boost::math::pdf(_gamma->_backend, x);
}

double
BasicGammaDistribution::untrCdf(double x){
  if(x >= 0) {
    return boost::math::cdf(_gamma->_backend, x);
  } else {
    return 0.0;
  }
}

double
BasicGammaDistribution::untrRandomNumberGenerator(double RNG){
  return boost::math::quantile(_gamma->_backend, RNG);
}

double
BasicGammaDistribution::Pdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   double low = _dis_parameters.find("low") ->second;

   double value;
   x = x - low; //Translate x value

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value=0;
	  else if (x>xMax)
		  value=0;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * untrPdf(x);
   else
      value=-1;

   return value;
}

double
BasicGammaDistribution::Cdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   double low = _dis_parameters.find("low") ->second;

   double value;
   x = x - low; //Translate x value

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value=0;
	  else if (x>xMax)
		  value=1;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * (untrCdf(x) - untrCdf(xMin));
   else
      value=-1;

   return value;
}

double
BasicGammaDistribution::RandomNumberGenerator(double RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   double low = _dis_parameters.find("low") ->second;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1){
      double temp = untrCdf(xMin) + RNG * (untrCdf(xMax)-untrCdf(xMin));
      value=untrRandomNumberGenerator(temp);
   }
   else
     return -1;
   //value=-1;
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = -1.0;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value+low;
}

/*
 * CLASS BETA DISTRIBUTION
 */


class BetaDistributionBackend {
public:
  BetaDistributionBackend(double alpha, double beta) : _backend(alpha, beta) {
    
  }
  boost::math::beta_distribution<> _backend;
};


BasicBetaDistribution::BasicBetaDistribution(double alpha, double beta)
{
  _dis_parameters["alpha"] = alpha;
  _dis_parameters["beta"] = beta;

  if ((alpha<0) || (beta<0))
    throwError("ERROR: incorrect value of alpha or beta for beta distribution");

  _beta = new BetaDistributionBackend(alpha, beta);
}

BasicBetaDistribution::~BasicBetaDistribution()
{
  delete _beta;
}

double
BasicBetaDistribution::untrPdf(double x){
  return boost::math::pdf(_beta->_backend, x);
}

double
BasicBetaDistribution::untrCdf(double x){
  if(x >= 0 and x <= 1) {
    return boost::math::cdf(_beta->_backend, x);
  } else if(x < 0){
    return 0.0;
  } else {
    return 1.0;
  }
}

double
BasicBetaDistribution::untrRandomNumberGenerator(double RNG){
  return boost::math::quantile(_beta->_backend, RNG);
}

double
BasicBetaDistribution::Pdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value=0;
	  else if (x>xMax)
		  value=0;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * untrPdf(x);
   else
      value=-1;

   return value;
}

double
BasicBetaDistribution::Cdf(double x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
	  if (x<xMin)
		  value=0;
	  else if (x>xMax)
		  value=1;
	  else
		  value = 1/(untrCdf(xMax) - untrCdf(xMin)) * (untrCdf(x) - untrCdf(xMin));
   else
      value=-1;

   return value;
}

double
BasicBetaDistribution::RandomNumberGenerator(double RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1){
      double temp = untrCdf(xMin) + RNG * (untrCdf(xMax)-untrCdf(xMin));
      value=untrRandomNumberGenerator(temp);
   }
   else
      value=-1;
   }
   else if(_force_dist == 1){
     value = xMin;
   }
   else if(_force_dist == 2){
     value = -1.0;
   }
   else if(_force_dist == 3){
     value = xMax;
   }
   else{
     throwError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

/*
 * CLASS CUSTOM DISTRIBUTION
 */


// BasicCustomDistribution::BasicCustomDistribution(double x_coordinates, double y_coordinates, int fitting_type, double n_points)
// {
//    _dis_parameters["x_coordinates"] = x_coordinates;
//    _dis_parameters["y_coordinates"] = y_coordinates;
//    _dis_parameters["fitting_type"] = fitting_type;
//    _dis_parameters["n_points"] = n_points;

// }

// BasicCustomDistribution::~BasicCustomDistribution()
// {
// }

// double
// BasicCustomDistribution::Pdf(double & x){
//    double value=_interpolation.interpolation(x);

//    return value;
// }

// double
// BasicCustomDistribution::Cdf(double & ){
//   //XXX implement
//    double value=-1;

//    return value;
// }

// double
// BasicCustomDistribution::RandomNumberGenerator(double & ){
//   //XXX implement
//    double value=-1;
//    return value;
// }



//
//   // Beta pdf
//      double distribution_1D::betaPdf (double x){
//         // parameter1=alpha   >0
//         // parameter2=beta    >0
//         // 0<x<1
//
//         double value;
//
//         /*if ((x > 0)&&(x < 1)&&(_parameter1 > 0)&&(_parameter2 > 0))
//              value = 1/betaFunc(_parameter1,_parameter2)*pow(x,_parameter1-1)*pow(1-x,_parameter2-1);
//              else */
//            value=-1;
//
//         return value;
//      }
//
//      double distribution_1D::betaCdf (double x){
//         // parameter1=alpha   >0
//         // parameter2=beta    >0
//         // 0<x<1
//
//         double value;
//
//         /*if ((x > 0)&&(x < 1)&&(_parameter1 > 0)&&(_parameter2 > 0))
//            value = betaInc(_parameter1,_parameter2 ,x);
//                else */
//            value=-1;
//
//         return value;
//      }

//
//   // Gamma pdf
//      double distribution_1D::gammaPdf(double x){
//         // parameter1= k   >0
//         // parameter2= theta     >0
//         // x>=0
//
//         double value;
//
//         /* if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
//            value=1/gammaFunc(_parameter1)/pow(_parameter2,_parameter1)*pow(x,_parameter1-1)*exp(-x/_parameter2);
//                else */
//            value=1;
//
//         return value;
//      }
//
//      double distribution_1D::gammaCdf(double x){
//         // parameter1=alpha, k   >0
//         // parameter2=beta, theta     >0
//         // x>=0
//
//         double value;
//
//         /* if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
//              value= gammp(_parameter1,x/_parameter2);
//              else */
//            value=1;
//
//         return value;
//      }


//      double distribution_1D::gammaRandNumberGenerator(){
//          double value=-1;//gammaRNG(_parameter1,_parameter2);
//         return value;
//      }
//
//      double distribution_1D::betaRandNumberGenerator(){
//          double value=-1;//betaRNG(_parameter1,_parameter2);
//         return value;
//      }
//
//      double distribution_1D::triangularRandNumberGenerator(){
//         double value;
//         double RNG = rand()/double(RAND_MAX);
//         double referenceValue=(_parameter1-_xMin)/(_xMax-_xMin);
//
//         if (RNG<referenceValue)
//            value= _xMin+sqrt(RNG*(_parameter1-_xMin)*(_xMax-_xMin));
//         else
//            value=_xMax-sqrt((1-RNG)*(_xMax-_parameter1)*(_xMax-_xMin));
//         return value;
//      }
