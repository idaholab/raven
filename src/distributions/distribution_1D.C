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
#include "Interpolation_Functions.h"
#include <string>


#define _USE_MATH_DEFINES   // needed in order to use M_PI = 3.14159

/*
 * CLASS UNIFORM DISTRIBUTION
 */

template<>
InputParameters validParams<UniformDistribution>(){

   InputParameters params = validParams<distribution>();
   return params;
}

class UniformDistribution;

UniformDistribution::UniformDistribution(const std::string & name, InputParameters parameters):
   distribution(name,parameters)
{
}
UniformDistribution::~UniformDistribution()
{
}
double
UniformDistribution::Pdf(double & x){
   double value;
   value = 1.0/((_dis_parameters.find("xMax") ->second) -
              (_dis_parameters.find("xMin") ->second));
   return value;
}
double
UniformDistribution::Cdf(double & x){
   double value;
   value = 1.0/(((_dis_parameters.find("xMax") ->second) -
              (_dis_parameters.find("xMin") ->second))*
              (x - (_dis_parameters.find("xMin") ->second)));
   return value;
}
double
UniformDistribution::RandomNumberGenerator(double & RNG){
   double value;
   if(_force_dist == 0){
     value = (_dis_parameters.find("xMin") ->second)+RNG*
               ((_dis_parameters.find("xMax") ->second)-
                (_dis_parameters.find("xMin") ->second));
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
     mooseError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

double  UniformDistribution::untrPdf(double & x){
   double value=UniformDistribution::Pdf(x);
   return value;
}

double  UniformDistribution::untrCdf(double & x){
   double value=UniformDistribution::Cdf(x);
   return value;
}

double  UniformDistribution::untrRandomNumberGenerator(double & RNG){
   double value=UniformDistribution::RandomNumberGenerator(RNG);
   return value;
}


/*
 * CLASS NORMAL DISTRIBUTION
 */
template<>
InputParameters validParams<NormalDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("mu", "Mean");
   params.addRequiredParam<double>("sigma", "Standard deviation");
   return params;
}

class NormalDistribution;

NormalDistribution::NormalDistribution(const std::string & name, InputParameters parameters):
   distribution(name,parameters){
   _dis_parameters["mu"] = getParam<double>("mu");
   _dis_parameters["sigma"] = getParam<double>("sigma");
}
NormalDistribution::~NormalDistribution(){
}

double
NormalDistribution::untrPdf(double & x){
   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   double value=1/(sqrt(2.0*M_PI*sigma*sigma))*exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
   return value;
}

double
NormalDistribution::untrCdf(double & x){
   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   double value=0.5*(1+erf((x-mu)/(sqrt(2*sigma*sigma))));
   return value;
}

double
NormalDistribution::untrRandomNumberGenerator(double & RNG){
   double stdNorm;
   double value;

   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

    if (RNG < 0.5)
        stdNorm = -AbramStegunApproximation( sqrt(-2.0*log(RNG)) );
    else
       stdNorm = AbramStegunApproximation( sqrt(-2.0*log(1-RNG)) );

    value = mu + sigma * stdNorm;

    return value;
}

double
NormalDistribution::Pdf(double & x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(NormalDistribution::untrCdf(xMax) - NormalDistribution::untrCdf(xMin)) * NormalDistribution::untrPdf(x);
   else
      value=-1;

   return value;
}

double
NormalDistribution::Cdf(double & x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(NormalDistribution::untrCdf(xMax) - NormalDistribution::untrCdf(xMin)) * NormalDistribution::untrCdf(x);
   else
      value=-1;

   return value;
}

double
NormalDistribution::RandomNumberGenerator(double & RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
     std::cerr << "TRUNCATION IS " << _dis_parameters.find("truncation") ->second << std::endl;
     if (_dis_parameters.find("truncation") ->second == 1){
       double temp=NormalDistribution::untrCdf(xMin)+RNG*(NormalDistribution::untrCdf(xMax)-NormalDistribution::untrCdf(xMin));
       value=NormalDistribution::untrRandomNumberGenerator(temp);
     }
     else{
       value=-1;
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
     mooseError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

/*
 * CLASS LOG NORMAL DISTRIBUTION
 */

template<>
InputParameters validParams<LogNormalDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("mu", "Mean");
   params.addRequiredParam<double>("sigma", "Standard deviation");
   return params;
}

class LogNormalDistribution;

LogNormalDistribution::LogNormalDistribution(const std::string & name, InputParameters parameters):
   distribution(name,parameters)
{
   _dis_parameters["mu"] = getParam<double>("mu");
   _dis_parameters["sigma"] = getParam<double>("sigma");
}

LogNormalDistribution::~LogNormalDistribution()
{
}

double
LogNormalDistribution::untrPdf(double & x){
   double value;
   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   if (x<=0)
      mooseError("Error: Lognormal pdf evaluated for x<=0");
   else
      value=1/(sqrt(x*2.0*M_PI*sigma*sigma))*exp(-(log(x)-mu)*(log(x)-mu)/(2*sigma*sigma));

   return value;
}

double
LogNormalDistribution::untrCdf(double & x){
   double value;
   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

   if (x<=0)
      mooseError("Error: Lognormal pdf evaluated for x<=0");
   else
      value=0.5*(1+erf((log(x)-mu)/(sqrt(2*sigma*sigma))));

   return value;
}

double
LogNormalDistribution::untrRandomNumberGenerator(double & RNG){
   double stdNorm;
   double value;

   double mu=_dis_parameters.find("mu") ->second;
   double sigma=_dis_parameters.find("sigma") ->second;

    if (RNG < 0.5)
        stdNorm = -AbramStegunApproximation( sqrt(-2.0*log(RNG)) );
    else
       stdNorm = AbramStegunApproximation( sqrt(-2.0*log(1-RNG)) );

   value=exp(mu + sigma * stdNorm);

   return value;
}

double
LogNormalDistribution::Pdf(double & x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(LogNormalDistribution::untrCdf(xMax) - LogNormalDistribution::untrCdf(xMin)) * LogNormalDistribution::untrPdf(x);
   else
      value=-1;

   return value;
}

double
LogNormalDistribution::Cdf(double & x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(LogNormalDistribution::untrCdf(xMax) - LogNormalDistribution::untrCdf(xMin)) * LogNormalDistribution::untrCdf(x);
   else
      value=-1;

   return value;
}

double
LogNormalDistribution::RandomNumberGenerator(double & RNG){
  double value;
  double xMin = _dis_parameters.find("xMin") ->second;
  double xMax = _dis_parameters.find("xMax") ->second;

   if(_force_dist == 0){
     if (_dis_parameters.find("truncation") ->second == 1){
       double temp=LogNormalDistribution::untrCdf(xMin) + RNG * (LogNormalDistribution::untrCdf(xMax)-LogNormalDistribution::untrCdf(xMin));
       value=LogNormalDistribution::untrRandomNumberGenerator(temp);
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
     mooseError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

/*
 * CLASS TRIANGULAR DISTRIBUTION
 */


template<>
InputParameters validParams<TriangularDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("xPeak", "Maximum coordinate");
   params.addRequiredParam<double>("lowerBound", "Lower bound");
   params.addRequiredParam<double>("upperBound", "Upper bound");
   return params;
}

class TriangularDistribution;

TriangularDistribution::TriangularDistribution(const std::string & name, InputParameters parameters):
   distribution(name,parameters)
{
   _dis_parameters["xPeak"] = getParam<double>("xPeak");
}
TriangularDistribution::~TriangularDistribution()
{
}

double
TriangularDistribution::untrPdf(double & x){
   double value;
   double lb = _dis_parameters.find("lowerBound") ->second;
   double ub = _dis_parameters.find("upperBound") ->second;
   double peak = _dis_parameters.find("xPeak") ->second;

   if (x<lb)
      value=0;
   if ((x>lb)&(x<peak))
      value=2*(x-lb)/(ub-lb)/(peak-lb);
   if ((x>peak)&(x<ub))
      value=2*(ub-x)/(ub-lb)/(ub-peak);
   if (x>ub)
      value=1;

   return value;
}

double  TriangularDistribution::untrCdf(double & x){
   double value;
   double lb = _dis_parameters.find("lowerBound") ->second;
   double ub = _dis_parameters.find("upperBound") ->second;
   double peak = _dis_parameters.find("xPeak") ->second;

   if (x<lb)
      value=0;
   if ((x>lb)&(x<peak))
      value=(x-lb)*(x-lb)/(ub-lb)/(peak-lb);
   if ((x>peak)&(x<ub))
      value=1-(ub-x)*(ub-x)/(ub-lb)/(ub-peak);
   if (x>ub)
      value=0;

   return value;
}

double
TriangularDistribution::untrRandomNumberGenerator(double & RNG){
   double value;
   double lb = _dis_parameters.find("lowerBound") ->second;
   double ub = _dis_parameters.find("upperBound") ->second;
   double peak = _dis_parameters.find("xPeak") ->second;

   double threshold = (peak-lb)/(ub-lb);

   if (RNG<threshold)
      value=lb+sqrt(RNG*(peak-lb)/(ub-lb));
   else
      value=ub-sqrt((1-RNG)*(ub-peak)/(ub-lb));

   return value;
}

double
TriangularDistribution::Pdf(double & x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(TriangularDistribution::untrCdf(xMax) - TriangularDistribution::untrCdf(xMin)) * TriangularDistribution::untrPdf(x);
   else
      value=-1;

   return value;
}

double
TriangularDistribution::Cdf(double & x){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(TriangularDistribution::untrCdf(xMax) - TriangularDistribution::untrCdf(xMin)) * TriangularDistribution::untrCdf(x);
   else
      value=-1;

   return value;
   }

double
TriangularDistribution::RandomNumberGenerator(double & RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
     if (_dis_parameters.find("truncation") ->second == 1){
       double temp=TriangularDistribution::untrCdf(xMin)+RNG*(TriangularDistribution::untrCdf(xMax)-TriangularDistribution::untrCdf(xMin));
       value=TriangularDistribution::untrRandomNumberGenerator(temp);
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
     mooseError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}



/*
 * CLASS EXPONENTIAL DISTRIBUTION
 */

template<>
InputParameters validParams<ExponentialDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("lambda", "lambda");
   return params;
}

class ExponentialDistribution;

ExponentialDistribution::ExponentialDistribution(const std::string & name, InputParameters parameters):
   distribution(name,parameters)
{
   _dis_parameters["lambda"] = getParam<double>("lambda");
}
ExponentialDistribution::~ExponentialDistribution()
{
}

double
ExponentialDistribution::untrPdf(double & x){
   double value;
   double lambda=_dis_parameters.find("lambda") ->second;

   if (x >= 0.0)
      value = lambda*exp(-x*lambda);
   else
      mooseError("Exponential distribution (pdf calculation): parameter x not valid (x>0).");

   return value;
}

double
ExponentialDistribution::untrCdf(double & x){
   double value;
   double lambda=_dis_parameters.find("lambda") ->second;

   if (x >= 0.0)
      value = 1-exp(-x*lambda);
   else
      mooseError("Exponential distribution (Cdf calculation): parameter x not valid (x>0).");

   return value;
}

double
ExponentialDistribution::untrRandomNumberGenerator(double & RNG){
   double lambda=_dis_parameters.find("lambda") ->second;
   double value=-log(1-RNG)/(lambda);
   return value;
}

double
ExponentialDistribution::Pdf(double & x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(ExponentialDistribution::untrCdf(xMax) - ExponentialDistribution::untrCdf(xMin)) * ExponentialDistribution::untrPdf(x);
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
     mooseError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

double
ExponentialDistribution::Cdf(double & x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(ExponentialDistribution::untrCdf(xMax) - ExponentialDistribution::untrCdf(xMin)) * ExponentialDistribution::untrCdf(x);
   else
      value=-1;

   return value;
}

double
ExponentialDistribution::RandomNumberGenerator(double & RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1){
      double temp = ExponentialDistribution::untrCdf(xMin)+RNG*(ExponentialDistribution::untrCdf(xMax)-ExponentialDistribution::untrCdf(xMin));
      value=ExponentialDistribution::untrRandomNumberGenerator(temp);
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
     mooseError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}


/*
 * CLASS WEIBULL DISTRIBUTION
 */

template<>
InputParameters validParams<WeibullDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("k", "shape parameter");
   params.addRequiredParam<double>("lambda", "scale parameter");
   return params;
}

class WeibullDistribution;

WeibullDistribution::WeibullDistribution(const std::string & name, InputParameters parameters):
   distribution(name,parameters)
{
   _dis_parameters["k"] = getParam<double>("k");
   _dis_parameters["lambda"] = getParam<double>("lambda");
}

WeibullDistribution::~WeibullDistribution()
{
}

double
WeibullDistribution::untrPdf(double & x){
   double lambda = _dis_parameters.find("lambda") ->second;
   double k = _dis_parameters.find("k") ->second;
   double value;

   if (x >= 0)
      value = k/lambda * pow(x/lambda,k-1) * exp(-pow(x/lambda,k));
   else
      value=0;

   return value;
}

double
WeibullDistribution::untrCdf(double & x){
   double lambda = _dis_parameters.find("lambda") ->second;
   double k = _dis_parameters.find("k") ->second;
   double value;

   if (x >= 0)
      value = 1.0 - exp(-pow(x/lambda,k));
   else
      value=0;

      return value;
}

double
WeibullDistribution::untrRandomNumberGenerator(double & RNG){
   double lambda = _dis_parameters.find("lambda") ->second;
   double k = _dis_parameters.find("k") ->second;

   double value = - lambda * pow(log(1.0 - RNG),1/k);
   return value;
}

double
WeibullDistribution::Pdf(double & x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(WeibullDistribution::untrCdf(xMax) - WeibullDistribution::untrCdf(xMin)) * WeibullDistribution::untrPdf(x);
   else
      value=-1;

   return value;
}

double
WeibullDistribution::Cdf(double & x){
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;

   double value;

   if (_dis_parameters.find("truncation") ->second == 1)
      value = 1/(WeibullDistribution::untrCdf(xMax) - WeibullDistribution::untrCdf(xMin)) * WeibullDistribution::untrCdf(x);
   else
      value=-1;

   return value;
}

double
WeibullDistribution::RandomNumberGenerator(double & RNG){
   double value;
   double xMin = _dis_parameters.find("xMin") ->second;
   double xMax = _dis_parameters.find("xMax") ->second;
   if(_force_dist == 0){
   if (_dis_parameters.find("truncation") ->second == 1){
      double temp = WeibullDistribution::untrCdf(xMin) + RNG * (WeibullDistribution::untrCdf(xMax)-WeibullDistribution::untrCdf(xMin));
      value=WeibullDistribution::untrRandomNumberGenerator(temp);
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
     mooseError("ERROR: not recognized force_dist flag (!= 0, 1 , 2, 3)");
   }
   return value;
}

/*
 * CLASS CUSTOM DISTRIBUTION
 */

template<>
InputParameters validParams<CustomDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam< vector<double> >("x_coordinates", "coordinates along x");
   params.addRequiredParam< vector<double> >("y_coordinates", "coordinates along y");
   params.addRequiredParam<custom_dist_fit_type>("fitting_type", "type of fitting");
   params.addParam<int>("n_points",3,"Number of fitting point (for spline only)");
   return params;
}

class CustomDistribution;

CustomDistribution::CustomDistribution(const std::string & name, InputParameters parameters):
   distribution(name,parameters)
{
   _dis_parameters["x_coordinates"] = getParam<double>("x_coordinates");
   _dis_parameters["y_coordinates"] = getParam<double>("y_coordinates");
   _dis_parameters["fitting_type"] = getParam<custom_dist_fit_type>("fitting_type");
   _dis_parameters["n_points"] = getParam<double>("n_points");

}

CustomDistribution::~CustomDistribution()
{
}

double
CustomDistribution::Pdf(double & x){
   double value=_interpolation.interpolation(x);

   return value;
}

double
CustomDistribution::Cdf(double & x){
   double value=-1;

   return value;
}

double
CustomDistribution::RandomNumberGenerator(double & RNG){
   double value=-1;
   return value;
}

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
