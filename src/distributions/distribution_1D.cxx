/* Copyright 2017 Battelle Energy Alliance, LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
#include <cmath>               // needed to use erfc error function
#include <string>
#include "dynamicArray.h"
#include <ctime>
#include <cstdlib>
//#include "InterpolationFunctions.h"
#include <string>
#include <limits>
#include <iso646.h>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/normal.hpp>

#define _USE_MATH_DEFINES   // needed in order to use M_PI = 3.14159

#define throwError(msg) { std::cerr << "\n\n" << msg << "\n\n"; throw std::runtime_error("Error"); }

class DistributionBackend {
public:
  virtual double pdf(double x) = 0;
  virtual double cdf(double x) = 0;
  virtual double cdfComplement(double x) = 0;
  virtual double quantile(double x) = 0;
  virtual double mean() = 0;
  virtual double standard_deviation() = 0;
  virtual double median() = 0;
  virtual double mode() = 0;
  virtual double hazard(double x) = 0;
  virtual ~DistributionBackend() {};
};

/*
 * Class Basic Truncated Distribution
 * This class implements a basic truncated distribution that can
 * be inherited from.
 */

BasicTruncatedDistribution::BasicTruncatedDistribution(double x_min, double x_max)
{
  if(not hasParameter("truncation"))
  {
    _dist_parameters["truncation"] = 1.0;
  }
    _dist_parameters["xMin"] = x_min;
    _dist_parameters["xMax"] = x_max;
}

double
BasicTruncatedDistribution::pdf(double x){
  double value;
  double x_min = _dist_parameters.find("xMin") ->second;
  double x_max = _dist_parameters.find("xMax") ->second;

  if (_dist_parameters.find("truncation") ->second == 1) {
    if ((x<x_min)||(x>x_max)) {
      value=0;
    } else {
      value = 1/(untrCdf(x_max) - untrCdf(x_min)) * untrPdf(x);
    }
  } else {
    value=-1;
  }

  return value;
}

double
BasicTruncatedDistribution::cdf(double x){
  double value;
  double x_min = _dist_parameters.find("xMin") ->second;
  double x_max = _dist_parameters.find("xMax") ->second;

  if (_dist_parameters.find("truncation") ->second == 1) {
    if (x<x_min) {
      value=0;
    } else if (x>x_max) {
      value=1;
    } else{
      value = 1/(untrCdf(x_max) - untrCdf(x_min)) * (untrCdf(x)- untrCdf(x_min));
    }
  } else {
    value=-1;
  }

  return value;
}

double
BasicTruncatedDistribution::inverseCdf(double x){
  double value;
  double x_min = _dist_parameters.find("xMin") ->second;
  double x_max = _dist_parameters.find("xMax") ->second;

  if(x == 0.0) {
    //Using == in floats is generally a bad idea, but
    // 0.0 can be represented exactly.
    //In this case, return the minimum value
    return x_min;
  }
  if(x == 1.0) {
    //Using == in floats is generally a bad idea, but
    // 1.0 can be represented exactly.
    //In this case, return the maximum value
    return x_max;
  }
  if (_dist_parameters.find("truncation") ->second == 1){
    double temp=untrCdf(x_min)+x*(untrCdf(x_max)-untrCdf(x_min));
    value=untrInverseCdf(temp);
  } else {
    throwError("A valid solution for inverseCdf was not found!");
  }
  return value;
}


double BasicTruncatedDistribution::untrPdf(double x) {
  return _backend->pdf(x);
}

double BasicTruncatedDistribution::untrCdf(double x) {
  return _backend->cdf(x);
}

double BasicTruncatedDistribution::untrCdfComplement(double x) {
  return _backend->cdfComplement(x);
}

double BasicTruncatedDistribution::untrInverseCdf(double x) {
  return _backend->quantile(x);
}

double BasicTruncatedDistribution::untrMean() {
  return _backend->mean();
}

/**
   Calculates the untruncated standard deviation
   \return the standard deviation
*/
double BasicTruncatedDistribution::untrStdDev() {
  return _backend->standard_deviation();
}

double BasicTruncatedDistribution::untrMedian() {
  return _backend->median();
}

double BasicTruncatedDistribution::untrMode() {
  return _backend->mode();
}

double BasicTruncatedDistribution::untrHazard(double x) {
  return _backend->hazard(x);
}



/*
 * Class Basic Discrete Distribution
 * This class implements a basic discrete distribution that can
 * be inherited from.
 */

double BasicDiscreteDistribution::untrPdf(double x) {
  return _backend->pdf(x);
}

double BasicDiscreteDistribution::untrCdf(double x) {
  return _backend->cdf(x);
}

double BasicDiscreteDistribution::untrCdfComplement(double x) {
  return _backend->cdfComplement(x);
}

double BasicDiscreteDistribution::untrInverseCdf(double x) {
  return _backend->quantile(x);
}

double BasicDiscreteDistribution::untrMean() {
  return _backend->mean();
}

/**
   Calculates the untruncated standard deviation
   \return the standard deviation
*/
double BasicDiscreteDistribution::untrStdDev() {
  return _backend->standard_deviation();
}

double BasicDiscreteDistribution::untrMedian() {
  return _backend->median();
}

double BasicDiscreteDistribution::untrMode() {
  return _backend->mode();
}

double BasicDiscreteDistribution::untrHazard(double x) {
  return _backend->hazard(x);
}

double BasicDiscreteDistribution::pdf(double x) {
  return untrPdf(x);
}

double BasicDiscreteDistribution::cdf(double x) {
  return untrCdf(x);
}

double BasicDiscreteDistribution::inverseCdf(double x) {
  return untrInverseCdf(x);
}

/*
 * Class DistributionBackendTemplate implements a template that
 * can be used to create a DistributionBackend from a boost distribution
 */

template <class T>
class DistributionBackendTemplate : public DistributionBackend {
public:
  double pdf(double x) { return boost::math::pdf(*_backend, x); };
  double cdf(double x) { return boost::math::cdf(*_backend, x); };
  double cdfComplement(double x) { return boost::math::cdf(boost::math::complement(*_backend, x)); };
  double quantile(double x) { return boost::math::quantile(*_backend, x); };
  double mean() { return boost::math::mean(*_backend); };
  double standard_deviation() { return boost::math::standard_deviation(*_backend); };
  double median() { return boost::math::median(*_backend); };
  double mode() { return boost::math::mode(*_backend); };
  double hazard(double x) { return boost::math::hazard(*_backend, x); };
protected:
  T *_backend;
};

/*
 * CLASS UNIFORM DISTRIBUTION
 */


class UniformDistributionBackend : public DistributionBackendTemplate<boost::math::uniform> {
public:
  UniformDistributionBackend(double x_min, double x_max) {
    _backend = new boost::math::uniform(x_min,x_max);
  }
  ~UniformDistributionBackend() {
    delete _backend;
  }
};


BasicUniformDistribution::BasicUniformDistribution(double x_min, double x_max)
{
  _dist_parameters["xMin"] = x_min;
  _dist_parameters["xMax"] = x_max;
  _backend = new UniformDistributionBackend(x_min, x_max);

  if (x_min>x_max)
    throwError("ERROR: bounds for uniform distribution are incorrect");
}

BasicUniformDistribution::~BasicUniformDistribution()
{
  delete _backend;
}

double
BasicUniformDistribution::pdf(double x){
  return untrPdf(x);
}

double
BasicUniformDistribution::cdf(double x){
  return untrCdf(x);
}

double
BasicUniformDistribution::inverseCdf(double x){
  return untrInverseCdf(x);
}

class NormalDistributionBackend : public DistributionBackendTemplate<boost::math::normal> {
public:
  NormalDistributionBackend(double mean, double sd) {
    _backend = new boost::math::normal(mean, sd);
  }
  ~NormalDistributionBackend() {
    delete _backend;
  }
};

/*
 * CLASS NORMAL DISTRIBUTION
 */

BasicNormalDistribution::BasicNormalDistribution(double mu, double sigma) {
  _dist_parameters["mu"] = mu; //mean
  _dist_parameters["sigma"] = sigma; //sd
  if(not hasParameter("truncation")) {
    _dist_parameters["truncation"] = 1.0;
  }
  if(not hasParameter("xMin")) {
    _dist_parameters["xMin"] = -std::numeric_limits<double>::max( );
  }
  if(not hasParameter("xMax")) {
    _dist_parameters["xMax"] = std::numeric_limits<double>::max( );
  }
  //std::cout << "mu " << mu << " sigma " << sigma
  //          << " truncation " << _dist_parameters["truncation"]
  //          << " xMin " << _dist_parameters["xMin"]
  //          << " xMax " << _dist_parameters["xMax"] << std::endl;
  _backend = new NormalDistributionBackend(mu, sigma);
}

BasicNormalDistribution::BasicNormalDistribution(double mu, double sigma, double x_min, double x_max):
  BasicTruncatedDistribution(x_min,x_max)
{
  _dist_parameters["mu"] = mu; //mean
  _dist_parameters["sigma"] = sigma; //sd
  //if(not hasParameter("truncation")) {
  //  _dist_parameters["truncation"] = 1.0;
  //}
  //_dist_parameters["xMin"] = x_min;
  //_dist_parameters["xMax"] = x_max;
  //std::cout << "mu " << mu << " sigma " << sigma
  //          << " truncation " << _dist_parameters["truncation"]
  //          << " xMin " << _dist_parameters["xMin"]
  //          << " xMax " << _dist_parameters["xMax"] << std::endl;
  _backend = new NormalDistributionBackend(mu, sigma);

}


BasicNormalDistribution::~BasicNormalDistribution(){
  delete _backend;
}


double
BasicNormalDistribution::inverseCdf(double x){
  return BasicTruncatedDistribution::inverseCdf(x);
}

