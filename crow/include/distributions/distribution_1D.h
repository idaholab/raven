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
#ifndef DISTRIBUTION_1D_H
#define DISTRIBUTION_1D_H

#include <string>
#include <vector>
#include "distribution.h"
#include "distributionFunctions.h"

class DistributionBackend;

class BasicTruncatedDistribution : public virtual BasicDistribution
{
public:
  BasicTruncatedDistribution(double x_min, double x_max);
  BasicTruncatedDistribution(){};
  virtual double pdf(double x);        ///< pdf function at coordinate x
  virtual double cdf(double x);        ///< cdf function at coordinate x
  virtual double inverseCdf(double x); ///< x

  virtual double untrPdf(double x);
  virtual double untrCdf(double x);
  virtual double untrCdfComplement(double x);
  virtual double untrInverseCdf(double x);
  virtual double untrMean();
  virtual double untrStdDev();
  virtual double untrMedian();
  virtual double untrMode();
  virtual double untrHazard(double x);
protected:
  DistributionBackend * _backend;
};

class BasicDiscreteDistribution : public virtual BasicDistribution
{
public:
  virtual double pdf(double x);           ///< pdf function at coordinate x
  virtual double cdf(double x);           ///< cdf function at coordinate x
  virtual double inverseCdf(double x);    ///< x

  virtual double untrPdf(double x);
  virtual double untrCdf(double x);
  virtual double untrCdfComplement(double x);
  virtual double untrInverseCdf(double x);
  virtual double untrMean();
  virtual double untrStdDev();
  virtual double untrMedian();
  virtual double untrMode();
  virtual double untrHazard(double x);
protected:
  DistributionBackend * _backend;
};

/*
 * CLASS UNIFORM DISTRIBUTION
 */
class UniformDistribution;


class UniformDistributionBackend;

class BasicUniformDistribution : public BasicTruncatedDistribution
{
public:
  BasicUniformDistribution(double x_min, double x_max);
  virtual ~BasicUniformDistribution();
  double  pdf(double x);                ///< pdf function at coordinate x
  double  cdf(double x);                ///< cdf function at coordinate x
  double  inverseCdf(double x);        ///< x
};


/*
 * CLASS NORMAL DISTRIBUTION
 */
class NormalDistribution;


class NormalDistributionBackend;

class BasicNormalDistribution : public  BasicTruncatedDistribution
{
public:
  BasicNormalDistribution(double mu, double sigma);
  BasicNormalDistribution(double mu, double sigma, double x_min, double x_max);
  virtual ~BasicNormalDistribution();

  double  inverseCdf(double x);        ///< x
};



#endif
