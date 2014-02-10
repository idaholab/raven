/*
 * distribution_ND.h
 *
 *  Created on: Feb 6, 2013
 *      Author: alfoa
 *
 */

#ifndef DISTRIBUTION_ND_H_
#define DISTRIBUTION_ND_H_

#include <string>
#include <vector>
#include "RavenObject.h"
#include "distribution_base_ND.h"


//enum PbFunctionType {PDF,CDF};

class distributionND;

template<>
InputParameters validParams<distributionND>();

class distributionND : public RavenObject, public BasicDistributionND
{
 public:
   //> constructor for built-in distributions
  distributionND(const std::string & name, InputParameters parameters);

  virtual ~distributionND();

};


/*
 * CLASS MultiDimensionalInverseWeight DISTRIBUTION
 */
class MultiDimensionalInverseWeight;

template<>
InputParameters validParams<MultiDimensionalInverseWeight>();


class MultiDimensionalInverseWeight : public distributionND {
public:
  MultiDimensionalInverseWeight(const std::string & name, InputParameters parameters);
  virtual ~MultiDimensionalInverseWeight();

};

/*
 * CLASS MultiDimensionalScatteredMS DISTRIBUTION
 */

class MultiDimensionalScatteredMS;

template<>
InputParameters validParams<MultiDimensionalScatteredMS>();


class MultiDimensionalScatteredMS : public distributionND {
public:
  MultiDimensionalScatteredMS(const std::string & name, InputParameters parameters);
  virtual ~MultiDimensionalScatteredMS();

};

/*
 * CLASS MultiDimensionalCartesianSpline DISTRIBUTION
 */
class MultiDimensionalCartesianSpline;

template<>
InputParameters validParams<MultiDimensionalCartesianSpline>();


class MultiDimensionalCartesianSpline : public distributionND {
public:
  MultiDimensionalCartesianSpline(const std::string & name, InputParameters parameters);
  virtual ~MultiDimensionalCartesianSpline();

};

#endif
