/*
 * distribution_base_ND.h
 *
 *  Created on: Feb 6, 2014
 *      Author: alfoa
 *
 */

#ifndef DISTRIBUTION_BASE_ND_H_
#define DISTRIBUTION_BASE_ND_H_
#include <map>
#include <string>
#include <vector>
#include "ND_Interpolation_Functions.h"
#include "distribution_min.h"

enum PbFunctionType{PDF,CDF};

class BasicDistributionND
{
public:
  BasicDistributionND();
   virtual ~BasicDistributionND();
   double  getVariable(std::string & variableName);                   	///< getVariable from mapping
   //std::vector<double>  getVariableVector(std::string  variableName);
   void updateVariable(std::string & variableName, double & newValue);
   virtual double  Pdf(std::vector<double> x) ;                           ///< Pdf function at coordinate x
   virtual double  Cdf(std::vector<double> x) ;                              ///< Cdf function at coordinate x
   virtual double  InverseCdf(std::vector<double> x) ;
   std::string & getType();

protected:
   std::string                   _type;                              ///< Distribution type
   std::string                   _data_filename;
   PbFunctionType                _function_type;
   std::map <std::string,double> _dis_parameters;
   bool                          _checkStatus;
};

class BasicMultiDimensionalInverseWeight: public virtual BasicDistributionND
{
public:
  BasicMultiDimensionalInverseWeight(std::string data_filename,double p): _interpolator(inverseDistanceWeigthing(data_filename,p)){};
  BasicMultiDimensionalInverseWeight(double p): _interpolator(inverseDistanceWeigthing(p)){};
  virtual ~BasicMultiDimensionalInverseWeight(){};
  double  Pdf(std::vector<double> x) {return _interpolator.interpolateAt(x);};
  double  Cdf(std::vector<double> x){return _interpolator.interpolateAt(x);};
  double  InverseCdf(std::vector<double> x){return -1.0;};
protected:
  inverseDistanceWeigthing _interpolator;
};

class BasicMultiDimensionalScatteredMS: public virtual BasicDistributionND
{
public:
  BasicMultiDimensionalScatteredMS(std::string data_filename,double p,int precision): _interpolator(microSphere(data_filename,p,precision)){};
  BasicMultiDimensionalScatteredMS(double p,int precision): _interpolator(microSphere(p,precision)){};
  virtual ~BasicMultiDimensionalScatteredMS(){};
  double  Pdf(std::vector<double> x) {return _interpolator.interpolateAt(x);};
  double  Cdf(std::vector<double> x){return _interpolator.interpolateAt(x);};
  double  InverseCdf(std::vector<double> x){return -1.0;};
protected:
  microSphere _interpolator;
};

class BasicMultiDimensionalCartesianSpline: public virtual BasicDistributionND
{
public:
  BasicMultiDimensionalCartesianSpline(std::string data_filename): _interpolator(NDspline(data_filename)){};
  BasicMultiDimensionalCartesianSpline(): _interpolator(NDspline()){};
  virtual ~BasicMultiDimensionalCartesianSpline(){};
  double  Pdf(std::vector<double> x) {return _interpolator.interpolateAt(x);};
  double  Cdf(std::vector<double> x){return _interpolator.interpolateAt(x);};
  double  InverseCdf(std::vector<double> x){return -1.0;};
protected:
  NDspline _interpolator;
};


#endif /* DISTRIBUTION_BASE_ND_H_ */
