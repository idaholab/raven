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
//#include "distribution_min.h"
#include <iostream>

enum PbFunctionType{PDF,CDF};

class distributionND;

class BasicDistributionND
{
public:
   BasicDistributionND();
   virtual ~BasicDistributionND();
   double  getVariable(std::string & variableName);                   	///< getVariable from mapping
   void updateVariable(std::string & variableName, double & newValue);
   virtual double  Pdf(std::vector<double> x) = 0;                           ///< Pdf function at coordinate x
   virtual double  Cdf(std::vector<double> x) = 0;                     ///< Cdf function at coordinate x
   virtual double  InverseCdf(std::vector<double> x) = 0;

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
  BasicMultiDimensionalInverseWeight(std::string data_filename,double p):  _interpolator(data_filename,p){};
  BasicMultiDimensionalInverseWeight(double p):  _interpolator(inverseDistanceWeigthing(p)){};
  virtual ~BasicMultiDimensionalInverseWeight(){};
  double  Pdf(std::vector<double> x) {return _interpolator.interpolateAt(x);};
  double  Cdf(std::vector<double> x) {return _interpolator.interpolateAt(x);};
  double  InverseCdf(std::vector<double> x){return -1.0;};
protected:
  inverseDistanceWeigthing  _interpolator;
};

class BasicMultiDimensionalScatteredMS: public virtual BasicDistributionND
{
public:
  BasicMultiDimensionalScatteredMS(std::string data_filename,double p,int precision): _interpolator(data_filename,p,precision){};
  BasicMultiDimensionalScatteredMS(double p,int precision): _interpolator(p,precision){};
  virtual ~BasicMultiDimensionalScatteredMS(){};
  double  Pdf(std::vector<double> x) {return _interpolator.interpolateAt(x);};
  double  Cdf(std::vector<double> x){return _interpolator.interpolateAt(x);};
  double  InverseCdf(std::vector<double> x){return -1.0;};
protected:
  microSphere _interpolator;
};

class BasicMultiDimensionalCartesianSpline: public  virtual BasicDistributionND
{
public:
  BasicMultiDimensionalCartesianSpline(std::string data_filename): _interpolator(data_filename){};
  BasicMultiDimensionalCartesianSpline(): _interpolator(){};
  virtual ~BasicMultiDimensionalCartesianSpline(){};
  double  Pdf(std::vector<double> x) {return _interpolator.interpolateAt(x);};
  double  Cdf(std::vector<double> x){return _interpolator.interpolateAt(x);};
  double  InverseCdf(std::vector<double> x){return -1.0;};
protected:
  NDspline _interpolator;
};



double DistributionPdf(BasicDistributionND & dist,std::vector<double> & x);
double DistributionCdf(BasicDistributionND & dist,std::vector<double> & x);
std::string getDistributionType(BasicDistributionND & dist);
double getDistributionVariable(BasicDistributionND & dist, std::string & variableName);
void DistributionUpdateVariable(BasicDistributionND & dist, std::string & variableName, double & newValue);

#endif /* DISTRIBUTION_BASE_ND_H_ */
