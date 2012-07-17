/*
 * distribution_1D.h
 *
 *  Created on: Mar 22, 2012
 *      Author: MANDD
 *      References:
 *
 *      Tests		: None for the custom
 *
 *      Problems	: Gamma, Beta and Custom distributions have no RNG in place yet
 *      Issues		: None
 *      Complaints	: None
 *      Compliments	: None
 *
 */

#ifndef DISTRIBUTION_1D_H_
#define DISTRIBUTION_1D_H_

#include <string>
#include <vector>
#include "Interpolation_Functions.h"

enum distribution_type { UNIFORM_DISTRIBUTION=1, NORMAL_DISTRIBUTION=2,  LOG_NORMAL_DISTRIBUTION=3, WEIBULL_DISTRIBUTION=4, EXPONENTIAL_DISTRIBUTION=5, GAMMA_DISTRIBUTION=6, BETA_DISTRIBUTION=7, CUSTOM_DISTRIBUTION=8, TRIANGULAR_DISTRIBUTION=9 , DISTRIBUTION_ERROR=-1};
//enum custom_dist_fit_type {STEP=1,LINEAR=2,QUADRATIC=3,SPLINE=4};

class distribution_1D{

private:

  distribution_type _type;	// type of distribution: 1-uniform, 2-normal, 3-log-normal, 4-Weibull, 5-exponential, 6-gamma, 7-beta, 8-custom, 9-triangular
  // if type >9 return -1

  double _xMin;	// the distribution is defined over the interval [_xMin , xMax]
  double _xMax;	// i did it to speed the sampling process

  double _parameter1;		// generic parameters that correspond to specific parameters for each distributions
  double _parameter2;

  int _numberOfPoints;

  Interpolation_Functions _interpolation;


public:

  distribution_1D ();																	// constructor (default: uniform within [0,1])
  distribution_1D (distribution_type type, double min, double max, double param1, double param2);	// constructor 1
  distribution_1D (std::vector<double> x_coordinates, std::vector<double> y_coordinates, int numberPoints, custom_dist_fit_type fitting_type); // constructor 2
  ~distribution_1D ();																// destructor

  distribution_type getType () {return _type;};			// return type of distribution _type
  double getMin ();		// return limits of the interval over which the distribution is defined
  double getMax();
  double getParamater1();	// return _parameter1
  double getParameter2(); // return _parameter1

  void changeParameter1(double newParameter1);	// to change on the fly paramter1
  void changeParameter2(double newParameter2);	// to change on the fly paramter1

  double pdfCalc(double x);	// return pdf value of the distribution _type as function of the position x within [_xMin , xMax]
  double cdfCalc(double x);	// return cdf value of the distribution _type as function of the position x within [_xMin , xMax]
  double randGen();   // return a random number distributed accordingly to the distribution given a random number [0,1]

protected:

  double uniformPdf (double x);			// Uniform pdf
  double normalPdf (double x);			// Normal pdf
  double logNormalPdf (double x);		// Log-Normal pdf
  double weibullPdf (double x);			// Weibull pdf
  double betaPdf (double x);			// Beta pdf
  double exponentialPdf (double x);		// Exponential pdf
  double gammaPdf(double x);			// Gamma pdf
  double customPdf(double x);			// custom pdf
  double triangPdf(double x);			// Triangular pdf

  double uniformCdf(double x);			// uniform CDF
  double normalCdf (double x);			// normal CDF
  double logNormalCdf (double x);		// lognormal CDF
  double weibullCdf (double x);			// weibull CDF
  double betaCdf (double x);			// beta CDF
  double exponentialCdf (double x);		// exponential CDF
  double gammaCdf(double x);			// gamma CDF
  double customCdf(double x);        	// custom CDF
  double triangCdf(double x);			// Triangular CDF

  double uniformRandNumberGenerator();			// uniform random number generator
  double normalRandNumberGenerator();			// normal random number generator
  double logNormalRandNumberGenerator();		// log-normal random number generator
  double weibullRandNumberGenerator();			// weibull random number generator
  double exponentialRandNumberGenerator();		// exponential random number generator
  double triangularRandNumberGenerator();		// triangular random number generator
  double gammaRandNumberGenerator();			// gamma random number generator
  double betaRandNumberGenerator();				// beta random number generator
};


//  double gammaFunc(double x);
//	double betaInc(double a, double b, double x);
//	double gammaFunc(double x);
//	double gammp(double a, double x);
//	double betaFunc(double alpha, double beta);
//	void LoadData(double** data, int dimensionality, int cardinality, std::string filename);
double calculateCustomCDF(double position, double fitting, double** dataSet, int numberSamples);
double calculateCustomPdf(double position, double fitting, double** dataSet, int numberSamples);
double normRNG(double mu, double sigma);
//	double gammaRNG(double shape, double scale);
//	double betaRNG(double alpha, double beta);
double STDgammaRNG(double shape);
double rk_gauss();

#endif
