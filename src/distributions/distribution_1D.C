/*
 * distribution_1D.cpp
 *
 *  Created on: Mar 22, 2012
 *      Author: MANDD
 *      References:
 *      1- G. Cassella, R.G. Berger, "Statistical Inference", 2nd ed. Pacific Grove, CA: Duxbury Press (2001).
 *
 *      Tests		: None for the custom
 *
 *      Problems	: None
 *      Issues		: None
 *      Complaints	: None
 *      Compliments	: None
 *
 */

#include "distribution_1D.h"
#include <math.h>
#include <cmath>					// needed to use erfc error function
#include <string>
#include "dynamicArray.h"
#include <ctime>
#include <cstdlib>


#define _USE_MATH_DEFINES	// needed in order to use M_PI = 3.14159

	distribution_1D::distribution_1D (){
		_type=UNIFORM_DISTRIBUTION;	// Default: uniform distribution
		_xMin=0;
		_xMax=1;
		_parameter1=1;
		_parameter2=1;
		srand ( time(NULL) );
	}

	distribution_1D::~distribution_1D (){
	}

	distribution_1D::distribution_1D (distribution_type type, double min, double max, double param1, double param2){
		_type=type;
		_xMin=min;
		_xMax=max;
		_parameter1=param1;
		_parameter2=param2;
		srand ( time(NULL) );
	}

	distribution_1D::distribution_1D (std::vector<double> x_coordinates, std::vector<double> y_coordinates, int numberPoints, custom_dist_fit_type fitting_type){
		_type=CUSTOM_DISTRIBUTION;
		_xMin=x_coordinates[0];
		_xMax=x_coordinates[x_coordinates.size()-1];

		_numberOfPoints= numberPoints;

		_interpolation=Interpolation_Functions(x_coordinates, y_coordinates, numberPoints, fitting_type);

		srand ( time(NULL) );
	}


	double distribution_1D::getMin (){
		return _xMin;
	}

	double distribution_1D::getMax(){
		return _xMax;
	}

	double distribution_1D::getParamater1(){
		return _parameter1;
	}

	double distribution_1D::getParameter2(){
		return _parameter2;
	}

	void distribution_1D::changeParameter1(double newParameter1){
		_parameter1=newParameter1;
	}

	void distribution_1D::changeParameter2(double newParameter2){
		_parameter2=newParameter2;
	}

	// return pdf coordinate
		double distribution_1D::pdfCalc(double x){
			double pdfValue;

			if ((x >= _xMin)&&(x <= _xMax))
				switch (_type) {
				  case UNIFORM_DISTRIBUTION:  // Uniform
					  pdfValue=uniformPdf(x);
					break;
				  case NORMAL_DISTRIBUTION:  // Normal
					  pdfValue=normalPdf(x);
					break;
				  case LOG_NORMAL_DISTRIBUTION:  // Lognormal
					  pdfValue=logNormalPdf(x);
					break;
				  case WEIBULL_DISTRIBUTION:  // Weibull
					  pdfValue=weibullPdf(x);
					break;
				  case EXPONENTIAL_DISTRIBUTION:  // Exponential
					  pdfValue=exponentialPdf(x);
					break;
				  case GAMMA_DISTRIBUTION:  // Gamma
					  pdfValue=gammaPdf(x);
					break;
				  case BETA_DISTRIBUTION:  // Beta
					  pdfValue=betaPdf(x);
					break;
				  case CUSTOM_DISTRIBUTION:  // custom
					  pdfValue=customPdf(x);
					break;
				  case TRIANGULAR_DISTRIBUTION:	// triangular
					  pdfValue=triangPdf(x);
					break;
				  default:	// otherwise return error pdfvalue =-1;
					  pdfValue=-1;
					break;
			  }
			else
				pdfValue=-1;

			return pdfValue;
		}

		// return cdf coordinate
			double distribution_1D::cdfCalc(double x){
				double pdfValue;

				if ((x >= _xMin)&&(x <= _xMax))
					switch (_type) {
					  case UNIFORM_DISTRIBUTION:  // Uniform
						  pdfValue=uniformCdf(x);
						break;
					  case NORMAL_DISTRIBUTION:  // Normal
						  pdfValue=normalCdf(x);
						break;
					  case LOG_NORMAL_DISTRIBUTION:  // Lognormal
						  pdfValue=logNormalCdf(x);
						break;
					  case WEIBULL_DISTRIBUTION:  // Weibull
						  pdfValue=weibullCdf(x);
						break;
					  case EXPONENTIAL_DISTRIBUTION:  // Exponential
						  pdfValue=exponentialCdf(x);
						break;
					  case GAMMA_DISTRIBUTION:  // Gamma
						  pdfValue=gammaCdf(x);
						break;
					  case BETA_DISTRIBUTION:  // Beta
						  pdfValue=betaCdf(x);
						break;
					  case CUSTOM_DISTRIBUTION:  // Custom
						  pdfValue=customCdf(x);
						break;
					  case TRIANGULAR_DISTRIBUTION:	// triangular
						  pdfValue=triangCdf(x);
						break;
					  default:	// otherwise return error pdfvalue =-1;
						  pdfValue=-1;
						break;
				  }
				else
					pdfValue=-1;

				return pdfValue;
			}

	// return random number distributed accordingly to that distribution
		double distribution_1D::randGen(){
			double RandomNumberValue;

			for(int i=0; i<5; i++){
				switch (_type) {
				  case 1:  // Uniform
					  RandomNumberValue=uniformRandNumberGenerator();
					break;
				  case 2:  // Normal
					  RandomNumberValue=normalRandNumberGenerator();
					break;
				  case 3:  // Log-normal
					  RandomNumberValue=logNormalRandNumberGenerator();
					break;
				  case 4:  // Weibull
					  RandomNumberValue=weibullRandNumberGenerator();
					break;
				  case 5:  // Exponential
					  RandomNumberValue=exponentialRandNumberGenerator();
					break;
				  case 6:  // Gamma
					  RandomNumberValue=gammaRandNumberGenerator();
					break;
				  case 7:  // Beta
					  RandomNumberValue=betaRandNumberGenerator();
					break;
				  case 9:  // Triangular
					  RandomNumberValue=triangularRandNumberGenerator();
					  break;
				  default:	// otherwise return error pdfvalue =-1;
					  RandomNumberValue=-1;
					break;
				}
			}

			return RandomNumberValue;
		}

	// Uniform pdf
		double distribution_1D::uniformPdf (double x){
			// _xMin<x<_xMax

			double value;

			value = 1/(_xMax-_xMin);

			return value;
		}

		double distribution_1D::uniformCdf(double x){
			double value;

			value=1/(_xMax-_xMin)*(x-_xMin);

			return value;
		}

	// Normal pdf
		double distribution_1D::normalPdf (double x){
			// parameter1=mu
			// parameter2=sigma	>0

			double value;

			if (_parameter2 > 0)
				value = 1/(sqrt(2.0*M_PI*_parameter2*_parameter2))*exp(-(x-_parameter1)*(x-_parameter1)/(2.0*_parameter2*_parameter2));
			else
				value=-1;

			return value;
		}

		double distribution_1D::normalCdf (double x){
			// parameter1=mu
			// parameter2=sigma	>0

			double value;

			if (_parameter2 > 0)
				value = 1/2 + erf((x-_parameter1)/sqrt(2*_parameter2*_parameter2));
			else
				value=-1;

			return value;
		}

	// Log-Normal pdf
		double distribution_1D::logNormalPdf (double x){
			// parameter1=mu
			// parameter2=sigma  >0

			double value;

			if (_parameter2 > 0){
				if (x == 0)
					value = 0;
				else
				value= 1/(x*sqrt(2*M_PI*_parameter2*_parameter2))*exp(-(log(x)-_parameter1)*(log(x)-_parameter1)/(2*_parameter2*_parameter2));
			}
			else
				value=-1;

			return value;
		}

		double distribution_1D::logNormalCdf (double x){
			// parameter1=mu
			// parameter2=sigma  >0

			double value;

			if (_parameter2 > 0)
				value = 0.5 + 0.5 * erf((log(x)-_parameter1)/sqrt(2.0*_parameter2*_parameter2));
			else
				value=-1;

			return value;
		}

	// Weibull pdf
		double distribution_1D::weibullPdf (double x){
			// parameter1=k   		>0     (source wikipedia: http://en.wikipedia.org/wiki/Weibull_distribution)
			// parameter2=lambda     >0
			// x>0

			double value;

			if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value = _parameter1/_parameter2*pow(x/_parameter2,_parameter1-1)*exp(-pow(x/_parameter2, _parameter1));
			else
				value=-1;

			return value;
		}

		double distribution_1D::weibullCdf (double x){
			// parameter1=k   >0
			// parameter2=lambda     >0
			// x>0

			double value;

			if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value = 1-exp(-pow(x/_parameter2, _parameter1));
			else
				value=-1;

			return value;
		}

	// Beta pdf
		double distribution_1D::betaPdf (double x){
			// parameter1=alpha   >0
			// parameter2=beta    >0
			// 0<x<1

			double value;

			/*if ((x > 0)&&(x < 1)&&(_parameter1 > 0)&&(_parameter2 > 0))
              value = 1/betaFunc(_parameter1,_parameter2)*pow(x,_parameter1-1)*pow(1-x,_parameter2-1);
              else */
				value=-1;

			return value;
		}

		double distribution_1D::betaCdf (double x){
			// parameter1=alpha   >0
			// parameter2=beta    >0
			// 0<x<1

			double value;

			/*if ((x > 0)&&(x < 1)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value = betaInc(_parameter1,_parameter2 ,x);
                else */
				value=-1;

			return value;
		}

	// Exponential pdf
		double distribution_1D::exponentialPdf (double x){
			// parameter1=lamda	>0
			// parameter2 not used
			// x>=0

			double value;

			if ((x >= 0.0)&&(_parameter1 > 0.0))
				value = _parameter1*exp(-x*_parameter1);
			else
				value=-1;

			return value;
		}

		double distribution_1D::exponentialCdf (double x){
			// parameter1=lambda	>0
			// parameter2 not used
			// x>=0

			double value;

			if ((x >= 0)&&(_parameter1 > 0))
				value = 1-exp(-x*_parameter1);
			else
				value=-1;

			return value;
		}

	// Gamma pdf
		double distribution_1D::gammaPdf(double x){
			// parameter1= k   >0
			// parameter2= theta     >0
			// x>=0

			double value;

			/* if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value=1/gammaFunc(_parameter1)/pow(_parameter2,_parameter1)*pow(x,_parameter1-1)*exp(-x/_parameter2);
                else */
				value=1;

			return value;
		}

		double distribution_1D::gammaCdf(double x){
			// parameter1=alpha, k   >0
			// parameter2=beta, theta     >0
			// x>=0

			double value;

			/* if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
              value= gammp(_parameter1,x/_parameter2);
              else */
				value=1;

			return value;
		}

	// Custom pdf
		double distribution_1D::customPdf(double x){
			double value=-1;
			return value;
		}

		double distribution_1D::customCdf(double x){
			double value=-1;
			return value;
		}

	// triangular pdf
		double distribution_1D::triangPdf(double x){
			// parameter1= peak coordinate
			// parameter2 not used

			double value;

			if ((_parameter1>_xMin)&&(_parameter1<_xMax)){
				if (x<=_parameter1)
					value=2*(x-_xMin)/(_xMax-_xMin)/(_parameter1-_xMin);
				else
					value=2*(_xMax-x)/(_xMax-_xMin)/(_xMax-_parameter1);
			}
			else
				value=-1;

			return value;
		}


		double distribution_1D::triangCdf(double x){
			// parameter1= peak coordinate
			// parameter2 not used

			double value;

			if ((_parameter1>_xMin)&&(_parameter1<_xMax)){
				if (x<=_parameter1)
					value=(x-_xMin)*(x-_xMin)/(_xMax-_xMin)/(_parameter1-_xMin);
				else
					value=1-(_xMax-x)*(_xMax-x)/(_xMax-_xMin)/(_xMax-_parameter1);
			}
			else
				value=-1;

			return value;
		}
		

		double distribution_1D::uniformRandNumberGenerator(){
			double value;
			double RNG=rand()/double(RAND_MAX);
			return value=_xMin+RNG*(_xMax-_xMin);
		}

		double distribution_1D::normalRandNumberGenerator(){
          double value=normRNG(_parameter1, _parameter2);
			return value;
		}

		double distribution_1D::logNormalRandNumberGenerator(){
			double value;
			return value=-1;//exp(normRNG(_parameter1, _parameter2));
		}

		double distribution_1D::exponentialRandNumberGenerator(){
			double value=-1/_parameter1 * log(1.0 - rand()/double(RAND_MAX));
			return value;
		}

		double distribution_1D::weibullRandNumberGenerator(){
			double value=-_parameter2 * pow(log(1.0 - rand()/double(RAND_MAX)),1/_parameter2);
			return value;
		}

		double distribution_1D::gammaRandNumberGenerator(){
          double value=-1;//gammaRNG(_parameter1,_parameter2);
			return value;
		}

		double distribution_1D::betaRandNumberGenerator(){
          double value=-1;//betaRNG(_parameter1,_parameter2);
			return value;
		}

		double distribution_1D::triangularRandNumberGenerator(){
			double value;
			double RNG = rand()/double(RAND_MAX);
			double referenceValue=(_parameter1-_xMin)/(_xMax-_xMin);

			if (RNG<referenceValue)
				value= _xMin+sqrt(RNG*(_parameter1-_xMin)*(_xMax-_xMin));
			else
				value=_xMax-sqrt((1-RNG)*(_xMax-_parameter1)*(_xMax-_xMin));
			return value;
		}
