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
#include "dynamicArray.h"
#include "inputFile.h"
#include "customDist.h"
#include <math.h>
#include <cmath>					// needed to use erfc error function
#include "beta_gamma_Func.h"		// this file contains auxiliary functions for the beta and gamma distributions

#define _USE_MATH_DEFINES	// needed in order to use M_PI = 3.14159

	distribution_1D::distribution_1D (){
		_type=1;	// Default: uniform distribution
		_xMin=0;
		_xMax=1;
		_parameter1=1;
		_parameter2=1;
	}

	distribution_1D::~distribution_1D (){
	}

	distribution_1D::distribution_1D (int type, double min, double max, double param1, double param2){
		_type=type;
		_xMin=min;
		_xMax=max;
		_parameter1=param1;
		_parameter2=param2;
	}

	distribution_1D::distribution_1D (int type, double min, double max, double param1, double param2, string fileName){
		_type=type;
		_xMin=min;
		_xMax=max;
		_filename=fileName;
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
				  case 1:  // Uniform
					  pdfValue=uniformPdf(x);
					break;
				  case 2:  // Normal
					  pdfValue=normalPdf(x);
					break;
				  case 3:  // Lognormal
					  pdfValue=logNormalPdf(x);
					break;
				  case 4:  // Weibull
					  pdfValue=weibullPdf(x);
					break;
				  case 5:  // Exponential
					  pdfValue=exponentialPdf(x);
					break;
				  case 6:  // Gamma
					  pdfValue=gammaPdf(x);
					break;
				  case 7:  // Beta
					  pdfValue=betaPdf(x);
					break;
				  case 8:  // custom
					  pdfValue=customPdf(x);
					break;
				  default:	// otherwise return error pdfvalue =-1;
					  pdfValue=-1;
					break;
			  }
			else
				pdfValue=-1;


			return pdfValue;
		}

		// return pdf coordinate
			double distribution_1D::cdfCalc(double x){
				double pdfValue;

				if ((x >= _xMin)&&(x <= _xMax))
					switch (_type) {
					  case 1:  // Uniform
						  pdfValue=uniformCdf(x);
						break;
					  case 2:  // Normal
						  pdfValue=normalCdf(x);
						break;
					  case 3:  // Lognormal
						  pdfValue=logNormalCdf(x);
						break;
					  case 4:  // Weibull
						  pdfValue=weibullCdf(x);
						break;
					  case 5:  // Exponential
						  pdfValue=exponentialCdf(x);
						break;
					  case 6:  // Gamma
						  pdfValue=gammaCdf(x);
						break;
					  case 7:  // Beta
						  pdfValue=betaCdf(x);
						break;
					  case 8:  // Custom
						  pdfValue=customCdf(x);
						break;
					  default:	// otherwise return error pdfvalue =-1;
						  pdfValue=-1;
						break;
				  }
				else
					pdfValue=-1;


				return pdfValue;
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
			// parameter2=sigma^2	>0

			double value;

			if (_parameter2 > 0)
				value = 1/(sqrt(2.0*M_PI*_parameter2))*exp(-(x-_parameter1)*(x-_parameter1)/(2.0*_parameter2));
			else
				value=-1;

			return value;
		}

		double distribution_1D::normalCdf (double x){
			// parameter1=mu
			// parameter2=sigma^2	>0

			double value;

			if (_parameter2 > 0)
				value = 1/2 + erf((x-_parameter1)/sqrt(2*_parameter2));
			else
				value=-1;

			return value;
		}

	// Log-Normal pdf
		double distribution_1D::logNormalPdf (double x){
			// parameter1=mu
			// parameter2=sigma^2  >0

			double value;

			if (_parameter2 > 0){
				if (x == 0)
					value = 0;
				else
				value= 1/(x*sqrt(2*M_PI*_parameter2))*exp(-(log(x)-_parameter1)*(log(x)-_parameter1)/(2*_parameter2));
			}
			else
				value=-1;

			return value;
		}

		double distribution_1D::logNormalCdf (double x){
			// parameter1=mu
			// parameter2=sigma^2  >0

			double value;

			if (_parameter2 > 0)
				value = 0.5 + 0.5 * erf((log(x)-_parameter1)/sqrt(2.0*_parameter2));
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

			if ((x > 0)&&(x < 1)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value = 1/betaFunc(_parameter1,_parameter2)*pow(x,_parameter1-1)*pow(1-x,_parameter2-1);
			else
				value=-1;

			return value;
		}

		double distribution_1D::betaCdf (double x){
			// parameter1=alpha   >0
			// parameter2=beta    >0
			// 0<x<1

			double value;

			if ((x > 0)&&(x < 1)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value = betaInc(_parameter1,_parameter2 ,x);
			else
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

			if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value=1/gammaFunc(_parameter1)/pow(_parameter2,_parameter1)*pow(x,_parameter1-1)*exp(-x/_parameter2);
			else
				value=1;

			return value;
		}

		double distribution_1D::gammaCdf(double x){
			// parameter1=alpha, k   >0
			// parameter2=beta, theta     >0
			// x>=0

			double value;

			if ((x >= 0)&&(_parameter1 > 0)&&(_parameter2 > 0))
				value= gammp(_parameter1,x/_parameter2);
			else
				value=1;

			return value;
		}

	// Custom pdf
		double distribution_1D::customPdf(double x){
			// parameter1= number of samples
			// parameter2= type of interpolation (0- constant, 1-linear)

			double value;

			double **dataSet = AllocateDynamicArray<double>((int)round(_parameter1),2);

			// retrieve data from file
			LoadData(dataSet, 2, (int)round(_parameter1), _filename);

			// calculate value
			value=calculateCustomPdf(x,_parameter2, dataSet,(int)round(_parameter1));

			return value;
		}

		double distribution_1D::customCdf(double x){
			// parameter1= number of samples
			// parameter2= type of interpolation (0- constant, 1-linear)

			double value;

			double **dataSet = AllocateDynamicArray<double>((int)round(_parameter1),2);

			// retrieve data from file
			LoadData(dataSet, 2, (int)round(_parameter1), _filename);

			// calculate value
			value=calculateCustomCDF(x,_parameter2, dataSet,(int)round(_parameter1));

			return value;
		}
