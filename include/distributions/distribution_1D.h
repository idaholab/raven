/*
 * distribution_1D.h
 *
 *  Created on: Apr 5, 2012
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

#include <string>

#ifndef DISTRIBUTION_1D_H_
#define DISTRIBUTION_1D_H_

using namespace std;

class distribution_1D{
																														//    _____ overload exists: a problem for SWIG?
	private:																											//    |
																														//    V
		int _type;	// type of distribution: 1-uniform, 2-normal, 3-lognormal, 4-Weibull, 5-exponential, 6-gamma, 7-beta, 8-custom
					// if type >8 return -1

		double _xMin;	// the distribution is defined over the interval [_xMin , xMax]
		double _xMax;	// i did it to speed the sampling process

		double _parameter1;		// generic parameters that correspond to specific parameters for each distributions
		double _parameter2;

		string _filename;

	public:

		distribution_1D ();																	// constructor (default: uniform within [0,1])
		distribution_1D (int type, double min, double max, double param1, double param2);	// constructor 1
		distribution_1D (int type, double min, double max, double param1, double param2, string fileName); // constructor 2
		~distribution_1D ();																	// destructor

		int getType ();			// return type of distribution _type
		double getMin ();		// return limits of the interval over which the distribution is defined
		double getMax();
		double getParamater1();	// return _parameter1
		double getParameter2(); // return _parameter1

		void changeParameter1(double newParameter1);	// to change on the fly paramter1
		void changeParameter2(double newParameter2);	// to change on the fly paramter1

		double pdfCalc(double x);	// return pdf value of the distribution _type as function of the position x within [_xMin , xMax]
		double cdfCalc(double x);	// return cdf value of the distribution _type as function of the position x within [_xMin , xMax]
        double randGen() { return 1/0;};   // return a random number distributed accordingly to the distribution given a random number [0,1]

	protected:

		double uniformPdf (double x);		// Uniform pdf
		double normalPdf (double x);		// Normal pdf
		double logNormalPdf (double x);		// Log-Normal pdf
		double weibullPdf (double x);		// Weibull pdf
		double betaPdf (double x);			// Beta pdf
		double exponentialPdf (double x);	// Exponential pdf
		double gammaPdf(double x);			// Gamma pdf
		double customPdf(double x);			// custom pdf

		double uniformCdf(double x);		// uniform CDF
		double normalCdf (double x);		// normal CDF
		double logNormalCdf (double x);		// lognormal CDF
		double weibullCdf (double x);		// weibull CDF
		double betaCdf (double x);			// beta CDF
		double exponentialCdf (double x);	// exponential CDF
		double gammaCdf(double x);			// gamma CDF
		double customCdf(double x);         // custom CDF

		double NormalRandNumberGenerator();		//random normal generator
};





#endif /* DISTRIBUTION_1D_H_ */
