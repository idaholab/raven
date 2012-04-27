/*
 * distribution_ND.h
 *
 *  Created on: Mar 27, 2012
 *      Author: MANDD
 *
 *      This class define a N-dimensional distribution function composed by N uncorrelated 1-D distribution functions
 *      References:
 *      1- G. Cassella, R.G. Berger, "Statistical Inference", 2nd ed. Pacific Grove, CA: Duxbury Press (2001).
 *
 *      Tests		: None
 *
 *      Problems	: None
 *      Issues		: None
 *      Complaints	: None
 *      Compliments	: None
 *
 */

#include <vector>
#include "distribution_1D.h"

using namespace std;

#ifndef DISTRIBUTION_ND_H_
#define DISTRIBUTION_ND_H_


class distribution_ND:distribution_1D{
	private:
		int _Dimensionality;			// dimensionality of the distribution
		std::vector<distribution_1D> _N_1Ddistribution;		// the distribution is coded as a vector of 1-dimensional pdf

	public:
		distribution_ND();
		~distribution_ND();
		distribution_ND(int dimensionality, std::vector<distribution_1D> N_1Ddistribution);
		int get_Dimensionality();

		double pdfCalcND(vector<double>& coordinate);
		double cdfCalcND(vector<double>& coordinate);
};

#endif /* PDFND_H_ */
