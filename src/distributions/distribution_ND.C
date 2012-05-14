/*
 * distribution_ND.cpp
 *
 *  Created on: Mar 27, 2012
 *      Author: MANDD
 *
 *      Tests		: None
 *
 *      Problems	: None
 *      Issues		: None
 *      Complaints	: None
 *      Compliments	: None
 *
 */

#include "distribution_ND.h"
#include "distribution_1D.h"
#include <vector>


//constructor 1
distribution_ND::distribution_ND (){
	_Dimensionality = 1;
}

//destructor
distribution_ND::~distribution_ND (){

}

// constructor 2
distribution_ND::distribution_ND(int dimensionality, std::vector<distribution_1D> N_1Ddistribution){
	_Dimensionality= dimensionality;
	_N_1Ddistribution = N_1Ddistribution;
}

int distribution_ND::get_Dimensionality(){
	return _Dimensionality;
}

double distribution_ND::pdfCalcND(vector<double>& coordinate){
	double value=1;
	for(int i=1; i<_Dimensionality; i++){
		value *= _N_1Ddistribution[i].pdfCalc(coordinate[i]);
	}
	return value;
}

double distribution_ND::cdfCalcND(vector<double>& coordinate){
	double value=1;
	for(int i=1; i<_Dimensionality; i++){
		value *= _N_1Ddistribution[i].cdfCalc(coordinate[i]);
	}
	return value;
}
