/*
 * customDist.h
 *
 *  Created on: Apr 17, 2012
 *      Author: MANDD
 */


#include <string>
#include <stdio.h>

#ifndef CUSTOMDIST_H_
#define CUSTOMDIST_H_


double calculateCustomPdf(double position, double fitting, double** dataSet, int numberSamples);


double calculateCustomCDF(double position, double fitting, double** dataSet, int numberSamples);

#endif /* CUSTOMDIST_H_ */
