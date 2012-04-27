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


double calculateCustomPdf(double position, double fitting, double** dataSet, int numberSamples){
	double value=-1;
	double min;
	double max;

	for (int i=1; i<numberSamples; i++){
		max=dataSet[i][1];
		min=dataSet[i-1][1];

		if((position>min)&(position<max)){
			if (fitting==1)
				value=dataSet[i-1][2];
			else
				value=dataSet[i-1][2]+(dataSet[i][2]-dataSet[i-1][2])/(dataSet[i][1]-dataSet[i-1][1])*(position-dataSet[i-1][1]);
		}
		else
			perror ("The following error occurred: distribution sampled out of its boundaries");
	}

	return value;
}

double calculateCustomCDF(double position, double fitting, double** dataSet, int numberSamples){
	double value=-1;
	double min;
	double max;
	double cumulative=0;

	for (int i=1; i<numberSamples; i++){
		max=dataSet[i][1];
		min=dataSet[i-1][1];

		if((position>min)&(position<max)){
			if (fitting==1)
				value=cumulative+dataSet[i-1][2]*(position-dataSet[i-1][1]);
			else{
				double pdfValueInPosition =dataSet[i-1][2]+(dataSet[i][2]-dataSet[i-1][2])/(dataSet[i][1]-dataSet[i-1][1])*(position-dataSet[i-1][1]);
				value=cumulative + (pdfValueInPosition+dataSet[i-1][2])*(position-dataSet[i-1][1])/2;
			}
		}
		else
			perror ("The following error occurred: distribution sampled out of its boundaries");

		if (fitting==1)
			cumulative=cumulative+dataSet[i-1][2]*(dataSet[i][1]-dataSet[i-1][1]);
		else
			cumulative=cumulative+(dataSet[i][2]+dataSet[i-1][2])*(dataSet[i][1]-dataSet[i-1][1])/2;
	}

	return value;
}



#endif /* CUSTOMDIST_H_ */
