#ifndef DISTRIBUTION_MIN_H_
#define DISTRIBUTION_MIN_H_

#include "distribution_type.h"

class distribution;


double getDistributionVariable(distribution & dist, std::string & variableName);
void DistributionUpdateVariable(distribution & dist, std::string & variableName, double & newValue);
double DistributionPdf(distribution & dist,double & x);
double DistributionCdf(distribution & dist,double & x);
double DistributionRandomNumberGenerator(distribution & dist);
std::string getDistributionType(distribution & dist);
std::vector<std::string> getDistributionVariableNames(distribution & dist);


#endif /* DISTRIBUTION_MIN_H_ */
