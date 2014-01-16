#ifndef DISTRIBUTION_MIN_H_
#define DISTRIBUTION_MIN_H_

#include "distribution_type.h"

class BasicDistribution;

double getDistributionVariable(BasicDistribution & dist, std::string & variableName);
void DistributionUpdateVariable(BasicDistribution & dist, std::string & variableName, double & newValue);
double DistributionPdf(BasicDistribution & dist,double & x);
double DistributionCdf(BasicDistribution & dist,double & x);
double DistributionInverseCdf(BasicDistribution & dist, double & x);
std::string getDistributionType(BasicDistribution & dist);
std::vector<std::string> getDistributionVariableNames(BasicDistribution & dist);


#endif /* DISTRIBUTION_MIN_H_ */
