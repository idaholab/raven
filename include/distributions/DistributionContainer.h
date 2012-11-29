/*
 * DistributionContainer.h
 *
 *  Created on: Jul 6, 2012
 *      Author: alfoa
 */

#ifndef DISTRIBUTIONCONTAINER_H_
#define DISTRIBUTIONCONTAINER_H_

//#include "distribution_1D.h"
#include "distribution_type.h"
#include <iostream>
#include <vector>
#include <map>
#include "distribution.h"
#include "Interpolation_Functions.h"

using namespace std;



class DistributionContainer;

//class distribution;

class InputParameters;

class DistributionContainer{
     public:
     static DistributionContainer & Instance();
     /*
      * Function to construct on the fly this class through the action system
      * @
      * @
      */
     //void addDistributionInContainer(const std::string & type, const std::string & name, InputParameters params);
     void addDistributionInContainer(const std::string & type, const std::string & name, distribution * dist); 

     void seedRandom(unsigned int seed);

//     void constructDistributionContainer(std::string DistAlias,
//                                           distribution_type type,
//                                           double xmin, double xmax,
//                                           double param1,
//                                           double param2,
//                                           unsigned int seed);
//     /*
//      * Function to construct on the fly this class through the action system
//      * @
//      * @
//      */
//     void constructDistributionContainerCustom(std::string DistAlias, distribution_type type,
//                                                 std::vector< double > dist_x,
//                                                 std::vector< double > dist_y,
//                                                 int numPoints,
//                                                 custom_dist_fit_type fit_type,
//                                                 unsigned int seed);

     /*
      * Function to get the enum of the distribution called DistAlias
      * @ DistAlias, alias of the distribution from which retrieving the parameter
      */
     std::string  getType (std::string DistAlias);


     double getVariable(std::string paramName,std::string DistAlias);

     void updateVariable(std::string paramName,double newValue,std::string DistAlias);

//     /*
//      * Function to get the lower limit of the interval over which the distribution is defined
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      */
//     double getMin (std::string DistAlias);          // return limits of the interval over which the distribution is defined
//     /*
//      * Function to get the upper limit of the interval over which the distribution is defined
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      */
//     double getMax(std::string DistAlias);
//     /*
//      * Function to get the first characteristic parameter of the distribution (ex. "Mu")
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      */
//     double getParamater1(std::string DistAlias);     // return _parameter1
//     /*
//      * Function to get the second characteristic parameter of the distribution (ex. "Sigma")
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      */
//     double getParameter2(std::string DistAlias); // return _parameter1
//     /*
//      * Function to change on the fly the first characteristic parameter of the distribution (ex. "Mu")
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      * @ newParameter1, value of the parameter
//      */
//     void changeParameter1(std::string DistAlias, double newParameter1);     // to change on the fly paramter1
//     /*
//      * Function to change on the fly the second characteristic parameter of the distribution (ex. "Sigma")
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      * @ newParameter2, value of the parameter
//      */
//     void changeParameter2(std::string DistAlias, double newParameter2);     // to change on the fly paramter1
//     /*
//      * Function to get the pdf value of the distribution called "DistAlias"
//      * as function of the position x within [_xMin , xMax]
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      * @ x, position
//      */
     double Pdf(std::string DistAlias, double x);     // return pdf value of the distribution _type as function of the position x within [_xMin , xMax]
     /*
      * Function to get the cdf value of the distribution called "DistAlias"
      * as function of the position x within [_xMin , xMax]
      * @ DistAlias, alias of the distribution from which retrieving the parameter
      * @ x, position
      */
     double Cdf(std::string DistAlias, double x);     // return cdf value of the distribution _type as function of the position x within [_xMin , xMax]
     /*
      * Function to get a random number distributed accordingly to the distribution
      * given a random number [0,1]
      * @ DistAlias, alias of the distribution from which retrieving the parameter
      */
     double randGen(std::string DistAlias, double RNG);   // return a random number distributed accordingly to the distribution given a random number [0,1]

     /*
      * Function to get a random number distributed accordingly to the distribution
      * given a random number [0,1]
      * @ DistAlias, alias of the distribution from which retrieving the parameter
      */
     double randGen(char * DistAlias, double RNG);   // return a random number distributed accordingly to the distribution given a random number [0,1]

     double random(); // return a random number

     bool checkCdf(double probability, std::vector<double> values);

     bool checkCdf(double probability, double value);

     protected:
     std::map < std::string, int > _vector_pos_map;
     std::vector < distribution * > _distribution_cont;
     /// mapping from distribution name and distribution
     std::map<std::string, distribution *> _dist_by_name;
     /// "Buckets" of distribution based on their types
     std::map<std::string, std::vector<distribution *> > _dist_by_type;
//     /*
//      * Function to get the position in the internal mapping
//      * @ DistAlias, alias of the distribution from which retrieving the parameter
//      */
//     int getPosition(std::string DistAlias);

     /*
      * Constructor(empty)
      */
     DistributionContainer();
     /*
      * Destructor
      */
     virtual ~DistributionContainer();
     static DistributionContainer * _instance; // = NULL 
};

std::string * str_to_string_p(char *s);
void free_string_p(std::string * s);

#endif /* DISTRIBUTIONCONTAINER_H_ */
