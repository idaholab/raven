/*
 * DistributionContainer.C
 *
 *  Created on: Jul 6, 2012
 *      Author: alfoa
 */
#include "DistributionContainer.h"
//#include "Factory.h"
//#include "distribution_1D.h"
#include "distribution_min.h"
#include <iostream>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <map>
using namespace std;

#ifndef mooseError
#define mooseError(msg) { std::cerr << "\n\n" << msg << "\n\n"; }
#endif

class DistributionContainer;

DistributionContainer::DistributionContainer()
{
}
DistributionContainer::~DistributionContainer()
{
}

/*void
DistributionContainer::addDistributionInContainer(const std::string & type, const std::string & name, InputParameters params){
   // create the distribution type
   distribution * dist = dynamic_cast<distribution *>(Factory::instance()->create(type, name, params));
   if (_dist_by_name.find(name) == _dist_by_name.end())
    _dist_by_name[name] = dist;
   else
     mooseError("Distribution with name " << name << " already exists");

   _dist_by_type[type].push_back(dist);

   }*/

void
DistributionContainer::addDistributionInContainer(const std::string & type, const std::string & name, distribution * dist){
   // create the distribution type
  //distribution * dist = dynamic_cast<distribution *>(Factory::instance()->create(type, name, params));
   if (_dist_by_name.find(name) == _dist_by_name.end())
    _dist_by_name[name] = dist;
   else
     mooseError("Distribution with name " << name << " already exists");

   _dist_by_type[type].push_back(dist);

}

//void
//DistributionContainer::constructDistributionContainer(std::string DistAlias, distribution_type type, double xmin, double xmax, double param1, double param2, unsigned int seed){
//	_distribution_cont.push_back(distribution(type, xmin, xmax, param1, param2, seed));
//	_vector_pos_map[DistAlias]=_distribution_cont.size()-1;
//}
//void
//DistributionContainer::constructDistributionContainerCustom(std::string DistAlias, distribution_type type, std::vector< double > dist_x, std::vector< double > dist_y, int numPoints, custom_dist_fit_type fit_type, unsigned int seed){
//
//	_distribution_cont.push_back(distribution(dist_x, dist_y, numPoints, fit_type, seed));
//	_vector_pos_map[DistAlias]=_distribution_cont.size()-1;
//}

std::string
DistributionContainer::getType(std::string DistAlias){

    if(_dist_by_name.find(DistAlias) != _dist_by_name.end()){
       distribution * dist = _dist_by_name.find(DistAlias)->second;
       std::string type = getDistributionType(*dist);
       if(type == "DistributionError"){
         mooseError("Type for distribution " << DistAlias << " not found");
       }
       return type;
    }
    else{
       mooseError("Distribution " << DistAlias << " not found in distribution container");
       return "DistributionError";
    }

}

void
DistributionContainer::seedRandom(unsigned int seed){
   srand( seed );
}
double
DistributionContainer::random(){
   return (rand()/RAND_MAX);
//   return -1.0;
}

// to be implemented
bool DistributionContainer::checkCdf(double probability, std::vector<double> values){
   return false;
}
// end to be implemented
// to be implemented
bool DistributionContainer::checkCdf(double probability, double value){
   return false;
}
// end to be implemented

double
DistributionContainer::getVariable(std::string paramName,std::string DistAlias){
    if(_dist_by_name.find(DistAlias) != _dist_by_name.end()){
       distribution * dist = _dist_by_name.find(DistAlias)->second;
       return getDistributionVariable(*dist,paramName);
    }
    mooseError("Distribution " << DistAlias << " not found in distribution container");
    return -1;
}

void
DistributionContainer::updateVariable(std::string paramName,double newValue,std::string DistAlias){
    if(_dist_by_name.find(DistAlias) != _dist_by_name.end()){
       distribution * dist = _dist_by_name.find(DistAlias)->second;
       DistributionUpdateVariable(*dist,paramName,newValue);
    }
    mooseError("Distribution " + DistAlias + " was not found in distribution container.");
}


//double
//DistributionContainer::getMin(std::string DistAlias){
//    int position;
//
//    position = getPosition(DistAlias);
//
//    if(position != -1){
//    	return _distribution_cont[position].getMin();
//    }
//    else{
//    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
//    	return DISTRIBUTION_ERROR;
//    }
//
//}
//
//double
//DistributionContainer::getMax(std::string DistAlias){
//    int position;
//
//    position = getPosition(DistAlias);
//
//    if(position != -1){
//    	return _distribution_cont[position].getMax();
//    }
//    else{
//    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
//    	return DISTRIBUTION_ERROR;
//    }
//
//}
//
//double
//DistributionContainer::getParamater1(std::string DistAlias){
//    int position;
//
//    position = getPosition(DistAlias);
//
//    if(position != -1){
//    	return _distribution_cont[position].getParamater1();
//    }
//    else{
//    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
//    	return DISTRIBUTION_ERROR;
//    }
//
//}
//
//double
//DistributionContainer::getParameter2(std::string DistAlias){
//    int position;
//
//    position = getPosition(DistAlias);
//
//    if(position != -1){
//    	return _distribution_cont[position].getParameter2();
//    }
//    else{
//    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
//    	return DISTRIBUTION_ERROR;
//    }
//
//}
//
//void
//DistributionContainer::changeParameter1(std::string DistAlias,double newParameter1){
//    int position;
//
//    position = getPosition(DistAlias);
//
//    if(position != -1){
//    	_distribution_cont[position].changeParameter1(newParameter1);
//    }
//    else{
//    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
//    }
//
//}
//
//void
//DistributionContainer::changeParameter2(std::string DistAlias,double newParameter2){
//    int position;
//
//    position = getPosition(DistAlias);
//
//    if(position != -1){
//    	_distribution_cont[position].changeParameter2(newParameter2);
//    }
//    else{
//    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
//    }
//
//}

double
DistributionContainer::Pdf(std::string DistAlias, double x){

    if(_dist_by_name.find(DistAlias) != _dist_by_name.end()){
       distribution * dist = _dist_by_name.find(DistAlias)->second;
       return DistributionPdf(*dist,x);
    }
    mooseError("Distribution " + DistAlias + " was not found in distribution container.");
    return -1.0;
}

double
DistributionContainer::Cdf(std::string DistAlias, double x){

   if(_dist_by_name.find(DistAlias) != _dist_by_name.end()){
       distribution * dist = _dist_by_name.find(DistAlias)->second;
       return DistributionCdf(*dist,x);
    }
    mooseError("Distribution " + DistAlias + " was not found in distribution container.");
    return -1.0;

}

double
DistributionContainer::randGen(char * DistAlias, double RNG){
  return randGen(std::string(DistAlias), RNG);
}

double
DistributionContainer::randGen(std::string DistAlias, double RNG){

    if(_dist_by_name.find(DistAlias) != _dist_by_name.end()){
        distribution * dist = _dist_by_name.find(DistAlias)->second;
        return dist->RandomNumberGenerator(RNG);
     }
     mooseError("Distribution " + DistAlias + " was not found in distribution container.");
     return -1.0;

}

//int DistributionContainer::getPosition(std::string DistAlias){
//	std::map <std::string, int>::iterator p;
//
//	p = _vector_pos_map.find(DistAlias);
//	if(p != _vector_pos_map.end()){
//       return (p->second);
//	}
//	else{
//		return -1;
//	}
//}


DistributionContainer & DistributionContainer::Instance() {
  if(_instance == NULL){
    _instance = new DistributionContainer();
  }
  return *_instance;
}

DistributionContainer * DistributionContainer::_instance;

/* the str_to_string_p and free_string_p are for python use */

std::string * str_to_string_p(char *s) {
  return new std::string(s);
}

void free_string_p(std::string * s) {
  delete s;
}



