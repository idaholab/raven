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
#include <MooseRandom.h>

using namespace std;

#ifndef mooseError
#define mooseError(msg) { std::cerr << "\n\n" << msg << "\n\n"; }
#endif

class DistributionContainer;

DistributionContainer::DistributionContainer()
{
  _at_least_a_dist_triggered = false;
  _last_dist_triggered = "";
}
DistributionContainer::~DistributionContainer()
{
}

/*void
DistributionContainer::addDistributionInContainer(const std::string & type, const std::string & name, InputParameters params){
   // create the distribution type
   distribution * dist = dynamic_cast<distribution *>(_factory.create(type, name, params));
   if (_dist_by_name.find(name) == _dist_by_name.end())
    _dist_by_name[name] = dist;
   else
     mooseError("Distribution with name " << name << " already exists");

   _dist_by_type[type].push_back(dist);

   }*/

void
DistributionContainer::addDistributionInContainer(const std::string & type, const std::string & name, distribution * dist){
   // create the distribution type
  //distribution * dist = dynamic_cast<distribution *>(_factory.create(type, name, params));
   if (_dist_by_name.find(name) == _dist_by_name.end())
    _dist_by_name[name] = dist;
   else
     mooseError("Distribution with name " << name << " already exists");

   _dist_by_type[type].push_back(dist);

}

//void
//DistributionContainer::constructDistributionContainer(std::string DistAlias, distribution_type type, double xmin, double xmax, double param1, double param2, unsigned int seed){
//  _distribution_cont.push_back(distribution(type, xmin, xmax, param1, param2, seed));
//  _vector_pos_map[DistAlias]=_distribution_cont.size()-1;
//}
//void
//DistributionContainer::constructDistributionContainerCustom(std::string DistAlias, distribution_type type, std::vector< double > dist_x, std::vector< double > dist_y, int numPoints, custom_dist_fit_type fit_type, unsigned int seed){
//
//  _distribution_cont.push_back(distribution(dist_x, dist_y, numPoints, fit_type, seed));
//  _vector_pos_map[DistAlias]=_distribution_cont.size()-1;
//}

std::string
DistributionContainer::getType(char *  DistAlias){
  return getType(std::string(DistAlias));
}

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
	//srand( seed );
	_random.seed(seed);
}
double
DistributionContainer::random(){
   //return (static_cast<double>(rand())/static_cast<double>(RAND_MAX));

	return _random.rand();
}

bool DistributionContainer::checkCdf(std::string DistAlias, double value){
  bool result;
  if (Cdf(std::string(DistAlias),value) >= getVariable("ProbabilityThreshold",DistAlias)){
    result=true;
    _dist_by_trigger_status[DistAlias] = true;
    _last_dist_triggered = DistAlias;
    _at_least_a_dist_triggered = true;
  }
  else{
    result=false;
    _dist_by_trigger_status[DistAlias] = false;
  }
  return result;
}

bool
DistributionContainer::getTriggerStatus(std::string DistAlias){
  bool st;
  if(_dist_by_trigger_status.find(DistAlias) != _dist_by_trigger_status.end()){
    st = _dist_by_trigger_status.find(DistAlias) -> second;
  }
  else{
    mooseError("Distribution " + DistAlias + " not found in Triggering event.");
  }
  return st;
}
bool
DistributionContainer::getTriggerStatus(char * DistAlias){
  return getTriggerStatus(std::string(DistAlias));
}
// to be implemented
bool DistributionContainer::checkCdf(char * DistAlias, double value){
  return checkCdf(std::string(DistAlias),value);
}
// end to be implemented

double
DistributionContainer::getVariable(char * paramName,char *DistAlias){
  return getVariable(std::string(paramName),std::string(DistAlias));
}

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
DistributionContainer::updateVariable(char * paramName,double newValue,char *DistAlias){
  updateVariable(std::string(paramName),newValue,std::string(DistAlias));
}

void
DistributionContainer::updateVariable(std::string paramName,double newValue,std::string DistAlias){
    if(_dist_by_name.find(DistAlias) != _dist_by_name.end()){
       distribution * dist = _dist_by_name.find(DistAlias)->second;
       DistributionUpdateVariable(*dist,paramName,newValue);
    }
    else{
       mooseError("Distribution " + DistAlias + " was not found in distribution container.");

    }
}


double
DistributionContainer::Pdf(char * DistAlias, double x){
   return Pdf(std::string(DistAlias),x);
}

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
DistributionContainer::Cdf(char * DistAlias, double x){
   return Cdf(std::string(DistAlias),x);
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


DistributionContainer & DistributionContainer::Instance() {
  if(_instance == NULL){
    _instance = new DistributionContainer();
  }
  return *_instance;
}

DistributionContainer * DistributionContainer::_instance = NULL;

/* the str_to_string_p and free_string_p are for python use */

std::string * str_to_string_p(char *s) {
  return new std::string(s);
}

const char * string_p_to_str(const std::string * s) {
  return s->c_str();
}

void free_string_p(std::string * s) {
  delete s;
}



