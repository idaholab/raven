/*
 * DistributionContainer.C
 *
 *  Created on: Jul 6, 2012
 *      Author: alfoa
 */
#include "DistributionContainer.h"

#include "distribution_1D.h"
#include <iostream>
using namespace std;

class DistributionContainer;

DistributionContainer::DistributionContainer()
{
}
DistributionContainer::~DistributionContainer()
{
}
void
DistributionContainer::constructDistributionContainer(std::string DistAlias, distribution_type type, double xmin, double xmax, double param1, double param2, unsigned int seed){
	_distribution_cont.push_back(distribution_1D(type, xmin, xmax, param1, param2, seed));
	_vector_pos_map[DistAlias]=_distribution_cont.size()-1;
}
void
DistributionContainer::constructDistributionContainerCustom(std::string DistAlias, distribution_type type, std::vector< double > dist_x, std::vector< double > dist_y, int numPoints, custom_dist_fit_type fit_type, unsigned int seed){

	_distribution_cont.push_back(distribution_1D(dist_x, dist_y, numPoints, fit_type, seed));
	_vector_pos_map[DistAlias]=_distribution_cont.size()-1;
}

distribution_type
DistributionContainer::getType(std::string DistAlias){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].getType();
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

double
DistributionContainer::getMin(std::string DistAlias){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].getMin();
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

double
DistributionContainer::getMax(std::string DistAlias){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].getMax();
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

double
DistributionContainer::getParamater1(std::string DistAlias){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].getParamater1();
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

double
DistributionContainer::getParameter2(std::string DistAlias){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].getParameter2();
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

void
DistributionContainer::changeParameter1(std::string DistAlias,double newParameter1){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	_distribution_cont[position].changeParameter1(newParameter1);
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    }

}

void
DistributionContainer::changeParameter2(std::string DistAlias,double newParameter2){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	_distribution_cont[position].changeParameter2(newParameter2);
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    }

}

double
DistributionContainer::pdfCalc(std::string DistAlias, double x){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].pdfCalc(x);
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

double
DistributionContainer::cdfCalc(std::string DistAlias, double x){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].cdfCalc(x);
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

double
DistributionContainer::randGen(char * DistAlias){
  return randGen(std::string(DistAlias));
}

double
DistributionContainer::randGen(std::string DistAlias){
    int position;

    position = getPosition(DistAlias);

    if(position != -1){
    	return _distribution_cont[position].randGen();
    }
    else{
    	std::cerr << " ERROR: distribution " << DistAlias << "not present in the mapping";
    	return DISTRIBUTION_ERROR;
    }

}

int DistributionContainer::getPosition(std::string DistAlias){
	std::map <std::string, int>::iterator p;

	p = _vector_pos_map.find(DistAlias);
	if(p != _vector_pos_map.end()){
       return (p->second);
	}
	else{
		return -1;
	}
}


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



