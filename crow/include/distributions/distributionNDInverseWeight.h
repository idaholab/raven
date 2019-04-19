/* Copyright 2017 Battelle Energy Alliance, LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
/*
 * distributionNDInverseWeight.h
 * Created by @wangc on Oct. 23, 2015
 * Extracted from  distribution_base_ND.h
 *
 */
#ifndef DISTRIBUTION_ND_INVERSE_WEIGHT_H
#define DISTRIBUTION_ND_INVERSE_WEIGHT_H
#include <map>
#include <string>
#include <vector>
#include "ND_Interpolation_Functions.h"
#include "distributionNDCartesianSpline.h"
#include "distributionFunctions.h"
#include "distributionNDBase.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iso646.h>

#define throwError(msg) { std::cerr << "\n\n" << msg << "\n\n"; throw std::runtime_error("Error"); }

class BasicMultiDimensionalInverseWeight: public virtual BasicDistributionND
{
public:
  BasicMultiDimensionalInverseWeight(const char * data_filename,double p, bool cdf_provided):  _interpolator(data_filename,p)
  {
    _cdf_provided = cdf_provided;
    basicMultiDimensionalInverseWeightInit();
  };

  BasicMultiDimensionalInverseWeight(std::string data_filename,double p, bool cdf_provided):  _interpolator(data_filename,p)
  {
    _cdf_provided = cdf_provided;
    basicMultiDimensionalInverseWeightInit();
  };

  BasicMultiDimensionalInverseWeight(double p):  _interpolator(InverseDistanceWeighting(p))
  {
  };

  void basicMultiDimensionalInverseWeightInit(){
    std::cout<<"Initialize BasicMultiDimensionalInverseWeight"<< std::endl;

    if (_cdf_provided) {
      bool LBcheck = _interpolator.checkLB(0.0);
      if (LBcheck == false)
        throwError("BasicMultiDimensionalInverseWeight Distribution error: CDF values given as input contain element below 0.0");

      bool UBcheck = _interpolator.checkUB(1.0);
      if (UBcheck == false)
        throwError("BasicMultiDimensionalInverseWeight Distribution error: CDF values given as input contain element above 1.0");
    }
    else{  // PDF is provided
      // Create ND spline for the CDF
      //BasicMultiDimensionalCartesianSpline(std::string data_filename,std::vector<double> alpha, std::vector<double> beta, bool cdf_provided)
      std::cout<<"Creating ND spline for inverseWeight"<< std::endl;
      int n_dimensions = _interpolator.returnDimensionality();
      std::vector<double> alpha (n_dimensions);
      std::vector<double> beta (n_dimensions);

      for(int i=0; i<n_dimensions; i++){
        alpha.at(i) = 0.0;
        beta.at(i) = 0.0;
      }
      // Here I am building a cartesian grid from a sparse set of points
      // TODO: give the possibility at the user to specify this value and add a ticket.
      int numberDiscretization = 15;

      std::vector<std::vector<double> > discretizations;
      std::vector<double> cellPoint0 = _interpolator.getCellPoint0();
      std::vector<double> cellDxs = _interpolator.getCellDxs();

      std::cout<<"Discretization points for ND spline for inverseWeight"<< std::endl;
      for(int i=0; i<n_dimensions; i++){
        std::vector<double> temp;
        for(int j=0; j<numberDiscretization; j++){
          // in the following I am building a cartesian grid from a sparse set of points
          double value = cellPoint0.at(i) + cellDxs.at(i)/numberDiscretization * j;
          temp.push_back(value);
        }
        discretizations.push_back(temp);
      }

      int totalNumberOfValues=1;
      std::vector<int> discretizationSizes(n_dimensions);
      for(int i=0; i<n_dimensions; i++){
        totalNumberOfValues *= numberDiscretization;
        discretizationSizes.at(i) = numberDiscretization;
      }

      std::vector<double> PDFvalues (totalNumberOfValues);

      for (int i=0; i<totalNumberOfValues; i++){
        std::vector<int> nd_coordinateIndex = oneDtoNDconverter(i, discretizationSizes);
        std::vector<double> nd_coordinate(n_dimensions);
        for (int j=0; j<n_dimensions; j++){
          nd_coordinate.at(j) = discretizations.at(j)[nd_coordinateIndex.at(j)];
        }
        PDFvalues.at(i) = _interpolator.interpolateAt(nd_coordinate);
      }
      _cdf_spline = BasicMultiDimensionalCartesianSpline(discretizations, PDFvalues, alpha, beta, false);
    }
  }

  virtual ~BasicMultiDimensionalInverseWeight()
  {
  };

  double
  pdf(std::vector<double> x)
  {
    if (_cdf_provided){
      return _interpolator.ndDerivative(x);
    }
    else
      return _interpolator.interpolateAt(x);
  };

  double
  cdf(std::vector<double> x)
  {
    double value;
    if (_cdf_provided){
      value = _interpolator.interpolateAt(x);
    }
    else
      value = _cdf_spline.cdf(x);

     if (value > 1.0)
   throwError("BasicMultiDimensionalInverseWeight Distribution error: CDF value calculated is above 1.0");

     return value;
  };

  void updateRNGparameter(double tolerance, double initial_divisions){
    _tolerance = tolerance;
    _initial_divisions = (int)initial_divisions;

    _interpolator.updateRNGParameters(_tolerance,_initial_divisions);

    _cdf_spline.updateRNGparameter(_tolerance,_initial_divisions);

    //std::cout<<"Distribution updateRNGparameter" << _tolerance <<  _initial_divisions << std::endl;
  };


  std::vector<double>
  inverseCdf(double f, double g)
  {
    if (_cdf_provided)
      return _interpolator.ndInverseFunctionGrid(f,g);
    else{
      return _cdf_spline.inverseCdf(f,g);
    }
  };

  double inverseMarginal(double f, int dimension){
    double value=0.0;

    if ((f<1.0) and (f>0.0)){
      if (_cdf_provided){
        throwError("BasicMultiDimensionalInverseWeight Distribution error: inverseMarginal calculation not available if CDF provided");
      }else{
        value = _cdf_spline.inverseMarginal(f, dimension);
      }
    }else
      throwError("BasicMultiDimensionalInverseWeight Distribution error: CDF value for inverse marginal distribution is above 1.0");

    return value;
  }

  double marginal(double x, int dimension)
    /**
     * Method to get the margina cdf give the dimension id
     * and the value x
     * @ In, double, x, the parameter value
     * @ In, int, dimension, the dimension id
     * @ Out, double, value, the marginal CDF value at x for dim dimension
     */
    {
        double value=0.0;
        if (_cdf_provided){
            throwError("BasicMultiDimensionalInverseWeight Distribution error: marginal calculation not available if CDF provided");
        }else{
            value = _cdf_spline.marginal(x, dimension);
        }
        return value;
    }

  int
  returnDimensionality()
  {
    return _interpolator.returnDimensionality();
  }

  double returnUpperBound(int dimension){
   /**
    * this function returns the upper bound of the distribution for a particular dimension
    */
    return _interpolator.returnUpperBound(dimension);
  }

  double returnLowerBound(int dimension){
    /**
     * this function returns the lower bound of the distribution for a particular dimension
     */
    return _interpolator.returnLowerBound(dimension);
  }

protected:
  InverseDistanceWeighting  _interpolator;
  BasicMultiDimensionalCartesianSpline  _cdf_spline;
  bool _cdf_provided;
};

#endif /* DISTRIBUTION_ND_INVERSE_WEIGHT_H */
