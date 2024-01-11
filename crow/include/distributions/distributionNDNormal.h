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
 * distributionNDNormal.h
 * Created by @wangc on Oct. 23, 2015
 * Extracted from  distribution_base_ND.h
 *
 */
#ifndef DISTRIBUTION_ND_NORMAL_H
#define DISTRIBUTION_ND_NORMAL_H
#include <map>
#include <string>
#include <vector>
#include "ND_Interpolation_Functions.h"
#include "distributionFunctions.h"
#include "distributionNDBase.h"
#include "distributionNDCartesianSpline.h"
#include <stdexcept>
//#include "distribution_min.h"
#include <iostream>
#include <fstream>

#define throwError(msg) { std::cerr << "\n\n" << msg << "\n\n"; throw std::runtime_error("Error"); }


class BasicMultivariateNormal: public virtual BasicDistributionND
{
public:
  BasicMultivariateNormal(const char * data_filename, std::vector<double> mu);
  BasicMultivariateNormal(std::string data_filename, std::vector<double> mu);
  BasicMultivariateNormal(std::vector<std::vector<double> > cov_matrix, std::vector<double> mu);
  BasicMultivariateNormal(std::vector<double> vec_cov_matrix, std::vector<double> mu);

  //void basicMultivariateNormalInit(std::string data_filename, std::vector<double> mu);
  void basicMultivariateNormalInit(unsigned int & rows, unsigned int &columns, std::vector<std::vector<double> > cov_matrix, std::vector<double> mu);

  virtual ~BasicMultivariateNormal();
  double  pdf(std::vector<double> x);
  double  cdf(std::vector<double> x);
  std::vector<double> inverseCdf(double f, double g);
  double inverseMarginal(double f, int dimension);
  void updateRNGparameter(double tolerance, double initial_divisions);
  double marginal(double x, int dimension);

  int returnDimensionality();

  //double MVNDST(std::vector<double> a, std::vector<double> b, double alpha, double epsilon, int Nmax);
  double phi(double x);
  double phiInv(double x);
  //double rn();
  double * cholesky(double *A, int n);
  std::vector<std::vector<double> > choleskyDecomposition(std::vector<std::vector<double> > matrix);
  void showMatrix(double *A, int n);

  double returnLowerBound(int dimension){
    return _lower_bounds.at(dimension);
  }

  double returnUpperBound(int dimension){
    return _upper_bounds.at(dimension);
  }

private:
  std::vector<double> _mu;
  std::vector<std::vector<double> > _cov_matrix;
  std::vector<std::vector<double> > _inverse_cov_matrix;
  std::vector<std::vector<double> > _cholesky_C;
  // parameters for singular value decomposition
  double _determinant_cov_matrix;

  std::vector<double> _upper_bounds;
  std::vector<double> _lower_bounds;

  BasicMultiDimensionalCartesianSpline _cartesian_distribution;

  void base10ToBaseN(int value_base10, int base, std::vector<int> & value_base_n);
  double getPdf(std::vector<double> x, std::vector<double> mu, std::vector<std::vector<double> > inverse_cov_matrix);
};


#endif /* DISTRIBUTION_ND_NORMAL_H */
