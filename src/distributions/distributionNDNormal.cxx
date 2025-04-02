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
 * distributionNDNormal.C
 * Created on Oct. 23, 2015
 * Author: @wangc
 * Extracted from @alfoa (Feb 6, 2014) distribution_base_ND.C
 *
 */

#include "distributionNDNormal.h"
#include "distributionNDBase.h"
#include "DistributionContainer.h"
#include <stdexcept>
#include <iostream>
#include "MDreader.h"
#include "distributionFunctions.h"
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions/normal.hpp>
#include "distribution_1D.h"
using boost::math::normal;

#include <ctime>

#define _use_math_defines

//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/lu.hpp>
//#include <boost/numeric/ublas/io.hpp>

//using namespace boost::numeric::ublas;

#define throwError(msg) { std::cerr << "\n\n" << msg << "\n\n"; throw std::runtime_error("Error"); }

#ifndef M_PI
//PI is not actually defined anywhere in the C++ standard.
#define M_PI 3.14159265358979323846
#endif

void BasicMultivariateNormal::base10ToBaseN(int value_base10, int base, std::vector<int> & value_base_n){
    /**
     * This function convert a number in base 10 to a new number in any base N
     */

     int index = 0 ;

     if (value_base10 == 0)
       value_base_n.push_back(0);
     else{
       while ( value_base10 != 0 ){
         int remainder = value_base10 % base ;  // assume K > 1
         value_base10  = value_base10 / base ;  // integer division
         value_base_n.push_back(remainder);
         index++ ;
      }
     }
}

//void BasicMultivariateNormal::basicMultivariateNormalInit(std::string data_filename, std::vector<double> mu){
void BasicMultivariateNormal::basicMultivariateNormalInit(unsigned int &rows, unsigned int &columns, std::vector<std::vector<double> > cov_matrix, std::vector<double> mu){
    /**
     * This is the base function that initializes the Multivariate normal distribution
     * Input Parameter
     * rows: first dimension of covariance matrix
     * columns: second dimension of covariance matrix
     * cov_matrix: covariance matrix stored in vector<vector<double> >
     * mu: mean value stored in vector<double>
     */

   _mu = mu;
   _cov_matrix = cov_matrix;

   std::vector<std::vector<double> > inverseCovMatrix (rows,std::vector< double >(columns));

   computeInverse(_cov_matrix, inverseCovMatrix);

   for (unsigned int i=0;i<rows;i++){
    std::vector<double> temp;
    for (unsigned int j=0;j<columns;j++)
     temp.push_back(inverseCovMatrix.at(i).at(j));
    _inverse_cov_matrix.push_back(temp);
   }

   unsigned int dimensions = _mu.size();
   //for(int i=0; i<dimensions; i++)
  //for(int j=0; j<dimensions; j++)
   //std::cerr<<_inverse_cov_matrix[i][j]<<std::endl;

   _determinant_cov_matrix = getDeterminant(_cov_matrix);

   _cholesky_C = choleskyDecomposition(_cov_matrix);

   if(rows != columns)
     throwError("MultivariateNormal error: covariance matrix in is not a square matrix.");

   // Creation BasicMultiDimensionalCartesianSpline(std::vector< std::vector<double> > & discretizations, std::vector<double> & values, std::vector<double> alpha, std::vector<double> beta, bool cdf_provided)

   int numberValues=1;
   std::vector< std::vector<double> > discretizations;
   std::vector<double> alpha (_mu.size());
   std::vector<double> beta (_mu.size());
   // for now we use this to be a bit more less problem dependent
   double floatDiscretizations = (1./std::pow(1.e-4,1./ (double)dimensions)) + 0.5;
   int numberOfDiscretizations = (int)floatDiscretizations;
   for(unsigned int i=0; i<dimensions; i++){
     alpha.at(i) = 0.0;
     beta.at(i)  = 0.0;
     numberValues = numberValues * numberOfDiscretizations;

     std::vector<double> discretization_temp;
     double sigma = sqrt(_cov_matrix[i][i]);
     double deltaSigma = 12.0*sigma/(double)numberOfDiscretizations;
     for(int n=0; n<numberOfDiscretizations; n++){
       double disc_value = mu.at(i) - 6.0 * sigma + deltaSigma * (double)n;
       discretization_temp.push_back(disc_value);
     }
     discretizations.push_back(discretization_temp);
     _lower_bounds.push_back(discretization_temp.at(0));
     _upper_bounds.push_back(discretization_temp.back());
   }
   std::vector< double > values (numberValues);
   for(int i=0; i<numberValues; i++){
     std::vector<int> intCoordinates;
     base10ToBaseN(i,numberOfDiscretizations,intCoordinates);
     std::vector<double> point_coordinates(dimensions);
     std::vector<int> intCoordinatesFormatted(dimensions);

     for(unsigned int j=0; j<dimensions; j++)
       intCoordinatesFormatted.at(j) = 0;
     for(unsigned int j=0; j<intCoordinates.size(); j++)
       intCoordinatesFormatted.at(j) = intCoordinates.at(j);

     for(unsigned int j=0; j<intCoordinates.size(); j++)
       point_coordinates.at(j) = discretizations.at(j).at(intCoordinatesFormatted.at(j));

     values.at(i) = getPdf(point_coordinates, _mu, _inverse_cov_matrix);
   }
   _cartesian_distribution = BasicMultiDimensionalCartesianSpline(discretizations,values,alpha,beta,false);

}

BasicMultivariateNormal::BasicMultivariateNormal(std::string data_filename, std::vector<double> mu){
    /**
     * This is the function that initializes the Multivariate normal distribution given:
     * - data_filename: it specifies the covariance matrix
     * - mu: the mean value vector
     */
  unsigned int rows,columns;
  std::vector<std::vector<double> > cov_matrix;
  readMatrix(data_filename, rows, columns, cov_matrix);
  basicMultivariateNormalInit(rows,columns,cov_matrix, mu);
}

BasicMultivariateNormal::BasicMultivariateNormal(const char * data_filename, std::vector<double> mu){
    /**
     * This is the function that initializes the Multivariate normal distribution given:
     * - data_filename: it specifies the covariance matrix
     * - mu: the mean value vector
     */
  unsigned int rows,columns;
  std::vector<std::vector<double> > cov_matrix;
  readMatrix(std::string(data_filename), rows, columns, cov_matrix);
  basicMultivariateNormalInit(rows,columns,cov_matrix, mu);
  //basicMultivariateNormalInit(std::string(data_filename) , mu);
}

BasicMultivariateNormal::BasicMultivariateNormal(std::vector<std::vector<double> > cov_matrix, std::vector<double> mu){
  /**
   * This is the function that initializes the Multivariate normal distribution given:
   * - cov_matrix: covariance matrix
   * - mu: the mean value vector
   */
  unsigned int rows, columns;
  rows = cov_matrix.size();
  columns = cov_matrix.at(0).size();

  basicMultivariateNormalInit(rows,columns,cov_matrix, mu);
  //_mu = mu;
  //_cov_matrix = cov_matrix;

  //computeInverse(_cov_matrix, _inverse_cov_matrix);

  //_determinant_cov_matrix = getDeterminant(_cov_matrix);
}

// Input Parameters: vectors of covariance and mu
BasicMultivariateNormal::BasicMultivariateNormal(std::vector<double> vec_cov_matrix, std::vector<double> mu){
  /**
   * This is the function that initializes the Multivariate normal distribution given:
   * Input Parameters
   * - vec_cov_matrix: covariance matrix stored in a vector<double>
   * - mu: the mean value vector
   */

  unsigned int rows, columns;
  std::vector<std::vector<double> > cov_matrix;
  // convert the vec_cov_matrix to cov_matrix, output the rows and columns of the covariance matrix
  vectorToMatrix(rows,columns,vec_cov_matrix,cov_matrix);

  basicMultivariateNormalInit(rows,columns,cov_matrix, mu);
}

double BasicMultivariateNormal::getPdf(std::vector<double> x, std::vector<double> mu, std::vector<std::vector<double> > inverse_cov_matrix){
  /**
   * This function calculates the pdf values at x of a MVN distribution
   */

  double value = 0;

   if(mu.size() == x.size()){
     int dimensions = mu.size();
     double expTerm=0;
     std::vector<double> tempVector (dimensions);
     for(int i=0; i<dimensions; i++){
       tempVector[i]=0;
       for(int j=0; j<dimensions; j++)
         tempVector[i] += inverse_cov_matrix[i][j]*(x[j]-mu[j]);
       expTerm += tempVector[i]*(x[i]-mu[i]);
     }
     value = 1/sqrt(_determinant_cov_matrix*pow(2*M_PI,dimensions))*exp(-0.5*expTerm);
   }else
     throwError("MultivariateNormal PDF error: evaluation point dimensionality is not correct");
   return value;
}

double BasicMultivariateNormal::pdf(std::vector<double> x){
    /**
     * This function calculates the pdf values at x of a MVN distribution
     */
  return getPdf(x, _mu, _inverse_cov_matrix);
}

double BasicMultivariateNormal::cdf(std::vector<double> x){
    /**
     * This function calculates the cdf values at x of a MVN distribution
     */
  return _cartesian_distribution.cdf(x);
}

std::vector<double> BasicMultivariateNormal::inverseCdf(double f, double g){
    /**
     * This function calculates the inverse CDF values at f of a MVN distribution
     */
  return _cartesian_distribution.inverseCdf(f,g);
}

double BasicMultivariateNormal::inverseMarginal(double f, int dimension){
    /**
     * This function calculates the inverse marginal distribution at f for a specific dimension of a MVN distribution
     */
  return _cartesian_distribution.inverseMarginal(f,dimension);
}

int BasicMultivariateNormal::returnDimensionality(){
    /**
     * This function returns the dimensionality of a MVN distribution
     */
  return _mu.size();
}

void BasicMultivariateNormal::updateRNGparameter(double tolerance, double initial_divisions){
    /**
     * This function updates the random number generator parameters of a MVN distribution
     */
  return _cartesian_distribution.updateRNGparameter(tolerance,initial_divisions);
}

double BasicMultivariateNormal::marginal(double x, int dimension){
    /**
     * This function calculates the marginal distribution at x for a specific dimension of a MVN distribution
     */
  return _cartesian_distribution.marginal(x,dimension);
}

//double BasicMultivariateNormal::cdf_(std::vector<double> x){
//// if(_mu.size() == x.size()){
////  int dimensions = _mu.size();
////  //boost::math::chi_squared chiDistribution(dimensions);
////
////  double mahalanobis=0.0;
////  std::vector<double> tempVector (dimensions);
////  for(int i=0; i<dimensions; i++)
////   tempVector[i]=0.0;
////
////  for(int i=0; i<dimensions; i++){
////   tempVector[i]=0.0;
////   for(int j=0; j<dimensions; j++)
////    tempVector[i] += _inverseCovMatrix[i][j]*(x[j]-_mu[j]);
////   mahalanobis += tempVector[i]*(x[i]-_mu[i]);
////  }
////  value = boost::math::gamma_p<double,double>(dimensions/2,mahalanobis/2);
//// }else
////  throwError("MultivariateNormal CDF error: evaluation point dimensionality is not correct");
//
// double alpha = 2.5;
// int Nmax = 50;
// double epsilon= 0.01;
// double delta;
//
// int dimensions = _cov_matrix.size();
// double Intsum=0;
// double Varsum=0;
// int N = 0;
// double error=10*epsilon;
// std::vector<double> d (dimensions);
// std::vector<double> e (dimensions);
// std::vector<double> f (dimensions);
//
// d[0] = 0.0;
// e[0] = phi(x[0]/_cholesky_C[0][0]);
// f[0] = e[0] - d[0];
//
// boost::random::mt19937 rng;
// rng.seed(time(NULL));
// double range = rng.max() - rng.min();
//
// while (error>epsilon or N<Nmax){
//  std::vector<double> w (dimensions-1);
//
//  for (int i=0; i<(dimensions-1); i++){
//   w.at(i) = (rng()-rng.min())/range;
//   //std::cout<< "value: " << w.at(i) << std::endl;
//  }
//
//  std::vector<double> y (dimensions-1);
//
//  for (int i=1; i<dimensions; i++){
//   double tempY = d.at(i-1) + w.at(i-1) * (e.at(i-1)-d.at(i-1));
//
//   y.at(i-1) = phiInv(tempY);
//
//   double tempE = x.at(i);
//
//   for (int j=0; j<(i-1); j++)
//    tempE = tempE - _cholesky_C[i][j] * y.at(j) / _cholesky_C[i][i];
//
//   e.at(i)=phi(tempE);
//   d.at(i)=0.0;
//   f.at(i)=(e.at(i)-d.at(i))*f.at(i-1);
//  }
//
//  N++;
//  delta = (f.at(dimensions-1)-Intsum)/double(N);
//  Intsum = Intsum + delta;
//  Varsum = (double(N-2))*Varsum/double(N) + delta*delta;
//  error = alpha * sqrt(Varsum);
//
//  std::cout << "N " << N << " ; f: " << f.at(dimensions-1) << " ; delta: " << delta << " ; Intsum: " << Intsum << " ; Varsum: " << Varsum << "; error: " << error << std::endl;
// }
//
// return Intsum;
//}

BasicMultivariateNormal::~BasicMultivariateNormal(){

}

double BasicMultivariateNormal::phi(double x){
 double value = 0.5 * (1.0 + boost::math::erf<double>(x/sqrt(2.0)));
 //double value = 0.5 * (boost::math::erf<double>(x/sqrt(2)));
 return value;
}

double BasicMultivariateNormal::phiInv(double x){
 normal s;
 double value = quantile(s,x);
 return value;
}

//double BasicMultivariateNormal::rn(){
//    boost::random::mt19937 rng;
// rng.seed(time(NULL));
// double range = rng.max() - rng.min();
// double value = (rng()-rng.min())/range;
// std::cout<< "value: " << value << std::endl;
// return value;
//}

//double BasicMultivariateNormal::MVNDST(std::vector<double> a, std::vector<double> b, double alpha, double epsilon, int Nmax){
// int dimensions = _cov_matrix.size();
// double Intsum=0;
// double Varsum=0;
// int N = 0;
// double error;
// std::vector<double> d (dimensions);
// std::vector<double> e (dimensions);
// std::vector<double> f (dimensions);
//
// std::vector<std::vector<double> > cholesky_C = choleskyDecomposition(_cov_matrix);
//
// d[0] = phi(a[0]/cholesky_C[0][0]);
// e[0] = phi(b[0]/cholesky_C[0][0]);
// f[0] = e[0] - d[0];
//
//    boost::random::mt19937 rng;
// rng.seed(time(NULL));
// double range = rng.max() - rng.min();
//
// do{
//  std::vector<double> w (dimensions-1);
//  for (int i=0; i<(dimensions-1); i++){
//   w.at(i) = (rng()-rng.min())/range;
//   std::cout<< "value: " << rng() << std::endl;
//  }
//
//  std::vector<double> y (dimensions-1);
//  for (int i=1; i<dimensions; i++){
//   double tempY = d.at(i-1) + w.at(i-1)*(e.at(i-1)-d.at(i-1));
//   y.at(i-1) = phiInv(tempY);
//
//   double tempD = a.at(i);
//   double tempE = b.at(i);
//
//   for (int j=0; j<(i-1); j++){
//    tempD = tempD - cholesky_C[i][j] * y[j] / cholesky_C[i][i];
//    tempE = tempE - cholesky_C[i][j] * y[j] / cholesky_C[i][i];
//   }
//
//   d[i]=phi(tempD);
//   e[i]=phi(tempE);
//   f[i]=(e[i]-d[i])/f[i-1];
//  }
//
//  N++;
//  double delta = (f[dimensions-1]-Intsum)/N;
//  Intsum = Intsum + delta;
//  Varsum = (N-2)*Varsum/N + delta*delta;
//  error = alpha * sqrt(Varsum);
//
// } while (error<epsilon and N<Nmax);
//
// return Intsum;
//}

//http://rosettacode.org/wiki/Cholesky_decomposition#C
double *BasicMultivariateNormal::cholesky(double *A, int n) {
    double *L = (double*)calloc(n * n, sizeof(double));
    if (L == NULL)
  exit(EXIT_FAILURE);

    for (int i = 0; i < n; i++)
  for (int j = 0; j < (i+1); j++) {
      double s = 0;
      for (int k = 0; k < j; k++)
    s += L[i * n + k] * L[j * n + k];
      L[i * n + j] = (i == j) ?
         sqrt(A[i * n + i] - s) :
         (1.0 / L[j * n + j] * (A[i * n + j] - s));
  }

    return L;
}

std::vector<std::vector<double> > BasicMultivariateNormal::choleskyDecomposition(std::vector<std::vector<double> > matrix){
 std::vector<std::vector<double> > cholesky_C;

 int dimensions = matrix.size();
 double * m1 = new double[dimensions*dimensions];

 for (int r=0; r<dimensions; r++)
  for (int c=0; c<dimensions; c++)
   m1[r*dimensions+c] = matrix[r][c];

 double *c1 = cholesky(m1, dimensions);
 //std::cout << "choleskyDecomposition" << std::endl;
 //showMatrix(c1,dimensions);

 for (int r=0; r<dimensions; r++){
  std::vector<double> temp;
  for (int c=0; c<dimensions; c++)
   temp.push_back(c1[r*dimensions+c]);
  cholesky_C.push_back(temp);
 }
 delete m1;
 return cholesky_C;
}

void BasicMultivariateNormal::showMatrix(double *A, int n) {
    for (int i = 0; i < n; i++) {
  for (int j = 0; j < n; j++)
      printf("%2.5f ", A[i * n + j]);
  printf("\n");
    }
}

//template<class T>
//bool InvertMatrix(const matrix<T>& input, matrix<T>& inverse)
//{
// typedef permutation_matrix<std::size_t> pmatrix;
//
// // create a working copy of the input
// matrix<T> A(input);
//
// // create a permutation matrix for the LU-factorization
// pmatrix pm(A.size1());
//
// // perform LU-factorization
// int res = lu_factorize(A, pm);
// if (res != 0)
//  return false;
//
// // create identity matrix of "inverse"
// inverse.assign(identity_matrix<T> (A.size1()));
//
// // backsubstitute to get the inverse
// lu_substitute(A, pm, inverse);
//
// return true;
//}
//
//void getInverse(std::vector<std::vector<double> > matrix, std::vector<std::vector<double> > inverse_matrix){
// int dimension = matrix.size();
// double initialValues[dimension][dimension];
// matrix<double> A(dimension, dimension), Z(dimension, dimension);
//
// for(int i=0; i<dimension; i++)
//  for(int j=0; j<dimension; j++)
//   initialValues[i][j]=matrix[i][j];
// A = make_matrix_from_pointer(initialValues);
// InvertMatrix(A, Z);
//
// for(int i=0; i<dimension; i++){
//  std::vector<double> temp;
//  for(int j=0; j<dimension; j++)
//   temp.push_back(Z[i][j]);
//  inverse_matrix.push_back(temp);
// }
//}
