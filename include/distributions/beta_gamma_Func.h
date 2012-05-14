/*
 * beta_gamma_Func.h
 *
 *  Created on: Mar 27, 2012
 *      Author: MANDD
 *      this file contains auxiliary functions for the beta and gamma distributions
 *
 *      Tests		: None
 *
 *      Problems	: None
 *      Issues		: None
 *      Complaints	: None
 *      Compliments	: None
 *
 *   source: Numerical Recipes in C++ 3rd edition
 */

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cmath>	// to use erfc error function

using namespace std;

#ifndef BETA_GAMMA_FUNC_H_
#define BETA_GAMMA_FUNC_H_

// Code for inverse gamma function
// source: Numerical Recipes in c++

static void nrerror2(const char error_text[]){					// added const to avoid "warning: deprecated conversion from string constant to ‘char*’"
/* Numerical Recipes standard error handler */
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
}

static void gcf(double *gammcf,double a,double x,double *gln);
static void gser(double *gamser,double a,double x,double *gln);

static double gammp(double a, double x){
/* high level function for incomplete gamma function */
   double gamser,gammcf,gln;
   if(x < 0.0 || a <= 0.0) nrerror2("Invalid arg in gammp");
   if(x < (a+1.0)){
/* here I change routine so that it returns \gamma(a,x)
   or P(a,x)-just take out comments to get P(a,x) vs \gamma(a,x)-
   to get latter use the exp(log(.)+gln) expression */
	  gser(&gamser,a,x,&gln);
//      return exp(log(gamser)+gln);
	  return gamser;
   }
   else{
	  gcf(&gammcf,a,x,&gln);
//      return exp(log(1.0-gammcf)+gln);
	  return 1.0-gammcf;
   }
}

static double loggam(double xx)
{
   double x,y,tmp,ser;
   static double cof[6]={76.18009172947146, -86.50532032941677,
      24.01409824083091,-1.231739572450155, 0.001208650973866179,
      -5.395239384953e-006};
   int j;
   y=x=xx;
   tmp=x+5.5;
   tmp -= (x+0.5)*log(tmp);
   ser=1.000000000190015;
   for(j=0;j<=5;j++) ser += cof[j]/++y;
   return -tmp+log(2.506628274631*ser/x);
}

#define ITMAX 100
#define EPSW 3.0e-7

static void gser(double *gamser,double a,double x,double *gln){
   int n;
   double sum,del,ap;
   *gln=loggam(a);
   if(x <= 0.0){
	  if(x < 0.0) nrerror2("x less than 0 in routine gser");
	  *gamser=0.0;
	  return;
   }
   else{
	  ap=a;
	  del=sum=1.0/a;
	  for(n=1;n<=ITMAX;n++){
		 ++ap;
		 del *= x/ap;
		 sum += del;
		 if(fabs(del) < fabs(sum)*EPSW){
			*gamser=sum*exp(-x+a*log(x)-(*gln));
			return;
		 }
	  }
	  nrerror2("a too large, ITMAX too small in routine gser");
	  return;
   }
}


#define FPMIN 1.0e-30

static void gcf(double *gammcf,double a,double x,double *gln){
   int i;
   double an,b,c,d,del,h;
   *gln=loggam(a);
   b=x+1.0-a;
   c=1.0/FPMIN;
   d=1.0/b;
   h=d;
   for(i=1;i<=ITMAX;i++){
	  an = -i*(i-a);
	  b += 2.0;
	  d=an*d+b;
	  if(fabs(d) < FPMIN) d=FPMIN;
	  c=b+an/c;
	  if(fabs(c) < FPMIN) c=FPMIN;
	  d=1.0/d;
	  del=d*c;
	  h *= del;
	  if(fabs(del-1.0) < EPSW) break;
   }
   if(i > ITMAX) nrerror2("a too large, ITMAX too small in gcf");
   *gammcf=exp(-x+a*log(x)-(*gln))*h;
}

// Gamma function
// source http://www.crbond.com/math.htm
static double gammaFunc(double x){
	int i,k,m;
	double ga,gr,r,z;

	static double g[] = {
		1.0,
		0.5772156649015329,
	   -0.6558780715202538,
	   -0.420026350340952e-1,
		0.1665386113822915,
	   -0.421977345555443e-1,
	   -0.9621971527877e-2,
		0.7218943246663e-2,
	   -0.11651675918591e-2,
	   -0.2152416741149e-3,
		0.1280502823882e-3,
	   -0.201348547807e-4,
	   -0.12504934821e-5,
		0.1133027232e-5,
	   -0.2056338417e-6,
		0.6116095e-8,
		0.50020075e-8,
	   -0.11812746e-8,
		0.1043427e-9,
		0.77823e-11,
	   -0.36968e-11,
		0.51e-12,
	   -0.206e-13,
	   -0.54e-14,
		0.14e-14};

	if (x > 171.0) return 1e308;    // This value is an overflow flag.
	if (x == (int)x) {
		if (x > 0.0) {
			ga = 1.0;               // use factorial
			for (i=2;i<x;i++) {
			   ga *= i;
			}
		 }
		 else
			ga = 1e308;
	 }
	 else {
		if (fabs(x) > 1.0) {
			z = fabs(x);
			m = (int)z;
			r = 1.0;
			for (k=1;k<=m;k++)
				r *= (z-k);
			z -= m;
		}
		else
			z = x;
		gr = g[24];
		for (k=23;k>=0;k--)
			gr = gr*z+g[k];

		ga = 1.0/(gr*z);
		if (fabs(x) > 1.0) {
			ga *= r;
			if (x < 0.0)
				ga = -M_PI/(x*ga*sin(M_PI*x));
		}
	}
	return ga;
}


// Beta function
static double betaFunc(double alpha, double beta){
	double value=gammaFunc(alpha)*gammaFunc(beta)/gammaFunc(alpha+beta);
	return value;
}


// log gamma using the Lanczos approximation
static double logGamma(double x) {
const double c[8] = { 676.5203681218851, -1259.1392167224028,
		 771.32342877765313, -176.61502916214059,
		 12.507343278686905, -0.13857109526572012,
		 9.9843695780195716e-6, 1.5056327351493116e-7 };
double sum = 0.99999999999980993;
double y = x;
for (int j = 0; j < 8; j++)
  sum += c[j] / ++y;
return log(sqrt(2*3.14159) * sum / x) - (x + 7.5) + (x + 0.5) * log(x + 7.5);
}

// helper function for incomplete beta
// computes continued fraction

static double betaContFrac(double a, double b, double x) {
	const int MAXIT = 1000;
	const double EPS = 3e-7;
	double qab = a + b;
	double qap = a + 1;
	double qam = a - 1;
	double c = 1;
	double d = 1 - qab * x / qap;
	if (fabs(d) < FPMIN) d = FPMIN;
	d = 1 / d;
	double h = d;
	int m;
	for (m = 1; m <= MAXIT; m++) {
	  int m2 = 2 * m;
	  double aa = m * (b-m) * x / ((qam + m2) * (a + m2));
	  d = 1 + aa * d;
	  if (fabs(d) < FPMIN) d = FPMIN;
	  c = 1 + aa / c;
	  if (fabs(c) < FPMIN) c = FPMIN;
	  d = 1 / d;
	  h *= (d * c);
	  aa = -(a+m) * (qab+m) * x / ((a+m2) * (qap+m2));
	  d = 1 + aa * d;
	  if (fabs(d) < FPMIN) d = FPMIN;
	  c = 1 + aa / c;
	  if (fabs(c) < FPMIN) c = FPMIN;
	  d = 1 / d;
	  double del = d*c;
	  h *= del;
	  if (fabs(del - 1) < EPS) break;
	}
	if (m > MAXIT) {
	  cerr << "betaContFrac: too many iterations\n";
	}
	return h;
}

// incomplete beta function
// must have 0 <= x <= 1
static double betaInc(double a, double b, double x) {
  if (x == 0)
	return 0;
  else if (x == 1)
	return 1;
  else {
	double logBeta = logGamma(a+b) - logGamma(a) - logGamma(b)
	  + a * log(x) + b * log(1-x);
	if (x < (a+1) / (a+b+2))
	  return exp(logBeta) * betaContFrac(a, b, x) / a;
	else
	  return 1 - exp(logBeta) * betaContFrac(b, a, 1-x) / b;
  }
}




#endif /* BETA_GAMMA_FUNC_H_ */
