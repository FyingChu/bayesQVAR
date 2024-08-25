#include "basicSamplers.h"
#include <Rcpp.h>

using namespace Rcpp;

/*
@brief random number generator for Generalized Inverse Gaussian distribution imported from the namespace of R package `GIGrvg`
@note https://cran.r-project.org/web/packages/GIGrvg/index.html
@param n number of random numbers to generate
@param lambda shape parameter
@param chi scale parameter
@param psi rate parameter
@return `Rcpp::NumericVector` of random vector
*/
Rcpp::Function rGIG = Rcpp::Environment::namespace_env("GIGrvg")["rgig"];

/*
@brief random number generator for inverse Wishart distribution imported from the namespace of R package 'LaplacesDemon'
@note https://cran.r-project.org/web/packages/LaplacesDemon/index.html
@note https://en.wikipedia.org/wiki/Inverse-Wishart_distribution
@param n number of random numbers to generate
@param df degrees of freedom
@param S scale matrix
@return `Rcpp::NumericMatrix` of positive definite random matrix
*/
Rcpp::Function rInvWishart = Rcpp::Environment::namespace_env("LaplacesDemon")["rinvwishart"];

/*
@brief random number generator for inverse gamma distribution
@note https://en.wikipedia.org/wiki/Inverse-gamma_distribution
@param n number of random numbers to generate
@param shape shape parameter
@param scale rate parameter
@return `double` of random number
*/
double rInvGamma(const double &shape, const double &scale)
{
  return 1.0 / R::rgamma(shape, 1.0 / scale); // Be careful with the order of shape and invRate
}
