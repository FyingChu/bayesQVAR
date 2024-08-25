# ifndef BASICSAMPLERS_H
# define BASICSAMPLERS_H

# include <Rcpp.h>
using namespace Rcpp;

extern Rcpp::Function rGIG;
extern Rcpp::Function rInvWishart;
double rInvGamma(const double &shape, const double &scale);

# endif
