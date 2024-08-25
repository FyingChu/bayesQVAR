#ifndef BAYESQVAR_TYPES_H
#define BAYESQVAR_TYPES_H

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Eigen;

typedef Eigen::MatrixXd EigenMat;
typedef Eigen::VectorXd EigenVec;
typedef Rcpp::DataFrame RcppDf;
typedef Rcpp::NumericMatrix RcppNumMat;
typedef Rcpp::NumericVector RcppNumVec;
typedef Rcpp::IntegerVector RcppIntVec;
typedef Rcpp::CharacterVector RcppCharVec;
typedef Rcpp::StringVector RcppStrVec;
typedef Rcpp::LogicalVector RcppLogiVec;

#endif
