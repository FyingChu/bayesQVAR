# ifndef QRBAYESSAMPLER_H
# define QRBAYESSAMPLER_H

# include <Rcpp.h>
# include "bayesQVAR_types.h"

using namespace Rcpp;

List qrGibbsSampler_al(
    const EigenVec& y,         // vector of explained variable, n * 1
    const EigenMat& X,         // matrix of data, n * k, the 1st column is the constant term 1
    const double& alpha,              // tail probability, scalar
    const List& prior,
    const List& samplerSetting,
    const int& printFreq,
    const bool& mute
);

List mqrGibbsSampler_mal(
    const EigenMat& Y,         // vector of explained variable, n * 1
    const EigenMat& X,         // matrix of data, n * k, the 1st column is the constant ter 1
    const RcppNumVec& alpha,              // tail probability, scalar
    const List& prior,
    const List& samplerSetting,
    const int& printFreq,
    const bool& mute
);

# endif
