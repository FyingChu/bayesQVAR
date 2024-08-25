#ifndef BQVAREST_H
#define BQVAREST_H

#include <Rcpp.h>
#include "bayesQVAR_types.h"

using namespace Rcpp;

List completePrior(
    const Rcpp::Nullable<List> &prior,
    const EigenMat &S2,
    const int &n_end,
    const int &n_x);

List completeSamplerSetting(
    const Rcpp::Nullable<List> &samplerSetting,
    const EigenMat &S2,
    const int &n_end,
    const int &n_x);

List estBQVAR(
    const RcppDf &data_end,
    const RcppIntVec &lag,
    const RcppNumVec &alpha,
    const Rcpp::Nullable<RcppDf> &data_exo,
    const Rcpp::Nullable<List> &prior,
    const Rcpp::Nullable<List> &samplerSetting,
    const std::string &method,
    const int &printFreq,
    const bool &mute);

List estMultiBQVAR(
    const List &modelSpecif,
    const EigenMat &alphaMat);

EigenMat estVARbyOLS(
    const EigenMat &Y,
    const EigenMat &X);

List estBVAR(
    const RcppDf &data_end,
    const RcppIntVec &lag,
    const Rcpp::Nullable<RcppDf> &data_exo,
    const List &prior,
    const List &samplerSetting);

EigenVec estSigmaOfBVAR(
    const RcppDf &data_end,
    const RcppIntVec &lag,
    const Rcpp::Nullable<RcppDf> &data_exo,
    const List &prior,
    const List &samplerSetting);

#endif
