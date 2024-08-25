#ifndef MANIPMATDFLIST_H
#define MANIPMATDFLIST_H

#include <Rcpp.h>
#include "bayesQVAR_types.h"

using namespace Rcpp;

EigenMat matPower(const EigenMat &X, const int &p);

RcppIntVec sequence(const int &start, const int &end, const int &stepSize);

RcppNumMat matSubset(
    RcppNumMat &X,
    Rcpp::Nullable<RcppIntVec &> rowIdx,
    Rcpp::Nullable<RcppIntVec &> colIdx);

RcppNumVec colMedianOfMat(const RcppNumMat &mat);

RcppNumVec vecSubset(
    RcppNumVec &x,
    Rcpp::Nullable<RcppIntVec &> idx);

RcppNumMat moveColumn(const RcppNumMat &X, const int &from, const int &to);

RcppNumMat replaceElement(const RcppNumMat &mat1, const RcppNumMat &mat2);

RcppDf convertMatrixToDataFrame(const RcppNumMat &mat);

RcppDf removeDuplicateColumns(const RcppDf &df);

RcppDf subtractDataFrames(const RcppDf &df1, const RcppDf &df2);

List removeElementFromNamedList(const List &lst, const RcppCharVec &names_removed);

#endif
