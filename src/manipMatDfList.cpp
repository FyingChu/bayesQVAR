// [[Rcpp::depends(RcppEigen)]]
#include "manipMatDfList.h"
#include <RcppEigen.h>
#include <Rcpp.h>
#include "bayesQVAR_types.h"

using namespace Rcpp;
using namespace Eigen;

EigenMat matPower(const EigenMat &X, const int &p)
{
  EigenMat out = EigenMat::Identity(X.rows(), X.cols());
  for (int i = 0; i < p; i++)
  {
    out = out * X;
  }
  return out;
}

RcppIntVec sequence(const int &start, const int &end, const int &stepSize)
{

  RcppIntVec out;
  for (int i = start; i <= end; i += stepSize)
  {
    out.push_back(i);
  }
  return out;
}

RcppNumMat matSubset(
    RcppNumMat &X,
    Rcpp::Nullable<RcppIntVec &> rowIdx = R_NilValue,
    Rcpp::Nullable<RcppIntVec &> colIdx = R_NilValue)
{
  int m = X.rows();
  int n = X.cols();
  RcppIntVec rows = Rcpp::seq(0, m - 1);
  RcppIntVec cols = Rcpp::seq(0, n - 1);
  if (rowIdx.isNotNull())
  {
    rows = rowIdx.get();
    m = rows.size();
  }
  if (colIdx.isNotNull())
  {
    cols = colIdx.get();
    n = cols.size();
  }

  RcppNumMat out(m, n);
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      out(i, j) = X(rows[i], cols[j]);
    }
  }
  return out;
}

RcppNumVec colMedianOfMat(const RcppNumMat &mat)
{
  RcppNumVec out(mat.cols());
  for (int i = 0; i < mat.cols(); i++)
  {
    RcppNumVec col_i = mat(_, i);
    out[i] = Rcpp::median(col_i);
  }
  return out;
}

RcppNumVec vecSubset(
    RcppNumVec &x,
    Rcpp::Nullable<RcppIntVec &> idx = R_NilValue)
{
  int m = x.length();
  RcppIntVec rows = Rcpp::seq(0, m - 1);
  if (idx.isNotNull())
  {
    rows = idx.get();
    m = rows.size();
  }
  RcppNumVec out(m);
  for (int i = 0; i < m; i++)
  {
    out[i] = x[rows[i]];
  }
  return out;
}

RcppNumMat moveColumn(
    const RcppNumMat &X,
    const int &from,
    const int &to)
{
  RcppNumVec col_from = X.column(from);
  RcppNumMat X_out = X;
  if (from < to)
  {
    for (int i = from; i < to; ++i)
    {
      X_out.column(i) = X.column(i + 1);
    }
    X_out.column(to) = col_from;
  }
  else
  {
    for (int i = from; i > to; --i)
    {
      X_out.column(i) = X.column(i - 1);
    }
    X_out.column(to) = col_from;
  }
  return X_out;
}

RcppNumMat replaceElement(const RcppNumMat &mat1, const RcppNumMat &mat2)
{
  RcppNumMat mat_out = mat1;
  for (int i = 0; i < mat1.nrow(); i++)
  {
    for (int j = 0; j < mat1.ncol(); j++)
    {
      if (!RcppNumMat::is_na(mat2(i, j)))
      {
        mat_out(i, j) = mat2(i, j);
      }
    }
  }
  return mat_out;
}

RcppDf convertMatrixToDataFrame(const RcppNumMat &mat)
{

  List df(mat.ncol());

  for (int i = 0; i < mat.ncol(); ++i)
  {
    df[i] = mat(_, i);
  }

  df.attr("names") = colnames(mat);
  df.attr("class") = "data.frame";
  df.attr("row.names") = RcppIntVec::create(NA_INTEGER, -mat.nrow());

  return df;
}

RcppDf removeDuplicateColumns(const RcppDf &df)
{

  RcppCharVec colnames = df.names();
  std::string colname_0 = Rcpp::as<std::string>(colnames[0]);
  List colUniqueList = List::create(Named(colname_0) = df[0]);

  for (int i = 1; i < df.size(); ++i)
  {
    RcppNumVec col_i = df[i];
    std::string colname_i = Rcpp::as<std::string>(colnames[i]);
    bool isUnique = true;

    for (int j = 0; j < colUniqueList.size(); ++j)
    {
      RcppNumVec colUnique_j = colUniqueList[j];
      if (Rcpp::is_true(Rcpp::all(colUnique_j == col_i)))
      {
        isUnique = false;
        break;
      }
    }

    if (isUnique)
    {
      colUniqueList.push_back(col_i, colname_i);
    }
  }
  return RcppDf(colUniqueList);
}

DataFrame subtractDataFrames(const RcppDf &df1, const RcppDf &df2)
{

  if (df1.ncol() != df2.ncol() || df1.nrow() != df2.nrow())
  {
    stop("Dimensions of input DataFrames do not match.");
  }

  RcppDf df_out = clone(df1);
  colnames(df_out) = colnames(df1);
  for (int i = 0; i < df1.ncol(); ++i)
  {
    RcppNumVec col_1i = clone(Rcpp::as<RcppNumVec>(df1[i]));
    RcppNumVec col_2i = clone(Rcpp::as<RcppNumVec>(df2[i]));
    df_out[i] = col_1i - col_2i;
  }

  return df_out;
}

List removeElementFromNamedList(const List &lst, const RcppCharVec &names_removed)
{
  List out;
  RcppCharVec names = lst.names();
  for (int i = 0; i < lst.size(); ++i)
  {
    std::string name_i = Rcpp::as<std::string>(names[i]);
    bool remove_i = false;
    for (int j = 0; j < names_removed.size(); ++j)
    {
      std::string name_removed_j = Rcpp::as<std::string>(names_removed[j]);
      if (name_i == name_removed_j)
      {
        remove_i = true;
        break;
      }
    }
    if (!remove_i)
    {
      out[name_i] = lst[i];
    }
  }
  return out;
}
