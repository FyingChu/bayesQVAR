// [[Rcpp::depends(RcppEigen)]]
#include "constDesignMat.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include "bayesQVAR_types.h"
#include "manipMatDfList.h"

using namespace std;
using namespace Rcpp;
using namespace Eigen;

// @brief Construct design matrices Y and X for QVAR model based on data frames of endogenous and exogenous variables and number of lags for each variable.
// @param data_end `T + lag_max` * `n_end` `Rcpp::DataFrame` of endogenous variables.
// @param data_exo (optional) `T + lag_max` * `n_exo` `Rcpp::DataFrame` of exogenous variables.
// @param lag_end `int` number of lags for endogenous variables.
// @param lag_exo `int` number of lags for exogenous variables. If `data_exo` is not provided, `lag_exo` is treated as 0 no matter what value is passed to it.
// @return `Rcpp::List` list of design matrices Y and X, Y is `n_end` * T matrix and X is (`n_end` * `lag_end` + `n_exo` * `lag_exo` + 1) * `T` matrix.
// [[Rcpp::export(.constDesignMat)]]
List constDesignMat(
    const RcppDf &data_end,
    const Rcpp::Nullable<RcppDf> &data_exo = R_NilValue,
    const int &lag_end = 1,
    const int &lag_exo = 0)
{

    /* #region Declare quantities for model demension, sample size */
    const int lag_max = max(lag_end, lag_exo);
    const int T = data_end.rows() - lag_max;
    const int n_end = data_end.cols();
    RcppDf data_exo_notNull = R_NilValue;
    int n_exo = 0;
    if (data_exo.isNotNull())
    {
        data_exo_notNull = Rcpp::as<RcppDf>(data_exo);
        n_exo = data_exo_notNull.cols();
    }
    /* #endregion */

    /* #region Remove duplicate columns from dataEnd */
    RcppDf data_end_unique = removeDuplicateColumns(data_end);
    const int n_end_unique = data_end_unique.cols();
    /* #endregion */

    /* #region Transpose and convert Rcpp::DataFrame of endogenous and exogenous variables into Eigen::MatrixXd */
    const RcppNumMat data_end_rcpp = Rcpp::internal::convert_using_rfunction(data_end, "as.matrix");
    const RcppNumMat data_end_unique_rcpp = Rcpp::internal::convert_using_rfunction(data_end_unique, "as.matrix");
    const EigenMat data_end_eigen = Rcpp::as<EigenMat>(transpose(data_end_rcpp));
    const EigenMat data_end_unique_eigen = Rcpp::as<EigenMat>(transpose(data_end_unique_rcpp));
    /* #endregion */

    const EigenMat Y = data_end_eigen.rightCols(T); // n_end * T design matrix of endogenous variables

    /* #region Loop: construct design matrix of explanatory variabels */
    EigenMat X(n_end_unique * lag_end + n_exo * lag_exo + 1, T); // n_end * lag_end + n_exo * lag_exo + 1 * T design matrix of explainatory variables
    for (int i = 0; i < lag_end; ++i)
    {
        X.middleRows(n_end_unique * i, n_end_unique) = data_end_unique_eigen.middleCols(lag_max - i - 1, T);
    }
    if (data_exo.isNotNull())
    {
        const RcppNumMat data_exo_rcpp = Rcpp::internal::convert_using_rfunction(data_exo_notNull, "as.matrix");
        const EigenMat data_exo_eigen = Rcpp::as<MatrixXd>(transpose(data_exo_rcpp));
        for (int i = 0; i < lag_exo; ++i)
        {
            X.middleRows(n_end_unique * lag_end + n_exo * i, n_exo) = data_exo_eigen.middleCols(lag_max - i - 1, T);
        }
        /* annotation:
        If data frame of exogenous variables is provided, the lagged term of exogenous variables are added to X after lagged endogenous variables. The order of lag is lag_exo.
        */
    }
    X.row(n_end_unique * lag_end + n_exo * lag_exo) = EigenVec::Ones(T); // add a row of 1s to the bottom of X
    /* #endregion */

    return List::create(
        Named("Y") = Y,
        Named("X") = X);
}
