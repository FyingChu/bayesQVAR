// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include "bayesQVAR_types.h"
#include "constDesignMat.h"
#include "manipMatDfList.h"

using namespace std;
using namespace Rcpp;
using namespace Eigen;

// @brief Simulate a new data set based on original data and a list of coefficient estimates of QVAR model
// @param data_end `Rcpp::DataFrame` of endogenous variables, (`lag_max` + `samplSize`) * `n_end`
// @param data_exo `Rcpp::DataFrame` data frame exogenous variables, (`lag_max` + `samplSize`) * `n_exo`
// @param AList_eachProb `Rcpp::List` list of coefficient matrices of QVAR model for each cumulative probability
// @param lag_end `int` number of lags for endogenous variables
// @param lag_exo `int` number of lags for exogenous variables
// @param sampleSize `int` number of observations to simulate
// @return `Rcpp::List` list of simulated data frames of endogenous and exogenous variables
// [[Rcpp::export(.bootsSimuOnce)]]
List bootsSimuOnce(
    const RcppDf &data_end,
    const RcppDf &data_exo,
    const int &lag_end,
    const int &lag_exo,
    const int &sampleSize,
    const List &AList_eachProb)
{

    /* #region Declare useful quantities */
    int n_end = data_end.cols();
    int n_exo = data_exo.cols();
    int lag_max = max(lag_end, lag_exo);
    int n_prob = AList_eachProb.size(); // how dense the probability space is devided
    /* #endregion */

    /* #region Contruct design matrices Y and X */
    List designMat = constDesignMat(data_end, data_exo, lag_end, lag_exo);
    EigenMat Y = designMat["Y"];
    EigenMat X = designMat["X"];
    /* #endregion */

    /* #region Calculate residuals for QVAR model estimated at each probability */
    List residList_eachProb(n_prob); // list to save residual matrix of QVAR for each probability
    for (int p = 0; p < n_prob; ++p)
    {
        EigenMat A_p = Rcpp::as<EigenMat>(AList_eachProb[p]); // coefficient matrix of QVAR model at probability p
        EigenMat resid_p = Y - A_p * X;
        residList_eachProb[p] = resid_p;
    }
    /* #endregion */

    EigenMat probIdx_realized(n_end, sampleSize); // matrix that save realized cumulative probabilities of each endogenous variable at each time point

    /* #region Determine which cumulative probability was realized for each endogenous variable at each time */
    for (int t = 0; t < sampleSize; ++t)
    {
        EigenMat resid_t(n_end, n_prob);
        for (int p = 0; p < n_prob; ++p)
        {
            EigenMat resid_p = residList_eachProb[p];
            resid_t.col(p) = resid_p.col(t);
        }
        EigenVec resid_min = resid_t.cwiseAbs().rowwise().minCoeff();
        for (int i = 0; i < n_end; ++i)
        {
            for (int p = 0; p < n_prob; ++p)
            {
                if (abs(resid_t(i, p)) == resid_min(i))
                {
                    probIdx_realized(i, t) = p;
                    break;
                }
            }
        }
    }
    /* #endregion */

    /* #region Transpose and convert DataFrame of original data into MatrixXd */
    RcppNumMat data_end_nm = Rcpp::internal::convert_using_rfunction(data_end, "as.matrix");
    RcppNumMat data_exo_nm = Rcpp::internal::convert_using_rfunction(data_exo, "as.matrix");
    EigenMat data_end_eigen = Rcpp::as<EigenMat>(transpose(data_end_nm));
    EigenMat data_exo_eigen = Rcpp::as<EigenMat>(transpose(data_exo_nm));
    /* #endregion */

    /* #region Initialize matrix of simulated data, assign the initial value to the first lag_end columns */
    EigenMat data_end_simulated_eigen(n_end, sampleSize + lag_max);
    EigenMat data_exo_simulated_eigen(n_exo, sampleSize + lag_max);
    data_end_simulated_eigen.leftCols(lag_max) = data_end_eigen.leftCols(lag_max);
    data_exo_simulated_eigen.leftCols(lag_max) = data_exo_eigen.leftCols(lag_max);
    /* #endregion */

    /* #region Sample indices from original ordered indices with replacement */
    RcppIntVec idx_original = seq_len(sampleSize) - 1;
    RcppIntVec idx_sample = Rcpp::sample(idx_original, sampleSize, true);
    /* #endregion */

    /* #region Loop: simulate data based on sample index */
    for (int t = 0; t < sampleSize; ++t)
    {

        int idx_t = idx_sample[t]; // the position at which the realized probability and exogenous variable to be sampled
        /* #region update X_h */
        EigenVec X_t(n_end * lag_end + n_exo * lag_exo + 1);
        for (int j = 0; j < lag_end; ++j)
        {
            X_t.middleRows(n_end * j, n_end) = data_end_simulated_eigen.col(t + lag_max - j - 1);
        }
        for (int j = 0; j < lag_exo; ++j)
        {
            X_t.middleRows(n_end * lag_end + n_exo * j, n_exo) = data_exo_simulated_eigen.col(t + lag_max - j - 1);
        }
        X_t(X_t.size() - 1) = 1;
        /* #endregion */
        /* #region Construct autoregressive matrix by realized probability */
        EigenMat A_t(n_end, n_end * lag_end + n_exo * lag_exo + 1);
        for (int i = 0; i < n_end; ++i)
        {
            int probIdx_it = probIdx_realized(i, idx_t);
            EigenMat A_p = Rcpp::as<EigenMat>(AList_eachProb[probIdx_it]);
            A_t.row(i) = A_p.row(i);
        }
        /* #endregion */
        /* #region Simulate endogenous variable recursively and extract the sampled observation of exogenous variable */
        data_end_simulated_eigen.col(t + lag_max) = A_t * X_t;
        data_exo_simulated_eigen.col(t + lag_max) = data_exo_eigen.col(idx_t + lag_max);
        /* #endregion */
    }
    /* #endregion */

    /* #region Transpose and convert MatrixXd of simulated data into DataFrame */
    RcppNumMat data_end_simulated_nm = wrap(data_end_simulated_eigen);
    RcppNumMat data_exo_simulated_nm = wrap(data_exo_simulated_eigen);
    RcppDf data_end_simulated = convertMatrixToDataFrame(transpose(data_end_simulated_nm));
    RcppDf data_exo_simulated = convertMatrixToDataFrame(transpose(data_exo_simulated_nm));
    data_end_simulated.names() = data_end.names();
    data_exo_simulated.names() = data_exo.names();
    /* #endregion */

    return List::create(
        Named("data_end") = data_end_simulated,
        Named("data_exo") = data_exo_simulated);
}
