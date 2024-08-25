// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include "bayesQVAR_types.h"
#include "constDesignMat.h"
#include "manipMatDfList.h"
#include "qrGibbsSampler.h"
#include "estBQVAR.h"

using namespace std;
using namespace Rcpp;
using namespace Eigen;

// @brief Forecast quantiles of endogenous variables in QVAR model given horizon.
// @param modelSpecif a `Rcpp::List` that contains QVAR model secification information, including `data_end`, `data_exo`, `data_exo_forecast`, `lag`, `pior`, `samplerSetting`, `method`. `data_end`: a `T` * `n_end` `Rcpp::DataFrame` of endogenous variables. `data_exo`: a `T` * `n_exo` `Rcpp::DataFrame` of exogenous variables. `data_exo_forecast` (optional): a `n_exo`-column `Rcpp::dataframe` with at least `horizon - 1` rows. `lag`: a `Rcpp::IntegerVector` of length 1 or 2 that specifies the lag order of endogenous and exogenous variables. `prior`: a `Rcpp::List` that contains prior information. `samplerSetting`: a `Rcpp::List` that contains sampling setting. `method`: a `std::string` that specifies the estimation method. "bayes-al" for Bayesian estimation based on AL distribution, "bayes-mal" for Bayesian estimation based on MAL distribution.
// @param horizon an `int` that represents the number of periods to forecast.
// @param probPath an `Eigen::MatrixXd` of probability paths. Each column contains the probability values for each variable at each period. The number of columns should be equal to `horizon`.
// @param mean a `bool`. If true, the mean forecast will be calculated. If false, the only quantile forecast along the probability path will be calculated.
// @return res a `Rcpp::List` that contains the forecasted quantiles for particular evolve path and forecasted mean quantiles.
// [[Rcpp::export(.forecastQuant)]]
List forecastQuant(
    const List &modelSpecif,
    const int &horizon,
    const EigenMat &probPath,
    const bool mean = false)
{

    List res;          // output List
    List res_forecast; // output List for forecasted quantiles

    /* #region Check input validity of probPath */
    if (probPath.cols() < horizon)
    {
        Rcpp::stop("The number of columns in probPath must be at least the same as forecast horizon.");
    }
    else if (probPath.cols() > horizon)
    {
        Rcpp::Rcout << "Redundant probabilities are provided, only " << horizon << " paths will be used." << std::endl;
    }
    /* #endregion */

    /* #region Fetch dataframe of endogenous and exogenous variables, forecast of exogenous variables */
    RcppDf data_end = modelSpecif["data_end"];
    RcppNumMat data_end_nm = Rcpp::internal::convert_using_rfunction(data_end, "as.matrix");
    EigenMat data_end_eigen = Rcpp::as<EigenMat>(data_end_nm);
    RcppDf data_exo;
    EigenMat data_exo_eigen;
    RcppNumMat data_exo_forecast_nm;
    EigenMat data_exo_forecast_eigen;
    if (modelSpecif.containsElementNamed("data_exo") && modelSpecif["data_exo"] != R_NilValue)
    {
        data_exo = modelSpecif["data_exo"];
        RcppNumMat data_exo_nm = Rcpp::internal::convert_using_rfunction(data_exo, "as.matrix");
        data_exo_eigen = Rcpp::as<EigenMat>(data_exo_nm);

        if (modelSpecif.containsElementNamed("data_exo_forecast") && modelSpecif["data_exo_forecast"] != R_NilValue)
        {
            RcppDf data_exo_forecast = modelSpecif["data_exo_forecast"];
            if (data_exo_forecast.rows() != horizon - 1)
            {
                stop("The number of rows in data_exo_forecast must be the same as forecast horizon - 1.");
            }
            data_exo_forecast_nm = Rcpp::internal::convert_using_rfunction(data_exo_forecast, "as.matrix");
            data_exo_forecast_eigen = Rcpp::as<EigenMat>(data_exo_forecast_nm);
        }
        else
        {
            RcppNumMat data_exo_nm = Rcpp::internal::convert_using_rfunction(data_exo, "as.matrix");
            EigenVec colMeans_exo = Rcpp::as<EigenVec>(Rcpp::colMeans(data_exo_nm));
            data_exo_forecast_eigen = colMeans_exo.replicate(1, horizon);
            data_exo_forecast_nm = Rcpp::wrap(data_exo_forecast_eigen.transpose());
        }
    }
    else
    {
        // do nothing
    }
    RcppCharVec colNames_exo = data_exo.names();
    colnames(data_exo_forecast_nm) = colNames_exo;
    res_forecast["data_exo"] = data_exo_forecast_nm;
    /* #endregion */

    /* #region Define dimensions and lag order of endogenous and exogenous variables */
    int n_end = data_end.cols();
    int n_exo = data_exo.cols();
    RcppIntVec lag = modelSpecif["lag"];
    int lag_end = lag[0];
    int lag_exo = 0;
    if (n_exo != 0)
    {
        if (lag.size() == 1)
        {
            lag_exo = lag[0];
        }
        else
        {
            lag_exo = lag[1];
        }
    }
    int n_x = n_end * lag_end + n_exo * lag_exo + 1;
    /* #endregion */

    /* #region Construct design matrices and initilize matrix to save forecast result  */
    List designMat = constDesignMat(data_end, data_exo, lag_end, lag_exo);
    EigenMat X = designMat["X"];
    EigenMat Y = designMat["Y"];
    EigenMat Y_forecast_eigen(n_end, horizon);
    /* #endregion */

    /* #region Estimate autoregressive matrix at all horizons */
    List AList_eachHorizon = estMultiBQVAR(modelSpecif, probPath)["AList"];
    /* #endregion */

    /* #region Initialize the temporary variable `X_h`, which will update during forecasting */
    EigenVec X_h = X.rightCols(1);
    for (int j = 0; j < lag_end; ++j)
    {
        X_h.middleRows(n_end * j, n_end) = data_end_eigen.row(data_end_eigen.rows() - 1 - j).transpose();
    }
    if (n_exo != 0)
    {
        for (int j = 0; j < lag_exo; ++j)
        {
            X_h.middleRows(n_end * lag_end + n_exo * j, n_exo) = data_exo_eigen.row(data_exo_eigen.rows() - 1 - j).transpose();
        }
    }
    /* #endregion */

    /* #region Loop: Forcast and update X_h at each horizon */
    for (int h = 0; h < horizon; ++h)
    {

        /* #region Forecast `Y_h` and save it to `Y_forecast_eigen` */
        EigenMat A_h = Rcpp::as<EigenMat>(AList_eachHorizon[h]);
        EigenVec Y_h = A_h * X_h;
        Y_forecast_eigen.col(h) = Y_h;
        /* #endregion */

        /* #region Update `X_h` */
        /* annotation:
        updating X_h needs the lastest updated Y_h.
        Move the first n_end * (lag_end - 1) rows of X_h by n_end rows down,
        then replace the first n_end rows of X_h with Y_h.
        If data_exo is provided, move the first n_exo * (lag_exo - 1) rows of exogenous part of X_h by n_exo rows down,
        then replace the first n_exo rows of the exogenous part with the current row of data_exo_predicted_eigen.
        */
        if (h < horizon - 1)
        {
            EigenMat x_end_unchanged = X_h.topRows(n_end * (lag_end - 1));
            X_h.middleRows(n_end, n_end * (lag_end - 1)) = x_end_unchanged;
            X_h.topRows(n_end) = Y_h;
            if (data_exo.cols() != 0)
            {
                EigenMat x_exo_unchanged = X_h.middleRows(n_end * lag_end, n_exo * (lag_exo - 1));
                X_h.middleRows(n_end * lag_exo, n_exo * (lag_exo - 1)) = x_exo_unchanged;
                X_h.middleRows(n_end * lag_end, n_exo) = data_exo_forecast_eigen.col(h);
            }
        }
        /* #endregion */
    }
    RcppNumMat Y_forecast = Rcpp::wrap(Y_forecast_eigen.transpose());
    Rcpp::CharacterVector colNames_end = data_end.names();
    colnames(Y_forecast) = colNames_end;
    res_forecast["quant"] = Y_forecast;
    /* #endregion */

    /* #region If mean is true, then use mean autoregressive matrix to perform forecast */
    if (mean == true)
    {

        /* #region Estimate mean autoregressive matrix */
        List designMat = constDesignMat(data_end, data_exo, lag_end, lag_exo);
        EigenMat X = designMat["X"];
        List prior_BVAR = modelSpecif["prior"];
        prior_BVAR["Sigma"] = 100 * EigenMat::Identity(n_end, n_end);
        prior_BVAR["nu"] = n_end;
        List samplerSetting_BVAR = modelSpecif["samplerSetting"];
        samplerSetting_BVAR["init_Sigma"] = 100 * EigenMat::Identity(n_end, n_end);
        List res_estBVAR = estBVAR(data_end, lag, data_exo, prior_BVAR, samplerSetting_BVAR);
        List estimates_mean = res_estBVAR["estimates"];
        EigenMat A_mean = estimates_mean["A"];
        /* #endregion */

        EigenMat X_mean_h = X.rightCols(1);
        for (int j = 0; j < lag_end; ++j)
        {
            X_mean_h.middleRows(n_end * j, n_end) = data_end_eigen.row(data_end_eigen.rows() - 1 - j);
        }
        if (data_exo.cols() != 0)
        {
            for (int j = 0; j < lag_exo; ++j)
            {
                X_mean_h.middleRows(n_end * lag_end + n_exo * j, n_exo) = data_exo_eigen.row(data_exo_eigen.rows() - 1 - j);
            }
        }

        EigenMat Y_mean_forecast_eigen(n_end, horizon);
        for (int h = 0; h < horizon; ++h)
        {

            EigenVec Y_mean_h = A_mean * X_mean_h;
            EigenMat A_h = Rcpp::as<EigenMat>(AList_eachHorizon[h]);
            EigenVec Y_h = A_h * X_mean_h;
            Y_mean_forecast_eigen.col(h) = Y_h;

            if (h < horizon - 1)
            {
                EigenMat x_end_unchanged = X_mean_h.topRows(n_end * (lag_end - 1));
                X_mean_h.middleRows(n_end, n_end * (lag_end - 1)) = x_end_unchanged;
                X_mean_h.topRows(n_end) = Y_mean_h;
                if (data_exo.cols() != 0)
                {
                    EigenMat x_exo_unchanged = X_mean_h.middleRows(n_end * lag_end, n_exo * (lag_exo - 1));
                    X_mean_h.middleRows(n_end * lag_exo, n_exo * (lag_exo - 1)) = x_exo_unchanged;
                    X_mean_h.middleRows(n_end * lag_end, n_exo) = data_exo_forecast_eigen.col(h);
                }
            }
        }

        RcppNumMat Y_mean_forecast = Rcpp::wrap(Y_mean_forecast_eigen.transpose());
        colnames(Y_mean_forecast) = colNames_end;
        res_forecast["quant_mean"] = Y_mean_forecast;
    }
    /* #endregion */

    /* #region Save other output into list `res` */
    res["forecastList"] = res_forecast;
    res["modelSpecif"] = modelSpecif;
    res["mean"] = mean;
    res["horizon"] = horizon;
    res["probPath"] = probPath;
    /* #endregion */

    return res;
}
