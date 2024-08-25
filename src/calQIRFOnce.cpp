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

// @brief Calculate impulse response function based on a list of autoregressive matrices
// @param AList_eachHorizon a `Rcpp::List` list of autoregressive matrices at each horizon
// @param B0 an `Eigen::MatrixXd` initial coefficient matrix
// @param idx_impulse a `Rcpp::IntegerVector` index of impulse variables
// @param idx_response a `Rcpp::IntegerVector` index of response variables
// @param n_end an `int` number of endogenous variables
// @param lag_end an `int` number of lags for endogenous variables
// @param mean a `bool` that represents whether to calculate mean impulse response function
// @param A_mean an `Eigen::MatrixXd` mean autoregressive matrix
// @return `Rcpp::DataFrame` impulse response function
// [[Rcpp::export(.calQIRFwithA)]]
RcppDf calQIRFwithA(
    const List &AList_eachHorizon,
    const EigenVec &sigmaSqrt,
    const EigenMat &A0,
    const RcppIntVec &idx_impulse,
    const RcppIntVec &idx_response,
    const int &n_end,
    const int &lag_end,
    const bool &mean,
    const EigenMat &A_mean)
{

    /* #region Declare horizon, number of response and impulse variables */
    int horizon = AList_eachHorizon.size();
    int n_response = idx_response.size();
    int n_impulse = idx_impulse.size();
    /* #endregion */

    /* #region Initialize DataFrame to save QIRF */
    std::vector<RcppNumVec> irfVec = std::vector<RcppNumVec>(
        n_impulse * n_response,
        RcppNumVec(horizon + 1));
    RcppDf irf(irfVec); // (horizon + 1) * (n_impulse * n_response) data frame
    /* #endregion */

    /* #region Construct selection matrix J */
    EigenMat J = EigenMat::Zero(n_end, n_end * lag_end);    // J matrix is used to extract the n * n top corner of prod_A
    J.leftCols(n_end) = EigenVec::Ones(n_end).asDiagonal(); // the first n columns of J is an identity matrix
    /* #endregion */

    /* #region Expand A_mean */
    EigenMat A_mean_h;
    if (mean == true)
    {
        A_mean_h = A_mean;
        A_mean_h.conservativeResize(n_end, n_end * lag_end); // delete the coefficients on exogenous variables and constant term
        A_mean_h.conservativeResize(n_end * lag_end, n_end * lag_end);
        A_mean_h.bottomLeftCorner(n_end * (lag_end - 1), n_end * (lag_end - 1)) = EigenMat::Identity(n_end * (lag_end - 1), n_end * (lag_end - 1));
        A_mean_h.bottomRightCorner(n_end * (lag_end - 1), n_end) = EigenMat::Zero(n_end * (lag_end - 1), n_end);
    }
    /* #endregion */

    /* #region Loop: Calcualte QIRF or mean QIRF, depending on `mean` */
    EigenMat prod_A_h = EigenMat::Identity(n_end * lag_end, n_end * lag_end); // initialize the product of autoregressive matrices
    for (int h = 0; h < horizon + 1; ++h)
    {
        if (h == 0)
        {
            // do nothing
            // h == 0 means the instantaneous response to the impulse
        }
        else
        {
            EigenMat A_h = Rcpp::as<EigenMat>(AList_eachHorizon[h - 1]); // extract autoregressive matrix at h-th horizon
            A_h.conservativeResize(n_end, n_end * lag_end);              // delete the coefficients on exogenous variables and constant term
            /* #region Expand A_h */
            A_h.conservativeResize(n_end * lag_end, n_end * lag_end);
            A_h.bottomLeftCorner(n_end * (lag_end - 1), n_end * (lag_end - 1)) = EigenMat::Identity(n_end * (lag_end - 1), n_end * (lag_end - 1));
            A_h.bottomRightCorner(n_end * (lag_end - 1), n_end) = EigenMat::Zero(n_end * (lag_end - 1), n_end);
            if (mean == true)
            {
                prod_A_h = A_h * matPower(A_mean_h, h - 1);
            }
            else
            {
                prod_A_h = (A_h * prod_A_h).eval();
            }
            /* #endregion */
        }

        for (int i = 0; i < n_impulse; ++i)
        {
            EigenVec shock_orthogonal_i = EigenVec::Zero(n_end);
            shock_orthogonal_i(idx_impulse[i]) = 1.0;
            EigenVec shock_i = A0 * sigmaSqrt.asDiagonal() * shock_orthogonal_i;
            EigenVec response_h = J * prod_A_h * J.transpose() * shock_i;
            for (int j = 0; j < n_response; ++j)
            {
                Rcpp::NumericVector irfCol_ij = Rcpp::clone<Rcpp::NumericVector>(irf[i * n_response + j]);
                // if (i == j)
                // {
                //     irfCol_ij[0] = sigmaSqrt(idx_impulse[i]);
                // }
                // else
                // {
                //     irfCol_ij[0] = 0.0;
                // }
                irfCol_ij[h] = response_h(idx_response[j]);
                irf[i * n_response + j] = irfCol_ij;
            } // end of loop: j (response variables)
        } // end of loop: i (impulse variables)
    } // end of loop: h (horizon)
    /* #endregion */

    return irf;
}

// @brief Calculate quantile impulse response function (QIRF) and mean QIRF, based on model specification list. If counterfactual model specification list is provided, calculate counterfactual QIRF, counterfactual mean QIRF and their differnce to actual counterpart.
// @param modelSpecif a `Rcpp::List` model specification list
// @param names_impulse a `Rcpp::CharacterVector` names of impulse variables
// @param names_response a `Rcpp::CharacterVector` names of response variables
// @param sigmaSqrt an `Eigen::VectorXd` square root of diagonal elements of covariance matrix of residuals
// @param horizon an `int` number of horizons
// @param probPath an `Eigen::MatrixXd` probability matrix of interest
// @param mean a `bool` that represents whether to calculate mean impulse response function
// @param counterfactual a `Rcpp::Nullable<Rcpp::List>` counterfactual model specification list
// @return `Rcpp::List` of quantile impulse response function, mean impulse response function, counterfactual impulse response function, and difference between counterfactual and actual impulse response function
// [[Rcpp::export(.calQIRFOnce)]]
List calQIRFOnce(
    const List &modelSpecif,
    const RcppCharVec &names_impulse,
    const RcppCharVec &names_response,
    const EigenVec &sigmaSqrt,
    const int &horizon,
    const EigenMat &probPath,
    const bool &mean = false,
    const Nullable<List> &counterfactual = R_NilValue)
{
    List res; // output List

    /* #region Fetch dataframe of endogenous and exogenous variables */
    RcppDf data_end = modelSpecif["data_end"];
    RcppDf data_exo;
    if (modelSpecif.containsElementNamed("data_exo") && modelSpecif["data_exo"] != R_NilValue)
    {
        data_exo = modelSpecif["data_exo"];
    }
    else
    {
        // do nothing
    }
    /* #endregion */

    /* #region Define dimensions and lag order of endogenous and exogenous variables */
    int n_end = data_end.cols();
    int n_exo = data_exo.cols(); // 0 if data_exo is not provided
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

    /* #region Construct design matrices */
    List designMat = constDesignMat(data_end, data_exo, lag_end, lag_exo);
    EigenMat Y = designMat["Y"];
    EigenMat X = designMat["X"];
    /* #endregion */

    /* #region Find the column position at which response and impulse variables are in `data_end` */
    int n_response = names_response.length();      // number of response variables
    int n_impulse = names_impulse.length();        // number of impulse variables
    const RcppCharVec colNames = data_end.names(); // column names of `data_end`
    RcppIntVec idx_response(n_response);
    RcppIntVec idx_impulse(n_impulse);
    for (int i = 0; i < n_response; ++i)
    {
        for (int j = 0; j < n_end; ++j)
        {
            if (colNames[j] == names_response[i])
            {
                idx_response[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < n_impulse; ++i)
    {
        for (int j = 0; j < n_end; ++j)
        {
            if (colNames[j] == names_impulse[i])
            {
                idx_impulse[i] = j;
                break;
            }
        }
    }
    /* #endregion */

    /* #region Initialize a `DataFrame` to save QIRF */
    std::vector<RcppNumVec> irfVec(n_impulse * n_response, RcppNumVec(horizon + 1));
    RcppDf irf(irfVec); // (horizon + 1) * (n_impulse * n_response) data frame
    RcppCharVec colNames_irf(n_impulse * n_response);
    for (int i = 0; i < n_impulse; ++i)
    {
        for (int j = 0; j < n_response; ++j)
        {
            colNames_irf[i * n_response + j] = "response " + names_response[j] + " to " + names_impulse[i];
        }
    }
    /* #endregion */

    /* #region Estimate autoregressive matrices along ProbMat of interest */
    List res_estMultiBQVAR = estMultiBQVAR(modelSpecif, probPath);
    List AList_eachHorizon = res_estMultiBQVAR["AList"];
    List mcmcChainList_A = res_estMultiBQVAR["mcmcChainList"];
    res["mcmcChainList_A"] = mcmcChainList_A;
    /* #endregion */

    /* #region Fetch A0 from `modelSpecif` */
    EigenMat A0 = EigenMat::Identity(n_end, n_end); // the diagonals of A0 should be 1
    if (modelSpecif.containsElementNamed("A0"))
    {
        A0 = modelSpecif["A0"];
    }
    else if (modelSpecif.containsElementNamed("B0"))
    {
        EigenMat B0 = modelSpecif["B0"];
        A0 = (EigenMat::Identity(n_end, n_end) - B0).inverse();
    }
    else
    {
        // do nothing
    }
    /* #endregion */

    /* #region Calculate QIRF along `probMat` of interest */
    irf = calQIRFwithA(
        AList_eachHorizon, sigmaSqrt, A0,
        idx_impulse, idx_response,
        n_end, lag_end,
        false, EigenMat::Identity(n_end, n_end));
    irf.names() = colNames_irf;
    res["irf"] = irf;
    /* #endregion */

    /* #region Calculate mean QIRF along `probMat` of interest */
    RcppDf irf_mean(irfVec);
    EigenMat A_mean;
    if (mean == true)
    {
        List prior_BVAR = clone(Rcpp::as<List>(modelSpecif["prior"]));
        prior_BVAR["Sigma"] = EigenMat::Identity(n_end, n_end);
        prior_BVAR["nu"] = 0;
        List samplerSetting_BVAR = modelSpecif["samplerSetting"];
        samplerSetting_BVAR["init_Sigma"] = EigenMat::Identity(n_end, n_end);
        List res_estBVAR = estBVAR(data_end, lag, data_exo, prior_BVAR, samplerSetting_BVAR);
        List estimates_mean = res_estBVAR["estimates"];
        A_mean = estimates_mean["A"];
        irf_mean = calQIRFwithA(
            AList_eachHorizon, sigmaSqrt, A0,
            idx_impulse, idx_response,
            n_end, lag_end,
            true, A_mean);
        irf_mean.names() = colNames_irf;
        res["irf_mean"] = irf_mean;

        List mcmcChainList_Amean = res_estBVAR["mcmcChains"];
        res["mcmcChain_Amean"] = mcmcChainList_Amean["a"];
    }
    /* #endregion */

    /* #region Calculate counterfactual QIRF */
    if (counterfactual.isNotNull())
    {
        List counterList = counterfactual.get();
        EigenMat A0_counter = A0;
        List AList_counter_eachHorizon(horizon);
        if (counterList.containsElementNamed("A0_counter") &&
            counterList.containsElementNamed("Ap_counter"))
        {
            RcppNumMat A0_counter_withNA_nm = counterList["A0_counter"];
            RcppNumMat Ap_counter_withNA_nm = counterList["Ap_counter"];
            RcppNumMat A0_nm = wrap(A0);
            RcppNumMat A0_counter_nm = replaceElement(A0_nm, A0_counter_withNA_nm);
            A0_counter = Rcpp::as<EigenMat>(A0_counter_nm);

            for (int h = 0; h < horizon; ++h)
            {
                EigenMat A_h = AList_eachHorizon[h];
                RcppNumMat A_h_nm = wrap(A_h);
                RcppNumMat A_counter_h_nm = replaceElement(A_h_nm, Ap_counter_withNA_nm);
                EigenMat A_counter_h = Rcpp::as<EigenMat>(A_counter_h_nm);
                AList_counter_eachHorizon[h] = A_counter_h;
            }
        }
        else if (counterList.containsElementNamed("B0_counter") &&
                 counterList.containsElementNamed("Bp_counter"))
        {
            RcppNumMat B0_counter_withNA_nm = counterList["B0_counter"];
            RcppNumMat Bp_counter_withNA_nm = counterList["Bp_counter"];
            RcppNumMat B0_nm = wrap(EigenMat::Identity(n_end, n_end) - A0.inverse());
            RcppNumMat B0_counter_nm = replaceElement(B0_nm, B0_counter_withNA_nm);
            EigenMat B0_counter = Rcpp::as<EigenMat>(B0_counter_nm);
            A0_counter = (EigenMat::Identity(n_end, n_end) - B0_counter).inverse();

            for (int h = 0; h < horizon; ++h)
            {
                EigenMat A_h = AList_eachHorizon[h];
                EigenMat B_h = A0.inverse() * A_h;
                RcppNumMat B_h_nm = wrap(B_h);
                RcppNumMat B_counter_h_nm = replaceElement(B_h_nm, Bp_counter_withNA_nm);
                EigenMat B_counter_h = Rcpp::as<EigenMat>(B_counter_h_nm);
                AList_counter_eachHorizon[h] = A0_counter * B_counter_h;
            }
        }
        else
        {
            stop("Invalid counterfactual matrices specification.");
        }

        RcppDf irf_counter = calQIRFwithA(
            AList_counter_eachHorizon, sigmaSqrt, A0_counter,
            idx_impulse, idx_response,
            n_end, lag_end,
            false, EigenMat::Identity(n_end, n_end));
        irf_counter.names() = colNames_irf;
        res["irf_counter"] = irf_counter;

        RcppDf diffIrf = subtractDataFrames(irf_counter, irf);
        res["diffIrf"] = diffIrf;

        if (mean == true)
        {
            EigenMat A_mean_counter;
            if (counterList.containsElementNamed("Ap_mean_counter"))
            {
                RcppNumMat A_mean_counter_withNA_nm = counterList["Ap_mean_counter"];
                RcppNumMat A_mean_nm = wrap(A_mean);
                RcppNumMat A_mean_counter_nm = replaceElement(A_mean_nm, A_mean_counter_withNA_nm);
                A_mean_counter = Rcpp::as<EigenMat>(A_mean_counter_nm);
            }
            else if (counterList.containsElementNamed("Bp_mean_counter"))
            {
                RcppNumMat B_mean_counter_withNA_nm = counterList["Bp_mean_counter"];
                EigenMat B_mean = A0.inverse() * A_mean;
                RcppNumMat B_mean_nm = wrap(B_mean);
                RcppNumMat B_mean_counter_nm = replaceElement(B_mean_nm, B_mean_counter_withNA_nm);
                EigenMat B_mean_counter = Rcpp::as<EigenMat>(B_mean_counter_nm);
                A_mean_counter = A0_counter * B_mean_counter;
            }
            else
            {
                if (counterList.containsElementNamed("Ap_counter"))
                {
                    RcppNumMat Ap_counter_withNA_nm = counterList["Ap_counter"];
                    RcppNumMat A_mean_nm = wrap(A_mean);
                    RcppNumMat A_mean_counter_nm = replaceElement(A_mean_nm, Ap_counter_withNA_nm);
                    A_mean_counter = Rcpp::as<EigenMat>(A_mean_counter_nm);
                }
                else if (counterList.containsElementNamed("Bp_counter"))
                {
                    RcppNumMat Bp_counter_withNA_nm = counterList["Bp_counter"];
                    EigenMat B_mean = A0.inverse() * A_mean;
                    RcppNumMat B_mean_nm = wrap(B_mean);
                    RcppNumMat B_mean_counter_nm = replaceElement(B_mean_nm, Bp_counter_withNA_nm);
                    EigenMat B_mean_counter = Rcpp::as<EigenMat>(B_mean_counter_nm);
                    A_mean_counter = A0_counter * B_mean_counter;
                }
            }

            RcppDf irf_mean_counter = calQIRFwithA(
                AList_counter_eachHorizon, sigmaSqrt, A0_counter,
                idx_impulse, idx_response,
                n_end, lag_end,
                true, A_mean_counter);
            irf_mean_counter.names() = colNames_irf;
            res["irf_mean_counter"] = irf_mean_counter;

            RcppDf diffIrf_mean = subtractDataFrames(irf_mean_counter, irf_mean);
            res["diffIrf_mean"] = diffIrf_mean;
        }
    }
    /* #endregion */

    return res;
}
