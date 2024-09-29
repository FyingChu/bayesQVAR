// [[Rcpp::depends(RcppEigen)]]
#include "estBQVAR.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Cholesky>    // Cholesky decomposition of posterior variance matrix
#include <cmath>             // pow, round
#include <iostream>          // debugging and printing
#include <chrono>            // timing
#include <variant>           // variant type
#include "bayesQVAR_types.h" // defining types of Eigen Matrix and Vector, Rcpp NumericMatrix and NumericVector
#include "constDesignMat.h"  // constructing design matrices
#include "basicSamplers.h"   // random number generators
#include "manipMatDfList.h"  // basic matrix manipulations
#include "qrGibbsSampler.h"  // Gibbs samplers for QVAR estimation

using namespace Rcpp;
using namespace Eigen;
using namespace std;

/*
@brief Complete prior specification for QVAR model, given a `Rcpp::List` of incomplete prior specification, even a `NULL`.
@param prior_incomplete: a `Rcpp::Nullable<Rcpp::List>` of incomplete prior specification.
@param S2: a `Eigen::MatrixXd` of prior parameter Sigma for residuals.
@param n_end: an `int` of the number of endogenous variables in QVAR model.
@param n_x: an `int` of the number of columns in the design matrix X.
*/
List completePrior(
    const Rcpp::Nullable<List> &prior_incomplete,
    const EigenMat &S2,
    const int &n_end,
    const int &n_x)
{
    List priorList_complete;
    if (prior_incomplete.isNotNull())
    {
        priorList_complete = prior_incomplete.get();
        if (!priorList_complete.containsElementNamed("mu_A"))
        {
            priorList_complete["mu_B"] = EigenMat::Zero(n_end, n_x); // why named "mu_B"? Since the function `qrgibbsSampler_al` uses "mu_b" as the name of the prior mean of the autoregressive coefficients.
        }
        else
        {
            priorList_complete["mu_B"] = priorList_complete["mu_A"];
        }
        if (!priorList_complete.containsElementNamed("Sigma_A"))
        {
            List prior_SigmaA(n_end);
            for (int i = 0; i < n_end; ++i)
            {
                prior_SigmaA[i] = 100 * EigenMat::Identity(n_x, n_x);
            }
            priorList_complete["Sigma_B"] = prior_SigmaA;
        }
        else
        {
            priorList_complete["Sigma_B"] = priorList_complete["Sigma_A"];
        }
        if (!priorList_complete.containsElementNamed("Sigma"))
        {
            priorList_complete["Sigma"] = S2; // n_end * n_end
        }
        if (!priorList_complete.containsElementNamed("nu"))
        {
            priorList_complete["nu"] = n_end + 1;
        }
        if (!priorList_complete.containsElementNamed("n_delta"))
        {
            priorList_complete["n_delta"] = EigenVec::Ones(n_end);
        }
        if (!priorList_complete.containsElementNamed("s_delta"))
        {
            priorList_complete["s_delta"] = EigenVec::Ones(n_end);
        }
    }
    else
    {
        List prior_SigmaA(n_end);
        for (int i = 0; i < n_end; ++i)
        {
            prior_SigmaA[i] = 100 * EigenMat::Identity(n_x, n_x);
        }
        priorList_complete = List::create(
            Named("mu_B") = EigenMat::Zero(n_end, n_x),
            Named("Sigma_B") = prior_SigmaA,
            Named("Sigma") = S2,
            Named("nu") = n_end + 1,
            Named("n_delta") = EigenVec::Ones(n_end),
            Named("s_delta") = EigenVec::Ones(n_end));
    }
    return priorList_complete;
}

/*
@brief Complete sampler setting for QVAR model, given a `Rcpp::List` of incomplete sampler setting, even a `NULL`.
@param samplerSetting_incomplete: a `Rcpp::Nullable<Rcpp::List>` of incomplete sampler setting.
@param S2: a `Eigen::MatrixXd` of prior parameter Sigma for residuals.
@param n_end: an `int` of the number of endogenous variables in QVAR model.
@param n_x: an `int` of the number of columns in the design matrix X.
@return a `Rcpp::List` of complete sampler setting.
*/
List completeSamplerSetting(
    const Rcpp::Nullable<List> &samplerSetting_incomplete,
    const EigenMat &S2,
    const int &n_end,
    const int &n_x)
{
    List samplerSettingList_complete;
    if (samplerSetting_incomplete.isNotNull())
    {
        samplerSettingList_complete = samplerSetting_incomplete.get();
        if (!samplerSettingList_complete.containsElementNamed("init_A"))
        {
            samplerSettingList_complete["init_B"] = EigenMat::Zero(n_end, n_x);
        }
        else
        {
            samplerSettingList_complete["init_B"] = samplerSettingList_complete["init_A"];
        }
        if (!samplerSettingList_complete.containsElementNamed("init_Sigma"))
        {
            samplerSettingList_complete["init_Sigma"] = S2;
        }
        if (!samplerSettingList_complete.containsElementNamed("init_delta"))
        {
            samplerSettingList_complete["init_delta"] = 0.1 * EigenVec::Ones(n_end);
        }
        if (!samplerSettingList_complete.containsElementNamed("n_sample"))
        {
            samplerSettingList_complete["n_sample"] = 500;
        }
        if (!samplerSettingList_complete.containsElementNamed("n_burn"))
        {
            samplerSettingList_complete["n_burn"] = 500;
        }
        if (!samplerSettingList_complete.containsElementNamed("n_thin"))
        {
            samplerSettingList_complete["n_thin"] = 1;
        }
    }
    else
    {
        samplerSettingList_complete = List::create(
            Named("init_B") = EigenMat::Zero(n_end, n_x),
            Named("init_Sigma") = S2,
            Named("init_delta") = 0.1 * EigenVec::Ones(n_end),
            Named("n_sample") = 500,
            Named("n_burn") = 500,
            Named("n_thin") = 1);
    }

    return samplerSettingList_complete;
}

// @brief Estimate QVAR model using Bayesian estimation based on AL distribution or MAL distribution.
// @param data_end a `Rcpp::DataFrame` of endogenous variables.
// @param lag a `Rcpp::IntegerVector` of lag order of endogenous and exogenous variables in QVAR model.
// @param alpha a `Rcpp::NumericVector` of tail probability vector.
// @param data_exo a `Rcpp::Nullable<Rcpp::DataFrame>` of exogenous variables.
// @param prior a `Rcpp::Nullable<Rcpp::List>` of prior information.
// @param samplerSetting a `Rcpp::Nullable<Rcpp::List>` of sampling setting.
// @param method a `std::string` that specifies the estimation method. "bayes-al" for Bayesian estimation based on AL distribution, "bayes-mal" for Bayesian estimation based on MAL distribution.
// @param printFreq an `int` that specifies the frequency of printing the progress of estimation.
// @param mute a `bool` that specifies whether to print the progress of estimation.
// @return a `Rcpp::List` that contains MCMC chains and posterior mean estimates of QVAR model.
// [[Rcpp::export(.estBQVAR)]]
List estBQVAR(
    const RcppDf &data_end,
    const RcppIntVec &lag,
    const RcppNumVec &alpha,
    const Rcpp::Nullable<RcppDf> &data_exo = R_NilValue,
    const Rcpp::Nullable<List> &prior = R_NilValue,
    const Rcpp::Nullable<List> &samplerSetting = R_NilValue,
    const std::string &method = "bayes-al",
    const int &printFreq = 10,
    const bool &mute = false)
{
    List res; // output list

    /* #region Declare useful quantities: dimension of variables, sample size, lag orders */
    int lag_end = lag[0];
    int lag_exo = 0;
    if (data_exo.isNotNull())
    {
        if (lag.size() == 1)
        {
            lag_exo = lag[0];
        }
        else if (lag.size() == 2)
        {
            lag_exo = lag[1];
        }
        else
        {
            stop("'lag' must be either a scalar or a vector of length 2.");
        }
    }
    const int lag_max = std::max(lag_end, lag_exo);
    const int T = data_end.rows() - lag_max;
    const int n_end = data_end.cols();
    int n_exo = 0;
    if (data_exo.isNotNull())
    {
        RcppDf data_exo_notNull = Rcpp::as<RcppDf>(data_exo);
        n_exo = data_exo_notNull.cols();
    }
    int n_x = n_end * lag_end + n_exo * lag_exo + 1;
    /* #endregion */

    /* #region Declare tail-specific parameters */
    RcppNumVec alpha_nv;
    EigenVec xi;
    EigenVec s2;
    EigenMat S2;
    if (alpha.size() == 1)
    {
        /*
        If alpha is a scalar, then take it granted that all endogenous variables have the same alpha value. xi and s2 are vectors with constant elements.
        */
        const double alpha_scalar = Rcpp::as<double>(alpha);
        alpha_nv = Rcpp::as<RcppNumVec>(wrap(EigenVec::Constant(n_end, alpha_scalar)));
        const double xi_scalar = (1. - 2. * alpha_scalar) / alpha_scalar / (1. - alpha_scalar);
        const double s2_scalar = 2. / alpha_scalar / (1. - alpha_scalar);
        xi = EigenVec::Constant(n_end, xi_scalar);
        s2 = EigenVec::Constant(n_end, s2_scalar);
        S2 = s2.asDiagonal();
    }
    else if (alpha.size() == n_end)
    {
        /*
        If alpha is a vector of length n_end, then each endogenous variable has its own alpha value. xi and s2 are vectors with varying elements.
        */
        alpha_nv = alpha;
        const RcppNumVec xi_nv = (1. - 2. * alpha) / alpha / (1. - alpha);
        const RcppNumVec s2_nv = 2. / alpha / (1. - alpha);
        xi = Rcpp::as<EigenVec>(xi_nv);
        s2 = Rcpp::as<EigenVec>(s2_nv);
        S2 = s2.asDiagonal();
    }
    else
    {
        stop("'alpha' must be a vector of length 1 or n_end (the number of endogenous variables in QVAR model).");
    }
    /* #endregion */

    /* #region  Construct Y and X matrices */
    List designMatList = constDesignMat(data_end, data_exo, lag_end, lag_exo);
    EigenMat Y = designMatList["Y"];
    EigenMat X = designMatList["X"];
    /* #endregion */

    List priorList = completePrior(prior, S2, n_end, n_x); // complete prior specification

    /* #region Complete sampler setting and fetch number of sampling, burn-in and thinning interval */
    List samplerSettingList = completeSamplerSetting(samplerSetting, S2, n_end, n_x);
    const int n_sample = samplerSettingList["n_sample"];
    const int n_burn = samplerSettingList["n_burn"];
    const int n_thin = samplerSettingList["n_thin"];
    const int n_iter = n_burn + n_sample * n_thin; // total number of iterations
    /* #endregion */

    if (method == "bayes-al")
    {
        /*
        If method == "bayes-al", then devide the prior specification and sampler setting into components specific for each row equation of QVAR and input them into `qrGibbsSampler_al` function one by one.
        */

        /* #region Declare initial */
        const EigenMat init_B = samplerSettingList["init_B"];
        const EigenVec init_delta = samplerSettingList["init_delta"];
        /* #endregion */

        /* #region Declare matrices to save mcmc chains and posterior estimates  */
        RcppNumMat mcmc_w(n_iter, n_end * T);
        RcppNumMat mcmc_A(n_iter + 1, n_end * n_x);
        RcppNumMat mcmc_delta(n_iter + 1, n_end);
        RcppNumMat mcmc_lambda;
        RcppNumMat w_est(T, n_end);
        RcppNumMat A_est(n_end, n_x);
        RcppNumVec delta_est(n_end);
        /* #endregion */

        /* #region Fetch prior from comlete list of prior */
        const EigenMat prior_muB = priorList["mu_B"];
        const List prior_SigmaB = priorList["Sigma_B"];
        const EigenVec prior_nd = priorList["n_delta"];
        const EigenVec prior_sd = priorList["s_delta"];
        EigenMat prior_nl;
        EigenMat prior_sl;
        if (priorList.containsElementNamed("n_lambda") && priorList.containsElementNamed("s_lambda"))
        {
            mcmc_lambda = RcppNumMat(n_iter + 1, n_end * n_x);
            prior_nl = priorList["n_lambda"];
            prior_sl = priorList["s_lambda"];
        }
        /* #endregion */

        /* #region Loop: estimate each equation of QVAR using `qrGibbsSampler` */
        std::chrono::system_clock::time_point start;
        if (mute == false)
        {
            start = std::chrono::system_clock::now();
            std::time_t start_time = std::chrono::system_clock::to_time_t(start);
            Rcpp::Rcout << "Estimation started at " << std::ctime(&start_time) << std::endl;
        }
        for (int i = 0; i < n_end; ++i)
        {
            /* #region Construct prior specification and sampler setting for i-th row equation of QVAR */
            List priorList_i = List::create(
                Named("mu_b") = prior_muB.row(i),
                Named("Sigma_b") = prior_SigmaB[i],
                Named("n_delta") = prior_nd(i),
                Named("s_delta") = prior_sd(i));
            if (priorList.containsElementNamed("n_lambda") && priorList.containsElementNamed("s_lambda"))
            {
                priorList_i["n_lambda"] = prior_nl.row(i);
                priorList_i["s_lambda"] = prior_sl.row(i);
            }
            List samplerSettingList_i = List::create(
                Named("init_b") = init_B.row(i),
                Named("init_delta") = init_delta(i),
                Named("n_sample") = n_sample,
                Named("n_burn") = n_burn,
                Named("n_thin") = n_thin);
            /* #endregion */

            /* #region Call Gibbs Sampelr and fetch MCMC chains and posterior estimates */
            EigenVec y_i = Y.row(i).transpose();
            List res_qr_i = qrGibbsSampler_al(y_i, X.transpose(), alpha_nv[i], priorList_i, samplerSettingList_i, 0, true);
            List res_mcmc_i = res_qr_i["mcmcChains"];
            List res_est_i = res_qr_i["estimates"];
            RcppNumMat mcmc_b_i = res_mcmc_i["b"];
            RcppNumVec mcmc_delta_i = res_mcmc_i["delta"];
            RcppNumMat mcmc_w_i = res_mcmc_i["w"];
            RcppNumVec a_est_i = res_est_i["b"];
            double delta_est_i = res_est_i["delta"];
            RcppNumVec w_est_i = res_est_i["w"];
            /* #endregion */

            /* #region Save MCMC chains and posterior estimates into the matrices for the whole QVAR */
            for (int j = 0; j < n_x; ++j)
            {
                mcmc_A(_, j * n_end + i) = mcmc_b_i(_, j);
            }
            mcmc_delta(_, i) = mcmc_delta_i;
            for (int j = 0; j < T; ++j)
            {
                mcmc_w(_, j * n_end + i) = mcmc_w_i(_, j);
            }
            A_est.row(i) = a_est_i;
            w_est(_, i) = w_est_i;
            delta_est[i] = delta_est_i;

            if (priorList.containsElementNamed("n_lambda") && priorList.containsElementNamed("s_lambda"))
            {
                RcppNumMat mcmc_lambda_i = res_mcmc_i["lambda"];
                mcmc_lambda(_, i * n_x) = mcmc_lambda_i;
            }
            /* #endregion */

            if (mute == false)
            {
                if (printFreq != 0)
                {
                    if ((i + 1) % printFreq == 0 or i + 1 == n_end)
                    {
                        if (i + 1 == printFreq)
                        {
                            auto lastStep = start;
                        }
                        auto currentStep = std::chrono::system_clock::now();
                        std::chrono::duration<double> elapsed_seconds = currentStep - start;
                        Rcpp::Rcout << i + 1 << " equations have been estimated: " << std::round(elapsed_seconds.count() * 100.0) / 100.0 << "s" << std::endl;
                    }
                }
            }

        } // end of loop for each equation
        if (mute == false)
        {
            auto end = std::chrono::system_clock::now();
            std::time_t end_time = std::chrono::system_clock::to_time_t(end);
            Rcpp::Rcout << "Estimation finished at " << std::ctime(&end_time) << std::endl;
        }
        /* #endregion */

        /* #region Save MCMC chains  */
        List res_mcmc = List::create(
            Named("w") = mcmc_w,
            Named("A") = mcmc_A,
            Named("delta") = mcmc_delta);
        if (priorList.containsElementNamed("n_lambda") && priorList.containsElementNamed("s_lambda"))
        {
            res_mcmc["lambda"] = mcmc_lambda;
        }
        /* #endregion */

        /* #region save posterior estimates */
        List res_est = List::create(
            Named("w") = w_est,
            Named("A") = A_est,
            Named("delta") = delta_est);
        res["mcmcChains"] = res_mcmc;
        res["estimates"] = res_est;
        /* #endregion */
    }
    else if (method == "bayes-mal")
    {

        /*
        If method == "bayes-mal", then the whole prior specification and sampler setting can be inputed into `mqrGibbsSampler_mal` function directly.
        */

        /* #region Declare matrices to save MCMC chains and posterior estimates  */
        RcppNumMat mcmc_w(n_iter, T);
        RcppNumMat mcmc_A(n_iter + 1, n_end * n_x);
        RcppNumMat mcmc_Sigma(n_iter + 1, n_end * n_end);
        RcppNumMat mcmc_delta(n_iter + 1, n_end);
        RcppNumMat mcmc_lambda;
        RcppNumVec w_est(T);
        RcppNumMat A_est(n_end, n_x);
        RcppNumMat Sigma_est(n_end, n_end);
        RcppNumVec delta_est(n_end);
        /* #endregion */

        /* #region Call Gibbs sampler and fetch MCMC chains and estimates */
        List res_qvar = mqrGibbsSampler_mal(Y, X, alpha_nv, priorList, samplerSettingList, printFreq, mute);
        List res_mcmc = res_qvar["mcmcChains"];
        List res_est = res_qvar["estimates"];
        /* #endregion */

        /* #region change the names of elements of MCMC chains and estimates list and save them into `res` */
        res_est.names() = RcppCharVec::create("A", "Sigma", "w", "delta");
        if (priorList.containsElementNamed("n_lambda") && priorList.containsElementNamed("s_lambda"))
        {
            res_mcmc.names() = RcppCharVec::create("w", "A", "Sigma", "delta", "lambda");
        }
        else
        {
            res_mcmc.names() = RcppCharVec::create("w", "A", "Sigma", "delta");
        }
        res["mcmcChains"] = res_mcmc;
        res["estimates"] = res_est;
        /* #endregion */
    }
    else
    {
        stop("Invalid method. Choose either 'bayes-al' or 'bayes-mal'.");
    }

    /* #region Estimate residuals and save it to `res` */
    List res_est = res["estimates"];
    EigenMat A_est_eigen = Rcpp::as<EigenMat>(res_est["A"]);
    EigenMat resid_est_eigen = (Y - A_est_eigen * X).transpose();
    RcppNumMat resid_est = Rcpp::wrap(resid_est_eigen);
    RcppCharVec names = data_end.names();
    colnames(resid_est) = names;
    res["residuals"] = resid_est;
    /* #endregion */

    /* #region Save other components to `res` */
    List dataDfList = List::create(
        Named("data_end") = data_end);
    if (data_exo.isNotNull())
    {
        dataDfList["data_exo"] = Rcpp::as<RcppDf>(data_exo);
    }
    res["data"] = dataDfList;
    res["designMat"] = designMatList;
    res["lag"] = RcppNumVec::create(lag_end, lag_exo);
    res["alpha"] = alpha;
    res["method"] = method;
    res["prior"] = removeElementFromNamedList(priorList, RcppCharVec::create("mu_B", "Sigma_B"));
    res["samplerSetting"] = removeElementFromNamedList(samplerSettingList, RcppCharVec::create("init_B"));
    /* #endregion */

    return res;
}

//@brief: Estimate multiple QVAR models at different tail probability vectors and save autoregressive matrices
//@param modelSpecif a `Rcpp::List` that contains QVAR model specification information, including `data_end`, `data_exo`, `lag`, `pior`, `samplerSetting`, `method`. `data_end`: a `T` * `n_end` `Rcpp::DataFrame` of endogenous variables. `data_exo`: a `T` * `n_exo` `Rcpp::DataFrame` of exogenous variables. `lag`: a `Rcpp::IntegerVector` of length 1 or 2 that specifies the lag order of endogenous and exogenous variables. `prior`: a `Rcpp::List` that contains prior information. `samplerSetting`: a `Rcpp::List` that contains sampling setting. `method`: a `std::string` that specifies the estimation method. "bayes-al" for Bayesian estimation based on AL distribution, "bayes-mal" for Bayesian estimation based on MAL distribution..
//@param alphaMat: `Rcpp::NumericMatrix` that contains tail probability vectors. Each column contains the tail probability vector for each QVAR model.
//@return a `Rcpp::List` that contains autoregressive coefficients matrices and MCMC chains of QVAR mdoels.
// [[Rcpp::export(.estMultiBQVAR)]]
List estMultiBQVAR(
    const List &modelSpecif,
    const EigenMat &alphaMat)
{
    List res; // output List

    /* #region Fetch dataframe of endogenous and exogenous variables */
    RcppDf data_end(modelSpecif["data_end"]);
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

    /* #region Complete prior setting for QVAR model */
    Rcpp::Nullable<List> prior_nullable;
    if (modelSpecif.containsElementNamed("prior") && modelSpecif["prior"] != R_NilValue)
    {
        List prior_inputed = modelSpecif["prior"];
        prior_nullable = Rcpp::as<Nullable<List>>(prior_inputed);
    }
    else
    {
        prior_nullable = R_NilValue;
    }
    /*
    The second argument of priorComplete S2, which is the prior parameter Sigma for residuals, is set to the identity matrix of n_end * n_end temporarily. It will not be used when QVAR model is estimated with "bayes-al" method but will be in the case of using Bayesian estimation method based on MAL distribution. In latter case, the Sigma matrix will be reset.
    */
    List prior_complete = completePrior(prior_nullable, EigenMat::Identity(n_end, n_end), n_end, n_x);
    EigenMat prior_muB = prior_complete["mu_B"];
    List prior_SigmaB = prior_complete["Sigma_B"];
    EigenVec prior_nd = prior_complete["n_delta"];
    EigenVec prior_sd = prior_complete["s_delta"];
    EigenMat prior_nl;
    EigenMat prior_sl;
    if (prior_complete.containsElementNamed("n_lambda") && prior_complete.containsElementNamed("s_lambda"))
    {
        prior_nl = prior_complete["n_lambda"];
        prior_sl = prior_complete["s_lambda"];
    }
    /* #endregion */

    /* #region Complete sampler setting */
    Rcpp::Nullable<List> samplerSetting_nullable;
    if (modelSpecif.containsElementNamed("samplerSetting"))
    {
        List samplerSetting_inputed = modelSpecif["samplerSetting"];
        samplerSetting_nullable = Rcpp::as<Rcpp::Nullable<List>>(samplerSetting_inputed);
    }
    else
    {
        samplerSetting_nullable = R_NilValue;
    }
    List samplerSetting_complete = completeSamplerSetting(samplerSetting_nullable, EigenMat::Identity(n_end, n_end), n_end, n_x);
    EigenMat init_B = samplerSetting_complete["init_B"];
    EigenMat init_Sigma = samplerSetting_complete["init_Sigma"];
    EigenVec init_delta = samplerSetting_complete["init_delta"];
    int n_sample = samplerSetting_complete["n_sample"];
    int n_burn = samplerSetting_complete["n_burn"];
    int n_thin = samplerSetting_complete["n_thin"];
    std::string method = "bayes-al";
    if (modelSpecif.containsElementNamed("method"))
    {
        std::string method_inputed = modelSpecif["method"];
        method = method_inputed;
    }
    /* #endregion */

    /* #region Initialize the list to save autoregressive matrix of each alpha vector */
    int n_alpha = alphaMat.cols(); // How many tail probability vector are to be estimated, d
    List AList_eachAlphaVec(n_alpha);
    List mcmcChainList_eachAlphaVec(n_alpha);
    /* #endregion */

    /* #region Loop: estimate QVAR at each alpha vector and save autoregressive matrices */
    for (int i = 0; i < n_alpha; ++i)
    {

        /* #region Define the tail probability and corresponding parameter xi and s2  */
        EigenVec alpha_i_eigen = alphaMat.col(i);
        RcppNumVec alpha_i_nv = Rcpp::wrap(alpha_i_eigen);
        /* #endregion */

        /* #region Check if QVAR at current tail probability has been estimated */
        bool haveBeenEstimated = false;
        if (i > 0)
        {
            for (int j = i - 1; j >= 0; --j)
            {
                EigenVec alpha_j_eigen = alphaMat.col(j);
                bool isEqual_ij = alpha_i_eigen.isApprox(alpha_j_eigen, 1e-4);
                if (isEqual_ij == true)
                {
                    AList_eachAlphaVec[i] = AList_eachAlphaVec[j];
                    mcmcChainList_eachAlphaVec[i] = mcmcChainList_eachAlphaVec[j];
                    haveBeenEstimated = true;
                    break;
                }
            }
        }
        /* #endregion */

        /* #region If QVAR at alpha_i has not been estimated, then estiamte QVAR at the alpha_i and save the MCMC chains and posterior estimates of A */
        if (haveBeenEstimated == false)
        {
            List res_estBQVAR_i;
            if (method == "bayes-al")
            {
                res_estBQVAR_i = estBQVAR(
                    data_end,
                    lag,
                    alpha_i_nv,
                    data_exo,
                    prior_complete,
                    samplerSetting_complete,
                    "bayes-al",
                    0,
                    true);
            }
            else if (method == "bayes-mal")
            {
                /* #region Define the tail probability and corresponding parameter xi and s2  */
                EigenVec alpha_i_eigen = alphaMat.col(i);
                RcppNumVec alpha_i_nv = Rcpp::wrap(alpha_i_eigen);
                RcppNumVec s2_nv = 2. / alpha_i_nv / (1. - alpha_i_nv);
                EigenVec s2 = Rcpp::as<EigenVec>(s2_nv);
                EigenMat S2 = s2.asDiagonal();
                /* #endregion */

                /* #region Reset the prior and initial value of Sigma */
                prior_complete["Sigma"] = S2;
                samplerSetting_complete["init_Sigma"] = S2;
                /* #endregion */

                /* #region Estimate and save A_h */
                res_estBQVAR_i = estBQVAR(
                    data_end,
                    lag,
                    alpha_i_nv,
                    data_exo,
                    prior_complete,
                    samplerSetting_complete,
                    "bayes-mal",
                    0,
                    true);
                /* #endregion */
            }
            else
            {
                Rcpp::stop("Invalid estimation method.");
            }
            /* #region Estimate and save A_h */
            List estimates_i = res_estBQVAR_i["estimates"];
            List mcmcChains_i = res_estBQVAR_i["mcmcChains"];
            RcppNumMat A_i = estimates_i["A"];
            RcppNumMat mcmc_A_i = mcmcChains_i["A"];
            AList_eachAlphaVec[i] = A_i;
            mcmcChainList_eachAlphaVec[i] = mcmc_A_i;
            /* #endregion */
        }
        /* #endregion */
    }
    res["AList"] = AList_eachAlphaVec;
    res["mcmcChainList"] = mcmcChainList_eachAlphaVec;
    /* #endregion */

    return res;
}
