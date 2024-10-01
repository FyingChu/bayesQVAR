// [[Rcpp::depends(RcppEigen)]]
#include "qrGibbsSampler.h"
#include <RcppEigen.h>
#include <Rcpp.h>
#include <Eigen/Cholesky>
#include <math.h>            // pow
#include <iostream>          // debugging and printing
#include <chrono>            // timing
#include "bayesQVAR_types.h" // defining types of Eigen Matrix and Vector, Rcpp NumericMatrix and NumericVector
#include "basicSamplers.h"   // random number generators
#include "manipMatDfList.h"  // basic matrix manipulations

using namespace Rcpp;
using namespace Eigen;

/*
@brief This function is a MCMC algorithm performing Bayesian estimation fot single-equation linear quantile regression.
@brief y = x * b + u, where u ~ AL(0, delta, xi, w)}
@brief where AL is asymmetric Laplace distribution, which has following mixture representation: u = delta * xi * w + sqrt(w) * delta * sigma * z
@note The detailed reparameterization of AL distribution refers to Kotz et al.(2012).
@param y a `n` * `1` `Eigen::VectorXd` of explained variable, where `n` is the number of observations.
@param X a `n` * `k` `Eigen::MatrixXd` of explanatory variables, of which the last column should be constant 1, where `n` is the number of observations and `k` is the number of explanatory variables.
@param alpha a `double` that represents the cumulative tail probability of quantile regression.
@param prior a `Rcpp:List` of prior components, including `mu_b`, `Sigma_b`, `n_delta`, `s_delta`, `n_lambda`. `mu_b`: a `k` * `1` `Eigen::VectorXd`, is the prior mean of coefficient vector b. `Sigma_b`: a `k` * `k` `Eigen::MatrixXd`, is the prior covariance matrix of coefficient vector b. `n_delta`: a `double`, is the prior shape parameter of inverse gamma distribution of delta. `s_delta`: a `double`, is the prior scale parameter of inverse gamma distribution of delta. `n_lambda`: a `double`, is the prior shape parameter of inverse gamma distribution of penalty. `s_lambda`: a `double` scalar, is the prior scale parameter of inverse gamma distribution of penalty.
@param samplerSetting: a `Rcpp::List` of sampler settings, including `init_b`, `init_delta`, `sampleSize`, `burnIn`, `thin`. `init_b`: a `k` * `1` `Eigen::VectorXd`, is the initial value of coefficient vector b. `init_delta`: a `double`, is the initial value of delta. `sampleSize`: a `int`, is the number of samples to draw. `burnIn`: an `int`, is the number of burn-in samples. `thin`: a `int`, is the thinning interval.
@param printFreq: an `int`, is the frequency of printing out the iteration number. Message will be printed every `printFreq` iterations.
@return a `Rcpp::List` of mcmc chains and posterior median estimates.
*/
List qrGibbsSampler_al(
    const EigenVec &y,   // vector of explained variable, n * 1
    const EigenMat &X,   // matrix of data, n * k, the last column is the constant term 1
    const double &alpha, // tail probability, scalar
    const List &prior,
    const List &samplerSetting,
    const int &printFreq = 500,
    const bool &mute = false)
{
    List res; // output list

    int k = X.cols(); // number of explanatory variables
    int n = X.rows(); // number of observations

    /* #region Declare tail-specified parameters */
    double xi = (1. - 2. * alpha) / alpha / (1. - alpha); // xi = (1 - 2 * alpha) / alpha / (1 - alpha)
    double s2 = 2. / alpha / (1. - alpha);                // s2 = 2 / alpha / (1 - alpha)
    /* #endregion */

    /* #region Extract prior setting components from `prior` */
    EigenVec prior_mub = prior["mu_b"];       // prior mean of b
    EigenMat prior_Sigmab = prior["Sigma_b"]; // prior covariance of b
    double prior_nd = prior["n_delta"];       // shape parameter of prior distribtuion of delta
    double prior_sd = prior["s_delta"];       // scale parameter of prior distribution of delta
    EigenVec prior_nl;                        // shape parameter of prior distribution of lambda
    EigenVec prior_sl;                        // scale parameter of prior distribution of lambda
    if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
    {
        prior_nl = prior["n_lambda"];
        prior_sl = prior["s_lambda"];
    }
    /* #endregion */

    /* #region Extract sampler setting components from `samplerSetting` */
    EigenVec init_b = samplerSetting["init_b"];       // initial value of b
    double init_delta = samplerSetting["init_delta"]; // initial value of delta
    int n_sample = samplerSetting["n_sample"];        // number of samples to draw
    int n_burn = samplerSetting["n_burn"];            // number of burn-in samples
    int n_thin = samplerSetting["n_thin"];            // thinning interval
    const int n_iter = n_burn + n_sample * n_thin;    // total number of iterations
    /* #endregion */

    /* #region Declare matrices to save MCMC chains */
    RcppNumMat mcmc_w(n_iter, n);          // w only need to save the result of `n_iter` sampling since it will be first updated in the Gibbs sampling
    RcppNumMat mcmc_b(n_iter + 1, k);      // the first row needs to save the initial value
    RcppNumVec mcmc_delta(n_iter + 1);     // the first element needs to save the initial value
    RcppNumMat mcmc_lambda(n_iter + 1, k); // the first row needs to save the initial value
    mcmc_b.row(0) = Rcpp::as<RcppNumVec>(wrap(init_b));
    mcmc_delta[0] = init_delta;
    mcmc_lambda.row(0) = Rcpp::as<RcppNumVec>(wrap(EigenVec::Ones(k)));
    /* #endregion */

    /* #region Declare temporary variables that will be updated with loop */
    EigenVec w_h(n);
    EigenVec b_h = init_b;
    double delta_h = init_delta;
    EigenVec lambda_h = EigenVec::Ones(k);
    /* #endregion */

    /* #region Loop: Gibbs sampling */
    std::chrono::system_clock::time_point start;
    if (mute == false)
    {
        start = std::chrono::system_clock::now();
        std::time_t start_time = std::chrono::system_clock::to_time_t(start);
        Rcout << "Gibbs sampling started at " << std::ctime(&start_time) << std::endl;
    }
    for (int h = 0; h < n_iter; ++h)
    {

        /* #region Update */
        for (int i = 0; i < n; ++i)
        {
            double ew_ih = y(i) - X.row(i) * b_h;
            double d = pow(xi, 2) / s2 + 2.0;
            double m = pow(ew_ih, 2) / pow(delta_h, 2) / s2;
            Rcpp::NumericVector w_i = rGIG(
                Named("n") = 1,
                Named("lambda") = 0.5,
                Named("psi") = d,
                Named("chi") = m);
            w_h[i] = w_i[0];
        }
        mcmc_w.row(h) = Rcpp::as<RcppNumVec>(wrap(w_h));
        /* #endregion */

        /* #region Update b_h */
        EigenVec ea_h = y - delta_h * xi * w_h;
        EigenMat Qa_1 = (X.transpose() * w_h.asDiagonal().inverse() * X) / pow(delta_h, 2) / s2;
        EigenMat Qa_2 = (X.transpose() * w_h.asDiagonal().inverse() * ea_h) / pow(delta_h, 2) / s2;
        Eigen::MatrixXd post_Vb = (Qa_1 + (prior_Sigmab * lambda_h.asDiagonal()).inverse()).inverse();
        Eigen::VectorXd post_mub = post_Vb * (Qa_2 + (prior_Sigmab * lambda_h.asDiagonal()).inverse() * prior_mub);
        Rcpp::NumericVector x_a = Rcpp::rnorm(k);
        b_h = Rcpp::as<Map<VectorXd>>(x_a);
        Eigen::MatrixXd L_a = post_Vb.llt().matrixL();
        b_h = (L_a * b_h + post_mub).eval();
        mcmc_b.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(b_h));
        /* #endregion */

        /* #region Ppdate delta_h */
        Eigen::VectorXd ed_h = y - X * b_h - delta_h * xi * w_h;
        Eigen::VectorXd vSqrt_h = (delta_h * w_h).cwiseSqrt();
        double post_nd = prior_nd + 3.0 * n;
        double Q_d = (ed_h.transpose().cwiseProduct(vSqrt_h.cwiseInverse())).squaredNorm() / s2;
        double post_sd = prior_sd + 2.0 * vSqrt_h.cwiseAbs2().sum() + Q_d;
        delta_h = rInvGamma(0.5 * post_nd, 0.5 * post_sd);
        mcmc_delta[h + 1] = delta_h;
        /* #endregion */

        /* #region Update penalty parameter if prior parameter of lambda is provided */
        if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
        {
            for (int i = 0; i < k; ++i)
            {
                double post_nl_ih = prior_nl(i) + 1;
                double post_sl_ih = pow(b_h[i], 2) / prior_Sigmab.diagonal()[i] + prior_sl(i);
                lambda_h[i] = rInvGamma(0.5 * post_nl_ih, 0.5 * post_sl_ih);
            }
            mcmc_lambda.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(lambda_h));
        }
        /* #endregion */

        /* #region Print message if `mute == false` */
        if (mute == false)
        {
            if (printFreq != 0)
            {
                if ((h + 1) % printFreq == 0 or h + 1 == n_iter)
                {
                    if (h + 1 == printFreq)
                    {
                        auto lastStep = start;
                    }
                    auto currentStep = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_seconds = currentStep - start;
                    Rcpp::Rcout << "Iterations " << h + 1 << ": " << std::round(static_cast<float>(elapsed_seconds.count() * 100)) / 100.0 << "s" << std::endl;
                }
            }
        }
        /* #endregion */
    } // end of Gibbs sampling
    /* #endregion */

    /* #region Print message when sampling finished if `mute == false` */
    if (mute == false)
    {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        Rcpp::Rcout << "Gibbs sampling finished at " << std::ctime(&end_time) << std::endl;
    }
    /* #endregion */

    /* #region Save mcmc chains into a `Rcpp::List` */
    List res_mcmc = List::create(
        Named("w") = mcmc_w,
        Named("b") = mcmc_b,
        Named("delta") = mcmc_delta);
    if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
    {
        res_mcmc["lambda"] = mcmc_lambda;
    }
    /* #endregion */

    /* #region Calculate posterior median and save them into `Rcpp::List` */
    RcppIntVec idx_thin = sequence(n_burn, n_iter - 1, n_thin);
    RcppNumVec b_est = colMedianOfMat(matSubset(mcmc_b, idx_thin, R_NilValue));
    RcppNumVec w_est = colMedianOfMat(matSubset(mcmc_w, idx_thin, R_NilValue));
    double delta_est = Rcpp::mean(vecSubset(mcmc_delta, idx_thin));
    List res_est = List::create(
        Named("b") = b_est,
        Named("w") = w_est,
        Named("delta") = delta_est);
    /* #endregion */

    res["mcmcChains"] = res_mcmc;
    res["estimates"] = res_est;

    return res;
}
