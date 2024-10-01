// [[Rcpp::depends(RcppEigen)]]
#include "qrGibbsSampler.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Cholesky> // Cholesky decomposition of posterior variance matrix
#include <cmath>
#include <iostream>          // debugging and printing
#include <chrono>            // timing
#include "bayesQVAR_types.h" // defining types of Eigen Matrix and Vector, Rcpp NumericMatrix and NumericVector
#include "basicSamplers.h"   // random number generators
#include "manipMatDfList.h"  // basic matrix manipulations

using namespace Rcpp;
using namespace Eigen;
using namespace std;

/*
@brief Gibbs sampler for Bayesian estimation of QVAR model based on MAL distribution.
@brief The model is defined as follows:
@brief Y = B X + U, U ~ MAL(0, D * xi(alpha), D * Sigma(alpha) * D ), where Y is `k_y` * `1` vector, X is `k_x` * `1` vector and B is `k_y` * `k_x` matrx. alpha is `k_y` * `1` cumulative probability vector, D(alpha) = diag(delta), xi(alpha) = (1 - 2 * alpha) / alpha / (1 - alpha), the diagonal elements of Sigma(alpha) are 2 / alpha / (1 - alpha).
@note The reparameterization of MAL distribution refers to Kotz et al.(2012).
@param Y a `k_y` * `n` matrix of explained variables, where `k_y` is the number of variables and `n` is the number of observations
@param X a `k_x` * `n` matrix of data, where `k_x` is the number of variables and `n` is the number of observations
@param alpha a `k_y` * `1` vector of tail probabilities
@param prior a `Rcpp::List` containing prior settings, including `mu_B`, `Sigma_B`, `Sigma`, `nu`, `n_delta`, `s_delta`, `n_lambda`. `mu_B`: a `k_y` * `k_x` `Eigen::MatrixXd`, is the prior mean of coefficient matrix B. `Sigma_b`: a `Rcpp::List` of `k_y` `k` * `k` `Eigen::MatrixXd`, each of them is the prior covariance matrix of one row of B. `Sigma`: a `k` * `k` `Eigen::MatrixXd` for the matrix parameter of the prior inverse Wishart distribtution of Sigma(alpha). `nu`: a `double`, which is the freedom parameter of prior IW distribution of Sigma(alpha). `n_delta`: a `double`, is the prior shape parameter of prior inverse gamma distribution of delta. `s_delta`: a `double`, is the prior scale parameter of the inverse gamma distribution of delta. `n_lambda`: a `double`, is the prior shape parameter of inverse gamma distribution of penalty. `s_lambda`: a `double`, is the prior scale parameter of inverse gamma distribution of penalty.
@param samplerSetting a `Rcpp::List` of sampler setting, including `init_B`, `init_Sigma`, `init_delta`, `samplerSize`, `burnIn`, `thin`. `init_B`: a `k_y` * `k_x` `Eigen::matrixXd`, is the initial value of coefficient matrix B. `init_Sigma`: a `k_y` * `k_y` `Eigen::MatrixXd`, is the initial value of Sigma(alpha). `init_delta`: a `double`, is the initial value of delta. `sampleSize`: a `int`, is the number of samples to draw. `burnIn`: a `int`, is the number of burn-in samples. `thin`: a `int`, is the thinning interval.
@param printFreq a `int`, frequency of printing out the iteration number. Message will be printed every `printFreq` iterations.
@return a list containing MCMC chains and posterior estimates.
*/
List mqrGibbsSampler_mal(
    const EigenMat &Y,
    const EigenMat &X,
    const RcppNumVec &alpha,
    const List &prior,
    const List &samplerSetting,
    const int &printFreq = 500,
    const bool &mute = false)
{
    List res; // output list

    const int k_y = Y.rows(); // number of explained variables
    const int k_x = X.rows(); // number of explanatory variables
    const int n = X.cols();   // number of observations

    /* #region Declare tail-specific quantities */
    EigenVec xi(k_y);
    EigenVec s2(k_y);
    for (int i = 0; i < k_y; ++i)
    {
        const double alpha_i = alpha[i];
        xi(i) = (1.0 - 2.0 * alpha_i) / alpha_i / (1.0 - alpha_i);
        s2(i) = 2.0 / alpha_i / (1.0 - alpha_i);
    }
    EigenMat S = s2.cwiseSqrt().asDiagonal();
    /* #endregion */

    /* #region Extract prior setting components from `prior` */
    EigenMat prior_muB = prior["mu_B"];    // prior mean of B, k_y * k_x
    List prior_SigmaB = prior["Sigma_B"];  // a list containing prior varaince of each row of B
    EigenMat prior_Sigma = prior["Sigma"]; // prior variance of Sigma, k_y * k_y
    int prior_nu = prior["nu"];            // prior degrees of freedom of Sigma
    EigenVec prior_nd = prior["n_delta"];  // prior shape parameter of delta
    EigenVec prior_sd = prior["s_delta"];  // prior scale parameter of delta
    EigenMat prior_nl;                     // prior shape parameter of penalty, k_y * k_x
    EigenMat prior_sl;                     // prior scale parameter of penalty, k_y * k_x
    if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
    {
        prior_nl = prior["n_lambda"];
        prior_sl = prior["s_lambda"];
    }
    /* #endregion */

    /* #region Extract sampler setting components from `samplerSetting` */
    const EigenMat init_B = samplerSetting["init_B"];
    const EigenMat init_Sigma = samplerSetting["init_Sigma"];
    const EigenVec init_delta = samplerSetting["init_delta"];
    const int n_sample = samplerSetting["n_sample"];
    const int n_burn = samplerSetting["n_burn"];
    const int n_thin = samplerSetting["n_thin"];
    const EigenVec init_b = Eigen::Map<const EigenVec>(init_B.data(), init_B.size());
    const EigenVec init_sigma = Eigen::Map<const EigenVec>(init_Sigma.data(), init_Sigma.size());
    const int n_iter = n_burn + n_sample * n_thin; // total number of iterations
    /* #endregion */

    /* #region Declare matrices to save mcmc chians and initialize the first row */
    RcppNumMat mcmc_w(n_iter, n);
    RcppNumMat mcmc_b(n_iter + 1, k_y * k_x);
    RcppNumMat mcmc_sigma(n_iter + 1, k_y * k_y);
    RcppNumMat mcmc_delta(n_iter + 1, k_y);
    RcppNumMat mcmc_lambda(n_iter + 1, k_y * k_x);
    mcmc_b.row(0) = Rcpp::as<RcppNumVec>(wrap(init_b));
    mcmc_sigma.row(0) = Rcpp::as<RcppNumVec>(wrap(init_sigma));
    mcmc_delta.row(0) = Rcpp::as<RcppNumVec>(wrap(init_delta));
    mcmc_lambda.row(0) = Rcpp::as<RcppNumVec>(wrap(EigenVec::Ones(k_y * k_x)));
    /* #endregion */

    /* #region Declare temporary variables that will be updated with loop */
    EigenVec w_h(n);
    EigenMat B_h = init_B;
    EigenVec b_h = init_b;
    EigenMat Sigma_h = init_Sigma;
    EigenVec sigma_h = init_sigma;
    EigenVec delta_h = init_delta;
    EigenMat Lambda_h = EigenMat::Ones(k_y, k_x);
    EigenVec lambda_h = Eigen::Map<EigenVec>(Lambda_h.data(), Lambda_h.size());
    const IntegerVector idx = Rcpp::seq(0, k_y - 1); // used for generation of random sampling w.r.t. index of endogenous variables
    /* #endregion */

    /* #region Loop: Gibbs sampling */
    std::chrono::system_clock::time_point start;
    if (mute == false)
    {
        start = std::chrono::system_clock::now();
        std::time_t start_time = std::chrono::system_clock::to_time_t(start);
        Rcpp::Rcout << "Gibbs sampling started at " << std::ctime(&start_time) << std::endl;
    }
    for (int h = 0; h < n_iter; ++h)
    {

        /* #region Update w */
        const EigenMat ew_h = Y - B_h * X;
        const EigenMat QwInv = (delta_h.asDiagonal() * Sigma_h * delta_h.asDiagonal()).inverse();
        double d_h = xi.dot(Sigma_h.inverse() * xi) + 2.0;
        double l_h = 1.0 - 0.5 * k_y;
        for (int t = 0; t < n; ++t)
        {
            const EigenVec ew_th = ew_h.col(t);
            const double m_th = ew_th.dot(QwInv * ew_th);
            const RcppNumVec w_th_rcpp = rGIG(
                Named("n") = 1,
                Named("lambda") = l_h,
                Named("chi") = m_th,
                Named("psi") = d_h);
            w_h(t) = w_th_rcpp[0];
        }
        mcmc_w.row(h) = Rcpp::as<RcppNumVec>(wrap(w_h));
        /* #endregion */

        /* #region Update each row of B, i.e. b_i, i = 1, ..., k_y */
        IntegerVector randomIdx = Rcpp::sample(idx, k_y, false);
        const EigenMat Qb_h = X * w_h.asDiagonal().inverse() * X.transpose();
        const EigenMat Omega_h = (delta_h.asDiagonal() * Sigma_h * delta_h.asDiagonal()).inverse();
        for (int i = 0; i < k_y; ++i)
        {
            const int j = randomIdx[i];
            const EigenMat eb_h = Y - B_h * X - delta_h.asDiagonal() * xi * w_h.transpose();
            const EigenVec Y_j = Y.row(j).transpose(); // T * 1 vector
            const double xi_j = xi(j);
            const double delta_jh = delta_h(j);
            const EigenMat prior_Sigmab_j = prior_SigmaB[j]; // (n_end + n_exo + 1) * (n_end + n_exo + 1) matrix
            const EigenMat Lambda_jh = Lambda_h.row(j).asDiagonal();
            const EigenVec prior_mub_j = prior_muB.row(j).transpose(); // (n_end + n_exo + 1) * 1 vector
            const EigenVec eb_jh = Y_j - delta_jh * xi_j * w_h;        // T * 1 vector
            const EigenVec omega_j = Omega_h.row(j).transpose();       // n_end * 1 vector
            const double omega_jj = Omega_h(j, j);
            EigenVec omega_other(k_y - 1);
            EigenMat eb_other(k_y - 1, n);
            if (j == 0)
            {
                omega_other = omega_j.segment(1, k_y - 1); // remove the first element
                eb_other = eb_h.bottomRows(k_y - 1);       // remove the first row
            }
            else if (j == k_y - 1)
            {
                omega_other = omega_j.segment(0, k_y - 1); // remove the last element
                eb_other = eb_h.topRows(k_y - 1);          // remove the last row
            }
            else
            {
                omega_other << omega_j.segment(0, j), omega_j.segment(j + 1, k_y - j - 1);
                eb_other.topRows(j) = eb_h.topRows(j);
                eb_other.bottomRows(k_y - j - 1) = eb_h.bottomRows(k_y - j - 1);
                /*
                Removing the j-th element of omega_j and the j-th row of eb_h respectively. When j != 0 and j != k_y - 1, the omega_other vector is constructed by concatenating the first j elements and the last k_y - j - 1 elements of omega_j. The eb_other matrix is constructed by concatenating the first j rows and the last k_y - j - 1 rows of eb_h.
                */
            }
            const EigenMat post_Sigmab_jh = (Qb_h * omega_jj + prior_Sigmab_j.inverse() * Lambda_jh.inverse()).inverse();
            const EigenVec post_mub_jh = post_Sigmab_jh * (X * w_h.cwiseInverse().asDiagonal() * (omega_jj * eb_jh + eb_other.transpose() * omega_other) + prior_Sigmab_j.inverse() * prior_mub_j);
            const EigenMat LowTriSigmab_jh = post_Sigmab_jh.llt().matrixL();
            const RcppNumVec x_b = Rcpp::rnorm(k_x);
            EigenVec b_jh = Rcpp::as<EigenVec>(x_b);
            b_jh = (LowTriSigmab_jh * b_jh + post_mub_jh).eval();
            B_h.row(j) = b_jh;
        }
        b_h = Eigen::Map<EigenVec>(B_h.data(), B_h.size());
        mcmc_b.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(b_h));
        /* #endregion */

        /* #region Update Sigma_h */
        EigenMat es_h = Y - B_h * X - delta_h.asDiagonal() * xi * w_h.transpose();
        es_h = (delta_h.asDiagonal().inverse() * es_h * w_h.cwiseSqrt().cwiseInverse().asDiagonal()).eval();
        EigenMat post_Sigma_h = es_h * es_h.transpose() + prior_nu * prior_Sigma;
        Eigen::EigenSolver<Eigen::MatrixXd> solver(post_Sigma_h);
        const int post_nu_h = n + prior_nu;
        const Rcpp::NumericMatrix Sigma_h_rcpp = rInvWishart(
            Named("nu") = post_nu_h,
            Named("S") = post_Sigma_h);
        Sigma_h = Rcpp::as<EigenMat>(Sigma_h_rcpp);
        EigenMat diagSqrtSigmaInv_h = Sigma_h.diagonal().cwiseSqrt().cwiseInverse().asDiagonal();
        Sigma_h = (S * diagSqrtSigmaInv_h * Sigma_h * diagSqrtSigmaInv_h * S).eval();
        sigma_h = Eigen::Map<EigenVec>(Sigma_h.data(), Sigma_h.size());
        mcmc_sigma.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(sigma_h));
        /* #endregion */

        /* #region Update delta_h */
        randomIdx = Rcpp::sample(idx, k_y, false);
        for (int i = 0; i < k_y; ++i)
        {
            const int j = randomIdx[i];
            const EigenMat ed_h = Y - B_h * X - delta_h.asDiagonal() * xi * w_h.transpose();
            const EigenVec ed_jh = ed_h.row(j).transpose();
            const double delta_jh = delta_h(j);
            const double s2_j = s2(j);
            const EigenVec vsqrt_jh = (delta_jh * w_h).cwiseSqrt();
            const double Qd = (ed_jh.cwiseProduct(vsqrt_jh.cwiseInverse())).squaredNorm() / s2_j;
            const double post_nd = 0.5 * (prior_nd(j) + 3.0 * n);
            const double post_sd = 0.5 * prior_sd(j) + 2.0 * vsqrt_jh.squaredNorm() + Qd;
            delta_h(j) = rInvGamma(post_nd, post_sd);
        }
        mcmc_delta.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(delta_h));
        /* #endregion */

        /* #region If penalty is specified, update penalty parameters */
        if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
        {
            for (int i = 0; i < k_y; ++i)
            {
                const EigenMat prior_SigmaB_i = prior_SigmaB[i];
                for (int j = 0; j < k_x; ++j)
                {
                    const double b_ij = B_h(i, j);
                    const double prior_Sigmab_ij = prior_SigmaB_i(j, j);
                    const double post_nl = 0.5 * (prior_nl(i, j) + 1);
                    const double post_sl = 0.5 * (std::pow(b_ij, 2) / prior_Sigmab_ij + prior_sl(i, j));
                    Lambda_h(i, j) = rInvGamma(post_nl, post_sl);
                }
            }
            lambda_h = Eigen::Map<EigenVec>(Lambda_h.data(), Lambda_h.size());
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
                    Rcpp::Rcout << "Iterations " << h + 1 << ": " << std::round(static_cast<float>(elapsed_seconds.count() * 100)) / 100 << "s" << std::endl;
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

    /* #region Save MCMC chains into a `Rcpp::List` */
    List res_mcmc = List::create(
        Named("w") = mcmc_w,
        Named("b") = mcmc_b,
        Named("sigma") = mcmc_sigma,
        Named("delta") = mcmc_delta);
    if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
    {
        res_mcmc["lambda"] = mcmc_lambda;
    }
    /* #endregion */

    /* #region Calculate posterior median and save them into `Rcpp::List` */
    RcppIntVec idx_thin = sequence(n_burn, n_iter - 1, n_thin);
    RcppNumVec b_est = colMedianOfMat(matSubset(mcmc_b, idx_thin, R_NilValue));
    RcppNumMat B_est(k_y, k_x, b_est.begin());
    RcppNumVec sigma_est = colMedianOfMat(matSubset(mcmc_sigma, idx_thin, R_NilValue));
    RcppNumMat Sigma_est(k_y, k_y, sigma_est.begin());
    RcppNumVec w_est = colMedianOfMat(matSubset(mcmc_w, idx_thin, R_NilValue));
    RcppNumVec delta_est = colMedianOfMat(matSubset(mcmc_delta, idx_thin, R_NilValue));
    List res_est = List::create(
        Named("B") = B_est,
        Named("Sigma") = Sigma_est,
        Named("w") = w_est,
        Named("delta") = delta_est);
    /* #endregion */

    res["mcmcChains"] = res_mcmc;
    res["estimates"] = res_est;

    return res;
}
