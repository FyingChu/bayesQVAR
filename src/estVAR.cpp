// [[Rcpp::depends(RcppEigen)]]
#include "estBQVAR.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include "bayesQVAR_types.h"
#include "basicSamplers.h"
#include "constDesignMat.h" // constructing design matrices
#include "manipMatDfList.h"

using namespace Rcpp;

// [[Rcpp::export(.estVARbyOLS)]]
EigenMat estVARbyOLS(
    const EigenMat &Y,
    const EigenMat &X)
{
    int n_y = Y.rows();
    int n_x = X.rows();
    EigenMat A(n_y, n_x);
    for (int i = 0; i < n_y; ++i)
    {
        EigenVec y_i = Y.row(i).transpose();
        EigenVec a_i = (X * X.transpose()).inverse() * X * y_i;
        A.row(i) = a_i.transpose();
    }

    return A;
}

// [[Rcpp::export(.estBVAR)]]
List estBVAR(
    const RcppDf &data_end,
    const RcppIntVec &lag,
    const Rcpp::Nullable<RcppDf> &data_exo = R_NilValue,
    const List &prior = NULL,
    const List &samplerSetting = NULL)
{
    List res;

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

    /* #region Construct design matrices */
    List designMatList = constDesignMat(data_end, data_exo, lag_end, lag_exo);
    EigenMat Y = designMatList["Y"];
    EigenMat X = designMatList["X"];
    /* #endregion */

    /* #region Fetch prior setting from `prior` */
    EigenMat prior_muA = prior["mu_A"];    // prior mean of A, k_y * k_x
    List prior_SigmaA = prior["Sigma_A"];  // list of prior covariance of A, contain k_y k_x * k_x matrices
    EigenMat prior_Sigma = prior["Sigma"]; // prior covariance of error term, k_y * k_y
    int prior_nu = prior["nu"];
    EigenMat prior_nl; // prior shape parameter of penalty, k_y * k_x
    EigenMat prior_sl; // prior scale parameter of penalty, k_y * k_x
    if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
    {
        prior_nl = prior["n_lambda"];
        prior_sl = prior["s_lambda"];
    }
    /* #endregion */

    /* #region Fetch sampler setting from `samplerSetting` */
    const EigenMat init_Sigma = samplerSetting["init_Sigma"];
    const int n_sample = samplerSetting["n_sample"];
    const int n_burn = samplerSetting["n_burn"];
    const int n_thin = samplerSetting["n_thin"];
    const EigenVec init_sigma = Eigen::Map<const EigenVec>(init_Sigma.data(), init_Sigma.size());
    const int n_iter = n_burn + n_sample * n_thin; // total number of iterations
    /* #endregion */

    /* #region Declare MCMC chain matrices */
    RcppNumMat mcmc_a(n_iter + 1, n_end * n_x);
    RcppNumMat mcmc_sigma(n_iter + 1, n_end * n_end);
    RcppNumMat mcmc_lambda(n_iter + 1, n_end * n_x);
    mcmc_a.row(0) = Rcpp::as<RcppNumVec>(wrap(EigenVec::Zero(n_end * n_x)));
    mcmc_sigma.row(0) = Rcpp::as<RcppNumVec>(wrap(init_sigma));
    mcmc_lambda.row(0) = Rcpp::as<RcppNumVec>(wrap(EigenVec::Ones(n_end * n_x)));
    /* #endregion */

    /* #region Declare 'xx_h', used to store the sample of variable 'xx' in the h-th iteration */
    EigenVec a_h;
    EigenMat A_h(n_end, n_x);
    EigenVec sigma_h = init_sigma;
    EigenMat Sigma_h = init_Sigma;
    EigenMat Lambda_h = EigenMat::Ones(n_end, n_x);
    EigenVec lambda_h = Eigen::Map<EigenVec>(Lambda_h.data(), Lambda_h.size());
    /* #endregion */

    // start loop
    for (int h = 0; h < n_iter; ++h)
    {

        /* #region Update muA_i */
        Eigen::MatrixXd Q = X * X.transpose();
        for (int i = 0; i < n_end; ++i)
        {

            // calculate posterior mean and variance of a_hi
            double sigma2_i = Sigma_h(i, i);
            const EigenMat prior_SigmaA_i = prior_SigmaA[i];
            const EigenVec prior_muA_i = prior_muA.row(i);
            const EigenMat Lambda_i = Lambda_h.row(i).asDiagonal();
            const EigenVec y_i = Y.row(i).transpose();
            const EigenMat post_SigmaA_i = (Q / sigma2_i + prior_SigmaA_i.inverse() * Lambda_i.inverse()).inverse();
            const EigenMat Q_i = X * y_i / sigma2_i;
            const EigenVec post_muA_i = post_SigmaA_i * (Q_i + prior_SigmaA_i.inverse() * prior_muA_i);

            // h-th sampling of a_hi
            Rcpp::NumericVector x_a(n_x); // save a_hi
            x_a = Rcpp::rnorm(n_x);       // draw dim_a standard univariate normal samples
            Eigen::Map<VectorXd> a_ih(Rcpp::as<Eigen::Map<VectorXd>>(x_a));
            Eigen::MatrixXd L_a = post_SigmaA_i.llt().matrixL();
            a_ih = (L_a * a_ih + post_muA_i).eval();
            A_h.row(i) = a_ih;
        }
        a_h = Eigen::Map<VectorXd>(A_h.data(), A_h.size());
        mcmc_a.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(a_h));
        /* #endregion */

        // update the covariance of error term
        EigenMat es_h = Y - A_h * X;
        EigenMat post_Sigma = es_h * es_h.transpose() + prior_nu * prior_Sigma;
        double post_nu = prior_nu + T;
        Rcpp::NumericMatrix Sigma_h_new = rInvWishart(
            Named("S") = post_Sigma,
            Named("nu") = post_nu);
        Sigma_h = Rcpp::as<EigenMat>(Sigma_h_new);
        sigma_h = Eigen::Map<VectorXd>(Sigma_h.data(), n_end * n_end);
        mcmc_sigma.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(sigma_h));

        // update the penalty parameter
        if (prior.containsElementNamed("n_lambda") && prior.containsElementNamed("s_lambda"))
        {
            for (int i = 0; i < n_end; ++i)
            {
                const EigenMat prior_SigmaA_i = prior_SigmaA[i];
                for (int j = 0; j < n_x; ++j)
                {
                    const double a_ij = A_h(i, j);
                    const double prior_SigmaA_ij = prior_SigmaA_i(j, j);
                    const double post_nl = 0.5 * (prior_nl(i, j) + 1);
                    const double post_sl = 0.5 * (pow(a_ij, 2) / prior_SigmaA_ij + prior_sl(i, j));
                    Lambda_h(i, j) = rInvGamma(post_nl, post_sl);
                }
            }
            lambda_h = Eigen::Map<EigenVec>(Lambda_h.data(), Lambda_h.size());
            mcmc_lambda.row(h + 1) = Rcpp::as<RcppNumVec>(wrap(lambda_h));
        }
    }

    /* #region Save MCMC chains into a `Rcpp::List` */
    List res_mcmc = List::create(
        Named("a") = mcmc_a,
        Named("sigma") = mcmc_sigma,
        Named("lambda") = mcmc_lambda);
    /* #endregion */

    /* #region Calculate posterior median and save them into `Rcpp::List` */
    RcppIntVec idx_thin = sequence(n_burn, n_iter - 1, n_thin);
    RcppNumVec a_est = colMedianOfMat(matSubset(mcmc_a, idx_thin, R_NilValue));
    RcppNumMat A_est(n_end, n_x, a_est.begin());
    RcppNumVec sigma_est = colMedianOfMat(matSubset(mcmc_sigma, idx_thin, R_NilValue));
    RcppNumMat Sigma_est(n_end, n_end, sigma_est.begin());
    List res_est = List::create(
        Named("A") = A_est,
        Named("Sigma") = Sigma_est);
    /* #endregion */

    res["mcmcChains"] = res_mcmc;
    res["estimates"] = res_est;
    res["designMat"] = designMatList;

    return res;
}

// [[Rcpp::export(.estSigmaOfBVAR)]]
EigenVec estSigmaOfBVAR(
    const RcppDf &data_end,
    const RcppIntVec &lag,
    const Rcpp::Nullable<RcppDf> &data_exo = R_NilValue,
    const List &prior = NULL,
    const List &samplerSetting = NULL)
{
    /* #region Estimate covariance matrix of residuals of VAR */
    const List res_estBVAR = estBVAR(data_end, lag, data_exo, prior, samplerSetting);
    const List estimates_BVAR = res_estBVAR["estimates"];
    const List designMatList = res_estBVAR["designMat"];
    const EigenMat Y = designMatList["Y"];
    const EigenMat X = designMatList["X"];
    const EigenMat A_est = estimates_BVAR["A"];
    const EigenMat residuals = Y - A_est * X;
    const EigenVec rowMean_resid = residuals.rowwise().mean();
    const EigenMat residuals_centered = residuals - rowMean_resid.replicate(1, Y.cols());
    const EigenMat Sigma_est = residuals_centered * residuals_centered.transpose() / (Y.cols() - X.rows());
    const EigenVec sigmaSqrt = Sigma_est.diagonal().cwiseSqrt();
    /* #endregion */

    return sigmaSqrt;
}
