#' @name estBQVAR
#' @title Estimate Bayesian QVAR model
#' @description Estimate Bayesian QVAR model using Gibbs algorithm. Two different versions of alogorithm are available: one uses Asymmetric Laplace (AL) distribution as likelihood, and the other one uses Multivariate Asymmetric Laplace (MAL) distribution. The Gibbs sampling algorithm is implemented in C++ for efficiency.
#' @usage
#' estBQVAR(
#'     data_end,
#'     lag,
#'     alpha,
#'     data_exo = NULL,
#'     prior = NULL,
#'     samplerSetting = NULL,
#'     method = "bayes-al",
#'     printFreq = 10,
#'     mute = FALSE
#' )
#' @param data_end \eqn{(T + \max(p, q)) \times n_Y} data frame of endogenous variables.
#' Time increases from top to bottom.
#' @param lag Integer scalar or vector that indicates lag order, \eqn{(p, q)}, of
#' the endogenous and exogenous variables. If a scalar is provided, it is assumed
#' that the lag order of the endogenous and exogenous variables(if included into
#' the model) are the same. If a vector of length 2 is provided, the first element
#' is the lag order of the endogenous variables and the second element is the lag
#' order of the exogenous variables.
#' @param alpha Numeric scalar or \eqn{n_Y\times 1} vector that indicates probability
#'  at which the QVAR is estimated. If a scalar is provided, it is assumed that all
#'  endogenous variables have the same alpha value. If a vector of length \eqn{n_Y}
#'  is provided, each endogenous variable has its own alpha value.
#' @param data_exo (optional, default is \code{NULL}) \eqn{(T + \max(p, q)) \times n_X}
#'  data frame of exogenous variables. If provided, the QVAR model will include the 1-st
#'  to q-th order of lagged term of \eqn{\bm{X}}.
#' @param prior (optional) List of prior distribution parameters. A complete prior setting list includes following components: \code{mu_A}, \code{Sigma_A}, \code{Sigma}, \code{nu}, \code{n_delta}, \code{s_delta}, \code{n_lambda}, \code{s_lambda}. If not provided, the default setting is used.
#' \itemize{
#' \item \code{mu_A} \eqn{n_Y \times (n_Y p + n_X q + 1)} matrix of prior mean of coefficient matrix \eqn{\mathbf{A}(\bm{\alpha})_+}, of which each row is \eqn{\underline{\bm{a}}_i}. The default is \eqn{\mathbf{0}}.
#' \item \code{Sigma_A} List of prior covariance matrix \eqn{\underline{\bm{\Sigma}}_{\bm{a}_i}}, the default is a list of \eqn{n_Y}  \eqn{100 \times \mathbf{I}_{n_Y p + n_X q + 1}}.
#' \item \code{Sigma} \eqn{n_Y \times n_Y} matrix parameter \eqn{\underline{\bm{\Sigma}}} of Inverse Wishart distribution, which will only be used when \code{method = "bayes-mal"}. The default is \eqn{\mathbf{S}^2=\text{diag}(\tilde{s}_1^2, \tilde{s}_2^2, \cdots, \tilde{s}_{n_Y}^2)}.
#' \item \code{nu} Integer that indicate parameter \eqn{\underline{\nu}} of Inverse Wishart distribution, which will only be used when \code{method = "bayes-mal"}. The default is \eqn{n_Y + 1}.
#' \item \code{n_delta} \eqn{n_Y\times 1} vector of shape parameter \eqn{\underline{n}_{\delta,i}} of Inverse Gamma distribution for \eqn{\delta_i}. The default is \eqn{\mathbf{1}_{n_Y}}.
#' \item \code{s_delta} \eqn{n_Y \times 1} vector of scale parameter \eqn{\underline{s}_{\delta,i}} of Inverse Gamma distribution for \eqn{\delta_i}. The default is \eqn{\mathbf{1}_{n_Y}}.
#' \item \code{n_lambda} \eqn{n_Y \times (n_Y p + n_X q + 1)} matrix of shape parameter \eqn{\underline{n}_{\lambda,ij}} of Inverse Gamma distribution for \eqn{\lambda_{ij}}, The larger, the stronger penalty on \eqn{a(\bm{\alpha})_{ij}} is.
#' \item \code{s_lambda} \eqn{n_Y \times (n_Y p + n_X q + 1)} matrix of scale parameter \eqn{\underline{s}_{\lambda,ij}} of Inverse Gamma distribution for \eqn{\lambda_{ij}}. The smaller, the stronger penalty on \eqn{a(\bm{\alpha})_{ij}} is.
#' }
#' If \code{n_lambda} and \code{s_lambda} are not provided, then no penalty imposed on \eqn{\mathbf{A}(\bm{\alpha})_+}.
#' @param samplerSetting (optional) List of sampler setting parameters that passing the initial values of Gibbs algorithm and the number of iterations. A complete sampler including \code{init_A}, \code{init_Sigma}, \code{init_delta}, \code{n_sample}, \code{n_burn}, \code{n_thin}. If not privided, the default setting is used.
#' \itemize{
#' \item \code{init_A} Initial value of coefficient matrix \eqn{\mathbf{A}(\bm{\alpha})_+}. The default is \eqn{\mathbf{0}}.
#' \item \code{init_Sigma} Initial value of variance-covariance matrix \eqn{\mathbf{\Sigma}}. The default is \eqn{\mathbf{S}^2}.
#' \item \code{init_delta} Initial value of shape parameter \eqn{\delta_i}. The default is \eqn{0.1 \times \bm{1}_{n_Y}}.
#' \item \code{n_sample} Number of total valid sampling. The default is \code{500}.
#' \item \code{n_burn} Number of burn-in period. The default is \code{500}.
#' \item \code{n_thin} Thinning interval. The default is \code{1}.
#' }
#' Total iteration number is \code{n_burn + n_sample * n_thin}.
#' @param method (optional, default is "bayes-al") Estimation method: "bayes-al" for AL-based algorithm, "bayes-mal" for MAL-based algorithm.
#' @param printFreq (optional, default is \code{10}) Frequency of printing the progress of the estimation.
#' @param mute (optional, default is \code{FALSE}) Whether to suppress the progress printing. If \code{method = "bayes-al"}, then the progress is printed every \code{printFreq} equations. If \code{method = "bayes-mal"}, then the progress is printed every \code{printFreq} samples.
#' @return An object of class \code{BQVAR}, which contains model specification information, MCMC chains and Bayesian estimates. The estimates of \eqn{\mathbf{A}(\bm{\alpha})_+} is sample mean of the last \code{n_sample} samples.
#' @details
#' \subsection{Model specification}{
#' \emph{Qauntile Vector AutoRgression} is used to estimate quantiles of \eqn{n_Y} endogenous variables and is especially useful for capturing the heterogenous affect of past shock to future distribution. An \eqn{\mathrm{QVAR}(\bm{\alpha})} takes the form
#'     \deqn{
#'         \bm{Y}_t = \bm{q}(\bm{\alpha}) + \sum\limits_{j=1}^p \mathbf{A}(\bm{\alpha})_j \bm{Y}_{t-j} + \sum\limits_{k=1}^q \mathbf{\Gamma}(\bm{\alpha})_k \bm{X}_{t-k} + \bm{u}(\bm{\alpha})_t, t = 1, 2 \cdots, T,
#'     }
#'     where \eqn{\bm{Y}_t} is the \eqn{n_Y}-dimensional vector of endogenous variables at time \eqn{t}, and \eqn{\bm{X}_t} is the \eqn{n_X}-dimensional vector of exogenous variables at time \eqn{t}. The lag order of the endogenous and exogenous variables are \eqn{p} and \eqn{q} respectively. \eqn{\bm{q}(\bm{\alpha})} is \eqn{n_Y \times 1} intercept term. The matrix \eqn{\mathbf{A}(\bm{\alpha})_j} is the \eqn{n_Y \times n_Y} matrix of coefficients for the endogenous variables at lag \eqn{j}, and \eqn{\mathbf{\Gamma}(\bm{\alpha})_k} is the \eqn{n_Y \times n_X} matrix of coefficients for the exogenous variables at lag \eqn{k}. \eqn{\bm{u}(\bm{\alpha})_t} is the \eqn{n_Y}-dimensional vector of error terms at time \eqn{t}. \eqn{\bm{\alpha} \equiv [\alpha_1, \alpha_2, \cdots, \alpha_{n_Y}]^\intercal \in (0,1)^{n_y}} is the probability vector that indicates the quantile of interest. All the notation \eqn{\bm{\alpha}} in parenthesis indicate that the parameter or variables varies with quantiles. \cr
#' Different to conventional VAR, of which the error term has zero mean, the error term \eqn{\bm{u}(\alpha)} is required to have zero \eqn{\bm{\alpha}}-quantile. For example, if \eqn{\alpha_i}-quantile of \eqn{Y_i} is estimated, then \eqn{u(\alpha_i)_t} satisfies \eqn{\mathrm{Pr}( u(\alpha_i)_t < 0 | I_t ) = \alpha_i } or \eqn{Q_{\alpha_i}(u(\alpha_i)_t|I_t) = 0}. The quantile of \eqn{\bm{Y}} is \eqn{Q_{\bm{\alpha}}(\bm{Y}_t | I_t) = \bm{q}(\bm{\alpha}) + \sum_{j=1}^p \mathbf{A}(\bm{\alpha})_j \bm{Y}_{t-j} + \sum_{k=1}^q \mathbf{\Gamma}(\bm{\alpha})_k \bm{X}_{t-k}}
#' }
#' \subsection{Likelihood}{
#' The error term \eqn{\bm{u}(\bm{\alpha})_t} is assumed that either each component \eqn{u(\alpha_i)} follows Asymmetric Laplace (AL) distribution or all components follow Multivariate Asymmetric Laplace (MAL) distribution jointly, as follows:
#' \deqn{
#'  \text{AL distribution: } u(\alpha)_{it} \sim \mathcal{AL}\left(0, \delta_i \tilde{\xi}_i, \tilde{\sigma}_i\right) i = 1, 2, \cdots, n_Y, \\
#'  \text{MAL distribution: } \bm{u}(\bm{\alpha})_t \sim \mathcal{MAL}\left(\bm{0}, \mathbf{D}\tilde{\bm{\xi}}, \mathbf{D}\tilde{\mathbf{\Sigma}}\mathbf{D} \right),
#' }
#' where \eqn{\tilde{\xi}_i = \frac{1 - 2 \alpha_i}{\alpha_i (1 - \alpha_i)}, \tilde{\sigma}_i^2 = \frac{2}{\alpha_i (1 - \alpha_i)}} and
#' \deqn{
#' \tilde{\bm{\xi}} = \begin{bmatrix} \tilde{\xi}_1 \\ \tilde{\xi}_2 \\ \vdots \\ \tilde{\xi}_{n_Y} \end{bmatrix},
#' \text{diag}\left(\tilde{\mathbf{\Sigma}}\right) = \begin{bmatrix} \tilde{\sigma}_1^2 \\ \tilde{\sigma}_2^2 \\ \vdots \\ \tilde{\sigma}_{n_Y}^2 \end{bmatrix},
#' \mathbf{D} = \begin{bmatrix} \delta_1 & 0 & \cdots & 0 \\ 0 & \delta_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \delta_{n_Y} \end{bmatrix}.
#' }
#' The parameterization of AL and MAL distribution follows \insertCite{kotz_laplace_2012;textual}{bayesQVAR}. The superscript \eqn{\tilde{\cdot}} means that corresponsding parameters are associated with \eqn{\bm{\alpha}}.
#' }
#' \subsection{Mixtrue representation}{
#' Bayesian estimation of QVAR model utilizes the mixture representation property of AL and MAL distribution, that is
#' \deqn{
#' \text{AL likelihood: } u(\bm{\alpha})_{it} = \delta_i \tilde{\xi} W_{it} + \sqrt{W_it} \delta_i \tilde{\sigma}_i Z_{it}, \\
#' \text{MAL likelihood: } \bm{u}(\bm{\alpha})_t = \mathbf{D}\tilde{\bm{\xi}} W_t +  \sqrt{W_t} \mathbf{D} \tilde{\mathbf{\Sigma}}^{1/2} \bm{Z}_t,
#' }
#' where \eqn{W_{it}, W_t \sim \mathcal{E}\mathcal{x}p(1)} and \eqn{Z_{it} \sim \mathcal{N}(0, 1)} are independent random variables, \eqn{\bm{Z}_t \equiv [Z_{1t}\ Z_{2t}\ \cdots\ Z_{n_Y,t}]^{\intercal}}.
#' }
#' \subsection{Prior}{
#' Prior distribution used in this function is based on existing works to large extent \insertCite{kozumi_gibbs_2011,schuler_asymmetric_2014}{bayesQVAR}. of the lies in three points: (1) a shrinkage prior \insertCite{van_erp_shrinkage_2019}{bayesQVAR} for coefficient matrics is introduced; (2) In MAL-based algorithm, we continue to use "equation-by-equation" strategy, i.e. specify prior and derives posterior for each row for coefficient matrices separately, other than operate on the whole matrix, (3) Avoiding the use of Metropolis-Hastings algorithm, which is computationally expensive. \cr
#' Following \insertCite{kozumi_gibbs_2011;textual}{bayesQVAR}, the prior specification based on AL likelihood is
#' \deqn{
#' W_{it} \sim \mathcal{E}\mathcal{x}p(1), i=1,2,\cdots, n_Y, t=1, 2,\cdots, T, \\
#' \bm{a}(\bm{\alpha})_i \sim \mathcal{N}\left(\underline{\bm{a}}_i, \bm{\Lambda}_i\underline{\bm{\Sigma}}_{\bm{a}_i} \right), i=1,2,\cdots, n_Y, \\
#' \delta_i \sim \mathcal{I}\mathcal{G}\left(\frac{\underline{n}_{\delta,i}}{2}, \frac{\underline{s}_{\delta,i}}{2} \right), i=1,2,\cdots, n_Y, \\
#' \lambda_{ij} \sim \mathcal{I}\mathcal{G}\left(\frac{\underline{n}_{\lambda,ij}}{2}, \frac{\underline{s}_{\lambda, ij}}{2}\right), i=1,2,\cdots, n_Y, j=1,2,\cdots, n_Y p + n_X q + 1,
#' }
#' where \eqn{\mathbf{A}(\bm{\alpha})_+ \equiv \left[ \mathbf{A}(\bm{\alpha}) \cdots \mathbf{A}(\bm{\alpha})_p\ \mathbf{\Gamma}(\bm{\alpha})_1 \cdots \mathbf{\Gamma}(\bm{\alpha})_q\ \bm{q}(\bm{\alpha})\right]}, and \eqn{\bm{a}(\bm{\alpha})_i} is the \eqn{i}-th row of \eqn{\mathbf{A}(\bm{\alpha})_+}, \eqn{\bm{\Sigma}_{\bm{\beta}_i}} is a diagonal variance matrix, \eqn{\lambda_{ij}} is penalty parameter imposed on \eqn{\bm{a}(\bm{\alpha})_{ij}}, \eqn{\mathbf{\Lambda}_i \equiv \mathrm{diag}(\lambda_{i1}, \lambda_{i2}, \cdots, \lambda_{i,n_Y p + n_X q + 1})}. \eqn{\mathcal{IG}(\alpha, \beta)} represents Inverse Gamma distribution with shape parameter \eqn{\alpha} and scale parameter \eqn{\beta}. \cr
#' The correspondence based on MAL likelihood, proposed by \insertCite{schuler_asymmetric_2014;textual}{bayesQVAR}, is
#' \deqn{
#' W_t \sim \mathcal{E}\mathcal{x}p(1), t= 1, 2,\cdots, T, \\
#' \bm{a}(\bm{\alpha})_i \sim \mathcal{N}\left(\underline{\bm{a}}_i, \bm{\Lambda}_i\underline{\bm{\Sigma}}_{\bm{a}_i} \right), i = 1,2, \cdots, n_Y \\
#' \tilde{\bm{\Sigma}} \sim \mathcal{IW}(\underline{\nu}, \underline{\mathbf{\Sigma}}), \\
#' \delta_i \sim \mathcal{I}\mathcal{G}\left(\frac{\underline{n}_{\delta,i}}{2}, \frac{\underline{s}_{\delta,i}}{2} \right), i=1,2,\cdots, n_Y, \\
#' \lambda_{ij} \sim \mathcal{I}\mathcal{G}\left(\frac{\underline{n}_{\lambda,ij}}{2}, \frac{\underline{s}_{\lambda, ij}}{2}\right), i=1,2,\cdots, n_Y, j=1,2,\cdots, n_Y p + n_X q + 1,
#' }
#' where \eqn{\mathcal{IW}(\nu, \mathbf{\Sigma})} represents Inverse Wishart distribution with degree of freedom \eqn{\nu} and scale matrix \eqn{\mathbf{\Sigma}}.
#' }
#' @example man/examples/estBQVAR.R
#' @references
#' \insertAllCited{}
estBQVAR <- function(
    data_end,
    lag,
    alpha,
    data_exo = NULL,
    prior = NULL,
    samplerSetting = NULL,
    method = "bayes-al",
    printFreq = 10,
    mute = FALSE
){
    # if alpha is a scalar or vector, then perform estimation once
    if(is.numeric(alpha)){
        res_estBQVAR <- .estBQVAR(
            data_end,
            lag,
            alpha,
            data_exo,
            prior,
            samplerSetting,
            method,
            printFreq,
            mute
        )
    }
    # if alpha is a list, then perform estimation for each element of alpha
    if(class(alpha) == "list"){

        n_cores <- min(length(alpha), round(parallel::detectCores() * 2 / 3))
        estQVARforEachAlpha <- function(i, data_end, lag, alpha, data_exo, prior, samplerSetting, method, printFreq, mute){
            res <- .estBQVAR(
                data_end,
                lag,
                alpha[[i]],
                data_exo,
                prior,
                samplerSetting,
                method,
                printFreq,
                mute
            )
            return(res)
        }
        cl <- parallel::makeCluster(n_cores)
        res_estBQVAR_parallel <- parallel::parLapply(
            cl = cl,
            X = 1:length(alpha),
            fun = estQVARforEachAlpha,
            data_end = data_end,
            lag = lag,
            alpha = alpha,
            data_exo = data_exo,
            prior = prior,
            samplerSetting = samplerSetting,
            method = method,
            printFreq = printFreq,
            mute = mute
        )
        on.exit(parallel::stopCluster(cl))
        res_estBQVAR <- res_estBQVAR_parallel[[1]][c("data", "designMat", "lag", "method", "prior", "samplerSetting")]
        mcmcChainList <- lapply(res_estBQVAR_parallel, function(x) x$mcmcChains )
        estimatesList <- lapply(res_estBQVAR_parallel, function(x) x$estimates )
        residualsList <- lapply(res_estBQVAR_parallel, function(x) x$residuals )
        res_estBQVAR$mcmcChains <- mcmcChainList
        res_estBQVAR$estimates <- estimatesList
        res_estBQVAR$residuals <- residualsList
        res_estBQVAR$alpha <- alpha
    }

    res <- BQVAR(res_estBQVAR)
    return(res)
}




