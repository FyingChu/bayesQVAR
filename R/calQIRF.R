#' @name calQIRF
#' @title Calculate Quantile Impulse Response Function
#' @description Calculate Quantile Impulse Response Function(QIRF) and its' bootstrap
#' confidence interval, given the probability paths along which the endogenous
#' variables in the QVAR system will evolve within horizon \eqn{H}.This function
#' also support counterfactual analysis.
#' @param modelSpecif List of model specification, which includes the following
#' components:
#' \itemize{
#'  \item{\code{data_end}} {Data frame of endogenous variables.}
#'  \item{\code{data_exo}} {Data frame of exogenous variables.}
#'  \item{\code{lag}} {Lag order of endogenous and exogenous variables.}
#'  \item{\code{prior}} {List of prior specification.}
#'  \item{\code{samplerSetting}} {List of initial values and of number of valid
#'  sampling, burn-in period and thinning.}
#'  \item{\code{method}} {String of estimation method.}
#'  \item{\code{B0}} {Instantaneous structural matrix.}
#' }
#' Refer to \code{\link{estBQVAR}} function for more details on all components
#' except for \code{B0}. About how \code{B0} works, see details.
#' @param names_impulse Vector of names of impulse variables.
#' @param names_response Vector of names of response variables.
#' @param horizon Integer that indicates forecast horizon \eqn{H}.
#' @param probPath \eqn{n_Y \times H} matrix of probability paths. Each column
#' corresponds to the quantile of interest at horizon \eqn{h}.
#' @param mean Logical value indicating whether to calculate the mean of QIRF,
#' see details.
#' @param counterfactual List that contains components for counterfactual analysis,
#' which includes \code{names}, \code{rules_inst} and \code{rules_inter}. See details.
#' @param confInt Logical value indicating whether to calculate the confidence
#' interval of QIRF. See details on how to the confidence interval are calculated.
#' @param credInt Logical value indicating whether to calculate the credibility
#' interval. See details on how to the credibility interval are calculated.
#' @param alpha Vector of significance level.
#' @param n_simu Integer that indicates the number of bootstrapping to calculate
#'  bootstrapping standard deviation of QIRF or/and how many samples of QIRF are drawn
#'  to calculate the posterior percentile of QIRF.
#' @param penalty_boots Penalty setting for estimation of QVAR based on bootstrapping
#' simulated data. If \code{NULL}, the original penalty setting in \code{modelSpecif}
#' will be used.
#' @return An object of class \code{BQVAR_QIRF}, which contains model specification
#' information, mean QIRF (\code{mean = TRUE}), counterfactual QIRF
#' (\code{!is.null(counterfactual)}), counterfactual mean QIRF (\code{mean = TRUE}
#' and \code{!is.null(counterfactual)}) and its' bootstrap confidence interval
#' (\code{confInt = TRUE}).
#' @details
#' \subsection{Quantile impulse response function for specific probability path}{
#' For following \eqn{n_Y}-dimensional QVAR process
#' \deqn{
#' Q_{\bm{\alpha}}(\bm{Y}_t|I_t) = \bm{q}(\bm{\alpha}) + \sum\limits_{j=1}^p
#' \mathbf{A}(\bm{\alpha}) \bm{Y}_{t-j} + \sum\limits_{k=1}^q \bm{X}_{t-k},
#' }
#' provided that the system evolves along the probability path
#' \eqn{\bm{\alpha}_1, \cdots, \bm{\alpha}_{h-1}} between \eqn{t+1} and \eqn{t+h-1}
#' exactly, then the response of conditional \eqn{\bm{\alpha}}-quantile at \eqn{t+h}
#' to the shock \eqn{\bm{\delta}} at time \eqn{t}, denoted as
#' \eqn{\text{QIRF}(\bm{\alpha}_{h}|\bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_{h-1})},
#' is defined as
#' \deqn{
#' Q_{\bm{\alpha}}(\bm{Y}_{t+h}|\bm{y}_{t+1}, \cdots, \bm{y}_{t+h-1}, I_t + \bm{\delta})
#'  - Q_{\bm{\alpha}}(\bm{Y}_{t+h}|\bm{y}_{t+1}, \cdots, \bm{y}_{t+h-1}, I_t)
#'  = \mathbf{J} \left[\prod\limits_{g=1}^h \tilde{\mathbf{A}}(\bm{\alpha}_{g}) \right]
#'  \mathbf{J}^{\intercal},
#' }
#' where \eqn{I_t + \bm{\delta}} is the new information set in which the \eqn{\bm{Y}_t}
#' is shocked by \eqn{\bm{\delta}}, \eqn{\mathbf{J} =
#' \left[\mathbf{I}_{n_Y}\ \bm{0}\ \cdots\ \bm{0}\right]}, and
#' \deqn{
#' \tilde{\mathbf{A}}(\bm{\alpha}) = \begin{bmatrix}
#' \mathbf{A}(\bm{\alpha})_1 & \mathbf{A}(\bm{\alpha})_2 & \cdots &
#' \mathbf{A}(\bm{\alpha})_{p-1} & \mathbf{A}(\bm{\alpha})_p \\
#' \mathbf{I}_{n_Y} & \bm{0} & \cdots & \bm{0} & \bm{0} \\
#' \bm{0} & \mathbf{I}_{n_Y} & \cdots & \bm{0} & \bm{0} \\
#' \vdots & \vdots & \ddots & \vdots  & \vdots \\
#' \bm{0} & \bm{0} & \cdots & \mathbf{I}_{n_Y} & \bm{0}
#' \end{bmatrix}.
#' }
#' \subsection{Mean Quantile Impulse Response function}{
#' The mean QIRF, \eqn{\bar{\text{QIRF}}(\bm{\alpha}_h)}, is defined as the mean
#' response of \eqn{\bm{\alpha}_H}-quantile to shock at time \eqn{t}, i.e.
#' \deqn{
#' \int_{\bm{\alpha}_{h-1}} \cdots\int_{\bm{\alpha}_2} \int_{\bm{\alpha}_1}
#' \text{QIRF}(\bm{\alpha}_h|\bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_{h-1})
#' \mathrm{d}\bm{\alpha}_1 \mathrm{d} \bm{\alpha}_2 \cdots \mathrm{d}\bm{\alpha}_{h-1} =
#' \mathbf{J} \tilde{\mathbf{A}}(\bm{\alpha}_H) {\bar{\tilde{\mathbf{A}}}}^{h-1}  \mathbf{J}^{\intercal},
#' }
#' where \eqn{{\bar{\tilde{\mathbf{A}}}}^{h-1} = \int_{\bm{\alpha}} \tilde{\mathbf{A}}(\bm{\alpha})
#' \mathrm{d} \bm{\alpha} }.
#' The technical details on (mean) QIRF calculation can be found in the
#' \insertCite{montes-rojas_multivariate_2019}{bayesQVAR}. In practice, the integral
#' can be approximated by the Monte Carlo method or directly autoregressive coefficient
#' matrices of VAR. In \code{calQIRF}, the latter method is used.
#' }
#' }
#' \subsection{Structural shock}{
#' When calculating the response to \eqn{i}, the default shock \eqn{\bm{\delta}} is
#' a \eqn{n_Y\times 1} vector whose \eqn{i}-th element is 1, that is
#' \eqn{[0_1\ 0_2\ \cdots\ 0_{i-1}\ 1_i\ 0_{i+1}\ \cdots\ 0_{n_Y}]^{\intercal}}.
#' This is not desirable because of ignorance of the correlation and variation
#' difference between variables. Follow the convention of SVAR analysis,
#' \code{calQIRF} allows users to add a \eqn{n_Y \times n_Y} matrix
#' \eqn{\mathbf{A}(\bm{\alpha})_0} in \code{modelSpecif} to tell how structural
#' shock is calculated. \eqn{\mathbf{A}(\bm{\alpha})_0} acts in following way:
#' \deqn{
#' \bm{Y}_t = \mathbf{A}(\bm{\alpha})_+ \bm{Z}_{t-1} +
#' \mathbf{A}(\bm{\alpha})_0 \bm{\varepsilon}_t,
#' }
#' where \eqn{\bm{\varepsilon}_t} is orthogonal structural shock, and
#' \deqn{
#' \bm{Z}_t = \begin{bmatrix}
#' \bm{Y}_{t-1}^{\intercal} & \cdots & \bm{Y}_{t-p}^{\intercal} &
#' \bm{X}_{t-1}^{\intercal} & \cdots & \bm{X}_{t-q}^{\intercal} & 1
#' \end{bmatrix}^{\intercal}, \\
#' \mathbf{A}(\bm{\alpha})_+ = \begin{bmatrix}
#' \mathbf{A}(\bm{\alpha})_1 & \cdots & \mathbf{A}(\bm{\alpha})_p &
#' \mathbf{\Gamma}(\bm{\alpha})_1 & \cdots &
#' \mathbf{\Gamma}(\bm{\alpha})_q & \bm{q}(\bm{\alpha})
#' \end{bmatrix}' \\
#' \mathrm{Cov}(\bm{\varepsilon}_t) = \begin{bmatrix}
#' \sigma_1^2 & 0 & \cdots & 0 \\
#' 0 & \sigma_2^2 & \cdots & 0 \\
#' \vdots & \vdots & \ddots & \vdots \\
#' 0 & 0 & \cdots & \sigma_{n_Y}^2
#' \end{bmatrix}.
#' }
#' The diagonals of \eqn{\mathbf{A}(\bm{\alpha})_0} are ones. users can add their
#' \code{A0} into \code{modelSpecif} to tell the function how to calculate the
#' structural shocks.
#' }
#' \subsection{Counterfactual analysis}{
#' Referring to \insertCite{chen_direct_2023}{bayesQVAR}, \code{calQIRF} can perform
#' counterfactual analysis through argument \code{counterfactual}. There are two
#' ways to specify counterfactual components: \code{counterfactual = list(A0_counter,
#' Ap_counter)} or \code{counterfactual = list(B0_counter, Bp_counter)}, which
#' correspond to following structural models respectively:
#' \deqn{
#' \bm{Y}_t = \mathbf{A}(\bm{\alpha})_+ \bm{Z}_{t-1} +
#' \mathbf{A}(\bm{\alpha})_0 \bm{\varepsilon}_t, \\
#' \bm{Y}_t = \mathbf{B}(\bm{\alpha})_0 \bm{Y}_t +
#' \mathbf{B}(\bm{\alpha})_+ \bm{Z}_{t-1} + \bm{\varepsilon}_t.
#' }
#' \code{A0_counter} and \code{B0_counter} are \eqn{n_Y \times n_Y} matrices
#' that states instantaneous counterfactual rules, where the elements subject to
#' counterfactual analysis are up to users; remaining elements are \code{NA}.
#' For example, if one would like to see the impulse response in the case of none
#' of variables reacts to orthogonal shocks of others, then \code{A0 = diag(n)}
#' or \code{B0 = matrix(0, n, n)} should be used. \cr
#' \code{Ap_counter} and \code{Bp_counter} are \eqn{n_Y \times n_Y p + n_X q + 1}
#' matrices that declare the counterfactual intertemporal rules. Similarly, the
#' elements subject to counterfactual analysis are up to users; remaining elements
#' are \code{NA}. \cr
#' When calculating the counterfactual mean QIRF, \code{Ap_counter} and \code{Bp_counter}
#' will act on the mean autoregressive matrix \eqn{\mathbf{A}}. \cr
#' }
#' \subsection{Bootstrap confidence interval}{
#' Different to bootstrap procedure proposed by \insertCite{lutkepohl_new_2005}{bayesQVAR},
#' which samples residuals directly to simulate new data, the bootstrap procedure
#' for QIRF is more complicated since there are infinite choice of \eqn{\mathbf{A}(\bm{\alpha})_+}
#' available to calculate residuals. If we chose a particular \eqn{\mathbf{A}(\bm{\alpha})}
#' to calculate residuals and perform bootstrap sampling, simulated data can only
#' restore the autoregressive feature of \eqn{\bm{\alpha}}-quantile of original data. \cr
#' To recover the whole distribution the original data, we propose a new bootstrap
#' procedure which samples cumulative probabilities that realized in real world.
#' The procedure of simulating a new data set is as follows:
#' \enumerate{
#' \item{\strong{Find the realized cumulative probabilities.}} {For each \eqn{\bm{y}_t},
#' find the \eqn{\hat{\bm{\alpha}}_t} such that \eqn{\hat{Q}_{\hat{\bm{\alpha}_t}}
#' (\bm{Y}_t|I_t) = \bm{y}_t }}, where \eqn{\hat{Q}_{\bm{\alpha}}(\bm{Y}_t|I_t) =
#' \hat{\bm{q}}(\bm{\alpha}) + \sum_{j=1}^{p}  \hat{\mathbf{A}}(\bm{\alpha})_j \bm{y}_{t-j} +
#' \sum_{k=1}^q \hat{\bm{\Gamma}}(\bm{\alpha}) \bm{x}_{t-k}  }.
#' \item{\strong{Sample the realized cumulative probabilities.}} {
#' Sample \eqn{T} probability vectors from \eqn{\hat{\bm{\alpha}}_1, \hat{\bm{\alpha}}_2,
#' \cdots, \hat{\bm{\alpha}}_T } and \eqn{T} exogenous variable observations from
#' \eqn{\hat{\bm{x}}_1, \hat{\bm{x}}_2, \cdots, \hat{\bm{x}}_T } with replacement,
#' denoted as \eqn{\bm{\alpha}_1^{(s)}, \bm{\alpha}_2^{(s)}, \cdots, \bm{\alpha}_T^{(s)}}
#' and \eqn{\bm{x}_1^{(s)}, \bm{x}_2^{(s)}, \cdots, \bm{x}_T^{(s)}}.
#' }
#' \item{\strong{Simulate data.}} {
#' For each \eqn{t}, calculate \eqn{\bm{y}_t^{(s)} = \bm{q}\left(\bm{\alpha}_t^{(s)}\right) +
#' \sum_{j=1}^p \hat{\mathbf{A}}\left(\bm{\alpha}_t^{(s)}\right)_j \bm{y}_{t-j} +
#' \sum_{k=1}^q \hat{\bm{\Gamma}}\left(\bm{\alpha}_t^{(s)}\right)_k \bm{x}_{t-k}^{(s)}
#' }.
#' For \eqn{t = 1}, the initial values can be the first \eqn{p} observations of
#' \eqn{\bm{Y}} and the first \eqn{q} observations of \eqn{\bm{X}}.
#' }
#' }
#' After \code{n_simu} simulations, calculate the standard deviation of QIRF at
#' each horizon, \eqn{\hat{\sigma}_{\mathrm{QIRF},h}}, and construct the confidence interval by
#' \deqn{
#' \left[
#' \hat{\mathrm{QIRF}}_h - z_{\frac{\alpha}{2}} \hat{\sigma}_{\mathrm{QIRF},h},
#' \hat{\mathrm{QIRF}}_h + z_{1 - \frac{\alpha}{2}} \hat{\sigma}_{\mathrm{QIRF},h}
#' \right].
#' }
#' Consider that Gibbs sampling can be time-consuming, the bootstrap procedure is
#' parallelized by calling \code{\link{parLapply}}. \cr
#' In bootstrapping stage, a new penalty setting can be used by specifying
#' \code{penalty_boots}. The main purpose to do so is to prevent the module of
#' eigenvalue of autoregressive matrix in bootstrapping stage to be greater than
#' 1 without introducing too much shrinkage bias in the original estimates of
#' \eqn{\hat{\mathbf{A}}(\bm{\alpha})_+}. \cr
#' }
#' \subsection{Credibility interval}{
#' The credibility interval is obtained by sampling from the posterior distribution
#' of QIRF and then estimate percentiles. Given the data, the source of uncertainty
#' of QIRF is from the parameter uncertainty of the autoregressive matrix at each
#' horizon, i.e. \eqn{\mathbf{A}(\bm{\alpha})_+ | \text{Data} \sim f(\mathbf{A}
#' (\bm{\alpha})_+ | \text{Data})}, where \eqn{f(\cdot)} is a certain distribution
#' function. \cr
#' Calculation of the impulse response of \eqn{j} to \eqn{i} at horizon \eqn{h} follows below steps:
#' \enumerate{
#' \item{\strong{Draw ramdom sample of autoregressive matrix from posterior dstribution. }}{Draw one random samples from the last \code{n_sample} MCMC iterations after thinning to represent a random sample of the posterior distribution of the autoregressive matrix at horizon \eqn{1, 2, \cdots, h}. Denote it as \eqn{\mathbf{A}(\bm{\alpha})_{+,1}^{(s)}, \mathbf{A}(\bm{\alpha})_{+,2}^{(s)}, \cdots, \mathbf{A}(\bm{\alpha})_{+,h}^{(s)}}.}
#' \item{\strong{Calculate QIRF based on sampled autoregressive matrices., denoted as \eqn{\hat{\text{QIRF}}_h^{(s)}}.}}{}
#' \item{\strong{Find percentiles of QIRF. }}{Repeat step 1 and 2 until get \code{n_simu} samples of \eqn{\hat{\text{QIRF}}_h}, i.e. \eqn{\hat{\text{QIRF}}_h^{(1)}, \hat{\text{QIRF}}_h^{(2)}, \cdots, \hat{\text{QIRF}}_h^{(n_\text{simu})}}. Use \eqn{\alpha / 2 \times 100\%}-percentile and \eqn{(1-\alpha / 2) \times 100\%}-percentile of them to represent the lower and upper bounds of credibility interval.}
#' }
#' In current version, \eqn{\mathbf{A}(\bm{\alpha})_0} is treated fixed, which is the original input by users or identity matrix.
#' }
#' \subsection{Be cautious!}{
#' The choice of penalty parameter can only be set manually in current version.
#' The results of bootstrap confidence interval may be sensitive to the penalty,
#' strongly biased and thus not very meaningful.
#' }
#' @examples man/examples/calQIRF.R
#' @references
#' \insertAllCited{}
#' @export
calQIRF <- function(
    modelSpecif,
    names_impulse,
    names_response,
    horizon,
    probPath,
    mean = FALSE,
    counterfactual = NULL,
    confInt = FALSE,
    credInt = TRUE,
    alpha = c(0.10, 0.32),
    n_simu = 100,
    penalty_boots = NULL)
{
  qirfList <- list() # output list

  # Fetch data frame of endogenous and exogenous variable from modelSpecif----
  data_end <- modelSpecif[["data_end"]]
  data_exo <- modelSpecif[["data_exo"]]

  # Declare dimension of variables, lag orders, and sample size----
  n_impulse <- length(names_impulse)
  n_response <- length(names_response)
  n_end <- ncol(data_end)
  if(is.null(data_exo)){
    n_exo <- 0
  }else{
    n_exo <- ncol(data_exo)
  }
  lag <- modelSpecif[["lag"]]
  lag_end <- lag[1]
  lag_exo <- 0
  if(n_exo != 0){
      if (length(lag) == 1) {
          lag_exo <- lag[1]
      } else {
          lag_exo <- lag[2]
      }
  }
  n_x <- n_end * lag_end + n_exo * lag_exo + 1
  lag_max <- max(lag)
  sampleSize <- nrow(data_end) - lag_max

  # Sampler setting----
  n_burn <- modelSpecif[["samplerSetting"]][["n_burn"]]
  n_thin <- modelSpecif[["samplerSetting"]][["n_thin"]]
  n_sample <- modelSpecif[["samplerSetting"]][["n_sample"]]

  # Estimate square root of covariance matrix of residuals----
  designMat <- .constDesignMat(data_end, data_exo, lag_end, lag_exo)
  Y <- designMat[["Y"]]
  X <- designMat[["X"]]
  prior_VAR <- list(
    mu_A = matrix(0, n_end, n_x),
    Sigma_A = rep(list(100 * diag(n_x)), n_end),
    Sigma = diag(n_end),
    nu = 0
  )
  if( all( c("n_lambda", "s_lambda") %in% names(modelSpecif[["prior"]]) ) ){
    prior_VAR[["n_lambda"]] <- modelSpecif[["prior"]][["n_lambda"]]
    prior_VAR[["s_lambda"]] <- modelSpecif[["prior"]][["s_lambda"]]
  }
  samplerSetting_VAR <- list(
    init_Sigma = diag(n_end),
    n_sample = modelSpecif[["samplerSetting"]][["n_sample"]],
    n_burn = modelSpecif[["samplerSetting"]][["n_burn"]],
    n_thin = modelSpecif[["samplerSetting"]][["n_thin"]]
  )
  sigmaSqrt <- .estSigmaOfBVAR(data_end, lag, data_exo, prior_VAR, samplerSetting_VAR)

  # Calculate QIRF of observed data and save them into qirfList----
  res_calQIRFOnce <- .calQIRFOnce(
    modelSpecif,
    names_impulse,
    names_response,
    sigmaSqrt,
    horizon,
    probPath,
    mean,
    counterfactual
  )
  qirfList[["irf"]] <- irf <- res_calQIRFOnce[["irf"]]
  if (mean == TRUE) {
    qirfList[["irf_mean"]] <- irf_mean <- res_calQIRFOnce[["irf_mean"]]
  }
  if (!is.null(counterfactual)) {
    qirfList[["irf_counter"]] <- irf_counter <- res_calQIRFOnce[["irf_counter"]]
    qirfList[["diffIrf"]] <- diffIrf <- res_calQIRFOnce[["diffIrf"]]
    if (mean == TRUE) {
      qirfList[["irf_mean_counter"]] <- irf_mean_counter <- res_calQIRFOnce[["irf_mean_counter"]]
      qirfList[["diffIrf_mean"]] <- diffIrf_mean <- res_calQIRFOnce[["diffIrf_mean"]]
    }
  }
  colNames_irf <- colnames(irf)

  # Construct Bootstrap confidence interval of QIRF----
  confIntList <- list()
  if (confInt == TRUE) {
    ## Function: estimate A with one probability vector-----
    estAwithOneProbVec <- function(
        prob,        # vector, which quantile is estimated
        n_end,       # integer, number of endogenous variables
        modelSpecif  # list, model specification
    ) {
      probVec <- as.matrix(rep(prob, n_end))
      A <- .estMultiBQVAR(modelSpecif, probVec)[["AList"]][[1]]
      return(A)
    }
    ## Parallel: estimate A for each probability vector----
    probSeq <- seq(0.001, 0.999, 0.001)
    cl <- parallel::makeCluster(floor(parallel::detectCores() / 3 * 2))
    AList_eachProb <- parallel::parLapply(
      cl = cl,
      X = probSeq,
      fun = estAwithOneProbVec,
      n_end = n_end,
      modelSpecif = modelSpecif
    )
    ## Function: calculate QIRF with simulated data----
    calQIRFOncewithSimuData <- function(
        arg_unused,     # it will not be used in this function
        # argument for .bootSimuOnce
        data_end,       # data frame of endogenous variables
        data_exo,       # data frame of exogenous variables
        lag_end,        # lag order of endogenous variables
        lag_exo,        # lag order of exogenous variables
        sampleSize,     # effective sample size of data
        AList_eachProb, # list of autoregressive matrices
        # argument for re-construct modelSpecif_boots
        modelSpecif,    # original model specification
        penalty_boots,  # penalty setting for bootstrap sampling
        # argument for .calQIRFOnce
        names_impulse,  # names of impulse variables
        names_response, # names of response variables
        sigmaSqrt,      # square root of covariance matrix of residuals
        horizon,        # forecast horizon
        probPath,       # probability path
        mean,           # logical value indicating whether to calculate the mean of QIRF
        counterfactual  # list of counterfactual components
    ){
      ### Simulate data----
      dataList_boots <- .bootsSimuOnce(
        data_end,
        data_exo,
        lag_end,
        lag_exo,
        sampleSize,
        AList_eachProb
      )
      ### Replace data in model specification by simulated data----
      modelSpecif_boots <- modelSpecif
      modelSpecif_boots[["data_end"]] <- dataList_boots[["data_end"]]
      modelSpecif_boots[["data_exo"]] <- dataList_boots[["data_exo"]]
      ### Replace penalty setting in model specification by new penalty setting----
      if(!is.null(penalty_boots)){
        modelSpecif_boots[["prior"]][["n_lambda"]] <- penalty_boots[["n_lambda"]]
        modelSpecif_boots[["prior"]][["s_lambda"]] <- penalty_boots[["s_lambda"]]
      }
      ### Calculate QIRF with new model specification----
      res_calOIRFOnce <- .calQIRFOnce(
        modelSpecif_boots,
        names_impulse,
        names_response,
        sigmaSqrt,
        horizon,
        probPath,
        mean,
        counterfactual
      )
      return(res_calOIRFOnce)
    }
    ## Parallel: calculate QIRF with simulated data sets----
    res_calBootsQIRF <- parallel::parLapply(
      cl = cl,
      X = 1:n_simu,
      fun = calQIRFOncewithSimuData,
      data_end = data_end,
      data_exo = data_exo,
      lag_end = lag_end,
      lag_exo = lag_exo,
      sampleSize = sampleSize,
      AList_eachProb = AList_eachProb,
      modelSpecif = modelSpecif,
      penalty_boots = penalty_boots,
      names_impulse = names_impulse,
      names_response = names_response,
      sigmaSqrt = sigmaSqrt,
      horizon = horizon,
      probPath = probPath,
      mean = mean,
      counterfactual = counterfactual
    )
    if(credInt == FALSE){
      parallel::stopCluster(cl)
    }

    ## Function: Extract QIRF from res_calQIRFboots----
    extractQIRFbyName <- function(res_calBootsQIRF, name_irf) {
      irfList <- lapply(
        res_calBootsQIRF,
        function(x){
          df_irf <- x[[name_irf]]
          colnames(df_irf) <- colNames_irf # colNames_irf is a global variable
          return(df_irf)
        }
      )
      return(irfList)
    }
    irfList_boots <- extractQIRFbyName(res_calBootsQIRF, "irf")
    if (mean == TRUE) {
      irfList_mean_boots <- extractQIRFbyName(res_calBootsQIRF, "irf_mean")
    }
    if (!is.null(counterfactual)) {
      irfList_counter_boots <- extractQIRFbyName(res_calBootsQIRF, "irf_counter")
      diffIrfList_boots <- extractQIRFbyName(res_calBootsQIRF, "diffIrf")
      if (mean == TRUE) {
        irfList_mean_counter_boots <- extractQIRFbyName(res_calBootsQIRF, "irf_mean_counter")
        diffIrfList_mean_boots <- extractQIRFbyName(res_calBootsQIRF, "diffIrf_mean")
      }
    }

    ## Loop: calculate bootstrap standard deviation and confidence interval----
    calSigmaBasedConfIntBoundsofQIRF <- function(irf, irfList_boots, alpha){
      upper <- lower <- irf
      for (i in 1:(horizon + 1)) {              # horizon is global variable
        for (j in 1:(n_impulse * n_response)) { # n_impulse and n_response are global variables
          irf_boots_ij <- unlist(
            lapply(
              irfList_boots,
              function(x, i, j) { x[i, j] },
              i = i, j = j
            )
          )
          sd_irf_ij <- sd(irf_boots_ij)
          upper[i, j] <- upper[i, j] + qnorm(1 - alpha / 2) * sd_irf_ij
          lower[i, j] <- lower[i, j] - qnorm(1 - alpha / 2) * sd_irf_ij
        }
      }
      return(list(upper = upper, lower = lower))
    }

    for (a in 1:length(alpha)) {
      alpha_a <- alpha[a]
      name_a <- paste0("ci_", format(round(alpha_a * 100), nsmall = 1))
      confIntList[["irf"]][[name_a]] <- calSigmaBasedConfIntBoundsofQIRF(irf, irfList_boots, alpha_a)
      if (mean == TRUE) {
        confIntList[["irf_mean"]][[name_a]] <- calSigmaBasedConfIntBoundsofQIRF(irf_mean, irfList_mean_boots, alpha_a)
      }
      if (!is.null(counterfactual)) {
        confIntList[["irf_counter"]][[name_a]] <- calSigmaBasedConfIntBoundsofQIRF(irf_counter, irfList_counter_boots, alpha_a)
        confIntList[["diffIrf"]][[name_a]] <- calSigmaBasedConfIntBoundsofQIRF(diffIrf, diffIrfList_boots, alpha_a)
        if (mean == TRUE) {
          confIntList[["irf_mean_counter"]][[name_a]] <- calSigmaBasedConfIntBoundsofQIRF(irf_mean_counter, irfList_mean_counter_boots, alpha_a)
          confIntList[["diffIrf_mean"]][[name_a]] <- calSigmaBasedConfIntBoundsofQIRF(diffIrf_mean, diffIrfList_mean_boots, alpha_a)
        }
      }
    }
  } # end of if(confInt == TRUE)

  # Construct credibility interval of QIRF----
  credIntList <- list()
  if(credInt == TRUE){

    ## Construct (counterfactual) A0 matrix----
    A0 <- diag(n_end)
    if( !is.null(modelSpecif[["A0"]]) ){
      A0 <- modelSpecif[["A0"]]
    }else if (!is.null(modelSpecif[["B0"]])){
      B0 <- modelSpecif[["B0"]]
      A0 <- solve(diag(n_end) - B0)
    }
    if(!is.null(counterfactual)){
      # validate counterfactual components
      if( sum(duplicated(names(counterfactual))) > 0 ){
        stop("Counterfactual matrices may be contradictary or duplicated.")
      }else{
        if(!all(names(counterfactual) %in% c("A0_counter", "Ap_counter", "B0_counter", "Bp_counter", "Ap_mean_counter", "Bp_mean_counter") )){
          stop("counterfactual should not contain any other components except A0_counter, Ap_counter, B0_counter, Bp_counter, Ap_mean_counter, Bp_mean_counter.")
        }
      }
      # fetch counterfactual components
      if(sum(names(counterfactual) %in% c("A0_counter", "Ap_counter")) == 2){
        A0_NA_counter <- counterfactual[["A0_counter"]]
        Ap_NA_counter <- counterfactual[["Ap_counter"]]
      }else if(sum(names(counterfactual) %in% c("B0_counter", "Bp_counter")) == 2){
        B0_NA_counter <- counterfactual[["B0_counter"]]
        Bp_NA_counter <- counterfactual[["Bp_counter"]]
      }
      if(sum(names(counterfactual) == c("Ap_mean_counter")) == 1){
        Ap_mean_NA_counter <- counterfactual[["Ap_mean_counter"]]
      }else if(sum(names(counterfactual) == c("Bp_mean_counter")) == 1){
        Bp_mean_NA_counter <- counterfactual[["Bp_mean_counter"]]
      }else{
        if(exists("Ap_NA_counter")){
          Ap_mean_NA_counter <- Ap_NA_counter
        }else if(exists("Bp_NA_counter")){
          Bp_mean_NA_counter <- Bp_NA_counter
        }
      }
      # construct counterfactual A0 matrix
      if(exists("A0_NA_counter")){
        A0_counter <- A0
        A0_counter[!is.na(A0_NA_counter)] <- A0_NA_counter[!is.na(A0_NA_counter)]
      }else if(exists("B0_NA_counter")){
        B0 <- diag(n_end) - solve(A0)
        B0_counter <- B0
        B0_counter[!is.na(B0_NA_counter)] <- B0_NA_counter[!is.na(B0_NA_counter)]
        A0_counter <- solve(diag(n_end) - B0_counter)
      }
    }

    ## Extract MCMC chains of A and A_mean matrices----
    mcmcChainList_A <- res_calQIRFOnce[["mcmcChainList_A"]]
    if(mean == TRUE){
      mcmcChain_Amean <- res_calQIRFOnce[["mcmcChain_Amean"]]
    }

    ## Construct list of posterior Ap and counterfactual Ap with MCMC chains-----
    AList_posterior_eachHorizon <- list()
    AList_counter_posterior_eachHorizon <- list()
    for(h in 1:horizon){
      mcmcChain_Ah_removeBurn <- mcmcChainList_A[[h]][-(1:(n_burn+1)), ]
      mcmcChain_Ah_thinning <- mcmcChain_Ah_removeBurn[
        seq(1, nrow(mcmcChain_Ah_removeBurn), n_thin),
      ]
      AList_posterior_eachHorizon[[h]] <- lapply(
        1:nrow(mcmcChain_Ah_thinning),
        function(i, y, n_end, n_x){
            matrix(y[i, ], n_end, n_x)
        },
        y = mcmcChain_Ah_thinning,
        n_end = n_end,
        n_x = n_x
      )
      if(!is.null(counterfactual)){
        AList_counter_posterior_eachHorizon[[h]] <- lapply(
          1:nrow(mcmcChain_Ah_thinning),
          function(i, y, n_end, n_x){
            Ap_actual <- matrix(y[i, ], n_end, n_x)
            Ap_counter <- Ap_actual
            if(exists("Ap_NA_counter")){
              Ap_counter[!is.na(Ap_NA_counter)] <- Ap_NA_counter[!is.na(Ap_NA_counter)]
            }else if(exists("Bp_NA_counter")){
              Bp_actual <- solve(A0) %*% Ap_actual
              Bp_counter <- Bp_actual
              Bp_counter[!is.na(Bp_NA_counter)] <- Bp_NA_counter[!is.na(Bp_NA_counter)]
              Ap_counter <- A0_counter %*% Bp_counter
            }
            return(Ap_counter)
          },
          y = mcmcChain_Ah_thinning,
          n_end = n_end,
          n_x = n_x
        )
      }
    }
    ## Construct list of posterior (counterfactual) A_mean with MCMC chains----
    if(mean == TRUE){
      mcmcChain_Amean_removeBurn <- mcmcChain_Amean[-(1:(n_burn+1)), ]
      mcmcChain_Amean_thinning <- mcmcChain_Amean_removeBurn[
        seq(1, nrow(mcmcChain_Amean_removeBurn), n_thin),
      ]
      AList_mean_posterior <- lapply(
        1:nrow(mcmcChain_Amean_thinning),
        function(i, y, n_end, n_x){
          matrix(y[i, ], n_end, n_x)
        },
        y = mcmcChain_Amean_thinning,
        n_end = n_end,
        n_x = n_x
      )
      if(!is.null(counterfactual)){
        AList_mean_counter_posterior <- lapply(
          1:nrow(mcmcChain_Amean_thinning),
          function(i, y, n_end, n_x){
            Ap_mean_actual <- matrix(y[i, ], n_end, n_x)
            Ap_mean_counter <- Ap_mean_actual
            if(exists("Ap_NA_counter")){
              Ap_mean_counter[!is.na(Ap_mean_NA_counter)] <- Ap_mean_NA_counter[!is.na(Ap_mean_NA_counter)]
            }else if(exists("Bp_NA_counter")){
              Bp_mean_actual <- solve(A0) %*% Ap_mean_actual
              Bp_mean_counter <- Bp_mean_actual
              Bp_mean_counter[!is.na(Bp_mean_NA_counter)] <- Bp_mean_NA_counter[!is.na(Bp_mean_NA_counter)]
              Ap_mean_counter <- A0_counter %*% Bp_mean_counter
            }
            return(Ap_mean_counter)
          },
          y = mcmcChain_Amean_thinning,
          n_end = n_end,
          n_x = n_x
        )
      }
    }

    ## Calculate QIRF for each randomly sampled A, A_mean, A_counter and A_mean_counter matrices----
    idx_impulse <- match(names_impulse, colnames(data_end)) - 1
    idx_response <- match(names_response, colnames(data_end)) - 1
    calQIRFwithSampledAPost <- function(
      aug_unused, AList_posterior_eachHorizon, A0, sigmaSqrt,
      idx_impulse, idx_response, colNames_irf,
      n_sample, n_end, n_x, lag_end, horizon,
      mean = FALSE, AList_mean_posterior = NULL
    ){
      idx_sample <- sample(1:n_sample, horizon, replace = TRUE)
      AList_eachHorizon_sample <- mapply(
        function(x, i){ x[[i]] },
        AList_posterior_eachHorizon, idx_sample,
        SIMPLIFY = FALSE
      )
      if(mean == FALSE){
        A_mean_sample <- diag(1)
      }else{
        A_mean_sample <- AList_mean_posterior[[sample(1:n_sample, 1)]]
      }
      res_calQIRFwithA <- .calQIRFwithA(
        AList_eachHorizon_sample,
        sigmaSqrt, A0,
        idx_impulse, idx_response,
        n_end, lag_end,
        mean, A_mean_sample
      )
      colnames(res_calQIRFwithA) <- colNames_irf
      return(res_calQIRFwithA)
    }

    if(confInt == FALSE){
      cl <- parallel::makeCluster(floor(parallel::detectCores() / 3 * 2))
    }
    res_sampleIrf <- parallel::parLapply(
      cl = cl,
      X = 1:n_simu,
      fun = calQIRFwithSampledAPost,
      AList_posterior_eachHorizon = AList_posterior_eachHorizon,
      A0 = A0,
      sigmaSqrt = sigmaSqrt,
      idx_impulse = idx_impulse,
      idx_response = idx_response,
      colNames_irf = colNames_irf,
      n_sample = n_sample,
      n_end = n_end,
      n_x = n_x,
      lag_end = lag_end,
      horizon = horizon,
      mean = FALSE,
      AList_mean_posterior = NULL
    )
    if(mean == TRUE){
      res_sampleIrf_mean <- parallel::parLapply(
        cl = cl,
        X = 1:n_simu,
        fun = calQIRFwithSampledAPost,
        AList_posterior_eachHorizon = AList_posterior_eachHorizon,
        A0 = A0,
        sigmaSqrt = sigmaSqrt,
        idx_impulse = idx_impulse,
        idx_response = idx_response,
        colNames_irf = colNames_irf,
        n_sample = n_sample,
        n_end = n_end,
        n_x = n_x,
        lag_end = lag_end,
        horizon = horizon,
        mean = TRUE,
        AList_mean_posterior = AList_mean_posterior
      )
    }
    if(!is.null(counterfactual)){
      res_sampleIrf_counter <- parallel::parLapply(
        cl = cl,
        X = 1:n_simu,
        fun = calQIRFwithSampledAPost,
        AList_posterior_eachHorizon = AList_counter_posterior_eachHorizon,
        A0 = A0_counter,
        sigmaSqrt = sigmaSqrt,
        idx_impulse = idx_impulse,
        idx_response = idx_response,
        colNames_irf = colNames_irf,
        n_sample = n_sample,
        n_end = n_end,
        n_x = n_x,
        lag_end = lag_end,
        horizon = horizon,
        mean = FALSE,
        AList_mean_posterior = NULL
      )
      if(mean == TRUE){
        res_sampleIrf_mean_counter <- parallel::parLapply(
          cl = cl,
          X = 1:n_simu,
          fun = calQIRFwithSampledAPost,
          AList_posterior_eachHorizon = AList_counter_posterior_eachHorizon,
          A0 = A0_counter,
          sigmaSqrt = sigmaSqrt,
          idx_impulse = idx_impulse,
          idx_response = idx_response,
          colNames_irf = colNames_irf,
          n_sample = n_sample,
          n_end = n_end,
          n_x = n_x,
          lag_end = lag_end,
          horizon = horizon,
          mean = TRUE,
          AList_mean_posterior = AList_mean_counter_posterior
        )
      }
    }
    on.exit(parallel::stopCluster(cl))
    ## Calculate difference between randomly sampled counterfactual QIRF and QIRF----
    if(!is.null(counterfactual)){
      res_sampleDiffIrf <- lapply(
        1:n_simu,
        function(i, res_sampleIrf, res_sampleIrf_counter){
          return(res_sampleIrf_counter[[i]] - res_sampleIrf[[i]])
        },
        res_sampleIrf = res_sampleIrf,
        res_sampleIrf_counter = res_sampleIrf_counter
      )
      if(mean == TRUE){
        res_sampleDiffIrf_mean <- lapply(
          1:n_simu,
          function(i, res_sampleIrf, res_sampleIrf_counter){
            return(res_sampleIrf_counter[[i]] - res_sampleIrf[[i]] )
          },
          res_sampleIrf = res_sampleIrf_mean,
          res_sampleIrf_counter = res_sampleIrf_mean_counter
        )
      }
    }

    ## Find percentile of QIRF or difference QIRF----
    calPercentileBasedCredIntBoundsofQIRF <- function(irfList_sample, alpha){
      upper <- lower <- irfList_sample[[1]]
      for (h in 1:(horizon + 1)) {              # horizon is global variable
        for (i in 1:(n_impulse * n_response)) { # n_impulse and n_response are global variables
          irf_sample_hi <- unlist(
            lapply(
              irfList_sample,
              function(x){ return(x[h, i]) }
            )
          )
          upper[h, i] <- quantile(irf_sample_hi, 1 - alpha_a / 2)
          lower[h, i] <- quantile(irf_sample_hi, alpha_a / 2)
        }
      }
      return(list(upper = upper, lower = lower))
    }
    for(a in 1:length(alpha)){
      alpha_a <- alpha[a]
      name_a <- paste0("ci_", format(round(alpha_a * 100), nsmall = 1))
      credIntList[["irf"]][[name_a]] <- calPercentileBasedCredIntBoundsofQIRF(res_sampleIrf, alpha_a)
      if(mean == TRUE){
        credIntList[["irf_mean"]][[name_a]] <- calPercentileBasedCredIntBoundsofQIRF(res_sampleIrf_mean, alpha_a)
      }
      if(!is.null(counterfactual)){
        credIntList[["irf_counter"]][[name_a]] <- calPercentileBasedCredIntBoundsofQIRF(res_sampleIrf_counter, alpha_a)
        credIntList[["diffIrf"]][[name_a]] <- calPercentileBasedCredIntBoundsofQIRF(res_sampleDiffIrf, alpha_a)
        if(mean == TRUE){
          credIntList[["irf_mean_counter"]][[name_a]] <- calPercentileBasedCredIntBoundsofQIRF(res_sampleIrf_mean_counter, alpha_a)
          credIntList[["diffIrf_mean"]][[name_a]] <- calPercentileBasedCredIntBoundsofQIRF(res_sampleDiffIrf_mean, alpha_a)
        }
      }
    }
  } # end of if(credInt == TRUE)

  # Construct BQVAR_QIRF object----
  res <- BQVAR_QIRF(
    list(
      modelSpecif = modelSpecif,
      horizon = horizon,
      probPath = probPath,
      names_impulse = names_impulse,
      names_response = names_response,
      shockScale = sigmaSqrt,
      confInt = confInt,
      credInt = credInt,
      alpha = alpha,
      qirfList = qirfList,
      confIntList = confIntList,
      credIntList = credIntList
    )
  )

  return(res)
}
