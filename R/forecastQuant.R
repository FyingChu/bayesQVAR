#' @title Forecast Quantiles
#' @name forecastQuant
#' @description Forecast quantile using QVAR, given the probability paths along
#' which the endogenous variables in the VAR system will evolve.
#' @param modelSpecif a List that contains QVAR model specification information,
#'  which includes the following components:
#' \itemize{
#'  \item{\code{data_end}} {Data frame of endogenous variables.}
#'  \item{\code{data_exo}} {Data frame of exogenous variables.}
#'  \item{\code{data_exo_forecast}} (optional) {Data frame of forecasts of exogenous
#'   variables within forecast horizons. If not provided, then the sample mean of
#'   \code{data_exo} will be used.}
#'  \item{\code{lag}} {Lag order of endogenous and exogenous variables.}
#'  \item{\code{prior}} {List of prior specification.}
#'  \item{\code{samplerSetting}} {List of initial values and of number of valid sampling,
#'  burn-in period and thinning.}
#'  \item{\code{method}} {String of estimation method.}
#' }
#' Refer to \code{\link{estBQVAR}} for more details.
#' @param horizon Integer that indicates forecast horizon \eqn{H}.
#' @param probPath \eqn{n_Y \times H} Matrix of probability paths. Each column
#' corresponds the quantile of interest at horizon \eqn{h}.
#' @param mean Logical value indicating whether to calculate the mean of QIRF, see details.
#' @return An object of class \code{BQVAR_forecast}, which contains the quantile
#' forecasts along particular evolve path, \code{quant}, and mean quantile forecasts,
#' \code{quant_mean}.
#' @details
#' \subsection{Quantile forecasts for specific probability path}{
#' For following \eqn{n_Y}-dimensional QVAR process
#' \deqn{
#' Q_{\bm{\alpha}}(\bm{Y}_t|I_t) = \bm{q}(\bm{\alpha}) +
#' \sum\limits_{j=1}^p\mathbf{A}(\bm{\alpha}) \bm{Y}_{t-j} +
#' \sum\limits_{k=1}^q \bm{X}_{t-k},
#' }
#' provided that the system evolves along the probability path
#' \eqn{\bm{\alpha}_1, \cdots, \bm{\alpha}_{h-1}} between \eqn{t+1} and \eqn{t+h-1}
#' exactly, i.e.
#' \deqn{
#' \bm{y}_{t+1} = Q_{ \bm{\alpha}_1}(\bm{Y}_{t+1} | I_{t+1} ), \\
#' \bm{y}_{t+2} = Q_{\bm{\alpha}_2}(\bm{Y}_{t+2} | \bm{y}_{t+1}, I_{t+1} ), \\
#' \vdots \\
#' \bm{y}_{t+h-1} = Q_{\bm{\alpha}_{h-1}}(\bm{Y}_{t+h-1} | \bm{y}_{t+1}, \cdots,
#' \bm{y}_{t+h-2}, I_{t+1} ),
#' }
#' then we can forecast the quantile for the path in recursive way:
#' \deqn{
#'  \hat{Q}_{\bm{\alpha}_h}(\bm{Y}_{t+h} | \bm{\alpha}_1, \bm{\alpha}_2, \cdots,
#'  \bm{\alpha}_{h-1}, I_{t+1} ) = \bm{q}(\bm{\alpha}_h) + \sum_{j=1}^p\mathbf{A}(\bm{\alpha}_h)_j
#'  \hat{Q}_{\bm{\alpha}_{h-j}}(\bm{Y}_{t+h-j} | I_{t+1}) + \sum_{k=1}^q
#'  \mathbf{\Gamma}(\bm{\alpha}_h)_k \hat{\bm{X}}_{t+h-j},\ h = 1, 2, \cdots, H,
#' }
#' where \eqn{ \hat{Q}_{\bm{\alpha}_{h-j}}(\bm{Y_{t+h-j}} | I_{t+1}) = \bm{y}_{t+h-j} }
#' and \eqn{\hat{\bm{X}}_{t + h - j} = \bm{x}_{t+h-j}} if \eqn{h-j \leq 0}. The
#' forecasts of exogenous variables, should be provided in \code{data_exo_forecast}.
#' Note that only \eqn{H - 1} forecasts of exogenous variables are needed since
#' the are fetched from the last \eqn{q} rows of \code{data_exo}.
#' }
#' \subsection{Mean quantile forecasts}{
#' Mean quantile forecast at time \eqn{t+h} is
#' \deqn{
#' \bar{\hat{Q}}_{\bm{\alpha}_h}(\bm{Y}_{t+h}|I_{t+1}) = \int_{\bm{\alpha}_{h-1}}
#' \cdots \int_{\bm{\alpha}_2} \int_{\bm{\alpha}_1} \hat{Q}_{\bm{\alpha}_h}(\bm{Y}_{t+h} |
#' \bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_{h-1}, I_{t+1} )
#' \mathrm{d}\bm{\alpha}_1 \mathrm{d}\bm{\alpha}_2 \cdots \cdots d\bm{\alpha}_{h-1}.
#' }
#' Integrating over the probabilities is not feasible, so we use the mean autoregressive
#' matrix to perform the forecast from \eqn{1} to \eqn{h-1}, then perform quantile
#' forecast at the last horizon. This is equivalent to forecasting the quantile at
#' \eqn{h}, assuming that the system evolves along the mean path from \eqn{1} to \eqn{h}.
#' }
#' @example man/examples/forecastQuant.R
forecastQuant <- function(
    modelSpecif,
    horizon,
    probPath,
    mean = FALSE
){

    res_forecastQuant <- .forecastQuant(modelSpecif, horizon, probPath, mean)
    res <- BQVAR_forecast(res_forecastQuant)
    return(res)
}