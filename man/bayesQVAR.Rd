\name{bayesQVAR}
\alias{bayesQVAR}
\alias{bayesQVAR}
\docType{package}
\title{
  Quantile AutoRegression.
}
\description{
  The package provides a toolkit of Bayesian estimation, forecast and impulse response analysis of Quantile Vector AutoRegression (QVAR) Model.
}
\details{
  Three functions are provided: \cr
  \code{\link{estBQVAR}}: Bayesian estimation of Quantile Vector AutoRegression (QVAR) Model using Gibbs sampler. The likelihood and prior are based on Asymmetric Laplace (AL) or Multivariate Asymmetric Laplace (MAL) distribution. \cr
  \code{  \link{forecastQuant}}: Forecasting of quantile using QVAR given the probability path. \cr
  \code{\link{calQIRF}}: calculate Quantile Impulse Response Function (QIRF) and its' bootstrap confidence interval, given the probability path. The function also supports counterfactual analysis.
}
\author{
Fengyi Zhu, zhufengyi810@live.com.

Maintainer: Fengyi Zhu <zhufengyi810@live.com>
}
\references{
  \insertRef{kozumi_gibbs_2011}{bayesQVAR} \cr
  \insertRef{kotz_laplace_2012}{bayesQVAR} \cr
  \insertRef{schuler_asymmetric_2014}{bayesQVAR} \cr
  \insertRef{montes-rojas_multivariate_2019}{bayesQVAR}
}
\keyword{ Bayesian, quantile autoregression, Gibbs sampler, forecast, quantile impulse response function}
