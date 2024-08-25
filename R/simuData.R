#' @title Simulate QVAR(1)
#' @description
#' The function simulates a \eqn{n}-dimensional \eqn{\text{QVAR}(1)} process with one exogenous variable,
#' given sample size \eqn{T}.
#' @details
#' The process is defined as
#' \deqn{
#' Q_{\bm{\alpha}}(\bm{Y}_t|I_t) = \bm{q}(\bm{\alpha}) + \mathbf{A}(\bm{\alpha}) \bm{Y}_{t-1} + \bm{\gamma} X_t,\ \bm{\alpha} \in (0, 1)^{n}, \\
#' \mathbf{A}(\bm{\alpha}) = \bar{\mathbf{A}} + 0.15( \bm{\alpha} - 0.5 \times \bm{1}_n ) \bm{1}_n^{\intercal},\\
#' \bm{Y}_{i0} \sim \mathcal{t}(2),\ q(\alpha_i) = F^{-1}(\alpha_i),\ \gamma_i \sim \mathcal{U}(-0.2, 0.2), i = 1, 2, \cdots, n,
#' }
#' where \eqn{F^{-1}(\cdot)} is the quantile function of \eqn{t(2)}.
#' @param n Dimension of the QVAR(1) process.
#' @param alpha Vector of the quantile level of interest, \eqn{\bm{\alpha}^*},.
#' @param sampleSize The sample size of simulated data.
#' @return The function exports following object into global environment:
#' \itemize{
#' \item \code{q_alpha}: \eqn{\bm{q}(\bm{\alpha}^*)},
#' \item \code{A_const}: \eqn{\bar{\mathbf{A}}},
#' \item \code{A_alpha}: \eqn{\mathbf{A}(\bm{\alpha}^*)},
#' \item \code{gamma_alpha}: \eqn{\bm{\gamma}},
#' \item \code{data_end}: data frame of endogenous variables, \eqn{\bm{y}_t, t=0, 1, \cdots, T},
#' \item \code{data_exo}: data frame of exogenous variables, \eqn{x_t, t=0, 1, \cdots, T-1},
#' \item \code{quant_alpha}: data frame of conditional quantile, \eqn{Q_{\bm{\alpha}^*}(\bm{Y}_t|I_t), t=0, 1, \cdots, T}.
#' }
#' @example man/examples/simuData.R
#' @export
simuData <- function(n, alpha, sampleSize){
  # set parameter
  A_const <<- matrix(
      c(0.15, 0.10, 0.00, 0.21, 0.15,
        0.05, 0.15, 0.10, 0.10, 0.15,
        0.11, -.10, 0.30, -.05, 0.00,
        0.21, 0.20, 0.10, 0.30, 0.20,
        0.07, 0.12, 0.04, -.08, 0.30),
      n, n, byrow = TRUE
  ) # diag(rep(0.15, n))
  A_vary <- matrix(0.15 * (alpha - 0.5), n, n)
  A_alpha <<- A_const + A_vary
  gamma_alpha <<- runif(n, -0.2, 0.2)
  q_alpha <<- 0.1 * qt(alpha, df = 2)
  Ap_alpha <- cbind(A_alpha, gamma_alpha, q_alpha)
  alphaSeq <- seq(0.0005, 0.9995, 0.0005)
  # data frame and initial value
  data_end <- as.data.frame(matrix(0.0, ncol = n, nrow = sampleSize + 1))
  data_exo <- as.data.frame(matrix(0, ncol = 1, nrow = sampleSize + 1))
  quant_alpha <- as.data.frame(matrix(0, ncol = n, nrow = sampleSize + 1))
  data_end[1, ] <- 0.1 * rt(n, df = 2)
  data_exo[1, ] <- 0.1 * rnorm(1)
  quant_alpha[1, ] <- 0.1 * qt(alpha, df = 2)
  # loop:simulation
  for(t in 2:(sampleSize + 1)){
    ylag <- t(data_end[t - 1, ])
    xlag <- data_exo[t - 1, ]
    x <- 0.1 * rnorm(1)
    data_exo[t, ] <- x
    quant_alpha[t - 1, ] <- as.numeric(Ap_alpha %*% rbind(ylag, xlag, 1))
    for(i in 1:n){
      cdf_it <- 0.1 * qt(alphaSeq, df = 2) + (A_const %*% ylag)[i] + colSums( 0.3 * ylag %*% t(alphaSeq - 0.5) ) + gamma_alpha[i] * xlag
      u <- sample(alphaSeq, 1)
      data_end[t, i] <- cdf_it[which(round(alphaSeq, 4) == round(u, 4))]
    }
  }
  data_end <<- data_end
  data_exo <<- data_exo
  quant_alpha <<- quant_alpha
}
