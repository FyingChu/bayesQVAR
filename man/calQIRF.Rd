% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/calQIRF.R
\name{calQIRF}
\alias{calQIRF}
\title{Calculate Quantile Impulse Response Function}
\usage{
calQIRF(
  modelSpecif,
  names_impulse,
  names_response,
  horizon,
  probPath,
  mean = FALSE,
  counterfactual = NULL,
  confInt = FALSE,
  credInt = TRUE,
  alpha = c(0.1, 0.32),
  n_simu = 100,
  penalty_boots = NULL
)
}
\arguments{
\item{modelSpecif}{List of model specification, which includes the following
components:
\itemize{
\item{\code{data_end}} {Data frame of endogenous variables.}
\item{\code{data_exo}} {Data frame of exogenous variables.}
\item{\code{lag}} {Lag order of endogenous and exogenous variables.}
\item{\code{prior}} {List of prior specification.}
\item{\code{samplerSetting}} {List of initial values and of number of valid
sampling, burn-in period and thinning.}
\item{\code{method}} {String of estimation method.}
\item{\code{B0}} {Instantaneous structural matrix.}
}
Refer to \code{\link{estBQVAR}} function for more details on all components
except for \code{B0}. About how \code{B0} works, see details.}

\item{names_impulse}{Vector of names of impulse variables.}

\item{names_response}{Vector of names of response variables.}

\item{horizon}{Integer that indicates forecast horizon \eqn{H}.}

\item{probPath}{\eqn{n_Y \times H} matrix of probability paths. Each column
corresponds to the quantile of interest at horizon \eqn{h}.}

\item{mean}{Logical value indicating whether to calculate the mean of QIRF,
see details.}

\item{counterfactual}{List that contains components for counterfactual analysis,
which includes \code{names}, \code{rules_inst} and \code{rules_inter}. See details.}

\item{confInt}{Logical value indicating whether to calculate the confidence
interval of QIRF. See details on how to the confidence interval are calculated.}

\item{credInt}{Logical value indicating whether to calculate the credibility
interval. See details on how to the credibility interval are calculated.}

\item{alpha}{Vector of significance level.}

\item{n_simu}{Integer that indicates the number of bootstrapping to calculate
bootstrapping standard deviation of QIRF or/and how many samples of QIRF are drawn
to calculate the posterior percentile of QIRF.}

\item{penalty_boots}{Penalty setting for estimation of QVAR based on bootstrapping
simulated data. If \code{NULL}, the original penalty setting in \code{modelSpecif}
will be used.}
}
\value{
An object of class \code{BQVAR_QIRF}, which contains model specification
information, mean QIRF (\code{mean = TRUE}), counterfactual QIRF
(\code{!is.null(counterfactual)}), counterfactual mean QIRF (\code{mean = TRUE}
and \code{!is.null(counterfactual)}) and its' bootstrap confidence interval
(\code{confInt = TRUE}).
}
\description{
Calculate Quantile Impulse Response Function(QIRF) and its' bootstrap
confidence interval, given the probability paths along which the endogenous
variables in the QVAR system will evolve within horizon \eqn{H}.This function
also support counterfactual analysis.
}
\details{
\subsection{Quantile impulse response function for specific probability path}{
For following \eqn{n_Y}-dimensional QVAR process
\deqn{
Q_{\bm{\alpha}}(\bm{Y}_t|I_t) = \bm{q}(\bm{\alpha}) + \sum\limits_{j=1}^p
\mathbf{A}(\bm{\alpha}) \bm{Y}_{t-j} + \sum\limits_{k=1}^q \bm{X}_{t-k},
}
provided that the system evolves along the probability path
\eqn{\bm{\alpha}_1, \cdots, \bm{\alpha}_{h-1}} between \eqn{t+1} and \eqn{t+h-1}
exactly, then the response of conditional \eqn{\bm{\alpha}}-quantile at \eqn{t+h}
to the shock \eqn{\bm{\delta}} at time \eqn{t}, denoted as
\eqn{\text{QIRF}(\bm{\alpha}_{h}|\bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_{h-1})},
is defined as
\deqn{
Q_{\bm{\alpha}}(\bm{Y}_{t+h}|\bm{y}_{t+1}, \cdots, \bm{y}_{t+h-1}, I_t + \bm{\delta})
 - Q_{\bm{\alpha}}(\bm{Y}_{t+h}|\bm{y}_{t+1}, \cdots, \bm{y}_{t+h-1}, I_t)
 = \mathbf{J} \left[\prod\limits_{g=1}^h \tilde{\mathbf{A}}(\bm{\alpha}_{g}) \right]
 \mathbf{J}^{\intercal},
}
where \eqn{I_t + \bm{\delta}} is the new information set in which the \eqn{\bm{Y}_t}
is shocked by \eqn{\bm{\delta}}, \eqn{\mathbf{J} =
\left[\mathbf{I}_{n_Y}\ \bm{0}\ \cdots\ \bm{0}\right]}, and
\deqn{
\tilde{\mathbf{A}}(\bm{\alpha}) = \begin{bmatrix}
\mathbf{A}(\bm{\alpha})_1 & \mathbf{A}(\bm{\alpha})_2 & \cdots &
\mathbf{A}(\bm{\alpha})_{p-1} & \mathbf{A}(\bm{\alpha})_p \\
\mathbf{I}_{n_Y} & \bm{0} & \cdots & \bm{0} & \bm{0} \\
\bm{0} & \mathbf{I}_{n_Y} & \cdots & \bm{0} & \bm{0} \\
\vdots & \vdots & \ddots & \vdots  & \vdots \\
\bm{0} & \bm{0} & \cdots & \mathbf{I}_{n_Y} & \bm{0}
\end{bmatrix}.
}
\subsection{Mean Quantile Impulse Response function}{
The mean QIRF, \eqn{\bar{\text{QIRF}}(\bm{\alpha}_h)}, is defined as the mean
response of \eqn{\bm{\alpha}_H}-quantile to shock at time \eqn{t}, i.e.
\deqn{
\int_{\bm{\alpha}_{h-1}} \cdots\int_{\bm{\alpha}_2} \int_{\bm{\alpha}_1}
\text{QIRF}(\bm{\alpha}_h|\bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_{h-1})
\mathrm{d}\bm{\alpha}_1 \mathrm{d} \bm{\alpha}_2 \cdots \mathrm{d}\bm{\alpha}_{h-1} =
\mathbf{J} \tilde{\mathbf{A}}(\bm{\alpha}_H) {\bar{\tilde{\mathbf{A}}}}^{h-1}  \mathbf{J}^{\intercal},
}
where \eqn{{\bar{\tilde{\mathbf{A}}}}^{h-1} = \int_{\bm{\alpha}} \tilde{\mathbf{A}}(\bm{\alpha})
\mathrm{d} \bm{\alpha} }.
The technical details on (mean) QIRF calculation can be found in the
\insertCite{montes-rojas_multivariate_2019}{bayesQVAR}. In practice, the integral
can be approximated by the Monte Carlo method or directly autoregressive coefficient
matrices of VAR. In \code{calQIRF}, the latter method is used.
}
}
\subsection{Structural shock}{
When calculating the response to \eqn{i}, the default shock \eqn{\bm{\delta}} is
a \eqn{n_Y\times 1} vector whose \eqn{i}-th element is 1, that is
\eqn{[0_1\ 0_2\ \cdots\ 0_{i-1}\ 1_i\ 0_{i+1}\ \cdots\ 0_{n_Y}]^{\intercal}}.
This is not desirable because of ignorance of the correlation and variation
difference between variables. Follow the convention of SVAR analysis,
\code{calQIRF} allows users to add a \eqn{n_Y \times n_Y} matrix
\eqn{\mathbf{A}(\bm{\alpha})_0} in \code{modelSpecif} to tell how structural
shock is calculated. \eqn{\mathbf{A}(\bm{\alpha})_0} acts in following way:
\deqn{
\bm{Y}_t = \mathbf{A}(\bm{\alpha})_+ \bm{Z}_{t-1} +
\mathbf{A}(\bm{\alpha})_0 \bm{\varepsilon}_t,
}
where \eqn{\bm{\varepsilon}_t} is orthogonal structural shock, and
\deqn{
\bm{Z}_t = \begin{bmatrix}
\bm{Y}_{t-1}^{\intercal} & \cdots & \bm{Y}_{t-p}^{\intercal} &
\bm{X}_{t-1}^{\intercal} & \cdots & \bm{X}_{t-q}^{\intercal} & 1
\end{bmatrix}^{\intercal}, \\
\mathbf{A}(\bm{\alpha})_+ = \begin{bmatrix}
\mathbf{A}(\bm{\alpha})_1 & \cdots & \mathbf{A}(\bm{\alpha})_p &
\mathbf{\Gamma}(\bm{\alpha})_1 & \cdots &
\mathbf{\Gamma}(\bm{\alpha})_q & \bm{q}(\bm{\alpha})
\end{bmatrix}' \\
\mathrm{Cov}(\bm{\varepsilon}_t) = \begin{bmatrix}
\sigma_1^2 & 0 & \cdots & 0 \\
0 & \sigma_2^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_{n_Y}^2
\end{bmatrix}.
}
The diagonals of \eqn{\mathbf{A}(\bm{\alpha})_0} are ones. users can add their
\code{A0} into \code{modelSpecif} to tell the function how to calculate the
structural shocks.
}
\subsection{Counterfactual analysis}{
Referring to \insertCite{chen_direct_2023}{bayesQVAR}, \code{calQIRF} can perform
counterfactual analysis through argument \code{counterfactual}. There are two
ways to specify counterfactual components: \code{counterfactual = list(A0_counter,
Ap_counter)} or \code{counterfactual = list(B0_counter, Bp_counter)}, which
correspond to following structural models respectively:
\deqn{
\bm{Y}_t = \mathbf{A}(\bm{\alpha})_+ \bm{Z}_{t-1} +
\mathbf{A}(\bm{\alpha})_0 \bm{\varepsilon}_t, \\
\bm{Y}_t = \mathbf{B}(\bm{\alpha})_0 \bm{Y}_t +
\mathbf{B}(\bm{\alpha})_+ \bm{Z}_{t-1} + \bm{\varepsilon}_t.
}
\code{A0_counter} and \code{B0_counter} are \eqn{n_Y \times n_Y} matrices
that states instantaneous counterfactual rules, where the elements subject to
counterfactual analysis are up to users; remaining elements are \code{NA}.
For example, if one would like to see the impulse response in the case of none
of variables reacts to orthogonal shocks of others, then \code{A0 = diag(n)}
or \code{B0 = matrix(0, n, n)} should be used. \cr
\code{Ap_counter} and \code{Bp_counter} are \eqn{n_Y \times n_Y p + n_X q + 1}
matrices that declare the counterfactual intertemporal rules. Similarly, the
elements subject to counterfactual analysis are up to users; remaining elements
are \code{NA}. \cr
When calculating the counterfactual mean QIRF, \code{Ap_counter} and \code{Bp_counter}
will act on the mean autoregressive matrix \eqn{\mathbf{A}}. \cr
}
\subsection{Bootstrap confidence interval}{
Different to bootstrap procedure proposed by \insertCite{lutkepohl_new_2005}{bayesQVAR},
which samples residuals directly to simulate new data, the bootstrap procedure
for QIRF is more complicated since there are infinite choice of \eqn{\mathbf{A}(\bm{\alpha})_+}
available to calculate residuals. If we chose a particular \eqn{\mathbf{A}(\bm{\alpha})}
to calculate residuals and perform bootstrap sampling, simulated data can only
restore the autoregressive feature of \eqn{\bm{\alpha}}-quantile of original data. \cr
To recover the whole distribution the original data, we propose a new bootstrap
procedure which samples cumulative probabilities that realized in real world.
The procedure of simulating a new data set is as follows:
\enumerate{
\item{\strong{Find the realized cumulative probabilities.}} {For each \eqn{\bm{y}_t},
find the \eqn{\hat{\bm{\alpha}}_t} such that \eqn{\hat{Q}_{\hat{\bm{\alpha}_t}}
(\bm{Y}_t|I_t) = \bm{y}_t }}, where \eqn{\hat{Q}_{\bm{\alpha}}(\bm{Y}_t|I_t) =
\hat{\bm{q}}(\bm{\alpha}) + \sum_{j=1}^{p}  \hat{\mathbf{A}}(\bm{\alpha})_j \bm{y}_{t-j} +
\sum_{k=1}^q \hat{\bm{\Gamma}}(\bm{\alpha}) \bm{x}_{t-k}  }.
\item{\strong{Sample the realized cumulative probabilities.}} {
Sample \eqn{T} probability vectors from \eqn{\hat{\bm{\alpha}}_1, \hat{\bm{\alpha}}_2,
\cdots, \hat{\bm{\alpha}}_T } and \eqn{T} exogenous variable observations from
\eqn{\hat{\bm{x}}_1, \hat{\bm{x}}_2, \cdots, \hat{\bm{x}}_T } with replacement,
denoted as \eqn{\bm{\alpha}_1^{(s)}, \bm{\alpha}_2^{(s)}, \cdots, \bm{\alpha}_T^{(s)}}
and \eqn{\bm{x}_1^{(s)}, \bm{x}_2^{(s)}, \cdots, \bm{x}_T^{(s)}}.
}
\item{\strong{Simulate data.}} {
For each \eqn{t}, calculate \eqn{\bm{y}_t^{(s)} = \bm{q}\left(\bm{\alpha}_t^{(s)}\right) +
\sum_{j=1}^p \hat{\mathbf{A}}\left(\bm{\alpha}_t^{(s)}\right)_j \bm{y}_{t-j} +
\sum_{k=1}^q \hat{\bm{\Gamma}}\left(\bm{\alpha}_t^{(s)}\right)_k \bm{x}_{t-k}^{(s)}
}.
For \eqn{t = 1}, the initial values can be the first \eqn{p} observations of
\eqn{\bm{Y}} and the first \eqn{q} observations of \eqn{\bm{X}}.
}
}
After \code{n_simu} simulations, calculate the standard deviation of QIRF at
each horizon, \eqn{\hat{\sigma}_{\mathrm{QIRF},h}}, and construct the confidence interval by
\deqn{
\left[
\hat{\mathrm{QIRF}}_h - z_{\frac{\alpha}{2}} \hat{\sigma}_{\mathrm{QIRF},h},
\hat{\mathrm{QIRF}}_h + z_{1 - \frac{\alpha}{2}} \hat{\sigma}_{\mathrm{QIRF},h}
\right].
}
Consider that Gibbs sampling can be time-consuming, the bootstrap procedure is
parallelized by calling \code{\link{parLapply}}. \cr
In bootstrapping stage, a new penalty setting can be used by specifying
\code{penalty_boots}. The main purpose to do so is to prevent the module of
eigenvalue of autoregressive matrix in bootstrapping stage to be greater than
1 without introducing too much shrinkage bias in the original estimates of
\eqn{\hat{\mathbf{A}}(\bm{\alpha})_+}. \cr
}
\subsection{Credibility interval}{
The credibility interval is obtained by sampling from the posterior distribution
of QIRF and then estimate percentiles. Given the data, the source of uncertainty
of QIRF is from the parameter uncertainty of the autoregressive matrix at each
horizon, i.e. \eqn{\mathbf{A}(\bm{\alpha})_+ | \text{Data} \sim f(\mathbf{A}
(\bm{\alpha})_+ | \text{Data})}, where \eqn{f(\cdot)} is a certain distribution
function. \cr
Calculation of the impulse response of \eqn{j} to \eqn{i} at horizon \eqn{h} follows below steps:
\enumerate{
\item{\strong{Draw ramdom sample of autoregressive matrix from posterior dstribution. }}{Draw one random samples from the last \code{n_sample} MCMC iterations after thinning to represent a random sample of the posterior distribution of the autoregressive matrix at horizon \eqn{1, 2, \cdots, h}. Denote it as \eqn{\mathbf{A}(\bm{\alpha})_{+,1}^{(s)}, \mathbf{A}(\bm{\alpha})_{+,2}^{(s)}, \cdots, \mathbf{A}(\bm{\alpha})_{+,h}^{(s)}}.}
\item{\strong{Calculate QIRF based on sampled autoregressive matrices., denoted as \eqn{\hat{\text{QIRF}}_h^{(s)}}.}}{}
\item{\strong{Find percentiles of QIRF. }}{Repeat step 1 and 2 until get \code{n_simu} samples of \eqn{\hat{\text{QIRF}}_h}, i.e. \eqn{\hat{\text{QIRF}}_h^{(1)}, \hat{\text{QIRF}}_h^{(2)}, \cdots, \hat{\text{QIRF}}_h^{(n_\text{simu})}}. Use \eqn{\alpha / 2 \times 100\%}-percentile and \eqn{(1-\alpha / 2) \times 100\%}-percentile of them to represent the lower and upper bounds of credibility interval.}
}
In current version, \eqn{\mathbf{A}(\bm{\alpha})_0} is treated fixed, which is the original input by users or identity matrix.
}
\subsection{Be cautious!}{
The choice of penalty parameter can only be set manually in current version.
The results of bootstrap confidence interval may be sensitive to the penalty,
strongly biased and thus not very meaningful.
}
}
\examples{
# simulate data
library(bayesQVAR)
set.seed(810)
n <- 5
alpha <- rep(0.95, n)
bayesQVAR::simuData(n, alpha, 150)

# set model specification, counterfactual matrices
n_end <- ncol(data_end)
n_exo <- ncol(data_exo)
lag_end <- 1
lag_exo <- 1
n_x <- n_end * lag_end + n_exo * lag_exo + 1
s <- sqrt(2 / alpha / (1 - alpha))
xi <- (1 - 2 * alpha) / alpha / (1 - alpha)
Sigma <- diag(s^2, n_end, n_end)
Sigma_A <- list()
for(i in 1:n_end){
  Sigma_A[[i]] = 100 * diag(n_x)
}
prior <- list(
  mu_A = matrix(0, n_end, n_x),
  Sigma_A = Sigma_A,
  Sigma = diag(n_end),
  nu = 1000 * n_end,
  n_delta = rep(1, n_end),
  s_delta = rep(1, n_end),
  n_lambda = matrix(2, n_end, n_x),
  s_lambda = matrix(0.0001, n_end, n_x)
)
samplerSetting <- list(
  # initial value
  init_A = matrix(0, n_end, n_x),
  init_Sigma = diag(n_end),
  init_delta = rep(0.1, n_end),
  n_sample = 200,
  # burn-in, thinning, step size of delta
  n_burn = 200,
  n_thin = 2
)
modelSpecif_mal <- list(
  data_end = data_end,
  data_exo = data_exo,
  lag = c(lag_end, lag_exo),
  prior = prior,
  samplerSetting = samplerSetting,
  method = "bayes-mal"
)
Ap_counter <- cbind(A_alpha, 0, 0) # the coefficient on exogenous and constant term does not matter
counterfactual_actual <- list(
  A0_counter = diag(n_end),
  Ap_counter = Ap_counter,
  Ap_mean_counter = cbind(A_const, 0, 0)
)
colNames_end <- colnames(data_end)
# estimate QIRFs and confidence, credibility intervals
QIRF_mal <- bayesQVAR::calQIRF(
  modelSpecif = modelSpecif_mal,
  names_impulse = colNames_end,
  names_response = colNames_end,
  horizon = 12,
  probPath = matrix(alpha, nrow = n_end, ncol = 12),
  mean = TRUE,
  counterfactual = counterfactual_actual, # the irf_counter will be the true irf
  confInt = TRUE,
  credInt = TRUE,
  alpha = c(0.10, 0.32, 0.50),
  n_simu = 100
)
# plot QIRFs and compared it to the true irf
bayesQVAR::gplot(
  QIRF_mal,
  response = "V1",
  impulse = "V2",
  type_irf = "irf vs irf_counter",
  type_int = "credibility",
  alpha = 0.32,
  color_irf = c("#1ED760", "#FF6597"),
  color_ci = c("#1ED760",  "#FF6597"),
  fontSize_lab = 15,
  fontSize_axis = 15
)
bayesQVAR::gplot(
  QIRF_mal,
  response = "V1",
  impulse = "V2",
  type_irf = "irf_mean vs irf_mean_counter",
  type_int = "credibility",
  alpha = 0.10,
  color_irf = c("#1ED760", "#FF6597"),
  color_ci = c("#1ED760",  "#FF6597"),
  fontSize_lab = 15,
  fontSize_axis = 15
)
}
\references{
\insertAllCited{}
}
