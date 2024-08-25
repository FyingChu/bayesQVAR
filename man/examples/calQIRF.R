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
