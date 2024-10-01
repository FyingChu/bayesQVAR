# simulate data
library(bayesQVAR)
set.seed(810)
n <- 5
alpha <- rep(0.95, n)
simuData(n, alpha, 300)
# set prior, samplerSetting
n_end <- ncol(data_end)
n_exo <- ncol(data_exo)
lag_end <- 1
lag_exo <- 1
n_x <- n_end * lag_end + n_exo * lag_exo + 1
alpha <- rep(rep(0.95, 5), n / 5)
s <- sqrt(2 / alpha / (1 - alpha))
xi <- (1 - 2 * alpha) / alpha / (1 - alpha)
Sigma <- diag(s^2)
Sigma_A <- list()
for(i in 1:n){
    Sigma_A[[i]] = 100 * diag(n_x)
}
prior <- list(
    mu_A = matrix(0, n_end, n_x),
    Sigma_A = Sigma_A,
    Sigma = Sigma,
    nu = 100 * n_end + 1,
    n_delta = rep(1, n_end),
    s_delta = rep(1, n_end)
)
samplerSetting <- list(
    # initial value
    init_A = matrix(0, n_end, n_x),
    init_Sigma = Sigma,
    init_delta = rep(0.1, n_end),
    n_sample = 250,
    # burn-in, thinning, step size of delta
    n_burn = 250,
    n_thin = 2
)
# estimate QVAR model
BQVAR_mal <- bayesQVAR::estBQVAR(
    data_end = data_end,
    data_exo = data_exo,
    alpha = alpha,
    lag = c(lag_end, lag_exo),
    method = "bayes-mal",
    prior = prior,
    samplerSetting = samplerSetting,
    printFreq = 250,
    mute = FALSE
)
BQVAR_al <- bayesQVAR::estBQVAR(
    data_end = data_end,
    data_exo = data_exo,
    alpha = alpha,
    lag = c(lag_end, lag_exo),
    method = "bayes-al",
    prior = prior,
    samplerSetting = samplerSetting,
    printFreq = 1,
    mute = FALSE
)
# Extract estimate matrices, and calculate quantile of interest
Y <- BQVAR_mal@designMat[["Y"]]
X <- BQVAR_mal@designMat[["X"]]
Ap_est_mal <- BQVAR_mal@estimates[["A"]]
Ap_est_al <- BQVAR_al@estimates[["A"]]
quant_est_mal <- Ap_est_mal %*% X
quant_est_al <- Ap_est_al %*% X
# plot the estimated quantile
for(i in 1:n_end){
    plot(Y[i, ], type = "l")
    lines(quant_alpha[-1, i], col = "blue")
    lines(quant_est_mal[i, ], col = "red")
}
for(i in 1:n_end){
    plot(Y[i, ], type = "l")
    lines(quant_alpha[-1, i], col = "blue")
    lines(quant_est_al[i, ], col = "red")
}
# performance evaluation
mse_a_al <- sqrt(sum((Ap_est_al[, 1:(n_x - 1)] - cbind(A_alpha, gamma_alpha))^2))
mse_a_mal <- sqrt(sum((Ap_est_mal[, 1:(n_x - 1)] - cbind(A_alpha, gamma_alpha))^2))
mse_quant_al <- sqrt(sum((quant_est_al - quant_alpha[-1, ])^2))
mse_quant_mal <- sqrt(sum((quant_est_mal - quant_alpha[-1, ])^2))
mse_quant_al; mse_quant_mal
mse_a_al; mse_a_mal