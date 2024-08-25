library(bayesQVAR)
set.seed(810)
n <- 5
alpha <- rep(0.95, n)
bayesQVAR::simuData(n, alpha, 300)
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
    s_delta = rep(1, n_end),
    n_lambda = matrix(2, n_end, n_x),
    s_lambda = matrix(0.0001, n_end, n_x)
)
samplerSetting <- list(
    # initial value
    init_A = matrix(0, n_end, n_x),
    init_Sigma = Sigma,
    init_delta = rep(0.1, n_end),
    n_sample = 250,
    # burn-in, thinning, step size of delta
    n_burn = 250,
    n_thin = 1
)
modelSpecif <- list(
    data_end = data_end,
    data_exo = data_exo,
    lag = c(lag_end, lag_exo),
    prior = prior,
    samplerSetting = samplerSetting,
    method = "bayes-mal"
)
# forecast quantiles
res_forecastQuant <- forecastQuant(
    modelSpecif = modelSpecif,
    horizon = 48,
    probPath = matrix(
        c(0.95, 0.95, 0.95, 0.95, 0.95),
        nrow = n_end, ncol = 48
    ),
    mean = TRUE
)
# plot forecasts
data_end <- res_forecastQuant@modelSpecif[["data_end"]]
x <- rbind(data_end, res_forecastQuant@forecastList[["quant"]])
x_mean <- rbind(data_end, res_forecastQuant@forecastList[["quant_mean"]])
for(i in 1:n){
    plot(x[, i], col = "purple", type = "l")
    lines(x_mean[, i], col = "red")
    lines(x[1:nrow(data_end), i])
    lines(quant_alpha[, i], col = "blue")
}

