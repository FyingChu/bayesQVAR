#' @title BQVAR class

#' @description Class for Bayesian quantile vector autoregression (BQVAR) model, which is only used to create an object of formally defined class when return the output of \link{estBQVAR}.
#' @param x List that contains estimation results of Bayesian quantile vector autoregression (BQVAR).
#' @return \code{data_end}: Data.frame for the endogenous variables.
#' @return \code{data_exo}: Data.frame for the exogenous variables.
#' @return \code{lag}: Integer for the lag order.
#' @return \code{alpha}: Double for the quantile level.
#' @return \code{prior}: List for the prior settings.
#' @return \code{samplerSetting}: List for the sampler settings.
#' @return \code{method}: Character string for the estimation method.
#' @return \code{estimates}: List for the estimation results.
#' @return \code{mcmcChains}: List for the MCMC chains.
#' @return \code{residuals}: List for the residuals.
BQVAR <- S7::new_class(
    name = "BQVAR",
    properties = list(
        data = S7::class_list,
        designMat = S7::class_list,
        lag = S7::class_numeric,
        alpha = S7::class_any,
        method = S7::class_character,
        prior = S7::class_list,
        samplerSetting = S7::class_list,
        mcmcChains = S7::class_list,
        estimates = S7::class_list,
        residuals = S7::class_any
    ),
    constructor = function(x){

        S7::new_object(
            S7::S7_object(),
            data = x$data,
            designMat = x$designMat,
            lag = x$lag,
            alpha = x$alpha,
            method = x$method,
            prior = x$prior,
            samplerSetting = x$samplerSetting,
            mcmcChains = x$mcmcChains,
            estimates = x$estimates,
            residuals = x$residuals
        )
    }
)

#' @title BQVAR_forecast class
#' @description convert a list of quantile forecasts into a formally defined S7 class,which is only used to create an object of formally defined class when return the output of \link{forecastQuant}.
#' @param x List that contains quantile forecast.
#' @returns An object of class \code{BQVAR_forecast}, which contains:
#' @returns \code{modelSpecif}: ModelSpecif list for the model specification.
#' @returns \code{horizon}: Integer for the forecast horizon.
#' @returns \code{probPath}: Matrix for the quantile path.
#' @returns \code{forecastList}: List for the forecasts of quantile and mean. quantile.
BQVAR_forecast <- S7::new_class(
    name = "BQVAR_forecast",
    properties = list(
        modelSpecif = S7::class_list,
        horizon = S7::class_numeric,
        probPath = S7::class_numeric,
        forecastList = S7::class_list
    ),
    constructor = function(x){
        S7::new_object(
            S7::S7_object(),
            modelSpecif = x$modelSpecif,
            horizon = x$horizon,
            probPath = x$probPath,
            forecastList = x$forecastList
        )

    }
)

#' @title BQVAR_QIRF class
#' @description Class for quantile forecasts based on QVAR, which is only used to create an object of formally defined class when return the output of \link{calQIRF}.
#' @param x List that contains the quantile impulse response functions and its' confidence intervals.
#' @returns An object of class \code{BQVAR_QIRF}, which contains:
#' @returns \code{modelSpecif}: List for the model specification.
#' @returns \code{horizon}: Integer for the forecast horizon.
#' @returns \code{probPath}: Matrix for the quantile path.
#' @returns \code{names_impulse}: Character vector for the impulse variable names.
#' @returns \code{names_response}: character vector for the response variable names.
#' @returns \code{shockScale}: Numeric vector for the shock scale of each variable.
#' @returns \code{confInt}: Logical for whether the confidence intervals are estimated.
#' @returns \code{credInt}: Logical for whether the credibility intervals are estimated.
#' @returns \code{alpha}: Numeric vector for the significant level.
#' @returns \code{qirfList}: List for the quantile impulse response functions and its' confidence intervals.
#' @returns \code{confIntList}: List of the confidence intervals.
#' @returns \code{credIntList}: List of the credibility intervals.
BQVAR_QIRF <- S7::new_class(
    name = "BQVAR_QIRF",
    properties = list(
        modelSpecif = S7::class_list,
        horizon = S7::class_numeric,
        probPath = S7::class_numeric,
        names_impulse = S7::class_character,
        names_response = S7::class_character,
        shockScale = S7::class_numeric,
        confInt = S7::class_logical,
        credInt = S7::class_logical,
        alpha = S7::class_numeric,
        qirfList = S7::class_list,
        confIntList = S7::class_list,
        credIntList = S7::class_list
    ),
    constructor = function(x){

        S7::new_object(
            S7::S7_object(),
            modelSpecif = x$modelSpecif,
            horizon = x$horizon,
            probPath = x$probPath,
            names_impulse = x$names_impulse,
            names_response = x$names_response,
            shockScale = x$shockScale,
            confInt = x$confInt,
            credInt = x$credInt,
            alpha = x$alpha,
            qirfList = x$qirfList,
            confIntList = x$confIntList,
            credIntList = x$credIntList
        )

    }
)



