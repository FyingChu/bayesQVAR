estBVAR <- function(
    data_end,
    lag,
    data_exo = NULL,
    prior,
    samplerSetting
){
    res_BVAR <- .estBVAR(
        data_end,
        lag,
        data_exo,
        prior,
        samplerSetting
    )
    return(res_BVAR)
    
}