gplot <- S7::new_generic(
    name = "gplot",
    dispatch_args = "x"
)

#' @name gplot
#' @title Plotting method for BQVAR_QIRF
#' @description Plotting Quantile Impulse Response Functions (QIRFs) and their confidence intervals.
#' @param x BQVAR_QIRF object.
#' @param response Character vector of names of response variable.
#' @param impulse Character vector of names of impulse variable.
#' @param type_irf Character that indicates the type of QIRF to be plotted. Four basic types of IRF are \code{"irf"}(QIRF), \code{"irf_counter"}(counterfactual QIRF), \code{"irf_mean"}(mean QIRF) and \code{"irf_mean_counter"}(counterfactual mean QIRF). Two different types of IRF can be drawn in the same plot by separating the names of basic types by \code{" vs "}, such as \code{"irf vs irf_mean"}. If two types of IRF are either \code{"irf"} and \code{"irf_counter"} or \code{"irf_mean"} and \code{"irf_mean_counter"}, then the difference between actual and counterfactual IRF will be plotted too.
#' @param type_int Character that indicates the which kind of interval is plotted, which should be one of \code{"none"}(no interval), \code{"confidence"}(bootstrapping confidence interval) and \code{"credibility"}(credibility interval).
#' @param alpha Numeric vector of that contains significance levels of confidence or credibility intervals. e.g. \code{alpha = 0.1} means 90% confidence or credibility interval. The levels that are not estimated in the \code{BQVAR_QIRF} object will be omitted.
#' @param lineWidth_irf Integer that indicates line width of IRF.
#' @param color_irf Character of the color of IRF line.
#' @param color_ci Character vector of the colors of confidence interval bands. The length of the vector should be equal to the number of IRF types multiplied by the number of significance levels. For example, if \code{type_irf = "irf vs irf_counter"} and \code{type_int = c(0.10, 0.32, 0.50)}, then 6 colors should be provided, where the first three colors are for the 90%, 68% and 50% band of QIRF respectively , and the last three colors are for counterfactual IRF.
#' @param labs_irf object of \code{labels} class for axis titles of QIRF plot.
#' @param labs_diff object of \code{labels} class for axis titles of difference in QIRF plot.
#' @param font_lab object of \code{element} class for axis titles.
#' @param font_axis object of \code{element} class for axis values.
#' @return ggplot object.
#' @import patchwork
#' @example man/examples/calQIRF.R
#' @export
S7::method(gplot, BQVAR_QIRF) <- function(
        x,
        response,
        impulse,
        type_irf = "irf",
        type_int = "none",
        alpha = x@alpha[1],
        lineWidth_irf = 1,
        color_irf = "red",
        color_ci = "red",
        labs_irf = labs(x = "Horizon", y = "Response"),
        labs_diff = labs(x = "Horizon", y = "Counterfactual QIRF - QIRF"),
        font_lab = element_text(size = 12, family = "serif"),
        font_axis = element_text(size = 12, family = "serif"),
        scale_x = scale_x_continuous(
          breaks = seq(0, horizon, 1),
          expand = c(0, 0)
        ),
        ...
){
  # Check if the QIRF types to be drawn are estimated----
  irfTypes_ploted <- as.vector(stringr::str_split(type_irf, " vs ", simplify = TRUE))
  irfTypes_estimated <- names(x@qirfList)
  if(!all(irfTypes_ploted %in% irfTypes_estimated)){
    irfTypes_ploted <- irfTypes_ploted[irfTypes_ploted %in% irfTypes_estimated]
    if(length(irfTypes_ploted) == 0){
      stop("None of IRF types is available.")
    }else{
      message("Only ", stringr::str_c(irfTypes_ploted, collapse = ", "), " will be ploted.")
    }
  }
  horizon <- x@horizon
  name_irf <- paste0("response ", response, " to ", impulse)

  # Construct data frame of lower and upper bounds of confidence intervals----
  if(type_int != "none"){
    if(type_int == "confidence" & length(x@confIntList) == 0){
      stop("Confidence interval is not available.")
    }else if(type_int == "credibility" & length(x@credIntList) == 0){
      stop("Credibility interval is not available.")
    }else{
      if(type_int == "confidence" ){
        ciList <- x@confIntList[irfTypes_ploted]
      }else if(type_int == "credibility"){
        ciList <- x@credIntList[irfTypes_ploted]
      }else{
        stop("Invalid interval type.")
      }
      alpha_estimated <- names(ciList[[1]])
      alpha_ploted <- stringr::str_c("ci_", format(round(alpha * 100), nsmall = 1) )
      alpha_ploted <- stringr::str_replace_all(alpha_ploted, " ", "")
      if(!all(alpha_ploted %in% alpha_estimated)){
        alpha_ploted <- alpha_ploted[alpha_ploted %in% alpha_estimated]
        if(length(alpha_ploted) == 0){
          stop("None of significance levels is available.")
        }
        message(
          "Only band of ",
          stringr::str_c(stringr::str_c(stringr::str_remove(alpha_ploted, "ci_"), "%"), collapse = ", "),
          " will be ploted."
        )
      }
      ciBoundsDf <- do.call(
        rbind,
        lapply(  # first level: type_irf
          ciList,
          function(x){
            ciBoundsList_eachAlpha <- lapply(  # second level: alpha
              x,
              function(y){
                ciBoundsDf_alpha <- data.frame(
                  upper = y$upper[, name_irf],
                  lower = y$lower[, name_irf],
                  check.names = FALSE
                )
                ciBoundsDf_alpha$horizon <- 0:horizon
                return(ciBoundsDf_alpha)
              }
            )
            ciBoundsDf_eachAlpha <- do.call(rbind, ciBoundsList_eachAlpha)
            ciBoundsDf_eachAlpha$alpha <- rep(alpha_estimated, each = horizon + 1)
            return(ciBoundsDf_eachAlpha)
          }
        )
      )
      ciBoundsDf$type_irf <- rep(irfTypes_ploted, each = length(alpha_estimated) * (horizon + 1))
      ciBoundsDf$confInt <- paste0(ciBoundsDf$type_irf, "_", ciBoundsDf$alpha)
      ciBoundsDf <- subset(ciBoundsDf, alpha %in% alpha_ploted)
    }
  }

  # Construct data frame of QIRFs----
  qirfList <- x@qirfList[irfTypes_ploted]
  qirfDf <- do.call(rbind, qirfList)
  qirfDf$horizon <- rep(0:horizon, length(irfTypes_ploted))
  qirfDf$type_irf <- rep(irfTypes_ploted, each = horizon + 1)

  # Loop: draw each irf type----
  require(ggplot2)
  for(i in 1:length(irfTypes_ploted)){
    if(!exists("p", inherits = FALSE)){
      p <- ggplot()
    }
    ## add confidence or credibility interval band----
    if(type_int %in% c("confidence", "credibility")){
      for(j in 1:length(alpha_ploted)){
        p <- p + geom_ribbon(
          data = subset(
            ciBoundsDf,
            type_irf == irfTypes_ploted[i] & alpha == alpha_ploted[j]
          ),
          aes(
              x = horizon,
              ymin = lower,
              ymax = upper,
          ),
          fill = color_ci[(i - 1) * length(alpha_ploted) + j],
          color = color_ci[(i - 1) * length(alpha_ploted) + j],
          alpha = 0.25
        )
      }
    }
    ## add QIRF lines----
    p <- p +
      geom_line(
        data = subset(qirfDf, type_irf == irfTypes_ploted[i]),
        aes(
            x = horizon,
            y = eval( parse(text = paste0("`", name_irf, "`") ) ),
        ),
        color = color_irf[i],
        linewidth = lineWidth_irf
      )
  }

  # add extra layers and theme-----
  p <- p +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
    ) +
    labs_irf +
    scale_x +
    scale_y_continuous(
      labels = function(x) formatC(x, width = 6)
    ) +
    scale_color_manual(
      values = color_irf
    ) +
    scale_fill_manual(
      values = color_ci
    ) +
    theme_linedraw() +
    theme(
      panel.grid.minor = element_blank(),
      axis.text = font_axis,
      axis.title = font_lab
    )

  # If type_irf is either c("irf", "irf_counter") or c("irf_mean", "irf_mean_counter"), add plot of difference in QIRF ----
  if(
    sum(irfTypes_ploted %in% c("irf", "irf_counter")) == 2 |
    sum(irfTypes_ploted %in% c("irf_mean", "irf_mean_counter")) == 2
  ){
    diffIrfType <- paste0(
      "diff",
      stringr::str_to_title(stringr::str_remove(irfTypes_ploted, "_counter")[1])
    )
    diffIrfDf <- x@qirfList[[diffIrfType]]
    diffIrfDf$horizon <- 0:horizon
    p_diff <- ggplot(
      data = diffIrfDf
    ) +
      geom_point(
        aes(
          x = horizon,
          y = eval( parse(text = paste0("`", name_irf, "`") ) )
        )
      )

    if(type_int %in% c("confidence", "credibility")){
      if(type_int == "confidence" ){
        ciList_diff <- x@confIntList[diffIrfType]
      }else if(type_int == "credibility"){
        ciList_diff <- x@credIntList[diffIrfType]
      }
      ciBoundsDf_diff <- do.call(
        rbind,
        lapply(  # first level: type_irf
          ciList_diff,
          function(x){
            ciBoundsList_diff_eachAlpha <- lapply(  # second level: alpha
              x,
              function(y){
                ciBoundsDf_diff_alpha <- data.frame(
                  upper = y$upper[, name_irf],
                  lower = y$lower[, name_irf],
                  check.names = FALSE
                )
                ciBoundsDf_diff_alpha$horizon <- 0:horizon
                return(ciBoundsDf_diff_alpha)
              }
            )
            ciBoundsDf_diff_eachAlpha <- do.call(rbind, ciBoundsList_diff_eachAlpha)
            ciBoundsDf_diff_eachAlpha$alpha <- rep(alpha_estimated, each = horizon + 1)
            return(ciBoundsDf_diff_eachAlpha)
          }
        )
      )
      ciBoundsDf_diff <- subset(ciBoundsDf_diff, alpha %in% alpha_ploted)
      p_diff <- p_diff +
        geom_errorbar(
          data = subset(ciBoundsDf_diff, alpha == max(alpha_ploted)),
          aes(
            x = horizon,
            ymin = lower,
            ymax = upper
          ),
          width = 0.4,
          linewidth = 0.6
        )
    }
    p_diff <- p_diff +
      geom_hline(
        yintercept = 0,
        linetype = "dashed",
      ) +
      labs_diff +
      scale_x +
      scale_y_continuous(
        labels = function(x) formatC(x, width = 6)
      ) +
      coord_cartesian(xlim = c(0, horizon)) +
      theme_linedraw() +
      theme(
        panel.grid.minor = element_blank(),
        axis.text = font_axis,
        axis.title = font_lab
      )

    require(patchwork)
    p_merge <- p + p_diff +
      plot_layout(
        ncol = 1,
        nrow = 2
      )

  }else{ # if type_irf is not specified as "actual vs counterfactual", then only draw the QIRF type specified
      p_merge <- p
  }
  print(p_merge)
}


