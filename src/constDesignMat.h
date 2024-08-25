# ifndef DESIGNMATCONST_H
# define DESIGNMATCONST_H

# include <Rcpp.h>
# include "bayesQVAR_types.h"

using namespace Rcpp;

List constDesignMat(
    const RcppDf &data_end,
    const Rcpp::Nullable<RcppDf> &data_exo,
    const int &lag_end,
    const int &lag_exo
);

# endif
