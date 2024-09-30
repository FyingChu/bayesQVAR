# Installation

Install the dependencies, then download the source package file or binary package file from Release and install it locally. Source package file needs compilation, but binary package does not. If the technique detail of algorithm is not your interests, binary package is more recommended for users.

```R

# install dependencies
install.packages(
  c("Rcpp", "parallel", "GIGrvg", "LaplacesDemon", "Rdpack", "S7", "patchwork", "ggplot2", "stringr")
)
# if you download source package file
install.packages(
  "path to your download folder/bayesQVAR_1.0.0.tar.gz",
  type = "source"
)
# if you download binary package file
install.packages(
  "path to your download folder/bayesQVAR_1.0.0.zip",
  type = "binary"
)
```

# Technical details

Please refer to [user manual](https://bookdown.org/zhufengyi810/bayesqvar_user_manual/).
