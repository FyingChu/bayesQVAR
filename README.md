# Installation

Step 1: install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) first.
Rtools provides tool to compile source files in the packages.

Step 2: For manual installation, you need to install the dependencies, then download the source package file or binary package file from Release and install it locally. Source package file needs compilation, but binary package does not. If the technique detail of algorithm is not your interests, binary package is more recommended for users.

For remote installation, install `remotes` package and then use `install_github` function to install `bayesQVAR`. The dependencies will be installed automatically. 

```R
# Manual Installation
# install dependencies
install.packages(
  c("Rcpp", "parallel", "GIGrvg", "LaplacesDemon", "S7", "patchwork", "ggplot2", "stringr")
)
# If you download source package file
install.packages(
  "path to your download folder/bayesQVAR_1.0.0.tar.gz",
  type = "source"
)
# if you download binary package file
install.packages(
  "path to your download folder/bayesQVAR_1.0.0.zip",
  type = "binary"
)
# Remote installation
remotes::install_github("FyingChu/bayesQVAR")
```

# Technical details

Please refer to [user manual](https://bookdown.org/zhufengyi810/bayesqvar_user_manual/).
