# Bayesian Qauntile VAR

## Installation

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

# Technique Details

## Model Specification of QVAR

The **$P$**-order QVAR at probability vector$\boldsymbol{\alpha}=[\alpha_1\ \alpha_2\ \cdots\ \alpha_N]^{\top} \in (0, 1)^{N}$without exogenous variable, denoted as $\mathrm{QVAR}(P)$takes the form of

$$
\begin{gather}
\boldsymbol{Y}_t = \boldsymbol{q}(\boldsymbol{\alpha}) + \sum_{p=1}^P \mathbf{A}(\boldsymbol{\alpha})_p Y_{t-p} + \boldsymbol{u}(\boldsymbol{\alpha})_t, Q_{\boldsymbol{\alpha}}(\boldsymbol{u}(\boldsymbol{\alpha})_t|I_t) = \boldsymbol{0}, \\
\boldsymbol{Y}_t = \begin{bmatrix}
Y_{1t} \\
Y_{2t} \\
\vdots \\
Y_{Nt}
\end{bmatrix},
\boldsymbol{u}(\boldsymbol{\alpha})_t = \begin{bmatrix}
u(\alpha_1)_{1t} \\
u(\alpha_2)_{2t} \\
\vdots \\
u(\alpha_N)_{Nt}
\end{bmatrix},
\mathbf{A}(\boldsymbol{\alpha})_p=\begin{bmatrix}
a(\alpha_1)_{11,p} & a(\alpha_1)_{12,p} & \cdots & a(\alpha_1)_{1N,p} \\
a(\alpha_2)_{21,p} & a(\alpha_2)_{22,p} & \cdots & a(\alpha_2)_{2N,p} \\
\vdots & \vdots & \ddots & \vdots \\
a(\alpha_N)_{N1,p} & a(\alpha_N)_{N2,p} & \cdots & a(\alpha_N)_{NN,p}
\end{bmatrix},
\end{gather}
$$

where $Q_{\boldsymbol{\alpha}}(\cdot)$ is **_quantile operator_**, which means that it calculates element-wise quantile of the random vector in the parenthesis according to the probability vector $\boldsymbol{\alpha}$, that is $Q_{\boldsymbol{\alpha}}(\boldsymbol{X}) = \begin{bmatrix}Q_{\alpha_1}(X_1) & Q_{\alpha_2}(X_2) & \cdots & Q_{\alpha_N}(X_N)\end{bmatrix}^{\top}$. $Q_{\boldsymbol{\alpha}}\left(\boldsymbol{u}(\boldsymbol{\alpha})_t|I_t\right)=\boldsymbol{0}$ implies that the conditional $\boldsymbol{\alpha}$-quantile of $\boldsymbol{Y}_t $ is $Q_{\boldsymbol{\alpha}}\left(\boldsymbol{Y}_t|I_t\right) = \boldsymbol{q}(\boldsymbol{\alpha}) + \sum_{p=1}^P \mathbf{A}(\boldsymbol{\alpha})_p\boldsymbol{Y}_{t-p}$. In other words, $\boldsymbol{u}(\boldsymbol{\alpha})_t$ is the forecast error that drive the observation of $\boldsymbol{y}_t$ to deviate from conditional $\boldsymbol{\alpha}$-quantile of $\boldsymbol{Y}_t$. In this sense, **$\boldsymbol{u}(\boldsymbol{\alpha})_t$ **can be named as **_quantile shock_**.

To consider the effect of common factors on $\boldsymbol{Y}_t$, we can extend the model by introducing lagged exogenous variables.

## Bayesian Estimation of QVAR

### Likelihood

#### Asymmetric Laplace Distribution

The first popular likelihood specification is to assume each element of quantile shocks obeies **asymmetric Laplace distribution** (ALD), i.e. $u(\alpha_i)_{it}\sim \mathcal{AL}(0, \alpha_i, \delta_i)$, where $\alpha_i$ is probability parameter and $\delta_i>0$ is scale parameter. According to the property of ALD, the conditional distribution of $Y_{it}$ is $\mathcal{AL}\left(q(\alpha_i)_i + \sum_{p=1}^P\sum_{j=1}^N a(\alpha_i)_{ij,p} Y_{j,t-p}, \alpha_i, \delta_i\right)$. The density function of $Y_{it}$ is

$$
\begin{equation}
f_{Y_{it}}(y_{it}) = \frac{\alpha_i(1 - \alpha_i)}{\delta_i}
\begin{cases}
\exp\left(-\dfrac{\alpha_i}{\delta_i} e(\alpha_i)_{it} \right) & \text{if } e(\alpha_i) \geq 0,  \\
\exp\left(\dfrac{1-\alpha_i}{\delta_i} e(\alpha_i)_{it} \right) & \text{if } e(\alpha_i) < 0.
\end{cases}
\end{equation}
$$

where $e(\alpha_i)_{it} = y_{it} - q(\alpha_i)_i - \sum_{p=1}^P\sum_{j=1}^N a(\alpha_i)_{ij,p} Y_{j,t-p}$. For the facility of implementation of Bayesian estimation, we use the _mixture representation property_ to write the distribtuion of $Y_{it}$ as the mixture of standard normal distribution and exponential distribution, that is

$$
\begin{equation}
Y_{it} = q(\alpha_i)_i + \sum_{p=1}^P\sum_{j=1}^N a(\alpha_i)_{ij,p} Y_{j,t-p} + \delta_i \tilde{\xi}_i W_{it} + \delta_i \tilde{\sigma}_i \sqrt{W_{it}} Z_{it},
\end{equation}
$$

where $W_{it}\sim \mathcal{EXP}(1)$, $Z_{it} \sim \mathcal{N}(0,1)$ and they are independent. $W_{it}$ is called the **_latent variable_**. The parameters $\tilde{\xi}_i$ and $\tilde{\sigma}_i$ should satisfy

$$
\begin{equation}
\tilde{\xi}_i = \frac{1 - 2 \alpha_i}{ \alpha_i (1 - \alpha_i)}, \tilde{\sigma}_i^2 = \frac{2}{\alpha_i(1 - \alpha_i)}.
\end{equation}
$$

The restriction is to ensure that $Q_{\alpha_i}(Y_{it}) = q(\alpha_i)_i + \sum_{p=1}^P\sum_{j=1}^N a(\alpha_i)_{ij,p} Y_{j,t-p}$, or equivalently, $Q_{\alpha_i}(u(\alpha_i)_{it})=0$.

#### Multivariate Asymmetric Laplace Distribution

Another feasible likelihood function setting is to assume that $\boldsymbol{Y}_t$ jointly obey **_multivariate asymmetric Laplace distribution_** (MALD), denoted as $\boldsymbol{Y}_t \sim \mathcal{MAL}\left(\boldsymbol{q}(\boldsymbol{\alpha}) + \sum_{p=1}^P \mathbf{A}(\boldsymbol{\alpha})_p \boldsymbol{y}_{t-p}, \mathbf{D}\tilde{\boldsymbol{\xi}}, \mathbf{D}\tilde{\mathbf{\Sigma}} \mathbf{D} \right)$, where $\mathbf{D}=\mathrm{diag}(\delta_1, \delta_2, \cdots, \delta_N)$ is the diagonal matrix of scale parameter, $\tilde{\boldsymbol{\xi}}=\left[\tilde{\xi}_1\ \tilde{\xi}_2\ \cdots\ \tilde{\xi}_N\right]^{\top}$ and $\tilde{\mathbf{\Sigma}}$ is a $N\times N$ positive definite matrix . The density function of $\boldsymbol{Y}_t$ is

$$
\begin{equation}
f_{\boldsymbol{Y}_t}(\boldsymbol{y}_t) = \frac{2 \exp\left( \boldsymbol{e}(\boldsymbol{\alpha})_t^{\top} \mathbf{D}^{-1} \tilde{\mathbf{\Sigma}}^{-1} \tilde{\boldsymbol{\xi}} \right) }{(2\pi)^{N/2} \left|\mathbf{D} \tilde{\mathbf{\Sigma}} \mathbf{D}\right|^{1/2} } \left(\frac{m_t}{2+d}\right)^{\nu/2} K_{\nu}\left[\sqrt{(2+d)m_t}\right],
\end{equation}
$$

where $\boldsymbol{e}(\boldsymbol{\alpha})_t = \boldsymbol{y}_{it} - \boldsymbol{q}(\boldsymbol{\alpha}) - \sum_{p=1}^P \mathbf{A}(\boldsymbol{\alpha})_p \boldsymbol{y}_{t-p}$, $m_t = \boldsymbol{e}(\boldsymbol{\alpha})^{\top} \mathbf{D}^{-1} \tilde{\mathbf{\Sigma}}^{-1} \mathbf{D}^{-1} \boldsymbol{e}(\boldsymbol{\alpha})$, $d=\tilde{\xi}^{\top} \tilde{\mathbf{\Sigma}}^{-1} \tilde{\boldsymbol{\xi}}.$ $K_{\nu}(\cdot)$ is the modified Bessel function of the third kind where $\nu = (2-N)/2$. To fix that the $\boldsymbol{\alpha}$-quantile of $\boldsymbol{u}(\boldsymbol{\alpha})_{t}$, $\tilde{\boldsymbol{\xi}}$ and the diagonals of $\tilde{\mathbf{\Sigma}}$ should satisfy

$$
\begin{equation}
\tilde{\xi}_i = \frac{1 - 2 \alpha_i}{ \alpha_i (1 - \alpha_i)}, \tilde{\sigma}_i^2 = \frac{2}{\alpha_i(1 - \alpha_i)}, i=1,2,\cdots,N.
\end{equation}
$$

Similarly, MALD also has a mixture representation, which takes the form of

$$
\begin{equation}
\boldsymbol{Y}_t = \boldsymbol{q}(\boldsymbol{\alpha}) + \sum_{p=1}^P \mathbf{A}(\boldsymbol{\alpha})_p \boldsymbol{Y}_{t-p} + \mathbf{D}\tilde{\xi}W_t + \sqrt{W_t} \mathbf{D} \tilde{\mathbf{\Sigma}}^{1
}\boldsymbol{Z}_t,
\end{equation}
$$

where $W_t \sim \mathcal{EXP}(1)$, $\mathbf{Z}_t \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I}_N)$. Compared to the case of ALD likelihood, the latent variable $W_t$ here is common for all cross-sectional units, which will reduce sampling burden of MCMC algorithm.

### Prior Distribution

#### AL-likelihood Case

Define $\boldsymbol{X}_t$$=[\boldsymbol{1}_N\ \boldsymbol{Y}_{t-1}\ \boldsymbol{Y}_{t-2}\ \cdots\ \boldsymbol{Y}_{t-P}]$, $\mathbf{B}(\boldsymbol{\alpha})=[\boldsymbol{q}(\boldsymbol{\alpha})\ \mathbf{A}(\boldsymbol{\alpha})_1\ \mathbf{A}(\boldsymbol{\alpha})_2\ \cdots\ \mathbf{A}(\boldsymbol{\alpha})_P]$ and $\boldsymbol{b}(\alpha_i)_{i\cdot}$ is the $i$-th row of the $\mathbf{B}(\boldsymbol{\alpha})$. The prior distributions of parameter in the case of ALD are

$$
\begin{align}
& \text{Prior:}
\begin{cases}
W_{it} \sim \mathcal{EXP}(1), \\
\boldsymbol{b}(\alpha_i)_{i\cdot} \sim \mathcal{N}\left(\boldsymbol{0}, \boldsymbol{\Lambda}_i \underline{\mathbf{V}}_i\right), \\
\delta_i \sim \mathcal{IG}\left(\underline{n}_{\delta,i}/2, \underline{s}_{\delta,i}/2 \right), \\
\lambda_{ij} \sim \mathcal{IG}\left(\underline{n}_{\lambda,ij}/2, \underline{s}_{\lambda,ij}/2 \right),
\end{cases}  \\
& \text{AL-Likelihood: } Y_{it} | \boldsymbol{b}(\alpha_i)_{i\cdot}, \delta_i, \tilde{\xi}_i, \tilde{\sigma}_i, \boldsymbol{x}_t, w_{it} \sim \mathcal{N}\left(\boldsymbol{x}_t^{\top} \boldsymbol{b}(\alpha_i)_{i\cdot} + \tilde{\xi}_i \delta_i w_{it},\ \delta_i^2 w_{it} \tilde{\sigma}_i^2 \right), \\
& i= 1,2,\cdots,N, j=1,2,\cdots,N, t=1,2,\cdots, T. \notag
\end{align}
$$

where $\mathbf{\Lambda}_i = \mathrm{diag}(\lambda_{i1}, \lambda_{i2}, \cdots, \lambda_{i,NP+1})$ is the diagonal matrix of penalty parameters for $\boldsymbol{b}(\alpha_i)_{i\cdot}$ The smaller is $\lambda_{ij}$, the more concentrated is $b(\alpha_i)_{ij}$ aounrd 0. $\mathcal{IG}(\alpha, \beta)$ represents Inverse Gamma distribution with shape parameter $\alpha > 0$ and scale parameter $\beta > 0$.

#### MAL-likelihood Case

In MALD case, the prior are

$$
\begin{align}
& \text{Prior: }
\begin{cases}
W_t \sim \mathcal{EXP}(1), \\
\boldsymbol{b}(\alpha_i)_{i\cdot} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{\Lambda}_i \underline{\mathbf{V}}_i), \\
\delta_i \sim \mathcal{IG}\left(\underline{n}_{\delta,i}/2, \underline{s}_{\delta,i}/2\right),\\
\tilde{\boldsymbol{\Sigma}} \sim \mathcal{IW}\left(\underline{\nu}, \underline{\mathbf{\Sigma}}\right), \\
\lambda_{ij} \sim \mathcal{IG}\left(\underline{n}_{\lambda,ij}/2, \underline{s}_{\lambda,ij}/2 \right),
\end{cases} \\
& \text{MAL-Likelihood: } \boldsymbol{Y}_t | \{\alpha_i\}_{i=1}^N, \left\{\boldsymbol{b}(\alpha_i)_{i\cdot}\right\}_{i=1}^N, \{\delta_i\}_{i=1}^N, \tilde{\mathbf{\Sigma}}, \boldsymbol{x}_t, w_t \sim \mathcal{N}\left( \mathbf{B}(\boldsymbol{\alpha})\boldsymbol{x}_t + \mathbf{D}\tilde{\boldsymbol{\xi}} w_t, w_t \mathbf{D}\tilde{\mathbf{\Sigma}} \mathbf{D} \right), \\
& i= 1,2,\cdots,N, j=1,2,\cdots,N, t=1,2,\cdots, T. \notag
\end{align}
$$

### Posterior Distribution

#### AL-likelihood Case

In the case of AL-likelihood, the posterior distribution of model parameters are

$$
\begin{align}
& W_{it} | \alpha_i, \boldsymbol{b}(\alpha_i)_{i\cdot}, \delta_i, y_{it} \sim \mathcal{GIG}\left(\frac{1}{2}, d_i + 2, m_{it}\right), \\
& \boldsymbol{b}(\alpha_i)_{i\cdot} | \alpha_i, \mathbf{\Lambda}_i, \delta_i, \{w_{it}\}_{t=1}^T, \{y_{it}\}_{t=1}^T, \{\boldsymbol{x}_t\}_{t=1}^T \sim \mathcal{N}\left(\bar{\boldsymbol{b}}_i, \bar{\mathbf{V}}_i\right), \\
& \delta_i | \alpha_i, \boldsymbol{b}(\alpha_i)_{i\cdot}, \{v_{it}\}_{t=1}^T, \{y_{it}\}_{t=1}^T, \{\boldsymbol{x}_t\}_{t=1}^T \sim \mathcal{IG}\left(\frac{\bar{n}_{\delta, i}}{2}, \frac{\bar{s}_{\delta,i}}{2}\right), \\
& \lambda_{ij}| b(\alpha)_{ij} \sim \mathcal{IG}\left(\frac{\bar{n}_{\lambda, ij}}{2}, \frac{\bar{s}_{\lambda, ij}}{2}\right), \\
& i=1,2,\cdots, N, j = 1, 2, \cdots, NP+1, t = 1,2,\cdots, T, \notag
\end{align}
$$

where $v_{it} = \delta_i w_{it}$, $\mathcal{GIG}(\alpha, \beta, n)$ represents Generalized Inverse Gaussian distribution. Other quantities in posterior distribution are defined as

$$
\begin{align}
& \bar{\mathbf{V}}_i = \left(\sum_{t=1}^T \frac{1}{w_{it} \delta_i^2 \tilde{\sigma}_i^2} \boldsymbol{x}_t \boldsymbol{x}_t^{\top} + \underline{\mathbf{V}}_i^{-1} \mathbf{\Lambda}_i^{-1}\right)^{-1}, \\
& \bar{\boldsymbol{b}}_i = \bar{\mathbf{V}}_i \left[\sum_{t=1}^T \frac{1}{w_t \delta_i^2 \tilde{\sigma}_i^2}\left(y_{it} - \delta_i\tilde{\xi}_i w_t\right) \boldsymbol{x}_t \right], \\
& \bar{n}_{\delta,i} = \underline{n}_{\delta,i} + 3T, \bar{s}_{\delta,i}=\underline{s}_{\delta,i} + 2 \sum_{t=1}^T v_{it} + \sum_{t=1}^T \frac{\left( y_{it} - \boldsymbol{x}_t^{\top} \boldsymbol{b}(\alpha_i)_{i\cdot} -\tilde{\xi} v_{it} \right)^2}{v_{it} \tilde{\sigma}_i^2}, \\
& i=1,2,\cdots, N, j = 1,2, \cdots, NP+1, t=1,2,\cdots, T. \notag
\end{align}
$$

#### MAL-likelihood Case

If you adopt MAL likelihood, then the posterior distribution of the parameters are

$$
\begin{align}
& W_t|\boldsymbol{\alpha}, \mathbf{B}(\boldsymbol{\alpha}), \tilde{\mathbf{\Sigma}}, \boldsymbol{\delta}, \boldsymbol{y}_t, \boldsymbol{x}_t \sim \mathcal{GIG}\left(\frac{2-N}{2}, d+2, m_t\right), \\
& \boldsymbol{b}(\alpha_i)_{i\cdot} | \boldsymbol{\alpha}, \boldsymbol{\Lambda}_i, \{\boldsymbol{b}(\alpha_j)_{j\cdot}\}_{j\neq i}, \tilde{\boldsymbol{\Sigma}}, \boldsymbol{\delta}, \{w_t\}_{t=1}^T, \{y_{it}\}_{t=1}^T, \{\boldsymbol{x}_t\}_{t=1}^T \sim \mathcal{N}\left(\bar{\boldsymbol{b}}_{i\cdot}, \bar{\mathbf{V}}_i \right), \\
& \tilde{\boldsymbol{\Sigma}} | \boldsymbol{\alpha}, \mathbf{B}(\boldsymbol{\alpha}), \boldsymbol{\delta}, \{w_t\}_{t=1}^T, \{\boldsymbol{y}_t\}_{t=1}^T, \{\boldsymbol{x}\}_{t=1}^T \sim \mathcal{IW}\left(\bar{\nu}, \bar{\boldsymbol{\Sigma}}\right), \\
& \delta_i | \alpha_i, \boldsymbol{b}(\alpha_i)_{i\cdot}, \{w_t\}_{t=1}^T, \{y_{it}\}_{t=1}^T, \{\boldsymbol{x}_t\}_{t=1}^T \sim \mathcal{IG}\left(\frac{\bar{n}_{\delta, i}}{2}, \frac{\bar{s}_{\delta,i}}{2}\right), \\
& \lambda_{ij} | b(\alpha_i)_{ij} \sim \mathcal{IG}\left(\frac{\bar{n}_{\lambda,ij}}{2}, \frac{\bar{s}_{\lambda,ij}}{2} \right), \\
& i = 1,2,\cdots, N, j = 1,2, \cdots, NP+1, t = 1,2,\cdots,T. \notag
\end{align}
$$

where $d=\tilde{\boldsymbol{\xi}}^{\top}\tilde{\boldsymbol{\Sigma}}^{-1}\tilde{\boldsymbol{\xi}}$, $m_t = \boldsymbol{e}(\boldsymbol{\alpha})_t^{\top} \left(\boldsymbol{\mathbf{D}}\tilde{\boldsymbol{\Sigma}} \mathbf{D} \right)^{-1} \boldsymbol{e}(\boldsymbol{\alpha})_t$, $\boldsymbol{e}(\boldsymbol{\alpha})_t = \boldsymbol{y}_t - \mathbf{B}(\boldsymbol{\alpha}) \boldsymbol{x}_t$, and

$$
\begin{align}
& \bar{\mathbf{V}}_i = \left(\sum_{t=1}^T \frac{\omega_{ii}}{w_t} \boldsymbol{x}_t \boldsymbol{x}_t^{\top} + \underline{\mathbf{V}}_i^{-1} \mathbf{\Lambda}_i^{-1}\right)^{-1}, \\
& \bar{\boldsymbol{b}}_i = \bar{\mathbf{V}}_i \left\{\sum_{t=1}^T \frac{1}{w_t}\left[\omega_{ii} \left(y_{it} - \delta_i\tilde{\xi}_i w_t\right) + \sum_{j\neq i, j =1}^N \omega_{ij} \left(e(\alpha_j)_{jt} - \delta_j \tilde{\xi}_j w_t\right) \right] \right\}, \\
& \bar{\boldsymbol{\Sigma}} = \sum_{t=1}^T \frac{1}{w_t} \mathbf{D}^{-1} \left( \boldsymbol{e}(\boldsymbol{\alpha})_t - \mathbf{D} \tilde{\boldsymbol{\xi}} w_t \right) \left( \boldsymbol{e}(\boldsymbol{\alpha})_t - \mathbf{D} \tilde{\boldsymbol{\xi}} w_t \right)^{\top} \mathbf{D}^{-1} + \underline{\mathbf{\Sigma}}, \bar{\nu} = \underline{\nu} + T - P, \\
& \bar{n}_{\delta,i} = \underline{n}_{\delta,i} + 3T, \bar{s}_{\delta,i}=\underline{s}_{\delta,i} + 2 \sum_{t=1}^T v_{it} + \sum_{t=1}^T \frac{\left( y_{it} - \boldsymbol{x}_t^{\top} \boldsymbol{b}(\alpha_i)_{i\cdot} -\tilde{\xi} v_{it} \right)^2}{v_{it} \tilde{\sigma}_i^2},
\end{align}
$$

where $v_{it} = \delta_i w_t$, $\omega_{ij}$ is the $(i,j)$ element of $\left(\mathbf{D}\tilde{\mathbf{\Sigma}} \mathbf{D}\right)^{-1}$.
