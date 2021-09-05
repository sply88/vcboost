# vcboost

Experimenting with estimation of varying coefficient models by gradient boosting.

The experimental implementation adapts the generic gradient boosting algorithm described in (Friedman 2001) and
can be found in [vcboost/boost.py](vcboost/boost.py).

The whole idea is mainly based on (Zhou 2019).

[sklearn decision trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) are used as base learners.

The [examples](examples) folder contains a few short scripts / notebooks.

- The [simple_interaction notebook](examples/simple_interaction.ipynb) illustrates the most basic case.
- In the [robust_loss notebook](examples/robust_loss.ipynb) the two currently available loss functions are compared
for the simple interaction example with added outliers.
- The [special_cases notebook](examples/special_cases.ipynb) shows how the algorithm relates to its special cases
*gradient boosting* and *ordinary least squares*.

## Background

**On GitHub see the [background notebook](background.ipynb) for this section, as I haven't figured out how to render equations
in the readme, yet.**

Given data $(y_i, x_i, z_i), i = 1, \dots, n$ with outcome $y_i \in \mathbb{R}$, covariates $x_i \in \mathbb{R}^p$ and effect modifiers $z_i \in \mathbb{R}^q$ one assumes the varying coefficient model

$$
y_i = \sum_{j=1}^p x_{ij} \beta_j(z_i) + \varepsilon_i = x_i^T \beta(z_i) + \varepsilon_i
$$

(Hastie 1993). The coefficients $\beta_1(\cdot), \dots, \beta_p(\cdot)$ determine a (rather simple) functional
relationship between outcome $y$ and covariate $x$. The coefficients themselves are considered to be
(potentially complex) functions of the effect modifier $z$.

Each varying coefficient mapping $\beta_j(\cdot)$ is estimated using an ensemble of gradient
boosted decision trees. In doing so a loss function

$$
    L(y,\beta) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, x_i^T\beta(z_i))
$$

is iteratively minimized.

## References
***********

- Hastie, T., & Tibshirani, R. (1993). Varying‐coefficient models. Journal of the Royal Statistical Society: Series B (Methodological), 55(4), 757-779.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
- Zhou, Y., & Hooker, G. (2019). Tree boosted varying coefficient models. arXiv preprint arXiv:1904.01058
