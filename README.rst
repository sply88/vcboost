*********
vcboost
*********
Experimenting with estimation of varying coefficient models by gradient boosting.s

Given data

.. math::
    (y_i, x_i, z_i), i = 1, \dots, n

with outcome :math:`y_i \in \mathbb{R}`, covariates :math:`x_i \in \mathbb{R}^p` and effect modifiers
:math:`\mathbb{R}^p, z_i \in \mathbb{R}^q`

one assumes the varying coefficient model

.. math::
    y_i = \sum_{j=1}^p x_ij \beta_j(z_i) + \varepsilon_i = x_i^T \beta(z_i) + \varepsilon_i

[1]_. The coefficients :math:`\beta_1(\cdot), \dots, \beta_p(\cdot)` determine a (rather simple) functional
relationship between outcome :math:`y` and covariate :math:`x`. The coefficients themselves are considered to be
(potentially complex) functions of the effect modifier :math:`z`.

Each varying coefficient mapping :math:`\beta_j(\cdot)` is estimated using an ensemble of gradient
boosted decision trees. In doing so a loss function

.. math::
    L(y,\beta) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, x_i^T\beta(z_i))

is iteratively minimized. The approach used adapts the generic gradient boosting algorithm described in [2]_.

The whole idea is mainly based on  [3]_.

`DecisionTreeRegressors <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html>`_
from `sklearn <https://scikit-learn.org/stable/>`_ are used as base learners.

References
***********

.. [1] Hastie, T., & Tibshirani, R. (1993). Varying‐coefficient models. Journal of the Royal Statistical Society: Series B (Methodological), 55(4), 757-779.
.. [2] Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
.. [3] Zhou, Y., & Hooker, G. (2019). Tree boosted varying coefficient models. arXiv preprint arXiv:1904.01058
