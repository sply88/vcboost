# vcboost

Experimenting with estimation of varying coefficient models by gradient boosting.

The experimental implementation adapts the generic gradient boosting algorithm described in (Friedman 2001) and
can be found in [vcboost/boost.py](vcboost/boost.py).

The whole idea is mainly based on (Zhou 2019).

[sklearn decision trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) are used as base learners.

The [examples](examples) folder contains a few short notebooks.

- The [simple_interaction notebook](examples/simple_interaction.ipynb) illustrates the most basic case.
- In the [robust_loss notebook](examples/robust_loss.ipynb) the two currently available loss functions are compared
for the simple interaction example with added outliers.
- The [special_cases notebook](examples/special_cases.ipynb) shows how the algorithm relates to its special cases
*gradient boosting* and *ordinary least squares*.
- In the [wang_hastie notebook](examples/wang_hastie.ipynb) the model is applied to an example problem from the literature

## Background

Some details on the method are described in [background.pdf](background/background.pdf).


## References
***********

- Hastie, T., & Tibshirani, R. (1993). Varying‐coefficient models. Journal of the Royal Statistical Society: Series B (Methodological), 55(4), 757-779.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
- Zhou, Y., & Hooker, G. (2019). Tree boosted varying coefficient models. arXiv preprint arXiv:1904.01058

