import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF

from .loss import LS


class VCBooster:
    """Uses ensembles of sklearn.tree.DecisionTreeRegressor to estimate boosted varying coefficient models.

    Parameters
    ----------
    n_stages : int
        Number of boosting stages
    learning_rate : float
        Learning rate
    max_depth : int or None, default=3
        Max depth of base learners. Passed to same parameter of DecisionTreeRegressor.
    min_samples_leaf : int or float, default=0.05
        Min size of leaves. Total number of observations if int is passed, fraction of observations if float is passed.
        Passed to same parameter of DecisionTreeRegressor.
    splitter : {'best', 'random'}, default='best'
        Splitting strategy for internal nodes. Passed to same parameter of DecisionTreeRegressor.
    line_search_strategy : {'global', 'leaf'}, default='global'
        How to do the line search after gradient approximation. Either 'global' to do a single line search for all
        observations (Friedman 2001, Algorithm 1, Line 5) or 'leaf' to do line search for each leaf
        (Friedman 2001, Eq. 18)
    mini_updates : bool, default=False
        If True current prediction y_hat, residuals and negative gradient will be recomputed after a single coordinate
        has been updated. Otherwise recompute only after current boosting stage has been run for all coordinates
    loss : loss.LossFunction
        Instance of a loss function that can be called and has methods negative_gradient, and line_search
    verbose : int
        If not 0 print progress during fit

    Attributes
    ----------
    ensembles : list of lists
        Coefficient models learned, ensembles[j][s] contains base learner s for coefficient j
    ncol_X, ncol_Z : int
        Number of parametric coefficients and effect modifiers. Used to check input dimensions after fitting.
    train_loss, validation_loss : list
        Training and (optionally) validation loss for all stages
    """

    def __init__(self, n_stages=100, learning_rate=0.1, max_depth=3, min_samples_leaf=0.05, splitter='best',
                 line_search_strategy='global', mini_updates=False, loss=LS(), verbose=0):

        self.n_stages = n_stages
        self.learning_rate = learning_rate

        # parameters that will be passed to sklearn.tree.DecisionTreeRegressor
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter

        # parameters controlling update strategy
        self.line_search_strategy = line_search_strategy
        self.mini_updates = mini_updates

        # loss function used
        self.loss = loss

        # more attributes set during fit
        self.ensembles = None
        self.ncol_X = None
        self.ncol_Z = None

        self.train_loss = None
        self.validation_loss = None

        self.verbose = verbose

    @property
    def fitted(self):
        return self.ensembles is not None

    @staticmethod
    def _validate_input(X, Z, y=None):

        if not X.ndim == 2:
            raise ValueError('Expected 2D Array for X but got {}D'.format(X.ndim))

        if not Z.ndim == 2:
            raise ValueError('Expected 2D Array for Z but got {}D'.format(Z.ndim))

        if not Z.shape[0] == X.shape[0]:
            raise ValueError('Number of observations (rows) mismatch in X and Z.')

        if y is not None:

            if not y.ndim == 1:
                ValueError('Expected 1D array for y but got {}D'.format(y.ndim))
            elif not y.shape[0] == X.shape[0]:
                raise ValueError('Number of observations mismatch in y and X')
            else:
                return X.astype(np.float32), Z.astype(np.float32), y. astype(np.float32)

        else:
            return X.astype(np.float32), Z.astype(np.float32)

    def _set_dimensions(self, X, Z):

        self.ncol_X = X.shape[1]
        self.ncol_Z = Z.shape[1]

        return self.ncol_X, self.ncol_Z

    def _validate_dimensions(self, X, Z):

        if not X.shape[1] == self.ncol_X:
            raise ValueError('Expected X to have {} columns but got {} instead.'.format(self.ncol_X, X.shape[1]))

        if not Z.shape[1] == self.ncol_Z:
            raise ValueError('Expected Z to have {} columns but got {} instead.'.format(self.ncol_Z, Z.shape[1]))

        return self.ncol_X, self.ncol_Z

    def fit(self, X, Z, y, validation_data=None):
        """Fit a varying coefficient model

        Parameters
        ----------
        X : numpy.array with shape (n_samples, n_coefficients)
            Design matrix for parametric coefficients.
        Z : numpy.array with shape (n_samples, n_effect_modifiers)
            Design matrix of effect modifiers
        y : numpy.array with shape (n_samples,)
            Dependent variable
        validation_data : None or tuple, default=None
            If not None pass tuple (X_val, Z_val, y_val) with validation data

        Returns
        -------
        self : VCBooster
            The fitted model
        """

        rng = np.random.default_rng()  # TODO: Random state can not be controlled

        X, Z, y = self._validate_input(X, Z, y)
        p, q = self._set_dimensions(X, Z)

        self.ensembles = [[] for j in range(p)]
        self.train_loss = []

        # matrix to hold coefficients for each observation
        B = np.zeros(X.shape)

        # check and process validation data
        if validation_data is not None:
            X_val, Z_val, y_val = self._validate_input(validation_data[0], validation_data[1], validation_data[2])
            B_val = np.zeros(X_val.shape)
            self.validation_loss = []
        else:
            B_val = None

        # update order will be permuted each stage. Only matters if self.mini_updates is True
        update_order = np.arange(p)

        # initial prediction, residual and negative gradient
        y_hat = (X * B).sum(axis=1)
        residual = y - y_hat
        ng = self.loss.negative_gradient(y, y_hat)

        # training loop
        for i in range(self.n_stages):

            if self.verbose:
                print('Stage {} / {}:'.format(i+1, self.n_stages))

            # keep track of loss for training data and validation data
            self.train_loss.append(self.loss(y, y_hat))

            if validation_data is not None:
                y_hat_val = (X_val * B_val).sum(axis=1)
                self.validation_loss.append(self.loss(y_val, y_hat_val))

            # varying coefficient function for each coordinate is updated
            rng.shuffle(update_order)
            for update_idx in range(p):

                # update current prediction, residual and negative gradient
                if update_idx == 0 or self.mini_updates:
                    y_hat = (X * B).sum(axis=1)
                    residual = y - y_hat
                    ng = self.loss.negative_gradient(y, y_hat)

                # current coordinate
                j = update_order[update_idx]

                # negative gradient of loss function for current coordinate
                ng_j = ng * X[:, j]

                # approximate negative gradient using least squares regression tree
                tree = DecisionTreeRegressor(criterion='mse', max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf, splitter=self.splitter)
                tree.fit(X=Z, y=ng_j, check_input=False)

                # adjust leaf values based on self.line_search_strategy
                self._finalize_tree(tree, X[:, j], Z, residual)

                # add tree to ensemble of current coordinate
                self.ensembles[j].append(tree)

                # update coefficients of current coordinate
                B[:, j] += self.learning_rate * tree.predict(Z, check_input=False)

                if validation_data is not None:
                    B_val[:, j] += self.learning_rate * tree.predict(Z_val, check_input=False)

        return self

    def _finalize_tree(self, tree, x_j, Z, residual):

        # leaf indices in array representation
        leaves = np.where(tree.tree_.children_left == TREE_LEAF)[0]

        if self.line_search_strategy == 'leaf':

            # get terminal node indices for all observations
            terminal_nodes = tree.apply(Z, check_input=False)

            # line search for each leaf (Friedman (2001) Eq. (18) but direction depends on values of covariate j)
            for leaf in leaves:
                tree.tree_.value[leaf, 0, 0] = self.loss.line_search(residual=residual[terminal_nodes == leaf],
                                                                     direction=x_j[terminal_nodes == leaf])

        elif self.line_search_strategy == 'global':

            ng_hat = tree.predict(Z, check_input=False)
            step_size = self.loss.line_search(residual=residual, direction=x_j*ng_hat)

            tree.tree_.value[leaves, 0, 0] *= step_size

        else:
            raise AttributeError('Unknown line_search_strategy {}.'.format(self.line_search_strategy))

    def predict_coefficients_at_stage(self, X, Z, stage=None):
        """Predict varying coefficients

        Parameters
        ----------
        X : numpy.array with shape (n_samples, n_coefficients)
            Design matrix for parametric coefficients.
        Z : numpy.array with shape (n_samples, n_effect_modifiers)
            Design matrix of effect modifiers
        stage : int or None, default=None
            Number of stages / base learners to use for prediction. If None use all.
        Returns
        -------
        B : numpy.array
            Predicted matrix with parametric coefficients for each observation
        """

        if not self.fitted:
            raise Exception('Model not fitted.')

        if stage is None:
            stage = self.n_stages
        elif stage > self.n_stages:
            raise ValueError('Only {} stages fitted but {} passed.'.format(self.n_stages, stage))

        X, Z = self._validate_input(X, Z)
        p, q = self._validate_dimensions(X, Z)

        B = np.zeros(X.shape)

        for j in range(p):
            for s in range(stage):
                tree = self.ensembles[j][s]
                B[:, j] += self.learning_rate * tree.predict(X=Z, check_input=False)

        return B

    def predict_stage(self, X, Z, stage):
        """Predict outcome for a specific stage

        Parameters
        ----------
        X : numpy.array with shape (n_samples, n_coefficients)
            Design matrix for parametric coefficients.
        Z : numpy.array with shape (n_samples, n_effect_modifiers)
            Design matrix of effect modifiers
        stage : int or None, default=None
            Number of stages / base learners to use for prediction. If None use all.
        Returns
        -------
        numpy.array
            Predicted outcome for each observation at passed stage
        """

        B = self.predict_coefficients_at_stage(X, Z, stage)

        return (X * B).sum(axis=1)

    def predict(self, X, Z):
        """Predict outcome

        Parameters
        ----------
        X : numpy.array with shape (n_samples, n_coefficients)
            Design matrix for parametric coefficients.
        Z : numpy.array with shape (n_samples, n_effect_modifiers)
            Design matrix of effect modifiers

        Returns
        -------
        numpy.array
            Predicted outcome for each observation
        """
        return self.predict_stage(X, Z, stage=self.n_stages)
