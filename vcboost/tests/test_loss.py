import pytest
import numpy as np

from vcboost.loss import LS, LAD


class TestLS:

    @pytest.mark.parametrize('y, y_hat, expected', [
        (np.zeros(1), np.zeros(1), 0),
        (np.zeros(1), np.ones(1), 1),
        (np.ones(1), np.zeros(1), 1),
        (-np.ones(1), np.zeros(1), 1),
        (np.ones(4), np.ones(4), 0),
        (np.ones(4), np.array([0, 1, 1, 2]), 0.5)
    ])
    def test__call__(self, y, y_hat, expected):

        ls = LS()

        assert ls(y, y_hat) == expected

    @pytest.mark.parametrize('y, y_hat, expected', [
        (np.zeros(1), np.zeros(1), np.array([0])),
        (np.zeros(1), np.ones(1), np.array([-1])),
        (np.ones(1), np.zeros(1), np.array([1])),
        (-np.ones(1), np.zeros(1), np.array([-1])),
        (np.ones(4), np.ones(4), np.zeros(4)),
        (np.ones(4), np.array([0, 1, 1, 2]), np.array([1, 0, 0, -1]))
    ])
    def test_negative_gradient(self, y, y_hat, expected):

        ls = LS()

        assert (ls.negative_gradient(y, y_hat) == expected).all()

    @pytest.mark.parametrize('residual, direction, expected', [
        (np.ones(1), np.zeros(1), 0),
        (np.ones(1), np.ones(1)*1e-8, 0),
        (np.ones(5), np.zeros(5), 0),
        (np.ones(5), np.ones(5)*1e-8, 0),
        (np.ones(1), np.ones(1), 1),
        (np.ones(4), np.ones(4), 1),
        (2*np.ones(4), np.ones(4), 2),
        (10*np.arange(5), np.arange(5), 10)
    ])
    def test_line_search(self, residual, direction, expected):

        ls = LS()

        assert ls.line_search(residual, direction) == expected


class TestLAD:

    @pytest.mark.parametrize('y, y_hat, expected', [
        (np.zeros(1), np.zeros(1), 0),
        (np.zeros(1), np.ones(1), 1),
        (np.ones(1), np.zeros(1), 1),
        (-np.ones(1), np.zeros(1), 1),
        (np.ones(4), np.ones(4), 0),
        (np.ones(4), np.array([1, 2, 3, 4]), 1.5),
        (np.ones(5), np.array([-5, -4, -3, -2, -1]), 4)
    ])
    def test__call__(self, y, y_hat, expected):

        lad = LAD()

        assert lad(y, y_hat) == expected

    @pytest.mark.parametrize('y, y_hat, expected', [
        (np.zeros(1), np.zeros(1), np.array([0])),
        (np.zeros(1), np.ones(1), np.array([-1])),
        (np.ones(1), np.zeros(1), np.array([1])),
        (-np.ones(1), np.zeros(1), np.array([-1])),
        (10*np.ones(1), np.ones(1), np.array([1])),
        (np.ones(1), 10*np.ones(1), np.array([-1])),
        (np.ones(4), np.array([-5, 4, -20, 7]), np.array([1, -1, 1, -1])),
        (np.array([12, -3.5, -9, 1.1]), -np.ones(4), np.array([1, -1, -1, 1]))
    ])
    def test_negative_gradient(self, y, y_hat, expected):

        lad = LAD()

        assert (lad.negative_gradient(y, y_hat) == expected).all()

    @pytest.mark.parametrize('residual, direction, expected', [
        (np.ones(1), np.zeros(1), 0),
        (np.ones(1), np.ones(1)*1e-9, 0),
        (np.ones(5), np.zeros(5), 0),
        (np.ones(5), np.ones(5) * 1e-9, 0),
        (np.ones(1), np.ones(1), 1),
        (np.ones(4), np.ones(4), 1),
        (np.array([16, 16, 16]), np.array([2, 4, 8]), 2)

    ])
    def test_line_search(self, residual, direction, expected):

        lad = LAD()

        assert lad.line_search(residual, direction) == expected
