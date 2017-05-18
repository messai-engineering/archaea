import unittest

import archaea.machine_learning.ridge_regression.ridge_reg_trainer as trainer
import numpy as num_py

import archaea.machine_learning.ridge_regression.ridge_reg_builder as lr_builder


class TestLinearRegressionTrainerTest(unittest.TestCase):

    def test_train_lr_function(self):
        linear_reg = lr_builder.RidgeRegressionBuilder(alpha=0.1,
                                                       fit_intercept=True,
                                                       normalize=False,
                                                       copy_X=True,
                                                       max_iter=50).build()
        f = lambda x: num_py.exp(3 * x)
        x = num_py.array([0, .1, .2, .5, .8, .9, 1])
        y = f(x) + num_py.random.randn(len(x))
        degree = 3
        x = num_py.vander(x, degree + 1)
        lr_trainer = trainer.RidgeRegressionTrainer(linear_reg)
        lr_trainer.train(x, y)
        self.assertEqual(len(lr_trainer.predict(x)), 7)

    def test_train_lr_function_multidimensional_input(self):
        linear_reg = lr_builder.RidgeRegressionBuilder(alpha=0.00001,
                                                       fit_intercept=True,
                                                       normalize=False,
                                                       copy_X=True,
                                                       max_iter=50).build()
        x = num_py.array([[0, .1, .2], [.5, .8, .9], [1, 1.2, 1.5]])
        y = [4.17197761, 30.38459717, 146.70090266]
        lr_trainer = trainer.RidgeRegressionTrainer(linear_reg)
        lr_trainer.train(x, y)
        coefficients = lr_trainer.regression_coefficients()
        self.assertEqual(len(coefficients), 3)
        self.assertEqual(len(lr_trainer.predict(x)), 3)
        self.assertEqual(lr_trainer.predict(x)[1], 30.431280870924571)
