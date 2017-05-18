import unittest

import archaea.machine_learning.ridge_regression.ridge_reg_builder as ridge_builder
import sklearn.linear_model as lm


class LinearRegressionBuilderTests(unittest.TestCase):

    def linear_reg_builder_tests(self):
        linear_reg = ridge_builder.RidgeRegressionBuilder(alpha=0.1,
                                                          fit_intercept=True,
                                                          normalize=False,
                                                          copy_X=True,
                                                          max_iter=50).build()
        self.assertEqual((lm.Ridge), type(linear_reg))
