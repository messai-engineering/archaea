import sklearn.linear_model as lm


class RidgeRegressionBuilder:

    def __init__(self, alpha=None,
                 fit_intercept=None,
                 normalize=None,
                 copy_X=None,
                 max_iter=None):
        """
            Initialize the logistic regression

            :param network_architecture: The architecture of the network
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter

    def build(self):
        """
        Ridge regression object builder

        :return:
        """
        return lm.Ridge(alpha=self.alpha, fit_intercept=True,
                        normalize=False, copy_X=True, max_iter=self.max_iter)
