class GBDTRegressor(GBDTEstimator):

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        grad = y_pred - y_true
        hess = np.ones(len(y_true))
        return grad, hess
