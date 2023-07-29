class GBDTClassifier(GBDTEstimator):

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        # (reference) regression_loss.h
        y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
        eps = 1e-16
        grad = y_pred_prob - y_true
        hess = np.maximum(y_pred_prob * (1.0 - y_pred_prob), eps)
        return grad, hess

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # apply sigmoid
        return 1.0 / (1.0 + np.exp(-self.predict(x)))
