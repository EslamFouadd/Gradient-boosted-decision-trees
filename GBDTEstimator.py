class GBDTEstimator:

    def __init__(self, params: dict):
        self.params: dict = params
        self.trees: List[Tree] = []

        # parameters
        self.n_round: int = params.get("n_round")
        self.eta: float = params.get("eta")

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        data = Data(x, y)
        self._fit(data)

    def _fit(self, data: Data):
        pred = np.zeros(len(data.values))
        for round in range(self.n_round):
            logger.info(f"construct tree[{round}] --------------------")
            grad, hess = self.calc_grad(data.target, pred)
            tree = Tree(self.params)
            tree.construct(data, grad, hess)
            self.trees.append(tree)
            # NOTE: predict only last tree
            pred += self._predict_last_tree(data)

    def predict(self, x: np.ndarray) -> np.ndarray:
        data = Data(x, None)
        return self._predict(data)

    def _predict(self, data: Data) -> np.ndarray:
        pred = np.zeros(len(data.values))
        for tree in self.trees:
            pred += tree.predict(data.values) * self.eta
        return pred

    def _predict_last_tree(self, data: Data) -> np.ndarray:
        assert(len(self.trees) > 0)
        tree = self.trees[-1]
        return tree.predict(data.values) * self.eta

    def dump_model(self) -> str:
        ret = []
        for i, tree in enumerate(self.trees):
            ret.append(f"booster[{i}]")
            ret.append(tree.dump())
        return "\n".join(ret)
