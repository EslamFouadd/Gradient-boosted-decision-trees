class Data:

    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):
        self.values: np.array = x
        self.target: Optional[np.array] = y
        self.sorted_indexes: Optional[np.array] = None

        # sort index for each feature
        # note: necessary only for training
        sorted_indexes = []
        for feature_id in range(self.values.shape[1]):
            sorted_indexes.append(np.argsort(self.values[:, feature_id]))
        self.sorted_indexes = np.array(sorted_indexes).T
