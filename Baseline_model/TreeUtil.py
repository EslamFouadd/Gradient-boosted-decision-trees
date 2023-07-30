class TreeUtil:

    @classmethod
    def left_child_id(cls, id: int) -> int:
        """node id of left child"""
        return id * 2 + 1

    @classmethod
    def right_child_id(cls, id: int) -> int:
        """node id of right child"""
        return id * 2 + 2

    @classmethod
    def loss(cls, sum_grad: float, sum_hess: float) -> Optional[float]:
        if np.isclose(sum_hess, 0.0, atol=1.e-8):
            return None
        return -0.5 * (sum_grad ** 2.0) / sum_hess

    @classmethod
    def weight(cls, sum_grad: float, sum_hess: float) -> Optional[float]:
        if np.isclose(sum_hess, 0.0, atol=1.e-8):
            return None
        return -1.0 * sum_grad / sum_hess

    @classmethod
    def node_ids_depth(self, d: int) -> List[int]:
        return list(range(2 ** d - 1, 2 ** (d + 1) - 1))
