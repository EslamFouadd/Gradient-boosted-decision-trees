class Node:

    def __init__(self, id: int, weight: float):
        self.id: int = id

        # note: necessary only for leaf node
        self.weight: float = weight

        # split information
        self.feature_id: int = None
        self.feature_value: float = None

    def is_leaf(self) -> bool:
        return self.feature_id is None
