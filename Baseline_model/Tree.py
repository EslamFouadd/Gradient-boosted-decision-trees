class Tree:

    def __init__(self, params: dict):
        self.params: dict = params
        self.nodes: List[Node] = []

        # parameters
        self.max_depth: int = params.get("max_depth")

        # add initial node
        node = Node(0, 0.0)
        self.nodes.append(node)

    def construct(self, data: Data, grad: np.ndarray, hess: np.ndarray):

        # data
        assert (data.sorted_indexes is not None)
        n = len(data.values)
        values = data.values
        sorted_indexes = data.sorted_indexes

        # node ids records belong to
        node_ids_data = np.zeros(n, dtype=int)

        # [comment with [] is important for understanding]
        # [for each depth]
        for depth in range(self.max_depth):

            # node ids in the depth
            node_ids_depth = TreeUtil.node_ids_depth(depth)

            # [1. find best split] -------------------

            # split information for each node
            feature_ids, feature_values = [], []
            left_weights, right_weights = [], []

            # [for each node]
            for node_id in node_ids_depth:

                node = self.nodes[node_id]
                # logger.debug(f"{node_id}: find split -----")

                # [calculate sum grad and hess of records in the node]
                sum_grad, sum_hess = 0.0, 0.0
                for i in range(n):
                    if node_ids_data[i] != node_id:
                        continue
                    sum_grad += grad[i]
                    sum_hess += hess[i]

                # [initialize best gain (set as all directed to left)]
                best_gain, best_feature_id, best_feature_value = 0.0, 0, -np.inf
                best_left_weight, best_right_weight = node.weight, 0.0

                if sum_hess > 0:
                    sum_loss = TreeUtil.loss(sum_grad, sum_hess)
                else:
                    sum_loss = 0.0

                # logger.debug(f"sum grad:{sum_grad} hess:{sum_hess} loss:{sum_loss}")

                # [for each feature]
                for feature_id in range(data.values.shape[1]):
                    prev_value = -np.inf
                    
                    # [have gradients/hessian of left child records(value of record < value of split)]
                    left_grad, left_hess = 0.0, 0.0

                    sorted_index = sorted_indexes[:, feature_id]

                    # [for each sorted record]
                    for i in sorted_index:
                        # skip if the record does not belong to the node
                        # NOTE: this calculation is redundant and inefficient.
                        if node_ids_data[i] != node_id:
                            continue

                        value = values[i, feature_id]

                        # [evaluate split, if split can be made at the record]
                        if value != prev_value and left_hess > 0 and (sum_hess - left_hess) > 0:
                            
                            # [calculate loss of the split using gradient and hessian]
                            right_grad = sum_grad - left_grad
                            right_hess = sum_hess - left_hess
                            left_loss = TreeUtil.loss(left_grad, left_hess)
                            right_loss = TreeUtil.loss(right_grad, right_hess)
                            if left_loss is not None and right_loss is not None:
                                gain = sum_loss - (left_loss + right_loss)
                                # logger.debug(f"'feature{feature_id} < {value}' " +
                                #       f"lg:{left_grad:.3f} lh:{left_hess:.3f} rg:{right_grad:.3f} rh:{right_hess:.3f} " +
                                #       f"ll:{left_loss:.3f} rl:{right_loss:.3f} gain:{gain:.3f}")
                                
                                # [update if the gain is better than current best gain]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_feature_id = feature_id
                                    best_feature_value = value
                                    best_left_weight = TreeUtil.weight(left_grad, left_hess)
                                    best_right_weight = TreeUtil.weight(right_grad, right_hess)

                        prev_value = value
                        left_grad += grad[i]
                        left_hess += hess[i]

                # logger.debug(f"node_id:{node_id} split - 'feature{best_feature_id} < {best_feature_value}'")
                feature_ids.append(best_feature_id)
                feature_values.append(best_feature_value)
                left_weights.append(best_left_weight)
                right_weights.append(best_right_weight)

            # [2. update nodes and create new nodes] ----------
            for i in range(len(node_ids_depth)):
                node_id = node_ids_depth[i]
                feature_id = feature_ids[i]
                feature_value = feature_values[i]
                left_weight = left_weights[i]
                right_weight = right_weights[i]

                # update current node
                node = self.nodes[node_id]
                node.feature_id = feature_id
                node.feature_value = feature_value

                # create new nodes
                left_node = Node(TreeUtil.left_child_id(node_id), left_weight)
                right_node = Node(TreeUtil.right_child_id(node_id), right_weight)
                self.nodes += [left_node, right_node]

            # [3. update node ids of records] ----------
            for i in range(len(node_ids_data)):
                # directed by split
                node_id = node_ids_data[i]
                node = self.nodes[node_id]
                feature_id, feature_value = node.feature_id, node.feature_value

                # update
                is_left = values[i, feature_id] < feature_value
                if is_left:
                    next_node_id = TreeUtil.left_child_id(node_id)
                else:
                    next_node_id = TreeUtil.right_child_id(node_id)
                node_ids_data[i] = next_node_id

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = x

        # node ids records belong to
        node_ids_data = np.zeros(len(values), dtype=int)

        for depth in range(self.max_depth):
            for i in range(len(node_ids_data)):
                # directed by split
                node_id = node_ids_data[i]
                node = self.nodes[node_id]
                feature_id, feature_value = node.feature_id, node.feature_value

                # update
                if feature_id is None:
                    next_node_id = node_id
                elif values[i, feature_id] < feature_value:
                    next_node_id = TreeUtil.left_child_id(node_id)
                else:
                    next_node_id = TreeUtil.right_child_id(node_id)
                node_ids_data[i] = next_node_id

        weights = np.array([self.nodes[node_id].weight for node_id in node_ids_data])

        return weights

    def dump(self) -> str:
        """dump tree information"""
        ret = []
        for depth in range(self.max_depth + 1):
            node_ids_depth = TreeUtil.node_ids_depth(depth)
            for node_id in node_ids_depth:
                node = self.nodes[node_id]
                if node.is_leaf():
                    ret.append(f"{node_id}:leaf={node.weight}")
                else:
                    ret.append(
                        f"{node_id}:[f{node.feature_id}<{node.feature_value}] " +
                        f"yes={TreeUtil.left_child_id(node_id)},no={TreeUtil.right_child_id(node_id)}")
        return "\n".join(ret)
