from __future__ import annotations

import itertools
from queue import PriorityQueue
from typing import Callable, Optional

import numpy as np

from transformers import PreTrainedTokenizer


class Node:
    token_ids: list[int]
    parent: Optional[Node] = None
    children: list[Node] = []
    value: float = 0.0
    count: int = 0

    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids

    def __lt__(self, other: Node) -> bool:
        return len(self.token_ids) > len(other.token_ids)

    def add_child(self, node: Node):
        node.parent = self
        self.children.append(node)

    def remove_child(self, node: Node):
        node.parent = None
        self.children.remove(node)

    def branch(self) -> tuple[Node, Node]:
        start = int(0.25 * len(self.token_ids))
        end = int(0.75 * len(self.token_ids))
        idx = np.random.randint(start, end)
        node = Node(self.token_ids[:idx])
        self.parent.add_child(node)
        self.parent.remove_child(self)
        self.token_ids = self.token_ids[idx:]
        node.add_child(self)
        return node, self


class Tree:
    root: Node
    node: Node
    leaf_nodes: list[Node] = []
    queue: PriorityQueue[Node] = PriorityQueue()

    def __init__(self, token_ids: list[int]):
        self.root = Node(token_ids)

    def agg_token_ids(self, node: Node, skip_root: bool = False) -> list[int]:
        token_ids_list = []
        while node is not None:
            if skip_root and node == self.root:
                break
            token_ids_list.append(node.token_ids)
            node = node.parent
        token_ids = list(itertools.chain.from_iterable(reversed(token_ids_list)))
        return token_ids

    def branch(self) -> list[int]:
        if self.queue.empty():
            self.node = self.root
        else:
            node = self.queue.get()
            self.node, node = node.branch()
            self.queue.put(self.node)
            self.queue.put(node)
        token_ids = self.agg_token_ids(self.node)
        return token_ids

    def update(self, token_ids: list[int]):
        node = Node(token_ids)
        self.node.add_child(node)
        self.leaf_nodes.append(node)
        self.queue.put(node)

    def compute_reward(
        self, data_source: str, ground_truth: str, tokenizer: PreTrainedTokenizer, compute_score: Callable[..., float]
    ):
        for node in self.leaf_nodes:
            response_ids = self.agg_token_ids(node, skip_root=True)
            response_str = tokenizer.decode(response_ids, skip_special_tokens=True)
            score = compute_score(data_source, response_str, ground_truth)
            node.value = score
            node.count = 1

    def compute_value(self):
        pass

    def compute_advantage(self):
        pass


class BatchTree:
    trees: list[Tree]

    def __init__(self, token_ids_list: list[list[int]]):
        self.trees = [Tree(token_ids) for token_ids in token_ids_list]

    def branch(self) -> list[list[int]]:
        token_ids_list = []
        for tree in self.trees:
            token_ids = tree.branch()
            token_ids_list.append(token_ids)
        return token_ids_list

    def update(self, token_ids_list: list[list[int]]):
        for tree, token_ids in zip(self.trees, token_ids_list, strict=True):
            tree.update(token_ids)

    def compute_reward(
        self,
        data_source_list: list[str],
        ground_truth_list: list[str],
        tokenizer: PreTrainedTokenizer,
        compute_score: Callable[..., float],
    ):
        for tree, data_source, ground_truth in zip(self.trees, data_source_list, ground_truth_list, strict=True):
            tree.compute_reward(data_source, ground_truth, tokenizer, compute_score)

    def compute_value(self):
        for tree in self.trees:
            tree.compute_value()

    def compute_advantage(self):
        for tree in self.trees:
            tree.compute_advantage()
