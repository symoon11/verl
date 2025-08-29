# Copyright 2025 SNU MLLAB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from verl.protocol import DataProto, DataProtoItem, collate_fn
from verl.workers.reward_manager.abstract import AbstractRewardManager

THINK_TOKEN_ID = 151668


@dataclass
class Node:
    token_ids: list[int]
    parent: Optional[Node] = None
    children: list[Node] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    advantage: float = 0.0

    def __len__(self) -> int:
        try:
            return len(self.token_ids) - self.token_ids[::-1].index(THINK_TOKEN_ID)
        except ValueError:
            return len(self.token_ids)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, node: Node):
        node.parent = self
        self.children.append(node)

    def remove_child(self, node: Node):
        node.parent = None
        self.children.remove(node)

    def branch(self) -> Node:
        idx = random.randint(0, len(self) - 1)
        if idx == 0:
            return self.parent
        else:
            node = Node(self.token_ids[:idx])
            self.token_ids = self.token_ids[idx:]
            self.parent.add_child(node)
            self.parent.remove_child(self)
            node.add_child(self)
            return node


@dataclass
class Tree:
    data_item: DataProtoItem
    root_node: Node = field(init=False)
    curr_node: Node = field(init=False)
    leaf_nodes: list[Node] = field(default_factory=list)

    def __post_init__(self):
        prompt_ids = self.data_item.non_tensor_batch["raw_prompt_ids"]
        self.root_node = Node(prompt_ids)
        self.curr_node = self.root_node

    def _get_ancestor_nodes(self, node: Node, skip_root_node: bool = False) -> list[Node]:
        nodes = []
        while node is not None and not (skip_root_node and node.is_root):
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _agg_token_ids(self, node: Node, skip_root_node: bool = False) -> list[int]:
        token_ids = []
        nodes = self._get_ancestor_nodes(node, skip_root_node=skip_root_node)
        for node in nodes:
            token_ids.extend(node.token_ids)
        return token_ids

    def _get_non_root_nodes(self, remove_duplicates: bool = False) -> list[Node]:
        nodes = []
        for node in self.leaf_nodes:
            nodes.extend(self._get_ancestor_nodes(node, skip_root_node=True))
        nodes = list(set(nodes)) if remove_duplicates else nodes
        return nodes

    def _branch(self) -> Node:
        nodes = self._get_non_root_nodes()
        lengths = [len(node) for node in nodes]
        node = random.choices(nodes, weights=lengths)[0]
        node = node.branch()
        return node

    def _compute_reward(self, reward_fn: AbstractRewardManager) -> float:
        data_source = self.data_item.non_tensor_batch["data_source"]
        ground_truth = self.data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = self.data_item.non_tensor_batch["extra_info"]
        scores = []
        for node in self.leaf_nodes:
            response_ids = self._agg_token_ids(node, skip_root_node=True)
            response_str = reward_fn.tokenizer.decode(response_ids, skip_special_tokens=True)
            score = reward_fn.compute_score(data_source, response_str, ground_truth, extra_info)
            node.values.append(score)
            scores.append(score)
        return np.mean(scores)

    def _compute_value(self, node: Optional[Node] = None):
        node = node or self.root_node
        for child in node.children:
            self._compute_value(node=child)
            node.values.extend(child.values)

    def _compute_traj_advantage(self, node: Node) -> float:
        return np.mean(node.values) - np.mean(self.root_node.values)

    def _compute_step_advantage(self, node: Node) -> float:
        return np.mean(node.values) - np.mean(node.parent.values)

    def _compute_advantage(self, node: Optional[Node] = None, weight: float = 1.0):
        node = node or self.root_node
        if not node.is_root:
            traj_advantage = self._compute_traj_advantage(node)
            step_advantage = self._compute_step_advantage(node)
            node.advantage = traj_advantage + weight * step_advantage
        for child in node.children:
            self._compute_advantage(node=child, weight=weight)

    def get_prompt(self) -> DataProtoItem:
        self.curr_node = self._branch() if self.leaf_nodes else self.root_node
        prompt_ids = self._agg_token_ids(self.curr_node)
        prompt = copy.deepcopy(self.data_item)
        prompt.non_tensor_batch["raw_prompt_ids"] = prompt_ids
        return prompt

    def update_output(self, output: DataProtoItem):
        response = output.batch["responses"]
        response_length = response.size(0)
        attention_mask = output.batch["attention_mask"]
        response_mask = attention_mask[-response_length:].bool()
        response_ids = torch.masked_select(response, response_mask).tolist()
        node = Node(response_ids)
        self.curr_node.add_child(node)
        self.leaf_nodes.append(node)

    def postprocess(self, reward_fn: AbstractRewardManager) -> float:
        score = self._compute_reward(reward_fn)
        return score


@dataclass
class BatchTree:
    data: DataProto
    trees: list[Tree] = field(default_factory=list)

    def __post_init__(self):
        for data_item in self.data:
            self.trees.append(Tree(data_item))

    def get_prompts(self) -> DataProto:
        prompts = [tree.get_prompt() for tree in self.trees]
        prompts = collate_fn(prompts)
        return prompts

    def update_outputs(self, outputs: DataProto):
        for tree, output in zip(self.trees, outputs, strict=True):
            tree.update_output(output)

    def postprocess(self, reward_fn: AbstractRewardManager):
        scores = []
        for tree in self.trees:
            score = tree.postprocess(reward_fn)
            scores.append(score)
        print(np.mean(scores))
