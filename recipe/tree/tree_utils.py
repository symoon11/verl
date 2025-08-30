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
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

THINK_TOKEN_ID = 151668


@dataclass
class Node:
    token_ids: list[int]
    parent: Optional[Node] = None
    children: list[Node] = field(default_factory=list)
    reward: float = 0.0
    count: int = 0
    value: float = 0.0
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
        return not self.children

    @property
    def rewards(self) -> list[float]:
        return [0.0 for _ in range(len(self.token_ids) - 1)] + [self.reward]

    @property
    def advantages(self) -> list[float]:
        return [self.advantage for _ in range(len(self.token_ids))]

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
    data: DataProto
    root_node: Node = field(init=False)
    curr_node: Node = field(init=False)
    leaf_nodes: list[Node] = field(default_factory=list)

    def __post_init__(self):
        token_ids = self.data.non_tensor_batch["raw_prompt_ids"][0]
        self.root_node = Node(token_ids)
        self.curr_node = self.root_node

    def _get_ancestor_nodes(self, node: Node, skip_root_node: bool = True) -> list[Node]:
        nodes = []
        while node is not None and not (skip_root_node and node.is_root):
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _aggregate(self, node: Node, name: str, skip_root_node: bool = True) -> list[Any]:
        attr = []
        nodes = self._get_ancestor_nodes(node, skip_root_node=skip_root_node)
        for node in nodes:
            attr.extend(getattr(node, name))
        return attr

    def _get_non_root_nodes(self, remove_duplicates: bool = False) -> list[Node]:
        nodes = []
        for node in self.leaf_nodes:
            nodes.extend(self._get_ancestor_nodes(node))
        nodes = list(set(nodes)) if remove_duplicates else nodes
        return nodes

    def _branch(self) -> Node:
        nodes = self._get_non_root_nodes()
        lengths = [len(node) for node in nodes]
        node = random.choices(nodes, weights=lengths)[0]
        node = node.branch()
        return node

    def _compute_reward(self, tokenizer: PreTrainedTokenizer, compute_score: Callable[..., float]):
        data_source = self.data.non_tensor_batch["data_source"][0]
        ground_truth = self.data.non_tensor_batch["reward_model"][0]["ground_truth"]
        extra_info = self.data.non_tensor_batch["extra_info"][0]
        for node in self.leaf_nodes:
            response_ids = self._aggregate(node, name="token_ids")
            response_str = tokenizer.decode(response_ids, skip_special_tokens=True)
            score = compute_score(data_source, response_str, ground_truth, extra_info)
            node.reward = score

    def _compute_value(self, node: Optional[Node] = None):
        node = node or self.root_node
        if node.is_leaf:
            node.count = 1
            node.value = node.reward
        else:
            count = 0
            value = 0.0
            for child in node.children:
                self._compute_value(node=child)
                count += child.count
                value += child.value * child.count
            node.count = count
            node.value = node.reward + value / count

    def _compute_traj_advantage(self, node: Node) -> float:
        return node.value - self.root_node.value

    def _compute_step_advantage(self, node: Node) -> float:
        return node.value - node.parent.value

    def _compute_advantage(self, node: Optional[Node] = None):
        node = node or self.root_node
        if not node.is_root:
            traj_advantage = self._compute_traj_advantage(node)
            step_advantage = self._compute_step_advantage(node)
            node.advantage = traj_advantage + step_advantage
        for child in node.children:
            self._compute_advantage(node=child)

    def get_prompts(self) -> DataProto:
        self.curr_node = self._branch() if self.leaf_nodes else self.root_node
        prompt_ids = self._aggregate(self.curr_node, name="token_ids", skip_root_node=False)
        self.data.non_tensor_batch["raw_prompt_ids"][0] = prompt_ids
        return self.data

    def update_output(self, output: DataProto):
        prev_response_ids = self._aggregate(self.curr_node, name="token_ids")
        prev_response_length = len(prev_response_ids)
        response_ids = output.batch["responses"][0].tolist()
        max_response_length = len(response_ids)
        attention_mask = output.batch["attention_mask"][0].tolist()
        response_length = sum(attention_mask[-max_response_length:])
        response_length = min(response_length, max_response_length - prev_response_length)
        token_ids = response_ids[:response_length]
        node = Node(token_ids)
        self.curr_node.add_child(node)
        self.leaf_nodes.append(node)

    def postprocess(
        self, tokenizer: PreTrainedTokenizer, compute_score: Callable[..., float], response_length: int
    ) -> DataProto:
        self._compute_reward(tokenizer, compute_score)
        self._compute_value()
        self._compute_advantage()

        batch_size = len(self.leaf_nodes)
        data = self.data.repeat(batch_size)
        prompt_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        non_tensor_batch = data.non_tensor_batch

        response_ids = []
        rewards = []
        advantages = []
        for node in self.leaf_nodes:
            response_ids.append(self._aggregate(node, name="token_ids"))
            rewards.append(self._aggregate(node, name="rewards"))
            advantages.append(self._aggregate(node, name="advantages"))
        response_ids = pad_2d_list_to_length(response_ids, tokenizer.pad_token_id, max_length=response_length)
        rewards = pad_2d_list_to_length(rewards, 0.0, max_length=response_length)
        advantages = pad_2d_list_to_length(advantages, 0.0, max_length=response_length)

        input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response_ids, eos_token=tokenizer.eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_level_scores": rewards,
                "token_level_rewards": rewards,
                "advantages": advantages,
                "returns": advantages,
            },
            batch_size=batch_size,
        )
        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        return data


@dataclass
class BatchTree:
    data: DataProto
    trees: list[Tree] = field(default_factory=list)

    def __post_init__(self):
        for data in self.data.chunk(len(self.data)):
            self.trees.append(Tree(data=data))

    def get_prompts(self) -> DataProto:
        prompts = [tree.get_prompts() for tree in self.trees]
        prompts = DataProto.concat(prompts)
        return prompts

    def update_outputs(self, outputs: DataProto):
        for tree, output in zip(self.trees, outputs.chunk(len(outputs)), strict=True):
            tree.update_output(output)

    def postprocess(
        self, tokenizer: PreTrainedTokenizer, compute_score: Callable[..., float], response_length: int
    ) -> DataProto:
        data = []
        for tree in self.trees:
            data.append(tree.postprocess(tokenizer, compute_score, response_length))
        data = DataProto.concat(data)
        return data
