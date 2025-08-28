from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from transformers import PreTrainedTokenizer

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
    tokenizer: PreTrainedTokenizer
    token_ids: list[int]
    root_node: Node = field(init=False)
    curr_node: Node = field(init=False)
    leaf_nodes: list[Node] = field(default_factory=list)

    def __post_init__(self):
        self.root_node = Node(self.token_ids)
        self.curr_node = self.root_node

    def get_ancestor_nodes(self, node: Node, skip_root_node: bool = False) -> list[Node]:
        nodes = []
        while node is not None and not (skip_root_node and node.is_root):
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def agg_token_ids(self, node: Node, skip_root_node: bool = False) -> list[int]:
        token_ids = []
        nodes = self.get_ancestor_nodes(node, skip_root_node=skip_root_node)
        for node in nodes:
            token_ids.extend(node.token_ids)
        return token_ids

    def get_non_root_nodes(self, remove_duplicates: bool = False) -> list[Node]:
        nodes = []
        for node in self.leaf_nodes:
            nodes.extend(self.get_ancestor_nodes(node, skip_root_node=True))
        if remove_duplicates:
            nodes = list(set(nodes))
        return nodes

    def branch(self) -> Node:
        nodes = self.get_non_root_nodes()
        lengths = [len(node) for node in nodes]
        node = random.choices(nodes, weights=lengths)[0]
        node = node.branch()
        return node

    def get_prompt(self) -> list[int]:
        if len(self.leaf_nodes) > 0:
            self.curr_node = self.branch()
        token_ids = self.agg_token_ids(self.curr_node)
        return token_ids

    def update_response(self, token_ids: list[int]):
        node = Node(token_ids)
        self.curr_node.add_child(node)
        self.leaf_nodes.append(node)

    def compute_reward(
        self,
        data_source: str,
        ground_truth: str,
        extra_info: dict[str, any],
        tokenizer: PreTrainedTokenizer,
        compute_score: Callable[..., float],
    ):
        for node in self.leaf_nodes:
            response_ids = self.agg_token_ids(node, skip_root_node=True)
            response_str = tokenizer.decode(response_ids, skip_special_tokens=True)
            reward = compute_score(data_source, response_str, ground_truth, extra_info)
            node.values.append(reward)

    def compute_value(self, node: Optional[Node] = None):
        if node is None:
            node = self.root_node
        for child in node.children:
            self.compute_value(node=child)
            node.values.extend(child.values)

    def compute_traj_advantage(self, node: Node) -> float:
        return np.mean(node.values) - np.mean(self.root_node.values)

    def compute_step_advantage(self, node: Node) -> float:
        return np.mean(node.values) - np.mean(node.parent.values)

    def compute_advantage(self, node: Optional[Node] = None, weight: float = 1.0, id = "0"):
        if node is None:
            node = self.root_node
        # if not node.is_root:
        #     traj_advantage = self.compute_traj_advantage(node)
        #     step_advantage = self.compute_step_advantage(node)
        #     node.advantage = traj_advantage + weight * step_advantage
        print(f"Node {id}")
        for i, child in enumerate(node.children):
            self.compute_advantage(node=child, weight=weight, id=f"{id}.{i}")


if __name__ == "__main__":
    from datasets import load_dataset
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model="qwen/Qwen3-0.6B",
        max_model_len=10240,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=8192,
    )
    dataset = load_dataset("CMU-AIRe/e3-math-easy", split="train")
    example = dataset[300]
    prompt = example["prompt"]
    ground_truth = example["reward_model"]["ground_truth"]
    token_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    tree = Tree(tokenizer, token_ids)
    for _ in range(8):
        prompt_token_ids = tree.get_prompt()
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        response = llm.generate(prompt, sampling_params=sampling_params)
        response_token_ids = response[0].outputs[0].token_ids
        tree.update_response(response_token_ids)
    tree.compute_advantage()
