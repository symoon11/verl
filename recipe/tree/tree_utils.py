from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Callable, Optional

import numpy as np
from datasets import load_dataset
from reward_function import compute_score
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams, TokensPrompt


@dataclass
class Node:
    token_ids: list[int]
    parent: Optional[Node] = None
    children: list[Node] = field(default_factory=list)
    reward: float = 0.0
    values: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        try:
            return len(self.token_ids) - self.token_ids[::-1].index(151668)
        except ValueError:
            return len(self.token_ids)

    def __lt__(self, other: Node) -> bool:
        return len(self) > len(other)

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

    def branch(self) -> tuple[Node, Node]:
        idx = np.random.randint(0, len(self))
        node = Node(self.token_ids[:idx])
        self.token_ids = self.token_ids[idx:]
        self.parent.add_child(node)
        self.parent.remove_child(self)
        node.add_child(self)
        return node, self


@dataclass
class Tree:
    token_ids: list[int]
    root_node: Node = field(init=False)
    curr_node: Node = field(init=False)
    leaf_nodes: list[Node] = field(default_factory=list)
    queue: PriorityQueue[Node] = field(default_factory=PriorityQueue)

    def __post_init__(self):
        self.root_node = Node(self.token_ids)
        self.curr_node = self.root_node

    def agg_token_ids(self, node: Node, skip_root_node: bool = False) -> list[int]:
        token_ids = []
        while node is not None and not (skip_root_node and node.is_root):
            token_ids.append(node.token_ids)
            node = node.parent
        token_ids = list(itertools.chain.from_iterable(reversed(token_ids)))
        return token_ids

    def branch(self) -> list[int]:
        if not self.queue.empty():
            node = self.queue.get()
            self.curr_node, node = node.branch()
            self.queue.put(self.curr_node)
            self.queue.put(node)
        token_ids = self.agg_token_ids(self.curr_node)
        return token_ids

    def update(self, token_ids: list[int]):
        node = Node(token_ids)
        self.curr_node.add_child(node)
        self.leaf_nodes.append(node)
        self.queue.put(node)

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
            node.reward = reward

    def compute_value(self):
        for node in self.leaf_nodes:
            reward = node.reward
            while node is not None:
                node.values.append(reward)
                node = node.parent

    def compute_advantage(self):
        pass


if __name__ == "__main__":
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        max_model_len=10240,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=1.0, max_tokens=4096)
    dataset = load_dataset("CMU-AIRe/e3-math-easy", split="train")
    example = dataset[300]
    prompt = example["prompt"]
    ground_truth = example["reward_model"]["ground_truth"]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    tree = Tree(token_ids=prompt_ids)
    for _ in range(8):
        prompt_ids = tree.branch()
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        response = llm.generate(prompt, sampling_params=sampling_params)
        response_ids = response[0].outputs[0].token_ids
        tree.update(response_ids)
    print("Compute reward")
    tree.compute_reward("", ground_truth, {}, tokenizer, compute_score)
    print("Compute value")
    tree.compute_value()
    for i, node in enumerate(tree.leaf_nodes):
        print(f"Leaf node {i}th val: {node.values}")
    print(f"Root node val: {tree.root_node.values}")
