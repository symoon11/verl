import os
from collections import defaultdict
from pprint import pprint

import hydra
import numpy as np
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from recipe.e3_math.reward_function import compute_score


def get_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # load dataset
    data_files = os.listdir(config.output_dir)
    dataset = load_dataset(**config.dataset, data_files=data_files)
    data_sources = dataset.unique("data_source")
    data_sources.sort()

    # compute pass@k
    for data_source in data_sources:
        print(f"Data Source: {data_source}")
        data = dataset.filter(lambda x: x["data_source"] == data_source)

        scores = defaultdict(list)
        for example in data:
            response = example["response"]
            answer = example["answer"]
            extra_info = example["extra_info"]
            index = extra_info["index"]
            score = compute_score(data_source, response, answer, extra_info)
            scores[index].append(score)

        pass_at_ks = defaultdict(list)
        for score in scores.values():
            n = len(score)
            c = sum(score)
            for k in [1, 2, 4, 8, 16, 32]:
                pass_at_k = get_pass_at_k(n, c, k)
                pass_at_ks[k].append(pass_at_k)

        for k, pass_at_k in pass_at_ks.items():
            print(f"Pass@{k}: {np.mean(pass_at_k):.3f}")


if __name__ == "__main__":
    main()
