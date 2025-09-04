from pprint import pprint

import hydra
import os
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams


@hydra.main(config_path="config", config_name="gen", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # load dataset
    dataset = load_dataset(**config.dataset)
    prompts = [example["prompt"] for example in dataset]

    # create LLM
    llm = LLM(**config.llm)
    sampling_params = SamplingParams(**config.sampling_params)

    # generate responses
    outputs = llm.chat(prompts, sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    dataset = dataset.add_column("response", responses)

    # save responses
    output_path = os.path.join(config.output_dir, f"gen_test_{config.seed}.parquet")
    dataset.to_parquet(output_path)


if __name__ == "__main__":
    main()
