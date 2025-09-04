from pprint import pprint

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams


@hydra.main(config_path="config", config_name="gen", version_base=None)
def main(config: DictConfig):
    # Config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Dataset
    dataset = load_dataset(**config.dataset)
    dataset = dataset.select(range(1))
    prompts = [example["prompt"] for example in dataset]

    # vLLM
    llm = LLM(**config.llm)
    sampling_params = SamplingParams(**config.sampling)

    # Generation
    outputs = llm.chat(prompts, sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    dataset = dataset.add_column("response", responses)

    # Save
    dataset.to_parquet(config.output_path)


if __name__ == "__main__":
    main()
