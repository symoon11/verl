from pprint import pprint

import hydra
from omegaconf import DictConfig, OmegaConf

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


@hydra.main(config_path="config", config_name="merge", version_base=None)
def main(config: DictConfig):
    # resolve config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # create model merger
    merger_config = ModelMergerConfig(**config.merger_config)
    merger = FSDPModelMerger(merger_config)

    # merge and save model
    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
