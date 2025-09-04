import hydra
from omegaconf import DictConfig

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


@hydra.main(config_path="config", config_name="merge", version_base=None)
def main(cfg: DictConfig):
    config = ModelMergerConfig(**cfg.merger)
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
