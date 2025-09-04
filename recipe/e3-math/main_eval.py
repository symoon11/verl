import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    pass


if __name__ == "__main__":
    main()
