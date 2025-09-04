import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    # load dataset
    os.listdir(cfg.output_path)


if __name__ == "__main__":
    main()
