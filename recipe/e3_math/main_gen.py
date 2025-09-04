import hydra
from omegaconf import DictConfig

import pandas as pd
from pprint import pprint
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="gen", version_base=None)
def main(config: DictConfig):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    # read data
    data = pd.read_parquet(config.data.path)
    print(data)

if __name__ == "__main__":
    main()
