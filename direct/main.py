import pyrallis

from direct.config import ExperimentConfig

from direct.experiments import run_experiment
from direct.utils import get_default_device


@pyrallis.wrap()
def main(config: ExperimentConfig):
    if config.device is None:
        config.device = get_default_device()
    print("Config: ")
    print(config)
    print(f"Running on device: {config.device}")
    run_experiment(config)


if __name__ == "__main__":
    main()
