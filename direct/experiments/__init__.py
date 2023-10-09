import dataclasses
import wandb

from direct.config import ExperimentConfig

from direct.experiments import (experiment5)


EXPERIMENTS = {
    "exp5": experiment5.run,
}


def run_experiment(config: ExperimentConfig):
    fn = EXPERIMENTS.get(config.experiment_name)
    if fn is None:
        raise NotImplementedError(f"Don't know how to run experiment {config.experiment_name}")

    if config.wandb_tags is not None:
        tags = config.wandb_tags.split(",")
    else:
        tags = None

    # optionally setup logging
    if config.log:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run,
            reinit=True,
            config=dataclasses.asdict(config),
            tags=tags
        )
        wandb.run.log_code(".")

    fn(config)

    if config.log:
        # Call this explicitly, since the automatic version is unreliable in a debug session
        wandb.finish()
