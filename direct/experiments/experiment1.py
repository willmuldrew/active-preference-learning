import itertools

import numpy as np
import torch
import torch.utils.data
import wandb
from torch.utils.data import Subset

from direct import utils as utils
from direct.config import ExperimentConfig
from direct.data import load_prompt_dataset
from direct.evaluate import evaluate_model
from direct.model import get_generative_model, TRAINER_REGISTRY
from direct.oracles import get_preference_oracle

from direct.utils import always_assert


def run(config: ExperimentConfig):
    always_assert("UNTESTED CODE - may be very broken!")

    # get pretrained generative model
    # noinspection PyUnreachableCode
    model, tokenizer = get_generative_model(config)

    # get training and test data loaders
    datasets = load_prompt_dataset(tokenizer=tokenizer, config=config)

    train_dataloader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=config.train.batch_size,
        shuffle=True
    )

    # setup trainer for fine-tuning
    trainer = TRAINER_REGISTRY[config.train.trainer_class](
        config=config,
        model=model,
        tokenizer=tokenizer,
    )

    # get preference model
    preference_model = get_preference_oracle(config=config)

    # run the fine-tuning
    print("Start main training loop")

    total_epochs = int(np.ceil(config.steps / config.train.batch_size))
    for epoch, batch in zip(range(total_epochs), iter(itertools.cycle(train_dataloader))):
        if len(batch["query_tokens"]) != config.train.batch_size:
            print("Skipping truncated batch")
            continue

        logs = dict()

        with utils.timeit("time/epoch", logs):
            with utils.timeit("time/optimization", logs):
                stats, scores = trainer.step(batch=batch, preference_model=preference_model, device=config.device)

            if isinstance(scores, list):
                scores = torch.tensor(scores)

        if epoch % config.eval.interim_eval_interval_steps == 0:
            print("Running interim eval")
            eval_stats = evaluate_model(
                model, trainer.ref_model, tokenizer,
                Subset(datasets["test"], range(config.eval.test_set_size)),
                preference_model, config, include_samples=True)
            logs.update(eval_stats)
            utils.print_stats(logs, f"Eval: ", ["eval/pref_mean", "eval/pref_std"])

        logs.update(stats)
        logs.update({
            "env/pref_mean": torch.mean(scores).item(),
            "env/pref_std": torch.std(scores).item(),
            "env/pref_dist": scores.cpu().numpy()
        })

        utils.print_stats(logs, f"Epoch {epoch}/{total_epochs}: ",
                          [trainer.loss_property_name, "env/pref_mean", "env/pref_std"])
        if config.log:
            wandb.log(logs)

    print("Main training loop finished")

    if config.eval.perform_final_full_eval:
        print("Running final eval:")
        final_eval_stats = evaluate_model(
            model, trainer.ref_model, tokenizer, datasets["test"], preference_model, config, num_batches=None,
            include_samples=False,
            prefix="final_eval")
        print(final_eval_stats)
        if config.log:
            wandb.log(final_eval_stats)
    else:
        print("Skipping final eval")
