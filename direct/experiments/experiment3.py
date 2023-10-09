import itertools

import torch
import torch.utils.data
import wandb
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

import direct.dpo_trainer
import direct.generate
import direct.ppo_trainer
from direct import utils as utils
from direct.config import ExperimentConfig
from direct.data import load_prompt_dataset
from direct.evaluate import evaluate_model
from direct.model import get_generative_model
from direct.oracles import get_preference_oracle


# VRAM high water-mark = ~4.3GB for 117M param model (below numbers are with batch_size 16 / 4 grad acc steps) - ((32x2 batch size -> 5.2GB))
# CUDA context/overhead ~= 460MB
# load model + tokenizer = ~500MB
# initialise trainer (cloning ref_model) = 500GB
# load sentiment model ~270MB
# forward/backward passes ~ 1.5GB (activations and accumulating gradients) <--- scales with batchsize and model size
# could maybe reduce by ~1/3rd by changing loss function to work on positive or negative examples, but not very material
# optimiser state (ADAM) ~ 1GB <--- scales with model size

# so with a 1.6B GPT2 (32bit), 8x4 batches
# 6GB + 6GB + 10GB + 12GB = ~34


# NOTE: trying to replicate the DPO paper as closely as possible
# lr = 1e-6, with ramp over first 150 steps
# 64 batch size
# They SFT using training reviews from imdb on gpt2-large (774M) - should be doable on a 4090 (confirmed - with 8x8 grad accumulation)

# TODO:
#  * make generated preference dataset deterministic and cachable - hmm though if we do generation in a collate fn then this isn't a dataset thing, but a runtime thing
#  * note: determinism when generating with sampling may be hard due to non-deterministic CUDA operations
#  * rationalise related config and add a train_n option
#  * write notes around this
#  * add online version of dpo that generates new prompts from the current model
#  * implement original DB DPL objective and run
#  * standardise the "prompt -> pair -> ranked pair" data pipeline - perhaps collators, perhaps custom dataloaders.  Maybe the "label" is 0 or 1, rather than having winner and loser fields
#  * perhaps our custom workflow could mix CausalLM and pref ranking data into a single batch (configurable splits) to unify the DPL training loop
#  * check in
#  * re-run (possible on big model) on cluster


def build_generated_pair_dataloader(config, training_prompt_dataloader, gen_model, gen_tokenizer, preference_oracle, train_n):
    # Hmm - generating from our SFT model - this drifts from our current LM
    print("generating training dataset...")

    preference_getter = direct.generate.RankedCompletionSetGenerator(
        gen_model, gen_tokenizer, preference_oracle,
        direct.utils.LengthSampler(*config.data.completion_len_range), config.generate_gpt2.to_kwargs(),
        num_completions=2)

    pair_dataset = []

    # FIXME - this doesn't do a good progress bar since we escape early!
    for prompt_batch in tqdm(training_prompt_dataloader):
        if len(pair_dataset) >= train_n:
            pair_dataset = pair_dataset[:train_n]
            break
        preference_batch = preference_getter.generate(prompt_batch, config.device, config.device)
        preference_batch["response_tokens_w"] = [t.cpu() for t in preference_batch["response_tokens_w"]]
        preference_batch["response_tokens_l"] = [t.cpu() for t in preference_batch["response_tokens_l"]]
        preference_batch["prompt_tokens"] = [t.cpu() for t in preference_batch["prompt_tokens"]]
        pair_dataset.extend(utils.cols_to_records(preference_batch))

    def collate(data) -> dict:
        res = dict((key, [datum[key] for datum in data]) for key in data[0])

        for k in ["query_tokens", "response_tokens_w", "response_tokens_l"]:
            res[k] = [t.to(config.device) for t in res[k]]
        return res

    class PreferencePairDataset(Dataset):
        def __getitem__(self, index):
            return pair_dataset[index]

        def __len__(self):
            return len(pair_dataset)

    return torch.utils.data.DataLoader(PreferencePairDataset(),
                                       batch_size=config.train.batch_size * config.train.grad_acc_steps,
                                       collate_fn=collate,
                                       drop_last=True,
                                       shuffle=True)


def run_dpo(config: ExperimentConfig):
    utils.always_assert("UNTESTED CODE ahead!")

    print("Starting DPO run")
    gen_model, gen_tokenizer = get_generative_model(config)
    gen_trainer = direct.model.DirectPreferenceTrainer(config, gen_model, gen_tokenizer)

    prompt_datasets = load_prompt_dataset(tokenizer=gen_tokenizer, config=config)
    preference_oracle = get_preference_oracle(config=config)

    effective_batch_size = config.train.batch_size * config.train.grad_acc_steps

    training_prompt_dataloader = torch.utils.data.DataLoader(
        prompt_datasets["train"],
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
    )

    train_dataloader = build_generated_pair_dataloader(
        config, training_prompt_dataloader, gen_model, gen_tokenizer, preference_oracle, config.exp3.dpo_train_set_size)

    train_loop(config,
               [train_dataloader],
               gen_trainer,
               gen_model,
               gen_tokenizer,
               Subset(prompt_datasets["test"], range(config.eval.test_set_size)),
               preference_oracle,
               lambda _batch: gen_trainer.step_pairs(_batch[0], config.train.grad_acc_steps))


def run_ppo_gt(config: ExperimentConfig):
    print("Starting run")
    gen_model, gen_tokenizer = get_generative_model(config)

    assert(config.train.grad_acc_steps == 1), "Don't use grad acc with PPOTrainer unless we understand it better"

    gen_trainer = direct.ppo_trainer.PPOPreferenceTrainer(
        config, config.train.ppo.get_trl_ppo_config(), gen_model, gen_tokenizer)

    prompt_datasets = load_prompt_dataset(tokenizer=gen_tokenizer, config=config)
    preference_oracle = get_preference_oracle(config=config)

    training_prompt_dataloader = torch.utils.data.DataLoader(
        prompt_datasets["train"],
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
    )

    train_loop(config,
               [training_prompt_dataloader],
               gen_trainer,
               gen_model,
               gen_tokenizer,
               Subset(prompt_datasets["test"], range(config.eval.test_set_size)),
               preference_oracle,
               lambda _batch: gen_trainer.step(_batch[0], preference_oracle, config.device))


def train_loop(config, train_dataloaders,
               gen_trainer, gen_model, gen_tokenizer, test_prompt_dataset, preference_oracle, step_fn):
    scheduler = torch.optim.lr_scheduler.LinearLR(gen_trainer.optimizer,
                                                  start_factor=config.train.lr_ramp_start_factor,
                                                  total_iters=config.train.lr_ramp_total_iters)

    i_train_dataloaders = [iter(itertools.cycle(dl)) for dl in train_dataloaders]

    for step in range(config.steps):
        batches = [next(itr) for itr in i_train_dataloaders]

        logs = dict()

        stats, scores = step_fn(batches)
        scores = torch.tensor(scores)

        logs.update(stats)
        logs.update({
            "env/pref_mean": torch.mean(scores).item(),
            "env/sigma_pref_mean": torch.mean(torch.sigmoid(scores)).item(),
            "env/pref_std": torch.std(scores).item(),
            "env/pref_dist": scores.numpy(),
            "lr": scheduler.get_last_lr(),
        })

        if (step+1) % config.eval.interim_eval_interval_steps == 0:
            print("Running interim eval")
            eval_stats = evaluate_model(
                gen_model,
                gen_trainer.ref_model,
                gen_tokenizer,
                test_prompt_dataset,
                preference_oracle,
                config,
                include_samples=True)

            logs.update(eval_stats)
            utils.print_stats(logs, f"Eval (step={step}): ",
                              ["eval/pref_mean",
                               "eval/sigma_pref_mean",
                               "eval/pref_std",
                               "eval/kl_mean"])

        utils.print_stats(logs, f"Step {step}/{config.steps}, stats: ",
                          [gen_trainer.loss_property_name,
                           "env/pref_mean",
                           "env/pref_std",
                           "env/kl_mean",
                           "env/sigma_pref_mean",
                           "lr"])
        scheduler.step(step)
        if config.log:
            wandb.log(logs)
