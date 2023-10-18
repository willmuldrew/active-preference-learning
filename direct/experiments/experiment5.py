import itertools
import json
import os
import random
from functools import partial

import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import wandb
from torch.utils.data import Subset, Dataset

import direct.dpo_trainer
import direct.generate
import direct.model
import direct.ppo_trainer
from direct import utils as utils
from direct.config import ExperimentConfig
from direct.data import load_prompt_dataset
from direct.evaluate import evaluate_model
from direct.model import get_generative_model
from direct.oracles import get_preference_oracle


def random_subset(dataset: Dataset, size: int):
    return Subset(
        dataset,
        random.sample(list(range(len(dataset))), k=size)
    )


def run(config: ExperimentConfig):
    print("Starting exp5 run")

    direct.utils.seed_everything(config.seed)

    gen_model, gen_tokenizer = get_generative_model(config)

    gen_model.gradient_checkpointing_enable()

    preference_oracle = get_preference_oracle(config)
    gen_trainer = direct.dpo_trainer.DirectPreferenceTrainer(config, gen_model, gen_tokenizer)

    prompt_datasets = load_prompt_dataset(tokenizer=gen_tokenizer, config=config)

    test_prompt_dataset = random_subset(prompt_datasets["test"], config.eval.test_set_size)
    train_prompt_dataset = random_subset(prompt_datasets["train"], len(prompt_datasets["train"]))

    def step_fn(batch):
        stats, _ = gen_trainer.step_pairs(batch, config.train.grad_acc_steps)
        return stats

    simple_training_loop(config, gen_trainer, gen_model, gen_tokenizer, train_prompt_dataset, test_prompt_dataset, preference_oracle, step_fn)


def is_scalar(v):
    if torch.is_tensor(v):
        return torch.squeeze(v).dim() == 0
    elif isinstance(v, np.ndarray):
        return np.squeeze(v).ndim == 0
    else:
        return isinstance(v, (int, float, bool, complex, np.generic))


def simple_training_loop(
        config,
        gen_trainer,
        gen_model, gen_tokenizer,
        train_prompt_dataset,
        test_prompt_dataset,
        preference_oracle, step_fn
):
    # We're going to use these prompts for generation - crank up the batch size to make it go faster
    prompt_dataloader = torch.utils.data.DataLoader(
        train_prompt_dataset,
        config.exp5.prompt_batch_size,
        shuffle=True,
        drop_last=True,
    )

    source_model = gen_model
    if config.exp5.acquire_pairs_function == "OFFLINE":
        source_model = gen_trainer.ref_model

    completion_pair_generator = direct.generate.CompletionSetGenerator(
        source_model, gen_tokenizer,
        direct.utils.LengthSampler(*config.data.completion_len_range),
        dict(config.generate_gpt2.to_kwargs()),
        num_completions=2
    )

    def reset_model():
        # Reset model and scheduler
        gen_model.load_state_dict(gen_trainer.ref_model.state_dict())

    def maybe_get_eval_save_path(filename):
        if wandb.run.dir is not None:
            return os.path.join(wandb.run.dir, filename)
        else:
            return None

    def reset_optimizer():
        gen_trainer.reset_optimizer()

    def print_training_progress(stats):
        scalar_stats = {k: v for (k, v) in stats.items() if is_scalar(v)}
        print(
            f"step: {scalar_stats['step']} epoch: {scalar_stats['epoch']} m: {scalar_stats['m']}",
            scalar_stats)

    def train_to_convergence(training_data):
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(gen_trainer.optimizer,
                                                         start_factor=config.train.lr_ramp_start_factor,
                                                         total_iters=config.train.lr_ramp_total_iters)

        early_stopper = direct.utils.LossMAEarlyStopper(threshold=config.exp5.loss_ma_early_stopper_threshold)

        train_dl = torch.utils.data.DataLoader(training_data, config.train.effective_batch_size, collate_fn=utils.records_to_cols, shuffle=True)

        # Need to do this since gradient checkpointing won't be done without being in the right mode
        gen_trainer.model.train()

        stop = False
        step = -1
        epoch = -1
        while not stop:
            epoch += 1
            for batch in train_dl:
                torch.cuda.reset_peak_memory_stats()
                step += 1

                if config.exp5.max_steps is not None and config.exp5.max_steps > 0:
                    if step > config.exp5.max_steps:
                        print("STOPPING - reached max_steps")
                        stop = True
                        break

                if config.exp5.max_epochs is not None and config.exp5.max_epochs > 0:
                    if epoch > config.exp5.max_epochs:
                        print("STOPPING - reached max_epochs")
                        stop = True
                        break

                stats = step_fn(batch)
                stats.update({
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": step,
                    "epoch": epoch,
                    "m": m,
                    "stopper_loss_ma": early_stopper.ma,
                    "stopper_target": early_stopper.target,
                })

                if config.log:
                    wandb.log(stats)

                lr_scheduler.step(step)

                print_training_progress(stats)
                early_stopper.update(stats)

                if config.eval.interim_eval_interval_steps > 0:
                    if step % config.eval.interim_eval_interval_steps == 0:
                        do_eval("interim_training",
                                save_path=maybe_get_eval_save_path(
                                    f"evaluation_m{len(training_data)}_step-{step}_midepoch-{epoch}"))

                print(f"Maximum GPU memory used: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")

            if config.eval.eval_epoch_interval is not None and config.eval.eval_epoch_interval > 0:
                if epoch % config.eval.eval_epoch_interval == 0:
                    do_eval("interim_training", save_path=maybe_get_eval_save_path(
                                        f"evaluation_m{len(training_data)}_step-{step}_endepoch-{epoch}"))

            if early_stopper.should_stop():
                break

    i_prompt_dataloader = iter(itertools.cycle(prompt_dataloader))

    def acquire_completion_pairs_random(minimum_count):
        """
        NB: May return more than minimum_count...
        """
        new_data = []
        with tqdm(total=minimum_count, desc=f"generating {minimum_count} datapoints using RANDOM") as pbar:
            while len(new_data) < minimum_count:
                new_data.extend(utils.cols_to_records(
                    completion_pair_generator.generate(next(i_prompt_dataloader), config.device, "cpu")
                ))
                pbar.update(len(new_data) - pbar.n)
        return new_data

    def acquire_completion_pairs_uncertainty(minimum_count, which, over_generate_factor=1):
        """
        May return more than minimum_count...
        """
        new_data = []

        print(f"generating {minimum_count} datapoints using {which} UNCERTAINTY")

        with tqdm(total=minimum_count * over_generate_factor, desc="Over-generating points") as pbar:
            while len(new_data) < minimum_count * over_generate_factor:
                new_data.extend(utils.cols_to_records(
                    completion_pair_generator.generate(next(i_prompt_dataloader), config.device, "cpu")
                ))
                pbar.update(len(new_data) - pbar.n)

        print("Computing certainty scores using gen_model")
        all_scores, all_rhats = compute_model_uncertainty(gen_model, gen_trainer.ref_model, gen_trainer.tokenizer, new_data,
                                                          config.train.dpo.beta, config.train.batch_size, config.device)

        for d, rhats in zip(new_data, all_rhats):
            d["r_hats"] = rhats

        new_data = [new_data[i] for i in torch.argsort(torch.Tensor(all_scores))]

        if which == 'most':
            print("Retaining MOST uncertain points")
            new_data = new_data[-minimum_count:]
        elif which == 'least':
            print("Retaining LEAST uncertain points")
            new_data = new_data[:minimum_count]
        elif which == 'mid':
            print("Retaining MIDDLE uncertain points")
            s = (len(new_data) // 2) - (minimum_count // 2)
            e = s + minimum_count
            new_data = new_data[s:e]
        elif which == 'tails':
            print("Retaining LEAST/MOST uncertain points")
            b = minimum_count // 2
            t = minimum_count - b
            new_data = new_data[:b] + new_data[-t:]
        else:
            raise NotImplementedError(f"Don't know how to select '{which}'")

        return new_data

    def acquire_completion_pairs_predictive_entropy(minimum_count, over_sample_prompts_factor=1, entropy_sample_n=16):
        """
        We're going to select our *prompts* using an MC estimate of the entropy of the conditional response distribution

        Simple version:  https://arxiv.org/pdf/2207.05221.pdf p29
        """
        print(f"generating {minimum_count} datapoints using ENTROPY")

        # over sample prompts from our loader
        prompts = []
        while len(prompts) < minimum_count * over_sample_prompts_factor:
            prompts.extend(next(i_prompt_dataloader)['prompt'])
        prompts = prompts[:minimum_count * over_sample_prompts_factor]

        # now score and take the best/worst
        h = []
        for prompt in tqdm(prompts, desc=f"estimating prompt entropy (n={entropy_sample_n}) for {len(prompts)} prompts"):
            h.append(estimate_prompt_entropy(prompt, gen_model, gen_tokenizer, config.exp5.prompt_batch_size, n=entropy_sample_n))

        # now we want to take the highest entropy (most uncertain) prompts
        prompts = [prompts[n] for n in torch.argsort(torch.Tensor(h))]
        prompts = prompts[-minimum_count:]
        random.shuffle(prompts)

        # Now need to generate pairs...
        new_data = []

        filtered_prompt_dataloader = torch.utils.data.DataLoader(prompts, config.exp5.prompt_batch_size)
        # for prompt_batch in tqdm(filtered_prompt_dataloader, desc="Generating completion pairs from filtered prompts"):
        #     new_data.extend(utils.cols_to_records(
        #         completion_pair_generator.generate(prompt_batch, config.device, "cpu")
        #     ))
        for prompt_batch in filtered_prompt_dataloader:
            new_data.extend(utils.cols_to_records(
                completion_pair_generator.generate(dict(prompt=prompt_batch), config.device, "cpu")
            ))

        return new_data

    @torch.no_grad()
    def estimate_prompt_entropy(prompt, gen_model, tokenizer, batch_size, n=32):
        assert (n % batch_size == 0), "n should be multiple of batch size"
        h_sum = 0.0
        for _ in range(n // batch_size):
            gen_lens = direct.utils.LengthSampler(*config.data.completion_len_range)(n=batch_size)
            gen_args = dict(config.generate_gpt2.to_kwargs())
            gen_args["temperature"] = 1.0
            generated_output = direct.generate.generate(
                gen_model, tokenizer, [prompt] * batch_size, gen_lens, gen_args, config.device, True)

            logprobs, response_mask, ref_logprobs = direct.model.batch_forward_pass(gen_model, None, tokenizer,
                                            generated_output.prompt_tokens, generated_output.completion_tokens)

            h_sum += -((logprobs * response_mask).sum()).item() / batch_size

        h = h_sum / (n // batch_size)
        return h

    acquire_completion_pairs = dict(
        RANDOM=acquire_completion_pairs_random,
        OFFLINE=acquire_completion_pairs_random,
        CERTAINTY=partial(acquire_completion_pairs_uncertainty, which='least', over_generate_factor=config.exp5.over_generate_factor),
        UNCERTAINTY=partial(acquire_completion_pairs_uncertainty, which='most', over_generate_factor=config.exp5.over_generate_factor),
        MID_UNCERTAINTY=partial(acquire_completion_pairs_uncertainty, which='mid', over_generate_factor=config.exp5.over_generate_factor),
        TAILS_UNCERTAINTY=partial(acquire_completion_pairs_uncertainty, which='tails', over_generate_factor=config.exp5.over_generate_factor),
        ENTROPY=partial(acquire_completion_pairs_predictive_entropy, over_sample_prompts_factor=config.exp5.over_sample_prompts_factor, entropy_sample_n=config.exp5.entropy_sample_n)
    )[config.exp5.acquire_pairs_function]

    def expand_training_dataset(current_dataset, target_m):
        print(f"Expanding dataset from {len(current_dataset)} to {target_m}")
        current_dataset = list(current_dataset)
        while len(current_dataset) < target_m:
            required_new = target_m - len(current_dataset)
            new_data = acquire_completion_pairs(minimum_count=required_new)

            # Now get labels
            oracle_response = preference_oracle.consult_the_oracle(
                [d['prompt'] for d in new_data],
                [[d[f"completion_{n}"] for d in new_data] for n in [0, 1]],
                random_tie_break=False)
            direct.oracles.apply_preferences_to_completion_set(new_data, oracle_response)

            for d in new_data:
                if d["rank"] is None:
                    print(f"dropping datapoint from from batch: {d['prompt']}\n,A: {d['completion_0']}\n,B: {d['completion_1']}\nRATIONALE:{d['rationale']}")
                else:
                    d["target_m"] = target_m
                    current_dataset.append(d)

        current_dataset = current_dataset[:target_m]
        random.shuffle(current_dataset)
        return current_dataset

    def do_eval(prefix, save_path=None, additional_stats=None):
        for t in config.eval.sampling_temperatures:
            eval_stats = evaluate_model(
                gen_model,
                gen_trainer.ref_model,
                gen_tokenizer,
                test_prompt_dataset,
                preference_oracle,
                config,
                sample_temperature=t,
                prefix=f"{prefix}/eval_T{t:0.2f}",
                vs_model=gen_trainer.ref_model if config.eval.vs_ref_model_eval else None,
                save_path=save_path + f"_T{t:0.2f}.json",
            )
            if additional_stats is not None:
                eval_stats.update(additional_stats)
            print("EVAL:", {k: v for (k, v) in eval_stats.items() if is_scalar(v)})
            if config.log:
                wandb.log(eval_stats)

    training_data = []
    m_schedule = config.exp5.m_schedule
    eval_m_schedule = config.exp5.eval_m_schedule or m_schedule

    # TODO
    #  [x] establish good systematic stopping criterion
    #     - let's evaluate the win rate every 10 steps for a single run with M=128 - see how win-rate and training loss co-evolve
    #     - training to some low training error seems to be a useful proxy and doesn't seem to overfit - though I should probably do some really long runs with small and larger m
    #  [x] add eval (end of training, but also optionally during training)
    #  [x] more metric logging
    #  [ ] checkpoint writing
    #  [ ] different datapoint acquisition functions
    #  [x] make seedable/more deterministic
    #  [ ] what if I discard the existing data each time (recycle vs reacquire)? Question I posed in PH meeting on 21st Sept - seems interesting, but manana

    print("m_schedule", m_schedule)
    print("eval_m_schedule", eval_m_schedule)

    for m in m_schedule:
        if config.exp5.reacquire_all_data:
            training_data.clear()
        training_data = expand_training_dataset(training_data, m)

        if wandb.run.dir is not None:
            data_dump_path = os.path.join(wandb.run.dir, f"training_data_m{m}.jsonl")
            print(f"Logging {len(training_data)} rows of training data -> {data_dump_path}")
            with open(data_dump_path, "wt", encoding="utf-8") as f:
                for p in training_data:
                    p2 = {k: v for k, v in p.items() if not isinstance(v, torch.Tensor)}
                    json.dump(p2, f)
                    f.write("\n")

        reset_model()
        reset_optimizer()
        if config.exp5.acquire_pairs_function == "OFFLINE" and m not in eval_m_schedule:
            print("Skipping training since we're OFFLINE and m not in eval_m_schedule")
            pass
        else:
            train_to_convergence(training_data)
        if m in eval_m_schedule:
            do_eval(f"post_training_m", save_path=maybe_get_eval_save_path(
                                        f"evaluation_m{len(training_data)}_post_training"), additional_stats={"m": m})

        # if wandb.run.dir is not None:
        #     checkpoint_dir = os.path.join(wandb.run.dir, f"checkpoint_final_m{m}")
        #     write_checkpoint(checkpoint_dir)


@torch.no_grad()
def compute_model_uncertainty(gen_model, gen_ref_model, tokenizer, pairs, beta, batch_size, device):
    all_scores = []
    rhats = []
    batch_size = min(batch_size, len(pairs))
    for b in range(len(pairs) // batch_size):
        pair_batch = pairs[b * batch_size:(b + 1) * batch_size]

        col_pairs = utils.tensor_cols_to(utils.records_to_cols(pair_batch), device)

        def fwd_pass(response_tokens):
            return direct.model.batch_forward_pass(
                gen_model, gen_ref_model, tokenizer,
                col_pairs["prompt_tokens"], response_tokens,
            )

        logprobs_a, response_mask_a, ref_logprobs_a = fwd_pass(col_pairs["completion_tokens_0"])
        logprobs_b, response_mask_b, ref_logprobs_b = fwd_pass(col_pairs["completion_tokens_1"])

        r_a = beta * ((logprobs_a * response_mask_a).sum(dim=1) - (ref_logprobs_a * response_mask_a).sum(dim=1))
        r_b = beta * ((logprobs_b * response_mask_b).sum(dim=1) - (ref_logprobs_b * response_mask_b).sum(dim=1))

        # This expression is just to try and have a symmetric curve that is 1.0 when r_a == r_b, and tails off as
        # abs(r_a - r_b) increases.  However... it results in the exact same ranking as abs(r_a - r_b)
        u = 1.0 - 2.0 * torch.abs(torch.sigmoid(r_a - r_b) - 0.5)

        for n, p in enumerate(pair_batch):
            rhats.append([r_a[n].item(), r_b[n].item()])

        all_scores.append(u)

    return torch.cat(all_scores).tolist(), rhats
