import json_tricks
import random

import torch
import torch.utils.data
import wandb

from tqdm import tqdm

import direct.generate
import direct.model
from direct import utils
from direct.config import *
from direct.oracles import PreferenceOracle


@torch.no_grad()
def evaluate_model(model, ref_model, tokenizer, dataset,
                   preference_oracle: PreferenceOracle, config: ExperimentConfig, num_batches=None,
                   prefix="eval", shuffle_data=True, do_sample=True,
                   sample_temperature=1.0, versus=None, save_path=None):
    """
    Evaluate generative model against a given dataset and preference model.  Samples a number of batches from the
    dataset. Returns various stats and (optionally) samples of the generation process
    """
    results = {}

    with utils.timeit(f"{prefix}/time", results):
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.eval.batch_size, shuffle=shuffle_data)

        print(f"evaluate_model: model {utils.hash_model_weights(model)}, ref_model {utils.hash_model_weights(ref_model)}")
        if num_batches is not None:
            print(f"Doing '{prefix}' eval with {num_batches} batches of size {config.eval.batch_size} "
                  f"drawn from dataset of size {len(dataset)} - T={sample_temperature}")
        else:
            print(f"Doing '{prefix}' eval with full dataset of size {len(dataset)} - T={sample_temperature}")

        scores = None
        all_kl_divs = []
        all_logprobs = []
        all_ref_logprobs = []
        vs_wins = []
        vs_rows = []
        vs_rows_cols = []
        prompts = []
        completions = []
        vs_completions = []

        model.eval()

        for n_batch, batch in tqdm(enumerate(test_dataloader), desc=f"Generating {len(dataset)} eval completions", total=len(dataset) // config.eval.batch_size):
            len_batch = len(batch["prompt"])
            if len_batch != config.eval.batch_size:
                break

            if num_batches is not None and n_batch >= num_batches:
                break

            gen_lens = utils.LengthSampler(*config.data.completion_len_range)(len_batch)
            gen_args = dict(config.generate_gpt2.to_kwargs())
            gen_args["temperature"] = sample_temperature
            if sample_temperature == 0.0:
                do_sample = False

            responses = direct.generate.generate(model, tokenizer, batch["prompt"], gen_lens, gen_args,
                                                 config.device, do_sample=do_sample)

            prompts.extend(batch["prompt"])
            completions.extend(responses.completion)

            logprobs, response_mask, ref_logprobs = direct.model.batch_forward_pass(
                model, ref_model, tokenizer,
                responses.prompt_tokens,
                responses.completion_tokens,
            )

            sum_logprobs = (logprobs * response_mask).sum(dim=1)
            sum_ref_logprobs = (ref_logprobs * response_mask).sum(dim=1)
            batch_kls = (sum_logprobs - sum_ref_logprobs).cpu().tolist()
            batch_logprobs = sum_logprobs.cpu().tolist()
            batch_ref_logprobs = sum_ref_logprobs.cpu().tolist()

            all_kl_divs.extend(batch_kls)
            all_logprobs.extend(batch_logprobs)
            all_ref_logprobs.extend(batch_ref_logprobs)

            if versus is not None:
                if versus not in ["ref_model", "label"]:
                    raise NotImplementedError(f"Don't know how to compute win-rate vs '{versus}'")

                if versus == "ref_model":
                    vs_responses = direct.generate.generate(
                        ref_model, tokenizer, batch["prompt"], gen_lens, gen_args, config.device, do_sample=do_sample)
                    vs_completions.extend(vs_responses.completion)
                elif versus == "label":
                    vs_completions.extend(batch["label"])

        try:
            if not config.eval.defer:
                scores = preference_oracle.get_scores(prompts, completions)
        except NotImplementedError:
            # Doesn't support scores... (i.e. only computes preference rank)
            pass

        if len(vs_completions) > 0:
            if config.eval.defer:
                # Don't consult the oracle - we can do this post-training, but we still need to dump out our samples
                vs_rows_cols = ["prompt", "completion", "completion_ref"]
                vs_rows = [list(r) for r in zip(prompts, completions, vs_completions)]
            else:
                vs_oracle_response = preference_oracle.consult_the_oracle(
                    prompts, [completions, vs_completions])

                for r in vs_oracle_response.rank:
                    if r is None:  # couldn't decide... choose one at random
                        vs_wins.append(random.choice([0, 1]))
                    else:
                        vs_wins.append(1 if r[0] == 0 else 0)
                # wins = [1 if r[0] == 0 else 0 for r in vs_oracle_response.rank]

                vs_rows_cols = ["prompt", "completion", "completion_ref", "win"]
                vs_rows = [list(r) for r in zip(
                    prompts, completions, vs_completions, vs_wins)]

                if vs_oracle_response.score is not None:
                    vs_rows_cols.extend(["score", "score_ref"])
                    for r, s in zip(vs_rows, vs_oracle_response.score):
                        r.extend([s[0], s[1]])

                if vs_oracle_response.rationale is not None:
                    vs_rows_cols.extend(["rationale"])
                    for r, s in zip(vs_rows, vs_oracle_response.rationale):
                        r.append(s)

    if scores is not None:
        scores = torch.Tensor(scores)
        results.update({
            f"{prefix}/pref_mean": torch.mean(scores).item(),
            f"{prefix}/pref_std": torch.std(scores).item(),
            f"{prefix}/sigma_pref_mean": torch.mean(torch.sigmoid(scores)).item(),
            f"{prefix}/pref_dist": torch.Tensor(scores).numpy(),
        })

    results.update({
        f"{prefix}/kl_mean": torch.mean(torch.Tensor(all_kl_divs)).item(),
        f"{prefix}/kl_dist": torch.Tensor(all_kl_divs).numpy(),
        f"{prefix}/logprobs_dist": torch.Tensor(all_logprobs).numpy(),
        f"{prefix}/ref_logprobs_dist": torch.Tensor(all_ref_logprobs).numpy(),
        f"{prefix}/logprobs_mean": torch.Tensor(all_logprobs).mean().item(),
        f"{prefix}/ref_logprobs_mean": torch.Tensor(all_ref_logprobs).mean().item(),
    })

    if versus is not None:
        if len(vs_wins) > 0:
            results[f"{prefix}/win_rate_vs_{versus}"] = sum(vs_wins) / len(vs_wins)
        results[f"{prefix}/vs_rows_vs_{versus}"] = wandb.Table(
            columns=vs_rows_cols,
            rows=vs_rows)

    if save_path is not None:
        o = dict(
            prompts=prompts,
            completions=completions,
            vs_completions=vs_completions,
        )

        if len(vs_wins) > 0:
            o["vs_wins"] = vs_wins

        with open(save_path, "w") as f:
            json_tricks.dump(o, f, indent=4)
        print(f"Written evaluation inputs and results to {save_path}")

    return results
