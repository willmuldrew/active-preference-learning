import itertools

import torch
import torch.utils.data
import wandb
from torch.utils.data import Subset

import direct.dpo_trainer
import direct.generate
import direct.ppo_trainer
from direct import utils as utils
from direct.config import ExperimentConfig
from direct.data import load_prompt_dataset, ActiveDataLoader
from direct.evaluate import evaluate_model
from direct.model import get_generative_model
from direct.oracles import get_preference_oracle


# Run DPO online vs offline
# i.e.
# online = generate a batch of completion pairs, ask oracle to rank them, train
#  vs
# offline = generate all pairs and rankings up front
#
# variants
#  a). unlimited labels - offline, generate from ref model, rank using oracle, consume and train
#                          online, generate from current model, rank using oracle, consume and train
#  b). limited labels   - offline, generate fixed number of pairs from ref model, rank using oracle, train (looping?)
#                          online, start with initial set of generations/labels, train with those, every N steps
#                          generate with current model and rank. When budget runs out, just loop
#                            * lots of arbitrary hyperparams to choose
#
#
# How to make "fair" -

# For active...
#  train with M initial labelled pairs for number of steps S
#  After S, every S_a steps, acquire B_a datapoints (smartly) and add to dataset
#

# S, S_a should correspond (possibly) to multiple passes through the available data
# may need to recalc S_a


@torch.no_grad()
def compute_model_uncertainty(gen_model, gen_ref_model, tokenizer, pairs, beta, batch_size, device, set_r_and_u=False):
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


def model_uncertainty(gen_model, gen_ref_model, tokenizer, pairs, beta, config):
    """
    Return pairs sorted in order of *increasing* model uncertainty according to its implicit reward function:
       r_hat(y|x) from the DPO paper is beta * ( sum(logprobs)  - sum(ref_logprobs)
    """

    all_scores = compute_model_uncertainty(gen_model, gen_ref_model, tokenizer, pairs, beta, config.train.batch_size, config.device, set_r_and_u=True)
    return [pairs[i] for i in torch.argsort(torch.Tensor(all_scores))]


def scores_baseline(pairs):
    """
    This is cheating... take a peek at the scores...
    """
    score_diff = torch.tensor([p["score_w"] - p["score_l"] for p in pairs])
    return [pairs[i] for i in torch.argsort(score_diff)]


def run(config: ExperimentConfig):
    print("Starting exp4 run [ACTIVE]")
    gen_model, gen_tokenizer = get_generative_model(config)
    preference_oracle = get_preference_oracle(config)

    gen_trainer = direct.model.DirectPreferenceTrainer(config, gen_model, gen_tokenizer)
    ref_model = gen_trainer.ref_model

    prompt_datasets = load_prompt_dataset(tokenizer=gen_tokenizer, config=config)

    completion_model = {
        "offline": ref_model,  # to get more data, generate it from the original model
        "active": gen_model,  # active+random is the same as online
    }[config.exp4.mode]

    generate_args = dict(config.generate_gpt2.to_kwargs())

    completion_pair_generator = direct.generate.CompletionSetGenerator(
        completion_model,
        gen_tokenizer,
        direct.utils.LengthSampler(*config.data.completion_len_range),
        generate_args,
        num_completions=2
    )

    # NB: we don't use a custom collate function - we just want to batch up the string prompts rather than
    # make things messy with pre-tokenized/collated batches... keep it simple
    prompt_dataloader = torch.utils.data.DataLoader(
        prompt_datasets["train"],
        config.train.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_prompt_dataset = Subset(prompt_datasets["test"], range(config.eval.test_set_size))

    class MyActiveDataLoader(ActiveDataLoader):
        def __init__(self):
            super().__init__(
                config.exp4.initial_m, config.exp4.max_m,
                prompt_dataloader, completion_pair_generator,
                preference_oracle,
                config.train.effective_batch_size,
                config,
                over_generate_factor=config.exp4.over_generate_factor,
                greedily_get_preferences=config.exp4.rank_pair_fn in ["scores"],
                log_acquired_data_path=(
                    f"{wandb.run.dir}/active_training_data.jsonl" if wandb.run is not None
                    else "active_training_data.jsonl"),
            )
            # self.loss_ma_high_water_mark = 0.0
            # self.latest_loss_ma = 0.0
            self.early_stopper = self.initialise_early_stopper()

        # noinspection PyMethodMayBeStatic
        def initialise_early_stopper(self):
            return utils.LossMAEarlyStopper()

        def select_completion_pairs(self, count, pairs):
            ranked_pairs = dict(
                random=lambda: utils.shuffled(pairs),
                uncertainty=lambda: model_uncertainty(
                    gen_model, ref_model, gen_tokenizer, pairs, config.train.dpo.beta, config),
                certainty=lambda: list(reversed(
                    model_uncertainty(gen_model, ref_model, gen_tokenizer, pairs, config.train.dpo.beta, config))),
                scores=lambda: scores_baseline(pairs),
            )[config.exp4.rank_pair_fn]()
            return ranked_pairs[-count:]

        def after_acquire_data(self):
            if config.exp4.reset_after_acquisition:
                print("Resetting model/optimiser after data acquisition")

                # force an eval before we reset and retrain
                print("Running eval")
                eval_stats = {}

                for t in config.eval.sampling_temperatures:
                    eval_stats.update(
                        evaluate_model(
                            gen_model,
                            gen_trainer.ref_model,
                            gen_tokenizer,
                            test_prompt_dataset,
                            preference_oracle,
                            config,
                            include_samples=True,
                            sample_temperature=t,
                            prefix=f"eval_T{t:0.2f}",
                            vs_model=gen_trainer.ref_model if config.eval.vs_ref_model_eval else None,
                        )
                    )

                print(f"Pre-reset eval")
                for t in config.eval.sampling_temperatures:
                    utils.print_stats(eval_stats, f"T={t}: ",
                                      [f"eval_T{t:0.2f}/{k}"
                                       for k in ["pref_mean", "sigma_pref_mean", "pref_std", "win_rate", "kl_mean"]])

                # TODO - we want to save out some checkpoints before each reset

                # Now do the reset
                gen_model.load_state_dict(ref_model.state_dict())
                gen_trainer.reset_optimizer()
                self.early_stopper = self.initialise_early_stopper()

        def acquire_batch_count(self, epoch):
            if config.exp4.defer_acquisition_using_loss:
                if self.early_stopper.should_stop():
                    return 1
                else:
                    return 0
            else:
                return 1

        def post_training_step(self, step, stats):
            # loss_ma = stats["direct/loss_ma"]
            # if loss_ma > self.loss_ma_high_water_mark:
            #     self.loss_ma_high_water_mark = loss_ma
            # self.latest_loss_ma = loss_ma
            self.early_stopper.update(stats)

            return self.should_stop()

        def post_eval_fn(self, step, eval_stats):
            pass

        def should_stop(self):
            return len(self.labelled_data) == self.max_m and self.early_stopper.should_stop()

    train_dataloader = MyActiveDataLoader()

    def step_fn(_step, _batch):
        stats, scores = gen_trainer.step_pairs(_batch[0], config.train.grad_acc_steps)
        stats["active/current_m"] = train_dataloader.get_current_m()
        return stats, scores

    def post_eval_fn(step, eval_stats):
        train_dataloader.post_eval_fn(step, eval_stats)

    def post_step_fn(_step, _stats):
        return train_dataloader.post_training_step(_step, _stats)

    train_loop(config,
               [train_dataloader],
               gen_trainer,
               gen_model,
               gen_tokenizer,
               test_prompt_dataset,
               preference_oracle,
               step_fn,
               post_eval_fn=post_eval_fn,
               post_step_fn=post_step_fn,
               )


# TODO: may want to factor this out...
def train_loop(config, train_dataloaders,
               gen_trainer, gen_model, gen_tokenizer, test_prompt_dataset,
               preference_oracle, step_fn,
               post_eval_fn=None,
               post_step_fn=None,
               ):

    scheduler = torch.optim.lr_scheduler.LinearLR(gen_trainer.optimizer,
                                                  start_factor=config.train.lr_ramp_start_factor,
                                                  total_iters=config.train.lr_ramp_total_iters)

    i_train_dataloaders = [iter(itertools.cycle(dl)) for dl in train_dataloaders]

    loss_ma = None
    loss_ma_alpha = 0.9

    for step in range(config.steps):
        batches = [next(itr) for itr in i_train_dataloaders]

        logs = dict()

        stats, scores = step_fn(step, batches)

        logs.update(stats)
        logs.update({
            "lr": scheduler.get_last_lr(),
            "step": step,
        })

        if (step + 1) % config.eval.interim_eval_interval_steps == 0:
            print("Running eval")
            eval_stats = {}

            for t in config.eval.sampling_temperatures:
                eval_stats.update(
                    evaluate_model(
                        gen_model,
                        gen_trainer.ref_model,
                        gen_tokenizer,
                        test_prompt_dataset,
                        preference_oracle,
                        config,
                        include_samples=True,
                        sample_temperature=t,
                        prefix=f"eval_T{t:0.1f}",
                        vs_model=gen_trainer.ref_model if config.eval.vs_ref_model_eval else None,
                    )
                )

            logs.update(eval_stats)
            print(f"Eval (step={step})")
            for t in config.eval.sampling_temperatures:
                utils.print_stats(logs, f"T={t}: ",
                                  [f"eval_T{t:0.1f}/{k}"
                                   for k in ["pref_mean", "sigma_pref_mean", "pref_std", "win_rate", "kl_mean"]])

            if post_eval_fn is not None:
                post_eval_fn(step, eval_stats)

        utils.print_stats(logs, f"Step {step}/{config.steps}, stats: ",
                          [gen_trainer.loss_property_name,
                           "env/pref_mean",
                           "env/pref_std",
                           "env/kl_mean",
                           "env/sigma_pref_mean",
                           "lr"])

        scheduler.step(step)

        if loss_ma is None:
            loss_ma = stats["direct/loss/total"]
        else:
            loss_ma = stats["direct/loss/total"] * (1.0 - loss_ma_alpha) + loss_ma_alpha * loss_ma

        stats["direct/loss_ma"] = loss_ma

        if config.log:
            wandb.log(logs)

        if post_step_fn is not None:
            if post_step_fn(step, stats):
                break

        # print(f"loss_ma: {loss_ma}")
        # if loss_ma < 0.05:
        #     print("loss_ma hit early-stopping criterion")
        #     break
