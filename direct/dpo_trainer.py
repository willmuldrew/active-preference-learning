from typing import Optional

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling
from trl import create_reference_model

import direct.model
from direct.config import ExperimentConfig
from direct.types import TModel


class DirectPreferenceTrainer:
    def __init__(
            self,
            config: ExperimentConfig,
            model: TModel,
            tokenizer: PreTrainedTokenizerBase,
            ref_model: Optional[TModel] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.optimizer = self.initialise_optimizer(model)
        # self.prompt_size_sampler = utils.LengthSampler(*config.data.prompt_len_range)
        # self.completion_size_sampler = utils.LengthSampler(*config.data.completion_len_range)
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.ref_model = ref_model
        self.loss_property_name = "direct/loss/total"
        self.beta = config.train.dpo.beta

        if self.ref_model is None:
            # Use trl function to create our reference model (copy and disable gradients)
            # noinspection PyTypeChecker
            self.ref_model = create_reference_model(self.model)

    def reset_optimizer(self):
        self.optimizer = self.initialise_optimizer(self.model)

    def initialise_optimizer(self, model):
        # return bnb.optim.PagedAdam8bit(model.parameters(), lr=self.config.train.lr)
        # return bnb.optim.RMSprop8bit(model.parameters(), lr=self.config.train.lr)
        # TODO: investigate other optimisers - DPO reference code suggests that RMSProp can be used without significant
        #       impact
        if self.config.train.optimizer == "Adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.train.lr)
        elif self.config.train.optimizer == "PagedAdam8bit":
            import bitsandbytes as bnb
            return bnb.optim.PagedAdam8bit(model.parameters(), lr=self.config.train.lr)
        else:
            raise NotImplementedError(f"Don't know how to create a {self.config.train.optimizer} instance")


    def loss(self,
             logprobs_w: Tensor, response_mask_w: Tensor, ref_logprobs_w: Tensor,
             logprobs_l: Tensor, response_mask_l: Tensor, ref_logprobs_l: Tensor,
             ) -> Tensor:

        # Use the mask to only use the completion tokens to compute gradients (ignoring the prompt and any padding)
        if self.config.train.loss_fn == "dpl":
            # Rasch style objective
            return -self.log_sigmoid(
                ((logprobs_w * response_mask_w).sum(dim=-1) / response_mask_w.sum(dim=-1))
                - ((logprobs_l * response_mask_l).sum(dim=-1) / response_mask_l.sum(dim=-1))
            ).mean()
        elif self.config.train.loss_fn == "dpo":
            # https://arxiv.org/pdf/2305.18290.pdf
            log_pi_w = (logprobs_w * response_mask_w).sum(dim=-1)
            log_pi_l = (logprobs_l * response_mask_l).sum(dim=-1)

            log_pi_ref_w = (ref_logprobs_w * response_mask_w).sum(dim=-1)
            log_pi_ref_l = (ref_logprobs_l * response_mask_l).sum(dim=-1)

            return -self.log_sigmoid(
                self.beta * ((log_pi_w - log_pi_ref_w) - (log_pi_l - log_pi_ref_l))
            ).mean()
        else:
            raise ValueError(f"Unknown loss function {self.config.train.loss_fn}")

    def fwd_pass(self, query_tokens, response_tokens):
        return direct.model.batch_forward_pass(
            self.model, self.ref_model, self.tokenizer,
            [t.to(self.config.device) for t in query_tokens],  # list of ragged prompts, no padding
            [t.to(self.config.device) for t in response_tokens],  # list of ragged responses, no padding
            use_cache=not self.model.is_gradient_checkpointing
        )

    def step_pairs(self, batch_pairs, grad_acc_steps: int):
        sub_batch_size = len(batch_pairs["prompt"]) // grad_acc_steps

        kls = []
        losses = []

        self.optimizer.zero_grad()
        mean_logp_w, mean_logp_l, mean_ref_logp_w, mean_ref_logp_l = 0.0, 0.0, 0.0, 0.0

        for grad_acc_step in range(grad_acc_steps):
            batch_start, batch_end = grad_acc_step * sub_batch_size, (grad_acc_step + 1) * sub_batch_size
            logprobs_w, response_mask_w, ref_logprobs_w = self.fwd_pass(
                batch_pairs["prompt_tokens"][batch_start:batch_end],
                batch_pairs["completion_tokens_w"][batch_start:batch_end]
            )
            logprobs_l, response_mask_l, ref_logprobs_l = self.fwd_pass(
                batch_pairs["prompt_tokens"][batch_start:batch_end],
                batch_pairs["completion_tokens_l"][batch_start:batch_end],
            )

            with torch.no_grad():
                mean_logp_w += ((logprobs_w * response_mask_w).sum() / response_mask_w.sum()).item() / grad_acc_steps
                mean_logp_l += ((logprobs_l * response_mask_l).sum() / response_mask_l.sum()).item() / grad_acc_steps

                mean_ref_logp_w += \
                    ((ref_logprobs_w * response_mask_w).sum() / response_mask_w.sum()).item() / grad_acc_steps
                mean_ref_logp_l += \
                    ((ref_logprobs_l * response_mask_l).sum() / response_mask_l.sum()).item() / grad_acc_steps

            loss = self.loss(
                logprobs_w=logprobs_w, response_mask_w=response_mask_w, ref_logprobs_w=ref_logprobs_w,
                logprobs_l=logprobs_l, response_mask_l=response_mask_l, ref_logprobs_l=ref_logprobs_l,
            ) / grad_acc_steps

            with torch.no_grad():
                kls.append((((logprobs_w - ref_logprobs_w) * response_mask_w).sum(dim=1).mean()).item())
                kls.append((((logprobs_l - ref_logprobs_l) * response_mask_l).sum(dim=1).mean()).item())

            loss.backward()
            losses.append(loss.item())
            # bit hacky, but keeps a lid on VRAM usage
            torch.cuda.empty_cache()

        self.optimizer.step()
        self.optimizer.zero_grad()

        stats = {
            self.loss_property_name: np.sum(losses),  # sum since we've already divided by grad_acc_steps
            "env/kl_mean": np.mean(kls),
            "env/mean_logp_w": mean_logp_w,
            "env/mean_logp_l": mean_logp_l,
            "env/mean_ref_logp_w": mean_ref_logp_w,
            "env/mean_ref_logp_l": mean_ref_logp_l,
        }

        if "score_w" in batch_pairs:
            scores = torch.Tensor(batch_pairs["score_w"] + batch_pairs["score_l"])
            stats.update({
                "env/pref_mean": torch.mean(scores).item(),
                "env/sigma_pref_mean": torch.mean(torch.sigmoid(scores)).item(),
                "env/pref_std": torch.std(scores).item(),
                "env/pref_dist": scores.numpy(),
            })
        else:
            scores = None

        # Need to return all scores - not just the cherry-picked winners! (TODO: sure?)
        return stats, scores
