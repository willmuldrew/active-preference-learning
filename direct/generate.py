from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizerBase

import direct.utils
from direct import oracles
from direct.utils import LengthSampler
from direct.oracles import PreferenceOracle
from direct.types import TModel


@dataclass
class GenerateBatchOutput:
    prompt: list[str]
    prompt_tokens: list[Tensor]
    completion_tokens: list[Tensor]
    completion: list[str]
    full_sequence_str: list[str]


def generate(model: TModel,
             tokenizer: PreTrainedTokenizerBase,
             prompts: list[str],
             gen_lens: Tensor,
             generate_kwargs: dict,
             device,
             do_sample: bool = True) -> GenerateBatchOutput:
    # It's a bit wasteful to keep tokenizing and collating, but it keeps the code a _lot_ simpler...
    tok_prompts = tokenizer(prompts)
    tok_prompts["prompt_tokens"] = tok_prompts["input_ids"]
    del tok_prompts["input_ids"]
    gen_batch = _collate_for_generation(direct.utils.cols_to_records(tok_prompts), tokenizer.pad_token_id)

    return _generate_batch(
        model,
        tokenizer,
        gen_batch["prompt_tokens"].to(device),
        gen_batch["attention_mask"].to(device),
        gen_lens,
        generate_kwargs,
        do_sample=do_sample
    )


def _generate_batch(model: TModel,
                    tokenizer: PreTrainedTokenizerBase,
                    prompt_batch: Tensor,
                    attention_mask: Tensor,
                    gen_lens: Tensor,
                    generate_kwargs: dict,
                    do_sample: bool = True) -> GenerateBatchOutput:
    # Although GPT2 has an absolute positional encoding, this PR made it so that generation using the attention
    # mask ensures that the position is computed from the correct location:
    # https://github.com/huggingface/transformers/pull/7552

    generated_batch = model.generate(input_ids=prompt_batch,
                                     attention_mask=attention_mask,
                                     max_new_tokens=max(gen_lens),
                                     do_sample=do_sample,
                                     pad_token_id=tokenizer.pad_token_id,
                                     **generate_kwargs)
    batch_len = prompt_batch.shape[0]

    all_completions_tokens = []
    all_completions_str = []
    all_full_sequences_str = []
    all_prompt_tokens = []
    all_prompts = []

    for n in range(batch_len):
        completion_tokens: torch.LongTensor = generated_batch[n][
                                              attention_mask.shape[1]:attention_mask.shape[1] + gen_lens[n]]
        # truncate after the first end of text token
        eos_ids = (completion_tokens == tokenizer.eos_token_id).nonzero()
        if len(eos_ids) > 1:
            completion_tokens = completion_tokens[:eos_ids[1]]

        prompt_tokens = prompt_batch[n][attention_mask[n].nonzero(as_tuple=True)]
        assert prompt_tokens.dim() == 1

        prompt_str = tokenizer.decode(prompt_tokens)
        completion_str = tokenizer.decode(completion_tokens)

        all_completions_str.append(completion_str)
        all_completions_tokens.append(completion_tokens)
        all_full_sequences_str.append(prompt_str + completion_str)
        all_prompt_tokens.append(prompt_tokens)
        all_prompts.append(prompt_str)

    return GenerateBatchOutput(
        prompt=all_prompts,
        prompt_tokens=all_prompt_tokens,
        completion_tokens=all_completions_tokens,
        completion=all_completions_str,
        full_sequence_str=all_full_sequences_str,
    )


def _collate_for_generation(data: list[dict], pad_token_id) -> dict:
    res = dict((key, [datum[key] for datum in data]) for key in data[0])
    if torch.is_tensor(res["prompt_tokens"][0]):
        flipped_query_tokens = [q.flip(dims=(0,)) for q in res["prompt_tokens"]]
    else:
        flipped_query_tokens = [torch.LongTensor(q[::-1]) for q in res["prompt_tokens"]]

    res["prompt_tokens"] = pad_sequence(
        flipped_query_tokens,
        batch_first=True,
        padding_value=pad_token_id).flip(dims=[1])

    # noinspection PyUnresolvedReferences
    res["attention_mask"] = (res["prompt_tokens"] != pad_token_id).long()
    return res


class CompletionSetGenerator:
    def __init__(self, model, tokenizer, output_size_sampler, generate_kwargs, num_completions):
        self.output_size_sampler = output_size_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.num_completions = num_completions

    @torch.no_grad()
    def generate(self, batch: dict, device: str, dest_device: Optional[str] = None) -> dict:
        completions = [generate(
            self.model,
            self.tokenizer,
            batch["prompt"],
            self.output_size_sampler(len(batch["prompt"])),
            self.generate_kwargs,
            device) for _ in range(self.num_completions)]

        res = dict(
            prompt=batch["prompt"],
            prompt_tokens=[t.to(dest_device) for t in completions[0].prompt_tokens]
        )
        for n in range(self.num_completions):
            res[f"completion_{n}"] = completions[n].completion
            res[f"completion_tokens_{n}"] = [t.to(dest_device) for t in completions[n].completion_tokens]

        return res


class RankedCompletionSetGenerator:
    def __init__(self,
                 model: TModel,
                 tokenizer: PreTrainedTokenizerBase,
                 preference_oracle: PreferenceOracle,
                 output_size_sampler: LengthSampler,
                 generate_kwargs: dict,
                 num_completions: int):

        self.num_completions = num_completions
        self.completion_set_generator = CompletionSetGenerator(
            model, tokenizer, output_size_sampler, generate_kwargs, self.num_completions)
        self.preference_oracle = preference_oracle

    def generate(self, prompt_batch: dict, device: str, dest_device: str):
        return self.apply_rank(self.completion_set_generator.generate(prompt_batch, device, dest_device))

    @torch.no_grad()
    def apply_rank(self, completion_sets):
        completion_sets = completion_sets.copy()
        oracle_response = self.preference_oracle.consult_the_oracle(
            completion_sets["prompt"], [completion_sets[f"completion_{n}"] for n in range(self.num_completions)])
        oracles.apply_preferences_to_completion_set(completion_sets, oracle_response)
        return completion_sets
