import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)
from trl.core import logprobs_from_logits
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
from tenacity import Retrying, stop_after_attempt, wait_fixed

from direct.config import *
from direct.types import TModel

MODEL_REGISTRY = {
    "gpt2_value": AutoModelForCausalLMWithValueHead,
    "gpt2": GPT2LMHeadModel,
    "gpt-neox": GPTNeoXForCausalLM,
}


def get_generative_model(config: ExperimentConfig) -> tuple[TModel, PreTrainedTokenizerBase]:
    print("Setting up generative model")
    model_class: TModel = MODEL_REGISTRY.get(config.model_class, GPT2LMHeadModel)

    for attempt in Retrying(wait=wait_fixed(10), stop=stop_after_attempt(5)):
        with attempt:
            model = model_class.from_pretrained(config.model_instance)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_instance)

    model.to(config.device)

    #  pad_token hack needed re: https://github.com/huggingface/transformers/issues/4122
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _get_logits_from_model_output(output):
    if hasattr(output, "logits"):
        return output.logits
    else:
        return output[0]


def batch_forward_pass(model: TModel,
                       ref_model: TModel,
                       tokenizer: PreTrainedTokenizerBase,
                       prompt_tokens: list[Tensor],
                       completion_tokens: list[Tensor]):
    assert(not hasattr(model, "is_encoder_decoder")), "not implemented for encoder-decoder models!"

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # n x num tokens
    input_ids: torch.Tensor = collator(
        [torch.cat([q, r]) for q, r in zip(prompt_tokens, completion_tokens)]
    )["input_ids"]

    # noinspection PyUnresolvedReferences
    assert len((input_ids[:, 0] == tokenizer.pad_token_id).nonzero()) == 0, \
        "Not expecting any left padding here - since we're not using an attention mask"

    model_output = model(input_ids)

    # n x num tokens x vocab size
    logits = _get_logits_from_model_output(model_output)

    # NOTE: the last element of logits is the completion for the token *after* the input_ids - which we're not
    # interested in. The model doesn't give us any logit corresponding to the first input token, so we drop that too
    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

    # Build a mask, so we can compute loss solely on the generated response - need to make sure we don't have an off
    # by one - e.g. if the query has 4 tokens, then the 3rd (0-indexed) logit corresponds to
    # the first token of the response
    response_mask = torch.zeros_like(logprobs)
    for i, (q, r) in enumerate(zip(prompt_tokens, completion_tokens)):
        response_mask[i, len(q) - 1: len(q) - 1 + len(r)] = 1.0

    ref_logprobs = None

    if ref_model is not None:
        # for KL penalty
        with torch.no_grad():
            ref_logits = _get_logits_from_model_output(ref_model(input_ids))
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])

    return logprobs, response_mask, ref_logprobs
