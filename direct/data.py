import itertools
import json
import random
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

from direct import utils as utils
from direct.config import ExperimentConfig
from direct.generate import CompletionSetGenerator
from direct.oracles import PreferenceOracle, apply_preferences_to_completion_set
from direct.utils import LengthSampler

EXCLUDE_TLDR_SUBREDDITS = set("offmychest tifu".split())


# def get_prompt_collator(pad_token_id: int) -> Callable[List[dict], dict]:
#     """
#     Batch collator function - converts ragged collection of query_tokens into left-padded 2D tensors suitable
#     for batch generation
#     """
#     def batch_collator(data: List[dict]) -> dict:
#         res = dict((key, [datum[key] for datum in data]) for key in data[0])
#         if torch.is_tensor(res["prompt_tokens"][0]):
#             flipped_query_tokens = [q.flip(dims=(0, )) for q in res["prompt_tokens"]]
#         else:
#             flipped_query_tokens = [torch.LongTensor(q[::-1]) for q in res["prompt_tokens"]]
#
#         res["prompt_tokens"] = pad_sequence(
#             flipped_query_tokens,
#             batch_first=True,
#             padding_value=pad_token_id).flip(dims=[1])
#
#         res["attention_mask"] = (res["prompt_tokens"] != pad_token_id).long()
#         return res
#
#     return batch_collator


def load_prompt_dataset(tokenizer: PreTrainedTokenizerBase, config: ExperimentConfig) -> dict[str, Dataset[dict]]:
    datasets = {}
    for split, limit_n in [("train", config.data.limit_train_n),
                           ("test", config.data.limit_test_n)]:

        if config.data.dataset_name == "imdb":
            dataset = load_dataset("imdb", split=split)
            dataset = dataset.rename_columns({"text": "prompt"})
            dataset = dataset.filter(lambda x: len(x["prompt"]) > 200, batched=False)
        elif config.data.dataset_name == "tldr":
            # NOTE: Some interesting facts about this dataset....
            #  * in the train split - max prompt length in gpt2 tokens ~= 510
            #                       - max string length ~= 2500
            #  * ~ 5 chars per token
            dataset = load_dataset("CarperAI/openai_summarize_tldr", split=split)
            # TODO: consider if we want to filter like this... but may be good for debugging
            dataset = dataset.filter(lambda x: 1000 > len(x["prompt"]) > 200, batched=False)

            def strip_trailing_spaces(x):
                x['prompt'] = x['prompt'].rstrip(" ")
                return x

            dataset = dataset.map(strip_trailing_spaces, num_proc=config.data.num_proc)

            # TODO: bit queasy about sending unfiltered reddit to openai on my personal account!
            dataset = dataset.filter(
                lambda x: x["prompt"].split("\n")[0].split()[1].split("/")[1].lower() not in EXCLUDE_TLDR_SUBREDDITS,
                batched=False)
        else:
            raise ValueError(f"Unsupported dataset {config.data.dataset_name}")

        if limit_n is not None:
            dataset = dataset.select(range(limit_n))

        prompt_length_sampler = LengthSampler(*config.data.prompt_len_range) if config.data.truncate_prompts else None

        def map_fn(samples: dict):
            # TODO - consider just truncating on word count... avoids need for tokenization here
            prompt_tokens = [p for p in tokenizer(samples["prompt"], truncation=True)["input_ids"]]
            # NOTE - this token truncation approach can fail to do exactly what you want - e.g.
            # ". . . is just" -> a 2 token truncated seq (if random len = 2) -> "..." -> 1 token seq
            if config.data.truncate_prompts:
                prompt_tokens = [p[:prompt_length_sampler()] for p in prompt_tokens]
            return dict(
                prompt=tokenizer.batch_decode(prompt_tokens),
                prompt_tokens=prompt_tokens
            )

        dataset = dataset.map(map_fn, batched=True, num_proc=config.data.num_proc)

        # dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["prompt", "prompt_tokens"]])
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["prompt"]])

        datasets[split] = dataset

    return datasets


class ActiveDataLoader:
    def __init__(self,
                 initial_m: int,
                 max_m: int,
                 prompt_dataloader: DataLoader,
                 pair_generator: CompletionSetGenerator,
                 preference_oracle: PreferenceOracle,
                 batch_size: int,
                 config: ExperimentConfig,
                 over_generate_factor: int,
                 greedily_get_preferences: bool,  # needed if for baseline acquisition filters like "score"
                 log_acquired_data_path: Optional[str] = None,
                 ):
        self.initial_m = initial_m
        self.i_prompt_dataloader = iter(itertools.cycle(prompt_dataloader))
        self.pair_generator = pair_generator
        self.config = config
        self.max_m = max_m
        self.batch_size = batch_size
        self.over_generate_factor = over_generate_factor
        self.preference_oracle = preference_oracle
        self.greedily_get_preferences = greedily_get_preferences
        self.log_acquired_data_path = log_acquired_data_path
        self.acquisition_phase = 0

        self.labelled_data = []

        assert (initial_m % batch_size == 0), f"initial_m {initial_m} is not a multiple of batch_size {batch_size}"

        print(f"Acquiring initial dataset of size {initial_m}")
        self._acquire_batches(initial_m // batch_size, False)

    def _acquire_batches(self, num_batches, rank):
        final_required_count = num_batches * self.batch_size
        sample_count = final_required_count * (self.over_generate_factor if rank else 1)
        print(f"Acquiring {final_required_count} datapoints - sampling {sample_count} datapoints")

        candidate_pairs = []

        while len(candidate_pairs) < sample_count:
            candidate_pairs.extend(
                utils.cols_to_records(
                    self.pair_generator.generate(next(self.i_prompt_dataloader), self.config.device, "cpu")
                ))

        if self.greedily_get_preferences:
            self.apply_preference_oracle(candidate_pairs)

        candidate_pairs = candidate_pairs[:sample_count]  # just in case we have any excess items
        selected_pairs = self.select_completion_pairs(final_required_count, candidate_pairs)

        if not self.greedily_get_preferences:
            self.apply_preference_oracle(selected_pairs)

        self.maybe_log_acquired_data(self.acquisition_phase, selected_pairs)
        self.labelled_data.extend(selected_pairs)
        self.acquisition_phase += 1

    def get_current_m(self):
        return len(self.labelled_data)

    def after_acquire_data(self):
        pass

    def select_completion_pairs(self, count, pairs):
        return pairs[-count:]

    def acquire_batch_count(self, epoch):
        """
        Called at the end of each epoch so that subclasses can choose how many batches (if any) to acquire
        """
        return 1

    def apply_preference_oracle(self, pairs):
        oracle_response = self.preference_oracle.consult_the_oracle(
            [p['prompt'] for p in pairs], [[p[f"completion_{n}"] for p in pairs] for n in [0, 1]])
        apply_preferences_to_completion_set(pairs, oracle_response)

    def __iter__(self):
        epoch = 0
        while True:
            for b in range(len(self.labelled_data) // self.batch_size):
                yield utils.records_to_cols(self.labelled_data[b*self.batch_size:(b+1)*self.batch_size])
            epoch += 1
            if self.get_current_m() < self.max_m:
                acquire_batch_count = self.acquire_batch_count(epoch)
                if acquire_batch_count > 0:
                    current_m = self.get_current_m()
                    acquire_n = min(acquire_batch_count, (self.max_m - current_m) // self.batch_size)
                    if acquire_n > 0:
                        self._acquire_batches(acquire_n, True)
                        print(f"Acquired {acquire_n} batches (m={current_m} -> {self.get_current_m()})")
                        self.after_acquire_data()
                    else:
                        print("Looped through data - but hit max_m")
            random.shuffle(self.labelled_data)

    def maybe_log_acquired_data(self, acquisition_phase: int, new_data: list[dict]) -> None:
        if self.log_acquired_data_path is None:
            return

        print(f"Logging {len(new_data)} rows of acquired data -> {self.log_acquired_data_path}")
        with open(self.log_acquired_data_path, "at", encoding="utf-8") as f:
            for p in new_data:
                p2 = {k: v for k, v in p.items() if not isinstance(v, torch.Tensor)}
                p2["acquisition_phase"] = acquisition_phase
                json.dump(p2, f)
                f.write("\n")
