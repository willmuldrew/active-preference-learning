from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from direct.config import ExperimentConfig
from direct.utils import LengthSampler

EXCLUDE_TLDR_SUBREDDITS = set("offmychest tifu".split())


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
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["prompt"]])

        datasets[split] = dataset

    return datasets
