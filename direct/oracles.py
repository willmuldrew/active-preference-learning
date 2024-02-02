import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from tenacity import retry, wait_fixed, stop_after_attempt
from transformers import pipeline

import direct.openai_ranking as openai_ranking
from direct import utils
from direct.config import ExperimentConfig
from direct.utils import argsort, JSONFileLogger


@dataclass
class OracleResponse:
    source: str
    rank: list[list[int]]
    score: Optional[list[list[float]]] = None
    rationale: Optional[list[str]] = None


class PreferenceOracle(ABC):
    @abstractmethod
    def consult_the_oracle(self, prompt_strs: list[str], response_strs: list[list[str]], random_tie_break: bool = True) -> OracleResponse:
        pass

    def get_scores(self, prompts: list[str], completions: list[str]):
        raise NotImplementedError("This oracle doesn't not implement scores")


class DummyStringLengthPreferenceOracle(PreferenceOracle):
    def get_scores(self, prompts: list[str], completions: list[str]):
        raise NotImplementedError("This oracle doesn't not implement scores")

    def __init__(self, target_length):
        self.target_length = target_length

    def consult_the_oracle(self, prompt_strs: list[str], response_strs: list[list[str]], random_tie_break: bool = True) -> OracleResponse:
        rank = []
        for n in range(len(prompt_strs)):
            rank.append(
                argsort([r[n] for r in response_strs], key=lambda resp_str: (len(resp_str.strip()) - self.target_length) ** 2.0))
        return OracleResponse(rank=rank, source="DummyStringLengthPreferenceOracle")


class OpenAIPreferenceOracle(PreferenceOracle):
    def __init__(self, pref_instance, request_logger, num_threads, provider):
        task_name, model_name = pref_instance.split("-", 1)

        if task_name not in ["tldr", "imdb"]:
            raise ValueError(f"Don't know how to create oracle for {pref_instance}")

        if model_name not in openai_ranking.OPENAI_PRICING:
            raise ValueError(f"Don't know about openai model {model_name}")

        self.model_name = model_name
        self.request_logger = request_logger
        self.task_name = task_name
        self.num_threads = num_threads
        self.provider = provider

    def consult_the_oracle(self, prompt_strs: list[str], response_strs: list[list[str]], random_tie_break=True) -> OracleResponse:
        assert (len(response_strs) == 2), "Only implemented for pairs"
        rank: list[list[int]] = []
        rationale: list[str] = []

        batch = [dict(prompt=p, completion_a=r_a, completion_b=r_b) for p, r_a, r_b in zip(prompt_strs, response_strs[0], response_strs[1])]
        responses = openai_ranking.get_preference_batch(batch, self.model_name, self.request_logger, self.num_threads, task_name=self.task_name, provider=self.provider)

        for resp in responses:
            best = resp.get("preferred")
            if best is None and random_tie_break:
                print("WARNING: oracle did not assign preferred - doing random tie-break")
                best = random.choice([0, 1])

            if best is None:
                print("WARNING: oracle did not assign preferred")
                rank.append(None)
            else:
                rank.append([best, (best + 1) % 2])
            rationale.append(resp.get("comparison", "No rationale provided"))
        return OracleResponse(rank=rank, rationale=rationale, source=self.model_name)  # we don't have a scalar score to return

    def get_scores(self, prompts: list[str], completions: list[str]):
        raise NotImplementedError("This oracle doesn't not implement scores")


class LocalHuggingFacePreferenceOracle(PreferenceOracle):
    def __init__(self, pref_class, pref_instance, device, batch_size):
        @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
        def initialise_pipeline():
            return pipeline(task=pref_class, model=pref_instance, device=device)

        self.pipeline = initialise_pipeline()
        self.batch_size = batch_size
        self.oracle_source = f"{pref_class}/{pref_instance}"

    def get_scores(self, prompts, responses):
        full_strs = [p + r for p, r in zip(prompts, responses)]
        scores = []
        for pref in self.pipeline(full_strs, top_k=None, function_to_apply="none", batch_size=self.batch_size):
            for label_score in pref:
                if label_score["label"] == "POSITIVE":
                    scores.append(label_score["score"])
                    break
            else:
                raise ValueError("could not find POSITIVE label in preference model output")

        assert len(scores) == len(full_strs)
        return scores

    def consult_the_oracle(self, prompt_strs: list[str], response_strs: list[list[str]], random_tie_break: bool = True) -> OracleResponse:
        assert (len(response_strs) == 2), "only handling pairs for now"

        scores_a = self.get_scores(prompt_strs, response_strs[0])
        scores_b = self.get_scores(prompt_strs, response_strs[1])
        all_ranks = []
        all_scores = []
        for n in range(len(prompt_strs)):
            all_scores.append([scores_a[n], scores_b[n]])
            all_ranks.append(utils.argsort([-scores_a[n], -scores_b[n]]))

        return OracleResponse(rank=all_ranks, score=all_scores, source=self.oracle_source)


def get_preference_oracle(config: ExperimentConfig) -> PreferenceOracle:
    if (config.pref_model_class == "dummy" and
            (config.pref_model_instance.startswith("tldr-wordcount") or
             config.pref_model_instance.startswith("imdb-wordcount"))):
        print("Using dummy wordcount preference 'oracle'")
        return DummyStringLengthPreferenceOracle(42)
    elif config.pref_model_class == "openai":
        print("Using external openai preference oracle")
        return OpenAIPreferenceOracle(config.pref_model_instance, JSONFileLogger("openai-request-log.jsonl"), config.exp5.num_openai_threads, config.exp5.openai_provider)
    else:
        # otherwise, assume it's a local model we can load from huggingface
        print(f"Using local preference model {config.pref_model_class}/{config.pref_model_instance}")
        return LocalHuggingFacePreferenceOracle(
            config.pref_model_class, config.pref_model_instance, config.device, batch_size=config.train.batch_size)


def apply_preferences_to_completion_set(completion_sets: list[dict], oracle_response: OracleResponse) -> None:
    assert len(completion_sets) == len(oracle_response.rank), "Lengths need to match"

    for p, r in zip(completion_sets, oracle_response.rank):
        p["oracle_source"] = oracle_response.source
        p["rank"] = r
        if r is not None:
            if len(r) != 2:
                raise NotImplementedError("Only really implemented for pairs at the moment - though could be extended")

            p["completion_w"] = p[f"completion_{r[0]}"]
            p["completion_l"] = p[f"completion_{r[1]}"]
            p["completion_tokens_w"] = p[f"completion_tokens_{r[0]}"]
            p["completion_tokens_l"] = p[f"completion_tokens_{r[1]}"]

    if oracle_response.score is not None:
        for p, s, r in zip(completion_sets, oracle_response.score, oracle_response.rank):
            p["score"] = s
            if r is not None:
                p["score_w"] = p["score"][r[0]]
                p["score_l"] = p["score"][r[1]]

    if oracle_response.rationale is not None:
        for p, r in zip(completion_sets, oracle_response.rationale):
            p["rationale"] = r

