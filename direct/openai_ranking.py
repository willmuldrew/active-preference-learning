import os
import random
import traceback
import json
from multiprocessing.pool import ThreadPool

import openai
import time
import tqdm

REQUEST_TIMEOUT = 60
BACKOFF_SECS = 30

OPENAI_PRICING = {
    # gpt3.5 is 20x cheaper than gpt-4
    "gpt-3.5-turbo": {  # 4k
        "prompt_tokens": 0.0015 / 1000,
        "completion_tokens": 0.002 / 1000,
    },
    "gpt-4": {
        "prompt_tokens": 0.03 / 1000,
        "completion_tokens": 0.06 / 1000,
    },
}

OPENAI_PRICING["gpt-4-0613"] = OPENAI_PRICING["gpt-4"]
OPENAI_PRICING["gpt-3.5-turbo-0613"] = OPENAI_PRICING["gpt-3.5-turbo"]


def preferred_letter_to_index(v: str):
    v = v.strip().upper()

    # I've seen randomly quoted response letters
    v = v.replace('"', '')
    v = v.replace("'", "")

    if v not in "AB":
        print(f"WARNING: openai model returned unexpected output {v}")
        return None
    else:
        return "AB".index(v)


def _apply_azure_config():
    openai.api_type = 'azure'
    # openai.api_base = 'https://willm-france-central.openai.azure.com/'
    openai.api_base = 'https://willm-us-east.openai.azure.com/'
    openai.api_version = '2023-06-01-preview'
    with open(os.environ["HOME"] + "/.openai-azure-gsa-us-east", "rt") as f:
        openai.api_key = f.read().strip()
    openai.api_key_path = None


def _apply_openai_config():
    openai.api_type = 'open_ai'
    openai.api_base = 'https://api.openai.com/v1'
    openai.api_version = None
    openai.api_key_path = os.environ["HOME"] + "/.openai"
    openai.api_key = None


def get_preference_batch(batch, model, request_logger, num_threads, task_name, oracle_temperature=0.05, provider="azure") -> list[dict]:
    desc = f"Getting preferences of batch of {len(batch)} using {num_threads} threads"

    with ThreadPool(num_threads) as pool:
        return list(
            tqdm.tqdm(
                pool.imap(
                    lambda x: get_preference(x["prompt"], x["completion_a"], x["completion_b"], task_name, model, request_logger, oracle_temperature=oracle_temperature, provider=provider), batch),
                total=len(batch), desc=desc))


def get_imdb_prompt(prompt, completion_a, completion_b):
    return (f"""\
Which of the following movie reviews is better? The best one will be the one with the most positive sentiment, which also is grammatically correct, consistent, and avoids repetition.

Review A:
{prompt} {completion_a}

Review B:
{prompt} {completion_b}

FIRST provide a one-sentence comparison of the two reviews, explaining which is better and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. 

You MUST choose A or B for the preferred answer even if neither review is very good.  

Your response should use the format:

Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">\
""".lstrip())


def get_tldr_prompt(prompt, completion_a, completion_b):
    return (f"""\
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details?

Post:
{prompt}

Summary A:
{completion_a}

Summary B:
{completion_b}

FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. 

You MUST choose A or B for the preferred answer even if neither summary is very good.

Your response should use the format:

Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">\
""").lstrip()


IMDB_SYSTEM_PROMPT = "You are a helpful assistant that evaluates the quality and positive sentiment of movie reviews"
TLDR_SYSTEM_PROMPT = "You are a helpful assistant that evaluates the quality of summaries for internet posts"


def get_preference(prompt: str, completion_a: str, completion_b: str, task_name: str, model: str = "gpt-3.5-turbo", request_logger=None, oracle_temperature=0.05, provider="azure") -> dict:
    if model not in OPENAI_PRICING.keys():
        raise ValueError(f"Don't know pricing for model {model}")

    # TODO: not threadsafe to use different providers in the same process!
    if provider == "azure":
        _apply_azure_config()
        provider_kwargs = {
            "deployment_id": model
        }
    elif provider == "openai":
        _apply_openai_config()
        provider_kwargs = {
            "model": model
        }
    else:
        raise NotImplementedError(f"Unknown provider {provider}")

    swap_responses = random.random() > 0.5

    if swap_responses:
        completion_a, completion_b = completion_b, completion_a

    question_fn, system_prompt = {
        "tldr": (get_tldr_prompt, TLDR_SYSTEM_PROMPT),
        "imdb": (get_imdb_prompt, IMDB_SYSTEM_PROMPT),
    }[task_name]
    question = question_fn(prompt, completion_a, completion_b)

    # print(system_prompt, "\n", question)

    try:
        t_start = time.time()
        resp = openai.ChatCompletion.create(
            **provider_kwargs,
            temperature=oracle_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            request_timeout=REQUEST_TIMEOUT,
        )
        resp["request_duration"] = time.time() - t_start
        resp['estimated_cost_usd'] = sum([OPENAI_PRICING[model].get(k, 0.0) * v for k, v in resp['usage'].items()])
    except (openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError, openai.error.RateLimitError) as e:
        traceback.print_exc()
        backoff = BACKOFF_SECS * (1 + random.random())
        # if isinstance(e, openai.error.RateLimitError):
        #     backoff *= 4
        print(f"openai error raised {e} - sleeping for {backoff} s...")
        time.sleep(backoff)
        return get_preference(
            prompt,
            completion_a, completion_b,
            task_name,
            model=model, request_logger=request_logger, oracle_temperature=oracle_temperature, provider=provider)

    resp_content = resp['choices'][0]['message']['content']
    # print(resp_content)

    res = {
        'prompt': prompt,
        'completion_a': completion_a,
        'completion_b': completion_b,
        'openai_response': json.loads(json.dumps(resp)),
    }

    for line in resp_content.strip().split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            if k == "Comparison":
                res['comparison'] = v
            if k == "Preferred":
                res['preferred'] = preferred_letter_to_index(v)
                res['preferred_response_str'] = v  # to debug if it comes back with something unexpected
        res['cost'] = sum([OPENAI_PRICING[model].get(k, 0.0) * v for k, v in resp['usage'].items()])

    if swap_responses and res.get('preferred') is not None:
        res['preferred'] = (res['preferred'] + 1) % 2

    if request_logger is not None:
        request_logger.log(res)
    return res
