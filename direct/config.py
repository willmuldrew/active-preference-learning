import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from trl.trainer import PPOConfig


@dataclass
class DPOConfig(object):
    beta: float = 0.1
    acquisition_filter: str = "none"  # rm-rank, gm-rank
    oversample_factor: int = 1


@dataclass
class PPOConfigOverrides:
    target: int = 6

    def get_trl_ppo_config(self):
        return PPOConfig(**{k: v for (k, v) in dataclasses.asdict(self).items() if v is not None})


@dataclass
class TrainerConfig:
    trainer_class: str = "direct"
    loss_fn: str = "dpo"
    lr: float = 1.41e-5
    lr_ramp_start_factor: float = 1.0
    lr_ramp_total_iters: int = 1
    optimizer: str = "Adam"
    batch_size: int = 256
    grad_acc_steps: int = 1
    # WJM - this is Peter's hack to disable KL (i.e. the P in PPO!)... seems to
    #       lead to grammatically poor results e.g. stuffing responses with
    #       repeated "excellent" and "great" to get high scores
    #  ppo: Optional[PPOConfig] = PPOConfig(init_kl_coef=0.0)
    ppo: PPOConfigOverrides = PPOConfigOverrides()
    dpo: Optional[DPOConfig] = DPOConfig()

    @property
    def effective_batch_size(self):
        return self.batch_size * self.grad_acc_steps


@dataclass
class GPT2GenerateArgsConfig:
    min_length: int = -1
    top_k: int = 0
    top_p: float = 1.0
    temperature: float = 1.0

    def to_kwargs(self):
        return dataclasses.asdict(self)


@dataclass
class EvalConfig:
    interim_eval_interval_steps: int = -1
    eval_epoch_interval: int = -1  # only in exp5
    test_set_size: int = 4096
    batch_size: int = 32
    perform_final_full_eval: bool = False
    vs_ref_model_eval: bool = False
    sampling_temperatures: list[float] = field(default_factory=lambda: [1.0, 0.7, 0.0])


@dataclass
class DataConfig:
    truncate_prompts: bool = True
    prompt_min_len: int = None
    prompt_max_len: int = None
    completion_min_len: int = 4
    completion_max_len: int = 16
    dataset_name: str = "imdb"
    limit_train_n: Optional[int] = None  # limit number of datapoints made available (for debugging)
    limit_test_n: Optional[int] = None
    num_proc: int = 32

    @property
    def prompt_len_range(self):
        return self.prompt_min_len, self.prompt_max_len

    @property
    def completion_len_range(self):
        return self.completion_min_len, self.completion_max_len


@dataclass
class Exp4Config:
    mode: str = None  # offline, online, active
    initial_m: int = 256
    max_m: int = 2048
    rank_pair_fn: str = "random"
    over_generate_factor: int = 1
    reset_after_acquisition: bool = False
    defer_acquisition_using_loss: bool = False


@dataclass
class Exp5Config:
    m_schedule: list[int] = None
    eval_m_schedule: list[int] = None
    reacquire_all_data: bool = False
    acquire_pairs_function: str = "RANDOM"
    over_generate_factor: int = 1
    over_sample_prompts_factor: int = 1
    entropy_sample_n: int = 32
    prompt_batch_size: int = 16
    loss_ma_early_stopper_threshold: float = 0.1
    max_steps: int = None
    max_epochs: int = None
    num_openai_threads: int = 1
    openai_provider: str = "openai"


@dataclass
class ExperimentConfig:
    seed: int = 42
    experiment_name: str = "exp1"
    train: TrainerConfig = TrainerConfig()
    data: DataConfig = DataConfig()
    steps: int = 500
    model_class: str = "gpt2"
    model_instance: str = "lvwerra/gpt2-imdb"  # 124M params? (~500MB)
    # model_instance: str = "edbeeching/gpt2-medium-imdb" # 355M params (~1.44GB)
    # model_instance: str = "edbeeching/gpt2-large-imdb" # 774M params (~3.13GB)
    # model_instance: str = "edbeeching/gpt2-xl-imdb" # 1.5B params (~6.28GB)

    # model_instance: str = "pvduy/pythia-125M-sft-summarize-tldr" # could do a 1B model (maybe) on an 80GB A100/H100
    # pvduy/pythia-1B-sft-summarize-tldr
    # jon-tow/pythia-1.4b-summarize-sft
    # lxuechen/tldr-gpt2-xl <- slightly larger again... might not fit onto an A100 without more tweaking

    pref_model_instance: str = "lvwerra/distilbert-imdb"
    pref_model_class: str = "sentiment-analysis"
    device: Optional[str] = None
    log: bool = True
    wandb_project: str = "preference-learning"
    wandb_run: str = None
    wandb_tags: str = None
    eval: EvalConfig = EvalConfig()
    generate_gpt2: GPT2GenerateArgsConfig = GPT2GenerateArgsConfig()
    # exp3: Exp3Config = Exp3Config()
    exp4: Exp4Config = Exp4Config()
    exp5: Exp5Config = Exp5Config()
    openai_request_log_path: str = "openai_request_log.jsonl"
