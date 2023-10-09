from typing import Callable, Optional

import trl
from transformers import PreTrainedTokenizerBase
from trl import PPOTrainer

from direct import utils as utils
from direct.config import ExperimentConfig
from direct.types import TModel


# TODO fix!
# from direct.generate import ScoredCompletionGenerator


class PPOPreferenceTrainer:
    def __init__(
            self,
            config: ExperimentConfig,
            ppo_config: trl.PPOConfig,
            model: TModel,
            tokenizer: PreTrainedTokenizerBase,
            data_collator: Callable = None,
            num_shared_layers: Optional[int] = None,
            ref_model: Optional[TModel] = None,
    ):
        ppo_config.batch_size = config.train.batch_size
        ppo_config.learning_rate = config.train.lr
        ppo_config.horizon = config.steps

        # noinspection PyTypeChecker
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            num_shared_layers=num_shared_layers,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.output_length_sampler = utils.LengthSampler(*config.data.completion_len_range)
        self.loss_property_name = "ppo/loss/total"
        self.optimizer = self.ppo_trainer.optimizer
        self.ref_model = self.ppo_trainer.ref_model
        self.config = config

    # def step(
    #         self,
    #         batch: Dict[str, Any],
    #         preference_model: Callable,
    #         device: str = "cpu"
    # ) -> Tuple[Dict[str, Any], List[Any]]:
    #     """
    #     Run a PPO optimisation step - generating responses, scoring them then using
    #     PPOTrainer to apply RL updates
    #
    #     Args:
    #         batch: Latest batch of training data yielded from data_loader - containing tokens
    #         preference_model: callable to get scores on generated prompt-completions
    #         device: device to use
    #
    #     Returns:
    #         train_stats (dict[str, Any]):
    #             a summary of the training statistics
    #         scores: a Tensor of preference scores for the responses
    #     """
    #
    #     scg = ScoredCompletionGenerator(self.model, self.tokenizer, preference_model, self.output_length_sampler, 4,
    #                                     self.config.generate_gpt2.to_kwargs())
    #     scored_completions = scg.generate(batch, device)
    #
    #     # for query, completion, score in zip(scored_completions["query_str"],
    #     #                                     scored_completions["response_str"],
    #     #                                     scored_completions["score"]):
    #     #     print(f'"{query}","{completion}",{score}')
    #
    #     # Use the trl PPOTrainer to update the model now we've got generated responses and scores
    #     stats = self.ppo_trainer.step(
    #         scored_completions["query_tokens"],
    #         scored_completions["response_tokens"],
    #         list(torch.Tensor(scored_completions["score"]).to(device)),
    #     )
    #
    #     stats["env/kl_mean"] = stats["objective/kl"]
    #     return stats, scored_completions["score"]
