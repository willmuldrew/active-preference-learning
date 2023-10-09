from typing import Union
from transformers import PreTrainedModel
from trl import PreTrainedModelWrapper

TModel = Union[PreTrainedModel, PreTrainedModelWrapper]
