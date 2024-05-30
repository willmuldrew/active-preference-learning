#!/bin/bash -l 

# run like: CUDA_VISIBLE_DEVICES=0 MODE=HIGH_ENTROPY_AND_CERTAINTY SEED=12029 sh/run-imdb2-runpod.sh


export PYTHONPATH=.
PYTHON=python3
#PYTHON=echo

M_SCHEDULE="[128,256,384,512,640,768]"
# Okay since we're deferring eval - just going to dump samples 
EVAL_SCHEDULE=$M_SCHEDULE

# NOTE - provide SEEDS on the command line via an environment variable
#SEEDS="42 41697 29716"
EVAL_TEST_SET_SIZE=1024

SHARED_ARGS="--experiment_name exp5 \
--log true --wandb_project wm-debug-imdb \
--data.dataset_name imdb --exp5.prompt_batch_size 32 \
--data.truncate_prompts true --data.prompt_min_len 8 --data.prompt_max_len 16 \
--data.completion_min_len 16 --data.completion_max_len 24 \
--model_class gpt-2 --model_instance edbeeching/gpt2-large-imdb \
--train.trainer direct --train.loss_fn dpo \
--train.batch_size 16 --train.grad_acc_steps 4 --train.lr 1e-06 \
--eval.batch_size 32 --eval.test_set_size $EVAL_TEST_SET_SIZE --eval.interim_eval_interval_steps -1 --eval.sampling_temperatures [0.25] \
--pref_model_class openai --pref_model_instance imdb-gpt-4-1106-preview \
--generate_gpt2.temperature 0.7 \
--exp5.m_schedule $M_SCHEDULE --exp5.eval_m_schedule $EVAL_SCHEDULE \
--exp5.loss_ma_early_stopper_threshold 0.0 --exp5.max_epochs 50 \
--train.optimizer Adam 
--eval.versus ref_model"


OFFLINE_ARGS="--exp5.acquire_pairs_function OFFLINE --exp5.over_generate_factor 1"
HIGH_ENTROPY_ARGS="--exp5.acquire_pairs_function HIGH_ENTROPY --exp5.over_sample_prompts_factor 8 --exp5.entropy_sample_n 8"
LOW_ENTROPY_ARGS="--exp5.acquire_pairs_function LOW_ENTROPY --exp5.over_sample_prompts_factor 8 --exp5.entropy_sample_n 8"
RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"
CERTAINTY_ARGS="--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 32"
UNCERTAINTY_ARGS="--exp5.acquire_pairs_function UNCERTAINTY --exp5.over_generate_factor 32"
HIGH_ENTROPY_AND_CERTAINTY_ARGS="--exp5.acquire_pairs_function HIGH_ENTROPY_AND_CERTAINTY --exp5.over_sample_prompts_factor 8 --exp5.entropy_sample_n 8 --exp5.over_generate_factor 32"
LOW_ENTROPY_AND_CERTAINTY_ARGS="--exp5.acquire_pairs_function LOW_ENTROPY_AND_CERTAINTY --exp5.over_sample_prompts_factor 8 --exp5.entropy_sample_n 8 --exp5.over_generate_factor 32"


BETA=0.2

ARGS_VAR="${MODE}_ARGS"
ADDITIONAL_ARGS=${!ARGS_VAR}
if [ -z "$ADDITIONAL_ARGS" ]; then
  echo "don't know about mode $MODE"
  exit 1
fi

$PYTHON direct/main.py --seed $SEED --eval.defer true --wandb_tags xmas-sweep2 --train.dpo.beta $BETA $SHARED_ARGS $ADDITIONAL_ARGS 

