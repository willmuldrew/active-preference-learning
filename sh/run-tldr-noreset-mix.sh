#!/bin/bash

# run like: CUDA_VISIBLE_DEVICES=0 MODE=CERTAINTY SEED=12029 sh/run-tldr-noreset.sh


export PYTHONPATH=.
PYTHON=python3
#PYTHON=echo

M_SCHEDULE="[0,128,256,384,512]"
# Okay since we're deferring eval - just going to dump samples
EVAL_SCHEDULE=$M_SCHEDULE

# NOTE - provide SEEDS on the command line via an environment variable
#SEEDS="42 41697 29716"
EVAL_TEST_SET_SIZE=1024

SHARED_ARGS="--experiment_name exp5 \
--log true --wandb_project wm-debug-tldr \
--data.dataset_name tldr --exp5.prompt_batch_size 16 \
--data.truncate_prompts false \
--data.completion_min_len 128 --data.completion_max_len 128 \
--model_class gpt-neox --model_instance pvduy/pythia-1B-sft-summarize-tldr \
--train.trainer direct --train.loss_fn dpo \
--train.batch_size 8 --train.grad_acc_steps 8 --train.lr 1e-06 \
--eval.batch_size 8 --eval.test_set_size $EVAL_TEST_SET_SIZE --eval.interim_eval_interval_steps -1 --eval.sampling_temperatures [0.25] \
--pref_model_class openai --pref_model_instance tldr-gpt-4-1106-preview \
--generate_gpt2.temperature 0.7 \
--exp5.m_schedule $M_SCHEDULE --exp5.eval_m_schedule $EVAL_SCHEDULE \
--exp5.loss_ma_early_stopper_threshold 0.0 --exp5.max_epochs 70 \
--train.optimizer Adam
--eval.versus label
--exp5.num_openai_threads 10
--exp5.no_reset true
--exp5.mix_data true
--exp5.mix_data_m 128
--exp5.mix_data_r 0.5"

# < 8 hours (or less)
#OFFLINE_ARGS="--exp5.acquire_pairs_function OFFLINE --exp5.over_generate_factor 1"
# 16 hours
RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"
CERTAINTY_ARGS="--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 16"
#UNCERTAINTY_ARGS="--exp5.acquire_pairs_function UNCERTAINTY --exp5.over_generate_factor 16"
# ~30 hours
#ENTROPY_ARGS="--exp5.acquire_pairs_function ENTROPY --exp5.over_sample_prompts_factor 4 --exp5.entropy_sample_n 16"
#HIGH_ENTROPY_AND_CERTAINTY_ARGS="--exp5.acquire_pairs_function HIGH_ENTROPY_AND_CERTAINTY --exp5.over_sample_prompts_factor 4 --exp5.entropy_sample_n 8 --exp5.over_generate_factor 16"
#LOW_ENTROPY_AND_CERTAINTY_ARGS="--exp5.acquire_pairs_function LOW_ENTROPY_AND_CERTAINTY --exp5.over_sample_prompts_factor 4 --exp5.entropy_sample_n 8 --exp5.over_generate_factor 16"


BETA=0.2

ARGS_VAR="${MODE}_ARGS"
ADDITIONAL_ARGS=${!ARGS_VAR}
if [ -z "$ADDITIONAL_ARGS" ]; then
  echo "don't know about mode $MODE"
  exit 1
fi

$PYTHON direct/main.py --seed $SEED --eval.defer true --wandb_tags xmas-noreset-mixdata1 --train.dpo.beta $BETA $SHARED_ARGS $ADDITIONAL_ARGS

