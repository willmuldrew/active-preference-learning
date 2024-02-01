#!/bin/bash

# run like: CUDA_VISIBLE_DEVICES=0 MIX_R=0.9 MODE=CERTAINTY SEED=12029 sh/run-tldr-noreset.sh


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
--eval.versus ref_model
--exp5.num_openai_threads 10
--exp5.no_reset true
--exp5.update_ref_model false
--exp5.mix_data true
--exp5.mix_data_m 128
--exp5.mix_data_r $MIX_R"


RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"
CERTAINTY_ARGS="--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 32"


BETA=0.2

ARGS_VAR="${MODE}_ARGS"
ADDITIONAL_ARGS=${!ARGS_VAR}
if [ -z "$ADDITIONAL_ARGS" ]; then
  echo "don't know about mode $MODE"
  exit 1
fi

$PYTHON direct/main.py --seed $SEED --eval.defer true --wandb_tags xmas-noreset-mixdata2 --train.dpo.beta $BETA $SHARED_ARGS $ADDITIONAL_ARGS

