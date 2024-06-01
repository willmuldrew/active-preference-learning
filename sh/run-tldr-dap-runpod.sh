#!/bin/bash -l 

# run like: CUDA_VISIBLE_DEVICES=0 MODE=HIGH_ENTROPY_AND_CERTAINTY SEED=12029 sh/run-tldr2-runpod.sh


export PYTHONPATH=.
PYTHON=python3
#PYTHON=echo

# NOTE - provide SEEDS on the command line via an environment variable
#SEEDS="42 41697 29716"
EVAL_TEST_SET_SIZE=512

SHARED_ARGS="--train.dpo.beta 0.2 \
--experiment_name exp5 \
--log true \
--wandb_project wm-debug-tldr \
--data.dataset_name tldr \
--exp5.prompt_batch_size 64 \
--data.truncate_prompts false \
--data.completion_min_len 128 --data.completion_max_len 128 \
--model_class gpt-neox --model_instance pvduy/pythia-1B-sft-summarize-tldr \
--train.trainer direct --train.loss_fn dpo \
--train.batch_size 8 --train.grad_acc_steps 8 --train.lr 1e-06 \
--eval.batch_size 8 --eval.test_set_size $EVAL_TEST_SET_SIZE --eval.interim_eval_interval_steps -1 --eval.sampling_temperatures [0.25] \
--pref_model_class openai --pref_model_instance tldr-gpt-4-1106-preview \
--generate_gpt2.temperature 0.7 \
--exp5.loss_ma_early_stopper_threshold 0.0 \
--exp5.max_epochs 1 --train.optimizer Adam 
--eval.versus label \
--exp5.no_reset true --exp5.reacquire_all_data true --eval.defer false \
--exp5.num_openai_threads 10 \
--exp5.fixed_m 64 --exp5.max_phases 200 \
--exp5.eval_interval 10"

RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"
CERTAINTY_ARGS="--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 16"



ARGS_VAR="${MODE}_ARGS"
ADDITIONAL_ARGS=${!ARGS_VAR}
if [ -z "$ADDITIONAL_ARGS" ]; then
  echo "don't know about mode $MODE"
  exit 1
fi

$PYTHON direct/main.py --seed $SEED --wandb_tags dap1 $SHARED_ARGS $ADDITIONAL_ARGS 

