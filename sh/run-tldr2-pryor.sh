#!/bin/bash -l 
      
#$ -l h_rt=20:00:00
#$ -l gpu=true,gpu_type=(a100|a100_80|a100_dgx)
#$ -l tmem=10G
#$ -N tldr2 
#$ -cwd

#$ -l tscratch=10G

if [ -n "${JOB_ID}" ]; then 
	TMPDIR=/scratch0/wmuldrew/$JOB_ID
	mkdir -p $TMPDIR 

	function finish {
	    rm -rf $TMPDIR 
	}

	trap finish EXIT ERR INT TERM
fi

#module -f unload compilers mpi gcc-libs
#module load beta-modules
#module load gcc-libs/10.2.0
#module load cuda/11.8.0/gnu-10.2.0
source /share/apps/source_files/cuda/cuda-11.8.source

export TMPDIR=${TMPDIR:-/tmp}
mkdir -p $TMPDIR
# Don't want these interfering across multiple runs or cluttering up my home dir  
export HF_DATASETS_CACHE=${TMPDIR}/cache/huggingface-datasets
export HUGGINGFACE_HUB_CACHE=${TMPDIR}/cache/huggingface-hub
export WANDB_CACHE_DIR=${TMPDIR}/cache/wandb

#SEEDS=42


export PYTHONPATH=.
PYTHON=./venv/bin/python3
#PYTHON=echo

M_SCHEDULE="[128,256,384,512]"
EVAL_SCHEDULE="[512]"

# NOTE - provide SEEDS on the command line via an environment variable
#SEEDS="42 41697 29716"
EVAL_TEST_SET_SIZE=1024

SHARED_ARGS="--experiment_name exp5 \
--log true --wandb_project wm-debug-tldr \
--data.dataset_name tldr --exp5.prompt_batch_size 32 \
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
--eval.versus label"

# < 8 hours (or less)
OFFLINE_ARGS="--exp5.acquire_pairs_function OFFLINE --exp5.over_generate_factor 1"
# 16 hours
RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"
CERTAINTY_ARGS="--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 16"
UNCERTAINTY_ARGS="--exp5.acquire_pairs_function UNCERTAINTY --exp5.over_generate_factor 16"
# ~30 hours
ENTROPY_ARGS="--exp5.acquire_pairs_function ENTROPY --exp5.over_sample_prompts_factor 4 --exp5.entropy_sample_n 16"
HIGH_ENTROPY_AND_CERTAINTY_ARGS="--exp5.acquire_pairs_function HIGH_ENTROPY_AND_CERTAINTY --exp5.over_sample_prompts_factor 4 --exp5.entropy_sample_n 16 --exp5.over_generate_factor 8"
LOW_ENTROPY_AND_CERTAINTY_ARGS="--exp5.acquire_pairs_function LOW_ENTROPY_AND_CERTAINTY --exp5.over_sample_prompts_factor 4 --exp5.entropy_sample_n 16 --exp5.over_generate_factor 8"


BETA=0.2

ARGS_VAR="${MODE}_ARGS"
ADDITIONAL_ARGS=${!ARGS_VAR}
if [ -z "$ADDITIONAL_ARGS" ]; then
  echo "don't know about mode $MODE"
  exit 1
fi

$PYTHON direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $ADDITIONAL_ARGS 

