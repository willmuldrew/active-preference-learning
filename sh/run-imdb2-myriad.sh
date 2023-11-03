#!/bin/bash -l 
      
#$ -l h_rt=10:00:00
#$ -l gpu=1
#$ -l mem=20G
#$ -ac allow=L                
#$ -N imdb2-myriad 
#$ -cwd

#$ -l tmpfs=10G

module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load cuda/11.8.0/gnu-10.2.0

export TMPDIR=${TMPDIR:-/tmp}
mkdir -p $TMPDIR
# Don't want these interfering across multiple runs or cluttering up my home dir  
export HF_DATASETS_CACHE=${TMPDIR}/cache/huggingface-datasets
export HUGGINGFACE_HUB_CACHE=${TMPDIR}/cache/huggingface-hub
export WANDB_CACHE_DIR=${TMPDIR}/cache/wandb

SEEDS=42


export PYTHONPATH=.
PYTHON=./venv/bin/python3

#M_SCHEDULE="[1024]"
M_SCHEDULE="[64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024]"
EVAL_SCHEDULE="[1024]"

# NOTE - provide SEEDS on the command line via an environment variable
#SEEDS="42 41697 29716"
EVAL_TEST_SET_SIZE=512

SHARED_ARGS="--experiment_name exp5 \
--log true --wandb_project wm-debug-imdb \
--data.dataset_name imdb --exp5.prompt_batch_size 32 \
--data.truncate_prompts true --data.prompt_min_len 8 --data.prompt_max_len 16 \
--data.completion_min_len 16 --data.completion_max_len 24 \
--model_class gpt-2 --model_instance edbeeching/gpt2-large-imdb \
--train.trainer direct --train.loss_fn dpo \
--train.batch_size 16 --train.grad_acc_steps 4 --train.lr 1e-06 \
--eval.batch_size 32 --eval.test_set_size $EVAL_TEST_SET_SIZE --eval.interim_eval_interval_steps -1 --eval.vs_ref_model_eval true --eval.sampling_temperatures [0.25] \
--pref_model_class openai --pref_model_instance imdb-gpt-4 \
--generate_gpt2.temperature 0.7 \
--exp5.m_schedule $M_SCHEDULE --exp5.eval_m_schedule $EVAL_SCHEDULE \
--exp5.loss_ma_early_stopper_threshold 0.0 --exp5.max_epochs 50 \
--train.optimizer Adam "


OFFLINE_ARGS="--exp5.acquire_pairs_function OFFLINE --exp5.over_generate_factor 1"
ENTROPY_ARGS="--exp5.acquire_pairs_function ENTROPY --exp5.over_sample_prompts_factor 16 --exp5.entropy_sample_n 32"
RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"
CERTAINTY_ARGS="--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 16"
UNCERTAINTY_ARGS="--exp5.acquire_pairs_function UNCERTAINTY --exp5.over_generate_factor 16"

BETA=0.2

for SEED in $SEEDS; do
#  $PYTHON direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $OFFLINE_ARGS
   $PYTHON direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $RANDOM_ARGS
   $PYTHON direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $CERTAINTY_ARGS
#  $PYTHON direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $UNCERTAINTY_ARGS
#  $PYTHON direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $ENTROPY_ARGS
done


