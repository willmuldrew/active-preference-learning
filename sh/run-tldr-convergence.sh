export PYTHONPATH=.

SEEDS="41697 42"
Ms="128 256 512"
EVAL_BATCH_SIZE=32
EVAL_TEST_SET_SIZE=1024

for M in $Ms; do

SHARED_ARGS="--experiment_name exp5 \
--log true --wandb_project wm-debug-tldr-convergence --wandb_tags tldr-exp5-convergence1 \
--data.dataset_name tldr --exp5.prompt_batch_size 32 \
--data.truncate_prompts false \
--data.completion_min_len 128 --data.completion_max_len 128 \
--model_class gpt-neox --model_instance pvduy/pythia-1B-sft-summarize-tldr \
--train.trainer direct --train.loss_fn dpo \
--train.batch_size 8 --train.grad_acc_steps 8 --train.lr 1e-06 \
--eval.batch_size $EVAL_BATCH_SIZE --eval.test_set_size $EVAL_TEST_SET_SIZE --eval.eval_epoch_interval 16 --eval.vs_ref_model_eval true --eval.sampling_temperatures [0.25] \
--eval.interim_eval_interval_steps -1 \
--pref_model_class openai --pref_model_instance tldr-gpt-4 \
--generate_gpt2.temperature 0.7 \
--exp5.m_schedule [$M] --exp5.eval_m_schedule [] \
--exp5.max_epochs 160 --exp5.loss_ma_early_stopper_threshold 0.0\
--train.dpo.beta 0.2"

RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"

#CMD=echo
CMD=

  for SEED in $SEEDS; do
    $CMD python direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $RANDOM_ARGS
  done

done





