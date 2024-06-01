export PYTHONPATH=.

export PYTHONPATH=.

SHARED_ARGS="--train.dpo.beta 0.2 \
--experiment_name exp5 \
--log true \
--wandb_project wm-debug-imdb \
--data.dataset_name imdb \
--exp5.prompt_batch_size 64 \
--data.truncate_prompts true \
--data.prompt_min_len 8 --data.prompt_max_len 16 --data.completion_min_len 16 --data.completion_max_len 24 \
--model_class gpt-2 --model_instance edbeeching/gpt2-large-imdb \
--train.trainer direct \
--train.loss_fn dpo --train.batch_size 16 --train.grad_acc_steps 4 --train.lr 1e-06 \
--eval.batch_size 32 --eval.test_set_size 1024 --eval.interim_eval_interval_steps -1 --eval.versus ref_model --eval.sampling_temperatures [0.25] \
--pref_model_class sentiment-analysis --pref_model_instance lvwerra/distilbert-imdb \
--generate_gpt2.temperature 0.7 \
--exp5.loss_ma_early_stopper_threshold 0.0 \
--exp5.max_epochs 1 --train.optimizer Adam \
--exp5.no_reset true --exp5.reacquire_all_data true --eval.defer false \
--exp5.num_openai_threads 10 \
--exp5.fixed_m 64 --exp5.max_phases 200 \
--exp5.eval_interval 10"

C_SEEDS="38967 1283 29952"
R_SEEDS="64165 21354 2343"

for SEED in $C_SEEDS; do
  python direct/main.py --seed $SEED --exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 8 $SHARED_ARGS
done

for SEED in $R_SEEDS; do
  python direct/main.py --seed $SEED --exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1 $SHARED_ARGS
done



#echo $SHARED_ARGS

#
#--seed 31697 \
#--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 8 \
#
#OFFLINE_ARGS="--exp5.acquire_pairs_function OFFLINE --exp5.over_generate_factor 1"
#ENTROPY_ARGS="--exp5.acquire_pairs_function ENTROPY --exp5.over_sample_prompts_factor 16 --exp5.entropy_sample_n 32"
#RANDOM_ARGS="--exp5.acquire_pairs_function RANDOM --exp5.over_generate_factor 1"
#CERTAINTY_ARGS="--exp5.acquire_pairs_function CERTAINTY --exp5.over_generate_factor 16"
#UNCERTAINTY_ARGS="--exp5.acquire_pairs_function UNCERTAINTY --exp5.over_generate_factor 16"
#
#BETA=0.2
#
#for SEED in $SEEDS; do
#  python direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $OFFLINE_ARGS
#  python direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $RANDOM_ARGS
#  python direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $CERTAINTY_ARGS
#  python direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $UNCERTAINTY_ARGS
#  python direct/main.py --seed $SEED --train.dpo.beta $BETA $SHARED_ARGS $ENTROPY_ARGS
#done


