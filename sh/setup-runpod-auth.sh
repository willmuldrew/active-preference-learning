rsync -avz $HOME/.openai* $HOST:.

WANDB_KEY=$(cat $HOME/.wandb_key)
ssh $HOST -t /root/miniconda3/bin/wandb login --relogin $WANDB_KEY 

HF_KEY=$(cat $HOME/.huggingface_key)
ssh $HOST -t /root/miniconda3/bin/huggingface-cli login --token $HF_KEY


