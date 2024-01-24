if [ -z "$HOST" ]; then
    echo "Please set HOST"
    exit 1
fi

ssh $HOST "cd /workspace; git clone -b wm-dev https://github.com/willmuldrew/active-preference-learning.git"

rsync -avz $HOME/.openai* $HOST:.

WANDB_KEY=$(cat $HOME/.wandb_key)
ssh $HOST -t /root/miniconda3/bin/wandb login --relogin $WANDB_KEY 

HF_KEY=$(cat $HOME/.huggingface_key)
ssh $HOST -t /root/miniconda3/bin/huggingface-cli login --token $HF_KEY


