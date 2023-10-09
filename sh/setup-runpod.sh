
#ssh $HOST unminimize
ssh $HOST apt update
ssh $HOST apt install -y rsync vim nvtop htop tmux
ssh $HOST "curl https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.1-0-Linux-x86_64.sh >/workspace/miniconda.sh"
ssh $HOST "chmod u+x /workspace/miniconda.sh"
ssh $HOST "/workspace/miniconda.sh -b -p /workspace/miniconda3"
ssh $HOST "/workspace/miniconda3/bin/conda init"

HOST=$HOST bash ./sync_runpod.sh

ssh $HOST /workspace/miniconda3/bin/python -m venv /workspace/preference-learning/venv
ssh $HOST /workspace/preference-learning/venv/bin/pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
ssh $HOST /workspace/preference-learning/venv/bin/pip install -r /workspace/preference-learning/requirements.txt
ssh $HOST /workspace/preference-learning/venv/bin/pip install --upgrade pip

rsync -avz $HOME/.openai* $HOST:.

WANDB_KEY=$(cat $HOME/.wandb_key)
ssh $HOST -t /workspace/preference-learning/venv/bin/wandb login --relogin $WANDB_KEY 

HF_KEY=$(cat $HOME/.huggingface_key)
ssh $HOST -t /workspace/preference-learning/venv/bin/huggingface-cli login --token $HF_KEY


