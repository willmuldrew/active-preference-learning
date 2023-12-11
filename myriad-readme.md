Getting Started on Myriad/UCL CS Cluster
-------------------------

1. Install miniconda for python 3.10 to home directory (I hit some issues with 3.11 which I haven't yet fixed) - 
   this can take quite a long time since there are lots of small files and lustre doesn't like this
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
    chmod u+x Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
    ./Miniconda3-py310_23.5.0-3-Linux-x86_64.sh -b
    ./miniconda3/bin/conda init
    # You may need to login/logout at this stage
    ```
1. Clone repo to home directory and cd into it
    ```
    git clone https://github.com/willmuldrew/active-preference-learning -b wm-dev
    cd active-preference-learning
    ```
1. Create a venv in the code directory and activate it
   ```
   python -m venv venv
   . venv/bin/activate
   ```
1. Install requirements - first manually installing the relevant torch version with cuda 11.8 support
   ```
   pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements-myriad.txt
   ```
1. Authenticate to huggingface and wandb using cli tools (follow their instructions)
   ```
   wandb login
   huggingface-cli login
   ```
1. Configure openai creds in $HOME/openai
   ``` 
   echo MY_OPENAI_KEY >$HOME/.openai
   ```
1. Submit a job from the **root of the code directory** - ensuring that the cluster run-time is set appropriately. 
   Note that concurrent jobs will compete for OpenAI rate limits!  
   ```
   qsub sh/run-imdb2-myriad.sh
   ```
   or for the UCL CS cluster
   ```
   qsub sh/run-imdb2-pryor.sh
   ```
1. Observe!
   ```
   qstat
   ```
