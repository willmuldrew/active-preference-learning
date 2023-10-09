rsync -avz --no-owner --no-group -c --modify-window=3 \
  --exclude *.pyc --exclude venv --exclude wandb --exclude "runpod-*.sh" --exclude __pycache__ \
  --exclude ".pytest_cache" --exclude ".git" --exclude ".ipynb_checkpoints" --exclude ".gitignore"
  --exclude "*.swp" --exclude "*.log" \
  /home/will/code/preference-learning $HOST:/workspace/.
