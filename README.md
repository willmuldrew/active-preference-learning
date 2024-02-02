Active Preference Learning
--------------------------

* Code in ./direct
* Setup and experimental launchers in ./sh
* Analysis notebooks in ./notebooks

Setup (python3.10)
-----
    # first install miniconda and put it in your path
  
    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt 

Running experiments
----
e.g.

    SEEDS=42 bash sh/run-imdb2.sh


Auth
----
Needs auth for W&B and OpenAI (with GPT-4 access) - keys in home directory

Infrastructure
----
Larger experiments require 80GB A100 or H100 (for now).
