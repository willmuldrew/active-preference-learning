## Dockerfile

FROM runpod/pytorch:3.10-2.0.1-117-devel

RUN apt update --yes 
RUN apt install -y rsync vim nvtop htop tmux

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.1-0-Linux-x86_64.sh >/root/miniconda.sh

RUN chmod u+x /root/miniconda.sh
RUN /root/miniconda.sh -b -p /root/miniconda3
RUN /root/miniconda3/bin/conda init

RUN /root/miniconda3/bin/pip3 install --upgrade pip
RUN /root/miniconda3/bin/pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /root/requirements.txt
RUN /root/miniconda3/bin/pip3 install -r /root/requirements.txt 

RUN echo export PATH=/root/miniconda3/bin:$PATH >> /root/.bashrc
