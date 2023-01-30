FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

LABEL maintainer="Kacper Kazaniecki"

ARG LPIPS_PATH
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=kacper
ARG USERID=1000
ARG EPE_DIR='/app'

ENV WANDB_USERNAME="kacper-kazaniecki"
ENV WANDB_USER_EMAIL="kacper@wayve.ai"
ENV WANDB_WANDB_ENTITY="wayve-ai"

# Instal basic utilities
RUN apt-get update -q -y && \
    apt-get -y upgrade

RUN DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends bzip2 build-essential ca-certificates \
        curl vim tmux git wget unzip sudo software-properties-common \
        libprotobuf-dev libprotobuf-c0-dev protobuf-c-compiler protobuf-compiler python-protobuf && \
        rm -rf /var/lib/apt/lists/*

RUN apt-get install --only-upgrade libstdc++6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# OpenCV fix
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Installing azure cli
RUN DEBIAN_FRONTEND=noninteractive curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

COPY EPE/environment.yaml /home/${USERNAME}/environment.yaml
RUN conda update -n base conda
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba
RUN conda env update --name base --file /home/${USERNAME}/environment.yaml &&\
    conda clean -tipy

COPY EPE ${EPE_DIR}
RUN pip install -e ${EPE_DIR}
WORKDIR ${EPE_DIR}

# installing some pip packages in conda environment
RUN conda config --set pip_interop_enabled True

RUN pip install git+https://github.com/EyalMichaeli/PerceptualSimilarity.git

COPY ../../WayveCode /home/${USERNAME}/WayveCode

RUN /home/${USERNAME}/WayveCode/wayve/ai/lib/conda.sh -e base -r ${EPE_DIR}/requirements-ailib.txt

# Running in azureml using python
RUN mkdir -p /app/python_runtime/python3/bin
RUN ln -s ${CONDA_DIR}/bin/python /app/python_runtime/python3/bin/python3

COPY ../../metadata/urban-driving /config

# For interactive shell
RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc

ENTRYPOINT [ "python", "/app/main.py"]