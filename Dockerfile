FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

LABEL maintainer="Kacper Kazaniecki"

ARG LPIPS_PATH
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=kacper
ARG USERID=1000
ARG EPE_DIR='/app'

# Instal basic utilities
RUN apt-get update && \
    apt-get -y upgrade

RUN DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends bzip2 build-essential ca-certificates \
        curl vim tmux git wget unzip sudo software-properties-common && \
        rm -rf /var/lib/apt/lists/*

RUN apt-get install --only-upgrade libstdc++6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# OpenCV fix
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# # Installing python
# RUN apt-get update && apt-get install -y python3.8 python3-pip

# Installing azure cli
RUN DEBIAN_FRONTEND=noninteractive curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# # Installing miniconda
# ENV PATH $CONDA_DIR/bin:$PATH
# RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
#     /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
#     rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# RUN conda install -y mamba -c conda-forge

# RUN conda install python=3.8
# RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# RUN conda install -c pytorch faiss-gpu
# RUN conda install scikit-image
# RUN conda install imageio
# RUN pip install tqdm

COPY EPE/environment.yaml /home/${USERNAME}/environment.yaml
RUN conda env update --name base --file /home/${USERNAME}/environment.yaml &&\
    conda clean -tipy

# RUN conda install -y cudatoolkit=11.5 pytorch-gpu torchvision -c pytorch -c conda-forge

RUN pip install kornia

COPY ../../PerceptualSimilarity /home/${USERNAME}/PerceptualSimilarity
RUN pip install -e /home/${USERNAME}/PerceptualSimilarity

COPY EPE ${EPE_DIR}
RUN pip install -e ${EPE_DIR}
WORKDIR ${EPE_DIR}

COPY ../../metadata /config

# For interactive shell
RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc

RUN mkdir -p /app/python_runtime/python3/bin
RUN ln -s ${CONDA_DIR}/bin/python /app/python_runtime/python3/bin/python3

ENTRYPOINT [ "python", "/app/main.py", "--log", "info", "--log_dir", "/app/log", "--action", "train", "--config", "/config/train_sim2real.yaml"]