FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -m -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} ${USERNAME} \
    && apt-get update \
    && apt-get install -y sudo git libgl1 libglib2.0-0 libsm6 libice6 libxrender1 libxext6 libxcb-xinerama0 \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
