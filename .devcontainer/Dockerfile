FROM python:3.12

RUN apt update && apt install -qy \
    aspell-en \
    git-core

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID vscode \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash vscode \
    && chown -R $USER_UID:$USER_GID /home/vscode
