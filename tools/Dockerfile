FROM python:3.6

RUN apt-get update \
    && apt-get install -qy \
       aspell-en

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
