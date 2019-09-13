FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -qy \
       g++ \
       openjdk-11-jdk-headless \
       wget \
       xz-utils

RUN wget https://nodejs.org/dist/v10.16.3/node-v10.16.3-linux-x64.tar.xz -nv -O /tmp/node.tar.xz \
    && mkdir /opt/node \
    && tar -C /opt/node --strip-components=1 -xf /tmp/node.tar.xz \
    && rm /tmp/node.tar.xz
