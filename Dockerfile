FROM anibali/pytorch:1.11.0-cuda11.5-ubuntu20.04 as builder

FROM python:3.10

USER root
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./Torchserve /usr/src/app/Torchserve
COPY ./Models /usr/src/app/Models
COPY ./utils /usr/src/app/utils
COPY ./requirements.txt /usr/src/app/

RUN pip install -r requirements.txt

WORKDIR /usr/src/app/Torchserve
CMD ["python", "predict.py"]