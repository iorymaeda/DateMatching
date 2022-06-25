FROM anibali/pytorch:1.11.0-cuda11.5-ubuntu20.04 as builder

FROM python:3.10

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./predict.py /usr/src/app/
COPY ./Models /usr/src/app/Models
COPY ./utils /usr/src/app/utils
COPY ./requirements.txt /usr/src/app/

RUN pip install -r requirements.txt

WORKDIR /usr/src/app/Torchserve
CMD ["python", "predict.py"]