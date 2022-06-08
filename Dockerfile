FROM pytorch/torchserve as builder

FROM python:3.10

USER root

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY ./ /usr/src/app/

RUN pip install -r requirements.txt

WORKDIR /usr/src/app/Torchserve
CMD ["python", "predict.py"]