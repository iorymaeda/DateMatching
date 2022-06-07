FROM python:3.10

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./ /usr/src/app/
RUN pip instsall -r requirements.txt

CMD ["python", "Torchserve/predict.py"]