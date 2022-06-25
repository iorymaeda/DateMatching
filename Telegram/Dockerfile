FROM python:3.10

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app/

RUN pip install -r requirements.txt

WORKDIR /usr/src/app/
CMD ["python", "main.py"]