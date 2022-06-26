FROM pytorch/torchserve
USER root

RUN mkdir -p /usr/src/app

COPY ./Models /usr/src/app/Models
COPY ./utils /usr/src/app/utils
COPY ./model-store /usr/src/app/model-store
COPY ./run.sh /usr/src/app/
COPY ./handler.py /usr/src/app/
COPY ./requirements.txt /usr/src/app/
WORKDIR /usr/src/app

RUN torch-model-archiver --model-name face_recognition \
					--version 1.0 \
					--model-file ./utils/model.py \
					--serialized-file ./Models/w/prod.torch \
					--handler ./handler.py \
					--extra-files ./Models/pca.pkl -f \
					--export-path ./model-store


RUN pip install -r requirements.txt
RUN chmod a+x ./run.sh
CMD ["torchserve", "--start", "--ncs", "--model-store model-store", "--models face_recognition.mar"]