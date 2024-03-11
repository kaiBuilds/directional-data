FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# pip install requirements
RUN pip3 install --upgrade pip
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

CMD [ "bin/bash" ]

