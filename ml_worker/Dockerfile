FROM python:3.10-rc-slim-buster

COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8

ENV JAVA_HOME /usr/local/openjdk-8
ENV PATH $PATH:$JAVA_HOME/bin

RUN apt-get update && \
    yes | apt-get upgrade && \
    apt-get -y install make \
                       gcc \
                       libc-dev \
                       musl-dev \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       tesseract-ocr \
                       libtesseract-dev \
                       libreoffice-common \
                       wkhtmltopdf \
                       libreoffice-writer && \
    rm -rf /var/lib/apt/lists/*

# Install Git
RUN apt-get update && apt-get install -y git && apt-get install -y build-essential

# Install paddlepaddle and paddlehub
RUN pip install paddlepaddle paddlehub

# Install the latest version of Pip
RUN python -m ensurepip --upgrade --default-pip && \
    pip install --no-cache-dir --upgrade pip==23.1.2

COPY requirements.txt /
RUN python -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r /requirements.txt

RUN python -m pip install -q 'git+https://github.com/jyotiyadav94/PaddleOCR.git'

ADD . /ml_worker
WORKDIR /ml_worker
RUN mkdir /nonexistent
RUN chmod -R 777 /nonexistent

ENTRYPOINT celery -A tasks worker -P solo --loglevel=info
