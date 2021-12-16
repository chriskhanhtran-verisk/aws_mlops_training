# Build an image that can do training and inference in SageMaker
# This is a Python 3.8 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM public.ecr.aws/bitnami/python:3.7

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/program
RUN mkdir -p /opt/ml
RUN mkdir -p /opt/ml/model

COPY requirements.txt /opt/program

RUN pip install -r /opt/program/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY decision_trees /opt/program
WORKDIR /opt/program

