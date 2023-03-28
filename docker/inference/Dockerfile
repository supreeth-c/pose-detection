# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM tensorflow/tensorflow

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    python3-pip \
    python3-setuptools \
    nginx \
    ca-certificates \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN export AWS_REGION="ap-south-1"

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.

RUN python -m pip --no-cache-dir install -U --force-reinstall pip
RUN pip --no-cache-dir install pandas flask gunicorn



COPY /src/inference_webserver/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --user -r /requirements.txt


# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml:${PATH}"

# Set up the program in the image
#COPY src/model_deployment/ /opt/ml

COPY /src/inference_webserver/ /opt/ml

RUN chmod 777 /opt/ml

RUN chmod +x /opt/ml/serve


# Download the movenet_thunder model
RUN wget -q -O model.tflite -P /opt/ml https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite \ 
    && mkdir /opt/ml/model/ \
    && mv model.tflite /opt/ml/model/ \
    && chmod 777 /opt/ml/model


WORKDIR /opt/ml