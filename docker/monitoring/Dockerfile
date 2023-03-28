FROM python:3.7-slim-buster

RUN pip3 install sagemaker
ENV PYTHONUNBUFFERED=TRUE

ADD /src/model_monitoring/evaluation.py /

ENTRYPOINT ["python3", "/evaluation.py"]