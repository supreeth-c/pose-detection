import os
import yaml
import logging
import sys

BASE_DIR = os.path.dirname(__file__)
push_ecr = BASE_DIR.replace("/push_inference_webserver_ecr","")
deployment_path = push_ecr.replace("/realtime_endpoint_deployment","")
sys.path.append(deployment_path)

from docker_utils import build_and_push_docker_image

logging.basicConfig(
    filename='/var/log/inference_docker_deployment.log', level=logging.INFO)


def read_config(file_path):
    with open(os.path.expanduser(file_path), 'r') as stream:
        project_params = yaml.safe_load(stream)

    return project_params


def build_push_docker_image(project_params, docker_file_path):
    
    ecrInferenceImageName = project_params["InferenceConfig"]["ecrInferenceImageName"]
    repository_short_name = ecrInferenceImageName
    logging.info(f"Repository Name, {repository_short_name}")
    image_name = build_and_push_docker_image(repository_short_name, docker_file_path)
    return image_name
