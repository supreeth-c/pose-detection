import os
import yaml
import logging
from deployment.docker_utils import build_and_push_docker_image
logging.basicConfig(
    filename='/var/log/inference_docker_deployment.log', level=logging.INFO)


def read_config(file_path):
    with open(os.path.expanduser(file_path), 'r') as stream:
        project_params = yaml.safe_load(stream)

    return project_params


def build_push_docker_image(project_params):
    ecrInferenceImageName = project_params["InferenceConfig"]["ecrInferenceImageName"]
    repository_short_name = ecrInferenceImageName
    logging.info("Repository Name", repository_short_name)
    image_name = build_and_push_docker_image(repository_short_name)
    return image_name
