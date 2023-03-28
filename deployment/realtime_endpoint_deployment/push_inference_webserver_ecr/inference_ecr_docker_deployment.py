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
    filename='inference_docker_deployment.log', level=logging.INFO)


def read_config(file_path):
    """Read a YAML configuration file and load its contents into a dictionary.
       function takes in a file path as input and uses the PyYAML library to safely 
       load the YAML contents into a dictionary.
    
    Args:
        file_path: A string representing the file path to the YAML configuration file.
    
    Returns: 
        project_params: A dictionary containing the project parameters loaded from the YAML configuration file.
    
    """
    with open(os.path.expanduser(file_path), 'r') as stream:
        project_params = yaml.safe_load(stream)

    return project_params


def build_push_docker_image(project_params, docker_file_path):
    """Used to build and push a Docker image to Amazon Elastic Container Registry (ECR). 
       This takes in a dictionary of project parameters and a file path to the Dockerfile
    
    Args:
        project_params: A dictionary containing the project parameters required for building and pushing the Docker image.
        docker_file_path: A string representing the file path to the Dockerfile
    
    Returns: 
        image_name: A string representing the URI of the Docker image in ECR.
    
    """
    ecrInferenceImageName = project_params["InferenceConfig"]["ecrInferenceImageName"]
    repository_short_name = ecrInferenceImageName
    logging.info(f"Repository Name, {repository_short_name}")
    image_name = build_and_push_docker_image(repository_short_name, docker_file_path)
    return image_name
