import os
import yaml
import json
import logging
import boto3
from datetime import datetime
from push_inference_webserver_ecr.inference_ecr_docker_deployment import read_config, build_push_docker_image

logging.basicConfig(
    filename='/var/log/inference_endpoint_deployment.log', level=logging.INFO)

access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
session_token = os.environ['AWS_SESSION_TOKEN']


sm_client = boto3.client(service_name='sagemaker',
                         aws_access_key_id=access_key, aws_secret_access_key=secret_key, aws_session_token=session_token)
runtime_sm_client = boto3.client(service_name='sagemaker-runtime',
                                 aws_access_key_id=access_key, aws_secret_access_key=secret_key, aws_session_token=session_token)
account_id = boto3.client('sts', aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key, aws_session_token=session_token).get_caller_identity()['Account']
region = boto3.Session().region_name


def _construct_inference_image_uri(config_path, project_params, docker_file_path):

    ecrInferenceImageName = build_push_docker_image(
        project_params, docker_file_path)
    ecr_inference_image_name = ecrInferenceImageName.split("/")[1]
    inference_image_uri = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(
        account_id, region, ecr_inference_image_name)

    print("INFERENCE_IMAGE_URI", inference_image_uri)

    logging.info(f"Inference Image URI, {inference_image_uri}")
    return inference_image_uri


def create_model_response(project_params, inference_image_uri):
    modelName = project_params["EndpointConfig"]["modelName"]
    roleArn = project_params["EndpointConfig"]["roleArn"]

    create_model_response = sm_client.create_model(
        ModelName=modelName,
        PrimaryContainer={
            'Image': inference_image_uri,
        },
        ExecutionRoleArn=roleArn
    )
    logging.info(f"Model Arn, {create_model_response['ModelArn']}")

    return create_model_response['ModelArn']


def create_endpoint_config_arn(project_params):

    capture_modes = ["Input",  "Output"]

    bucketName = project_params["S3Config"]["bucketName"]
    endpointConfigName = project_params["EndpointConfig"]["endpointConfigName"]
    instanceType = project_params["Instance"]["instanceType"]
    instanceCount = project_params["Instance"]["instanceCount"]
    modelName = project_params["EndpointConfig"]["modelName"]
    enableCapture = project_params["DataCapture"]["enableCapture"]
    initialSamplingPercentage = project_params["DataCapture"]["initialSamplingPercentage"]
    realtimeS3Prefix = project_params["S3Config"]["realtimeS3Prefix"]
    dataCapture = project_params["S3Config"]["dataCapture"]
    s3Prefix = project_params["S3Config"]["s3Prefix"]
    _s3Prefix = s3Prefix + realtimeS3Prefix

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpointConfigName,
        ProductionVariants=[
            {
                'InstanceType': instanceType,
                'InitialInstanceCount': instanceCount,
                'InitialVariantWeight': 1,
                'ModelName': modelName,
                'VariantName': 'AllTraffic'
            }
        ],
        DataCaptureConfig={
            # Whether data should be captured or not.
            'EnableCapture': enableCapture,
            'InitialSamplingPercentage': initialSamplingPercentage,
            'DestinationS3Uri': f's3://'+bucketName+"/"+_s3Prefix+dataCapture,
            "CaptureContentTypeHeader": {
                "JsonContentTypes": ["application/json"]
            },
            # Example - Use list comprehension to capture both Input and Output
            'CaptureOptions': [{"CaptureMode": capture_mode} for capture_mode in capture_modes]
        }
    )
    logging.info(
        f"Endpoint Configuration Arn, {create_endpoint_config_response['EndpointConfigArn']}")
    return create_endpoint_config_response['EndpointConfigArn']


def _create_endpoint_arn(project_params):
    endpointName = project_params["EndpointConfig"]["endpointName"]
    endpointConfigName = project_params["EndpointConfig"]["endpointConfigName"]
    create_endpoint_arn = sm_client.create_endpoint(
        EndpointName=endpointName,
        EndpointConfigName=endpointConfigName
    )
    logging.info(f"Endpoint Arn, {create_endpoint_arn['EndpointArn']}")
    return create_endpoint_arn['EndpointArn']


def create_endpoint(project_params):
    endpointName = project_params["EndpointConfig"]["endpointName"]
    endpoint_arn = _create_endpoint_arn(project_params)

    print("ENDPOINT ARN", endpoint_arn)

    resp = sm_client.describe_endpoint(EndpointName=endpointName)
    status = resp['EndpointStatus']
    logging.info(f"Endpoint Status, {status}")
    logging.info(f"Waiting for {endpointName} endpoint to be in service...")
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpointName)
    logging.info(f"Endpoint Arn, {endpoint_arn}")
    return endpoint_arn


if __name__ == '__main__':

    config_path = os.path.join(
        os.getcwd(), "deployment", "realtime_endpoint_deployment", "config.yml")
    docker_file_path = os.path.join(
        os.getcwd(), "docker", "inference", "Dockerfile")
    logging.info(f"Config Path, {config_path}")
    logging.info(f"Docker file Path, {docker_file_path}")
    project_params = read_config(config_path)

    inference_image_uri = _construct_inference_image_uri(
        config_path, project_params, docker_file_path)
    model_arn = create_model_response(project_params, inference_image_uri)
    create_endpoint_config_arn(project_params)
    create_endpoint(project_params)
