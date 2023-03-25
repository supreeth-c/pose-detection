import os
import yaml
import json
import logging
import boto3
from datetime import datetime
from push_inference_webserver_ecr.inference_ecr_docker_deployment import read_config, build_push_docker_image

logging.basicConfig(
    filename='/var/log/inference_endpoint_deployment.log', level=logging.INFO)

sm_client = boto3.client(service_name='sagemaker')
runtime_sm_client = boto3.client(service_name='sagemaker-runtime')

account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name


def _construct_inference_image_uri(project_params):

    ecrInferenceImageName = build_push_docker_image(project_params)
    inference_image_uri = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(
        account_id, region, ecrInferenceImageName)
    logging.info("Inference Image URI", inference_image_uri)
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
    logging.info("Model Arn", create_model_response['ModelArn'])

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
    logging.info("Endpoint Configuration Arn",
                 create_endpoint_config_response['EndpointConfigArn'])
    return create_endpoint_config_response['EndpointConfigArn']


def _create_endpoint_arn(project_params):
    endpointName = project_params["EndpointConfig"]["endpointName"]
    endpointConfigName = project_params["EndpointConfig"]["endpointConfigName"]
    create_endpoint_arn = sm_client.create_endpoint(
        EndpointName=endpointName,
        EndpointConfigName=endpointConfigName
    )
    logging.info("Endpoint Arn", create_endpoint_arn['EndpointArn'])
    return create_endpoint_arn['EndpointArn']


def create_endpoint(project_params):
    endpointName = project_params["EndpointConfig"]["endpointName"]
    resp = sm_client.describe_endpoint(EndpointName=endpointName)
    status = resp['EndpointStatus']
    logging.info("Endpoint Status", status)
    logging.info(
        'Waiting for {} endpoint to be in service...'.format(endpointName))
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpointName)
    endpoint_arn = _create_endpoint_arn(project_params)
    logging.info("Endpoint Arn", endpoint_arn)
    return endpoint_arn


if __name__ == '__main__':

    config_path = os.path.join(
        os.getcwd(), "deployment", "realtime_endpoint_deployment", "config.yml")
    logging.info("Config Path", config_path)
    project_params = read_config(config_path)
    inference_image_uri = _construct_inference_image_uri(
        config_path, project_params)
    model_arn = create_model_response(project_params, inference_image_uri)
    create_endpoint_config_arn(project_params)
    create_endpoint(project_params)
