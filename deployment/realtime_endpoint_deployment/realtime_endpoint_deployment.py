import os
import yaml
import json
import logging
import boto3
from datetime import datetime
from push_inference_webserver_ecr.inference_ecr_docker_deployment import read_config, build_push_docker_image

cwd = os.getcwd()

logging.basicConfig(
    filename='{}/inference_endpoint_deployment.log'.format(cwd), level=logging.INFO)

# reading the environment variables 
access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['region']


sm_client = boto3.client(service_name='sagemaker',
                         aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
runtime_sm_client = boto3.client(service_name='sagemaker-runtime',
                                 aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
account_id = boto3.client('sts', aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key, region_name=region).get_caller_identity()['Account']

s3_client = boto3.client(service_name='s3',
                             aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)





def construct_inference_image_uri(config_path, project_params, docker_file_path):
    """Construct the URI for a Inferencing image based on the given project parameters and Docker file path. 
    Args:
        project_params: A dictionary containing the project parameters required for building and pushing the Docker image.
        docker_file_path: A string representing the file path to the Docker file.
    Return:
        inference_image_uri: A string representing the URI for the inference image.
    Output:
        "123456789012.dkr.ecr.us-west-2.amazonaws.com/my_ecr_repo:latest"
    """

    ecrInferenceImageName = build_push_docker_image(
        project_params, docker_file_path)
    ecr_inference_image_name = ecrInferenceImageName.split("/")[1]
    inference_image_uri = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(
        account_id, region, ecr_inference_image_name)

    print("INFERENCE_IMAGE_URI", inference_image_uri)

    logging.info(f"Inference Image URI, {inference_image_uri}")
    return inference_image_uri


def create_model_response(project_params, inference_image_uri):
    """Create an Amazon SageMaker model with the specified name and Docker image URI for inference. 
    Function takes in a dictionary of project parameters and the URI for the Docker image to use for the model's primary container
    
    Args:
        project_params: A dictionary containing the project parameters required for creating the SageMaker model.
        inference_image_uri: A string representing the URI for the Docker image to use for the model's primary container.
    Return:
        create_model_response['ModelArn']: A string representing the Amazon Resource Name (ARN) for the created SageMaker model

    """
    
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
    
    """Create an Amazon SageMaker endpoint configuration. 
    The endpoint configuration defines the details of the endpoint such as instance type, number of instances, model name,
    and data capture settings.
        
    Args:
        project_params: A dictionary containing the project parameters required for creating the endpoint configuration.
    
    Return:
        A string containing the Amazon Resource Name (ARN) of the newly created endpoint configuration.

    """

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
    """Create an Amazon SageMaker endpoint based on the given project parameters.
    
    Args:
        project_params: A dictionary containing the project parameters required for creating the endpoint configuration.
    
    Returns:

        endpoint_arn: A string representing the ARN of the created endpoint.
    
    """
    
    endpointName = project_params["EndpointConfig"]["endpointName"]
    endpoint_arn = _create_endpoint_arn(project_params)
    resp = sm_client.describe_endpoint(EndpointName=endpointName)
    status = resp['EndpointStatus']
    logging.info(f"Endpoint Status, {status}")
    logging.info(f"Waiting for {endpointName} endpoint to be in service...")
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpointName)
    logging.info(f"Endpoint Arn, {endpoint_arn}")
    return endpoint_arn

def endpoint_url(project_params):
    """Create an Amazon SageMaker endpoint URL using endpoint name and region for a project parameters.
    
    Args:
        project_params: A dictionary containing the project parameters required for creating the endpoint configuration.
    
    Returns:

        endpoint_url:  https:// url to be used for triggering the prediction
    
    """
    
    endpointName = project_params["EndpointConfig"]["endpointName"]
    endpoint_url = "https://runtime.sagemaker.{}.amazonaws.com/endpoints/{}/invocations".format(region, endpointName)
    logging.info(f"Endpoint URL, {endpoint_url}")
    print("SAGEMAKER ENDPOINT URL:", endpoint_url)
    return endpoint_url
    

def create_s3_bucket(project_params):
    """Create an Amazon SageMaker S3 bucket in a region for a project parameters.
    
    Args:
        project_params: A dictionary containing the project parameters required for creating the endpoint configuration.
    
    Returns:

        bucketName:  Name of the S3 bucket created
    
    """
    cors_configuration = {
    "CORSRules": [
        {
            "AllowedHeaders": [
                "Authorization",
                "Content-Range",
                "Accept",
                "Content-Type",
                "Origin",
                "Range",
            ],
            "AllowedMethods": ["GET", "PUT"],
            "AllowedOrigins": ["*"],
            "ExposeHeaders": ["Content-Range", "Content-Length", "ETag"],
            "MaxAgeSeconds": 3000,
        }
    ]
    }
    bucketName = project_params["S3Config"]["bucketName"]
    folderName = project_params["S3Config"]["inputDir"]
    s3_client.create_bucket(Bucket=bucketName,CreateBucketConfiguration={'LocationConstraint': region})
    s3_client.put_object(Bucket=bucketName, Key=folderName)
    s3_client.put_bucket_cors(Bucket=bucketName, CORSConfiguration=cors_configuration)
    logging.info(f"Bucket Name, {bucketName}")
    print("INPUT S3 BUCKET:", endpoint_url)
    return bucketName

if __name__ == '__main__':
    # Deployment workflow for inference end point creation
    config_path = os.path.join(os.getcwd(), "deployment", "config.yml")
    docker_file_path = os.path.join(
        os.getcwd(), "docker", "inference", "Dockerfile")
    logging.info(f"Config Path, {config_path}")
    logging.info(f"Docker file Path, {docker_file_path}")
    project_params = read_config(config_path)
    create_s3_bucket(project_params)
    inference_image_uri = construct_inference_image_uri(
        config_path, project_params, docker_file_path)
    model_arn = create_model_response(project_params, inference_image_uri)
    create_endpoint_config_arn(project_params)
    create_endpoint(project_params)
    endpoint_url(project_params)
    