import os
import boto3
import json
import logging
import sagemaker
from sagemaker.model_monitor import ModelMonitor
from sagemaker.model_monitor import CronExpressionGenerator, MonitoringOutput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from push_monitoring_container_ecr.monitoring_container_ecr_deployment import read_config, build_push_docker_image

cwd = os.getcwd()
logging.basicConfig(
    filename='{}/monitoring_deployment.log'.format(cwd), level=logging.INFO)

logging.info("Starting Model Monitoring Deployment")

# reading the environment variables 
access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
region = os.environ['region']


boto_session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
s3_client = boto3.client("s3",aws_access_key_id=access_key, aws_secret_access_key=secret_key)
account_id = boto3.client('sts',aws_access_key_id=access_key, aws_secret_access_key=secret_key).get_caller_identity()['Account']



def construct_monitoring_image_uri(project_params, docker_file_path):
    """Construct the URI for a monitoring image based on the given project parameters and Docker file path. 
    
    Args:
        project_params: A dictionary containing the project parameters required for building and pushing the Docker image.
        docker_file_path: A string representing the file path to the Docker file.
    
    Returns:
        monitoring_image_uri: A string representing the URI for the monitoring image.
        
    Output:
    
        "123456789012.dkr.ecr.us-west-2.amazonaws.com/my_ecr_repo:latest"
    """

    ecr_monitoring_imageName = build_push_docker_image(project_params,docker_file_path)
    monitoring_image_uri = f"{ecr_monitoring_imageName}:latest"
    logging.info(f"Monitoring Image URI, {monitoring_image_uri}")
    return monitoring_image_uri


def _attach_model_monitoring(project_params, monitoring_image_uri):
    baseJobName = project_params["ModelMonitoring"]["baseJobName"]
    roleArn = project_params["EndpointConfig"]["roleArn"]
    instanceCount = project_params["Instance"]["instanceCount"]
    instanceType = project_params["Instance"]["instanceType"]
    threshold = project_params["ModelMonitoring"]["threshold"]
    monitor = ModelMonitor(
        base_job_name= baseJobName,
        role=roleArn,
        image_uri=monitoring_image_uri,
        instance_count=instanceCount,
        instance_type=instanceType,
        env={ 'THRESHOLD':threshold },
    )
    return monitor


def create_schedule_monitoring_in_house(project_params,monitor_object):
    """Create a monitoring schedule for an Amazon SageMaker endpoint, 
       monitoring schedule is set up to process the default output from the endpoint 
       and store the monitoring results in an S3 bucket 
        
    Args:

        project_params: A dictionary containing the project parameters required for creating the monitoring schedule.
        monitor_object: An instance of the Amazon SageMaker ModelMonitor class, used to create the monitoring schedule.
        
    Returns:
        None
    
    """
    
    bucketName = project_params["S3Config"]["bucketName"]
    s3Prefix = project_params["S3Config"]["s3Prefix"]
    realtimeS3Prefix = project_params["S3Config"]["realtimeS3Prefix"]
    s3DestinationPrefix = project_params["ModelMonitoring"]["processingOutput"]["s3DestinationPrefix"]
    monitoringScheduleName = project_params["ModelMonitoring"]["monitoringScheduleName"]
    outputName = project_params["ModelMonitoring"]["processingOutput"]["outputName"]
    mlDefaultResultSource = project_params["ModelMonitoring"]["processingOutput"]["mlDefaultResultSource"]
    endpointName = project_params["EndpointConfig"]["endpointName"]
    _s3Prefix = s3Prefix + realtimeS3Prefix
    destination = f's3://'+bucketName+"/"+_s3Prefix+s3DestinationPrefix
    processing_output = ProcessingOutput(
        output_name=outputName,
        source=mlDefaultResultSource,
        destination=destination,
    )

    output = MonitoringOutput(source=processing_output.source, destination=processing_output.destination)

    monitor_object.create_monitoring_schedule(
        monitor_schedule_name=monitoringScheduleName,
        output=output,
        endpoint_input=endpointName,
        schedule_cron_expression=CronExpressionGenerator.hourly(),  #need to checck the expression generator logic
    )


if __name__ == '__main__':
    
    # Deployment workflow for model monitoring
    config_path = os.path.join(os.getcwd(), "deployment", "config.yml")
    logging.info(f"Config Path, {config_path}")
    docker_file_path = os.path.join(os.getcwd(), "docker", "monitoring", "Dockerfile")
    logging.info(f"Docker file Path, {docker_file_path}")
    project_params = read_config(config_path)
    monitoring_image_uri_details = construct_monitoring_image_uri(project_params, docker_file_path)
    attache_monitoring_object_status = _attach_model_monitoring(project_params, monitoring_image_uri_details)
    create_schedule_monitoring_in_house_status = create_schedule_monitoring_in_house(project_params,attache_monitoring_object_status)