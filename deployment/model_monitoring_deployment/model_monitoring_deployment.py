import os
import boto3
import json
import logging
import sagemaker
from sagemaker.model_monitor import ModelMonitor
from sagemaker.model_monitor import CronExpressionGenerator, MonitoringOutput
from sagemaker.processing import ProcessingInput, ProcessingOutput


from push_monitoring_container_ecr.monitoring_container_ecr_deployment import read_config, build_push_docker_image

logging.basicConfig(
    filename='/var/log/monitoring_deployment.log', level=logging.INFO)


boto_session = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=boto_session)

s3_client = boto3.client("s3")
role = sagemaker.get_execution_role()

account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name


def _construct_inference_image_uri(project_params, docker_file_path):

    ecr_monitoring_imageName = build_push_docker_image(project_params,docker_file_path)
    monitoring_image_uri = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(
        account_id, region, ecr_monitoring_imageName)
    logging.info(f"Monitoring Image URI, {monitoring_image_uri}")
    return monitoring_image_uri


def _attach_model_monitoring(project_params, monitoring_image_uri):
    baseJobName = project_params["ModelMonitoring"]["baseJobName"]
    roleArn = project_params["EndpointConfig"]["roleArn"]
    instanceCount = project_params["Instance"]["instanceCount"]
    instanceType = project_params["Instance"]["instanceType"]
    threshold = project_params["ModelMonitoring"]["threshold"]

    monitor = ModelMonitor(
        base_job_name=baseJobName,
        role=roleArn,
        image_uri=monitoring_image_uri,
        instance_count=instanceCount,
        instance_type=instanceType,
        env={'THRESHOLD': threshold},
    )

    return monitor


def create_monitoring_schedule(project_params, monitoring_image_uri):
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

    monitor = _attach_model_monitoring(project_params, monitoring_image_uri)

    processing_output = ProcessingOutput(
        output_name=outputName,
        source=mlDefaultResultSource,
        destination=destination,
    )
    output = MonitoringOutput(
        source=processing_output.source, destination=processing_output.destination)

    monitor.create_monitoring_schedule(
        monitor_schedule_name=monitoringScheduleName,
        output=output,
        endpoint_input=endpointName,
        # need to check the expression generator logic
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )

    logging.info(f"Describe Monitor Schedule: , {monitor.describe_schedule()}")

    jobs = monitor.list_executions()
    logging.info(f"Model Monitoring job Object:, {jobs}")

    if len(jobs) > 0:
        last_execution_desc = monitor.list_executions()[-1].describe()
        logging.info(f"Monitoring Job execution description:, {last_execution_desc}")
        execu_desc = last_execution_desc.get("ExitMessage", "None")
        logging.info(f"\nExit Message: {execu_desc}")

    else:
        logging.info("""No processing job has been executed yet. 
        This means that one hour has not passed yet. 
        You can go to the next code cell and run the processing job manually""")

    return jobs


if __name__ == '__main__':

    config_path = os.path.join(
        os.getcwd(), "deployment", "realtime_endpoint_deployment", "config.yml")
    logging.info(f"Config Path, {config_path}")
    docker_file_path = os.path.join(os.getcwd(), "docker", "inference", "Dockerfile")
    logging.info(f"Docker file Path, {docker_file_path}")
    project_params = read_config(config_path)
    monitoring_image_uri = _construct_inference_image_uri(project_params, docker_file_path)
    create_monitoring_schedule(project_params, monitoring_image_uri)
