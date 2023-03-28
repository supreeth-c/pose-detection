"""Custom Model Monitoring script for infrastructure monitoring"""
# Python Built-Ins:
from collections import defaultdict
import datetime
import json
import os
import traceback
from types import SimpleNamespace

# External Dependencies:
import numpy as np
import boto3
from dateutil.tz import tzutc
from datetime import timedelta, datetime
cw_client = boto3.client('cloudwatch', region_name='ap-south-1')


def get_environment():
    """Load configuration variables for SM Model Monitoring job
    """
    try:
        with open("/opt/ml/config/processingjobconfig.json", "r") as conffile:
            defaults = json.loads(conffile.read())["Environment"]
    except Exception as e:
        traceback.print_exc()
        print("Unable to read environment vars from SM processing config file")
        defaults = {}

    return SimpleNamespace(
        dataset_format=os.environ.get(
            "dataset_format", defaults.get("dataset_format")),
        dataset_source=os.environ.get(
            "dataset_source",
            defaults.get("dataset_source",
                         "/opt/ml/processing/input/endpoint"),
        ),
        end_time=os.environ.get("end_time", defaults.get("end_time")),
        output_path=os.environ.get(
            "output_path",
            defaults.get("output_path", "/opt/ml/processing/resultdata"),
        ),
        publish_cloudwatch_metrics=os.environ.get(
            "publish_cloudwatch_metrics",
            defaults.get("publish_cloudwatch_metrics", "Enabled"),
        ),
        sagemaker_endpoint_name=os.environ.get(
            "sagemaker_endpoint_name",
            defaults.get("sagemaker_endpoint_name"),
        ),
        sagemaker_monitoring_schedule_name=os.environ.get(
            "sagemaker_monitoring_schedule_name",
            defaults.get("sagemaker_monitoring_schedule_name"),
        ),
        start_time=os.environ.get("start_time", defaults.get("start_time")),

        max_ratio_threshold=float(os.environ.get(
            "THRESHOLD", defaults.get("THRESHOLD", "nan"))),
    )


def get_infra_stats(endpoint_name, start_time, end_time):
    
    """Retrieves infrastructure statistics for a given SageMaker endpoint between a specified start and end time. 
    The function takes in the endpoint name, start time and end time as parameters and returns a list of dictionaries 
    containing the requested statistics
    
    Args:
        endpoint_name: A string representing the name of the SageMaker endpoint for which to retrieve infrastructure statistics.
        start_time: A datetime object representing the start time of the period for which to retrieve statistics.
        end_time: A datetime object representing the end time of the period for which to retrieve statistics.
    
    Returns:
        metrics_report: A list of dictionaries containing the infrastructure statistics for the specified endpoint within the specified time period. 
        Each dictionary contains the metric name and the corresponding datapoints for that metric.
    """
    
    metrics = [
        {'namespace': 'AWS/SageMaker', 'unit': 'Microseconds', 'name': 'ModelLatency', 'dimensions': [
            {'Name': 'EndpointName', 'Value': endpoint_name}, {'Name': 'VariantName', 'Value': 'AllTraffic'}]},

        {'namespace': '/aws/sagemaker/Endpoints', 'unit': 'Percent', 'name': 'CPUUtilization', 'dimensions': [
            {'Name': 'EndpointName', 'Value': endpoint_name}, {'Name': 'VariantName', 'Value': 'AllTraffic'}]},

        {'namespace': '/aws/sagemaker/Endpoints', 'unit': 'Percent', 'name': 'MemoryUtilization',
            'dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}, {'Name': 'VariantName', 'Value': 'AllTraffic'}]}
    ]

    metrics_report = []
    for metric in metrics:
        response = cw_client.get_metric_statistics(
            Namespace=metric['namespace'],
            MetricName=metric['name'],
            Dimensions=metric['dimensions'],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Unit=metric['unit'],
            Statistics=['SampleCount', 'Average']
        )

        metrics_report.append({metric['name']: response['Datapoints']})

    return metrics_report


def average_and_time(matrix, len_matrix):
    """Takes in a matrix and its length as input and returns the maximum average value and the corresponding timestamp from the matrix. 
        The matrix is assumed to be a list of dictionaries where each dictionary contains two keys 'Average' and 'Timestamp'.
        
    Args:
        matrix: A list of dictionaries where each dictionary contains two keys 'Average' and 'Timestamp'.
        len_matrix: An integer representing the length of the matrix.
        
    Returns:
        max_avg: A float representing the maximum average value in the matrix.
        max_time: A string representing the timestamp corresponding to the maximum average value.

    """
    
    for i in range(len_matrix):
        if i == 0:
            max_avg = matrix[i]['Average']
            max_time = matrix[i]['Timestamp']

        else:
            if matrix[i]['Average'] > max_avg:
                max_avg = matrix[i]['Average']
                max_time = matrix[i]['Timestamp']
        return max_avg, max_time


def _iterator(list_metrics, key_name):
    for key, value in list_metrics.items():
        if key == key_name:
            refined_list = list_metrics[key_name]
        else:
            return ""
    return refined_list


def _check_field(list_metrics):
    if len(list_metrics) == 3:
        model_lat = _iterator(list_metrics[0], "ModelLatency")
        cpu_util = _iterator(list_metrics[1], "CPUUtilization")
        mem_util = _iterator(list_metrics[2], "MemoryUtilization")
    else:
        return None

    return model_lat, cpu_util, mem_util


if __name__ == "__main__":
    env = get_environment()
    print(f"Starting evaluation with config:\n{env}")

    print("Analyzing collected data...")
    end_point_name = "human-pose-prediction-endpoint"
    start_time = datetime.now() - timedelta(hours=10)
    end_time = datetime.now()

    result = get_infra_stats(end_point_name, start_time, end_time)

    model_lat, cpu_util, mem_util = _check_field(result)

    avg_model_latency, time = average_and_time(model_lat, len(model_lat))
    avg_cpu_util, time = average_and_time(cpu_util, len(cpu_util))
    avg_mem_util, time = average_and_time(mem_util, len(mem_util))

    avg_model_latency = (avg_model_latency/1000000)

    print("Checking for constraint violations...")
    violations = []
    if avg_model_latency > env.max_ratio_threshold:
        violations.append({
            "feature_name": "ModelLatency",
            "constraint_check_type": "baseline_infra_drift_check",
            "description": "Model Latency actual {:.2f}% in seconds : Exceeded {:.2f}% threshold".format(
                avg_model_latency,
                env.max_ratio_threshold,
            ),
        })

    print("Writing violations file...")
    with open(os.path.join(env.output_path, "constraints_violations.json"), "w") as outfile:
        outfile.write(json.dumps(
            {"metrics": avg_model_latency},
            indent=4,
        ))

    print("Writing overall status output...")
    with open("/opt/ml/output/message", "w") as outfile:
        if len(violations):
            msg = f"CompletedWithViolations: {violations[0]['description']}"
        else:
            msg = "Completed: Job completed successfully with no violations."
        outfile.write(msg)
        print(msg)

    if env.publish_cloudwatch_metrics:
        print("Writing CloudWatch metrics...")
        with open("/opt/ml/output/metrics/cloudwatch/cloudwatch_metrics.jsonl", "a+") as outfile:
            # One metric per line (JSONLines list of dictionaries)
            # Remember these metrics are aggregated in graphs, so we report them as statistics on our dataset
            json.dump(
                {
                    "MetricName": f"Average CPU Utlization",
                    "Timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "Dimensions": [
                        {"Name": "Endpoint",
                            "Value": env.sagemaker_endpoint_name or "unknown"},
                        {
                            "Name": "MonitoringSchedule",
                            "Value": env.sagemaker_monitoring_schedule_name or "unknown",
                        },
                    ],
                    "StatisticValues": {
                        "Average": avg_cpu_util
                    },
                },
                outfile
            )
            outfile.write("\n")

            json.dump(
                {
                    "MetricName": f"Average Mememory Utlization",
                    "Timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "Dimensions": [
                        {"Name": "Endpoint",
                            "Value": env.sagemaker_endpoint_name or "unknown"},
                        {
                            "Name": "MonitoringSchedule",
                            "Value": env.sagemaker_monitoring_schedule_name or "unknown",
                        },
                    ],
                    "StatisticValues": {
                        "Average": avg_mem_util
                    },
                },
                outfile
            )
            outfile.write("\n")
