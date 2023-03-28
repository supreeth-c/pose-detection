import os
import sys
import yaml
import json
# cf_stack_name = os.environ['STACK_NAME']


# gateway_url = os.environ['Gateway_URL']


def upate_dict(config_path, project_params, gateway_url):
    endpointName = project_params["EndpointConfig"]["endpointName"]
    bucketName = project_params["S3Config"]["bucketName"]
    replace_dict_values = {
        'REPLACE_GATEWAY_URL': gateway_url,
        'REPLACE_BUCKET_NAME': bucketName,
        'REPLACE_ENDPOINT_NAME': endpointName
    }
    return replace_dict_values


def read_config(config_path):
    with open(os.path.expanduser(config_path), 'r') as stream:
        project_params = yaml.safe_load(stream)
    return project_params


def read_replace_html(index_path, replace_dict_values):
    # Read HTML file as string
    with open(index_path, 'r') as file:
        html_string = file.read()

    # Replace content using dictionary of key-value pairs
    for key, value in replace_dict_values.items():
        html_string = html_string.replace(key, value)

    # Write updated HTML string back to file
    with open(index_path, 'w') as file:
        file.write(html_string)


if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'deployment', 'config.yml')
    index_path = os.path.join(os.getcwd(), 'website', 'index.html')
    gateway_url = json.loads(sys.argv[1])
    project_params = read_config(config_path)
    replace_dict_values = upate_dict(config_path, project_params, gateway_url)
    read_replace_html(index_path, replace_dict_values)
