import os
import io
import logging
import json
import base64
import flask
import boto3
import cv2
import imageio
import argparse
from io import BytesIO
from PIL import Image
from flask import request
from flask import send_file
from urllib.parse import urlparse
import urllib.request
from botocore.client import Config
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from helper import *


# Some modules to display an animation using imageio.
input_size = 256


# Configure logging
logging.basicConfig(filename='/var/log/flask.log', level=logging.INFO)

# The flask app for serving predictions
app = flask.Flask(__name__)

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")


def create_presigned_url(bucket_name, object_name, expiration=3600):
    s3_client = boto3.client(
        's3', region_name="ap-south-1", config=Config(signature_version='s3v4'))
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name}, ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None
    return response


def load_model():
    path = os.path.join(model_path, 'model.tflite')
    logging.info(f"Model Path, {path}")
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def predict_movenet_for_image(input_image):
    """Runs detection on an input image.

    Args:
        input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
    """
    # load the model
    interpreter = load_model()
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    return keypoints_with_scores


def load_input_image_resize_pad(image_path):
    """Loads image, resizes and pads it to keep the aspect ratio."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    return input_image, image


def prediction(input_image, image):

    keypoints_with_scores = predict_movenet_for_image(input_image)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    plt.savefig('result.png')
    _ = plt.axis('off')
    plt.savefig("output_plot.png")

    im = Image.open("output_plot.png")
    data = io.BytesIO()
    im.save(data, "png")
    encoded_img_data = base64.b64encode(data.getvalue())
    print(f"encoded_image : {encoded_img_data}")
    decode_utf = encoded_img_data.decode('utf-8')
    return decode_utf


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy.
    In this sample container, we declare
    it healthy if we can load the model successfully."""
    load_model()
    logging.info("Model loaded")
    status = 200
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def inference():
    """Performed an inference on incoming data.
    In this sample server, we take data as application/json,
    print it out to confirm that the server received it.
    """

    # log content_type using logger
    logging.info(f"Flask request, {flask.request}")
    logging.info(f"content_type, {flask.request.content_type}")

    if flask.request.content_type == "application/x-npy":
        input_data = flask.request.data
        logging.info(f"input_data, {input_data}")
        result = {
            "resourceType": "OperationOutcome",
            "issue": [
                {
                            "severity": "error",
                            "code": "invalid",
                            "diagnostics": "Invalid JSON format",
                            "details": {
                                "text": "The JSON format is invalid. Please ensure that the request body contains valid JSON."
                            }
                }
            ]
        }

        return flask.Response(response=result, status=400, mimetype="application/json")

    elif flask.request.content_type == "application/json":
        logging.info(f"Flask request data, {flask.request.data}")
        json_data = json.loads(flask.request.data)
        logging.info(f"json data, {json_data}")
        input_path = json_data["image_ref"]
        logging.info(f"input_path, {input_path}")
        objectPath = urlparse(input_path)
        bucket = objectPath.netloc
        key = objectPath.path[1:]
        fileName = os.path.basename(objectPath.path)

        logging.info(f"bucket, {bucket}")
        logging.info(f"Object key, {key}")
        logging.info(f"File Name, {fileName}")

        preSignedUrl = create_presigned_url(bucket, key)
        urllib.request.urlretrieve(preSignedUrl, fileName)

        input_image, image = load_input_image_resize_pad(fileName)
        result = prediction(input_image, image)

        result = json.dumps(result)

        return flask.Response(response=result, status=200, mimetype="application/json")

    else:
        return flask.Response(response="This predictor only images of size 256", status=415, mimetype="application/json")
