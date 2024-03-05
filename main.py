from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path
from load_model import auto_transforms
from model import vision_output, audio_output, vision_classes, audio_classes
from utils import preprocess_single_image, handle_blobs, save_images, save_load_audio
from torch.nn.functional import softmax

app = Flask(__name__)
CORS(app, origins="*")

# endpoint to receive frames of video

THRESHOLD = 0.90


def decide_output(output):
    y_pred = []
    for i in range(output.shape[0]):
        max_arg = torch.argmax(output[i])
        if output[i][max_arg] >= THRESHOLD:
            y_pred.append(vision_classes[max_arg])
        else:
            y_pred.append("null")
    return y_pred


@app.route("/vision", methods=["POST"])
def predict_vision():

    blobs = request.files

    input_tensor = handle_blobs(blobs)

    output = softmax(vision_output(input_tensor))

    # y_pred = [vision_classes[torch.argmax(item).detach().item()] for item in output]
    y_pred = decide_output(output)
    print(output.shape)
    return jsonify({'msg': 'success', 'predictions': y_pred})


@app.route("/save_frames", methods=["POST"])
def save_frames():
    blobs = request.files
    try:
        save_images(blobs)
        return jsonify({"msg": "success"})
    except:
        print("couldn't save images")
        return jsonify({"msg": "failure"})


@app.route("/vision_single", methods=["POST"])
def predict_single_image():
    image = request.files['image']
    image = Image.open(image.stream)
    image = preprocess_single_image(image)
    image = torch.unsqueeze(image, 0)
    output = vision_output(image)
    type = vision_classes[torch.argmax(output)]
    return jsonify({'msg': 'success', 'output': type})


@app.route("/audio", methods=["POST"])
def predict_audio():
    blob = request.files['audio']
    input_tensor = save_load_audio(blob)
    output = softmax(audio_output(input_tensor))
    y_pred = decide_output(output)
    # y_pred = [vision_classes[torch.argmax(
    #     item).detach().item()] for item in output]

    return jsonify({'msg': 'success', 'output': y_pred})


if __name__ == "__main__":
    app.run(debug=True)
