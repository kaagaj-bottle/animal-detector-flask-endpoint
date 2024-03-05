from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
from torchvision.transforms.functional import pil_to_tensor

from load_model import auto_transforms
from model import vision_output, audio_output, vision_classes, audio_classes


app = Flask(__name__)
CORS(app, origins="*")

# preprocesses single image to tensor of size and transformation as per pre-trained model


def preprocess_single_image(img):
    img_as_tensor = pil_to_tensor(img)
    transformed_img = auto_transforms(img_as_tensor)
    return transformed_img

# takes input as blobs and outputs the input tensor for the model comprising of all the images


def handle_blobs(blobs):
    no_of_frames = len(blobs)
    input_tensor = torch.zeros(
        [no_of_frames, 3, 224, 224], dtype=torch.float32)
    idx = 0

    for key in blobs:
        try:
            img = Image.open(blobs[key].stream)
        except Exception as e:
            print('----------------------------------------------')
            print(e)
            print('----------------------------------------------')
            return input_tensor
        input_tensor[idx] = preprocess_single_image(img)
        idx+=1
    return input_tensor

def save_images(blobs):
    i=0
    for key in blobs:
        img=Image.open(blobs[key].stream)
        try:
            img.save(f"img{i}.bmp")
            i+=1
        except:
            print(f"couldn't save the {i}th image")
# endpoint to receive frames of video
@app.route("/vision", methods=["POST"])
def predict_vision():

    blobs = request.files

    input_tensor = handle_blobs(blobs)

    output = vision_output(input_tensor)

    y_pred = [vision_classes[torch.argmax(
        item).detach().item()] for item in output]
    return jsonify({'msg': 'success', 'predictions': y_pred})

@app.route("/save_frames",methods=["POST"])
def save_frames():
    blobs=request.files
    try:
        save_images(blobs)
        return jsonify({"msg":"success"})
    except:
        print("couldn't save images")
        return jsonify({"msg":"failure"})

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
    blobs = request.files

    return jsonify({'msg': 'success', 'output': 'under construction'})


if __name__ == "__main__":
    app.run(debug=True)
