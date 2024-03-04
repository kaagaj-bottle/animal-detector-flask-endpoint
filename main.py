from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from load_model import auto_transforms
from model import output_from_model


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
    return input_tensor


# endpoint to receive frames of video
@app.route("/vision", methods=["POST"])
def predict():

    blobs = request.files

    input_tensor = handle_blobs(blobs)

    output = output_from_model(input_tensor)

    y_pred = [torch.argmax(item).detach().item() for item in output]
    return jsonify({'msg': 'success', 'predictions': y_pred})


if __name__ == "__main__":
    app.run(debug=True)
