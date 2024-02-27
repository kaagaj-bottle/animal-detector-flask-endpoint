from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from load_model import auto_transforms
from model import output_from_model

import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, origins="*")


def preprocess(img):
    img_as_tensor = pil_to_tensor(img)
    transformed_img = auto_transforms(img_as_tensor)
    unsqueezed_img = torch.unsqueeze(transformed_img, 0)
    return unsqueezed_img


@app.route("/", methods=["POST"])
def predict():
    input = request.files['image']
    img = Image.open(input.stream)

    preprocessed_img = preprocess(img)
    output = output_from_model(preprocessed_img)
    return jsonify({'msg': 'success', 'id': int(torch.argmax(output))})


if __name__ == "__main__":
    app.run(debug=True)
