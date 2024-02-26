from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins="*")


@app.route("/", methods=["POST"])
def predict():
    input = request.get_json()
    output = input

    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
