import os
import gc
import io
import time
import base64
import logging

import numpy as np
from PIL import Image

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

import detect


app = Flask(__name__)
CORS(app)

net = detect.load_model(model_name="u2net")

logging.basicConfig(level=logging.INFO)


@app.route("/", methods=["GET"])
def ping():
    return "U^2-Net!"


@app.route("/remove", methods=["POST"])
def remove():

    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'missing file'}), 400

    if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ["jpg", "png", "jpeg"]:
        return jsonify({'error': 'invalid file format'}), 400

    data = request.files['file'].read()
    
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    img = Image.open(io.BytesIO(data))

    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.BILINEAR) # remove resample

    empty_img = Image.new("RGBA", (img.size), (83,40,130))
    new_img = Image.composite(img, empty_img, output.convert("L"))

    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)

    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

@app.route("/remove_nearest", methods=["POST"])
def remove_nearest():

    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'missing file'}), 400

    if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ["jpg", "png", "jpeg"]:
        return jsonify({'error': 'invalid file format'}), 400

    data = request.files['file'].read()
    
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    img = Image.open(io.BytesIO(data))

    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.NEAREST) # remove resample

    empty_img = Image.new("RGBA", (img.size), (83,40,130))
    new_img = Image.composite(img, empty_img, output.convert("L"))

    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)

    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

@app.route("/remove_box", methods=["POST"])
def remove_box():

    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'missing file'}), 400

    if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ["jpg", "png", "jpeg"]:
        return jsonify({'error': 'invalid file format'}), 400

    data = request.files['file'].read()
    
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    img = Image.open(io.BytesIO(data))

    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.BOX) # remove resample

    empty_img = Image.new("RGBA", (img.size), (83,40,130))
    new_img = Image.composite(img, empty_img, output.convert("L"))

    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)

    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

@app.route("/remove_hamming", methods=["POST"])
def remove_hamming():

    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'missing file'}), 400

    if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ["jpg", "png", "jpeg"]:
        return jsonify({'error': 'invalid file format'}), 400

    data = request.files['file'].read()
    
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    img = Image.open(io.BytesIO(data))

    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.HAMMING) # remove resample

    empty_img = Image.new("RGBA", (img.size), (83,40,130))
    new_img = Image.composite(img, empty_img, output.convert("L"))

    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)

    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

@app.route("/remove_bicubic", methods=["POST"])
def remove_bicubic():

    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'missing file'}), 400

    if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ["jpg", "png", "jpeg"]:
        return jsonify({'error': 'invalid file format'}), 400

    data = request.files['file'].read()
    
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    img = Image.open(io.BytesIO(data))

    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.BICUBIC) # remove resample

    empty_img = Image.new("RGBA", (img.size), (83,40,130))
    new_img = Image.composite(img, empty_img, output.convert("L"))

    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)

    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

@app.route("/remove_lanczos", methods=["POST"])
def remove_lanczos():

    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'missing file'}), 400

    if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ["jpg", "png", "jpeg"]:
        return jsonify({'error': 'invalid file format'}), 400

    data = request.files['file'].read()
    
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    img = Image.open(io.BytesIO(data))

    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.LANCZOS) # remove resample

    empty_img = Image.new("RGBA", (img.size), (83,40,130))
    new_img = Image.composite(img, empty_img, output.convert("L"))

    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)

    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

if __name__ == "__main__":
    app.run(debug=True,host='localhost',port=int(os.environ.get('PORT', 8888)))
