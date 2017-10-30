from flask import Flask, jsonify, request
from flask_cors import CORS

import argparse
import sys
import numpy as np
import tensorflow as tf

import tensor_process as tp
from configs.tensor_config import path

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return jsonify(
        'pong'
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['images']
        print(f)
        return 'uploaded'

@app.route('/dummy', methods=['GET', 'POST'])
def dummy():
    if request.method == 'POST':
        name = request.form.get('name')
        print(name)
        return name

@app.route('/recog_digits', methods=['POST'])
def tensor():
    file_name = request.form.get('file_path') or "/home/karim/learn_tensor_flow/pure_tensor/tf_files/metering_images/9/1.jpg"
    model_file = request.form.get('model_file_path') or path()['model_path']
    label_file = request.form.get('label_file_path') or path()['label_path']
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    graph = tp.load_graph(model_file)
    t = tp.read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]

    labels = tp.load_labels(label_file)


    accuration = float(results[top_k[0]])
    return jsonify(
        text=labels[top_k[0]],
        accuration=accuration
    )
