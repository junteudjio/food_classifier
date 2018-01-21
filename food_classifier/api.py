import argparse

from flask import Flask, jsonify, request

import inference


__author__ = 'Junior Teudjio'


app = Flask(__name__)
inference_model = inference.Model()


@app.route('/predict', methods=['POST', 'GET'])
def predict():
     request_args = request.args
     prediction = inference_model.predict(image_path=request_args['image_path'],
                                          model_type=request_args['model_type'])
     return jsonify(prediction)

def _setup_args():
     parser = argparse.ArgumentParser()
     parser.add_argument('--port', type=int, default=8086)
     parser.add_argument('--host', type=str, default='0.0.0.0')
     return parser.parse_args()

if __name__ == '__main__':
     args = _setup_args()
     app.run(port=args.port,host=args.host)