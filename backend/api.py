from flask import Flask
from flask_restful import Resource, Api, reqparse
from model.evaluate import evaluate
from model.train import train
import werkzeug
import numpy as np
import io
from scipy.io.wavfile import read


app = Flask(__name__)
api = Api(app)


class EvaluateModel(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument(
            'audio', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        stream = args['audio'].stream
        a = read(stream)
        img = evaluate(a[1][100:200])
        return {'img': img}


class TrainModel(Resource):
    def get(self):
        raw = train()
        return 'Training complete'


api.add_resource(EvaluateModel, '/eval')
api.add_resource(TrainModel, '/train')

if __name__ == '__main__':
    app.run(debug=True)
