from flask import Flask, render_template, make_response
from flask_restful import Resource, Api, reqparse
from model.evaluate import evaluate
from model.train import train
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
api = Api(app)

app._static_folder = os.path.abspath("templates/static")


class EvaluateModel(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('audio', type=list, location="json")
        parser = parse.parse_args()
        audio = parser['audio']
        img = evaluate(audio)
        return {'img': img}


class TrainModel(Resource):
    def get(self):
        train()
        return {'modelTrained': True}


class RenderIndex(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'), 200, headers)


api.add_resource(EvaluateModel, '/eval')
api.add_resource(TrainModel, '/train')
api.add_resource(RenderIndex, '/')

if __name__ == '__main__':
    # app.run(debug=True)
    train()
