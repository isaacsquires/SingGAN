from flask import Flask
from flask_restful import Resource, Api, reqparse
from model.evaluate import evaluate
from model.train import train
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


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


api.add_resource(EvaluateModel, '/eval')
api.add_resource(TrainModel, '/train')

if __name__ == '__main__':
    # app.run(debug=True)
    train()
